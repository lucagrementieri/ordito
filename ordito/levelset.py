"""
Surfaces from scalar fields, and the surface operations whose answer is a *different* surface.

[`marching_cubes`][ordito.levelset.marching_cubes] is the primitive: the ``iso`` level set of a
dense lattice, as a triangle mesh. It is the extraction tail of every implicit-surface pipeline in
the package -- [`proximity.signed_distance_grid`][ordito.proximity.signed_distance_grid],
[`voxels.to_field`][ordito.voxels.to_field],
[`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] and
[`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] all end in it -- which
is why it lives here with its consumers rather than with the point-cloud reconstructors: none of
those callers is reconstructing from a cloud, they are extracting a level set.

[`offset_mesh`][ordito.levelset.offset_mesh] moves a closed surface a fixed distance along its own
normal direction, in the only way that stays well defined where the surface curves back on itself:
through the signed distance field, whose level set at ``d`` is exactly the set of points at distance
``d`` -- so it is ``marching_cubes`` over a shifted SDF. That is why an offset lives here rather
than in [`ordito.remesh`][ordito.remesh]: it is not a vertex displacement, and its output topology
is not the input's. A sphere offset inward past its radius vanishes; a thin plate offset outward
merges into one shell. Both are correct, and no per-vertex method produces either.

[`thicken_mesh`][ordito.levelset.thicken_mesh] is the *topology-preserving* counterpart and the one
member here that is **not** a level-set operation. Where an open surface has to become a solid of
known thickness and the input's own triangulation should survive, it extrudes along the vertex
normals and closes the rim, so the output is the input plus a copy plus a band -- nothing is
resampled and no field is built. The choice between the two is the choice between keeping the
triangulation and keeping the distance. It can self-intersect where the thickness exceeds the local
radius of curvature, and it does not guard against that --
[`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
names the condition exactly, and repairing it is a separate operation.

For the fields these consume, see
[`ordito.proximity.signed_distance_grid`][ordito.proximity.signed_distance_grid]; for the binary
occupancy lattice, [`ordito.voxels`][ordito.voxels].
"""

from __future__ import annotations

import functools
import math
from typing import Literal, cast

import numpy as np
import warp as wp
from warp.geometry import sparse_marching_cubes

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.kernels import levelset as kernel_levelset

# Bounds on the automatic ``voxel_size``, both expressed as a sample count across a span, and both
# load-bearing rather than defensive. The spans differ, which is the part to read carefully: the
# floor counts across the mesh's own bounding-box diagonal, the cap across the padded lattice.
#
# The **cap** stops a small offset distance on a large mesh from asking for a lattice nobody can
# allocate: a distance field costs 4 bytes a sample, so the memory is cubic in it. It is measured
# against the *padded* extent -- the diagonal plus the two outward margins the offset band needs --
# rather than against the mesh's own diagonal, because the padding is what actually grows without
# bound: an outward offset of ten times the diagonal pads by ``distance / spacing`` cells on every
# side, so a cap that only counts the mesh permits a lattice orders of magnitude past it.
#
# The **floor** guards a real failure mode. Tying the spacing to the offset distance alone
# resolves the *band* the level set sits in but not necessarily what is left of the object: an
# inward offset of 0.9 on a unit sphere leaves a sphere of radius 0.1, which at a spacing of 0.9/3
# is smaller than one cell -- so marching cubes finds nothing and the call returns **empty** for a
# level set that exists. At least 64 samples across the mesh avoids that, for a 64 ** 3 lattice.
#
# Where the two conflict -- a large outward offset, where resolving the mesh to 64 samples would
# blow the allocation -- the **cap wins**, because the floor only buys accuracy where the cap
# decides whether the call runs at all.
_MAX_AUTO_RESOLUTION = 256
_MIN_AUTO_RESOLUTION = 64

# Samples across the offset distance in the automatic ``voxel_size``. Three is the smallest number
# that puts a lattice cell strictly inside the offset band, which is what marching cubes needs to
# find the level set at all.
_SAMPLES_PER_DISTANCE = 3

# Lattice nodes from which a closed, consistently wound mesh's distance level set is extracted
# sparsely -- a Lipschitz octree brackets the surface and the distance is queried only at the
# corners of the cells it keeps -- instead of sampling every node: for ``sign_mode="parity"``, and
# for ``"winding"`` only from ``_LATTICE_WINDING_BELOW_NODES`` on. See
# ``signed_distance_level_set``'s Notes for why the input has to be closed. Measured on CUDA
# against the dense lattice through ``offset_mesh``: 0.84-0.93x on ``dragon`` at 0.6-0.7 M nodes,
# 1.95x at 2.4 M and 4.5x at 9 M, 1.84x on ``happy_buddha`` at the benchmark's finest cell,
# 5.7-11.8x on a 20 k-face sphere at 10-37 M nodes. The closedness test it pays on an open input
# is two key sorts, 0.98x at worst (``lucy``, 28 M faces).
_SPARSE_LEVEL_SET_FROM_NODES = 1 << 21

# The winding-signed extraction that replaces sampling the whole lattice (``kernels/levelset.py``'s
# header has the method) is taken below this many nodes, whatever the input's topology; from here a
# closed input goes back to the octree above. It reproduces the dense lattice's surface bit for bit,
# where the octree matches it only up to rounding at nodes whose distance rounds onto the level. It
# holds about 44 bytes a node. Against the octree on closed inputs: 5.5-6.4x on ``happy_buddha``
# from 17 to 55 M nodes, 3.0x / 2.4x / 1.6x on a 20 k-face sphere at 29 / 56 / 95 M.
_LATTICE_WINDING_BELOW_NODES = 1 << 26

# A node whose exact winding number is within this of 1/2 takes Warp's own Barnes-Hut sign rather
# than the exact one, since only there can the two differ. Warp's approximation error is two orders
# below it on every lattice node probed (``kernels/levelset.py``).
_WINDING_UNDECIDED_DELTA = 0.1

# Undecided nodes summed exactly over every face, at most this many and at most this many
# (node, face) pairs; past either bound they all take Warp's sign, building the solid-angle BVH.
_EXACT_WINDING_CAPACITY = 1 << 16
_EXACT_WINDING_WORK = 1 << 32

# The cone over the boundary is evaluated at every node, so a boundary of very many edges or
# components (a triangle soup) falls back to sampling the whole lattice with Warp's sign.
_MAX_CONE_EDGES = 1 << 16
_MAX_CONE_WORK = 1 << 34


def marching_cubes(
    field: odt.Array3dFloat32,
    iso: float = 0.0,
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    edge_margin: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract the ``iso`` level set of a dense scalar lattice as a triangle mesh.

    The extraction tail of every implicit-surface pipeline, exposed on its own so a caller with a
    field of their own — an SDF, an occupancy volume, a simulation state — does not have to route it
    through [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] to get a
    surface out. It is what
    [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] is built from.

    Parameters
    ----------
    field
        ``(nx, ny, nz)`` lattice of scalar values, with ``x`` the slowest axis. The surface is
        extracted where the field crosses ``iso``; the sign convention is the caller's, and the
        winding follows it (with ordito's outside-positive
        [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] convention the
        normals come out pointing outward).
    iso
        Level to extract. Defaults to ``0``, which is the zero level set of a signed distance field.
    bounds
        ``(lower, upper)`` world-space corners the lattice spans, so ``field[0, 0, 0]`` sits at
        ``lower`` and ``field[nx - 1, ny - 1, nz - 1]`` at ``upper``. When ``None`` the result is in
        *index* space: vertex coordinates are lattice indices. The mapping is positional and is not
        checked, so a pair passed the other way round is honoured rather than rejected: it mirrors
        the result along every axis it inverts, which flips the winding with it, and a pair whose
        corners coincide collapses every vertex onto that point.
    edge_margin
        Fraction of a lattice edge every vertex keeps from both of the edge's nodes
        (``0 <= edge_margin < 0.5``). At ``0`` a vertex can land on, or arbitrarily close to, a
        node where the field meets ``iso`` there, and the triangles around it degenerate into
        slivers whose area is rounding noise; a positive margin moves such a vertex at most that
        fraction of an edge and bounds every triangle's altitudes below by a fixed fraction of
        ``edge_margin`` times the lattice spacing, without changing the topology.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_vertices,)`` level-set vertices on ``field.device``. Empty when the field does not
        cross ``iso``.
    faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat triangle index buffer.

    Raises
    ------
    TypeError
        If ``field`` is not a rank-3 ``wp.float32`` array.
    ValueError
        If any of ``field``'s dimensions is below 2, or ``edge_margin`` is outside ``[0, 0.5)``.

    See Also
    --------
    [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform]
    [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson]
    [`ordito.proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]
    [`ordito.voxels.to_field`][ordito.voxels.to_field]
    [`ordito.voxels.grid_points`][ordito.voxels.grid_points]

    Notes
    -----
    The triangulation is Warp's ``warp.geometry.IsoSurfaceMarchingCubes``: its case tables, its
    vertex numbering (one vertex per crossing lattice edge, in row-major order of the edge's lower
    node and then by axis), its triangle order and its interpolation, so the result equals
    ``IsoSurfaceMarchingCubes.extract`` on the same lattice and bounds bit for bit, buffer order
    included. What differs is the scratch it needs: a pair of counters per lattice node, about twice
    the field, where Warp's extraction needs about eleven times the field -- the margin between
    fitting a fine lattice on the device and not.

    The consequence worth knowing is that the result is **not guaranteed manifold** at an
    ambiguous cell, and can carry duplicate vertices where two cells agree on a crossing —
    [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] runs
    [`ordito.repair`][ordito.repair] over it for exactly that reason.
    """
    field = odt.as_array3d(field, wp.float32)
    shape = tuple(int(dim) for dim in field.shape)
    if min(shape) < 2:
        raise ValueError(f"field must be at least 2 wide along every axis, got {shape}")
    if not 0.0 <= edge_margin < 0.5:
        raise ValueError(f"edge_margin must be in [0, 0.5), got {edge_margin}")
    nx, ny, nz = shape

    if bounds is None:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(float(nx - 1), float(ny - 1), float(nz - 1))
    else:
        lower, upper = wp.vec3(bounds[0]), wp.vec3(bounds[1])
    # The spacing as ``IsoSurfaceMarchingCubes.extract`` forms it (Warp's ``resolve_domain_bounds``:
    # a float32 difference divided by the cell count), so every interpolated vertex rounds as
    # Warp's does.
    cells_np = np.array((nx - 1, ny - 1, nz - 1), dtype=np.float32)
    delta = wp.vec3(
        (np.asarray(upper, dtype=np.float32) - np.asarray(lower, dtype=np.float32)) / cells_np
    )

    device = wp.get_device(field.device)
    table = _marching_cubes_table(device.alias)
    counts = _launch.empty(nx * ny * nz, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_levelset.marching_cubes_counts,
        dim=(nx, ny, nz),
        inputs=[field, wp.float32(iso), table],
        outputs=[counts],
        device=device,
    )
    _launch.array_scan(counts, counts, inclusive=True)
    # One read sizes both outputs: the scanned pair's last entry is the vertex and triangle totals.
    totals = read_scalar(counts)
    n_vertices, n_faces = int(totals[0]), int(totals[1])
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    if n_vertices > 0:
        _launch.launch(
            kernel_levelset.marching_cubes_emit,
            dim=(nx, ny, nz),
            inputs=[field, wp.float32(iso), lower, delta, wp.float32(edge_margin), table, counts],
            outputs=[vertices, faces],
            device=device,
        )
    return vertices, faces


@functools.cache
def _marching_cubes_table(device: str) -> wp.array[wp.int32]:
    """``kernels/levelset.MARCHING_CUBES_TABLE`` on the device ``device`` names, uploaded once."""
    return _launch.array(kernel_levelset.MARCHING_CUBES_TABLE, dtype=wp.int32, device=device)


def offset_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    distance: float,
    voxel_size: float | None = None,
    *,
    sign_mode: Literal["parity", "winding"] = "winding",
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Offset a surface by a signed distance, through the level set of its distance field.

    Positive grows the solid, negative shrinks it. The result is the exact set of points at distance
    ``distance`` from the input, sampled at ``voxel_size`` -- so it handles the cases a per-vertex
    displacement cannot: a shrink that makes a thin feature disappear, a growth that merges two
    nearby sheets into one, and any offset of a surface with concave regions, where neighbouring
    vertices moving along their own normals would cross.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer. Should describe a closed
        surface; an open one still offsets, but only ``sign_mode="winding"`` gives it a meaningful
        inside.
    distance
        Signed offset. Positive is outward.
    voxel_size
        Lattice spacing for the distance field. ``None`` derives it from ``distance`` --
        ``|distance| / 3``, so three samples span the band the level set has to be found in --
        clamped to between 64 and 256 samples across the mesh's bounding-box diagonal. The lower
        bound matters: a *large* inward offset leaves a small object, and a spacing set by the
        distance alone can be coarser than what is left of it.
    sign_mode
        Forwarded to
        [`ordito.proximity.signed_distance_grid`][ordito.proximity.signed_distance_grid]. Defaults
        to ``"winding"`` rather than ``"parity"``: an offset of a mesh with a few open rims is a
        common ask, and parity is the mode that goes wrong on one.
    bounds
        ``(lower, upper)`` box to sample before padding. ``None`` uses the mesh's own box, which the
        padding then extends to cover the offset.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(m,)`` vertices and ``(3 * k,)`` faces of the offset surface, ``m`` and ``k`` its vertex
        and face counts, on ``vertices.device``. **Empty** when the level set does not exist -- an
        inward offset larger than the object's own half-thickness has no points at that distance,
        which is the right answer rather than an error.

    Raises
    ------
    ValueError
        If ``distance`` is zero, ``voxel_size`` is not positive, ``faces`` is empty, or
        ``voxel_size`` is left to be derived on a mesh whose bounding box has zero extent.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    grown_v, grown_f = od.levelset.offset_mesh(v, f, 0.1)
    ```

    Notes
    -----
    **The defaults are the whole value of this function over composing the two calls it makes.** The
    field has to extend past the level set being extracted or that level set is clipped by the
    lattice boundary, so the padding is ``ceil(distance / voxel_size) + 2`` cells for an outward
    offset; and the spacing has to resolve the band, which is what ties it to ``distance`` rather
    than to the mesh. Getting either wrong yields a *plausible* surface with a hole in it, which is
    why they are computed here rather than left to the caller. A derived ``voxel_size`` is also
    bounded so that the lattice **including its padding** stays allocatable, which coarsens a very
    large outward offset rather than refusing it; pass ``voxel_size`` to override that.

    The output is a resampled surface: its triangulation is the marching-cubes lattice's, not the
    input's, and its vertex count is set by ``voxel_size`` rather than by the input's. Where the
    input's own triangulation must survive, the operation wanted is
    [`thicken_mesh`][ordito.levelset.thicken_mesh], not this.

    Accuracy is the lattice's, improved by marching cubes' linear interpolation across a cell.

    The extraction is
    [`signed_distance_level_set`][ordito.levelset.signed_distance_level_set], which searches for
    distances only near the offset surface.

    See Also
    --------
    [`thicken_mesh`][ordito.levelset.thicken_mesh]
        The topology-preserving shell, when the input triangulation should survive.
    [`ordito.proximity.signed_distance_grid`][ordito.proximity.signed_distance_grid]
        The field this extracts a level set from, when the field itself is wanted.
    [`ordito.levelset.marching_cubes`][ordito.levelset.marching_cubes]
        The extraction, and where the triangulation's own caveats live.
    """
    require_same_device(vertices=vertices, faces=faces)
    if distance == 0.0:
        raise ValueError("distance must be non-zero; an offset of zero is a resampling")
    if voxel_size is not None and voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if faces.size == 0:
        raise ValueError("offset_mesh needs at least one face")

    spacing = voxel_size
    if spacing is None:
        diagonal = float(od.bounds.enclosing_diagonal(vertices))
        # The band sets the spacing, and the two resolution bounds keep it usable: fine enough to
        # resolve what survives the offset, coarse enough to allocate. The cap is applied last, so
        # it wins over the floor. See the constants.
        padded_extent = diagonal + 2.0 * max(distance, 0.0)
        spacing = max(
            min(abs(distance) / _SAMPLES_PER_DISTANCE, diagonal / _MIN_AUTO_RESOLUTION),
            padded_extent / _MAX_AUTO_RESOLUTION,
        )
        if spacing <= 0.0:
            raise ValueError(
                "cannot derive a voxel_size for a mesh whose bounding box has zero extent; "
                "pass voxel_size explicitly"
            )
    # Only an outward offset leaves the input's box; an inward one needs just the two cells the
    # field itself wants so that the surface is enclosed.
    pad = 2 + (math.ceil(distance / spacing) if distance > 0.0 else 0)

    if bounds is None:
        bounds = od.bounds.aabb(vertices)
    shape, box = od.proximity.signed_distance_lattice(vertices, spacing, bounds=bounds, pad=pad)
    return signed_distance_level_set(
        vertices, faces, distance, shape, bounds=box, sign_mode=sign_mode
    )


def signed_distance_level_set(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    iso: float,
    shape: tuple[int, int, int],
    *,
    bounds: tuple[wp.vec3, wp.vec3],
    sign_mode: Literal["parity", "winding"] = "winding",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract the ``iso`` level set of a mesh's signed distance, sampled on a corner lattice.

    The surface [`marching_cubes`][ordito.levelset.marching_cubes] extracts from the field
    [`ordito.voxels.grid_points`][ordito.voxels.grid_points] and
    [`ordito.proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] would
    sample on the lattice of ``shape`` spanning ``bounds`` -- without necessarily sampling all of
    it. It is the extraction behind [`offset_mesh`][ordito.levelset.offset_mesh],
    [`ordito.reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] and
    [`ordito.repair.fix_self_intersections`][ordito.repair.fix_self_intersections]'s voxel method.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer, at least one face.
    iso
        Signed distance of the level to extract (Warp's convention: positive outside).
    shape
        ``(nx, ny, nz)`` lattice samples per axis, each at least 2.
    bounds
        ``(lower, upper)`` corners the lattice spans: sample ``(0, 0, 0)`` sits on ``lower`` and
        sample ``(nx - 1, ny - 1, nz - 1)`` on ``upper``. The spacing may differ per axis.
    sign_mode
        How the distance is signed, as in
        [`ordito.proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh].

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(m,)`` vertices and ``(3 * k,)`` faces of the level set, ``m`` and ``k`` its vertex and
        face counts, on ``vertices.device``. Empty where the lattice holds no point at distance
        ``iso``.

    Raises
    ------
    ValueError
        If ``faces`` is empty or any of ``shape`` is below 2.
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

    Notes
    -----
    The surface is always the one marching cubes extracts from the lattice of signed distances,
    but that lattice is not necessarily sampled node by node.

    With ``sign_mode="winding"`` the closest-point search at each node stops one cell diagonal
    past ``|iso|``: marching cubes interpolates only along edges that cross ``iso``, and both ends
    of such an edge lie within one cell diagonal of the level, so a farther node only has to land
    on the right side of it. The sign comes from the exact generalized winding number, counted as
    signed ray crossings along the lattice's columns with the input's boundary closed by a cone.
    Wherever that count is in doubt, or the winding number is close enough to 1/2 for Warp's own
    approximation of it to decide differently, the node takes Warp's sign, and a node past the
    search's reach that still ends a crossing edge (the sign of an open surface's field can change
    away from the surface) gets its full distance. The result is the dense lattice's surface,
    vertex for vertex and face for face. A boundary of very many edges or components, as a
    triangle soup has, samples every node instead.

    Otherwise, on a large lattice, a closed, consistently wound input is not sampled on every node
    either: an octree over the lattice discards every cell whose centre is farther from the level
    set than the cell is wide -- sound because a signed distance changes no faster than the
    distance travelled -- and the distance is evaluated only at the corners of the cells left, so
    the cost follows the surface's area rather than the lattice's volume. The surface is the dense
    lattice's, up to the order of its vertices and faces -- except where a lattice node's distance
    rounds onto the level itself, which the two place one rounding apart and so can triangulate
    differently, equally validly. An input with a boundary or an inconsistent winding samples every
    node: there the sign of its distance can jump away from the surface, across the region a hole
    spans, and such a field is not bounded by the distance travelled, so the octree could discard
    cells the dense extraction keeps.
    """
    require_same_device(vertices=vertices, faces=faces)
    if faces.size == 0:
        raise ValueError("signed_distance_level_set needs at least one face")
    if min(shape) < 2:
        raise ValueError(f"shape must be at least 2 along every axis, got {shape}")
    lower, upper = bounds
    n_nodes = math.prod(shape)
    if sign_mode == "winding" and n_nodes < _LATTICE_WINDING_BELOW_NODES:
        field = _winding_band_field(vertices, faces, iso, shape, bounds)
        if field is not None:
            return od.levelset.marching_cubes(field, iso, bounds=bounds)
    if n_nodes >= _SPARSE_LEVEL_SET_FROM_NODES and _is_closed_and_consistent(vertices, faces):
        mesh = wp.Mesh(
            points=vertices, indices=faces, support_winding_number=sign_mode == "winding"
        )

        def signed_distance(points: wp.array[wp.vec3]) -> wp.array[wp.float32]:
            return od.proximity.signed_distance_on_mesh(
                vertices, faces, points, sign_mode=sign_mode, mesh=mesh
            )

        # Typed as a union with the ``return_stats=True`` triple, which is not asked for here.
        return cast(
            "tuple[wp.array[wp.vec3], wp.array[wp.int32]]",
            sparse_marching_cubes(
                signed_distance,
                *shape,
                lower=lower,
                upper=upper,
                threshold=iso,
                device=vertices.device,
            ),
        )

    samples = od.voxels.grid_points(shape, bounds=bounds, device=vertices.device)
    distances = od.proximity.signed_distance_on_mesh(vertices, faces, samples, sign_mode=sign_mode)
    field = odt.as_array3d(distances.reshape(shape), wp.float32)
    return od.levelset.marching_cubes(field, iso, bounds=bounds)


def _winding_band_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    iso: float,
    shape: tuple[int, int, int],
    bounds: tuple[wp.vec3, wp.vec3],
) -> odt.Array3dFloat32 | None:
    """
    Return the winding-signed distance lattice as marching cubes reads it at ``iso``, or ``None``.

    Equal to ``proximity.signed_distance_on_mesh(sign_mode="winding")`` on every node an edge
    crossing ``iso`` ends at, and on the same side of ``iso`` everywhere else; ``kernels/
    levelset.py``'s header has the method. ``None`` when the boundary is too large for the cone
    evaluation or the lattice has no extent on an axis, for the caller to sample every node.
    """
    device = vertices.device
    n_faces = faces.size // 3
    lower, upper = bounds
    lower_f, upper_f = odt.vec3_floats(lower), odt.vec3_floats(upper)
    steps = [(upper_f[axis] - lower_f[axis]) / float(shape[axis] - 1) for axis in range(3)]
    if min(steps) <= 0.0:
        return None
    chain = _boundary_chain(faces, vertices.size)
    if chain.shape[0] > _MAX_CONE_EDGES:
        return None
    labels, sorted_keys, order, starts, radius = _cone_components(vertices, chain)
    n_nodes = math.prod(shape)
    if n_nodes * (starts.size - 1) > _MAX_CONE_WORK:
        return None

    # A search capped one cell diagonal past ``|iso|`` reaches every node an edge crossing ``iso``
    # can end at; the margin covers the rounding of the distances against the cap.
    diagonal = math.sqrt(sum(step * step for step in steps))
    scale = max(abs(component) for component in (*lower_f, *upper_f))
    rounding = 64.0 * float(np.spacing(np.float32(scale)))
    cap = abs(iso) + 1.01 * diagonal + rounding
    points = odt.as_array3d(
        od.voxels.grid_points(shape, bounds=bounds, device=device).reshape(shape), wp.vec3
    )
    mesh = wp.Mesh(points=vertices, indices=faces)
    distance = odt.empty_3d(shape, wp.float32, device=device)
    _launch.launch(
        kernel_levelset.capped_distance,
        dim=shape,
        inputs=[mesh.id, points, wp.float32(cap), distance],
        device=device,
    )

    # The column counts along z and, as their cross-check, along x.
    inv_step = wp.vec3(*(1.0 / step for step in steps))
    crossings = [_launch.zeros(shape, dtype=wp.int32, device=device) for _ in range(2)]
    ambiguous = _launch.zeros(shape, dtype=wp.int32, device=device)
    columns = [(shape[0], shape[1]), (shape[1], shape[2])]
    for axis in range(2):
        _launch.launch(
            kernel_levelset.rasterize_face_crossings,
            dim=n_faces,
            inputs=[vertices, faces, lower, inv_step, wp.int32(axis), crossings[axis], ambiguous],
            device=device,
        )
        if chain.shape[0] > 0:
            _launch.launch(
                kernel_levelset.rasterize_cone_crossings,
                dim=chain.shape[0],
                inputs=[
                    vertices,
                    chain,
                    labels,
                    lower,
                    inv_step,
                    wp.int32(axis),
                    crossings[axis],
                    ambiguous,
                ],
                device=device,
            )
        _launch.launch(
            kernel_levelset.column_suffix_sums,
            dim=columns[axis],
            inputs=[wp.int32(axis), crossings[axis]],
            device=device,
        )

    winding = odt.empty_3d(shape, wp.float32, device=device)
    slots = odt.empty_3d(shape, wp.int32, device=device)
    undecided = _launch.empty(_EXACT_WINDING_CAPACITY, dtype=wp.vec3, device=device)
    counts = _launch.zeros(2, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.lattice_winding_numbers,
        dim=shape,
        inputs=[
            points,
            crossings[0],
            crossings[1],
            ambiguous,
            distance,
            vertices,
            chain,
            order,
            sorted_keys,
            starts,
            radius,
            wp.float32(_WINDING_UNDECIDED_DELTA),
            wp.float32(abs(iso) - 1.01 * diagonal - rounding),
            wp.float32(rounding),
            winding,
            slots,
            undecided,
            counts,
        ],
        device=device,
    )
    # Whether any node is in doubt decides whether a second pass (and possibly Warp's solid-angle
    # BVH) is needed at all, which the host has to know to launch it.
    n_undecided = int(read_scalar(counts, 0))
    if n_undecided > 0:
        _settle_undecided_winding(
            vertices, faces, points, undecided, n_undecided, winding, slots, counts
        )

    values = odt.empty_3d(shape, wp.float32, device=device)
    _launch.launch(
        kernel_levelset.signed_band_values,
        dim=shape,
        inputs=[distance, winding, values],
        device=device,
    )
    field = odt.empty_3d(shape, wp.float32, device=device)
    _launch.launch(
        kernel_levelset.resolve_crossing_endpoints,
        dim=shape,
        inputs=[mesh.id, points, wp.float32(cap), wp.float32(iso), winding, values, field],
        device=device,
    )
    return field


def _boundary_chain(faces: wp.array[wp.int32], n_vertices: int) -> odt.Array2dInt32:
    """
    Return the ``(m, 2)`` boundary 1-chain of the faces.

    Each edge whose halfedges do not cancel, repeated by its net multiplicity and directed the way
    its net runs. Empty for a closed, consistently wound surface; an edge two faces traverse the
    same way appears twice.
    """
    device = faces.device
    n_halfedges = faces.size // 3 * 3
    mates = od.halfedge.halfedge_mates(faces, n_vertices)
    net = _launch.zeros(n_halfedges, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.boundary_chain_multiplicity,
        dim=n_halfedges,
        inputs=[faces, mates, net],
        device=device,
    )
    counts = _launch.empty(n_halfedges, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.boundary_chain_counts, dim=n_halfedges, inputs=[net, counts], device=device
    )
    # The chain's length sizes it.
    offsets, total = od.array.counts_to_offsets(counts)
    chain = odt.empty_2d((total, 2), wp.int32, device=device)
    if total > 0:
        _launch.launch(
            kernel_levelset.emit_boundary_chain,
            dim=n_halfedges,
            inputs=[faces, net, offsets, chain],
            device=device,
        )
    return chain


def _cone_components(
    vertices: wp.array[wp.vec3], chain: odt.Array2dInt32
) -> tuple[
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.float32],
]:
    """
    Group the chain into connected boundary components, each closed by its own cone.

    Returns ``(labels, sorted_keys, order, starts, radius)``: ``labels`` the smallest vertex of each
    vertex's component (the apex of its cone), the chain edges in component order as
    ``chain[order[t]]`` with ``sorted_keys[t]`` their apex, ``starts`` the total-terminated offsets
    of each component's run, and ``radius`` a ball about each apex holding its cone.
    """
    device = vertices.device
    m = int(chain.shape[0])
    if m == 0:
        empty = _launch.empty(0, dtype=wp.int32, device=device)
        return (
            empty,
            empty,
            empty,
            _launch.zeros(1, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.float32, device=device),
        )
    labels = od.graph.connected_component_labels_from_edges(
        chain, node_count=vertices.size, validate=False
    )
    keys = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.cone_component_keys, dim=m, inputs=[chain, labels, keys], device=device
    )
    sorted_keys, order = od.array.sort_and_argsort(keys)
    flags = _launch.empty(m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.mark_run_starts, dim=m, inputs=[sorted_keys, flags], device=device
    )
    # The component count sizes the cone tables.
    first = od.array.flatnonzero(flags)
    n_components = first.size
    starts = _launch.full(n_components + 1, m, dtype=wp.int32, device=device)
    _launch.copy(starts, first, count=n_components)
    radius = _launch.empty(n_components, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_levelset.cone_bounds,
        dim=n_components,
        inputs=[vertices, chain, order, sorted_keys, starts, radius],
        device=device,
    )
    return labels, sorted_keys, order, starts, radius


def _settle_undecided_winding(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3, Literal[3]],
    undecided: wp.array[wp.vec3],
    n_undecided: int,
    winding: odt.Array3dFloat32,
    slots: wp.array[wp.int32, Literal[3]],
    counts: wp.array[wp.int32],
) -> None:
    """
    Give every node in doubt the sign Warp's own winding number gives it, in place.

    The exact winding number, summed over every face with Warp's per-triangle solid angle, settles
    a node unless it too lies within the margin of 1/2; only those, or all of them when they are
    too many to sum, are evaluated by Warp itself, which needs its solid-angle BVH.
    """
    device = vertices.device
    n_faces = faces.size // 3
    n_warp = n_undecided
    if n_undecided <= _EXACT_WINDING_CAPACITY and n_undecided * n_faces <= _EXACT_WINDING_WORK:
        exact = _launch.zeros(n_undecided, dtype=wp.float32, device=device)
        n_slices = max(1, min(n_faces, (1 << 20) // n_undecided))
        _launch.launch(
            kernel_levelset.exact_winding_slices,
            dim=(n_undecided, n_slices),
            inputs=[vertices, faces, undecided[:n_undecided], wp.int32(n_slices), exact],
            device=device,
        )
        _launch.launch(
            kernel_levelset.take_exact_winding,
            dim=tuple(int(n) for n in winding.shape),
            inputs=[exact, wp.float32(_WINDING_UNDECIDED_DELTA), winding, slots, counts[1:]],
            device=device,
        )
        # Whether Warp's BVH must be built at all.
        n_warp = int(read_scalar(counts, 1))
    if n_warp > 0:
        winding_mesh = wp.Mesh(points=vertices, indices=faces, support_winding_number=True)
        _launch.launch(
            kernel_levelset.warp_winding_sign,
            dim=tuple(int(n) for n in winding.shape),
            inputs=[winding_mesh.id, points, slots, winding],
            device=device,
        )


def _is_closed_and_consistent(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> bool:
    """Whether every edge has exactly two faces that traverse it in opposite directions."""
    # Winding first: it is the cheaper of the two sorts, and an inconsistent input stops there.
    n_vertices = vertices.size
    return od.validation.is_winding_consistent(
        faces, n_vertices=n_vertices
    ) and od.validation.is_edge_manifold(faces, allow_boundary_edges=False, n_vertices=n_vertices)


def thicken_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    thickness: float,
    *,
    outside: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Turn a surface into a solid shell of given thickness, keeping the input's triangulation.

    The counterpart of [`offset_mesh`][ordito.levelset.offset_mesh] for the case where the answer
    should still be *this* mesh: every vertex is displaced along its own angle-weighted normal, a
    reversed copy of the surface becomes the shell's other side, and the two are joined along every
    boundary edge by a quad band. So the output is the input's connectivity twice over plus the band
    -- ``2 * n_faces + 2 * n_boundary_edges`` triangles -- and a per-vertex attribute follows
    through by duplication, which no resampled offset allows.

    On a **closed** surface there is no boundary to band, so the result is two nested shells: a
    solid with a cavity, which is what thickening a closed surface means.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer, consistently wound. The
        winding is what decides which side is "outside", so an inconsistent input gives a shell
        turned inside out in places -- run
        [`ordito.repair.make_winding_consistent`][ordito.repair.make_winding_consistent] first.
    thickness
        Distance the shell extends **inward**, opposite the vertex normals. Must be positive.
    outside
        Distance the shell also extends outward, so the input surface ends up ``outside`` in from
        the outer face. Zero (the default) leaves the input surface as the outer face exactly.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(2 * n_vertices,)`` vertices and ``(3 * k,)`` faces on ``vertices.device``,
        ``k = 2 * n_faces + 2 * n_boundary_edges`` triangles. The vertices hold the outward layer
        first, then the inward one, so input vertex ``v`` is at ``v`` and at ``v + n_vertices``.

    Raises
    ------
    ValueError
        If ``thickness`` is not positive, ``outside`` is negative, or ``faces`` is empty.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    shell_v, shell_f = od.levelset.thicken_mesh(v, f, 0.05)
    ```

    Notes
    -----
    **It can self-intersect, and it does not check.** Displacing along vertex normals folds the
    surface wherever ``thickness`` exceeds the local radius of curvature -- the inward layer of a
    tube thickened past its own radius passes through itself -- and no per-vertex method avoids
    that.
    The condition is exactly
    [`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask],
    so it is detectable in one call; where it happens, the operation wanted is a level-set
    [`offset_mesh`][ordito.levelset.offset_mesh], which cannot self-intersect by construction. Not
    guarding is deliberate: the guard would be a whole-mesh intersection test on every call, and the
    caller who needs it can run the one that names the faces.

    The normals are **angle-weighted** (the pseudonormal), which is the weighting that makes the
    displacement independent of how the incident triangles happen to be subdivided;
    [`ordito.vertices`][ordito.vertices] documents the three choices.

    See Also
    --------
    [`offset_mesh`][ordito.levelset.offset_mesh]
        The resampling offset, for a shell that must not self-intersect.
    [`vertices.vertex_normals`][ordito.vertices.vertex_normals] at `weighting="angle"`
        The displacement direction.
    [`ordito.boundary.oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]
        Where the band's orientation comes from.
    """
    require_same_device(vertices=vertices, faces=faces)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if outside < 0.0:
        raise ValueError("outside must be non-negative")
    if faces.size == 0:
        raise ValueError("thicken_mesh needs at least one face")

    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    normals = od.vertices.vertex_normals(vertices, faces, weighting="angle")
    rim = od.boundary.oriented_boundary_edges(vertices, faces)
    n_rim = int(rim.shape[0])

    out_vertices = _launch.empty(2 * n_vertices, dtype=wp.vec3, device=device)
    out_faces = _launch.empty(3 * (2 * n_faces + 2 * n_rim), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_levelset.shell_mesh,
        dim=n_vertices + n_faces + n_rim,
        inputs=[vertices, normals, wp.float32(outside), wp.float32(thickness), faces, rim],
        outputs=[out_vertices, out_faces],
        device=device,
    )
    return out_vertices, out_faces
