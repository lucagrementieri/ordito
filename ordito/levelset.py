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

import math
from typing import Literal

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
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


def marching_cubes(
    field: odt.Array3dFloat32, iso: float = 0.0, *, bounds: tuple[wp.vec3, wp.vec3] | None = None
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
        If any of ``field``'s dimensions is below 2.

    See Also
    --------
    [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform]
    [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson]
    [`ordito.proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]
    [`ordito.voxels.to_field`][ordito.voxels.to_field]
    [`ordito.voxels.grid_points`][ordito.voxels.grid_points]

    Notes
    -----
    A thin wrapper over Warp's own ``warp.MarchingCubes``, so the triangulation, its vertex
    deduplication and its handling of the ambiguous cube cases are Warp's rather than ordito's. The
    consequence worth knowing is that the result is **not guaranteed manifold** at an ambiguous
    cell, and can carry duplicate vertices where two cells agree on a crossing —
    [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] runs
    [`ordito.repair`][ordito.repair] over it for exactly that reason.
    """
    field = odt.as_array3d(field, wp.float32)
    shape = tuple(int(dim) for dim in field.shape)
    if min(shape) < 2:
        raise ValueError(f"field must be at least 2 wide along every axis, got {shape}")

    if bounds is None:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(float(shape[0] - 1), float(shape[1] - 1), float(shape[2] - 1))
    else:
        lower, upper = bounds
    return wp.MarchingCubes.extract_surface_marching_cubes(field, float(iso), lower, upper)


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

    field, box = od.proximity.signed_distance_grid(
        vertices, faces, spacing, bounds=bounds, pad=pad, sign_mode=sign_mode
    )
    return od.levelset.marching_cubes(field, distance, bounds=box)


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
