"""A cached, composition-based triangle mesh container (mirrors `trimesh.Trimesh`)."""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
import weakref
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Generic, Literal, NoReturn, TypeVar, cast, overload

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._common import read_scalar, require_nonempty_mesh, require_same_device
from ordito.config import is_deterministic
from ordito.kernels import mesh as kernel_mesh
from ordito.kernels import repair as kernel_repair
from ordito.kernels import transform as kernel_transform
from ordito.kernels.algorithms import connected_components as kernel_connected_components

_R = TypeVar("_R")

_LOGGER = logging.getLogger(__name__)

# `Trimesh.mesh_for_rays`' budget: the rays whose saving would pay for `traced_mesh`'s build, as a
# fixed part plus a part per face. On an RTX 5090 (Warp 1.18) the cuBQL build costs ~6 ms over the
# default plus ~0.1 us a face, and its tree saves ~0.4-1.6 ns a traced ray (any-hit bundles at the
# low end, closest-hit at the high end); the measured break-even sat at 84-255 rays a face from
# 0.87 M to 28 M faces and at 230-2 150 on 16-69 k faces, which this pair of constants follows.
_TRACED_MESH_FIXED_RAYS = 8_000_000
_TRACED_MESH_RAYS_PER_FACE = 160
# The normal matrix ``Trimesh.transform``'s launch receives when it carries no direction (unread).
_ZERO_MAT33 = wp.mat33()
# The attribute table of a `Trimesh` given none, shared: it is read-only.
_NO_ATTRIBUTES: Mapping[str, odt.ArrayNd] = MappingProxyType({})
# The attribute names a `Trimesh` reads for itself (`vertex_colors`, `face_colors`, `uv`), and the
# one dtype each must have: float RGBA colours, which blend through an edit, and float UVs.
_RESERVED_ATTRIBUTE_DTYPES: Mapping[str, type] = MappingProxyType({"color": wp.vec4, "uv": wp.vec2})

# Cached quantities that depend only on `faces` (topology), not on vertex positions. A
# functional update that keeps the same faces and the same vertex count (`with_vertices`)
# can carry these forward instead of recomputing them. Every new cached property below must
# be added to this set (if faces-only) or left out of it (if it also depends on `vertices`).
# `face_adjacency_unshared` is faces-only and belongs here; its neighbour `face_adjacency_angles`
# reads `face_normals` and is therefore correctly absent. The halfedge trio -- `halfedge_twins`,
# `vertex_one_rings`, `vertex_face_adjacency` -- is pure connectivity and belongs here too, while
# every discrete operator (`cotmatrix` through `vector_heat_operators`) is assembled from the
# connectivity *and* the geometry and must not: `tests/test_mesh.py` pins both halves of that rule,
# this set by parametrizing over it and its complement by name.
_TOPOLOGY_KEYS: frozenset[str] = frozenset(
    {
        "edges",
        "edges_sorted",
        "edges_face",
        "edges_unique",
        "edges_unique_inverse",
        "face_adjacency",
        "face_adjacency_edges",
        "face_adjacency_unshared",
        "face_connected_component_labels",
        "vertex_face_adjacency",
        "halfedge_twins",
        "halfedge_mates",
        "vertex_one_rings",
        "boundary_edges",
        "oriented_boundary_edges",
        "boundary_loops",
        "boundary_vertex_indices",
        "euler_characteristic",
        "is_edge_manifold",
        "is_vertex_manifold",
        "is_winding_consistent",
        "is_orientable",
        "faces_unique_edges",  # a reshaped view of `edges_unique_inverse`
        "body_count",
        # Unit weights on the directed edge adjacency, so the operator reads `faces` and not the
        # positions, and is invariant under any affine remap of the vertices. It is
        # winding-dependent on an *open* mesh, which `with_vertices` never changes but a mirroring
        # `transform` does; `_ORIENTATION_DEPENDENT_KEYS` carries that half.
        "laplacian_operator",
    }
)

# What `with_vertices` carries: the topology, plus the operators' sparsity. The pattern reads
# `faces` alone but stays out of `_TOPOLOGY_KEYS`, and so out of every transform stratum: those
# are gated by an elementwise comparison against recomputation, and the pattern's sort buffers
# hold scratch upper halves (and its `columns` an unwritten tail) that differ between two builds.
# Its corner slots `3f + e` also name corners of this very face buffer, so a winding flip would
# need it dropped anyway.
_WITH_VERTICES_CARRY: frozenset[str] = _TOPOLOGY_KEYS | frozenset({"mesh_operator_pattern"})

# --- Transform cache strata -------------------------------------------------
#
# How much of the cache survives depends on what the transform preserves.
# `tests/test_transform.py::test_carried_cache_matches_recomputation` checks each set against
# recomputation on both a closed and an open fixture, and fails if a key is added here that
# recomputation would disagree with.
#
# Each set is what is carried **verbatim**. A quantity that survives only up to a factor (areas
# under a scale) is dropped rather than rescaled, so "carried" always means "bit-identical to
# recomputation, up to float32 noise" -- one contract a test can check uniformly. Directions are
# the exception and are *rotated*; see `Trimesh.transform`.

# Dropped whenever face winding reverses, on top of whatever the metric class allows. Reversing a
# face's corners renumbers halfedges and permutes every per-corner table, which is invisible on a
# closed convex fixture and wrong in general.
_ORIENTATION_DEPENDENT_KEYS: frozenset[str] = frozenset(
    {
        "edges",  # each face's three rows reverse direction and permute
        "edges_sorted",
        "edges_unique_inverse",
        "halfedge_twins",  # halfedge `3f + k` names a different corner
        "halfedge_mates",  # likewise
        "vertex_one_rings",  # and the rotation around each vertex runs the other way
        "oriented_boundary_edges",  # directed by construction
        "boundary_edges",  # rows in halfedge order, which a flip permutes within each face
        "boundary_loops",  # so the loops come back reversed
        "face_angles",  # per-corner table, permuted
        "cotmatrix_entries",  # likewise -- the assembled `cotmatrix` is a sum and survives
        # Gauge is built from a reference halfedge, which moves -- and `basis_y = normal x basis_x`
        # flips with the winding, so after a mirror a carried frame is nowhere near the recomputed
        # one while the frame's own normal still agrees, being mapped by the inverse transpose.
        # Dropping the triple here is also what keeps `_carry_directions` from rotating it, which is
        # why that helper tests `survived` rather than the cache.
        "vertex_tangent_frames",
        "laplacian_operator",  # directed adjacency, asymmetric at an open boundary
        "faces_unique_edges",  # a view of `edges_unique_inverse`; carrying it alone would stale
    }
)

# Survives any *invertible* affine map: connectivity, plus the predicates a bijection preserves.
_AFFINE_CARRY: frozenset[str] = _TOPOLOGY_KEYS | frozenset(
    {"nondegenerate_faces", "is_watertight", "is_self_intersecting", "is_volume"}
)

# `face_adjacency_convex` is in no set. Convexity of a face pair *is* preserved by every
# invertible affine map, and the quantity it thresholds -- `face_adjacency_projections` --
# carries through an isometry closely. The **boolean** does not: on a mesh with coplanar
# neighbours the projection is exactly zero, so the tiny perturbation a rotation introduces
# crosses the threshold and flips the answer. Visible on `cave_cube`, whose box faces are coplanar
# by construction; invisible on any curved mesh. A thresholded quantity is not carryable just
# because the quantity is.

# Adds the angle functions. A similarity preserves angles, so it preserves cotangent weights --
# which is why the heaviest object here, the assembled `cotmatrix`, survives a scale.
#
# `vertex_tangent_frames` is here rather than one rung up because the gauge is a set of *unit*
# tangent directions: the fan's edges scale, the fit in the tangent plane scales with them, and the
# normalization divides the scale back out, so a similarity leaves the frame where
# an isometry does. It is the one entry in this set that `transform` **rotates** rather than
# carrying verbatim (see the header's note and `_carry_directions`), and membership here is
# load-bearing twice over: it is also what subjects the frame to the `_ORIENTATION_DEPENDENT_KEYS`
# subtraction, which a mirror needs -- a reflected frame's `basis_x` is nowhere near the recomputed
# one, where the *normal* still agrees because `transform_normals` maps it by the inverse
# transpose. Not one rung further down either: an affine map tilts the tangent plane by an amount
# that depends on the surface, so neither the frame nor the normal survives it.
_SIMILARITY_CARRY: frozenset[str] = _AFFINE_CARRY | frozenset(
    {
        "face_angles",
        "vertex_defects",
        "face_adjacency_angles",
        "cotmatrix_entries",
        "cotmatrix",
        "vertex_tangent_frames",
    }
)

# Adds the length and area quantities, which only an isometry leaves alone.
_ISOMETRY_CARRY: frozenset[str] = _SIMILARITY_CARRY | frozenset(
    {"face_areas", "area", "mean_edge_length", "edges_unique_length", "mass_matrix_entries"}
)

# `face_adjacency_projections` is in no set either, and not because it fails to survive an
# isometry -- it carries through cleanly. It is dropped so it cannot contradict
# `face_adjacency_convex`, which is exactly `projections <= TOLERANCE_MERGE` and is *not*
# carryable for the threshold reason above. Carrying one while recomputing the other lets a
# transformed mesh report a projection next to a `convex` that disagrees with it, on precisely the
# coplanar meshes where the threshold is fragile: one rotation of `cave_cube` leaves pairs within
# nanometres of the `1e-8` threshold band. The pair is cheap -- one kernel over the adjacency
# rows -- so recomputing both keeps them consistent for less than the bug is worth.

# The mass properties -- `volume`, `center_mass`, `moment_inertia` -- are in **no** set, and the
# reason is not the obvious one. Each is an integral over the tetrahedra from the origin to every
# face, which telescopes to an origin-independent answer only when the surface is *closed*. On an
# open mesh they are origin-dependent, so a translation changes all three, where the same move on a
# closed surface leaves them alone to float32 rounding. Carrying them would need the stratum
# to depend on `is_watertight`, which is a device readback and a second axis through every set;
# recomputing them is three readbacks and always right.

# A translation moves no direction at all, so normals, frames and the operator bundles built from
# them survive untouched -- the one rung where nothing has to be recomputed *or* rotated. The
# bounding box survives too, and `transform` shifts it on the host rather than reducing again.
#
# `vertex_tangent_frames` is *not* relisted here: it comes in through `_SIMILARITY_CARRY`, and a
# translation is the one kind for which `_carry_directions` returns before rotating anything, so
# the frame arrives verbatim exactly as this comment says.
_TRANSLATION_CARRY: frozenset[str] = _ISOMETRY_CARRY | frozenset(
    {
        "face_normals",
        "vertex_normals",
        "heat_operators",
        "vector_heat_operators",
        "enclosing_diagonal",
        "extents",  # the box shifts rather than changing shape
    }
)

# What survives reversing every face's winding with the positions untouched --
# [`Trimesh.invert`][ordito.mesh.Trimesh.invert]. Enumerated positively rather than by
# subtraction so a new cached property is *not* carried until someone classifies it.
#
# Three groups are absent and each for its own reason. The orientation-dependent tables go for the
# reason `_ORIENTATION_DEPENDENT_KEYS` lists. `face_normals`, `vertex_normals` and the mass
# properties **negate** rather than surviving, so `invert` flips the two normal buffers itself and
# drops the rest. And everything derived from a sign -- `is_volume`, `face_adjacency_convex`,
# `face_adjacency_projections`, the tangent frames and both heat bundles -- goes with them.
_INVERT_CARRY: frozenset[str] = frozenset(
    {
        # positions are untouched, so every quantity of the point set itself survives
        "face_areas",
        "area",
        "triangles_center",
        "centroid",
        "bounds",
        "extents",
        "enclosing_diagonal",
        "mean_edge_length",
        "vertex_defects",
        "nondegenerate_faces",
        # the undirected topology: a flip permutes each face's corners and changes none of these
        "edges_face",
        "edges_unique",
        "edges_unique_length",
        "face_adjacency",
        "face_adjacency_edges",
        "face_adjacency_unshared",
        "face_adjacency_angles",
        "face_connected_component_labels",
        "body_count",
        "vertex_face_adjacency",
        "boundary_vertex_indices",
        "euler_characteristic",
        "is_edge_manifold",
        "is_vertex_manifold",
        "is_winding_consistent",
        "is_orientable",
        "is_watertight",
        "is_self_intersecting",
        # the assembled operators, which are sums over faces and so corner-permutation invariant --
        # unlike the `cotmatrix_entries` table they are built from, which is per corner
        "cotmatrix",
        "mass_matrix_entries",
    }
)

_TRANSFORM_CARRY: dict[str, frozenset[str]] = {
    "translation": _TRANSLATION_CARRY,
    "rigid": _ISOMETRY_CARRY,
    "reflection": _ISOMETRY_CARRY,
    "similarity": _SIMILARITY_CARRY,
    "affine": _AFFINE_CARRY,
    # A singular map flattens the mesh, so even the predicates a bijection preserves are gone:
    # every face becomes degenerate, and a flattened surface self-intersects.
    "singular": _TOPOLOGY_KEYS,
}


class _CachedProperty(Generic[_R]):
    """
    Read-only descriptor storing its value in the owning `Trimesh`'s shared ``_cache`` dict.

    Unlike `functools.cached_property`, the value lives in a dict shared across all cached
    properties of the instance rather than in a dedicated instance attribute. This lets one
    property computation stash extra by-product values under sibling keys (e.g. computing
    `Trimesh.face_normals` also caches `Trimesh.face_areas`), and lets `Trimesh.invalidate`
    clear every cached value in a single ``dict.clear()``.
    """

    def __init__(self, func: Callable[[Trimesh], _R]) -> None:
        self._func = func
        self._key = func.__name__
        self.__doc__ = func.__doc__

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> _CachedProperty[_R]: ...
    @overload
    def __get__(self, obj: Trimesh, objtype: type | None = None) -> _R: ...
    def __get__(self, obj: Trimesh | None, objtype: type | None = None) -> _CachedProperty[_R] | _R:
        if obj is None:
            return self
        cache = obj._cache  # pyright: ignore[reportPrivateUsage]  # the owner class's own cache
        if self._key not in cache:
            cache[self._key] = self._func(obj)
        return cast(_R, cache[self._key])

    def __set__(self, obj: Trimesh, value: object) -> NoReturn:
        raise AttributeError(f"'{self._key}' is read-only; call invalidate() after in-place edits")


class _RowOrderedCachedProperty(_CachedProperty[tuple[wp.array[wp.int32], wp.array[wp.int32]]]):
    """
    `_CachedProperty` for a ``(values, offsets)`` CSR whose order within a row the setting decides.

    Under [`is_deterministic`][ordito.config.is_deterministic] such a property's rows are
    ascending, which a reproducible consumer relies on; with the setting off they are in arrival
    order. A value cached with the setting off is therefore rebuilt on its first read with it on,
    and the ascending one replaces it, serving both settings from then on. Which cached values are
    ascending is recorded on their ``values`` array, so the mark travels with the value when
    `with_vertices` or `transform` carries it forward and goes when `invalidate` drops it.
    """

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> _RowOrderedCachedProperty: ...
    @overload
    def __get__(
        self, obj: Trimesh, objtype: type | None = None
    ) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
    def __get__(
        self, obj: Trimesh | None, objtype: type | None = None
    ) -> _RowOrderedCachedProperty | tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        if obj is None:
            return self
        cache = obj._cache  # pyright: ignore[reportPrivateUsage]  # the owner class's own cache
        ordered = is_deterministic()
        cached = cast("tuple[wp.array[wp.int32], wp.array[wp.int32]] | None", cache.get(self._key))
        if cached is not None and (not ordered or cached[0] in _ASCENDING_ROWS):
            return cached
        value = self._func(obj)
        if ordered:
            _ASCENDING_ROWS.add(value[0])
        cache[self._key] = value
        return value


# The ``values`` arrays of the row-ordered values `_RowOrderedCachedProperty` built with the
# deterministic setting on, by identity.
_ASCENDING_ROWS: weakref.WeakSet[wp.array[wp.int32]] = weakref.WeakSet()


class Trimesh:
    """
    A triangle mesh with lazily cached derived quantities.

    Holds a ``vertices``/``faces`` pair (the same flat-``int32`` convention used throughout
    `ordito`) and computes derived geometry, topology and validity predicates on first access,
    caching the result. `Trimesh` is frozen: there are no setters, so a cached value can never
    silently go stale from a Python-level mutation. The ``warp.Mesh`` BVH is held by composition
    rather than inheritance and built lazily behind [`warp_mesh`][ordito.mesh.Trimesh.warp_mesh],
    since building it eagerly would waste GPU memory on meshes that never issue a ray or proximity
    query.

    Every cached property returns the same object on each access -- an array, a tuple of arrays or a
    sparse matrix, aliased across repeated accesses and (for `warp_mesh`) with the mesh's own
    `vertices`/`faces` buffers -- so callers must not mutate a returned buffer in place. That
    matters most where they are passed *into* the free functions as precomputed arguments: a wrapper
    that rewrites such a buffer copies it first. If a buffer is deliberately mutated in place by a
    kernel, call [`invalidate`][ordito.mesh.Trimesh.invalidate] afterward to drop every cached
    value (including the BVH); otherwise use [`with_vertices`][ordito.mesh.Trimesh.with_vertices] /
    [`with_faces`][ordito.mesh.Trimesh.with_faces], which return a new `Trimesh` and carry forward
    whichever cached values are still valid.

    A mesh also carries named per-vertex and per-face **attributes** -- a label, a colour, a UV, a
    scalar field, any array of one entry per vertex or per face -- in
    [`vertex_attributes`][ordito.mesh.Trimesh.vertex_attributes] and
    [`face_attributes`][ordito.mesh.Trimesh.face_attributes]. They are data the mesh holds, not
    quantities it derives, and they follow it: every method returning a new mesh carries them
    (through the edit's [`Provenance`][ordito.interpolation.Provenance] where the numbering
    changes), [`with_attributes`][ordito.mesh.Trimesh.with_attributes] sets or removes them, and
    [`with_edit`][ordito.mesh.Trimesh.with_edit] carries them through any free function called with
    ``return_provenance=True``.

    Two attribute names are the mesh's **visuals**, as in trimesh: ``"color"``, a ``wp.vec4`` RGBA
    colour in ``[0, 1]`` per vertex or per face, read as
    [`vertex_colors`][ordito.mesh.Trimesh.vertex_colors] and
    [`face_colors`][ordito.mesh.Trimesh.face_colors], and ``"uv"``, a ``wp.vec2`` texture
    coordinate per vertex, read as [`uv`][ordito.mesh.Trimesh.uv]. A textured mesh also holds a
    [`Material`][ordito.texture.Material], its texture image, as
    [`material`][ordito.mesh.Trimesh.material]. Being attributes, colours blend and UVs interpolate
    through every edit. UVs are per vertex, so a UV seam is a row of vertices split in two:
    [`ordito.repair.split_vertices_by_corner_values`][] makes one from per-corner texcoords.

    Scalar-valued properties (`area`, `centroid`, `bounds`, `enclosing_diagonal`,
    `mean_edge_length`, `euler_characteristic`, and every `is_*` predicate) synchronize from device
    to host on first access; the synchronized Python value is then cached like any other.

    Most of these are also what the free functions accept as an optional precomputed argument, so
    the cache is worth more than the repeat accesses on this class: pass `face_adjacency`,
    `halfedge_twins`, `vertex_one_rings`, `vertex_face_adjacency`, `face_normals`
    / `face_areas`, `cotmatrix_entries`, `bounds`, `laplacian_operator` or an operator bundle into
    the wrapper that takes it and the whole assembly is skipped. The discrete operators at the
    bottom of the class (`cotmatrix` through `vector_heat_operators`) are the heaviest of these and
    the reason a solver run over one mesh should go through a `Trimesh`. Caching pays off most when
    the same mesh is reused across many calls, and least where a single solve dominates the cost of
    any one call.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` or ``(n_faces, 3)`` triangle indices (flattened on construction).
    vertex_attributes
        Named ``(n_vertices,)`` arrays, one entry per vertex, of a dtype a
        [`Provenance`][ordito.interpolation.Provenance] carries: ``wp.bool``, ``wp.uint8``,
        ``wp.int32``, ``wp.int64``, ``wp.float32``, ``wp.float64``, or a ``wp.vec2`` / ``vec3`` /
        ``vec4`` / ``vec2d`` / ``vec3d`` / ``vec2i`` / ``vec3i`` or four-byte colour vector. Held,
        not copied (a non-contiguous view is made contiguous).
    face_attributes
        Named ``(n_faces,)`` arrays, one entry per face, of the same dtypes. In both tables
        ``"color"`` must be ``wp.vec4`` and ``"uv"`` ``wp.vec2``.
    material
        The texture the ``"uv"`` vertex attribute reads, or ``None``.
    initial_cache
        Optional pre-populated cache entries, keyed by property name. Used by
        [`from_warp_mesh`][ordito.mesh.Trimesh.from_warp_mesh] to seed `warp_mesh` without
        rebuilding its BVH, and by
        [`with_vertices`][ordito.mesh.Trimesh.with_vertices] to carry forward
        topology-only caches.

    Raises
    ------
    TypeError
        If ``vertices`` is not a rank-1 ``wp.vec3`` array, ``faces`` is not rank-1 or
        rank-2 ``wp.int32`` after flattening, an attribute is not a ``wp.array`` of a dtype
        listed above under a string name, a ``"color"`` or ``"uv"`` attribute has another dtype
        than its own, or ``material`` is not a [`Material`][ordito.texture.Material].
    ValueError
        If a rank-2 ``faces`` array does not have shape ``(n_faces, 3)``, the flattened
        face buffer's size is not a multiple of 3, or an attribute does not have one entry per
        vertex or face.
    RuntimeError
        If an attribute or the material's image is not on the device of ``vertices``.

    See Also
    --------
    [`trimesh.Trimesh`][]

    Examples
    --------
    Derived quantities are computed on first access and cached; methods return new meshes:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    mesh = od.Trimesh(vertices, faces)
    print(mesh.n_vertices, mesh.n_faces)  # 8 12
    print(f"{mesh.area:.4f} {mesh.volume:.4f}")  # 6.0000 1.0000
    print(mesh.is_watertight, mesh.euler_characteristic, mesh.edges_unique.shape)  # True 2 (18, 2)
    print(mesh.face_normals is mesh.face_normals)  # True
    points = wp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
    print(mesh.contains(points).numpy())  # [ True False]
    ```

    Attributes ride along: a submesh keeps the labels of the vertices and faces it keeps:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    labels = od.array.arange(8, device=vertices.device)
    mesh = od.Trimesh(vertices, faces, vertex_attributes={"label": labels})
    piece = mesh.submesh(wp.array([0, 1], dtype=wp.int32))
    print(sorted(piece.vertex_attributes), piece.vertex_attributes["label"].shape)  # ['label'] (4,)
    ```
    """

    # ``__weakref__``: a ``HeatSolver`` kept in the cache refers back to its mesh weakly, so the two
    # are no reference cycle and the mesh's device memory goes when the mesh does.
    __slots__ = (
        "__weakref__",
        "_cache",
        "_face_attributes",
        "_faces",
        "_material",
        "_vertex_attributes",
        "_vertices",
    )

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        *,
        vertex_attributes: Mapping[str, odt.ArrayNd] | None = None,
        face_attributes: Mapping[str, odt.ArrayNd] | None = None,
        material: od.texture.Material | None = None,
        initial_cache: dict[str, object] | None = None,
    ) -> None:
        """Construct a `Trimesh` from a vertex buffer and a face index buffer."""
        odt.ensure_ndim(vertices, 1, dtype=wp.vec3)
        if int(faces.ndim) == 2:
            if int(faces.shape[1]) != 3:
                raise ValueError(f"2D faces must have shape (n_faces, 3), got {tuple(faces.shape)}")
            faces = faces.reshape((-1,))
        odt.ensure_ndim(faces, 1, dtype=wp.int32)
        if int(faces.size) % 3 != 0:
            raise ValueError(f"faces size must be a multiple of 3, got {int(faces.size)}")

        self._vertices = vertices.contiguous()
        self._faces = faces.contiguous()
        self._cache: dict[str, object] = dict(initial_cache) if initial_cache is not None else {}
        self._vertex_attributes = _attribute_table(
            vertex_attributes, self.n_vertices, "vertex", self._vertices
        )
        self._face_attributes = _attribute_table(
            face_attributes, self.n_faces, "face", self._vertices
        )
        if material is not None:
            if not isinstance(material, od.texture.Material):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"material must be an ordito Material, got {type(material).__name__}"
                )
            if material.image is not None:
                require_same_device(vertices=self._vertices, material=material.image)
        self._material = material

    @classmethod
    def from_warp_mesh(cls, mesh: wp.Mesh) -> Trimesh:
        """
        Wrap an existing `warp.Mesh`, seeding `warp_mesh` so its BVH is never rebuilt.

        Parameters
        ----------
        mesh
            Source mesh; ``mesh.points`` and ``mesh.indices`` become the new `Trimesh`'s
            `vertices` and `faces`.

        Returns
        -------
        Trimesh
            New instance whose `warp_mesh` property is `mesh` itself.

        Examples
        --------
        The wrapped ``warp.Mesh`` becomes the cached BVH, so it is not rebuilt:

        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.icosphere(subdivisions=1)
        warp_mesh = wp.Mesh(points=vertices, indices=faces)
        mesh = od.Trimesh.from_warp_mesh(warp_mesh)
        print(mesh.warp_mesh is warp_mesh, mesh.n_faces)  # True 80
        ```
        """
        return cls(mesh.points, mesh.indices, initial_cache={"warp_mesh": mesh})

    @property
    def vertices(self) -> wp.array[wp.vec3]:
        """``(n_vertices,)`` mesh vertex positions."""
        return self._vertices

    @property
    def faces(self) -> wp.array[wp.int32]:
        """``(3 * n_faces,)`` flat triangle index buffer."""
        return self._faces

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the `vertices` buffer (its length, not the referenced count)."""
        return self._vertices.size

    @property
    def n_faces(self) -> int:
        """Number of triangles (``faces.size // 3``)."""
        return self._faces.size // 3

    @property
    def device(self) -> wp.Device:
        """Warp device holding `vertices` and `faces`."""
        return cast("wp.Device", self._vertices.device)

    @property
    def vertex_attributes(self) -> Mapping[str, odt.ArrayNd]:
        """
        Named ``(n_vertices,)`` per-vertex arrays the mesh carries, read-only.

        Set and removed with [`with_attributes`][ordito.mesh.Trimesh.with_attributes]; carried by
        every method returning a new mesh. An empty mapping when the mesh has none.

        Examples
        --------
        ```python
        import ordito as od

        vertices, faces = od.creation.box()
        heights = od.array.arange(8, device=vertices.device)
        mesh = od.Trimesh(vertices, faces, vertex_attributes={"height": heights})
        print(list(mesh.vertex_attributes), mesh.vertex_attributes["height"] is heights)
        # ['height'] True
        ```
        """
        return self._vertex_attributes

    @property
    def face_attributes(self) -> Mapping[str, odt.ArrayNd]:
        """
        Named ``(n_faces,)`` per-face arrays the mesh carries, read-only.

        Set and removed with [`with_attributes`][ordito.mesh.Trimesh.with_attributes]; carried by
        every method returning a new mesh. An empty mapping when the mesh has none.

        Examples
        --------
        ```python
        import ordito as od

        vertices, faces = od.creation.box()
        mesh = od.Trimesh(vertices, faces)
        print(len(mesh.face_attributes))  # 0
        ```
        """
        return self._face_attributes

    @property
    def vertex_colors(self) -> wp.array[wp.vec4] | None:
        """
        ``(n_vertices,)`` RGBA vertex colours in ``[0, 1]``, or ``None`` when the mesh has none.

        The ``"color"`` vertex attribute. Unlike trimesh, a mesh with no colours reads ``None``
        rather than a default grey, and face colours are not averaged onto the vertices on read:
        [`ordito.interpolation.average_onto_vertices`][] does that.

        See Also
        --------
        [`face_colors`][ordito.mesh.Trimesh.face_colors]
        [`to_color`][ordito.mesh.Trimesh.to_color]

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        red = wp.full(8, wp.vec4(1.0, 0.0, 0.0, 1.0), dtype=wp.vec4, device=vertices.device)
        mesh = od.Trimesh(vertices, faces, vertex_attributes={"color": red})
        print(mesh.vertex_colors is red, od.Trimesh(vertices, faces).vertex_colors)  # True None
        ```
        """
        return cast("wp.array[wp.vec4] | None", self._vertex_attributes.get("color"))

    @property
    def face_colors(self) -> wp.array[wp.vec4] | None:
        """
        ``(n_faces,)`` RGBA face colours in ``[0, 1]``, or ``None`` when the mesh has none.

        The ``"color"`` face attribute; [`ordito.interpolation.average_onto_faces`][] gives the
        mean of the vertex colours instead.

        See Also
        --------
        [`vertex_colors`][ordito.mesh.Trimesh.vertex_colors]

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        blue = wp.full(12, wp.vec4(0.0, 0.0, 1.0, 1.0), dtype=wp.vec4, device=vertices.device)
        mesh = od.Trimesh(vertices, faces, face_attributes={"color": blue})
        print(mesh.face_colors.shape)  # (12,)
        ```
        """
        return cast("wp.array[wp.vec4] | None", self._face_attributes.get("color"))

    @property
    def uv(self) -> wp.array[wp.vec2] | None:
        """
        ``(n_vertices,)`` texture coordinates, or ``None`` when the mesh has none.

        The ``"uv"`` vertex attribute, read by [`material`][ordito.mesh.Trimesh.material].

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        uv = wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)
        mesh = od.Trimesh(vertices, faces, vertex_attributes={"uv": uv})
        print(mesh.uv.shape)  # (8,)
        ```
        """
        return cast("wp.array[wp.vec2] | None", self._vertex_attributes.get("uv"))

    @property
    def material(self) -> od.texture.Material | None:
        """
        The texture [`uv`][ordito.mesh.Trimesh.uv] reads, or ``None`` when the mesh has none.

        Set with [`with_material`][ordito.mesh.Trimesh.with_material]; carried by every method
        returning a new mesh.

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.box()
        material = od.texture.Material(base_color=wp.vec4(0.8, 0.2, 0.2, 1.0))
        mesh = od.Trimesh(vertices, faces, material=material)
        print(mesh.material is material, mesh.submesh(mesh.faces[:3]).material is material)
        # True True
        ```
        """
        return self._material

    @_CachedProperty
    def warp_mesh(self) -> wp.Mesh:
        """
        Build the `vertices` / `faces` pair into a `warp.Mesh`, constructing its BVH.

        Raises
        ------
        ValueError
            If the mesh has zero faces — building a ``warp.Mesh`` with an empty BVH silently
            corrupts CUDA state.

        See Also
        --------
        [`ordito.ray`][ordito.ray]
        [`ordito.proximity`][ordito.proximity]
        """
        require_nonempty_mesh(self._faces, "Trimesh.warp_mesh")
        return wp.Mesh(points=self._vertices, indices=self._faces)

    @_CachedProperty
    def traced_mesh(self) -> wp.Mesh:
        """
        Build `warp_mesh`'s geometry into a `warp.Mesh` whose BVH is built for tracing many rays.

        On a CUDA device it is built by Warp's cuBQL builder (``bvh_constructor="cubql"``), whose
        tree returns the same hits as the default builder's and traces a ray bundle markedly faster,
        for a build that costs many times more. It therefore pays only on a mesh that will trace of
        the order of a hundred rays a face or more over its life;
        [`mesh_for_rays`][ordito.mesh.Trimesh.mesh_for_rays] chooses between the two automatically.

        Where cuBQL cannot help, this is `warp_mesh` itself, so no BVH is built twice: on the CPU
        device, whose default builder already produces the faster tree, and on a Warp build without
        cuBQL. The tree cannot carry Warp's winding-number data, so pass `warp_mesh` (or a mesh
        built with ``support_winding_number=True``) to the winding-signed queries.

        Raises
        ------
        ValueError
            If the mesh has zero faces.

        See Also
        --------
        [`warp_mesh`][ordito.mesh.Trimesh.warp_mesh]
        [`mesh_for_rays`][ordito.mesh.Trimesh.mesh_for_rays]
        [`ambient_occlusion`][ordito.visibility.ambient_occlusion]
        """
        require_nonempty_mesh(self._faces, "Trimesh.traced_mesh")
        if self.device.is_cpu or not wp.is_cubql_available():
            return self.warp_mesh
        return wp.Mesh(points=self._vertices, indices=self._faces, bvh_constructor="cubql")

    def mesh_for_rays(self, n_rays: int) -> wp.Mesh:
        """
        `warp_mesh` or `traced_mesh`, whichever tracing ``n_rays`` more rays should go through.

        The faster tree's extra build is paid once while its saving grows with every ray, and how
        many rays a mesh will trace over its life is not known when it is built: the rent-or-buy
        problem, decided the standard online way. Rays go through `warp_mesh` and are counted; once
        the count reaches the number whose saving would have paid for `traced_mesh`'s build, that
        mesh is built and returned from then on, which keeps the total within about twice the cost
        of the better fixed choice made in hindsight. A call that alone traces enough rays switches
        at once. An already built `traced_mesh` is always returned (its build is spent), and on the
        CPU device both properties are the same mesh.

        The count belongs to this `Trimesh` and is cleared by
        [`invalidate`][ordito.mesh.Trimesh.invalidate]. The switch is logged at ``INFO`` on the
        ``ordito.mesh`` logger. A caller who has already decided passes `warp_mesh` or
        `traced_mesh` directly instead.

        Parameters
        ----------
        n_rays
            Rays about to be traced through the returned mesh: for a hemisphere or cone bundle, the
            number of points times the rays per point.

        Returns
        -------
        wp.Mesh
            `traced_mesh` once the counted rays reach the budget, `warp_mesh` before.

        Raises
        ------
        ValueError
            If ``n_rays`` is negative, or the mesh has zero faces.

        Examples
        --------
        A small ray count goes through `warp_mesh`, whose BVH is the cheap one to build:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=2))
        rays = 16
        traced = mesh.mesh_for_rays(mesh.n_vertices * rays)
        print(traced is mesh.warp_mesh)  # True
        occlusion = od.visibility.ambient_occlusion(traced, mesh.vertices, n_rays=rays)
        print(occlusion.shape)  # (162,)
        ```
        """
        if n_rays < 0:
            raise ValueError(f"n_rays must be non-negative, got {n_rays}")
        if "traced_mesh" in self._cache or self.device.is_cpu or not wp.is_cubql_available():
            return self.traced_mesh
        counted = cast("int", self._cache.get("_rays_traced", 0)) + n_rays
        budget = _TRACED_MESH_FIXED_RAYS + _TRACED_MESH_RAYS_PER_FACE * self.n_faces
        if counted < budget:
            self._cache["_rays_traced"] = counted
            return self.warp_mesh
        _LOGGER.info(
            "Trimesh.mesh_for_rays: %d rays counted against a budget of %d for %d faces; "
            "building the cuBQL tree",
            counted,
            budget,
            self.n_faces,
        )
        return self.traced_mesh

    @_CachedProperty
    def face_normals(self) -> wp.array[wp.vec3]:
        """
        ``(n_faces,)`` unit face normals (zero for degenerate faces).

        Computed together with [`face_areas`][ordito.mesh.Trimesh.face_areas]; whichever is
        accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.face_normals`][]
        """
        normals, areas = od.triangles.face_normals_and_areas(self._vertices, self._faces)
        self._cache.setdefault("face_areas", areas)
        return normals

    @_CachedProperty
    def face_areas(self) -> wp.array[wp.float32]:
        """
        ``(n_faces,)`` triangle areas.

        Computed together with [`face_normals`][ordito.mesh.Trimesh.face_normals]; whichever
        is accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.area_faces`][]
        """
        # Compute the pair rather than reading the slot `face_normals` would have filled: that
        # only happens when `face_normals` is itself cold, and it can be cached *alone* -- a
        # rigid `transform` rotates it while dropping the areas a scale would have changed.
        # `setdefault` so an already-cached (carried or rotated) normal array is kept.
        normals, areas = od.triangles.face_normals_and_areas(self._vertices, self._faces)
        self._cache.setdefault("face_normals", normals)
        return areas

    @_CachedProperty
    def area(self) -> float:
        """
        Total surface area (sum of `face_areas`).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`trimesh.Trimesh.area`][]
        """
        return od.reduce.sum(cast("odt.Array1dFloat32", self.face_areas))

    @_CachedProperty
    def face_angles(self) -> odt.Array2dFloat32:
        """
        ``(n_faces, 3)`` interior angles in radians, aligned with each face's corners.

        See Also
        --------
        [`ordito.triangles.face_angles`][]
        [`trimesh.Trimesh.face_angles`][]
        """
        return od.triangles.face_angles(self._vertices, self._faces)

    @_CachedProperty
    def triangles_center(self) -> wp.array[wp.vec3]:
        """
        ``(n_faces,)`` barycentre of each triangle (the mean of its three corners).

        Notes
        -----
        Named for ``trimesh.Trimesh.triangles_center`` rather than for the function it calls;
        matching trimesh's property names is this facade's job, and the module-level function is
        [`face_centroids`][ordito.triangles.face_centroids].

        See Also
        --------
        [`ordito.triangles.face_centroids`][]
        [`centroid`][ordito.mesh.Trimesh.centroid]
            The area-weighted centre of the whole surface, which is not the mean of these.
        """
        return od.triangles.face_centroids(self._vertices, self._faces)

    @_CachedProperty
    def centroid(self) -> wp.vec3:
        """
        Area-weighted centroid of the mesh surface (all-``NaN`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`ordito.measures.surface_centroid`][]
        [`trimesh.Trimesh.centroid`][]
        """
        return od.measures.surface_centroid(self._vertices, self._faces)

    @_CachedProperty
    def volume(self) -> float:
        """
        Signed volume enclosed by the surface (meaningful only for a closed, consistent mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Accumulated in ``float32``,
        unlike the volume [`ordito.measures.moments`][] computes alongside
        [`center_mass`][ordito.mesh.Trimesh.center_mass] -- the two agree to that precision and
        are not the same number, which is why this property does not fill that call's cache.

        See Also
        --------
        [`ordito.measures.volume`][]
        [`is_volume`][ordito.mesh.Trimesh.is_volume]
            The check for the precondition this quantity needs.
        [`trimesh.Trimesh.volume`][]

        Examples
        --------
        Positive for outward-facing normals, negative once the winding is reversed:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box(extents=(1.0, 2.0, 3.0)))
        print(f"{mesh.volume:.4f} {mesh.invert().volume:.4f}")  # 6.0000 -6.0000
        ```
        """
        return od.measures.volume(self._vertices, self._faces)

    @_CachedProperty
    def center_mass(self) -> wp.vec3:
        """
        Centre of mass of the enclosed solid at unit density (all-``NaN`` for an empty mesh).

        **Not** [`centroid`][ordito.mesh.Trimesh.centroid], which is the area-weighted centre of
        the *surface*: the two differ on any solid whose mass is not distributed like its shell.

        Computed together with [`moment_inertia`][ordito.mesh.Trimesh.moment_inertia]; whichever
        is accessed first also caches the other.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (three, in fact -- every return
        of [`ordito.measures.moments`][] is a host value).

        See Also
        --------
        [`ordito.measures.moments`][]
        [`centroid`][ordito.mesh.Trimesh.centroid]
        [`trimesh.Trimesh.center_mass`][]
        """
        _volume, center, inertia = od.measures.moments(self._vertices, self._faces)
        self._cache.setdefault("moment_inertia", inertia)
        return center

    @_CachedProperty
    def moment_inertia(self) -> wp.mat33d:
        """
        ``(3, 3)`` inertia tensor about the centre of mass, at unit density.

        ``float64``: the second moments scale as ``length ** 5``, so a ``float32`` accumulation
        loses their low digits on any sizeable mesh.

        Computed together with [`center_mass`][ordito.mesh.Trimesh.center_mass]; whichever is
        accessed first also caches the other.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Being referred to the centre of
        mass rather than to the origin, it is unchanged by a translation --
        [`transform`][ordito.mesh.Trimesh.transform] carries it through one.

        See Also
        --------
        [`ordito.measures.moments`][]
        [`center_mass`][ordito.mesh.Trimesh.center_mass]
        [`trimesh.Trimesh.moment_inertia`][]

        Examples
        --------
        A unit cube's tensor has ``1 / 6`` on each diagonal entry:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        inertia = mesh.moment_inertia
        print(f"{float(inertia[0, 0]):.4f} {float(inertia[2, 2]):.4f}")  # 0.1667 0.1667
        ```
        """
        _volume, center, inertia = od.measures.moments(self._vertices, self._faces)
        self._cache.setdefault("center_mass", center)
        return inertia

    @_CachedProperty
    def bounds(self) -> tuple[wp.vec3, wp.vec3]:
        """
        Axis-aligned bounding box of `vertices`, as ``(min_bound, max_bound)``.

        Pass it to any wrapper taking a ``bounds=`` argument --
        [`marching_cubes`][ordito.levelset.marching_cubes],
        [`signed_distance_grid`][ordito.proximity.signed_distance_grid],
        [`query_nearest`][ordito.neighbors.query_nearest],
        [`grid_points`][ordito.voxels.grid_points] -- so the box is reduced once per mesh rather
        than once per call.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. ``(+inf, -inf)`` for an empty
        mesh, the [`aabb`][ordito.bounds.aabb] convention.

        See Also
        --------
        [`ordito.bounds.aabb`][]
        [`enclosing_diagonal`][ordito.mesh.Trimesh.enclosing_diagonal]
        """
        return od.bounds.aabb(self._vertices)

    @_CachedProperty
    def extents(self) -> wp.vec3:
        """
        Side lengths of the axis-aligned [`bounds`][ordito.mesh.Trimesh.bounds] box.

        Notes
        -----
        Costs no device work of its own: it subtracts the cached box on the host. ``(-inf, ...)``
        for an empty mesh, following [`bounds`][ordito.mesh.Trimesh.bounds]' ``(+inf, -inf)``
        convention.

        See Also
        --------
        [`bounds`][ordito.mesh.Trimesh.bounds]
        [`enclosing_diagonal`][ordito.mesh.Trimesh.enclosing_diagonal]
        [`trimesh.Trimesh.extents`][]
        """
        lower, upper = self.bounds
        return upper - lower

    @_CachedProperty
    def enclosing_diagonal(self) -> float:
        """
        Diagonal length of `bounds`: this package's default mesh-query search radius.

        Notes
        -----
        Costs no device work of its own, unlike
        [`enclosing_diagonal`][ordito.bounds.enclosing_diagonal], which reduces the box again --
        it reads the cached `bounds` on the host. ``inf`` for an empty mesh, that function's
        convention.

        A query wrapper's own default is the diagonal of the box around the mesh **and the query
        points**, which this is not; it is the right radius only when the queries lie inside the
        mesh's own box.

        See Also
        --------
        [`ordito.bounds.enclosing_diagonal`][]
        [`bounds`][ordito.mesh.Trimesh.bounds]
        ``trimesh.Trimesh.scale``
            The same quantity under trimesh's name for it (no Sphinx inventory entry to link).
        """
        lower, upper = self.bounds
        # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
        # Warp builtin at Python scope each route through builtin dispatch, several times dearer.
        # It computes in float64 where ``wp.length`` is float32, i.e. the correctly-rounded answer
        # for float32 corners.
        return math.dist(lower, upper)

    @_CachedProperty
    def vertex_normals(self) -> wp.array[wp.vec3]:
        """
        ``(n_vertices,)`` angle-weighted unit vertex normals.

        Matches `trimesh`'s default vertex-normal weighting (interior-angle weighted).

        See Also
        --------
        [`normals.vertex_normals`][ordito.normals.vertex_normals] at ``weighting="angle"``
        [`trimesh.Trimesh.vertex_normals`][]
        """
        # Only tables already held are passed: a missing one is formed inside the scatter, bit for
        # bit the property's own, rather than written to a cache slot just to be read back once.
        return od.normals.vertex_normals(
            self._vertices,
            self._faces,
            weighting="angle",
            face_normals=cast("wp.array[wp.vec3] | None", self._cache.get("face_normals")),
            face_weights=cast("odt.Array2dFloat32 | None", self._cache.get("face_angles")),
        )

    @_CachedProperty
    def vertex_defects(self) -> wp.array[wp.float32]:
        """
        ``(n_vertices,)`` discrete angle defect (``2π`` minus incident corner angles).

        See Also
        --------
        [`ordito.curvature.vertex_defects`][]

        Examples
        --------
        By Gauss-Bonnet the defects of a closed genus-0 surface sum to ``4 * pi``:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        print(f"{mesh.vertex_defects.numpy().sum():.4f}")  # 12.5664
        ```
        """
        return od.curvature.vertex_defects(self.n_vertices, self._faces, self.face_angles)

    @_CachedProperty
    def nondegenerate_faces(self) -> wp.array[wp.bool]:
        """
        ``(n_faces,)`` mask; ``True`` where the triangle has non-zero area.

        Named for ``trimesh.Trimesh.nondegenerate_faces`` rather than for the function it calls:
        matching trimesh's property names is this facade's whole job, so the property keeps the
        spelling a reader arrives with while the module-level function is
        [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask].

        See Also
        --------
        [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask]
        """
        return od.triangles.face_nondegenerate_mask(self._vertices, self._faces)

    @_CachedProperty
    def mean_edge_length(self) -> float:
        """
        Mean length over all ``3 * n_faces`` per-face edges (``0.0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`ordito.edges.mean_edge_length`][]
        """
        return od.edges.mean_edge_length(self._vertices, self._faces)

    @_CachedProperty
    def edges(self) -> odt.Array2dInt32:
        """
        ``(3 * n_faces, 2)`` directed triangle edges, in face-corner order.

        See Also
        --------
        [`ordito.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges`][]
        """
        return od.edges.faces_to_edges(self._faces, sorted=False)

    @_CachedProperty
    def edges_sorted(self) -> odt.Array2dInt32:
        """
        ``(3 * n_faces, 2)`` edges from `edges`, each row sorted (smaller index first).

        See Also
        --------
        [`ordito.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges_sorted`][]
        """
        return od.edges.faces_to_edges(self._faces, sorted=True)

    @_CachedProperty
    def edges_face(self) -> wp.array[wp.int32]:
        """
        ``(3 * n_faces,)`` face index for each row of `edges` / `edges_sorted`.

        See Also
        --------
        [`ordito.edges.edges_face`][]
        [`trimesh.Trimesh.edges_face`][]
        """
        return od.edges.edges_face(self._faces)

    @_CachedProperty
    def edges_unique(self) -> odt.Array2dInt32:
        """
        ``(m, 2)`` unique undirected edges, ``m <= 3 * n_faces``.

        Also caches [`edges_unique_inverse`][ordito.mesh.Trimesh.edges_unique_inverse] as a
        by-product.

        See Also
        --------
        [`ordito.edges.edges_unique`][]
        [`trimesh.Trimesh.edges_unique`][]

        Examples
        --------
        A cube's 12 triangles have 36 directed edges but 18 undirected ones:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        print(mesh.edges.shape, mesh.edges_unique.shape)  # (36, 2) (18, 2)
        ```
        """
        unique, inverse = od.edges.edges_unique(self._faces, n_vertices=self.n_vertices)
        self._cache.setdefault("edges_unique_inverse", inverse)
        return unique

    @_CachedProperty
    def edges_unique_inverse(self) -> wp.array[wp.int32]:
        """
        ``(3 * n_faces,)`` inverse mapping into `edges_unique` (reconstructs `edges_sorted`).

        See Also
        --------
        [`ordito.edges.edges_unique_inverse`][]
        [`trimesh.Trimesh.edges_unique_inverse`][]
        """
        # As in `face_areas`: `edges_unique` fills this only when it is itself cold, and a
        # mirroring `transform` carries it while dropping this one, whose row order reverses.
        unique, inverse = od.edges.edges_unique(self._faces, n_vertices=self.n_vertices)
        self._cache.setdefault("edges_unique", unique)
        return inverse

    @_CachedProperty
    def faces_unique_edges(self) -> odt.Array2dInt32:
        """
        ``(n_faces, 3)`` index into [`edges_unique`][ordito.mesh.Trimesh.edges_unique].

        Row ``f`` holds the unique-edge slot of each of face ``f``'s three edges, in face-corner
        order, so ``edges_unique[faces_unique_edges[f, k]]`` is that corner's edge.

        Notes
        -----
        Costs no device work: it is
        [`edges_unique_inverse`][ordito.mesh.Trimesh.edges_unique_inverse] viewed as ``(n, 3)``,
        and shares that buffer rather than copying it.

        See Also
        --------
        [`edges_unique_inverse`][ordito.mesh.Trimesh.edges_unique_inverse]
        [`trimesh.Trimesh.faces_unique_edges`][]
        """
        return odt.as_array2d(self.edges_unique_inverse.reshape((-1, 3)), wp.int32)

    @_CachedProperty
    def edges_unique_length(self) -> wp.array[wp.float32]:
        """
        ``(m,)`` Euclidean length of each `edges_unique` row.

        See Also
        --------
        [`ordito.edges.edges_unique_length`][]
        [`trimesh.Trimesh.edges_unique_length`][]
        """
        return od.edges.edges_unique_length(
            self._vertices, self._faces, unique_edges=self.edges_unique
        )

    @_CachedProperty
    def face_adjacency(self) -> odt.Array2dInt32:
        """
        ``(m, 2)`` face index pairs that share an undirected edge.

        Also caches [`face_adjacency_edges`][ordito.mesh.Trimesh.face_adjacency_edges] as a
        by-product.

        See Also
        --------
        [`ordito.adjacency.face_adjacency`][]
        [`trimesh.graph.face_adjacency`][]
        """
        adjacency, adjacency_edges = od.adjacency.face_adjacency(
            self._faces,
            return_edges=True,
            # The mesh knows its own vertex count, so the row-hash radix never has to be inferred
            # from a device reduction ending in a host readback.
            n_vertices=self._vertices.size,
        )
        self._cache.setdefault("face_adjacency_edges", adjacency_edges)
        return adjacency

    @_CachedProperty
    def face_adjacency_edges(self) -> odt.Array2dInt32:
        """
        ``(m, 2)`` shared vertex pair for each `face_adjacency` row.

        See Also
        --------
        [`ordito.adjacency.face_adjacency`][]
        """
        _ = self.face_adjacency
        # Both halves live in `_TOPOLOGY_KEYS`, so nothing splits this pair today; the read stays
        # a plain one rather than recomputing, and `face_adjacency` uses `setdefault` so a future
        # stratum that does split it fails loudly here instead of silently returning a stale edge.
        return cast("odt.Array2dInt32", self._cache["face_adjacency_edges"])

    @_CachedProperty
    def face_adjacency_unshared(self) -> odt.Array2dInt32:
        """
        ``(m, 2)`` vertex on each adjacent face not on their shared edge.

        See Also
        --------
        [`ordito.adjacency.face_adjacency_unshared`][]
        [`trimesh.graph.face_adjacency_unshared`][]
        """
        return od.adjacency.face_adjacency_unshared(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
        )

    @_CachedProperty
    def face_adjacency_angles(self) -> wp.array[wp.float32]:
        """
        ``(m,)`` unsigned angle in radians between each `face_adjacency` pair.

        See Also
        --------
        [`ordito.adjacency.face_adjacency_angles`][]
        [`trimesh.Trimesh.face_adjacency_angles`][]
        """
        return od.adjacency.face_adjacency_angles(
            self._vertices,
            self._faces,
            face_adjacency=self.face_adjacency,
            face_normals=self.face_normals,
        )

    @_CachedProperty
    def face_adjacency_projections(self) -> wp.array[wp.float32]:
        """
        ``(m,)`` projection of each adjacent pair's unshared vertices onto the other's plane.

        Negative where the pair is convex, which is the sign
        [`face_adjacency_convex`][ordito.mesh.Trimesh.face_adjacency_convex] thresholds.

        See Also
        --------
        [`ordito.adjacency.face_adjacency_projections`][]
        [`face_adjacency_convex`][ordito.mesh.Trimesh.face_adjacency_convex]
        [`trimesh.Trimesh.face_adjacency_projections`][]
        """
        return od.adjacency.face_adjacency_projections(
            self._vertices,
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
            face_adjacency_unshared=self.face_adjacency_unshared,
            face_normals=self.face_normals,
        )

    @_CachedProperty
    def face_adjacency_convex(self) -> wp.array[wp.bool]:
        """
        ``(m,)`` mask; ``True`` where an adjacent face pair meets convexly.

        See Also
        --------
        [`ordito.adjacency.face_adjacency_convex`][]
        [`face_adjacency_projections`][ordito.mesh.Trimesh.face_adjacency_projections]
        [`trimesh.Trimesh.face_adjacency_convex`][]

        Examples
        --------
        Every face pair on a convex solid meets convexly:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=1))
        print(mesh.face_adjacency.shape)  # (120, 2)
        print(bool(mesh.face_adjacency_convex.numpy().all()))  # True
        ```
        """
        return od.adjacency.face_adjacency_convex(
            self._vertices,
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
            face_adjacency_unshared=self.face_adjacency_unshared,
            face_normals=self.face_normals,
        )

    @_CachedProperty
    def face_connected_component_labels(self) -> wp.array[wp.int32]:
        """
        ``(n_faces,)`` connected-component label per face (face-adjacency graph).

        Notes
        -----
        With `face_adjacency` already cached, calls
        [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
        on it; otherwise
        [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels],
        which labels straight off the halfedge mates (the cached `halfedge_mates` when there are
        some) and builds no adjacency table. Same labelling engine, same answer.

        See Also
        --------
        [`ordito.adjacency.face_connected_component_labels`][]
        """
        adjacency = self._cache.get("face_adjacency")
        if adjacency is None:
            return od.adjacency.face_connected_component_labels(
                self._faces,
                n_vertices=self.n_vertices,
                mates=cast("wp.array[wp.int32] | None", self._cache.get("halfedge_mates")),
            )
        # ``validate=False``: the rows are face ids ``face_adjacency`` derived as ``e // 3``,
        # so they are below ``n_faces`` by construction and the check would only add a sync.
        return od.graph.connected_component_labels_from_edges(
            cast("odt.Array2dInt32", adjacency), node_count=self.n_faces, validate=False
        )

    @_CachedProperty
    def body_count(self) -> int:
        """
        Component count of the face-adjacency graph (``0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Counts components of the
        **face** graph, so an isolated vertex referenced by no face is not a body -- unlike
        ``igl.connected_components`` over the vertex adjacency, which counts each one.

        See Also
        --------
        [`face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels]
        [`ordito.combine.split`][]
            Materializes the bodies this counts.
        [`trimesh.Trimesh.body_count`][]

        Examples
        --------
        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box()) + od.Trimesh(*od.creation.tetrahedron())
        print(mesh.body_count)  # 2
        ```
        """
        if self.n_faces == 0:
            return 0
        # Both labelling paths name each component by its smallest face id, so the components are
        # the faces that are their own label: one launch and one read, no deduplication.
        labels = self.face_connected_component_labels
        count = _launch.zeros(1, dtype=wp.int32, device=labels.device)
        _launch.launch(
            kernel_connected_components.count_component_roots,
            dim=self.n_faces,
            inputs=[labels, count],
            device=labels.device,
        )
        # The count is the returned host integer.
        return int(read_scalar(count, 0))

    @_RowOrderedCachedProperty
    def vertex_face_adjacency(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        """
        Incidence CSR of the faces touching each vertex, as ``(vertex_faces, offsets)``.

        Each row is a *set*: use `vertex_one_rings` where the rotational order around the vertex
        is what matters, at the price of needing an edge-manifold mesh.

        See Also
        --------
        [`ordito.adjacency.vertex_face_adjacency`][]
        [`vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings]
        """
        return od.adjacency.vertex_face_adjacency(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def halfedge_twins(self) -> wp.array[wp.int32]:
        """
        ``(3 * n_faces,)`` opposite halfedge of every halfedge, or ``-1`` on a boundary.

        Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to ``faces[3f + (k + 1) % 3]``, so
        ``next`` and ``prev`` are index arithmetic and this array is all a walk needs to cross an
        edge.

        Raises
        ------
        ValueError
            Propagated from [`halfedge_twins`][ordito.halfedge.halfedge_twins] when an undirected
            edge carries three or more halfedges, i.e. the mesh is not edge-manifold.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`ordito.halfedge.halfedge_twins`][]
        [`vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings]
        """
        return od.halfedge.halfedge_twins(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def halfedge_mates(self) -> wp.array[wp.int32]:
        """
        ``(3 * n_faces,)`` partner of every halfedge on its undirected edge, with no validation.

        The pairing the boundary and validity properties share: each is computed from this one
        table instead of pairing the halfedges again. Pass it as ``mates=`` to the
        [`ordito.boundary`][] and [`ordito.validation`][] functions that take it.

        See Also
        --------
        [`ordito.halfedge.halfedge_mates`][]
        [`halfedge_twins`][ordito.mesh.Trimesh.halfedge_twins]
        """
        return od.halfedge.halfedge_mates(self._faces, self.n_vertices)

    @_CachedProperty
    def vertex_one_rings(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]:
        """
        Counter-clockwise outgoing halfedges per vertex: ``(ring_halfedges, offsets, is_boundary)``.

        Built from the cached `halfedge_twins`, or together with it (and caching it) when it is not
        cached yet, so accessing either first pays for that array once.

        Raises
        ------
        ValueError
            Propagated from [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] when a vertex's
            rotation closes before its whole fan is covered (a pinched, vertex-non-manifold
            vertex), or from `halfedge_twins` on an edge-non-manifold mesh.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`ordito.halfedge.vertex_one_rings`][]
        [`halfedge_twins`][ordito.mesh.Trimesh.halfedge_twins]
        [`vertex_face_adjacency`][ordito.mesh.Trimesh.vertex_face_adjacency]
        """
        twins = cast("wp.array[wp.int32] | None", self._cache.get("halfedge_twins"))
        if twins is not None:
            return od.halfedge.vertex_one_rings(
                self._faces, twins=twins, n_vertices=self.n_vertices
            )
        # Built here, the twin table comes back with the rings for one synchronization, rather than
        # one for its own check and another for the check of passing it back in.
        ring_halfedges, offsets, is_boundary, twins = od.halfedge.vertex_one_rings(
            self._faces, n_vertices=self.n_vertices, return_twins=True
        )
        self._cache.setdefault("halfedge_twins", twins)
        return ring_halfedges, offsets, is_boundary

    @_CachedProperty
    def boundary_edges(self) -> odt.Array2dInt32:
        """
        ``(n_boundary, 2)`` undirected boundary edges (each row sorted, min-first).

        See Also
        --------
        [`ordito.boundary.boundary_edges`][]
        """
        return od.boundary.boundary_edges(self._vertices, self._faces, mates=self.halfedge_mates)

    @_CachedProperty
    def oriented_boundary_edges(self) -> odt.Array2dInt32:
        """
        ``(n_boundary, 2)`` directed boundary edges, preserving face winding.

        See Also
        --------
        [`ordito.boundary.oriented_boundary_edges`][]
        """
        return od.boundary.oriented_boundary_edges(
            self._vertices, self._faces, mates=self.halfedge_mates
        )

    @_CachedProperty
    def boundary_loops(self) -> list[wp.array[wp.int32]]:
        """
        One ordered vertex-index array per boundary loop (empty list if the mesh is closed).

        See Also
        --------
        [`ordito.boundary.boundary_loops`][]

        Examples
        --------
        A tube has two boundary loops; a closed solid has none:

        ```python
        import warp as wp

        import ordito as od

        profile = wp.array([[1.0, 0.0], [1.0, 1.0]], dtype=wp.vec2)
        tube = od.Trimesh(*od.creation.revolve(profile, sections=8))
        print([loop.size for loop in tube.boundary_loops])  # [8, 8]
        print(od.Trimesh(*od.creation.box()).boundary_loops)  # []
        ```
        """
        return od.boundary.boundary_loops(self._vertices, self._faces, mates=self.halfedge_mates)

    @_CachedProperty
    def boundary_vertex_indices(self) -> wp.array[wp.int32]:
        """
        ``(n_boundary_vertices,)`` sorted unique vertex indices lying on the mesh boundary.

        See Also
        --------
        [`ordito.boundary.boundary_vertex_indices`][]
        """
        return od.boundary.boundary_vertex_indices(
            self._vertices, self._faces, mates=self.halfedge_mates
        )

    @_CachedProperty
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` (``0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Calls
        [`euler_characteristic`][ordito.measures.euler_characteristic], except that a cached
        `edges_unique` supplies the edge count instead of a fresh sort.

        See Also
        --------
        [`ordito.measures.euler_characteristic`][]
        [`trimesh.Trimesh.euler_number`][]
        """
        if self.n_faces == 0:
            return 0
        unique_edges = self._cache.get("edges_unique")
        if unique_edges is None:
            return od.measures.euler_characteristic(self._faces, n_vertices=self.n_vertices)
        # The referenced count is the membership mask's popcount, as in the measures function.
        mask = od.array.indices_to_mask(self._faces, self.n_vertices)
        n_edges = int(cast("odt.Array2dInt32", unique_edges).shape[0])
        return od.reduce.sum(mask) - n_edges + self.n_faces

    @_CachedProperty
    def is_edge_manifold(self) -> bool:
        """
        Whether every undirected edge is shared by one or two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Read off `halfedge_mates` when
        they are already cached -- one pass over the mate codes -- and otherwise by counting the
        edges, which is cheaper than pairing the halfedges for it.

        See Also
        --------
        [`ordito.validation.is_edge_manifold`][]
        """
        return od.validation.is_edge_manifold(
            self._faces,
            n_vertices=self.n_vertices,
            mates=cast("wp.array[wp.int32] | None", self._cache.get("halfedge_mates")),
        )

    @_CachedProperty
    def is_vertex_manifold(self) -> bool:
        """
        Whether every referenced vertex has a single edge-connected fan of faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Reuses `face_adjacency` when
        it is already cached; otherwise the corner graph is hooked straight off the sorted edge
        keys, which is cheaper than building the adjacency table for it.

        See Also
        --------
        [`ordito.validation.is_vertex_manifold`][]
        """
        if "face_adjacency" not in self._cache:
            return od.validation.is_vertex_manifold(
                self._faces, n_vertices=self.n_vertices, mates=self.halfedge_mates
            )
        return od.validation.is_vertex_manifold(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
            n_vertices=self.n_vertices,
        )

    @_CachedProperty
    def is_winding_consistent(self) -> bool:
        """
        Whether every shared edge is traversed in opposite directions by its two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`ordito.validation.is_winding_consistent`][]
        [`trimesh.Trimesh.is_winding_consistent`][]
        """
        return od.validation.is_winding_consistent(
            self._faces, n_vertices=self.n_vertices, mates=self.halfedge_mates
        )

    @_CachedProperty
    def is_orientable(self) -> bool:
        """
        Whether the faces admit a consistent orientation (allowing per-face flips).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. The orientation is propagated
        over `halfedge_mates`, the table the other mate-based predicates here share, so no
        face-adjacency table is built for it.

        See Also
        --------
        [`ordito.validation.is_orientable`][]
        """
        return od.validation.is_orientable(
            self._faces, n_vertices=self.n_vertices, mates=self.halfedge_mates
        )

    @_CachedProperty
    def is_watertight(self) -> bool:
        """
        Whether the mesh bounds a closed volume with no self-intersections.

        Follows Open3D's ``IsWatertight`` semantics (edge-manifold-closed, vertex-manifold, and
        not self-intersecting) — **not** the same predicate as `trimesh.Trimesh.is_watertight`,
        which only checks that every edge is shared by exactly two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. The self-intersection broad
        phase runs over `warp_mesh` -- the cached BVH when there is one, and otherwise one built
        and cached only once the manifold tests pass, so an open mesh never pays for it -- the
        same reuse [`is_self_intersecting`][ordito.mesh.Trimesh.is_self_intersecting] gets by
        taking a `wp.Mesh` directly.

        See Also
        --------
        [`ordito.validation.is_watertight`][]
        [`is_volume`][ordito.mesh.Trimesh.is_volume]
        """
        if self.n_faces == 0:
            return True
        # A cached BVH is reused; otherwise it is built -- and cached for later readers -- only
        # once the manifold tests pass, since an open mesh never reaches the broad phase.
        mesh = self._cache.get("warp_mesh")
        return od.validation.is_watertight(
            self._vertices,
            self._faces,
            mesh=cast("wp.Mesh", mesh) if mesh is not None else lambda: self.warp_mesh,
            mates=self.halfedge_mates,
        )

    @_CachedProperty
    def is_self_intersecting(self) -> bool:
        """
        Whether any two non-adjacent triangles of the mesh intersect.

        Raises
        ------
        ValueError
            If the mesh has zero faces (see [`warp_mesh`][ordito.mesh.Trimesh.warp_mesh]).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Reuses the cached
        `warp_mesh` BVH for its broad phase.

        See Also
        --------
        [`ordito.validation.is_self_intersecting`][]
        """
        return od.validation.is_self_intersecting(self.warp_mesh)

    @_CachedProperty
    def is_volume(self) -> bool:
        """
        Whether the mesh is a valid closed volume with outward-facing normals.

        Watertight in the trimesh sense (every edge shared by exactly two faces),
        winding-consistent, and enclosing a positive signed volume.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`ordito.validation.is_volume`][]
        [`trimesh.Trimesh.is_volume`][]
        """
        return od.validation.is_volume(self._vertices, self._faces, mates=self.halfedge_mates)

    # Discrete operators: the assemblies every solve over this mesh shares. Heavier than everything
    # above (a sparse build each, and the two heat bundles several) and reached by fewer callers,
    # so they sit last -- but they are also where the cache pays most, since each is what a whole
    # family of wrappers accepts as its precomputed argument.

    @_CachedProperty
    def cotmatrix_entries(self) -> odt.Array2dFloat32:
        """
        ``(n_faces, 3)`` per-triangle half-cotangent weights, in igl's edge order.

        The ``float32`` table, and that costs a ``float64`` consumer nothing: the free function
        computes these weights in ``float32`` -- the vertex precision -- whatever dtype is asked
        for, and casts on write, so a ``float64`` request would return this same table widened.
        Every consumer accepts either precision and casts to the *matrix* dtype in one build, so
        pass this to [`cotmatrix`][ordito.laplacian.cotmatrix],
        [`connection_laplacian`][ordito.laplacian.connection_laplacian],
        [`heat_operators`][ordito.heat.heat_operators] or
        [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix] at either
        precision -- which is what this class's own ``float32`` `cotmatrix` and ``float64``
        `heat_operators` both do.

        See Also
        --------
        [`ordito.laplacian.cotmatrix_entries`][]
        [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix]
        """
        return od.laplacian.cotmatrix_entries(self._vertices, self._faces)

    @_CachedProperty
    def mesh_operator_pattern(self) -> od.laplacian.MeshOperatorPattern:
        """
        Sparsity of the cotangent Laplacian on these faces, with each entry's contributing corners.

        Depends on the faces alone, so every operator over it -- `cotmatrix`, `heat_operators` and
        `vector_heat_operators`, and a [`HeatSolver`][ordito.heat.HeatSolver] at any diffusion
        time -- is assembled from this one pattern and pays only for its values, and the matrices
        share its ``offsets`` and ``columns``.

        Notes
        -----
        Kept for the mesh's lifetime like every cached property: the sorted entry keys and their
        corner slots, arrays proportional to the face count that a one-shot assembly would free.
        It survives [`with_vertices`][ordito.mesh.Trimesh.with_vertices]; the meshes
        [`transform`][ordito.mesh.Trimesh.transform], [`invert`][ordito.mesh.Trimesh.invert] and
        [`with_faces`][ordito.mesh.Trimesh.with_faces] return build their own. Pass it as
        ``pattern=`` to [`cotmatrix`][ordito.laplacian.cotmatrix],
        [`connection_laplacian`][ordito.laplacian.connection_laplacian] or the heat assemblies
        when building one of them for this mesh at another dtype or diffusion time.

        See Also
        --------
        [`ordito.laplacian.mesh_operator_pattern`][]
        [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix]

        Examples
        --------
        The cached operators share the pattern's row offsets:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=1))
        pattern = mesh.mesh_operator_pattern
        print(pattern.n_vertices, mesh.cotmatrix.offsets is pattern.offsets)  # 42 True
        ```
        """
        return od.laplacian.mesh_operator_pattern(self._faces, self.n_vertices)

    @_CachedProperty
    def cotmatrix(self) -> odt.BsrMatrix[wp.float32]:
        """
        ``(n_vertices, n_vertices)`` cotangent stiffness (discrete Laplace-Beltrami) matrix.

        Assembled from the cached `cotmatrix_entries`. Diagonal entries are negative and each row
        sums to zero, so ``-L`` is positive semi-definite on a closed mesh.

        Notes
        -----
        ``float32``, which is the right precision for a *product* (an energy, a residual, a filter
        weight) and not for a solve -- the heat solvers and
        [`ordito.parametrization`][ordito.parametrization] assemble their own ``float64``
        operators, and `heat_operators` caches the one they share.

        See Also
        --------
        [`ordito.laplacian.cotmatrix`][]
        [`cotmatrix_entries`][ordito.mesh.Trimesh.cotmatrix_entries]
        [`mass_matrix_entries`][ordito.mesh.Trimesh.mass_matrix_entries]
        [`ordito.energies.k_harmonic`][]
            Takes this operator and `mass_matrix_entries` as its two arguments.

        Examples
        --------
        Each row of the cotangent matrix sums to zero:

        ```python
        import numpy as np
        import warp as wp

        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=1))
        ones = wp.full(mesh.n_vertices, 1.0, dtype=wp.float32)
        row_sums = wp.zeros(mesh.n_vertices, dtype=wp.float32)
        wp.sparse.bsr_mv(mesh.cotmatrix, ones, row_sums)
        print(mesh.cotmatrix.shape, bool(np.abs(row_sums.numpy()).max() < 1e-5))  # (42, 42) True
        ```
        """
        return od.laplacian.cotmatrix(
            self._vertices,
            self._faces,
            cot_entries=self.cotmatrix_entries,
            pattern=self.mesh_operator_pattern,
        )

    @_CachedProperty
    def mass_matrix_entries(self) -> wp.array[wp.float32]:
        """
        ``(n_vertices,)`` barycentric lumped mass: a third of each incident triangle's area.

        Built from the cached `face_areas`, so this is a scatter and nothing else.

        Notes
        -----
        ``float32``. A ``float64`` solve wants
        [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries] at that dtype, which
        accumulates in the requested precision rather than casting this -- the two are not the same
        array widened.

        See Also
        --------
        [`ordito.laplacian.mass_matrix_entries`][]
        [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix]

        Examples
        --------
        The lumped masses add up to the surface area:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        print(f"{mesh.mass_matrix_entries.numpy().sum():.4f} {mesh.area:.4f}")  # 6.0000 6.0000
        ```
        """
        return od.laplacian.mass_matrix_entries(
            self._vertices, self._faces, face_areas=self.face_areas
        )

    @_CachedProperty
    def laplacian_operator(self) -> odt.BsrMatrix[wp.float32]:
        """
        ``(n_vertices, n_vertices)`` row-normalized 1-ring averaging (uniform / umbrella) Laplacian.

        Exactly what every position filter in [`ordito.smoothing`][ordito.smoothing] builds for
        itself when its ``laplacian_operator=`` argument is ``None``, so passing this hoists the
        assembly out of a multi-filter or multi-call pass.

        Notes
        -----
        The **directed** ``mesh.edges`` adjacency, the shared default.
        [`filter_neighborhood_average`][ordito.smoothing.filter_neighborhood_average] is the one
        filter that wants the *symmetric* adjacency instead
        ([`laplacian`][ordito.laplacian.laplacian] at ``symmetric=True``) and must not be handed
        this one; the two differ only on a mesh with an open boundary.

        See Also
        --------
        [`ordito.laplacian.laplacian`][]
        [`ordito.smoothing.filter_laplacian`][]
        [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix]
        """
        return od.laplacian.laplacian(self._vertices, self._faces)

    @_CachedProperty
    def halfedge_transport_angles(self) -> wp.array[wp.float64]:
        """
        ``(3 * n_faces,)`` rotation carrying a tangent vector across each halfedge, in (-pi, pi].

        [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles] formed over
        this mesh's cached `halfedge_twins` and `vertex_one_rings`, so the rings are walked once for
        the tangent frames and the connection alike. `vector_heat_operators` and a
        [`HeatSolver`][ordito.heat.HeatSolver] at any diffusion time assemble their connection
        from it.

        See Also
        --------
        [`ordito.tangent_space.halfedge_transport_angles`][]
        [`vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames]

        Examples
        --------
        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=1))
        print(mesh.halfedge_transport_angles.shape)  # (240,)
        ```
        """
        # Imported here so that importing this module does not load the kernel module.
        from ordito.kernels import tangent_space as kernel_tangent_space

        device = self._faces.device
        if self._faces.size == 0:
            return _launch.empty(0, dtype=wp.float64, device=device)
        # The ring walk forms the corner angles in ``float64`` itself; a cached ``float32``
        # `face_angles` table is not read, so the answer does not depend on what ran first.
        tangent_angles = od.tangent_space.halfedge_tangent_angles(
            self._vertices, self._faces, rings=self.vertex_one_rings
        )
        # The transport kernel itself, since `halfedge_transport_angles` verifies a supplied twin
        # table with a launch and a readback: these are this mesh's own `halfedge_twins`, which the
        # `vertex_one_rings` build above already verified.
        twins = self.halfedge_twins
        rho = _launch.empty(twins.size, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_tangent_space.halfedge_transport_angles,
            dim=twins.size,
            inputs=[twins, tangent_angles, rho],
            device=device,
        )
        return rho

    @_CachedProperty
    def vertex_tangent_frames(
        self,
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]:
        """
        ``(n_vertices,)`` orthonormal tangent frame per vertex as ``(basis_x, basis_y, normal)``.

        The gauge every 2-D tangent quantity on this mesh is measured in. Built over the cached
        `vertex_one_rings`, and its third element **is** `vertex_normals`: the cached normals when
        there are any, otherwise the frames form them and `vertex_normals` then serves those.

        Raises
        ------
        ValueError
            Propagated from `vertex_one_rings` on a non-manifold mesh.

        See Also
        --------
        [`ordito.tangent_space.vertex_tangent_frames`][]
        [`vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators]
        """
        # Without cached normals the frame kernel gathers the angle-weighted sums over the ring it
        # walks: one launch where building `vertex_normals` first costs four and its face normal
        # and corner angle tables. The ``float64`` sum is taken in ring order where
        # `vertex_normals` scatters in face order; ``float32`` terms leave the narrowed normal
        # unchanged by the order (bit-identical on the CPU over the fixtures and scan meshes).
        frames = od.tangent_space.vertex_tangent_frames(
            self._vertices,
            self._faces,
            normals=cast("wp.array[wp.vec3] | None", self._cache.get("vertex_normals")),
            rings=self.vertex_one_rings,
        )
        self._cache.setdefault("vertex_normals", frames[2])
        return frames

    @_CachedProperty
    def heat_operators(self) -> od.heat.HeatOperators:
        """
        Source-independent operator bundle for the heat method, at the default diffusion time.

        What [`heat_geodesic`][ordito.heat.heat_geodesic] and the other heat solvers use when
        given this mesh (through `heat_solver`): everything in the bundle depends on the mesh
        alone, so distance from many different source sets costs one assembly.

        Notes
        -----
        At ``t = None`` (the squared mean unique-edge length, ``igl::heat_geodesics``' default) and
        ``use_robust=False``. Another ``t`` or ``use_robust`` is
        [`heat_solver`][ordito.mesh.Trimesh.heat_solver]'s, which keeps its own bundle. Every
        bundle is assembled over the cached `mesh_operator_pattern`, which this mesh keeps.

        See Also
        --------
        [`ordito.heat.heat_operators`][]
        [`vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators]
        """
        # The cached half-cotangents when there are any; otherwise the assembly forms them in its
        # own pass over the faces (``cotmatrix_entries``' table, bit for bit), and
        # `cotmatrix_entries` then serves those.
        operators = od.heat.heat_operators(
            self._vertices,
            self._faces,
            cot_entries=cast("odt.Array2dFloat32 | None", self._cache.get("cotmatrix_entries")),
            pattern=self.mesh_operator_pattern,
        )
        self._cache.setdefault("cotmatrix_entries", operators[5])
        return operators

    @_CachedProperty
    def vector_heat_operators(self) -> od.heat.VectorHeatOperators:
        """
        Operator bundle for the vector heat method, at the default diffusion time.

        ``(vector_system, scalar, frames, preconditioner)``: the ``2 x 2``-block connection system,
        the scalar `heat_operators`, the `vertex_tangent_frames`, and the connection system's own
        Jacobi preconditioner. What
        [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors],
        [`log_map`][ordito.heat.log_map] and
        [`heat_signed_distance`][ordito.heat.heat_signed_distance] use when given this mesh.

        Its second and third fields are this class's own `heat_operators` and
        `vertex_tangent_frames`, so the three properties share one assembly however they are
        reached: built first, this assembles the scalar bundle itself and caches it as
        `heat_operators`. Both systems are assembled over the cached `mesh_operator_pattern`, and
        the connection is formed over the cached `halfedge_twins` and `vertex_one_rings`.

        Notes
        -----
        At ``t = None``, the same default `heat_operators` uses -- which is load-bearing rather
        than incidental: [`log_map`][ordito.heat.log_map]'s radius is asserted to *be* the
        [`heat_geodesic`][ordito.heat.heat_geodesic] distance, so the two systems must
        share a diffusion time.

        See Also
        --------
        [`ordito.heat.vector_heat_operators`][]
        [`heat_operators`][ordito.mesh.Trimesh.heat_operators]
        [`vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames]
        """
        bundle = od.heat.vector_heat_operators(
            self._vertices,
            self._faces,
            scalar_operators=cast(
                "od.heat.HeatOperators | None", self._cache.get("heat_operators")
            ),
            frames=self.vertex_tangent_frames,
            pattern=self.mesh_operator_pattern,
            transport_angles=self.halfedge_transport_angles,
        )
        # The scalar bundle the vector assembly built alongside (one sparsity pattern for both)
        # is this class's own `heat_operators`, and its half-cotangents are `cotmatrix_entries`.
        self._cache.setdefault("heat_operators", bundle[1])
        self._cache.setdefault("cotmatrix_entries", bundle[1][5])
        return bundle

    def heat_solver(
        self,
        t: float | None = None,
        *,
        use_robust: bool = False,
        method: Literal["potpourri3d", "meshlab", "igl"] = "potpourri3d",
    ) -> od.heat.HeatSolver:
        """
        Return this mesh's [`HeatSolver`][ordito.heat.HeatSolver] at diffusion time ``t``.

        Kept on the mesh, one per ``(t, use_robust, method)``: the operators the heat solvers
        assemble and the factorizations they build live there, so the functions of
        [`ordito.heat`][ordito.heat] given this mesh reuse them on every later call. At the
        defaults its operators are `heat_operators` and `vector_heat_operators`.

        Parameters
        ----------
        t
            Diffusion time; ``None`` for the squared mean unique-edge length, lengthened on a mesh
            too fine for it (see [`HeatSolver`][ordito.heat.HeatSolver]).
        use_robust
            Build the scalar operators from mollified edge lengths (see
            [`heat_operators`][ordito.heat.heat_operators]), or at ``method="potpourri3d"`` on
            geometry-central's intrinsic Delaunay tufted cover (see
            [`heat_geodesic`][ordito.heat.heat_geodesic]).
        method
            Whose discretization the operators follow (see
            [`heat_operators`][ordito.heat.heat_operators]): ``"potpourri3d"`` (default, the
            vector heat method's), or ``"meshlab"`` / ``"igl"``, what
            [`heat_geodesic`][ordito.heat.heat_geodesic] runs on at its own ``method``.

        Returns
        -------
        HeatSolver
            The kept solver; the same object on every call with the same arguments.

        Raises
        ------
        ValueError
            If ``method`` is not ``"potpourri3d"``, ``"meshlab"`` or ``"igl"``.

        Notes
        -----
        A solver's factorizations hold device memory for as long as this mesh lives;
        [`release_factorizations`][ordito.mesh.Trimesh.release_factorizations] returns it without
        dropping the mesh. They are not carried into the meshes `transform`, `invert` or
        `with_vertices` return.

        See Also
        --------
        [`ordito.heat.HeatSolver`][]
        [`release_factorizations`][ordito.mesh.Trimesh.release_factorizations]

        Examples
        --------
        The solver is kept on the mesh, so repeated distance queries share its operators:

        ```python
        import warp as wp

        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=3))
        solver = mesh.heat_solver(method="meshlab")
        print(mesh.heat_solver(method="meshlab") is solver)  # True
        distances = od.heat.heat_geodesic(mesh, wp.array([0], dtype=wp.int32))
        print(f"{distances.numpy().max():.1f}")  # 3.1
        ```
        """
        solvers = cast(
            "dict[tuple[float | None, bool, str], od.heat.HeatSolver]",
            self._cache.setdefault("_heat_solvers", {}),
        )
        key = (None if t is None else float(t), bool(use_robust), method)
        solver = solvers.get(key)
        if solver is None:
            solver = solvers[key] = od.heat.HeatSolver(
                self, t, use_robust=use_robust, method=method
            )
        return solver

    def fixed_vertex_solver(self) -> od.parametrization.FixedVertexSolver:
        """
        Return this mesh's [`FixedVertexSolver`][ordito.parametrization.FixedVertexSolver].

        Kept on the mesh: the factorizations of the fixed-vertex maps' systems
        ([`harmonic`][ordito.parametrization.harmonic], [`tutte`][ordito.parametrization.tutte],
        [`lscm`][ordito.parametrization.lscm]) live there, so a call given this mesh that fixes the
        same vertices as a kept factorization runs no iteration. Build one ahead of the calls with
        its [`factor`][ordito.parametrization.FixedVertexSolver.factor], as the example shows.

        Returns
        -------
        FixedVertexSolver
            The kept solver; the same object on every call.

        Notes
        -----
        Its factorizations hold device memory for as long as this mesh lives;
        [`release_factorizations`][ordito.mesh.Trimesh.release_factorizations] returns it without
        dropping the mesh. They are not carried into the meshes `transform`, `invert` or
        `with_vertices` return.

        See Also
        --------
        [`heat_solver`][ordito.mesh.Trimesh.heat_solver]
        [`release_factorizations`][ordito.mesh.Trimesh.release_factorizations]

        Examples
        --------
        Factor the harmonic system once; later maps fixing the same boundary reuse it:

        ```python
        import ordito as od

        vertices, faces = od.creation.grid(count=(8, 8))
        mesh = od.Trimesh(vertices, faces)
        loop = od.boundary.longest_boundary_loop(vertices, faces)
        print(mesh.fixed_vertex_solver().factor("harmonic", loop, k=2))  # True
        circle = od.parametrization.map_vertices_to_circle(vertices, loop)
        uv = od.parametrization.harmonic(mesh, loop, circle, k=2)
        print(uv.shape)  # (64,)
        uv_again = od.parametrization.harmonic(mesh, loop, circle, k=2)  # no iteration either
        ```
        """
        solver = self._cache.get("_fixed_vertex_solver")
        if solver is None:
            solver = self._cache["_fixed_vertex_solver"] = od.parametrization.FixedVertexSolver(
                self
            )
        return cast("od.parametrization.FixedVertexSolver", solver)

    def contains(self, points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
        """
        Test which query points lie inside the mesh, by ray parity against the cached BVH.

        Parameters
        ----------
        points
            ``(n,)`` query positions.

        Returns
        -------
        wp.array[wp.bool]
            ``(n,)`` mask; ``True`` where the point is inside.

        Raises
        ------
        ValueError
            If the mesh has zero faces (see [`warp_mesh`][ordito.mesh.Trimesh.warp_mesh]).

        Notes
        -----
        Only meaningful for a closed, consistently wound surface --
        [`is_volume`][ordito.mesh.Trimesh.is_volume] is the check for it. Reuses the cached
        `warp_mesh` rather than building a BVH per call, which is the whole reason to reach for
        this instead of the free function.

        See Also
        --------
        [`ordito.ray.contains_points`][]
        [`ordito.proximity.signed_distance_on_mesh`][]
            A signed distance rather than a bit, from the same BVH.
        [`trimesh.Trimesh.contains`][]

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=2))
        points = wp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3)
        print(mesh.contains(points).numpy())  # [ True  True False]
        ```
        """
        return od.ray.contains_points(self.warp_mesh, points)

    def sample(
        self, count: int, *, seed: int | None = None
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
        """
        Sample points uniformly over the surface, area-weighted.

        Parameters
        ----------
        count
            Number of samples to draw.
        seed
            Seed for the sampler; non-reproducible when ``None``.

        Returns
        -------
        points : wp.array[wp.vec3]
            ``(count,)`` sampled positions.
        face_indices : wp.array[wp.int32]
            ``(count,)`` index of the face each sample landed on.

        Notes
        -----
        Passes the cached [`face_areas`][ordito.mesh.Trimesh.face_areas] as the sampling weight,
        so repeated draws from one mesh share that reduction. Returns the face indices as well as
        the points, where ``trimesh.Trimesh.sample`` returns them only on request.

        See Also
        --------
        [`ordito.sample.sample_surface`][]
        [`ordito.sample.sample_surface_blue_noise`][]
            Sampling with a minimum separation rather than independently.
        [`trimesh.Trimesh.sample`][]

        Examples
        --------
        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        points, face_indices = mesh.sample(100, seed=0)
        print(points.shape, face_indices.shape)  # (100,) (100,)
        print(bool((abs(points.numpy()) <= 0.5 + 1e-6).all()))  # True
        ```
        """
        return od.sample.sample_surface(
            self._vertices, self._faces, count, face_weight=self.face_areas, seed=seed
        )

    def transform(
        self, matrix: wp.mat44 | wp.array[wp.mat44], *, assume: str | None = None
    ) -> Trimesh:
        """
        Return a new `Trimesh` under an affine transform, carrying forward whatever survives it.

        `Trimesh` is frozen, so this returns a new instance rather than moving this one -- and that
        is the *fast* path, not merely the safe one. How much of the cache survives is decided by
        what the transform preserves ([`classify_transform`][ordito.transform.classify_transform]),
        and for a rigid motion that is everything expensive: angles, areas, the cotangent table and
        the assembled [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix] are all isometry invariants, so
        only the directions and the bounding box are touched.

        | transform | recomputed on the new mesh |
        |---|---|
        | translation | the BVH, and nothing else |
        | rigid, reflection | the BVH and `bounds`; directions are *rotated*, not rebuilt |
        | similarity | the above, plus the length and area quantities |
        | affine | everything but connectivity |

        Face winding is reversed when ``matrix`` mirrors, which keeps normals outward and costs the
        orientation-dependent caches on top of the row above --
        [`vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings], the directed edge tables and
        the per-corner tables among them.

        Carrying the cache forward is cheaper than transforming the buffers and rebuilding a
        `Trimesh` around them whenever the mesh is reused for further queries: the saving is a count
        of assembly launches skipped, so it is worth reaching for on a mesh carried through a
        sequence of poses, and worth little on a mesh transformed once and used once.

        Parameters
        ----------
        matrix
            ``4x4`` transform, as a scalar ``wp.mat44`` or a ``(1,)`` ``wp.array[wp.mat44]`` --
            an [`icp`][ordito.registration.icp] result can be passed straight through. Build one
            with [`translation_matrix`][ordito.transform.translation_matrix],
            [`rotation_matrix`][ordito.transform.rotation_matrix],
            [`scale_matrix`][ordito.transform.scale_matrix] or
            [`reflection_matrix`][ordito.transform.reflection_matrix].
        assume
            A [`TransformKind`][ordito.transform.TransformKind] value promising what ``matrix``
            preserves, skipping the classification. Use it when a matrix composed from many
            ``float32`` factors has drifted far enough off orthogonality to be classified
            `TransformKind.AFFINE`, which is safe but discards the cache. **An incorrect promise
            silently corrupts every carried value** -- there is no check.

        Returns
        -------
        Trimesh
            New instance on the transformed vertices, sharing whatever cached values survive.
            ``self`` when ``matrix`` is the identity. `faces` is the same buffer as this mesh's
            unless ``matrix`` mirrors, in which case it is a new, corner-reversed buffer.

        Raises
        ------
        ValueError
            If ``assume`` is not a `TransformKind` value.

        Notes
        -----
        Carried values are **aliased**, not copied, exactly as with
        [`with_vertices`][ordito.mesh.Trimesh.with_vertices]: the returned mesh and this one hold
        the same array objects. That is safe because neither can mutate them, and it is why this
        class offers no in-place transform -- an edit to a shared buffer would corrupt every mesh
        derived from it, with nothing to invalidate them. Use
        [`transform_points`][ordito.transform.transform_points] with ``out=`` for in-place work on
        raw buffers.

        See Also
        --------
        [`ordito.transform.transform_mesh`][]
            The buffer-level form, without the cache.
        [`with_vertices`][ordito.mesh.Trimesh.with_vertices]
        [`invalidate`][ordito.mesh.Trimesh.invalidate]

        Examples
        --------
        A rotation is rigid, so the assembled cotangent matrix is carried over, not rebuilt:

        ```python
        import math

        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=2))
        cotmatrix = mesh.cotmatrix
        spun = mesh.transform(od.transform.rotation_matrix((0.0, 0.0, 1.0), math.pi / 3))
        print(spun.cotmatrix is cotmatrix)  # True
        moved = mesh.transform(od.transform.translation_matrix((0.0, 0.0, 5.0)))
        print(f"{moved.bounds[0][2]:.4f} {moved.bounds[1][2]:.4f}")  # 4.0000 6.0000
        ```
        """
        # Every use below -- the classification, the `reverses_orientation` check, the normal
        # matrix in `_carry_directions`, up to two `as_mat44` calls in `_carry_box` -- is a host
        # branch over a matrix that does not change between them. Resolving it once (the one
        # readback a device array costs, none for an assumed identity) and passing the host value
        # down reads the 16 floats once.
        matrix_host = od.transform.as_mat44(matrix) if assume is None else None
        kind = str(
            od.transform.TransformKind(assume)
            if matrix_host is None
            else od.transform.classify_transform(matrix_host)
        )
        if kind == od.transform.TransformKind.IDENTITY:
            return self
        if matrix_host is None:
            matrix_host = od.transform.as_mat44(matrix)

        carry = _TRANSFORM_CARRY[kind]
        reverse = od.transform.reverses_orientation(matrix_host)
        if reverse:
            carry = carry - _ORIENTATION_DEPENDENT_KEYS
        survived = {key: value for key, value in self._cache.items() if key in carry}
        new_vertices, new_faces = self._carry_directions(kind, matrix_host, reverse, survived)
        self._carry_box(kind, matrix_host, survived)
        return self._derived(new_vertices, new_faces, survived)

    def _carry_directions(
        self, kind: str, matrix: wp.mat44, reverse: bool, survived: dict[str, object]
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
        """
        Transform the buffers, and rotate the cached direction quantities into ``survived``.

        Returns the new ``(vertices, faces)``: the positions placed by ``matrix`` and, when
        ``reverse`` (a mirror), the faces corner-reversed as
        [`transform_mesh`][ordito.transform.transform_mesh] writes them; otherwise connectivity is
        untouched, so the face buffer is shared rather than copied, as `invert` shares the
        vertices: neither mesh can mutate it. On CUDA, positions, faces and every carried direction
        are one ``kernels/mesh.transform_mesh_and_directions`` launch; the CPU takes a launch a
        buffer (`_carry_directions_by_buffer`) unless a small mesh carries directions.

        The directions are rotated rather than dropped only for a similarity or stronger, where a
        direction's image is determined by the transform alone. A translation moves no direction
        at all and carries these verbatim through `_TRANSLATION_CARRY`; an affine map tilts them
        by an amount that depends on the surface, so there they are recomputed -- and recomputing
        `face_normals` yields `face_areas` with it, which an affine map does not preserve anyway.

        [`transform_normals`][ordito.transform.transform_normals] is the right map for the frame's
        tangents too, not just for its normal: for a similarity ``M = sR`` the inverse transpose is
        ``R / s``, which normalizes to the same unit vector the forward map does.

        **The two normal branches test `self._cache` and the frame branch tests `survived`, and
        that asymmetry is deliberate.** `survived` is the carry set after the
        `_ORIENTATION_DEPENDENT_KEYS` subtraction, so testing it is what stops a *mirror* from
        rotating a gauge that a mirror does not preserve. The normals need no such gate --
        `transform_normals` maps them by the inverse transpose, which is already correct under a
        reflection. So do
        not "simplify" the frame branch to read `self._cache` like its siblings; it would silently
        carry a mirrored frame. Mutation-probed: that edit fails the reflection arm of
        `test_transform_rotates_the_tangent_frames_where_it_can` on every fixture, closed and open
        alike, with "carried a frame it does not preserve".
        """
        device = self._vertices.device
        # Every cached direction maps through the same launch: up to four arrays, an absent one
        # passed as a null array its guard skips. Each output is allocated only for an input that
        # is present.
        face_normals = vertex_normals = basis_x = basis_y = None
        if kind in ("rigid", "reflection", "similarity"):
            face_normals = cast("wp.array[wp.vec3] | None", self._cache.get("face_normals"))
            frames = self._cache.get("vertex_tangent_frames")
            if frames is not None and "vertex_tangent_frames" in survived:
                basis_x, basis_y, vertex_normals = cast(
                    "tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]", frames
                )
            else:
                vertex_normals = cast("wp.array[wp.vec3] | None", self._cache.get("vertex_normals"))
        inputs = [face_normals, basis_x, basis_y, vertex_normals]
        outputs = [
            None if array is None else _launch.empty(array.size, dtype=wp.vec3, device=device)
            for array in inputs
        ]
        new_vertices = _launch.empty(self.n_vertices, dtype=wp.vec3, device=device)
        new_faces = self._faces
        face_mode = kernel_transform.FACES_KEEP
        if reverse:
            new_faces = _launch.empty(self._faces.size, dtype=wp.int32, device=device)
            face_mode = kernel_transform.FACES_REVERSE
        directed = any(array is not None for array in inputs)
        n = max(
            self.n_vertices,
            self.n_faces if reverse else 0,
            *(0 if array is None else array.size for array in inputs),
        )
        if not wp.get_device(device).is_cuda and (
            not directed or n >= kernel_transform.CPU_TWO_PASS_FROM_FACES
        ):
            self._carry_directions_by_buffer(
                matrix, reverse, inputs, new_vertices, new_faces, outputs
            )
        elif n > 0:
            _launch.launch(
                kernel_mesh.transform_mesh_and_directions,
                dim=n,
                inputs=[
                    self._vertices,
                    self._faces,
                    matrix,
                    face_mode,
                    # A normal matrix is a host inverse; it is formed only when a direction maps.
                    od.transform.normal_matrix(matrix) if directed else _ZERO_MAT33,
                    *inputs,
                    new_vertices,
                    new_faces,
                    *outputs,
                ],
                device=device,
            )
        out_face, out_x, out_y, out_normal = outputs
        if face_normals is not None:
            survived["face_normals"] = out_face
        if basis_x is not None:
            # The frame's third element *is* `vertex_normals` (that property's documented
            # invariant), so rotate it once and fill both slots with the same array.
            survived["vertex_tangent_frames"] = (out_x, out_y, out_normal)
        if vertex_normals is not None:
            survived["vertex_normals"] = out_normal
        return new_vertices, new_faces

    def _carry_directions_by_buffer(
        self,
        matrix: wp.mat44,
        reverse: bool,
        inputs: list[wp.array[wp.vec3] | None],
        new_vertices: wp.array[wp.vec3],
        new_faces: wp.array[wp.int32],
        outputs: list[wp.array[wp.vec3] | None],
    ) -> None:
        """
        `_carry_directions`' launch on the CPU, as one pass a buffer.

        The point map, the normal map of each carried direction and, for a mirror, the face
        reversal. The CPU runs a launch's threads in turn, so one thread placing a vertex, a face
        and up to four directions runs slower there than a launch for each buffer.
        """
        device = self._vertices.device
        if self.n_vertices > 0:
            _launch.map(
                kernel_transform.transform_point_mat44, self._vertices, matrix, out=new_vertices
            )
        if any(array is not None for array in inputs):
            normal_map = od.transform.normal_matrix(matrix)
            for array, out in zip(inputs, outputs, strict=True):
                if array is not None and out is not None and array.size > 0:
                    _launch.map(kernel_transform.transform_normal_mat33, array, normal_map, out=out)
        if reverse and self.n_faces > 0:
            _launch.launch(
                kernel_repair.reverse_face_winding,
                dim=self.n_faces,
                inputs=[self._faces, new_faces],
                device=device,
            )

    def _carry_box(
        self, kind: str, matrix: wp.mat44 | wp.array[wp.mat44], survived: dict[str, object]
    ) -> None:
        """
        Map the cached `centroid`, and the `bounds` where the transform allows it, on the host.

        Both are single values, so this is float arithmetic rather than a launch -- and it is what
        keeps a translation from re-reducing the vertex buffer and paying a readback for a box it
        already knows.

        The area-weighted `centroid` maps exactly under any similarity: a uniform scale multiplies
        every weight equally, so the normalized weights are unchanged and the weighted mean follows
        the points. Under an affine map the weights change per face and it does not.

        The axis-aligned `bounds` survive a *translation* only. A rotated box is not derivable from
        the old one, so `bounds` and the `enclosing_diagonal` read off it are recomputed for every
        other kind -- `_TRANSLATION_CARRY` is the only set holding the diagonal for that reason.
        """
        if kind == "singular":
            return
        # ``wp.transform_point`` is Python-scope builtin dispatch. Doing it as a NumPy
        # ``3x3 @ v + t`` is measurably cheaper and declined: a few microseconds on a cache-carry
        # path, for a spelling that hides what the line means.
        # ``odt.transform_point`` is that same builtin, re-exported over the concrete vector and
        # matrix types so a value held in a variable resolves against it.
        if (centroid := self._cache.get("centroid")) is not None and kind != "affine":
            survived["centroid"] = odt.transform_point(
                od.transform.as_mat44(matrix), cast("wp.vec3", centroid)
            )
        if kind != "translation":
            return
        if (bounds := self._cache.get("bounds")) is not None:
            lower, upper = cast("tuple[wp.vec3, wp.vec3]", bounds)
            offset = odt.transform_point(od.transform.as_mat44(matrix), wp.vec3(0.0, 0.0, 0.0))
            survived["bounds"] = (lower + offset, upper + offset)

    def invert(self) -> Trimesh:
        """
        Return a new `Trimesh` with every face's winding reversed, flipping the surface's outside.

        Positions are untouched and shared, so this changes only which side of the surface is the
        outside: normals point the other way and the enclosed
        [`volume`][ordito.mesh.Trimesh.volume] changes sign. `faces` is a new buffer.

        Because nothing moves, the cache survives this far better than any transform does -- every
        quantity of the point set itself is carried (areas, the box, the centroid, edge lengths)
        along with the whole undirected topology and the assembled
        [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix]. What goes is the orientation-dependent half:
        the directed edge tables, [`vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings], the
        per-corner tables, and everything reading a sign. The two normal buffers are **negated**
        rather than dropped, which is exact and saves rebuilding `vertex_normals` from the
        `face_angles` this drops.

        Cheaper than reversing the buffer and rebuilding a `Trimesh` around it: nothing here has to
        be rotated or reassembled, since positions do not move -- only the two normal buffers are
        touched, and by negation rather than recomputation.

        Returns
        -------
        Trimesh
            New instance sharing `vertices`, on a reversed face buffer.

        Notes
        -----
        An involution up to the cache: ``mesh.invert().invert()`` has the same buffers' contents as
        ``mesh``.

        This is not [`repair.make_normals_outward`][ordito.repair.make_normals_outward]. That one
        decides per connected component and leaves an already-outward mesh alone; this flips
        unconditionally, so it turns a solid inside out.

        See Also
        --------
        [`ordito.repair.reverse_winding`][]
            The buffer-level form, without the cache.
        [`ordito.repair.make_normals_outward`][]
        [`transform`][ordito.mesh.Trimesh.transform]
            Also reverses winding, when its matrix mirrors.
        ``trimesh.Trimesh.invert``
            The same operation under trimesh's name for it, which mutates in place (no Sphinx
            inventory entry to link).

        Examples
        --------
        What was the outside is now the inside:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        flipped = mesh.invert()
        print(f"{mesh.volume:.4f} {flipped.volume:.4f}")  # 1.0000 -1.0000
        print(flipped.vertices is mesh.vertices)  # True
        ```
        """
        survived = {key: value for key, value in self._cache.items() if key in _INVERT_CARRY}
        # Negated, not recomputed: exact, and it saves rebuilding `vertex_normals` from the
        # `face_angles` this flip drops. Both buffers in one launch; an absent one is zero-length.
        device = self._vertices.device
        none = _launch.empty(0, dtype=wp.vec3, device=device)
        face_normals = cast("wp.array[wp.vec3] | None", self._cache.get("face_normals"))
        vertex_normals = cast("wp.array[wp.vec3] | None", self._cache.get("vertex_normals"))
        if face_normals is not None or vertex_normals is not None:
            inputs = [none if a is None else a for a in (face_normals, vertex_normals)]
            outputs = [_launch.empty(a.size, dtype=wp.vec3, device=device) for a in inputs]
            n = max(a.size for a in inputs)
            if n > 0:
                _launch.launch(
                    kernel_mesh.negate_cached_normals,
                    dim=n,
                    inputs=[*inputs, *outputs],
                    device=device,
                )
            if face_normals is not None:
                survived["face_normals"] = outputs[0]
            if vertex_normals is not None:
                survived["vertex_normals"] = outputs[1]
        return self._derived(self._vertices, od.repair.reverse_winding(self._faces), survived)

    def with_vertices(self, new_vertices: wp.array[wp.vec3], *, refit: bool = False) -> Trimesh:
        """
        Return a new `Trimesh` with different vertex positions but the same topology.

        Cached quantities that depend only on `faces` (edges, adjacency, boundary,
        manifold/orientation predicates) are carried forward; quantities that depend on
        vertex positions (normals, areas, angles, the BVH, ...) are dropped and recomputed
        lazily on the new instance. `faces` is shared (aliased), not copied.

        Parameters
        ----------
        new_vertices
            ``(n_vertices,)`` replacement vertex positions; must have the same length as the
            current `vertices`.
        refit
            Hand this mesh's built [`warp_mesh`][ordito.mesh.Trimesh.warp_mesh] to the new one,
            its BVH refitted to ``new_vertices`` rather than built again: for a deformation loop
            that queries the mesh after every move. This mesh then no longer holds it (it builds
            its own if asked again), and a caller still holding that ``warp.Mesh`` sees it answer
            for ``new_vertices``. A refit keeps the tree's shape, so it fits a moved mesh less
            tightly than a fresh build and its queries slow as the motion grows; build afresh
            after a large deformation. Does nothing when `warp_mesh` was never built.

        Returns
        -------
        Trimesh
            New instance sharing `faces` with the topology-only caches carried forward.

        Raises
        ------
        ValueError
            If ``new_vertices`` has a different vertex count than the current mesh.
        RuntimeError
            If ``new_vertices`` is not on the mesh's device.

        Examples
        --------
        Scaling the positions keeps the topology caches and recomputes the geometry:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        edges = mesh.edges_unique
        doubled = od.transform.transform_points(mesh.vertices, od.transform.scale_matrix(2.0))
        bigger = mesh.with_vertices(doubled)
        print(bigger.edges_unique is edges)  # True
        print(f"{bigger.area:.4f}")  # 24.0000
        ```

        A deformation loop hands the search tree on instead of building one each step:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=3))
        tree = mesh.warp_mesh
        moved = mesh.with_vertices(
            od.transform.transform_points(mesh.vertices, od.transform.scale_matrix(1.1)),
            refit=True,
        )
        print(moved.warp_mesh is tree, mesh.warp_mesh is tree)  # True False
        ```
        """
        require_same_device(vertices=self._vertices, new_vertices=new_vertices)
        if new_vertices.size != self.n_vertices:
            raise ValueError(
                f"with_vertices requires the same vertex count ({self.n_vertices}), "
                f"got {new_vertices.size}"
            )
        survived = {key: value for key, value in self._cache.items() if key in _WITH_VERTICES_CARRY}
        if refit and "warp_mesh" in self._cache:
            tree = cast("wp.Mesh", self._cache.pop("warp_mesh"))
            if self._cache.get("traced_mesh") is tree:
                # The CPU device's `traced_mesh` is `warp_mesh` itself.
                del self._cache["traced_mesh"]
            # Assigning the points refits the BVH to them.
            tree.points = new_vertices
            survived["warp_mesh"] = tree
        return self._derived(new_vertices, self._faces, survived)

    def with_faces(self, new_faces: wp.array[wp.int32]) -> Trimesh:
        """
        Return a new `Trimesh` with different faces; no cached quantities are carried forward.

        Parameters
        ----------
        new_faces
            ``(3 * n_faces,)`` or ``(n_faces, 3)`` replacement triangle indices.

        Returns
        -------
        Trimesh
            New instance sharing `vertices` and the
            [`vertex_attributes`][ordito.mesh.Trimesh.vertex_attributes], with an empty cache and
            no face attributes: nothing relates the new faces to the old ones. Use
            [`with_edit`][ordito.mesh.Trimesh.with_edit] when an edit reports where they came from.

        Examples
        --------
        Keeping only the first two triangles of a box; the vertex buffer is kept whole:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        front = mesh.with_faces(mesh.faces[:6])
        print(front.n_vertices, front.n_faces, len(front.boundary_loops))  # 8 2 1
        ```
        """
        return Trimesh(
            self._vertices,
            new_faces,
            vertex_attributes=self._vertex_attributes,
            material=self._material,
        )

    def with_attributes(
        self,
        *,
        vertex_attributes: Mapping[str, odt.ArrayNd | None] | None = None,
        face_attributes: Mapping[str, odt.ArrayNd | None] | None = None,
    ) -> Trimesh:
        """
        Return a new `Trimesh` with attributes set or removed, sharing the whole cache.

        Each entry of ``vertex_attributes`` / ``face_attributes`` sets the attribute of its name,
        and an entry whose value is ``None`` removes it; attributes not named are kept. The
        geometry and every cached value are shared with this mesh, since no attribute changes
        either.

        Parameters
        ----------
        vertex_attributes
            Named ``(n_vertices,)`` arrays to set, or ``None`` values to remove.
        face_attributes
            Named ``(n_faces,)`` arrays to set, or ``None`` values to remove.

        Returns
        -------
        Trimesh
            New instance on the same buffers and cache, with the updated attributes.

        Raises
        ------
        TypeError
            If a value is not a ``wp.array`` of a dtype the constructor accepts.
        ValueError
            If a value does not have one entry per vertex or face.
        RuntimeError
            If a value is not on the mesh's device.

        See Also
        --------
        [`vertex_attributes`][ordito.mesh.Trimesh.vertex_attributes]
        [`face_attributes`][ordito.mesh.Trimesh.face_attributes]

        Examples
        --------
        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        ids = od.array.arange(12, device=mesh.device)
        labelled = mesh.with_attributes(face_attributes={"id": ids})
        print(list(labelled.face_attributes), labelled.face_normals is mesh.face_normals)
        # ['id'] True
        cleared = labelled.with_attributes(face_attributes={"id": None})
        print(len(cleared.face_attributes))  # 0
        ```
        """
        # Private entries (``_heat_solvers``, ``_fixed_vertex_solver``) hold solvers that refer back
        # to this very mesh, so they stay with it; every public cached value is shared.
        survived = {key: value for key, value in self._cache.items() if not key.startswith("_")}
        return Trimesh(
            self._vertices,
            self._faces,
            vertex_attributes=_updated_attributes(self._vertex_attributes, vertex_attributes),
            face_attributes=_updated_attributes(self._face_attributes, face_attributes),
            material=self._material,
            initial_cache=survived,
        )

    def with_material(self, material: od.texture.Material | None) -> Trimesh:
        """
        Return a new `Trimesh` with ``material`` as its texture, sharing the whole cache.

        Parameters
        ----------
        material
            The new material, or ``None`` to remove it.

        Returns
        -------
        Trimesh
            New instance on the same buffers, attributes and cache.

        Raises
        ------
        TypeError
            If ``material`` is not a [`Material`][ordito.texture.Material].
        RuntimeError
            If the material's image is not on the mesh's device.

        See Also
        --------
        [`material`][ordito.mesh.Trimesh.material]

        Examples
        --------
        ```python
        import warp as wp

        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        painted = mesh.with_material(od.texture.Material(base_color=wp.vec4(1.0, 0.5, 0.0, 1.0)))
        print(mesh.material, painted.material.name)  # None material
        ```
        """
        # As in `with_attributes`: the solvers refer back to this mesh and stay with it.
        survived = {key: value for key, value in self._cache.items() if not key.startswith("_")}
        return Trimesh(
            self._vertices,
            self._faces,
            vertex_attributes=self._vertex_attributes,
            face_attributes=self._face_attributes,
            material=material,
            initial_cache=survived,
        )

    def to_color(self) -> Trimesh:
        """
        Return a `Trimesh` whose vertex colours are its material read at its UVs.

        Returns
        -------
        Trimesh
            New instance on the same buffers and cache, its ``"color"`` vertex attribute set to
            [`ordito.texture.material_colors`][] of [`material`][ordito.mesh.Trimesh.material] at
            [`uv`][ordito.mesh.Trimesh.uv]; the UVs and the material are kept.

        Raises
        ------
        ValueError
            If the mesh has no material or no UVs.

        See Also
        --------
        [`ordito.texture.material_colors`][]
        ``trimesh.visual.TextureVisuals.to_color``
            The same conversion in trimesh, which samples its nearest texel (no Sphinx inventory
            entry to link).

        Examples
        --------
        A texture whose left half is red and right half blue, read along the bottom row of a grid:

        ```python
        import numpy as np
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.grid(count=(3, 3), center=False)
        texels = np.zeros((4, 4, 4), dtype=np.float32)
        texels[:, :2] = [1.0, 0.0, 0.0, 1.0]
        texels[:, 2:] = [0.0, 0.0, 1.0, 1.0]
        mesh = od.Trimesh(
            vertices,
            faces,
            vertex_attributes={"uv": wp.array(vertices.numpy()[:, :2] * 0.6 + 0.2, dtype=wp.vec2)},
            material=od.texture.Material(image=wp.array(texels, dtype=wp.float32)),
        )
        colors = mesh.to_color().vertex_colors.numpy()
        print(colors[::3, 0].tolist(), colors[::3, 2].tolist())  # [1.0, 0.5, 0.0] [0.0, 0.5, 1.0]
        ```
        """
        uv = self.uv
        if self._material is None or uv is None:
            raise ValueError("to_color needs a mesh with both a material and UVs")
        colors = od.texture.material_colors(self._material, uv)
        return self.with_attributes(vertex_attributes={"color": colors})

    def with_edit(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        provenance: od.interpolation.Provenance,
    ) -> Trimesh:
        """
        Return the output of an edit of this mesh as a `Trimesh`, its attributes carried through.

        Every free function that rebuilds a mesh's connectivity takes ``return_provenance=True``
        and returns the new ``(vertices, faces)`` with a
        [`Provenance`][ordito.interpolation.Provenance]; this wraps them and carries every
        attribute across. A float attribute (``wp.float32``, ``wp.vec2``, ``wp.vec3``,
        ``wp.vec4``) is interpolated with the weights the edit formed its new vertices with, so a
        midpoint takes the mean of its edge's values; every other one (a label, a mask, a
        ``float64`` field) and every face attribute is copied from its source. An element with no
        source (a face filling a hole) reads zero. An interpolated direction is not renormalized.

        Parameters
        ----------
        vertices
            ``(n_out_vertices,)`` the edit's output vertex positions.
        faces
            ``(3 * n_out_faces,)`` or ``(n_out_faces, 3)`` the edit's output triangle indices.
        provenance
            The edit's provenance from this mesh to that output.

        Returns
        -------
        Trimesh
            New instance on ``vertices`` and ``faces`` with an empty cache and the carried
            attributes.

        Raises
        ------
        ValueError
            If ``provenance`` does not describe an edit of a mesh of this one's vertex and face
            counts, or does not name one source per output vertex and face.
        RuntimeError
            If ``vertices``, ``faces`` and the mesh are not all on one device.

        See Also
        --------
        [`ordito.interpolation.Provenance`][]
        [`with_faces`][ordito.mesh.Trimesh.with_faces]

        Examples
        --------
        Vertex labels through a subdivision, which keeps the original vertices first:

        ```python
        import ordito as od

        vertices, faces = od.creation.box()
        labels = od.array.arange(8, device=vertices.device)
        mesh = od.Trimesh(vertices, faces, vertex_attributes={"label": labels})
        finer = mesh.with_edit(*od.remesh.subdivide(vertices, faces, return_provenance=True))
        print(finer.n_vertices, finer.vertex_attributes["label"].numpy()[:8].tolist())
        # 26 [0, 1, 2, 3, 4, 5, 6, 7]
        ```
        """
        require_same_device(vertices=vertices, faces=faces, mesh=self._vertices)
        if provenance.n_vertices != self.n_vertices or provenance.n_faces != self.n_faces:
            raise ValueError(
                f"provenance describes an edit of a mesh of {provenance.n_vertices} vertices and "
                f"{provenance.n_faces} faces, this one has {self.n_vertices} and {self.n_faces}"
            )
        n_out_faces = faces.size // 3
        if (
            provenance.vertex_index.size != vertices.size
            or provenance.face_index.size != n_out_faces
        ):
            raise ValueError(
                f"provenance names {provenance.vertex_index.size} vertices and "
                f"{provenance.face_index.size} faces, the output has {vertices.size} and "
                f"{n_out_faces}"
            )
        return Trimesh(
            vertices,
            faces,
            vertex_attributes={
                name: provenance.vertex_values(values)
                for name, values in self._vertex_attributes.items()
            },
            face_attributes={
                name: provenance.face_values(values)
                for name, values in self._face_attributes.items()
            },
            material=self._material,
        )

    def submesh(self, faces: wp.array[wp.int32] | wp.array[wp.bool]) -> Trimesh:
        """
        Extract the selected faces as a new `Trimesh` with vertices reindexed from zero.

        Parameters
        ----------
        faces
            ``(k,)`` ``wp.int32`` face indices or a ``(n_faces,)`` ``wp.bool`` mask; the dtype
            selects which.

        Returns
        -------
        Trimesh
            Compact submesh with an empty cache. No cached value is carried: the vertex numbering
            changes, so every index-valued quantity on this mesh names different vertices there.
            The attributes of the kept vertices and faces are.

        Raises
        ------
        TypeError
            If ``faces`` is neither a ``wp.int32`` nor a ``wp.bool`` array.

        See Also
        --------
        [`ordito.selection.submesh_from_face_indices`][]
        [`ordito.selection.submesh_from_face_mask`][]
        [`split`][ordito.mesh.Trimesh.split]
        [`trimesh.Trimesh.submesh`][]

        Examples
        --------
        Select faces by index or by mask; unused vertices are dropped:

        ```python
        import warp as wp

        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        piece = mesh.submesh(wp.array([0, 1], dtype=wp.int32))
        print(piece.n_vertices, piece.n_faces)  # 4 2
        ```
        """
        if faces.dtype is not wp.bool and faces.dtype is not wp.int32:
            raise TypeError(f"submesh needs a wp.int32 or wp.bool array, got {faces.dtype}")
        if not self._vertex_attributes and not self._face_attributes:
            # With nothing to carry, the selection reports no provenance and issues no more.
            if faces.dtype is wp.bool:
                parts = od.selection.submesh_from_face_mask(
                    self._vertices, self._faces, cast("wp.array[wp.bool]", faces)
                )
            else:
                parts = od.selection.submesh_from_face_indices(
                    self._vertices, self._faces, cast("wp.array[wp.int32]", faces)
                )
            return Trimesh(*parts, material=self._material)
        if faces.dtype is wp.bool:
            edit = od.selection.submesh_from_face_mask(
                self._vertices,
                self._faces,
                cast("wp.array[wp.bool]", faces),
                return_provenance=True,
            )
        else:
            edit = od.selection.submesh_from_face_indices(
                self._vertices,
                self._faces,
                cast("wp.array[wp.int32]", faces),
                return_provenance=True,
            )
        return self.with_edit(*edit)

    def split(self, *, copy: bool = False) -> list[Trimesh]:
        """
        Split into connected components by face adjacency, as one `Trimesh` per body.

        Parameters
        ----------
        copy
            Give each component independent buffers. By default the components are **views** into
            two shared allocations, so holding one keeps both alive -- the
            [`ordito.combine.split`][] convention, carried through unchanged.

        Returns
        -------
        list[Trimesh]
            One mesh per body, each with vertices reindexed from zero, an empty cache and its
            share of the attributes. The count is [`body_count`][ordito.mesh.Trimesh.body_count].

        See Also
        --------
        [`ordito.combine.split`][]
        [`body_count`][ordito.mesh.Trimesh.body_count]
            The same number without materializing the bodies.
        [`trimesh.Trimesh.split`][]

        Examples
        --------
        ```python
        import ordito as od

        first = od.Trimesh(*od.creation.box())
        second = od.Trimesh(*od.creation.box(bounds=[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]))
        parts = (first + second).split()
        print([part.n_faces for part in parts])  # [12, 12]
        ```
        """
        # The labels are this mesh's own cached property, so a mesh that has counted its bodies
        # splits without labelling them again, and one that has not keeps them for later.
        labels = self.face_connected_component_labels
        if not self._vertex_attributes and not self._face_attributes:
            return [
                Trimesh(vertices, faces, material=self._material)
                for vertices, faces in od.combine.split(
                    self._vertices, self._faces, copy=copy, labels=labels
                )
            ]
        return [
            self.with_edit(vertices, faces, provenance)
            for vertices, faces, provenance in od.combine.split(
                self._vertices, self._faces, copy=copy, labels=labels, return_provenance=True
            )
        ]

    def copy(self) -> Trimesh:
        """
        Return a `Trimesh` on independent copies of its buffers, with an empty cache.

        Returns
        -------
        Trimesh
            New instance sharing nothing with this one: `vertices`, `faces`, every attribute and
            the material's image are copied.

        Notes
        -----
        Reach for this before deliberately rewriting a buffer from a kernel: every other method
        here **aliases** its buffers into the meshes it returns, so an in-place edit would reach
        them too. Where the mesh is only being read, the copy is pure cost -- nothing on this
        class mutates.

        See Also
        --------
        [`with_vertices`][ordito.mesh.Trimesh.with_vertices]
        [`invalidate`][ordito.mesh.Trimesh.invalidate]
        [`trimesh.Trimesh.copy`][]

        Examples
        --------
        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        duplicate = mesh.copy()
        print(duplicate.vertices is mesh.vertices, duplicate.n_faces)  # False 12
        ```
        """
        return Trimesh(
            _launch.clone(self._vertices),
            _launch.clone(self._faces),
            vertex_attributes={
                name: _launch.clone(values) for name, values in self._vertex_attributes.items()
            },
            face_attributes={
                name: _launch.clone(values) for name, values in self._face_attributes.items()
            },
            material=_copied_material(self._material),
        )

    def __add__(self, other: Trimesh) -> Trimesh:
        """
        Concatenate two meshes into one, offsetting the second's face indices.

        Parameters
        ----------
        other
            Mesh to append; its vertices follow this mesh's in the result.

        Returns
        -------
        Trimesh
            Combined mesh with an empty cache. Vertices are **not** merged, so a shared surface
            stays two coincident sheets -- [`ordito.repair.remove_duplicated_vertices`][] is the
            weld. An attribute both meshes carry is concatenated; one only a single mesh carries is
            dropped, having no values on the other's elements. The material is kept when both meshes
            hold the same one.

        Warns
        -----
        UserWarning
            When the meshes hold different materials (or only one holds one): the result holds
            none, since one material cannot texture both, and keeps the UVs.

        Raises
        ------
        TypeError
            If an attribute both meshes carry has a different dtype on each.
        RuntimeError
            If the two meshes are not on one device.

        See Also
        --------
        [`ordito.combine.concatenate`][]
        [`split`][ordito.mesh.Trimesh.split]
            The inverse, up to ordering.
        ``trimesh.Trimesh.__add__``
            The same operator under trimesh's name for it (no Sphinx inventory entry to link).

        Examples
        --------
        ```python
        import ordito as od

        combined = od.Trimesh(*od.creation.box()) + od.Trimesh(*od.creation.tetrahedron())
        print(combined.n_vertices, combined.n_faces, combined.body_count)  # 12 16 2
        ```
        """
        vertices, faces = od.combine.concatenate(
            [(self._vertices, self._faces), (other.vertices, other.faces)]
        )
        material = self._material
        if other.material is not material:
            material = None
            warnings.warn(
                "the meshes hold different materials, so their sum holds none (its UVs are kept)",
                UserWarning,
                stacklevel=2,
            )
        return Trimesh(
            vertices,
            faces,
            vertex_attributes=_joined_attributes(
                self._vertex_attributes, other.vertex_attributes, "vertex"
            ),
            face_attributes=_joined_attributes(
                self._face_attributes, other.face_attributes, "face"
            ),
            material=material,
        )

    def __repr__(self) -> str:
        """
        Describe the mesh: its sizes and device, its attributes, and the values it has cached.

        Returns
        -------
        str
            A first line ``Trimesh(<n> vertices, <m> faces, <device>)``, then one indented line per
            vertex and face attribute with its dtype, then one for the material, if any, with its
            image size, then the names of the cached values.

        Examples
        --------
        ```python
        import ordito as od

        vertices, faces = od.creation.box()
        labels = od.array.arange(8, device=vertices.device)
        mesh = od.Trimesh(vertices, faces, vertex_attributes={"label": labels})
        _ = mesh.area
        lines = repr(mesh).splitlines()
        print(lines[0].split(",")[:2])  # ['Trimesh(8 vertices', ' 12 faces']
        print(lines[1].strip())  # vertex attribute 'label': int32
        print(lines[2].strip())  # cached: area, face_areas, face_normals
        ```
        """
        lines = [f"Trimesh({self.n_vertices} vertices, {self.n_faces} faces, {self.device})"]
        for element, table in (
            ("vertex", self._vertex_attributes),
            ("face", self._face_attributes),
        ):
            lines.extend(
                f"    {element} attribute {name!r}: {_dtype_name(values.dtype)}"
                for name, values in table.items()
            )
        if self._material is not None:
            image = self._material.image
            texture = "no image" if image is None else f"{image.shape[1]}x{image.shape[0]} image"
            lines.append(f"    material {self._material.name!r}: {texture}")
        cached = sorted(key for key in self._cache if not key.startswith("_"))
        if cached:
            lines.append(f"    cached: {', '.join(cached)}")
        return "\n".join(lines)

    def _derived(
        self, vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], cache: dict[str, object]
    ) -> Trimesh:
        """Build a mesh on ``vertices`` and ``faces`` with ``cache`` and these attributes."""
        return Trimesh(
            vertices,
            faces,
            vertex_attributes=self._vertex_attributes,
            face_attributes=self._face_attributes,
            material=self._material,
            initial_cache=cache,
        )

    def release_factorizations(self) -> None:
        """
        Drop every factorization the solvers keep on this mesh, keeping the mesh's cache.

        Each [`heat_solver`][ordito.mesh.Trimesh.heat_solver] and the
        [`fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver] is released (its
        factorizations, solve counts and remembered refusals dropped) and forgotten; the operators
        cached as properties stay. A later solve starts counting afresh, so it factors again only
        when that pays.

        See Also
        --------
        [`heat_solver`][ordito.mesh.Trimesh.heat_solver]
        [`fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver]
        [`invalidate`][ordito.mesh.Trimesh.invalidate]

        Examples
        --------
        The kept solver is forgotten, so the next call builds a fresh one:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.icosphere(subdivisions=2))
        solver = mesh.heat_solver()
        mesh.release_factorizations()
        print(mesh.heat_solver() is solver)  # False
        ```
        """
        solvers = cast(
            "dict[tuple[float | None, bool, str], od.heat.HeatSolver]",
            self._cache.pop("_heat_solvers", {}),
        )
        for solver in solvers.values():
            solver.release()
        fixed = self._cache.pop("_fixed_vertex_solver", None)
        if fixed is not None:
            cast("od.parametrization.FixedVertexSolver", fixed).release()

    def invalidate(self) -> None:
        """
        Clear every cached value, including the `warp_mesh` BVH.

        Call this after mutating `vertices` or `faces` in place from a kernel — `Trimesh` has
        no way to detect such an edit on its own, so cached values would otherwise silently
        go stale.

        Examples
        --------
        After an in-place edit of the vertex buffer, drop the stale cache:

        ```python
        import ordito as od

        mesh = od.Trimesh(*od.creation.box())
        print(f"{mesh.area:.4f}")  # 6.0000
        double = od.transform.scale_matrix(2.0)
        od.transform.transform_points(mesh.vertices, double, out=mesh.vertices)
        mesh.invalidate()
        print(f"{mesh.area:.4f}")  # 24.0000
        ```
        """
        self._cache.clear()


def _attribute_table(
    attributes: Mapping[str, odt.ArrayNd] | None,
    count: int,
    element: str,
    vertices: wp.array[wp.vec3],
) -> Mapping[str, odt.ArrayNd]:
    """
    Validate a `Trimesh` attribute mapping and return it as a read-only table.

    Each value must be a rank-1 ``wp.array`` of ``count`` entries on the device of ``vertices``, of
    a dtype a ``Provenance`` carries (so every method can carry it), under a string name.
    """
    if not attributes:
        return _NO_ATTRIBUTES
    # Local: the dtype list lives with the gather kernels, which a mesh with no attributes never
    # needs to import.
    from ordito.kernels import interpolation as kernel_interpolation

    table: dict[str, odt.ArrayNd] = {}
    for name, values in attributes.items():
        if not isinstance(name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"{element} attribute names must be strings, got {name!r}")
        if not isinstance(values, wp.array):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"{element} attribute {name!r} must be a wp.array, got {type(values).__name__}"
            )
        if values.dtype not in kernel_interpolation.ATTRIBUTE_DTYPES:
            raise TypeError(
                f"{element} attribute {name!r} has dtype {_dtype_name(values.dtype)}, which no "
                "provenance carries"
            )
        if values.ndim != 1 or values.size != count:
            raise ValueError(
                f"{element} attribute {name!r} must have shape ({count},), got "
                f"{tuple(values.shape)}"
            )
        reserved = _RESERVED_ATTRIBUTE_DTYPES.get(name)
        if reserved is not None and values.dtype is not reserved:
            raise TypeError(
                f"{element} attribute {name!r} must be {_dtype_name(reserved)}, got "
                f"{_dtype_name(values.dtype)}"
            )
        require_same_device(vertices=vertices, **{f"{element} attribute {name!r}": values})
        table[name] = values.contiguous()
    return MappingProxyType(table)


def _copied_material(material: od.texture.Material | None) -> od.texture.Material | None:
    """``material`` on a copy of its image (``Trimesh.copy``); ``None`` stays ``None``."""
    if material is None or material.image is None:
        return material
    return dataclasses.replace(material, image=_launch.clone(material.image))


def _updated_attributes(
    table: Mapping[str, odt.ArrayNd], updates: Mapping[str, odt.ArrayNd | None] | None
) -> dict[str, odt.ArrayNd]:
    """``table`` with ``updates`` applied: a value sets its name, ``None`` removes it."""
    merged = dict(table)
    for name, values in (updates or {}).items():
        if values is None:
            merged.pop(name, None)
        else:
            merged[name] = values
    return merged


def _joined_attributes(
    first: Mapping[str, odt.ArrayNd], second: Mapping[str, odt.ArrayNd], element: str
) -> dict[str, odt.ArrayNd]:
    """Concatenate the attributes both tables hold, in order (``Trimesh.__add__``)."""
    joined: dict[str, odt.ArrayNd] = {}
    for name, values in first.items():
        other = second.get(name)
        if other is None:
            continue
        if other.dtype != values.dtype:
            raise TypeError(
                f"{element} attribute {name!r} is {_dtype_name(values.dtype)} on one mesh and "
                f"{_dtype_name(other.dtype)} on the other"
            )
        joined[name] = od.array.concatenate([values, other])
    return joined


def _dtype_name(dtype: object) -> str:
    """Name a Warp dtype briefly (``int32``, ``vec3f``, ``bool``), for messages and ``repr``."""
    return "bool" if dtype is wp.bool else wp.types.type_repr(dtype)


def mesh_arguments(
    name: str,
    mesh: object,
    vertices: object,
    faces: object,
    arguments: tuple[object, ...],
    n_positional: int,
) -> tuple[Trimesh, bool, tuple[object, ...]]:
    """
    Resolve a call that takes a `Trimesh` or, in its place, ``vertices`` and ``faces``.

    For a function declared ``f(mesh, faces, *arguments, *, vertices)`` so that both
    ``f(trimesh, a, b, *, c)`` and ``f(vertices, faces, a, b, c)`` bind: in the `Trimesh` form the
    first ``n_positional`` arguments may be positional, and each lands one parameter early (the
    first in ``faces``); this moves them back, taking each from its own keyword where the caller
    named it instead. Every later argument of the `Trimesh` form is keyword-only, so it is bound in
    place. ``None`` stands for an argument not given, and the positional ones are never ``None``.

    Parameters
    ----------
    name
        The calling function's name, for the error messages.
    mesh
        The first parameter: a `Trimesh`, the vertex buffer of the other form, or ``None`` when
        the caller passed ``vertices`` by keyword.
    vertices
        The ``vertices`` keyword, or ``None``.
    faces
        The second parameter: the face buffer, or in the `Trimesh` form the first argument after
        the mesh.
    arguments
        The parameters after ``faces``, in order, as bound.
    n_positional
        Leading arguments of the `Trimesh` form that may be passed by position.

    Returns
    -------
    mesh : Trimesh
        The mesh: the caller's, or a new one over ``vertices`` and ``faces``.
    owned : bool
        Whether the caller passed the `Trimesh` (and so keeps what it caches).
    arguments : tuple
        ``arguments`` in the `Trimesh` form's order: the positional ones moved back into place.

    Raises
    ------
    TypeError
        If neither a `Trimesh` nor both ``vertices`` and ``faces`` are given, if a `Trimesh` comes
        with ``vertices``, or if a `Trimesh` comes with more than ``n_positional`` arguments by
        position.

    Examples
    --------
    Both calling forms resolve to a `Trimesh` and the arguments in one order:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    mesh = od.Trimesh(vertices, faces)
    resolved, owned, arguments = od.mesh.mesh_arguments("f", mesh, None, 0.5, (None,), 1)
    print(resolved is mesh, owned, arguments)  # True True (0.5,)
    resolved, owned, arguments = od.mesh.mesh_arguments("f", vertices, None, faces, (0.5,), 1)
    print(resolved.n_faces, owned, arguments)  # 12 False (0.5,)
    ```
    """
    if isinstance(mesh, Trimesh):
        if vertices is not None:
            raise TypeError(f"{name}: pass a Trimesh or vertices and faces, not both")
        if faces is None:
            return mesh, True, arguments
        moved = [faces]
        for j in range(1, n_positional):
            moved.append(arguments[j] if arguments[j - 1] is None else arguments[j - 1])
        # A positional argument past the last one the `Trimesh` form takes by position lands in the
        # slot of that last one.
        if (
            n_positional <= len(arguments)
            and arguments[n_positional - 1] is not None
            and (n_positional == 1 or arguments[n_positional - 2] is not None)
        ):
            raise TypeError(
                f"{name} takes {n_positional} positional argument(s) after a Trimesh; pass the "
                "rest by keyword"
            )
        return mesh, True, (*moved, *arguments[n_positional:])
    if mesh is not None:
        if vertices is not None:
            raise TypeError(f"{name}: vertices given by position and by keyword")
        vertices = mesh
    if vertices is None or faces is None:
        raise TypeError(f"{name}: pass a Trimesh, or vertices and faces")
    return (
        Trimesh(cast("wp.array[wp.vec3]", vertices), cast("wp.array[wp.int32]", faces)),
        False,
        arguments,
    )
