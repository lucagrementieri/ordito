"""A cached, composition-based triangle mesh container (mirrors `trimesh.Trimesh`)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Generic, NoReturn, TypeVar, cast, overload

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh

_R = TypeVar("_R")

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
        "vertex_one_rings",  # and the rotation around each vertex runs the other way
        "oriented_boundary_edges",  # directed by construction
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
# tangent directions: the reference halfedge scales, the projection into the tangent plane scales
# with it, and the normalization divides the scale back out, so a similarity leaves the frame where
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
# [`Trimesh.invert`][triwarp.mesh.Trimesh.invert]. Enumerated positively rather than by
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
        "boundary_edges",
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
        if self._key not in obj._cache:
            obj._cache[self._key] = self._func(obj)
        return cast(_R, obj._cache[self._key])

    def __set__(self, obj: Trimesh, value: object) -> NoReturn:
        raise AttributeError(f"'{self._key}' is read-only; call invalidate() after in-place edits")


class Trimesh:
    """
    A triangle mesh with lazily cached derived quantities.

    Holds a ``vertices``/``faces`` pair (the same flat-``int32`` convention used throughout
    `triwarp`) and computes derived geometry, topology and validity predicates on first access,
    caching the result. `Trimesh` is frozen: there are no setters, so a cached value can never
    silently go stale from a Python-level mutation. The ``warp.Mesh`` BVH is held by composition
    rather than inheritance and built lazily behind [`warp_mesh`][triwarp.mesh.Trimesh.warp_mesh],
    since building it eagerly would waste GPU memory on meshes that never issue a ray or proximity
    query.

    Every cached property returns the same object on each access -- an array, a tuple of arrays or a
    sparse matrix, aliased across repeated accesses and (for `warp_mesh`) with the mesh's own
    `vertices`/`faces` buffers -- so callers must not mutate a returned buffer in place. That
    matters most where they are passed *into* the free functions as precomputed arguments: a wrapper
    that rewrites such a buffer copies it first. If a buffer is deliberately mutated in place by a
    kernel, call [`invalidate`][triwarp.mesh.Trimesh.invalidate] afterward to drop every cached
    value (including the BVH); otherwise use [`with_vertices`][triwarp.mesh.Trimesh.with_vertices] /
    [`with_faces`][triwarp.mesh.Trimesh.with_faces], which return a new `Trimesh` and carry forward
    whichever cached values are still valid.

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
        Triangle indices as ``wp.int32``, either flat (length ``3 * n_faces``) or shape
        ``(n_faces, 3)`` (flattened on construction).
    initial_cache
        Optional pre-populated cache entries, keyed by property name. Used by
        [`from_warp_mesh`][triwarp.mesh.Trimesh.from_warp_mesh] to seed `warp_mesh` without
        rebuilding its BVH, and by
        [`with_vertices`][triwarp.mesh.Trimesh.with_vertices] to carry forward
        topology-only caches.

    Raises
    ------
    TypeError
        If ``vertices`` is not a rank-1 ``wp.vec3`` array, or ``faces`` is not rank-1 or
        rank-2 ``wp.int32`` after flattening.
    ValueError
        If a rank-2 ``faces`` array does not have shape ``(n_faces, 3)``, or the flattened
        face buffer's size is not a multiple of 3.

    Examples
    --------
    ```python
    mesh = tw.Trimesh.from_warp_mesh(warp_mesh)
    mesh.face_normals  # computed once, cached
    mesh.face_areas  # already cached as a by-product of face_normals
    hits = tw.ray.intersects_any(mesh.warp_mesh, origins, directions)
    ```

    See Also
    --------
    [`trimesh.Trimesh`][]
    """

    __slots__ = ("_cache", "_faces", "_vertices")

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        *,
        initial_cache: dict[str, object] | None = None,
    ) -> None:
        """Construct a `Trimesh` from a vertex buffer and a face index buffer."""
        twt.ensure_ndim(vertices, 1, dtype=wp.vec3)
        if int(faces.ndim) == 2:
            if int(faces.shape[1]) != 3:
                raise ValueError(f"2D faces must have shape (n_faces, 3), got {tuple(faces.shape)}")
            faces = faces.reshape((-1,))
        twt.ensure_ndim(faces, 1, dtype=wp.int32)
        if int(faces.size) % 3 != 0:
            raise ValueError(f"faces size must be a multiple of 3, got {int(faces.size)}")

        self._vertices = vertices.contiguous()
        self._faces = faces.contiguous()
        self._cache: dict[str, object] = dict(initial_cache) if initial_cache is not None else {}

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
        """
        return cls(mesh.points, mesh.indices, initial_cache={"warp_mesh": mesh})

    @property
    def vertices(self) -> wp.array[wp.vec3]:
        """``(n_vertices,)`` mesh vertex positions."""
        return self._vertices

    @property
    def faces(self) -> wp.array[wp.int32]:
        """Length-``3 * n_faces`` flat ``wp.int32`` triangle index buffer."""
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
        return cast(wp.Device, self._vertices.device)

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
        [`triwarp.ray`][triwarp.ray]
        [`triwarp.proximity`][triwarp.proximity]
        """
        require_nonempty_mesh(self._faces, "Trimesh.warp_mesh")
        return wp.Mesh(points=self._vertices, indices=self._faces)

    @_CachedProperty
    def face_normals(self) -> wp.array[wp.vec3]:
        """
        Length-``n_faces`` unit face normals (zero for degenerate faces).

        Computed together with [`face_areas`][triwarp.mesh.Trimesh.face_areas]; whichever is
        accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.face_normals`][]
        """
        normals, areas = tw.triangles.face_normals_and_areas(self._vertices, self._faces)
        self._cache.setdefault("face_areas", areas)
        return normals

    @_CachedProperty
    def face_areas(self) -> wp.array[wp.float32]:
        """
        Length-``n_faces`` triangle areas.

        Computed together with [`face_normals`][triwarp.mesh.Trimesh.face_normals]; whichever
        is accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.area_faces`][]
        """
        # Compute the pair rather than reading the slot `face_normals` would have filled: that
        # only happens when `face_normals` is itself cold, and it can be cached *alone* -- a
        # rigid `transform` rotates it while dropping the areas a scale would have changed.
        # `setdefault` so an already-cached (carried or rotated) normal array is kept.
        normals, areas = tw.triangles.face_normals_and_areas(self._vertices, self._faces)
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
        return tw.reduce.sum(cast("twt.Array1dFloat32", self.face_areas))

    @_CachedProperty
    def face_angles(self) -> twt.Array2dFloat32:
        """
        Shape ``(n_faces, 3)`` interior angles in radians, aligned with each face's corners.

        See Also
        --------
        [`triwarp.triangles.face_angles`][]
        [`trimesh.Trimesh.face_angles`][]
        """
        return tw.triangles.face_angles(self._vertices, self._faces)

    @_CachedProperty
    def triangles_center(self) -> wp.array[wp.vec3]:
        """
        Length-``n_faces`` barycentre of each triangle (the mean of its three corners).

        Notes
        -----
        Named for ``trimesh.Trimesh.triangles_center`` rather than for the function it calls;
        matching trimesh's property names is this facade's job, and the module-level function is
        [`face_centroids`][triwarp.triangles.face_centroids].

        See Also
        --------
        [`triwarp.triangles.face_centroids`][]
        [`centroid`][triwarp.mesh.Trimesh.centroid]
            The area-weighted centre of the whole surface, which is not the mean of these.
        """
        return tw.triangles.face_centroids(self._vertices, self._faces)

    @_CachedProperty
    def centroid(self) -> wp.vec3:
        """
        Area-weighted centroid of the mesh surface (all-``NaN`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.measures.surface_centroid`][]
        [`trimesh.Trimesh.centroid`][]
        """
        return tw.measures.surface_centroid(self._vertices, self._faces)

    @_CachedProperty
    def volume(self) -> float:
        """
        Signed volume enclosed by the surface (meaningful only for a closed, consistent mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Accumulated in ``float32``,
        unlike the volume [`triwarp.measures.moments`][] computes alongside
        [`center_mass`][triwarp.mesh.Trimesh.center_mass] -- the two agree to that precision and
        are not the same number, which is why this property does not fill that call's cache.

        See Also
        --------
        [`triwarp.measures.volume`][]
        [`is_volume`][triwarp.mesh.Trimesh.is_volume]
            The check for the precondition this quantity needs.
        [`trimesh.Trimesh.volume`][]
        """
        return tw.measures.volume(self._vertices, self._faces)

    @_CachedProperty
    def center_mass(self) -> wp.vec3:
        """
        Centre of mass of the enclosed solid at unit density (all-``NaN`` for an empty mesh).

        **Not** [`centroid`][triwarp.mesh.Trimesh.centroid], which is the area-weighted centre of
        the *surface*: the two differ on any solid whose mass is not distributed like its shell.

        Computed together with [`moment_inertia`][triwarp.mesh.Trimesh.moment_inertia]; whichever
        is accessed first also caches the other.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (three, in fact -- every return
        of [`triwarp.measures.moments`][] is a host value).

        See Also
        --------
        [`triwarp.measures.moments`][]
        [`centroid`][triwarp.mesh.Trimesh.centroid]
        [`trimesh.Trimesh.center_mass`][]
        """
        _volume, center, inertia = tw.measures.moments(self._vertices, self._faces)
        self._cache.setdefault("moment_inertia", inertia)
        return center

    @_CachedProperty
    def moment_inertia(self) -> wp.mat33d:
        """
        ``(3, 3)`` inertia tensor about the centre of mass, at unit density.

        ``float64``: the second moments scale as ``length ** 5``, so a ``float32`` accumulation
        loses their low digits on any sizeable mesh.

        Computed together with [`center_mass`][triwarp.mesh.Trimesh.center_mass]; whichever is
        accessed first also caches the other.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Being referred to the centre of
        mass rather than to the origin, it is unchanged by a translation --
        [`transform`][triwarp.mesh.Trimesh.transform] carries it through one.

        See Also
        --------
        [`triwarp.measures.moments`][]
        [`center_mass`][triwarp.mesh.Trimesh.center_mass]
        [`trimesh.Trimesh.moment_inertia`][]
        """
        _volume, center, inertia = tw.measures.moments(self._vertices, self._faces)
        self._cache.setdefault("center_mass", center)
        return inertia

    @_CachedProperty
    def bounds(self) -> tuple[wp.vec3, wp.vec3]:
        """
        Axis-aligned bounding box of `vertices`, as ``(min_bound, max_bound)``.

        Pass it to any wrapper taking a ``bounds=`` argument --
        [`marching_cubes`][triwarp.levelset.marching_cubes],
        [`signed_distance_grid`][triwarp.proximity.signed_distance_grid],
        [`query_nearest`][triwarp.neighbors.query_nearest],
        [`grid_points`][triwarp.voxels.grid_points] -- so the box is reduced once per mesh rather
        than once per call.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. ``(+inf, -inf)`` for an empty
        mesh, the [`aabb`][triwarp.bounds.aabb] convention.

        See Also
        --------
        [`triwarp.bounds.aabb`][]
        [`enclosing_diagonal`][triwarp.mesh.Trimesh.enclosing_diagonal]
        """
        return tw.bounds.aabb(self._vertices)

    @_CachedProperty
    def extents(self) -> wp.vec3:
        """
        Side lengths of the axis-aligned [`bounds`][triwarp.mesh.Trimesh.bounds] box.

        Notes
        -----
        Costs no device work of its own: it subtracts the cached box on the host. ``(-inf, ...)``
        for an empty mesh, following [`bounds`][triwarp.mesh.Trimesh.bounds]' ``(+inf, -inf)``
        convention.

        See Also
        --------
        [`bounds`][triwarp.mesh.Trimesh.bounds]
        [`enclosing_diagonal`][triwarp.mesh.Trimesh.enclosing_diagonal]
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
        [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal], which reduces the box again --
        it reads the cached `bounds` on the host. ``inf`` for an empty mesh, that function's
        convention.

        A query wrapper's own default is the diagonal of the box around the mesh **and the query
        points**, which this is not; it is the right radius only when the queries lie inside the
        mesh's own box.

        See Also
        --------
        [`triwarp.bounds.enclosing_diagonal`][]
        [`bounds`][triwarp.mesh.Trimesh.bounds]
        ``trimesh.Trimesh.scale``
            The same quantity under trimesh's name for it (no Sphinx inventory entry to link).
        """
        lower, upper = self.bounds
        # ``math.dist`` rather than ``float(wp.length(upper - lower))``: a Warp operator and a
        # Warp builtin at Python scope each route through builtin dispatch, several times dearer.
        # It computes in float64 where ``wp.length`` is float32, i.e. the correctly-rounded answer
        # for float32 corners. Section 13.1.
        return math.dist(lower, upper)

    @_CachedProperty
    def vertex_normals(self) -> wp.array[wp.vec3]:
        """
        Length-``n_vertices`` angle-weighted unit vertex normals.

        Matches `trimesh`'s default vertex-normal weighting (interior-angle weighted).

        See Also
        --------
        [`vertices.vertex_normals`][triwarp.vertices.vertex_normals] at ``weighting="angle"``
        [`trimesh.Trimesh.vertex_normals`][]
        """
        return tw.vertices.vertex_normals(
            self._vertices,
            self._faces,
            weighting="angle",
            face_normals=self.face_normals,
            face_weights=self.face_angles,
        )

    @_CachedProperty
    def vertex_defects(self) -> wp.array[wp.float32]:
        """
        Length-``n_vertices`` discrete angle defect (``2π`` minus incident corner angles).

        See Also
        --------
        [`triwarp.vertices.vertex_defects`][]
        """
        return tw.vertices.vertex_defects(self.n_vertices, self._faces, self.face_angles)

    @_CachedProperty
    def nondegenerate_faces(self) -> wp.array[wp.bool]:
        """
        Length-``n_faces`` mask; ``True`` where the triangle has non-zero area.

        Named for ``trimesh.Trimesh.nondegenerate_faces`` rather than for the function it calls:
        matching trimesh's property names is this facade's whole job, so the property keeps the
        spelling a reader arrives with while the module-level function is
        [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask].

        See Also
        --------
        [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask]
        """
        return tw.triangles.face_nondegenerate_mask(self._vertices, self._faces)

    @_CachedProperty
    def mean_edge_length(self) -> float:
        """
        Mean length over all ``3 * n_faces`` per-face edges (``0.0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.edges.mean_edge_length`][]
        """
        if self.n_faces == 0:
            return 0.0
        lengths = tw.edges.edges_length(self._vertices, self._faces)
        return tw.reduce.mean(cast("twt.Array1dFloat32", lengths))

    @_CachedProperty
    def edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_faces * 3, 2)`` directed triangle edges, in face-corner order.

        See Also
        --------
        [`triwarp.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges`][]
        """
        return tw.edges.faces_to_edges(self._faces, sorted=False)

    @_CachedProperty
    def edges_sorted(self) -> twt.Array2dInt32:
        """
        Shape ``(n_faces * 3, 2)`` edges from `edges`, each row sorted (smaller index first).

        See Also
        --------
        [`triwarp.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges_sorted`][]
        """
        return tw.edges.faces_to_edges(self._faces, sorted=True)

    @_CachedProperty
    def edges_face(self) -> wp.array[wp.int32]:
        """
        Length-``n_faces * 3`` face index for each row of `edges` / `edges_sorted`.

        See Also
        --------
        [`triwarp.edges.edges_face`][]
        [`trimesh.Trimesh.edges_face`][]
        """
        return tw.edges.edges_face(self._faces)

    @_CachedProperty
    def edges_unique(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` unique undirected edges, ``m <= n_faces * 3``.

        Also caches [`edges_unique_inverse`][triwarp.mesh.Trimesh.edges_unique_inverse] as a
        by-product.

        See Also
        --------
        [`triwarp.edges.edges_unique`][]
        [`trimesh.Trimesh.edges_unique`][]
        """
        unique, inverse = tw.edges.edges_unique(self._faces, n_vertices=self.n_vertices)
        self._cache.setdefault("edges_unique_inverse", inverse)
        return unique

    @_CachedProperty
    def edges_unique_inverse(self) -> wp.array[wp.int32]:
        """
        Length-``n_faces * 3`` inverse mapping into `edges_unique` (reconstructs `edges_sorted`).

        See Also
        --------
        [`triwarp.edges.edges_unique_inverse`][]
        [`trimesh.Trimesh.edges_unique_inverse`][]
        """
        # As in `face_areas`: `edges_unique` fills this only when it is itself cold, and a
        # mirroring `transform` carries it while dropping this one, whose row order reverses.
        unique, inverse = tw.edges.edges_unique(self._faces, n_vertices=self.n_vertices)
        self._cache.setdefault("edges_unique", unique)
        return inverse

    @_CachedProperty
    def faces_unique_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_faces, 3)`` index into [`edges_unique`][triwarp.mesh.Trimesh.edges_unique].

        Row ``f`` holds the unique-edge slot of each of face ``f``'s three edges, in face-corner
        order, so ``edges_unique[faces_unique_edges[f, k]]`` is that corner's edge.

        Notes
        -----
        Costs no device work: it is
        [`edges_unique_inverse`][triwarp.mesh.Trimesh.edges_unique_inverse] viewed as ``(n, 3)``,
        and shares that buffer rather than copying it.

        See Also
        --------
        [`edges_unique_inverse`][triwarp.mesh.Trimesh.edges_unique_inverse]
        [`trimesh.Trimesh.faces_unique_edges`][]
        """
        return twt.as_array2d(self.edges_unique_inverse.reshape((-1, 3)), wp.int32)

    @_CachedProperty
    def edges_unique_length(self) -> wp.array[wp.float32]:
        """
        Length-``m`` Euclidean length of each `edges_unique` row.

        See Also
        --------
        [`triwarp.edges.edges_unique_length`][]
        [`trimesh.Trimesh.edges_unique_length`][]
        """
        return tw.edges.edges_unique_length(
            self._vertices, self._faces, unique_edges=self.edges_unique
        )

    @_CachedProperty
    def face_adjacency(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` face index pairs that share an undirected edge.

        Also caches [`face_adjacency_edges`][triwarp.mesh.Trimesh.face_adjacency_edges] as a
        by-product.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency`][]
        [`trimesh.graph.face_adjacency`][]
        """
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            self._faces,
            return_edges=True,
            # The mesh knows its own vertex count, so the row-hash radix never has to be inferred
            # from a device reduction ending in a host readback.
            n_vertices=self._vertices.size,
        )
        self._cache.setdefault("face_adjacency_edges", adjacency_edges)
        return adjacency

    @_CachedProperty
    def face_adjacency_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` shared vertex pair for each `face_adjacency` row.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency`][]
        """
        _ = self.face_adjacency
        # Both halves live in `_TOPOLOGY_KEYS`, so nothing splits this pair today; the read stays
        # a plain one rather than recomputing, and `face_adjacency` uses `setdefault` so a future
        # stratum that does split it fails loudly here instead of silently returning a stale edge.
        return cast("twt.Array2dInt32", self._cache["face_adjacency_edges"])

    @_CachedProperty
    def face_adjacency_unshared(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` vertex on each adjacent face not on their shared edge.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_unshared`][]
        [`trimesh.graph.face_adjacency_unshared`][]
        """
        return tw.adjacency.face_adjacency_unshared(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
        )

    @_CachedProperty
    def face_adjacency_angles(self) -> wp.array[wp.float32]:
        """
        Length-``m`` unsigned angle in radians between each `face_adjacency` pair.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_angles`][]
        [`trimesh.Trimesh.face_adjacency_angles`][]
        """
        return tw.adjacency.face_adjacency_angles(
            self._vertices,
            self._faces,
            face_adjacency=self.face_adjacency,
            face_normals=self.face_normals,
        )

    @_CachedProperty
    def face_adjacency_projections(self) -> wp.array[wp.float32]:
        """
        Length-``m`` projection of each adjacent pair's unshared vertices onto the other's plane.

        Negative where the pair is convex, which is the sign
        [`face_adjacency_convex`][triwarp.mesh.Trimesh.face_adjacency_convex] thresholds.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_projections`][]
        [`face_adjacency_convex`][triwarp.mesh.Trimesh.face_adjacency_convex]
        [`trimesh.Trimesh.face_adjacency_projections`][]
        """
        return tw.adjacency.face_adjacency_projections(
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
        Length-``m`` mask; ``True`` where an adjacent face pair meets convexly.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_convex`][]
        [`face_adjacency_projections`][triwarp.mesh.Trimesh.face_adjacency_projections]
        [`trimesh.Trimesh.face_adjacency_convex`][]
        """
        return tw.adjacency.face_adjacency_convex(
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
        Length-``n_faces`` connected-component label per face (face-adjacency graph).

        Notes
        -----
        With `face_adjacency` already cached, calls
        [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
        on it; otherwise
        [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels],
        which labels straight off the sorted edge keys and builds no adjacency table. Same
        labelling engine, same answer.

        See Also
        --------
        [`triwarp.adjacency.face_connected_component_labels`][]
        """
        adjacency = self._cache.get("face_adjacency")
        if adjacency is None:
            return tw.adjacency.face_connected_component_labels(
                self._faces, n_vertices=self.n_vertices
            )
        # ``validate=False``: the rows are face ids ``face_adjacency`` derived as ``e // 3``,
        # so they are below ``n_faces`` by construction and the check would only add a sync.
        return tw.graph.connected_component_labels_from_edges(
            cast("twt.Array2dInt32", adjacency), node_count=self.n_faces, validate=False
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
        [`face_connected_component_labels`][triwarp.mesh.Trimesh.face_connected_component_labels]
        [`triwarp.combine.split`][]
            Materializes the bodies this counts.
        [`trimesh.Trimesh.body_count`][]
        """
        if self.n_faces == 0:
            return 0
        return tw.grouping.unique_1d(self.face_connected_component_labels).size

    @_CachedProperty
    def vertex_face_adjacency(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        """
        Incidence CSR of the faces touching each vertex, as ``(vertex_faces, offsets)``.

        Each row is a *set*: use `vertex_one_rings` where the rotational order around the vertex
        is what matters, at the price of needing an edge-manifold mesh.

        See Also
        --------
        [`triwarp.adjacency.vertex_face_adjacency`][]
        [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings]
        """
        return tw.adjacency.vertex_face_adjacency(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def halfedge_twins(self) -> wp.array[wp.int32]:
        """
        Length-``3 * n_faces`` opposite halfedge of every halfedge, or ``-1`` on a boundary.

        Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to ``faces[3f + (k + 1) % 3]``, so
        ``next`` and ``prev`` are index arithmetic and this array is all a walk needs to cross an
        edge.

        Raises
        ------
        ValueError
            Propagated from [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when an undirected
            edge carries three or more halfedges, i.e. the mesh is not edge-manifold.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`triwarp.halfedge.halfedge_twins`][]
        [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings]
        """
        return tw.halfedge.halfedge_twins(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def vertex_one_rings(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]:
        """
        Counter-clockwise outgoing halfedges per vertex: ``(ring_halfedges, offsets, is_boundary)``.

        Built from the cached `halfedge_twins`, so accessing either first pays for that array once.

        Raises
        ------
        ValueError
            Propagated from [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] when a vertex's
            rotation closes before its whole fan is covered (a pinched, vertex-non-manifold
            vertex), or from `halfedge_twins` on an edge-non-manifold mesh.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`triwarp.halfedge.vertex_one_rings`][]
        [`halfedge_twins`][triwarp.mesh.Trimesh.halfedge_twins]
        [`vertex_face_adjacency`][triwarp.mesh.Trimesh.vertex_face_adjacency]
        """
        return tw.halfedge.vertex_one_rings(
            self._faces, twins=self.halfedge_twins, n_vertices=self.n_vertices
        )

    @_CachedProperty
    def boundary_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_boundary, 2)`` undirected boundary edges (each row sorted, min-first).

        See Also
        --------
        [`triwarp.boundary.boundary_edges`][]
        """
        return tw.boundary.boundary_edges(self._vertices, self._faces)

    @_CachedProperty
    def oriented_boundary_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_boundary, 2)`` directed boundary edges, preserving face winding.

        See Also
        --------
        [`triwarp.boundary.oriented_boundary_edges`][]
        """
        return tw.boundary.oriented_boundary_edges(self._vertices, self._faces)

    @_CachedProperty
    def boundary_loops(self) -> list[wp.array[wp.int32]]:
        """
        One ordered vertex-index array per boundary loop (empty list if the mesh is closed).

        See Also
        --------
        [`triwarp.boundary.boundary_loops`][]
        """
        return tw.boundary.boundary_loops(self._vertices, self._faces)

    @_CachedProperty
    def boundary_vertex_indices(self) -> wp.array[wp.int32]:
        """
        Sorted unique vertex indices lying on the mesh boundary.

        See Also
        --------
        [`triwarp.boundary.boundary_vertex_indices`][]
        """
        return tw.boundary.boundary_vertex_indices(self._vertices, self._faces)

    @_CachedProperty
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` (``0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Equivalent to
        [`euler_characteristic`][triwarp.measures.euler_characteristic], recomposed here to
        reuse the cached `edges_unique` count instead of recomputing it.

        See Also
        --------
        [`triwarp.measures.euler_characteristic`][]
        [`trimesh.Trimesh.euler_number`][]
        """
        if self.n_faces == 0:
            return 0
        n_referenced = tw.grouping.unique_1d(self._faces).size
        return n_referenced - int(self.edges_unique.shape[0]) + self.n_faces

    @_CachedProperty
    def is_edge_manifold(self) -> bool:
        """
        Whether every undirected edge is shared by one or two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.validation.is_edge_manifold`][]
        """
        return tw.validation.is_edge_manifold(self._faces, n_vertices=self.n_vertices)

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
        [`triwarp.validation.is_vertex_manifold`][]
        """
        if "face_adjacency" not in self._cache:
            return tw.validation.is_vertex_manifold(self._faces)
        return tw.validation.is_vertex_manifold(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
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
        [`triwarp.validation.is_winding_consistent`][]
        [`trimesh.Trimesh.is_winding_consistent`][]
        """
        return tw.validation.is_winding_consistent(self._faces)

    @_CachedProperty
    def is_orientable(self) -> bool:
        """
        Whether the faces admit a consistent orientation (allowing per-face flips).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Unlike the other
        adjacency-based predicates on this class, this does not reuse `face_adjacency`: the
        underlying propagation engine rebuilds it internally and does not accept a
        precomputed adjacency.

        See Also
        --------
        [`triwarp.validation.is_orientable`][]
        """
        return tw.validation.is_orientable(self._faces)

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
        same reuse [`is_self_intersecting`][triwarp.mesh.Trimesh.is_self_intersecting] gets by
        taking a `wp.Mesh` directly.

        See Also
        --------
        [`triwarp.validation.is_watertight`][]
        [`is_volume`][triwarp.mesh.Trimesh.is_volume]
        """
        if self.n_faces == 0:
            return True
        # A cached BVH is reused; otherwise it is built -- and cached for later readers -- only
        # once the manifold tests pass, since an open mesh never reaches the broad phase.
        mesh = self._cache.get("warp_mesh")
        return tw.validation.is_watertight(
            self._vertices,
            self._faces,
            mesh=cast("wp.Mesh", mesh) if mesh is not None else lambda: self.warp_mesh,
        )

    @_CachedProperty
    def is_self_intersecting(self) -> bool:
        """
        Whether any two non-adjacent triangles of the mesh intersect.

        Raises
        ------
        ValueError
            If the mesh has zero faces (see [`warp_mesh`][triwarp.mesh.Trimesh.warp_mesh]).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Reuses the cached
        `warp_mesh` BVH for its broad phase.

        See Also
        --------
        [`triwarp.validation.is_self_intersecting`][]
        """
        return tw.validation.is_self_intersecting(self.warp_mesh)

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
        [`triwarp.validation.is_volume`][]
        [`trimesh.Trimesh.is_volume`][]
        """
        return tw.validation.is_volume(self._vertices, self._faces)

    # Discrete operators: the assemblies every solve over this mesh shares. Heavier than everything
    # above (a sparse build each, and the two heat bundles several) and reached by fewer callers,
    # so they sit last -- but they are also where the cache pays most, since each is what a whole
    # family of wrappers accepts as its precomputed argument.

    @_CachedProperty
    def cotmatrix_entries(self) -> twt.Array2dFloat32:
        """
        Shape ``(n_faces, 3)`` per-triangle half-cotangent weights, in igl's edge order.

        The ``float32`` table, and that costs a ``float64`` consumer nothing: the free function
        computes these weights in ``float32`` -- the vertex precision -- whatever dtype is asked
        for, and casts on write, so a ``float64`` request would return this same table widened.
        Every consumer accepts either precision and casts to the *matrix* dtype in one build, so
        pass this to [`cotmatrix`][triwarp.laplacian.cotmatrix],
        [`connection_laplacian`][triwarp.laplacian.connection_laplacian],
        [`heat_operators`][triwarp.heat.heat_operators] or
        [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix] at either
        precision -- which is what this class's own ``float32`` `cotmatrix` and ``float64``
        `heat_operators` both do.

        See Also
        --------
        [`triwarp.laplacian.cotmatrix_entries`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.cotmatrix_entries(self._vertices, self._faces)

    @_CachedProperty
    def cotmatrix(self) -> wps.BsrMatrix[wp.float32]:
        """
        Cotangent stiffness matrix: the ``float32`` discrete Laplace-Beltrami operator.

        Assembled from the cached `cotmatrix_entries`. Diagonal entries are negative and each row
        sums to zero, so ``-L`` is positive semi-definite on a closed mesh.

        Notes
        -----
        ``float32``, which is the right precision for a *product* (an energy, a residual, a filter
        weight) and not for a solve -- the heat solvers and
        [`triwarp.parametrization`][triwarp.parametrization] assemble their own ``float64``
        operators, and `heat_operators` caches the one they share.

        See Also
        --------
        [`triwarp.laplacian.cotmatrix`][]
        [`cotmatrix_entries`][triwarp.mesh.Trimesh.cotmatrix_entries]
        [`mass_matrix_entries`][triwarp.mesh.Trimesh.mass_matrix_entries]
        [`triwarp.energies.k_harmonic`][]
            Takes this operator and `mass_matrix_entries` as its two arguments.
        """
        return tw.laplacian.cotmatrix(
            self._vertices, self._faces, cot_entries=self.cotmatrix_entries
        )

    @_CachedProperty
    def mass_matrix_entries(self) -> wp.array[wp.float32]:
        """
        Length-``n_vertices`` barycentric lumped mass: a third of each incident triangle's area.

        Built from the cached `face_areas`, so this is a scatter and nothing else.

        Notes
        -----
        ``float32``. A ``float64`` solve wants
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries] at that dtype, which
        accumulates in the requested precision rather than casting this -- the two are not the same
        array widened.

        See Also
        --------
        [`triwarp.laplacian.mass_matrix_entries`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.mass_matrix_entries(
            self._vertices, self._faces, face_areas=self.face_areas
        )

    @_CachedProperty
    def laplacian_operator(self) -> wps.BsrMatrix[wp.float32]:
        """
        Row-normalized 1-ring averaging operator (the uniform / umbrella Laplacian).

        Exactly what every position filter in [`triwarp.smoothing`][triwarp.smoothing] builds for
        itself when its ``laplacian_operator=`` argument is ``None``, so passing this hoists the
        assembly out of a multi-filter or multi-call pass.

        Notes
        -----
        The **directed** ``mesh.edges`` adjacency, the shared default.
        [`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average] is the one
        filter that wants the *symmetric* adjacency instead
        ([`laplacian`][triwarp.laplacian.laplacian] at ``symmetric=True``) and must not be handed
        this one; the two differ only on a mesh with an open boundary.

        See Also
        --------
        [`triwarp.laplacian.laplacian`][]
        [`triwarp.smoothing.filter_laplacian`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.laplacian(self._vertices, self._faces)

    @_CachedProperty
    def vertex_tangent_frames(
        self,
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]:
        """
        Orthonormal tangent frame at every vertex as ``(basis_x, basis_y, normal)``.

        The gauge every 2-D tangent quantity on this mesh is measured in. Built from the cached
        `vertex_normals` and `vertex_one_rings`, and its third element **is** `vertex_normals`.

        Raises
        ------
        ValueError
            Propagated from `vertex_one_rings` on a non-manifold mesh.

        See Also
        --------
        [`triwarp.tangent_space.vertex_tangent_frames`][]
        [`vector_heat_operators`][triwarp.mesh.Trimesh.vector_heat_operators]
        """
        return tw.tangent_space.vertex_tangent_frames(
            self._vertices, self._faces, normals=self.vertex_normals, rings=self.vertex_one_rings
        )

    @_CachedProperty
    def heat_operators(self) -> tw.heat.HeatOperators:
        """
        Source-independent operator bundle for the heat method, at the default diffusion time.

        Pass it to [`heat_geodesic`][triwarp.heat.heat_geodesic] or
        [`geodesic_path`][triwarp.geodesic_walk.geodesic_path] through their ``operators=``
        argument: everything in the bundle depends on the mesh alone, so distance from many
        different source sets costs one assembly.

        Notes
        -----
        At ``t = None`` (the squared mean unique-edge length, ``igl::heat_geodesics``' default) and
        ``use_robust=False``. A mesh with degenerate triangles, or a caller sweeping ``t``, wants
        [`heat_operators`][triwarp.heat.heat_operators] directly -- and can still pass this
        class's `cotmatrix_entries` into it.

        See Also
        --------
        [`triwarp.heat.heat_operators`][]
        [`vector_heat_operators`][triwarp.mesh.Trimesh.vector_heat_operators]
        """
        return tw.heat.heat_operators(
            self._vertices, self._faces, cot_entries=self.cotmatrix_entries
        )

    @_CachedProperty
    def vector_heat_operators(self) -> tw.heat.VectorHeatOperators:
        """
        Operator bundle for the vector heat method, at the default diffusion time.

        ``(vector_system, scalar, frames, preconditioner)``: the ``2 x 2``-block connection system,
        the scalar `heat_operators`, the `vertex_tangent_frames`, and the connection system's own
        Jacobi preconditioner. Pass it to
        [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors],
        [`log_map`][triwarp.heat.log_map],
        [`extend_scalar`][triwarp.heat.extend_scalar] or
        [`heat_signed_distance`][triwarp.heat.heat_signed_distance] through their
        ``operators=`` argument.

        Its second and third fields are this class's own `heat_operators` and
        `vertex_tangent_frames`, so the three properties share one assembly however they are
        reached.

        Notes
        -----
        At ``t = None``, the same default `heat_operators` uses -- which is load-bearing rather
        than incidental: [`log_map`][triwarp.heat.log_map]'s radius is asserted to *be* the
        [`heat_geodesic`][triwarp.heat.heat_geodesic] distance, so the two systems must
        share a diffusion time.

        See Also
        --------
        [`triwarp.heat.vector_heat_operators`][]
        [`heat_operators`][triwarp.mesh.Trimesh.heat_operators]
        [`vertex_tangent_frames`][triwarp.mesh.Trimesh.vertex_tangent_frames]
        """
        return tw.heat.vector_heat_operators(
            self._vertices,
            self._faces,
            scalar_operators=self.heat_operators,
            frames=self.vertex_tangent_frames,
        )

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
            Length-``n`` mask; ``True`` where the point is inside.

        Raises
        ------
        ValueError
            If the mesh has zero faces (see [`warp_mesh`][triwarp.mesh.Trimesh.warp_mesh]).

        Notes
        -----
        Only meaningful for a closed, consistently wound surface --
        [`is_volume`][triwarp.mesh.Trimesh.is_volume] is the check for it. Reuses the cached
        `warp_mesh` rather than building a BVH per call, which is the whole reason to reach for
        this instead of the free function.

        See Also
        --------
        [`triwarp.ray.contains_points`][]
        [`triwarp.proximity.signed_distance_on_mesh`][]
            A signed distance rather than a bit, from the same BVH.
        [`trimesh.Trimesh.contains`][]
        """
        return tw.ray.contains_points(self.warp_mesh, points)

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
            Length-``count`` index of the face each sample landed on.

        Notes
        -----
        Passes the cached [`face_areas`][triwarp.mesh.Trimesh.face_areas] as the sampling weight,
        so repeated draws from one mesh share that reduction. Returns the face indices as well as
        the points, where ``trimesh.Trimesh.sample`` returns them only on request.

        See Also
        --------
        [`triwarp.sample.sample_surface`][]
        [`triwarp.sample.sample_surface_blue_noise`][]
            Sampling with a minimum separation rather than independently.
        [`trimesh.Trimesh.sample`][]
        """
        return tw.sample.sample_surface(
            self._vertices, self._faces, count, face_weight=self.face_areas, seed=seed
        )

    def transform(
        self, matrix: wp.mat44 | wp.array[wp.mat44], *, assume: str | None = None
    ) -> Trimesh:
        """
        Return a new `Trimesh` under an affine transform, carrying forward whatever survives it.

        `Trimesh` is frozen, so this returns a new instance rather than moving this one -- and that
        is the *fast* path, not merely the safe one. How much of the cache survives is decided by
        what the transform preserves ([`classify_transform`][triwarp.transform.classify_transform]),
        and for a rigid motion that is everything expensive: angles, areas, the cotangent table and
        the assembled [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix] are all isometry invariants, so
        only the directions and the bounding box are touched.

        | transform | recomputed on the new mesh |
        |---|---|
        | translation | the BVH, and nothing else |
        | rigid, reflection | the BVH and `bounds`; directions are *rotated*, not rebuilt |
        | similarity | the above, plus the length and area quantities |
        | affine | everything but connectivity |

        Face winding is reversed when ``matrix`` mirrors, which keeps normals outward and costs the
        orientation-dependent caches on top of the row above --
        [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings], the directed edge tables and
        the per-corner tables among them.

        Carrying the cache forward is cheaper than transforming the buffers and rebuilding a
        `Trimesh` around them whenever the mesh is reused for further queries: the saving is a count
        of assembly launches skipped, so it is worth reaching for on a mesh carried through a
        sequence of poses, and worth little on a mesh transformed once and used once.

        Parameters
        ----------
        matrix
            ``4x4`` transform, as a scalar ``wp.mat44`` or a ``(1,)`` ``wp.array[wp.mat44]`` --
            an [`icp`][triwarp.registration.icp] result can be passed straight through. Build one
            with [`translation_matrix`][triwarp.transform.translation_matrix],
            [`rotation_matrix`][triwarp.transform.rotation_matrix],
            [`scale_matrix`][triwarp.transform.scale_matrix] or
            [`reflection_matrix`][triwarp.transform.reflection_matrix].
        assume
            A [`TransformKind`][triwarp.transform.TransformKind] value promising what ``matrix``
            preserves, skipping the classification. Use it when a matrix composed from many
            ``float32`` factors has drifted far enough off orthogonality to be classified
            `TransformKind.AFFINE`, which is safe but discards the cache. **An incorrect promise
            silently corrupts every carried value** -- there is no check.

        Returns
        -------
        Trimesh
            New instance on the transformed vertices, sharing whatever cached values survive.
            ``self`` when ``matrix`` is the identity.

        Raises
        ------
        ValueError
            If ``assume`` is not a `TransformKind` value.

        Notes
        -----
        Carried values are **aliased**, not copied, exactly as with
        [`with_vertices`][triwarp.mesh.Trimesh.with_vertices]: the returned mesh and this one hold
        the same array objects. That is safe because neither can mutate them, and it is why this
        class offers no in-place transform -- an edit to a shared buffer would corrupt every mesh
        derived from it, with nothing to invalidate them. Use
        [`transform_points`][triwarp.transform.transform_points] with ``out=`` for in-place work on
        raw buffers.

        Examples
        --------
        ```python
        mesh = tw.Trimesh(v, f)
        spun = mesh.transform(tw.transform.rotation_matrix((0.0, 0.0, 1.0), 0.5, mesh.centroid))
        spun.cotmatrix  # carried from `mesh`, not reassembled
        ```

        See Also
        --------
        [`triwarp.transform.transform_mesh`][]
            The buffer-level form, without the cache.
        [`with_vertices`][triwarp.mesh.Trimesh.with_vertices]
        [`invalidate`][triwarp.mesh.Trimesh.invalidate]
        """
        kind = str(
            tw.transform.TransformKind(assume)
            if assume is not None
            else tw.transform.classify_transform(matrix)
        )
        if kind == tw.transform.TransformKind.IDENTITY:
            return self

        # Every use below -- `transform_mesh`'s own `reverses_orientation` check, the explicit one
        # on the next line, up to four `transform_normals` calls in `_carry_directions`, up to two
        # `as_mat44` calls in `_carry_box` -- is a host branch over a matrix that does not change
        # between them. Resolving it once here (paying the one readback `classify_transform`/
        # `reverses_orientation` already forces when `matrix` is a device array) and passing the
        # resolved host value down turns what was up to 7-8 independent host syncs of the same 16
        # floats into exactly one.
        matrix_host = tw.transform.as_mat44(matrix)
        new_vertices, new_faces = tw.transform.transform_mesh(
            self._vertices, self._faces, matrix_host
        )
        carry = _TRANSFORM_CARRY[kind]
        if tw.transform.reverses_orientation(matrix_host):
            carry = carry - _ORIENTATION_DEPENDENT_KEYS
        survived = {key: value for key, value in self._cache.items() if key in carry}
        self._carry_directions(kind, matrix_host, survived)
        self._carry_box(kind, matrix_host, survived)
        return Trimesh(new_vertices, new_faces, initial_cache=survived)

    def _carry_directions(
        self, kind: str, matrix: wp.mat44 | wp.array[wp.mat44], survived: dict[str, object]
    ) -> None:
        """
        Rotate the cached direction quantities into ``survived`` instead of dropping them.

        Only for a similarity or stronger, where a direction's image is determined by the
        transform alone. A translation moves no direction at all and carries these verbatim
        through `_TRANSLATION_CARRY`; an affine map tilts them by an amount that depends on the
        surface, so there they are recomputed -- and recomputing `face_normals` yields
        `face_areas` with it, which an affine map does not preserve anyway.

        [`transform_normals`][triwarp.transform.transform_normals] is the right map for the frame's
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
        if kind not in ("rigid", "reflection", "similarity"):
            return
        rotate = tw.transform.transform_normals
        if "face_normals" in self._cache:
            survived["face_normals"] = rotate(
                cast("wp.array[wp.vec3]", self._cache["face_normals"]), matrix
            )
        frames = self._cache.get("vertex_tangent_frames")
        if frames is not None and "vertex_tangent_frames" in survived:
            basis_x, basis_y, normal = cast(
                "tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]", frames
            )
            rotated_normal = rotate(normal, matrix)
            # The frame's third element *is* `vertex_normals` (that property's documented
            # invariant), so rotate it once and fill both slots with the same array.
            survived["vertex_tangent_frames"] = (
                rotate(basis_x, matrix),
                rotate(basis_y, matrix),
                rotated_normal,
            )
            survived["vertex_normals"] = rotated_normal
        elif "vertex_normals" in self._cache:
            survived["vertex_normals"] = rotate(
                cast("wp.array[wp.vec3]", self._cache["vertex_normals"]), matrix
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
        # ``twt.transform_point`` is that same builtin, re-exported over the concrete vector and
        # matrix types so a value held in a variable resolves against it.
        if (centroid := self._cache.get("centroid")) is not None and kind != "affine":
            survived["centroid"] = twt.transform_point(
                tw.transform.as_mat44(matrix), cast("wp.vec3", centroid)
            )
        if kind != "translation":
            return
        if (bounds := self._cache.get("bounds")) is not None:
            lower, upper = cast("tuple[wp.vec3, wp.vec3]", bounds)
            offset = twt.transform_point(tw.transform.as_mat44(matrix), wp.vec3(0.0, 0.0, 0.0))
            survived["bounds"] = (lower + offset, upper + offset)

    def invert(self) -> Trimesh:
        """
        Return a new `Trimesh` with every face's winding reversed, flipping the surface's outside.

        Positions are untouched and shared, so this changes only which side of the surface is the
        outside: normals point the other way and the enclosed
        [`volume`][triwarp.mesh.Trimesh.volume] changes sign. `faces` is a new buffer.

        Because nothing moves, the cache survives this far better than any transform does -- every
        quantity of the point set itself is carried (areas, the box, the centroid, edge lengths)
        along with the whole undirected topology and the assembled
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]. What goes is the orientation-dependent half:
        the directed edge tables, [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings], the
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

        This is not [`repair.make_normals_outward`][triwarp.repair.make_normals_outward]. That one
        decides per connected component and leaves an already-outward mesh alone; this flips
        unconditionally, so it turns a solid inside out.

        Examples
        --------
        ```python
        mesh = tw.Trimesh(v, f)
        flipped = mesh.invert()
        assert flipped.volume < 0.0 < mesh.volume  # what was the outside is now the inside
        ```

        See Also
        --------
        [`triwarp.repair.reverse_winding`][]
            The buffer-level form, without the cache.
        [`triwarp.repair.make_normals_outward`][]
        [`transform`][triwarp.mesh.Trimesh.transform]
            Also reverses winding, when its matrix mirrors.
        ``trimesh.Trimesh.invert``
            The same operation under trimesh's name for it, which mutates in place (no Sphinx
            inventory entry to link).
        """
        survived = {key: value for key, value in self._cache.items() if key in _INVERT_CARRY}
        for key in ("face_normals", "vertex_normals"):
            # Negated, not recomputed: exact, and it saves rebuilding `vertex_normals` from the
            # `face_angles` this flip drops.
            cached = self._cache.get(key)
            if cached is not None:
                normals = cast("wp.array[wp.vec3]", cached)
                flipped = wp.empty(normals.size, dtype=wp.vec3, device=normals.device)
                if normals.size > 0:
                    wp.map(wp.neg, normals, out=flipped)
                survived[key] = flipped
        return Trimesh(
            self._vertices, tw.repair.reverse_winding(self._faces), initial_cache=survived
        )

    def with_vertices(self, new_vertices: wp.array[wp.vec3]) -> Trimesh:
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

        Returns
        -------
        Trimesh
            New instance sharing `faces` with the topology-only caches carried forward.

        Raises
        ------
        ValueError
            If ``new_vertices`` has a different vertex count than the current mesh.
        """
        if new_vertices.size != self.n_vertices:
            raise ValueError(
                f"with_vertices requires the same vertex count ({self.n_vertices}), "
                f"got {new_vertices.size}"
            )
        survived = {key: value for key, value in self._cache.items() if key in _TOPOLOGY_KEYS}
        return Trimesh(new_vertices, self._faces, initial_cache=survived)

    def with_faces(self, new_faces: wp.array[wp.int32]) -> Trimesh:
        """
        Return a new `Trimesh` with different faces; no cached quantities are carried forward.

        Parameters
        ----------
        new_faces
            Replacement triangle indices, flat or ``(n_faces, 3)``.

        Returns
        -------
        Trimesh
            New instance sharing `vertices` with an empty cache.
        """
        return Trimesh(self._vertices, new_faces)

    def submesh(self, faces: wp.array[wp.int32] | wp.array[wp.bool]) -> Trimesh:
        """
        Extract the selected faces as a new `Trimesh` with vertices reindexed from zero.

        Parameters
        ----------
        faces
            Either a ``wp.int32`` array of face indices or a length-``n_faces`` ``wp.bool`` mask;
            the dtype selects which.

        Returns
        -------
        Trimesh
            Compact submesh with an empty cache. No cached value is carried: the vertex numbering
            changes, so every index-valued quantity on this mesh names different vertices there.

        Raises
        ------
        TypeError
            If ``faces`` is neither a ``wp.int32`` nor a ``wp.bool`` array.

        See Also
        --------
        [`triwarp.selection.submesh_from_face_indices`][]
        [`triwarp.selection.submesh_from_face_mask`][]
        [`split`][triwarp.mesh.Trimesh.split]
        [`trimesh.Trimesh.submesh`][]
        """
        if faces.dtype is wp.bool:
            parts = tw.selection.submesh_from_face_mask(self._vertices, self._faces, faces)
        elif faces.dtype is wp.int32:
            parts = tw.selection.submesh_from_face_indices(self._vertices, self._faces, faces)
        else:
            raise TypeError(f"submesh needs a wp.int32 or wp.bool array, got {faces.dtype}")
        return Trimesh(*parts)

    def split(self, *, copy: bool = False) -> list[Trimesh]:
        """
        Split into connected components by face adjacency, as one `Trimesh` per body.

        Parameters
        ----------
        copy
            Give each component independent buffers. By default the components are **views** into
            two shared allocations, so holding one keeps both alive -- the
            [`triwarp.combine.split`][] convention, carried through unchanged.

        Returns
        -------
        list[Trimesh]
            One mesh per body, each with vertices reindexed from zero and an empty cache. The
            count is [`body_count`][triwarp.mesh.Trimesh.body_count].

        See Also
        --------
        [`triwarp.combine.split`][]
        [`body_count`][triwarp.mesh.Trimesh.body_count]
            The same number without materializing the bodies.
        [`trimesh.Trimesh.split`][]
        """
        return [
            Trimesh(vertices, faces)
            for vertices, faces in tw.combine.split(self._vertices, self._faces, copy=copy)
        ]

    def copy(self) -> Trimesh:
        """
        Return a `Trimesh` on independent copies of `vertices` and `faces`, with an empty cache.

        Returns
        -------
        Trimesh
            New instance sharing nothing with this one.

        Notes
        -----
        Reach for this before deliberately rewriting a buffer from a kernel: every other method
        here **aliases** its buffers into the meshes it returns, so an in-place edit would reach
        them too. Where the mesh is only being read, the copy is pure cost -- nothing on this
        class mutates.

        See Also
        --------
        [`with_vertices`][triwarp.mesh.Trimesh.with_vertices]
        [`invalidate`][triwarp.mesh.Trimesh.invalidate]
        [`trimesh.Trimesh.copy`][]
        """
        return Trimesh(wp.clone(self._vertices), wp.clone(self._faces))

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
            stays two coincident sheets -- [`triwarp.repair.remove_duplicated_vertices`][] is the
            weld.

        See Also
        --------
        [`triwarp.combine.concatenate`][]
        [`split`][triwarp.mesh.Trimesh.split]
            The inverse, up to ordering.
        ``trimesh.Trimesh.__add__``
            The same operator under trimesh's name for it (no Sphinx inventory entry to link).
        """
        return Trimesh(
            *tw.combine.concatenate([(self._vertices, self._faces), (other.vertices, other.faces)])
        )

    def invalidate(self) -> None:
        """
        Clear every cached value, including the `warp_mesh` BVH.

        Call this after mutating `vertices` or `faces` in place from a kernel — `Trimesh` has
        no way to detect such an edit on its own, so cached values would otherwise silently
        go stale.
        """
        self._cache.clear()
