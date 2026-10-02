"""
Changing a mesh's triangulation: subdividing it, coarsening it, and improving its triangle shapes.

Four families, in decreasing order of how much they rearrange:

- **Remeshing.** [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] runs the Botsch-Kobbelt
  split / collapse / flip / smooth / reproject loop until every edge is near a target length. It is
  the only entry point here that does all four of the others' jobs at once.
- **Decimation.** [`quadric_decimate`][triwarp.remesh.quadric_decimate] collapses edges in
  quadric-error order to a target face count; [`cluster_decimate`][triwarp.remesh.cluster_decimate]
  instead welds each voxel of a uniform grid to a single vertex, which is far cheaper and far
  blunter.
- **Edge flipping**, which moves no vertex and changes no vertex count:
  [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] toward the Delaunay criterion,
  [`flip_by_objective`][triwarp.remesh.flip_by_objective] toward a triangle-shape or flatness
  objective, and [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay] toward the *intrinsic*
  Delaunay triangulation, which flips the connectivity a Laplacian sees without touching the
  embedding.
- **Subdivision**, which only ever adds: [`subdivide`][triwarp.remesh.subdivide] (one-to-four
  splits), [`subdivide_loop`][triwarp.remesh.subdivide_loop] (Loop's approximating scheme),
  [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] and
  [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size] (until every edge, or every
  edge of a face region, is under a target length).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple, cast, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, read_values, require_nonempty_mesh, require_same_device
from triwarp.constants import INT32_MAX, TOLERANCE_MOLLIFY, UINT64_MAX
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels import remesh as kernel_remesh
from triwarp.kernels import scatter as kernel_scatter
from triwarp.laplacian import mollify_intrinsic

# Callback that launches a predicate kernel filling ``out_flip``/``out_quad`` for one flip
# iteration. Supplied by each consumer of ``_flip_interior_edges`` (3D Delone / 2D incircle).
_LaunchCandidates = Callable[
    [
        twt.Array2dInt32,  # adjacency (m, 2)
        twt.Array2dInt32,  # adjacency_edges (m, 2)
        twt.Array2dInt32,  # unshared (m, 2)
        "wp.array[wp.uint64]",  # sorted edge keys of the last build
        "wp.array[wp.uint64]",  # edge-key set (grouping.hash_find_or_insert table)
        "wp.int32",  # edge-key set mask, -1 while the sorted keys are current
        "wp.uint64",  # key base (n_vertices)
        "wp.array[wp.bool]",  # out_flip (m,)
        twt.Array2dInt32,  # out_quad (m, 4)
    ],
    None,
]


class _EdgeIncidence(NamedTuple):
    """
    The unique undirected edges of one triangulation, and which faces meet along each of them.

    Everything the collapse passes and ``_classify`` need about edge topology, grouped once:
    ``unique_edges`` comes straight from [`edges_unique`][triwarp.edges.edges_unique], and one
    scatter over that call's corner -> unique-edge map fills both ``face_count`` (1 on a boundary
    edge, 2 on an interior one) and ``faces``. The map itself is not carried: it is scratch for
    that scatter, and no consumer of this tuple has ever read it.
    """

    unique_edges: twt.Array2dInt32
    """``(m, 2)`` unique undirected vertex pairs, each row min-first."""
    face_count: wp.array[wp.int32]
    """Length ``m`` face-corners per unique edge."""
    faces: twt.Array2dInt32
    """``(m, 2)`` incident face indices, the second column unwritten where ``face_count`` is 1."""


# Backstop on the independent-set rounds per geometry rebuild in ``quadric_decimate``.
#
# One round commits only a fraction of the scored candidates -- each winner locks the closed 1-rings
# of both endpoints, so a hashed-key round takes on the order of ``m / 50`` of them -- and the
# rebuild that follows is much more expensive than another round. So the pass loop runs rounds
# against the same scoring **until one finds nothing new**, which is the real stopping rule; this
# constant only bounds it.
_QUADRIC_ROUNDS = 8

# ``quadric_decimate`` records its pass once and replays it, and the recording fixes every face-,
# corner- and edge-indexed width, the two sorts' included. So once the live face count has fallen
# to ``_DECIMATION_RECAPTURE`` of the recorded width, *and* that frees at least
# ``_DECIMATION_RECAPTURE_FACES`` faces of width, the pass is recorded again at the live width. The
# second condition is the one that usually decides: below it the pass's launches are too small for
# their width to show in the replay, and a recording costs a few replays.
_DECIMATION_RECAPTURE = 0.5
_DECIMATION_RECAPTURE_FACES = 250_000

# Rounds of ``_flip_interior_edges`` issued before the round is recorded and replayed. Recording
# and instantiating a round costs more than issuing it, so a call whose first round finds nothing
# to flip -- a mesh already at the fixpoint -- is cheapest never recording. One issued round is all
# it takes to learn that, and a later threshold only moves the recording's cost around rather than
# removing it: calls that flip at all run a handful of rounds, and the replays repay it.
_FLIP_CAPTURE_FROM_ROUND = 1


def isotropic_remesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_length: float | wp.array[wp.float32] | None = None,
    iterations: int = 10,
    feature_angle: float = 30.0,
    split: bool = True,
    collapse: bool = True,
    swap: bool = True,
    smooth: bool = True,
    reproject: bool = True,
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """

    Isotropic explicit remeshing (Botsch-Kobbelt split / collapse / flip / smooth / reproject).

    GPU port of the classic incremental isotropic remesher (PyMeshLab's
    ``meshing_isotropic_explicit_remeshing``, vcglib ``IsotropicRemeshing``): each iteration drives
    all edge lengths toward ``target_length`` by (1) **splitting** every edge longer than
    ``4/3 * target_length`` at its midpoint (crack-free, reusing
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]); (2) **collapsing** every edge shorter
    than ``4/5 * target_length`` (a parallel primitive with full 1-ring locking and a manifold
    link-condition guard); (3) **flipping** interior edges toward the ideal vertex valence (6
    interior, 4 boundary); (4) **tangentially smoothing** free vertices (area-equalizing Laplacian
    projected onto the tangent plane); and (5) **reprojecting** free vertices back onto the original
    surface. Feature and boundary structure is preserved: each vertex is classified FREE / CREASE
    (on a boundary loop or a dihedral crease sharper than ``feature_angle``) / CORNER (feature
    junction or endpoint), corners are frozen, crease vertices only move along their feature, and
    feature edges are never flipped. Passing ``max_deviation`` relaxes that last guarantee on
    purpose — see its own entry below.

    Inputs are cloned and never mutated.


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer. Whenever a reference
        ``wp.Mesh`` is needed — under ``reproject``, under ``max_deviation``, or with an array
        ``target_length``, any one of which is enough — it aliases ``vertices`` and ``faces``
        rather than copying them; do not mutate them for the duration of the call.
    target_length
        Desired edge length. A **scalar** is the uniform target, defaulting to ``1 %`` of the
        bounding-box diagonal. A ``(n_vertices,)`` ``wp.float32`` array is an **adaptive sizing
        field** over the *input* vertices: every stage then reads its own local target, so the
        result is fine where the field is small and coarse where it is large. This is the general
        form of PyMeshLab's ``adaptive`` flag — rather than deriving the field from curvature
        internally, the caller supplies it, which also covers a painted field, a distance-to-feature
        field, or a field carried from another mesh. Build a curvature-driven one from
        [`triwarp.curvature`][triwarp.curvature], or an interpolated one from
        [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points].
    iterations
        Number of full remeshing passes.
    feature_angle
        Dihedral angle in **degrees** above which an interior edge is treated as a sharp feature
        (protected from flipping and collapsing across).
    split, collapse, swap, smooth, reproject
        Enable/disable each stage of the per-iteration pipeline.
    max_deviation
        Bound on how far the result may move off the input surface, in model units. At the end of
        every iteration each vertex further than this from the input surface is pulled straight back
        toward its own closest point until it is exactly this far, bounding the result's one-sided
        Hausdorff distance to the input. ``None`` (the default) leaves fidelity to ``reproject``
        alone, as before. This is PyMeshLab's ``checksurfdist`` / ``maxsurfdist`` pair as a single
        optional bound. See the Notes for how tightly the bound actually holds.

        **The clamp overrides feature preservation, deliberately, and it is the one thing here
        that does.** It runs over *every* vertex with no regard for its FREE / CREASE / CORNER
        classification, because a crease or a corner is exactly the kind of vertex that drifts and
        that ``reproject`` refuses to touch. So a crease or corner vertex that has strayed past the
        bound is pulled back toward its closest point on the whole input surface, which need not
        lie on its own feature curve. It moves no further than the bound requires, and a vertex
        already inside the band is untouched — but if the feature curves must be honoured exactly,
        leave this ``None``.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` remeshed vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat triangle index buffer.

    Raises
    ------
    ValueError
        If ``target_length`` is non-positive, a sizing field does not have one entry per vertex or
        holds a non-positive value, or ``max_deviation`` is non-positive.
    RuntimeError
        If ``vertices``, ``faces`` and ``target_length`` are not all on one device.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`split_edges`][triwarp.remesh.split_edges]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]

    Notes
    -----
    Hysteresis (split above ``4/3 t``, collapse below ``4/5 t``) keeps split and collapse from
    fighting. The collapse primitive guarantees manifoldness through the link condition but has no
    normal-flip guard in this version, relying on the reprojection step to keep free vertices on the
    original surface.

    An adaptive field is **re-sampled from the input surface** before each stage that reads it, with
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices], rather than
    transported through the split / collapse operations. The field is a property of the input
    geometry, so re-sampling keeps it exact under an arbitrary sequence of operations where
    transport would accumulate error; the cost is one closest-point query per vertex per stage.
    Inside a single stage the field *is* transported, because there the correspondence is known
    exactly: a split midpoint takes the mean of the endpoints it splits, and a collapse compacts the
    bands alongside the vertices.

    A **constant** field is not quite the scalar path: the two agree on the face buffer exactly and
    on positions to within float rounding. The gap is float rounding in the threshold alone -- the
    array path forms ``4/3 * t`` per vertex in ``float32`` where the scalar path forms it in Python
    ``float64`` and narrows once. Pass a scalar when the target is uniform; it is also one
    closest-point query per stage cheaper.

    ``max_deviation`` is a **positional bound applied per iteration**, not a per-operation rejection
    test: an individual collapse or flip is never vetoed for moving the surface too far, it is the
    accumulated vertex position that is corrected afterwards. A mesh whose *edges* must never sweep
    past the bound mid-iteration needs the flip stage's own gate as well
    ([`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] takes one).

    How tightly the bound holds is set by ``wp.mesh_query_point_no_sign`` rather than by this
    function. Against that query -- the one the clamp is implemented with -- the result is within
    the bound almost exactly. Against an independent closest-point query, the bound controls
    deviation proportionally, but at a bound near Warp's own query accuracy it becomes approximate
    rather than hard, because the two queries can disagree by a small absolute amount and iterating
    the clamp does not converge further (Warp's answer is a fixed point). Ask for a bound
    comfortably above the scale of a single-precision closest-point query, or scale the model up.

    One limitation worth stating: PyMeshLab's ``selectedonly`` has no equivalent here, and a
    region-restricted refinement is
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size] rather than a mode of this
    function.
    """
    require_same_device(vertices=vertices, faces=faces, target_length=target_length)
    n_faces = faces.size // 3
    device = vertices.device
    n_vertices = vertices.size
    current_vertices = _launch.clone(vertices)
    current_faces = _launch.clone(faces)
    if n_faces == 0 or iterations <= 0:
        return current_vertices, current_faces

    diag = tw.bounds.enclosing_diagonal(vertices)
    sizing_input: wp.array[wp.float32] | None = None
    if target_length is None or isinstance(target_length, int | float):
        target = float(target_length) if target_length is not None else 0.01 * diag
        if target <= 0.0:
            raise ValueError(f"isotropic_remesh requires target_length > 0, got {target}.")
    else:
        field = target_length
        if field.size != n_vertices:
            raise ValueError(
                f"isotropic_remesh requires one target_length per vertex ({n_vertices}), "
                f"got {field.size}."
            )
        # One readback, on a buffer the caller just built: a non-positive entry makes the split
        # stage diverge (every edge over-long), so it is worth catching here rather than at
        # ``max_iter``.
        #
        # Two reductions over one buffer, and merging them into a single min-and-sum kernel is
        # declined: this runs once per call on the optional sizing-field path, against a call that
        # issues hundreds of launches per iteration, so the pair is well under a percent of it and
        # a new reduction kernel would exist for one site.
        smallest = float(tw.reduce.min(field))
        if smallest <= 0.0:
            raise ValueError(
                f"isotropic_remesh requires a positive target_length everywhere, got {smallest}."
            )
        target = float(tw.reduce.mean(field))
        sizing_input = field
    if max_deviation is not None and max_deviation <= 0.0:
        raise ValueError(f"isotropic_remesh requires max_deviation > 0, got {max_deviation}.")
    feature = wp.float32(math.radians(feature_angle))

    # Original surface, built once, for reprojecting free vertices and for the deviation bound
    # (never a 0-triangle mesh).
    original_mesh = None
    if reproject or max_deviation is not None or sizing_input is not None:
        require_nonempty_mesh(faces, "isotropic_remesh")
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy: the
        # loop below rebinds ``current_vertices`` / ``current_faces`` and never writes ``vertices``.
        original_mesh = wp.Mesh(points=vertices, indices=faces)
    query_radius = max(diag, 1.0)
    clamp_kernel = None
    if max_deviation is not None:
        # Hoisted out of the loop: the generated kernel is cached, but the per-call Python is not.
        clamp_kernel = cast(
            "wp.Kernel",
            wp.map(
                kernel_remesh.clamp_to_surface_band,
                current_vertices,
                wp.uint64(0),
                wp.float32(0.0),
                wp.float32(0.0),
                out=_launch.empty_like(current_vertices),
                return_kernel=True,
            ),
        )

    for _ in range(iterations):
        # The sizing field is re-sampled from the *input* surface immediately before each stage that
        # reads it, because the stage before it changed the vertex set. Two closest-point passes per
        # iteration is the price of keeping the field exact; see this function's Notes.
        if split:
            # Only the *high* band gates a split, and with no sizing field it is one constant at
            # every vertex -- which ``subdivide_to_size`` takes as a scalar. So the uniform path
            # builds neither the two per-vertex band buffers nor the ``low`` band that only the
            # collapse stage below reads.
            split_limit: float | wp.array[wp.float32] = 4.0 / 3.0 * target
            if sizing_input is not None:
                split_limit = _length_bands(
                    _sizing_at(current_vertices, vertices, faces, sizing_input, query_radius),
                    target,
                    current_vertices.size,
                    device,
                )[1]
            current_vertices, current_faces = subdivide_to_size(
                current_vertices, current_faces, split_limit, max_iter=20
            )
        # The edge grouping of ``current_faces`` whenever a stage leaves one behind, so the next
        # stage that classifies the same faces need not regroup them.
        incidence: _EdgeIncidence | None = None
        if collapse:
            low, high = _length_bands(
                _sizing_at(current_vertices, vertices, faces, sizing_input, query_radius),
                target,
                current_vertices.size,
                device,
            )
            current_vertices, current_faces, incidence = _collapse_pass(
                current_vertices, current_faces, low, high, feature
            )
        if current_faces.size == 0:
            break
        if swap and _valence_flip_pass(
            current_vertices, current_faces, feature, incidence=incidence
        ):
            incidence = None  # a flip rewrote the faces in place, so the grouping is stale
        if smooth or reproject:
            # One edge grouping serves both the classification and the smoothing ring: the two read
            # the same faces, and the smooth step would otherwise hash and sort them a second time.
            if incidence is None:
                incidence = _edge_incidence(current_faces, current_vertices.size)
            codes, _boundary = _classify(current_vertices, current_faces, feature, incidence)
            if smooth:
                current_vertices = _smooth_pass(
                    current_vertices, current_faces, codes, incidence.unique_edges
                )
            if reproject and original_mesh is not None:
                current_vertices = _reproject_pass(
                    current_vertices, codes, original_mesh, query_radius
                )
        if clamp_kernel is not None and original_mesh is not None and max_deviation is not None:
            bounded = _launch.empty_like(current_vertices)
            _launch.launch(
                clamp_kernel,
                dim=current_vertices.size,
                inputs=[
                    current_vertices,
                    wp.uint64(original_mesh.id),
                    wp.float32(max_deviation),
                    wp.float32(query_radius),
                ],
                outputs=[bounded],
                device=device,
            )
            current_vertices = bounded

    return current_vertices, current_faces


def _sizing_at(
    positions: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sizing_input: wp.array[wp.float32] | None,
    query_radius: float,
) -> wp.array[wp.float32] | None:
    """
    Sample the input mesh's sizing field at ``positions``, via their closest points on that surface.

    ``None`` in, ``None`` out, so the uniform-target path costs nothing and the caller needs no
    branch of its own.
    """
    if sizing_input is None:
        return None
    return tw.interpolation.transfer_onto_vertices(
        vertices, faces, sizing_input, positions, max_dist=query_radius
    )[0]


def _length_bands(
    sizing: wp.array[wp.float32] | None, target: float, n_vertices: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Per-vertex collapse-below and split-above length bands, from a sizing field or a uniform target.

    The Botsch-Kobbelt hysteresis (``4/5 t`` and ``4/3 t``) is applied per vertex so the uniform
    case is literally the constant field, letting the collapse kernel keep a single code path.
    """
    if sizing is None:
        low = _launch.full(n_vertices, 4.0 / 5.0 * target, dtype=wp.float32, device=device)
        high = _launch.full(n_vertices, 4.0 / 3.0 * target, dtype=wp.float32, device=device)
        return low, high
    low = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    high = _launch.empty(n_vertices, dtype=wp.float32, device=device)
    # One map, two outputs: both bands are one scale of the same entry, so a second pass would
    # only re-read the sizing field to multiply it by the other constant.
    _launch.map(kernel_remesh.hysteresis_bands, sizing, out=[low, high])
    return low, high


def _classify(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature: wp.float32,
    incidence: _EdgeIncidence | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Per-vertex FREE / CREASE / CORNER codes plus a boundary-vertex mask.

    A boundary edge or an interior edge sharper than ``feature`` counts as one incident feature
    edge; a vertex with zero is FREE, exactly two is CREASE (a smooth feature/boundary line), and
    anything else (a feature endpoint or a junction) is a frozen CORNER.

    Both questions are answered by **one launch** over ``incidence``, which the collapse passes
    have already built for their own scoring and pass in. That matters because computing the
    answer from ``boundary.boundary_edges`` plus ``adjacency.face_adjacency`` would hash, sort and
    group the same ``3 * n_faces`` edge rows the incidence was already grouped from — sharing the
    grouping avoids that repeated work whenever a caller has it in hand.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    boundary_vertex = _launch.zeros(n_vertices, dtype=wp.bool, device=device)
    if n_faces == 0:
        return _launch.zeros(n_vertices, dtype=wp.int32, device=device), boundary_vertex

    if incidence is None:
        incidence = _edge_incidence(faces, n_vertices)
    m = int(incidence.unique_edges.shape[0])
    if m == 0:
        return _launch.zeros(n_vertices, dtype=wp.int32, device=device), boundary_vertex

    # The feature counts become the codes in place.
    feature_count = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.scatter_feature_edge_counts,
        dim=m,
        inputs=[
            vertices,
            faces,
            incidence.unique_edges,
            incidence.face_count,
            incidence.faces,
            feature,
            feature_count,
            boundary_vertex,
        ],
        device=device,
    )
    _launch.map(kernel_remesh.finalize_vertex_codes, feature_count, out=feature_count)
    return feature_count, boundary_vertex


def _edge_incidence(faces: wp.array[wp.int32], n_vertices: int) -> _EdgeIncidence:
    """
    Group a triangulation's edge rows once, into unique edges and their incident faces.

    The face table costs one scatter over the corners on top of the ``edges_unique`` the collapse
    passes run anyway -- and it replaces the ``scatter.count_occurrences`` launch they used to make
    for the count alone, so it is very nearly free. It is what lets ``_classify`` skip a second and
    third grouping of the same rows; see ``scatter_edge_incidence`` in ``kernels/scatter.py``.
    """
    device = faces.device
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    m = int(unique_edges.shape[0])
    face_count = _launch.zeros(m, dtype=wp.int32, device=device)
    edge_faces = twt.empty_2d((m, 2), wp.int32, device=device)
    if m > 0:
        _launch.launch(
            kernel_scatter.scatter_edge_incidence,
            dim=inverse.size,
            inputs=[inverse, face_count, edge_faces],
            device=device,
        )
    return _EdgeIncidence(unique_edges, face_count, twt.as_array2d(edge_faces, wp.int32))


def _collapse_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    feature: wp.float32,
    max_passes: int = 5,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], _EdgeIncidence | None]:
    """
    Collapse short edges in parallel with 1-ring locking; returns compacted (vertices, faces).

    ``low`` and ``high`` are the per-vertex length bands, so one path serves both a uniform target
    and a sizing field. They are compacted alongside the vertices at the end of each pass rather
    than re-sampled, which keeps the whole loop free of closest-point queries.

    The third return is the edge grouping of the returned faces when the loop has one in hand --
    it stopped on a pass that committed nothing, so the faces it grouped are the ones it returns --
    and ``None`` otherwise, so the next stage can classify the same faces without regrouping them.
    """
    device = vertices.device
    # One commit counter for the whole loop, never reset: it accumulates across passes and a pass
    # committed nothing exactly when it did not move.
    count = _launch.zeros(1, dtype=wp.int32, device=device)
    committed = 0
    current: _EdgeIncidence | None = None
    for _ in range(max_passes):
        current = None
        n_vertices = vertices.size
        n_faces = faces.size // 3
        if n_faces == 0:
            break

        # One edge grouping per pass, shared by the candidate scoring and ``_classify``.
        incidence = _edge_incidence(faces, n_vertices)
        unique_edges = incidence.unique_edges
        m = int(unique_edges.shape[0])
        if m == 0:
            break
        codes, _boundary = _classify(vertices, faces, feature, incidence)
        # Only the neighbour *sets* are read -- the link condition counts shared neighbours, the
        # band walks and the ring locks exit or take a minimum over a whole row -- so the rows need
        # no order, and a degree count and a cursor fill build them with no sort. The unique edges
        # came from this very face buffer, so they are in range.
        ring_neighbors, ring_offsets = tw.graph.edges_to_neighbor_lists(
            n_vertices, unique_edges, validate=False
        )
        # The vertex-face CSR exists only for ``collapse_candidates``' fold veto, which needs the
        # faces incident to a vertex where the rings above have only its neighbours. It is about a
        # percent of the collapse stage, and the veto's own per-candidate work does not show above
        # run-to-run noise -- see the veto itself in ``kernels/remesh.collapse_candidates``.
        vertex_faces, face_offsets = tw.adjacency.vertex_face_adjacency(
            faces, n_vertices=n_vertices
        )

        survivor = _launch.empty(m, dtype=wp.int32, device=device)
        removed = _launch.empty(m, dtype=wp.int32, device=device)
        target_pos = _launch.empty(m, dtype=wp.vec3, device=device)
        # Seeded by ``collapse_candidates``: unclaimed keys (64-bit, because the lock key is -- see
        # ``kernel_remesh.scramble_index`` for why it has to be injective), the identity map and the
        # working positions.
        claim = _launch.empty(n_vertices, dtype=wp.int64, device=device)
        remap = _launch.empty(n_vertices, dtype=wp.int32, device=device)
        positions = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_remesh.collapse_candidates,
            dim=m,
            inputs=[
                unique_edges,
                vertices,
                faces,
                codes,
                incidence.face_count,
                ring_offsets,
                ring_neighbors,
                face_offsets,
                vertex_faces,
                low,
                high,
                survivor,
                removed,
                target_pos,
                claim,
                remap,
                positions,
            ],
            device=device,
        )

        # **Folding this launch into the candidate kernel above is declined**, on the same
        # measurement as the flip engine's equivalent in ``_flip_interior_edges``: the claim reads
        # ``survivor[k]`` / ``removed[k]`` at its own thread and its claim table is seeded by the
        # candidate kernel itself, so it would fuse -- but a pass is a topology rebuild plus a
        # handful of launches around one readback, and this is one of them. The claim/commit pair
        # after it is not fusible at all.
        _launch.launch(
            kernel_remesh.claim_collapse_key,
            dim=m,
            inputs=[survivor, removed, ring_offsets, ring_neighbors, claim],
            device=device,
        )
        _launch.launch(
            kernel_remesh.commit_collapses,
            dim=m,
            inputs=[
                survivor,
                removed,
                target_pos,
                ring_offsets,
                ring_neighbors,
                claim,
                remap,
                positions,
                count,
            ],
            device=device,
        )
        total = int(read_scalar(count, 0))
        if total == committed:
            current = incidence
            break
        committed = total

        # Drop the faces the collapses degenerated and the vertices no face names any more, in one
        # scan: the positions and both length bands are compacted together, so neither band is
        # re-sampled.
        ranks, faces, n_kept = _compact_remapped_faces(faces, remap, n_vertices)
        vertices = _launch.empty(n_kept, dtype=wp.vec3, device=device)
        kept_low = _launch.empty(n_kept, dtype=wp.float32, device=device)
        kept_high = _launch.empty(n_kept, dtype=wp.float32, device=device)
        if n_kept > 0:
            _launch.launch(
                kernel_remesh.compact_collapse_vertices,
                dim=n_vertices,
                inputs=[positions, low, high, ranks, vertices, kept_low, kept_high],
                device=device,
            )
        low, high = kept_low, kept_high

    return vertices, faces, current


def _valence_flip_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature: wp.float32,
    max_iter: int = 10,
    incidence: _EdgeIncidence | None = None,
) -> int:
    """
    Flip interior edges toward ideal valence (6 interior, 4 boundary); mutates ``faces``.

    ``incidence``, when given, is the edge grouping of ``faces`` as they are on entry; the boundary
    classification reads it rather than grouping the rows again. Returns the number of flips, so a
    caller holding that grouping knows whether it still describes ``faces``.
    """
    device = faces.device
    n_vertices = vertices.size
    _codes, boundary_vertex = _classify(vertices, faces, feature, incidence)

    # Valence is the loop's own: seeded by its one build, from the keys that build radix-sorts,
    # and kept current by every committed flip (a flip moves one edge from ``a``, ``c`` to ``b``,
    # ``d``), rather than recounted from a re-sort each round.
    valence = _launch.empty(n_vertices, dtype=wp.int32, device=device)

    def launch(
        adjacency: twt.Array2dInt32,
        adjacency_edges: twt.Array2dInt32,
        unshared: twt.Array2dInt32,
        sorted_keys: wp.array[wp.uint64],
        edge_set: wp.array[wp.uint64],
        edge_set_mask: wp.int32,
        key_base: wp.uint64,
        out_flip: wp.array[wp.bool],
        out_quad: twt.Array2dInt32,
    ) -> None:
        _launch.launch(
            kernel_remesh.valence_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                edge_set,
                edge_set_mask,
                key_base,
                valence,
                boundary_vertex,
                feature,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    return _flip_interior_edges(faces, n_vertices, launch, max_iter, valence=valence)


def _smooth_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    codes: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.vec3]:
    """
    One area-equalizing tangential relaxation step over free vertices.

    Each neighbour is weighted by its own barycentric area, which is the form Botsch-Kobbelt
    specify and the one ``isotropic_remesh``'s Notes describe; the plain one-ring centroid this
    replaced was a fixed point on exactly the graded input the stage exists for. Every proposed
    move is vetoed if it would invert an incident face, by the same rule the collapse stage runs --
    see ``kernels/remesh.accumulate_one_ring`` for the quality measurement and
    ``kernels/remesh.smooth_free_vertices`` for why the veto is load-bearing rather than defensive.
    """
    device = vertices.device
    n_vertices = vertices.size
    normals = tw.vertices.vertex_normals(vertices, faces)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    vertex_areas = tw.laplacian.mass_matrix_entries(vertices, faces)
    # The same vertex-face CSR the collapse stage builds for its own fold veto, and for the same
    # reason: ``unique_edges`` above carries a vertex's *neighbours*, never its faces.
    vertex_faces, face_offsets = tw.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)

    ring_sum = _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
    ring_weight = _launch.zeros(n_vertices, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_remesh.accumulate_one_ring,
        dim=int(unique_edges.shape[0]),
        inputs=[unique_edges, vertices, vertex_areas, ring_sum, ring_weight],
        device=device,
    )
    out_positions = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_remesh.smooth_free_vertices,
        dim=n_vertices,
        inputs=[
            vertices,
            faces,
            codes,
            normals,
            ring_sum,
            ring_weight,
            face_offsets,
            vertex_faces,
            wp.float32(1.0),
            out_positions,
        ],
        device=device,
    )
    return out_positions


def _reproject_pass(
    vertices: wp.array[wp.vec3], codes: wp.array[wp.int32], original_mesh: wp.Mesh, max_dist: float
) -> wp.array[wp.vec3]:
    """Snap free vertices onto the closest point of the original surface."""
    device = vertices.device
    n_vertices = vertices.size
    out_positions = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    # ``wp.uint64(...)`` is required: a bare ``wp.Mesh.id`` is a Python int and ``wp.map`` would
    # infer ``int32`` for it (see tests/test_map_uniform_probe.py).
    _launch.map(
        kernel_remesh.reproject_vertices,
        vertices,
        codes,
        wp.uint64(original_mesh.id),
        wp.float32(max_dist),
        out=out_positions,
    )
    return out_positions


def _flip_interior_edges(
    faces: wp.array[wp.int32],
    n_vertices: int,
    launch_candidates: _LaunchCandidates,
    max_iter: int,
    topology: _FlipTopology | None = None,
    valence: wp.array[wp.int32] | None = None,
) -> int:
    """
    Repeatedly flip an independent set of interior edges until none is a candidate.

    ``faces`` is mutated in place. Each iteration lets ``launch_candidates`` mark flippable edges
    (predicate-specific) and commits a conflict-free subset (no two committed flips touch a shared
    face or create the same new edge). The first flipping round is followed by a regroup that also
    builds ``_FlipTopology``'s incremental state; every later round refreshes the adjacency rows
    from the halfedge maps the commit kept current instead of regrouping. Returns the total number
    of flips performed. The winding rewrite matches ``igl::flip_edge``.

    ``launch_candidates(adjacency, adjacency_edges, unshared, sorted_keys, edge_set, edge_set_mask,
    key_base, out_flip, out_quad)`` receives the current rows and the current edge keys for the
    duplicate-edge guard, which ``kernel_remesh.edge_key_exists`` reads: the last build's sorted
    keys while ``edge_set_mask`` is ``-1``, and the hashed set once a flip has made the sort stale.
    ``valence``, when given, is a per-vertex buffer the loop seeds with each vertex's unique-edge
    count and keeps current across flips, for a predicate that reads it.

    [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay] does not use this engine: it carries
    an edge-length table beside the face buffer and needs to create a second edge between two
    already-adjacent vertices, which this loop's vertex-pair-keyed topology cannot represent. It
    drives its own halfedge-twin-based loop instead (see ``kernel_remesh.build_intrinsic_twins``).

    A caller may pass its own ``topology`` over this very ``faces`` buffer (and ``n_vertices``).
    Either way it is left describing the face buffer as this call leaves it, so the caller can read
    the unique edges afterwards (``_FlipTopology.edges_unique``, which re-sorts if a flip made the
    sort stale) or hand the same object to the next call on the same buffer, which then skips its
    opening build.
    """
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0 or max_iter <= 0:
        return 0
    if topology is None:
        topology = _FlipTopology(faces, n_vertices)
    key_base = wp.uint64(n_vertices)
    # The one readback of the row count. A committed flip trades one two-face edge for a new one
    # (the duplicate-edge guard every candidate kernel opens with rejects a flip whose new edge
    # already exists, and the claim table stops two flips creating the same one) and leaves the
    # quad's four sides alone, so the interior-edge count is fixed from here on and every row keeps
    # describing one edge. A tracked valence is seeded by a build, so it forces one.
    topology.valence = valence
    m = topology.rows if topology.built and valence is None else topology.rebuild()
    if m == 0:
        return 0

    def flip_round(progress: wp.array[wp.int32], *, refresh: bool = True) -> None:
        """Mark, claim and commit one independent set of flips, then refresh the rows they moved."""
        launch_candidates(
            topology.adjacency,
            topology.adjacency_edges,
            topology.unshared,
            topology.sorted_keys,
            topology.edge_set,
            wp.int32(topology.active_edge_set_mask),
            key_base,
            topology.flip,
            topology.quad,
        )
        # Independent-set selection: a flip commits only if it wins both incident faces and the
        # hashed slot of its new edge (prevents two disjoint flips creating the same edge).
        #
        # **Folding this launch into the candidate kernels above is measured and declined.** It
        # passes the index-locality test -- ``claim_flips`` reads ``flip[k]``, ``quad[k, *]`` and
        # ``adjacency[k, *]`` at its own thread, and the two ``fill_`` calls touch buffers the
        # candidate kernel never reads, so they would simply move ahead of a fused launch. What
        # it is not is worth it: every one of the four candidate kernels decides ``out_flip`` at
        # two or three separate exits (``objective_flip_candidates`` at three), so each would
        # have to be restructured around a single exit for the claim to hang off, in predicate
        # code where a mistake silently changes which edges flip -- for one launch of a round's
        # dozen. The claim/commit pair below is not fusible at all (a commit must see every
        # claim).
        #
        # A refresh re-arms both claim tables for the round after it, so only a round that follows
        # a build or another issued round fills them itself.
        topology.arm_claims()
        _launch.launch(
            kernel_remesh.claim_flips,
            dim=m,
            inputs=[
                topology.flip,
                topology.quad,
                topology.adjacency,
                topology.adjacency_edges,
                wp.int32(topology.edge_claim_mask),
                key_base,
                topology.face_claim,
                topology.edge_claim,
            ],
            device=device,
        )
        _launch.launch(
            kernel_remesh.commit_flips,
            dim=m,
            inputs=[
                topology.flip,
                topology.quad,
                topology.adjacency,
                topology.adjacency_edges,
                topology.face_claim,
                topology.edge_claim,
                wp.int32(topology.edge_claim_mask),
                key_base,
                wp.int32(topology.active_edge_set_mask),
                topology.row_halfedges,
                topology.halfedge_row,
                topology.edge_set,
                valence,
                faces,
                progress,
            ],
            device=device,
        )
        topology.claims_armed = False
        # The commit kept the halfedge <-> row maps, the duplicate-edge set and the valences
        # current, so the rows are rebuilt in place from them: one launch over the rows instead of
        # a whole-mesh regroup (a key launch, a radix sort, a mark, a scan and an emit). An issued
        # round leaves it to the caller, which refreshes only if something flipped.
        if refresh:
            topology.refresh()

    # The commits accumulate across rounds and are never reset: a round's flips are what the
    # running total moved by.
    count = _launch.zeros(1, dtype=wp.int32, device=device)
    graph = None
    capturable = wp.get_device(device).is_cuda
    total = 0
    for round_index in range(max_iter):
        if graph is not None:
            wp.capture_launch(graph)
        elif capturable and round_index >= _FLIP_CAPTURE_FROM_ROUND:
            # Every round is the same launch sequence over the same buffers -- only the face
            # buffer's contents change -- so a long call records one round and replays it, and a
            # round costs one graph launch rather than a dozen launches' worth of Python. Replayed
            # from the host, which reads each round's count: it is what tells
            # ``topology.flipped`` when the duplicate-edge set's tombstones call for a rebuild.
            # The claim tables are armed before recording, so the recorded round -- whose refresh
            # re-arms them -- carries no fill.
            topology.arm_claims()
            with wp.ScopedCapture(device) as capture:
                flip_round(count)
            graph = capture.graph
            assert graph is not None
            wp.capture_launch(graph)
        else:
            # An issued round reads its count before refreshing, so a round that flipped nothing --
            # the common last round, and the only one a call at the fixpoint runs -- skips it.
            flip_round(count, refresh=False)
            n = int(read_scalar(count, 0)) - total
            total += n
            if n == 0:
                break
            if topology.incremental:
                topology.refresh()
                topology.flipped(n)
            else:
                # A plain build's first flipping round regroups, as every round once did, and that
                # regroup builds the incremental state the later rounds -- the recorded ones --
                # run on. A call that flips at most once never pays for it.
                topology.rebuild(read_count=False, incremental=True)
            continue
        n = int(read_scalar(count, 0)) - total
        total += n
        if n == 0:
            break
        topology.flipped(n)
    return total


class _FlipTopology:
    """
    Persistent working set of the parallel edge-flip loop.

    A flip rewrites two triangles' corners and leaves the vertex, face and interior-edge counts
    alone, so every buffer here is allocated once and rewritten in place, and a long flip loop
    maintains the face adjacency rather than rebuilding it every round.

    A build is one key launch, one radix sort, a run-length mark, a scan and a single emit that
    writes the adjacency pairs, their shared-edge endpoints and the opposite apexes together. An
    *incremental* build also keeps the two halfedges of each row in both directions and seeds the
    set of every edge key, and from then on a flip round never regroups: ``commit_flips`` moves the
    halfedges of the rows a flip touched, swaps the old diagonal's key for the new one in the set,
    and ``refresh`` rewrites every row from its halfedges in one launch. The flip loop starts
    plain and takes the incremental build as the regroup after its first flipping round, so a call
    that flips at most once pays for none of it.

    Each edge keeps its row, so the rows stop being in ascending key order after the first
    incremental flip -- which no consumer depends on, since the independent-set claims rank
    candidates by *key* (``kernel_remesh.flip_priority``) and the smallest key was exactly the
    smallest row index of a fresh build. The loop is therefore byte-identical to one that regroups
    every round. ``edges_unique`` needs the sort itself and re-sorts once if a flip made it stale.

    The key set is a ``grouping.hash_find_or_insert`` table with tombstone deletes, sized at twice
    the corner count. A delete never frees a slot, so ``flipped`` rebuilds everything once the flips
    since the last build could have filled it past three quarters.

    [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay] uses this class for exactly one
    build: its input is still a simplicial complex at that point, so the vertex-pair key this class
    groups on is trustworthy, and the one build seeds an incrementally-maintained halfedge twin
    table that the rest of its loop drives instead (see ``kernel_remesh.build_intrinsic_twins``).

    Every intermediate of a build is byte-identical to the composed path: the keys match
    [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] over
    [`faces_to_edges`][triwarp.edges.faces_to_edges] rows (see
    [`face_edge_keys`][triwarp.kernels.adjacency.face_edge_keys]), and the scan reproduces the
    ascending-key row order [`group`][triwarp.grouping.group] gets from its ``flatnonzero``
    compaction.

    Attributes
    ----------
    sorted_keys : wp.array[wp.uint64]
        Length ``3 * n_faces`` ascending undirected-edge keys as of the last build -- stale once a
        flip commits (``edges_unique`` re-sorts first).
    edge_set : wp.array[wp.uint64]
        Every current undirected edge key, as a ``grouping.hash_find_or_insert`` set, for the
        candidate predicates' duplicate-edge guard once an incremental build seeded it.
    edge_set_mask : int
        Power-of-two mask for ``edge_set`` slots.
    adjacency : twt.Array2dInt32
        ``(m, 2)`` ascending face pairs sharing an interior edge.
    adjacency_edges : twt.Array2dInt32
        ``(m, 2)`` sorted endpoints of each shared edge, row-aligned with ``adjacency``.
    unshared : twt.Array2dInt32
        ``(m, 2)`` apex of each incident face opposite the shared edge.
    row_halfedges : twt.Array2dInt32 | None
        ``(m, 2)`` the two halfedges (``3f + corner``) each row's edge is made of; ``None`` until an
        incremental build.
    halfedge_row : wp.array[wp.int32] | None
        Length ``3 * n_faces`` row of each halfedge's edge, or ``-1`` for an edge that is not a
        two-face interior edge; ``None`` until an incremental build.
    flip : wp.array[wp.bool]
        Length ``m`` candidate mask. Every predicate kernel opens by writing all of it, so it is
        deliberately not zeroed between passes.
    quad : twt.Array2dInt32
        ``(m, 4)`` flip quad ``(a, b, c, d)``, written only where ``flip`` is set.
    face_claim : wp.array[wp.uint64]
        Length ``n_faces`` per-face winning key of the independent-set round.
    edge_claim : wp.array[wp.uint64]
        Open-addressed claim table over the new edges, one slot per hashed key.
    edge_claim_mask : int
        Power-of-two mask for ``edge_claim`` slots.
    claims_armed : bool
        Whether ``face_claim`` and ``edge_claim`` currently hold the unclaimed key everywhere.
    valence : wp.array[wp.int32] | None
        Per-vertex unique-edge count a build seeds and ``commit_flips`` keeps current, when the
        loop's caller asked for one.
    """

    def __init__(self, faces: wp.array[wp.int32], n_vertices: int) -> None:
        """Allocate the fixed working set for the ``faces`` buffer the loop will mutate in place."""
        self._faces = faces
        self._device = faces.device
        self._n_faces = faces.size // 3
        self._n_corners = self._n_faces * 3
        self._n_vertices = n_vertices
        self._radix = wp.uint64(n_vertices)
        # A key packs an edge as ``min + max * n_vertices``, so every key is below
        # ``n_vertices ** 2`` and the sort orders only that many low bits: the stable sort's
        # permutation is identical, and a 64-bit sort's fixed per-digit passes dominate a build.
        self._key_bits = max(1, (n_vertices * n_vertices - 1).bit_length())
        n = self._n_corners
        # ``radix_sort_pairs`` ping-pongs through the upper half of both buffers, so each is
        # double width and only ``[:n]`` is data.
        self._keys = _launch.empty(2 * n, dtype=wp.uint64, device=self._device)
        self._order = _launch.empty(2 * n, dtype=wp.int32, device=self._device)
        self._starts = _launch.empty(n, dtype=wp.int32, device=self._device)
        self._ranks = _launch.empty(n, dtype=wp.int32, device=self._device)
        self._ranks_tail = twt.as_dense(self._ranks[n - 1 :])
        self.sorted_keys = twt.as_dense(self._keys[:n])
        self.face_claim = _launch.empty(self._n_faces, dtype=wp.uint64, device=self._device)
        # Whether both claim tables hold the unclaimed key: a refresh arms them for the next round,
        # a commit spends them.
        self.claims_armed = False
        # The incremental state is allocated by the first incremental build; a plain build passes
        # ``None`` for it (a null descriptor, which the kernels read as "no incremental state").
        self.incremental = False
        self.halfedge_row: wp.array[wp.int32] | None = None
        self.row_halfedges: twt.Array2dInt32 | None = None
        table = 1
        while table < 2 * n:
            table <<= 1
        self._edge_set_table = table
        # Until an incremental build seeds it, ``edge_set`` stands in with a buffer of the right
        # dtype that no thread reads (the guard takes the sorted keys while the mask is ``-1``).
        self.edge_set = self._keys
        self.edge_set_mask = table - 1
        # Unique edges at a build are at most ``n``, and each flip spends one empty slot, so
        # ``table * 3 // 4 - n`` flips keep the set under three-quarters full.
        self._flip_budget = max(0, table * 3 // 4 - n)
        self._flips_since_build = 0
        self._sorted_stale = False
        self.valence: wp.array[wp.int32] | None = None
        self._rows = -1
        self._allocate_rows(0)
        self.built = False

    @property
    def rows(self) -> int:
        """Interior-edge row count of the last build (``0`` before one)."""
        return self._rows

    def rebuild(self, *, read_count: bool = True, incremental: bool = False) -> int:
        """
        Build every table from the face buffer, and return the interior-edge row count.

        Every public attribute is rewritten; the returned row count is also the launch dimension
        for the candidate, claim and commit kernels. ``read_count=False`` skips the one readback,
        for a rebuild of a face buffer whose row count is already known (a flip loop's), and
        returns the known count. ``incremental=True`` also builds the state a flip round keeps
        current -- the halfedge <-> row maps and the duplicate-edge set -- and the topology stays
        incremental from then on.

        **It is whole-mesh, not region-scoped**: ``_flip_region_faces`` builds over the whole mesh
        even though only edges with *both* faces in the region are flippable, because the
        duplicate-edge set must hold every edge -- a flip's new edge may already exist outside the
        region.
        """
        n = self._n_corners
        self._sort()
        self.built = True
        self._flips_since_build = 0
        self.incremental = self.incremental or incremental
        if self.incremental:
            if self.halfedge_row is None:
                self.halfedge_row = _launch.empty(n, dtype=wp.int32, device=self._device)
                self.edge_set = _launch.zeros(
                    self._edge_set_table, dtype=wp.uint64, device=self._device
                )
            else:
                _launch.zero_(self.edge_set)
        if self.valence is not None:
            _launch.zero_(self.valence)
        _launch.launch(
            kernel_remesh.mark_edge_pair_starts,
            dim=n,
            inputs=[
                self._keys,
                self._order,
                wp.int32(n),
                wp.int32(self.active_edge_set_mask),
                self._radix,
                self.edge_set,
            ],
            outputs=[self._starts, self.halfedge_row, self.valence],
            device=self._device,
        )
        # Inclusive, so the row count is one 4-byte tail read and the emit kernel's row is
        # ``ranks[i] - 1`` -- the contract ``array.flatnonzero`` uses for the same reason.
        _launch.array_scan(self._starts, out_array=self._ranks, inclusive=True)
        if read_count:
            m = int(read_scalar(self._ranks_tail, 0))
            if m != self._rows:
                self._allocate_rows(m)
        else:
            m = self._rows
            # The invariant the skipped read rests on, checked where Warp's debug mode is already
            # paying for bounds checks: a rebuild whose row count moved would index past the tables.
            if wp.config.mode == "debug" and not wp.get_device(self._device).is_capturing:
                assert int(read_scalar(self._ranks_tail, 0)) == m, "interior-edge count changed"
        if self.incremental and (self.row_halfedges is None or self.row_halfedges.shape[0] != m):
            self.row_halfedges = twt.empty_2d((m, 2), wp.int32, device=self._device)
        if m > 0:
            _launch.launch(
                kernel_remesh.emit_flip_topology,
                dim=n,
                inputs=[
                    self._faces,
                    self._order,
                    self._starts,
                    self._ranks,
                    self.adjacency,
                    self.adjacency_edges,
                    self.unshared,
                    self.row_halfedges,
                    self.halfedge_row,
                ],
                device=self._device,
            )
        return m

    def _sort(self) -> None:
        """Pack every corner's edge key and radix-sort them, with the corner order as payload."""
        n = self._n_corners
        # Keys and identity payload in one launch, into the leading halves only: the upper halves
        # are the sort's scratch and need no fill.
        _launch.launch(
            kernel_adjacency.face_edge_keys_and_order,
            dim=self._n_faces,
            inputs=[self._faces, self._radix, self._keys, self._order],
            device=self._device,
        )
        _launch.radix_sort_pairs(self._keys, self._order, count=n, end_bit=self._key_bits)
        self._sorted_stale = False

    def refresh(self) -> None:
        """
        Rewrite every row from the halfedges ``commit_flips`` kept current, in one launch.

        The same launch re-arms both claim tables for the next round.
        """
        _launch.launch(
            kernel_remesh.refresh_flip_rows,
            dim=self._rows,
            inputs=[
                self._faces,
                self.row_halfedges,
                self.adjacency,
                self.adjacency_edges,
                self.unshared,
                self.face_claim,
                self.edge_claim,
            ],
            device=self._device,
        )
        self.claims_armed = True

    def arm_claims(self) -> None:
        """Fill both claim tables with the unclaimed key, unless a refresh already did."""
        if not self.claims_armed:
            _launch.fill_(self.face_claim, UINT64_MAX)
            _launch.fill_(self.edge_claim, UINT64_MAX)
            self.claims_armed = True

    @property
    def active_edge_set_mask(self) -> int:
        """``edge_set_mask`` once an incremental build seeded the key set, else ``-1``."""
        return self.edge_set_mask if self.incremental else -1

    def flipped(self, count: int) -> None:
        """
        Record ``count`` committed flips, rebuilding once the key set's tombstones call for it.

        A flip makes the sort stale (``edges_unique`` re-sorts), and a plain build's first flipping
        round is followed by the incremental build the loop runs from then on. Once the flips since
        the last build could have filled the key set past three quarters, everything is rebuilt.
        """
        self._sorted_stale = True
        self._flips_since_build += count
        if self._flips_since_build > self._flip_budget:
            self.rebuild(read_count=False, incremental=True)

    def edges_unique(self) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
        """
        ``edges.edges_unique(faces, n_vertices=n_vertices)`` for the face buffer as it is now.

        A build already radix-sorted every corner's undirected edge key against the same radix
        ``edges_unique`` packs with, so the unique edges are the runs of that sort: a run-start
        mark, a scan and one emit give the identical ascending-key rows and corner map, without
        re-hashing the whole mesh. The sort goes stale when a flip commits, and is redone first
        then. Reuses the pair-start scratch, which only a build reads. Requires a build.
        """
        if self._sorted_stale:
            # Only the sort is stale: the rows, the halfedge maps and the key set were kept current.
            self._sort()
        n = self._n_corners
        _launch.launch(
            kernel_grouping.MARK_SORTED_RUN_STARTS[wp.uint64],
            dim=n,
            inputs=[self._keys, self._starts],
            device=self._device,
        )
        _launch.array_scan(self._starts, out_array=self._ranks, inclusive=True)
        # The edge count sizes the returned table.
        n_edges = int(read_scalar(self._ranks_tail, 0))
        unique_edges = twt.empty_2d((n_edges, 2), wp.int32, device=self._device)
        inverse = _launch.empty(n, dtype=wp.int32, device=self._device)
        _launch.launch(
            kernel_edges.emit_sorted_unique_edges,
            dim=n,
            inputs=[self._faces, self._keys, self._order, self._ranks, unique_edges, inverse],
            device=self._device,
        )
        return unique_edges, inverse

    def _allocate_rows(self, m: int) -> None:
        """(Re)allocate the ``m``-row tables and the hashed edge-claim table sized from them."""
        self._rows = m
        self.adjacency = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.adjacency_edges = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.unshared = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.flip = _launch.empty(m, dtype=wp.bool, device=self._device)
        self.quad = twt.empty_2d((m, 4), wp.int32, device=self._device)
        table = 1
        while table < 4 * m + 1:
            table <<= 1
        self.edge_claim = _launch.empty(table, dtype=wp.uint64, device=self._device)
        self.edge_claim_mask = table - 1
        self.claims_armed = False


def cluster_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    voxel_size: float | None = None,
    contraction: Literal["average", "closest"] = "average",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Decimate by snapping vertices to a uniform voxel grid and welding each cell to one vertex.

    The one decimation scheme that is *naturally* parallel: there is no priority queue and no
    sequential dependence anywhere, so the whole thing is a handful of kernel launches whatever the
    mesh size. Every vertex is binned into a cell of width ``voxel_size``, each occupied cell
    becomes a single output vertex, faces are remapped onto those, and the faces that collapsed
    (two or three corners in the same cell) or duplicated are dropped.

    Ports MeshLab's ``meshing_decimation_clustering`` and Open3D's ``simplify_vertex_clustering``;
    the grid is anchored half a cell below the bounding box, which is Open3D's convention, so both
    libraries produce the same cell assignment for the same ``voxel_size``. The vertex output alone
    is also MeshLab's ``generate_sampling_clustered_vertex``.

    !!! warning "This does not preserve topology"
        Two sheets of the surface that pass within ``voxel_size`` of each other get welded
        together, and a thin feature narrower than a cell disappears. That is the *point* of the
        algorithm — it is a resampling, not a simplification — but it means the result can be
        non-manifold even when the input is not. Use
        [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] when the topology matters and
        [`quadric_decimate`][triwarp.remesh.quadric_decimate] when a face budget does.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    voxel_size
        Cell width. Defaults to ``1 %`` of the bounding-box diagonal, matching MeshLab's
        ``threshold`` default of ``1 %``. Larger cells decimate harder.
    contraction
        How each cell picks its output position:

        - ``"average"`` (default) — the mean of the cell's vertices, which is Open3D's
          ``SimplificationContraction.Average``. Smooths slightly and cannot land off the input's
          convex hull.
        - ``"closest"`` — the input vertex nearest the cell centre, which is MeshLab's
          ``'Closest to center'`` sampling. Keeps every output vertex *on* the input surface, so it
          is the right choice when the positions must stay exact (ties break to the lowest index).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` one position per occupied cell that still carries a face, compacted
        from index zero.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat triangle index buffer, free of collapsed and duplicate faces.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, or ``contraction`` is not one of the two names.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`quadric_decimate`][triwarp.remesh.quadric_decimate]
        The other way to simplify: a face budget and a quadric error metric, rather than a voxel
        size.
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`triwarp.repair.remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]
    [`triwarp.grouping.unique_faces`][triwarp.grouping.unique_faces]
    [`triwarp.voxels.voxel_down_sample`][triwarp.voxels.voxel_down_sample]

    Notes
    -----
    Cells whose every face collapsed are dropped from the output, where Open3D keeps them as
    unreferenced vertices. So the face counts agree exactly and the vertex counts can differ by the
    number of such cells — usually zero, and never in a way that changes the surface.

    The binning is [`triwarp.voxels.cell_indices`][triwarp.voxels.cell_indices]' cell assignment.
    Output vertices are numbered in the order of their packed cell keys -- the order
    [`triwarp.grouping.unique_1d`][triwarp.grouping.unique_1d] would sort them in -- rather than a
    NanoVDB grid's leaf-major order, and the faces come out in
    [`triwarp.grouping.unique_faces`][triwarp.grouping.unique_faces]' order over them.
    """
    require_same_device(vertices=vertices, faces=faces)
    if contraction not in ("average", "closest"):
        raise ValueError(f"contraction must be 'average' or 'closest', got {contraction!r}")

    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_vertices == 0 or n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    # One definition of "the default voxel grid for these points", shared with ``triwarp.voxels``
    # so the two modules cannot drift on the cell size or on Open3D's half-cell anchor.
    # The origin is always derived here, so every cell coordinate is non-negative and below the
    # bound the grid reports, which is the radix the cell keys are packed in.
    voxel_size, origin, cell_bound = tw.voxels.resolve_voxel_grid(
        vertices, voxel_size, caller="cluster_decimate", return_cell_bound=True
    )
    inverse_size = wp.float32(1.0 / voxel_size)
    bound = wp.uint64(cell_bound)
    cell_mask = kernel_grouping.hash_table_mask(n_vertices)
    face_mask = kernel_grouping.hash_table_mask(n_faces)
    n_cells = cell_mask + 1
    # Three ``-1``-filled tables in one allocation: the cell table (a vertex index per cluster),
    # each cluster's state (unreferenced, kept, then its output vertex) and the face table (the
    # first occurrence of each distinct surviving face).
    tables = _launch.full(2 * n_cells + face_mask + 1, -1, dtype=wp.int32, device=device)
    cell_table = twt.as_dense(tables[:n_cells])
    cell_rank = twt.as_dense(tables[n_cells : 2 * n_cells])
    face_table = twt.as_dense(tables[2 * n_cells :])
    # Every vertex's cluster, then the kept clusters (sized for the sort that orders them, at most
    # one per vertex), then the distinct surviving faces' table slots.
    scratch = _launch.empty(3 * n_vertices + n_faces, dtype=wp.int32, device=device)
    vertex_cell = twt.as_dense(scratch[:n_vertices])
    kept_cells = twt.as_dense(scratch[n_vertices : 3 * n_vertices])
    face_slots = twt.as_dense(scratch[3 * n_vertices :])
    # The kept clusters' cell keys, at the sort's double width.
    sort_keys = _launch.empty(2 * n_vertices, dtype=wp.uint64, device=device)
    counters = _launch.zeros(2, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.cluster_insert_vertices,
        dim=n_vertices,
        inputs=[vertices, origin, inverse_size, bound, wp.int32(cell_mask), cell_table],
        outputs=[vertex_cell],
        device=device,
    )
    # Welding can map two distinct input faces onto the same triple, which would leave a duplicated
    # face rather than a manifold one, so the face dedup is part of the algorithm rather than
    # polish. It happens here, before any count is read, in a table of surviving faces only: at a
    # coarse voxel size nearly every face collapses, and a collapsed face never enters it.
    _launch.launch(
        kernel_remesh.cluster_mark_faces,
        dim=n_faces,
        inputs=[
            faces,
            vertex_cell,
            vertices,
            origin,
            inverse_size,
            bound,
            cell_table,
            wp.int32(face_mask),
            cell_rank,
            face_table,
            counters,
        ],
        outputs=[sort_keys, kept_cells, face_slots],
        device=device,
    )
    # The two counts size both outputs, so they come back together, in one copy; nothing after
    # this reads the host.
    n_kept, n_unique = read_values(counters, 0, 2)
    if n_kept == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
        )
    # The output vertices are the kept clusters in cell-key order, which is the order the sorted
    # unique cell keys of a whole-mesh dedup would give them. The keys are distinct, so the
    # permutation does not depend on the order the claims arrived in.
    _launch.radix_sort_pairs(
        sort_keys,
        kept_cells,
        count=n_kept,
        end_bit=max(1, (min(cell_bound**3, 1 << 64) - 1).bit_length()),
    )
    _launch.launch(
        kernel_remesh.cluster_rank_cells, dim=n_kept, inputs=[kept_cells, cell_rank], device=device
    )
    kept_vertices = _cluster_positions(
        vertices, vertex_cell, cell_rank, origin, voxel_size, contraction, n_kept
    )
    return kept_vertices, _cluster_faces(
        faces,
        vertex_cell,
        cell_rank,
        face_table,
        face_slots,
        n_kept,
        n_unique,
        sort_keys,
        kept_cells,
    )


def _cluster_positions(
    vertices: wp.array[wp.vec3],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    origin: wp.vec3,
    voxel_size: float,
    contraction: Literal["average", "closest"],
    n_kept: int,
) -> wp.array[wp.vec3]:
    """One representative position per kept cluster, by cell mean or by nearest-to-centre."""
    device = vertices.device
    n_vertices = vertices.size
    if contraction == "average":
        sums = _launch.zeros(n_kept, dtype=wp.vec3, device=device)
        counts = _launch.zeros(n_kept, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_remesh.cluster_accumulate,
            dim=n_vertices,
            inputs=[vertex_cell, cell_rank, vertices],
            outputs=[sums, counts],
            device=device,
        )
        out = _launch.empty(n_kept, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_remesh.cluster_means,
            dim=n_kept,
            inputs=[sums, counts],
            outputs=[out],
            device=device,
        )
        return out

    closest = _launch.full(n_kept, UINT64_MAX, dtype=wp.uint64, device=device)
    _launch.launch(
        kernel_remesh.cluster_pick_closest,
        dim=n_vertices,
        inputs=[vertex_cell, cell_rank, vertices, origin, wp.float32(voxel_size)],
        outputs=[closest],
        device=device,
    )
    out = _launch.empty(n_kept, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_remesh.cluster_gather_representatives,
        dim=n_kept,
        inputs=[vertices, closest],
        outputs=[out],
        device=device,
    )
    return out


def _cluster_faces(
    faces: wp.array[wp.int32],
    vertex_cell: wp.array[wp.int32],
    cell_rank: wp.array[wp.int32],
    face_table: wp.array[wp.int32],
    face_slots: wp.array[wp.int32],
    n_kept: int,
    n_unique: int,
    sort_keys: wp.array[wp.uint64],
    kept_cells: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """
    Distinct surviving faces over the kept clusters, in ``grouping.unique_faces``' order.

    That is the order of the sorted output-vertex triple packed in radix ``n_kept``, each face in
    its first occurrence's winding. The kept-cluster sort's buffers are free by now and serve the
    face sort whenever they are wide enough, which they are unless almost nothing merged.
    """
    device = faces.device
    if 2 * n_unique > sort_keys.size:
        sort_keys = _launch.empty(2 * n_unique, dtype=wp.uint64, device=device)
        kept_cells = _launch.empty(2 * n_unique, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.cluster_face_keys,
        dim=n_unique,
        inputs=[faces, vertex_cell, cell_rank, face_table, face_slots, wp.uint64(n_kept)],
        outputs=[sort_keys, kept_cells],
        device=device,
    )
    # Three sorted indices below ``n_kept`` pack below ``n_kept ** 3``.
    _launch.radix_sort_pairs(
        sort_keys,
        kept_cells,
        count=n_unique,
        end_bit=max(1, (min(n_kept**3, 1 << 64) - 1).bit_length()),
    )
    out = _launch.empty(3 * n_unique, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.cluster_emit_faces,
        dim=n_unique,
        inputs=[faces, vertex_cell, cell_rank, kept_cells],
        outputs=[out],
        device=device,
    )
    return out


def _compact_remapped_faces(
    faces: wp.array[wp.int32], remap: wp.array[wp.int32], n_targets: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    Faces pushed through a vertex remap, with the collapsed ones dropped and the targets compacted.

    Returns ``(ranks, compacted_faces, n_kept)``. A face survives when its three remapped corners
    are distinct; the targets kept are those a surviving face names, in target order, and
    ``compacted_faces`` holds the surviving faces in input order renumbered onto them -- a
    face-mask ``submesh`` followed by ``repair.remove_unreferenced_vertices``. ``ranks`` is the
    ``n_targets + 1`` scan ``kernels/array.scanned_slot`` reads each target's slot from.

    One marking launch, one scan over the face flags and the target marks together, one compaction
    that also publishes both counts, and one read of them. The compacted buffer is sized for every
    face so that launch can publish the counts before the host knows them; the result is its
    leading view.
    """
    device = faces.device
    n_faces = faces.size // 3
    # Behind a leading zero: the face flags, then the target marks, so one inclusive scan of the
    # tail makes each half an exclusive scan ending in its running total.
    scan = _launch.zeros(1 + n_faces + n_targets, dtype=wp.int32, device=device)
    marks = scan[1:]
    _launch.launch(
        kernel_remesh.mark_surviving_faces, dim=n_faces, inputs=[faces, remap, marks], device=device
    )
    _launch.array_scan(marks, marks, inclusive=True)
    ranks = twt.as_dense(scan[n_faces:])
    compacted = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    totals = _launch.empty(2, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.compact_surviving_faces,
        dim=n_faces,
        inputs=[faces, remap, scan[: n_faces + 1], ranks, compacted, totals],
        device=device,
    )
    # Both counts size what the caller allocates next, so they come back -- together, in one copy.
    n_kept, n_surviving = (int(count) for count in totals.numpy())
    if n_surviving == 0:
        return ranks, _launch.empty(0, dtype=wp.int32, device=device), n_kept
    return ranks, twt.as_dense(compacted[: 3 * n_surviving]), n_kept


@overload
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = ...,
    target_ratio: float | None = ...,
    feature_angle: float = ...,
    max_iter: int = ...,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = ...,
    target_ratio: float | None = ...,
    feature_angle: float = ...,
    max_iter: int = ...,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]: ...
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = None,
    target_ratio: float | None = None,
    feature_angle: float = 30.0,
    max_iter: int = 100,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """

    Simplify to a target face count by quadric-error edge collapses (Garland-Heckbert).

    The decimation to reach for when the requirement is a **face budget** rather than an edge
    length: [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] targets a length and
    [`cluster_decimate`][triwarp.remesh.cluster_decimate] a voxel size, and neither lets a caller
    ask for "this mesh at 10 % of its triangles". This does, and it is the method every comparable
    library exposes for the purpose (MeshLab's ``meshing_decimation_quadric_edge_collapse``,
    ``igl.decimate``, Open3D's ``simplify_quadric_decimation``).

    Each vertex accumulates the area-weighted plane quadrics of its incident faces; the cost of
    collapsing an edge is the residual of the summed quadric at its own minimizer, which is also
    where the surviving vertex is placed. Cheap collapses are the ones that barely move the surface,
    so the flat regions go first and the features last — the property that makes this the standard
    method.


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device. Never mutated.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    target_faces
        Desired face count. Mutually exclusive with ``target_ratio``; exactly one must be given.
        A value at or above the input count collapses nothing, but still returns an independent
        copy with the same compaction and provenance every other target gets — see Returns.
    target_ratio
        Desired face count as a fraction of the input's, so ``0.1`` is MeshLab's usual "10 %". Must
        be in ``(0, 1]``.
    feature_angle
        Dihedral angle in **degrees** above which an edge is a feature. Feature and boundary
        structure is preserved exactly as in
        [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]: corners are frozen, a crease vertex
        only collapses along its own feature, and a crease is never dragged off it.
    max_iter
        Cap on collapse passes. Each pass commits a conflict-free independent set, so a large
        reduction needs many; the loop also stops early once the target is met or a pass commits
        nothing.
    return_index
        If ``True``, also return the two provenance maps below, which is how a per-vertex or
        per-face attribute survives the decimation.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` simplified vertex positions on ``vertices.device``, compacted from
        index zero.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat triangle index buffer. The count is a best effort at
        ``target_faces`` and is **not** bounded by it: a mesh whose remaining edges all fail the
        link condition or the normal-flip guard stops above the target — see Notes. Read the
        returned count rather than assuming it.
    vertex_index : wp.array[wp.int32]
        ``(n_vertices,)`` **output** vertex each input vertex ended up in, or ``-1`` for an input
        vertex that survives in no output face (one that was already unreferenced); only when
        ``return_index`` is ``True``. Many-to-one, since that is what a collapse is, so it is
        the direction a scatter or a segmented reduction wants.
    face_index : wp.array[wp.int32]
        ``(n_out_faces,)`` **input** face each output face came from, only when ``return_index`` is
        ``True`` -- the same output-to-input direction
        [`split_edges`][triwarp.remesh.split_edges] uses, so a per-face attribute follows through
        ``tw.array.gather``. A collapse only deletes faces and never creates one, so every output
        face has exactly one source.

    Raises
    ------
    ValueError
        If neither or both of ``target_faces`` / ``target_ratio`` is given, ``target_faces`` is
        negative, or ``target_ratio`` is outside ``(0, 1]``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`cluster_decimate`][triwarp.remesh.cluster_decimate]
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`triwarp.repair.collapse_small_triangles`][triwarp.repair.collapse_small_triangles]

    Notes
    -----
    **This is a batched-parallel greedy method, not the textbook serial one, and the difference is
    visible in the output.** Textbook QEM pops one edge at a time from a global priority queue,
    which is inherently sequential. Here each pass scores every edge, ranks the candidates by cost,
    and commits the cheapest *independent set* of them -- two collapses may commit together only if
    their closed 1-rings are disjoint. So the sequence of collapses differs from a serial run's and
    the resulting triangulation is not the same mesh, even though both are driven by the same
    metric. Compare the two by deviation from the input rather than by equality. In exchange the
    *quality* is competitive: committing an independent set spreads the error over the surface where
    draining a priority queue concentrates it, and a max-norm error measure rewards that.

    A pass commits **several independent sets against one scoring**, not one. A single hashed-key
    round takes a small fraction of the candidates, because each winner locks the closed 1-rings of
    both its endpoints, and rebuilding the geometry between rounds is comparatively expensive. So
    the pass retires only the candidates the previous round's commits invalidated -- those whose
    closed 1-rings touch a collapsed neighbourhood -- and runs another round until one finds nothing
    new. That round loop runs **entirely on device**, as one ``wp.capture_while`` graph.

    The per-pass rebuild cost is dominated by the number of launches it issues rather than by the
    mesh size, which is why one edge grouping answers the feature classification, the incidence
    and both adjacencies, and why ``_DecimationBuffers`` replays the whole rebuild as one captured
    graph with fixed-width buffers and live sizes carried in a device array. Read that class before
    changing anything here. ``return_index`` costs nothing when off and next to nothing when on: the
    two provenance maps are folded per pass by two launches and one copy, and the branch is
    evaluated when the pass is *issued*, so the captured graph does not contain it.

    Four consequences to plan around:

    - **The target is reached exactly whenever it is reachable, and it is ``feature_angle`` that
      decides
      whether it is.** A pass is budgeted at half the remaining surplus (an interior collapse
      removes two faces), shared across its rounds, and the loop stops early when a pass can commit
      nothing. What stops it is almost always the *feature* rule rather than the link condition or
      the normal-flip guard: a surface's own dihedral angles grow as it is coarsened, so past some
      face count every edge of a smooth mesh is sharper than ``feature_angle``, every vertex becomes
      a frozen corner, and no collapse is legal at any ``max_iter``. **That floor is the parameter
      working, not a limitation to route around** -- it is the same rule that keeps a cylinder's rim
      and a box's creases intact. Raising ``feature_angle`` lowers it; at 180 degrees nothing is a
      feature and the target is reached. Check the returned face count if it matters.
    - Every collapse is also checked against a **normal-flip guard**: an incident face whose normal
      would turn too far vetoes it. That is what keeps the output free of the inverted,
      self-intersecting triangles an unguarded quadric method produces at high reduction ratios. It
      is **not** usually what stops a decimation short, and is deliberately not exposed as a keyword
      -- see ``COLLAPSE_MIN_NORMAL_DOT`` in ``kernels/remesh.py``, which records the veto census
      this claim rests on.
    - The independent set is chosen under a **hashed** lock key rather than by cost rank. That looks
      like a detail and is not: on a structured mesh both the edge index and the quadric cost are
      spatially monotone fields, and a monotone key has one local minimum, so either would commit
      only a single collapse per pass. See ``scramble_index`` in ``kernels/remesh.py``.
    - **The output is not bit-reproducible on a mesh with tied costs.** The vertex-face incidence
      CSR is
      built by an atomic counting scatter, so a row's order varies run to run; where two candidate
      edges tie on cost, which one the sort keeps varies with it. On a mesh with many ties this can
      move the two-sided Hausdorff distance noticeably between otherwise identical runs, so **treat
      the max-norm as a band, not a value** -- the mean deviation is far more stable.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    target = _resolve_decimation_target(target_faces, target_ratio, n_faces)

    if n_faces == 0 or target >= n_faces:
        # Nothing to collapse, but the *output* contract still holds: vertices compacted from index
        # zero, and ``vertex_index`` reporting -1 for an input vertex no output face references.
        # Returning the buffer verbatim with an identity map would make the shape of the answer
        # depend on whether the target happened to clear the input's face count -- a caller sweeping
        # a ratio would see an already-unreferenced vertex appear and disappear across that
        # boundary. It is the answer ``_DecimationBuffers``' per-pass compaction gives: survivors
        # in order and the referenced vertices kept in index order.
        kept_vertices, kept_faces, remap = tw.repair.remove_unreferenced_vertices(vertices, faces)
        if not return_index:
            return kept_vertices, kept_faces
        return kept_vertices, kept_faces, remap, tw.array.arange(n_faces, device=faces.device)

    buffers = _DecimationBuffers(
        vertices, faces, target, wp.float32(math.radians(feature_angle)), track_index=return_index
    )
    for _ in range(max_iter):
        if not buffers.run_pass():
            break
    out_vertices, out_faces, face_source = buffers.result()
    if not return_index:
        return out_vertices, out_faces
    return out_vertices, out_faces, buffers.vertex_index, face_source


class _DecimationBuffers:
    """
    The whole working set of [`quadric_decimate`][triwarp.remesh.quadric_decimate], allocated once.

    A decimation pass is dominated by the launches and wrapper calls that rebuild the geometry
    rather than by the device work itself, and that cost barely tracks the mesh size. So the pass is
    issued once against fixed-capacity buffers sized at the pass-0 width, and replayed as a captured
    CUDA graph for every pass after, which pays for no Python at all on replay.

    Nothing in a pass reads a count back: every size lives in ``state`` on the device, and each
    kernel stops at the live prefix it reads from there. Nor does a pass allocate, beyond the two
    radix sorts' own scratch -- every buffer is allocated here and the scans are
    [`_ExclusiveScan`][triwarp.remesh._ExclusiveScan]s -- because a replayed graph pays for every
    node it holds, and the round loop's conditional body may not allocate at all.
    ``kernels/remesh.py`` lists the pass's launches in order at ``begin_decimation_pass``.

    Padding is carried by a **dummy vertex** at index ``n_vertices`` that every padded face corner
    points at, and the edge buffers keep a **dummy edge slot** at index ``n_edges``; see the kernel
    section in ``kernels/remesh.py`` for why that is enough.

    Attributes
    ----------
    state : wp.array[wp.int32]
        ``[n_faces, n_vertices, n_edges, commits]``, the live prefix lengths and the pass's collapse
        count. The one array the host reads, once per pass, to decide whether to run another.
    """

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        target: int,
        feature: wp.float32,
        *,
        track_index: bool = False,
    ) -> None:
        """Allocate at the input's size, which bounds every later pass, and seed the live counts."""
        device = cast("wp.Device", faces.device)
        self._device = device
        self._target = target
        self._feature = feature
        self._track_index = track_index
        self._graph = None
        self._passes = 0
        self._retain: list[object] = []
        self.n_faces = faces.size // 3
        self.n_vertices = vertices.size
        self.n_corners = 3 * self.n_faces
        # The corner keys pack an edge as ``min + max * base``, so every real key is below
        # ``base ** 2`` and the sort need only order that many low bits. The padding sentinel is
        # all ones, so its truncation still sorts at or past every real key -- and a tie can only be
        # with a real key of lower corner index, which the stable sort keeps first either way.
        base = self.n_vertices + 1
        self._key_bits = max(1, (base * base - 1).bit_length())

        # The dummy vertex lives one past the capacity, so every per-vertex buffer is one longer.
        v_cap = self.n_vertices + 1
        self.vertices = _launch.zeros(v_cap, dtype=wp.vec3, device=device)
        _launch.copy(self.vertices, vertices, count=self.n_vertices)
        self._faces_store = _launch.empty(self.n_corners, dtype=wp.int32, device=device)
        _launch.copy(self._faces_store, faces, count=self.n_corners)
        # Allocated holding its seed rather than zeroed and then assigned: the zeroing is
        # discarded and the assign is a second upload of the same bytes. ``[faces, vertices,
        # edges, commits]``: the last slot is the pass's collapse count, which the pass kernels
        # reach through the ``_count`` view, so the one readback a pass ends with answers both
        # "did it commit anything" and the next pass's size test.
        self.state = _launch.array(
            [self.n_faces, self.n_vertices, 0, 0], dtype=wp.int32, device=device
        )
        self._host_state = None
        self._count = self.state[3:4]

        n = self.n_corners
        self._keys_store = _launch.empty(2 * n, dtype=wp.uint64, device=device)
        self._order_store = _launch.empty(2 * n, dtype=wp.int32, device=device)
        self._starts_store = _launch.empty(n, dtype=wp.int32, device=device)
        self._corner_slots_store = _launch.empty(n, dtype=wp.int32, device=device)
        self._remapped_store = _launch.empty(n, dtype=wp.int32, device=device)
        # Per-vertex state the pass accumulates into or starts from; ``begin_decimation_pass``
        # resets all of it.
        self._positions = _launch.empty(v_cap, dtype=wp.vec3, device=device)
        self._collapse_remap = _launch.empty(v_cap, dtype=wp.int32, device=device)
        self._feature_count = _launch.empty(v_cap, dtype=wp.int32, device=device)
        self._quadrics = _launch.empty(v_cap, dtype=wp.mat44d, device=device)
        self._locked = _launch.empty(v_cap, dtype=wp.int32, device=device)
        self._min_key = _launch.empty(v_cap, dtype=wp.int64, device=device)
        # Vertex-vertex then vertex-face incidence counts, one row per vertex each and a closing
        # zero, so one exclusive scan yields both CSRs' offset tables (see ``kernels/remesh.py``).
        self._adjacency_counts = _launch.empty(2 * v_cap + 1, dtype=wp.int32, device=device)
        self._adjacency_scan = _ExclusiveScan(2 * v_cap + 1, self._device)
        self._adjacency_offsets = _launch.empty(2 * v_cap + 1, dtype=wp.int32, device=device)
        self._round_state = _launch.empty(
            kernel_remesh.COLLAPSE_STATE_SIZE, dtype=wp.int32, device=device
        )

        # Provenance, only when a caller asked for it: one entry per *input* vertex composed pass by
        # pass, and a column beside the face buffer compacted with it (both inside
        # ``compact_decimation_pass``). Both are fixed width -- the vertex map by construction, the
        # face column because the face buffer is -- so tracking them does not stop the pass being
        # captured; the scratch exists because the compaction cannot read and write one buffer.
        self.vertex_index = (
            tw.array.arange(self.n_vertices, device=device)
            if track_index
            else _launch.empty(0, dtype=wp.int32, device=device)
        )
        self._face_source_store = (
            tw.array.arange(self.n_faces, device=device)
            if track_index
            else _launch.empty(0, dtype=wp.int32, device=device)
        )
        self._face_source_scratch_store = (
            _launch.empty(self.n_faces, dtype=wp.int32, device=device)
            if track_index
            else _launch.empty(0, dtype=wp.int32, device=device)
        )
        self.face_source = self._face_source_store
        self._face_source_scratch = self._face_source_scratch_store

        # The capacity is the structural bound until the first grouping measures the true count.
        self.n_edges = 0
        self._allocate_face_buffers(self.n_faces)
        self._allocate_edge_buffers(self.n_corners)

    def _allocate_face_buffers(self, faces: int) -> None:
        """
        Narrow every buffer indexed by face or corner to a capacity of ``faces``.

        The live faces are always a prefix of the face buffer and every row past them is the dummy
        triangle, so a narrower capacity is a prefix view of each corner buffer; only the two
        layouts that place something *after* the faces -- the compaction's flags and the scans
        sized by them -- are allocated anew.
        """
        device = self._device
        self.n_faces = faces
        self.n_corners = 3 * faces
        n = self.n_corners
        self.faces = twt.as_dense(self._faces_store[:n])
        self._keys = twt.as_dense(self._keys_store[: 2 * n])
        self._order = twt.as_dense(self._order_store[: 2 * n])
        self._starts = twt.as_dense(self._starts_store[:n])
        self._corner_slots = twt.as_dense(self._corner_slots_store[:n])
        self._remapped = twt.as_dense(self._remapped_store[:n])
        self._start_scan = _ExclusiveScan(n, device)
        # The compaction's keep flags, faces first and vertices after, and their one scan.
        self._compact_flags = _launch.empty(faces + self.n_vertices, dtype=wp.int32, device=device)
        self._compact_scan = _ExclusiveScan(faces + self.n_vertices, device)
        if self._track_index:
            self.face_source = twt.as_dense(self._face_source_store[:faces])
            self._face_source_scratch = twt.as_dense(self._face_source_scratch_store[:faces])

    def _allocate_edge_buffers(self, edges: int) -> None:
        """
        (Re)allocate every buffer indexed by unique edge at a capacity of ``edges``.

        An edge collapse removes at least three undirected edges and adds none, so a count one pass
        measured bounds every later one; until the first pass has run, ``3 * n_faces`` does.
        """
        device = self._device
        self.n_edges = edges
        self._unique_edges = twt.empty_2d((edges, 2), wp.int32, device=device)
        # One row longer than the edge capacity: the dummy slot an overflowing corner lands in.
        self._edge_face_count = _launch.empty(edges + 1, dtype=wp.int32, device=device)
        self._edge_faces = twt.empty_2d((edges + 1, 2), wp.int32, device=device)
        self._edge_slots = _launch.empty(edges, dtype=wp.vec2i, device=device)
        # Vertex-vertex entries (two per edge) followed by vertex-face entries (one per corner).
        self._adjacency = _launch.empty(2 * edges + self.n_corners, dtype=wp.int32, device=device)
        self._candidates = _launch.empty(edges, dtype=wp.int32, device=device)
        self._survivor = _launch.empty(edges, dtype=wp.int32, device=device)
        self._removed = _launch.empty(edges, dtype=wp.int32, device=device)
        self._target_pos = _launch.empty(edges, dtype=wp.vec3, device=device)
        self._cost = _launch.empty(edges, dtype=wp.float32, device=device)
        self._cost_rank = _launch.empty(edges, dtype=wp.int32, device=device)
        # ``radix_sort_pairs`` wants double-width key and payload buffers.
        self._sort_keys = _launch.empty(2 * edges, dtype=wp.float32, device=device)
        self._sort_order = _launch.empty(2 * edges, dtype=wp.int32, device=device)
        # Two bitmasks of the round's winners, in cost order and in edge order, one word per 32.
        self._winner_words = _launch.empty(2 * (-(-edges // 32)), dtype=wp.uint32, device=device)
        self._winner_scan = _ExclusiveScan(self._winner_words.size, device)

    def run_pass(self) -> bool:
        """
        Run one decimation pass, and report whether another is worth running.

        The first pass is issued -- it is the one that measures the true edge count, which the
        allocation could only bound structurally at ``3 * n_faces`` -- and the second is captured
        at that narrower width and replayed by every pass after it. The one host readback per pass
        is here rather than in the pass body, taken after it, and covers the "nothing legal left
        to collapse" exit of this pass and the "target reached" exit of the next.

        Without a conditional-graph device there is nothing to replay, so the width is re-tightened
        every pass instead and the pass is issued: that keeps the CPU path tracking the live mesh
        rather than paying the pass-0 width forever.
        """
        counts = self._host_state if self._host_state is not None else self.state.numpy()
        if int(counts[0]) <= self._target or int(counts[1]) == 0:
            return False
        live = int(counts[0])
        if (
            self._graph is not None
            and live <= _DECIMATION_RECAPTURE * self.n_faces
            and self.n_faces - live >= _DECIMATION_RECAPTURE_FACES
        ):
            # Every per-face, per-corner and per-edge launch and both sorts run at the recorded
            # width, which the live mesh has by now fallen well below: record the pass again at
            # the live width.
            self._graph = None
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._tighten(int(counts[0]), int(counts[2]))
            if self._passes > 0 and self._device.is_cuda and wp.is_conditional_graph_supported():
                with wp.ScopedCapture(self._device) as capture:
                    self._issue_pass()
                self._graph = capture.graph
                assert self._graph is not None
                wp.capture_launch(self._graph)
            else:
                self._issue_pass()
        self._passes += 1
        self._host_state = self.state.numpy()
        return int(self._host_state[3]) != 0

    def _tighten(self, faces: int, edges: int) -> None:
        """
        Narrow the face- and edge-indexed buffers to the bounds the previous pass measured.

        A collapse removes faces and at least three undirected edges and adds neither, so the
        previous pass's counts bound every later one. The allocation can only bound the
        unique-edge count by ``3 * n_faces``, which is 2x the true value on a closed mesh -- and
        every per-edge kernel, the candidate cost sort and each round's scan run at that width.
        """
        if 0 < faces < self.n_faces:
            self._allocate_face_buffers(faces)
        if 0 < edges < self.n_edges:
            self._allocate_edge_buffers(edges)

    def result(self) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
        """Copy the live prefixes out of the fixed buffers, which is the only place a size leaks."""
        counts = self._host_state if self._host_state is not None else self.state.numpy()
        n_faces, n_vertices = int(counts[0]), int(counts[1])
        vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=self._device)
        faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=self._device)
        if n_vertices > 0:
            _launch.copy(vertices, self.vertices, count=n_vertices)
        if n_faces > 0:
            _launch.copy(faces, self.faces, count=3 * n_faces)
        face_source = _launch.empty(0, dtype=wp.int32, device=self._device)
        if self._track_index:
            face_source = _launch.empty(n_faces, dtype=wp.int32, device=self._device)
            if n_faces > 0:
                _launch.copy(face_source, self.face_source, count=n_faces)
        return vertices, faces, face_source

    def _issue_pass(self) -> None:
        """
        Issue every launch of one pass, in order. Called once, when the graph is captured.

        The views this creates are kept in ``self._retain``. ``warp``'s ``Graph`` references the
        *modules* a captured launch needs but **not its arrays**; the buffers themselves belong to
        this object, and the list costs only a few object references.
        """
        device = self._device
        v_cap = self.n_vertices + 1
        _launch.launch(
            kernel_remesh.begin_decimation_pass,
            dim=max(self.n_faces, self.n_edges + 1, 2 * v_cap + 1),
            inputs=[
                self.faces,
                self.vertices,
                self.state,
                wp.uint64(v_cap),
                self._keys,
                self._order,
                self._edge_face_count,
                self._adjacency_counts,
                self._feature_count,
                self._quadrics,
                self._locked,
                self._min_key,
                self._collapse_remap,
                self._positions,
                self._compact_flags[self.n_faces :],
                self._round_state,
                self.state,
            ],
            device=device,
        )
        offsets = self._group_edges()
        csr_offsets = twt.as_dense(offsets[: v_cap + 1])
        face_offsets = offsets[v_cap:]
        _launch.launch(
            kernel_remesh.quadric_collapse_candidates,
            dim=self.n_edges,
            inputs=[
                self._unique_edges,
                self.vertices,
                self.faces,
                self._quadrics,
                self._feature_count,
                self._edge_face_count,
                csr_offsets,
                self._adjacency,
                face_offsets,
                self._adjacency,
                self.state,
                self._candidates,
                self._removed,
                self._target_pos,
                self._cost,
                self._sort_keys,
                self._sort_order,
            ],
            device=device,
        )

        # Two-stage selection, and both stages matter.
        #
        # Stage one narrows the field to the cheapest *half* of the candidate edges, which is what
        # makes the method quadric-driven. Stage two picks a maximal independent set from those,
        # locking each winner's closed 2-ring under a **hashed** key -- see ``scramble_index`` for
        # why the obvious keys (edge index, or the cost itself) both collapse to one winner a pass
        # on a structured mesh.
        _launch.radix_sort_pairs(self._sort_keys, self._sort_order, count=self.n_edges)
        _launch.launch(
            kernel_remesh.drop_past_half,
            dim=self.n_edges,
            inputs=[self._sort_order, self.state, self._cost_rank, self._candidates],
            device=device,
        )
        self._run_collapse_rounds(csr_offsets)
        self._compact()
        self._retain = [offsets, csr_offsets, face_offsets]

    def _group_edges(self) -> wp.array[wp.int32]:
        """
        Group the live face corners into unique edges and build the pass's adjacency, readback-free.

        The same edges as [`_edge_incidence`][triwarp.remesh._edge_incidence] -- and in the same
        ascending-key order, which the lock keys depend on -- from one radix sort instead of
        ``edges_unique``'s hash table, a compaction scan and a second sort; plus, from the same
        launches, the per-vertex feature counts, the vertex-vertex and vertex-face CSRs and the
        vertex quadrics. Returns the shared offset table, whose first ``n_vertices + 2`` entries
        index the vertex-vertex rows and whose remainder indexes the vertex-face rows.
        """
        device = self._device
        n = self.n_corners
        v_cap = self.n_vertices + 1
        _launch.radix_sort_pairs(self._keys, self._order, count=n, end_bit=self._key_bits)
        _launch.launch(
            kernel_remesh.mark_unique_edge_starts,
            dim=n,
            inputs=[self._keys, self.state, self._starts],
            device=device,
        )
        self._start_scan.launch(self._starts)
        _launch.launch(
            kernel_remesh.emit_pass_edges,
            dim=n,
            inputs=[
                self.faces,
                self._order,
                self._starts,
                self._start_scan.prefix,
                self._start_scan.chunk_offsets,
                self.state,
                wp.int32(self.n_edges),
                wp.int32(v_cap),
                self._unique_edges,
                self._edge_face_count,
                self._edge_faces,
                self._adjacency_counts,
                self._corner_slots,
                self.state,
            ],
            device=device,
        )
        _launch.launch(
            kernel_remesh.count_pass_edges,
            dim=self.n_edges,
            inputs=[
                self.vertices,
                self.faces,
                self._unique_edges,
                self._edge_face_count,
                self._edge_faces,
                self.state,
                self._feature,
                self._feature_count,
                self._adjacency_counts,
                self._edge_slots,
            ],
            device=device,
        )
        self._adjacency_scan.launch(self._adjacency_counts)
        _launch.launch(
            kernel_remesh.scatter_pass_adjacency,
            dim=max(n, 2 * v_cap + 1),
            inputs=[
                self.vertices,
                self.faces,
                self._unique_edges,
                self._edge_slots,
                self._corner_slots,
                self._adjacency_scan.prefix,
                self._adjacency_scan.chunk_offsets,
                self.state,
                wp.int32(v_cap),
                self._adjacency,
                self._adjacency_offsets,
                self._quadrics,
            ],
            device=device,
        )
        return self._adjacency_offsets

    def _run_collapse_rounds(self, csr_offsets: wp.array[wp.int32]) -> None:
        """
        Commit independent sets of collapses against one scoring, until a round finds nothing new.

        Everything the loop decides with lives in device arrays -- the pass's face count in
        ``state``, and ``_round_state``, the shared round-loop state (``kernels/array.py``'s
        ``LOOP_ROUND`` / ``LOOP_CONDITION``) with a third slot appended for the commits so far --
        so the body holds no host readback and the whole loop is a single ``wp.capture_while``
        node, nested as an inner ``while`` node of the pass graph the caller is capturing.

        A round is four launches and a scan: restore-and-claim, the win test, a scan of the
        winners' flags that ranks them by cost, the budgeted commit (which also locks the
        committed neighbourhoods and re-arms the claim keys), and the loop test. Every round runs
        the identical body, which is what makes one graph enough: on the first round ``locked`` is
        all-zero and the restore is the candidate list unchanged, and on the last the locks and
        re-armed keys are scratch nobody reads again. A budget-exhausted pass stops the way a
        saturated one does -- nothing commits, and ``end_collapse_round`` sees no progress.
        """
        device = self._device
        m = self.n_edges
        width = max(m, self.n_vertices + 1)

        def round_body() -> None:
            _launch.launch(
                kernel_remesh.drop_locked_and_claim,
                dim=max(m, self._winner_words.size),
                inputs=[
                    self._candidates,
                    self._removed,
                    csr_offsets,
                    self._adjacency,
                    self._locked,
                    self._survivor,
                    self._min_key,
                    self._winner_words,
                ],
                device=device,
            )
            _launch.launch(
                kernel_remesh.mark_collapse_winners,
                dim=m,
                inputs=[
                    self._survivor,
                    self._removed,
                    csr_offsets,
                    self._adjacency,
                    self._min_key,
                    self._cost_rank,
                    self._survivor,
                    self._winner_words,
                ],
                device=device,
            )
            self._winner_scan.launch(self._winner_words, words=True)
            _launch.launch(
                kernel_remesh.commit_budgeted_collapses,
                dim=width,
                inputs=[
                    self._survivor,
                    self._removed,
                    self._target_pos,
                    self._cost,
                    self._cost_rank,
                    self._winner_words,
                    self._winner_scan.prefix,
                    self._winner_scan.chunk_offsets,
                    csr_offsets,
                    self._adjacency,
                    self.state,
                    wp.int32(self._target),
                    self._round_state,
                    self._collapse_remap,
                    self._positions,
                    self._count,
                    self._locked,
                    self._min_key,
                ],
                device=device,
            )
            _launch.launch(
                kernel_remesh.end_collapse_round,
                dim=1,
                inputs=[wp.int32(_QUADRIC_ROUNDS), self._count, self._round_state],
                device=device,
            )

        wp.capture_while(
            twt.as_dense(self._round_state[kernel_array.LOOP_CONDITION_VIEW]), round_body
        )

    def _compact(self) -> None:
        """
        Rebuild the face and vertex buffers in place, publishing both new counts to ``state``.

        The tail of ``quadric_decimate``'s pass with its host readbacks removed: the face
        compaction's ``flatnonzero`` and
        [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices] become one
        inclusive scan over both keep masks, and one launch moves the surviving faces -- already
        renumbered -- and the surviving vertices together.
        """
        device = self._device
        _launch.launch(
            kernel_remesh.remap_and_mark_faces,
            dim=self.n_faces,
            inputs=[self.faces, self._collapse_remap, self._remapped, self._compact_flags],
            device=device,
        )
        self._compact_scan.launch(self._compact_flags)
        if self._track_index:
            # The compaction moves the provenance column with the faces, so it reads a copy.
            _launch.copy(self._face_source_scratch, self.face_source)
        _launch.launch(
            kernel_remesh.compact_decimation_pass,
            dim=max(self.n_faces, self.n_vertices),
            inputs=[
                self._remapped,
                self._positions,
                self._compact_flags,
                self._compact_scan.prefix,
                self._compact_scan.chunk_offsets,
                wp.int32(self.n_vertices),
                self._collapse_remap,
                self._face_source_scratch,
                self.vertex_index,
            ],
            outputs=[self.faces, self.vertices, self.face_source, self.state],
            device=device,
        )


class _ExclusiveScan:
    """
    An exclusive scan of a fixed-length ``int32`` buffer that allocates nothing when it runs.

    ``wp.utils.array_scan`` allocates its scratch on every call, which a ``wp.capture_while`` body
    may not do and which adds a memory node to any other captured graph. This one owns its
    buffers: ``scan_chunks_exclusive`` scans each ``SCAN_CHUNK`` into ``prefix`` and publishes the
    chunk totals, and ``scan_chunk_totals_exclusive`` scans those into ``chunk_offsets``. The
    result is split between the two -- a kernel reads entry ``i`` as
    ``kernel_remesh.scanned_prefix(prefix, chunk_offsets, i)``.
    """

    def __init__(self, n: int, device: wp.Device) -> None:
        """Allocate the scan of an ``n``-element buffer on ``device``."""
        chunks = max(1, -(-n // int(kernel_remesh.SCAN_CHUNK)))
        self._device = device
        self.prefix = _launch.empty(n, dtype=wp.int32, device=device)
        self.chunk_totals = _launch.empty(chunks, dtype=wp.int32, device=device)
        self.chunk_offsets = _launch.empty(chunks, dtype=wp.int32, device=device)

    def launch(
        self, values: wp.array[wp.int32] | wp.array[wp.uint32], *, words: bool = False
    ) -> None:
        """
        Scan ``values``, which must have the length this scan was allocated for.

        With ``words`` the input is a ``uint32`` bitmask and the scan is of its words' set-bit
        counts.
        """
        _launch.launch_tiled(
            kernel_remesh.scan_word_counts_exclusive
            if words
            else kernel_remesh.scan_chunks_exclusive,
            dim=[self.chunk_totals.size],
            inputs=[values, self.prefix, self.chunk_totals],
            block_dim=kernel_remesh.SCAN_BLOCK_DIM,
            device=self._device,
        )
        _launch.launch_tiled(
            kernel_remesh.scan_chunk_totals_exclusive,
            dim=[1],
            inputs=[self.chunk_totals, self.chunk_offsets],
            block_dim=kernel_remesh.SCAN_BLOCK_DIM,
            device=self._device,
        )


def _resolve_decimation_target(
    target_faces: int | None, target_ratio: float | None, n_faces: int
) -> int:
    """Validate the mutually exclusive target arguments and reduce them to a face count."""
    if (target_faces is None) == (target_ratio is None):
        raise ValueError("pass exactly one of target_faces and target_ratio")
    if target_faces is not None:
        if target_faces < 0:
            raise ValueError(f"target_faces must be non-negative, got {target_faces}")
        return int(target_faces)
    assert target_ratio is not None
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
    return math.ceil(target_ratio * n_faces)


def flip_to_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool] | None = None,
    max_angle_change: float | None = None,
    max_deviation: float | None = None,
    critical_aspect_ratio: float = 1000.0,
    max_iter: int = 100,
) -> wp.array[wp.int32]:
    """
    Improve triangle quality by flipping interior edges toward the Delaunay criterion.

    For every interior edge whose two incident faces are both in ``region``, the shared diagonal
    is flipped when doing so satisfies the local Delone (empty-circumcircle) test — subject to an
    optional dihedral-angle-change gate and a surface-deviation gate, so the flips never distort
    the surface. Rim edges (with a face
    outside the region, or on the mesh boundary) are never flipped. Vertices, face count and
    region membership are unchanged; only the triangulation of the region is rewritten.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    region
        ``(n_faces,)`` mask; only edges interior to the ``True`` faces are flippable. ``None``
        treats the whole mesh as flippable.
    max_angle_change
        Maximum dihedral-angle change (radians) a flip may introduce. ``None`` disables the gate.
    max_deviation
        Maximum surface deviation a flip may introduce. ``None`` disables the gate.
    critical_aspect_ratio
        Triangle aspect ratio above which the dihedral-angle gate is lifted, so degenerate
        triangles can still be repaired.
    max_iter
        Maximum number of parallel flip passes.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat face buffer with the region re-triangulated, on ``faces.device`` (a
        copy; the input is not modified).

    Raises
    ------
    ValueError
        If ``region`` is given and is not length ``n_faces``.
    RuntimeError
        If ``vertices``, ``faces`` and ``region`` are not all on one device.

    See Also
    --------
    [`flip_by_objective`][triwarp.remesh.flip_by_objective]
        The same flip engine with a shape or flatness predicate in front of it instead of the
        Delone test.
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]
    [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    """
    require_same_device(vertices=vertices, faces=faces, region=region)
    device = faces.device
    setup = _flip_setup(faces, region)
    if setup is None:
        return _launch.clone(faces)
    out_faces, n_vertices, region_flags = setup
    mac, mdsq, car = _flip_gates(max_angle_change, max_deviation, critical_aspect_ratio)

    def launch(
        adjacency: twt.Array2dInt32,
        adjacency_edges: twt.Array2dInt32,
        unshared: twt.Array2dInt32,
        sorted_keys: wp.array[wp.uint64],
        edge_set: wp.array[wp.uint64],
        edge_set_mask: wp.int32,
        key_base: wp.uint64,
        out_flip: wp.array[wp.bool],
        out_quad: twt.Array2dInt32,
    ) -> None:
        _launch.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                out_faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                edge_set,
                edge_set_mask,
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(out_faces, n_vertices, launch, max_iter)
    return out_faces


_OBJECTIVE_QUALITY_METRICS = ("radius_ratio", "area_max_side", "mean_ratio")


def flip_by_objective(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    objective: Literal["planarity", "curvature", "t_vertex"] = "planarity",
    region: wp.array[wp.bool] | None = None,
    planar_angle: float = 1.0,
    metric: Literal["radius_ratio", "area_max_side", "mean_ratio"] = "area_max_side",
    aspect_threshold: float = 40.0,
    max_iter: int = 100,
) -> wp.array[wp.int32]:
    """
    Flip interior edges to optimize triangle shape or surface flatness, instead of the Delone test.

    Runs the same parallel flip engine as [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] —
    independent-set selection over the face adjacency, iterated until no edge is a candidate — with
    a different predicate at the front. That is the whole port: the machinery was already there, and
    these two objectives are what MeshLab's ``meshing_edge_flip_by_planar_optimization`` and
    ``meshing_edge_flip_by_curvature_optimization`` put in front of it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions. Never modified — only the triangulation changes.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    objective
        Which predicate to flip on:

        - ``"planarity"`` (default) — **shape only, surface preserved.** A quad is eligible when its
          two triangles meet within ``planar_angle`` of flat, and its diagonal is flipped when doing
          so raises the ``metric`` score of the *worse* of the two triangles. Because the quad is
          near-planar to begin with, the rewrite is a retriangulation and not a deformation. This is
          the one to reach for after any operation that leaves thin triangles across a flat region.
        - ``"curvature"`` — **flatness, surface changed.** The diagonal is flipped whenever the
          other one bends less, i.e. whenever the dihedral angle across it is smaller. This *does*
          move the surface (it chooses between two interpolations of the same four points) and is
          what makes a coarse triangulation of a curved shape follow its principal directions.
          ``planar_angle`` and ``metric`` are ignored.
        - ``"t_vertex"`` — **slivers only.** A quad is eligible only when one of its two triangles
          is a sliver: its ``aspect_ratio`` (circumradius over twice the inradius) exceeds
          ``aspect_threshold``; the diagonal is then flipped if that improves the worse of the two.
          A T-vertex — a vertex sitting in the interior of a neighbouring edge — is exactly what
          produces such a sliver, which is why this is the repair for one; see
          [`flip_t_vertices`][triwarp.repair.flip_t_vertices] for the wrapper that says so.
          ``planar_angle`` and ``metric`` are ignored.
    region
        ``(n_faces,)`` mask; only edges interior to the ``True`` faces are flippable. ``None``
        treats the whole mesh as flippable.
    planar_angle
        Planarity tolerance in **degrees** for ``objective="planarity"``: a quad whose dihedral
        exceeds it is left alone. MeshLab's ``pthreshold``, whose default of ``1`` is this one. Must
        be in ``[0, 180]``.
    metric
        Which [`face_quality`][triwarp.triangles.face_quality] measure ``objective="planarity"``
        maximizes. Only the three larger-is-better shape measures are accepted — ``"aspect_ratio"``
        runs the other way and ``"area"`` is not a shape measure at all. MeshLab's ``planartype``,
        whose default ``'area/max side'`` is this one.
    aspect_threshold
        Sliver threshold for ``objective="t_vertex"``: only a quad whose worse triangle has an
        ``aspect_ratio`` above this is eligible. MeshLab's ``meshing_remove_t_vertices`` threshold,
        whose default of ``40`` is this one. Must be positive.
    max_iter
        Maximum number of parallel flip passes. Each pass commits a conflict-free independent set,
        so a mesh needing many local rewrites needs several.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat face buffer with the region re-triangulated, on ``faces.device`` (a
        copy; the input is not modified).

    Raises
    ------
    ValueError
        If ``objective`` or ``metric`` is unknown, ``planar_angle`` is outside ``[0, 180]``, or
        ``region`` has the wrong length.
    RuntimeError
        If ``vertices``, ``faces`` and ``region`` are not all on one device.

    See Also
    --------
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`triwarp.triangles.face_quality`][triwarp.triangles.face_quality]
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]

    Notes
    -----
    A quad whose two diagonals score *equally* — every quad of a regular grid — would flip back and
    forth forever, one pass each way, so a flip must beat the incumbent by a relative ``1e-6``
    rather than merely tie it. That margin is what makes ``max_iter`` a safety net rather than the
    normal stopping condition.
    """
    require_same_device(vertices=vertices, faces=faces, region=region)
    if objective not in ("planarity", "curvature", "t_vertex"):
        raise ValueError(
            f"objective must be 'planarity', 'curvature' or 't_vertex', got {objective!r}"
        )
    if aspect_threshold <= 0.0:
        raise ValueError(f"aspect_threshold must be positive, got {aspect_threshold}")
    if metric not in _OBJECTIVE_QUALITY_METRICS:
        raise ValueError(
            f"metric must be one of {list(_OBJECTIVE_QUALITY_METRICS)}, got {metric!r}"
        )
    if not 0.0 <= planar_angle <= 180.0:
        raise ValueError(f"planar_angle must be in [0, 180] degrees, got {planar_angle}")

    device = faces.device
    setup = _flip_setup(faces, region)
    if setup is None:
        return _launch.clone(faces)
    out_faces, n_vertices, region_flags = setup

    objective_flag = {
        "planarity": kernel_remesh.OBJECTIVE_PLANARITY,
        "curvature": kernel_remesh.OBJECTIVE_CURVATURE,
        "t_vertex": kernel_remesh.OBJECTIVE_T_VERTEX,
    }[objective]
    metric_flag = tw.triangles._QUALITY_METRICS[metric]  # pyright: ignore[reportPrivateUsage]
    # The gate is on the dihedral's cosine so the kernel needs no inverse trigonometry.
    planar_cos = wp.float32(math.cos(math.radians(planar_angle)))

    def launch(
        adjacency: twt.Array2dInt32,
        adjacency_edges: twt.Array2dInt32,
        unshared: twt.Array2dInt32,
        sorted_keys: wp.array[wp.uint64],
        edge_set: wp.array[wp.uint64],
        edge_set_mask: wp.int32,
        key_base: wp.uint64,
        out_flip: wp.array[wp.bool],
        out_quad: twt.Array2dInt32,
    ) -> None:
        _launch.launch(
            kernel_remesh.objective_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                out_faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                edge_set,
                edge_set_mask,
                key_base,
                objective_flag,
                metric_flag,
                planar_cos,
                wp.float32(aspect_threshold),
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(out_faces, n_vertices, launch, max_iter)
    return out_faces


def _flip_setup(
    faces: wp.array[wp.int32], region: wp.array[wp.bool] | None
) -> tuple[wp.array[wp.int32], int, wp.array[wp.int32]] | None:
    """
    Working face buffer, vertex count and per-face region flags shared by the two flip drivers.

    Both [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] and
    [`flip_by_objective`][triwarp.remesh.flip_by_objective] flip in place on a clone of the input
    and gate every candidate on the same ``int32`` region mask, which is all ones when the caller
    named no region.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Optional length-``n_faces`` ``wp.bool`` mask restricting which faces may flip.

    Returns
    -------
    tuple[wp.array[wp.int32], int, wp.array[wp.int32]] | None
        ``(out_faces, n_vertices, region_flags)``, or ``None`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``region`` is given and is not length ``n_faces``.
    """
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return None
    if region is not None and region.size != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {region.size}")

    if region is None:
        region_flags = _launch.full(n_faces, 1, dtype=wp.int32, device=device)
    else:
        region_flags = tw.array.astype(region, wp.int32)
    return _launch.clone(faces), tw.array.index_bound(faces), region_flags


def intrinsic_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    max_iter: int = 100,
) -> tuple[wp.array[wp.int32], twt.Array2dFloat32, int]:
    """
    Retriangulate to the intrinsic Delaunay triangulation, without moving a vertex.

    Flips edges whose two opposite angles sum past ``pi`` — exactly the edges whose cotangent weight
    is negative — until none is left. The flip is *intrinsic*: the new edge is not a straight line
    in space but the geodesic across the two triangles, and its length comes from unfolding them
    into a plane and measuring the other diagonal. The surface, its vertices and its metric are all
    untouched; only which pairs of vertices count as connected changes, so every operator built from
    the result is a better-behaved operator for the *same* geometry.

    Its practical effect is that the cotangent weights all become non-negative, which is what a
    Laplacian needs to satisfy a maximum principle: no spurious extrema, no negative diffusion, far
    better conditioned solves on a badly-shaped mesh.

    Flips run in parallel rounds, each committing a conflict-free independent set (the same engine
    behind [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]) with the edge-length table carried
    alongside the connectivity.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Not modified.
    epsilon
        Mollification margin applied before flipping, relative to the mean edge length: a degenerate
        triangle has no well-defined angles to test.
    max_iter
        Cap on the number of parallel flip rounds.

    Returns
    -------
    intrinsic_faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` connectivity of the intrinsic triangulation, over the same vertices.
    edge_lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` intrinsic edge lengths for those faces, column ``e`` opposite corner ``e``.
    n_flips : int
        How many edges were flipped. Zero means the input was already intrinsically Delaunay.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Unlike the other flip passes in this module, this one *can* create a second edge between two
    vertices some other edge already connects — intrinsically that is a different geodesic, not a
    duplicate, and the flip loop tracks connectivity through an incrementally-maintained halfedge
    twin table rather than a vertex-pair key, so the two never collide. The one thing that still
    cannot be flipped away is a negative weight on a *boundary* edge: the Delaunay two-opposite-
    angles condition has nothing to compare a boundary edge's one incident angle against, so no
    flip of any kind addresses it. A caller that needs the maximum principle should check the
    weights it got ([`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic])
    rather than inferring them from convergence.

    See Also
    --------
    [`robust_laplacian`][triwarp.laplacian.robust_laplacian]
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    intrinsic_faces = _launch.clone(faces)
    if n_faces == 0:
        return intrinsic_faces, lengths, 0

    n_half = n_faces * 3
    # The one point in this loop where a vertex-pair-keyed adjacency table is trustworthy: the
    # caller's input is still a simplicial complex, so ``_FlipTopology``'s ordinary rebuild can
    # derive it, once. Every later round instead maintains ``twin`` in place, because a flip can
    # make the vertex-pair key ambiguous (see ``kernel_remesh.build_intrinsic_twins``).
    initial = _FlipTopology(intrinsic_faces, n_vertices)
    m0 = initial.rebuild()
    twin = _launch.full(n_half, -1, dtype=wp.int32, device=device)
    if m0 > 0:
        _launch.launch(
            kernel_remesh.build_intrinsic_twins,
            dim=m0,
            inputs=[intrinsic_faces, initial.adjacency, initial.unshared, twin],
            device=device,
        )
    del initial  # its vertex-pair-keyed tables cannot represent what a flip may do from here on

    flip = _launch.empty(n_half, dtype=wp.bool, device=device)
    quad = twt.empty_2d((n_half, 4), wp.int32, device=device)
    new_length = _launch.empty(n_half, dtype=wp.float32, device=device)
    neighbors = twt.empty_2d((n_half, 4), wp.int32, device=device)
    # Armed here for the first round; ``fixup_twin_remap`` re-arms it for every later one.
    face_claim = _launch.full(n_faces, INT32_MAX, dtype=wp.int32, device=device)
    remap = _launch.empty(n_half, dtype=wp.int32, device=device)
    no_remap = _launch.zeros(n_half, dtype=wp.bool, device=device)
    count = _launch.zeros(1, dtype=wp.int32, device=device)

    total = 0
    for _ in range(max_iter):
        # The per-iteration resets ride the launches next to their readers: ``remap`` /
        # ``no_remap`` / ``count`` in the candidate pass, the claim table in the previous
        # iteration's ``fixup_twin_remap``. The candidate pass also claims: independent-set
        # selection over just the flipping pair -- that kernel's comment says why that is enough
        # here, unlike the vertex-pair-keyed flip loops.
        _launch.launch(
            kernel_remesh.intrinsic_delaunay_candidates,
            dim=n_half,
            inputs=[intrinsic_faces, lengths, twin],
            outputs=[flip, quad, new_length, neighbors, face_claim, remap, no_remap, count],
            device=device,
        )
        _launch.launch(
            kernel_remesh.commit_intrinsic_flips,
            dim=n_half,
            inputs=[
                flip,
                quad,
                neighbors,
                face_claim,
                new_length,
                intrinsic_faces,
                lengths,
                twin,
                remap,
                no_remap,
                count,
            ],
            device=device,
        )
        _launch.launch(
            kernel_remesh.fixup_twin_remap,
            dim=n_half,
            inputs=[remap, no_remap, twin],
            outputs=[face_claim],
            device=device,
        )
        n = int(read_scalar(count, 0))
        total += n
        if n == 0:
            break
    return intrinsic_faces, lengths, total


def subdivide(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Subdivide a mesh by splitting every face into four triangles.

    Each triangle is split by placing a new vertex at the midpoint of each
    edge. The four child triangles share these midpoints and preserve the
    original winding order, matching [`trimesh.remesh.subdivide`][] exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices + n_edges,)`` ``new_vertices`` and ``(12 * n_faces,)`` ``new_faces`` on
        ``vertices.device``, ``n_edges`` the unique edge count.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`trimesh.remesh.subdivide`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size

    n_faces = faces.size // 3
    if n_faces == 0:
        # Cloned, not aliased: every other entry point in this module returns independent buffers,
        # and a caller that mutates a "subdivided" mesh must not reach back into its own input.
        return _launch.clone(vertices), _launch.clone(faces)

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    n_unique = int(unique_edges.shape[0])

    # One buffer sized for its final use: the originals copied into the prefix and one midpoint per
    # unique edge written straight into the tail, which is the layout ``_split_faces_four`` indexes.
    new_vertices = _launch.empty(n_vertices + n_unique, dtype=wp.vec3, device=device)
    _launch.copy(new_vertices, vertices, count=n_vertices)
    _launch.launch(
        kernel_remesh.compute_midpoints,
        dim=n_unique,
        inputs=[vertices, unique_edges],
        outputs=[new_vertices[n_vertices:]],
        device=device,
    )
    return new_vertices, _split_faces_four(faces, inverse, n_vertices)


@overload
def subdivide_loop(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    return_operator: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_loop(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_operator: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], twt.BsrMatrix[wp.float32]]: ...
def subdivide_loop(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_operator: bool = False
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], twt.BsrMatrix[wp.float32]]
):
    """

    Subdivide a mesh with one pass of Loop subdivision.

    Same 1-to-4 split as [`subdivide`][triwarp.remesh.subdivide] -- identical face table, identical
    index layout -- but the positions are the smooth Loop stencils rather than midpoints, so the
    surface is *approximated* instead of interpolated: original vertices move, and repeated
    application converges to a C² limit surface (C¹ at irregular vertices).

    - **Odd (edge) vertices**, one per unique edge: ``3/8`` on each endpoint and ``1/8`` on each of
      the two vertices opposite the edge. A boundary edge takes the midpoint instead.
    - **Even (original) vertices**, interior: ``(1 - n * beta) * v + beta * sum(ring)`` with
      **Warren's** ``beta`` -- ``3/16`` at valence 3, ``3/(8n)`` above -- which is the variant
      ``igl.loop`` uses, rather than Loop's original trigonometric weight.
    - **Even vertices on a boundary**: ``3/4 * v`` plus ``1/8`` of each of the two neighbours along
      the boundary, with interior neighbours excluded, so a boundary curve subdivides identically
      from either side of a seam.

    The mesh should be edge-manifold. Where it is not, the stencils are not defined and the
    fallbacks are conservative rather than arbitrary: an edge with three or more incident faces
    takes the midpoint rule, and a vertex where one or three-plus boundary edges meet -- or one with
    no edges at all -- keeps its position.


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    return_operator
        If ``True``, also return the sparse interpolation operator ``P`` this pass applies, so that
        ``new_vertices == P @ vertices`` and **any** per-vertex attribute can be carried through the
        subdivision by the same matrix (see
        [`triwarp.interpolation.transfer_through_operator`][triwarp.interpolation.transfer_through_operator]).

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_vertices + n_edges,)`` positions on ``vertices.device``, ``n_edges`` the unique edge
        count: the ``n_vertices`` relocated originals first and then one vertex per unique edge, so
        the leading ``n_vertices`` rows are the input vertex set *displaced* -- unlike
        ``subdivide``, where that prefix is unchanged.
    new_faces : wp.array[wp.int32]
        ``(12 * n_faces,)`` flat triangle index buffer.
    operator : warp.sparse.BsrMatrix
        ``(n_vertices + n_edges, n_vertices)`` matrix of Loop weights, only when
        ``return_operator`` is ``True``, one row per output vertex in the same layout as
        ``new_vertices``. Every row sums to 1.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    **Why an operator rather than an index map.** Every other topology edit here reports provenance
    as one ``int32`` per output element, because each output comes from exactly one input. A Loop
    vertex does not: an odd vertex is an affine combination of four inputs and an even vertex of its
    whole 1-ring, so no index map can express it, and an attribute cannot otherwise be carried
    through this function at all. The operator is the honest form, it is the standard prolongation
    object, and it costs nothing unless asked for.

    The operator is assembled from the same three grids and through the same two weight functions
    (``kernels/remesh.loop_odd_weights`` / ``loop_even_weights``) that the position kernels use, so
    the two cannot drift onto different surfaces. It is *not* used to compute the positions -- those
    stay two direct kernels, since a sparse build plus a ``bsr_mv`` would make every
    caller pay for the matrix.

    Four launches over three grids -- faces, unique edges, vertices -- plus the shared topology.
    Neither stencil needs an ordered 1-ring: the valence and the ring sum come from an atomic pass
    over the *unique* edges, which is the deduplicated neighbour count ``igl::loop`` reads off a
    sorted adjacency list, and the two boundary neighbours are found as the ones joined by boundary
    edges rather than as the ends of that list.

    For several passes, call this repeatedly -- that is what ``igl.loop``'s ``number_of_subdivs``
    does internally, and each pass multiplies the face count by four.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    ``igl.loop``
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_faces == 0:
        # No faces means no edges and no relocation, so the pass is the identity -- but it
        # returns independent buffers all the same, as every other entry point here does.
        if return_operator:
            return (
                _launch.clone(vertices),
                _launch.clone(faces),
                twt.bsr_identity(n_vertices, wp.float32, device=device),
            )
        return _launch.clone(vertices), _launch.clone(faces)

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    n_unique = int(unique_edges.shape[0])

    # How many faces each edge carries, and the sum of the vertices opposite it.
    edge_opposite_sum = _launch.zeros(n_unique, dtype=wp.vec3, device=device)
    edge_face_count = _launch.zeros(n_unique, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.loop_edge_opposites,
        dim=n_faces,
        inputs=[vertices, faces, inverse, edge_opposite_sum, edge_face_count],
        device=device,
    )

    valence = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    ring_sum = _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
    boundary_count = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
    boundary_sum = _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
    # One buffer sized for its final use, written through two views: the relocated originals in the
    # prefix and the new edge vertices after them, which is the index layout `_split_faces_four`
    # assumes and the one ``igl.loop`` returns. The edge vertices are written by the ring pass,
    # which has their endpoints in hand.
    new_vertices = _launch.empty(n_vertices + n_unique, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_remesh.loop_vertex_rings,
        dim=n_unique,
        inputs=[vertices, unique_edges, edge_face_count, edge_opposite_sum],
        outputs=[valence, ring_sum, boundary_count, boundary_sum, new_vertices[n_vertices:]],
        device=device,
    )
    # The operator's triplets, when asked for: the even self-weights are written by the even
    # position pass below, the rest by ``_loop_operator``.
    operator_triplets = (
        tw.array.triplet_buffers(
            _loop_triplet_count(n_vertices, n_unique, n_faces), wp.float32, device
        )
        if return_operator
        else None
    )
    _launch.launch(
        kernel_remesh.loop_even_positions,
        dim=n_vertices,
        inputs=[vertices, valence, ring_sum, boundary_count, boundary_sum],
        outputs=[new_vertices[:n_vertices], *(operator_triplets or (None, None, None))],
        device=device,
    )
    new_faces = _split_faces_four(faces, inverse, n_vertices)
    if operator_triplets is None:
        return new_vertices, new_faces
    return (
        new_vertices,
        new_faces,
        _loop_operator(
            faces,
            unique_edges,
            inverse,
            edge_face_count,
            valence,
            boundary_count,
            n_vertices,
            operator_triplets,
        ),
    )


def _loop_operator(
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32,
    inverse: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    valence: wp.array[wp.int32],
    boundary_count: wp.array[wp.int32],
    n_vertices: int,
    triplets: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.float32]],
) -> twt.BsrMatrix[wp.float32]:
    """
    Assemble one Loop pass as a sparse interpolation matrix, from the pass's own intermediates.

    Launches over the three grids the positions come from -- vertices, unique edges, faces --
    writing into one triplet buffer, sized by ``_loop_triplet_count``, whose vertex rows (the even
    self-weights) the even position pass has already written. ``array.csr_from_triplets`` sums
    coincident entries, which is what lets the odd rows' 3/8 endpoints (edge grid) and 1/8 wings
    (face grid) be emitted independently.
    """
    device = faces.device
    n_faces = faces.size // 3
    n_unique = int(unique_edges.shape[0])
    edge_base = n_vertices
    face_base = edge_base + 4 * n_unique
    rows, cols, values = triplets
    _launch.launch(
        kernel_remesh.loop_edge_triplets,
        dim=n_unique,
        inputs=[
            unique_edges,
            edge_face_count,
            valence,
            boundary_count,
            wp.int32(n_vertices),
            wp.int32(edge_base),
            rows,
            cols,
            values,
        ],
        device=device,
    )
    _launch.launch(
        kernel_remesh.loop_opposite_triplets,
        dim=n_faces,
        inputs=[
            faces,
            inverse,
            edge_face_count,
            wp.int32(n_vertices),
            wp.int32(face_base),
            rows,
            cols,
            values,
        ],
        device=device,
    )
    return tw.array.csr_from_triplets(n_vertices + n_unique, n_vertices, rows, cols, values)


def _loop_triplet_count(n_vertices: int, n_unique: int, n_faces: int) -> int:
    """Triplet count of one Loop pass's operator: one per vertex, four per edge, three per face."""
    return n_vertices + 4 * n_unique + 3 * n_faces


def _split_faces_four(
    faces: wp.array[wp.int32], inverse: wp.array[wp.int32], n_vertices: int
) -> wp.array[wp.int32]:
    """
    Build the 1-to-4 face table both uniform subdivisions share, from the per-corner edge map.

    New vertex ``n_vertices + e`` belongs to unique edge ``e``, and corner ``j`` of face ``f`` spans
    ``(fv[j], fv[j + 1])``, so shifting ``inverse`` by ``n_vertices`` is the whole index translation
    [`subdivide`][triwarp.remesh.subdivide] and [`subdivide_loop`][triwarp.remesh.subdivide_loop]
    need before the split -- the two differ only in where they put the new positions.
    """
    device = faces.device
    n_faces = faces.size // 3
    out_new_faces = _launch.empty(n_faces * 12, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.subdivide_faces,
        dim=n_faces,
        inputs=[faces, inverse, wp.int32(n_vertices), out_new_faces],
        device=device,
    )
    return out_new_faces


@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Subdivide a mesh until every edge is at most ``max_edge`` long.

    Every edge longer than ``max_edge`` is bisected at a single shared midpoint,
    so the two faces on either side stay in sync and a watertight input stays
    watertight — no T-junctions (cracks) are introduced. Faces already small
    enough are left untouched. Each pass splits every over-long edge once and
    re-triangulates the incident faces with per-face templates (1, 2, or 3 split
    edges; the 2-split quad is cut along its shorter diagonal), iterating until
    no edge exceeds the threshold, matching
    [`trimesh.remesh.subdivide_to_size`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_edge
        Maximum length of any edge in the result. A **scalar** gives the uniform target; a
        ``(n_vertices,)`` ``wp.float32`` array is a per-vertex **sizing field**, and an edge's own
        target is then the mean of its two endpoints', so the refinement is fine where the field is
        small and coarse where it is large. The field is *extended* to each inserted midpoint as the
        mean of the endpoints it splits, so no resampling is needed between passes and a field that
        satisfies the target cannot be driven past it by a later pass.
    max_iter
        Maximum number of subdivision passes. A ``ValueError`` is raised if the
        mesh still has an over-long edge after this many passes.
    return_index
        If ``True``, also return the source face index of each output face.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` refined vertex positions on ``vertices.device`` (original vertices
        first, then the inserted edge midpoints).
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat buffer of the refined faces.
    index : wp.array[wp.int32]
        ``(n_out_faces,)`` index of the original face each output face was refined from; only
        returned when ``return_index`` is ``True``.

    Raises
    ------
    ValueError
        If any edge is still longer than ``max_edge`` after ``max_iter`` passes.
    RuntimeError
        If ``vertices``, ``faces`` and ``max_edge`` are not all on one device.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.remesh.subdivide_to_size`][]
    """
    require_same_device(vertices=vertices, faces=faces, max_edge=max_edge)
    device = vertices.device
    sizing = max_edge if isinstance(max_edge, wp.array) else None
    max_edge_f = wp.float32(0.0 if isinstance(max_edge, wp.array) else max_edge)

    current_vertices = vertices
    current_faces = faces
    n_faces = faces.size // 3
    index = tw.array.arange(n_faces, device=device)

    if n_faces == 0:
        if return_index:
            return _launch.clone(vertices), _launch.clone(faces), index
        return _launch.clone(vertices), _launch.clone(faces)

    for i in range(max_iter + 1):
        n_vertices = current_vertices.size

        unique_edges, inverse = tw.edges.edges_unique(
            current_faces, n_vertices=n_vertices, validate=False
        )
        m = int(unique_edges.shape[0])

        # Flag the edges that are longer than the target length, measuring each where it is tested.
        long_mask = _launch.empty(m, dtype=wp.bool, device=device)
        rank_buffer = _launch.zeros(m + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_remesh.mark_long_edges,
            dim=m,
            inputs=[
                current_vertices,
                unique_edges,
                max_edge_f,
                sizing,
                wp.bool(sizing is not None),
                long_mask,
                rank_buffer[1:],
            ],
            device=device,
        )

        # A sizing field must grow with the vertex buffer, and it is *extended* rather than
        # re-sampled: a midpoint's target is the mean of the endpoints it splits, which is the same
        # value the edge was tested against, so a run of passes cannot drift the field. Computed
        # before the split because it reads the pre-split edge rows.
        # The ranks are taken once here and shared by the sizing extension and the split.
        offsets, n_split = _ranks_from_flags(rank_buffer)
        # Every edge is short enough: we are done. The count is already on the host, so the pass
        # that finds nothing to split stops here rather than running a split that only copies.
        if n_split == 0:
            break
        # Ran out of passes with over-long edges still present.
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")
        next_sizing = (
            None
            if sizing is None
            else _extend_sizing_field(sizing, unique_edges, long_mask, offsets, n_split)
        )

        # ``index`` rides through the split rather than being gathered afterwards.
        new_vertices, new_faces, new_index = _split_ranked_edges(
            current_vertices,
            current_faces,
            long_mask,
            offsets,
            n_split,
            unique_edges,
            inverse,
            index,
            index,
            None,
            True,
        )
        current_vertices, current_faces, index, sizing = (
            new_vertices,
            new_faces,
            new_index,
            next_sizing,
        )

    # Nothing ever split, so ``current_*`` are still the caller's own buffers. Every entry point in
    # this module returns independent ones -- ``subdivide`` says so in as many words -- and a caller
    # that mutates a "subdivided" mesh must not reach back into its own input. Cloning here rather
    # than up front keeps the path that *did* split free, since ``split_edges`` already handed back
    # fresh buffers there.
    if current_vertices is vertices:
        current_vertices, current_faces = _launch.clone(vertices), _launch.clone(faces)
    if return_index:
        return current_vertices, current_faces, index
    return current_vertices, current_faces


def _extend_sizing_field(
    sizing: wp.array[wp.float32],
    unique_edges: twt.Array2dInt32,
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    n_split: int,
) -> wp.array[wp.float32]:
    """
    Append one sizing value per edge about to be split: the mean of the edge's two endpoints.

    Called before the split rather than after, because it needs the *pre-split* edge rows, and the
    value it writes is exactly the target the edge was just tested against — so a midpoint inherits
    the size that justified inserting it and repeated passes converge instead of drifting.
    """
    device = sizing.device
    if n_split == 0:
        return sizing
    # Sized for its final use: the current field copied into the prefix and the new values written
    # straight into the tail, in the same order the split appends the midpoints.
    n_vertices = sizing.size
    extended = _launch.empty(n_vertices + n_split, dtype=wp.float32, device=device)
    _launch.copy(extended, sizing, count=n_vertices)
    _launch.launch(
        kernel_remesh.fill_edge_mean_sizing,
        dim=int(unique_edges.shape[0]),
        inputs=[sizing, unique_edges, split_mask, offsets],
        outputs=[extended[n_vertices:]],
        device=device,
    )
    return extended


def subdivide_region_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    max_edge: float,
    max_iter: int = 10,
    max_splits: int | None = None,
    delaunay: bool = True,
    max_angle_change: float | None = math.pi / 6.0,
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Subdivide only a face region until its edges are at most ``max_edge`` long.

    The region-restricted form of [`subdivide_to_size`][triwarp.remesh.subdivide_to_size], and the
    refinement stage of the smooth-patch pipeline: every edge with at least one incident region
    face and length
    greater than ``max_edge`` is bisected, the incident faces are re-triangulated crack-free
    (the [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] 1/2/3-split templates, so faces
    outside the region that touch a split edge stay watertight), and — unless disabled — a
    parallel Delaunay edge-flip pass ([`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay])
    improves the region triangulation after each pass. New vertices are appended after the
    originals, so the caller derives the new-vertex set as the index range
    ``[len(vertices), len(new_vertices))``.

    Where a sequential implementation would drive this from a longest-edge-first priority queue,
    splitting is done in parallel passes; ``max_splits`` is honoured as a soft budget by keeping
    only the longest eligible edges of the pass that would exceed it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    region
        ``(n_faces,)`` mask; only edges touching a ``True`` face are refined.
    max_edge
        Target maximum edge length inside the region.
    max_iter
        Maximum number of subdivision passes.
    max_splits
        Optional soft cap on the total number of edge splits. ``None`` keeps
        splitting until convergence and raises if ``max_iter`` is exhausted first.
    delaunay
        When ``True`` (default), interleave and finish with the Delaunay flip pass.
    max_angle_change
        Dihedral-angle-change gate (radians) for the flip pass (default 30°). ``None`` disables
        the gate.
    max_deviation
        Surface-deviation gate for the flip pass. ``None`` disables it.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` original vertices followed by the inserted midpoints, on
        ``vertices.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat buffer of the refined faces.
    new_region : wp.array[wp.bool]
        ``(n_out_faces,)`` region mask; child faces inherit their parent's membership.

    Raises
    ------
    ValueError
        If ``region`` length does not match the face count, or if over-long region edges remain
        after ``max_iter`` passes and ``max_splits`` is ``None``.
    RuntimeError
        If ``vertices``, ``faces`` and ``region`` are not all on one device.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`fill_smooth`][triwarp.holes.fill_smooth]
    """
    require_same_device(vertices=vertices, faces=faces, region=region)
    device = vertices.device
    n_faces = faces.size // 3
    if region.size != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {region.size}")
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces), _launch.clone(region)

    max_edge_f = wp.float32(max_edge)
    current_vertices = vertices
    current_faces = faces
    region_flags = tw.array.astype(region, wp.int32)
    splits_done = 0
    # The flip pass's working set over ``current_faces``, kept while that buffer lives: its edge
    # sort already is the next pass's ``edges_unique``, and the closing flip pass starts from it.
    topology: _FlipTopology | None = None

    for i in range(max_iter + 1):
        n_faces = current_faces.size // 3
        n_vertices = current_vertices.size

        if topology is not None and topology.built:
            unique_edges, inverse = topology.edges_unique()
        else:
            unique_edges, inverse = tw.edges.edges_unique(
                current_faces, n_vertices=n_vertices, validate=False
            )
        m = int(unique_edges.shape[0])

        long_mask = _launch.zeros(m, dtype=wp.bool, device=device)
        rank_buffer = _launch.zeros(m + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_remesh.mark_long_region_edges,
            dim=3 * n_faces,
            inputs=[current_vertices, unique_edges, region_flags, inverse, max_edge_f],
            outputs=[long_mask, rank_buffer[1:]],
            device=device,
        )

        # The count drives the stopping test, the budget and the running total. The ranks are the
        # split's own and their scan total is the count, so the split below reuses them rather than
        # scanning the mask a second time; a budget-trimmed mask re-ranks.
        offsets, n_long = _ranks_from_flags(rank_buffer)

        if n_long == 0:
            break
        if i >= max_iter:
            if max_splits is None:
                raise ValueError("max_iter exceeded!")
            break

        if max_splits is not None:
            remaining = max_splits - splits_done
            if remaining <= 0:
                break
            if n_long > remaining:
                # It keeps exactly ``remaining`` of the flagged edges, by construction, so the
                # new count needs no second reduction.
                # The lengths are measured only here, on the pass the budget actually trims:
                # the flagging above computes each one where it tests it.
                lengths = tw.edges.edges_unique_length(
                    current_vertices, current_faces, unique_edges=unique_edges
                )
                long_mask = _keep_longest_edges(long_mask, lengths, remaining, m, device)
                offsets, n_long = tw.array.mask_to_compact_ranks(long_mask)

        # The crack-free split itself is [`split_edges`][triwarp.remesh.split_edges], which is
        # exactly what this loop contributes nothing new to: all this function decides is *which*
        # edges (long, and inside the region) and what rides along (``region_flags``, carried
        # through ``index`` so the grown region comes back resolved onto the new faces).
        current_vertices, current_faces, region_flags = _split_ranked_edges(
            current_vertices,
            current_faces,
            long_mask,
            offsets,
            n_long,
            unique_edges,
            inverse,
            region_flags,
            region_flags,
            None,
            True,
        )
        splits_done += n_long

        topology = None
        if delaunay:
            topology = _FlipTopology(current_faces, current_vertices.size)
            _flip_region_faces(
                current_vertices,
                current_faces,
                region_flags,
                max_angle_change,
                max_deviation,
                8,
                topology,
            )

    # Nothing in the region needed splitting, so ``current_*`` are still the caller's own buffers;
    # see ``subdivide_to_size``'s tail for why that has to be broken here. ``new_region`` is always
    # freshly allocated by ``astype``, so only the two mesh buffers are at stake. This has to run
    # *above* the closing flip pass, not below it: ``_flip_region_faces`` rewrites its face buffer
    # in place, so cloning afterwards would hand back a copy of an already-mutated input.
    if current_vertices is vertices:
        current_vertices, current_faces = _launch.clone(vertices), _launch.clone(faces)

    if delaunay:
        _flip_region_faces(
            current_vertices,
            current_faces,
            region_flags,
            max_angle_change,
            max_deviation,
            50,
            topology,
        )

    new_region = tw.array.astype(region_flags, wp.bool)
    return current_vertices, current_faces, new_region


def _keep_longest_edges(
    long_mask: wp.array[wp.bool],
    lengths: wp.array[wp.float32],
    remaining: int,
    m: int,
    device: wp.DeviceLike,
) -> wp.array[wp.bool]:
    """
    Keep only the ``remaining`` longest edges currently flagged in ``long_mask``.

    Sorts the eligible lengths on the device rather than reading ``long_mask`` and ``lengths`` back
    to pick the top ``remaining`` with ``numpy.argsort``, the same spelling
    [`sample_surface_poisson_disk`][triwarp.sample.sample_surface_poisson_disk]'s final round uses.
    ``m`` is the unique-edge count and grows with the mesh, which is what makes a host readback here
    the wrong side of the trade at scale.

    Ties are not ordered by contract on either path -- ``numpy.argsort``'s introsort is unstable and
    ``sort_and_argsort`` is a stable radix sort, and the budget is documented as soft and as keeping
    "the longest eligible edges", which every tie-break satisfies equally.
    """
    eligible = tw.array.flatnonzero(long_mask)
    # Ascending on the negated length is descending on the length, and ``sort_and_argsort`` is the
    # package's one radix-sort spelling. ``order[:remaining]`` is a contiguous *prefix* slice, which
    # is the case CLAUDE.md section 3.4 says a gather may index through directly -- it is a column
    # (``arr[:, k]``) or a step slice whose stride Warp ignores. Cloning it dense first is a
    # measurable loss on the gather for byte-identical output.
    descending = _launch.empty(eligible.size, dtype=wp.float32, device=device)
    _launch.map(wp.neg, tw.array.gather(lengths, eligible), out=descending)
    _sorted, order = tw.array.sort_and_argsort(descending)
    keep = tw.array.gather(eligible, twt.as_dense(order[:remaining]))
    return tw.array.indices_to_mask(keep, m, device=device)


def refine_region_to_density(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    *,
    max_iter: int = 10,
    alpha: float = math.sqrt(2.0),
    delaunay: bool = True,
    max_angle_change: float | None = math.radians(30.0),
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Refine a face region until its sampling matches the surrounding mesh's, not a target length.

    The density-driven sibling of
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size], and the difference is the
    criterion rather than the mechanism. That one bisects every region edge longer than a single
    global ``max_edge``; this one gives each vertex a **scale attribute** -- the average length of
    the edges incident to it -- and splits a region triangle only while its own scale is coarse
    relative to its corners'. On a uniformly sampled neighbourhood the two agree; on a *graded* one
    they do not, because a single length cannot be right at both ends of the grading.

    The rule is Liepa's (see Notes). Writing ``sigma(v)`` for the scale attribute, ``c`` for a
    triangle's centroid and ``sigma(c)`` for the mean of its three corners' attributes, the triangle
    is split at ``c`` when

        ``alpha * |c - v_m| > sigma(c)``  and  ``alpha * sigma(c) > sigma(v_m)``

    holds for every corner ``m``. The first clause refines; the second is what makes the process
    *terminate at the surrounding sampling* rather than at a tolerance. A pass that splits nothing
    ends the loop.

    The split is a **1 -> 3 centroid split**, which is what makes this cheap in parallel: the new
    vertex is interior to the triangle and no edge is divided, so there is nothing to agree with the
    neighbours about and no crack-free template is needed -- unlike edge bisection, which is why
    ``subdivide_region_to_size`` carries the 1/2/3 split families. One pass is one prefix scan and
    one kernel.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    region
        ``(n_faces,)`` mask; only ``True`` faces are refined. Child faces inherit
        their parent's membership.
    max_iter
        Cap on refinement passes. Unlike ``subdivide_region_to_size`` this does **not** raise when
        the cap is reached: the criterion is a density match rather than a hard bound, so stopping
        early leaves a coarser patch and not a wrong one.
    alpha
        The criterion's constant, ``sqrt(2)`` in the paper. Larger refines further, smaller stops
        sooner; it is the only real tuning knob here.
    delaunay
        When ``True`` (default), run the parallel Delone edge-flip pass over the region after each
        split pass, which is the relaxation step the paper pairs with the criterion.
    max_angle_change
        Dihedral-angle-change gate for that flip pass (default 30 degrees). ``None`` disables the
        gate.
    max_deviation
        Surface-deviation gate for the flip pass. ``None`` disables it.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` positions with the centroid vertices appended after the originals, so
        the caller derives the new-vertex set as the index range
        ``[len(vertices), len(new_vertices))``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat buffer of the refined faces.
    new_region : wp.array[wp.bool]
        ``(n_out_faces,)`` region mask.

    Raises
    ------
    ValueError
        If ``region`` length does not match the face count.
    RuntimeError
        If ``vertices``, ``faces`` and ``region`` are not all on one device.

    See Also
    --------
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]
        The same operation driven by a target edge length instead.
    [`smoothing.refine_and_smooth_region`][triwarp.smoothing.refine_and_smooth_region]
        Where the two criteria are selected between, and what the hole fillers reach through.
    [`holes.fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    The criterion is section 3 of P. Liepa, *"Filling holes in meshes"*, Eurographics/ACM SIGGRAPH
    Symposium on Geometry Processing (2003).

    The scale attribute is computed **once**, from the mesh as given, and only *extended* as
    vertices are added -- a centroid inherits the mean of its parents'. That is the point of it: it
    carries the surrounding sampling inward across the patch instead of being re-measured from the
    increasingly fine triangles it is producing, which would never converge.
    """
    require_same_device(vertices=vertices, faces=faces, region=region)
    device = vertices.device
    n_faces = faces.size // 3
    if region.size != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {region.size}")
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces), _launch.clone(region)

    alpha_f = wp.float32(alpha)
    current_vertices = vertices
    current_faces = faces
    current_region = region
    scale = _vertex_scale_attribute(vertices, faces, region)

    for _ in range(max_iter):
        n_faces = current_faces.size // 3
        # The flags are written into the tail of the ranks buffer and scanned there, so the emit
        # reads each face's verdict back as a step in the scan.
        split_offsets = _launch.zeros(n_faces + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_remesh.mark_density_splits,
            dim=n_faces,
            inputs=[current_vertices, current_faces, current_region, scale, alpha_f],
            outputs=[split_offsets[1:]],
            device=device,
        )
        _ranks, n_split = _ranks_from_flags(split_offsets)
        if n_split == 0:
            break

        # A split face becomes three, so the output face count needs no second scan; and the new
        # centroids and their scales are written straight into the tails of buffers sized for the
        # grown mesh, whose prefixes are the current ones.
        n_vertices = current_vertices.size
        n_out_faces = n_faces + 2 * n_split
        new_vertices = _launch.empty(n_vertices + n_split, dtype=wp.vec3, device=device)
        new_scale = _launch.empty(n_vertices + n_split, dtype=wp.float32, device=device)
        _launch.copy(new_vertices, current_vertices, count=n_vertices)
        _launch.copy(new_scale, scale, count=n_vertices)
        out_faces = _launch.empty(3 * n_out_faces, dtype=wp.int32, device=device)
        out_region = _launch.empty(n_out_faces, dtype=wp.bool, device=device)
        region_flags = (
            _launch.empty(n_out_faces, dtype=wp.int32, device=device) if delaunay else None
        )
        _launch.launch(
            kernel_remesh.emit_density_splits,
            dim=n_faces,
            inputs=[
                current_vertices,
                current_faces,
                current_region,
                scale,
                split_offsets,
                wp.int32(n_vertices),
            ],
            outputs=[
                new_vertices[n_vertices:],
                new_scale[n_vertices:],
                out_faces,
                out_region,
                region_flags,
            ],
            device=device,
        )
        current_vertices = new_vertices
        scale = new_scale
        current_faces = out_faces
        current_region = out_region

        if region_flags is not None:
            _flip_region_faces(
                current_vertices, current_faces, region_flags, max_angle_change, max_deviation, 8
            )

    # No face was dense enough to split, so all three are still the caller's own buffers -- this one
    # aliases ``region`` outright, with no ``astype`` in between. See ``subdivide_to_size``'s tail.
    if current_vertices is vertices:
        current_vertices, current_faces, current_region = (
            _launch.clone(vertices),
            _launch.clone(faces),
            _launch.clone(region),
        )
    return current_vertices, current_faces, current_region


def _vertex_scale_attribute(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], region: wp.array[wp.bool]
) -> wp.array[wp.float32]:
    """
    Liepa's per-vertex scale attribute: the mean incident edge length in the *surrounding* mesh.

    "Surrounding" is load-bearing, not a synonym for "whole": the edges counted are those of the
    faces **outside** ``region``. Including the region's own edges is the natural-looking mistake
    and it defeats the criterion -- a minimum-weight patch spans its rim with long chords, so a rim
    vertex that happens to carry two of them reads a scale several times its neighbourhood's, the
    ``alpha * sigma(c) > sigma(v_m)`` clause fails there, and the patch is left unrefined.

    Over the **unique** edge list, so an interior edge counts once at each endpoint rather than
    twice -- which is why the scatter reads one edge per run of the sorted edge keys rather than
    one per halfedge. A vertex with no surrounding edge at all keeps ``0`` (the guarded
    division rather than ``nan``), which makes its clause fail and leaves its triangles alone; when
    the region is the *whole* mesh there is no surrounding mesh to measure and every edge is
    counted instead, so the criterion degrades to the mesh's own average rather than to zero.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = region.size
    n = 3 * n_faces
    # Every halfedge's edge key, the region's lifted past the surrounding mesh's, sorted once: the
    # unique surrounding edges are the leading runs, in ``edges_unique``'s order, so the class the
    # attribute measures is chosen on the device and nothing is compacted or read back. The sort's
    # buffers are double width, the upper halves its scratch; its payload is carried and never
    # read, so it is not initialized.
    keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n, dtype=wp.int32, device=device)
    base = wp.uint64(n_vertices)
    _launch.launch(
        kernel_remesh.scale_attribute_edge_keys,
        dim=n_faces,
        inputs=[faces, region, base, keys],
        device=device,
    )
    _launch.radix_sort_pairs(
        keys, order, count=n, end_bit=max(1, (2 * n_vertices * n_vertices - 1).bit_length())
    )
    # The sum and the valence share one zeroed buffer.
    sums = _launch.zeros(2 * n_vertices, dtype=wp.float32, device=device)
    total = twt.as_dense(sums[:n_vertices])
    valence = twt.as_dense(sums[n_vertices:])
    _launch.launch(
        kernel_remesh.scatter_scale_attribute,
        dim=n,
        inputs=[vertices, keys, base, total, valence],
        device=device,
    )
    _launch.map(kernel_array.divide_if_positive, total, valence, out=total)
    return total


def _ranks_from_flags(rank_buffer: wp.array[wp.int32]) -> tuple[wp.array[wp.int32], int]:
    """
    Scan the 0/1 flags in ``rank_buffer[1:]`` in place; return the exclusive ranks and their count.

    The buffer is zero-filled and one longer than the flags, so after an inclusive scan of its tail
    the leading zero makes ``rank_buffer[:n]`` the exclusive ranks and ``rank_buffer[n]`` the
    count -- ``array.mask_to_compact_ranks`` for a caller whose marking kernel already wrote the
    flags, which saves converting a mask into them.
    """
    n = rank_buffer.size - 1
    flags = twt.as_dense(rank_buffer[1:])
    _launch.array_scan(flags, flags, inclusive=True)
    # The count sizes the split's outputs and decides whether the loop stops, so it has to come
    # back to the host.
    return twt.as_dense(rank_buffer[:n]), int(read_scalar(rank_buffer))


@overload
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Split a chosen set of edges in one crack-free pass, inserting one vertex per edge.

    The primitive the whole ``subdivide_*`` family is built from, exposed because the *choice* of
    edges and the *position* of the new vertex are the only things that differ between its members:
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] iterates this with a length test and
    midpoints, and
    [`triwarp.intersection.split_mesh_with_plane`][triwarp.intersection.split_mesh_with_plane] calls
    it once with the edges a plane crosses and the crossing points. A caller with a different
    criterion — a curvature threshold, a paint selection, an isovalue — needs no new machinery.

    Crack-free means the new vertex of an edge is inserted **once** and both incident faces
    reference it, so a watertight input stays watertight and no T-junction is introduced. Each face
    is re-triangulated by how many of its three edges were split: 1 gives two triangles, 2 gives
    three (the quad cut along its shorter diagonal), 3 gives the regular 1-to-4 split, and 0 passes
    through unchanged.

    Every return is freshly allocated and independently owned, on the path where nothing was
    flagged and the mesh comes back unchanged as much as on the splitting one -- including
    ``index``, which is a copy of the caller's rather than the array they passed in.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device. Never mutated.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    split_mask
        ``(n_edges,)`` mask over the **unique undirected edges** in
        [`edges_unique`][triwarp.edges.edges_unique] order, ``True`` for each edge to split.
    split_positions
        ``(n_split,)`` positions for the new vertices, ``n_split`` the number of ``True`` entries of
        ``split_mask``, indexed by the **exclusive scan of** ``split_mask`` — that is, in ascending
        unique-edge order among the flagged edges, which is where a kernel writing
        ``out[offsets[e]]`` naturally puts them. ``None`` uses each edge's midpoint.
    unique_edges, inverse
        ``(n_edges, 2)`` and ``(3 * n_faces,)`` [`edges_unique`][triwarp.edges.edges_unique] pair
        for ``faces``, when the caller has already built it to compute ``split_mask``. Both must be
        given together; either being ``None`` rebuilds them.
    index
        ``(n_faces,)`` per-face values to carry through the split: each output face
        receives the value of the input face it came from. ``None`` means the identity, so
        ``return_index`` then reports provenance into ``faces``. Passing the *previous* pass's index
        is how an iterated caller composes provenance without a gather per pass.
    return_index
        If ``True``, also return ``index`` resolved onto the output faces (provenance into ``faces``
        when ``index`` is ``None``).

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_vertices + n_split,)`` original vertices followed by the ``n_split`` inserted ones, in
        ascending edge order.
    new_faces : wp.array[wp.int32]
        ``(3 * m,)`` flat triangle index buffer for the refined mesh, ``m`` the output face count.
    index : wp.array[wp.int32]
        ``(m,)`` index into the **input** ``faces`` of the face each output face came from; only
        when ``return_index`` is ``True``.

    Raises
    ------
    ValueError
        If ``split_mask`` does not have one entry per unique edge, ``split_positions`` does not have
        one entry per flagged edge, or ``index`` does not have one entry per face.
    RuntimeError
        If ``vertices``, ``faces``, ``split_mask``, ``split_positions``, ``unique_edges``,
        ``inverse`` and ``index`` are not all on one device.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`subdivide`][triwarp.remesh.subdivide]
    [`triwarp.intersection.split_mesh_with_plane`][triwarp.intersection.split_mesh_with_plane]
    [`triwarp.edges.edges_unique`][triwarp.edges.edges_unique]

    Examples
    --------
    Splitting *every* edge is the regular 1-to-4 subdivision, so the face count quadruples:

    ```python
    unique_edges, inverse = tw.edges.edges_unique(f)
    every_edge = wp.full(int(unique_edges.shape[0]), True, dtype=wp.bool, device=f.device)
    fine_v, fine_f = tw.remesh.split_edges(
        v, f, every_edge, unique_edges=unique_edges, inverse=inverse
    )
    print(int(fine_f.shape[0]) // 3 == 4 * (int(f.shape[0]) // 3))
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        split_mask=split_mask,
        split_positions=split_positions,
        unique_edges=unique_edges,
        inverse=inverse,
        index=index,
    )
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    # Bound in one expression rather than an ``if`` that reassigns the parameters, so the optional
    # annotations narrow for the type checker without an ``assert``.
    edges, corner_edge = (
        (unique_edges, inverse)
        if unique_edges is not None and inverse is not None
        else tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    )
    n_edges = int(edges.shape[0])
    if split_mask.size != n_edges:
        raise ValueError(
            f"split_mask must have one entry per unique edge ({n_edges}), got {split_mask.size}."
        )

    carried = index if index is not None else tw.array.arange(n_faces, device=device)
    if carried.size != n_faces:
        raise ValueError(f"index must have one entry per face ({n_faces}), got {carried.size}.")

    # The exclusive scan both counts the split edges and assigns each one its new vertex slot, which
    # is the indexing ``split_positions`` is documented against.
    offsets, n_split = tw.array.mask_to_compact_ranks(split_mask)
    new_vertices, new_faces, new_index = _split_ranked_edges(
        vertices,
        faces,
        split_mask,
        offsets,
        n_split,
        edges,
        corner_edge,
        carried,
        index,
        split_positions,
        return_index,
    )
    if return_index:
        return new_vertices, new_faces, new_index
    return new_vertices, new_faces


def _split_ranked_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    n_split: int,
    edges: twt.Array2dInt32,
    corner_edge: wp.array[wp.int32],
    carried: wp.array[wp.int32],
    index: wp.array[wp.int32] | None,
    split_positions: wp.array[wp.vec3] | None,
    return_index: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Split the ranked edges: the body of [`split_edges`][triwarp.remesh.split_edges] past its scan.

    ``offsets`` / ``n_split`` are ``mask_to_compact_ranks(split_mask)``; the two refinement loops
    already hold them for their own bookkeeping, so they hand them in rather than have the scan and
    its count readback run twice per pass. ``carried`` is the provenance to thread through
    (``index`` or a fresh ``arange``); the no-split path copies it when it is the caller's ``index``
    and ``return_index`` asks for it back. The provenance is always the third element;
    ``split_edges`` drops it when not asked.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    n_edges = int(edges.shape[0])
    if n_split == 0 or n_faces == 0:
        # ``carried`` is still the caller's own ``index`` buffer when one was supplied, so it is
        # copied for the same reason ``vertices`` and ``faces`` are: every return is independently
        # owned on the no-split path exactly as on the splitting one. When ``index`` was ``None``
        # the ``arange`` already allocated it fresh.
        owned = _launch.clone(carried) if return_index and carried is index else carried
        return _launch.clone(vertices), _launch.clone(faces), owned

    # One buffer sized for its final use: the originals in the prefix and the new vertices written
    # straight into the tail, so the face emission below resolves every index against it.
    new_vertices = _launch.empty(n_vertices + n_split, dtype=wp.vec3, device=device)
    if n_vertices > 0:
        _launch.copy(new_vertices, vertices, count=n_vertices)
    if split_positions is not None:
        if split_positions.size != n_split:
            raise ValueError(
                f"split_positions must have one entry per flagged edge ({n_split}), "
                f"got {split_positions.size}."
            )
        _launch.copy(new_vertices, split_positions, dest_offset=n_vertices, count=n_split)

    # Each face's child count, scanned, is its first output row -- so the emission writes the
    # compact face buffer directly and the provenance with it, rather than four fixed slots per face
    # and a compaction of both. The same launch writes the midpoints when the caller gave none.
    child_counts = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.split_child_counts_and_midpoints,
        dim=max(n_faces, n_edges),
        inputs=[
            corner_edge,
            split_mask,
            vertices,
            edges,
            offsets,
            wp.int32(1 if split_positions is None else 0),
        ],
        outputs=[child_counts, new_vertices[n_vertices:]],
        device=device,
    )
    face_offsets, n_out_faces = tw.array.counts_to_offsets(child_counts)
    out_faces = twt.empty_2d((n_out_faces, 3), wp.int32, device=device)
    out_index = _launch.empty(n_out_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_remesh.emit_size_faces,
        dim=n_faces,
        inputs=[
            faces,
            corner_edge,
            split_mask,
            offsets,
            wp.int32(n_vertices),
            new_vertices,
            carried,
            face_offsets,
            out_faces,
            out_index,
        ],
        device=device,
    )

    return new_vertices, out_faces.reshape(-1), out_index


def _flip_region_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region_flags: wp.array[wp.int32],
    max_angle_change: float | None,
    max_deviation: float | None,
    max_iter: int,
    topology: _FlipTopology | None = None,
) -> int:
    """
    Run the parallel Delone flip pass over the region, mutating ``faces`` in place.

    ``topology`` is handed through to ``_flip_interior_edges``, which leaves it describing
    ``faces`` as the pass leaves it.
    """
    device = faces.device
    n_vertices = vertices.size
    mac, mdsq, car = _flip_gates(max_angle_change, max_deviation)

    def launch(
        adjacency: twt.Array2dInt32,
        adjacency_edges: twt.Array2dInt32,
        unshared: twt.Array2dInt32,
        sorted_keys: wp.array[wp.uint64],
        edge_set: wp.array[wp.uint64],
        edge_set_mask: wp.int32,
        key_base: wp.uint64,
        out_flip: wp.array[wp.bool],
        out_quad: twt.Array2dInt32,
    ) -> None:
        _launch.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                edge_set,
                edge_set_mask,
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    return _flip_interior_edges(faces, n_vertices, launch, max_iter, topology)


def _flip_gates(
    max_angle_change: float | None,
    max_deviation: float | None,
    critical_aspect_ratio: float = 1000.0,
) -> tuple[wp.float32, wp.float32, wp.float32]:
    """
    Build the three gate scalars ``delone_flip_candidates`` takes from a caller's optional bounds.

    Both flip drivers that reach that kernel --
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] and ``_flip_region_faces`` -- built this
    triple inline and identically, differing only in that the region driver exposes no
    ``critical_aspect_ratio`` and so takes the public default.

    ``None`` means "no gate", and each is disabled by a sentinel the kernel cannot exceed rather
    than by a branch: a full turn for the dihedral change, and a squared deviation near the top of
    ``float32``. The deviation is squared here so the kernel compares against a squared length and
    needs no root per candidate.
    """
    mac = wp.float32(max_angle_change if max_angle_change is not None else float(2.0 * math.pi))
    mdsq = wp.float32(max_deviation * max_deviation if max_deviation is not None else 3.0e38)
    return mac, mdsq, wp.float32(critical_aspect_ratio)
