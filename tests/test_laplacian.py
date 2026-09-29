"""Regression tests for ``triwarp.laplacian`` against igl (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.comparisons import assert_nonconstant
from tests.conftest import MESHES
from tests.conversions import (
    bsr_to_csr,
    bsr_to_dense,
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
)
from triwarp.constants import TOLERANCE_MOLLIFY

# Not ``conftest.MESHES``: this predates that constant and has never carried ``cave_cube``.
_LAPLACIAN_MESHES = ["icosahedron", "half_torus", "hemisphere"]

# -----------------------------------------------------------------------------------------
# face_gradients (libigl `grad`, pyvista `compute_derivative`)
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("face_gradients", "igl")
def test_face_gradients_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (matrix form applied): ``igl.grad`` is the same operator as a sparse ``(3F, V)`` map.

    igl returns the operator; triwarp returns its product with the field. The named transform is
    therefore to *apply* igl's matrix and unstack the result, which comes back as
    ``[all x; all y; all z]`` rather than interleaved -- getting that wrong yields a permutation of
    the right numbers, so the test also checks the defining property below, which no permutation
    satisfies.

    The tolerance is the package's standard ``1e-5`` rather than something tighter, and the reason
    is structural rather than a fudge: triwarp accumulates in ``float64`` but takes its normals and
    areas from ``face_normals_and_areas``, which is ``float32``, so the geometry enters at single
    precision where igl's is double throughout. Measured worst deviation across these fixtures is
    **1.4e-6 relative** (on ``half_torus``, whose faces are the smallest), so the bound has a 7x
    margin -- and it is a float32-vs-float64 gap, not a disagreement about the operator.

    The second assert is the gradient's defining identity, checked without igl:
    ``dot(grad, v1 - v0) == values[v1] - values[v0]`` for every face. A finite difference along the
    edges, or a gradient left in the wrong plane, fails it.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    values_np = np.ascontiguousarray(vertices_np[:, 2])

    n_faces = faces_np.shape[0]
    stacked_igl = igl.grad(vertices_np, faces_np) @ values_np
    gradients_igl = np.stack(
        [stacked_igl[:n_faces], stacked_igl[n_faces : 2 * n_faces], stacked_igl[2 * n_faces :]],
        axis=1,
    )

    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)
    gradients_wp = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, values_wp)

    assert np.allclose(gradients_wp.numpy(), gradients_igl, rtol=1e-5, atol=1e-5)

    # The identity that defines a piecewise-linear gradient, independent of either library.
    edges_np = vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]]
    differences_np = values_np[faces_np[:, 1]] - values_np[faces_np[:, 0]]
    assert np.allclose(
        np.einsum("ij,ij->i", gradients_wp.numpy(), edges_np), differences_np, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("face_gradients", "pyvista")
def test_face_gradients_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (averaged onto vertices): VTK reports the same gradient, at the *points*.

    ``compute_derivative`` returns a per-**point** gradient for a point-data field, and
    ``preference='cell'`` does not move it (measured: the array stays ``(n_vertices, 3)``), so the
    named transform is triwarp's answer averaged onto vertices with
    ``interpolation.average_onto_vertices`` -- which turns out to be exactly what VTK computes,
    element-wise to 1.8e-07 on ``icosphere(3)``.

    The quantity is the **tangential** surface gradient on both sides, which is the thing most
    likely to be misread as a bug: for ``f = x`` on the unit sphere the mean is ``2/3 e_x``, not
    ``e_x``, because the ambient x-direction is only partly tangent to the surface. That value is
    pinned below so the convention is asserted rather than described.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_tm.vertices.shape[0]
    values_np = np.ascontiguousarray(mesh_tm.vertices[:, 2], dtype=np.float64)

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.point_data["field"] = values_np
    gradients_pv = np.asarray(
        mesh_pv.compute_derivative(scalars="field", gradient=True).point_data["gradient"]
    )

    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)
    gradients_np = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, values_wp).numpy()
    averaged_np = np.stack(
        [
            tw.interpolation.average_onto_vertices(
                n_vertices,
                mesh_wp.indices,
                wp.array(
                    np.ascontiguousarray(gradients_np[:, axis], dtype=np.float32),
                    dtype=wp.float32,
                    device=mesh_wp.device,
                ),
            ).numpy()
            for axis in range(3)
        ],
        axis=1,
    )
    assert np.allclose(averaged_np, gradients_pv, rtol=1e-4, atol=1e-4)

    # The gradient is *tangential*, on both sides: 2/3 e_x for f = x on a unit sphere.
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    sphere_pv = trimesh_to_pyvista(sphere_tm)
    sphere_pv.point_data["field"] = np.ascontiguousarray(sphere_tm.vertices[:, 0])
    sphere_gradient_pv = np.asarray(
        sphere_pv.compute_derivative(scalars="field", gradient=True).point_data["gradient"]
    )
    assert np.allclose(sphere_gradient_pv.mean(axis=0), [2.0 / 3.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity(
    "face_gradients",
    "meshlib",
    benchmarked=False,
    reason="gradientInTri takes one triangle's three corners and three values, so "
    "a batched row would be a Python loop over the face buffer and would time the "
    "loop rather than MeshLib -- the same reason section 6 bars a per-element loop "
    "from a benchmark row, and the same call shape as triCenter and "
    "triangleAspectRatio in tests/test_triangles.py. It is nonetheless the only "
    "reference in the suite that returns the gradient itself rather than the "
    "operator, which is what makes it worth comparing here. igl and pyvista carry "
    "the timed rows for this group.",
)
def test_face_gradients_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the *applied* gradient, face by face, against the only reference that returns it.

    igl gives this operator as a sparse ``(3F, V)`` matrix and pyvista as a smoothed point field,
    so both need a named transform before they can be compared (see the two tests above).
    ``gradientInTri`` is the third form -- one triangle's three corners and three values in, one
    vector out -- and needs none: it is the same quantity triwarp returns, in the same units, per
    face.

    Measured on a non-uniformly scaled ``icosphere(2)``: **4.09e-06** absolute, **2.49e-07**
    relative, against gradient norms spanning 16.5 -- so the residual is the float32 vertex buffer
    on both sides (MeshLib stores points in float32 too; section 6 records its own 2.58e-08
    round-trip) rather than a disagreement about the formula.

    The four-argument overload is the one to call: ``gradientInTri(b, c, vb, vc)`` assumes the value
    at ``a`` is zero and returns a different vector, and nothing in the signature says so.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.asarray(mesh_tm.faces)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    rng = np.random.default_rng(7)
    field_np = rng.standard_normal(len(mesh_tm.vertices))

    field_wp = wp.array(field_np, dtype=wp.float64, device=mesh_wp.device)
    gradients_wp = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, field_wp)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    assert mesh_ml.topology.numValidFaces() == len(faces_np) > 0  # non-vacuity
    gradients_ml = np.array(
        [
            [
                *mm.gradientInTri(
                    mm.Vector3f(*vertices_np[a].tolist()),
                    mm.Vector3f(*vertices_np[b].tolist()),
                    mm.Vector3f(*vertices_np[c].tolist()),
                    float(field_np[a]),
                    float(field_np[b]),
                    float(field_np[c]),
                )
            ]
            for a, b, c in faces_np
        ]
    )
    # Non-vacuity: a constant field would make every row zero and pass any tolerance.
    assert_nonconstant(np.linalg.norm(gradients_ml, axis=1), tol=1.0)
    assert np.allclose(gradients_wp.numpy(), gradients_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_gradients_of_a_constant_field_is_zero(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """A constant field has no gradient, and a degenerate face has none either."""
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_wp.points.shape[0])
    constant_wp = wp.array(
        np.full(n_vertices, 3.25, dtype=np.float64), dtype=wp.float64, device=mesh_wp.device
    )
    gradients_wp = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, constant_wp)
    assert np.allclose(gradients_wp.numpy(), 0.0, atol=1e-9)


def test_face_gradients_precomputed_face_data(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Passing the face normals and areas must not change the answer -- only skip recomputing."""
    _mesh_tm, mesh_wp = half_torus
    values_wp = wp.array(
        np.ascontiguousarray(mesh_wp.points.numpy()[:, 2], dtype=np.float64),
        dtype=wp.float64,
        device=mesh_wp.device,
    )
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    derived = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, values_wp)
    supplied = tw.laplacian.face_gradients(
        mesh_wp.points, mesh_wp.indices, values_wp, face_normals=normals_wp, face_areas=areas_wp
    )
    assert np.array_equal(derived.numpy(), supplied.numpy())


def test_face_gradients_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    values_wp = wp.array(np.array([], dtype=np.float64), dtype=wp.float64, device=device)
    assert tw.laplacian.face_gradients(vertices_wp, faces_wp, values_wp).shape == (0,)


# -----------------------------------------------------------------------------------------
# cotmatrix_entries
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("cotmatrix_entries", "igl")
def test_cotmatrix_entries(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the per-face half-cotangent table against ``igl.cotmatrix_entries``, column order too.

    igl uses the same ``(n_faces, 3)`` opposite-corner convention, so no permutation is needed
    -- which makes this the test that pins the column order every downstream assembly depends
    on.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    cot_entries_igl = igl.cotmatrix_entries(vertices_np, faces_np)
    cot_entries_wp = tw.laplacian.cotmatrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity(
    "cotmatrix_entries",
    "meshlib",
    benchmarked=False,
    reason="leftCotan answers one directed edge per call, so a batched row would "
    "be a Python loop over the edge buffer and would price the loop rather than "
    "MeshLib's arithmetic -- the per-element rule from section 6. igl carries the "
    "timed row for this group, and it is the same quantity.",
)
def test_cotmatrix_entries_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (a factor of two, and an edge-to-corner mapping): ``leftCotan`` per directed edge.

    Two named transforms, both of which the loop below performs rather than assumes. MeshLib keys
    the weight by the *directed edge* whose left face owns it, so the ``(face, corner)`` slot is
    recovered from the edge's endpoints -- the corner is the face vertex that is neither
    ``org(e)`` nor ``dest(e)``. And ``leftCotan`` is the plain cotangent where triwarp's table
    holds **half** of it (igl's convention), so the comparison carries the factor 2.

    Measured on the three fixtures: **2.38e-07** absolute, 1.35e-07 relative. The loop also asserts
    it filled every slot, which is what makes the mapping a claim rather than a hope -- a mapping
    that silently missed a corner would leave a zero there and still pass an ``allclose`` on a
    nearly-flat mesh.

    ``cotan(topology, points, ue)`` is *not* the pairing: it is the sum of the two ``leftCotan``
    values across an undirected edge, i.e. the assembled off-diagonal, which is what
    [`test_cotmatrix`] compares through igl's matrix.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.asarray(mesh_tm.faces)
    entries_wp = tw.laplacian.cotmatrix_entries(mesh_wp.points, mesh_wp.indices).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml, points_ml = mesh_ml.topology, mesh_ml.points
    cot_ml = np.zeros_like(entries_wp)
    filled = np.zeros(entries_wp.shape, dtype=bool)
    for undirected in range(topology_ml.undirectedEdgeSize()):
        for edge_ml in (mm.EdgeId(2 * undirected), mm.EdgeId(2 * undirected + 1)):
            face_ml = topology_ml.left(edge_ml)
            if not face_ml.valid():
                continue  # a boundary edge has no left face, so it owns no corner
            org, dest = int(topology_ml.org(edge_ml)), int(topology_ml.dest(edge_ml))
            row = faces_np[int(face_ml)]
            corner = int(np.flatnonzero((row != org) & (row != dest))[0])
            cot_ml[int(face_ml), corner] = mm.leftCotan(topology_ml, points_ml, edge_ml)
            filled[int(face_ml), corner] = True

    assert filled.all()  # the mapping reached every (face, corner) slot
    # Non-vacuity: the table must carry real weights, and on the two irregular fixtures it must
    # vary. The icosahedron is regular, so its 0.2887 is constant by construction and a spread
    # check there would be testing the fixture rather than the function. Individual entries do sit
    # near zero -- a right-angled corner has cotangent 0 -- so only the maximum is bounded away.
    assert np.abs(entries_wp).max() > 0.1
    if mesh_name != "icosahedron":
        assert_nonconstant(entries_wp, tol=0.1)
    assert np.allclose(2.0 * entries_wp, cot_ml, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# cotmatrix_entries_intrinsic
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("cotmatrix_entries_intrinsic", "igl")
def test_cotmatrix_entries_intrinsic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the length-only overload, fed igl's own ``edge_lengths`` so only the formula differs.

    Passing the reference's lengths in rather than triwarp's isolates the cotangent formula
    from [`triwarp.edges`], which has its own oracle. igl overloads the same name on the
    argument shape.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    edge_lengths_igl = igl.edge_lengths(vertices_np, faces_np)
    cot_entries_igl = igl.cotmatrix_entries(edge_lengths_igl)
    edge_lengths_wp = wp.array(
        edge_lengths_igl.astype(np.float32), dtype=wp.float32, device=mesh_wp.device
    )
    cot_entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths_wp)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


def test_cotmatrix_entries_intrinsic_float64(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on the ``float64`` overload: the dtype must change without the values changing.

    The dtype is asserted as well as the values, because a silent ``float32`` return would
    still pass the ``1e-5`` comparison and lose precision only where it matters -- in a
    downstream solve.
    """
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    edge_lengths_igl = igl.edge_lengths(vertices_np, faces_np)
    edge_lengths_wp = wp.array(
        edge_lengths_igl.astype(np.float32), dtype=wp.float32, device=mesh_wp.device
    )
    cot_entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths_wp, dtype=wp.float64)

    assert cot_entries_wp.dtype == wp.float64
    cot_entries_igl = igl.cotmatrix_entries(edge_lengths_igl)
    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# cotmatrix (and the mass matrix it is compared alongside)
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("cotmatrix", "igl")
def test_cotmatrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the assembled cotangent operator against ``igl.cotmatrix``, dense and elementwise.

    Both are densified first, because the two builds order their CSR entries differently while
    holding the same matrix. This is the igl sign convention (negative diagonal), which the
    module's other operators deliberately do not share.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))

    assert laplacian_wp.shape == laplacian_igl.shape
    assert np.allclose(laplacian_wp.toarray(), laplacian_igl.toarray(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("cotmatrix", "potpourri3d")
@pytest.mark.parity("mass_matrix_entries", "potpourri3d")
def test_cotmatrix_and_mass_match_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The two operator builds against geometry-central's bindings, which already agree with libigl.

    A third independent implementation of quantities libigl already pins is worth having precisely
    because it is cheap: these are the operators every solver in the library is built on, so a
    regression here surfaces as a wrong answer several modules away.

    ``vertex_areas`` is Class A: it is one third of the incident face areas, which is exactly the
    barycentric lumped mass diagonal ``mass_matrix_entries`` returns.

    ``cotan_laplacian`` is Class B, and the transform is a **sign flip**. geometry-central builds
    the positive-semidefinite Laplacian while libigl -- and triwarp with it -- builds the negative
    one: measured on ``icosahedron``, ``pp3d.cotan_laplacian`` is ``-igl.cotmatrix`` to 1e-9 entry
    for entry, with a ``+2.887`` diagonal against igl's ``-2.887``. Neither is wrong, but handing
    one to a solver expecting the other flips the sign of every diffusion step, so the negation
    here is the substance of the comparison rather than bookkeeping.

    Read potpourri3d as the *weakest* of the three references rather than the strongest: it
    assembles both of these in vectorized numpy into a scipy COO, not in geometry-central's C++, so
    it is closer to an independent re-derivation of the same formula than to a separate codebase.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)

    cotmatrix_pp = pp3d.cotan_laplacian(vertices_np, faces_np).tocsr()
    cotmatrix_wp = bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))
    assert cotmatrix_wp.shape == cotmatrix_pp.shape
    assert np.allclose(cotmatrix_wp.toarray(), -cotmatrix_pp.toarray(), rtol=1e-5, atol=1e-5)

    mass_pp = pp3d.vertex_areas(vertices_np, faces_np)
    mass_wp = tw.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(mass_wp.numpy(), mass_pp, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("cotmatrix", "pytorch3d")
@pytest.mark.parity("mass_matrix", "pytorch3d")
def test_cotmatrix_and_mass_match_pytorch3d(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: ``cot_laplacian`` returns **twice** triwarp's off-diagonal and a **zero** diagonal.

    Two conventions in one call and neither is guessable from the name. The off-diagonal ratio is
    measured at exactly ``0.5`` -- triwarp keeps the *half*-cotangent table that
    ``cotmatrix_entries`` produces, pytorch3d assembles the sum -- and pytorch3d never writes the
    row sum onto the diagonal at all (``max|diag|`` is ``0.0``, not merely small), where triwarp's
    ``cotmatrix`` assembles it. So the comparison is off-diagonals only, and the zero diagonal is
    asserted rather than sidestepped, because it is what a caller assembling from
    ``cot_laplacian`` has to know: section 4's ``laplacian_smoothing_loss`` variants need that
    matrix and not this one.

    Its second return value is the mass matrix in the reciprocal: ``1 / inv_areas`` is **three
    times** triwarp's ``mass_matrix`` diagonal, because the barycentric lumped mass is a third of
    the incident area sum. Measured 5.96e-08 absolute / 2.45e-07 relative.
    """
    mesh_tm, mesh_wp = icosphere
    n_vertices = len(mesh_tm.vertices)
    mesh_p3d = trimesh_to_pytorch3d(mesh_tm)
    cotangent_p3d, inv_areas_p3d = p3d_ops.cot_laplacian(
        mesh_p3d.verts_packed(), mesh_p3d.faces_packed()
    )
    cotangent_p3d = cotangent_p3d.to_dense().numpy()
    inv_areas_p3d = inv_areas_p3d.numpy().reshape(-1)

    cotangent_np = bsr_to_dense(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices), n_vertices)
    mass_np = bsr_to_dense(tw.laplacian.mass_matrix(mesh_wp.points, mesh_wp.indices), n_vertices)

    off_diagonal = ~np.eye(n_vertices, dtype=bool)
    assert float(np.abs(np.diag(cotangent_p3d)).max()) == 0.0
    assert float(np.abs(cotangent_p3d[off_diagonal]).max()) > 1e-2
    assert np.allclose(
        cotangent_np[off_diagonal], 0.5 * cotangent_p3d[off_diagonal], rtol=1e-5, atol=1e-6
    )
    assert np.allclose(1.0 / inv_areas_p3d, 3.0 * np.diag(mass_np), rtol=1e-5, atol=1e-6)


@pytest.mark.parity("laplacian_equal_weight", "pytorch3d")
@pytest.mark.parity("laplacian_inverse_distance", "pytorch3d")
def test_laplacian_operators_match_pytorch3d(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: the two ``ops`` graph Laplacians are triwarp's two ``equal_weight`` branches.

    Both transforms were recovered by measurement rather than read off the docstrings. For the
    uniform operator, pytorch3d writes **-1** on the diagonal where triwarp writes 0, so
    ``L_p3d == L_tw - I``; the off-diagonals are bit-identical (**0.0**), which is what makes the
    diagonal claim a claim about the convention and not a tolerance. For the inverse-distance one,
    pytorch3d leaves the rows *unnormalized* -- it is ``1 / (||vi - vj|| + 1e-12)`` raw, the same
    formula and the same literal ``eps`` triwarp's docstring quotes -- so dividing by the row sum
    is the transform, and the two then agree to 1.49e-08 including the diagonal, which is 0 on
    both sides.

    That second measurement is why there is no ``edge_weights=`` keyword to add here: pytorch3d's
    ``norm_laplacian`` *is* ``laplacian(equal_weight=False)``, and ``filter_taubin`` already takes
    it through ``laplacian_operator=``.
    """
    mesh_tm, mesh_wp = icosphere
    n_vertices = len(mesh_tm.vertices)
    mesh_p3d = trimesh_to_pytorch3d(mesh_tm)
    uniform_p3d = (
        p3d_ops.laplacian(mesh_p3d.verts_packed(), mesh_p3d.edges_packed()).to_dense().numpy()
    )
    inverse_p3d = (
        p3d_ops.norm_laplacian(mesh_p3d.verts_packed(), mesh_p3d.edges_packed()).to_dense().numpy()
    )
    row_sums_p3d = inverse_p3d.sum(axis=1, keepdims=True)
    normalized_p3d = inverse_p3d / np.where(row_sums_p3d == 0.0, 1.0, row_sums_p3d)

    uniform_np = bsr_to_dense(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=True), n_vertices
    )
    inverse_np = bsr_to_dense(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=False), n_vertices
    )

    assert np.array_equal(np.diag(uniform_p3d), np.full(n_vertices, -1.0, dtype=np.float32))
    assert np.array_equal(uniform_np, uniform_p3d + np.eye(n_vertices))
    assert float(np.abs(np.diag(inverse_p3d)).max()) == 0.0
    assert np.allclose(inverse_np, normalized_p3d, rtol=1e-5, atol=1e-7)


def test_cotmatrix_null_space(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the constant vector must be in the operator's null space.

    Both sides are checked against the *property* rather than against each other -- igl at
    1e-10 in float64 and triwarp at 1e-4 in float32 -- which is what makes the differing
    thresholds honest rather than a hidden tolerance. Catches a row that does not sum to zero,
    which a matrix comparison at 1e-5 can miss on a large-valued row.
    """
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    ones = np.ones(vertices_np.shape[0], dtype=np.float64)
    assert np.linalg.norm(laplacian_igl @ ones) < 1e-10

    laplacian_wp = bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))
    ones_wp = np.ones(int(mesh_wp.points.shape[0]), dtype=np.float32)
    assert np.linalg.norm(laplacian_wp @ ones_wp) < 1e-4


def test_cotmatrix_empty_mesh(device: str) -> None:
    """
    Class A on the degenerate case: three vertices and no faces give an empty operator, sized 3x3.

    The *shape* is the claim, not the values: igl sizes by ``len(V)`` and so must triwarp,
    rather than returning a 0x0 matrix that would break a caller's dimensions.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.empty((0, 3), dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    laplacian_wp = bsr_to_csr(tw.laplacian.cotmatrix(vertices_wp, faces_wp))

    assert laplacian_wp.shape == laplacian_igl.shape == (3, 3)
    assert laplacian_wp.nnz == 0
    assert laplacian_igl.nnz == 0


@pytest.mark.parametrize(
    "operator",
    [
        "cotmatrix",
        "connection_laplacian",
        "graph_laplacian",
        "laplacian_uniform",
        "laplacian_inverse",
    ],
)
def test_mesh_operator_pattern_builds_agree(
    operator: str, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """
    Triwarp against triwarp: the directed and undirected pattern builds give the same matrix.

    ``_mesh_operator_pattern`` switches builds at ``_UNDIRECTED_PATTERN_FROM_FACES`` and every
    fixture here sits below it, so the threshold is forced each way. ``test_cotmatrix`` carries the
    igl oracle for the directed build; this pins the other to it bit for bit -- offsets, columns
    and values. The mesh carries an unreferenced vertex and a degenerate face -- whose self-edge the
    symmetric ``laplacian`` keeps as a diagonal entry, which the undirected build flags rather than
    sorts -- except for the connection Laplacian, which requires an edge-manifold mesh.
    """
    sphere = tm.creation.icosphere(3)
    vertices_np, faces_np = sphere.vertices, sphere.faces
    if operator != "connection_laplacian":
        vertices_np = np.vstack([vertices_np, [[3.0, 3.0, 3.0]]])
        faces_np = np.vstack([faces_np, [[0, 0, 1]]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    builds = {
        "laplacian_uniform": lambda v, f: tw.laplacian.laplacian(v, f, symmetric=True),
        "laplacian_inverse": lambda v, f: tw.laplacian.laplacian(v, f, equal_weight=False),
    }
    build = builds.get(operator) or getattr(tw.laplacian, operator)
    arrays = []
    for threshold in (10**12, 0):
        monkeypatch.setattr(tw.laplacian, "_UNDIRECTED_PATTERN_FROM_FACES", threshold)
        matrix = build(vertices_wp, faces_wp)
        n_entries = matrix.nnz_sync()
        arrays.append(
            (
                matrix.offsets.numpy(),
                matrix.columns.numpy()[:n_entries],
                matrix.values.numpy()[:n_entries],
            )
        )
    directed, undirected = arrays
    assert directed[0][-1] > vertices_np.shape[0]  # a real pattern, not an empty one
    for directed_np, undirected_np in zip(directed, undirected, strict=True):
        assert np.array_equal(directed_np, undirected_np)


# -----------------------------------------------------------------------------------------
# robust_laplacian (libigl reference)
#
# test_cotmatrix_entries_are_zero_for_a_zero_area_face is here rather than with cotmatrix_entries:
# its subject is the degeneracy that motivates this path, not the entry formula.
# -----------------------------------------------------------------------------------------


# --- robust_laplacian / mollify_intrinsic (libigl reference) ---------------------------
@pytest.mark.parametrize("mesh_name", MESHES)
def test_robust_laplacian_is_unchanged_on_a_clean_mesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    robust = tw.laplacian.robust_laplacian(
        mesh_wp.points, mesh_wp.indices, use_intrinsic_delaunay=False
    )
    plain = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices)

    # Mollification adds nothing when every triangle is already non-degenerate, so with the flips
    # turned off the operator must be the ordinary one up to the intrinsic route's rounding.
    assert np.allclose(
        bsr_to_dense(robust, n_vertices), bsr_to_dense(plain, n_vertices), rtol=1e-4, atol=1e-5
    )


def test_robust_laplacian_keeps_couplings_the_plain_one_drops(sliver_patch: tuple) -> None:
    """
    Mollification's purpose, on a mesh with a triangle too thin to have cotangents.

    Both operators are *finite* -- ``cot_entries_from_l2`` refuses to divide by a zero area, so a
    degenerate face contributes nothing rather than an infinity. Contributing nothing is the only
    finite choice available (a zero-area triangle's angles are 0 or pi), but it is not free: that
    face's edge couplings vanish from the operator, which is what mollification exists to avoid.
    Here vertices 0 and 1 share an edge of the degenerate face and end up **uncoupled** in the
    plain operator while the mollified one couples them.
    """
    vertices_np, _, vertices_wp, faces_wp = sliver_patch
    n_vertices = len(vertices_np)
    plain = bsr_to_dense(tw.laplacian.cotmatrix(vertices_wp, faces_wp), n_vertices)
    robust = bsr_to_dense(
        tw.laplacian.robust_laplacian(vertices_wp, faces_wp, use_intrinsic_delaunay=False),
        n_vertices,
    )

    assert np.isfinite(plain).all()
    assert np.isfinite(robust).all()

    # The coupling the degenerate face should have provided.
    assert plain[0, 1] == 0.0
    assert robust[0, 1] != 0.0
    off_diagonal = ~np.eye(n_vertices, dtype=bool)
    assert (np.abs(plain[off_diagonal]) > 0).sum() < (np.abs(robust[off_diagonal]) > 0).sum()


def test_robust_laplacian_barely_perturbs_the_rows_it_did_not_need_to(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Triwarp against triwarp: the price mollification charges the *undegenerate* part of the mesh.

    ``cotmatrix`` carries the oracle here -- it is what
    ``test_cotmatrix`` pins against ``igl.cotmatrix`` -- and this bounds how far
    ``robust_laplacian`` may drift from it on the rows that no degenerate face touches.

    ``delta`` is one **global** constant added to every edge length, so ``TOLERANCE_MOLLIFY`` buys
    a finite cotangent on the bad face by perturbing the whole mesh, and the size of that
    perturbation is what bounds the constant from *above*. Without this the constant is guarded on
    one side only: raising it 1000x (to ``1e-2``) leaves the whole of ``test_laplacian``,
    ``test_heat`` and ``test_remesh`` green, because every other fixture in those files is clean
    enough that ``delta`` comes out zero and the constant is never exercised at all.

    The bar bites on a 10x raise, not a 3x one: the deviation is linear in epsilon (~0.66 * eps
    here), so ``3e-5`` sits ~4.4x above the shipped value's measured 6.8e-06 and is crossed once
    epsilon reaches ~4.5e-5. The companion lower bound is
    ``test_robust_laplacian_keeps_couplings_the_plain_one_drops``, which fails if the constant is
    *lowered* until ``delta`` no longer survives float32 storage.
    """
    mesh_tm, _ = icosphere_coarse
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64).copy()
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32)
    # Collapse face 0 exactly, by snapping its apex onto the midpoint of its own opposite edge.
    apex_a, apex_b, apex_c = (int(index) for index in faces_np[0])
    vertices_np[apex_c] = 0.5 * (vertices_np[apex_a] + vertices_np[apex_b])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.reshape(-1), device)
    n_vertices = len(vertices_np)

    _lengths, delta = tw.laplacian.mollify_intrinsic(vertices_wp, faces_wp)
    # Non-vacuity: if the collapse did not actually break the inequality there is nothing to price.
    assert delta > 0.0

    plain = bsr_to_dense(tw.laplacian.cotmatrix(vertices_wp, faces_wp), n_vertices)
    robust = bsr_to_dense(
        tw.laplacian.robust_laplacian(vertices_wp, faces_wp, use_intrinsic_delaunay=False),
        n_vertices,
    )
    clean = np.setdiff1d(np.arange(n_vertices), np.unique(faces_np[0]))
    block = np.ix_(clean, clean)
    scale = np.abs(plain[block]).max()
    assert scale > 0.0

    deviation = np.abs(robust[block] - plain[block]).max() / scale
    assert deviation < 3e-5


def test_cotmatrix_entries_are_zero_for_a_zero_area_face(device: str) -> None:
    """
    A collinear triangle yields zero weights, not infinities.

    Regression guard for the division in ``cot_entries_from_l2``: its denominator is
    ``4 * doublearea``, and ``doublearea_from_lengths`` deliberately reports ``0.0`` for a
    degenerate triangle, so an unguarded divide sends the whole assembled operator -- and any
    solve against it -- to NaN off a single bad face.
    """
    collinear_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)

    entries_wp = tw.laplacian.cotmatrix_entries(collinear_wp, faces_wp)
    assert np.array_equal(entries_wp.numpy(), np.zeros((1, 3), dtype=np.float32))
    assert np.isfinite(bsr_to_dense(tw.laplacian.cotmatrix(collinear_wp, faces_wp), 3)).all()


@pytest.mark.parity(
    "mollify_intrinsic",
    "igl",
    benchmarked=False,
    reason="libigl binds no mollification: igl.cotmatrix_intrinsic takes the length table as "
    "given, so there is nothing on its side to time. What it can still check is the table "
    "mollify_intrinsic emits -- delta is asserted zero on these clean fixtures, and igl's "
    "intrinsic assembly over those lengths then confirms they are the plain edge lengths.",
)
@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_robust_laplacian_matches_igl_intrinsic_assembly(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A: the intrinsic assembly against ``igl.cotmatrix_intrinsic`` on identical lengths.

    ``delta == 0.0`` is asserted first, which is what makes this a comparison of the *assembly*
    alone: on a clean mesh mollification is the identity, so both sides see the same length
    table and any difference is in how the entries are placed.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths_wp, delta = tw.laplacian.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)
    assert delta == 0.0  # these fixtures are clean, so the comparison is against the plain lengths

    # igl takes the same (n_faces, 3) opposite-edge-length table, which pins the column order.
    laplacian_igl = igl.cotmatrix_intrinsic(
        np.ascontiguousarray(lengths_wp.numpy(), dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(lengths_wp)
    laplacian_wp = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, cot_entries=entries_wp)

    dense_igl = np.asarray(laplacian_igl.todense())
    assert np.allclose(
        bsr_to_dense(laplacian_wp, len(mesh_tm.vertices)), dense_igl, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
@pytest.mark.parity("robust_laplacian", "igl")
def test_robust_laplacian_matches_igl_intrinsic_delaunay(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A against ``igl.intrinsic_delaunay_cotmatrix`` -- the strongest oracle in this module.

    It pins the flip's new-edge-length formula and its winding bookkeeping at once, and it can
    be elementwise because the intrinsic Delaunay triangulation is *unique*: the two
    implementations need not perform the same flips in the same order to agree on the matrix.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)

    # With the flips on, this *is* ``igl::intrinsic_delaunay_cotmatrix`` — the strongest oracle in
    # the module, and one that pins the flip's new-edge-length formula and its winding bookkeeping
    # at once. The intrinsic Delaunay triangulation is unique, so the two implementations need not
    # (and do not) perform the same flips in the same order to agree on the matrix.
    laplacian_igl, _, _ = igl.intrinsic_delaunay_cotmatrix(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    laplacian_wp = tw.laplacian.robust_laplacian(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        bsr_to_dense(laplacian_wp, n_vertices),
        np.asarray(laplacian_igl.todense()),
        rtol=1e-4,
        atol=1e-5,
    )


# -----------------------------------------------------------------------------------------
# mollify_intrinsic
# -----------------------------------------------------------------------------------------


def test_mollify_intrinsic_is_a_no_op_on_a_clean_mesh(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    original = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices)
    mollified, delta = tw.laplacian.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)

    assert delta == 0.0
    assert np.array_equal(mollified.numpy(), original.numpy())


def test_mollify_intrinsic_restores_the_triangle_inequality(sliver_patch: tuple) -> None:
    _, _, vertices_wp, faces_wp = sliver_patch
    original = tw.edges.face_edge_lengths(vertices_wp, faces_wp).numpy()
    mollified, delta = tw.laplacian.mollify_intrinsic(vertices_wp, faces_wp)

    assert delta > 0.0

    # Every mollified triangle satisfies the inequality strictly; at least one original did not.
    def worst_slack(lengths: np.ndarray) -> np.ndarray:
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        return np.minimum(np.minimum(a + b - c, b + c - a), c + a - b)

    assert worst_slack(original).min() <= 0.0
    # Against the *documented* margin, not merely against zero. ``> 0.0`` alone passes for any
    # positive ``delta`` and so cannot tell a correct mollification from one that adds half of what
    # the inequality needs -- which is exactly the defect this assert was rewritten to catch.
    # ``mollify_intrinsic`` promises ``epsilon * mean_edge_length``; the slack it delivers is
    # ``delta`` above the original's, and the float32 sum of two lengths of ~1 with a ~1e-5 delta
    # is what sets the 5 % slop.
    #
    # That slop is also what makes this the *lower* guard on ``TOLERANCE_MOLLIFY``: at 1e-6 the
    # constant is only ~13 ULP of the mean edge, so rounding the stored sum eats far more than 5 %
    # of it and this assert fails -- which is the float32 floor
    # ``kernels/laplacian.triangle_inequality_slack`` documents, caught rather than reasoned about.
    margin = TOLERANCE_MOLLIFY * float(original.mean())
    assert worst_slack(mollified.numpy()).min() >= margin * (1.0 - 5e-2)
    # One global constant, added to every length: that is what keeps the operator symmetric. The
    # tolerance is loose because ``delta`` here is ~1e-5 against lengths of ~1, so the sum lands at
    # the edge of float32's resolution.
    assert np.allclose(mollified.numpy() - original, delta, rtol=5e-2, atol=1e-9)


# -----------------------------------------------------------------------------------------
# connection_laplacian
# -----------------------------------------------------------------------------------------


# --- connection_laplacian / laplacian_entries -------------------------------------------


def _dense_blocks_2x2(matrix: object, n_vertices: int) -> np.ndarray:
    """
    Densify a ``(n, n)`` matrix of ``mat22d`` blocks into a plain ``(2n, 2n)`` array.

    [`tests.conversions.bsr_to_dense`][] writes one scalar per entry and so cannot read a blocked
    matrix; the row-offset walk is the same, and is still the only safe way in (``values`` is
    allocated at the triplet count).
    """
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    dense = np.zeros((2 * n_vertices, 2 * n_vertices))
    for row in range(n_vertices):
        for slot in range(offsets[row], offsets[row + 1]):
            column = columns[slot]
            dense[2 * row : 2 * row + 2, 2 * column : 2 * column + 2] = values[slot]
    return dense


def _connection_complex(matrix, n_vertices: int) -> np.ndarray:
    """
    Fold the ``2 x 2`` real blocks into one complex matrix, the form potpourri3d returns.

    Each block is a rotation-scale, i.e. ``[[a, -b], [b, a]]``, which is complex multiplication by
    ``a + ib``. That is asserted rather than assumed by the caller, since the fold is only valid in
    that form.
    """
    nnz = matrix.nnz_sync()
    offsets = matrix.offsets.numpy()[: n_vertices + 1]
    values = matrix.values.numpy()[:nnz]
    rows = np.repeat(np.arange(n_vertices), np.diff(offsets))
    entries = values[:, 0, 0] + 1j * values[:, 1, 0]
    dense = np.zeros((n_vertices, n_vertices), dtype=complex)
    dense[rows, matrix.columns.numpy()[:nnz]] = entries
    return dense


# ``cave_cube`` and ``half_torus`` cannot be used here, for the reason recorded at the top of
# ``tests/test_tangent_space.py``: every face of a diagonal-split quad grid carries an edge whose
# two opposite angles are right angles, so that edge's cotangent weight is exactly zero and its
# phase is erased from the operator -- a limit of the oracle, not of the computation.
@pytest.mark.parity(
    "connection_laplacian",
    "potpourri3d",
    benchmarked=False,
    reason="the operator is only reachable through MeshVectorHeatSolver, whose construction builds "
    "a halfedge mesh, tangent frames and a factorization before get_connection_laplacian returns a "
    "cached matrix -- so a row would price that build rather than the assembly, exactly as in the "
    "halfedge_transport_angles claim. The operator itself is what is comparable.",
)
@pytest.mark.parametrize("mesh_name", ["icosahedron", "icosphere_coarse", "hemisphere"])
def test_connection_laplacian_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class B (block fold, then gauge invariants): the same operator, up to the tangent-frame gauge.

    Two named transforms. The first is representational -- triwarp stores real ``2 x 2``
    rotation-scale blocks where potpourri3d stores complex scalars, and ``_connection_complex``
    folds one into the other (asserted exactly: ``v01 == -v10`` and ``v00 == v11`` to ``0.0``).

    The second is the real content. A connection Laplacian is defined only up to a choice of
    reference direction per vertex, and the two libraries choose differently, so **individual
    entries are not comparable** -- CLAUDE.md section 7.6's rule about tangent-space quantities.
    What a gauge change cannot touch is a diagonal unitary conjugation's invariants, and two are
    checked here: the entry **magnitudes** ``|L_ij|``, which are the cotangent weights, and the
    **whole spectrum**, since ``L`` is Hermitian and conjugation by a unitary preserves eigenvalues.

    That makes this stronger than a correlation bound, which is why it is Class B rather than the
    Class C the plan projected: measured max abs difference 3.7e-07 on the magnitudes and 2.7e-07 on
    the eigenvalues (5.1e-08 relative) on ``icosphere_coarse``, i.e. at triwarp's float32 vertex
    floor. The bug class it excludes is a wrong weight, a wrong sparsity pattern, a non-Hermitian
    assembly, and any phase error large enough to move the spectrum -- a *global* gauge shift is the
    one thing it cannot see, and that is because a global gauge shift is not an error.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])

    matrix = tw.laplacian.connection_laplacian(mesh_wp.points, mesh_wp.indices)
    blocks = matrix.values.numpy()[: matrix.nnz_sync()]
    # The fold below is only valid for a rotation-scale block; pin that rather than trusting it.
    assert np.array_equal(blocks[:, 0, 1], -blocks[:, 1, 0])
    assert np.array_equal(blocks[:, 0, 0], blocks[:, 1, 1])
    connection_wp = _connection_complex(matrix, n_vertices)

    connection_pp = np.asarray(
        pp3d.MeshVectorHeatSolver(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            use_intrinsic_delaunay=False,
        )
        .get_connection_laplacian()
        .todense()
    )

    # Non-vacuity: a comparison of two all-zero operators would pass everything below.
    assert np.abs(connection_pp).max() > 1e-3
    assert np.abs(connection_wp - connection_wp.conj().T).max() < 1e-9

    # Gauge-invariant 1: the magnitudes are the cotangent weights, phases removed.
    assert np.allclose(np.abs(connection_wp), np.abs(connection_pp), rtol=1e-4, atol=1e-5)
    # Gauge-invariant 2: the spectrum, since a gauge change is a diagonal unitary conjugation.
    assert np.allclose(
        np.linalg.eigvalsh(connection_wp), np.linalg.eigvalsh(connection_pp), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "half_torus", "hemisphere"])
def test_connection_laplacian_with_zero_transport_is_the_cotangent_laplacian(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: with every transport angle zeroed, each ``2 x 2`` block collapses to ``-w * I``.

    The rotations are the *only* thing separating this operator from
    [`cotmatrix`][triwarp.laplacian.cotmatrix] -- same weights, same sparsity -- so feeding it a
    zero angle per halfedge has to reproduce that matrix exactly, in both diagonal components and
    with nothing off-diagonal inside a block. Measured on ``icosphere_coarse``: the intra-block
    off-diagonals are identically ``0.0``, the two diagonal components agree bit-for-bit, and each
    equals ``-cotmatrix`` to 5.7e-07 (the sign flip is documented -- this operator is assembled
    positive semi-definite for its CG consumers, against ``cotmatrix``'s igl convention).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])
    zero_angles_wp = wp.zeros(
        int(mesh_wp.indices.shape[0]), dtype=wp.float32, device=mesh_wp.points.device
    )

    connection_np = _dense_blocks_2x2(
        tw.laplacian.connection_laplacian(
            mesh_wp.points, mesh_wp.indices, transport_angles=zero_angles_wp
        ),
        n_vertices,
    )
    cotmatrix_np = bsr_to_dense(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices), n_vertices)

    assert np.array_equal(connection_np[0::2, 1::2], np.zeros((n_vertices, n_vertices)))
    assert np.array_equal(connection_np[1::2, 0::2], np.zeros((n_vertices, n_vertices)))
    assert np.array_equal(connection_np[0::2, 0::2], connection_np[1::2, 1::2])
    assert np.allclose(connection_np[0::2, 0::2], -cotmatrix_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosphere_coarse", "half_torus", "hemisphere"])
def test_connection_laplacian_is_symmetric_psd_and_a_rotation_per_block(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The three properties its docstring promises, and the one comparison the gauge permits.

    A single ``2 x 2`` block is gauge-dependent -- it re-expresses a vector in the neighbour's
    frame, and that frame is fixed by a convention no reference library shares -- so §6 rules out an
    element-wise oracle here (see ``test_tangent_space.py``, where the holonomy around a face is
    what can be compared). What is gauge-*invariant* is the block's spectral norm: the rotation is
    orthogonal, so ``‖block‖`` must be the cotangent weight itself, whatever frame it maps between.
    Measured on ``icosphere_coarse``: agreement with ``|cotmatrix|`` to 5.7e-07, exact symmetry, a
    minimum diagonal of 3.43 and a minimum eigenvalue of 7.6e-02.

    The PSD clause is a dense ``np.linalg.eigvalsh`` on a ``2n x 2n`` matrix, and replacing it with
    a Cholesky of ``A + eps*I`` was measured and declined. It reports the *margin* (7.6e-02 above),
    which a factorization does not, and it is no longer expensive: with ``conftest.py`` capping
    OpenBLAS to 8 threads this stopped being the most expensive non-Poisson test in the suite on
    either device. The ``eigvalsh`` behind ``half_torus`` alone is orders of magnitude slower at one
    thread per core than at eight -- the cost is the thread count, not the algorithm.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])
    connection = tw.laplacian.connection_laplacian(mesh_wp.points, mesh_wp.indices)
    cotmatrix = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices)

    # Same sparsity as the scalar operator: it is the same stencil with a rotation per entry.
    assert np.array_equal(connection.offsets.numpy(), cotmatrix.offsets.numpy())
    nnz = connection.nnz_sync()
    assert nnz == cotmatrix.nnz_sync()
    assert np.array_equal(
        connection.columns.numpy()[:nnz], cotmatrix.columns.numpy()[: cotmatrix.nnz_sync()]
    )

    connection_np = _dense_blocks_2x2(connection, n_vertices)
    assert np.array_equal(connection_np, connection_np.T)
    assert np.diag(connection_np).min() > 0.0
    assert np.linalg.eigvalsh(connection_np).min() > -1e-8

    block_norms_np = np.linalg.norm(connection.values.numpy()[:nnz], ord=2, axis=(1, 2))
    cotmatrix_np = bsr_to_dense(cotmatrix, n_vertices)
    rows_np = np.repeat(
        np.arange(n_vertices), np.diff(connection.offsets.numpy()[: n_vertices + 1])
    )
    weights_np = np.abs(cotmatrix_np[rows_np, connection.columns.numpy()[:nnz]])
    assert np.allclose(block_norms_np, weights_np, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# laplacian_entries
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("equal_weight", [True, False])
@pytest.mark.parametrize("symmetric", [True, False])
def test_laplacian_entries_assemble_into_the_laplacian(
    half_torus: tuple[tm.Trimesh, wp.Mesh], equal_weight: bool, symmetric: bool
) -> None:
    """
    Class A: the unassembled triplets are the row-normalized entries ``laplacian`` builds from.

    ``laplacian_entries`` is the umbrella analogue of ``cotmatrix_entries`` and its only in-repo
    caller is ``laplacian`` itself, so the contract worth pinning is that the two agree: summing
    the triplets per row and dividing by the row's total weight has to reproduce the assembled
    operator, for each of the four ``(equal_weight, symmetric)`` combinations.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = int(mesh_tm.vertices.shape[0])
    rows_wp, cols_wp, vals_wp = tw.laplacian.laplacian_entries(
        mesh_wp.points, mesh_wp.indices, equal_weight=equal_weight, symmetric=symmetric
    )
    rows_np, cols_np, vals_np = rows_wp.numpy(), cols_wp.numpy(), vals_wp.numpy()
    assert rows_np.size > 0
    assert equal_weight == bool(np.array_equal(vals_np, np.ones_like(vals_np)))

    weights_np = np.zeros((n_vertices, n_vertices))
    np.add.at(weights_np, (rows_np, cols_np), vals_np)
    totals_np = weights_np.sum(axis=1, keepdims=True)
    expected_np = np.where(
        totals_np > 0.0, weights_np / np.where(totals_np > 0.0, totals_np, 1.0), 0.0
    )
    # 3 072-3 136 non-zero entries here; an all-zero table would compare equal to an empty operator.
    assert np.count_nonzero(expected_np) == rows_np.size

    assembled_np = bsr_to_dense(
        tw.laplacian.laplacian(
            mesh_wp.points, mesh_wp.indices, equal_weight=equal_weight, symmetric=symmetric
        ),
        n_vertices,
    )
    assert np.allclose(assembled_np, expected_np, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# laplacian / graph_laplacian
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parametrize("equal_weight", [True, False])
@pytest.mark.parity("laplacian_equal_weight", "trimesh")
@pytest.mark.parity("laplacian_inverse_distance", "trimesh")
def test_laplacian_operator(
    request: pytest.FixtureRequest, mesh_name: str, equal_weight: bool
) -> None:
    """
    Class A: the row-normalized umbrella operator against ``trimesh.smoothing``, both weightings.

    Both ``equal_weight`` branches are compared, which matters because the inverse-distance
    branch also changes trimesh's default adjacency shape -- ``laplacian_entries``'
    ``symmetric`` default tracks that, and
    [`test_laplacian_entries_assemble_into_the_laplacian`] pins the tracking.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    operator_coo_tm = tms.laplacian_calculation(mesh_tm, equal_weight=equal_weight)
    assert operator_coo_tm is not None
    operator_tm = operator_coo_tm.tocsr()
    operator_wp = bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=equal_weight)
    )

    assert operator_wp.shape == operator_tm.shape
    assert np.allclose(operator_wp.toarray(), operator_tm.toarray(), rtol=1e-5, atol=1e-5)


def test_laplacian_symmetric_flag(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    # On an open mesh the directed and symmetric adjacencies differ; forcing ``symmetric``
    # overrides the ``not equal_weight`` default, so uniform+symmetric is row-stochastic and
    # symmetric while uniform+directed matches trimesh's (asymmetric) ``edges_to_coo``.
    _, mesh_wp = half_torus
    directed = bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=True, symmetric=False)
    )
    symmetric = bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=True, symmetric=True)
    )

    directed_dense = directed.toarray()
    symmetric_dense = symmetric.toarray()
    assert not np.allclose(directed_dense, symmetric_dense)
    # Symmetric adjacency has a symmetric sparsity pattern; directed does not (open boundary).
    assert np.array_equal(symmetric_dense != 0.0, (symmetric_dense != 0.0).T)
    assert not np.array_equal(directed_dense != 0.0, (directed_dense != 0.0).T)
    # Both operators are row-stochastic.
    assert np.allclose(symmetric_dense.sum(axis=1), 1.0)
    assert np.allclose(directed_dense.sum(axis=1), 1.0)


# -----------------------------------------------------------------------------------------
# mass_matrix_entries (test_mass_matrix asserts the diagonal it returns)
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("mass_matrix_entries", "igl")
def test_mass_matrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: ``mass_matrix_entries`` against the *diagonal* of ``igl.massmatrix``.

    The named transform is taking igl's diagonal -- triwarp returns the lumped vector, not a
    matrix. Despite the name this tests the entries; the assembled form is
    [`test_mass_matrix_assembled_matches_igl`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    mass_wp = tw.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(mass_wp.numpy(), mass_igl, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# mass_matrix
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
@pytest.mark.parity("mass_matrix", "igl")
def test_mass_matrix_assembled_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The *assembled* barycentric mass matrix, not just its diagonal.

    ``test_mass_matrix`` already pins ``mass_matrix_entries`` against ``igl.massmatrix(...)
    .diagonal()``, which is the same numbers; what this adds is the sparse build around them, and
    that is a separate benchmark group for the same reason. Class A on the dense form: the matrix is
    diagonal, so every off-diagonal entry must be zero, and asserting on the full array rather than
    on ``.diagonal()`` is what makes a stray off-diagonal triplet visible.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).tocsr()
    mass_wp = bsr_to_csr(tw.laplacian.mass_matrix(mesh_wp.points, mesh_wp.indices))

    assert mass_wp.shape == mass_igl.shape
    assert np.allclose(mass_wp.toarray(), mass_igl.toarray(), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------------------
# Cross-cutting
#
# Spans six public functions at once, so it mirrors no single position above.
# -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _LAPLACIAN_MESHES)
def test_operators_float64_match_float32(request: pytest.FixtureRequest, mesh_name: str) -> None:
    # Every operator exposes a ``dtype`` parameter for native float64 assembly (used by the
    # linear-system solvers). The float64 build must carry float64 values and agree with the
    # float32 build within float32 precision.
    _, mesh_wp = request.getfixturevalue(mesh_name)
    points, indices = mesh_wp.points, mesh_wp.indices

    cot_entries_f32 = tw.laplacian.cotmatrix_entries(points, indices)
    cot_entries_f64 = tw.laplacian.cotmatrix_entries(points, indices, dtype=wp.float64)
    assert cot_entries_f64.dtype == wp.float64
    assert np.allclose(cot_entries_f64.numpy(), cot_entries_f32.numpy(), rtol=1e-5, atol=1e-5)

    mass_f32 = tw.laplacian.mass_matrix_entries(points, indices)
    mass_f64 = tw.laplacian.mass_matrix_entries(points, indices, dtype=wp.float64)
    assert mass_f64.dtype == wp.float64
    assert np.allclose(mass_f64.numpy(), mass_f32.numpy(), rtol=1e-5, atol=1e-5)

    builders = [
        tw.laplacian.cotmatrix,
        tw.laplacian.laplacian,
        tw.laplacian.graph_laplacian,
        tw.laplacian.mass_matrix,
    ]
    for builder in builders:
        matrix_f32 = builder(points, indices)
        matrix_f64 = builder(points, indices, dtype=wp.float64)
        assert matrix_f64.values.dtype == wp.float64
        dense_f32 = bsr_to_csr(matrix_f32).toarray()
        dense_f64 = bsr_to_csr(matrix_f64).toarray()
        assert dense_f64.shape == dense_f32.shape
        assert np.allclose(dense_f64, dense_f32, rtol=1e-5, atol=1e-5)
