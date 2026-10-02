"""
Benchmarks for ``ordito.energies``: the quadratic forms assembled from a Laplacian.

Split out of [`test_laplacian.py`](test_laplacian.py) with the module. The distinction that makes it
its own file is what the timed callable *takes*: everything here consumes an already-assembled
operator, so its inputs are cached per mesh on both sides and the row measures the assembly of the
second-order form alone. ``test_laplacian.py`` times the first-order builders these are fed.

All five rows run the **scale** axis (``sphere_small`` -> ``sphere_med`` -> ``sphere_large``) rather
than the scan sweep, and the ratio against igl grows with size on every one of them, because igl's
side is Eigen sparse products where ordito's is a fixed number of launches. libigl is the only
reference that exposes these operators at all -- no other library has a Crouzeix-Raviart pair, a
curved Hessian energy or an integrated k-harmonic form.

The LSCM operators (``lscm_hessian``, ``vector_area_matrix``) are **not** timed here: neither has an
igl binding of its own, and their cost is measured through the ``lscm`` solve in
[`test_parametrization.py`](test_parametrization.py), which is the only way a caller reaches them.
Recorded in ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import igl
import numpy as np
import pytest
import pytorch3d.loss as p3d_loss
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase

if TYPE_CHECKING:
    import pytorch3d.structures as p3d_structures
    import scipy.sparse as sp
    import torch

# igl's stubs spell its sparse returns ``csc_matrix[float]``, outside scipy-stubs' scalar bound.
_OperatorInputsNp = tuple["sp.csc_matrix[float]", "sp.csc_matrix[float]"]  # pyright: ignore[reportInvalidTypeArguments]
_OperatorInputsWp = tuple[odt.BsrMatrix[wp.float64], wp.array[wp.float64]]
_operator_inputs_np_cache: dict[str, _OperatorInputsNp] = {}
_operator_inputs_wp_cache: dict[tuple[str, str], _OperatorInputsWp] = {}
_edge_numbering_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_edge_numbering_wp_cache: dict[tuple[str, str], tuple[odt.Array2dInt32, wp.array[wp.int32]]] = {}


def _laplacian_and_mass_np(bench_case: BenchCase) -> _OperatorInputsNp:
    """Prebuild igl's cotangent Laplacian and barycentric mass, cached per mesh (the inputs)."""
    name = bench_case.mesh_name
    if name not in _operator_inputs_np_cache:
        _operator_inputs_np_cache[name] = (
            igl.cotmatrix(bench_case.vertices_np, bench_case.faces_np),
            igl.massmatrix(
                bench_case.vertices_np, bench_case.faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC
            ),
        )
    return _operator_inputs_np_cache[name]


def _laplacian_and_mass_wp(bench_case: BenchCase) -> _OperatorInputsWp:
    """Ordito's float64 Laplacian and mass diagonal, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_inputs_wp_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _operator_inputs_wp_cache[key] = (
            od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64),
            od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64),
        )
    return _operator_inputs_wp_cache[key]


def _run_loss_pytorch3d(
    bench_case: BenchCase, loss_fn: Callable[[p3d_structures.Meshes], torch.Tensor]
) -> None:
    """
    Time one ``pytorch3d.loss`` regularizer with the ``Meshes`` built inside the timed callable.

    Every one of the three reads a **memoized** derivation -- ``edges_packed``,
    ``faces_packed_to_edges_packed``, ``laplacian_packed`` -- so a shared container would have
    rounds 2..n hit the cache and the row would report the reduction alone. Building it inside is
    what makes the row comparable with ordito's, which reassembles per call. The tensors are
    hoisted; only the container and its derivations are timed.
    """
    import pytorch3d.structures as p3d_structures
    import torch

    vertices_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
        device=bench_case.torch_device,
    )
    faces_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.faces_np, dtype=np.int64), device=bench_case.torch_device
    )
    loss_p3d = bench_case.run(
        lambda: loss_fn(p3d_structures.Meshes(verts=[vertices_p3d], faces=[faces_p3d]))
    )
    assert float(loss_p3d) >= 0.0


@pytest.mark.benchmark(group="edge_length_loss")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pytorch3d")
def test_edge_length_loss(bench_case: BenchCase) -> None:
    """
    The edge regularizer: a unique-edge pass, a squared deviation and one reduction.

    Read against ``edges_unique_length`` in [`test_edges.py`](test_edges.py) -- the deduplication
    is the whole cost on both sides and the reduction is the difference between the two groups.

    **pytorch3d** is the only reference and it is the one that pins the value
    (``tests/test_energies.py``). Two things separate the rows and neither is the arithmetic: its
    ``edges_packed()`` is a memoized accessor, so the ``Meshes`` is built inside the timed callable
    or the row reports nothing; and its own docstring flags the per-mesh weight gather as a
    bottleneck, which ordito has no counterpart for because a single mesh needs none.
    """
    if bench_case.kind == "pytorch3d":
        _run_loss_pytorch3d(bench_case, lambda mesh_p3d: p3d_loss.mesh_edge_loss(mesh_p3d))
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loss = bench_case.run(lambda: od.energies.edge_length_loss(vertices, faces))
    assert loss > 0.0


@pytest.mark.benchmark(group="normal_consistency_loss")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pytorch3d")
def test_normal_consistency_loss(bench_case: BenchCase) -> None:
    """
    The dihedral regularizer: face adjacency, one angle per pair, one reduction.

    The adjacency is inside the timed callable on both sides, and it is the row -- read this
    against ``face_adjacency_angles`` in [`test_adjacency.py`](test_adjacency.py), where the same
    build is timed without the reduction.

    **pytorch3d** does measurably more here than ordito, and the extra is not a constant: it
    enumerates every *pair* of faces per edge through a C++ helper over a per-edge vertex list,
    where ordito reads one pair per adjacency. The two agree on edge-manifold input
    (``tests/test_energies.py``) and this row prices that generality.
    """
    if bench_case.kind == "pytorch3d":
        _run_loss_pytorch3d(bench_case, lambda mesh_p3d: p3d_loss.mesh_normal_consistency(mesh_p3d))
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loss = bench_case.run(lambda: od.energies.normal_consistency_loss(vertices, faces))
    assert loss >= 0.0


@pytest.mark.benchmark(group="laplacian_smoothing_loss")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "pytorch3d")
@pytest.mark.parametrize("method", ["uniform", "cotcurv"])
def test_laplacian_smoothing_loss(
    bench_case: BenchCase, method: Literal["uniform", "cotcurv"]
) -> None:
    """
    The smoothness regularizer, at the cheap and the expensive end of its ``method`` axis.

    The axis is the *operator*, which is the only thing that separates the three methods in cost:
    ``uniform`` assembles the row-normalized 1-ring average and ``cotcurv`` assembles the cotangent
    stiffness matrix **and** the lumped mass, so the pair brackets the range and ``cot`` sits
    between them (it is the same stiffness assembly with a diagonal read instead of a mass pass).
    Everything after the assembly is one CSR pass and one reduction on both sides.

    **pytorch3d** reassembles per call as ordito does, so this is a like-for-like race between two
    sparse assemblies -- the only group in this module where that is true, since the four operator
    groups below hand both sides prebuilt inputs.
    """
    if bench_case.kind == "pytorch3d":
        _run_loss_pytorch3d(
            bench_case, lambda mesh_p3d: p3d_loss.mesh_laplacian_smoothing(mesh_p3d, method=method)
        )
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loss = bench_case.run(lambda: od.energies.laplacian_smoothing_loss(vertices, faces, method))
    assert loss > 0.0


@pytest.mark.benchmark(group="k_harmonic")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_k_harmonic(bench_case: BenchCase) -> None:
    """
    The biharmonic operator ``L M^-1 L`` from a prebuilt Laplacian and mass (``k = 2``).

    Both sides consume cached, prebuilt inputs — the matrices *are* the function's arguments in
    both APIs — so the row times only the composition: igl's two Eigen sparse products against
    ordito's counted-and-emitted triplet pass plus one ``bsr_from_triplets``.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        laplacian, mass = _laplacian_and_mass_wp(bench_case)
        operator = bench_case.run(lambda: od.energies.k_harmonic(laplacian, mass, k=2))
        assert operator.nrow == n_vertices
    else:
        laplacian_igl, mass_igl = _laplacian_and_mass_np(bench_case)
        operator_igl = bench_case.run(
            lambda: igl.harmonic_integrated_from_laplacian_and_mass(laplacian_igl, mass_igl, 2)
        )
        assert operator_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="hessian_energy")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_hessian_energy(bench_case: BenchCase) -> None:
    """
    The natural-boundary Hessian smoothness energy, geometry to assembled matrix.

    igl materializes the ``(9 F, V)`` stacked Hessian and squares it through Eigen sparse
    products; ordito contracts the component pairs analytically and emits ``9 * valence^2``
    triplets per vertex. Uniform valence 6 on this axis keeps that quadratic cost benign.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = bench_case.run(lambda: od.energies.hessian_energy(vertices, faces))
        assert operator.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        operator_igl = bench_case.run(lambda: igl.hessian_energy(vertices_np, faces_np))
        assert operator_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="curved_hessian_energy")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_curved_hessian_energy(bench_case: BenchCase) -> None:
    """
    The Crouzeix-Raviart curved Hessian energy, geometry to assembled matrix.

    igl assembles four ``(2 E, ...)`` CR operators and chains four sparse products; ordito emits
    each face's sandwiched 6x6 block directly (144 triplets per face) and never materializes an
    edge-indexed matrix.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = bench_case.run(lambda: od.energies.curved_hessian_energy(vertices, faces))
        assert operator.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        operator_igl = bench_case.run(lambda: igl.curved_hessian_energy(vertices_np, faces_np))
        assert operator_igl.shape == (n_vertices, n_vertices)


def _edge_numbering_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Igl's ``unique_edge_map`` numbering, cached per mesh: the CR bindings' explicit input."""
    name = bench_case.mesh_name
    if name not in _edge_numbering_np_cache:
        unique_edge_map = igl.unique_edge_map(bench_case.faces_np)
        _edge_numbering_np_cache[name] = (unique_edge_map[1], unique_edge_map[2].ravel())
    return _edge_numbering_np_cache[name]


def _edge_numbering_wp(bench_case: BenchCase) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]:
    """Ordito's ``edges_unique`` numbering, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_numbering_wp_cache:
        _edge_numbering_wp_cache[key] = od.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
    return _edge_numbering_wp_cache[key]


@pytest.mark.benchmark(group="crouzeix_raviart_cotmatrix")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_crouzeix_raviart_cotmatrix(bench_case: BenchCase) -> None:
    """
    Edge-based CR stiffness matrix with the edge numbering prebuilt on both sides.

    ``igl.crouzeix_raviart_cotmatrix`` takes ``(E, EMAP)`` explicitly, so the numbering is the
    function's input, not its work; ordito is handed its own precomputed ``edges_unique`` pair
    for the same reason. Each side uses its native numbering — the timing is unaffected, and the
    value-level correspondence is pinned in ``tests/test_laplacian.py``.
    """
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        unique_edges, edge_map = _edge_numbering_wp(bench_case)
        matrix = bench_case.run(
            lambda: od.energies.crouzeix_raviart_cotmatrix(
                vertices, faces, unique_edges=unique_edges, edge_map=edge_map
            )
        )
        assert matrix.nrow == unique_edges.shape[0]
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        edges_igl, edge_map_igl = _edge_numbering_np(bench_case)
        matrix_igl = bench_case.run(
            lambda: igl.crouzeix_raviart_cotmatrix(vertices_np, faces_np, edges_igl, edge_map_igl)
        )
        assert matrix_igl.shape == (len(edges_igl), len(edges_igl))


@pytest.mark.benchmark(group="crouzeix_raviart_massmatrix")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_crouzeix_raviart_massmatrix(bench_case: BenchCase) -> None:
    """The diagonal CR mass, same prebuilt-numbering convention as the cotmatrix row."""
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        unique_edges, edge_map = _edge_numbering_wp(bench_case)
        matrix = bench_case.run(
            lambda: od.energies.crouzeix_raviart_massmatrix(
                vertices, faces, unique_edges=unique_edges, edge_map=edge_map
            )
        )
        assert matrix.nrow == unique_edges.shape[0]
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        edges_igl, edge_map_igl = _edge_numbering_np(bench_case)
        matrix_igl = bench_case.run(
            lambda: igl.crouzeix_raviart_massmatrix(vertices_np, faces_np, edges_igl, edge_map_igl)
        )
        assert matrix_igl.shape == (len(edges_igl), len(edges_igl))
