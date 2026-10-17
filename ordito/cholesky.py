"""
Sparse Cholesky factorization of symmetric positive-(semi-)definite operators, solved on the device.

[`sparse_cholesky`][ordito.cholesky.sparse_cholesky] orders the operator by nested dissection,
factors it level by level on the device and returns a
[`SparseCholesky`][ordito.cholesky.SparseCholesky] whose
[`solve`][ordito.cholesky.SparseCholesky.solve] is a fixed sequence of matrix-vector products,
recorded once and replayed. Where a Krylov solver's cost grows with the operator's condition number
and with the distance information must travel across the mesh, a factorization's does not: one
factor serves every later right-hand side and every operator of the same sparsity
([`refactor`][ordito.cholesky.SparseCholesky.refactor]).

Every solve is refined against the operator itself until its residual passes the requested test.
A solution spanning hundreds of orders of magnitude -- the heat method's diffusions -- can be
solved *componentwise*, each entry to its own size, which no iteration controlled by a residual
norm achieves: on an M-matrix the factor and both triangular solves add terms of one sign, so no
entry is formed by cancellation.

A component of the operator's graph whose rows all sum to zero (a Laplacian's constant null space)
is solved up to that constant, which is then set to the initial guess's mean: the answer conjugate
gradient gives from the same guess on a consistent right-hand side.

Notes
-----
The ordering is geometric nested dissection when vertex coordinates are given and the same
bisection over breadth-first distances from three mutually far vertices otherwise. The numeric
factorization is multifrontal (Liu 1992); each supernode stores the inverse of its diagonal block
and the product of that inverse with its off-diagonal rows, so both triangular solves are
level-scheduled products with no substitution inside a level (the partitioned inverse of Alvarado,
Pothen and Schreiber 1993). Every partial sum lands in its own slot and is reduced in a fixed
order, so solves are reproducible on both devices.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, record_device_loop, require_same_device
from ordito.array import arange
from ordito.graph import connected_component_labels
from ordito.kernels import cholesky as kernel_cholesky
from ordito.kernels import grouping as kernel_grouping

# Largest leaf of the nested-dissection tree, in rows.
CHOLESKY_LEAF_ROWS = 32
# The top of each dissection tree is merged into one supernode while it holds at most this many
# rows: fewer levels for every solve, one dense front for the factorization.
CHOLESKY_TOP_ROWS = 512
# Device memory a factorization may take at its peak, in bytes -- what it holds once built plus the
# transient arena its first numeric factorization works in (or a first solve's scratch, when that
# is larger); ``sparse_cholesky`` refuses a larger one.
CHOLESKY_MEMORY_BUDGET = 4 << 30
# Refinement rounds a solve runs at most.
CHOLESKY_REFINE_ROUNDS = 12

# A row sums to "zero" below this fraction of its diagonal.
_SINGULAR_ROW_SUM = 1e-10
# Rows whose diagonal ``sparse_cholesky(negated=None)`` sums to choose the sign it tries first.
_SIGN_PROBE_ROWS = 64


class NotPositiveDefiniteError(ValueError):
    """
    The operator met a pivot that is not positive: it is not definite, at least in ``float64``.

    Raised by [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] and
    [`SparseCholesky.refactor`][ordito.cholesky.SparseCholesky.refactor]; a ``ValueError``, so a
    caller catching that sees it too. Distinct from the operator being too large to factor, which
    an iteration may still solve: a symmetric operator this cannot factor either way round is past
    what conjugate gradient converges on as well.

    Examples
    --------
    A cotangent Laplacian is negative semi-definite, so factoring it as given fails:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    L = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    try:
        od.cholesky.sparse_cholesky(L)
        outcome = "factored"
    except od.cholesky.NotPositiveDefiniteError:
        outcome = "not positive definite"
    print(outcome)  # not positive definite
    ```
    """


class SparseCholesky:
    """
    A sparse Cholesky factorization on the device.

    Built by [`sparse_cholesky`][ordito.cholesky.sparse_cholesky].

    Attributes
    ----------
    n : int
        Rows of the factored operator.
    device : wp.context.Device
        Device the factor lives on.
    nbytes : int
        Device memory the factorization holds, in bytes: its analysis (shared with every
        factorization [`with_operator`][ordito.cholesky.SparseCholesky.with_operator] builds from
        it; [`held_nbytes`][ordito.cholesky.held_nbytes] counts it once), its factor, the scratch
        of every right-hand-side count solved so far, and the storage a repeated
        [`refactor`][ordito.cholesky.SparseCholesky.refactor] keeps. Building also takes a
        transient arena, not held, which
        [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET] counts.
    negated : bool
        Whether ``-matrix`` was factored (a negative semi-definite operator).
    records_at_once : bool
        Whether a solve records its refinement graph on its first run (a factorization solved many
        times), rather than on its second against the same operator storage (the default: a
        one-shot solve then runs its rounds directly and records nothing).

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    system = od.heat.heat_operators(vertices, faces)[0]  # M - t L, float64
    factor = od.cholesky.sparse_cholesky(system, vertices)
    print(factor.n, factor.negated, factor.nbytes > 0)  # 162 False True
    ```
    """

    def __init__(self, plan: _Plan, matrix: odt.BsrMatrix[wp.float64], *, negated: bool) -> None:
        """Factor ``matrix`` (or its negation) over ``plan``'s structure."""
        self.negated = negated
        self.records_at_once = False
        self.n = plan.n
        self.device = plan.device
        self.nbytes = plan.nbytes + plan.numeric_nbytes()
        self._plan = plan
        self._numeric = plan.numeric()
        self._solves: dict[int, _Solve] = {}
        self._matrix = matrix
        # A caller that refactors repeatedly gets stable storage: the operator copied into arrays
        # this factorization owns, the fronts arena kept, and the numeric factorization recorded
        # once and replayed, as the refined solve then is too (``_Solve.run`` replays for one
        # operator storage). Built on the first ``refactor`` after construction.
        self._stable: odt.BsrMatrix[wp.float64] | None = None
        self._fronts: wp.array[wp.float64] | None = None
        self._factor_graph: Any = None
        # Whether the numeric factor is one ``solve`` may use: false from a ``refactor`` that
        # raised until one that succeeds.
        self._valid = False
        self._constructing = True
        self.refactor(matrix)
        self._constructing = False

    def refactor(self, matrix: odt.BsrMatrix[wp.float64]) -> None:
        """
        Factor a new operator of the same sparsity pattern in place.

        If the new operator is not positive definite, the factorization holds no usable factor
        afterwards: [`solve`][ordito.cholesky.SparseCholesky.solve] raises until a later
        ``refactor`` succeeds.

        Parameters
        ----------
        matrix
            ``(n, n)`` symmetric ``float64`` operator whose stored pattern is the one this
            factorization was built for (the same row offsets and column indices). Only its
            shape and its storage's sizes are checked; another pattern of as many stored entries
            is not detected and factors a wrong operator.

        Raises
        ------
        ValueError
            If ``matrix`` is not ``(n, n)`` scalar ``float64``, or stores fewer entries than the
            pattern this factorization was built for.
        NotPositiveDefiniteError
            If the operator is not positive definite on its non-singular components (a pivot that
            is not positive).
        RuntimeError
            If ``matrix`` is not on this factorization's device.

        Examples
        --------
        The heat system at another diffusion time has the same pattern, so the analysis is reused:

        ```python
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.icosphere(subdivisions=2)
        factor = od.cholesky.sparse_cholesky(od.heat.heat_operators(vertices, faces, t=0.01)[0])
        factor.refactor(od.heat.heat_operators(vertices, faces, t=0.1)[0])
        mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
        solution = wp.zeros_like(mass)
        factor.solve(mass, solution)  # (M - t L) 1 = M 1
        print(f"{solution.numpy().min():.6f} {solution.numpy().max():.6f}")  # 1.000000 1.000000
        ```
        """
        require_same_device(matrix=matrix, factor=self)
        self._require_pattern(matrix, "refactor")
        plan = self._plan
        sign = -1.0 if self.negated else 1.0
        # The factor is overwritten from here on: unusable until the pivots are read back good.
        self._valid = False
        if self._constructing:
            plan.factor(matrix, self._numeric, sign)
        else:
            matrix = self._stable_copy(matrix)
            fronts = self._fronts
            if fronts is None:
                fronts = self._fronts = _launch.empty(
                    plan.front_entries, dtype=wp.float64, device=self.device
                )
                self.nbytes += 8 * plan.front_entries
            device = wp.get_device(self.device)
            if device.is_cuda:
                if self._factor_graph is None:
                    with wp.ScopedCapture(device) as capture:
                        plan.factor(matrix, self._numeric, sign, fronts)
                    self._factor_graph = capture.graph
                wp.capture_launch(self._factor_graph)
            else:
                plan.factor(matrix, self._numeric, sign, fronts)
        # Read back: a bad pivot raises, and whether any component is singular decides whether
        # the solves project a null space at all -- both in one read.
        bad_pivot, singular = read_values(self._numeric.status, 0, 2)
        if bad_pivot != 0:
            raise NotPositiveDefiniteError("sparse_cholesky: the operator is not positive definite")
        self._numeric.any_singular = singular != 0
        self._matrix = matrix
        for solve in self._solves.values():
            solve.matrix = matrix
        self._valid = True

    def _require_pattern(self, matrix: odt.BsrMatrix[wp.float64], caller: str) -> None:
        """Raise unless ``matrix`` has this factorization's shape and storage for its pattern."""
        n = self.n
        rows, cols = int(matrix.nrow), int(matrix.ncol)
        if matrix.values.dtype != wp.float64 or rows != n or cols != n:
            raise ValueError(
                f"SparseCholesky.{caller}: expected an ({n}, {n}) scalar float64 operator, got "
                f"({rows}, {cols}) of {matrix.values.dtype.__name__}"
            )
        stored = self._plan.n_stored
        if matrix.offsets.size < n + 1 or min(matrix.columns.size, matrix.values.size) < stored:
            raise ValueError(
                f"SparseCholesky.{caller}: the operator stores fewer entries than the {stored} of "
                "the pattern this factorization was built for"
            )

    def _stable_copy(self, matrix: odt.BsrMatrix[wp.float64]) -> odt.BsrMatrix[wp.float64]:
        """Copy ``matrix``'s values into this factorization's own storage of its pattern."""
        stable = self._stable
        if stable is None:
            stable = odt.bsr_matrix_t(wp.float64)()
            stable.nrow, stable.ncol = int(matrix.nrow), int(matrix.ncol)
            stable.offsets = _launch.clone(matrix.offsets)
            stable.columns = _launch.clone(matrix.columns)
            stable.values = _launch.clone(matrix.values)
            stable.row_counts = None
            stable.notify_nnz_changed(nnz=int(matrix.nnz))
            self.nbytes += sum(
                held.capacity for held in (stable.offsets, stable.columns, stable.values)
            )
            self._stable = stable
        elif self._plan.n_stored:
            # The pattern's entries only: a caller's storage may be larger than the pattern.
            _launch.copy(stable.values, matrix.values, count=self._plan.n_stored)
        return stable

    def with_operator(
        self, matrix: odt.BsrMatrix[wp.float64], *, negated: bool | None = None
    ) -> SparseCholesky:
        """
        Factor another operator of this one's sparsity pattern, sharing this analysis.

        Only the numeric factorization is paid: the ordering and every structure map are this
        factorization's, held by both. The two factorizations are independent otherwise.

        Parameters
        ----------
        matrix
            ``(n, n)`` symmetric ``float64`` operator stored over the pattern this factorization
            was built for (the same row offsets and column indices; checked as
            [`refactor`][ordito.cholesky.SparseCholesky.refactor] checks it).
        negated
            As [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] takes it: ``None`` factors
            whichever sign is positive definite.

        Returns
        -------
        SparseCholesky
            The new factorization.

        Raises
        ------
        ValueError
            If ``matrix`` is not ``(n, n)`` scalar ``float64``, or stores fewer entries than the
            pattern this factorization was built for.
        NotPositiveDefiniteError
            If the operator (with ``negated=None``: either sign of it) is not positive definite.
        RuntimeError
            If ``matrix`` is not on this factorization's device.

        Examples
        --------
        Two diffusion times, one analysis, two independent factorizations:

        ```python
        import ordito as od

        vertices, faces = od.creation.icosphere(subdivisions=2)
        short = od.cholesky.sparse_cholesky(od.heat.heat_operators(vertices, faces, t=0.01)[0])
        long = short.with_operator(od.heat.heat_operators(vertices, faces, t=0.1)[0])
        print(long is short, long.n)  # False 162
        ```
        """
        require_same_device(matrix=matrix, factor=self)
        self._require_pattern(matrix, "with_operator")
        return _factored(self._plan, matrix, negated)

    def solve(
        self,
        rhs: odt.ArrayNd,
        solution: odt.ArrayNd,
        *,
        tol: float = 1e-12,
        componentwise: bool = False,
    ) -> None:
        """
        Solve the factored system for one or several right-hand sides, refined in ``float64``.

        Parameters
        ----------
        rhs
            ``(n,)`` right-hand side, or ``(n_columns, n)`` ``float64`` columns solved together.
        solution
            Same shape as ``rhs``: the initial guess, overwritten with the answer. Refinement
            starts from it, so a guess that already passes ``tol`` costs one residual and no
            triangular solve; on a singular component it supplies the constant (see the module
            notes).
        tol
            Refinement stops once every row's residual passes the backward-error test
            ``|b - A x|_i <= tol * max(s_i, max_j |b_j|)``, with ``s_i = (|A| |x| + |b|)_i`` the
            row's Oettli-Prager scale and ``max_j |b_j|`` the column's largest right-hand-side
            entry -- or, with ``componentwise``, ``|b - A x|_i <= tol * s_i``. A row dropped to
            pin a singular component is exempt; a residual that is not a number never passes.
            At most [`CHOLESKY_REFINE_ROUNDS`][ordito.cholesky.CHOLESKY_REFINE_ROUNDS]
            refinement solves run: a solution still failing the test then is returned as it
            stands, without a warning.
        componentwise
            Refine every entry to its own size: for solutions spanning many orders of magnitude.

        Raises
        ------
        ValueError
            If ``rhs`` or ``solution`` is not ``float64`` of a matching shape.
        RuntimeError
            If ``rhs`` and ``solution`` are not on this factorization's device, or the last
            [`refactor`][ordito.cholesky.SparseCholesky.refactor] raised, leaving no factor.

        Examples
        --------
        Two right-hand sides solved together: ``(M - t L) c = c M 1`` for a constant ``c``:

        ```python
        import numpy as np
        import warp as wp

        import ordito as od

        vertices, faces = od.creation.icosphere(subdivisions=2)
        factor = od.cholesky.sparse_cholesky(od.heat.heat_operators(vertices, faces)[0], vertices)
        mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64).numpy()
        rhs = wp.array(np.stack([mass, 2.0 * mass]), dtype=wp.float64)
        solution = wp.zeros_like(rhs)
        factor.solve(rhs, solution)
        print(solution.numpy().mean(axis=1).round(6))  # [1. 2.]
        ```
        """
        require_same_device(rhs=rhs, solution=solution, factor=self)
        if not self._valid:
            raise RuntimeError(
                "SparseCholesky.solve: the last refactor failed and left no factor; refactor "
                "with a positive definite operator first"
            )
        if rhs.dtype != wp.float64 or solution.dtype != wp.float64 or rhs.shape != solution.shape:
            raise ValueError("SparseCholesky.solve takes float64 rhs and solution of one shape")
        n_columns = 1 if rhs.ndim == 1 else int(rhs.shape[0])
        if int(rhs.shape[-1]) != self.n:
            raise ValueError(
                f"SparseCholesky.solve: rhs has {rhs.shape[-1]} rows, expected {self.n}"
            )
        if self.n == 0 or n_columns == 0:
            return
        key = (n_columns, float(tol), bool(componentwise))
        solve = self._solves.get(n_columns)
        if solve is None or solve.key != key:
            if solve is not None:
                self.nbytes -= self._plan.solve_nbytes(n_columns)
            solve = _Solve(
                self._plan,
                self._numeric,
                self._matrix,
                n_columns,
                float(tol),
                bool(componentwise),
                -1.0 if self.negated else 1.0,
            )
            self._solves[n_columns] = solve
            self.nbytes += self._plan.solve_nbytes(n_columns)
        solve.run(rhs.flatten(), solution.flatten(), self.records_at_once)


def sparse_cholesky(
    matrix: odt.BsrMatrix[wp.float64],
    coordinates: wp.array[wp.vec3] | None = None,
    *,
    negated: bool | None = False,
) -> SparseCholesky:
    """
    Sparse Cholesky factorization of a symmetric positive-(semi-)definite ``float64`` operator.

    Orders the operator by nested dissection, factors it on the device and returns the
    factorization, whose [`solve`][ordito.cholesky.SparseCholesky.solve] answers any later
    right-hand side in a fixed number of device passes. The ordering and the symbolic analysis run
    on the device too, reading back a few small counts per level of the dissection. The analysis
    belongs to the returned factorization: [`refactor`][ordito.cholesky.SparseCholesky.refactor]
    reuses it for another operator of the same pattern, and nothing is kept across calls.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric scalar ``float64`` operator, positive definite apart from components
        whose rows all sum to zero (solved up to a constant, see the module notes). Its stored
        pattern must be symmetric too: where ``(i, j)`` is stored, so is ``(j, i)``.
    coordinates
        ``(n,)`` positions of the rows, for a geometric ordering: a mesh's vertices for a vertex
        operator. When ``None``, the ordering is computed from the pattern alone.
    negated
        Factor ``-matrix`` instead, for a negative (semi-)definite operator such as
        [`cotmatrix`][ordito.laplacian.cotmatrix]; solves still answer ``matrix x = rhs``. When
        ``None``, factor whichever of ``matrix`` and ``-matrix`` is positive definite, over one
        analysis: the sign of the leading diagonal is tried first.

    Returns
    -------
    SparseCholesky
        The factorization, on ``matrix.device``.

    Raises
    ------
    ValueError
        If ``matrix`` is not square scalar ``float64``, its stored pattern is not symmetric,
        ``coordinates`` does not have ``n`` rows, or the factorization's peak device memory would
        exceed [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET].
    NotPositiveDefiniteError
        If the operator is not positive definite (with ``negated=None``: neither sign is).
    RuntimeError
        If ``matrix`` and ``coordinates`` are not on one device.

    See Also
    --------
    [`SparseCholesky`][ordito.cholesky.SparseCholesky]
    [`solve_spd`][ordito.linalg.solve_spd]

    Examples
    --------
    The heat method's system ``M - t L`` maps the constant ``1`` to the lumped masses, since every
    row of ``L`` sums to zero; solving against the masses recovers it:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    system = od.heat.heat_operators(vertices, faces)[0]  # M - t L, float64
    factor = od.cholesky.sparse_cholesky(system, vertices)
    mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    solution = wp.zeros_like(mass)
    factor.solve(mass, solution)
    print(f"{solution.numpy().min():.6f} {solution.numpy().max():.6f}")  # 1.000000 1.000000
    ```

    A cotangent Laplacian is negative semi-definite: ``negated=None`` finds the sign, and the
    singular constant component is solved up to a constant:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(6, 6))
    L = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    factor = od.cholesky.sparse_cholesky(L, vertices, negated=None)
    print(factor.negated)  # True
    x = wp.array(vertices.numpy()[:, 0], dtype=wp.float64)
    solution = wp.zeros_like(x)
    factor.solve(L @ x, solution)
    print(np.ptp(solution.numpy() - x.numpy()) < 1e-9)  # True
    ```
    """
    require_same_device(matrix=matrix, coordinates=coordinates)
    return _factored(_plan_cholesky(matrix, coordinates), matrix, negated)


def _factored(
    plan: _Plan, matrix: odt.BsrMatrix[wp.float64], negated: bool | None
) -> SparseCholesky:
    """Factor ``matrix`` over ``plan``; ``negated=None`` tries the leading diagonal's sign first."""
    if negated is not None:
        return SparseCholesky(plan, matrix, negated=negated)
    first = _leading_diagonal_negative(matrix)
    try:
        return SparseCholesky(plan, matrix, negated=first)
    except NotPositiveDefiniteError:
        # Retried outside the handler: the exception's traceback holds the first attempt (its
        # solve blocks and fronts), released once the handler ends, before the second allocates.
        pass
    return SparseCholesky(plan, matrix, negated=not first)


def _leading_diagonal_negative(matrix: odt.BsrMatrix[wp.float64]) -> bool:
    """Whether the first rows' diagonal sums below zero: the sign a definite operator has."""
    device = matrix.device
    total = _launch.empty(1, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_cholesky.leading_diagonal_sum,
        dim=1,
        inputs=[matrix.offsets, matrix.columns, matrix.values,
                wp.int32(min(int(matrix.nrow), _SIGN_PROBE_ROWS))],
        outputs=[total],
        device=device,
    )  # fmt: skip
    # Read back: which sign to factor first is the host's choice of launch arguments.
    return float(read_scalar(total, 0)) < 0.0


def held_nbytes(factorizations: Iterable[SparseCholesky]) -> int:
    """
    Device memory several factorizations hold together, in bytes: a shared analysis counted once.

    Each factorization's [`nbytes`][ordito.cholesky.SparseCholesky] counts its analysis, which
    every factorization built from it by
    [`with_operator`][ordito.cholesky.SparseCholesky.with_operator] holds too; summing ``nbytes``
    counts that analysis once a factorization.

    Parameters
    ----------
    factorizations
        The factorizations, each counted once however often it appears.

    Returns
    -------
    int
        The bytes of every distinct allocation the factorizations hold.

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    short = od.cholesky.sparse_cholesky(od.heat.heat_operators(vertices, faces, t=0.01)[0])
    long = short.with_operator(od.heat.heat_operators(vertices, faces, t=0.1)[0])
    together = od.cholesky.held_nbytes([short, long])
    print(short.nbytes < together < short.nbytes + long.nbytes)  # True
    ```
    """
    analyses: dict[int, int] = {}
    own = 0
    for factor in {id(factor): factor for factor in factorizations}.values():
        # The analysis is this module's own state of a factorization.
        plan = factor._plan  # pyright: ignore[reportPrivateUsage]
        analyses[id(plan)] = plan.nbytes
        own += factor.nbytes - plan.nbytes
    return own + sum(analyses.values())


def _plan_cholesky(
    matrix: odt.BsrMatrix[wp.float64], coordinates: wp.array[wp.vec3] | None = None
) -> _Plan:
    """Return the symbolic half of ``sparse_cholesky``: ordering, structure, every device map."""
    if matrix.values.dtype != wp.float64 or int(matrix.nrow) != int(matrix.ncol):
        raise ValueError("sparse_cholesky factors a square scalar float64 operator")
    n = int(matrix.nrow)
    if coordinates is not None and coordinates.size != n:
        raise ValueError(f"sparse_cholesky: coordinates has {coordinates.size} rows, expected {n}")
    labels = (
        connected_component_labels(matrix)
        if n > 0
        else _launch.empty(0, dtype=wp.int32, device=matrix.device)
    )
    if coordinates is None:
        coordinates = _landmark_coordinates(matrix, labels)
    return _Plan(matrix, coordinates, labels)


class _Plan:
    """Device maps for one sparsity pattern: factorization arenas, scatter maps, solve tasks."""

    def __init__(
        self,
        matrix: odt.BsrMatrix[wp.float64],
        coordinates: wp.array[wp.vec3],
        labels: wp.array[wp.int32],
    ) -> None:
        n = int(matrix.nrow)
        device = matrix.device
        self.n = n
        self.device = device
        if n == 0:
            self._empty(labels)
            return
        offsets, columns = matrix.offsets, matrix.columns
        # Components, for the singular-component pins and the null-space projection: the rows in
        # label order (one stable sort) and the run starts' scan, issued ahead of the dissection
        # so the component count is read with the structure's counts.
        component_keys = _launch.empty(2 * n, dtype=wp.int32, device=device)
        component_rows = _launch.empty(2 * n, dtype=wp.int32, device=device)
        self.component_rows = component_rows
        flags = _launch.empty(n, dtype=wp.int32, device=device)
        ends = _launch.empty(n, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_grouping.KEYS_AND_IDENTITY[wp.int32],
            dim=n,
            inputs=[labels],
            outputs=[component_keys, component_rows],
            device=device,
        )
        _launch.radix_sort_pairs(component_keys, component_rows, n, end_bit=_bits(n))
        _launch.launch(
            kernel_grouping.MARK_SORTED_RUN_STARTS[wp.int32],
            dim=n,
            inputs=[component_keys],
            outputs=[flags],
            device=device,
        )
        _launch.array_scan(flags, ends, inclusive=True)
        node, tree, state, depths = _nested_dissection(offsets, columns, n, coordinates)
        s = _structure(offsets, columns, n, node, tree, state, depths, ends)
        n_sn, n_levels = s.n_sn, s.n_levels
        self.n_levels = n_levels
        panel = int(kernel_cholesky.PANEL)

        # The schedule, on the device: each supernode's front size and prefix-summed terms (front
        # and solve-block offsets, chunk bases, task totals), the level order (one sort by height),
        # the level-major prefixes the solve maps are laid out by, the extend-add groups (children
        # by parent, then by ``(height, slot)``) and every level's panel work.
        def empty(count: int, dtype: Any) -> wp.array[Any]:
            return _launch.empty(count, dtype=dtype, device=device)

        sn_bits = _bits(n_sn * (n_sn + 1) * (n_levels + 1))
        sentinel = wp.uint64((1 << sn_bits) - 1)
        self.front_size = empty(n_sn, wp.int32)
        parent_clamped = empty(n_sn, wp.int32)
        terms = _launch.zeros(n_sn + 1, dtype=kernel_cholesky.front_terms, device=device)
        level_keys = empty(2 * n_sn, wp.uint64)
        level_nodes_all = empty(2 * n_sn, wp.int32)
        child_keys = empty(2 * n_sn, wp.uint64)
        child_values = empty(2 * n_sn, wp.int32)
        level_stats = odt.as_array2d(
            _launch.zeros((n_levels, 3), dtype=wp.int32, device=device), wp.int32
        )
        _launch.launch(
            kernel_cholesky.front_terms_and_keys,
            dim=n_sn,
            inputs=[s.ncol, s.nrow, s.height, s.parent, sentinel],
            outputs=[self.front_size, parent_clamped, terms, level_keys, level_nodes_all,
                     child_keys, child_values, level_stats],
            device=device,
        )  # fmt: skip
        _launch.radix_sort_pairs(
            level_keys, level_nodes_all, n_sn, end_bit=_bits(n_sn * (n_levels + 1))
        )
        _launch.radix_sort_pairs(child_keys, child_values, n_sn, end_bit=sn_bits)
        level_nodes = _launch.view(level_nodes_all, 0, n_sn)
        level_sums = _launch.zeros(n_sn + 1, dtype=wp.vec4i, device=device)
        _launch.launch(
            kernel_cholesky.level_terms,
            dim=n_sn,
            inputs=[level_nodes, s.ncol, self.front_size],
            outputs=[level_sums],
            device=device,
        )
        _launch.array_scan(terms, terms, inclusive=True)
        _launch.array_scan(level_sums, level_sums, inclusive=True)
        front_offset = empty(n_sn + 1, wp.int64)
        block_offset = empty(n_sn + 1, wp.int64)
        f_base_d = empty(n_sn + 1, wp.int32)
        b_base_d = empty(n_sn + 1, wp.int32)
        level_offsets = odt.empty_2d((4, n_sn + 1), wp.int32, device=device)
        column_base_d = empty(n_sn, wp.int32)
        _launch.launch(
            kernel_cholesky.split_terms,
            dim=n_sn + 1,
            inputs=[terms, level_sums, level_nodes],
            outputs=[front_offset, block_offset, f_base_d, b_base_d, level_offsets,
                     column_base_d],
            device=device,
        )  # fmt: skip
        sibling_sorted = empty(2 * n_sn, wp.uint64)
        members = empty(2 * n_sn, wp.int32)
        child_keys_view = _launch.view(child_keys, 0, n_sn)
        _launch.launch(
            kernel_cholesky.sibling_keys,
            dim=n_sn,
            inputs=[child_keys_view, _launch.view(child_values, 0, n_sn), s.height, sentinel],
            outputs=[sibling_sorted, members],
            device=device,
        )
        _launch.radix_sort_pairs(sibling_sorted, members, n_sn, end_bit=sn_bits)
        sibling_view = _launch.view(sibling_sorted, 0, n_sn)
        group_ends = empty(n_sn, wp.int32)
        _launch.launch(
            kernel_cholesky.extend_group_flags,
            dim=n_sn,
            inputs=[sibling_view, sentinel],
            outputs=[group_ends],
            device=device,
        )
        _launch.array_scan(group_ends, group_ends, inclusive=True)
        groups = odt.as_array2d(_launch.zeros((n_sn, 4), dtype=wp.int32, device=device), wp.int32)
        _launch.launch(
            kernel_cholesky.extend_group_tables,
            dim=n_sn,
            inputs=[sibling_view, members, group_ends, s.nrow, sentinel],
            outputs=[groups],
            device=device,
        )
        header = int(kernel_cholesky.SCHEDULE_HEADER)
        fields = int(kernel_cholesky.LEVEL_FIELDS)
        work_base = header + (n_levels + 1) * fields
        schedule = empty(work_base + n_levels + n // panel + 1 + 4 * n_sn, wp.int32)
        _launch.launch(
            kernel_cholesky.schedule_levels,
            dim=1,
            inputs=[level_stats, level_offsets, s.row_ends, group_ends, terms],
            outputs=[schedule],
            device=device,
        )
        _launch.launch(
            kernel_cholesky.schedule_work,
            dim=n_sn,
            inputs=[level_nodes, s.ncol, self.front_size, s.height, groups, wp.int32(n_levels)],
            outputs=[schedule],
            device=device,
        )
        # Read back, the analysis' last two syncs: the header (distinct rows, groups, panels and
        # the terms' totals), which sizes the rest of the schedule (a row a level; the panels'
        # work; the groups), read second.
        head = [int(value) for value in read_values(schedule, 0, header)]
        n_rows, n_groups, n_panels = head[:3]
        terms_at = int(kernel_cholesky.SCHEDULE_TERMS)
        front_entries, block_entries, f_parts, b_parts, f_tasks, b_tasks = (
            (head[terms_at + 2 * k] << 31) + head[terms_at + 2 * k + 1] for k in range(6)
        )
        table = odt.as_dense(schedule[: work_base + n_panels + 4 * n_groups]).numpy()
        levels = table[header:work_base].reshape(n_levels + 1, fields)
        work = table[work_base : work_base + n_panels].tolist()
        group_table = table[work_base + n_panels :].reshape(n_groups, 4)
        self.front_entries = front_entries
        self.block_entries = block_entries
        last_terms = levels[n_levels]
        self.n_pairs = max(n_rows, 1)
        self.n_forward_parts = max(f_parts, 1)
        self.n_backward_parts = max(b_parts, 1)
        self.n_stored = s.n_stored
        n_entries = s.n_entries
        n_f, n_r, n_b, n_c = (int(value) for value in last_terms[5:9])
        widest = max(int(levels[h][1]) for h in range(n_levels))
        n_components = s.n_components
        self.chunk_bound = n_components + n // int(kernel_cholesky.NULL_SPACE_CHUNK)
        self.n_components = n_components
        self.nbytes = _analysis_nbytes(
            n=n,
            n_sn=n_sn,
            row_capacity=s.rows.size,
            n_rows=n_rows,
            n_entries=n_entries,
            # The task totals from the header, which no ``int32`` level sum wraps; the front rows
            # are at most ``front_entries``, refused below long before they could wrap.
            tasks=(f_tasks, max(n_r, 0), b_tasks, n_c),
            widest=widest,
            n_components=n_components,
            extends=n_groups > 0,
        )
        # The budget is the first factorization's peak: what it holds plus the larger of the
        # transient fronts arena and a one-column solve's scratch. It also bounds every count the
        # maps index in ``int32``: each costs at least 8 bytes an item there, so one past ``2^31``
        # is past any budget below 16 GiB.
        needed = self.nbytes + self.numeric_nbytes() + max(8 * front_entries, self.solve_nbytes(1))
        if needed > CHOLESKY_MEMORY_BUDGET:
            raise ValueError(
                f"sparse_cholesky: the factorization needs {needed} bytes, above "
                f"CHOLESKY_MEMORY_BUDGET ({CHOLESKY_MEMORY_BUDGET})"
            )
        self.front_offset = _launch.view(front_offset, 0, n_sn)
        self.block_offset = _launch.view(block_offset, 0, n_sn)
        self.ncol = s.ncol
        self.c0 = s.c0
        self.parent = parent_clamped
        self.perm = s.perm
        self.owner = s.owner
        self.rows = _launch.view(s.rows, 0, n_rows + 1)
        self.row_offsets = s.row_offsets
        self.entry_source = empty(n_entries, wp.int32)
        self.entry_target = empty(n_entries, wp.int64)
        self.entry_row = empty(n_entries, wp.int32)
        self.entry_col = empty(n_entries, wp.int32)
        sizes = [self.c0, self.ncol, self.front_size]
        _launch.launch(
            kernel_cholesky.write_lower_entries,
            dim=n,
            inputs=[offsets, columns, s.pos, self.owner, *sizes, self.front_offset,
                    self.row_offsets, self.rows, s.lower_ends],
            outputs=[self.entry_source, self.entry_target, self.entry_row, self.entry_col],
            device=device,
        )  # fmt: skip
        self.parent_local = _launch.zeros(n_rows + 1, dtype=wp.int32, device=device)
        if n_rows:
            _launch.launch(
                kernel_cholesky.parent_locals,
                dim=n_rows,
                inputs=[self.rows, s.row_node, s.parent, *sizes, self.row_offsets],
                outputs=[self.parent_local],
                device=device,
            )
        self.extend: list[list[tuple[wp.array[wp.int32], int]]] = [[] for _ in range(n_levels)]
        for level, start, end, width in group_table.tolist():
            self.extend[level].append((_launch.view(members, start, end), width * width))
        # The per-level task maps, every level in one launch per map.
        f_offsets, r_offsets, b_offsets, c_offsets = (
            odt.as_dense(level_offsets[k]) for k in range(4)
        )
        f_block, f_stride, f_count_d, f_source, f_slot = (
            empty(n_f, wp.int64),
            *(empty(n_f, wp.int32) for _ in range(4)),
        )
        r_slot, r_stride, r_count, r_target = (empty(n_r, wp.int32) for _ in range(4))
        b_block = empty(n_b, wp.int64)
        b_first, b_ncol, b_size, b_start, b_rows, b_slot = (empty(n_b, wp.int32) for _ in range(6))
        c_slot, c_stride, c_first, c_count, c_target = (empty(n_c, wp.int32) for _ in range(5))
        _launch.launch(
            kernel_cholesky.forward_task_map,
            dim=n_f,
            inputs=[level_nodes, f_offsets, self.ncol, self.front_size, self.c0,
                    self.block_offset, f_base_d],
            outputs=[f_block, f_stride, f_count_d, f_source, f_slot],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.forward_row_map,
            dim=n_r,
            inputs=[level_nodes, r_offsets, self.ncol, self.front_size, self.c0,
                    self.row_offsets, f_base_d],
            outputs=[r_slot, r_stride, r_count, r_target],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.backward_task_map,
            dim=n_b,
            inputs=[level_nodes, b_offsets, self.ncol, self.front_size, self.c0,
                    self.block_offset, self.row_offsets, b_base_d],
            outputs=[b_block, b_first, b_ncol, b_size, b_start, b_rows, b_slot],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.backward_row_map,
            dim=n_c,
            inputs=[level_nodes, c_offsets, self.ncol, self.front_size, self.c0, b_base_d],
            outputs=[c_slot, c_stride, c_first, c_count, c_target],
            device=device,
        )  # fmt: skip
        # Each column's contributions (the row-structure entries naming it), listed level-major
        # by one stable sort: the forward gather's offsets and pairs.
        contribution_offsets = _launch.zeros(n + 1, dtype=wp.int32, device=device)
        contribution_pairs = empty(2 * n_rows, wp.int32)
        if n_rows:
            contribution_keys = _launch.empty(2 * n_rows, dtype=wp.int32, device=device)
            contribution_counts = _launch.zeros(n, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.contribution_keys,
                dim=n_rows,
                inputs=[self.rows, self.owner, self.c0, column_base_d],
                outputs=[contribution_keys, contribution_pairs, contribution_counts],
                device=device,
            )  # fmt: skip
            _launch.radix_sort_pairs(
                contribution_keys, contribution_pairs, n_rows, end_bit=_bits(n)
            )
            _launch.array_scan(contribution_counts, contribution_offsets[1:], inclusive=True)
        # Each level's launches, from its schedule row: its fronts (a view of the level order),
        # the panels' work, the most rows below a first panel and the widest front; and its solve
        # maps as the ``[start, start + count)`` entries of the level-major arrays.
        self.big: list[tuple[wp.array[wp.int32], list[int], int, int] | None] = []
        self.forward: list[tuple[int, int, int, int, int, int]] = []
        self.backward: list[tuple[int, int, int, int]] = []
        for h in range(n_levels):
            start, count, panels, rows_below, max_ncol, f0, r0, b0, c0_, slot = (
                int(value) for value in levels[h]
            )
            f1, r1, b1, c1 = (int(value) for value in levels[h + 1][5:9])
            self.big.append(
                (
                    _launch.view(level_nodes, start, start + count),
                    work[slot : slot + panels],
                    max(rows_below, 1),
                    max_ncol,
                )
            )
            self.forward.append((c0_, c1 - c0_, f0, f1 - f0, r0, r1 - r0))
            self.backward.append((b0, b1 - b0, c0_, c1 - c0_))
        self.forward_maps = (
            c_target, contribution_offsets, contribution_pairs,
            (f_block, f_stride, f_count_d, f_source, f_slot), (r_slot, r_stride, r_count, r_target),
        )  # fmt: skip
        self.backward_maps = (
            (b_block, b_first, b_ncol, b_size, b_start, b_rows, b_slot),
            (c_slot, c_stride, c_first, c_count, c_target),
        )
        # ``factor_panel_tiled``'s panel inverses: one ``PANEL``-square tile a front of a level.
        self.panel_inverse = _launch.empty(widest * panel * panel, dtype=wp.float64, device=device)
        self.panel_inverse_tiles = self.panel_inverse.reshape((widest * panel, panel))
        self.panel_scratch = odt.empty_2d((widest * panel, panel), wp.float64, device=device)
        # Each component's label, start and last-eliminated row.
        self.labels = labels
        self.component_label = _launch.empty(n_components, dtype=wp.int32, device=device)
        self.component_offsets = _launch.full(n_components + 1, n, dtype=wp.int32, device=device)
        self.component_last = _launch.full(n_components, -1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.component_tables,
            dim=n,
            inputs=[component_keys, flags, ends, component_rows, s.pos],
            outputs=[self.component_label, self.component_offsets, self.component_last],
            device=device,
        )
        # The null-space projection's chunks: each component's rows in ``NULL_SPACE_CHUNK``s,
        # at most ``chunk_bound`` of them, so no count is read back.
        self.chunk_offsets = _launch.zeros(n_components + 1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.component_chunk_counts,
            dim=n_components,
            inputs=[self.component_offsets],
            outputs=[self.chunk_offsets],
            device=device,
        )
        _launch.array_scan(self.chunk_offsets, self.chunk_offsets, inclusive=True)
        self.component_diagonal = _launch.empty(n_components, dtype=wp.int64, device=device)
        _launch.launch(
            kernel_cholesky.component_diagonals,
            dim=n_components,
            inputs=[self.component_last, self.owner, self.front_offset, self.c0,
                    self.front_size],
            outputs=[self.component_diagonal],
            device=device,
        )  # fmt: skip

    def numeric(self) -> _Numeric:
        """Allocate the per-factorization state of this pattern."""
        n, device = self.n, self.device
        return _Numeric(
            singular=_launch.zeros(n, dtype=wp.int32, device=device),
            pinned=_launch.zeros(n, dtype=wp.int32, device=device),
            dropped=_launch.zeros(n, dtype=wp.int32, device=device),
            # A bad pivot (slot 0) and whether any component is singular (slot 1).
            status=_launch.zeros(2, dtype=wp.int32, device=device),
            blocks=_launch.zeros(self.block_entries, dtype=wp.float64, device=device),
        )

    def numeric_nbytes(self) -> int:
        """Device memory [`numeric`][ordito.cholesky._Plan.numeric] allocates, in bytes."""
        return 12 * self.n + 8 + 8 * self.block_entries

    def solve_nbytes(self, n_columns: int) -> int:
        """Device memory one ``n_columns`` solve configuration (``_Solve``) allocates, in bytes."""
        # ``rhs``, ``solution``, ``initial`` and ``residual``, and the work's ``gathered``,
        # ``forward`` and ``correction``: seven ``n * n_columns`` arrays.
        per_column = 7 * self.n + 1 + self.n_forward_parts + self.n_backward_parts + self.n_pairs
        null_space = max(self.chunk_bound * n_columns, 1) + max(self.n_components * n_columns, 1)
        return 8 * (per_column * n_columns + null_space) + 12

    def _empty(self, labels: wp.array[wp.int32]) -> None:
        """Plan an operator with no rows: every table empty, no level, no component."""
        device = self.device

        def empty(dtype: Any) -> wp.array[Any]:
            return _launch.empty(0, dtype=dtype, device=device)

        self.n_levels = 0
        self.front_entries = self.block_entries = 0
        self.n_stored = 0
        self.n_pairs = self.n_forward_parts = self.n_backward_parts = 1
        self.perm, self.owner, self.ncol, self.c0, self.front_size, self.parent = (
            empty(wp.int32) for _ in range(6)
        )
        self.front_offset, self.block_offset = empty(wp.int64), empty(wp.int64)
        self.rows = _launch.zeros(1, dtype=wp.int32, device=device)
        self.row_offsets = empty(wp.int32)
        self.entry_source, self.entry_row, self.entry_col = (empty(wp.int32) for _ in range(3))
        self.entry_target = empty(wp.int64)
        self.parent_local = _launch.zeros(1, dtype=wp.int32, device=device)
        self.extend = []
        self.big = []
        self.forward = []
        self.backward = []
        nothing = empty(wp.int32)
        self.forward_maps = (nothing, _launch.zeros(1, dtype=wp.int32, device=device), nothing,
                             (empty(wp.int64), *(nothing,) * 4), (nothing,) * 4)  # fmt: skip
        self.backward_maps = ((empty(wp.int64), *(nothing,) * 6), (nothing,) * 5)
        self.panel_inverse = _launch.empty(256, dtype=wp.float64, device=device)
        self.panel_inverse_tiles = self.panel_inverse.reshape((16, 16))
        self.panel_scratch = odt.empty_2d((16, 16), wp.float64, device=device)
        self.labels = labels
        self.component_rows = nothing
        self.component_label = nothing
        self.component_offsets = _launch.zeros(1, dtype=wp.int32, device=device)
        self.chunk_offsets = _launch.zeros(1, dtype=wp.int32, device=device)
        self.chunk_bound = 0
        self.component_last = nothing
        self.component_diagonal = empty(wp.int64)
        self.n_components = 0
        self.nbytes = sum(
            held.capacity
            for held in (self.rows, self.parent_local, self.forward_maps[1], self.panel_inverse,
                         self.panel_scratch, self.component_offsets, self.chunk_offsets)
        )  # fmt: skip

    def factor(
        self,
        matrix: odt.BsrMatrix[wp.float64],
        numeric: _Numeric,
        sign: float,
        fronts: wp.array[wp.float64] | None = None,
    ) -> None:
        """
        Numeric factorization of ``matrix`` into ``blocks``; ``status`` flags a bad pivot.

        The fronts arena is ``fronts`` when given (zeroed here; nothing is allocated, so the
        launches can be recorded), else allocated for this factorization alone.
        """
        device = self.device
        n = self.n
        _launch.zero_(numeric.status)
        _launch.fill_(numeric.singular, 1)
        _launch.launch(
            kernel_cholesky.mark_nonsingular_components,
            dim=n,
            inputs=[matrix.offsets, matrix.columns, matrix.values, self.labels,
                    wp.float64(_SINGULAR_ROW_SUM)],
            outputs=[numeric.singular, numeric.pinned, numeric.dropped],
            device=device,
        )  # fmt: skip
        if fronts is None:
            fronts = _launch.zeros(self.front_entries, dtype=wp.float64, device=device)
        else:
            _launch.zero_(fronts)
        _launch.launch(
            kernel_cholesky.pin_singular_components,
            dim=self.n_components,
            inputs=[self.component_label, self.component_last, self.perm, numeric.singular,
                    self.component_diagonal],
            outputs=[numeric.pinned, numeric.dropped, fronts, numeric.status],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.scatter_entries,
            dim=self.entry_source.size,
            inputs=[matrix.values, self.entry_source, self.entry_target, self.entry_row,
                    self.entry_col, numeric.pinned, wp.float64(sign)],
            outputs=[fronts],
            device=device,
        )  # fmt: skip
        _launch.zero_(numeric.blocks)
        _launch.launch(
            kernel_cholesky.seed_blocks,
            dim=self.n,
            inputs=[self.block_offset, self.front_size, self.owner, self.c0, numeric.blocks],
            device=device,
        )
        tables = [self.front_offset, self.ncol, self.front_size]
        cuda = wp.get_device(device).is_cuda
        panel = int(kernel_cholesky.PANEL)
        for h in range(self.n_levels):
            big = self.big[h]
            if big is not None:
                nodes, work, rows_below, max_ncol = big
                for p, width in enumerate(work):
                    row_dim = (nodes.size, rows_below + min((p + 1) * panel, max_ncol))
                    if cuda:
                        # The diagonal block as a tile, and the rows by its inverse. Three
                        # launches a panel: the factor is device-bound, and every fusion measured
                        # (the next block factored inside the update or rows launch, or per row
                        # group) slowed it. The tile factor's latency is least at 64 lanes.
                        _launch.launch_tiled(
                            kernel_cholesky.factor_panel_tiled,
                            dim=[nodes.size],
                            inputs=[
                                nodes,
                                *tables,
                                wp.int32(p),
                                fronts,
                                numeric.status,
                                self.panel_scratch,
                            ],
                            outputs=[self.panel_inverse_tiles],
                            block_dim=64,
                            device=device,
                        )
                        _launch.launch_tiled(
                            kernel_cholesky.panel_rows_inverse,
                            dim=[nodes.size, -(-row_dim[1] // panel)],
                            inputs=[nodes, self.front_offset, self.ncol, self.front_size,
                                    self.block_offset, wp.int32(p), wp.int32(rows_below),
                                    self.panel_inverse, fronts, numeric.blocks],
                            block_dim=panel * panel,
                            device=device,
                        )  # fmt: skip
                    else:
                        _launch.launch_tiled(
                            kernel_cholesky.factor_panel,
                            dim=[nodes.size],
                            inputs=[nodes, *tables, wp.int32(p), fronts, numeric.status],
                            block_dim=1,
                            device=device,
                        )
                        _launch.launch(
                            kernel_cholesky.panel_rows,
                            dim=row_dim,
                            inputs=[nodes, self.front_offset, self.ncol, self.front_size,
                                    self.block_offset, wp.int32(p), wp.int32(rows_below),
                                    fronts, numeric.blocks],
                            device=device,
                        )  # fmt: skip
                    _launch.launch(
                        kernel_cholesky.update_panel,
                        dim=(nodes.size, width),
                        inputs=[
                            nodes,
                            *tables,
                            self.block_offset,
                            wp.int32(p),
                            fronts,
                            numeric.blocks,
                        ],
                        device=device,
                    )
            for children, width in self.extend[h]:
                _launch.launch(
                    kernel_cholesky.extend_add,
                    dim=(children.size, width),
                    inputs=[children, self.parent, self.front_offset, self.ncol, self.front_size,
                            self.row_offsets, self.parent_local, fronts],
                    device=device,
                )  # fmt: skip

    def triangular_solve(
        self,
        rhs: wp.array[wp.float64],
        n_columns: int,
        work: _Work,
        numeric: _Numeric,
        solution: wp.array[wp.float64],
    ) -> None:
        """
        ``work.correction = A^{-1} rhs`` in the permuted order, from the factored blocks.

        The correction is also added to ``solution`` at the operator's rows, as each level of the
        backward sweep finishes its entries.
        """
        device = self.device
        n = wp.int32(self.n)
        nf, nb, npairs = (
            wp.int32(self.n_forward_parts), wp.int32(self.n_backward_parts), wp.int32(self.n_pairs)
        )  # fmt: skip
        c_target, contribution_offsets, contribution_pairs, f_tasks, f_rows = self.forward_maps
        b_tasks, c_rows = self.backward_maps
        for row_start, n_rows, task_start, n_tasks, block_row_start, n_block_rows in self.forward:
            _launch.launch(
                kernel_cholesky.forward_gather,
                dim=(n_rows, n_columns),
                inputs=[c_target, rhs, self.perm, numeric.pinned, contribution_offsets,
                        contribution_pairs, work.pair_values, n, npairs, wp.int32(row_start)],
                outputs=[work.gathered],
                device=device,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.forward_partial,
                dim=(n_tasks, n_columns),
                inputs=[*f_tasks, numeric.blocks, work.gathered, n, nf, wp.int32(task_start)],
                outputs=[work.forward_parts],
                device=device,
                block_dim=128,
            )
            _launch.launch(
                kernel_cholesky.forward_finalize,
                dim=(n_block_rows, n_columns),
                inputs=[*f_rows, work.forward_parts, n, nf, npairs, wp.int32(block_row_start)],
                outputs=[work.forward, work.pair_values],
                device=device,
            )
        for task_start, n_tasks, row_start, n_rows in reversed(self.backward):
            _launch.launch(
                kernel_cholesky.backward_partial,
                dim=(n_tasks, n_columns),
                inputs=[*b_tasks, self.rows, numeric.blocks, work.forward, work.correction, n, nb,
                        wp.int32(task_start)],
                outputs=[work.backward_parts],
                device=device,
                block_dim=128,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.backward_finalize,
                dim=(n_rows, n_columns),
                inputs=[*c_rows, numeric.pinned, work.backward_parts, self.perm,
                        solution, n, nb, wp.int32(row_start)],
                outputs=[work.correction],
                device=device,
            )  # fmt: skip


def _analysis_nbytes(
    *,
    n: int,
    n_sn: int,
    row_capacity: int,
    n_rows: int,
    n_entries: int,
    tasks: tuple[int, int, int, int],
    widest: int,
    n_components: int,
    extends: bool,
) -> int:
    """
    Device memory a ``_Plan`` of these counts holds once built, in bytes, before it allocates it.

    Every array the plan keeps, by its own allocation (a view counts its owner's): the labels and
    the components' rows, sort width; the supernode tables and the level order, the extend-add
    members (held only when some level extends); the elimination order (sort width), positions'
    owners, columns and first columns (one a row); the row structure (``row_capacity``, sized by
    the structure pairs) and each supernode's first row; the entry maps; the parents' local rows;
    the four solve task maps; the forward gather's offsets and pairs; the panel inverses and their
    scratch; and the component tables.
    """
    n_f, n_r, n_b, n_c = tasks
    # Labels; the components' rows and the elimination order (both of sort width); owners,
    # columns and first columns.
    per_row = 4 * n + 8 * n + 8 * n + 3 * 4 * n
    # Front sizes, parents and first rows; the level order (sort width); the front and solve-block
    # offsets; the extend-add members (sort width).
    per_supernode = 3 * 4 * n_sn + 8 * n_sn + 2 * 8 * (n_sn + 1) + (8 * n_sn if extends else 0)
    # The row structure; the parents' local rows; the forward gather's pairs and offsets.
    structure = 4 * row_capacity + 4 * (n_rows + 1) + 8 * n_rows + 4 * (n + 1)
    task_maps = 24 * n_f + 16 * n_r + 32 * n_b + 20 * n_c
    panels = 2 * 8 * widest * int(kernel_cholesky.PANEL) ** 2
    # Each component's label, last row and diagonal slot; its offsets and its chunks' offsets.
    components = (4 + 4 + 8) * n_components + 2 * 4 * (n_components + 1)
    return per_row + per_supernode + structure + 20 * n_entries + task_maps + panels + components


@dataclass
class _Numeric:
    """One factorization's state: its solve blocks, pins and pivot status."""

    singular: wp.array[wp.int32]
    pinned: wp.array[wp.int32]
    dropped: wp.array[wp.int32]
    status: wp.array[wp.int32]
    blocks: wp.array[wp.float64]
    any_singular: bool = True


@dataclass
class _Work:
    """Scratch of one solve configuration."""

    gathered: wp.array[wp.float64]
    forward: wp.array[wp.float64]
    correction: wp.array[wp.float64]
    forward_parts: wp.array[wp.float64]
    backward_parts: wp.array[wp.float64]
    pair_values: wp.array[wp.float64]


class _Solve:
    """One right-hand-side count's refined solve, recorded once on CUDA and replayed."""

    def __init__(
        self,
        plan: _Plan,
        numeric: _Numeric,
        matrix: odt.BsrMatrix[wp.float64],
        n_columns: int,
        tol: float,
        componentwise: bool,
        sign: float,
    ) -> None:
        device = plan.device
        self.sign = sign
        total = plan.n * n_columns
        self.key = (n_columns, tol, componentwise)
        self.plan, self.numeric, self.matrix, self.n_columns = plan, numeric, matrix, n_columns
        self.tol, self.componentwise = tol, componentwise
        # Written before any read: ``run`` copies in ``rhs`` and ``initial``, ``_body`` zeroes
        # ``rhs_scale``, ``refine_start`` writes all of ``solution`` and the state, and every
        # ``residual_test`` all of ``residual``.
        self.rhs = _launch.empty(total, dtype=wp.float64, device=device)
        self.solution = _launch.empty(total, dtype=wp.float64, device=device)
        self.initial = _launch.empty(total, dtype=wp.float64, device=device)
        self.residual = _launch.empty(total, dtype=wp.float64, device=device)
        self.rhs_scale = _launch.empty(n_columns, dtype=wp.float64, device=device)
        self.null_space_partials = _launch.empty(
            max(plan.chunk_bound * n_columns, 1), dtype=wp.float64, device=device
        )
        self.null_space_means = _launch.empty(
            max(plan.n_components * n_columns, 1), dtype=wp.float64, device=device
        )
        self.state = _launch.empty(3, dtype=wp.int32, device=device)
        # ``wp.capture_while`` reads the condition slot, the state's first.
        self.condition = cast("wp.array[int]", self.state[0:1])
        self.work = _Work(
            _launch.zeros(total, dtype=wp.float64, device=device),
            _launch.zeros(total, dtype=wp.float64, device=device),
            _launch.zeros(total, dtype=wp.float64, device=device),
            _launch.zeros(plan.n_forward_parts * n_columns, dtype=wp.float64, device=device),
            _launch.zeros(plan.n_backward_parts * n_columns, dtype=wp.float64, device=device),
            _launch.zeros(plan.n_pairs * n_columns, dtype=wp.float64, device=device),
        )
        self.graph: Any = None
        self.recorded_for: Any = None
        self.ran_for: Any = None

    def _test(self) -> None:
        plan, device = self.plan, self.plan.device
        matrix = self.matrix
        _launch.launch(
            kernel_cholesky.residual_test,
            dim=(plan.n, self.n_columns),
            inputs=[matrix.offsets, matrix.columns, matrix.values, self.solution, self.rhs,
                    self.rhs_scale, self.numeric.dropped, wp.float64(self.sign), wp.int32(plan.n),
                    wp.float64(self.tol), wp.int32(1 if self.componentwise else 0), self.state],
            outputs=[self.residual],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.refine_advance,
            dim=1,
            inputs=[wp.int32(CHOLESKY_REFINE_ROUNDS), self.state],
            device=device,
        )

    def _round(self) -> None:
        plan = self.plan
        plan.triangular_solve(self.residual, self.n_columns, self.work, self.numeric, self.solution)
        self._test()

    def _body(self) -> None:
        plan, device = self.plan, self.plan.device
        # Refinement starts from the caller's solution: a warm start that already passes the test
        # costs one residual and no triangular solve. ``refine_start`` copies it in.
        _launch.zero_(self.rhs_scale)
        _launch.launch(
            kernel_cholesky.refine_start,
            dim=(plan.n, self.n_columns),
            inputs=[
                self.rhs,
                wp.int32(plan.n),
                self.state,
                self.rhs_scale,
                wp.int32(1 if self.componentwise else 0),
                self.initial,
            ],
            outputs=[self.solution],
            device=device,
        )
        self._test()
        wp.capture_while(self.condition, self._round)
        if plan.n_components and self.numeric.any_singular:
            # The chunk sums and the shift run over the whole device, not a block per component:
            # a mesh's one component is every row. Between them one thread a component adds its
            # chunks, in order, into its mean.
            chunk_bound = wp.int32(plan.chunk_bound)
            singular = self.numeric.singular
            _launch.launch_tiled(
                kernel_cholesky.null_space_partials,
                dim=[plan.chunk_bound * self.n_columns],
                inputs=[plan.component_offsets, plan.component_rows, plan.component_label,
                        plan.chunk_offsets, chunk_bound, singular, self.initial, self.solution,
                        wp.int32(plan.n)],
                outputs=[self.null_space_partials],
                block_dim=256 if wp.get_device(device).is_cuda else 1,
                device=device,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.null_space_means,
                dim=(plan.n_components, self.n_columns),
                inputs=[plan.component_offsets, plan.component_label, plan.chunk_offsets,
                        chunk_bound, singular, self.null_space_partials],
                outputs=[self.null_space_means],
                device=device,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.null_space_shift,
                dim=(plan.n, self.n_columns),
                inputs=[plan.component_offsets, plan.component_rows, plan.component_label,
                        singular, self.null_space_means, wp.int32(plan.n), self.solution],
                device=device,
            )  # fmt: skip

    def run(self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64], at_once: bool) -> None:
        device = wp.get_device(self.plan.device)
        _launch.copy(self.rhs, rhs)
        _launch.copy(self.initial, solution)
        # The storage a recording replays: the operator's device pointers, which the graph holds
        # (an array's ``id`` can be reused by a new array at another address).
        identity = (
            self.matrix.offsets.ptr,
            self.matrix.columns.ptr,
            self.matrix.values.ptr,
            self.numeric.any_singular,
        )
        # Recorded on the second run against one operator storage, not the first, unless the
        # factorization says it is solved many times: a one-shot solve (a verified fallback, a call
        # that drops its factorization) runs its few refinement rounds directly, one four-byte
        # read each, rather than paying a recording it never replays.
        repeat = at_once or self.ran_for == identity
        self.ran_for = identity
        if device.is_cuda and wp.is_conditional_graph_supported() and repeat:
            if self.graph is None or self.recorded_for != identity:
                with wp.ScopedCapture(device) as capture:
                    self._body()
                self.graph = capture.graph
                self.recorded_for = identity
            wp.capture_launch(self.graph)
        else:
            self._body()
        _launch.copy(solution, self.solution)


def _structure(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    n: int,
    node: wp.array[wp.int32],
    tree: odt.Array2dInt32,
    state: wp.array[wp.int32],
    depths: int,
    component_ends: wp.array[wp.int32],
) -> _Structure:
    """
    Merge each tree's top, lay the supernodes out in postorder and derive the row structure.

    All on the device, from the dissection's tree: one read of the counts that size what
    follows (supernodes, levels, structure pairs, lower entries and, from ``component_ends``,
    components) with the stored entry count and the pattern's symmetry flag, and the row
    structure's distinct count left on the device for the schedule's.
    """
    device = node.device
    # Supernodes: each tree's top merged into its root while it holds at most
    # ``CHOLESKY_TOP_ROWS`` rows, whole depths at a time (the root's at least), every other node
    # a supernode of its own; the groups in elimination order (one sort of the nodes by their
    # group's first position, then their own), each supernode's columns, parent and height.
    width = depths + 2
    depth_rows = _launch.zeros(n * width, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.top_depth_rows,
        dim=tree.shape[0],
        inputs=[tree, state, wp.int32(width)],
        outputs=[depth_rows],
        device=device,
    )
    group_bits = _bits((n + 1) * (n + 1))
    sentinel = wp.uint64((1 << group_bits) - 1)
    group_keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
    nodes = _launch.empty(2 * n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.supernode_keys,
        dim=n,
        inputs=[tree, state, depth_rows, wp.int32(width), wp.int32(CHOLESKY_TOP_ROWS),
                wp.int32(n), sentinel],
        outputs=[group_keys, nodes],
        device=device,
    )  # fmt: skip
    _launch.radix_sort_pairs(group_keys, nodes, n, end_bit=group_bits)
    group_ends = _launch.empty(n, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_cholesky.supernode_flags,
        dim=n,
        inputs=[group_keys, nodes, tree, state, wp.int32(n)],
        outputs=[group_ends],
        device=device,
    )
    _launch.array_scan(group_ends, group_ends, inclusive=True)
    node_start = _launch.empty(n, dtype=wp.int32, device=device)
    node_supernode = _launch.empty(n, dtype=wp.int32, device=device)
    c0 = _launch.empty(n, dtype=wp.int32, device=device)
    ncol = _launch.zeros(n, dtype=wp.int32, device=device)
    kept = _launch.empty(n, dtype=wp.int32, device=device)
    header = _launch.zeros(kernel_cholesky.HEADER_SLOTS, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.supernode_tables,
        dim=n,
        inputs=[group_keys, nodes, group_ends, tree, state, wp.int32(n)],
        outputs=[node_start, node_supernode, c0, ncol, kept, header],
        device=device,
    )
    parent = _launch.empty(n, dtype=wp.int32, device=device)
    column_end = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.supernode_links,
        dim=n,
        inputs=[kept, tree, node_supernode, c0, ncol, header],
        outputs=[parent, column_end],
        device=device,
    )
    height = _launch.zeros(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.supernode_heights,
        dim=n,
        inputs=[parent],
        outputs=[height, header],
        device=device,
    )
    # The elimination order: a stable sort of the vertices by their node's first position.
    keys = _launch.empty(2 * n, dtype=wp.int32, device=device)
    vertices = _launch.empty(2 * n, dtype=wp.int32, device=device)
    pos = _launch.empty(n, dtype=wp.int32, device=device)
    owner = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.node_order_keys,
        dim=n,
        inputs=[node, node_start],
        outputs=[keys, vertices],
        device=device,
    )
    _launch.radix_sort_pairs(keys, vertices, n, end_bit=_bits(n))
    _launch.launch(
        kernel_cholesky.scatter_positions,
        dim=n,
        inputs=[vertices, node, node_supernode],
        outputs=[pos, owner],
        device=device,
    )
    perm = odt.as_dense(vertices[:n])
    # The operator's lower entries in the new order, for the entry maps and the estimate, and
    # whether its pattern is structurally symmetric.
    lower_ends = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.count_lower_entries,
        dim=n,
        inputs=[offsets, columns, pos],
        outputs=[lower_ends, header],
        device=device,
    )
    _launch.array_scan(lower_ends, lower_ends, inclusive=True)
    # Row structure as paths: every operator entry's row up the tree from its column's supernode
    # to its own, all pairs written, sorted by ``(height, supernode, row)`` and deduplicated once
    # -- the rows level-major, each supernode's ascending.
    path_inputs = [offsets, columns, pos, owner, column_end, parent]
    path_counts = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.count_structure_paths,
        dim=n,
        inputs=path_inputs,
        outputs=[path_counts],
        device=device,
    )
    _launch.array_scan(path_counts, path_counts, inclusive=True)
    _launch.launch(
        kernel_cholesky.structure_counts,
        dim=1,
        inputs=[path_counts, lower_ends, component_ends, offsets],
        outputs=[header],
        device=device,
    )
    # Read back, the analysis' first sync, one copy: the supernode and level counts, the structure
    # pairs' count (their sort's size), the lower entries, the components, the stored entries and
    # the symmetry flag.
    n_sn, n_levels, n_pairs, n_entries, n_components, n_stored, asymmetric = (
        int(value) for value in read_values(header, 0, kernel_cholesky.HEADER_SLOTS)
    )
    if asymmetric:
        raise ValueError(
            "sparse_cholesky: the operator's stored pattern is not symmetric (an entry (i, j) is "
            "stored without (j, i)); store both, explicit zeros included"
        )
    shift = _bits(n_sn * n)
    if shift + _bits(n_levels) > 64:
        raise ValueError("sparse_cholesky: the factorization exceeds 64-bit row keys")
    row_start = _launch.zeros(n_sn, dtype=wp.int32, device=device)
    nrow = _launch.zeros(n_sn, dtype=wp.int32, device=device)
    # Sized by the pairs, which bound the distinct rows: the schedule reads their count.
    rows = _launch.zeros(n_pairs + 1, dtype=wp.int32, device=device)
    row_node = _launch.empty(max(n_pairs, 1), dtype=wp.int32, device=device)
    row_ends = _launch.zeros(max(n_pairs, 1), dtype=wp.int32, device=device)
    if n_pairs:
        keys_pairs = _launch.empty(2 * n_pairs, dtype=wp.uint64, device=device)
        values = _launch.empty(2 * n_pairs, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.write_structure_paths,
            dim=n,
            inputs=[*path_inputs, height, wp.int32(n), wp.int32(shift), path_counts],
            outputs=[keys_pairs, values],
            device=device,
        )
        _launch.radix_sort_pairs(keys_pairs, values, n_pairs, end_bit=shift + _bits(n_levels))
        _launch.launch(
            kernel_grouping.MARK_SORTED_RUN_STARTS[wp.uint64],
            dim=n_pairs,
            inputs=[keys_pairs],
            outputs=[row_ends],
            device=device,
        )
        _launch.array_scan(row_ends, row_ends, inclusive=True)
        _launch.launch(
            kernel_cholesky.emit_structure_rows,
            dim=n_pairs,
            inputs=[keys_pairs, row_ends, wp.uint64((1 << shift) - 1), wp.int32(n)],
            outputs=[rows, row_node, row_start, nrow],
            device=device,
        )
    return _Structure(
        perm, pos, owner, _launch.view(ncol, 0, n_sn), nrow, _launch.view(c0, 0, n_sn),
        _launch.view(parent, 0, n_sn), _launch.view(height, 0, n_sn), rows, row_node, row_start,
        row_ends, lower_ends, n_entries, n_stored, n_sn, n_levels, n_components,
    )  # fmt: skip


@dataclass
class _Structure:
    """The supernode tree in elimination order with each supernode's row structure, on device."""

    perm: wp.array[wp.int32]
    pos: wp.array[wp.int32]
    owner: wp.array[wp.int32]
    ncol: wp.array[wp.int32]
    nrow: wp.array[wp.int32]
    c0: wp.array[wp.int32]
    parent: wp.array[wp.int32]
    height: wp.array[wp.int32]
    # Every supernode's rows, ascending, height by height (``row_offsets`` is each supernode's
    # first, ``row_node`` each entry's supernode), in a buffer sized by the structure pairs; the
    # distinct count is ``row_ends``' last entry.
    rows: wp.array[wp.int32]
    row_node: wp.array[wp.int32]
    row_offsets: wp.array[wp.int32]
    row_ends: wp.array[wp.int32]
    # Each operator row's lower entries, scanned inclusively, and their total.
    lower_ends: wp.array[wp.int32]
    n_entries: int
    # The operator's stored entries, ``offsets[n]``.
    n_stored: int
    n_sn: int
    n_levels: int
    n_components: int


def _nested_dissection(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], n: int, coordinates: wp.array[wp.vec3]
) -> tuple[wp.array[wp.int32], odt.Array2dInt32, wp.array[wp.int32], int]:
    """
    Every subset of one depth bisected at once: each vertex's node, and the tree.

    A subset is split at the median of its projection onto its principal axis, and the separator
    is the smaller of the two endpoint sets of the pattern edges crossing the split. A subset of at
    most ``CHOLESKY_LEAF_ROWS`` rows is a leaf; a split with no crossing edge (two components)
    makes no node, and its halves hang from the subset's parent. Each subset owns an interval of
    the elimination order -- its lower half's, its upper half's, then its separator's rows -- so a
    node's first position orders the tree in postorder.

    Returns each row's node, the tree -- one row a node: first position, rows, parent, depth,
    root -- the dissection state whose first slot counts the nodes, and the bound on the depths.
    The tree stays on the device, its nodes numbered by scans: a depth's separators in segment
    order, then its leaves in child order. Nothing is read back: every buffer has its largest
    size, so the depths are issued back to back -- a segment's halves have at most
    ``ceil(count / 2)`` rows, so the depth count is bounded before the first one runs, and a
    depth past the last segment launches over empty slots.
    """
    device = coordinates.device
    node = _launch.empty(n, dtype=wp.int32, device=device)
    if n <= CHOLESKY_LEAF_ROWS:
        # One leaf rooting its own tree (``n >= 1``): every row in node 0.
        _launch.zero_(node)
        (leaf,) = _upload(device, wp.int32, np.array([0, n, -1, 0, 0]))
        (state,) = _upload(device, wp.int32, np.array([1, 0]))
        return node, odt.as_array2d(leaf.reshape((1, 5)), wp.int32), state, 0
    depths = 0
    largest = n
    while largest > CHOLESKY_LEAF_ROWS:
        depths += 1
        largest -= largest // 2
    cuda = wp.get_device(device).is_cuda
    chunk = kernel_cholesky.DISSECTION_CHUNK_CUDA if cuda else kernel_cholesky.DISSECTION_CHUNK_CPU
    # Every depth's buffers, allocated once at their largest: a depth's live vertices are at most
    # ``n``, and its segments each hold more than ``CHOLESKY_LEAF_ROWS`` of them. The survivors are
    # compacted into ``order`` itself, which the depth has finished reading by then.
    slots = max(1, n // (CHOLESKY_LEAF_ROWS + 1))
    max_tasks = slots + n // chunk + 1
    order = arange(n, device=device)
    order_segment = _launch.zeros(n, dtype=wp.int32, device=device)
    tag = _launch.full(n, -1, dtype=wp.int32, device=device)
    lower_mark = _launch.zeros(n, dtype=wp.int32, device=device)
    upper_mark = _launch.zeros(n, dtype=wp.int32, device=device)
    partials = _launch.empty(max_tasks, dtype=kernel_cholesky.moment_vector, device=device)
    centre = _launch.empty(slots, dtype=wp.vec3d, device=device)
    axis = _launch.empty(slots, dtype=wp.vec3d, device=device)
    keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
    vertices = _launch.empty(2 * n, dtype=wp.int32, device=device)
    # The crossing counts start zeroed; ``dissection_link`` clears each segment's once read.
    counts = _launch.zeros(2 * slots, dtype=wp.int32, device=device)
    keep = _launch.empty(n, dtype=wp.int32, device=device)
    next_segment = _launch.empty(n, dtype=wp.int32, device=device)
    separator_side = _launch.empty(slots, dtype=wp.int32, device=device)
    separator_node = _launch.empty(slots, dtype=wp.int32, device=device)
    child_next = _launch.empty(2 * slots, dtype=wp.int32, device=device)
    child_leaf = _launch.empty(2 * slots, dtype=wp.int32, device=device)
    # Two generations of segment tables -- sizes after a zero slot, then task counts after one,
    # scanned as one into the starts and the live-shifted task offsets -- and of segment rows
    # (first position, parent, its depth and root). The first depth's one segment is every vertex.
    tables = [_launch.zeros(2 * (slots + 1), dtype=wp.int32, device=device) for _ in range(2)]
    segments = [odt.empty_2d((4, slots), wp.int32, device=device) for _ in range(2)]
    _launch.launch(
        kernel_cholesky.seed_dissection,
        dim=tables[0].size,
        inputs=[wp.int32(n), wp.int32(chunk)],
        outputs=[tables[0], segments[0]],
        device=device,
    )
    # The tree has at most one node a row (every node holds at least one).
    tree = odt.empty_2d((n, 5), wp.int32, device=device)
    state = _launch.zeros(2, dtype=wp.int32, device=device)
    leaf_rows = wp.int32(CHOLESKY_LEAF_ROWS)
    # The sentinel key past the live vertices carries segment ``slots + 1`` (above every real one).
    sort_bits = 32 + _bits(slots + 1)
    block_dim = 128 if cuda else 1
    # The bookkeeping block's lanes: its packed tallies hold a chunk of at most 512 slots.
    bookkeeping_lanes = 512 if cuda else 1

    def split(
        depth: int,
        table: wp.array[wp.int32],
        segment: odt.Array2dInt32,
        next_table: wp.array[wp.int32],
        next_segments: odt.Array2dInt32,
    ) -> None:
        """Split every segment of one depth; the depth itself lives in ``state``."""
        segment_start = _launch.view(table, 0, slots + 1)
        tasks = _launch.view(table, slots + 1, table.size)
        # A depth has at most ``2^depth`` segments, each a task more than its chunks.
        live = min(1 << depth, slots)
        _launch.launch_tiled(
            kernel_cholesky.dissection_partials,
            dim=[min(live + n // chunk + 1, max_tasks)],
            inputs=[order, coordinates, segment_start, tasks, wp.int32(chunk)],
            outputs=[partials],
            block_dim=block_dim,
            device=device,
        )
        _launch.launch(
            kernel_cholesky.dissection_axes,
            dim=live,
            inputs=[order, coordinates, segment_start, tasks, partials],
            outputs=[centre, axis],
            device=device,
        )
        _launch.launch(
            kernel_cholesky.dissection_keys,
            dim=n,
            inputs=[order, order_segment, coordinates, centre, axis, segment_start],
            outputs=[keys, vertices],
            device=device,
        )
        # The top 32 bits only -- the segment and the projection's leading bits -- in four radix
        # passes instead of six: projections closer than that keep their order (a stable sort),
        # an equally valid median.
        _launch.radix_sort_pairs(
            keys, vertices, n, begin_bit=max(sort_bits - 32, 0), end_bit=sort_bits
        )
        _launch.launch(
            kernel_cholesky.dissection_sides,
            dim=n,
            inputs=[keys, vertices, segment_start, state],
            outputs=[tag],
            device=device,
        )
        _launch.launch(
            kernel_cholesky.dissection_crossings,
            dim=n,
            inputs=[keys, vertices, segment_start, offsets, columns, tag, state],
            outputs=[lower_mark, upper_mark, counts],
            device=device,
        )
        _launch.launch_tiled(
            kernel_cholesky.dissection_bookkeeping,
            dim=[1],
            inputs=[segment_start, segment, wp.int32(live), leaf_rows,
                    wp.int32(chunk), counts, state],
            outputs=[tree, separator_side, separator_node, child_next, child_leaf, next_table,
                     next_segments],
            block_dim=bookkeeping_lanes,
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.dissection_relabel,
            dim=n,
            inputs=[keys, vertices, segment_start, separator_side, separator_node, child_next,
                    child_leaf, lower_mark, upper_mark, state],
            outputs=[node, keep, next_segment],
            device=device,
        )  # fmt: skip
        _launch.array_scan(keep, keep, inclusive=True)
        _launch.launch(
            kernel_cholesky.dissection_compact,
            dim=n,
            inputs=[vertices, keep, next_segment, state],
            outputs=[order, order_segment],
            device=device,
        )

    # Issued directly: recording two depths and replaying them was slower (the scans' and the
    # sort's allocation nodes make the recording cost more than the replays save).
    for depth in range(depths):
        split(depth, tables[depth % 2], segments[depth % 2],
              tables[(depth + 1) % 2], segments[(depth + 1) % 2])  # fmt: skip
    return node, tree, state, depths


def _bits(value: int) -> int:
    """Bits a radix sort needs for keys up to ``value``."""
    return max(int(value).bit_length(), 1)


def _upload(device: wp.DeviceLike, dtype: Any, *tables: npt.NDArray[Any]) -> list[wp.array[Any]]:
    """Upload host tables in one copy as ``dtype``; return each as a view of the one buffer."""
    host = np.concatenate([np.ravel(table) for table in tables] + [np.zeros(1)]).astype(
        wp.dtype_to_numpy(dtype)
    )
    buffer = _launch.array(host, dtype=dtype, device=device)
    views: list[wp.array[Any]] = []
    start = 0
    for table in tables:
        # The trailing pad keeps every start inside the buffer, an empty table's included.
        views.append(_launch.view(buffer, start, start + table.size))
        start += table.size
    return views


def _landmark_coordinates(
    matrix: odt.BsrMatrix[wp.float64], labels: wp.array[wp.int32]
) -> wp.array[wp.vec3]:
    """Return hop distances from three mutually far vertices per component: a pattern embedding."""
    n = int(matrix.nrow)
    device = matrix.device
    coordinates = _launch.zeros(n, dtype=wp.vec3, device=device)
    if n == 0:
        # No row, no search: slicing an empty axis would raise.
        return coordinates
    if not wp.get_device(device).is_cuda:
        # The CPU device runs a launch grid as one loop, so a search sweeping every row per level
        # costs ``n`` times the graph's diameter; the host's frontier search is linear.
        coordinates.numpy()[:] = _host_landmark_coordinates(matrix, labels)
        return coordinates
    axes = coordinates.view(wp.float32)
    size = _launch.zeros(n, dtype=wp.int32, device=device)
    first = _launch.full(n, n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_cholesky.component_sizes,
        dim=n,
        inputs=[labels],
        outputs=[size, first],
        device=device,
    )
    distance = _launch.empty(n, dtype=wp.int32, device=device)
    state = _launch.empty(3, dtype=wp.int32, device=device)
    seeds = _launch.empty(n, dtype=wp.int32, device=device)
    key = _launch.empty(n, dtype=wp.uint64, device=device)
    offsets, columns = matrix.offsets, matrix.columns

    def level_round() -> None:
        for step in range(int(kernel_cholesky.BFS_LEVELS_PER_ROUND)):
            _launch.launch(
                kernel_cholesky.breadth_first_level,
                dim=n,
                inputs=[offsets, columns, wp.int32(step), state, distance],
                device=device,
            )
        _launch.launch(kernel_cholesky.breadth_first_advance, dim=1, inputs=[state], device=device)

    # Every component is searched at once: a search from one seed per component.
    search = record_device_loop(device, odt.as_dense(state[0:1]), level_round)
    # The first search, from each component's lowest row, finds the first landmark; each later
    # one is the row farthest from the landmarks so far (by the nearer of the last two).
    # Each search's distances become a coordinate in the launch that reads them for the next
    # seed (``farthest_rows``), and the last search's in one of their own after the loop.
    for axis, (a, b) in enumerate(((None, None), (0, 0), (0, 0), (0, 1))):
        if a is None or b is None:
            seed_inputs = [key, size, first, wp.int32(CHOLESKY_LEAF_ROWS), wp.int32(1)]
        else:
            _launch.zero_(key)
            _launch.launch(
                kernel_cholesky.farthest_rows,
                dim=n,
                inputs=[labels, distance, axes[:, a], axes[:, b], wp.int32(1), wp.int32(1)],
                outputs=[axes[:, max(axis - 2, 0)], key],
                device=device,
            )
            seed_inputs = [key, size, first, wp.int32(CHOLESKY_LEAF_ROWS), wp.int32(0)]
        _launch.launch(
            kernel_cholesky.farthest_seeds,
            dim=n,
            inputs=seed_inputs,
            outputs=[seeds],
            device=device,
        )
        _launch.launch(
            kernel_cholesky.seed_breadth_first,
            dim=n,
            inputs=[labels, seeds],
            outputs=[distance, state],
            device=device,
        )
        search()
    _launch.launch(
        kernel_cholesky.farthest_rows,
        dim=n,
        inputs=[labels, distance, None, None, wp.int32(1), wp.int32(0)],
        outputs=[axes[:, 2], None],
        device=device,
    )
    return coordinates


def _host_landmark_coordinates(
    matrix: odt.BsrMatrix[wp.float64], labels: wp.array[wp.int32]
) -> npt.NDArray[np.float32]:
    """
    ``_landmark_coordinates`` by frontier searches on the host, every component at once.

    The device path's sequence: a search from each component's lowest row, then from the row
    farthest from it, from the row farthest from that, and from the row farthest from the nearer
    of the last two (the lowest row on a tie), each a single multi-source search over all
    components -- linear in the pattern however many components there are.
    """
    n = int(matrix.nrow)
    offsets = matrix.offsets.numpy()[: n + 1].astype(np.int64)
    columns = matrix.columns.numpy()[: int(offsets[-1])].astype(np.int64)
    labels_np = labels.numpy().astype(np.int64)
    coords = np.zeros((n, 3), np.float32)
    sizes = np.bincount(labels_np, minlength=n)
    first = np.full(n, n, np.int64)
    np.minimum.at(first, labels_np, np.arange(n, dtype=np.int64))
    searched = np.flatnonzero(sizes > CHOLESKY_LEAF_ROWS)
    if searched.size == 0:
        return coords
    inside = sizes[labels_np] > CHOLESKY_LEAF_ROWS
    d0 = _hop_distances(offsets, columns, n, first[searched])
    a = _hop_distances(offsets, columns, n, _farthest_rows(d0, labels_np, inside, searched))
    b = _hop_distances(offsets, columns, n, _farthest_rows(a, labels_np, inside, searched))
    nearer = np.minimum(a, b)
    c = _hop_distances(offsets, columns, n, _farthest_rows(nearer, labels_np, inside, searched))
    coords[inside] = np.stack([a[inside], b[inside], c[inside]], 1)
    return coords


def _farthest_rows(
    distance: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    inside: npt.NDArray[np.bool_],
    searched: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Each searched component's farthest row by ``distance``, its lowest on a tie."""
    n = distance.size
    rows = np.flatnonzero(inside)
    # The larger distance first, then the lower row: one key per row, maximized per component.
    key = distance[rows] * (n + 1) + (n - rows)
    best = np.full(n, -1, np.int64)
    np.maximum.at(best, labels[rows], key)
    return n - best[searched] % (n + 1)


def _hop_distances(
    offsets: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
    n: int,
    seeds: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Breadth-first hop distance from the nearest of ``seeds``; ``-1`` where none reaches."""
    distance = np.full(n, -1, np.int64)
    frontier = np.unique(seeds)
    distance[frontier] = 0
    level = 0
    while frontier.size:
        level += 1
        reached = _neighbors(offsets, columns, frontier)
        reached = np.unique(reached[distance[reached] < 0])
        distance[reached] = level
        frontier = reached
    return distance


def _neighbors(
    offsets: npt.NDArray[np.int64], columns: npt.NDArray[np.int64], rows: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Return the pattern's column indices of ``rows``, concatenated."""
    return columns[_ranges(offsets[rows], offsets[rows + 1] - offsets[rows])]


def _ranges(starts: npt.NDArray[np.int64], counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Concatenated ``arange(starts[i], starts[i] + counts[i])``."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    first = np.cumsum(counts) - counts
    return np.repeat(starts.astype(np.int64) - first, counts) + np.arange(total)
