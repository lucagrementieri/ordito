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

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.constants import TILE_1D
from ordito.graph import connected_component_labels
from ordito.kernels import cholesky as kernel_cholesky
from ordito.kernels import reduce as kernel_reduce

# Largest leaf of the nested-dissection tree, in rows.
CHOLESKY_LEAF_ROWS = 32
# The top of each dissection tree is merged into one supernode while it holds at most this many
# rows: fewer levels for every solve, one dense front for the factorization.
CHOLESKY_TOP_ROWS = 512
# Device memory a factorization may take, in bytes; ``sparse_cholesky`` refuses a larger one.
CHOLESKY_MEMORY_BUDGET = 2 << 30
# Refinement rounds a solve runs at most.
CHOLESKY_REFINE_ROUNDS = 12

# A row sums to "zero" below this fraction of its diagonal.
_SINGULAR_ROW_SUM = 1e-10
_INT32_MAX = 2**31 - 1


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
        Device memory the factorization holds, in bytes.
    negated : bool
        Whether ``-matrix`` was factored (a negative semi-definite operator).
    """

    def __init__(self, plan: _Plan, matrix: odt.BsrMatrix[wp.float64], *, negated: bool) -> None:
        """Factor ``matrix`` (or its negation) over ``plan``'s structure."""
        self.negated = negated
        self.n = plan.n
        self.device = plan.device
        self.nbytes = plan.nbytes
        self._plan = plan
        self._numeric = plan.numeric()
        self._solves: dict[int, _Solve] = {}
        self._matrix = matrix
        self.refactor(matrix)

    def refactor(self, matrix: odt.BsrMatrix[wp.float64]) -> None:
        """
        Factor a new operator of the same sparsity pattern in place.

        Parameters
        ----------
        matrix
            ``(n, n)`` symmetric ``float64`` operator whose stored pattern is the one this
            factorization was built for (the same row offsets and column indices).

        Raises
        ------
        ValueError
            If the operator is not positive definite on its non-singular components (a pivot that
            is not positive).
        RuntimeError
            If ``matrix`` is not on this factorization's device.
        """
        require_same_device(matrix=matrix, factor=self)
        plan = self._plan
        plan.factor(matrix, self._numeric, -1.0 if self.negated else 1.0)
        if read_scalar(self._numeric.status, 0) != 0:
            raise ValueError("sparse_cholesky: the operator is not positive definite")
        self._matrix = matrix
        for solve in self._solves.values():
            solve.matrix = matrix

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
            Refinement stops once no entry moves by more than ``tol`` times the column's largest
            entry -- or, with ``componentwise``, times its own magnitude.
        componentwise
            Refine every entry to its own size: for solutions spanning many orders of magnitude.

        Raises
        ------
        ValueError
            If ``rhs`` or ``solution`` is not ``float64`` of a matching shape.
        RuntimeError
            If ``rhs`` and ``solution`` are not on this factorization's device.
        """
        require_same_device(rhs=rhs, solution=solution, factor=self)
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
        solve.run(rhs.flatten(), solution.flatten())


def sparse_cholesky(
    matrix: odt.BsrMatrix[wp.float64],
    coordinates: wp.array[wp.vec3] | None = None,
    *,
    negated: bool = False,
) -> SparseCholesky:
    """
    Sparse Cholesky factorization of a symmetric positive-(semi-)definite ``float64`` operator.

    Orders the operator by nested dissection, factors it on the device and returns the
    factorization, whose [`solve`][ordito.cholesky.SparseCholesky.solve] answers any later
    right-hand side in a fixed number of device passes. Costs a host pass over the sparsity pattern
    (and the coordinates, when given), each read back once.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric scalar ``float64`` operator, positive definite apart from components
        whose rows all sum to zero (solved up to a constant, see the module notes).
    coordinates
        ``(n,)`` positions of the rows, for a geometric ordering: a mesh's vertices for a vertex
        operator. When ``None``, the ordering is computed from the pattern alone.
    negated
        Factor ``-matrix`` instead, for a negative (semi-)definite operator such as
        [`cotmatrix`][ordito.laplacian.cotmatrix]; solves still answer ``matrix x = rhs``.

    Returns
    -------
    SparseCholesky
        The factorization, on ``matrix.device``.

    Raises
    ------
    ValueError
        If ``matrix`` is not square scalar ``float64``, ``coordinates`` does not have ``n`` rows,
        the factorization would exceed
        [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or the operator is not
        positive definite.
    RuntimeError
        If ``matrix`` and ``coordinates`` are not on one device.

    See Also
    --------
    [`SparseCholesky`][ordito.cholesky.SparseCholesky]
    [`solve_spd`][ordito.linalg.solve_spd]

    Examples
    --------
    ```python
    import ordito as od

    system = od.heat.heat_operators(v, f)[0]  # the heat method's M - t L
    factor = od.cholesky.sparse_cholesky(system, v)
    rhs = wp.ones(system.nrow, dtype=wp.float64, device=v.device)
    solution = wp.zeros_like(rhs)
    factor.solve(rhs, solution)
    ```
    """
    require_same_device(matrix=matrix, coordinates=coordinates)
    plan = _plan_cholesky(matrix, coordinates)
    return SparseCholesky(plan, matrix, negated=negated)


def reused_sparse_cholesky(
    matrix: odt.BsrMatrix[wp.float64],
    coordinates: wp.array[wp.vec3] | None = None,
    *,
    negated: bool = False,
) -> SparseCholesky | None:
    """
    Sparse Cholesky factorization of ``matrix`` once its sparsity pattern repeats, else ``None``.

    For a caller that assembles a fresh operator on every call over one mesh, where a single solve
    does not repay a factorization's analysis but every later one does: the first request for a
    pattern records it and returns ``None`` (the caller iterates instead); from the second on, the
    pattern's analysis is built once and each request factors the operator's current values. The
    pattern is recognized on the device (one 24-byte read), not by reading it back.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric scalar ``float64`` operator, as for
        [`sparse_cholesky`][ordito.cholesky.sparse_cholesky].
    coordinates
        ``(n,)`` positions of the rows, for the ordering, or ``None``.
    negated
        Factor ``-matrix`` instead.

    Returns
    -------
    SparseCholesky | None
        The factorization, or ``None`` on a pattern's first request -- and where none can be built
        (over [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or not definite).

    Raises
    ------
    RuntimeError
        If ``matrix`` and ``coordinates`` are not on one device.

    See Also
    --------
    [`sparse_cholesky`][ordito.cholesky.sparse_cholesky]
    """
    require_same_device(matrix=matrix, coordinates=coordinates)
    if matrix.values.dtype != wp.float64 or int(matrix.nrow) != int(matrix.ncol):
        return None
    key = (str(matrix.device), int(matrix.nrow), *_pattern_key(matrix))
    if key in _PATTERNS_REFUSED:
        return None
    if key not in _PLAN_CACHE and key not in _PATTERNS_SEEN:
        if len(_PATTERNS_SEEN) >= _PATTERNS_SEEN_ENTRIES:
            _PATTERNS_SEEN.clear()
        _PATTERNS_SEEN.add(key)
        return None
    try:
        return SparseCholesky(_plan_cholesky(matrix, coordinates, key), matrix, negated=negated)
    except ValueError:
        # Over the budget or not definite: not retried for this pattern.
        if len(_PATTERNS_REFUSED) >= _PATTERNS_SEEN_ENTRIES:
            _PATTERNS_REFUSED.clear()
        _PATTERNS_REFUSED.add(key)
        return None


# The patterns ``reused_sparse_cholesky`` was asked for once, and those it could not factor;
# bounded like linalg's shape census.
_PATTERNS_SEEN: set[tuple[object, ...]] = set()
_PATTERNS_REFUSED: set[tuple[object, ...]] = set()
_PATTERNS_SEEN_ENTRIES = 256


def _plan_cholesky(
    matrix: odt.BsrMatrix[wp.float64],
    coordinates: wp.array[wp.vec3] | None = None,
    key: tuple[object, ...] | None = None,
) -> _Plan:
    """Return the symbolic half of ``sparse_cholesky``: ordering, structure, every device map."""
    if matrix.values.dtype != wp.float64 or int(matrix.nrow) != int(matrix.ncol):
        raise ValueError("sparse_cholesky factors a square scalar float64 operator")
    n = int(matrix.nrow)
    if coordinates is not None and coordinates.size != n:
        raise ValueError(f"sparse_cholesky: coordinates has {coordinates.size} rows, expected {n}")
    if key is None:
        key = (str(matrix.device), n, *_pattern_key(matrix))
    plan = _PLAN_CACHE.pop(key, None)
    if plan is None:
        offsets = matrix.offsets.numpy()[: n + 1].astype(np.int64)
        columns = matrix.columns.numpy()[: int(offsets[-1])].astype(np.int64)
        coords = None if coordinates is None else coordinates.numpy().astype(np.float64)
        labels = connected_component_labels(matrix).numpy() if n > 0 else np.zeros(0, np.int32)
        plan = _Plan(offsets, columns, n, coords, labels, matrix.device)
        while len(_PLAN_CACHE) >= _PLAN_CACHE_ENTRIES:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = plan
    if plan.nbytes > CHOLESKY_MEMORY_BUDGET:
        raise ValueError(
            f"sparse_cholesky: the factorization needs {plan.nbytes} bytes, above "
            f"CHOLESKY_MEMORY_BUDGET ({CHOLESKY_MEMORY_BUDGET})"
        )
    return plan


def _pattern_key(matrix: odt.BsrMatrix[wp.float64]) -> tuple[int, int, int]:
    """Return ``matrix``'s stored-entry count and a 128-bit fingerprint of its pattern."""
    n = int(matrix.nrow)
    key = _launch.zeros(3, dtype=wp.uint64, device=matrix.device)
    if n > 0:
        _launch.launch_tiled(
            kernel_cholesky.pattern_checksum,
            dim=[kernel_reduce.blocks_1d(n)],
            inputs=[matrix.offsets, matrix.columns, wp.int32(n)],
            outputs=[key],
            block_dim=TILE_1D,
            device=matrix.device,
        )
    count, first, second = (int(x) for x in read_values(key, 0, 3))
    return count, first, second


# ``_plan_cholesky``'s analyses, least recently used first: each holds its pattern's device maps
# (not a factor), bounded so a long session of distinct meshes does not accumulate them.
_PLAN_CACHE: dict[tuple[object, ...], _Plan] = {}
_PLAN_CACHE_ENTRIES = 4


class _Plan:
    """Device maps for one sparsity pattern: factorization arenas, scatter maps, solve tasks."""

    def __init__(
        self,
        offsets: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        n: int,
        coords: npt.NDArray[np.float64] | None,
        labels: npt.NDArray[np.int32],
        device: Any,
    ) -> None:
        self.n = n
        self.device = device
        if coords is None:
            coords = _landmark_coordinates(offsets, columns, n, labels)
        s = _structure(offsets, columns, n, coords)
        chunk, panel = int(kernel_cholesky.CHUNK), int(kernel_cholesky.PANEL)
        ncol, nrow, c0 = s.ncol, s.nrow, s.c0
        size = ncol + nrow
        n_sn = ncol.size
        front_offset = np.r_[0, np.cumsum(size * size)[:-1]].astype(np.int64)
        block_offset = np.r_[0, np.cumsum(size * ncol)[:-1]].astype(np.int64)
        self.front_entries = int((size * size).sum())
        self.block_entries = int((size * ncol).sum())
        owner = np.repeat(np.arange(n_sn), ncol)
        row_key = s.row_node * n + s.row_value

        def local(node: npt.NDArray[np.int64], p: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            inside = p < c0[node] + ncol[node]
            in_rows = np.searchsorted(row_key, node * n + p) - s.row_offsets[node]
            return np.where(inside, p - c0[node], ncol[node] + in_rows)

        # Operator entries: the lower triangle of the permuted operator, into their fronts.
        lower = np.flatnonzero(s.entry_row >= s.entry_col)
        entry_node = owner[s.entry_col[lower]]
        entry_target = (
            front_offset[entry_node]
            + local(entry_node, s.entry_row[lower]) * size[entry_node]
            + (s.entry_col[lower] - c0[entry_node])
        )
        # Extend-add: each child's lower Schur complement into its parent's front, one launch per
        # (child height, sibling slot) so no two children of a launch share a parent.
        child = np.flatnonzero((s.parent >= 0) & (nrow > 0))
        sibling_order = np.lexsort((child, s.parent[child]))
        sorted_parent = s.parent[child][sibling_order]
        group_start = np.searchsorted(sorted_parent, sorted_parent)
        slot = np.empty(child.size, np.int64)
        slot[sibling_order] = np.arange(child.size) - group_start
        row_parent = np.where(s.parent[s.row_node] >= 0, s.parent[s.row_node], 0)
        parent_local = local(row_parent, s.row_value)
        extend_key = s.height[child] * (n_sn + 1) + slot
        extend_order = np.argsort(extend_key, kind="stable")
        extend_groups, extend_starts = np.unique(extend_key[extend_order], return_index=True)
        extend_bounds = np.r_[extend_starts, child.size]
        n_levels = int(s.height.max(initial=0)) + 1
        self.n_levels = n_levels
        # Solve tasks. Forward: a solve block's rows times chunks of its columns, then each row's
        # chunks summed; a lower row is a contribution to an ancestor row, kept in the slot of its
        # row pair, and each permuted row sums the slots of the pairs that name it. Backward: the
        # block's columns times chunks of its rows, then each column's chunks summed.
        f_chunks = -(-ncol // chunk)
        f_base = np.r_[0, np.cumsum(f_chunks * size)[:-1]].astype(np.int64)
        f_node = np.repeat(np.arange(n_sn), f_chunks * size)
        f_local = _ranges(np.zeros(n_sn, np.int64), f_chunks * size)
        f_chunk, f_row = f_local // size[f_node], f_local % size[f_node]
        keep = (f_row >= ncol[f_node]) | (f_chunk * chunk <= f_row)
        f_node, f_local, f_chunk, f_row = f_node[keep], f_local[keep], f_chunk[keep], f_row[keep]
        r_node = np.repeat(np.arange(n_sn), size)
        r_row = _ranges(np.zeros(n_sn, np.int64), size)
        r_lower = r_row >= ncol[r_node]
        r_count = np.where(r_lower, f_chunks[r_node], r_row // chunk + 1)
        r_target = np.where(
            r_lower, -1 - (s.row_offsets[r_node] + r_row - ncol[r_node]), c0[r_node] + r_row
        )
        b_chunks = -(-size // chunk)
        b_base = np.r_[0, np.cumsum(b_chunks * ncol)[:-1]].astype(np.int64)
        b_node = np.repeat(np.arange(n_sn), b_chunks * ncol)
        b_local = _ranges(np.zeros(n_sn, np.int64), b_chunks * ncol)
        b_chunk, b_col = b_local // ncol[b_node], b_local % ncol[b_node]
        keep = b_chunk * chunk + chunk > b_col
        b_node, b_local, b_chunk, b_col = b_node[keep], b_local[keep], b_chunk[keep], b_col[keep]
        c_order = np.argsort(s.row_value, kind="stable")
        c_start = np.searchsorted(s.row_value[c_order], np.arange(n + 1))
        self.n_pairs = max(int(s.row_value.size), 1)
        self.n_forward_parts = max(int((f_chunks * size).sum()), 1)
        self.n_backward_parts = max(int((b_chunks * ncol).sum()), 1)
        self._estimate(size, ncol, f_node.size, b_node.size, lower.size, 0)
        if self.nbytes > CHOLESKY_MEMORY_BUDGET:
            raise ValueError(
                f"sparse_cholesky: the factorization needs {self.nbytes} bytes, above "
                f"CHOLESKY_MEMORY_BUDGET ({CHOLESKY_MEMORY_BUDGET})"
            )

        def put(values: npt.NDArray[Any], dtype: Any) -> wp.array[Any]:
            if dtype == wp.int32 and values.size and values.max(initial=0) > _INT32_MAX:
                raise ValueError("sparse_cholesky: the factorization exceeds int32 indexing")
            return _launch.array(np.ascontiguousarray(values), dtype=dtype, device=device)

        self.perm = put(s.perm.astype(np.int32), wp.int32)
        self.front_offset = put(front_offset, wp.int64)
        self.block_offset = put(block_offset, wp.int64)
        self.ncol = put(ncol.astype(np.int32), wp.int32)
        self.owner = put(owner.astype(np.int32), wp.int32)
        self.c0 = put(c0.astype(np.int32), wp.int32)
        self.front_size = put(size.astype(np.int32), wp.int32)
        self.entry_source = put(lower.astype(np.int32), wp.int32)
        self.entry_target = put(entry_target, wp.int64)
        self.entry_row = put(s.entry_row[lower].astype(np.int32), wp.int32)
        self.entry_col = put(s.entry_col[lower].astype(np.int32), wp.int32)
        self.rows = put(np.r_[s.row_value, 0].astype(np.int32), wp.int32)
        self.parent_local = put(np.r_[parent_local, 0].astype(np.int32), wp.int32)
        self.row_offsets = put(s.row_offsets[:-1].astype(np.int32), wp.int32)
        self.parent = put(np.maximum(s.parent, 0).astype(np.int32), wp.int32)
        self.extend: list[list[tuple[wp.array[wp.int32], int]]] = [[] for _ in range(n_levels)]
        for g, key in enumerate(extend_groups.tolist()):
            members = child[extend_order[extend_bounds[g] : extend_bounds[g + 1]]]
            width = int(nrow[members].max())
            self.extend[key // (n_sn + 1)].append(
                (put(members.astype(np.int32), wp.int32), width * width)
            )
        self.big: list[tuple[wp.array[wp.int32], list[int], int, int] | None] = []
        self.forward: list[dict[str, Any]] = []
        self.backward: list[dict[str, Any]] = []

        def by_level(node: npt.NDArray[np.int64]) -> list[npt.NDArray[np.int64]]:
            """Return the indices of ``node``'s entries per tree level, in their own order."""
            level = s.height[node]
            order = np.argsort(level, kind="stable")
            bounds = np.searchsorted(level[order], np.arange(n_levels + 1))
            return [order[bounds[h] : bounds[h + 1]] for h in range(n_levels)]

        f_levels, r_levels = by_level(f_node), by_level(r_node)
        b_levels, col_levels = by_level(b_node), by_level(owner)
        for h in range(n_levels):
            nodes = np.flatnonzero(s.height == h)
            big = nodes
            if big.size:
                work = []
                for p in range(int(-(-ncol[big].max() // panel))):
                    last = np.minimum((p + 1) * panel, ncol[big])
                    span = size[big] - last
                    work.append(max(int((span * span + span * last).max()), 1))
                rows_below = int((size[big] - np.minimum(panel, ncol[big])).max())
                self.big.append(
                    (
                        put(big.astype(np.int32), wp.int32),
                        work,
                        max(rows_below, 1),
                        int(ncol[big].max()),
                    )
                )
            else:
                self.big.append(None)
            level_rows = _ranges(c0[nodes], ncol[nodes])
            counts = c_start[level_rows + 1] - c_start[level_rows]
            contributions = _ranges(c_start[level_rows], counts)
            f, r, b, cols = f_levels[h], r_levels[h], b_levels[h], col_levels[h]
            fn, fc, fr = f_node[f], f_chunk[f], f_row[f]
            rn, rr = r_node[r], r_row[r]
            bn, bc, bi = b_node[b], b_chunk[b], b_col[b]
            cn = owner[cols]
            ci = cols - c0[cn]
            self.forward.append(
                {
                    "rows": put(level_rows.astype(np.int32), wp.int32),
                    "offsets": put(np.r_[0, np.cumsum(counts)].astype(np.int32), wp.int32),
                    "pairs": put(c_order[contributions].astype(np.int32), wp.int32),
                    "task_block": put(block_offset[fn] + fc * chunk * size[fn] + fr, wp.int64),
                    "task_stride": put(size[fn].astype(np.int32), wp.int32),
                    "task_count": put(
                        np.minimum(chunk, ncol[fn] - fc * chunk).astype(np.int32), wp.int32
                    ),
                    "task_source": put((c0[fn] + fc * chunk).astype(np.int32), wp.int32),
                    "task_slot": put((f_base[fn] + f_local[f]).astype(np.int32), wp.int32),
                    "row_slot": put((f_base[rn] + rr).astype(np.int32), wp.int32),
                    "row_stride": put(size[rn].astype(np.int32), wp.int32),
                    "row_count": put(r_count[r].astype(np.int32), wp.int32),
                    "row_target": put(r_target[r].astype(np.int32), wp.int32),
                    "n_rows": int(level_rows.size),
                    "n_tasks": int(fn.size),
                    "n_block_rows": int(rn.size),
                }
            )
            self.backward.append(
                {
                    "task_block": put(block_offset[bn] + bi * size[bn], wp.int64),
                    "task_first": put((bc * chunk).astype(np.int32), wp.int32),
                    "task_ncol": put(ncol[bn].astype(np.int32), wp.int32),
                    "task_size": put(size[bn].astype(np.int32), wp.int32),
                    "task_start": put(c0[bn].astype(np.int32), wp.int32),
                    "task_rows": put(s.row_offsets[bn].astype(np.int32), wp.int32),
                    "task_slot": put((b_base[bn] + b_local[b]).astype(np.int32), wp.int32),
                    "row_slot": put((b_base[cn] + ci).astype(np.int32), wp.int32),
                    "row_stride": put(ncol[cn].astype(np.int32), wp.int32),
                    "row_first": put((ci // chunk).astype(np.int32), wp.int32),
                    "row_count": put(b_chunks[cn].astype(np.int32), wp.int32),
                    "row_target": put(cols.astype(np.int32), wp.int32),
                    "n_tasks": int(bn.size),
                    "n_rows": int(cn.size),
                }
            )
        # Components, for the singular-component pins and the null-space projection.
        component_ids, component_label = np.unique(labels, return_inverse=True)
        n_components = component_ids.size
        last = np.full(n_components, -1, np.int64)
        np.maximum.at(last, component_label, s.perm.argsort())
        by_component = np.argsort(component_label, kind="stable")
        self.n_components = n_components
        self.labels = put(labels.astype(np.int32), wp.int32)
        self.component_label = put(component_ids.astype(np.int32), wp.int32)
        self.component_last = put(last.astype(np.int32), wp.int32)
        last_owner = owner[last]
        self.component_diagonal = put(
            front_offset[last_owner] + (last - c0[last_owner]) * (size[last_owner] + 1), wp.int64
        )
        self.component_offsets = put(
            np.r_[0, np.cumsum(np.bincount(component_label, minlength=n_components))].astype(
                np.int32
            ),
            wp.int32,
        )
        self.component_rows = put(by_component.astype(np.int32), wp.int32)

    def numeric(self) -> _Numeric:
        """Allocate the per-factorization state of this pattern."""
        n, device = self.n, self.device
        return _Numeric(
            singular=_launch.zeros(n, dtype=wp.int32, device=device),
            pinned=_launch.zeros(n, dtype=wp.int32, device=device),
            dropped=_launch.zeros(n, dtype=wp.int32, device=device),
            status=_launch.zeros(1, dtype=wp.int32, device=device),
            blocks=_launch.zeros(self.block_entries, dtype=wp.float64, device=device),
        )

    def _estimate(
        self,
        size: npt.NDArray[np.int64],
        ncol: npt.NDArray[np.int64],
        forward_tasks: int,
        backward_tasks: int,
        entries: int,
        extend_pairs: int,
    ) -> None:
        nbytes = (
            4 * int((size * size).sum())
            + 4 * int((size * ncol).sum())
            + 8 * (self.n_forward_parts + self.n_backward_parts + self.n_pairs)
            + 24 * forward_tasks
            + 32 * backward_tasks
            + 20 * entries
            + 16 * extend_pairs
            + 64 * self.n
        )
        self.nbytes = nbytes

    def factor(self, matrix: odt.BsrMatrix[wp.float64], numeric: _Numeric, sign: float) -> None:
        """Numeric factorization of ``matrix`` into ``blocks``; ``status`` flags a bad pivot."""
        device = self.device
        n = self.n
        _launch.zero_(numeric.status)
        _launch.fill_(numeric.singular, 1)
        _launch.launch(
            kernel_cholesky.mark_nonsingular_components,
            dim=n,
            inputs=[matrix.offsets, matrix.columns, matrix.values, self.labels,
                    wp.float64(_SINGULAR_ROW_SUM)],
            outputs=[numeric.singular],
            device=device,
        )  # fmt: skip
        _launch.zero_(numeric.pinned)
        _launch.zero_(numeric.dropped)
        _launch.launch(
            kernel_cholesky.pin_singular_components,
            dim=self.n_components,
            inputs=[self.component_label, self.component_last, self.perm, numeric.singular],
            outputs=[numeric.pinned, numeric.dropped],
            device=device,
        )
        fronts = _launch.zeros(self.front_entries, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_cholesky.scatter_entries,
            dim=self.entry_source.size,
            inputs=[matrix.values, self.entry_source, self.entry_target, self.entry_row,
                    self.entry_col, numeric.pinned, wp.float64(sign)],
            outputs=[fronts],
            device=device,
        )  # fmt: skip
        _launch.launch(
            kernel_cholesky.set_pinned_diagonals,
            dim=self.n_components,
            inputs=[self.component_label, self.component_diagonal, numeric.singular],
            outputs=[fronts],
            device=device,
        )
        _launch.zero_(numeric.blocks)
        _launch.launch(
            kernel_cholesky.seed_blocks,
            dim=self.n,
            inputs=[self.block_offset, self.front_size, self.owner, self.c0, numeric.blocks],
            device=device,
        )
        tables = [self.front_offset, self.ncol, self.front_size, self.block_offset]
        block_dim = 256 if wp.get_device(device).is_cuda else 1
        panel = int(kernel_cholesky.PANEL)
        for h in range(self.n_levels):
            big = self.big[h]
            if big is not None:
                nodes, work, rows_below, max_ncol = big
                for p, width in enumerate(work):
                    _launch.launch_tiled(
                        kernel_cholesky.factor_panel,
                        dim=[nodes.size],
                        inputs=[nodes, *tables, wp.int32(p), fronts, numeric.status],
                        # One warp: the diagonal block is ``PANEL`` columns, and one warp's
                        # barrier is the cheap one.
                        block_dim=min(block_dim, 32),
                        device=device,
                    )
                    _launch.launch(
                        kernel_cholesky.panel_rows,
                        dim=(nodes.size, rows_below + min((p + 1) * panel, max_ncol)),
                        inputs=[nodes, self.front_offset, self.ncol, self.front_size,
                                self.block_offset, wp.int32(p), wp.int32(rows_below), fronts,
                                numeric.blocks],
                        device=device,
                    )  # fmt: skip
                    _launch.launch(
                        kernel_cholesky.update_panel,
                        dim=(nodes.size, width),
                        inputs=[nodes, *tables, wp.int32(p), fronts, numeric.blocks],
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
        self, rhs: wp.array[wp.float64], n_columns: int, work: _Work, numeric: _Numeric
    ) -> None:
        """``work.correction = A^{-1} rhs`` in the permuted order, from the factored blocks."""
        device = self.device
        n = wp.int32(self.n)
        nf, nb, npairs = (
            wp.int32(self.n_forward_parts), wp.int32(self.n_backward_parts), wp.int32(self.n_pairs)
        )  # fmt: skip
        for f in self.forward:
            _launch.launch(
                kernel_cholesky.forward_gather,
                dim=(f["n_rows"], n_columns),
                inputs=[f["rows"], rhs, self.perm, numeric.pinned, f["offsets"], f["pairs"],
                        work.pair_values, n, npairs],
                outputs=[work.gathered],
                device=device,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.forward_partial,
                dim=(f["n_tasks"], n_columns),
                inputs=[f["task_block"], f["task_stride"], f["task_count"], f["task_source"],
                        f["task_slot"], numeric.blocks, work.gathered, n, nf],
                outputs=[work.forward_parts],
                device=device,
                block_dim=128,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.forward_finalize,
                dim=(f["n_block_rows"], n_columns),
                inputs=[f["row_slot"], f["row_stride"], f["row_count"], f["row_target"],
                        work.forward_parts, n, nf, npairs],
                outputs=[work.forward, work.pair_values],
                device=device,
            )  # fmt: skip
        for b in reversed(self.backward):
            _launch.launch(
                kernel_cholesky.backward_partial,
                dim=(b["n_tasks"], n_columns),
                inputs=[b["task_block"], b["task_first"], b["task_ncol"], b["task_size"],
                        b["task_start"], b["task_rows"], b["task_slot"], self.rows, numeric.blocks,
                        work.forward, work.correction, n, nb],
                outputs=[work.backward_parts],
                device=device,
                block_dim=128,
            )  # fmt: skip
            _launch.launch(
                kernel_cholesky.backward_finalize,
                dim=(b["n_rows"], n_columns),
                inputs=[b["row_slot"], b["row_stride"], b["row_first"], b["row_count"],
                        b["row_target"], numeric.pinned, work.backward_parts, n, nb],
                outputs=[work.correction],
                device=device,
            )  # fmt: skip


@dataclass
class _Numeric:
    """One factorization's state: its solve blocks, pins and pivot status."""

    singular: wp.array[wp.int32]
    pinned: wp.array[wp.int32]
    dropped: wp.array[wp.int32]
    status: wp.array[wp.int32]
    blocks: wp.array[wp.float64]


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
        self.rhs = _launch.zeros(total, dtype=wp.float64, device=device)
        self.solution = _launch.zeros(total, dtype=wp.float64, device=device)
        self.initial = _launch.zeros(total, dtype=wp.float64, device=device)
        self.residual = _launch.zeros(total, dtype=wp.float64, device=device)
        self.rhs_scale = _launch.zeros(n_columns, dtype=wp.float64, device=device)
        self.state = _launch.zeros(3, dtype=wp.int32, device=device)
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
        plan.triangular_solve(self.residual, self.n_columns, self.work, self.numeric)
        _launch.launch(
            kernel_cholesky.scatter_correction,
            dim=(plan.n, self.n_columns),
            inputs=[self.work.correction, plan.perm, wp.int32(plan.n), self.solution],
            device=plan.device,
        )
        self._test()

    def _body(self) -> None:
        plan, device = self.plan, self.plan.device
        # Refinement starts from the caller's solution: a warm start that already passes the test
        # costs one residual and no triangular solve.
        _launch.copy(self.solution, self.initial)
        _launch.zero_(self.rhs_scale)
        _launch.launch(
            kernel_cholesky.refine_start,
            dim=(plan.n, self.n_columns),
            inputs=[self.rhs, wp.int32(plan.n), self.state, self.rhs_scale],
            device=device,
        )
        self._test()
        wp.capture_while(self.condition, self._round)
        if plan.n_components:
            _launch.launch_tiled(
                kernel_cholesky.project_null_space,
                dim=[plan.n_components * self.n_columns],
                inputs=[plan.component_offsets, plan.component_rows, plan.component_label,
                        self.numeric.singular, self.initial, wp.int32(plan.n)],
                outputs=[self.solution],
                block_dim=256 if wp.get_device(device).is_cuda else 1,
                device=device,
            )  # fmt: skip

    def run(self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64]) -> None:
        device = wp.get_device(self.plan.device)
        _launch.copy(self.rhs, rhs)
        _launch.copy(self.initial, solution)
        identity = (id(self.matrix.offsets), id(self.matrix.columns), id(self.matrix.values))
        if device.is_cuda and wp.is_conditional_graph_supported():
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
    offsets: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
    n: int,
    coords: npt.NDArray[np.float64],
) -> _Structure:
    """Dissect, merge each tree's top, order by postorder and derive the row structure."""
    node_rows, parent = _nested_dissection(offsets, columns, n, coords)
    n_nodes = parent.size
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    roots: list[int] = []
    for k, p in enumerate(parent.tolist()):
        (children[p] if p >= 0 else roots).append(k)
    postorder: list[int] = []
    stack = [(r, False) for r in reversed(roots)]
    while stack:
        k, done = stack.pop()
        if done:
            postorder.append(k)
            continue
        stack.append((k, True))
        stack.extend((c, False) for c in reversed(children[k]))
    merged = np.arange(n_nodes)
    for r in roots:
        group, size, frontier = [r], node_rows[r].size, list(children[r])
        while frontier:
            extra = sum(node_rows[c].size for c in frontier)
            if size + extra > CHOLESKY_TOP_ROWS:
                break
            group += frontier
            size += extra
            frontier = [g for c in frontier for g in children[c]]
        merged[group] = r
    kept = [k for k in postorder if merged[k] == k]
    index = np.full(n_nodes, -1, np.int64)
    index[kept] = np.arange(len(kept))
    pieces: list[list[npt.NDArray[np.int64]]] = [[] for _ in kept]
    for k in postorder:
        pieces[index[merged[k]]].append(node_rows[k])
    supernode_rows = [np.concatenate(p) for p in pieces]
    sn_parent = np.full(len(kept), -1, np.int64)
    for k in kept:
        if parent[k] >= 0:
            sn_parent[index[k]] = index[merged[parent[k]]]
    perm = np.concatenate(supernode_rows) if supernode_rows else np.zeros(0, np.int64)
    pos = np.empty(n, np.int64)
    pos[perm] = np.arange(n)
    ncol = np.array([r.size for r in supernode_rows], np.int64)
    c0 = np.r_[0, np.cumsum(ncol)[:-1]].astype(np.int64)
    c1 = c0 + ncol
    n_sn = ncol.size
    height = np.zeros(n_sn, np.int64)
    for k in range(n_sn):  # postorder: children first
        if sn_parent[k] >= 0:
            height[sn_parent[k]] = max(height[sn_parent[k]], height[k] + 1)
    # Row structure, one height at a time: a supernode's rows are its columns' later neighbours
    # and its children's rows beyond its own columns.
    entry_row = pos[np.repeat(np.arange(n), np.diff(offsets))]
    entry_col = pos[columns]
    owner = np.repeat(np.arange(n_sn), ncol)
    own = entry_row > c1[owner[entry_col]] - 1
    own_node, own_row = owner[entry_col[own]], entry_row[own]
    by_height = np.argsort(height, kind="stable")
    level_start = np.searchsorted(height[by_height], np.arange(int(height.max(initial=0)) + 2))
    node_parts: list[npt.NDArray[np.int64]] = []
    row_parts: list[npt.NDArray[np.int64]] = []
    pending_node = np.zeros(0, np.int64)
    pending_row = np.zeros(0, np.int64)
    own_height = height[own_node]
    for h in range(level_start.size - 1):
        at = np.flatnonzero(own_height == h)
        inherit = height[pending_node] == h
        nodes = np.concatenate([own_node[at], pending_node[inherit]])
        values = np.concatenate([own_row[at], pending_row[inherit]])
        keep = values >= c1[nodes]
        key = np.unique(nodes[keep] * n + values[keep])
        node_h, row_h = key // n, key % n
        node_parts.append(node_h)
        row_parts.append(row_h)
        pending_node, pending_row = pending_node[~inherit], pending_row[~inherit]
        up = sn_parent[node_h] >= 0
        pending_node = np.concatenate([pending_node, sn_parent[node_h[up]]])
        pending_row = np.concatenate([pending_row, row_h[up]])
    row_node = np.concatenate(node_parts) if node_parts else np.zeros(0, np.int64)
    row_value = np.concatenate(row_parts) if row_parts else np.zeros(0, np.int64)
    order = np.lexsort((row_value, row_node))
    row_node, row_value = row_node[order], row_value[order]
    nrow = np.bincount(row_node, minlength=n_sn).astype(np.int64)
    row_offsets = np.r_[0, np.cumsum(nrow)].astype(np.int64)
    return _Structure(
        perm, ncol, nrow, c0, sn_parent, height, row_node, row_value, row_offsets,
        entry_row, entry_col,
    )  # fmt: skip


@dataclass
class _Structure:
    """The supernode tree in elimination order, with each supernode's row structure."""

    perm: npt.NDArray[np.int64]
    ncol: npt.NDArray[np.int64]
    nrow: npt.NDArray[np.int64]
    c0: npt.NDArray[np.int64]
    parent: npt.NDArray[np.int64]
    height: npt.NDArray[np.int64]
    row_node: npt.NDArray[np.int64]
    row_value: npt.NDArray[np.int64]
    row_offsets: npt.NDArray[np.int64]
    entry_row: npt.NDArray[np.int64]
    entry_col: npt.NDArray[np.int64]


def _nested_dissection(
    offsets: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
    n: int,
    coords: npt.NDArray[np.float64],
) -> tuple[list[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """
    Every subset of one depth bisected in one vectorized pass: ``(node rows, node parent)``.

    A subset is split at the median of its projection onto its principal axis, and the separator
    is the smaller of the two endpoint sets of the pattern edges crossing the split. A subset of at
    most ``CHOLESKY_LEAF_ROWS`` rows is a leaf; a split with no crossing edge (two components)
    makes no node, and its halves hang from the subset's parent.
    """
    rows_of = np.repeat(np.arange(n), np.diff(offsets))
    off_diagonal = rows_of != columns
    edge_u, edge_v = rows_of[off_diagonal], columns[off_diagonal]
    label = np.zeros(n, np.int64)
    subset_parent = [-1]
    node_rows: list[npt.NDArray[np.int64]] = []
    node_parent: list[int] = []
    n_subsets = 1
    while True:
        live = np.flatnonzero(label >= 0)
        if live.size == 0:
            break
        ids, sizes = np.unique(label[live], return_counts=True)
        is_leaf = np.zeros(n_subsets, bool)
        is_leaf[ids[sizes <= CHOLESKY_LEAF_ROWS]] = True
        leaf_mask = is_leaf[label[live]]
        leaves = live[leaf_mask]
        rows = live[~leaf_mask]
        if leaves.size:
            leaves = leaves[np.argsort(label[leaves], kind="stable")]
            leaf_ids, leaf_start = np.unique(label[leaves], return_index=True)
            for s, part in zip(leaf_ids, np.split(leaves, leaf_start[1:]), strict=True):
                node_rows.append(part)
                node_parent.append(subset_parent[s])
            label[leaves] = -1
        if rows.size == 0:
            break
        rows = rows[np.argsort(label[rows], kind="stable")]
        big, start, count = np.unique(label[rows], return_index=True, return_counts=True)
        segment = np.repeat(np.arange(big.size), count)
        p = coords[rows]
        centred = p - (np.add.reduceat(p, start) / count[:, None])[segment]
        covariance = np.add.reduceat(centred[:, :, None] * centred[:, None, :], start)
        axis = np.linalg.eigh(covariance)[1][:, :, -1]
        projection = np.einsum("ij,ij->i", centred, axis[segment])
        order = np.lexsort((projection, segment))
        rank = np.empty(rows.size, np.int64)
        rank[order] = np.arange(rows.size) - start[segment[order]]
        upper = rank >= count[segment] // 2
        side = np.zeros(n, np.int8)
        side[rows] = np.where(upper, 2, 1)
        crossing = (side[edge_u] == 1) & (side[edge_v] == 2) & (label[edge_u] == label[edge_v])
        lower_end = np.zeros(n, bool)
        lower_end[edge_u[crossing]] = True
        upper_end = np.zeros(n, bool)
        upper_end[edge_v[crossing]] = True
        lower_count = np.bincount(segment[lower_end[rows]], minlength=big.size)
        upper_count = np.bincount(segment[upper_end[rows]], minlength=big.size)
        take_lower = lower_count <= upper_count
        separator = np.where(take_lower[segment], lower_end[rows], upper_end[rows])
        new_ids = n_subsets + 2 * segment + upper.astype(np.int64)
        n_subsets += 2 * big.size
        separator_parts = np.split(
            rows[separator], np.cumsum(np.bincount(segment[separator], minlength=big.size))[:-1]
        )
        for s, part in zip(big.tolist(), separator_parts, strict=True):
            if part.size:
                node_rows.append(part)
                node_parent.append(subset_parent[s])
                me = len(node_rows) - 1
            else:
                me = subset_parent[s]
            subset_parent.extend([me, me])
        label[rows] = np.where(separator, -1, new_ids)
        # A split that shrank neither half (every projection equal) would never terminate: such a
        # subset becomes one node.
        kept = label[rows] >= 0
        half_sizes = np.bincount(
            label[rows][kept] - (n_subsets - 2 * big.size), minlength=2 * big.size
        ).reshape(-1, 2)
        for i in np.flatnonzero(half_sizes.max(1) >= count):
            part = rows[start[i] : start[i] + count[i]]
            part = part[label[part] >= 0]
            node_rows.append(part)
            node_parent.append(subset_parent[big[i]])
            label[part] = -1
    return node_rows, np.array(node_parent, np.int64)


def _landmark_coordinates(
    offsets: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
    n: int,
    labels: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Return hop distances from three mutually far vertices per component: a pattern embedding."""
    coords = np.zeros((n, 3))
    first = np.unique(labels, return_index=True)[1]
    sizes = np.bincount(labels, minlength=n)
    for seed in first[sizes[labels[first]] > CHOLESKY_LEAF_ROWS]:
        d0 = _hop_distances(offsets, columns, n, int(seed))
        inside = d0 >= 0
        a = _hop_distances(offsets, columns, n, int(np.argmax(d0)))
        b = _hop_distances(offsets, columns, n, int(np.argmax(np.where(inside, a, -1))))
        c = _hop_distances(
            offsets, columns, n, int(np.argmax(np.where(inside, np.minimum(a, b), -1)))
        )
        coords[inside] = np.stack([a[inside], b[inside], c[inside]], 1)
    return coords


def _hop_distances(
    offsets: npt.NDArray[np.int64], columns: npt.NDArray[np.int64], n: int, seed: int
) -> npt.NDArray[np.int64]:
    """Breadth-first hop distance from ``seed``; ``-1`` off its component."""
    distance = np.full(n, -1, np.int64)
    frontier = np.array([seed], np.int64)
    distance[seed] = 0
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
