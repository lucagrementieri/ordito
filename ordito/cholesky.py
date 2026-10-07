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
from ordito._device import read_scalar, read_values, record_device_loop, require_same_device
from ordito.array import arange
from ordito.graph import connected_component_labels
from ordito.kernels import cholesky as kernel_cholesky

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
    right-hand side in a fixed number of device passes. The ordering and the symbolic analysis run
    on the device too, reading back a few small counts per level of the dissection; an analysis
    is kept per sparsity pattern, so a later operator of the same pattern skips it.

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
    plan = _Plan(matrix, coordinates, labels)
    if plan.nbytes > CHOLESKY_MEMORY_BUDGET:
        raise ValueError(
            f"sparse_cholesky: the factorization needs {plan.nbytes} bytes, above "
            f"CHOLESKY_MEMORY_BUDGET ({CHOLESKY_MEMORY_BUDGET})"
        )
    return plan


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
        offsets, columns = matrix.offsets, matrix.columns
        node, tree = _nested_dissection(offsets, columns, n, coordinates)
        s = _structure(offsets, columns, n, node, tree)
        chunk, panel = int(kernel_cholesky.CHUNK), int(kernel_cholesky.PANEL)
        ncol, nrow, c0 = s.ncol, s.nrow, s.c0
        size = ncol + nrow
        n_sn = ncol.size
        front_offset = np.r_[0, np.cumsum(size * size)[:-1]].astype(np.int64)
        block_offset = np.r_[0, np.cumsum(size * ncol)[:-1]].astype(np.int64)
        self.front_entries = int((size * size).sum())
        self.block_entries = int((size * ncol).sum())
        n_levels = int(s.height.max(initial=-1)) + 1
        self.n_levels = n_levels
        # Supernodes level by level (height, then elimination order): every per-level map below is
        # one array in this order, each level a view of it.
        level_node = np.argsort(s.height, kind="stable")
        level_bounds = np.searchsorted(s.height[level_node], np.arange(n_levels + 1))
        # Solve tasks. Forward: a solve block's rows times chunks of its columns, then each row's
        # chunks summed; a lower row is a contribution to an ancestor row, kept in the slot of its
        # row pair, and each permuted row sums the slots of the pairs that name it. Backward: the
        # block's columns times chunks of its rows, then each column's chunks summed. A chunk
        # skips the rows (forward) or columns (backward) where ``L11^-1`` is zero.
        f_chunks = -(-ncol // chunk)
        b_chunks = -(-size // chunk)
        f_count = f_chunks * size - (chunk // 2) * f_chunks * (f_chunks - 1)
        full = ncol // chunk
        b_count = (chunk // 2) * full * (full + 1) + (b_chunks - full) * ncol
        f_base = np.r_[0, np.cumsum(f_chunks * size)[:-1]].astype(np.int64)
        b_base = np.r_[0, np.cumsum(b_chunks * ncol)[:-1]].astype(np.int64)
        self.n_pairs = max(s.n_rows, 1)
        self.n_forward_parts = max(int((f_chunks * size).sum()), 1)
        self.n_backward_parts = max(int((b_chunks * ncol).sum()), 1)
        # Extend-add: each child's lower Schur complement into its parent's front, one launch per
        # (child height, sibling slot) so no two children of a launch share a parent.
        child = np.flatnonzero((s.parent >= 0) & (nrow > 0))
        sibling_order = np.lexsort((child, s.parent[child]))
        sorted_parent = s.parent[child][sibling_order]
        group_start = np.searchsorted(sorted_parent, sorted_parent)
        slot = np.empty(child.size, np.int64)
        slot[sibling_order] = np.arange(child.size) - group_start
        extend_key = s.height[child] * (n_sn + 1) + slot
        extend_order = np.argsort(extend_key, kind="stable")
        extend_groups, extend_starts = np.unique(extend_key[extend_order], return_index=True)
        extend_bounds = np.r_[extend_starts, child.size]
        # The operator's lower entries: counted first, for the memory estimate.
        lower_counts = _launch.empty(n, dtype=wp.int32, device=device)
        lower_ends = _launch.empty(n, dtype=wp.int32, device=device)
        if n:
            _launch.launch(
                kernel_cholesky.count_lower_entries,
                dim=n,
                inputs=[offsets, columns, s.pos],
                outputs=[lower_counts],
                device=device,
            )
            _launch.array_scan(lower_counts, lower_ends, inclusive=True)
        # Read back: the entry count sizes the entry maps and the estimate.
        n_entries = int(read_scalar(lower_ends, -1)) if n else 0
        self._estimate(size, ncol, int(f_count.sum()), int(b_count.sum()), n_entries, 0)
        if self.nbytes > CHOLESKY_MEMORY_BUDGET:
            raise ValueError(
                f"sparse_cholesky: the factorization needs {self.nbytes} bytes, above "
                f"CHOLESKY_MEMORY_BUDGET ({CHOLESKY_MEMORY_BUDGET})"
            )
        if max(self.n_forward_parts, self.n_backward_parts, self.n_pairs, n_entries) > _INT32_MAX:
            raise ValueError("sparse_cholesky: the factorization exceeds int32 indexing")

        def put(values: npt.NDArray[Any], dtype: Any) -> wp.array[Any]:
            host = np.ascontiguousarray(values, dtype=wp.dtype_to_numpy(dtype))
            return _launch.array(host, dtype=dtype, device=device)

        def level_offsets(count: npt.NDArray[np.int64]) -> tuple[wp.array[wp.int32], list[int]]:
            """Return the level-major prefix of ``count`` on the device and its level bounds."""
            prefix = np.r_[0, np.cumsum(count[level_node])].astype(np.int64)
            return put(prefix.astype(np.int32), wp.int32), prefix[level_bounds].tolist()

        def empty(count: int, dtype: Any) -> wp.array[Any]:
            return _launch.empty(count, dtype=dtype, device=device)

        self.perm = s.perm
        self.front_offset = put(front_offset, wp.int64)
        self.block_offset = put(block_offset, wp.int64)
        self.ncol = put(ncol.astype(np.int32), wp.int32)
        self.owner = s.owner
        self.c0 = put(c0.astype(np.int32), wp.int32)
        self.front_size = put(size.astype(np.int32), wp.int32)
        self.rows = s.rows
        self.row_offsets = s.row_offsets
        self.parent = put(np.maximum(s.parent, 0).astype(np.int32), wp.int32)
        self.entry_source = empty(n_entries, wp.int32)
        self.entry_target = empty(n_entries, wp.int64)
        self.entry_row = empty(n_entries, wp.int32)
        self.entry_col = empty(n_entries, wp.int32)
        sizes = [self.c0, self.ncol, self.front_size]
        if n:
            _launch.launch(
                kernel_cholesky.write_lower_entries,
                dim=n,
                inputs=[offsets, columns, s.pos, self.owner, *sizes, self.front_offset,
                        self.row_offsets, self.rows, lower_ends],
                outputs=[self.entry_source, self.entry_target, self.entry_row, self.entry_col],
                device=device,
            )  # fmt: skip
        self.parent_local = _launch.zeros(s.n_rows + 1, dtype=wp.int32, device=device)
        if s.n_rows:
            _launch.launch(
                kernel_cholesky.parent_locals,
                dim=s.n_rows,
                inputs=[self.rows, s.row_node, put(s.parent.astype(np.int32), wp.int32), *sizes,
                        self.row_offsets],
                outputs=[self.parent_local],
                device=device,
            )  # fmt: skip
        self.extend: list[list[tuple[wp.array[wp.int32], int]]] = [[] for _ in range(n_levels)]
        if child.size:
            members = put(child[extend_order].astype(np.int32), wp.int32)
            for g, key in enumerate(extend_groups.tolist()):
                first, last = int(extend_bounds[g]), int(extend_bounds[g + 1])
                width = int(nrow[child[extend_order[first:last]]].max())
                self.extend[key // (n_sn + 1)].append(
                    (odt.as_dense(members[first:last]), width * width)
                )
        # The per-level task maps, every level in one launch per map.
        level_nodes = put(level_node.astype(np.int32), wp.int32)
        f_offsets, f_levels = level_offsets(f_count)
        r_offsets, r_levels = level_offsets(size)
        b_offsets, b_levels = level_offsets(b_count)
        c_offsets, c_levels = level_offsets(ncol)
        n_f, n_r, n_b, n_c = f_levels[-1], r_levels[-1], b_levels[-1], c_levels[-1]
        f_block, f_stride, f_count_d, f_source, f_slot = (
            empty(n_f, wp.int64),
            *(empty(n_f, wp.int32) for _ in range(4)),
        )
        r_slot, r_stride, r_count, r_target = (empty(n_r, wp.int32) for _ in range(4))
        b_block = empty(n_b, wp.int64)
        b_first, b_ncol, b_size, b_start, b_rows, b_slot = (empty(n_b, wp.int32) for _ in range(6))
        c_slot, c_stride, c_first, c_count, c_target = (empty(n_c, wp.int32) for _ in range(5))
        f_base_d = put(f_base.astype(np.int32), wp.int32)
        b_base_d = put(b_base.astype(np.int32), wp.int32)
        if n_levels:
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
        column_base = np.empty(n_sn, np.int64)
        column_base[level_node] = np.r_[0, np.cumsum(ncol[level_node])[:-1]]
        contribution_offsets = _launch.zeros(n + 1, dtype=wp.int32, device=device)
        contribution_pairs = empty(2 * s.n_rows, wp.int32)
        if s.n_rows:
            contribution_keys = _launch.empty(2 * s.n_rows, dtype=wp.int32, device=device)
            contribution_counts = _launch.zeros(n, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.contribution_keys,
                dim=s.n_rows,
                inputs=[self.rows, self.owner, self.c0, put(column_base, wp.int32)],
                outputs=[contribution_keys, contribution_pairs, contribution_counts],
                device=device,
            )  # fmt: skip
            _launch.radix_sort_pairs(
                contribution_keys, contribution_pairs, s.n_rows, end_bit=_bits(n)
            )
            _launch.array_scan(contribution_counts, contribution_offsets[1:], inclusive=True)
        self.big: list[tuple[wp.array[wp.int32], list[int], int, int] | None] = []
        self.forward: list[dict[str, Any]] = []
        self.backward: list[dict[str, Any]] = []
        for h in range(n_levels):
            nodes = level_node[level_bounds[h] : level_bounds[h + 1]]
            work = []
            for p in range(int(-(-ncol[nodes].max() // panel))):
                last = np.minimum((p + 1) * panel, ncol[nodes])
                span = size[nodes] - last
                work.append(max(int((span * span + span * last).max()), 1))
            rows_below = int((size[nodes] - np.minimum(panel, ncol[nodes])).max())
            self.big.append(
                (
                    odt.as_dense(level_nodes[int(level_bounds[h]) : int(level_bounds[h + 1])]),
                    work,
                    max(rows_below, 1),
                    int(ncol[nodes].max()),
                )
            )
            f = slice(f_levels[h], f_levels[h + 1])
            r = slice(r_levels[h], r_levels[h + 1])
            b = slice(b_levels[h], b_levels[h + 1])
            c = slice(c_levels[h], c_levels[h + 1])
            self.forward.append(
                {
                    "rows": c_target[c],
                    "offsets": contribution_offsets[c_levels[h] : c_levels[h + 1] + 1],
                    "pairs": contribution_pairs,
                    "task_block": f_block[f],
                    "task_stride": f_stride[f],
                    "task_count": f_count_d[f],
                    "task_source": f_source[f],
                    "task_slot": f_slot[f],
                    "row_slot": r_slot[r],
                    "row_stride": r_stride[r],
                    "row_count": r_count[r],
                    "row_target": r_target[r],
                    "n_rows": c_levels[h + 1] - c_levels[h],
                    "n_tasks": f_levels[h + 1] - f_levels[h],
                    "n_block_rows": r_levels[h + 1] - r_levels[h],
                }
            )
            self.backward.append(
                {
                    "task_block": b_block[b],
                    "task_first": b_first[b],
                    "task_ncol": b_ncol[b],
                    "task_size": b_size[b],
                    "task_start": b_start[b],
                    "task_rows": b_rows[b],
                    "task_slot": b_slot[b],
                    "row_slot": c_slot[c],
                    "row_stride": c_stride[c],
                    "row_first": c_first[c],
                    "row_count": c_count[c],
                    "row_target": c_target[c],
                    "n_tasks": b_levels[h + 1] - b_levels[h],
                    "n_rows": c_levels[h + 1] - c_levels[h],
                }
            )
        # Components, for the singular-component pins and the null-space projection: the vertices
        # in label order (one stable sort), each component's label, start and last-eliminated row.
        self.labels = labels
        component_keys = _launch.empty(2 * max(n, 1), dtype=wp.int32, device=device)
        component_rows = _launch.empty(2 * max(n, 1), dtype=wp.int32, device=device)
        self.component_rows = component_rows
        n_components = 0
        if n:
            _launch.launch(
                kernel_cholesky.seed_index_sort,
                dim=n,
                inputs=[labels],
                outputs=[component_keys, component_rows],
                device=device,
            )
            _launch.radix_sort_pairs(component_keys, component_rows, n, end_bit=_bits(n))
            flags = _launch.empty(n, dtype=wp.int32, device=device)
            ends = _launch.empty(n, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.mark_label_runs,
                dim=n,
                inputs=[component_keys],
                outputs=[flags],
                device=device,
            )
            _launch.array_scan(flags, ends, inclusive=True)
            # Read back: the component count sizes the component tables and their launches.
            n_components = int(read_scalar(ends, n - 1))
            self.component_label = _launch.empty(n_components, dtype=wp.int32, device=device)
            self.component_offsets = _launch.full(
                n_components + 1, n, dtype=wp.int32, device=device
            )
            self.component_last = _launch.full(n_components, -1, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.component_tables,
                dim=n,
                inputs=[component_keys, flags, ends, component_rows, s.pos],
                outputs=[self.component_label, self.component_offsets, self.component_last],
                device=device,
            )
            self.component_diagonal = _launch.empty(n_components, dtype=wp.int64, device=device)
            _launch.launch(
                kernel_cholesky.component_diagonals,
                dim=n_components,
                inputs=[self.component_last, self.owner, self.front_offset, self.c0,
                        self.front_size],
                outputs=[self.component_diagonal],
                device=device,
            )  # fmt: skip
        else:
            self.component_label = _launch.empty(0, dtype=wp.int32, device=device)
            self.component_offsets = _launch.zeros(1, dtype=wp.int32, device=device)
            self.component_last = _launch.empty(0, dtype=wp.int32, device=device)
            self.component_diagonal = _launch.empty(0, dtype=wp.int64, device=device)
        self.n_components = n_components

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
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    n: int,
    node: wp.array[wp.int32],
    tree: _Tree,
) -> _Structure:
    """Merge each tree's top, lay the supernodes out in postorder and derive the row structure."""
    device = node.device
    n_nodes = tree.size.size
    # Each tree's top is merged into its root while it holds at most ``CHOLESKY_TOP_ROWS`` rows,
    # whole depths at a time (the root at least).
    width = int(tree.depth.max(initial=0)) + 1
    depth_key, depth_index = np.unique(tree.root * width + tree.depth, return_inverse=True)
    depth_rows = np.cumsum(np.bincount(depth_index, weights=tree.size).astype(np.int64))
    depth_root = depth_key // width
    first = np.searchsorted(depth_root, depth_root)
    depth_rows -= np.r_[0, depth_rows][first]
    fitting = np.bincount(depth_root[depth_rows <= CHOLESKY_TOP_ROWS], minlength=max(n_nodes, 1))
    merged = tree.depth <= np.maximum(fitting[tree.root] - 1, 0)
    # The nodes in elimination order: a node's rows end its subtree's interval, so ordering by
    # first row is a postorder; a merged top moves to its root's place, its nodes in postorder.
    group_start = np.where(merged, tree.start[tree.root], tree.start)
    sequence = np.lexsort((tree.start, group_start))
    node_start = np.empty(n_nodes, np.int64)
    node_start[sequence] = np.cumsum(tree.size[sequence]) - tree.size[sequence]
    kept = np.flatnonzero(~merged | (tree.root == np.arange(n_nodes)))
    kept = kept[np.argsort(group_start[kept])]
    node_supernode = np.searchsorted(group_start[kept], group_start)
    n_sn = kept.size
    ncol = np.bincount(node_supernode, weights=tree.size, minlength=n_sn).astype(np.int64)
    c0 = np.r_[0, np.cumsum(ncol)[:-1]].astype(np.int64)
    parent = np.where(tree.parent[kept] >= 0, node_supernode[np.maximum(tree.parent[kept], 0)], -1)
    height = np.zeros(n_sn, np.int64)
    depth = tree.depth[kept]
    for d in range(int(depth.max(initial=0)), 0, -1):
        at = np.flatnonzero(depth == d)
        np.maximum.at(height, parent[at], height[at] + 1)
    n_levels = int(height.max(initial=-1)) + 1

    def put(values: npt.NDArray[Any]) -> wp.array[wp.int32]:
        return _launch.array(values.astype(np.int32), dtype=wp.int32, device=device)

    # The elimination order: a stable sort of the vertices by their node's first position.
    keys = _launch.empty(2 * n, dtype=wp.int32, device=device)
    vertices = _launch.empty(2 * n, dtype=wp.int32, device=device)
    pos = _launch.empty(n, dtype=wp.int32, device=device)
    owner = _launch.empty(n, dtype=wp.int32, device=device)
    column_end = put(c0 + ncol)
    height_d = put(height)
    if n:
        _launch.launch(
            kernel_cholesky.node_order_keys,
            dim=n,
            inputs=[node, put(node_start)],
            outputs=[keys, vertices],
            device=device,
        )
        _launch.radix_sort_pairs(keys, vertices, n, end_bit=_bits(n))
        _launch.launch(
            kernel_cholesky.scatter_positions,
            dim=n,
            inputs=[vertices, node, put(node_supernode)],
            outputs=[pos, owner],
            device=device,
        )
    perm = odt.as_dense(vertices[:n]) if n else vertices
    # Row structure, one height at a time: a supernode's rows are its columns' later neighbours
    # and its children's rows beyond its own columns. Every candidate is packed as ``supernode *
    # n + row``, sorted and deduplicated per height; a height's rows past its parents' columns
    # wait in ``pending`` for the parent's height.
    key_bits = _bits(n_sn * n)
    shift = key_bits
    if shift + _bits(n_levels) > 64:
        raise ValueError("sparse_cholesky: the factorization exceeds 64-bit row keys")
    row_start = _launch.zeros(max(n_sn, 1), dtype=wp.int32, device=device)
    nrow_d = _launch.zeros(max(n_sn, 1), dtype=wp.int32, device=device)
    parent_d = put(parent)
    levels: list[tuple[wp.array[wp.uint64], int, int]] = []
    level_base = 0
    if n_levels:
        own_counts = _launch.empty(n, dtype=wp.int32, device=device)
        own_ends = _launch.empty(n, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.count_own_entries,
            dim=n,
            inputs=[offsets, columns, pos, owner, column_end],
            outputs=[own_counts],
            device=device,
        )
        _launch.array_scan(own_counts, own_ends, inclusive=True)
        # Read back: the own-entry count sizes their sort.
        n_own = int(read_scalar(own_ends, -1))
        own_keys = _launch.empty(2 * max(n_own, 1), dtype=wp.uint64, device=device)
        own_values = _launch.empty(2 * max(n_own, 1), dtype=wp.int32, device=device)
        starts = _launch.full(n_levels + 1, n_own, dtype=wp.int32, device=device)
        if n_own:
            _launch.launch(
                kernel_cholesky.write_own_entries,
                dim=n,
                inputs=[offsets, columns, pos, owner, column_end, height_d, wp.int32(n),
                        wp.int32(shift), own_ends],
                outputs=[own_keys, own_values],
                device=device,
            )  # fmt: skip
            _launch.radix_sort_pairs(own_keys, own_values, n_own, end_bit=shift + _bits(n_levels))
            _launch.launch(
                kernel_cholesky.height_starts,
                dim=n_own,
                inputs=[own_keys, wp.int32(shift)],
                outputs=[starts],
                device=device,
            )
        # Read back: each height's own entries are one slice of the sorted keys.
        own_start = starts.numpy().astype(np.int64)
        key_mask = wp.uint64((1 << shift) - 1)
        pending = _launch.empty(1, dtype=wp.uint64, device=device)
        n_pending = 0
        for h in range(n_levels):
            n_own_h = int(own_start[h + 1] - own_start[h])
            capacity = n_own_h + n_pending
            if capacity == 0:
                continue
            candidates = _launch.empty(2 * capacity, dtype=wp.uint64, device=device)
            values = _launch.empty(2 * capacity, dtype=wp.int32, device=device)
            kept_pending = _launch.empty(max(n_pending, 1), dtype=wp.uint64, device=device)
            cursors = _launch.zeros(3, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.gather_candidates,
                dim=capacity,
                inputs=[own_keys, wp.int32(own_start[h]), wp.int32(n_own_h), key_mask, pending,
                        height_d, wp.int32(n), wp.int32(h), cursors],
                outputs=[candidates, values, kept_pending],
                device=device,
            )  # fmt: skip
            # Read back: how many pending rows this height takes, and how many wait.
            taken, waiting = (int(x) for x in read_values(cursors, 0, 2))
            count = n_own_h + taken
            if count == 0:
                pending, n_pending = kept_pending, waiting
                continue
            _launch.radix_sort_pairs(candidates, values, count, end_bit=key_bits)
            flags = _launch.empty(count, dtype=wp.int32, device=device)
            ends = _launch.empty(count, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.mark_key_runs,
                dim=count,
                inputs=[candidates],
                outputs=[flags],
                device=device,
            )
            _launch.array_scan(flags, ends, inclusive=True)
            # Read back: the height's distinct rows size its row buffer.
            n_unique = int(read_scalar(ends, count - 1))
            unique = _launch.empty(n_unique, dtype=wp.uint64, device=device)
            _launch.launch(
                kernel_cholesky.compact_key_runs,
                dim=count,
                inputs=[candidates, flags, ends],
                outputs=[unique],
                device=device,
            )
            pending = _launch.empty(waiting + n_unique, dtype=wp.uint64, device=device)
            if waiting:
                _launch.copy(pending, kept_pending, count=waiting)
            _launch.launch(
                kernel_cholesky.emit_rows,
                dim=n_unique,
                inputs=[unique, wp.int32(n), wp.int32(level_base), parent_d, column_end,
                        wp.int32(waiting), cursors],
                outputs=[row_start, nrow_d, pending],
                device=device,
            )  # fmt: skip
            # Read back: the rows passed up size the next height's candidates.
            n_pending = waiting + int(read_scalar(cursors, 2))
            levels.append((unique, level_base, n_unique))
            level_base += n_unique
    n_rows = level_base
    rows = _launch.zeros(n_rows + 1, dtype=wp.int32, device=device)
    row_node = _launch.empty(max(n_rows, 1), dtype=wp.int32, device=device)
    for unique, base, count in levels:
        _launch.launch(
            kernel_cholesky.split_rows,
            dim=count,
            inputs=[unique, wp.int32(n), wp.int32(base)],
            outputs=[rows, row_node],
            device=device,
        )
    # Read back: the row counts size every front.
    nrow = nrow_d.numpy()[:n_sn].astype(np.int64)
    return _Structure(
        perm, pos, owner, ncol, nrow, c0, parent, height, rows, row_node, row_start, n_rows
    )


@dataclass
class _Structure:
    """The supernode tree in elimination order, with each supernode's row structure."""

    perm: wp.array[wp.int32]
    pos: wp.array[wp.int32]
    owner: wp.array[wp.int32]
    ncol: npt.NDArray[np.int64]
    nrow: npt.NDArray[np.int64]
    c0: npt.NDArray[np.int64]
    parent: npt.NDArray[np.int64]
    height: npt.NDArray[np.int64]
    # Every supernode's rows, ascending, height by height (``row_offsets`` is each supernode's
    # first, ``row_node`` each entry's supernode); one trailing zero.
    rows: wp.array[wp.int32]
    row_node: wp.array[wp.int32]
    row_offsets: wp.array[wp.int32]
    n_rows: int


@dataclass
class _Tree:
    """The nested-dissection tree: per node its first position, rows, parent, depth and root."""

    start: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    parent: npt.NDArray[np.int64]
    depth: npt.NDArray[np.int64]
    root: npt.NDArray[np.int64]


def _nested_dissection(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], n: int, coordinates: wp.array[wp.vec3]
) -> tuple[wp.array[wp.int32], _Tree]:
    """
    Every subset of one depth bisected at once: each vertex's node, and the tree.

    A subset is split at the median of its projection onto its principal axis, and the separator
    is the smaller of the two endpoint sets of the pattern edges crossing the split. A subset of at
    most ``CHOLESKY_LEAF_ROWS`` rows is a leaf; a split with no crossing edge (two components)
    makes no node, and its halves hang from the subset's parent. Each subset owns an interval of
    the elimination order -- its lower half's, its upper half's, then its separator's rows -- so a
    node's first position orders the tree in postorder.
    """
    device = coordinates.device
    node = _launch.empty(n, dtype=wp.int32, device=device)
    nodes = _TreeBuilder()
    if n <= CHOLESKY_LEAF_ROWS:
        if n:
            one = np.ones(1, np.int64)
            nodes.add(0 * one, n * one, -one, 0 * one, -one)
            _launch.zero_(node)
        return node, nodes.tree()
    # The live vertices, grouped by segment (a subset of the current depth), and per segment its
    # count, first position in the elimination order, parent node, that node's depth and root.
    order = arange(n, device=device)
    order_segment = _launch.zeros(n, dtype=wp.int32, device=device)
    count = np.array([n], np.int64)
    first = np.zeros(1, np.int64)
    parent = np.full(1, -1, np.int64)
    parent_depth = np.full(1, -1, np.int64)
    root = np.full(1, -1, np.int64)
    tag = _launch.full(n, -1, dtype=wp.int32, device=device)
    lower_mark = _launch.zeros(n, dtype=wp.int32, device=device)
    upper_mark = _launch.zeros(n, dtype=wp.int32, device=device)
    chunk = int(kernel_cholesky.DISSECTION_CHUNK)
    base = 0
    stamp = 0
    while count.size:
        stamp += 1
        n_segments = count.size
        live = int(count.sum())
        segment_start = np.r_[0, np.cumsum(count)]
        tasks = -(-count // chunk)
        task_offsets = np.r_[0, np.cumsum(tasks)]
        task_segment = np.repeat(np.arange(n_segments), tasks)
        task_begin = segment_start[task_segment] + chunk * (
            np.arange(task_offsets[-1]) - task_offsets[task_segment]
        )
        task_end = np.minimum(task_begin + chunk, segment_start[task_segment + 1])
        table = _launch.array(
            np.concatenate(
                [segment_start, task_offsets, task_segment, task_begin, task_end]
            ).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        n_tasks = int(task_offsets[-1])
        segment_start_d = table[: n_segments + 1]
        task_offsets_d = table[n_segments + 1 : 2 * n_segments + 2]
        task_table = [
            table[2 * n_segments + 2 + k * n_tasks : 2 * n_segments + 2 + (k + 1) * n_tasks]
            for k in range(3)
        ]
        partials = _launch.empty(n_tasks, dtype=kernel_cholesky.moment_vector, device=device)
        _launch.launch(
            kernel_cholesky.dissection_partials,
            dim=n_tasks,
            inputs=[order, coordinates, segment_start_d, *task_table],
            outputs=[partials],
            device=device,
        )
        centre = _launch.empty(n_segments, dtype=wp.vec3d, device=device)
        axis = _launch.empty(n_segments, dtype=wp.vec3d, device=device)
        _launch.launch(
            kernel_cholesky.dissection_axes,
            dim=n_segments,
            inputs=[order, coordinates, segment_start_d, task_offsets_d, partials],
            outputs=[centre, axis],
            device=device,
        )
        keys = _launch.empty(2 * live, dtype=wp.uint64, device=device)
        vertices = _launch.empty(2 * live, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.dissection_keys,
            dim=live,
            inputs=[order, order_segment, coordinates, centre, axis],
            outputs=[keys, vertices],
            device=device,
        )
        _launch.radix_sort_pairs(keys, vertices, live, end_bit=32 + _bits(n_segments - 1))
        _launch.launch(
            kernel_cholesky.dissection_sides,
            dim=live,
            inputs=[keys, vertices, segment_start_d, wp.int32(base)],
            outputs=[tag],
            device=device,
        )
        counts = _launch.zeros(2 * n_segments, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.dissection_crossings,
            dim=live,
            inputs=[keys, vertices, offsets, columns, tag, wp.int32(stamp)],
            outputs=[lower_mark, upper_mark, counts],
            device=device,
        )
        # Read back: the crossing endpoints per segment and side decide every separator, and the
        # sizes they leave allocate the next depth.
        crossing = counts.numpy().reshape(-1, 2).astype(np.int64)
        take_lower = crossing[:, 0] <= crossing[:, 1]
        separator = np.where(take_lower, crossing[:, 0], crossing[:, 1])
        half = count // 2
        lower = half - np.where(take_lower, separator, 0)
        upper = count - half - np.where(take_lower, 0, separator)
        has = separator > 0
        separator_node = nodes.add(
            (first + count - separator)[has],
            separator[has],
            parent[has],
            parent_depth[has] + 1,
            root[has],
        )
        me = parent.copy()
        me[has] = separator_node
        me_depth = np.where(has, parent_depth + 1, parent_depth)
        me_root = root.copy()
        me_root[has] = np.where(root[has] >= 0, root[has], separator_node)
        child_count = np.stack([lower, upper], 1).ravel()
        child_first = np.stack([first, first + lower], 1).ravel()
        child_parent = np.repeat(me, 2)
        child_depth = np.repeat(me_depth, 2)
        child_root = np.repeat(me_root, 2)
        leaf = (child_count > 0) & (child_count <= CHOLESKY_LEAF_ROWS)
        child_leaf = np.full(child_count.size, -1, np.int64)
        child_leaf[leaf] = nodes.add(
            child_first[leaf],
            child_count[leaf],
            child_parent[leaf],
            child_depth[leaf] + 1,
            child_root[leaf],
        )
        going_on = child_count > CHOLESKY_LEAF_ROWS
        child_next = np.where(going_on, np.cumsum(going_on) - 1, -1)
        separator_full = np.full(n_segments, -1, np.int64)
        separator_full[has] = separator_node
        relabel = _launch.array(
            np.concatenate([
                np.where(has, np.where(take_lower, 1, 2), 0), separator_full, child_next,
                child_leaf,
            ]).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )  # fmt: skip
        keep = _launch.empty(live, dtype=wp.int32, device=device)
        next_segment = _launch.empty(live, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_cholesky.dissection_relabel,
            dim=live,
            inputs=[keys, vertices, segment_start_d, relabel[:n_segments],
                    relabel[n_segments : 2 * n_segments],
                    relabel[2 * n_segments : 4 * n_segments], relabel[4 * n_segments :],
                    lower_mark, upper_mark, wp.int32(stamp)],
            outputs=[node, keep, next_segment],
            device=device,
        )  # fmt: skip
        count = child_count[going_on]
        survivors = int(count.sum())
        if survivors:
            kept_through = _launch.empty(live, dtype=wp.int32, device=device)
            _launch.array_scan(keep, kept_through, inclusive=True)
            order = _launch.empty(survivors, dtype=wp.int32, device=device)
            order_segment = _launch.empty(survivors, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_cholesky.dissection_compact,
                dim=live,
                inputs=[vertices, keep, kept_through, next_segment],
                outputs=[order, order_segment],
                device=device,
            )
        first = child_first[going_on]
        parent = child_parent[going_on]
        parent_depth = child_depth[going_on]
        root = child_root[going_on]
        base += 2 * n_segments
    return node, nodes.tree()


class _TreeBuilder:
    """The dissection tree's nodes, appended a depth at a time."""

    def __init__(self) -> None:
        self.parts: list[tuple[npt.NDArray[np.int64], ...]] = []
        self.count = 0

    def add(
        self,
        start: npt.NDArray[np.int64],
        size: npt.NDArray[np.int64],
        parent: npt.NDArray[np.int64],
        depth: npt.NDArray[np.int64],
        root: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Append nodes (a ``root`` of ``-1`` roots its own tree) and return their ids."""
        ids = self.count + np.arange(start.size, dtype=np.int64)
        root = np.where(root >= 0, root, ids)
        self.parts.append((start, size, parent, depth, root))
        self.count += start.size
        return ids

    def tree(self) -> _Tree:
        if not self.parts:
            return _Tree(*(np.zeros(0, np.int64) for _ in range(5)))
        return _Tree(*(np.concatenate(p).astype(np.int64) for p in zip(*self.parts, strict=True)))


def _bits(value: int) -> int:
    """Bits a radix sort needs for keys up to ``value``."""
    return max(int(value).bit_length(), 1)


def _landmark_coordinates(
    matrix: odt.BsrMatrix[wp.float64], labels: wp.array[wp.int32]
) -> wp.array[wp.vec3]:
    """Return hop distances from three mutually far vertices per component: a pattern embedding."""
    n = int(matrix.nrow)
    device = matrix.device
    coordinates = _launch.zeros(n, dtype=wp.vec3, device=device)
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
    for axis, (a, b) in enumerate(((None, None), (0, 0), (0, 0), (0, 1))):
        if a is None or b is None:
            seed_inputs = [key, size, first, wp.int32(CHOLESKY_LEAF_ROWS), wp.int32(1)]
        else:
            _launch.zero_(key)
            _launch.launch(
                kernel_cholesky.farthest_rows,
                dim=n,
                inputs=[labels, axes[:, a], axes[:, b]],
                outputs=[key],
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
            kernel_cholesky.store_distance,
            dim=n,
            inputs=[distance],
            outputs=[axes[:, max(axis - 1, 0)]],
            device=device,
        )
    return coordinates


def _host_landmark_coordinates(
    matrix: odt.BsrMatrix[wp.float64], labels: wp.array[wp.int32]
) -> npt.NDArray[np.float32]:
    """``_landmark_coordinates`` by frontier searches on the host."""
    n = int(matrix.nrow)
    offsets = matrix.offsets.numpy()[: n + 1].astype(np.int64)
    columns = matrix.columns.numpy()[: int(offsets[-1])].astype(np.int64)
    labels_np = labels.numpy()
    coords = np.zeros((n, 3), np.float32)
    first = np.unique(labels_np, return_index=True)[1]
    sizes = np.bincount(labels_np, minlength=n)
    for seed in first[sizes[labels_np[first]] > CHOLESKY_LEAF_ROWS]:
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
