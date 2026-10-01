"""
Warp array type aliases, runtime rank checks, and typed vector builtins for Python wrappers.

Use these aliases in ``triwarp`` Python APIs instead of ``wp.array2d[dtype]``, which
static checkers treat as Warp annotation objects (no ``.shape`` / indexing).

Kernels should continue to use ``wp.array2d[dtype]`` in ``@wp.kernel`` signatures.

[`normalize`][triwarp.typing.normalize], [`cross`][triwarp.typing.cross] and
[`transform_point`][triwarp.typing.transform_point] are the Warp builtins of those names,
re-exported unchanged so that a vector held in a variable resolves against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypeVar, cast, overload

import warp as wp
import warp.sparse as wps

from triwarp import _launch

T = TypeVar("T")
DType = TypeVar("DType")
NDim = TypeVar("NDim", bound=int)
# Any Warp array type, kept whole: a union of array types in is the same union out.
ArrayT = TypeVar("ArrayT", bound="wp.array[Any, Any]")
# The block type of a sparse matrix: the element dtype of its ``values``.
Block = TypeVar("Block")

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import TypeIs

    # Anything ``wp.launch`` accepts as its kernel. ``wp.kernel`` carries no return annotation, so
    # a checker infers a ``@wp.kernel``-decorated name as the Python function it wraps, while
    # ``wp.overload``, ``wp.map(..., return_kernel=True)`` and ``wp.kernel(f, name=...)`` in a
    # factory are real ``wp.Kernel`` objects. Every kernel is launched with ``None`` as its value.
    Kernel: TypeAlias = wp.Kernel | Callable[..., None]

    # A ``warp.sparse`` matrix whose blocks are ``Block`` (``wp.float32``, ``wp.float64``,
    # ``wp.mat22d``), with the storage fields declared. Warp declares them only on the class
    # ``bsr_matrix_t`` generates at runtime, of which every real matrix is an instance, so reading
    # ``matrix.values`` off the base class is an unknown attribute to a checker.
    class BsrMatrix(wps.BsrMatrix[Block]):
        nrow: int
        ncol: int
        nnz: int
        offsets: wp.array[wp.int32]
        columns: wp.array[wp.int32]
        row_counts: wp.array[wp.int32] | None
        values: wp.array[Block]

    # A scalar CSR matrix at either precision, and a sparse operator of any block type the solvers
    # take (adding the ``wp.mat22d`` tangent-field operators). Unions rather than a ``TypeVar``:
    # a function accepting any of them is called with the union itself, which no ``TypeVar``
    # (constrained or not) accepts. Narrow one with ``has_blocks``; a private helper that returns
    # the block type it was given is an overload set, across whose arms a union argument expands.
    CsrMatrix: TypeAlias = BsrMatrix[wp.float32] | BsrMatrix[wp.float64]
    SparseMatrix: TypeAlias = CsrMatrix | BsrMatrix[wp.mat22d]

    Array1dInt32: TypeAlias = wp.array[wp.int32, Literal[1]]
    Array1dFloat32: TypeAlias = wp.array[wp.float32, Literal[1]]
    Array1dFloat64: TypeAlias = wp.array[wp.float64, Literal[1]]
    Array2dInt32: TypeAlias = wp.array[wp.int32, Literal[2]]
    Array2dFloat32: TypeAlias = wp.array[wp.float32, Literal[2]]
    Array2dFloat64: TypeAlias = wp.array[wp.float64, Literal[2]]
    Array2dVec3: TypeAlias = wp.array[wp.vec3, Literal[2]]
    Array3dFloat32: TypeAlias = wp.array[wp.float32, Literal[3]]
    Array3dBool: TypeAlias = wp.array[wp.bool, Literal[3]]

    # wp.Int / wp.Float / wp.Scalar are TypeVars; subscripting wp.array[...] with them
    # yields a generic alias that requires type arguments under basedpyright.
    Array1dInt: TypeAlias = Array1dInt32
    Array2dInt: TypeAlias = Array2dInt32
    Array1dFloat: TypeAlias = Array1dFloat32 | Array1dFloat64
    Array2dFloat: TypeAlias = Array2dFloat32 | Array2dFloat64
    Array1dScalar: TypeAlias = Array1dInt32 | Array1dFloat
    Array2dScalar: TypeAlias = Array2dInt32 | Array2dFloat

    IntArray: TypeAlias = Array1dInt | Array2dInt
    FloatArray: TypeAlias = Array1dFloat | Array2dFloat
    ScalarArray: TypeAlias = Array1dScalar | Array2dScalar

    # Any rank, for the handful of operations that are genuinely rank-agnostic because they
    # flatten and reshape back (``array.isin``, ``array.gather``). Prefer a concrete rank alias
    # everywhere else -- this one deliberately gives up the rank check.
    ArrayNdInt32: TypeAlias = wp.array[wp.int32, Any]
    ArrayNd: TypeAlias = wp.array[Any, Any]

    # The parameter-position family. ``NDim`` is invariant, so ``wp.array[wp.float32]`` -- this
    # package's usual rank-1 spelling, and what ``wp.empty`` and most public signatures produce --
    # is ``array[float32, int]`` and is not assignable to ``Array1dFloat32``
    # (``array[float32, Literal[1]]``) in either direction. A callee that pins the rank therefore
    # rejects its own callers. The rule these exist to express: **accept wide, return narrow** --
    # a parameter takes the ``ArrayNd*`` form, a return keeps the ``Literal``-ranked one, so a
    # caller may hand over either spelling while the value it gets back still carries its rank.
    # The dtype is still discriminated, which is what keeps an overload set resolvable.
    ArrayNdInt64: TypeAlias = wp.array[wp.int64, Any]
    ArrayNdUInt32: TypeAlias = wp.array[wp.uint32, Any]
    ArrayNdUInt64: TypeAlias = wp.array[wp.uint64, Any]
    ArrayNdFloat32: TypeAlias = wp.array[wp.float32, Any]
    ArrayNdFloat64: TypeAlias = wp.array[wp.float64, Any]
    ArrayNdFloat: TypeAlias = ArrayNdFloat32 | ArrayNdFloat64
    ArrayNdInt: TypeAlias = ArrayNdInt32 | ArrayNdInt64 | ArrayNdUInt32 | ArrayNdUInt64
    ArrayNdScalar: TypeAlias = ArrayNdInt | ArrayNdFloat

    # The six scalar dtypes ``warp.utils.radix_sort_pairs`` accepts as keys, which is what
    # ``sortable_dtype`` returns. It cannot be spelled ``type[wp.Scalar]``: ``wp.Scalar`` is a
    # ``TypeVar``, so using it in both the parameter and the return position would claim the
    # function is dtype-preserving, and widening a narrow dtype is the whole point of it.
    SortableDType: TypeAlias = (
        type[wp.int32]
        | type[wp.int64]
        | type[wp.uint32]
        | type[wp.uint64]
        | type[wp.float32]
        | type[wp.float64]
    )
else:
    BsrMatrix = wps.BsrMatrix
    CsrMatrix = wps.BsrMatrix
    SparseMatrix = wps.BsrMatrix
    Kernel = wp.Kernel
    Array1dInt32 = wp.array
    Array1dFloat32 = wp.array
    Array1dFloat64 = wp.array
    Array2dInt32 = wp.array
    Array2dFloat32 = wp.array
    Array2dFloat64 = wp.array
    Array2dVec3 = wp.array
    Array3dFloat32 = wp.array
    Array3dBool = wp.array
    Array1dInt = wp.array
    Array2dInt = wp.array
    Array1dFloat = wp.array
    Array2dFloat = wp.array
    Array1dScalar = wp.array
    Array2dScalar = wp.array
    IntArray = wp.array
    FloatArray = wp.array
    ScalarArray = wp.array
    ArrayNdInt32 = wp.array
    ArrayNd = wp.array
    ArrayNdInt64 = wp.array
    ArrayNdUInt32 = wp.array
    ArrayNdUInt64 = wp.array
    ArrayNdFloat32 = wp.array
    ArrayNdFloat64 = wp.array
    ArrayNdFloat = wp.array
    ArrayNdInt = wp.array
    ArrayNdScalar = wp.array
    SortableDType = type


# Typed views of Warp functions whose stubs cannot describe a triwarp call. Each is the Warp
# function itself, bound once through a ``cast`` to a ``Protocol`` spelling its signature over the
# concrete types, so a call costs exactly what it always did. They are deliberately absent from
# ``__all__``: mkdocstrings would render them as bare module attributes. They stay importable as
# ``twt.<name>``. Drop each once the upstream stub describes it.
#
# The vector builtins' stubs take the ``Vector`` / ``Matrix`` hint shells, which no concrete Warp
# type derives from (``wp.vec3`` is ``vec3f``, based on ``ctypes.Array``), so a vector held in a
# variable never satisfies them -- only a value coming straight out of another builtin does.
_V = TypeVar("_V", wp.vec2, wp.vec3, wp.vec4)


class _Normalize(Protocol):
    def __call__(self, v: _V, /) -> _V:
        """Vector of unit length along ``v``, at ``v``'s own precision."""
        ...


class _Cross(Protocol):
    def __call__(self, a: _V, b: _V, /) -> _V:
        """Cross product of two vectors of the same type."""
        ...


class _Dot(Protocol):
    def __call__(self, a: _V, b: _V, /) -> float:
        """Dot product of two vectors of the same type."""
        ...


class _TransformPoint(Protocol):
    def __call__(self, matrix: wp.mat44, point: wp.vec3, /) -> wp.vec3:
        """``point`` carried through the affine transform ``matrix``, translation included."""
        ...


normalize = cast("_Normalize", wp.normalize)
cross = cast("_Cross", wp.cross)
dot = cast("_Dot", wp.dot)
transform_point = cast("_TransformPoint", wp.transform_point)

# ``warp.sparse`` parameterizes its matrices by a phantom ``BlockType[Rows, Cols, Scalar]`` that no
# function returns, so no matrix a caller holds satisfies a ``bsr_*`` parameter, and every result
# is ``BsrMatrix[Unknown]``. Below, ``BsrMatrix[Block]`` is parameterized by the element dtype of
# its ``values`` and the functions triwarp calls take and return it. Only the parameters triwarp
# passes are spelled; the rest keep Warp's defaults at runtime.
_S = TypeVar("_S")


class _BsrMatrixT(Protocol):
    def __call__(self, dtype: type[Block], /) -> type[BsrMatrix[Block]]:
        """Return the concrete matrix class whose blocks are ``dtype``."""
        ...


class _BsrDiag(Protocol):
    def __call__(
        self, diag: wp.array[Block, Any], *, device: wp.DeviceLike = None
    ) -> BsrMatrix[Block]:
        """Square block-diagonal matrix with ``diag`` on its diagonal."""
        ...


class _BsrIdentity(Protocol):
    def __call__(
        self, rows_of_blocks: int, block_type: type[Block], device: wp.DeviceLike = None
    ) -> BsrMatrix[Block]:
        """Identity matrix of ``rows_of_blocks`` blocks of ``block_type``."""
        ...


class _BsrAxpy(Protocol):
    def __call__(
        self,
        x: BsrMatrix[Block],
        y: BsrMatrix[Block] | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> BsrMatrix[Block]:
        """``alpha * x + beta * y``, written into ``y`` when it is given."""
        ...


class _BsrMm(Protocol):
    def __call__(self, x: BsrMatrix[Block], y: BsrMatrix[Block], /) -> BsrMatrix[Block]:
        """Return the matrix product ``x @ y``."""
        ...


class _BsrMv(Protocol):
    def __call__(
        self,
        matrix: SparseMatrix,
        x: wp.array[_S, Any],
        y: wp.array[_S, Any],
        /,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        tile_size: int = 0,
    ) -> wp.array[_S, Any]:
        """``alpha * matrix @ x + beta * y``, written into ``y``."""
        ...


class _BsrUnary(Protocol):
    def __call__(self, matrix: BsrMatrix[Block], /) -> BsrMatrix[Block]:
        """Return a new matrix derived from ``matrix``, with its block type."""
        ...


class _BsrGetDiag(Protocol):
    def __call__(self, matrix: BsrMatrix[Block], /) -> wp.array[Block]:
        """Return the diagonal blocks of ``matrix``."""
        ...


class _BsrCompress(Protocol):
    def __call__(
        self, matrix: BsrMatrix[Block], /, *, prune_numerical_zeros: bool = True
    ) -> BsrMatrix[Block]:
        """``matrix`` with its active blocks compacted, optionally dropping explicit zeros."""
        ...


bsr_matrix_t = cast("_BsrMatrixT", wps.bsr_matrix_t)
bsr_diag = cast("_BsrDiag", wps.bsr_diag)
bsr_identity = cast("_BsrIdentity", wps.bsr_identity)
bsr_axpy = cast("_BsrAxpy", wps.bsr_axpy)
bsr_mm = cast("_BsrMm", wps.bsr_mm)
bsr_mv = cast("_BsrMv", wps.bsr_mv)
bsr_copy = cast("_BsrUnary", wps.bsr_copy)
bsr_transposed = cast("_BsrUnary", wps.bsr_transposed)
bsr_get_diag = cast("_BsrGetDiag", wps.bsr_get_diag)
bsr_compress = cast("_BsrCompress", wps.bsr_compress)

__all__ = [
    "Array1dFloat",
    "Array1dFloat32",
    "Array1dFloat64",
    "Array1dInt",
    "Array1dInt32",
    "Array1dScalar",
    "Array2dFloat",
    "Array2dFloat32",
    "Array2dFloat64",
    "Array2dInt",
    "Array2dInt32",
    "Array2dScalar",
    "Array2dVec3",
    "Array3dBool",
    "Array3dFloat32",
    "ArrayNd",
    "ArrayNdFloat",
    "ArrayNdFloat32",
    "ArrayNdFloat64",
    "ArrayNdInt",
    "ArrayNdInt32",
    "ArrayNdInt64",
    "ArrayNdScalar",
    "ArrayNdUInt32",
    "ArrayNdUInt64",
    "CsrMatrix",
    "FloatArray",
    "IntArray",
    "Kernel",
    "ScalarArray",
    "SortableDType",
    "SparseMatrix",
    "as_array2d",
    "as_array3d",
    "as_dense",
    "dtype_max",
    "dtype_min",
    "dtype_zero",
    "empty_1d",
    "empty_2d",
    "empty_3d",
    "ensure_edge_pairs",
    "ensure_ndim",
    "has_blocks",
    "sortable_dtype",
    "vec3_floats",
]


def ensure_ndim(arr: ArrayT, ndim: int, *, dtype: type | None = None) -> ArrayT:
    """
    Validate the rank (and optionally the dtype) of a Warp array.

    The runtime counterpart to the aliases in this module: `isinstance(x, wp.array2d)` is always
    ``False``, so rank is checked here instead.

    Parameters
    ----------
    arr
        Array to validate.
    ndim
        Required rank.
    dtype
        When given, the required element dtype.

    Returns
    -------
    wp.array
        ``arr`` unchanged, so this can wrap an argument in place.

    Raises
    ------
    TypeError
        If the rank or dtype does not match.
    """
    if int(arr.ndim) != ndim:
        raise TypeError(f"expected {ndim}D array, got ndim={arr.ndim}")
    if dtype is not None and arr.dtype != dtype:
        raise TypeError(f"expected dtype {dtype}, got {arr.dtype}")
    return arr


def ensure_edge_pairs(arr: wp.array[T, Any], name: str) -> None:
    """
    Validate that ``arr`` is a rank-2 ``wp.int32`` array with exactly two columns.

    The shared check every ``(k, 2)`` edge-list argument in the package runs before use, factored
    because six call sites (``triangles.corner_normals``, ``seams.cut_along_edges``,
    ``graph.connected_component_parity_from_edges`` and ``graph._validate_edge_list``,
    ``proximity.closest_point_on_edges``, ``selection.contour_side_mask``) wrote it out by hand,
    with the same two exceptions and only the parameter name differing between them.

    Parameters
    ----------
    arr
        Array to validate.
    name
        Parameter name to use in the raised message.

    Raises
    ------
    TypeError
        If ``arr`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``arr`` does not have exactly two columns.
    """
    ensure_ndim(arr, 2, dtype=wp.int32)
    if int(arr.shape[1]) != 2:
        raise ValueError(f"{name} must have shape (k, 2), got {arr.shape}")


def as_array2d(arr: wp.array[T, Any], dtype: type[DType]) -> wp.array[DType, Literal[2]]:
    """
    Validate and narrow a Warp array to the rank-2 alias for ``dtype``.

    One function rather than a per-dtype family: the ``dtype`` argument carries the element type
    into the return, so a call site keeps the narrow type it had -- ``as_array2d(x, wp.int32)`` is
    an [`Array2dInt32`][triwarp.typing.Array2dInt32], not a union.

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 with element type ``dtype``.
    dtype
        Expected element type.

    Returns
    -------
    wp.array
        ``arr`` unchanged, narrowed to ``wp.array[dtype, Literal[2]]``.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 with element type ``dtype``.
    """
    return _as_ranked(arr, 2, dtype)


def as_array3d(arr: wp.array[T, Any], dtype: type[DType]) -> wp.array[DType, Literal[3]]:
    """
    Validate and narrow a Warp array to the rank-3 alias for ``dtype``.

    The rank-3 counterpart of [`as_array2d`][triwarp.typing.as_array2d].

    Parameters
    ----------
    arr
        Warp array expected to be rank-3 with scalar type ``dtype``.
    dtype
        Expected scalar type.

    Returns
    -------
    wp.array
        ``arr`` unchanged, narrowed to ``wp.array[dtype, Literal[3]]``.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-3 with scalar type ``dtype``.
    """
    return _as_ranked(arr, 3, dtype)


def _as_ranked(arr: wp.array[T, Any], ndim: int, dtype: type[DType]) -> wp.array[DType, Any]:
    """Shared body of the ``as_array*`` pair: check the rank and the dtype, then narrow to both."""
    ensure_ndim(arr, ndim, dtype=dtype)
    return cast("wp.array[DType, Any]", arr)


def as_dense(view: wp.array[DType, NDim] | wp.indexedarray[DType, NDim]) -> wp.array[DType, NDim]:
    """
    Narrow a Python-scope index expression to the dense array a slice of one always is.

    ``wp.array.__getitem__`` carries no annotations, so basedpyright infers its return from the two
    branches of the body and every subscript comes back as ``indexedarray | array`` -- including a
    plain slice, which can only ever produce the dense arm (§3.4: an ``indexedarray`` is what an
    *integer-array* key yields, and a slice is not one). This narrows the union back, and unlike a
    bare ``cast`` it checks: ``wp.indexedarray`` is not a subclass of ``wp.array``, so the
    ``isinstance`` genuinely discriminates rather than restating the assumption.

    Use it on a slice. A gather (``src[indices]``) is an ``indexedarray`` by design and is
    materialized with ``wp.copy`` instead, not narrowed here.

    Parameters
    ----------
    view
        Result of a Python-scope subscript of a ``wp.array``.

    Returns
    -------
    wp.array
        ``view`` unchanged, typed as the dense array it is.

    Raises
    ------
    TypeError
        If ``view`` is a ``wp.indexedarray`` -- that is, if the subscript was a gather rather than
        a slice.

    See Also
    --------
    [`as_array2d`][triwarp.typing.as_array2d]
    [`triwarp.array.gather`][]
    """
    if not isinstance(view, wp.array):
        raise TypeError(f"expected a dense wp.array slice, got {type(view).__name__}")
    return view


def vec3_floats(v: wp.vec3) -> tuple[float, float, float]:
    """
    Components of a Python-scope 3-vector, as Python floats.

    At Python scope indexing a ``wp.vec3`` already returns a native ``float``, but
    Warp's stubs type the index as ``vec_t | bool | float32 | int32``, so arithmetic on two
    components does not type-check. This names that fact once for host-side bounds arithmetic.

    Parameters
    ----------
    v
        Vector to unpack.

    Returns
    -------
    tuple[float, float, float]
        ``(v[0], v[1], v[2])``.
    """
    return cast("tuple[float, float, float]", (v[0], v[1], v[2]))


def has_blocks(matrix: object, dtype: type[Block]) -> TypeIs[BsrMatrix[Block]]:
    """
    Whether ``matrix`` is a sparse matrix whose blocks are ``dtype``.

    The runtime test a solver branches on (``wp.float32`` / ``wp.float64`` CSR, ``wp.mat22d``
    blocks), and it narrows a [`SparseMatrix`][triwarp.typing.SparseMatrix] on both branches:
    to ``BsrMatrix[dtype]`` where it holds and to the remaining block types where it does not.

    Parameters
    ----------
    matrix
        Value to test.
    dtype
        Block type to test for: the element dtype of ``matrix.values``.

    Returns
    -------
    bool
        ``True`` if ``matrix`` is a ``warp.sparse`` matrix and ``matrix.values.dtype`` is
        ``dtype``.
    """
    return isinstance(matrix, BsrMatrix) and matrix.values.dtype == dtype


@overload
def dtype_max(dtype: type[wp.Int]) -> int: ...
@overload
def dtype_max(dtype: type[wp.Float]) -> float: ...
def dtype_max(dtype: type[wp.Scalar]) -> int | float:
    """
    Largest representable value for a Warp scalar type.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``float("inf")`` for floating-point types; the maximum representable
        integer for integer types.
    """
    if not wp.types.type_is_int(dtype):
        return float("inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if not dtype.__name__.lower().startswith("u"):
        bits -= 1
    return (1 << bits) - 1


@overload
def dtype_min(dtype: type[wp.Int]) -> int: ...
@overload
def dtype_min(dtype: type[wp.Float]) -> float: ...
def dtype_min(dtype: type[wp.Scalar]) -> int | float:
    """
    Smallest representable value for a Warp scalar type.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``float("-inf")`` for floating-point types; the minimum representable
        integer for integer types.
    """
    if not wp.types.type_is_int(dtype):
        return float("-inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if dtype.__name__.lower().startswith("u"):
        return 0
    return -(1 << (bits - 1))


def dtype_zero(dtype: type[wp.Scalar]) -> int | float:
    """
    Zero value for a Warp scalar type, typed to match Python's ``int``/``float`` split.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``0`` for integer types, ``0.0`` for floating-point types.
    """
    if wp.types.type_is_int(dtype):
        return 0
    return 0.0


def sortable_dtype(dtype: type[wp.Scalar]) -> SortableDType:
    """
    Same-width dtype that ``warp.utils.radix_sort_pairs`` accepts, preserving ``dtype``'s order.

    The single widening rule for every radix sort in this package. Sorting a sub-32-bit dtype is
    not supported by Warp, and sorting the reinterpreted *bits* of a float or an unsigned value
    gets the order wrong, so callers ask here rather than widening ad hoc.

    The hash table works in one common signed-integer key space (see
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]), which is fine for equality but wrong for
    ordering: negative floats have descending bit patterns, and a ``uint64`` with its top bit set
    reads as a negative ``int64``. Warp sorts ``int32`` / ``int64`` / ``uint32`` / ``uint64`` /
    ``float32`` / ``float64`` keys directly, so the sort is done in this dtype instead of on the
    reinterpreted bits. Every narrower width -- ``int8`` / ``uint8`` / ``int16`` / ``uint16`` /
    ``float16`` -- raises ``Unsupported keys and values data types``, so the widening table below
    has a case for each of them.

    Parameters
    ----------
    dtype
        Any Warp scalar dtype.

    Returns
    -------
    SortableDType
        ``dtype`` itself when Warp can already sort it, otherwise the narrowest same-signedness,
        same-kind dtype it can (``float32`` / ``float64``, ``uint32`` / ``uint64``, ``int32`` /
        ``int64``), chosen by whether ``dtype`` is wider than four bytes.

    See Also
    --------
    [`sort_and_argsort`][triwarp.array.sort_and_argsort]
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
    """
    wide = wp.types.type_size_in_bytes(dtype) > 4
    if wp.types.type_is_float(dtype):
        return wp.float64 if wide else wp.float32
    if dtype.__name__.lower().startswith("u"):
        return wp.uint64 if wide else wp.uint32
    return wp.int64 if wide else wp.int32


def empty_1d(
    n: int, dtype: type[DType], *, device: wp.DeviceLike = None
) -> wp.array[DType, Literal[1]]:
    """
    Allocate an uninitialized rank-1 Warp array of the given element type.

    The rank-1 member of the ``empty_*`` family, and the checked replacement for
    ``cast(twt.Array1dFloat32, wp.empty(n, dtype=wp.float32, device=...))``. ``wp.empty`` is
    annotated as returning ``warp.array`` -- element type unknown and rank unspecified -- and the
    rank is what a caller cannot recover afterwards: ``NDim`` is invariant, so
    ``wp.array[wp.float32]`` and [`Array1dFloat32`][triwarp.typing.Array1dFloat32] are not
    assignable to one another in either direction. A ``cast`` is an unchecked assertion of both;
    this fixes the rank by construction and carries ``dtype`` through to the return.

    Prefer plain ``wp.empty`` where the surrounding annotations are ``wp.array[dtype]``, which is
    this package's usual rank-1 spelling and which ``wp.empty`` already satisfies -- the aliases
    are for the modules that carry the rank in their signatures ([`triwarp.reduce`][],
    [`triwarp.metrics`][], [`triwarp.neighbors`][]).

    Parameters
    ----------
    n
        Length of the allocated array.
    dtype
        Element type. Any Warp scalar, vector or matrix type; a call site holding a runtime
        ``dtype`` gets ``wp.array[Unknown, Literal[1]]``, which still carries the rank.
    device
        Target Warp device.

    Returns
    -------
    wp.array
        ``(n,)`` uninitialized array of element type ``dtype`` on ``device``.

    See Also
    --------
    [`empty_2d`][triwarp.typing.empty_2d]
    [`empty_3d`][triwarp.typing.empty_3d]
    """
    return _empty_ranked((n,), 1, dtype, device)


def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type[DType], *, device: wp.DeviceLike = None
) -> wp.array[DType, Literal[2]]:
    """
    Allocate an uninitialized rank-2 Warp array of the given element type.

    The rank-2 member of the ``empty_*`` family; see [`empty_1d`][triwarp.typing.empty_1d] for why
    the rank has to be fixed at the allocation rather than asserted afterwards.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    dtype
        Element type. ``wp.vec3`` is what an ``(m, 2)`` table of segment endpoints wants --
        ``triwarp.intersection``'s two public segment returns are the callers.
    device
        Target Warp device.

    Returns
    -------
    wp.array
        ``(rows, cols)`` uninitialized array of element type ``dtype`` on ``device``.

    Raises
    ------
    ValueError
        If ``shape`` does not have length 2.

    See Also
    --------
    [`empty_1d`][triwarp.typing.empty_1d]
    [`empty_3d`][triwarp.typing.empty_3d]
    """
    return _empty_ranked(shape, 2, dtype, device)


def empty_3d(
    shape: tuple[int, int, int] | list[int], dtype: type[DType], *, device: wp.DeviceLike = None
) -> wp.array[DType, Literal[3]]:
    """
    Allocate an uninitialized rank-3 Warp array of the given element type.

    The rank-3 member of the ``empty_*`` family; see [`empty_1d`][triwarp.typing.empty_1d] for why
    the rank has to be fixed at the allocation rather than asserted afterwards.

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` shape of the allocated array.
    dtype
        Element type.
    device
        Target Warp device.

    Returns
    -------
    wp.array
        ``(nx, ny, nz)`` uninitialized array of element type ``dtype`` on ``device``.

    Raises
    ------
    ValueError
        If ``shape`` does not have length 3.

    See Also
    --------
    [`empty_1d`][triwarp.typing.empty_1d]
    [`empty_2d`][triwarp.typing.empty_2d]
    """
    return _empty_ranked(shape, 3, dtype, device)


def _empty_ranked(
    shape: tuple[int, ...] | list[int], ndim: int, dtype: type[DType], device: wp.DeviceLike
) -> wp.array[DType, Any]:
    """Shared body of the ``empty_*`` family: check the rank of ``shape``, then allocate at it."""
    dims = tuple(int(extent) for extent in shape)
    if len(dims) != ndim:
        raise ValueError(f"{ndim}D shape must have length {ndim}, got {shape!r}")
    return _launch.empty(dims, dtype=dtype, device=device)
