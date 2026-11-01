"""
Triangular mesh utilities on NVIDIA Warp.

Submodules and ``Trimesh`` resolve **lazily**, through the :pep:`562` module ``__getattr__``
below, so ``import ordito`` costs almost nothing and a caller pays only for what it reaches.

Every ``@wp.kernel`` decoration parses its function and builds an ``Adjoint`` at *import* time, so
eagerly importing every public module would pull in every kernel module along with it and make
``import ordito`` pay for all of them up front. Importing one submodule instead does not avoid
this either: Python imports a parent package before its child, so ``import ordito.edges`` still
runs this file first. Lazy resolution is what keeps both cheap: ``import ordito as od`` (how most
wrappers open) is nearly free, and ``od.laplacian`` costs one ``__getattr__`` on first touch, after
which it is a plain global lookup because the resolved module is cached into this module's
namespace.

Two things this deliberately does not change. Nothing is imported at a *different* time relative to
its own kernel module, so a kernel module's overload registration still runs before the first
launch through it -- a wrapper cannot be reached without importing it. And ``wp.load_module`` /
``wp.force_load`` are still not called here: registration is not compilation, and eager loading at
import would defeat the whole point of this module.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import sys
import threading
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Re-imported for the type checker and for IDE completion only; at runtime these names come
    # from ``__getattr__``. basedpyright reads this branch, so the public surface stays typed.
    from ordito import (
        adjacency,
        array,
        boundary,
        bounds,
        cholesky,
        combine,
        config,
        constants,
        creation,
        curvature,
        edges,
        energies,
        equality,
        geodesic_walk,
        graph,
        grouping,
        halfedge,
        heat,
        holes,
        homology,
        interpolation,
        intersection,
        io,
        laplacian,
        levelset,
        linalg,
        measures,
        mesh,
        metrics,
        neighbors,
        parametrization,
        points,
        polyline,
        proximity,
        ray,
        reconstruction,
        reduce,
        registration,
        remesh,
        repair,
        sample,
        seams,
        selection,
        smoothing,
        tangent_space,
        texture,
        transform,
        triangles,
        typing,
        validation,
        vertices,
        visibility,
        voxels,
    )
    from ordito.config import deterministic, is_deterministic, set_deterministic
    from ordito.mesh import Trimesh

__all__ = [
    "Trimesh",
    "adjacency",
    "array",
    "boundary",
    "bounds",
    "cholesky",
    "combine",
    "config",
    "constants",
    "creation",
    "curvature",
    "deterministic",
    "edges",
    "energies",
    "equality",
    "geodesic_walk",
    "graph",
    "grouping",
    "halfedge",
    "heat",
    "holes",
    "homology",
    "interpolation",
    "intersection",
    "io",
    "is_deterministic",
    "laplacian",
    "levelset",
    "linalg",
    "measures",
    "mesh",
    "metrics",
    "neighbors",
    "parametrization",
    "points",
    "polyline",
    "proximity",
    "ray",
    "reconstruction",
    "reduce",
    "registration",
    "remesh",
    "repair",
    "sample",
    "seams",
    "selection",
    "set_deterministic",
    "smoothing",
    "tangent_space",
    "texture",
    "transform",
    "triangles",
    "typing",
    "validation",
    "vertices",
    "visibility",
    "voxels",
]

# The submodules ``__getattr__`` will resolve. ``Trimesh`` is not one of them -- it is a class in
# ``ordito.mesh`` and is handled separately below, because importing it eagerly here is exactly
# the whole-package pull this file exists to avoid.
# ``ordito.config``'s three setting functions are reachable at the top level too, like
# ``torch.use_deterministic_algorithms``; they resolve to the same objects, on first touch.
_CONFIG_NAMES = frozenset({"deterministic", "is_deterministic", "set_deterministic"})
_LAZY_SUBMODULES = frozenset(__all__) - {"Trimesh"} - _CONFIG_NAMES


def __getattr__(name: str) -> object:
    """
    Resolve a submodule, ``Trimesh``, a setting or ``__version__`` on first access (:pep:`562`).

    The result is cached in this module's namespace, so only the first access reaches here and
    every later ``ordito.<name>`` is an ordinary attribute lookup.

    Parameters
    ----------
    name
        The attribute being looked up.

    Returns
    -------
    object
        The imported submodule, the ``Trimesh`` class, one of ``ordito.config``'s setting
        functions, or the installed version string.

    Raises
    ------
    AttributeError
        If ``name`` is not a public name or ``__version__``. Raising this rather
        than letting an ``ImportError`` escape is what keeps ``hasattr`` and
        ``getattr(..., default)`` working.
    """
    if name == "__version__":
        # Read from the installed distribution metadata rather than duplicated as a literal, so
        # ``pyproject.toml``'s ``[project] version`` stays the single source of truth. Resolved
        # here rather than at module scope for the same reason everything else is: an eager
        # ``importlib.metadata`` import costs every caller who never asks for the version.
        from importlib.metadata import PackageNotFoundError, version

        try:
            value: object = version("ordito")
        except PackageNotFoundError:
            # Running from a source tree that was never installed. A sentinel is friendlier than
            # an exception from an attribute every packaging tool probes.
            value = "0.0.0+unknown"
    elif name == "Trimesh":
        value = _import_submodule("ordito.mesh").Trimesh
    elif name in _CONFIG_NAMES:
        value = getattr(_import_submodule("ordito.config"), name)
    elif name in _LAZY_SUBMODULES:
        value = _import_submodule(f"ordito.{name}")
    else:
        raise AttributeError(f"module 'ordito' has no attribute {name!r}")
    globals()[name] = value
    return value


def _import_submodule(qualified: str) -> ModuleType:
    """
    Import ``qualified`` on a helper thread, so its frames do not reach the caller's.

    Warp's ``@wp.func`` decorator keeps its own frame in a local, a cycle only the garbage
    collector breaks, and every frame holds its callers through ``f_back``: imported from inside a
    call (the first ``od.<module>`` an ordito function touches), the module's thousand-odd
    decorations kept that call's frame, and every array it had bound, outputs included, until a
    full collection (26.9 MiB after a depth-9 octree reconstruction). On a helper thread the
    chain ends at the thread's start; an import error is raised again on the caller's thread. A
    touch from the helper thread, or while an ordito module is still executing its own import
    (which may be what ``qualified`` imports, and whose lock this thread holds), imports in place.
    """
    if threading.current_thread().name.startswith(_IMPORT_THREAD) or any(
        getattr(getattr(module, "__spec__", None), "_initializing", False)
        for key, module in list(sys.modules.items())
        if key.startswith("ordito.")
    ):
        return importlib.import_module(qualified)
    with concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=_IMPORT_THREAD) as pool:
        return pool.submit(importlib.import_module, qualified).result()


_IMPORT_THREAD = "ordito-lazy-import"


def __dir__() -> list[str]:
    """
    Public names, whether or not they have been resolved yet.

    Without this, ``dir(ordito)`` would list only the submodules some caller happened to touch,
    which is what makes a lazy package hard to explore interactively.

    Returns
    -------
    list of str
        Every name in ``__all__``, plus ``__version__``, sorted.
    """
    # ``__version__`` is named here rather than added to ``__all__``: it is resolvable through
    # ``__getattr__`` and should be discoverable interactively, but it is not part of the
    # star-import surface.
    return sorted([*__all__, "__version__"])
