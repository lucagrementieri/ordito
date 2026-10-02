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

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Re-imported for the type checker and for IDE completion only; at runtime these names come
    # from ``__getattr__``. basedpyright reads this branch, so the public surface stays typed.
    from ordito import (
        adjacency,
        array,
        boundary,
        bounds,
        combine,
        constants,
        creation,
        curvature,
        edges,
        energies,
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
    from ordito.mesh import Trimesh

__all__ = [
    "Trimesh",
    "adjacency",
    "array",
    "boundary",
    "bounds",
    "combine",
    "constants",
    "creation",
    "curvature",
    "edges",
    "energies",
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
_LAZY_SUBMODULES = frozenset(__all__) - {"Trimesh"}


def __getattr__(name: str) -> object:
    """
    Resolve a submodule, ``Trimesh``, or ``__version__`` on first access (:pep:`562`).

    The result is cached in this module's namespace, so only the first access reaches here and
    every later ``ordito.<name>`` is an ordinary attribute lookup.

    Parameters
    ----------
    name
        The attribute being looked up.

    Returns
    -------
    object
        The imported submodule, the ``Trimesh`` class, or the installed version string.

    Raises
    ------
    AttributeError
        If ``name`` is not a public submodule, ``Trimesh`` or ``__version__``. Raising this rather
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
        value = importlib.import_module("ordito.mesh").Trimesh
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"ordito.{name}")
    else:
        raise AttributeError(f"module 'ordito' has no attribute {name!r}")
    globals()[name] = value
    return value


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
