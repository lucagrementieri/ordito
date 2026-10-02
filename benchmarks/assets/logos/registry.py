"""
Map a benchmark ``library`` id to its rendered logo (if one exists) and its display label.

Keyed by the exact ``id`` strings in ``benchmarks/conftest.py``'s ``LIBRARIES`` table -- that is
the only place library ids are defined, so a new library added there needs one new row here (a
missing id falls back to the id itself as the label, with no logo, rather than raising: a chart
should degrade to a plain text bar for an unmapped library, not fail to render).

Not every library has a distinct mark of its own to show. Three rows deliberately carry
``logo=None``:

- ``pymeshfix`` -- no distinct PyMeshFix project mark exists; its own docs reuse PyVista's logo
  (``reference/pymeshfix/doc/_static/pyvista_logo_sm.png``), which would misattribute the mark.
- ``igl`` (libigl) -- no small icon-only mark is bound in this project's vendored ``reference/``
  copy or found on the project's own site.
- ``potpourri3d`` -- potpourri3d is geometry-central's Python binding and has no separate mark of
  its own; geometry-central likewise has none vendored or found.

``pymeshlab`` is the one row that looks like it should be in that list but isn't: PyMeshLab wraps
MeshLab and ships no mark of its own, but MeshLab's own site (``meshlab.net/img/meshlabLogo.png``)
does -- and using it is the right call, not a stretch, because a PyMeshLab call in this suite is
never anything but a thin binding over the same MeshLab filters the desktop app runs. Labelled
"MeshLab" rather than "PyMeshLab" for that reason -- see ``raster/meshlab.png``'s entry in
``SOURCES.md``, and the caption in any chart that plots it.

A library gaining a mark later is a one-line change here plus a new source under ``svg/`` (or
``raster/``, for a project with no SVG to vendor) and a ``render.py`` re-run -- nothing in
``plot_comparison.py`` needs to change.
"""

from __future__ import annotations

from pathlib import Path

PNG_DIR = Path(__file__).parent / "png"

# id -> (png filename under png/, or None for a text-only fallback; display label)
_REGISTRY: dict[str, tuple[str | None, str]] = {
    "ordito-cpu": ("ordito.png", "ordito (CPU)"),
    "ordito-cuda": ("ordito.png", "ordito (CUDA)"),
    "trimesh": ("trimesh.png", "trimesh"),
    "igl": (None, "libigl"),
    "open3d": ("open3d.png", "Open3D"),
    "scipy": ("scipy.png", "SciPy"),
    "numpy": ("numpy.png", "NumPy"),
    "potpourri3d": (None, "potpourri3d"),
    "pymeshlab": ("meshlab.png", "MeshLab"),
    "pyvista": ("pyvista.png", "PyVista"),
    "meshlib": ("meshlib.png", "MeshLib"),
    "pymeshfix": (None, "PyMeshFix"),
    # There is deliberately no ``pytorch3d-cpu`` row in ``benchmarks/conftest.py``'s ``LIBRARIES``
    # -- pytorch3d is registered for its CUDA kernels alone and is Θ(N²) with no spatial structure
    # on either device, so its CPU rows would be minutes per call at scan-mesh sizes. Keep this
    # entry mapped for good measure (a `--library` filter or a future row could still name it) but
    # do not expect it to appear in real benchmark JSON.
    "pytorch3d-cuda": ("pytorch3d.png", "PyTorch3D (CUDA)"),
}


def logo_path(library_id: str) -> Path | None:
    """Return the PNG path for `library_id`'s logo, or `None` if it has no vendored mark."""
    filename, _ = _REGISTRY.get(library_id, (None, library_id))
    return (PNG_DIR / filename) if filename is not None else None


def display_label(library_id: str) -> str:
    """Return the human-readable label for `library_id`, falling back to the id itself."""
    _, label = _REGISTRY.get(library_id, (None, library_id))
    return label


def is_subject(library_id: str) -> bool:
    """Whether `library_id` is ordito itself rather than a reference library."""
    return library_id.startswith("ordito")
