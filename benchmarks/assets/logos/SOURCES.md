# Logo sources

Every file under `svg/` and `raster/` is copied verbatim from the project it identifies (or
fetched from that project's own public site) — none is redrawn or modified beyond the
cropping and scaling `render.py` applies at rasterization time. See `NOTICE.md` for the
licensing/trademark statement these vendored marks are subject to.

| File | Source | Note |
|---|---|---|
| `trimesh.svg` | `reference/trimesh/docs/static/images/favicon.svg` (trimesh's own repository) | The icon-only mark, not the `logotype-*.svg` wordmarks |
| `open3d.svg` | `reference/Open3D/cpp/apps/Open3DViewer/icon.svg` (Open3D's own repository) | The desktop viewer app icon — square, unlike the docs' horizontal wordmark |
| `pyvista.svg` | `reference/pyvista/logo/pyvista_sq.svg` (PyVista's own repository) | The square variant PyVista itself ships alongside a wordmark one |
| `meshlib.svg` | `reference/MeshLib/wasm/logo.svg` (MeshLib's own repository) | |
| `numpy.svg` | <https://numpy.org/images/logo.svg> (NumPy's own site) | Fetched directly; NumPy publishes this for reuse |
| `scipy.svg` | <https://scipy.org/images/logo.svg> (SciPy's own site) | Fetched directly; SciPy publishes this for reuse |
| `ordito.png` | `docs/assets/logo.svg` (this repository), rendered directly | ordito's own mark, for the subject bars; there is no copy under `svg/` |
| `raster/pytorch3d.png` | <https://pytorch3d.org/img/pytorch3dfavicon.png> (PyTorch3D's own site) | The site's favicon: the coloured mark, not the all-white `pytorch3dicon.svg` |
| `raster/meshlab.png` | <https://www.meshlab.net/img/meshlabLogo.png> (MeshLab's own site) | Used for the `pymeshlab` row — MeshLab publishes no SVG, only this PNG; see `registry.py` for why MeshLab's mark (not a separate PyMeshLab one) is the right choice there |

Three benchmarked libraries have no entry here and render as a text-only bar instead — see the
module docstring in `registry.py` for which ones and why (in short: `igl`, `potpourri3d`, and
`pymeshfix` have no small icon-only mark either vendored in this repository's `reference/` mirror
or found at a stable URL on the project's own site).

Regenerate `png/` after touching anything here: `uv run python benchmarks/assets/logos/render.py`.
