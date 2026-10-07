from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="R6",
    title="Delaunay flips and the intrinsic Delaunay triangulation",
    summary="""
    The patch is cut along the long diagonal of every sheared cell and carries a few needles and
    caps, so most of its edges have a negative cotangent weight (red): the cotangent Laplacian
    then violates the maximum principle.
    [`flip_to_delaunay`][ordito.remesh.flip_to_delaunay] flips edges in space toward the empty
    circumcircle property, with a dihedral gate so the surface barely moves;
    [`intrinsic_delaunay`][ordito.remesh.intrinsic_delaunay] flips *intrinsically*, along
    geodesics across the two triangles, so no vertex or surface point moves at all; what is
    left is on the boundary, where no flip can help.
    [`robust_laplacian`][ordito.laplacian.robust_laplacian] builds the cotangent Laplacian on
    that intrinsic triangulation (after mollifying the degenerate faces).
    """,
    credits=(
        ("libigl 716", "https://libigl.github.io/tutorial/#intrinsic-delaunay-triangulation"),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import math

    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("sliver_patch", device)
    flipped = od.remesh.flip_to_delaunay(vertices, faces, max_angle_change=math.radians(30.0))
    intrinsic_faces, lengths, n_flips = od.remesh.intrinsic_delaunay(vertices, faces)

    weights = {
        "input": od.laplacian.cotmatrix_entries(vertices, faces),
        "flip_to_delaunay": od.laplacian.cotmatrix_entries(vertices, flipped),
        "intrinsic_delaunay": od.laplacian.cotmatrix_entries_intrinsic(lengths),
    }
    for name, w in weights.items():
        print(f"{name}: {(w.numpy() < 0).sum()} negative half-cotangents")
    print(f"{n_flips} intrinsic flips")

    for build in (od.laplacian.cotmatrix, od.laplacian.robust_laplacian):
        matrix = build(vertices, faces)
        n = matrix.nnz_sync()  # the value buffer may be longer than the matrix
        rows = np.repeat(np.arange(matrix.nrow), np.diff(matrix.offsets.numpy()[: matrix.nrow + 1]))
        off_diagonal = matrix.values.numpy()[:n][rows != matrix.columns.numpy()[:n]]
        print(f"{build.__name__}: {(off_diagonal < 0).sum()} negative off-diagonal entries")
    # --8<-- [end:code]
    return {
        "faces": {
            "input": faces.numpy().reshape(-1, 3),
            "flip_to_delaunay": flipped.numpy().reshape(-1, 3),
            "intrinsic_delaunay": intrinsic_faces.numpy().reshape(-1, 3),
        },
        "weights": {name: w.numpy() for name, w in weights.items()},
    }


def _negative_edges(vertices: np.ndarray, faces: np.ndarray, half_cot: np.ndarray) -> np.ndarray:
    """Segments of the edges whose summed cotangent weight is negative."""
    a = faces[:, [1, 2, 0]].ravel()
    b = faces[:, [2, 0, 1]].ravel()
    keys = np.minimum(a, b).astype(np.int64) * vertices.shape[0] + np.maximum(a, b)
    unique, inverse = np.unique(keys, return_inverse=True)
    total = np.zeros(unique.size)
    np.add.at(total, inverse, np.nan_to_num(half_cot.ravel(), nan=-1.0, neginf=-1.0))
    bad = unique[total < 0]
    i, j = bad // vertices.shape[0], bad % vertices.shape[0]
    return np.stack([vertices[i], vertices[j]], axis=1)


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, _ = data.arrays("sliver_patch")
    panels = []
    for name, title in (
        ("input", "Input"),
        ("flip_to_delaunay", "flip_to_delaunay"),
        ("intrinsic_delaunay", "intrinsic_delaunay"),
    ):
        faces = result["faces"][name]
        bad = _negative_edges(vertices, faces, result["weights"][name])
        panels.append(
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN, show_edges=True, smooth=False),
                    r.Segments(bad, color=r.RED, width=3.0),
                ],
                title=f"{title}: {len(bad)} negative edges",
            )
        )
    return r.Figure(panels, camera=data.camera("sliver_patch"), panel_size=(620, 560))
