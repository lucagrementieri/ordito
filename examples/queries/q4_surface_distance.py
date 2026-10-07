from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="Q4",
    title="Distance between two surfaces",
    summary="""
    How far one surface lies from another, two ways. Per vertex:
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] measures every vertex of
    a smoothed bunny ([`filter_laplacian`][ordito.smoothing.filter_laplacian]) against the
    original, with a sign (negative inside) that shows where smoothing ate into the shape and
    where it pushed out. As one number:
    [`mesh_to_mesh_distance`][ordito.proximity.mesh_to_mesh_distance] returns the exact clearance
    between two disjoint meshes and the pair of faces that realises it.
    """,
    credits=(
        (
            "PyVista: distance between surfaces",
            "https://docs.pyvista.org/examples/01-filter/distance_between_surfaces",
        ),
        (
            "MeshLib: signed distance",
            "https://meshlib.io/documentation/ExampleSignedDistances.html",
        ),
        ("PyMeshLab: distance", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    smoothed = od.smoothing.filter_laplacian(vertices, faces, lamb=0.5, iterations=40)
    signed = od.proximity.signed_distance_on_mesh(vertices, faces, smoothed, sign_mode="winding")
    print(f"smoothed vs original: {signed.numpy().min():.5f} to {signed.numpy().max():.5f}")

    # A second bunny, turned and set beside the first: how close do they come?
    turned = vertices.numpy() @ data.rotation((0.0, 1.0, 0.0), 120.0).T + [0.12, 0.0, 0.02]
    turned = wp.array(turned, dtype=wp.vec3, device=device)
    clearance, face_a, face_b = od.proximity.mesh_to_mesh_distance(vertices, faces, turned, faces)
    print(f"clearance {clearance:.5f} between faces {face_a} and {face_b}")
    # --8<-- [end:code]
    return {
        "smoothed": smoothed.numpy(),
        "signed": signed.numpy(),
        "turned": turned.numpy(),
        "face_a": face_a,
        "face_b": face_b,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    smoothed, signed, turned = result["smoothed"], result["signed"], result["turned"]
    witness_a = vertices[faces[result["face_a"]]].mean(axis=0)
    witness_b = turned[faces[result["face_b"]]].mean(axis=0)
    center = 0.5 * (witness_a + witness_b)
    half = 0.6 * float(np.linalg.norm(witness_b - witness_a)) + 0.004
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(
                        smoothed,
                        faces,
                        scalars=signed,
                        cmap="RdBu_r",
                        clim=r.symmetric_clim(signed),
                        scalar_bar="signed distance",
                    )
                ],
                title="Signed distance to the original",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREEN),
                    r.Mesh(turned, faces, color=r.GREY),
                    r.Points(np.stack([witness_a, witness_b]), color=r.RED, size=16),
                ],
                title="Two bunnies: the closest pair of faces",
                bounds=np.stack(
                    [
                        np.minimum(vertices.min(0), turned.min(0)),
                        np.maximum(vertices.max(0), turned.max(0)),
                    ]
                ),
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREEN, show_edges=True),
                    r.Mesh(turned, faces, color=r.GREY, show_edges=True),
                    r.Segments(np.stack([witness_a, witness_b])[None], color=r.RED, width=4),
                ],
                title="Close-up of the gap",
                bounds=np.stack([center - half, center + half]),
            ),
        ],
        camera=data.camera("bunny"),
        ncols=3,
        panel_size=(600, 560),
    )
