from __future__ import annotations

import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T5",
    title="Edge structure: dihedral angles, convexity and boundaries",
    summary="""
    Every pair of faces sharing an edge is a row of
    [`face_adjacency`][ordito.mesh.Trimesh.face_adjacency], with the shared vertex pair in
    [`face_adjacency_edges`][ordito.mesh.Trimesh.face_adjacency_edges]. The angle between the two
    face normals is [`face_adjacency_angles`][ordito.mesh.Trimesh.face_adjacency_angles], and
    [`face_adjacency_convex`][ordito.mesh.Trimesh.face_adjacency_convex] says whether the edge
    folds outward (a ridge) or inward (a valley). Thresholding the angle picks out the creases of
    a machined part. On the scanned bunny, the signed angle averaged around each vertex separates
    ridges from folds. [`boundary_edges`][ordito.mesh.Trimesh.boundary_edges] counts the edges on
    the rims of the holes in the bunny's base.
    """,
    credits=(
        ("trimesh: quick start", "https://trimesh.org/quick_start.html"),
        ("trimesh: examples", "https://trimesh.org/examples.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    out = {}
    for name, threshold in [("cad_part", 30.0), ("bunny", 40.0)]:
        mesh = od.Trimesh(*data.load(name, device))
        angles = np.degrees(mesh.face_adjacency_angles.numpy())
        convex = mesh.face_adjacency_convex.numpy()
        sharp = angles > threshold
        print(
            f"{name}: {mesh.edges_unique.shape[0]} edges, {sharp.sum()} sharper than "
            f"{threshold:.0f}° ({(sharp & convex).sum()} convex, "
            f"{(sharp & ~convex).sum()} concave), {mesh.boundary_edges.shape[0]} on the boundary"
        )
        out[name] = (mesh.face_adjacency_edges.numpy(), angles, convex, sharp)
    # --8<-- [end:code]
    return out


def _by_angle(vertices: np.ndarray, pairs: np.ndarray, angles: np.ndarray) -> list[r.Layer]:
    """Edges binned by dihedral angle, one coloured segment layer per bin (flat edges skipped)."""
    cmap = plt.get_cmap("plasma")
    bins = np.linspace(1.0, 90.0, 9)
    layers: list[r.Layer] = []
    for lo, hi in itertools.pairwise(bins):
        pick = (angles >= lo) & ((angles < hi) | (hi == bins[-1]))
        if pick.any():
            color = mpl_hex(cmap((lo - bins[0]) / (bins[-1] - bins[0])))
            layers.append(r.Segments(vertices[pairs[pick]], color=color, width=2.5))
    return layers


def mpl_hex(rgba: tuple[float, ...]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in rgba[:3]))


def figure(result: dict[str, Any]) -> r.Figure:
    part_v, part_f = data.arrays("cad_part")
    bunny_v, bunny_f = data.arrays("bunny")
    pairs, angles, convex, sharp = result["cad_part"]
    b_pairs, b_angles, b_convex, _ = result["bunny"]
    signed = np.zeros(bunny_v.shape[0])
    count = np.zeros(bunny_v.shape[0])
    for k in range(2):
        np.add.at(signed, b_pairs[:, k], np.where(b_convex, b_angles, -b_angles))
        np.add.at(count, b_pairs[:, k], 1.0)
    signed /= np.maximum(count, 1.0)
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(part_v, part_f, color=r.GREY, smooth=False),
                    *_by_angle(part_v, pairs, angles),
                ],
                title="Dihedral angle: 1° purple to 90° yellow",
                camera=data.camera("cad_part"),
            ),
            r.Panel(
                [
                    r.Mesh(part_v, part_f, color=r.GREY, smooth=False),
                    r.Segments(part_v[pairs[sharp & convex]], color=r.BLUE, width=3.0),
                    r.Segments(part_v[pairs[sharp & ~convex]], color=r.RED, width=3.0),
                ],
                title="Creases: convex blue, concave red",
                camera=data.camera("cad_part"),
            ),
            r.Panel(
                [
                    r.Mesh(
                        bunny_v,
                        bunny_f,
                        scalars=signed,
                        cmap="RdBu_r",
                        clim=r.symmetric_clim(signed),
                        scalar_bar="mean signed angle (°)",
                    )
                ],
                title="Bunny: ridges red, folds blue",
                camera=data.camera("bunny"),
            ),
        ],
        link_bounds=False,
    )
