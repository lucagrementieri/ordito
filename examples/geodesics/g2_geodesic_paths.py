from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

TAIL, EAR, NOSE = 12217, 22820, 11842

META = Meta(
    id="G2",
    title="Shortest paths: along the edges and across the faces",
    summary="""
    Two answers to "how do I get from the ear to the tail". The edge graph's shortest path comes
    from [`shortest_path_envelope`][ordito.graph.shortest_path_envelope] (Dijkstra's distances,
    computed by parallel relaxation over the [`edges_to_csr`][ordito.graph.edges_to_csr]
    adjacency) followed by a short backtrack; it zigzags because it may only follow edges.
    [`geodesic_path`][ordito.geodesic_walk.geodesic_path] instead descends one heat-method
    distance field from every target, so its paths cut straight across faces and are shorter.
    """,
    credits=(
        ("potpourri3d: geodesic paths", "https://github.com/nmwsharp/potpourri3d"),
        ("libigl 206", "https://libigl.github.io/tutorial/#exact-discrete-geodesics"),
        ("PyVista: geodesic", "https://docs.pyvista.org/examples/01-filter/geodesic"),
        (
            "trimesh: shortest path",
            "https://github.com/mikedh/trimesh/blob/main/examples/shortest.ipynb",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    tail, targets = 12217, [22820, 11842]  # from the ear tip and the chin to the tail

    # Across the faces: descend the heat distance to the tail from every target.
    source = wp.array([tail], dtype=wp.int32, device=device)
    points, offsets = od.geodesic_walk.geodesic_path(
        vertices, faces, source, wp.array(targets, dtype=wp.int32, device=device)
    )
    geodesics = od.array.split(points, offsets)

    # Along the edges: Dijkstra distances to the tail, then step to the closest neighbour.
    n = vertices.shape[0]
    edges = od.edges.edges_unique(faces, n_vertices=n, validate=False)[0]
    graph = od.graph.edges_to_csr(n, edges, od.edges.edges_unique_length(vertices, faces, edges))
    seed = np.full(n, 1e6, dtype=np.float32)
    seed[tail] = 0.0
    dist = od.graph.shortest_path_envelope(graph, wp.array(seed, device=device)).numpy()
    rows, cols, weights = graph.offsets.numpy(), graph.columns.numpy(), graph.values.numpy()
    edge_paths = []
    for v in targets:
        path = [v]
        while v != tail:
            ring = slice(rows[v], rows[v + 1])
            v = cols[ring][np.argmin(dist[cols[ring]] + weights[ring])]
            path.append(v)
        edge_paths.append(np.array(path))

    for name, geodesic, path in zip(["ear", "chin"], geodesics, edge_paths, strict=True):
        along_edges = np.linalg.norm(np.diff(vertices.numpy()[path], axis=0), axis=1).sum()
        print(
            f"{name} -> tail: {along_edges:.4f} along edges, "
            f"{od.polyline.polyline_length(geodesic):.4f} across faces"
        )
    # --8<-- [end:code]
    return {"geodesics": [g.numpy() for g in geodesics], "edge_paths": edge_paths}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    ends = r.Points(vertices[[TAIL, EAR, NOSE]], color=r.RED, size=18)
    edge_lines = [vertices[p] for p in result["edge_paths"]]
    lines = r.Lines(result["geodesics"], color=r.ORANGE, width=3.0)
    kink = vertices[result["edge_paths"][1][len(result["edge_paths"][1]) * 2 // 3]]
    head = np.stack([kink - 0.025, kink + 0.025])
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Lines(edge_lines, color=r.BLUE, width=3.0),
                    ends,
                ],
                title="Along the edges",
            ),
            r.Panel(
                [r.Mesh(vertices, faces, color=r.LIGHT_GREEN), lines, ends],
                title="Across the faces",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN, show_edges=True, line_width=0.4),
                    r.Lines(edge_lines, color=r.BLUE, width=4.0),
                    r.Lines(result["geodesics"], color=r.ORANGE, width=4.0),
                ],
                title="Both (close-up)",
                bounds=head,
            ),
        ],
        camera=data.camera("bunny"),
    )
