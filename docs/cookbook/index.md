# Cookbook

Task-oriented recipes. Each page is self-contained and runnable end to end — copy the code block,
`uv run python` it (with `ordito` installed), and it prints real diagnostics from a real,
intentionally imperfect input.

| Recipe | What it covers |
|---|---|
| [Cleaning and remeshing a rough mesh](clean-and-remesh.md) | `repair.make_solid` → `remesh.isotropic_remesh` on a mesh with a hole and stray debris |
| [Point cloud to watertight surface](point-cloud-to-surface.md) | Normal estimation, then `reconstruction.screened_poisson` and `reconstruction.ball_pivoting` side by side, and why sampling decides ball pivoting's holes |
| [Geodesic distance fields](geodesic-distance.md) | `heat.heat_geodesic` checked against the exact distance on a sphere, reusing operators, and the vector heat family |
| [Aligning two scans](align-two-scans.md) | `registration.procrustes` with known correspondences, `registration.icp` and `icp_point_to_plane` without, and how to tell when ICP stopped too early |

Every recipe here is also a starting point rather than a finished pipeline — each links back into
the relevant API reference page for the full set of keyword arguments a real use case will
eventually need (a target edge length that varies over the surface, a feature-angle threshold, a
solver tolerance). See also [Concepts](../concepts.md) for the design principles these recipes all
lean on, and [Migrating from another library](../migrating-from/index.md) if you're translating an
existing pipeline built on trimesh, Open3D, libigl, MeshLab, potpourri3d, or PyTorch3D.
