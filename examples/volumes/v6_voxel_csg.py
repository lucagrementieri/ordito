from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V6",
    title="Voxel CSG (approximate booleans)",
    summary="""
    ordito has no exact mesh booleans. What it has is set algebra on voxels: both solids are
    voxelized on one lattice with [`voxelize_mesh`][ordito.voxels.voxelize_mesh], combined
    cell by cell with [`union`][ordito.voxels.union],
    [`intersection`][ordito.voxels.intersection] and [`difference`][ordito.voxels.difference],
    and meshed back through [`to_field`][ordito.voxels.to_field] and
    [`marching_cubes`][ordito.levelset.marching_cubes]. The result is only as accurate as the
    voxel size: sharp edges are rounded off at that scale and the output is resampled
    everywhere, not just near the intersection curve.
    """,
    credits=(
        (
            "PyVista: boolean operations",
            "https://docs.pyvista.org/examples/01-filter/boolean_operations",
        ),
        ("libigl 609 (CSG)", "https://libigl.github.io/tutorial/#boolean-operations-on-meshes"),
        ("MeshLib: boolean", "https://meshlib.io/documentation/ExampleMeshBoolean.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    def csg(
        a: tuple[wp.array[wp.vec3], wp.array[wp.int32]],
        b: tuple[wp.array[wp.vec3], wp.array[wp.int32]],
        voxel_size: float,
        origin: wp.vec3,
    ):
        solid_a = od.voxels.voxelize_mesh(*a, voxel_size, origin=origin, mode="solid")
        solid_b = od.voxels.voxelize_mesh(*b, voxel_size, origin=origin, mode="solid")
        results = {}
        for operation in (od.voxels.union, od.voxels.intersection, od.voxels.difference):
            field, bounds = od.voxels.to_field(operation(solid_a, solid_b))
            results[operation.__name__] = od.levelset.marching_cubes(field, 0.5, bounds=bounds)
        return results

    # Two overlapping spheres, split into their two components.
    sphere_a, sphere_b = od.combine.split(*data.load("two_spheres", device))
    spheres = csg(sphere_a, sphere_b, 0.02, origin=wp.vec3(-1.1, -1.1, -1.1))

    # The bunny (its open base sealed first) and a box over its back.
    bunny_vertices, bunny_faces = data.load("bunny", device)
    bunny = (bunny_vertices, od.holes.fill_min_weight(bunny_vertices, bunny_faces))
    box = od.creation.box(bounds=[[-0.01, 0.02, -0.08], [0.08, 0.12, 0.08]], device=device)
    # The origin is half a cell off round numbers so that no box face lies on a cell boundary.
    carved = csg(bunny, box, 0.0012, origin=wp.vec3(-0.1006, -0.0006, -0.0906))
    for name, (_, f) in carved.items():
        print(f"bunny {name} box: {f.shape[0] // 3} faces")
    # --8<-- [end:code]

    def as_np(mesh: tuple[wp.array[wp.vec3], wp.array[wp.int32]]) -> tuple[np.ndarray, np.ndarray]:
        return mesh[0].numpy(), mesh[1].numpy().reshape(-1, 3)

    return {
        "spheres_in": [as_np(sphere_a), as_np(sphere_b)],
        "bunny_in": [as_np(bunny), as_np(box)],
        "spheres": {k: as_np(m) for k, m in spheres.items()},
        "carved": {k: as_np(m) for k, m in carved.items()},
    }


def figure(result: dict[str, Any]) -> r.Figure:
    panels = []
    for inputs, results, camera in (
        ("spheres_in", "spheres", data.camera("two_spheres")),
        ("bunny_in", "carved", data.camera("bunny")),
    ):
        (av, af), (bv, bf) = result[inputs]
        both = np.concatenate([av, bv])
        frame = np.stack([both.min(axis=0), both.max(axis=0)])
        panels.append(
            r.Panel(
                [
                    r.Mesh(av, af, color=r.GREEN, opacity=0.6),
                    r.Mesh(bv, bf, color=r.BLUE, opacity=0.6),
                ],
                title="Inputs",
                camera=camera,
                bounds=frame,
            )
        )
        for name, (v, f) in result[results].items():
            panels.append(
                r.Panel([r.Mesh(v, f, color=r.GREEN)], title=name, camera=camera, bounds=frame)
            )
    return r.Figure(panels, ncols=4, link_bounds=False, panel_size=(500, 440))
