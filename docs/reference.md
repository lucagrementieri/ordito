# API reference

Every function is reached as `od.<module>.<function>` after `import ordito as od`. Each module
loads on first use, so `import ordito` stays cheap however many modules exist. The pages below are
generated from the docstrings. Each opens with a table of the module's functions; each function
then lists its parameters (array shapes first), what it returns, what it raises, related functions,
and an example. The examples run in the test suite on both CPU and CUDA, and every value they
print is checked, so the output shown is the output you will get.

Three conventions hold across the whole reference (see [Concepts](concepts.md) for the reasons):

- A mesh is two arrays: `vertices`, a `wp.array[wp.vec3]` of shape `(n_vertices,)`, and `faces`, a
  **flat** `wp.array[wp.int32]` of shape `(3 * n_faces,)`. A per-face result has shape
  `(n_faces,)`, and a per-vertex result has shape `(n_vertices,)`.
- Outputs live on the device of the inputs. Pass inputs that are all on the same device; mixing
  devices raises `RuntimeError`.
- A variable-length result per item (one boundary loop, one neighbour list) comes back packed as
  `(values, offsets)`, where item `i` is `values[offsets[i]:offsets[i + 1]]`. A plain-list
  variant exists alongside where it is convenient (`boundary_loops` next to
  `boundary_loops_with_offsets`).

## Find a function by task

| I want to... | Use |
|---|---|
| **Create and load** | |
| make a primitive (box, sphere, cylinder, torus...) | [`creation`][ordito.creation]: [`box`][ordito.creation.box], [`icosphere`][ordito.creation.icosphere], [`uv_sphere`][ordito.creation.uv_sphere], [`cylinder`][ordito.creation.cylinder], [`torus`][ordito.creation.torus], [`grid`][ordito.creation.grid] |
| revolve, extrude or sweep a profile | [`revolve`][ordito.creation.revolve], [`extrude_polygon`][ordito.creation.extrude_polygon], [`sweep_polygon`][ordito.creation.sweep_polygon] |
| load a mesh file (OBJ, PLY, STL, ...) | [`io.load_mesh_data`][ordito.io.load_mesh_data], [`io.load_mesh`][ordito.io.load_mesh] (needs `ordito[io]`) |
| work with an object that caches derived quantities | [`Trimesh`][ordito.mesh.Trimesh] |
| **Inspect** | |
| face normals, areas, angles, centroids | [`triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas], [`face_angles`][ordito.triangles.face_angles], [`face_centroids`][ordito.triangles.face_centroids] |
| vertex normals | [`vertices.vertex_normals`][ordito.vertices.vertex_normals] |
| volume, centre of mass, inertia, Euler characteristic | [`measures`][ordito.measures]: [`volume`][ordito.measures.volume], [`moments`][ordito.measures.moments], [`euler_characteristic`][ordito.measures.euler_characteristic] |
| is it watertight / manifold / orientable / self-intersecting? | [`validation`][ordito.validation]: [`is_watertight`][ordito.validation.is_watertight], [`is_edge_manifold`][ordito.validation.is_edge_manifold], [`is_orientable`][ordito.validation.is_orientable], [`is_self_intersecting`][ordito.validation.is_self_intersecting] |
| are two meshes the same up to vertex and face numbering? | [`comparison.meshes_allclose`][ordito.equality.meshes_allclose], [`faces_equal`][ordito.equality.faces_equal] |
| unique edges and their lengths | [`edges.edges_unique`][ordito.edges.edges_unique], [`edges_unique_length`][ordito.edges.edges_unique_length] |
| boundary edges and hole rims | [`boundary.boundary_edges`][ordito.boundary.boundary_edges], [`boundary_loops`][ordito.boundary.boundary_loops] |
| neighbouring faces and connected components | [`adjacency.face_adjacency`][ordito.adjacency.face_adjacency], [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels], [`combine.split`][ordito.combine.split] |
| curvature | [`curvature`][ordito.curvature]: [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature], [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature], [`principal_curvature`][ordito.curvature.principal_curvature] |
| bounding boxes | [`bounds.aabb`][ordito.bounds.aabb], [`bounds.oriented_bounding_box`][ordito.bounds.oriented_bounding_box] |
| triangle quality | [`triangles.face_quality`][ordito.triangles.face_quality] |
| **Repair** | |
| make a broken mesh a closed solid in one call | [`repair.make_solid`][ordito.repair.make_solid] |
| merge duplicate vertices, drop degenerate or unused elements | [`repair.remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices], [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces], [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices] |
| fix needle and cap triangles within a deviation bound | [`repair.fix_degeneracies`][ordito.repair.fix_degeneracies], [`collapse_small_triangles`][ordito.repair.collapse_small_triangles] |
| fix the winding, or point normals outward | [`repair.make_winding_consistent`][ordito.repair.make_winding_consistent], [`make_normals_outward`][ordito.repair.make_normals_outward] |
| remove small floating pieces | [`repair.remove_small_components`][ordito.repair.remove_small_components] |
| fill holes | [`holes`][ordito.holes]: [`fill_small`][ordito.holes.fill_small], [`fill_min_weight`][ordito.holes.fill_min_weight], [`fill_smooth`][ordito.holes.fill_smooth] |
| remove self-intersections | [`repair.fix_self_intersections`][ordito.repair.fix_self_intersections] |
| **Edit** | |
| move, rotate, scale, mirror | [`transform`][ordito.transform]: [`transform_mesh`][ordito.transform.transform_mesh], [`rotation_matrix`][ordito.transform.rotation_matrix] |
| combine meshes, or split into parts | [`combine.concatenate`][ordito.combine.concatenate], [`combine.split`][ordito.combine.split] |
| keep a subset of faces | [`selection.submesh_from_face_mask`][ordito.selection.submesh_from_face_mask] |
| remesh to uniform edge lengths | [`remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh] |
| simplify (decimate) | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate], [`cluster_decimate`][ordito.remesh.cluster_decimate] |
| subdivide | [`remesh.subdivide`][ordito.remesh.subdivide], [`subdivide_loop`][ordito.remesh.subdivide_loop], [`subdivide_to_size`][ordito.remesh.subdivide_to_size] |
| smooth | [`smoothing`][ordito.smoothing]: [`filter_taubin`][ordito.smoothing.filter_taubin], [`filter_laplacian`][ordito.smoothing.filter_laplacian], [`filter_implicit_fairing`][ordito.smoothing.filter_implicit_fairing] |
| offset or thicken a surface | [`levelset.offset_mesh`][ordito.levelset.offset_mesh], [`levelset.thicken_mesh`][ordito.levelset.thicken_mesh] |
| cut with a plane | [`intersection.slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane], [`split_mesh_with_plane`][ordito.intersection.split_mesh_with_plane], [`mesh_with_plane`][ordito.intersection.mesh_with_plane] |
| extract an isosurface | [`levelset.marching_cubes`][ordito.levelset.marching_cubes], [`intersection.marching_triangles`][ordito.intersection.marching_triangles] |
| **Query** | |
| closest point on a surface, signed distance | [`proximity.closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh], [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh], [`winding_number`][ordito.proximity.winding_number] |
| cast rays | [`ray.intersects_first`][ordito.ray.intersects_first], [`intersects_location`][ordito.ray.intersects_location] |
| test which points are inside | [`ray.contains_points`][ordito.ray.contains_points] |
| nearest neighbours, radius search | [`neighbors.query_nearest`][ordito.neighbors.query_nearest], [`query_ball`][ordito.neighbors.query_ball] |
| Chamfer and Hausdorff distances | [`metrics`][ordito.metrics]: [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh], [`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh] |
| intersection curve of two meshes, colliding faces | [`intersection.mesh_with_mesh`][ordito.intersection.mesh_with_mesh], [`collision_masks`][ordito.intersection.collision_masks] |
| **Point clouds** | |
| sample points on a surface | [`sample.sample_surface`][ordito.sample.sample_surface], [`sample_surface_poisson_disk`][ordito.sample.sample_surface_poisson_disk] |
| estimate normals | [`points.estimate_normals`][ordito.points.estimate_normals] |
| down-sample | [`voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample], [`points.farthest_point_sample`][ordito.points.farthest_point_sample] |
| remove outliers | [`points.statistical_outlier_mask`][ordito.points.statistical_outlier_mask], [`radius_outlier_mask`][ordito.points.radius_outlier_mask] |
| convex hull ([example](examples/points.md#pc8)) | [`points.convex_hull`][ordito.points.convex_hull] (exact), [`convex_superset_mask`][ordito.points.convex_superset_mask] (prefilter) |
| reconstruct a surface | [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson], [`ball_pivoting`][ordito.reconstruction.ball_pivoting] |
| align two scans | [`registration.icp`][ordito.registration.icp], [`icp_point_to_plane`][ordito.registration.icp_point_to_plane], [`procrustes`][ordito.registration.procrustes] |
| voxelize | [`voxels.voxelize_mesh`][ordito.voxels.voxelize_mesh], [`voxelize_points`][ordito.voxels.voxelize_points] |
| **Differential geometry** | |
| cotangent Laplacian and mass matrix | [`laplacian.cotmatrix`][ordito.laplacian.cotmatrix], [`laplacian.mass_matrix`][ordito.laplacian.mass_matrix] |
| solve a constrained quadratic (harmonic, biharmonic fields) | [`linalg.min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] |
| geodesic distance | [`heat.heat_geodesic`][ordito.heat.heat_geodesic] (approximate, fast), [`geodesic_walk.geodesic_path`][ordito.geodesic_walk.geodesic_path] |
| flatten to the plane (UV parametrization) | [`parametrization`][ordito.parametrization]: [`harmonic`][ordito.parametrization.harmonic], [`lscm`][ordito.parametrization.lscm], [`arap`][ordito.parametrization.arap] |
| bake or sample textures | [`texture`][ordito.texture] |
| **Arrays** | |
| NumPy-like `unique`, `sort`, `isin`, `searchsorted`-style helpers | [`grouping`][ordito.grouping], [`array`][ordito.array] |
| reductions (`min`, `max`, `sum`, `mean`) without a host round trip | [`reduce`][ordito.reduce] |
| per-group sums (`bincount` with weights, `add.reduceat`), reproducible under the deterministic setting | [`reduce.scatter_sum`][ordito.reduce.scatter_sum], [`reduce.segment_sum`][ordito.reduce.segment_sum] |

## Modules

### Primitives & creation

| Module | Contents |
|---|---|
| [creation](api/ordito/creation.md) | Parametric mesh generators: boxes, Platonic solids, surfaces of revolution and extrusions. |

### Mesh structure & topology

| Module | Contents |
|---|---|
| [mesh](api/ordito/mesh.md) | The optional [`Trimesh`][ordito.mesh.Trimesh] container, which caches derived quantities. |
| [vertices](api/ordito/vertices.md) | Per-vertex quantities: normals under three weighting rules, and the angle defect. |
| [edges](api/ordito/edges.md) | Triangle edges: the directed corner edges, the deduplicated undirected set, and their lengths. |
| [triangles](api/ordito/triangles.md) | Quantities carried by one triangle at a time: its normal, area, angles, shape and barycentre. |
| [halfedge](api/ordito/halfedge.md) | Implicit halfedge connectivity: edge twins and counter-clockwise vertex one-rings. |
| [adjacency](api/ordito/adjacency.md) | Which faces share an edge, what each adjacent pair looks like, connected components. |
| [boundary](api/ordito/boundary.md) | Boundary edges, vertices and loops. |
| [selection](api/ordito/selection.md) | Face-subset extraction and vertex-selection morphology. |
| [validation](api/ordito/validation.md) | Validity predicates (manifold, watertight, orientable, self-intersecting) and their per-element masks. |
| [equality](api/ordito/equality.md) | Equality of meshes up to the numbering of their vertices and faces. |
| [homology](api/ordito/homology.md) | Homology generators: the loops that make a surface's topology. |
| [tangent_space](api/ordito/tangent_space.md) | Per-vertex and per-face tangent frames, and the angles that transport vectors between them. |

### Measures & shape descriptors

| Module | Contents |
|---|---|
| [measures](api/ordito/measures.md) | What the whole mesh adds up to: enclosed volume, moments, Euler characteristic. |
| [curvature](api/ordito/curvature.md) | Discrete curvature: principal directions and magnitudes, Gaussian and mean. |
| [bounds](api/ordito/bounds.md) | Bounding boxes: building them, and selecting or cropping with them. |
| [visibility](api/ordito/visibility.md) | Ambient occlusion, shape diameter and thickness: how far the surface is from a point. |

### Mesh editing & repair

| Module | Contents |
|---|---|
| [transform](api/ordito/transform.md) | Affine transforms: build a `4x4` matrix, apply it, classify what it preserves. |
| [repair](api/ordito/repair.md) | Removing what should not be in a mesh: duplicates, bad triangles, inconsistent winding. |
| [holes](api/ordito/holes.md) | Closing boundary holes, and stitching two open meshes along a rim. |
| [combine](api/ordito/combine.md) | Assembling meshes from parts and splitting them back apart. |
| [remesh](api/ordito/remesh.md) | Changing a mesh's triangulation: subdividing, decimating, and improving triangle shapes. |
| [smoothing](api/ordito/smoothing.md) | Laplacian smoothing filters, for vertex positions and for per-vertex scalar fields. |
| [seams](api/ordito/seams.md) | Feature and UV-seam edges, and the topological cut along them. |
| [levelset](api/ordito/levelset.md) | Surfaces from scalar fields: marching cubes, offsets and thickening. |

### Spatial queries

| Module | Contents |
|---|---|
| [proximity](api/ordito/proximity.md) | Where a point stands relative to a mesh: closest point, signed distance, winding number. |
| [ray](api/ordito/ray.md) | Ray casting and point-in-mesh tests. |
| [neighbors](api/ordito/neighbors.md) | Point-set acceleration structures (BVH, hash grid), k-nearest and radius queries. |
| [intersection](api/ordito/intersection.md) | Where a mesh meets a plane, another mesh, or an isovalue. |
| [metrics](api/ordito/metrics.md) | Chamfer and Hausdorff distances between point clouds and meshes. |

### Point clouds, voxels & reconstruction

| Module | Contents |
|---|---|
| [points](api/ordito/points.md) | Point-cloud geometry: plane and line fits, covariance, normals, outliers, down-sampling, convex hull. |
| [sample](api/ordito/sample.md) | New points on a mesh surface, inside its volume, or over a sphere. |
| [voxels](api/ordito/voxels.md) | Sparse voxel grids: voxelize, test membership, morphology, booleans, and mesh the result. |
| [reconstruction](api/ordito/reconstruction.md) | Surfaces from points: Delaunay, screened Poisson, ball pivoting, uniform resampling. |
| [registration](api/ordito/registration.md) | Point-cloud registration: Procrustes analysis and ICP. |

### Operators & solvers

| Module | Contents |
|---|---|
| [laplacian](api/ordito/laplacian.md) | Discrete Laplacians, gradients and mass matrices, with intrinsic repairs for bad meshes. |
| [energies](api/ordito/energies.md) | Mesh energies: the scalar regularization losses, and the operators the solvers minimize. |
| [linalg](api/ordito/linalg.md) | Sparse solvers: constrained quadratic minimization, conjugate gradients, preconditioners. |
| [cholesky](api/ordito/cholesky.md) | Sparse Cholesky factorization of symmetric positive-definite operators, on the device. |
| [interpolation](api/ordito/interpolation.md) | Moving a field between faces, vertices and edges, or from scattered points. |
| [parametrization](api/ordito/parametrization.md) | Flattening a mesh into the plane: harmonic, Tutte, ARAP and LSCM. |

### Geodesics & heat-method solvers

| Module | Contents |
|---|---|
| [geodesic_walk](api/ordito/geodesic_walk.md) | Straightest geodesics, shortest paths and loop shortening across the surface. |
| [heat](api/ordito/heat.md) | Heat-method distances, signed distances, vector transport and log maps. |

### Curves

| Module | Contents |
|---|---|
| [polyline](api/ordito/polyline.md) | Open and closed 3D polylines: length, resampling, simplification, triangulation. |

### Attributes & I/O

| Module | Contents |
|---|---|
| [texture](api/ordito/texture.md) | Bake per-vertex attributes into UV-space textures and sample them back. |
| [io](api/ordito/io.md) | Mesh file loading through the optional `meshio` dependency. |

### Arrays & infrastructure

| Module | Contents |
|---|---|
| [array](api/ordito/array.md) | NumPy-style structural and elementwise operations on Warp arrays, and sparse assembly. |
| [reduce](api/ordito/reduce.md) | Global reductions on Warp arrays: min, max, sum, mean, any, all, median, and per-group sums. |
| [grouping](api/ordito/grouping.md) | Grouping and deduplication of values and rows. |
| [graph](api/ordito/graph.md) | Graph algorithms on sparse adjacency matrices and edge lists. |
| [typing](api/ordito/typing.md) | Array type aliases and runtime rank checks. |
| [constants](api/ordito/constants.md) | Shared numeric tolerances and sentinel values. |
| [config](api/ordito/config.md) | Runtime settings: whether float reductions and scans repeat bit for bit on CUDA (`od.set_deterministic`, `with od.deterministic(False):`); see [Reproducible results](reproducibility.md). |
