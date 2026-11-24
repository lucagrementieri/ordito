# Changelog

All notable changes to ordito are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and ordito adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a pre-1.0 caveat: while the
version is `0.x`, a **minor** bump may change a public signature, add or remove a public module,
or raise the `warp-lang` floor, and a **patch** bump is limited to bug fixes and documentation.

ordito is pre-1.0. The public API is safe to build on — it is covered by an extensive regression
suite and a nine-library parity gate — but until 1.0 a public signature may move a positional
argument to a keyword or gain a required parameter between minor versions. 1.0 will be the point
at which public function signatures become a commitment: module, name and positional argument
order stable, with change limited to additive keyword arguments.

Entries are written by hand rather than generated from commit messages: the git history is mostly
internal engineering (kernel-shape refactors, measurement rounds, review passes) that is not a
user-facing change, and a generated log would bury the handful of entries that are.

Every level-2 heading below is a released version, and `release.yml` cuts the GitHub Release body
from the one matching its tag — so prose sections do not belong between them.

## [Unreleased]

- Colours, UVs and textures on `Trimesh`, as in trimesh. A `"color"` vertex or face attribute
  (`wp.vec4`, RGBA in `[0, 1]`) reads as `vertex_colors` / `face_colors`, a `"uv"` vertex
  attribute (`wp.vec2`) as `uv`; both now blend through every edit like any float attribute. The
  new `texture.Material` (an RGBA image and a base-colour factor) is held as `Trimesh.material`,
  set with `with_material` and carried by every method returning a mesh; `+` keeps a material
  both operands share and warns and drops it otherwise. `Trimesh.to_color` and
  `texture.material_colors` read a material at UVs (bilinear, tiling, times the factor).
- `ordito.io` reads and writes through trimesh instead of meshio: the `io` extra installs
  `trimesh` and `Pillow`. The new `io.load_trimesh` returns a `Trimesh` with the file's colours,
  UVs and texture (OBJ materials, glTF base colours, several materials packed into one image),
  and `io.save_mesh` writes one, warning about any visual the format cannot hold. Breaking:
  `load_mesh_data` splits a UV seam's vertices and returns per-vertex `uv` instead of
  `corner_uv`, returns `colors` as `wp.vec4` always and face colours as `face_colors`, and no
  longer returns file normals (`vertex_normals`, `face_normals`), which trimesh does not keep;
  polygons are triangulated rather than skipped. VTK, Gmsh and the other volume formats are read
  only when `meshio` is installed as well.
- Python 3.15 is a supported version. Before it publishes, the release pipeline now installs the
  built wheel on every supported Python, 3.11 to 3.15, and runs it there.
- `reconstruction.delaunay_triangulation` triangulates repeated points. A copy of the
  lexicographically first point returned no faces, and a third copy of any point lost every point
  after it; coincident points now triangulate once, through their first copy.
- `array.block_diag` stacks a zero-row operator, which raised a `TypeError`.
- `boolean.union_all` given an empty mesh beside one clear of the origin returns Manifold's face
  order; the empty mesh's box was the origin point, where Manifold's holds nothing.
- `linalg.solve_spd_columns` and `linalg.spd_column_solver` reject a `float32` operator under any
  preconditioner but `"diag"` with a `ValueError` before solving. `"adaptive"` raised only once its
  Jacobi probe had stalled and overwritten `solution`, and `"auto"` failed with a `RuntimeError`.
- `sample.sample_surface_blue_noise` and `sample.sample_surface_poisson_disk` document the
  `ValueError` a zero-area mesh raises, which now says the mesh has zero surface area rather than
  naming a face weight the call never passed.
- `linalg.solve_spd` and `linalg.jacobi_preconditioner` are annotated to take an operator of any
  square block type, as they did and documented.

## [0.5.2] - 2026-11-23

- ordito can now run in one thread while PyTorch runs on its default stream in another. A graph
  recording used to break both: PyTorch raised `CUDA error: operation would make the legacy stream
  depend on a capturing blocking stream` and ordito `Warp CUDA error 901: operation failed due to
  a previous error during capture`. The recordings now happen on a private non-blocking stream.
  `linalg.solve_spd` when it does not batch, and `smoothing.filter_laplacian`'s implicit solve
  when its fixed-point step does not apply, still need the side stream described in
  [Sharing the GPU](https://lucagrementieri.github.io/ordito/streams/).
- That page now states that ordito must not be called from two threads at once, why, and the lock
  that makes a threaded process safe.

## [0.5.1] - 2026-11-23

### Added

- Documentation: a [Sharing the GPU](https://lucagrementieri.github.io/ordito/streams/) page,
  which states the stream ordito runs on, how to keep it in order with PyTorch, what to do when
  PyTorch runs in another thread, and how the two memory pools share the device.

### Fixed

- Every function that records a CUDA graph (solvers, smoothing, remeshing, hole filling, heat
  geodesics, `arap_parametrization` and more) now runs when Warp's current stream is CUDA's legacy
  default stream, as it is under `wp.ScopedStream(wp.stream_from_torch())` while PyTorch runs on
  its default stream. They raised `RuntimeError: Warp error: unknown stream` there.

## [0.5.0] - 2026-11-23

### Added

- `ordito.boolean`, exact Booleans of closed, consistently wound meshes on the GPU: `union`,
  `intersection` and `difference` of two solids, `union_all`, `intersection_all` and
  `difference_all` of many, `split_by_mesh` (both pieces of a solid cut by another), and
  `split_by_plane` / `trim_by_plane` (capped cuts by a plane). A port of the Manifold library's
  Boolean, the engine behind `trimesh.boolean` and `manifold3d`: exact `float32` predicates with
  one symbolic perturbation for touching and coplanar inputs, so every result is closed and
  manifold, and the same mesh Manifold returns, vertex and triangle order included.
  `collapse_collinear=False` keeps the crossing vertices an exact-arithmetic Boolean (libigl's,
  PyMeshLab's) keeps along straight creases. Each input is cleaned as Manifold cleans a mesh it
  builds a solid from (triangles naming a vertex twice dropped, pinched vertices split, edges
  shorter than the precision collapsed, degenerate triangles swapped, unused vertices dropped),
  so a raw scan gives Manifold's answer too; a clean mesh is left as it is.
- `levelset.function_level_set`: the closed solid where a `@wp.func` of position is positive,
  meshed from a body-centred cubic lattice with no grid to allocate (Manifold's `level_set`).
- `remesh.subdivide_smooth`: refinement onto a smooth surface through the vertices that keeps
  edges sharper than `min_sharp_angle` as creases and coplanar regions flat, by a count, an edge
  length or a tolerance (Manifold's `smooth_out` and `refine*`). `remesh.simplify`: the vertices
  that do not shape a closed mesh (inside flat regions, along straight edges) removed within a
  tolerance (Manifold's `simplify`).
- `polyline.triangulate_polygons` and `triangulate_polygons_from_offsets`: triangles of a planar
  region given as outer loops and holes, every triangle indexing the input points, robust to
  touching loops (Manifold's triangulator: the same triangles in the same order).
- `ray.segment_hits`: every crossing of each segment with a closed mesh, in order along it,
  decided by the Booleans' exact predicates, so a segment through an edge or a vertex crosses
  once there (Manifold's `ray_cast`, for many segments at once).
- Point-cloud preprocessing in `ordito.points`, each matching its Open3D, PCL or pytorch3d
  counterpart: `orient_normals` (minimum-spanning-tree propagation),
  `smooth_normals`, `smooth_points` (Laplacian or Taubin over a fixed neighbour table),
  `smooth_points_mls`, `smooth_points_bilateral`, `cluster_labels` (DBSCAN, Open3D's labels),
  `point_boundary_mask`, `point_visible_mask` (hidden point removal on the exact hull),
  `point_covariances` and `point_frames` (principal frames and spreads).
- `points.local_triangulation`: each point's star in the Delaunay triangulation of its neighbours
  in its tangent plane, as one triangle buffer (`potpourri3d.PointCloudLocalTriangulation`); it
  takes `frames=` to build the stars in given tangent frames.
- `heat.PointCloudHeatSolver`: the heat method on a bare point cloud (geometry-central's
  point-cloud solver): `distance`, `signed_distance` to curves through the cloud,
  `extend_scalar`, `transport_tangent_vectors` and `log_map`, on the tufted cover of the local
  triangulations, its factorizations kept for every later call.
- Documentation: a [Solids and Booleans](https://lucagrementieri.github.io/ordito/examples/solids/)
  gallery page, point-cloud and point-cloud-geodesic examples, a manifold3d migration guide, and
  the new modules and functions on the benchmark pages.

### Changed

- `remesh.flip_to_delaunay` flips on the local Delaunay test of a surface (the two angles
  opposite an edge sum to at most pi) instead of comparing circumcircles. Planar results are
  unchanged; on a curved mesh every edge a flip can reach now ends locally Delaunay, as
  `igl.is_intrinsic_delaunay` reads it, and the pass settles in fewer rounds. Edges whose flip
  would duplicate an existing edge (around a valence-3 vertex) are still left as they are.

### Fixed

- `equality.meshes_allclose` and `points_allclose` compare inputs that hold one place twice (an
  unwelded seam, a triangle soup, duplicate points): they returned `False` even for identical
  arrays, since every copy matched the same copy of the other side.

## [0.4.0] - 2026-11-20

### Added

- `parametrization.slim_parametrization`: a whole unwrap to the least symmetric Dirichlet
  distortion in one call, from a start it builds (`"free"`, `"harmonic"`, `"tutte"` or
  `"lscm"`); any topology from the free start, closed meshes included. `parametrization.slim`
  refines a chart you already have (scalable locally injective maps, as `igl::slim`): a face the
  start has the right way round never folds.
- `parametrization.low_distortion_parametrization`: a pin-free ARAP chart in the mesh's units for
  any topology (handles, several boundary loops, several components), and
  `parametrization.arap_parametrization`, ARAP iterated to convergence from a harmonic, Tutte or
  LSCM start on any connected surface with a boundary.
- `Trimesh` carries named per-vertex and per-face attributes (`vertex_attributes=`,
  `face_attributes=`, `with_attributes`); `Trimesh.with_edit` carries them through any edit called
  with `return_provenance=True`, and `submesh`, `split`, `+` and the transforms keep them.
  `Trimesh` prints a summary of its sizes, device, attributes and cached values.
- `Trimesh.with_vertices(refit=True)` refits the existing BVH for moved vertices instead of
  building a new one.
- `mask=` / `weights=` on `smoothing.filter_laplacian`, `filter_humphrey`, `filter_taubin`,
  `filter_neighborhood_average` and `filter_scalar_laplacian`: smooth only a region, or blend
  per vertex; a vertex of weight 0 stays in place.
- `texture.rasterize_face_indices`: the face covering every pixel of a window of UV space (a
  visibility buffer), with an optional per-face priority for overlaps.
- `ray.first_hits`: every ray's first hit densely (location, distance, face, barycentrics),
  optionally cast both ways.
- `proximity.signed_distance_on_mesh(sign_mode="pseudonormal")`: signs by the angle-weighted
  pseudonormal, as libigl's `PSEUDONORMAL`, so an open surface gets a meaningful sign; and
  `proximity.closest_point_on_mesh(return_barycentric=True)`.
- `registration.procrustes_from_offsets`: many weighted similarity fits over packed point sets in
  one launch, with optional Huber reweighting rounds.
- `registration.icp_point_to_plane(max_distance_factor=)`: rejects correspondences beyond a
  multiple of each round's median distance.
- `neighbors.query_ball_sum`: a per-query sum over each ball's neighbours (a value, or the
  neighbours' offsets along a direction), storing no pair list.
- `sample.sample_surface_in_ball`: uniform, Poisson-disk or blue-noise samples of the part of a
  surface inside a ball, with no submesh.
- `selection.vertex_hop_distance`: breadth-first hop counts from a vertex mask.
- `polyline.polyline_resample(interpolation="catmull_rom")` (the centripetal curve) and
  `polyline.polyline_upsample_count`.
- `nonnegative_weights=` on `laplacian.cotmatrix_entries`, `laplacian.cotmatrix` and
  `parametrization.harmonic`: clamps obtuse angles' cotangent weights at zero.
- `cholesky.factorization_nbytes`: the peak memory a factorization would take, from the analysis
  alone; `linalg.OperatorFactorization.exceeds_budget` and `iterate_rejected`.
- `cholesky.SparseCholesky.solve(refine=False)`: one triangular solve with no refinement rounds,
  for a caller that iterates to its own fixpoint.

### Changed

- **Breaking:** the boolean `region` argument of `smoothing.relax_keep_volume`, `relax_approx`
  and `equalize_triangle_areas` is renamed `mask` (same position; a keyword call must be renamed).
- A settled heat diffusion too large to factor within the memory budget is now solved band by
  band outward from its source, every entry to its own precision, instead of returning a
  far field that failed its accuracy check: `heat.heat_geodesic` and `log_map` are accurate on
  very large meshes, and the 0.3.0 warning for that case is gone.
- `parametrization.arap` charts move by rounding (within `3e-7`); it now shares the new charts'
  ARAP pieces.
- Faster sparse Cholesky solves on the CPU device (the whole triangular solve in one launch), and
  so the heat, parametrization and hole-fill calls built on them.

### Fixed

- `polyline.polyline_upsample` no longer overflows `int32` on a very fine step (the CPU returned
  a two-point answer, CUDA a shape error); a non-positive or non-finite `step_size` raises.
- `reduce.quantile` and `median` (and every sort of float keys) rank NaN above `+inf` on CUDA as
  on the CPU, where CUDA ranked it below `-inf`.

## [0.3.1] - 2026-11-17

### Fixed

- `reconstruction.screened_poisson` (`method="octree"`, and `method="adaptive"` with a band) no
  longer fails with CUDA error 700 on Blackwell GPUs of compute capability 12.0 (RTX 50 series,
  RTX PRO 6000) under Warp's CUDA 12 build (`warp-lang` `+cu12`), whose compiler mis-builds the
  iso-value sum for that architecture. Other GPUs, and Warp's CUDA 13 build, were not affected.
- `sparse_cholesky` (and the dense hole-fill solve) no longer fails to build its kernels when a
  CUDA 12 and a CUDA 13 build of Warp share one kernel cache: it clears the stale linked objects
  and rebuilds once, with a warning.
- Three kernels read a value before setting it (an unrolled loop index, a module constant):
  release builds happened to give the right answer, debug builds (`wp.config.mode = "debug"`)
  did not — `screened_poisson(method="octree")` read out of bounds there, and the sparse Cholesky
  panel updates and exact group sums read garbage bounds. Results of release builds are unchanged.
- The `warp-lang` requirement is now `>=1.19`. 0.3.0 declared `>=1.18` but uses Warp 1.19
  features and does not build on 1.18; this corrects the metadata rather than dropping a
  supported version.

## [0.3.0] - 2026-11-16

### Added

- `interpolation.Provenance`: where each output vertex and face of a topology edit came from, so
  vertex and face attributes (UVs, colours, labels, per-corner values) carry through it.
  `return_provenance=True` returns one from every function that rebuilds connectivity — the
  subdivisions, decimations, `isotropic_remesh` and the region refiners, `split_edges`, the
  `selection.submesh_*` family, `delete_region_keep_boundary`, the face-dropping and
  vertex-merging repairs, `make_solid`, `fix_degeneracies`, `fix_self_intersections`, the hole
  fills, extensions, bridges, stitches and joins, `seams.cut_along_edges`, `bounds.crop_mesh`,
  `combine.split`, and the `intersection` splits, slices and clips. Edits chain with
  `Provenance.then`; `vertex_values` / `face_values` carry an attribute through. Built from
  `Provenance.from_rows`, `from_edge_points`, `from_clusters` and `identity`, over the new
  `interpolation.transfer_through_index`, `operator_from_index`, `operator_representatives` and
  `compose_operators`.
- `interpolation.interpolate_barycentric`, and `sample.sample_surface(return_barycentric=True)`
  to read a vertex field at the sampled points.
- `repair.split_vertices_by_corner_values`: unweld vertices whose corners carry different values
  (UV seams); `repair.remove_duplicated_vertices(vertex_values=)` welds only vertices whose values
  also agree.
- `io.load_mesh_data` reads OBJ files itself and returns per-corner UVs (`corner_uv`); PLY
  texture-coordinate face lists are read too.
- `precompile`: `python -m ordito.precompile` (or `od.precompile.precompile`) builds every kernel
  ordito launches into Warp's kernel cache ahead of time, for any CUDA architecture without a GPU
  (`--arch`) and for the CPU, so a first call compiles nothing.
- `levelset.marching_cubes(method="openvdb")`: OpenVDB's `volumeToMesh` at adaptivity 0 (dual
  marching cubes), equal to MeshLib's `gridToMesh` vertex for vertex and face for face.
- `levelset.offset_mesh(method=)`: `"meshlab"` is MeshLab's offset; `"meshlib"` the exact signed
  distance on OpenVDB's lattice, extracted as above.
- `levelset.signed_distance_level_set` takes `wp.vec3d` bounds for a `float64` lattice, and
  classic `marching_cubes` a `float64` field: `resample_uniform(method="igl")` now matches
  `igl.offset_surface` to ~2e-7.
- `method="meshlib"` on `holes.fill_smooth`, `refill_region`, `stitch_smooth`,
  `bridge_edges_smooth` and `smoothing.smooth_region_boundary`: MeshLib's `fillHoleNicely`,
  `patchMesh`, `stitchHolesNicely`, `makeSmoothBridge` and `smoothRegionBoundary`.
- `heat.heat_geodesic(method="potpourri3d")`: geometry-central's heat method with its intrinsic
  tufted cover; `geodesic_walk.geodesic_path(method="potpourri3d")`: geometry-central's edge-flip
  geodesics (the default stays `"heat"`).
- Method aliases named for the library a default already reproduces:
  `filter_laplacian(method="trimesh")`, `filter_humphrey(method="trimesh")`,
  `icp(method="trimesh" | "open3d" | "pytorch3d")`, `marching_cubes(method="igl" | "pyvista")` and
  `principal_curvature(method="igl")`.
- `voxels.space_filling_codes` / `cells_from_space_filling_codes` (Hilbert and Morton codes of
  integer cells) and `points.space_filling_order`.
- `reduce.scatter_min` / `scatter_max` / `scatter_mean` and `segment_min` / `segment_max` /
  `segment_mean`.
- `neighbors.mutual_nearest`: reciprocal nearest-neighbour pairs between two clouds.
- `equality.triangles_allclose`: two meshes' triangles equal up to face order and rotation, with
  no vertex matching.
- `laplacian.cotmatrix_entries_and_pattern`; `array.gather(out=)`;
  `halfedge.halfedge_twins(defect_counts=)`; `graph.shortest_path_envelope(order=)`;
  `linalg.OperatorFactorization.refused`.

### Changed

- **Breaking:** the `vertices` module is now `normals` (`vertex_normals`, `mean_vertex_normals`,
  `weighted_vertex_normals`), and `vertex_defects` moved to `curvature`.
- **Breaking:** `return_index`, `return_operator`, `return_inverse` and `split_edges(index=)` are
  replaced by `return_provenance` (on `quadric_decimate`, `subdivide_to_size`, `subdivide_loop`,
  `split_edges`, `submesh_from_face_indices` / `_mask` and `remove_unreferenced_vertices`).
- **Breaking:** `halfedge.key_ordered_halfedges` and the `key_order` keyword of
  `boundary.oriented_boundary_edges` are removed. Boundary, oriented-boundary, region-boundary
  and crease edge rows come out in halfedge order, `resolve_duplicated_faces(method="cancel")`
  keeps faces in input order, and `cluster_decimate` numbers cells by arrival (by first
  occurrence under the deterministic setting). These outputs are sets: compare them through
  `ordito.equality`.
- Vertex normals (and everything built on them) repeat bit for bit on CUDA under the deterministic
  setting.
- The texture rasterizer gives the same images on CPU and CUDA.
- `smoothing.filter_normals(method="meshlib")` (the Mumford-Shah denoiser) follows MeshLib 3.2;
  `heat.fast_marching_distance(method="meshlib")` follows MeshLib 3.2's surface-distance front;
  `registration.procrustes` weights its cross-covariance as trimesh 5.2.1 does;
  `creation.parametric_surface("klein")` is the closed Klein bottle, as pyvista 0.49 builds it.
- A settled heat diffusion that fails its accuracy check and cannot be factored within the memory
  budget now warns that its far field is unreliable, instead of returning silently.
- Faster across the library — host overhead, launch counts, hub vertices of very high valence,
  tree searches on scrambled point orders, and the CPU device — and tested on `warp-lang` 1.19.

### Fixed

- `heat` `method="meshlab"` / `"igl"`: the default diffusion time is floored on large meshes, where
  the far-field distance stopped growing.
- `smoothing.smooth_region_fixed_rim` no longer collapses a free component that touches no fixed
  vertex.
- `intersection.clip_mesh_with_field(cap=True)`: sections passing a point twice, and faces the
  cut collapses, are capped correctly.
- Closest points on slivers (`icp(method="pyvista")`, the `float64` level-set band,
  `max_tangent_sphere`) no longer jump to a face's first corner.
- `linalg` multigrid no longer crashes on a fragmented free set, and carries the near-nullspace
  through its levels (fewer CG iterations on deep hierarchies).
- `energies.hessian_energy` on fans of more than 21 faces, and `smoothing.smooth_region` past the
  `int32` range, no longer write out of bounds.
- `remesh.refine_region_to_density` is reproducible on CUDA under the deterministic setting.
- `boundary.boundary_loops(pinches="gap")` keeps dead-end walks in their fan.

## [0.2.0] - 2026-11-03

### Added

- `heat.fast_marching_distance`: geodesic distance by a front marched across the faces, as
  MeshLib's `computeSurfaceDistances` computes it with no cap on re-opened vertices, solved in
  parallel rounds. `method="meshlab"` is MeshLab's geodesic front instead (vcg's face unfolding,
  in `float64`), equal to `compute_scalar_by_geodesic_distance_from_given_point_per_vertex` from one
  source.
- `curvature.vertex_curvatures`: pointwise mean and Gaussian curvature (Meyer et al.), as MeshLab's
  `compute_scalar_by_discrete_curvature_per_vertex` forms them.
- `curvature.principal_curvature(method="pyvista")`: VTK's 1-ring extreme curvatures, equal to
  `pyvista.PolyData.curvature("maximum")` / `("minimum")`.
- `curvature.principal_curvature(method="meyer")`: the principal pair implied by Meyer et al.'s
  pointwise mean and Gaussian curvature.
- `smoothing.filter_windowed_sinc`: the windowed-sinc low-pass filter of Taubin, Zhang and Golub,
  equal to `pyvista.PolyData.smooth_taubin` (VTK's `vtkWindowedSincPolyDataFilter`).
- `repair.resolve_duplicated_faces(method=)`: `"igl"` drops a group no cancellation resolves, as
  libigl does; `"open3d"` and `"meshlab"` keep the first face of each duplicate group, as
  `remove_duplicated_triangles` and `meshing_remove_duplicate_faces` do. The default still raises.
- `reconstruction.resample_uniform(method="igl")`: `igl.offset_surface`'s lattice.
- `reconstruction.resample_uniform(method="meshlab")`: MeshLab's uniform resampling, equal to
  `generate_resampled_uniform_mesh` mesh for mesh on an ordinary mesh: its lattice, its
  point-face distance and sign rule, and Lewiner's topologically controlled marching cubes
  (MC33). Where faces wound against each other coincide (duplicated or seamed faces), a few
  vertices around the affected lattice nodes can differ.
- `levelset.marching_cubes(method="lewiner")`: Lewiner et al.'s topologically controlled marching
  cubes (MC33), its ambiguous faces and cell interiors resolved by the field, on a `float32` or
  `float64` lattice.
- `array.bit_ranks`: the running set-bit count of a `uint32` bitmask's words, the rank table of
  a flag stored as one bit (`out=` writes it into a caller's buffer).
- `texture.rasterize_attribute` / `rasterize_discrete_attribute(subpixel_bits=)`: snap vertices
  to a sub-pixel grid as a GPU rasterizer does; `8` reproduces OpenGL.

### Changed

- `smoothing.filter_two_step(method="meshlab")` runs in `float64`, MeshLab's precision.
- `halfedge.key_ordered_halfedges` takes its flags as a bitmask and the bitmask's word ranks
  (`words, word_ranks`, from `array.bit_ranks`), a bit a halfedge and each class row padded to
  whole words, in place of an `int32` inclusive scan: a 32nd of the memory.
- `reconstruction.screened_poisson(method="octree")` runs to depth 10, matching open3d's
  reconstruction there, and is faster and leaner at every depth: its sweeps visit only bricks
  holding an active node, its index tables are rank-packed bitsets, and it shares its solve buffers
  with the dense and adaptive methods.
- Lower peak memory across `reconstruction`, `voxels`, `levelset`, `visibility`, `intersection`,
  `linalg` and `smoothing`; `triangulate_point_cloud` and `ball_pivoting` size their buffers for
  what a run holds.
- `energies.hessian_energy` / `curved_hessian_energy` build their sparsity pattern first, with far
  lower peak memory.
- On the CPU device, iterative loops (CG, Poisson V-cycle, fan flood, ear clipping and others) run
  as one launch, with identical results.

### Fixed

- `energies.hessian_energy` / `curved_hessian_energy` refuse a triplet count that overflows `int32`
  instead of faulting on very large meshes.
- `reconstruction.ball_pivoting` seeding reads only the failures of earlier waves.
- `sample.sample_surface_poisson_disk` is reproducible under the deterministic setting.
- The lazy submodule import no longer keeps the calling frame's arrays alive.
- `heat`: the vector heat solvers and `heat_signed_distance(method="potpourri3d")` lengthen their
  default time on fine meshes.
- `cholesky`: a solve into a caller's buffer no longer replays a stale recording.

## [0.1.0] - 2026-10-27

The first public release.

### Added

- GPU geometry processing on NVIDIA Warp: 52 public modules covering mesh creation, topology and
  validation, measures and curvature, repair, remeshing and smoothing, spatial queries and
  distances, point clouds and reconstruction, registration, Laplacians and sparse solvers
  (CG, multigrid, Cholesky), parametrization, geodesics and heat methods, and polylines.
- An optional `Trimesh` object API with cached derived quantities, over the same functions.
- Opt-in bit-reproducible results on CUDA (`ordito.config.set_deterministic`).
- Inline type information (`py.typed`).

### Notes

- Requires Python 3.11+ and `warp-lang` 1.18+; the only other runtime dependency is `numpy`.
  Mesh I/O needs the optional `io` extra (meshio).
- Runs on CUDA when available and on Warp's CPU backend otherwise; see
  [Platform support](https://lucagrementieri.github.io/ordito/getting-started/#platform-support).

[Unreleased]: https://github.com/lucagrementieri/ordito/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/lucagrementieri/ordito/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/lucagrementieri/ordito/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/lucagrementieri/ordito/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/lucagrementieri/ordito/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/lucagrementieri/ordito/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/lucagrementieri/ordito/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lucagrementieri/ordito/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucagrementieri/ordito/releases/tag/v0.1.0
