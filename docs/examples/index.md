# Examples

Each example below is a complete, runnable snippet together with the image it produced. The
inputs are shared across examples so that related operations can be compared on the same shape:
most use the Stanford bunny, and the rest use the dragon, the happy Buddha, or meshes built with
[`ordito.creation`][ordito.creation].

`data` in the snippets is the gallery's input module
([`examples/data.py`](https://github.com/lucagrementieri/ordito/blob/main/examples/data.py)). It
loads a mesh into ordito's layout: `wp.vec3` vertices and a flat `wp.int32` face buffer. The code
runs on whichever Warp `device` you pass, CPU or CUDA. The images come from
`python -m examples.run`, which renders them with PyVista.

Each example also links the reference-library examples it was modelled on. The images are ordito's
own output.

## [The Trimesh object](trimesh-object.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Quick start: build a mesh and read its properties](../assets/examples/t1-thumb.webp)](trimesh-object.md#t1)

    [Quick start: build a mesh and read its properties](trimesh-object.md#t1)

-   [![Bodies: split, explode and recombine](../assets/examples/t2-thumb.webp)](trimesh-object.md#t2)

    [Bodies: split, explode and recombine](trimesh-object.md#t2)

-   [![Rigid and mirror transforms](../assets/examples/t3-thumb.webp)](trimesh-object.md#t3)

    [Rigid and mirror transforms](trimesh-object.md#t3)

-   [![Inside tests and surface samples](../assets/examples/t4-thumb.webp)](trimesh-object.md#t4)

    [Inside tests and surface samples](trimesh-object.md#t4)

-   [![Edge structure: dihedral angles, convexity and boundaries](../assets/examples/t5-thumb.webp)](trimesh-object.md#t5)

    [Edge structure: dihedral angles, convexity and boundaries](trimesh-object.md#t5)

-   [![Caching and functional updates](../assets/examples/t6-thumb.webp)](trimesh-object.md#t6)

    [Caching and functional updates](trimesh-object.md#t6)

-   [![Reusing cached operators](../assets/examples/t7-thumb.webp)](trimesh-object.md#t7)

    [Reusing cached operators](trimesh-object.md#t7)

</div>

## [Creating meshes](creation.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Primitive gallery](../assets/examples/c1-thumb.webp)](creation.md#c1)

    [Primitive gallery](creation.md#c1)

-   [![Parametric surfaces](../assets/examples/c2-thumb.webp)](creation.md#c2)

    [Parametric surfaces](creation.md#c2)

-   [![Revolve, extrude and sweep](../assets/examples/c3-thumb.webp)](creation.md#c3)

    [Revolve, extrude and sweep](creation.md#c3)

-   [![Polygon triangulation](../assets/examples/c4-thumb.webp)](creation.md#c4)

    [Polygon triangulation](creation.md#c4)

-   [![Terrain triangulation](../assets/examples/c5-thumb.webp)](creation.md#c5)

    [Terrain triangulation](creation.md#c5)

</div>

## [Inspecting meshes](inspect.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Validation masks: boundaries, non-manifold elements, orientation](../assets/examples/i1-thumb.webp)](inspect.md#i1)

    [Validation masks: boundaries, non-manifold elements, orientation](inspect.md#i1)

-   [![Bounding boxes, principal axes and a fitted plane](../assets/examples/i2-thumb.webp)](inspect.md#i2)

    [Bounding boxes, principal axes and a fitted plane](inspect.md#i2)

-   [![Connected components: label, split, remove debris](../assets/examples/i3-thumb.webp)](inspect.md#i3)

    [Connected components: label, split, remove debris](inspect.md#i3)

-   [![Feature edges: creases, convexity and boundaries](../assets/examples/i4-thumb.webp)](inspect.md#i4)

    [Feature edges: creases, convexity and boundaries](inspect.md#i4)

-   [![Triangle quality](../assets/examples/i5-thumb.webp)](inspect.md#i5)

    [Triangle quality](inspect.md#i5)

-   [![Normals: per face, per vertex, per corner](../assets/examples/i6-thumb.webp)](inspect.md#i6)

    [Normals: per face, per vertex, per corner](inspect.md#i6)

-   [![Homology generators: handles and tunnels](../assets/examples/i7-thumb.webp)](inspect.md#i7)

    [Homology generators: handles and tunnels](inspect.md#i7)

-   [![Self-intersections and mesh-mesh collisions](../assets/examples/i8-thumb.webp)](inspect.md#i8)

    [Self-intersections and mesh-mesh collisions](inspect.md#i8)

</div>

## [Curvature and differential operators](differential.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Gaussian and mean curvature](../assets/examples/d1-thumb.webp)](differential.md#d1)

    [Gaussian and mean curvature](differential.md#d1)

-   [![Principal curvatures and directions](../assets/examples/d2-thumb.webp)](differential.md#d2)

    [Principal curvatures and directions](differential.md#d2)

-   [![Gradient of a scalar field](../assets/examples/d3-thumb.webp)](differential.md#d3)

    [Gradient of a scalar field](differential.md#d3)

-   [![Laplace equation with Dirichlet conditions](../assets/examples/d4-thumb.webp)](differential.md#d4)

    [Laplace equation with Dirichlet conditions](differential.md#d4)

-   [![Polyharmonic surfaces (k = 1, 2, 3)](../assets/examples/d5-thumb.webp)](differential.md#d5)

    [Polyharmonic surfaces (k = 1, 2, 3)](differential.md#d5)

-   [![Handle-based biharmonic deformation](../assets/examples/d6-thumb.webp)](differential.md#d6)

    [Handle-based biharmonic deformation](differential.md#d6)

-   [![Isolines of a scalar field](../assets/examples/d7-thumb.webp)](differential.md#d7)

    [Isolines of a scalar field](differential.md#d7)

-   [![Smoothing a noisy scalar field](../assets/examples/d8-thumb.webp)](differential.md#d8)

    [Smoothing a noisy scalar field](differential.md#d8)

-   [![Ambient occlusion, obscurance, shape diameter and thickness](../assets/examples/d9-thumb.webp)](differential.md#d9)

    [Ambient occlusion, obscurance, shape diameter and thickness](differential.md#d9)

-   [![Maximal inscribed spheres and the medial axis](../assets/examples/d10-thumb.webp)](differential.md#d10)

    [Maximal inscribed spheres and the medial axis](differential.md#d10)

</div>

## [Geodesics and tangent fields](geodesics.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Geodesic distance with the heat method](../assets/examples/g1-thumb.webp)](geodesics.md#g1)

    [Geodesic distance with the heat method](geodesics.md#g1)

-   [![Shortest paths: along the edges and across the faces](../assets/examples/g2-thumb.webp)](geodesics.md#g2)

    [Shortest paths: along the edges and across the faces](geodesics.md#g2)

-   [![Tracing straightest geodesics (the exponential map)](../assets/examples/g3-thumb.webp)](geodesics.md#g3)

    [Tracing straightest geodesics (the exponential map)](geodesics.md#g3)

-   [![Signed distance to curves on the surface](../assets/examples/g4-thumb.webp)](geodesics.md#g4)

    [Signed distance to curves on the surface](geodesics.md#g4)

-   [![Extending values from a few points (geodesic Voronoi cells)](../assets/examples/g5-thumb.webp)](geodesics.md#g5)

    [Extending values from a few points (geodesic Voronoi cells)](geodesics.md#g5)

-   [![Parallel transport and the logarithmic map](../assets/examples/g6-thumb.webp)](geodesics.md#g6)

    [Parallel transport and the logarithmic map](geodesics.md#g6)

-   [![Smoothing a tangent vector field](../assets/examples/g7-thumb.webp)](geodesics.md#g7)

    [Smoothing a tangent vector field](geodesics.md#g7)

-   [![Shortening non-contractible loops](../assets/examples/g8-thumb.webp)](geodesics.md#g8)

    [Shortening non-contractible loops](geodesics.md#g8)

</div>

## [Parametrization and textures](parametrization.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Flattening a disk: Tutte, harmonic, LSCM and ARAP](../assets/examples/p1-thumb.webp)](parametrization.md#p1)

    [Flattening a disk: Tutte, harmonic, LSCM and ARAP](parametrization.md#p1)

-   [![Cutting a closed surface open along seams](../assets/examples/p2-thumb.webp)](parametrization.md#p2)

    [Cutting a closed surface open along seams](parametrization.md#p2)

-   [![Baking a field into a texture and reading it back](../assets/examples/p3-thumb.webp)](parametrization.md#p3)

    [Baking a field into a texture and reading it back](parametrization.md#p3)

</div>

## [Smoothing and fairing](smoothing.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Denoising a scan: four smoothing filters](../assets/examples/s1-thumb.webp)](smoothing.md#s1)

    [Denoising a scan: four smoothing filters](smoothing.md#s1)

-   [![Smoothing without shrinking](../assets/examples/s2-thumb.webp)](smoothing.md#s2)

    [Smoothing without shrinking](smoothing.md#s2)

-   [![Mean-curvature flow](../assets/examples/s3-thumb.webp)](smoothing.md#s3)

    [Mean-curvature flow](smoothing.md#s3)

-   [![Removing spikes and sharpening detail](../assets/examples/s4-thumb.webp)](smoothing.md#s4)

    [Removing spikes and sharpening detail](smoothing.md#s4)

-   [![Fairing a region and smoothing its rim](../assets/examples/s5-thumb.webp)](smoothing.md#s5)

    [Fairing a region and smoothing its rim](smoothing.md#s5)

-   [![Relaxation: even triangle areas and a local surface fit](../assets/examples/s6-thumb.webp)](smoothing.md#s6)

    [Relaxation: even triangle areas and a local surface fit](smoothing.md#s6)

</div>

## [Remeshing and simplification](remesh.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Quadric decimation](../assets/examples/r1-thumb.webp)](remesh.md#r1)

    [Quadric decimation](remesh.md#r1)

-   [![Vertex clustering](../assets/examples/r2-thumb.webp)](remesh.md#r2)

    [Vertex clustering](remesh.md#r2)

-   [![Subdivision: midpoint and Loop](../assets/examples/r3-thumb.webp)](remesh.md#r3)

    [Subdivision: midpoint and Loop](remesh.md#r3)

-   [![Isotropic remeshing](../assets/examples/r4-thumb.webp)](remesh.md#r4)

    [Isotropic remeshing](remesh.md#r4)

-   [![Refining to a size or to the surrounding density](../assets/examples/r5-thumb.webp)](remesh.md#r5)

    [Refining to a size or to the surrounding density](remesh.md#r5)

-   [![Delaunay flips and the intrinsic Delaunay triangulation](../assets/examples/r6-thumb.webp)](remesh.md#r6)

    [Delaunay flips and the intrinsic Delaunay triangulation](remesh.md#r6)

</div>

## [Repairing meshes](repair.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Filling holes: fan, minimum weight and smooth](../assets/examples/h1-thumb.webp)](repair.md#h1)

    [Filling holes: fan, minimum weight and smooth](repair.md#h1)

-   [![Filling only the small holes](../assets/examples/h2-thumb.webp)](repair.md#h2)

    [Filling only the small holes](repair.md#h2)

-   [![Stitching two boundaries](../assets/examples/h3-thumb.webp)](repair.md#h3)

    [Stitching two boundaries](repair.md#h3)

-   [![Joining nearby open components](../assets/examples/h4-thumb.webp)](repair.md#h4)

    [Joining nearby open components](repair.md#h4)

-   [![Degenerate triangles, duplicate vertices and T-vertices](../assets/examples/h5-thumb.webp)](repair.md#h5)

    [Degenerate triangles, duplicate vertices and T-vertices](repair.md#h5)

-   [![Consistent and outward orientation](../assets/examples/h6-thumb.webp)](repair.md#h6)

    [Consistent and outward orientation](repair.md#h6)

-   [![Non-manifold repair](../assets/examples/h7-thumb.webp)](repair.md#h7)

    [Non-manifold repair](repair.md#h7)

-   [![Fixing self-intersections](../assets/examples/h8-thumb.webp)](repair.md#h8)

    [Fixing self-intersections](repair.md#h8)

-   [![Removing tunnels](../assets/examples/h9-thumb.webp)](repair.md#h9)

    [Removing tunnels](repair.md#h9)

-   [![One-call watertight solid](../assets/examples/h10-thumb.webp)](repair.md#h10)

    [One-call watertight solid](repair.md#h10)

</div>

## [Offsets, voxels and implicit surfaces](volumes.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Offset, thicken and shell](../assets/examples/v1-thumb.webp)](volumes.md#v1)

    [Offset, thicken and shell](volumes.md#v1)

-   [![Signed distance field and its level sets](../assets/examples/v2-thumb.webp)](volumes.md#v2)

    [Signed distance field and its level sets](volumes.md#v2)

-   [![Meshing an implicit surface: the gyroid](../assets/examples/v3-thumb.webp)](volumes.md#v3)

    [Meshing an implicit surface: the gyroid](volumes.md#v3)

-   [![Voxelizing meshes and point clouds](../assets/examples/v4-thumb.webp)](volumes.md#v4)

    [Voxelizing meshes and point clouds](volumes.md#v4)

-   [![Voxel morphology](../assets/examples/v5-thumb.webp)](volumes.md#v5)

    [Voxel morphology](volumes.md#v5)

-   [![Voxel CSG (approximate booleans)](../assets/examples/v6-thumb.webp)](volumes.md#v6)

    [Voxel CSG (approximate booleans)](volumes.md#v6)

-   [![Generalized winding number](../assets/examples/v7-thumb.webp)](volumes.md#v7)

    [Generalized winding number](volumes.md#v7)

</div>

## [Spatial queries](queries.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Ray casting a depth and a normal image](../assets/examples/q1-thumb.webp)](queries.md#q1)

    [Ray casting a depth and a normal image](queries.md#q1)

-   [![Rays against a mesh: hits and misses](../assets/examples/q2-thumb.webp)](queries.md#q2)

    [Rays against a mesh: hits and misses](queries.md#q2)

-   [![Closest points on a surface](../assets/examples/q3-thumb.webp)](queries.md#q3)

    [Closest points on a surface](queries.md#q3)

-   [![Distance between two surfaces](../assets/examples/q4-thumb.webp)](queries.md#q4)

    [Distance between two surfaces](queries.md#q4)

-   [![Inside / outside classification](../assets/examples/q5-thumb.webp)](queries.md#q5)

    [Inside / outside classification](queries.md#q5)

-   [![Nearest neighbours and ball queries](../assets/examples/q6-thumb.webp)](queries.md#q6)

    [Nearest neighbours and ball queries](queries.md#q6)

-   [![Chamfer and Hausdorff distances](../assets/examples/q7-thumb.webp)](queries.md#q7)

    [Chamfer and Hausdorff distances](queries.md#q7)

</div>

## [Cutting, slicing and selection](cutting.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Plane sections and slice stacks](../assets/examples/x1-thumb.webp)](cutting.md#x1)

    [Plane sections and slice stacks](cutting.md#x1)

-   [![Clipping and capping](../assets/examples/x2-thumb.webp)](cutting.md#x2)

    [Clipping and capping](cutting.md#x2)

-   [![Clipping with a scalar field](../assets/examples/x3-thumb.webp)](cutting.md#x3)

    [Clipping with a scalar field](cutting.md#x3)

-   [![Cropping to a box](../assets/examples/x4-thumb.webp)](cutting.md#x4)

    [Cropping to a box](cutting.md#x4)

-   [![Growing and shrinking selections](../assets/examples/x5-thumb.webp)](cutting.md#x5)

    [Growing and shrinking selections](cutting.md#x5)

-   [![Polyline processing](../assets/examples/x6-thumb.webp)](cutting.md#x6)

    [Polyline processing](cutting.md#x6)

</div>

## [Point clouds](points.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Down-sampling: voxel grid, farthest point and blue noise](../assets/examples/pc1-thumb.webp)](points.md#pc1)

    [Down-sampling: voxel grid, farthest point and blue noise](points.md#pc1)

-   [![Surface sampling: uniform, Poisson disk and blue noise](../assets/examples/pc2-thumb.webp)](points.md#pc2)

    [Surface sampling: uniform, Poisson disk and blue noise](points.md#pc2)

-   [![Normal estimation](../assets/examples/pc3-thumb.webp)](points.md#pc3)

    [Normal estimation](points.md#pc3)

-   [![Outlier removal: statistical, radius and probabilistic](../assets/examples/pc4-thumb.webp)](points.md#pc4)

    [Outlier removal: statistical, radius and probabilistic](points.md#pc4)

-   [![Cloud-to-cloud distance](../assets/examples/pc5-thumb.webp)](points.md#pc5)

    [Cloud-to-cloud distance](points.md#pc5)

-   [![Interpolating sparse samples onto a surface](../assets/examples/pc6-thumb.webp)](points.md#pc6)

    [Interpolating sparse samples onto a surface](points.md#pc6)

-   [![Convex-hull points and half-space tests](../assets/examples/pc7-thumb.webp)](points.md#pc7)

    [Convex-hull points and half-space tests](points.md#pc7)

</div>

## [Reconstruction and registration](reconstruction.md)

<div class="grid cards ordito-gallery" markdown>

-   [![Screened Poisson reconstruction](../assets/examples/rr1-thumb.webp)](reconstruction.md#rr1)

    [Screened Poisson reconstruction](reconstruction.md#rr1)

-   [![Ball pivoting](../assets/examples/rr2-thumb.webp)](reconstruction.md#rr2)

    [Ball pivoting](reconstruction.md#rr2)

-   [![Local triangulation and uniform resampling](../assets/examples/rr3-thumb.webp)](reconstruction.md#rr3)

    [Local triangulation and uniform resampling](reconstruction.md#rr3)

-   [![Rigid ICP: point-to-point vs point-to-plane](../assets/examples/rr4-thumb.webp)](reconstruction.md#rr4)

    [Rigid ICP: point-to-point vs point-to-plane](reconstruction.md#rr4)

-   [![Robust ICP with outliers](../assets/examples/rr5-thumb.webp)](reconstruction.md#rr5)

    [Robust ICP with outliers](reconstruction.md#rr5)

-   [![Known correspondences: Procrustes alignment](../assets/examples/rr6-thumb.webp)](reconstruction.md#rr6)

    [Known correspondences: Procrustes alignment](reconstruction.md#rr6)

</div>
