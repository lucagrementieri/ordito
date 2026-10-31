"""Gallery page: the trimesh object."""

TITLE = "The Trimesh object"
HEAD_TITLE = "The Trimesh object: cached mesh properties on the GPU"
DESCRIPTION = (
    "Use ordito's trimesh-style Trimesh class: lazily cached normals, adjacency, boundaries, "
    "watertightness, transforms and reusable operators, all on the GPU."
)
ORDER = 1
INTRO = """
[`ordito.Trimesh`][ordito.mesh.Trimesh] bundles a mesh's vertex and face buffers with lazily
computed, cached properties, in the style of `trimesh.Trimesh`. Every property runs on the
mesh's device, and each is computed once and reused until the mesh changes.
"""
