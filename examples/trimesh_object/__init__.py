"""Gallery page: the trimesh object."""

TITLE = "The Trimesh object"
ORDER = 1
INTRO = """
[`ordito.Trimesh`][ordito.mesh.Trimesh] bundles a mesh's vertex and face buffers with lazily
computed, cached properties, in the style of `trimesh.Trimesh`. Every property runs on the
mesh's device, and each is computed once and reused until the mesh changes.
"""
