"""
The ordito example gallery: one module per example, rendered into ``docs/assets/examples``.

Run ``python -m examples.run`` to execute every example, render its image and regenerate the
gallery pages under ``docs/examples``.
"""

GALLERY_INTRO = """
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
"""
