"""
Run a documentation example the way a reader would, and hold its printed values to the page.

Shared by the two example gates: ``tests/test_api_conventions.py`` runs every fenced ``python``
block in a docstring, and ``tests/test_docs_examples.py`` every block of a hand-written page. Both
run a block statement by statement, so that a top-level ``print(...)`` whose line ends in a comment
can be checked against it:

```python
print(mesh.is_watertight)  # True
```

asserts that the call printed exactly ``True``. That comment is the documented output; without the
check it is the first thing to go stale. A ``print`` with no trailing comment is not checked, and
neither is one inside a loop or a function body, nor one printing more than one line. A printed
float is shown rounded (``f"{area:.4f}"``), because the last digits of a reduction differ between
CPU and CUDA and between CUDA runs.
"""

from __future__ import annotations

import ast
import contextlib
import io
import tokenize

import warp as wp


class ExampleOutputMismatchError(AssertionError):
    """A ``print`` in an example printed something other than its trailing comment says."""


def _trailing_comments(code: str) -> dict[int, str]:
    """Map a 1-based line number to the text of the comment ending that line, ``#`` stripped."""
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string.removeprefix("#").strip()
    return comments


def _is_print(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "print"
    )


def run_example(code: str, site: str, namespace: dict[str, object]) -> None:
    """
    Run ``code`` in ``namespace``, checking each commented top-level ``print`` against its comment.

    Raises
    ------
    ExampleOutputMismatchError
        If a checked ``print`` printed something other than its trailing comment.
    """
    tree = ast.parse(code, filename=site)
    comments = _trailing_comments(code)
    # Warp reports each module load on stdout at its default ``LOG_INFO``; a reader sees those
    # lines too, but they are not the example's output.
    log_level = wp.config.log_level
    wp.config.log_level = wp.LOG_WARNING
    try:
        _run_statements(tree, comments, site, namespace)
    finally:
        wp.config.log_level = log_level


def _run_statements(
    tree: ast.Module, comments: dict[int, str], site: str, namespace: dict[str, object]
) -> None:
    for statement in tree.body:
        compiled = compile(ast.Module(body=[statement], type_ignores=[]), site, "exec")
        expected = comments.get(statement.end_lineno or statement.lineno)
        if not _is_print(statement) or expected is None:
            exec(compiled, namespace)
            continue
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compiled, namespace)
        printed = captured.getvalue().rstrip("\n")
        if printed != expected:
            line = statement.lineno
            raise ExampleOutputMismatchError(
                f"{site} (line {line} of the block) prints {printed!r}, "
                f"but its comment documents {expected!r}"
            )
