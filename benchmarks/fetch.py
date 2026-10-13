"""
Download the scan meshes of [`meshes.py`](meshes.py) into ``DATA_DIR``, verifying each by SHA-256.

The harness calls [`ensure_mesh`][] the first time a case reads a scan mesh, so a run filtered by
``--size`` downloads only what it uses. Run this file to prefetch ahead of a benchmark run (a CI
cache step, a machine about to go offline):

```bash
uv run python benchmarks/fetch.py              # every scan mesh
uv run python benchmarks/fetch.py bunny dragon # just these
```

A file whose hash does not match is an error, never a skip, both for a download and for a file
already on disk: a benchmark row that silently disappears, or silently times different bytes,
is the failure this module exists to prevent.
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from meshes import DATA_DIR, MESH_RELEASE_URL, MESHES, MESHES_BY_NAME

_CHUNK = 1 << 20
_ATTEMPTS = 4


@functools.cache
def ensure_mesh(name: str) -> Path:
    """
    Return the local path of scan mesh ``name``, downloading it first if it is missing.

    The file on disk is hashed once per process, so a corrupted or replaced file fails here rather
    than benchmarking different bytes.

    Raises
    ------
    RuntimeError
        If the downloaded or cached file does not match the recorded SHA-256, or the download
        fails after retrying.
    """
    spec = MESHES_BY_NAME[name]
    path = DATA_DIR / spec["filename"]
    if not path.exists():
        _download(f"{MESH_RELEASE_URL}/{spec['filename']}", path, spec["sha256"])
    else:
        digest = _sha256(path)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{path} has SHA-256 {digest}, expected {spec['sha256']}; delete it to download "
                "the pinned file again."
            )
    return path


def _download(url: str, path: Path, sha256: str) -> None:
    """Stream ``url`` into a temporary file beside ``path``, check its hash, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_ATTEMPTS):
        print(f"Downloading {url}", file=sys.stderr)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".part")
        try:
            digest = hashlib.sha256()
            with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=60) as src:
                while chunk := src.read(_CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise RuntimeError(
                    f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}: the release "
                    "asset changed, which a published meshes-vN release must never do."
                )
            # Atomic on one filesystem, so concurrent workers never read a partial file.
            Path(temporary).replace(path)
            return
        except OSError as error:
            Path(temporary).unlink(missing_ok=True)
            if attempt == _ATTEMPTS - 1:
                raise RuntimeError(f"could not download {url}: {error}") from error
            time.sleep(2 ** (attempt + 1))
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    names = sys.argv[1:] or [spec["name"] for spec in MESHES]
    unknown = sorted(set(names) - set(MESHES_BY_NAME))
    if unknown:
        sys.exit(f"unknown scan meshes {unknown}; choose from {list(MESHES_BY_NAME)}")
    for mesh_name in names:
        print(ensure_mesh(mesh_name))
