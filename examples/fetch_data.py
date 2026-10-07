"""
Download the Stanford scans the examples use into ``benchmarks/data/``.

Usage::

    python -m examples.fetch_data

The meshes are from the Stanford 3D Scanning Repository
(https://graphics.stanford.edu/data/3Dscanrep/); please credit it when you reuse them. Files that
are already present are skipped.
"""

from __future__ import annotations

import gzip
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "benchmarks" / "data"
BASE = "http://graphics.stanford.edu/pub/3Dscanrep"

# (target filename, archive URL, member inside a tar archive or None for a gzip-compressed file)
SOURCES = [
    ("bunny.ply", f"{BASE}/bunny.tar.gz", "bunny/reconstruction/bun_zipper.ply"),
    ("dragon.ply", f"{BASE}/dragon/dragon_recon.tar.gz", "dragon_recon/dragon_vrip.ply"),
    ("happy_buddha.ply", f"{BASE}/happy/happy_recon.tar.gz", "happy_recon/happy_vrip.ply"),
    ("Armadillo.ply", f"{BASE}/armadillo/Armadillo.ply.gz", None),
]


def fetch(filename: str, url: str, member: str | None) -> None:
    """Download one scan unless it is already in ``DEST``."""
    target = DEST / filename
    if target.exists():
        print(f"{filename}: present")
        return
    print(f"{filename}: downloading {url}")
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    DEST.mkdir(parents=True, exist_ok=True)
    if member is None:
        target.write_bytes(gzip.decompress(payload))
        return
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"{member} not in {url}")
        with target.open("wb") as out:
            shutil.copyfileobj(source, out)


def main() -> None:
    for filename, url, member in SOURCES:
        fetch(filename, url, member)


if __name__ == "__main__":
    main()
