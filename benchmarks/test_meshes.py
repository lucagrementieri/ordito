"""
Invariant checks for the synthetic feature meshes in [`meshes.py`](meshes.py).

Not benchmarks -- these are the guard rails that keep the benchmarks *meaningful*. Every feature
mesh exists to perturb one property of the ``sphere_med`` control while pinning the others, and
a group's timing spread is only attributable if that pinning actually holds. A silent change in
``trimesh.creation`` (a different section count, a cap that stops being a fan, an icosphere that
starts merging vertices) would leave every benchmark still running and still green while quietly
comparing two meshes that differ in more than one way.

So the recorded ``MeshSpec`` counts are checked against the built geometry, the topology each
mesh is chosen for is asserted directly, and the cross-mesh pinning invariants -- equal ``F``
across the component axis, equal ``V`` across the diameter and valence axes, identical
connectivity across the quality axis -- are checked as their own tests.

The scan meshes are downloaded rather than built, so what is checked for them is the contract
that makes a download reproducible: every one is pinned by a SHA-256, and [`fetch.py`](fetch.py)
refuses bytes that do not match it (exercised offline through ``file://`` URLs).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import fetch
import numpy as np
import numpy.typing as npt
import pytest
import trimesh as tm
from meshes import AXES, BUILDERS, FEATURE_MESHES, FEATURE_MESHES_BY_NAME, MESHES, MeshSpec

_mesh_cache: dict[str, tm.Trimesh] = {}

_NAMES = [spec["name"] for spec in FEATURE_MESHES]

# The topology each mesh is chosen for: (bodies, watertight, boundary loops, max vertex valence).
# Reading down the valence column shows the design: everything is a regular 6 except the two-row
# ribbon and the open tubes (3, from their degree-limited grids) and ``fan_hub``, which is the
# whole point of that mesh.
_TOPOLOGY: dict[str, tuple[int, bool, int, int]] = {
    "sphere_small": (1, True, 0, 6),
    "sphere_med": (1, True, 0, 6),
    "sphere_large": (1, True, 0, 6),
    "parts_64": (64, True, 0, 6),
    "parts_1024": (1024, True, 0, 6),
    # Open counterparts of the two above: one face dropped per shell, so every component
    # carries exactly one 3-edge rim and the joining functions have something to bridge.
    "open_parts_4": (4, False, 4, 6),
    "open_parts_16": (16, False, 16, 6),
    "open_parts_64": (64, False, 64, 6),
    "ribbon_long": (1, False, 1, 3),
    "fan_hub": (1, True, 0, 40_960),
    "rim_long": (1, False, 2, 3),
    "holes_many": (1, False, 512, 6),
    "holes_dense": (1, False, 8_192, 6),
    "rim_short": (1, False, 2, 3),
    "saddle_small": (1, False, 1, 6),
    "saddle": (1, False, 1, 6),
    "saddle_graded": (1, False, 1, 6),
    "hemisphere": (1, False, 1, 6),
    "shells_8": (8, True, 0, 6),
    "tangle_2": (2, True, 0, 6),
    # The self-intersection is geometric, not topological: both tori are one closed watertight
    # component with regular valence, and only the *embedding* crosses itself. That is exactly
    # what distinguishes them from ``tangle_2``, whose two bodies several repair references
    # decline rather than handle.
    "tangle_torus_small": (1, True, 0, 6),
    "tangle_torus": (1, True, 0, 6),
    # The two genus meshes come out of a boolean, so their valence peaks are whatever the
    # retriangulation around a tunnel produced -- recorded rather than designed, unlike the rest.
    "handles_1": (1, True, 0, 16),
    "handles_64": (1, True, 0, 14),
}


def _built(name: str) -> tm.Trimesh:
    """Build a feature mesh once per session, without trimesh's vertex-merging preprocessing."""
    if name not in _mesh_cache:
        vertices_np, faces_np = BUILDERS[name]()
        _mesh_cache[name] = tm.Trimesh(vertices_np, faces_np, process=False)
    return _mesh_cache[name]


def _max_valence(mesh_tm: tm.Trimesh) -> int:
    return int(np.bincount(mesh_tm.faces.reshape(-1), minlength=len(mesh_tm.vertices)).max())


def _n_loops(mesh_tm: tm.Trimesh) -> int:
    return 0 if mesh_tm.is_watertight else len(mesh_tm.outline().entities)


def _worst_aspect_ratio(mesh_tm: tm.Trimesh) -> float:
    """Longest-over-shortest edge of the worst triangle."""
    triangles_np = mesh_tm.vertices[mesh_tm.faces]
    edges_np = np.linalg.norm(np.roll(triangles_np, -1, axis=1) - triangles_np, axis=2)
    return float((edges_np.max(axis=1) / edges_np.min(axis=1)).max())


@pytest.mark.parametrize("spec", FEATURE_MESHES, ids=_NAMES)
def test_recorded_counts_match(spec: MeshSpec) -> None:
    """``MeshSpec`` records counts so a mesh can be described without building it -- verify them."""
    mesh_tm = _built(spec["name"])
    assert len(mesh_tm.vertices) == spec["n_vertices"]
    assert len(mesh_tm.faces) == spec["n_faces"]
    assert mesh_tm.faces.min() >= 0
    assert len(mesh_tm.faces) > 0, (
        "a zero-triangle mesh corrupts CUDA state when wrapped in wp.Mesh"
    )


@pytest.mark.parametrize("spec", FEATURE_MESHES, ids=_NAMES)
def test_topology_is_what_the_axis_claims(spec: MeshSpec) -> None:
    """Components, watertightness, boundary loop count and peak valence, per ``_TOPOLOGY``."""
    mesh_tm = _built(spec["name"])
    bodies, watertight, loops, valence = _TOPOLOGY[spec["name"]]
    assert mesh_tm.body_count == bodies
    assert mesh_tm.is_watertight == watertight
    assert _n_loops(mesh_tm) == loops
    assert _max_valence(mesh_tm) == valence


def test_component_axis_pins_face_count() -> None:
    """1 -> 64 -> 1024 components with the face count held at the control's 81 920."""
    counts = {name: len(_built(name).faces) for name in AXES["components"]}
    assert len(set(counts.values())) == 1, counts
    assert [_built(name).body_count for name in AXES["components"]] == [1, 64, 1024]


def test_diameter_and_valence_axes_pin_vertex_count() -> None:
    """``ribbon_long`` and ``fan_hub`` match the control vertex for vertex."""
    control = len(_built("sphere_med").vertices)
    assert len(_built("ribbon_long").vertices) == control
    assert len(_built("fan_hub").vertices) == control
    # fan_hub additionally matches on faces, so valence is the only difference at all.
    assert len(_built("fan_hub").faces) == len(_built("sphere_med").faces)


def test_genus_axis_pins_the_face_count_and_varies_only_the_genus() -> None:
    """
    0 -> 1 -> 64 handles at ~90 000 faces, on one footprint at one tessellation scale.

    The face counts cannot be pinned *exactly* -- a boolean decides how many triangles a tunnel
    costs -- so what is asserted is that they stay within 15 % of each other, which is what keeps a
    timing spread across this axis attributable to the handles rather than to size. The genus
    itself comes from the Euler characteristic, and both ends must be closed for it to mean
    anything.
    """
    handles_1, handles_64 = _built("handles_1"), _built("handles_64")
    assert handles_1.is_watertight
    assert handles_64.is_watertight
    assert (2 - handles_1.euler_number) // 2 == 1
    assert (2 - handles_64.euler_number) // 2 == 64
    counts = [len(_built(name).faces) for name in AXES["genus"]]
    assert max(counts) / min(counts) < 1.15, counts


def test_quality_axis_pins_connectivity() -> None:
    """The graded saddle is the same mesh re-spaced: identical faces, far worse triangles."""
    saddle, graded = _built("saddle"), _built("saddle_graded")
    assert np.array_equal(saddle.faces, graded.faces)
    assert _worst_aspect_ratio(saddle) < 2.0
    assert _worst_aspect_ratio(graded) > 1_000.0


def test_loop_axes_trade_length_against_count() -> None:
    """Few long loops against many short ones, at a comparable total boundary length."""
    long_rim, many = _built("rim_long"), _built("holes_many")
    assert _n_loops(long_rim) == 2
    assert _n_loops(many) == 512
    assert all(len(entity.points) - 1 == 3 for entity in many.outline().entities)
    # Within 2x on total boundary edges, so loop count is what differs rather than sheer size.
    total_long, total_many = len(long_rim.outline().entities), len(many.outline().entities)
    assert total_long < total_many


def test_axes_reference_known_meshes() -> None:
    """Every ``AXES`` entry names real feature meshes, control first."""
    for axis, names in AXES.items():
        assert len(names) >= 2, f"{axis} is not a comparison"
        unknown = [name for name in names if name not in FEATURE_MESHES_BY_NAME]
        assert not unknown, f"{axis} references unknown meshes: {unknown}"


def test_every_feature_mesh_is_reachable_from_an_axis() -> None:
    """No orphan meshes: a mesh nothing selects is a mesh nobody maintains."""
    used = {name for names in AXES.values() for name in names}
    assert set(_NAMES) == used, f"unused: {sorted(set(_NAMES) - used)}"


def test_scan_and_feature_names_are_disjoint() -> None:
    """``ALL_MESHES_BY_NAME`` merges the two registries, so a collision would shadow silently."""
    from meshes import MESHES_BY_NAME

    assert not set(MESHES_BY_NAME) & set(FEATURE_MESHES_BY_NAME)


@pytest.mark.parametrize("spec", FEATURE_MESHES, ids=_NAMES)
def test_vertices_are_finite(spec: MeshSpec) -> None:
    """Guards the graded spacing and the slice-plane cut, which are the two easiest to break."""
    vertices_np: npt.NDArray[np.floating] = _built(spec["name"]).vertices
    assert np.isfinite(vertices_np).all()


def test_scan_meshes_are_pinned() -> None:
    """Every scan mesh names a distinct release asset and a SHA-256; no feature mesh does."""
    assert all(re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) for spec in MESHES)
    assert len({spec["filename"] for spec in MESHES}) == len(MESHES)
    assert all(spec["filename"] == spec["sha256"] == "" for spec in FEATURE_MESHES)


@pytest.fixture
def fake_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    """Serve ``bunny.ply`` from a ``file://`` release into an empty cache; yield both folders."""
    release = tmp_path / "release"
    release.mkdir()
    content = b"ply\nformat ascii 1.0\nend_header\n"
    (release / "bunny.ply").write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    cache = tmp_path / "cache"
    monkeypatch.setattr(fetch, "DATA_DIR", cache)
    monkeypatch.setattr(fetch, "MESH_RELEASE_URL", release.as_uri())
    monkeypatch.setitem(
        fetch.MESHES_BY_NAME, "bunny", {**fetch.MESHES_BY_NAME["bunny"], "sha256": sha256}
    )
    fetch.ensure_mesh.cache_clear()
    yield release, cache
    fetch.ensure_mesh.cache_clear()


def test_ensure_mesh_downloads_a_missing_mesh(fake_release: tuple[Path, Path]) -> None:
    """A missing file is downloaded into the cache, with no partial file left beside it."""
    release, cache = fake_release
    path = fetch.ensure_mesh("bunny")
    assert path == cache / "bunny.ply"
    assert path.read_bytes() == (release / "bunny.ply").read_bytes()
    assert sorted(item.name for item in cache.iterdir()) == ["bunny.ply"]


def test_ensure_mesh_rejects_a_changed_release_asset(fake_release: tuple[Path, Path]) -> None:
    """A download whose bytes differ from the pin raises and leaves nothing in the cache."""
    release, cache = fake_release
    (release / "bunny.ply").write_bytes(b"something else")
    with pytest.raises(RuntimeError, match="SHA-256"):
        fetch.ensure_mesh("bunny")
    assert list(cache.iterdir()) == []


def test_ensure_mesh_rejects_a_corrupted_cached_file(fake_release: tuple[Path, Path]) -> None:
    """A cached file is re-hashed, so a corrupted copy raises instead of being benchmarked."""
    _, cache = fake_release
    cache.mkdir()
    (cache / "bunny.ply").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="delete it"):
        fetch.ensure_mesh("bunny")
