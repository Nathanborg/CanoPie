"""
Polygon sidecars are written ONCE per save, through the C JSON encoder (BUG-030).

THE BUG THIS PINS
-----------------
Importing the 2,801-crown BCI shapefile into C:\\Nova pasta (6) froze the app
for 105 s, 93 s of it on the GUI thread. cProfile: 88 s inside the pure-Python
JSON encoder (encoder.py _iterencode*), because

  * every sidecar writer used ``json.dump(data, f, indent=...)`` -- json.dump
    streams through the pure-Python encoder, and indent disables the C one
    anyway; and
  * every file was written TWICE: the import's deferred save_incremental wrote
    all 2,801 (64 s) and cleared only ITS dirty set; 200 ms later the debounced
    _flush_dirty_polygons found the same 2,801 in ITS set and wrote them again
    (24 s) -- dumping the raw in-memory record, a different format from the
    first write.

Simplifying the shapefile in QGIS "helped" only because it cut the number of
floats to encode. The fix keeps the full geometry and makes writing cheap.
"""
import builtins
import json
import os

import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.io, pytest.mark.perf]

N_POLYGONS = 40
N_VERTICES = 120          # above polygon_lod.MIN_POINTS_FOR_LOD


def _ring(cx, cy, r, n):
    import math
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


@pytest.fixture
def dirty_crowns(synthetic_project, monkeypatch):
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    pdir = os.path.join(pt.project_folder, "polygons")
    os.makedirs(pdir, exist_ok=True)
    groups = [f"sidecarw_{i}" for i in range(N_POLYGONS)]
    for i, g in enumerate(groups):
        pt.all_polygons[g] = {fp: {
            'name': str(1000 + i), 'type': 'polygon', 'coord_space': 'image',
            'image_ref_size': {'w': 64, 'h': 64}, 'root': '',
            'points': _ring(32.0, 32.0, 5.0 + 0.5 * i, N_VERTICES),
            'properties': {'tag': str(1000 + i), 'species': 'oleifera'},
        }}
        pt._mark_polygon_dirty(g, fp)
    t = getattr(pt, "_save_dirty_timer", None)
    if t is not None:
        t.stop()                       # the test drives the flush itself
    base = os.path.splitext(os.path.basename(fp))[0]
    paths = {os.path.normcase(os.path.join(pdir, f"{g}_{base}_polygons.json")): g for g in groups}
    try:
        yield pt, fp, paths
    finally:
        for g in groups:
            pt.all_polygons.pop(g, None)
            for s in ("_dirty_polys", "_dirty_polygon_entries"):
                getattr(pt, s, set()).discard((g, fp))
        for p in paths:
            for q in (p, p + ".tmp"):
                try:
                    os.remove(q)
                except OSError:
                    pass
        pt._poly_norm_index_invalid = True


def _count_sidecar_writes(monkeypatch, paths):
    from .. import project_tab as _pt_mod
    from .. import polygon_lod as _lod_mod
    writes = {p: 0 for p in paths}
    real_open = builtins.open

    def counting_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax"):
            key = os.path.normcase(os.fspath(file))
            if key.endswith(".tmp"):
                key = key[:-4]
            if key in writes:
                writes[key] += 1
        return real_open(file, mode, *a, **k)

    # Module-level `open` shadows the builtin for that module only.
    monkeypatch.setattr(_pt_mod, "open", counting_open, raising=False)
    monkeypatch.setattr(_lod_mod, "open", counting_open, raising=False)
    return writes


def test_each_dirty_sidecar_is_written_exactly_once(dirty_crowns, monkeypatch):
    pt, fp, paths = dirty_crowns
    writes = _count_sidecar_writes(monkeypatch, paths)

    pt.save_incremental(show_status=False, update_project_json=False)
    pt._flush_dirty_polygons()          # the debounced flush that used to re-write everything

    twice = sorted(paths[p] for p, n in writes.items() if n > 1)
    never = sorted(paths[p] for p, n in writes.items() if n == 0)
    assert not never, f"{len(never)} dirty polygon(s) were never saved: {never[:3]}"
    assert not twice, (
        f"{len(twice)} of {len(paths)} sidecars were written more than once in one save "
        "(save_incremental, then _flush_dirty_polygons again 200 ms later) -- an import "
        "of 2,801 crowns paid for every file twice.")

    for p, g in paths.items():
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        assert d.get('properties', {}).get('species') == 'oleifera', f"{g}: properties lost"
        # One format from both writers. The display pyramid is derived on the fly
        # now; storing it cost 41% of every sidecar and 2.5 s of every import.
        assert 'lod' not in d and 'display_points' not in d and 'group' not in d, (
            f"{g}: sidecar keys {sorted(d)} -- a writer dumped the raw in-memory record "
            "instead of the canonical payload")
        assert len(d.get('points') or []) == N_VERTICES


def test_sidecars_are_encoded_by_the_c_encoder(dirty_crowns, monkeypatch):
    pt, fp, paths = dirty_crowns
    import json.encoder as enc
    calls = {"n": 0}
    real = enc._make_iterencode

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(enc, "_make_iterencode", spy)
    pt.save_incremental(show_status=False, update_project_json=False)
    pt._flush_dirty_polygons()
    assert calls["n"] == 0, (
        f"the pure-Python JSON encoder ran {calls['n']} time(s) while saving {len(paths)} "
        "sidecars. json.dump(...) and any indent= force it; on the 2,801-crown import it "
        "was 88 of the 93 seconds the GUI thread was frozen. Use polygon_lod.dump_sidecar.")
