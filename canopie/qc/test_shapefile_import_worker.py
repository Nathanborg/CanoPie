"""
End-to-end coverage for ShapefileImportWorker.

WHY THIS FILE EXISTS
--------------------
Nothing in the suite drove the import worker end to end. `test_shapefile_import_perf.py`
does static source checks and exercises `match_feature_geometry` directly;
`test_shapefile_import_render.py` checks what happens to already-imported
records. So a speed-up attempt could add a call with the wrong number of
arguments, and every test still passed:

    spatial_index.py   def intersects_project_bbox(self, bbox_wgs84)     # 1 arg
    polygon_manager.py     intersects_project_bbox(feat_bbox, crs)       # 2 args

That only escaped being a hard failure by accident -- `decompose_shapefile_features`
silently dropped the `bbox` key, so the guard `if feat_bbox and ...` short-circuited
and the whole "optimisation" was dead code. Fixing the arity WITHOUT noticing that
would have turned a no-op into a crash on the first feature of every import.

These tests run the real worker against a real georeferenced GeoTIFF and a real
shapefile, so the wiring itself is under test.

WHAT THEY PIN
-------------
* the worker completes and emits `finished`, never `error`
* the bbox pre-filter compares in the SHAPEFILE's CRS (the feature bbox is in
  source coordinates, e.g. UTM metres; the project extent is WGS84 degrees --
  comparing them directly matches nothing and would drop everything)
* it FAILS OPEN: no `.prj`, or no project extent, means "do not filter", because
  discarding data must require positive evidence
* discarded features are COUNTED and reported, never silently lost
* caches are keyed on content, not `id()` -- CPython reuses addresses, and these
  caches are module-level and never cleared
"""
import json
import os
import struct

import numpy as np
import pytest
from PyQt5 import QtWidgets

pytestmark = [pytest.mark.io, pytest.mark.polygons]

# A small UTM 17N patch (Panama), matching the real project's CRS.
EPSG = 32617
ORIGIN_X, ORIGIN_Y = 620000.0, 1010000.0
PIX = 0.10
W = H = 512


@pytest.fixture
def geo_project(qapp, tmp_path):
    """A georeferenced GeoTIFF plus the target_images_info the worker expects."""
    import tifffile

    from ..shapefile_io import wkt_for_epsg

    img = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)
    tif = tmp_path / "ortho.tif"
    tifffile.imwrite(
        str(tif), img, photometric='rgb',
        extratags=[
            (33922, 'd', 6, (0.0, 0.0, 0.0, ORIGIN_X, ORIGIN_Y, 0.0), True),
            (33550, 'd', 3, (PIX, PIX, 0.0), True),
            (34735, 'H', 12, (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, EPSG), True),
        ])
    return {
        'tif': str(tif),
        'info': [{'filepath': str(tif), 'ref_size': {'w': W, 'h': H}, 'ax_data': None}],
        'wkt': wkt_for_epsg(EPSG),
    }


def _write_shp(tmp_path, name, rings, crs_wkt):
    """Write a polygon shapefile; omit the .prj when crs_wkt is None."""
    from ..shapefile_io import write_shapefile

    feats = [{'geometry': r,
              'properties': {'name': f'f{i}', 'group': 'trees', 'type': 'polygon'}}
             for i, r in enumerate(rings)]
    stem = str(tmp_path / name)
    write_shapefile(feats, stem, crs_wkt=crs_wkt, shape_type=5)
    if crs_wkt is None:
        prj = stem + ".prj"
        if os.path.exists(prj):
            os.remove(prj)
    return stem + ".shp"


def _ring(cx, cy, r=2.0):
    return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r),
            (cx - r, cy + r), (cx - r, cy - r)]


def _inside(n=12):
    """Rings well inside the raster's footprint.

    The raster is only W*PIX = 51.2 m across, so the spacing has to keep every
    ring inside it -- otherwise the off-image extent check legitimately drops
    the last few and the test looks like a code bug.
    """
    span = W * PIX
    step = (span - 12.0) / max(n, 1)
    return [_ring(ORIGIN_X + 6.0 + i * step, ORIGIN_Y - 6.0 - i * step) for i in range(n)]


def _far_away(n=8):
    """Rings ~200 km east -- same CRS, nowhere near the image."""
    return [_ring(ORIGIN_X + 200_000 + i * 5, ORIGIN_Y - 20 - i * 3) for i in range(n)]


def _run(shp_paths, info, tmp_path):
    """Drive the real worker synchronously; return (data, warnings, errors)."""
    from ..polygon_manager import ShapefileImportWorker

    w = ShapefileImportWorker(file_paths=shp_paths, project_folder=str(tmp_path),
                              target_images_info=info)
    out, errs, prog = {}, [], []
    w.finished.connect(lambda d, warns: out.update(data=d, warns=warns))
    w.error.connect(lambda e: errs.append(e))
    w.progress.connect(lambda p, m: prog.append((p, m)))
    w.run()
    return out.get('data'), out.get('warns'), errs, prog


# ---------------------------------------------------------------------------
# THE regression: the worker must actually run
# ---------------------------------------------------------------------------
def test_worker_imports_overlapping_features(geo_project, tmp_path):
    shp = _write_shp(tmp_path, "inside", _inside(12), geo_project['wkt'])
    data, warns, errs, prog = _run([shp], geo_project['info'], tmp_path)

    assert errs == [], (
        f"the import worker raised instead of completing: {errs}. This is the "
        "class of failure an arity/wiring mistake in the per-feature loop "
        "produces, and nothing else in the suite exercises it.")
    assert data, "finished fired with no polygons"
    assert len(data) == 12, f"expected 12 groups, got {len(data)}"
    assert len(prog) >= 2, "no progress was emitted"


def test_imported_geometry_lands_inside_the_image(geo_project, tmp_path):
    shp = _write_shp(tmp_path, "inside", _inside(6), geo_project['wkt'])
    data, _, errs, _ = _run([shp], geo_project['info'], tmp_path)
    assert errs == []

    for group, fmap in data.items():
        for fp, rec in fmap.items():
            pts = rec['points']
            assert pts, f"{group}: no points"
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            assert 0 <= min(xs) and max(xs) <= W, f"{group}: x out of range {min(xs)}..{max(xs)}"
            assert 0 <= min(ys) and max(ys) <= H, f"{group}: y out of range {min(ys)}..{max(ys)}"
            assert rec['coord_space'] == 'image'


def test_imported_records_are_json_serialisable(geo_project, tmp_path):
    """Coordinates must be plain floats -- the reader builds them from a numpy
    buffer, and mixed scalar types are a trap for anything downstream."""
    shp = _write_shp(tmp_path, "inside", _inside(4), geo_project['wkt'])
    data, _, errs, _ = _run([shp], geo_project['info'], tmp_path)
    assert errs == []

    blob = json.dumps(data)          # must not raise
    assert json.loads(blob)
    for fmap in data.values():
        for rec in fmap.values():
            for x, y in rec['points']:
                assert type(x) is float and type(y) is float, (
                    f"coordinate types are {type(x).__name__}/{type(y).__name__}, "
                    "not plain float")


# ---------------------------------------------------------------------------
# The bbox pre-filter
# ---------------------------------------------------------------------------
def test_features_outside_the_project_are_skipped_and_reported(geo_project, tmp_path):
    """THE behaviour the user chose: drop, but never silently."""
    shp = _write_shp(tmp_path, "far", _far_away(8), geo_project['wkt'])
    data, warns, errs, _ = _run([shp], geo_project['info'], tmp_path)

    assert errs == []
    assert not data, f"features 200 km away were imported: {len(data)} group(s)"
    assert warns, "features were discarded with NO warning -- silent data loss"
    joined = " ".join(warns)
    assert "not imported" in joined and "8" in joined, (
        f"the summary does not say how many features were dropped: {warns}")


def test_filter_compares_in_the_shapefile_crs(geo_project, tmp_path):
    """A feature bbox is in SOURCE coordinates (UTM metres here) while the
    project extent is WGS84 degrees. Comparing them directly matches nothing,
    so a CRS-blind filter would discard even overlapping features."""
    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    mgr.build_index(geo_project['info'])
    assert mgr.project_bbox_wgs84, "no project extent was computed"
    assert max(abs(v) for v in mgr.project_bbox_wgs84) < 400, (
        "project extent is not in degrees")

    assert mgr.prepare_bbox_filter(geo_project['wkt']) is True
    fb = mgr._filter_bbox
    assert fb[0] > 1000, (
        f"the filter bbox {fb} is still in degrees -- it was never projected "
        "into the shapefile's CRS, so UTM feature bboxes can never match it")

    inside_bbox = (ORIGIN_X + 10, ORIGIN_Y - 30, ORIGIN_X + 20, ORIGIN_Y - 20)
    far_bbox = (ORIGIN_X + 200_000, ORIGIN_Y - 30, ORIGIN_X + 200_010, ORIGIN_Y - 20)
    assert mgr.bbox_outside_project(inside_bbox) is False
    assert mgr.bbox_outside_project(far_bbox) is True


def test_filter_fails_open_without_a_prj(geo_project, tmp_path):
    """No .prj means unknown coordinates. Import everything rather than guess."""
    shp = _write_shp(tmp_path, "noprj", _inside(5), None)
    assert not os.path.exists(shp[:-4] + ".prj")

    data, _, errs, _ = _run([shp], geo_project['info'], tmp_path)
    assert errs == []
    assert data and len(data) == 5, (
        f"a shapefile with no .prj imported {0 if not data else len(data)}/5 "
        "features -- the pre-filter must switch itself off when the CRS is "
        "unknown, not discard the data")


def test_filter_fails_open_with_no_project_extent():
    """No footprints -> no extent -> filter must not reject anything."""
    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    mgr.build_index([])
    assert mgr.prepare_bbox_filter("EPSG:32617") is False
    assert mgr.bbox_outside_project((0, 0, 1, 1)) is False, (
        "with no known project extent the filter rejected a feature -- that "
        "would discard every feature in the file")


def test_malformed_bbox_is_not_a_reason_to_drop():
    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    mgr._filter_bbox = (0.0, 0.0, 10.0, 10.0)
    for bad in (None, (), (1, 2), ("a", "b", "c", "d")):
        assert mgr.bbox_outside_project(bad) is False, f"{bad!r} caused a drop"


# ---------------------------------------------------------------------------
# Cache keys must be content-derived, not id()
# ---------------------------------------------------------------------------
def test_ax_cache_token_is_content_based():
    """`id()` is unique only among LIVE objects; these caches are module-level
    and never cleared, so a recycled address silently applies the wrong
    transform."""
    from ..shapefile_io import _ax_cache_token

    a = {'crop_enabled': True, 'crop_rect': {'x': 1, 'y': 2, 'width': 3, 'height': 4}}
    b = dict(a)
    c = {'crop_enabled': True, 'crop_rect': {'x': 9, 'y': 2, 'width': 3, 'height': 4}}

    assert _ax_cache_token(a) == _ax_cache_token(b), (
        "two equal .ax dicts produced different cache tokens -- the cache can "
        "never hit, so every call rebuilds the transform")
    assert _ax_cache_token(a) != _ax_cache_token(c), (
        "different crops share a cache token -- polygons would be placed with "
        "the wrong transform")
    assert _ax_cache_token(None) == _ax_cache_token(None)


def test_crs_cache_token_is_content_based():
    pyproj = pytest.importorskip("pyproj")
    from ..shapefile_io import _crs_cache_token

    a = pyproj.CRS.from_epsg(32617)
    b = pyproj.CRS.from_epsg(32617)
    c = pyproj.CRS.from_epsg(32618)

    assert a is not b
    assert _crs_cache_token(a) == _crs_cache_token(b), (
        "two separately-parsed but identical CRSs produced different tokens. "
        "pyproj CRS objects expose __dict__, so an id()-based key took this "
        "branch every time and the transformer cache MISSED on every call -- "
        "a pessimisation, not a cache")
    assert _crs_cache_token(a) != _crs_cache_token(c)


# ---------------------------------------------------------------------------
# The accidental quadratic
# ---------------------------------------------------------------------------
def test_identity_resolution_does_not_copy_the_key_set():
    """`existing_keys` is the caller's accumulating set, passed once per
    feature. Copying it made identity resolution O(features^2): 0.293 s of pure
    set-copying at 6000 features, ~12 s at 40k."""
    import inspect
    import textwrap

    from ..shapefile_io import resolve_feature_identity

    src = textwrap.dedent(inspect.getsource(resolve_feature_identity))
    assert "set(existing_keys) if existing_keys else set()" not in src, (
        "resolve_feature_identity still copies the caller's key set on every "
        "call, making import quadratic in feature count")

    # and behaviourally: the caller's set must be the one consulted
    keys = {"trees_f0"}
    out = resolve_feature_identity({'name': 'f0', 'group': 'trees'}, 1, "shp", keys)
    assert out['entry_key'] != "trees_f0", "collision with an existing key not avoided"
