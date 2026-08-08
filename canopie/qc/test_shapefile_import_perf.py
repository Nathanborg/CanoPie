"""
Shapefile import must be O(features), not O(features x PROJ pipeline build).

THE BUG THIS PINS (reported as "shapefile importing is just too slow to load
in canopie and it lags the program when the shape has many features -- it
should be nearly instant like QGIS", plus "the loading bar is not working
properly"):

1. PER-FEATURE TRANSFORMER CONSTRUCTION.
   `SpatialIndexManager.match_feature_geometry` built its source-CRS ->
   EPSG:4326 hop inline:

       src_crs    = pyproj.CRS.from_user_input(shapefile_crs)
       transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", ...)

   Both ran once PER FEATURE, even though every feature in a shapefile shares
   one CRS. Building a PROJ transformation pipeline is genuinely expensive --
   it hits the PROJ database and searches candidate operations/grids.

   Profiling the REAL worker on 2000 synthetic features measured:

       total 29.08 s, of which 26.09 s (90%) inside Transformer.__init__

   i.e. ~13 ms per feature spent reconstructing an object identical to the
   previous one. After caching: 2.81 s total (10.4x faster). At 6000 features
   the same import went from an extrapolated ~87 s to 4.04 s.

2. PROGRESS EMITTED PER FILE, NOT PER FEATURE.
   The worker emitted progress once per shapefile. Importing a single .shp --
   the normal case -- therefore produced exactly two events (0% and 100%) no
   matter how many features it held, so the bar sat frozen at 0% for the whole
   import. Measured before: 2 events for 2000 features. After: 102.
"""
import inspect
import os
import textwrap

import pytest

pytestmark = [pytest.mark.io, pytest.mark.polygons]


# ---------------------------------------------------------------------------
# 1. The transformer must be built once per CRS, not once per feature
# ---------------------------------------------------------------------------
def test_transformer_is_cached_across_features(monkeypatch):
    """THE regression, at the call-count level.

    Drives the real `match_feature_geometry` 400 times with the same CRS and
    requires at most one PROJ pipeline construction.
    """
    pyproj = pytest.importorskip("pyproj")
    from shapely.geometry import Polygon

    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    # A manager with no image records short-circuits, so give it one. The
    # match result is irrelevant here -- only the transformer cost is.
    mgr.image_records = [{'filepath': 'fake.tif', 'center': None}]
    mgr.footprint_polygons = []
    mgr.str_tree = None
    mgr.kd_tree = None
    mgr.kd_coordinates = []

    builds = {"n": 0}
    real_from_crs = pyproj.Transformer.from_crs

    def _spy(*a, **k):
        builds["n"] += 1
        return real_from_crs(*a, **k)

    monkeypatch.setattr(pyproj.Transformer, "from_crs", staticmethod(_spy))

    crs = pyproj.CRS.from_epsg(32617)     # UTM 17N -- not 4326, so a real hop
    for i in range(400):
        poly = Polygon([(620000 + i, 1010000), (620010 + i, 1010000),
                        (620010 + i, 1010010), (620000 + i, 1010010)])
        mgr.match_feature_geometry(poly, shapefile_crs=crs)

    assert builds["n"] <= 1, (
        f"pyproj.Transformer.from_crs was called {builds['n']} times for 400 "
        "features sharing ONE crs -- building a PROJ pipeline per feature was "
        "90% of total import time (26.1s of 29.1s measured on 2000 features)")


def test_transformer_cache_is_per_crs_not_global():
    """Correctness guard on the cache: two different source CRSs must not
    share a transformer, or geometries land in the wrong place."""
    pyproj = pytest.importorskip("pyproj")
    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    utm17 = pyproj.CRS.from_epsg(32617)
    utm18 = pyproj.CRS.from_epsg(32618)

    t17 = mgr._get_wgs84_transformer(utm17)
    t18 = mgr._get_wgs84_transformer(utm18)
    assert t17 is not None and t18 is not None
    assert t17 is not t18, "two different source CRSs shared one transformer"
    assert mgr._get_wgs84_transformer(utm17) is t17, "cache did not hit"


def test_a_4326_source_needs_no_transformer():
    """When the shapefile is already WGS84 there is nothing to build, and the
    cache must record that rather than rebuilding a None each time."""
    pyproj = pytest.importorskip("pyproj")
    from ..spatial_index import SpatialIndexManager

    mgr = SpatialIndexManager()
    assert mgr._get_wgs84_transformer(pyproj.CRS.from_epsg(4326)) is None
    key = mgr._crs_cache_key(pyproj.CRS.from_epsg(4326))
    assert key in mgr._wgs84_transformer_cache, (
        "the 'no transform needed' answer was not cached, so every feature "
        "re-parses the CRS to rediscover it")


def test_crs_cache_key_does_not_serialise_wkt():
    """`str(pyproj.CRS)` returns the full WKT -- kilobytes. Using it as a
    per-feature cache key would itself be a measurable cost."""
    pyproj = pytest.importorskip("pyproj")
    from ..spatial_index import SpatialIndexManager

    crs = pyproj.CRS.from_epsg(32617)
    key = SpatialIndexManager._crs_cache_key(crs)
    assert len(str(key)) < 100, (
        f"CRS cache key is {len(str(key))} chars -- it is serialising the WKT "
        "on every feature")


# ---------------------------------------------------------------------------
# 2. The progress bar must actually move
# ---------------------------------------------------------------------------
def test_worker_emits_progress_per_feature_not_per_file():
    """THE regression, at the source level.

    A behavioural version would need a real .shp on disk plus a georeferenced
    raster; the defect is structural and unambiguous, so assert that the
    per-feature loop emits, rather than only the per-file one.
    """
    from ..polygon_manager import ShapefileImportWorker

    src = textwrap.dedent(inspect.getsource(ShapefileImportWorker.run))
    # The feature loop is `for feat_idx, feat in enumerate(decomposed...)`.
    feat_loop_at = src.index("for feat_idx, feat in enumerate(decomposed")
    after_loop = src[feat_loop_at:]

    assert "self.progress.emit" in after_loop, (
        "ShapefileImportWorker.run emits progress only in the per-FILE loop. "
        "Importing a single shapefile then produces just two events (0% and "
        "100%) regardless of feature count, so the bar sits frozen at 0% for "
        "the entire import")


def test_progress_emission_is_throttled():
    """Emitting once per feature across a queued cross-thread connection would
    make the signal itself a bottleneck at 6000+ features, so the emit must be
    rate-limited."""
    from ..polygon_manager import ShapefileImportWorker

    src = textwrap.dedent(inspect.getsource(ShapefileImportWorker.run))
    assert "step" in src and "% step" in src, (
        "the per-feature progress emit appears unthrottled -- at 6000 features "
        "that is 6000 queued cross-thread signal emissions")


def test_progress_reaches_100_on_the_last_feature():
    """The throttle must not skip the final update, or the bar stalls just
    short of done."""
    from ..polygon_manager import ShapefileImportWorker

    src = textwrap.dedent(inspect.getsource(ShapefileImportWorker.run))
    assert "feat_idx == n_feats" in src, (
        "the throttled emit has no explicit final-feature case, so the last "
        "partial block of features produces no progress update")
