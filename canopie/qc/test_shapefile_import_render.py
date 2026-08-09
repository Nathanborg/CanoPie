"""
Shapefile-imported polygons must actually appear in the viewer.

THE BUG THIS PINS (reported as "polygons were imported to the polygon manager
but I don't see their paint in the imageviewer" on a 3000-feature import):

Two independent defects in the shapefile-import write path, either one of
which alone hides every imported polygon, PLUS a batching/robustness gap in
the immediate on-import draw that only bites at large feature counts.

1. coord_space stamped 'pixel' instead of 'image' (shapefile_io.py,
   polygon_manager.py -- two write sites for two import code paths). Every
   CONSUMER (load_polygons, machine_learning_manager,
   _normalize_points_for_save) checks for the literal string 'image' to mean
   "points are full-resolution raster pixels, scale by image_ref_size".
   Nothing recognizes 'pixel'. It fell into the catch-all 'scene' branch,
   which uses the points AS SCENE COORDINATES with NO SCALING. For any raster
   the viewer displays decimated -- which is every raster over the lazy
   threshold, i.e. most real drone mosaics -- that places imported polygons
   at full-resolution pixel coordinates directly in scene space: tens of
   thousands of scene units away from a preview that is a few thousand pixels
   wide. Visually indistinguishable from "nothing imported".

2. Root-ID filter treats '' as a real value. resolve_feature_identity's
   default for a shapefile with no ROOT/PLANT_ID/PLOT_ID column is the empty
   string, not None. load_polygons's relaxed root check only skips filtering
   when saved_root IS None -- so '' compared against any real root id (e.g.
   "1") reads as a mismatch, and the polygon is silently dropped on every
   navigation to the image. This is independent of bug #1: fixing coord_space
   alone still loses every polygon from a shapefile with no root column.

3. The immediate live-draw loop in _on_shapefile_import_finished (the path
   that paints polygons into any ALREADY-OPEN viewer without waiting for
   load_polygons on next navigation) had no signal blocking / index-method
   batching during the add loop, no per-polygon error handling, and no final
   scene.update()/viewport().update(). At 3000 features this is what made a
   large import "very very slow", and the missing repaint could leave
   correctly-placed items simply unpainted.
"""
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

pytestmark = [pytest.mark.polygons, pytest.mark.contract]


# ---------------------------------------------------------------------------
# Bug #1: coord_space
# ---------------------------------------------------------------------------
def test_shapefile_io_stamps_image_not_pixel():
    """Static guard on the shapefile_io.py write site: 'pixel' must never
    reappear as the coord_space import stamps, since no consumer honours it."""
    import inspect

    from .. import shapefile_io

    src = inspect.getsource(shapefile_io)
    # Scope to the region that builds the imported poly_dict, not the whole
    # module (which legitimately mentions 'pixel' elsewhere, e.g. CRS labels).
    idx = src.index("'root': identity['root_val']")
    window = src[idx: idx + 1000]
    assert "'coord_space': 'image'" in window, (
        "shapefile_io.py's imported poly_dict no longer stamps coord_space="
        "'image' -- if it went back to 'pixel', load_polygons will use "
        "unscaled full-resolution pixel coordinates as scene coordinates")
    assert "'coord_space': 'pixel'" not in window


def test_polygon_manager_import_stamps_image_not_pixel():
    """Same guard for polygon_manager.py's own (duplicate) import path."""
    import inspect

    from .. import polygon_manager

    src = inspect.getsource(polygon_manager)
    idx = src.index("'root': identity['root_val']")
    window = src[idx: idx + 1000]
    assert "'coord_space': 'image'" in window
    assert "'coord_space': 'pixel'" not in window


def test_load_polygons_scales_image_and_legacy_pixel_coord_space(
        synthetic_project, viewer_factory, monkeypatch):
    """THE consumer-side proof: drive the REAL load_polygons() with a polygon
    whose points sit at the middle of a LARGER reference frame than the
    displayed image (exactly what a full-resolution shapefile import looks
    like against a decimated preview), tagged 'image' and tagged 'pixel'.

    BOTH must land inside the displayed pixmap. 'pixel' is the legacy value
    the shapefile import stamped before it was fixed to write 'image'; real
    projects (2,333 files measured in one) still carry it on disk, so the
    consumers now normalise it to 'image' instead of letting it fall into the
    unscaled 'scene' branch -- which placed every polygon far off-screen,
    indistinguishable from "polygon never rendered".
    """
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")     # a real 64x64 fixture
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    assert imgd is not None and imgd.image is not None
    viewer.image_data = imgd
    pm = synthetic_project.convert_cv_to_pixmap(imgd.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)

    monkeypatch.setattr(synthetic_project, "get_root_by_filepath", lambda f: None)

    def _seed(coord_space):
        group = f"shp_import_test_{coord_space}"
        synthetic_project.all_polygons[group] = {
            fp: {
                'name': group, 'group': group, 'root': '',
                'type': 'polygon', 'coord_space': coord_space,
                # A reference frame 100x the fixture's 64x64 -- the scale gap
                # between a full-res shapefile reprojection and a decimated
                # preview. Points near the middle of THAT frame.
                'image_ref_size': {'w': 6400, 'h': 6400},
                'points': [(3100, 3100), (3300, 3100), (3300, 3300), (3100, 3300)],
                'properties': {},
            }
        }
        synthetic_project._poly_norm_index_invalid = True
        synthetic_project.load_polygons(viewer, imgd)
        items = [p['item'] for p in viewer.polygons if p.get('name') == group]
        assert items, f"coord_space={coord_space!r}: polygon was not added to the scene at all"
        return items[0].polygon.boundingRect()

    try:
        image_rect = _seed('image')
        pixel_rect = _seed('pixel')

        scene_bounds = QtCore.QRectF(0, 0, pm.width(), pm.height()).adjusted(-100, -100, 100, 100)
        assert scene_bounds.intersects(image_rect), (
            f"coord_space='image' placed the polygon at {image_rect}, outside "
            f"the displayed image bounds {scene_bounds} -- scaling regressed")
        assert scene_bounds.intersects(pixel_rect), (
            f"legacy coord_space='pixel' placed the polygon at {pixel_rect}, "
            f"outside the displayed image bounds {scene_bounds} -- the "
            "'pixel'->'image' normalisation regressed, so every polygon in an "
            "existing imported project renders off-screen again")
        # And identically: the alias must be exact, not merely "also visible".
        assert pixel_rect == image_rect, (
            f"'pixel' ({pixel_rect}) and 'image' ({image_rect}) scaled "
            "differently for identical points")
    finally:
        for g in ("shp_import_test_image", "shp_import_test_pixel"):
            synthetic_project.all_polygons.pop(g, None)
        synthetic_project._poly_norm_index_invalid = True


# ---------------------------------------------------------------------------
# Bug #2: empty-string root
# ---------------------------------------------------------------------------
def test_empty_string_root_is_not_filtered(synthetic_project, viewer_factory, monkeypatch):
    """THE regression, driven through the real load_polygons().

    A polygon with root='' (shapefile import's default when there is no
    ROOT/PLANT_ID/PLOT_ID column) must render even when the image's owning
    root id is a real, non-empty value -- '' means "no root info", not "root
    zero" or "wrong root".
    """
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pm = synthetic_project.convert_cv_to_pixmap(imgd.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)

    # Force a real, non-empty owning root id -- the exact condition under
    # which '' previously compared unequal and got dropped.
    monkeypatch.setattr(synthetic_project, "get_root_by_filepath", lambda f: "Root_1")
    monkeypatch.setattr(synthetic_project, "root_id_mapping", {"Root_1": 1}, raising=False)

    group = "shp_import_no_root_column"
    synthetic_project.all_polygons[group] = {
        fp: {
            'name': group, 'group': group, 'root': '',      # <-- the bug
            'type': 'polygon', 'coord_space': 'image',
            'image_ref_size': {'w': 64, 'h': 64},
            'points': [(10, 10), (25, 10), (25, 25), (10, 25)],
            'properties': {},
        }
    }
    synthetic_project._poly_norm_index_invalid = True
    try:
        synthetic_project.load_polygons(viewer, imgd)
        names = {p.get('name') for p in viewer.polygons}
        assert group in names, (
            "a polygon with root='' was dropped even though it carries no "
            "real root conflict -- the empty-string default from shapefile "
            "import is being treated as an actual (mismatching) root id")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        synthetic_project._poly_norm_index_invalid = True


def test_genuine_root_mismatch_is_still_filtered(synthetic_project, viewer_factory, monkeypatch):
    """The fix must not defeat the filter entirely -- a REAL, non-empty root
    that actually differs still has to be skipped."""
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pm = synthetic_project.convert_cv_to_pixmap(imgd.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)

    monkeypatch.setattr(synthetic_project, "get_root_by_filepath", lambda f: "Root_1")
    monkeypatch.setattr(synthetic_project, "root_id_mapping", {"Root_1": 1}, raising=False)

    group = "shp_import_wrong_root"
    synthetic_project.all_polygons[group] = {
        fp: {
            'name': group, 'group': group, 'root': '999',   # genuinely different
            'type': 'polygon', 'coord_space': 'image',
            'image_ref_size': {'w': 64, 'h': 64},
            'points': [(10, 10), (25, 10), (25, 25), (10, 25)],
            'properties': {},
        }
    }
    synthetic_project._poly_norm_index_invalid = True
    try:
        synthetic_project.load_polygons(viewer, imgd)
        names = {p.get('name') for p in viewer.polygons}
        assert group not in names, (
            "a polygon whose root genuinely differs from the image's owning "
            "root was rendered anyway -- the empty-string fix over-relaxed "
            "the filter")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        synthetic_project._poly_norm_index_invalid = True


# ---------------------------------------------------------------------------
# Bug #3: the live-draw loop's batching / robustness / repaint
# ---------------------------------------------------------------------------
def test_import_finished_batches_signal_blocking_per_viewer():
    """Static guard: the live-draw loop must block/restore signals and the
    scene index method around the add loop, mirroring load_polygons's own
    batching -- not add 3000 polygons one at a time at full cost each."""
    import inspect

    from ..polygon_manager import PolygonManager

    src = inspect.getsource(PolygonManager._on_shapefile_import_finished)
    assert "blockSignals(True)" in src, (
        "_on_shapefile_import_finished no longer blocks viewer/scene signals "
        "during the import draw loop -- reintroduces the per-polygon signal "
        "cost that made a 3000-feature import very slow")
    assert "NoIndex" in src, (
        "_on_shapefile_import_finished no longer disables scene indexing "
        "during the bulk add")
    assert "blockSignals(False)" in src


def test_import_finished_repaints_touched_viewers():
    """Static guard: a scene/viewport update must follow the draw loop, or
    correctly-placed polygons can simply not be painted."""
    import inspect

    from ..polygon_manager import PolygonManager

    src = inspect.getsource(PolygonManager._on_shapefile_import_finished)
    assert "scene.update()" in src or ".update()" in src
    assert "viewport().update()" in src, (
        "_on_shapefile_import_finished no longer forces a viewport repaint "
        "after the import draw loop")


def test_import_finished_catches_a_bad_polygon_and_continues(
        synthetic_project, viewer_factory, monkeypatch):
    """THE regression, behaviourally: one malformed polygon in the imported
    batch must not abort the rest of the import's live-draw pass.

    Before the fix, an exception raised while drawing any single polygon
    propagated out of the whole nested loop uncaught, silently abandoning
    every remaining group -- not just the bad one.
    """
    from ..polygon_manager import PolygonManager
    from .fixtures_manifest import fixture_image_path

    found = synthetic_project.findChildren(PolygonManager)
    if not found:
        pytest.skip("no PolygonManager instance attached to this ProjectTab build")
    pm = found[0]

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pixmap = synthetic_project.convert_cv_to_pixmap(imgd.image)
    assert pixmap is not None and not pixmap.isNull()
    viewer.set_image(pixmap)

    monkeypatch.setattr(pm, "_iter_viewers",
                        lambda: [({"image_data": imgd}, viewer)])

    class _Progress:
        def close(self):
            pass

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    good_pts = [(10, 10), (25, 10), (25, 25), (10, 25)]
    imported_data = {
        "import_bad": {fp: {'name': 'import_bad', 'points': "not-a-list-of-points",
                            'image_ref_size': {'w': 64, 'h': 64}, 'root': ''}},
        "import_good": {fp: {'name': 'import_good', 'points': good_pts,
                             'image_ref_size': {'w': 64, 'h': 64}, 'root': ''}},
    }
    try:
        pm._on_shapefile_import_finished(imported_data, [], _Progress())
        names = {p.get('name') for p in viewer.polygons}
        assert 'import_good' in names, (
            "the malformed 'import_bad' polygon aborted the loop before "
            "'import_good' (which comes after it) was ever drawn")
    finally:
        synthetic_project.all_polygons.pop("import_bad", None)
        synthetic_project.all_polygons.pop("import_good", None)
        synthetic_project._poly_norm_index_invalid = True


# ---------------------------------------------------------------------------
# Scene-index restore must account for the FINAL polygon count, not the
# image-size heuristic captured before the batch add
# ---------------------------------------------------------------------------
def test_load_polygons_restores_the_scene_index_after_a_large_batch(
        synthetic_project, viewer_factory, monkeypatch):
    """NoIndex is for the ADD LOOP only; it must not become the steady state.

    An earlier revision of this test asserted the opposite -- that a batch
    above 1000 items should be left on NoIndex permanently, on the theory that
    the BspTree rebuild was what made large imports slow. Benchmarking says
    otherwise: building the tree costs 3.5 ms at 3000 items and 16.7 ms at
    6000 (negligible), while leaving the scene on NoIndex makes every viewport
    query 2.7x slower at 3000 items -- i.e. it degrades panning, which is
    exactly the symptom it was supposed to help. The real cost was the
    inflated item boundingRect (see test_polygon_bounding_rect_perf.py).

    So: whatever the scene had before the bulk add must be restored after it.
    """
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pm = synthetic_project.convert_cv_to_pixmap(imgd.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)
    # Force the small-image assumption regardless of how set_image sized it.
    viewer._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)

    monkeypatch.setattr(synthetic_project, "get_root_by_filepath", lambda f: None)

    N = 1200
    groups = {}
    for i in range(N):
        g = f"bulk_{i:04d}"
        groups[g] = {fp: {
            'name': g, 'group': g, 'root': '', 'type': 'point',
            'coord_space': 'image', 'image_ref_size': {'w': 64, 'h': 64},
            'points': [(float(i % 60) + 1, float((i * 7) % 60) + 1)],
            'properties': {},
        }}
    synthetic_project.all_polygons.update(groups)
    synthetic_project._poly_norm_index_invalid = True
    try:
        synthetic_project.load_polygons(viewer, imgd)
        assert viewer._polygon_item_count >= N, (
            f"expected at least {N} items to have been added, got "
            f"{viewer._polygon_item_count}")
        assert viewer._scene.itemIndexMethod() == QtWidgets.QGraphicsScene.BspTreeIndex, (
            "the scene was left on NoIndex after the bulk add -- NoIndex is "
            "correct only DURING the add loop; as a steady state it makes "
            "every viewport repaint and hit-test a linear scan of the whole "
            "scene, which is measurably worse for panning")
    finally:
        for g in groups:
            synthetic_project.all_polygons.pop(g, None)
        synthetic_project._poly_norm_index_invalid = True


# ---------------------------------------------------------------------------
# Import draws DECIMATED geometry but must persist the EXACT coordinates
# ---------------------------------------------------------------------------
def test_import_draws_decimated_geometry_but_saves_exact(
        synthetic_project, viewer_factory, monkeypatch):
    """The import live-draw pass builds one QPointF per vertex, and the BCI
    crown map carries 2,147,273 of them -- 3.22 s of pure object construction.

    Drawing the same pyramid level the project-open path already uses keeps
    7.5% of the vertices on real crown geometry and cuts that to 0.91 s. The
    invariant that makes it safe: the DISPLAYED outline may be coarse, but
    all_polygons must still hold the exact coordinates (they are what gets
    saved, exported and measured), and the item must be flagged is_lod so the
    first drag upgrades it before any edit can be written back.
    """
    import numpy as np
    from ..polygon_manager import PolygonManager
    from .. import polygon_lod
    from .fixtures_manifest import fixture_image_path

    found = synthetic_project.findChildren(PolygonManager)
    if not found:
        pytest.skip("no PolygonManager instance attached to this ProjectTab build")
    pm = found[0]

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pixmap = synthetic_project.convert_cv_to_pixmap(imgd.image)
    viewer.set_image(pixmap)
    monkeypatch.setattr(pm, "_iter_viewers",
                        lambda: [({"image_data": imgd}, viewer)])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    class _Progress:
        def close(self):
            pass

    # A dense smooth ring, like a real crown outline: many vertices far closer
    # together than the level-4 (16 px) tolerance.
    n = 900
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    exact = [(float(32 + 28 * np.cos(t)), float(32 + 28 * np.sin(t))) for t in a]
    assert n >= polygon_lod.MIN_POINTS_FOR_LOD

    imported_data = {
        "crown_lod": {fp: {'name': 'crown_lod', 'points': exact,
                           'image_ref_size': {'w': 64, 'h': 64}, 'root': ''}},
    }
    try:
        pm._on_shapefile_import_finished(imported_data, [], _Progress())

        rec = synthetic_project.all_polygons["crown_lod"][fp]
        assert rec['points'] == exact, (
            "import stored the DECIMATED outline -- the exact coordinates must "
            "survive, they are what gets saved and measured")

        items = [p for p in viewer.polygons if p.get('name') == 'crown_lod']
        assert items, "the imported polygon was never drawn"
        item = items[0]['item']
        assert len(item.polygon) < n, (
            f"display geometry was not decimated ({len(item.polygon)} of {n} "
            f"vertices) -- the import draw loop is still building every point")
        assert item.is_lod_geometry is True, (
            "a decimated item MUST be flagged is_lod, otherwise a drag would "
            "edit the coarse outline and update_all_polygons would write it "
            "back over the real one")
    finally:
        synthetic_project.all_polygons.pop("crown_lod", None)
        synthetic_project._poly_norm_index_invalid = True
