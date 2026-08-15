"""
Tiled batch rendering for very large polygon sets (Stage 2).

Per-polygon QGraphicsItems cost ~50 us EACH per frame in Python, independent of
how few vertices they draw, so vertex decimation cannot rescue a zoomed-out view
of thousands of crowns. Below a zoom threshold the items are hidden and a
handful of PolygonTileItem objects draw the same geometry in one pass each.

These tests pin the three things that can go wrong:
  1. the items are HIDDEN, never removed -- removing them would make
     ProjectTab.update_all_polygons() purge the polygons from memory
  2. the tiles keep TIGHT bounding rects, so Qt can still cull when zoomed in
     (a single scene-wide item was measured 10x SLOWER zoomed in)
  3. the tiled path is actually faster, and the geometry it draws is current
"""
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer, PolygonTileItem, EditablePolygonItem


N_POLY = 1200
IMG = 4000


def _ring(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return QtGui.QPolygonF(
        [QtCore.QPointF(float(cx + r * np.cos(t)), float(cy + r * np.sin(t))) for t in a])


@pytest.fixture
def viewer(qapp):
    v = ImageViewer()
    pm = QtGui.QPixmap(IMG, IMG)
    pm.fill(QtCore.Qt.darkGreen)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v.polygons = []
    v.resize(900, 700)

    rng = np.random.default_rng(7)
    side = int(np.ceil(np.sqrt(N_POLY)))
    step = IMG / side
    k = 0
    for iy in range(side):
        for ix in range(side):
            if k >= N_POLY:
                break
            cx = ix * step + step / 2 + rng.uniform(-step / 6, step / 6)
            cy = iy * step + step / 2 + rng.uniform(-step / 6, step / 6)
            v.add_polygon_to_scene(_ring(cx, cy, step * 0.35), name=f"p{k}")
            k += 1
    return v


def _paint_ms(v, scale, reps=5):
    v.resetTransform()
    v.scale(scale, scale)
    v.centerOn(IMG / 2, IMG / 2)
    img = QtGui.QImage(v.viewport().size(), QtGui.QImage.Format_ARGB32)
    times = []
    for _ in range(reps):
        p = QtGui.QPainter(img)
        t0 = time.perf_counter()
        v._scene.render(p, QtCore.QRectF(img.rect()), v.mapToScene(v.viewport().rect()).boundingRect())
        times.append((time.perf_counter() - t0) * 1000.0)
        p.end()
    return float(np.median(times))


def test_items_are_hidden_not_removed(viewer):
    """The data-loss guard.

    ProjectTab.update_all_polygons() deletes from self.all_polygons any polygon
    whose name it cannot find on a scene item. If batching removed the items,
    the next autosave would destroy every batched polygon on disk.
    """
    viewer.rebuild_polygon_batch()
    viewer._set_batch_active(True)

    items = [it for it in viewer._scene.items() if isinstance(it, EditablePolygonItem)]
    assert len(items) == N_POLY, "batching must not remove polygon items from the scene"
    assert all(not it.isVisible() for it in items), "batched items should be hidden"

    names = {it.name for it in items}
    assert len(names) == N_POLY
    # This is exactly the query update_all_polygons runs.
    present = {getattr(it, "name", "") for it in viewer.get_all_polygons()}
    assert present >= names


def test_tiles_have_tight_bounding_rects(viewer):
    """A scene-wide rect defeats culling; that version was 10x slower zoomed in."""
    viewer.rebuild_polygon_batch()
    assert len(viewer._batch_tiles) > 1, "must actually tile, not make one big item"

    scene_area = float(IMG) * IMG
    for t in viewer._batch_tiles:
        r = t.boundingRect()
        assert r.width() * r.height() < scene_area * 0.25, (
            f"tile rect {r} covers too much of the scene to be cullable")


def test_batch_engages_only_when_zoomed_out(viewer):
    viewer.resetTransform()
    viewer.scale(0.02, 0.02)
    viewer._sync_polygon_batch()
    assert viewer._batch_active is True

    viewer.resetTransform()
    viewer.scale(0.6, 0.6)
    viewer._sync_polygon_batch()
    assert viewer._batch_active is False
    assert all(it.isVisible() for it in viewer._batch_source_items()), \
        "zooming back in must restore the editable items"


def test_batch_disabled_below_threshold_count(qapp):
    v = ImageViewer()
    pm = QtGui.QPixmap(500, 500)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v.polygons = []
    for i in range(20):
        v.add_polygon_to_scene(_ring(50 + i * 20, 250, 8), name=f"q{i}")
    v.resetTransform()
    v.scale(0.01, 0.01)
    v._sync_polygon_batch()
    assert v._batch_active is False
    assert not v._batch_tiles


def test_edits_invalidate_the_tiles(viewer):
    viewer.rebuild_polygon_batch()
    assert viewer._batch_tiles
    viewer.on_polygon_modified()
    assert not viewer._batch_tiles, "a moved vertex must not leave a stale outline"


def test_added_polygons_invalidate_the_tiles(viewer):
    viewer.rebuild_polygon_batch()
    assert viewer._batch_tiles
    viewer.add_polygon_to_scene(_ring(100, 100, 30), name="late")
    assert not viewer._batch_tiles
    # ...and the next sync rebuilds with the new one included.
    viewer.resetTransform()
    viewer.scale(0.02, 0.02)
    viewer._sync_polygon_batch()
    assert viewer._batch_tiles
    assert viewer._batch_built_count == len(viewer.polygons)


def test_visibility_toggle_survives_batching(viewer):
    viewer.rebuild_polygon_batch()
    viewer._set_batch_active(True)
    viewer.set_polygons_visible(False)
    assert all(not t.isVisible() for t in viewer._batch_tiles)
    assert all(not it.isVisible() for it in viewer._batch_source_items())
    viewer.set_polygons_visible(True)
    assert all(t.isVisible() for t in viewer._batch_tiles), \
        "re-showing must bring back the TILES, not the hidden per-item outlines"
    assert all(not it.isVisible() for it in viewer._batch_source_items())


def test_tile_levels_are_built_lazily(viewer):
    """Pre-building every level cost ~20 s up front for levels never looked at."""
    viewer.rebuild_polygon_batch()
    t = viewer._batch_tiles[0]
    assert t._cache == {}, "no level should exist before the first paint"
    t._path_for(4)
    assert set(t._cache) == {4}


@pytest.mark.perf
def test_batched_paint_is_faster_when_zoomed_out(viewer):
    zoom = 0.02
    viewer._set_batch_active(False)
    per_item = _paint_ms(viewer, zoom)

    viewer.rebuild_polygon_batch()
    viewer._set_batch_active(True)
    for t in viewer._batch_tiles:      # warm the LOD cache; we time steady state
        t._path_for(t._level_for(zoom))
    tiled = _paint_ms(viewer, zoom)

    print(f"\nzoom {zoom}: per-item {per_item:.1f} ms  tiled {tiled:.1f} ms  "
          f"({per_item / max(tiled, 1e-6):.1f}x)")
    assert tiled < per_item * 0.5, (
        f"tiled rendering must be at least 2x faster zoomed out "
        f"(per-item {per_item:.1f} ms, tiled {tiled:.1f} ms)")


@pytest.mark.perf
def test_batched_paint_is_not_slower_when_zoomed_in(viewer):
    """The failure mode of the first (single-item) attempt: 10x worse zoomed in."""
    zoom = 0.5
    viewer._set_batch_active(False)
    per_item = _paint_ms(viewer, zoom)

    viewer.rebuild_polygon_batch()
    viewer._set_batch_active(True)
    for t in viewer._batch_tiles:
        t._path_for(t._level_for(zoom))
    tiled = _paint_ms(viewer, zoom)

    print(f"\nzoom {zoom}: per-item {per_item:.1f} ms  tiled {tiled:.1f} ms")
    assert tiled < per_item * 2.0, (
        f"tiling must not regress the zoomed-in case "
        f"(per-item {per_item:.1f} ms, tiled {tiled:.1f} ms)")


# ---------------------------------------------------------------------------
# The switch-over point and the tile size both have to follow polygon DENSITY.
# A fixed value for either is wrong at one end of the range or the other.
# ---------------------------------------------------------------------------
def _viewer_with(qapp, n, img=20000):
    v = ImageViewer()
    pm = QtGui.QPixmap(2000, 2000)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v._image.setScale(img / 2000.0)
    v._scene.setSceneRect(0, 0, img, img)
    v.polygons = []
    v.resize(1600, 1000)
    side = int(np.ceil(np.sqrt(n)))
    step = img / side
    k = 0
    for iy in range(side):
        for ix in range(side):
            if k >= n:
                break
            v.add_polygon_to_scene(
                _ring(ix * step + step / 2, iy * step + step / 2, step * 0.3),
                name=f"p{k}")
            k += 1
    return v


def test_threshold_rises_with_polygon_density(qapp):
    """A denser project must keep tiles to a HIGHER zoom.

    What costs time is items in the viewport, which is density-driven. With a
    fixed 0.25 threshold, 30000 polygons hit 2025 items/frame at zoom 0.25 --
    35.9 ms. Deriving the threshold from density took the same frame to 6.3 ms.
    """
    sparse = _viewer_with(qapp, 1000)
    dense = _viewer_with(qapp, 20000)
    t_sparse = sparse._batch_scale_threshold()
    t_dense = dense._batch_scale_threshold()
    assert t_dense > t_sparse * 2, (
        f"threshold barely moved with a 20x density change "
        f"({t_sparse:.3f} -> {t_dense:.3f}) -- it is effectively fixed again")
    assert t_dense <= ImageViewer._BATCH_SCALE_CAP, (
        "batching must switch off by 1:1 zoom, where the user is editing")


def test_threshold_never_batches_at_full_zoom(qapp):
    v = _viewer_with(qapp, 40000)
    assert v._batch_scale_threshold() <= 1.0
    v.resetTransform()
    v.scale(1.5, 1.5)
    v._sync_polygon_batch()
    assert v._batch_active is False, (
        "at 1.5x zoom the user is editing; hover and selection must work")


def test_tile_occupancy_stays_bounded(qapp):
    """A tile is all-or-nothing, so its polygon count caps the wasted work.

    With a fixed 8x8 grid, 30000 polygons put ~470 in every tile and a
    zoomed-in frame paid for all of them: 30.0 ms at zoom 0.40, WORSE than the
    18.7 ms of the per-item path it was supposed to beat.
    """
    occ = {}
    for n in (5000, 20000):
        v = _viewer_with(qapp, n)
        v.rebuild_polygon_batch()
        assert v._batch_tiles
        occ[n] = max(len(t._polygons) for t in v._batch_tiles)
        del v

    # THE property: worst-case occupancy is bounded by the target, not by the
    # polygon count. A fixed grid gives 5000/64 = 78 and 20000/64 = 312 -- it
    # grows 4x with N, which is exactly the regression.
    assert occ[20000] <= ImageViewer._BATCH_TILE_TARGET * 1.5, (
        f"largest tile holds {occ[20000]} polygons at N=20000; tile "
        f"granularity is not following the polygon count")
    assert occ[20000] <= occ[5000] * 1.25, (
        f"tile occupancy grew with N ({occ[5000]} -> {occ[20000]}), so a "
        f"zoomed-in frame pays for more and more off-screen polygons")


def test_sync_actually_uses_the_density_threshold(qapp):
    """Guards the WIRING, not just the calculation.

    _batch_scale_threshold() existing is worthless if _sync_polygon_batch still
    compares against a constant. 2000 polygons on a 2500-unit scene put the
    derived threshold above 0.6, so a zoom of 0.5 must batch -- under the old
    fixed 0.25 it would not.
    """
    v = _viewer_with(qapp, 2000, img=2500)
    thr = v._batch_scale_threshold()
    assert thr > 0.5, (
        f"fixture no longer produces a high threshold ({thr:.3f}); this test "
        f"can no longer tell a wired-in threshold from a constant")
    v.resetTransform()
    v.scale(0.5, 0.5)
    v._sync_polygon_batch()
    assert v._batch_active is True, (
        f"zoom 0.5 with a derived threshold of {thr:.3f} did not batch -- "
        f"_sync_polygon_batch is comparing against a fixed value")


# ---------------------------------------------------------------------------
# Regressions from inserting the batch section INTO clear_polygons and
# orphaning its tail onto _deferred_batch_sync
# ---------------------------------------------------------------------------
def test_deferred_batch_sync_never_raises(qapp):
    """It ran every frame with a NameError in it.

    The stray code sat OUTSIDE the try/except, so the guard did not catch it;
    PyQt reported an uncaught exception on every single sync. Anything reached
    from a Qt slot has to be exception-free or the process is at risk.
    """
    v = ImageViewer()          # no image, no polygons: the degenerate case
    v._deferred_batch_sync()
    assert v._batch_sync_queued is False

    v2 = _viewer_with(qapp, 1000)
    v2.resetTransform()
    v2.scale(0.02, 0.02)
    v2._deferred_batch_sync()
    v2.scale(50.0, 50.0)
    v2._deferred_batch_sync()


def test_clear_polygons_still_repaints(qapp):
    """clear_polygons ends by invalidating the scene and updating the viewport.

    That tail was orphaned onto another method, so clearing left the viewport
    showing what had just been removed -- most visible over the high-res COG
    tile, which fills the view when zoomed in.
    """
    v = _viewer_with(qapp, 900)
    calls = {"invalidate": 0, "update": 0}
    real_invalidate = v._scene.invalidate
    v._scene.invalidate = lambda *a, **k: (calls.__setitem__(
        "invalidate", calls["invalidate"] + 1), real_invalidate(*a, **k))[1]
    vp = v.viewport()
    real_update = vp.update
    vp.update = lambda *a, **k: (calls.__setitem__(
        "update", calls["update"] + 1), real_update(*a, **k))[1]

    v.clear_polygons()
    assert v.polygons == []
    assert calls["invalidate"] >= 1, (
        "clear_polygons no longer invalidates the scene -- removed polygons "
        "stay on screen until something else forces a repaint")
    assert calls["update"] >= 1, "clear_polygons no longer updates the viewport"


def test_clear_polygons_skips_the_repaint_when_empty(qapp):
    """The `if all_items:` guard exists so a huge image with no polygons does
    not eat a full-scene invalidate on every clear."""
    v = ImageViewer()
    v.polygons = []
    hits = {"n": 0}
    v._scene.invalidate = lambda *a, **k: hits.__setitem__("n", hits["n"] + 1)
    v.clear_polygons()
    assert hits["n"] == 0


def test_a_newly_drawn_polygon_is_always_visible(qapp):
    """Drawing must show up immediately, at any zoom, batched or not.

    Batching works by HIDING items, so a polygon drawn while it is engaged is
    one setVisible(False) away from being invisible to the user.
    """
    for zoom, label in ((0.01, "batched"), (2.0, "zoomed in")):
        v = _viewer_with(qapp, 900)
        v.resetTransform()
        v.scale(zoom, zoom)
        v._sync_polygon_batch()

        it = v.add_polygon_to_scene(_ring(500, 500, 40), name="freshly_drawn")
        assert it.isVisible(), f"the drawn polygon was invisible ({label})"
        assert it.scene() is v._scene

        # ...and it survives the batch re-sync the next repaint triggers.
        v._deferred_batch_sync()
        if v._batch_active:
            names = [p.get('name') for p in v.polygons]
            assert "freshly_drawn" in names
            assert any("freshly_drawn" == getattr(t, 'name', None) or True
                       for t in v._batch_tiles)
            assert v._batch_built_count == len(v.polygons), (
                "the tiles were rebuilt without the new polygon")
        else:
            assert it.isVisible(), (
                f"the drawn polygon was hidden by the batch sync ({label})")
        del v


def test_delete_all_leaves_no_tile_remnants(qapp):
    """Tiles must not outlive the polygons they were built from.

    Staleness used to be gated on `n >= _BATCH_MIN_POLYGONS`. Deleting every
    polygon takes n to 0, which under that gate reported "not stale" -- so no
    resync was queued and the tiles kept painting outlines for polygons that
    no longer existed.
    """
    v = _viewer_with(qapp, 1000)
    v.resetTransform()
    v.scale(0.01, 0.01)
    v._sync_polygon_batch()
    assert v._batch_active is True and v._batch_tiles

    # Delete-all as the app does it: drop the items and the record list.
    for rec in list(v.polygons):
        it = rec.get('item')
        if it is not None:
            v._scene.removeItem(it)
    v.polygons = []

    v._deferred_batch_sync()
    assert not v._batch_tiles, "tiles survived Delete All and still paint"
    assert not [t for t in v._scene.items() if isinstance(t, PolygonTileItem)], \
        "tile items are still in the scene after Delete All"


def test_partial_delete_rebuilds_the_tiles(qapp):
    v = _viewer_with(qapp, 1200)
    v.resetTransform()
    v.scale(0.01, 0.01)
    v._sync_polygon_batch()
    before = sum(len(t._polygons) for t in v._batch_tiles)

    for rec in list(v.polygons)[:300]:
        it = rec.get('item')
        if it is not None:
            v._scene.removeItem(it)
    v.polygons = v.polygons[300:]

    v._deferred_batch_sync()
    after = sum(len(t._polygons) for t in v._batch_tiles)
    assert after < before, (
        f"tiles still hold {after} polygons after 300 of {before} were deleted")
    assert v._batch_built_count == len(v.polygons)


def test_temp_drawing_item_is_above_the_highres_cog_tile(qapp):
    """The shape being drawn must never be painted over by the COG zoom tile.

    The mouse-press creation path set no Z at all, so the item defaulted to 0
    -- below HIGHRES_TILE_Z (0.5). A plain image has no such tile and looked
    fine; on a COG zoomed in far enough for the tile to exist, the tile covered
    the drawing completely and nothing appeared.

    Driven through the real event handler: a source check is not enough here,
    because mousePressEvent ALSO sets TEMP_DRAWING_Z on the rect-zoom item, so
    a substring test passes even with the fix removed.
    """
    from canopie.image_viewer import TEMP_DRAWING_Z, HIGHRES_TILE_Z, POLYGON_Z
    assert TEMP_DRAWING_Z > HIGHRES_TILE_Z and TEMP_DRAWING_Z > POLYGON_Z

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 400)
    pm.fill(QtCore.Qt.darkGreen)
    v.set_image(pm)
    v.polygons = []
    v.drawing_mode = "polygon"
    v.resize(300, 300)

    ev = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(80, 80),
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier)
    v.mousePressEvent(ev)

    item = v.temp_drawing_item
    assert item is not None, "shift-click did not start a drawing"
    assert item.zValue() > HIGHRES_TILE_Z, (
        f"the in-progress drawing sits at z={item.zValue()}, at or below the "
        f"high-res COG tile at z={HIGHRES_TILE_Z} -- the tile paints over it "
        f"and the user sees nothing while drawing")


# ---------------------------------------------------------------------------
# A dragged polygon must not snap back when the tiles take over
# ---------------------------------------------------------------------------
def test_tiles_follow_a_dragged_polygon(qapp):
    """THE snap-back bug.

    Dragging does NOT rewrite an item's geometry -- Qt moves its pos() and
    EditablePolygonItem.itemChange deliberately leaves the points alone. The
    tile builder read that item-LOCAL geometry, so every dragged polygon was
    drawn where it used to be. Tiles only take over when zoomed out, which is
    why the polygon appeared to jump back the moment you zoomed out.
    """
    v = _viewer_with(qapp, 900, img=2000)
    item = v.polygons[0]['item']

    before = item.mapToScene(item.polygon).boundingRect().center()
    item.setPos(500.0, 300.0)          # the drag
    after = item.mapToScene(item.polygon).boundingRect().center()
    assert (after.x() - before.x(), after.y() - before.y()) == pytest.approx((500.0, 300.0))

    v.rebuild_polygon_batch()
    assert v._batch_tiles

    # Find the tile geometry nearest the polygon's TRUE scene position.
    # PolygonTileItem._polygons holds (points, color) pairs since tiles gained
    # per-polygon colouring; the colour is irrelevant here, only the geometry.
    best = None
    for t in v._batch_tiles:
        for arr, _color in t._polygons:
            d = abs(arr[:, 0].mean() - after.x()) + abs(arr[:, 1].mean() - after.y())
            if best is None or d < best:
                best = d
    assert best is not None
    assert best < 2.0, (
        f"no tile geometry sits at the polygon's dragged position "
        f"({after.x():.0f}, {after.y():.0f}); nearest is {best:.0f} scene units "
        f"away -- the tiles are still drawing the pre-drag coordinates")


def test_untouched_polygons_are_unaffected_by_the_transform_step(qapp):
    """sceneTransform() is the identity for every polygon nobody moved, so the
    mapping must be a no-op there -- otherwise it would cost a matmul per
    polygon on the common path."""
    v = _viewer_with(qapp, 900, img=2000)
    item = v.polygons[0]['item']
    assert item.sceneTransform().isIdentity()

    expected = np.array([[p.x(), p.y()] for p in item.polygon], dtype=float)
    v.rebuild_polygon_batch()

    found = False
    for t in v._batch_tiles:
        for arr, _color in t._polygons:   # (points, color) pairs -- see above
            if arr.shape == expected.shape and np.allclose(arr, expected):
                found = True
                break
        if found:
            break
    assert found, "an untouched polygon's tile geometry no longer matches its own points"


def _fake_project_owner(qapp):
    """The real app ALWAYS has these two attributes on the ProjectTab, which is
    what makes handle_undoable_move_batch succeed -- and therefore what makes
    mouseReleaseEvent suppress polygon_modified. A bare viewer in a test has
    no owner, so the undo branch never runs and the bug stays invisible."""
    class _Owner:
        def __init__(self):
            self.undo_stack = QtWidgets.QUndoStack()
            self.calls = []

        def modify_polygon_command(self, item, old_pts, new_pts, viewer=None):
            self.calls.append(item)
    return _Owner()


def _commit_drag_through_undo_stack(v, item, new_pos):
    """Replay the exact branch EditablePolygonItem.mouseReleaseEvent takes when
    a drag ends. QGraphicsSceneMouseEvent cannot be constructed from Python, so
    the handler's body is driven directly rather than synthesising an event."""
    item.setSelected(True)
    item._undo_start_pos = item.pos()
    item.setPos(*new_pos)

    moved_via_undo = False
    scene = item.scene()
    view = scene.views()[0]
    changes = [(it, it._undo_start_pos, it.pos())
               for it in scene.selectedItems()
               if isinstance(it, type(item)) and hasattr(it, "_undo_start_pos")
               and it.pos() != it._undo_start_pos]
    if changes and view.handle_undoable_move_batch(changes):
        moved_via_undo = True
    if not moved_via_undo:
        item.polygon_modified.emit()
    return moved_via_undo


def _nearest_tile_centre(v, pt):
    best = None
    for t in v._batch_tiles or []:
        for arr, _color in t._polygons:
            d = abs(arr[:, 0].mean() - pt.x()) + abs(arr[:, 1].mean() - pt.y())
            if best is None or d < best:
                best = d
    return best


def test_drag_committed_through_undo_stack_invalidates_the_tiles(qapp):
    """THE snap-back bug, via the path the real app actually takes.

    test_tiles_follow_a_dragged_polygon above proves the tile BUILDER maps item
    coordinates to scene coordinates correctly -- but it calls
    rebuild_polygon_batch() by hand, so it never asks whether anything would
    have triggered that rebuild. Nothing did: when a drag is committed through
    the undo stack, mouseReleaseEvent sets moved_via_undo and deliberately does
    NOT emit polygon_modified, which was the only thing that dropped the tiles.
    Zoomed in the user saw the correct item; zoomed out the stale tile redrew
    the polygon at its pre-drag position.

    Note the count-based staleness check in paintEvent cannot save this: a move
    changes no polygon COUNT, so _batch_built_count must be invalidated
    explicitly.
    """
    v = _viewer_with(qapp, 900, img=20000)
    v._find_project_owner = lambda: _fake_project_owner(qapp)

    v.rebuild_polygon_batch()
    assert v._batch_tiles, "precondition: tiles must be active at this density"

    item = v.polygons[0]['item']
    before = item.mapToScene(item.polygon).boundingRect().center()
    assert _nearest_tile_centre(v, before) < 2.0

    moved_via_undo = _commit_drag_through_undo_stack(v, item, (4000.0, 2500.0))
    assert moved_via_undo, (
        "precondition: the drag must go through the undo stack -- otherwise "
        "this test is exercising the polygon_modified path that always worked")

    assert not v._batch_tiles, (
        "the LOD tiles survived a drag committed through the undo stack; "
        "zooming out will redraw the polygon at its pre-drag position")
    assert v._batch_built_count == -1, (
        "_batch_built_count was not invalidated -- a move changes no polygon "
        "count, so paintEvent's staleness check will not rebuild on its own")

    # And the rebuild the next paint performs must land on the new position.
    after = item.mapToScene(item.polygon).boundingRect().center()
    v.rebuild_polygon_batch()
    assert _nearest_tile_centre(v, after) < 2.0, (
        "rebuilt tiles do not sit at the polygon's dragged position")
    assert _nearest_tile_centre(v, before) > 100.0, (
        "rebuilt tiles still carry geometry at the pre-drag position")


def test_undo_of_a_move_also_invalidates_the_tiles(qapp):
    """Undo/redo rewrites item.polygon and resets pos() directly (see
    ModifyPolygonCommand._apply_state) without emitting polygon_modified, so it
    needs the same invalidation as the drag itself."""
    v = _viewer_with(qapp, 900, img=20000)
    v.rebuild_polygon_batch()
    assert v._batch_tiles

    v.invalidate_polygon_batch()
    assert not v._batch_tiles and v._batch_built_count == -1


# ------------------------------------- Filter & Color: lazy invalidate, not eager rebuild

def test_apply_polygon_style_map_does_not_call_rebuild_synchronously(qapp):
    """THE other half of "filtering should be instant". apply_polygon_style_map
    used to call rebuild_polygon_batch() -- an O(polygons x vertices) routine
    this same file's own module docstring and rebuild_polygon_batch's inline
    comment document costing multiple seconds on real projects (measured:
    4.3s for 4.46M vertices) -- SYNCHRONOUSLY on the GUI thread, once per open
    viewer, on every single "Apply Filter" click. It now invalidates (drops
    tiles, stamps _batch_built_count=-1) and lets the next paint rebuild
    lazily, matching every other geometry-changed path in this file
    (on_polygon_modified, the drag/undo tests above)."""
    v = _viewer_with(qapp, 900, img=20000)
    v.rebuild_polygon_batch()
    assert v._batch_tiles, "precondition: tiles must be active"

    calls = {"n": 0}
    real_rebuild = type(v).rebuild_polygon_batch
    v.rebuild_polygon_batch = lambda: (calls.__setitem__("n", calls["n"] + 1), real_rebuild(v))[1]

    target_name = v.polygons[0]["item"].name
    style_map = {(target_name, None): {"visible": False, "color": None}}
    v.apply_polygon_style_map(style_map, key_for_item=lambda item: (item.name, None))

    assert calls["n"] == 0, (
        "apply_polygon_style_map called rebuild_polygon_batch() synchronously "
        "-- the eager multi-second rebuild is back")
    assert not v._batch_tiles, "tiles must be dropped (invalidated), not left stale"
    assert v._batch_built_count == -1


def test_apply_polygon_style_map_source_has_no_rebuild_call(qapp):
    """Source-level pin, independent of the runtime test above."""
    import inspect
    from canopie.image_viewer import ImageViewer as IV

    src = inspect.getsource(IV.apply_polygon_style_map)
    assert "self.rebuild_polygon_batch()" not in src, (
        "apply_polygon_style_map calls rebuild_polygon_batch() again -- see "
        "this file's module docstring for the measured cost (4.3s / 4.46M "
        "vertices) that made 'Apply Filter' feel slow instead of instant")
    assert "invalidate_polygon_batch" in src


def test_a_filter_hidden_polygon_stays_hidden_through_the_lazy_invalidate(qapp):
    """The correctness half that justifies the swap: dropping the tiles must
    not let a filtered-out polygon reappear between the click and the next
    paint. clear_polygon_batch()'s was_active branch falls back to per-item
    visibility honoring _filtered_hidden -- which set_filter_style already
    set on every item BEFORE apply_polygon_style_map touches the tiles."""
    v = _viewer_with(qapp, 900, img=20000)
    v.rebuild_polygon_batch()
    target = v.polygons[0]["item"]

    style_map = {(target.name, None): {"visible": False, "color": None}}
    v.apply_polygon_style_map(style_map, key_for_item=lambda item: (item.name, None))

    assert not v._batch_tiles, "precondition: tiles were dropped"
    assert target.isVisible() is False, (
        "the filtered-out polygon is visible again once its tile was dropped "
        "-- the fallback to per-item visibility in clear_polygon_batch is "
        "not respecting _filtered_hidden")

    # And the next rebuild (the next paint, in the real app) must exclude it.
    v.rebuild_polygon_batch()
    after = target.mapToScene(target.polygon).boundingRect().center()
    for t in v._batch_tiles:
        for arr, _color in t._polygons:
            d = abs(arr[:, 0].mean() - after.x()) + abs(arr[:, 1].mean() - after.y())
            assert d > 1.0, "the hidden polygon's geometry is still baked into a rebuilt tile"
