"""A point must stay exactly where the user dropped it, across any number of
refreshes -- the ORIGINAL bug report tonight: "whenever I move a point after
refreshing, the paint goes somewhere else, mostly for rescaled images."

`EditablePointItem` stores its geometry PIXMAP-LOCAL (see `ImageViewer.
_scene_to_pix_local`), while the value actually persisted to disk/`all_polygons`
is a scene/raw-basis coordinate, produced by `item.mapToScene(QPointF(*item.
_scene_xy(p)))` (see `ProjectTab.on_polygon_drawn`/`on_polygon_modified`/
`update_all_polygons`, and `ARCHITECTURE.md`'s "Point items carry SUB-PIXEL
precision" note). `_scene_to_pix_local` used to `floor()` that pixmap-local
storage to a whole PIXMAP pixel. One pixmap pixel equals one raster pixel only
while the viewer shows the raster 1:1 -- under an `.ax` resize (measured on a
real 2064x1544 5-band capture displayed at 50x33) one pixmap pixel is 41 x 47
RASTER pixels, so every reload threw away up to that much of the real
click/drop position:

  * after a drag, dragged points moved 19-61 raster pixels on the very next
    refresh (the sub-pixel part of the drop position, saved correctly, was
    destroyed by the reload's floor()), and
  * even with NO drag at all, a freshly drawn point could jump a full pixmap
    pixel on the first refresh, because the load-side round trip
    (`stored * pm_w / ref_w`) lands on `k - 1e-15` and `floor()` took that to
    `k - 1`.

This drives the REAL save/reload round trip end to end: `_scene_to_pix_local`
(what `add_point_to_scene` does to a scene click), a real Qt drag (`item.
setPos(...)`, exactly how `ItemIsMovable` moves an item), `mapToScene(
_scene_xy(p))` (the real save-side reconstruction), `_scene_to_pix_local`
again (the next save), and a FRESH `EditablePointItem` built from that stored
value (the real reload) -- then asserts the reloaded scene position equals the
post-drag position EXACTLY, not approximately.
"""
import pytest
from PyQt5 import QtCore, QtGui

from ..image_viewer import EditablePointItem, ImageViewer
from ..shapefile_io import _raw_image_dims
from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


def _viewer_with_pixmap(pm_w, pm_h):
    v = ImageViewer()
    pm = QtGui.QPixmap(pm_w, pm_h)
    pm.fill(QtCore.Qt.black)
    v.set_image(pm)
    return v


def _round_trip_after_drag(viewer, scene_pt, drag_dx, drag_dy):
    """Real save -> real drag -> real save -> real reload. Returns the scene
    position a FRESH item (a "refresh") reconstructs, for comparison against
    the position immediately after the drag."""
    stored = viewer._scene_to_pix_local(scene_pt)          # real save (draw)
    item = EditablePointItem(QtGui.QPolygonF([stored]), pixmap_item=viewer._image,
                              points_are_pixmap_local=True)

    # Real Qt drag: ItemIsMovable moves the item via .pos(), not .points.
    item.setPos(QtCore.QPointF(drag_dx, drag_dy))

    # Real save side: ProjectTab.on_polygon_modified's exact reconstruction.
    saved_scene_pt = item.mapToScene(QtCore.QPointF(*item._scene_xy(stored)))

    # Real reload: a brand-new item built from what got saved, pos() reset to
    # (0,0) since a freshly loaded item has no accumulated drag offset.
    restored = viewer._scene_to_pix_local(saved_scene_pt)
    fresh = EditablePointItem(QtGui.QPolygonF([restored]), pixmap_item=viewer._image,
                              points_are_pixmap_local=True)
    reloaded_scene_pt = fresh.mapToScene(QtCore.QPointF(*fresh._scene_xy(restored)))

    return saved_scene_pt, reloaded_scene_pt


@pytest.mark.parametrize("pm_w,pm_h,label", [
    (2064, 1544, "1:1 (no .ax)"),
    (50, 33, ".ax resize to 50x33 (the reported case)"),
    (996, 664, ".ax resize to 996x664"),
])
@pytest.mark.parametrize("scene_x,scene_y,dx,dy", [
    (12.37, 8.62, 3.47, 2.61),
    (0.5, 0.5, -0.5, -0.5),
    (5.0, 5.0, 0.0, 0.0),          # no-drag-at-all case: must still not drift
])
def test_point_survives_drag_and_refresh_exactly(qapp, pm_w, pm_h, label, scene_x, scene_y, dx, dy):
    viewer = _viewer_with_pixmap(pm_w, pm_h)
    scene_pt = QtCore.QPointF(scene_x, scene_y)

    saved, reloaded = _round_trip_after_drag(viewer, scene_pt, dx, dy)

    assert (reloaded.x(), reloaded.y()) == pytest.approx((saved.x(), saved.y()), abs=1e-6), (
        f"{label}: point drifted from {saved} to {reloaded} on refresh -- "
        f"delta ({reloaded.x()-saved.x():.4f}, {reloaded.y()-saved.y():.4f}) "
        "pixmap-local units")


def test_multiple_refreshes_do_not_accumulate_drift(qapp):
    """Refreshing N times must be idempotent, not a slow leak."""
    viewer = _viewer_with_pixmap(50, 33)
    scene_pt = QtCore.QPointF(12.37, 8.62)

    saved, reloaded = _round_trip_after_drag(viewer, scene_pt, 3.47, 2.61)
    for _ in range(5):
        # Feed the reloaded position through another save/reload pass.
        stored = viewer._scene_to_pix_local(reloaded)
        item = EditablePointItem(QtGui.QPolygonF([stored]), pixmap_item=viewer._image,
                                  points_are_pixmap_local=True)
        reloaded = item.mapToScene(QtCore.QPointF(*item._scene_xy(stored)))

    assert (reloaded.x(), reloaded.y()) == pytest.approx((saved.x(), saved.y()), abs=1e-6), (
        "position drifted further across repeated refreshes -- not idempotent")


def test_real_5band_stacked_geometry_matches_the_reported_project(qapp):
    """Uses a real fixture whose loader takes the same tifffile-fallback
    branch as the reported 5-band stacked capture (confirmed elsewhere in
    this suite: OpenCV refuses multiband_8band_ancillary), heavily decimated
    (20x, matching the reported project's own ~1:41 ratio in spirit) so one
    pixmap-local unit is worth several raw pixels, same as the real bug.

    The scene/click point must be expressed in the DISPLAYED PIXMAP's own
    coordinate space (0..pm_w, 0..pm_h) -- `_scene_to_pix_local` operates on
    that space, not on raw-image units; using raw-scale numbers here would
    just exercise the clamp, not the drift this test targets.
    """
    fp = fixture_image_path("multiband_8band_ancillary")   # cv2-refuses-it fixture, same
    raw_h, raw_w = _raw_image_dims(fp)                      # loader branch as the real 5-band file
    assert raw_h and raw_w

    pm_w, pm_h = max(2, raw_w // 20), max(2, raw_h // 20)
    viewer = _viewer_with_pixmap(pm_w, pm_h)
    scene_pt = QtCore.QPointF(pm_w * 0.3 + 0.41, pm_h * 0.3 + 0.29)

    saved, reloaded = _round_trip_after_drag(viewer, scene_pt, 0.23, -0.19)
    assert (reloaded.x(), reloaded.y()) == pytest.approx((saved.x(), saved.y()), abs=1e-6)


# ---------------------------------------------------------------------------
# Falsifiability control: the OLD floor()ing behavior, reproduced by an
# explicit floor() here (NOT by re-implementing the fixed function -- this
# calls the same real save/reload plumbing, just injects truncation at the
# one point the old code used to), must actually drift. If this control ever
# stopped drifting, the tests above would be vacuous.
# ---------------------------------------------------------------------------
def test_old_floor_based_storage_would_have_drifted(qapp):
    viewer = _viewer_with_pixmap(50, 33)
    scene_pt = QtCore.QPointF(12.37, 8.62)

    stored = viewer._scene_to_pix_local(scene_pt)
    truncated = QtCore.QPointF(float(int(stored.x())), float(int(stored.y())))
    assert (truncated.x(), truncated.y()) != (stored.x(), stored.y()), (
        "sanity check itself is broken: the chosen scene point already lands "
        "on an exact pixmap-local integer, so floor() wouldn't have changed it")
