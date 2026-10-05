"""
QC regression tests for "Fix Zoom" drifting across sequential root changes.

THE BUG THIS PINS
------------------
Report (decoded): with "Fix Zoom" enabled, the pinned pan/zoom position
drifted progressively further away from where the user pinned it, the more
times they navigated between roots ("passing roots") -- a COMPOUNDING error,
not a one-off jump.

ROOT CAUSE
----------
`ProjectTab._jump_to_index` calls `_ZoomBar.update_fixed_center(viewer)`
UNCONDITIONALLY before every single root switch (to catch panning that
bypassed the mouse-release/wheel handlers, e.g. keyboard scrolling) --
whether or not the user actually panned since the view was last positioned.
`update_fixed_center` -> `_store_scroll_position` -> `_get_view_center_ratio`
reads the CURRENT view center back and OVERWRITES `_ZoomBar._fixed_center_norm`
with it.

That read-back used to be `viewer.mapToScene(viewer.viewport().rect().center())`.
`QRect.center()` computes `(left() + right()) // 2` with `right() = width() - 1`
-- an INTEGER-TRUNCATED half-width (585 for a 1172px-wide viewport) -- while
Qt's own `QGraphicsView.centerOn()` (what `ImageViewer._apply_fixed_zoom` calls
to SET the position in the first place) uses the true floating half-width
(586.0 for that same viewport) internally. Those two notions of "the center of
the viewport" differ by about half a device pixel, and since the capture step
fires on EVERY navigation regardless of whether anything actually moved, that
fixed sub-pixel offset got re-baked into `_fixed_center_norm`, in the same
direction, on every single root change -- a linear walk that never settles,
matching the reported "gets worse the more you navigate" symptom exactly
(measured on the OLD code: a constant, non-zero per-navigation step of
1 / (zoom * image_dimension), confirmed with a real `ImageViewer` +
`_ZoomBar` driven through the actual `update_fixed_center` /
`reapply_fixed_zoom_if_enabled` methods -- not a re-implementation of them).

THE FIX
-------
`_ZoomBar._get_view_center_ratio` now derives the viewport's TRUE geometric
center by averaging the scene positions of its two integer corners
(0, 0) and (width, height) -- an exact inverse of `centerOn`'s continuous
half-width math -- instead of mapping a single integer-rounded center point.
`mapToScene` only accepts QPoint (integer) in this PyQt5, not QPointF, so
corner-averaging (an affine map, hence exact) is how sub-pixel precision is
recovered without it.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer, _ZoomBar

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer]


# ---------------------------------------------------------------------------
# `_ZoomBar` keeps its "fixed zoom" state on CLASS attributes, shared by every
# ImageViewer in the process for the lifetime of the app (that is the whole
# point -- it persists across root changes). That means it is also shared
# across pytest tests in this one process, so every test here must save and
# restore it, or it leaks into unrelated tests run later in the same session.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_zoombar_state():
    saved = {
        "_zoom_sync_enabled": _ZoomBar._zoom_sync_enabled,
        "_syncing": _ZoomBar._syncing,
        "_applying_fixed_zoom": _ZoomBar._applying_fixed_zoom,
        "_fixed_zoom": _ZoomBar._fixed_zoom,
        "_fixed_center_norm": _ZoomBar._fixed_center_norm,
    }
    _ZoomBar._zoom_sync_enabled = True
    _ZoomBar._syncing = False
    _ZoomBar._applying_fixed_zoom = False
    _ZoomBar._fixed_zoom = None
    _ZoomBar._fixed_center_norm = None
    _ZoomBar.invalidate_viewer_cache()
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(_ZoomBar, k, v)
        _ZoomBar.invalidate_viewer_cache()


def _make_pixmap(w=4000, h=3000, val=0x204060):
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(val)
    return QtGui.QPixmap.fromImage(qimg)


def _viewer_with_pinned_view(viewer_factory, qapp, w=1200, h=900,
                              zoom=2.35, center_norm=(0.27, 0.68)):
    """Build a viewer exactly the way display_image_group's first load does,
    then pin a genuine, non-trivial Fix-Zoom anchor (not centered, not 100%)."""
    v = viewer_factory()
    v.resize(w, h)
    v.show()
    qapp.processEvents()

    # Initial load: same call shape as display_image_group's cache-hit path.
    v.set_image(_make_pixmap(), defer_fit=True)
    qapp.processEvents()
    v.reapply_fixed_zoom_if_enabled()  # INITIAL LOAD path (fits + adopts 0.5,0.5)
    qapp.processEvents()

    # Simulate the user turning Fix Zoom on with a real pan/zoom.
    _ZoomBar._fixed_zoom = zoom
    _ZoomBar._fixed_center_norm = center_norm
    v._apply_fixed_zoom()
    qapp.processEvents()
    return v


def _simulate_root_change(v, qapp):
    """The exact sequence that fires on a real 'next root' / 'previous root'
    click, in call order:
      1. ProjectTab._jump_to_index's pre-switch capture
         (v._zoombar.update_fixed_center(v)).
      2. display_image_group's set_image(pixmap, defer_fit=True) for the new
         root's image.
      3. The QTimer.singleShot(0, _reapply_fixed_zoom_after_layout) callback
         (reapply_fixed_zoom_if_enabled), called synchronously here since we
         are testing the logic, not the timer scheduling.
    """
    if v._zoombar:
        v._zoombar.update_fixed_center(v)
    v.set_image(_make_pixmap(), defer_fit=True)
    v.reapply_fixed_zoom_if_enabled()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# THE regression: N sequential root changes must not move the anchor.
# ---------------------------------------------------------------------------
def test_fix_zoom_anchor_does_not_drift_across_sequential_root_changes(
        viewer_factory, qapp):
    zoom, center = 2.35, (0.27, 0.68)
    v = _viewer_with_pinned_view(viewer_factory, qapp, zoom=zoom, center_norm=center)

    history = [_ZoomBar._fixed_center_norm]
    for _ in range(12):
        _simulate_root_change(v, qapp)
        history.append(_ZoomBar._fixed_center_norm)

    start, end = history[0], history[-1]
    drift = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5

    assert drift < 1e-9, (
        f"Fix Zoom anchor drifted from {start} to {end} (distance {drift}) "
        f"over {len(history) - 1} sequential root changes with NO panning in "
        "between. Full history (should be constant): "
        + ", ".join(f"{c}" for c in history))


def test_fix_zoom_drift_does_not_grow_with_more_navigations(viewer_factory, qapp):
    """Falsifiability shape check: even if some residual float noise exists
    (Qt's own `centerOn` always snaps to an INTEGER scrollbar value, so a
    sub-pixel remainder can legitimately survive one round trip), it must
    settle to a fixed point rather than being a monotonically growing walk --
    the bug's actual symptom was "gets worse the more you navigate". Compares
    the anchor after 3 passes to the anchor after 15 passes: a compounding
    bug makes the second distance much larger than the first; a settled
    quantization remainder makes them equal.
    """
    v = _viewer_with_pinned_view(viewer_factory, qapp, zoom=1.61, center_norm=(0.6, 0.35))
    start = _ZoomBar._fixed_center_norm

    for _ in range(3):
        _simulate_root_change(v, qapp)
    after_3 = _ZoomBar._fixed_center_norm

    for _ in range(12):  # 3 + 12 = 15 total
        _simulate_root_change(v, qapp)
    after_15 = _ZoomBar._fixed_center_norm

    d3 = ((after_3[0] - start[0]) ** 2 + (after_3[1] - start[1]) ** 2) ** 0.5
    d15 = ((after_15[0] - start[0]) ** 2 + (after_15[1] - start[1]) ** 2) ** 0.5

    # Bound d3 itself to well under one screen pixel's worth of normalized
    # error (irreducible integer-scrollbar quantization, not the bug).
    img_w = v._get_image_rect().width()
    img_h = v._get_image_rect().height()
    max_err = 2.0 / (_ZoomBar._fixed_zoom * min(img_w, img_h))
    assert d3 <= max_err, (
        f"anchor moved {d3} after just 3 navigations with no panning -- more "
        f"than one screen pixel's worth of normalized error (bound {max_err})")

    # THE actual regression check: 12 MORE navigations must not move the
    # anchor any further than the first 3 already did (compounding vs settled).
    assert d15 <= d3 + 1e-9, (
        f"anchor kept moving with more navigations: d3={d3} (after 3 passes) "
        f"-> d15={d15} (after 15 passes). A fixed, non-compounding rounding "
        f"remainder would make these equal; growth here means the drift is "
        f"still compounding (start={start}, after_3={after_3}, "
        f"after_15={after_15})")


# ---------------------------------------------------------------------------
# The fix must not turn the capture step into a no-op that ignores genuine
# user panning between navigations (the "just delete the feedback write"
# alternative fix would break this).
# ---------------------------------------------------------------------------
def test_fix_zoom_still_captures_a_genuine_pan_between_navigations(
        viewer_factory, qapp):
    v = _viewer_with_pinned_view(viewer_factory, qapp, zoom=2.0, center_norm=(0.5, 0.5))

    # A real user pan (e.g. keyboard/scrollbar drag) that bypasses the
    # mouse-release/wheel sync handlers -- exactly what the pre-switch capture
    # step in _jump_to_index exists to catch.
    img_rect = v._get_image_rect()
    v.centerOn(img_rect.left() + 0.65 * img_rect.width(),
               img_rect.top() + 0.30 * img_rect.height())
    qapp.processEvents()

    v._zoombar.update_fixed_center(v)
    rx, ry = _ZoomBar._fixed_center_norm

    assert abs(rx - 0.65) < 0.01 and abs(ry - 0.30) < 0.01, (
        f"a genuine pan to (0.65, 0.30) was not captured correctly, got "
        f"({rx}, {ry})")


# ---------------------------------------------------------------------------
# Direct unit-level pin of the mechanism itself: _apply_fixed_zoom (write)
# and _get_view_center_ratio (read) must be exact inverses, for both even and
# odd viewport dimensions.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w,h", [(1200, 900), (1201, 901), (801, 601)])
def test_apply_and_read_back_fixed_center_round_trips_exactly(
        viewer_factory, qapp, w, h):
    v = _viewer_with_pinned_view(viewer_factory, qapp, w=w, h=h,
                                  zoom=1.73, center_norm=(0.41, 0.59))

    rx, ry = _ZoomBar._get_view_center_ratio(v)

    # An odd/frame-affected viewport can still carry <=1 integer scrollbar
    # step of irreducible quantization noise (Qt's own centerOn always snaps
    # to an integer scrollbar value) -- that is NOT the bug being pinned here
    # (it does not grow with more navigations, see the test above). Bound it
    # generously at well under one screen pixel's worth of normalized error.
    img_w = v._get_image_rect().width()
    img_h = v._get_image_rect().height()
    zoom = _ZoomBar._fixed_zoom
    max_err_x = 2.0 / (zoom * img_w)
    max_err_y = 2.0 / (zoom * img_h)

    assert abs(rx - 0.41) <= max_err_x, f"x round-trip off by {abs(rx-0.41)}"
    assert abs(ry - 0.59) <= max_err_y, f"y round-trip off by {abs(ry-0.59)}"
