"""QC regression tests for "edit vertices crashes the program when I have
too many vertices."

THE BUG: dragging a single vertex handle went through
`ImageViewer._apply_vertex_update` -> the `EditablePolygonItem.polygon`
SETTER on every throttled frame (on_vertex_moved already limits this to
60/sec, but does not limit polygon SIZE). That setter -- and
`_apply_vertex_update`'s own point-copy loop before it -- do THREE full
O(n) passes over the polygon for moving exactly one point:

  1. `_apply_vertex_update`: `for i in range(poly.count()): new_poly.append(...)`
     -- a Python-level rebuild of the entire point list.
  2. `_invalidate_geometry_cache`: `for i in range(n): p = poly.at(i)` --
     rebuilding the `_pts_np` numpy array one Python-level call at a time.
  3. `_invalidate_geometry_cache`: `path.addPolygon(poly)` -- rebuilding the
     cached QPainterPath used by `shape()`/hit-testing.

Vertex HANDLES are already capped at 100 regardless of the real point count
(`start_vertex_editing`'s MAX_HANDLES, so dragging one handle always stays
responsive on its own) -- but the underlying polygon a handle drags can
still have far more real vertices than that (a detailed shapefile import
easily reaches the tens or hundreds of thousands; the real project measured
elsewhere in this codebase averaged 433 vertices/polygon with a max of
2312, and that was already enough to matter for painting alone). All three
of the passes above scale with the REAL point count, not the 100-handle
cap, and all three ran on the GUI thread on every throttled frame for as
long as the drag continued -- for a big enough polygon, each frame becomes
slow enough that new drag events queue up faster than they drain, and the
UI never recovers: the freeze the report describes as a crash.

THE FIX: `EditablePolygonItem._set_vertex_point(index, new_point)` replaces
all three passes with O(1) work per frame -- an in-place `QPolygonF`
`__setitem__`, an in-place `_pts_np` write, and invalidating (not
rebuilding) the decimation cache, which the next paint recomputes with
vectorized numpy rather than a Python loop. `_cached_shape`/`_cached_brect`
are deliberately left stale during the drag (exactly like the existing
`is_moving` fast path for a whole-polygon drag already does) and brought
back in sync with ONE full `_invalidate_geometry_cache()` call in
`ImageViewer.stop_vertex_editing`, once per edit SESSION rather than once
per FRAME. `stop_vertex_editing` was also fixed to apply, rather than
silently discard, a throttled update still pending when the drag ends.
"""
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .test_export_and_ax_regressions import _func_tree, _names_in, _calls_in
from ..image_viewer import EditablePolygonItem, ImageViewer

pytestmark = [pytest.mark.viewer, pytest.mark.polygons]


def _big_ring(n, cx=1000.0, cy=1000.0, r=500.0):
    """A polygon with `n` real vertices -- big enough to force _pts_np to
    exist (n >= EditablePolygonItem._SIMPLIFY_MIN_POINTS)."""
    import math
    pts = [QtCore.QPointF(cx + r * math.cos(2 * math.pi * i / n),
                          cy + r * math.sin(2 * math.pi * i / n))
           for i in range(n)]
    return QtGui.QPolygonF(pts)


# ---------------------------------------------------------------------------
# _set_vertex_point: correctness
# ---------------------------------------------------------------------------
def test_set_vertex_point_updates_the_polygon_in_place():
    item = EditablePolygonItem(_big_ring(200), "crown")
    new_pt = QtCore.QPointF(12345.0, 6789.0)
    item._set_vertex_point(5, new_pt)

    got = item.polygon.at(5)
    assert got.x() == pytest.approx(12345.0)
    assert got.y() == pytest.approx(6789.0)
    # every other point must be untouched
    assert item.polygon.at(4) != new_pt
    assert item.polygon.count() == 200


def test_set_vertex_point_updates_pts_np_in_place():
    item = EditablePolygonItem(_big_ring(200), "crown")
    assert item._pts_np is not None, "200 points must be above _SIMPLIFY_MIN_POINTS"
    new_pt = QtCore.QPointF(-500.0, -600.0)
    item._set_vertex_point(3, new_pt)
    assert item._pts_np[3, 0] == pytest.approx(-500.0)
    assert item._pts_np[3, 1] == pytest.approx(-600.0)


def test_set_vertex_point_leaves_cached_shape_and_brect_stale():
    """Deliberate: matching the existing is_moving fast path. A full resync
    only happens once, via _invalidate_geometry_cache (stop_vertex_editing),
    not on every frame."""
    item = EditablePolygonItem(_big_ring(200), "crown")
    shape_before = item._cached_shape
    brect_before = item._cached_brect
    item._set_vertex_point(0, QtCore.QPointF(99999.0, 99999.0))
    assert item._cached_shape is shape_before
    assert item._cached_brect is brect_before


def test_invalidate_geometry_cache_resyncs_after_vertex_edits():
    """The deferred full rebuild (what stop_vertex_editing calls) must bring
    boundingRect/shape back in line with the moved point."""
    item = EditablePolygonItem(_big_ring(200), "crown")
    item._set_vertex_point(0, QtCore.QPointF(99999.0, 99999.0))
    item._invalidate_geometry_cache()

    br = item.boundingRect()
    assert br.right() >= 99999.0 - 100, (
        "boundingRect was not resynced to include the moved vertex")


def test_set_vertex_point_out_of_range_falls_back_safely():
    """An out-of-range index must not corrupt the polygon or raise past this
    call -- the fallback path in the docstring."""
    item = EditablePolygonItem(_big_ring(10), "crown")
    before = item.polygon.count()
    item._set_vertex_point(999, QtCore.QPointF(1.0, 1.0))
    assert item.polygon.count() == before


# ---------------------------------------------------------------------------
# The actual regression: this must stay fast regardless of vertex count
# ---------------------------------------------------------------------------
def test_dragging_a_vertex_on_a_huge_polygon_stays_fast(qapp):
    """THE regression, stated as a timing bound generous enough to never be
    flaky on real hardware but tight enough that the OLD three-O(n)-passes-
    per-frame code would blow through it by orders of magnitude: 50,000
    vertices x 300 simulated drag frames is 15,000,000 point-touches for the
    OLD per-frame full rebuild (three passes: rebuild list, rebuild
    _pts_np, rebuild path) vs 300 O(1) writes for the fix.
    """
    n = 50_000
    item = EditablePolygonItem(_big_ring(n), "crown")
    assert item._pts_np is not None and len(item._pts_np) == n

    t0 = time.perf_counter()
    for i in range(300):
        idx = i % n
        item._set_vertex_point(idx, QtCore.QPointF(1000.0 + i, 1000.0 + i))
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, (
        f"300 vertex updates on a {n}-vertex polygon took {elapsed:.2f}s -- "
        "each _set_vertex_point call must be O(1), not O(n); this is the "
        "'edit vertices crashes with too many vertices' regression")


def test_finishing_a_huge_vertex_edit_resyncs_once_not_per_frame():
    """The deferred full rebuild itself (paid once, at drag-end) must also
    complete promptly for a large polygon -- it is O(n) by nature (it is the
    per-FRAME repetition that was the bug, not this one-time cost), but a
    single pass over even a very large polygon should still be well under a
    second on any real hardware."""
    n = 200_000
    item = EditablePolygonItem(_big_ring(n), "crown")
    item._set_vertex_point(0, QtCore.QPointF(0.0, 0.0))

    t0 = time.perf_counter()
    item._invalidate_geometry_cache()
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"one-time resync of a {n}-vertex polygon took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# AST-level: the fix's shape is actually wired in, not just present
# ---------------------------------------------------------------------------
def test_apply_vertex_update_calls_set_vertex_point_not_a_rebuild_loop():
    assert _calls_in(ImageViewer._apply_vertex_update, "_set_vertex_point"), (
        "_apply_vertex_update must delegate to the O(1) _set_vertex_point, "
        "not rebuild the whole polygon per frame")


def test_stop_vertex_editing_resyncs_the_geometry_cache():
    assert _calls_in(ImageViewer.stop_vertex_editing, "_invalidate_geometry_cache"), (
        "stop_vertex_editing must trigger the deferred full cache rebuild "
        "once editing ends, or boundingRect/shape stay stale forever")


def test_stop_vertex_editing_applies_the_last_pending_update():
    """Guards the adjacent correctness fix: a throttled update still
    pending when the drag ends must be applied, not discarded."""
    names = _names_in(ImageViewer.stop_vertex_editing)
    assert "_apply_vertex_update" in names
    assert "pending" in names


# ---------------------------------------------------------------------------
# Functional: dragging a vertex on a huge polygon end-to-end through the
# real ImageViewer methods (not just the item in isolation)
# ---------------------------------------------------------------------------
class _FakeViewerHost:
    """Just enough of ImageViewer's own state for start/stop/apply to run
    without a live QGraphicsScene/View -- vertex editing itself doesn't
    paint anything synchronously, so a real scene isn't required to
    exercise the update path, only to add/remove handle items."""


def test_full_drag_session_on_a_huge_polygon_via_real_viewer_methods(qapp):
    from PyQt5 import QtWidgets

    scene = QtWidgets.QGraphicsScene()
    n = 20_000
    item = EditablePolygonItem(_big_ring(n), "crown")
    scene.addItem(item)

    viewer = ImageViewer.__new__(ImageViewer)
    viewer._scene = scene
    viewer._vertex_editing_item = None
    viewer._vertex_handles = []
    viewer._vertex_resampled = False
    viewer._original_polygon = None

    try:
        t0 = time.perf_counter()
        viewer.start_vertex_editing(item)
        assert item._vertex_editing_active is True
        assert len(viewer._vertex_handles) == 100, "MAX_HANDLES cap must still apply"
        moved_poly_index = getattr(viewer._vertex_handles[0], "_poly_index", 0)

        # Simulate a real drag: many rapid moves of the SAME handle.
        for i in range(200):
            viewer._apply_vertex_update(0, QtCore.QPointF(2000.0 + i, 2000.0))

        viewer.stop_vertex_editing()
        elapsed = time.perf_counter() - t0

        assert item._vertex_editing_active is False
        assert elapsed < 3.0, (
            f"a full start->200-frame-drag->stop session on a {n}-vertex "
            f"polygon took {elapsed:.2f}s")
        # The final position must have actually landed (not the discarded
        # snap-back this session also fixed) -- last write was i=199.
        final_pt = item.polygon.at(moved_poly_index)
        assert final_pt.x() == pytest.approx(2199.0)
        assert final_pt.y() == pytest.approx(2000.0)
    finally:
        scene.clear()
