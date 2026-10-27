"""
QC tests: smooth vertex dragging in vertex-edit mode (ImageViewer._vertex_drag_begin /
_apply_vertex_update).

A dense outline (crown maps, Smart Canopy outlines: hundreds of vertices) only
gets 100 handles, and dragging one used to move that single vertex -- its
neighbours, handle-less, stayed put and the outline grew a spike. Now a drag
pulls every vertex within a radius (screen px, [ / ] to change) along with a
smooth falloff; Shift+drag still moves the one vertex.
"""
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from ..image_viewer import VERTEX_SOFT_RADIUS_SCREEN_PX

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]

N, CX, CY, RAD = 360, 200.0, 200.0, 100.0


@pytest.fixture()
def editing(viewer_factory):
    v = viewer_factory()
    v.resize(600, 600)
    v.show()
    qimg = QtGui.QImage(400, 400, QtGui.QImage.Format_RGB32)
    qimg.fill(0x305020)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    v.resetTransform()
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    circle = QtGui.QPolygonF([QtCore.QPointF(CX + RAD * np.cos(a), CY + RAD * np.sin(a)) for a in ang])
    item = v.add_polygon_to_scene(circle, "crown")
    QtWidgets.QApplication.processEvents()
    v.start_vertex_editing(item)
    assert len(v._vertex_handles) == 100, "precondition: a dense outline gets 100 handles"
    return v, item


def _xy(item):
    return np.array([[p.x(), p.y()] for p in item.polygon])


def _handle(v, poly_index):
    return next(h for h in v._vertex_handles if h._poly_index == poly_index)


def _drag(v, handle, dx, single=False):
    v._vertex_drag_begin(handle, single=single)
    p = handle.pos()
    v._apply_vertex_update(handle.index, QtCore.QPointF(p.x() + dx, p.y()))


def test_a_drag_pulls_the_outline_around_it_smoothly(editing):
    v, item = editing
    before = _xy(item)
    _drag(v, _handle(v, 0), 30.0)
    moved = _xy(item) - before
    r = VERTEX_SOFT_RADIUS_SCREEN_PX                 # scale 1: screen px == scene units
    dist = np.hypot(*(before - before[0]).T)
    assert moved[0, 0] == pytest.approx(30.0)
    near = (dist > 0) & (dist < r)
    assert near.sum() >= 10, "several vertices, most without a handle, are within reach"
    assert np.all((moved[near, 0] > 0) & (moved[near, 0] < 30.0))
    order = np.argsort(dist[near])
    assert np.all(np.diff(moved[near, 0][order]) <= 1e-9), "the pull falls off with distance"
    assert np.allclose(moved[dist >= r], 0.0), "vertices beyond the radius do not move"


def test_neighbouring_handles_follow_their_vertices(editing):
    v, item = editing
    h_near = _handle(v, 7)                           # ~12 px from vertex 0
    before = h_near.pos()
    _drag(v, _handle(v, 0), 30.0)
    vtx = item.mapToScene(item.polygon.at(7))
    assert h_near.pos().x() > before.x() and h_near.pos() == vtx


def test_shift_drag_moves_one_vertex_only(editing):
    v, item = editing
    before = _xy(item)
    _drag(v, _handle(v, 0), 30.0, single=True)
    moved = np.abs(_xy(item) - before).sum(1)
    assert moved[0] > 0 and np.count_nonzero(moved) == 1


def test_the_reach_ring_shows_while_dragging(editing):
    v, item = editing
    _drag(v, _handle(v, 0), 10.0)
    assert v._vx_ring is not None and v._vx_ring.rect().width() == pytest.approx(2 * VERTEX_SOFT_RADIUS_SCREEN_PX)
    v._vertex_drag_end()
    assert v._vx_ring is None


def test_brackets_change_the_reach_down_to_one_vertex(editing):
    v, item = editing
    key = lambda k: v.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, k, QtCore.Qt.NoModifier))
    key(QtCore.Qt.Key_BracketRight)
    assert v._vertex_soft_radius_screen() > VERTEX_SOFT_RADIUS_SCREEN_PX
    for _ in range(30):
        key(QtCore.Qt.Key_BracketLeft)
    assert v._vertex_soft_radius_screen() == 0.0
    before = _xy(item)
    _drag(v, _handle(v, 0), 30.0)
    assert np.count_nonzero(np.abs(_xy(item) - before).sum(1)) == 1


def test_a_real_handle_drag_through_the_throttle_ends_smooth(editing):
    """Moving the handle itself (itemChange -> on_vertex_moved, throttled to 60/s) in quick steps, then
    releasing: the final position is applied with the smooth pull, not as a lone-vertex kink."""
    v, item = editing
    before = _xy(item)
    h = _handle(v, 0)
    v._vertex_drag_begin(h)
    p0 = h.pos()
    for k in range(1, 11):                              # faster than the throttle lets through
        h.setPos(QtCore.QPointF(p0.x() + 3.0 * k, p0.y()))
    v._vertex_drag_end()                                 # what the mouse release calls
    moved = _xy(item) - before
    assert moved[0, 0] == pytest.approx(30.0)
    assert moved[1, 0] == pytest.approx(30.0 * (1 - (np.hypot(*(before[1] - before[0])) /
                                                    VERTEX_SOFT_RADIUS_SCREEN_PX) ** 2) ** 2, rel=1e-6)


def test_finishing_the_edit_still_saves_it(editing):
    v, item = editing
    got = []
    item.polygon_modified.connect(lambda: got.append(True))
    _drag(v, _handle(v, 0), 30.0)
    v._vertex_drag_end()
    v.finish_vertex_editing()
    assert got and v._vertex_handles == [] and v._vertex_editing_item is None
