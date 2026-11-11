"""
Cancelling the "Polygon Locked" prompt left the pan stuck on.

THE BUG
-------
Clicking a locked polygon opens a modal QMessageBox from inside the item's
mousePressEvent. The item then ignores the press, so QGraphicsView's
ScrollHandDrag reads it as a click on the background and starts hand-scrolling.
The modal box eats the matching release, so nothing ever ended that pan: the
closed-hand cursor stuck, and every later hover moved the image, button held
or not.

These tests drive real mouse events through a real ImageViewer. Only
QMessageBox.exec_ is replaced, by a stand-in that clicks the button the
test names.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest

from canopie.image_viewer import ImageViewer

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


def _answer_prompt_with(monkeypatch, label):
    def fake_exec(box):
        for b in box.buttons():
            if b.text() == label:
                b.click()
        return 0
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", fake_exec)


@pytest.fixture
def locked_viewer(qapp, monkeypatch):
    # Zoom sync is class-wide state over every viewer ever made; earlier
    # tests' deleted viewers would be touched on a pan. Not under test here.
    from canopie.image_viewer import _ZoomBar
    monkeypatch.setattr(_ZoomBar, "_zoom_sync_enabled", False)
    v = ImageViewer()
    v.resize(300, 300)
    pm = QtGui.QPixmap(2000, 2000)
    pm.fill(QtCore.Qt.gray)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v._scene.setSceneRect(0, 0, 2000, 2000)
    v._empty = False            # as after a real image load
    v.polygons = []
    v.add_polygon_to_scene(QtGui.QPolygonF([
        QtCore.QPointF(900, 900), QtCore.QPointF(1100, 900),
        QtCore.QPointF(1100, 1100), QtCore.QPointF(900, 1100)]), name="crown")
    v.set_polygons_locked(True)
    v.show()
    qapp.processEvents()
    v.centerOn(1000, 1000)
    qapp.processEvents()
    yield v
    v.hide()
    v._scene.clear()
    v.deleteLater()


def _hover(v, pos):
    """A move with NO button held, i.e. the user just moving the mouse."""
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(pos),
                           QtCore.Qt.NoButton, QtCore.Qt.NoButton,
                           QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(v.viewport(), ev)


def _click_polygon_and_hover(v, qapp):
    pt = v.mapFromScene(QtCore.QPointF(1000, 1000))
    h0, s0 = v.horizontalScrollBar().value(), v.verticalScrollBar().value()
    # press only: in the app the modal prompt swallows the release
    QTest.mousePress(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pt)
    _hover(v, pt + QtCore.QPoint(-80, -40))
    qapp.processEvents()
    return h0, s0


@pytest.mark.parametrize("answer", ["Cancel", "Unlock This", "Unlock Group"])
def test_lock_prompt_leaves_canvas_idle(locked_viewer, qapp, monkeypatch, answer):
    v = locked_viewer
    _answer_prompt_with(monkeypatch, answer)
    item = v.get_all_polygons()[0]
    p0 = item.pos()

    h0, s0 = _click_polygon_and_hover(v, qapp)

    assert (v.horizontalScrollBar().value(), v.verticalScrollBar().value()) == (h0, s0), (
        f"after '{answer}' on the lock prompt, hovering scrolled the image: "
        "the pan is stuck on")
    assert v.viewport().cursor().shape() != QtCore.Qt.ClosedHandCursor, (
        "the closed-hand (panning) cursor stuck after the lock prompt")
    assert v._is_panning is False and v.left_button_pressed is False
    assert v._scene.mouseGrabberItem() is None, (
        "an item still grabs the mouse; it would follow the cursor")
    assert item.pos() == p0, "the polygon moved without a drag"
    assert item.is_locked is (answer == "Cancel")


def test_after_cancel_background_pan_still_works(locked_viewer, qapp, monkeypatch):
    """The canvas is back to its default interaction: a real background drag pans."""
    v = locked_viewer
    _answer_prompt_with(monkeypatch, "Cancel")
    _click_polygon_and_hover(v, qapp)

    bg = v.mapFromScene(QtCore.QPointF(800, 800))
    h0 = v.horizontalScrollBar().value()
    QTest.mousePress(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, bg)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(bg + QtCore.QPoint(-50, 0)),
                           QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(v.viewport(), ev)
    QTest.mouseRelease(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                       bg + QtCore.QPoint(-50, 0))
    qapp.processEvents()
    assert v.horizontalScrollBar().value() == h0 + 50, "a background drag no longer pans"
    assert v._is_panning is False


def test_unlocked_polygon_drags_on_the_next_click(locked_viewer, qapp, monkeypatch):
    v = locked_viewer
    _answer_prompt_with(monkeypatch, "Unlock This")
    item = v.get_all_polygons()[0]
    _click_polygon_and_hover(v, qapp)
    assert item.is_locked is False

    p0 = item.pos()
    pt = v.mapFromScene(QtCore.QPointF(1000, 1000))
    QTest.mousePress(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pt)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(pt + QtCore.QPoint(30, 0)),
                           QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(v.viewport(), ev)
    QTest.mouseRelease(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                       pt + QtCore.QPoint(30, 0))
    qapp.processEvents()
    assert item.pos() != p0, "an unlocked polygon did not drag on the next click"
