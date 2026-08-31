"""
QC regression tests: the in-progress polygon/point/line drawing preview
("rubber band") disappearing while the user draws, worse the higher the
zoom -- reported as "whenever I press shift and draw the line disappears
depending on the zoom (If the zoom is too high the line just disappears)".

Shift+Left-click is literally the ONLY mouse gesture that starts freehand
drawing (image_viewer.py's mousePressEvent) -- a plain left-click without
Shift pans or selects instead -- so "press shift and draw" just means
"draw"; the report is not describing a Shift-specific code path.

THE ACTUAL BUG
---------------
`temp_drawing_item`'s pen is created COSMETIC -- pen.setCosmetic(True),
width 2.0 -- specifically so it renders as a constant 2 DEVICE pixels at
ANY zoom (mousePressEvent's own comment: "so the width never has to be
recomputed when the user zooms mid-draw"). But FOUR separate call sites
that run *during* drawing -- update_temp_drawing() (called on every added
vertex and, in "polygon" mode, on every hover mouse-move too),
zoom_out_to_fit(), the wheel-zoom handler (_apply_wheel_zoom), and
start_drawing_with_group_name() itself -- re-derived the pen width as
`2 / view_scale` on every call, a formula that is only correct for a
NON-cosmetic (scene-space) pen. Applied to the actually-cosmetic pen, the
resulting on-screen width becomes `2 / scale` DEVICE pixels: it shrinks
towards zero as the user zooms IN, and well before this app's 50x zoom
cap it rounds down to sub-pixel/invisible. (It also comes out visibly
*too thick* at less-than-1x zoom, for the same reason -- confirmed below
by the very first Shift-click, before any further zoom change.)

Fixed in image_viewer.py: all four sites now leave the cosmetic pen's
width alone; start_drawing_with_group_name's own pen is now created
cosmetic too (previously it was the one non-cosmetic pen in the codebase,
sized once via `desired_width/scale` at creation only).
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

pytestmark = [pytest.mark.viewer]


def _viewer_with_image(viewer_factory, w=1200, h=900, img=(2000, 1500)):
    v = viewer_factory()
    v.resize(w, h)
    v.show()
    qimg = QtGui.QImage(img[0], img[1], QtGui.QImage.Format_RGB32)
    qimg.fill(0x204060)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    QtWidgets.QApplication.processEvents()
    return v


def _shift_click(viewer, x, y):
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(x, y),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.ShiftModifier,
    )
    viewer.mousePressEvent(ev)


def _hover_move(viewer, x, y):
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(x, y),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    viewer.mouseMoveEvent(ev)


def _effective_device_width(viewer):
    pen = viewer.temp_drawing_item.pen()
    if pen.isCosmetic():
        return pen.widthF()
    return pen.widthF() * viewer.get_current_scale_factor()


def test_shift_click_starts_drawing_with_a_cosmetic_pen(viewer_factory):
    v = _viewer_with_image(viewer_factory)
    _shift_click(v, 100, 100)
    assert v.drawing is True
    assert v.temp_drawing_item is not None
    pen = v.temp_drawing_item.pen()
    assert pen.isCosmetic() is True
    assert pen.widthF() == pytest.approx(2.0)


@pytest.mark.parametrize("scale", [1.0, 2.0, 5.0, 20.0, 50.0])
def test_preview_line_width_survives_zoom_while_drawing(viewer_factory, scale):
    v = _viewer_with_image(viewer_factory)
    _shift_click(v, 100, 100)
    assert v.drawing is True

    v.resetTransform()
    v.scale(scale, scale)
    assert v.get_current_scale_factor() == pytest.approx(scale, rel=1e-3)

    _hover_move(v, 150, 150)

    width = _effective_device_width(v)
    assert width == pytest.approx(2.0, rel=1e-6), (
        f"at view scale {scale}x the preview line's effective on-screen "
        f"width was {width} device px -- it must stay a constant 2px, "
        f"not shrink with zoom")


@pytest.mark.parametrize("scale", [1.0, 10.0, 50.0])
def test_update_temp_drawing_never_touches_pen_width(viewer_factory, scale):
    for mode in ("polygon", "rectangle", "circle", "point"):
        v = _viewer_with_image(viewer_factory)
        v.drawing_mode = mode
        _shift_click(v, 100, 100)
        v.resetTransform()
        v.scale(scale, scale)
        v.currentPolygon.append(QtCore.QPointF(120, 140))
        v._shape_start_point = QtCore.QPointF(100, 100)
        v._shape_end_point = QtCore.QPointF(140, 160)

        v.update_temp_drawing()

        width = _effective_device_width(v)
        assert width == pytest.approx(2.0, rel=1e-6), (
            f"mode={mode} scale={scale}: preview pen width corrupted by "
            f"update_temp_drawing (got {width} effective device px)")


def test_zoom_out_to_fit_does_not_shrink_the_preview_pen(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(4000, 3000))
    _shift_click(v, 500, 500)
    v.resetTransform()
    v.scale(30.0, 30.0)

    v.zoom_out_to_fit()

    width = _effective_device_width(v)
    assert width == pytest.approx(2.0, rel=1e-6)


def test_wheel_zoom_in_while_drawing_does_not_shrink_the_preview_pen(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(4000, 3000))
    _shift_click(v, 500, 500)

    before = v.get_current_scale_factor()
    for _ in range(6):
        wheel_ev = QtGui.QWheelEvent(
            QtCore.QPointF(400, 300),
            v.mapToGlobal(QtCore.QPoint(400, 300)),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 480),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.NoScrollPhase,
            False,
        )
        v.wheelEvent(wheel_ev)
        QtWidgets.QApplication.processEvents()

    after = v.get_current_scale_factor()
    assert after > before * 2, "test setup sanity: the view should have actually zoomed in"

    width = _effective_device_width(v)
    assert width == pytest.approx(2.0, rel=1e-6), (
        f"after zooming in to {after:.1f}x via the real wheel-zoom path, "
        f"preview pen effective width was {width} device px")


@pytest.mark.parametrize("scale", [1.0, 20.0, 50.0])
def test_start_drawing_with_group_name_pen_survives_zoom(viewer_factory, scale):
    """The Polygon Manager "Draw"/"Edit Group" entry point
    (start_drawing_with_group_name) used to build the ONE non-cosmetic
    temp_drawing_item pen in the codebase, sized once via
    `desired_width / scale` at the CURRENT zoom -- so it stayed a
    reasonable width until the user changed zoom afterwards, at which
    point every OTHER call site's `2/scale` recompute (all operating on a
    pen that was, everywhere else, cosmetic) mismatched it too. Confirm
    it's now cosmetic and fixed-width like every other entry point,
    regardless of the zoom level active when drawing starts."""
    v = _viewer_with_image(viewer_factory)
    v.resetTransform()
    v.scale(scale, scale)

    v.start_drawing_with_group_name("some_group")

    assert v.drawing is True
    pen = v.temp_drawing_item.pen()
    assert pen.isCosmetic() is True
    assert pen.widthF() == pytest.approx(2.0, rel=1e-6)
