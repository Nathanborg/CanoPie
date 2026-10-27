"""
QC tests: the Smart Canopy brush (canopie/canopy_tracer.py + the
"smart_canopy" mode of ImageViewer).

Brush strokes on a crown must select that crown: checked against synthetic
crowns with known masks (round, elliptic, lobed, two touching crowns), then
end to end through a real ImageViewer driven by mouse and key events --
strokes accumulate, Backspace undoes one, Esc cancels, and only Enter commits
(polygon_drawn, which is what opens ProjectTab's name dialog).
"""
import numpy as np
import cv2
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from ..canopy_tracer import CanopyParams, select_canopy

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]

H, W = 600, 800
CROWNS = {                       # name: (centre, radius, mask builder)
    "round": ((180, 170), 70, lambda xx, yy: (xx - 180) ** 2 + (yy - 170) ** 2 <= 70 ** 2),
    "ellipse": ((520, 160), 85, lambda xx, yy: ((xx - 520) / 110.0) ** 2 + ((yy - 160) / 65.0) ** 2 <= 1),
    "lobed": ((250, 420), 75, lambda xx, yy: np.hypot(xx - 250, yy - 420)
              <= 75 * (1 + 0.18 * np.cos(5 * np.arctan2(yy - 420, xx - 250)))),
    "touchA": ((470, 470), 45, lambda xx, yy: (xx - 470) ** 2 + (yy - 470) ** 2 <= 45 ** 2),
    "touchB": ((555, 470), 42, lambda xx, yy: (xx - 555) ** 2 + (yy - 470) ** 2 <= 42 ** 2),
}
COLOURS = {"round": (60, 130, 50), "ellipse": (90, 150, 60), "lobed": (50, 115, 45),
           "touchA": (60, 135, 50), "touchB": (110, 160, 70)}


def _scene():
    """Brown textured soil with internally textured, soft-edged crowns (RGB uint8) + true masks."""
    rng = np.random.default_rng(3)
    img = np.dstack([c + 20 * rng.standard_normal((H, W)) for c in (140, 110, 80)]).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    yy, xx = np.mgrid[:H, :W]
    truth = {}
    for name, (_, _, build) in CROWNS.items():
        mask = build(xx, yy)
        m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)[..., None]
        tex = cv2.GaussianBlur((rng.standard_normal((H, W)) * 25).astype(np.float32), (0, 0), 2.5)[..., None]
        img = img * (1 - m) + (np.float32(COLOURS[name]) + tex) * m
        truth[name] = mask
    return np.clip(img, 0, 255).astype(np.uint8), truth


@pytest.fixture(scope="module")
def scene():
    return _scene()


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


def _zigzag(cx, cy, R):
    t = np.linspace(-0.45 * R, 0.45 * R, 30)
    return np.c_[cx + t, cy + 0.35 * R * np.sign(np.sin(np.linspace(0, 4 * np.pi, 30)))]


@pytest.mark.parametrize("name", ["round", "ellipse", "lobed"])
def test_painting_a_crown_selects_it(scene, name):
    img, truth = scene
    (cx, cy), R, _ = CROWNS[name]
    res = select_canopy(img, [(_zigzag(cx, cy, R), R / 5, False)])
    iou = _iou(res.mask((H, W)), truth[name])
    assert iou >= 0.88, f"{name}: IoU {iou:.3f} ({res.note})"


@pytest.mark.parametrize("name", ["round", "lobed"])
def test_a_click_with_a_big_brush_selects_the_crown(scene, name):
    img, truth = scene
    (cx, cy), R, _ = CROWNS[name]
    res = select_canopy(img, [(np.array([[cx + 4, cy - 3]]), R / 3, False)])
    iou = _iou(res.mask((H, W)), truth[name])
    assert iou >= 0.8, f"{name}: IoU {iou:.3f} ({res.note})"


def test_the_paint_is_always_inside_the_outline(scene):
    """A stroke reaching past the visible edge (the user saying 'this is crown too') is never cut off."""
    img, _ = scene
    stroke = np.c_[np.linspace(150, 270, 30), np.full(30, 170.0)]       # 20 px past the round crown's edge
    res = select_canopy(img, [(stroke, 6, False)])
    m = res.mask((H, W))
    pts = np.round(stroke).astype(int)
    assert cv2.dilate(m, np.ones((5, 5), np.uint8))[pts[:, 1], pts[:, 0]].all(), res.note


def test_touching_crowns_stay_separate_with_the_stop_mask(scene):
    img, truth = scene
    a = select_canopy(img, [(_zigzag(470, 470, 45), 9, False)])
    b = select_canopy(img, [(_zigzag(555, 470, 42), 8, False)], stop_mask=a.mask((H, W)))
    ma, mb = a.mask((H, W)), b.mask((H, W))
    assert (ma & mb).sum() <= 0.01 * mb.sum(), "the second crown took part of the first"
    assert _iou(ma, truth["touchA"]) >= 0.8, a.note
    assert _iou(mb, truth["touchB"]) >= 0.8, b.note


def test_an_erase_stroke_takes_its_band_out(scene):
    img, truth = scene
    (cx, cy), R, _ = CROWNS["round"]
    paint = (_zigzag(cx, cy, R), R / 5, False)
    cut = (np.c_[np.full(20, cx + 0.7 * R), np.linspace(cy - R, cy + R, 20)], 8, True)
    before = select_canopy(img, [paint]).mask((H, W))
    after = select_canopy(img, [paint, cut]).mask((H, W))
    assert after.sum() < before.sum()
    assert after[cy, int(cx + 0.7 * R)] == 0                    # the erased band is out
    assert after[cy, cx] == 1                                   # the crown is still there


def test_grow_bounds_how_far_past_the_paint_the_crown_goes(scene):
    """grow = 0 is exactly the paint; otherwise nothing is selected beyond grow x brush from it."""
    img, _ = scene
    (cx, cy), R, _ = CROWNS["round"]
    clicks = [(np.array([[cx + dx, cy + dy]]), 8.0, False) for dx, dy in ((0, 0), (20, 5), (-15, 18))]
    paint = np.zeros((H, W), np.uint8)
    for q, r, _ in clicks:
        cv2.circle(paint, tuple(int(v) for v in q[0]), 8, 1, -1)
    d = cv2.distanceTransform((paint == 0).astype(np.uint8), cv2.DIST_L2, 3)
    exact = select_canopy(img, clicks, CanopyParams(grow=0)).mask((H, W)) > 0
    assert exact[paint > 0].mean() >= 0.97                     # all the paint, joined into one shape ...
    assert d[exact].max() <= 12.0                               # ... by bridges between the clicks only
    for g in (0.5, 1.0, 2.0):
        m = select_canopy(img, clicks, CanopyParams(grow=g)).mask((H, W)) > 0
        assert d[m].max() <= g * 8.0 + 3.0, f"grow {g}: selected {d[m].max():.1f} px past the paint"


def test_many_clicks_inside_a_crown_converge_onto_it(scene):
    """The many-clicks workflow: each click refines the same crown, and with enough of them the outline
    covers the crown without spilling over."""
    img, truth = scene
    (cx, cy), R, _ = CROWNS["lobed"]
    rng = np.random.default_rng(1)
    clicks = []
    while len(clicks) < 12:
        q = np.array([cx, cy]) + rng.uniform(-0.8 * R, 0.8 * R, 2)
        if truth["lobed"][int(round(q[1])), int(round(q[0]))]:
            clicks.append((q[None, :], R / 6, False))           # the optimised brush; default grow
    ious = [_iou(select_canopy(img, clicks[:k]).mask((H, W)), truth["lobed"]) for k in (1, 4, 12)]
    assert ious[0] < ious[1] <= ious[2] + 0.02 and ious[2] >= 0.85, ious


def test_only_erasing_selects_nothing(scene):
    img, _ = scene
    res = select_canopy(img, [(np.array([[100.0, 100.0]]), 10, True)])
    assert len(res.polygon) == 0


def test_a_flat_image_gives_about_the_painted_area_not_a_blow_up():
    flat = np.full((300, 300, 3), 120, np.uint8)
    res = select_canopy(flat, [(np.array([[150.0, 150.0]]), 10, False)])
    assert len(res.polygon) >= 3
    assert res.area_px <= np.pi * 40 ** 2, res.note


def test_single_band_and_16_bit_inputs_work(scene):
    img, truth = scene
    grey16 = (img.mean(2) * 200).astype(np.uint16)[..., None]   # an HxWx1 radiance-like band
    (cx, cy), R, _ = CROWNS["round"]
    res = select_canopy(grey16, [(_zigzag(cx, cy, R), R / 5, False)])
    assert _iou(res.mask((H, W)), truth["round"]) >= 0.8, res.note


# ------------------------------------------------------------------------------
# End to end through a real ImageViewer
# ------------------------------------------------------------------------------

def _viewer_with(viewer_factory, rgb):
    v = viewer_factory()
    v.resize(1000, 800)
    v.show()
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    v.resetTransform()
    QtWidgets.QApplication.processEvents()
    v.drawing_mode = "smart_canopy"
    return v


def _mouse(viewer, kind, scene_pt, buttons, mods=QtCore.Qt.NoModifier):
    vp = viewer.mapFromScene(QtCore.QPointF(*scene_pt))
    ev = QtGui.QMouseEvent(kind, QtCore.QPointF(vp), QtCore.Qt.LeftButton, buttons, mods)
    {QtCore.QEvent.MouseButtonPress: viewer.mousePressEvent,
     QtCore.QEvent.MouseMove: viewer.mouseMoveEvent,
     QtCore.QEvent.MouseButtonRelease: viewer.mouseReleaseEvent}[kind](ev)


def _stroke(viewer, pts, mods=QtCore.Qt.NoModifier):
    _mouse(viewer, QtCore.QEvent.MouseButtonPress, pts[0], QtCore.Qt.LeftButton, mods)
    for p in pts[1:]:
        _mouse(viewer, QtCore.QEvent.MouseMove, p, QtCore.Qt.LeftButton, mods)
    _mouse(viewer, QtCore.QEvent.MouseButtonRelease, pts[-1], QtCore.Qt.NoButton, mods)


def _key(viewer, key):
    viewer.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier))


def _scene_mask(poly):
    pts = np.array([[p.x() - 0.5, p.y() - 0.5] for p in poly])          # pixmap-local -> array coords
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
    return m


def _item_mask(item):
    return _scene_mask(item.mapToScene(item.polygon))


def _zig_scene(name):
    (cx, cy), R, _ = CROWNS[name]
    return [tuple(p + 0.5) for p in _zigzag(cx, cy, R)]


def test_smart_canopy_is_not_a_point_mode(viewer_factory):
    v = viewer_factory()
    v.drawing_mode = "smart_canopy"
    assert v._is_point_drawing_mode() is False


def test_strokes_accumulate_and_only_enter_commits(viewer_factory, scene):
    img, truth = scene
    v = _viewer_with(viewer_factory, img)
    drawn = []
    v.polygon_drawn.connect(drawn.append)
    zig = _zig_scene("round")
    _stroke(v, zig[:15], QtCore.Qt.ShiftModifier)               # Shift starts the session
    assert v.drawing is True and drawn == []
    first = _scene_mask(v._sc_selection).sum()
    _stroke(v, zig[15:])                                        # later strokes need no Shift
    assert drawn == [] and len(v._sc_strokes) == 2
    assert v._sc_sel_item is not None and not v._sc_sel_item.path().isEmpty()
    assert _scene_mask(v._sc_selection).sum() >= first
    _key(v, QtCore.Qt.Key_Return)
    assert v.drawing is False and v.temp_drawing_item is None and v._sc_sel_item is None
    assert len(drawn) == 1
    assert _iou(_item_mask(drawn[0]), truth["round"]) >= 0.85


def test_backspace_undoes_a_stroke_and_escape_cancels(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    drawn = []
    v.polygon_drawn.connect(drawn.append)
    zig = _zig_scene("lobed")
    _stroke(v, zig[:15], QtCore.Qt.ShiftModifier)
    one = QtGui.QPolygonF(v._sc_selection)
    _stroke(v, [(80.5, 80.5)])                                  # a stray click on soil
    _key(v, QtCore.Qt.Key_Backspace)
    assert len(v._sc_strokes) == 1
    assert _iou(_scene_mask(v._sc_selection), _scene_mask(one)) > 0.99
    _key(v, QtCore.Qt.Key_Escape)
    assert v.drawing is False and v._sc_sel_item is None and drawn == []


def test_ctrl_drag_erases_inside_the_session(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    (cx, cy), R, _ = CROWNS["round"]
    _stroke(v, _zig_scene("round"), QtCore.Qt.ShiftModifier)
    before = _scene_mask(v._sc_selection)
    x = cx + 0.7 * R + 0.5
    _stroke(v, [(x, cy - R + 0.5 + 10 * k) for k in range(15)], QtCore.Qt.ControlModifier)
    assert v._sc_strokes[-1]["erase"] is True
    after = _scene_mask(v._sc_selection)
    assert after.sum() < before.sum() and after[cy, int(x)] == 0


def test_double_click_does_not_commit(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    drawn = []
    v.polygon_drawn.connect(drawn.append)
    _stroke(v, _zig_scene("round"), QtCore.Qt.ShiftModifier)
    vp = v.mapFromScene(QtCore.QPointF(180.5, 170.5))
    v.mouseDoubleClickEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseButtonDblClick, QtCore.QPointF(vp),
                                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert v.drawing is True and drawn == []


def test_brackets_resize_the_brush(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    _stroke(v, [(180.5, 170.5)], QtCore.Qt.ShiftModifier)
    r0 = v._smart_canopy_brush_scene()
    _key(v, QtCore.Qt.Key_BracketRight)
    assert v._smart_canopy_brush_scene() > r0
    _key(v, QtCore.Qt.Key_BracketLeft)
    _key(v, QtCore.Qt.Key_BracketLeft)
    assert v._smart_canopy_brush_scene() < r0


def test_minus_plus_change_the_grow_and_reselect_at_once(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    _stroke(v, [(180.5, 170.5)], QtCore.Qt.ShiftModifier)
    g0 = v._smart_canopy_grow()
    for _ in range(12):                                        # down to "exactly the paint"
        v.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Minus, QtCore.Qt.NoModifier, "-"))
    assert v._smart_canopy_grow() == 0.0
    small = _scene_mask(v._sc_selection).sum()
    for _ in range(3):
        v.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Plus, QtCore.Qt.NoModifier, "+"))
    assert v._smart_canopy_grow() > 0.0
    assert _scene_mask(v._sc_selection).sum() > small          # re-selected without a new stroke
    assert len(v._sc_strokes) == 1 and g0 > 0


def test_on_a_cog_the_sharp_viewport_tile_is_what_is_traced(viewer_factory, scene):
    """Zoomed in on a COG the viewer shows a sharp tile over a decimated preview: the brush must read the
    tile.  Here the preview is flat grey and only the tile holds the crowns."""
    img, truth = scene
    v = _viewer_with(viewer_factory, np.full_like(img, 128))
    h, w = img.shape[:2]
    tile = QtGui.QImage(np.ascontiguousarray(img).data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    v._highres_enabled = True
    v.update_highres_overlay(QtGui.QPixmap.fromImage(tile), 0.0, 0.0, 1.0, 1.0)
    drawn = []
    v.polygon_drawn.connect(drawn.append)
    _stroke(v, _zig_scene("round"), QtCore.Qt.ShiftModifier)
    assert v._smart_canopy_source(v._sc_strokes) is v._highres_front_item
    _key(v, QtCore.Qt.Key_Return)
    assert len(drawn) == 1
    assert _iou(_item_mask(drawn[0]), truth["round"]) >= 0.85


def test_a_second_crown_stops_at_the_first(viewer_factory, scene):
    img, _ = scene
    v = _viewer_with(viewer_factory, img)
    drawn = []
    v.polygon_drawn.connect(drawn.append)
    v._sc_brush_screen = 8.0                                   # ~1/6 of these small crowns' radius, as with [ / ]
    for name in ("touchA", "touchB"):
        _stroke(v, _zig_scene(name), QtCore.Qt.ShiftModifier)
        _key(v, QtCore.Qt.Key_Enter)
    assert len(drawn) == 2
    ma, mb = _item_mask(drawn[0]), _item_mask(drawn[1])
    assert (ma & mb).sum() <= 0.02 * mb.sum()
