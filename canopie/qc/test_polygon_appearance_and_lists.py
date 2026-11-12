"""
Polygon & label appearance, lists that follow viewer clicks, a non-blocking
ML manager.

* Labels: in the default "screen" mode a label is label_px SCREEN pixels at
  every zoom -- also on a COG, whose scene unit is a 1/16 preview pixel. The
  old code set the size in scene units clamped to 14..220, so at high zoom a
  BCI label was hundreds of pixels tall. "zoom" mode keeps the old formula.
* apply_polygon_style reaches the outline pens, the label space reserved in
  the bounding rect and the zoomed-out tiles; the dialog previews live and
  Cancel restores the previous style; it saves to QSettings.
* Configuration menu + the gear next to Lock Polygons open the dialog.
* A press on a polygon in a viewer selects (and scrolls to) its row in the
  open Polygon Manager and ML manager; Ctrl adds to the selection.
* The ML manager opens non-modal (the canvas stays usable), one per tab.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie import image_viewer as iv
from canopie.image_viewer import ImageViewer, POLYGON_STYLE, apply_polygon_style

pytestmark = [pytest.mark.viewer, pytest.mark.polygons]


class _RecordingPainter:
    """Stands in for QPainter in draw_item_label: records the font size and
    the scale it is drawn at, to compute the on-screen size."""

    def __init__(self):
        self.scale_stack = [1.0]
        self.drawn = []
        self._font = QtGui.QFont()

    def save(self):
        self.scale_stack.append(self.scale_stack[-1])

    def restore(self):
        self.scale_stack.pop()

    def translate(self, *a):
        pass

    def scale(self, sx, sy):
        self.scale_stack[-1] *= sx

    def setFont(self, f):
        self._font = QtGui.QFont(f)

    def setPen(self, p):
        self._pen = p

    def drawText(self, *a):
        self.drawn.append((self._font.pixelSize(), self.scale_stack[-1], self._pen.color().name()))


class _Item:
    label_text = "guayacan_21294"
    is_rgb = True
    _cached_res_boost = 1.0
    _LABEL_SCALE_THRESHOLD = 0.35
    label_offset = QtCore.QPointF(10, -10)


def _screen_px(view_scale):
    p = _RecordingPainter()
    iv.draw_item_label(_Item(), p, QtCore.QPointF(0, 0), view_scale)
    font_px, painter_scale, _c = p.drawn[-1]
    return font_px * painter_scale * view_scale       # device pixels


@pytest.mark.parametrize("zoom", [0.5, 1.0, 8.0, 50.0, 800.0])
def test_screen_labels_keep_their_size_at_any_zoom(zoom):
    assert POLYGON_STYLE["label_mode"] == "screen"
    assert _screen_px(zoom) == pytest.approx(POLYGON_STYLE["label_px"], rel=0.01), (
        f"at zoom {zoom} the label is {_screen_px(zoom):.0f} px on screen")


def test_old_labels_grew_with_zoom_and_zoom_mode_still_does():
    apply_polygon_style(dict(POLYGON_STYLE, label_mode="zoom"), save=False)
    assert _screen_px(50.0) >= 14 * 50 * 0.99, "zoom mode must keep the old scene-unit size"
    apply_polygon_style(dict(POLYGON_STYLE, label_mode="screen"), save=False)
    assert _screen_px(50.0) == pytest.approx(14, rel=0.01)


def test_label_colour_and_size_follow_the_style():
    apply_polygon_style(dict(POLYGON_STYLE, label_px=22, label_color="#00ff00", label_halo=False), save=False)
    p = _RecordingPainter()
    iv.draw_item_label(_Item(), p, QtCore.QPointF(0, 0), 3.0)
    font_px, sc, color = p.drawn[-1]
    assert font_px * sc * 3.0 == pytest.approx(22, rel=0.01)
    assert color == "#00ff00"
    assert len(p.drawn) == 1, "no halo passes when the halo is off"


@pytest.fixture
def viewer(qapp):
    v = ImageViewer()
    v.resize(300, 300)
    pm = QtGui.QPixmap(400, 400)
    pm.fill(QtCore.Qt.gray)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v.polygons = []
    for i in range(3):
        x = 50 + 100 * i
        v.add_polygon_to_scene(QtGui.QPolygonF([QtCore.QPointF(x, 50), QtCore.QPointF(x + 40, 50),
                                                QtCore.QPointF(x + 40, 90)]), name=f"g{i}")
    yield v
    v._scene.clear()
    v.deleteLater()


def test_style_reaches_outline_pens_and_label_space(viewer):
    it = viewer.get_all_polygons()[0]
    it._ensure_pens()
    assert it._pen_base.color() == QtGui.QColor(QtCore.Qt.red) or it._pen_base.color() == QtGui.QColor(QtCore.Qt.blue)
    apply_polygon_style(dict(POLYGON_STYLE, outline_color="#00ffff", selected_color="#ffff00",
                             outline_width=3.0), save=False)
    it._ensure_pens()
    assert it._pen_base.color().name() == "#00ffff"
    assert it._pen_highlight.color().name() == "#ffff00"

    it.show_label = True
    it._update_bounding_rect()
    small = it.boundingRect()
    apply_polygon_style(dict(POLYGON_STYLE, label_px=40), save=False)
    big = it.boundingRect()
    assert big.height() > small.height(), "a bigger label must reserve more space (or it is clipped)"


def test_style_reaches_the_zoomed_out_tiles():
    apply_polygon_style(dict(POLYGON_STYLE, outline_color="#123456"), save=False)
    import numpy as np
    tile = iv.PolygonTileItem(QtCore.QRectF(0, 0, 10, 10),
                              [(np.array([[0, 0], [5, 0], [5, 5]], float), None)], is_rgb=True)
    assert tile._default_color.name() == "#123456"


def test_dialog_previews_live_and_cancel_restores(qapp):
    before = dict(POLYGON_STYLE)
    dlg = iv.PolygonAppearanceDialog()
    dlg.label_px.setValue(30)
    assert POLYGON_STYLE["label_px"] == 30, "changes must show live"
    dlg.reject()
    assert POLYGON_STYLE == before, "Cancel must put the previous style back"


def test_saved_style_round_trips_through_qsettings(qapp, monkeypatch):
    monkeypatch.setattr(iv, "_POLYGON_STYLE_SETTINGS", ("CanoPieQC", "polygon_style_test"))
    QtCore.QSettings(*iv._POLYGON_STYLE_SETTINGS).clear()
    try:
        apply_polygon_style(dict(POLYGON_STYLE, label_px=19, outline_width=2.5), save=True)
        apply_polygon_style(dict(iv.POLYGON_STYLE_DEFAULTS), save=False)
        st = iv.load_polygon_style()
        assert st["label_px"] == 19 and st["outline_width"] == 2.5
    finally:
        QtCore.QSettings(*iv._POLYGON_STYLE_SETTINGS).clear()


# --------------------------------------------------------- entry points

def test_configuration_menu_and_manager_gear_open_the_dialog(qapp, monkeypatch):
    from canopie.main_window import MainWindow
    opened = []
    monkeypatch.setattr(iv.PolygonAppearanceDialog, "exec_", lambda self: opened.append(self) or 0)
    win = MainWindow()
    try:
        menus = {a.text(): a.menu() for a in win.menuBar().actions()}
        assert "Configuration" in menus, "no Configuration menu"
        texts = [a.text() for a in menus["Configuration"].actions()]
        assert any("Appearance" in t for t in texts)
        win.polygonAppearanceAct.trigger()
        pt = win.get_current_tab() or None
        if pt is None:
            win.add_new_tab("t")
            pt = win.get_current_tab()
        pm = pt.polygon_manager
        assert pm.appearance_button.parent() is not None
        idx = pm.top_controls_layout.indexOf(pm.appearance_button)
        assert idx == pm.top_controls_layout.indexOf(pm.lock_polys_checkbox) + 1, "the gear sits beside Lock Polygons"
        pm.appearance_button.click()
        assert len(opened) == 2
    finally:
        win.close()
        win.deleteLater()


# ----------------------------------------------- lists follow viewer clicks

@pytest.fixture()
def project_tab(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    pt = build_project_tab(str(tmp_path))
    yield pt
    for attr in ("ml_manager",):
        w = getattr(pt, attr, None)
        if w is not None:
            w.close()
            w.deleteLater()
    pt.polygon_manager.hide()


def _first_group_and_fp(pt):
    for g, fm in pt.all_polygons.items():
        for fp in fm:
            return g, fp


def test_a_viewer_click_selects_the_row_in_both_managers(project_tab, qapp, monkeypatch):
    pt = project_tab
    pt.show_polygon_manager()
    mlm = pt.show_machine_learning_manager()
    g, fp = _first_group_and_fp(pt)
    other = next(x for x in pt.all_polygons if x != g)
    v = ImageViewer()
    v.image_data = type("D", (), {"filepath": fp, "image": None})()
    v._scene.clear()
    v._image = v._scene.addPixmap(QtGui.QPixmap(64, 64))
    v.polygons = []
    pt._wire_viewer_for_inspection(v)
    v.add_polygon_to_scene(QtGui.QPolygonF([QtCore.QPointF(2, 2), QtCore.QPointF(12, 2),
                                            QtCore.QPointF(12, 12)]), name=g)
    item = v.get_all_polygons()[0]

    zoomed = []
    monkeypatch.setattr(type(pt.polygon_manager), "zoom_to_groups", lambda self, gs: zoomed.append(gs))
    pt.polygon_manager.highlight_group(other)              # some other row selected first
    iv._notify_polygon_click(item)
    assert pt.polygon_manager.get_selected_polygon_groups() == [g], "the Polygon Manager row did not follow"
    assert pt.polygon_manager.list_widget.currentItem().data(QtCore.Qt.UserRole) == g
    assert mlm.get_selected_groups() == [g], "the ML manager row did not follow"
    assert zoomed == [], "highlighting a row must not re-zoom the viewers"

    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                        staticmethod(lambda: QtCore.Qt.ControlModifier))
    pt.polygon_manager.highlight_group(other, additive=True)
    assert set(pt.polygon_manager.get_selected_polygon_groups()) == {g, other}
    v.deleteLater()


def test_ml_manager_no_longer_blocks_the_main_window(project_tab, qapp):
    pt = project_tab
    mlm = pt.show_machine_learning_manager()
    qapp.processEvents()
    assert mlm.isVisible() and not mlm.isModal()
    assert QtWidgets.QApplication.activeModalWidget() is None, "the ML manager is modal: the canvas is blocked"
    assert pt.show_machine_learning_manager() is mlm, "a second open must raise the same window"

    pt.all_polygons["__new_group__"] = {}
    mlm.refresh_polygon_groups()
    assert "__new_group__" in [mlm.list_widget.item(i).text() for i in range(mlm.list_widget.count())]
    pt.all_polygons.pop("__new_group__")


# ------------------------------------- list click / arrows move the viewer

def _wired_viewer(pt, fp, groups):
    """A shown viewer with one polygon per group, far apart, labels ON, in
    pt.viewer_widgets (what the list handlers walk)."""
    v = ImageViewer()
    v.resize(300, 300)
    v.image_data = type("D", (), {"filepath": fp, "image": None, "ax_config": {},
                                  "raw_shape": (2000, 2000, 3)})()
    pm = QtGui.QPixmap(2000, 2000)
    pm.fill(QtCore.Qt.gray)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v._scene.setSceneRect(0, 0, 2000, 2000)
    v._cached_pixmap_size = (2000, 2000)
    v.polygons = []
    centres = {}
    for i, g in enumerate(groups):
        x, y = 300 + 1200 * i, 300 + 1200 * i
        pts = [(x, y), (x + 60, y), (x + 60, y + 60), (x, y + 60)]
        pt.all_polygons[g] = {fp: {"name": g, "type": "polygon", "coord_space": "image",
                                   "image_ref_size": {"w": 2000, "h": 2000}, "points": pts}}
        v.add_polygon_to_scene(QtGui.QPolygonF([QtCore.QPointF(a, b) for a, b in pts]), name=g)
        centres[g] = QtCore.QPointF(x + 30, y + 30)
    pt._poly_norm_index_invalid = True
    v.set_labels_visible(True)                       # the label space is what threw centerOn off
    v.show()
    v.resetTransform()
    v.scale(2.0, 2.0)
    return v, centres


def _view_centre(v):
    return v.mapToScene(v.viewport().rect().center())


@pytest.fixture()
def two_crowns(project_tab, qapp):
    pt = project_tab
    # the project's deferred image-group load rebuilds viewer_widgets; let it
    # run now, or it replaces the test's viewer mid-test
    import time
    t0 = time.time()
    while time.time() - t0 < 1.0:
        qapp.processEvents()
        time.sleep(0.01)
    _g, fp = _first_group_and_fp(pt)
    groups = ["__crown_a__", "__crown_b__"]
    v, centres = _wired_viewer(pt, fp, groups)
    saved = pt.viewer_widgets
    pt.viewer_widgets = [{"viewer": v, "image_data": v.image_data}]
    qapp.processEvents()
    try:
        yield pt, v, centres, groups
    finally:
        pt.viewer_widgets = saved
        for g in groups:
            pt.all_polygons.pop(g, None)
        v.hide()
        v.deleteLater()


def test_one_click_in_the_polygon_manager_centres_the_polygon_at_the_same_zoom(two_crowns, qapp):
    from PyQt5.QtTest import QTest
    pt, v, centres, (a, b) = two_crowns
    pm = pt.polygon_manager
    pm.set_polygons(pt.all_polygons)
    pt.show_polygon_manager()
    lw = pm.list_widget
    row = next(lw.item(i) for i in range(lw.count()) if lw.item(i).data(QtCore.Qt.UserRole) == b)
    zoom0 = v.transform().m11()
    QTest.mouseClick(lw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, lw.visualItemRect(row).center())
    qapp.processEvents()
    c = _view_centre(v)
    assert abs(c.x() - centres[b].x()) < 3 and abs(c.y() - centres[b].y()) < 3, (
        f"one click left the view at ({c.x():.0f}, {c.y():.0f}), the polygon is at "
        f"({centres[b].x():.0f}, {centres[b].y():.0f})")
    assert v.transform().m11() == pytest.approx(zoom0), "a list click must keep the zoom"


def test_arrow_down_in_the_polygon_manager_moves_to_the_next_polygon(two_crowns, qapp):
    from PyQt5.QtTest import QTest
    pt, v, centres, (a, b) = two_crowns
    pm = pt.polygon_manager
    pm.set_polygons(pt.all_polygons)
    pt.show_polygon_manager()
    lw = pm.list_widget
    rows = {lw.item(i).data(QtCore.Qt.UserRole): i for i in range(lw.count())}
    assert rows[b] == rows[a] + 1, "precondition: b follows a in the list"
    QTest.mouseClick(lw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     lw.visualItemRect(lw.item(rows[a])).center())
    qapp.processEvents()
    QTest.keyClick(lw, QtCore.Qt.Key_Down)
    qapp.processEvents()
    c = _view_centre(v)
    assert abs(c.x() - centres[b].x()) < 3 and abs(c.y() - centres[b].y()) < 3, "Down did not move to the next polygon"
    assert pm.get_selected_polygon_groups() == [b]


def test_ml_manager_list_click_and_arrows_move_the_viewer(two_crowns, qapp):
    from PyQt5.QtTest import QTest
    pt, v, centres, (a, b) = two_crowns
    mlm = pt.show_machine_learning_manager()
    mlm.refresh_polygon_groups()
    lw = mlm.list_widget
    rows = {lw.item(i).text(): i for i in range(lw.count())}
    zoom0 = v.transform().m11()
    QTest.mouseClick(lw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     lw.visualItemRect(lw.item(rows[a])).center())
    qapp.processEvents()
    c = _view_centre(v)
    assert abs(c.x() - centres[a].x()) < 3 and abs(c.y() - centres[a].y()) < 3, "an ML list click did not move the viewer"
    assert v.transform().m11() == pytest.approx(zoom0)
    QTest.keyClick(lw, QtCore.Qt.Key_Down)
    qapp.processEvents()
    c = _view_centre(v)
    assert abs(c.x() - centres[b].x()) < 3 and abs(c.y() - centres[b].y()) < 3, "Down in the ML list did not move"
