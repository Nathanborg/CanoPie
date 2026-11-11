"""
Annotator brush patches and the single Annotator window.

* PatchCanvas: strokes are clipped to the crown, separate strokes stay
  separate polygons, Ctrl (erase) cuts holes that survive the round trip
  through polygons, the fraction is painted area / crown area.
* PatchBrush on a REAL viewer, driven by real mouse events: a drag paints, the
  patches land on the polygon record under annotator_patches[property name],
  in the crown's own coordinates, and reach the sidecar JSON on save.
* Finishing a drawing on a numeric property asks to use the % as its value;
  "Always" / "Never" (or the remembered answer) skip the question.
* The Polygon Properties panel is part of the Annotator window, and the
  window moves off the viewer showing the polygon a list click zoomed to.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest

from canopie.annotator import (
    KIND_NUMERIC, PATCHES_KEY, AnnotationStore, PatchCanvas, find_record, norm_fp)

pytestmark = [pytest.mark.polygons]

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


# ------------------------------------------------------------- PatchCanvas

def test_strokes_are_clipped_to_the_crown():
    c = PatchCanvas(SQUARE)
    c.add_stroke([(-50, 50), (150, 50)], radius=10)       # runs off both sides
    polys = c.polygons()
    assert len(polys) == 1
    xs = np.array(polys[0][0])[:, 0]
    assert xs.min() >= -0.5 and xs.max() <= 100.5, "paint leaked outside the crown"
    assert c.fraction == pytest.approx(0.20, abs=0.02)    # a 20-wide band of a 100x100 crown


def test_separate_strokes_are_separate_patches_and_erase_cuts_holes():
    c = PatchCanvas(SQUARE)
    c.add_stroke([(20, 20)], radius=8)
    c.add_stroke([(80, 80)], radius=8)
    assert len(c.polygons()) == 2, "two unconnected patches must stay two polygons"

    c.clear()
    c.add_stroke([(50, 50)], radius=30)
    c.add_stroke([(50, 50)], radius=8, erase=True)
    polys = c.polygons()
    assert len(polys) == 1 and len(polys[0]) == 2, "the erased centre must be a hole"
    f = c.fraction
    again = PatchCanvas(SQUARE, polys)                     # rebuild from what is saved
    assert again.fraction == pytest.approx(f, abs=0.01), "the hole was lost on reload"


def test_transfer_follows_a_moved_and_rescaled_crown():
    from canopie.annotator import transfer_patches
    src_crown = SQUARE
    dst_crown = [(500, 300), (700, 300), (700, 500), (500, 500)]      # moved, 2x bigger
    patch = [[[10, 10], [30, 10], [30, 30], [10, 30]]]
    out = transfer_patches([patch], src_crown, dst_crown)
    assert out == [[[[520.0, 320.0], [560.0, 320.0], [560.0, 360.0], [520.0, 360.0]]]]
    same = transfer_patches([patch], SQUARE, SQUARE)
    assert same == [[[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]]], (
        "a fixed camera (same crown) must get the drawing back unchanged")


def test_canvas_record_is_in_the_crowns_basis():
    c = PatchCanvas([(1000, 2000), (1400, 2000), (1400, 2400), (1000, 2400)])
    c.add_stroke([(1200, 2200)], radius=50)
    rec = c.to_record((5000, 4000))
    ring = np.array(rec["polygons"][0][0])
    assert 1140 < ring[:, 0].mean() < 1260 and 2140 < ring[:, 1].mean() < 2260
    assert rec["crown_area_px"] == pytest.approx(160000, rel=0.01)
    assert rec["fraction"] == pytest.approx(np.pi * 50 ** 2 / 160000, rel=0.05)
    assert rec["basis"] == {"w": 5000, "h": 4000}


# ---------------------------------------------- brush on a real viewer + tab

@pytest.fixture()
def project_tab(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    from canopie.image_viewer import _ZoomBar
    saved = _ZoomBar._zoom_sync_enabled
    _ZoomBar._zoom_sync_enabled = False
    pt = build_project_tab(str(tmp_path))
    # let the project's deferred image-group load run now: it rebuilds the
    # viewer grid, which would delete the test's own viewer mid-test
    import time
    t0 = time.time()
    while time.time() - t0 < 1.0:
        qapp.processEvents()
        time.sleep(0.01)
    yield pt
    _ZoomBar._zoom_sync_enabled = saved
    dlg = getattr(pt, "annotator", None)
    if dlg is not None:
        dlg.close()
        dlg.deleteLater()


def _crown(pt):
    """A real polygon of the project + a viewer showing its image, at 1:1."""
    from canopie.image_viewer import ImageViewer
    for g, fm in pt.all_polygons.items():
        for fp, rec in fm.items():
            if rec.get("type", "polygon") == "polygon" and len(rec.get("points") or []) >= 3:
                ref = rec.get("image_ref_size") or {}
                w, h = int(ref.get("w") or 64), int(ref.get("h") or 64)
                v = ImageViewer()
                v.resize(400, 400)
                v.image_data = type("D", (), {"filepath": fp, "image": None})()
                v._scene.clear()
                v._image = v._scene.addPixmap(QtGui.QPixmap(w, h))
                v._scene.setSceneRect(0, 0, w, h)
                v.polygons = []
                box = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(box)
                lay.addWidget(QtWidgets.QLabel(os.path.basename(fp)))
                lay.addWidget(v)
                box.resize(420, 440)
                box.show()
                v.resetTransform()
                return g, fp, rec, v, {"container": box, "viewer": v, "image_data": v.image_data}
    raise AssertionError("no polygon in the fixture project")


def _drag(viewer, scene_pts, ctrl=False):
    vp = viewer.viewport()
    mods = QtCore.Qt.ControlModifier if ctrl else QtCore.Qt.NoModifier
    pts = [viewer.mapFromScene(QtCore.QPointF(*p)) for p in scene_pts]
    QTest.mousePress(vp, QtCore.Qt.LeftButton, mods, pts[0])
    for p in pts[1:]:
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p), QtCore.Qt.NoButton,
                               QtCore.Qt.LeftButton, mods)
        QtWidgets.QApplication.sendEvent(vp, ev)
    QTest.mouseRelease(vp, QtCore.Qt.LeftButton, mods, pts[-1])


def test_brush_paints_saves_and_sets_the_value(project_tab, qapp, monkeypatch, tmp_path):
    pt = project_tab
    g, fp, rec, viewer, w = _crown(pt)
    saved_widgets = pt.viewer_widgets
    pt.viewer_widgets = [w]
    try:
        dlg = pt.show_annotator()
        assert dlg.panel.window() is dlg, "the Polygon Properties panel must be part of the Annotator window"
        pid = dlg.store.add_property("flower", KIND_NUMERIC, step=0.1)
        dlg.refresh()
        dlg.open_polygon(g, fp, navigate=False)
        qapp.processEvents()

        row = dlg.panel.rows[pid]
        row.draw.setChecked(True)
        assert dlg._brush is not None and not dlg.panel.brush_bar.isHidden()
        dlg.panel.brush_slider.setValue(6)

        pts = np.asarray(rec["points"], dtype=float)
        cx, cy = pts.mean(axis=0)
        ref = rec.get("image_ref_size") or {}
        assert viewer._image.pixmap().width() == int(ref.get("w") or 64)    # basis == scene here
        _drag(viewer, [(cx - 2, cy), (cx + 2, cy)])
        qapp.processEvents()

        patch = rec.get(PATCHES_KEY, {}).get("flower")
        assert patch and patch["polygons"], "the stroke did not reach the polygon record"
        assert 0 < patch["fraction"] <= 1
        ring = np.asarray(patch["polygons"][0][0])
        assert abs(ring[:, 0].mean() - cx) < 6 and abs(ring[:, 1].mean() - cy) < 6, (
            "patch coordinates are not in the crown's own basis")

        asked = {}

        def fake_exec(box):
            asked["text"] = box.text()
            box.checkBox().setChecked(True)               # remember: Always
            return QtWidgets.QMessageBox.Yes
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", fake_exec)
        dlg.panel.done_btn.click()
        assert dlg._brush is None and dlg.panel.brush_bar.isHidden()
        assert "%" in asked.get("text", ""), "finishing did not offer the percentage"
        want = round(100 * patch["fraction"], 1)
        assert dlg.store.get_value(g, fp, pid) == pytest.approx(want, abs=0.051)
        assert rec["properties"]["flower"] == pytest.approx(want, abs=0.051)
        assert dlg.store.property(pid)["patch_sets_value"] is True

        # remembered: no question the second time
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_",
                            lambda *a: pytest.fail("asked again after 'remember'"))
        dlg.panel.rows[pid].draw.setChecked(True)
        _drag(viewer, [(cx, cy - 3), (cx, cy + 3)], ctrl=True)     # erase part of it
        dlg.panel.rows[pid].draw.setChecked(False)
        assert dlg.store.get_value(g, fp, pid) == pytest.approx(
            round(100 * rec[PATCHES_KEY]["flower"]["fraction"], 1), abs=0.051)

        pt.save_incremental(show_status=False)
        found = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "polygons").glob("*_polygons.json")]
        assert any((d.get(PATCHES_KEY) or {}).get("flower", {}).get("polygons") for d in found), (
            "the patches did not reach the polygon's JSON file")
    finally:
        pt.viewer_widgets = saved_widgets
        w["container"].hide()
        w["container"].deleteLater()


def test_never_does_not_touch_the_value(project_tab, qapp, monkeypatch):
    pt = project_tab
    g, fp, rec, viewer, w = _crown(pt)
    saved_widgets = pt.viewer_widgets
    pt.viewer_widgets = [w]
    try:
        dlg = pt.show_annotator()
        pid = dlg.store.add_property("flower", KIND_NUMERIC, patch_sets_value=False)
        dlg.open_polygon(g, fp, navigate=False)
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda *a: pytest.fail("asked although 'Never'"))
        dlg.start_drawing(pid)
        cx, cy = np.asarray(rec["points"], dtype=float).mean(axis=0)
        _drag(viewer, [(cx, cy)])
        dlg.finish_drawing()
        assert rec[PATCHES_KEY]["flower"]["polygons"]
        assert dlg.store.get_value(g, fp, pid) is not None and "flower" not in (rec.get("properties") or {})
    finally:
        pt.viewer_widgets = saved_widgets
        w["container"].hide()
        w["container"].deleteLater()


def test_window_moves_off_the_viewer(project_tab, qapp):
    pt = project_tab
    g, fp, rec, viewer, w = _crown(pt)
    try:
        dlg = pt.show_annotator()
        vp = viewer.viewport()
        vg = QtCore.QRect(vp.mapToGlobal(QtCore.QPoint(0, 0)), vp.size())
        dlg.move(vg.center() - QtCore.QPoint(dlg.width() // 2, dlg.height() // 2))   # right on top of it
        qapp.processEvents()
        assert dlg.frameGeometry().intersects(vg)
        poly = viewer.mapToScene(vp.rect().center())
        dlg._keep_clear_of(viewer, QtCore.QRectF(poly.x() - 5, poly.y() - 5, 10, 10))
        qapp.processEvents()
        r = viewer.mapFromScene(QtCore.QRectF(poly.x() - 5, poly.y() - 5, 10, 10)).boundingRect()
        pr = QtCore.QRect(vp.mapToGlobal(r.topLeft()), r.size())
        assert not dlg.frameGeometry().intersects(pr), "the Annotator still covers the polygon"
    finally:
        w["container"].hide()
        w["container"].deleteLater()


def _second_image_of(pt, g, fp, rec):
    """The same canopy (group) on another image: a phenocam / time series."""
    other = next(f for fm in pt.all_polygons.values() for f in fm if norm_fp(f) != norm_fp(fp))
    twin = {k: v for k, v in rec.items() if k not in (PATCHES_KEY, "properties")}
    pt.all_polygons[g][other] = twin
    return other, twin


def test_copy_drawing_from_the_canopys_previous_image(project_tab, qapp, monkeypatch):
    pt = project_tab
    g, fp, rec, viewer, w = _crown(pt)
    saved_widgets = pt.viewer_widgets
    try:
        dlg = pt.show_annotator()
        pid = dlg.store.add_property("flower", KIND_NUMERIC, patch_sets_value=False)
        other, twin = _second_image_of(pt, g, fp, rec)
        first, second = dlg._canopy_images(g)            # time order
        src_rec = pt.all_polygons[g][first]
        # the earlier image already has a drawing
        c = PatchCanvas(src_rec["points"])
        cx, cy = np.asarray(src_rec["points"], dtype=float).mean(axis=0)
        c.add_stroke([(cx, cy)], radius=3)
        src_rec[PATCHES_KEY] = {"flower": c.to_record((64, 64))}

        viewer.image_data.filepath = second
        w["image_data"] = viewer.image_data
        pt.viewer_widgets = [w]
        dlg.open_polygon(g, second, navigate=False)
        dlg.start_drawing(pid)
        assert dlg.panel.copy_prev_btn.isEnabled(), "an earlier drawing exists, the button must be on"
        dlg.panel.copy_prev_btn.click()
        got = pt.all_polygons[g][second].get(PATCHES_KEY, {}).get("flower")
        assert got and got["polygons"], "nothing was copied"
        assert got["fraction"] == pytest.approx(src_rec[PATCHES_KEY]["flower"]["fraction"], abs=0.01)
        assert pt.all_polygons[g][first][PATCHES_KEY]["flower"]["fraction"] == pytest.approx(
            c.fraction, abs=1e-5), "copying changed the source drawing"
        dlg.finish_drawing()

        # the earliest image has nothing before it
        viewer.image_data.filepath = first
        dlg.open_polygon(g, first, navigate=False)
        dlg.start_drawing(pid)
        assert not dlg.panel.copy_prev_btn.isEnabled()
        dlg.finish_drawing()
    finally:
        pt.viewer_widgets = saved_widgets
        w["container"].hide()
        w["container"].deleteLater()


def test_strip_is_red_until_complete_then_green(project_tab, qapp):
    from canopie.annotator import _ACCENT, _PENDING
    pt = project_tab
    g, fp, rec, viewer, w = _crown(pt)
    saved_widgets = pt.viewer_widgets
    pt.viewer_widgets = [w]
    try:
        dlg = pt.show_annotator()
        pid = dlg.store.add_property("cover", KIND_NUMERIC)
        dlg.open_polygon(g, fp, navigate=False)
        head = w["container"].layout().itemAt(0).widget()
        assert _PENDING in viewer.styleSheet() and _PENDING in head.styleSheet(), "pending must be red"
        dlg.panel.rows[pid].slider.setValue(10)
        assert _ACCENT in viewer.styleSheet() and _ACCENT in head.styleSheet(), "complete must turn green"
        dlg.panel.rows[pid].na.setChecked(True)
        assert _ACCENT in head.styleSheet(), "N/A is an answer: still green"
    finally:
        pt.viewer_widgets = saved_widgets
        w["container"].hide()
        w["container"].deleteLater()


def test_arrow_and_a_d_keys_walk_the_polygons_to_fill(project_tab, qapp):
    pt = project_tab
    dlg = pt.show_annotator()
    dlg.store.add_property("cover", KIND_NUMERIC)
    dlg.refresh()
    dlg.activateWindow()
    dlg.tree.setFocus()
    qapp.processEvents()
    seen = []
    for key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
        QTest.keyClick(dlg.tree, key)
        qapp.processEvents()
        seen.append((dlg.panel.group, norm_fp(dlg.panel.filepath)))
    assert seen[0] != seen[1] != seen[2], "→ / D did not move to the next polygon to fill"
    QTest.keyClick(dlg.tree, QtCore.Qt.Key_A)
    qapp.processEvents()
    assert (dlg.panel.group, norm_fp(dlg.panel.filepath)) == seen[1], "A did not go back"
    QTest.keyClick(dlg.tree, QtCore.Qt.Key_Left)
    qapp.processEvents()
    assert (dlg.panel.group, norm_fp(dlg.panel.filepath)) == seen[0], "← did not go back"

    # In the project window A/D/←/→ step roots; the Annotator borrows them
    # while open (you draw in a viewer, then press D) and gives them back.
    assert pt.leftKeyAct.shortcut().isEmpty() and pt.nextAct.shortcut().isEmpty()
    dlg.close()
    qapp.processEvents()
    assert pt.leftKeyAct.shortcut().toString() == "A"
    assert pt.rightKeyAct.shortcut().toString() == "D"
    assert pt.nextAct.shortcut().toString() == "Right"
    assert pt.prevAct.shortcut().toString() == "Left"
