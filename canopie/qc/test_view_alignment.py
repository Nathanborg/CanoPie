"""
Viewer alignment (view_alignment.py, view_alignment_dialog.py): a display-only
"fake registration" for images of one root that cover the same ground on
different pixel grids (Nathan, 2026-11-16: orthomosaics that are georeferenced
but do not match in row / column count, so polygons drawn on one land off
position on the other).

5 spots are matched with the canopy finder, a similarity (shift + rotation +
one scale) is fitted per image in RAW file pixels, and then:
  * zoom / pan / view rotation of the aligned viewers follow it,
  * a polygon drawn, replicated or edited on one image lands on the same
    ground in the others.

Built on test_canopy_finder's Scene: a source GeoTIFF and a target with a
known 7.3 m georeference error, a -3 degree rotation and 0.2 m pixels
(source 0.15 m).
"""
import math
import types

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import view_alignment as VA
from .test_canopy_finder import Scene

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return Scene(str(tmp_path_factory.mktemp("align_scene")))


def _sim(s, deg, tx, ty):
    th = math.radians(deg)
    return np.array([[s * math.cos(th), -s * math.sin(th), tx], [s * math.sin(th), s * math.cos(th), ty]])


def _spots(M, n=5, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.uniform(100, 900, 2)
        sq = c + np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]])
        out.append({"src": sq, "dst": VA.apply(M, sq) + rng.normal(0, noise, (4, 2)), "weight": 1.0})
    return out


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
def test_fit_recovers_a_known_similarity():
    M = _sim(0.5, -3.0, 120.0, -40.0)
    f = VA.fit_similarity(_spots(M))
    assert np.abs(f["M"] - M).max() < 1e-6
    s, ang, _tx, _ty = VA.similarity_params(f["M"])
    assert s == pytest.approx(0.5) and ang == pytest.approx(-3.0)
    assert f["rms_px"] < 1e-6 and f["n_used"] == 5


def test_a_wrong_spot_is_left_out():
    M = _sim(1.2, 4.0, -30.0, 15.0)
    spots = _spots(M, noise=0.3)
    spots[2]["dst"] = spots[2]["dst"] + 35.0
    f = VA.fit_similarity(spots)
    assert f["used"] == [True, True, False, True, True]
    assert f["rms_px"] < 1.0
    assert np.abs(VA.apply(f["M"], [[500, 500]]) - VA.apply(M, [[500, 500]])).max() < 1.0


def test_two_spots_are_enough_one_is_not():
    M = _sim(0.8, 10.0, 5.0, 5.0)
    assert VA.fit_similarity(_spots(M, n=2)) is not None
    assert VA.fit_similarity(_spots(M, n=1)) is None


def test_spots_are_spread_textured_and_never_on_nodata():
    rng = np.random.default_rng(1)
    img = rng.random((400, 500)).astype(np.float32) * 100
    img[:, :120] = 0.0                       # a NoData band on the left
    valid = img != 0
    pts = VA.pick_spots(img, valid, n=5, side=30)
    assert len(pts) == 5
    P = np.asarray(pts)
    assert (P[:, 0] > 120 + 14).all(), "a spot window touches NoData"
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) + np.eye(5) * 1e9
    assert d.min() > 30, "spots on top of each other"
    assert np.ptp(P[:, 0]) > 200 and np.ptp(P[:, 1]) > 200, "spots not spread"


# ---------------------------------------------------------------------------
# stored alignments
# ---------------------------------------------------------------------------
def test_sets_round_trip_and_compose_through_the_reference(tmp_path):
    Mb = _sim(0.75, 3.0, 24.0, -27.0)
    Mc = _sim(1.5, -2.0, -10.0, 40.0)
    A = VA.AlignmentSets()
    A.put("ref.tif", {"b.tif": {"M": Mb.tolist()}, "c.tif": {"M": Mc.tolist()}}, root="r1")
    B = VA.AlignmentSets(__import__("json").loads(__import__("json").dumps(A.to_json())))
    p = np.array([[10.0, 20.0], [300.0, 50.0]])
    # b -> c goes through the reference: c = Mc(ref) and ref = Mb^-1(b)
    want = VA.apply(Mc, VA.apply(VA.invert(Mb), p))
    assert np.abs(VA.apply(B.chain("b.tif", "c.tif"), p) - want).max() < 1e-9
    assert np.abs(VA.apply(B.chain("ref.tif", "b.tif"), p) - VA.apply(Mb, p)).max() < 1e-9
    assert B.chain("b.tif", "x.tif") is None
    B.sets[next(iter(B.sets))]["enabled"] = False
    assert B.chain("ref.tif", "b.tif") is None, "a disabled alignment still maps"


def test_record_map_goes_through_both_images_ax(scene):
    """Records are fractions of the post-.ax frame: a target cropped to a
    non-square window (anisotropic record basis) still gets the right ground."""
    from .. import canopy_finder as CF
    n = scene.B.shape[0]
    m = scene.m
    M = _sim(0.75, 3.0, 24.0, -27.0)
    dst_ax = {"crop_rect": {"x": 40, "y": 10, "width": 400, "height": 250},
              "crop_rect_ref_size": {"w": m, "h": m}}
    pm = {"M": M.tolist(), "src_ax": {}, "dst_ax": dst_ax, "src_raw_wh": (n, n), "dst_raw_wh": (m, m)}
    rec_src = scene.P                                       # no .ax on the source: record == raw
    got = VA.map_record_points(rec_src, pm)
    raw_t = VA.apply(M, scene.P)
    want, ref, _inside = CF.raw_to_record(raw_t, dst_ax, m, m)
    assert np.abs(got - np.asarray(want)).max() < 1e-9
    rec = {"points": got.tolist(), "coord_space": "image", "image_ref_size": ref}
    assert np.abs(CF.record_to_raw(rec, dst_ax, m, m) - raw_t).max() < 1e-9


def test_a_bulk_copy_with_a_point_map_is_mapped_and_never_linked(tmp_path):
    from .. import polygon_bulk, polygon_lod
    M = _sim(0.5, 0.0, 10.0, 20.0)
    src_rec = {"points": [[0, 0], [100, 0], [100, 50]], "coord_space": "image",
               "image_ref_size": {"w": 1000, "h": 800}, "type": "polygon", "name": "g"}
    side = tmp_path / "src.json"
    side.write_text("{}")
    job = {"group": "g", "dst_fp": str(tmp_path / "b.tif"), "sidecar": str(tmp_path / "out.json"),
           "src": src_rec, "src_wh": (1000, 800), "dst_wh": (600, 500), "root_id": "1", "coords": None,
           "point_map": {"M": M.tolist(), "src_ax": {}, "dst_ax": {}, "src_raw_wh": (1000, 800),
                         "dst_raw_wh": (600, 500)}}
    meta, _trash, linked = polygon_bulk.run_job(job)
    assert not linked
    out = polygon_lod.read_sidecar(job["sidecar"])
    assert np.abs(np.asarray(out["points"]) - VA.apply(M, src_rec["points"])).max() < 1e-6
    assert out["image_ref_size"] == {"w": 600, "h": 500}


# ---------------------------------------------------------------------------
# end to end: dialog, viewers, polygons
# ---------------------------------------------------------------------------
@pytest.fixture
def aligned(synthetic_project, scene, qapp):
    """Two real viewers (source, target) in the project tab, matched and
    aligned through the real AlignmentDialog."""
    import time
    from ..image_viewer import ImageViewer, _ZoomBar
    from ..view_alignment_dialog import AlignmentDialog
    pt = synthetic_project
    saved = (list(pt.viewer_widgets or []), pt.__dict__.get("_view_align_sets"),
             getattr(pt, "view_alignment_on", True), _ZoomBar._zoom_sync_enabled,
             _ZoomBar._fixed_zoom, _ZoomBar._fixed_center_norm)
    pt._view_align_sets = VA.AlignmentSets()
    pt._save_alignment_sets = lambda: None              # never write the session's global.ax
    views = []
    for fp in (scene.src, scene.tgt):
        lite = pt._imagedata_or_fallback(fp)
        v = ImageViewer(pt)
        v.resize(600, 500)
        v.show()
        v.set_image(pt._render_with_viewer_stretch(lite.image, v))
        v.image_data = lite
        views.append((v, lite))
    pt.viewer_widgets = [{"viewer": v, "image_data": l} for v, l in views]
    qapp.processEvents()
    vs, vt = views[0][0], views[1][0]
    dlg = AlignmentDialog(pt, vs, [scene.tgt], cfg={"processes": False})
    dlg.start()
    t0 = time.monotonic()
    while dlg.worker is not None and dlg.worker.isRunning() and time.monotonic() - t0 < 60:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    pt.viewer_widgets = [{"viewer": v, "image_data": l} for v, l in views]
    yield types.SimpleNamespace(pt=pt, vs=vs, vt=vt, dlg=dlg, scene=scene)
    dlg.close(); dlg.deleteLater()
    for v, _l in views:
        v.close(); v.setParent(None); v.deleteLater()
    pt.viewer_widgets = saved[0]
    pt._view_align_sets = saved[1]
    pt.view_alignment_on = saved[2]
    (_ZoomBar._zoom_sync_enabled, _ZoomBar._fixed_zoom, _ZoomBar._fixed_center_norm) = saved[3:]
    pt.__dict__.pop("_save_alignment_sets", None)
    qapp.processEvents()


def _centre(v):
    vp = v.viewport().rect()
    a, b = v.mapToScene(0, 0), v.mapToScene(vp.width(), vp.height())
    return np.array([(a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0])


def test_matching_and_fit_recover_the_scene(aligned):
    f = aligned.dlg.fits.get(aligned.scene.tgt)
    assert f is not None and f["n_used"] >= 4
    P = np.array([[100.0, 100.0], [800, 150], [450, 450], [200, 700], [700, 750]])
    err = np.abs(VA.apply(f["M"], P) - aligned.scene.truth(P)).max()
    assert err < 1.0, f"fit off by {err:.2f} px"
    s, ang, _tx, _ty = VA.similarity_params(f["M"])
    assert s == pytest.approx(0.15 / 0.2, rel=0.01) and ang == pytest.approx(3.0, abs=0.3)


def test_the_target_view_shows_the_same_ground(aligned, qapp):
    from ..image_viewer import _ZoomBar
    pt, vs, vt, sc = aligned.pt, aligned.vs, aligned.vt, aligned.scene
    aligned.dlg.apply()
    qapp.processEvents()
    assert _ZoomBar._zoom_sync_enabled, "Apply did not switch Fix on"
    vs.set_zoom_factor(2.0, anchor=QtWidgets.QGraphicsView.AnchorViewCenter)
    vs.centerOn(450, 450)
    _ZoomBar._sync_zoom_to_all_viewers(vs._zoombar, 2.0)
    qapp.processEvents()
    rec_s = pt._scene_to_image_xy(vs, [_centre(vs)])[0]
    rec_t = pt._scene_to_image_xy(vt, [_centre(vt)])[0]
    assert np.abs(rec_t - sc.truth(np.array([rec_s]))[0]).max() < 1.5
    assert vt.current_zoom_factor() == pytest.approx(2.0 / 0.75, rel=0.02)
    assert vt._view_rotation_deg == pytest.approx(-3.0, abs=0.3)
    # the rotation survives a plain zoom change on the target
    vt.set_zoom_factor(1.0, anchor=QtWidgets.QGraphicsView.AnchorViewCenter)
    tr = vt.transform()
    assert math.degrees(math.atan2(tr.m12(), tr.m11())) == pytest.approx(vt._view_rotation_deg, abs=1e-6)
    # pan the TARGET: the reference follows, unrotated
    vt.centerOn(*_centre(vt) + (40.0, 0.0))
    _ZoomBar.update_fixed_center(vt)
    qapp.processEvents()
    trs = vs.transform()
    assert abs(math.degrees(math.atan2(trs.m12(), trs.m11()))) < 0.05
    rec_s = pt._scene_to_image_xy(vs, [_centre(vs)])[0]
    rec_t = pt._scene_to_image_xy(vt, [_centre(vt)])[0]
    assert np.abs(rec_t - sc.truth(np.array([rec_s]))[0]).max() < 1.5


def test_alignment_off_keeps_the_plain_fraction_sync(aligned, qapp):
    from ..image_viewer import _ZoomBar
    aligned.dlg.apply()
    aligned.pt.set_view_alignment_enabled(False)
    assert not _ZoomBar._pose_aligned(aligned.vs, aligned.vt)
    assert getattr(aligned.vt, "_view_rotation_deg", 0.0) == 0.0


def test_a_drawn_polygon_lands_on_the_same_ground(aligned):
    pt, sc = aligned.pt, aligned.scene
    aligned.dlg.apply()
    fmap = pt._aligned_record_map(sc.src, sc.tgt)
    assert fmap is not None
    got = fmap(sc.P)
    assert np.abs(got - sc.truth()).max() < 1.0


def test_an_edit_moves_the_copy_and_one_undo_reverts_it(aligned):
    pt, sc = aligned.pt, aligned.scene
    aligned.dlg.apply()
    grp = "__align_test_crown"
    n, m = sc.B.shape[0], sc.m
    pt.all_polygons[grp] = {
        sc.src: {"points": sc.P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n},
                 "name": grp, "type": "polygon", "root": "1"},
        sc.tgt: {"points": sc.truth().tolist(), "coord_space": "image", "image_ref_size": {"w": m, "h": m},
                 "name": grp, "type": "polygon", "root": "1"}}
    saved = pt.__dict__.get("_save_specific_polygon_file")
    pt._save_specific_polygon_file = lambda *a, **k: None
    try:
        moved = sc.P + (25.0, -10.0)
        cmds = pt._propagate_aligned_edit(aligned.vs, grp, sc.src, moved.tolist(), undoable=True)
        assert len(cmds) == 1
        pt.undo_stack.beginMacro("t")
        for c in cmds:
            pt.undo_stack.push(c)
        pt.undo_stack.endMacro()
        tgt_pts = np.asarray(pt.all_polygons[grp][sc.tgt]["points"])
        assert np.abs(tgt_pts - sc.truth(moved)).max() < 1.0, "the copy did not follow the edit"
        pt.undo_stack.undo()
        tgt_pts = np.asarray(pt.all_polygons[grp][sc.tgt]["points"])
        assert np.abs(tgt_pts - sc.truth()).max() < 1e-6, "undo did not revert the copy"
        # the non-undoable path (vertex drag) updates the record directly
        pt._propagate_aligned_edit(aligned.vs, grp, sc.src, moved.tolist())
        tgt_pts = np.asarray(pt.all_polygons[grp][sc.tgt]["points"])
        assert np.abs(tgt_pts - sc.truth(moved)).max() < 1.0
    finally:
        pt.all_polygons.pop(grp, None)
        if saved is None:
            pt.__dict__.pop("_save_specific_polygon_file", None)
        else:
            pt._save_specific_polygon_file = saved


def test_the_button_sits_beside_fix(viewer_factory):
    v = viewer_factory()
    zb = v._zoombar
    lay = zb.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert order.index(zb._btn_align) == order.index(zb._btn_fix) + 1
    assert zb._btn_align.width() <= 22


def test_clicked_spots_come_back_as_scene_points(viewer_factory, qapp):
    from PyQt5.QtTest import QTest
    from ..view_alignment_dialog import ClickSpotsMode
    v = viewer_factory()
    v.resize(400, 300)
    q = QtGui.QImage(200, 150, QtGui.QImage.Format_RGB32)
    q.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(q))
    v.show()
    qapp.processEvents()
    got = []
    mode = ClickSpotsMode(v, on_done=got.append)
    for x, y in ((50, 60), (200, 80), (120, 200)):
        QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, pos=QtCore.QPoint(x, y))
    QTest.keyClick(v, QtCore.Qt.Key_Backspace)
    QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, pos=QtCore.QPoint(300, 220))
    QTest.keyClick(v, QtCore.Qt.Key_Return)
    qapp.processEvents()
    assert len(got) == 1 and got[0].shape == (3, 2)
    want = v.mapToScene(QtCore.QPoint(300, 220))
    assert np.abs(got[0][2] - (want.x(), want.y())).max() < 1e-6
    assert not [it for it in v.scene().items() if isinstance(it, QtWidgets.QGraphicsPathItem)
                and it.zValue() >= 1e6], "the crosses were left on the scene"
