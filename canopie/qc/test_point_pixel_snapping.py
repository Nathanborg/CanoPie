"""
A point is ONE exported-frame pixel: clicked, highlighted, drawn, exported and
fed to ML as the same pixel -- and it stays on that pixel when the crop,
rotation or resize changes.

THE BUGS THIS PINS (Nathan, 2026-10-06: "points are completely buggy on
rescaled and cropped images ... I had a way where the clicked pixel would
highlight and be drawn"):

  * the marker was sized raw_w/pixmap_w pixmap units -- the ratio used as a
    WIDTH (inverted) and taken from the raw FILE rather than the export frame:
    3 x 3 export pixels per point on the BCI mosaic's 1/3 preview, 28 x 32 on a
    cropped + px_w-resized one;
  * clicks were no longer snapped to a pixel (5ec209c removed the old floor()
    to a PREVIEW pixel), so neither the hover preview nor the marker showed
    the pixel CSV export would read, and boundary clicks could tip either way;
  * applying a new crop after drawing slid every point to the same RELATIVE
    position in the new frame (110 / 461 raw pixels off on the BCI mosaic);
  * ML export/training clamped out-of-frame points to the edge pixel.
"""
import json
import math
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.extraction, pytest.mark.viewer]


def _click(viewer, vpos):
    for et, bt in ((QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton),
                   (QtCore.QEvent.MouseButtonRelease, QtCore.Qt.NoButton)):
        QtWidgets.QApplication.sendEvent(viewer.viewport(), QtGui.QMouseEvent(
            et, QtCore.QPointF(vpos), QtCore.Qt.LeftButton, bt, QtCore.Qt.ShiftModifier))


def _csv_pixel(pt, group, fp, rec):
    rows, _ = pt.process_polygon(group, fp, rec, {}, [], False, opts={"stats": {"mean": True}})
    r = next((r for r in rows if isinstance(r, dict)), None)
    return None if r is None else (int(r["Centroid X"]), int(r["Centroid Y"]))


@pytest.fixture
def decimated_view(synthetic_project, viewer_factory, force_preview_decimation, monkeypatch):
    pt = synthetic_project
    fp = fixture_image_path("hyperspectral_200band")          # 32 x 32, preview decimated to 16 x 16
    for attr in ("_imgdata_cache", "_pixmap_cache", "_export_image_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    imgd = pt._imagedata_or_fallback(fp)
    assert imgd.image.shape[1] < 32, "fixture no longer decimates -- test would be vacuous"
    v = viewer_factory()
    v.resize(500, 500)
    v.show()
    v.image_data = imgd
    v.set_image(pt.convert_cv_to_pixmap(imgd.image))
    rec = {"viewer": v, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec]
    v.polygon_drawn.connect(pt.on_polygon_drawn)
    # In the app a viewer lives inside its ProjectTab, which is how it finds
    # its owner; a factory viewer has no parent, so point it there.
    monkeypatch.setattr(v, "_find_project_owner", lambda: pt)
    monkeypatch.setattr(pt, "_mark_polygon_dirty", lambda g, f: None)
    v.resetTransform()
    v.scale(20.0, 20.0)
    v.centerOn(v._image.mapToScene(QtCore.QPointF(8.0, 8.0)))
    QtWidgets.QApplication.processEvents()
    groups = []
    try:
        yield pt, v, fp, groups
    finally:
        try:
            v.polygon_drawn.disconnect(pt.on_polygon_drawn)
        except Exception:
            pass
        try:
            pt.viewer_widgets.remove(rec)
        except ValueError:
            pass
        for g in groups:
            pt.all_polygons.pop(g, None)
        for attr in ("_imgdata_cache", "_pixmap_cache", "_export_image_cache"):
            c = getattr(pt, attr, None)
            if isinstance(c, dict):
                c.clear()


def test_click_hover_marker_csv_and_ml_are_the_same_pixel(decimated_view):
    from ..image_viewer import EditablePointItem
    from ..machine_learning_manager import MachineLearningManager
    pt, v, fp, groups = decimated_view
    kx, ky = v._raw_to_pixmap_scale()
    assert kx == pytest.approx(2.0) and ky == pytest.approx(2.0)
    img, info = pt._get_export_image(fp)
    mgr = MachineLearningManager(pt)
    vp = v.viewport().rect()
    for n, (fx, fy) in enumerate(((0.5, 0.5), (0.43, 0.61), (0.58, 0.37), (0.51, 0.49))):
        group = f"snaptest_{n}"
        groups.append(group)
        vpos = QtCore.QPoint(int(vp.width() * fx), int(vp.height() * fy))
        scene = v.mapToScene(vpos)
        lp = v._image.mapFromScene(scene)
        expect = (math.floor(lp.x() * kx), math.floor(lp.y() * ky))
        # hover highlight = that pixel
        hover = v._point_marker_path([], hover_point=scene).boundingRect()
        hl = v._image.mapRectFromScene(hover)
        assert (math.floor(hl.left() * kx + 1e-9), math.floor(hl.top() * ky + 1e-9)) == expect
        assert hl.width() * kx == pytest.approx(1.0) and hl.height() * ky == pytest.approx(1.0)
        # click + commit
        v.drawing_mode = "point"
        v.pending_group_name = group
        _click(v, vpos)
        v.finish_polygon()
        QtWidgets.QApplication.processEvents()
        rec = pt.all_polygons[group][fp]
        it = next(i for i in v._scene.items() if isinstance(i, EditablePointItem) and i.name == group)
        cell = it._cell_rect_item(it.points[0])
        cl = cell.translated(it.pos() - v._image.pos())
        assert (round(cl.left() * kx), round(cl.top() * ky)) == expect, "marker is not the clicked pixel"
        assert cl.width() * kx == pytest.approx(1.0), (
            f"marker is {cl.width() * kx:.2f} export pixels wide; it must be exactly the one pixel "
            "the export reads (it was the raw/pixmap RATIO used as a width).")
        # stored at the pixel centre
        ex = rec["points"][0][0] * info["W"] / rec["image_ref_size"]["w"]
        ey = rec["points"][0][1] * info["H"] / rec["image_ref_size"]["h"]
        assert (ex % 1.0, ey % 1.0) == pytest.approx((0.5, 0.5), abs=1e-6), (
            f"stored point is at export ({ex:.4f}, {ey:.4f}) -- not snapped to a pixel centre")
        assert _csv_pixel(pt, group, fp, rec) == expect
        assert tuple(mgr._point_pixels(fp, rec["points"], img.shape, rec)[0]) == expect


def test_grid_comes_from_the_export_frame_not_the_raw_file(decimated_view, monkeypatch):
    """With a crop/resize the export frame differs from the raw file. The grid
    must follow the export frame (here 3x the 16-px preview), not the raw file
    (2x) -- the raw basis is what gave 28 x 32-pixel markers on a cropped,
    resized BCI mosaic."""
    pt, v, fp, groups = decimated_view
    pm = v._image.pixmap()
    monkeypatch.setattr(pt, "export_frame_hw", lambda viewer, image_data=None: (3 * pm.height(), 3 * pm.width()))
    assert v._raw_to_pixmap_scale() == pytest.approx((3.0, 3.0))


def test_drag_release_snaps_to_the_pixel_under_the_drop(decimated_view):
    from ..image_viewer import EditablePointItem
    pt, v, fp, groups = decimated_view
    kx, _ = v._raw_to_pixmap_scale()
    group = "snaptest_drag"
    groups.append(group)
    v.drawing_mode = "point"
    v.pending_group_name = group
    _click(v, v.viewport().rect().center())
    v.finish_polygon()
    it = next(i for i in v._scene.items() if isinstance(i, EditablePointItem) and i.name == group)
    before = QtCore.QPointF(it.points[0])
    it.moveBy(1.3 / kx, -0.8 / kx)                 # a drag of a fraction of a cell
    it.snap_points_to_cells()
    p = it.points[0]
    assert it.pos() == QtCore.QPointF(0, 0)
    assert ((p.x() * kx) % 1.0, (p.y() * kx) % 1.0) == pytest.approx((0.5, 0.5), abs=1e-9)
    assert (math.floor(p.x() * kx), math.floor(p.y() * kx)) == (
        math.floor((before.x() + 1.3 / kx) * kx), math.floor((before.y() - 0.8 / kx) * kx))


# --------------------------------------------------------------------------
# Crop / rotate / resize applied AFTER drawing: the point stays on its pixel
# --------------------------------------------------------------------------
RAW_W, RAW_H = 70, 53


@pytest.fixture
def raster(tmp_path):
    import tifffile
    rng = np.random.default_rng(5)
    arr = rng.integers(1, 60000, size=(RAW_H, RAW_W, 3), dtype=np.uint16)
    p = str(tmp_path / "pointfollow.tif")
    tifffile.imwrite(p, arr, photometric="rgb")
    return p, arr


def _write_ax(pt, fp, ax):
    axp = pt._ax_path_for(fp)
    if ax is None:
        if os.path.exists(axp):
            os.remove(axp)
    else:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(ax, f)
    for attr in ("_export_image_cache", "_ax_cache", "_ax_json_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    if hasattr(pt, "_invalidate_cached_ax"):
        pt._invalidate_cached_ax(fp)


@pytest.mark.parametrize("new_ax", [
    {"crop_rect": {"x": 11, "y": 6, "width": 40, "height": 33}, "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}},
    {"rotate": 90, "crop_rect": {"x": 4, "y": 9, "width": 40, "height": 50}, "crop_rect_ref_size": {"w": RAW_H, "h": RAW_W}},
    {"rotate": 270},
], ids=["crop", "rot90+crop", "rot270"])
def test_point_stays_on_its_raw_pixel_when_geometry_changes(synthetic_project, raster, monkeypatch, new_ax,
                                                            tmp_path):
    pt = synthetic_project
    fp, raw = raster
    monkeypatch.setattr(pt, "_mark_polygon_dirty", lambda g, f: None)
    # re-projection writes sidecars + the overview: keep them out of the shared project
    (tmp_path / "proj" / "polygons").mkdir(parents=True)
    monkeypatch.setattr(pt, "project_folder", str(tmp_path / "proj"), raising=False)
    _write_ax(pt, fp, None)
    rx, ry = 37, 29                                   # the pixel the user clicked, no .ax
    group = "pointfollow"
    pt.all_polygons[group] = {fp: {'name': 'p', 'type': 'point', 'coord_space': 'image', 'root': '',
                                   'image_ref_size': {'w': RAW_W, 'h': RAW_H}, 'points': [(rx + 0.5, ry + 0.5)]}}
    try:
        snap = pt._snapshot_point_ax_geometry()
        assert fp in snap
        _write_ax(pt, fp, new_ax)
        n = pt._reproject_points_after_ax_change(snap)
        assert n == 1
        rec = pt.all_polygons[group][fp]
        img, info = pt._get_export_image(fp)
        px = _csv_pixel(pt, group, fp, rec)
        assert px is not None, "point dropped although its pixel is still inside the frame"
        assert np.array_equal(np.asarray(img)[px[1], px[0], :3], raw[ry, rx, :3]), (
            f"after the new crop/rotate the point reads export {px} = {np.asarray(img)[px[1], px[0], :3]}, "
            f"not its original pixel's value {raw[ry, rx, :3]} -- it slid to the same RELATIVE position.")
    finally:
        pt.all_polygons.pop(group, None)
        _write_ax(pt, fp, None)


def test_ml_does_not_clamp_an_out_of_frame_point(synthetic_project, raster):
    from ..machine_learning_manager import MachineLearningManager
    pt = synthetic_project
    fp, _ = raster
    rec = {'type': 'point', 'coord_space': 'image', 'image_ref_size': {'w': RAW_W, 'h': RAW_H},
           'points': [(-3.5, 10.5), (RAW_W + 2.5, 10.5), (12.5, 10.5)]}
    px = MachineLearningManager(pt)._point_pixels(fp, rec['points'], (RAW_H, RAW_W, 3), rec)
    assert px[0][0] < 0 and px[1][0] >= RAW_W, (
        f"out-of-frame points mapped to {px[:2]} -- clamped onto the edge pixel, so ML exported/"
        "trained on a pixel the point is not on")
    assert tuple(px[2]) == (12, 10)
