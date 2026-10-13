"""
"Find canopy in other images…" (2026-10-10): a crown drawn on one GeoTIFF is
taken through world coordinates into other GeoTIFFs and refined by the
notebook's registration engines (canopie/canopy_finder.py), then saved / zoomed /
exported / browsed (canopy_finder_dialog.py, canopy_browser.py).

Synthetic GeoTIFFs are written at runtime: a texture of crowns, and targets
showing the same ground with a KNOWN georeference error (shift + rotation) at
another resolution and CRS -- so the true position of the crown is known.
"""
import json
import math
import os
import time

import cv2
import numpy as np
import pytest
import tifffile

from .. import canopy_finder as CF

pytestmark = [pytest.mark.polygons]


@pytest.fixture(autouse=True, scope="module")
def _private_exif_cache(tmp_path_factory):
    """The EXIF cache lives in the user's profile; the tests keep theirs apart."""
    from .. import canopy_finder_meta as M
    old = M._SHARED_DIR
    M.set_shared_cache_dir(str(tmp_path_factory.mktemp("exif_cache")))
    yield
    M.set_shared_cache_dir(old)

GSD = 0.15
OX, OY = 500000.0, 9700000.0          # UTM 21S (EPSG:32721) origin of the source


def _write_geotiff(path, arr, origin_x, origin_y, gsd_x, gsd_y=None, epsg=32721):
    gk = [1, 1, 0, 3, 1024, 0, 1, 2 if epsg == 4326 else 1, 1025, 0, 1, 1,
          2048 if epsg == 4326 else 3072, 0, 1, int(epsg)]
    tifffile.imwrite(path, arr, photometric="rgb" if arr.ndim == 3 and arr.shape[2] == 3 else "minisblack",
                     extratags=[(33922, "d", 6, (0, 0, 0, float(origin_x), float(origin_y), 0), True),
                                (33550, "d", 3, (float(gsd_x), float(gsd_y or gsd_x), 0), True),
                                (34735, "H", len(gk), tuple(gk), True)])
    return path


def _world(n=900, seed=5):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 3), np.float32) + (40, 70, 30)
    for _ in range(int(n * n / 10000)):
        cx, cy = rng.uniform(0, n, 2)
        col = rng.uniform((30, 90, 20), (120, 200, 80))
        cv2.circle(img, (int(cx), int(cy)), int(rng.uniform(12, 45)), tuple(float(c) for c in col), -1)
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.2), 0, 255).astype(np.uint8)


class Scene:
    """Source GeoTIFF + a target with a known georeference error."""

    def __init__(self, d, shift=(7.3, -2.0), theta=-3.0, tgsd=0.2, epsg_t=32721):
        self.B = _world()
        n = self.B.shape[0]
        self.src = _write_geotiff(os.path.join(d, "src.tif"), self.B, OX, OY, GSD)
        self.c = np.array([OX + n / 2 * GSD, OY - n / 2 * GSD])
        th = math.radians(theta)
        self.R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        self.s = np.array(shift, np.float64)
        self.tgsd, self.epsg_t = tgsd, epsg_t
        self.tox, self.toy = OX + 6.0, OY - 4.0
        m = self.m = int(n * GSD / tgsd) - 40
        yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
        w = np.column_stack(((self.tox + (xx + 0.5) * tgsd).ravel(), (self.toy - (yy + 0.5) * tgsd).ravel()))
        w = (w - self.c - self.s) @ self.R + self.c            # target-georef world -> true world
        mx = ((w[:, 0] - OX) / GSD - 0.5).reshape(m, m).astype(np.float32)
        my = ((OY - w[:, 1]) / GSD - 0.5).reshape(m, m).astype(np.float32)
        T = cv2.remap(self.B, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        T = np.clip(T.astype(np.float32) * 0.8 + 20, 0, 255).astype(np.uint8)
        if epsg_t == 32721:
            self.tgt = _write_geotiff(os.path.join(d, "tgt.tif"), T, self.tox, self.toy, tgsd)
        else:
            raise ValueError(epsg_t)
        a = np.linspace(0, 2 * math.pi, 13)[:-1]
        self.P = np.column_stack((470 + 40 * np.cos(a), 430 + 33 * np.sin(a)))

    def truth(self, P=None):
        P = self.P if P is None else P
        w = np.column_stack((OX + P[:, 0] * GSD, OY - P[:, 1] * GSD))
        w = (w - self.c) @ self.R.T + self.c + self.s
        return np.column_stack(((w[:, 0] - self.tox) / self.tgsd, (self.toy - w[:, 1]) / self.tgsd))

    def record(self):
        n = self.B.shape[0]
        return {"points": self.P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n},
                "type": "polygon"}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return Scene(str(tmp_path_factory.mktemp("canopy_scene")))


def test_georeference_round_trip_across_crs_and_resolution(tmp_path):
    from pyproj import Transformer
    a = _write_geotiff(str(tmp_path / "utm.tif"), np.zeros((50, 60, 3), np.uint8), OX, OY, 0.15)
    lon, lat = Transformer.from_crs(32721, 4326, always_xy=True).transform(OX - 3, OY + 2)
    b = _write_geotiff(str(tmp_path / "geo.tif"), np.zeros((40, 40, 3), np.uint8), lon, lat,
                       1.6e-6, 1.6e-6, epsg=4326)
    ga, gb = CF.georef(a), CF.georef(b)
    assert (ga.epsg, gb.epsg) == (32721, 4326)
    p = np.array([[0.0, 0.0], [12.5, 7.25], [60, 50]])
    back = CF.map_px(gb, ga, CF.map_px(ga, gb, p))
    assert np.abs(back - p).max() < 0.01
    A = CF.local_affine(ga, gb, 0, 0, 60, 50)
    assert np.abs(p @ A[:, :2].T + A[:, 2] - CF.map_px(ga, gb, p)).max() < 0.5
    gx, gy = CF.gsd_m(gb, 10, 10)
    assert 0.17 < gx < 0.18 and 0.17 < gy < 0.18, (gx, gy)       # 1.6e-6 deg near the equator


@pytest.mark.parametrize("method", ["exhaustive", "stackreg"])
def test_the_crown_is_found_through_a_known_shift_rotation_and_resolution(scene, method):
    ctx = CF.prepare_source(scene.src, scene.record(), {}, {"method": method}, group="g")
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": method})
    truth = scene.truth()
    geo_err = np.abs(r.geo_pts - truth).max()
    assert geo_err > 20, "precondition: the georeference error is large"
    assert r.status == "found", (r.status, r.message, r.score)
    err = np.abs(np.asarray(r.raw_pts) - truth).max()
    assert err < 1.0, f"{method}: crown {err:.2f} px off (geo only: {geo_err:.1f} px)"
    assert abs(abs(r.angle) - 3.0) < 0.6, r.angle
    assert 6.5 < r.offset_m < 8.5, r.offset_m


def test_geo_only_is_the_plain_projection(scene):
    ctx = CF.prepare_source(scene.src, scene.record(), {}, {"method": "geo"})
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": "geo"})
    assert r.status == "geo"
    w = np.column_stack((OX + scene.P[:, 0] * GSD, OY - scene.P[:, 1] * GSD))
    exp = np.column_stack(((w[:, 0] - scene.tox) / scene.tgsd, (scene.toy - w[:, 1]) / scene.tgsd))
    assert np.abs(r.raw_pts - exp).max() < 1e-6


def test_records_map_through_the_ax_crop_both_ways(scene):
    n = scene.B.shape[0]
    ax = {"crop_rect": {"x": 100, "y": 50, "width": 600, "height": 500}, "crop_rect_ref_size": {"w": n, "h": n}}
    # a record of the cropped frame: stored relative to that frame, in raw-size units
    e = scene.P - (100, 50)
    rec = {"points": (e * (n / 600.0, n / 500.0)).tolist(), "coord_space": "image",
           "image_ref_size": {"w": n, "h": n}}
    assert np.abs(CF.record_to_raw(rec, ax, n, n) - scene.P).max() < 1e-9
    pts, ref, inside = CF.raw_to_record(scene.P, ax, n, n)
    assert inside and ref == {"w": n, "h": n}
    assert np.abs(np.asarray(pts) - np.asarray(rec["points"])).max() < 1e-9
    _p, _r, inside = CF.raw_to_record(scene.P, {"crop_rect": {"x": 0, "y": 0, "width": 200, "height": 200},
                                                "crop_rect_ref_size": {"w": n, "h": n}}, n, n)
    assert not inside
    # end to end: the source works in its post-.ax (cropped) frame -- what its viewer shows
    ctx = CF.prepare_source(scene.src, rec, ax, {"method": "geo"})
    assert ctx.geo.frame == "export" and (ctx.geo.raw_w, ctx.geo.raw_h) == (600, 500)
    assert np.abs(ctx.raw_pts - e).max() < 1e-9
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": "geo"})
    w = np.column_stack((OX + scene.P[:, 0] * GSD, OY - scene.P[:, 1] * GSD))
    exp = np.column_stack(((w[:, 0] - scene.tox) / scene.tgsd, (scene.toy - w[:, 1]) / scene.tgsd))
    assert r.status == "geo" and np.abs(r.raw_pts - exp).max() < 1e-6, "the crop is undone through the georeference"
    # a target whose .ax crop misses the crown says so
    r = CF.find_in_target(ctx, scene.tgt, {"crop_rect": {"x": 0, "y": 0, "width": 50, "height": 50},
                                           "crop_rect_ref_size": {"w": 999, "h": 999}}, {"method": "geo"})
    assert r.status == "outside_frame"
    # and one whose crop holds it: found in ITS cropped frame, stored for that frame
    tax = {"crop_rect": {"x": 200, "y": 150, "width": 300, "height": 300},
           "crop_rect_ref_size": {"w": scene.m, "h": scene.m}}
    r = CF.find_in_target(ctx, scene.tgt, tax, {"method": "exhaustive"})
    truth = scene.truth() - (200, 150)
    assert r.status == "found" and np.abs(np.asarray(r.raw_pts) - truth).max() < 1.0
    assert r.image_ref_size == {"w": scene.m, "h": scene.m}
    assert np.abs(np.asarray(r.record_points) - truth * (scene.m / 300.0)).max() < 1.0 * scene.m / 300.0


def test_pixels_come_from_the_ax_replay(scene, tmp_path):
    """A 90 degree .ax rotation + a NoData value: the crown is read and matched in
    the rotated frame, the NoData value is not data, and the exported square is
    georeferenced through the rotation."""
    n = scene.B.shape[0]
    ax = {"rotate": 90, "nodata_values": [0], "nodata_enabled": True}
    fr = CF.frame_for(scene.src, ax=ax)
    assert (fr.raw_w, fr.raw_h) == (n, n) and fr.frame == "export"
    e = np.asarray(CF.G.forward_np(CF.G.frame_plan(ax, n, n)[0], scene.P))
    rec = {"points": e.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    ctx = CF.prepare_source(scene.src, rec, ax, {"method": "exhaustive"})
    r0, c0, r1, c1 = ctx.bbox
    raw = tifffile.imread(scene.src)
    want = np.rot90(raw, k=3)[r0:r1, c0:c1]                     # the viewer's frame
    got, _v, _f, _o = CF.Frames().read(scene.src, ax, c0, r0, c1, r1, [0, 1, 2], zero_is_nodata=False)
    assert np.allclose(got, want), "the crop must come from the rotated (post-.ax) frame"
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": "exhaustive"})
    assert r.status == "found" and np.abs(np.asarray(r.raw_pts) - scene.truth()).max() < 1.0
    files = CF.export_source_square(ctx, str(tmp_path))
    from ..shapefile_io import get_geotiff_transform
    t = tifffile.TiffFile(files).pages[0].tags
    assert 34264 in t or 33922 in t
    if 34264 in t:                                               # rotated frame: a ModelTransformation
        M = t[34264].value
        assert abs(M[0]) < 1e-9 and abs(M[1]) > 0


def test_a_target_not_covering_the_crown_or_without_georeference(scene, tmp_path):
    far = _write_geotiff(str(tmp_path / "far.tif"), np.full((50, 50, 3), 90, np.uint8), OX + 5000, OY, 0.2)
    plain = str(tmp_path / "plain.tif")
    tifffile.imwrite(plain, np.zeros((20, 20, 3), np.uint8), photometric="rgb")
    ctx = CF.prepare_source(scene.src, scene.record(), {}, {})
    assert CF.find_in_target(ctx, far, {}, {}).status == "outside_footprint"
    assert CF.find_in_target(ctx, plain, {}, {}).status == "no_georef"


def test_the_exported_square_is_georeferenced_and_holds_the_target_pixels(scene, tmp_path):
    from ..shapefile_io import get_geotiff_transform
    ctx = CF.prepare_source(scene.src, scene.record(), {}, {"method": "stackreg"})
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": "stackreg"})
    files = CF.export_result(ctx, r, str(tmp_path), native=True, aligned=True, quicklook=True)
    sq = next(f for f in files if f.endswith("_square.tif"))
    x0, y0, x1, y1 = r.square
    t = get_geotiff_transform(sq)[0]
    assert abs(t[0] - (scene.tox + x0 * scene.tgsd)) < 1e-6 and abs(t[3] - (scene.toy - y0 * scene.tgsd)) < 1e-6
    assert CF.georef(sq).epsg == 32721
    full = tifffile.imread(scene.tgt)
    assert np.array_equal(tifffile.imread(sq), full[y0:y1, x0:x1])
    # the square holds the found crown
    c = np.asarray(r.raw_pts).mean(axis=0)
    assert x0 <= c[0] <= x1 and y0 <= c[1] <= y1
    CF.write_summary([r], str(tmp_path), ctx)
    s = json.load(open(tmp_path / "summary.json", encoding="utf-8"))
    assert s["targets"][0]["status"] == "found" and s["targets"][0]["square"] == list(r.square)


# ───────────────────────── the app: menus, dialog, browser ─────────────────────────
@pytest.fixture
def geo_project(tmp_path, qapp, scene):
    """A real ProjectTab: one root of source + target + a second shifted target."""
    import shutil
    from PyQt5 import QtWidgets
    from ..project_tab import ProjectTab
    d = tmp_path / "proj"
    d.mkdir()
    imgs = []
    for name, srcp in (("a_src.tif", scene.src), ("b_tgt.tif", scene.tgt), ("c_tgt.tif", scene.tgt)):
        p = str(tmp_path / name)
        shutil.copyfile(srcp, p)
        imgs.append(p.replace("\\", "/"))
    n = scene.B.shape[0]
    data = {"project_name": "geo", "mode": "rgb_only", "root_offset": 0, "current_root_index": 0,
            "root_coordinates": {}, "rgb_folder_path": str(tmp_path), "batch_size": 3,
            "multispectral_root_names": ["R1"], "multispectral_image_data_groups": {"R1": imgs},
            "all_polygons": {"crown": {imgs[0]: dict(scene.record(), name="crown", root="1")},
                             "other": {imgs[2]: {"points": [[5, 5], [30, 5], [30, 30]], "coord_space": "image",
                                                 "image_ref_size": {"w": 999, "h": 999}, "name": "other",
                                                 "root": "1", "type": "polygon"}}}}
    json.dump(data, open(d / "project.json", "w", encoding="utf-8"))
    tabs = QtWidgets.QTabWidget()
    pt = ProjectTab("geo", tabs)
    pt.load_project(str(d))
    _pump(qapp, 10.0, lambda: len(pt.viewer_widgets) == 3 and all(
        getattr(w["viewer"], "_image", None) is not None for w in pt.viewer_widgets) and any(
        getattr(p.get("item"), "name", "") == "crown" for p in pt.viewer_widgets[0]["viewer"].polygons))
    yield pt, imgs
    dlg = getattr(pt, "_canopy_finder_dialog", None)
    if dlg is not None:
        dlg.close()
        if dlg.browser is not None:
            dlg.browser.close()
    pt.deleteLater()


def _pump(qapp, sec, until=None):
    t = time.time()
    while time.time() - t < sec:
        qapp.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until() if until is not None else True


def _wait_scan(qapp, dlg):
    """The target list is filled off the UI thread."""
    _pump(qapp, 60, lambda: (dlg.scan_worker is None or not dlg.scan_worker.isRunning())
          and dlg.btn_find.isEnabled())
    _pump(qapp, 0.1)


def _targets(dlg):
    from PyQt5 import QtCore
    return [dlg.target_list.item(i) for i in range(dlg.target_list.count())
            if dlg.target_list.item(i).data(QtCore.Qt.UserRole)]


def _run(qapp, dlg):
    _wait_scan(qapp, dlg)
    dlg.start()
    assert dlg.worker is not None
    _pump(qapp, 60, lambda: not dlg.worker.isRunning())
    _pump(qapp, 0.6)


def test_both_menus_offer_the_finder(geo_project, monkeypatch, qapp):
    from PyQt5 import QtWidgets
    from .. import image_viewer as IV
    pt, imgs = geo_project
    seen = []

    def fake_exec(menu, *a, **k):
        seen.append([act.text() for act in menu.actions()])
        return None
    monkeypatch.setattr(QtWidgets.QMenu, "exec_", fake_exec)
    v = pt.viewer_widgets[0]["viewer"]
    item = next(p["item"] for p in v.polygons if getattr(p.get("item"), "name", "") == "crown")
    item.contextMenuEvent(None)
    acts = seen[-1]
    assert acts.index("Find canopy in other images…") == acts.index("Edit all polygons in this group") + 1
    pm = pt.polygon_manager
    pt.show_polygon_manager()
    lw = pm.list_widget
    row = next(i for i in range(lw.count()) if lw.item(i).data(0x0100) == "crown")
    pm.show_context_menu(lw.visualItemRect(lw.item(row)).center())
    assert seen[-1].index("Find canopy in other images…") == seen[-1].index("Edit") + 1
    pm.hide()


def test_the_dialog_finds_saves_zooms_and_keeps_existing(geo_project, qapp, scene, monkeypatch):
    from PyQt5 import QtWidgets
    from ..canopy_finder_dialog import existing_record, live_viewer_for, record_scene_polygon
    pt, imgs = geo_project
    pt.polygon_manager.find_canopy_for_group("crown")
    dlg = pt._canopy_finder_dialog
    assert dlg.source_fp() == imgs[0]
    _wait_scan(qapp, dlg)
    assert len(_targets(dlg)) == 2
    dlg.cb_browser.setChecked(False)
    _run(qapp, dlg)
    assert [r.status for r, _t, _s in dlg.results] == ["found", "found"]
    assert [s for _r, _t, s in dlg.results] == ["saved", "saved"]
    truth = scene.truth()
    for fp in imgs[1:]:
        _k, rec = existing_record(pt, "crown", fp)
        assert rec is not None and rec["image_ref_size"] == {"w": scene.m, "h": scene.m}
        assert np.abs(np.asarray(rec["points"]) - truth).max() < 1.0
        assert os.path.isfile(os.path.join(pt.project_folder, "polygons",
                                           f"crown_{os.path.splitext(os.path.basename(fp))[0]}_polygons.json"))
        v = live_viewer_for(pt, fp)
        rect = record_scene_polygon(v, rec["points"], rec["image_ref_size"]).boundingRect()
        c = v.mapToScene(v.viewport().rect().center())
        assert rect.adjusted(-3, -3, 3, 3).contains(c), "the viewer was not zoomed onto the crown"
    # again: the records exist -> kept, unless replaced (zip backup first)
    _k, before = existing_record(pt, "crown", imgs[1])
    before_pts = [tuple(p) for p in before["points"]]
    dlg.method.setCurrentIndex(dlg.method.findData("geo"))
    _run(qapp, dlg)
    assert [s for _r, _t, s in dlg.results] == ["exists", "exists"]
    assert [tuple(p) for p in existing_record(pt, "crown", imgs[1])[1]["points"]] == before_pts
    row = next(i for i, (_r, t, _s) in enumerate(dlg.results) if t["fp"] == imgs[1])     # completion order
    assert dlg.save_row(row, replace=True) == "replaced"
    bk = os.path.join(pt.project_folder, "polygons", ".reproject_backup")
    assert os.path.isdir(bk) and any("b_tgt" in f for f in os.listdir(bk))
    assert [tuple(p) for p in existing_record(pt, "crown", imgs[1])[1]["points"]] != before_pts


def test_other_project_images_are_never_written(geo_project, qapp, scene, tmp_path):
    import shutil
    pt, imgs = geo_project
    ext = tmp_path / "ext"
    ext.mkdir()
    shutil.copyfile(scene.tgt, ext / "e.tif")
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(False)
    dlg.add_folders([str(ext)])
    assert dlg.rb_ext.isChecked()
    _wait_scan(qapp, dlg)
    assert len(_targets(dlg)) == 1
    _run(qapp, dlg)
    (r, t, s), = dlg.results
    assert r.status == "found" and s == "external"
    pdir = os.path.join(pt.project_folder, "polygons")
    assert not (os.path.isdir(pdir) and any("e_polygons" in f for f in os.listdir(pdir)))
    assert not any(f.endswith("_polygons.json") for f in os.listdir(ext))
    assert all(os.path.normcase(k) != os.path.normcase(str(ext / "e.tif"))
               for k in pt.all_polygons.get("crown", {}))


def test_the_browser_pages_and_its_edits_update_the_record(geo_project, qapp):
    from PyQt5 import QtCore
    from ..canopy_finder_dialog import existing_record
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(True)
    _run(qapp, dlg)
    b = dlg.browser
    assert b is not None and b.isVisible()
    assert [e["kind"] for e in b.entries] == ["source", "target", "target"]
    b.cols.setValue(1); b.rows.setValue(1)
    assert b.n_pages() == 3 and len(b.tiles) == 1
    b.set_page(1)
    tile = b.tiles[0][0]
    fp = tile.entry["fp"]
    b.edit.setChecked(True)
    h = tile._handles[0]
    h.setPos(h.pos() + QtCore.QPointF(4, 0))
    tile._edit_finished()
    _k, rec = existing_record(pt, "crown", fp)
    g = CF.georef(fp)
    pts, ref, _ = CF.raw_to_record(tile.entry["pts"], {}, g.raw_w, g.raw_h)
    assert np.abs(np.asarray(rec["points"]) - np.asarray(pts)).max() < 1e-9
    # sync zoom: same ground extent in metres
    b.cols.setValue(3)
    t0, t1 = b.tiles[0][0], b.tiles[1][0]
    t0.scale(2.0, 2.0)
    b._on_view_changed(t0)
    assert abs(t0.ground_state()[0] - t1.ground_state()[0]) < 1e-6


# ───────────────────── round 2: raw drone frames, phenocams, COGs ─────────────────────
FOCAL, F35 = 8.8, 24.0                  # sensor width 13.2 mm (35 mm equivalent)
RW, RH = 800, 600                       # raw frame size
AGL = 60.0                              # -> 0.1125 m / px


def _raw_frame(d, name, B, E, N, yaw, err_m=(0.0, 0.0), err_agl=1.0, err_yaw=0.0):
    """A synthetic nadir frame of the source world (EPSG:32721, GSD) seen by a
    camera at (E, N) heading `yaw`; its EXIF is injected with the given errors."""
    from pyproj import Transformer
    from .. import canopy_finder_meta as M
    to_ll = Transformer.from_crs(32721, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 32721, always_xy=True)
    lon, lat = to_ll.transform(E, N)
    tr, _ = M.exif_transform({"lat": lat, "lon": lon, "alt_rel": AGL, "focal": FOCAL, "focal35": F35, "yaw": yaw},
                             RW, RH)
    g = CF.GeoRef(path=name, transform=tr, epsg=4326, crs_wkt=None, raw_w=RW, raw_h=RH, kind="exif")
    yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float64)
    wll = g.px_to_world(np.column_stack(((xx + 0.5).ravel(), (yy + 0.5).ravel())))
    e, n = to_utm.transform(wll[:, 0], wll[:, 1])
    mx = ((np.asarray(e) - OX) / GSD - 0.5).reshape(RH, RW).astype(np.float32)
    my = ((OY - np.asarray(n)) / GSD - 0.5).reshape(RH, RW).astype(np.float32)
    img = cv2.remap(B, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = np.clip(img.astype(np.float32) * 0.9 + 15, 0, 255).astype(np.uint8)
    p = os.path.join(d, name + ".jpg")
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    lon2, lat2 = to_ll.transform(E + err_m[0], N + err_m[1])
    M._put(p, {"lat": lat2, "lon": lon2, "alt_rel": AGL * err_agl, "focal": FOCAL, "focal35": F35,
               "yaw": yaw + err_yaw, "full": True})
    return p, g


def _truth_in(g, P):
    from pyproj import Transformer
    to_ll = Transformer.from_crs(32721, 4326, always_xy=True)
    w = np.column_stack((OX + P[:, 0] * GSD, OY - P[:, 1] * GSD))
    lon, lat = to_ll.transform(w[:, 0], w[:, 1])
    return g.world_to_px(np.column_stack((lon, lat)))


class RawScene:
    def __init__(self, d):
        self.B = _world(1100, seed=9)
        n = self.B.shape[0]
        self.ortho = _write_geotiff(os.path.join(d, "ortho.tif"), self.B, OX, OY, GSD)
        a = np.linspace(0, 2 * math.pi, 13)[:-1]
        self.P = np.column_stack((560 + 40 * np.cos(a), 520 + 33 * np.sin(a)))
        rng = np.random.default_rng(3)
        self.frames = {}
        for i in range(4):
            for j in range(3):
                name = f"f{i}{j}"
                self.frames[name] = _raw_frame(d, name, self.B, OX + 25 + i * 40.0, OY - 30 - j * 40.0, 30.0,
                                               err_m=tuple(rng.normal(0, 2.5, 2)), err_agl=1.0 + rng.normal(0, 0.04),
                                               err_yaw=rng.normal(0, 3))
        self.rec = {"points": self.P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}

    def g_true(self, path):
        return self.frames[os.path.splitext(os.path.basename(path))[0]][1]


@pytest.fixture(scope="module")
def raw_scene(tmp_path_factory):
    return RawScene(str(tmp_path_factory.mktemp("canopy_raw")))


def test_exif_frame_model_turns_with_the_heading():
    from .. import canopy_finder_meta as M
    tr, info = M.exif_transform({"lat": 0.0, "lon": 0.0, "alt_rel": 60, "focal": 8.8, "focal35": 24, "yaw": 90},
                                800, 600)
    assert abs(info["gsd_m"] - 0.1125) < 1e-9
    g = CF.GeoRef(path="x", transform=tr, epsg=4326, crs_wkt=None, raw_w=800, raw_h=600, kind="exif")
    c, top, right = g.px_to_world([(400, 300), (400, 0), (800, 300)])
    assert top[0] > c[0] and abs(top[1] - c[1]) < 1e-9, "heading east: the top of the frame is east"
    assert right[1] < c[1] and abs(right[0] - c[0]) < 1e-9, "heading east: the right of the frame is south"
    # no height above ground: no model from EXIF (the frame's overlap measures it instead)
    tr0, why = M.exif_transform({"lat": 0, "lon": 0, "focal": 8.8, "focal35": 24}, 800, 600)
    assert tr0 is None and "overlap" in why


def test_the_kdtree_picks_the_frames_nearest_the_crown(raw_scene):
    from pyproj import Transformer
    from .. import canopy_finder_meta as M
    paths = [v[0] for v in raw_scene.frames.values()]
    pos = M.positions(paths)
    assert len(pos) == len(paths)
    cE, cN = OX + raw_scene.P[:, 0].mean() * GSD, OY - raw_scene.P[:, 1].mean() * GSD
    lon, lat = Transformer.from_crs(32721, 4326, always_xy=True).transform(cE, cN)
    near = M.nearest(pos, lat, lon, k=3)
    assert len(near) == 3 and near[0][1] <= near[1][1] <= near[2][1]
    # the truly nearest frames (by their TRUE centres, not the noisy EXIF)
    true_d = sorted((math.hypot(25 + i * 40 - (cE - OX), -30 - j * 40 - (cN - OY)), f"f{i}{j}")
                    for i in range(4) for j in range(3))
    picked = {os.path.splitext(os.path.basename(p))[0] for p, _d in near}
    assert true_d[0][1] in picked and true_d[1][1] in picked


@pytest.mark.parametrize("method", ["stackreg", "exhaustive"])
def test_orthomosaic_to_raw_frames_at_a_finer_resolution(raw_scene, method):
    from pyproj import Transformer
    from .. import canopy_finder_meta as M
    pos = M.positions([v[0] for v in raw_scene.frames.values()])
    cE, cN = OX + raw_scene.P[:, 0].mean() * GSD, OY - raw_scene.P[:, 1].mean() * GSD
    lon, lat = Transformer.from_crs(32721, 4326, always_xy=True).transform(cE, cN)
    ctx = CF.prepare_source(raw_scene.ortho, raw_scene.rec, {}, {"method": method})
    for p, _d in M.nearest(pos, lat, lon, k=2):
        r = CF.find_in_target(ctx, p, {}, {"method": method})
        truth = _truth_in(raw_scene.g_true(p), raw_scene.P)
        assert r.extra.get("pair") == "exif"
        assert r.status == "found", (r.status, r.message, r.score)
        geo_err = np.abs(r.geo_pts - truth).max()
        err = np.abs(np.asarray(r.raw_pts) - truth).max()
        assert geo_err > 15, "precondition: the EXIF projection is off"
        assert err < 2.5, f"{method}: {err:.2f} raw px off (EXIF only: {geo_err:.1f})"
        assert abs(r.extra["work_gsd_m"] - GSD) < 1e-6, "matched at the coarser (ortho) resolution"


def test_raw_frame_to_orthomosaic_and_to_another_frame(raw_scene):
    from pyproj import Transformer
    from .. import canopy_finder_meta as M
    pos = M.positions([v[0] for v in raw_scene.frames.values()])
    cE, cN = OX + raw_scene.P[:, 0].mean() * GSD, OY - raw_scene.P[:, 1].mean() * GSD
    lon, lat = Transformer.from_crs(32721, 4326, always_xy=True).transform(cE, cN)
    (p0, _), (p1, _) = M.nearest(pos, lat, lon, k=2)
    praw = _truth_in(raw_scene.g_true(p0), raw_scene.P)
    rec = {"points": praw.tolist(), "coord_space": "image", "image_ref_size": {"w": RW, "h": RH}}
    ctx = CF.prepare_source(p0, rec, {}, {"method": "stackreg"})
    assert ctx.geo.kind == "exif"
    r = CF.find_in_target(ctx, raw_scene.ortho, {}, {"method": "stackreg"})
    assert r.status == "found" and np.abs(np.asarray(r.raw_pts) - raw_scene.P).max() < 1.5
    r = CF.find_in_target(ctx, p1, {}, {"method": "stackreg"})
    truth = _truth_in(raw_scene.g_true(p1), raw_scene.P)
    # worst vertex of two EXIF-estimated frames: within 3 px (0.34 m at 0.11 m / px)
    assert r.status == "found" and np.abs(np.asarray(r.raw_pts) - truth).max() < 3.0


def test_phenocam_frames_without_location_match_in_image_coordinates(tmp_path):
    B = _world(500, seed=11)
    M_ = np.float32([[1, 0, 9.0], [0, 1, -6.0]])                      # the camera moved 9 / -6 px
    T = cv2.warpAffine(B, M_, (500, 500), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    T = np.clip(T.astype(np.float32) * 0.85 + 25, 0, 255).astype(np.uint8)
    a, b = str(tmp_path / "cam_day1.png"), str(tmp_path / "cam_day2.png")
    cv2.imwrite(a, cv2.cvtColor(B, cv2.COLOR_RGB2BGR)); cv2.imwrite(b, cv2.cvtColor(T, cv2.COLOR_RGB2BGR))
    t = np.linspace(0, 2 * math.pi, 11)[:-1]
    P = np.column_stack((240 + 35 * np.cos(t), 260 + 28 * np.sin(t)))
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": 500, "h": 500}}
    ctx = CF.prepare_source(a, rec, {}, {"method": "exhaustive"})
    assert ctx.geo.kind == "pixel"
    r = CF.find_in_target(ctx, b, {}, {"method": "exhaustive"})
    assert r.extra["pair"] == "pixel" and r.status == "found", (r.status, r.message)
    assert np.abs(np.asarray(r.raw_pts) - (P + (9.0, -6.0))).max() < 1.0
    # a located image and a phenocam can't be related
    geo = _write_geotiff(str(tmp_path / "geo.tif"), B, OX, OY, GSD)
    assert CF.find_in_target(ctx, geo, {}, {}).status == "no_georef"


def _write_cog(path, arr, gsd, levels=2):
    gk = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32721]
    tags = [(33922, "d", 6, (0, 0, 0, OX, OY, 0), True), (33550, "d", 3, (gsd, gsd, 0), True),
            (34735, "H", len(gk), tuple(gk), True)]
    with tifffile.TiffWriter(path) as tif:
        tif.write(arr, subifds=levels, extratags=tags, photometric="rgb", tile=(256, 256))
        lv = arr
        for _ in range(levels):
            lv = cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA)
            kw = {"tile": (256, 256)} if min(lv.shape[:2]) >= 256 else {}
            tif.write(lv, subfiletype=1, photometric="rgb", **kw)
    return path


def test_a_cog_target_is_read_from_its_overview_at_the_working_resolution(tmp_path, monkeypatch):
    from .. import raster_reader as RR
    B = _world(700, seed=13)                                           # 0.15 m source
    src = _write_geotiff(str(tmp_path / "src.tif"), B, OX, OY, GSD)
    fine = cv2.resize(B, (2100, 2100), interpolation=cv2.INTER_CUBIC)  # 0.05 m COG of the same ground
    cog = _write_cog(str(tmp_path / "fine_cog.tif"), fine, GSD / 3.0)
    a = np.linspace(0, 2 * math.pi, 11)[:-1]
    P = np.column_stack((350 + 35 * np.cos(a), 330 + 30 * np.sin(a)))
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": 700, "h": 700}}
    levels = []
    orig = RR.TiledRasterReader.read_window

    def spy(self, *a, **k):
        levels.append(k.get("level", 0))
        return orig(self, *a, **k)
    monkeypatch.setattr(RR.TiledRasterReader, "read_window", spy)
    ctx = CF.prepare_source(src, rec, {}, {"method": "stackreg"})
    levels.clear()
    r = CF.find_in_target(ctx, cog, {}, {"method": "stackreg"})
    assert r.status == "found", (r.status, r.message)
    assert levels and max(levels) == 1, f"read at levels {levels}, expected the 2x overview"
    assert np.abs(np.asarray(r.raw_pts) - P * 3.0).max() < 3.0          # within one working pixel


def test_save_all_and_a_dragged_grid_outline(geo_project, qapp, monkeypatch):
    from PyQt5 import QtCore, QtWidgets
    from ..canopy_finder_dialog import existing_record, TempOutline
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(False)
    dlg.cb_save.setChecked(False)                 # nothing saved automatically
    dlg.cb_zoom.setChecked(True)
    _run(qapp, dlg)
    _pump(qapp, 0.5)
    outlines = [it for _v, it in dlg._temp_items if isinstance(it, TempOutline)]
    assert len(outlines) == 2, "unsaved results are drawn as dashed outlines"
    # drag the outline on b_tgt by 10 scene px to the right
    it = next(o for o in outlines if "b_tgt" in o.viewer.image_data.filepath)
    r = dlg.results[it.row][0]
    before = np.asarray(r.record_points, np.float64).copy()
    it.setPos(it.pos() + QtCore.QPointF(10, 0))
    dlg.temp_outline_moved(it)
    after = np.asarray(r.record_points, np.float64)
    assert np.allclose(after[:, 1], before[:, 1]) and (after[:, 0] - before[:, 0]).min() > 1, \
        "the result follows the dragged outline"
    # c_tgt already holds a crown of this group? no -> plant one, Save all asks once
    pt.all_polygons["crown"][imgs[2]] = dict(pt.all_polygons["other"][imgs[2]], name="crown")
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No))
    assert dlg.save_all() == 1 and len(asked) == 1
    _k, rec = existing_record(pt, "crown", imgs[1])
    assert np.abs(np.asarray(rec["points"]) - after).max() < 1e-9, "saved where it was dropped"
    assert [s for _r, _t, s in dlg.results][it.row] == "saved"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    assert dlg.save_all() == 1                    # now the existing one, replaced
    assert all(s in ("saved", "replaced") for _r, _t, s in dlg.results)


def test_a_browser_outline_is_moved_as_a_whole(geo_project, qapp):
    from PyQt5 import QtCore
    from ..canopy_finder_dialog import existing_record
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(True)
    _run(qapp, dlg)
    b = dlg.browser
    b.edit.setChecked(True)
    tile = next(t for t, _w in b.tiles if t.entry["kind"] == "target")
    before = np.asarray(tile.entry["pts"], np.float64).copy()
    tile.poly_item.setPos(QtCore.QPointF(5, -3))
    tile._outline_moved(5, -3)
    assert np.allclose(np.asarray(tile.entry["pts"]) - before, (5, -3))
    g = CF.frame_for(tile.entry["fp"], fetch_missing=False)
    pts, _ref, _ = CF.raw_to_record(tile.entry["pts"], {}, g.raw_w, g.raw_h)
    _k, rec = existing_record(pt, "crown", tile.entry["fp"])
    assert np.abs(np.asarray(rec["points"]) - np.asarray(pts)).max() < 1e-9


def test_the_target_scan_does_not_block_the_dialog(geo_project, qapp):
    pt, imgs = geo_project
    fake = [f"C:/nowhere/frame_{i:05d}.JPG" for i in range(2000)]
    pt.multispectral_image_data_groups["R_big"] = fake
    pt.multispectral_root_names.append("R_big")
    try:
        dlg = pt.open_canopy_finder("crown", imgs[0])
        _wait_scan(qapp, dlg)
        t0 = time.perf_counter()
        dlg.rb_project.setChecked(True)
        dt = time.perf_counter() - t0
        assert dt < 0.3, f"switching to all roots blocked the UI for {dt:.2f} s"
        _wait_scan(qapp, dlg)
        assert len(_targets(dlg)) >= 2
        t0 = time.perf_counter()
        dlg.rb_root.setChecked(True); dlg.rb_project.setChecked(True)          # cached per scope
        assert time.perf_counter() - t0 < 0.3 and not dlg.scan_worker.isRunning()
    finally:
        pt.multispectral_image_data_groups.pop("R_big", None)
        pt.multispectral_root_names.remove("R_big")


def test_what_a_real_flir_frame_taught():
    """FLIR Duo Pro R (20221212_094602): FocalLength in IFD0, FocalPlaneXResolution
    10.88 px/mm for a 4000 px frame (a 368 mm 'sensor'), GPSTrack but no yaw, and
    GPSAltitude above sea level only."""
    from .. import canopy_finder_meta as M
    meta = {"lat": -2.93, "lon": -59.96, "alt_abs": 202.7, "focal": 8.0, "fp_xres": 10.88, "fp_unit": 4,
            "track": 322.66, "yaw": None, "alt_rel": None}
    assert M.sensor_width_mm(meta, 4000) is None, "an implausible sensor size is not used"
    assert M.exif_transform(meta, 4000, 3000)[0] is None, "no height above ground: needs the overlap"
    # with the frame's own measurement: its ground size and heading, not the GPS track
    tr, info = M.exif_transform(meta, 4000, 3000, calib={"gsd_m": 0.025, "yaw_deg": 95.0})
    assert tr is not None and not info["yaw_known"] and "overlap" in info["note"]
    assert abs(info["gsd_m"] - 0.025) < 1e-12
    g = CF.GeoRef(path="x", transform=tr, epsg=4326, crs_wkt=None, raw_w=4000, raw_h=3000, kind="exif")
    c, top = g.px_to_world([(2000, 1500), (2000, 0)])
    assert top[0] > c[0], "heading 95 deg: the top of the frame points (nearly) east"


def test_a_crown_on_no_data_is_refused(tmp_path):
    B = _world(300, seed=2)
    B[:, :150] = 0                                                     # mosaic edge: no data
    src = _write_geotiff(str(tmp_path / "edge.tif"), B, OX, OY, GSD)
    a = np.linspace(0, 2 * math.pi, 9)[:-1]
    P = np.column_stack((70 + 20 * np.cos(a), 150 + 20 * np.sin(a)))
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": 300, "h": 300}}
    with pytest.raises(ValueError, match="no-data"):
        CF.prepare_source(src, rec, {}, {})


# ───────────── round 3: GPS-only frames (no height / heading), several crowns, style ─────────────
def _flight_line(d, B, n=7, step_m=8.0, motion_deg=120.0, cam_yaw=30.0, jitter=1.0, seed=4):
    """Frames along a straight line (motion_deg) of a camera turned to cam_yaw,
    whose EXIF has ONLY a jittery GPS (like the FLIR Duo Pro R / MicaSense)."""
    from pyproj import Transformer
    from .. import canopy_finder_meta as M
    to_ll = Transformer.from_crs(32721, 4326, always_xy=True)
    rng = np.random.default_rng(seed)
    E0, N0 = OX + 50.0, OY - 60.0
    th = math.radians(motion_deg)
    out = []
    for k in range(n):
        E, N = E0 + k * step_m * math.sin(th), N0 + k * step_m * math.cos(th)
        p, g = _raw_frame(d, f"IMG_{k + 100:05d}", B, E, N, cam_yaw)
        lon, lat = to_ll.transform(E + rng.normal(0, jitter), N + rng.normal(0, jitter))
        M._put(p, {"lat": lat, "lon": lon, "alt_abs": 190.0, "alt_rel": None, "focal": None, "focal35": None,
                   "yaw": None, "track": None, "model": "TestCam", "full": True})
        out.append((p, g))
    return out


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("canopy_flight"))
    B = _world(1300, seed=21)
    return d, B, _flight_line(d, B)


def test_a_gps_only_flight_is_calibrated_from_its_own_overlap(flight):
    from .. import canopy_finder_meta as M
    d, B, frames = flight
    p, g_true = frames[3]
    c = CF.calibrate_frame(p)
    assert c is not None, M.get_calibration("frame|" + os.path.normcase(os.path.abspath(p)))
    true_gsd = AGL * 13.2 / (FOCAL * RW)
    assert abs(c["gsd_m"] / true_gsd - 1) < 0.06, (c["gsd_m"], true_gsd)
    assert abs((c["yaw_deg"] - 30.0 + 180) % 360 - 180) < 3.0, c["yaw_deg"]
    fr = CF.frame_for(p)
    assert fr.kind == "exif" and "overlap" in fr.note
    # frames of the measured stretch share it
    assert M.get_calibration("frame|" + os.path.normcase(os.path.abspath(frames[2][0]))) is not None


def test_an_orthomosaic_crown_is_found_in_gps_only_frames(flight, tmp_path):
    d, B, frames = flight
    ortho = _write_geotiff(str(tmp_path / "ortho.tif"), B, OX, OY, GSD)
    p, g_true = frames[3]
    c = g_true.world_to_px([g_true.px_to_world([(RW / 2, RH / 2)])[0]])[0]
    from pyproj import Transformer
    to_utm = Transformer.from_crs(4326, 32721, always_xy=True)
    lon, lat = g_true.px_to_world([(RW / 2 + 40, RH / 2 - 30)])[0]
    E, N = to_utm.transform(lon, lat)
    cx, cy = (E - OX) / GSD, (OY - N) / GSD
    a = np.linspace(0, 2 * math.pi, 13)[:-1]
    P = np.column_stack((cx + 35 * np.cos(a), cy + 30 * np.sin(a)))
    n = B.shape[0]
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    ctx = CF.prepare_source(ortho, rec, {}, {"method": "stackreg"})
    r = CF.find_in_target(ctx, p, {}, {"method": "stackreg"})
    truth = _truth_in(g_true, P)
    assert r.status == "found", (r.status, r.message)
    assert np.abs(np.asarray(r.raw_pts) - truth).max() < 3.0


def test_gps_written_as_zero_is_no_gps(tmp_path):
    from .. import canopy_finder_meta as M
    m = M._no_null_island({"lat": 0.0, "lon": 0.0})
    assert m["lat"] is None and m["lon"] is None
    p = str(tmp_path / "phenocam.jpg")
    cv2.imwrite(p, np.full((40, 60, 3), 128, np.uint8))
    M._put(p, {"lat": None, "lon": None, "full": True})
    assert M.positions([p]) == {}
    assert CF.frame_for(p, fetch_missing=False).kind == "pixel"


def test_several_crowns_in_one_run(geo_project, qapp, scene):
    from ..canopy_finder_dialog import existing_record
    pt, imgs = geo_project
    # a second crown on the source image
    a = np.linspace(0, 2 * math.pi, 13)[:-1]
    P2 = np.column_stack((330 + 30 * np.cos(a), 300 + 25 * np.sin(a)))
    n = scene.B.shape[0]
    pt.all_polygons["crown2"] = {imgs[0]: {"points": P2.tolist(), "coord_space": "image",
                                           "image_ref_size": {"w": n, "h": n}, "name": "crown2", "root": "1",
                                           "type": "polygon"}}
    pt._ensure_polygon_index() if hasattr(pt, "_ensure_polygon_index") else None
    try:
        pt._add_to_polygon_index("crown2", imgs[0])
    except Exception:
        pass
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(False)
    _wait_scan(qapp, dlg)
    names = [dlg.group_list.item(i).data(0x0100) for i in range(dlg.group_list.count())]
    assert set(names) >= {"crown", "crown2"} and names[0] == "crown", names
    for i in range(dlg.group_list.count()):
        if dlg.group_list.item(i).data(0x0100) == "crown2":
            dlg.group_list.item(i).setCheckState(2)
    assert set(dlg.selected_groups()) == {"crown", "crown2"}
    _run(qapp, dlg)
    by = {}
    for r, t, s in dlg.results:
        by.setdefault(t["group"], []).append((r.status, s))
    assert set(by) == {"crown", "crown2"} and all(st == "found" for g in by for st, _s in by[g]), by
    assert dlg.table.item(0, 0).text() in ("crown", "crown2")
    _k, rec2 = existing_record(pt, "crown2", imgs[1])
    assert rec2 is not None and np.abs(np.asarray(rec2["points"]) - scene.truth(P2)).max() < 1.0


def test_the_dialog_wears_the_image_editors_palette(geo_project, qapp):
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    assert "#FFD700" in dlg.styleSheet() and "#006400" in dlg.styleSheet()
    # CanoPie's dark green and its tints only -- not the yellow-green of round 3
    for off in ("#9acd32", "#eef7d6", "#d9f0a3", "#d6e9b0"):
        assert off not in dlg.styleSheet().lower(), off
    assert dlg.btn_find.objectName() == "primary"


# ───────────── round 4: fast EXIF, no typed priors, the aligned crown stack ─────────────
def _jpg_with_gps(path, lat, lon, track=123.4, model="TestCam", size=(300, 400)):
    from PIL import Image
    im = Image.fromarray(np.random.default_rng(abs(hash(path)) % 2 ** 32).integers(0, 255, size + (3,),
                                                                                       dtype=np.uint8))
    ex = Image.Exif()
    ex[0x0110] = model

    def dms(v):
        v = abs(v)
        d = int(v); m = int((v - d) * 60); s_ = (v - d - m / 60.0) * 3600
        return (float(d), float(m), round(s_, 4))
    ex[0x8825] = {1: "S" if lat < 0 else "N", 2: dms(lat), 3: "W" if lon < 0 else "E", 4: dms(lon),
                  6: 190.5, 0x0E: "T", 0x0F: float(track)}
    im.save(path, exif=ex, quality=85)
    return path


def test_the_header_reader_gives_what_the_whole_file_gives(tmp_path):
    import exifread
    from .. import canopy_finder_meta as M
    p = _jpg_with_gps(str(tmp_path / "a.jpg"), -2.9296, -59.9750, track=45.5)
    head = M._head_meta(p)
    with open(p, "rb") as f:
        whole = M._meta_from_tags(exifread.process_file(f, details=False), b"")
    for k in ("lat", "lon", "alt_abs", "track", "model"):
        assert head[k] == whole[k], (k, head[k], whole[k])
    assert abs(head["lat"] + 2.9296) < 1e-6 and abs(head["lon"] + 59.975) < 1e-6 and head["track"] == 45.5


def test_reading_gps_runs_in_threads_and_keeps_every_file(tmp_path, monkeypatch):
    from .. import canopy_finder_meta as M
    paths = [_jpg_with_gps(str(tmp_path / f"f_{i:05d}.jpg"), -2.93 - i * 1e-5, -59.97, size=(40, 60))
             for i in range(300)]
    monkeypatch.setattr(M, "_exiftool", lambda: None)
    t0 = time.perf_counter()
    pos = M.positions(paths)
    assert len(pos) == 300
    assert all(abs(pos[p][0] - (-2.93 - i * 1e-5)) < 1e-7 for i, p in enumerate(paths))
    assert time.perf_counter() - t0 < 20


def test_the_scan_lists_the_nearest_frames_without_measuring_them(flight, monkeypatch):
    from ..canopy_finder_dialog import classify_targets
    d, B, frames = flight
    ortho = _write_geotiff(os.path.join(d, "scan_ortho.tif"), B, OX, OY, GSD)
    n = B.shape[0]
    a = np.linspace(0, 2 * math.pi, 9)[:-1]
    P = np.column_stack((700 + 30 * np.cos(a), 600 + 30 * np.sin(a)))
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}

    def boom(*a, **k):
        raise AssertionError("the scan must not calibrate frames")
    monkeypatch.setattr(CF, "calibrate_frame", boom)
    monkeypatch.setattr(CF, "_borrowed_calibration", boom)
    targets = [{"fp": p, "root": "f", "project": None, "external": True} for p, _g in frames]
    t0 = time.perf_counter()
    items, summary, kind = classify_targets(ortho, rec, {}, targets, {"k": 3})
    assert time.perf_counter() - t0 < 5
    assert kind == "geotiff" and len(items) == 3 and all(ok for _t, _s, ok in items)
    assert "nearest #1" in items[0][1]


def test_the_removed_settings_are_gone(geo_project, qapp):
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    cfg = dlg.cfg()
    for k in ("default_agl_m", "default_sensor_mm", "scale_range_exif"):
        assert k not in cfg
    assert not hasattr(dlg, "agl") and not hasattr(dlg, "sensor") and not hasattr(dlg, "cb_flight")
    assert "flight" not in dlg._scan_opts()
    assert (cfg["scale_min_exif"], cfg["scale_max_exif"]) == (0.7, 1.4)
    assert dlg.method.currentData() == "stackreg"


@pytest.mark.parametrize("bias", [0.8, 1.25])
def test_a_biased_ground_size_is_still_found(flight, tmp_path, monkeypatch, bias):
    """A frame whose measured ground size is 20 % too small or 25 % too large
    (the bias the palm test showed breaking the old +-15 % scale search)."""
    from pyproj import Transformer
    d, B, frames = flight
    p, g_true = frames[3]
    true_gsd = AGL * 13.2 / (FOCAL * RW)
    monkeypatch.setattr(CF, "calibrate_frame",
                        lambda path, cancel=None, span=3: {"gsd_m": true_gsd * bias, "yaw_deg": 30.0})
    ortho = _write_geotiff(str(tmp_path / "ortho.tif"), B, OX, OY, GSD)
    to_utm = Transformer.from_crs(4326, 32721, always_xy=True)
    lon, lat = g_true.px_to_world([(RW / 2 + 40, RH / 2 - 30)])[0]
    E, N = to_utm.transform(lon, lat)
    cx, cy = (E - OX) / GSD, (OY - N) / GSD
    a = np.linspace(0, 2 * math.pi, 13)[:-1]
    P = np.column_stack((cx + 35 * np.cos(a), cy + 30 * np.sin(a)))
    n = B.shape[0]
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    ctx = CF.prepare_source(ortho, rec, {}, {})
    r = CF.find_in_target(ctx, p, {}, {})
    assert r.status == "found", (r.status, r.message)
    assert np.abs(np.asarray(r.raw_pts) - _truth_in(g_true, P)).max() < 3.0


@pytest.mark.parametrize("rotate", [0, 90])
def test_the_aligned_crown_stack(scene, tmp_path, rotate):
    """Every found target resampled onto the source crown window: the same pixels
    as the source on the crown (within interpolation), a stack with page 0 = the
    source square, and the source square's georeference -- also from a rotated
    source frame."""
    from ..shapefile_io import get_geotiff_transform
    n = scene.B.shape[0]
    ax = {"rotate": rotate} if rotate else {}
    e = np.asarray(CF.G.forward_np(CF.G.frame_plan(ax, n, n)[0], scene.P)) if rotate else scene.P
    rec = {"points": e.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    ctx = CF.prepare_source(scene.src, rec, ax, {"method": "stackreg"})
    r = CF.find_in_target(ctx, scene.tgt, {}, {"method": "stackreg"})
    assert r.status == "found"
    files = CF.export_aligned_stack(ctx, [r], str(tmp_path))
    stack = next(f for f in files if f.endswith("aligned_stack.tif"))
    one = next(f for f in files if f.endswith("_aligned.tif"))
    pages = tifffile.TiffFile(stack).pages
    assert len(pages) == 2
    src_sq = pages[0].asarray().astype(np.float32)
    al = tifffile.imread(one).astype(np.float32)
    assert al.shape == src_sq.shape
    # the target is 0.8 x + 20 of the source (other light): undo it and compare on the crown
    cm = ctx.cmask
    ok = cm & np.isfinite(al).all(axis=2)
    assert ok.sum() > 0.9 * cm.sum()
    diff = np.abs((al[ok] - 20.0) / 0.8 - src_sq[ok])
    assert np.median(diff) < 6.0, np.median(diff)
    t_stack = get_geotiff_transform(stack)[0] if not rotate else None
    t_src = get_geotiff_transform(CF.export_source_square(ctx, str(tmp_path / "sq")))[0] if not rotate else None
    if not rotate:
        assert np.allclose(t_stack, t_src)
    else:
        assert 34264 in pages[0].tags, "a rotated source frame: a ModelTransformation"
    rows = open(os.path.join(str(tmp_path), "aligned_stack.csv"), encoding="utf-8").read().splitlines()
    assert rows[1].startswith("0,") and "source" in rows[1] and "found" in rows[2]


# ───────────── round 5: a reusable EXIF cache, several folders, searches in parallel ─────────────
def test_the_exif_cache_is_kept_per_folder_and_reused(tmp_path, monkeypatch):
    """A folder read once is not read again -- next session, from any project --
    and nothing is written beside the images."""
    from .. import canopy_finder_meta as M
    d = tmp_path / "flight_a"
    d.mkdir()
    paths = [_jpg_with_gps(str(d / f"g_{i:03d}.jpg"), -2.93 - i * 1e-5, -59.97, size=(40, 60)) for i in range(20)]
    monkeypatch.setattr(M, "_exiftool", lambda: None)
    pos1 = M.positions(paths)
    assert len(pos1) == 20
    M.save()
    assert any(f.endswith("_flight_a.json") for f in os.listdir(M.shared_cache_dir()))
    assert sorted(os.listdir(d)) == sorted(os.path.basename(p) for p in paths)
    # a new session: nothing in memory, and reading a file is an error
    nd = os.path.normcase(os.path.abspath(str(d)))
    for k in [k for k in M._MEM if os.path.dirname(k) == nd]:
        del M._MEM[k]
    M._LOADED.discard(nd)

    def boom(*a, **k):
        raise AssertionError("the folder was read again")
    monkeypatch.setattr(M, "_head_meta", boom)
    assert M.positions(paths) == pos1
    # a changed file is read again (its stamp no longer matches)
    monkeypatch.undo()
    monkeypatch.setattr(M, "_exiftool", lambda: None)
    time.sleep(0.05)
    _jpg_with_gps(paths[0], -3.0, -60.0, size=(40, 60))
    M._DIRSTAT.clear()
    assert abs(M.positions(paths[:1])[paths[0]][0] + 3.0) < 1e-6


def test_each_folder_gets_its_own_nearest_frames(flight, tmp_path):
    """Two flights over the same crown: k nearest from EACH folder, not k overall."""
    import shutil
    from .. import canopy_finder_meta as M
    from ..canopy_finder_dialog import classify_targets
    d, B, frames = flight
    d2 = tmp_path / "flight_b"
    d2.mkdir()
    targets = []
    for p, _g in frames:
        q = str(d2 / os.path.basename(p))
        shutil.copyfile(p, q)
        m = dict(M.get_meta(p, full=False))
        m.pop("stamp", None)
        m["lat"] += 2e-6                       # the second flight a little off
        M._put(q, m)
        targets += [{"fp": p, "root": "a", "project": None, "external": True},
                    {"fp": q, "root": "b", "project": None, "external": True}]
    ortho = _write_geotiff(str(tmp_path / "ortho.tif"), B, OX, OY, GSD)
    n = B.shape[0]
    a = np.linspace(0, 2 * math.pi, 9)[:-1]
    P = np.column_stack((700 + 30 * np.cos(a), 600 + 30 * np.sin(a)))
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    items, summary, kind = classify_targets(ortho, rec, {}, targets, {"k": 2})
    dirs = [os.path.normcase(os.path.dirname(t["fp"])) for t, _s, ok in items if ok]
    assert len(dirs) == 4 and dirs.count(os.path.normcase(str(d2))) == 2, dirs
    assert any("[flight_b]" in st for _t, st, _ok in items)
    assert "2 folders" in summary, summary
    # the same frames, nearest first, in each flight
    names = lambda folder: [os.path.basename(t["fp"]) for t, _s, _ok in items
                            if os.path.normcase(os.path.dirname(t["fp"])) == folder]
    assert names(os.path.normcase(str(d2))) == names(os.path.normcase(os.path.abspath(d)))


def test_several_folders_searched_in_parallel_give_the_sequential_result(geo_project, qapp, scene, tmp_path):
    import shutil
    from ..canopy_finder_dialog import image_folders_under, search_workers
    pt, imgs = geo_project
    parent = tmp_path / "flights"
    for f in ("day1", "day2"):
        (parent / f).mkdir(parents=True)
        for k in range(2):
            shutil.copyfile(scene.tgt, parent / f / f"{f}_{k}.tif")
    (parent / "notes").mkdir()
    (parent / "notes" / "readme.txt").write_text("no images")
    folders = image_folders_under(str(parent))
    assert [os.path.basename(f) for f in folders] == ["day1", "day2"]
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(False)
    assert dlg.workers.value() == search_workers() and 1 <= search_workers() <= 8
    dlg.add_folders(folders)
    _wait_scan(qapp, dlg)
    assert len(_targets(dlg)) == 4
    runs = {}
    old_cv = cv2.getNumThreads()
    for w in (1, 4):
        dlg.workers.setValue(w)
        assert dlg.cfg()["workers"] == w
        _run(qapp, dlg)
        assert cv2.getNumThreads() == old_cv            # OpenCV's own pool is given back
        runs[w] = {os.path.normcase(t["fp"]): (r.status, s, np.asarray(r.raw_pts))
                   for r, t, s in dlg.results}
    assert len(runs[1]) == 4 and set(runs[1]) == set(runs[4])
    for fp, (st, sv, pts) in runs[1].items():
        assert st == "found" and sv == "external", (fp, st, sv)
        assert runs[4][fp][0] == st and np.abs(runs[4][fp][2] - pts).max() < 1e-6
    for f in folders:                                   # read only
        assert not any(x.endswith("_polygons.json") for x in os.listdir(f))
    dlg.ext_list.item(0).setSelected(True)
    dlg._remove_folders()
    assert dlg._ext_paths == folders[1:]
    _wait_scan(qapp, dlg)
    assert len(_targets(dlg)) == 2


def _forget_calibrations(folder):
    from .. import canopy_finder_meta as M
    nd = os.path.normcase(os.path.abspath(folder))
    with M._LOCK:
        for k in [k for k in M._CALIB if nd in k]:
            del M._CALIB[k]
        M._DIRTY_DIRS.add(nd)
    M.save()                    # the worker processes read the cache file: forget there too


def test_a_frozen_build_searches_in_threads(monkeypatch):
    import sys
    assert CF.processes_available()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not CF.processes_available()
    with pytest.raises(RuntimeError):
        CF.ProcessSearch(2)


def test_worker_processes_load_only_the_finder_at_low_priority():
    import sys
    from .. import canopy_finder_meta as M
    main = sys.modules["__main__"].__dict__
    before = (main.get("__file__"), main.get("__spec__", "absent"))
    pool = CF.ProcessSearch(2)
    try:
        info = pool.submit(CF._process_probe).result(timeout=180)
    finally:
        pool.close()
    assert (main.get("__file__"), main.get("__spec__", "absent")) == before
    assert info["pid"] != os.getpid()
    assert not info["qt"] and not info["app"], info           # not the app, not Qt
    assert info["cv_threads"] == 1
    assert os.path.normcase(info["cache"]) == os.path.normcase(M.shared_cache_dir())
    if os.name == "nt":
        assert info["priority"] == 0x4000                     # BELOW_NORMAL_PRIORITY_CLASS
        # a search thread here too (and the shared ctypes functions are left alone)
        import ctypes
        import threading
        seen = []

        def th():
            seen.append(CF.lower_priority())
            k = CF._kernel32()
            seen.append(k.GetThreadPriority(k.GetCurrentThread()))
        t = threading.Thread(target=th)
        t.start()
        t.join()
        assert seen == [True, -1]
        assert ctypes.windll.kernel32.GetCurrentProcess.restype is ctypes.c_int


def test_a_flight_measured_in_the_processes_is_measured_as_here(flight):
    d, B, frames = flight
    p = frames[3][0]
    _forget_calibrations(d)
    here = CF.calibrate_frame(p)
    pool = CF.ProcessSearch(3)
    try:
        _forget_calibrations(d)
        with CF.pair_shifts_with(pool.pair_shifts):
            there = CF.calibrate_frame(p)
        # a pool that stopped: its pairs are registered here, not cached as a failure
        pool.close()
        _forget_calibrations(d)
        with CF.pair_shifts_with(pool.pair_shifts):
            after = CF.calibrate_frame(p)
    finally:
        pool.close()
    assert here and there and after
    for k in ("gsd_m", "yaw_deg"):
        assert abs(here[k] - there[k]) < 1e-9 and abs(here[k] - after[k]) < 1e-9, (k, here, there, after)


def test_gps_only_frames_give_the_same_crowns_with_one_or_four_workers(flight, tmp_path, qapp):
    """Measured in a fixed order (nearest first, per folder) and searched in
    worker processes: the same answer as one search after another."""
    from PyQt5 import QtCore
    from pyproj import Transformer
    from ..canopy_finder_dialog import CanopyFindWorker
    d, B, frames = flight
    ortho = _write_geotiff(str(tmp_path / "ortho.tif"), B, OX, OY, GSD)
    g3 = frames[3][1]
    to_utm = Transformer.from_crs(4326, 32721, always_xy=True)
    lon, lat = g3.px_to_world([(RW / 2 + 40, RH / 2 - 30)])[0]
    E, N = to_utm.transform(lon, lat)
    cx, cy = (E - OX) / GSD, (OY - N) / GSD
    a = np.linspace(0, 2 * math.pi, 13)[:-1]
    P = np.column_stack((cx + 35 * np.cos(a), cy + 30 * np.sin(a)))
    n = B.shape[0]
    rec = {"points": P.tolist(), "coord_space": "image", "image_ref_size": {"w": n, "h": n}}
    targets = [{"fp": p, "root": "f", "project": None, "external": True, "ax": {}} for p, _g in frames[1:6]]
    runs = {}
    for w in (1, 4):
        _forget_calibrations(d)
        got = []
        wk = CanopyFindWorker(ortho, [{"group": "c", "rec": rec, "targets": targets}], {}, {"workers": w})
        wk.result.connect(lambda r, t: got.append((t["fp"], r)), QtCore.Qt.DirectConnection)
        wk.run()
        assert wk.used_processes == (4 if w == 4 else 0)
        runs[w] = dict(got)
    assert set(runs[1]) == set(runs[4]) and len(runs[1]) == 5
    assert sum(r.status == "found" for r in runs[1].values()) >= 3
    for fp, r in runs[1].items():
        q = runs[4][fp]
        assert q.status == r.status, (fp, r.status, q.status)
        if r.raw_pts is not None:
            assert np.abs(np.asarray(q.raw_pts) - np.asarray(r.raw_pts)).max() < 1e-3   # float order
    p3 = frames[3][0]
    assert np.abs(np.asarray(runs[4][p3].raw_pts) - _truth_in(g3, P)).max() < 3.0


def test_a_plain_tiff_is_not_opened_again_to_look_for_a_georeference(scene, tmp_path, monkeypatch):
    import shutil
    from ..canopy_finder_dialog import classify_targets
    d = tmp_path / "ir"
    d.mkdir()
    for k in range(4):
        tifffile.imwrite(str(d / f"IR_{k}.tiff"), np.full((20, 30), 7 + k, np.uint16))
    shutil.copyfile(scene.tgt, d / "geo.tif")
    targets = [{"fp": str(p), "root": "ir", "project": None, "external": True} for p in sorted(d.iterdir())]
    items1, _s, _k = classify_targets(scene.src, scene.record(), {}, targets, {"k": 5})
    opened = []
    real = CF.georef
    monkeypatch.setattr(CF, "georef", lambda p: opened.append(os.path.basename(p)) or real(p))
    items2, _s, _k = classify_targets(scene.src, scene.record(), {}, targets, {"k": 5})
    assert "geo.tif" in opened and not [o for o in opened if o.startswith("IR_")], opened
    assert [(t["fp"], st, ok) for t, st, ok in items1] == [(t["fp"], st, ok) for t, st, ok in items2]


# ───────────── round 6: replace in the viewer, folding sections, rotated browser tiles ─────────────
def _drawn_outlines(v, group):
    out = []
    for p in v.polygons:
        it = p.get("item")
        if getattr(it, "name", None) == group and it.scene() is v.scene():
            poly = it.polygon
            out.append(np.array([[it.mapToScene(poly[i]).x(), it.mapToScene(poly[i]).y()]
                                 for i in range(poly.count())]))
    return out


def _scene_pts(v, rec):
    from ..canopy_finder_dialog import record_scene_polygon
    q = record_scene_polygon(v, rec["points"], rec["image_ref_size"])
    return np.array([[p.x(), p.y()] for p in q])


def test_a_replaced_crown_replaces_the_outline_in_the_viewer(geo_project, qapp, scene):
    """A crown already drawn in the viewer: the found one replaces it there too
    (load_polygons skips a group the viewer already shows -- the old outline
    stayed on screen)."""
    from ..canopy_finder_dialog import existing_record, live_viewer_for
    pt, imgs = geo_project
    n = scene.B.shape[0]
    wrong = [[20, 20], [80, 20], [80, 80], [20, 80]]
    for fp in imgs[1:]:
        pt.all_polygons["crown"][fp] = {"points": wrong, "coord_space": "image", "image_ref_size": {"w": n, "h": n},
                                        "name": "crown", "root": "1", "type": "polygon"}
        pt._add_to_polygon_index("crown", fp)
        pt._mark_polygon_dirty("crown", fp)
    pt._flush_dirty_polygons()
    for fp in imgs[1:]:
        v = live_viewer_for(pt, fp)
        pt._schedule_polygons_for_viewer(v, v.image_data, delay_ms=0)
    _pump(qapp, 1.5)
    assert all(len(_drawn_outlines(live_viewer_for(pt, fp), "crown")) == 1 for fp in imgs[1:])
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(False)
    dlg.cb_replace.setChecked(True)
    _run(qapp, dlg)
    assert [s_ for _r, _t, s_ in dlg.results] == ["replaced", "replaced"]
    _pump(qapp, 1.5)
    truth = scene.truth()
    for fp in imgs[1:]:
        _k, rec = existing_record(pt, "crown", fp)
        assert np.abs(np.asarray(rec["points"]) - truth).max() < 1.0
        v = live_viewer_for(pt, fp)
        got = _drawn_outlines(v, "crown")
        want = _scene_pts(v, rec)
        assert len(got) == 1 and got[0].shape == want.shape and np.abs(got[0] - want).max() < 1.0
    # the record survives the viewer's next save pass (no "deleted by the user" purge)
    pt.update_all_polygons() if hasattr(pt, "update_all_polygons") else None
    assert all(existing_record(pt, "crown", fp)[1] is not None for fp in imgs[1:])


def test_the_dialog_sections_fold(geo_project, qapp):
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.show()
    secs = (dlg.sec_crowns, dlg.sec_targets, dlg.sec_reg, dlg.sec_found)
    try:
        for sec in secs:
            assert sec.is_open() and sec.body.isVisible() and sec.title().startswith("▾")
        h_open = dlg.sec_reg.sizeHint().height()
        dlg.sec_reg.set_open(False)
        qapp.processEvents()
        assert not dlg.sec_reg.body.isVisible() and dlg.sec_reg.title().startswith("▸")
        assert dlg.sec_reg.sizeHint().height() < h_open / 3
        assert dlg.cfg()["min_ncc"] == dlg.min_score.value()          # folded settings still apply
        dlg.sec_reg.set_open(True)
        qapp.processEvents()
        assert dlg.sec_reg.body.isVisible() and dlg.method.isEnabled()
        # a control disabled by the method stays disabled after unfolding
        dlg.method.setCurrentIndex(dlg.method.findData("geo"))
        assert not dlg.radius.isEnabled()
        dlg.sec_reg.set_open(False); dlg.sec_reg.set_open(True)
        assert not dlg.radius.isEnabled() and dlg.method.isEnabled()
        dlg.method.setCurrentIndex(dlg.method.findData("stackreg"))
        assert dlg.radius.isEnabled()
        # remembered
        from ..canopy_finder_dialog import Section
        dlg.sec_targets.set_open(False)
        assert not Section("Search in").is_open()
    finally:
        for sec in secs:
            sec.set_open(True)


def _drag(view, p0, p1, steps=6):
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtTest import QTest
    vp = view.viewport()
    QTest.mousePress(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, p0)
    for k in range(1, steps + 1):
        p = QtCore.QPoint(int(round(p0.x() + (p1.x() - p0.x()) * k / steps)),
                          int(round(p0.y() + (p1.y() - p0.y()) * k / steps)))
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(p), QtCore.QPointF(vp.mapToGlobal(p)),
                               QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(vp, ev)
    QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, p1)


@pytest.mark.parametrize("deg", [90, 35])
def test_a_crown_moved_in_a_rotated_browser_tile_lands_on_the_image(geo_project, qapp, deg):
    """The tile view turns, the image does not: a drag to the RIGHT on screen in
    a tile turned 90 deg moves the crown along the image's y axis, and the record
    (and the grid viewer) get exactly that move."""
    from PyQt5 import QtCore
    from ..canopy_finder_dialog import existing_record, live_viewer_for
    pt, imgs = geo_project
    dlg = pt.open_canopy_finder("crown", imgs[0])
    dlg.cb_browser.setChecked(True)
    _run(qapp, dlg)
    b = dlg.browser
    b.cols.setValue(1); b.rows.setValue(1)
    b.set_page(1)
    tile = b.tiles[0][0]
    fp = tile.entry["fp"]
    with open(fp, "rb") as f:
        file_before = f.read()
    _pump(qapp, 0.3)                       # the tile's first fit
    s0 = tile.view_scale()
    tile.set_view_rotation(deg)
    _pump(qapp, 0.2)
    assert abs(tile.view_scale() - s0) < 1e-6 * s0 and abs(tile.view_rotation() - deg) < 1e-9
    before = np.asarray(tile.entry["pts"], np.float64).copy()
    c = tile.poly_item.mapToScene(tile.poly_item.boundingRect().center())
    p0 = tile.mapFromScene(c)
    p1 = p0 + QtCore.QPoint(40, 0)
    want = tile.mapToScene(p1) - tile.mapToScene(p0)                 # the drag in image pixels
    _drag(tile, p0, p1)
    _pump(qapp, 0.3)
    moved = np.asarray(tile.entry["pts"], np.float64) - before
    assert np.allclose(moved, moved[0]), "the crown must move as a whole"
    assert np.abs(moved[0] - (want.x(), want.y())).max() < 1.0, (moved[0], want)
    if deg == 90:
        assert abs(moved[0][1]) > 5 * abs(moved[0][0])              # screen-right = image y
    g = CF.frame_for(fp, fetch_missing=False)
    pts, _ref, _ = CF.raw_to_record(tile.entry["pts"], {}, g.raw_w, g.raw_h)
    _k, rec = existing_record(pt, "crown", fp)
    assert np.abs(np.asarray(rec["points"]) - np.asarray(pts)).max() < 1e-9
    _pump(qapp, 1.0)
    v = live_viewer_for(pt, fp)
    got = _drawn_outlines(v, "crown")
    assert len(got) == 1 and np.abs(got[0] - _scene_pts(v, rec)).max() < 1.0
    with open(fp, "rb") as f:
        assert f.read() == file_before                               # the image is never rotated
    # rotation survives paging; sync zoom compares true scales
    b.set_page(0); b.set_page(1)
    assert abs(b.tiles[0][0].view_rotation() - deg) < 1e-9
    b.cols.setValue(3)
    t0, t1 = b.tiles[0][0], b.tiles[1][0]
    t1.set_view_rotation(deg)
    t0.scale(2.0, 2.0)
    b._on_view_changed(t0)
    assert abs(t0.ground_state()[0] - t1.ground_state()[0]) < 1e-6


@pytest.mark.parametrize("method", ["stackreg", "exhaustive"])
def test_a_mosaic_with_no_data_under_the_crown_does_not_hold_it(scene, tmp_path, method):
    """pasta (10): two mosaics hold no data where the crown projects. StackReg
    reported score -1, the exhaustive search 'found' a patch 25 m away."""
    from ..shapefile_io import get_geotiff_transform
    arr = tifffile.imread(scene.tgt)
    T = scene.truth()
    x0, y0 = np.floor(T.min(axis=0)).astype(int) - 25
    x1, y1 = np.ceil(T.max(axis=0)).astype(int) + 25
    arr = arr.copy()
    arr[max(0, y0):y1, max(0, x0):x1] = 0
    tr, _wkt = get_geotiff_transform(scene.tgt)
    p = _write_geotiff(str(tmp_path / "gap.tif"), arr, tr[0], tr[3], tr[1], -tr[5])
    ctx = CF.prepare_source(scene.src, scene.record(), {}, {"method": method})
    r = CF.find_in_target(ctx, p, {}, {"method": method})
    assert r.status == "outside_footprint" and "no data under the crown" in r.message, (r.status, r.score, r.message)
    ok = CF.find_in_target(ctx, scene.tgt, {}, {"method": method})
    assert ok.status == "found"
