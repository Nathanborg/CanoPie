"""
Polygons and points stay on their trees when the crop / rotation / resize of
an image changes -- in memory, on disk, and through the editor's Reset -- and
the machinery that makes that affordable and visible stays honest.

THE BUGS THIS PINS (Nathan, 2026-10-06, C:\\Nova pasta (5) and C:\\New Folder223):

  * polygons are stored relative to the post-.ax frame, so a new crop slid
    every crown to the same RELATIVE position over the wrong trees (points
    were fixed the round before; polygons were not);
  * "polygons disappear on resetting the image editor": the Reset refresh kept
    the old scene coordinates on a 16x larger image (a black corner), and the
    first fix attempt's re-projection could be undone by update_all_polygons
    reading the old-frame items back;
  * the image editor took 60 s to open on the BCI project -- 55 s of it the
    mask-polygon menu paging in all 2,802 crowns just to list their names --
    and re-decoded the 2.9 GB mosaic on every open;
  * zoomed in, the sharp tile showed SOURCE pixels even when the export is
    downsampled (566 x 511 export of a 28,315 px mosaic: 2,500 source pixels
    under a one-pixel point marker -- "bigger drawing than the pixels I have");
  * a drag in point mode appended 23 unsnapped, partly duplicated points.

Every case runs over the image types the app reads (PNG, JPEG, strip TIFF,
tiled float TIFF, a COG with overviews, a band-first planar stack).
"""
import json
import os
import zipfile

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.extraction, pytest.mark.viewer]

RAW_W, RAW_H = 70, 53


# --------------------------------------------------------------------------
# rasters of every type, written per test
# --------------------------------------------------------------------------
def _write_raster(kind, path_base):
    import cv2
    import tifffile
    rng = np.random.default_rng(11)
    if kind == "png_u8":
        p = path_base + ".png"
        cv2.imwrite(p, rng.integers(0, 255, (RAW_H, RAW_W, 3), dtype=np.uint8))
    elif kind == "jpeg_u8":
        p = path_base + ".jpg"
        # smooth content so JPEG decodes stably
        y, x = np.mgrid[0:RAW_H, 0:RAW_W]
        img = np.dstack([(x * 3) % 256, (y * 4) % 256, ((x + y) * 2) % 256]).astype(np.uint8)
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif kind == "tif_strip_u16":
        p = path_base + ".tif"
        tifffile.imwrite(p, rng.integers(1, 60000, (RAW_H, RAW_W, 3), dtype=np.uint16), photometric="rgb")
    elif kind == "tif_tiled_f32_6band":
        p = path_base + ".tif"
        tifffile.imwrite(p, rng.uniform(0, 1, (RAW_H, RAW_W, 6)).astype(np.float32),
                         tile=(16, 16), photometric="minisblack", planarconfig="contig")
    elif kind == "cog_u8_overviews":
        p = path_base + ".tif"
        arr = rng.integers(0, 255, (RAW_H, RAW_W, 3), dtype=np.uint8)
        with tifffile.TiffWriter(p) as tw:
            tw.write(arr, tile=(16, 16), photometric="rgb", subifds=1)
            tw.write(arr[::2, ::2], tile=(16, 16), photometric="rgb", subfiletype=1)
    elif kind == "tif_planar_u16_5band":
        p = path_base + ".tif"
        tifffile.imwrite(p, rng.integers(1, 60000, (5, RAW_H, RAW_W), dtype=np.uint16),
                         photometric="minisblack", planarconfig="separate")
    else:
        raise ValueError(kind)
    return p


KINDS = ["png_u8", "jpeg_u8", "tif_strip_u16", "tif_tiled_f32_6band", "cog_u8_overviews",
         "tif_planar_u16_5band"]

GEOMS = {
    "crop": {"crop_rect": {"x": 11, "y": 6, "width": 40, "height": 33},
             "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}},
    "rot90+crop": {"rotate": 90, "crop_rect": {"x": 4, "y": 9, "width": 40, "height": 50},
                   "crop_rect_ref_size": {"w": RAW_H, "h": RAW_W}},
    "rot270": {"rotate": 270},
    "crop+resize60": {"crop_rect": {"x": 11, "y": 6, "width": 40, "height": 33},
                      "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}, "resize": {"scale": 60}},
}


@pytest.fixture
def iso(synthetic_project, tmp_path, monkeypatch):
    """The shared ProjectTab, writing its polygons into a private folder."""
    pt = synthetic_project
    folder = tmp_path / "proj"
    (folder / "polygons").mkdir(parents=True)
    monkeypatch.setattr(pt, "project_folder", str(folder), raising=False)
    monkeypatch.setattr(pt, "_mark_polygon_dirty", lambda g, f: None)
    added = []
    yield pt, folder, added
    for g in added:
        pt.all_polygons.pop(g, None)
    for attr in ("_export_image_cache", "_ax_cache", "_ax_json_cache", "_export_frame_hw_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()


def _write_ax(pt, fp, ax):
    axp = pt._ax_path_for(fp)
    if ax is None:
        if os.path.exists(axp):
            os.remove(axp)
    else:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(ax, f)
    for attr in ("_export_image_cache", "_ax_cache", "_ax_json_cache", "_export_frame_hw_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    if hasattr(pt, "_invalidate_cached_ax"):
        pt._invalidate_cached_ax(fp)


def _to_raw(pt, fp, rec):
    from .. import ax_geometry as G
    ops, (W, H) = G.frame_plan(pt._ax_geometry_subset(fp), RAW_W, RAW_H)
    ref = rec["image_ref_size"]
    a = np.asarray(rec["points"], float) * (W / ref["w"], H / ref["h"])
    return G.inverse_np(ops, a)


def _csv_values(pt, group, fp, rec):
    rows, _ = pt.process_polygon(group, fp, rec, {}, [], False, opts={"stats": {"mean": True}})
    rows = [r for r in rows if isinstance(r, dict)]
    return {r["Channel"]: r["Mean"] for r in rows}, rows


@pytest.mark.parametrize("geom", list(GEOMS), ids=list(GEOMS))
@pytest.mark.parametrize("kind", KINDS)
def test_polygons_and_points_stay_on_their_raw_pixels(iso, tmp_path, kind, geom):
    pt, folder, added = iso
    fp = _write_raster(kind, str(tmp_path / f"img_{kind}"))
    _write_ax(pt, fp, None)
    rx, ry = 37, 29
    square = [(rx - 3.0, ry - 2.0), (rx + 4.0, ry - 2.0), (rx + 4.0, ry + 3.0), (rx - 3.0, ry + 3.0)]
    ref = {"w": RAW_W, "h": RAW_H}
    pt.all_polygons["follow_pt"] = {fp: {"name": "p", "type": "point", "coord_space": "image", "root": "",
                                         "image_ref_size": dict(ref), "points": [(rx + 0.5, ry + 0.5)]}}
    pt.all_polygons["follow_poly"] = {fp: {"name": "q", "type": "polygon", "coord_space": "image", "root": "",
                                           "image_ref_size": dict(ref), "points": list(square)}}
    added += ["follow_pt", "follow_poly"]
    from .. import polygon_lod
    base = os.path.splitext(os.path.basename(fp))[0]
    for g in ("follow_pt", "follow_poly"):           # a saved project: files on disk
        polygon_lod.dump_sidecar(str(folder / "polygons" / f"{g}_{base}_polygons.json"),
                                 pt._sidecar_payload(g, fp, pt.all_polygons[g][fp]))
    try:
        truth, _ = _csv_values(pt, "follow_pt", fp, pt.all_polygons["follow_pt"][fp])
        assert truth, f"{kind}: no CSV value for the point before any change"
        snap = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, GEOMS[geom])
        assert pt._reproject_after_ax_change(snap) == 2
        # geometry: back on the same raw pixels, exactly
        assert np.allclose(_to_raw(pt, fp, pt.all_polygons["follow_poly"][fp]), square, atol=1e-6), (
            f"{kind}/{geom}: the polygon slid off its raw pixels")
        assert np.allclose(_to_raw(pt, fp, pt.all_polygons["follow_pt"][fp]), [(rx + 0.5, ry + 0.5)], atol=1e-6)
        # values: the point reads its original pixel (a resize averages, so only without one)
        if "resize" not in geom:
            got, _ = _csv_values(pt, "follow_pt", fp, pt.all_polygons["follow_pt"][fp])
            assert got == pytest.approx(truth), (
                f"{kind}/{geom}: after the change the point reads {got}, its pixel read {truth}")
        # on disk: the files say the same, and a backup of the old ones exists
        base = os.path.splitext(os.path.basename(fp))[0]
        side = json.load(open(folder / "polygons" / f"follow_poly_{base}_polygons.json", encoding="utf-8"))
        assert np.allclose(side["points"], pt.all_polygons["follow_poly"][fp]["points"])
        zips = list((folder / "polygons" / ".reproject_backup").glob(f"*_{base}.zip"))
        assert len(zips) == 1
        # and back: undoing the change restores the original numbers
        snap2 = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, None)
        assert pt._reproject_after_ax_change(snap2) == 2
        assert np.allclose(pt.all_polygons["follow_poly"][fp]["points"], square, atol=1e-6)
    finally:
        _write_ax(pt, fp, None)


@pytest.mark.parametrize("kind", ["png_u8", "cog_u8_overviews"])
def test_a_resize_alone_moves_nothing_and_writes_nothing(iso, tmp_path, kind):
    """Records hold fractions of the frame: a uniform resize leaves them right,
    so there must be no rewrite and no backup (the BCI crowns cost ~20 s)."""
    pt, folder, added = iso
    fp = _write_raster(kind, str(tmp_path / f"rs_{kind}"))
    _write_ax(pt, fp, {"crop_rect": {"x": 5, "y": 4, "width": 50, "height": 40},
                       "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}})
    pts = [(10.25, 7.5), (30.0, 7.5), (30.0, 22.0)]
    pt.all_polygons["rs_only"] = {fp: {"name": "r", "type": "polygon", "coord_space": "image", "root": "",
                                       "image_ref_size": {"w": RAW_W, "h": RAW_H}, "points": list(pts)}}
    added.append("rs_only")
    try:
        snap = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, {"crop_rect": {"x": 5, "y": 4, "width": 50, "height": 40},
                           "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}, "resize": {"px_w": 17}})
        assert pt._reproject_after_ax_change(snap) == 0
        assert pt.all_polygons["rs_only"][fp]["points"] == pts
        assert not (folder / "polygons" / ".reproject_backup").exists()
    finally:
        _write_ax(pt, fp, None)


@pytest.mark.skipif(os.name != "nt", reason="two spellings of one path (forward and back slashes) "
                                            "exist only on Windows")
def test_two_spellings_of_one_image_are_moved_once(iso, tmp_path):
    """Records of one image keyed 'C:/x/a.tif' and 'C:\\x\\a.tif' (the BCI
    project has both): a snapshot keyed by spelling moved every crown twice --
    18,036 px off after one crop."""
    pt, folder, added = iso
    fp = _write_raster("tif_strip_u16", str(tmp_path / "spell"))
    alt = fp.replace("\\", "/") if "\\" in fp else fp.replace("/", "\\")
    assert os.path.normcase(os.path.normpath(alt)) == os.path.normcase(os.path.normpath(fp)) and alt != fp
    _write_ax(pt, fp, None)
    ref = {"w": RAW_W, "h": RAW_H}
    pt.all_polygons["sp_a"] = {fp: {"name": "a", "type": "polygon", "coord_space": "image", "root": "",
                                    "image_ref_size": dict(ref), "points": [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)]}}
    pt.all_polygons["sp_b"] = {alt: {"name": "b", "type": "polygon", "coord_space": "image", "root": "",
                                     "image_ref_size": dict(ref), "points": [(21.0, 21.0), (31.0, 21.0), (31.0, 31.0)]}}
    added += ["sp_a", "sp_b"]
    try:
        snap = pt._snapshot_ax_geometry()
        mine = [k for k in snap if os.path.normcase(os.path.normpath(k)) == os.path.normcase(os.path.normpath(fp))]
        assert len(mine) == 1, f"one image, {len(mine)} snapshot entries: {mine}"
        _write_ax(pt, fp, GEOMS["crop"])
        pt._reproject_after_ax_change(snap)
        for g, key, orig in (("sp_a", fp, (20.0, 20.0)), ("sp_b", alt, (21.0, 21.0))):
            assert np.allclose(_to_raw(pt, fp, pt.all_polygons[g][key])[0], orig, atol=1e-6), (
                f"{g} moved {_to_raw(pt, fp, pt.all_polygons[g][key])[0]} -- re-projected more than once")
    finally:
        _write_ax(pt, fp, None)


def test_lazy_crowns_are_reprojected_file_to_file_and_stay_lazy(iso, tmp_path):
    from .. import polygon_lod
    pt, folder, added = iso
    fp = _write_raster("cog_u8_overviews", str(tmp_path / "lazy"))
    _write_ax(pt, fp, None)
    base = os.path.splitext(os.path.basename(fp))[0]
    pdir = folder / "polygons"
    pts = [(12.0, 9.0), (25.5, 9.0), (25.5, 19.25), (12.0, 19.25)]
    for i in range(3):
        g = f"lz_{i}"
        data = {"points": [(x + i, y) for x, y in pts], "name": g, "type": "polygon", "root": "",
                "coord_space": "image", "image_ref_size": {"w": RAW_W, "h": RAW_H}, "properties": {"k": i}}
        src = str(pdir / f"{g}_{base}_polygons.json")
        polygon_lod.dump_sidecar(src, data)
        pt.all_polygons[g] = {fp: polygon_lod.LazyPolygonRecord(
            {"name": g, "group": g, "type": "polygon", "coord_space": "image", "root": "",
             "image_ref_size": {"w": RAW_W, "h": RAW_H}, "display_points": [list(p) for p in data["points"][:3]],
             "full_point_count": 4}, src)}
        added.append(g)
    originals = {f"lz_{i}": open(pdir / f"lz_{i}_{base}_polygons.json", "rb").read() for i in range(3)}
    try:
        snap = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, GEOMS["rot90+crop"])
        assert pt._reproject_after_ax_change(snap) == 3
        for i in range(3):
            rec = pt.all_polygons[f"lz_{i}"][fp]
            assert not rec.is_materialised, "re-projecting paged the crown in (20 s / ~200 MB on the BCI project)"
            disp = np.asarray(dict.get(rec, "display_points"), float)
            assert np.allclose(_to_raw(pt, fp, {"points": disp, "image_ref_size": {"w": RAW_W, "h": RAW_H}}),
                               [(x + i, y) for x, y in pts[:3]], atol=1e-6), "coarse display outline not moved"
            assert np.allclose(_to_raw(pt, fp, rec), [(x + i, y) for x, y in pts], atol=1e-6)  # pages in the new file
            assert rec["properties"] == {"k": i}
        z = next((pdir / ".reproject_backup").glob(f"*_{base}.zip"))
        with zipfile.ZipFile(z) as zf:
            for g, raw in originals.items():
                assert zf.read(f"{g}_{base}_polygons.json") == raw, "backup is not the files as they were"
    finally:
        _write_ax(pt, fp, None)


def test_written_sidecars_are_served_from_memory_until_the_file_changes(tmp_path, monkeypatch):
    """The virus scanner holds every just-written file on its first read (500
    BCI sidecars: 9.7 s). What a bulk writer just wrote is read back from memory
    -- but only while the file is still exactly that."""
    import builtins
    from .. import polygon_lod
    p = str(tmp_path / "g_img_polygons.json")
    data = {"points": [[1.5, 2.25], [3.0, 4.0]], "name": "g", "properties": {"a": 1}}
    polygon_lod.dump_sidecar(p, data, remember=True)
    real_open = builtins.open

    def _no_open(f, *a, **k):
        if os.path.abspath(str(f)) == os.path.abspath(p):
            raise AssertionError("read the file instead of the remembered text")
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", _no_open)
    assert polygon_lod.read_sidecar(p) == data
    monkeypatch.setattr(builtins, "open", real_open)
    import time
    time.sleep(0.02)
    with open(p, "w", encoding="utf-8") as f:            # someone else rewrites it
        json.dump({"points": [[9.0, 9.0]], "name": "other"}, f)
    assert polygon_lod.read_sidecar(p)["name"] == "other"


def test_reset_inside_the_editor_is_reprojected_once(iso, tmp_path):
    """Reset deletes the .ax while the dialog is open and calls
    _after_editor_ax_write; OK / Cancel afterwards must not move them again."""
    pt, folder, added = iso
    fp = _write_raster("tif_strip_u16", str(tmp_path / "reset"))
    _write_ax(pt, fp, GEOMS["crop"])
    pt.all_polygons["rs_once"] = {fp: {"name": "r", "type": "polygon", "coord_space": "image", "root": "",
                                       "image_ref_size": {"w": RAW_W, "h": RAW_H},
                                       "points": [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]}}
    added.append("rs_once")
    try:
        raw0 = _to_raw(pt, fp, pt.all_polygons["rs_once"][fp])
        pt._editor_ax_geom_snapshot = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, None)                         # the Reset
        assert pt._after_editor_ax_write() == 1
        assert pt._reproject_after_ax_change(pt._editor_ax_geom_snapshot) == 0
        assert np.allclose(_to_raw(pt, fp, pt.all_polygons["rs_once"][fp]), raw0, atol=1e-6)
    finally:
        pt._editor_ax_geom_snapshot = None
        _write_ax(pt, fp, None)


def test_a_stale_viewer_is_not_read_back_over_the_reprojected_records(iso, tmp_path, viewer_factory, monkeypatch):
    """update_all_polygons copies every on-screen item back into its record.
    Between a re-projection and the redraw the items still show the OLD frame
    -- reading them back undid the re-projection (Reset's own refresh does it)."""
    pt, folder, added = iso
    fp = _write_raster("tif_strip_u16", str(tmp_path / "stale"))
    _write_ax(pt, fp, None)
    imgd = pt._imagedata_or_fallback(fp)
    v = viewer_factory()
    v.image_data = imgd
    v.set_image(pt.convert_cv_to_pixmap(imgd.image))
    rec_v = {"viewer": v, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec_v]
    monkeypatch.setattr(v, "_find_project_owner", lambda: pt)
    sq = [(30.0, 20.0), (40.0, 20.0), (40.0, 30.0), (30.0, 30.0)]
    pt.all_polygons["stale_q"] = {fp: {"name": "stale_q", "type": "polygon", "coord_space": "image", "root": "",
                                       "image_ref_size": {"w": RAW_W, "h": RAW_H}, "points": list(sq)}}
    added.append("stale_q")
    pt._poly_norm_index_invalid = True       # as the app does when it adds a record
    try:
        pt.load_polygons(v, imgd)
        assert any(getattr(i, "name", None) == "stale_q" for i in v.scene().items())
        snap = pt._snapshot_ax_geometry()
        _write_ax(pt, fp, GEOMS["crop"])
        pt._reproject_after_ax_change(snap)
        moved = [tuple(p) for p in pt.all_polygons["stale_q"][fp]["points"]]
        assert v._ax_frame_stale
        pt.update_all_polygons()
        assert np.allclose(pt.all_polygons["stale_q"][fp]["points"], moved), (
            "update_all_polygons read the old-frame item back over the re-projected record")
        # redrawn on the new frame, reading back changes nothing
        imgd2 = pt._imagedata_or_fallback(fp)
        imgd2.image = pt.apply_aux_modifications(fp, imgd2.image, project_folder=pt.project_folder,
                                                 global_mode=False)
        v.image_data = imgd2
        rec_v["image_data"] = imgd2
        v.set_image(pt.convert_cv_to_pixmap(imgd2.image))
        pt.load_polygons(v, imgd2)
        assert not v._ax_frame_stale
        pt.update_all_polygons()
        assert np.allclose(pt.all_polygons["stale_q"][fp]["points"], moved, atol=1e-4)
    finally:
        try:
            pt.viewer_widgets.remove(rec_v)
        except ValueError:
            pass
        _write_ax(pt, fp, None)


def test_editor_mask_menu_lists_names_without_paging_crowns_in(tmp_path):
    from .. import polygon_lod
    from ..image_editor_dialog import ImageEditorDialog
    recs = {}
    for i in range(4):
        r = polygon_lod.LazyPolygonRecord({"name": f"crown{i}", "full_point_count": 3 + i},
                                          str(tmp_path / f"missing_{i}.json"))
        recs[f"g{i}"] = {"C:/img.tif": r}

    class _Parent:
        all_polygons = recs

    class _Fake:
        image_filepath = "C:/img.tif"

        def parent(self):
            return _Parent()
    names = ImageEditorDialog._get_all_polygon_names_in_project(_Fake())
    assert names == {f"crown{i}" for i in range(4)}
    assert ImageEditorDialog._get_polygon_names_for_current_image(_Fake()) == names
    assert not any(r["C:/img.tif"].is_materialised for r in recs.values()), (
        "listing names paged the crowns in -- 55 of the 60 s the editor took to open")


def test_preview_is_read_once_and_handed_out_as_copies(tmp_path):
    from .. import raster_reader
    p = str(tmp_path / "prev.bin")
    open(p, "wb").write(b"x")
    calls = []

    class _R:
        def read_bands_full(self, bands, level=0, step=1):
            calls.append((tuple(bands), level, step))
            return np.arange(24, dtype=np.uint16).reshape(2, 4, 3)
    raster_reader.clear_preview_cache()
    a = raster_reader.read_preview(_R(), p, [0, 1, 2], level=0, step=3)
    a[0, 0, 0] = 999                                   # a caller edits its copy in place
    b = raster_reader.read_preview(_R(), p, [0, 1, 2], level=0, step=3)
    assert len(calls) == 1, "the preview was decoded again (4 s per editor open / refresh on BCI)"
    assert b[0, 0, 0] == 0, "the cached preview was modified through a caller's copy"
    raster_reader.read_preview(_R(), p, [0, 1, 2], level=0, step=2)
    assert len(calls) == 2, "a different decimation must not be served from the cache"
    raster_reader.clear_preview_cache()


def test_area_resample_is_cv2_inter_area_also_mid_cell():
    import cv2
    from ..project_tab import _area_resample
    rng = np.random.default_rng(3)
    for _ in range(25):
        H, W = (int(v) for v in rng.integers(30, 200, size=2))
        img = rng.integers(0, 255, size=(H, W, 3)).astype(np.float64)
        nw, nh = int(rng.integers(3, W)), int(rng.integers(3, H))
        ref = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        got = _area_resample(img, np.arange(nw + 1) * (W / nw), np.arange(nh + 1) * (H / nh))
        assert np.abs(ref - got).max() < 1e-3
        j0, j1, i0, i1 = nw // 4, nw // 2 + 1, nh // 4, nh // 2 + 1
        rx0, ry0 = int(np.floor(j0 * W / nw)), int(np.floor(i0 * H / nh))
        sub = img[ry0:int(np.ceil(i1 * H / nh)), rx0:int(np.ceil(j1 * W / nw))]
        got2 = _area_resample(sub, np.arange(j0, j1 + 1) * (W / nw) - rx0, np.arange(i0, i1 + 1) * (H / nh) - ry0)
        assert np.abs(ref[i0:i1, j0:j1] - got2).max() < 1e-3


# --------------------------------------------------------------------------
# The zoom tile shows exported pixels, never finer ones
# --------------------------------------------------------------------------
def _tile_request(pt, fp, ax, disp_shape, rect, monkeypatch):
    """Run one tile request synchronously; returns (placement, pre-stretch array, windows read)."""
    import time
    import types
    from ..raster_reader import probe, open_reader
    prof = probe(fp)
    reader = open_reader(fp, prof)
    image_data = types.SimpleNamespace(
        filepath=fp, image=np.zeros(disp_shape, np.float32), reader=reader, profile=prof,
        preview_bands=[0, 1, 2], full_shape=(prof.height, prof.width, prof.count),
        display_scale=prof.width / float(disp_shape[1]))
    got = {}
    reads = []
    original = type(reader).read_window

    def _spy(self, x0, y0, x1, y1, bands=None, level=0, step=1):
        reads.append((x0, y0, x1, y1, step))
        return original(self, x0, y0, x1, y1, bands=bands, level=level, step=step)
    monkeypatch.setattr(type(reader), "read_window", _spy)

    def _render(arr, viewer, *a, **k):
        got["arr"] = np.array(arr)
        return np.zeros(arr.shape[:2] + (3,), np.uint8)
    monkeypatch.setattr(pt, "_render_with_viewer_stretch", _render)

    class _VP:
        def width(self): return 400
        def height(self): return 300

    class _Sig:
        def emit(self, pm, sx, sy, scx, scy, rid):
            got["place"] = (sx, sy, scx, scy, pm.width(), pm.height())
    viewer = types.SimpleNamespace(viewport=lambda: _VP(), image_data=image_data,
                                   highres_tile_ready=_Sig(), _clear_highres_overlay=lambda: None)
    pt._request_highres_viewport_region(viewer, image_data, rect, request_id=None, ax=ax)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and "place" not in got and (reads or time.monotonic() < deadline - 9.5):
        time.sleep(0.02)
    return got, reads


@pytest.mark.parametrize("name", ["rgb_16bit_tiled_bip_cog", "hyperspectral_200band"])
def test_zoom_tile_is_the_exported_pixels_when_the_export_is_downsized(synthetic_project, name, monkeypatch):
    pt = synthetic_project
    fp = fixture_image_path(name)
    from ..raster_reader import probe
    prof = probe(fp)
    W, H = prof.width, prof.height
    ax = {"resize": {"scale": 75}, "resize_enabled": True}
    eW, eH = round(W * 0.75), round(H * 0.75)
    disp = (eH // 3, eW // 3, 3)                       # a preview 3x coarser than the export
    kx, ky = eW / disp[1], eH / disp[0]
    rect = QtCore.QRectF(disp[1] * 0.3, disp[0] * 0.3, disp[1] * 0.35, disp[0] * 0.35)
    got, reads = _tile_request(pt, fp, ax, disp, rect, monkeypatch)
    assert "place" in got, "no tile produced"
    sx, sy, scx, scy, tw, th = got["place"]
    assert scx == pytest.approx(1.0 / kx) and scy == pytest.approx(1.0 / ky), (
        f"one tile pixel is {scx:.4f} x {scy:.4f} display units; one exported pixel is "
        f"{1 / kx:.4f} -- the tile shows source pixels finer than the export")
    gx0, gy0 = sx * kx, sy * ky
    assert gx0 == pytest.approx(round(gx0)) and gy0 == pytest.approx(round(gy0)), "tile off the exported grid"
    _write_ax(pt, fp, ax)
    try:
        img, info = pt._get_export_image(fp)
        assert (info["W"], info["H"]) == (eW, eH)
        ref = np.asarray(img)[int(round(gy0)):int(round(gy0)) + th, int(round(gx0)):int(round(gx0)) + tw, :3]
        tile = got["arr"][..., :3]
        assert np.abs(ref.astype(float) - tile.astype(float)).max() <= 1.0, (
            "the tile's pixels are not the exported pixels")
    finally:
        _write_ax(pt, fp, None)


def test_no_tile_when_the_preview_already_is_the_export_resolution(synthetic_project, monkeypatch):
    """pasta (5): a 566 x 511 export shown as a 566 x 511 pixmap. Any tile could
    only show detail the export does not have."""
    pt = synthetic_project
    fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    from ..raster_reader import probe
    prof = probe(fp)
    ax = {"resize": {"px_w": prof.width // 4}, "resize_enabled": True}
    from .. import ax_geometry as G
    eh, ew = G.frame_hw(ax, prof.width, prof.height)
    rect = QtCore.QRectF(2, 2, ew * 0.5, eh * 0.5)
    got, reads = _tile_request(pt, fp, ax, (eh, ew, 3), rect, monkeypatch)
    assert not reads and "place" not in got, f"a tile was read ({reads}) although the preview is the export"


# --------------------------------------------------------------------------
# Drawing points by dragging paints each pixel once
# --------------------------------------------------------------------------
def test_point_drag_paints_each_exported_pixel_once(synthetic_project, viewer_factory, force_preview_decimation,
                                                   monkeypatch):
    pt = synthetic_project
    fp = fixture_image_path("hyperspectral_200band")
    for attr in ("_imgdata_cache", "_pixmap_cache", "_export_image_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    imgd = pt._imagedata_or_fallback(fp)
    v = viewer_factory()
    v.resize(500, 500)
    v.show()
    v.image_data = imgd
    v.set_image(pt.convert_cv_to_pixmap(imgd.image))
    monkeypatch.setattr(v, "_find_project_owner", lambda: pt)
    v.resetTransform()
    v.scale(20.0, 20.0)
    v.centerOn(v._image.mapToScene(QtCore.QPointF(8.0, 8.0)))
    QtWidgets.QApplication.processEvents()
    kx, ky = v._raw_to_pixmap_scale()
    v.drawing_mode = "point"
    c = v.viewport().rect().center()
    path = [c + QtCore.QPoint(dx, 0) for dx in range(0, 60, 3)] + [c + QtCore.QPoint(dx, 0) for dx in range(57, -1, -3)]
    QtWidgets.QApplication.sendEvent(v.viewport(), QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(path[0]), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
        QtCore.Qt.ShiftModifier))
    for p in path[1:]:
        QtWidgets.QApplication.sendEvent(v.viewport(), QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(p), QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
            QtCore.Qt.ShiftModifier))
    pts = [v._image.mapFromScene(v.currentPolygon.at(i)) for i in range(v.currentPolygon.count())]
    cells = [(int(np.floor(q.x() * kx)), int(np.floor(q.y() * ky))) for q in pts]
    v.drawing = False
    v.currentPolygon = QtGui.QPolygonF()
    assert len(cells) == len(set(cells)), f"pixels repeated along the drag: {cells}"
    assert len(cells) >= 3, f"the drag should have painted a run of pixels, got {cells}"
    for q in pts:
        assert ((q.x() * kx) % 1.0, (q.y() * ky) % 1.0) == pytest.approx((0.5, 0.5), abs=1e-6), (
            "a dragged point is not at its pixel's centre")
