"""
CSV export speed work (2026-10-07) -- faster with the SAME bytes -- and the
defects found on the way:

  * median and quantiles come from ONE partition and keep numpy's own bits (a
    zero picked where the data holds -0.0 and +0.0 is handed back to numpy);
  * the band-expression evaluator skips its NaN copies when there is nothing
    to replace, and never hands back the caller's own memory;
  * the band-math cache belongs to the export image it was computed on: a
    re-export after an .ax edit used the OLD image's band math (background
    export, BCI copy: every sum/ratio row wrong, R/G/B right); the background
    worker drops it per file (it kept whole-frame arrays for every file);
  * the label-band guess runs once per file, not once per polygon;
  * a polygon thumbnail on a windowed (large) raster is drawn -- it has no
    `.size`, was taken for an empty image, and every one was skipped;
  * the augmentation preview of a windowed raster reads one window, not the
    whole image;
  * a whole-image .ax replay that does not fit in memory is refused with a
    reason (a windowable one is read window by window instead);
  * the viewer's preview holds at most 16 Mpx; saving a polygon's size never
    builds the export image.
"""
import copy
import json
import os
import threading

import numpy as np
import pytest
from PyQt5 import QtGui

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name

pytestmark = [pytest.mark.extraction]


def _clear(pt):
    for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                 "_per_band_nd_cache", "_master_nd_cache", "_export_frame_hw_cache",
                 "_bm_expr_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


@pytest.fixture
def ax_on(synthetic_project):
    """Write a temporary .ax for a fixture of the shared project that has none."""
    pt = synthetic_project
    written = []

    def _make(name, ax):
        fp = fixture_image_path(name)
        axp = pt._ax_path_for(fp)
        assert not os.path.exists(axp) or axp in written, "precondition: fixture has no .ax"
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(ax, f)
        written.append(axp)
        return fp

    yield _make
    for axp in written:
        if os.path.exists(axp):
            os.remove(axp)
    _clear(pt)


@pytest.fixture
def app_export_caches(synthetic_project, monkeypatch):
    """The per-export caches save_polygons_to_csv sets up before an export."""
    from ..raster_reader import ByteBudgetedLRUDict
    from .. import project_tab as PT
    pt = synthetic_project
    monkeypatch.setattr(pt, "_export_cache", ByteBudgetedLRUDict(PT._EXPORT_CACHE_BUDGET_BYTES), raising=False)
    monkeypatch.setattr(pt, "_export_cache_lock", threading.Lock(), raising=False)
    return pt


def _poly(pt, name):
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    return group, fp, pt.all_polygons[group][fp]


# ---------------------------------------------------------------------------
# statistics: one partition, numpy's bits
# ---------------------------------------------------------------------------
def _arrays():
    rng = np.random.default_rng(11)
    for n in (1, 2, 3, 4, 5, 8, 9, 101, 1000, 4097):
        yield rng.normal(size=n)
        yield rng.integers(0, 65535, size=n).astype(np.float64)
        yield rng.integers(0, 2, size=n).astype(np.float64)
        yield rng.normal(size=n) * 1e-30 + rng.integers(0, 2, size=n) * 1e30
        yield rng.choice([-0.0, 0.0, 1.0, -1.0], size=n)
        yield np.full(n, -0.0)


@pytest.mark.parametrize("qs", [[], [10, 25, 75, 90], [0, 100], [0.1, 33.3, 99.9]])
def test_median_and_percentiles_are_numpys_bits(qs):
    from ..performance import median_and_percentiles
    for a in _arrays():
        med, vals = median_and_percentiles(a, True, qs)
        assert np.float64(med).view(np.uint64) == np.float64(np.median(a)).view(np.uint64), (a[:6], med)
        if qs:
            ref = np.percentile(a, qs)
            assert [np.float64(v).view(np.uint64) for v in vals] == [r.view(np.uint64) for r in ref], (a[:6], qs)


def test_fast_stats_matches_numpy_directly():
    from ..performance import fast_stats
    for a in _arrays():
        for arr in (a, a.astype(np.float32), np.where(np.arange(a.size) % 7 == 3, np.nan, a)):
            got = fast_stats(arr, True, True, True, [10, 25, 75, 90])
            v = np.asarray(arr, dtype=np.float64).ravel()
            v = v[np.isfinite(v)]
            if not v.size:
                assert got["Mean"] is None
                continue
            ref = {"Mean": float(np.mean(v)), "Standard Deviation": float(np.std(v, ddof=0)),
                   "Median": float(np.median(v))}
            ref.update({f"Q{q}": float(x) for q, x in zip((10, 25, 75, 90), np.percentile(v, [10, 25, 75, 90]))})
            assert repr(got) == repr({k: ref[k] for k in got}), (got, ref)


# ---------------------------------------------------------------------------
# band-expression evaluator
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    out = np.zeros(np.broadcast(a, b).shape, np.float32)
    ok = (b != 0) & np.isfinite(a) & np.isfinite(b)
    np.divide(a, b, out=out, where=ok)
    return out


@pytest.mark.parametrize("with_nan", [False, True], ids=["finite", "nan_inf"])
def test_eval_band_expression_values_and_no_aliasing(with_nan):
    from ..utils import eval_band_expression
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(17, 23, 3)).astype(np.float32)
    if with_nan:
        img[2, 3, 0] = np.nan
        img[5, 7, 1] = np.inf
        img[9, 9, 2] = -np.inf
    before = img.copy()
    x = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    b1, b2, b3 = x[..., 0], x[..., 1], x[..., 2]
    cases = {"b1": b1, "b1+b2": b1 + b2, "b1>100": (b1 > 100).astype(np.float32),
             "(b2-b1)/(b2+b1)": _safe_div(b2 - b1, b2 + b1), "b3/0": _safe_div(b3, np.float32(0))}
    for expr, ref in cases.items():
        got = eval_band_expression(img, expr)
        assert got.dtype == np.float32 and got.shape == img.shape[:2], expr
        assert np.array_equal(got.view(np.uint32), np.nan_to_num(ref).astype(np.float32).view(np.uint32)), expr
        assert not np.may_share_memory(got, img), f"{expr}: the result is the caller's own memory"
        got[...] = -1                      # the caller may write into what it gets
    assert np.array_equal(img, before, equal_nan=True), "the evaluator changed its input"


# ---------------------------------------------------------------------------
# band-math cache: tied to the export image
# ---------------------------------------------------------------------------
BM_OPTS = {"stats": {"mean": True, "median": True},
           "band_math": {"enabled": True, "formulas": {"sum3": "b1+b2+b3"}}}


def _bm_rows(rows):
    return {r["Channel"]: (r.get("Mean"), r.get("Median"), r.get("Pixel Count"))
            for r in rows if r.get("Channel") == "sum3"}


def test_band_math_after_an_ax_edit_is_computed_on_the_new_image(synthetic_project, ax_on):
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, poly = _poly(pt, name)
    _clear(pt)
    ax_on(name, {"crop_enabled": True, "crop_rect": {"x": 0, "y": 0, "width": 90, "height": 90},
                 "crop_rect_ref_size": {"w": 96, "h": 96}})
    rows1, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=BM_OPTS)
    assert _bm_rows(rows1), "no band-math row -- fixture/test problem"
    # the user edits the crop and exports again in the same session: a new
    # export resets the export-image cache, the band-math cache stays
    for attr in ("_export_image_cache", "_export_cache", "_export_frame_hw_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()
    ax_on(name, {"crop_enabled": True, "crop_rect": {"x": 7, "y": 5, "width": 85, "height": 88},
                 "crop_rect_ref_size": {"w": 96, "h": 96}})
    os.utime(pt._ax_path_for(fp), None)
    rows_same, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=BM_OPTS)
    _clear(pt)
    rows_fresh, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=BM_OPTS)
    assert _bm_rows(rows_same) == _bm_rows(rows_fresh), (
        "the re-export's band math came from the image before the .ax edit",
        _bm_rows(rows_same), _bm_rows(rows_fresh))
    rgb = lambda rows: {r["Channel"]: r.get("Mean") for r in rows if r.get("Channel") in ("R", "G", "B")}
    assert rgb(rows_same) == rgb(rows_fresh)


def test_background_export_drops_each_files_band_math(synthetic_project, tmp_path):
    from ..project_tab import ExportWorker
    pt = synthetic_project
    name = "rgb_8bit_untiled"
    group, fp, poly = _poly(pt, name)
    _clear(pt)
    w = ExportWorker(pt, str(tmp_path / "out.csv"), dict(BM_OPTS), [(group, fp, copy.deepcopy(poly))],
                     {}, [], False, False, "sequential", 1, snapshot_dir=str(tmp_path))
    w.run()
    assert os.path.exists(tmp_path / "out.csv")
    key = os.path.normcase(os.path.abspath(fp))
    assert key not in (getattr(pt, "_bm_expr_cache", None) or {}), \
        "the file's whole-frame band-math arrays outlived its part of the export"


# ---------------------------------------------------------------------------
# label-band guess: once per file
# ---------------------------------------------------------------------------
def test_label_band_guess_runs_once_per_file(synthetic_project, force_lazy_export, app_export_caches, monkeypatch):
    from ..lazy_analysis import LazyAnalysis
    pt = synthetic_project
    name = "hyperspectral_200band"
    group, fp, poly = _poly(pt, name)
    calls = {"n": 0}
    orig = LazyAnalysis.sample_bands

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(LazyAnalysis, "sample_bands", counting)
    opts = {"stats": {"mean": True, "count01": True}}
    _clear(pt)
    rows1, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=opts)
    first = calls["n"]
    assert rows1 and first >= 1, "the guess did not sample the image -- vacuous"
    rows2, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=opts)
    assert calls["n"] == first, f"the second polygon of the file sampled again ({calls['n'] - first} more)"
    strip = lambda rows: [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]
    assert strip(rows1) == strip(rows2)
    _clear(pt)


# ---------------------------------------------------------------------------
# windowed images in the thumbnail and augmentation-preview paths
# ---------------------------------------------------------------------------
def test_thumbnail_is_drawn_on_a_windowed_raster(synthetic_project, force_lazy_export, monkeypatch, tmp_path):
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, _poly_rec = _poly(pt, name)
    _clear(pt)
    img, info = pt._get_export_image(fp)
    assert info.get("lazy"), "lazy path not engaged"
    monkeypatch.setattr(pt, "project_folder", str(tmp_path))

    class _Data:
        filepath = fp
        is_rgb = True
        is_thermal = False

    pt.generate_thumbnail(group, _Data())
    out = list((tmp_path / "thumbnails").glob("*")) if (tmp_path / "thumbnails").exists() else []
    assert out, "no thumbnail written for a windowed raster"
    _clear(pt)


def test_augmentation_preview_reads_one_window(synthetic_project, force_lazy_export):
    from ..ml_augmentation_options_dialog import MLAugmentationOptionsDialog as D
    pt = synthetic_project
    fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    _clear(pt)

    class _Fake:
        _preview_img_cache = {}

        def _get_export_image(self, f):
            return pt._get_export_image(f)

    fake = _Fake()
    img = D._load_preview_source(fake, fp)
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[:2] == (96, 96), getattr(img, "shape", img)
    _clear(pt)


# ---------------------------------------------------------------------------
# memory: a whole-image replay that cannot fit is refused; previews are capped
# ---------------------------------------------------------------------------
def test_whole_image_replay_that_cannot_fit_is_refused(synthetic_project, ax_on, monkeypatch):
    from .. import project_tab as PT
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    monkeypatch.setattr(PT, "_available_ram_bytes", lambda: 4096)
    # an up-size cannot be computed window by window
    fp = ax_on(name, {"resize_enabled": True, "resize": {"width": 200, "height": 200}})
    _clear(pt)
    img, info = pt._get_export_image(fp)
    assert img is None and "error" in info, (type(img), info)
    assert os.path.normcase(os.path.abspath(fp)) in pt._export_errors
    # a windowable .ax on the same machine is read window by window instead
    fp = ax_on(name, {"band_enabled": True, "band_expression": "b1>100"})
    os.utime(pt._ax_path_for(fp), None)
    _clear(pt)
    img, info = pt._get_export_image(fp)
    assert img is not None and info.get("lazy"), info
    _clear(pt)


def test_a_refused_image_is_reported_by_the_export(synthetic_project, ax_on, monkeypatch, tmp_path):
    """No rows and no exception: the completion message said "(0 errors)"."""
    from .. import project_tab as PT
    from ..project_tab import ExportWorker
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, poly = _poly(pt, name)
    monkeypatch.setattr(PT, "_available_ram_bytes", lambda: 4096)
    ax_on(name, {"resize_enabled": True, "resize": {"width": 200, "height": 200}})
    other_group, other_fp, other_poly = _poly(pt, "rgb_8bit_untiled")
    _clear(pt)
    w = ExportWorker(pt, str(tmp_path / "out.csv"), dict(BM_OPTS),
                     [(group, fp, copy.deepcopy(poly)), (other_group, other_fp, copy.deepcopy(other_poly))],
                     {}, [], False, False, "sequential", 1, snapshot_dir=str(tmp_path))
    got = []
    w.finished.connect(lambda p, ok, msg: got.append((ok, msg)))
    w.run()
    assert got and got[-1][0], got
    msg = got[-1][1]
    assert "1 image(s) skipped" in msg and os.path.basename(fp) in msg, msg
    _clear(pt)


def test_a_refused_image_is_reported_by_the_ml_export(synthetic_project, ml_manager_factory, ax_on,
                                                      monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from .. import project_tab as PT
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, _rec = _poly(pt, name)
    monkeypatch.setattr(PT, "_available_ram_bytes", lambda: 4096)
    ax_on(name, {"resize_enabled": True, "resize": {"width": 200, "height": 200}})
    _clear(pt)
    mgr = ml_manager_factory(pt)
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() == group)
    answers = iter([("Average Pixel Value", True), ("1x1", True)])
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a, **k: next(answers)))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "ml.csv"), "")))
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind, **k: shown.append((_k, a[1] if len(a) > 1 else "",
                                                                                a[2] if len(a) > 2 else ""))))
    mgr.export_csv_data()
    done = [s for s in shown if s[1] == "Export Complete"]
    assert done, shown
    assert done[-1][0] == "warning" and "1 image(s) skipped" in done[-1][2] \
        and os.path.basename(fp) in done[-1][2], done[-1]
    _clear(pt)


def _ix_read_bands_full(rd, bands, step):
    """read_bands_full as it was (np.ix_ gather per band per tile)."""
    lv = rd._levels[0]
    out = np.empty((len(bands), (lv["height"] + step - 1) // step, (lv["width"] + step - 1) // step),
                   dtype=rd.profile.dtype)
    for tr in range(lv["down"]):
        ty0 = tr * lv["th"]; ty1 = min(lv["height"], ty0 + lv["th"])
        sy0 = ((ty0 + step - 1) // step) * step
        if sy0 >= ty1:
            continue
        rows = np.arange(sy0, ty1, step)
        seg = ([rd._segment_index(lv, b, tr, tc) for b in bands for tc in range(lv["across"])]
               if lv["planar"] == 2 else [rd._segment_index(lv, 0, tr, tc) for tc in range(lv["across"])])
        dec = rd._decode_many(0, seg, lv["workers"])
        for tc in range(lv["across"]):
            tx0 = tc * lv["tw"]; tx1 = min(lv["width"], tx0 + lv["tw"])
            sx0 = ((tx0 + step - 1) // step) * step
            if sx0 >= tx1:
                continue
            cols = np.arange(sx0, tx1, step)
            for bi, b in enumerate(bands):
                if lv["planar"] == 2:
                    tile = dec[rd._segment_index(lv, b, tr, tc)]
                else:
                    t = dec[rd._segment_index(lv, 0, tr, tc)]
                    tile = t[..., b] if t.ndim == 3 else t
                p = tile[np.ix_(rows - ty0, cols - tx0)]
                out[bi, sy0 // step:sy0 // step + p.shape[0], sx0 // step:sx0 // step + p.shape[1]] = p
    return np.moveaxis(out, 0, -1)


@pytest.mark.parametrize("name", ["rgb_8bit_untiled", "rgb_16bit_tiled_bip_cog", "multiband_8band_ancillary",
                                  "hyperspectral_200band", "ax_crop_nodata_source"])
def test_decimated_full_read_is_the_same_pixels(name):
    """read_bands_full(step) takes strided views instead of an np.ix_ gather per
    band per tile (the editor's all-band preview of the BCI mosaic: 0.95 -> 0.29 s)."""
    from .. import raster_reader as RR
    fp = fixture_image_path(name)
    prof = RR.probe(fp)
    rd = RR.open_reader(fp, prof)
    assert rd is not None
    for step in (2, 3, 5, 7):
        for bands in ([0], list(range(min(prof.count, 4))), list(range(prof.count))[::-1][:3]):
            got = rd.read_bands_full(bands, 0, step)
            ref = _ix_read_bands_full(rd, [int(b) for b in bands], step)
            assert got.shape == ref.shape and got.dtype == ref.dtype, (step, bands, got.shape, ref.shape)
            assert np.array_equal(got, ref, equal_nan=True), (name, step, bands)


def test_editor_save_takes_the_pre_resize_size_from_the_geometry_alone(synthetic_project, qapp):
    """save_modifications_to_file ran the WHOLE editor pipeline (band expression
    included) only to read the size after crop/rotation: ~1 s per Apply on the
    BCI mosaic preview. Same size now, and the display state of the image on
    screen is left as it was."""
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: fixture has no .ax"
    dlg = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
    try:
        crop = {"x": 10, "y": 5, "width": 60, "height": 40}
        dlg.modifications.update({"rotate": 90, "crop_rect": crop, "crop_rect_ref_size": {"w": 96, "h": 96},
                                  "band_expression": "b1>100", "band_enabled": True,
                                  "nodata_values": ["b2<300"], "nodata_enabled": True})
        dlg.crop_rect = dict(crop)
        dlg.last_crop_ref_size = (96, 96)
        full = dict(dlg.modifications)
        full.pop("resize", None)
        ref = dlg.apply_all_modifications_to_image(dlg.base_image, full)     # the old way
        ref_wh = (int(ref.shape[1]), int(ref.shape[0]))
        dlg.resize_width_input.setText("17")
        dlg.resize_height_input.setText("13")
        marker = np.zeros((3, 3), bool)
        dlg._has_expression_band, dlg._display_nodata_mask = "screen-state", marker
        assert dlg.save_modifications_to_file() is not None
        with open(axp, encoding="utf-8") as f:
            ax = json.load(f)
        assert ax.get("resize_ref_size") == {"w": ref_wh[0], "h": ref_wh[1]}, (ax.get("resize_ref_size"), ref_wh)
        assert dlg._has_expression_band == "screen-state" and dlg._display_nodata_mask is marker, \
            "saving changed the display state of the image on screen"
    finally:
        try:
            dlg.close(); dlg.deleteLater()
        except Exception:
            pass
        if os.path.exists(axp):
            os.remove(axp)
        _clear(pt)


def test_preview_is_capped_at_16_megapixels():
    from types import SimpleNamespace
    from .. import project_tab as PT
    small_many_px = SimpleNamespace(bytes_if_fully_loaded=200 * 1024 * 1024, width=9439, height=8516, dtype="uint8")
    assert PT._wants_preview(small_many_px), "a 9439 x 8516 raster under 512 MB was loaded whole"
    assert not PT._wants_preview(SimpleNamespace(bytes_if_fully_loaded=10 * 1024 * 1024, width=2000, height=1500,
                                                 dtype="uint8"))
    budget = PT._preview_budget_bytes(small_many_px, 3)
    assert budget == PT._PREVIEW_MAX_PIXELS * 3 * 1
    assert PT._preview_budget_bytes(SimpleNamespace(dtype="float32"), 200) == PT._PREVIEW_LOAD_THRESHOLD_BYTES


def test_saving_a_polygons_size_never_builds_the_export_image(synthetic_project, viewer_factory, monkeypatch):
    pt = synthetic_project
    name = "rgb_8bit_untiled"
    group, fp, _rec = _poly(pt, name)
    saved = {g: copy.deepcopy(files.get(sfp)) for g, sfp in pt._poly_index_lookup(fp)
             for files in [pt.all_polygons.get(g, {})] if sfp in files}
    assert saved, "no polygon on the fixture -- vacuous"

    def boom(*a, **k):
        raise AssertionError("_get_export_image called to save a polygon's size")

    monkeypatch.setattr(pt, "_get_export_image", boom)
    v = viewer_factory()
    qimg = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))

    class _D:
        filepath = fp

    v.image_data = _D()
    try:
        pt.update_current_polygons_pixmap_size(v)
        rec = pt.all_polygons[group][fp]
        assert rec.get("image_size") == [64, 64], rec.get("image_size")
    finally:
        for g, sfp in pt._poly_index_lookup(fp):
            if g in saved and saved[g] is not None:
                pt.all_polygons[g][sfp] = saved[g]


# ---------------------------------------------------------------------------
# final review (2026-10-07): a polygon too large to read never escapes the
# training slot; the background export drops each file's analysis image
# ---------------------------------------------------------------------------
def test_training_skips_a_polygon_too_large_to_read(synthetic_project, ml_manager_factory, force_lazy_export,
                                                    monkeypatch):
    from .. import lazy_analysis as LA
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, _rec = _poly(pt, name)
    _clear(pt)
    mgr = ml_manager_factory(pt)
    img, _c = mgr._get_export_image(fp)
    assert not isinstance(img, np.ndarray), "lazy path not engaged"
    monkeypatch.setattr(LA, "_FULL_MATERIALIZE_LIMIT_BYTES", 16)     # every polygon window is "too large"
    mgr._train_skipped_polygons = []
    out = mgr._sample_training_file(
        group, fp, img, mgr._snapshot_polygons_by_group([group]),
        selected_band_indices=[0, 1, 2], selected_expressions=[], window_size=1, samples_per_poly=50,
        num_features=3, augmentation_cfg={"enabled": False}, augmentation_enabled=False,
        n_variants=1, rng=np.random.default_rng(0))
    assert len(out[0]) == 0, "rows from a polygon that could not be read"
    assert mgr._train_skipped_polygons and group in mgr._train_skipped_polygons[0]
    _clear(pt)


def test_background_export_drops_each_files_analysis_image(synthetic_project, tmp_path, force_lazy_export):
    from ..project_tab import ExportWorker
    pt = synthetic_project
    name = "rgb_16bit_tiled_bip_cog"
    group, fp, poly = _poly(pt, name)
    _clear(pt)
    w = ExportWorker(pt, str(tmp_path / "out.csv"), dict(BM_OPTS), [(group, fp, copy.deepcopy(poly))],
                     {}, [], False, False, "sequential", 1, snapshot_dir=str(tmp_path))
    w.run()
    key = os.path.normcase(os.path.abspath(fp))
    left = [k for k in (getattr(pt, "_export_image_cache", None) or {})
            if isinstance(k, tuple) and len(k) > 1 and os.path.normcase(os.path.abspath(str(k[1]))) == key]
    assert not left, "the file's (windowed) analysis image outlived its part of the export"
