"""
Display stretch: NoData never enters the statistics, and Stretch to view.

Nathan, 2026-11-16: "Viewer stretch not adapting to the new range and
adapting badly with -9999 values", plus an ENVI-style "stretch to view"
button on the viewer's small bottom-left square, and a stretch window that
grew with the band count.

  * NaN, ArcGIS float sentinels (|v| > 1e37), -9999 and the file's effective
    NoData (.ax UNION the GDAL_NODATA tag) are left out of every stretch's
    statistics: the render engine, the stretch bar, the Stretch Viewer dialog.
  * A zoom tile is read raw from the file; its NoData becomes NaN like the
    preview's, so it is never drawn as data.
  * Stretch to view takes the statistics from the pixels on screen only.
  * The Stretch Viewer stays compact on a 298-band cube.
"""
import types

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import stretch_stats as ss
from ..project_tab import _StretchParams, ImageStretchDialog
from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer, pytest.mark.extraction]


def _dem(h=100, w=100):
    """An elevation model (50..150 m) with a -9999 border, NaN holes and
    ArcGIS float sentinels."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = (50.0 + 100.0 * (xx + yy) / float(h + w - 2)).astype(np.float32)
    a[:8, :] = -9999.0
    a[:, :8] = -9999.0
    a[40:45, 40:45] = np.nan
    a[60, 60:70] = -3.4028235e38
    a[70, 60:70] = 3.4028235e38
    return a


def _sp(**kw):
    base = dict(mode="percentile", low_p=0.0, high_p=100.0, k_sigma=1.0, min_val=None,
                max_val=None, per_channel=True, clip=True, scope="viewer", display_mode="auto")
    base.update(kw)
    return _StretchParams(**base)


def _valid(a):
    return a[np.isfinite(a) & (np.abs(a) < 1e37) & (a != -9999.0)]


# ---------------------------------------------------------------------------
# stretch_stats itself
# ---------------------------------------------------------------------------
def test_valid_mask_drops_nan_sentinels_minus9999_and_declared_values():
    a = np.array([1.0, np.nan, np.inf, -9999.0, -3.4e38, 3.4e38, -32768.0, 7.0], np.float32)
    ok = ss.valid_mask(a, [-32768])
    assert ok.tolist() == [True, False, False, False, False, False, False, True]


def test_expression_rules_are_not_numbers():
    """``b1<123`` names a file band; it must not be read as a value."""
    assert ss.numeric_nodata(["b1<123", "nan", -1], implicit=False) == [-1.0]


def test_integer_data_ignores_out_of_range_nodata():
    a = np.array([[0, 10, 255]], np.uint8)
    assert ss.valid_mask(a, [-9999]).all()
    assert ss.valid_mask(a, [255]).tolist() == [[True, True, False]]


def test_bounds_of_a_dem_come_from_valid_pixels_only():
    a = _dem()
    v = _valid(a)
    lo, hi = ss.channel_bounds(a, "minmax")
    assert (lo, hi) == (pytest.approx(float(v.min())), pytest.approx(float(v.max())))
    lo, hi = ss.channel_bounds(a, "percentile", 2, 98)
    want = np.percentile(v.astype(np.float64), [2, 98])
    assert lo == pytest.approx(want[0], rel=1e-4) and hi == pytest.approx(want[1], rel=1e-4)


def test_a_channel_with_no_valid_pixel_gets_a_usable_span():
    a = np.full((10, 10, 2), -9999.0, np.float32)
    a[..., 1] = np.linspace(0, 1, 100).reshape(10, 10)
    lo, hi = ss.channel_bounds(a, "minmax")
    assert (float(lo[0]), float(hi[0])) == (0.0, 1.0)
    assert float(lo[1]) == pytest.approx(0.0) and float(hi[1]) == pytest.approx(1.0)


def test_window_restricts_the_statistics():
    a = np.zeros((20, 40), np.float32)
    a[:, 20:] = 1000.0 + np.arange(20, dtype=np.float32)[None, :]
    lo_all, _ = ss.channel_bounds(a, "minmax")
    lo_win, hi_win = ss.channel_bounds(a, "minmax", window=(25, 0, 40, 20))
    assert lo_all == 0.0
    assert lo_win == pytest.approx(1005.0) and hi_win == pytest.approx(1019.0)


def test_nan_nodata_window_turns_fill_into_nan_per_band():
    w = np.array([[[1, -9999, 5], [2, 3, 3.4e38]]], np.float32)       # (1, 2, 3)
    out = ss.nan_nodata_window(w, [-9999], file_bands=[0, 1, 2])
    assert np.isnan(out[0, 0, 1]) and np.isnan(out[0, 1, 2])
    assert out[0, 0, 0] == 1 and out[0, 1, 1] == 3
    # per-band expression: b2 (file band 1) < 4 -> only that channel
    out2 = ss.nan_nodata_window(np.array([[[1, 3, 1]]], np.int16), ["b2<4"], file_bands=[0, 1, 2])
    assert out2.dtype == np.float32 and np.isnan(out2[0, 0, 1]) and out2[0, 0, 0] == 1


# ---------------------------------------------------------------------------
# The render engine (C0: _compute_stretched_preview)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["percentile", "absolute", "stddev"])
def test_render_bounds_ignore_nodata(synthetic_project, mode):
    pt = synthetic_project
    a = _dem()
    v = _valid(a).astype(np.float64)
    b = {}
    out = pt._compute_stretched_preview(a, _sp(mode=mode, k_sigma=2.0), bounds_out=b)
    assert out is not None
    lo, hi = float(np.asarray(b["lo"])), float(np.asarray(b["hi"]))
    if mode == "stddev":
        want = (v.mean() - 2 * v.std(), v.mean() + 2 * v.std())
    else:
        want = (v.min(), v.max())
    assert lo == pytest.approx(want[0], rel=1e-3) and hi == pytest.approx(want[1], rel=1e-3), (
        f"{mode}: bounds {lo, hi} took NoData in (valid range {v.min()}..{v.max()})")
    # the valid ramp spans the full grey range
    inner = np.asarray(out)[10:, 10:]
    assert inner.min() <= 2 and inner.max() >= 253


def test_declared_nodata_is_honoured_in_render(synthetic_project):
    pt = synthetic_project
    a = np.linspace(100, 200, 400, dtype=np.float32).reshape(20, 20)
    a[:3] = -32768.0
    b = {}
    pt._compute_stretched_preview(a, _sp(), bounds_out=b, nodata_values=[-32768])
    assert float(np.asarray(b["lo"])) >= 100.0


def test_render_with_viewer_stretch_uses_the_files_own_nodata_tag(synthetic_project):
    """A tagged GDAL_NODATA (no .ax value) reaches the viewer's stretch."""
    pt = synthetic_project
    fp = fixture_image_path("nodata_fragmented_multiband")
    assert -9999.0 in pt._stretch_nodata_values(fp)


# ---------------------------------------------------------------------------
# Zoom tiles
# ---------------------------------------------------------------------------
def test_zoom_tile_turns_the_files_nodata_into_nan(synthetic_project, monkeypatch):
    """The tile is read raw; its -9999 holes must reach the render as NaN
    (white, never stretched as data), like the preview underneath."""
    import time
    from ..raster_reader import probe, open_reader
    pt = synthetic_project
    fp = fixture_image_path("nodata_fragmented_multiband")
    prof = probe(fp)
    reader = open_reader(fp, prof)
    disp = np.zeros((prof.height // 2, prof.width // 2, 3), np.float32)
    image_data = types.SimpleNamespace(
        filepath=fp, image=disp, reader=reader, profile=prof, preview_bands=[0, 1, 2],
        full_shape=(prof.height, prof.width, prof.count), display_scale=2.0)
    got = {}

    def _render(arr, viewer, *a, **k):
        got["arr"] = np.array(arr, copy=True)
        return np.zeros(arr.shape[:2] + (3,), np.uint8)
    monkeypatch.setattr(pt, "_render_with_viewer_stretch", _render)

    class _VP:
        def width(self): return 400
        def height(self): return 400

    class _Sig:
        def emit(self, *a):
            got["done"] = True
    viewer = types.SimpleNamespace(viewport=lambda: _VP(), image_data=image_data, highres_tile_ready=_Sig(),
                                   _clear_highres_overlay=lambda: None, _preview_stretch_bounds=None)
    rect = QtCore.QRectF(0, 0, disp.shape[1], disp.shape[0])
    pt._request_highres_viewport_region(viewer, image_data, rect, request_id=None, ax={})
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and "done" not in got:
        time.sleep(0.02)
    arr = got.get("arr")
    assert arr is not None, "no tile was rendered"
    assert not np.any(arr == -9999.0), "the tile still carries the -9999 fill"
    assert np.isnan(arr).any(), "the NoData holes vanished from the tile"


# ---------------------------------------------------------------------------
# Stretch bar and Stretch Viewer dialog
# ---------------------------------------------------------------------------
def _viewer_with(viewer_factory, arr, filepath=None):
    h, w = arr.shape[:2]
    viewer = viewer_factory()
    viewer.resize(420, 340)
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = types.SimpleNamespace(
        image=arr, channel_order="rgb", filepath=filepath, ax_config={},
        preview_bands=None, profile=None)
    return viewer


def test_stretch_bar_range_skips_dem_fill(viewer_factory):
    viewer = _viewer_with(viewer_factory, _dem())
    sb = viewer._stretchbar
    sb.refresh_range()
    v = _valid(_dem())
    assert sb._data_min == pytest.approx(float(v.min()), rel=1e-4)
    assert sb._data_max == pytest.approx(float(v.max()), rel=1e-4)


def test_stretch_bar_reads_the_raster_tag_without_a_project(viewer_factory):
    fp = fixture_image_path("nodata_fragmented_multiband")
    viewer = _viewer_with(viewer_factory, np.zeros((4, 4), np.float32), filepath=fp)
    assert -9999.0 in [float(x) for x in viewer._stretchbar._nodata_values()]


def test_stretch_bar_range_follows_a_replaced_array(viewer_factory):
    """Same file, new pixels (an edit): the bar's cached range is not reused."""
    viewer = _viewer_with(viewer_factory, np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10))
    viewer._stretchbar.refresh_range()
    assert viewer._stretchbar._data_max == pytest.approx(1.0)
    viewer.image_data = types.SimpleNamespace(
        image=np.linspace(0, 500, 100, dtype=np.float32).reshape(10, 10), channel_order="rgb",
        filepath=None, ax_config={}, preview_bands=None, profile=None)
    viewer._stretchbar.refresh_range()
    assert viewer._stretchbar._data_max == pytest.approx(500.0)


def test_stretch_dialog_range_skips_dem_fill(synthetic_project):
    dlg = ImageStretchDialog(synthetic_project, _dem())
    try:
        v = _valid(_dem())
        assert dlg._real_min == pytest.approx(float(v.min()), rel=1e-4)
        assert dlg._real_max == pytest.approx(float(v.max()), rel=1e-4)
    finally:
        dlg.close(); dlg.setParent(None); dlg.deleteLater()


def test_stretch_dialog_stays_compact_with_298_bands(synthetic_project, qapp):
    names = [f"AVIRIS-3 reflectance band {i + 1} ({380 + i * 7.4:.2f} nm) radiance-corrected"
             for i in range(298)]
    img = np.random.default_rng(0).random((40, 50, 3)).astype(np.float32)
    dlg = ImageStretchDialog(synthetic_project, img, band_names=names, preview_bands=[50, 30, 10],
                             reload_bands_cb=lambda *a, **k: None)
    try:
        dlg.show()
        qapp.processEvents()
        assert dlg.cb_band.count() == 298
        for combo in (dlg.cb_band, dlg.cb_r, dlg.cb_g, dlg.cb_b):
            assert combo.sizeHint().width() < 300, "a band combo grows with its longest name"
        assert dlg.width() <= ImageStretchDialog._MAX_W_PX
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        assert dlg.height() <= int(scr.height() * ImageStretchDialog._MAX_SCREEN_H) + 1
    finally:
        dlg.close(); dlg.setParent(None); dlg.deleteLater()


# ---------------------------------------------------------------------------
# Stretch to view
# ---------------------------------------------------------------------------
def _two_halves():
    a = np.empty((200, 200), np.float32)
    a[:, :100] = np.linspace(0, 100, 100, dtype=np.float32)[None, :]
    a[:, 100:] = np.linspace(1000, 2000, 100, dtype=np.float32)[None, :]
    return a


def test_no_view_button_at_the_bottom_left(viewer_factory):
    """The switch lives beside Auto only (Nathan asked for the bottom-left
    "View" square to go)."""
    viewer = viewer_factory()
    assert not hasattr(viewer, "_stretch_view_btn")
    btns = [w for w in viewer.findChildren(QtWidgets.QToolButton) if w.text() == "View"]
    assert not btns


def test_stretch_to_view_takes_the_statistics_on_screen(synthetic_project, viewer_factory, qapp):
    pt = synthetic_project
    arr = _two_halves()
    viewer = _viewer_with(viewer_factory, arr)
    viewer.stretch_params = _sp()
    viewer.fitInView(QtCore.QRectF(130, 70, 60, 60), QtCore.Qt.KeepAspectRatio)
    qapp.processEvents()

    viewer.stretch_to_view = False
    pt._render_with_viewer_stretch(arr, viewer, _as_array=True)
    lo_full = float(np.asarray(viewer._preview_stretch_bounds["lo"]))

    viewer.stretch_to_view = True
    win = ss.view_window(viewer, arr.shape)
    assert win is not None and win[0] >= 100, f"view window {win} is not the bright half"
    pt._render_with_viewer_stretch(arr, viewer, _as_array=True)
    lo_view = float(np.asarray(viewer._preview_stretch_bounds["lo"]))
    hi_view = float(np.asarray(viewer._preview_stretch_bounds["hi"]))
    assert lo_full == pytest.approx(0.0, abs=1e-3)
    assert lo_view >= 1000.0, "Stretch to view still used the whole image"
    want = ss.channel_bounds(arr, "percentile", 0, 100, window=win)
    assert (lo_view, hi_view) == (pytest.approx(want[0]), pytest.approx(want[1]))


def test_stretch_to_view_overrides_a_fixed_range(synthetic_project, viewer_factory, qapp):
    pt = synthetic_project
    arr = _two_halves()
    viewer = _viewer_with(viewer_factory, arr)
    viewer.stretch_params = _sp(mode="absolute", min_val=0.0, max_val=2000.0)
    viewer.fitInView(QtCore.QRectF(130, 70, 60, 60), QtCore.Qt.KeepAspectRatio)
    qapp.processEvents()
    viewer.stretch_to_view = True
    pt._render_with_viewer_stretch(arr, viewer, _as_array=True)
    assert float(np.asarray(viewer._preview_stretch_bounds["lo"])) >= 1000.0


def test_toggle_and_pan_run_the_view_stretch(viewer_factory, qapp):
    from PyQt5.QtTest import QTest
    arr = _two_halves()
    viewer = _viewer_with(viewer_factory, arr)
    calls = []
    viewer.set_view_stretch_callback(lambda v: calls.append(v.stretch_to_view))
    viewer._stretchbar._btn_view.click()
    assert viewer.stretch_to_view is True and calls == [True]
    key0 = viewer._view_key()
    viewer.setTransform(QtGui.QTransform.fromScale(4.0, 4.0))
    viewer.centerOn(30, 30)
    assert viewer._view_key() != key0, "the test did not move the view"
    for _ in range(60):                  # the re-stretch is debounced
        if len(calls) >= 2:
            break
        QTest.qWait(50)
    assert len(calls) == 2, "panning / zooming did not re-stretch the view"
    QTest.qWait(400)
    assert len(calls) == 2, "the view stretch re-ran without the view moving"
    viewer._stretchbar._btn_view.click()
    assert viewer.stretch_to_view is False and calls[-1] is False


def test_square_beside_auto_switches_stretch_to_view(viewer_factory):
    """The stretch bar's small square beside Auto switches Stretch to view and
    shows the viewer's state when it is switched from elsewhere."""
    arr = _two_halves()
    viewer = _viewer_with(viewer_factory, arr)
    calls = []
    viewer.set_view_stretch_callback(lambda v: calls.append(v.stretch_to_view))
    sb = viewer._stretchbar
    lay = sb.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert order.index(sb._btn_view) == order.index(sb._btn_auto) + 1, "the square is not beside Auto"
    sb._btn_view.click()
    assert viewer.stretch_to_view is True and calls == [True]
    viewer.set_stretch_to_view(False)
    assert viewer.stretch_to_view is False and not sb._btn_view.isChecked()


def test_stretch_bar_is_only_as_wide_as_its_content(viewer_factory):
    viewer = _viewer_with(viewer_factory, _two_halves())
    viewer.resize(1200, 600)
    sb = viewer._stretchbar
    sb.reposition()
    assert sb.width() <= sb.sizeHint().width()


# ---------------------------------------------------------------------------
# Each band keeps its stretch (band bar clicks)
# ---------------------------------------------------------------------------
@pytest.fixture
def band_viewer(synthetic_project, viewer_factory, tmp_path):
    """A real viewer showing a COPY of an 8-band fixture (its .ax is written)."""
    import shutil, os
    src = fixture_image_path("multiband_8band_ancillary")
    fp = str(tmp_path / f"bandmem_{tmp_path.name}.tif")
    shutil.copy(src, fp)
    pt = synthetic_project
    lite = pt._imagedata_or_fallback(fp)
    arr = lite.image
    h, w = arr.shape[:2]
    viewer = viewer_factory()
    viewer.resize(420, 340)
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer.image_data = lite
    yield pt, viewer, fp
    try:
        axp = pt._ax_path_for(fp)
        if os.path.exists(axp):
            os.remove(axp)
    except Exception:
        pass


def _set_by_hand(pt, viewer, lo, hi):
    """What releasing a stretch-bar slider does."""
    sp = viewer.stretch_params
    params = _StretchParams(mode="absolute", min_val=lo, max_val=hi, scope="viewer",
                            display_mode="single", display_band=sp.display_band)
    viewer.stretch_params = params
    pt._on_viewer_stretch_applied(params, viewer, "root_x")


def test_a_band_comes_back_with_the_stretch_it_was_left_with(band_viewer):
    pt, viewer, fp = band_viewer
    pt._on_band_bar_clicked(viewer, "root_x", 1)
    assert viewer.stretch_params.mode == "percentile"
    _set_by_hand(pt, viewer, 10.0, 20.0)

    pt._on_band_bar_clicked(viewer, "root_x", 2)
    assert viewer.stretch_params.mode == "percentile", "band 3 took band 2's stretch"

    pt._on_band_bar_clicked(viewer, "root_x", -1)          # composite
    pt._on_band_bar_clicked(viewer, "root_x", 1)
    sp = viewer.stretch_params
    assert (sp.mode, sp.min_val, sp.max_val) == ("absolute", 10.0, 20.0), (
        "band 2 lost the stretch the user left it with")
    assert str(sp.display_mode).lower() == "single"


def test_auto_forgets_the_bands_stretch(band_viewer):
    pt, viewer, fp = band_viewer
    pt._on_band_bar_clicked(viewer, "root_x", 1)
    _set_by_hand(pt, viewer, 10.0, 20.0)
    viewer.stretch_params = None                             # the bar's Auto
    pt._on_viewer_stretch_applied(None, viewer, "root_x")
    pt._on_band_bar_clicked(viewer, "root_x", 2)
    pt._on_band_bar_clicked(viewer, "root_x", 1)
    assert viewer.stretch_params.mode == "percentile", "Auto did not reset the band"


def test_reset_image_forgets_every_band(band_viewer):
    pt, viewer, fp = band_viewer
    pt._on_band_bar_clicked(viewer, "root_x", 1)
    _set_by_hand(pt, viewer, 10.0, 20.0)
    pt._clear_file_stretch(fp)
    assert pt._load_band_stretch(fp, "1") is None
