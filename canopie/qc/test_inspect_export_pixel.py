"""
Inspect in the app reports what the CSV / ML export read at the clicked
EXPORT pixel (ProjectTab.inspect_export_pixel, 2026-10-06):

  * full resolution: on a decimated preview it used to report the preview's
    (coarser) pixel -- a different value from the one the CSV writes there;
  * named like the CSV columns (R, G, B, band_4 ..., the expression text),
    with no second re-evaluated "index" column;
  * NoData per band (canopie.nodata): a `b1<40` rule flags b1 and the bands
    reading it, never the whole readout; a raster's own GDAL_NODATA tag counts
    on the lazy read path too.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .fixtures_manifest import fixture_image_path, get_fixture

pytestmark = [pytest.mark.viewer, pytest.mark.extraction]


def _clear(pt):
    for attr in ("_export_image_cache", "_export_cache", "_per_band_nd_cache",
                 "_master_nd_cache", "_export_frame_hw_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def _viewer(pt, viewer_factory, lite, data):
    """A viewer INSIDE the project (so Inspect finds its ProjectTab), showing
    `data` as its image data with a pixmap of the same size."""
    v = viewer_factory()
    v.setParent(pt)
    lite.image = data
    h, w = data.shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    v.image_data = lite
    return v


def _inspect(v, x, y):
    got = []
    v.pixel_clicked.connect(lambda p, payload: got.append(payload))
    ok = v._inspect_at_scene_point(QtCore.QPointF(x + 0.5, y + 0.5))
    return ok, got[-1] if got else None


def _export(pt, fp):
    img, info = pt._get_export_image(fp)
    return np.asarray(img, dtype=np.float64), info


@pytest.fixture
def ax_on(synthetic_project):
    """Write a temporary .ax for a no-.ax fixture of the shared project."""
    pt = synthetic_project
    written = []

    def _make(name, ax):
        fp = fixture_image_path(name)
        axp = pt._ax_path_for(fp)
        assert not os.path.exists(axp) or axp in written, "precondition: fixture has no .ax"
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(ax, f)
        written.append(axp)
        _clear(pt)
        return fp

    yield _make
    for axp in written:
        if os.path.exists(axp):
            os.remove(axp)
    _clear(pt)


def test_inspect_names_and_per_band_nodata_like_the_csv(synthetic_project, viewer_factory, ax_on):
    pt = synthetic_project
    fp = ax_on("rgb_8bit_untiled", {"nodata_enabled": True, "nodata_values": ["b1<40"],
                                    "band_enabled": True, "band_expression": "b1>100"})
    lite = pt._imagedata_or_fallback(fp)
    data = pt.__class__.apply_aux_modifications(fp, lite.image, pt.project_folder,
                                                src_channel_order=lite.channel_order)
    export, info = _export(pt, fp)
    v = _viewer(pt, viewer_factory, lite, data)
    ys, xs = np.nonzero(np.isnan(export[..., 0]))
    assert ys.size, "fixture has no b1<40 pixel -- vacuous"
    for (y, x) in [(int(ys[0]), int(xs[0])), (export.shape[0] // 2, export.shape[1] // 2)]:
        ok, payload = _inspect(v, x, y)
        assert ok and payload is not None
        assert payload["names"] == ["R", "G", "B", "b1>100"], payload["names"]
        assert payload["is_nodata"] is False
        assert payload["channel_nodata"] == [bool(np.isnan(t)) for t in export[y, x]]
        got = np.array(payload["values"], dtype=np.float64)
        assert np.allclose(got, export[y, x], equal_nan=True), (got, export[y, x])
        assert tuple(payload["pixel"]) == (x, y)
    # at the b1<40 pixel: b1 and the band reading it are NoData, G and B are not
    _ok, payload = _inspect(v, int(xs[0]), int(ys[0]))
    assert payload["channel_nodata"] == [True, False, False, True]


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_inspect_on_a_decimated_preview_reads_the_exported_pixel(
        synthetic_project, viewer_factory, monkeypatch, lazy):
    """The preview holds every 2nd pixel; Inspect must still report the
    exported pixel under the click, not the preview's."""
    from .. import project_tab as project_tab_module
    pt = synthetic_project
    name = "multiband_8band_ancillary"           # band 8: GDAL_NODATA tag fill
    fp = fixture_image_path(name)
    _clear(pt)
    if lazy:
        monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    try:
        lite = pt._imagedata_or_fallback(fp)
        full = np.asarray(lite.image)
        preview = np.ascontiguousarray(full[::2, ::2])
        export, info = _export(pt, fp) if not lazy else (None, None)
        if lazy:
            img, info = pt._get_export_image(fp)
            assert info.get("lazy"), "lazy path not engaged"
            export = np.dstack([np.asarray(c, dtype=np.float64) for c in img.read_window(0, 0, img.width, img.height)])
            export[..., 7] = np.nan                       # the tag's fill, per band
        v = _viewer(pt, viewer_factory, lite, preview)
        hit_diff = False
        for (i, j) in [(3, 4), (10, 7), (15, 15)]:
            ok, payload = _inspect(v, i, j)
            assert ok and payload is not None
            ex, ey = payload["pixel"]
            assert (ex, ey) == (2 * i + 1, 2 * j + 1), payload["pixel"]
            got = np.array(payload["values"], dtype=np.float64)
            assert np.allclose(got, export[ey, ex, :got.size], equal_nan=True), (got, export[ey, ex])
            assert payload["channel_nodata"][7] is True and not any(payload["channel_nodata"][:7])
            assert payload["names"][:4] == ["R", "G", "B", "band_4"]
            hit_diff |= not np.allclose(got[:3], preview[j, i, :3])
        assert hit_diff, "preview and export agree at every probe -- the test is vacuous"
    finally:
        monkeypatch.undo()
        _clear(pt)
