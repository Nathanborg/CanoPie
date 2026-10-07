"""
Navigation stays responsive (QGIS-style: never read or draw more than the
screen can show) -- the contracts behind the 2026-10-05 navigation fix.

Measured on C:\\Nova pasta (6) (BCI mosaic: 28315x25548 uncompressed, one row
per strip, NO overviews) before the fix, with real wheel/drag input: 6.9 s of
GUI stalls (longest 1.5 s) for 8 zoom notches, 3.1 s for a 12-step pan, sharp
detail ~3 s after input. Causes, each pinned here:

  * every zoom above 1.05x fit asked for a "sharp" viewport tile, although the
    1/3-resolution preview already had more pixels than the screen up to zoom
    1.0 -- and with no overviews that tile was read, stretched and turned into
    a pixmap at FULL resolution (8,900 x 5,500 for a 1,600 x 1,000 viewport);
  * the polygon LOD refiner swapped crowns for sub-2-pixel gains, reading one
    sidecar per crown in view on 8 threads that starved the GUI thread;
  * per-vertex Python QPointF loops building outlines (tiles, refine swaps).

After: 0.65 s / 0.09 s of stalls; sharp detail ~0.2 s after input.
"""
import math

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .fixtures_manifest import FIXTURES, fixture_image_path

pytestmark = [pytest.mark.perf, pytest.mark.viewer]


# --------------------------------------------------------------------------
# Strided reads: a raster without overviews can be read at screen resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", [f["name"] for f in FIXTURES])
@pytest.mark.parametrize("step", [2, 3])
def test_strided_window_equals_sliced_full_read(name, step):
    from ..raster_reader import probe, open_reader
    path = fixture_image_path(name)
    prof = probe(path)
    if prof is None or not getattr(prof, "is_windowable", True):
        pytest.skip("not readable through raster_reader")
    r = open_reader(path, prof)
    if r is None:
        pytest.skip("no reader")
    lv = r._levels[0]
    w, h = lv["width"], lv["height"]
    bands = list(range(min(3, lv["count"])))
    x0, y0, x1, y1 = 1, 2, max(3, w - 1), max(4, h - 3)
    full = r.read_window(x0, y0, x1, y1, bands=bands)[::step, ::step]
    got = r.read_window(x0, y0, x1, y1, bands=bands, step=step)
    assert got.shape == full.shape and np.array_equal(got, full), (
        f"{name} ({prof.strategy}): strided read differs from read_window()[::{step}, ::{step}]")


@pytest.mark.parametrize("planar", ["contig", "separate"])
@pytest.mark.parametrize("rows_per_strip", [1, 7])
def test_strided_read_on_strips(tmp_path, planar, rows_per_strip):
    """The BCI mosaic's layout: strips (one row each), read through the
    segment path, where the strided read must decode only the sampled strips."""
    import tifffile
    from ..raster_reader import probe, open_reader
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 60000, size=(97, 131, 4), dtype=np.uint16)
    path = str(tmp_path / f"strips_{planar}_{rows_per_strip}.tif")
    data = arr if planar == "contig" else np.moveaxis(arr, -1, 0)
    tifffile.imwrite(path, data, compression="zlib", rowsperstrip=rows_per_strip,
                     planarconfig=planar, photometric="rgb" if planar == "contig" else "minisblack",
                     extrasamples=[2] if planar == "contig" else None)
    prof = probe(path)
    assert prof is not None and prof.strategy == "strips", f"fixture is {getattr(prof, 'strategy', None)}"
    r = open_reader(path, prof)
    for step in (2, 3, 5):
        full = r.read_window(5, 3, 120, 90, bands=[0, 2, 3])[::step, ::step]
        got = r.read_window(5, 3, 120, 90, bands=[0, 2, 3], step=step)
        assert np.array_equal(got, full), f"strips/{planar}/rps={rows_per_strip} step {step}"


# --------------------------------------------------------------------------
# No sharp tile while the preview already out-resolves the screen
# --------------------------------------------------------------------------
def test_no_sharp_tile_request_until_the_preview_is_visibly_coarse(viewer_factory, qapp):
    v = viewer_factory()
    v.resize(800, 600)
    v.show()
    qimg = QtGui.QImage(4000, 3000, QtGui.QImage.Format_RGB32)
    qimg.fill(0x336699)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    qapp.processEvents()
    calls = []
    v.enable_highres_viewport(lambda viewer, rect, req=None: calls.append(v.transform().m11()))
    try:
        for scale, expect in ((0.5, False), (1.0, False), (2.0, True)):
            v.resetTransform()
            v.scale(scale, scale)
            calls.clear()
            v._on_highres_timer_timeout()
            assert bool(calls) == expect, (
                f"at {scale} device px per preview px a sharp tile was "
                f"{'requested' if calls else 'not requested'}: below ~1.1 the preview already has "
                "more detail than the screen, and on a raster without overviews each request read, "
                "stretched and pixmapped the whole viewport at full resolution (1.5 s stalls).")
    finally:
        v.disable_highres_viewport()


# --------------------------------------------------------------------------
# numpy <-> QPolygonF helpers are exact
# --------------------------------------------------------------------------
def test_qpolygonf_helpers_are_exact():
    from ..image_viewer import (qpolygonf_from_xy, xy_from_qpolygonf,
                                _qpolygonf_from_xy_slow, _xy_from_qpolygonf_slow)
    xy = np.random.default_rng(1).random((777, 2)) * 50000 - 1000
    fast, slow = qpolygonf_from_xy(xy), _qpolygonf_from_xy_slow(xy)
    assert fast.size() == slow.size() == 777
    assert all(fast.at(i) == slow.at(i) for i in range(777))
    assert np.array_equal(xy_from_qpolygonf(fast), _xy_from_qpolygonf_slow(slow))
    assert qpolygonf_from_xy(np.empty((0, 2))).size() == 0


# --------------------------------------------------------------------------
# The LOD refiner leaves crowns alone when the error is not visible
# --------------------------------------------------------------------------
def test_refiner_does_not_swap_for_an_invisible_gain(synthetic_project, viewer_factory, monkeypatch):
    """At 1.6 device px of overview error (below the 2 px threshold) no crown
    is swapped and no sidecar read is queued; at 4 px it refines."""
    from .. import polygon_lod
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    imgd = pt._imagedata_or_fallback(fp)
    v = viewer_factory()
    v.resize(400, 400)
    v.show()
    v.image_data = imgd
    v.set_image(pt.convert_cv_to_pixmap(imgd.image))
    rec = {"viewer": v, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec]
    monkeypatch.setattr(pt, "get_root_by_filepath", lambda f: None)
    ring = [(128 + 100 * math.cos(2 * math.pi * k / 600), 128 + 100 * math.sin(2 * math.pi * k / 600))
            for k in range(600)]
    group = "navspeed_crown"
    pt.all_polygons[group] = {fp: {'name': 'c', 'root': '', 'type': 'polygon', 'coord_space': 'image',
                                   'image_ref_size': {'w': 256, 'h': 256}, 'points': ring}}
    pt._poly_norm_index_invalid = True
    try:
        v.clear_polygons()
        pt.load_polygons(v, imgd)
        it = next(i for i in v.get_all_polygons() if i.name == group)
        n0 = it.polygon.count()
        ov = polygon_lod.OVERVIEW_LEVEL
        # device px per record unit = view scale * 64/256
        for scale, should_swap in ((0.4, False), (1.0, True)):     # error 1.6 px, then 4 px
            v.resetTransform()
            v.scale(scale, scale)
            v.centerOn(it)
            err = (2 ** ov) * scale * (64 / 256)
            before = it.polygon.count()
            pt._refine_visible_polygon_lod(v)
            changed = it.polygon.count() != before
            assert changed == should_swap, (
                f"overview error {err:.1f} device px: crown {'was' if changed else 'was not'} swapped "
                f"({before} -> {it.polygon.count()} vertices); refine only when > 2 px.")
        assert it.polygon.count() > n0
    finally:
        try:
            pt.viewer_widgets.remove(rec)
        except ValueError:
            pass
        pt.all_polygons.pop(group, None)
        pt._poly_norm_index_invalid = True
