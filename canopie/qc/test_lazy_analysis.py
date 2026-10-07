"""
The windowed analysis image (canopie/lazy_analysis.py, 2026-10-07) gives the
pixels of the whole-image analysis replay -- for every `.ax` step it supports,
however the frame is cut into windows.

Large rasters (COGs) are read tile by tile; band math and booleans used to
force a whole-image read (17.9 GiB for the BCI COG -- a crash). The windowed
image computes crop, rotation, down-size, NoData per band, mask polygons, the
band expression and appended expressions per window. This pins it to the
full replay on real tiled TIFFs, both as one window and assembled from a grid
of odd-sized windows (window edges must not show).
"""
import numpy as np
import pytest
import tifffile

from ..analysis_replay import replay_for_analysis, _mask_polygon_records
from ..lazy_analysis import LazyAnalysis, supported
from ..raster_reader import probe, open_reader

pytestmark = [pytest.mark.extraction]

H0, W0 = 157, 203


def _raster(tmp_path, kind):
    rng = np.random.default_rng(11)
    if kind == "uint8":
        a = rng.integers(0, 256, size=(H0, W0, 4), dtype=np.uint8)
        a[20:30, 40:60, 0] = 0                     # NoData 0 in b1 only
        a[90:95, 10:200, 2] = 0                    # ... and in b3 only
    elif kind == "uint16":
        a = rng.integers(0, 4000, size=(H0, W0, 3), dtype=np.uint16)
        a[50:70, 50:70, :] = 65535
    else:
        a = rng.normal(100.0, 30.0, size=(H0, W0, 5)).astype(np.float32)
        a[:15, :, 4] = np.nan                      # native NaN in one band
        a[100:110, 120:150, 1] = -9999.0
    fp = str(tmp_path / f"{kind}.tif")
    tifffile.imwrite(fp, a, photometric="minisblack", planarconfig="contig", tile=(64, 64))
    return fp, a


CROP = {"crop_enabled": True, "crop_rect": {"x": 13, "y": 9, "width": 170, "height": 131},
        "crop_rect_ref_size": {"w": W0, "h": H0}}
CASES = {
    "none": {},
    "crop": dict(CROP),
    "rot90": {"rotate_enabled": True, "rotate": 90},
    "rot180_crop": {**CROP, "rotate_enabled": True, "rotate": 180},
    "crop_rot270": {**CROP, "rotate_enabled": True, "rotate": 270},
    "half": {"resize_enabled": True, "resize": {"width": 50, "height": 50}},
    "odd_scale_crop_rot90": {**CROP, "rotate_enabled": True, "rotate": 90,
                             "resize_enabled": True, "resize": {"width": 37, "height": 41}},
    "px_w": {"resize_enabled": True, "resize": {"px_w": 61}},
    "nodata": {"nodata_enabled": True, "nodata_values": [0, -9999, 65535]},
    "nodata_expr_half": {"nodata_enabled": True, "nodata_values": ["b1<40", 0, -9999],
                         "resize_enabled": True, "resize": {"width": 50, "height": 50}},
    "boolean": {"nodata_enabled": True, "nodata_values": [0, -9999], "band_enabled": True,
                "band_expression": "b1>100"},
    "band_math_appended_crop_rot": {**CROP, "rotate_enabled": True, "rotate": 90,
                                    "nodata_enabled": True, "nodata_values": [0, -9999],
                                    "band_enabled": True, "band_expression": "(b2-b1)/(b2+b1+1)",
                                    "appended_bands": [{"type": "boolean_expression", "expression": "b2>120", "index": 0},
                                                       {"type": "band_expression", "expression": "b3-b2", "index": 1}],
                                    "resize_enabled": True, "resize": {"width": 64, "height": 64}},
    "mask_polygon": {"band_enabled": True, "band_expression": "b1>100",
                     "mask_polygon": {"enabled": True,
                                      "points": [[10, 10], [60, 12], [55, 70], [12, 60]]}},
}


def _lazy(fp, ax):
    prof = probe(fp)
    vals = list(ax.get("nodata_values") or []) if ax.get("nodata_enabled", True) else []
    return LazyAnalysis(open_reader(fp, prof), prof, ax, filepath=fp, values=vals,
                        mask_records=_mask_polygon_records(None, ax, fp, None))


def _assert_same(got, want, msg):
    assert got.shape == want.shape, f"{msg}: {got.shape} vs {want.shape}"
    nan_g, nan_w = np.isnan(got), np.isnan(want)
    assert np.array_equal(nan_g, nan_w), (
        f"{msg}: NoData differs at {int((nan_g != nan_w).sum())} of {nan_w.size} values")
    for k in range(want.shape[2]):
        ok = ~nan_w[..., k]
        d = np.abs(got[..., k][ok] - want[..., k][ok])
        if not d.size:
            continue
        # float32 precision of this band's magnitude (one ulp at 65535 is 0.004)
        scale = max(1.0, float(np.abs(want[..., k][ok]).max()))
        assert d.max() <= 1.0 + 1e-4, f"{msg} band {k}: max |diff| {d.max()}"
        assert (d > 1e-6 * scale + 1e-4).mean() < 0.002, (
            f"{msg} band {k}: {(d > 1e-6 * scale + 1e-4).mean():.2%} of values differ "
            "(only integer rounding ties may)")


@pytest.mark.parametrize("kind", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_windowed_image_equals_the_full_replay(tmp_path, kind, case):
    fp, raw = _raster(tmp_path, kind)
    ax = CASES[case]
    assert supported(ax, W0, H0)
    want, layout = replay_for_analysis(None, raw, ax, filepath=fp, src_channel_order="rgb")
    want = np.asarray(want, dtype=np.float64)
    lz = _lazy(fp, ax)
    assert lz.layout.names() == layout.names()
    assert (lz.height, lz.width, len(lz)) == want.shape
    whole = np.dstack(lz.read_window(0, 0, lz.width, lz.height)).astype(np.float64)
    _assert_same(whole, want, f"{kind}/{case} one window")
    # assembled from odd-sized windows: window edges must not show
    tiled = np.full(want.shape, -12345.0)
    for y in range(0, lz.height, 23):
        for x in range(0, lz.width, 29):
            w = np.dstack(lz.read_window(x, y, x + 29, y + 23))
            tiled[y:y + w.shape[0], x:x + w.shape[1]] = w
    _assert_same(tiled, want, f"{kind}/{case} 23x29 windows")


def test_offset_bands_index_in_the_export_frame(tmp_path):
    fp, raw = _raster(tmp_path, "uint8")
    ax = CASES["boolean"]
    want, _ = replay_for_analysis(None, raw, ax, filepath=fp, src_channel_order="rgb")
    bands = _lazy(fp, ax).read_window_offset(40, 30, 90, 70)
    for k, b in enumerate(bands):
        assert b.shape == want.shape[:2]
        for (y, x) in [(30, 40), (69, 89), (50, 61)]:
            assert np.allclose(b[y, x], want[y, x, k], equal_nan=True)


def test_steps_needing_the_whole_image_are_not_windowed():
    assert not supported({"hist_enabled": True, "hist_match": {"mode": "meanstd", "ref_stats": []}}, W0, H0)
    assert supported({"hist_enabled": False, "hist_match": {"mode": "meanstd"}}, W0, H0)
    assert not supported({"resize_enabled": True, "resize": {"width": 150, "height": 150}}, W0, H0)
    assert not supported({"classification": {"mode": "sklearn", "enabled": True}}, W0, H0)
    assert not supported({"registration": {"enabled": True, "matrix": [[1, 0, 1], [0, 1, 0], [0, 0, 1]]}}, W0, H0)
    assert not supported({"appended_bands": [{"type": "classification"}]}, W0, H0)


def test_a_huge_full_band_is_refused_not_attempted(tmp_path, monkeypatch):
    from .. import lazy_analysis as la
    fp, _raw = _raster(tmp_path, "uint8")
    monkeypatch.setattr(la, "_FULL_MATERIALIZE_LIMIT_BYTES", 1000)
    with pytest.raises(MemoryError):
        _lazy(fp, {})[0]


@pytest.mark.parametrize("case", ["odd_scale_crop_rot90", "nodata_expr_half", "band_math_appended_crop_rot"])
def test_row_strips_give_the_same_pixels(tmp_path, monkeypatch, case):
    """A window is computed in row strips so the raw pixels decoded at once
    stay bounded (a 20x down-sized 139-band stack); strip edges must not show."""
    from .. import lazy_analysis as la
    fp, raw = _raster(tmp_path, "float32")
    ax = CASES[case]
    want, _ = replay_for_analysis(None, raw, ax, filepath=fp, src_channel_order="rgb")
    monkeypatch.setattr(la, "_STRIP_RAW_BYTES", 3000)
    lz = _lazy(fp, ax)
    assert lz._rows_per_strip(0, lz.width) < 5, "strips not engaged"
    got = np.dstack(lz.read_window(0, 0, lz.width, lz.height)).astype(np.float64)
    _assert_same(got, np.asarray(want, dtype=np.float64), f"{case} in strips")
