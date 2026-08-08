"""
QC regression tests for histogram matching under NoData.

Everything here was measured on a real 15-band AVIRIS prediction stack
(project `C:\\New Folder176`) whose `.ax` carried `nodata_values: []` while the
raster itself declares `GDAL_NODATA = -9999`. Three separate defects stacked up:

1. **The declared NoData was ignored.** Statistics were taken from
   `ax["nodata_values"]` alone, so the -9999 fill counted as real data in BOTH
   the reference and the source. Measured: band 11's stored reference mean was
   -9356 for a sensor-azimuth band whose true valid mean is 55.4, and matching
   moved the real data to -1362.

2. **The fill sentinel was rescaled.** Because the fill was not masked, the
   transform rewrote it: -9999 -> -10110.4. 172470 px per band then matched no
   NoData rule at all and silently became "valid data" for CSV export, ML
   extraction, Inspect and scene stats.

3. **The mask was a whole-pixel OR across every channel.** Two ancillary bands
   are 100% fill, so the OR was True for 100.00% of the image -- and since the
   same mask drives the "restore original pixels" step, fixing (1) alone turned
   histogram matching into a complete no-op for the whole file. Statistics are
   per band, so the mask has to be per band too (which is what
   `process_polygon` and `_per_band_nodata_masks` already do).

Science bands that mark fill as NaN were never affected, because numpy's
nan-aware reductions skip NaN for free -- which is exactly why only the
sentinel-based bands looked broken and the problem went unnoticed.
"""
import numpy as np
import pytest

from ..project_tab import ProjectTab, hist_nodata_values, declared_file_nodata
from ..image_editor_dialog import contaminated_reference_bands

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor, pytest.mark.extraction]

FILL = -9999.0


def _stack_with_mixed_fill(h=40, w=40, bands=6):
    """Mirrors the real stack's shape of the problem:
      band 0,1 : real data, NaN fill in a corner
      band 2   : real data, -9999 fill over most of the frame
      band 3   : 100% -9999  (the band that made the whole-pixel OR total)
      band 4,5 : real data, no fill
    """
    rng = np.random.default_rng(7)
    a = rng.uniform(10.0, 20.0, (h, w, bands)).astype(np.float32)
    a[:5, :5, 0] = np.nan
    a[:5, :5, 1] = np.nan
    a[10:, :, 2] = FILL              # majority fill, some real data left
    a[:, :, 3] = FILL                # entirely fill
    return a


REF = {"mode": "meanstd", "bands": 6,
       "ref_stats": [{"mean": 100.0, "std": 5.0}] * 6}


def _match(arr, nodata_values):
    return ProjectTab._apply_hist_match(
        arr, {"hist_match": REF}, nodata_values=nodata_values)


# ---------------------------------------------------------------------------
# The fill sentinel must survive
# ---------------------------------------------------------------------------
def test_fill_value_is_not_rescaled():
    """THE data-corruption regression: after matching, pixels that were the
    NoData sentinel must STILL be the sentinel, or every downstream NoData
    check silently stops working."""
    arr = _stack_with_mixed_fill()
    before = int(np.isclose(arr[..., 2], FILL, atol=1e-3).sum())
    out = _match(arr, [FILL])
    after = int(np.isclose(out[..., 2], FILL, atol=1e-3).sum())

    assert before > 0, "fixture has no fill pixels to test"
    assert after == before, (
        f"{before - after} fill pixels were rewritten by histogram matching; "
        f"they no longer match the -9999 sentinel and read as valid data")


def test_nan_fill_is_preserved():
    arr = _stack_with_mixed_fill()
    out = _match(arr, [FILL])
    assert not np.isfinite(out[:5, :5, 0]).any(), "NaN fill did not survive matching"


# ---------------------------------------------------------------------------
# Per-band masking
# ---------------------------------------------------------------------------
def test_an_all_fill_band_does_not_disable_matching_for_every_other_band():
    """THE no-op regression. The mask was a whole-pixel OR, so a single band
    that is 100% fill made it True everywhere -- and the restore step then put
    the original values back for the entire image."""
    arr = _stack_with_mixed_fill()
    out = _match(arr, [FILL])

    assert not np.allclose(out[..., 4], arr[..., 4]), (
        "band 5 was left untouched: a 100%-fill band is still cancelling the "
        "whole transform")


def test_valid_pixels_land_on_the_reference_statistics():
    """The numbers, not just 'something changed'."""
    arr = _stack_with_mixed_fill()
    out = _match(arr, [FILL])

    for b in (4, 5):
        v = out[..., b]
        v = v[np.isfinite(v)]
        assert v.mean() == pytest.approx(100.0, abs=1.0), f"band {b+1} mean {v.mean()}"
        assert v.std() == pytest.approx(5.0, abs=1.0), f"band {b+1} std {v.std()}"


def test_partially_filled_band_uses_only_its_valid_pixels():
    """Band 3 is mostly fill. Its statistics must come from the remaining real
    pixels, so those land on the reference rather than being dragged toward
    -9999."""
    arr = _stack_with_mixed_fill()
    out = _match(arr, [FILL])

    valid = ~np.isclose(arr[..., 2], FILL, atol=1e-3)
    got = out[..., 2][valid]
    assert got.mean() == pytest.approx(100.0, abs=1.5), (
        f"valid pixels of a mostly-filled band landed at {got.mean():.2f}, not 100")


def test_all_fill_band_is_left_alone():
    arr = _stack_with_mixed_fill()
    out = _match(arr, [FILL])
    assert np.allclose(out[..., 3], FILL, atol=1e-3), (
        "a band with no valid pixels should be passed through untouched")


def test_without_nodata_the_corruption_is_reproduced():
    """Guards the tests above from passing vacuously: with NoData NOT supplied,
    the old behaviour (sentinel rewritten) must still be observable."""
    arr = _stack_with_mixed_fill()
    out = _match(arr, [])          # no NoData -> fill counted as data

    after = int(np.isclose(out[..., 2], FILL, atol=1e-3).sum())
    before = int(np.isclose(arr[..., 2], FILL, atol=1e-3).sum())
    assert after < before, (
        "fixture no longer demonstrates the problem -- with NoData ignored the "
        "sentinel should be rewritten")


# ---------------------------------------------------------------------------
# The .ax / declared-NoData union
# ---------------------------------------------------------------------------
def test_declared_nodata_is_unioned_with_the_ax_list(tmp_path):
    """`nodata_values: []` in the .ax must not mean "no NoData" when the raster
    declares its own -- that is the exact configuration of the real project."""
    import tifffile
    fp = tmp_path / "declared.tif"
    tifffile.imwrite(str(fp), np.zeros((16, 16), dtype=np.float32),
                     extratags=[(42113, 's', 0, "-9999", True)])   # GDAL_NODATA

    assert declared_file_nodata(str(fp)) == [FILL], (
        "the raster's own GDAL_NODATA tag was not read")
    assert hist_nodata_values({"nodata_values": []}, str(fp)) == [FILL]


def test_explicit_values_are_kept_alongside_the_declared_one(tmp_path):
    import tifffile
    fp = tmp_path / "both.tif"
    tifffile.imwrite(str(fp), np.zeros((16, 16), dtype=np.float32),
                     extratags=[(42113, 's', 0, "-9999", True)])

    got = hist_nodata_values({"nodata_values": [-1]}, str(fp))
    assert -1 in got and FILL in got, got


def test_disabled_nodata_is_respected(tmp_path):
    import tifffile
    fp = tmp_path / "off.tif"
    tifffile.imwrite(str(fp), np.zeros((16, 16), dtype=np.float32),
                     extratags=[(42113, 's', 0, "-9999", True)])

    assert hist_nodata_values({"nodata_enabled": False}, str(fp)) == []


# ---------------------------------------------------------------------------
# The WIRING: the replay paths must actually use the union
#
# Testing `hist_nodata_values` as a pure function and `_apply_hist_match` with
# an explicit list does NOT cover the thing that was broken -- which was that
# the replay paths never asked for the declared value in the first place. These
# go end to end through the real `.ax` replay, with `nodata_values: []` exactly
# as the real project has it.
# ---------------------------------------------------------------------------
def _declared_nodata_raster(tmp_path, name="stack.tif"):
    """A float raster that declares GDAL_NODATA = -9999, with one band 100%
    fill and one band partially filled -- the real stack's shape."""
    import tifffile
    arr = _stack_with_mixed_fill()
    fp = tmp_path / name
    tifffile.imwrite(str(fp), arr, extratags=[(42113, 's', 0, "-9999", True)])
    return str(fp), arr


AX_NO_NODATA = {
    "nodata_values": [],          # exactly what the real project's .ax carries
    "nodata_enabled": True,
    "hist_enabled": True,
    "crop_enabled": True, "rotate_enabled": True, "resize_enabled": True,
    "band_enabled": True,
    "hist_match": REF,
}


def test_ax_replay_excludes_the_declared_nodata(synthetic_project, tmp_path):
    """THE wiring regression for the export/CSV path: `_apply_ax_to_raw` must
    obtain the declared NoData itself, because the .ax lists none."""
    fp, arr = _declared_nodata_raster(tmp_path)

    out, _C = synthetic_project._apply_ax_to_raw(arr, dict(AX_NO_NODATA), filepath=fp)

    before = int(np.isclose(arr[..., 2], FILL, atol=1e-3).sum())
    after = int(np.isclose(np.asarray(out)[..., 2], FILL, atol=1e-3).sum())
    assert after == before, (
        f"{before - after} fill pixels were rewritten: the replay path is not "
        "honouring the raster's declared NoData")


def test_ax_replay_still_normalises_the_real_pixels(synthetic_project, tmp_path):
    """Excluding NoData must not turn matching into a no-op (which is what the
    whole-pixel mask did once the fill was finally excluded)."""
    fp, arr = _declared_nodata_raster(tmp_path)

    out = np.asarray(synthetic_project._apply_ax_to_raw(
        arr, dict(AX_NO_NODATA), filepath=fp)[0])

    v = out[..., 4][np.isfinite(out[..., 4])]
    assert v.mean() == pytest.approx(100.0, abs=1.5), (
        f"band 5 landed at {v.mean():.2f}, not the reference mean 100 -- "
        "matching became a no-op")


def test_viewer_replay_excludes_the_declared_nodata(synthetic_project, tmp_path):
    """Same wiring, the other replay path (`apply_aux_modifications`, used by
    the viewer's loader). Both must agree or the viewer and CSV disagree."""
    import json
    fp, arr = _declared_nodata_raster(tmp_path, "viewer.tif")
    with open(tmp_path / "viewer.ax", "w", encoding="utf-8") as f:
        json.dump(AX_NO_NODATA, f)

    out = np.asarray(synthetic_project.__class__.apply_aux_modifications(
        fp, arr, project_folder=str(tmp_path), global_mode=False))

    before = int(np.isclose(arr[..., 2], FILL, atol=1e-3).sum())
    after = int(np.isclose(out[..., 2], FILL, atol=1e-3).sum())
    assert after == before, (
        f"{before - after} fill pixels rewritten by the viewer replay path")


# ---------------------------------------------------------------------------
# Stale, fill-contaminated stored references
# ---------------------------------------------------------------------------
def test_contaminated_reference_is_detected():
    """Both shapes seen in the real .ax: a 100%-fill band stored -9988 (0.1%
    from the sentinel) and the partially-filled angle bands stored -9356 (6%
    away) -- an exact-match tolerance would miss the second, which is the one
    that actually corrupts real data."""
    hm = {"ref_stats": [{"mean": -9988.5}, {"mean": -9356.4},
                        {"mean": 0.2877}, {"mean": 54.17}]}
    bad = {i for i, _mu, _nv in contaminated_reference_bands(hm, [FILL])}

    assert bad == {0, 1}, f"expected bands 1-2 flagged, got {sorted(bad)}"


def test_clean_reference_is_not_flagged():
    """Must not cry wolf on a correctly computed reference."""
    hm = {"ref_stats": [{"mean": 0.2912}, {"mean": 54.17}, {"mean": 211.4},
                        {"mean": 0.827}, {"mean": -1.5}]}
    assert contaminated_reference_bands(hm, [FILL]) == []


def test_no_nodata_means_nothing_to_flag():
    hm = {"ref_stats": [{"mean": -9988.5}]}
    assert contaminated_reference_bands(hm, []) == []


# ---------------------------------------------------------------------------
# The plot sampler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bands", [1, 3, 4, 15, 200])
def test_stats_sampler_handles_any_band_count(bands):
    """`_sample_for_stats` was defined TWICE in ImageEditorDialog; the later
    definition (bare cv2.resize) won and OpenCV asserts `cn <= 4`, so the
    histogram plot raised on every >4-band stack and rendered an error tab."""
    import types
    from ..image_editor_dialog import ImageEditorDialog as IED

    stub = types.SimpleNamespace(STRETCH_SAMPLE_MAX=IED.STRETCH_SAMPLE_MAX)
    fn = IED._sample_for_stats.__get__(stub, IED)

    arr = (np.random.rand(600, 700, bands) if bands > 1
           else np.random.rand(600, 700)).astype(np.float32)
    out = fn(arr)          # must not raise

    assert out.shape[:2] == (max(1, round(600 * IED.STRETCH_SAMPLE_MAX / 700)),
                             IED.STRETCH_SAMPLE_MAX)
    if bands > 1:
        assert out.shape[2] == bands, "band count must survive the downsample"


def test_only_one_sample_for_stats_definition():
    """A second `def` silently replaced the first. Pin the count so the
    shadowing cannot come back."""
    import inspect
    from .. import image_editor_dialog as ied

    src = inspect.getsource(ied)
    assert src.count("    def _sample_for_stats(self, arr):") == 1, (
        "_sample_for_stats is defined more than once; the LAST definition wins "
        "and the earlier, correct one becomes dead code")
