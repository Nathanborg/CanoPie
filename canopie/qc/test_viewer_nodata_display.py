"""
The viewer side of the one NoData rule (canopie/nodata.py, Nathan 2026-10-06).

The viewer's DATA holds NoData as NaN per band, exactly where the analysis
image (CSV / ML export) does -- see test_contract_ax_consumers for that
contract under every .ax op. This file pins the two things around it:

  * the DISPLAY paints a pixel white when a band it SHOWS is NoData, and only
    then (it used to build one mask over every band of the image, so NoData in
    a band that was not on screen whitened the pixel);
  * a raster that declares NoData only in its own GDAL_NODATA tag, with no
    .ax at all, gets the same NaN in the viewer as in the export.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture

pytestmark = [pytest.mark.extraction, pytest.mark.viewer]


def _params(**kw):
    from ..project_tab import _StretchParams
    base = dict(mode="percentile", low_p=0.5, high_p=99.5, k_sigma=1.0,
                min_val=None, max_val=None, per_channel=True, clip=True,
                scope="viewer", display_mode="auto")
    base.update(kw)
    return _StretchParams(**base)


def _stack():
    rng = np.random.default_rng(7)
    img = rng.uniform(10, 200, size=(40, 50, 5)).astype(np.float32)
    img[5, 5, 0] = np.nan          # a displayed band (b1) is NoData here
    img[9, 9, 4] = np.nan          # only a band NOT on screen (b5) is NoData here
    return img


def test_display_nodata_helper_uses_only_the_bands_it_is_given():
    from ..project_tab import _with_display_nodata
    img = _stack()
    m = _with_display_nodata(img[..., :3])
    assert m is not None and m[5, 5] and not m[9, 9]
    assert int(m.sum()) == 1
    # an integer stack has no NaN: nothing to add
    assert _with_display_nodata(np.zeros((4, 4, 3), np.uint8)) is None
    # a mask covering every pixel (an all-NaN band on screen) is dropped
    allnan = np.full((4, 4), np.nan, np.float32)
    assert _with_display_nodata(allnan) is None
    # an incoming mask is kept and OR-ed
    given = np.zeros((40, 50), bool)
    given[1, 1] = True
    m2 = _with_display_nodata(img[..., :3], given)
    assert m2[1, 1] and m2[5, 5] and not m2[9, 9]


@pytest.mark.parametrize("engine", ["unified", "legacy"])
def test_display_whitens_only_where_a_shown_band_is_nodata(synthetic_project, engine):
    pt = synthetic_project
    img = _stack()
    if engine == "unified":
        out = pt._compute_stretched_preview(img, _params())
    else:
        out = pt._apply_viewer_stretch_numpy(img[..., :3], _params())
    out = np.asarray(out)
    assert out.shape[:2] == (40, 50)
    assert np.all(out[5, 5] == 255), f"{engine}: NoData in a shown band must be white"
    assert not np.all(out[9, 9] == 255), (
        f"{engine}: NoData in a band that is NOT shown must not whiten the pixel")

    # showing b5 alone flips it: (9, 9) is white, (5, 5) is not
    if engine == "unified":
        out1 = np.asarray(pt._compute_stretched_preview(img, _params(display_mode="single", display_band=4)))
        assert out1[9, 9] == 255 and out1[5, 5] != 255


@pytest.mark.parametrize("name", ["multiband_8band_ancillary", "nodata_fragmented_multiband"])
def test_tag_only_nodata_reaches_the_viewer_without_an_ax(synthetic_project, name):
    """GDAL_NODATA tag, no .ax: viewer data NaN exactly where the export's is."""
    import os
    pt = synthetic_project
    fp = fixture_image_path(name)
    assert get_fixture(name)["nodata_value"] is not None
    assert not os.path.exists(pt._ax_path_for(fp)), "precondition: no .ax for this fixture"
    lite = pt._imagedata_or_fallback(fp)
    order = str(getattr(lite, "channel_order", "bgr") or "bgr").lower()
    viewer = np.asarray(pt.__class__.apply_aux_modifications(
        fp, lite.image, pt.project_folder, global_mode=False,
        src_channel_order=order), dtype=np.float64)
    if order != "rgb" and viewer.shape[2] >= 3:
        viewer = viewer.copy()
        viewer[..., :3] = viewer[..., 2::-1]
    img, _c = pt._get_export_image(fp)
    chans = pt._channels_in_export_order(img)
    export = np.stack([np.asarray(c, dtype=np.float64) for c in chans[:viewer.shape[2]]], axis=-1)
    assert np.isnan(export).any(), "fixture has no NoData -- vacuous"
    assert np.array_equal(np.isnan(viewer), np.isnan(export))
    ok = ~np.isnan(export)
    assert np.allclose(viewer[ok], export[ok], atol=1e-4)
    assert not np.any(np.isclose(viewer[ok], get_fixture(name)["nodata_value"])), \
        "the fill value is still in the viewer data"
