"""
QC regression tests for Scene Mean / Scene Median / Scene Standard Deviation
under NoData, on the LAZY/windowed read path.

THE BUG THIS PINS (reported from a real 15-band prediction stack):

    The exported "Scene Mean" / "Scene Median" columns reported the NoData
    FILL VALUE ITSELF on any raster large enough to take the lazy path --
    e.g. Scene Median exactly -9999.0 and Scene Mean ~-9365 on the ancillary
    angle bands -- while the polygon statistics on the very same CSV row were
    correctly masked. Scene stats are meant to be the whole-image context for
    a polygon's values, so a -9999 there is not just wrong, it silently
    poisons any downstream normalisation against the scene.

ROOT CAUSE: `_band_scene_stats` masked using `nd_masks[bi]`, but `nd_masks`
is only ever built on the EAGER path -- both of its construction branches in
`process_polygon` are guarded by `not _chans_are_lazy`. On the lazy path
NoData is applied per-ROI instead (`lazy_nd_vals` / `_build_roi_nodata`), so
`nd_masks` stays None, `_scene_stats_for_band` received `band_mask=None`, and
the raw band -- fill values included -- went straight into nanmean/nanmedian.

The fix builds the mask from the band that scene stats already read, so it
costs no extra I/O: `chans[bi]` is a full-band read either way.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_raw_npz, assert_close

SCENE_OPTS = {"stats": {"mean": True, "scene_mean": True,
                        "scene_median": True, "scene_std": True}}


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


def _scene_rows(project, name, force_lazy_marker=None):
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = project.all_polygons[group][fp]
    rows, _ = project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=SCENE_OPTS)
    return _rows_by_channel(rows)


def test_scene_stats_exclude_fill_value_on_lazy_path(
        synthetic_project, force_lazy_export):
    """THE regression: the 100%-fill ancillary band aside, no band's scene
    stats may equal (or be dragged toward) the -9999 fill value."""
    name = "nodata_fragmented_multiband"    # -9999 holes in EVERY band
    by_channel = _scene_rows(synthetic_project, name)

    assert by_channel, "no rows produced"
    for ch, row in by_channel.items():
        for col in ("Scene Mean", "Scene Median"):
            val = row.get(col)
            if val is None or not np.isfinite(val):
                continue  # an all-fill band legitimately has no scene value
            assert val > -9000.0, (
                f"{ch} {col} = {val} -- the NoData fill value is leaking into "
                "scene statistics")


def test_scene_stats_match_independently_masked_ground_truth(
        synthetic_project, force_lazy_export):
    """Exact values, not just 'not -9999': each band's scene stats must equal
    nanmean/nanmedian over that band's NON-fill pixels, computed straight from
    the committed ground-truth array."""
    name = "nodata_fragmented_multiband"
    spec = get_fixture(name)
    raw = load_raw_npz(name)
    by_channel = _scene_rows(synthetic_project, name)

    channel_names = {0: "R", 1: "G", 2: "B"}
    for b in range(spec["bands"]):
        ch = channel_names.get(b, f"band_{b + 1}")
        row = by_channel.get(ch)
        if row is None:
            continue
        band = raw[:, :, b].astype(np.float64)
        valid = band[~np.isclose(band, -9999.0, atol=1e-3)]
        if valid.size == 0:
            continue  # all-fill band: nothing to compare
        assert_close(row.get("Scene Mean"), float(np.nanmean(valid)),
                     tol=1e-3, msg=f"{ch} Scene Mean")
        assert_close(row.get("Scene Median"), float(np.nanmedian(valid)),
                     tol=1e-3, msg=f"{ch} Scene Median")


def test_lazy_and_eager_scene_stats_agree(synthetic_project, monkeypatch):
    """The eager path already masked scene stats correctly; the lazy path must
    now produce the same numbers rather than a differently-masked answer."""
    from .. import project_tab as project_tab_module

    name = "nodata_fragmented_multiband"
    pt = synthetic_project

    def _clear():
        cache = getattr(pt, "_export_image_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        for attr in ("_scene_stats_cache", "_per_band_nd_cache", "_master_nd_cache"):
            c = getattr(pt, attr, None)
            if hasattr(c, "clear"):
                c.clear()

    _clear()
    eager = _scene_rows(pt, name)

    _clear()
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    lazy = _scene_rows(pt, name)

    for ch in eager:
        if ch not in lazy:
            continue
        for col in ("Scene Mean", "Scene Median", "Scene Standard Deviation"):
            a, b = eager[ch].get(col), lazy[ch].get(col)
            if a is None or b is None:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            assert_close(a, b, tol=1e-3, msg=f"{ch} {col}: eager vs lazy")


def test_expression_nodata_only_affects_its_own_band_in_scene_stats(
        synthetic_project, force_lazy_export):
    """An expression names one band, so it must mask only that band's scene
    stats -- matching _per_band_nodata_masks' per-band semantics. If it leaked
    across bands it would silently shrink every other band's scene sample."""
    name = "nodata_fragmented_multiband"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    raw = load_raw_npz(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    band1 = raw[:, :, 1].astype(np.float64)
    threshold = float(np.median(band1))

    opts = dict(SCENE_OPTS)
    opts["nodata_enabled"] = True
    opts["nodata_values"] = [f"b2<{threshold}"]     # b2 == band index 1
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=opts)
    by_channel = _rows_by_channel(rows)

    # Band 1 ("G") must reflect the exclusion...
    g_expected = float(np.nanmean(band1[band1 >= threshold]))
    assert_close(by_channel["G"].get("Scene Mean"), g_expected, tol=1e-3,
                 msg="G Scene Mean under b2<threshold")

    # ...while band 0 ("R") is NOT touched by a b2 expression.
    #
    # DOCUMENTED, STILL-UNFIXED BEHAVIOR (flagged separately to the user):
    # supplying ANY explicit nodata_values REPLACES the file's auto-detected
    # GDAL_NODATA rather than adding to it -- the auto fallback only runs when
    # nodata_values is otherwise empty. So with only `b2<threshold` supplied,
    # band 0 keeps its -9999 fill pixels and its scene mean is dragged
    # negative. That is the CURRENT contract, pinned here so the behavior
    # cannot change silently; if custom rules are ever made additive, this
    # assertion flips to the fill-excluded expectation.
    band0 = raw[:, :, 0].astype(np.float64)
    r_expected_current = float(np.nanmean(band0))          # fill INCLUDED
    r_if_auto_were_additive = float(np.nanmean(band0[~np.isclose(band0, -9999.0, atol=1e-3)]))
    assert r_expected_current != pytest.approx(r_if_auto_were_additive), (
        "fixture no longer distinguishes the two behaviors -- pick a band that "
        "actually contains fill pixels")
    assert_close(by_channel["R"].get("Scene Mean"), r_expected_current, tol=1e-3,
                 msg="R Scene Mean must be untouched by a b2 expression "
                     "(and, per current behavior, still contains -9999 fill)")
