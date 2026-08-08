"""
QC regression tests for the main CSV export engine, ProjectTab.process_polygon
-- called DIRECTLY here (bypassing ExportWorker/QThread; confirmed safe, it's
a plain synchronous method) against every fixture's polygon and point
annotations, comparing per-band stats against ground truth.

Also pins down process_polygon's documented NoData-masking asymmetry (bands
0/1/2 gated by band 0's own NoData status whenever there are 3+ bands; bands
>= 3 gated by their own band) as the current regression baseline -- NOT a bug
to fix here, see the plan's "Bug fixes" section for what IS in scope.
"""
import pytest

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from .project_builder import polygon_group_name, point_group_name, degenerate_group_name
from ._helpers import load_ground_truth, assert_close, expected_channel_names

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.extraction]

ALL_NAMES = [f["name"] for f in FIXTURES]

STATS_OPTS = {"stats": {"mean": True, "median": True, "std": True, "quantiles": [25, 75]}}

_STAT_KEY_MAP = {
    "mean": "Mean",
    "median": "Median",
    "std": "Standard Deviation",
    "q25": "Q25",
    "q75": "Q75",
}


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


@pytest.mark.parametrize("name", ALL_NAMES)
def test_polygon_stats_match_ground_truth(synthetic_project, name):
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=STATS_OPTS,
    )
    by_channel = _rows_by_channel(rows)

    channel_names = expected_channel_names(spec["bands"])
    for b, ch_name in enumerate(channel_names):
        band_gt = gt["polygon"]["bands"][str(b)]["process_polygon"]

        if band_gt["count"] == 0:
            # Zero valid pixels -> the row is silently omitted, never emitted
            # with NaN/zero-count placeholders (confirmed by direct read of
            # process_polygon's early-return branches).
            assert ch_name not in by_channel, f"{name} band {b} ({ch_name}): expected omitted row, but got one"
            continue

        assert ch_name in by_channel, f"{name} band {b} ({ch_name}): expected a row, got none. Rows: {list(by_channel)}"
        row = by_channel[ch_name]
        assert row["Pixel Count"] == band_gt["count"], f"{name} band {b} Pixel Count"
        for stat_key, col in _STAT_KEY_MAP.items():
            assert_close(row.get(col), band_gt[stat_key], tol=1e-2, msg=f"{name} band {b} ({ch_name}) {col}")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_point_values_match_ground_truth(synthetic_project, name):
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)

    for p in spec["points"]:
        group = point_group_name(name, p["name"])
        point_dict = synthetic_project.all_polygons[group][fp]
        rows, _ = synthetic_project.process_polygon(
            group, fp, point_dict, {}, [], False, opts={"stats": {"mean": True}},
        )
        by_channel = _rows_by_channel(rows)
        expected_vals = gt["points"][p["name"]]["values"]

        channel_names = expected_channel_names(spec["bands"])
        for b, ch_name in enumerate(channel_names):
            expected = expected_vals[b]
            nodata_value = gt["nodata_value"]
            is_nodata_here = nodata_value is not None and abs(expected - nodata_value) < 1e-3

            if is_nodata_here:
                # Point mode drops a point entirely only if EVERY band is
                # NoData; but within the RGB sub-block, if ANY of R/G/B is
                # individually NaN, that point's RGB rows are skipped anyway.
                # For bands >= 3 a NoData value is skipped individually.
                # Either way: this channel's row must be absent here.
                assert ch_name not in by_channel, (
                    f"{name}/{p['name']} band {b} ({ch_name}): expected omitted (NoData), got a row"
                )
                continue

            assert ch_name in by_channel, f"{name}/{p['name']} band {b} ({ch_name}): expected a row"
            assert_close(by_channel[ch_name].get("Mean"), expected, tol=1e-2,
                         msg=f"{name}/{p['name']} band {b} ({ch_name})")


def test_nodata_asymmetry_documented_baseline(synthetic_project):
    """Pins down the CURRENT, documented masking asymmetry as the regression
    baseline (not something this suite fixes): for a 100%-fill ancillary band
    (auto-detected GDAL_NODATA -> genuinely per-band masks), R/G/B are
    band-0-gated (band 0 itself is never NoData here, so they see full valid
    counts) while the fill band itself is correctly excluded via its own
    mask."""
    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=STATS_OPTS,
    )
    by_channel = _rows_by_channel(rows)

    assert "R" in by_channel, "band 0 (R) should have full valid pixels (band-0-gated, band 0 is clean)"
    assert by_channel["R"]["Pixel Count"] == gt["polygon"]["pixel_count_total"]
    assert "band_8" not in by_channel, "the 100%-fill ancillary band (index 7) must be entirely excluded"


def test_fragmented_nodata_polygon(synthetic_project):
    """A polygon straddling a valid patch and a NoData hole: R/G/B (band-0-
    gated) must exclude exactly the hole's pixels; the count must be strictly
    less than the polygon's total rasterized pixel count."""
    name = "nodata_fragmented_multiband"
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=STATS_OPTS,
    )
    by_channel = _rows_by_channel(rows)

    total = gt["polygon"]["pixel_count_total"]
    r_count = by_channel["R"]["Pixel Count"]
    assert r_count < total, f"expected some pixels excluded by the hole, got full count {r_count} == {total}"
    assert r_count == gt["polygon"]["bands"]["0"]["process_polygon"]["count"]


def test_degenerate_point_vs_1px_polygon_agree(synthetic_project):
    """A point and a co-located 1-vertex 'polygon' must produce identical
    per-band values -- process_polygon handles both via the same
    single-pixel-sample code path."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    deg_name = spec["degenerate_point_name"]

    point_group = point_group_name(name, deg_name)
    point_dict = synthetic_project.all_polygons[point_group][fp]
    point_rows, _ = synthetic_project.process_polygon(
        point_group, fp, point_dict, {}, [], False, opts={"stats": {"mean": True}},
    )

    deg_group = degenerate_group_name(name, deg_name)
    deg_dict = synthetic_project.all_polygons[deg_group][fp]
    deg_rows, _ = synthetic_project.process_polygon(
        deg_group, fp, deg_dict, {}, [], False, opts={"stats": {"mean": True}},
    )

    point_by_ch = _rows_by_channel(point_rows)
    deg_by_ch = _rows_by_channel(deg_rows)
    assert set(point_by_ch) == set(deg_by_ch)
    for ch, row in point_by_ch.items():
        assert_close(row["Mean"], deg_by_ch[ch]["Mean"], tol=1e-6, msg=f"point vs 1px-polygon {ch}")
