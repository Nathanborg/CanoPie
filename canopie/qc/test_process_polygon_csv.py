"""
QC regression tests for the main CSV export engine, ProjectTab.process_polygon
-- called DIRECTLY here (bypassing ExportWorker/QThread; confirmed safe, it's
a plain synchronous method) against every fixture's polygon and point
annotations, comparing per-band stats against ground truth.

NoData is per band (canopie/nodata.py, Nathan 2026-10-06): band b's
statistics exclude only the pixels where band b itself is NoData, and a band
that is NoData everywhere in the polygon (or at the point) KEEPS its row, with
empty statistics and Pixel Count 0. (Before, R/G/B were gated by band 0 and
such rows were silently omitted.)
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
        band_gt = gt["polygon"]["bands"][str(b)]["per_band"]

        assert ch_name in by_channel, f"{name} band {b} ({ch_name}): expected a row, got none. Rows: {list(by_channel)}"
        row = by_channel[ch_name]
        assert row["Pixel Count"] == band_gt["count"], f"{name} band {b} Pixel Count"
        if band_gt["count"] == 0:
            # Zero valid pixels -> the row is KEPT with empty statistics.
            for col in _STAT_KEY_MAP.values():
                assert row.get(col) is None, (
                    f"{name} band {b} ({ch_name}): {col} must be empty, got {row.get(col)!r}")
            continue
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

            assert ch_name in by_channel, f"{name}/{p['name']} band {b} ({ch_name}): expected a row"
            if is_nodata_here:
                # Per band: only this band is NoData here; its row is kept, empty.
                assert by_channel[ch_name].get("Mean") is None, (
                    f"{name}/{p['name']} band {b} ({ch_name}): NoData must give an empty Mean")
                assert by_channel[ch_name].get("Pixel Count") == 0
                continue
            assert_close(by_channel[ch_name].get("Mean"), expected, tol=1e-2,
                         msg=f"{name}/{p['name']} band {b} ({ch_name})")


def test_all_fill_band_keeps_an_empty_row(synthetic_project):
    """A 100%-fill ancillary band (GDAL_NODATA tag) is NoData only in ITSELF:
    R/G/B keep their full valid counts, and the fill band keeps its row with
    Pixel Count 0 and empty statistics (Nathan 2026-10-06)."""
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

    assert "R" in by_channel, "band 0 (R) should have full valid pixels (band 0 is clean)"
    assert by_channel["R"]["Pixel Count"] == gt["polygon"]["pixel_count_total"]
    assert "band_8" in by_channel, "the 100%-fill ancillary band (index 7) must keep its row"
    assert by_channel["band_8"]["Pixel Count"] == 0
    assert by_channel["band_8"]["Mean"] is None


def test_fragmented_nodata_polygon(synthetic_project):
    """A polygon straddling a valid patch and a NoData hole: R must exclude
    exactly the hole's pixels (its own NoData); the count must be strictly
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
    assert r_count == gt["polygon"]["bands"]["0"]["per_band"]["count"]


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


# --------------------------------------------------------------------------
# `properties` -> prop_* row keys
#
# process_polygon never read polygon_dict['properties'] at all before this --
# every row it emitted silently dropped whatever a shapefile import or the
# viewer's "Edit properties" dialog had stored. These pin the fix directly
# against the real function (not a reimplementation), using a real fixture's
# polygon so the row-shape/dedup machinery around it is exercised for real.
# --------------------------------------------------------------------------

def _polygon_with_properties(synthetic_project, name, properties):
    """A COPY of a real fixture's polygon dict with `properties` added --
    never mutate the live dict living on the session-scoped fixture."""
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = dict(synthetic_project.all_polygons[group][fp])
    poly_dict["properties"] = properties
    return group, fp, poly_dict


def test_polygon_properties_become_prefixed_row_keys(synthetic_project):
    props = {"DBH_CM": 31.5, "SPECIES": "Ceiba"}
    group, fp, poly_dict = _polygon_with_properties(
        synthetic_project, "rgb_8bit_untiled", props)
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=STATS_OPTS,
    )
    assert rows, "precondition: this fixture must produce at least one row"
    for row in rows:
        assert row.get("prop_DBH_CM") == 31.5, row
        assert row.get("prop_SPECIES") == "Ceiba", row


def test_polygon_without_properties_adds_no_prop_columns(synthetic_project):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]  # unmodified, no 'properties' key
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=STATS_OPTS,
    )
    assert rows
    for row in rows:
        assert not any(str(k).startswith("prop_") for k in row), row


def test_property_key_colliding_with_a_stat_name_does_not_overwrite_it():
    """THE collision-safety pin. A property literally named 'Mean' (or any
    other reserved column) must land in 'prop_Mean', never overwrite the
    real computed statistic in 'Mean'."""
    from canopie.utils import polygon_property_cells
    poly_dict = {"properties": {"Mean": "BOGUS", "Object ID": "BOGUS", "Channel": "BOGUS"}}
    cells = polygon_property_cells(poly_dict)
    assert cells == {"prop_Mean": "BOGUS", "prop_Object ID": "BOGUS", "prop_Channel": "BOGUS"}
    # And a real row already carrying the true stat is untouched by update():
    row = {"Mean": 42.0, "Object ID": "crown_1", "Channel": "R"}
    row.update(cells)
    assert row["Mean"] == 42.0, "prop_ cells overwrote the real Mean column"
    assert row["Object ID"] == "crown_1"
    assert row["Channel"] == "R"
    assert row["prop_Mean"] == "BOGUS"


def test_properties_do_not_change_row_count_or_dedup(synthetic_project):
    """Adding properties must not change WHICH rows are emitted or how many
    -- only add columns to rows that already exist."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    base_dict = synthetic_project.all_polygons[group][fp]

    rows_before, _ = synthetic_project.process_polygon(
        group, fp, base_dict, {}, [], False, opts=STATS_OPTS,
    )

    with_props = dict(base_dict)
    with_props["properties"] = {"DBH_CM": 31.5}
    rows_after, _ = synthetic_project.process_polygon(
        group, fp, with_props, {}, [], False, opts=STATS_OPTS,
    )

    key_cols = ("Object ID", "Channel", "File Name", "Label Band Index")
    before_keys = [tuple(r.get(k) for k in key_cols) for r in rows_before]
    after_keys = [tuple(r.get(k) for k in key_cols) for r in rows_after]
    assert before_keys == after_keys, "row identity/order changed by adding properties"


# --------------------------------------------------------------------------
# BUG-025: a "Label" row only with a classification
# --------------------------------------------------------------------------
def _clear_export_caches(pt):
    for attr in ("_export_image_cache", "_export_cache", "_per_band_nd_cache",
                 "_master_nd_cache", "_scene_stats_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def test_no_label_row_without_classification_and_counts_on_the_bands_own_row(synthetic_project):
    """With "Count 0/1" on and NO classification, the 0/1 band the counting
    finds (here a boolean band expression) keeps its OWN row, with the counts
    -- it used to be taken for "the label": its row was replaced by a "Label"
    row (BUG-025)."""
    import json
    import os
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    pt = synthetic_project
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: no .ax"
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"band_enabled": True, "band_expression": "b1>100"}, f)
        _clear_export_caches(pt)
        rows, _ = pt.process_polygon(group, fp, pt.all_polygons[group][fp], {}, [], False,
                                     opts={"stats": {"mean": True, "count01": True}})
    finally:
        os.remove(axp)
        _clear_export_caches(pt)
    by = _rows_by_channel(rows)
    assert "Label" not in by, f"a Label row without classification: {sorted(by)}"
    assert "b1>100" in by, f"the expression lost its own row: {sorted(by)}"
    row = by["b1>100"]
    assert (row.get("counts_0") or 0) + (row.get("counts_1") or 0) == row["Pixel Count"]


def test_negative_expression_values_are_kept_with_count01(synthetic_project):
    """The old "last band = label" guess also turned an expression's negative
    values (NDVI-style) into NaN."""
    name = "ax_band_expression_source"
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    pt = synthetic_project
    _clear_export_caches(pt)
    rows, _ = pt.process_polygon(group, fp, pt.all_polygons[group][fp], {}, [], False,
                                 opts={"stats": {"mean": True, "count01": True}})
    by = _rows_by_channel(rows)
    expr = spec["ax"]["band_expression"]
    assert "Label" not in by
    band_gt = gt["polygon"]["bands"][str(spec["bands"])]["per_band"]
    assert by[expr]["Pixel Count"] == band_gt["count"]
    assert_close(by[expr]["Mean"], band_gt["mean"], tol=1e-3, msg=f"{expr} Mean")
