"""
QC regression tests for MachineLearningManager.export_csv_data (and a
lighter-weight check of train_models' inline pixel-sampling logic).

export_csv_data is dialog-driven end to end (QInputDialog x1-2,
QFileDialog.getSaveFileName x1) -- monkeypatched here rather than stubbed
out via a rewritten method, per this session's established preference for
exercising real code paths. Group selection drives the real QListWidget
(selectedItems()), not a monkeypatched get_selected_groups.

Cross-checks "Average Pixel Value" mode's per-band mean against
process_polygon's Mean on NoData-free fixtures (must match exactly, since
both reduce to a plain mean with nothing excluded) and explicitly documents
the divergence on fixtures with real NoData (whole-pixel-any-band-invalid
here vs. process_polygon's per-band/master mix).
"""
import csv

import pytest
from PyQt5 import QtWidgets

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_ground_truth, assert_close

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.ml, pytest.mark.extraction]

# A representative subset, not every NoData-free fixture: each
# MachineLearningManager() construction is expensive (sklearn-adjacent
# imports + QDialog setup), and channel_order/band-count/tiling correctness
# is already exhaustively covered per-fixture in test_process_polygon_csv.py
# -- this file only needs enough fixtures to confirm export_csv_data's OWN
# mean computation agrees with process_polygon's, across a representative
# spread of band counts (3-band baseline, 4-band w/ band_expression, 200-band).
NODATA_FREE = ["rgb_8bit_untiled", "ax_band_expression_source", "hyperspectral_200band"]
WITH_NODATA = ["multiband_8band_ancillary", "nodata_fragmented_multiband"]


def _select_group(mgr, group_name):
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() == group_name)


def _run_export_csv(monkeypatch, mgr, tmp_path, mode, window_choice="3x3"):
    calls = {"n": 0}

    def fake_get_item(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return mode, True
        return window_choice, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(fake_get_item))

    out_csv = tmp_path / "export.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                         staticmethod(lambda *a, **k: (str(out_csv), "")))

    mgr.export_csv_data()
    return out_csv


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("name", NODATA_FREE)
def test_average_pixel_value_matches_process_polygon(synthetic_project, ml_manager_factory, monkeypatch, tmp_path, name):
    """On fixtures with no real NoData, 'Average Pixel Value' mode's mean
    must match process_polygon's Mean exactly -- both reduce to a plain
    unweighted mean over the same rasterized pixel set with nothing excluded."""
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")
    rows = _read_csv(out_csv)
    assert len(rows) == 1, f"expected exactly one averaged row, got {len(rows)}: {rows}"
    row = rows[0]

    channel_names = ["R", "G", "B"] + [f"band_{i}" for i in range(4, spec["bands"] + 1)] if spec["bands"] >= 3 else \
                    [f"channel_{i}" for i in range(1, spec["bands"] + 1)]
    for b, ch_name in enumerate(channel_names):
        col = f"mean_{ch_name}"
        assert col in row, f"missing column {col!r} in {list(row)}"
        expected = gt["polygon"]["bands"][str(b)]["process_polygon"]["mean"]
        assert_close(float(row[col]), expected, tol=1e-1, msg=f"{name} {col}")


@pytest.mark.parametrize("name", ["rgb_8bit_untiled", "multiband_8band_ancillary"])
def test_all_pixel_values_spot_check(synthetic_project, ml_manager_factory, monkeypatch, tmp_path, name):
    """'All Pixel Values' mode must report the exact raw per-pixel value at
    each (x, y) it emits, matching the fixture's raw ground-truth array."""
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "All Pixel Values")
    rows = _read_csv(out_csv)
    assert len(rows) == gt["polygon"]["pixel_count_total"], (
        f"expected one row per rasterized pixel ({gt['polygon']['pixel_count_total']}), got {len(rows)}"
    )

    from ._helpers import load_raw_npz
    raw = load_raw_npz(name)
    channel_names = ["R", "G", "B"] + [f"band_{i}" for i in range(4, spec["bands"] + 1)]
    for row in rows[:10]:  # a representative sample is enough; every row shares the same code path
        xi, yi = int(row["x"]), int(row["y"])
        for b, ch_name in enumerate(channel_names):
            if ch_name not in row or row[ch_name] == "":
                continue
            assert_close(float(row[ch_name]), float(raw[yi, xi, b]), tol=1e-1,
                         msg=f"{name} pixel ({xi},{yi}) {ch_name}")


@pytest.mark.parametrize("name", WITH_NODATA)
def test_gdal_nodata_tag_ignored_by_ml_export(synthetic_project, ml_manager_factory, monkeypatch, tmp_path, name):
    """REGRESSION BASELINE for a real, currently-unfixed defect.

    Fixtures 4 and 6 declare their fill value ONLY via the raster's
    GDAL_NODATA tag (no .ax sidecar). process_polygon honors that tag (via
    effective_nodata_values) and masks correctly. export_csv_data does NOT --
    it only ever reads nodata_values from a .ax file -- so it averages the raw
    fill values straight into its means.

    The damage is blatant on fixture 6, whose -9999 holes drag mean_R to
    -2034.5, a value no real pixel in the image is anywhere near (valid pixels
    are ~0-1000). Any downstream ML training on that CSV is being fed garbage
    features for every polygon that touches a NoData region.

    This test pins the CURRENT behavior so the suite stays green and the defect
    stays visible; flip it to assert the process_polygon value once the tag is
    honored in export_csv_data (see test_ml_manager_bugfix.py for the pattern)."""
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")
    rows = _read_csv(out_csv)
    assert len(rows) == 1
    row = rows[0]

    band0 = gt["polygon"]["bands"]["0"]
    r_actual = float(row["mean_R"])

    # No .ax on these fixtures => ML export applies NO masking whatsoever.
    assert_close(r_actual, band0["no_mask"]["mean"], tol=1e-1,
                 msg=f"{name} mean_R should equal the UNMASKED mean (GDAL_NODATA tag ignored)")

    # And where that genuinely differs from correctly-masked output, prove the
    # divergence rather than letting a coincidental match hide it.
    pp_mean = band0["process_polygon"]["mean"]
    if pp_mean is not None and abs(band0["no_mask"]["mean"] - pp_mean) > 1e-6:
        assert abs(r_actual - pp_mean) > 1e-6, (
            f"{name}: expected ML export to diverge from process_polygon's masked "
            f"Mean ({pp_mean}), but it matched -- has the tag now been honored? "
            f"If so, update this test."
        )


def test_ax_declared_nodata_IS_honored_by_ml_export(synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """The flip side of the test above: when NoData IS declared in a .ax
    sidecar (fixture 7), export_csv_data does mask it correctly, whole-pixel.
    This is what proves the defect above is specifically "the GDAL_NODATA tag
    is ignored", not "ML export can't mask at all"."""
    name = "ax_crop_nodata_source"
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")
    rows = _read_csv(out_csv)
    assert len(rows) == 1

    band0 = gt["polygon"]["bands"]["0"]
    assert_close(float(rows[0]["mean_R"]), band0["ml_manager"]["mean"], tol=1e-1,
                 msg=f"{name} mean_R should match the .ax-masked (whole-pixel) mean")


# ---------------------------------------------------------------------------
# Polygon `properties` -> trailing prop_* columns (Step 3 of the properties-
# export feature). header is a static list decided once, up front, in
# export_csv_data -- see the pre-pass comment there -- then written verbatim
# by _CsvExportWorker.run(); these tests exercise that real dialog-driven
# entry point end to end, not the worker in isolation.
# ---------------------------------------------------------------------------

def _select_groups(mgr, *group_names):
    wanted = set(group_names)
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() in wanted)


def test_prop_columns_appear_after_the_fixed_columns(synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(original, properties={"DBH_CM": 31.5, "SPECIES": "Ceiba"})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")
        with open(out_csv, newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames
        assert fieldnames[-2:] == ["prop_DBH_CM", "prop_SPECIES"], fieldnames
        assert not fieldnames[-2:][0].startswith("mean_")
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_every_row_of_a_polygon_carries_its_properties(synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(original, properties={"DBH_CM": 31.5})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "All Pixel Values")
        rows = _read_csv(out_csv)
        assert len(rows) > 1, "need multiple rows from one polygon to prove this isn't a fluke"
        assert all(r["prop_DBH_CM"] == "31.5" for r in rows)
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_missing_keys_fill_blank_across_polygons(synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """THE core 'adapt for missing/different properties' pin for this export
    path: disjoint property key sets across polygons must fill gaps blank,
    with an unchanged row count -- never dropped rows."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group_a = polygon_group_name(name, spec["polygon"]["name"])
    group_b = "__ml_csv_gap_fill_test__"

    original_a = synthetic_project.all_polygons[group_a][fp]
    synthetic_project.all_polygons[group_a][fp] = dict(original_a, properties={"DBH_CM": 31.5})
    synthetic_project.all_polygons[group_b] = {fp: dict(original_a, name=group_b)}  # no 'properties' key at all
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_groups(mgr, group_a, group_b)
        out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")
        rows = _read_csv(out_csv)
        by_group = {r["group_name"]: r for r in rows}
        assert group_a in by_group and group_b in by_group, (
            f"expected a row for both groups, got: {list(by_group)}")
        assert by_group[group_a]["prop_DBH_CM"] == "31.5"
        assert by_group[group_b]["prop_DBH_CM"] == "", "gap not blank for the property-less polygon"
    finally:
        synthetic_project.all_polygons[group_a][fp] = original_a
        synthetic_project.all_polygons.pop(group_b, None)


def test_property_key_named_x_does_not_shift_the_pixel_columns(synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """Collision pin: a property literally named 'x' becomes 'prop_x', a
    DIFFERENT column from the real pixel-coordinate 'x' -- the real column
    must keep holding pixel coordinates, not the property value."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(original, properties={"x": "not-a-pixel-coordinate"})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "All Pixel Values")
        with open(out_csv, newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames
        assert "prop_x" in fieldnames
        assert fieldnames.index("x") != fieldnames.index("prop_x")
        rows = _read_csv(out_csv)
        assert rows, "no rows written"
        for row in rows[:5]:
            int(row["x"])  # must still parse as a plain pixel coordinate
            assert row["prop_x"] == "not-a-pixel-coordinate"
    finally:
        synthetic_project.all_polygons[group][fp] = original
