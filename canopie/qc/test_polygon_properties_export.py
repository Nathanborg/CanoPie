"""
QC tests for polygon `properties` reaching CSV exports, end to end through
the REAL entry points -- not process_polygon in isolation
(canopie/qc/test_process_polygon_csv.py already covers that).

WHY THIS FILE EXISTS
--------------------
process_polygon merging `prop_*` keys into its row dicts is only HALF the
fix. The main CSV export has two independent write paths and only one of
them is dynamic:

  * The background path (ExportWorker) accumulates a union of every row
    key it ever sees and writes with csv.DictWriter(extrasaction="ignore")
    -- any new row key becomes a column automatically. No code change was
    needed there beyond process_polygon itself.
  * The foreground path (save_polygons_to_csv, non-background branch) uses
    a STATIC fieldnames list built up front, ALSO with
    extrasaction="ignore" -- which means a row key not in that static list
    is SILENTLY DROPPED, not defaulted. process_polygon's fix alone does
    nothing on this path; it needs its own pre-pass over the source
    polygons to widen the header (mirroring the existing Class_* dynamic
    column precedent already in that function).

These tests drive both real entry points -- save_polygons_to_csv() for the
foreground path, ExportWorker.run() called directly (bypassing the QThread
event loop, exactly like calling .run() synchronously) for the background
path -- and assert on the actual CSV bytes each one writes.
"""
import csv
import os

import pytest

from .project_builder import build_project_tab

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _any_existing_fixture_filepath(pt):
    for file_map in pt.all_polygons.values():
        for fp in file_map:
            return fp
    raise AssertionError("no polygons available to borrow a filepath from")


def _poly(name, ref_w=100, ref_h=100, properties=None):
    d = {
        'points': [[0, 0], [10, 0], [10, 10]], 'name': name, 'root': '1',
        'coordinates': {'latitude': 0.0, 'longitude': 0.0}, 'type': 'polygon',
        'coord_space': 'image', 'image_ref_size': {'w': ref_w, 'h': ref_h},
    }
    if properties is not None:
        d['properties'] = properties
    return d


@pytest.fixture()
def project_tab(qapp, tmp_path):
    """A fresh, non-session-scoped ProjectTab per test -- these tests write
    real CSV files to the project's exports/ folder and mutate all_polygons,
    so a dedicated instance is simpler and safer than borrowing the shared
    synthetic_project and restoring it."""
    pt = build_project_tab(str(tmp_path))
    pt.analysis_options = {}
    yield pt


def test_foreground_csv_writes_prop_columns(project_tab):
    fp = _any_existing_fixture_filepath(project_tab)
    group = "__prop_export_foreground__"
    props = {"DBH_CM": 31.5, "SPECIES": "Ceiba"}
    project_tab.all_polygons[group] = {fp: _poly(group, properties=props)}

    save_path = project_tab.save_polygons_to_csv(
        options={"processing_params": ("sequential", 1, False)})
    assert save_path and os.path.exists(save_path)

    with open(save_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert "prop_DBH_CM" in reader.fieldnames
        assert "prop_SPECIES" in reader.fieldnames
        rows = [r for r in reader if r.get("Object ID") == group]
    assert rows, "no rows written for the test polygon"
    for row in rows:
        assert row["prop_DBH_CM"] == "31.5"
        assert row["prop_SPECIES"] == "Ceiba"


def test_foreground_and_background_csv_agree_on_prop_columns(project_tab, tmp_path):
    """This is the test that catches the extrasaction="ignore" trap
    reappearing: if the foreground pre-pass (Step 2c) is ever removed, this
    fails while the background-only equivalent keeps passing -- exactly the
    asymmetric signature that bug has."""
    from canopie.project_tab import ExportWorker

    fp = _any_existing_fixture_filepath(project_tab)
    group = "__prop_export_agree__"
    props = {"DBH_CM": 31.5, "SPECIES": "Ceiba"}
    poly_dict = _poly(group, properties=props)
    project_tab.all_polygons[group] = {fp: poly_dict}

    fg_path = project_tab.save_polygons_to_csv(
        options={"processing_params": ("sequential", 1, False)})
    assert fg_path

    bg_path = str(tmp_path / "background_export.csv")
    worker = ExportWorker(
        project_tab, bg_path, {"stats": {"mean": True}}, [(group, fp, poly_dict)],
        {}, [], False, True, "sequential", 1,
    )
    worker.run()
    assert os.path.exists(bg_path)

    with open(fg_path, encoding="utf-8-sig") as f:
        fg_fields = set(csv.DictReader(f).fieldnames)
    with open(bg_path, encoding="utf-8-sig") as f:
        bg_fields = set(csv.DictReader(f).fieldnames)

    fg_props = {c for c in fg_fields if c.startswith("prop_")}
    bg_props = {c for c in bg_fields if c.startswith("prop_")}
    assert fg_props == bg_props == {"prop_DBH_CM", "prop_SPECIES"}, (
        f"foreground prop_* columns {fg_props} != background {bg_props}")


def test_mixed_property_key_sets_fill_gaps_not_dropped_rows(project_tab):
    """THE core 'adapt for missing properties' pin: polygons across a
    project with different (or absent) property keys must all still get
    exported, with gaps blank, not silently dropped."""
    fp = _any_existing_fixture_filepath(project_tab)
    g_a, g_b, g_c = (
        "__prop_export_mixed_a__", "__prop_export_mixed_b__", "__prop_export_mixed_c__",
    )
    project_tab.all_polygons[g_a] = {fp: _poly(g_a, properties={"DBH_CM": 31.5})}
    project_tab.all_polygons[g_b] = {fp: _poly(g_b, properties={"SPECIES": "Ceiba"})}
    project_tab.all_polygons[g_c] = {fp: _poly(g_c, properties=None)}  # hand-drawn: no key at all

    save_path = project_tab.save_polygons_to_csv(
        options={"processing_params": ("sequential", 1, False)})
    assert save_path

    with open(save_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert {"prop_DBH_CM", "prop_SPECIES"} <= set(reader.fieldnames)
        rows_by_group = {}
        for row in reader:
            rows_by_group.setdefault(row.get("Object ID"), []).append(row)

    for g in (g_a, g_b, g_c):
        assert rows_by_group.get(g), f"polygon {g} produced no rows at all"

    assert rows_by_group[g_a][0]["prop_DBH_CM"] == "31.5"
    assert rows_by_group[g_a][0]["prop_SPECIES"] == "", "gap not blank for group A"
    assert rows_by_group[g_b][0]["prop_SPECIES"] == "Ceiba"
    assert rows_by_group[g_b][0]["prop_DBH_CM"] == "", "gap not blank for group B"
    assert rows_by_group[g_c][0]["prop_DBH_CM"] == ""
    assert rows_by_group[g_c][0]["prop_SPECIES"] == ""
