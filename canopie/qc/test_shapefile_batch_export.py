"""
QC regression tests for shapefile export during a BACKGROUND ("batched") CSV run,
and for the attribute-ordering / feature-scoping of the embedded statistics.

THE BUG THIS PINS (reported as: "not exporting shapefile when file batched
(running in the background)"):

`save_polygons_to_csv` did the shapefile export inline at its very END, but the
background branch returns roughly 800 lines earlier:

    if run_background:
        worker = ExportWorker(...)
        mgr.add_export(worker, save_path)
        return None                       # <-- never reaches the shapefile code

and `ExportWorker` contained no shapefile logic whatsoever (grep for
"shapefile" in the class came back empty). So ticking "Also export as
Shapefile" and choosing "run in background" produced a CSV and, silently, no
shapefile at all -- no error, no warning.

The export is now a method, `ProjectTab.export_shapefile_bundle`, called by both
paths. It deliberately returns `(message, ok)` rather than popping a
QMessageBox, because the worker calls it from a QThread and Qt widgets may only
be touched on the GUI thread -- a dialog there is a crash, not a cosmetic issue.

Also pinned: geometry is taken from the shapes the export actually processed,
not from `self.all_polygons` (the whole project), which used to drag in shapes
the user had excluded -- shapes with no CSV row, and potentially on rasters in
an entirely different CRS.
"""
import os

import numpy as np
import pytest
import tifffile

from ..shapefile_io import json_polygons_to_features, write_feature_collection
from .test_shapefile_crs import _write_geotiff, _shape

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.io]


# ---------------------------------------------------------------------------
# The background worker path
# ---------------------------------------------------------------------------
def test_background_export_worker_writes_a_shapefile(synthetic_project, tmp_path,
                                                     fixtures_ready):
    """THE regression, driven BEHAVIOURALLY through the real ExportWorker.

    `run()` is called directly rather than via `start()` so it executes
    synchronously on this thread -- no QThread scheduling, no Export Manager
    dialog -- while still exercising the actual production code path.

    A source-text assertion is deliberately NOT used here: it passes against
    dead code (`if False:` around the call still contains every expected
    identifier), which is exactly the mistake that would let this regress."""
    from ..project_tab import ExportWorker
    from .fixtures_manifest import fixture_image_path, get_fixture
    from .project_builder import polygon_group_name

    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    out_csv = str(tmp_path / "batched.csv")
    worker = ExportWorker(
        synthetic_project, out_csv,
        {"stats": {"mean": True}, "export_shapefile": True},
        [(group, fp, poly_dict)],
        {}, [], False, False,
        "sequential", 1,
        snapshot_dir=str(tmp_path),
    )
    worker.run()          # synchronous: this IS the background code path

    assert os.path.exists(out_csv), "the background export produced no CSV"
    produced = sorted(p.name for p in tmp_path.glob("*.shp"))
    assert produced, (
        "background export wrote a CSV but NO shapefile, even though "
        "export_shapefile was requested -- files present: "
        f"{sorted(p.name for p in tmp_path.iterdir())}")


def test_background_and_foreground_use_the_same_export(synthetic_project, tmp_path,
                                                       fixtures_ready):
    """Both paths must funnel through export_shapefile_bundle, or they drift --
    which is how the background path came to have no shapefile step at all."""
    import ast
    import inspect
    import textwrap
    from ..project_tab import ExportWorker

    tree = ast.parse(textwrap.dedent(inspect.getsource(ExportWorker.run)))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "export_shapefile_bundle" in called, (
        "ExportWorker.run must call ProjectTab.export_shapefile_bundle")


def test_export_shapefile_bundle_exists_and_is_dialog_free():
    """It runs on a QThread. A QMessageBox there is a crash, not a nuisance --
    so the method must return its message instead of showing it."""
    import ast
    import inspect
    import textwrap
    from ..project_tab import ProjectTab

    assert hasattr(ProjectTab, "export_shapefile_bundle")

    # Parse rather than grep: the docstring legitimately mentions QMessageBox to
    # explain why there isn't one, and a substring check flags that as a hit.
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(ProjectTab.export_shapefile_bundle)))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    assert "QMessageBox" not in used, (
        "export_shapefile_bundle is called from ExportWorker's thread; it must "
        "not touch Qt widgets")


def test_bundle_writes_a_shapefile_and_reports_success(synthetic_project, tmp_path):
    """End to end through the real method, with real rows."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    polygons_to_process = [("g", a, _shape(a))]
    data_rows = [{
        "File Name": "a.tif", "Object ID": "a.tif", "Channel": "Gray",
        "Mean": 12.5, "Median": 12.0, "Pixel Count": 100,
    }]

    save_path = str(tmp_path / "run.csv")
    msg, ok = synthetic_project.export_shapefile_bundle(
        save_path, polygons_to_process, data_rows)

    assert ok, f"export reported failure: {msg}"
    assert os.path.exists(str(tmp_path / "run.shp")), f"no shapefile written: {msg}"


def test_bundle_reports_failure_without_raising(synthetic_project, tmp_path):
    """A shapefile problem must never take down a CSV export that already
    succeeded -- the CSV is the primary artifact."""
    msg, ok = synthetic_project.export_shapefile_bundle(
        str(tmp_path / "run.csv"), [], [])

    assert ok is False
    assert isinstance(msg, str) and msg


# ---------------------------------------------------------------------------
# Feature scoping: only the shapes this export processed
# ---------------------------------------------------------------------------
def test_only_processed_shapes_become_features(synthetic_project, tmp_path):
    """Geometry used to come from self.all_polygons -- the WHOLE project -- so a
    filtered export still wrote every polygon in the project, including ones
    with no CSV row and ones on rasters in another CRS."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    b = _write_geotiff(str(tmp_path / "b.tif"), 32620, 601000.0, 1000000.0)

    # The project knows about both; the export processes only 'a'.
    synthetic_project.all_polygons["ga"] = {a: _shape(a)}
    synthetic_project.all_polygons["gb"] = {b: _shape(b)}
    try:
        save_path = str(tmp_path / "subset.csv")
        msg, ok = synthetic_project.export_shapefile_bundle(
            save_path, [("ga", a, _shape(a))],
            [{"File Name": "a.tif", "Object ID": "a.tif", "Mean": 1.0}])

        assert ok, msg
        assert "1 feature(s)" in msg, (
            f"expected exactly the one processed shape, got: {msg}")
    finally:
        synthetic_project.all_polygons.pop("ga", None)
        synthetic_project.all_polygons.pop("gb", None)


# ---------------------------------------------------------------------------
# Attribute ordering vs. the CSV
# ---------------------------------------------------------------------------
def _dbf_field_names(dbf_path):
    """Field names in header order, read straight back out of the DBF."""
    with open(dbf_path, "rb") as f:
        header = f.read(32)
        header_len = int.from_bytes(header[8:10], "little")
        n_fields = (header_len - 33) // 32
        names = []
        for _ in range(n_fields):
            names.append(f.read(32)[:11].split(b"\x00")[0].decode("ascii", "ignore"))
    return names


def test_attribute_columns_follow_the_csv_column_order(tmp_path):
    """"Organize how the features appear regarding the stats from export_csv":
    the DBF used to be ordered by whatever order process_polygon happened to
    build each row's dict, so the attribute table bore no relation to the CSV."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    stats = {("g", a): [{"Mean": 1.0, "Pixel Count": 10, "Median": 2.0,
                         "Standard Deviation": 0.5, "Project": "p"}]}
    features, _crs, _w = json_polygons_to_features(
        {"g": {a: _shape(a)}}, str(tmp_path), stats_lookup=stats)

    csv_order = ["Project", "Pixel Count", "Mean", "Median", "Standard Deviation"]
    stem = str(tmp_path / "out")
    written, _notes = write_feature_collection(features, stem, field_order=csv_order)

    names = _dbf_field_names(os.path.splitext(written[0])[0] + ".dbf")
    stat_positions = {n: i for i, n in enumerate(names)}

    seen = [c for c in ("Project", "Pixel Coun", "Mean", "Median") if c in stat_positions]
    assert len(seen) >= 3, f"expected the CSV stat columns in the DBF: {names}"
    ordered = [stat_positions[c] for c in seen]
    assert ordered == sorted(ordered), (
        f"DBF columns {seen} are not in CSV order; full header: {names}")


def test_channel_flattened_columns_group_with_their_base_statistic(tmp_path):
    """Multi-channel rows flatten to R_Mean/G_Mean/B_Mean. Those must sit
    together with Mean rather than scattering across the table."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    stats = {("g", a): [{"R_Mean": 1.0, "G_Mean": 2.0, "B_Mean": 3.0,
                         "R_Median": 4.0, "G_Median": 5.0, "Pixel Count": 9}]}
    features, _crs, _w = json_polygons_to_features(
        {"g": {a: _shape(a)}}, str(tmp_path), stats_lookup=stats)

    stem = str(tmp_path / "out")
    written, _notes = write_feature_collection(
        features, stem, field_order=["Pixel Count", "Mean", "Median"])

    names = _dbf_field_names(os.path.splitext(written[0])[0] + ".dbf")
    means = [i for i, n in enumerate(names) if n.endswith("_Mean")]
    medians = [i for i, n in enumerate(names) if n.endswith("_Median")]

    assert means and medians, f"channel columns missing: {names}"
    assert max(means) < min(medians), (
        f"the _Mean group should precede the _Median group: {names}")


def test_unlisted_fields_are_still_written(tmp_path):
    """Ordering must never DROP a column the caller didn't think to list."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    stats = {("g", a): [{"Mean": 1.0, "SomethingNew": 42.0}]}
    features, _crs, _w = json_polygons_to_features(
        {"g": {a: _shape(a)}}, str(tmp_path), stats_lookup=stats)

    stem = str(tmp_path / "out")
    written, _notes = write_feature_collection(
        features, stem, field_order=["Mean"])

    names = _dbf_field_names(os.path.splitext(written[0])[0] + ".dbf")
    assert any(n.startswith("Something") for n in names), (
        f"an unlisted field was dropped: {names}")
