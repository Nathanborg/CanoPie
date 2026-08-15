"""
QC tests for polygon `properties` reaching generate_thumbnails' new
thumbnail_properties.csv sidecar (Step 5 of the properties-export feature).

No sidecar existed before this feature -- it is always written now (same
"unconditional, matches segmentation's existing behaviour" decision as
segmentation_labels.csv, see test_ml_manager_segmentation_csv.py), one row
per OUTPUT FILE rather than per polygon, since tiling can turn one polygon
into 0..N files and every file maps back to exactly one polygon.

generate_thumbnails() shows a folder picker and a modal ThumbnailOptionsDialog
(QMessageBox.information at the end is already neutralized process-wide by
conftest's autouse _no_modal_dialogs) -- both patched here so the REAL
dialog-driven entry point runs headlessly, same convention as
test_ml_manager_segmentation_csv.py's _run_generate_segmentation_images.
Group selection drives the real QListWidget (selectedItems()), not a
monkeypatched get_selected_groups.
"""
import csv
import os

import pytest
from PyQt5 import QtWidgets

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.ml, pytest.mark.io]


def _select_group(mgr, *group_names):
    wanted = set(group_names)
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() in wanted)


def _run_generate_thumbnails(monkeypatch, mgr, save_dir):
    from canopie.thumbnail_options_dialog import ThumbnailOptionsDialog
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                         staticmethod(lambda *a, **k: str(save_dir)))
    # Real dialog, real (unmodified) defaults -- "press Generate and nothing
    # changes", same convention test_thumbnail_options.py pins elsewhere.
    monkeypatch.setattr(ThumbnailOptionsDialog, "exec_",
                         lambda self: QtWidgets.QDialog.Accepted)
    mgr.generate_thumbnails()


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_thumbnail_sidecar_has_identity_columns_first(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    save_dir = tmp_path / "thumb_out1"
    save_dir.mkdir()
    _run_generate_thumbnails(monkeypatch, mgr, save_dir)

    csv_path = save_dir / "thumbnail_properties.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    assert fieldnames[:4] == ["thumbnail_file", "group_name", "image_file", "polygon_index"]


def test_thumbnail_sidecar_carries_polygon_properties(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(
        original, properties={"DBH_CM": 31.5, "SPECIES": "Ceiba"})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        save_dir = tmp_path / "thumb_out2"
        save_dir.mkdir()
        _run_generate_thumbnails(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "thumbnail_properties.csv")
        assert rows, "no rows written"
        for row in rows:
            assert row.get("prop_DBH_CM") == "31.5"
            assert row.get("prop_SPECIES") == "Ceiba"
            assert row["thumbnail_file"], "thumbnail_file must not be blank"
            assert (save_dir / row["thumbnail_file"]).exists(), (
                f"sidecar names {row['thumbnail_file']!r} but no such file was written")
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_thumbnail_sidecar_fills_gaps_for_polygons_without_properties(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """THE core 'adapt for missing properties' pin for this export path."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group_a = polygon_group_name(name, spec["polygon"]["name"])
    group_b = "__thumb_csv_gap_fill_test__"

    original_a = synthetic_project.all_polygons[group_a][fp]
    synthetic_project.all_polygons[group_a][fp] = dict(original_a, properties={"DBH_CM": 31.5})
    synthetic_project.all_polygons[group_b] = {fp: dict(original_a, name=group_b)}  # no 'properties' key
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group_a, group_b)
        save_dir = tmp_path / "thumb_out3"
        save_dir.mkdir()
        _run_generate_thumbnails(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "thumbnail_properties.csv")
        by_group = {r["group_name"]: r for r in rows}
        assert group_a in by_group and group_b in by_group, (
            f"expected a row for both groups, got: {list(by_group)}")
        assert by_group[group_a]["prop_DBH_CM"] == "31.5"
        assert by_group[group_b]["prop_DBH_CM"] == "", "gap not blank for the property-less polygon"
    finally:
        synthetic_project.all_polygons[group_a][fp] = original_a
        synthetic_project.all_polygons.pop(group_b, None)


def test_thumbnail_sidecar_property_named_group_name_does_not_collide(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """Collision pin: a property literally named 'group_name' becomes
    'prop_group_name', a DIFFERENT column -- the real group_name column must
    keep holding the actual group, not the property value."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(
        original, properties={"group_name": "not-the-real-group"})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        save_dir = tmp_path / "thumb_out4"
        save_dir.mkdir()
        _run_generate_thumbnails(monkeypatch, mgr, save_dir)

        with open(save_dir / "thumbnail_properties.csv", newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames
        assert "prop_group_name" in fieldnames
        assert fieldnames.index("group_name") != fieldnames.index("prop_group_name")

        rows = _read_csv(save_dir / "thumbnail_properties.csv")
        assert rows
        for row in rows:
            assert row["group_name"] == group
            assert row["prop_group_name"] == "not-the-real-group"
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_thumbnail_sidecar_repeats_properties_across_tiles(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """One row per OUTPUT FILE, not per polygon: a tiled polygon's property
    values must repeat identically across every tile it produces."""
    from canopie.machine_learning_manager import MachineLearningManager

    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(original, properties={"DBH_CM": 31.5})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)

        # Force this polygon to yield 3 output files regardless of its real
        # geometry -- isolates "does a tiled polygon repeat its properties
        # across every file" from tiling geometry itself, which
        # test_thumbnail_options.py already covers exhaustively.
        def fake_render(self, src, pts_img, x0, y0, new_w, new_h, W, H, color, opts, out_name):
            import numpy as np
            stem, ext = os.path.splitext(out_name)
            for i in range(3):
                yield f"{stem}_t{i}{ext}", np.zeros((4, 4, 3), dtype=np.uint8)

        monkeypatch.setattr(MachineLearningManager, "_render_thumbnail_outputs", fake_render)

        save_dir = tmp_path / "thumb_out5"
        save_dir.mkdir()
        _run_generate_thumbnails(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "thumbnail_properties.csv")
        assert len(rows) == 3, f"expected 3 rows (one per tile file), got {len(rows)}"
        assert len({r["thumbnail_file"] for r in rows}) == 3, "tile filenames must be distinct"
        assert all(r["prop_DBH_CM"] == "31.5" for r in rows)
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_no_properties_thumbnail_run_still_writes_a_sidecar_with_no_prop_columns(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """When nothing selected has `properties`, the sidecar is still written
    (it's independently useful as a filename->polygon map) but carries no
    prop_* columns at all."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    save_dir = tmp_path / "thumb_out6"
    save_dir.mkdir()
    _run_generate_thumbnails(monkeypatch, mgr, save_dir)

    csv_path = save_dir / "thumbnail_properties.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    assert fieldnames == ["thumbnail_file", "group_name", "image_file", "polygon_index"]


# ---------------------------------------------------------------------------
# Pure-function unit test + source-level pin, no Qt/fixtures required.
# ---------------------------------------------------------------------------
def test_thumbnail_sidecar_rows_pure_function_shape():
    from canopie.machine_learning_manager import _thumbnail_sidecar_rows

    records = [
        ("a.jpg", "GroupA", "/img/1.tif", 1, [31.5, "Ceiba"]),
        ("b.jpg", "GroupB", "/img/1.tif", 1, ["", ""]),
    ]
    header, rows = _thumbnail_sidecar_rows(["DBH_CM", "SPECIES"], records)
    assert header == ["thumbnail_file", "group_name", "image_file", "polygon_index",
                       "prop_DBH_CM", "prop_SPECIES"]
    assert rows == [
        ["a.jpg", "GroupA", "/img/1.tif", 1, 31.5, "Ceiba"],
        ["b.jpg", "GroupB", "/img/1.tif", 1, "", ""],
    ]


def test_thumbnail_sidecar_rows_with_no_properties_at_all():
    from canopie.machine_learning_manager import _thumbnail_sidecar_rows

    header, rows = _thumbnail_sidecar_rows([], [("a.jpg", "GroupA", "/img/1.tif", 1, [])])
    assert header == ["thumbnail_file", "group_name", "image_file", "polygon_index"]
    assert rows == [["a.jpg", "GroupA", "/img/1.tif", 1]]


def test_render_thumbnail_outputs_never_references_properties():
    """Source-level pin, same style as test_no_per_pixel_loop_or_dict_allocation_remains
    in test_ml_manager_csv_export_perf.py: the tile-rendering path stays a
    separate layer from property handling -- prop_vals is fetched once per
    polygon in generate_thumbnails and threaded around this method, never
    into it.

    Checks for "properties"/"prop_" specifically, not the substring
    "propert" -- _ThumbSource legitimately uses the unrelated @property
    decorator, which would otherwise false-positive this pin."""
    import inspect
    from canopie.machine_learning_manager import MachineLearningManager, _ThumbSource

    src = inspect.getsource(MachineLearningManager._render_thumbnail_outputs)
    assert "properties" not in src.lower() and "prop_" not in src

    src2 = inspect.getsource(_ThumbSource)
    assert "properties" not in src2.lower() and "prop_" not in src2
