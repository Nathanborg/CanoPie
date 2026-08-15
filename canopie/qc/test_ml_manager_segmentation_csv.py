"""
QC tests for polygon `properties` reaching `generate_segmentation_images`'s
CSV sidecar.

segmentation_labels.csv already existed (one row per polygon: object_id,
group_name, image_file, polygon_index) -- this widens it with prop_* columns,
using the same header-must-be-finalized-up-front pre-pass as
export_csv_data's own header. No options dialog exists for this flow (goes
straight from folder picker to work), so the CSV was already unconditional
and stays that way.

Group selection drives the real QListWidget (selectedItems()), matching
test_ml_manager_csv_and_training.py's established convention, not a
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


def _run_generate_segmentation_images(monkeypatch, mgr, save_dir):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                         staticmethod(lambda *a, **k: str(save_dir)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                         staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         staticmethod(lambda *a, **k: None))
    mgr.generate_segmentation_images()


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_segmentation_csv_keeps_its_original_four_columns_first(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    save_dir = tmp_path / "seg_out1"
    save_dir.mkdir()
    _run_generate_segmentation_images(monkeypatch, mgr, save_dir)

    csv_path = save_dir / "segmentation_labels.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    assert fieldnames[:4] == ["object_id", "group_name", "image_file", "polygon_index"]


def test_segmentation_csv_carries_polygon_properties(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    props = {"DBH_CM": 31.5, "SPECIES": "Ceiba"}
    synthetic_project.all_polygons[group][fp] = dict(original, properties=props)
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        save_dir = tmp_path / "seg_out2"
        save_dir.mkdir()
        _run_generate_segmentation_images(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "segmentation_labels.csv")
        assert rows, "no rows written"
        for row in rows:
            assert row.get("prop_DBH_CM") == "31.5"
            assert row.get("prop_SPECIES") == "Ceiba"
    finally:
        synthetic_project.all_polygons[group][fp] = original


def test_segmentation_csv_fills_gaps_for_polygons_without_properties(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """THE core 'adapt for missing properties' pin for this export path."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group_a = polygon_group_name(name, spec["polygon"]["name"])

    # A second group at the SAME file/geometry, so both polygons rasterize
    # onto the same mask -- one with properties, one without (mimicking a
    # hand-drawn polygon alongside a shapefile-imported one).
    original_a = synthetic_project.all_polygons[group_a][fp]
    group_b = "__seg_csv_gap_fill_test__"
    synthetic_project.all_polygons[group_a][fp] = dict(original_a, properties={"DBH_CM": 31.5})
    synthetic_project.all_polygons[group_b] = {fp: dict(original_a, name=group_b)}  # no 'properties' key
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group_a, group_b)
        save_dir = tmp_path / "seg_out3"
        save_dir.mkdir()
        _run_generate_segmentation_images(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "segmentation_labels.csv")
        by_group = {r["group_name"]: r for r in rows}
        assert group_a in by_group and group_b in by_group, (
            f"expected both polygons to produce a row, got groups: {list(by_group)}")
        assert by_group[group_a]["prop_DBH_CM"] == "31.5"
        assert by_group[group_b]["prop_DBH_CM"] == "", "gap not blank for the property-less polygon"
    finally:
        synthetic_project.all_polygons[group_a][fp] = original_a
        synthetic_project.all_polygons.pop(group_b, None)


def test_object_id_still_matches_the_mask_pixel_values(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """The invariant this widening must not disturb: object_id in the CSV
    is the exact integer label value burned into the mask file."""
    import numpy as np

    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    original = synthetic_project.all_polygons[group][fp]
    synthetic_project.all_polygons[group][fp] = dict(original, properties={"DBH_CM": 31.5})
    try:
        mgr = ml_manager_factory(synthetic_project)
        _select_group(mgr, group)
        save_dir = tmp_path / "seg_out4"
        save_dir.mkdir()
        _run_generate_segmentation_images(monkeypatch, mgr, save_dir)

        rows = _read_csv(save_dir / "segmentation_labels.csv")
        object_ids = {int(r["object_id"]) for r in rows}

        stem = os.path.splitext(os.path.basename(fp))[0]
        mask_png = save_dir / f"{stem}_mask.png"
        mask_tif = save_dir / f"{stem}_mask.tif"
        mask_path = mask_png if mask_png.exists() else mask_tif
        assert mask_path.exists(), "no mask file written"

        import cv2
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        mask_values = set(np.unique(mask).tolist()) - {0}
        assert mask_values == object_ids, (
            f"mask pixel values {mask_values} != CSV object_id values {object_ids}")
    finally:
        synthetic_project.all_polygons[group][fp] = original
