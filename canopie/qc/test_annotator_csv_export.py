"""
Annotator values in exports.

The values live in each polygon's ``properties``, so they leave with the
polygon like any other property: ``prop_<name>`` columns in the main CSV
(the foreground writer and the background ExportWorker) and in the ML
manager's CSVs, and shapefile attributes. Pinned here, through the real
entry points:

* a filled value and N/A (null -> blank cell) both reach every writer;
* a property with "Include in exports" off is left out of the copy each export
  works on, but stays on the live polygon.
"""
import csv
import os

import pytest

from canopie.annotator import AnnotationStore, KIND_CATEGORICAL, KIND_NUMERIC

pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _poly(name):
    return {'points': [[0, 0], [10, 0], [10, 10]], 'name': name, 'root': '1',
            'coordinates': {'latitude': 0.0, 'longitude': 0.0}, 'type': 'polygon',
            'coord_space': 'image', 'image_ref_size': {'w': 100, 'h': 100}}


@pytest.fixture()
def project_tab(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    pt = build_project_tab(str(tmp_path))
    pt.analysis_options = {}
    yield pt


def _fixture_fp(pt):
    for fm in pt.all_polygons.values():
        for fp in fm:
            return fp
    raise AssertionError("fixture project has no polygons")


def _rows(path, object_id):
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r.get("Object ID") == object_id]
    return fields, rows


def _annotate(pt, group, fp, store):
    """Fill values the way the panel does: index + the polygon's properties."""
    cover = store.add_property("flower cover", KIND_NUMERIC)
    stage = store.add_property("stage", KIND_CATEGORICAL, choices=["bud", "flower"])
    private = store.add_property("private", KIND_NUMERIC, export=False)
    rec = pt.all_polygons[group][fp]
    store.set_value(group, fp, cover, 45)
    store.set_value(group, fp, stage, "flower")
    store.set_value(group, fp, private, 9)
    for pid in (cover, stage, private):
        store.write_record(group, fp, rec, pid)
    pt._mark_polygon_dirty(group, fp)


def test_main_csv_both_writers_carry_annotator_values(project_tab, tmp_path):
    from canopie.annotator import attach_export_values, store_for
    from canopie.project_tab import ExportWorker

    pt = project_tab
    fp = _fixture_fp(pt)
    g = "__annot_csv__"
    pt.all_polygons[g] = {fp: _poly(g)}
    _annotate(pt, g, fp, store_for(pt))

    fg = pt.save_polygons_to_csv(options={"processing_params": ("sequential", 1, False)})
    assert fg and os.path.exists(fg)
    fields, rows = _rows(fg, g)
    assert {"prop_flower cover", "prop_stage"} <= set(fields)
    assert "prop_private" not in fields, "a property with 'Include in exports' off was exported"
    assert rows and all(r["prop_flower cover"] == "45.0" and r["prop_stage"] == "flower" for r in rows)
    assert pt.all_polygons[g][fp]["properties"]["private"] == 9.0, (
        "excluding it from the export removed it from the polygon")

    rec = attach_export_values(pt, g, fp, dict(pt.all_polygons[g][fp]))
    bg = str(tmp_path / "bg.csv")
    ExportWorker(pt, bg, {"stats": {"mean": True}}, [(g, fp, rec)],
                 {}, [], False, True, "sequential", 1).run()
    bg_fields, bg_rows = _rows(bg, g)
    assert {"prop_flower cover", "prop_stage"} <= set(bg_fields)
    assert "prop_private" not in bg_fields
    assert bg_rows and bg_rows[0]["prop_stage"] == "flower"


def test_ml_segmentation_csv_carries_annotator_values(synthetic_project, ml_manager_factory,
                                                      monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from .fixtures_manifest import fixture_image_path, get_fixture
    from .project_builder import polygon_group_name

    name = "rgb_8bit_untiled"
    fp = fixture_image_path(name)
    group = polygon_group_name(name, get_fixture(name)["polygon"]["name"])
    pt = synthetic_project
    saved_store = getattr(pt, "annotator_store", None)
    original = pt.all_polygons[group][fp]
    pt.all_polygons[group][fp] = dict(original)
    s = pt.annotator_store = AnnotationStore()
    try:
        cover = s.add_property("flower cover", KIND_NUMERIC)
        hidden = s.add_property("private", KIND_NUMERIC, export=False)
        rec = pt.all_polygons[group][fp]
        for pid, v in ((cover, 70), (hidden, 1)):
            s.set_value(group, fp, pid, v)
            s.write_record(group, fp, rec, pid)

        mgr = ml_manager_factory(pt)
        for i in range(mgr.list_widget.count()):
            it = mgr.list_widget.item(i)
            it.setSelected(it.text() == group)
        out = tmp_path / "seg"
        out.mkdir()
        monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: str(out)))
        mgr.generate_segmentation_images()

        with open(out / "segmentation_labels.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        assert rows, "no rows written"
        assert "prop_private" not in fields
        assert all(r.get("prop_flower cover") == "70.0" for r in rows)
        assert pt.all_polygons[group][fp]["properties"]["private"] == 1.0
    finally:
        pt.all_polygons[group][fp] = original
        pt.annotator_store = saved_store
