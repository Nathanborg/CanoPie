"""
The Annotator (canopie/annotator.py): custom per-polygon properties.

What these tests pin:

* SCOPE. A property applies to the whole project, one root or one image, and
  optionally to one label only. Nothing outside its scope is asked for it.
* RETROACTIVE. A property defined after the polygons exist covers all of them
  at once, and covers a polygon drawn later without anyone touching it.
  Applicability is worked out when asked, never stamped onto records.
* N/A. N/A stores null, counts as an answer (the polygon can still be
  complete), survives a JSON round trip as null, and disables the slider.
* COMPLETION. The per-polygon OK, the per-label "% filled" across images,
  the per-property counts and the to-do list all agree.
* PERSISTENCE. Definitions and values live in project.json's "annotator"
  key; Ctrl+Q (save_incremental) writes them and load_project reads them back
  into a fresh ProjectTab.
* UI. The Annotator opens from the tab, its tree reports completion, the
  Polygon Properties panel edits values, and a polygon clicked in a viewer
  (locked or not) opens in the panel with no unlock prompt.
"""
import json
import os

import pytest
from PyQt5 import QtCore, QtWidgets

from canopie.annotator import (
    MISSING, AnnotationStore, KIND_CATEGORICAL, KIND_NUMERIC, SCOPE_IMAGE,
    SCOPE_ROOT, PolygonPropertiesPanel, find_record, norm_fp, persist_to_project_json)

pytestmark = [pytest.mark.polygons]


def _rec(name, kind="polygon"):
    return {"name": name, "type": kind, "points": [(0, 0), (5, 0), (5, 5)]}


# Two roots, three images; labels "palm" (3 polygons) and "fig" (1).
ROOT = {"C:/p/r1/a.tif": "r1", "C:/p/r1/b.tif": "r1", "C:/p/r2/c.tif": "r2"}


def _project():
    return {
        "palm_1": {"C:/p/r1/a.tif": _rec("palm"), "C:/p/r2/c.tif": _rec("palm")},
        "palm_2": {"C:/p/r1/b.tif": _rec("palm")},
        "fig_1": {"C:/p/r1/a.tif": _rec("fig")},
        "mask": {"C:/p/r1/a.tif": dict(_rec("mask"), is_mask_polygon=True)},
    }


def root_of(fp):
    return ROOT.get(fp)


# ---------------------------------------------------------------- scope

def test_project_scope_applies_to_every_polygon_but_masks():
    s = AnnotationStore()
    s.add_property("flower cover", KIND_NUMERIC)
    polys = list(s.iter_polygons(_project(), root_of))
    assert len(polys) == 4, "image-editor mask polygons are not annotatable objects"
    assert all(s.status(g, f, lb, r) == (0, 1) for g, f, lb, r in polys)


def test_root_scope_only_asks_polygons_of_that_root():
    s = AnnotationStore()
    s.add_property("cover", KIND_NUMERIC, scope=SCOPE_ROOT, scope_value="r2")
    asked = {(g, f) for g, f, lb, r in s.iter_polygons(_project(), root_of)
             if s.status(g, f, lb, r)[1]}
    assert asked == {("palm_1", "C:/p/r2/c.tif")}


def test_image_scope_matches_either_path_spelling():
    """A shapefile import stores os.path.normpath; the viewer uses slashes."""
    s = AnnotationStore()
    s.add_property("cover", KIND_NUMERIC, scope=SCOPE_IMAGE,
                   scope_value=os.path.normpath("C:/p/r1/a.tif"))
    asked = {(g, f) for g, f, lb, r in s.iter_polygons(_project(), root_of)
             if s.status(g, f, lb, r)[1]}
    assert asked == {("palm_1", "C:/p/r1/a.tif"), ("fig_1", "C:/p/r1/a.tif")}


def test_label_restriction():
    s = AnnotationStore()
    s.add_property("fruit", KIND_CATEGORICAL, label="fig", choices=["green", "ripe"])
    asked = {g for g, f, lb, r in s.iter_polygons(_project(), root_of)
             if s.status(g, f, lb, r)[1]}
    assert asked == {"fig_1"}


def test_scoped_property_needs_a_target_and_names_are_unique():
    s = AnnotationStore()
    with pytest.raises(ValueError):
        s.add_property("x", scope=SCOPE_ROOT)            # no root given
    with pytest.raises(ValueError):
        s.add_property("y", KIND_CATEGORICAL, choices=[])
    s.add_property("Cover")
    with pytest.raises(ValueError):
        s.add_property("cover")


# ---------------------------------------------------------- retroactive

def test_new_property_covers_existing_and_later_polygons():
    project = _project()
    s = AnnotationStore()
    pid = s.add_property("cover", KIND_NUMERIC)

    pending = {(t["group"], t["filepath"]) for t in s.pending(project, root_of)}
    assert pending == {("palm_1", "C:/p/r1/a.tif"), ("palm_1", "C:/p/r2/c.tif"),
                       ("palm_2", "C:/p/r1/b.tif"), ("fig_1", "C:/p/r1/a.tif")}, (
        "a property added after the polygons must cover every existing one")

    # drawn afterwards: nothing written to it, it is asked anyway
    project["palm_3"] = {"C:/p/r2/c.tif": _rec("palm")}
    assert s.status("palm_3", "C:/p/r2/c.tif", "palm", "r2") == (0, 1)
    assert any(t["group"] == "palm_3" and t["missing_ids"] == [pid]
               for t in s.pending(project, root_of))
    assert "annotations" not in project["palm_3"]["C:/p/r2/c.tif"], (
        "applicability must be computed, not stamped onto the polygon record")


# ------------------------------------------------------------------ N/A

def test_na_is_null_counts_as_filled_and_round_trips():
    s = AnnotationStore()
    pid = s.add_property("cover", KIND_NUMERIC)
    s.set_na("fig_1", "C:/p/r1/a.tif", pid)
    assert s.get_value("fig_1", "C:/p/r1/a.tif", pid) is None
    assert s.is_complete("fig_1", "C:/p/r1/a.tif", "fig", "r1")

    blob = json.loads(json.dumps(s.to_dict()))
    assert blob["values"]["fig_1"][norm_fp("C:/p/r1/a.tif")][pid] is None
    back = AnnotationStore(blob)
    assert back.get_value("fig_1", "C:/p/r1/a.tif", pid) is None
    assert back.get_value("palm_2", "C:/p/r1/b.tif", pid) is MISSING


def test_values_are_validated():
    s = AnnotationStore()
    num = s.add_property("cover", KIND_NUMERIC, minimum=0, maximum=100)
    cat = s.add_property("stage", KIND_CATEGORICAL, choices=["bud", "flower"])
    tags = s.add_property("visitors", KIND_CATEGORICAL, choices=["bee", "bird"], multi=True)
    s.set_value("g", "f.tif", num, 140)
    assert s.get_value("g", "f.tif", num) == 100.0, "numbers clamp to the range"
    with pytest.raises(ValueError):
        s.set_value("g", "f.tif", cat, "fruit")
    s.set_value("g", "f.tif", tags, ["bird", "bee"])
    assert s.get_value("g", "f.tif", tags) == ["bee", "bird"], "tags keep the choice order"
    s.set_value("g", "f.tif", tags, [])
    assert s.get_value("g", "f.tif", tags) is MISSING, "no tag picked is pending, not an answer"


def test_rename_keeps_values_and_remove_drops_them():
    s = AnnotationStore()
    pid = s.add_property("cover")
    s.set_value("g", "f.tif", pid, 3)
    s.update_property(pid, name="flower cover")
    assert s.get_value("g", "f.tif", pid) == 3.0
    s.remove_property(pid)
    assert s.properties == []
    assert s.get_value("g", "f.tif", pid) is MISSING


# ------------------------------------------------------------ completion

def test_completion_by_label_and_todo_list():
    project = _project()
    s = AnnotationStore()
    cover = s.add_property("cover", KIND_NUMERIC)
    fruit = s.add_property("fruit", KIND_CATEGORICAL, label="fig", choices=["green", "ripe"])

    s.set_value("palm_1", "C:/p/r1/a.tif", cover, 10)
    s.set_value("palm_1", "C:/p/r2/c.tif", cover, 20)
    s.set_value("palm_2", "C:/p/r1/b.tif", cover, 30)
    s.set_value("fig_1", "C:/p/r1/a.tif", cover, 40)       # fig: 1 of 2 values

    comp = s.completion_by_label(project, root_of)
    assert comp["palm"]["pct"] == pytest.approx(100.0)
    assert comp["palm"]["polygons"] == 3 and comp["palm"]["complete"] == 3
    assert comp["fig"]["pct"] == pytest.approx(50.0)
    assert comp["fig"]["complete"] == 0

    todo = s.pending(project, root_of)
    assert [(t["group"], t["missing"]) for t in todo] == [("fig_1", ["fruit"])]

    st = s.property_stats(project, root_of)
    assert st[cover] == {"applies": 4, "filled": 4, "na": 0, "missing": 0}
    assert st[fruit] == {"applies": 1, "filled": 0, "na": 0, "missing": 1}

    s.set_na("fig_1", "C:/p/r1/a.tif", fruit)
    assert s.pending(project, root_of) == []
    assert s.completion_by_label(project, root_of)["fig"]["pct"] == pytest.approx(100.0)
    assert s.property_stats(project, root_of)[fruit]["na"] == 1


def test_prune_drops_values_of_deleted_polygons():
    project = _project()
    s = AnnotationStore()
    pid = s.add_property("cover")
    s.set_value("palm_2", "C:/p/r1/b.tif", pid, 5)
    s.set_value("gone", "C:/p/r1/b.tif", pid, 5)
    assert s.prune(project) == 1
    assert s.get_value("palm_2", "C:/p/r1/b.tif", pid) == 5.0


def test_persist_keeps_the_rest_of_project_json(tmp_path):
    (tmp_path / "project.json").write_text(json.dumps({"mode": "rgb_only", "root_offset": 3}))
    s = AnnotationStore()
    s.add_property("cover")
    tab = type("Tab", (), {"annotator_store": s, "project_folder": str(tmp_path)})()
    assert persist_to_project_json(tab)
    data = json.loads((tmp_path / "project.json").read_text())
    assert data["mode"] == "rgb_only" and data["root_offset"] == 3
    assert data["annotator"]["properties"][0]["name"] == "cover"


# ------------------------------------------- values on the polygon itself

def test_values_round_trip_through_the_polygon_properties():
    s = AnnotationStore()
    cover = s.add_property("flower cover", KIND_NUMERIC)
    stage = s.add_property("stage", KIND_CATEGORICAL, choices=["bud", "flower"])
    vis = s.add_property("visitors", KIND_CATEGORICAL, choices=["bee", "bird"], multi=True)
    rec = {"name": "palm", "properties": {"species": "Ceiba"}}
    s.set_value("g", "a.tif", cover, 45)
    s.set_na("g", "a.tif", stage)
    s.set_value("g", "a.tif", vis, ["bird", "bee"])
    for pid in (cover, stage, vis):
        s.write_record("g", "a.tif", rec, pid)
    assert rec["properties"] == {"species": "Ceiba", "flower cover": 45.0,
                                 "stage": None, "visitors": "bee;bird"}

    fresh = AnnotationStore(s.to_dict())
    fresh.values = {}                                   # index lost: the polygon is the truth
    assert fresh.sync_record("g", "a.tif", rec) == "synced"
    assert fresh.get_value("g", "a.tif", cover) == 45.0
    assert fresh.get_value("g", "a.tif", stage) is None
    assert fresh.get_value("g", "a.tif", vis) == ["bee", "bird"]

    s.clear_value("g", "a.tif", vis)
    s.write_record("g", "a.tif", rec, vis)
    assert "visitors" not in rec["properties"], "pending removes the key"


def test_a_property_named_like_a_shapefile_field_reads_that_field():
    s = AnnotationStore()
    dbh = s.add_property("dbh", KIND_NUMERIC, maximum=500)
    sp = s.add_property("species", KIND_NUMERIC)       # text in a numeric property
    rec = {"properties": {"dbh": "35.5", "species": "Ceiba"}}
    s.sync_record("g", "a.tif", rec)
    assert s.get_value("g", "a.tif", dbh) == 35.5
    assert s.get_value("g", "a.tif", sp) is MISSING, "a value the property cannot hold reads as pending"
    assert rec["properties"]["species"] == "Ceiba", "and it is left on the polygon untouched"


def test_unread_lazy_records_use_the_index_until_forced():
    from canopie.polygon_lod import LazyPolygonRecord
    s = AnnotationStore()
    pid = s.add_property("cover")
    s.set_value("g", "a.tif", pid, 7)
    lazy = LazyPolygonRecord({"name": "palm"}, "does_not_exist.json")
    assert s.sync_record("g", "a.tif", lazy) == "skipped"
    assert not lazy.is_materialised, "stats must not page a lazy polygon in"
    assert s.get_value("g", "a.tif", pid) == 7.0


def test_version_1_project_values_move_onto_the_polygons():
    blob = {"version": 1, "properties": [{"id": "p1", "name": "cover", "kind": "numeric"}],
            "values": {"g": {norm_fp("a.tif"): {"p1": 12}}}}
    s = AnnotationStore(blob)
    assert s.legacy
    rec = {"properties": {}}
    assert s.sync_record("g", "a.tif", rec) == "migrated"
    assert rec["properties"] == {"cover": 12}


def test_unticked_properties_are_taken_out_of_export_copies():
    from canopie.annotator import attach_export_values
    s = AnnotationStore()
    s.add_property("cover")
    s.add_property("private", export=False)
    live = {"properties": {"cover": 3.0, "private": 9.0, "species": "Ceiba"}}
    tab = type("Tab", (), {"annotator_store": s, "get_root_by_filepath": lambda self, fp: None})()
    copy = attach_export_values(tab, "g", "a.tif", dict(live))
    assert copy["properties"] == {"cover": 3.0, "species": "Ceiba"}
    assert live["properties"]["private"] == 9.0, "the live record must not lose the value"
    assert AnnotationStore(s.to_dict()).properties[1]["export"] is False


# ----------------------------------------------------------- panel (UI)

def test_panel_slider_and_na_checkbox(qapp):
    s = AnnotationStore()
    pid = s.add_property("cover", KIND_NUMERIC, minimum=0, maximum=10, step=1)
    panel = PolygonPropertiesPanel(s)
    panel.show_polygon("g", "f.tif", "palm", "r1")
    row = panel.rows[pid]
    assert s.get_value("g", "f.tif", pid) is MISSING
    assert "0/1" in panel.status.text()

    row.slider.setValue(7)
    assert s.get_value("g", "f.tif", pid) == 7.0
    assert "OK" in panel.status.text()

    row.na.setChecked(True)
    assert s.get_value("g", "f.tif", pid) is None, "N/A must save null"
    assert not row.slider.isEnabled(), "N/A must disable the slider"
    assert "OK" in panel.status.text(), "N/A is an answer"

    row.na.setChecked(False)
    assert row.slider.isEnabled()
    assert s.get_value("g", "f.tif", pid) == 7.0

    # reopening the polygon shows what was stored
    row.na.setChecked(True)
    panel.show_polygon("g", "f.tif", "palm", "r1")
    assert panel.rows[pid].na.isChecked() and not panel.rows[pid].slider.isEnabled()
    panel.deleteLater()


def test_panel_dropdown_and_tags(qapp):
    s = AnnotationStore()
    stage = s.add_property("stage", KIND_CATEGORICAL, choices=["bud", "flower"])
    vis = s.add_property("visitors", KIND_CATEGORICAL, choices=["bee", "bird"], multi=True)
    panel = PolygonPropertiesPanel(s)
    panel.show_polygon("g", "f.tif", "palm", None)
    combo = panel.rows[stage].combo
    combo.setCurrentIndex(combo.findData("flower"))
    assert s.get_value("g", "f.tif", stage) == "flower"
    combo.setCurrentIndex(0)
    assert s.get_value("g", "f.tif", stage) is MISSING, "the blank entry means pending"
    chips = {b.text(): b for b in panel.rows[vis].chips}
    chips["bird"].setChecked(True)
    assert s.get_value("g", "f.tif", vis) == ["bird"]
    panel.deleteLater()


# --------------------------------------------- real ProjectTab end to end

@pytest.fixture()
def project_tab(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    pt = build_project_tab(str(tmp_path))
    yield pt
    dlg = getattr(pt, "annotator", None)
    if dlg is not None:
        dlg.close()
        if dlg.dock is not None:
            dlg.dock.deleteLater()
        dlg.deleteLater()


def _first_polygon(pt):
    from canopie.annotator import AnnotationStore as S
    for g, f, lb, r in S.iter_polygons(pt.all_polygons, pt.get_root_by_filepath):
        return g, f, lb, r
    raise AssertionError("fixture project has no polygons")


def test_annotator_opens_reports_and_persists(project_tab, tmp_path):
    from canopie.project_tab import ProjectTab
    pt = project_tab
    assert pt.annotatorAct.text() == "Annotator"
    assert pt.annotatorAct not in pt.toolbar.actions(), "the Annotator belongs under Modules"
    modules = [w.menu() for w in pt.toolbar.findChildren(QtWidgets.QToolButton)
               if w.text() == "Modules" and w.menu() is not None]
    assert modules and pt.annotatorAct in modules[0].actions(), "Annotator missing from Modules"
    dlg = pt.show_annotator()
    assert dlg.isVisible()
    pid = dlg.store.add_property("cover", KIND_NUMERIC)
    tags = dlg.store.add_property("visitors", KIND_CATEGORICAL, choices=["bee", "bird"], multi=True)
    dlg.refresh()

    labels = [dlg.tree.topLevelItem(i) for i in range(dlg.tree.topLevelItemCount())]
    assert labels and all(it.text(1) == "0% filled" for it in labels)

    g, f, lb, r = _first_polygon(pt)
    dlg.open_polygon(g, f, navigate=True)
    assert dlg.panel.group == g and pid in dlg.panel.rows
    dlg.panel.rows[pid].na.setChecked(True)
    {b.text(): b for b in dlg.panel.rows[tags].chips}["bird"].setChecked(True)
    assert pt.annotator_store.get_value(g, f, pid) is None

    # THE values live on the polygon itself -- what Edit properties shows,
    # what the shapefile DBF and the prop_* CSV columns are built from.
    props = pt.all_polygons[g][f]["properties"]
    assert "cover" in props and props["cover"] is None, "N/A must be null in the polygon's properties"
    assert props["visitors"] == "bird"
    row = dlg._tree_rows[(g, norm_fp(f))]
    assert "OK" in row.text(1), "the polygon's row did not get its OK mark"
    n = len(dlg._label_members[lb])
    assert row.parent().text(1) == f"{100.0 / n:.0f}% filled", (
        "the label's % filled did not follow the edit")

    pt.save_incremental(show_status=False)
    data = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    assert data["annotator"]["properties"][0]["name"] == "cover"
    sidecars = [p for p in (tmp_path / "polygons").glob("*_polygons.json")
                if json.loads(p.read_text(encoding="utf-8")).get("properties", {}).get("visitors") == "bird"]
    assert sidecars, "the value did not reach the polygon's sidecar JSON"

    pt2 = ProjectTab("qc2", QtWidgets.QTabWidget())
    pt2.load_project(str(tmp_path))
    st = pt2.annotator_store
    assert st is not None and st.property(pid)["name"] == "cover"
    assert st.get_value(g, f, pid) is None, "N/A (null) did not survive save + load"
    assert st.status(g, f, lb, r) == (2, 2)
    key, rec = find_record(pt2.all_polygons, g, f)
    assert rec["properties"]["visitors"] == "bird"


def test_edit_properties_dialog_and_annotator_are_one_thing(project_tab, monkeypatch):
    """Typed in the viewer's Edit properties dialog -> shows in the panel;
    N/A (null) survives a trip through that dialog untouched."""
    from canopie import image_viewer as iv
    from PyQt5 import QtGui
    pt = project_tab
    dlg = pt.show_annotator()
    cover = dlg.store.add_property("cover", KIND_NUMERIC)
    stage = dlg.store.add_property("stage", KIND_CATEGORICAL, choices=["bud", "flower"])
    g, f, lb, r = _first_polygon(pt)
    dlg.open_polygon(g, f, navigate=False)
    dlg.panel.rows[stage].na.setChecked(True)          # null on the polygon

    viewer = iv.ImageViewer()
    viewer.image_data = type("D", (), {"filepath": f, "image": None})()
    viewer._scene.clear()
    viewer._image = viewer._scene.addPixmap(QtGui.QPixmap(64, 64))
    viewer.polygons = []
    viewer.setParent(pt)                               # _find_project_owner walks parents
    viewer.add_polygon_to_scene(QtGui.QPolygonF([
        QtCore.QPointF(2, 2), QtCore.QPointF(12, 2), QtCore.QPointF(12, 12)]), name=g)
    item = viewer.get_all_polygons()[0]

    seen = {}

    def fake_exec(self):
        seen["shown"] = {self.table.item(i, 0).text(): self.table.item(i, 1).text()
                         for i in range(self.table.rowCount())}
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).text() == "cover":
                self.table.item(i, 1).setText("35")
                break
        else:
            self._append_row("cover", "35")
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(iv.PolygonPropertiesDialog, "exec_", fake_exec)
    monkeypatch.setattr(pt, "save_incremental", lambda *a, **k: None)
    viewer.edit_polygon_properties(item)

    assert "stage" in seen["shown"], "the Annotator value is not in Edit properties"
    assert pt.all_polygons[g][f]["properties"]["stage"] is None, (
        "N/A (null) became an empty string after Edit properties -> Save")
    assert dlg.store.get_value(g, f, cover) == 35.0, "a value typed in Edit properties did not reach the Annotator"
    assert dlg.panel.rows[cover].value_label.text() == "35"
    viewer.setParent(None)
    viewer.deleteLater()


def _add_image_of_same_canopy(pt, g, f):
    """Put the same polygon group on a second image (a phenocam root has one
    canopy in every frame)."""
    other = next(fp for fm in pt.all_polygons.values() for fp in fm if norm_fp(fp) != norm_fp(f))
    rec = dict(pt.all_polygons[g][f])
    rec["properties"] = {}
    pt.all_polygons[g][other] = rec
    return other


def test_current_canopy_is_highlighted_and_navigable(project_tab):
    pt = project_tab
    dlg = pt.show_annotator()
    pid = dlg.store.add_property("cover", KIND_NUMERIC)
    g, f, lb, r = _first_polygon(pt)
    other = _add_image_of_same_canopy(pt, g, f)
    dlg.refresh()

    dlg.open_polygon(g, f, navigate=False)
    row = dlg._tree_rows[(g, norm_fp(f))]
    sib = dlg._tree_rows[(g, norm_fp(other))]
    assert dlg.tree.currentItem() is row, "the polygon being annotated is not the list's current row"
    assert row.font(0).bold() and row.parent().isExpanded()
    assert row.background(0).color().alpha() > sib.background(0).color().alpha() > 0, (
        "current row strongest tint, the same canopy in other images a lighter one")
    assert not dlg._tree_rows[next(k for k in dlg._tree_rows if k[0] != g)].font(0).bold()
    assert dlg.panel.where.text() in ("1 / 2", "2 / 2")

    old_rows = list(dlg.panel.rows.values())
    dlg.step_same_canopy(1)
    assert norm_fp(dlg.panel.filepath) == norm_fp(other), "the arrow did not move to the canopy's next image"
    assert dlg.tree.currentItem() is sib
    assert all(r.isHidden() and r.parent() is None for r in old_rows), (
        "the previous polygon's rows are still in the panel (painted under the new ones)")
    on_panel = [w for w in dlg.panel.form_host.findChildren(QtWidgets.QWidget)
                if type(w).__name__ == "_NumericRow"]
    assert len(on_panel) == 1

    # Next to fill: lands on a pending polygon, never on a finished one
    dlg.store.set_value(g, other, pid, 5)
    dlg.go_next_pending()
    assert dlg.store.get_value(dlg.panel.group, dlg.panel.filepath, pid) is MISSING


def test_viewer_of_the_annotated_image_is_marked(project_tab):
    from canopie import image_viewer as iv
    from PyQt5 import QtGui
    pt = project_tab
    dlg = pt.show_annotator()
    dlg.store.add_property("cover", KIND_NUMERIC)
    g, f, lb, r = _first_polygon(pt)
    other = _add_image_of_same_canopy(pt, g, f)
    saved = pt.viewer_widgets
    widgets = []
    for fp in (f, other):
        v = iv.ImageViewer()
        v.image_data = type("D", (), {"filepath": fp, "image": None})()
        head = QtWidgets.QLabel(os.path.basename(fp))
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.addWidget(head)
        lay.addWidget(v)
        widgets.append({"container": box, "viewer": v, "image_data": v.image_data})
    pt.viewer_widgets = widgets
    try:
        dlg.open_polygon(g, other, navigate=False)
        assert "border" in widgets[1]["viewer"].styleSheet(), "the annotated image's viewer is not framed"
        assert widgets[0]["viewer"].styleSheet() == "", "a different image's viewer got the frame"
        dlg.open_polygon(g, f, navigate=False)
        assert widgets[1]["viewer"].styleSheet() == "", "the old frame was not taken off"
        assert "border" in widgets[0]["viewer"].styleSheet()
        dlg.close()
        assert widgets[0]["viewer"].styleSheet() == "", "closing the Annotator must clear the mark"
    finally:
        pt.viewer_widgets = saved


def test_viewer_click_opens_polygon_in_panel_even_when_locked(project_tab, qapp, monkeypatch):
    from canopie import image_viewer as iv
    pt = project_tab
    dlg = pt.show_annotator()
    dlg.store.add_property("cover", KIND_NUMERIC)
    # A real viewer showing one of the project's images, wired by the same
    # ProjectTab method every viewer goes through.
    from PyQt5 import QtGui
    g, f, _lb, _r = _first_polygon(pt)
    viewer = iv.ImageViewer()
    viewer.image_data = type("D", (), {"filepath": f, "image": None})()
    viewer._scene.clear()
    viewer._image = viewer._scene.addPixmap(QtGui.QPixmap(64, 64))
    viewer.polygons = []
    pt._wire_viewer_for_inspection(viewer)
    assert viewer.polygon_click_listener is not None, "ProjectTab did not wire the viewer"
    viewer.add_polygon_to_scene(QtGui.QPolygonF([
        QtCore.QPointF(2, 2), QtCore.QPointF(12, 2), QtCore.QPointF(12, 12)]), name=g)
    item = viewer.get_all_polygons()[0]

    item.is_locked = True
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_",
                        lambda *a: pytest.fail("unlock prompt shown while the Annotator is open"))
    assert iv._notify_polygon_click(item) is True
    qapp.processEvents()
    assert dlg.panel.group == item.name
    assert norm_fp(dlg.panel.filepath) == norm_fp(viewer.image_data.filepath)
    assert item.is_locked, "picking a locked polygon for its properties must not unlock it"

    dlg.close()
    assert iv._notify_polygon_click(item) is False, "a closed Annotator must not take clicks"
    viewer._scene.clear()
    viewer.deleteLater()
