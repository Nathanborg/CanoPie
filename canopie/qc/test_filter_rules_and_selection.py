"""
Filter & Color rules: property / value suggestions, hidden attributes, speed;
selection that survives zooming out; the last_modified stamp.

* PolygonAttributeIndex: every polygon's properties plus hidden area,
  perimeter, last_modified (area_px / perimeter_px always computed). A lazily
  loaded crown is read from its file WITHOUT touching the live record, and an
  entry is reused until the polygon changes -- the second Apply reads nothing.
* The catalog suggests property names (hidden first) and values: the
  commonest labels, or min / quartiles / max for numbers.
* RuleRow builds valid expressions (names with spaces -> props["..."],
  contains, numbers vs text) and parses saved ones back.
* Zoomed out, the tiles hide every item and Qt cannot select a hidden item:
  select_polygon_item shows it first, a selected item stays drawn over the
  tiles, and a deselected one goes back to them.
* ProjectTab stamps record["last_modified"] on every edit (dirty marking, the
  drag command) and the sidecar keeps it.
"""
import json
import os
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.polygon_manager import (HIDDEN_FILTER_ATTRIBUTES, PolygonAttributeIndex,
                                     PolygonFilterWorker, RuleRow, sample_catalog)

pytestmark = [pytest.mark.polygons]

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def _rec(name, props=None, pts=SQ, **kw):
    d = {"name": name, "type": "polygon", "points": list(pts), "properties": dict(props or {})}
    d.update(kw)
    return d


# ------------------------------------------------------------------ index

def test_hidden_attributes_and_suggestions():
    snap = {"a": {"f.tif": _rec("a", {"species": "oleifera", "dbh": "35"})},
            "b": {"f.tif": _rec("b", {"species": "oleifera", "dbh": 12.0})},
            "c": {"f.tif": _rec("c", {"species": "ekmanii", "area": 999.0}, last_modified="2026-11-01 10:00:00")}}
    idx = PolygonAttributeIndex()
    entries, computed = idx.refresh(snap)
    ns_a = entries[("a", "f.tif")][1]
    assert ns_a["area"] == pytest.approx(100.0) and ns_a["perimeter"] == pytest.approx(40.0)
    assert ns_a["dbh"] == 35.0, "numeric text must compare as a number"
    assert entries[("c", "f.tif")][1]["area"] == 999.0, "a polygon's own area property wins"
    assert entries[("c", "f.tif")][1]["area_px"] == pytest.approx(100.0)
    assert entries[("c", "f.tif")][1]["last_modified"] == "2026-11-01 10:00:00"
    assert ("a", "f.tif") in computed and ("c", "f.tif") not in computed

    cat = idx.catalog()
    names = list(cat)
    for h in ("area", "perimeter", "last_modified"):
        assert h in names, f"hidden attribute {h!r} missing from the property list"
    assert names.index("area") < names.index("species"), "hidden attributes are listed first"
    assert cat["species"][0] == "oleifera", "the commonest label comes first"
    assert set(cat["species"]) == {"oleifera", "ekmanii"}
    assert cat["dbh"] == sorted(cat["dbh"]) and len(cat["dbh"]) == 5, "numbers: min, quartiles, max"


def test_lazy_crowns_are_read_from_file_without_touching_the_record(tmp_path):
    from canopie.polygon_lod import LazyPolygonRecord, dump_sidecar
    src = tmp_path / "g_img_polygons.json"
    dump_sidecar(str(src), {"points": SQ, "name": "g", "properties": {"species": "pine"},
                            "last_modified": "2026-10-10 08:00:00"})
    lazy = LazyPolygonRecord({"name": "g"}, str(src))
    entries, _ = PolygonAttributeIndex().refresh({"g": {"img.tif": lazy}})
    ns = entries[("g", "img.tif")][1]
    assert ns["species"] == "pine" and ns["area_px"] == pytest.approx(100.0)
    assert ns["last_modified"] == "2026-10-10 08:00:00"
    assert not lazy.is_materialised, "the live record was paged in by the index"


def test_second_apply_reads_nothing_and_is_fast(monkeypatch):
    snap = {f"g{i}": {"f.tif": _rec(f"g{i}", {"species": ["a", "b", "c"][i % 3], "dbh": i % 50})}
            for i in range(5000)}
    idx = PolygonAttributeIndex()
    rules = [{"expression": "species == 'b' and dbh > 20", "color": (1, 2, 3)}]

    def run():
        w = PolygonFilterWorker(snap, rules, index=idx)
        got = {}
        w.finished.connect(lambda sm, ca, zm: got.update(sm=sm))
        t0 = time.perf_counter()
        w.run()
        return got["sm"], time.perf_counter() - t0

    sm, _t_first = run()
    assert sum(1 for s in sm.values() if s["color"] is not None) == sum(
        1 for i in range(5000) if i % 3 == 1 and i % 50 > 20)
    import canopie.polygon_manager as pmod
    monkeypatch.setattr(pmod, "_area_perimeter", lambda pts: pytest.fail("geometry re-read on the 2nd Apply"))
    sm2, t_second = run()
    assert sm2.keys() == sm.keys()
    assert t_second < 1.0, f"re-applying a filter to 5000 polygons took {t_second:.2f}s"


def test_an_edited_polygon_is_re_indexed():
    rec = _rec("a", {"species": "oak"})
    idx = PolygonAttributeIndex()
    idx.refresh({"a": {"f": rec}})
    rec["properties"] = {"species": "pine"}            # edits replace the dict
    entries, _ = idx.refresh({"a": {"f": rec}})
    assert entries[("a", "f")][1]["species"] == "pine"


def test_sample_catalog_is_quick_and_has_the_hidden_names():
    allp = {f"g{i}": {"f.tif": _rec(f"g{i}", {"species": "x"})} for i in range(2000)}
    t0 = time.perf_counter()
    cat = sample_catalog(allp)
    assert time.perf_counter() - t0 < 0.5
    assert "area" in cat and "species" in cat


# ------------------------------------------------------------- rule editor

def test_rule_row_builds_and_parses_expressions(qapp):
    cat = {"area": [1.0, 5.0], "species": ["oleifera", "ekmanii"], "flower cover": [0.0, 50.0]}
    r = RuleRow(cat)
    r.prop.setEditText("species")
    r.op.setCurrentText("==")
    assert [r.value.itemText(i) for i in range(r.value.count())] == ["oleifera", "ekmanii"], (
        "the value list must suggest the property's labels")
    r.value.setEditText("oleifera")
    assert r.expression() == "species == 'oleifera'"

    r.prop.setEditText("flower cover")
    r.op.setCurrentText(">")
    r.value.setEditText("20")
    assert r.expression() == "props['flower cover'] > 20"

    r.prop.setEditText("species")
    r.op.setCurrentText("contains")
    r.value.setEditText("ole")
    assert r.expression() == "'ole' in str(species)"

    r2 = RuleRow(cat)
    r2.set_expression('species == "ekmanii"')
    assert (r2.prop.currentText(), r2.op.currentText(), r2.value.currentText()) == ("species", "==", "ekmanii")
    r2.set_expression("area > 5 and species == 'x'")
    assert r2.op.currentText() == "expr" and r2.expression() == "area > 5 and species == 'x'"


def test_rule_expressions_evaluate_against_the_index():
    snap = {"a": {"f": _rec("a", {"species": "oleifera", "flower cover": 60})},
            "b": {"f": _rec("b", {"species": "ekmanii", "flower cover": 5})}}
    for expr, want in (("props['flower cover'] > 20", {"a"}),
                       ("'ole' in str(species)", {"a"}),
                       ("perimeter > 39", {"a", "b"}),
                       ("last_modified >= ''", {"a", "b"})):
        w = PolygonFilterWorker(snap, [{"expression": expr, "color": (9, 9, 9)}])
        got = {}
        w.finished.connect(lambda sm, ca, zm: got.update(sm=sm))
        w.run()
        assert {g for (g, _f), s in got["sm"].items() if s["color"] is not None} == want, expr


def test_manager_rule_rows_use_suggestions_and_apply_does_not_save(qapp, monkeypatch):
    from canopie.polygon_manager import PolygonManager

    class _Owner(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.all_polygons = {"a": {"f": _rec("a", {"species": "oleifera"})}}
            self.project_folder = None
            self.viewer_widgets = []

        def save_incremental(self, *a, **k):
            pytest.fail("Apply saved the project (it used to rewrite every polygon's area)")

    owner = _Owner()
    pm = PolygonManager(owner)
    pm._add_filter_rule_row("")
    editor = pm.filter_rules_table.cellWidget(0, 0)
    assert isinstance(editor, RuleRow)
    names = [editor.prop.itemText(i) for i in range(editor.prop.count())]
    assert "species" in names and "area" in names
    editor.prop.setEditText("species")
    editor.op.setCurrentText("==")
    editor.value.setEditText("oleifera")
    assert pm._collect_filter_rules()[0]["expression"] == "species == 'oleifera'"
    pm._on_filter_finished({("a", "f"): {"visible": True, "color": None}}, {("a", "f"): 100.0}, [])
    assert "area" not in owner.all_polygons["a"]["f"]["properties"], "computed area written into properties"
    # let the background index finish before tearing the dialog down
    t0 = time.time()
    while (pm._index_running or pm._active_filter_threads) and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    assert not pm._index_running
    assert "species" in [editor.prop.itemText(i) for i in range(editor.prop.count())], (
        "the full index did not reach the rule row")
    pm.deleteLater()
    owner.deleteLater()
    qapp.processEvents()


# ------------------------------------------------- selection while zoomed out

@pytest.fixture
def zoomed_out_viewer(qapp, monkeypatch):
    from canopie.image_viewer import ImageViewer, _ZoomBar
    monkeypatch.setattr(_ZoomBar, "_zoom_sync_enabled", False)
    v = ImageViewer()
    v.resize(300, 300)
    pm = QtGui.QPixmap(4000, 4000)
    pm.fill(QtCore.Qt.gray)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v._empty = False                       # as after a real image load
    v.polygons = []
    n = v._BATCH_MIN_POLYGONS + 10
    for i in range(n):
        x, y = 40 * (i % 90), 40 * (i // 90)
        v.add_polygon_to_scene(QtGui.QPolygonF([QtCore.QPointF(x, y), QtCore.QPointF(x + 20, y),
                                                QtCore.QPointF(x + 20, y + 20)]), name=f"p{i}")
    v.show()
    v.resetTransform()
    v.scale(0.02, 0.02)
    v._sync_polygon_batch()
    qapp.processEvents()
    assert v._batch_active, "precondition: zoomed out, tiles drawn"
    yield v
    v.hide()
    v._scene.clear()
    v.deleteLater()


def test_selecting_while_zoomed_out_works_and_survives(zoomed_out_viewer, qapp):
    v = zoomed_out_viewer
    it = v.get_all_polygons()[5]
    assert not it.isVisible()
    it.setSelected(True)
    assert not it.isSelected(), "Qt refuses to select a hidden item -- the reported bug"
    v.select_polygon_item(it, True)
    assert it.isSelected() and it.isVisible(), "selection while zoomed out did nothing"

    v.resetTransform()
    v._sync_polygon_batch()                # zoom in ...
    v.scale(0.02, 0.02)
    v._sync_polygon_batch()                # ... and out again
    assert it.isSelected(), "zooming out wiped the selection"

    it.setSelected(False)
    qapp.processEvents()
    assert not it.isVisible(), "a deselected crown must go back to the tile"


def test_polygon_manager_selects_zoomed_out_items(zoomed_out_viewer, qapp):
    from canopie.polygon_manager import PolygonManager

    class _Owner(QtWidgets.QWidget):
        all_polygons = {}
        viewer_widgets = []
    owner = _Owner()
    owner.viewer_widgets = [{"viewer": zoomed_out_viewer, "image_data": None}]
    pm = PolygonManager(owner)
    pm._select_group_in_viewers("p7", center=False)
    it = next(i for i in zoomed_out_viewer.get_all_polygons() if i.name == "p7")
    assert it.isSelected()
    pm.deleteLater()
    owner.deleteLater()


# ------------------------------------------------------------ last_modified

def test_edits_stamp_last_modified_and_the_sidecar_keeps_it(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    pt = build_project_tab(str(tmp_path))
    g = next(iter(pt.all_polygons))
    fp = next(iter(pt.all_polygons[g]))
    rec = pt.all_polygons[g][fp]
    before = time.strftime("%Y-%m-%d %H:%M:%S")
    pt._mark_polygon_dirty(g, fp)
    assert rec.get("last_modified", "") >= before, "an edit did not stamp last_modified"
    payload = pt._sidecar_payload(g, fp, rec)
    assert payload.get("last_modified") == rec["last_modified"]
    assert "last_modified" not in (rec.get("properties") or {}), "it must stay out of the properties"


def test_index_cache_skips_unchanged_files_next_session(tmp_path, monkeypatch):
    """The first filter of a session used to parse every sidecar (BCI: ~4.8 s).
    The index keeps a small cache next to them, checked by mtime + size."""
    from canopie.polygon_lod import LazyPolygonRecord, dump_sidecar
    import canopie.polygon_lod as lod
    files = []
    for i in range(3):
        src = tmp_path / f"g{i}_img_polygons.json"
        dump_sidecar(str(src), {"points": SQ, "name": f"g{i}", "properties": {"species": f"sp{i}"}})
        files.append(src)
    snap = lambda: {f"g{i}": {"img.tif": LazyPolygonRecord({"name": f"g{i}"}, str(files[i]))} for i in range(3)}
    PolygonAttributeIndex().refresh(snap(), polygons_dir=str(tmp_path))
    assert (tmp_path / PolygonAttributeIndex.CACHE_NAME).exists()
    assert not list(tmp_path.glob("*_filter_index*_polygons.json")), "the cache must not look like a polygon file"

    real = lod.read_sidecar
    reads = []
    monkeypatch.setattr(lod, "read_sidecar", lambda p: reads.append(p) or real(p))
    entries, _ = PolygonAttributeIndex().refresh(snap(), polygons_dir=str(tmp_path))   # a new session
    assert reads == [], "unchanged polygon files were parsed again"
    assert entries[("g1", "img.tif")][1]["species"] == "sp1"

    time.sleep(0.02)
    dump_sidecar(str(files[1]), {"points": SQ, "name": "g1", "properties": {"species": "changed"}})
    os.utime(files[1], None)
    entries, _ = PolygonAttributeIndex().refresh(snap(), polygons_dir=str(tmp_path))
    assert [os.path.basename(p) for p in reads] == ["g1_img_polygons.json"], "only the changed file is read"
    assert entries[("g1", "img.tif")][1]["species"] == "changed"


# ------------------------------------- canvas clicks / remnants, zoomed out

def _crown_centre_on_screen(v, it):
    """The crown's centroid (a triangle's bbox centre lies ON its edge)."""
    pts = it.mapToScene(it.polygon)
    cx = sum(p.x() for p in pts) / len(pts)
    cy = sum(p.y() for p in pts) / len(pts)
    return v.mapFromScene(QtCore.QPointF(cx, cy))


def test_a_canvas_click_selects_a_crown_while_zoomed_out(zoomed_out_viewer, qapp):
    from PyQt5.QtTest import QTest
    v = zoomed_out_viewer
    z = v._batch_scale_threshold() * 0.9  # just zoomed out enough for the tiles
    v.resetTransform()
    v.scale(z, z)
    v._sync_polygon_batch()
    qapp.processEvents()
    assert v._batch_active
    by = {i.name: i for i in v.get_all_polygons()}
    a, b = by["p3"], by["p4"]                       # neighbours: both on screen
    v.centerOn(QtCore.QPointF(150, 10))
    qapp.processEvents()
    QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, _crown_centre_on_screen(v, a))
    qapp.processEvents()
    assert a.isSelected() and a.isVisible(), "a click on a crown while zoomed out selected nothing"
    QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, _crown_centre_on_screen(v, b))
    qapp.processEvents()
    assert b.isSelected() and not a.isSelected(), "a second click must move the selection"
    assert not a.isVisible(), "the deselected crown goes back to the tile"
    # a click between crowns still pans and selects nothing new
    gap = v.mapFromScene(QtCore.QPointF(30, 30))
    QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, gap)
    assert [i for i in v.get_all_polygons() if i.isSelected()] == [b]


def test_ctrl_drag_selects_several_crowns_while_zoomed_out(zoomed_out_viewer, qapp):
    from PyQt5.QtTest import QTest
    v = zoomed_out_viewer
    z = v._batch_scale_threshold() * 0.9
    v.resetTransform()
    v.scale(z, z)
    v._sync_polygon_batch()
    assert v._batch_active
    v.centerOn(QtCore.QPointF(100, 30))
    qapp.processEvents()
    first = [i for i in v.get_all_polygons() if i.name in ("p0", "p1", "p2", "p3")]
    p0 = v.mapFromScene(QtCore.QPointF(-5, -5))
    p1 = v.mapFromScene(QtCore.QPointF(200, 35))   # one screen px ~ 8 scene units here
    QTest.mousePress(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.ControlModifier, p0)
    for k in range(1, 6):
        q = p0 + (p1 - p0) * k / 5
        QtWidgets.QApplication.sendEvent(v.viewport(), QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(q), QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
            QtCore.Qt.ControlModifier))
    QTest.mouseRelease(v.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.ControlModifier, p1)
    qapp.processEvents()
    assert all(i.isSelected() for i in first), "Ctrl+drag while zoomed out did not select the crowns in the box"


def test_a_deleted_crown_never_stays_in_the_tiles(zoomed_out_viewer, qapp):
    """THE remnant: the delete paths filtered viewer.polygons by p['name'],
    which is the LABEL for imported crowns, not the group the item is named
    by -- so the entry survived, the count never changed, the tiles were
    never rebuilt and kept drawing the deleted crown."""
    v = zoomed_out_viewer
    victim = v.get_all_polygons()[10]
    rec = next(r for r in v.polygons if r.get("item") is victim)
    rec["name"] = "a label, not the group"          # as on a shapefile import
    v._scene.removeItem(victim)                      # what the delete command does
    v.invalidate_polygon_batch()
    v._sync_polygon_batch()
    drawn = sum(len(t._polygons) for t in v._batch_tiles)
    assert drawn == len(v.get_all_polygons()), "a crown no longer in the scene is still in the tiles"


# ------------------------------------------- applied filter narrows the lists

def test_applied_filter_narrows_the_polygon_and_ml_lists(qapp, monkeypatch):
    """After Apply the Polygon Manager and ML lists show only the polygons the
    filter kept (coloured and visible; or visible when rules only hide); a
    polygon drawn afterwards still shows; Clear Filters shows all again."""
    from types import SimpleNamespace
    from canopie.polygon_manager import PolygonManager
    from canopie.machine_learning_manager import MachineLearningManager

    class _Owner(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.all_polygons = {g: {"f": _rec(g)} for g in ("a", "b", "c", "d")}
            self.project_folder = None
            self.viewer_widgets = []
            self.ml_manager = None

    owner = _Owner()
    pm = PolygonManager(owner)
    owner.polygon_manager = pm
    refreshed = []
    owner.ml_manager = SimpleNamespace(refresh_polygon_groups=lambda: refreshed.append(1))
    pm.set_polygons(owner.all_polygons)

    def rows():
        return [pm.list_widget.item(i).data(QtCore.Qt.UserRole) for i in range(pm.list_widget.count())]

    def ml_rows():
        return MachineLearningManager._listed_groups(SimpleNamespace(parent=lambda: owner))

    assert rows() == ["a", "b", "c", "d"] and pm.list_filter_label.isHidden()
    # a and c match (coloured), b doesn't, d matches but is hidden
    pm._on_filter_finished({("a", "f"): {"visible": True, "color": "#f00"},
                            ("b", "f"): {"visible": True, "color": None},
                            ("c", "f"): {"visible": True, "color": "#0f0"},
                            ("d", "f"): {"visible": False, "color": "#f00"}}, {}, [])
    assert rows() == ["a", "c"] and ml_rows() == ["a", "c"]
    assert refreshed, "the ML list was not refreshed"
    assert not pm.list_filter_label.isHidden() and "2 of 4" in pm.list_filter_label.text()

    # a polygon drawn after the filter is listed; the filter survives refreshes
    owner.all_polygons["e"] = {"f": _rec("e")}
    pm.set_polygons(owner.all_polygons)
    assert rows() == ["a", "c", "e"] and ml_rows() == ["a", "c", "e"]

    # rules that only hide: the visible ones are kept
    pm._on_filter_finished({("a", "f"): {"visible": False, "color": None},
                            ("b", "f"): {"visible": True, "color": None}}, {}, [])
    assert rows() == ["b", "c", "d", "e"]

    pm._clear_filters()
    assert rows() == ["a", "b", "c", "d", "e"] and ml_rows() == ["a", "b", "c", "d", "e"]
    assert pm.list_filter_label.isHidden()
    t0 = time.time()
    while (pm._index_running or pm._active_filter_threads) and time.time() - t0 < 5:
        qapp.processEvents()
        time.sleep(0.01)
    pm.deleteLater()
    owner.deleteLater()
    qapp.processEvents()
