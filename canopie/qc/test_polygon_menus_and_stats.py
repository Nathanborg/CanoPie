"""
Polygon right-click menus + "Polygon Stats" for selected polygons.

* Export Manager lives under Modules (not on the toolbar).
* A polygon's menu in a viewer and the Polygon Manager's menu offer the same
  entries (Edit vertices / properties, Find canopy, Lock, Polygon Stats,
  Polygon Info ...).
* "Polygon Stats" = the CSV export's statistics (process_polygon) for just
  the selected polygons -- several at once, from the viewer (Ctrl-selected
  items) and from the Polygon Manager (selected groups, on every image).
  The old Polygon Manager "Polygon Stats" (counts) is now "Polygon Info".
"""
import os

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

pytestmark = [pytest.mark.polygons, pytest.mark.extraction]

SHARED = {"Edit vertices", "Edit properties", "Find canopy in other images…"}


@pytest.fixture()
def project_tab(qapp, fixtures_ready, tmp_path):
    from .project_builder import build_project_tab
    pt = build_project_tab(str(tmp_path))
    pt.analysis_options = {"include_exif": False}     # no options dialog / exiftool
    yield pt


def _polygon_pairs(pt, n=2):
    out = []
    for g, fm in pt.all_polygons.items():
        for fp, rec in fm.items():
            if rec.get("type", "polygon") == "polygon" and len(rec.get("points") or []) >= 3:
                out.append((g, fp))
                break
        if len(out) == n:
            return out
    raise AssertionError("fixture project lacks polygons")


def _base(text):
    return text.split(" (")[0]


def _capture_menus(monkeypatch, choose=None):
    """Replace QMenu.exec_: record each menu's entries; pick the entry
    starting with `choose` (or nothing)."""
    seen = []

    def fake_exec(menu, *a, **k):
        acts = [x for x in menu.actions() if not x.isSeparator()]
        seen.append([x.text() for x in acts])
        if choose:
            return next((x for x in acts if x.text().startswith(choose)), None)
        return None
    monkeypatch.setattr(QtWidgets.QMenu, "exec_", fake_exec)
    return seen


def test_export_manager_is_under_modules(project_tab):
    pt = project_tab
    assert pt.exportManagerAct not in pt.toolbar.actions(), "Export Manager is still on the toolbar"
    modules = [b.menu() for b in pt.toolbar.findChildren(QtWidgets.QToolButton)
               if b.text() == "Modules" and b.menu() is not None]
    assert modules and pt.exportManagerAct in modules[0].actions()
    assert pt.exportManagerAct.shortcut().toString() == "Ctrl+E"


def test_polygon_stats_rows_are_the_csv_exports(project_tab):
    """Same numbers as process_polygon itself, for each selected polygon."""
    pt = project_tab
    (g1, f1), (g2, f2) = _polygon_pairs(pt, 2)
    rows = pt.stats_rows_for_polygons([(g1, f1), (g2, f2)])
    assert rows, "no stats rows"
    assert {r.get("Object ID") for r in rows} >= {g1, g2}
    inputs = pt._stats_inputs([f1])
    ref, _ = pt.process_polygon(group_name=g1, filepath=f1, polygon_dict=pt.all_polygons[g1][f1],
                                exif_data_dict=inputs[1], exif_tags=inputs[2],
                                model_loaded=inputs[3], opts=inputs[0])
    got = [r for r in rows if r.get("Object ID") == g1]
    assert [r.get("Mean") for r in got] == [r.get("Mean") for r in ref]


def test_polygon_stats_finds_an_imported_crown_under_the_other_spelling(project_tab):
    pt = project_tab
    (g, f), = _polygon_pairs(pt, 1)
    alt = f.replace("/", "\\") if "/" in f else f.replace("\\", "/")
    rec = pt.all_polygons[g].pop(f)
    pt.all_polygons[g][alt] = rec
    pt._poly_norm_index_invalid = True
    try:
        rows = pt.stats_rows_for_polygons([(g, f)])
        assert rows and all(r.get("Object ID") == g for r in rows)
    finally:
        pt.all_polygons[g] = {f: pt.all_polygons[g].pop(alt)}
        pt._poly_norm_index_invalid = True


def _viewer_with(pt, fp, groups):
    from canopie.image_viewer import ImageViewer
    v = ImageViewer()
    v.image_data = type("D", (), {"filepath": fp, "image": None})()
    v._scene.clear()
    v._image = v._scene.addPixmap(QtGui.QPixmap(200, 200))
    v.polygons = []
    v.setParent(pt)                             # _find_project_owner walks parents
    for i, g in enumerate(groups):
        x = 10 + 50 * i
        v.add_polygon_to_scene(QtGui.QPolygonF([QtCore.QPointF(x, 10), QtCore.QPointF(x + 30, 10),
                                                QtCore.QPointF(x + 30, 40)]), name=g)
    return v


def test_viewer_and_manager_menus_offer_the_same_entries(project_tab, monkeypatch):
    pt = project_tab
    (g, f), = _polygon_pairs(pt, 1)
    v = _viewer_with(pt, f, [g])
    seen = _capture_menus(monkeypatch)
    v.get_all_polygons()[0].contextMenuEvent(None)
    viewer_entries = {_base(t) for t in seen[-1]}

    pm = pt.polygon_manager
    pm.set_polygons(pt.all_polygons)
    item = next(pm.list_widget.item(i) for i in range(pm.list_widget.count())
                if pm.list_widget.item(i).data(QtCore.Qt.UserRole) == g)
    pm.show_context_menu(pm.list_widget.visualItemRect(item).center())
    manager_entries = {_base(t) for t in seen[-1]}

    for name in SHARED | {"Polygon Stats…", "Polygon Info…"}:
        assert name in viewer_entries, f"viewer menu lacks '{name}'"
        assert name in manager_entries, f"Polygon Manager menu lacks '{name}'"
    assert any(t.startswith("Lock") or t.startswith("Unlock") for t in viewer_entries)
    assert any(t.startswith("Lock") or t.startswith("Unlock") for t in manager_entries)
    v.setParent(None)
    v.deleteLater()


def test_polygon_stats_for_several_selected_polygons_in_a_viewer(project_tab, monkeypatch):
    pt = project_tab
    (g1, f), (g2, _f2) = _polygon_pairs(pt, 2)
    v = _viewer_with(pt, f, [g1, g2])
    a, b = v.get_all_polygons()[:2]
    a.setSelected(True)
    b.setSelected(True)                          # Ctrl-click / Ctrl-drag selection
    calls = []
    monkeypatch.setattr(pt, "show_stats_for_polygons", lambda pairs, **k: calls.append(sorted(pairs)))
    seen = _capture_menus(monkeypatch, choose="Polygon Stats")
    a.contextMenuEvent(None)
    assert any("(2 selected)" in t for t in seen[-1]), "the menu should say it acts on the selection"
    assert calls == [sorted([(g1, f), (g2, f)])], f"stats were not asked for both: {calls}"
    v.setParent(None)
    v.deleteLater()


def test_polygon_stats_for_several_groups_in_the_manager(project_tab, monkeypatch):
    pt = project_tab
    (g1, _f1), (g2, _f2) = _polygon_pairs(pt, 2)
    pm = pt.polygon_manager
    pm.set_polygons(pt.all_polygons)
    lw = pm.list_widget
    items = {lw.item(i).data(QtCore.Qt.UserRole): lw.item(i) for i in range(lw.count())}
    lw.clearSelection()
    items[g1].setSelected(True)
    items[g2].setSelected(True)
    calls = []
    monkeypatch.setattr(pt, "show_stats_for_polygons", lambda pairs, **k: calls.append(sorted(pairs)))
    _capture_menus(monkeypatch, choose="Polygon Stats")
    pm.show_context_menu(lw.visualItemRect(items[g1]).center())
    want = sorted((g, fp) for g in (g1, g2) for fp in pt.all_polygons[g])
    assert calls == [want], "Polygon Stats did not cover every image of both selected groups"


def test_old_polygon_stats_is_now_polygon_info(project_tab, monkeypatch):
    pt = project_tab
    pm = pt.polygon_manager
    titles = []
    monkeypatch.setattr(QtWidgets.QDialog, "exec_", lambda self: titles.append(self.windowTitle()) or 0)
    pm.show_polygon_info([_polygon_pairs(pt, 1)[0][0]])
    assert titles == ["Polygon Info"]
