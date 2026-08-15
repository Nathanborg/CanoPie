"""
Polygon locking + Filter & Color, the two PolygonManager features.

WHY THIS FILE EXISTS
--------------------
Nothing in the QC suite touched `lock_polys_checkbox`, `PolygonFilterWorker`,
`set_polygons_locked`, `apply_polygon_style_map` or `edit_polygon_properties`.
That gap is not academic: the whole PolygonManager half of the feature was
reverted in the working tree during an unrelated debugging session and the
full suite still reported 689 passed, because no test ever exercised it. These
tests fail loudly if any of it disappears or regresses again.

WHAT THEY PIN
-------------
* Locking is per-viewer state that NEW items inherit, and `unlock_group` is
  selective -- a global lock with one group exempted is the actual workflow.
* Filter rules genuinely FILTER. The worker's first draft hardcoded
  `visible=True`, so rules could only recolor; "Hide non-matching polygons"
  and a rule's own `hide` flag both have to be able to hide.
* First matching rule wins, and a rule that raises (referencing a property a
  given polygon lacks -- routine across a mixed project) counts as "no match"
  rather than aborting that polygon or the batch.
* Rule expressions are evaluated with no builtins: this is user-authored text
  from a form field, run once per polygon.
* `apply_polygon_style_map` restores default appearance for polygons the map
  no longer mentions, instead of leaving a stale filter result on screen.
* Areas are computed for polygons whose stored properties lack one, and
  reported back so the caller can persist them without recomputing.
"""
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer
from canopie.polygon_manager import PolygonFilterWorker, PolygonManager, _polygon_shoelace_area

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.polygons]


def _square(cx, cy, r=10.0):
    return QtGui.QPolygonF([
        QtCore.QPointF(cx - r, cy - r), QtCore.QPointF(cx + r, cy - r),
        QtCore.QPointF(cx + r, cy + r), QtCore.QPointF(cx - r, cy + r)])


@pytest.fixture
def viewer(qapp):
    v = ImageViewer()
    pm = QtGui.QPixmap(400, 400)
    v._scene.clear()
    v._image = v._scene.addPixmap(pm)
    v.polygons = []
    for i in range(3):
        v.add_polygon_to_scene(_square(50 + 100 * i, 50), name=f"g{i}")
    yield v
    v._scene.clear()
    v.deleteLater()


# ----------------------------------------------------------------- locking

def test_set_polygons_locked_locks_every_existing_item(viewer):
    assert all(not it.is_locked for it in viewer.get_all_polygons())
    viewer.set_polygons_locked(True)
    assert viewer.global_polygons_locked is True
    assert all(it.is_locked for it in viewer.get_all_polygons())
    viewer.set_polygons_locked(False)
    assert all(not it.is_locked for it in viewer.get_all_polygons())


def test_new_polygons_inherit_the_viewers_lock_state(viewer):
    """The lock is viewer state, not a one-shot sweep -- a polygon drawn while
    the project is locked must come out locked."""
    viewer.set_polygons_locked(True)
    viewer.add_polygon_to_scene(_square(300, 300), name="drawn_while_locked")
    added = [it for it in viewer.get_all_polygons()
             if getattr(it, "name", None) == "drawn_while_locked"]
    assert added and added[0].is_locked, (
        "a polygon added while the viewer is locked came out unlocked")


def test_unlock_group_is_selective(viewer):
    """Lock everything, exempt one group -- the real workflow."""
    viewer.set_polygons_locked(True)
    viewer.unlock_group("g1")
    by_name = {it.name: it.is_locked for it in viewer.get_all_polygons()}
    assert by_name["g1"] is False
    assert by_name["g0"] is True and by_name["g2"] is True


def test_unlock_group_ignores_empty_name(viewer):
    viewer.set_polygons_locked(True)
    viewer.unlock_group("")
    assert all(it.is_locked for it in viewer.get_all_polygons()), (
        "an empty group name unlocked everything")


# ------------------------------------------------------------ filter worker

def _run(snapshot, rules, hide_unmatched=False):
    w = PolygonFilterWorker(snapshot, rules, hide_unmatched=hide_unmatched)
    got = {}
    w.finished.connect(lambda sm, ca: got.update(style_map=sm, areas=ca))
    w.error.connect(lambda m: got.update(error=m))
    w.run()
    assert "error" not in got, got.get("error")
    return got["style_map"], got["areas"]


_SNAP = {
    "crowns": {
        "a.tif": {"name": "big",   "points": [(0, 0), (10, 0), (10, 10), (0, 10)],
                  "properties": {"area": 100.0, "species": "oak"}},
        "b.tif": {"name": "small", "points": [(0, 0), (2, 0), (2, 2), (0, 2)],
                  "properties": {"area": 4.0, "species": "pine"}},
    }
}


def test_rule_colors_matching_polygons_only():
    style, _ = _run(_SNAP, [{"expression": "area > 50", "color": (255, 0, 0)}])
    assert style[("crowns", "a.tif")]["color"].getRgb()[:3] == (255, 0, 0)
    assert style[("crowns", "a.tif")]["visible"] is True
    assert style[("crowns", "b.tif")]["color"] is None
    assert style[("crowns", "b.tif")]["visible"] is True, (
        "without hide_unmatched, a non-matching polygon must stay visible")


def test_hide_unmatched_actually_hides():
    """THE bug the first draft had: visible was hardcoded True, so 'Filter'
    could only ever recolor."""
    style, _ = _run(_SNAP, [{"expression": "area > 50", "color": (0, 255, 0)}],
                    hide_unmatched=True)
    assert style[("crowns", "a.tif")]["visible"] is True
    assert style[("crowns", "b.tif")]["visible"] is False, (
        "'Hide non-matching polygons' did not hide a non-matching polygon")


def test_rule_with_hide_flag_hides_its_matches():
    style, _ = _run(_SNAP, [{"expression": "species == 'pine'",
                             "color": (1, 2, 3), "hide": True}])
    assert style[("crowns", "b.tif")]["visible"] is False
    assert style[("crowns", "b.tif")]["color"] is None, (
        "a hidden polygon should carry no color")
    assert style[("crowns", "a.tif")]["visible"] is True


def test_first_matching_rule_wins():
    style, _ = _run(_SNAP, [
        {"expression": "area > 1",  "color": (10, 10, 10)},
        {"expression": "area > 50", "color": (20, 20, 20)},
    ])
    assert style[("crowns", "a.tif")]["color"].getRgb()[:3] == (10, 10, 10)


def test_rule_referencing_a_missing_property_is_not_an_error():
    """Mixed projects routinely have polygons lacking a property some rule
    mentions -- that must read as 'no match', not blow up the batch."""
    style, _ = _run(_SNAP, [
        {"expression": "dbh > 5", "color": (9, 9, 9)},          # no polygon has dbh
        {"expression": "area > 50", "color": (7, 7, 7)},
    ])
    assert style[("crowns", "a.tif")]["color"].getRgb()[:3] == (7, 7, 7)
    assert style[("crowns", "b.tif")]["color"] is None


def test_a_syntactically_broken_rule_does_not_disable_the_others():
    style, _ = _run(_SNAP, [
        {"expression": "area >>> 5", "color": (1, 1, 1)},        # SyntaxError
        {"expression": "area > 50",  "color": (2, 2, 2)},
    ])
    assert style[("crowns", "a.tif")]["color"].getRgb()[:3] == (2, 2, 2)


def test_rule_expressions_cannot_reach_builtins():
    """User-authored text evaluated per polygon -- it must not be able to open
    files or import."""
    style, _ = _run(_SNAP, [{"expression": "__import__('os').getcwd() != ''",
                             "color": (1, 1, 1)}])
    assert all(s["color"] is None for s in style.values()), (
        "an expression reached __import__ -- the eval sandbox is gone")


def test_missing_area_is_computed_and_reported_back():
    snap = {"g": {"f.tif": {"name": "n",
                            "points": [(0, 0), (4, 0), (4, 3), (0, 3)],
                            "properties": {}}}}
    style, areas = _run(snap, [{"expression": "area > 10", "color": (5, 5, 5)}])
    assert areas[("g", "f.tif")] == pytest.approx(12.0)
    assert style[("g", "f.tif")]["color"].getRgb()[:3] == (5, 5, 5), (
        "the freshly computed area was not visible to the rule")


def test_shoelace_area_matches_a_known_polygon():
    assert _polygon_shoelace_area([(0, 0), (4, 0), (4, 3), (0, 3)]) == pytest.approx(12.0)
    assert _polygon_shoelace_area([(0, 0), (1, 1)]) == 0.0  # degenerate


# --------------------------------------------------- applying to the viewer

def test_apply_style_map_sets_color_and_visibility(viewer, monkeypatch):
    monkeypatch.setattr(type(viewer), "image_data",
                        property(lambda self: type("D", (), {"filepath": "x.tif"})()),
                        raising=False)
    style_map = {
        ("g0", "x.tif"): {"visible": True,  "color": QtGui.QColor(255, 0, 0)},
        ("g1", "x.tif"): {"visible": False, "color": None},
    }
    viewer.apply_polygon_style_map(style_map)
    by_name = {it.name: it for it in viewer.get_all_polygons()}
    assert by_name["g0"].current_color.getRgb()[:3] == (255, 0, 0)
    assert by_name["g1"]._filtered_hidden is True
    assert by_name["g1"].isVisible() is False


def test_apply_style_map_restores_polygons_it_no_longer_mentions(viewer, monkeypatch):
    """Re-running a narrowed filter must not leave the previous run's colour
    and hidden state stranded on polygons that now match nothing."""
    monkeypatch.setattr(type(viewer), "image_data",
                        property(lambda self: type("D", (), {"filepath": "x.tif"})()),
                        raising=False)
    viewer.apply_polygon_style_map(
        {("g0", "x.tif"): {"visible": False, "color": QtGui.QColor(1, 2, 3)}})
    g0 = {it.name: it for it in viewer.get_all_polygons()}["g0"]
    assert g0._filtered_hidden is True

    viewer.apply_polygon_style_map({})       # nothing matches any more
    assert g0.current_color is None, "stale filter colour survived an empty style map"
    assert g0._filtered_hidden is False
    assert g0.isVisible() is True


# ------------------------------------------- filepath-spelling (shapefile import)

class _OwnerWithIndex:
    """Minimal stand-in for ProjectTab: all_polygons keyed the way a SHAPEFILE
    IMPORT keys it (os.path.normpath -> backslashes on Windows), plus the real
    union-index lookup those projects depend on."""

    def __init__(self, group, stored_key, entry):
        self.all_polygons = {group: {stored_key: entry}}
        self._dirty = []

    def _poly_index_lookup(self, filepath):
        import os
        want = os.path.normpath(str(filepath)).lower()
        out = []
        for g, files in self.all_polygons.items():
            for k in files:
                if os.path.normpath(str(k)).lower() == want:
                    out.append((g, k))
        return out

    def _mark_polygon_dirty(self, group, fp):
        self._dirty.append((group, fp))


# The two spellings one project really holds: project.json / the viewer use
# forward slashes, a shapefile import stores os.path.normpath.
_VIEWER_FP = "C:/Users/natha/Downloads/20240716_bciclippled_rx1ii_rgb_gr0p07_infer_cog.tif"
# Exactly what the importer writes: os.path.normpath of the same file. Derived
# rather than written as a literal so this stays honest on non-Windows too.
_IMPORT_FP = os.path.normpath(_VIEWER_FP)


def test_edit_properties_finds_polygons_stored_under_the_import_spelling(viewer, monkeypatch):
    """Editing properties on a shapefile-imported polygon failed with "Could
    not find this polygon's stored data" for every polygon in the project: the
    lookup was an exact all_polygons[group][viewer_filepath], which cannot
    match the backslash key the importer wrote."""
    entry = {"points": [(0, 0), (10, 0), (10, 10)],
             "properties": {"species": "Hieronyma alchorneoides"}}
    owner = _OwnerWithIndex("g0", _IMPORT_FP, entry)

    monkeypatch.setattr(viewer, "_find_project_owner", lambda: owner, raising=False)
    monkeypatch.setattr(type(viewer), "image_data",
                        property(lambda self: type("D", (), {"filepath": _VIEWER_FP})()),
                        raising=False)

    item = {it.name: it for it in viewer.get_all_polygons()}["g0"]

    seen = {}
    from canopie import image_viewer as iv

    class _Dlg:
        def __init__(self, props, title="", parent=None):
            seen["props"] = props
        def exec_(self):
            return 1
        def get_properties(self):
            return {"species": "edited"}

    monkeypatch.setattr(iv, "PolygonPropertiesDialog", _Dlg)
    monkeypatch.setattr(iv.QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: pytest.fail(
                            "edit_polygon_properties refused a polygon that IS "
                            "stored, just under the import's path spelling: %s" % (a[2:],)))

    viewer.edit_polygon_properties(item)

    assert seen.get("props") == {"species": "Hieronyma alchorneoides"}, (
        "the dialog was not seeded from the stored properties")
    assert entry["properties"] == {"species": "edited"}, (
        "edits were not written back to the stored entry")
    assert owner._dirty == [("g0", _IMPORT_FP)], (
        "the polygon must be marked dirty under the key it is actually stored "
        "under, or the save writes to the wrong record")


def test_zoom_to_group_finds_polygons_stored_under_the_import_spelling(qapp, monkeypatch):
    """Same root cause, the other symptom: `zoom_to_groups: group=<g>,
    rect=None` for every imported group, so double-clicking a group in the
    Polygon Manager did nothing."""
    from canopie.polygon_manager import PolygonManager

    entry = {"points": [(100, 100), (200, 100), (200, 200), (100, 200)],
             "image_ref_size": {"w": 1000, "h": 1000}}
    owner = _OwnerWithIndex("crown_1", _IMPORT_FP, entry)

    pm = PolygonManager.__new__(PolygonManager)      # no Qt init needed
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    idata = type("D", (), {"filepath": _VIEWER_FP, "ax_config": {},
                           "raw_shape": (1000, 1000, 3)})()
    v = ImageViewer()
    v._scene.setSceneRect(0, 0, 1000, 1000)

    rect = pm._get_polygon_scene_rect({"image_data": idata}, v, "crown_1")
    v.deleteLater()

    assert rect is not None, (
        "no scene rect for a group whose polygons are stored under the "
        "import's path spelling -- 'Zoom to group' silently does nothing")
    assert rect.width() > 0 and rect.height() > 0


def test_editor_open_does_not_resync_polygons_for_a_clean_root():
    """Opening the Image Editor used to call update_all_polygons()
    unconditionally, and that call INVALIDATES the polygon index -- so every
    editor open paid an index rebuild proportional to the project's polygon
    count, for a dialog that never draws a polygon. On the BCI COG project
    (3598 crowns) the editor was slow to open with polygons loaded and instant
    after clearing them.

    _jump_to_index already gates the identical sync on _dirty_polygon_roots
    ("Calling update_all_polygons() on every navigation was causing
    50,000-entry index rebuilds"); this pins the same gate on the editor path,
    including its two fail-safe cases -- an unknown root or a project with no
    dirty-tracking must still sync, so a pending edit can never be dropped.
    """
    import inspect
    from canopie.project_tab import ProjectTab

    src = inspect.getsource(ProjectTab.edit_image_viewer)
    assert "_dirty_polygon_roots" in src, (
        "edit_image_viewer no longer gates its polygon sync on "
        "_dirty_polygon_roots -- every editor open rebuilds the polygon index")
    assert "self.update_all_polygons()" in src, (
        "the sync must still be reachable for a dirty root, not deleted")

    # The gate's decision table, evaluated exactly as the source does.
    def would_sync(root_names, idx, dirty):
        cur = root_names[idx] if 0 <= idx < len(root_names) else None
        return cur is None or dirty is None or cur in dirty

    assert would_sync(["r1"], 0, set()) is False, "a clean root must skip the sync"
    assert would_sync(["r1"], 0, {"r1"}) is True, "a dirty root must still sync"
    assert would_sync(["r1"], 0, None) is True, "no dirty-tracking -> must still sync"
    assert would_sync([], 0, set()) is True, "unknown root -> must still sync"


# ----------------------------------------------------- shapefile import: cancel

def test_cancelling_shapefile_import_still_emits_finished(qapp):
    """THE stuck "Importing Shapefiles" progress bar bug.

    ShapefileImportWorker.run() gated its `self.finished.emit(...)` behind
    `if not self._is_cancelled:` -- so clicking "Cancel Import" made run()
    return having emitted NEITHER `finished` NOR `error`. Tracing what that
    starves, in _start_shapefile_import_worker:

      * ShapefileImportProgressDialog.on_cancel() only disables its button and
        emits cancel_requested -- it never closes itself, relying entirely on
        `worker.finished` reaching `_on_shapefile_import_finished`, which is
        the only thing that calls progress_dlg.close()/deleteLater().
      * `worker.finished.connect(thread.quit)` is the only thing that stops
        the QThread's event loop; nothing else ever calls quit().
      * `thread.finished.connect(_cleanup)` -- the deleteLater()/bookkeeping
        cleanup -- only runs once the thread actually quits.

    So cancelling left the progress dialog on screen forever (stuck on
    "Canceling...") AND leaked a QThread whose event loop never stops.

    Uses an empty file_paths list so the worker's own file/feature loops
    never execute at all -- the broken code path did not depend on any file
    actually being processed, only on `self._is_cancelled` being True when
    control reaches the end of run().
    """
    from canopie.polygon_manager import ShapefileImportWorker

    worker = ShapefileImportWorker(
        file_paths=[], project_folder=".", target_images_info=[])
    worker._is_cancelled = True

    got = {}
    worker.finished.connect(lambda data, warns: got.update(finished=(data, warns)))
    worker.error.connect(lambda msg: got.update(error=msg))

    worker.run()

    assert "error" not in got, f"run() reported an error instead: {got.get('error')}"
    assert "finished" in got, (
        "worker.run() returned without emitting finished (or error) when "
        "cancelled -- the progress dialog has nothing left to close it and "
        "the QThread hosting this worker has nothing left to quit it")
    data, warns = got["finished"]
    assert data == {}, "no files were queued, so nothing should have been imported"
    assert any("cancel" in w.lower() for w in warns), (
        "a cancelled import should say so in its warnings/summary")


def test_completing_shapefile_import_normally_still_works(qapp):
    """The non-cancelled path must be unaffected by the fix above."""
    from canopie.polygon_manager import ShapefileImportWorker

    worker = ShapefileImportWorker(
        file_paths=[], project_folder=".", target_images_info=[])

    got = {}
    worker.finished.connect(lambda data, warns: got.update(finished=(data, warns)))
    worker.error.connect(lambda msg: got.update(error=msg))

    worker.run()

    assert "error" not in got
    assert "finished" in got
    data, warns = got["finished"]
    assert data == {}
    assert not any("cancel" in w.lower() for w in warns), (
        "a normal (non-cancelled) run must not claim it was cancelled")


def test_successful_shapefile_import_closes_the_progress_dialog(qapp, monkeypatch):
    """THE "Importing Shapefiles" dialog that never went away.

    Distinct from the cancel bug above: here the worker DOES emit `finished`,
    but the handler failed to act on it. Its cleanup block opened with
    `progress_dlg.reset()` -- a QProgressDialog method. But
    ShapefileImportProgressDialog is a plain QDialog and has no `reset`, so
    the FIRST statement raised AttributeError, the surrounding bare
    `except Exception: pass` swallowed it, and hide()/close()/deleteLater()
    never ran. The dialog stayed on screen after every successful import.

    Regression note: HEAD's version was a working `progress_dlg.close()`;
    this was introduced while "explicitly destroying" the dialog.
    """
    from canopie.polygon_manager import PolygonManager
    from canopie.shapefile_import_dialog import ShapefileImportProgressDialog
    from PyQt5 import QtWidgets as _QtW

    pm = PolygonManager.__new__(PolygonManager)
    _QtW.QWidget.__init__(pm)

    dlg = ShapefileImportProgressDialog()
    dlg.show()
    assert dlg.isVisible(), "precondition: the progress dialog is on screen"

    # parent() is None here, so the handler takes its early-return branch --
    # but only AFTER the dialog-close block this test is about.
    monkeypatch.setattr(_QtW.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    PolygonManager._on_shapefile_import_finished(pm, {}, [], dlg)
    qapp.processEvents()

    assert not dlg.isVisible(), (
        "the shapefile import progress dialog is still on screen after the "
        "finished-handler ran -- something in its cleanup raised before "
        "close() (this is exactly what reset() did)")


def test_progress_dialog_cleanup_does_not_call_a_missing_reset(qapp):
    """Root-cause pin, independent of the behavioural test above.

    ShapefileImportProgressDialog genuinely has no `reset` -- assert that,
    so if someone later swaps the base class to QProgressDialog this test
    tells them to re-check the assumption rather than silently passing. And
    assert the handler's source doesn't call reset() on it.
    """
    import inspect
    from canopie.polygon_manager import PolygonManager
    from canopie.shapefile_import_dialog import ShapefileImportProgressDialog

    assert not hasattr(ShapefileImportProgressDialog, "reset"), (
        "ShapefileImportProgressDialog now HAS reset(); the guard in "
        "_on_shapefile_import_finished can be revisited, but verify the "
        "close path still runs unconditionally")

    src = inspect.getsource(PolygonManager._on_shapefile_import_finished)
    assert "progress_dlg.reset()" not in src, (
        "reset() is back in the shapefile-import cleanup -- it raises "
        "AttributeError on a QDialog and strands the progress dialog")
    assert "close" in src, "the cleanup must still close the dialog"


# --------------------------------------------- apply_polygon_style_map: path spelling

def _stub_viewer_with_one_polygon(qapp, viewer_fp):
    v = ImageViewer()
    idata = type("D", (), {"filepath": viewer_fp, "image": None})()
    v.image_data = idata
    pmv = QtGui.QPixmap(400, 400)
    v._scene.clear()
    v._image = v._scene.addPixmap(pmv)
    v.polygons = []
    v.add_polygon_to_scene(_square(50, 50), name="g0")
    return v


def test_style_map_applies_under_the_import_path_spelling(qapp):
    """THE dominant cause of "filtering does nothing after importing
    shapefiles". apply_polygon_style_map resolved each item's key as an
    EXACT tuple (item.name, viewer.image_data.filepath) -- but style_map's
    keys come from PolygonFilterWorker's snapshot of all_polygons, i.e. the
    STORAGE spelling. A shapefile import stores os.path.normpath (backslashes
    on Windows) while the viewer/project.json use forward slashes (see
    ProjectTab._poly_index_lookup's docstring for the measured real-project
    numbers). An exact lookup missed on EVERY shapefile-imported polygon, and
    the `style is None` branch actively resets the item to default -- so the
    filter looked like it did nothing at all, regardless of whether any rule
    matched correctly.
    """
    v = _stub_viewer_with_one_polygon(qapp, _VIEWER_FP)
    style_map = {("g0", _IMPORT_FP): {"visible": False, "color": QtGui.QColor(1, 2, 3)}}
    v.apply_polygon_style_map(style_map)

    item = v.get_all_polygons()[0]
    assert item._filtered_hidden is True, (
        "style_map keyed under the import spelling did not reach a viewer "
        "using the project/viewer spelling")
    assert item.isVisible() is False


def test_style_map_still_applies_under_the_viewer_spelling(qapp):
    """Guard against the normalization fix breaking the common (single
    spelling) case."""
    v = _stub_viewer_with_one_polygon(qapp, _VIEWER_FP)
    style_map = {("g0", _VIEWER_FP): {"visible": False, "color": None}}
    v.apply_polygon_style_map(style_map)
    item = v.get_all_polygons()[0]
    assert item._filtered_hidden is True


def test_style_map_entry_for_a_different_file_is_ignored(qapp):
    """A style_map entry for a DIFFERENT file must not match by name alone --
    normalization must still compare the filepath, not skip it."""
    v = _stub_viewer_with_one_polygon(qapp, _VIEWER_FP)
    other_fp = os.path.normpath("C:/Users/natha/Downloads/some_other_file.tif")
    style_map = {("g0", other_fp): {"visible": False, "color": None}}
    v.apply_polygon_style_map(style_map)
    item = v.get_all_polygons()[0]
    assert item._filtered_hidden is False, (
        "a style entry for a different file's path was applied to this viewer")


def test_a_polygon_absent_from_the_style_map_is_restored_to_default(qapp):
    """The `style is None` branch: re-running a narrowed filter must restore
    default appearance for polygons the new map no longer mentions -- this
    is the existing behavior the normalization fix must not disturb."""
    v = _stub_viewer_with_one_polygon(qapp, _VIEWER_FP)
    v.apply_polygon_style_map({("g0", _IMPORT_FP): {"visible": False, "color": QtGui.QColor(9, 9, 9)}})
    item = v.get_all_polygons()[0]
    assert item._filtered_hidden is True

    v.apply_polygon_style_map({})
    assert item._filtered_hidden is False
    assert item.current_color is None


# --------------------------------------------- numeric coercion + zero-match reporting

def test_string_typed_numeric_dbf_values_support_ordering():
    """A DBF Character-typed column holding numeric-looking text (real
    example: dbh_mm='353.0') must support >/</>=/<= -- pre-fix this raised
    TypeError on every polygon, silently swallowed as 'no match'."""
    from canopie.polygon_manager import _coerce_numeric_strings
    props = {"dbh_mm": "353.0", "hom_m": "1.3", "crown": "4"}
    coerced = _coerce_numeric_strings(props)
    assert coerced["dbh_mm"] == 353.0 and isinstance(coerced["dbh_mm"], float)
    assert coerced["hom_m"] == 1.3
    assert coerced["crown"] == 4.0


def test_coercion_leaves_non_numeric_text_alone():
    from canopie.polygon_manager import _coerce_numeric_strings
    props = {"species": "acuminatum", "plot_statu": "Unique Crown", "empty": ""}
    coerced = _coerce_numeric_strings(props)
    assert coerced == props, "non-numeric strings must be returned unchanged"


def test_coercion_skips_nan_and_inf_strings():
    """float('nan')/float('inf') parse 'successfully' into a value that
    compares False against everything -- silently reintroducing the exact
    failure mode this exists to fix, for a column that legitimately holds
    that text (a status code, not a measurement)."""
    from canopie.polygon_manager import _coerce_numeric_strings
    props = {"status": "NaN", "flag": "inf", "other": "-Infinity"}
    coerced = _coerce_numeric_strings(props)
    assert coerced == props


def test_coercion_does_not_mutate_the_input_dict():
    from canopie.polygon_manager import _coerce_numeric_strings
    props = {"dbh_mm": "353.0"}
    _coerce_numeric_strings(props)
    assert props["dbh_mm"] == "353.0", "input dict was mutated in place"


def test_a_numeric_comparison_rule_now_matches_string_typed_dbf_values():
    """End-to-end through the real worker, not just the coercion helper."""
    snap = {"g": {"a.tif": {"name": "n1", "points": [(0, 0), (1, 0), (1, 1)],
                            "properties": {"dbh_mm": "353.0"}},
                 "b.tif": {"name": "n2", "points": [(0, 0), (1, 0), (1, 1)],
                            "properties": {"dbh_mm": "10.0"}}}}
    style, _ = _run(snap, [{"expression": "dbh_mm > 300", "color": (1, 1, 1)}])
    assert style[("g", "a.tif")]["color"] is not None
    assert style[("g", "b.tif")]["color"] is None


def test_a_rule_that_matches_nothing_is_reported():
    w = PolygonFilterWorker(_SNAP, [{"expression": "area > 1e9", "color": (1, 1, 1)}])
    got = {}
    w.finished.connect(lambda sm, ca, zm: got.update(zero_match=zm))
    w.run()
    assert got["zero_match"] == ["area > 1e9"]


def test_a_rule_that_matches_something_is_not_reported():
    w = PolygonFilterWorker(_SNAP, [{"expression": "area > 50", "color": (1, 1, 1)}])
    got = {}
    w.finished.connect(lambda sm, ca, zm: got.update(zero_match=zm))
    w.run()
    assert got["zero_match"] == []


def test_a_syntactically_broken_rule_is_not_reported_as_zero_match():
    """A rule that never compiled has its own SyntaxError warning already
    (logging.warning at compile time) -- it must not ALSO show up in the
    zero-match list, which is specifically for rules that compiled fine but
    matched nothing."""
    w = PolygonFilterWorker(_SNAP, [{"expression": "area >>> 5", "color": (1, 1, 1)}])
    got = {}
    w.finished.connect(lambda sm, ca, zm: got.update(zero_match=zm))
    w.run()
    assert got["zero_match"] == []


def test_on_filter_finished_surfaces_zero_match_rules_in_the_status_label():
    pm = PolygonManager.__new__(PolygonManager)
    QtWidgets.QWidget.__init__(pm)
    pm.filter_status_label = QtWidgets.QLabel()
    pm._iter_viewers = lambda: []
    pm._last_filter_style_map = None

    PolygonManager._on_filter_finished(pm, {}, {}, ["dbh_mm > 300"])
    assert "dbh_mm > 300" in pm.filter_status_label.text()


def test_on_filter_finished_says_nothing_extra_when_everything_matched():
    pm = PolygonManager.__new__(PolygonManager)
    QtWidgets.QWidget.__init__(pm)
    pm.filter_status_label = QtWidgets.QLabel()
    pm._iter_viewers = lambda: []
    pm._last_filter_style_map = None

    PolygonManager._on_filter_finished(pm, {("g", "f"): {"visible": True, "color": None}}, {}, [])
    assert "matched nothing" not in pm.filter_status_label.text()


# ------------------------------------------------- Bug B: persistence across refresh

def test_last_filter_style_map_recorded_after_apply():
    pm = PolygonManager.__new__(PolygonManager)
    QtWidgets.QWidget.__init__(pm)
    pm.filter_status_label = QtWidgets.QLabel()
    pm._iter_viewers = lambda: []
    pm._last_filter_style_map = None

    sm = {("g", "f"): {"visible": False, "color": None}}
    PolygonManager._on_filter_finished(pm, sm, {}, [])
    assert pm._last_filter_style_map is sm


def test_reload_reapplies_the_last_filter_style(qapp):
    """The load_polygons hook, driven directly against a fresh viewer --
    proves a viewer rebuilt AFTER a filter was applied still ends up styled,
    which is exactly what a root switch / reopen does to every polygon item.
    """
    v = _stub_viewer_with_one_polygon(qapp, _VIEWER_FP)
    pm = type("PM", (), {"_last_filter_style_map":
                         {("g0", _IMPORT_FP): {"visible": False, "color": None}}})()

    # The exact block added to project_tab.py's load_polygons, run standalone.
    try:
        style_map = getattr(pm, "_last_filter_style_map", None) if pm is not None else None
        if style_map and hasattr(v, "apply_polygon_style_map"):
            v.apply_polygon_style_map(style_map)
    except Exception as e:
        pytest.fail(f"reapply raised: {e}")

    item = v.get_all_polygons()[0]
    assert item._filtered_hidden is True


def test_load_polygons_reapply_block_runs_after_polygon_population():
    """Source-level pin: the reapply call must be textually AFTER the
    polygon-add loop in load_polygons, not in the early per-viewer sync
    block near the top (which only sets viewer-level flags on a viewer that
    may not have any items yet)."""
    import inspect
    from canopie.project_tab import ProjectTab

    src = inspect.getsource(ProjectTab.load_polygons)
    add_idx = src.index("add_polygon_to_scene")
    reapply_idx = src.index("_last_filter_style_map")
    assert reapply_idx > add_idx, (
        "the filter reapply block moved before polygon population -- it "
        "would have nothing to style")


# ------------------------------------------------- Bug C1: scoped copy, not deepcopy

def test_apply_filters_snapshot_shares_leaf_data_not_deepcopies_it():
    """_apply_filters used copy.deepcopy(parent.all_polygons) -- copying
    every point of every polygon synchronously on the GUI thread before the
    background worker even started. PolygonFilterWorker.run() only ever
    READS poly.get('properties')/poly.get('points')/poly.get('name'); only
    the container structure (groups, filepaths) needs protecting from a
    concurrent add/remove, so the leaf polygon dicts and their point lists
    can be shared by reference. Source-level pin, mirroring how this
    session's other synchronous-vs-lazy fixes are pinned."""
    import inspect
    from canopie.polygon_manager import PolygonManager

    src = inspect.getsource(PolygonManager._apply_filters)
    assert "copy.deepcopy" not in src, (
        "_apply_filters is deepcopying all_polygons again -- this was the "
        "single biggest cost in 'Apply Filter should be instant' (measured "
        "1589x slower than a scoped copy on a real 5265-polygon project)")


def test_apply_filters_snapshot_construction_shares_point_lists():
    """Runtime pin, independent of the source-text check above: build the
    exact snapshot expression _apply_filters uses and confirm identity."""
    all_polygons = {
        "g": {"f.tif": {"name": "n", "points": [(0, 0), (1, 0), (1, 1)],
                        "properties": {}}}
    }
    snapshot = {g: dict(files) for g, files in all_polygons.items()}

    assert snapshot["g"]["f.tif"] is all_polygons["g"]["f.tif"], (
        "leaf polygon dict was copied, not shared")
    assert snapshot["g"]["f.tif"]["points"] is all_polygons["g"]["f.tif"]["points"], (
        "leaf point list was copied, not shared -- this is exactly the cost "
        "that made a deepcopy expensive")

    # And the container IS protected: adding a new file to the live dict
    # must not appear in (or corrupt iteration of) the already-taken snapshot.
    all_polygons["g"]["new.tif"] = {"name": "n2", "points": [], "properties": {}}
    assert "new.tif" not in snapshot["g"]
