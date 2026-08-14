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
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from canopie.image_viewer import ImageViewer
from canopie.polygon_manager import PolygonFilterWorker, _polygon_shoelace_area

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
