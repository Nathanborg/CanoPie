"""QC regression tests for the LIVE viewer's point-item -> saved-coordinate
path: the step BEFORE process_polygon ever runs, where a drawn/moved point's
on-screen position gets turned into the (x, y) pair written to
all_polygons / polygons/*.json.

THE BUG THIS PINS ("points drawn in the past go stray on reload"):

EditablePolygonItem stores genuine SCENE coordinates directly
(`self._polygon = polygon`, no indirection -- see its __init__).
EditablePointItem does NOT: `self.points` is pixmap-LOCAL, floor()-truncated
INTEGER pixel coordinates (see ImageViewer._scene_to_pix_local_int /
add_point_to_scene), and the class provides `_scene_xy(p)` specifically to
convert one of those back into a genuine scene position -- which is what its
own paint()/boundingRect()/hit-testing already do internally.

Both places that turn a live item into the coordinate saved to disk --
ProjectTab.on_polygon_drawn and .on_polygon_modified -- used to read
`item.points` (on_polygon_drawn) or call `item.mapToScene(item.points)`
(on_polygon_modified) and treat the result as if it were already a scene
coordinate. Because EditablePointItem's own transform is the identity,
`mapToScene` is a silent no-op there, so both paths fed raw pixmap-local
integers straight into `scene_to_image_coords()` -- which expects a genuine
scene point -- as though no conversion were needed. The wrong "image pixel"
coordinate that produced was what got WRITTEN to disk, at draw/move time, so
every reload was faithfully (and correctly) re-displaying an already-wrong
saved value.

EditablePolygonItem's `polygon` attribute holds genuine scene coordinates
already, so it never had this defect -- which is exactly why the user's
report was specific to points and not polygons.
"""
import ast
import inspect

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from ..image_viewer import EditablePointItem
from ..project_tab import ProjectTab
from .test_export_and_ax_regressions import _func_tree, _names_in, _calls_in

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


class _FakePixmapItem:
    """A minimal stand-in for the QGraphicsPixmapItem EditablePointItem is
    anchored to. Only `.pos()` is used by `_scene_xy`."""

    def __init__(self, x, y):
        self._pos = QtCore.QPointF(x, y)

    def pos(self):
        return self._pos


# ---------------------------------------------------------------------------
# 1. _scene_xy is the correct reconstruction; raw .points is NOT
# ---------------------------------------------------------------------------

def test_scene_xy_reconstructs_the_true_position_when_the_pixmap_is_offset():
    """The premise the fix depends on: when the anchoring pixmap sits away
    from the scene origin, raw pixel-local storage and the true scene
    position genuinely differ -- so a test asserting they must be treated
    differently cannot pass vacuously."""
    fake_pixmap = _FakePixmapItem(500.0, 300.0)
    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(12, 7)]), "pt1", True,
        pixmap_item=fake_pixmap, points_are_pixmap_local=True,
    )

    raw = item.points[0]
    true_scene = item._scene_xy(raw)

    assert (raw.x(), raw.y()) == (12, 7), "stored value must stay pixel-local"
    assert true_scene == (512, 307), (
        f"expected the pixmap offset (500, 300) added to the pixel-local "
        f"point (12, 7); got {true_scene}")
    assert true_scene != (raw.x(), raw.y()), (
        "raw .points and the true scene position must differ here, or this "
        "test cannot distinguish the fix from the bug")


def test_scene_xy_passes_through_unchanged_when_not_pixmap_local():
    """The non-pixmap-local branch (points_are_pixmap_local=False) already
    existed and is untouched by this fix -- confirm it still behaves as a
    plain passthrough."""
    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(40, 20)]), "pt2", True,
        pixmap_item=None, points_are_pixmap_local=False,
    )
    assert item._scene_xy(item.points[0]) == (40, 20)


# ---------------------------------------------------------------------------
# 2. the live-item -> saved-coordinate paths must use _scene_xy
# ---------------------------------------------------------------------------

def test_on_polygon_drawn_reconstructs_point_scene_coords():
    """THE regression: on_polygon_drawn must not treat a point item's raw
    .points as already being scene coordinates."""
    assert _calls_in(ProjectTab.on_polygon_drawn, "_scene_xy"), (
        "on_polygon_drawn no longer reconstructs point positions via "
        "_scene_xy -- it will feed pixmap-local integer pixels into "
        "scene_to_image_coords() as though they were scene coordinates, "
        "corrupting every newly drawn point's saved position")


def test_on_polygon_modified_reconstructs_point_scene_coords():
    """THE regression's twin: dragging/moving an existing point must go
    through the same reconstruction, not item.mapToScene() (a silent no-op
    for a point item, since its own transform is the identity)."""
    assert _calls_in(ProjectTab.on_polygon_modified, "_scene_xy"), (
        "on_polygon_modified no longer reconstructs point positions via "
        "_scene_xy -- moving an existing point will save the wrong "
        "coordinate the same way drawing a new one used to")


def test_polygon_items_are_unaffected_by_the_point_fix():
    """Guard against a regression in the other direction: EditablePolygonItem
    already stores genuine scene coordinates, so its branch must keep using
    `.polygon` / `mapToScene`, not `_scene_xy` (which polygons do not have)."""
    for fn in (ProjectTab.on_polygon_drawn, ProjectTab.on_polygon_modified):
        names = _names_in(fn)
        assert "polygon" in names, (
            f"{fn.__name__} lost its polygon-item branch while fixing the "
            "point-item one")


def test_scene_xy_helper_is_only_invoked_for_the_point_branch():
    """Sanity check on the fix's shape: the point-item reconstruction must be
    reachable only from the branch that already established `points`/`item`
    is a point item (via hasattr checks), not applied unconditionally."""
    for fn in (ProjectTab.on_polygon_drawn, ProjectTab.on_polygon_modified):
        tree = _func_tree(fn)
        # There must be at least one `hasattr(..., "points")` guard -- the
        # existing dispatch this fix was layered onto, not replaced.
        has_points_guard = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "hasattr"
            and len(n.args) == 2
            and isinstance(n.args[1], ast.Constant) and n.args[1].value == "points"
            for n in ast.walk(tree)
        )
        assert has_points_guard, (
            f"{fn.__name__} no longer gates the point-reconstruction branch "
            "on hasattr(..., 'points')")


# ---------------------------------------------------------------------------
# 3. dragging an existing point must not discard the drag
# ---------------------------------------------------------------------------
#
# THE REGRESSION THIS SECTION PINS: the first fix (this same session) for
# "points go stray on reload" corrected on_polygon_modified's use of
# _scene_xy() but dropped mapToScene() entirely -- silently assuming the
# item's own transform is always identity. It is not: EditablePointItem sets
# ItemIsMovable and never overrides itemChange/mouseMoveEvent, so Qt's
# DEFAULT drag translates item.pos() genuinely. on_polygon_modified fires on
# drag RELEASE, so without composing mapToScene() on top of _scene_xy(), a
# dragged-and-released point silently saved its PRE-drag position -- an
# active bug, not a historical one, introduced by fixing the first bug.

def test_scene_xy_alone_ignores_a_completed_drag():
    """Establishes the premise: _scene_xy() reads item-LOCAL coordinates and
    knows nothing about item.pos(). If a point item has been dragged (pos()
    is no longer the origin), _scene_xy() must NOT already reflect that --
    otherwise a test asserting mapToScene() is still required could pass
    vacuously."""
    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(12, 7)]), "pt3", True,
        pixmap_item=_FakePixmapItem(0, 0), points_are_pixmap_local=True,
    )
    before = item._scene_xy(item.points[0])

    item.setPos(QtCore.QPointF(200, 150))   # simulates a completed Qt drag
    after = item._scene_xy(item.points[0])

    assert before == after == (12, 7), (
        "_scene_xy() must be blind to item.pos() -- if it already reflected "
        "the drag, mapToScene() would be redundant and this whole test file "
        "would be testing nothing")


def test_maps_the_scene_xy_result_through_the_items_own_transform():
    """The actual fix: composing _scene_xy() (pixmap-local -> item-local)
    with item.mapToScene() (item-local -> scene, honouring item.pos()) is
    what correctly reflects a completed drag."""
    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(12, 7)]), "pt4", True,
        pixmap_item=_FakePixmapItem(0, 0), points_are_pixmap_local=True,
    )
    item.setPos(QtCore.QPointF(200, 150))

    local = QtCore.QPointF(*item._scene_xy(item.points[0]))
    true_scene = item.mapToScene(local)

    assert (true_scene.x(), true_scene.y()) == (212.0, 157.0), (
        f"expected item.pos() (200, 150) composed with the item-local point "
        f"(12, 7); got ({true_scene.x()}, {true_scene.y()})")
    assert (true_scene.x(), true_scene.y()) != (local.x(), local.y()), (
        "the mapped scene position must differ from the raw _scene_xy() "
        "result after a drag, or this test cannot distinguish the fix from "
        "the regression it pins")


def test_on_polygon_modified_composes_scene_xy_with_map_to_scene():
    """THE regression, at the source level: the point branch must call BOTH
    _scene_xy() and item.mapToScene() -- calling _scene_xy() alone silently
    discards any completed drag."""
    tree = _func_tree(ProjectTab.on_polygon_modified)

    map_to_scene_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "mapToScene"
    ]
    assert len(map_to_scene_calls) >= 2, (
        "on_polygon_modified must call item.mapToScene() at least twice -- "
        "once for the polygon branch (item.polygon) and once for the point "
        "branch (wrapping the _scene_xy()-reconstructed points) -- found "
        f"{len(map_to_scene_calls)}")

    assert _calls_in(ProjectTab.on_polygon_modified, "_scene_xy"), (
        "on_polygon_modified no longer reconstructs point-local coordinates "
        "via _scene_xy() at all")


def test_on_polygon_drawn_also_maps_through_the_items_transform():
    """The drawn-point branch is unaffected numerically (a fresh item's
    pos() is always the origin), but it should use the identical
    _scene_xy()-then-mapToScene() shape as on_polygon_modified rather than a
    second, subtly different reconstruction for the same underlying item
    type."""
    map_to_scene_calls = [
        n for n in ast.walk(_func_tree(ProjectTab.on_polygon_drawn))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "mapToScene"
    ]
    assert map_to_scene_calls, (
        "on_polygon_drawn's point branch does not call mapToScene() -- "
        "harmless today (a freshly created item's pos() is always the "
        "origin) but leaves two different reconstructions for the same "
        "EditablePointItem coordinate model")
