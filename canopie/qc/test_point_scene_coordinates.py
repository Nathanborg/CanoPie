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


# ---------------------------------------------------------------------------
# 4. Other consumers of a point/polygon item's position must ALSO compose
#    _scene_xy() (or `.polygon`) with mapToScene() -- on_polygon_modified is
#    not the only reader. Found during a re-review of this fix: two more
#    call sites had their OWN independent pixmap/item-local -> scene
#    reconstruction, and both had the identical bug (computed the item-local
#    position correctly, then used it directly as "the scene position"
#    without ever composing item.mapToScene() on top -- so a completed drag
#    was silently discarded there too, even though the AST-level checks
#    above only ever looked at on_polygon_drawn/on_polygon_modified).
#
# These drive the REAL ImageViewer + real ProjectTab methods end-to-end
# (not a reimplementation), and simulate a completed drag the same way Qt's
# own default ItemIsMovable handling does: item.setPos(...), leaving the
# item's stored geometry (`.points` / `.polygon`) untouched -- see
# EditablePointItem.mouseReleaseEvent / EditablePolygonItem's itemChange,
# neither of which bakes the drag back into stored geometry.
# ---------------------------------------------------------------------------

class _FakeImageDataInMemory:
    """image_data with a non-existent filepath: polygon_basis_hw's file-probe
    (shapefile_io._raw_image_dims) fails to open it and falls back to
    `.image.shape`, so the basis is exactly the synthetic array below --
    keeping the expected scene<->image scale factor at a clean 1:1."""

    def __init__(self, arr):
        self.filepath = "C:/nonexistent/does_not_exist_qc_fixture.png"
        self.image = arr


def test_update_all_polygons_reflects_a_completed_point_drag(qapp):
    """THE regression this re-review found: update_all_polygons has its OWN,
    independent pixmap-local -> scene reconstruction for point items
    (separate from on_polygon_modified's), reached from root navigation and
    project save -- not the point_modified signal. It computed the
    item-local position (pixmap-item offset + stored pixel, i.e. exactly
    what _scene_xy does) but used that directly as the scene position,
    never composing item.mapToScene() on top. So dragging a point and then
    switching roots (which flushes dirty polygons through
    update_all_polygons before saving) silently reverted the point to its
    PRE-drag position on disk -- even though on_polygon_modified had just
    saved the correct post-drag position moments earlier.
    """
    import numpy as np
    from PyQt5 import QtGui, QtCore
    from ..image_viewer import ImageViewer

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 300)
    pm.fill(QtCore.Qt.black)
    v.set_image(pm)
    v.image_data = _FakeImageDataInMemory(np.zeros((300, 400, 3), np.uint8))

    point_item = v.add_point_to_scene(
        QtGui.QPolygonF([QtCore.QPointF(50, 60)]), name="pt_group")

    # Simulate a completed drag: Qt's default ItemIsMovable behaviour
    # translates item.pos() and never touches `.points` (confirmed by
    # EditablePointItem having no itemChange/mouseMoveEvent override that
    # would bake it back in).
    point_item.setPos(QtCore.QPointF(20, 5))

    pt = ProjectTab.__new__(ProjectTab)
    pt.all_polygons = {}
    pt.viewer_widgets = [{"viewer": v, "image_data": v.image_data}]
    # __new__ never runs QWidget.__init__, so any INSTANCE attribute that is
    # not pre-set here and gets touched (even via hasattr/getattr-with-
    # default) raises RuntimeError instead of the AttributeError plain
    # Python would give -- PyQt's own attribute-resolution fallback needs
    # the C++-side object, which does not exist. Pre-seed exactly what
    # _ensure_polygon_index/_poly_index_lookup/_add_to_polygon_index read,
    # so the index machinery takes its already-valid early-return path
    # instead of touching anything unset.
    pt._poly_exact_index = {}
    pt._poly_norm_index = {}
    pt._poly_norm_index_invalid = False

    pt.update_all_polygons()

    saved = pt.all_polygons.get("pt_group", {}).get(v.image_data.filepath)
    assert saved is not None, (
        "update_all_polygons did not write the dragged point at all -- "
        "check for a newly-introduced exception in the point branch")
    assert saved["points"] == [(70.0, 65.0)], (
        f"expected the drawn position (50, 60) plus the drag delta (20, 5) "
        f"= (70, 65); got {saved['points']} -- the drag was discarded, "
        f"which is the bug this test pins")


def test_serialize_item_reflects_a_completed_point_drag(qapp):
    """The same missing-mapToScene bug in ImageViewer._serialize_item, used
    by 'Copy points' / Ctrl+C (`copy_selection`/`copy_specific_items`) and
    'Replicate to all viewers' (`replicate_toviewer`): copying or
    replicating a point that had already been dragged silently used its
    PRE-drag position."""
    from PyQt5 import QtGui, QtCore
    from ..image_viewer import ImageViewer

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 300)
    pm.fill(QtCore.Qt.black)
    v.set_image(pm)

    point_item = v.add_point_to_scene(
        QtGui.QPolygonF([QtCore.QPointF(50, 60)]), name="pt")
    point_item.setPos(QtCore.QPointF(20, 5))

    payload = v._serialize_item(point_item)
    assert payload["points"] == [(70.0, 65.0)], (
        f"_serialize_item ignored the completed drag (item.pos() == "
        f"(20, 5)) and serialized the pre-drag position instead; got "
        f"{payload['points']}")


def test_serialize_item_reflects_a_completed_polygon_drag(qapp):
    """Same bug, polygon branch: `.polygon` holds genuine scene coordinates
    ONLY at construction time; after a drag it is item-local and still
    needs mapToScene()."""
    from PyQt5 import QtGui, QtCore
    from ..image_viewer import ImageViewer

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 300)
    pm.fill(QtCore.Qt.black)
    v.set_image(pm)

    poly_item = v.add_polygon_to_scene(
        QtGui.QPolygonF([QtCore.QPointF(10, 10), QtCore.QPointF(20, 10),
                          QtCore.QPointF(20, 20)]),
        name="poly")
    poly_item.setPos(QtCore.QPointF(5, 8))

    payload = v._serialize_item(poly_item)
    assert payload["points"] == [(15.0, 18.0), (25.0, 18.0), (25.0, 28.0)], (
        f"_serialize_item ignored the completed drag (item.pos() == "
        f"(5, 8)) on the polygon branch; got {payload['points']}")
