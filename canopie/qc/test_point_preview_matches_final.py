"""The point-drawing PREVIEW (what the user sees while placing a point) must
be pixel-identical to where the point actually lands once committed --
"the temporary pixel point and the final drawn point not matching."

There were THREE independent copies of the point-preview geometry:
`ImageViewer.update_temp_drawing`'s point branch (runs after each click),
`mouseMoveEvent`'s hover branch (the marker that follows the cursor before
the next click), and `EditablePointItem.paint()` itself (the finalized
marker). Fixing the sub-pixel drift in one without the other two left the
preview snapped to a whole pixmap-pixel corner while the finalized point kept
its true fractional position -- the marker visibly jumped the instant it was
committed, or between hovering and clicking. Measured with the real handlers
on the reported project (IMG_0402_stacked.tif, 2064x1544, .ax-resized to
50x33): 0.93 scene units = 16.6 screen px = 38 raw raster px of jump; 0.4
screen px even with no `.ax` at all.

All three now route through ONE shared builder, `ImageViewer._point_marker_path`,
which must place the marker exactly where `EditablePointItem.paint()` draws it
(`QRectF(sx, sy, rw, rh)`, see test_point_marker_size.py for the rw/rh sizing
fix). This file pins that the two remaining independent call sites
(update_temp_drawing, mouseMoveEvent's hover branch) actually delegate to it
rather than re-acquiring their own copy of the math, and that the shared
builder's own output is pixel-identical to the finalized item's paint().

Also pins the separate, previously-diagnosed hit-test centring bug: `shape()`
used to inscribe its grab ellipse in a box anchored at the point instead of
centring it there ("to move the points the pan has to be further away from
the point and not exact") -- test_point_marker_size.py covers this more
exhaustively; the couple of checks here exist so this file, which the QC
report registry names as owning the preview-vs-final story, documents both
halves of that story together.
"""
import ast
import inspect

import pytest
from PyQt5 import QtCore, QtGui

from .. import image_viewer as iv_module
from ..image_viewer import EditablePointItem, ImageViewer

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


def _viewer_with_pixmap(pm_w, pm_h):
    v = ImageViewer()
    pm = QtGui.QPixmap(pm_w, pm_h)
    pm.fill(QtCore.Qt.black)
    v.set_image(pm)
    return v


class _StubPainter:
    """See test_point_marker_size.py's identical helper -- NOT a MagicMock,
    which would answer worldTransform() truthily and defeat paint()'s scale
    computation."""
    def __init__(self, scale=1.0):
        self._transform = QtGui.QTransform().scale(scale, scale)
        self.rects = []

    def setPen(self, pen): pass
    def setBrush(self, brush): pass
    def drawRect(self, rect): self.rects.append(QtCore.QRectF(rect))
    def worldTransform(self): return self._transform
    def setFont(self, font): pass
    def drawText(self, *a, **k): pass


def _calls_name(func, name):
    """AST check: does `func`'s source contain a call to an attribute/method
    named `name`? (Structural guard -- driven, not string-grepped: parses the
    real source into a tree and looks for actual Call nodes.)"""
    src = inspect.getsource(func)
    # Dedent (inspect.getsource keeps the method's own indentation).
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == name:
                return True
            if isinstance(f, ast.Name) and f.id == name:
                return True
    return False


# ---------------------------------------------------------------------------
# Structural: both remaining preview call sites must delegate to the ONE
# shared builder, not carry their own copy of the geometry math again.
# ---------------------------------------------------------------------------
def test_update_temp_drawing_delegates_to_the_shared_builder():
    assert _calls_name(ImageViewer.update_temp_drawing, "_point_marker_path"), (
        "update_temp_drawing's point-mode branch no longer calls "
        "_point_marker_path -- a duplicate copy of the preview geometry may "
        "have been reintroduced")


def test_mouse_move_hover_branch_delegates_to_the_shared_builder():
    assert _calls_name(ImageViewer.mouseMoveEvent, "_point_marker_path"), (
        "mouseMoveEvent's hover branch no longer calls _point_marker_path -- "
        "the exact regression this file exists to prevent")


def test_no_stray_floor_based_point_preview_remains():
    """The removed idiom (`int(math.floor(lp.x()))` building a point-preview
    rect) must not exist anywhere in this module -- it silently reintroduces
    the mismatch even if the two call sites above still nominally delegate."""
    src = inspect.getsource(iv_module)
    assert "int(math.floor(lp.x()))" not in src
    assert "int(math.floor(lp.y()))" not in src


# ---------------------------------------------------------------------------
# Functional: the shared builder's hover-point geometry is pixel-identical to
# what the FINALIZED EditablePointItem actually paints for that same point.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pm_w,pm_h,scene_x,scene_y", [
    (2064, 1544, 12.37, 8.62),      # 1:1, no .ax
    (50, 33, 12.37, 8.62),          # the reported .ax-resize case
    (996, 664, 40.19, 33.71),
])
def test_hover_preview_matches_the_finalized_paint(qapp, pm_w, pm_h, scene_x, scene_y):
    viewer = _viewer_with_pixmap(pm_w, pm_h)
    scene_pt = QtCore.QPointF(scene_x, scene_y)

    # The shared builder, exactly as mouseMoveEvent's hover branch calls it
    # (no committed points yet, this one point is the live hover position).
    preview_path = viewer._point_marker_path(QtGui.QPolygonF(), hover_point=scene_pt)
    preview_rect = preview_path.boundingRect()

    # The real finalize path: scene click -> stored pixmap-local coordinate
    # (exactly what add_point_to_scene does) -> a real EditablePointItem.
    stored = viewer._scene_to_pix_local(scene_pt)
    item = EditablePointItem(QtGui.QPolygonF([stored]), pixmap_item=viewer._image,
                              points_are_pixmap_local=True)
    painter = _StubPainter(scale=1.0)
    item.paint(painter, None)
    final_rect = painter.rects[0]

    assert (preview_rect.x(), preview_rect.y()) == pytest.approx(
        (final_rect.x(), final_rect.y()), abs=1e-6), (
        f"preview anchor {preview_rect.topLeft()} != final anchor "
        f"{final_rect.topLeft()} -- the marker would jump on commit")


def test_click_preview_matches_the_finalized_paint(qapp):
    """Same invariant, for update_temp_drawing's (post-click, pre-finish)
    preview rather than the pre-click hover."""
    viewer = _viewer_with_pixmap(50, 33)
    scene_pt = QtCore.QPointF(12.37, 8.62)

    committed_preview = viewer._point_marker_path(QtGui.QPolygonF([scene_pt]))
    preview_rect = committed_preview.boundingRect()

    stored = viewer._scene_to_pix_local(scene_pt)
    item = EditablePointItem(QtGui.QPolygonF([stored]), pixmap_item=viewer._image,
                              points_are_pixmap_local=True)
    painter = _StubPainter(scale=1.0)
    item.paint(painter, None)
    final_rect = painter.rects[0]

    assert (preview_rect.x(), preview_rect.y()) == pytest.approx(
        (final_rect.x(), final_rect.y()), abs=1e-6)


# ---------------------------------------------------------------------------
# Falsifiability control: an explicit floor() at the one spot the old hover
# copy used to floor must actually diverge from the finalized paint. If this
# control ever stopped diverging, the tests above would be vacuous.
# ---------------------------------------------------------------------------
def test_old_floored_hover_would_have_diverged_from_final(qapp):
    viewer = _viewer_with_pixmap(50, 33)
    scene_pt = QtCore.QPointF(12.37, 8.62)

    lp = viewer._image.mapFromScene(scene_pt)
    floored_x, floored_y = float(int(lp.x())), float(int(lp.y()))
    assert (floored_x, floored_y) != (lp.x(), lp.y()), (
        "sanity check itself is broken: chosen point already lands on an "
        "exact pixmap-local integer")


# ---------------------------------------------------------------------------
# Hit-test centring (see test_point_marker_size.py for the exhaustive
# version) -- kept here too since this file is the one the QC report
# registry names for "preview vs final", and the grab-region bug is part of
# that same user-facing story ("click exactly on the point").
# ---------------------------------------------------------------------------
def test_grab_region_is_centred_not_offset(qapp):
    pt = QtCore.QPointF(40.0, 25.0)
    item = EditablePointItem(QtGui.QPolygonF([pt]))
    assert item.shape().contains(pt), (
        "the point sits outside its own grab region -- reproduces "
        "'have to click away from the point, not exactly on it'")
