"""
A polygon item's boundingRect must not be wildly larger than the polygon.

THE BUG THIS PINS (reported as "with more than 5000 polygons it is impossible
to pan around the imagery" and "everything in the viewer drags, even moving a
polygon around"):

`EditablePolygonItem._invalidate_geometry_cache` unconditionally reserved space
for a worst-case label inside boundingRect:

    label_width  = len(self.name) * 50 + 100
    label_height = 250
    ... plus a 50 unit margin on every side

For a real name from the field ("guayacan_21294.0", 16 chars) that is a
1150 x 350 rect around a 40 x 40 polygon -- 251x the polygon's own area --
whether or not a label was ever going to be drawn.

Qt uses boundingRect for two things that both then blow up:

  * culling: which items count as exposed by a repaint region;
  * dirtying: under MinimalViewportUpdate (which this viewer uses), changing
    an item marks its whole boundingRect dirty. 1150x350 instead of 60x60 is
    ~112x more pixels to repaint per item, per frame.

Measured with a real QGraphicsView, 1200x800 viewport, 6000 items:

    inflated boundingRect : 67.3 ms/frame   (~15 FPS -- "impossible to pan")
    tight boundingRect    : 27.4 ms/frame   (2.5x faster)

THE FIX: only reserve the label area when a label can ACTUALLY be painted --
`paint()` draws it only when `show_label or _hover_showing_label`, so when both
are false nothing is drawn outside the polygon and the tight rect is not just
faster but strictly correct. Labels off is the normal state for a project with
thousands of polygons, i.e. exactly when it matters.

Because boundingRect now DEPENDS on those two flags, both are properties that
call prepareGeometryChange() when flipped -- otherwise Qt keeps the stale rect
and leaves label artifacts behind (this codebase already carries several
full-scene invalidate() calls added to chase exactly that class of ghosting).
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from ..image_viewer import EditablePolygonItem

pytestmark = [pytest.mark.viewer, pytest.mark.polygons]

LONG_NAME = "guayacan_21294.0"      # a real name from the user's data


def _poly(x=0.0, y=0.0, w=40.0, h=40.0):
    return QtGui.QPolygonF([QtCore.QPointF(x, y), QtCore.QPointF(x + w, y),
                            QtCore.QPointF(x + w, y + h), QtCore.QPointF(x, y + h)])


def test_labels_off_does_not_reserve_label_space():
    """THE regression, stated as the invariant that actually matters.

    With no label to draw, boundingRect must be the polygon plus a BOUNDED
    CONSTANT (the stroke margin) -- never a term that grows with the name.
    Asserting a raw area ratio would be a bad test here: for a small polygon
    the constant stroke margin legitimately dominates, so the ratio says more
    about the polygon's size than about the bug.
    """
    item = EditablePolygonItem(_poly(), LONG_NAME)
    try:
        item.show_label = False
        br = item.boundingRect()
        poly = _poly().boundingRect()
        # Whatever margin the item adds must be symmetric and modest -- and
        # crucially must not encode the label box (len(name)*50 + 100 = 900
        # for this name).
        margin_x = (br.width() - poly.width()) / 2.0
        margin_y = (br.height() - poly.height()) / 2.0
        assert margin_x < 100.0 and margin_y < 100.0, (
            f"boundingRect adds {margin_x:.0f}x{margin_y:.0f} of padding around "
            "a polygon whose label is not drawn. Qt dirties this whole rect on "
            "every change under MinimalViewportUpdate, so padding multiplies "
            "the pixels repainted per pan frame")
    finally:
        item.setParent(None)


def test_bounding_rect_is_independent_of_name_length_when_labels_are_off():
    """THE sharpest form of the regression: the old code sized the rect from
    `len(name) * 50`, so a long name cost far more to repaint than a short one
    even with labels off. Two items differing ONLY in name length must now
    produce identical rects."""
    short = EditablePolygonItem(_poly(), "a")
    long_ = EditablePolygonItem(_poly(), LONG_NAME * 3)
    try:
        short.show_label = False
        long_.show_label = False
        assert short.boundingRect() == long_.boundingRect(), (
            f"a {len(LONG_NAME)*3}-char name produced "
            f"{long_.boundingRect().width():.0f} wide vs "
            f"{short.boundingRect().width():.0f} for a 1-char name, with "
            "labels off -- boundingRect still scales with the label text")
    finally:
        short.setParent(None)
        long_.setParent(None)


def test_bounding_rect_still_covers_the_label_when_labels_are_on():
    """The fix must not trade correctness for speed: when a label IS drawn, its
    area must still be inside boundingRect, or Qt leaves ghost text behind."""
    item = EditablePolygonItem(_poly(), LONG_NAME)
    try:
        item.show_label = True
        br = item.boundingRect()
        # paint() puts the label at bbox.topRight() + label_offset
        anchor = _poly().boundingRect().topRight() + item.label_offset
        assert br.contains(anchor), (
            f"label anchor {anchor} is outside boundingRect {br}")
        assert br.width() > 200, (
            "boundingRect no longer reserves any horizontal room for the "
            "label while labels are ON -- text painted outside boundingRect "
            "leaves artifacts that only a full-scene invalidate can clear")
    finally:
        item.setParent(None)


def test_toggling_labels_updates_the_bounding_rect():
    """boundingRect depends on show_label, so flipping it must recompute."""
    item = EditablePolygonItem(_poly(), LONG_NAME)
    try:
        item.show_label = True
        wide = item.boundingRect()
        item.show_label = False
        narrow = item.boundingRect()
        assert narrow.width() < wide.width(), (
            "turning labels off did not shrink boundingRect -- the cached rect "
            "is stale, so the item keeps paying the inflated repaint cost")
        item.show_label = True
        assert item.boundingRect().width() == pytest.approx(wide.width()), (
            "turning labels back on did not restore the label allowance")
    finally:
        item.setParent(None)


def test_hover_label_also_reserves_room():
    """Hover temporarily shows a label on an item whose labels are off, so it
    must expand the rect the same way."""
    item = EditablePolygonItem(_poly(), LONG_NAME)
    try:
        item.show_label = False
        tight = item.boundingRect().width()
        item._hover_showing_label = True
        assert item.boundingRect().width() > tight, (
            "a hover-revealed label is painted outside a rect that was never "
            "expanded for it -- classic ghost-label artifact")
        item._hover_showing_label = False
        assert item.boundingRect().width() == pytest.approx(tight)
    finally:
        item.setParent(None)


def test_geometry_change_is_announced_when_labels_toggle(qapp):
    """prepareGeometryChange() must fire, or the SCENE keeps the old rect in
    its index no matter what boundingRect() now returns."""
    scene = QtWidgets.QGraphicsScene()
    item = EditablePolygonItem(_poly(), LONG_NAME)
    scene.addItem(item)
    try:
        item.show_label = True
        before = item.sceneBoundingRect().width()
        item.show_label = False
        after = item.sceneBoundingRect().width()
        assert after < before, (
            "the scene's view of this item's bounds did not shrink when labels "
            "were turned off -- prepareGeometryChange() was not called, so the "
            "scene index still holds the inflated rect")
    finally:
        scene.removeItem(item)
        scene.clear()


def test_unnamed_polygon_never_reserves_label_space():
    """No name means no label at any time, labels-on or not."""
    item = EditablePolygonItem(_poly(), "")
    try:
        poly = _poly().boundingRect()
        for flag in (True, False):
            item.show_label = flag
            br = item.boundingRect()
            assert (br.width() - poly.width()) / 2.0 < 100.0, (
                f"an unnamed polygon reserved label room (show_label={flag})")
    finally:
        item.setParent(None)


@pytest.mark.perf
def test_many_items_with_labels_off_stay_cheap_to_dirty(qapp):
    """Aggregate guard: the total dirty area of a screenful of polygons with
    labels off must stay proportional to the polygons, not to a worst-case
    label box each."""
    scene = QtWidgets.QGraphicsScene()
    scene.setSceneRect(0, 0, 4000, 4000)
    items = []
    try:
        for i in range(300):
            it = EditablePolygonItem(_poly((i * 13) % 3900, (i * 29) % 3900), f"{LONG_NAME}_{i}")
            it.show_label = False
            scene.addItem(it)
            items.append(it)

        total = sum(it.boundingRect().width() * it.boundingRect().height() for it in items)
        # With the label box included this was (40+900+100) x (40+250+100)
        # = ~406k per item. Without it, it is bounded by the stroke margin.
        per_item = total / len(items)
        assert per_item < 40000.0, (
            f"average boundingRect area is {per_item:.0f} scene-units^2 per "
            "40x40 polygon with labels off (label-inclusive was ~406000); "
            "panning cost scales directly with this")
    finally:
        for it in items:
            scene.removeItem(it)
        scene.clear()
