"""
Two viewer-throughput fixes that are easy to silently undo.

1. LABELS DEFAULT OFF. A label makes the item reserve `len(name)*50 + 100` by
   250 scene units of boundingRect for text that may never be drawn, so Qt's
   culling hands back far more items than are on screen AND each one rasterises
   text. Measured on 10000 generated circles with realistic crown names:

       zoom   labels ON   labels OFF
       0.40      51.2 ms      10.0 ms      5.1x
       0.70      27.1 ms       9.4 ms      2.9x
       1.00      18.9 ms       6.3 ms      3.0x

   At zoom 0.25, labels-on painted 967 items where labels-off painted 696 --
   39% of the work was for polygons that were not visible at all.

2. IMAGE->SCENE MAPPING uses the whole-polygon mapToScene overload, not one
   call per vertex. 1.6x on 2.45M vertices, bit-identical.
"""
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import (
    ImageViewer, EditablePolygonItem, EditablePointItem, image_points_to_scene)


NAME = "guayacan_21294.0_20240716_bciclippled_rx1ii"


def _ring(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return QtGui.QPolygonF(
        [QtCore.QPointF(float(cx + r * np.cos(t)), float(cy + r * np.sin(t))) for t in a])


# ---------------------------------------------------------------------------
# 1. label defaults
# ---------------------------------------------------------------------------
def test_polygon_item_labels_default_off(qapp):
    it = EditablePolygonItem(_ring(100, 100, 20), NAME)
    assert it.show_label is False, (
        "labels back on by default: every polygon pays label-sized "
        "boundingRect inflation and text rasterisation")


def test_point_item_labels_default_off(qapp):
    poly = QtGui.QPolygonF([QtCore.QPointF(10, 10)])
    it = EditablePointItem(poly, NAME)
    assert it.show_label is False


def test_viewer_labels_default_off(qapp):
    assert ImageViewer()._labels_visible is False


def test_manager_checkbox_defaults_unchecked(qapp):
    """The user turns labels on deliberately; they are not the default state."""
    from canopie.polygon_manager import PolygonManager
    import inspect
    src = inspect.getsource(PolygonManager)
    assert "self.show_labels_checkbox.setChecked(False)" in src, (
        "'Show Labels' is checked by default again -- the viewer default and "
        "the checkbox must agree, or the first repaint re-enables labels")


def test_label_off_keeps_the_bounding_rect_tight(qapp):
    """The actual mechanism behind the speedup."""
    it = EditablePolygonItem(_ring(1000, 1000, 20), NAME)
    tight = it.boundingRect()
    it.show_label = True
    wide = it.boundingRect()

    assert wide.width() > tight.width() * 2, (
        "turning labels on should visibly inflate the rect -- if it does not, "
        "this test can no longer detect the regression it guards")
    poly_area = 40.0 * 40.0
    assert tight.width() * tight.height() < poly_area * 20, (
        f"default rect {tight} is not tight around a 40x40 polygon")


@pytest.mark.perf
def test_labels_off_is_faster_when_zoomed_in(qapp):
    IMG = 20000
    N = 2500

    def build(labels):
        v = ImageViewer()
        pm = QtGui.QPixmap(2000, 2000)
        v._scene.clear()
        v._image = v._scene.addPixmap(pm)
        v._image.setScale(IMG / 2000.0)
        v._scene.setSceneRect(0, 0, IMG, IMG)
        v.polygons = []
        v.resize(900, 700)
        side = int(np.ceil(np.sqrt(N)))
        step = IMG / side
        k = 0
        for iy in range(side):
            for ix in range(side):
                if k >= N:
                    break
                it = v.add_polygon_to_scene(
                    _ring(ix * step + step / 2, iy * step + step / 2, step * 0.3),
                    name=f"{NAME}_{k}")
                it.show_label = labels
                k += 1
        return v

    def painted_and_ms(v, scale, reps=5):
        """Count items Qt actually paints, and time the frame.

        The COUNT is the deterministic part and what the assertion uses. A
        wall-clock comparison at this size lands in the 1-3 ms range where
        run-to-run noise exceeds the effect -- that made this test flaky, and a
        flaky perf test is worse than none.
        """
        v.resetTransform()
        v.scale(scale, scale)
        v._set_batch_active(False)      # this is about per-item cost
        img = QtGui.QImage(v.viewport().size(), QtGui.QImage.Format_ARGB32)
        step = (v.viewport().width() / scale) * 0.25

        count = {"n": 0}
        orig = EditablePolygonItem.paint

        def counting(self, painter, option, widget=None):
            count["n"] += 1
            return orig(self, painter, option, widget)

        EditablePolygonItem.paint = counting
        try:
            ts = []
            for k in range(reps):
                v.centerOn(IMG / 2 + k * step, IMG / 2)
                p = QtGui.QPainter(img)
                t = time.perf_counter()
                v._scene.render(p, QtCore.QRectF(img.rect()),
                                v.mapToScene(v.viewport().rect()).boundingRect())
                ts.append((time.perf_counter() - t) * 1000.0)
                p.end()
        finally:
            EditablePolygonItem.paint = orig
        return count["n"], float(np.median(ts))

    n_on, ms_on = painted_and_ms(build(True), 0.5)
    n_off, ms_off = painted_and_ms(build(False), 0.5)
    print(f"\nzoom 0.5, {N} polygons: labels ON {n_on} paints / {ms_on:.1f} ms, "
          f"OFF {n_off} paints / {ms_off:.1f} ms")

    assert n_off < n_on, (
        f"labels-off must make Qt cull MORE items ({n_off} vs {n_on} paints). "
        f"The label box inflates boundingRect, so Qt hands back polygons that "
        f"are nowhere near the viewport -- that is the mechanism behind the "
        f"3-5x, and it is what this asserts.")


# ---------------------------------------------------------------------------
# 2. image -> scene mapping
# ---------------------------------------------------------------------------
def _per_vertex_reference(pixitem, pts, hw):
    pm = pixitem.pixmap()
    pw, ph = pm.width(), pm.height()
    ih, iw = hw
    return QtGui.QPolygonF([
        pixitem.mapToScene(QtCore.QPointF(float(x) * (pw / float(iw)),
                                          float(y) * (ph / float(ih))))
        for x, y in pts])


@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_mapping_matches_per_vertex_maptoscene(qapp, rotation):
    """Bit-identical to the loop it replaced -- including under rotation.

    An inline-affine version using only m11/m22 was 1.8x faster still, but
    silently drops m12/m21: with a rotated pixmap item every imported polygon
    would land in the wrong place. That is why this is the whole-polygon Qt
    overload and not hand-rolled maths.
    """
    scene = QtWidgets.QGraphicsScene()
    pm = QtGui.QPixmap(256, 256)
    pm.fill(QtCore.Qt.darkGreen)
    pixitem = scene.addPixmap(pm)
    pixitem.setScale(4.0)
    pixitem.setPos(17.0, -23.0)
    pixitem.setRotation(rotation)

    rng = np.random.default_rng(11)
    pts = rng.uniform(0, 1024, size=(500, 2)).tolist()
    hw = (1024, 1024)

    got = image_points_to_scene(pixitem, pts, hw)
    ref = _per_vertex_reference(pixitem, pts, hw)

    assert len(got) == len(ref)
    worst = max(abs(got[i].x() - ref[i].x()) + abs(got[i].y() - ref[i].y())
                for i in range(len(ref)))
    assert worst == 0.0, f"mapping drifted from Qt's own by {worst:.3e}"


def test_mapping_handles_empty_and_missing_pixmap(qapp):
    assert image_points_to_scene(None, []) .isEmpty()
    got = image_points_to_scene(None, [(1.0, 2.0), (3.0, 4.0)])
    assert len(got) == 2 and got[1].x() == 3.0
