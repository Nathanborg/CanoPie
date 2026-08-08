"""
Zoom-aware vertex decimation must speed up drawing WITHOUT changing any data.

WHY THIS EXISTS (reported as "still very slow -- how does QGIS load shapefiles
instantly without harming the viewer?"):

The real project's polygons are tree crowns traced at full raster resolution:

    2333 polygons, 1,010,806 vertices, mean 433 each, max 2312

Every one of those vertices was handed to the painter on every frame, at every
zoom. Measured cost of drawing them at three zooms:

    zoom 0.02   555 ms/frame ( 1.8 FPS)
    zoom 0.05   228 ms/frame ( 4.4 FPS)
    zoom 0.15    40 ms/frame (25.0 FPS)

QGIS does not do this: it applies distance-based simplification by default
(QgsVectorSimplifyMethod), discarding vertices that fall within about one
device pixel of their predecessor because they cannot be told apart on screen.
`_display_polygon` does the same thing here.

TWO THINGS THIS TEST GUARDS, both of which a careless version gets wrong:

1. IT IS DISPLAY-ONLY. `self._polygon`, `shape()` and every stored coordinate
   must remain bit-exact -- statistics, CSV export and hit-testing all read
   them. Simplification that mutates the geometry would silently corrupt
   scientific output, which is far worse than being slow.

2. IT MUST NOT FIRE WHEN IT CANNOT PAY. The expensive step is rebuilding a
   QPolygonF from Python QPointF objects, and during a pan that cost lands on
   almost every paint -- instrumenting a 30-frame pan measured 1556 rebuilds
   against 608 cache hits, because panning keeps revealing unseen polygons. A
   naive "simplify whenever anything can be dropped" rule therefore made
   zoomed-in panning SLOWER (10.0 -> 20.5 ms/frame at zoom 0.40). The guard
   requires ~4x more vertices than the outline has device pixels, i.e. only
   simplifies when ~75%+ of vertices go.
"""
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from ..image_viewer import EditablePolygonItem

pytestmark = [pytest.mark.viewer, pytest.mark.polygons]


def _dense_poly(n=600, r=1000.0, cx=5000.0, cy=5000.0):
    """A crown-like ring with many vertices, similar to the real data."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    wob = r * (1.0 + 0.05 * np.sin(ang * 7))
    return QtGui.QPolygonF([QtCore.QPointF(cx + wob[i] * np.cos(ang[i]),
                                           cy + wob[i] * np.sin(ang[i]))
                            for i in range(n)])


# ---------------------------------------------------------------------------
# 1. Display-only: the stored geometry is never touched
# ---------------------------------------------------------------------------
def test_stored_polygon_is_never_modified():
    """THE correctness guard. Statistics and export read `polygon`; if
    simplification mutated it, every downstream number would change."""
    item = EditablePolygonItem(_dense_poly(), "crown")
    try:
        before = [(item.polygon.at(i).x(), item.polygon.at(i).y())
                  for i in range(item.polygon.size())]
        for scale in (0.001, 0.01, 0.1, 1.0, 10.0):
            item._display_polygon(scale)
        after = [(item.polygon.at(i).x(), item.polygon.at(i).y())
                 for i in range(item.polygon.size())]
        assert after == before, (
            "simplification mutated the stored polygon -- polygon statistics "
            "and CSV export would silently change")
    finally:
        item.setParent(None)


def test_shape_used_for_hit_testing_keeps_full_precision():
    """Clicking/selecting must use the exact outline, not the drawn one."""
    poly = _dense_poly()
    item = EditablePolygonItem(poly, "crown")
    try:
        item._display_polygon(0.001)          # force an aggressive simplify
        shape_poly = item.shape().toFillPolygon()
        # toFillPolygon may append a closing point; allow one extra.
        assert shape_poly.size() >= poly.size(), (
            f"shape() dropped to {shape_poly.size()} points from "
            f"{poly.size()} -- hit testing would no longer match the real "
            "polygon outline")
    finally:
        item.setParent(None)


# ---------------------------------------------------------------------------
# 2. It actually simplifies when zoomed out
# ---------------------------------------------------------------------------
def test_simplifies_hard_when_zoomed_far_out():
    item = EditablePolygonItem(_dense_poly(n=600), "crown")
    try:
        drawn = item._display_polygon(0.002)   # whole scene on screen
        assert drawn.size() < 600 * 0.5, (
            f"only reduced 600 -> {drawn.size()} vertices at extreme zoom-out; "
            "the painter is still being handed detail no screen can show")
        assert drawn.size() >= 4, "simplified below a drawable polygon"
    finally:
        item.setParent(None)


def test_returns_the_exact_polygon_when_zoomed_in():
    """At high zoom every vertex is visible, so the guard must return the
    original object -- not a rebuilt copy, which is the expensive path."""
    poly = _dense_poly(n=600)
    item = EditablePolygonItem(poly, "crown")
    try:
        assert item._display_polygon(50.0) is item.polygon, (
            "a rebuilt QPolygonF was produced at high zoom where nothing can "
            "be dropped -- that rebuild cost is what made zoomed-in panning "
            "slower than doing nothing at all")
    finally:
        item.setParent(None)


def test_small_polygons_are_left_alone_entirely():
    """Below the vertex threshold there is nothing worth decimating."""
    poly = QtGui.QPolygonF([QtCore.QPointF(0, 0), QtCore.QPointF(10, 0),
                            QtCore.QPointF(10, 10), QtCore.QPointF(0, 10)])
    item = EditablePolygonItem(poly, "tiny")
    try:
        for scale in (0.001, 1.0, 100.0):
            assert item._display_polygon(scale) is item.polygon
    finally:
        item.setParent(None)


# ---------------------------------------------------------------------------
# 3. Caching behaviour
# ---------------------------------------------------------------------------
def test_repeated_paints_at_one_zoom_reuse_the_cache():
    """Panning at a fixed zoom must not rebuild the same polygon each frame."""
    item = EditablePolygonItem(_dense_poly(), "crown")
    try:
        first = item._display_polygon(0.002)
        for _ in range(20):
            assert item._display_polygon(0.002) is first, (
                "the simplified polygon was rebuilt on a repeat paint at the "
                "same zoom")
    finally:
        item.setParent(None)


def test_changing_geometry_invalidates_the_cache():
    item = EditablePolygonItem(_dense_poly(cx=5000.0), "crown")
    try:
        item._display_polygon(0.002)
        item.polygon = _dense_poly(cx=90000.0)      # move it far away
        drawn = item._display_polygon(0.002)
        assert drawn.boundingRect().center().x() > 50000, (
            "a stale simplified polygon from the previous geometry was reused "
            "after the polygon changed")
    finally:
        item.setParent(None)


def test_simplified_outline_still_tracks_the_original():
    """Decimation must not visibly deform the shape: the simplified bounding
    box should stay close to the true one."""
    poly = _dense_poly(n=600, r=1000.0)
    item = EditablePolygonItem(poly, "crown")
    try:
        true_r = poly.boundingRect()
        drawn_r = item._display_polygon(0.01).boundingRect()
        tol = 0.05 * max(true_r.width(), true_r.height())
        assert abs(drawn_r.width() - true_r.width()) < tol
        assert abs(drawn_r.height() - true_r.height()) < tol
    finally:
        item.setParent(None)
