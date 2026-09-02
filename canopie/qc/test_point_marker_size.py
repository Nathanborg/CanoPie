"""A drawn point must represent exactly ONE RAW raster pixel -- on screen and
in its hit-test region -- regardless of how heavily the displayed pixmap is
decimated.

THE BUG (reported on project C:\\New Folder221, a real 28315x25548x4 orthomosaic
with ZERO embedded pyramid levels, forcing raster_reader to live-decimate the
only available preview 3x): `EditablePointItem.paint()` drew every point as a
hardcoded `QRectF(sx, sy, 1, 1)` -- 1 unit in PIXMAP-LOCAL (displayed) space,
not raw-pixel space. Points are stored/exported in raw coordinates everywhere
else in the app (CSV stats, ML classification), so a marker meant "1 pixel of
whatever preview happens to be on screen", not "1 raw pixel". On a lightly
decimated or 1:1 image this is imperceptible; on this project's forced 3x
decimation the marker rendered 3x too large linearly (9x in area) -- "way
bigger point square pixels than what the viewer shows". `shape()`'s hit region
had the same defect, plus a separate, independent centring bug
(`addEllipse(sx, sy, 6, 6)` inscribes rather than centers the ellipse, putting
the point itself outside its own grab region).

None of this project's earlier, already-fixed point-POSITION tests
(test_point_scene_coordinates.py, test_point_drag_refresh_stability.py) caught
it, because every one of them used `.ax`-driven crop/rotate/resize or a small
image shown near 1:1 -- never `raster_reader`'s live lazy-preview decimation
path, the ONLY mechanism that forces a large raw/pixmap ratio with no `.ax`
involved at all. This file exercises exactly that path, reusing the same
conftest fixtures `test_forced_preview_decimation` already proves force real
spatial decimation (empirically: step=2, a (32,32,3) preview decimated to
(16,16,3), no multi-GB file needed).
"""
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from ..image_viewer import EditablePointItem, POINT_GRAB_RADIUS
from ..shapefile_io import _raw_image_dims
from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]


class _FakeImageData:
    def __init__(self, filepath, arr):
        self.filepath = filepath
        self.image = arr


class _StubPainter:
    """A narrow, explicit stand-in for QPainter -- NOT a MagicMock, which
    would answer worldTransform() with a truthy Mock and silently defeat the
    scale computation paint() depends on (a documented trap in this repo's
    own QC conventions)."""

    def __init__(self, scale=1.0):
        self._transform = QtGui.QTransform().scale(scale, scale)
        self.rects = []
        self.pen = None
        self.brush = None
        self.font = None

    def setPen(self, pen):
        self.pen = pen

    def setBrush(self, brush):
        self.brush = brush

    def drawRect(self, rect):
        self.rects.append(QtCore.QRectF(rect))

    def worldTransform(self):
        return self._transform

    def setFont(self, font):
        self.font = font

    def drawText(self, *a, **k):
        pass


def _real_ratio_from_decimated_pixmap(fp, decimation):
    """A REAL raw_to_pixmap_scale, computed the same way _raw_to_pixmap_scale
    does (raw header dims / a chosen pixmap size) -- not hand-set on the item."""
    raw_h, raw_w = _raw_image_dims(fp)
    assert raw_h and raw_w
    pm_w, pm_h = max(1, raw_w // decimation), max(1, raw_h // decimation)
    return (raw_w / float(pm_w), raw_h / float(pm_h))


# ---------------------------------------------------------------------------
# 1) End-to-end: the real add_point_to_scene -> _raw_to_pixmap_scale ->
#    polygon_basis_hw chain, under REAL forced raster_reader decimation.
# ---------------------------------------------------------------------------
def test_add_point_to_scene_computes_ratio_under_real_forced_decimation(
        synthetic_project, force_preview_decimation, viewer_factory):
    from .fixtures_manifest import get_fixture

    name = "hyperspectral_200band"
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    assert synthetic_project._last_loader == "raster_reader-preview"

    step = max(1, int(round(lite.display_scale)))
    if step <= 1:
        pytest.skip("decode budget did not force decimation on this fixture")

    spec = get_fixture(name)
    raw_h, raw_w = spec["height"], spec["width"]
    pm_h, pm_w = lite.image.shape[:2]

    pm = QtGui.QPixmap(pm_w, pm_h)
    pm.fill(QtCore.Qt.black)
    viewer = viewer_factory()
    viewer.setParent(synthetic_project)   # so _find_project_owner() resolves polygon_basis_hw
    viewer.set_image(pm)
    viewer.image_data = lite

    item = viewer.add_point_to_scene(
        QtGui.QPolygonF([QtCore.QPointF(2.0, 2.0)]), name="p")

    assert item.raw_to_pixmap_scale == pytest.approx(
        (raw_w / pm_w, raw_h / pm_h), rel=1e-6)
    # And it must actually equal the decimation step, not just be self-consistent.
    assert item.raw_to_pixmap_scale == pytest.approx((float(step), float(step)), rel=0.5)


# ---------------------------------------------------------------------------
# 2) paint() draws a marker sized to a real, non-trivial ratio.
# ---------------------------------------------------------------------------
def test_paint_draws_marker_sized_to_raw_pixmap_ratio():
    fp = fixture_image_path("rgb_8bit_untiled")
    ratio = _real_ratio_from_decimated_pixmap(fp, decimation=4)

    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(5.0, 5.0)]),
        raw_to_pixmap_scale=ratio)

    painter = _StubPainter(scale=1.0)   # scale=1.0 -> the min-device-px floor cannot engage
    item.paint(painter, None)

    assert len(painter.rects) == 1
    r = painter.rects[0]
    assert r.width() == pytest.approx(ratio[0])
    assert r.height() == pytest.approx(ratio[1])
    assert (r.width(), r.height()) != (1.0, 1.0)


# ---------------------------------------------------------------------------
# 3) paint()'s drawn size is INDEPENDENT of the live view scale -- it must
#    stay purely a function of item state (raw_to_pixmap_scale). This is
#    deliberate, not an oversight: an earlier version of this fix floored the
#    size at a minimum device-pixel count (`POINT_MIN_DEVICE_PX / scale`), but
#    boundingRect()/shape() are cached once from item state alone and cannot
#    see the view transform -- test 6 below caught that no fixed cache could
#    then guarantee containment at extreme zoom-out (view scale 0.02, exactly
#    what "fit to window" needs on the real 28315px-wide file this bug was
#    reported against). Removed rather than patched around.
# ---------------------------------------------------------------------------
def test_paint_size_does_not_depend_on_view_scale():
    item = EditablePointItem(
        QtGui.QPolygonF([QtCore.QPointF(0.0, 0.0)]),
        raw_to_pixmap_scale=(4.0, 4.0))

    zoomed_in = _StubPainter(scale=20.0)
    zoomed_out = _StubPainter(scale=0.001)
    item.paint(zoomed_in, None)
    item.paint(zoomed_out, None)

    r_in, r_out = zoomed_in.rects[0], zoomed_out.rects[0]
    assert (r_in.width(), r_in.height()) == pytest.approx((4.0, 4.0))
    assert (r_out.width(), r_out.height()) == pytest.approx((4.0, 4.0))


# ---------------------------------------------------------------------------
# 4) shape()'s grab ellipse is centred on the point (the hit-test bug).
# ---------------------------------------------------------------------------
def test_shape_grab_ellipse_is_centred_on_the_point():
    pt = QtCore.QPointF(40.0, 25.0)
    item = EditablePointItem(QtGui.QPolygonF([pt]))

    shape = item.shape()
    assert shape.contains(pt), (
        "the point is outside its own grab region -- this is the "
        "'have to click away from the point, not exactly on it' bug")

    bounds = shape.boundingRect()
    cx, cy = bounds.center().x(), bounds.center().y()
    assert cx == pytest.approx(pt.x(), abs=0.5)
    assert cy == pytest.approx(pt.y(), abs=0.5)


# ---------------------------------------------------------------------------
# 5) Grab radius grows with decimation but never shrinks below the legacy size.
# ---------------------------------------------------------------------------
def test_shape_grab_radius_grows_with_decimation_but_never_shrinks():
    pt = QtCore.QPointF(10.0, 10.0)
    small = EditablePointItem(QtGui.QPolygonF([pt]), raw_to_pixmap_scale=(1.0, 1.0))
    big = EditablePointItem(QtGui.QPolygonF([pt]), raw_to_pixmap_scale=(30.0, 30.0))

    small_extent = small.shape().boundingRect().width()
    big_extent = big.shape().boundingRect().width()

    assert big_extent > small_extent
    # `pytest.approx()` only supports `==`/`!=`, not `>=` -- compare directly.
    assert small_extent >= 2 * POINT_GRAB_RADIUS - 0.01


# ---------------------------------------------------------------------------
# 6) boundingRect() actually contains what paint() draws, across a matrix of
#    ratios x view scales -- the concrete, numeric version of "don't trust the
#    padding constant by inspection alone".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [1.0, 4.0, 30.0])
@pytest.mark.parametrize("view_scale", [1.0, 0.02])
def test_bounding_rect_contains_the_actual_painted_marker(ratio, view_scale):
    pt = QtCore.QPointF(100.0, 80.0)
    item = EditablePointItem(QtGui.QPolygonF([pt]), raw_to_pixmap_scale=(ratio, ratio))

    painter = _StubPainter(scale=view_scale)
    item.paint(painter, None)
    painted = painter.rects[0]

    brect = item.boundingRect()
    assert brect.contains(painted), (
        f"boundingRect() {brect} does not contain the painted marker {painted} "
        f"(ratio={ratio}, view_scale={view_scale})")


# ---------------------------------------------------------------------------
# Falsifiability control: a fixed (1.0, 1.0) ratio must NOT satisfy test 2/6
# at a real decimation ratio -- i.e. these tests genuinely depend on the fix.
# ---------------------------------------------------------------------------
def test_old_hardcoded_1x1_marker_would_fail_the_ratio_check():
    fp = fixture_image_path("rgb_8bit_untiled")
    ratio = _real_ratio_from_decimated_pixmap(fp, decimation=4)
    assert ratio[0] > 1.0 and ratio[1] > 1.0, "fixture choice no longer forces decimation"

    item = EditablePointItem(QtGui.QPolygonF([QtCore.QPointF(5.0, 5.0)]))  # default (1,1)
    painter = _StubPainter(scale=1.0)
    item.paint(painter, None)
    r = painter.rects[0]
    assert (r.width(), r.height()) != pytest.approx(ratio), (
        "sanity check itself is broken: a (1,1)-ratio item accidentally "
        "matches the real decimation ratio")
