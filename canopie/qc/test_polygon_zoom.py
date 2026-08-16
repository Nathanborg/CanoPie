"""
QC regression tests for "click group in list" / "zoom to polygon" behavior.

THE PROBLEM THIS FIXES
-----------------------
Both actions used to be two independent, hand-duplicated implementations,
sizing the zoom rectangle with HARDCODED ABSOLUTE SCENE-UNIT CONSTANTS
(`min_dim = 100.0`, `pad = max(50.0, dim*0.20)`) and forcing a NEW computed
zoom on every click. A first pass unified them behind one shared
fit-based method with a native-resolution-aware cap -- but on a COG, ANY
forced re-fit is a jarring, unwanted zoom jump away from whatever the user
was already comfortably looking at, even capped at 200% of native
resolution. That's not what was actually wanted.

THE ACTUAL DESIGN (per explicit product direction)
-----------------------------------------------------
  * Clicking a group in the list (`PolygonManager._select_group_in_viewers`)
    is ALWAYS pan-only: it never changes the current zoom level, for a
    polygon OR a point -- exactly like clicking between rows in a list
    keeps its own scroll position. `ImageViewer.smart_zoom_to_scene_rect`
    does nothing but `centerOn(rect.center())`.
  * "Zoom to Polygon" / double-click (`PolygonManager.zoom_to_groups`)
    actually changes zoom -- but ONLY for a target with real extent. A
    POINT has no natural "fit" scale (any zoom "fits" a dimensionless
    target), so forcing one was the actual root cause of "zoom is too high"
    -- a point-only (or all-points) selection is routed to the same
    pan-only `smart_zoom_to_scene_rect` list-click uses. A real polygon
    goes through `ImageViewer.smart_fit_to_scene_rect`: relative/
    resolution-scaled padding (never the old absolute min_dim=100.0), a
    fitInView, and a scale clamp to `max_native_scale` (200%) of NATIVE
    resolution -- not of whatever pixmap happens to be cached right now,
    which on a COG is often a decimated preview (confirmed in this
    codebase's own comments, project_tab.py:20338-20340: a 3001x3131
    preview for a 48031x50101 native raster, a ~16x gap). Tests below
    simulate exactly that gap by passing a `native_hw` deliberately larger
    than the synthetic image's own pixel dimensions -- no real COG needed.

Two additional, non-obvious Qt bugs were found and fixed while wiring this
up (pinned separately below, not just implied by the main tests):
  * QRectF.isValid() is False for an exact zero-width/zero-height rect --
    which is exactly what a single point's bounding rect is. The OLD zoom
    methods' own `if r and r.isValid():` guard would have skipped a true
    point entirely; the new code checks `r is not None` instead.
  * QRectF.united() silently DROPS a null/zero-size operand instead of
    expanding to include its position (a point unioned with anything just
    returns the other operand unchanged; two points union to a degenerate
    rect at the origin) -- so a multi-group zoom mixing points and polygons
    would have silently lost every point's position. zoom_to_groups now
    unions by min/max corners instead of QRectF.united().

THE REAL PRODUCTION BUG, FOUND BY REPRODUCING AGAINST A REAL USER PROJECT
---------------------------------------------------------------------------
Reported: "zoom to polygon points is failing miserably ... zooming out of
the image boundaries" on two real projects (C:\\RGB, C:\\New Folder196).
Reproduced directly against C:\\RGB's real project.json + real polygon
sidecar + the real external source image
(C:/Multispect_analyser/test/Jpeg/20221212_103009_759_8b.JPG, 4000x3000):

`_get_polygon_scene_rect` (polygon_manager.py) computed its scale
denominator from `viewer.sceneRect()` -- the QGraphicsView's PADDED
scrollable extent (image + ImageViewer._SCENE_PAD_FRACTION, 20%, on every
side: a 4000x3000 image gets a 5600x4200 sceneRect) -- instead of the
DISPLAYED PIXMAP's own width/height. The pixmap item always sits at scene
position (0,0) with an identity transform (image_viewer.py's set_image
calls `_image.setPos(0, 0)`), so a point's fraction-of-image position must
be rescaled against the pixmap's own size to land back on the image;
scaling by the padded sceneRect instead placed every point ~1.4x further
out, with no correction for the padding's negative left/top offset either.
Reproduced exactly: a real polygon whose points are (1960-2068, 2411-2493)
in its own 4000x3000 image mapped to scene (2744-2895, 3376-3491) -- Y
entirely past the image's own 3000px height, landing the "zoomed to" view
fully below the image, showing nothing but blank padding. Confirmed as
THE cause (not just a contributing factor) by revert-verification: with
only this one fix reverted, the reported symptom reproduces exactly and
every other change stays in place. This bug predates this session's zoom
work entirely (same buggy `_get_polygon_scene_rect`, untouched by earlier
fixes here) -- the OLD hardcoded-constant zoom (min_dim=100/pad=50, loose
and imprecise) happened to still show SOME of the image despite the
mis-centering; this session's more precise centerOn/fitInView made the
pre-existing bug fully visible for the first time.

A SECOND THING FOUND WHILE INVESTIGATING -- A SIMPLIFICATION, NOT A BUG
---------------------------------------------------------------------------
`smart_fit_to_scene_rect`'s pre-fitInView "floor" (meant only to keep a
near-zero-area rect from breaking fitInView) computed
`vp_w*cached_pw / (max_native_scale*native_w)`, which for any image where
the cached pixmap is already at (or near) native resolution -- any plain
JPEG/TIFF, no COG decimation gap -- collapses to roughly "half the
viewport size", disconnected from the polygon's or image's own size. This
LOOKED like a second, independent bug (a large floor could in principle
push the framed rect past an off-center polygon's image edge) -- but
revert-verification of this change ALONE (with the real bug above still
fixed) showed NO test in this file distinguishes the two floor formulas:
floor-expansion is always symmetric around the rect's own center, so it
never moves centerOn's target, and whenever native_hw is supplied and a
small polygon needs real zoom-in, the exact post-fitInView clamp (step 3)
corrects the final scale to precisely max_native_scale*settle_back
regardless of what the floor produced first -- and when native_hw is
None, the old and new formulas are the literal same expression. Kept as a
code-quality simplification (removes a formula that looks
resolution-aware but never needed to be) rather than framed as an
independent bug fix.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer
from canopie.polygon_manager import PolygonManager

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer, pytest.mark.polygons]


def _viewer_with_image(viewer_factory, w=1200, h=900, img=(2000, 1500)):
    """Same convention as test_wheel_zoom_repaint.py's helper of the same
    name, plus an actual show()/processEvents(): that file only ever reads
    v.transform().m11() and paintEvent counts, which don't depend on the
    viewport's real settled geometry -- centering assertions here do, and
    without show() the viewport can stay at its pre-resize default size
    under the offscreen platform, throwing off mapFromScene() math."""
    v = viewer_factory()
    v.resize(w, h)
    v.show()
    qimg = QtGui.QImage(img[0], img[1], QtGui.QImage.Format_RGB32)
    qimg.fill(0x204060)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    QtWidgets.QApplication.processEvents()
    return v


def _screen_px_per_native_px(v, native_hw):
    """What smart_fit_to_scene_rect's own clamp measures against."""
    cached_pw, _cached_ph = v._cached_pixmap_size
    _native_h, native_w = native_hw
    return v.transform().m11() * (cached_pw / float(native_w))


# ---------------------------------------------------------------------------
# ImageViewer.smart_zoom_to_scene_rect -- PAN ONLY, never changes zoom
# ---------------------------------------------------------------------------
def test_smart_zoom_never_changes_scale(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(20000, 15000))
    before = v.transform().m11()

    v.smart_zoom_to_scene_rect(QtCore.QRectF(9000, 7000, 5000, 4000))  # a real, large rect too

    assert v.transform().m11() == before, (
        "smart_zoom_to_scene_rect must never change the current zoom level")


def test_smart_zoom_centers_on_the_rect(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(20000, 15000))
    rect = QtCore.QRectF(9000, 7000, 40, 40)

    v.smart_zoom_to_scene_rect(rect)

    got_center = v.mapFromScene(rect.center())
    vp_center = v.viewport().rect().center()
    assert abs(got_center.x() - vp_center.x()) <= 3
    assert abs(got_center.y() - vp_center.y()) <= 3


def test_smart_zoom_handles_a_zero_area_point_without_change_in_scale(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(3001, 3131))
    before = v.transform().m11()

    v.smart_zoom_to_scene_rect(QtCore.QRectF(1500, 1565, 0, 0))  # a single point

    assert v.transform().m11() == before, (
        "a point must pan at the CURRENT zoom, not force a computed zoom-in")


def test_smart_zoom_rect_none_is_a_noop(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(500, 400))
    before = v.transform().m11()
    v.smart_zoom_to_scene_rect(None)  # must not raise
    assert v.transform().m11() == before


# ---------------------------------------------------------------------------
# ImageViewer.smart_fit_to_scene_rect -- actually zooms, used only for
# real-extent targets
# ---------------------------------------------------------------------------
def test_smart_fit_small_polygon_on_huge_image_zooms_in_and_centers(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(20000, 15000))
    rect = QtCore.QRectF(9000, 7000, 40, 40)

    v.smart_fit_to_scene_rect(rect, native_hw=(15000, 20000))

    assert v.transform().m11() > 1.0, "a tiny rect on a huge image should zoom IN"
    got_center = v.mapFromScene(rect.center())
    vp_center = v.viewport().rect().center()
    assert abs(got_center.x() - vp_center.x()) <= 3
    assert abs(got_center.y() - vp_center.y()) <= 3


def test_smart_fit_respects_native_cap_on_a_cog_preview(viewer_factory):
    """THE key regression test for the cap itself. img= is the PREVIEW size
    (what's actually cached); native_hw is the REAL, much larger native
    raster size -- the same ~16x gap the codebase's own COG comments
    describe. Checking m11() <= 2.0 directly would be wrong here (that's
    200% of the preview, not native) -- the cap must be computed via the
    native ratio, which is exactly what would have caught the original bug."""
    v = _viewer_with_image(viewer_factory, img=(3001, 3131))
    rect = QtCore.QRectF(1490, 1555, 20, 20)  # small but real extent
    native_hw = (50101, 48031)

    v.smart_fit_to_scene_rect(rect, native_hw, max_native_scale=2.0)

    ratio = _screen_px_per_native_px(v, native_hw)
    assert ratio <= 2.0 + 1e-6, (
        f"zoomed to {ratio:.2f}x NATIVE resolution, past the 200% cap")
    assert v.transform().m11() > 5.0, (
        "sanity check: raw preview-relative scale should be large here, "
        "confirming the clamp had real work to do")


def test_smart_fit_large_polygon_zooms_out_to_fit(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(2000, 1500))
    rect = QtCore.QRectF(0, 0, 1900, 1400)

    v.smart_fit_to_scene_rect(rect, native_hw=(1500, 2000))

    assert v.transform().m11() < 1.0, "a near-whole-image rect should zoom OUT"
    ratio = _screen_px_per_native_px(v, (1500, 2000))
    assert ratio <= 2.0 + 1e-6


def test_smart_fit_small_low_res_image_backward_compat(viewer_factory):
    """The old min_dim=100 rarely engaged on a small image like this one --
    confirm the new relative/floor logic still produces a sane,
    non-degenerate fit, not a regression.

    Deliberately doesn't assert tight centering here: on an image this
    small relative to the viewport, the native-cap floor can legitimately
    exceed the image's own extent, and Qt's centerOn() can't fully center
    a point near the scene's edge when there's nowhere further to scroll --
    that's expected QGraphicsView boundary behavior, not a bug."""
    v = _viewer_with_image(viewer_factory, img=(500, 400))
    rect = QtCore.QRectF(240, 190, 20, 20)

    v.smart_fit_to_scene_rect(rect, native_hw=(400, 500))

    m11 = v.transform().m11()
    assert m11 > 0 and m11 == m11 and m11 not in (float("inf"), float("-inf"))
    ratio = _screen_px_per_native_px(v, (400, 500))
    assert ratio <= 2.0 + 1e-6, f"exceeded the native cap: {ratio:.2f}x"


def test_smart_fit_native_hw_none_skips_clamp_gracefully(viewer_factory):
    """Unsaved/synthetic image with no file -- polygon_basis_hw can't
    resolve a native size, so native_hw is None. The fit must still work
    (padding/settle-back alone), no exception, no clamp attempted."""
    v = _viewer_with_image(viewer_factory, img=(20000, 15000))
    rect = QtCore.QRectF(9000, 7000, 40, 40)

    v.smart_fit_to_scene_rect(rect, native_hw=None)  # must not raise

    m11 = v.transform().m11()
    assert m11 > 0 and m11 == m11


def test_smart_fit_zero_area_rect_gets_nonzero_floor(viewer_factory):
    """Even smart_fit_to_scene_rect itself (not just its point-avoiding
    caller) must not produce a degenerate/NaN transform on a zero-area
    input, with or without native_hw -- the resolution-scaled floor exists
    to guarantee this before fitInView ever runs."""
    for native_hw in [(400, 500), None]:
        v = _viewer_with_image(viewer_factory, img=(500, 400))
        rect = QtCore.QRectF(250, 200, 0, 0)

        v.smart_fit_to_scene_rect(rect, native_hw)

        tr = v.transform()
        for val in (tr.m11(), tr.m22()):
            assert val > 0 and val == val, f"degenerate transform with native_hw={native_hw}: {tr}"


def test_smart_fit_rect_none_is_a_noop(viewer_factory):
    v = _viewer_with_image(viewer_factory, img=(500, 400))
    before = v.transform().m11()
    v.smart_fit_to_scene_rect(None, native_hw=(400, 500))
    assert v.transform().m11() == before


# ---------------------------------------------------------------------------
# The two Qt rect-quirks found and fixed while wiring this up
# ---------------------------------------------------------------------------
def test_qrectf_isvalid_is_false_for_a_true_point():
    """Documents WHY the callers check `r is not None`, not `r.isValid()`:
    a single point's bounding rect is exactly zero-size, and Qt considers
    that invalid. The old code's `if r and r.isValid():` guard would have
    silently skipped every true point group."""
    r = QtCore.QRectF(5.0, 5.0, 0.0, 0.0)
    assert r.isValid() is False


def test_qrectf_united_drops_a_point_operand():
    """Documents the OTHER Qt quirk zoom_to_groups now works around: a
    zero-size rect unioned with anything is silently discarded rather than
    expanding the union to include its position."""
    point = QtCore.QRectF(10.0, 10.0, 0.0, 0.0)
    real = QtCore.QRectF(100.0, 100.0, 20.0, 20.0)
    assert point.united(real) == real, (
        "if this ever changes, zoom_to_groups's manual min/max union "
        "workaround may no longer be necessary")


# ---------------------------------------------------------------------------
# PolygonManager.zoom_to_groups / _select_group_in_viewers, end to end:
# the point-vs-polygon dispatch, and multi-viewer fan-out
# ---------------------------------------------------------------------------
class _ZoomOwner:
    """Minimal ProjectTab stand-in: real polygon payloads, a fixed
    polygon_basis_hw, no actual file I/O."""

    def __init__(self, all_polygons, native_hw=(2000, 3000)):
        self.all_polygons = all_polygons
        self._native_hw = native_hw

    def _poly_index_lookup(self, filepath):
        return []

    def polygon_basis_hw(self, viewer, image_data=None):
        return self._native_hw


def _make_viewer_widget(viewer_factory, filepath, img=(3000, 2000)):
    v = _viewer_with_image(viewer_factory, img=img)
    idata = type("D", (), {"filepath": filepath, "ax_config": {}, "raw_shape": None})()
    return {"viewer": v, "image_data": idata}, v


def test_select_group_in_viewers_is_always_pan_only_even_for_a_real_polygon(viewer_factory, monkeypatch):
    """THE core of the product requirement: a plain list click must NEVER
    change zoom, even when the clicked group is a real polygon with
    substantial extent -- only zoom_to_groups (Zoom to Polygon) does that,
    and only for real polygons."""
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(100, 100), (2900, 100), (2900, 1900), (100, 1900)],
             "image_ref_size": {"w": 3000, "h": 2000}, "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp: entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp)
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)
    pm.list_widget = None  # not touched by center=False path below

    before = v.transform().m11()
    pm._select_group_in_viewers("g0", additive=False, center=True)

    assert v.transform().m11() == before, (
        "clicking a group in the list changed zoom -- it must only pan")


def test_zoom_to_groups_point_only_selection_is_pan_only(viewer_factory, monkeypatch):
    """"Zoom polygon to points should assume the same behavior as click on
    the list" -- a point-only (or all-points) selection through
    zoom_to_groups must NOT change zoom either."""
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(1500, 1000)], "image_ref_size": {"w": 3000, "h": 2000},
             "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp: entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp)
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    before = v.transform().m11()
    pm.zoom_to_groups(["g0"])

    assert v.transform().m11() == before, (
        "zoom_to_groups changed zoom for a point-only selection -- points "
        "have no natural fit scale, this must be pan-only like a list click")


def test_zoom_to_groups_real_polygon_actually_zooms(viewer_factory, monkeypatch):
    """The flip side: "Zoom to Polygon" on a REAL polygon (not a point)
    must actually change zoom -- that's the whole point of the feature."""
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(100, 100), (200, 100), (200, 200), (100, 200)],
             "image_ref_size": {"w": 3000, "h": 2000}, "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp: entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp)
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    before = v.transform().m11()
    pm.zoom_to_groups(["g0"])

    assert v.transform().m11() != before, (
        "zoom_to_groups on a real polygon must actually zoom, not just pan")


def test_multi_viewer_fanout_pans_every_viewer(viewer_factory, monkeypatch):
    """Both viewers -- not just the first -- must be panned to center on the
    group's polygon when it's clicked in the list, with neither changing
    zoom. Points are placed near the CENTER of the 3000x2000 image
    (deliberately not near a corner): a target too close to the padded
    scene's edge can hit the scrollbar's own range limit, so centerOn()
    can't fully reach it regardless of correctness -- same expected
    QGraphicsView boundary behavior noted in
    test_smart_fit_small_low_res_image_backward_compat above, not
    something this test should be confounded by."""
    fp_a, fp_b = "C:/proj/imgA.tif", "C:/proj/imgB.tif"
    entry = {"points": [(1400, 900), (1600, 900), (1600, 1100), (1400, 1100)],
             "image_ref_size": {"w": 3000, "h": 2000}, "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp_a: entry, fp_b: entry}})

    vw_a, v_a = _make_viewer_widget(viewer_factory, fp_a)
    vw_b, v_b = _make_viewer_widget(viewer_factory, fp_b)
    owner.viewer_widgets = [vw_a, vw_b]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    # Derive the expected pan target the same way _select_group_in_viewers
    # itself does (_get_polygon_scene_rect's own scene-space mapping,
    # including the viewer's padded sceneRect) -- rather than assuming the
    # raw image-pixel center maps 1:1 to scene coordinates, which it does
    # not (the scene rect is offset/padded 20% around the image).
    target_a = pm._get_polygon_scene_rect(vw_a, v_a, "g0").center()
    target_b = pm._get_polygon_scene_rect(vw_b, v_b, "g0").center()

    m11_a_before, m11_b_before = v_a.transform().m11(), v_b.transform().m11()

    pm._select_group_in_viewers("g0", additive=False, center=True)

    # Loose tolerance deliberately: scrollbar/viewport rounding near a
    # boundary can land centerOn a little off pixel-perfect center (same
    # category of QGraphicsView behavior noted above) -- what actually
    # matters here is "did every viewer pan to roughly the target", not
    # sub-pixel precision (that's covered exactly by
    # test_smart_zoom_centers_on_the_rect, which has no boundary confound).
    for v, target, m11_before, label in [(v_a, target_a, m11_a_before, "A"),
                                          (v_b, target_b, m11_b_before, "B")]:
        assert v.transform().m11() == m11_before, f"viewer {label} changed zoom -- must be pan-only"
        got = v.mapFromScene(target)
        vp_center = v.viewport().rect().center()
        assert abs(got.x() - vp_center.x()) <= 60 and abs(got.y() - vp_center.y()) <= 60, (
            f"viewer {label} was not panned anywhere near the group: got={got}, center={vp_center}")


def test_zoom_to_groups_multi_group_union_includes_a_point(viewer_factory, monkeypatch):
    """A polygon group and a POINT group selected together must produce a
    combined rect covering both (routed to smart_fit_to_scene_rect, since
    the union has real extent from the polygon) -- pins the QRectF.united()
    workaround: without it, the point's position would be silently dropped
    from the union and the view would fit the polygon alone."""
    fp = "C:/proj/imgA.tif"
    poly_entry = {"points": [(2900, 1900), (2950, 1900), (2950, 1950), (2900, 1950)],
                  "image_ref_size": {"w": 3000, "h": 2000}, "coord_space": "image"}
    point_entry = {"points": [(50, 50)],
                   "image_ref_size": {"w": 3000, "h": 2000}, "coord_space": "image"}
    owner = _ZoomOwner({"poly_group": {fp: poly_entry}, "point_group": {fp: point_entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp)
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    r_poly = pm._get_polygon_scene_rect(vw, v, "poly_group")
    r_point = pm._get_polygon_scene_rect(vw, v, "point_group")
    assert r_poly is not None and r_point is not None

    pm.zoom_to_groups(["poly_group", "point_group"])

    # The combined view must be wide enough to have actually included the
    # far-apart point, not just the small polygon -- checked indirectly via
    # the resulting scale: a rect spanning both corners of the 3000x2000
    # image is close to the whole image, so the fit scale must be small
    # (zoomed OUT), not the tight zoomed-in scale a poly_group-only fit
    # would produce.
    only_poly_scale_upper_bound = 5.0  # a ~50x50 rect on a 3000px image fits at a large scale
    assert v.transform().m11() < only_poly_scale_upper_bound, (
        "the view zoomed in as if only the small polygon was framed -- "
        "the point's position was likely dropped from the union")


# ---------------------------------------------------------------------------
# coord_space handling in _get_polygon_scene_rect
# ---------------------------------------------------------------------------
def test_coord_space_scene_routes_through_map_points_scene_to_image(viewer_factory, monkeypatch):
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(10, 10), (20, 20)], "image_ref_size": {"w": 100, "h": 100},
             "coord_space": "scene"}
    owner = _ZoomOwner({"g0": {fp: entry}})
    calls = []

    def fake_mapper(filepath, points, size_hint, polygon_data=None, viewer=None):
        calls.append((filepath, points, size_hint))
        return [(30, 30), (40, 40)]  # distinct from the input, so we can tell it was used

    owner._map_points_scene_to_image = fake_mapper

    vw, v = _make_viewer_widget(viewer_factory, fp, img=(100, 100))
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    rect = pm._get_polygon_scene_rect(vw, v, "g0")

    assert calls, "coord_space == 'scene' must call _map_points_scene_to_image"
    assert calls[0][0] == fp
    assert calls[0][1] == [(10, 10), (20, 20)]
    assert rect is not None


def test_coord_space_image_default_does_not_call_mapper(viewer_factory, monkeypatch):
    """The common/default case (coord_space absent, or explicitly 'image')
    must NOT call the scene->image mapper -- preserves today's behavior for
    every live polygon-creation path, which already writes 'image'."""
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(10, 10), (20, 20)], "image_ref_size": {"w": 100, "h": 100}}
    owner = _ZoomOwner({"g0": {fp: entry}})
    calls = []
    owner._map_points_scene_to_image = lambda *a, **k: calls.append(1)

    vw, v = _make_viewer_widget(viewer_factory, fp, img=(100, 100))
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    rect = pm._get_polygon_scene_rect(vw, v, "g0")

    assert not calls, "coord_space defaulting to 'image' must not call the scene->image mapper"
    assert rect is not None


# ---------------------------------------------------------------------------
# THE TWO REAL PRODUCTION BUGS -- reproduced directly against C:\RGB's
# actual project.json / polygon sidecar / 4000x3000 source image (see the
# module docstring for the full writeup and exact numbers).
# ---------------------------------------------------------------------------
def test_get_polygon_scene_rect_maps_to_the_pixmap_not_the_padded_scene_rect(
        viewer_factory, monkeypatch):
    """THE core regression test: with coord_space='image', an empty
    ax_config, and image_ref_size matching the displayed pixmap exactly (a
    plain, non-.ax-edited image -- true for both C:\\RGB and
    C:\\New Folder196), the mapping must be the identity: a point at raw
    image pixel (x, y) lands at scene (x, y), because the pixmap item
    always sits at scene position (0,0) (image_viewer.py's set_image calls
    _image.setPos(0, 0)). It must NOT be rescaled against
    viewer.sceneRect(), which is the image PLUS 20% padding on every side
    (a 4000x3000 image gets a 5600x4200 sceneRect) -- that was the actual
    bug: every point landed ~1.4x further from the origin than its real
    position, which for a polygon anywhere but dead-center pushes "zoom to
    polygon" outside the image entirely.

    Uses the EXACT real points/image_ref_size from C:\\RGB's own
    dsaz_..._polygons.json sidecar."""
    fp = "C:/Multispect_analyser/test/Jpeg/20221212_103009_759_8b.JPG"
    # Exact values from the real sidecar file.
    real_points = [
        [2068.0015851628086, 2445.1213383186614],
        [1960.2530842611363, 2455.8961884088285],
    ]
    entry = {"points": real_points, "image_ref_size": {"w": 4000, "h": 3000},
             "coord_space": "image"}
    owner = _ZoomOwner({"dsaz": {fp: entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp, img=(4000, 3000))
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    rect = pm._get_polygon_scene_rect(vw, v, "dsaz")

    assert rect is not None
    # Identity mapping: the rect's corners must match the raw points
    # directly, not scaled by the padded (5600x4200) sceneRect.
    xs = [p[0] for p in real_points]
    ys = [p[1] for p in real_points]
    assert abs(rect.left() - min(xs)) < 1e-6
    assert abs(rect.top() - min(ys)) < 1e-6
    assert abs(rect.right() - max(xs)) < 1e-6
    assert abs(rect.bottom() - max(ys)) < 1e-6
    # And therefore inside the actual 4000x3000 image -- the direct
    # assertion the bug report's "outside the image boundaries" was about.
    assert 0 <= rect.left() and rect.right() <= 4000
    assert 0 <= rect.top() and rect.bottom() <= 3000


def test_get_polygon_scene_rect_does_not_scale_by_the_padded_scene_rect(viewer_factory, monkeypatch):
    """Narrower, more mechanical pin of the same bug: the OLD formula's
    signature was `nsx * sr.width()` where sr.width() = image_w * 1.4
    (padding). Directly assert the returned rect is NOT anywhere near what
    that padded multiplication would produce."""
    fp = "C:/proj/imgA.tif"
    entry = {"points": [(2068.0, 2445.12)], "image_ref_size": {"w": 4000, "h": 3000},
             "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp: entry}})

    vw, v = _make_viewer_widget(viewer_factory, fp, img=(4000, 3000))
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    rect = pm._get_polygon_scene_rect(vw, v, "g0")

    buggy_x = (2068.0 / 4000.0) * v.sceneRect().width()   # what the old code produced
    assert abs(rect.center().x() - buggy_x) > 100, (
        "rect was computed against the padded sceneRect, not the pixmap")


def test_smart_fit_floor_is_small_and_content_relative_without_native_hw(viewer_factory):
    """Simplification, not an independent second bug: an earlier version of
    this floor tried to pre-empt the native-resolution cap itself, sized
    off the viewport (vp_w*cached_pw/(max_native_scale*native_w)). Revert-
    verification showed that formula change alone does NOT alter any
    observable output in this test suite: whenever native_hw is supplied
    and a small polygon needs real zoom-in, the exact post-fitInView clamp
    (step 3) corrects the final scale to precisely max_native_scale*
    settle_back regardless of what the floor produced first (floor-
    expansion is always symmetric around the original center, so it never
    moves centerOn's target either) -- and when native_hw is None, both the
    old and new formulas are the literal same expression. So this is pinned
    as a code-quality simplification (replacing a formula that LOOKS
    resolution-aware but, per the math above, never actually needed to be)
    rather than a fix with its own independently-observable symptom: just
    confirm the floor stays a small, sane, image-proportional size, with no
    native_hw to invoke the (separately and independently verified) clamp."""
    v = _viewer_with_image(viewer_factory, img=(4000, 3000))
    rect = QtCore.QRectF(1960.25, 2411.26, 107.75, 82.35)  # the real C:\RGB polygon extent

    v.smart_fit_to_scene_rect(rect, native_hw=None)

    vp = v.viewport().rect()
    tl = v.mapToScene(vp.topLeft())
    br = v.mapToScene(vp.bottomRight())
    view_w, view_h = br.x() - tl.x(), br.y() - tl.y()

    # The floor is 2% of the image's longest side (80 for this 4000px
    # image) -- confirm the final view is in that ballpark, not hundreds of
    # scene units disconnected from the image/polygon size.
    assert view_w < 200 and view_h < 200

    # And still inside the image -- the polygon (1960-2068, 2411-2493) is
    # nowhere near an edge of a 4000x3000 image, so a correct, proportionate
    # fit must never approach the boundary here.
    assert 0 <= tl.x() and br.x() <= 4000
    assert 0 <= tl.y() and br.y() <= 3000


def test_zoom_to_groups_keeps_an_off_center_real_scale_polygon_inside_the_image(
        viewer_factory, monkeypatch):
    """End-to-end reproduction of the reported symptom, through the actual
    PolygonManager.zoom_to_groups entry point: a real-scale (4000x3000, no
    COG gap) polygon positioned in the lower-right area of the image (not
    dead-center, where the old bugs could still coincidentally land inside
    bounds) must result in a final view that overlaps the image, not one
    that lands entirely outside it."""
    fp = "C:/proj/imgA.tif"
    # Positioned like the real C:\RGB polygon: well off-center (x~52%,
    # y~82% of the image), which is exactly where the padded-sceneRect bug
    # pushed the view below the image entirely.
    entry = {"points": [[2068.0, 2445.12], [1960.25, 2493.6], [1979.49, 2411.26]],
             "image_ref_size": {"w": 4000, "h": 3000}, "coord_space": "image"}
    owner = _ZoomOwner({"g0": {fp: entry}}, native_hw=(3000, 4000))

    vw, v = _make_viewer_widget(viewer_factory, fp, img=(4000, 3000))
    owner.viewer_widgets = [vw]

    pm = PolygonManager.__new__(PolygonManager)
    monkeypatch.setattr(type(pm), "parent", lambda self: owner, raising=False)

    pm.zoom_to_groups(["g0"])

    vp = v.viewport().rect()
    tl = v.mapToScene(vp.topLeft())
    br = v.mapToScene(vp.bottomRight())
    overlaps_image = not (br.x() < 0 or tl.x() > 4000 or br.y() < 0 or tl.y() > 3000)
    assert overlaps_image, (
        f"final view (({tl.x():.0f},{tl.y():.0f}) to ({br.x():.0f},{br.y():.0f})) "
        f"does not overlap the 4000x3000 image at all -- reproduces the "
        f"reported 'zooming out of the image boundaries' bug")
