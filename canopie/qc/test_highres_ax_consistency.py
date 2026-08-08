"""
The zoom overlay must never show pixels the displayed image does not have.

CONTEXT. For an oversized raster the viewer shows a DECIMATED preview, and
zooming asks for a sharper tile of the visible region. Those two images come
from completely different pipelines:

    preview : reader -> apply_aux_modifications  (crop, rotate, hist match,
                                                  resize, band expression,
                                                  classification) -> stretch
    tile    : reader.read_window -> stretch

The tile skips `apply_aux_modifications` entirely. So the moment the `.ax`
carries an op that changes geometry or pixel VALUES, the sharpened region
disagrees with the image underneath it:

  * band math / rescaling "don't translate to the viewer" -- the sharpened
    patch is RAW data while the surrounding preview is the computed index;
  * with a CROP the coordinate mapping is wrong outright, because
    `display_scale` and `full_shape` describe the UNCROPPED raster (they are
    set in `_imagedata_or_fallback`, before the crop is applied). Measured on
    the real 19499x17481 COG with a crop: the tile landed at scene
    (-529.6, -428.7) -- outside the 500x438 cropped preview -- and was
    entirely black. That is the reported "cropping then zooming gives a black
    viewer".

THE FIX: give the tile its own windowed replay of the `.ax`, and its own
windowability predicate, because the viewer is not asking export's question:

    export asks:   can I compute EXACT STATISTICS from a window?
    the tile asks: can I draw a FAITHFUL PICTURE of a window?

Answering the second with `_ax_is_windowable` (the first) is what stranded
every edited image on the coarse pyramid level -- reported as "rescaling is
performed in the lower pyramid so I see very broad pixels" and "rotate,
rescaling and band math all return the lower resolution".

Three of the four ops survive windowing exactly:

  resize  only changes the SCALE between scene and raster coordinates, so
          taking the ratio against the displayed shape absorbs it and the tile
          needs no resize-specific code at all
  rotate  is rigid: the rotated sub-window equals the sub-window of the
          rotated image, so un-rotate the request and rot90 the result
  band    math is pointwise: evaluating on a window equals the window of
          evaluating on everything
  crop    is an offset, undone via `_ax_crop_in_full_pixels`

What still cannot be tiled is anything needing GLOBAL information the window
does not contain -- hist_match (per-band gain/offset from whole-image stats),
registration, classification, mask_polygon, appended_bands. For those the
overlay stays off and the viewer stays at preview resolution: softer, but
correct. A sharp tile of the wrong pixels is far worse than a blurry one of
the right pixels.
"""
import json

import numpy as np
import pytest

from ..project_tab import ProjectTab
from ..raster_reader import open_reader, probe
from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer, pytest.mark.contract]


# .ax configurations, and whether each predicate accepts them.
#                                                        (tileable, windowable)
AX_CASES = {
    "plain":            ({}, True, True),
    "crop":             ({"crop_enabled": True,
                          "crop_rect": {"x": 8, "y": 8, "width": 40, "height": 40},
                          "crop_rect_ref_size": {"w": 96, "h": 96}}, True, True),
    "nodata":           ({"nodata_enabled": True, "nodata_values": [0]}, True, True),
    # These three are the reported symptom: drawable per-window, but NOT
    # exactly computable per-window, so the two predicates must disagree.
    "band_expression":  ({"band_enabled": True, "band_expression": "b1+b2"}, True, False),
    "resize":           ({"resize_enabled": True, "resize": {"width": 50, "height": 50}}, True, False),
    "rotate":           ({"rotate_enabled": True, "rotate": 90}, True, False),
    # Needs whole-image statistics -- untileable by nature.
    "hist_match":       ({"hist_enabled": True,
                          "hist_match": {"mode": "meanstd", "bands": 3,
                                         "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3}}, False, False),
}


@pytest.mark.parametrize("case", sorted(AX_CASES))
def test_tileability_and_windowability_are_separate_questions(synthetic_project, case):
    """THE gate, and the distinction the whole fix rests on.

    Collapsing these two predicates into one is what cost the resolution: the
    export predicate correctly refuses resize/rotate/band math (they change
    pixel VALUES or GEOMETRY, so exact statistics need the whole image), but
    the viewer only needs to draw the same picture, which all three allow.
    """
    ax, want_tileable, want_windowable = AX_CASES[case]
    assert synthetic_project._ax_is_tileable(ax) is want_tileable, (
        f"{case}: _ax_is_tileable said {synthetic_project._ax_is_tileable(ax)}, "
        f"expected {want_tileable}")
    assert synthetic_project._ax_is_windowable(ax) is want_windowable, (
        f"{case}: _ax_is_windowable said "
        f"{synthetic_project._ax_is_windowable(ax)}, expected {want_windowable} "
        "-- this predicate gates the lazy EXPORT path; changing it changes "
        "scientific output, not just the picture")


def test_hist_match_still_disables_the_overlay(synthetic_project):
    """The one op that genuinely cannot be tiled. Its per-band gain/offset come
    from whole-image statistics, so a window cannot reproduce them; painting a
    raw tile over a matched preview would be visibly wrong."""
    ax = {"hist_enabled": True,
          "hist_match": {"mode": "meanstd", "bands": 3,
                         "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3}}
    assert not synthetic_project._ax_is_tileable(ax)


@pytest.mark.parametrize("key,flag", [
    ("band_expression", "band_enabled"),
    ("resize", "resize_enabled"),
    ("rotate", "rotate_enabled"),
    ("hist_match", "hist_enabled"),
])
def test_a_disabled_op_never_costs_resolution(synthetic_project, key, flag):
    """An op switched OFF cannot alter pixels, so it must not gate anything.
    (A leftover config block with its flag cleared is the normal state once a
    user has experimented in the editor and backed the change out.)"""
    ax = {flag: False, key: {"mode": "meanstd", "bands": 1,
                             "ref_stats": [{"mean": 1.0, "std": 1.0}]}
          if key == "hist_match" else "b1+b2"}
    assert synthetic_project._ax_is_tileable(ax), (
        f"a disabled {key} still disables the zoom overlay")


# ---------------------------------------------------------------------------
# Rotation: the coordinate inverse must be pixel-exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("rect", [
    (0, 0, None, None),          # whole frame
    (3, 5, 11, 20),              # interior
    (7, 0, 9, 3),                # touching an edge
])
def test_unrotate_window_inverts_rot90_exactly(k, rect):
    """The tile reads an UN-rotated window and then rotates the result. That is
    only valid if the un-rotated rect is exactly the pre-image of the rect the
    viewer asked for -- so compare pixel VALUES, not just shapes.

    Uses a non-square, non-round frame (37x53): with H == W every off-by-one in
    a width/height swap cancels out and a wrong formula still passes.
    """
    H, W = 37, 53
    m = np.arange(H * W).reshape(H, W)
    rot = np.rot90(m, k=k)
    rh, rw = rot.shape

    ri0, rj0, ri1, rj1 = rect
    if ri1 is None:
        ri1, rj1 = rh, rw
    if ri1 > rh or rj1 > rw:
        pytest.skip("rect does not fit this orientation")

    want = rot[ri0:ri1, rj0:rj1]
    cr0, cc0, cr1, cc1 = ProjectTab._unrotate_window(k, W, H, ri0, rj0, ri1, rj1)
    got = np.rot90(m[cr0:cr1, cc0:cc1], k=k)

    assert got.shape == want.shape, (
        f"k={k}: un-rotated window has shape {got.shape}, expected {want.shape}")
    assert np.array_equal(got, want), (
        f"k={k}: the tile would show different PIXELS than the region the "
        f"viewer asked for (src rect {(cr0, cc0, cr1, cc1)})")


def test_rotate_degrees_map_to_the_same_turns_the_editor_uses():
    """`_do_rotate` uses cv2-style clockwise degrees (90 -> np.rot90 k=3). The
    tile must use the identical mapping or the sharpened patch is rotated the
    wrong way relative to the preview."""
    assert ProjectTab._ROT_K == {0: 0, 90: 3, 180: 2, 270: 1}


# ---------------------------------------------------------------------------
# crop + rotate: WHICH ORDER, and therefore which frame the crop rect is in
# ---------------------------------------------------------------------------
RAW_W, RAW_H = 400, 300          # the dims the pipeline actually sees


@pytest.mark.parametrize("ref,rot,expected,why", [
    ({"w": RAW_W, "h": RAW_H}, 90,  False, "ref matches raw dims -> crop drawn first"),
    ({"w": RAW_H, "h": RAW_W}, 90,  True,  "ref matches rotated dims -> rotate first"),
    ({"w": RAW_W, "h": RAW_H}, 180, False, "180 does not swap; raw ref -> crop first"),
    ({"w": 999, "h": 999},     90,  True,  "ambiguous -> pipeline's default"),
    (None,                     90,  True,  "no ref -> pipeline's default"),
])
def test_rotate_ordering_matches_the_pipeline(ref, rot, expected, why):
    """`apply_aux_modifications` decides crop-then-rotate vs rotate-then-crop
    from `crop_rect_ref_size`, and the crop rect is expressed in whichever
    frame it was drawn in. The tile MUST take the same decision -- taking the
    other one puts the crop origin in the wrong coordinate system.

    This mirrors the pipeline including its DEFAULT, deliberately: the tile has
    to reproduce what the preview actually did, not what would be most sensible.
    """
    ax = {"crop_enabled": True, "rotate_enabled": True, "rotate": rot,
          "crop_rect": {"x": 10, "y": 20, "width": 100, "height": 80}}
    if ref is not None:
        ax["crop_rect_ref_size"] = ref
    assert ProjectTab._ax_do_rotate_first(ax, RAW_W, RAW_H) is expected, why


def test_rotate_ordering_is_asked_about_the_array_the_pipeline_saw():
    """The comparison is against the DECIMATED preview's dims, not the full
    raster's. Asking with the full dims answers a different question."""
    ax = {"crop_enabled": True, "rotate_enabled": True, "rotate": 90,
          "crop_rect": {"x": 1, "y": 1, "width": 10, "height": 10},
          "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}}
    assert ProjectTab._ax_do_rotate_first(ax, RAW_W, RAW_H) is False
    # Same .ax, asked about a differently-sized array -> different answer.
    assert ProjectTab._ax_do_rotate_first(ax, RAW_W * 8, RAW_H * 8) is True


def test_no_crop_means_ordering_cannot_matter():
    """With no crop there is nothing to order, so rotation alone must never be
    treated as ambiguous."""
    ax = {"rotate_enabled": True, "rotate": 270}
    assert ProjectTab._ax_do_rotate_first(ax, RAW_W, RAW_H) is True


# ---------------------------------------------------------------------------
# The full scene -> raster window map
# ---------------------------------------------------------------------------
class _Rect:
    """Minimal QRectF stand-in (avoids constructing Qt objects for pure math)."""
    def __init__(self, l, t, r, b):
        self._v = (l, t, r, b)

    def left(self):   return self._v[0]
    def top(self):    return self._v[1]
    def right(self):  return self._v[2]
    def bottom(self): return self._v[3]


class _Prof:
    def __init__(self, w, h):
        self.width, self.height = w, h


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_window_stays_inside_the_raster_for_every_rotation(synthetic_project, rot):
    """A rotation must never push the request outside the raster.

    The bug this catches: `_unrotate_window` needs the dims of the array the
    rotation was applied TO, not its rotated extent. Passing the rotated dims
    yields a plausible-looking but TRANSPOSED window -- on the real 19499x17481
    COG that drove tile-vs-preview correlation to 0.01 for 90 and 270, while
    180 (where the dims do not swap, so the mistake cancels) stayed correct.
    Hence all four angles here, not just one.
    """
    prof = _Prof(400, 300)
    ax = {"rotate_enabled": True, "rotate": rot}
    disp_h, disp_w = (200, 150) if rot in (90, 270) else (150, 200)

    plan = synthetic_project._tile_window_for_scene(
        ax, prof, (disp_h, disp_w, 3),
        _Rect(disp_w * 0.25, disp_h * 0.25, disp_w * 0.75, disp_h * 0.75),
        raw_shape=(prof.height, prof.width, 3))
    assert plan is not None, f"rot={rot}: no window produced"
    (x0, y0, x1, y1), rot_k, _ = plan

    assert 0 <= x0 < x1 <= prof.width, f"rot={rot}: x window {(x0, x1)} outside raster"
    assert 0 <= y0 < y1 <= prof.height, f"rot={rot}: y window {(y0, y1)} outside raster"
    assert rot_k == ProjectTab._ROT_K[rot]

    # A centred request must land centred, not against an edge -- a transposed
    # window still fits inside a non-square raster, so bounds alone are weak.
    assert x0 > 0 and y0 > 0 and x1 < prof.width and y1 < prof.height, (
        f"rot={rot}: an interior request reached the raster edge {(x0, y0, x1, y1)}")


@pytest.mark.parametrize("frame", ["crop_then_rotate", "rotate_then_crop"])
def test_window_respects_the_crop_in_whichever_frame_it_was_drawn(
        synthetic_project, frame):
    """Both orderings must keep the read inside the cropped extent."""
    prof = _Prof(400, 300)
    cx, cy, cw, ch = 40, 30, 200, 150
    ax = {"crop_enabled": True, "rotate_enabled": True, "rotate": 90,
          "crop_rect": {"x": cx, "y": cy, "width": cw, "height": ch}}
    if frame == "crop_then_rotate":
        ax["crop_rect_ref_size"] = {"w": prof.width, "h": prof.height}
        disp_h, disp_w = cw, ch            # crop, then rotated -> axes swap
    else:
        ax["crop_rect_ref_size"] = {"w": prof.height, "h": prof.width}
        disp_h, disp_w = ch, cw            # crop drawn in the rotated frame

    plan = synthetic_project._tile_window_for_scene(
        ax, prof, (disp_h, disp_w, 3),
        _Rect(disp_w * 0.2, disp_h * 0.2, disp_w * 0.8, disp_h * 0.8),
        raw_shape=(prof.height, prof.width, 3))
    assert plan is not None, f"{frame}: no window produced"
    (x0, y0, x1, y1), _rot_k, _ = plan

    assert 0 <= x0 < x1 <= prof.width and 0 <= y0 < y1 <= prof.height, (
        f"{frame}: window {(x0, y0, x1, y1)} left the raster")
    # The read must cover strictly less than the whole raster: a wrong frame
    # typically produces a window spanning the full extent.
    assert (x1 - x0) < prof.width and (y1 - y0) < prof.height, (
        f"{frame}: window {(x0, y0, x1, y1)} is not constrained by the crop")


def test_upscaling_resize_does_not_break_the_map(synthetic_project):
    """A resize ABOVE 100% makes the preview hold more pixels than the source
    region. The map must still land inside the raster (the tile is simply not
    sharper than the preview in that case, which is correct)."""
    prof = _Prof(400, 300)
    ax = {"resize_enabled": True, "resize": {"width": 200, "height": 200}}
    disp_h, disp_w = 600, 800                      # 200% of the raster

    plan = synthetic_project._tile_window_for_scene(
        ax, prof, (disp_h, disp_w, 3),
        _Rect(disp_w * 0.4, disp_h * 0.4, disp_w * 0.6, disp_h * 0.6),
        raw_shape=(prof.height, prof.width, 3))
    assert plan is not None
    (x0, y0, x1, y1), _, _ = plan
    assert 0 <= x0 < x1 <= prof.width and 0 <= y0 < y1 <= prof.height
    # 20% of the display must map to ~20% of the raster regardless of the scale.
    assert abs((x1 - x0) - prof.width * 0.2) <= 2


# ---------------------------------------------------------------------------
# Band math: the tile must compute the same index the preview shows
# ---------------------------------------------------------------------------
def test_tile_appends_the_band_expression(synthetic_project):
    """Pointwise means windowable. The expression band must land in the same
    LAST position the preview puts it in, so the band bar's selection keeps
    addressing the same channel in both."""
    rng = np.random.default_rng(0)
    win = (rng.random((8, 9, 4), dtype=np.float32) + 0.5)
    ax = {"band_enabled": True, "band_expression": "b1+b2"}

    out = synthetic_project._tile_apply_band_expression(win, ax)
    assert out.shape[:2] == win.shape[:2]
    assert out.shape[2] == win.shape[2] + 1, (
        "the expression band was not appended -- the viewer's last-band "
        "selection would address a raw band instead of the index")
    np.testing.assert_allclose(out[..., -1], win[..., 0] + win[..., 1], rtol=1e-5)
    # The original bands must survive: the band bar can still switch back.
    np.testing.assert_allclose(out[..., :4], win, rtol=1e-6)


def test_tile_band_math_equals_windowing_the_whole_image_result(synthetic_project):
    """THE property that makes this legal, asserted directly: evaluating on a
    window == the window of evaluating on everything."""
    rng = np.random.default_rng(1)
    full = (rng.random((20, 24, 3), dtype=np.float32) + 0.5)
    ax = {"band_enabled": True, "channel_order": "rgb",
          "band_expression": "(b2-b1)/(b2+b1)"}

    whole = synthetic_project._tile_apply_band_expression(full, ax)[..., -1]
    win = synthetic_project._tile_apply_band_expression(full[4:12, 6:18], ax)[..., -1]

    np.testing.assert_allclose(win, whole[4:12, 6:18], rtol=1e-6, atol=1e-7)


def test_tile_band_math_honours_the_enable_flag(synthetic_project):
    """A disabled expression must append nothing, or the tile grows a channel
    the preview does not have and the band indices diverge."""
    win = np.ones((4, 4, 3), dtype=np.float32)
    out = synthetic_project._tile_apply_band_expression(
        win, {"band_enabled": False, "band_expression": "b1+b2"})
    assert out.shape == win.shape


def test_tile_band_math_uses_the_same_bgr_rule_as_the_editor(synthetic_project):
    """`_do_band_expr` treats an exactly-3-channel image as cv2-style BGR unless
    the .ax overrides it, so `b1` is RED. The tile must agree or a 3-band index
    comes out with its operands swapped."""
    win = np.zeros((2, 2, 3), dtype=np.float32)
    win[..., 0] = 1.0        # cv2 blue plane
    win[..., 2] = 9.0        # cv2 red plane

    default = synthetic_project._tile_apply_band_expression(
        win, {"band_enabled": True, "band_expression": "b1"})
    assert np.allclose(default[..., -1], 9.0), (
        "b1 did not resolve to the RED plane for a 3-channel image")

    forced = synthetic_project._tile_apply_band_expression(
        win, {"band_enabled": True, "band_expression": "b1",
              "channel_order": "rgb"})
    assert np.allclose(forced[..., -1], 1.0), (
        'channel_order "rgb" was ignored')


# ---------------------------------------------------------------------------
# The crop coordinate fix
# ---------------------------------------------------------------------------
def test_cropped_tile_reads_inside_the_crop(synthetic_project, tmp_path, monkeypatch):
    """THE black-window regression, at the coordinate level.

    Captures the window `_request_highres_viewport_region` actually asks the
    reader for, and requires it to fall inside the crop rectangle. Before the
    fix the request was offset by the crop origin and scaled against the
    UNCROPPED preview, so it addressed the wrong part of the raster (and, for a
    crop far from the origin, nothing at all).
    """
    import types

    name = "rgb_16bit_tiled_bip_cog"          # a real tiled COG fixture
    fp = fixture_image_path(name)
    prof = probe(fp)
    reader = open_reader(fp, prof)
    assert reader is not None

    crop_x, crop_y, crop_w, crop_h = 20, 16, 48, 40
    ax = {"crop_enabled": True,
          "crop_rect": {"x": crop_x, "y": crop_y, "width": crop_w, "height": crop_h},
          "crop_rect_ref_size": {"w": prof.width, "h": prof.height}}

    # A preview of exactly the cropped region, decimated 2x -- the shape
    # apply_aux_modifications would have produced.
    disp = np.zeros((crop_h // 2, crop_w // 2, 3), dtype=np.float32)
    image_data = types.SimpleNamespace(
        filepath=fp, image=disp, reader=reader, profile=prof,
        preview_bands=[0, 1, 2],
        full_shape=(prof.height, prof.width, prof.count),
        # Deliberately the UNCROPPED basis, as _imagedata_or_fallback sets it.
        display_scale=prof.width / float(disp.shape[1]),
    )

    seen = {}
    original = type(reader).read_window

    def _spy(self, x0, y0, x1, y1, bands=None, level=0):
        seen["win"] = (x0, y0, x1, y1)
        return original(self, x0, y0, x1, y1, bands=bands, level=level)

    monkeypatch.setattr(type(reader), "read_window", _spy)

    from PyQt5 import QtCore
    # Zoomed onto the middle of the cropped preview.
    rect = QtCore.QRectF(disp.shape[1] * 0.25, disp.shape[0] * 0.25,
                         disp.shape[1] * 0.5, disp.shape[0] * 0.5)

    class _VP:
        def width(self): return 400
        def height(self): return 300

    viewer = types.SimpleNamespace(viewport=lambda: _VP(), image_data=image_data)

    synthetic_project._request_highres_viewport_region(
        viewer, image_data, rect, request_id=None, ax=ax)

    # The fetch runs on an executor; wait for the spy.
    import time
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and "win" not in seen:
        time.sleep(0.02)

    assert "win" in seen, "the reader was never asked for a region"
    x0, y0, x1, y1 = seen["win"]

    assert x0 >= crop_x and y0 >= crop_y, (
        f"requested window ({x0},{y0})-({x1},{y1}) starts BEFORE the crop "
        f"origin ({crop_x},{crop_y}) -- the crop offset is not being applied, "
        "which is what produced an all-black tile at negative scene coords")
    assert x1 <= crop_x + crop_w and y1 <= crop_y + crop_h, (
        f"requested window ({x0},{y0})-({x1},{y1}) extends beyond the crop "
        f"({crop_x},{crop_y},{crop_w}x{crop_h})")
    assert x1 > x0 and y1 > y0, "empty window requested"


def test_request_accepts_a_preloaded_ax(synthetic_project):
    """display_image_group resolves the .ax once and passes it down, so the
    per-zoom request must not re-read it from disk on every wheel tick."""
    import inspect
    sig = inspect.signature(synthetic_project._request_highres_viewport_region)
    assert "ax" in sig.parameters, (
        "_request_highres_viewport_region should accept the already-resolved "
        ".ax rather than re-reading it for every zoom event")
