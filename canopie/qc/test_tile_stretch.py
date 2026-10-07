"""
BUG-007 D2 / D3 (2026-10-06): the zoom tile is stretched like the preview.

  D2  A zoom tile used to be stretched by its OWN percentiles, so zooming in
      changed the colours of the same pixels. The preview's render now keeps
      the bounds it mapped with (viewer._preview_stretch_bounds) and the tile
      maps with exactly those -- one stretch for the whole image.
  D3  A 0/1 band (a boolean band expression) collapsed to black on a tile that
      was all True (lo == hi == 1). A band holding only 0 and 1 now always maps
      0 -> black and 1 -> white, in both stretch engines.
"""
import types

import numpy as np
import pytest
from PyQt5 import QtCore

pytestmark = [pytest.mark.viewer]


def _sp(**kw):
    from ..project_tab import _StretchParams
    base = dict(mode="percentile", low_p=2.0, high_p=98.0, k_sigma=1.0, min_val=None,
                max_val=None, per_channel=True, clip=True, scope="viewer", display_mode="auto")
    base.update(kw)
    return _StretchParams(**base)


@pytest.mark.parametrize("fill", [1.0, 0.0])
@pytest.mark.parametrize("engine", ["C0", "F"])
def test_a_constant_boolean_band_is_white_for_true_black_for_false(synthetic_project, engine, fill):
    pt = synthetic_project
    band = np.full((30, 40), fill, np.float32)
    if engine == "C0":
        out = np.asarray(pt._compute_stretched_preview(band, _sp()))
    else:
        out = np.asarray(pt._apply_viewer_stretch_numpy(band, _sp()))
    want = 255 if fill == 1.0 else 0
    assert np.all(out == want), f"{engine}: an all-{fill:g} 0/1 band rendered {np.unique(out)}"


@pytest.mark.parametrize("engine", ["C0", "F"])
def test_a_mostly_true_boolean_band_keeps_both_colours(synthetic_project, engine):
    pt = synthetic_project
    band = np.ones((50, 50), np.uint8)
    band[0, 0] = 0                      # 0.04% False: beyond the 2/98 percentiles
    f = pt._compute_stretched_preview if engine == "C0" else pt._apply_viewer_stretch_numpy
    out = np.asarray(f(band, _sp()))
    assert out[1, 1] == 255 and out[0, 0] == 0, f"{engine}: True={out[1, 1]} False={out[0, 0]}"


def _viewer_with(preview, sp):
    return types.SimpleNamespace(image_data=types.SimpleNamespace(image=preview, filepath=None,
                                                                  channel_order="rgb"),
                                 stretch_params=sp)


@pytest.mark.parametrize("prefer_last", [False, True], ids=["rgb", "last_band"])
def test_a_tile_cut_from_the_preview_renders_like_that_part_of_the_preview(synthetic_project, prefer_last):
    pt = synthetic_project
    rng = np.random.default_rng(3)
    preview = rng.gamma(2.0, 400.0, size=(80, 100, 4)).astype(np.float32)
    v = _viewer_with(preview, _sp())
    whole = np.asarray(pt._render_with_viewer_stretch(preview, v, prefer_last_band=prefer_last, _as_array=True))
    pb = getattr(v, "_preview_stretch_bounds", None)
    assert pb is not None and pb["img_id"] == id(preview), "the preview's bounds were not kept"
    # the same pixels as the preview get the preview's colours
    same = np.array(preview[20:50, 30:90], copy=True)
    got_same = np.asarray(pt._render_with_viewer_stretch(same, v, prefer_last_band=prefer_last,
                                                         _as_array=True, bounds=pb))
    assert np.array_equal(got_same, whole[20:50, 30:90]), "a tile re-stretched its own pixels"
    # ...which is NOT what a per-tile stretch gives (the test would be vacuous)
    own = np.asarray(pt._render_with_viewer_stretch(same, v, prefer_last_band=prefer_last, _as_array=True))
    assert not np.array_equal(own, whole[20:50, 30:90])


def test_the_tile_request_hands_the_preview_bounds_to_the_render(synthetic_project, monkeypatch):
    """Wiring: _request_highres_viewport_region renders its tile with the
    bounds the preview was drawn with."""
    import time
    from .fixtures_manifest import fixture_image_path
    from ..raster_reader import probe, open_reader
    pt = synthetic_project
    fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    prof = probe(fp)
    reader = open_reader(fp, prof)
    disp = np.zeros((prof.height // 3, prof.width // 3, 3), np.float32)
    image_data = types.SimpleNamespace(
        filepath=fp, image=disp, reader=reader, profile=prof, preview_bands=[0, 1, 2],
        full_shape=(prof.height, prof.width, prof.count), display_scale=3.0)
    got = {}

    def _render(arr, viewer, *a, **k):
        got["bounds"] = k.get("bounds")
        return np.zeros(arr.shape[:2] + (3,), np.uint8)
    monkeypatch.setattr(pt, "_render_with_viewer_stretch", _render)

    class _VP:
        def width(self): return 400
        def height(self): return 300

    class _Sig:
        def emit(self, *a):
            got["done"] = True
    pb = {"img_id": id(disp), "engine": "C0", "lo": np.zeros(3, np.float32), "hi": np.ones(3, np.float32)}
    viewer = types.SimpleNamespace(viewport=lambda: _VP(), image_data=image_data, highres_tile_ready=_Sig(),
                                   _clear_highres_overlay=lambda: None, _preview_stretch_bounds=pb)
    rect = QtCore.QRectF(disp.shape[1] * 0.3, disp.shape[0] * 0.3, disp.shape[1] * 0.3, disp.shape[0] * 0.3)
    pt._request_highres_viewport_region(viewer, image_data, rect, request_id=None, ax={})
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and "done" not in got:
        time.sleep(0.02)
    assert "bounds" in got, "no tile was rendered"
    assert got["bounds"] is pb


def _dialog_viewer(preview, sp):
    shown = {}
    v = types.SimpleNamespace(
        image_data=types.SimpleNamespace(image=preview, filepath=None, channel_order="rgb"),
        stretch_params=sp,
        show_preview_array=lambda arr, channel_order=None: shown.update(arr=np.asarray(arr).copy(),
                                                                        order=channel_order))
    return v, shown


def test_back_to_rgb_in_the_stretch_dialog_gives_the_tiles_the_rgb_bounds(synthetic_project):
    """Nova pasta (5), 2026-10-07: a boolean shown, then RGB put back with the
    stretch dialog -- every zoom tile came out white / red / blue: the dialog
    redrew the preview without recording its bounds, so the tiles kept mapping
    R, G and B with the boolean's 0..1 (measured on the BCI mosaic: 100 % of
    the tile's pixels 0 or 255 per channel, 0.4 % in the preview)."""
    pt = synthetic_project
    rng = np.random.default_rng(5)
    preview = rng.gamma(2.0, 60.0, size=(60, 80, 4)).astype(np.float32)
    preview[..., 3] = (preview[..., 0] > 120).astype(np.float32)        # the boolean band
    boolean = _sp(display_mode="single", display_band=3)
    v, shown = _dialog_viewer(preview, boolean)
    pt._render_with_viewer_stretch(preview, v, _as_array=True)          # the boolean on screen
    assert float(np.max(v._preview_stretch_bounds["hi"])) == 1.0, "precondition: the boolean's bounds"
    rgb = _sp(display_mode="rgb", r_band=0, g_band=1, b_band=2, per_channel=False, low_p=0.5, high_p=99.5)
    pt._stretch_apply_preview(rgb, v, None)
    pb = v._preview_stretch_bounds
    assert pb is not None and pb["img_id"] == id(preview)
    assert float(np.min(pb["hi"])) > 100, f"the tiles still get the boolean's bounds: {pb['hi']}"
    # a tile cut from the preview renders like that part of the preview
    tile = np.array(preview[10:40, 20:70], copy=True)
    got = np.asarray(pt._render_with_viewer_stretch(tile, v, _as_array=True, bounds=pb))
    want = shown["arr"][10:40, 20:70]
    if shown["order"] == "rgb":
        want = want[..., ::-1]                                          # tiles come back BGR
    assert np.array_equal(got, want), "a tile after Apply is drawn differently from the preview"


@pytest.mark.parametrize("engine", ["C0", "F"])
def test_bounds_from_another_stretch_are_not_reused(synthetic_project, engine):
    """Whatever redraws the preview, a tile maps with recorded bounds only when
    they belong to the stretch it is drawn with."""
    pt = synthetic_project
    rng = np.random.default_rng(9)
    preview = rng.gamma(2.0, 60.0, size=(40, 50, 3)).astype(np.float32)
    rgb = _sp(display_mode="rgb", r_band=0, g_band=1, b_band=2)
    v = _viewer_with(preview, rgb)
    prefer_last = engine == "F"
    own = np.asarray(pt._render_with_viewer_stretch(preview, v, prefer_last_band=prefer_last, _as_array=True))
    stale = dict(v._preview_stretch_bounds, lo=np.zeros(3, np.float32), hi=np.ones(3, np.float32),
                 sp_sig=("a", "different", "stretch"))
    got = np.asarray(pt._render_with_viewer_stretch(preview, v, prefer_last_band=prefer_last, _as_array=True,
                                                    bounds=stale))
    assert np.array_equal(got, own), "bounds recorded for another stretch were applied"


@pytest.mark.parametrize("engine", ["C0", "F"])
def test_a_band_with_a_black_border_is_not_taken_for_a_boolean(synthetic_project, engine):
    """Nova pasta (10), 2026-10-07: a single band's sample is a 2-D image, and
    its COLUMNS were read as channels -- an all-0 column (an orthomosaic's black
    border) turned the whole band into 0..1: every value above 1 white, so a
    band change in Percentile gave a binary picture and Percentile looked dead
    after Absolute."""
    pt = synthetic_project
    rng = np.random.default_rng(4)
    band = rng.integers(20, 200, size=(60, 80)).astype(np.uint16)
    band[:, :12] = 0                                   # the border: whole columns of 0
    sp = _sp(per_channel=False)
    f = pt._compute_stretched_preview if engine == "C0" else pt._apply_viewer_stretch_numpy
    out = np.asarray(f(band, sp)).astype(np.float32)
    out = out if out.ndim == 2 else out[..., 0]
    assert float(np.mean(out[:, 12:] == 255)) < 0.2, f"{engine}: the band rendered as a boolean (all white)"
    assert len(np.unique(out[:, 12:])) > 20, f"{engine}: {len(np.unique(out[:, 12:]))} grey levels"


def test_binary_bounds_count_channels_from_the_bounds():
    from ..project_tab import _binary_stretch_bounds
    img = np.zeros((10, 6), np.float32)
    img[:, 3:] = np.arange(30, dtype=np.float32).reshape(10, 3) + 5      # not a 0/1 band
    assert _binary_stretch_bounds(img, 3.0, 40.0) == (3.0, 40.0), "an image's columns were read as channels"
    assert _binary_stretch_bounds((img > 10).astype(np.float32), 1.0, 1.0) == (0.0, 1.0)
    rgb = np.stack([img, (img > 10).astype(np.float32), img + 1], axis=2)    # (H, W, 3): channel 2 is 0/1
    lo, hi = _binary_stretch_bounds(rgb, np.array([3, 1, 4], np.float32), np.array([40, 1, 41], np.float32))
    assert list(lo) == [3, 0, 4] and list(hi) == [40, 1, 41], (lo, hi)
