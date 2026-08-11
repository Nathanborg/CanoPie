"""
generate_thumbnails' options dialog and DL tiling.

Two properties matter here and they pull in opposite directions:

  * DEFAULTS MUST NOT MOVE. The dialog replaced three hardcoded constants
    (zoom 1.4, 200x200 output, 2 px outline). Pressing Generate without
    touching anything has to write exactly what the old code wrote, so this is
    a UI addition rather than a behaviour change.
  * TILING MUST NOT SILENTLY DROP POLYGONS. Tiles are cut from a polygon's
    zoomed bounding box, not from the whole image. Measured over the 3504 real
    BCI crowns those boxes run p25=324, p50=424, p75=571 px, so discarding
    partial tiles would write nothing at all for 7.8% of crowns at tile=256
    (66.9% at tile=512). _grow_to_tile_grid rounds the window up instead.

The render step is deliberately separated from the writing step
(_render_thumbnail_outputs returns [(filename, array)] and touches no disk), so
all of this is checked headlessly -- generate_thumbnails itself shows two modal
dialogs and cannot be.
"""
import os

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.io]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mlm():
    from ..machine_learning_manager import MachineLearningManager
    return MachineLearningManager.__new__(MachineLearningManager)


class _FakeSrc:
    """Stands in for _ThumbSource: slices an in-memory image, counts crops."""

    def __init__(self, img):
        self.img = img
        self.calls = 0

    def crop(self, x0, y0, w, h):
        self.calls += 1
        return self.img[y0:y0 + h, x0:x0 + w].copy()


def _scene(h=400, w=400):
    """A deterministic BGR image with structure in every tile."""
    yy, xx = np.mgrid[0:h, 0:w]
    b = ((xx * 3) % 256).astype(np.uint8)
    g = ((yy * 5) % 256).astype(np.uint8)
    r = (((xx + yy) * 7) % 256).astype(np.uint8)
    return np.dstack([b, g, r])


DEFAULT_OPTS = {
    'thumbnail_size': (200, 200),
    'zoom_factor': 1.4,
    'draw_outline': True,
    'tile_enabled': False,
    'tile_size': 256,
    'tiles_inside_only': False,
}


def _opts(**over):
    o = dict(DEFAULT_OPTS)
    o.update(over)
    return o


PTS = [(120, 130), (260, 135), (255, 270), (115, 262)]
COLOR = (255, 0, 0)


# ---------------------------------------------------------------------------
# the dialog
# ---------------------------------------------------------------------------
def test_dialog_defaults_match_the_old_hardcoded_behaviour(qapp):
    """What pins "press Generate and nothing changes"."""
    from ..thumbnail_options_dialog import (ThumbnailOptionsDialog,
                                            LEGACY_THUMBNAIL_SIZE,
                                            LEGACY_ZOOM_FACTOR)
    d = ThumbnailOptionsDialog()
    got = d.get_options()
    assert got['thumbnail_size'] == LEGACY_THUMBNAIL_SIZE == (200, 200)
    assert got['zoom_factor'] == LEGACY_ZOOM_FACTOR == 1.4
    assert got['draw_outline'] is True
    assert got['tile_enabled'] is False, (
        "tiling defaults ON -- the default run would stop producing the "
        "single 200x200 thumbnail it has always produced")


def test_dialog_disables_controls_that_would_be_ignored(qapp):
    """Tiles are tile_size square, so an output width/height cannot apply."""
    from ..thumbnail_options_dialog import ThumbnailOptionsDialog
    d = ThumbnailOptionsDialog()
    assert d.tile_spin.isEnabled() is False
    assert d.width_spin.isEnabled() and d.height_spin.isEnabled()

    d.tile_cb.setChecked(True)
    assert d.tile_spin.isEnabled() is True
    assert not d.width_spin.isEnabled() and not d.height_spin.isEnabled(), (
        "output size stays editable while tiling; it would be silently ignored")


def test_legacy_constants_are_shared_not_duplicated():
    """The render path and the dialog must not drift apart."""
    from .. import machine_learning_manager as M
    from .. import thumbnail_options_dialog as T
    assert M.LEGACY_THUMBNAIL_SIZE is T.LEGACY_THUMBNAIL_SIZE
    assert M.LEGACY_ZOOM_FACTOR == T.LEGACY_ZOOM_FACTOR
    assert M.LEGACY_OUTLINE_THICKNESS == T.LEGACY_OUTLINE_THICKNESS


# ---------------------------------------------------------------------------
# the default path must not move
# ---------------------------------------------------------------------------
def test_default_path_is_identical_to_the_old_algorithm(qapp):
    """Independent reimplementation of the pre-dialog code, inline.

    Not tautological: the steps and constants below are written out from the
    old generate_thumbnails body rather than read from the new one.
    """
    import cv2
    img = _scene()
    m = _mlm()
    x0, y0, nw, nh = 100, 110, 180, 175

    got = m._render_thumbnail_outputs(
        _FakeSrc(img), PTS, x0, y0, nw, nh, img.shape[1], img.shape[0],
        COLOR, _opts(), "crown_1_base_RGB.jpg")

    # --- the old algorithm, verbatim ---
    crop = img[y0:y0 + nh, x0:x0 + nw].copy()
    adj = [[int(round(px - x0)), int(round(py - y0))] for (px, py) in PTS]
    cv2.polylines(crop, [np.array([adj], dtype=np.int32)], isClosed=True,
                  color=COLOR, thickness=2)
    expect = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)

    assert len(got) == 1
    assert got[0][0] == "crown_1_base_RGB.jpg", "the filename changed"
    assert np.array_equal(got[0][1], expect), (
        "default thumbnail output changed; this was meant to be a UI addition, "
        "not a behaviour change")


def test_outline_off_skips_the_drawing_step(qapp):
    """Off must skip cv2.polylines entirely, not draw it in a hidden colour."""
    import cv2
    img = _scene()
    m = _mlm()
    x0, y0, nw, nh = 100, 110, 180, 175

    off = m._render_thumbnail_outputs(
        _FakeSrc(img), PTS, x0, y0, nw, nh, img.shape[1], img.shape[0],
        COLOR, _opts(draw_outline=False), "a.jpg")[0][1]
    on = m._render_thumbnail_outputs(
        _FakeSrc(img), PTS, x0, y0, nw, nh, img.shape[1], img.shape[0],
        COLOR, _opts(draw_outline=True), "a.jpg")[0][1]

    undrawn = cv2.resize(img[y0:y0 + nh, x0:x0 + nw].copy(), (200, 200),
                         interpolation=cv2.INTER_AREA)
    assert np.array_equal(off, undrawn), "pixels were still altered with the outline off"
    assert not np.array_equal(off, on), "the outline toggle did nothing"


def test_output_size_is_honoured(qapp):
    img = _scene()
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), PTS, 100, 110, 180, 175, img.shape[1], img.shape[0],
        COLOR, _opts(thumbnail_size=(64, 96)), "a.jpg")
    assert out[0][1].shape[:2] == (96, 64)      # (h, w)


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bbox", [16, 60, 128, 200, 256, 300, 700])
def test_every_polygon_yields_at_least_one_tile(qapp, bbox):
    """THE 7.8% regression.

    A polygon whose zoomed bbox is smaller than one tile must still produce a
    tile. Dropping the remainder instead wrote nothing at all for 7.8% of the
    real BCI crowns at tile=256, and 66.9% at tile=512 -- silently.
    """
    img = _scene(1024, 1024)
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), PTS, 300, 300, bbox, bbox, img.shape[1], img.shape[0],
        COLOR, _opts(tile_enabled=True, tile_size=128), "a.jpg")
    assert len(out) >= 1, (
        f"a {bbox}x{bbox} crop produced NO tiles at tile=128; partial tiles are "
        "being dropped instead of the window being grown")
    for _name, tile in out:
        assert tile.shape[:2] == (128, 128), "tiles are not uniformly sized"


def test_tiles_are_non_overlapping_and_reassemble(qapp):
    """Row-major tiles must rebuild the grown crop exactly."""
    from ..machine_learning_manager import _grow_to_tile_grid, _tile_grid

    img = _scene(1024, 1024)
    tile = 64
    gx, gy, gw, gh = _grow_to_tile_grid(300, 300, 200, 170, tile,
                                        img.shape[1], img.shape[0])
    assert gw % tile == 0 and gh % tile == 0, "the grid is not a whole number of tiles"
    crop = img[gy:gy + gh, gx:gx + gw]

    rebuilt = np.zeros_like(crop)
    seen = set()
    for r, c, t in _tile_grid(crop, tile):
        assert (r, c) not in seen, "the same cell was emitted twice"
        seen.add((r, c))
        rebuilt[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    assert len(seen) == (gh // tile) * (gw // tile)
    assert np.array_equal(rebuilt, crop), "tiles do not reassemble into the crop"


def test_grown_grid_stays_inside_the_image(qapp):
    """Growing must never read outside the raster, including in corners."""
    from ..machine_learning_manager import _grow_to_tile_grid
    W = H = 500
    for (x, y, w, h) in [(0, 0, 30, 30), (470, 470, 30, 30), (240, 0, 20, 20),
                         (0, 240, 20, 20), (100, 100, 400, 400)]:
        gx, gy, gw, gh = _grow_to_tile_grid(x, y, w, h, 128, W, H)
        assert gx >= 0 and gy >= 0
        assert gx + gw <= W and gy + gh <= H, (
            f"grid {(gx, gy, gw, gh)} runs outside the {W}x{H} image")
        assert gw > 0 and gh > 0


def test_tiny_image_still_yields_one_padded_tile(qapp):
    """When the image itself is smaller than a tile, pad -- do not emit nothing."""
    img = _scene(40, 40)
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), [(5, 5), (30, 5), (30, 30)], 0, 0, 40, 40, 40, 40,
        COLOR, _opts(tile_enabled=True, tile_size=128), "a.jpg")
    assert len(out) == 1
    assert out[0][1].shape[:2] == (128, 128)
    assert out[0][1].any(), "the padded tile is entirely blank"


def test_tile_names_are_zero_padded_and_sort_row_major(qapp):
    """These feed a training pipeline that globs the folder, so order matters."""
    img = _scene(1024, 1024)
    tile = 32
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), PTS, 100, 100, 400, 400, img.shape[1], img.shape[0],
        COLOR, _opts(tile_enabled=True, tile_size=tile), "crown_1_base_RGB_p2.jpg")
    names = [n for n, _ in out]

    assert len(names) > 100, "not enough tiles to exercise two-digit indices"
    assert names[0] == "crown_1_base_RGB_p2_r00c00.jpg", (
        f"unexpected tile name {names[0]!r}; the _p2 polygon suffix must stay "
        "ahead of the tile suffix")
    assert sorted(names) == names, (
        "tile names do not sort into row-major order -- indices are probably "
        "not zero-padded, which breaks past 10 rows/columns")
    assert all(n.endswith(".jpg") for n in names)


def test_tiling_replaces_the_single_thumbnail(qapp):
    """Tiling on writes tiles only -- no stray differently-sized image."""
    img = _scene(1024, 1024)
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), PTS, 100, 100, 300, 300, img.shape[1], img.shape[0],
        COLOR, _opts(tile_enabled=True, tile_size=128), "a.jpg")
    assert len(out) > 1
    assert all(t.shape[:2] == (128, 128) for _n, t in out)
    assert not any(n == "a.jpg" for n, _ in out)


# ---------------------------------------------------------------------------
# "only squares that fit inside the polygon"
# ---------------------------------------------------------------------------
def _square(x0, y0, side):
    return [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0 + y0 * 0 + x0 * 0 + x0, y0 + side)]


def test_inside_only_keeps_only_fully_contained_squares(qapp):
    """Every kept tile must be entirely interior, not merely overlapping."""
    from ..machine_learning_manager import _tiles_fully_inside
    import cv2

    # A 400x400 square polygon inside a 512x512 grid at tile=64.
    pts = [(100, 100), (500, 100), (500, 500), (100, 500)]
    gx, gy, gw, gh = 60, 60, 512, 512
    keep = _tiles_fully_inside(pts, gx, gy, gw, gh, 64)

    mask = np.zeros((gh, gw), np.uint8)
    cv2.fillPoly(mask, np.array([[[px - gx, py - gy] for px, py in pts]], np.int32), 1)
    for r in range(keep.shape[0]):
        for c in range(keep.shape[1]):
            block = mask[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]
            assert keep[r, c] == bool(block.all()), (
                f"tile ({r},{c}) kept={keep[r, c]} but interior={bool(block.all())}")
    assert keep.any() and not keep.all(), "the fixture must have both kinds of tile"


def test_inside_only_is_a_strict_subset_of_the_full_grid(qapp):
    img = _scene(1024, 1024)
    m = _mlm()
    pts = [(300, 300), (700, 300), (700, 700), (300, 700)]
    args = (_FakeSrc(img), pts, 300, 300, 400, 400, img.shape[1], img.shape[0], COLOR)

    allt = m._render_thumbnail_outputs(*args, _opts(tile_enabled=True, tile_size=64), "a.jpg")
    inside = m._render_thumbnail_outputs(
        *args, _opts(tile_enabled=True, tile_size=64, tiles_inside_only=True), "a.jpg")

    names_all = {n for n, _ in allt}
    names_in = {n for n, _ in inside}
    assert names_in and names_in < names_all, (
        "inside-only did not reduce the grid, or produced names outside it")
    # A kept tile is byte-identical to the same cell of the unfiltered grid:
    # this is a filter, not a different grid.
    by_name = dict(allt)
    for n, t in inside:
        assert np.array_equal(t, by_name[n])


def test_inside_only_keeps_grid_positions_in_the_names(qapp):
    """Names keep their (row, col), so gaps appear rather than a renumbering."""
    img = _scene(1024, 1024)
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), [(300, 300), (700, 300), (700, 700), (300, 700)],
        300, 300, 400, 400, img.shape[1], img.shape[0], COLOR,
        _opts(tile_enabled=True, tile_size=64, tiles_inside_only=True), "a.jpg")
    names = [n for n, _ in out]
    assert names[0] != "a_r00c00.jpg", (
        "the first kept tile is at grid origin; the fixture polygon does not "
        "reach the grid edge, so an interior-only run must start later")
    assert sorted(names) == names


def test_polygon_narrower_than_a_tile_keeps_nothing(qapp):
    """The documented cost of the option -- and why the dialog estimates it."""
    img = _scene(1024, 1024)
    out = _mlm()._render_thumbnail_outputs(
        _FakeSrc(img), [(500, 500), (560, 500), (560, 560), (500, 560)],
        500, 500, 84, 84, img.shape[1], img.shape[0], COLOR,
        _opts(tile_enabled=True, tile_size=128, tiles_inside_only=True), "a.jpg")
    assert out == [], "a 60 px polygon cannot contain a 128 px square"


def test_inside_only_off_is_unchanged(qapp):
    """The new flag must not disturb the existing tiled path."""
    img = _scene(1024, 1024)
    m = _mlm()
    args = (_FakeSrc(img), PTS, 100, 100, 300, 300, img.shape[1], img.shape[0], COLOR)
    a = m._render_thumbnail_outputs(*args, _opts(tile_enabled=True, tile_size=128), "a.jpg")
    b = m._render_thumbnail_outputs(
        *args, _opts(tile_enabled=True, tile_size=128, tiles_inside_only=False), "a.jpg")
    assert [n for n, _ in a] == [n for n, _ in b]
    assert all(np.array_equal(x, y) for (_n, x), (_m2, y) in zip(a, b))


# ---------------------------------------------------------------------------
# the live estimate
# ---------------------------------------------------------------------------
def _bbox_args(pts, zoom, W, H):
    """The (x0, y0, new_w, new_h) generate_thumbnails derives for a polygon.

    Written out here rather than imported so the estimate is checked against
    the documented geometry, not against whatever the estimator happens to do.
    """
    import cv2
    x, y, w, h = cv2.boundingRect(np.array(pts, dtype=np.int32))
    nw = max(16, int(round(w * zoom)))
    nh = max(16, int(round(h * zoom)))
    x0 = max(0, min(int(round(x + w / 2.0 - nw / 2.0)), W - nw))
    y0 = max(0, min(int(round(y + h / 2.0 - nh / 2.0)), H - nh))
    return x0, y0, nw, nh


def test_estimate_matches_what_the_run_actually_writes(qapp):
    """The readout must not promise a number the render path won't deliver."""
    from ..machine_learning_manager import estimate_tile_yield

    img = _scene(1024, 1024)
    m = _mlm()
    pts = [(300, 300), (700, 300), (700, 700), (300, 700)]
    polys = [(pts, 1024, 1024)]
    x0, y0, nw, nh = _bbox_args(pts, 1.4, 1024, 1024)

    for tile in (64, 128, 256):
        for inside in (False, True):
            est = estimate_tile_yield(polys, tile, 1.4, inside, sample_cap=None)
            out = m._render_thumbnail_outputs(
                _FakeSrc(img), pts, x0, y0, nw, nh, 1024, 1024, COLOR,
                _opts(tile_enabled=True, tile_size=tile, tiles_inside_only=inside),
                "a.jpg")
            assert est['tiles'] == len(out), (
                f"tile={tile} inside={inside}: estimate said {est['tiles']}, "
                f"the render produced {len(out)}")


def test_estimate_falls_as_the_tile_grows(qapp):
    """'the number will vary with the window size' -- monotonically."""
    from ..machine_learning_manager import estimate_tile_yield
    pts = [(1000, 1000), (1400, 1000), (1400, 1400), (1000, 1400)]
    polys = [(pts, 5000, 5000)] * 20

    counts = [estimate_tile_yield(polys, t, 1.4, False, sample_cap=None)['tiles']
              for t in (64, 128, 256, 512)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1], counts

    inside = [estimate_tile_yield(polys, t, 1.4, True, sample_cap=None)['tiles']
              for t in (64, 128, 256, 512)]
    assert inside[-1] == 0, "a 512 square cannot fit inside a 400 px crown"
    assert all(i <= a for i, a in zip(inside, counts))


def test_estimate_reports_polygons_that_yield_nothing(qapp):
    from ..machine_learning_manager import estimate_tile_yield
    small = [([(0, 0), (60, 0), (60, 60), (0, 60)], 5000, 5000)] * 10
    est = estimate_tile_yield(small, 128, 1.4, True, sample_cap=None)
    assert est['tiles'] == 0
    assert est['empty'] == 10, (
        "polygons that produce no tile are not being counted; that number is "
        "the whole point of the readout")


def test_estimate_scales_from_a_sample(qapp):
    from ..machine_learning_manager import estimate_tile_yield
    pts = [(1000, 1000), (1400, 1000), (1400, 1400), (1000, 1400)]
    polys = [(pts, 5000, 5000)] * 1000
    exact = estimate_tile_yield(polys, 128, 1.4, False, sample_cap=None)
    sampled = estimate_tile_yield(polys, 128, 1.4, False, sample_cap=50)
    assert sampled['sampled'] == 50 and sampled['total'] == 1000
    assert sampled['tiles'] == exact['tiles'], (
        "identical polygons must extrapolate exactly")


def test_dialog_shows_a_live_estimate_that_moves_with_tile_size(qapp):
    from ..thumbnail_options_dialog import ThumbnailOptionsDialog
    pts = [(1000, 1000), (1400, 1000), (1400, 1400), (1000, 1400)]
    d = ThumbnailOptionsDialog([(pts, 5000, 5000)] * 30)

    d.tile_cb.setChecked(True)
    d.tile_spin.setValue(64)
    d.update_estimate()
    at64 = d.estimate_lbl.text()
    d.tile_spin.setValue(256)
    d.update_estimate()
    at256 = d.estimate_lbl.text()
    assert at64 and at256 and at64 != at256, (
        f"the estimate does not change with the tile size ({at64!r} at 64 vs "
        f"{at256!r} at 256)")

    d.inside_cb.setChecked(True)
    d.update_estimate()
    assert "without the inside-only filter" in d.estimate_lbl.text()
    assert d.get_options()['tiles_inside_only'] is True


def test_inside_only_defaults_off_and_needs_tiling(qapp):
    from ..thumbnail_options_dialog import ThumbnailOptionsDialog
    d = ThumbnailOptionsDialog()
    assert d.get_options()['tiles_inside_only'] is False
    assert d.inside_cb.isEnabled() is False, (
        "the inside-only box is usable while tiling is off, where it does nothing")
    d.tile_cb.setChecked(True)
    assert d.inside_cb.isEnabled() is True


def test_dialog_works_without_polygons(qapp):
    """The estimate is optional; the dialog must not require it."""
    from ..thumbnail_options_dialog import ThumbnailOptionsDialog
    d = ThumbnailOptionsDialog()
    d.tile_cb.setChecked(True)
    d.update_estimate()
    assert d.estimate_lbl.text() == ""
    assert d.get_options()['tile_enabled'] is True


# ---------------------------------------------------------------------------
# COG safety
# ---------------------------------------------------------------------------
def test_tiling_reads_the_crop_once_not_once_per_tile(qapp):
    """The constraint carried over from the COG out-of-memory fix."""
    img = _scene(1024, 1024)
    src = _FakeSrc(img)
    out = _mlm()._render_thumbnail_outputs(
        src, PTS, 100, 100, 400, 400, img.shape[1], img.shape[0],
        COLOR, _opts(tile_enabled=True, tile_size=64), "a.jpg")
    assert len(out) >= 36, "expected a many-tile grid"
    assert src.calls == 1, (
        f"the tiled path issued {src.calls} crop reads for {len(out)} tiles; it "
        "must fetch the grown window once and slice it in memory")


def test_tiling_issues_one_window_read_on_a_lazy_raster(qapp, fixtures_ready,
                                                        monkeypatch):
    """Same property, driven through the real _ThumbSource and reader."""
    from .. import raster_reader
    from ..raster_reader import probe, open_reader, LazyChannels
    from ..machine_learning_manager import _ThumbSource

    fp = fixture_image_path("rgb_8bit_untiled")
    profile = probe(fp)
    if profile is None or not profile.is_windowable:
        pytest.skip("fixture is not windowable")
    reader = open_reader(fp, profile)
    if reader is None:
        pytest.skip("no tiled reader for this fixture")
    lazy = LazyChannels(reader, order=list(range(profile.count)))

    m = _mlm()
    # Built before the spy: the one-off contrast sample is not a per-tile read.
    src = _ThumbSource(m, lazy)
    assert src.ok

    calls = []
    real = raster_reader.TiledRasterReader.read_window

    def spy(rself, x0, y0, x1, y1, bands=None, level=0):
        calls.append((x0, y0, x1, y1))
        return real(rself, x0, y0, x1, y1, bands=bands, level=level)

    monkeypatch.setattr(raster_reader.TiledRasterReader, "read_window", spy)

    out = m._render_thumbnail_outputs(
        src, [(10, 10), (50, 10), (50, 50)], 0, 0, 64, 64,
        profile.width, profile.height,
        COLOR, _opts(tile_enabled=True, tile_size=16), "a.jpg")

    assert len(out) >= 16, "expected a many-tile grid"
    assert len(calls) == 1, (
        f"{len(calls)} raster reads for {len(out)} tiles; the tiled path is "
        "re-reading the image per tile, which is what the COG fix removed")
