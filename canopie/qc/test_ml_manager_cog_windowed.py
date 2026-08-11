"""
generate_thumbnails and generate_segmentation_images must read only what they
use, never the whole raster.

THE INCIDENT, on the BCI COG (48031 x 50101, 4 bands, uint16, 17.93 GiB, on a
15.7 GiB machine):

  * generate_thumbnails built ONE whole-image 8-bit BGR preview
    (`_as_8bit_bgr`) and sliced 200x200 crops out of it. Reproduced verbatim:
    `read_window asked for 1 band x 50101 x 48031 = 4.48 GiB`, matching the
    reported `_ArrayMemoryError`. Its `except` branch then fell back to
    `np.asarray(arr)`, which asks the reader for exactly the same frame, so it
    could not rescue itself.
  * generate_segmentation_images called `_get_geometry_only_image` -> tifffile
    `asarray()` over the whole file (17.93 GiB, then float32 on top) to obtain
    NOTHING BUT THE SHAPE -- it never reads a pixel of the image.
  * the mask itself is 2.41 Gpx, which is 2.24x OpenCV's CV_IO_MAX_IMAGE_PIXELS,
    so cv2.imwrite could not have written it as PNG at any memory budget.

The rule these tests enforce: an eager (small) image must behave EXACTLY as
before -- this is a memory fix, not a behaviour change -- while a lazily-read
raster must never request more than a polygon-sized window.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture

pytestmark = [pytest.mark.io]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mlm(parent=None):
    from ..machine_learning_manager import MachineLearningManager
    m = MachineLearningManager.__new__(MachineLearningManager)
    m.parent_tab = parent
    return m


def _lazy_view(fp):
    """LazyChannels over a small fixture, so the windowed path is exercised
    without needing a multi-GB file (same trick as conftest.force_lazy_export)."""
    from ..raster_reader import probe, open_reader, LazyChannels
    profile = probe(fp)
    if profile is None or not profile.is_windowable:
        pytest.skip(f"{fp} is not windowable")
    reader = open_reader(fp, profile)
    if reader is None:
        pytest.skip("no tiled reader for this fixture")
    return LazyChannels(reader, order=list(range(profile.count))), profile


class _ReadSpy:
    """Records every reader window, so a test can assert what was requested."""

    def __init__(self, monkeypatch):
        from .. import raster_reader
        self.calls = []
        real = raster_reader.TiledRasterReader.read_window

        def spy(rself, x0, y0, x1, y1, bands=None, level=0):
            n = len(bands) if bands is not None else rself.profile.count
            h, w = max(0, y1 - y0), max(0, x1 - x0)
            self.calls.append(
                n * h * w * np.dtype(rself.profile.dtype).itemsize)
            return real(rself, x0, y0, x1, y1, bands=bands, level=level)

        monkeypatch.setattr(raster_reader.TiledRasterReader, "read_window", spy)

    @property
    def peak(self):
        return max(self.calls) if self.calls else 0


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------
def test_eager_thumbnail_crops_are_byte_identical_to_the_old_path(qapp, fixtures_ready):
    """The no-behaviour-change guarantee for every image that works today.

    _ThumbSource on an eager array must be the same whole-frame preview,
    sliced the same way -- so existing thumbnails do not shift by a single
    byte because this fix landed.
    """
    from ..machine_learning_manager import _ThumbSource
    import tifffile

    fp = fixture_image_path("rgb_8bit_untiled")
    arr = tifffile.imread(fp)
    m = _mlm()

    vis_full = m._as_8bit_bgr(arr)
    assert vis_full is not None

    src = _ThumbSource(m, arr)
    assert src.ok
    for (x0, y0, w, h) in [(0, 0, 16, 16), (5, 7, 20, 12), (10, 10, 8, 8)]:
        expect = vis_full[y0:y0 + h, x0:x0 + w].copy()
        got = src.crop(x0, y0, w, h)
        assert np.array_equal(got, expect), (
            f"eager crop at ({x0},{y0}) changed; this path must be untouched")


def test_lazy_thumbnail_crops_match_the_eager_whole_frame_crop(qapp, fixtures_ready):
    """Lazy vs eager agreement, the pattern test_lazy_eager_agreement uses.

    A crop read through the windowed path must equal the same crop taken out
    of the whole-frame preview. The contrast scale is sampled, and for a
    fixture this small sample_bands reads the region exactly, so the two are
    pixel-identical rather than merely close.
    """
    from ..machine_learning_manager import _ThumbSource
    import tifffile

    fp = fixture_image_path("rgb_8bit_untiled")
    arr = tifffile.imread(fp)
    m = _mlm()

    vis_full = m._as_8bit_bgr(arr)
    lazy, profile = _lazy_view(fp)
    src = _ThumbSource(m, lazy)
    assert src.ok and src._lazy

    h, w = arr.shape[:2]
    for (x0, y0, cw, ch) in [(0, 0, 16, 16), (4, 6, 12, 12), (w - 10, h - 10, 10, 10)]:
        got = src.crop(x0, y0, cw, ch)
        expect = vis_full[y0:y0 + ch, x0:x0 + cw]
        assert got.shape == expect.shape
        assert np.array_equal(got, expect), (
            "the windowed crop differs from the whole-frame-then-crop result; "
            "this is meant to be a memory fix, not a pixel change")


def test_lazy_thumbnails_never_request_the_whole_frame(qapp, fixtures_ready, monkeypatch):
    """The actual crash condition: no read may approach the full frame."""
    from ..machine_learning_manager import _ThumbSource

    fp = fixture_image_path("rgb_8bit_untiled")
    lazy, profile = _lazy_view(fp)
    m = _mlm()

    # Built BEFORE the spy: the one-off contrast sample is allowed to cover the
    # frame (on a fixture this small sample_bands reads the whole region in one
    # pass, which is the point -- it is bounded by a byte budget, not by the
    # raster size). What must never happen is a whole-frame read PER CROP.
    src = _ThumbSource(m, lazy)
    spy = _ReadSpy(monkeypatch)
    for i in range(4):
        assert src.crop(i, i, 8, 8) is not None

    assert spy.calls, "nothing was read at all"
    crop_bytes = 8 * 8 * profile.count * np.dtype(profile.dtype).itemsize
    assert spy.peak <= crop_bytes, (
        f"a crop read {spy.peak} bytes for an 8x8 window that needs "
        f"{crop_bytes}; generate_thumbnails is materialising the raster again")


def test_as_8bit_bgr_refuses_an_oversized_whole_frame(qapp, fixtures_ready, monkeypatch):
    """The guard that turns a machine-killing allocation into a log line."""
    from .. import machine_learning_manager as M

    fp = fixture_image_path("rgb_8bit_untiled")
    lazy, _ = _lazy_view(fp)
    monkeypatch.setattr(M, "_WHOLE_FRAME_VIS_MAX_BYTES", 1)

    spy = _ReadSpy(monkeypatch)
    assert _mlm()._as_8bit_bgr(lazy) is None, (
        "_as_8bit_bgr attempted an oversized whole-frame read instead of refusing")
    assert not spy.calls, "it refused but still read the raster"


def test_as_8bit_bgr_has_no_function_local_logging_import():
    """A regression with teeth, because this defeated the guard silently.

    `import logging` inside _as_8bit_bgr made `logging` a LOCAL name for the
    whole function, so the size guard -- which logs before that line executes
    -- raised UnboundLocalError into its own `except Exception: pass` and let
    the 4.48 GiB read proceed exactly as before. The guard looked correct and
    did nothing.
    """
    import inspect
    from ..machine_learning_manager import MachineLearningManager

    src = inspect.getsource(MachineLearningManager._as_8bit_bgr)
    # Statements only -- the function's own comment explains the hazard and
    # would otherwise match a naive substring search.
    offenders = [ln.strip() for ln in src.splitlines()
                 if ln.strip().startswith(("import logging", "from logging import"))]
    assert not offenders, (
        f"_as_8bit_bgr imports logging locally again ({offenders}); that rebinds "
        "the name for the whole function and silently disables the whole-frame "
        "size guard")


# ---------------------------------------------------------------------------
# segmentation: shape without pixels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["rgb_8bit_untiled", "ax_crop_nodata_source",
                                  "hyperspectral_200band"])
def test_geometry_only_shape_matches_the_full_pipeline(qapp, fixtures_ready, name):
    """The cheap shape must equal what actually decoding the image produces."""
    fp = fixture_image_path(name)
    m = _mlm()
    m._load_ax_mods = lambda p: {}

    img, C = m._get_geometry_only_image(fp)
    if img is None:
        pytest.skip(f"{name} could not be decoded")
    expect = (img.shape[0], img.shape[1], C)

    assert m._geometry_only_shape(fp) == expect, (
        "_geometry_only_shape disagrees with _apply_ax_geometry_only; the two "
        "shape derivations have drifted apart")


@pytest.mark.parametrize("ax,expect", [
    ({}, (100, 200, 3)),
    ({"rotate": 90}, (200, 100, 3)),
    ({"rotate": 180}, (100, 200, 3)),
    ({"rotate": 90, "rotate_enabled": False}, (100, 200, 3)),
    ({"crop_rect": {"x": 10, "y": 20, "width": 50, "height": 30}}, (30, 50, 3)),
    ({"crop_rect": {"x": 10, "y": 20, "width": 50, "height": 30},
      "crop_enabled": False}, (100, 200, 3)),
    ({"resize": {"scale": 50.0}}, (50, 100, 3)),
    ({"resize": {"px_w": 80, "px_h": 40}}, (40, 80, 3)),
    ({"resize": {"px_w": 100}}, (50, 100, 3)),
    ({"rotate": 90, "resize": {"scale": 50.0}}, (100, 50, 3)),
])
def test_geometry_shape_arithmetic(qapp, ax, expect):
    """Each geometric op, checked against the shape maths directly."""
    assert _mlm()._geometry_shape_after_ax((100, 200, 3), ax) == expect


def test_geometry_only_shape_reads_no_pixels(qapp, fixtures_ready, monkeypatch):
    """The 17.93 GiB read that produced nothing but two integers."""
    fp = fixture_image_path("hyperspectral_200band")
    m = _mlm()
    m._load_ax_mods = lambda p: {}

    def _boom(*a, **k):
        raise AssertionError(
            "_geometry_only_shape decoded image data; it needs the header only")

    monkeypatch.setattr(m, "_load_raw_image", _boom)
    monkeypatch.setattr(m, "_get_geometry_only_image", _boom)

    shape = m._geometry_only_shape(fp)
    assert shape is not None and len(shape) == 3


# ---------------------------------------------------------------------------
# segmentation: the mask
# ---------------------------------------------------------------------------
def _shapes(h, w, n=60):
    out = []
    for i in range(n):
        cx = (i * 197) % max(1, w - 120)
        cy = (i * 43) % max(1, h - 120)
        if i % 7 == 0:
            out.append((i + 1, [(cx, cy)]))                     # point shape
        else:
            out.append((i + 1, [(cx, cy), (cx + 90, cy + 12),
                                (cx + 75, cy + 95), (cx + 8, cy + 70)]))
    return out


def test_strip_built_mask_equals_a_single_allocation_build(tmp_path):
    """The strip path must be pixel-exact, not merely close.

    It is not automatic. Handing cv2.fillPoly a STRIP as its canvas lets it
    clip the polygon against the strip edge, and that cut adds boundary pixels
    a full-canvas fill never produces -- measured at 3102 wrong pixels over 120
    polygons before _stamp_mask_shape rasterised into each polygon's own bbox
    instead. Wrong pixels in a segmentation mask are wrong training labels.
    """
    import tifffile
    from ..machine_learning_manager import (_fill_mask_shape, _write_mask_tiled)

    H, W = 1600, 1400                      # spans several 512-row strips
    shapes = _shapes(H, W)

    dense = np.zeros((H, W), np.uint16)
    for lbl, pts in shapes:
        _fill_mask_shape(dense, pts, lbl, H, W)

    out = tmp_path / "mask.tif"
    _write_mask_tiled(str(out), H, W, shapes)
    back = tifffile.imread(str(out))

    assert back.shape == dense.shape and back.dtype == dense.dtype
    assert int(dense.max()) > 0, "the fixture drew nothing"
    assert np.array_equal(back, dense), (
        f"strip-built mask differs from the dense build in "
        f"{int((back != dense).sum())} pixel(s)")


def test_tiled_mask_is_written_at_full_resolution(tmp_path):
    """Downscaling a segmentation mask would silently break its alignment."""
    import tifffile
    from ..machine_learning_manager import _write_mask_tiled

    H, W = 1300, 1100                      # deliberately not tile multiples
    out = tmp_path / "mask.tif"
    _write_mask_tiled(str(out), H, W, _shapes(H, W))

    with tifffile.TiffFile(str(out)) as tf:
        page = tf.pages[0]
        assert tuple(page.shape) == (H, W)
        assert page.dtype == np.uint16
        assert page.is_tiled, "not tiled, so it cannot be read back windowed"


def test_polygon_spanning_many_strips_is_not_cut(tmp_path):
    """A single tall polygon crossing several strip boundaries stays solid."""
    import tifffile
    from ..machine_learning_manager import _fill_mask_shape, _write_mask_tiled

    H, W = 2048, 600
    tall = [(1, [(100, 30), (500, 30), (500, H - 30), (100, H - 30)])]

    dense = np.zeros((H, W), np.uint16)
    _fill_mask_shape(dense, tall[0][1], 1, H, W)

    out = tmp_path / "tall.tif"
    _write_mask_tiled(str(out), H, W, tall)
    back = tifffile.imread(str(out))

    assert np.array_equal(back, dense)
    # No blank row where a strip boundary falls.
    rows = (back > 0).sum(axis=1)
    assert (rows[31:H - 31] > 0).all(), "a strip boundary left an empty row"
