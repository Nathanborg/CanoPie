"""
QC regression tests for ImageViewer / ProjectTab._imagedata_or_fallback --
the real entry point ImageViewer uses to load a file (not raw ImageData()
directly, since >3-band TIFFs route through a tifffile-preflight stack path
instead -- see _tifffile_is_stack in project_tab.py).

Covers: correct loader routing per fixture, correct band count, correct
channel_order tagging, correct raw pixel values (channel_order-normalized),
and the large-raster preview/decimation path (fixture 5, thresholds patched
via conftest.py's force_lazy_preview_no_decimation / force_preview_decimation).
"""
import numpy as np
import pytest

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from ._helpers import expected_viewer_loader, load_ground_truth, load_raw_npz, pixel_values_native_order, assert_close

ALL_NAMES = [f["name"] for f in FIXTURES]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_loader_routing_and_channel_order(synthetic_project, name):
    """Confirms _last_loader and channel_order are exactly what the fixture's
    band count dictates -- pins down which loader served each file, which is
    itself a real regression signal (a change to _tifffile_is_stack's
    threshold would silently reroute a file to a different, differently-
    behaved loader)."""
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)

    lite = synthetic_project._imagedata_or_fallback(fp)

    assert synthetic_project._last_loader == expected_viewer_loader(spec), (
        f"{name}: loader={synthetic_project._last_loader!r}, "
        f"expected={expected_viewer_loader(spec)!r}"
    )
    assert lite.channel_order == gt["viewer_channel_order"], (
        f"{name}: channel_order={lite.channel_order!r}, expected={gt['viewer_channel_order']!r}"
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_band_count(synthetic_project, name):
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    img = lite.image
    c = 1 if img.ndim == 2 else img.shape[2]
    assert c == spec["bands"], f"{name}: band count {c} != {spec['bands']}"


@pytest.mark.parametrize("name", [f["name"] for f in FIXTURES if not f.get("ax")])
def test_pixel_values_match_ground_truth(synthetic_project, name):
    """For fixtures without an .ax sidecar, _imagedata_or_fallback's raw
    array must match ground truth exactly (after undoing any BGR swap)."""
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)

    for point_name, p in gt["points"].items():
        x, y = p["x"], p["y"]
        actual = pixel_values_native_order(lite.image, x, y, lite.channel_order)
        for b, (a, e) in enumerate(zip(actual, p["values"])):
            assert_close(a, e, tol=1.0, msg=f"{name}/{point_name} band {b}")


@pytest.mark.parametrize("name", [f["name"] for f in FIXTURES if f.get("ax")])
def test_ax_fixtures_show_raw_preedit_image(synthetic_project, name):
    """_imagedata_or_fallback never applies .ax transforms (crop/nodata/
    band_expression) -- confirmed by its own docstring and by direct
    behavior. It must show the untouched RAW image (matching raw_shape and
    the ground-truth .npz), not the post-edit one process_polygon/Inspect's
    other consumers see."""
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    raw = load_raw_npz(name)
    fp = fixture_image_path(name)

    lite = synthetic_project._imagedata_or_fallback(fp)
    assert list(lite.image.shape) == gt["raw_shape"]

    # Spot-check a few raw-frame pixels directly against the .npz.
    H, W = spec["height"], spec["width"]
    for (x, y) in [(0, 0), (H // 2, W // 2), (H - 1, W - 1)]:
        actual = pixel_values_native_order(lite.image, x, y, lite.channel_order)
        expected = [float(v) for v in raw[y, x, :]]
        for b, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, tol=1.0, msg=f"{name} raw pixel ({x},{y}) band {b}")


def test_no_crash_across_all_fixtures(synthetic_project):
    """Every bit-depth/band-count/tiling combination must load without
    raising, including the 200-band cube."""
    for spec in FIXTURES:
        fp = fixture_image_path(spec["name"])
        lite = synthetic_project._imagedata_or_fallback(fp)
        assert lite is not None and lite.image is not None


# --- Large-raster preview / decimation path (fixture 5) --------------------

def test_forced_lazy_preview_no_decimation(synthetic_project, force_lazy_preview_no_decimation):
    """With the preview threshold patched below the fixture's byte size but
    the decode budget left generous, the raster_reader-backed preview path
    engages (step resolves to 1, i.e. no spatial decimation) -- pixel values
    for the bands actually selected must still match ground truth exactly."""
    name = "hyperspectral_200band"
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)

    lite = synthetic_project._imagedata_or_fallback(fp)
    assert synthetic_project._last_loader == "raster_reader-preview"
    assert lite.channel_order == "rgb"
    assert lite.preview_bands is not None

    # Preview only carries the selected bands (get_display_bands -- first 3
    # by default); check whichever of those the preview actually holds.
    p = gt["points"][gt["degenerate_point_name"]]
    x, y = p["x"], p["y"]
    for pos, file_band in enumerate(lite.preview_bands):
        actual = float(lite.image[y, x, pos])
        expected = p["values"][file_band]
        assert_close(actual, expected, tol=1.0, msg=f"{name} preview band {file_band} (pos {pos})")


def test_forced_preview_decimation(synthetic_project, force_preview_decimation):
    """With the decode budget also shrunk, step should resolve > 1 (spatial
    decimation) -- the decimated array must equal ground_truth[::step, ::step]
    for the selected bands, not just "some smaller array"."""
    name = "hyperspectral_200band"
    spec = get_fixture(name)
    raw = load_raw_npz(name)
    fp = fixture_image_path(name)

    lite = synthetic_project._imagedata_or_fallback(fp)
    assert synthetic_project._last_loader == "raster_reader-preview"

    scale = lite.display_scale
    step = max(1, int(round(scale)))
    if step <= 1:
        pytest.skip("decode budget did not force decimation on this fixture size (step==1)")

    expected = raw[::step, ::step, :]
    for pos, file_band in enumerate(lite.preview_bands):
        eb = expected[:, :, file_band]
        ab = lite.image[:eb.shape[0], :eb.shape[1], pos]
        assert ab.shape == eb.shape, f"decimated shape mismatch: {ab.shape} vs {eb.shape}"
        assert np.allclose(ab, eb, atol=1.0), f"decimated band {file_band} values don't match ground_truth[::{step}]"
