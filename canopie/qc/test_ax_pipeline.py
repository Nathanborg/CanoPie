"""
QC tests for the .ax non-destructive edit pipeline (_apply_ax_to_raw).

The .ax sidecar is CanoPie's whole editing model -- crop, rotate, resize,
hist-match and band_expression are replayed onto RAW pixels on every load and
every export. Order matters (canonically crop -> rotate -> hist -> resize ->
band_expression) and a change in geometry silently invalidates every polygon
coordinate mapped against it.

Only crop and band_expression had fixture coverage before this file; rotate,
resize and the enabled-flags are covered here by driving _apply_ax_to_raw
directly with a synthetic array whose pixels encode their own position, so a
wrong rotation/flip is unambiguous rather than plausible-looking.
"""
import numpy as np
import pytest

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor, pytest.mark.extraction]


def _positional_array(h=4, w=6, c=3):
    """value = row*100 + col*10 + band -- every pixel identifies itself, so
    any transpose/flip/rotation error is directly readable from the output."""
    arr = np.zeros((h, w, c), dtype=np.float32)
    for r in range(h):
        for col in range(w):
            for b in range(c):
                arr[r, col, b] = r * 100 + col * 10 + b
    return arr


def _apply(pt, arr, ax):
    img, _c = pt._apply_ax_to_raw(arr, ax)
    return np.asarray(img)


def test_empty_ax_is_identity(synthetic_project):
    arr = _positional_array()
    out = _apply(synthetic_project, arr.copy(), {})
    assert out.shape == arr.shape
    assert np.allclose(out, arr), "an empty .ax must not alter pixels at all"


def test_crop_takes_the_requested_window(synthetic_project):
    arr = _positional_array(h=8, w=8)
    ax = {"crop_rect": {"x": 2, "y": 3, "width": 4, "height": 2},
          "crop_rect_ref_size": {"w": 8, "h": 8}, "crop_enabled": True}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[:2] == (2, 4), f"expected 2x4 crop, got {out.shape[:2]}"
    # Top-left of the crop must be the source pixel at (row=3, col=2).
    assert np.allclose(out[0, 0, :], arr[3, 2, :]), (
        f"crop origin wrong: got {out[0, 0, :]}, expected {arr[3, 2, :]}")


def test_crop_disabled_flag_is_honored(synthetic_project):
    arr = _positional_array(h=8, w=8)
    ax = {"crop_rect": {"x": 2, "y": 3, "width": 4, "height": 2},
          "crop_rect_ref_size": {"w": 8, "h": 8}, "crop_enabled": False}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[:2] == (8, 8), "crop_enabled=False must leave geometry untouched"


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_rotate_matches_numpy_reference(synthetic_project, angle):
    """Rotation must match an independent numpy rot90 reference exactly --
    including which direction it turns, which is the easy thing to get
    backwards and the hard thing to notice by eye on real imagery."""
    arr = _positional_array(h=4, w=6)
    out = _apply(synthetic_project, arr.copy(), {"rotate": angle, "rotate_enabled": True})

    expected_shape = (arr.shape[1], arr.shape[0]) if angle in (90, 270) else arr.shape[:2]
    assert out.shape[:2] == expected_shape, (
        f"rotate {angle}: shape {out.shape[:2]}, expected {expected_shape}")

    # np.rot90 turns counter-clockwise; try both directions and require one to
    # match, then assert WHICH one so the convention is pinned, not guessed.
    ccw = np.rot90(arr, k=angle // 90)
    cw = np.rot90(arr, k=-(angle // 90))
    assert np.allclose(out, ccw) or np.allclose(out, cw), (
        f"rotate {angle} matched neither rotation direction -- geometry is wrong")


def test_rotate_disabled_flag_is_honored(synthetic_project):
    arr = _positional_array(h=4, w=6)
    out = _apply(synthetic_project, arr.copy(), {"rotate": 90, "rotate_enabled": False})
    assert out.shape[:2] == (4, 6)
    assert np.allclose(out, arr)


def test_rotate_360_and_0_are_identity(synthetic_project):
    arr = _positional_array()
    for angle in (0, 360):
        out = _apply(synthetic_project, arr.copy(), {"rotate": angle, "rotate_enabled": True})
        assert np.allclose(out, arr), f"rotate {angle} must be a no-op"


def test_resize_absolute_pixels(synthetic_project):
    """`px_w`/`px_h` are ABSOLUTE target pixels."""
    arr = _positional_array(h=8, w=8)
    ax = {"resize": {"px_w": 4, "px_h": 4}, "resize_enabled": True}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[:2] == (4, 4), f"expected 4x4, got {out.shape[:2]}"
    assert out.shape[2] == arr.shape[2], "resize must not change band count"


def test_resize_absolute_single_axis_preserves_aspect(synthetic_project):
    """Supplying only px_w scales the other axis proportionally."""
    arr = _positional_array(h=8, w=16)
    out = _apply(synthetic_project, arr.copy(),
                 {"resize": {"px_w": 8}, "resize_enabled": True})
    assert out.shape[:2] == (4, 8), f"expected aspect-preserved 4x8, got {out.shape[:2]}"


def test_resize_width_height_are_PERCENTAGES_not_pixels(synthetic_project):
    """SCHEMA TRAP, verified empirically: in a .ax `resize` block, bare
    `width`/`height` are PERCENT values (default 100), NOT target pixel
    counts -- absolute sizing uses `px_w`/`px_h`. So {"width": 50} on an
    8px-wide image yields 4px, and a naive {"width": 4} would yield
    max(1, round(8 * 0.04)) == 1px, silently collapsing the image.

    Pinned because the two key sets look interchangeable at a glance and the
    failure mode (a 1x1 image) is far from the mistake."""
    arr = _positional_array(h=8, w=8)
    out = _apply(synthetic_project, arr.copy(),
                 {"resize": {"width": 50, "height": 50}, "resize_enabled": True})
    assert out.shape[:2] == (4, 4), f"50%% of 8x8 should be 4x4, got {out.shape[:2]}"

    collapsed = _apply(synthetic_project, arr.copy(),
                       {"resize": {"width": 4, "height": 4}, "resize_enabled": True})
    assert collapsed.shape[:2] == (1, 1), (
        "4%% of 8px should clamp to 1px -- if this changed, the percent/pixel "
        "semantics of `width`/`height` may have been altered")


def test_resize_scale_percentage(synthetic_project):
    arr = _positional_array(h=8, w=8)
    out = _apply(synthetic_project, arr.copy(),
                 {"resize": {"scale": 25}, "resize_enabled": True})
    assert out.shape[:2] == (2, 2), f"25%% of 8x8 should be 2x2, got {out.shape[:2]}"


def test_resize_disabled_flag_is_honored(synthetic_project):
    arr = _positional_array(h=8, w=8)
    ax = {"resize": {"px_w": 4, "px_h": 4}, "resize_enabled": False}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[:2] == (8, 8)


def test_band_expression_appends_a_band(synthetic_project):
    """band_expression appends its result as a NEW trailing band rather than
    replacing the data -- the property that lets exports carry both the source
    bands and the derived index."""
    arr = _positional_array(h=4, w=4, c=4)
    ax = {"band_expression": "(b4-b1)/(b4+b1)", "band_enabled": True}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[2] == arr.shape[2] + 1, (
        f"expected one appended band, got {out.shape[2]} from {arr.shape[2]}")
    assert np.allclose(out[:, :, :arr.shape[2]], arr), "source bands must be preserved as-is"

    b1, b4 = arr[:, :, 0], arr[:, :, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where((b4 + b1) != 0, (b4 - b1) / (b4 + b1), 0.0)
    assert np.allclose(np.nan_to_num(out[:, :, -1]), np.nan_to_num(expected), atol=1e-5)


def test_band_expression_disabled_flag_is_honored(synthetic_project):
    arr = _positional_array(h=4, w=4, c=4)
    ax = {"band_expression": "(b4-b1)/(b4+b1)", "band_enabled": False}
    out = _apply(synthetic_project, arr.copy(), ax)
    assert out.shape[2] == arr.shape[2], "band_enabled=False must append nothing"


def test_crop_then_rotate_composition(synthetic_project):
    """Crop and rotate together: the result must be the rotation OF THE CROP,
    not a crop of the rotation -- i.e. the pipeline's ordering is observable
    and must stay stable, since polygon coordinates are mapped against it."""
    arr = _positional_array(h=8, w=8)
    ax = {"crop_rect": {"x": 0, "y": 0, "width": 4, "height": 2},
          "crop_rect_ref_size": {"w": 8, "h": 8}, "crop_enabled": True,
          "rotate": 90, "rotate_enabled": True}
    out = _apply(synthetic_project, arr.copy(), ax)
    # A 2x4 crop rotated by 90 is 4x2; a 8x8 rotated then cropped would be 2x4.
    assert out.shape[:2] == (4, 2), (
        f"expected the rotation of the crop (4x2), got {out.shape[:2]} -- "
        "pipeline order may have changed")


def test_nodata_values_do_not_alter_geometry(synthetic_project):
    """NoData config must never change the pixel grid -- it only marks pixels.
    (A NoData mask that resized or shifted the array would silently break
    every polygon mapped onto the image.)"""
    arr = _positional_array(h=6, w=6)
    out = _apply(synthetic_project, arr.copy(),
                 {"nodata_values": [0], "nodata_enabled": True})
    assert out.shape[:2] == (6, 6)
