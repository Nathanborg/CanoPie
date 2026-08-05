"""
QC tests for display stretch rendering (_compute_stretched_preview).

This is the DISPLAY-ONLY path: it turns the scientific array into an 8-bit
frame for the screen. The property that matters most is that it is display
-only -- if it ever mutated the underlying array, every exported statistic
would silently inherit a contrast stretch, which is exactly the kind of
corruption this suite exists to prevent.
"""
import numpy as np
import pytest

from ..project_tab import _StretchParams


def _gradient(h=16, w=16, c=3, lo=0.0, hi=1000.0):
    """A deterministic ramp per band, so percentile behavior is predictable."""
    base = np.linspace(lo, hi, h * w, dtype=np.float32).reshape(h, w)
    return np.stack([base + (b * 10.0) for b in range(c)], axis=2)


def test_output_is_uint8_and_display_shaped(synthetic_project):
    img = _gradient()
    out = synthetic_project._compute_stretched_preview(img, _StretchParams())
    out = np.asarray(out)
    assert out.dtype == np.uint8, f"display frame must be uint8, got {out.dtype}"
    assert out.shape[:2] == img.shape[:2], "stretch must not change spatial dims"
    assert out.ndim in (2, 3), f"expected (H,W) or (H,W,3), got {out.shape}"


def test_source_array_is_never_mutated(synthetic_project):
    """THE critical guarantee. Inspect and every CSV export read the same
    array this is handed; mutating it in place would corrupt exported data
    with a display-only transform."""
    img = _gradient()
    before = img.copy()
    for params in (_StretchParams(mode="percentile"),
                   _StretchParams(mode="stddev", k_sigma=2.0),
                   _StretchParams(mode="absolute", min_val=100.0, max_val=900.0)):
        synthetic_project._compute_stretched_preview(img, params)
        assert np.array_equal(img, before), (
            f"_compute_stretched_preview mutated its input under {params.mode!r}")


def test_percentile_stretch_spans_full_display_range(synthetic_project):
    """A wide-percentile stretch of a smooth ramp should use most of 0-255 --
    otherwise the display would look washed out / clipped."""
    img = _gradient()
    out = np.asarray(synthetic_project._compute_stretched_preview(
        img, _StretchParams(mode="percentile", low_p=0.0, high_p=100.0)))
    assert int(out.min()) <= 5, f"expected near-black present, min={out.min()}"
    assert int(out.max()) >= 250, f"expected near-white present, max={out.max()}"


def test_absolute_stretch_clips_outside_range(synthetic_project):
    """With an explicit absolute window, values beyond it must saturate rather
    than wrap or produce noise."""
    img = _gradient(lo=0.0, hi=1000.0)
    out = np.asarray(synthetic_project._compute_stretched_preview(
        img, _StretchParams(mode="absolute", min_val=400.0, max_val=600.0, clip=True)))
    assert int(out.min()) == 0, f"values below the window should clip to 0, got {out.min()}"
    assert int(out.max()) == 255, f"values above the window should clip to 255, got {out.max()}"


def _shape_varying_stack():
    """Bands that differ in SPATIAL SHAPE, not merely in scale.

    This distinction is essential and easy to get wrong: a per-band stretch
    normalizes each band independently, so it is SCALE-INVARIANT -- bands that
    are all linear ramps differing only in magnitude (0..100, 0..200, ...) all
    render to the exact same 0-255 frame. A test built on such data cannot
    detect band selection at all and will report a false failure (verified
    empirically while writing these tests). Varying the spatial pattern is
    what makes band choice observable through a normalizing stretch.
    """
    img = np.zeros((8, 8, 4), dtype=np.float32)
    img[:, :, 0] = np.tile(np.arange(8, dtype=np.float32), (8, 1))                 # horizontal ramp
    img[:, :, 1] = np.tile(np.arange(8, dtype=np.float32).reshape(8, 1), (1, 8))   # vertical ramp
    img[:, :, 2] = 42.0                                                            # flat
    img[:, :, 3] = np.add.outer(np.arange(8), np.arange(8)).astype(np.float32)     # diagonal
    return img


def test_single_band_mode_renders_the_selected_band(synthetic_project):
    """display_mode="single" is what the band-selector bar and the Stretch
    dialog's band picker both drive. Rendering different bands of a stack
    whose bands differ in pattern must produce different frames."""
    img = _shape_varying_stack()

    frames = []
    for band in range(4):
        out = synthetic_project._compute_stretched_preview(
            img, _StretchParams(display_mode="single", display_band=band))
        frames.append(np.asarray(out))

    for f in frames:
        assert f.shape[:2] == (8, 8)
    assert not np.array_equal(frames[0], frames[1]), (
        "horizontal-ramp and vertical-ramp bands rendered identically -- "
        "display_band is being ignored")
    assert not all(np.array_equal(frames[0], f) for f in frames[1:]), (
        "every band rendered identically -- display_band is being ignored")


def test_rgb_mode_composes_the_named_bands(synthetic_project):
    """display_mode="rgb" must honor r_band/g_band/b_band; swapping two of
    them must visibly change the composite."""
    img = _shape_varying_stack()

    normal = np.asarray(synthetic_project._compute_stretched_preview(
        img, _StretchParams(display_mode="rgb", r_band=0, g_band=1, b_band=3)))
    swapped = np.asarray(synthetic_project._compute_stretched_preview(
        img, _StretchParams(display_mode="rgb", r_band=3, g_band=1, b_band=0)))

    assert normal.ndim == 3 and normal.shape[2] == 3, f"rgb mode must yield 3 channels, got {normal.shape}"
    assert not np.array_equal(normal, swapped), (
        "swapping r_band and b_band produced an identical frame -- band "
        "assignment is being ignored")


def test_all_constant_image_does_not_crash_or_nan(synthetic_project):
    """Degenerate input (zero dynamic range) is common in real data -- an
    all-fill band, a masked tile. It must render, not divide by zero."""
    img = np.full((8, 8, 3), 42.0, dtype=np.float32)
    out = np.asarray(synthetic_project._compute_stretched_preview(img, _StretchParams()))
    assert out.dtype == np.uint8
    assert np.all(np.isfinite(out.astype(np.float32))), "constant input produced non-finite output"


def test_nan_input_is_tolerated(synthetic_project):
    """NaN is one of the NoData conventions in this project's real rasters."""
    img = _gradient()
    img[0, 0, :] = np.nan
    out = np.asarray(synthetic_project._compute_stretched_preview(img, _StretchParams()))
    assert out.dtype == np.uint8
    assert np.all(np.isfinite(out.astype(np.float32)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_all_dtypes_render(synthetic_project, dtype):
    """uint8 / uint16 / float32 all reach this function in practice; the
    stretch must not assume a range tied to one of them."""
    img = (_gradient(hi=200.0)).astype(dtype)
    out = np.asarray(synthetic_project._compute_stretched_preview(img, _StretchParams()))
    assert out.dtype == np.uint8, f"{dtype.__name__} input did not render to uint8"
    assert out.shape[:2] == img.shape[:2]
