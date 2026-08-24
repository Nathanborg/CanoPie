"""QC tests for `canopie/ml_augmentation.py` -- the training-time brightness
and shadow/illumination augmentation engine.

Shadow/illumination reimplements `uneven_illumination_rgb()` /
`generate_noisy_images()` from
https://github.com/Nathanborg/Cloud_shadow_correction (random
linear/circular/diagonal gradient mask, Gaussian-blurred, applied
multiplicatively per channel). Brightness reuses the exact linear-rescale
formula `ProjectTab._apply_hist_match`'s meanstd mode uses
(project_tab.py:4698-4702), duplicated locally since the .ax/hist-match
reference-provenance machinery doesn't apply to a randomly-perturbed target.
"""
import cv2
import numpy as np
import pytest

from ..ml_augmentation import (
    apply_brightness_image_level,
    apply_brightness_patch_level,
    apply_shadow_illumination,
    augment_image_for_training,
    _generate_illumination_mask,
    _linear_rescale,
)

pytestmark = [pytest.mark.ml]


def _band_mean_std(arr, band=0):
    a = arr if arr.ndim == 3 else arr[..., None]
    ch = a[..., band].astype(np.float64)
    return float(ch.mean()), float(ch.std())


# ---------------------------------------------------------------------------
# _linear_rescale (private, but the shared core of everything else here)
# ---------------------------------------------------------------------------
def test_linear_rescale_lands_on_the_requested_target_stats():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=50.0, scale=10.0, size=(64, 64, 1)).astype(np.float32)
    mu_s, sd_s = arr.mean(), arr.std()
    out = _linear_rescale(arr, [mu_s], [sd_s], [200.0], [5.0])
    assert abs(out.mean() - 200.0) < 1.0
    assert abs(out.std() - 5.0) < 1.0


def test_linear_rescale_restores_keep_mask_positions_exactly():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    keep = np.zeros((4, 4), dtype=bool)
    keep[1, 1] = True
    out = _linear_rescale(arr, [8.0], [4.0], [0.0], [1.0], keep_mask=keep)
    assert out[1, 1, 0] == arr[1, 1, 0], "a keep_mask position must be restored to its original value"
    # And a non-masked position must actually have moved (not a vacuous pass):
    assert out[0, 0, 0] != arr[0, 0, 0]


def test_linear_rescale_never_mutates_input():
    arr = np.ones((4, 4, 2), dtype=np.float32) * 5.0
    original = arr.copy()
    _linear_rescale(arr, [5.0, 5.0], [1.0, 1.0], [50.0, 50.0], [2.0, 2.0])
    np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# apply_brightness_image_level
# ---------------------------------------------------------------------------
def test_image_level_brightness_lands_within_the_configured_jitter_range():
    rng = np.random.default_rng(1)
    img = rng.normal(loc=100.0, scale=20.0, size=(80, 80, 3)).astype(np.float32)
    mu_s = img[..., 0].mean()
    sd_s = img[..., 0].std()

    out = apply_brightness_image_level(img, np.random.default_rng(2), mu_jitter=0.15, sd_jitter=0.25)
    new_mu, new_sd = _band_mean_std(out, 0)

    assert (1 - 0.15) * mu_s - 2.0 <= new_mu <= (1 + 0.15) * mu_s + 2.0
    assert (1 - 0.25) * sd_s - 2.0 <= new_sd <= (1 + 0.25) * sd_s + 2.0


def test_image_level_brightness_leaves_nodata_pixels_byte_identical():
    rng = np.random.default_rng(3)
    img = rng.normal(loc=100.0, scale=20.0, size=(40, 40, 3)).astype(np.float32)
    nodata_mask = np.zeros((40, 40), dtype=bool)
    nodata_mask[:10, :10] = True
    original_patch = img[:10, :10, :].copy()

    out = apply_brightness_image_level(img, np.random.default_rng(4), nodata_mask=nodata_mask)
    np.testing.assert_array_equal(out[:10, :10, :], original_patch)


def test_image_level_brightness_never_mutates_input():
    rng = np.random.default_rng(5)
    img = rng.normal(size=(20, 20, 3)).astype(np.float32)
    original = img.copy()
    apply_brightness_image_level(img, np.random.default_rng(6))
    np.testing.assert_array_equal(img, original)


def test_image_level_brightness_accepts_2d_single_band_images():
    rng = np.random.default_rng(7)
    img = rng.normal(loc=50, scale=5, size=(30, 30)).astype(np.float32)
    out = apply_brightness_image_level(img, np.random.default_rng(8))
    assert out.shape == img.shape
    assert out.ndim == 2


# ---------------------------------------------------------------------------
# apply_brightness_patch_level
# ---------------------------------------------------------------------------
def test_patch_level_gives_different_tiles_different_perturbations():
    # A uniform-valued source image: every tile starts with IDENTICAL stats,
    # so any difference in the OUTPUT between tiles must come from the
    # independent per-tile random draw, not from different input content.
    img = np.full((64, 64, 3), 100.0, dtype=np.float32)
    out = apply_brightness_patch_level(img, np.random.default_rng(9), tile_size=16)

    tile_a = out[0:16, 0:16, 0]
    tile_b = out[16:32, 16:32, 0]
    assert not np.allclose(tile_a.mean(), tile_b.mean(), atol=0.5), (
        "two different tiles of a uniform source image landed on the same "
        "mean -- patch-level augmentation is not perturbing tiles independently")


def test_patch_level_tile_matches_image_level_on_the_same_subregion():
    """Proves patch-level reuses the SAME core math as image-level (not a
    reimplementation): isolating one tile and running it through
    apply_brightness_image_level with a freshly-seeded rng in the same state
    must reproduce exactly what patch-level did to that tile."""
    rng_seed = 42
    img = np.random.default_rng(0).normal(loc=80, scale=10, size=(32, 16, 2)).astype(np.float32)

    patched = apply_brightness_patch_level(img, np.random.default_rng(rng_seed), tile_size=32)
    tile = img[0:32, 0:16, :]
    standalone = apply_brightness_image_level(tile, np.random.default_rng(rng_seed))

    np.testing.assert_allclose(patched, standalone, rtol=1e-5)


def test_patch_level_never_mutates_input():
    img = np.random.default_rng(10).normal(size=(50, 50, 3)).astype(np.float32)
    original = img.copy()
    apply_brightness_patch_level(img, np.random.default_rng(11), tile_size=20)
    np.testing.assert_array_equal(img, original)


def test_patch_level_cell_with_no_eligible_pixels_is_left_unperturbed():
    img = np.random.default_rng(12).normal(loc=50, size=(32, 32, 1)).astype(np.float32)
    nodata_mask = np.ones((32, 32), dtype=bool)  # everything excluded
    out = apply_brightness_patch_level(img, np.random.default_rng(13), tile_size=8, nodata_mask=nodata_mask)
    np.testing.assert_array_equal(out, img)


def test_patch_level_handles_a_grid_that_does_not_evenly_divide_the_image():
    img = np.random.default_rng(14).normal(size=(50, 37, 3)).astype(np.float32)
    out = apply_brightness_patch_level(img, np.random.default_rng(15), tile_size=16)
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# _generate_illumination_mask / apply_shadow_illumination
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gradient_type", ["linear", "circular", "diagonal"])
def test_illumination_mask_stays_within_the_configured_range(gradient_type):
    rng = np.random.default_rng(20)
    mask = _generate_illumination_mask(64, 96, rng, gradient_type, min_illum=0.2, max_illum=0.9)
    assert mask.shape == (64, 96)
    assert mask.min() >= 0.2 - 1e-5
    assert mask.max() <= 0.9 + 1e-5


def test_illumination_mask_unknown_gradient_type_raises():
    with pytest.raises(ValueError):
        _generate_illumination_mask(8, 8, np.random.default_rng(0), "spiral", 0.1, 0.9)


def test_shadow_illumination_forces_an_odd_gaussian_kernel_even_when_smoothness_is_even():
    """cv2.GaussianBlur requires an odd kernel -- a dialog spinbox could pass
    an even 'smoothness' value; this must not crash."""
    img = np.random.default_rng(21).uniform(50, 200, size=(40, 40, 3)).astype(np.float32)
    out = apply_shadow_illumination(img, np.random.default_rng(22), smoothness=74)  # even
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))


def test_shadow_illumination_darkens_or_brightens_but_stays_finite_and_nonnegative():
    img = np.random.default_rng(23).uniform(50, 200, size=(50, 50, 3)).astype(np.float32)
    out = apply_shadow_illumination(img, np.random.default_rng(24), smoothness=15)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0, "multiplying by an illumination mask in [0,1]-ish range must not go negative"
    assert not np.allclose(out, img), "shadow augmentation had no visible effect"


def test_shadow_illumination_leaves_nodata_pixels_byte_identical():
    img = np.random.default_rng(25).uniform(50, 200, size=(30, 30, 3)).astype(np.float32)
    nodata_mask = np.zeros((30, 30), dtype=bool)
    nodata_mask[5:15, 5:15] = True
    original_patch = img[5:15, 5:15, :].copy()
    out = apply_shadow_illumination(img, np.random.default_rng(26), nodata_mask=nodata_mask)
    np.testing.assert_array_equal(out[5:15, 5:15, :], original_patch)


def test_shadow_illumination_never_mutates_input():
    img = np.random.default_rng(27).uniform(50, 200, size=(20, 20, 3)).astype(np.float32)
    original = img.copy()
    apply_shadow_illumination(img, np.random.default_rng(28))
    np.testing.assert_array_equal(img, original)


def test_shadow_illumination_accepts_2d_single_band_images():
    img = np.random.default_rng(29).uniform(50, 200, size=(25, 25)).astype(np.float32)
    out = apply_shadow_illumination(img, np.random.default_rng(30))
    assert out.shape == img.shape
    assert out.ndim == 2


# ---------------------------------------------------------------------------
# augment_image_for_training -- the orchestrator
# ---------------------------------------------------------------------------
def test_orchestrator_mode_none_and_shadow_disabled_returns_unchanged_image():
    img = np.random.default_rng(40).uniform(0, 255, size=(20, 20, 3)).astype(np.float32)
    cfg = {"brightness": {"mode": "none"}, "shadow": {"enabled": False}}
    out = augment_image_for_training(img, cfg, np.random.default_rng(41))
    np.testing.assert_array_equal(out, img)


def test_orchestrator_applies_brightness_then_shadow_in_that_order():
    """Composing both stages manually (brightness first, then shadow, with
    fresh rng draws matching what the orchestrator would consume in order)
    must reproduce the orchestrator's output exactly."""
    img = np.random.default_rng(42).uniform(50, 200, size=(40, 40, 3)).astype(np.float32)
    cfg = {
        "brightness": {"mode": "image", "mu_jitter": 0.1, "sd_jitter": 0.1},
        "shadow": {"enabled": True, "smoothness": 21},
    }
    rng_seed = 99
    combined = augment_image_for_training(img, cfg, np.random.default_rng(rng_seed))

    rng2 = np.random.default_rng(rng_seed)
    stepwise = apply_brightness_image_level(img, rng2, mu_jitter=0.1, sd_jitter=0.1)
    stepwise = apply_shadow_illumination(stepwise, rng2, smoothness=21)

    np.testing.assert_allclose(combined, stepwise, rtol=1e-5)


def test_orchestrator_patch_mode_dispatches_to_patch_level():
    img = np.full((32, 32, 1), 100.0, dtype=np.float32)
    cfg = {"brightness": {"mode": "patch", "tile_size": 8}, "shadow": {"enabled": False}}
    out = augment_image_for_training(img, cfg, np.random.default_rng(50))
    tile_a = out[0:8, 0:8, 0]
    tile_b = out[8:16, 8:16, 0]
    assert not np.allclose(tile_a.mean(), tile_b.mean(), atol=0.5)


def test_orchestrator_unknown_brightness_mode_raises():
    img = np.zeros((5, 5, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        augment_image_for_training(img, {"brightness": {"mode": "bogus"}}, np.random.default_rng(0))


def test_orchestrator_never_mutates_input():
    img = np.random.default_rng(51).uniform(0, 255, size=(20, 20, 3)).astype(np.float32)
    original = img.copy()
    cfg = {"brightness": {"mode": "image"}, "shadow": {"enabled": True}}
    augment_image_for_training(img, cfg, np.random.default_rng(52))
    np.testing.assert_array_equal(img, original)
