"""QC tests for `MLAugmentationOptionsDialog` (canopie/ml_augmentation_options_dialog.py).

Mirrors this codebase's established options-dialog convention/tests (see
test_thumbnail_options.py): defaults must reproduce "no augmentation, no
normalization" exactly -- accepting the dialog without touching anything must
be a behavioral no-op for train_models, matching this session's convention
that a new options dialog never silently changes existing behavior.
"""
import pytest

from ..ml_augmentation_options_dialog import (
    MLAugmentationOptionsDialog,
    DEFAULT_BRIGHTNESS_MODE, DEFAULT_TILE_SIZE, DEFAULT_MU_JITTER,
    DEFAULT_SD_JITTER, DEFAULT_SHADOW_SMOOTHNESS, DEFAULT_ROW_POLICY,
    DEFAULT_N_VARIANTS, DEFAULT_NORMALIZATION,
)

pytestmark = [pytest.mark.ml]


def _dlg(entries=None):
    return MLAugmentationOptionsDialog(
        entries or [], lambda fp: (None, 0), lambda img: [])


def test_defaults_are_a_behavioral_no_op(qapp):
    d = _dlg()
    opts = d.get_options()
    assert opts["augmentation"]["enabled"] is False
    assert opts["augmentation"]["brightness"]["mode"] == DEFAULT_BRIGHTNESS_MODE == "none"
    assert opts["augmentation"]["brightness"]["tile_size"] == DEFAULT_TILE_SIZE
    assert opts["augmentation"]["brightness"]["mu_jitter"] == DEFAULT_MU_JITTER
    assert opts["augmentation"]["brightness"]["sd_jitter"] == DEFAULT_SD_JITTER
    assert opts["augmentation"]["shadow"]["enabled"] is False
    assert opts["augmentation"]["shadow"]["smoothness"] == DEFAULT_SHADOW_SMOOTHNESS
    assert opts["augmentation"]["row_policy"] == DEFAULT_ROW_POLICY == "add"
    assert opts["augmentation"]["n_variants"] == DEFAULT_N_VARIANTS
    assert opts["normalization"] == DEFAULT_NORMALIZATION == "none"


def test_enabling_brightness_only_sets_augmentation_enabled(qapp):
    d = _dlg()
    d.brightness_enabled_cb.setChecked(True)
    opts = d.get_options()
    assert opts["augmentation"]["enabled"] is True
    assert opts["augmentation"]["brightness"]["mode"] == "image"
    assert opts["augmentation"]["shadow"]["enabled"] is False


def test_patch_level_radio_selects_patch_mode(qapp):
    d = _dlg()
    d.brightness_enabled_cb.setChecked(True)
    d.patch_level_rb.setChecked(True)
    opts = d.get_options()
    assert opts["augmentation"]["brightness"]["mode"] == "patch"


def test_enabling_shadow_only_sets_augmentation_enabled(qapp):
    d = _dlg()
    d.shadow_enabled_cb.setChecked(True)
    opts = d.get_options()
    assert opts["augmentation"]["enabled"] is True
    assert opts["augmentation"]["shadow"]["enabled"] is True
    assert opts["augmentation"]["brightness"]["mode"] == "none"


def test_row_policy_replace_is_selectable(qapp):
    d = _dlg()
    d.replace_rb.setChecked(True)
    opts = d.get_options()
    assert opts["augmentation"]["row_policy"] == "replace"


@pytest.mark.parametrize("rb_attr,expected", [
    ("norm_none_rb", "none"), ("norm_l2_rb", "l2"), ("norm_zscore_rb", "zscore")])
def test_normalization_radios_are_mutually_exclusive(qapp, rb_attr, expected):
    d = _dlg()
    getattr(d, rb_attr).setChecked(True)
    assert d.get_options()["normalization"] == expected


def test_tile_size_control_disabled_until_patch_level_and_brightness_are_on(qapp):
    d = _dlg()
    assert not d.tile_size_spin.isEnabled()
    d.brightness_enabled_cb.setChecked(True)
    assert not d.tile_size_spin.isEnabled(), "image-level is the default -- tile size stays disabled"
    d.patch_level_rb.setChecked(True)
    assert d.tile_size_spin.isEnabled()


def test_n_variants_control_disabled_when_replace_mode_selected(qapp):
    d = _dlg()
    assert d.n_variants_spin.isEnabled()  # "add" is the default row policy
    d.replace_rb.setChecked(True)
    assert not d.n_variants_spin.isEnabled()


def test_preview_combo_is_populated_from_entries(qapp):
    entries = [("groupA", "C:/imgs/a.tif"), ("groupB", "C:/imgs/b.tif")]
    d = _dlg(entries)
    assert d.preview_combo.count() == 2
    assert d.preview_combo.itemData(0) == "C:/imgs/a.tif"
    assert d.preview_combo.itemData(1) == "C:/imgs/b.tif"


def test_no_entries_does_not_crash_construction(qapp):
    d = _dlg([])
    assert d.preview_combo.count() == 0
