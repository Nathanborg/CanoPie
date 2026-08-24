"""QC regression test for "hist match mean and std is not working whatsoever
-- after applying to the viewer, scene mean and std keep the same."

Every existing histogram-matching test (test_hist_match.py) drives
`ProjectTab._apply_hist_match` against a HAND-WRITTEN `.ax` `hist_match`
block. That proves the replay MATH is correct, but it never drives the
actual UI path a user takes for cross-image matching:

    1. Open reference image A in the Image Editor.
    2. Set Mean/Std mode, click "Calc" (`on_hist_match_clicked`) -- this
       computes `ref_stats` FROM A's own pixels.
    3. Check "Apply modifications to all roots" (`apply_all_groups_checkbox`)
       so the SAME reference gets broadcast to every other image.
    4. Click "Apply All Changes" (`save_modifications_to_file`).
    5. Expect OTHER image B -- never opened in the editor -- to now have
       `hist_match` in its own `.ax`, and to visibly shift toward A's
       mean/std whenever it is viewed or exported.

That whole chain -- on_hist_match_clicked -> save_modifications_to_file's
scope="all"/"group" broadcast loop -> B's .ax on disk -> ProjectTab replay
for B -- had ONLY ever been checked at the AST level
(test_image_editor_apply_and_band_expr.py's A3 cluster asserts on SOURCE
TEXT, e.g. "_PER_IMAGE_AX_KEYS in src"; it never actually runs the broadcast
against two real files and reads back the result). This is the first test
to run it end to end with real pixels.

A and B are built with every channel equal to the same per-pixel value (a
flat/grayscale-like RGB image), so the well-known, ALREADY-covered RGB<->BGR
channel-order question (cv2 vs tifffile) cannot mask or fake a pass here --
whichever order either loader picks, band i's value is identical to band 0's
in both A and B.
"""
import json

import numpy as np
import pytest
import tifffile

from .project_builder import build_project_tab
from ..image_editor_dialog import ImageEditorDialog
from ..project_tab import ProjectTab

pytestmark = [pytest.mark.editor, pytest.mark.extraction]


def _flat_rgb(h, w, mean, std, seed):
    """An HxWx3 uint8 image with every channel equal to the same per-pixel
    value, so RGB/BGR reordering cannot change any band's statistics."""
    rng = np.random.default_rng(seed)
    plane = rng.normal(mean, std, size=(h, w)).clip(0, 255).astype(np.uint8)
    return np.stack([plane, plane, plane], axis=-1)


@pytest.fixture
def reference_and_target(tmp_path):
    """Two real images, each its OWN root (the realistic drone-photo layout:
    build_project_data.py gives every fixture a single-file root too) -- A
    bright (mean ~200) as the hist-match reference, B dark (mean ~40) as the
    untouched target that should shift toward A after 'Apply to all roots'.
    """
    arr_a = _flat_rgb(48, 48, mean=200.0, std=15.0, seed=1)
    arr_b = _flat_rgb(48, 48, mean=40.0, std=10.0, seed=2)

    fp_a = str(tmp_path / "images" / "reference_A.tif")
    fp_b = str(tmp_path / "images" / "target_B.tif")
    import os
    os.makedirs(tmp_path / "images", exist_ok=True)
    tifffile.imwrite(fp_a, arr_a)
    tifffile.imwrite(fp_b, arr_b)

    project_folder = tmp_path / "project"
    pt = build_project_tab(str(project_folder))
    pt.multispectral_image_data_groups["reference_A"] = [fp_a]
    pt.multispectral_image_data_groups["target_B"] = [fp_b]
    pt.multispectral_root_names = list(pt.multispectral_root_names) + [
        "reference_A", "target_B"]

    return pt, arr_a, arr_b, fp_a, fp_b


def _raw_band0_mean_std(arr):
    band0 = arr[..., 0].astype(np.float64)
    return float(band0.mean()), float(band0.std())


def test_apply_to_all_roots_writes_hist_match_into_the_OTHER_images_ax(
        reference_and_target, qapp):
    """The .ax on disk for B (never opened in the editor) must gain the SAME
    hist_match block A's own Calc computed, once 'Apply to all roots' runs."""
    pt, arr_a, arr_b, fp_a, fp_b = reference_and_target

    dlg = ImageEditorDialog(pt, image_data=arr_a, image_filepath=fp_a)
    idx = dlg.hist_mode_combo.findText("Mean/Std")
    assert idx >= 0, "Mean/Std is no longer an option in hist_mode_combo"
    dlg.hist_mode_combo.setCurrentIndex(idx)

    dlg.on_hist_match_clicked()

    hm = dlg.modifications.get("hist_match")
    assert isinstance(hm, dict) and hm.get("mode") == "meanstd", (
        "Calc did not store a meanstd hist_match block on the reference image")
    ref_stats = hm.get("ref_stats")
    assert ref_stats and len(ref_stats) >= 1

    # Sanity: the captured reference actually describes A, not some default.
    a_mean, a_std = _raw_band0_mean_std(arr_a)
    assert abs(ref_stats[0]["mean"] - a_mean) < 3.0, (
        f"captured ref mean {ref_stats[0]['mean']} does not match A's own "
        f"pixels (~{a_mean})")

    dlg.apply_all_groups_checkbox.setChecked(True)
    hint = dlg.save_modifications_to_file()
    assert hint == {"scope": "all", "root_name": None}, hint

    ax_path_b = pt._ax_path_for(fp_b)
    with open(ax_path_b, "r", encoding="utf-8") as f:
        ax_b = json.load(f) or {}

    assert "hist_match" in ax_b, (
        "'Apply to all roots' did not write a hist_match block into B's own "
        f".ax ({ax_path_b}) -- B was never told to match anything, which is "
        "exactly 'hist match ... not working whatsoever' for any image "
        "other than the one Calc was clicked on")
    assert ax_b["hist_match"].get("mode") == "meanstd"
    assert ax_b["hist_match"].get("ref_stats") == ref_stats, (
        "B's .ax hist_match block does not carry the SAME ref_stats A's "
        "Calc computed -- the reference was altered or dropped in transit")
    assert ax_b.get("hist_enabled", True) is True, (
        "B's .ax has hist_enabled=False, so replay will skip the match "
        "even though the block is present")


@pytest.mark.parametrize("replay", ["export", "viewer"])
def test_apply_to_all_roots_actually_shifts_the_OTHER_images_pixels(
        reference_and_target, qapp, replay):
    """Behavioural: after the broadcast, B's OWN rendered/exported pixels --
    not just its .ax text -- must land near A's mean/std, and must have
    genuinely moved from B's original (~40) statistics. This is the guard
    against a vacuous pass: a hist_match block that is written but silently
    skipped at replay time would leave B's stats exactly where they started,
    reproducing the user's report even though the .ax "looks" correct."""
    pt, arr_a, arr_b, fp_a, fp_b = reference_and_target

    dlg = ImageEditorDialog(pt, image_data=arr_a, image_filepath=fp_a)
    idx = dlg.hist_mode_combo.findText("Mean/Std")
    dlg.hist_mode_combo.setCurrentIndex(idx)
    dlg.on_hist_match_clicked()
    ref_stats = dlg.modifications["hist_match"]["ref_stats"]

    dlg.apply_all_groups_checkbox.setChecked(True)
    dlg.save_modifications_to_file()

    b_raw_mean, _b_raw_std = _raw_band0_mean_std(arr_b)

    if replay == "export":
        img, _C = pt._get_export_image(fp_b)
        chans = pt._channels_in_export_order(img)
        band0 = np.asarray(chans[0]).astype(np.float64)
    else:
        lite = pt._imagedata_or_fallback(fp_b)
        lite.image = pt.__class__.apply_aux_modifications(
            fp_b, lite.image, pt.project_folder, global_mode=False)
        arr = np.asarray(lite.image).astype(np.float64)
        band0 = arr[..., 0] if arr.ndim == 3 else arr

    new_mean = float(band0.mean())
    new_std = float(band0.std())

    assert abs(new_mean - b_raw_mean) > 20.0, (
        f"{replay}: B's band-0 mean is still {new_mean:.1f}, essentially "
        f"unchanged from its original {b_raw_mean:.1f} -- histogram "
        "matching from the reference image had NO effect on this other "
        "image, matching the reported bug exactly")
    assert abs(new_mean - ref_stats[0]["mean"]) < 5.0, (
        f"{replay}: B's band-0 mean landed at {new_mean:.1f}, expected "
        f"close to the reference's {ref_stats[0]['mean']:.1f}")
    assert abs(new_std - ref_stats[0]["std"]) < 5.0, (
        f"{replay}: B's band-0 std landed at {new_std:.1f}, expected close "
        f"to the reference's {ref_stats[0]['std']:.1f}")


# ---------------------------------------------------------------------------
# THE ACTUAL ROOT CAUSE FOUND AGAINST A REAL PROJECT (C:\New Folder215):
# every image's .ax carried a stale 'cdf' reference from an EARLIER Calc,
# even though the reporter's most recent work was on Mean/Std -- because
# switching hist_mode_combo alone never recomputes or clears the stored
# reference; only clicking "Calc" does. save_modifications_to_file then
# broadcasts whatever Calc last stored, silently, regardless of what the
# dropdown currently shows. These tests pin the fix: a live warning label
# plus a hard confirm-or-cancel guard before any group/all broadcast.
# ---------------------------------------------------------------------------
def _dlg_with_stale_cdf_reference(reference_and_target):
    """A' has clicked Calc once on CDF, then (WITHOUT clicking Calc again)
    switched the dropdown to Mean/Std -- the exact sequence that leaves a
    stale reference stored while the dropdown shows something else."""
    pt, arr_a, arr_b, fp_a, fp_b = reference_and_target
    dlg = ImageEditorDialog(pt, image_data=arr_a, image_filepath=fp_a)

    idx_cdf = dlg.hist_mode_combo.findText("CDF")
    dlg.hist_mode_combo.setCurrentIndex(idx_cdf)
    dlg.on_hist_match_clicked()
    assert dlg.modifications["hist_match"]["mode"] == "cdf"

    idx_meanstd = dlg.hist_mode_combo.findText("Mean/Std")
    dlg.hist_mode_combo.setCurrentIndex(idx_meanstd)  # NOT followed by Calc
    return dlg, pt, fp_b


def test_combo_hist_mode_matches_on_hist_match_clicked_mapping(
        reference_and_target, qapp):
    """_combo_hist_mode is the single source of truth on_hist_match_clicked
    now delegates to -- pin the text->mode mapping itself."""
    pt, arr_a, _arr_b, fp_a, _fp_b = reference_and_target
    dlg = ImageEditorDialog(pt, image_data=arr_a, image_filepath=fp_a)

    for label, expected in (("None", "none"), ("Mean/Std", "meanstd"), ("CDF", "cdf")):
        dlg.hist_mode_combo.setCurrentIndex(dlg.hist_mode_combo.findText(label))
        assert dlg._combo_hist_mode() == expected


def test_base_label_warns_when_dropdown_disagrees_with_stored_reference(
        reference_and_target, qapp):
    dlg, pt, fp_b = _dlg_with_stale_cdf_reference(reference_and_target)

    txt = dlg.hist_base_label.text()
    assert "cdf" in txt.lower() and "mean/std" in txt.lower(), (
        f"label does not name both the stale stored mode and the dropdown's "
        f"current mode: {txt!r}")
    assert "\u26a0" in txt or "stale" in txt.lower() or "click calc" in txt.lower(), (
        f"label does not read as a warning: {txt!r}")


def test_base_label_is_quiet_once_calc_resyncs_it(reference_and_target, qapp):
    """Clicking Calc again (on the now-selected mode) must clear the warning --
    the label should not stay red forever once the user does the right thing."""
    dlg, pt, fp_b = _dlg_with_stale_cdf_reference(reference_and_target)
    dlg.on_hist_match_clicked()  # now computes meanstd, matching the dropdown
    txt = dlg.hist_base_label.text()
    assert "\u26a0" not in txt, f"label still warns after Calc resynced it: {txt!r}"
    assert dlg.modifications["hist_match"]["mode"] == "meanstd"


def test_apply_to_group_is_blocked_when_user_declines_the_stale_reference(
        reference_and_target, qapp, monkeypatch):
    """The guard must stop the broadcast (and must not close the dialog) when
    the user answers Cancel -- otherwise the stale CDF reference still ships
    to every other file exactly as it did on the real project."""
    from PyQt5 import QtWidgets
    dlg, pt, fp_b = _dlg_with_stale_cdf_reference(reference_and_target)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: QtWidgets.QMessageBox.Cancel)

    dlg.apply_all_groups_checkbox.setChecked(True)
    result = dlg.save_modifications_to_file()
    assert result is None, "declining the mismatch warning must abort the save"

    ax_path_b = pt._ax_path_for(fp_b)
    assert not __import__("os").path.exists(ax_path_b), (
        "B's .ax was written even though the user declined the stale-"
        "reference warning")

    accepted = {"called": False}
    monkeypatch.setattr(dlg, "accept", lambda: accepted.__setitem__("called", True))
    dlg.apply_all_changes()
    assert not accepted["called"], (
        "apply_all_changes() closed the editor even though nothing was "
        "saved -- the user's chance to click Calc and retry was discarded")


def test_apply_to_group_proceeds_when_user_accepts_the_stale_reference(
        reference_and_target, qapp, monkeypatch):
    """Answering Yes is a deliberate override and must still work -- the
    guard warns, it does not silently block forever."""
    from PyQt5 import QtWidgets
    dlg, pt, fp_b = _dlg_with_stale_cdf_reference(reference_and_target)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: QtWidgets.QMessageBox.Yes)

    dlg.apply_all_groups_checkbox.setChecked(True)
    result = dlg.save_modifications_to_file()
    assert result == {"scope": "all", "root_name": None}

    ax_path_b = pt._ax_path_for(fp_b)
    with open(ax_path_b, "r", encoding="utf-8") as f:
        ax_b = json.load(f)
    assert ax_b["hist_match"]["mode"] == "cdf", (
        "answering Yes should broadcast the OLD stored reference (cdf), "
        "exactly as the user explicitly confirmed")


def test_apply_to_single_scope_is_not_blocked_by_the_guard(
        reference_and_target, qapp, monkeypatch):
    """The confirm dialog only guards a BROADCAST (scope all/group); a
    single-file save must never be interrupted by it -- the live warning
    label already covers that case without a blocking prompt."""
    from PyQt5 import QtWidgets
    dlg, pt, fp_b = _dlg_with_stale_cdf_reference(reference_and_target)

    def _fail(*a, **k):
        raise AssertionError("QMessageBox.warning must not be shown for scope='single'")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)

    # Neither apply_all_groups_checkbox nor global_mods_checkbox is checked.
    result = dlg.save_modifications_to_file()
    assert result is not None
