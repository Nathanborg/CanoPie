"""
QC regression test for the ImageEditorDialog opening "signal storm".

THE BUG THIS PINS
------------------
`_sync_ui_from_modifications()` set 8 checkboxes (NoData/resize/rotate/crop/
band/histogram/registration/classification "enabled" toggles) via bare
`.setChecked(...)`, with no `blockSignals()` -- unlike `hist_mode_combo`
right below them in the same method, which correctly blocks. Every one of
those 8 checkboxes is wired to a slot that calls `reapply_modifications()`
(or, for the classification checkbox, an equivalent refresh chain), which
runs the FULL image-processing pipeline (`apply_all_modifications_to_image`:
crop, rotate, histogram matching, resize, band expression). So loading a
saved `.ax` whose flags differ from the freshly-constructed widgets'
defaults fired up to 8 full pipeline runs -- and `init_ui()` called
`_sync_ui_from_modifications()` TWICE, unconditionally, doubling that again
to ~16 -- all before the dialog ever appears on screen.

Fixed by: wrapping the sync in blockSignals(True)/finally-blockSignals(False)
around all 8 checkboxes (see `_sync_ui_from_modifications`'s docstring), and
removing the duplicate call in `init_ui()`. After the fix, constructing the
dialog runs the pipeline at most once (from `showEvent()`'s own explicit
call, only when no fast-preview cache is available -- 0 times otherwise).

Verified against the ACTUAL current source before this fix was written (two
independent investigations plus a personal read of the code), not against
the multi-agent report that originally raised this -- that report also
claimed caller-side bottlenecks in project_tab.py (a full project-wide
polygon rebuild, synchronous disk flushes, a duplicated image copy) which
turned out to be overstated or false on inspection and are NOT addressed
here; see the plan file for that reasoning.
"""
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from canopie.image_editor_dialog import ImageEditorDialog

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor]


# Checkboxes whose freshly-constructed Qt default is `True` (see their
# QCheckBox() + .setChecked(True) construction in init_ui()). Setting the
# .ax to False for each guarantees .setChecked() during sync actually
# changes the widget's state, so stateChanged/toggled fires pre-fix.
_DEFAULT_TRUE_FLAGS = {
    "resize_enabled": False,
    "rotate_enabled": False,
    "crop_enabled": False,
    "nodata_enabled": False,
    "band_enabled": False,
    "hist_enabled": False,
}


def _write_ax(tmp_path, image_filepath, extra=None):
    ax = dict(_DEFAULT_TRUE_FLAGS)
    ax["nodata_values"] = []
    ax["band_expression"] = ""
    ax["hist_match"] = None
    if extra:
        ax.update(extra)
    ax_path = os.path.splitext(image_filepath)[0] + ".ax"
    with open(ax_path, "w", encoding="utf-8") as f:
        json.dump(ax, f)
    return ax_path


def _build_dialog_and_count(tmp_path, extra_ax=None):
    """Constructs a real ImageEditorDialog (parent=None, exactly like the
    existing test_image_editor_ml.py convention) against a .ax with several
    non-default flags, wrapping apply_all_modifications_to_image so its call
    count during construction + first show can be read afterward."""
    image_filepath = str(tmp_path / "editor_perf_test.tif")
    _write_ax(tmp_path, image_filepath, extra=extra_ax)

    artificial_image = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch.object(ImageEditorDialog, "apply_all_modifications_to_image",
                      autospec=True, side_effect=ImageEditorDialog.apply_all_modifications_to_image
                      ) as wrapped:
        dialog = ImageEditorDialog(parent=None, image_data=artificial_image,
                                   image_filepath=image_filepath)
        # Exercise the guarded first-show path too (showEvent's own
        # reapply_modifications() call, when no fast-preview cache exists).
        dialog.show()
        call_count_after_construct_and_show = wrapped.call_count

    return dialog, call_count_after_construct_and_show


def test_opening_editor_with_nondefault_ax_runs_pipeline_at_most_once(qapp, tmp_path):
    """THE regression pin. Pre-fix this was as high as ~14-16; post-fix it
    must be 0 or 1."""
    dialog, call_count = _build_dialog_and_count(tmp_path)
    dialog.close()

    assert call_count <= 1, (
        f"apply_all_modifications_to_image ran {call_count} time(s) opening "
        "the editor -- the checkbox signal storm during _sync_ui_from_modifications "
        "is back (or the duplicate sync call in init_ui() is back)")


def test_use_sklearn_checkbox_signal_does_not_fire_during_sync(qapp, tmp_path):
    """use_sklearn_checkbox specifically -- the 8th checkbox, missed by the
    original bug report, whose slot (_on_use_sklearn_toggled) does its own
    refresh chain (_update_append_button_state, _persist_classification_enabled,
    and -- when unchecking -- _clear_classification_preview) when toggled.

    Deliberately does NOT construct a dialog with classification actually
    enabled end-to-end: doing so exercises a real, separate, and much
    heavier mechanism (showEvent()'s classification-preview cache, which can
    load a real sklearn model bundle from disk) that has nothing to do with
    the checkbox-signal-storm bug this file exists to pin, and is out of
    scope for this fix. Instead this checks the ONE thing Fix 1a actually
    changed: that toggling this checkbox during
    `_sync_ui_from_modifications()` does not invoke its slot at all.
    """
    image_filepath = str(tmp_path / "editor_perf_test.tif")
    _write_ax(tmp_path, image_filepath, extra={"classification": {"enabled": True}})
    artificial_image = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch.object(ImageEditorDialog, "_on_use_sklearn_toggled",
                      autospec=True) as slot_mock:
        dialog = ImageEditorDialog(parent=None, image_data=artificial_image,
                                   image_filepath=image_filepath)
        call_count_after_construct = slot_mock.call_count
        dialog.close()

    assert call_count_after_construct == 0, (
        f"_on_use_sklearn_toggled ran {call_count_after_construct} time(s) "
        "during dialog construction -- use_sklearn_checkbox's signal is "
        "firing during _sync_ui_from_modifications()")


def test_checkbox_states_still_reflect_the_loaded_ax(qapp, tmp_path):
    """Blocking signals must not silently break the actual UI sync -- the
    correctness half of the fix, paired with the performance half above."""
    dialog, _ = _build_dialog_and_count(tmp_path)
    try:
        assert dialog.resize_enabled_checkbox.isChecked() is False
        assert dialog.rotate_enabled_checkbox.isChecked() is False
        assert dialog.crop_enabled_checkbox.isChecked() is False
        assert dialog.nodata_enabled_checkbox.isChecked() is False
        assert dialog.band_enabled_checkbox.isChecked() is False
        assert dialog.hist_enabled_checkbox.isChecked() is False
    finally:
        dialog.close()


def test_sync_ui_from_modifications_called_once_in_init_ui():
    """Source-level pin for the duplicate-call removal -- deterministic,
    unlike the call-count tests above which depend on which checkboxes
    happen to differ from their Qt-constructed defaults."""
    import inspect
    from canopie import image_editor_dialog as ied

    src = inspect.getsource(ied.ImageEditorDialog.init_ui)
    assert src.count("self._sync_ui_from_modifications()") == 1, (
        "init_ui() calls _sync_ui_from_modifications() more than once again")
