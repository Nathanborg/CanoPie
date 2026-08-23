"""QC regression tests for ExportImagesOptionsDialog and its wiring into
export_project_images.

Before this dialog existed, export_project_images asked two plain
QMessageBox questions (copy EXIF? / run in background?) and never offered a
format or band-order choice at all -- every non-jpg source always became a
classic TIFF, with the resulting band order decided implicitly by whichever
save branch happened to run (see test_export_images_band_order.py for the
real R/B swap bug that hid in that implicitness).

DEFAULTS MUST MATCH THE OLD PROMPTS' DEFAULTS, so opening the dialog and
immediately pressing Export changes nothing about format/EXIF/background
choices from before -- only "Keep original" band order is new, and it is
the fix on its own.
"""
import pytest

pytestmark = [pytest.mark.io]


def test_dialog_defaults_match_the_old_prompts(qapp):
    from ..export_images_options_dialog import ExportImagesOptionsDialog
    d = ExportImagesOptionsDialog(n_images=10)
    got = d.get_options()
    assert got == {
        'format': 'tif',        # classic TIFF, unchanged default
        'band_order': 'keep',   # the fix -- use each source's true order
        'copy_exif': False,     # old QMessageBox default answer was No
        'background': True,     # old QMessageBox default answer was Yes
    }


def test_dialog_offers_all_three_formats(qapp):
    from ..export_images_options_dialog import ExportImagesOptionsDialog
    d = ExportImagesOptionsDialog()
    values = [d.format_combo.itemData(i) for i in range(d.format_combo.count())]
    assert set(values) == {"tif", "jpg", "png"}


def test_dialog_offers_keep_rgb_bgr_band_order(qapp):
    from ..export_images_options_dialog import ExportImagesOptionsDialog
    d = ExportImagesOptionsDialog()
    values = [d.band_order_combo.itemData(i) for i in range(d.band_order_combo.count())]
    assert values == ["keep", "rgb", "bgr"]
    assert d.band_order_combo.currentData() == "keep"


def test_dialog_get_options_reflects_every_control(qapp):
    from ..export_images_options_dialog import ExportImagesOptionsDialog
    d = ExportImagesOptionsDialog()
    d.format_combo.setCurrentIndex(2)       # png
    d.band_order_combo.setCurrentIndex(1)   # rgb
    d.copy_exif_cb.setChecked(True)
    d.background_cb.setChecked(False)
    assert d.get_options() == {
        'format': 'png',
        'band_order': 'rgb',
        'copy_exif': True,
        'background': False,
    }


def test_ok_button_is_relabeled_export():
    """Same CanoPie convention as the thumbnail dialog relabeling OK ->
    Generate -- here it's Export, so the button describes the action."""
    from PyQt5.QtWidgets import QDialogButtonBox
    from .test_export_and_ax_regressions import _names_in
    from ..export_images_options_dialog import ExportImagesOptionsDialog
    names = _names_in(ExportImagesOptionsDialog.setup_ui)
    assert "Export" in names


# ---------------------------------------------------------------------------
# Wiring: export_project_images uses the dialog instead of two QMessageBoxes
# ---------------------------------------------------------------------------
def test_export_project_images_uses_the_new_dialog_not_two_messageboxes():
    from .test_export_and_ax_regressions import _names_in, _calls_in
    from ..project_tab import ProjectTab
    names = _names_in(ProjectTab.export_project_images)
    assert "ExportImagesOptionsDialog" in names
    assert _calls_in(ProjectTab.export_project_images, "get_options")


def test_export_project_images_threads_format_and_band_order_to_the_worker():
    from .test_export_and_ax_regressions import _names_in
    from ..project_tab import ProjectTab
    names = _names_in(ProjectTab.export_project_images)
    assert "export_format" in names
    assert "band_order" in names
    assert "run_background" in names
