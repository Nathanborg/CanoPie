"""
BUG-060 (2026-10-09): a pixel resize (`px_w` / `px_h`, pixels of the FULL
frame -- what the export applies) on a DECIMATED preview.

  * the viewer and the editor resized the preview to the absolute pixel size,
    i.e. blew it back up to the full export size: New Folder213's 3.3 m mosaic
    preview (2438 x 2186) became 19499 x 17481 float -- 1.36 GB plus a 341 Mpx
    pixmap per band shown -- defeating the 16 Mpx preview cap;
  * that .ax was written by the editor itself: opened on a preview and saved
    with no edit, it compared the full-frame size its boxes showed with the
    PREVIEW's natural size and stored `px 19499 x 17481, ref 2438 x 2186`
    (reproduced on a copy); after an edit its boxes showed the preview's size,
    so a typed size was 8-16x too small at export.

Now: on a preview the export's resize FACTOR applies; the boxes and the save
work in full-frame pixels; a full image keeps the exact pixel size.
"""
import json
import os

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer]

NAME = "rgb_16bit_tiled_bip_cog"          # 96 x 96, tiled: previews read windows


def _clear(pt):
    for attr in ("_ax_cache", "_imgdata_cache", "_pixmap_cache", "_raw_cache", "_raw_image_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


@pytest.fixture
def preview_ax(synthetic_project, monkeypatch):
    """The fixture shown as a 48 x 48 preview, with a temporary .ax."""
    from .. import project_tab as PT
    from ..raster_reader import clear_reader_cache
    pt = synthetic_project
    fp = fixture_image_path(NAME)
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: fixture has no .ax"
    monkeypatch.setattr(PT, "_PREVIEW_MAX_PIXELS", 48 * 48)

    def _write(ax):
        json.dump(ax, open(axp, "w", encoding="utf-8"))
        _clear(pt)
        clear_reader_cache()
        return fp

    yield _write
    if os.path.exists(axp):
        os.remove(axp)
    _clear(pt)


def test_the_helper_applies_the_factor_on_a_preview_only():
    from ..ax_geometry import preview_px_resize
    ax = {"resize_enabled": True, "resize": {"px_w": 48, "px_h": 48}}
    assert preview_px_resize(ax, 96, 96, 96, 96) is None, "the full frame keeps the pixel size"
    assert preview_px_resize(ax, 96, 96, 48, 48) == (24, 24)
    assert preview_px_resize({"resize": {"width": 50, "height": 50}}, 96, 96, 48, 48) is None
    crop = dict(ax, crop_rect={"x": 0, "y": 0, "width": 64, "height": 32}, crop_rect_ref_size={"w": 96, "h": 96})
    assert preview_px_resize(crop, 96, 96, 32, 16) == (24, 24)          # factor 48/64, 48/32


def test_the_viewer_applies_a_pixel_resize_as_a_factor_on_a_preview(synthetic_project, preview_ax):
    pt = synthetic_project
    fp = preview_ax({"resize_enabled": True, "resize": {"px_w": 48, "px_h": 48}})
    lite = pt._imagedata_or_fallback(fp)
    assert lite.image.shape[:2] == (48, 48), f"precondition: a 48 px preview, got {lite.image.shape}"
    out = pt.__class__.apply_aux_modifications(fp, lite.image, pt.project_folder, src_channel_order="rgb")
    assert out.shape[:2] == (24, 24), f"the preview was resized to the full-frame pixel size: {out.shape}"
    full = np.zeros((96, 96, 3), np.uint16)
    out_full = pt.__class__.apply_aux_modifications(fp, full, pt.project_folder, src_channel_order="rgb")
    assert out_full.shape[:2] == (48, 48), "a full image must keep the exact pixel size"


def test_the_editor_keeps_its_preview_and_works_in_full_frame_pixels(synthetic_project, preview_ax, qapp):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = preview_ax({"resize_enabled": True, "resize": {"px_w": 48, "px_h": 48},
                     "resize_ref_size": {"w": 96, "h": 96}})
    dlg = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
    try:
        assert getattr(dlg, "_editor_full_shape", None), "precondition: the editor holds a preview"
        dlg.reapply_modifications()
        assert dlg.original_image.shape[:2] == (24, 24), f"editor preview blown up: {dlg.original_image.shape}"
        assert (dlg.resize_width_input.text(), dlg.resize_height_input.text()) == ("48", "48"), \
            "the resize boxes must show full-frame pixels"
    finally:
        dlg.close(); dlg.deleteLater()


def test_opening_the_editor_on_a_preview_and_saving_writes_no_resize(synthetic_project, preview_ax, qapp):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = preview_ax({"band_enabled": True})
    dlg = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
    try:
        assert getattr(dlg, "_editor_full_shape", None), "precondition: the editor holds a preview"
        assert dlg.save_modifications_to_file() is not None
        ax = json.load(open(pt._ax_path_for(fp), encoding="utf-8"))
        assert "resize" not in ax, f"a resize was written without being asked for: {ax.get('resize')}"
    finally:
        dlg.close(); dlg.deleteLater()
