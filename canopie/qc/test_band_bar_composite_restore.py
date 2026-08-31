"""
Regression test for a real user-reported bug: on a preview-loaded (windowed)
raster -- i.e. a large multi-band TIFF/COG shown through the
raster_reader-backed preview path -- clicking a single band on the in-viewport
band bar and then clicking the leading "Auto"/composite button back did NOT
actually restore the original RGB/full-band composite. It looked stuck on the
single band's grayscale render.

Root cause (project_tab.py, ProjectTab._on_band_bar_clicked): switching to a
single band calls set_viewer_preview_image, which REPLACES
viewer.image_data.image/preview_bands with just that one band's pixels --
the composite's actual pixel data is discarded, only its _StretchParams
(scope/mode/percentiles/etc, stashed as viewer._composite_stretch_params)
survived. The composite/"-1" restore branch only ever re-applied THAT stretch
object via _stretch_apply_preview, which renders whatever is currently
resident in viewer.image_data.image -- still the single band. Because
_compute_stretched_preview's "auto" fallback renders anything with < 3
channels as grayscale regardless of display_mode, the result silently stayed
a single-band grayscale image no matter what the (correctly restored)
stretch params said.

The fix (see _on_band_bar_clicked) reloads the actual composite bands --
stashed alongside the stretch params as viewer._composite_preview_bands, with
a self.get_display_bands(...) fallback -- back into
viewer.image_data.image/preview_bands before re-applying the stretch, and a
matching fix in ImageViewer.image_data's setter clears both stashed
attributes when a different file is assigned to the same viewer, so they can
never leak across images/roots either.

This bug does NOT reproduce on a fully-loaded (non-preview) image -- there,
the whole array stays resident across a band-bar click and only the display
slice changes, so the composite render was always correct. It is specific to
the preview/windowed-raster path, which is why the existing
test_composite_click_restores_previous_stretch (test_band_bar.py, exercised
against small non-preview QC fixtures) passed throughout -- it only ever
checked the restored _StretchParams object, never the actual resident pixel
data/rendered image the bug corrupts.
"""
import numpy as np
import pytest
from PyQt5 import QtGui

from .fixtures_manifest import fixture_image_path

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer, pytest.mark.io]

NAME = "hyperspectral_200band"  # only fixture big enough to force the preview path


def _viewer_with_lite(viewer_factory, lite):
    viewer = viewer_factory()
    viewer.resize(400, 320)
    h, w = lite.image.shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = lite          # property setter repopulates the band bar
    return viewer


def _pixmap_to_rgb_array(pixmap):
    """Read back exactly what was pushed to the viewer's pixmap item."""
    qimg = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(h * qimg.bytesPerLine())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, qimg.bytesPerLine())[:, : w * 3]
    return arr.reshape(h, w, 3).copy()


def test_composite_restore_reloads_original_bands_not_just_stretch(
    synthetic_project, viewer_factory, force_lazy_preview_no_decimation
):
    """Switch to a single band NOT in the default preview, then click back to
    composite -- the resident preview_bands/image must return to the ORIGINAL
    default composite, not stay parked on the single band."""
    fp = fixture_image_path(NAME)
    lite = synthetic_project._imagedata_or_fallback(fp)
    assert lite.profile is not None, "fixture must engage the preview/windowed path"
    default_bands = list(lite.preview_bands)
    assert default_bands == [0, 1, 2], f"unexpected default preview bands: {default_bands}"

    viewer = _viewer_with_lite(viewer_factory, lite)
    root = NAME

    # Establish the actual composite/auto stretch, as if the user had never
    # touched the band bar yet (mirrors test_band_bar.py's own setup).
    from canopie.project_tab import _StretchParams
    composite_params = _StretchParams(mode="percentile", low_p=1.0, high_p=99.0,
                                       display_mode="auto")
    viewer.stretch_params = composite_params

    # Click a single band far outside the default [0,1,2] composite.
    other_band = 50
    synthetic_project._on_band_bar_clicked(viewer, root, other_band)
    assert list(viewer.image_data.preview_bands) == [other_band]
    assert viewer.image_data.image.shape[-1] == 1
    single_band_pixmap = _pixmap_to_rgb_array(viewer._image.pixmap())

    # Click the leading composite/"Auto" button (-1 sentinel) to go back.
    synthetic_project._on_band_bar_clicked(viewer, root, -1)

    # --- The actual regression assertions -------------------------------
    assert list(viewer.image_data.preview_bands) == default_bands, (
        f"composite click left preview_bands={viewer.image_data.preview_bands!r} "
        f"instead of restoring the original {default_bands!r} -- pixel data for "
        "the composite was never reloaded, only the stretch parameters were."
    )
    assert viewer.image_data.image.shape[-1] == len(default_bands), (
        f"resident array still has {viewer.image_data.image.shape[-1]} channel(s) "
        f"after restoring the composite; expected {len(default_bands)}"
    )
    assert str(viewer.stretch_params.display_mode).lower() != "single"

    # --- Actual rendered pixels must match a fresh, independent render of ---
    # --- the SAME band selection, not stay stuck on the single-band frame ---
    restored_pixmap = _pixmap_to_rgb_array(viewer._image.pixmap())
    assert not np.array_equal(restored_pixmap, single_band_pixmap), (
        "the viewer's actual displayed pixmap did not change after clicking "
        "the composite button -- still showing the single-band render"
    )

    # Build the expected composite render completely independently of
    # _on_band_bar_clicked: load the default bands directly via the raster
    # reader plumbing and stretch them with the exact same params object.
    from canopie.raster_reader import probe, open_reader
    profile = probe(fp)
    reader = open_reader(fp, profile)
    fresh_arr = reader.read_bands_full(default_bands, level=0, step=1)
    expected_preview = synthetic_project._compute_stretched_preview(
        fresh_arr, composite_params, None)
    actual_preview = synthetic_project._compute_stretched_preview(
        viewer.image_data.image, viewer.stretch_params, None)
    assert actual_preview.shape == expected_preview.shape
    assert np.array_equal(actual_preview, expected_preview), (
        "restored composite render does not pixel-match an independent fresh "
        "render of the same band selection"
    )


def test_composite_restore_does_not_leak_across_images(synthetic_project, viewer_factory,
                                                          force_lazy_preview_no_decimation):
    """A composite stashed while viewing a single band of file A must not be
    used to restore file B's composite once the SAME viewer is reassigned to
    a genuinely DIFFERENT image_data (root/image navigation reusing the same
    widget -- ImageViewer instances are reused across roots)."""
    from canopie.project_tab import _StretchParams

    fp = fixture_image_path(NAME)
    lite_a = synthetic_project._imagedata_or_fallback(fp)
    viewer = _viewer_with_lite(viewer_factory, lite_a)
    root = NAME

    tuned = _StretchParams(mode="stddev", k_sigma=2.5, display_mode="auto")
    viewer.stretch_params = tuned
    synthetic_project._on_band_bar_clicked(viewer, root, 50)
    assert getattr(viewer, "_composite_stretch_params", None) is tuned
    assert getattr(viewer, "_composite_preview_bands", None) == [0, 1, 2]

    # Reassign to a DIFFERENT file's image_data (a different root/navigation
    # target reusing this same viewer widget) -- must not inherit fixture A's
    # stashed composite state.
    fp_b = fixture_image_path("multiband_8band_ancillary")
    lite_b = synthetic_project._imagedata_or_fallback(fp_b)
    assert lite_b.filepath != lite_a.filepath
    viewer.image_data = lite_b

    assert getattr(viewer, "_composite_stretch_params", None) is None, (
        "stale composite stretch from the previous image_data leaked across "
        "the image_data property setter"
    )
    assert getattr(viewer, "_composite_preview_bands", None) is None, (
        "stale composite preview bands from the previous image_data leaked "
        "across the image_data property setter"
    )
