"""
QC tests for the in-viewport band-selector bar (_BandBar), the feature added
in this session.

Covers the behaviors that were specified for it: it appears for every image
(including single-band), lists one button per FILE band, switches the viewer
to single-band grayscale on click, restores the composite stretch via its
leading Auto/RGB button, hides immediately when drawing starts, and stays
docked to the bottom of the viewport below the zoom bar at any size.
"""
import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture

# One fixture per interesting band count: single-band, 3-band, 8-band, 200-band.
BAR_FIXTURES = ["gray_8bit_png", "rgb_8bit_untiled",
                "multiband_8band_ancillary", "hyperspectral_200band"]


def _viewer_with_image(synthetic_project, viewer_factory, name):
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    viewer = viewer_factory()
    viewer.resize(400, 320)
    h, w = lite.image.shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = lite          # property setter repopulates the bar
    return viewer


@pytest.mark.parametrize("name", BAR_FIXTURES)
def test_bar_lists_one_button_per_file_band(synthetic_project, viewer_factory, name):
    """The bar is indexed by FILE band, so a 200-band cube gets 200 buttons
    even though only ~3 are resident in a preview."""
    spec = get_fixture(name)
    viewer = _viewer_with_image(synthetic_project, viewer_factory, name)
    bar = viewer._bandbar
    assert bar is not None, "band bar was never attached"
    assert len(bar._buttons_by_band) == spec["bands"], (
        f"{name}: {len(bar._buttons_by_band)} buttons for {spec['bands']} bands")


def test_bar_shown_even_for_single_band(synthetic_project, viewer_factory):
    """Explicit product decision: the bar appears for EVERY image, not only
    multi-band ones -- so a 1-band image still gets a 1-button bar."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "gray_8bit_png")
    assert len(viewer._bandbar._buttons_by_band) == 1


def test_composite_button_exists_and_is_checked_by_default(synthetic_project, viewer_factory):
    """Leading button restores the real Auto/RGB composite; with no
    single-band stretch active it should read as the selected state."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    bar = viewer._bandbar
    assert bar._composite_btn.text() in ("Auto", "RGB")
    assert bar._composite_btn.isChecked(), (
        "composite should be the active selection when no band is being viewed")


def test_clicking_a_band_emits_its_file_index(synthetic_project, viewer_factory):
    """The signal must carry the FILE band index (not a position within a
    preview array), because that is what ProjectTab reloads against."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "multiband_8band_ancillary")
    received = []
    viewer.band_selected.connect(lambda i: received.append(i))

    viewer._bandbar._buttons_by_band[5].click()
    assert received == [5], f"expected band index 5, got {received}"


def test_composite_button_emits_sentinel(synthetic_project, viewer_factory):
    """-1 is the agreed sentinel for "restore the composite"."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    received = []
    viewer.band_selected.connect(lambda i: received.append(i))

    viewer._bandbar._composite_btn.click()
    assert received == [-1], f"expected the -1 composite sentinel, got {received}"


def test_selection_is_exclusive(synthetic_project, viewer_factory):
    """Clicking a band must deselect the composite (and vice versa) -- the bar
    should never show two active selections."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "multiband_8band_ancillary")
    bar = viewer._bandbar

    bar._buttons_by_band[2].click()
    assert bar._buttons_by_band[2].isChecked()
    assert not bar._composite_btn.isChecked(), "composite stayed checked after a band click"

    bar._buttons_by_band[4].click()
    assert bar._buttons_by_band[4].isChecked()
    assert not bar._buttons_by_band[2].isChecked(), "two band buttons checked at once"


def test_bar_hides_immediately_when_drawing_starts(synthetic_project, viewer_factory):
    """Specified behavior: the bar vanishes the instant a draw begins -- not
    after the auto-hide timeout -- so it never overlaps an in-progress shape.
    Driven through the `drawing` property, which is the single choke point all
    the drawing entry points go through."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    bar = viewer._bandbar

    bar.show_briefly()
    # isHidden(), not isVisible(): isVisible() is False whenever no ancestor
    # window is on screen, which is always true in a headless test -- so an
    # isVisible()-based "is hidden" assertion passes vacuously and proves
    # nothing. isHidden() reflects the widget's OWN explicit hide state, which
    # is exactly the behavior under test.
    assert not bar.isHidden(), "precondition failed: bar should not be hidden"

    viewer.drawing = True
    assert bar.isHidden(), "band bar must hide immediately when drawing starts"


def test_bar_docks_to_bottom_below_zoom_bar(synthetic_project, viewer_factory):
    """Layout contract: band bar pinned to the viewport bottom, zoom bar
    directly above it, never overlapping."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    bar, zoom = viewer._bandbar, viewer._zoombar
    bar.show_briefly()
    zoom.show()
    zoom.reposition()
    bar.reposition()

    vp_h = viewer.viewport().height()
    br, zr = bar.geometry(), zoom.geometry()

    assert br.bottom() <= vp_h, "band bar extends past the bottom of the viewport"
    assert vp_h - br.bottom() <= 12, f"band bar not docked to the bottom (gap {vp_h - br.bottom()}px)"
    assert zr.top() < br.top(), "zoom bar must sit above the band bar"
    assert not zr.intersects(br), "zoom bar and band bar overlap"


def test_bar_redocks_after_viewport_resize(synthetic_project, viewer_factory, qapp):
    """Zooming past fit brings in scrollbars, which resizes the VIEWPORT
    without necessarily resizing the view -- the bar must follow. (This is the
    exact case that the viewport eventFilter was added for.)"""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    bar = viewer._bandbar
    bar.show_briefly()

    viewer.viewport().resize(viewer.viewport().width(), viewer.viewport().height() - 60)
    qapp.processEvents()

    vp_h = viewer.viewport().height()
    gap = vp_h - bar.geometry().bottom()
    assert gap <= 12, f"band bar did not re-dock after a viewport-only resize (gap {gap}px)"


def test_repopulates_when_image_changes(synthetic_project, viewer_factory):
    """Assigning a new image_data must rebuild the bar for the new band count
    -- the property setter is what guarantees this without every call site
    remembering to refresh."""
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    assert len(viewer._bandbar._buttons_by_band) == 3

    other = synthetic_project._imagedata_or_fallback(
        fixture_image_path("multiband_8band_ancillary"))
    viewer.image_data = other
    assert len(viewer._bandbar._buttons_by_band) == 8, (
        "band bar did not repopulate after image_data changed")


def test_bar_hides_when_image_cleared(synthetic_project, viewer_factory):
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    viewer._bandbar.show_briefly()
    assert not viewer._bandbar.isHidden(), "precondition: bar should be showing first"
    viewer.image_data = None
    assert viewer._bandbar.isHidden(), "bar should hide when there is no image"


def test_band_click_switches_viewer_to_that_band(synthetic_project, viewer_factory):
    """End-to-end through ProjectTab's handler: clicking a band must leave the
    viewer in single-band mode pointed at that band, using a plain percentile
    stretch (not the composite's tuned parameters)."""
    name = "multiband_8band_ancillary"
    viewer = _viewer_with_image(synthetic_project, viewer_factory, name)
    root = name  # project_builder uses the fixture name as the root name

    synthetic_project._on_band_bar_clicked(viewer, root, 4)

    sp = getattr(viewer, "stretch_params", None)
    assert sp is not None, "no stretch_params set after a band click"
    assert str(sp.display_mode).lower() == "single", f"display_mode={sp.display_mode!r}"
    assert sp.mode == "percentile", (
        f"band clicks should apply a plain percentile stretch, got {sp.mode!r}")


def test_composite_click_restores_previous_stretch(synthetic_project, viewer_factory):
    """After viewing a single band, the composite button must restore the
    ACTUAL previous composite params, not a generic default."""
    from ..project_tab import _StretchParams

    name = "multiband_8band_ancillary"
    viewer = _viewer_with_image(synthetic_project, viewer_factory, name)
    root = name

    tuned = _StretchParams(mode="stddev", k_sigma=2.5, per_channel=False, display_mode="auto")
    viewer.stretch_params = tuned

    synthetic_project._on_band_bar_clicked(viewer, root, 3)
    assert str(viewer.stretch_params.display_mode).lower() == "single"

    synthetic_project._on_band_bar_clicked(viewer, root, -1)
    restored = viewer.stretch_params
    assert str(restored.display_mode).lower() == "auto", f"got {restored.display_mode!r}"
    assert restored.mode == "stddev" and restored.k_sigma == 2.5, (
        "composite restore lost the original tuned parameters "
        f"(mode={restored.mode!r}, k_sigma={restored.k_sigma})")


def test_top_stretch_bar_attached_and_emits_absolute_stretch(synthetic_project, viewer_factory):
    """The top stretch bar (_StretchBar) is attached to ImageViewer and allows
    adjusting absolute contrast stretch ranges via Min/Max sliders."""
    from PyQt5 import QtWidgets
    from ..image_viewer import attach_stretch_bar
    viewer = _viewer_with_image(synthetic_project, viewer_factory, "rgb_8bit_untiled")
    QtWidgets.QApplication.processEvents()
    sb = getattr(viewer, "_stretchbar", None) or attach_stretch_bar(viewer)
    assert sb is not None, "_StretchBar overlay was not attached"

    received_params = []
    viewer.stretch_applied.connect(lambda p: received_params.append(p))

    # Move slider min/max
    sb._slider_min.setValue(100)
    sb._slider_max.setValue(900)
    sb._apply_stretch()

    assert len(received_params) > 0, "stretch_applied was not emitted"
    last = received_params[-1]
    assert last is not None
    assert last.mode == "absolute"
    assert last.min_val is not None
    assert last.max_val is not None
    assert last.min_val < last.max_val


