import pytest
from PyQt5 import QtCore, QtGui

from canopie.image_viewer import ImageViewer, attach_zoom_bar, attach_band_bar, attach_stretch_bar
from .fixtures_manifest import fixture_image_path

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer]

def _viewer_with_image(synthetic_project, viewer_factory, name="rgb_8bit_untiled"):
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    viewer = viewer_factory()
    viewer.resize(800, 600)
    h, w = lite.image.shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = lite
    
    # Ensure all bars are attached
    attach_zoom_bar(viewer)
    attach_band_bar(viewer)
    attach_stretch_bar(viewer)
    return viewer


def test_global_overlay_toggle(synthetic_project, viewer_factory):
    """
    Tests that the _OverlayToggleButton correctly toggles the global `overlays_muted` 
    flag on ImageViewer, and that bars respect this flag by not showing.
    """
    # Ensure default is unmuted
    ImageViewer.overlays_muted = False
    viewer1 = _viewer_with_image(synthetic_project, viewer_factory)
    viewer2 = _viewer_with_image(synthetic_project, viewer_factory)
    
    btn1 = viewer1._overlay_toggle_btn
    assert not ImageViewer.overlays_muted
    assert btn1.text() == "👁"
    
    # Show bars on both viewers
    viewer1._zoombar.show_briefly()
    viewer2._zoombar.show_briefly()
    assert not viewer1._zoombar.isHidden()
    assert not viewer2._zoombar.isHidden()
    
    # Toggle off via viewer 1
    btn1.toggle_overlays()
    assert ImageViewer.overlays_muted is True
    
    # The icon updates (though the cross-viewer sync in the UI uses topLevelWidgets,
    # which might not catch orphaned test widgets, but we can verify the class flag and current viewer)
    assert btn1.text() == "✕"
    
    # Bars on viewer1 should immediately hide
    assert viewer1._zoombar.isHidden()
    
    # Showing briefly while muted does nothing
    viewer1._zoombar.show_briefly()
    assert viewer1._zoombar.isHidden()
    
    viewer2._zoombar.show_briefly()
    assert viewer2._zoombar.isHidden()
    
    # Reset state for other tests
    ImageViewer.overlays_muted = False


def test_drawing_hides_overlays(synthetic_project, viewer_factory):
    """
    Tests that entering drawing mode hides active overlays, and they don't pop up again.
    """
    ImageViewer.overlays_muted = False
    viewer = _viewer_with_image(synthetic_project, viewer_factory)
    
    viewer._zoombar.show_briefly()
    viewer._bandbar.show_briefly()
    assert not viewer._zoombar.isHidden()
    assert not viewer._bandbar.isHidden()
    
    # Simulate start drawing
    viewer.drawing = True
    
    # Simulate mouse press to start a polygon
    # The viewer should hide the bars.
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(50, 50),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.ShiftModifier
    )
    viewer.mousePressEvent(ev)
    
    # Check that they are hidden
    assert viewer._zoombar.isHidden()
    assert viewer._bandbar.isHidden()
