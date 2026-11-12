"""
A COG zooms in to its real pixels, like any other image.

The viewer shows a large raster from a decimated preview (BCI: 48,031 x 50,101
shown from its 1/16 overview, image_data.display_scale = 16). The zoom caps
were measured against that preview: the wheel stopped at 50x and the zoom bar
at 20x, i.e. about 3 and 1.25 screen pixels per REAL pixel -- "it won't let me
zoom in onto the pixels". The sharp viewport tile already reads full
resolution, so the caps are now per real pixel (ImageViewer.max_zoom_factor);
a plain image keeps exactly its old limits.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer, attach_zoom_bar, _ZoomBar

pytestmark = [pytest.mark.viewer]


@pytest.fixture
def make_viewer(qapp, monkeypatch):
    monkeypatch.setattr(_ZoomBar, "_zoom_sync_enabled", False)
    made = []

    def _make(display_scale):
        v = ImageViewer()
        v.resize(400, 400)
        v.image_data = type("D", (), {"filepath": "x.tif", "image": None,
                                      "display_scale": display_scale})()
        pm = QtGui.QPixmap(300, 300)
        pm.fill(QtCore.Qt.gray)
        v.set_image(pm)
        attach_zoom_bar(v)
        v.show()
        qapp.processEvents()
        made.append(v)
        return v

    yield _make
    for v in made:
        v.hide()
        v.deleteLater()


def _wheel_in(v, qapp, notches=60):
    vp = v.viewport()
    c = QtCore.QPointF(vp.rect().center())
    for _ in range(notches):
        ev = QtGui.QWheelEvent(c, QtCore.QPointF(vp.mapToGlobal(c.toPoint())), QtCore.QPoint(0, 0),
                               QtCore.QPoint(0, 120), QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                               QtCore.Qt.NoScrollPhase, False)
        v.wheelEvent(ev)
        v._last_wheel_time = 0.0                 # no coalescing: apply every notch now
        v._apply_wheel_zoom()
        qapp.processEvents()
    return v.transform().m11()


def test_cog_wheel_zooms_to_real_pixels(make_viewer, qapp):
    v = make_viewer(16.0)
    z = _wheel_in(v, qapp)
    assert z > 50.0, "the wheel still stops at 50x on a COG preview"
    assert z == pytest.approx(50.0 * 16.0, rel=0.01)
    assert z / 16.0 >= 40, f"only {z / 16.0:.1f} screen px per real pixel"


def test_plain_image_keeps_its_50x_wheel_limit(make_viewer, qapp):
    v = make_viewer(1.0)
    assert _wheel_in(v, qapp) == pytest.approx(50.0, rel=0.01)


def test_zoom_bar_reaches_real_pixels_on_a_cog_only(make_viewer, qapp):
    cog, plain = make_viewer(16.0), make_viewer(1.0)
    cog.set_zoom_factor(400.0)
    plain.set_zoom_factor(400.0)
    assert cog.transform().m11() == pytest.approx(400.0)
    assert plain.transform().m11() == pytest.approx(50.0), "a plain image went past its old limit"

    zb = cog._zoombar
    zb._refresh_range()
    top = zb._slider_to_zoom(1000, zb._min_zoom, zb._max_zoom)
    assert top == pytest.approx(20.0 * 16.0), f"the zoom bar slider tops out at {top:.0f}x on a COG"
    pzb = plain._zoombar
    pzb._refresh_range()
    assert pzb._max_zoom == pytest.approx(20.0), "the plain image's zoom bar range changed"
