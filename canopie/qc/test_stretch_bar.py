"""
QC tests for the in-viewport stretch bar (_StretchBar), the top-docked overlay
that stretches the display against the raster's absolute data range.

TWO BUGS ARE PINNED HERE.

1. THE MIN/MAX VALUES NEVER APPEARED. `ImageViewer._refresh_stretch_bar()` was
   CALLED from two places -- the `image_data` property setter and
   `attach_stretch_bar()` -- but was never actually defined. Both call sites
   wrap the call in a bare `try/except` that logs at debug level, so the
   AttributeError was swallowed on every single image load and the bar kept
   its construction defaults: the readout stuck on its "-" placeholder and the
   sliders mapping over an assumed 0-255.

2. THE RANGE DID NOT FOLLOW THE SELECTED BAND. The bounds came from a single
   pair of `_stretch_data_min` / `_stretch_data_max` attributes cached on the
   viewer, computed once over `_preview_take3` (the first three channels) and
   then reused unconditionally. Switching bands with the band-selector bar left
   the readout showing the previous band's range -- and on a cube whose band 1
   is reflectance (0-1) while band 11 is a sensor azimuth angle (0-360), the
   sliders were mapping over an interval unrelated to the pixels on screen, so
   dragging them did something arbitrary. The range is now derived from, and
   cached per, the band actually displayed.

The bar's bounds must also AGREE with the Stretch Viewer dialog's own data
range (same k=400 stride sampler, same NoData exclusion) -- that was the stated
requirement for the feature, and `test_bar_range_matches_stretch_dialog` is
what holds the two implementations together.
"""
import types

import numpy as np
import pytest
from PyQt5 import QtGui

from .fixtures_manifest import fixture_image_path, get_fixture
from ..project_tab import _StretchParams

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer]

# Bands with genuinely DIFFERENT bounds. Every committed fixture spans roughly
# 0-999 in every band, which would make a per-band range assertion pass even if
# band selection were ignored entirely -- the same scale-invariance trap
# documented for the stretch-rendering tests.
BAND_RANGES = [(0.0, 1.0), (0.0, 360.0), (-50.0, 50.0)]


def _multi_range_stack(h=32, w=32):
    arr = np.zeros((h, w, len(BAND_RANGES)), dtype=np.float32)
    for b, (lo, hi) in enumerate(BAND_RANGES):
        arr[..., b] = np.linspace(lo, hi, h * w).reshape(h, w)
    return arr


def _image_data(arr, *, filepath=None, ax_config=None, preview_bands=None):
    """Minimal stand-in for ImageData. _StretchBar reads exactly these
    attributes; the hand-built arrays here have no file on disk."""
    return types.SimpleNamespace(
        image=arr, channel_order="rgb", filepath=filepath,
        ax_config=ax_config or {}, preview_bands=preview_bands, profile=None)


def _viewer_with(viewer_factory, idata):
    arr = idata.image
    h, w = arr.shape[:2]
    viewer = viewer_factory()
    viewer.resize(420, 340)
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = idata          # setter -> _refresh_stretch_bar
    return viewer


def _select_band(viewer, pos):
    viewer.stretch_params = _StretchParams(
        scope="viewer", display_mode="single", display_band=pos)
    viewer._stretchbar.refresh_range()


def _select_composite(viewer):
    viewer.stretch_params = _StretchParams(scope="viewer", display_mode="auto")
    viewer._stretchbar.refresh_range()


# ---------------------------------------------------------------------------
# Bug 1 -- the range is populated at all
# ---------------------------------------------------------------------------
def test_refresh_stretch_bar_is_defined(viewer_factory):
    """THE regression for bug 1. Both call sites swallow AttributeError, so a
    missing method is invisible at runtime -- assert it exists and is callable
    rather than relying on any downstream symptom."""
    viewer = viewer_factory()
    assert callable(getattr(viewer, "_refresh_stretch_bar", None)), (
        "_refresh_stretch_bar is called by the image_data setter and by "
        "attach_stretch_bar, but is not defined on ImageViewer")


def test_bar_is_attached(viewer_factory):
    viewer = viewer_factory()
    assert viewer._stretchbar is not None, "stretch bar was never attached"


def test_range_is_populated_on_image_load(viewer_factory):
    """Loading an image must leave real bounds on the bar, not the constructed
    0-255 placeholder -- this is what "min and max not appearing" meant."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    sb = viewer._stretchbar

    assert (sb._data_min, sb._data_max) != (0.0, 255.0), (
        "bar still holds its placeholder range after an image was loaded")
    assert sb._data_min == pytest.approx(-50.0, abs=1e-3)
    assert sb._data_max == pytest.approx(360.0, abs=1e-3)


def test_label_shows_the_numeric_bounds(viewer_factory):
    """The user-visible readout, not just the internal attributes."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    text = viewer._stretchbar._lbl_vals.text()

    assert text.strip() not in ("", "-", "–"), (
        f"readout is still the placeholder: {text!r}")
    assert "-50" in text and "360" in text, f"unexpected readout: {text!r}"


# ---------------------------------------------------------------------------
# Bug 2 -- the range follows the selected band
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pos,expected", list(enumerate(BAND_RANGES)))
def test_range_adapts_to_the_selected_band(viewer_factory, pos, expected):
    """THE regression for bug 2: each band reports its OWN bounds."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    _select_band(viewer, pos)
    sb = viewer._stretchbar

    assert sb._data_min == pytest.approx(expected[0], abs=1e-3), (
        f"band {pos + 1} min: got {sb._data_min}, expected {expected[0]}")
    assert sb._data_max == pytest.approx(expected[1], abs=1e-3), (
        f"band {pos + 1} max: got {sb._data_max}, expected {expected[1]}")


def test_switching_bands_changes_the_range(viewer_factory):
    """Ordering matters: the stale global cache meant whichever band was
    measured FIRST won for the rest of the session. Walk the bands and assert
    the value actually moves each time."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    seen = []
    for pos in range(len(BAND_RANGES)):
        _select_band(viewer, pos)
        seen.append((viewer._stretchbar._data_min, viewer._stretchbar._data_max))

    assert len(set(seen)) == len(BAND_RANGES), (
        f"band switches did not change the range: {seen}")


def test_returning_to_composite_restores_the_full_range(viewer_factory):
    """The per-band cache must not leak the last single band's bounds back
    into the composite view."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    _select_band(viewer, 0)                      # narrow 0-1 band
    assert viewer._stretchbar._data_max == pytest.approx(1.0, abs=1e-3)

    _select_composite(viewer)
    assert viewer._stretchbar._data_min == pytest.approx(-50.0, abs=1e-3)
    assert viewer._stretchbar._data_max == pytest.approx(360.0, abs=1e-3)


def test_slider_positions_map_into_the_selected_bands_range(viewer_factory):
    """The bounds are not cosmetic -- _pos_to_val turns slider travel into data
    values, so a wrong range silently applies a wrong absolute stretch."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    _select_band(viewer, 0)                      # 0-1
    sb = viewer._stretchbar

    assert sb._pos_to_val(0) == pytest.approx(0.0, abs=1e-6)
    assert sb._pos_to_val(1000) == pytest.approx(1.0, abs=1e-6)
    assert sb._pos_to_val(500) == pytest.approx(0.5, abs=1e-3)


def test_band_ranges_are_tracked_per_file(viewer_factory):
    """Viewers are reused as the user pages through roots. A cache keyed only
    by band index would hand the next file the previous file's bounds."""
    viewer = _viewer_with(viewer_factory,
                          _image_data(_multi_range_stack(), filepath="a.tif"))
    _select_band(viewer, 1)
    assert viewer._stretchbar._data_max == pytest.approx(360.0, abs=1e-3)

    other = np.full((16, 16, 3), 7.5, dtype=np.float32)
    other[..., 1] = np.linspace(100.0, 200.0, 16 * 16).reshape(16, 16)
    viewer.image_data = _image_data(other, filepath="b.tif")
    _select_band(viewer, 1)

    assert viewer._stretchbar._data_max == pytest.approx(200.0, abs=1e-3), (
        "second file reused the first file's cached band range")


# ---------------------------------------------------------------------------
# NoData must not pollute the bounds
# ---------------------------------------------------------------------------
def test_range_excludes_nan(viewer_factory):
    """A NaN-filled border would otherwise make nanmin/nanmax meaningless and,
    worse, propagate NaN bounds into the slider mapping."""
    arr = _multi_range_stack()
    arr[0, :, :] = np.nan
    viewer = _viewer_with(viewer_factory, _image_data(arr))
    _select_band(viewer, 1)
    sb = viewer._stretchbar

    assert np.isfinite(sb._data_min) and np.isfinite(sb._data_max)
    assert sb._data_max == pytest.approx(360.0, abs=1.0)


def test_range_excludes_declared_nodata(viewer_factory):
    """-9999 fill must not drag the low bound down to -9999; the bar reads the
    same in-memory ax_config the rest of the app does."""
    arr = _multi_range_stack()
    arr[0, 0, 1] = -9999.0
    idata = _image_data(arr, ax_config={"nodata_enabled": True,
                                        "nodata_values": [-9999]})
    viewer = _viewer_with(viewer_factory, idata)
    _select_band(viewer, 1)

    assert viewer._stretchbar._data_min > -9000.0, (
        f"declared NoData leaked into the low bound: {viewer._stretchbar._data_min}")


def test_all_fill_band_does_not_report_a_data_range(synthetic_project, viewer_factory):
    """Real fixture: multiband_8band_ancillary's band 8 is 100% -9999 fill.
    Selecting it must not report the other bands' 0-999 range, which is exactly
    what the shared global cache did."""
    name = "multiband_8band_ancillary"
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    viewer = _viewer_with(viewer_factory, lite)

    _select_band(viewer, 0)
    normal_max = viewer._stretchbar._data_max

    last = get_fixture(name)["bands"] - 1
    if lite.image.ndim == 3 and last < lite.image.shape[2]:
        _select_band(viewer, last)
        assert viewer._stretchbar._data_max != pytest.approx(normal_max), (
            "the all-fill ancillary band reported the science bands' range")


# ---------------------------------------------------------------------------
# Agreement with the Stretch Viewer dialog
# ---------------------------------------------------------------------------
def test_bar_range_matches_stretch_dialog(synthetic_project, viewer_factory):
    """The stated requirement: the bar's bounds must be the same numbers the
    Stretch Viewer shows. Both now run the same k=400 stride sampler and the
    same NoData exclusion, so a drift in either implementation fails here."""
    from ..project_tab import ImageStretchDialog

    name = "multiband_8band_ancillary"
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    viewer = _viewer_with(viewer_factory, lite)
    _select_composite(viewer)
    sb = viewer._stretchbar

    dlg = ImageStretchDialog(synthetic_project, lite.image, image_filepath=fp)
    try:
        assert sb._data_min == pytest.approx(dlg._real_min, rel=1e-6, abs=1e-6), (
            f"bar min {sb._data_min} != dialog min {dlg._real_min}")
        assert sb._data_max == pytest.approx(dlg._real_max, rel=1e-6, abs=1e-6), (
            f"bar max {sb._data_max} != dialog max {dlg._real_max}")
    finally:
        dlg.close()
        dlg.setParent(None)
        dlg.deleteLater()


def test_seed_range_applies_only_to_the_current_band(viewer_factory):
    """open_stretch_dialog hands the dialog's range over via seed_range. It
    must land under the band context on screen -- writing it to one viewer-wide
    pair is precisely what made every band show the same bounds."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    _select_band(viewer, 0)
    viewer._stretchbar.seed_range(-7.0, 7.0)
    assert (viewer._stretchbar._data_min, viewer._stretchbar._data_max) == (-7.0, 7.0)

    _select_band(viewer, 1)
    assert viewer._stretchbar._data_max == pytest.approx(360.0, abs=1e-3), (
        "a range seeded for band 1 overwrote band 2's range")

    _select_band(viewer, 0)
    assert (viewer._stretchbar._data_min, viewer._stretchbar._data_max) == (-7.0, 7.0), (
        "the seeded range was lost when returning to its own band")


# ---------------------------------------------------------------------------
# Overlay behaviour
# ---------------------------------------------------------------------------
def test_bar_hides_when_drawing_starts(viewer_factory):
    """Matches _BandBar: the overlay must get out of the way for polygon work.
    isHidden(), not isVisible() -- the latter is False in a headless test
    regardless of the widget's own state."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    viewer._stretchbar.show_briefly()
    assert not viewer._stretchbar.isHidden()

    viewer.drawing = True
    assert viewer._stretchbar.isHidden(), "stretch bar stayed up while drawing"


def test_bar_docks_to_the_top_of_the_viewport(viewer_factory):
    """Top edge is what distinguishes it from the zoom/band bars at the bottom;
    an overlap would make one of them unclickable."""
    viewer = _viewer_with(viewer_factory, _image_data(_multi_range_stack()))
    sb = viewer._stretchbar
    sb.show_briefly()
    sb.reposition()

    vp_h = viewer.viewport().height()
    assert sb.geometry().top() < vp_h / 2, (
        f"stretch bar is not docked to the top: {sb.geometry()}")

    bb = viewer._bandbar
    if bb is not None and getattr(bb, "_buttons_by_band", None):
        bb.reposition()
        assert not sb.geometry().intersects(bb.geometry()), (
            f"stretch bar {sb.geometry()} overlaps band bar {bb.geometry()}")
