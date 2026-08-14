"""
QC regression tests for rendering a REAL multi-band, fill-heavy raster in the
Image Editor.

WHY THIS FILE EXISTS
--------------------
test_image_editor_open_perf.py pins the checkbox "signal storm" -- but it does
so with a 10x10 synthetic uint8 array pointed at a filepath that does not
exist, an .ax with every flag False, no band expression and no NoData values,
and it asserts a *call count*. That test cannot observe whether anything is
actually drawn, so it stayed green while the editor mis-rendered every image
in a real project (C:\\Final_liana_project_canopie_2: 385 scenes, each a
1759x1930 float32 stack of 15 bands). These tests use imagery shaped like that
project's, and assert on what reaches the screen.

The scenes there have a structure that is ordinary for a prediction stack and
that none of the existing fixtures had: bands 8 and 9 are -9999 in EVERY pixel
(all-fill ancillary planes), bands 1-7 are ~94% NaN (outside the flight
swath), and the .ax carries ``nodata_values: ['b2>1', -9999, 'b2<0.70']``.

THE BUGS THESE PIN
------------------
1. ``_pixmap_with_ax_stretch``'s no-parent fallback called
   ``_pixmap_from_uint8(disp)``, which is defined in project_tab.py and was
   never imported into image_editor_dialog.py. The fallback therefore ran the
   ENTIRE multi-band stretch and then raised ``NameError`` on its last line --
   swallowed by every caller's ``except Exception``, which quietly fell back
   to ``normalize_image_for_display``. The .ax stretch was silently ignored
   and the work was pure waste.

2. That stretch computed percentile bounds and a full uint8 conversion for
   ALL C channels, then discarded everything except the 1 or 3 channels it
   displays -- 12 wasted percentile passes and 12 wasted stretches per render
   on a 15-band scene. The main viewer's equivalent
   (``ProjectTab._compute_stretched_preview``) has always picked its display
   stack *before* computing statistics.

3. It seeded its output with ``y = x.copy()`` -- float32 -- and assigned uint8
   results into it, so the array handed to ``_pixmap_from_uint8`` was float32.
   That helper wraps the raw buffer via ``sip.voidptr()`` as Grayscale8/BGR888,
   i.e. it reinterprets float bytes as pixels.

4. NaN survived the stretch arithmetic and ``np.clip``, and casting NaN to
   uint8 is undefined: it emitted "invalid value encountered in cast" and
   produced arbitrary bytes, so NoData pixels rendered as random speckle.

5. ``resizeEvent`` called ``display_image(self.display_image_data)``
   unconditionally. Qt delivers a resize while the dialog is being shown,
   before showEvent() has populated that attribute, so every editor open
   logged "No image provided to display." at WARNING level -- which reads as a
   failed load and was reported as one.
"""
import json
import logging
import os

import numpy as np
import pytest
import tifffile

from ..image_editor_dialog import ImageEditorDialog

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor]

# Mirrors C:\Final_liana_project_canopie_2's scenes, at a size that keeps the
# suite fast. Band count is what matters here: it must exceed the 3 channels
# the editor can actually display, or bug #2 is invisible.
_BANDS = 15
_H, _W = 48, 64
_ALL_FILL_BANDS = (7, 8)   # 0-based: bands 8 and 9, -9999 everywhere
_NAN_BANDS = range(0, 7)   # 0-based: bands 1-7, mostly NaN


def _write_scene(tmp_path, name="scene.tif", with_ax=True):
    """A float32 stack shaped like the real prediction scenes."""
    rng = np.random.RandomState(0)
    arr = rng.rand(_BANDS, _H, _W).astype(np.float32)
    for b in _ALL_FILL_BANDS:
        arr[b, :, :] = -9999.0
    for b in _NAN_BANDS:
        arr[b, : int(_H * 0.94), :] = np.nan

    tif = str(tmp_path / name)
    tifffile.imwrite(tif, arr, photometric="minisblack", planarconfig="separate")

    if with_ax:
        ax = {
            "orig_size": {"h": _H, "w": _W, "c": _BANDS},
            "nodata_values": ["b2>1", -9999, "b2<0.70"],
            "nodata_enabled": True,
            "resize_enabled": True, "rotate_enabled": True, "crop_enabled": True,
            "band_expression": "", "band_enabled": True,
            "hist_enabled": True, "hist_match": None,
        }
        with open(os.path.splitext(tif)[0] + ".ax", "w", encoding="utf-8") as f:
            json.dump(ax, f)
    return tif


def _open(tmp_path, **kw):
    tif = _write_scene(tmp_path, **kw)
    return ImageEditorDialog(parent=None, image_data=None, image_filepath=tif)


def test_local_stretch_returns_a_pixmap_instead_of_raising(qapp, tmp_path):
    """Bug #1. Pre-fix this raised NameError on every call."""
    from PyQt5 import QtGui

    dlg = _open(tmp_path)
    try:
        pm = dlg._pixmap_with_ax_stretch(
            np.random.RandomState(0).rand(_H, _W, _BANDS).astype(np.float32))
        assert isinstance(pm, QtGui.QPixmap), (
            "_pixmap_with_ax_stretch did not produce a QPixmap -- the no-parent "
            "fallback is broken again (it is reached whenever the dialog has no "
            "ProjectTab renderer)")
        assert (pm.width(), pm.height()) == (_W, _H)
    finally:
        dlg.close()


@pytest.mark.parametrize("shape", [(_H, _W), (_H, _W, 2), (_H, _W, 3), (_H, _W, _BANDS)])
def test_local_stretch_handles_every_channel_count(qapp, tmp_path, shape):
    """The band-selection rewrite has to keep the 2D / 2-band / 3-band /
    many-band assembly paths working, not just the 15-band one."""
    from PyQt5 import QtGui

    dlg = _open(tmp_path)
    try:
        pm = dlg._pixmap_with_ax_stretch(
            np.random.RandomState(0).rand(*shape).astype(np.float32))
        assert isinstance(pm, QtGui.QPixmap) and not pm.isNull()
    finally:
        dlg.close()


def test_stretch_computes_bounds_only_for_displayed_bands(qapp, tmp_path, monkeypatch):
    """Bug #2. At most 3 channels are ever shown, so at most 3 sets of
    percentile bounds should be computed -- not one per band."""
    from canopie import utils

    calls = []
    real = utils._nanpct
    monkeypatch.setattr(utils, "_nanpct",
                        lambda a, p, axis=None: (calls.append(1), real(a, p, axis))[1])

    dlg = _open(tmp_path)
    try:
        dlg._pixmap_with_ax_stretch(
            np.random.RandomState(0).rand(_H, _W, _BANDS).astype(np.float32))
    finally:
        dlg.close()

    assert len(calls) <= 3, (
        f"stretch computed percentile bounds {len(calls)} times for a "
        f"{_BANDS}-band image; only the <=3 displayed bands should be measured")


def test_stretch_hands_uint8_to_the_pixmap_builder(qapp, tmp_path, monkeypatch):
    """Bug #3. _pixmap_from_uint8 reinterprets the raw buffer as 8-bit pixels,
    so anything but uint8 renders as noise."""
    from canopie import project_tab

    seen = {}
    real = project_tab._pixmap_from_uint8

    def spy(disp):
        seen["dtype"] = disp.dtype
        seen["shape"] = disp.shape
        return real(disp)

    monkeypatch.setattr(project_tab, "_pixmap_from_uint8", spy)

    dlg = _open(tmp_path)
    try:
        dlg._pixmap_with_ax_stretch(
            np.random.RandomState(0).rand(_H, _W, _BANDS).astype(np.float32))
    finally:
        dlg.close()

    assert seen.get("dtype") == np.uint8, (
        f"pixmap builder received {seen.get('dtype')}, not uint8")
    assert seen.get("shape") == (_H, _W, 3)


def test_fill_and_nan_pixels_do_not_produce_undefined_bytes(qapp, tmp_path):
    """Bug #4. All-NaN and all-fill bands must yield deterministic pixels, not
    whatever the undefined NaN->uint8 cast happens to emit."""
    from canopie import project_tab

    arr = np.random.RandomState(0).rand(_H, _W, _BANDS).astype(np.float32)
    arr[..., 0] = np.nan              # an entirely-NaN displayed band
    arr[..., 1] = -9999.0             # an all-fill displayed band

    dlg = _open(tmp_path)
    captured = []
    real = project_tab._pixmap_from_uint8
    project_tab._pixmap_from_uint8 = lambda d: (captured.append(d.copy()), real(d))[1]
    try:
        with np.errstate(all="raise"):
            dlg._pixmap_with_ax_stretch(arr)
    finally:
        project_tab._pixmap_from_uint8 = real
        dlg.close()

    # `disp` is the assembled display stack in `sel` order, which for a
    # >3-band image in auto mode is bands 0,1,2 -- so the doctored bands land
    # at ch0 (all-NaN) and ch1 (all -9999), and ch2 is untouched real data.
    disp = captured[0]
    assert disp.dtype == np.uint8
    assert np.all(disp[..., 0] == 0), (
        "an all-NaN band should stretch to a constant 0, got varying bytes -- "
        "the NaN->uint8 cast is undefined again")
    assert np.all(disp[..., 1] == 0), (
        "an all-fill (-9999) band has zero spread, so it should stretch to a "
        "constant, not to arbitrary bytes")
    assert np.ptp(disp[..., 2]) > 0, (
        "the one band holding real data rendered flat -- the degenerate bands "
        "are contaminating the bands that still have signal")


def test_opening_editor_on_fill_heavy_stack_displays_an_image(qapp, tmp_path, caplog):
    """Bugs #4/#5 end-to-end: opening a scene whose NoData masks 100% of pixels
    must still put a real image on screen and must not log a load failure."""
    dlg = _open(tmp_path)
    seen_none = []
    real_display = ImageEditorDialog.display_image

    def spy(self, img):
        if img is None:
            seen_none.append(1)
        return real_display(self, img)

    ImageEditorDialog.display_image = spy
    try:
        with caplog.at_level(logging.WARNING):
            dlg.show()
            qapp.processEvents()
    finally:
        ImageEditorDialog.display_image = real_display
        dlg.close()

    assert not seen_none, (
        "display_image(None) was called while opening the editor -- the "
        "resizeEvent guard is gone (this is the 'No image provided to "
        "display.' warning users reported as a failed load)")
    assert dlg.display_image_data is not None, "editor finished with no image"
    assert "No image provided to display" not in caplog.text


def test_resize_before_first_render_is_a_noop(qapp, tmp_path):
    """Bug #5 directly: a resize delivered before the first render must not
    push None through the display path.

    The handler is invoked explicitly rather than via ``dlg.resize(...)``:
    Qt only delivers a resize event to a dialog that has been shown, and
    showing it runs the very render whose absence this test is about, so the
    widget-level call quietly tests nothing (verified -- it passed against the
    unfixed source).
    """
    from PyQt5 import QtCore, QtGui

    dlg = _open(tmp_path)
    calls = []
    real_display = ImageEditorDialog.display_image
    ImageEditorDialog.display_image = lambda self, img: calls.append(img)
    try:
        assert dlg.display_image_data is None, "precondition: nothing rendered yet"
        dlg.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(400, 300),
                                           QtCore.QSize(980, 650)))
    finally:
        ImageEditorDialog.display_image = real_display
        dlg.close()

    assert None not in calls, "resizeEvent still displays None before first render"
