"""
QC regression tests for how the Pixel Inspector reports NON-FINITE pixels.

THE BUG THIS PINS (reported from the viewer: "there are pixels the image
viewer clearly marks as NaN but the inspect tool reports them as a number"):

`_inspect_at_scene_point` only ever ran its `math.isfinite` check inside

    if nodata_values:          # .ax declares SOMETHING
        ...
        if numeric_vals:       # ...and at least one of them is a NUMBER
            for i, v in enumerate(vals):
                if not math.isfinite(v):
                    channel_is_nodata[i] = True

so a NaN pixel was flagged only when the file happened to have an `.ax`
sidecar declaring a numeric fill value such as -9999. On a raster with NO
`.ax` at all -- or one declaring only boolean expressions -- every NaN read
back as an ordinary value while the viewer rendered the very same pixel as a
masked hole.

That combination is the normal case, not an edge case: float prediction
stacks mark their science bands' fill as NaN (and their ancillary planes'
as -9999), and CanoPie auto-detects the file's own GDAL_NODATA tag rather
than requiring an `.ax`. Every other subsystem already treats non-finite as
unconditionally invalid -- `utils.build_nodata_mask` ORs `~np.isfinite` in as
its final pass, `_per_band_nodata_masks` seeds its mask with `~np.isfinite`,
and the polygon statistics go through nanmean/nanmedian. The Inspector was
the sole holdout.

Also pinned here: an index computed from a `band_expression` over NaN inputs
used to be `np.nan_to_num`'d to 0.0 before being reported, making "no data
here" indistinguishable from a genuine zero index.
"""
import math
import types

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from ..project_tab import ProjectTab

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer, pytest.mark.extraction]


def _image_data(arr, *, filepath=None, channel_order="rgb"):
    """Minimal stand-in for ImageData.

    `_inspect_at_scene_point` reads exactly four attributes off it, and the
    arrays under test here (hand-built NaN/Inf patterns) deliberately have no
    file on disk, so a real loader-produced object cannot be used.
    """
    return types.SimpleNamespace(
        image=arr, channel_order=channel_order,
        filepath=filepath, ax_config={}, preview_bands=None, profile=None)


def _viewer_with_array(viewer_factory, arr, *, filepath=None):
    h, w = arr.shape[:2]
    viewer = viewer_factory()
    viewer.resize(320, 260)
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = _image_data(arr, filepath=filepath)
    return viewer


def _inspect(viewer, x, y):
    """Click pixel (x, y) and return the emitted payload."""
    got = []
    viewer.pixel_clicked.connect(lambda pt, payload: got.append(payload))
    viewer._inspect_at_scene_point(QtCore.QPointF(x + 0.5, y + 0.5))
    assert got, "pixel_clicked never fired"
    return got[-1]


def _nan_stack():
    """8x8x3 float32. (2,3) is NaN in every band, (5,5) in band 1 only,
    (6,1) is +Inf in band 2. Everything else is finite and distinct."""
    arr = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    arr[3, 2, :] = np.nan
    arr[5, 5, 0] = np.nan
    arr[1, 6, 1] = np.inf
    return arr


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------
def test_nan_pixel_is_nodata_with_no_ax_sidecar(viewer_factory):
    """THE regression: no `.ax`, so nodata_values is empty -- the NaN check
    used to be skipped entirely and every channel reported a bare `nan`."""
    viewer = _viewer_with_array(viewer_factory, _nan_stack(), filepath=None)
    payload = _inspect(viewer, 2, 3)

    assert payload["channel_nodata"] == [True, True, True], (
        "an all-NaN pixel must read as NoData in every channel, got "
        f"{payload['channel_nodata']} for values {payload['values']}")


def test_nan_in_one_band_only_flags_that_band(viewer_factory):
    """Per-channel, not whole-pixel: the two finite bands still report their
    real values, matching the per-band NoData semantics used everywhere else."""
    viewer = _viewer_with_array(viewer_factory, _nan_stack())
    payload = _inspect(viewer, 5, 5)

    assert payload["channel_nodata"] == [True, False, False]
    assert math.isfinite(payload["values"][1])
    assert math.isfinite(payload["values"][2])


def test_infinity_is_also_nodata(viewer_factory):
    """+-Inf is as unusable as NaN and `~np.isfinite` catches both, so the
    Inspector must not special-case only NaN."""
    viewer = _viewer_with_array(viewer_factory, _nan_stack())
    payload = _inspect(viewer, 6, 1)

    assert payload["channel_nodata"][1] is True, (
        f"+Inf must be NoData, got {payload['values']}")


def test_finite_pixels_are_untouched(viewer_factory):
    """The fix must not start flagging good data -- the failure mode that
    would silently blank the readout everywhere."""
    viewer = _viewer_with_array(viewer_factory, _nan_stack())
    payload = _inspect(viewer, 0, 0)

    assert payload["channel_nodata"] == [False, False, False]
    assert all(math.isfinite(v) for v in payload["values"])


def test_nan_flagged_even_when_ax_declares_only_expressions(viewer_factory, tmp_path):
    """The second half of the gate: `nodata_values` was non-empty but held no
    NUMERIC entry, so `if numeric_vals:` was False and the NaN pass never ran.
    An expression-only .ax is a normal configuration."""
    import json
    fp = tmp_path / "expr_only.tif"
    fp.write_bytes(b"")                       # only the .ax beside it is read
    (tmp_path / "expr_only.ax").write_text(
        json.dumps({"nodata_enabled": True, "nodata_values": ["b1>1e9"]}),
        encoding="utf-8")

    viewer = _viewer_with_array(viewer_factory, _nan_stack(), filepath=str(fp))
    payload = _inspect(viewer, 2, 3)

    assert payload["channel_nodata"] == [True, True, True]


def test_numeric_ax_nodata_still_works(viewer_factory, tmp_path):
    """The path that DID work before must keep working: a declared -9999 is
    still matched per channel, alongside the new unconditional NaN check."""
    import json
    arr = _nan_stack()
    arr[0, 1, 2] = -9999.0

    fp = tmp_path / "numeric.tif"
    fp.write_bytes(b"")
    (tmp_path / "numeric.ax").write_text(
        json.dumps({"nodata_enabled": True, "nodata_values": [-9999]}),
        encoding="utf-8")

    viewer = _viewer_with_array(viewer_factory, arr, filepath=str(fp))
    assert _inspect(viewer, 1, 0)["channel_nodata"] == [False, False, True]
    assert _inspect(viewer, 2, 3)["channel_nodata"] == [True, True, True]


def test_band_expression_index_over_nan_is_nodata_not_zero(viewer_factory, tmp_path):
    """A band_expression index used to be nan_to_num'd to 0.0 before readout,
    so an index over NoData inputs reported a hard `0` -- indistinguishable
    from a genuine zero index (an NDVI of 0 is a real, meaningful value)."""
    import json
    fp = tmp_path / "expr.tif"
    fp.write_bytes(b"")
    (tmp_path / "expr.ax").write_text(
        json.dumps({"band_enabled": True, "band_expression": "(b2-b1)/(b2+b1)"}),
        encoding="utf-8")

    viewer = _viewer_with_array(viewer_factory, _nan_stack(), filepath=str(fp))
    payload = _inspect(viewer, 2, 3)

    assert "index" in payload["names"], (
        f"band_expression produced no index channel: {payload['names']}")
    i = payload["names"].index("index")
    assert payload["channel_nodata"][i] is True, (
        f"index over NaN inputs reported {payload['values'][i]!r} instead of NoData")


# ---------------------------------------------------------------------------
# The readout the user actually sees
# ---------------------------------------------------------------------------
def test_status_bar_renders_nodata_channels_as_text():
    """End of the chain: ProjectTab.display_pixel_value must turn the payload's
    channel_nodata flags into "NoData" rather than printing `nan`.

    Called unbound against a stub `self` -- the method touches only
    `self.window()` and `self.sender`, and re-parenting the shared ProjectTab
    into a QMainWindow just to reach a status bar would destabilise the rest
    of the suite."""
    shown = []
    fake_self = types.SimpleNamespace(
        window=lambda: types.SimpleNamespace(
            statusBar=lambda: types.SimpleNamespace(
                showMessage=lambda m: shown.append(m))),
        sender=None)

    ProjectTab.display_pixel_value(
        fake_self, QtCore.QPointF(2, 3),
        {"values": [float("nan"), 136.0, 137.0],
         "names": ["b1", "b2", "b3"],
         "is_nodata": False,
         "channel_nodata": [True, False, False]})

    assert shown, "nothing was pushed to the status bar"
    msg = shown[-1]
    assert "b1=NoData" in msg, f"NaN channel not rendered as NoData: {msg!r}"
    assert "nan" not in msg.lower().replace("nodata", ""), (
        f"a raw nan leaked into the readout: {msg!r}")
    assert "b2=136" in msg, f"finite channels must still show values: {msg!r}"
