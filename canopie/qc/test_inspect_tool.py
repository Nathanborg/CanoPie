"""
QC regression tests for the Pixel Inspector tool
(ImageViewer._inspect_at_scene_point).

Covers: correct per-band values at known coordinates (channel_order-aware,
post the session's channel_order fix -- all 8 fixtures should now agree, not
just the 3 originally-correct ones), the two distinct NoData display
mechanisms (numeric per-channel vs. string-expression whole-pixel), and the
band_expression "index" value for the .ax fixture that carries one.
"""
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from ._helpers import load_ground_truth, pixel_values_native_order, assert_close

ALL_NAMES = [f["name"] for f in FIXTURES]


def _make_viewer_for(synthetic_project, viewer_factory, name):
    """Real ImageViewer (via viewer_factory, so it's torn down at test
    teardown -- see conftest.py), real image_data (via the same
    _imagedata_or_fallback the app itself uses), and a same-size trivial
    pixmap so the scene<->image coordinate mapping in
    _inspect_at_scene_point is exactly 1:1."""
    fp = fixture_image_path(name)
    lite = synthetic_project._imagedata_or_fallback(fp)
    viewer = viewer_factory()
    viewer.resize(300, 300)
    h, w = lite.image.shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    viewer.set_image(QtGui.QPixmap.fromImage(qimg))
    viewer._image.setPos(0, 0)
    viewer.image_data = lite
    return viewer


def _inspect(viewer, x, y):
    captured = {}
    conn = viewer.pixel_clicked.connect(lambda pt, payload: captured.update(payload=payload))
    ok = viewer._inspect_at_scene_point(QtCore.QPointF(x + 0.5, y + 0.5))
    viewer.pixel_clicked.disconnect(conn)
    return ok, captured.get("payload")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_pixel_values_at_spot_points(synthetic_project, viewer_factory, name):
    """Inspect must report the true raw/edited data value at every declared
    spot point, in the correct (channel_order-aware) band order, for every
    fixture -- not just the 3 that happened to be correct before the
    channel_order fix."""
    gt = load_ground_truth(name)
    viewer = _make_viewer_for(synthetic_project, viewer_factory, name)

    # .ax fixtures: the viewer shows the RAW pre-edit image (see
    # test_image_viewer.py), so ground truth's post-edit "points" frame does
    # not apply here -- skip point-value comparison for those (band_expression
    # index value is covered separately below).
    if get_fixture(name).get("ax"):
        pytest.skip("ax fixture -- viewer shows raw pre-edit image, different frame")

    for point_name, p in gt["points"].items():
        x, y = p["x"], p["y"]
        ok, payload = _inspect(viewer, x, y)
        assert ok, f"{name}/{point_name}: inspect call failed"
        assert not payload["is_nodata"], f"{name}/{point_name}: unexpectedly flagged whole-pixel NoData"
        # payload["values"] is already in ch_names ("b1"=R, "b2"=G, "b3"=B, ...)
        # semantic order -- i.e. native file-band order -- regardless of
        # channel_order, since the fix makes the tool consult channel_order
        # internally and only swap when genuinely BGR. No further reordering
        # needed here (unlike the raw-array comparisons in test_image_viewer.py).
        actual = payload["values"][: len(p["values"])]
        for b, (a, e) in enumerate(zip(actual, p["values"])):
            assert_close(a, e, tol=1.0, msg=f"{name}/{point_name} band {b}")


def test_numeric_nodata_per_channel(synthetic_project, viewer_factory):
    """fixture 7's .ax carries numeric nodata_values=[9999], stamped at RAW
    coordinates (25,25) and (45,45) across all 4 bands. The viewer shows the
    RAW (uncropped) image, so Inspect at those exact raw coordinates must
    report every band as the literal text-flagged NoData, while a nearby
    valid pixel must show real numbers."""
    name = "ax_crop_nodata_source"
    viewer = _make_viewer_for(synthetic_project, viewer_factory, name)

    ok, payload = _inspect(viewer, 25, 25)
    assert ok
    assert not payload["is_nodata"], "numeric NoData is per-channel, not whole-pixel"
    assert payload["channel_nodata"] is not None
    assert all(payload["channel_nodata"]), f"expected every band flagged NoData at (25,25): {payload['channel_nodata']}"

    ok2, payload2 = _inspect(viewer, 2, 2)
    assert ok2
    assert not any(payload2["channel_nodata"]), f"expected no NoData at a clean pixel: {payload2['channel_nodata']}"


def test_expression_nodata_whole_pixel(synthetic_project, viewer_factory):
    """String-expression NoData (e.g. "b1>threshold") is a distinct mechanism
    from numeric per-channel NoData: it sets a whole-pixel is_nodata flag that
    suppresses ALL band values, not just the offending one. Verified in
    isolation (monkeypatched .ax mods) rather than via a dedicated fixture
    file, since it's a pure Inspect-side behavior independent of what's on
    disk."""
    name = "multiband_8band_ancillary"  # any clean, no-.ax fixture works
    gt = load_ground_truth(name)
    viewer = _make_viewer_for(synthetic_project, viewer_factory, name)

    p = gt["points"][gt["degenerate_point_name"]]
    r_value = p["values"][0]  # file band 0 == "b1" for this channel_order="rgb" fixture
    threshold = r_value - 1  # guarantees b1 > threshold is True at this point

    viewer._load_ax_mods = lambda path: {
        "nodata_enabled": True,
        "nodata_values": [f"b1>{threshold}"],
        "band_enabled": False,
    }

    ok, payload = _inspect(viewer, p["x"], p["y"])
    assert ok
    assert payload["is_nodata"] is True, "expression match should set the whole-pixel flag"


def test_band_expression_index_value(synthetic_project, viewer_factory):
    """fixture 8's .ax band_expression "(b4-b1)/(b4+b1)" must be evaluated by
    Inspect using file-native band order (channel_order="rgb", 4 bands -- the
    b1<->b3 swap only ever applies for exactly-3-channel BGR arrays), matching
    the ground truth's independently-computed NDVI-style value."""
    name = "ax_band_expression_source"
    gt = load_ground_truth(name)
    viewer = _make_viewer_for(synthetic_project, viewer_factory, name)

    # Ground truth's points are in the (identical, since no crop) post-edit
    # frame; raw_shape == working_shape's spatial dims for this fixture.
    p = gt["points"][gt["degenerate_point_name"]]
    ok, payload = _inspect(viewer, p["x"], p["y"])
    assert ok
    assert "index" in payload["names"], f"expected an 'index' channel, got {payload['names']}"
    idx_pos = payload["names"].index("index")
    actual_index = payload["values"][idx_pos]
    expected_index = p["values"][-1]  # ground truth appended the index value last
    assert_close(actual_index, expected_index, tol=1e-3, msg=f"{name} band_expression index")
