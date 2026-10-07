"""
BUG-008 / E1 (2026-10-06): what the image editor knows when it opens.

  * A reopen served from a cache keeps the pixels' channel order: the viewer's
    _raw_cache and the editor's own _raw_image_cache used to hand over the
    array alone, and the editor then assumed B,G,R -- `b1` meant Blue on a
    tifffile-loaded stack.
  * `orig_size` is the raster's header size (a stale value from a preview was
    never corrected).
  * The editor's band expression is NoData wherever a band it reads is.
"""
import json
import os

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.editor]


def _order(dlg):
    from ..utils import channel_order_attr
    return str(channel_order_attr(dlg, "_editor_channel_order", "bgr") or "bgr").lower()


def _close(dlg):
    try:
        dlg.close()
        dlg.deleteLater()
    except Exception:
        pass


@pytest.mark.parametrize("name, order", [("multiband_8band_ancillary", "rgb"), ("rgb_8bit_png", "bgr")])
def test_an_editor_opened_from_the_viewers_cache_knows_the_channel_order(synthetic_project, qapp, name, order):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = fixture_image_path(name)
    lite = pt._imagedata_or_fallback(fp)              # the viewer load caches the raw array
    key = os.path.normcase(os.path.abspath(fp))
    assert key in pt._raw_cache, "precondition: the viewer load did not cache the raw array"
    dlg = ImageEditorDialog(pt, image_data=lite.image, image_filepath=fp)
    try:
        assert _order(dlg) == order, f"{name}: editor thinks {_order(dlg)}, the pixels are {order}"
    finally:
        _close(dlg)


def test_a_second_open_from_the_editors_own_cache_keeps_the_order(synthetic_project, qapp):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = fixture_image_path("multiband_8band_ancillary")
    key = os.path.normcase(os.path.abspath(fp))
    try:
        pt._raw_cache.pop(key, None)                  # force: disk, then the editor's cache
        d1 = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
        o1 = _order(d1)
        _close(d1)
        pt._raw_cache.pop(key, None)
        d2 = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
        o2 = _order(d2)
        _close(d2)
    finally:
        pt._imagedata_or_fallback(fp)
    assert o1 == "rgb" and o2 == "rgb", f"first open {o1}, second (cached) open {o2}"


def test_orig_size_is_the_raster_header_size(synthetic_project, qapp):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: no .ax"
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"orig_size": {"h": 10, "w": 12, "c": 3}, "rotate": 0}, f)
        dlg = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
        try:
            assert dlg.modifications.get("orig_size") == {"h": 64, "w": 64, "c": 3}, (
                f"stale orig_size kept: {dlg.modifications.get('orig_size')}")
        finally:
            _close(dlg)
    finally:
        if os.path.exists(axp):
            os.remove(axp)


def test_the_editors_expression_is_nodata_where_a_band_it_reads_is(synthetic_project, qapp):
    from ..image_editor_dialog import ImageEditorDialog
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    dlg = ImageEditorDialog(pt, image_data=None, image_filepath=fp)
    try:
        img = np.full((6, 6, 3), 150, np.uint8)        # file order R,G,B
        img[1, 1, 0] = 0                                # b1 NoData here
        img[2, 2, 1] = 0                                # only b2 NoData here
        out = dlg.process_band_expression(img, "b1>100", nodata_values=[0], src_channel_order="rgb")
        band = np.asarray(out)[..., -1].astype(np.float64)
        assert np.isnan(band[1, 1]), f"b1>100 over a NoData b1 is {band[1, 1]!r}, not NoData"
        assert band[2, 2] == 1.0, "a NoData b2 must not affect an expression that only reads b1"
        assert band[0, 0] == 1.0
    finally:
        _close(dlg)
