"""
BUG-009, viewer side (2026-10-07): the band bar names bands like the CSV.

On a fully loaded image the buttons were the array's POSITIONS labelled 1..C;
on a cv2-loaded image (B,G,R first) button "1" therefore showed file Blue, and
the band expression was "band 4" on screen while the CSV called it by its
expression. Buttons now follow FILE band numbers, derived bands come last with
their expression text, and a click still selects the right array position.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtGui

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer]


def test_band_bar_buttons_follow_file_bands_and_name_the_expression(synthetic_project, viewer_factory, qapp):
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_png")            # cv2-loaded: B,G,R array
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: no .ax"
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"band_enabled": True, "band_expression": "b1>100"}, f)
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
        lite = pt._imagedata_or_fallback(fp)
        assert str(lite.channel_order).lower() == "bgr"
        lite.image = pt.__class__.apply_aux_modifications(fp, lite.image, pt.project_folder,
                                                          src_channel_order=lite.channel_order)
        v = viewer_factory()
        v.setParent(pt)
        h, w = lite.image.shape[:2]
        v.set_image(QtGui.QPixmap.fromImage(QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)))
        v.image_data = lite
        v._refresh_band_bar()
        bb = v._bandbar
        order = [(bb._row.itemAt(k).widget()) for k in range(bb._row.count())]
        btns = [b for b in order if b is not None and b in bb._buttons_by_band.values()]
        texts = [b.text() for b in btns]
        assert texts == ["1", "2", "3", "b1>100"], texts
        pos_of = {b.text(): p for p, b in bb._buttons_by_band.items()}
        # button "1" is file band 1 (Red) = array position 2 on a B,G,R array
        assert pos_of["1"] == 2 and pos_of["3"] == 0 and pos_of["b1>100"] == 3
        got = []
        v.band_selected.connect(lambda i: got.append(i))
        [b for b in btns if b.text() == "1"][0].click()
        assert got and got[-1] == 2
        assert np.array_equal(lite.image[..., 2], lite.image[..., got[-1]])
    finally:
        if os.path.exists(axp):
            os.remove(axp)
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
