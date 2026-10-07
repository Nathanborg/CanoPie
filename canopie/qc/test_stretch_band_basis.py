"""
BUG-024 (2026-10-06): a saved stretch names FILE bands, not array positions.

The viewer's stretch picks bands by position in whatever array the loader
produced (B,G,R first for cv2; file order for tifffile and previews; a band
subset for a decimated preview; derived bands appended). Saving those
positions made a stretch saved on a cv2-loaded image (r/g/b = 2/1/0, natural
colour) swap red and blue once the file loaded in file order -- measured on
Nova pasta (5): screen red correlated 1.000 with file Blue. A root / project
stretch applied one image's positions to every other image.

Now: `band_basis: "file"` with file bands / derived band names on disk, and a
draw maps them to the positions of the array being drawn.
"""
import json
import os
import types

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer]


def _sp(**kw):
    from ..project_tab import _StretchParams
    base = dict(mode="percentile", low_p=1.0, high_p=99.0, scope="viewer", display_mode="rgb")
    base.update(kw)
    return _StretchParams(**base)


def _viewer(img, order, fp=None, preview_bands=None, full_shape=None):
    idata = types.SimpleNamespace(image=img, channel_order=order, filepath=fp,
                                  preview_bands=preview_bands, full_shape=full_shape)
    return types.SimpleNamespace(image_data=idata, stretch_params=None)


def _rgb_image():
    rng = np.random.default_rng(5)
    r = rng.uniform(0, 255, (40, 50)).astype(np.float32)
    g = np.full((40, 50), 128, np.float32)
    b = (255 - r).astype(np.float32)                 # anti-correlated with R
    return np.dstack([r, g, b])                      # FILE order


def test_a_stretch_saved_on_a_bgr_viewer_draws_natural_colour_on_a_file_order_viewer(synthetic_project):
    pt = synthetic_project
    rgb = _rgb_image()
    bgr_v = _viewer(np.ascontiguousarray(rgb[..., ::-1]), "bgr")
    rgb_v = _viewer(rgb, "rgb")
    natural_on_bgr = _sp(r_band=2, g_band=1, b_band=0)          # what the dialog gives on cv2 data
    d = pt._stretch_to_dict(natural_on_bgr, bgr_v)
    assert d["band_basis"] == "file" and (d["r_band"], d["g_band"], d["b_band"]) == (0, 1, 2), d
    sp = pt._sp_to_positions(pt._stretch_from_dict(d), rgb_v)
    assert (sp.r_band, sp.g_band, sp.b_band) == (0, 1, 2)
    sp_back = pt._sp_to_positions(pt._stretch_from_dict(d), bgr_v)
    assert (sp_back.r_band, sp_back.g_band, sp_back.b_band) == (2, 1, 0)


def test_the_legacy_2_1_0_entry_is_natural_colour_on_a_file_order_image(synthetic_project, tmp_path):
    """THE BUG-024 case: a legacy .ax entry (no band_basis) with r/g/b 2/1/0,
    drawn on a file-order array, must show file Red as red."""
    pt = synthetic_project
    rgb = _rgb_image()
    fp = str(tmp_path / "legacy.tif")
    v = _viewer(rgb, "rgb", fp=fp)
    old_folder = pt.project_folder
    try:
        pt.project_folder = str(tmp_path)
        with open(os.path.join(str(tmp_path), "legacy.ax"), "w", encoding="utf-8") as f:
            json.dump({"viewer_stretch": {"mode": "percentile", "low_p": 1.0, "high_p": 99.0,
                                          "display_mode": "rgb", "r_band": 2, "g_band": 1, "b_band": 0}}, f)
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
        out = np.asarray(pt._render_with_viewer_stretch(rgb, v, _as_array=True), dtype=np.float64)
    finally:
        pt.project_folder = old_folder
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
    red_on_screen = out[..., 2]                       # the array goes to a BGR pixmap
    c = np.corrcoef(red_on_screen.ravel(), rgb[..., 0].ravel())[0, 1]
    assert c > 0.99, f"screen red correlates {c:.3f} with file Red (BUG-024: it was file Blue)"


def test_a_root_stretch_follows_each_images_own_band_order(synthetic_project):
    pt = synthetic_project
    rgb = _rgb_image()
    made_on = _viewer(np.ascontiguousarray(rgb[..., ::-1]), "bgr")
    params = pt._sp_with_refs(_sp(r_band=2, g_band=1, b_band=0), made_on)
    assert params.band_refs == {"display_band": None, "r_band": 0, "g_band": 1, "b_band": 2}
    # an image loaded in file order, and a preview holding file bands 3, 2, 1 then 0
    other = _viewer(rgb, "rgb")
    sp = pt._sp_to_positions(params, other)
    assert (sp.r_band, sp.g_band, sp.b_band) == (0, 1, 2)
    prev = _viewer(np.zeros((10, 10, 4), np.float32), "rgb", preview_bands=[3, 2, 1, 0])
    sp = pt._sp_to_positions(params, prev)
    assert (sp.r_band, sp.g_band, sp.b_band) == (3, 2, 1), "positions of file bands 0/1/2 in the preview"
    # the shared root-map object is never rewritten
    assert (params.r_band, params.g_band, params.b_band) == (2, 1, 0)


def test_a_derived_band_is_stored_by_name_and_found_wherever_it_sits(synthetic_project, tmp_path):
    pt = synthetic_project
    fp = str(tmp_path / "expr.tif")
    old_folder = pt.project_folder
    try:
        pt.project_folder = str(tmp_path)
        with open(os.path.join(str(tmp_path), "expr.ax"), "w", encoding="utf-8") as f:
            json.dump({"band_enabled": True, "band_expression": "b1>100",
                       "appended_bands": [{"type": "band_expression", "expression": "b2-b3", "index": 0}]}, f)
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
        v = _viewer(np.zeros((8, 8, 6), np.float32), "rgb", fp=fp)     # 4 file bands + 2 derived
        assert pt._display_band_map(v) == [("file", 0), ("file", 1), ("file", 2), ("file", 3),
                                           ("derived", "b1>100"), ("derived", "b2-b3")]
        d = pt._stretch_to_dict(_sp(display_mode="single", display_band=4), v)
        assert d["display_band"] == "b1>100" and d["band_basis"] == "file"
        cv = _viewer(np.zeros((8, 8, 5), np.float32), "bgr", fp=fp)    # cv2: 3 file bands + 2 derived
        sp = pt._sp_to_positions(pt._stretch_from_dict(d), cv)
        assert sp.display_band == 3
    finally:
        pt.project_folder = old_folder
        if hasattr(pt, "_ax_cache"):
            pt._ax_cache.clear()
