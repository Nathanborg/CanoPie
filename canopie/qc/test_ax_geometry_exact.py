"""
ax_geometry predicts the exported frame EXACTLY (size and pixel mapping).

Points are snapped to -- and drawn as -- one pixel of the exported frame (the
frame CSV export and the ML manager sample). That is only right if
ax_geometry.frame_plan reproduces ProjectTab._apply_ax_to_raw: the same frame
size to the pixel, and a forward map that lands every original pixel on the
export pixel holding its value. Checked here against the real export pipeline
on a non-square raster for every crop / rotate / resize mode and combination.
"""
import json
import os

import numpy as np
import pytest

from .. import ax_geometry as G

pytestmark = [pytest.mark.extraction, pytest.mark.contract]

RAW_W, RAW_H = 70, 53

VARIANTS = {
    "none": {},
    "crop": {"crop_rect": {"x": 7, "y": 5, "width": 40, "height": 30},
             "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}},
    "crop_on_half_preview": {"crop_rect": {"x": 3, "y": 2, "width": 21, "height": 15},
                             "crop_rect_ref_size": {"w": 35, "h": 27}},
    "rot90": {"rotate": 90},
    "rot180": {"rotate": 180},
    "rot270": {"rotate": 270},
    "rot90_crop_after": {"rotate": 90, "crop_rect": {"x": 4, "y": 9, "width": 30, "height": 41},
                         "crop_rect_ref_size": {"w": RAW_H, "h": RAW_W}},
    "rot90_crop_before": {"rotate": 90, "crop_rect": {"x": 6, "y": 3, "width": 44, "height": 35},
                          "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}},
    "resize_scale50": {"resize": {"scale": 50}},
    "resize_pct": {"resize": {"width": 75, "height": 60}},
    "resize_px_w": {"resize": {"px_w": 33}},
    "resize_px_h": {"resize": {"px_h": 20}},
    "resize_px_both": {"resize": {"px_w": 31, "px_h": 29}},
    "crop_resize": {"crop_rect": {"x": 7, "y": 5, "width": 40, "height": 30},
                    "crop_rect_ref_size": {"w": RAW_W, "h": RAW_H}, "resize": {"scale": 150}},
    "rot270_crop_resize": {"rotate": 270, "crop_rect": {"x": 2, "y": 8, "width": 37, "height": 50},
                           "crop_rect_ref_size": {"w": RAW_H, "h": RAW_W}, "resize": {"px_w": 25}},
    "disabled_flags": {"crop_rect": {"x": 7, "y": 5, "width": 40, "height": 30}, "crop_enabled": False,
                       "rotate": 90, "rotate_enabled": False, "resize": {"scale": 50}, "resize_enabled": False},
}


@pytest.fixture(scope="module")
def raster(tmp_path_factory):
    import tifffile
    d = tmp_path_factory.mktemp("axgeom")
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 60000, size=(RAW_H, RAW_W, 3), dtype=np.uint16)
    p = str(d / "axgeom_nonsquare.tif")
    tifffile.imwrite(p, arr, photometric="rgb")
    return p, arr


def _export(pt, fp, ax):
    axp = pt._ax_path_for(fp)
    with open(axp, "w", encoding="utf-8") as f:
        json.dump(ax, f)
    for attr in ("_export_image_cache", "_ax_cache", "_ax_json_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    if hasattr(pt, "_invalidate_cached_ax"):
        pt._invalidate_cached_ax(fp)
    try:
        img, info = pt._get_export_image(fp)
        return np.asarray(img), (int(info["H"]), int(info["W"]))
    finally:
        os.remove(axp)
        for attr in ("_export_image_cache", "_ax_cache", "_ax_json_cache"):
            c = getattr(pt, attr, None)
            if isinstance(c, dict):
                c.clear()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_frame_size_matches_the_export_pipeline(synthetic_project, raster, name):
    fp, _ = raster
    img, (H, W) = _export(synthetic_project, fp, VARIANTS[name])
    assert G.frame_hw(VARIANTS[name], RAW_W, RAW_H) == (H, W) == img.shape[:2], (
        f"{name}: ax_geometry predicts {G.frame_hw(VARIANTS[name], RAW_W, RAW_H)}, export produced "
        f"{(H, W)} -- a point snapped to the predicted grid would read a neighbouring pixel")


@pytest.mark.parametrize("name", [n for n, a in VARIANTS.items() if "resize" not in a or a.get("resize_enabled") is False])
def test_forward_map_lands_on_the_same_pixel_value(synthetic_project, raster, name):
    """Without resampling, export pixel floor(forward(raw centre)) must hold
    exactly the raw pixel's value."""
    fp, raw = raster
    img, (H, W) = _export(synthetic_project, fp, VARIANTS[name])
    ops, _ = G.frame_plan(VARIANTS[name], RAW_W, RAW_H)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(400):
        rx, ry = int(rng.integers(0, RAW_W)), int(rng.integers(0, RAW_H))
        (ex, ey), = G.forward(ops, [(rx + 0.5, ry + 0.5)])
        i, j = int(np.floor(ex)), int(np.floor(ey))
        if not (0 <= i < W and 0 <= j < H):
            continue                      # cropped away
        hits += 1
        assert np.array_equal(img[j, i, :3], raw[ry, rx, :3]), (
            f"{name}: raw ({rx},{ry}) -> export ({i},{j}) holds {img[j, i, :3]} not {raw[ry, rx, :3]}")
    assert hits > 50


@pytest.mark.parametrize("name", list(VARIANTS))
def test_inverse_undoes_forward(name):
    ops, _ = G.frame_plan(VARIANTS[name], RAW_W, RAW_H)
    pts = [(0.5, 0.5), (12.25, 40.75), (69.5, 52.5), (33.0, 0.0)]
    back = G.inverse(ops, G.forward(ops, pts))
    assert np.allclose(back, pts, atol=1e-9)
