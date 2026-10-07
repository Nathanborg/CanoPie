"""
BUG-020 / P4 (2026-10-06): the post-.ax size helpers used when copying or
replicating polygons to other images.

  * `_raw_hw_quick` was declared without `self`; every call through the
    instance raised TypeError.
  * The copy helpers trusted the .ax's `orig_size` (stale after a preview
    stamped it), simulated the geometry their own way (no px_w / px_h, no
    crop/rotate order rule, no enabled flags) and returned the RAW size as the
    post-.ax size whenever the two disagreed by more than 5 % -- after any
    real crop.

The size is now ONE definition: ax_geometry on the raster's header size, and it
must equal the exported image's own size.
"""
import json
import os

import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.extraction]

NAME = "rgb_8bit_untiled"           # 64 x 64, no .ax of its own

CASES = {
    "crop": {"crop_enabled": True, "crop_rect": {"x": 8, "y": 6, "width": 40, "height": 30},
             "crop_rect_ref_size": {"w": 64, "h": 64}},
    "crop_rotate": {"crop_enabled": True, "crop_rect": {"x": 8, "y": 6, "width": 40, "height": 30},
                    "crop_rect_ref_size": {"w": 64, "h": 64}, "rotate_enabled": True, "rotate": 90},
    "px_w": {"resize_enabled": True, "resize": {"px_w": 50}},
    "crop_disabled": {"crop_enabled": False, "crop_rect": {"x": 8, "y": 6, "width": 40, "height": 30},
                      "crop_rect_ref_size": {"w": 64, "h": 64}},
    "stale_orig_size": {"orig_size": {"h": 10, "w": 12, "c": 3}, "crop_enabled": True,
                        "crop_rect": {"x": 0, "y": 0, "width": 32, "height": 32},
                        "crop_rect_ref_size": {"w": 64, "h": 64}},
}


def _clear(pt):
    for attr in ("_export_image_cache", "_export_cache", "_ax_cache", "_ax_json_cache", "_copy_size_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def test_raw_hw_quick_reads_the_header(synthetic_project):
    assert synthetic_project._raw_hw_quick(fixture_image_path(NAME)) == (64, 64)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_post_ax_size_is_the_exported_images_size(synthetic_project, case):
    pt = synthetic_project
    fp = fixture_image_path(NAME)
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: no .ax"
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(CASES[case], f)
        _clear(pt)
        img, info = pt._get_export_image(fp)
        want = (int(info["H"]), int(info["W"]))
        assert pt._size_after_ax_fast(fp) == want, f"{case}: {pt._size_after_ax_fast(fp)} != export {want}"
        assert pt.post_ax_frame_hw_for_file(fp) == want
    finally:
        if os.path.exists(axp):
            os.remove(axp)
        _clear(pt)
