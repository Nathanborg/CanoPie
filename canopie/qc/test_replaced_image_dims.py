"""
An image replaced on disk while CanoPie runs must be measured again.

The image-size authority (shapefile_io._raw_image_dims) was cached by PATH.
On a real phenocam project an exported crop (1197x923) was copied over its
6080x3420 original, keeping the name; for the rest of the session every
reader of that cache -- the polygon basis, Inspect, the crown finder,
shapefile export -- still used 6080x3420. A crown drawn on the new image was
stored as if on the old one and Inspect called it outside the image.
"""
import os

import cv2
import numpy as np


def _write(fp, h, w, mtime):
    cv2.imwrite(fp, np.full((h, w, 3), 128, np.uint8))
    os.utime(fp, (mtime, mtime))


def test_raw_dims_follow_a_replaced_file(tmp_path):
    from ..shapefile_io import _raw_image_dims
    fp = str(tmp_path / "WSCT1803.JPG")
    _write(fp, 342, 608, 1_700_000_000)
    assert _raw_image_dims(fp) == (342, 608)

    _write(fp, 92, 119, 1_700_000_100)          # an exported crop copied over it
    assert _raw_image_dims(fp) == (92, 119)


def test_polygon_basis_follows_a_replaced_file(tmp_path):
    from ..project_tab import ProjectTab
    fp = str(tmp_path / "frame.jpg")
    _write(fp, 342, 608, 1_700_000_000)

    class _ImageData:
        filepath = fp
        image = np.zeros((342, 608, 3), np.uint8)

    pt = ProjectTab.__new__(ProjectTab)
    assert pt.polygon_basis_hw(None, _ImageData) == (342, 608)

    _write(fp, 92, 119, 1_700_000_100)
    _ImageData.image = np.zeros((92, 119, 3), np.uint8)
    assert pt.polygon_basis_hw(None, _ImageData) == (92, 119), (
        "the polygon basis kept the replaced file's old size: crowns drawn now "
        "would be stored in a frame the image no longer has")


def test_cache_clear_still_works(tmp_path):
    from ..shapefile_io import _raw_image_dims, clear_shapefile_io_caches
    fp = str(tmp_path / "a.png")
    _write(fp, 10, 20, 1_700_000_000)
    assert _raw_image_dims(fp) == (10, 20)
    _raw_image_dims.cache_clear()
    clear_shapefile_io_caches()
    assert _raw_image_dims(fp) == (10, 20)
