"""
A polygon drawn with sync on is copied to the other images of the root at the
SAME FRACTION of each frame -- what "copy to root" does (2026-10-07).

add_polygon_to_other_images scaled between images by the size of the array
each viewer DISPLAYS, while the coordinates themselves are in each image's
polygon basis (polygon_basis_hw: the raster's own size, stretched over the
displayed frame). An image shown .ax-resized, cropped or as a decimated preview
displays a smaller array than its basis, so every synced copy was scaled by
that factor -- on New Folder223 / New Folder129 the copies landed at ~6x the
frame, outside the image, and that is what was saved.
"""
import types

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.viewer]


def _viewer(viewer_factory, fp, shape):
    v = viewer_factory()
    h, w = shape
    v.image_data = types.SimpleNamespace(filepath=fp, image=np.zeros((h, w, 3), np.uint8))
    q = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    q.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(q))
    v._image.setPos(0, 0)
    return v


def _fracs_on_screen(v, name):
    pm = v._image.pixmap()
    for it in v._scene.items():
        if getattr(it, "name", None) == name and hasattr(it, "polygon"):
            out = []
            for p in it.polygon:
                q = v._image.mapFromScene(it.mapToScene(p))
                out.append((q.x() / pm.width(), q.y() / pm.height()))
            return out
    return None


@pytest.mark.parametrize("src_shown, tgt_shown", [((48, 48), (64, 64)),     # source decimated / resized
                                                  ((96, 96), (32, 32)),     # target shown smaller
                                                  ((40, 72), (64, 64))],    # source cropped (aspect differs)
                         ids=["source_preview", "target_preview", "source_cropped"])
def test_a_synced_polygon_keeps_its_place_on_every_image(synthetic_project, viewer_factory, monkeypatch,
                                                          src_shown, tgt_shown):
    pt = synthetic_project
    src_fp = fixture_image_path("rgb_16bit_tiled_bip_cog")    # 96 x 96 raster
    tgt_fp = fixture_image_path("rgb_8bit_untiled")           # 64 x 64 raster
    vs = _viewer(viewer_factory, src_fp, src_shown)
    vt = _viewer(viewer_factory, tgt_fp, tgt_shown)
    root = "R_sync"
    monkeypatch.setattr(pt, "all_polygons", {}, raising=False)
    monkeypatch.setattr(pt, "multispectral_image_data_groups", {root: [src_fp, tgt_fp]}, raising=False)
    monkeypatch.setattr(pt, "thermal_rgb_image_data_groups", {}, raising=False)
    monkeypatch.setattr(pt, "root_id_mapping", {root: 1}, raising=False)
    monkeypatch.setattr(pt, "get_root_by_filepath", lambda fp: root)
    monkeypatch.setattr(pt, "get_viewer_by_filepath", lambda fp: {src_fp: vs, tgt_fp: vt}.get(fp))
    monkeypatch.setattr(pt, "_add_to_polygon_index", lambda *a, **k: None)
    assert pt.polygon_basis_hw(vs) == (96, 96) and pt.polygon_basis_hw(vt) == (64, 64), "precondition"

    fr = [(0.25, 0.20), (0.60, 0.20), (0.60, 0.70), (0.25, 0.70)]
    sh, sw = src_shown
    poly = QtGui.QPolygonF([vs._image.mapToScene(QtCore.QPointF(fx * sw, fy * sh)) for fx, fy in fr])
    pt.add_polygon_to_other_images(vs, poly, "sq", action="copy", shape_type="polygon")

    rec = pt.all_polygons["sq"][tgt_fp]
    ref = rec["image_ref_size"]
    assert (ref["w"], ref["h"]) == (64, 64), f"stamped with the displayed size, not the target's basis: {ref}"
    got = [(x / ref["w"], y / ref["h"]) for x, y in rec["points"]]
    assert np.allclose(got, fr, atol=1e-6), f"saved at {got}, drawn at {fr}"
    shown = _fracs_on_screen(vt, "sq")
    assert shown is not None, "nothing drawn on the target viewer"
    assert np.allclose(shown, fr, atol=1e-3), f"drawn on the target at {shown}, on the source at {fr}"


def test_synced_points_keep_their_place_too(synthetic_project, viewer_factory, monkeypatch):
    pt = synthetic_project
    src_fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    tgt_fp = fixture_image_path("rgb_8bit_untiled")
    vs = _viewer(viewer_factory, src_fp, (48, 48))         # shown at half size
    vt = _viewer(viewer_factory, tgt_fp, (64, 64))
    root = "R_sync_pts"
    monkeypatch.setattr(pt, "all_polygons", {}, raising=False)
    monkeypatch.setattr(pt, "multispectral_image_data_groups", {root: [src_fp, tgt_fp]}, raising=False)
    monkeypatch.setattr(pt, "thermal_rgb_image_data_groups", {}, raising=False)
    monkeypatch.setattr(pt, "root_id_mapping", {root: 1}, raising=False)
    monkeypatch.setattr(pt, "get_root_by_filepath", lambda fp: root)
    monkeypatch.setattr(pt, "get_viewer_by_filepath", lambda fp: {src_fp: vs, tgt_fp: vt}.get(fp))
    monkeypatch.setattr(pt, "_add_to_polygon_index", lambda *a, **k: None)
    fr = [(0.25, 0.40), (0.70, 0.55)]
    pts = QtGui.QPolygonF([vs._image.mapToScene(QtCore.QPointF(fx * 48, fy * 48)) for fx, fy in fr])
    pt.add_polygon_to_other_images(vs, pts, "pp", action="copy", shape_type="point")
    rec = pt.all_polygons["pp"][tgt_fp]
    ref = rec["image_ref_size"]
    got = [(x / ref["w"], y / ref["h"]) for x, y in rec["points"]]
    assert rec["type"] == "point" and np.allclose(got, fr, atol=1e-6), (got, fr, ref)
