"""
QC tests: growing / shrinking a selected polygon with + / - (ProjectTab.resize_selected_polygons).

THE BUG THIS PINS ("+ and - delete polygons in the BCI project"): the redraw
after a resize scaled the new outline into the size of the PREVIEW the viewer
holds, then mapped it with image_to_scene_coords -- which expects FULL-raster
coordinates and scales down again. On a plain image the preview is the full
image and both factors are 1; on a COG (BCI: a 48,031 x 50,101 raster shown as
a ~1/16 preview) the outline was redrawn ~(1/16)^2 its size in the image's
top-left corner -- gone from where the crown is. The resize also rebuilt the
record from scratch (losing species / tag / class ...), kept the stale coarse
overview geometry, and could not be undone.
"""
import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from .generate_fixtures import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]

GROUP = "resize_crown"


@pytest.fixture()
def decimated_viewer(synthetic_project, viewer_factory, force_preview_decimation, monkeypatch):
    """A viewer showing the 200-band fixture as a half-size preview (like a COG), with one crown in it."""
    pt = synthetic_project
    # Earlier tests leave the shared project's refresh timers pending; one firing mid-test would clear and
    # redraw the viewer under us.
    for attr in ("_poly_idle_timer", "_autosave_timer", "_poly_resize_save_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            t.stop()
    if hasattr(pt, "_cancel_all_scheduled_poly_loads"):
        pt._cancel_all_scheduled_poly_loads()
    fp = fixture_image_path("hyperspectral_200band")
    for attr in ("_imgdata_cache", "_pixmap_cache", "_export_image_cache"):
        c = getattr(pt, attr, None)
        if isinstance(c, dict):
            c.clear()
    imgd = pt._imagedata_or_fallback(fp)
    v = viewer_factory()
    v.resize(500, 500)
    v.show()
    v.image_data = imgd
    v.set_image(pt.convert_cv_to_pixmap(imgd.image))
    monkeypatch.setattr(v, "_find_project_owner", lambda: pt)
    H, W = pt.polygon_basis_hw(v)
    assert imgd.image.shape[0] < H, "precondition: the viewer holds a decimated preview"
    rec_v = {"viewer": v, "image_data": imgd}
    # The only viewer (restored afterwards): with no widget focused, resize_selected_polygons
    # acts on the one viewer that has a selection -- this one.
    monkeypatch.setattr(pt, "viewer_widgets", [rec_v])
    cx, cy, r = W / 2.0, H / 2.0, W / 6.0
    square = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    pt.all_polygons[GROUP] = {fp: {"name": "7114", "type": "polygon", "coord_space": "image", "root": "",
                                   "image_ref_size": {"w": W, "h": H}, "points": list(square),
                                   "properties": {"species": "oleifera", "tag": "7114"}, "class": "tree",
                                   "lod": {"3": [[0, 0], [1, 1], [0, 1]]}}}
    pt._poly_norm_index_invalid = True
    pt.load_polygons(v, imgd)
    item = next(i for i in v.scene().items() if getattr(i, "name", None) == GROUP)
    item.setSelected(True)
    try:
        yield pt, v, fp, item, np.array(square), (cx, cy)
    finally:
        pt.all_polygons.pop(GROUP, None)
        pt._poly_norm_index_invalid = True


def _scene_bbox(v):
    items = [i for i in v.scene().items() if getattr(i, "name", None) == GROUP]
    assert len(items) == 1, f"expected exactly one '{GROUP}' item on screen, found {len(items)}"
    return items[0].mapToScene(items[0].polygon).boundingRect()


@pytest.mark.parametrize("sign", [+1, -1])
def test_resize_keeps_the_crown_where_it_is_on_a_decimated_preview(decimated_viewer, sign):
    pt, v, fp, item, square, (cx, cy) = decimated_viewer
    before = _scene_bbox(v)
    pt.resize_selected_polygons(sign * 0.07)
    after = _scene_bbox(v)
    assert after.center().x() == pytest.approx(before.center().x(), abs=0.05 * before.width())
    assert after.center().y() == pytest.approx(before.center().y(), abs=0.05 * before.height())
    if sign > 0:
        assert after.width() > before.width() * 1.05, "grow did not grow it on screen"
    else:
        assert 0.6 * before.width() < after.width() < before.width() * 0.95, "shrink did not shrink it on screen"
    # the record grew / shrank around the same centre, in full-raster pixels
    pts = np.asarray(pt.all_polygons[GROUP][fp]["points"], float)
    assert not np.allclose(pts[0], pts[-1]), "the closing vertex is stored twice"
    from shapely.geometry import Polygon
    c = Polygon(pts).centroid
    assert (c.x, c.y) == pytest.approx((cx, cy), abs=0.05)
    span, span0 = np.ptp(pts[:, 0]), np.ptp(square[:, 0])
    assert (span > span0) if sign > 0 else (span < span0)


def test_resize_keeps_the_record_and_drops_stale_overview_geometry(decimated_viewer):
    pt, v, fp, item, square, _ = decimated_viewer
    pt.resize_selected_polygons(+0.07)
    rec = pt.all_polygons[GROUP][fp]
    assert rec["properties"] == {"species": "oleifera", "tag": "7114"} and rec["class"] == "tree"
    assert rec["name"] == "7114"
    assert "lod" not in rec and "display_points" not in rec, "the old outline's overview would come back on reload"
    assert item.isSelected(), "the resized crown should stay selected for the next + / -"


def test_resize_can_be_undone_exactly(decimated_viewer):
    pt, v, fp, item, square, _ = decimated_viewer
    before = _scene_bbox(v)
    pt.resize_selected_polygons(+0.07)
    pt.resize_selected_polygons(+0.07)
    pt.undo_stack.undo()
    pt.undo_stack.undo()
    assert np.allclose(pt.all_polygons[GROUP][fp]["points"], square, atol=1e-6)
    after = _scene_bbox(v)
    assert after.width() == pytest.approx(before.width(), rel=1e-3)
