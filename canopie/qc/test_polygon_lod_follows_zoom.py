"""
Polygon detail follows the zoom, QGIS-style (BUG-031).

THE BUG THIS PINS
-----------------
Crowns are drawn from a coarse pyramid level (`polygon_lod`, OVERVIEW_LEVEL =
16 record units) so thousands of them load and paint fast. But
`polygon_lod.level_for_scale` -- the function meant to choose the level for
the current zoom -- was never called: after a reopen every crown kept its
overview outline at EVERY zoom. On the BCI orthomosaic that is up to 16 raw px
(~1 m) off the real crown when zoomed in to native resolution, until the user
tried to drag it. And "Edit Vertices" / copy / replicate read that simplified
outline as if it were the polygon, so editing vertices of an imported crown
saved the overview over the real geometry.

The contract:
  * zoomed in past ~1 device px per record unit, crowns in view show their
    EXACT geometry; at intermediate zooms, the coarsest level that is still
    under one device pixel; never coarser than the overview;
  * the device-px-per-unit includes how the displayed pixmap maps to the
    record basis, so a decimated preview (COG) is handled like a full-res one;
  * this is display only: the record is untouched and update_all_polygons does
    not write the display outline back;
  * Edit Vertices upgrades to the exact geometry first.
"""
import json
import math
import os

import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]

N = 600                 # vertices of the exact outline
REF = 256               # record basis = 4x the 64-px pixmap, like a decimated COG preview
GROUP = "lodzoom_crown"


def _circle(cx=128.0, cy=128.0, r=100.0, n=N):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _record(points):
    return {'name': "crown", 'root': '', 'type': 'polygon', 'coord_space': 'image',
            'image_ref_size': {'w': REF, 'h': REF}, 'points': [tuple(p) for p in points],
            'properties': {'tag': '1'}}


@pytest.fixture
def lod_view(synthetic_project, viewer_factory, monkeypatch, tmp_path):
    from .. import polygon_lod
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    imgd = pt._imagedata_or_fallback(fp)
    viewer = viewer_factory()
    viewer.resize(400, 400)
    viewer.show()
    viewer.image_data = imgd
    pm = pt.convert_cv_to_pixmap(imgd.image)
    viewer.set_image(pm)
    rec = {"viewer": viewer, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec]
    viewer.set_full_geometry_callback(pt._upgrade_item_to_full_geometry)
    monkeypatch.setattr(pt, "get_root_by_filepath", lambda f: None)
    monkeypatch.setattr(pt, "_mark_polygon_dirty", lambda g, f: None)

    def load(lazy):
        pts = _circle()
        if lazy:
            side = tmp_path / f"{GROUP}_side_polygons.json"
            side.write_text(json.dumps(_record(pts)), encoding="utf-8")
            meta = {k: v for k, v in _record(pts).items() if k not in ('points', 'properties')}
            meta['display_points'] = polygon_lod.decimate(pts, 2.0 ** polygon_lod.OVERVIEW_LEVEL).tolist()
            meta['display_level'] = polygon_lod.OVERVIEW_LEVEL
            data = polygon_lod.LazyPolygonRecord(meta, str(side))
        else:
            data = _record(pts)
        pt.all_polygons[GROUP] = {fp: data}
        pt._poly_norm_index_invalid = True
        viewer.clear_polygons()
        pt.load_polygons(viewer, imgd)
        it = next(i for i in viewer.get_all_polygons() if i.name == GROUP)
        return it, data

    try:
        yield pt, viewer, load
    finally:
        try:
            pt.viewer_widgets.remove(rec)
        except Exception:
            pass
        pt.all_polygons.pop(GROUP, None)
        pt._poly_norm_index_invalid = True


def _zoom(viewer, item, s):
    viewer.resetTransform()
    viewer.scale(s, s)
    viewer.centerOn(item)


def _n(item):
    return item.polygon.count()


def _refine(pt, viewer, timeout_s=10.0):
    """Run refine passes the way the viewer does (it re-runs while reads are in
    flight on the background pool) until there is nothing left to do."""
    import time
    from PyQt5 import QtWidgets
    end = time.time() + timeout_s
    while pt._refine_visible_polygon_lod(viewer):
        assert time.time() < end, "refine never settled"
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)


@pytest.mark.parametrize("lazy", [False, True], ids=["in-memory", "lazy-from-disk"])
def test_detail_follows_zoom_and_stays_display_only(lod_view, lazy):
    pt, viewer, load = lod_view
    it, data = load(lazy)
    coarse = _n(it)
    assert it.is_lod_geometry and coarse < N / 4, (
        f"loaded with {coarse} vertices; expected the overview level (a crown of {N} "
        "vertices drawn coarse, so thousands load and paint fast)")

    # Zoomed in: 8 device px per pixmap px = 2 device px per record unit -> exact.
    _zoom(viewer, it, 8.0)
    _refine(pt, viewer)
    assert _n(it) == N, (
        f"zoomed in to 2 device px per record unit the crown still has {_n(it)} of {N} "
        "vertices: the overview outline was kept at every zoom (level_for_scale was "
        "never called), up to 16 record units off the real crown.")
    br_full = it.mapToScene(it.polygon).boundingRect()

    # Intermediate: 0.5 device px per pixmap px = 0.125 per unit -> level 3 (8 units).
    _zoom(viewer, it, 0.5)
    _refine(pt, viewer)
    mid = _n(it)
    assert coarse <= mid < N, f"at 0.125 device px per unit got {mid} vertices (overview {coarse}, full {N})"
    br_mid = it.mapToScene(it.polygon).boundingRect()
    # Same crown, same place (within one level's tolerance, in scene units).
    tol = (2 ** 3) * (64.0 / REF) + 1e-6
    for a, b in ((br_mid.left(), br_full.left()), (br_mid.top(), br_full.top()),
                 (br_mid.right(), br_full.right()), (br_mid.bottom(), br_full.bottom())):
        assert abs(a - b) <= tol, "the refined outline moved"

    # Display only: the record keeps every exact vertex, and a sync must not
    # write the display outline back over it.
    pt.update_all_polygons()
    stored = pt.all_polygons[GROUP][next(iter(pt.all_polygons[GROUP]))]
    assert len(stored['points']) == N and stored.get('properties') == {'tag': '1'}


def test_edit_vertices_upgrades_to_exact_geometry_first(lod_view):
    pt, viewer, load = lod_view
    it, data = load(lazy=True)
    assert it.is_lod_geometry and _n(it) < N
    viewer.start_vertex_editing(it)
    try:
        assert not it.is_lod_geometry and _n(it) == N, (
            f"Edit Vertices started on the {_n(it)}-vertex display outline; finishing the "
            "edit would save it over the crown's real geometry.")
    finally:
        viewer.stop_vertex_editing()
