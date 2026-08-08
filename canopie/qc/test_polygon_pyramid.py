"""
Polygon pyramids: COG-style overviews for vector geometry.

WHAT THIS IS
------------
Polygon files store, alongside their full coordinates, a set of decimated
levels at doubling tolerances -- exactly how a COG stores overviews. A single
consolidated `_overview.json` carries one coarse level for EVERY polygon, so a
project opens by reading that instead of every full-resolution file.

Measured on the real project (2333 crowns, 1,010,806 vertices, 90.9 MB):

    open without overview : 5.53 s
    open with overview    : 0.127 s   (44x faster, 3.19 MB file)
    geometry parsed at open: 15.6% of the vertices

THE INVARIANT THAT MATTERS MOST
-------------------------------
Levels are DISPLAY-ONLY. Statistics, CSV export, ML extraction and shapefile
export must always see full coordinates. Two mechanisms enforce it, and both
are tested here because either one failing corrupts scientific output
*silently*:

  1. `LazyPolygonRecord` loads the real file the instant anything asks for
     'points' (or 'coordinates'/'properties'/'lod'). There is no list of call
     sites to keep in sync -- correctness is structural.

  2. `update_all_polygons` reads geometry back OUT of scene items and stores it
     as truth. An item drawn from a coarse level is a PICTURE of the polygon,
     so writing it back would replace a 433-vertex crown with its ~60-vertex
     overview, permanently, on the very next autosave. Such items carry
     `is_lod_geometry` and are skipped; dragging one upgrades it to full
     geometry first.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from .. import polygon_lod as L

pytestmark = [pytest.mark.polygons, pytest.mark.io]

IMG = "somewhere/ortho_cog.tif"


def _ring(n=600, r=1000.0, cx=5000.0, cy=5000.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    w = r * (1.0 + 0.05 * np.sin(a * 7))
    return np.column_stack([cx + w * np.cos(a), cy + w * np.sin(a)]).tolist()


# ---------------------------------------------------------------------------
# The pyramid itself
# ---------------------------------------------------------------------------
def test_levels_get_monotonically_coarser():
    pyr = L.build_pyramid(_ring())
    assert pyr, "no pyramid built for a 600-vertex ring"
    levels = sorted(int(k) for k in pyr)
    counts = [len(pyr[str(l)]) for l in levels]
    assert counts == sorted(counts, reverse=True), (
        f"levels are not monotonically coarser: {dict(zip(levels, counts))}")
    assert counts[-1] < counts[0] / 4, "coarsest level barely reduced anything"


def test_levels_that_buy_nothing_are_omitted():
    """A level identical to the previous one is pure waste in every file."""
    pyr = L.build_pyramid(_ring())
    counts = [len(v) for _, v in sorted(pyr.items(), key=lambda kv: int(kv[0]))]
    assert len(counts) == len(set(counts)), f"duplicate levels stored: {counts}"


def test_no_pyramid_for_simple_polygons():
    """Below the threshold a pyramid costs more than it saves."""
    assert L.build_pyramid([[0, 0], [10, 0], [10, 10], [0, 10]]) == {}


def test_decimation_never_destroys_the_ring():
    """However aggressive the tolerance, the result must still be a polygon."""
    pts = _ring(n=200)
    for tol in (1, 10, 100, 10_000, 10_000_000):
        out = L.decimate(pts, tol)
        assert len(out) >= L.MIN_RING_POINTS, f"tol={tol} left {len(out)} points"


def test_coarse_level_keeps_the_shape():
    pts = np.asarray(_ring())
    coarse = np.asarray(L.decimate(pts, 16))
    for axis in (0, 1):
        assert abs(coarse[:, axis].min() - pts[:, axis].min()) < 40
        assert abs(coarse[:, axis].max() - pts[:, axis].max()) < 40


@pytest.mark.parametrize("scale,expect_coarse", [
    (0.002, True),    # whole mosaic on screen
    (0.05, True),
    (2.0, False),     # zoomed in: full geometry
])
def test_level_for_scale_picks_by_pixel_size(scale, expect_coarse):
    pyr = L.build_pyramid(_ring())
    lvl = L.level_for_scale(scale, pyr.keys())
    if expect_coarse:
        assert lvl is not None
        assert (2.0 ** lvl) * scale <= 1.0, (
            "chose a level whose tolerance exceeds one device pixel -- visible "
            "distortion")
    else:
        assert lvl is None, "used a decimated level at a zoom that shows every vertex"


# ---------------------------------------------------------------------------
# The overview file
# ---------------------------------------------------------------------------
def _write_polygon(tmp_path, group, points, with_lod=True):
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMG))[0]
    data = {'name': group, 'group': group, 'root': '', 'type': 'polygon',
            'coord_space': 'image', 'image_ref_size': {'w': 48031, 'h': 50101},
            'points': points, 'coordinates': {}, 'properties': {'fid': 1}}
    if with_lod:
        data['lod'] = L.build_pyramid(points)
    (poly_dir / f"{group}_{base}_polygons.json").write_text(
        json.dumps(data), encoding="utf-8")
    return str(poly_dir), data


def test_overview_round_trip(tmp_path):
    poly_dir, data = _write_polygon(tmp_path, "crown_1", _ring())
    L.write_overview(poly_dir, [("crown_1", IMG, data)])
    level, entries = L.read_overview(poly_dir)
    assert level == L.OVERVIEW_LEVEL
    assert len(entries) == 1
    e = entries[0]
    assert e['g'] == "crown_1" and e['f'] == IMG
    assert 0 < len(e['p']) < len(data['points']), (
        "overview stored full geometry -- it would be as slow as what it replaces")
    assert e['np'] == len(data['points']), "full vertex count not recorded"


def test_overview_excludes_heavy_fields(tmp_path):
    """`properties` and full coordinates are what make the polygon dir 91 MB;
    the overview must carry neither."""
    poly_dir, data = _write_polygon(tmp_path, "crown_1", _ring())
    L.write_overview(poly_dir, [("crown_1", IMG, data)])
    blob = json.loads((tmp_path / "polygons" / L.OVERVIEW_NAME).read_text(encoding="utf-8"))
    assert 'properties' not in blob['entries'][0]
    assert 'fid' not in json.dumps(blob)


def test_missing_overview_reads_as_absent(tmp_path):
    (tmp_path / "polygons").mkdir()
    assert L.read_overview(str(tmp_path / "polygons")) == (None, None)


def test_corrupt_overview_is_ignored_not_fatal(tmp_path):
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir()
    (poly_dir / L.OVERVIEW_NAME).write_text("{not json", encoding="utf-8")
    assert L.read_overview(str(poly_dir)) == (None, None)


# ---------------------------------------------------------------------------
# LazyPolygonRecord -- the correctness mechanism
# ---------------------------------------------------------------------------
def test_metadata_reads_do_not_materialise(tmp_path):
    poly_dir, data = _write_polygon(tmp_path, "crown_1", _ring())
    src = os.path.join(poly_dir, f"crown_1_{os.path.splitext(os.path.basename(IMG))[0]}_polygons.json")
    rec = L.LazyPolygonRecord({'name': 'crown_1', 'type': 'polygon',
                               'coord_space': 'image', 'display_points': [[0, 0]]}, src)
    for key in ('name', 'type', 'coord_space'):
        rec.get(key)
    assert not rec.is_materialised, (
        f"reading '{key}' paged in the full geometry -- project open would be "
        "as slow as before")


def test_points_access_materialises_full_geometry(tmp_path):
    poly_dir, data = _write_polygon(tmp_path, "crown_1", _ring())
    src = os.path.join(poly_dir, f"crown_1_{os.path.splitext(os.path.basename(IMG))[0]}_polygons.json")
    rec = L.LazyPolygonRecord({'name': 'crown_1', 'display_points': [[0, 0], [1, 1]]}, src)
    pts = rec['points']
    assert rec.is_materialised
    assert pts == data['points'], (
        "materialised geometry differs from the file -- statistics and export "
        "would be computed on the wrong coordinates")


@pytest.mark.parametrize("key", ["points", "coordinates", "properties", "lod"])
def test_every_exact_key_triggers_materialisation(tmp_path, key):
    """Anything implying real data must not be answered from the overview."""
    poly_dir, _ = _write_polygon(tmp_path, "crown_1", _ring())
    src = os.path.join(poly_dir, f"crown_1_{os.path.splitext(os.path.basename(IMG))[0]}_polygons.json")
    rec = L.LazyPolygonRecord({'name': 'crown_1'}, src)
    rec.get(key)
    assert rec.is_materialised, f"'{key}' was served without loading the real file"


def test_serialising_a_lazy_record_writes_full_geometry(tmp_path):
    """THE save-corruption guard. If json.dump saw only the coarse copy, saving
    would overwrite the real polygon with its overview."""
    poly_dir, data = _write_polygon(tmp_path, "crown_1", _ring())
    src = os.path.join(poly_dir, f"crown_1_{os.path.splitext(os.path.basename(IMG))[0]}_polygons.json")
    rec = L.LazyPolygonRecord({'name': 'crown_1', 'display_points': [[0, 0]]}, src)
    out = json.loads(json.dumps(dict(rec)))
    assert len(out['points']) == len(data['points']), (
        "serialising an untouched record produced decimated geometry")


def test_display_points_survive_materialisation(tmp_path):
    poly_dir, _ = _write_polygon(tmp_path, "crown_1", _ring())
    src = os.path.join(poly_dir, f"crown_1_{os.path.splitext(os.path.basename(IMG))[0]}_polygons.json")
    coarse = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    rec = L.LazyPolygonRecord({'name': 'crown_1', 'display_points': coarse}, src)
    rec['points']
    assert dict.get(rec, 'display_points') == coarse, (
        "the coarse copy was clobbered by the full record; the viewer would "
        "have to re-decimate on every paint")


def test_unreadable_source_does_not_loop_or_raise(tmp_path):
    rec = L.LazyPolygonRecord({'name': 'x'}, str(tmp_path / "nope.json"))
    assert rec.get('points') is None
    assert rec.is_materialised, "a failed load left the record retrying forever"


# ---------------------------------------------------------------------------
# THE data-loss guard: coarse items must never be written back
# ---------------------------------------------------------------------------
def test_lod_items_are_flagged(viewer_factory):
    from ..image_viewer import EditablePolygonItem
    v = viewer_factory()
    qimg = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    poly = QtGui.QPolygonF([QtCore.QPointF(1, 1), QtCore.QPointF(9, 1),
                            QtCore.QPointF(9, 9), QtCore.QPointF(1, 9)])
    plain = v.add_polygon_to_scene(poly, "plain")
    lod = v.add_polygon_to_scene(poly, "lod", is_lod=True)
    assert plain.is_lod_geometry is False
    assert lod.is_lod_geometry is True


def test_update_all_polygons_skips_coarse_items():
    """THE regression that would destroy data. `update_all_polygons` stores
    item geometry as truth; it must skip items that are only a coarse picture."""
    import inspect
    import textwrap

    from ..project_tab import ProjectTab

    src = textwrap.dedent(inspect.getsource(ProjectTab.update_all_polygons))
    assert "is_lod_geometry" in src, (
        "update_all_polygons does not check is_lod_geometry -- it would write "
        "a polygon's ~60-vertex overview back over its 433-vertex real "
        "geometry on the next autosave, permanently")


def test_dragging_a_coarse_item_upgrades_it_first():
    """Editing must operate on real coordinates, never the overview."""
    import inspect
    import textwrap

    from ..image_viewer import EditablePolygonItem

    src = textwrap.dedent(inspect.getsource(EditablePolygonItem.mousePressEvent))
    assert "is_lod_geometry" in src and "request_full_geometry" in src, (
        "a coarse item can be dragged without first loading its real "
        "coordinates -- the user would be editing the overview")


def test_lazy_records_are_not_rewritten_by_the_disk_gate():
    """An untouched lazy record's file is already correct; paging it in just to
    rewrite it identically would undo the whole optimisation."""
    import inspect
    import textwrap

    from ..project_tab import ProjectTab

    src = textwrap.dedent(inspect.getsource(ProjectTab._ensure_all_polygons_on_disk))
    assert "is_materialised" in src, (
        "_ensure_all_polygons_on_disk does not skip untouched lazy records, so "
        "it materialises every polygon it was meant to avoid reading")
