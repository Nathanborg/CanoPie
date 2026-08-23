"""QC regression tests for find_suspect_point_coordinates -- the DETECT-ONLY
migration helper for point coordinates that were saved by the pre-fix
on_polygon_drawn / on_polygon_modified (see test_point_scene_coordinates.py
for the live-viewer fix this method cleans up after).

WHY THIS CANNOT BE A migrate_polygon_basis-STYLE REPAIR:

migrate_polygon_basis can algebraically fix a mis-stamped polygon because the
file records exactly what basis it was (wrongly) written against
(`image_ref_size`), and that can be diffed against the TRUE raster size and
inverted by a clean scale factor.

The "points go stray on reload" bug is different: `image_ref_size` was
stamped identically by the buggy code and the fixed code (the corruption was
in `on_polygon_drawn`/`on_polygon_modified`'s coordinate RECONSTRUCTION, not
in what basis got recorded). The runtime state that produced a bad point
(which decimated preview level was on screen, and/or mid-drag item.pos(), at
the exact moment of the save) was never persisted anywhere. So there is no
"recorded-wrong-basis" to invert -- only DETECTION is possible:
  1. out of bounds vs image_ref_size -- unambiguous, removable.
  2. far outside every sibling point/polygon on the same image -- a
     heuristic, report-only, never auto-removed.

A second, independent thing this file pins: unlike migrate_polygon_basis's
sidecar schema, a point/polygon entry's JSON body carries NO "filepath" or
"image_path" key of its own -- the filepath is only ever the dict KEY in
`all_polygons[group][filepath]` in memory, and on disk it must be resolved
from the filename ("<group>_<image base>_polygons.json") the same way
migrate_polygon_basis does it: match the longest known image base the stem
ends with. An earlier draft of this method assumed a "filepath" key existed
in the JSON body and would have silently found nothing on every real
project -- see test_matches_the_right_image_for_underscored_group_names.
"""
import json
import os

import pytest

pytestmark = [pytest.mark.polygons]


def _tab(tmp_path):
    from ..project_tab import ProjectTab
    pt = ProjectTab.__new__(ProjectTab)
    pt.project_folder = str(tmp_path)
    pt.all_polygons = {}
    pt.image_data_groups = {}
    pt.multispectral_image_data_groups = {}
    pt.thermal_rgb_image_data_groups = {}
    return pt


def _prepare(tmp_path, fp="img1.tif"):
    tab = _tab(tmp_path)
    tab.image_data_groups = {"Root1": [fp]}
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    return tab, fp, poly_dir


def _write_point_sidecar(poly_dir, group, img_fp, pts, ref_w, ref_h):
    base = os.path.splitext(os.path.basename(img_fp))[0]
    path = poly_dir / f"{group}_{base}_polygons.json"
    path.write_text(json.dumps({
        'name': group, 'root': '1', 'type': 'point',
        'coord_space': 'image',
        'image_ref_size': {'w': ref_w, 'h': ref_h},
        'points': pts,
    }), encoding="utf-8")
    return path


def _write_polygon_sidecar(poly_dir, group, img_fp, pts, ref_w, ref_h):
    base = os.path.splitext(os.path.basename(img_fp))[0]
    path = poly_dir / f"{group}_{base}_polygons.json"
    path.write_text(json.dumps({
        'name': group, 'root': '1', 'type': 'polygon',
        'coord_space': 'image',
        'image_ref_size': {'w': ref_w, 'h': ref_h},
        'points': pts,
    }), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Filepath resolution (the bug in the first draft of this method)
# ---------------------------------------------------------------------------
def test_matches_the_right_image_for_underscored_group_names(tmp_path):
    """Group names contain underscores, so the image cannot be found by
    splitting the filename -- it must match a KNOWN image base, exactly like
    migrate_polygon_basis. If find_suspect_point_coordinates instead looked
    for a "filepath"/"image_path" KEY inside the JSON body, it would find
    nothing here (that key does not exist in real sidecars) and silently
    report zero suspects even though this point is 10x out of bounds."""
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_point_sidecar(poly_dir, "guayacan_21294.0_crown", fp,
                         [[9999.0, 9999.0]], 100, 100)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_checked == 1
    assert n_suspect == 1, (
        "the sidecar's image was not resolved -- filepath must come from "
        "filename-stem matching against known_bases, not a nonexistent "
        "'filepath' key in the JSON body")
    assert details[0]["filepath"] == fp


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_out_of_bounds_point_is_flagged(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_point_sidecar(poly_dir, "pts", fp, [[50.0, 50.0], [500.0, 500.0]], 100, 100)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_checked == 2
    assert n_suspect == 1
    assert details[0]["reason"].startswith("out of bounds")
    assert details[0]["point_index"] == 1


def test_plausible_in_bounds_point_is_not_flagged(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_point_sidecar(poly_dir, "pts", fp,
                         [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0]], 100, 100)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_checked == 3
    assert n_suspect == 0, details


def test_negative_coordinate_is_out_of_bounds(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_point_sidecar(poly_dir, "pts", fp, [[-5.0, 10.0]], 100, 100)

    _, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_suspect == 1
    assert details[0]["reason"].startswith("out of bounds")


def test_sibling_heuristic_flags_a_far_but_in_bounds_outlier(tmp_path):
    """A tight cluster of points plus one point that's in-bounds but far
    from every other one on the same image -- report-only."""
    tab, fp, poly_dir = _prepare(tmp_path)
    cluster = [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0], [9.0, 12.0]]
    outlier = [[950.0, 950.0]]
    _write_point_sidecar(poly_dir, "pts", fp, cluster + outlier, 1000, 1000)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_suspect == 1
    assert "far outside" in details[0]["reason"]
    assert details[0]["point_index"] == 4


def test_sibling_extent_is_built_across_files_including_polygons(tmp_path):
    """The "sibling" comparison must consider every OTHER polygon/point
    entry on the same image, not just entries within the same file."""
    tab, fp, poly_dir = _prepare(tmp_path)
    # A polygon elsewhere on the image establishes the plausible extent.
    _write_polygon_sidecar(poly_dir, "crown_a", fp,
                           [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
                           1000, 1000)
    # A point far outside that extent, but still in-bounds.
    _write_point_sidecar(poly_dir, "pts", fp, [[900.0, 900.0]], 1000, 1000)

    _, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_suspect == 1
    assert "far outside" in details[0]["reason"]


def test_out_of_bounds_siblings_do_not_poison_the_extent(tmp_path):
    """An out-of-bounds point must not widen the "plausible extent" used to
    judge other points -- otherwise one corrupted point would mask another."""
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_point_sidecar(poly_dir, "corrupt", fp, [[99999.0, 99999.0]], 1000, 1000)
    _write_point_sidecar(poly_dir, "far_but_inbounds", fp, [[900.0, 900.0]], 1000, 1000)
    # Establish a tight, plausible extent via a third entry.
    _write_point_sidecar(poly_dir, "cluster", fp,
                         [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0]], 1000, 1000)

    _, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    reasons = {d["group_name"]: d["reason"] for d in details}
    assert reasons["corrupt"].startswith("out of bounds")
    assert "far outside" in reasons["far_but_inbounds"], (
        "the out-of-bounds point must have been excluded from the extent "
        "used to judge the in-bounds-but-far point, or this would pass "
        "silently")


def test_polygon_entries_are_never_flagged_only_points_are(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    _write_polygon_sidecar(poly_dir, "crown", fp, [[9999.0, 9999.0]], 100, 100)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_checked == 0
    assert n_suspect == 0


# ---------------------------------------------------------------------------
# dry_run / remove_flagged semantics
# ---------------------------------------------------------------------------
def test_dry_run_writes_nothing(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    path = _write_point_sidecar(poly_dir, "pts", fp, [[9999.0, 9999.0]], 100, 100)
    before = path.read_text(encoding="utf-8")

    tab.find_suspect_point_coordinates(dry_run=True, remove_flagged=True)
    assert path.read_text(encoding="utf-8") == before, (
        "dry_run=True must never write, even if remove_flagged=True")


def test_remove_flagged_deletes_only_the_out_of_bounds_point(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    cluster = [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0], [9.0, 12.0]]
    path = _write_point_sidecar(poly_dir, "pts", fp,
                                cluster + [[9999.0, 9999.0]], 100, 100)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(
        dry_run=False, remove_flagged=True)
    assert n_suspect == 1
    assert details[0]["removed"] is True

    got = json.loads(path.read_text(encoding="utf-8"))
    assert len(got["points"]) == 4
    assert [9999.0, 9999.0] not in got["points"]
    assert got["points"] == cluster


def test_remove_flagged_never_touches_sibling_heuristic_points(tmp_path):
    """Only the unambiguous (out-of-bounds) case is ever auto-removed --
    the sibling heuristic is advisory only, per the plan."""
    tab, fp, poly_dir = _prepare(tmp_path)
    cluster = [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0]]
    outlier = [950.0, 950.0]
    path = _write_point_sidecar(poly_dir, "pts", fp, cluster + [outlier], 1000, 1000)

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(
        dry_run=False, remove_flagged=True)
    assert n_suspect == 1
    assert details[0]["removed"] is False, (
        "a far-but-in-bounds point must never be auto-deleted")

    got = json.loads(path.read_text(encoding="utf-8"))
    assert outlier in got["points"]
    assert len(got["points"]) == 4


def test_entry_emptied_by_removal_deletes_the_file(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    path = _write_point_sidecar(poly_dir, "pts", fp, [[9999.0, 9999.0]], 100, 100)

    tab.find_suspect_point_coordinates(dry_run=False, remove_flagged=True)
    assert not path.exists(), (
        "an entry with zero points remaining after removal must be "
        "deleted, not left behind as an empty 'points': []")


def test_report_only_mode_flags_but_does_not_remove(tmp_path):
    """remove_flagged=False (the default) must report every suspect point,
    including out-of-bounds ones, without writing anything."""
    tab, fp, poly_dir = _prepare(tmp_path)
    path = _write_point_sidecar(poly_dir, "pts", fp, [[9999.0, 9999.0]], 100, 100)
    before = path.read_text(encoding="utf-8")

    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=False)
    assert n_suspect == 1
    assert details[0]["removed"] is False
    assert path.read_text(encoding="utf-8") == before


def test_rerun_after_removal_finds_nothing_left(tmp_path):
    tab, fp, poly_dir = _prepare(tmp_path)
    cluster = [[10.0, 10.0], [12.0, 11.0], [11.0, 13.0]]
    _write_point_sidecar(poly_dir, "pts", fp, cluster + [[9999.0, 9999.0]], 100, 100)

    tab.find_suspect_point_coordinates(dry_run=False, remove_flagged=True)
    n_checked, n_suspect, _ = tab.find_suspect_point_coordinates(dry_run=True)
    assert n_suspect == 0
    assert n_checked == 3


def test_no_polygons_dir_returns_empty_result(tmp_path):
    tab = _tab(tmp_path)
    n_checked, n_suspect, details = tab.find_suspect_point_coordinates(dry_run=True)
    assert (n_checked, n_suspect) == (0, 0)


def test_underscore_prefixed_files_are_skipped(tmp_path):
    """Same convention as migrate_polygon_basis -- e.g. an overview/index
    sidecar prefixed with "_" must not be scanned as a polygon file."""
    tab, fp, poly_dir = _prepare(tmp_path)
    (poly_dir / "_overview_polygons.json").write_text(json.dumps({
        'name': 'ignored', 'type': 'point',
        'image_ref_size': {'w': 10, 'h': 10},
        'points': [[9999.0, 9999.0]],
    }), encoding="utf-8")

    n_checked, n_suspect, _ = tab.find_suspect_point_coordinates(dry_run=True)
    assert (n_checked, n_suspect) == (0, 0)
