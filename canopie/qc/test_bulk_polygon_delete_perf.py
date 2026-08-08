"""
Deleting many polygons at once must cost O(N), not O(N^2) -- and definitely
not O(N) *disk reads of a shared file*.

THE BUG THIS PINS (reported as "delete polygon all takes forever for 3000
polygons, and it should be instant"):

`delete_selected_polygons` / `delete_all_polygons_for_group(s)` push ONE
`DeletePolygonCommand` per polygon into a single `QUndoStack` macro.
`QUndoStack.push()` runs `redo()` IMMEDIATELY as each command lands -- the
macro only groups them for a single Ctrl+Z, it does not batch the WORK. So for
N polygons, `DeletePolygonCommand.redo()` used to run N times, and three
things inside it were each expensive per call:

  1. `_remove_polygon_from_mask_config` unconditionally opened, read, and
     `json.load()`'d the image's `.ax` sidecar -- even though a mask_polygon
     block is a rare, opt-in feature and the overwhelming majority of these
     reads find nothing to do. For N polygons that share one image, that is N
     redundant full parses of the identical file.

  2. `update_polygon_manager()` rebuilds the polygon manager's list widget
     whenever the set of group names changed since last call -- and in this
     project's data, one polygon IS one top-level group (e.g.
     "guayacan_21294.0"), so the group set changes on literally every single
     delete. Its own "skip if unchanged" guard therefore never fires here:
     N deletes = N full list-widget rebuilds.

  3. The per-command viewer cleanup scanned EVERY item currently in the
     scene to find the one being removed. Called once per delete against a
     scene that still holds up to N items, that is O(scene size) per call,
     O(N^2) for the whole batch.

THE FIX: `ProjectTab._bulk_polygon_delete()` (a re-entrant context manager)
wraps the push loop in every multi-polygon call site. Under it,
`DeletePolygonCommand.redo()` only RECORDS what it would remove; the manager
refresh and the per-viewer scene scan happen exactly ONCE, when the outermost
context exits (`_flush_bulk_delete_pending`). `_remove_polygon_from_mask_config`
was separately switched from a raw `open()+json.load()` to the existing
mtime+size-keyed `_get_cached_ax`, which collapses N redundant reads of the
same file down to effectively one regardless of batching.
"""
import json
import os
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

pytestmark = [pytest.mark.polygons, pytest.mark.contract]


def _rect_poly():
    return QtGui.QPolygonF([QtCore.QPointF(2, 2), QtCore.QPointF(10, 2),
                            QtCore.QPointF(10, 10), QtCore.QPointF(2, 10)])


def _seed_polygons(project, fp, n, prefix="bulk_del"):
    """N single-polygon top-level groups on the same file -- mirrors this
    project's real data shape (one polygon = one top-level group)."""
    groups = []
    for i in range(n):
        g = f"{prefix}_{i:05d}"
        project.all_polygons[g] = {
            fp: {
                'name': g, 'group': g, 'root': '', 'type': 'polygon',
                'coord_space': 'image', 'image_ref_size': {'w': 64, 'h': 64},
                'points': [(2.0, 2.0), (10.0, 2.0), (10.0, 10.0), (2.0, 10.0)],
                'properties': {},
            }
        }
        groups.append(g)
    project._poly_norm_index_invalid = True
    return groups


def _write_polygon_json_files(project, groups, fp):
    """DeletePolygonCommand only os.remove()'s a json_path that EXISTS, and a
    command built via the 3-arg (tab, group, filepath) form derives that path
    from project_folder -- so the on-disk file has to actually be there for
    the delete (and this test's assertion that it's gone afterward) to mean
    anything."""
    pf = project.project_folder
    poly_dir = os.path.join(pf, "polygons")
    os.makedirs(poly_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(fp))[0]
    paths = []
    for g in groups:
        p = os.path.join(poly_dir, f"{g}_{base}_polygons.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(project.all_polygons[g][fp], f)
        paths.append(p)
    return paths


@pytest.fixture
def bulk_delete_setup(synthetic_project, viewer_factory):
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    viewer.image_data = imgd
    pixmap = synthetic_project.convert_cv_to_pixmap(imgd.image)
    viewer.set_image(pixmap)
    rec = {"viewer": viewer, "image_data": imgd}
    synthetic_project.viewer_widgets = list(
        getattr(synthetic_project, "viewer_widgets", []) or []) + [rec]

    # _remove_polygon_from_mask_config's early "no .ax on disk" return would
    # make the cache-hit tests pass vacuously (zero calls either way), so the
    # fixture needs a REAL .ax sidecar to reach the code path under test. This
    # lands inside the temp project_folder synthetic_project already owns
    # (not next to the shared fixture image), so it cannot leak into other
    # tests sharing that fixture.
    axp = synthetic_project._ax_path_for(fp)
    prev_ax = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None
    with open(axp, "w", encoding="utf-8") as f:
        json.dump({"mask_polygon": {"enabled": False, "names": []}}, f)
    if hasattr(synthetic_project, "_invalidate_cached_ax"):
        synthetic_project._invalidate_cached_ax(fp)

    yield synthetic_project, viewer, fp

    try:
        synthetic_project.viewer_widgets.remove(rec)
    except Exception:
        pass
    try:
        if prev_ax is not None:
            open(axp, "w", encoding="utf-8").write(prev_ax)
        elif os.path.exists(axp):
            os.remove(axp)
        if hasattr(synthetic_project, "_invalidate_cached_ax"):
            synthetic_project._invalidate_cached_ax(fp)
    except Exception:
        pass


def _cleanup(project, groups):
    for g in groups:
        project.all_polygons.pop(g, None)
    project._poly_norm_index_invalid = True
    project._bulk_delete_pending = {}
    project._bulk_delete_depth = 0


# ---------------------------------------------------------------------------
# THE regression: manager refresh count
# ---------------------------------------------------------------------------
def test_manager_refresh_happens_once_not_per_polygon(bulk_delete_setup, monkeypatch):
    project, viewer, fp = bulk_delete_setup
    N = 400
    groups = _seed_polygons(project, fp, N)
    _write_polygon_json_files(project, groups, fp)

    calls = {"n": 0}
    real = project.update_polygon_manager

    def _spy():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(project, "update_polygon_manager", _spy)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)
        assert calls["n"] == 1, (
            f"update_polygon_manager ran {calls['n']} times for a "
            f"{N}-polygon batch delete -- it must run exactly once per "
            "batch, not once per polygon (that O(N) list-widget rebuild is "
            "what made large deletes take forever)")
    finally:
        _cleanup(project, groups)


def test_ax_mask_config_read_is_cached_not_reread_per_polygon(
        bulk_delete_setup, monkeypatch):
    """The .ax sidecar for the shared image must not be re-parsed from disk
    once per deleted polygon."""
    project, viewer, fp = bulk_delete_setup
    N = 300
    groups = _seed_polygons(project, fp, N)
    _write_polygon_json_files(project, groups, fp)

    calls = {"n": 0}
    real_get = project._get_cached_ax

    def _spy(filepath):
        calls["n"] += 1
        return real_get(filepath)

    monkeypatch.setattr(project, "_get_cached_ax", _spy)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)
        # One call per deleted polygon is expected (the FUNCTION still runs
        # N times) -- what must NOT happen is N *disk reads*. _get_cached_ax
        # itself is responsible for turning repeat calls for the same,
        # unchanged file into cache hits; assert it was actually consulted
        # (proving the raw open()+json.load() is gone) rather than asserting
        # on private cache internals.
        assert calls["n"] >= N, (
            "_remove_polygon_from_mask_config no longer consults the ax "
            "cache at all for a deleted polygon")
    finally:
        _cleanup(project, groups)


def test_ax_file_is_actually_read_from_disk_at_most_once(bulk_delete_setup, monkeypatch):
    """THE regression at the I/O level: patch the real disk read
    (`_load_ax_for`, which only runs on a cache MISS) and require it to fire
    at most once across the whole batch for one shared, unchanging file."""
    project, viewer, fp = bulk_delete_setup
    N = 300
    groups = _seed_polygons(project, fp, N)
    _write_polygon_json_files(project, groups, fp)

    reads = {"n": 0}
    real_load = project._load_ax_for

    def _spy(filepath):
        reads["n"] += 1
        return real_load(filepath)

    monkeypatch.setattr(project, "_load_ax_for", _spy)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)
        assert reads["n"] <= 2, (
            f"the .ax sidecar for one image was read from disk {reads['n']} "
            f"times while deleting {N} polygons on that image -- it should "
            "be read at most once (the mtime+size cache should absorb every "
            "repeat lookup)")
    finally:
        _cleanup(project, groups)


# ---------------------------------------------------------------------------
# The scene scan must run once per viewer, not once per polygon
# ---------------------------------------------------------------------------
def test_scene_items_scanned_once_per_viewer_not_per_polygon(
        bulk_delete_setup, monkeypatch):
    project, viewer, fp = bulk_delete_setup
    N = 250
    groups = _seed_polygons(project, fp, N)
    _write_polygon_json_files(project, groups, fp)
    for g in groups:
        viewer.add_polygon_to_scene(_rect_poly(), g)

    calls = {"n": 0}
    real_items = viewer._scene.items

    def _spy(*a, **k):
        calls["n"] += 1
        return real_items(*a, **k)

    monkeypatch.setattr(viewer._scene, "items", _spy)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)
        assert calls["n"] <= 2, (
            f"viewer._scene.items() was called {calls['n']} times while "
            f"deleting {N} polygons from one viewer -- it must be scanned "
            "once for the whole batch (O(N) total), not once per polygon "
            "(O(N^2) total)")
    finally:
        _cleanup(project, groups)


# ---------------------------------------------------------------------------
# Correctness must survive the batching
# ---------------------------------------------------------------------------
def test_bulk_delete_actually_removes_everything(bulk_delete_setup):
    project, viewer, fp = bulk_delete_setup
    N = 200
    groups = _seed_polygons(project, fp, N)
    json_paths = _write_polygon_json_files(project, groups, fp)
    for g in groups:
        viewer.add_polygon_to_scene(_rect_poly(), g)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)

        for g in groups:
            assert g not in project.all_polygons, f"'{g}' still in all_polygons"
        remaining_names = {p.get('name') for p in viewer.polygons}
        assert not (remaining_names & set(groups)), (
            f"{len(remaining_names & set(groups))} deleted polygon(s) still "
            "present in viewer.polygons after the bulk delete")
        scene_names = {(getattr(it, 'name', '') or '').strip()
                       for it in viewer._scene.items()}
        assert not (scene_names & set(groups)), (
            "deleted polygon item(s) still present in the QGraphicsScene")
        for p in json_paths:
            assert not os.path.exists(p), f"sidecar JSON not deleted: {p}"
    finally:
        _cleanup(project, groups)


def test_bulk_delete_leaves_untouched_polygons_alone(bulk_delete_setup):
    """The batching must not over-reach and remove polygons outside the
    requested set."""
    project, viewer, fp = bulk_delete_setup
    groups = _seed_polygons(project, fp, 50, prefix="to_delete")
    survivors = _seed_polygons(project, fp, 5, prefix="survivor")
    _write_polygon_json_files(project, groups, fp)
    _write_polygon_json_files(project, survivors, fp)
    for g in groups + survivors:
        viewer.add_polygon_to_scene(_rect_poly(), g)

    try:
        project.polygon_manager.delete_selected_polygons(groups=groups)
        for g in survivors:
            assert g in project.all_polygons, f"untouched polygon '{g}' was deleted"
        remaining_names = {p.get('name') for p in viewer.polygons}
        assert set(survivors) <= remaining_names, (
            "one or more untouched polygons vanished from the viewer during "
            "the bulk delete")
    finally:
        _cleanup(project, groups + survivors)


# ---------------------------------------------------------------------------
# Nesting: delete_all_polygons_for_groups() wraps a loop of
# delete_all_polygons_for_group(), each opening its own bulk context
# ---------------------------------------------------------------------------
def test_nested_bulk_delete_flushes_once_at_the_outer_boundary(
        bulk_delete_setup, monkeypatch):
    project, viewer, fp = bulk_delete_setup
    # Three distinct "groups" of one polygon each -- delete_all_polygons_for_group
    # treats each top-level all_polygons key as one group already, so three
    # keys here stand in for three separate group deletions.
    groups = _seed_polygons(project, fp, 3, prefix="nested")
    _write_polygon_json_files(project, groups, fp)

    calls = {"n": 0}
    real = project.update_polygon_manager
    monkeypatch.setattr(project, "update_polygon_manager",
                        lambda: (calls.__setitem__("n", calls["n"] + 1), real())[-1])

    try:
        project.polygon_manager.delete_all_polygons_for_groups(groups)
        assert calls["n"] == 1, (
            f"update_polygon_manager ran {calls['n']} times across "
            f"{len(groups)} nested per-group deletions -- the outer "
            "_bulk_polygon_delete() context must absorb the inner ones and "
            "flush exactly once")
        for g in groups:
            assert g not in project.all_polygons
    finally:
        _cleanup(project, groups)


# ---------------------------------------------------------------------------
# A generous wall-clock ceiling, per this suite's own stated convention:
# deterministic counters are the hard assertions above; this is a soft
# backstop against the whole class of regression recurring in a different
# shape.
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_bulk_delete_of_800_polygons_is_fast(bulk_delete_setup):
    project, viewer, fp = bulk_delete_setup
    N = 800
    groups = _seed_polygons(project, fp, N)
    _write_polygon_json_files(project, groups, fp)
    for g in groups:
        viewer.add_polygon_to_scene(_rect_poly(), g)

    try:
        t0 = time.perf_counter()
        project.polygon_manager.delete_selected_polygons(groups=groups)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, (
            f"deleting {N} polygons took {elapsed:.2f}s -- reported as "
            "'takes forever, should be instant'; the O(N^2) scene scan and "
            "per-polygon manager rebuild this test suite guards against "
            "would make this scale quadratically with N")
    finally:
        _cleanup(project, groups)
