"""
QC tests: bulk copies of crowns -- Replicate to all viewers, Copy to next /
previous / all roots, shapefile import -- through polygon_bulk.

THE BUG THIS PINS: on BCI2 (3,506 imported crowns, 2.1 M vertices) "Replicate
to all viewers" froze CanoPie for good. It worked item by item on the GUI
thread: it read every crown's full outline, wrote an indented JSON per crown,
mapped every vertex through Qt, and refreshed every list and record per crown,
which made it O(n^2). It also dropped the crowns' attributes. Copy to next
root froze for 18.5 s and dropped them too. The import froze for 8.7 s after
its worker finished.

Now a copy is record level and runs off the GUI thread:

* A crown is copied by image-size ratio, i.e. at the same FRACTION of the
  frame, and records are stored as fractions (points in units of their
  image_ref_size). So the copy's sidecar is the source's bytes, and the copy
  is a HARD LINK (~0.2 ms, no data written). Sidecar writers replace files
  rather than writing into them, so editing either crown later separates them.
* Where memory is ahead of the file (unsaved edits), the copy is written from
  memory, scaled to the target's full-resolution basis, attributes and
  Annotator patches included.
* The GUI thread installs lazy records, draws them in chunks, and pushes ONE
  undo step that renames files to and from polygons/.trash.
"""
import json
import os
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from .generate_fixtures import fixture_image_path
from canopie import polygon_bulk, polygon_lod

pytestmark = [pytest.mark.polygons, pytest.mark.viewer]

PREFIX = "bulkqc_"


# ---------------------------------------------------------------- pure parts

def test_copy_record_keeps_everything_and_scales_in_full_resolution():
    full = {
        "name": "7114", "type": "polygon", "coord_space": "image", "root": "1",
        "image_ref_size": {"w": 1000, "h": 500},
        "points": [[100, 50], [200, 50], [200, 100], [100, 100]],
        "properties": {"species": "oleifera", "sdff": 75.0, "tags": "a;b"},
        "coordinates": {"latitude": 9.1, "longitude": -79.8},
        "is_mask_polygon": True,
        "annotator_patches": {"sdff": {"polygons": [[[[110, 60], [150, 60], [150, 90]]]],
                                        "fraction": 0.25, "area_px": 400.0, "crown_area_px": 1600.0,
                                        "basis": {"w": 1000, "h": 500}}},
        "display_points": [[0, 0]], "json_path": "x", "lod": {"3": []},
    }
    out = polygon_bulk.copy_record(full, (1000, 500), (2000, 1500), stamp="2026-11-13 10:00:00")
    assert out["points"] == [[200, 150], [400, 150], [400, 300], [200, 300]]
    assert out["image_ref_size"] == {"w": 2000, "h": 1500}
    assert out["properties"] == full["properties"] and out["properties"] is not full["properties"]
    assert out["coordinates"] == full["coordinates"] and out["is_mask_polygon"] is True
    assert out["name"] == "7114" and out["last_modified"] == "2026-11-13 10:00:00"
    p = out["annotator_patches"]["sdff"]
    assert p["polygons"] == [[[[220.0, 180.0], [300.0, 180.0], [300.0, 270.0]]]]
    assert p["fraction"] == 0.25 and p["area_px"] == pytest.approx(400 * 2 * 3)
    assert p["basis"] == {"w": 2000, "h": 1500}
    for k in ("display_points", "json_path", "lod"):
        assert k not in out, f"in-memory key {k} leaked into the copy"
    assert full["points"][0] == [100, 50], "the source record was modified"


def test_record_basis_and_coarse_points():
    assert polygon_bulk.record_basis_wh({"image_ref_size": {"w": 48031, "h": 50101}}) == (48031, 50101)
    assert polygon_bulk.record_basis_wh({}, (3, 4)) == (3, 4)
    assert polygon_bulk.record_basis_wh({"coord_space": "norm_pixmap"}) == (1.0, 1.0)
    ring = [[float(i), float(i % 7)] for i in range(600)]
    coarse = polygon_bulk.coarse_points(ring)
    assert 4 <= len(coarse) < len(ring)
    assert polygon_bulk.coarse_points([[1, 2]], "point") == [[1.0, 2.0]]


def test_link_job_shares_the_file_and_fallback_writes_a_scaled_copy(tmp_path, monkeypatch):
    src = tmp_path / "g_A_polygons.json"
    rec = {"name": "g", "type": "polygon", "coord_space": "image", "root": "1",
           "image_ref_size": {"w": 100, "h": 100}, "points": [[10, 10], [20, 10], [20, 20]],
           "properties": {"species": "x"}}
    polygon_lod.dump_sidecar(str(src), rec)
    meta = polygon_bulk.link_meta(rec, "g", "2")
    assert meta["root"] == "2" and meta["full_point_count"] == 3
    job = {"group": "g", "dst_fp": "B.tif", "sidecar": str(tmp_path / "g_B_polygons.json"),
           "link_from": str(src), "meta": meta, "src_wh": (100, 100), "dst_wh": (200, 200),
           "root_id": "2", "coords": None}
    m, trash, linked = polygon_bulk.run_job(job)
    assert linked and trash is None and m is meta
    assert os.path.samefile(job["sidecar"], src), "a hard link was expected"

    # a file system without hard links (FAT/exFAT): a real, scaled copy
    def _no_links(*a, **k):
        raise OSError("hard links not supported")
    monkeypatch.setattr(polygon_bulk.os, "link", _no_links)
    job2 = dict(job, sidecar=str(tmp_path / "g_C_polygons.json"))
    m2, _t, linked2 = polygon_bulk.run_job(job2)
    assert not linked2
    data = json.load(open(job2["sidecar"], encoding="utf-8"))
    assert data["points"][0] == [20.0, 20.0] and data["image_ref_size"] == {"w": 200, "h": 200}
    assert data["root"] == "2" and data["properties"] == {"species": "x"}
    assert m2["full_point_count"] == 3


def test_saving_one_name_of_a_linked_crown_never_changes_the_other(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a_polygons.json"), str(tmp_path / "b_polygons.json")
    polygon_lod.dump_sidecar(a, {"points": [[1, 1]], "name": "a"})
    os.link(a, b)
    polygon_lod.dump_sidecar(b, {"points": [[9, 9]], "name": "b"})
    assert json.load(open(a))["name"] == "a" and json.load(open(b))["name"] == "b"
    # the last-resort in-place write (target held open) unlinks first
    os.remove(b)
    os.link(a, b)

    def _held(*a_, **k):
        raise PermissionError("held open")
    monkeypatch.setattr(polygon_lod.os, "replace", _held)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    polygon_lod.dump_sidecar(b, {"points": [[7, 7]], "name": "b2"})
    assert json.load(open(a))["name"] == "a", "the shared file was written into"
    assert json.load(open(b))["name"] == "b2"


def test_overview_written_in_slices_is_the_same_json(tmp_path):
    recs = [(f"g{i}", "f.tif", {"name": f"n{i}", "points": [[0, 0], [5, 0], [5, 5], [0, 5]] * 3})
            for i in range(437)]
    polygon_lod.write_overview(str(tmp_path), recs)
    o = json.load(open(tmp_path / "_overview.json", encoding="utf-8"))
    assert o["version"] == 1 and o["level"] == polygon_lod.OVERVIEW_LEVEL
    assert [e["g"] for e in o["entries"]] == [f"g{i}" for i in range(437)]
    polygon_lod.write_overview(str(tmp_path), [])
    assert json.load(open(tmp_path / "_overview.json"))["entries"] == []


def test_a_linked_record_keeps_its_own_root_when_read():
    rec = polygon_lod.LazyPolygonRecord({"root": "2", "display_points": [[0, 0]]}, "unused", pinned=("root",))
    rec.materialise_from({"root": "1", "points": [[1, 1], [2, 2], [3, 1]]})
    assert rec["root"] == "2" and rec["points"][0] == [1, 1]
    plain = polygon_lod.LazyPolygonRecord({"root": "2"}, "unused")
    plain.materialise_from({"root": "1"})
    assert plain["root"] == "1"


# ------------------------------------------------------------- in a project

def _ring(cx, cy, r, n=300):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((cx + r * np.cos(t), cy + r * (0.8 + 0.2 * np.sin(5 * t)) * np.sin(t))).round(2).tolist()


class _Beat:
    """Longest gap between 10 ms timer ticks = the longest GUI freeze."""

    def __init__(self):
        self.t = None
        self.max = 0.0
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        now = time.perf_counter()
        if self.t is not None:
            self.max = max(self.max, now - self.t)
        self.t = now

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *a):
        self.timer.stop()


def _idle(qapp, pt, limit=60):
    t0 = time.time()
    while (getattr(pt, "_bulk_threads", None) or getattr(pt, "_chunk_draw_timers", None)) \
            and time.time() - t0 < limit:
        qapp.processEvents()
        time.sleep(0.002)
    fut = getattr(pt, "_overview_future", None)
    if fut is not None:
        fut.result(timeout=30)


@pytest.fixture()
def two_viewers(synthetic_project, viewer_factory, monkeypatch, qapp):
    """Two viewers on two fixture images of different sizes, both in the
    project's viewer grid; crowns written as sidecars on image A and loaded as
    lazy records (as after a project open)."""
    pt = synthetic_project
    for attr in ("_poly_idle_timer", "_autosave_timer", "_poly_resize_save_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            t.stop()
    if hasattr(pt, "_cancel_all_scheduled_poly_loads"):
        pt._cancel_all_scheduled_poly_loads()
    fa = fixture_image_path("rgb_16bit_tiled_bip_cog")
    fb = fixture_image_path("rgb_8bit_untiled")
    widgets = []
    for fp in (fa, fb):
        imgd = pt._imagedata_or_fallback(fp)
        v = viewer_factory()
        v.resize(400, 400)
        v.show()
        v.image_data = imgd
        v.set_image(pt.convert_cv_to_pixmap(imgd.image))
        monkeypatch.setattr(v, "_find_project_owner", lambda: pt)
        widgets.append({"viewer": v, "image_data": imgd})
    stray = viewer_factory()      # an ImageViewer NOT in the grid: never a target
    stray.image_data = pt._imagedata_or_fallback(fixture_image_path("gray_8bit_png"))
    monkeypatch.setattr(pt, "viewer_widgets", widgets)
    monkeypatch.setattr(pt, "_ask_replace_existing", lambda n, w=None: "replace")
    pdir = os.path.join(pt.project_folder, "polygons")
    os.makedirs(pdir, exist_ok=True)
    wa, ha = pt._file_basis_wh(fa)
    root_a = pt._sidecar_root_id(fa)
    groups = []
    for i in range(40):
        g = f"{PREFIX}{i:03d}"
        rec = {"name": f"tree{i}", "type": "polygon", "coord_space": "image", "root": root_a,
               "image_ref_size": {"w": wa, "h": ha},
               "points": _ring(wa * (0.2 + 0.6 * (i % 8) / 7), ha * (0.2 + 0.6 * (i // 8) / 4), wa / 30),
               "properties": {"species": f"sp{i % 3}", "tag": i}}
        path = polygon_bulk.sidecar_path(pdir, g, fa)
        polygon_lod.dump_sidecar(path, rec)
        meta = polygon_bulk.lazy_meta(rec, g)
        # stored under the import's spelling (os.path.normpath): replicate must find it
        pt.all_polygons[g] = {os.path.normpath(fa): polygon_lod.LazyPolygonRecord(meta, path)}
        groups.append(g)
    pt._poly_norm_index_invalid = True
    pt.load_polygons(widgets[0]["viewer"], widgets[0]["image_data"])
    try:
        yield pt, widgets[0]["viewer"], widgets[1]["viewer"], fa, fb, groups, stray
    finally:
        _idle(qapp, pt)
        for g in list(pt.all_polygons):
            if g.startswith(PREFIX):
                pt.all_polygons.pop(g, None)
        for f in os.listdir(pdir):
            if f.startswith(PREFIX):
                try:
                    os.remove(os.path.join(pdir, f))
                except OSError:
                    pass
        pt._poly_norm_index_invalid = True
        pt.undo_stack.clear()


def _records_on(pt, fp):
    out = {}
    for g, k in pt._poly_index_lookup(fp):
        if g.startswith(PREFIX):
            out[g] = (k, pt.all_polygons[g][k])
    return out


def test_replicate_links_crowns_keeps_attributes_and_never_freezes(two_viewers, qapp):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    # one crown edited in memory, not saved: its copy must come from memory
    edited = groups[0]
    key = os.path.normpath(fa)
    rec = pt.all_polygons[edited][key]
    rec["points"] = [[float(x) + 1.0, float(y)] for x, y in rec["points"]]
    pt._dirty_polys = set(getattr(pt, "_dirty_polys", set()) or set()) | {(edited, key)}

    items = [it for it in va.get_all_polygons() if getattr(it, "name", "").startswith(PREFIX)]
    assert len(items) == len(groups)
    with _Beat() as beat:
        t0 = time.perf_counter()
        va.replicate_toviewer(items)
        _idle(qapp, pt)
        took = time.perf_counter() - t0
    assert took < 20, f"replicating {len(groups)} crowns took {took:.1f}s"
    assert beat.max < 0.75, f"the window froze for {beat.max:.2f}s"

    on_b = _records_on(pt, fb)
    assert set(on_b) == set(groups)
    assert not _records_on(pt, stray.image_data.filepath), "a viewer outside the grid got copies"
    wa, ha = pt._file_basis_wh(fa)
    wb, hb = pt._file_basis_wh(fb)
    pdir = os.path.join(pt.project_folder, "polygons")
    for g in groups:
        k, r = on_b[g]
        assert isinstance(r, polygon_lod.LazyPolygonRecord)
        src_path = polygon_bulk.sidecar_path(pdir, g, fa)
        dst_path = polygon_bulk.sidecar_path(pdir, g, fb)
        if g == edited:
            assert not os.path.samefile(src_path, dst_path)
            assert r["image_ref_size"] == {"w": wb, "h": hb}
            assert r["points"][0][0] == pytest.approx((rec["points"][0][0]) * wb / wa)
        else:
            assert os.path.samefile(src_path, dst_path), f"{g} was not hard-linked"
            assert dict.get(r, "root") == pt._sidecar_root_id(fb)
        assert r["properties"]["species"] == f"sp{int(g[-3:]) % 3}"
        assert dict.get(r, "root") == pt._sidecar_root_id(fb), "the copy kept the source's root"
    drawn = [it for it in vb.get_all_polygons() if getattr(it, "name", "").startswith(PREFIX)]
    assert len(drawn) == len(groups), "the copies were not drawn on the target viewer"
    assert all(getattr(it, "is_lod_geometry", False) or len(it.polygon) <= 300 for it in drawn)

    # one undo step removes every copy (files to .trash), redo brings them back
    pt.undo_stack.undo()
    _idle(qapp, pt)
    assert not _records_on(pt, fb)
    assert not any(os.path.exists(polygon_bulk.sidecar_path(pdir, g, fb)) for g in groups)
    assert not [it for it in vb.get_all_polygons() if getattr(it, "name", "").startswith(PREFIX)]
    assert all(os.path.exists(polygon_bulk.sidecar_path(pdir, g, fa)) for g in groups), \
        "undo touched the source crowns"
    pt.undo_stack.redo()
    _idle(qapp, pt)
    assert set(_records_on(pt, fb)) == set(groups)
    assert len([it for it in vb.get_all_polygons() if getattr(it, "name", "").startswith(PREFIX)]) == len(groups)


@pytest.mark.parametrize("answer", ["replace", "skip", None])
def test_existing_crowns_on_the_target_are_asked_about_once(two_viewers, qapp, monkeypatch, answer):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    pdir = os.path.join(pt.project_folder, "polygons")
    g = groups[3]
    wb, hb = pt._file_basis_wh(fb)
    old = {"name": "OLD", "type": "polygon", "coord_space": "image", "root": "",
           "image_ref_size": {"w": wb, "h": hb}, "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
           "properties": {"species": "old"}}
    old_path = polygon_bulk.sidecar_path(pdir, g, fb)
    polygon_lod.dump_sidecar(old_path, old)
    pt.all_polygons[g][fb] = polygon_lod.LazyPolygonRecord(polygon_bulk.lazy_meta(old, g), old_path)
    pt._add_to_polygon_index(g, fb)
    asked = []
    monkeypatch.setattr(pt, "_ask_replace_existing", lambda n, w=None: asked.append(n) or answer)

    n = pt.replicate_polygons(groups[:5], fa, [fb], wait=True)
    _idle(qapp, pt)
    assert asked == [1], "asked about the one existing crown, once"
    on_b = _records_on(pt, fb)
    if answer is None:
        assert n == 0 and set(on_b) == {g} and on_b[g][1]["properties"]["species"] == "old"
        return
    assert set(on_b) == set(groups[:5])
    species = on_b[g][1]["properties"]["species"]
    assert species == ("sp0" if answer == "replace" else "old")
    if answer == "replace":
        pt.undo_stack.undo()
        _idle(qapp, pt)
        on_b = _records_on(pt, fb)
        assert set(on_b) == {g}, "undo did not remove the copies"
        assert on_b[g][1]["properties"]["species"] == "old", "undo did not restore the replaced crown"
        assert json.load(open(old_path, encoding="utf-8"))["name"] == "OLD"


def test_polygon_manager_replicates_selected_groups(two_viewers, qapp):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    from canopie.polygon_manager import PolygonManager
    pm = PolygonManager(pt)
    try:
        n = pm.replicate_groups_to_viewers(groups[:7], wait=True)
        _idle(qapp, pt)
        assert n == 7 and set(_records_on(pt, fb)) == set(groups[:7])
    finally:
        pm.deleteLater()
        qapp.processEvents()


def test_copy_to_next_root_links_and_keeps_attributes(two_viewers, qapp):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    ra, rb = pt.get_root_by_filepath(fa), pt.get_root_by_filepath(fb)
    assert ra and rb and ra != rb, "precondition: the two fixtures are separate roots"
    n = pt._copy_roots_bulk(ra, [rb], broadcast=False, title="Copy to next root", wait=True)
    _idle(qapp, pt)
    try:
        on_b = _records_on(pt, fb)
        # the fixture's own test polygons on root A are copied too
        assert n >= len(groups) and set(on_b) == set(groups), \
            "crowns stored under the import's path spelling were skipped"
        k, r = on_b[groups[5]]
        assert r["properties"] == {"species": "sp2", "tag": 5}
        assert dict.get(r, "root") == pt._sidecar_root_id(fb)
    finally:
        pt.undo_stack.undo()          # the shared project's own polygons back as they were
        _idle(qapp, pt)


def test_import_bulk_writes_off_the_gui_thread_and_installs_lazy_records(two_viewers, qapp, monkeypatch):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    wb, hb = pt._file_basis_wh(fb)
    imported = {}
    for i in range(30):
        g = f"{PREFIX}imp{i:03d}"
        imported[g] = {os.path.normpath(fb): {
            "name": f"imp{i}", "group": g, "root": "", "type": "polygon", "coord_space": "image",
            "image_ref_size": {"w": wb, "h": hb}, "points": _ring(wb / 2, hb / 2, wb / 5, 250),
            "coordinates": {"latitude": 0.0, "longitude": 0.0}, "properties": {"k": i}}}
    # one malformed feature must not stop the others
    imported[f"{PREFIX}imp_bad"] = {os.path.normpath(fb): {
        "name": "bad", "type": "polygon", "coord_space": "image",
        "image_ref_size": {"w": wb, "h": hb}, "points": "not-a-list-of-points"}}
    monkeypatch.setattr(pt, "save_incremental",
                        lambda *a, **k: pytest.fail("import ran the synchronous full save"))
    before = pt.undo_stack.count()
    done = []
    n = pt.import_polygons_bulk(imported, wait=True, on_done=lambda ok, err: done.append((ok, err)))
    _idle(qapp, pt)
    assert n == 31 and done == [(30, 1)], done
    imported.pop(f"{PREFIX}imp_bad")
    assert os.path.normpath(fb) not in pt.all_polygons.get(f"{PREFIX}imp_bad", {})
    pdir = os.path.join(pt.project_folder, "polygons")
    for g in imported:
        r = pt.all_polygons[g][os.path.normpath(fb)]
        assert isinstance(r, polygon_lod.LazyPolygonRecord) and not r.is_materialised
        assert os.path.exists(polygon_bulk.sidecar_path(pdir, g, fb))
    assert pt.undo_stack.count() == before + 1, "the import is one undo step"
    ov = json.load(open(os.path.join(pdir, "_overview.json"), encoding="utf-8"))
    assert {e["g"] for e in ov["entries"]} >= set(imported)


def test_selecting_many_crowns_notifies_once(two_viewers, qapp):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    calls = []
    va._scene.selectionChanged.connect(lambda: calls.append(1))
    items = [it for it in va.get_all_polygons() if getattr(it, "name", "").startswith(PREFIX)]
    with va.bulk_selection():
        for it in items:
            va.select_polygon_item(it, True)
    assert len(calls) == 1 and all(it.isSelected() for it in items)


def test_index_buckets_are_sets_too(synthetic_project):
    from canopie.project_tab import _IndexBucket
    b = _IndexBucket(["a", "b", "a"])
    assert list(b) == ["a", "b"] and "a" in b and "z" not in b
    b.append("c")
    b.remove("a")
    assert list(b) == ["b", "c"] and "a" not in b
    with pytest.raises(ValueError):
        b.remove("zz")


# ------------------------------------------- Polygon Manager with many groups

def test_lock_state_of_many_groups_scans_each_viewer_once(synthetic_project, qapp):
    """Right-clicking 3,506 selected BCI crowns took 8 s before the menu
    showed: the Lock / Unlock label scanned every item once per group."""
    from canopie.polygon_manager import PolygonManager

    class _Item:
        def __init__(self, n):
            self.name, self.is_locked = n, False

    class _Viewer:
        calls = 0

        def __init__(self, items):
            self.items = items

        def get_all_polygons(self):
            _Viewer.calls += 1
            return self.items

    items = [_Item(f"g{i}") for i in range(500)]
    pm = PolygonManager(synthetic_project)
    try:
        pm._iter_viewers = lambda: [({}, _Viewer(items)), ({}, _Viewer(items))]
        groups = [f"g{i}" for i in range(500)]
        assert pm._groups_locked(groups) is False
        assert _Viewer.calls == 2, f"{_Viewer.calls} scans for 2 viewers"
        pm._set_groups_locked(groups, True)
        assert all(it.is_locked for it in items) and pm._groups_locked(groups)
    finally:
        pm.deleteLater()
        qapp.processEvents()


def test_info_and_zoom_never_read_crown_files(two_viewers, qapp, monkeypatch):
    """Polygon Info (58 s) and Zoom on 3,506 selected crowns read every
    crown's file from disk; lazy records answer both without it."""
    pt, va, vb, fa, fb, groups, stray = two_viewers
    from canopie.polygon_manager import PolygonManager
    pm = PolygonManager(pt)
    try:
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: 0)
        monkeypatch.setattr(QtWidgets.QDialog, "exec_", lambda self: 0)
        pm.show_polygon_info(groups)
        pm._iter_viewers = lambda: [({"image_data": va.image_data}, va)]
        pm.zoom_to_groups(groups)
        read = [g for g in groups if pt.all_polygons[g][os.path.normpath(fa)].is_materialised]
        assert not read, f"{len(read)} crowns were read from disk"
    finally:
        pm.deleteLater()
        qapp.processEvents()


def test_copy_and_move_selected_groups_use_the_bulk_engine(two_viewers, qapp, monkeypatch):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    from canopie.polygon_manager import PolygonManager
    ra, rb = pt.get_root_by_filepath(fa), pt.get_root_by_filepath(fb)
    pm = PolygonManager(pt)
    pm.current_root = ra
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a, **k: (rb, True))
    try:
        pm.copy_selected_polygons(groups[:3])
        _idle(qapp, pt)
        assert set(_records_on(pt, fb)) == set(groups[:3]), "copied more or less than the selection"
        assert pt.all_polygons[groups[1]][fb]["properties"]["tag"] == 1
        pm.move_selected_polygons(groups[3:6])
        _idle(qapp, pt)
        assert set(_records_on(pt, fb)) == set(groups[:6])
        on_a = {g for g, _k in pt._poly_index_lookup(fa)}
        assert not (set(groups[3:6]) & on_a), "Move left the crowns on the source image"
        assert set(groups[:3]) <= on_a
        pdir = os.path.join(pt.project_folder, "polygons")
        assert os.path.exists(polygon_bulk.sidecar_path(pdir, groups[4], fb)), \
            "deleting the source broke the moved (hard-linked) copy"
        # undo: the source crowns come back, then the copies go
        pt.undo()
        _idle(qapp, pt)
        assert set(groups[3:6]) <= {g for g, _k in pt._poly_index_lookup(fa)}
    finally:
        pm.deleteLater()
        qapp.processEvents()


def test_bulk_delete_checks_an_image_without_ax_once(two_viewers, qapp, monkeypatch):
    pt, va, vb, fa, fb, groups, stray = two_viewers
    calls = []
    real = pt._ax_path_for
    monkeypatch.setattr(pt, "_ax_path_for", lambda fp: calls.append(fp) or real(fp))
    pt.delete_polygons_on_files(groups[:10], [fa], title="QC delete")
    _idle(qapp, pt)
    assert not _records_on(pt, fa).keys() & set(groups[:10])
    assert len(calls) <= 2, f"the .ax lookup ran {len(calls)} times for one image"
    assert "_bulk_no_ax_files" not in pt.__dict__, "the per-delete cache outlived the delete"
    pt.undo()
    _idle(qapp, pt)
    assert set(groups[:10]) <= set(_records_on(pt, fa))
