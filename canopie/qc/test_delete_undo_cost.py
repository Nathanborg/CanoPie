"""
Deleting N polygons must cost O(N) cheap operations -- and stay undoable.

THE BUG THIS PINS (reported as "clean root polys / clear polygons / delete all
polygons takes forever -- I suspect the redo/undo implementation"). That
suspicion was correct, and profiling a real 500-polygon delete found two
distinct costs, both inside the undo machinery:

1. THE SNAPSHOT WAS BUILT UP FRONT, BY RE-SERIALISING MEMORY.
   `DeletePolygonCommand.__init__` did
   `json.dumps(all_polygons[group][filepath])` to capture the polygon for
   undo. That serialises ~40 KB of coordinates AND -- since polygons load
   lazily (polygon_lod.LazyPolygonRecord) -- MATERIALISES the record, i.e.
   reads its file off disk purely to re-serialise what that file already
   contains. Measured:

       construct 500 commands : 14.92 s  (29.8 ms each)
       records materialised   : 500/500
       snapshots held in RAM  : 9.2 MB

   ~70 s for a 2333-polygon delete, spent before a single file was removed.

2. THE SNAPSHOT READ THE FILE AT ALL.
   Deferring the snapshot to redo() removed the materialise, but the read
   remained -- and on Windows, where real-time AV scanning hits every file
   open, that measured ~28 ms PER FILE: 14.2 s of `_io.open` for 500 polygons.

   The sidecar does not need to be READ to be preserved. redo() now MOVES it
   into `polygons/.trash/` and undo() moves it back: metadata operations
   (~0.6 ms) that keep the exact original bytes.

Result on the same 500-polygon delete:

       before  ~28.4 s   (14.92 construct + 13.48 push)
       after     0.59 s  (48x)

The trash directory is inside `polygons/` so the rename is guaranteed
same-volume (a temp dir may be on another drive, where os.replace fails), is a
dot-directory, and every scan filters on '*_polygons.json' so it stays
invisible to the loader. It is purged at project open, where anything left
belongs to a dead undo stack.
"""
import json
import os
import time

import pytest
from PyQt5 import QtWidgets

pytestmark = [pytest.mark.polygons, pytest.mark.perf]

IMG = "somewhere/ortho_cog.tif"


@pytest.fixture
def project_factory(qapp, tmp_path):
    """Builds ProjectTabs and tears them down.

    ProjectTab is a QWidget; leaving several of them alive across a module
    crashes the interpreter at shutdown in this PyQt build (the same rc-127
    teardown fault the QC conftest documents), so each one is closed and
    deleted explicitly.
    """
    made = []

    def _make(n):
        pt = _project(qapp, tmp_path, n)
        made.append(pt)
        return pt

    yield _make

    for pt in made:
        try:
            pt.close()
            pt.setParent(None)
            pt.deleteLater()
        except Exception:
            pass
    qapp.processEvents()


def _project(qapp, tmp_path, n):
    """A ProjectTab with `n` polygons on disk, loaded lazily."""
    from ..project_tab import ProjectTab

    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMG))[0]
    for i in range(n):
        g = f"grp_{i:05d}"
        (poly_dir / f"{g}_{base}_polygons.json").write_text(json.dumps({
            'name': g, 'group': g, 'root': '', 'type': 'polygon',
            'coord_space': 'image', 'image_ref_size': {'w': 1000, 'h': 1000},
            'points': [[float(j), float(j * 2)] for j in range(120)],
            'coordinates': {}, 'properties': {'fid': i},
        }), encoding="utf-8")

    pt = ProjectTab.__new__(ProjectTab)
    QtWidgets.QWidget.__init__(pt)
    pt.project_folder = str(tmp_path)
    pt.multispectral_image_data_groups = {"R1": [IMG]}
    pt.thermal_rgb_image_data_groups = {}
    # First load takes the migration path (reads every file, writes the
    # overview) and returns PLAIN dicts; only a subsequent load returns
    # LazyPolygonRecords. Real projects behave the same way -- the first open
    # after upgrading migrates, every open after that is lazy -- and the lazy
    # state is exactly what these tests are about.
    pt._load_polygons_from_dir()
    pt.all_polygons = pt._load_polygons_from_dir()
    pt.viewer_widgets = []
    pt.undo_stack = QtWidgets.QUndoStack()
    pt._dirty_polygon_entries = set()
    pt._poly_norm_index_invalid = True
    pt.polygon_manager = None
    return pt


def _delete(pt, groups):
    from ..project_tab import DeletePolygonCommand
    with pt._bulk_polygon_delete():
        pt.undo_stack.beginMacro("del")
        for g in groups:
            pt.undo_stack.push(DeletePolygonCommand(pt, g, IMG))
        pt.undo_stack.endMacro()


# ---------------------------------------------------------------------------
# 1. Construction must not touch geometry
# ---------------------------------------------------------------------------
def test_command_construction_does_not_materialise_or_serialise(project_factory, qapp, tmp_path):
    """THE first regression. Building the undo command must be bookkeeping
    only -- no file read, no json.dumps."""
    from ..project_tab import DeletePolygonCommand

    pt = project_factory(40)
    groups = sorted(pt.all_polygons)
    cmds = [DeletePolygonCommand(pt, g, IMG) for g in groups]

    materialised = sum(1 for g in groups
                       if getattr(pt.all_polygons[g][IMG], 'is_materialised', True))
    assert materialised == 0, (
        f"{materialised}/{len(groups)} polygons were read off disk just to "
        "build their undo commands -- measured 29.8 ms each on the real "
        "project, ~70 s for a full delete before anything is removed")
    assert all(c.json_content is None for c in cmds), (
        "commands pre-serialised the polygon into memory; the sidecar on disk "
        "already is the snapshot")
    assert all(c.json_path for c in cmds), "json_path was not resolved"


# ---------------------------------------------------------------------------
# 2. Deleting must not read the files
# ---------------------------------------------------------------------------
def test_delete_moves_sidecars_instead_of_reading_them(project_factory, qapp, tmp_path, monkeypatch):
    """THE second regression, at the I/O level: opening each file to snapshot
    it cost ~28 ms apiece under Windows AV scanning."""
    import builtins

    pt = project_factory(30)
    groups = sorted(pt.all_polygons)

    opened = []
    real_open = builtins.open

    def _spy(path, *a, **k):
        if str(path).endswith("_polygons.json"):
            opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", _spy)
    _delete(pt, groups)
    monkeypatch.setattr(builtins, "open", real_open)

    assert not opened, (
        f"{len(opened)} polygon file(s) were opened during the delete; the "
        "sidecars should be MOVED to the trash, not read")


def test_deleted_files_are_gone_from_the_polygons_dir(project_factory, qapp, tmp_path):
    pt = project_factory(25)
    groups = sorted(pt.all_polygons)
    _delete(pt, groups)

    remaining = [f for f in os.listdir(tmp_path / "polygons")
                 if f.endswith("_polygons.json")]
    assert remaining == [], f"{len(remaining)} sidecar(s) survived the delete"
    for g in groups:
        assert g not in pt.all_polygons, f"'{g}' still in memory"


# ---------------------------------------------------------------------------
# 3. Undo must still work -- byte-for-byte
# ---------------------------------------------------------------------------
def test_undo_restores_files_byte_for_byte(project_factory, qapp, tmp_path):
    """The whole optimisation is only legitimate if undo is unharmed."""
    pt = project_factory(20)
    groups = sorted(pt.all_polygons)
    base = os.path.splitext(os.path.basename(IMG))[0]
    before = {g: (tmp_path / "polygons" / f"{g}_{base}_polygons.json").read_text(encoding="utf-8")
              for g in groups}

    _delete(pt, groups)
    pt.undo_stack.undo()

    for g in groups:
        p = tmp_path / "polygons" / f"{g}_{base}_polygons.json"
        assert p.exists(), f"undo did not restore {g}"
        assert p.read_text(encoding="utf-8") == before[g], (
            f"{g} was restored with DIFFERENT bytes than it had")


def test_undo_restores_memory_with_full_geometry(project_factory, qapp, tmp_path):
    pt = project_factory(15)
    groups = sorted(pt.all_polygons)
    _delete(pt, groups)
    pt.undo_stack.undo()

    for g in groups:
        assert g in pt.all_polygons, f"'{g}' not restored to memory"
        pts = pt.all_polygons[g][IMG].get('points')
        assert pts and len(pts) == 120, (
            f"'{g}' came back with {0 if not pts else len(pts)} vertices "
            "instead of 120 -- undo restored partial geometry")


def test_redo_after_undo_still_deletes(project_factory, qapp, tmp_path):
    """The snapshot is captured on the FIRST redo; a second redo must find the
    restored file and still work."""
    pt = project_factory(10)
    groups = sorted(pt.all_polygons)
    base = os.path.splitext(os.path.basename(IMG))[0]

    _delete(pt, groups)
    pt.undo_stack.undo()
    pt.undo_stack.redo()

    remaining = [f for f in os.listdir(tmp_path / "polygons")
                 if f.endswith("_polygons.json")]
    assert remaining == [], f"redo left {len(remaining)} sidecar(s) behind"
    for g in groups:
        assert g not in pt.all_polygons


# ---------------------------------------------------------------------------
# 4. Housekeeping
# ---------------------------------------------------------------------------
def test_trash_is_invisible_to_the_loader(project_factory, qapp, tmp_path):
    pt = project_factory(12)
    groups = sorted(pt.all_polygons)
    _delete(pt, groups[:6])

    trash = tmp_path / "polygons" / ".trash"
    assert trash.is_dir() and any(trash.iterdir()), "nothing was moved to trash"

    pt2 = project_factory(0)
    loaded = pt2._load_polygons_from_dir()
    assert len(loaded) == 6, (
        f"loader saw {len(loaded)} groups; trashed files must not be loaded")


def test_stale_trash_is_purged_at_project_open(project_factory, qapp, tmp_path):
    """Trash from a previous session is unrecoverable (its undo stack is gone)
    and must not accumulate."""
    pt = project_factory(8)
    _delete(pt, sorted(pt.all_polygons)[:4])
    trash = tmp_path / "polygons" / ".trash"
    assert any(trash.iterdir())

    pt2 = project_factory(0)
    pt2._load_polygons_from_dir()

    assert not any(trash.iterdir()), "stale delete-undo files were not purged"


@pytest.mark.slow
def test_bulk_delete_of_400_polygons_is_fast(project_factory, qapp, tmp_path):
    """Wall-clock backstop. Before the fix this same shape measured ~28 s for
    500 polygons; the ceiling here is deliberately loose."""
    pt = project_factory(400)
    groups = sorted(pt.all_polygons)
    t0 = time.perf_counter()
    _delete(pt, groups)
    elapsed = time.perf_counter() - t0
    assert elapsed < 8.0, (
        f"deleting 400 polygons took {elapsed:.1f}s -- the undo snapshot is "
        "reading or serialising per polygon again")
