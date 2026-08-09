"""
project.json must not carry a second copy of every polygon.

THE BUG THIS PINS (reported as "I still have very slow responses, lagged, for
this project C:\\New Folder182"):

Polygons were persisted TWICE -- once as `<project>/polygons/{group}_{image}_
polygons.json` (one file per polygon, written incrementally) and again as a
complete `all_polygons` blob embedded in project.json. Measured on the real
project:

    project.json                310 MB
    parse of project.json        3.0 s  (14.1 s cold)
    polygons/ dir                 95 MB, 2333 files

Two consequences, both matching the report:

  * SPEED. Every save path rewrote that blob, pretty-printed at indent=4, ON
    THE GUI THREAD -- save_incremental (Ctrl+Q and every autosave tick) even
    re-READ the whole 310 MB file first just to preserve the other keys. That
    is a multi-second freeze on a timer.

  * CORRECTNESS. Deleting a polygon removes its file, but nothing pruned the
    embedded blob, so the two stores drifted. In the real project the blob
    held 6091 groups against 2333 files on disk -- reopening the project
    RESURRECTED 3759 previously deleted polygons, which then had to be
    rendered and re-saved, compounding the slowness.

THE FIX: the polygons/ dir is the single source of truth.
`_load_polygons_from_dir()` restores from it (6.5 s vs 14.1 s on the real
project, and 2333 groups instead of 6091), and the save paths write
`"all_polygons": {}` + `"polygons_external": true` -- but only after
`_ensure_all_polygons_on_disk()` confirms every in-memory entry is backed by a
file, since entries loaded from an OLD embedded blob are never marked dirty and
would otherwise be dropped.
"""
import json
import os

import pytest

pytestmark = [pytest.mark.io, pytest.mark.polygons]

IMG = "somewhere/my_orthomosaic_cog.tif"


def _mk(tmp_path, groups):
    """Build a fake project folder with one polygon file per (group, image)."""
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMG))[0]
    for g in groups:
        (poly_dir / f"{g}_{base}_polygons.json").write_text(
            json.dumps({'name': g.split('_')[-1], 'group': g, 'root': '',
                        'type': 'polygon', 'coord_space': 'pixel',
                        'image_ref_size': {'w': 48031, 'h': 50101},
                        'points': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        'properties': {}}),
            encoding="utf-8")
    return poly_dir


def _tab(tmp_path):
    from ..project_tab import ProjectTab
    pt = ProjectTab.__new__(ProjectTab)          # no Qt needed for these methods
    pt.project_folder = str(tmp_path)
    pt.multispectral_image_data_groups = {"Root1": [IMG]}
    pt.thermal_rgb_image_data_groups = {}
    pt.all_polygons = {}
    return pt


# ---------------------------------------------------------------------------
# Loading from the dir
# ---------------------------------------------------------------------------
def test_loads_every_polygon_file(tmp_path):
    groups = [f"alicastrum_{i}" for i in range(25)]
    _mk(tmp_path, groups)
    got = _tab(tmp_path)._load_polygons_from_dir()
    assert set(got) == set(groups)
    assert all(IMG in v for v in got.values())


def test_group_name_survives_underscores_in_both_halves(tmp_path):
    """The filename is `{group}_{imagebase}_polygons.json` and BOTH halves
    contain underscores, so the split cannot be done positionally.

    The old per-file fallback in load_polygons split on the first '_' (or fell
    back to the JSON's 'name' field, which holds the DISPLAY name -- "4471" for
    group "alicastrum_4471"), so groups were misfiled under a bare number.
    """
    _mk(tmp_path, ["alicastrum_4471", "crowns_20m_filtered_only_liana_load_3129.0"])
    got = _tab(tmp_path)._load_polygons_from_dir()
    assert "alicastrum_4471" in got, sorted(got)
    assert "crowns_20m_filtered_only_liana_load_3129.0" in got, sorted(got)
    assert "4471" not in got, "display name leaked in as a group name"


def test_missing_or_empty_dir_returns_empty(tmp_path):
    tab = _tab(tmp_path)
    assert tab._load_polygons_from_dir() == {}      # no dir at all
    (tmp_path / "polygons").mkdir()
    assert tab._load_polygons_from_dir() == {}      # dir present but empty


def test_unreadable_file_does_not_abort_the_load(tmp_path):
    poly_dir = _mk(tmp_path, ["good_1", "good_2"])
    base = os.path.splitext(os.path.basename(IMG))[0]
    (poly_dir / f"corrupt_9_{base}_polygons.json").write_text("{not json", encoding="utf-8")
    got = _tab(tmp_path)._load_polygons_from_dir()
    assert {"good_1", "good_2"} <= set(got), (
        "one corrupt polygon file aborted the whole project load")


# ---------------------------------------------------------------------------
# The safety gate before externalising
# ---------------------------------------------------------------------------
def test_backfills_entries_that_have_no_file(tmp_path):
    """THE data-loss guard. Entries restored from an old embedded blob are
    never marked dirty, so nothing would ever write them to disk -- blanking
    the blob without backfilling them first would delete them."""
    _mk(tmp_path, ["already_on_disk"])
    tab = _tab(tmp_path)
    tab.all_polygons = {
        "already_on_disk": {IMG: {'points': [[0, 0]], 'coord_space': 'image'}},
        "only_in_memory": {IMG: {'points': [[9, 9]], 'coord_space': 'image'}},
    }

    assert tab._ensure_all_polygons_on_disk() is True

    base = os.path.splitext(os.path.basename(IMG))[0]
    written = tmp_path / "polygons" / f"only_in_memory_{base}_polygons.json"
    assert written.exists(), (
        "an in-memory polygon with no file was not backfilled; externalising "
        "project.json would have destroyed it")
    assert json.loads(written.read_text(encoding="utf-8"))["points"] == [[9, 9]]


def test_round_trip_memory_to_disk_to_memory(tmp_path):
    """Whatever the gate writes must be exactly what the loader reads back."""
    tab = _tab(tmp_path)
    tab.all_polygons = {
        f"grp_{i}": {IMG: {'name': str(i), 'points': [[i, i], [i + 1, i + 1]],
                           'coord_space': 'image',
                           'image_ref_size': {'w': 100, 'h': 100}}}
        for i in range(30)
    }
    assert tab._ensure_all_polygons_on_disk() is True

    reloaded = _tab(tmp_path)._load_polygons_from_dir()
    assert set(reloaded) == set(tab.all_polygons)
    for g, fm in tab.all_polygons.items():
        assert reloaded[g][IMG]["points"] == fm[IMG]["points"], f"{g} points changed"


def test_gate_reports_failure_when_the_dir_is_unusable(tmp_path, monkeypatch):
    """On failure the caller must keep the embedded copy rather than blank it."""
    tab = _tab(tmp_path)
    tab.all_polygons = {"g": {IMG: {'points': [[1, 1]]}}}

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "makedirs", _boom)
    assert tab._ensure_all_polygons_on_disk() is False, (
        "the gate reported success despite being unable to write -- the caller "
        "would then blank the embedded polygons and lose them")


def test_no_project_folder_is_not_externalisable(tmp_path):
    tab = _tab(tmp_path)
    tab.project_folder = ""
    assert tab._ensure_all_polygons_on_disk() is False
    assert tab._load_polygons_from_dir() == {}


# ---------------------------------------------------------------------------
# The save paths must stop embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn_name", [
    "save_project", "save_project_quick", "save_incremental",
])
def test_save_paths_do_not_embed_all_polygons(fn_name):
    """Static guard across every writer of project.json.

    A behavioural test would need a full Qt ProjectTab plus folder dialogs;
    the defect is a literal dict field, so assert on the source that none of
    them writes the full blob unconditionally.
    """
    import inspect
    import textwrap

    from ..project_tab import ProjectTab

    src = textwrap.dedent(inspect.getsource(getattr(ProjectTab, fn_name)))
    assert '"polygons_external"' in src, (
        f"{fn_name} does not mark project.json as having external polygons")
    # The only permitted embed is the guarded fallback when the gate failed.
    if '"all_polygons": {k: v for k, v in self.all_polygons.items()}' in src:
        assert "_ensure_all_polygons_on_disk" in src, (
            f"{fn_name} embeds the full polygon blob without first confirming "
            "every polygon is backed by a file")


def test_load_prefers_the_dir_over_an_embedded_copy():
    """Static guard on both load sites: a populated polygons/ dir must win, or
    deleted polygons come back from a stale blob."""
    import inspect
    import textwrap

    from ..project_tab import ProjectTab

    for fn_name in ("load_project", "load_project_file"):
        src = textwrap.dedent(inspect.getsource(getattr(ProjectTab, fn_name)))
        assert "_load_polygons_from_dir" in src, (
            f"{fn_name} still restores polygons only from the embedded "
            "project.json copy, which is never pruned on delete")


# ---------------------------------------------------------------------------
# The overview is a CACHE, never a FILTER
# ---------------------------------------------------------------------------
def _write_overview_covering(tab, tmp_path, groups):
    """Write an _overview.json that lists ONLY `groups`."""
    from .. import polygon_lod
    base = os.path.splitext(os.path.basename(IMG))[0]
    records = [(g, IMG, {'name': g, 'group': g, 'root': '', 'type': 'polygon',
                         'coord_space': 'image',
                         'image_ref_size': {'w': 48031, 'h': 50101},
                         'points': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})
               for g in groups]
    polygon_lod.write_overview(str(tmp_path / "polygons"), records)


def test_a_polygon_drawn_after_import_survives_a_refresh(tmp_path):
    """THE data-loss bug, reproduced.

    Import a shapefile (many groups, all listed in the overview), then draw one
    more polygon. Its sidecar is written correctly, but the overview still lists
    only the imported ones. The load path used to accept the overview whenever
    the entry count was within 5% of the file count -- so the drawn polygon was
    never read, and it vanished on the next refresh while every imported one
    came back.
    """
    imported = [f"crown_{i}" for i in range(200)]
    _mk(tmp_path, imported)
    tab = _tab(tmp_path)
    _write_overview_covering(tab, tmp_path, imported)

    # ...now the user draws one more polygon; on_polygon_drawn writes its file.
    _mk(tmp_path, ["hand_drawn_tree1"])

    got = tab._load_polygons_from_dir()
    assert "hand_drawn_tree1" in got, (
        "the polygon drawn after the import was dropped on load -- the "
        "overview was treated as the authoritative list of polygons instead "
        "of a cache of the ones it happens to describe")
    assert len(got) == len(imported) + 1


def test_many_drawn_polygons_survive(tmp_path):
    """Well inside the old 5% tolerance, so every one of these was lost."""
    imported = [f"crown_{i}" for i in range(400)]
    _mk(tmp_path, imported)
    tab = _tab(tmp_path)
    _write_overview_covering(tab, tmp_path, imported)

    drawn = [f"drawn_{i}" for i in range(15)]      # 15/415 = 3.6%, under 5%
    _mk(tmp_path, drawn)

    got = tab._load_polygons_from_dir()
    for g in drawn:
        assert g in got, f"{g} was dropped by the overview fast path"


def test_overview_covered_polygons_still_load_lazily(tmp_path):
    """The patching must not cost the fast path its laziness."""
    from .. import polygon_lod
    imported = [f"crown_{i}" for i in range(50)]
    _mk(tmp_path, imported)
    tab = _tab(tmp_path)
    _write_overview_covering(tab, tmp_path, imported)

    got = tab._load_polygons_from_dir()
    rec = got["crown_0"][IMG]
    assert isinstance(rec, polygon_lod.LazyPolygonRecord), (
        "overview-covered polygons must still come back as lazy records; "
        "reading them all eagerly is the 44x regression this path exists to avoid")
