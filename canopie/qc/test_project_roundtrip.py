"""
QC tests for project persistence (project.json load path).

Everything a user annotates lives in project.json + the polygons/ sidecars.
If a load silently drops a group, remaps a root, or mangles a coordinate, the
work is gone with no error -- so these tests assert the state that actually
comes back into a live ProjectTab, not just that the file parsed.
"""
import json
import os

import pytest

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from .project_builder import (
    build_project_data, write_project_json, build_project_tab,
    point_group_name, polygon_group_name,
)

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.io]


def test_all_polygon_groups_survive_load(synthetic_project):
    """Every group written to project.json must be present after load --
    a silent drop here loses annotations with no error shown."""
    expected = set(build_project_data()["all_polygons"].keys())
    actual = set(synthetic_project.all_polygons.keys())
    missing = expected - actual
    assert not missing, f"{len(missing)} polygon group(s) lost on load, e.g. {sorted(missing)[:5]}"


def test_polygon_geometry_survives_load_exactly(synthetic_project):
    """Vertex coordinates must round-trip bit-for-bit; drift here silently
    moves every downstream statistic to different pixels."""
    written = build_project_data()["all_polygons"]
    for name in ("rgb_8bit_untiled", "multiband_8band_ancillary"):
        spec = get_fixture(name)
        fp = fixture_image_path(name)
        group = polygon_group_name(name, spec["polygon"]["name"])

        src = written[group][fp]
        loaded = synthetic_project.all_polygons[group][fp]

        assert loaded["type"] == src["type"]
        assert len(loaded["points"]) == len(src["points"])
        for (sx, sy), (lx, ly) in zip(src["points"], loaded["points"]):
            assert float(sx) == float(lx) and float(sy) == float(ly), (
                f"{group}: vertex drifted from ({sx},{sy}) to ({lx},{ly})")


def test_point_and_polygon_types_are_preserved(synthetic_project):
    """`type` drives which extraction branch runs -- a point silently loaded
    as a polygon (or vice versa) changes what gets sampled."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)

    pt_group = point_group_name(name, spec["degenerate_point_name"])
    assert synthetic_project.all_polygons[pt_group][fp]["type"] == "point"

    poly_group = polygon_group_name(name, spec["polygon"]["name"])
    assert synthetic_project.all_polygons[poly_group][fp]["type"] == "polygon"


def test_image_ref_size_preserved(synthetic_project):
    """coord_space="image" + image_ref_size is what makes point mapping a
    pure scale; losing ref size would rescale every polygon."""
    name = "ax_crop_nodata_source"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    loaded = synthetic_project.all_polygons[group][fp]

    assert loaded.get("coord_space") == "image"
    ref = loaded.get("image_ref_size") or {}
    # This fixture's annotations are authored in the POST-crop frame.
    assert ref.get("w") == spec["ax"]["crop_rect"]["width"]
    assert ref.get("h") == spec["ax"]["crop_rect"]["height"]


# --------------------------------------------------------------------------
# `properties` surviving the disk round trip
#
# save_polygons_to_json and save_incremental each build their on-disk record
# from a hardcoded dict literal that never included a 'properties' key --
# so every property edit made through the viewer's "Edit properties" dialog
# (image_viewer.py's edit_polygon_properties, which always falls through to
# save_incremental() since ProjectTab has no request_save_polygons method)
# was silently discarded on the very next save. The in-memory copy looked
# fine until the next disk sync, at which point the edit was gone with no
# error anywhere. These tests pin the fix directly against the real
# ProjectTab writers, not a reimplementation of them.
# --------------------------------------------------------------------------

def _poly_with_properties(name, fp, ref_w=100, ref_h=100, properties=None):
    d = {
        'points': [[0, 0], [10, 0], [10, 10]], 'name': name, 'root': '1',
        'coordinates': {'latitude': 0.0, 'longitude': 0.0}, 'type': 'polygon',
        'coord_space': 'image', 'image_ref_size': {'w': ref_w, 'h': ref_h},
    }
    if properties is not None:
        d['properties'] = properties
    return d


def _sidecar_path(project_folder, group, filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(project_folder, 'polygons', f"{group}_{base}_polygons.json")


def _any_existing_fixture_filepath(synthetic_project):
    for file_map in synthetic_project.all_polygons.values():
        for fp in file_map:
            return fp
    raise AssertionError("synthetic_project has no polygons to borrow a filepath from")


def test_properties_survive_save_polygons_to_json(synthetic_project):
    fp = _any_existing_fixture_filepath(synthetic_project)
    group = "__prop_roundtrip_save_json__"
    props = {"DBH_CM": 31.5, "SPECIES": "Ceiba"}
    synthetic_project.all_polygons[group] = {fp: _poly_with_properties(group, fp, properties=props)}
    jf = _sidecar_path(synthetic_project.project_folder, group, fp)
    try:
        synthetic_project.save_polygons_to_json()
        data = json.load(open(jf, encoding="utf-8"))
        assert data.get("properties") == props, (
            f"properties dropped by save_polygons_to_json: wrote {data.get('properties')!r}")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        if os.path.exists(jf):
            os.remove(jf)


def test_properties_survive_save_incremental(synthetic_project):
    fp = _any_existing_fixture_filepath(synthetic_project)
    group = "__prop_roundtrip_save_incremental__"
    props = {"DBH_CM": 42.0, "PLOT": "p9"}
    synthetic_project.all_polygons[group] = {fp: _poly_with_properties(group, fp, properties=props)}
    jf = _sidecar_path(synthetic_project.project_folder, group, fp)
    try:
        synthetic_project._mark_polygon_dirty(group, fp)
        synthetic_project.save_incremental(show_status=False, update_project_json=False)
        data = json.load(open(jf, encoding="utf-8"))
        assert data.get("properties") == props, (
            f"properties dropped by save_incremental: wrote {data.get('properties')!r}")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        if os.path.exists(jf):
            os.remove(jf)


def test_edit_properties_round_trips_through_a_disk_sync(synthetic_project):
    """THE actual user-visible bug: a property edit must still be there after
    the same save -> reload cycle a running app performs constantly (root
    navigation, project reopen). Reproduces it end-to-end via the real
    ProjectTab writer and the real disk-sync reader, not a shortcut."""
    fp = _any_existing_fixture_filepath(synthetic_project)
    group = "__prop_roundtrip_edit_cycle__"
    props = {"CENSUS_STATUS": "Completed", "TAG": "193935"}
    synthetic_project.all_polygons[group] = {fp: _poly_with_properties(group, fp, properties=props)}
    jf = _sidecar_path(synthetic_project.project_folder, group, fp)
    try:
        synthetic_project._mark_polygon_dirty(group, fp)
        synthetic_project.save_incremental(show_status=False, update_project_json=False)

        synthetic_project._sync_all_polygons_from_disk()

        reloaded = synthetic_project.all_polygons.get(group, {}).get(fp, {})
        assert reloaded.get("properties") == props, (
            "properties did not survive save -> _sync_all_polygons_from_disk -- "
            f"got {reloaded.get('properties')!r}, expected {props!r}. This is the "
            "bug: an edit made through the viewer's properties dialog looks fine "
            "until the next disk sync, then silently reverts.")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        if os.path.exists(jf):
            os.remove(jf)


def test_hand_drawn_polygon_without_properties_key_saves_cleanly(synthetic_project):
    """A hand-drawn polygon never gets a 'properties' key at all (confirmed:
    project_tab.py's drawing path never sets one) -- the writer must not
    crash on that, and must not invent a non-empty dict."""
    fp = _any_existing_fixture_filepath(synthetic_project)
    group = "__prop_roundtrip_hand_drawn__"
    synthetic_project.all_polygons[group] = {fp: _poly_with_properties(group, fp, properties=None)}
    jf = _sidecar_path(synthetic_project.project_folder, group, fp)
    try:
        synthetic_project._mark_polygon_dirty(group, fp)
        synthetic_project.save_incremental(show_status=False, update_project_json=False)
        data = json.load(open(jf, encoding="utf-8"))
        assert data.get("properties") == {}, (
            f"hand-drawn polygon (no properties key) wrote properties={data.get('properties')!r}, "
            "expected an empty dict")
    finally:
        synthetic_project.all_polygons.pop(group, None)
        if os.path.exists(jf):
            os.remove(jf)


def test_roots_and_mode_restored(synthetic_project):
    pt = synthetic_project
    assert pt.mode == "rgb_only", f"mode came back as {pt.mode!r}"
    assert len(pt.multispectral_root_names) == len(FIXTURES), (
        f"{len(pt.multispectral_root_names)} roots loaded, expected {len(FIXTURES)}")
    for spec in FIXTURES:
        assert spec["name"] in pt.multispectral_root_names, f"root {spec['name']} missing"


def test_each_root_maps_to_its_own_image(synthetic_project):
    """Root -> file grouping must survive; a mixed-up mapping would export
    polygons against the wrong raster."""
    pt = synthetic_project
    for spec in FIXTURES:
        files = pt.multispectral_image_data_groups.get(spec["name"], [])
        assert len(files) == 1, f"root {spec['name']} has {len(files)} files, expected 1"
        assert os.path.normcase(files[0]) == os.path.normcase(fixture_image_path(spec["name"]))


def test_root_id_mapping_is_dense_and_unique(synthetic_project):
    """Root IDs appear in every exported CSV row; duplicates or gaps make
    exports ambiguous."""
    mapping = synthetic_project.root_id_mapping
    assert mapping, "no root_id_mapping built on load"
    ids = list(mapping.values())
    assert len(ids) == len(set(ids)), f"duplicate root IDs: {ids}"


def test_load_is_idempotent(qapp, tmp_path):
    """Loading the same project twice must give the same state -- a loader
    that appends instead of replacing would double every group."""
    folder = tmp_path / "twice"
    pt = build_project_tab(str(folder))
    first_groups = set(pt.all_polygons.keys())
    first_roots = list(pt.multispectral_root_names)

    pt.load_project(str(folder))

    assert set(pt.all_polygons.keys()) == first_groups, "group set changed on reload"
    assert list(pt.multispectral_root_names) == first_roots, "roots changed on reload"
    for group, per_file in pt.all_polygons.items():
        for fp, entry in per_file.items():
            assert isinstance(entry, dict), (
                f"{group}: entry became {type(entry).__name__} after reload -- "
                "the loader may be wrapping/accumulating instead of replacing")


def test_written_json_is_valid_and_complete(tmp_path):
    """The on-disk artifact must contain the keys load_project requires;
    missing any of them makes the project unopenable."""
    folder = tmp_path / "schema"
    path = write_project_json(str(folder))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("all_polygons", "current_root_index", "root_offset",
                "root_coordinates", "mode", "rgb_folder_path"):
        assert key in data, f"project.json missing required key {key!r}"

    assert data["all_polygons"], "project.json wrote no polygons at all"


def test_missing_project_json_is_handled(qapp, tmp_path):
    """Pointing the loader at a folder with no project.json must fail
    gracefully (the app shows a warning) rather than raising."""
    from PyQt5 import QtWidgets
    from ..project_tab import ProjectTab

    empty = tmp_path / "not_a_project"
    empty.mkdir()
    pt = ProjectTab("qc-empty", QtWidgets.QTabWidget())
    pt.load_project(str(empty))  # must not raise
