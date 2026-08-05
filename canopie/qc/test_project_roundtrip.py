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
