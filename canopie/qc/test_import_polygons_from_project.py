"""
QC tests for "Import Polygons from Project" (GPS-based nearest-image
matching) and the two bugs behind it that this pass fixes:

1. GPS AUTO-ASSIGNMENT REGRESSION -- a polygon drawn on a non-GeoTIFF image
   used to get its real-world `coordinates` (EXIF GPS) written synchronously
   at creation time. A later performance refactor split saving into a fast
   dirty-flag autosave (`_flush_dirty_polygons`) and Quick Save
   (`save_incremental`), neither of which ever called the EXIF fallback --
   only the rarely-triggered full `save_polygons_to_json()` did. Since the
   whole GPS-matching pipeline below silently skips any polygon whose
   `coordinates` are null, a freshly drawn polygon was invisible to it
   unless a full save happened to run first. Fixed by extracting the
   existing cache-first-then-EXIF logic into `ProjectTab._resolve_root_coordinates`
   and calling it from both fast paths too.

2. root_mapping.json PATH MISMATCH -- written by ProjectTab into the
   PROJECT folder, but read by polygon_manager.py from the app's own
   install directory, so it (almost) never loaded -- silently disabling
   RGB/thermal fan-out (`run_correction_process`) and the root-mapping type
   guess (layer 2 of `import_polygons_from_files`'s 3-layer filename
   fallback). Fixed by resolving the project folder via
   `self.parent().project_folder` at both read sites.

`import_polygons_from_files`'s 3-layer fallback (exact/canonicalized
filename match -> root_mapping type guess -> current-viewer last resort)
was NOT broken and is exercised here directly, in isolation per layer, to
pin it against a real ProjectTab (not the file's existing MagicMock-based
tests) now that layer 2 is actually reachable.
"""
import json
import os

import pytest
from PyQt5 import QtWidgets

from .fixtures_manifest import fixture_image_path
from .project_builder import build_project_tab

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.polygons, pytest.mark.io]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def project_tab(qapp, tmp_path):
    """A fresh, non-session-scoped ProjectTab per test -- these tests write
    real sidecar files (polygons/, root_mapping.json, gps_cache.json) and
    mutate all_polygons/_dirty_polys, so a dedicated instance per test is
    simpler and safer than borrowing the shared synthetic_project."""
    pt = build_project_tab(str(tmp_path))
    yield pt
    for attr in ("_poly_idle_timer", "_autosave_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass


@pytest.fixture()
def polygon_manager_factory(qapp):
    """Same pattern as ml_manager_factory in conftest.py -- also a QDialog
    with its own UI; leaving instances un-closed across a file was the
    documented cause of escalating slowness in that file."""
    created = []

    def _make(parent_tab):
        from canopie.polygon_manager import PolygonManager
        mgr = PolygonManager(parent_tab)
        created.append(mgr)
        qapp.processEvents()
        return mgr

    yield _make

    for mgr in created:
        try:
            mgr.close()
        except Exception:
            pass
        try:
            mgr.setParent(None)
        except Exception:
            pass
        try:
            mgr.deleteLater()
        except Exception:
            pass
    qapp.processEvents()
    qapp.processEvents()


def _draw_like_polygon(pt, group, fp):
    """Build the exact same on-disk-shape dict on_polygon_drawn builds --
    no 'coordinates' key at all -- and mark it dirty the same way, without
    needing a real viewer/sender."""
    pt.all_polygons.setdefault(group, {})[fp] = {
        'points': [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]],
        'coord_space': 'image',
        'image_ref_size': {'w': 100, 'h': 100},
        'name': group,
        'root': pt.root_id_mapping.get(pt.get_root_by_filepath(fp), 0),
        'type': 'polygon',
    }
    pt._mark_polygon_dirty(group, fp)


def _read_sidecar(pt, group, fp):
    base = os.path.splitext(os.path.basename(fp))[0]
    path = os.path.join(pt.project_folder, "polygons", f"{group}_{base}_polygons.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fix 1 -- GPS auto-assignment on the fast save paths
# ---------------------------------------------------------------------------
def test_flush_dirty_polygons_populates_coordinates_from_exif(project_tab, monkeypatch):
    """THE regression test: the save that fires ~200ms after a draw must now
    enrich `coordinates`, not just write the raw dict."""
    fp = fixture_image_path("rgb_8bit_untiled")
    group = "__gps_autosave_test__"
    monkeypatch.setattr(project_tab, "get_gps_coordinates",
                         lambda filepath: {'latitude': 12.5, 'longitude': -70.25})

    _draw_like_polygon(project_tab, group, fp)
    project_tab._flush_dirty_polygons()

    saved = _read_sidecar(project_tab, group, fp)
    assert saved.get("coordinates") == {'latitude': 12.5, 'longitude': -70.25}


def test_save_incremental_populates_coordinates_from_exif(project_tab, monkeypatch):
    """Same regression, the other fast path (Ctrl+Q / Quick Save)."""
    fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    group = "__gps_quicksave_test__"
    monkeypatch.setattr(project_tab, "get_gps_coordinates",
                         lambda filepath: {'latitude': 33.1, 'longitude': 44.2})

    _draw_like_polygon(project_tab, group, fp)
    project_tab.save_incremental(show_status=False, update_project_json=False)

    saved = _read_sidecar(project_tab, group, fp)
    assert saved.get("coordinates") == {'latitude': 33.1, 'longitude': 44.2}


def test_flush_dirty_polygons_does_not_re_resolve_existing_coordinates(project_tab, monkeypatch):
    """A polygon that already carries valid coordinates (e.g. re-saved after
    an edit) must not trigger another EXIF lookup -- correctness AND the
    perf property the whole cache-first design exists for."""
    fp = fixture_image_path("rgb_8bit_untiled")
    group = "__gps_no_reresolve_test__"
    calls = {"n": 0}

    def spy(filepath):
        calls["n"] += 1
        return {'latitude': 1.0, 'longitude': 2.0}
    monkeypatch.setattr(project_tab, "get_gps_coordinates", spy)

    _draw_like_polygon(project_tab, group, fp)
    project_tab.all_polygons[group][fp]['coordinates'] = {'latitude': 9.0, 'longitude': 8.0}
    project_tab._flush_dirty_polygons()

    assert calls["n"] == 0, "EXIF was consulted even though coordinates were already present"
    saved = _read_sidecar(project_tab, group, fp)
    assert saved.get("coordinates") == {'latitude': 9.0, 'longitude': 8.0}


def test_no_gps_available_does_not_crash_fast_save(project_tab, monkeypatch):
    """Graceful degradation: EXIF genuinely has nothing -- the save must
    still complete and write a file, not raise."""
    fp = fixture_image_path("rgb_8bit_png")
    group = "__gps_missing_test__"
    monkeypatch.setattr(project_tab, "get_gps_coordinates",
                         lambda filepath: {'latitude': None, 'longitude': None})

    _draw_like_polygon(project_tab, group, fp)
    project_tab._flush_dirty_polygons()  # must not raise

    saved = _read_sidecar(project_tab, group, fp)
    assert saved.get("points"), "the polygon itself must still be written"
    coords = saved.get("coordinates") or {}
    assert coords.get("latitude") is None and coords.get("longitude") is None


# ---------------------------------------------------------------------------
# Fix 2 -- root_mapping.json path mismatch
# ---------------------------------------------------------------------------
def test_run_correction_process_reads_root_mapping_from_project_folder(project_tab, tmp_path):
    """Real proof the path fix works: a REAL root_mapping.json on disk in
    the project folder (no builtins.open mocking) drives fan-out to BOTH
    listed images, not just the one closest image."""
    from canopie.polygon_manager import PolygonManager

    manager = PolygonManager.__new__(PolygonManager)
    manager.parent = lambda: project_tab

    fp_a = fixture_image_path("rgb_8bit_untiled")
    fp_b = fixture_image_path("rgb_8bit_png")
    root_a = project_tab.get_root_by_filepath(fp_a)
    root_id = str(project_tab.root_id_mapping[root_a])

    # Real file, real project folder -- exactly what save_root_mapping_json writes.
    with open(os.path.join(project_tab.project_folder, "root_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({root_id: {"RGB": os.path.basename(fp_a), "thermal": os.path.basename(fp_b)}}, f)

    jsons_dir = tmp_path / "jsons"
    jsons_dir.mkdir()
    manager.jsons_dir = str(jsons_dir)
    manager.polygons_dir = str(jsons_dir)
    manager.corrected_dir = str(jsons_dir)

    with open(jsons_dir / "Grp_root1_polygons_results.json", "w", encoding="utf-8") as f:
        json.dump([{"closest_image": os.path.basename(fp_a), "distance_meters": 0.1}], f)
    with open(jsons_dir / "Grp_root1_polygons.json", "w", encoding="utf-8") as f:
        json.dump({"points": [[1, 1], [2, 2]], "root": root_id,
                   "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}, f)

    created = manager.run_correction_process("src", "tgt")

    names = {os.path.basename(p) for p in created}
    assert f"Grp_{os.path.splitext(os.path.basename(fp_a))[0]}_polygons.json" in names
    assert f"Grp_{os.path.splitext(os.path.basename(fp_b))[0]}_polygons.json" in names, (
        "fan-out to the second (thermal) image never happened -- "
        "root_mapping.json was not found at the project-folder path")


def test_import_layer2_root_mapping_fallback_now_reachable(project_tab, polygon_manager_factory, tmp_path):
    """Layer 2 of the 3-layer filename fallback: a corrected JSON whose
    image base cannot be found by name at all, resolved via a REAL
    root_mapping.json in the project folder.

    Deliberately targets a fixture that is NOT current_root_index's root
    (rgb_16bit_tiled_bip_cog, not rgb_8bit_untiled) and pins current_root to
    a third, unrelated fixture -- otherwise layer 3's "current viewer / root"
    last resort can coincidentally land on the right image even when layer 2
    itself is dead, silently defeating this test's isolation."""
    mgr = polygon_manager_factory(project_tab)
    target_fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    root_name = project_tab.get_root_by_filepath(target_fp)
    root_id = project_tab.root_id_mapping[root_name]
    mgr.current_root = project_tab.get_root_by_filepath(fixture_image_path("hyperspectral_200band"))

    with open(os.path.join(project_tab.project_folder, "root_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({str(root_id): {"RGB": os.path.basename(target_fp)}}, f)

    corrected = tmp_path / f"LayerTwo_totally_unresolvable_name_8b_polygons.json"
    with open(corrected, "w", encoding="utf-8") as f:
        json.dump({"points": [[5, 5], [15, 15]], "root": str(root_id),
                   "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}, f)

    mgr.import_polygons_from_files([str(corrected)])

    assert "LayerTwo" in project_tab.all_polygons
    assert target_fp in project_tab.all_polygons["LayerTwo"], (
        "layer-2 root_mapping fallback did not resolve to the mapped image")


# ---------------------------------------------------------------------------
# import_polygons_from_files -- the other two fallback layers, in isolation
# ---------------------------------------------------------------------------
def test_import_layer1_exact_filename_match(project_tab, polygon_manager_factory, tmp_path):
    mgr = polygon_manager_factory(project_tab)
    target_fp = fixture_image_path("rgb_16bit_tiled_bip_cog")
    root_name = project_tab.get_root_by_filepath(target_fp)
    root_id = project_tab.root_id_mapping[root_name]

    corrected = tmp_path / "LayerOne_rgb_16bit_tiled_bip_cog_polygons.json"
    with open(corrected, "w", encoding="utf-8") as f:
        json.dump({"points": [[3, 3], [6, 6]], "root": str(root_id),
                   "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}, f)

    mgr.import_polygons_from_files([str(corrected)])

    assert target_fp in project_tab.all_polygons.get("LayerOne", {})


def test_import_layer1_canonicalized_suffix_match(project_tab, polygon_manager_factory, tmp_path):
    """'rgb_8bit_untiled_radiance' has no file of that exact name, but
    canonicalizes (strips the trailing modality token 'radiance') to the
    same key as the real 'rgb_8bit_untiled' image."""
    mgr = polygon_manager_factory(project_tab)
    target_fp = fixture_image_path("rgb_8bit_untiled")
    root_name = project_tab.get_root_by_filepath(target_fp)
    root_id = project_tab.root_id_mapping[root_name]

    corrected = tmp_path / "LayerOneCanon_rgb_8bit_untiled_radiance_polygons.json"
    with open(corrected, "w", encoding="utf-8") as f:
        json.dump({"points": [[7, 7], [9, 9]], "root": str(root_id),
                   "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}, f)

    mgr.import_polygons_from_files([str(corrected)])

    assert target_fp in project_tab.all_polygons.get("LayerOneCanon", {})


def test_import_layer3_current_viewer_last_resort(project_tab, polygon_manager_factory, tmp_path):
    """Nothing resolves by name or root_mapping -- falls back to whatever
    PolygonManager.current_root points at (no open viewer needed for this
    branch, see _fallback_current_root_and_first_viewer)."""
    mgr = polygon_manager_factory(project_tab)
    target_fp = fixture_image_path("rgb_8bit_untiled")
    mgr.current_root = project_tab.get_root_by_filepath(target_fp)

    corrected = tmp_path / "LayerThree_no_such_image_anywhere_polygons.json"
    with open(corrected, "w", encoding="utf-8") as f:
        json.dump({"points": [[1, 2], [3, 4]], "root": "9999",
                   "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}, f)

    mgr.import_polygons_from_files([str(corrected)])

    assert target_fp in project_tab.all_polygons.get("LayerThree", {}), (
        "last-resort current-viewer fallback did not place the polygon")


# ---------------------------------------------------------------------------
# Graceful degradation in the GPS-matching step itself
# ---------------------------------------------------------------------------
def test_polygon_without_coordinates_is_skipped_not_crashed(project_tab, tmp_path):
    """A source JSON with coordinates: null (e.g. from before Fix 1, or a
    project that was never saved) must be skipped, not raise -- matching
    today's documented 'No valid target coordinates to process' behavior."""
    from canopie.polygon_manager import PolygonManager
    from scipy.spatial import KDTree

    manager = PolygonManager.__new__(PolygonManager)
    manager.parent = lambda: project_tab

    src = tmp_path / "NoGps_root1_polygons.json"
    with open(src, "w", encoding="utf-8") as f:
        json.dump({"points": [[1, 1]], "coordinates": {"latitude": None, "longitude": None}}, f)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    tree = KDTree([[0.0, 0.0]])
    manager.process_single_json_with_processor(str(src), tree, ["dummy.tif"], [(0.0, 0.0)], str(out_dir))  # must not raise

    out_path = out_dir / "NoGps_root1_polygons_results.json"
    # No valid target coordinates -> the function returns before writing anything.
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# End to end: the real button handler, source project -> target project
# ---------------------------------------------------------------------------
def test_on_import_polygons_from_project_end_to_end(project_tab, polygon_manager_factory,
                                                       monkeypatch, tmp_path, qapp):
    """Drives on_import_polygons_from_project() for real: GPS matching,
    correction/fan-out, and final import all connect, with the source
    polygon's own coordinates populated through the now-fixed Fix-1 path
    (not hand-typed into the source JSON)."""
    from canopie.loaders import ImageProcessor

    # ---- source project: a second, independent ProjectTab sharing the same
    # underlying fixture images on disk (build_project_tab always points
    # current_folder_path at the one shared fixtures directory). ----
    source_pt = build_project_tab(str(tmp_path / "source_project"))

    all_fps = sorted({fp for fps in source_pt.multispectral_image_data_groups.values() for fp in fps})
    canned = {os.path.normpath(fp): {'latitude': 10.0 + i, 'longitude': 20.0 + i}
              for i, fp in enumerate(all_fps)}

    target_fp = fixture_image_path("rgb_8bit_untiled")
    monkeypatch.setattr(source_pt, "get_gps_coordinates",
                         lambda filepath: canned.get(os.path.normpath(filepath)))

    src_group = "E2EImportGroup"
    _draw_like_polygon(source_pt, src_group, target_fp)
    source_pt._flush_dirty_polygons()  # writes polygons/E2EImportGroup_..._polygons.json with real coordinates

    # ---- target project (the one under test) ----
    def fake_batch_extract(self, selected_files):
        out = []
        for fp in selected_files:
            c = canned.get(os.path.normpath(fp))
            if c:
                out.append({'filename': os.path.normpath(fp), **c})
        return out
    monkeypatch.setattr(ImageProcessor, "batch_extract_gps_with_exiftool", fake_batch_extract)

    mgr = polygon_manager_factory(project_tab)
    source_polygons_dir = os.path.join(source_pt.project_folder, "polygons")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                         staticmethod(lambda *a, **k: source_polygons_dir))

    mgr.on_import_polygons_from_project()

    assert src_group in project_tab.all_polygons, (
        f"no group named {src_group!r} landed in the target project; "
        f"got groups: {list(project_tab.all_polygons)}")
    assert target_fp in project_tab.all_polygons[src_group], (
        "the polygon landed on the wrong image -- GPS nearest-match picked "
        f"something other than {os.path.basename(target_fp)}: "
        f"{list(project_tab.all_polygons[src_group])}")
