"""
all_polygons is keyed by filepath STRING, and one project routinely holds two
spellings of the same file.

Measured on the real BCI project (C:\\New Folder185):

    project.json multispectral_image_data_groups
        'C:/Users/natha/Downloads/20240716_..._cog.tif'    <- QFileDialog form
    polygons/_overview.json, all 3504 imported crowns
        'C:\\\\Users\\\\natha\\\\Downloads\\\\20240716_..._cog.tif'  <- os.path.normpath

The viewer's image_data.filepath comes from the first; shapefile import writes
the second (polygon_manager._collect_image_info -> os.path.normpath, and
shapefile_io.shapefile_to_json_polygons' target_list). Both spellings are
registered in the polygon index -- the exact index under the literal string,
the normalised index under normpath().lower().

THE INCIDENT: every lookup consulted the two indices in FALLBACK order
(`if not memory_hit`, `else`, `elif`, `if not results`), which makes them
mutually exclusive and lets whichever spelling the exact index happens to hold
shadow the other entirely:

  1. Import 3504 crowns. The forward-slash key is absent from the exact index,
     so the normalised fallback runs and all 3504 render. Correct.
  2. Draw ONE polygon. on_polygon_drawn stores it under the VIEWER's
     forward-slash spelling and registers that key in the exact index.
  3. Refresh. The exact lookup now succeeds -- with exactly one group -- sets
     memory_hit, and the normalised index is never consulted. The viewer draws
     1 polygon instead of 3505, while the Polygon Manager still lists every
     group because all_polygons was never touched.

The same defect on the save side (save_polygons_to_json used the exact index
alone) is why a dragged imported crown snapped back: the move was applied in
memory but its sidecar was never written.
"""
import json
import os

import pytest

from ..project_tab import ProjectTab
from .fixtures_manifest import fixture_image_path


def _two_spellings(fp):
    """The same path written two ways, the second being normpath of the first.

    Uses a redundant '.' component so the two genuinely differ on every
    platform. On Windows the real project's pair is forward- vs back-slashes;
    that exact case is covered separately below.
    """
    raw = os.path.join(os.path.dirname(fp), ".", os.path.basename(fp))
    return raw, os.path.normpath(raw)


def _bare_tab(all_polygons):
    """A ProjectTab shell for the pure-lookup tests.

    ProjectTab.__new__ skips QObject.__init__, so reading an attribute that is
    not already in __dict__ raises RuntimeError from sip rather than
    AttributeError -- which means even hasattr()/getattr(..., default) blow up.
    _ensure_polygon_index probes exactly those, so seed them.
    """
    pt = ProjectTab.__new__(ProjectTab)
    pt.all_polygons = all_polygons
    pt._poly_exact_index = {}
    pt._poly_norm_index = {}
    pt._poly_norm_index_invalid = True
    return pt


class _ImageDataView:
    """image_data seen under a particular spelling of its filepath.

    A view rather than a mutation, so the session-scoped synthetic_project's
    cached ImageData is not altered for other tests.
    """

    def __init__(self, src, filepath):
        self.filepath = filepath
        self.image = src.image


# ---------------------------------------------------------------------------
# The index lookup itself
# ---------------------------------------------------------------------------
def test_exact_key_does_not_shadow_normalised_key():
    """The unit at the heart of it: the lookup is a UNION, not a fallback."""
    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer, fp_import = _two_spellings(fp)
    assert fp_viewer != fp_import, "the two spellings must actually differ"

    pt = _bare_tab({
        "imported_crown": {fp_import: {}},     # shapefile import's spelling
        "drawn_tree1": {fp_viewer: {}},        # the viewer's spelling
    })

    found = dict(pt._poly_index_lookup(fp_viewer))
    assert found.get("drawn_tree1") == fp_viewer
    assert found.get("imported_crown") == fp_import, (
        "the polygon stored under os.path.normpath's spelling of this same "
        "file was not found, because a group registered under the viewer's "
        "spelling shadowed the normalised index -- this is the bug that made "
        "3504 imported crowns vanish the moment one polygon was drawn")


def test_lookup_returns_the_storage_key_not_the_query_key():
    """Callers index all_polygons with what comes back, so it must be the key
    the record is actually stored under, not the path that was asked for."""
    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer, fp_import = _two_spellings(fp)

    pt = _bare_tab({"imported_crown": {fp_import: {"marker": 1}}})

    (group, stored_fp), = pt._poly_index_lookup(fp_viewer)
    assert group == "imported_crown"
    assert pt.all_polygons[group][stored_fp] == {"marker": 1}


def test_lookup_does_not_duplicate_a_group_present_in_both_indices():
    """The common case -- one spelling -- must return each group exactly once."""
    fp = fixture_image_path("rgb_8bit_untiled")
    pt = _bare_tab({"g": {fp: {}}})
    assert pt._poly_index_lookup(fp) == [("g", fp)]


@pytest.mark.skipif(os.name != "nt", reason="forward/backslash pair is Windows-specific")
def test_the_real_projects_slash_pair():
    """The literal pair measured on C:\\New Folder185."""
    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer = fp.replace(os.sep, "/")          # QFileDialog / project.json
    fp_import = os.path.normpath(fp)             # shapefile import
    assert fp_viewer != fp_import

    pt = _bare_tab({
        "imported_crown": {fp_import: {}},
        "drawn_tree1": {fp_viewer: {}},
    })
    assert set(dict(pt._poly_index_lookup(fp_viewer))) == {
        "imported_crown", "drawn_tree1"}


# ---------------------------------------------------------------------------
# Driven through the real load_polygons
# ---------------------------------------------------------------------------
def test_drawing_one_polygon_does_not_hide_imported_ones(
        synthetic_project, viewer_factory, monkeypatch):
    """THE reported bug, driven through the real load_polygons().

    Import N crowns under normpath's spelling, draw one polygon under the
    viewer's spelling, refresh -- all N+1 must render.
    """
    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer, fp_import = _two_spellings(fp)

    src = synthetic_project._imagedata_or_fallback(fp)
    imgd = _ImageDataView(src, fp_viewer)

    viewer = viewer_factory()
    viewer.image_data = imgd
    pm = synthetic_project.convert_cv_to_pixmap(src.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)

    # No root filtering -- this test is about the lookup, not the root filter.
    monkeypatch.setattr(synthetic_project, "get_root_by_filepath", lambda f: None)

    def _poly(name):
        return {
            'name': name, 'group': name, 'root': '', 'type': 'polygon',
            'coord_space': 'image', 'image_ref_size': {'w': 64, 'h': 64},
            'points': [(10, 10), (25, 10), (25, 25), (10, 25)],
        }

    imported = [f"shadow_crown_{i}" for i in range(5)]
    drawn = "shadow_drawn_tree1"
    for g in imported:
        synthetic_project.all_polygons[g] = {fp_import: _poly(g)}
    synthetic_project.all_polygons[drawn] = {fp_viewer: _poly(drawn)}
    synthetic_project._poly_norm_index_invalid = True

    try:
        synthetic_project.load_polygons(viewer, imgd)
        names = {p.get('name') for p in viewer.polygons}
        assert drawn in names, "the newly drawn polygon is missing"
        missing = [g for g in imported if g not in names]
        assert not missing, (
            f"{len(missing)} of {len(imported)} imported polygons did not "
            "render. They are stored under os.path.normpath's spelling of the "
            "image path; the one drawn polygon is stored under the viewer's "
            "spelling, and its exact-index hit suppressed the normalised "
            "lookup entirely. This is the 3504-crowns-vanish bug.")
    finally:
        for g in imported + [drawn]:
            synthetic_project.all_polygons.pop(g, None)
        synthetic_project._poly_norm_index_invalid = True


# ---------------------------------------------------------------------------
# The save side -- why a dragged imported polygon snapped back
# ---------------------------------------------------------------------------
def _save_tab(tmp_path, fp_viewer, all_polygons):
    """A REAL ProjectTab -- save_polygons_to_json reads too many attributes
    through getattr(..., default) for a __new__ shell to survive (see
    _bare_tab)."""
    from .project_builder import build_project_tab

    pt = build_project_tab(str(tmp_path))
    pt.project_folder = str(tmp_path)
    pt.mode = "rgb_only"
    pt.all_polygons = all_polygons
    pt._poly_norm_index_invalid = True
    pt.image_data_groups = {"Root_1": [fp_viewer]}
    pt._dirty_polygon_roots = {"Root_1"}
    pt.root_id_mapping = {"Root_1": 1}
    pt.root_coordinates = {"Root_1": {'latitude': 9.15, 'longitude': -79.85}}
    pt.get_root_by_filepath = lambda f: "Root_1"
    pt.get_viewer_by_filepath = lambda f: None
    return pt


def test_save_writes_polygons_keyed_under_normpath(qapp, fixtures_ready, tmp_path):
    """A root whose filepaths use the viewer's spelling must still save the
    polygons stored under normpath's spelling."""
    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer, fp_import = _two_spellings(fp)
    base = os.path.splitext(os.path.basename(fp))[0]

    pt = _save_tab(tmp_path, fp_viewer, {
        "imported_crown": {fp_import: {
            'name': "imported_crown", 'root': '', 'type': 'polygon',
            'coord_space': 'image', 'image_ref_size': {'w': 64, 'h': 64},
            'points': [(10, 10), (25, 10), (25, 25), (10, 25)],
        }},
    })

    pt.save_polygons_to_json("Root_1")

    out = tmp_path / "polygons" / f"imported_crown_{base}_polygons.json"
    assert out.exists(), (
        "the imported polygon was never written. save_polygons_to_json looked "
        "it up in the exact index using the root's filepath spelling, which "
        "never matches the normpath spelling shapefile import stored it under "
        "-- so a dragged imported crown was updated in memory, silently not "
        "saved, and snapped back to its old position on the next load")
    assert json.loads(out.read_text(encoding='utf-8'))['points']


def test_unmodified_pyramid_records_are_not_rewritten(qapp, fixtures_ready, tmp_path):
    """Saving must not page in every untouched polygon just to write it back.

    A LazyPolygonRecord that has never materialised cannot have been modified,
    and _normalize_points_for_save's first act is to ask for 'points' -- which
    materialises it. On the real project that is 3504 files and ~90 MB of JSON
    parsed on the GUI thread every time one polygon is drawn, because drawing
    marks the whole root dirty.
    """
    from .. import polygon_lod

    fp = fixture_image_path("rgb_8bit_untiled")
    fp_viewer, fp_import = _two_spellings(fp)
    base = os.path.splitext(os.path.basename(fp))[0]

    polygons_dir = tmp_path / "polygons"
    polygons_dir.mkdir()
    src_file = polygons_dir / f"lazy_crown_{base}_polygons.json"
    src_file.write_text(json.dumps({
        'name': "lazy_crown", 'type': 'polygon', 'coord_space': 'image',
        'image_ref_size': {'w': 64, 'h': 64},
        'points': [[10, 10], [25, 10], [25, 25], [10, 25]],
    }), encoding='utf-8')
    mtime_before = src_file.stat().st_mtime_ns

    lazy = polygon_lod.LazyPolygonRecord({
        'name': "lazy_crown", 'group': "lazy_crown", 'root': '',
        'type': 'polygon', 'coord_space': 'image',
        'image_ref_size': {'w': 64, 'h': 64},
        'display_points': [[10, 10], [25, 10], [10, 25]],
        'display_level': polygon_lod.OVERVIEW_LEVEL,
        'full_point_count': 4,
    }, str(src_file))

    pt = _save_tab(tmp_path, fp_viewer, {"lazy_crown": {fp_import: lazy}})
    pt.save_polygons_to_json("Root_1")

    assert lazy.is_materialised is False, (
        "saving materialised an untouched pyramid record -- that is the ~5 s / "
        "90 MB parse polygon_lod exists to avoid, and it now runs on every draw")
    assert src_file.stat().st_mtime_ns == mtime_before, (
        "an untouched polygon's sidecar was rewritten")
