"""
Builds a synthetic CanoPie project (project.json) from fixtures_manifest.py
and loads it into a real ProjectTab via ProjectTab.load_project(), which
skips its QFileDialog entirely when project_folder is passed explicitly
(confirmed at project_tab.py:24152) -- no GUI interaction needed.

Mode "rgb_only" is used with explicit multispectral_root_names /
multispectral_image_data_groups (one root per fixture) so grouping is exact
and doesn't depend on folder-scan + batch-size heuristics.

all_polygons is written directly into project.json (confirmed at
project_tab.py: `self.all_polygons = defaultdict(dict, project_data["all_polygons"])`
inside load_project) -- no separate polygons/*.json sidecar files are needed
for load_project itself; those are only consulted by other features (e.g.
mask_polygon lookups) that none of these fixtures use.

IMPORTANT -- all_polygons[group][filepath] is a PLAIN DICT, not a list of
dicts. Confirmed directly against the actually-exercised CSV-export
preprocessing (project_tab.py, the polygons_to_process builder feeding
ExportWorker): `for filepath, polygon_dict in polygons_data.items(): ...
polygons_to_process.append((group_name, filepath, polygon_dict))` passes
polygon_dict straight through with no list-unwrapping, and
`ProjectTab.process_polygon`'s first line is `polygon_dict.get("type", ...)`
-- a dict method call that would crash on a list. MachineLearningManager's
own reads (train_models, export_csv_data, generate_thumbnails) make the same
assumption. An earlier draft of this module wrapped each entry in a
single-element list based on a stale, never-independently-verified claim
from an earlier phase of development; that was wrong and made
MachineLearningManager crash immediately on any group (AttributeError: 'list'
object has no attribute 'get') even though ProjectTab.process_polygon
"worked" only because the test code that called it happened to unwrap `[0]`
manually first.
"""
import json
import os

from .fixtures_manifest import FIXTURES, FIXTURE_IMAGES_DIR, fixture_image_path


def _ref_size_for(spec):
    """The (w, h) polygon coordinates were authored against. For the two
    .ax fixtures, points/polygon are given in the POST-edit frame (see
    fixtures_manifest.py's per-fixture comments), so image_ref_size must
    match that post-edit shape -- coord_space="image" scales ref_size
    against whatever _get_export_image actually returns."""
    ax = spec.get("ax") or {}
    crop = ax.get("crop_rect")
    if crop:
        return crop["width"], crop["height"]
    return spec["width"], spec["height"]


def build_project_data():
    """Returns the project.json dict (also written to disk by build_project)."""
    root_names = []
    groups = {}
    all_polygons = {}

    for idx, spec in enumerate(FIXTURES):
        root_name = spec["name"]
        root_id = str(idx + 1)
        root_names.append(root_name)
        fp = fixture_image_path(spec["name"])
        groups[root_name] = [fp]

        ref_w, ref_h = _ref_size_for(spec)

        def _polygon_dict(name, points, ptype):
            return {
                "points": [[float(x), float(y)] for (x, y) in points],
                "coord_space": "image",
                "image_ref_size": {"w": ref_w, "h": ref_h},
                "name": name,
                "root": root_id,
                "type": ptype,
            }

        for p in spec["points"]:
            group = f"{p['name']}__{root_name}"
            all_polygons.setdefault(group, {})[fp] = _polygon_dict(group, [(p["x"], p["y"])], "point")

        poly = spec["polygon"]
        group = f"{poly['name']}__{root_name}"
        all_polygons.setdefault(group, {})[fp] = _polygon_dict(group, poly["points"], "polygon")

        deg_name = spec["degenerate_point_name"]
        deg_point = next(p for p in spec["points"] if p["name"] == deg_name)
        deg_group = f"degenerate_poly_{deg_name}__{root_name}"
        all_polygons.setdefault(deg_group, {})[fp] = _polygon_dict(
            deg_group, [(deg_point["x"], deg_point["y"])], "polygon"
        )

    return {
        "project_name": "qc",
        "mode": "rgb_only",
        "root_offset": 0,
        "all_polygons": all_polygons,
        "current_root_index": 0,
        "root_coordinates": {},
        "rgb_folder_path": FIXTURE_IMAGES_DIR,
        "batch_size": 1,
        "multispectral_root_names": root_names,
        "multispectral_image_data_groups": groups,
    }


def write_project_json(project_folder):
    os.makedirs(project_folder, exist_ok=True)
    data = build_project_data()
    path = os.path.join(project_folder, "project.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def build_project_tab(project_folder, tab_widget=None):
    """Writes project.json into project_folder and loads it into a real,
    headless ProjectTab instance."""
    from PyQt5 import QtWidgets
    from ..project_tab import ProjectTab

    write_project_json(project_folder)

    if tab_widget is None:
        tab_widget = QtWidgets.QTabWidget()
    pt = ProjectTab("qc", tab_widget)
    pt.load_project(project_folder)
    return pt


def point_group_name(fixture_name, point_name):
    """Exact all_polygons group-name key this module used for a point entry."""
    return f"{point_name}__{fixture_name}"


def polygon_group_name(fixture_name, polygon_name):
    """Exact all_polygons group-name key this module used for the interior
    polygon entry (polygon_name is spec["polygon"]["name"], e.g.
    "poly_interior" or fixture 6's "poly_straddle_hole_a")."""
    return f"{polygon_name}__{fixture_name}"


def degenerate_group_name(fixture_name, degenerate_point_name):
    """Exact all_polygons group-name key for the 1-vertex degenerate
    "polygon" co-located with a fixture's degenerate_point_name."""
    return f"degenerate_poly_{degenerate_point_name}__{fixture_name}"
