"""
Two silent data-loss / wrong-display defects from the 2026-10-05 review plan.

P5 / BUG-021  Undoing a bulk delete reloads the polygon folder to re-read the
              overview -- through the same loader the project open uses, which
              PURGES polygons/.trash. Every other pending undo lost its files:
              undo the second of two deletes and the first can no longer come
              back.
S1 / BUG-023  ProjectTab.apply_aux_modifications is a staticmethod but looked
              up `self._get_polygons_for_filepath` when a mask polygon was only
              in memory -> NameError -> the whole .ax replay failed and the viewer
              showed the unedited image.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtWidgets

pytestmark = [pytest.mark.polygons]

IMG = "somewhere/ortho_cog.tif"


@pytest.fixture
def small_project(qapp, tmp_path):
    from ..project_tab import ProjectTab
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMG))[0]
    for i in range(6):
        g = f"grp_{i}"
        (poly_dir / f"{g}_{base}_polygons.json").write_text(json.dumps({
            "name": g, "group": g, "root": "", "type": "polygon", "coord_space": "image",
            "image_ref_size": {"w": 100, "h": 100}, "points": [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0]],
            "coordinates": {}, "properties": {}}), encoding="utf-8")
    pt = ProjectTab.__new__(ProjectTab)
    QtWidgets.QWidget.__init__(pt)
    pt.project_folder = str(tmp_path)
    pt.multispectral_image_data_groups = {"R1": [IMG]}
    pt.thermal_rgb_image_data_groups = {}
    pt._load_polygons_from_dir()
    pt.all_polygons = pt._load_polygons_from_dir()
    pt.viewer_widgets = []
    pt.undo_stack = QtWidgets.QUndoStack()
    pt._dirty_polygon_entries = set()
    pt._poly_norm_index_invalid = True
    pt.polygon_manager = None
    yield pt, poly_dir
    try:
        pt.close()
        pt.setParent(None)
        pt.deleteLater()
    except Exception:
        pass
    qapp.processEvents()


def _delete(pt, groups):
    from ..project_tab import DeletePolygonCommand
    with pt._bulk_polygon_delete():
        pt.undo_stack.beginMacro("del")
        for g in groups:
            pt.undo_stack.push(DeletePolygonCommand(pt, g, IMG))
        pt.undo_stack.endMacro()


def test_undoing_one_bulk_delete_keeps_the_other_deletes_undoable(small_project):
    import types
    pt, poly_dir = small_project
    # a viewer showing the image: that is what makes undo reload the folder
    viewer = types.SimpleNamespace(image_data=types.SimpleNamespace(filepath=IMG))
    pt.viewer_widgets = [{"viewer": viewer, "image_data": viewer.image_data}]
    _delete(pt, ["grp_0", "grp_1"])
    _delete(pt, ["grp_2", "grp_3"])
    trash = poly_dir / ".trash"
    assert len(list(trash.iterdir())) == 4
    pt.undo()                                  # Ctrl+Z: brings back grp_2, grp_3
    assert {"grp_2", "grp_3"} <= set(pt.all_polygons)
    left = list(trash.iterdir())
    assert len(left) == 2, (
        f"undoing the second delete emptied .trash ({len(left)} file(s) left): the first delete "
        "can no longer be undone")
    pt.undo()                                  # and the first one still can
    assert {"grp_0", "grp_1"} <= set(pt.all_polygons)
    base = os.path.splitext(os.path.basename(IMG))[0]
    for g in ("grp_0", "grp_1", "grp_2", "grp_3"):
        assert (poly_dir / f"{g}_{base}_polygons.json").exists(), f"{g}'s file did not come back"


def test_opening_a_project_still_purges_dead_undo_files(small_project):
    pt, poly_dir = small_project
    _delete(pt, ["grp_4"])
    assert any((poly_dir / ".trash").iterdir())
    pt._load_polygons_from_dir()               # what a project open does
    assert not any((poly_dir / ".trash").iterdir())


def test_mask_polygon_only_in_memory_does_not_break_the_replay(tmp_path):
    from ..project_tab import ProjectTab
    fp = str(tmp_path / "masked.png")
    img = np.full((40, 50, 3), 100, np.uint8)
    base = os.path.splitext(os.path.basename(fp))[0]
    (tmp_path / f"{base}.ax").write_text(json.dumps({
        "mask_polygon": {"enabled": True, "names": ["holes"]},
        "crop_rect": {"x": 2, "y": 3, "width": 40, "height": 30},
        "crop_rect_ref_size": {"w": 50, "h": 40}}), encoding="utf-8")
    all_polygons = {"holes": {fp: {"name": "holes", "type": "polygon", "coord_space": "image",
                                   "image_ref_size": {"w": 50, "h": 40},
                                   "points": [(10, 10), (20, 10), (20, 20), (10, 20)]}}}
    out = ProjectTab.apply_aux_modifications(fp, img.copy(), project_folder=str(tmp_path),
                                             global_mode=False, all_polygons=all_polygons)
    assert out is not None
    assert out.shape[:2] == (30, 40), (
        f"replay returned {getattr(out, 'shape', None)}: the crop was not applied, i.e. the .ax "
        "replay failed (NameError on `self` in the staticmethod)")
