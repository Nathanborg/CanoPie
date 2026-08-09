"""
Saving must NOT rescale polygons to whatever the viewer is displaying.

THE INCIDENT (project C:/New Folder189, 3504 imported crowns): drawing a single
polygon made every shapefile-derived polygon vanish. They were not deleted --
`save_polygons_to_json` normalises every polygon it writes into "the file's
current effective basis", and that basis came from `image_data.image.shape`.
On a COG that is a ~16x decimated PREVIEW (3001x3131 for a 48031x50101
raster), so the save divided all 3504 crowns' coordinates by 16 and collapsed
them into a blob in the top-left corner.

It fired on DRAWING because drawing marks the root dirty, and a dirty root
saves the WHOLE root -- every polygon on the file, not just the new one.
"""
import json
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from ..project_tab import ProjectTab
from ..shapefile_io import _raw_image_dims
from .fixtures_manifest import fixture_image_path


class _FakeImageData:
    def __init__(self, filepath, arr):
        self.filepath = filepath
        self.image = arr


class _FakeViewer:
    """A viewer showing a DECIMATED preview of a much larger raster."""
    def __init__(self, filepath, preview):
        self.image_data = _FakeImageData(filepath, preview)


def _tab(tmp_path, fp, viewer, root="Root1"):
    # __new__ + QWidget.__init__ gives a valid C++ object (the save path calls
    # Qt on self) without running ProjectTab.__init__, which would need a whole
    # main window.
    pt = ProjectTab.__new__(ProjectTab)
    QtWidgets.QWidget.__init__(pt)
    pt.project_folder = str(tmp_path)
    pt.all_polygons = {}
    pt.image_data_groups = {root: [fp]}
    pt.multispectral_image_data_groups = {root: [fp]}
    pt.thermal_rgb_image_data_groups = {}
    pt.root_names = [root]
    pt.multispectral_root_names = [root]
    pt.current_root_index = 0
    pt.root_id_mapping = {root: 1}
    pt.root_coordinates = {}
    pt.viewer_widgets = [{"viewer": viewer, "image_data": viewer.image_data}]
    pt._dirty_polygon_roots = {root}
    pt.get_viewer_by_filepath = lambda p: viewer if p == fp else None
    return pt


@pytest.fixture
def setup(tmp_path, qapp):
    fp = fixture_image_path("rgb_8bit_untiled")
    true_h, true_w = _raw_image_dims(fp)
    assert true_h and true_w
    # The viewer holds a 4x-decimated preview, exactly as a COG does.
    preview = np.zeros((max(1, true_h // 4), max(1, true_w // 4), 3), np.uint8)
    viewer = _FakeViewer(fp, preview)
    tab = _tab(tmp_path, fp, viewer)
    return tab, fp, viewer, true_h, true_w


def _saved(tmp_path, group, fp):
    base = os.path.splitext(os.path.basename(fp))[0]
    p = tmp_path / "polygons" / f"{group}_{base}_polygons.json"
    assert p.exists(), f"{p} was not written"
    return json.loads(p.read_text(encoding="utf-8"))


def test_save_does_not_rescale_to_the_preview(setup, tmp_path):
    """THE regression, reduced to one polygon."""
    tab, fp, viewer, true_h, true_w = setup
    pts = [(1000.0, 2000.0), (1200.0, 2000.0), (1200.0, 2300.0)]
    tab.all_polygons["crown_0"] = {fp: {
        "points": list(pts), "coord_space": "image",
        "image_ref_size": {"w": true_w, "h": true_h},
        "name": "crown_0", "root": "", "type": "polygon",
    }}
    tab._poly_exact_index = {fp: {"crown_0"}}
    tab._ensure_polygon_index = lambda *a, **k: None

    tab.save_polygons_to_json(root_name="Root1")

    got = _saved(tmp_path, "crown_0", fp)
    assert got["image_ref_size"] == {"w": true_w, "h": true_h}, (
        f"the saved basis became {got['image_ref_size']} -- the viewer's "
        f"decimated preview was treated as the coordinate space")
    for (ax, ay), (bx, by) in zip(pts, got["points"]):
        assert (ax, ay) == pytest.approx((bx, by)), (
            f"saving MOVED the polygon: {(ax, ay)} -> {(bx, by)}. On the real "
            f"project this divided 3504 crowns by 16 and collapsed them into "
            f"the top-left corner.")


def test_save_is_idempotent_across_repeated_saves(setup, tmp_path):
    """Drawing repeatedly must not walk polygons toward the origin.

    Each save rescaled by eff/basis, so the damage compounded: 16x per save.
    """
    tab, fp, viewer, true_h, true_w = setup
    pts = [(1000.0, 2000.0), (1200.0, 2000.0), (1200.0, 2300.0)]
    tab.all_polygons["crown_0"] = {fp: {
        "points": list(pts), "coord_space": "image",
        "image_ref_size": {"w": true_w, "h": true_h},
        "name": "crown_0", "root": "", "type": "polygon",
    }}
    tab._poly_exact_index = {fp: {"crown_0"}}
    tab._ensure_polygon_index = lambda *a, **k: None

    for _ in range(3):
        tab._dirty_polygon_roots = {"Root1"}
        tab.save_polygons_to_json(root_name="Root1")

    got = _saved(tmp_path, "crown_0", fp)
    for (ax, ay), (bx, by) in zip(pts, got["points"]):
        assert (ax, ay) == pytest.approx((bx, by)), (
            f"three saves moved the polygon {(ax, ay)} -> {(bx, by)}")


def test_a_polygon_stamped_with_a_stale_basis_is_still_rebased(setup, tmp_path):
    """The rescale itself is correct and must survive -- it is what repairs a
    polygon whose recorded basis genuinely differs from the raster."""
    tab, fp, viewer, true_h, true_w = setup
    # Deliberately stamped in preview space, as pre-fix drawing did.
    tab.all_polygons["legacy"] = {fp: {
        "points": [(100.0, 200.0)], "coord_space": "image",
        "image_ref_size": {"w": true_w // 4, "h": true_h // 4},
        "name": "legacy", "root": "", "type": "polygon",
    }}
    tab._poly_exact_index = {fp: {"legacy"}}
    tab._ensure_polygon_index = lambda *a, **k: None

    tab.save_polygons_to_json(root_name="Root1")

    got = _saved(tmp_path, "legacy", fp)
    assert got["image_ref_size"] == {"w": true_w, "h": true_h}
    sx = true_w / float(true_w // 4)
    sy = true_h / float(true_h // 4)
    assert got["points"][0] == pytest.approx([100.0 * sx, 200.0 * sy]), (
        "a genuinely stale basis must still be rebased onto the raster")


# ---------------------------------------------------------------------------
# Every writer must agree on ONE basis
# ---------------------------------------------------------------------------
def test_import_stamps_the_same_basis_polygon_basis_hw_resolves(qapp, tmp_path):
    """Imported and drawn polygons must land in the same coordinate space.

    Shapefile import stamps `image_ref_size` from the raster header; everything
    else resolves it through ProjectTab.polygon_basis_hw. If those two ever
    diverge, a project ends up with two incompatible coordinate spaces -- which
    is exactly what produced 3504 crowns at 48031x50101 sitting alongside
    hand-drawn polygons at 3001x3131.
    """
    fp = fixture_image_path("rgb_8bit_untiled")
    true_h, true_w = _raw_image_dims(fp)

    preview = np.zeros((max(1, true_h // 4), max(1, true_w // 4), 3), np.uint8)
    viewer = _FakeViewer(fp, preview)
    tab = _tab(tmp_path, fp, viewer)

    basis_h, basis_w = tab.polygon_basis_hw(viewer)
    assert (basis_h, basis_w) == (true_h, true_w), (
        "polygon_basis_hw no longer resolves the raster header, so it no "
        "longer matches what shapefile import stamps")


def test_import_does_not_use_a_post_ax_size(qapp):
    """The import basis must not become a post-crop/resize size.

    `_size_after_ax_fast_from_file` is a nested function, never a ProjectTab
    method, so the import branch calling it never ran and the raw header was
    used by accident. If that helper is ever promoted to a real method, the
    import would silently start stamping a post-.ax size while every other
    writer still used the raw header.
    """
    import inspect
    from ..polygon_manager import PolygonManager
    src = inspect.getsource(PolygonManager.on_import_shapefile)
    # Strip comments: the fix documents the dead call in prose, and a naive
    # substring check would match its own explanation.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_size_after_ax_fast_from_file" not in code, (
        "on_import_shapefile CALLS _size_after_ax_fast_from_file again -- the "
        "import basis must stay the raw raster header so it matches "
        "ProjectTab.polygon_basis_hw")


def test_saving_refreshes_the_overview(setup, tmp_path):
    """A dragged polygon must not reappear at its old position after reopening.

    The overview caches each polygon's DECIMATED display points and the fast
    load path reads them INSTEAD of the sidecars. save_incremental refreshed
    it; save_polygons_to_json (the path a refresh takes) did not -- so a
    polygon could be dragged, written correctly to its sidecar, and still come
    back at its old position on the next open.
    """
    tab, fp, viewer, true_h, true_w = setup
    tab.all_polygons["crown_0"] = {fp: {
        "points": [(1000.0, 2000.0), (1200.0, 2000.0), (1200.0, 2300.0)],
        "coord_space": "image",
        "image_ref_size": {"w": true_w, "h": true_h},
        "name": "crown_0", "root": "", "type": "polygon",
    }}
    tab._poly_exact_index = {fp: {"crown_0"}}
    tab._ensure_polygon_index = lambda *a, **k: None

    calls = []
    tab._write_polygon_overview = lambda *a, **k: calls.append(1)

    tab.save_polygons_to_json(root_name="Root1")

    assert calls, (
        "save_polygons_to_json wrote sidecars without refreshing the overview; "
        "the stale display points it leaves behind are what the fast load path "
        "reads, so a moved polygon reverts on the next open")
