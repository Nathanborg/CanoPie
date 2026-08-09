"""
Polygons are stored in FULL-RESOLUTION raster pixels -- never in whatever the
viewer happens to be displaying.

For a COG, `image_data.image` is a decimated PREVIEW. On the real BCI project
that is 3001x3131 for a 48031x50101 raster, a ~16x factor. Everything that
stamped `image_ref_size` from the viewer image wrote polygons in preview
pixels, while shapefile import (which probes the file header) wrote them in
raster pixels -- two incompatible bases in one project, measured on disk:

    newly drawn (258 files)   root '1'   image_ref_size  3001 x 3131
    imported   (2687 files)   root ''    image_ref_size 48031 x 50101

which produced: new polygons vanishing (rescaled ~16x on load, landing off the
image), and a dragged polygon snapping back (the preview LEVEL changes with
zoom, so the polygon was re-encoded against a different basis than it was read
with).
"""
import json
import os

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path


def _tab(tmp_path):
    from ..project_tab import ProjectTab
    pt = ProjectTab.__new__(ProjectTab)
    pt.project_folder = str(tmp_path)
    pt.all_polygons = {}
    pt.image_data_groups = {}
    pt.multispectral_image_data_groups = {}
    pt.thermal_rgb_image_data_groups = {}
    return pt


class _FakeImageData:
    def __init__(self, filepath, arr):
        self.filepath = filepath
        self.image = arr


def test_basis_is_the_raster_not_the_preview(tmp_path):
    """The core invariant."""
    fp = fixture_image_path("rgb_8bit_untiled")
    from ..shapefile_io import _raw_image_dims
    true_h, true_w = _raw_image_dims(fp)
    assert true_h and true_w

    tab = _tab(tmp_path)
    # A viewer showing a 4x-decimated preview, as a COG does.
    preview = np.zeros((max(1, true_h // 4), max(1, true_w // 4), 3), np.uint8)
    idata = _FakeImageData(fp, preview)

    h, w = tab.polygon_basis_hw(None, idata)
    assert (h, w) == (true_h, true_w), (
        f"basis came back as {w}x{h}; the decimated preview "
        f"({preview.shape[1]}x{preview.shape[0]}) must never define the "
        f"coordinate space polygons are stored in")


def test_basis_is_stable_across_preview_levels(tmp_path):
    """THE test that would have caught the drag-snaps-back bug.

    The preview level changes with zoom. If the basis follows it, the same
    polygon is written against one basis and read against another.
    """
    fp = fixture_image_path("rgb_8bit_untiled")
    tab = _tab(tmp_path)
    from ..shapefile_io import _raw_image_dims
    true_h, true_w = _raw_image_dims(fp)

    bases = set()
    for step in (1, 2, 4, 8):
        arr = np.zeros((max(1, true_h // step), max(1, true_w // step), 3), np.uint8)
        bases.add(tab.polygon_basis_hw(None, _FakeImageData(fp, arr)))

    assert bases == {(true_h, true_w)}, (
        f"the basis moved with the preview level ({bases}) -- a polygon saved "
        f"at one zoom would be reinterpreted at another, which is exactly why "
        f"a dragged polygon snapped back to its old position")


def test_in_memory_only_image_still_works(tmp_path):
    """No file to probe (unsaved/synthetic): the array IS full resolution."""
    tab = _tab(tmp_path)
    arr = np.zeros((37, 41, 3), np.uint8)
    assert tab.polygon_basis_hw(None, _FakeImageData(None, arr)) == (37, 41)


# ---------------------------------------------------------------------------
# Migration of the already mis-stamped files
# ---------------------------------------------------------------------------
def _write_sidecar(poly_dir, group, img_fp, pts, ref_w, ref_h):
    base = os.path.splitext(os.path.basename(img_fp))[0]
    (poly_dir / f"{group}_{base}_polygons.json").write_text(json.dumps({
        'name': group, 'group': group, 'root': '1', 'type': 'polygon',
        'coord_space': 'image',
        'image_ref_size': {'w': ref_w, 'h': ref_h},
        'points': pts, 'lod': {'4': [[0, 0]]},
    }), encoding="utf-8")


def _prepare(tmp_path):
    fp = fixture_image_path("rgb_8bit_untiled")
    from ..shapefile_io import _raw_image_dims
    true_h, true_w = _raw_image_dims(fp)
    tab = _tab(tmp_path)
    tab.image_data_groups = {"Root1": [fp]}
    poly_dir = tmp_path / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)
    return tab, fp, poly_dir, true_h, true_w


def test_migration_rescales_preview_basis_polygons(tmp_path):
    tab, fp, poly_dir, true_h, true_w = _prepare(tmp_path)
    pw, ph = true_w // 4, true_h // 4
    _write_sidecar(poly_dir, "drawn_on_preview", fp,
                   [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0]], pw, ph)

    n_fixed, n_skip, _ = tab.migrate_polygon_basis(dry_run=False)
    assert n_fixed == 1 and n_skip == 0

    base = os.path.splitext(os.path.basename(fp))[0]
    got = json.loads((poly_dir / f"drawn_on_preview_{base}_polygons.json")
                     .read_text(encoding="utf-8"))
    assert got["image_ref_size"] == {"w": true_w, "h": true_h}
    sx, sy = true_w / float(pw), true_h / float(ph)
    assert got["points"][0] == pytest.approx([10.0 * sx, 20.0 * sy])
    assert "lod" not in got, "the stored pyramid was built on the OLD basis"


def test_migration_is_idempotent(tmp_path):
    """Running it twice must not double-scale -- that would destroy the data
    it just repaired."""
    tab, fp, poly_dir, true_h, true_w = _prepare(tmp_path)
    _write_sidecar(poly_dir, "drawn", fp, [[10.0, 20.0]], true_w // 4, true_h // 4)

    tab.migrate_polygon_basis(dry_run=False)
    base = os.path.splitext(os.path.basename(fp))[0]
    path = poly_dir / f"drawn_{base}_polygons.json"
    once = json.loads(path.read_text(encoding="utf-8"))

    n_fixed, n_skip, _ = tab.migrate_polygon_basis(dry_run=False)
    assert n_fixed == 0 and n_skip == 1
    assert json.loads(path.read_text(encoding="utf-8")) == once


def test_migration_leaves_correct_files_alone(tmp_path):
    tab, fp, poly_dir, true_h, true_w = _prepare(tmp_path)
    _write_sidecar(poly_dir, "imported", fp, [[1.0, 2.0]], true_w, true_h)
    n_fixed, n_skip, _ = tab.migrate_polygon_basis(dry_run=False)
    assert n_fixed == 0 and n_skip == 1


def test_migration_dry_run_writes_nothing(tmp_path):
    tab, fp, poly_dir, true_h, true_w = _prepare(tmp_path)
    _write_sidecar(poly_dir, "drawn", fp, [[10.0, 20.0]], true_w // 4, true_h // 4)
    base = os.path.splitext(os.path.basename(fp))[0]
    path = poly_dir / f"drawn_{base}_polygons.json"
    before = path.read_text(encoding="utf-8")

    n_fixed, _, details = tab.migrate_polygon_basis(dry_run=True)
    assert n_fixed == 1 and details
    assert path.read_text(encoding="utf-8") == before


def test_migration_matches_the_right_image_for_underscored_groups(tmp_path):
    """Group names contain underscores, so the image cannot be found by
    splitting the filename -- it must match a KNOWN image base."""
    tab, fp, poly_dir, true_h, true_w = _prepare(tmp_path)
    _write_sidecar(poly_dir, "guayacan_21294.0_crown", fp,
                   [[8.0, 9.0]], true_w // 4, true_h // 4)
    n_fixed, n_skip, _ = tab.migrate_polygon_basis(dry_run=False)
    assert n_fixed == 1, "the sidecar's image was not resolved"
