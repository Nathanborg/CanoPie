"""
The CSV export dialog's NoData field ADDS to each image's own NoData
(2026-10-06).

It used to REPLACE it, and it was prefilled from the first .ax the dialog
found in the project folder -- so one image's NoData list was applied to
every image of the export, and an image's own fill value (what the viewer
masks) could be dropped. Each image now always keeps its own rule (.ax values
+ the file's NoData tag, per band); the dialog's values are excluded in
addition, on the full read and on the lazy (windowed) read alike.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_raw_npz

pytestmark = [pytest.mark.extraction]

NAME = "ax_crop_nodata_source"          # .ax: crop 20..60, nodata_values [9999]
SQUARE = [[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]]   # holds the (5, 5) stamp


def _clear(pt):
    for attr in ("_export_image_cache", "_export_cache", "_per_band_nd_cache",
                 "_master_nd_cache", "_scene_stats_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def _poly(pt):
    spec = get_fixture(NAME)
    fp = fixture_image_path(NAME)
    group = polygon_group_name(NAME, spec["polygon"]["name"])
    d = dict(pt.all_polygons[group][fp])
    d["points"] = [list(p) for p in SQUARE]
    return group, fp, d


def _expected(extra_r):
    from .generate_fixtures import _rasterize_polygon_mask
    from ..nodata import tolerance          # how close counts as "the NoData value"
    raw = load_raw_npz(NAME).astype(np.float64)[20:60, 20:60]
    m = _rasterize_polygon_mask(SQUARE, 40, 40)
    out = {}
    for b, ch in enumerate(["R", "G", "B", "band_4"]):
        v = raw[..., b][m]
        keep = np.abs(v - 9999.0) > tolerance(9999.0)          # the image's own NoData
        if b == 0:
            keep &= np.abs(v - extra_r) > tolerance(extra_r)    # the dialog's value
        out[ch] = (float(v[keep].mean()), int(keep.sum()))
    return out


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_dialog_values_are_added_to_each_images_own_nodata(synthetic_project, monkeypatch, lazy):
    from .. import project_tab as project_tab_module
    pt = synthetic_project
    group, fp, d = _poly(pt)
    raw = load_raw_npz(NAME).astype(np.float64)[20:60, 20:60]
    extra_r = float(raw[3, 3, 0])                    # a real R value inside the square
    from ..nodata import tolerance
    assert not np.any(np.abs(raw[2:11, 2:11, 1:] - extra_r) <= tolerance(extra_r)), (
        "pick a value only R holds")
    if lazy:
        monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    _clear(pt)
    try:
        rows, _ = pt.process_polygon(group, fp, d, {}, [], False,
                                     opts={"stats": {"mean": True}, "nodata_values": [extra_r]})
    finally:
        monkeypatch.undo()
        _clear(pt)
    by = {r["Channel"]: r for r in rows if isinstance(r, dict)}
    exp = _expected(extra_r)
    for ch, (mean, n) in exp.items():
        assert by[ch]["Pixel Count"] == n, f"{ch}: {by[ch]['Pixel Count']} px, expected {n}"
        assert by[ch]["Mean"] == pytest.approx(mean, abs=1e-3), f"{ch}: {by[ch]['Mean']} vs {mean}"


def test_the_dialog_does_not_prefill_another_images_nodata(synthetic_project, qapp, tmp_path, monkeypatch):
    """The project folder holds one image's .ax with nodata_values; the dialog
    must not copy them into the field (they would be applied to EVERY image)."""
    import json
    from ..machine_learning_manager import AnalysisOptionsDialog
    (tmp_path / "some_image.ax").write_text(
        json.dumps({"nodata_enabled": True, "nodata_values": [123]}), encoding="utf-8")
    monkeypatch.setattr(synthetic_project, "project_folder", str(tmp_path))
    dlg = AnalysisOptionsDialog(synthetic_project)
    try:
        assert not dlg.chk_nodata.isChecked()
        assert dlg.le_nodata.text().strip() == ""
        assert dlg.get_options()["nodata_values"] == []
    finally:
        dlg.close()
        dlg.deleteLater()
