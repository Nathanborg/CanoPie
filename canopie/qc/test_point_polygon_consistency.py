"""
QC regression tests for point-vs-polygon extraction consistency.

The core promise this suite exists to protect: sampling the SAME pixels two
different ways must give the same numbers. Specifically --
  1. a 1-vertex "polygon" and a "point" at the same coordinate agree,
  2. a polygon's Mean equals the arithmetic mean of individually-sampled
     points at every one of its interior pixels,
  3. process_polygon and MachineLearningManager.export_csv_data agree
     per-pixel where neither is masking anything.

(2) is deliberately done against a TINY inline 3x3 polygon rather than the
manifest's larger ones: it needs one process_polygon call per interior pixel,
so 9 calls stays fast while still being an exhaustive, not sampled, check.
"""
import csv

import pytest

from .fixtures_manifest import FIXTURES, fixture_image_path, get_fixture
from .project_builder import point_group_name, polygon_group_name, degenerate_group_name
from ._helpers import load_ground_truth, load_raw_npz, assert_close, expected_channel_names

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.extraction, pytest.mark.polygons]

# One representative fixture per distinct extraction shape: 3-band cv2/BGR,
# 8-band tifffile/native, and single-band ("Gray" channel) -- the three
# branches process_polygon actually has.
CONSISTENCY_FIXTURES = ["rgb_8bit_untiled", "multiband_8band_ancillary", "gray_8bit_png"]

MEAN_OPTS = {"stats": {"mean": True}}


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


def _polygon_dict(name, points, ptype, ref_w, ref_h):
    """Same shape project_builder writes -- coord_space="image" with
    image_ref_size matching the export frame makes point mapping a no-op."""
    return {
        "points": [[float(x), float(y)] for (x, y) in points],
        "coord_space": "image",
        "image_ref_size": {"w": ref_w, "h": ref_h},
        "name": name,
        "root": "1",
        "type": ptype,
    }


@pytest.mark.parametrize("name", CONSISTENCY_FIXTURES)
def test_point_equals_colocated_1px_polygon(synthetic_project, name):
    """A `type: "point"` annotation and a 1-vertex `type: "polygon"` at the
    identical coordinate must produce byte-identical per-band values."""
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    deg = spec["degenerate_point_name"]

    pg = point_group_name(name, deg)
    prows, _ = synthetic_project.process_polygon(
        pg, fp, synthetic_project.all_polygons[pg][fp], {}, [], False, opts=MEAN_OPTS)

    dg = degenerate_group_name(name, deg)
    drows, _ = synthetic_project.process_polygon(
        dg, fp, synthetic_project.all_polygons[dg][fp], {}, [], False, opts=MEAN_OPTS)

    pby, dby = _rows_by_channel(prows), _rows_by_channel(drows)
    assert set(pby) == set(dby) and pby, f"{name}: channel sets differ ({set(pby)} vs {set(dby)})"
    for ch in pby:
        assert_close(pby[ch]["Mean"], dby[ch]["Mean"], tol=0.0,
                     msg=f"{name}/{ch}: point vs 1px-polygon")


@pytest.mark.parametrize("name", CONSISTENCY_FIXTURES)
def test_polygon_mean_equals_mean_of_its_points(synthetic_project, name):
    """EXHAUSTIVE (not sampled) check on a small 3x3 region: extract every
    interior pixel as its own point annotation, average those, and require it
    to equal what polygon-mode extraction reports as the Mean."""
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    ref_w, ref_h = spec["width"], spec["height"]

    # A 3x3 block well inside the image, and safely away from fixture 6-style
    # NoData holes (these three fixtures have none in this region anyway).
    x0, y0 = 6, 6
    corners = [(x0, y0), (x0 + 2, y0), (x0 + 2, y0 + 2), (x0, y0 + 2)]

    poly_rows, _ = synthetic_project.process_polygon(
        "consistency_poly", fp,
        _polygon_dict("consistency_poly", corners, "polygon", ref_w, ref_h),
        {}, [], False, opts=MEAN_OPTS)
    poly_by = _rows_by_channel(poly_rows)
    assert poly_by, f"{name}: polygon extraction produced no rows"

    interior = [(x, y) for y in range(y0, y0 + 3) for x in range(x0, x0 + 3)]
    per_channel_values = {}
    for (x, y) in interior:
        rows, _ = synthetic_project.process_polygon(
            f"consistency_pt_{x}_{y}", fp,
            _polygon_dict(f"consistency_pt_{x}_{y}", [(x, y)], "point", ref_w, ref_h),
            {}, [], False, opts=MEAN_OPTS)
        for ch, row in _rows_by_channel(rows).items():
            per_channel_values.setdefault(ch, []).append(float(row["Mean"]))

    for ch, poly_row in poly_by.items():
        vals = per_channel_values.get(ch)
        assert vals, f"{name}/{ch}: polygon reported this channel but no point did"
        assert len(vals) == len(interior), (
            f"{name}/{ch}: expected {len(interior)} point samples, got {len(vals)}")
        assert_close(poly_row["Mean"], sum(vals) / len(vals), tol=1e-6,
                     msg=f"{name}/{ch}: polygon Mean vs mean-of-points")
        assert poly_row["Pixel Count"] == len(interior), (
            f"{name}/{ch}: polygon Pixel Count {poly_row['Pixel Count']} != {len(interior)}")


def test_process_polygon_and_ml_export_agree_per_pixel(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """Cross-tool: on a fixture with no NoData anywhere, the per-pixel values
    ML export writes for a polygon must equal what process_polygon samples
    point-by-point at those same coordinates. This is the check that would
    catch a channel-order or coordinate-convention drift between the two
    independent extraction implementations."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() == group)

    calls = {"n": 0}
    def fake_get_item(*a, **k):
        calls["n"] += 1
        return ("All Pixel Values", True) if calls["n"] == 1 else ("3x3", True)

    from PyQt5 import QtWidgets
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(fake_get_item))
    out_csv = tmp_path / "cross.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out_csv), "")))
    mgr.export_csv_data()

    with open(out_csv, newline="", encoding="utf-8") as f:
        ml_rows = list(csv.DictReader(f))
    assert ml_rows, "ML export produced no rows"

    ref_w, ref_h = spec["width"], spec["height"]
    channel_names = expected_channel_names(spec["bands"])

    # Spot-check a spread of pixels rather than all of them: both paths are
    # uniform per-pixel loops, so a handful across the polygon is sufficient
    # to catch an ordering/offset drift, and keeps this to a few extractions.
    for ml_row in ml_rows[:: max(1, len(ml_rows) // 8)][:8]:
        xi, yi = int(ml_row["x"]), int(ml_row["y"])
        pt_rows, _ = synthetic_project.process_polygon(
            f"cross_pt_{xi}_{yi}", fp,
            _polygon_dict(f"cross_pt_{xi}_{yi}", [(xi, yi)], "point", ref_w, ref_h),
            {}, [], False, opts=MEAN_OPTS)
        pt_by = _rows_by_channel(pt_rows)
        for ch in channel_names:
            if ch not in ml_row or ml_row[ch] == "":
                continue
            assert ch in pt_by, f"{name} ({xi},{yi}): process_polygon emitted no {ch} row"
            assert_close(float(ml_row[ch]), pt_by[ch]["Mean"], tol=1e-6,
                         msg=f"{name} pixel ({xi},{yi}) {ch}: ML export vs process_polygon point")
