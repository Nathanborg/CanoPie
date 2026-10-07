"""
A point on a NoData pixel is left out of the CSV on BOTH read paths (BUG-036).

THE BUG THIS PINS
-----------------
`process_polygon` reads large rasters (> _EXPORT_LAZY_THRESHOLD_BYTES -- the
BCI orthomosaic, any COG) through the windowed LazyChannels path, where the
full-frame NoData masks are deliberately NOT built: they are deferred to each
ROI via `lazy_nd_vals`. The polygon branch applies them (`_build_roi_nodata`);
the POINT branch never did. So on every large file a point sitting on NoData
was exported as a row whose mean IS the fill value -- 9999.0 in every band of
the ax_crop_nodata_source fixture -- while the eager path (small files) omits
it. Same point, same file, different CSV, depending only on file size.

It surfaced as an order-dependent failure of test_process_polygon_csv: the
export cache is keyed by (path, mtime) only, so a lazy entry left behind by
test_lazy_eager_agreement was served to the eager point test.
"""
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import point_group_name
from ._helpers import load_ground_truth, expected_channel_names, assert_close

pytestmark = [pytest.mark.extraction, pytest.mark.io]

NAME = "ax_crop_nodata_source"      # crop + NoData 9999; two of its points sit on NoData


def _clear(pt):
    c = getattr(pt, "_export_image_cache", None)
    if isinstance(c, dict):
        c.clear()


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_points_on_nodata_are_empty_rows_on_both_read_paths(synthetic_project, monkeypatch, lazy):
    from .. import project_tab as project_tab_module
    if lazy:
        monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    pt = synthetic_project
    spec = get_fixture(NAME)
    gt = load_ground_truth(NAME)
    fp = fixture_image_path(NAME)
    nodata = gt["nodata_value"]
    channel_names = expected_channel_names(spec["bands"])
    _clear(pt)
    try:
        seen_nodata_point = False
        for p in spec["points"]:
            group = point_group_name(NAME, p["name"])
            rows, _ = pt.process_polygon(group, fp, pt.all_polygons[group][fp], {}, [], False,
                                         opts={"stats": {"mean": True}})
            by_channel = {r.get("Channel"): r for r in rows if isinstance(r, dict)}
            truth = gt["points"][p["name"]]["values"]
            for b, ch in enumerate(channel_names):
                if nodata is not None and abs(truth[b] - nodata) < 1e-3:
                    seen_nodata_point = True
                    # kept, EMPTY (Nathan 2026-10-06: one row per band, empty
                    # where NoData) -- never the fill value as data
                    row = by_channel.get(ch)
                    assert row is not None, (
                        f"{'lazy' if lazy else 'eager'} path: point {p['name']} on NoData lost "
                        f"its {ch} row")
                    assert row.get("Mean") is None and row.get("Pixel Count") == 0, (
                        f"{'lazy' if lazy else 'eager'} path: point {p['name']} sits on NoData "
                        f"({nodata}) but channel {ch} was exported with Mean={row.get('Mean')}, "
                        f"Pixel Count={row.get('Pixel Count')}")
                else:
                    assert ch in by_channel, f"{p['name']} {ch}: expected a row"
                    assert_close(by_channel[ch].get("Mean"), truth[b], tol=1e-2,
                                 msg=f"{p['name']} {ch}")
        assert seen_nodata_point, "fixture no longer has a point on NoData -- test is vacuous"
    finally:
        _clear(pt)       # never leave a lazy-path entry for the next test
