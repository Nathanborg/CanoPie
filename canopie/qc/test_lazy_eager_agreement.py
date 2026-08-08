"""
QC regression tests for the lazy (windowed raster_reader) vs eager (whole-
array) read paths.

_get_export_image silently routes a file down one of two completely different
readers depending on its size and .ax contents. Both must produce identical
numbers -- a divergence here would mean CSV values change purely because a
file crossed a size threshold, which is exactly the class of silent drift this
suite exists to catch. The thresholds are plain module-level ints, so the
lazy path is forced on small fixtures via conftest's force_lazy_export rather
than by generating multi-GB files.

CACHE HAZARD: _get_export_image memoizes on (path, mtimes) with no awareness
of which reader produced the entry, so the eager result would otherwise be
served straight back to the lazy run. Every test here clears
_export_image_cache between the two halves; without that these tests would
pass vacuously by comparing a result against itself.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_ground_truth, assert_close, expected_channel_names

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.io, pytest.mark.extraction]

STATS_OPTS = {"stats": {"mean": True, "median": True, "std": True, "quantiles": [25, 75]}}

# Fixtures whose .ax (if any) leaves them windowable, so the lazy path can
# actually engage: fixture 5 has no .ax at all; fixture 7's .ax is crop+nodata,
# neither of which is in _AX_BLOCKS_WINDOWING.
LAZY_ELIGIBLE = ["hyperspectral_200band", "ax_crop_nodata_source"]


def _clear_export_cache(pt):
    cache = getattr(pt, "_export_image_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


def _run_polygon(pt, name):
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    rows, _ = pt.process_polygon(
        group, fp, pt.all_polygons[group][fp], {}, [], False, opts=STATS_OPTS)
    return _rows_by_channel(rows)


@pytest.mark.parametrize("name", LAZY_ELIGIBLE)
def test_process_polygon_lazy_matches_eager(synthetic_project, monkeypatch, name):
    """Same polygon, same fixture, both readers -- every stat must agree, and
    both must agree with the committed ground truth."""
    from .. import project_tab as project_tab_module

    pt = synthetic_project
    gt = load_ground_truth(name)
    spec = get_fixture(name)

    _clear_export_cache(pt)
    eager = _run_polygon(pt, name)
    assert eager, f"{name}: eager run produced no rows"

    _clear_export_cache(pt)
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    lazy = _run_polygon(pt, name)
    assert lazy, f"{name}: lazy run produced no rows"

    assert set(eager) == set(lazy), (
        f"{name}: different channels between readers: {set(eager)} vs {set(lazy)}")

    for ch in eager:
        for col in ("Mean", "Median", "Standard Deviation", "Q25", "Q75", "Pixel Count"):
            if col not in eager[ch]:
                continue
            assert_close(eager[ch][col], lazy[ch].get(col), tol=1e-6,
                         msg=f"{name}/{ch} {col}: eager vs lazy")

    # ...and both agree with ground truth, so "they match each other" can't be
    # satisfied by both being wrong in the same way.
    channel_names = expected_channel_names(spec["bands"])
    for b, ch in enumerate(channel_names):
        band_gt = gt["polygon"]["bands"][str(b)]["process_polygon"]
        if band_gt["count"] == 0 or ch not in eager:
            continue
        assert_close(eager[ch]["Mean"], band_gt["mean"], tol=1e-2,
                     msg=f"{name}/{ch} eager Mean vs ground truth")


def test_lazy_path_actually_engaged(synthetic_project, monkeypatch):
    """Guard against the above passing vacuously: prove that with the
    threshold patched, _get_export_image really does hand back a
    raster_reader.LazyChannels rather than a plain ndarray."""
    from .. import project_tab as project_tab_module
    from ..raster_reader import LazyChannels

    pt = synthetic_project
    fp = fixture_image_path("hyperspectral_200band")

    _clear_export_cache(pt)
    eager_img, _ = pt._get_export_image(fp)
    assert isinstance(eager_img, np.ndarray), (
        f"expected a plain ndarray at default threshold, got {type(eager_img).__name__}")

    _clear_export_cache(pt)
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    lazy_img, _ = pt._get_export_image(fp)
    assert isinstance(lazy_img, LazyChannels), (
        f"expected LazyChannels with threshold patched, got {type(lazy_img).__name__}")


def test_ax_windowability_gate():
    """_ax_is_windowable decides which files may take the lazy path at all.
    It's a plain key-membership check that is easy to silently break by adding
    a new .ax operation without extending _AX_BLOCKS_WINDOWING -- which would
    route edited files down a reader that cannot reproduce that edit."""
    from ..project_tab import ProjectTab

    crop_ax = get_fixture("ax_crop_nodata_source")["ax"]
    expr_ax = get_fixture("ax_band_expression_source")["ax"]

    assert ProjectTab._ax_is_windowable(ProjectTab, crop_ax) is True, (
        "crop + nodata must stay windowable -- a crop is just an offset window")
    assert ProjectTab._ax_is_windowable(ProjectTab, expr_ax) is False, (
        "band_expression appends a band derived from the full cube and must "
        "force the eager path")
    assert ProjectTab._ax_is_windowable(ProjectTab, {}) is True
    assert ProjectTab._ax_is_windowable(ProjectTab, None) is True


def test_crop_offset_identical_in_both_readers(synthetic_project, monkeypatch):
    """The .ax crop is applied by SHIFTING THE READ WINDOW on the lazy path but
    by SLICING THE ARRAY on the eager path -- two genuinely different
    implementations of the same offset. An off-by-one in either would show up
    as different pixel values for the same polygon."""
    from .. import project_tab as project_tab_module

    name = "ax_crop_nodata_source"
    gt = load_ground_truth(name)
    pt = synthetic_project

    _clear_export_cache(pt)
    eager = _run_polygon(pt, name)

    _clear_export_cache(pt)
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    lazy = _run_polygon(pt, name)

    # Ground truth for this fixture is computed in the POST-crop frame, so
    # agreeing with it proves both readers cropped to the same origin.
    band0 = gt["polygon"]["bands"]["0"]["process_polygon"]
    assert_close(eager["R"]["Mean"], band0["mean"], tol=1e-2, msg="eager crop offset")
    assert_close(lazy["R"]["Mean"], band0["mean"], tol=1e-2, msg="lazy crop offset")
