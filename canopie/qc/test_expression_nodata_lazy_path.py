"""
QC regression tests for expression-based NoData values (e.g. "b7==0",
"b1<123") on the LAZY/windowed read path.

THE BUG THIS PINS (found by a user running a real classification band on a
real, large prediction stack -- NOT by this suite's original fixtures):

    Setting nodata_values=["b7==0"] (exclude class 0 of a 0-3 categorical
    "leaf_age" band) had ZERO effect on process_polygon's exported stats for
    ANY file large enough to take the lazy/windowed read path
    (`_EXPORT_LAZY_THRESHOLD_BYTES`, i.e. most real multi-band scenes).
    Separately noted and NOT fixed here: setting an explicit nodata_values at
    all disables the file's own auto-detected GDAL_NODATA (-9999) exclusion on
    OTHER bands, since that fallback only runs when nodata_values is otherwise
    empty -- so ancillary fill bands leak their -9999 into the stats.

ROOT CAUSE: `process_polygon`'s lazy path (`_build_roi_nodata`) calls
`_per_band_nodata_masks`, which tried `float(v)` on every NoData value and
silently `continue`d past anything that failed to convert -- which is EVERY
expression string, with no warning logged anywhere. The non-lazy/eager path
(`_cached_master_nodata` -> `utils.build_nodata_mask`) already supported
expressions correctly; the lazy path's separate implementation simply never
had that capability.

SYNTAX: equality is written `==`. A bare `=` is deliberately NOT accepted as
a comparison anywhere (NoData or band math) -- see
test_bare_single_equals_is_not_a_comparison.

WHY THE ORIGINAL SUITE MISSED IT: none of this suite's original fixtures
combined (a) an explicit expression-based NoData value with (b) a file large
enough to trigger the lazy path -- the two NoData fixtures (4, 6) both use
numeric GDAL_NODATA auto-detection, which never carries expression strings
in practice. This file specifically forces the lazy path via conftest's
force_lazy_export on an ordinary fixture to close that gap.
"""
import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .generate_fixtures import _rasterize_polygon_mask
from .project_builder import polygon_group_name
from ._helpers import load_raw_npz, assert_close

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.extraction, pytest.mark.io]


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


# ---------------------------------------------------------------------------
# Unit-level: the fixed function in isolation.
# ---------------------------------------------------------------------------
def test_per_band_nodata_masks_supports_expressions():
    from ..project_tab import _per_band_nodata_masks

    band0 = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    band1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    masks = _per_band_nodata_masks([band0, band1], ["b1==0"])

    assert masks[0].tolist() == [[True, False], [False, True]], (
        "b1==0 should flag exactly the two zero pixels in band 0")
    assert not masks[1].any(), "b1==0 must not touch band 1 at all (per-band, not master)"


def test_bare_single_equals_is_not_a_comparison():
    """DECIDED BEHAVIOR: equality must be written `==`. A single `=` is NOT
    accepted as a NoData comparison -- it is silently ignored as an
    unrecognized value, exactly as any other unparseable entry would be."""
    from ..project_tab import _per_band_nodata_masks

    band0 = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    double = _per_band_nodata_masks([band0], ["b1==0"])
    single = _per_band_nodata_masks([band0], ["b1=0"])
    assert double[0].any(), "'b1==0' must mask the zero pixels"
    assert not single[0].any(), "'b1=0' must NOT be treated as a comparison"


def test_per_band_nodata_masks_still_supports_numeric_literals():
    """The fix must not regress the pre-existing numeric-literal behavior,
    which applies to every band (unlike expressions)."""
    from ..project_tab import _per_band_nodata_masks

    band0 = np.array([[-9999.0, 1.0]], dtype=np.float32)
    band1 = np.array([[2.0, -9999.0]], dtype=np.float32)
    masks = _per_band_nodata_masks([band0, band1], [-9999])
    assert masks[0].tolist() == [[True, False]]
    assert masks[1].tolist() == [[False, True]]


def test_per_band_nodata_masks_out_of_range_band_warns_not_crashes(caplog):
    from ..project_tab import _per_band_nodata_masks

    band0 = np.zeros((2, 2), dtype=np.float32)
    masks = _per_band_nodata_masks([band0], ["b5==0"])  # only 1 band present
    assert not masks[0].any(), "an out-of-range band reference must be a no-op, not a crash"


@pytest.mark.parametrize("op,expr", [
    ("<", "b1<50"), ("<=", "b1<=50"), (">", "b1>50"),
    (">=", "b1>=50"), ("!=", "b1!=50"),
])
def test_per_band_nodata_masks_all_operators(op, expr):
    from ..project_tab import _per_band_nodata_masks

    band0 = np.array([30.0, 50.0, 70.0], dtype=np.float32).reshape(1, 3)
    mask = _per_band_nodata_masks([band0], [expr])[0][0]
    expected = {"<": [True, False, False], "<=": [True, True, False],
               ">": [False, False, True], ">=": [False, True, True],
               "!=": [True, False, True]}[op]
    assert mask.tolist() == expected, f"{expr}: got {mask.tolist()}, expected {expected}"


def test_shared_regex_requires_double_equals():
    """`==` parses; a bare `=` deliberately does not."""
    from ..utils import _NODATA_EXPR_RE
    assert _NODATA_EXPR_RE.match("b7==0").groups() == ("b7", "==", "0")
    assert _NODATA_EXPR_RE.match("b7=0") is None, (
        "a single '=' must not parse as a NoData comparison")


def test_parse_nodata_text_accepts_double_equals_expression():
    """The actual UI text-input parser must carry a `==` expression through
    alongside ordinary numeric values."""
    from ..utils import parse_nodata_text
    result = parse_nodata_text("b7==0, -9999")
    assert "b7==0" in result, f"'b7==0' was dropped by parse_nodata_text: {result}"
    assert -9999 in result or -9999.0 in result


# ---------------------------------------------------------------------------
# Integration: process_polygon on the LAZY path, the exact path that was
# completely broken for expressions before the fix.
# ---------------------------------------------------------------------------
def test_process_polygon_expression_nodata_on_lazy_path(
        synthetic_project, force_lazy_export):
    """The real regression: an expression-based NoData value, on a file
    forced through the windowed reader, must actually exclude the pixels it
    targets -- computed independently from the ground-truth array, not by
    trusting process_polygon's own output."""
    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    raw = load_raw_npz(name)  # (H, W, C), native file band order

    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    # Use the SAME rasterization ground truth already relies on elsewhere in
    # this suite, rather than a naive bounding-box slice -- cv2.fillPoly's
    # scanline fill is inclusive on both ends, so a naive half-open numpy
    # slice under-counts the true pixel set (verified: 144 vs the real 169
    # for this exact polygon).
    H, W = spec["height"], spec["width"]
    poly_mask = _rasterize_polygon_mask(spec["polygon"]["points"], H, W)
    band0_roi = raw[:, :, 0][poly_mask]
    threshold = float(np.median(band0_roi))
    expected_excluded = int((band0_roi < threshold).sum())
    assert 0 < expected_excluded < band0_roi.size, (
        "test threshold must split the region non-trivially -- adjust if the "
        "fixture's formula ever changes")

    opts = {"stats": {"mean": True}, "nodata_enabled": True,
           "nodata_values": [f"b1<{threshold}"]}
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=opts)
    by_channel = _rows_by_channel(rows)

    total_pixels = band0_roi.size
    r_count = by_channel["R"]["Pixel Count"]
    assert r_count == total_pixels - expected_excluded, (
        f"expression NoData on the lazy path: got {r_count} valid pixels, "
        f"expected {total_pixels - expected_excluded} "
        f"(total={total_pixels}, excluded={expected_excluded})")


def test_process_polygon_double_equals_excludes_matching_pixels_on_lazy_path(
        synthetic_project, force_lazy_export):
    """A `==` NoData expression must actually drop the matching pixels on the
    lazy path (a bare `=` is not a comparison -- see
    test_bare_single_equals_is_not_a_comparison)."""
    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    raw = load_raw_npz(name)
    H, W = spec["height"], spec["width"]
    poly_mask = _rasterize_polygon_mask(spec["polygon"]["points"], H, W)
    band0_roi = raw[:, :, 0][poly_mask]
    target_value = float(band0_roi.flat[0])

    n_matching = int((band0_roi == target_value).sum())
    assert n_matching >= 1

    opts = {"stats": {"mean": True}, "nodata_enabled": True,
            "nodata_values": [f"b1=={target_value:g}"]}
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=opts)
    count = _rows_by_channel(rows)["R"]["Pixel Count"]

    assert count == band0_roi.size - n_matching, (
        f"'b1=={target_value:g}' should have excluded {n_matching} pixel(s): "
        f"got {count} of {band0_roi.size}")
