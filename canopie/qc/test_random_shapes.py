"""
QC tests for random shape generation (random_shapes_generator).

This module has a history of "the shapes are all bunched in one corner"
reports, which turned out to be correct area-proportional sampling over a
fragmented valid mask rather than a bug -- so these tests pin the properties
that actually matter (count, containment, spacing, stratification) instead of
eyeballing distribution.
"""
import numpy as np
import pytest

from ..random_shapes_generator import generate_random_shape_pointlists


def _params(**over):
    p = {
        "shape_type": "Point",
        "count": 20,
        "diameter": 1.0,
        "width": 1.0,
        "height": 1.0,
        "min_dist_border": 0.0,
        "min_dist_shapes": 0.0,
        "scope": "Current Image Only",
        "restrict_valid": False,
        "stratify": False,
    }
    p.update(over)
    return p


def _points_of(pointlists):
    return [pt for pts in pointlists for pt in pts]


def test_generates_requested_count_unrestricted():
    """Fast path (no valid-area restriction): every requested shape should be
    placed, since the whole frame is available."""
    out = generate_random_shape_pointlists(_params(count=25), H=64, W=64)
    assert len(out) == 25, f"expected 25 shapes, got {len(out)}"


def test_all_points_inside_image_bounds():
    out = generate_random_shape_pointlists(_params(count=200), H=40, W=50)
    for (x, y) in _points_of(out):
        assert 0 <= x <= 50, f"x={x} outside [0,50]"
        assert 0 <= y <= 40, f"y={y} outside [0,40]"


@pytest.mark.parametrize("shape_type,min_pts", [
    ("Point", 1),
    ("Rectangle", 4),
    ("Circle", 8),
])
def test_shape_types_produce_expected_geometry(shape_type, min_pts):
    """Points are single vertices; rectangles are corner lists; circles are
    polygon approximations with many vertices."""
    out = generate_random_shape_pointlists(
        _params(shape_type=shape_type, count=5, diameter=6.0, width=6.0, height=4.0),
        H=64, W=64)
    assert len(out) == 5
    for pts in out:
        assert len(pts) >= min_pts, (
            f"{shape_type}: got {len(pts)} vertices, expected >= {min_pts}")


def test_shapes_stay_inside_valid_mask_when_restricted():
    """THE core promise of restrict_valid: with a valid region occupying only
    part of the frame, no shape may be centered outside it."""
    H = W = 60
    image = np.zeros((H, W, 1), dtype=np.float32)
    image[:, :] = -9999.0
    image[10:30, 10:30, :] = 100.0          # the only valid block

    out = generate_random_shape_pointlists(
        _params(count=40, restrict_valid=True), H, W,
        image=image, nodata_mask=None, nodata_values=[-9999])

    assert out, "restricted generation produced nothing at all"
    for (x, y) in _points_of(out):
        assert 10 <= x <= 30 and 10 <= y <= 30, (
            f"shape at ({x:.1f},{y:.1f}) landed outside the valid block")


def test_all_nodata_image_falls_back_without_crashing():
    """REGRESSION TEST for a real crash found by this suite.

    When restricted placement finds zero valid pixels, the generator logs
    "falling back to full frame" and calls its internal _sample_rect helper --
    but passed only ONE of the two required insets, so it raised
    `TypeError: _sample_rect() missing 1 required positional argument: 'iy'`
    instead of falling back. The path is reached exactly on heavily/entirely
    masked rasters, which is precisely the data the restrict-to-valid option
    exists for.

    Documented intent (from the code's own comment) is a graceful full-frame
    fallback, so that is what is asserted here."""
    H = W = 32
    image = np.full((H, W, 1), -9999.0, dtype=np.float32)
    out = generate_random_shape_pointlists(
        _params(count=10, restrict_valid=True), H, W,
        image=image, nodata_mask=None, nodata_values=[-9999])

    assert isinstance(out, list), f"expected a list fallback, got {type(out).__name__}"
    assert len(out) == 10, f"fallback should still place the requested shapes, got {len(out)}"
    for (x, y) in _points_of(out):
        assert 0 <= x <= W and 0 <= y <= H, f"fallback point ({x},{y}) outside the frame"


def test_min_distance_between_shapes_is_respected():
    """Spacing constraint must actually hold pairwise, not just be accepted
    as a parameter."""
    min_dist = 8.0
    out = generate_random_shape_pointlists(
        _params(shape_type="Point", count=12, min_dist_shapes=min_dist), H=100, W=100)

    centers = _points_of(out)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = (dx * dx + dy * dy) ** 0.5
            assert d >= min_dist - 1e-6, (
                f"points {i} and {j} are {d:.2f}px apart, below min_dist {min_dist}")


def test_impossible_spacing_degrades_gracefully():
    """Asking for more spacing than the frame can hold must return fewer
    shapes, not hang or raise."""
    out = generate_random_shape_pointlists(
        _params(shape_type="Point", count=50, min_dist_shapes=40.0), H=50, W=50)
    assert isinstance(out, list)
    assert len(out) < 50, "impossible spacing should yield fewer shapes"


def test_stratify_spreads_across_disconnected_patches():
    """Without stratification, sampling is area-proportional, so a patch
    holding most of the valid pixels legitimately receives most of the shapes.
    With stratify=True, a much smaller patch must still get meaningful
    representation -- that is the entire point of the option."""
    H = W = 80
    image = np.full((H, W, 1), -9999.0, dtype=np.float32)
    image[0:40, 0:40, :] = 100.0       # big patch: 1600 px
    image[70:76, 70:76, :] = 100.0     # small patch: 36 px (~2%)

    def small_patch_share(stratify):
        out = generate_random_shape_pointlists(
            _params(count=60, restrict_valid=True, stratify=stratify), H, W,
            image=image, nodata_mask=None, nodata_values=[-9999])
        pts = _points_of(out)
        assert pts, "no shapes generated"
        in_small = sum(1 for (x, y) in pts if x >= 69 and y >= 69)
        return in_small / len(pts)

    proportional = small_patch_share(False)
    stratified = small_patch_share(True)

    # The meaningful, load-bearing assertion: stratification must measurably
    # favor the small patch relative to pure area-proportional sampling.
    assert stratified > proportional, (
        f"stratified share ({stratified:.2%}) should exceed area-proportional "
        f"({proportional:.2%}) for a small patch")
    # Deliberately NOT asserting an idealized ~50% round-robin split. Measured
    # behavior is a few times the area-proportional share (small patch is ~2%
    # of valid pixels; stratified gives it high single digits), because the
    # eroded small patch can only hold so many centers. Pinning an invented
    # target here would encode a guess as a requirement.
    assert stratified > proportional * 1.5, (
        f"stratified ({stratified:.2%}) barely differs from proportional "
        f"({proportional:.2%}) -- round-robin may not be engaging at all")


def test_generation_is_intentionally_non_deterministic():
    """CHARACTERIZATION: shape placement uses an UNSEEDED
    `np.random.default_rng()`, so repeated calls differ by design -- users
    expect a fresh layout each time they hit Generate.

    Consequence for this QC suite: random shapes can never be ground-truthed
    by value, only by invariant (count / containment / spacing), which is why
    every other test in this file asserts properties rather than coordinates.
    If this ever becomes seeded, this test fails and the value-based options
    become available -- a deliberate decision point, not a silent change."""
    a = generate_random_shape_pointlists(_params(count=15), H=64, W=64)
    b = generate_random_shape_pointlists(_params(count=15), H=64, W=64)
    assert _points_of(a) != _points_of(b), (
        "shape generation became deterministic -- if intentional, this suite "
        "can now assert exact coordinates")


def test_zero_count_returns_nothing():
    out = generate_random_shape_pointlists(_params(count=0), H=32, W=32)
    assert len(out) == 0
