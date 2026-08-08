"""
CONTRACT: the two band-math engines must be interchangeable.

`process_polygon` prefers `performance.FastBandMathEngine` and only falls back
to `utils.eval_band_expression` when the fast engine *raises*
(project_tab.py, `_get_bm_arr`). Which engine runs therefore depends on whether
`numexpr` is installed and on whether the expression happens to parse -- neither
of which the user controls or sees. If the two disagree, the SAME project with
the SAME formula exports DIFFERENT numbers on different machines. For a
scientific tool that is a reproducibility failure, not a performance detail.

THREE DIVERGENCES THIS PINS (all measured, all reached real exports):

  b1/b2 with b2 == 0     fast -> inf     reference -> 0.0
      `_transform_expr_for_eval` never rewrote `/` to the `safe_div` it
      defines, despite its comment saying it did. `GCC = b2/(b1+b2+b3)` on a
      zero-sum pixel (shadow, NoData edge) exported inf, and that inf flowed
      straight into Mean/Median.

  b1+b2 with b1 == NaN   fast -> nan     reference -> 2.0
      the reference sanitises its INPUT (NaN->0) and its OUTPUT; the fast
      engine did neither.

  (b1 >150) & (b2>165)   fast -> raises  reference -> works
      the fast rewriter was regex-based and `(\\S+)\\s*&\\s*(\\S+)` cannot see
      parentheses, so it produced invalid Python. This is the shipped
      "boolean2" default in the indices box, so the advertised example silently
      took the slow path on every single export.

Fixed by making the numpy branch DELEGATE to the reference implementation
rather than being a second parser kept in sync by hand, and by sanitising the
numexpr branch's input and output the same way.
"""
import numpy as np
import pytest

from ..performance import FastBandMathEngine, numexpr_available
from ..utils import eval_band_expression

pytestmark = [pytest.mark.contract, pytest.mark.extraction]


# --- inputs chosen to break things, not to be representative -----------------
def _random_stack():
    rng = np.random.default_rng(0)
    return (rng.random((16, 16, 8), dtype=np.float32) * 200).astype(np.float32)


def _zero_divisor_stack():
    """b2 is entirely zero, so anything dividing by it hits the safe_div path."""
    a = np.zeros((6, 6, 3), dtype=np.float32)
    a[..., 0] = 10.0
    a[..., 2] = 5.0
    return a


def _nan_stack():
    a = np.arange(6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
    a[0, 0, 0] = np.nan
    a[1, 1, 1] = np.inf
    a[2, 2, 2] = -np.inf
    return a


def _negative_stack():
    a = np.linspace(-50.0, 50.0, 6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
    return a


def _constant_stack():
    """Zero variance in every band -- reducers and normalised indices degenerate."""
    a = np.empty((5, 5, 3), dtype=np.float32)
    a[..., 0] = 7.0
    a[..., 1] = 0.0
    a[..., 2] = -3.0
    return a


def _single_band():
    return np.linspace(0.0, 10.0, 25, dtype=np.float32).reshape(5, 5, 1)


STACKS = {
    "random": _random_stack,
    "zero_divisor": _zero_divisor_stack,
    "nan_inf": _nan_stack,
    "negative": _negative_stack,
    "constant": _constant_stack,
}

# Arithmetic, the shipped index defaults, and every boolean form the indices box
# advertises.
EXPRESSIONS = [
    "b1 + b2 + b3",
    "b2 / (b1 + b2 + b3)",          # GCC -- the zero-sum case
    "b3 / (b1 + b2 + b3)",          # RCC
    "b1 / (b1 + b2 + b3)",          # BCC
    "2*b2 - (b1 + b3)",             # EXG
    "(2*b1) + b3 - (2*b2)",         # WDX_2
    "b1 + 2*b3 - 2*b2",             # WDX_3
    "b1/b2",                        # bare divide by an all-zero band
    "b1 > 150",
    "(b1 >150) & (b2>165)",         # boolean2 -- used to RAISE in the fast engine
    "(b2 / (b1 + b2 + b3))>0.41",   # boolean3
    "b1==7",                        # boolean4
    "(b1==7) | (b2==2)",            # boolean5
    "b1 != 0",
    "~(b1 > 10)",
    "clip(b1, 0, 50)",
    "where(b1 > 1, b2, b3)",
    "abs(b1 - b2)",
    "sqrt(b1)",
    "min(b1, b2)",
    "max(b1, b2, b3)",
]


def _both(img, expr):
    fast = np.asarray(FastBandMathEngine().eval_expression(img, expr), dtype=np.float64)
    ref = np.asarray(eval_band_expression(img, expr), dtype=np.float64)
    return fast, ref


@pytest.mark.parametrize("stack_name", sorted(STACKS))
@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_engines_agree(stack_name, expr):
    """THE contract. Every expression x every hostile input, both engines."""
    img = STACKS[stack_name]()
    fast, ref = _both(img, expr)

    assert fast.shape == ref.shape, (
        f"{stack_name} / {expr!r}: shape {fast.shape} vs {ref.shape}")
    assert np.allclose(fast, ref, rtol=1e-5, atol=1e-6, equal_nan=True), (
        f"{stack_name} / {expr!r}: engines disagree -- "
        f"fast[0,0]={fast.flat[0]!r} ref[0,0]={ref.flat[0]!r}, "
        f"max|diff|={np.nanmax(np.abs(fast - ref))}")


def test_single_band_image_agrees():
    """C == 1 takes a different band-mapping branch in both engines."""
    img = _single_band()
    for expr in ("b1 * 2", "b1 > 5", "clip(b1, 1, 9)"):
        fast, ref = _both(img, expr)
        assert np.allclose(fast, ref, equal_nan=True), f"{expr!r} disagrees"


# ---------------------------------------------------------------------------
# The specific defects, asserted on absolute values rather than only on
# agreement -- so "both engines are wrong in the same new way" still fails.
# ---------------------------------------------------------------------------
def test_divide_by_zero_is_zero_not_inf():
    """safe_div's contract: x/0 -> 0. inf here poisons Mean/Median downstream."""
    img = _zero_divisor_stack()
    for engine_name, out in (("fast", FastBandMathEngine().eval_expression(img, "b1/b2")),
                             ("reference", eval_band_expression(img, "b1/b2"))):
        arr = np.asarray(out, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"{engine_name}: non-finite leaked out"
        assert np.allclose(arr, 0.0), f"{engine_name}: expected 0 for x/0, got {arr.flat[0]}"


def test_zero_sum_index_is_finite():
    """GCC on an all-zero pixel is the real-world form of the above."""
    img = np.zeros((4, 4, 3), dtype=np.float32)
    for engine_name, out in (
            ("fast", FastBandMathEngine().eval_expression(img, "b2 / (b1 + b2 + b3)")),
            ("reference", eval_band_expression(img, "b2 / (b1 + b2 + b3)"))):
        arr = np.asarray(out, dtype=np.float64)
        assert np.all(np.isfinite(arr)), (
            f"{engine_name}: GCC over an all-zero pixel produced {arr.flat[0]!r}")


def test_output_is_always_finite_for_every_expression():
    """No expression may hand a non-finite value to the statistics layer."""
    for stack_name, factory in STACKS.items():
        img = factory()
        for expr in EXPRESSIONS:
            for engine_name, out in (
                    ("fast", FastBandMathEngine().eval_expression(img, expr)),
                    ("reference", eval_band_expression(img, expr))):
                arr = np.asarray(out, dtype=np.float64)
                assert np.all(np.isfinite(arr)), (
                    f"{engine_name} / {stack_name} / {expr!r} produced non-finite "
                    f"values (e.g. {arr[~np.isfinite(arr)].flat[0]!r})")


def test_parenthesised_boolean_does_not_raise_in_the_fast_engine():
    """`(b1 >150) & (b2>165)` is the shipped boolean2 default. It used to raise
    here and fall back silently, so the advertised example never ran on the
    fast path."""
    img = _random_stack()
    out = FastBandMathEngine().eval_expression(img, "(b1 >150) & (b2>165)")
    assert np.asarray(out).shape == img.shape[:2]


def test_nan_input_is_treated_identically_by_both():
    """The reference maps NaN->0 on input; the fast engine must too, or a
    NoData pixel contributes a different value depending on the engine."""
    img = np.array([[[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]]], dtype=np.float32)
    fast, ref = _both(img, "b1 + b2")
    assert np.allclose(fast, ref, equal_nan=True), f"fast={fast.ravel()} ref={ref.ravel()}"


# ---------------------------------------------------------------------------
# Caching must not leak between images
# ---------------------------------------------------------------------------
def test_expression_cache_does_not_leak_across_images():
    """eval_expression caches by expression string. A second, DIFFERENT image
    evaluated with the same expression must not receive the first one's
    result -- that would silently export image A's numbers for image B."""
    eng = FastBandMathEngine()
    a = np.full((4, 4, 3), 1.0, dtype=np.float32)
    b = np.full((4, 4, 3), 9.0, dtype=np.float32)

    ra = np.asarray(eng.eval_expression(a, "b1 + b2", cache_key="b1 + b2"))
    rb = np.asarray(eng.eval_expression(b, "b1 + b2", cache_key="b1 + b2"))

    assert np.allclose(ra, 2.0)
    assert np.allclose(rb, 18.0), (
        f"cache leaked image A's result into image B: got {rb.flat[0]}, expected 18.0. "
        "cache_key must incorporate the image identity, or callers must scope "
        "the engine per image.")


def test_band_math_is_correct_for_every_image_in_a_multi_image_export(synthetic_project):
    """THE production form of the cache-leak bug, driven through process_polygon.

    `get_band_math_engine()` is a process-wide singleton and `process_polygon`
    caches by expression string alone, so a CSV export over several images
    reused the FIRST image's band-math array for all the rest -- with each
    image's own polygon mask applied to it, which is why the numbers looked
    plausible instead of obviously broken. Measured before the fix:

        rgb_8bit_untiled            382.0  (correct -- first image)
        rgb_16bit_tiled_bip_cog     383.8  (should be 86022.0)
        multiband_8band_ancillary   366.0  (should be 1527.0)

    The unit-level cache test above does NOT catch this on its own: it exercises
    the engine directly, whereas the damage happens through the singleton that
    only production wires up. Both are needed.
    """
    from .fixtures_manifest import fixture_image_path, get_fixture
    from .generate_fixtures import _rasterize_polygon_mask
    from .project_builder import polygon_group_name
    from ._helpers import load_raw_npz

    opts = {"stats": {"mean": True},
            "band_math": {"enabled": True, "formulas": {"sum3": "b1+b2+b3"}}}

    # Order matters: the first image is the one whose array used to leak.
    names = ["rgb_8bit_untiled", "rgb_16bit_tiled_bip_cog",
             "multiband_8band_ancillary", "rgb_12in16_tiled_bip"]

    for name in names:
        spec = get_fixture(name)
        fp = fixture_image_path(name)
        group = polygon_group_name(name, spec["polygon"]["name"])
        poly = synthetic_project.all_polygons[group][fp]

        rows, _ = synthetic_project.process_polygon(
            group, fp, poly, {}, [], False, opts=opts)
        row = next((r for r in rows if r.get("Channel") == "sum3"), None)
        assert row is not None, f"{name}: no sum3 row"

        raw = load_raw_npz(name).astype(np.float64)
        mask = _rasterize_polygon_mask(
            spec["polygon"]["points"], spec["height"], spec["width"])
        expected = (raw[..., 0] + raw[..., 1] + raw[..., 2])[mask].mean()

        assert row["Mean"] == pytest.approx(expected, abs=1e-3), (
            f"{name}: exported sum3 Mean {row['Mean']} but the pixels say "
            f"{expected}. A previous image's band-math array is being reused.")


@pytest.mark.skipif(not numexpr_available(),
                    reason="numexpr not installed -- the numexpr branch cannot be exercised here")
def test_numexpr_branch_agrees_too():
    """When numexpr IS installed it takes a different code path with different
    divide/NaN semantics, so it needs its own equivalence check. Skipped when
    numexpr is absent, which means CI must run at least once WITH it."""
    for stack_name, factory in STACKS.items():
        img = factory()
        for expr in EXPRESSIONS:
            fast, ref = _both(img, expr)
            assert np.allclose(fast, ref, rtol=1e-5, atol=1e-6, equal_nan=True), (
                f"numexpr path disagrees for {stack_name} / {expr!r}")
