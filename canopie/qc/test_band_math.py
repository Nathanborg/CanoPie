"""
QC tests for band math / vegetation-index evaluation (utils.eval_band_expression).

This is the engine behind the .ax `band_expression` field and the Image
Editor's index preview, so a regression here silently changes every derived
index value in every export. Tests use small hand-verifiable arrays rather
than fixtures, since the function is pure and array-shaped.
"""
import numpy as np
import pytest

from ..utils import eval_band_expression

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.extraction]


def _img(*bands):
    """Stack 1-D-ish constant bands into a tiny HxWxC image."""
    return np.stack([np.full((2, 2), float(b), dtype=np.float32) for b in bands], axis=2)


def test_single_band_reference():
    out = eval_band_expression(_img(10, 20, 30), "b1")
    assert np.allclose(out, 10.0)


def test_band_indices_are_1_based_and_positional():
    """b1..bN map to channel 0..N-1 in array order -- NOT to R/G/B semantics.
    (The BGR<->RGB remap, where it applies, happens in the CALLER before this
    is invoked -- see project_tab._do_band_expr -- so this function must stay
    purely positional.)"""
    img = _img(10, 20, 30)
    assert np.allclose(eval_band_expression(img, "b1"), 10.0)
    assert np.allclose(eval_band_expression(img, "b2"), 20.0)
    assert np.allclose(eval_band_expression(img, "b3"), 30.0)


def test_arithmetic_and_ndvi_form():
    """The canonical NDVI shape used throughout CanoPie."""
    img = _img(100, 0, 0, 300)          # b1=Red=100, b4=NIR=300
    out = eval_band_expression(img, "(b4-b1)/(b4+b1)")
    assert np.allclose(out, (300 - 100) / (300 + 100))


def test_operator_precedence_is_pythonic():
    img = _img(2, 3, 4)
    assert np.allclose(eval_band_expression(img, "b1+b2*b3"), 2 + 3 * 4)
    assert np.allclose(eval_band_expression(img, "(b1+b2)*b3"), (2 + 3) * 4)


def test_comparison_yields_boolean_mask():
    """Threshold expressions like `b1>182` are used as NoData rules, so they
    must produce a usable 0/1 mask, not raise."""
    img = np.zeros((2, 2, 1), dtype=np.float32)
    img[0, 0, 0] = 200.0
    out = np.squeeze(np.asarray(eval_band_expression(img, "b1>100")))
    assert out.shape == (2, 2)
    assert bool(out[0, 0]) is True
    assert bool(out[1, 1]) is False


def test_single_band_result_keeps_trailing_axis():
    """DOCUMENTED INCONSISTENCY (current behavior, verified empirically):
    for a single-band image the result keeps a trailing length-1 axis
    ((H, W, 1)), while for a multi-band image it does not ((H, W)).

    Cause: the band mapping is built as
        {'b1': x} if C == 1 else {f"b{i+1}": x[:, :, i] ...}
    so in the 1-band case `b1` is the whole (H, W, 1) array rather than a
    2-D slice of it.

    Harmless downstream today -- project_tab._do_band_expr promotes 2-D
    results with `res[..., None]` anyway, and numpy broadcasting absorbs the
    difference in arithmetic -- but any new consumer that assumes a 2-D
    result will silently misbehave on single-band images only. Pinned here so
    that if the shapes are ever unified, this test fails and the decision is
    made deliberately rather than by accident."""
    one_band = np.zeros((2, 2, 1), dtype=np.float32)
    three_band = np.zeros((2, 2, 3), dtype=np.float32)
    assert np.asarray(eval_band_expression(one_band, "b1+1")).shape == (2, 2, 1)
    assert np.asarray(eval_band_expression(three_band, "b1+1")).shape == (2, 2)


@pytest.mark.parametrize("expr,expected", [
    ("clip(b1,0,50)", 50.0),
    ("abs(0-b1)", 100.0),
    ("sqrt(b1)", 10.0),
    ("where(b1>50,1,0)", 1.0),
])
def test_supported_functions(expr, expected):
    out = eval_band_expression(_img(100), expr)
    assert np.allclose(out, expected), f"{expr} -> {np.asarray(out).ravel()[0]}, expected {expected}"


def test_single_arg_reducers_are_global_scalars():
    """Documented semantics: a 1-arg reducer collapses the whole band to a
    scalar, while multi-arg reducers work pixelwise across the args. Getting
    these backwards would silently turn a per-pixel index into a constant."""
    img = np.zeros((2, 2, 1), dtype=np.float32)
    img[:, :, 0] = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert np.allclose(np.asarray(eval_band_expression(img, "mean(b1)")), 15.0)
    assert np.allclose(np.asarray(eval_band_expression(img, "max(b1)")), 30.0)


def test_multi_arg_reducers_are_pixelwise():
    img = _img(10, 30)
    out = np.asarray(eval_band_expression(img, "mean(b1,b2)"))
    assert out.shape == (2, 2), "multi-arg reducer must stay pixelwise, not collapse"
    assert np.allclose(out, 20.0)


def test_nonfinite_inputs_are_neutralized():
    """eval_band_expression nan_to_num's its input up front; a NaN pixel must
    not poison the whole result array."""
    img = np.zeros((2, 2, 1), dtype=np.float32)
    img[0, 0, 0] = np.nan
    img[0, 1, 0] = np.inf
    img[1, 0, 0] = 5.0
    out = np.asarray(eval_band_expression(img, "b1+1"))
    assert np.all(np.isfinite(out)), f"non-finite leaked through: {out}"
    assert np.isclose(out[1, 0], 6.0)


def test_unknown_name_is_rejected():
    """Expressions are user-supplied; referencing something that isn't a band
    must raise rather than silently evaluate to garbage."""
    with pytest.raises(Exception):
        eval_band_expression(_img(1, 2, 3), "b1 + nonsense_token")


def test_empty_expression_rejected():
    with pytest.raises(ValueError):
        eval_band_expression(_img(1), "")
