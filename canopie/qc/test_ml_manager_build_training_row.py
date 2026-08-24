"""QC tests for `MachineLearningManager._build_training_row` and
`_select_bands_and_expressions` -- the two helpers extracted from
`train_models`' inline pixel-sampling loop so they can be (a) unit-tested in
isolation without driving the full QDialog training wizard, and (b) reused
verbatim against an AUGMENTED image variant's chans/expr_images, not just the
pristine pass -- see the augmentation wiring in train_models, which calls
BOTH of these for every generated variant.

`_build_training_row` is a faithful, unchanged extraction of what used to be
inline code (window offsets row-major, then bands then expressions per
offset, NoData-tolerance checks, edge-case counters) -- these tests pin that
exact behavior so nothing drifts during the augmentation refactor.
"""
import numpy as np
import pytest

from ..machine_learning_manager import MachineLearningManager

pytestmark = [pytest.mark.ml]


def _mgr():
    """A MachineLearningManager instance without running __init__ (no QDialog
    construction needed) -- same pattern already used elsewhere in this
    suite for testing a single method in isolation (e.g.
    ImageEditorDialog.__new__(ImageEditorDialog) in
    test_image_editor_apply_and_band_expr.py)."""
    return MachineLearningManager.__new__(MachineLearningManager)


# ---------------------------------------------------------------------------
# _build_training_row
# ---------------------------------------------------------------------------
def test_window_1_single_pixel_row_matches_band_values_directly():
    ch1 = np.arange(25, dtype=np.float32).reshape(5, 5)
    ch2 = (np.arange(25, dtype=np.float32) * 10).reshape(5, 5)
    row, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch1, ch2], [], yy=2, xx=3, window_size=1, nodata_values=[], num_features=2)
    assert ok
    assert row == [float(ch1[2, 3]), float(ch2[2, 3])]
    assert n_nan == 0 and n_missing == 0


def test_window_3_column_order_is_position_major_then_band():
    """Order: iterate window offsets row-major (dr outer, dc inner), THEN
    bands at each offset -- pins generate_window_feature_names' assumed
    column order."""
    ch = np.arange(49, dtype=np.float32).reshape(7, 7)
    row, ok, _, _ = MachineLearningManager._build_training_row(
        [ch], [], yy=3, xx=3, window_size=3, nodata_values=[], num_features=9)
    assert ok
    expected = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            expected.append(float(ch[3 + dr, 3 + dc]))
    assert row == expected


def test_expressions_come_after_bands_at_each_window_position():
    ch = np.ones((5, 5), dtype=np.float32) * 1.0
    expr = np.ones((5, 5), dtype=np.float32) * 2.0
    row, ok, _, _ = MachineLearningManager._build_training_row(
        [ch], [expr], yy=2, xx=2, window_size=1, nodata_values=[], num_features=2)
    assert ok
    assert row == [1.0, 2.0]


def test_missing_band_is_counted_and_row_rejected():
    ch = np.zeros((5, 5), dtype=np.float32)
    row, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch, None], [], yy=2, xx=2, window_size=1, nodata_values=[], num_features=2)
    assert not ok
    assert n_missing == 1
    assert n_nan == 0


def test_nan_or_inf_band_value_is_counted_and_row_rejected():
    ch = np.zeros((5, 5), dtype=np.float32)
    ch[2, 2] = np.nan
    row, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch], [], yy=2, xx=2, window_size=1, nodata_values=[], num_features=1)
    assert not ok
    assert n_nan == 1


def test_nodata_value_excludes_the_pixel():
    ch = np.zeros((5, 5), dtype=np.float32)
    ch[2, 2] = -9999.0
    row, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch], [], yy=2, xx=2, window_size=1, nodata_values=[-9999], num_features=1)
    assert not ok
    assert n_nan == 1  # NoData is folded into the same counter as NaN/Inf


def test_missing_expression_is_counted_and_row_rejected():
    ch = np.ones((5, 5), dtype=np.float32)
    row, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch], [None], yy=2, xx=2, window_size=1, nodata_values=[], num_features=2)
    assert not ok
    assert n_missing == 1


def test_only_one_failure_is_counted_per_row_even_with_multiple_bad_bands():
    ch1 = np.full((5, 5), np.nan, dtype=np.float32)
    ch2 = np.full((5, 5), np.nan, dtype=np.float32)
    _, ok, n_nan, n_missing = MachineLearningManager._build_training_row(
        [ch1, ch2], [], yy=2, xx=2, window_size=1, nodata_values=[], num_features=2)
    assert not ok
    assert n_nan == 1, "the loop must stop at the FIRST failure, not count every bad band"


# ---------------------------------------------------------------------------
# _select_bands_and_expressions
# ---------------------------------------------------------------------------
def test_select_bands_picks_requested_indices_in_order():
    mgr = _mgr()
    chans = [np.full((3, 3), i, dtype=np.float32) for i in range(4)]
    img = np.stack(chans, axis=-1)
    selected, exprs = mgr._select_bands_and_expressions(img, chans, [2, 0], [])
    assert len(selected) == 2
    assert np.all(selected[0] == 2)
    assert np.all(selected[1] == 0)
    assert exprs == []


def test_select_bands_out_of_range_index_yields_none():
    mgr = _mgr()
    chans = [np.zeros((3, 3), dtype=np.float32)]
    img = np.stack(chans, axis=-1)
    selected, _ = mgr._select_bands_and_expressions(img, chans, [0, 5], [])
    assert selected[0] is not None
    assert selected[1] is None


def test_select_bands_evaluates_expressions_against_img_not_chans():
    mgr = _mgr()
    # img has 2 bands; chans deliberately only exposes ONE of them, to prove
    # expression evaluation reads img directly rather than being limited to
    # whatever's in chans (band expressions can reference bands beyond the
    # "selected" set for classification features).
    img = np.zeros((4, 4, 2), dtype=np.float32)
    img[..., 0] = 3.0
    img[..., 1] = 5.0
    chans = [img[..., 0]]
    selected, exprs = mgr._select_bands_and_expressions(img, chans, [0], [("sum", "b1+b2")])
    assert len(exprs) == 1
    assert exprs[0] is not None
    np.testing.assert_allclose(exprs[0], 8.0)


def test_select_bands_expression_failure_yields_none_not_a_crash():
    mgr = _mgr()
    img = np.zeros((4, 4, 1), dtype=np.float32)
    chans = [img[..., 0]]
    selected, exprs = mgr._select_bands_and_expressions(img, chans, [0], [("bad", "b99*!!!")])
    assert exprs == [None]
