"""QC tests for the shared feature-normalization helpers
(`utils.fit_normalization` / `utils.apply_normalization`).

THE FEATURE: a trained model's .pkl bundle can now record how features were
normalized before `.fit()` -- "none", "l2" (per-sample, parameterless), or
"zscore" (per-feature mean/std fitted once at train time and stored so every
later predict call applies the SAME fitted transform, not a re-fit on its own,
usually much smaller, feature matrix). These two functions are the single
shared implementation used at every one of the ~6 independent classification
call sites across project_tab.py/image_editor_dialog.py/
machine_learning_manager.py (see test_ml_normalization_end_to_end.py for the
cross-site agreement checks) -- this file pins the primitives themselves.
"""
import numpy as np
import pytest

from ..utils import fit_normalization, apply_normalization

pytestmark = [pytest.mark.ml]


# ---------------------------------------------------------------------------
# fit_normalization
# ---------------------------------------------------------------------------
def test_fit_none_and_l2_carry_no_fitted_params():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    for method in ("none", "l2"):
        cfg = fit_normalization(X, method)
        assert cfg == {"method": method, "mean": None, "scale": None}


def test_fit_zscore_matches_hand_computed_mean_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[10.0, -5.0, 100.0], scale=[2.0, 0.5, 20.0], size=(500, 3)).astype(np.float32)
    cfg = fit_normalization(X, "zscore")
    assert cfg["method"] == "zscore"
    np.testing.assert_allclose(cfg["mean"], X.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(cfg["scale"], X.std(axis=0), rtol=1e-5)


def test_fit_zscore_floors_zero_variance_columns():
    X = np.stack([
        np.full(10, 7.0, dtype=np.float32),   # constant column -> std == 0
        np.arange(10, dtype=np.float32),
    ], axis=1)
    cfg = fit_normalization(X, "zscore")
    assert cfg["scale"][0] == 1.0, "a constant column's std must be floored to 1.0, not left at 0"


def test_fit_unknown_method_raises():
    X = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_normalization(X, "minmax")


def test_fit_normalization_returns_plain_python_types_not_ndarray():
    """The bundle is pickled and expected to hold plain Python types (matching
    every other bundle value's convention) -- an ndarray inside would still
    pickle fine, but would decouple the bundle format from NumPy's binary
    layout for no benefit; pin the .tolist() conversion explicitly."""
    X = np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32)
    cfg = fit_normalization(X, "zscore")
    assert isinstance(cfg["mean"], list)
    assert isinstance(cfg["scale"], list)
    assert all(isinstance(v, float) for v in cfg["mean"])


# ---------------------------------------------------------------------------
# apply_normalization
# ---------------------------------------------------------------------------
def test_apply_none_returns_unchanged_values():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = apply_normalization(X, {"method": "none"})
    np.testing.assert_array_equal(out, X)


def test_apply_none_handles_missing_or_empty_cfg():
    """Every predict call site passes bundle.get("normalization") directly --
    for an old bundle predating this feature that's None; a membership-style
    dict is also possible ({} or missing 'method'). Both must behave as 'none'."""
    X = np.arange(6, dtype=np.float32).reshape(2, 3)
    for cfg in (None, {}, {"method": None}):
        out = apply_normalization(X, cfg)
        np.testing.assert_array_equal(out, X)


def test_apply_l2_matches_sklearn_normalizer():
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import Normalizer
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 5)).astype(np.float32)
    expected = Normalizer(norm="l2").fit_transform(X)
    out = apply_normalization(X, {"method": "l2"})
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_apply_l2_floors_near_zero_rows_instead_of_producing_nan():
    X = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    out = apply_normalization(X, {"method": "l2"})
    assert np.all(np.isfinite(out)), "a zero-norm row must not produce inf/nan"
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.6, 0.8, 0.0], rtol=1e-5)


def test_apply_zscore_matches_hand_computed_transform():
    X_train = np.random.default_rng(3).normal(loc=5.0, scale=2.0, size=(100, 4)).astype(np.float32)
    cfg = fit_normalization(X_train, "zscore")

    X_new = np.random.default_rng(4).normal(loc=5.0, scale=2.0, size=(10, 4)).astype(np.float32)
    out = apply_normalization(X_new, cfg)
    expected = (X_new - np.asarray(cfg["mean"])) / np.asarray(cfg["scale"])
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_apply_zscore_on_fitted_data_lands_near_zero_mean_unit_std():
    X = np.random.default_rng(5).normal(loc=[1.0, 2.0], scale=[3.0, 4.0], size=(2000, 2)).astype(np.float32)
    cfg = fit_normalization(X, "zscore")
    out = apply_normalization(X, cfg)
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(out.std(axis=0), [1.0, 1.0], atol=1e-3)


def test_apply_zscore_raises_on_feature_count_mismatch():
    """Today's ~6 predict sites have NO guard at all for a stale bundle paired
    with a mismatched feature set -- they'd fail with an opaque NumPy
    broadcasting error somewhere downstream. Pin the clear error instead."""
    cfg = {"method": "zscore", "mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
    X_wrong_width = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="[Ff]eature count"):
        apply_normalization(X_wrong_width, cfg)


def test_apply_zscore_raises_when_mean_scale_missing():
    with pytest.raises(ValueError):
        apply_normalization(np.zeros((3, 2), dtype=np.float32), {"method": "zscore"})


def test_apply_handles_a_single_row_not_just_a_batch():
    """Several predict call sites slice a single sample out of a batch before
    predicting (e.g. process_polygon's per-point path) -- apply_normalization
    must accept a bare (F,) row, not only (N, F)."""
    X_train = np.random.default_rng(6).normal(size=(50, 3)).astype(np.float32)
    cfg = fit_normalization(X_train, "zscore")

    row = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_row = apply_normalization(row, cfg)
    out_batch = apply_normalization(row[None, :], cfg)
    assert out_row.shape == (3,)
    np.testing.assert_allclose(out_row, out_batch[0], rtol=1e-5)


def test_apply_never_mutates_the_input_array():
    """Several call sites keep using the pre-normalization X afterward (e.g.
    for label mapping keyed by row index) -- apply_normalization must return a
    new array, never edit X in place."""
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    original = X.copy()
    for cfg in ({"method": "none"}, {"method": "l2"},
                fit_normalization(X, "zscore")):
        apply_normalization(X, cfg)
        np.testing.assert_array_equal(X, original)


def test_apply_returns_float32():
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    for cfg in ({"method": "none"}, {"method": "l2"}, fit_normalization(X, "zscore")):
        out = apply_normalization(X, cfg)
        assert out.dtype == np.float32
