"""QC regression test for a perf regression in the classification RENDER
path (`_render_with_viewer_stretch` -> `_apply_sklearn_classification[_with_
indices]`, used on every root navigation for an image with an active
sklearn classification `.ax`), introduced alongside the "b1=Red / viewer-vs-
CSV classification eligibility" fix.

THE BUG: `_apply_sklearn_classification_with_indices`/
`_apply_sklearn_classification` UNCONDITIONALLY filtered the feature tile
through a boolean mask before predicting --

    valid = np.all(np.isfinite(X_tile), axis=1)
    X_valid = np.ascontiguousarray(X_tile[valid], dtype=np.float32)
    ...
    where = np.flatnonzero(valid) + out_offset
    out[where] = preds

-- even when EVERY row in the tile was eligible, which is the common case:
a plain, valid image with no NoData/window-edge pixels. `X_tile[valid]` is a
boolean fancy-index, which ALWAYS makes a fresh copy of the whole tile
(never a view, even when the mask is all-True), and `out[where] = preds` is
an indexed scatter instead of a contiguous slice assignment. Both are
full-tile-sized allocations/scans paid on EVERY classification render for
no behavioral benefit whenever nothing actually needs excluding. Measured
on a real RandomForestClassifier: ~15-30% slower than before the
eligibility fix at window_size=1 (the default/common case) on a fully-valid
1200x1600 synthetic image with no NoData values declared.

THE FIX: `_predict_and_write_tile` (shared by both call sites) takes a fast
path when `valid.sum() == len(X_tile)`: it hands `model.predict` a
contiguous VIEW of `X_tile` directly (no boolean-index copy) and writes
results back with a plain contiguous slice (no flatnonzero+scatter). The
general (some-rows-ineligible) path is unchanged. Output values are
identical either way -- `flatnonzero(all-True) + out_offset` is exactly the
contiguous range `[out_offset, out_offset + n_tile)` the fast path uses
instead.
"""
import numpy as np
import pytest

from ..project_tab import ProjectTab, _predict_and_write_tile
from .. import utils

pytestmark = [pytest.mark.ml, pytest.mark.perf]


# ---------------------------------------------------------------------------
# The core regression: no boolean-index copy, no scatter, when all rows are
# eligible.
# ---------------------------------------------------------------------------
class _BufferSpyModel:
    """Records, for every predict() call, whether the array handed in
    SHARES MEMORY with the original tile buffer (fast path: a view) or is
    an independent copy (the old always-filter behavior). `np.shares_memory`
    -- not `is`/`.base` identity -- is the only check that's correct
    regardless of how deep the view chain is (e.g. `X_tile` itself being a
    `.reshape()` of another array already makes `.base` point past `X_tile`
    to THAT array, not to `X_tile`)."""

    def __init__(self, tile):
        self.tile = tile
        self.calls = []  # list of bool: True if X shares memory with X_tile

    def predict(self, X):
        self.calls.append(bool(np.shares_memory(X, self.tile)))
        return np.zeros(X.shape[0], dtype=np.int64)


def test_predict_and_write_tile_avoids_copy_and_scatter_when_all_rows_valid():
    H, W, F = 200, 150, 3
    n = H * W
    X_tile = np.arange(n * F, dtype=np.float32).reshape(n, F)
    assert X_tile.flags["C_CONTIGUOUS"]

    model = _BufferSpyModel(X_tile)
    out = np.empty((n,), dtype=object)
    classified = np.zeros((n,), dtype=bool)

    _predict_and_write_tile(model, X_tile, chunk=10_000, out=out,
                             classified=classified, out_offset=0)

    assert classified.all(), "every row was eligible -- all must be marked classified"
    assert model.calls, "predict() was never called"
    assert all(model.calls), (
        "predict() was called on a COPY of X_tile even though every row was "
        "eligible -- the all-valid fast path regressed back to the "
        "boolean-fancy-index copy this test exists to catch")


def test_predict_and_write_tile_still_filters_when_some_rows_invalid():
    """The general path (some rows ineligible) must still exclude exactly
    those rows -- the fast path must never apply when it would be wrong."""
    H, W, F = 20, 15, 3
    n = H * W
    X_tile = np.arange(n * F, dtype=np.float32).reshape(n, F)
    bad_rows = np.array([0, 5, 17, n - 1])
    X_tile[bad_rows, 0] = np.nan

    class _RecordingModel:
        def predict(self, X):
            calls.append(X.shape[0])
            return np.zeros(X.shape[0], dtype=np.int64)

    calls = []
    out = np.empty((n,), dtype=object)
    classified = np.zeros((n,), dtype=bool)
    _predict_and_write_tile(_RecordingModel(), X_tile, chunk=10_000, out=out,
                             classified=classified, out_offset=0)

    expected_valid = n - len(bad_rows)
    assert sum(calls) == expected_valid, "predict() must only ever see eligible rows"
    assert not classified[bad_rows].any(), "NaN-marked (ineligible) rows must not be marked classified"
    good_rows = np.setdiff1d(np.arange(n), bad_rows)
    assert classified[good_rows].all(), "every eligible row must be marked classified"


def test_predict_and_write_tile_writes_are_offset_correctly():
    """Sanity: out_offset must still land in the right place for a
    non-zero-offset tile (the row-band tiling case)."""
    n_tile = 30
    F = 3
    X_tile = np.ones((n_tile, F), dtype=np.float32)

    class _ConstModel:
        def predict(self, X):
            return np.full(X.shape[0], 7, dtype=np.int64)

    total_n = 100
    out = np.empty((total_n,), dtype=object)
    classified = np.zeros((total_n,), dtype=bool)
    _predict_and_write_tile(_ConstModel(), X_tile, chunk=1000, out=out,
                             classified=classified, out_offset=50)

    assert classified[50:80].all()
    assert not classified[:50].any() and not classified[80:].any()
    assert all(v == 7 for v in out[50:80])


# ---------------------------------------------------------------------------
# End-to-end: the real ProjectTab render-path methods must agree with a
# plain, unfiltered model.predict() call when nothing is ineligible, and
# must still exclude declared-NoData / window-edge pixels when they exist.
# (Mirrors the eligibility-rule tests in test_classification_pixel_
# eligibility.py, scoped to this specific fast-path change.)
# ---------------------------------------------------------------------------
class _ThresholdModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (X[:, 0] > 128).astype(np.int64)


def _stub_tab(bundle):
    class _Tab:
        pass

    t = _Tab()
    t.random_forest_model = bundle
    t.shared_random_forest_model = None
    t._get_sklearn_bundle = ProjectTab._get_sklearn_bundle.__get__(t, ProjectTab)
    t._make_feature_stack_for_model = ProjectTab._make_feature_stack_for_model.__get__(t, ProjectTab)
    t._colorize_class_map = ProjectTab._colorize_class_map.__get__(t, ProjectTab)
    t.UNCLASSIFIED_INDEX = ProjectTab.UNCLASSIFIED_INDEX
    t._apply_sklearn_classification_with_indices = (
        ProjectTab._apply_sklearn_classification_with_indices.__get__(t, ProjectTab))
    return t


def test_apply_sklearn_classification_with_indices_matches_plain_predict_when_fully_valid():
    H, W = 40, 30
    rng = np.random.default_rng(1)
    img = (rng.random((H, W, 3)) * 255).astype(np.uint8)  # cv2-style BGR

    base_names = ["red_channel", "green_channel", "blue_channel"]
    bundle = {
        "model": _ThresholdModel(),
        "feature_names": base_names,
        "label_names": [0, 1],
        "expressions": [],
        "window_size": 1,
        "base_feature_names": base_names,
    }
    tab = _stub_tab(bundle)
    _, idx = tab._apply_sklearn_classification_with_indices(
        img, nodata_values=[], src_channel_order="bgr")

    expected = (img[:, :, 2].astype(np.int64) > 128).astype(np.int32)  # red is BGR-index 2
    assert np.array_equal(idx, expected)
    assert (idx >= 0).all(), "no pixel should be unclassified when nothing is ineligible"


def test_apply_sklearn_classification_with_indices_still_excludes_declared_nodata():
    H, W = 40, 30
    rng = np.random.default_rng(1)
    img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    img[5:10, 5:10, 2] = 7  # force a block's BGR-red channel to the NoData literal

    base_names = ["red_channel", "green_channel", "blue_channel"]
    bundle = {
        "model": _ThresholdModel(),
        "feature_names": base_names,
        "label_names": [0, 1],
        "expressions": [],
        "window_size": 1,
        "base_feature_names": base_names,
    }
    tab = _stub_tab(bundle)
    _, idx = tab._apply_sklearn_classification_with_indices(
        img, nodata_values=[7], src_channel_order="bgr")

    # Ground truth from the SAME shared eligibility rule the fix relies on.
    img_rgb = utils.to_rgb_band_order(img.astype(np.float32), "bgr")
    X_gt = np.column_stack([img_rgb[:, :, 0].reshape(-1),
                            img_rgb[:, :, 1].reshape(-1),
                            img_rgb[:, :, 2].reshape(-1)])
    bad = utils.classification_invalid_rows(X_gt, H, W, window_size=1, nodata_values=[7]).reshape(H, W)

    assert bad[5:10, 5:10].all(), "test setup: forced block must be part of 'bad'"
    assert (idx[bad] == tab.UNCLASSIFIED_INDEX).all()
    expected = (img[:, :, 2].astype(np.int64) > 128).astype(np.int32)
    assert np.array_equal(idx[~bad], expected[~bad])
