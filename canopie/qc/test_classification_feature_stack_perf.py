"""QC regression tests for "classify using the pkl files is too slow" / "a
very big file will take forever and crash the program."

THE BUG: `_make_feature_stack_for_model` exists as two near-duplicate copies
-- `ImageEditorDialog`'s (the live "Classify" button, full-resolution open
image) and `ProjectTab`'s (render/preview + the `.ax` "appended
classification band" replay in `_apply_ax_to_raw`/`apply_aux_modifications`,
also reachable at full resolution). In both, the `window_size > 1` branch
(spatial-context models) was a pure-Python `for y: for x: for dr: for dc:
for base_name: ... arr[py,px] ...` loop touching every pixel individually --
O(H*W*window_size^2*n_features) Python-level operations, minutes on a modest
image. Both branches ALSO unconditionally allocated a whole-image feature
array up front (`H*W*F*4` bytes, unbounded) -- a 10000x10000/3-band/window=3
image is ~10.8GB for that array alone, a straightforward OOM crash.

Why `process_polygon` (CSV export) never has this problem: it's always
bounded to one polygon's ROI via `LazyChannels.read_window`, which has its
own memory budget (`max_materialize`/`_DEFAULT_CACHE_BYTES` in
raster_reader.py) and falls back to decoding one band at a time past it.
There's no small ROI to exploit for a live "classify the whole image"
request, so the fix instead (1) vectorizes the window-feature construction
(`utils.build_windowed_feature_cube`) and (2) tiles by a row-band memory
budget when the feature matrix would exceed it
(`utils.iter_tiled_feature_matrices`, `_CLASSIFY_FEATURE_BUDGET_BYTES`),
mirroring that same "bound peak memory, decode/build in pieces past a
threshold" principle.
"""
import ast
import gc
import time
import tracemalloc

import numpy as np
import pytest

from .test_export_and_ax_regressions import _func_tree, _names_in
from .. import utils
from ..image_editor_dialog import ImageEditorDialog
from ..project_tab import ProjectTab

pytestmark = [pytest.mark.editor, pytest.mark.ml, pytest.mark.perf]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _old_loop_reference(padded, base_names, window_size, H, W):
    """Independent re-implementation of the ORIGINAL per-pixel loop's
    documented semantics (edge-pad, (dr,dc)-outer/base_name-inner column
    order) -- not a call into production code, which no longer contains
    this shape. Deliberately slow and obviously correct; only used on small
    images in these tests."""
    half = window_size // 2
    F = window_size * window_size * len(base_names)
    X_flat = np.zeros((H * W, F), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            flat_idx = y * W + x
            feat_idx = 0
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    py = y + half + dr
                    px = x + half + dc
                    for name in base_names:
                        X_flat[flat_idx, feat_idx] = padded[name][py, px]
                        feat_idx += 1
    return X_flat


def _rgb_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# utils.py primitives: correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window_size", [3, 5])
def test_build_windowed_feature_cube_matches_the_old_loop(window_size):
    H, W = 14, 11
    half = window_size // 2
    base_names = ["r", "g", "b"]
    rng = np.random.default_rng(42)
    arrs = {n: rng.random((H, W)).astype(np.float32) for n in base_names}
    padded = {n: np.pad(a, half, mode="edge") for n, a in arrs.items()}

    ref = _old_loop_reference(padded, base_names, window_size, H, W)
    cube = utils.build_windowed_feature_cube(padded, base_names, window_size, H, W)
    got = cube.reshape(H * W, -1)

    assert got.shape == ref.shape
    assert np.array_equal(got, ref), "vectorized builder must match the old loop exactly"


def test_iter_classification_row_tiles_single_tile_when_it_fits():
    tiles = list(utils.iter_classification_row_tiles(20, 15, 9, budget_bytes=10**9, half=1))
    assert tiles == [(0, 20, 0, 20)]


def test_iter_classification_row_tiles_covers_every_row_exactly_once():
    H = 37
    tiles = list(utils.iter_classification_row_tiles(H, 9, 9, budget_bytes=300, half=2))
    assert len(tiles) > 1, "budget was too small to force tiling -- test setup is wrong"
    covered = []
    for row0, row1, halo0, halo1 in tiles:
        assert halo0 <= row0 and row1 <= halo1
        covered.extend(range(row0, row1))
    assert covered == list(range(H))


def test_iter_tiled_feature_matrices_matches_untiled_on_a_forced_small_budget():
    H, W = 16, 9
    base_names = ["r", "g", "b"]
    img = _rgb_image(H, W, seed=7)
    feat_names = list(range(9 * 3))

    def builder(sub_img, fnames, exprs, wsize, bnames):
        a = np.asarray(sub_img)
        h, w = a.shape[0], a.shape[1]
        half = wsize // 2
        padded = {"r": np.pad(a[:, :, 0].astype(np.float32), half, mode="edge"),
                  "g": np.pad(a[:, :, 1].astype(np.float32), half, mode="edge"),
                  "b": np.pad(a[:, :, 2].astype(np.float32), half, mode="edge")}
        cube = utils.build_windowed_feature_cube(padded, bnames, wsize, h, w)
        return cube.reshape(h * w, -1), (h, w)

    ref_X, _ = builder(img, feat_names, [], 3, base_names)

    tiles = list(utils.iter_tiled_feature_matrices(
        img, feat_names, [], 3, base_names, budget_bytes=400, builder_fn=builder))
    assert len(tiles) > 1, "budget was too small to force tiling -- test setup is wrong"

    reassembled = np.zeros((H, W, ref_X.shape[1]), dtype=np.float32)
    for row0, row1, X_tile, w in tiles:
        assert X_tile is not None
        reassembled[row0:row1] = X_tile.reshape(row1 - row0, w, -1)
    assert np.array_equal(reassembled.reshape(H * W, -1), ref_X)


def test_iter_tiled_feature_matrices_single_tile_when_budget_is_huge():
    H, W = 10, 8
    img = _rgb_image(H, W, seed=3)
    feat_names = list(range(3))
    base_names = ["r", "g", "b"]

    def builder(sub_img, fnames, exprs, wsize, bnames):
        a = np.asarray(sub_img)
        return a.reshape(-1, 3).astype(np.float32), (a.shape[0], a.shape[1])

    tiles = list(utils.iter_tiled_feature_matrices(
        img, feat_names, [], 1, base_names, budget_bytes=10**9, builder_fn=builder))
    assert len(tiles) == 1
    assert tiles[0][0] == 0 and tiles[0][1] == H


def test_iter_tiled_feature_matrices_propagates_builder_failure():
    def failing_builder(sub_img, *a, **k):
        return None, (0, 0)

    img = _rgb_image(30, 30)
    tiles = list(utils.iter_tiled_feature_matrices(
        img, [1] * 9, [], 3, ["r"], budget_bytes=10, builder_fn=failing_builder))
    assert all(t[2] is None for t in tiles)


# ---------------------------------------------------------------------------
# Both real _make_feature_stack_for_model copies: correctness + no divergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_both_copies_agree_with_each_other(window_size):
    img = _rgb_image(9, 13, seed=11)
    base_names = ["red_channel", "green_channel", "blue_channel"]
    n_base = len(base_names)
    feat_names = list(range((window_size * window_size if window_size > 1 else 1) * n_base))

    X_editor, hw_editor = ImageEditorDialog._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=base_names)
    X_pt, hw_pt = ProjectTab._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=base_names)

    assert hw_editor == hw_pt == (9, 13)
    assert X_editor is not None and X_pt is not None
    assert np.array_equal(X_editor, X_pt), (
        "the two copies must produce identical features for a plain "
        "3-channel image with no NaN/Inf -- the only documented "
        "difference between them (project_tab's NaN sanitization pass) "
        "must not change output when there's nothing to sanitize")


@pytest.mark.parametrize("window_size", [3, 5])
def test_editor_copy_matches_independent_reference(window_size):
    H, W = 8, 6
    img = _rgb_image(H, W, seed=21)
    base_names = ["red_channel", "green_channel", "blue_channel"]
    feat_names = list(range(window_size * window_size * 3))

    X, (h, w) = ImageEditorDialog._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=base_names)

    half = window_size // 2
    a = np.asarray(img)
    chans = [a[:, :, 2].astype(np.float32), a[:, :, 1].astype(np.float32), a[:, :, 0].astype(np.float32)]
    padded = {"red_channel": np.pad(chans[0], half, mode="edge"),
              "green_channel": np.pad(chans[1], half, mode="edge"),
              "blue_channel": np.pad(chans[2], half, mode="edge")}
    ref = _old_loop_reference(padded, base_names, window_size, H, W)
    assert np.array_equal(X, ref)


# ---------------------------------------------------------------------------
# AST-level: the old per-pixel loop is really gone, the vectorized builder
# is really wired in
# ---------------------------------------------------------------------------
def _contains_quadruple_nested_pixel_loop(func):
    """Detect the OLD shape: a For loop over range(H)/range(W)-like names
    containing 2 more nested For loops inside it. A generic structural
    check (nesting depth), not a name/string match, so it can't be fooled
    by a comment or a differently-named variable."""
    tree = _func_tree(func)
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        depth1 = [n for n in ast.walk(node) if isinstance(n, ast.For) and n is not node]
        # count strictly-nested For loops reachable from this one
        nested_fors = sum(1 for n in ast.walk(node) if isinstance(n, ast.For)) - 1
        if nested_fors >= 3:
            return True
    return False


def test_editor_copy_has_no_quadruple_nested_pixel_loop():
    assert not _contains_quadruple_nested_pixel_loop(ImageEditorDialog._make_feature_stack_for_model), (
        "found a 4-deep nested For loop -- looks like the old per-pixel "
        "window-feature construction is back")


def test_project_tab_copy_has_no_quadruple_nested_pixel_loop():
    assert not _contains_quadruple_nested_pixel_loop(ProjectTab._make_feature_stack_for_model)


def test_editor_copy_calls_the_vectorized_builder():
    names = _names_in(ImageEditorDialog._make_feature_stack_for_model)
    assert "build_windowed_feature_cube" in names


def test_project_tab_copy_calls_the_vectorized_builder():
    names = _names_in(ProjectTab._make_feature_stack_for_model)
    assert "build_windowed_feature_cube" in names


def test_run_sklearn_classification_uses_tiled_iteration():
    names = _names_in(ImageEditorDialog.run_sklearn_classification)
    assert "iter_tiled_feature_matrices" in names
    assert "_CLASSIFY_FEATURE_BUDGET_BYTES" in names


@pytest.mark.parametrize("func", [
    ProjectTab._apply_sklearn_classification_with_indices,
    ProjectTab._apply_sklearn_classification,
    ProjectTab._apply_ax_to_raw,
    ProjectTab.apply_aux_modifications,
])
def test_project_tab_classification_call_sites_use_tiled_iteration(func):
    names = _names_in(func)
    assert "iter_tiled_feature_matrices" in names, (
        f"{func.__qualname__} must build/predict via the tiled, memory-"
        "bounded path, not a single whole-image builder call")
    assert "_CLASSIFY_FEATURE_BUDGET_BYTES" in names


# ---------------------------------------------------------------------------
# Memory: peak allocation must scale with the BUDGET/tile, not H*W*F
# ---------------------------------------------------------------------------
def test_tiled_path_peak_memory_is_bounded_not_whole_image(monkeypatch):
    """THE crash regression. Force a budget far below the full feature
    matrix's size, and assert peak traced memory stays within a small
    multiple of that budget -- never anywhere near H*W*F*4.

    Deliberately does NOT accumulate a full-size result array inside the
    traced region -- doing that would itself allocate H*W*F*4 bytes and
    defeat the whole point of the measurement. Each tile's array is
    touched (summed) and then allowed to go out of scope; correctness is
    checked SEPARATELY afterward, outside tracemalloc, by re-iterating and
    reassembling.
    """
    H, W = 2000, 100
    window_size = 3
    base_names = ["red_channel", "green_channel", "blue_channel"]
    feat_names = list(range(window_size * window_size * 3))
    img = _rgb_image(H, W, seed=99)

    full_bytes = H * W * len(feat_names) * 4
    budget = 400_000  # deliberately far below full_bytes
    assert budget < full_bytes / 20, "test setup: budget must be much smaller than the full matrix"

    def builder(i, fn, ex, ws, bn):
        return ImageEditorDialog._make_feature_stack_for_model(None, i, fn, ex, ws, bn)

    n_tiles = 0
    checksum = 0.0
    gc.collect()
    tracemalloc.start()
    try:
        for row0, row1, X_tile, w in utils.iter_tiled_feature_matrices(
                img, feat_names, [], window_size, base_names, budget, builder):
            n_tiles += 1
            checksum += float(X_tile.sum())  # touch it, then let it be discarded
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert n_tiles > 1, "budget was too small to force tiling -- test setup is wrong"
    # Generous ceiling (per test_perf_budgets.py's convention: distinguish
    # "tiled" from "whole image", not a tight byte-exact bound) -- a few
    # tiles' worth of working arrays, nowhere near full_bytes.
    assert peak < full_bytes / 3, (
        f"peak allocation {peak/1e6:.2f} MB is not far below the full "
        f"{full_bytes/1e6:.2f} MB feature matrix -- tiling is not actually "
        "bounding memory")
    assert checksum != 0.0  # sanity: the loop really ran and touched data

    # Correctness, checked separately (outside the memory measurement): the
    # generator is single-use, so re-iterate it fresh.
    reassembled = np.zeros((H * W, len(feat_names)), dtype=np.float32)
    for row0, row1, X_tile, w in utils.iter_tiled_feature_matrices(
            img, feat_names, [], window_size, base_names, budget, builder):
        reassembled[row0 * w:row1 * w] = X_tile
    X_full, _ = ImageEditorDialog._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=base_names)
    assert np.array_equal(reassembled, X_full)


# ---------------------------------------------------------------------------
# End to end: run_sklearn_classification, tiled, via a real (minimal) host
# ---------------------------------------------------------------------------
class _FakeModel:
    """classes_ = [0, 1]; predicts 1 wherever the RED feature (column 0,
    since base_names=[red,green,blue]) exceeds 128, else 0 -- deterministic
    and easy to check against a known image."""
    classes_ = np.array([0, 1])

    def get_params(self, **k):
        return {}  # no n_jobs -> forces the ThreadPoolExecutor path

    def predict(self, X):
        return (X[:, 0] > 128).astype(np.int64)


def test_run_sklearn_classification_tiled_matches_untiled(qapp, monkeypatch):
    from PyQt5 import QtWidgets

    H, W = 40, 20
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:H // 2, :, 2] = 200   # top half: cv2 B-channel (index 2) high -> RGB red high after BGR->RGB flip
    img[H // 2:, :, 2] = 10    # bottom half: red low

    window_size = 3
    base_names = ["red_channel", "green_channel", "blue_channel"]
    feat_names = list(range(window_size * window_size * 3))

    host = QtWidgets.QWidget()
    host.base_image = img
    host.modifications = {}
    host._cls_snapshot = None
    host.apply_all_modifications_to_image = lambda base, mods: base
    host._resolve_sklearn_model_bundle = lambda: {
        "model": _FakeModel(),
        "feature_names": feat_names,
        "label_names": [0, 1],
        "expressions": [],
        "window_size": window_size,
        "base_feature_names": base_names,
    }
    host._make_feature_stack_for_model = ImageEditorDialog._make_feature_stack_for_model.__get__(
        host, ImageEditorDialog)
    host.reapply_modifications = lambda: None  # out of scope for this fix

    def run_with_budget(budget):
        h2 = QtWidgets.QWidget()
        for attr in ("base_image", "modifications", "apply_all_modifications_to_image",
                     "_resolve_sklearn_model_bundle", "_make_feature_stack_for_model",
                     "reapply_modifications"):
            setattr(h2, attr, getattr(host, attr))
        h2._cls_snapshot = None
        h2._CLASSIFY_FEATURE_BUDGET_BYTES = budget
        ImageEditorDialog.run_sklearn_classification(h2)
        return h2._classification_result.copy()

    result_untiled = run_with_budget(10**9)     # fits in one tile
    result_tiled = run_with_budget(2_000)       # forces many tiles

    assert np.array_equal(result_untiled, result_tiled), (
        "tiled classification must produce the exact same class map as "
        "the untiled path")
    # Sanity: the top half (red high) should classify differently from the
    # bottom half (red low) somewhere away from the window-edge seam.
    assert result_untiled[2, 5] != result_untiled[H - 3, 5]


def test_run_sklearn_classification_stays_fast_on_a_moderately_large_image(qapp):
    """Timing bound generous enough to never be flaky, but tight enough
    that the OLD per-pixel loop (O(H*W*window^2*n_features) Python-level
    ops) would blow through it by orders of magnitude."""
    from PyQt5 import QtWidgets

    H, W = 220, 180
    img = _rgb_image(H, W, seed=55)
    window_size = 3
    base_names = ["red_channel", "green_channel", "blue_channel"]
    feat_names = list(range(window_size * window_size * 3))

    host = QtWidgets.QWidget()
    host.base_image = img
    host.modifications = {}
    host._cls_snapshot = None
    host.apply_all_modifications_to_image = lambda base, mods: base
    host._resolve_sklearn_model_bundle = lambda: {
        "model": _FakeModel(),
        "feature_names": feat_names,
        "label_names": [0, 1],
        "expressions": [],
        "window_size": window_size,
        "base_feature_names": base_names,
    }
    host._make_feature_stack_for_model = ImageEditorDialog._make_feature_stack_for_model.__get__(
        host, ImageEditorDialog)
    host.reapply_modifications = lambda: None
    host._CLASSIFY_FEATURE_BUDGET_BYTES = ImageEditorDialog._CLASSIFY_FEATURE_BUDGET_BYTES

    t0 = time.perf_counter()
    ImageEditorDialog.run_sklearn_classification(host)
    elapsed = time.perf_counter() - t0

    assert host._classification_result is not None
    assert elapsed < 5.0, (
        f"classifying a {H}x{W} image with window={window_size} took "
        f"{elapsed:.2f}s -- the old per-pixel loop would take vastly "
        "longer than this on an image this size")


# ---------------------------------------------------------------------------
# The two .ax "appended classification band" replay sites --
# _apply_ax_to_raw (export/CSV path) and apply_aux_modifications (the
# viewer's @staticmethod display path) -- both reachable at FULL export
# resolution, not just via the editor's Classify button.
# ---------------------------------------------------------------------------
class _WindowedFakeModel:
    """A window_size=3 model whose prediction genuinely depends on the
    input (mean of all 27 features vs a threshold) -- unlike a
    constant-output stub, this can actually expose a tile-boundary bug."""
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (X.mean(axis=1) > 128).astype(np.int64)


def _windowed_bundle():
    base_names = ["red_channel", "green_channel", "blue_channel"]
    return {
        "model": _WindowedFakeModel(),
        "feature_names": list(range(9 * 3)),
        "base_feature_names": base_names,
        "expressions": [],
        "window_size": 3,
        "label_names": [0, 1],
    }


def _stub_project_tab(tmp_path):
    from ..project_tab import ProjectTab

    class _Tab:
        pass

    tab = _Tab()
    tab.project_folder = str(tmp_path)
    tab._load_ax_json = ProjectTab._load_ax_json.__get__(tab, ProjectTab)
    tab._ax_path_for_fp = ProjectTab._ax_path_for_fp.__get__(tab, ProjectTab)
    tab._ax_path_for = ProjectTab._ax_path_for.__get__(tab, ProjectTab)
    tab._apply_ax_to_raw = ProjectTab._apply_ax_to_raw.__get__(tab, ProjectTab)
    tab._get_sklearn_bundle = ProjectTab._get_sklearn_bundle.__get__(tab, ProjectTab)
    tab._make_feature_stack_for_model = ProjectTab._make_feature_stack_for_model.__get__(tab, ProjectTab)
    tab._record_appended_band_error = ProjectTab._record_appended_band_error.__get__(tab, ProjectTab)
    tab.exiftool_path = None
    tab.random_forest_model = _windowed_bundle()
    return tab


def test_apply_ax_to_raw_classification_replay_tiled_matches_untiled(tmp_path, monkeypatch):
    from .. import project_tab as pt_module

    H, W = 60, 30
    img = _rgb_image(H, W, seed=17)
    ax = {"appended_bands": [{"type": "classification", "index": 1,
                              "label_names": [0, 1]}]}

    def classify(budget):
        tab = _stub_project_tab(tmp_path)
        monkeypatch.setattr(pt_module, "_CLASSIFY_FEATURE_BUDGET_BYTES", budget)
        out, C = tab._apply_ax_to_raw(img.copy(), ax, filepath=str(tmp_path / "a.tif"))
        assert C == 4, "RGB + 1 appended classification band"
        return out[:, :, 3]

    band_untiled = classify(10**9)
    band_tiled = classify(500)  # forces many tiny row-band tiles

    assert np.array_equal(band_untiled, band_tiled), (
        "the appended classification band must be identical whether built "
        "in one shot or tiled")


def test_apply_aux_modifications_classification_replay_tiled_matches_untiled(tmp_path, monkeypatch):
    """apply_aux_modifications loads its .ax from a REAL sidecar path (it's
    a @staticmethod, no self._load_ax_json to hand it a dict directly) --
    write one next to a (non-existent, that's fine -- only the pixel array
    passed in is actually read) image path in tmp_path."""
    import json
    from ..project_tab import ProjectTab
    from .. import project_tab as pt_module

    H, W = 50, 26
    img = _rgb_image(H, W, seed=29)
    image_filepath = str(tmp_path / "a.tif")
    ax_path = str(tmp_path / "a.ax")
    with open(ax_path, "w", encoding="utf-8") as f:
        json.dump({"appended_bands": [{"type": "classification", "index": 1,
                                       "label_names": [0, 1]}]}, f)

    def classify(budget):
        monkeypatch.setattr(pt_module, "_CLASSIFY_FEATURE_BUDGET_BYTES", budget)
        monkeypatch.setattr(ProjectTab, "shared_random_forest_model", _windowed_bundle(), raising=False)
        return ProjectTab.apply_aux_modifications(
            image_filepath, img.copy(), project_folder=None, global_mode=False,
            export_label_band=True,
        )

    out_untiled = classify(10**9)
    out_tiled = classify(500)

    assert out_untiled.shape[2] == out_tiled.shape[2] == 4
    assert np.array_equal(out_untiled[:, :, 3], out_tiled[:, :, 3]), (
        "the appended classification band must be identical whether built "
        "in one shot or tiled")
