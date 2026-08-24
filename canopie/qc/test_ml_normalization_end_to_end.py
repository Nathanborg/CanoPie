"""QC regression test: a bundle's "normalization" config must be applied
IDENTICALLY at every classification call site that later loads it and
predicts -- not just by the primitives in test_ml_normalization_helpers.py,
but by the actual wired call sites across project_tab.py,
image_editor_dialog.py, and machine_learning_manager.py.

The falsifiability trick: the model is fit on Z-SCORE-NORMALIZED features
with a decision boundary calibrated to roughly [-2, 2]-range values. Raw
(unnormalized, ~0-255 range) pixel values fed straight into that model
without normalization land WAY outside the values the tree ever split on,
so a call site that forgets to normalize doesn't just produce "slightly
different" predictions -- it collapses to (near-)one class everywhere,
which is trivially distinguishable from the correct, spatially-varying
ground truth this test checks against.

Covers 3 of the 6 documented predict sites (per the ML augmentation plan's
Part 7 test list): MachineLearningManager._predict_class_map_from_bundle
(simplest signature, bundle passed directly), and both
_make_feature_stack_for_model copies (project_tab.py's and
image_editor_dialog.py's) -- which together also cover
_apply_sklearn_classification[_with_indices] and both .ax "appended
classification band" replay blocks, since those all route through one or
the other of these two functions (see the plan's Part 5 for the full
6-site map).
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from ..machine_learning_manager import MachineLearningManager
from ..project_tab import ProjectTab
from ..image_editor_dialog import ImageEditorDialog
from ..utils import fit_normalization, apply_normalization

pytestmark = [pytest.mark.ml]

FEATURE_NAMES = ["red_channel", "green_channel", "blue_channel"]
LABEL_NAMES = ["low", "high"]


@pytest.fixture(scope="module")
def normalized_bundle():
    """A REAL, small, fitted sklearn model + a REAL fitted zscore
    normalization config -- not a mock. The model is trained purely on
    NORMALIZED features, so it only predicts correctly when callers
    normalize their own raw features the same way before calling .predict()."""
    rng = np.random.default_rng(123)
    X_raw = rng.uniform(0.0, 255.0, size=(400, 3)).astype(np.float32)
    y = (X_raw[:, 0] > 128.0).astype(int)  # ground truth: red_channel > 128

    normalization_cfg = fit_normalization(X_raw, "zscore")
    X_norm = apply_normalization(X_raw, normalization_cfg)

    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_norm, y)
    # Sanity: the model must actually be good on normalized data, or this
    # whole test proves nothing.
    assert (clf.predict(X_norm) == y).mean() > 0.95

    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "band_indices": [0, 1, 2],
        "expressions": [],
        "window_size": 1,
        "base_feature_names": FEATURE_NAMES,
        "normalization": normalization_cfg,
    }
    return bundle, normalization_cfg


def _test_image(h=20, w=20, seed=0):
    """A real RGB image with a spatially-varying, KNOWN ground truth:
    left half red_channel < 128 ("low"), right half > 128 ("high")."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(20.0, 100.0, size=(h, w, 3)).astype(np.float32)
    img[:, w // 2:, 0] = rng.uniform(160.0, 250.0, size=(h, w - w // 2)).astype(np.float32)
    return img


def _ground_truth(img):
    return (img[..., 0] > 128.0).astype(int)


# ---------------------------------------------------------------------------
# Site: MachineLearningManager._predict_class_map_from_bundle
# ---------------------------------------------------------------------------
def test_predict_class_map_from_bundle_applies_normalization(normalized_bundle):
    bundle, _cfg = normalized_bundle
    mgr = MachineLearningManager.__new__(MachineLearningManager)
    img = _test_image(seed=1)
    # chans in export order: R, G, B
    chans = [img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()]

    result = mgr._predict_class_map_from_bundle(chans, bundle)
    assert result is not None

    truth = _ground_truth(img)
    agree = (result.astype(int) == truth).mean()
    assert agree > 0.9, (
        f"only {agree:.0%} of pixels matched ground truth -- normalization "
        "was likely NOT applied before predict()")


def test_predict_class_map_from_bundle_WITHOUT_normalization_key_is_wrong(normalized_bundle):
    """Negative control: strip the normalization key (simulating an old
    bundle, or a regression that drops it) and confirm predictions collapse
    toward one class -- proving the positive test above isn't vacuous."""
    bundle, _cfg = normalized_bundle
    stripped = dict(bundle)
    stripped.pop("normalization", None)
    mgr = MachineLearningManager.__new__(MachineLearningManager)
    img = _test_image(seed=1)
    chans = [img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()]

    result = mgr._predict_class_map_from_bundle(chans, stripped)
    truth = _ground_truth(img)
    agree = (result.astype(int) == truth).mean()
    assert agree < 0.9, (
        "predictions matched ground truth even WITHOUT normalization -- "
        "this negative control is supposed to fail, which would mean the "
        "positive test above proves nothing")


# ---------------------------------------------------------------------------
# Site: ProjectTab._make_feature_stack_for_model
# ---------------------------------------------------------------------------
def test_project_tab_feature_stack_applies_normalization(normalized_bundle):
    bundle, cfg = normalized_bundle
    pt = ProjectTab.__new__(ProjectTab)
    img = _test_image(seed=2)  # RGB, will be BGR-remapped internally then un-remapped by feature order

    # _make_feature_stack_for_model treats a 3-channel img as BGR (OpenCV
    # convention) and remaps to RGB internally -- feed it BGR so the
    # resulting feature columns land on the same RGB semantics the model
    # (and _ground_truth, which reads channel 0 = R) expects.
    img_bgr = img[..., ::-1].copy()

    X, (H, W) = pt._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    assert (H, W) == img.shape[:2]

    preds = bundle["model"].predict(X).reshape(H, W)
    truth = _ground_truth(img)
    agree = (preds == truth).mean()
    assert agree > 0.9, f"only {agree:.0%} matched ground truth"


def test_project_tab_feature_stack_matches_manual_normalization(normalized_bundle):
    """The feature matrix X_normalized this function returns must be
    IDENTICAL to manually building it unnormalized and then calling
    apply_normalization -- not just 'close enough to predict correctly'."""
    bundle, cfg = normalized_bundle
    pt = ProjectTab.__new__(ProjectTab)
    img = _test_image(seed=3)
    img_bgr = img[..., ::-1].copy()

    X_norm, _ = pt._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    X_raw, _ = pt._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=None)
    expected = apply_normalization(X_raw, cfg)

    np.testing.assert_allclose(X_norm, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Site: ImageEditorDialog._make_feature_stack_for_model
# ---------------------------------------------------------------------------
def test_image_editor_feature_stack_applies_normalization(normalized_bundle):
    bundle, cfg = normalized_bundle
    dlg = ImageEditorDialog.__new__(ImageEditorDialog)
    img = _test_image(seed=4)
    img_bgr = img[..., ::-1].copy()

    X, (H, W) = dlg._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    assert (H, W) == img.shape[:2]

    preds = bundle["model"].predict(X).reshape(H, W)
    truth = _ground_truth(img)
    agree = (preds == truth).mean()
    assert agree > 0.9, f"only {agree:.0%} matched ground truth"


def test_image_editor_and_project_tab_feature_stacks_agree_when_normalized(normalized_bundle):
    """Both independent _make_feature_stack_for_model implementations must
    apply the SAME normalization transform, even though they're separate
    copies (per this codebase's documented, deliberate NaN/Inf-imputation
    asymmetry) -- their post-normalization X must still match."""
    bundle, cfg = normalized_bundle
    pt = ProjectTab.__new__(ProjectTab)
    dlg = ImageEditorDialog.__new__(ImageEditorDialog)
    img = _test_image(seed=5)
    img_bgr = img[..., ::-1].copy()

    X_pt, _ = pt._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    X_dlg, _ = dlg._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)

    np.testing.assert_allclose(X_pt, X_dlg, rtol=1e-5)


# ---------------------------------------------------------------------------
# Cross-site agreement: the SAME image/bundle must predict identically
# whichever of the 3 sites classifies it -- proves normalization doesn't
# introduce a site-specific divergence.
# ---------------------------------------------------------------------------
def test_all_three_sites_agree_on_the_same_image(normalized_bundle):
    bundle, cfg = normalized_bundle
    img = _test_image(seed=6)
    img_bgr = img[..., ::-1].copy()

    mgr = MachineLearningManager.__new__(MachineLearningManager)
    chans = [img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()]
    pred_mgr = mgr._predict_class_map_from_bundle(chans, bundle)

    pt = ProjectTab.__new__(ProjectTab)
    X_pt, (H, W) = pt._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    pred_pt = bundle["model"].predict(X_pt).reshape(H, W)

    dlg = ImageEditorDialog.__new__(ImageEditorDialog)
    X_dlg, _ = dlg._make_feature_stack_for_model(
        img_bgr, FEATURE_NAMES, expressions=[], window_size=1,
        base_feature_names=FEATURE_NAMES, normalization_cfg=cfg)
    pred_dlg = bundle["model"].predict(X_dlg).reshape(H, W)

    np.testing.assert_array_equal(pred_mgr.astype(int), pred_pt.astype(int))
    np.testing.assert_array_equal(pred_pt.astype(int), pred_dlg.astype(int))
