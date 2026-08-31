"""
CONTRACT: the viewer and the CSV must classify the SAME pixels.

Three independent rules decide whether a pixel is eligible for classification,
and `MachineLearningManager.train_models` applies all three when it builds the
training set. A pixel is DROPPED when:

    1. any value in its window is a declared numeric NoData literal
    2. any value in its window is non-finite (NaN/Inf)
    3. `window_size > 1` and its window would extend past the image edge

`ProjectTab.process_polygon` -- the CSV's `Class N %` columns -- already applied
all three. Three inference paths did not, each in a different way:

    _make_feature_stack_for_model   no NoData check at all, MEDIAN-IMPUTED
    (viewer render + editor          non-finite features, and edge-padded the
     Classify + appended-band         border
     replay)
    _apply_ax_to_raw / apply_aux_   handled NoData and non-finite, but still
    modifications' classification    edge-padded the border
    _predict_class_map_from_bundle  returned None for the WHOLE image if any
                                     single pixel was non-finite

The consequence was not an error message: it was the user reading one set of
class percentages off the screen and a different set out of the CSV, for the
same polygon on the same image, because the two were describing different
pixel populations.

WHY "MATCH TRAINING" IS THE RIGHT RULE
-------------------------------------
Inference has to be made on the kind of feature row the model was actually fit
on. A median-imputed value, an edge-padded neighbour, and a NoData sentinel are
all rows `train_models` deliberately refused to learn from, so a prediction
made from one is unsupported by the model -- and, separately, it is a pixel the
CSV does not count. Excluding them satisfies both concerns at once; imputing
them satisfies neither.

Excluded pixels now come back as `utils.UNCLASSIFIED_CLASS_INDEX` (-1) rather
than silently receiving a class.

These tests drive the REAL production functions end to end and compare the
numbers a user would actually see.
"""
import json
import os

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from ..project_tab import ProjectTab
from ..utils import (UNCLASSIFIED_CLASS_INDEX, classification_invalid_rows,
                     nodata_match_tolerance, window_edge_rows)

pytestmark = [pytest.mark.contract, pytest.mark.extraction, pytest.mark.ml]


H = W = 40
BASE_FEATURES = ["red_channel", "green_channel", "blue_channel"]

#: a 6x6 all-NaN block, in the LEFT (low-red) half
NAN_BLOCK = (slice(5, 11), slice(5, 11))
#: a 6x6 block of the declared NoData literal, also in the left half.
#: 9999 is deliberately ABOVE the model's decision threshold (128) while its
#: neighbourhood is below it, so counting these pixels visibly moves the
#: percentages -- a NoData value that classified like its surroundings would
#: make the whole comparison vacuous.
ND_BLOCK = (slice(25, 31), slice(6, 12))
ND_VALUE = 9999.0


def _window_feature_names(base_names, win):
    """The exact column naming/order `train_models`' own
    generate_window_feature_names produces (row-major (dr,dc) outer, band
    inner). Reproduced here only to LABEL a bundle -- no pixel math."""
    if win == 1:
        return list(base_names)
    half = win // 2
    return [f"{b}_r{dr:+d}c{dc:+d}"
            for dr in range(-half, half + 1)
            for dc in range(-half, half + 1)
            for b in base_names]


def _bundle(win):
    """A REAL fitted sklearn model whose only meaningful feature is column 0
    (the centre-window red channel), thresholded at 128."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 255.0, size=(800, 3 * win * win)).astype(np.float32)
    y = (X[:, 0] > 128.0).astype(int)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.95, "fixture model is not usable"
    return {"model": clf,
            "feature_names": _window_feature_names(BASE_FEATURES, win),
            # label_names must match model.classes_, or the label->index map
            # cannot resolve a prediction.
            "label_names": [0, 1],
            "expressions": [],
            "window_size": win,
            "base_feature_names": BASE_FEATURES}


def _image(with_nodata):
    """4 bands so both the viewer and the export path load it through
    tifffile (channel_order "rgb"), keeping band order out of this file's
    variables -- that is test_band_convention_contract.py's subject.

    Left 50% has red < 128, right 50% has red > 128, so the model's output is
    genuinely spatially structured rather than a single class.
    """
    rng = np.random.default_rng(7)
    red = rng.uniform(20.0, 100.0, size=(H, W)).astype(np.float32)
    red[:, W // 2:] = rng.uniform(160.0, 250.0,
                                  size=(H, W - W // 2)).astype(np.float32)
    green = rng.uniform(0.0, 255.0, size=(H, W)).astype(np.float32)
    blue = rng.uniform(0.0, 255.0, size=(H, W)).astype(np.float32)
    band4 = np.full((H, W), 7.0, dtype=np.float32)
    img = np.stack([red, green, blue, band4], axis=-1)
    img[NAN_BLOCK[0], NAN_BLOCK[1], :] = np.nan
    if with_nodata:
        img[ND_BLOCK[0], ND_BLOCK[1], :] = ND_VALUE
    return img


@pytest.fixture(scope="module")
def cls_project(qapp, tmp_path_factory):
    """ONE real ProjectTab holding two variants of the same scene.

    A fresh ProjectTab per test is what the rest of this suite avoids (see
    test_contract_ax_consumers' _ax_tab), so the `.ax` is rewritten per test
    instead.
    """
    import tifffile
    from PyQt5 import QtWidgets

    root = tmp_path_factory.mktemp("cls_eligibility")
    images = root / "images"
    images.mkdir()

    paths = {}
    for key, with_nd in (("nodata", True), ("nan", False)):
        fp = str(images / f"cls_{key}.tif")
        tifffile.imwrite(_as_path(fp), _image(with_nd),
                         photometric="minisblack", planarconfig="contig")
        paths[key] = fp

    pts = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
    all_polygons, groups, root_names = {}, {}, []
    for i, (key, fp) in enumerate(sorted(paths.items())):
        gname = "poly__" + os.path.basename(fp)
        all_polygons[gname] = {fp: {
            "points": [[float(x), float(y)] for (x, y) in pts],
            "coord_space": "image",
            "image_ref_size": {"w": W, "h": H},
            "name": gname, "root": str(i + 1), "type": "polygon"}}
        rn = f"root_{i}"
        root_names.append(rn)
        groups[rn] = [fp]

    project_folder = root / "project"
    project_folder.mkdir()
    with open(project_folder / "project.json", "w", encoding="utf-8") as f:
        json.dump({"project_name": "cls_eligibility", "mode": "rgb_only",
                   "root_offset": 0, "all_polygons": all_polygons,
                   "current_root_index": 0, "root_coordinates": {},
                   "rgb_folder_path": str(images), "batch_size": 1,
                   "multispectral_root_names": root_names,
                   "multispectral_image_data_groups": groups}, f)

    saved_shared = getattr(ProjectTab, "shared_random_forest_model", None)
    pt = ProjectTab("cls_eligibility", QtWidgets.QTabWidget())
    pt.load_project(str(project_folder))

    yield {"tab": pt, "nodata": paths["nodata"], "nan": paths["nan"]}

    # _get_sklearn_bundle COPIES the per-tab bundle onto the CLASS attribute
    # ProjectTab.shared_random_forest_model. Leaving a test model there would
    # silently enable classification for every later test in the process.
    ProjectTab.shared_random_forest_model = saved_shared
    for attr in ("_poly_idle_timer", "_autosave_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass


def _as_path(p):
    return p


def _clear_caches(pt):
    for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                 "_bm_expr_cache", "_file_nodata_cache", "_per_band_nd_cache",
                 "_master_nd_cache", "_nodata_mask_by_filepath"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            try:
                c.clear()
            except Exception:
                pass
    for attr in ("_last_export_nodata_mask", "_last_export_nodata_filepath"):
        if hasattr(pt, attr):
            setattr(pt, attr, None)


@pytest.fixture
def scenario(cls_project):
    """Point the shared project at one image + one .ax + one model bundle,
    and clean all of that up afterwards."""
    pt = cls_project["tab"]
    state = {}

    def _setup(which, window_size, nodata_values):
        fp = cls_project[which]
        ax = ({"nodata_enabled": True, "nodata_values": list(nodata_values)}
              if nodata_values else {})
        with open(pt._ax_path_for(fp), "w", encoding="utf-8") as f:
            json.dump(ax, f)
        _clear_caches(pt)
        bundle = _bundle(window_size)
        pt.random_forest_model = bundle
        ProjectTab.shared_random_forest_model = bundle
        state["fp"] = fp
        return pt, fp, bundle

    yield _setup

    fp = state.get("fp")
    if fp:
        try:
            p = pt._ax_path_for(fp)
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    pt.random_forest_model = None
    ProjectTab.shared_random_forest_model = None
    _clear_caches(pt)


# ---------------------------------------------------------------------------
# The two paths, each through its real entry point
# ---------------------------------------------------------------------------
def _csv_class_percentages(pt, fp):
    """What process_polygon writes into the CSV's `Class N %` columns."""
    group = "poly__" + os.path.basename(fp)
    poly = pt.all_polygons[group][fp]
    rows, _ = pt.process_polygon(group, fp, poly, {}, [], True,
                                 opts={"stats": {"mean": True}})
    out = {}
    for r in rows:
        if isinstance(r, dict):
            for k, v in r.items():
                if str(k).startswith("Class ") and v is not None:
                    out[str(k)] = float(v)
    assert out, "process_polygon produced no Class % columns"
    return out


def _viewer_class_map(pt, fp, nodata_values):
    """The per-pixel class map the viewer renders, via the real replay +
    the real classifier entry point."""
    lite = pt._imagedata_or_fallback(fp)
    base = ProjectTab.apply_aux_modifications(
        fp, lite.image, pt.project_folder, global_mode=False,
        src_channel_order=lite.channel_order)
    result = pt._apply_sklearn_classification_with_indices(
        base, nodata_values=list(nodata_values or []),
        src_channel_order=lite.channel_order)
    assert result is not None, "classifier returned None (no bundle?)"
    _colorized, idx = result
    return np.asarray(idx)


def _percentages_from_class_map(idx, classes=(0, 1)):
    classified = idx[idx >= 0]
    assert classified.size, "no pixel was classified at all"
    return {f"Class {k} %": 100.0 * float((classified == k).sum()) / classified.size
            for k in classes}


# ---------------------------------------------------------------------------
# 1. NoData + non-finite, window_size == 1
# ---------------------------------------------------------------------------
def test_window1_viewer_and_csv_report_the_same_class_percentages(scenario):
    pt, fp, _b = scenario("nodata", 1, [ND_VALUE])

    csv = _csv_class_percentages(pt, fp)
    idx = _viewer_class_map(pt, fp, [ND_VALUE])
    viewer = _percentages_from_class_map(idx)

    for key in sorted(set(csv) | set(viewer)):
        assert key in csv and key in viewer, f"{key} missing from one path"
        assert viewer[key] == pytest.approx(csv[key], abs=1e-6), (
            f"{key}: viewer says {viewer[key]}, CSV says {csv[key]} -- the two "
            "are describing different pixel populations")


def test_window1_excludes_exactly_the_nodata_and_nonfinite_pixels(scenario):
    pt, fp, _b = scenario("nodata", 1, [ND_VALUE])
    idx = _viewer_class_map(pt, fp, [ND_VALUE])

    excluded = idx == UNCLASSIFIED_CLASS_INDEX
    expected = np.zeros((H, W), dtype=bool)
    expected[NAN_BLOCK[0], NAN_BLOCK[1]] = True
    expected[ND_BLOCK[0], ND_BLOCK[1]] = True

    assert np.array_equal(excluded, expected), (
        f"excluded {int(excluded.sum())} pixels, expected "
        f"{int(expected.sum())} (the NaN block and the NoData block, and "
        "nothing else)")


def test_window1_exclusion_actually_changes_the_numbers(scenario):
    """Anti-vacuum guard. If the excluded pixels had been classified anyway,
    the percentages would be measurably different -- so the agreement above
    is a real constraint, not an accident of a symmetric fixture."""
    pt, fp, _b = scenario("nodata", 1, [ND_VALUE])
    idx = _viewer_class_map(pt, fp, [ND_VALUE])
    csv = _csv_class_percentages(pt, fp)

    n_excluded = int((idx == UNCLASSIFIED_CLASS_INDEX).sum())
    assert n_excluded > 0, "nothing was excluded -- fixture is not exercising the rule"

    classified = idx[idx >= 0]
    n1 = int((classified == 1).sum())
    # Hypothetical: count the excluded pixels too (as either class).
    for hypothetical in (100.0 * (n1 + n_excluded) / (classified.size + n_excluded),
                         100.0 * n1 / (classified.size + n_excluded)):
        assert abs(hypothetical - csv["Class 1 %"]) > 0.5, (
            "including the excluded pixels would not change Class 1 % -- this "
            "fixture cannot distinguish the two populations")


# ---------------------------------------------------------------------------
# 2. Window edge + non-finite, window_size == 3
# ---------------------------------------------------------------------------
def test_window3_viewer_and_csv_report_the_same_class_percentages(scenario):
    """No numeric NoData here on purpose: at window_size > 1 the two paths are
    compared on the two rules they BOTH implement pixel-for-pixel -- window
    edge and non-finite propagation through the window."""
    pt, fp, _b = scenario("nan", 3, [])

    csv = _csv_class_percentages(pt, fp)
    idx = _viewer_class_map(pt, fp, [])
    viewer = _percentages_from_class_map(idx)

    for key in sorted(set(csv) | set(viewer)):
        assert viewer[key] == pytest.approx(csv[key], abs=1e-6), (
            f"{key}: viewer says {viewer[key]}, CSV says {csv[key]}")


def test_window3_excludes_the_border_and_the_nonfinite_neighbourhood(scenario):
    """The exclusion set at window_size=3 is exactly:
    the 1-pixel image border (the window would leave the image) UNION the NaN
    block dilated by 1 (the window would contain a NaN)."""
    pt, fp, _b = scenario("nan", 3, [])
    idx = _viewer_class_map(pt, fp, [])
    excluded = idx == UNCLASSIFIED_CLASS_INDEX

    expected = window_edge_rows(H, W, 3).reshape(H, W).copy()
    nan_mask = np.zeros((H, W), dtype=bool)
    nan_mask[NAN_BLOCK[0], NAN_BLOCK[1]] = True
    dilated = np.zeros((H, W), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dilated |= np.roll(np.roll(nan_mask, dy, axis=0), dx, axis=1)
    expected |= dilated

    assert np.array_equal(excluded, expected), (
        f"excluded {int(excluded.sum())} pixels, expected {int(expected.sum())}")
    assert excluded[0, :].all() and excluded[-1, :].all(), "border not excluded"
    assert (~excluded).sum() > 0.5 * H * W, (
        "more than half the image was excluded -- the fixture is degenerate")


def test_window1_does_not_exclude_a_border(scenario):
    """Control for the geometry rule: with window_size == 1 there is no
    window to run off the edge, so the border must stay classified."""
    pt, fp, _b = scenario("nan", 1, [])
    idx = _viewer_class_map(pt, fp, [])
    excluded = idx == UNCLASSIFIED_CLASS_INDEX
    assert not excluded[0, :].any() and not excluded[:, 0].any(), (
        "window_size == 1 must not exclude the image border")


# ---------------------------------------------------------------------------
# 3. The exported label band follows the same rule
# ---------------------------------------------------------------------------
def test_export_label_band_leaves_the_window_border_unclassified(scenario):
    """`_apply_ax_to_raw`'s own classification block edge-padded the border
    too, so the exported label band disagreed with the CSV percentages built
    from the same run. It fills excluded pixels with its `nodata_fill`
    (the declared numeric NoData literal when there is one)."""
    pt, fp, _b = scenario("nodata", 3, [ND_VALUE])

    ax = {"nodata_enabled": True, "nodata_values": [ND_VALUE],
          "classification": {"mode": "sklearn", "enabled": True}}
    raw = np.asarray(_image(True), dtype=np.float32)
    out, _chans = pt._apply_ax_to_raw(raw, ax, export_label_band=True,
                                      filepath=fp, src_channel_order="rgb")
    out = np.asarray(out)
    assert out.shape[2] == 5, (
        f"expected 4 source bands + 1 label band, got {out.shape[2]}")
    label = out[..., -1]

    border = window_edge_rows(H, W, 3).reshape(H, W)
    assert np.all(label[border] == ND_VALUE), (
        "the exported label band still classifies the window border from "
        "edge-padded neighbours")
    interior_classes = np.unique(label[~border])
    assert set(interior_classes.tolist()) - {ND_VALUE}, (
        "no interior pixel received a class -- the classifier did not run")


# ---------------------------------------------------------------------------
# 4. The editor's separate feature builder follows the same rule
# ---------------------------------------------------------------------------
@pytest.mark.editor
@pytest.mark.parametrize("window_size", [1, 3])
def test_editor_builder_marks_the_same_rows_as_project_tab(window_size):
    """Two independent copies of `_make_feature_stack_for_model` exist. The
    editor's had NO non-finite and NO NoData handling at all, so the editor's
    Classify preview covered a different pixel set from everything else."""
    from ..image_editor_dialog import ImageEditorDialog

    img = _image(True)
    feat_names = _window_feature_names(BASE_FEATURES, window_size)

    X_pt, hw_pt = ProjectTab._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=BASE_FEATURES, src_channel_order="rgb",
        nodata_values=[ND_VALUE])
    X_ed, hw_ed = ImageEditorDialog._make_feature_stack_for_model(
        None, img, feat_names, expressions=None, window_size=window_size,
        base_feature_names=BASE_FEATURES, src_channel_order="rgb",
        nodata_values=[ND_VALUE])

    assert hw_pt == hw_ed == (H, W)
    bad_pt = ~np.all(np.isfinite(X_pt), axis=1)
    bad_ed = ~np.all(np.isfinite(X_ed), axis=1)
    assert np.array_equal(bad_pt, bad_ed), (
        "the two feature builders disagree about which pixels are eligible")
    assert bad_pt.any(), "fixture excluded nothing -- test is vacuous"
    assert np.array_equal(X_pt, X_ed, equal_nan=True)


# ---------------------------------------------------------------------------
# 5. The eligibility rule itself matches TRAINING, value for value
# ---------------------------------------------------------------------------
def test_nodata_tolerance_matches_the_real_training_row_builder():
    """`classification_invalid_rows` and
    `MachineLearningManager._build_training_row` must agree on whether a given
    value counts as the NoData literal. The training function is called for
    real here -- no reimplementation of its ladder."""
    from ..machine_learning_manager import MachineLearningManager

    for nd in (0.0, 5.0, 250.0, -9999.0, 1e12, 1e35):
        tol = nodata_match_tolerance(nd)
        inside = nd + tol * 0.5
        outside = nd + tol * 2.0
        for value, expect_dropped in ((nd, True), (inside, True), (outside, False)):
            chans = [np.full((3, 3), float(value), dtype=np.float32)]
            _row, ok, _n_nan, _n_missing = MachineLearningManager._build_training_row(
                chans, None, 1, 1, 1, [nd], 1)
            assert ok is not expect_dropped, (
                f"training row for value {value} vs NoData {nd}: ok={ok}, "
                f"expected dropped={expect_dropped}")

            X = np.array([[float(value)]], dtype=np.float32)
            got = bool(classification_invalid_rows(X, 1, 1, 1, [nd])[0])
            assert got == expect_dropped, (
                f"classification_invalid_rows disagrees with training for "
                f"value {value} vs NoData {nd}")


def test_expression_nodata_entries_are_not_matched_against_feature_values():
    """"b1>100" is a spatial rule resolved against the IMAGE by
    build_nodata_mask, not a literal to compare feature values with --
    `_build_training_row` makes the same split (its float() conversion of an
    expression string raises and is swallowed)."""
    X = np.array([[0.0, 50.0, 100.0]], dtype=np.float32)
    assert not classification_invalid_rows(X, 1, 1, 1, ["b1>100"]).any()
    assert not classification_invalid_rows(X, 1, 1, 1, ["b2<=50"]).any()


def test_window_edge_rows_matches_the_training_skip_condition():
    """train_models skips `yy - half < 0 or yy + half >= H or xx - half < 0 or
    xx + half >= W`. Assert the helper reproduces that predicate exactly, for
    every pixel, at several window sizes."""
    for h, w in ((7, 5), (9, 9), (4, 12)):
        for win in (1, 3, 5):
            half = win // 2
            got = window_edge_rows(h, w, win).reshape(h, w)
            for y in range(h):
                for x in range(w):
                    expected = (win > 1) and (
                        y - half < 0 or y + half >= h
                        or x - half < 0 or x + half >= w)
                    assert bool(got[y, x]) == expected, (
                        f"({x},{y}) h={h} w={w} win={win}: got {got[y, x]}, "
                        f"expected {expected}")
