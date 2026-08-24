"""QC tests pinning "augmented pixels must never leak into the holdout/report
split" -- the core new-requirement behind the ML Manager's training
augmentation feature.

`ml_augmentation.assemble_pixel_rows` is the policy function
`machine_learning_manager.py::train_models` calls, once per successfully-
sampled pixel, to decide what (if anything) gets added to the TRAINING row
collection. The REPORT/holdout row collection is built separately and
unconditionally from the pristine row alone, by train_models itself, on
every call -- this file pins assemble_pixel_rows' policy in isolation
(pure-Python, no image/GUI state needed), which is the piece most likely to
silently regress: a naive single `is_augmented` mask column (rather than two
separate row collections) can shrink or exhaust the holdout-eligible pool in
Replace mode -- see the "add vs replace" design note this test file guards.
"""
import pytest

from ..ml_augmentation import assemble_pixel_rows

pytestmark = [pytest.mark.ml]


# ---------------------------------------------------------------------------
# pristine row itself failed -- nothing to train on, augmentation can't help
# ---------------------------------------------------------------------------
def test_failed_pristine_row_yields_nothing_regardless_of_augmented_results():
    out = assemble_pixel_rows(
        pristine_ok=False, pristine_row=[1.0, 2.0],
        augmented_results=[(True, [9.0, 9.0])], row_policy="add")
    assert out == []


# ---------------------------------------------------------------------------
# "add" mode
# ---------------------------------------------------------------------------
def test_add_mode_with_no_augmented_variants_returns_just_the_pristine_row():
    out = assemble_pixel_rows(True, [1.0, 2.0], [], "add")
    assert out == [[1.0, 2.0]]


def test_add_mode_appends_every_successful_augmented_variant():
    out = assemble_pixel_rows(
        True, [1.0, 2.0],
        [(True, [10.0, 20.0]), (True, [30.0, 40.0]), (True, [50.0, 60.0])],
        "add")
    assert out == [[1.0, 2.0], [10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], (
        "N=3 successful augmented variants must yield exactly 3 extra rows "
        "alongside the original -- this is what 'increase sample size "
        "artificially' means")


def test_add_mode_skips_failed_augmented_variants_but_keeps_successful_ones():
    out = assemble_pixel_rows(
        True, [1.0, 2.0],
        [(False, [0.0, 0.0]), (True, [10.0, 20.0]), (False, [0.0, 0.0])],
        "add")
    assert out == [[1.0, 2.0], [10.0, 20.0]]


def test_add_mode_with_every_variant_failed_still_keeps_the_pristine_row():
    out = assemble_pixel_rows(
        True, [1.0, 2.0], [(False, [0.0, 0.0]), (False, [0.0, 0.0])], "add")
    assert out == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# "replace" mode -- must ALWAYS return exactly one row, never zero, never
# more than one, regardless of how many variants were attempted or how many
# succeeded (the pool-exhaustion / size-drift regression this design avoids)
# ---------------------------------------------------------------------------
def test_replace_mode_uses_the_first_successful_augmented_row():
    out = assemble_pixel_rows(
        True, [1.0, 2.0],
        [(False, [0.0, 0.0]), (True, [10.0, 20.0]), (True, [30.0, 40.0])],
        "replace")
    assert out == [[10.0, 20.0]], "replace must use the FIRST successful variant, not the last"


def test_replace_mode_falls_back_to_the_pristine_row_when_every_variant_fails():
    out = assemble_pixel_rows(
        True, [1.0, 2.0], [(False, [0.0, 0.0]), (False, [0.0, 0.0])], "replace")
    assert out == [[1.0, 2.0]], (
        "a pixel must never be silently dropped just because its augmented "
        "attempt(s) happened to fail")


def test_replace_mode_with_no_augmented_variants_falls_back_to_pristine():
    out = assemble_pixel_rows(True, [1.0, 2.0], [], "replace")
    assert out == [[1.0, 2.0]]


@pytest.mark.parametrize("n_variants,n_successful", [(1, 0), (1, 1), (5, 0), (5, 3), (5, 5)])
def test_replace_mode_always_returns_exactly_one_row(n_variants, n_successful):
    """The regression test for the pool-exhaustion bug a single is_augmented
    mask would introduce: replace mode's row COUNT per pixel must be
    invariant to how many variants were generated or how many succeeded."""
    results = [(i < n_successful, [float(i), float(i)]) for i in range(n_variants)]
    out = assemble_pixel_rows(True, [1.0, 2.0], results, "replace")
    assert len(out) == 1


@pytest.mark.parametrize("n_variants", [0, 1, 3, 10])
def test_add_mode_row_count_matches_original_plus_successful_variants(n_variants):
    results = [(True, [float(i), float(i)]) for i in range(n_variants)]
    out = assemble_pixel_rows(True, [9.0, 9.0], results, "add")
    assert len(out) == 1 + n_variants


# ---------------------------------------------------------------------------
# End-to-end-in-miniature: simulate what train_models' per-pixel loop does
# for a handful of pixels, and assert the REPORT collection is completely
# untouched by augmentation while the TRAIN collection reflects row_policy --
# this is the exact shape of the invariant train_models itself must uphold.
# ---------------------------------------------------------------------------
def _simulate_one_image(pixel_specs, row_policy):
    """pixel_specs: list of (pristine_ok, pristine_row, augmented_results).
    Mirrors train_models' bookkeeping: X_rows_report always grows by exactly
    one entry per ok pixel; X_rows_train is either the SAME list object
    (augmentation disabled) or grows per assemble_pixel_rows' policy."""
    X_report, y_report = [], []
    augmentation_enabled = any(spec[2] for spec in pixel_specs)  # any variants generated at all
    if augmentation_enabled:
        X_train, y_train = [], []
    else:
        X_train, y_train = X_report, y_report

    for (ok, row, aug_results) in pixel_specs:
        if not ok:
            continue
        X_report.append(row)
        y_report.append("cls")
        if augmentation_enabled:
            for tr in assemble_pixel_rows(ok, row, aug_results, row_policy):
                X_train.append(tr)
                y_train.append("cls")

    return X_report, X_train


def test_report_collection_is_identical_with_augmentation_on_or_off():
    pixels_off = [(True, [1.0], []), (True, [2.0], []), (False, [3.0], [])]
    pixels_on = [(True, [1.0], [(True, [1.5])]), (True, [2.0], [(True, [2.5])]), (False, [3.0], [])]

    report_off, _ = _simulate_one_image(pixels_off, "add")
    report_on, _ = _simulate_one_image(pixels_on, "add")

    assert report_off == report_on == [[1.0], [2.0]], (
        "the report/holdout row collection must be byte-identical whether "
        "or not augmentation ran -- augmentation must never be able to "
        "change what the holdout-metrics split is computed from")


def test_train_collection_is_the_same_list_object_as_report_when_augmentation_never_ran():
    pixel_specs = [(True, [1.0], []), (True, [2.0], [])]
    X_report, X_train = _simulate_one_image(pixel_specs, "add")
    assert X_train is X_report, (
        "when no pixel produced any augmented variant, X_rows_train must be "
        "the SAME list object as X_rows_report (zero extra cost for the "
        "common case) -- not a separately-populated duplicate")
