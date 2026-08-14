"""
QC regression tests for the export_csv_data performance/threading rewrite.

THE BUG THIS PINS
------------------
`export_csv_data` used to loop `for yi, xi in zip(ys, xs): ...` over every
valid pixel of a polygon, calling a bounds-checking alignment function per
pixel per channel and allocating one dict per pixel
(`all_rows.append(dict(zip(header, row)))`), then deduplicated by hashing
every column of every row into a Python `set`. For a large polygon that is
tens of millions of Python-level operations on the GUI thread -- reported as
the export "grinding to a halt" compared to project_tab.py's vectorized CSV
export.

Fixed by (see machine_learning_manager.py's `_csvexp_*` module functions and
`_CsvExportWorker`):
  1. Vectorized per-channel gathering (O(channels) or O(channels * k^2) numpy
     calls, not O(pixels)).
  2. The extraction now runs on a background QThread, not the GUI thread.
  3. Rows stream directly to the destination CSV as each file finishes.
  4. Deduplication is local to point-shape entries only (vectorized via
     np.unique), not a whole-export hash-every-row set -- see
     _CsvExportWorker's docstring for why that is provably equivalent to the
     original's observable behavior.

A SEPARATE, MORE SEVERE latent bug found and fixed along the way: on a
lazily-read (large/windowed) file, `chans[i]` is a raster_reader.OffsetBand,
which supports [y, x] tuple/scalar indexing but NOT a boolean-array key.
`chans[i][valid_mask]` (the pre-existing "Average Pixel Value" polygon-mean
code, and the old per-pixel loop's equivalent) raises TypeError on such a
file with a multi-point polygon. The vectorized gather helpers fix this too,
since they replace that call site.
"""
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_ground_truth, assert_close

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.ml, pytest.mark.extraction]


def _select_group(mgr, group_name):
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() == group_name)


def _run_export_csv(monkeypatch, mgr, tmp_path, mode, window_choice="3x3"):
    calls = {"n": 0}

    def fake_get_item(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return mode, True
        return window_choice, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(fake_get_item))

    out_csv = tmp_path / "export.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                         staticmethod(lambda *a, **k: (str(out_csv), "")))

    mgr.export_csv_data()
    return out_csv


def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# The lazy/OffsetBand path -- the more severe latent bug
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["Average Pixel Value", "All Pixel Values"])
def test_lazy_windowed_read_does_not_crash_on_multipoint_polygon(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path, force_lazy_export, mode):
    """THE severe latent bug: `chans[i][valid_mask]` on an OffsetBand used to
    raise TypeError for ANY multi-point polygon on a lazily-read file --
    force_lazy_export routes every file through that path regardless of its
    actual size, so this reproduces it on a small fixture."""
    name = "multiband_8band_ancillary"  # has real NoData (band 7 100% fill) too
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, mode)

    rows = _read_csv(out_csv)
    assert rows, f"{mode}: export produced no rows on the lazy path"

    if mode == "Average Pixel Value":
        band0 = gt["polygon"]["bands"]["0"]
        assert_close(float(rows[0]["mean_R"]), band0["no_mask"]["mean"], tol=1e-1,
                     msg=f"lazy-path {mode} mean_R")
    else:
        expected_n = gt["polygon"]["pixel_count_total"]
        assert len(rows) == expected_n, (
            f"lazy path produced {len(rows)} rows, expected {expected_n}")


def test_lazy_path_matches_ground_truth(synthetic_project, ml_manager_factory, monkeypatch, tmp_path, force_lazy_export):
    """Not just 'doesn't crash' -- the lazy (OffsetBand) path must produce the
    exact raw per-pixel values, same as test_all_pixel_values_spot_check
    already verifies for the eager (dense-array) path.

    Deliberately does NOT run a second export with force_lazy_export's patch
    undone to compare "lazy vs eager" directly in one test: `monkeypatch`
    is one shared instance per test, undoing it here would ALSO undo the
    autouse `_no_modal_dialogs` fixture's patch (same instance), and the
    second export's closing `QMessageBox.information(...)` would then call
    the REAL exec_() and hang forever waiting for a click -- exactly the
    documented hang _no_modal_dialogs exists to prevent. Comparing against
    the fixture's own ground truth is an equally strong check without that
    hazard."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "All Pixel Values")
    rows = _read_csv(out_csv)
    assert rows

    from ._helpers import load_raw_npz
    raw = load_raw_npz(name)
    for row in rows[:20]:
        xi, yi = int(row["x"]), int(row["y"])
        for b, ch in enumerate(("R", "G", "B")):
            assert_close(float(row[ch]), float(raw[yi, xi, b]), tol=1e-1,
                         msg=f"lazy path pixel ({xi},{yi}) {ch}")


# ---------------------------------------------------------------------------
# Off-GUI-thread execution
# ---------------------------------------------------------------------------
def test_extraction_runs_off_the_gui_thread(synthetic_project, ml_manager_factory, monkeypatch, tmp_path, qapp):
    """The actual freeze fix: per-file extraction must happen on a QThread,
    not the thread export_csv_data() was called from."""
    from canopie.machine_learning_manager import _CsvExportWorker

    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    gui_thread = QtCore.QThread.currentThread()
    seen_threads = []

    original = _CsvExportWorker._process_one_file

    def spying(self, group_name, filepath, writer):
        seen_threads.append(QtCore.QThread.currentThread())
        return original(self, group_name, filepath, writer)

    monkeypatch.setattr(_CsvExportWorker, "_process_one_file", spying)

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    _run_export_csv(monkeypatch, mgr, tmp_path, "Average Pixel Value")

    assert seen_threads, "extraction never ran -- test fixture problem, not a pass"
    assert all(t is not gui_thread for t in seen_threads), (
        "per-file extraction ran on the GUI thread -- the freeze this worker "
        "exists to fix would still happen")


def test_export_csv_data_is_still_synchronous_from_the_callers_perspective(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """The background thread must not change the calling convention: the CSV
    is fully written by the time export_csv_data() returns (existing tests
    in test_ml_manager_csv_and_training.py depend on exactly this)."""
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])

    mgr = ml_manager_factory(synthetic_project)
    _select_group(mgr, group)
    out_csv = _run_export_csv(monkeypatch, mgr, tmp_path, "All Pixel Values")

    assert out_csv.exists() and out_csv.stat().st_size > 0
    rows = _read_csv(out_csv)
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# Source-level pins -- see the wheel-zoom lesson this session: on fixtures
# this small, wall-clock cannot reliably distinguish "fixed" from "still
# broken", so the literal patterns are pinned directly.
# ---------------------------------------------------------------------------
def test_no_per_pixel_loop_or_dict_allocation_remains():
    import inspect
    from canopie import machine_learning_manager as mlm

    src = inspect.getsource(mlm)
    assert "for yi, xi in zip(ys, xs)" not in src, (
        "the per-pixel (yi, xi) loop is back in export_csv_data")
    assert "all_rows.append(dict(zip(header, row)))" not in src, (
        "the per-pixel dict-allocation pattern is back")
    assert "make_hashable" not in src, (
        "the whole-row hash-and-set dedup helper is back")


def test_gather_helpers_are_actually_used_by_the_worker():
    import inspect
    from canopie import machine_learning_manager as mlm

    src = inspect.getsource(mlm._CsvExportWorker)
    assert "_csvexp_gather_all_channels" in src
    assert "_csvexp_gather_window_all_channels" in src


# ---------------------------------------------------------------------------
# Vectorization actually pays off, on a scale the tiny QC fixtures cannot
# show -- built directly rather than via the fixture manifest.
# ---------------------------------------------------------------------------
def _old_scalar_align_point(chans, xi, yi, any_rgb, max_extras, max_small):
    """The exact pre-rewrite scalar algorithm (_align_point_values),
    reconstructed here (not imported -- it no longer exists in the module)
    purely as the 'before' side of a measured comparison."""
    def in_bounds(W, H, xi, yi):
        return (0 <= xi < W) and (0 <= yi < H)
    vals = []
    if any_rgb:
        rgb = [np.nan, np.nan, np.nan]
        for i in range(min(3, len(chans))):
            ch = chans[i]
            if in_bounds(ch.shape[1], ch.shape[0], xi, yi):
                rgb[i] = float(ch[yi, xi])
        vals.extend(rgb)
        actual_extras = max(0, len(chans) - 3)
        for i in range(max_extras):
            if i < actual_extras:
                ch = chans[3 + i]
                vals.append(float(ch[yi, xi]) if in_bounds(ch.shape[1], ch.shape[0], xi, yi) else np.nan)
            else:
                vals.append(np.nan)
    else:
        for i in range(max_small):
            if i < len(chans):
                ch = chans[i]
                vals.append(float(ch[yi, xi]) if in_bounds(ch.shape[1], ch.shape[0], xi, yi) else np.nan)
            else:
                vals.append(np.nan)
    return vals


@pytest.mark.slow
def test_vectorized_gather_is_faster_than_the_old_scalar_loop():
    """Direct, honest before/after on a scale the QC fixtures (32-96 px)
    cannot demonstrate: a 600x600 polygon mask, 4 bands. Measures the OLD
    algorithm's actual per-pixel cost (reconstructed above) against the
    replacement, on this machine, to show the fix is not merely present but
    effective -- not a hard ceiling (see this session's wheel-zoom lesson
    about small/unreliable margins); the deterministic source pins above are
    the real regression guard."""
    from canopie.machine_learning_manager import _csvexp_gather_all_channels

    rng = np.random.default_rng(0)
    H, W, C = 600, 600, 4
    chans = [rng.uniform(0, 255, size=(H, W)).astype(np.float32) for _ in range(C)]
    yy, xx = np.mgrid[50:550, 50:550]
    ys, xs = yy.ravel(), xx.ravel()  # 250,000 pixels

    t0 = time.perf_counter()
    old_rows = [_old_scalar_align_point(chans, int(x), int(y), True, 1, 0)
                for y, x in zip(ys[:20000], xs[:20000])]  # capped: the old way is genuinely this slow
    t_old_partial = time.perf_counter() - t0
    t_old_estimated_full = t_old_partial * (len(ys) / 20000.0)

    t0 = time.perf_counter()
    cols = _csvexp_gather_all_channels(True, 1, 0, chans, ys, xs)
    new_rows = np.stack(cols, axis=1)
    t_new = time.perf_counter() - t0

    assert len(old_rows) == 20000
    assert new_rows.shape == (len(ys), 4)
    # Spot-check equivalence on the sampled subset.
    for i in range(0, 20000, 2000):
        got = new_rows[i]
        want = old_rows[i]
        assert np.allclose(got, want, equal_nan=True), f"row {i} mismatch: {got} vs {want}"

    assert t_new < t_old_estimated_full, (
        f"vectorized gather ({t_new:.3f}s for {len(ys)} px) is not faster than "
        f"the old scalar loop (estimated {t_old_estimated_full:.3f}s)")


def test_build_nodata_mask_2d_accepts_an_offset_band():
    """CSV export aborted on any NoData-carrying file whose bands came back as
    OffsetBand -- i.e. exactly the big/stacked images the lazy windowed reader
    exists for.

    ``_build_nodata_mask_2d`` handed the band straight to
    ``utils.build_nodata_mask``, whose first statement is
    ``np.asarray(img, dtype=np.float32)``. OffsetBand.__array__ deliberately
    raises rather than silently dropping its origin, so the export died with
    "[CSV export] Error processing <file>: OffsetBand cannot be converted to
    an array" and wrote zero rows for that file. Small images that the reader
    materialized in full were unaffected, which is why this looked like a
    "stacked images only" bug.

    The mask must come back in FULL-image coordinates with the window's own
    mask placed at the window origin -- not at (0, 0), which would mask the
    wrong pixels and silently drop real samples.
    """
    from ..raster_reader import OffsetBand
    from ..machine_learning_manager import _CsvExportWorker

    full_h, full_w = 40, 40
    x0, y0 = 6, 6
    win = np.zeros((8, 8), dtype=np.float32)
    win[2, 3] = -9999.0
    band = OffsetBand(win, x0, y0, full_h, full_w)

    mask = _CsvExportWorker._build_nodata_mask_2d(band, [-9999])

    assert mask.shape == (full_h, full_w), (
        "mask must be in full-image coordinates so it can be OR-ed with the "
        "other bands' masks and indexed by full-image point coordinates")
    assert mask.dtype == np.bool_
    assert mask[y0 + 2, x0 + 3], "the fill pixel was not masked at its true position"
    assert mask.sum() == 1, (
        f"expected exactly one masked pixel, got {mask.sum()} -- the window "
        "mask was probably written at (0,0) instead of the window origin")


def test_build_nodata_mask_2d_still_handles_plain_arrays():
    """The OffsetBand branch must not change the dense-array path."""
    from ..machine_learning_manager import _CsvExportWorker

    ch = np.zeros((12, 10), dtype=np.float32)
    ch[4, 5] = -9999.0
    mask = _CsvExportWorker._build_nodata_mask_2d(ch, [-9999])
    assert mask.shape == (12, 10)
    assert mask[4, 5] and mask.sum() == 1
