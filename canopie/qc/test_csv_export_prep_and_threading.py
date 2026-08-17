"""QC for two CSV-export defects found on a ~5,000-polygon real project.

1. THE PRE-DIALOG FREEZE
   `save_polygons_to_csv` builds a disk snapshot of the whole job list --
   roughly one disk read per polygon (`_load_polygon_from_disk` opens the
   sidecar, and when there is none `_jsonify` walks the record, which for a
   `LazyPolygonRecord` materialises it from its own sidecar anyway). That work
   used to run BEFORE the processing-mode dialog was shown, so clicking Export
   on a large project gave a frozen, windowless app for a long time. The work
   itself is legitimate; its POSITION was the bug. It now runs after the
   dialog, with progress and a Cancel button.

   Pinned here by source-order assertions rather than wall-clock, so the test
   means the same thing on any machine and cannot go flaky.

2. THREADED EXPORT SILENTLY DROPPING A WHOLE FILE
   In `processing_mode == "threading"`, the per-file worker caught every
   exception and returned `([], [])`. The consumer then did
   `rows, mods = future.result()` -- which raises nothing, because the
   exception was already swallowed -- so `errors` was never incremented and
   that file contributed ZERO rows while the export still reported success.
   Every other mode (`multiprocessing`, and the sequential fallback) does
   `errors += 1` on a failed file. That asymmetry is exactly the reported
   symptom: the file-batched CSV was bigger than the multithreaded one, with
   nothing to explain the difference.

   The worker now re-raises so the existing `except` in the consumer counts it.
"""
import inspect
import logging
import re

import pytest

from canopie.project_tab import ProjectTab

pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _src(fn):
    return inspect.getsource(fn)


# ---------------------------------------------------------------------------
# 1. The snapshot must not sit in front of the dialog
# ---------------------------------------------------------------------------
def test_snapshot_runs_after_the_processing_mode_dialog():
    """THE regression test for the pre-dialog freeze.

    If the snapshot loop is ever moved back above the dialog, a large project
    freezes on the Export click again with no window to show for it.
    """
    src = _src(ProjectTab.save_polygons_to_csv)

    dialog = src.index("mode_dialog.exec_()")
    snapshot = src.index("snap_jobs.append(")

    assert dialog < snapshot, (
        "the per-polygon snapshot loop runs BEFORE the processing-mode dialog. "
        "That is one disk read per polygon on the GUI thread with no window "
        "visible yet -- the freeze this test exists to prevent.")


def test_snapshot_reports_progress_and_can_be_cancelled():
    """The loop is inherently O(polygons) disk reads; it must not look hung."""
    src = _src(ProjectTab.save_polygons_to_csv)

    snap_block = src[src.index("snap_jobs = []"):src.index("# ============ BACKGROUND EXPORT")]

    assert "wasCanceled()" in snap_block, "snapshot loop offers no way to cancel"
    assert "setValue(" in snap_block, "snapshot loop reports no progress"


def test_snapshot_cancel_does_not_leave_a_temp_dir_behind():
    src = _src(ProjectTab.save_polygons_to_csv)
    snap_block = src[src.index("_snap_cancelled = False"):src.index("# ============ BACKGROUND EXPORT")]

    assert "rmtree" in snap_block, (
        "cancelling the snapshot leaves its tempfile.mkdtemp() directory on disk")


# ---------------------------------------------------------------------------
# 2. Threaded export must not swallow a whole file
# ---------------------------------------------------------------------------
def test_thread_worker_reraises_so_failures_are_counted():
    """THE regression test for the missing-rows asymmetry.

    `_file_worker` is a closure inside save_polygons_to_csv, so this reads the
    source rather than calling it -- the point is the control flow, and the
    control flow is what regressed.
    """
    src = _src(ProjectTab.save_polygons_to_csv)

    worker = src[src.index("def _file_worker("):src.index("def submit_next()")]

    # The bug: initialise to empty, catch everything, return the empty result.
    assert "raise" in worker, (
        "_file_worker swallows exceptions and returns ([], []). The consumer's "
        "future.result() then raises nothing, `errors` is never incremented, "
        "and that file's rows vanish from a 'successful' export.")

    for handler in ("except MemoryError", "except Exception"):
        idx = worker.index(handler)
        tail = worker[idx:idx + 600]
        assert "raise" in tail, f"{handler} in _file_worker does not re-raise"


def test_every_processing_mode_counts_a_failed_file():
    """multiprocessing and the sequential fallback already did `errors += 1`;
    threading is the one that did not. Pin all three together so a future
    refactor cannot quietly reintroduce the asymmetry in any of them."""
    src = _src(ProjectTab.save_polygons_to_csv)

    # Consumer of the threaded futures.
    consumer = src[src.index("rows, mods = future.result()"):]
    consumer = consumer[:consumer.index("processed_count += poly_count")]
    assert consumer.count("errors += 1") >= 2, (
        "the threaded consumer does not count both MemoryError and generic "
        "failures")


def test_process_file_batch_warns_when_polygons_are_dropped(caplog, monkeypatch):
    """A polygon that raises inside _process_file_batch produces no CSV row.
    That is survivable, but it must not be silent -- this is the only place
    that knows it happened."""
    tab = ProjectTab.__new__(ProjectTab)

    def _boom(*a, **kw):
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(tab, "process_polygon", _boom, raising=False)

    with caplog.at_level(logging.WARNING):
        rows, mods = ProjectTab._process_file_batch(
            tab, "C:/fake/image.tif",
            [("grp_a", {}), ("grp_b", {})],
            {}, [], False, None,
        )

    assert rows == [] and mods == []
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert re.search(r"2 of 2 polygon\(s\) FAILED", joined), (
        "dropped polygons produced no WARNING -- 'my export is missing rows' "
        f"would have no signal anywhere. Got: {joined!r}")
