"""
Shared pytest fixtures for the QC suite.

Note on QT_QPA_PLATFORM: this module deliberately does NOT force
QT_QPA_PLATFORM=offscreen. Every real-QApplication script used to develop and
verify this codebase's PyQt widgets during this session (ImageViewer,
ProjectTab, MachineLearningManager, ...) already ran headlessly without it in
this environment, so forcing a platform override here would be an untested
behavior change, not a fix for an actual problem. If a future environment
needs it, set the env var before the test process starts rather than here.
"""
import os

import pytest
from PyQt5 import QtWidgets

from .. import performance as performance_module

# Disable NumExpr/Numba acceleration for this test session, BEFORE any test
# imports/uses them via _calc_stats/FastBandMathEngine. NumExpr in particular
# keeps its own persistent internal thread pool; empirically, that pool (or
# Numba's JIT-compilation thread) appears to be the actual source of the
# native-level "Windows fatal exception: access violation" crash chased
# above (flaky, ~1-in-3 runs, stack traces rooted in
# concurrent.futures.thread._worker) -- disabling both eliminates it. This
# only affects THIS test process; the real app still uses whichever
# accelerators are installed. Basic stats (fast_stats: mean/median/std/
# quantiles) never used numba/numexpr in the first place (confirmed by
# direct read of performance.py) -- only band-math expression evaluation and
# sklearn prediction acceleration do, so this doesn't change what pixel
# values/stats the suite is actually verifying.
performance_module.USE_NUMBA = False
performance_module.USE_NUMEXPR = False

from ._report import QCReporter as _QCReporter
from .generate_fixtures import generate_all
from .project_builder import build_project_tab
from .. import project_tab as project_tab_module


# ---------------------------------------------------------------------------
# KNOWN LIMITATION: a residual native-crash flakiness in THIS dev environment
# ---------------------------------------------------------------------------
# Despite the QTimer/viewer/executor cleanup in this file (and disabling
# NumExpr/Numba above, which measurably helped), this suite can still hit a
# genuine native-level RACE CONDITION on this environment (Windows + Python
# 3.12 + PyQt5 5.15 + OpenCV) -- empirically anywhere from ~0% to ~100% of
# runs depending on what else was tried (gc.disable() made it dramatically
# WORSE, ruling out "GC firing mid-test" as the cause; disabling NumExpr/
# Numba measurably helped, implicating their internal thread pools /
# JIT-compilation threads as at least A contributor). It manifests as
# "Windows fatal exception: access violation" and exit code 139, with
# faulthandler stack traces that are misleading -- they show whatever the
# MAIN thread was doing at the moment (usually mundane pytest internals),
# not the actual native thread that faulted, and it can strike WHILE the
# last test is still finishing (inside pytest's runtestprotocol), not only
# at final interpreter shutdown as first suspected -- so pytest_unconfigure's
# os._exit() below is a best-effort safety net for the shutdown-only cases,
# not a complete fix.
#
# Crucially: in every single observed crash, all test results were already
# correctly computed and printed (every PASSED/FAILED line, the full
# "[100%]" progress, and the "N passed" summary) before the fault -- this
# never corrupts or hides a result, it only sometimes makes the process's
# own exit code unreliable in automation. If you see this, the actual
# per-test pass/fail lines above the crash dump are still the ground truth;
# re-running is a reasonable workaround since it's non-deterministic.
_exit_status = {"code": 0}

# Feature-grouped end-of-run report (see _report.py). Built incrementally from
# each test's result so it survives even when the native crash truncates
# pytest's own terminal summary.
_reporter = _QCReporter(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "last_run_report.txt"))


def pytest_runtest_logreport(report):
    _reporter.record(report)


def pytest_sessionfinish(session, exitstatus):
    _exit_status["code"] = int(exitstatus)
    text = _reporter.write()
    # Print directly rather than via the terminal reporter: this runs before
    # pytest's own summary, and going straight to the stream means the report
    # is already flushed if the process dies during finalization.
    try:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write_line("")
            for line in text.splitlines():
                tr.write_line(line)
    except Exception:
        print("\n" + text)


def pytest_unconfigure(config):
    os._exit(_exit_status["code"])


@pytest.fixture(scope="session")
def qapp():
    """Exactly one QApplication for the whole test process -- PyQt allows
    only one per process, so this must be session-scoped and reused."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["canopie-qc"])
    yield app


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Neutralize every blocking modal dialog for the whole suite.

    THIS IS NOT COSMETIC -- it fixes a real hang. QMessageBox.information/
    warning/critical are static convenience methods that internally construct
    a QMessageBox and call exec_(), which blocks until a human clicks a
    button. CanoPie calls these on ordinary success paths, not just errors --
    e.g. MachineLearningManager.export_csv_data ends with
    `QMessageBox.information(self, "Export Complete", ...)`
    (machine_learning_manager.py:3735) and ProjectTab.save_project does the
    same. A test that drives those functions with no human present hangs
    forever on that line, with every assertion already satisfied and no
    output to explain why.

    This presented as a maddening intermittent "some tests pass, then one
    hangs" (and, under `timeout`, as a mysterious exit 124) -- intermittent
    only because whether a modal blocks forever depends on focus/window-
    manager races on Windows, not on anything in the test.

    QInputDialog.getItem / QFileDialog.getSaveFileName are deliberately NOT
    patched here: individual tests patch those per-call to feed specific
    return values (that IS the test input). This fixture only covers the
    fire-and-forget notification dialogs, which no test needs to inspect.
    """
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok), raising=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok), raising=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok), raising=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes), raising=False)
    # QProgressDialog is non-modal in CanoPie's usage (never exec_'d), but its
    # repeated setValue()/processEvents() churn is pure overhead in tests.
    monkeypatch.setattr(QtWidgets.QProgressDialog, "setValue",
                        lambda self, v: None, raising=False)


@pytest.fixture(scope="session", autouse=True)
def fixtures_ready():
    """Regenerates only whatever fixture images/ground-truth are missing --
    fast no-op on repeat runs. Runs before every test via autouse."""
    generate_all(force=False)


@pytest.fixture(scope="session", autouse=True)
def _shutdown_perf_executors():
    """process_polygon's stats/band-math delegate into performance.py, which
    lazily spins up module-level executors (_cpu_executor/_io_executor) on
    first use and never tears them down on their own. Left running, their
    worker threads are still alive at Python interpreter shutdown and crash
    there (observed: exit code 139, "Windows fatal exception: access
    violation", stack traces rooted in concurrent.futures.thread._worker)
    once enough of this suite's tests have exercised process_polygon to spin
    the pools up.

    performance.shutdown_executors() exists for exactly this but calls
    .shutdown(wait=False) -- it returns immediately without actually joining
    the worker threads, so by the time the interpreter starts finalizing,
    the threads can still be mid-teardown. Shut the executors down directly
    with wait=True instead, so this fixture's teardown doesn't return until
    they're genuinely gone."""
    yield
    try:
        from .. import performance
        if getattr(performance, "_io_executor", None) is not None:
            performance._io_executor.shutdown(wait=True)
            performance._io_executor = None
        if getattr(performance, "_cpu_executor", None) is not None:
            performance._cpu_executor.shutdown(wait=True)
            performance._cpu_executor = None
    except Exception:
        pass


@pytest.fixture(scope="session")
def synthetic_project(qapp, fixtures_ready, tmp_path_factory):
    """A single real ProjectTab, loaded once for the whole session against
    the synthetic project (all 8 fixtures as separate roots). Session-scoped
    for speed -- reused read-only across tests, which also naturally warms
    ProjectTab's own _export_image_cache between tests touching the same
    fixture."""
    project_folder = tmp_path_factory.mktemp("qc_project")
    pt = build_project_tab(str(project_folder))
    yield pt
    # Stop ProjectTab's background QTimers before interpreter shutdown --
    # otherwise Qt/sip can tear down C++ objects in an order that trips a
    # native-level exception during Python exit (harmless -- pytest still
    # exits 0, since it happens strictly after results are reported -- but
    # quieter this way).
    for attr in ("_poly_idle_timer", "_autosave_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass


@pytest.fixture()
def viewer_factory(qapp):
    """Tracks every ImageViewer a test creates and explicitly closes/deletes
    them at teardown (close(), setParent(None), deleteLater(), then
    processEvents()) before the next test or interpreter shutdown runs.

    Leaving ImageViewer instances to be garbage-collected in whatever order
    Python happens to pick -- rather than explicitly torn down while the
    QApplication is still fully alive -- was observed to segfault (exit code
    139, not just the benign post-report shutdown quirk seen with a single
    QApplication/ProjectTab) once more than a handful of viewers existed in
    one process; each ImageViewer owns a QGraphicsScene plus the zoom/band
    overlay bars' deferred QTimer.singleShot(0, ...) calls and a viewport
    event filter, which is much more native-side state per instance than a
    single ProjectTab, and pytest's parametrization instantiates one viewer
    per fixture per test.
    """
    created = []

    def _make():
        from canopie.image_viewer import ImageViewer
        v = ImageViewer()
        created.append(v)
        # ImageViewer schedules its zoom/band overlay bars' setup via
        # QTimer.singleShot(0, ...) rather than attaching them synchronously
        # in __init__ (see attach_zoom_bar/attach_band_bar). Flush the event
        # queue now so that setup runs deterministically while this viewer is
        # still fully alive, instead of being left pending and potentially
        # firing later -- e.g. during this fixture's own teardown, after a
        # DIFFERENT already-closed/deleteLater'd viewer's turn -- which is a
        # plausible source of the native-level crash this fixture exists to
        # avoid (a pending singleShot touching a half-destroyed C++ object).
        qapp.processEvents()
        return v

    yield _make

    for v in created:
        try:
            v.close()
        except Exception:
            pass
        try:
            v.setParent(None)
        except Exception:
            pass
        try:
            v.deleteLater()
        except Exception:
            pass
    qapp.processEvents()
    qapp.processEvents()


@pytest.fixture()
def ml_manager_factory(qapp):
    """Same pattern as viewer_factory, for MachineLearningManager -- also a
    QDialog with its own toolbar/actions/QListWidget. Leaving several of
    these un-closed across a parametrized test file was observed to cause
    escalating slowness culminating in an apparent hang (in practice: not a
    true infinite loop, but slow enough per-instance -- likely sklearn-
    related import/UI-construction overhead compounding -- that a whole
    parametrized file could exceed even a generous timeout)."""
    created = []

    def _make(parent_tab):
        from canopie.machine_learning_manager import MachineLearningManager
        mgr = MachineLearningManager(parent_tab)
        created.append(mgr)
        qapp.processEvents()
        return mgr

    yield _make

    for mgr in created:
        try:
            mgr.close()
        except Exception:
            pass
        try:
            mgr.setParent(None)
        except Exception:
            pass
        try:
            mgr.deleteLater()
        except Exception:
            pass
    qapp.processEvents()
    qapp.processEvents()


@pytest.fixture()
def force_lazy_export(monkeypatch):
    """Routes ProjectTab._get_export_image (and therefore process_polygon /
    MachineLearningManager.export_csv_data / train_models) through the
    windowed raster_reader.LazyChannels path, regardless of actual file size."""
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)


@pytest.fixture()
def force_lazy_preview_no_decimation(monkeypatch):
    """Routes ImageViewer/Inspect's display-load path (_imagedata_or_fallback)
    through the raster_reader-backed preview branch, while keeping step
    resolved to 1 (no spatial decimation) -- isolates "the lazy mechanism was
    used" from "pixels were decimated".

    _PREVIEW_LOAD_THRESHOLD_BYTES double-duties as BOTH the trigger
    ("bytes_if_fully_loaded > threshold" engages the preview branch at all)
    AND the OUTPUT byte budget handed to level_for_bytes/step_for_bytes --
    setting it to something tiny (e.g. 1) forces both at once. 200_000 sits
    comfortably below the 200-band fixture's ~800KB full size (triggers the
    preview branch) while still covering a 3-band 32x32 float32 preview
    output (~12KB, get_display_bands' default band count) with room to
    spare, so step resolves to 1."""
    monkeypatch.setattr(project_tab_module, "_PREVIEW_LOAD_THRESHOLD_BYTES", 200_000)


@pytest.fixture()
def force_preview_decimation(monkeypatch):
    """Shrinks the OUTPUT byte budget enough that TiledRasterReader.step_for_bytes
    resolves to step=2 for the 200-band fixture's default 3-band preview --
    empirically confirmed (4000 bytes -> (16,16,3) from a (32,32,3) full
    preview, scale=2.0) -- used by the one test that specifically wants to
    exercise spatial decimation, as opposed to force_lazy_preview_no_decimation's
    step=1 case."""
    monkeypatch.setattr(project_tab_module, "_PREVIEW_LOAD_THRESHOLD_BYTES", 4000)
