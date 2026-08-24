"""QC regression tests for "polygons disappearing after Image Editor applies
classification and others."

THE BUG: two places in the debounced polygon-loading pipeline bailed out
whenever `ProjectTab._nav_seq` -- a single GLOBAL counter bumped by ANY
navigation or refresh anywhere in the app (a different root, a different
single-image edit, a polygon-edit-finished full-grid reload via
reload_current_root, or the classification/crop/rotate/etc. refresh itself)
-- had moved since the load was scheduled:

  1. `display_image_group`'s `_fire_when_idle` (a 50ms singleShot closure)
     bailed with `if seq != self._nav_seq: return` -- its own prior comment
     called this out directly: "THE bail-out that leaves a refreshed viewer
     with no polygons: set_image() has already cleared the scene, and
     nothing else ever calls load_polygons for it." Firing this bails for
     EVERY viewer in the just-rebuilt grid, not just ones actually affected
     by whatever bumped `_nav_seq`.
  2. `_schedule_polygons_for_viewer`'s inner `go()` closure had the
     identical `expected_seq != self._nav_seq` bail, independently of (1) --
     its own prior comment: "one of only two places the load can be dropped
     without a trace."

Applying an edit (classification or otherwise) in the Image Editor with
"apply to group/all" scope refreshes the viewer via `display_image_group`,
which rebuilds every viewer in the grid from scratch (brand-new
`ImageViewer()` instances -- see `display_image_group`, `viewer = ImageViewer()`)
and schedules (1) to (re)populate their polygons ~50ms later. If ANYTHING
else in the app bumps `_nav_seq` in that window -- a keyboard nav, a slider
move, another edit elsewhere, `reload_current_root` (wired to every viewer's
`editing_finished` signal) -- the ENTIRE scheduled polygon load for that
grid was silently dropped, leaving every viewer in it with an empty scene
and nothing left to ever redraw them. This is a genuine race, not a typo:
`_nav_seq` is deliberately global (it also guards navigation itself), but
using it as the SOLE staleness signal for "is this specific grid/viewer
still current" conflated "something navigated" with "THIS grid was
replaced" -- two different questions with the same counter.

THE FIX: both bail-outs now ask the precise question instead.
  1. `_fire_when_idle` checks `self.viewer_widgets is not expected_widgets`
     (identity against the specific list THIS display_image_group call
     built, captured in the closure) -- true bailing ONLY when a NEWER
     display_image_group call has already replaced the grid, in which case
     that newer call's own `_fire_when_idle` is the one that should (and
     does) load polygons for what's actually on screen.
  2. `_schedule_polygons_for_viewer`'s `go()` drops the `_nav_seq` check
     entirely and relies solely on its OTHER, already-precise guard: does
     THIS viewer still show the SAME file it was scheduled to load for
     (`v.image_data.filepath == idata.filepath`)? That question is
     inherently per-viewer and correct regardless of what happened
     elsewhere.

Neither function is directly reachable as a bound method for AST purposes
(`_fire_when_idle` and `go` are both `<locals>` closures), so these checks
walk the FULL subtree of their enclosing methods -- `_func_tree` already
does this for any nested `def`.
"""
import ast

import pytest

from .test_export_and_ax_regressions import _func_tree, _names_in
from ..project_tab import ProjectTab

pytestmark = [pytest.mark.viewer, pytest.mark.polygons]


def _compares_navseq_with_noteq(func):
    """True if `func`'s subtree contains a `Compare` node using `!=`
    (`ast.NotEq`) where one side names `_nav_seq` -- the old bail-out shape
    (`seq != self._nav_seq` / `expected_seq != getattr(self, "_nav_seq", 0)`).
    """
    def _mentions_navseq(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == "_nav_seq":
                return True
            if isinstance(n, ast.Attribute) and n.attr == "_nav_seq":
                return True
            if isinstance(n, ast.Constant) and n.value == "_nav_seq":
                return True
        return False

    for node in ast.walk(_func_tree(func)):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, ast.NotEq) for op in node.ops):
            continue
        sides = [node.left] + list(node.comparators)
        if any(_mentions_navseq(s) for s in sides):
            return True
    return False


def _compares_viewer_widgets_by_identity(func):
    """True if `func`'s subtree contains an `is`/`is not` comparison naming
    `viewer_widgets` -- the new, precise bail-out shape."""
    for node in ast.walk(_func_tree(func)):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            continue
        sides = [node.left] + list(node.comparators)
        for s in sides:
            for n in ast.walk(s):
                if isinstance(n, ast.Attribute) and n.attr == "viewer_widgets":
                    return True
                if isinstance(n, ast.Name) and n.id == "viewer_widgets":
                    return True
    return False


# ---------------------------------------------------------------------------
# display_image_group's _fire_when_idle
# ---------------------------------------------------------------------------
def test_fire_when_idle_no_longer_bails_on_global_nav_seq():
    assert not _compares_navseq_with_noteq(ProjectTab.display_image_group), (
        "_fire_when_idle (nested in display_image_group) still bails on a "
        "global _nav_seq mismatch -- this is THE bug: any unrelated "
        "navigation/refresh anywhere in the app silently drops polygon "
        "loading for a grid it never touched")


def test_fire_when_idle_gates_on_viewer_widgets_identity():
    assert _compares_viewer_widgets_by_identity(ProjectTab.display_image_group), (
        "_fire_when_idle must gate on `self.viewer_widgets is expected_widgets` "
        "(precise: only true when a NEWER display_image_group call has "
        "already replaced this grid) instead of the removed nav_seq check")


def test_fire_when_idle_still_schedules_polygon_loads():
    """Guard against an over-fix that removes the bail-out but also breaks
    the actual scheduling loop."""
    assert "_schedule_polygons_for_viewer" in _names_in(ProjectTab.display_image_group)


# ---------------------------------------------------------------------------
# _schedule_polygons_for_viewer's go()
# ---------------------------------------------------------------------------
def test_schedule_polygons_go_no_longer_bails_on_global_nav_seq():
    assert not _compares_navseq_with_noteq(ProjectTab._schedule_polygons_for_viewer), (
        "_schedule_polygons_for_viewer's go() still bails on a global "
        "_nav_seq mismatch -- the second of 'only two places the load can "
        "be dropped without a trace'")


def test_schedule_polygons_go_still_checks_the_viewers_own_filepath():
    """The per-viewer filepath check is the ONE guard this function should
    still have -- it's precise (only bails when THIS viewer's own content
    actually changed) unlike the removed global nav_seq check."""
    names = _names_in(ProjectTab._schedule_polygons_for_viewer)
    assert "filepath" in names
    assert "image_data" in names


def test_schedule_polygons_for_viewer_signature_unchanged():
    """nav_seq stays an accepted (if now vestigial) parameter so existing
    call sites (`_fire_when_idle`, `_on_nav_idle`) don't need updating."""
    import inspect
    sig = inspect.signature(ProjectTab._schedule_polygons_for_viewer)
    assert "nav_seq" in sig.parameters


# ---------------------------------------------------------------------------
# Functional: _schedule_polygons_for_viewer actually loads polygons even
# after an unrelated _nav_seq bump -- the part of this fix that doesn't
# require driving the full display_image_group widget-creation machinery.
# ---------------------------------------------------------------------------
class _FakeImageData:
    def __init__(self, filepath):
        self.filepath = filepath


class _FakeViewer:
    """Minimal stand-in with just what go() touches: .image_data, .scene(),
    .viewportUpdateMode()/.setViewportUpdateMode(), .viewport()."""
    def __init__(self, filepath):
        self.image_data = _FakeImageData(filepath)

    def scene(self):
        return None

    def viewportUpdateMode(self):
        return None

    def setViewportUpdateMode(self, mode):
        pass


def _make_fake_tab():
    """`_schedule_polygons_for_viewer` does `QtCore.QTimer(self)` -- the
    timer's parent must be a real QObject, so the stand-in needs to be one
    too (a plain Python object raises TypeError there)."""
    from PyQt5 import QtCore

    class _FakeTab(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self._nav_seq = 0
            self.load_polygons_calls = []

        def load_polygons(self, viewer, image_data, ax_cfg=None):
            self.load_polygons_calls.append((viewer, image_data))

    return _FakeTab()


def test_schedule_polygons_for_viewer_still_loads_after_unrelated_nav_bump(qapp):
    """THE functional regression: schedule a load, bump _nav_seq (simulating
    a navigation/refresh that has nothing to do with this viewer), let the
    timer fire, and confirm load_polygons was still called. Before the fix,
    this silently did nothing."""
    from PyQt5 import QtWidgets

    tab = _make_fake_tab()
    viewer = _FakeViewer("a.tif")
    idata = _FakeImageData("a.tif")

    tab._nav_seq = 5
    ProjectTab._schedule_polygons_for_viewer(tab, viewer, idata, delay_ms=0, nav_seq=5)

    # Simulate an unrelated navigation/refresh elsewhere bumping the GLOBAL
    # counter before the scheduled timer fires.
    tab._nav_seq = 6

    # Pump the event loop long enough for the 0ms singleShot timer to fire.
    for _ in range(20):
        QtWidgets.QApplication.processEvents()

    assert tab.load_polygons_calls, (
        "load_polygons was never called -- the unrelated _nav_seq bump "
        "silently cancelled a load for a viewer it had nothing to do with")
    called_viewer, called_idata = tab.load_polygons_calls[0]
    assert called_viewer is viewer
    assert called_idata is idata


def test_schedule_polygons_for_viewer_still_skips_when_the_viewer_itself_moved_on(qapp):
    """The ONE case that must still be skipped: the viewer this load was
    scheduled for has genuinely switched to a different file by the time
    the timer fires."""
    from PyQt5 import QtWidgets

    tab = _make_fake_tab()
    viewer = _FakeViewer("a.tif")
    idata = _FakeImageData("a.tif")

    ProjectTab._schedule_polygons_for_viewer(tab, viewer, idata, delay_ms=0)

    # This SPECIFIC viewer moved on to a different file before the timer fired.
    viewer.image_data = _FakeImageData("b.tif")

    for _ in range(20):
        QtWidgets.QApplication.processEvents()

    assert not tab.load_polygons_calls, (
        "load_polygons was called for a viewer that had already switched "
        "to a different file -- the per-viewer filepath guard must still work")


# ---------------------------------------------------------------------------
# End-to-end: the ORIGINAL reported scenario, not just the isolated mechanism
# ---------------------------------------------------------------------------
#
# Everything above pins the specific `_nav_seq` bail-out at the AST level (for
# `_fire_when_idle`, which is a closure nested inside display_image_group and
# so cannot be called on its own) plus one real functional test of
# `_schedule_polygons_for_viewer` in isolation. Neither drives
# `display_image_group` itself, so neither would catch a regression specific
# to how IT wires the timer/closure up (e.g. a wrong `expected_widgets`
# capture, or the closure never being connected at all).
#
# This test drives a REAL ProjectTab through the REAL async
# load_image_group -> display_image_group pipeline (QThreadPool image
# loading, then the actual grid rebuild with real ImageViewer/QGraphicsScene
# instances), and reproduces the race by bumping `_nav_seq` for real in the
# ~50ms window before the debounced polygon load fires -- the same thing an
# unrelated keyboard nav / slider move / other-viewer edit would do in the
# running app.

def _pump_until(app, condition, timeout_s=5.0, interval_s=0.01):
    """Process Qt events + sleep in real wall-clock time until `condition()`
    is true or `timeout_s` elapses.

    QThreadPool workers run on background threads and QTimer.singleShot
    callbacks fire on the event loop -- `processEvents()` alone (with no
    sleep) never gives a background thread time to finish, and a sleep alone
    never delivers the signal it queued. Both are needed together.
    """
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(interval_s)
    app.processEvents()
    return bool(condition())


def test_display_image_group_survives_an_unrelated_nav_seq_bump_end_to_end(qapp, tmp_path):
    """THE full end-to-end repro, not just the mechanism.

    Before the fix, `_fire_when_idle` bailed the instant `self._nav_seq`
    (captured when the 50ms idle timer started) no longer matched
    `self._nav_seq` at fire time -- which this test forces to happen by
    bumping it directly, exactly as an UNRELATED navigation elsewhere in the
    app would. `viewer_widgets` itself is untouched (still the same list),
    so the scheduled polygon load for this grid must still run: nothing
    about this specific grid went stale.
    """
    from canopie.qc.project_builder import build_project_tab

    pt = build_project_tab(str(tmp_path))
    try:
        def _grid_shown():
            return bool(pt.viewer_widgets) and pt.viewer_widgets[0].get("viewer") is not None

        assert _pump_until(qapp, _grid_shown, timeout_s=8.0), (
            "the initial load_project navigation never produced a viewer grid -- "
            "cannot exercise the race without one")

        viewer = pt.viewer_widgets[0]["viewer"]
        assert not viewer.get_all_polygons(), (
            "test setup assumption violated: polygons already drawn before "
            "the race window even opened")

        # THE race: something ELSE in the app navigates while
        # display_image_group's 50ms idle timer is still pending for THIS
        # grid. viewer_widgets is untouched (same list, same objects) --
        # this bump must NOT cancel the scheduled polygon load.
        pt._nav_seq += 1

        def _polygons_drawn():
            return bool(viewer.get_all_polygons())

        assert _pump_until(qapp, _polygons_drawn, timeout_s=8.0), (
            "no polygon/point items ever appeared in the real QGraphicsScene "
            "after the unrelated _nav_seq bump -- the scheduled polygon load "
            "was silently dropped, which is the bug this whole file is about")
    finally:
        # Test-owned ProjectTab (its own QThreadPool loads / QTimers), not
        # the shared session-scoped synthetic_project -- stop its background
        # timers before it is garbage collected, mirroring
        # synthetic_project's own teardown in conftest.py.
        for attr in ("_poly_idle_timer", "_autosave_timer"):
            t = getattr(pt, attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        for t in list((getattr(pt, "_poly_load_timers", {}) or {}).values()):
            try:
                t.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The "single" scope: edit_image_viewer's OTHER refresh path
# ---------------------------------------------------------------------------
#
# edit_image_viewer calls `viewer.set_image(pm)` directly the moment the
# editor closes (BEFORE any scope dispatch), and set_image() always calls
# clear_polygons() -- so the scene is emptied for every scope, not just
# group/all. For "single" scope (the DEFAULT -- apply_all_groups_checkbox /
# global_mods_checkbox are both opt-in, see image_editor_dialog.py's
# save_modifications_to_file), the only thing that ever redraws them is
# refresh_single_viewer's own load_polygons() call at the very end -- fully
# synchronous, no timer, no `_nav_seq`, so no race of the kind above. Nothing
# else in this file drove that path, so this confirms it actually holds with
# a real ImageViewer/QGraphicsScene and the project's real polygons.

def test_refresh_single_viewer_restores_polygons_after_apply(synthetic_project, viewer_factory):
    """The 'single' scope counterpart to the end-to-end test above."""
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    # This fixture has real point/polygon groups seeded by
    # project_builder.build_project_data -- confirm at least one exists for
    # this file before trusting the "restored" assertion below (otherwise an
    # empty scene before AND after would pass vacuously).
    has_polygon_for_fp = any(
        fp in group for group in synthetic_project.all_polygons.values())
    assert has_polygon_for_fp, (
        "test fixture assumption violated: no seeded polygon for this file")

    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    assert imgd is not None and imgd.image is not None
    viewer.image_data = imgd
    rec = {"viewer": viewer, "image_data": imgd}
    synthetic_project.viewer_widgets = list(
        getattr(synthetic_project, "viewer_widgets", []) or []) + [rec]

    try:
        assert not viewer.get_all_polygons(), (
            "viewer_factory() viewer must start with an empty scene, or this "
            "test cannot tell 'restored' from 'never cleared'")

        synthetic_project.refresh_single_viewer(
            viewer, reapply_mods=True, preserve_view=True)

        assert viewer.get_all_polygons(), (
            "refresh_single_viewer's own set_image() call cleared the scene "
            "and nothing redrew the polygons afterwards -- the 'single' "
            "scope path of the polygons-vanishing bug, and the DEFAULT scope "
            "for Apply All Changes")
    finally:
        try:
            synthetic_project.viewer_widgets.remove(rec)
        except Exception:
            pass
