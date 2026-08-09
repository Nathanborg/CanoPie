"""
The polygon manager, and the two "clean" delete paths, at thousands of groups.

THE BUGS THIS PINS (reported as "opening the polygon manager takes forever with
thousands of polygons", "clean root polys is very slow", "delete all polygons is
very slow"):

1. PER-ROW SELECTION ON A REALISED WIDGET.
   `update_selection_based_on_root` called `item.setSelected(True)` once per
   row, and `set_polygons` rebuilt the list row by row with painting live.
   Each call makes the view recompute and repaint its selection. Measured on a
   SHOWN dialog with 2333 groups:

       rebuild + select, per-row : 1.63 s
       same work, batched        : 0.036 s   (45x)

   An isolated (never-shown) QListWidget does this in 0.013 s, which is why the
   cost is invisible in a naive benchmark -- the widget has to be realised for
   the layout/paint work to happen at all. This runs on every navigation, every
   save and after every bulk delete, not only when the dialog is opened.

2. THE "CLEAN" PATHS WERE NEVER BATCHED.
   `clean_all_polygons` ("clean root") and the per-viewer clean push one
   DeletePolygonCommand per polygon, and QUndoStack.push() runs redo()
   immediately -- so each one performed a full scene scan AND a full
   polygon-manager rebuild. That is the same O(N^2)-plus-O(N)-rebuilds problem
   already fixed for delete_selected_polygons; these two call sites were simply
   missed.

3. THEY READ EVERY POLYGON FILE TWICE.
   Both built a QPolygonF from `poly_data['points']` for undo AND read the
   polygon's sidecar JSON. Since polygons are now loaded lazily
   (polygon_lod.LazyPolygonRecord), touching 'points' pages the full file in --
   so each polygon was read once for the undo snapshot and again to
   materialise. The sidecar alone is everything undo needs, so the in-memory
   geometry is now only consulted when there is no file.
"""
import inspect
import textwrap

import pytest
from PyQt5 import QtCore, QtWidgets

pytestmark = [pytest.mark.polygons, pytest.mark.perf]


def _manager(qapp, n_groups, shown=True):
    """A PolygonManager with a REALISED list widget -- see the module docstring
    for why an unshown widget hides the cost entirely."""
    from ..polygon_manager import PolygonManager

    pm = PolygonManager.__new__(PolygonManager)
    QtWidgets.QDialog.__init__(pm)
    pm.list_widget = QtWidgets.QListWidget()
    pm.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    lay = QtWidgets.QVBoxLayout(pm)
    lay.addWidget(pm.list_widget)
    pm.current_root = "R1"
    pm.current_root_filepaths = {"img.tif"}
    pm._cached_group_names = None
    pm._groups_by_filepath = {"img.tif": {f"g_{i:05d}" for i in range(n_groups)}}

    # set_polygons invalidates the reverse index, which is then rebuilt from
    # parent().all_polygons -- so the stub parent has to carry it, or every
    # group looks like it belongs to no root and nothing gets selected.
    class _Parent:
        all_polygons = {f"g_{i:05d}": {"img.tif": {}} for i in range(n_groups)}

    _p = _Parent()
    pm.parent = lambda: _p

    if shown:
        pm.show()
        qapp.processEvents()
    return pm


# ---------------------------------------------------------------------------
# 1. Batched rebuild + range selection
# ---------------------------------------------------------------------------
def test_select_rows_fast_selects_exactly_the_requested_rows(qapp):
    pm = _manager(qapp, 200)
    try:
        for g in (f"g_{i:05d}" for i in range(200)):
            it = QtWidgets.QListWidgetItem(g)
            it.setData(QtCore.Qt.UserRole, g)
            pm.list_widget.addItem(it)

        wanted = [0, 1, 2, 7, 8, 50, 199]          # deliberately non-contiguous
        pm._select_rows_fast(wanted)
        got = sorted(pm.list_widget.row(i) for i in pm.list_widget.selectedItems())
        assert got == wanted, (
            f"range-collapsing selected {got}, expected {wanted} -- the "
            "contiguous-run grouping is wrong")
    finally:
        pm.close()


def test_select_rows_fast_handles_empty_and_full(qapp):
    pm = _manager(qapp, 50)
    try:
        for g in (f"g_{i:05d}" for i in range(50)):
            it = QtWidgets.QListWidgetItem(g)
            it.setData(QtCore.Qt.UserRole, g)
            pm.list_widget.addItem(it)

        pm._select_rows_fast([])
        assert pm.list_widget.selectedItems() == []
        pm._select_rows_fast(list(range(50)))
        assert len(pm.list_widget.selectedItems()) == 50
        pm._select_rows_fast([])
        assert pm.list_widget.selectedItems() == [], (
            "a subsequent empty selection did not clear the previous one")
    finally:
        pm.close()


def test_rebuild_suppresses_updates_and_signals():
    """Static guard: the row-by-row rebuild must not paint per row."""
    from ..polygon_manager import PolygonManager

    src = textwrap.dedent(inspect.getsource(PolygonManager.set_polygons))
    assert "setUpdatesEnabled(False)" in src and "blockSignals(True)" in src, (
        "set_polygons rebuilds the list with painting and signals live; on a "
        "realised dialog with 2333 groups that is 1.63 s instead of 0.036 s")
    assert "setUpdatesEnabled(True)" in src and "blockSignals(False)" in src, (
        "updates/signals are disabled but never restored")


def test_selection_does_not_call_setselected_per_row():
    from ..polygon_manager import PolygonManager

    src = textwrap.dedent(inspect.getsource(PolygonManager.update_selection_based_on_root))
    assert "_select_rows_fast" in src, (
        "update_selection_based_on_root still selects row by row")
    assert "setSelected(True)" not in src, (
        "a per-row setSelected loop remains -- that is the 45x cost")


@pytest.mark.slow
def test_manager_rebuild_stays_fast_at_scale(qapp):
    """Behavioural ceiling. Deliberately generous: the point is to catch a
    return to per-row painting (seconds), not to police milliseconds."""
    import time

    n = 2000
    pm = _manager(qapp, n)
    try:
        groups = {f"g_{i:05d}": {"img.tif": {}} for i in range(n)}
        t0 = time.perf_counter()
        pm.set_polygons(groups)
        qapp.processEvents()
        elapsed = time.perf_counter() - t0

        assert pm.list_widget.count() == n
        assert len(pm.list_widget.selectedItems()) == n, (
            "current-root groups were not all selected")
        assert elapsed < 3.0, (
            f"rebuilding the manager for {n} groups took {elapsed:.2f}s; "
            "per-row painting/selection has returned")
    finally:
        pm.close()


# ---------------------------------------------------------------------------
# 2. The clean paths must be batched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn_name", ["clean_all_polygons", "delete_polygons_for_viewer"])
def test_clean_paths_use_the_bulk_delete_context(fn_name):
    """Each pushed DeletePolygonCommand runs redo() immediately; without the
    bulk context that is one scene scan and one manager rebuild PER POLYGON."""
    from ..project_tab import ProjectTab

    fn = getattr(ProjectTab, fn_name, None)
    if fn is None:
        pytest.skip(f"{fn_name} not present in this build")
    src = textwrap.dedent(inspect.getsource(fn))
    assert "_bulk_polygon_delete" in src, (
        f"{fn_name} pushes deletions without _bulk_polygon_delete, so cleaning "
        "a root of thousands of polygons is O(N^2) in the scene plus N full "
        "polygon-manager rebuilds")


@pytest.mark.parametrize("fn_name", ["clean_all_polygons", "delete_polygons_for_viewer"])
def test_clean_paths_do_not_read_any_polygon_file(fn_name):
    """THE clean-root cost, at the mechanism level.

    An earlier revision of this test asserted the OPPOSITE -- that these paths
    must read the sidecar (`f.read()`) to snapshot it for undo. That was the
    slow design: reading costs a full file open per polygon, ~28 ms each under
    Windows AV scanning, i.e. ~115 s for the 4111 sidecars in a real project,
    all before a single file is deleted.

    Reading is also unnecessary. `DeletePolygonCommand.redo()` snapshots by
    MOVING the sidecar into polygons/.trash/ -- a metadata rename that
    preserves the exact bytes -- but that path only arms when `json_content`
    is None. So these callers must hand over NEITHER the content nor `points`
    (which, on a lazily loaded polygon, would materialise the record and
    re-read the very file being avoided).
    """
    from ..project_tab import ProjectTab

    fn = getattr(ProjectTab, fn_name, None)
    if fn is None:
        pytest.skip(f"{fn_name} not present in this build")
    src = textwrap.dedent(inspect.getsource(fn))

    assert "f.read()" not in src, (
        f"{fn_name} reads every polygon sidecar to snapshot it for undo. That "
        "is ~28 ms per file on Windows (~115 s for 4111 polygons) and is "
        "redundant -- redo() preserves the file by renaming it into .trash")
    assert 'poly_data.get("points"' not in src, (
        f"{fn_name} touches `points`, which materialises every lazily loaded "
        "polygon and re-reads the file the rename exists to avoid")
    assert "json_content=None" in src, (
        f"{fn_name} must pass json_content=None so redo()'s trash-rename fast "
        "path arms (see DeletePolygonCommand.__init__)")


def test_undo_batches_under_the_bulk_context():
    """Undo replays one command per polygon, exactly as redo does, so it needs
    the same escape hatch.

    Without it, undoing a 4111-polygon "Clean Root" runs 4111 polygon-manager
    rebuilds AND 4111 viewer refreshes -- and because the clean paths no longer
    carry `item_points`, each refresh is a full load_polygons() reload. Undo
    would then be far slower than the delete it reverses.
    """
    from ..project_tab import DeletePolygonCommand

    src = textwrap.dedent(inspect.getsource(DeletePolygonCommand.undo))
    assert "_bulk_delete_depth" in src, (
        "DeletePolygonCommand.undo has no bulk-batching guard, so undoing a "
        "large delete does one manager rebuild and one viewer reload per "
        "polygon")
    assert "_bulk_reload_viewers" in src, (
        "undo must record viewers for a RELOAD, not reuse redo's "
        "label-removal set -- that set removes items, which would delete "
        "exactly the polygons undo just restored")
