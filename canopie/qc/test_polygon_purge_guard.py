"""
`update_all_polygons` may only infer "the user deleted this" for polygons the
viewer's last load actually tried to draw.

THE INCIDENT: on C:/New Folder185 each refresh loaded one subset of the
polygons; the purge then removed the OTHER subset from memory, the save moved
their sidecars to polygons/.trash/, and the overview was rewritten from the
survivors -- so the next refresh flipped which half survived. The directory
went 6824 -> 3507 files with 7012 in .trash. Nothing in the suite covered it.
"""
import pytest

from ..project_tab import ProjectTab


class _Item:
    def __init__(self, name):
        self.name = name


class _Viewer:
    def __init__(self, items, loaded=None):
        self._items = items
        if loaded is not None:
            self._loaded_polygon_names = set(loaded)

    def get_all_polygons(self):
        return list(self._items)


FP = "C:/img/ortho_cog.tif"


def _tab(all_polygons):
    pt = ProjectTab.__new__(ProjectTab)
    pt.all_polygons = all_polygons
    pt._poly_exact_index = {FP: set(all_polygons)}
    pt._poly_norm_index = {}
    return pt


def _purge(tab, viewer):
    """Run just the prune step the way update_all_polygons does."""
    present = {getattr(it, "name", None) for it in viewer.get_all_polygons()}
    loadable = getattr(viewer, "_loaded_polygon_names", None)
    for group in list(tab._poly_exact_index.get(FP, ())):
        if loadable is None or group not in loadable:
            continue
        if group not in present:
            tab.all_polygons.pop(group, None)
    return tab.all_polygons


def test_a_partial_load_never_purges_what_it_did_not_draw():
    """The incident, reduced."""
    mem = {f"imported_{i}": {FP: {}} for i in range(5)}
    mem.update({f"drawn_{i}": {FP: {}} for i in range(3)})
    tab = _tab(dict(mem))

    # A refresh that drew only the imported ones.
    drew = [f"imported_{i}" for i in range(5)]
    viewer = _Viewer([_Item(n) for n in drew], loaded=drew)

    survived = _purge(tab, viewer)
    for i in range(3):
        assert f"drawn_{i}" in survived, (
            "a polygon this load never attempted to draw was treated as "
            "user-deleted -- this is what trashed 3318 sidecars")
    assert len(survived) == 8


def test_a_genuine_deletion_is_still_purged():
    """The guard must not disable the feature it protects.

    The purge exists so 'clear polygons' does not come back after a reload.
    """
    mem = {f"g{i}": {FP: {}} for i in range(4)}
    tab = _tab(dict(mem))

    # The load drew all four; the user then deleted g2 in the viewer.
    viewer = _Viewer([_Item(n) for n in ("g0", "g1", "g3")],
                     loaded=("g0", "g1", "g2", "g3"))

    survived = _purge(tab, viewer)
    assert "g2" not in survived, "a genuinely deleted polygon must still be purged"
    assert set(survived) == {"g0", "g1", "g3"}


def test_unknown_load_state_purges_nothing():
    """A viewer that never recorded a load cannot be reasoned about."""
    mem = {f"g{i}": {FP: {}} for i in range(4)}
    tab = _tab(dict(mem))
    viewer = _Viewer([])          # no _loaded_polygon_names at all
    survived = _purge(tab, viewer)
    assert len(survived) == 4, (
        "with no record of what was loaded, an empty scene must not be read "
        "as 'the user deleted everything'")


def test_load_polygons_records_what_it_drew():
    """The guard is only as good as the record load_polygons leaves."""
    import inspect
    src = inspect.getsource(ProjectTab.load_polygons)
    assert "_loaded_polygon_names" in src, (
        "load_polygons no longer records which polygons it drew, so the purge "
        "guard in update_all_polygons has nothing to check against")


def test_update_all_polygons_consults_the_record():
    """NOTE ON WHAT THE TESTS ABOVE DO AND DO NOT PROVE.

    `_purge` in this file is a MODEL of the prune step, not the real one --
    driving update_all_polygons needs a live viewer, image_data, an .ax and a
    populated index. So those tests document the intended semantics; they
    cannot detect the guard being removed from the real function.

    This one can, and is deliberately literal about the guard it requires.
    """
    import inspect
    src = inspect.getsource(ProjectTab.update_all_polygons)
    assert 'loadable = getattr(viewer, "_loaded_polygon_names", None)' in src, (
        "update_all_polygons no longer reads the load record")
    assert "if loadable is None or group_name not in loadable:" in src, (
        "the purge guard is gone: update_all_polygons will again infer "
        "'user deleted this' for polygons the load never attempted to draw, "
        "which is what moved 3318 sidecars to polygons/.trash/")


# ---------------------------------------------------------------------------
# The load record must not outlive the scene it describes
# ---------------------------------------------------------------------------
def test_clearing_the_scene_invalidates_the_load_record(qapp):
    """THE refresh bug, reproduced at the viewer level.

    set_image() calls clear_polygons() on every refresh. If the record of
    "what the last load drew" survives that, then for a moment the scene is
    empty while the record still names every polygon -- and update_all_polygons
    reads all of them as user deletions.

    Observed on a fresh project (New Folder190): import 3504 crowns, draw one
    polygon, refresh -> the 3504 vanished and only the drawn one remained. The
    drawn one survived precisely because it was NOT in the stale record.
    """
    from PyQt5 import QtGui, QtCore
    from ..image_viewer import ImageViewer

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 400)
    pm.fill(QtCore.Qt.darkGreen)
    v.set_image(pm)
    v.polygons = []
    for i in range(5):
        v.add_polygon_to_scene(
            QtGui.QPolygonF([QtCore.QPointF(10 + i, 10), QtCore.QPointF(20 + i, 10),
                             QtCore.QPointF(20 + i, 20)]), name=f"crown_{i}")

    # ProjectTab.load_polygons stamps this after drawing.
    v._loaded_polygon_names = {f"crown_{i}" for i in range(5)}

    v.clear_polygons()
    assert getattr(v, "_loaded_polygon_names", "MISSING") is None, (
        "the load record survived clear_polygons -- update_all_polygons will "
        "now read every name in it as a user deletion, because the scene it "
        "described has been emptied")


def test_set_image_invalidates_the_load_record(qapp):
    """set_image is the path a refresh actually takes."""
    from PyQt5 import QtGui, QtCore
    from ..image_viewer import ImageViewer

    v = ImageViewer()
    pm = QtGui.QPixmap(400, 400)
    pm.fill(QtCore.Qt.darkGreen)
    v.set_image(pm)
    v._loaded_polygon_names = {"crown_0", "crown_1"}

    v.set_image(pm)          # a refresh
    assert getattr(v, "_loaded_polygon_names", "MISSING") is None, (
        "refreshing left a stale load record behind")
