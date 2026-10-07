"""
A polygon item is identified by its GROUP key, not by its label (BUG-017).

THE BUG THIS PINS
-----------------
A shapefile crown lives in group "oleifera_7114" and its record's `name` -- the
label -- is "7114" (or the species, "oleifera", when that was the chosen name
field). `load_polygons` named scene items by that label, while the shapefile
importer, every delete path (`DeletePolygonCommand`, the bulk-delete flush),
`update_all_polygons`, `on_polygon_modified`, select / zoom-to and "Edit
properties" all treat `item.name` as the group key. So after any refresh or
reopen, measured on the real projects C:\\Nova pasta (5) and (6):

  * Polygon Manager -> Delete removed the record and file but left the crown
    on screen (until the next refresh);
  * moving a crown saved it as a NEW group "7114" with no properties -- 2,800
    such property-less duplicates were found in the (6) trash: "properties
    are kept at import, but once I move a polygon they vanish";
  * the load-time de-duplication skipped every crown after the first one that
    shares a label, so crowns labelled by species vanished on refresh;
  * a moved crown's record `name` was overwritten with the group key.

The contract: items are named by group key and labelled by the record name;
moving keeps the group, the name and the properties; delete removes the item.
"""
import pytest

from .fixtures_manifest import fixture_image_path

pytestmark = [pytest.mark.polygons]

CROWNS = {
    # group key        label (record name)   properties
    "oleifera_7114": ("7114", {"tag": "7114", "species": "oleifera", "area_m2": 12.5}),
    "ekmanii_23": ("ekmanii", {"tag": "23", "species": "ekmanii"}),
    "ekmanii_24": ("ekmanii", {"tag": "24", "species": "ekmanii"}),
}


def _record(label, props, x0):
    return {
        'name': label, 'root': '', 'type': 'polygon', 'coord_space': 'image',
        'image_ref_size': {'w': 64, 'h': 64},
        'points': [(x0, 10.0), (x0 + 12.0, 10.0), (x0 + 12.0, 22.0), (x0, 22.0)],
        'properties': dict(props),
    }


@pytest.fixture
def loaded(synthetic_project, viewer_factory, monkeypatch):
    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    imgd = pt._imagedata_or_fallback(fp)
    viewer = viewer_factory()
    viewer.image_data = imgd
    pm = pt.convert_cv_to_pixmap(imgd.image)
    assert pm is not None and not pm.isNull()
    viewer.set_image(pm)
    rec = {"viewer": viewer, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec]
    viewer.polygon_changed.connect(pt.on_polygon_modified)

    # No root filtering, no disk writes: this is about identity, not IO.
    monkeypatch.setattr(pt, "get_root_by_filepath", lambda f: None)
    dirty = []
    monkeypatch.setattr(pt, "_mark_polygon_dirty", lambda g, f: dirty.append((g, f)))
    monkeypatch.setattr(pt, "save_incremental", lambda *a, **k: None)

    for i, (g, (label, props)) in enumerate(CROWNS.items()):
        pt.all_polygons[g] = {fp: _record(label, props, 4.0 + 18.0 * i)}
    pt._poly_norm_index_invalid = True
    try:
        pt.load_polygons(viewer, imgd)
        yield pt, viewer, fp, dirty
    finally:
        try:
            viewer.polygon_changed.disconnect(pt.on_polygon_modified)
        except Exception:
            pass
        try:
            pt.viewer_widgets.remove(rec)
        except Exception:
            pass
        for g in list(CROWNS) + ["7114", "ekmanii", "oleifera"]:
            pt.all_polygons.pop(g, None)
        pt._poly_norm_index_invalid = True
        try:
            pt.undo_stack.clear()
        except Exception:
            pass


def _items(viewer):
    return {it.name: it for it in viewer.get_all_polygons()}


def test_items_are_named_by_group_and_labelled_by_record_name(loaded):
    pt, viewer, fp, _ = loaded
    items = _items(viewer)
    labels = {label for label, _ in CROWNS.values()}
    # (the shared synthetic project carries its own fixture polygons on this image)
    assert set(CROWNS) <= set(items) and not (labels & set(items)), (
        f"scene items are named {sorted(items)}; expected the group keys {sorted(CROWNS)}. "
        "load_polygons named items by the record's label, so every lookup that treats "
        "item.name as the group key (delete, sync, save, select, Edit properties) missed "
        "them -- and crowns sharing a label (two 'ekmanii') were de-duplicated away.")
    for g, (label, _) in CROWNS.items():
        assert items[g].label_text == label, f"{g} is labelled {items[g].label_text!r}, not {label!r}"


def test_moving_a_reloaded_crown_keeps_group_name_and_properties(loaded):
    pt, viewer, fp, dirty = loaded
    it = _items(viewer)["oleifera_7114"]
    before = [tuple(p) for p in pt.all_polygons["oleifera_7114"][fp]['points']]
    it.moveBy(6.0, 0.0)
    it.polygon_modified.emit()          # what a drag release emits

    assert "7114" not in pt.all_polygons, (
        "moving the crown created a new group '7114' -- the edit was filed under the "
        "label instead of the group key, as a record with no properties (the 2,800 "
        "property-less duplicates found in C:\\Nova pasta (6)).")
    rec = pt.all_polygons["oleifera_7114"][fp]
    assert rec.get('properties') == CROWNS["oleifera_7114"][1], (
        f"properties after the move: {rec.get('properties')!r}")
    assert rec.get('name') == "7114", f"the record's name was overwritten with {rec.get('name')!r}"
    after = [tuple(p) for p in rec['points']]
    assert after != before, "the move did not reach the stored geometry"
    assert ("oleifera_7114", fp) in dirty

    # A full sync (navigation, save, editor open) must not rename or duplicate it either.
    pt.update_all_polygons()
    assert "7114" not in pt.all_polygons
    assert pt.all_polygons["oleifera_7114"][fp].get('name') == "7114"
    assert pt.all_polygons["oleifera_7114"][fp].get('properties') == CROWNS["oleifera_7114"][1]


def test_polygon_manager_delete_removes_the_reloaded_crown(loaded):
    pt, viewer, fp, _ = loaded
    pt.polygon_manager.delete_selected_polygons(["ekmanii_23"])
    names = set(_items(viewer))
    assert "ekmanii_23" not in names, (
        "Polygon Manager delete removed the record but the crown is still on screen: "
        "the delete matched item.name against the group key, and the item carried its label.")
    assert {"ekmanii_24", "oleifera_7114"} <= names, (
        f"delete took other crowns with it; left on screen: {sorted(names)}")
    assert "ekmanii_23" not in pt.all_polygons or not pt.all_polygons["ekmanii_23"]


def test_clean_vector_removes_every_crown_from_the_screen(loaded, monkeypatch):
    """Clean Vector (delete_polygons_for_viewer) queued each crown's display
    LABEL for removal from the scene. Items named by label (the old reload
    path) went; items named by group key (shapefile import, now every path)
    stayed on screen with their records and files gone -- "Clear vector still
    visible", reproduced on 2,800 real crowns."""
    pt, viewer, fp, _ = loaded
    real_lookup = pt._poly_index_lookup
    # Only this test's crowns: the shared synthetic project keeps its own
    # fixture polygons on this image, which other tests rely on.
    monkeypatch.setattr(pt, "_poly_index_lookup",
                        lambda f: [(g, k) for g, k in real_lookup(f) if g in CROWNS])
    pt.delete_polygons_for_viewer(viewer)
    left = set(CROWNS) & set(_items(viewer))
    assert not left, (
        f"after Clean Vector {sorted(left)} are still on screen (records gone): the delete "
        "matched scene items by display label, but items are identified by group key.")
    assert not any(pt.all_polygons.get(g) for g in CROWNS)
    assert not ({r.get('name') for r in viewer.polygons} & set(CROWNS)), \
        "Clean Vector left stale viewer.polygons records (they feed the zoomed-out tiles)"
