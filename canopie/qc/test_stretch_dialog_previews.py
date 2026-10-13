"""
The stretch dialog on rasters shown as a PREVIEW (a few bands in memory) --
Nova pasta (10), 2026-10-07: five 4-band RGBT orthomosaics, a project stretch
of file band 4 (thermal), Absolute 7446..7546.

  * every viewer was black: the preview held bands 1-3, so band 4 had no
    position and band 3 was drawn with the thermal range;
  * "apply to all viewers" never loaded band 4 into the other viewers, and the
    stretch was SAVED with band 1 (a band without a position became None);
  * the dialog opened on band 1 (the combos list FILE bands, it selected the
    array position) and reset the saved Absolute range to the band's data
    range on every open ("not keeping the range values").
"""
import types

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture

pytestmark = [pytest.mark.viewer]

NAME = "multiband_8band_ancillary"          # 8 bands, tiled float32: previews read windows
BAND = 5                                     # the FILE band the stretch shows (band 6)


def _sp(**kw):
    from ..project_tab import _StretchParams
    base = dict(mode="absolute", min_val=0.2, max_val=0.6, per_channel=False, clip=True,
                display_mode="single", display_band=None, scope="viewer")
    base.update(kw)
    sp = _StretchParams(**base)
    sp.band_refs = {"display_band": BAND, "r_band": None, "g_band": None, "b_band": None}
    return sp


def _preview_viewer(pt, bands=(0, 1, 2)):
    from ..raster_reader import probe
    fp = fixture_image_path(NAME)
    prof = probe(fp)
    arr = pt.reload_preview_bands(fp, list(bands), prof)
    shown = {}
    v = types.SimpleNamespace(
        image_data=types.SimpleNamespace(filepath=fp, image=arr, profile=prof, preview_bands=list(bands),
                                         channel_order="rgb", full_shape=(prof.height, prof.width, prof.count)),
        stretch_params=None,
        show_preview_array=lambda a, channel_order=None: shown.update(arr=np.asarray(a).copy()))
    return v, shown, prof


@pytest.fixture
def no_band_persist(synthetic_project, monkeypatch):
    # set_viewer_preview_image persists the band choice into the image's .ax:
    # never into the shared fixtures
    monkeypatch.setattr(synthetic_project, "save_display_bands", lambda *a, **k: None)
    return synthetic_project


def test_the_preview_loads_the_band_the_stretch_shows(synthetic_project, monkeypatch):
    from ..raster_reader import probe
    pt = synthetic_project
    fp = fixture_image_path(NAME)
    monkeypatch.setattr(pt, "_resolve_stretch_for_viewer", lambda f, r: _sp())
    bands = pt.get_display_bands(fp, probe(fp))
    assert BAND in bands, f"a preview for a stretch of file band {BAND + 1} holds {[b + 1 for b in bands]}"


def test_a_band_the_viewer_does_not_hold_keeps_its_file_band_when_saved(no_band_persist):
    pt = no_band_persist
    v, _shown, _prof = _preview_viewer(pt)
    sp = pt._sp_to_positions(_sp(), v)
    assert sp.display_band is None, "precondition: file band 6 has no position in a 1-2-3 preview"
    d = pt._stretch_to_dict(sp, v)
    assert d["band_basis"] == "file" and d["display_band"] == BAND, d


def test_apply_to_all_viewers_loads_the_band_into_each_preview(no_band_persist):
    pt = no_band_persist
    va, shown_a, _ = _preview_viewer(pt)
    vb, shown_b, _ = _preview_viewer(pt)
    params = _sp(scope="project")
    old = list(getattr(pt, "viewer_widgets", []) or [])
    pt.viewer_widgets = [{"viewer": va}, {"viewer": vb}]
    try:
        pt._stretch_apply_preview(params, va, None)
    finally:
        pt.viewer_widgets = old
    for v, shown in ((va, shown_a), (vb, shown_b)):
        pb = v.image_data.preview_bands
        assert BAND in pb, f"the viewer still holds bands {[b + 1 for b in pb]}"
        pos = pb.index(BAND)
        assert v.stretch_params.display_band == pos
        band = np.asarray(v.image_data.image[..., pos], np.float32)
        want = np.clip((band - 0.2) / (0.6 - 0.2) * 255.0, 0, 255)
        got = np.asarray(shown["arr"], np.float32)
        got = got if got.ndim == 2 else got[..., 0]
        ok = np.isfinite(band)
        assert np.mean(np.abs(got[ok] - want[ok])) < 1.0, "the viewer is not drawing file band 6 with the range"


def _dialog(pt, v, init):
    from ..project_tab import ImageStretchDialog
    idata = v.image_data
    prof = idata.profile
    return ImageStretchDialog(
        pt, idata.image, initial_params=init, image_filepath=idata.filepath,
        band_names=[None] * prof.count, preview_bands=idata.preview_bands,
        reload_bands_cb=lambda bands: pt.reload_preview_bands(idata.filepath, bands, prof),
        set_preview_cb=lambda arr, bands: pt.set_viewer_preview_image(v, arr, bands),
        src_channel_order="rgb")


def test_the_dialog_opens_on_the_saved_band_and_range(no_band_persist, qapp):
    pt = no_band_persist
    v, _shown, _ = _preview_viewer(pt)
    init = pt._sp_to_positions(_sp(min_val=0.25, max_val=0.55), v)
    dlg = _dialog(pt, v, init)
    try:
        assert dlg.cb_band.currentData() == BAND, f"opened on band {dlg.cb_band.currentData() + 1}"
        assert (dlg.sp_min.value(), dlg.sp_max.value()) == pytest.approx((0.25, 0.55)), \
            f"the saved range was replaced by {dlg.sp_min.value()}..{dlg.sp_max.value()}"
        assert BAND in (dlg._file_bands or []), "the band was not loaded for the dialog"
        p = dlg.get_params()
        assert p.display_band == dlg._file_bands.index(BAND) and (p.min_val, p.max_val) == pytest.approx((0.25, 0.55))
    finally:
        dlg.close(); dlg.deleteLater()


def test_the_dialog_keeps_a_saved_range_on_a_full_image(synthetic_project, qapp):
    """Not only previews: Single + Absolute reset the range on every open."""
    from ..project_tab import ImageStretchDialog
    pt = synthetic_project
    rng = np.random.default_rng(2)
    img = rng.integers(7300, 7800, size=(40, 50, 4)).astype(np.uint16)
    from ..project_tab import _StretchParams
    init = _StretchParams(mode="absolute", min_val=7446.0, max_val=7546.0, per_channel=False,
                          display_mode="single", display_band=3)
    dlg = ImageStretchDialog(pt, img, initial_params=init, image_filepath="")
    try:
        assert dlg.cb_band.currentData() == 3
        assert (dlg.sp_min.value(), dlg.sp_max.value()) == (7446.0, 7546.0), \
            f"opened at {dlg.sp_min.value()}..{dlg.sp_max.value()}"
        # a deliberate band change still adopts the new band's range
        dlg.cb_band.setCurrentIndex(dlg.cb_band.findData(0))
        assert (dlg.sp_min.value(), dlg.sp_max.value()) != (7446.0, 7546.0), \
            "a band picked in the dialog no longer takes that band's range"
    finally:
        dlg.close(); dlg.deleteLater()


def test_opening_the_dialog_leaves_the_viewer_alone_until_apply(no_band_persist, qapp):
    """The band the saved stretch needs is loaded for the DIALOG when it opens;
    the viewer gets it with the first apply / OK -- a Cancel leaves the viewer
    exactly as it was (it used to keep the new band under its old picture)."""
    pt = no_band_persist
    v, _shown, _ = _preview_viewer(pt)
    before = list(v.image_data.preview_bands)
    init = pt._sp_to_positions(_sp(), v)
    dlg = _dialog(pt, v, init)
    try:
        assert BAND in dlg._file_bands, "precondition: the dialog loaded the band"
        assert v.image_data.preview_bands == before, "opening the dialog changed the viewer"
        dlg.reject()
        assert v.image_data.preview_bands == before, "Cancel left the dialog's bands in the viewer"
    finally:
        dlg.close(); dlg.deleteLater()
    dlg = _dialog(pt, v, init)
    try:
        dlg.accept()
        assert BAND in v.image_data.preview_bands, "OK did not hand the viewer the band"
    finally:
        dlg.close(); dlg.deleteLater()


def test_a_band_choice_from_the_view_menu_is_what_the_preview_loads(synthetic_project, monkeypatch):
    """get_display_bands loads the bands the stretch shows: the View menu's
    choice must become the stretch's, or the reload loads the old ones again."""
    import json, os
    from ..raster_reader import probe
    pt = synthetic_project
    v, _shown, prof = _preview_viewer(pt)
    fp = v.image_data.filepath
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: fixture has no .ax"
    v.stretch_params = _sp()                                   # shows file band 6
    monkeypatch.setattr(pt, "viewer_widgets", [{"viewer": v}], raising=False)
    monkeypatch.setattr(pt, "load_image_group", lambda *a, **k: None)
    monkeypatch.setattr(pt, "_resolve_stretch_for_viewer",
                        lambda f, r: pt._load_file_stretch(f) or _sp())
    try:
        pt.set_display_bands([1, 2, 3])
        assert pt.get_display_bands(fp, probe(fp)) == [1, 2, 3], "the menu's band choice was overridden"
        saved = json.load(open(axp, encoding="utf-8"))["viewer_stretch"]
        assert (saved["display_mode"], saved["r_band"], saved["g_band"], saved["b_band"]) == ("rgb", 1, 2, 3)
    finally:
        if os.path.exists(axp):
            os.remove(axp)
        pt._display_bands = None


# ---------------------------------------------------------------------------
# Nathan, 2026-10-09: "should RGB_june keep its own display when an all-viewers
# stretch names a band it doesn't have? Yes"
# ---------------------------------------------------------------------------
RGB3 = "rgb_16bit_tiled_bip_cog"            # 3 bands: has no file band 6


def test_a_shared_stretch_naming_a_missing_band_is_not_used_for_that_image(synthetic_project, monkeypatch):
    pt = synthetic_project
    fp3, fp8 = fixture_image_path(RGB3), fixture_image_path(NAME)
    monkeypatch.setattr(pt, "_project_default_stretch", _sp(), raising=False)
    monkeypatch.setattr(pt, "_root_stretch_map", {"__none__": None}, raising=False)
    assert pt._resolve_stretch_for_viewer(fp8, None) is not None, "an 8-band image takes the project stretch"
    assert pt._resolve_stretch_for_viewer(fp3, None) is None, \
        "a 3-band image took a project stretch of band 6 (drawn as its band 1 with that range)"


def test_all_viewers_skips_an_image_without_the_band_and_restores_it(no_band_persist, monkeypatch):
    from ..raster_reader import probe
    pt = no_band_persist
    va, shown_a, _ = _preview_viewer(pt)                       # 8 bands
    fp3 = fixture_image_path(RGB3)
    prof3 = probe(fp3)
    arr3 = pt.reload_preview_bands(fp3, [0, 1, 2], prof3)
    shown3 = {}
    v3 = types.SimpleNamespace(
        image_data=types.SimpleNamespace(filepath=fp3, image=arr3, profile=prof3, preview_bands=[0, 1, 2],
                                         channel_order="rgb", full_shape=(prof3.height, prof3.width, prof3.count)),
        stretch_params=None,
        show_preview_array=lambda a, channel_order=None: shown3.update(arr=np.asarray(a).copy(), n=shown3.get("n", 0) + 1))
    monkeypatch.setattr(pt, "viewer_widgets", [{"viewer": va}, {"viewer": v3}], raising=False)
    monkeypatch.setattr(pt, "_project_default_stretch", None, raising=False)
    monkeypatch.setattr(pt, "_root_stretch_map", {"__none__": None}, raising=False)
    # 1) an earlier live state of the dialog -- band 1, which the 3-band image has
    band1 = _sp(scope="project")
    band1.band_refs = {"display_band": 0, "r_band": None, "g_band": None, "b_band": None}
    pt._stretch_apply_preview(band1, va, None)
    assert shown3.get("n") == 1 and v3.stretch_params.band_refs["display_band"] == 0
    # 2) then band 6, which it has not: it goes back to its own (here: default) display
    pt._stretch_apply_preview(_sp(scope="project"), va, None)
    assert shown3.get("n") == 2, "the 3-band viewer was left on the earlier shared stretch"
    assert str(v3.stretch_params.display_mode) != "single" or v3.stretch_params.band_refs is None, \
        "the 3-band viewer took the band-6 stretch"
    assert BAND in va.image_data.preview_bands, "the 8-band viewer did not take the stretch"


def test_all_viewers_keeps_the_own_stretch_of_an_image_without_the_band(synthetic_project, monkeypatch):
    import json, os
    pt = synthetic_project
    fp3, fp8 = fixture_image_path(RGB3), fixture_image_path(NAME)
    written = []
    try:
        for fp in (fp3, fp8):
            axp = pt._ax_path_for(fp)
            assert not os.path.exists(axp), "precondition: fixture has no .ax"
            json.dump({"viewer_stretch": {"mode": "absolute", "min_val": 1.0, "max_val": 2.0, "display_mode": "single",
                                          "display_band": 1, "band_basis": "file"}}, open(axp, "w", encoding="utf-8"))
            written.append(axp)
        monkeypatch.setattr(pt, "multispectral_image_data_groups", {"r": [fp3, fp8]}, raising=False)
        monkeypatch.setattr(pt, "thermal_rgb_image_data_groups", {}, raising=False)
        monkeypatch.setattr(pt, "image_data_groups", {}, raising=False)
        pt._clear_all_file_stretches(keep_for=_sp())
        kept = {os.path.basename(p): "viewer_stretch" in json.load(open(p, encoding="utf-8")) for p in written}
        assert kept[os.path.basename(written[0])] is True, "the 3-band image's own stretch was cleared"
        assert kept[os.path.basename(written[1])] is False, "the 8-band image kept a stretch the project one replaces"
    finally:
        for p in written:
            if os.path.exists(p):
                os.remove(p)
        pt._ax_cache.clear() if hasattr(getattr(pt, "_ax_cache", None), "clear") else None


def test_reloaded_bands_keep_the_ax_crop(synthetic_project):
    """reload_preview_bands returned RAW bands: a cropped image came back
    uncropped in the viewer (and the dialog could not stack it with the
    processed bands it held)."""
    import json, os
    from ..raster_reader import probe
    pt = synthetic_project
    fp = fixture_image_path(NAME)
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: fixture has no .ax"
    prof = probe(fp)
    try:
        json.dump({"crop_enabled": True, "crop_rect": {"x": 4, "y": 6, "width": 30, "height": 20},
                   "crop_rect_ref_size": {"w": prof.width, "h": prof.height}}, open(axp, "w", encoding="utf-8"))
        if hasattr(getattr(pt, "_ax_cache", None), "clear"):
            pt._ax_cache.clear()
        arr = pt.reload_preview_bands(fp, [BAND], prof)
        assert arr is not None and arr.shape[:2] == (20, 30), f"reloaded band not cropped: {getattr(arr, 'shape', None)}"
    finally:
        if os.path.exists(axp):
            os.remove(axp)
        if hasattr(getattr(pt, "_ax_cache", None), "clear"):
            pt._ax_cache.clear()
