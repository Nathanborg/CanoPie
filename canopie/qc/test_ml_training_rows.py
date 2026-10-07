"""
ML TRAINING reads the same pixels as the ML export and the CSV (2026-10-06).

MachineLearningManager._sample_training_file is train_models' per-image pixel
sampling (moved out of train_models verbatim so it can be called here). These
tests drive it on the real fixtures and pin:

  * a training row is the export's pixel: band features equal the
    "All Pixel Values" export rows and the raw ground-truth array (file band
    order on every loader -- per image, never the shared channel order, M2);
  * NoData is per band (canopie.nodata) with ONE eligibility rule, the one
    prediction uses -- a row is dropped exactly when one of ITS features is
    NoData: a fill band that is not a feature drops nothing, a selected fill
    band drops every row, an expression reading a NoData band is NoData (not
    "false") -- on the eager AND the lazy (raw read) path (T3);
  * every row is labelled with its own class, also when the image has a mask
    polygon (T1: the mask lookup used to rebind the class label).
"""
import json
import os

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_ground_truth, load_raw_npz

pytestmark = [pytest.mark.ml, pytest.mark.extraction]


def _sample(mgr, group, fp, *, bands, exprs=(), window=1, stale_order=None):
    pbg = mgr._snapshot_polygons_by_group([group])
    img, _c = mgr._get_export_image(fp)
    assert img is not None
    if stale_order is not None:
        # what another export (another thread / file) leaves in the SHARED
        # attribute; training must not read it (M2)
        mgr.parent_tab._last_export_channel_order = stale_order
    n_feat = (len(bands) + len(exprs)) * window * window
    xr, yr, _xt, _yt, dn, dm = mgr._sample_training_file(
        group, fp, img, pbg,
        selected_band_indices=list(bands), selected_expressions=list(exprs),
        window_size=window, samples_per_poly=10 ** 9, num_features=n_feat,
        augmentation_cfg={"enabled": False}, augmentation_enabled=False,
        n_variants=1, rng=np.random.default_rng(0))
    return np.asarray(xr, dtype=np.float64).reshape(-1, n_feat), list(yr), dn, dm


def _clear(pt):
    for attr in ("_export_image_cache", "_export_cache", "_per_band_nd_cache", "_master_nd_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


@pytest.mark.parametrize("name", ["rgb_8bit_png", "rgb_8bit_untiled", "rgb_16bit_tiled_bip_cog",
                                  "multiband_8band_ancillary"])
def test_training_rows_are_the_raw_pixels_in_file_band_order(synthetic_project, ml_manager_factory, name):
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    raw = load_raw_npz(name)
    clean = [b for b in range(spec["bands"]) if gt["polygon"]["bands"][str(b)]["per_band"]["count"] > 0]

    mgr = ml_manager_factory(synthetic_project)
    X, y, _dn, _dm = _sample(mgr, group, fp, bands=clean, stale_order="bgr")
    assert len(X) == gt["polygon"]["pixel_count_total"], (
        f"{name}: {len(X)} rows, expected one per polygon pixel")
    assert set(y) == {group}
    # every row is a raw pixel of the file, bands in FILE order
    lut = {tuple(np.round(raw[yy, xx, clean], 3)): 1 for yy in range(raw.shape[0]) for xx in range(raw.shape[1])}
    miss = [r for r in X if tuple(np.round(r, 3)) not in lut]
    assert not miss, f"{name}: {len(miss)} training rows are not raw file pixels (band order?), e.g. {miss[:1]}"
    # and the per-band means equal the CSV's
    for k, b in enumerate(clean):
        assert X[:, k].mean() == pytest.approx(gt["polygon"]["bands"][str(b)]["per_band"]["mean"], abs=1e-2)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_nodata_drops_a_row_only_when_one_of_its_features_is_nodata(
        synthetic_project, ml_manager_factory, monkeypatch, lazy):
    """multiband_8band_ancillary: band 8 is 100% fill (GDAL_NODATA tag only)."""
    from .. import project_tab as project_tab_module
    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    gt = load_ground_truth(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    total = gt["polygon"]["pixel_count_total"]
    pt = synthetic_project
    _clear(pt)
    if lazy:
        monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    try:
        mgr = ml_manager_factory(pt)
        img, _c = mgr._get_export_image(fp)
        assert bool(mgr._export_info(fp).get("lazy")) == lazy, "path not engaged"

        X, _y, _dn, _dm = _sample(mgr, group, fp, bands=[0, 1, 2])
        assert len(X) == total, "a fill band that is not a feature must not drop rows"

        X, _y, dn, _dm = _sample(mgr, group, fp, bands=[0, 7])
        assert len(X) == 0 and dn == total, "a selected all-NoData band must drop every row"

        # an expression reading the NoData band is NoData, not "false" (0)
        X, _y, _dn, _dm = _sample(mgr, group, fp, bands=[0], exprs=[("fill_pos", "b8 > -1")])
        assert len(X) == 0, "b8 > -1 on a NoData b8 must be NoData, not 0"
        # one that does not read it is fine
        X, _y, _dn, _dm = _sample(mgr, group, fp, bands=[0], exprs=[("r_pos", "b1 > -1")])
        assert len(X) == total and np.all(X[:, 1] == 1.0)
    finally:
        monkeypatch.undo()
        _clear(pt)


def test_rows_keep_their_class_label_when_the_image_has_a_mask_polygon(
        synthetic_project, ml_manager_factory, tmp_path):
    """T1: the mask-polygon lookup loop used `group_name` as its own loop
    variable, so every row of an image with a mask polygon was labelled with
    the LAST group of all_polygons."""
    pt = synthetic_project
    name = "rgb_8bit_untiled"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    axp = pt._ax_path_for(fp)
    assert not os.path.exists(axp), "precondition: this fixture has no .ax"
    mask_group = "zz_mask_for_training_test"
    pt.all_polygons[mask_group] = {fp: {"points": [(0, 0), (3, 0), (3, 3), (0, 3)], "type": "polygon",
                                        "name": mask_group, "image_ref_size": {"w": 64, "h": 64}}}
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"mask_polygon": {"enabled": True, "names": [mask_group]}}, f)
        _clear(pt)
        mgr = ml_manager_factory(pt)
        X, y, _dn, _dm = _sample(mgr, group, fp, bands=[0, 1, 2])
        assert len(X) > 0
        assert set(y) == {group}, f"rows relabelled: {sorted(set(y))}"
    finally:
        pt.all_polygons.pop(mask_group, None)
        if os.path.exists(axp):
            os.remove(axp)
        _clear(pt)
