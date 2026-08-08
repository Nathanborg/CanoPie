"""
A per-band NoData EXPRESSION must affect the band it names -- including R/G/B.

THE BUG THIS PINS (reported as "I plug in b2>1 and it changes nothing"):

`process_polygon` builds `nd_roi` as a PER-BAND list of masks
(`_per_band_nodata_masks`). Bands 4+ consume it correctly, one mask each:

    valid = mask_b & ~nd_roi[j]

but the R/G/B branch collapsed all three channels onto `nd_roi[0]`:

    master_nodata_mask = nd_roi[0]          # band 1's mask
    red_vec = r_roi[valid]; green_vec = g_roi[valid]; blue_vec = b_roi[valid]

justified by the comment "All nd_roi masks should be identical when using
boolean NoData expressions". That WAS true while an expression produced a
single whole-pixel mask. It stopped being true once `_per_band_nodata_masks`
gained real per-band expression support (each band also seeds its own
`~np.isfinite`), and nothing flagged the change.

Consequence: a rule naming ANY band other than b1 had NO EFFECT on the R/G/B
rows, silently. `b1>...` appeared to work, so the feature looked functional.
Measured on a real 15-band stack with `nodata_values: ["b2>1", -9999]`, where
band 2 (a pre-computed NDVI) contains a 215.6 divide-by-zero artefact inside
the polygon:

    before:  G row  Mean = 1.5205356    (artefact included -- rule ignored)
    after:   G row  Mean = 0.8848386    (artefact excluded -- rule applied)

WHY THE EXISTING SUITE MISSED IT: every committed fixture is a smooth
deterministic ramp with no outliers, so masking a handful of pixels barely
moves a mean -- an assertion could pass whether or not the rule fired. These
tests therefore plant an EXTREME value and assert the mean moves by an amount
only correct masking can produce.
"""
import numpy as np
import pytest

from ..project_tab import _per_band_nodata_masks

pytestmark = [pytest.mark.extraction, pytest.mark.contract]


def _roi_with_outlier():
    """4 bands, 10x10. Band 2 (index 1) has ONE huge artefact at (0,0)."""
    rng = np.random.default_rng(0)
    roi = [(rng.random((10, 10), dtype=np.float32) * 0.2 + 0.8) for _ in range(4)]
    roi[1][0, 0] = 215.6
    return roi


# ---------------------------------------------------------------------------
# The mask builder itself
# ---------------------------------------------------------------------------
def test_expression_masks_only_the_band_it_names():
    roi = _roi_with_outlier()
    masks = _per_band_nodata_masks(roi, ["b2>1"])

    assert masks[1][0, 0], "b2>1 did not mask band 2's own artefact"
    assert not masks[0].any(), "b2>1 leaked into band 1's mask"
    assert not masks[2].any(), "b2>1 leaked into band 3's mask"


@pytest.mark.parametrize("band_no", [1, 2, 3, 4])
def test_every_band_can_be_targeted(band_no):
    """b1 worked while b2+ silently did not, so cover each position."""
    roi = _roi_with_outlier()
    idx = band_no - 1
    roi[idx][5, 5] = 999.0

    masks = _per_band_nodata_masks(roi, [f"b{band_no}>99"])
    assert masks[idx][5, 5], f"b{band_no}>99 did not mask band {band_no}"
    for other in range(4):
        if other != idx:
            assert not masks[other][5, 5], (
                f"b{band_no}>99 wrongly masked band {other + 1}")


# ---------------------------------------------------------------------------
# The consumer -- the part that was actually broken
# ---------------------------------------------------------------------------
def test_rgb_rows_use_their_own_bands_mask_not_band_ones():
    """THE regression, at the level the bug lived.

    Collapsing to nd_roi[0] leaves band 2's artefact in the G statistics; using
    each band's own mask removes it. The gap between the two means is enormous
    (>20x), so this cannot pass by accident.
    """
    roi = _roi_with_outlier()
    mask_b = np.ones((10, 10), dtype=bool)
    nd_roi = _per_band_nodata_masks(roi, ["b2>1"])

    # What the old code did: band 1's mask for every channel.
    old_g = roi[1][mask_b & ~nd_roi[0]].astype(np.float64)
    # What it does now: each band's own mask.
    new_g = roi[1][mask_b & ~nd_roi[1]].astype(np.float64)

    assert old_g.max() > 200, "fixture no longer contains the artefact"
    assert new_g.max() < 1.0, "the artefact survived per-band masking"
    assert old_g.mean() > 3.0, f"old behaviour should be badly skewed, got {old_g.mean()}"
    assert new_g.mean() == pytest.approx(0.9, abs=0.1), (
        f"per-band masked mean should sit with the median, got {new_g.mean()}")


def test_rgb_masking_matches_how_bands_four_and_up_already_work():
    """Bands 4+ have always used `mask_b & ~nd_roi[j]`. R/G/B must agree with
    that rule rather than having their own."""
    roi = _roi_with_outlier()
    roi[3][2, 2] = 500.0                       # artefact in band 4 as well
    mask_b = np.ones((10, 10), dtype=bool)
    nd_roi = _per_band_nodata_masks(roi, ["b2>1", "b4>99"])

    for j in range(4):
        got = roi[j][mask_b & ~nd_roi[j]].astype(np.float64)
        assert got.max() < 1.0, (
            f"band {j + 1}: per-band masking left an artefact behind (max={got.max()})")


def test_real_process_polygon_applies_a_b2_rule_to_the_G_row(
        synthetic_project, force_lazy_export):
    """THE regression, driven through the REAL consumer, on the REAL path.

    Two things this test had to get right, both of which a weaker version gets
    wrong and passes anyway:

    1. It must call `process_polygon`, not re-implement the masking. The other
       tests in this file exercise `_per_band_nodata_masks` (the mask BUILDER,
       which was always correct) and then simulate the consumer inline --
       reverting the actual fix leaves all of them green, verified directly.

    2. It must force the LAZY path (`force_lazy_export`). The committed
       fixtures are a few KB, so they take the EAGER path, and on that path
       `nd_masks` is a single whole-pixel mask replicated per band
       (`nd_masks = [master_nodata_mask] * len(chans)`) -- so every per-band
       index holds the SAME mask and the bug is invisible. The per-band masks
       only exist on the lazy path, which is where the user's 195 MB stack
       goes and where the bug was reported.

    Asserts the G row (band 2) is masked by `b2>900` while the R row (band 1)
    is untouched. Under the old "everything uses nd_roi[0]" rule the b2 rule
    reached neither row and G's mean was identical with and without it.
    """
    import json
    import os

    from .fixtures_manifest import fixture_image_path, get_fixture
    from .project_builder import polygon_group_name

    name = "multiband_8band_ancillary"          # bands span 0..999
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = synthetic_project.all_polygons[group][fp]
    axp = synthetic_project._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None

    def _run(nodata_values):
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"nodata_enabled": True, "nodata_values": nodata_values}, f)
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                     "_file_nodata_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()
        rows, _ = synthetic_project.process_polygon(
            group, fp, poly, {}, [], False, opts={"stats": {"mean": True}})
        return {r.get("Channel"): r for r in rows if isinstance(r, dict)}

    try:
        # Threshold must sit INSIDE band 2's actual range over this polygon
        # (281..737 here) or the rule matches nothing and the test passes
        # vacuously whichever masking rule is in force.
        from ._helpers import load_raw_npz
        from .generate_fixtures import _rasterize_polygon_mask
        _raw = load_raw_npz(name).astype(np.float64)
        _m = _rasterize_polygon_mask(spec["polygon"]["points"],
                                     spec["height"], spec["width"])
        _b2 = _raw[..., 1][_m]
        thr = float(np.median(_b2))
        assert (_b2 > thr).sum() > 0, "threshold matches nothing -- fixture changed"

        base = _run([])
        ruled = _run([f"b2>{thr}"])

        assert "G" in base and "G" in ruled, "no G row produced"
        assert base["G"]["Pixel Count"] > 0

        # Band 2 must lose its high pixels...
        assert ruled["G"]["Pixel Count"] < base["G"]["Pixel Count"], (
            "the b2 rule did not remove any pixel from the G row -- it is "
            "being ignored (R/G/B collapsed onto band 1's mask again)")
        assert ruled["G"]["Mean"] < base["G"]["Mean"], (
            f"G Mean unchanged by the b2 rule: {ruled['G']['Mean']} vs {base['G']['Mean']}")

        # ...while band 1 keeps every pixel, because no rule names b1.
        assert ruled["R"]["Pixel Count"] == base["R"]["Pixel Count"], (
            "a b2 rule leaked into the R row")
        assert ruled["R"]["Mean"] == pytest.approx(base["R"]["Mean"]), (
            "a b2 rule changed the R row's statistics")
    finally:
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                     "_file_nodata_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()


def test_numeric_nodata_still_applies_to_every_band():
    """A plain numeric fill value is band-independent by definition -- the
    per-band change must not break it."""
    roi = _roi_with_outlier()
    for b in range(4):
        roi[b][7, 7] = -9999.0

    masks = _per_band_nodata_masks(roi, [-9999])
    for b in range(4):
        assert masks[b][7, 7], f"numeric -9999 not masked in band {b + 1}"


def test_non_finite_is_masked_per_band():
    """Each band seeds its own ~isfinite, so a NaN in one band must not
    invalidate the same pixel in the others."""
    roi = _roi_with_outlier()
    roi[2][3, 3] = np.nan

    masks = _per_band_nodata_masks(roi, ["b2>1"])
    assert masks[2][3, 3], "NaN in band 3 was not masked"
    assert not masks[0][3, 3], "band 3's NaN leaked into band 1"
    assert not masks[1][3, 3], "band 3's NaN leaked into band 2"
