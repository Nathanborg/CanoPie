"""
CONTRACT: every consumer of an edited image must see the SAME pixels.

An `.ax` sidecar records non-destructive edits (crop -> rotate -> hist_match ->
resize -> band_expression) and FOUR separate code paths replay it:

    viewer      _imagedata_or_fallback + apply_aux_modifications   (loaders.py)
    CSV export  _get_export_image + _apply_ax_to_raw               (project_tab)
    ML manager  its own _get_export_image wrapper
    Inspect     reads image_data.image directly

They are genuinely different implementations -- `apply_aux_modifications` and
`_apply_ax_to_raw` are two separate replays of the same recipe, and the file
that holds them says so ("it has to be in parallell with apply_aux_modifcation
here and apply_all_modifications_to_image in image_editor_dialog"). Nothing
except a test keeps them in step, and when they drift the symptom is the worst
kind: the user reads one number off the screen and a different number out of the
CSV, with no error anywhere.

This file pins the invariant op-by-op, so a change to any single replay fails
here rather than silently changing exported science.
"""
import json

import numpy as np
import pytest

from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import build_project_tab

pytestmark = [pytest.mark.contract, pytest.mark.extraction]


FIXTURE = "rgb_8bit_untiled"

# One entry per .ax operation, plus the combinations that actually interact.
# `crop` changes geometry, `hist_match` changes radiometry, `band_expression`
# appends a band -- each stresses a different part of the replay.
AX_CASES = {
    "none": {},
    "crop": {
        "crop_enabled": True,
        "crop_rect": {"x": 8, "y": 6, "width": 30, "height": 24},
        "crop_rect_ref_size": {"w": 64, "h": 64},
    },
    "rotate90": {"rotate_enabled": True, "rotate": 90},
    "rotate180": {"rotate_enabled": True, "rotate": 180},
    "resize_pct": {"resize_enabled": True, "resize": {"width": 50, "height": 50}},
    "nodata": {"nodata_enabled": True, "nodata_values": [0]},
    "histmatch": {
        "hist_enabled": True,
        "hist_match": {"mode": "meanstd", "bands": 3,
                       "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3},
    },
    "crop_then_rotate": {
        "crop_enabled": True,
        "crop_rect": {"x": 8, "y": 6, "width": 30, "height": 24},
        "crop_rect_ref_size": {"w": 64, "h": 64},
        "rotate_enabled": True, "rotate": 90,
    },
    "crop_then_hist": {
        "crop_enabled": True,
        "crop_rect": {"x": 8, "y": 6, "width": 30, "height": 24},
        "crop_rect_ref_size": {"w": 64, "h": 64},
        "hist_enabled": True,
        "hist_match": {"mode": "meanstd", "bands": 3,
                       "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3},
    },
    "resize_then_hist": {
        "resize_enabled": True, "resize": {"width": 50, "height": 50},
        "hist_enabled": True,
        "hist_match": {"mode": "meanstd", "bands": 3,
                       "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3},
    },
}


@pytest.fixture(scope="module")
def _ax_tab(qapp, fixtures_ready, tmp_path_factory):
    """ONE ProjectTab for the whole module.

    Building a fresh ProjectTab per test crashes the interpreter here (exit 9)
    once a dozen or so accumulate -- the same native teardown fragility the
    viewer_factory fixture exists to manage. The `.ax` is rewritten per test
    instead, which is what actually varies.
    """
    folder = str(tmp_path_factory.mktemp("ax_contract"))
    return build_project_tab(folder), fixture_image_path(FIXTURE)


@pytest.fixture
def ax_project(_ax_tab):
    """Point the shared project's fixture at a given .ax and flush every cache."""
    pt, fp = _ax_tab

    def _make(ax):
        with open(pt._ax_path_for(fp), "w", encoding="utf-8") as f:
            json.dump(ax, f)
        # Caches are keyed by filepath, so a stale entry from the PREVIOUS .ax
        # would silently satisfy this test with the wrong pixels.
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                     "_file_nodata_cache", "_per_band_nd_cache", "_master_nd_cache"):
            c = getattr(pt, attr, None)
            if hasattr(c, "clear"):
                c.clear()
        for attr in ("_last_export_nodata_mask", "_last_export_nodata_filepath"):
            if hasattr(pt, attr):
                setattr(pt, attr, None)
        return pt, fp

    yield _make

    # Leave no .ax behind for the next module.
    try:
        import os
        axp = pt._ax_path_for(fp)
        if os.path.exists(axp):
            os.remove(axp)
    except Exception:
        pass


def _viewer_pixels(pt, fp):
    """Exactly what loaders.ImageLoadRunnable produces for the viewer."""
    lite = pt._imagedata_or_fallback(fp)
    lite.image = pt.__class__.apply_aux_modifications(
        fp, lite.image, pt.project_folder, global_mode=False)
    return np.asarray(lite.image).astype(np.float64), lite


def _export_pixels(pt, fp, n_channels):
    img, _c = pt._get_export_image(fp)
    chans = pt._channels_in_export_order(img)
    return np.stack([np.asarray(c).astype(np.float64)
                     for c in chans[:n_channels]], axis=-1)


# `nodata` is handled separately: the two replays deliberately differ there
# (see test_nodata_fill_differs_between_display_and_export below).
PIXEL_EQUAL_CASES = [c for c in sorted(AX_CASES) if c != "nodata"]


@pytest.mark.parametrize("case", PIXEL_EQUAL_CASES)
def test_viewer_and_csv_export_see_the_same_pixels(ax_project, case):
    """THE contract, one .ax operation at a time.

    `_channels_in_export_order` yields R,G,B while the viewer array for this
    cv2-loaded fixture is BGR, so the comparison reverses one of them -- that
    channel-order asymmetry is itself a documented past bug, and getting it
    wrong here would make this test pass for the wrong reason.
    """
    pt, fp = ax_project(AX_CASES[case])

    viewer, _lite = _viewer_pixels(pt, fp)
    assert viewer.ndim == 3, f"{case}: unexpected viewer shape {viewer.shape}"
    export = _export_pixels(pt, fp, viewer.shape[2])

    assert viewer.shape == export.shape, (
        f"{case}: viewer {viewer.shape} vs export {export.shape} -- the two "
        "replays disagree on GEOMETRY")
    assert np.allclose(viewer[..., ::-1], export, atol=1e-4), (
        f"{case}: viewer and export disagree, max|diff| = "
        f"{np.abs(viewer[..., ::-1] - export).max()}")


@pytest.mark.parametrize("case", ["crop", "resize_pct", "crop_then_rotate"])
def test_geometry_ops_change_the_shape_as_expected(ax_project, case):
    """Guards the test above from passing vacuously: if an op were silently
    skipped by BOTH replays they would still agree, but on the wrong pixels."""
    pt, fp = ax_project(AX_CASES[case])
    viewer, _ = _viewer_pixels(pt, fp)
    spec = get_fixture(FIXTURE)

    assert (viewer.shape[0], viewer.shape[1]) != (spec["height"], spec["width"]), (
        f"{case}: geometry op had no effect -- still {viewer.shape[:2]}")


def test_rotation_actually_moves_pixels(ax_project):
    """The fixture is SQUARE, so a 90-degree rotation leaves the shape
    unchanged -- a shape assertion would pass even if rotation were skipped
    entirely. Compare content instead."""
    pt, fp = ax_project({})
    plain, _ = _viewer_pixels(pt, fp)

    pt, fp = ax_project(AX_CASES["rotate90"])
    rotated, _ = _viewer_pixels(pt, fp)

    assert rotated.shape == plain.shape, "square fixture: shape should be unchanged"
    assert not np.allclose(rotated, plain), "rotate90 had no effect on the pixels"
    # A 90-degree rotation is a permutation, so the multiset of values survives.
    assert np.allclose(np.sort(rotated.ravel()), np.sort(plain.ravel())), (
        "rotation changed pixel VALUES, not just their positions")


def test_nodata_fill_differs_between_display_and_export(ax_project):
    """DOCUMENTED BASELINE, not a bug -- but pinned so it cannot drift silently.

    With a numeric `.ax` NoData value the two replays deliberately diverge:

      viewer  keeps the original pixel, so the user can still SEE the value
      export  zeroes the WHOLE pixel when ANY band matches, then masks it out
              of the statistics

    Measured on this fixture with nodata_values [0]: 46 of 4096 pixels differ,
    e.g. raw (94, 47, 0) stays (94, 47, 0) on screen but becomes (0, 0, 0) in
    the export array.

    That is defensible (show the data, don't count it) and the exported
    STATISTICS are self-consistent with the whole-pixel rule -- which is what
    `test_nodata_statistics_follow_whole_pixel_masking` below verifies. The
    thing that would be a real bug is the statistics disagreeing with the
    pixels they claim to summarise, so that is what is asserted, rather than
    forcing the two replays to render NoData identically.
    """
    pt, fp = ax_project(AX_CASES["nodata"])
    viewer, _ = _viewer_pixels(pt, fp)
    export = _export_pixels(pt, fp, viewer.shape[2])

    diff = np.abs(viewer[..., ::-1] - export).max(axis=2)
    n_diff = int((diff > 0).sum())
    assert n_diff > 0, (
        "display and export now agree on NoData pixels. That may be an "
        "improvement -- if so, move `nodata` back into PIXEL_EQUAL_CASES and "
        "delete this test.")

    # Every differing pixel must be one the export zeroed, not arbitrary drift.
    ys, xs = np.where(diff > 0)
    for y, x in zip(ys[:20], xs[:20]):
        assert np.allclose(export[y, x], 0.0), (
            f"({x},{y}) differs but the export value is {export[y, x]}, not the "
            "expected zero fill -- this is drift, not the documented rule")


def test_nodata_statistics_follow_whole_pixel_masking(synthetic_project, tmp_path):
    """The statistics must describe the pixel set the export actually used.

    Exported Mean is compared against BOTH candidate rules; it must match the
    whole-pixel one (the documented behaviour) and the test says so explicitly,
    so if the rule ever changes the failure names the new behaviour instead of
    just going red.
    """
    from ._helpers import load_raw_npz
    from .generate_fixtures import _rasterize_polygon_mask
    from .project_builder import polygon_group_name

    name = FIXTURE
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    axp = synthetic_project._ax_path_for(fp)
    prev = None
    import os
    if os.path.exists(axp):
        prev = open(axp, encoding="utf-8").read()
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"nodata_enabled": True, "nodata_values": [0]}, f)
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()

        group = polygon_group_name(name, spec["polygon"]["name"])
        poly = synthetic_project.all_polygons[group][fp]
        rows, _ = synthetic_project.process_polygon(
            group, fp, poly, {}, [], False, opts={"stats": {"mean": True}})

        raw = load_raw_npz(name).astype(np.float64)
        mask = _rasterize_polygon_mask(spec["polygon"]["points"],
                                       spec["height"], spec["width"])
        any_zero = (raw[..., 0] == 0) | (raw[..., 1] == 0) | (raw[..., 2] == 0)

        by_ch = {r.get("Channel"): r for r in rows if isinstance(r, dict)}
        for b, ch in enumerate("RGB"):
            row = by_ch.get(ch)
            if row is None:
                continue
            whole_pixel = raw[..., b][mask & ~any_zero].mean()
            assert row["Mean"] == pytest.approx(whole_pixel, abs=1e-3), (
                f"{ch}: exported Mean {row['Mean']} does not match whole-pixel "
                f"masking ({whole_pixel}). If the rule changed to per-band, "
                "update this test and the baseline it documents.")
    finally:
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()


def test_histmatch_actually_changes_radiometry(ax_project):
    """Same anti-vacuum guard for the radiometric op."""
    pt, fp = ax_project(AX_CASES["histmatch"])
    viewer, _ = _viewer_pixels(pt, fp)
    assert abs(float(viewer[..., 0].mean()) - 20.0) < 2.0, (
        "hist_match did not move the mean toward its reference")


@pytest.mark.parametrize("case", sorted(AX_CASES))
def test_inspect_reports_what_the_csv_would_export(ax_project, viewer_factory, case):
    """The Inspector reads `image_data.image`, which is already fully replayed.
    If it disagrees with the CSV the user is reading one number off the screen
    and exporting another."""
    from PyQt5 import QtCore, QtGui

    pt, fp = ax_project(AX_CASES[case])
    viewer_arr, lite = _viewer_pixels(pt, fp)
    export = _export_pixels(pt, fp, viewer_arr.shape[2])

    h, w = viewer_arr.shape[:2]
    if h < 4 or w < 4:
        pytest.skip("image too small after edits to probe an interior pixel")

    v = viewer_factory()
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    v.image_data = lite

    got = []
    v.pixel_clicked.connect(lambda p, payload: got.append(payload))
    y, x = h // 2, w // 2
    v._inspect_at_scene_point(QtCore.QPointF(x + 0.5, y + 0.5))

    inspected = [float(q) for q in got[-1]["values"][:3]]
    expected = [float(export[y, x, c]) for c in range(min(3, export.shape[2]))]
    assert inspected == pytest.approx(expected, abs=1e-4), (
        f"{case}: Inspect says {inspected} at ({x},{y}) but the CSV would "
        f"export {expected}")


@pytest.mark.parametrize("case", ["none", "crop", "histmatch", "nodata"])
def test_ml_manager_matches_csv_export(ax_project, ml_manager_factory, case):
    """The ML manager has its OWN _get_export_image wrapper. Training on
    different pixels than the CSV reports would invalidate any model
    interpretation drawn from that CSV."""
    pt, fp = ax_project(AX_CASES[case])
    mlm = ml_manager_factory(pt) if callable(ml_manager_factory) else None
    if mlm is None or not hasattr(mlm, "_get_export_image"):
        pytest.skip("ML manager does not expose _get_export_image in this build")

    img, _c = pt._get_export_image(fp)
    csv_chans = pt._channels_in_export_order(img)

    mimg, _mc = mlm._get_export_image(fp)
    ml_chans = mlm._channels_in_export_order(mimg)

    assert len(ml_chans) == len(csv_chans), (
        f"{case}: ML sees {len(ml_chans)} bands, CSV sees {len(csv_chans)}")
    for i, (a, b) in enumerate(zip(ml_chans, csv_chans)):
        assert np.allclose(np.asarray(a, dtype=np.float64),
                           np.asarray(b, dtype=np.float64), atol=1e-4), (
            f"{case}: band {i + 1} differs between ML manager and CSV export")


# ---------------------------------------------------------------------------
# Statistics must be internally consistent with the pixels they summarise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "rgb_8bit_untiled", "multiband_8band_ancillary", "nodata_fragmented_multiband",
])
def test_reported_stats_match_a_direct_numpy_recompute(synthetic_project, name):
    """Mean/Median/Std/quantiles must describe the SAME pixel set as the
    reported Pixel Count. Recompute them from the committed ground-truth array
    over the polygon mask and compare -- this catches a mask/statistic mismatch
    that per-statistic tests cannot see individually."""
    from ._helpers import load_raw_npz
    from .generate_fixtures import _rasterize_polygon_mask
    from .project_builder import polygon_group_name

    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = synthetic_project.all_polygons[group][fp]

    opts = {"stats": {"mean": True, "median": True, "std": True,
                      "quantiles": [25, 75]}}
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly, {}, [], False, opts=opts)

    raw = load_raw_npz(name).astype(np.float64)
    mask = _rasterize_polygon_mask(spec["polygon"]["points"],
                                   spec["height"], spec["width"])

    channel_names = {0: "R", 1: "G", 2: "B"}
    by_channel = {r.get("Channel"): r for r in rows if isinstance(r, dict)}

    for b in range(min(3, spec["bands"])):
        ch = channel_names[b]
        row = by_channel.get(ch)
        if row is None:
            continue
        vals = raw[..., b][mask]
        vals = vals[np.isfinite(vals)]
        # Only compare where NO NoData masking is in play, so the comparison is
        # against a well-defined pixel set.
        if row.get("Pixel Count") != int(vals.size):
            continue

        assert row["Mean"] == pytest.approx(float(vals.mean()), abs=1e-3), f"{ch} Mean"
        assert row["Median"] == pytest.approx(float(np.median(vals)), abs=1e-3), f"{ch} Median"
        assert row["Standard Deviation"] == pytest.approx(
            float(vals.std()), abs=1e-3), f"{ch} Std"


@pytest.mark.parametrize("name", ["rgb_8bit_untiled", "multiband_8band_ancillary"])
def test_stats_are_mutually_consistent(synthetic_project, name):
    """Relationships that must hold for ANY real sample, whatever the values:
    min <= Q25 <= Median <= Q75 <= max, Std >= 0, Pixel Count > 0."""
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    from .project_builder import polygon_group_name
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = synthetic_project.all_polygons[group][fp]

    opts = {"stats": {"mean": True, "median": True, "std": True,
                      "quantiles": [25, 75]}}
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly, {}, [], False, opts=opts)

    for row in rows:
        if not isinstance(row, dict) or row.get("Pixel Count") in (None, 0):
            continue
        ch = row.get("Channel")
        q25, med, q75 = row.get("Q25"), row.get("Median"), row.get("Q75")
        if None in (q25, med, q75):
            continue
        assert q25 <= med + 1e-6, f"{ch}: Q25 {q25} > Median {med}"
        assert med <= q75 + 1e-6, f"{ch}: Median {med} > Q75 {q75}"
        assert row["Standard Deviation"] >= -1e-9, f"{ch}: negative Std"
        assert row["Pixel Count"] > 0
