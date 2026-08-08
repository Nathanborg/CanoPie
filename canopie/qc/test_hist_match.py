"""
QC tests for histogram matching: the editor's diagnostic PLOTS, and agreement
between every consumer of the matched pixels.

PLOT BUGS PINNED HERE (reported as "the cdf and mean/std plots don't seem very
correct"):

1. The CDF reference curve is obtained by differentiating the stored cumulative
   curve. The guard against a zero-width step was
   `dx = np.maximum(np.diff(xr), 1e-12)`, which does not avoid the blow-up --
   it converts `dy/0` into `dy/1e-12`, i.e. a density around 1e10 instead of
   inf. Flat runs in a stored CDF are completely ordinary (a saturated or
   constant-valued region yields many identical quantiles), and ONE such spike
   sets the y-axis scale, squashing the real and corrected curves flat onto the
   axis. With the y-ticks hidden there was no clue why.

2. Real and corrected were histogrammed with `bins=80` each, i.e. each series
   got its own edges over its own min..max. Histogram matching deliberately
   shifts and rescales values, so the two curves were binned at different widths
   over different ranges and an apparent difference in shape could be a pure
   binning artefact.

3. Mean/std matching is a LINEAR rescale: it forces the mean and the standard
   deviation and preserves the source's shape. The reference was drawn as a
   Gaussian PDF, implying the corrected curve should become bell-shaped -- it
   never will, so the plot looked permanently wrong.

Also pinned: `_ax_is_windowable` decided whether an op blocks windowed reads by
looking for an "enabled" key INSIDE the op's dict, but the editor writes the
switch as a TOP-LEVEL sibling (`hist_enabled`). Turning histogram matching off
therefore left the stored `hist_match` block behind and every large raster kept
falling back to the slow whole-image read path.
"""
import json

import numpy as np
import pytest

from ..project_tab import ProjectTab
# The REAL functions the plot calls -- not a copy of their logic. An earlier
# draft of this file reimplemented them here, which meant reverting the fix in
# the app left every one of these tests green.
from ..image_editor_dialog import cdf_reference_density as _reference_density
from ..image_editor_dialog import shared_hist_edges

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor, pytest.mark.extraction]


def _cdf_with_flat_run():
    """A stored reference CDF containing a flat run -- what a saturated or
    constant-valued image region actually produces."""
    x = np.sort(np.concatenate([
        np.linspace(0.0, 0.30, 20),
        np.full(24, 0.30),              # 24 identical quantiles
        np.linspace(0.30, 1.0, 20),
    ]))
    y = np.linspace(0.0, 1.0, x.size)
    return x, y


def test_flat_cdf_run_does_not_produce_a_density_spike():
    """THE plot regression. `np.maximum(dx, 1e-12)` turned a zero-width step
    into a density of ~1e10, which alone set the y-axis scale."""
    x, y = _cdf_with_flat_run()
    _xc, dens = _reference_density(x, y, 0.0, 1000.0)

    assert dens is not None and dens.size
    assert np.all(np.isfinite(dens)), "non-finite density leaked into the plot"
    assert dens.max() < 1.0, (
        f"reference density peaked at {dens.max():.3e} -- a zero-width CDF step "
        "is being divided by a clamped epsilon instead of dropped")


def test_reference_density_stays_comparable_to_a_real_histogram():
    """The point of the plot is comparing three curves on one axis; a reference
    orders of magnitude larger makes the other two invisible."""
    x, y = _cdf_with_flat_run()
    _xc, dens = _reference_density(x, y, 0.0, 1000.0)

    real = np.random.default_rng(0).normal(500, 120, 20000)
    counts, _edges = np.histogram(real, bins=80, density=True)

    ratio = dens.max() / max(counts.max(), 1e-30)
    assert ratio < 100.0, (
        f"reference peak is {ratio:.3e}x the real curve's -- the real and "
        "corrected curves would be flattened onto the axis")


def test_zero_width_steps_are_dropped_not_clamped():
    """A zero-width step is a point mass and has no finite density, so those
    samples must be omitted rather than assigned an invented huge value."""
    x = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
    y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    xc, dens = _reference_density(x, y, 0.0, 10.0)

    assert xc is not None
    assert xc.size == 2, f"expected only the 2 positive-width steps, got {xc.size}"
    assert np.all(np.isfinite(dens))


def test_degenerate_cdf_returns_nothing_rather_than_garbage():
    """An all-constant reference has no density anywhere; the plot must skip the
    reference line, not draw an epsilon-driven spike."""
    x = np.full(8, 0.4)
    y = np.linspace(0.0, 1.0, 8)
    xc, dens = _reference_density(x, y, 0.0, 100.0)

    assert xc is None and dens is None


def test_shared_bins_make_real_and_corrected_comparable():
    """With per-series bins, a pure linear shift changed the bin WIDTH too, so
    the two curves were not measured on the same ruler."""
    rng = np.random.default_rng(1)
    real = rng.normal(120.0, 30.0, 5000)
    corrected = (real - real.mean()) * (5.0 / real.std()) + 20.0   # meanstd match

    edges = shared_hist_edges(real, corrected, bins=80)
    assert edges is not None
    assert edges[0] <= min(real.min(), corrected.min())
    assert edges[-1] >= max(real.max(), corrected.max()), (
        "the shared edges must span BOTH series, or the corrected curve is "
        "clipped out of the plot")

    c_real, _ = np.histogram(real, bins=edges, density=True)
    c_corr, _ = np.histogram(corrected, bins=edges, density=True)

    assert c_real.size == c_corr.size == 80
    # Both are densities over the SAME edges, so both integrate to ~1.
    w = np.diff(edges)
    assert np.isclose((c_real * w).sum(), 1.0, atol=1e-6)
    assert np.isclose((c_corr * w).sum(), 1.0, atol=1e-6)


def test_meanstd_reference_is_not_drawn_as_a_gaussian():
    """Mean/std matching is a linear rescale -- it preserves the source's SHAPE.
    Drawing a Gaussian as "the reference" implied the corrected curve should
    become bell-shaped, which cannot happen; a bimodal input stays bimodal."""
    import inspect
    from ..image_editor_dialog import ImageEditorDialog

    src = inspect.getsource(ImageEditorDialog._update_hist_plot)
    meanstd_branch = src.split('if mode == "meanstd"', 1)[1].split("elif mode ==", 1)[0]
    assert "np.exp(-0.5" not in meanstd_branch, (
        "the mean/std reference is still drawn as a Gaussian PDF; mean/std "
        "matching does not make the data Gaussian")
    assert "axvline" in meanstd_branch, (
        "the mean/std reference should mark the target mean it actually matches")


def test_meanstd_preserves_shape_so_a_gaussian_reference_would_mislead():
    """Demonstrates the above numerically: a bimodal band stays bimodal after
    mean/std matching, so a Gaussian reference line can never be met."""
    rng = np.random.default_rng(2)
    bimodal = np.concatenate([rng.normal(50, 5, 3000), rng.normal(200, 5, 3000)])
    corrected = (bimodal - bimodal.mean()) * (5.0 / bimodal.std()) + 20.0

    assert np.isclose(corrected.mean(), 20.0, atol=0.1)
    assert np.isclose(corrected.std(), 5.0, atol=0.1)

    # Still two separated clusters: nothing lands in the middle of the range,
    # which no Gaussian with this mean/std could ever reproduce.
    lo_cluster = corrected[corrected < corrected.mean()]
    hi_cluster = corrected[corrected >= corrected.mean()]
    gap = hi_cluster.min() - lo_cluster.max()
    assert gap > 2.0, (
        f"expected a clear gap between the two modes, got {gap:.3f}")

    # A Gaussian(20, 5) would put ~38% of its mass within ±0.5σ of the mean;
    # the real corrected data puts essentially none there.
    near_mean = np.mean(np.abs(corrected - 20.0) < 2.5)
    assert near_mean < 0.01, (
        f"{near_mean:.1%} of corrected values sit near the mean -- the fixture "
        "is no longer bimodal, so it cannot demonstrate the point")


# ---------------------------------------------------------------------------
# Windowing gate
# ---------------------------------------------------------------------------
def test_disabled_hist_match_does_not_block_windowed_reads(synthetic_project):
    """`_ax_is_windowable` looked for an "enabled" key INSIDE hist_match, but the
    editor writes the switch as the top-level sibling `hist_enabled`. So turning
    matching OFF left the stored block behind and every large raster kept
    falling back to the slow whole-image read."""
    ax = {"hist_enabled": False,
          "hist_match": {"mode": "meanstd", "bands": 3,
                         "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3}}
    assert synthetic_project._ax_is_windowable(ax) is True, (
        "a DISABLED hist_match still blocks the lazy/windowed read path")


def test_enabled_hist_match_still_blocks_windowed_reads(synthetic_project):
    """The fix must not go the other way: matching needs whole-image statistics,
    so an ACTIVE hist_match must keep forcing the eager path."""
    ax = {"hist_enabled": True,
          "hist_match": {"mode": "meanstd", "bands": 3,
                         "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3}}
    assert synthetic_project._ax_is_windowable(ax) is False


@pytest.mark.parametrize("key,flag", [
    ("resize", "resize_enabled"),
    ("rotate", "rotate_enabled"),
    ("band_expression", "band_enabled"),
])
def test_other_ops_honour_their_top_level_enable_flag(synthetic_project, key, flag):
    """Same top-level-sibling convention throughout the .ax schema."""
    ax = {key: ({"width": 50} if key == "resize" else 90 if key == "rotate" else "b1+b2"),
          flag: False}
    assert synthetic_project._ax_is_windowable(ax) is True, (
        f"a disabled {key} still blocks windowing")


# ---------------------------------------------------------------------------
# Consumer agreement: viewer == CSV export == ML manager == Inspect
# ---------------------------------------------------------------------------
HIST_AXES = {
    "meanstd": {"hist_enabled": True,
                "hist_match": {"mode": "meanstd", "bands": 3,
                               "ref_stats": [{"mean": 20.0, "std": 5.0}] * 3}},
    "cdf": {"hist_enabled": True,
            "hist_match": {"mode": "cdf", "bands": 3,
                           "ref_cdf": {"per_band": [
                               {"lo": 0.0, "hi": 255.0,
                                "x": list(np.linspace(0, 1, 64)),
                                "y": list(np.linspace(0, 1, 64))}] * 3}}},
}


@pytest.fixture
def hist_project(tmp_path, fixtures_ready, request):
    """A project whose one fixture carries a histogram-matching .ax."""
    from .project_builder import build_project_tab
    from .fixtures_manifest import fixture_image_path

    def _make(ax):
        pt = build_project_tab(str(tmp_path))
        fp = fixture_image_path("rgb_8bit_untiled")
        with open(pt._ax_path_for(fp), "w", encoding="utf-8") as f:
            json.dump(ax, f)
        return pt, fp
    return _make


@pytest.mark.parametrize("mode", ["meanstd", "cdf"])
def test_viewer_and_csv_export_agree_pixel_for_pixel(hist_project, mode):
    """The matched pixels a user clicks on must be the matched pixels the CSV
    reports. A double-apply or a skipped apply on either side shows up here."""
    pt, fp = hist_project(HIST_AXES[mode])

    # Viewer path: the loader runs _imagedata_or_fallback THEN
    # apply_aux_modifications (see loaders.py ImageLoadRunnable.run).
    lite = pt._imagedata_or_fallback(fp)
    lite.image = pt.__class__.apply_aux_modifications(
        fp, lite.image, pt.project_folder, global_mode=False)
    viewer = np.asarray(lite.image).astype(np.float64)

    img, _C = pt._get_export_image(fp)
    chans = pt._channels_in_export_order(img)
    export = np.stack([np.asarray(c).astype(np.float64)
                       for c in chans[:viewer.shape[2]]], axis=-1)

    # _channels_in_export_order yields R,G,B; the viewer array is BGR here.
    assert np.allclose(viewer[..., ::-1], export, atol=1e-4), (
        f"{mode}: viewer and CSV export disagree, max diff "
        f"{np.abs(viewer[..., ::-1] - export).max()}")


@pytest.mark.parametrize("mode", ["meanstd", "cdf"])
def test_hist_match_actually_changed_the_pixels(hist_project, mode):
    """Guards the test above from passing vacuously: if matching were skipped
    everywhere, viewer and export would agree on the UNMATCHED data."""
    from ._helpers import load_raw_npz

    pt, fp = hist_project(HIST_AXES[mode])
    img, _C = pt._get_export_image(fp)
    export0 = np.asarray(pt._channels_in_export_order(img)[0]).astype(np.float64)
    raw0 = load_raw_npz("rgb_8bit_untiled")[..., 0].astype(np.float64)

    assert not np.allclose(export0, raw0, atol=1e-3), (
        f"{mode}: histogram matching had no effect at all")


def test_meanstd_export_hits_the_requested_statistics(hist_project):
    """The numbers, not just 'something changed': mean/std matching must land on
    the reference mean and std it was given."""
    pt, fp = hist_project(HIST_AXES["meanstd"])
    img, _C = pt._get_export_image(fp)
    band = np.asarray(pt._channels_in_export_order(img)[0]).astype(np.float64)

    assert abs(band.mean() - 20.0) < 1.0, f"mean landed at {band.mean():.3f}, wanted 20"
    assert abs(band.std() - 5.0) < 1.0, f"std landed at {band.std():.3f}, wanted 5"


def test_inspect_reports_the_matched_values(hist_project, viewer_factory):
    """The Inspector reads image_data.image, which is already fully .ax-processed;
    it must not re-apply or bypass the match."""
    from PyQt5 import QtCore, QtGui

    pt, fp = hist_project(HIST_AXES["meanstd"])
    lite = pt._imagedata_or_fallback(fp)
    lite.image = pt.__class__.apply_aux_modifications(
        fp, lite.image, pt.project_folder, global_mode=False)

    img, _C = pt._get_export_image(fp)
    chans = pt._channels_in_export_order(img)

    v = viewer_factory()
    h, w = np.asarray(lite.image).shape[:2]
    qimg = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    v.image_data = lite

    got = []
    v.pixel_clicked.connect(lambda p, payload: got.append(payload))
    v._inspect_at_scene_point(QtCore.QPointF(10.5, 10.5))

    inspected = [float(x) for x in got[-1]["values"][:3]]
    expected = [float(np.asarray(chans[c])[10, 10]) for c in range(3)]
    assert inspected == pytest.approx(expected, abs=1e-4), (
        f"Inspect reports {inspected} but the CSV would report {expected}")
