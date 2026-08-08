"""
QC regression tests for band-math Scene Mean/Median/Std on the LAZY export path.

THE BUG THIS PINS (reported as: "the scene statistics are not being calculated"
for band-expression indices):

`process_polygon`'s band-math scene-stats block called `_eval_band_expr(expr)`
unconditionally. On the lazy path that function (`_get_bm_arr`) deliberately
returns `None` -- evaluating a formula over the whole image would decode the
exact multi-GB cube the lazy path exists to avoid
(`if _chans_are_lazy: return None`, a real and correct guard). But nothing
downstream of it had a fallback, so `arrf = np.asarray(None)` flowed straight
into `nanmean`/`nanmedian` and came out silently `NaN` -- for EVERY band-math
formula, on EVERY raster large enough to take the lazy path (the export lazy
threshold is 64 MB, so this is every real multi-band prediction stack).

Measured before the fix, `hyperspectral_200band` forced onto the lazy path:

    ch=R      Mean=535.0   SceneMean=501.8   SceneMedian=503.5   (raw band: fine)
    ch=sum3   Mean=1536.0  SceneMean=nan     SceneMedian=nan     (band math: broken)
    ch=GCC    Mean=0.333   SceneMean=nan     SceneMedian=nan     (band math: broken)

THE FIX: on the lazy path, Scene Mean/Median/Std for a band-math formula are
now computed from a DECIMATED SAMPLE of just the bands that formula references
(`LazyChannels.sample_bands`, budgeted by `_SCENE_STATS_SAMPLE_BUDGET_BYTES` /
`_SCENE_STATS_SAMPLE_DECODE_BUDGET_BYTES`), evaluated through the same
expression engine used everywhere else. This is consistent with how every
other expensive whole-image statistic in this app already works -- the
Stretch dialog's percentiles and the histogram-match reference/source stats
are both sample-based, not exact. The per-polygon row for the same formula is
NOT affected: it stays exact, computed from the polygon's own lazily-read
pixels (`_eval_band_expr_on_roi`), never from this sample. The eager path is
also unaffected: unchanged, still the exact full-image evaluation it always
was.
"""
import numpy as np
import pytest

from ..raster_reader import LazyChannels, probe, open_reader
from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name
from ._helpers import load_raw_npz

pytestmark = [pytest.mark.extraction, pytest.mark.contract]

SCENE_BM_OPTS = {
    "stats": {"mean": True, "scene_mean": True, "scene_median": True, "scene_std": True},
    "band_math": {"enabled": True, "formulas": {"sum3": "b1+b2+b3", "GCC": "b2/(b1+b2+b3)"}},
}


def _rows_by_channel(rows):
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


def _run(pt, name, force_lazy, opts=SCENE_BM_OPTS):
    from .. import project_tab as pt_module
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = pt.all_polygons[group][fp]

    old = pt_module._EXPORT_LAZY_THRESHOLD_BYTES
    pt_module._EXPORT_LAZY_THRESHOLD_BYTES = 1 if force_lazy else old
    try:
        for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                     "_bm_expr_cache"):
            c = getattr(pt, attr, None)
            if hasattr(c, "clear"):
                c.clear()
        rows, _ = pt.process_polygon(group, fp, poly, {}, [], False, opts=opts)
    finally:
        pt_module._EXPORT_LAZY_THRESHOLD_BYTES = old
    return _rows_by_channel(rows)


# ---------------------------------------------------------------------------
# LazyChannels.sample_bands -- the primitive
# ---------------------------------------------------------------------------
def test_sample_bands_matches_truth_when_budget_covers_the_whole_image():
    """With a generous budget the sample IS the whole (tiny) fixture, so this
    also validates the coordinate math has zero error in the degenerate case."""
    name = "hyperspectral_200band"
    fp = fixture_image_path(name)
    profile = probe(fp)
    reader = open_reader(fp, profile)
    lazy = LazyChannels(reader, order=list(range(profile.count)))

    sample = lazy.sample_bands([0, 1, 2], max_bytes=8 * 1024 * 1024,
                               decode_budget=64 * 1024 * 1024)
    raw = load_raw_npz(name).astype(np.float64)

    assert sample.shape == raw.shape[:2] + (3,)
    assert np.allclose(sample[..., 0].astype(np.float64), raw[..., 0])


def test_sample_bands_actually_decimates_under_a_tight_budget():
    """THE decimation guard: a tiny budget must produce a SMALLER array, not
    silently read the full image. Without this, a regression back to a
    full-image read would be invisible in the other tests here (which use
    fixtures too small to tell the difference)."""
    name = "hyperspectral_200band"
    fp = fixture_image_path(name)
    spec = get_fixture(name)
    profile = probe(fp)
    reader = open_reader(fp, profile)
    lazy = LazyChannels(reader, order=list(range(profile.count)))

    sample = lazy.sample_bands(list(range(profile.count)),
                               max_bytes=8 * 1024, decode_budget=64 * 1024)

    full_pixels = spec["height"] * spec["width"]
    assert sample.shape[0] * sample.shape[1] < full_pixels, (
        f"sample {sample.shape[:2]} is not smaller than the full frame "
        f"{(spec['height'], spec['width'])} -- decimation did not happen")


def test_sample_bands_honours_a_crop():
    """A cropped LazyChannels (origin/size set) must sample WITHIN the crop,
    not the whole underlying raster -- this is the path a cropped .ax takes."""
    name = "hyperspectral_200band"
    fp = fixture_image_path(name)
    spec = get_fixture(name)
    profile = probe(fp)
    reader = open_reader(fp, profile)

    H, W = spec["height"], spec["width"]
    ox, oy, cw, ch = W // 4, H // 4, W // 2, H // 2
    cropped = LazyChannels(reader, order=list(range(profile.count)),
                          origin=(ox, oy), size=(cw, ch))

    sample = cropped.sample_bands([0], max_bytes=8 * 1024 * 1024,
                                  decode_budget=64 * 1024 * 1024)
    raw = load_raw_npz(name).astype(np.float64)
    expected = raw[oy:oy + ch, ox:ox + cw, 0]

    assert sample.shape[:2] == expected.shape
    assert np.allclose(sample[..., 0].astype(np.float64), expected)


# ---------------------------------------------------------------------------
# process_polygon -- the production path
# ---------------------------------------------------------------------------
def test_lazy_band_math_scene_stats_are_not_nan(synthetic_project):
    """THE regression, exactly as reported: no NaN in Scene Mean/Median/Std
    for a band-math formula on the lazy path."""
    by_ch = _run(synthetic_project, "hyperspectral_200band", force_lazy=True)

    for fname in ("sum3", "GCC"):
        row = by_ch.get(fname)
        assert row is not None, f"no CSV row for {fname!r}"
        for col in ("Scene Mean", "Scene Median", "Scene Standard Deviation"):
            val = row.get(col)
            assert val is not None and np.isfinite(val), (
                f"{fname} {col} is {val!r} on the lazy path -- band-math scene "
                "stats are broken again")


def test_lazy_band_math_scene_stats_agree_with_eager(synthetic_project):
    """The sampled (lazy) value must be CLOSE to the exact (eager) value.
    On these small fixtures the budget covers the whole image, so the two are
    expected to match closely, not just "in the right ballpark"."""
    eager = _run(synthetic_project, "hyperspectral_200band", force_lazy=False)
    lazy = _run(synthetic_project, "hyperspectral_200band", force_lazy=True)

    for fname in ("sum3", "GCC"):
        e, l = eager[fname], lazy[fname]
        for col in ("Scene Mean", "Scene Median"):
            assert l[col] == pytest.approx(e[col], rel=0.05, abs=1e-6), (
                f"{fname} {col}: lazy(sampled)={l[col]} vs eager(exact)={e[col]} "
                "-- too far apart for a sample that should cover the whole "
                "(tiny) fixture")


def test_lazy_band_math_scene_stats_exclude_nodata(synthetic_project):
    """The sampled scene stats must still exclude fill pixels -- this is the
    same regression class as the raw-band lazy-path NoData fix earlier in this
    session, now for band-math formulas specifically."""
    by_ch = _run(synthetic_project, "nodata_fragmented_multiband", force_lazy=True)
    row = by_ch.get("sum3")
    assert row is not None

    assert row["Scene Mean"] > -9000.0, (
        f"Scene Mean = {row['Scene Mean']} -- the -9999 fill value leaked "
        "into the sampled band-math scene statistic")


def test_lazy_and_eager_nodata_exclusion_agree(synthetic_project):
    """Exact agreement here (not just 'not leaked'): both paths mask the same
    logical rule, so on a fixture small enough for zero sampling error they
    should produce the identical number."""
    eager = _run(synthetic_project, "nodata_fragmented_multiband", force_lazy=False)
    lazy = _run(synthetic_project, "nodata_fragmented_multiband", force_lazy=True)

    e, l = eager["sum3"], lazy["sum3"]
    assert l["Scene Mean"] == pytest.approx(e["Scene Mean"], rel=1e-6), (
        f"lazy {l['Scene Mean']} vs eager {e['Scene Mean']}")
    assert l["Scene Median"] == pytest.approx(e["Scene Median"], rel=1e-6)


def test_expression_nodata_masks_the_correct_band_within_the_formula(synthetic_project):
    """A per-band EXPRESSION rule (as opposed to a plain numeric fill value)
    must mask the BAND IT NAMES, not whichever band happens to sit first among
    the formula's referenced bands.

    `_scene_vals_for_band(vals, bi)` rewrites every rule as if it targeted
    "b1" (position 0), by design -- that is only safe when
    `_per_band_nodata_masks` is then called ONCE PER BAND with a SINGLE-band
    slice list, exactly mirroring the raw-band lazy branch. Combining several
    bands' rewritten rules into ONE call with a multi-band slice list applies
    whichever band's rule fired to SLICE POSITION 0 always (because every
    rewritten rule literally reads "b1<op>val"), not to the position the
    firing band actually occupies among the formula's referenced bands.

    A plain numeric NoData value (e.g. -9999) cannot expose this: it is
    applied identically to every band by POSITION, independent of the
    combining bug, so a naive "did the fill value leak" check on a fixture
    whose fill is numeric stays green even with the bug present -- this was
    confirmed directly: an earlier draft of this test asserted only
    `Scene Mean > -9000` and passed against the deliberately-reintroduced
    buggy version. Comparing against the EAGER path's exact value closes that
    gap, because eager scene stats are built from `nd_masks` (whole-image,
    keyed by real band index directly) -- a completely separate
    implementation, unaffected by this lazy-only rewriting, so any
    disagreement can only come from the lazy side being wrong.
    """
    from .. import project_tab as pt_module

    name = "nodata_fragmented_multiband"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = synthetic_project.all_polygons[group][fp]
    raw = load_raw_npz(name).astype(np.float64)

    # b2 (index 1) gets an expression rule; sum3 = b1+b2+b3 references [0,1,2].
    threshold = float(np.nanmedian(raw[..., 1][np.isfinite(raw[..., 1])]))
    opts = {"stats": {"scene_mean": True},
            "band_math": {"enabled": True, "formulas": {"sum3": "b1+b2+b3"}},
            "nodata_enabled": True,
            "nodata_values": [-9999, f"b2<{threshold}"]}

    def _run_forced(force_lazy):
        old = pt_module._EXPORT_LAZY_THRESHOLD_BYTES
        pt_module._EXPORT_LAZY_THRESHOLD_BYTES = 1 if force_lazy else old
        try:
            for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache"):
                c = getattr(synthetic_project, attr, None)
                if hasattr(c, "clear"):
                    c.clear()
            rows, _ = synthetic_project.process_polygon(
                group, fp, poly, {}, [], False, opts=opts)
        finally:
            pt_module._EXPORT_LAZY_THRESHOLD_BYTES = old
        return _rows_by_channel(rows).get("sum3")

    eager_row = _run_forced(force_lazy=False)
    lazy_row = _run_forced(force_lazy=True)

    assert eager_row is not None and lazy_row is not None
    assert lazy_row["Scene Mean"] == pytest.approx(eager_row["Scene Mean"], rel=1e-6), (
        f"lazy Scene Mean {lazy_row['Scene Mean']} != eager (ground truth) "
        f"{eager_row['Scene Mean']} -- the per-band expression rule for b2 is "
        "being applied to the wrong band on the lazy sampled path")


def test_per_polygon_row_stays_exact_on_the_lazy_path(synthetic_project):
    """The sampling change must be scoped to SCENE stats only. The polygon's
    own Mean for the same formula is computed from the polygon's own
    (exactly, lazily read) pixels and must be byte-identical whichever path
    computed the scene columns alongside it."""
    eager = _run(synthetic_project, "hyperspectral_200band", force_lazy=False)
    lazy = _run(synthetic_project, "hyperspectral_200band", force_lazy=True)

    for fname in ("sum3", "GCC"):
        assert lazy[fname]["Mean"] == pytest.approx(eager[fname]["Mean"], rel=1e-6), (
            f"{fname}: the polygon's own Mean changed between eager and lazy "
            "-- scene-stat sampling must not affect the exact per-polygon row")


def test_scene_stats_off_costs_no_sample_reads(synthetic_project, monkeypatch):
    """When scene stats are not requested at all, no sample should be read --
    `_want_scene` must gate the new sampling path the same way it already
    gates the raw-band one."""
    calls = {"n": 0}
    original = LazyChannels.sample_bands

    def counting(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(LazyChannels, "sample_bands", counting)

    opts = {"stats": {"mean": True},   # no scene_mean/median/std
            "band_math": {"enabled": True, "formulas": {"sum3": "b1+b2+b3"}}}
    by_ch = _run(synthetic_project, "hyperspectral_200band", force_lazy=True, opts=opts)

    assert by_ch.get("sum3") is not None, "band-math row itself must still appear"
    assert calls["n"] == 0, (
        f"sample_bands was called {calls['n']} time(s) even though no scene "
        "stat was requested")
