"""
QC regression tests for `utils.build_nodata_mask`'s per-channel-loop -> single
vectorized pass rewrite.

WHY THIS FILE EXISTS
--------------------
Investigating a report that the Image Editor was still slow to open on a
project whose images are 15-band float32 prediction stacks with NoData
configured (`nodata_values: ['b2>1', -9999, 'b2<0.70']`) -- distinct from the
render-path bugs fixed in test_image_editor_multiband_render.py, which live in
`image_editor_dialog.py`'s own no-parent stretch fallback and are DEAD CODE
whenever a real ProjectTab parent is present (i.e. always, in the running
app). The real app's render path is `ProjectTab._render_with_viewer_stretch`,
profiled directly (parent=a real ProjectTab shell, not parent=None) against
the actual scene: `_get_cached_nodata_mask` -> `utils.build_nodata_mask` was
73% of one render (0.45s of 0.61s total), on the exact real 15-band array.

`build_nodata_mask`'s numeric-literal and NaN/Inf branches looped over every
channel individually (2*C ufunc dispatches -- 30 for a 15-band image) instead
of one vectorized call across all channels at once (2 dispatches). Rewritten
to `(np.abs(x - fv) <= tol).any(axis=2)` and `(~np.isfinite(x)).any(axis=2)`.

Note on WHY the measured cost is fixed here rather than dismissed as noise:
the exact same bytes (verified via np.array_equal(..., equal_nan=True) AND a
NaN-bit-pattern census -- both arrays carry only the single quiet-NaN encoding
0x7fc00000) took ~0.08s in a bare script but ~0.45-0.50s reproducibly through
the real dialog-open call chain, on a memory-aligned (64-byte), C-contiguous,
non-lazy plain ndarray -- alignment, NaN payload, Qt initialization, and
system load were all checked and ruled out as the cause. Regardless of that
mechanism, cutting the dispatch count from 2*C to 2 measured faster on BOTH
the fast-path and slow-path array (0.425s -> 0.269s isolated on the captured
slow-path array; corroborated in-process by a live capture showing the
combined numeric+NaN cost drop from ~0.45-0.50s to ~0.28-0.30s per call, held
across 3 runs) -- fewer, larger numpy calls are simply less exposed to
whatever the per-call overhead source is.
"""
import numpy as np
import pytest

from ..utils import build_nodata_mask

pytestmark = [pytest.mark.io, pytest.mark.perf]


def _old_loop_impl(x, nd_vals, bgr_input=True, include_nonfinite=False):
    """The exact pre-fix algorithm (per-channel loops), reimplemented here as
    a ground-truth oracle -- NOT imported from utils, since that code no
    longer exists after the fix. Mirrors build_nodata_mask's own band-index
    mapping and expression regex so the comparison is apples to apples."""
    import re
    if x.ndim == 2:
        x = x[..., None]
    H, W, C = x.shape
    mask = np.zeros((H, W), dtype=bool)

    def ch_idx(band_num):
        if C == 1:
            return 0
        if C == 2:
            return band_num - 1
        if bgr_input and C == 3 and band_num <= 3:
            return 2 - (band_num - 1)
        return band_num - 1

    # Mirrors utils._NODATA_EXPR_RE exactly: group 1 is the letter+digits
    # ("b2"), stripped of its leading letter below -- NOT digits alone.
    expr_re = re.compile(r'^([bB]\d+)\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)$')
    for v in nd_vals:
        if isinstance(v, str):
            m = expr_re.match(v)
            if not m:
                continue
            band_name, op, threshold = m.groups()
            ci = ch_idx(int(band_name[1:]))
            if ci >= C:
                continue
            ch = x[..., ci]
            th = float(threshold)
            if op == '<':
                mask |= (ch < th)
            elif op == '<=':
                mask |= (ch <= th)
            elif op == '>':
                mask |= (ch > th)
            elif op == '>=':
                mask |= (ch >= th)
            elif op == '==':
                mask |= np.isclose(ch, th, rtol=0.0, atol=1e-6)
            elif op == '!=':
                mask |= ~np.isclose(ch, th, rtol=0.0, atol=1e-6)
        else:
            try:
                fv = float(v)
                abs_fv = abs(fv)
                if abs_fv > 1e+30:
                    tol = abs_fv * 0.01
                elif abs_fv > 1e+10:
                    tol = abs_fv * 0.001
                elif abs_fv > 100:
                    tol = abs_fv * 0.001
                else:
                    tol = 0.01
                diff = np.empty((H, W), dtype=np.float32)
                for c in range(C):
                    np.subtract(x[..., c], fv, out=diff)
                    np.abs(diff, out=diff)
                    mask |= (diff <= tol)
            except Exception:
                pass

    if np.issubdtype(x.dtype, np.floating):
        for c in range(C):
            mask |= ~np.isfinite(x[..., c])
    elif include_nonfinite:
        pass  # integer imagery can never hold NaN/Inf

    return mask


def _real_style_scene(rng, H=200, W=180, C=15):
    """Shaped like the real project's rasters: some bands are entirely
    NaN-free with real values, some are ~majority NaN, and one is entirely a
    fill value -- the structure that made the per-channel branch-prediction
    behavior relevant in the first place."""
    arr = rng.rand(H, W, C).astype(np.float32) * 400 - 50
    for b in (0, 1, 2, 6):        # majority-NaN bands
        arr[: int(H * 0.9), :, b] = np.nan
    arr[..., 7] = -9999.0          # all-fill band
    return arr


@pytest.mark.parametrize("nd_vals", [
    [-9999],
    [-9999.0, 1e12],
    ["b2>1", -9999, "b2<0.70"],
    ["b1<0", "b3>=200", -9999],
    [],
])
@pytest.mark.parametrize("include_nonfinite", [False, True])
def test_matches_the_original_per_channel_loop_bit_for_bit(nd_vals, include_nonfinite):
    rng = np.random.RandomState(3)
    x = _real_style_scene(rng)

    got = build_nodata_mask(x, nd_vals, bgr_input=True, include_nonfinite=include_nonfinite)

    if not nd_vals and not include_nonfinite:
        # build_nodata_mask's own documented fast path: nothing to mask and
        # nonfinite-detection wasn't requested -> cheap None, no full-image
        # scan. This oracle (unlike the real function) has no such early
        # return, so it is not a fair ground truth here -- assert the
        # contract directly instead.
        assert got is None
        return

    want = _old_loop_impl(x, nd_vals, bgr_input=True, include_nonfinite=include_nonfinite)
    assert got is not None
    assert np.array_equal(got, want), (
        "vectorized build_nodata_mask disagrees with the original per-channel "
        f"loop for nd_vals={nd_vals!r}, include_nonfinite={include_nonfinite}")


def test_matches_the_original_loop_on_a_2d_single_band_image():
    rng = np.random.RandomState(4)
    x = rng.rand(64, 64).astype(np.float32)
    x[:10, :10] = np.nan
    x[50:, 50:] = -9999.0

    got = build_nodata_mask(x, [-9999], include_nonfinite=True)
    want = _old_loop_impl(x, [-9999], include_nonfinite=True)
    assert np.array_equal(got, want)


def test_all_fill_band_is_masked_entirely():
    """The scenario that triggered this investigation: an ancillary band that
    is the NoData value in EVERY pixel."""
    rng = np.random.RandomState(5)
    x = _real_style_scene(rng)
    mask = build_nodata_mask(x, [-9999], include_nonfinite=True)
    # Band 7 is all -9999 -- every pixel must be masked via that band alone.
    assert mask.all()


def test_uses_a_single_vectorized_pass_per_check(monkeypatch):
    """Source-level pin for the dispatch-count reduction itself: the fix's
    entire value proposition is fewer, larger numpy calls instead of one per
    channel, so a per-channel Python loop reappearing must fail this test even
    if its OUTPUT still happens to be correct."""
    import inspect
    from .. import utils

    src = inspect.getsource(utils.build_nodata_mask)
    assert "for c in range(C)" not in src, (
        "a per-channel Python loop is back in build_nodata_mask -- this was "
        "the actual bottleneck (2*C ufunc dispatches instead of 2), not "
        "merely a style preference; see this file's module docstring")
    assert ".any(axis=2)" in src, (
        "the vectorized any(axis=2) reduction is gone")
