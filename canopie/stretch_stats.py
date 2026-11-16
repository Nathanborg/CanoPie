"""
Display-stretch statistics that never see NoData.

Every stretch in CanoPie (the viewer's render, the zoom tiles, the in-viewer
stretch bar, the Stretch Viewer dialog and Stretch to view) takes its bounds
from here, so they all agree on which pixels count:

  * NaN / Inf never count;
  * ArcGIS float sentinels (|v| > 1e37, e.g. -3.4028235e38) never count;
  * -9999 never counts -- the de-facto fill value of elevation models and
    ancillary planes, declared or not;
  * every NoData NUMBER passed in (the .ax list UNIONed with the raster's own
    GDAL_NODATA tag -- ``ProjectTab.effective_nodata_values``) never counts,
    matched with the same tolerance as ``canopie.nodata``.

NoData EXPRESSIONS (``b1<123``) name file bands, which a display stack does
not know, so they are not applied here; ``apply_aux_modifications`` already
turned their pixels into NaN upstream.

This only decides which values the statistics see. The pixels themselves are
left alone (except in ``nan_nodata_window``, which is used for zoom tiles read
raw from the file and must look like the preview they are drawn on).
"""
import numpy as np

__all__ = ["SENTINEL_ABS", "IMPLICIT_NODATA", "numeric_nodata", "valid_mask",
           "stats_sample", "channel_bounds", "data_range", "nan_nodata_window",
           "view_window"]

#: Anything larger in magnitude is a float NoData sentinel, not data.
SENTINEL_ABS = 1e37
#: Fill values that are never data for a display stretch.
IMPLICIT_NODATA = (-9999.0,)
#: Pixels sampled for statistics (a strided view; never a blended resample).
STATS_BUDGET = 250_000


def _tolerance(value):
    try:
        from .nodata import tolerance
        return tolerance(value)
    except Exception:
        return 0.01


def numeric_nodata(values, implicit=True):
    """The finite NoData NUMBERS in `values` (strings such as ``b1<123`` and
    NaN are skipped), plus IMPLICIT_NODATA when `implicit`. Deduplicated."""
    out = []
    seen = set()
    src = list(values or [])
    if implicit:
        src += list(IMPLICIT_NODATA)
    for v in src:
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except (TypeError, ValueError):
                continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(f) or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def valid_mask(a, nodata_values=None, implicit=True):
    """Boolean array, same shape as `a`: True where the value is real data."""
    arr = np.asarray(a)
    if arr.dtype.kind not in "fiub":
        arr = arr.astype(np.float32)
    if arr.dtype.kind == "f":
        ok = np.isfinite(arr)
        with np.errstate(invalid="ignore"):
            ok &= np.abs(arr) <= SENTINEL_ABS
    else:
        ok = np.ones(arr.shape, dtype=bool)
    if arr.dtype.kind == "b":
        return ok
    nums = numeric_nodata(nodata_values, implicit=implicit)
    if not nums:
        return ok
    if arr.dtype.kind in "iu":
        info = np.iinfo(arr.dtype)
        nums = [v for v in nums if info.min - 1 <= v <= info.max + 1]
        if not nums:
            return ok
    af = arr if arr.dtype == np.float32 else arr.astype(np.float32)
    with np.errstate(invalid="ignore"):
        for nv in nums:
            ok &= ~(np.abs(af - np.float32(nv)) <= _tolerance(nv))
    return ok


def stats_sample(a, mask=None, budget=STATS_BUDGET):
    """A strided view of `a` (and of `mask`) with about `budget` pixels.

    Strided, never area-resampled: averaging a -9999 pixel with its valid
    neighbours makes values (-3000, -500 ...) that no NoData test can find,
    and those dragged the low percentile of every DEM with a NoData border."""
    arr = np.asarray(a)
    if arr.ndim < 2:
        return arr, mask
    H, W = arr.shape[:2]
    n = H * W
    if n <= budget:
        return arr, mask
    stride = int(np.ceil(np.sqrt(n / float(budget))))
    s = arr[::stride, ::stride]
    m = np.asarray(mask)[::stride, ::stride] if mask is not None else None
    return s, m


def _channels(a):
    arr = np.asarray(a)
    if arr.ndim == 3:
        return [arr[..., c] for c in range(arr.shape[2])], True
    return [arr], False


def _excluded(mask, shape2):
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m.any(axis=2)
    return m if m.shape == tuple(shape2) else None


def _window(arr, window):
    if window is None:
        return arr, None
    x0, y0, x1, y1 = [int(v) for v in window]
    H, W = arr.shape[:2]
    x0 = max(0, min(W - 1, x0)); x1 = max(x0 + 1, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(y0 + 1, min(H, y1))
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def channel_bounds(a, mode="percentile", low_p=0.5, high_p=99.5, k_sigma=1.0,
                   nodata_values=None, mask=None, window=None,
                   budget=STATS_BUDGET, implicit=True):
    """Stretch bounds of `a` from its VALID pixels only.

    a       (H, W) or (H, W, C) display stack
    mode    "percentile" | "stddev"/"std" | "minmax"/"absolute"
    mask    optional (H, W) bool, True = exclude (mask polygons, NoData)
    window  optional (x0, y0, x1, y1): statistics from that part only
            (Stretch to view)

    Returns (lo, hi): floats for a 2-D input, float32 arrays of length C for a
    3-D one. A channel without valid pixels gets (0, 1); a constant channel a
    non-degenerate span around its value."""
    arr = np.asarray(a)
    if window is not None:
        arr, win = _window(arr, window)
        if mask is not None:
            m = np.asarray(mask)
            x0, y0, x1, y1 = win
            mask = m[y0:y1, x0:x1] if m.shape[:2] == np.asarray(a).shape[:2] else None
    s, m = stats_sample(arr, mask, budget)
    ex = _excluded(m, s.shape[:2])
    planes, multi = _channels(s)
    mode = str(mode or "percentile").lower()
    los, his = [], []
    for p in planes:
        ok = valid_mask(p, nodata_values, implicit=implicit)
        if ex is not None:
            ok &= ~ex
        v = np.asarray(p[ok], dtype=np.float64)
        if v.size == 0:
            los.append(0.0); his.append(1.0)
            continue
        if mode == "percentile":
            lo, hi = np.percentile(v, [float(low_p), float(high_p)])
        elif mode in ("stddev", "std"):
            mu, sd = float(v.mean()), float(v.std())
            lo, hi = mu - float(k_sigma) * sd, mu + float(k_sigma) * sd
        else:
            lo, hi = float(v.min()), float(v.max())
        lo, hi = float(lo), float(hi)
        if not hi > lo:
            span = abs(lo) * 1e-3 + 1e-6
            lo, hi = lo - span, hi + span
        los.append(lo); his.append(hi)
    if not multi:
        return los[0], his[0]
    return np.asarray(los, np.float32), np.asarray(his, np.float32)


def data_range(a, nodata_values=None, mask=None, per_channel=False,
               budget=STATS_BUDGET, implicit=True):
    """Valid min / max of `a`: pooled over all channels, or per channel
    (lists) with `per_channel`. None (pooled) when nothing is valid."""
    arr = np.asarray(a)
    s, m = stats_sample(arr, mask, budget)
    ex = _excluded(m, s.shape[:2])
    planes, _ = _channels(s)
    los, his = [], []
    for p in planes:
        ok = valid_mask(p, nodata_values, implicit=implicit)
        if ex is not None:
            ok &= ~ex
        if ok.any():
            v = p[ok]
            los.append(float(np.min(v))); his.append(float(np.max(v)))
        else:
            los.append(None); his.append(None)
    if per_channel:
        return los, his
    good = [(l, h) for l, h in zip(los, his) if l is not None]
    if not good:
        return None
    return min(g[0] for g in good), max(g[1] for g in good)


def nan_nodata_window(arr, values=None, file_bands=None):
    """A window read raw from the file, with NoData turned into NaN the way
    ``apply_aux_modifications`` does for the preview -- so a zoom tile drawn
    over the preview shows NoData where the preview does (white) and its
    statistics never see the fill values.

    values      the effective NoData list (.ax UNION the raster tag)
    file_bands  0-based file band of each channel (the preview_bands the
                window was read with), for per-band ``bN<x`` expressions"""
    a = np.asarray(arr)
    if a.size == 0:
        return arr
    try:
        from .nodata import rules_from_values, band_mask
        rules = rules_from_values(values or [])
    except Exception:
        rules = None
    a3 = a if a.ndim == 3 else a[..., None]
    masks = []
    any_bad = False
    for k in range(a3.shape[2]):
        plane = a3[..., k]
        fb = int(file_bands[k]) if (file_bands and k < len(file_bands)) else k
        if rules:
            m = band_mask(plane, fb, rules)
        elif plane.dtype.kind == "f":
            m = ~np.isfinite(plane)
        else:
            m = np.zeros(plane.shape, bool)
        if plane.dtype.kind == "f":
            with np.errstate(invalid="ignore"):
                m |= np.abs(plane) > SENTINEL_ABS
        masks.append(m)
        any_bad = any_bad or bool(m.any())
    if not any_bad:
        return arr
    out = a3.astype(np.float32, copy=True)
    for k, m in enumerate(masks):
        if m.any():
            out[..., k][m] = np.nan
    return out if a.ndim == 3 else out[..., 0]


def view_window(viewer, arr_shape, min_px=8):
    """The part of the viewer's displayed array that is on screen, as
    (x0, y0, x1, y1) in ARRAY pixels (the whole array when all of it is on
    screen), or None when it cannot be told.

    The pixmap may be smaller than the array (display downsampling), so the
    visible rect is mapped into pixmap coordinates and scaled by
    array / pixmap size."""
    try:
        item = getattr(viewer, "_image", None)
        if item is None:
            return None
        pm = item.pixmap()
        if pm is None or pm.isNull():
            return None
        pw, ph = float(pm.width()), float(pm.height())
        H, W = int(arr_shape[0]), int(arr_shape[1])
        vp_rect = viewer.viewport().rect()
        scene_rect = viewer.mapToScene(vp_rect).boundingRect()
        r = item.mapFromScene(scene_rect).boundingRect()
        sx, sy = W / pw, H / ph
        x0 = int(np.floor(max(0.0, r.left()) * sx))
        y0 = int(np.floor(max(0.0, r.top()) * sy))
        x1 = int(np.ceil(min(pw, r.right()) * sx))
        y1 = int(np.ceil(min(ph, r.bottom()) * sy))
        if x1 <= x0 or y1 <= y0:
            return None
        # A view deeper than a few preview pixels still needs a sample:
        # grow it around its centre.
        if x1 - x0 < min_px:
            c = (x0 + x1) // 2
            x0, x1 = max(0, c - min_px // 2), min(W, c + min_px // 2)
        if y1 - y0 < min_px:
            c = (y0 + y1) // 2
            y0, y1 = max(0, c - min_px // 2), min(H, c + min_px // 2)
        return (x0, y0, x1, y1)
    except Exception:
        return None
