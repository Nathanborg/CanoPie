"""
The analysis image of a LARGE raster, computed window by window (2026-10-07).

A COG is read tile by tile; this applies the same idea to the `.ax` replay.
Every step that is local -- crop, rotation, a down-size (area average), NoData
per band, mask polygons, the band expression and appended expression bands
(booleans included) -- is computed on the window a consumer asks for, from the
raster tiles under it, and gives the pixels canopie.analysis_replay gives for
the whole image (same rules; see qc/test_lazy_analysis.py). Steps that need the
whole image (histogram matching, registration, classification, an up-size)
are not windowed: `supported(ax)` says no and the caller uses the full read.

The object is a raster_reader.LazyChannels, so every existing lazy consumer
(CSV, ML export, Inspect, scene statistics) takes it unchanged; its values are
float32, FILE band order, NoData as NaN per band, derived bands appended and
named by `layout` -- export info `"nodata": "nan"`, nothing left to apply.
"""
import logging
import math

import numpy as np

from . import ax_geometry as G
from .band_layout import BandLayout
from .nodata import band_mask, expression_source_bands, rules_from_values
from .raster_reader import LazyChannels, OffsetBand

__all__ = ["LazyAnalysis", "supported"]

#: a full band / stack of the export frame is refused above this (a 48031 x
#: 50101 COG band alone is 9.6 GB as float32): consumers read windows
_FULL_MATERIALIZE_LIMIT_BYTES = 1024 * 1024 * 1024


def _active(ax, key, flag=None):
    v = ax.get(key)
    if not v:
        return False
    if flag is not None and not ax.get(flag, True):
        return False
    if isinstance(v, dict) and "enabled" in v and not v.get("enabled"):
        return False
    return True


def supported(ax, raw_w, raw_h):
    """True when every active `.ax` step can be computed on a window."""
    ax = ax if isinstance(ax, dict) else {}
    if _active(ax, "hist_match", "hist_enabled"):
        return False
    from .ax_registration import registration_active
    if registration_active(ax):
        return False
    c = ax.get("classification") or {}
    if isinstance(c, dict) and str(c.get("mode", "")).lower() == "sklearn" and c.get("enabled", False):
        return False
    for b in (ax.get("appended_bands") or []):
        if isinstance(b, dict) and b.get("type") == "classification":
            return False
    ops, _wh = G.frame_plan(ax, int(raw_w), int(raw_h))
    for op in ops:
        if op[0] == "resize" and (op[1] > 1.0 or op[2] > 1.0):
            return False                      # an up-size interpolates across windows
    return True


def _band_expression_of(ax):
    be = ax.get("band_expression")
    if isinstance(be, dict):
        return str(be.get("expression") or "").strip()
    return be.strip() if isinstance(be, str) else ""


def _rot_inv_rect(k, w, h, x0, y0, x1, y1):
    """Pre-rotation rect of the post-rotation rect [x0,x1) x [y0,y1) (edge
    coordinates; (w, h) are the dims BEFORE the rotation)."""
    pts = [G._rot_inv(k, w, h, x, y) for (x, y) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


_ROT_K = {90: 3, 180: 2, 270: 1}


def _area_matrix(b, n):
    """Sparse (len(b)-1, n) area weights: row i averages the input pixels
    [j, j+1) overlapping the cell [b[i], b[i+1]) by their overlap --
    cv2.INTER_AREA's rule for a down-size, for a window whose cells need not
    start on a pixel edge (the same integration as project_tab._area_resample).
    Banded: each cell touches only the few pixels under it."""
    from scipy import sparse
    b = np.clip(np.asarray(b, np.float64), 0.0, float(n))
    lo_i = np.floor(b[:-1]).astype(np.int64)
    hi_i = np.minimum(np.ceil(b[1:]).astype(np.int64), n)
    counts = np.maximum(hi_i - lo_i, 0)
    rows = np.repeat(np.arange(b.size - 1), counts)
    starts = np.repeat(lo_i - np.cumsum(np.r_[0, counts[:-1]]), counts)
    cols = np.arange(int(counts.sum())) + starts
    lo = np.maximum(b[:-1][rows], cols)
    hi = np.minimum(b[1:][rows], cols + 1.0)
    w = np.clip(hi - lo, 0.0, None) / np.maximum(np.diff(b), 1e-12)[rows]
    return sparse.csr_matrix((w, (rows, cols)), shape=(b.size - 1, n))


def _area_apply(a3, ry, rx):
    """out[i, l, c] = sum_j sum_k ry[i, j] a3[j, k, c] rx[l, k]: rows, then
    columns, each a sparse x dense product."""
    h, w, C = a3.shape
    t = ry @ a3.reshape(h, w * C)                                   # (oh, w*C)
    oh = t.shape[0]
    t = np.ascontiguousarray(np.asarray(t).reshape(oh, w, C).transpose(1, 0, 2)).reshape(w, oh * C)
    out = rx @ t                                                    # (ow, oh*C)
    ow = out.shape[0]
    return np.ascontiguousarray(np.asarray(out).reshape(ow, oh, C).transpose(1, 0, 2))


def _area_nan(a, bx, by, integer_dtype=None):
    """Area average onto the cells between boundaries `bx`/`by` (window
    pixel coordinates), NaN where any NaN pixel overlaps the cell -- what
    cv2.INTER_AREA gives on NaN data. Values and the NaN indicator are
    averaged separately (two small matrix products per window); integer input
    is rounded back like the full-image resize."""
    a3 = a if a.ndim == 3 else a[..., None]
    ry = _area_matrix(by, a3.shape[0])
    rx = _area_matrix(bx, a3.shape[1])
    if np.issubdtype(a3.dtype, np.floating):
        # float32 products: the full-image resize (cv2.INTER_AREA) accumulates
        # float images in float32 as well
        a32 = a3.astype(np.float32, copy=False)
        nan = ~np.isfinite(a32)
        has_nan = bool(nan.any())
        ry32, rx32 = ry.astype(np.float32), rx.astype(np.float32)   # sparse, float32
        out = _area_apply(np.where(nan, np.float32(0.0), a32) if has_nan else a32, ry32, rx32)
        out = out.astype(np.float32, copy=False)
        if has_nan:
            cnt = _area_apply(nan.astype(np.float32), ry32, rx32)
            out[cnt > 1e-6] = np.nan
    else:
        vals = _area_apply(a3.astype(np.float64), ry, rx)
        info = np.iinfo(a3.dtype)
        out = np.clip(np.rint(vals), info.min, info.max).astype(a3.dtype)
    return out if a.ndim == 3 else out[..., 0]


#: scene statistics are EXACT (the whole export frame, not a sample) when the
#: requested bands of the frame fit in this many bytes
_EXACT_SCENE_BYTES = 256 * 1024 * 1024

#: computed windows kept per image (most consumers re-read the ROI they just read)
_WINDOW_CACHE_BYTES = 256 * 1024 * 1024

#: raw pixels decoded at once for one strip of a window (bytes, as float32)
_STRIP_RAW_BYTES = 256 * 1024 * 1024


class LazyAnalysis(LazyChannels):
    """See the module docstring."""

    def __init__(self, reader, profile, ax, *, pt=None, filepath=None, values=None,
                 extra_values=None, mask_records=None):
        super().__init__(reader, order=list(range(int(profile.count))), level=0)
        ax = ax if isinstance(ax, dict) else {}
        self._pt = pt
        self._filepath = filepath
        self._profile = profile
        self.n_file = int(profile.count)
        self._raw_w, self._raw_h = int(profile.width), int(profile.height)
        self._ops, (W, H) = G.frame_plan(ax, self._raw_w, self._raw_h)
        self._geo = [op for op in self._ops if op[0] in ("rotate", "crop")]
        self._resize = next((op for op in self._ops if op[0] == "resize"), None)
        if self._resize is not None:
            # the frame the resize starts from
            self._pre_w = int(round(self._resize[3] / self._resize[1]))
            self._pre_h = int(round(self._resize[4] / self._resize[2]))
            # exact: frame_plan's pre-resize size
            pw, ph = self._raw_w, self._raw_h
            for op in self._geo:
                if op[0] == "rotate" and op[1] in (90, 270):
                    pw, ph = ph, pw
                elif op[0] == "crop":
                    pw, ph = op[3] - op[1], op[4] - op[2]
            self._pre_w, self._pre_h = pw, ph
        self.width, self.height = int(W), int(H)
        vals = list(values or [])
        for v in (extra_values or []):
            if v not in vals:
                vals.append(v)
        self._rules = rules_from_values(vals) if ax.get("nodata_enabled", True) or extra_values else None
        if not self._rules:
            self._rules = None
        self._mask_records = list(mask_records or [])
        self.layout = BandLayout(self.n_file)
        self._derived = []
        expr = _band_expression_of(ax)
        if ax.get("band_enabled", True) and expr:
            self._derived.append(expr)
            self.layout.add_expression(expr)
        for info in (ax.get("appended_bands") or []):
            if isinstance(info, dict) and info.get("type") in ("boolean_expression", "band_expression"):
                e = str(info.get("expression") or "").strip()
                if e:
                    self._derived.append(e)
                    self.layout.add_expression(e, kind="appended")
        self._order = list(range(len(self.layout)))       # len(self) = every band
        self.shape = (self.height, self.width, len(self.layout))
        self._integer_raw = np.issubdtype(np.dtype(profile.dtype), np.integer) and self._rules is None

    # ---- what LazyChannels exposes -------------------------------------
    @property
    def dtype(self):
        return np.dtype(np.float32)

    def __array__(self, dtype=None):
        a = np.dstack(self.read_window(0, 0, self.width, self.height))
        return a.astype(dtype, copy=False) if dtype is not None else a

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.read_window(0, 0, self.width, self.height)[int(index)]

    def read_window_offset(self, x0, y0, x1, y1):
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.width, int(x1)); y1 = min(self.height, int(y1))
        bands = self.read_window(x0, y0, x1, y1)
        return [OffsetBand(b, x0, y0, self.height, self.width) for b in bands]

    def iter_bands(self, x0=None, y0=None, x1=None, y1=None):
        x0 = 0 if x0 is None else x0
        y0 = 0 if y0 is None else y0
        x1 = self.width if x1 is None else x1
        y1 = self.height if y1 is None else y1
        for b in self.read_window(x0, y0, x1, y1):
            yield b

    def sample_bands(self, bands, max_bytes, decode_budget=None, target_dim=None):
        """A spatially spread SAMPLE of `bands` for scene statistics: up to 16
        windows on a grid over the export frame, computed like any window.
        Returns (N, 1, len(bands)) -- consumers are elementwise."""
        bands = [int(b) for b in bands]
        key = (tuple(bands), int(max_bytes))
        kept = self.__dict__.setdefault("_sample_cache", {})
        if key in kept:
            return kept[key].copy()
        out = self._sample_bands(bands, max_bytes)
        if out.nbytes <= _EXACT_SCENE_BYTES:
            while len(kept) >= 2:
                kept.pop(next(iter(kept)))
            kept[key] = out
            return out.copy()
        return out

    def _sample_bands(self, bands, max_bytes):
        nb = max(1, len(bands))
        budget_px = max(64, int(max_bytes) // (4 * len(self)))
        W, H = self.width, self.height
        if W * H * nb * 4 <= _EXACT_SCENE_BYTES:
            # the whole export frame fits: exact (read in row strips, so the
            # all-band window stays bounded), as the full-image path computes it
            rows = max(1, int(_EXACT_SCENE_BYTES // max(1, W * len(self) * 4)))
            parts = []
            for y0 in range(0, H, rows):
                win = self.read_window(0, y0, W, min(H, y0 + rows))
                parts.append(np.stack([win[b].reshape(-1) for b in bands], axis=-1))
            return np.concatenate(parts, axis=0)[:, None, :]
        n = 16
        side = max(8, int(math.sqrt(budget_px / n)))
        gx = gy = 4
        out = []
        for j in range(gy):
            for i in range(gx):
                cx = int((i + 0.5) * W / gx); cy = int((j + 0.5) * H / gy)
                x0 = max(0, cx - side // 2); y0 = max(0, cy - side // 2)
                win = self.read_window(x0, y0, min(W, x0 + side), min(H, y0 + side))
                out.append(np.stack([win[b].reshape(-1) for b in bands], axis=-1))
        return np.concatenate(out, axis=0)[:, None, :]

    # ---- the computation -------------------------------------------------
    def _raw_rect_for(self, px0, py0, px1, py1):
        """Raw-raster rect under the pre-resize frame rect (edge coords)."""
        x0, y0, x1, y1 = float(px0), float(py0), float(px1), float(py1)
        for op in reversed(self._geo):
            if op[0] == "crop":
                x0 += op[1]; x1 += op[1]; y0 += op[2]; y1 += op[2]
            elif op[0] == "rotate":
                x0, y0, x1, y1 = _rot_inv_rect(op[1], op[2], op[3], x0, y0, x1, y1)
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    def read_window(self, x0, y0, x1, y1, max_materialize=None):
        """Every band (file bands, then derived) of export-frame window
        [x0, x1) x [y0, y1), float32, NoData = NaN per band. Recently
        computed windows are kept (a byte-budgeted LRU): consumers ask for
        the same window more than once. Callers get copies."""
        W, H = self.width, self.height
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(W, int(x1)); y1 = min(H, int(y1))
        key = (x0, y0, x1, y1)
        cache = self.__dict__.setdefault("_win_cache", {})
        hit = cache.get(key)
        if hit is not None:
            cache[key] = cache.pop(key)                 # most recent
            return [b.copy() for b in hit]
        bands = self._compute_window(x0, y0, x1, y1)
        nb = sum(b.nbytes for b in bands)
        if nb <= _WINDOW_CACHE_BYTES // 4:
            total = self.__dict__.get("_win_cache_bytes", 0) + nb
            while cache and total > _WINDOW_CACHE_BYTES:
                old = cache.pop(next(iter(cache)))
                total -= sum(b.nbytes for b in old)
            cache[key] = bands
            self._win_cache_bytes = total
            return [b.copy() for b in bands]
        return bands

    def _compute_window(self, x0, y0, x1, y1):
        W, H = self.width, self.height
        if x1 <= x0 or y1 <= y0:
            return [np.zeros((max(0, y1 - y0), max(0, x1 - x0)), np.float32) for _ in range(len(self))]
        nbytes = (x1 - x0) * (y1 - y0) * 4 * len(self)
        if nbytes > _FULL_MATERIALIZE_LIMIT_BYTES:
            raise MemoryError(
                f"{(x1 - x0)} x {(y1 - y0)} x {len(self)} window of a lazily read image "
                f"({nbytes / 1e9:.1f} GB) -- read it in smaller windows")

        # 1-4) in row strips, so the raw pixels decoded at once stay bounded
        # however strongly the frame is down-sized
        strips = []
        rows_per = self._rows_per_strip(x0, x1)
        for sy0 in range(y0, y1, rows_per):
            strips.append(self._strip(x0, sy0, x1, min(y1, sy0 + rows_per)))
        win = strips[0] if len(strips) == 1 else np.concatenate(strips, axis=0)
        win = np.ascontiguousarray(win, dtype=np.float32)

        # 5) mask polygons: excluded from every band
        if self._mask_records:
            pm = self._mask_window(x0, y0, x1, y1)
            if pm is not None:
                win[pm] = np.nan

        # 6) derived bands, in layout order
        for expr in self._derived:
            res = self._eval(win, expr)
            if res is None or res.shape != win.shape[:2]:
                res = np.full(win.shape[:2], np.nan, np.float32)
            win = np.concatenate([win, res[..., None]], axis=2)
        return [win[..., k] for k in range(win.shape[2])]

    def _pre_rect(self, x0, y0, x1, y1):
        """The pre-resize frame rect covering export rect [x0,x1) x [y0,y1),
        and the area-cell boundaries in that rect's pixel coordinates."""
        if self._resize is None:
            return (x0, y0, x1, y1), None, None
        sx, sy = float(self._resize[1]), float(self._resize[2])
        px0 = int(math.floor(x0 / sx)); px1 = min(self._pre_w, int(math.ceil(x1 / sx)))
        py0 = int(math.floor(y0 / sy)); py1 = min(self._pre_h, int(math.ceil(y1 / sy)))
        bx = np.arange(x0, x1 + 1, dtype=np.float64) / sx - px0
        by = np.arange(y0, y1 + 1, dtype=np.float64) / sy - py0
        return (px0, py0, px1, py1), bx, by

    def _rows_per_strip(self, x0, x1):
        (px0, _py0, px1, _py1), _bx, _by = self._pre_rect(x0, 0, x1, 1)
        sy = float(self._resize[2]) if self._resize is not None else 1.0
        raw_row_bytes = max(1, (px1 - px0)) * self.n_file * 4 / max(sy, 1e-9)
        return max(1, int(_STRIP_RAW_BYTES // max(1.0, raw_row_bytes)))

    def _strip(self, x0, y0, x1, y1):
        """Steps 1-4 for export rows [y0, y1): raw tiles, rotation, NoData of
        the raw values, the down-size."""
        (px0, py0, px1, py1), bx, by = self._pre_rect(x0, y0, x1, y1)
        rx0, ry0, rx1, ry1 = self._raw_rect_for(px0, py0, px1, py1)
        raw = self._reader.read_window(rx0, ry0, rx1, ry1, bands=list(range(self.n_file)), level=0)
        raw = np.asarray(raw)
        if raw.ndim == 2:
            raw = raw[..., None]
        for op in self._geo:
            if op[0] == "rotate":
                raw = np.rot90(raw, k=_ROT_K[op[1]])
        win = np.ascontiguousarray(raw)
        if self._rules is not None or not self._integer_raw:
            f = win.astype(np.float32, copy=True)
            if self._rules is not None:
                for k in range(self.n_file):
                    m = band_mask(f[..., k], k, self._rules)
                    if m.any():
                        f[..., k][m] = np.nan
            win = f
        if bx is not None:
            win = _area_nan(win, bx, by)
        return win

    def _mask_polygons(self):
        """Each mask polygon rasterised ONCE, on its own bounding box inside
        the export frame -- exactly the pixels a whole-frame fillPoly gives
        (drawing per window clips the polygon differently at window edges)."""
        if getattr(self, "_mask_cache", None) is not None:
            return self._mask_cache
        import cv2
        out = []
        for pts, ref in self._mask_records:
            try:
                a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
                rw = float((ref or {}).get("w") or 0) if isinstance(ref, dict) else 0.0
                rh = float((ref or {}).get("h") or 0) if isinstance(ref, dict) else 0.0
                if rw > 0 and rh > 0:
                    a = a * (self.width / rw, self.height / rh)
                v = np.round(a).astype(np.int64)
                bx0 = max(0, int(v[:, 0].min())); by0 = max(0, int(v[:, 1].min()))
                bx1 = min(self.width, int(v[:, 0].max()) + 1); by1 = min(self.height, int(v[:, 1].max()) + 1)
                if bx1 <= bx0 or by1 <= by0:
                    continue
                m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                cv2.fillPoly(m, [(v - (bx0, by0)).astype(np.int32).reshape(-1, 1, 2)], 1)
                out.append((bx0, by0, m.astype(bool)))
            except Exception as e:
                logging.debug("[lazy_analysis] mask polygon skipped: %s", e)
        self._mask_cache = out
        return out

    def _mask_window(self, x0, y0, x1, y1):
        m = None
        for bx0, by0, pm in self._mask_polygons():
            ix0, iy0 = max(x0, bx0), max(y0, by0)
            ix1, iy1 = min(x1, bx0 + pm.shape[1]), min(y1, by0 + pm.shape[0])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if m is None:
                m = np.zeros((y1 - y0, x1 - x0), bool)
            m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] |= pm[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        return m if (m is not None and m.any()) else None

    def _eval(self, stack, expr):
        from .analysis_replay import _eval_expression, _with_nan_where_sources_are
        try:
            res = _eval_expression(self._pt, stack, expr)
        except Exception as e:
            logging.warning("[lazy_analysis] expression %r failed: %s", expr, e)
            return None
        if res is None:
            return None
        return _with_nan_where_sources_are(np.asarray(res, np.float32), stack, expression_source_bands(expr))
