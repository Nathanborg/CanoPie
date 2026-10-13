"""
Find one tree crown in other georeferenced images (orthomosaics).

A polygon drawn on one GeoTIFF is taken to the ORIGINAL raster (undoing its
.ax crop / rotate / resize, ax_geometry), to world coordinates (the GeoTIFF
transform), into every other GeoTIFF (pyproj when the CRS differs), and is then
refined there by image registration -- the exhaustive engine of
Raw_image_drone_aligniment_exhaustive.ipynb (brute-force rotation +
translation, gated masked NCC) or its StackReg variant -- because two mosaics
of the same forest are usually a few metres apart.

Frames used here (all continuous pixel-EDGE coordinates, pixel i = [i, i+1)):

  * raw      -- the file's own pixels at level 0 (what the georeference describes)
  * record   -- a polygon record: the post-.ax frame in units of image_ref_size
  * search   -- the SOURCE pixel grid around the crown; the target is resampled
                into it (so GSD, CRS and grid rotation differences vanish) and the
                registration only has to find the residual shift + rotation.

No Qt in this module: the dialog / worker / browser drive it.
"""
import contextlib
import functools
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ax_geometry as G

__all__ = ["DEFAULTS", "GeoRef", "georef", "map_px", "local_affine", "gsd_m",
           "record_to_raw", "raw_to_record", "read_raw_window", "prepare_source",
           "find_in_target", "frame_for", "pair_kind", "write_geotiff_crop", "export_crowns", "SourceContext",
           "Result", "compute_polygon_bbox", "build_canopy_mask", "exhaustive_align",
           "stackreg_align"]

DEFAULTS = {
    "method": "stackreg",            # "stackreg" | "exhaustive" | "geo". StackReg: the same answers on the
                                     # real FLIR / mosaic tests, 2-4x faster; exhaustive is used anyway when
                                     # a frame's heading is unknown (a rotation search StackReg can't do)
    "search_radius_m": 15.0,         # translation searched around the geo-projected crown (GeoTIFF pairs)
    "angle_range": 5.0,              # +/- degrees (exhaustive), GeoTIFF pairs
    "search_radius_exif_m": 40.0,    # an EXIF-estimated frame on either side (GPS error + camera tilt)
    "angle_range_exif": 20.0,        # (180 when a frame has no heading)
    "scale_min_exif": 0.7,           # relative scale searched for raw frames: a ground size 20 % too small
    "scale_max_exif": 1.4,           # was still found, 25 % too large broke the match (palm test, 2026-10-12);
                                     # the palm frames needed 1.15-1.2: margin above that
    "scale_step": 0.05,
    "search_radius_pct": 15.0,       # two non-located images (phenocam): % of the image size
    "angle_range_pixel": 2.0,
    "min_crown_px": 12,              # warn below this crown size at the work resolution
    "search_max_px": 1600,           # the search image is matched at most this big (coarser work grid)
    "angle_step": 1.0,
    "refine_step": 0.5,
    "topn": 5,
    "min_canopy_frac": 0.5,          # share of the crown that must land on real pixels
    "ncc_mask_weight": 0.2,          # notebook value: combined = w*mask + (1-w)*window
    "min_ncc": 0.2,                  # below: status "low_score" (outline still returned); correct
                                     # cross-season matches on pasta (10) scored 0.23-0.55
    "coarse_downscale": 2,
    "work_max_px": 900,              # search image larger than this is searched downscaled
    "reg_band": "all",               # "all" = B+2R-G (notebook), "mean", or a 1-based band
    "bands": None,                   # file bands (0-based) used for matching; None = first 3
    "zero_is_nodata": True,          # orthomosaic fill: all bands 0 = no data
    # crop context (notebook compute_polygon_bbox)
    "buffer_ratio": 0.3, "min_buffer": 30, "context_ring_ratio": 2.0,
    "min_context_ring": 0, "max_context_ring": 5300,
}


# ════════════════════════════════════════════════════════════════════════
# Georeference
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GeoRef:
    """Where an image's raw pixels lie. kind:
      "geotiff" -- its own georeference;
      "exif"    -- approximate, from the GPS / height / focal / heading of a raw
                   drone frame (canopy_finder_meta.exif_transform, EPSG:4326);
      "pixel"   -- not located (a phenocam): only its own pixel frame."""
    path: str
    transform: tuple          # GDAL order: X = t0 + c*t1 + r*t2 ; Y = t3 + c*t4 + r*t5
    epsg: object              # int or None
    crs_wkt: object
    raw_w: int                # size of the FRAME these coordinates are in (post-.ax when composed)
    raw_h: int
    kind: str = "geotiff"
    note: str = ""
    ref_w: int = 0            # raw file size = the basis polygon records are stamped with
    ref_h: int = 0
    frame: str = "raw"        # "raw" | "export" (the post-.ax frame the viewer / CSV use)
    base: object = None       # the raw-frame GeoRef an export frame was composed from

    @property
    def located(self):
        return self.kind != "pixel"

    @property
    def basis(self):
        return (self.ref_w or self.raw_w), (self.ref_h or self.raw_h)

    def from_record(self, rec, points=None):
        """(N, 2) frame coordinates of a polygon record of this image. Records
        are fractions of the post-.ax frame in units of image_ref_size, so in
        the export frame this is a plain scale (no .ax replay needed)."""
        pts = points if points is not None else (rec.get("points") or [])
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not len(a):
            raise ValueError("the polygon has no vertices")
        cs = str(rec.get("coord_space", "image") or "image").lower()
        if self.frame != "export":
            return record_to_raw(rec, {}, self.raw_w, self.raw_h, points=pts)
        if cs == "norm_pixmap":
            return a * (self.raw_w, self.raw_h)
        if cs in ("image", "pixel"):
            bw, bh = self.basis
            rw, rh = _ref_wh(rec, bw, bh)
            return a * (self.raw_w / rw, self.raw_h / rh)
        raise ValueError(f"polygons stored in '{cs}' space are not supported here")

    def to_record(self, pts):
        """Frame coordinates -> (record points, image_ref_size, centroid inside the frame)."""
        e = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        bw, bh = self.basis
        cx, cy = e.mean(axis=0)
        inside = bool(0 <= cx <= self.raw_w and 0 <= cy <= self.raw_h)
        rec = e * (bw / float(max(1, self.raw_w)), bh / float(max(1, self.raw_h)))
        return [(float(x), float(y)) for x, y in rec], {"w": int(bw), "h": int(bh)}, inside

    def px_to_world(self, pts):
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        t = self.transform
        return np.column_stack((t[0] + a[:, 0] * t[1] + a[:, 1] * t[2],
                                t[3] + a[:, 0] * t[4] + a[:, 1] * t[5]))

    def world_to_px(self, pts):
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        t = self.transform
        det = t[1] * t[5] - t[2] * t[4]
        dx, dy = a[:, 0] - t[0], a[:, 1] - t[3]
        return np.column_stack(((t[5] * dx - t[2] * dy) / det, (-t[4] * dx + t[1] * dy) / det))


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@functools.lru_cache(maxsize=512)
def _georef_cached(path, _mtime_key):
    from .shapefile_io import get_geotiff_transform, epsg_from_geotiff, _raw_image_dims
    if not str(path).lower().endswith((".tif", ".tiff")):
        return None
    tr, wkt = get_geotiff_transform(path)
    if not tr or not (tr[1] * tr[5] - tr[2] * tr[4]):
        return None
    try:
        h, w = _raw_image_dims(path)
    except Exception:
        h = w = None
    if not (h and w):
        return None
    return GeoRef(path=path, transform=tuple(float(v) for v in tr), epsg=epsg_from_geotiff(path),
                  crs_wkt=wkt, raw_w=int(w), raw_h=int(h))


def georef(path):
    """GeoRef of a GeoTIFF, or None (not a TIFF / no ModelTiepoint+PixelScale)."""
    return _georef_cached(str(path), _mtime(path))


class NotLocated(ValueError):
    pass


def _geometry_active(ax, raw_w, raw_h):
    ops, _wh = G.frame_plan(ax or {}, raw_w, raw_h)
    return bool(ops)


def compose_ax(g, ax):
    """The frame of `g` after the image's .ax crop / rotate / resize: the frame the
    viewer shows and the CSV / ML read. Every step is affine, so the result is
    again a GeoRef, its transform composed with export -> raw."""
    if g is None:
        return None
    ops, (W, H) = G.frame_plan(ax or {}, g.raw_w, g.raw_h)
    if not ops:
        return GeoRef(path=g.path, transform=g.transform, epsg=g.epsg, crs_wkt=g.crs_wkt, raw_w=g.raw_w,
                      raw_h=g.raw_h, kind=g.kind, note=g.note, ref_w=g.raw_w, ref_h=g.raw_h, frame="export",
                      base=g)
    r = G.inverse_np(ops, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    L = np.column_stack((r[1] - r[0], r[2] - r[0]))            # export -> raw linear part
    c = r[0]
    t = g.transform
    T = np.array([[t[1], t[2]], [t[4], t[5]]], np.float64)
    TL = T @ L
    o = T @ c + (t[0], t[3])
    tr = (float(o[0]), float(TL[0, 0]), float(TL[0, 1]), float(o[1]), float(TL[1, 0]), float(TL[1, 1]))
    return GeoRef(path=g.path, transform=tr, epsg=g.epsg, crs_wkt=g.crs_wkt, raw_w=int(W), raw_h=int(H),
                  kind=g.kind, note=g.note, ref_w=g.raw_w, ref_h=g.raw_h, frame="export", base=g)


def frame_for(path, fetch_missing=True, ax=None):
    """The frame of an image: its GeoTIFF georeference, else an EXIF camera-model
    estimate, else its plain pixel frame -- composed with its .ax when `ax` is
    given (None = the raw file frame). None only for an unreadable image."""
    g = _base_frame(path, fetch_missing)
    return compose_ax(g, ax) if (g is not None and ax is not None) else g


def _base_frame(path, fetch_missing=True):
    g = georef(path)
    if g is not None:
        return g
    from . import canopy_finder_meta as M
    wh = M.image_size(path)
    if not wh:
        return None
    w, h = wh
    meta = M.get_meta(path)
    if meta is None and fetch_missing:
        M.fetch([path])
        meta = M.get_meta(path)
    tr, info = M.exif_transform(meta, w, h) if meta else (None, "no EXIF")
    if tr is None and meta and meta.get("lat") is not None:
        # GPS but no height above ground / sensor: measure the collection itself
        calib = calibrate_frame(path) or _borrowed_calibration(path)
        if calib and calib.get("gsd_m"):
            tr, info = M.exif_transform(meta, w, h, calib=calib)
            if calib.get("borrowed_from"):
                info["note"] += f"; borrowed from {calib['borrowed_from']}"
    if tr is not None:
        return GeoRef(path=str(path), transform=tuple(float(v) for v in tr), epsg=4326, crs_wkt=None,
                      raw_w=int(w), raw_h=int(h), kind="exif", note=info.get("note", ""))
    # not located: its "world" is the fraction of its own raw frame, so two
    # phenocam frames relate by position in the image whatever their .ax does
    return GeoRef(path=str(path), transform=(0.0, 1.0 / w, 0.0, 0.0, 0.0, 1.0 / h), epsg=None, crs_wkt=None,
                  raw_w=int(w), raw_h=int(h), kind="pixel", note=str(info))


def _pair_shift(a_fp, b_fp, cancel=None):
    """Where the centre of frame a shows up in frame b (same camera, same
    scale): (shift px (dx, dy) at full resolution, score)."""
    A = _decode(a_fp)
    B = _decode(b_fp)
    H, W = A.shape[:2]
    k = max(1.0, max(W, H) / 900.0)
    sw, sh = max(8, int(round(W / k))), max(8, int(round(H / k)))
    nb = min(3, A.shape[2], B.shape[2])

    def small(X):
        x = cv2.resize(np.ascontiguousarray(X[:, :, :nb]).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        if x.ndim == 2:
            x = x[:, :, None]
        v = np.isfinite(x).all(axis=2) & (x != 0).any(axis=2)
        return to_uint8(x, v), v
    a8, av = small(A)
    b8, bv = small(B)
    x0, y0 = sw // 4, sh // 4
    crop, cv_ = a8[y0:y0 + sh // 2, x0:x0 + sw // 2], av[y0:y0 + sh // 2, x0:x0 + sw // 2]
    cm = np.ones(crop.shape[:2], bool)
    q2t, info = exhaustive_align(np.ascontiguousarray(crop), cv_, np.ascontiguousarray(b8), bv, cm,
                                 # consecutive frames of a flight line barely turn
                                 {"angle_range": 4.0, "angle_step": 2.0, "refine_step": 0.5, "topn": 3,
                                  "min_canopy_frac": 0.3, "ncc_mask_weight": 0.2, "coarse_downscale": 2,
                                  "work_max_px": 900, "reg_band": "all"}, cancel)
    c = np.array([crop.shape[1] / 2.0, crop.shape[0] / 2.0])
    cb = q2t[:, :2] @ c + q2t[:, 2]
    d = (cb - (c + (x0, y0))) * k
    return (float(d[0]), float(d[1])), float(info.get("score") or -1)


def _borrowed_calibration(path, reach=8):
    """The calibration of the nearest measured frame of the same collection
    (its heading turned by the GPS-track difference), for a frame whose own
    stretch could not be measured (a turn, a weak overlap)."""
    from . import canopy_finder_meta as M
    members = M.collection_members(path)
    norm = [os.path.normcase(m) for m in members]
    try:
        i = norm.index(os.path.normcase(os.path.abspath(path)))
    except ValueError:
        return None
    me = M.get_meta(path, full=False) or {}
    for d in range(1, reach + 1):
        for j in (i - d, i + d):
            if 0 <= j < len(members):
                c = M.get_calibration("frame|" + norm[j])
                if c and c.get("gsd_m"):
                    out = dict(c, borrowed_from=os.path.basename(members[j]))
                    other = M.get_meta(members[j], full=False) or {}
                    if me.get("track") is not None and other.get("track") is not None and c.get("yaw_deg") is not None:
                        out["yaw_deg"] = (c["yaw_deg"] + me["track"] - other["track"]) % 360.0
                    return out
    # nothing measured nearby yet: measure a neighbour's stretch, then borrow it
    for j in (i + 4, i - 4):
        if 0 <= j < len(members) and calibrate_frame(members[j]) is not None:
            return _borrowed_calibration(path, reach=reach) if reach > 0 else None
    return None


_CALIB_LOCKS = {}
_CALIB_LOCKS_LOCK = threading.Lock()
_PAIR_MAPPER = threading.local()


@contextlib.contextmanager
def pair_shifts_with(mapper):
    """Measurements of flight lines started in this thread register their frame
    pairs through mapper([(a, b), ...]) -> [((dx, dy), score) or an exception]
    (the worker processes) instead of one after another here."""
    old = getattr(_PAIR_MAPPER, "fn", None)
    _PAIR_MAPPER.fn = mapper
    try:
        yield
    finally:
        _PAIR_MAPPER.fn = old


def calibrate_frame(path, cancel=None, span=3):
    """See _calibrate_frame. One measurement per folder at a time: the frames
    nearest a crown share a flight stretch, so parallel searches would measure
    it once each; the later ones find it cached."""
    nd = os.path.normcase(os.path.dirname(os.path.abspath(path)))
    with _CALIB_LOCKS_LOCK:
        lk = _CALIB_LOCKS.setdefault(nd, threading.Lock())
    with lk:
        return _calibrate_frame(path, cancel, span)


def _calibrate_frame(path, cancel=None, span=3):
    """Ground size and heading of a raw frame whose EXIF has GPS but no height
    above ground (FLIR Duo Pro R, MicaSense: GPSAltitude is above sea level).

    The frames around it on the same flight line (same camera / band, within
    +-`span` in time, heading within 20 deg) are registered pair by pair. The
    cumulative pixel shifts u_k against the cumulative GPS displacements D_k
    give, by least squares, one similarity D = c u: |c| is m / px, arg c the
    heading. One GPS step is NOT enough: the FLIR GPS updates out of step with
    the shutter (consecutive frames even repeat a position), so single pairs gave
    0.0085 / 0.014 / 0.030 m / px for three frames of one cruise line; over a
    20-40 m baseline the noise averages out. Every frame of the stretch is
    cached with the result. None when the stretch can't be measured."""
    from . import canopy_finder_meta as M
    M.fetch([path], full=False)
    key = "frame|" + os.path.normcase(os.path.abspath(path))
    hit = M.get_calibration(key)
    if hit is not None:
        return hit if hit.get("gsd_m") else None
    members = M.collection_members(path)
    norm = [os.path.normcase(m) for m in members]
    try:
        i = norm.index(os.path.normcase(os.path.abspath(path)))
    except ValueError:
        i = 0
    lo, hi = max(0, i - span), min(len(members), i + span + 1)
    seg = members[lo:hi]
    pos = M.positions(seg)
    seg = [m for m in seg if m in pos]
    why = "no neighbour frames with GPS"
    if len(seg) >= 2:
        # keep the straight stretch around the frame (a turn breaks one similarity)
        ref = seg.index(members[i]) if members[i] in seg else 0

        def brg(a, b):
            return M.bearing_deg(pos[a][0], pos[a][1], pos[b][0], pos[b][1])
        if len(seg) >= 3:
            h0 = brg(seg[max(0, ref - 1)], seg[min(len(seg) - 1, ref + 1)])
            keep = [m for k, m in enumerate(seg)
                    if k == ref or abs((brg(seg[min(k, ref)], seg[max(k, ref)]) - h0 + 180) % 360 - 180) < 20
                    or pos[m] == pos[seg[ref]]]
            seg = keep
    best = None
    if len(seg) >= 2:
        # pair shifts along the stretch; a weak pair splits it into runs
        pairs = []
        todo = list(zip(seg, seg[1:]))
        mapper = getattr(_PAIR_MAPPER, "fn", None)
        shifts = mapper(todo) if (mapper is not None and len(todo) > 1) else None
        for k, (a, b) in enumerate(todo):
            if cancel is not None and cancel():
                break
            try:
                if shifts is None:
                    (sx, sy), score = _pair_shift(a, b, cancel)
                elif isinstance(shifts[k], BaseException):
                    raise shifts[k]
                else:
                    (sx, sy), score = shifts[k]
            except Exception as e:
                why = f"pair registration failed: {e}"
                pairs.append(None)
                continue
            if score < 0.3:
                why = f"pair registration weak ({os.path.basename(a)} -> {os.path.basename(b)}: {score:.2f})"
                pairs.append(None)
                continue
            pairs.append((sx, sy, score))
        runs, cur = [], []
        for k, p in enumerate(pairs):
            if p is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(k)
        if cur:
            runs.append(cur)
        ref = seg.index(members[i]) if members[i] in seg else 0
        # the run touching the frame, else the longest
        run = next((r for r in runs if r[0] <= ref <= r[-1] + 1), None) or (max(runs, key=len) if runs else [])
        if run:
            u, D, scores = [0j], [0j], []
            cu = 0j
            la0, lo0 = pos[seg[run[0]]]
            for k in run:
                sx, sy, score = pairs[k]
                scores.append(score)
                # the ground moved (sx, sy) px between the frames: the camera moved
                # (-sx, -sy) px, i.e. (-sx, +sy) with y up
                cu += complex(-sx, sy)
                b = seg[k + 1]
                dx, dy = M._local_m(la0, lo0, pos[b][0], pos[b][1])
                u.append(cu)
                D.append(complex(float(dx), float(dy)))
            u, D = np.array(u), np.array(D)
            if len(u) >= 3 and np.abs(u).max() >= 30 and np.abs(D - D[0]).max() >= 5:
                # least squares D ~ c u + d (d absorbs the first frame's own GPS error)
                uc, Dc = u - u.mean(), D - D.mean()
                c = (Dc * np.conj(uc)).sum() / max(1e-9, (np.abs(uc) ** 2).sum())
                resid = np.abs(Dc - c * uc)
                rel = float(np.sqrt((resid ** 2).mean()) / max(1e-9, np.abs(Dc).max()))
                yaw = (-math.degrees(math.atan2(c.imag, c.real))) % 360.0
                if rel < 0.35:
                    best = {"gsd_m": float(abs(c)), "yaw_deg": yaw, "score": float(np.mean(scores)),
                            "frames": len(u), "baseline_m": float(np.abs(D - D[0]).max()), "fit_rel_err": rel}
                    seg = [seg[k] for k in run] + [seg[run[-1] + 1]]
                else:
                    why = f"GPS and image shifts disagree (relative misfit {rel:.2f})"
            else:
                why = "too short a stretch to measure"
    if cancel is not None and cancel():
        return None
    if best is not None:
        for m in seg:
            k = "frame|" + os.path.normcase(os.path.abspath(m))
            if M.get_calibration(k) is None:
                M.put_calibration(k, dict(best, via=os.path.basename(path)))
    else:
        M.put_calibration(key, {"failed": why})
    M.save()
    logging.info("[canopy_finder] %s: %s", os.path.basename(path), best or why)
    return best


@functools.lru_cache(maxsize=64)
def _transformer(src_epsg, dst_epsg):
    from pyproj import Transformer
    return Transformer.from_crs(int(src_epsg), int(dst_epsg), always_xy=True)


def _same_crs(a, b):
    if a.epsg and b.epsg:
        return int(a.epsg) == int(b.epsg)
    return True       # unknown on one side: assume the same (logged by the caller)


def map_px(src, dst, pts):
    """Raw pixels of `src` -> raw pixels of `dst` through world coordinates
    (two pixel frames: the same fraction of each frame)."""
    if src.located != dst.located:
        raise NotLocated("a located and a non-located image can't be related")
    w = src.px_to_world(pts)
    if not src.located:
        return dst.world_to_px(w)
    if not _same_crs(src, dst):
        x, y = _transformer(src.epsg, dst.epsg).transform(w[:, 0], w[:, 1])
        w = np.column_stack((np.asarray(x, np.float64), np.asarray(y, np.float64)))
    return dst.world_to_px(w)


def crown_latlon(g, pt):
    """(lat, lon) of a raw pixel of a located frame, or None."""
    if not g.located:
        return None
    w = g.px_to_world([pt])[0]
    if g.epsg and int(g.epsg) != 4326:
        lon, lat = _transformer(int(g.epsg), 4326).transform(float(w[0]), float(w[1]))
    elif g.epsg or _is_geographic(g):
        lon, lat = float(w[0]), float(w[1])
    else:
        return None
    return float(lat), float(lon)


def local_affine(src, dst, x0, y0, x1, y1):
    """2x3 affine A with dst_px ~= A @ [x, y, 1] over the source box -- exact for
    two grids in the same CRS, a least-squares fit over a 5x5 grid otherwise."""
    gx, gy = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
    p = np.column_stack((gx.ravel(), gy.ravel()))
    q = map_px(src, dst, p)
    X = np.column_stack((p, np.ones(len(p))))
    sol, *_ = np.linalg.lstsq(X, q, rcond=None)
    return sol.T                                  # (2, 3)


def _is_geographic(g):
    if g.kind == "exif":
        return True
    try:
        from pyproj import CRS
        if g.epsg:
            return bool(CRS.from_epsg(int(g.epsg)).is_geographic)
        if g.crs_wkt:
            return bool(CRS.from_wkt(g.crs_wkt).is_geographic)
    except Exception:
        pass
    t = g.transform
    return abs(t[1]) < 1e-2 and -180 <= t[0] <= 360      # degrees-sized pixels


def gsd_m(g, x, y):
    """Ground size (m) of one raw pixel at (x, y): (along x, along y); a pixel
    frame has no ground size (1, 1)."""
    if not g.located:
        return 1.0, 1.0
    w = g.px_to_world([(x, y), (x + 1, y), (x, y + 1)])
    if _is_geographic(g):
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        _, _, dx = geod.inv(w[0, 0], w[0, 1], w[1, 0], w[1, 1])
        _, _, dy = geod.inv(w[0, 0], w[0, 1], w[2, 0], w[2, 1])
        return abs(dx), abs(dy)
    return float(np.hypot(*(w[1] - w[0]))), float(np.hypot(*(w[2] - w[0])))


# ════════════════════════════════════════════════════════════════════════
# Polygon records <-> raw pixels (the same maps load_polygons / the CSV use)
# ════════════════════════════════════════════════════════════════════════
def _ref_wh(rec, raw_w, raw_h):
    ref = rec.get("image_ref_size") or rec.get("image_size") or rec.get("pixmap_size")
    if isinstance(ref, dict) and ref.get("w") and ref.get("h"):
        return float(ref["w"]), float(ref["h"])
    if isinstance(ref, (list, tuple)) and len(ref) >= 2 and ref[0] and ref[1]:
        return float(ref[0]), float(ref[1])
    return float(raw_w), float(raw_h)


def record_to_raw(rec, ax, raw_w, raw_h, points=None):
    """(N, 2) raw-pixel vertices of a polygon record."""
    pts = points if points is not None else (rec.get("points") or [])
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if not len(a):
        raise ValueError("the polygon has no vertices")
    ops, (W, H) = G.frame_plan(ax or {}, raw_w, raw_h)
    cs = str(rec.get("coord_space", "image") or "image").lower()
    if cs == "norm_pixmap":
        e = a * (W, H)
    elif cs in ("image", "pixel"):
        rw, rh = _ref_wh(rec, raw_w, raw_h)
        e = a * (W / rw, H / rh)
    else:
        raise ValueError(f"polygons stored in '{cs}' space are not supported here")
    return G.inverse_np(ops, e)


def raw_to_record(raw_pts, ax, raw_w, raw_h):
    """Raw-pixel vertices -> (record points, image_ref_size, inside_frame)."""
    ops, (W, H) = G.frame_plan(ax or {}, raw_w, raw_h)
    e = G.forward_np(ops, np.asarray(raw_pts, dtype=np.float64).reshape(-1, 2))
    cx, cy = e.mean(axis=0)
    inside = bool(0 <= cx <= W and 0 <= cy <= H)
    rec = e * (raw_w / float(max(1, W)), raw_h / float(max(1, H)))
    return [(float(x), float(y)) for x, y in rec], {"w": int(raw_w), "h": int(raw_h)}, inside


# ════════════════════════════════════════════════════════════════════════
# Pixels
# ════════════════════════════════════════════════════════════════════════
_FULL_CACHE = {}                # key -> array, oldest first
_FULL_BYTES = 1024 * 1024 * 1024  # decoded whole images kept (searches run in parallel)
_FULL_LOCK = threading.Lock()
_FULL_KEY_LOCKS = {}


def _is_tiff(path):
    return str(path).lower().endswith((".tif", ".tiff"))


def _decode(path):
    """Whole image (h, w, C) of a non-windowable file -- a TIFF via tifffile
    (memmap when possible), anything else via cv2 (BGR -> RGB, EXIF orientation
    ignored like everywhere in CanoPie). A byte-budgeted LRU shared by the
    search threads; one decode per file even when several threads want it."""
    key = (os.path.normcase(os.path.abspath(path)), _mtime(path))
    with _FULL_LOCK:
        arr = _FULL_CACHE.get(key)
        if arr is not None:
            _FULL_CACHE[key] = _FULL_CACHE.pop(key)
            return arr
        klock = _FULL_KEY_LOCKS.setdefault(key, threading.Lock())
    with klock:
        with _FULL_LOCK:
            arr = _FULL_CACHE.get(key)
        if arr is not None:
            return arr
        if _is_tiff(path):
            import tifffile
            try:
                arr = tifffile.memmap(path)
            except Exception:
                arr = tifffile.imread(path)
            if arr.ndim == 3 and arr.shape[0] < min(arr.shape[1:]) and arr.shape[0] <= 64:
                arr = np.moveaxis(arr, 0, -1)
        else:
            buf = np.fromfile(path, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"cannot decode {os.path.basename(path)}")
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        with _FULL_LOCK:
            _FULL_CACHE[key] = arr
            total = sum(int(getattr(a, "nbytes", 0)) for a in _FULL_CACHE.values())
            while len(_FULL_CACHE) > 1 and total > _FULL_BYTES:
                old = _FULL_CACHE.pop(next(iter(_FULL_CACHE)))
                total -= int(getattr(old, "nbytes", 0))
            _FULL_KEY_LOCKS.pop(key, None)
    return arr


_full_array = _decode       # round-1 name


def _probe(path):
    if not _is_tiff(path):
        return None
    try:
        from .raster_reader import probe
        return probe(path)
    except Exception:
        return None


def file_band_count(path):
    p = _probe(path)
    if p is not None and p.count:
        return int(p.count)
    if not _is_tiff(path):
        try:
            from PIL import Image
            with Image.open(path) as im:
                return len(im.getbands())
        except Exception:
            pass
    return int(_decode(path).shape[2])


def file_nodata(path):
    p = _probe(path)
    return getattr(p, "nodata", None) if p is not None else None


def _reader(path):
    if not _is_tiff(path):
        return None
    try:
        from .raster_reader import open_reader
        return open_reader(path)
    except Exception:
        return None


def read_raw_window(path, x0, y0, x1, y1, bands=None, factor=1.0):
    """Raw window [y0:y1, x0:x1] (level-0 pixels), ZERO-PADDED where it leaves the
    raster, plus its real-pixel mask -- so the caller's origin is always (x0, y0).

    factor > 1 asks for about one output pixel per `factor` raw pixels: a COG
    is read from the overview level closest below it. Returns (arr, real,
    (fx, fy), (ox, oy)) then: raw pixels per returned pixel and the raw position
    of the returned top-left corner (the caller finishes the reduction). With
    factor == 1 it returns (arr, real) as before."""
    x0, y0, x1, y1 = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
    rd = _reader(path)
    level, sx, sy = 0, 1.0, 1.0
    if rd is not None:
        ih, iw = rd.level_shape(0)[:2]
        n = rd.level_shape(0)[2] if len(rd.level_shape(0)) > 2 else 1
        if factor > 1.0:
            for L in range(1, int(getattr(rd.profile, "n_levels", 1) or 1)):
                try:
                    lh, lw = rd.level_shape(L)[:2]
                except Exception:
                    break
                if iw / float(lw) <= factor * 1.01:
                    level, sx, sy = L, iw / float(lw), ih / float(lh)
                else:
                    break
        dtype = rd.profile.dtype
    else:
        full = _decode(path)
        ih, iw, n = full.shape
        dtype = full.dtype
    bands = list(range(min(n, 3))) if bands is None else [int(b) for b in bands if 0 <= int(b) < n]
    # the window at the chosen level (its own pixels)
    lx0, ly0 = int(math.floor(x0 / sx)), int(math.floor(y0 / sy))
    lx1, ly1 = int(math.ceil(x1 / sx)), int(math.ceil(y1 / sy))
    liw, lih = (int(round(iw / sx)), int(round(ih / sy))) if level else (iw, ih)
    h, w = max(0, ly1 - ly0), max(0, lx1 - lx0)
    out = np.zeros((h, w, len(bands)), dtype=dtype)
    real = np.zeros((h, w), dtype=bool)
    cx0, cy0, cx1, cy1 = max(0, lx0), max(0, ly0), min(liw, lx1), min(lih, ly1)
    if cx1 > cx0 and cy1 > cy0:
        if rd is not None:
            arr = rd.read_window(cx0, cy0, cx1, cy1, bands=bands, level=level)
        else:
            arr = np.asarray(full[cy0:cy1, cx0:cx1][:, :, bands])
        if arr.ndim == 2:
            arr = arr[:, :, None]
        hh, ww = arr.shape[:2]
        out[cy0 - ly0:cy0 - ly0 + hh, cx0 - lx0:cx0 - lx0 + ww] = arr
        real[cy0 - ly0:cy0 - ly0 + hh, cx0 - lx0:cx0 - lx0 + ww] = True
    if factor == 1.0:
        return out, real
    # (lx0, ly0) * s may differ from (x0, y0) by < one level pixel: report the true origin
    return out, real, (sx, sy), (lx0 * sx, ly0 * sy)


def _pixel_steps(ax):
    """The .ax changes pixel VALUES (not only geometry / which bands are shown)."""
    ax = ax or {}
    from .lazy_analysis import _band_expression_of
    if ax.get("band_enabled", True) and _band_expression_of(ax):
        return True
    for k in ("hist_match", "registration", "classification", "appended_bands", "nodata_values",
              "nodata_expression", "nodata_expressions", "mask_polygon"):
        v = ax.get(k)
        if isinstance(v, dict) and not v.get("enabled", True):
            continue
        if v:
            return True
    return False


class Frames:
    """Pixels of an image in its post-.ax frame -- what the viewer shows and the
    CSV / ML read -- by window:

      * no .ax step at all: the raw file, by window (COG: from an overview);
      * local steps (crop, rotation, down-size, NoData, mask polygons, band
        math): canopie.lazy_analysis.LazyAnalysis windows (windowable TIFFs);
      * otherwise (histogram matching, band registration, classification, a JPG
        with an .ax): the app's own whole-image replay (ProjectTab._get_export_image).

    `pt` (a ProjectTab) supplies NoData rules, mask polygons and the full replay;
    without it only the first two kinds apply. Values: float32 with NaN for
    NoData on the .ax paths, native dtype on the raw path."""

    def __init__(self, pt=None):
        self.pt = pt
        self._lazy = {}
        self._lazy_lock = threading.Lock()
        self._read_locks = {}            # one window reader per image at a time (its caches)

    def mode(self, fp, ax):
        if not ax:
            return "raw"
        wh = _file_wh(fp)
        if wh and not _geometry_active(ax, *wh) and not _pixel_steps(ax):
            return "raw"
        rd = _reader(fp)
        if rd is not None:
            from .lazy_analysis import supported
            if supported(ax, rd.level_shape(0)[1], rd.level_shape(0)[0]):
                return "lazy"
        return "full" if self.pt is not None else "raw"

    def _lazy_for(self, fp, ax):
        key = (os.path.normcase(os.path.abspath(fp)), _mtime(fp), repr(sorted((ax or {}).items(), key=str)))
        with self._lazy_lock:
            return self._lazy_locked(key, fp, ax)

    def _lazy_locked(self, key, fp, ax):
        lz = self._lazy.get(key)
        if lz is None:
            from .lazy_analysis import LazyAnalysis
            rd = _reader(fp)
            vals, masks = [], []
            if self.pt is not None:
                try:
                    vals = self.pt.effective_nodata_values(fp, ax) if ax.get("nodata_enabled", True) else []
                except Exception:
                    vals = []
                try:
                    from .analysis_replay import _mask_polygon_records
                    masks = _mask_polygon_records(self.pt, ax, fp, getattr(self.pt, "all_polygons", None))
                except Exception:
                    masks = []
            else:
                vals = [v for v in (ax.get("nodata_values") or []) if v is not None] \
                    if ax.get("nodata_enabled", True) else []
                nd = file_nodata(fp)
                if nd is not None and nd not in vals:
                    vals.append(nd)
            lz = LazyAnalysis(rd, rd.profile, ax, pt=self.pt, filepath=fp, values=vals, mask_records=masks)
            if len(self._lazy) > 8:
                self._lazy.clear()
            self._lazy[key] = lz
        return lz

    def size(self, fp, ax):
        g = frame_for(fp, fetch_missing=False, ax=ax or {})
        return (g.raw_w, g.raw_h) if g is not None else None

    def read(self, fp, ax, x0, y0, x1, y1, bands=None, factor=1.0, zero_is_nodata=True):
        """(arr (h, w, C), valid (h, w), (fx, fy), (ox, oy)) of frame window
        [x0, x1) x [y0, y1): zero-padded / invalid where it leaves the frame;
        (fx, fy) frame pixels per returned pixel, (ox, oy) its top-left."""
        m = self.mode(fp, ax)
        if m == "raw":
            if factor > 1.0:
                arr, real, f, o = read_raw_window(fp, x0, y0, x1, y1, bands, factor=factor)
            else:
                arr, real = read_raw_window(fp, x0, y0, x1, y1, bands)
                f, o = (1.0, 1.0), (float(math.floor(x0)), float(math.floor(y0)))
            nd = file_nodata(fp)
            if ax and ax.get("nodata_enabled", True):
                extra = [v for v in (ax.get("nodata_values") or []) if v is not None]
            else:
                extra = []
            v = valid_mask(arr, real, nd, zero_is_nodata)
            for val in extra:
                try:
                    v &= ~(arr == np.asarray(val, dtype=arr.dtype)).all(axis=2)
                except Exception:
                    pass
            return arr, v, f, o
        x0, y0, x1, y1 = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
        h, w = max(0, y1 - y0), max(0, x1 - x0)
        with self._lazy_lock:
            rlock = self._read_locks.setdefault(os.path.normcase(os.path.abspath(fp)), threading.Lock())
        if m == "lazy":
            src = self._lazy_for(fp, ax)
            W, H = src.width, src.height
            nfile = src.n_file
        else:
            with rlock:
                img, info = self.pt._get_export_image(fp)
            if img is None:
                logging.warning("[canopy_finder] %s: the .ax replay failed (%s) -- raw pixels used",
                                os.path.basename(fp), (info or {}).get("error"))
                return self.read(fp, {}, x0, y0, x1, y1, bands, factor, zero_is_nodata)
            src = img
            H, W = (img.height, img.width) if hasattr(img, "read_window") else img.shape[:2]
            nfile = file_band_count(fp)
        bands = list(range(min(3, nfile))) if bands is None else [int(b) for b in bands if 0 <= int(b) < nfile]
        out = np.zeros((h, w, len(bands)), np.float32)
        real = np.zeros((h, w), bool)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 > cx0 and cy1 > cy0:
            if hasattr(src, "read_window"):
                with rlock:
                    win = src.read_window(cx0, cy0, cx1, cy1)
                block = np.dstack([np.asarray(win[b], np.float32) for b in bands])
            else:
                a = np.asarray(src[cy0:cy1, cx0:cx1])
                if a.ndim == 2:
                    a = a[:, :, None]
                block = a[:, :, bands].astype(np.float32)
            out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = block
            real[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = True
        v = real & np.isfinite(out).all(axis=2)
        if zero_is_nodata:
            v &= (np.nan_to_num(out) != 0).any(axis=2)
        return np.nan_to_num(out, nan=0.0), v, (1.0, 1.0), (float(x0), float(y0))


def _file_wh(fp):
    from . import canopy_finder_meta as M
    return M.image_size(fp)


def valid_mask(arr, real, nodata=None, zero_is_nodata=True):
    v = real.copy()
    a = arr
    if np.issubdtype(a.dtype, np.floating):
        v &= ~np.isnan(a).any(axis=2)
    if nodata is not None:
        try:
            v &= ~(a == np.asarray(nodata, dtype=a.dtype)).all(axis=2)
        except Exception:
            pass
    if zero_is_nodata:
        v &= (a != 0).any(axis=2)
    return v


def to_uint8(arr, valid, lo=0.5, hi=99.5):
    """Per-band percentile stretch to uint8 over the valid pixels."""
    a = arr.astype(np.float32)
    out = np.zeros(a.shape, np.uint8)
    for b in range(a.shape[2]):
        band = a[:, :, b]
        vals = band[valid] if valid is not None and valid.any() else band.ravel()
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        p0, p1 = np.percentile(vals, [lo, hi])
        if p1 <= p0:
            p1 = p0 + 1.0
        out[:, :, b] = np.clip((np.nan_to_num(band, nan=p0) - p0) * (255.0 / (p1 - p0)), 0, 255).astype(np.uint8)
    return out


# ════════════════════════════════════════════════════════════════════════
# Registration helpers (ported from the notebook, cell 0)
# ════════════════════════════════════════════════════════════════════════
def compute_polygon_bbox(points, img_h, img_w, cfg):
    """Notebook crop box with its adaptive context ring: (r0, c0, r1, c1),
    r1/c1 EXCLUSIVE. Not clipped to the image here (the window read pads)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return None
    c0, c1 = int(math.floor(pts[:, 0].min())), int(math.ceil(pts[:, 0].max()))
    r0, r1 = int(math.floor(pts[:, 1].min())), int(math.ceil(pts[:, 1].max()))
    span = max(r1 - r0 + 1, c1 - c0 + 1)
    base_r = int(np.ceil(span / 2.0))
    ring = max(cfg.get("min_context_ring", 60),
               min(int(round(base_r * cfg.get("context_ring_ratio", 1.0))),
                   cfg.get("max_context_ring", 200)))
    min_crop = 2 * base_r + 2 * ring
    base_buf = max(cfg.get("min_buffer", 30), int(round(base_r * cfg.get("buffer_ratio", 0.3))))
    buf = max(base_buf, max(0, (min_crop - 2 * base_r + 1) // 2))
    return (r0 - buf, c0 - buf, r1 + buf, c1 + buf)


def build_canopy_mask(points, bbox):
    r0, c0, r1, c1 = bbox
    local = np.asarray(points, dtype=np.float64) - (c0, r0)
    mask = np.zeros((r1 - r0, c1 - c0), np.uint8)
    cv2.fillPoly(mask, [np.round(local).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def to_reg_gray(img, band="all"):
    """Registration grey of a uint8 (h, w, C) image: B + 2R - G for colour
    (notebook default), the band itself for one band."""
    if img.ndim == 2:
        return img
    if isinstance(band, int) and 1 <= band <= img.shape[2]:
        return img[:, :, band - 1]
    if img.shape[2] < 3 or band == "mean":
        return img.mean(axis=2).astype(np.uint8)
    R = img[:, :, 0].astype(np.float32)
    Gc = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    return np.clip(B + 2.0 * R - Gc, 0, 255).astype(np.uint8)


def hist_match_cdf(source, reference, src_valid=None, ref_valid=None):
    """Per-channel CDF matching (notebook), computed on valid pixels only."""
    if source.dtype == np.uint8 and reference.dtype == np.uint8:
        return _hist_match_cdf_u8(source, reference, src_valid, ref_valid)
    src = source.astype(np.float64)
    out = np.zeros_like(src)
    for ch in range(source.shape[2]):
        s = src[:, :, ch]
        r = reference[:, :, ch].astype(np.float64)
        sv_all = s[src_valid] if src_valid is not None and src_valid.any() else s.ravel()
        rv_all = r[ref_valid] if ref_valid is not None and ref_valid.any() else r.ravel()
        if not sv_all.size or not rv_all.size:
            out[:, :, ch] = s
            continue
        rv, rc = np.unique(rv_all, return_counts=True)
        rcdf = np.cumsum(rc).astype(np.float64); rcdf /= rcdf[-1]
        sv, sc = np.unique(sv_all, return_counts=True)
        scdf = np.cumsum(sc).astype(np.float64); scdf /= scdf[-1]
        out[:, :, ch] = np.interp(s, sv, np.interp(scdf, rcdf, rv))
    return np.clip(out, 0, 255).astype(np.uint8)


_LEVELS = np.arange(256, dtype=np.float64)


def _hist_match_cdf_u8(source, reference, src_valid=None, ref_valid=None):
    """hist_match_cdf for uint8 images, identical output: the same CDFs from
    256-bin counts and the same interpolation, evaluated once per grey level and
    applied as a lookup table (np.unique + interp per pixel was two thirds of a
    registration's time)."""
    out = np.empty(source.shape, np.uint8)
    for ch in range(source.shape[2]):
        s = source[:, :, ch]
        r = reference[:, :, ch]
        sv_all = s[src_valid] if src_valid is not None and src_valid.any() else s.ravel()
        rv_all = r[ref_valid] if ref_valid is not None and ref_valid.any() else r.ravel()
        if not sv_all.size or not rv_all.size:
            out[:, :, ch] = s
            continue
        rc = np.bincount(rv_all, minlength=256)
        sc = np.bincount(sv_all, minlength=256)
        rv = np.nonzero(rc)[0].astype(np.float64)
        sv = np.nonzero(sc)[0].astype(np.float64)
        rcdf = np.cumsum(rc[rc > 0]).astype(np.float64); rcdf /= rcdf[-1]
        scdf = np.cumsum(sc[sc > 0]).astype(np.float64); scdf /= scdf[-1]
        lut = np.interp(_LEVELS, sv, np.interp(scdf, rcdf, rv))
        out[:, :, ch] = np.clip(lut, 0, 255).astype(np.uint8)[s]
    return out


def _ncc_local(a, b):
    if len(a) < 10:
        return -1.0
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return -1.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _combine(nw, nm, w):
    if nw < 0 and nm < 0:
        return -1.0
    if nm < 0:
        return nw
    if nw < 0:
        return nm
    return w * nm + (1 - w) * nw


def _gated_match(refA, S, R, ch, cw, gy0, gx0, minfrac):
    """matchTemplate where >= minfrac of the canopy box lands on real pixels."""
    th, tw = refA.shape
    if S.shape[0] < th or S.shape[1] < tw:
        return None
    res = cv2.matchTemplate(S, refA, cv2.TM_CCOEFF_NORMED)
    frac = cv2.boxFilter((R > 0.999).astype(np.float32), -1, (max(1, cw), max(1, ch)),
                         anchor=(0, 0), normalize=True, borderType=cv2.BORDER_CONSTANT)
    g = frac[gy0:gy0 + res.shape[0], gx0:gx0 + res.shape[1]]
    if g.shape != res.shape:
        return None
    keep = g >= minfrac
    if not keep.any():
        return None
    return np.where(keep, np.nan_to_num(res, nan=-1.0), -1.0).astype(np.float32)


def _affine_inv(M):
    A = np.vstack([M, [0.0, 0.0, 1.0]])
    return np.linalg.inv(A)[:2]


def _score_pose(ref_u8, ref_valid, tgt_u8, tgt_real, q2t, cmask, band, w):
    """Warp the target onto the crop through q2t (crop -> target coords) and
    score it like the notebook: masked NCC over real pixels / over the canopy."""
    rh, rw = ref_u8.shape[:2]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    al = cv2.warpAffine(tgt_u8, q2t, (rw, rh), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if al.ndim == 2:
        al = al[:, :, None]
    realm = cv2.warpAffine(tgt_real.astype(np.float32), q2t, (rw, rh), flags=flags,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    realm &= ref_valid
    try:
        al = hist_match_cdf(al, ref_u8, realm, ref_valid)
    except Exception:
        pass
    ag = to_reg_gray(al, band).astype(np.float64)
    rg = to_reg_gray(ref_u8, band).astype(np.float64)
    cm = cmask & realm
    cf = cm.sum() / float(max(1, cmask.sum()))
    nw = _ncc_local(ag[realm], rg[realm]) if realm.sum() >= 10 else -1.0
    nm = _ncc_local(ag[cm], rg[cm]) if cm.sum() >= 10 else -1.0
    return _combine(nw, nm, w), nw, nm, cf, al


def _rot_q2t(TW, TH, ang, FX, FY):
    """Crop -> target map of the notebook's pose: the target rotated by `ang`
    about its centre, then cut at (FX, FY): t = M^-1 (q + F)."""
    M = cv2.getRotationMatrix2D((TW / 2.0, TH / 2.0), float(ang), 1.0)
    Mi = _affine_inv(M)
    q2t = Mi.copy()
    q2t[:, 2] = Mi[:, :2] @ np.array([FX, FY], np.float64) + Mi[:, 2]
    return q2t


def exhaustive_align(ref_u8, ref_valid, tgt_u8, tgt_real, cmask, cfg, cancel=None):
    """The notebook's brute-force engine on a search image: coarse (downscaled)
    gated matchTemplate over a rotation grid, top-N peaks refined at full
    resolution in finer angle steps, winner = best combined masked NCC among
    poses keeping >= min_canopy_frac of the crown on real pixels.
    Returns (q2t 2x3, info)."""
    band = cfg.get("reg_band", "all")
    AR = float(cfg.get("angle_range", 5.0))
    AS = max(0.05, float(cfg.get("angle_step", 1.0)))
    RS = max(0.05, float(cfg.get("refine_step", 0.5)))
    topn = max(1, int(cfg.get("topn", 5)))
    minfrac = float(cfg.get("min_canopy_frac", 0.5))
    w = float(cfg.get("ncc_mask_weight", 0.2))
    cds = max(1, int(cfg.get("coarse_downscale", 2)))
    TH, TW = tgt_u8.shape[:2]
    big = max(TH, TW)
    WS = max(1, int(math.ceil(big / float(max(64, int(cfg.get("work_max_px", 900)))))))
    cds_eff = cds * WS
    rh, rw = ref_u8.shape[:2]
    ref_g = to_reg_gray(ref_u8, band).astype(np.float32)
    rgm = float(ref_g[ref_valid].mean()) if ref_valid.any() else float(ref_g.mean())
    ref_g = np.where(ref_valid, ref_g, rgm).astype(np.float32)
    tg = to_reg_gray(tgt_u8, band).astype(np.float32)
    real = tgt_real.astype(np.float32)
    tmean = float(tg[tgt_real].mean()) if tgt_real.any() else 0.0
    tg = np.where(tgt_real, tg, tmean).astype(np.float32)
    ys, xs = np.where(cmask)
    if len(ys):
        cy0, cy1, cx0, cx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        cy0, cy1, cx0, cx1 = 0, rh, 0, rw

    def _topk(res, k, mind):
        out, r = [], res.copy()
        for _ in range(k):
            _, mv, _, ml = cv2.minMaxLoc(r)
            if mv <= -1.0:
                break
            out.append((float(mv), ml[0], ml[1]))
            x, y = ml
            r[max(0, y - mind):y + mind + 1, max(0, x - mind):x + mind + 1] = -2.0
        return out

    # ---- coarse ----
    ref_c = cv2.resize(ref_g, (max(1, rw // cds_eff), max(1, rh // cds_eff)), interpolation=cv2.INTER_AREA) \
        if cds_eff > 1 else ref_g
    rch, rcw = ref_c.shape
    P = max(rch, rcw)
    ccy0, ccx0 = cy0 // cds_eff, cx0 // cds_eff
    cch, ccw = max(1, (cy1 - cy0) // cds_eff), max(1, (cx1 - cx0) // cds_eff)
    a0 = float(cfg.get("angle_center", 0.0))
    angles = (np.arange(-AR, AR + 1e-6, AS) if AR > 0 else np.array([0.0])) + a0
    cand = []
    # coarse: rotate the DOWNSCALED image (about the same centre, scaled)
    if cds_eff > 1:
        TWc, THc = max(1, TW // cds_eff), max(1, TH // cds_eff)
        tg_c = cv2.resize(tg, (TWc, THc), interpolation=cv2.INTER_AREA)
        real_c = cv2.resize(real, (TWc, THc), interpolation=cv2.INTER_AREA)
    else:
        TWc, THc, tg_c, real_c = TW, TH, tg, real
    ctr_c = (TW / 2.0 / cds_eff, TH / 2.0 / cds_eff)
    for a in angles:
        if cancel is not None and cancel():
            raise _Cancelled()
        Mc = cv2.getRotationMatrix2D(ctr_c, float(a), 1.0)
        S = cv2.warpAffine(tg_c, Mc, (TWc, THc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=tmean)
        R = cv2.warpAffine(real_c, Mc, (TWc, THc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        Sp = cv2.copyMakeBorder(S, P, P, P, P, cv2.BORDER_CONSTANT, value=float(tmean))
        Rp = cv2.copyMakeBorder(R, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
        res = _gated_match(ref_c, Sp, Rp, cch, ccw, ccy0, ccx0, minfrac)
        if res is None:
            continue
        for corr, xx, yy in _topk(res, topn, max(rch, rcw) // 2):
            cand.append((corr, float(a), (xx - P) * cds_eff, (yy - P) * cds_eff))
    cand.sort(reverse=True)

    # ---- fine (full resolution, around each candidate) ----
    Pf = max(rh, rw) + 2
    best = fallback = None
    half = max(AS, RS) if AR > 0 else 0.0
    for corr0, a, FXc, FYc in (cand[:topn] or [(0.0, a0, (TW - rw) // 2, (TH - rh) // 2)]):
        for ang in (np.arange(a - half, a + half + 1e-6, RS) if half > 0 else [a]):
            if cancel is not None and cancel():
                raise _Cancelled()
            # only the window around the candidate, rotated directly (same pixels as
            # cutting it out of the padded, rotated image -- at O(window) cost)
            m = cds_eff * 2
            wx, wy = FXc + Pf, FYc + Pf
            x0, y0 = max(0, wx - m), max(0, wy - m)
            x1, y1 = min(TW + 2 * Pf, wx + rw + m), min(TH + 2 * Pf, wy + rh + m)
            if x1 <= x0 or y1 <= y0:
                continue
            Ms = cv2.getRotationMatrix2D((TW / 2.0, TH / 2.0), float(ang), 1.0)
            Ms[0, 2] -= (x0 - Pf)
            Ms[1, 2] -= (y0 - Pf)
            sub = cv2.warpAffine(tg, Ms, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=tmean)
            subr = cv2.warpAffine(real, Ms, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            res = _gated_match(ref_g, sub, subr, cy1 - cy0, cx1 - cx0, cy0, cx0, minfrac)
            if res is None:
                continue
            _, mv, _, ml = cv2.minMaxLoc(res)
            FX, FY = (x0 + ml[0]) - Pf, (y0 + ml[1]) - Pf
            q2t = _rot_q2t(TW, TH, ang, FX, FY)
            comb, nw, nm, cf, _al = _score_pose(ref_u8, ref_valid, tgt_u8, tgt_real, q2t, cmask, band, w)
            entry = (comb, q2t, nw, nm, cf, float(ang), float(mv))
            if cf >= minfrac and (best is None or comb > best[0]):
                best = entry
            if fallback is None or cf > fallback[4]:
                fallback = entry
    pick = best or fallback
    if pick is None:
        raise RuntimeError("no admissible position: the crown never lands on real target pixels")
    comb, q2t, nw, nm, cf, ang, mv = pick
    return q2t, {"angle": ang, "ncc_window": nw, "ncc_mask": nm, "score": comb,
                 "canopy_frac": cf, "match_corr": mv, "accepted": best is not None}


def stackreg_align(ref_u8, ref_valid, tgt_u8, tgt_real, cmask, cfg, cancel=None):
    """Notebook use_stackreg path: coarse gated matchTemplate (no rotation)
    recentres the crop, then pystackreg RIGID_BODY on the registration grey."""
    from pystackreg import StackReg
    band = cfg.get("reg_band", "all")
    minfrac = float(cfg.get("min_canopy_frac", 0.5))
    w = float(cfg.get("ncc_mask_weight", 0.2))
    cds = max(1, int(cfg.get("coarse_downscale", 2)))
    TH, TW = tgt_u8.shape[:2]
    rh, rw = ref_u8.shape[:2]
    ref_g = to_reg_gray(ref_u8, band).astype(np.float32)
    rgm = float(ref_g[ref_valid].mean()) if ref_valid.any() else float(ref_g.mean())
    ref_g = np.where(ref_valid, ref_g, rgm).astype(np.float32)
    tg = to_reg_gray(tgt_u8, band).astype(np.float32)
    tmean = float(tg[tgt_real].mean()) if tgt_real.any() else 0.0
    tg = np.where(tgt_real, tg, tmean).astype(np.float32)
    ys, xs = np.where(cmask)
    cy0, cy1, cx0, cx1 = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1) \
        if len(ys) else (0, rh, 0, rw)
    rc = cv2.resize(ref_g, (max(1, rw // cds), max(1, rh // cds)), interpolation=cv2.INTER_AREA)
    Sc = cv2.resize(tg, (max(1, TW // cds), max(1, TH // cds)), interpolation=cv2.INTER_AREA)
    Rc = cv2.resize(tgt_real.astype(np.float32), (Sc.shape[1], Sc.shape[0]), interpolation=cv2.INTER_AREA)
    res = _gated_match(rc, Sc, Rc, max(1, (cy1 - cy0) // cds), max(1, (cx1 - cx0) // cds),
                       cy0 // cds, cx0 // cds, minfrac)
    if res is None:
        FX, FY, mv = (TW - rw) // 2, (TH - rh) // 2, 0.0
    else:
        _, mv, _, ml = cv2.minMaxLoc(res)
        FX, FY = ml[0] * cds, ml[1] * cds
    FX = int(min(max(0, FX), max(0, TW - rw)))
    FY = int(min(max(0, FY), max(0, TH - rh)))
    if cancel is not None and cancel():
        raise _Cancelled()
    center = tg[FY:FY + rh, FX:FX + rw].astype(np.float64) + 1.0
    if center.shape != ref_g.shape:
        pad = np.full(ref_g.shape, tmean + 1.0)
        pad[:center.shape[0], :center.shape[1]] = center
        center = pad
    sr = StackReg(StackReg.RIGID_BODY)
    tmat = sr.register(ref_g.astype(np.float64) + 1.0, center)     # crop q -> center-crop tmat.q
    q2t = np.asarray(tmat, np.float64)[:2].copy()
    q2t[:, 2] += (FX, FY)
    comb, nw, nm, cf, _ = _score_pose(ref_u8, ref_valid, tgt_u8, tgt_real, q2t, cmask, band, w)
    return q2t, {"angle": math.degrees(math.atan2(q2t[1, 0], q2t[0, 0])), "ncc_window": nw, "ncc_mask": nm, "score": comb,
                 "canopy_frac": cf, "match_corr": float(mv), "accepted": cf >= minfrac}


class _Cancelled(Exception):
    pass


# ════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════
@dataclass
class SourceContext:
    fp: str
    group: str
    geo: GeoRef
    raw_pts: np.ndarray          # (N, 2) source raw px
    bbox: tuple                  # (r0, c0, r1, c1) source raw px, exclusive
    crop_u8: np.ndarray
    crop_valid: np.ndarray
    cmask: np.ndarray
    bands: list
    gsd: tuple                   # (m, m) per source pixel
    shape_type: str = "polygon"
    ax: dict = None              # the source's .ax (its frame is the post-.ax frame)


@dataclass
class Result:
    target_fp: str
    status: str                  # found | low_score | geo | outside_footprint | outside_frame | no_georef | error
    message: str = ""
    raw_pts: object = None       # (N, 2) target raw px
    geo_pts: object = None       # (N, 2) target raw px of the plain geo projection
    record_points: object = None
    image_ref_size: object = None
    square: object = None        # (x0, y0, x1, y1) registration square, target raw px
    offset_px: tuple = (0.0, 0.0)
    offset_m: float = 0.0
    angle: float = 0.0
    score: object = None
    ncc_window: object = None
    ncc_mask: object = None
    method: str = ""
    seconds: float = 0.0
    q2raw: object = None         # 2x3: source-crop coords -> target raw px
    extra: dict = field(default_factory=dict)

    @property
    def has_polygon(self):
        return self.raw_pts is not None and self.status in ("found", "low_score", "geo")


def merged_cfg(cfg=None):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None or k == "bands"})
    return out


def _match_bands(n, cfg):
    b = cfg.get("bands")
    if b:
        return [int(x) for x in b if 0 <= int(x) < n]
    return list(range(min(3, n)))


_DEFAULT_FRAMES = Frames()


def frames_of(cfg):
    return (cfg or {}).get("frames") or _DEFAULT_FRAMES


def prepare_source(fp, rec, ax, cfg=None, group=""):
    """Everything about the source crown the targets need (read once). All
    coordinates are in the source's post-.ax frame (what its viewer shows)."""
    cfg = merged_cfg(cfg)
    ax = ax or {}
    geo = frame_for(fp, ax=ax)
    if geo is None:
        raise ValueError(f"{os.path.basename(fp)} cannot be read")
    raw_pts = geo.from_record(rec)
    bbox = compute_polygon_bbox(raw_pts, geo.raw_h, geo.raw_w, cfg)
    r0, c0, r1, c1 = bbox
    bands = _match_bands(file_band_count(fp), cfg)
    arr, valid, _f, _o = frames_of(cfg).read(fp, ax, c0, r0, c1, r1, bands,
                                             zero_is_nodata=cfg.get("zero_is_nodata", True))
    cmask = build_canopy_mask(raw_pts, bbox)
    if cmask.any() and (valid & cmask).sum() < 0.5 * cmask.sum():
        raise ValueError(f"the crown lies on no-data pixels of {os.path.basename(fp)}")
    cx, cy = raw_pts.mean(axis=0)
    return SourceContext(fp=fp, group=group, geo=geo, raw_pts=raw_pts, bbox=bbox,
                         crop_u8=to_uint8(arr, valid), crop_valid=valid,
                         cmask=build_canopy_mask(raw_pts, bbox), bands=bands,
                         gsd=gsd_m(geo, cx, cy),
                         shape_type=str(rec.get("type", "polygon") or "polygon"), ax=ax)


def pair_kind(a, b):
    """'pixel' (two non-located frames), 'exif' (an EXIF estimate on either
    side) or 'geotiff'; None when one is located and the other is not."""
    if a.located != b.located:
        return None
    if not a.located:
        return "pixel"
    return "exif" if "exif" in (a.kind, b.kind) else "geotiff"


def _affine_scaled(A, sigma, c):
    """A applied after scaling about source point c by sigma."""
    out = A.copy()
    out[:, :2] = A[:, :2] * sigma
    out[:, 2] = A[:, :2] @ ((1.0 - sigma) * np.asarray(c, np.float64)) + A[:, 2]
    return out


def find_in_target(src, target_fp, target_ax=None, cfg=None, cancel=None):
    """Locate the source crown in one target image. Never raises: problems come
    back as a Result status.

    Matching happens on a WORK grid: the source grid, made coarser when the
    target is coarser (both images then meet at the coarser resolution -- an
    orthomosaic is never upsampled to a 2 cm raw frame, a raw frame is reduced to
    the mosaic's 15 cm). The target is read (COG: from the overview level
    nearest that resolution) and resampled into the work grid through the geo
    prior, so only a residual shift + rotation (+ scale, for EXIF frames) is
    left to the registration."""
    t0 = time.perf_counter()
    cfg = merged_cfg(cfg)
    method = str(cfg.get("method", "exhaustive")).lower()
    res = Result(target_fp=target_fp, status="error", method=method)
    try:
        target_ax = target_ax or {}
        frames = frames_of(cfg)
        tg = frame_for(target_fp, ax=target_ax)
        if tg is None:
            res.status, res.message = "no_georef", "cannot read this image"
            return res
        kind = pair_kind(src.geo, tg)
        res.extra["pair"] = kind
        if kind is None:
            res.status = "no_georef"
            res.message = ("the source is located but this image is not" if src.geo.located
                           else "this image is located but the source is not")
            return res
        if (src.geo.epsg is None) != (tg.epsg is None) and kind != "pixel":
            logging.warning("[canopy_finder] CRS unknown on one side (%s / %s): assumed equal",
                            src.geo.epsg, tg.epsg)
        r0, c0, r1, c1 = src.bbox
        ch, cw = r1 - r0, c1 - c0
        if kind == "pixel":
            Rsrc = float(cfg.get("search_radius_pct", 15.0)) / 100.0 * max(src.geo.raw_w, src.geo.raw_h)
            AR, SR = float(cfg.get("angle_range_pixel", 2.0)), None
        elif kind == "exif":
            Rsrc = float(cfg.get("search_radius_exif_m", 40.0)) / max(1e-6, min(src.gsd))
            AR = float(cfg.get("angle_range_exif", 20.0))
            if "no heading" in (src.geo.note or "") or "no heading" in (tg.note or ""):
                AR = 180.0
            SR = (float(cfg.get("scale_min_exif", 0.7)), float(cfg.get("scale_max_exif", 1.4)))
        else:
            Rsrc = float(cfg.get("search_radius_m", 15.0)) / max(1e-6, min(src.gsd))
            AR, SR = float(cfg.get("angle_range", 5.0)), None
        for note in (src.geo.note, tg.note):
            if note and kind == "exif":
                res.message = (res.message + "; " if res.message else "") + note
        A0 = local_affine(src.geo, tg, c0 - Rsrc, r0 - Rsrc, c1 + Rsrc, r1 + Rsrc)
        geo_pts = map_px(src.geo, tg, src.raw_pts)
        res.geo_pts = geo_pts
        res.extra["ax"], res.extra["frame"] = target_ax, tg
        gcx, gcy = geo_pts.mean(axis=0)
        if not (0 <= gcx <= tg.raw_w and 0 <= gcy <= tg.raw_h):
            b = tg.base
            bx, by = map_px(src.geo, b, [src.raw_pts.mean(axis=0)])[0] if b is not None else (-1, -1)
            if b is not None and 0 <= bx <= b.raw_w and 0 <= by <= b.raw_h:
                res.status, res.message = "outside_frame", "in the file, but outside this image's crop (.ax)"
            else:
                res.status, res.message = "outside_footprint", "the crown is outside this image"
            return res

        # ---- the work grid ----
        f = math.sqrt(abs(np.linalg.det(A0[:, :2])))           # target px per source px
        s = max(1.0, 1.0 / max(f, 1e-9))                        # source px per work px
        # two fine images (raw <-> raw, ~2.6 cm) searched +-40 m made a 4500 px search
        # image matched at full detail for every scale (5 min / frame): cap the search
        # image, but keep >= ~24 work px across the crown
        crown_src = float(np.ptp(src.raw_pts, axis=0).max())
        cap = (max(cw, ch) + 2.0 * Rsrc) / float(cfg.get("search_max_px", 1600))
        s = max(s, min(cap, crown_src / 24.0))
        cw_w, ch_w = max(8, int(round(cw / s))), max(8, int(round(ch / s)))
        sxw, syw = cw / float(cw_w), ch / float(ch_w)
        q_pts = (src.raw_pts - (c0, r0)) / (sxw, syw)            # crown in work-crop coords
        if s > 1.0 + 1e-6:
            crop_w = cv2.resize(src.crop_u8, (cw_w, ch_w), interpolation=cv2.INTER_AREA)
            if crop_w.ndim == 2:
                crop_w = crop_w[:, :, None]
            valid_w = cv2.resize(src.crop_valid.astype(np.float32), (cw_w, ch_w),
                                 interpolation=cv2.INTER_AREA) > 0.999
            cmask_w = build_canopy_mask(q_pts, (0, 0, ch_w, cw_w))
        else:
            crop_w, valid_w, cmask_w = src.crop_u8, src.crop_valid, src.cmask
        crown_px = float(np.ptp(q_pts, axis=0).max())
        res.extra["work_gsd_m"] = float(min(src.gsd)) * s if kind != "pixel" else None
        res.extra["crown_work_px"] = crown_px
        if crown_px < float(cfg.get("min_crown_px", 12)):
            res.message = (res.message + "; " if res.message else "") + \
                f"the crown is only {crown_px:.0f} px at this resolution"
        Dg = np.diag([sxw, syw])
        Rw = int(math.ceil(Rsrc / s))
        o = np.array([c0 - Rw * sxw, r0 - Rw * syw], np.float64)   # source raw of work u = 0
        SW, SH = cw_w + 2 * Rw, ch_w + 2 * Rw
        centre = src.raw_pts.mean(axis=0)

        if method == "geo":
            raw_t = geo_pts
            q2raw = np.empty((2, 3))
            q2raw[:, :2] = A0[:, :2] @ Dg
            q2raw[:, 2] = A0[:, :2] @ np.array([c0, r0], np.float64) + A0[:, 2]
            info = {"angle": 0.0, "score": None, "ncc_window": None, "ncc_mask": None}
            status = "geo"
        else:
            sigmas = [1.0]
            if SR is not None:
                step = max(0.01, float(cfg.get("scale_step", 0.05)))
                lo, hi = min(SR), max(SR)
                ks = range(int(math.floor((lo - 1.0) / step + 1e-9)), int(math.ceil((hi - 1.0) / step - 1e-9)) + 1)
                sigmas = sorted({round(1.0 + k * step, 6) for k in ks if lo - 1e-9 <= 1.0 + k * step <= hi + 1e-9}
                                | {1.0}, key=lambda v: abs(v - 1.0))
            # one target window covering every scale tried
            corners_u = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], np.float64)
            corners_raw = corners_u @ Dg.T + o
            tcs = []
            for sg in (min(sigmas), max(sigmas)):
                Asg = _affine_scaled(A0, sg, centre)
                tcs.append(corners_raw @ Asg[:, :2].T + Asg[:, 2])
            tc = np.vstack(tcs)
            wx0, wy0 = np.floor(tc.min(axis=0)) - 2
            wx1, wy1 = np.ceil(tc.max(axis=0)) + 2
            tb = _match_bands(file_band_count(target_fp), {"bands": src.bands})
            if len(tb) != len(src.bands):
                tb = _match_bands(file_band_count(target_fp), {"bands": None})
            f_t = f * s                                          # target px per work px (>= ~1)
            win, wvalid, (fx, fy), (ox, oy) = frames.read(target_fp, target_ax, wx0, wy0, wx1, wy1, tb,
                                                          factor=f_t if f_t > 1.5 else 1.0,
                                                          zero_is_nodata=cfg.get("zero_is_nodata", True))
            k = f_t / fx
            if k > 1.25 and win.shape[0] > 2 and win.shape[1] > 2:
                nw_, nh_ = max(2, int(round(win.shape[1] / k))), max(2, int(round(win.shape[0] / k)))
                kx, ky = win.shape[1] / float(nw_), win.shape[0] / float(nh_)
                win = np.stack([cv2.resize(win[:, :, b].astype(np.float32), (nw_, nh_),
                                           interpolation=cv2.INTER_AREA) for b in range(win.shape[2])], axis=2)
                wvalid = cv2.resize(wvalid.astype(np.float32), (nw_, nh_), interpolation=cv2.INTER_AREA) > 0.999
                fx, fy = fx * kx, fy * ky
            Fi = np.diag([1.0 / fx, 1.0 / fy])
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            # StackReg converges from a small rotation only: with no heading, search the circle
            use_sr = method == "stackreg" and AR <= 45.0
            if method == "stackreg" and not use_sr:
                res.message = (res.message + "; " if res.message else "") + \
                    "heading unknown: exhaustive rotation search used"
            engine = stackreg_align if use_sr else exhaustive_align
            ecfg = dict(cfg, angle_range=AR)
            best = None
            for sg in sigmas:
                if cancel is not None and cancel():
                    raise _Cancelled()
                Asg = _affine_scaled(A0, sg, centre)
                # work u -> window px:  Fi (Asg (Dg u + o) - win origin)
                Aw = np.empty((2, 3))
                Aw[:, :2] = Fi @ Asg[:, :2] @ Dg
                Aw[:, 2] = Fi @ (Asg[:, :2] @ o + Asg[:, 2] - (ox, oy))
                T = np.stack([cv2.warpAffine(win[:, :, b].astype(np.float32), Aw, (SW, SH), flags=flags,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                              for b in range(win.shape[2])], axis=2)
                Treal = cv2.warpAffine(wvalid.astype(np.float32), Aw, (SW, SH), flags=flags,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
                if not Treal.any():
                    continue
                if sg == 1.0 and kind == "geotiff":
                    # a mosaic with no data where the crown projects (its edge, a
                    # gap): not in this image. The engines otherwise reported a
                    # meaningless -1 (StackReg) or "found" a patch tens of metres
                    # away (exhaustive) -- and that outline was saved.
                    under = Treal[Rw:Rw + ch_w, Rw:Rw + cw_w]
                    cm = cmask_w[:under.shape[0], :under.shape[1]]
                    if cm.any() and float(under[cm].mean()) < 0.25:
                        res.status = "outside_footprint"
                        res.message = (f"no data under the crown in this image ({100 * float(under[cm].mean()):.0f} % "
                                       f"valid: mosaic edge or gap)")
                        return res
                T8 = to_uint8(T, Treal)
                src_u8 = crop_w
                if src_u8.shape[2] != T8.shape[2]:
                    kk = min(src_u8.shape[2], T8.shape[2])
                    src_u8, T8 = src_u8[:, :, :kk], T8[:, :, :kk]
                try:
                    q2t, inf = engine(np.ascontiguousarray(src_u8), valid_w, np.ascontiguousarray(T8),
                                      Treal, cmask_w, ecfg, cancel)
                except RuntimeError:
                    continue
                if sg == 1.0 and len(sigmas) > 1 and method != "stackreg":
                    # the other scales only refine around the rotation found at scale 1
                    ecfg = dict(ecfg, angle_center=float(inf.get("angle") or 0.0),
                                angle_range=max(float(cfg.get("angle_step", 1.0)), 1.0))
                key = (bool(inf.get("accepted", True)), inf.get("score") if inf.get("score") is not None else -2)
                if best is None or key > best[0]:
                    best = (key, sg, Asg, q2t, inf, T8, src_u8)
            if best is None:
                res.status, res.message = "outside_footprint", "no valid target pixels around the crown"
                return res
            _key_, sg, Asg, q2t, info, T8, src_u8 = best
            # one sign convention for both engines: the crown's rotation in the
            # target relative to the source, in the work grid (y down)
            info["angle"] = math.degrees(math.atan2(q2t[1, 0], q2t[0, 0]))
            info["scale"] = sg
            # work crop q -> search u -> source raw -> target raw:  Asg (Dg (q2t q) + o)
            q2raw = np.empty((2, 3))
            q2raw[:, :2] = Asg[:, :2] @ Dg @ q2t[:, :2]
            q2raw[:, 2] = Asg[:, :2] @ (Dg @ q2t[:, 2] + o) + Asg[:, 2]
            raw_t = q_pts @ q2raw[:, :2].T + q2raw[:, 2]
            score = info.get("score")
            status = "found" if (score is not None and score >= float(cfg.get("min_ncc", 0.3))
                                 and info.get("accepted", True)) else "low_score"
            res.extra["aligned_u8"] = cv2.warpAffine(T8, q2t, (cw_w, ch_w), flags=flags,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            res.extra["source_u8"] = src_u8
            res.extra["scale"] = sg

        res.raw_pts = raw_t
        res.q2raw = q2raw
        res.extra["work_scale"] = (sxw, syw)
        qc = np.array([[0, 0], [cw_w, 0], [cw_w, ch_w], [0, ch_w]], np.float64)
        sq = qc @ q2raw[:, :2].T + q2raw[:, 2]
        res.square = (int(max(0, math.floor(sq[:, 0].min()))), int(max(0, math.floor(sq[:, 1].min()))),
                      int(min(tg.raw_w, math.ceil(sq[:, 0].max()))), int(min(tg.raw_h, math.ceil(sq[:, 1].max()))))
        d = raw_t.mean(axis=0) - geo_pts.mean(axis=0)
        res.offset_px = (float(d[0]), float(d[1]))
        tgsd = gsd_m(tg, *raw_t.mean(axis=0))
        res.offset_m = float(math.hypot(d[0] * tgsd[0], d[1] * tgsd[1])) if tg.located else 0.0
        res.angle = float(info.get("angle") or 0.0)
        res.score = info.get("score")
        res.ncc_window, res.ncc_mask = info.get("ncc_window"), info.get("ncc_mask")
        pts, ref, inside = tg.to_record(raw_t)
        res.record_points, res.image_ref_size = pts, ref
        res.status = status if inside else "outside_frame"
        if not inside:
            res.message = "found, but outside this image's crop (.ax)"
        return res
    except _Cancelled:
        res.status, res.message = "error", "cancelled"
        return res
    except NotLocated as e:
        res.status, res.message = "no_georef", str(e)
        return res
    except Exception as e:
        logging.exception("[canopy_finder] %s failed", target_fp)
        res.status, res.message = "error", str(e)
        return res
    finally:
        res.seconds = time.perf_counter() - t0


# ════════════════════════════════════════════════════════════════════════
# Searches in worker processes
# ════════════════════════════════════════════════════════════════════════
# A search holds Python's GIL for much of its time (pystackreg's C code does not
# release it; the percentile stretch of every scale is numpy glue), so threads
# reach ~2x however many cores there are. Separate processes do not share a
# GIL. Each worker process loads only the finder modules -- not the app, not
# Qt (~0.7 s, ~60 MB) -- runs at below-normal priority with OpenCV
# single-threaded, and reads the EXIF / ground-size cache the parent saved
# (frames are located in the parent first: one measurement per flight).
def processes_available():
    """Worker processes are used when CanoPie runs from its sources. A frozen /
    Nuitka build would start its own .exe as each worker (a second app unless
    the build supports multiprocessing): it searches in threads."""
    main = sys.modules.get("__main__")
    return not (getattr(sys, "frozen", False) or "__compiled__" in globals()
                or "__compiled__" in getattr(main, "__dict__", {}))


def search_in_process_ok(frames, fp, ax):
    """A target a worker process can search: its pixels come straight from the
    file (no .ax replay that needs the open project)."""
    try:
        return frames.mode(fp, ax or {}) == "raw"
    except Exception:
        return False


def _process_boot(cache_dir):
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg = __name__.rsplit(".", 1)[0]
    lines = ["import os, sys, types",
             "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):",
             "    os.environ.setdefault(_v, '1')"]
    if os.path.isfile(os.path.join(pkg_dir, "canopy_finder.py")):
        # the package without its __init__ (which imports the whole app)
        lines += [f"if {pkg!r} not in sys.modules:",
                  f"    _p = types.ModuleType({pkg!r}); _p.__path__ = [{pkg_dir!r}]; sys.modules[{pkg!r}] = _p"]
    lines += [f"from {pkg} import canopy_finder as _cf", f"_cf._process_init({cache_dir!r})"]
    return "\n".join(lines) + "\n"


_K32 = None


def _kernel32():
    """kernel32 with HANDLE-typed calls -- a private instance: ctypes' default int
    return truncates the pseudo handles of GetCurrentProcess / GetCurrentThread
    on 64-bit, and the shared windll functions must not be retyped for others."""
    global _K32
    if _K32 is None:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.GetCurrentThread.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k.GetPriorityClass.restype = wintypes.DWORD
        k.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        k.GetThreadPriority.argtypes = [wintypes.HANDLE]
        _K32 = k
    return _K32


def lower_priority(process=False):
    """Below-normal priority for the calling thread (or, process=True, the whole
    process) on Windows, so a long run never makes the computer sluggish."""
    if os.name != "nt":
        return False
    try:
        k = _kernel32()
        if process:
            return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x4000))    # BELOW_NORMAL_PRIORITY_CLASS
        return bool(k.SetThreadPriority(k.GetCurrentThread(), -1))            # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:
        return False


def _process_init(cache_dir):
    global _FULL_BYTES
    lower_priority(process=True)
    cv2.setNumThreads(1)
    from . import canopy_finder_meta as M
    M.set_shared_cache_dir(cache_dir)
    _FULL_BYTES = 384 * 1024 * 1024


def _process_search(ctx, fp, ax, cfg):
    return find_in_target(ctx, fp, ax, cfg)


def _process_pair_shift(a_fp, b_fp):
    return _pair_shift(a_fp, b_fp)


def _process_probe():
    """What a worker process looks like (tests)."""
    from . import canopy_finder_meta as M
    pri = None
    if os.name == "nt":
        try:
            pri = int(_kernel32().GetPriorityClass(_kernel32().GetCurrentProcess()))
        except Exception:
            pass
    return {"pid": os.getpid(), "qt": any(m.startswith("PyQt5") for m in sys.modules),
            "app": any(m.endswith(("project_tab", "main_window")) for m in sys.modules),
            "cv_threads": cv2.getNumThreads(), "cache": M.shared_cache_dir(), "priority": pri}


_MAIN_LOCK = threading.Lock()


@contextlib.contextmanager
def _main_not_reimported():
    """A spawned process re-imports the parent's __main__ (main.py -> the whole
    app, ~2 s and ~160 MB each); hide it while the workers start. Frozen /
    compiled builds start their workers their own way: left alone."""
    main = sys.modules.get("__main__")
    d = getattr(main, "__dict__", None)
    if d is None or getattr(sys, "frozen", False) or "__compiled__" in d:
        yield
        return
    with _MAIN_LOCK:                             # several threads submit at once
        saved = {k: d[k] for k in ("__file__", "__spec__") if k in d}
        d.pop("__file__", None)
        d["__spec__"] = None
        try:
            yield
        finally:
            d.pop("__spec__", None)
            d.update(saved)


class ProcessSearch:
    """find_in_target in `workers` processes. submit() -> Future of a Result."""

    def __init__(self, workers):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from . import canopy_finder_meta as M
        if not processes_available():
            raise RuntimeError("worker processes are not used in a frozen build")
        M.save()                                 # the workers read what the parent knows
        self.ex = ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context("spawn"),
                                      initializer=functools.partial(exec, _process_boot(M.shared_cache_dir()), {}))

    def submit(self, fn, *args):
        with _main_not_reimported():
            return self.ex.submit(fn, *args)

    def pair_shifts(self, pairs):
        """The pair registrations of a flight-line measurement, in the workers
        (see pair_shifts_with). A pair the pool could not run is registered here
        -- a broken pool must not be cached as a failed measurement."""
        from concurrent.futures import BrokenExecutor
        futs = []
        for a, b in pairs:
            try:
                futs.append(self.submit(_process_pair_shift, a, b))
            except Exception as e:
                futs.append(e)
        out = []
        for (a, b), f in zip(pairs, futs):
            try:
                if isinstance(f, BaseException):
                    raise f
                out.append(f.result())
            except (BrokenExecutor, RuntimeError):
                try:
                    out.append(_pair_shift(a, b))
                except Exception as e:
                    out.append(e)
            except Exception as e:
                out.append(e)
        return out

    def search(self, ctx, fp, ax, cfg):
        cfg = {k: v for k, v in (cfg or {}).items() if k != "frames" and not callable(v)}
        return self.submit(_process_search, ctx, fp, ax or {}, cfg)

    def close(self, cancel=False):
        if cancel:
            for p in list((getattr(self.ex, "_processes", None) or {}).values()):
                try:
                    p.terminate()
                except Exception:
                    pass
        try:
            self.ex.shutdown(wait=not cancel, cancel_futures=cancel)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════
# Export
# ════════════════════════════════════════════════════════════════════════
def write_geotiff_crop(src_fp, arr, x0, y0, out_path, geo=None):
    """Write `arr` (h, w, C), cut at (x0, y0) of the frame `geo` (default: the
    file's own GeoTIFF frame), as a GeoTIFF (see _geo_extratags)."""
    import tifffile
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, :, None]
    photometric = "rgb" if (a.shape[2] == 3 and a.dtype == np.uint8) else "minisblack"
    data = a[:, :, 0] if a.shape[2] == 1 else a
    kw = {} if a.shape[2] == 1 else {"planarconfig": "contig"}
    tifffile.imwrite(out_path, data, photometric=photometric, compression="deflate",
                     extratags=_geo_extratags(src_fp, x0, y0, geo), **kw)
    return out_path


def _geo_extratags(src_fp, x0, y0, geo=None):
    """GeoTIFF tags of a window cut at (x0, y0) of the frame `geo`: the source's
    GeoKeys (EPSG:4326 for an EXIF-located frame) and the frame's transform
    shifted to the window -- a ModelTransformation when the .ax rotated it. A
    not-located frame gets none."""
    import tifffile
    own = georef(src_fp)
    geo = geo if geo is not None else own
    extratags = []
    if geo is not None and geo.kind == "exif":
        keys = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
        extratags.append((34735, "H", len(keys), tuple(keys), True))
    elif geo is not None and geo.kind == "geotiff" and own is not None:
        with tifffile.TiffFile(src_fp) as tif:
            tags = tif.pages[0].tags
            for code, dtype in ((34735, "H"), (34736, "d"), (34737, "s"), (42113, "s"), (42112, "s")):
                t = tags.get(code)
                if t is not None:
                    v = t.value
                    if code == 34737 and isinstance(v, bytes):
                        v = v.decode("ascii", "replace")
                    extratags.append((code, dtype, 0 if dtype == "s" else len(v), v, True))
    if geo is not None and geo.located:
        t = geo.transform
        X0 = t[0] + x0 * t[1] + y0 * t[2]
        Y0 = t[3] + x0 * t[4] + y0 * t[5]
        if abs(t[2]) > 1e-15 or abs(t[4]) > 1e-15 or t[1] < 0 or t[5] > 0:
            M = (t[1], t[2], 0.0, X0, t[4], t[5], 0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            extratags.append((34264, "d", 16, M, True))
        else:
            extratags.append((33550, "d", 3, (abs(t[1]), abs(t[5]), 0.0), True))
            extratags.append((33922, "d", 6, (0.0, 0.0, 0.0, X0, Y0, 0.0), True))
    return extratags


# ════════════════════════════════════════════════════════════════════════
# Aligned crown stack
# ════════════════════════════════════════════════════════════════════════
def aligned_crown(src, r, frames=None):
    """The target's crown window resampled onto the SOURCE crown window: the
    source square's own pixel grid (size, orientation, resolution), every file
    band of the target, through its post-.ax frame. float32, NaN outside the
    target. The found pose maps source-crop pixels to target frame pixels."""
    frames = frames or _DEFAULT_FRAMES
    r0, c0, r1, c1 = src.bbox
    ch, cw = r1 - r0, c1 - c0
    sx, sy = r.extra.get("work_scale", (1.0, 1.0))
    M = np.empty((2, 3))
    M[:, :2] = r.q2raw[:, :2] @ np.diag([1.0 / sx, 1.0 / sy])
    M[:, 2] = r.q2raw[:, 2]
    corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64) @ M[:, :2].T + M[:, 2]
    x0, y0 = np.floor(corners.min(axis=0)) - 2
    x1, y1 = np.ceil(corners.max(axis=0)) + 2
    f = math.sqrt(abs(np.linalg.det(M[:, :2])))                # target px per source px
    n = file_band_count(r.target_fp)
    arr, valid, (fx, fy), (ox, oy) = frames.read(r.target_fp, r.extra.get("ax") or {}, x0, y0, x1, y1,
                                                 list(range(n)), factor=f if f > 1.5 else 1.0,
                                                 zero_is_nodata=False)
    arr = arr.astype(np.float32)
    k = f / fx
    if k > 1.25 and arr.shape[0] > 2 and arr.shape[1] > 2:          # area-reduce a much finer target first
        nw_, nh_ = max(2, int(round(arr.shape[1] / k))), max(2, int(round(arr.shape[0] / k)))
        kx, ky = arr.shape[1] / float(nw_), arr.shape[0] / float(nh_)
        arr = np.stack([cv2.resize(arr[:, :, b], (nw_, nh_), interpolation=cv2.INTER_AREA)
                        for b in range(arr.shape[2])], axis=2)
        valid = cv2.resize(valid.astype(np.float32), (nw_, nh_), interpolation=cv2.INTER_AREA) > 0.999
        fx, fy = fx * kx, fy * ky
    Mw = np.empty((2, 3))
    Fi = np.diag([1.0 / fx, 1.0 / fy])
    Mw[:, :2] = Fi @ M[:, :2]
    Mw[:, 2] = Fi @ (M[:, 2] - (ox, oy))
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    out = np.stack([cv2.warpAffine(arr[:, :, b], Mw, (cw, ch), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0) for b in range(arr.shape[2])], axis=2)
    ok = cv2.warpAffine(valid.astype(np.float32), Mw, (cw, ch), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0) > 0.999
    out[~ok] = np.nan
    return out


def _date_of(path):
    import re
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


EXPORT_FORMATS = {"tif": "GeoTIFF (all bands)", "jpg": "JPEG (compressed)"}
_META_NAME = "{group}_metadata.csv"


def _safe(name):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name)).strip("_") or "crown"


def _jpeg_u8(arr, bands, keep_u8):
    """(h, w, 1 or 3) uint8 of the shown bands for a JPEG: 8-bit data as it is,
    anything else stretched (0.5-99.5 %) over its valid pixels; outside = 0."""
    a = np.asarray(arr)
    b = [k for k in bands if k < a.shape[2]] or [0]
    a = a[:, :, b[:3]]
    valid = np.isfinite(a).all(axis=2) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape[:2], bool)
    if keep_u8:
        out = np.where(valid[:, :, None], np.nan_to_num(a), 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = to_uint8(np.nan_to_num(a.astype(np.float32)), valid)
        out[~valid] = 0
    if out.shape[2] == 2:
        out = out[:, :, :1]
    return out


def _write_crop(src, arr, path, fmt, keep_u8):
    """One crop on the SOURCE crown grid: a GeoTIFF with the source crop's
    georeference (all bands), or a JPEG of the shown bands."""
    r0, c0, _r1, _c1 = src.bbox
    if fmt == "jpg":
        q = _jpeg_u8(arr, src.bands, keep_u8)
        img = q[:, :, 0] if q.shape[2] == 1 else cv2.cvtColor(q, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError(f"cannot encode {os.path.basename(path)}")
        buf.tofile(path)
        return path
    return write_geotiff_crop(src.fp, arr, c0, r0, path, geo=src.geo)


def _previous_export(meta_path):
    """Files listed by an earlier export's metadata CSV (they are ours)."""
    import csv
    try:
        with open(meta_path, encoding="utf-8", newline="") as f:
            return [v for row in csv.DictReader(f) for v in (row.get("file"), row.get("polygon_json")) if v]
    except Exception:
        return []


def _write_polygon_json(path, crown, shape_type, role, image, points, ref, crop_pts, crop_file, extra=None):
    """The crown's polygon on one exported image: in that IMAGE's frame as a
    CanoPie polygon record (points / image_ref_size / coord_space, what the
    project's polygons/*.json hold), and in the aligned CROP's pixels."""
    import json
    rec = {"name": crown, "type": shape_type or "polygon", "coord_space": "image",
           "points": [[round(float(x), 4), round(float(y), 4)] for x, y in points],
           "image_ref_size": {"w": int(ref["w"]), "h": int(ref["h"])},
           "image": image, "role": role, "crop_file": crop_file,
           "crop_points": [[round(float(x), 3), round(float(y), 3)] for x, y in crop_pts]}
    rec.update(extra or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return path


def export_crowns(src, results, out_dir, fmt="tif", frames=None, saved=None):
    """The crown of `src` in every image it was found in, as crops ALIGNED on
    the source crown crop (same grid, size and orientation: they stack), into
    <out_dir>/<crown>_<source>/:

      <crown>_00_SOURCE_<image>.<tif|jpg>   the source crop
      <crown>_01_<image>.<tif|jpg> ...      each target, by date then name
      <crown>_00_SOURCE_<image>.json ...    the crown's polygon on each image
      <crown>_metadata.csv                  one row per image

    fmt "tif": GeoTIFF, every band, with the source crop's georeference (the
    targets resampled onto it are float32, NaN outside the image); "jpg": the
    shown bands, compressed (8-bit data as it is, else stretched per image).
    The metadata holds the crop's geotransform, so JPEGs are located too.
    `results`: the Results to write (the caller leaves rejected ones out);
    `saved`: optional {target path: saved state} for the metadata. Files of an
    earlier export of this crown into the same folder are replaced. Returns
    (folder, written paths)."""
    import csv
    fmt = "jpg" if str(fmt).lower() in ("jpg", "jpeg") else "tif"
    frames = frames or _DEFAULT_FRAMES
    group = _safe(src.group or "crown")
    sstem = os.path.splitext(os.path.basename(src.fp))[0]
    d = os.path.join(out_dir, f"{group}_{_safe(sstem)}")
    os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(d, _META_NAME.format(group=group))
    for name in _previous_export(meta_path):
        p = os.path.join(d, os.path.basename(name))
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError as e:
                logging.warning("[canopy_finder] could not replace %s: %s", p, e)
    r0, c0, r1, c1 = src.bbox
    n = file_band_count(src.fp)
    s_arr, s_valid, _f, _o = frames.read(src.fp, src.ax or {}, c0, r0, c1, r1, list(range(n)), zero_is_nodata=False)
    keep_u8 = np.asarray(s_arr).dtype == np.uint8
    geo = src.geo
    t = geo.transform if geo is not None and geo.located else None
    gt = ((t[0] + c0 * t[1] + r0 * t[2], t[1], t[2], t[3] + c0 * t[4] + r0 * t[5], t[4], t[5])
          if t is not None else ("",) * 6)
    common = {"crown": src.group, "source_image": src.fp, "width": c1 - c0, "height": r1 - r0,
              "epsg": (geo.epsg or "") if geo is not None and geo.located else "",
              "gt0_x": gt[0], "gt1_dx": gt[1], "gt2_rx": gt[2], "gt3_y": gt[3], "gt4_ry": gt[4], "gt5_dy": gt[5],
              "pixel_size_m": round(float(min(src.gsd)), 6) if src.gsd else ""}
    ext = ".jpg" if fmt == "jpg" else ".tif"
    rows, written = [], []
    name = f"{group}_00_SOURCE_{_safe(sstem)}{ext}"
    written.append(_write_crop(src, s_arr, os.path.join(d, name), fmt, keep_u8))
    crop_pts = np.asarray(src.raw_pts, np.float64) - (c0, r0)           # the same on every (aligned) crop
    s_pts, s_ref, _in = geo.to_record(src.raw_pts)
    jname = os.path.splitext(name)[0] + ".json"
    written.append(_write_polygon_json(os.path.join(d, jname), src.group, src.shape_type, "source", src.fp,
                                       s_pts, s_ref, crop_pts, name))
    cm = src.cmask
    rows.append(dict(common, file=name, polygon_json=jname, role="source", image=os.path.basename(src.fp),
                     image_path=src.fp,
                     date=_date_of(src.fp), status="source", score="", offset_m="", rotation_deg="", scale="",
                     method="", saved="", bands=int(np.asarray(s_arr).shape[2]),
                     crown_valid_frac=round(float((s_valid & cm).sum()) / max(1, int(cm.sum())), 4)))
    found = [r for r in results if r.has_polygon]
    found.sort(key=lambda r: (_date_of(r.target_fp) or "9999", os.path.basename(r.target_fp).lower()))
    for k, r in enumerate(found, start=1):
        try:
            a = aligned_crown(src, r, frames)
        except Exception as e:
            logging.warning("[canopy_finder] aligned crown of %s: %s", r.target_fp, e)
            continue
        tstem = os.path.splitext(os.path.basename(r.target_fp))[0]
        name = f"{group}_{k:02d}_{_safe(tstem)}{ext}"
        t_u8 = keep_u8 and file_dtype_is_u8(r.target_fp)
        written.append(_write_crop(src, a, os.path.join(d, name), fmt, t_u8))
        jname = os.path.splitext(name)[0] + ".json"
        if r.record_points is not None and r.image_ref_size:
            written.append(_write_polygon_json(
                os.path.join(d, jname), src.group, src.shape_type, "target", r.target_fp, r.record_points,
                r.image_ref_size, crop_pts, name,
                {"status": r.status, "score": None if r.score is None else round(float(r.score), 4),
                 "offset_m": round(r.offset_m, 3), "rotation_deg": round(r.angle, 3), "method": r.method}))
        else:
            jname = ""
        ok = np.isfinite(a).all(axis=2)
        rows.append(dict(common, file=name, polygon_json=jname, role="target", image=os.path.basename(r.target_fp),
                         image_path=r.target_fp, date=_date_of(r.target_fp), status=r.status,
                         score="" if r.score is None else round(float(r.score), 4),
                         offset_m=round(r.offset_m, 3), rotation_deg=round(r.angle, 3),
                         scale=r.extra.get("scale", ""), method=r.method,
                         saved=(saved or {}).get(r.target_fp, ""), bands=int(a.shape[2]),
                         crown_valid_frac=round(float((ok & cm).sum()) / max(1, int(cm.sum())), 4)))
    cols = ["file", "polygon_json", "role", "crown", "image", "date", "status", "score", "offset_m", "rotation_deg", "scale",
            "method", "saved", "bands", "width", "height", "crown_valid_frac", "pixel_size_m", "epsg",
            "gt0_x", "gt1_dx", "gt2_rx", "gt3_y", "gt4_ry", "gt5_dy", "image_path", "source_image"]
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            wr.writerow(row)
    written.append(meta_path)
    return d, written


def file_dtype_is_u8(fp):
    """The file's samples are 8-bit (a JPEG / 8-bit TIFF): kept as they are in a JPEG export."""
    if not str(fp).lower().endswith((".tif", ".tiff")):
        return True
    try:
        rd = _reader(fp)
        if rd is not None:
            return np.dtype(rd.profile.dtype) == np.uint8
        import tifffile
        with tifffile.TiffFile(fp) as tif:
            return tif.pages[0].dtype == np.uint8
    except Exception:
        return False
