"""
Viewer alignment: a display-only "fake registration" between images of one
root that cover the same ground but do not share a pixel grid (two
orthomosaics of one forest with different row / column counts, resolution, a
small rotation or a residual georeference error).

A few spots of the reference image are matched in each other image with the
canopy finder, and a SIMILARITY (shift + rotation + one scale) is fitted per
image, from reference RAW file pixels to that image's RAW file pixels. Raw
pixels, not records: a polygon record is a fraction of the post-.ax frame in
raw-basis units, which is anisotropic once a crop is applied, and a later .ax
edit would invalidate anything stored in it. Raw pixels never change.

The fit is then used by the viewers (zoom, pan and the view's rotation follow
it; nothing is resampled or reprojected) and by polygon sync / copy (a crown
drawn on one image lands on the same ground in the others).

No Qt here: the dialog and the viewer glue live in view_alignment_dialog.py,
image_viewer.py and project_tab.py.
"""
import math
import time

import numpy as np

from . import ax_geometry as G

__all__ = ["pick_spots", "spot_side_px", "fit_similarity", "similarity_params",
           "invert", "compose", "apply", "record_to_raw_pts", "raw_to_record_pts",
           "AlignmentSets", "map_record_points", "map_record", "spot_residuals"]

#: spot side (raw px of the reference's export frame), and in metres of ground
SPOT_FRAC = 0.06
SPOT_MIN_M, SPOT_MAX_M = 25.0, 60.0
SPOT_MAX_PX = 800
SPOT_MIN_PX = 48


# ════════════════════════════════════════════════════════════════════════
# Affine helpers (2x3)
# ════════════════════════════════════════════════════════════════════════
def _h(M):
    M = np.asarray(M, np.float64).reshape(2, 3)
    return np.vstack((M, (0.0, 0.0, 1.0)))


def apply(M, pts):
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    M = np.asarray(M, np.float64).reshape(2, 3)
    return a @ M[:, :2].T + M[:, 2]


def invert(M):
    return np.linalg.inv(_h(M))[:2]


def compose(A, B):
    """A after B (B applied first)."""
    return (_h(A) @ _h(B))[:2]


def similarity_params(M):
    """(scale, angle in degrees, tx, ty) of a 2x3 similarity."""
    M = np.asarray(M, np.float64).reshape(2, 3)
    s = math.sqrt(abs(np.linalg.det(M[:, :2])))
    ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return s, ang, float(M[0, 2]), float(M[1, 2])


# ════════════════════════════════════════════════════════════════════════
# The fit
# ════════════════════════════════════════════════════════════════════════
def _umeyama(src, dst, w):
    W = float(w.sum())
    mu_s = (src * w[:, None]).sum(0) / W
    mu_d = (dst * w[:, None]).sum(0) / W
    S, D = src - mu_s, dst - mu_d
    cov = (D * w[:, None]).T @ S / W
    U, sig, Vt = np.linalg.svd(cov)
    d = 1.0 if np.linalg.det(U) * np.linalg.det(Vt) >= 0 else -1.0
    Dm = np.diag((1.0, d))
    R = U @ Dm @ Vt
    var_s = float((w * (S ** 2).sum(1)).sum() / W)
    c = float(np.trace(np.diag(sig) @ Dm) / max(var_s, 1e-12))
    t = mu_d - c * (R @ mu_s)
    M = np.empty((2, 3))
    M[:, :2] = c * R
    M[:, 2] = t
    return M, d < 0


def spot_residuals(M, spots):
    """Mean corner residual (px of the destination) of each spot."""
    out = []
    for s in spots:
        p = apply(M, s["src"])
        out.append(float(np.hypot(*(p - np.asarray(s["dst"], np.float64)).T).mean()))
    return out


def fit_similarity(spots, min_spots=2, side_px=None):
    """Least-squares similarity (no reflection) from matched spots.

    spots: [{"src": (k, 2) points, "dst": (k, 2) points, "weight": float}]
    -- every spot contributes all its corners, so even two spots fix the
    rotation well. Outliers are dropped one spot at a time while the worst
    residual exceeds max(3 x the median, 1.5 % of the spot side) and at least
    three spots remain.

    Returns {"M", "used" (bool per spot), "rms_px", "residuals", "reflected",
    "n_used"} or None when fewer than `min_spots` spots are usable."""
    use = [i for i, s in enumerate(spots) if s.get("dst") is not None and len(s.get("src", [])) >= 1]
    if len(use) < int(min_spots):
        return None
    if side_px is None:
        try:
            side_px = float(np.median([np.ptp(np.asarray(spots[i]["src"]), axis=0).max() for i in use]))
        except Exception:
            side_px = 100.0

    def _fit(idx):
        src = np.vstack([np.asarray(spots[i]["src"], np.float64).reshape(-1, 2) for i in idx])
        dst = np.vstack([np.asarray(spots[i]["dst"], np.float64).reshape(-1, 2) for i in idx])
        w = np.concatenate([np.full(len(np.asarray(spots[i]["src"]).reshape(-1, 2)),
                                    max(1e-3, float(spots[i].get("weight", 1.0)))) for i in idx])
        return _umeyama(src, dst, w)

    idx = list(use)
    M, refl = _fit(idx)
    while len(idx) > 3:
        # leave-one-out: a wrong spot drags a plain least-squares fit toward
        # itself and hides; judged against the fit of the OTHERS it stands out
        loo = []
        for j in range(len(idx)):
            rest = idx[:j] + idx[j + 1:]
            Mj, _r = _fit(rest)
            loo.append(spot_residuals(Mj, [spots[idx[j]]])[0])
        k = int(np.argmax(loo))
        med = float(np.median(loo))
        if loo[k] <= max(3.0 * med, 0.015 * side_px):
            break
        idx.pop(k)
        M, refl = _fit(idx)
    res_all = spot_residuals(M, [spots[i] if i in use else {"src": [[0, 0]], "dst": [[0, 0]]}
                                  for i in range(len(spots))])
    used = [i in idx for i in range(len(spots))]
    r_used = [res_all[i] for i in idx]
    rms = float(math.sqrt(np.mean(np.square(r_used)))) if r_used else float("nan")
    return {"M": M, "used": used, "rms_px": rms, "residuals": res_all, "reflected": bool(refl),
            "n_used": len(idx)}


# ════════════════════════════════════════════════════════════════════════
# Spots
# ════════════════════════════════════════════════════════════════════════
def spot_side_px(frame_w, frame_h, gsd=None):
    """Side of a spot square in export-frame px: 6 % of the frame, kept
    between 25 and 60 m of ground (located images) and under 800 px."""
    side = SPOT_FRAC * min(frame_w, frame_h)
    if gsd and gsd > 0:
        side = min(max(side, SPOT_MIN_M / gsd), SPOT_MAX_M / gsd)
    return float(max(SPOT_MIN_PX, min(SPOT_MAX_PX, side, 0.45 * min(frame_w, frame_h))))


def _gray(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :min(3, a.shape[2])].astype(np.float32).mean(axis=2)
    else:
        a = a.astype(np.float32)
    return a


def pick_spots(img, valid=None, n=5, side=None, region=None):
    """`n` well-spread, textured spot centres in the (preview) array `img`.

    Score = local standard deviation x gradient energy over a spot-sized
    window; a window touching an invalid pixel (NaN, NoData, outside
    `region`) never counts. The best spot of each of five cells (the four
    corners, inset, and the centre) is taken, then the farthest remaining good
    spots fill in.

    img     (H, W[, C]) array (the viewer's in-memory preview)
    valid   (H, W) bool, True = real data (default: finite and not all-zero)
    side    window side in `img` px (default 6 % of the smaller side)
    region  (H, W) bool where spots may go (e.g. the overlap with the targets)
    Returns [(x, y), ...] centres in `img` px."""
    import cv2
    g = _gray(img)
    H, W = g.shape[:2]
    if valid is None:
        a = np.asarray(img)
        valid = np.isfinite(g)
        if a.ndim == 3:
            valid &= np.any(a[..., :min(3, a.shape[2])] != 0, axis=2)
        else:
            valid &= a != 0
    valid = np.asarray(valid, bool)
    if region is not None:
        valid = valid & np.asarray(region, bool)
    # work at <= 600 px
    k = max(1.0, max(H, W) / 600.0)
    if k > 1.0:
        sw, sh = max(8, int(round(W / k))), max(8, int(round(H / k)))
        gs = cv2.resize(np.nan_to_num(g), (sw, sh), interpolation=cv2.INTER_AREA)
        vs = cv2.resize(valid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.999
    else:
        gs, vs = np.nan_to_num(g), valid
    sh, sw = gs.shape
    side_s = (side / k) if side else SPOT_FRAC * min(sw, sh)
    win = max(5, int(round(side_s)) | 1)
    gs = np.where(vs, gs, 0.0).astype(np.float32)
    m1 = cv2.blur(gs, (win, win))
    m2 = cv2.blur(gs * gs, (win, win))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    ge = cv2.blur(np.sqrt(gx * gx + gy * gy), (win, win))
    score = std * np.sqrt(ge)
    ok = cv2.erode(vs.astype(np.uint8), np.ones((win, win), np.uint8)) > 0
    half = win // 2
    ok[:half, :] = ok[-half:, :] = False
    ok[:, :half] = ok[:, -half:] = False
    score = np.where(ok, score, -1.0)
    if not (score > 0).any():
        return []
    picks = []
    cells = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5)]
    for cx, cy in cells[:max(1, n)]:
        x0, x1 = int((cx - 0.17) * sw), int((cx + 0.17) * sw)
        y0, y1 = int((cy - 0.17) * sh), int((cy + 0.17) * sh)
        x0, y0 = max(0, x0), max(0, y0)
        sub = score[y0:y1, x0:x1]
        if sub.size and sub.max() > 0:
            yy, xx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            p = (x0 + xx, y0 + yy)
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > win for q in picks):
                picks.append(p)
    # fill in: good spots far from the ones already picked
    if len(picks) < n:
        thr = np.percentile(score[score > 0], 70) if (score > 0).sum() > 10 else 0.0
        ys, xs = np.nonzero(score >= thr)
        cand = np.column_stack((xs, ys)).astype(np.float64)
        if len(cand) > 4000:
            cand = cand[np.random.default_rng(0).choice(len(cand), 4000, replace=False)]
        while len(picks) < n and len(cand):
            if picks:
                P = np.asarray(picks, np.float64)
                d = np.sqrt(((cand[:, None, :] - P[None]) ** 2).sum(-1)).min(1)
            else:
                d = np.ones(len(cand))
            j = int(np.argmax(d))
            if d[j] <= win:
                break
            picks.append((int(cand[j, 0]), int(cand[j, 1])))
    def _up(v):
        return float((v + 0.5) * k - 0.5) if k > 1.0 else float(v)
    return [(_up(x), _up(y)) for x, y in picks[:n]]


# ════════════════════════════════════════════════════════════════════════
# Records <-> raw pixels (one image)
# ════════════════════════════════════════════════════════════════════════
def record_to_raw_pts(pts, ax, raw_w, raw_h, ref_wh=None):
    """Record points (fractions of the post-.ax frame, in `ref_wh` units,
    default the raw basis) -> raw file pixels."""
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    if not len(a):
        return a
    rw, rh = ref_wh if ref_wh else (raw_w, raw_h)
    ops, (W, H) = G.frame_plan(ax or {}, int(raw_w), int(raw_h))
    e = a * (W / float(rw), H / float(rh))
    return G.inverse_np(ops, e) if ops else e


def raw_to_record_pts(raw, ax, raw_w, raw_h):
    """Raw file pixels -> record points in the raw basis (raw_w, raw_h)."""
    a = np.asarray(raw, np.float64).reshape(-1, 2)
    if not len(a):
        return a
    ops, (W, H) = G.frame_plan(ax or {}, int(raw_w), int(raw_h))
    e = G.forward_np(ops, a) if ops else a
    return e * (raw_w / float(max(1, W)), raw_h / float(max(1, H)))


def export_to_raw(pts, ax, raw_w, raw_h):
    ops, _wh = G.frame_plan(ax or {}, int(raw_w), int(raw_h))
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    return G.inverse_np(ops, a) if ops else a


def raw_to_export(pts, ax, raw_w, raw_h):
    ops, _wh = G.frame_plan(ax or {}, int(raw_w), int(raw_h))
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    return G.forward_np(ops, a) if ops else a


def map_record_points(pts, pm, ref_wh=None):
    """Record points of the source image -> record points of the target,
    through a point map {"M" (raw src -> raw dst), "src_ax", "dst_ax",
    "src_raw_wh", "dst_raw_wh"} (plain data: it travels to worker threads)."""
    sw, sh = pm["src_raw_wh"]
    dw, dh = pm["dst_raw_wh"]
    raw = record_to_raw_pts(pts, pm.get("src_ax"), sw, sh, ref_wh=ref_wh)
    return raw_to_record_pts(apply(pm["M"], raw), pm.get("dst_ax"), dw, dh)


def map_record(full, pm, src_wh=None, stamp=None):
    """A whole record (points + annotator patches) re-expressed on the target
    through `pm`; the polygon_bulk copy_record counterpart."""
    import copy as _copy
    out = {}
    for k, v in full.items():
        if k in ("points", "annotator_patches", "lod", "display_points"):
            continue
        out[k] = _copy.deepcopy(v)
    ref = full.get("image_ref_size") or {}
    rwh = None
    try:
        if ref.get("w") and ref.get("h"):
            rwh = (float(ref["w"]), float(ref["h"]))
    except AttributeError:
        rwh = None
    rwh = rwh or (tuple(src_wh) if src_wh else None)
    out["coord_space"] = "image"
    pts = full.get("points") or []
    out["points"] = map_record_points(pts, pm, rwh).tolist() if len(pts) else []
    dw, dh = pm["dst_raw_wh"]
    out["image_ref_size"] = {"w": int(dw), "h": int(dh)}
    s2 = float(abs(np.linalg.det(np.asarray(pm["M"], np.float64).reshape(2, 3)[:, :2])))
    patches = full.get("annotator_patches")
    if patches:
        newp = {}
        for prop, entry in patches.items():
            if not isinstance(entry, dict):
                continue
            e = _copy.deepcopy(entry)
            b = entry.get("basis") or {}
            bwh = (float(b["w"]), float(b["h"])) if isinstance(b, dict) and b.get("w") else rwh
            e["polygons"] = [[np.round(map_record_points(r, pm, bwh), 3).tolist() for r in rings]
                             for rings in (entry.get("polygons") or [])]
            for k in ("area_px", "crown_area_px"):
                if isinstance(e.get(k), (int, float)):
                    e[k] = float(e[k]) * s2
            e["basis"] = {"w": int(dw), "h": int(dh)}
            newp[prop] = e
        out["annotator_patches"] = newp
    out["last_modified"] = stamp or time.strftime("%Y-%m-%d %H:%M:%S")
    return out


# ════════════════════════════════════════════════════════════════════════
# Stored alignments
# ════════════════════════════════════════════════════════════════════════
def _norm(fp):
    import os
    return os.path.normcase(os.path.abspath(str(fp)))


class AlignmentSets:
    """{set id: {"reference": fp, "enabled": bool, "root": name,
                  "targets": {fp: {"M", "rms_px", "rms_m", "n_used", "spots", "method", "created"}}}}

    Kept in global.ax under "view_alignment" (plain JSON). Any two images of
    one set map through the reference: M_ab = M_rb o M_ra^-1."""

    def __init__(self, data=None):
        self.sets = {}
        for sid, s in ((data or {}).get("sets") or {}).items():
            if isinstance(s, dict) and s.get("reference"):
                self.sets[str(sid)] = s
        self._index = None

    def to_json(self):
        return {"sets": self.sets}

    def _idx(self):
        if self._index is None:
            idx = {}
            for sid, s in self.sets.items():
                idx.setdefault(_norm(s["reference"]), []).append(sid)
                for fp in (s.get("targets") or {}):
                    idx.setdefault(_norm(fp), []).append(sid)
            self._index = idx
        return self._index

    def changed(self):
        self._index = None

    def put(self, reference, targets, root=None, enabled=True):
        """Store one alignment (replacing any set with the same reference)."""
        for sid in [k for k, s in self.sets.items() if _norm(s["reference"]) == _norm(reference)]:
            self.sets.pop(sid, None)
        sid = "%s_%d" % (str(root or "set"), int(time.time() * 1000))
        self.sets[sid] = {"reference": reference, "root": root, "enabled": bool(enabled),
                          "targets": targets}
        self.changed()
        return sid

    def remove_for(self, fps):
        keys = {_norm(f) for f in fps}
        for sid in [k for k, s in self.sets.items()
                    if _norm(s["reference"]) in keys or keys & {_norm(f) for f in s.get("targets") or {}}]:
            self.sets.pop(sid, None)
        self.changed()

    def set_for(self, fp_a, fp_b=None, enabled_only=True):
        """The set holding fp_a (and fp_b when given)."""
        idx = self._idx()
        a = set(idx.get(_norm(fp_a), []))
        if fp_b is not None:
            a &= set(idx.get(_norm(fp_b), []))
        for sid in sorted(a, reverse=True):          # the newest first
            s = self.sets.get(sid)
            if s and (s.get("enabled", True) or not enabled_only):
                return sid, s
        return None, None

    @staticmethod
    def _to_ref(s, fp):
        """2x3 raw ref -> raw fp, or None."""
        if _norm(fp) == _norm(s["reference"]):
            return np.array([[1.0, 0, 0], [0, 1.0, 0]])
        for k, t in (s.get("targets") or {}).items():
            if _norm(k) == _norm(fp) and t.get("M") is not None:
                return np.asarray(t["M"], np.float64).reshape(2, 3)
        return None

    def chain(self, fp_a, fp_b, enabled_only=True):
        """2x3 raw a -> raw b, or None when no (enabled) set holds both."""
        if _norm(fp_a) == _norm(fp_b):
            return None
        _sid, s = self.set_for(fp_a, fp_b, enabled_only=enabled_only)
        if s is None:
            return None
        Ma, Mb = self._to_ref(s, fp_a), self._to_ref(s, fp_b)
        if Ma is None or Mb is None:
            return None
        return compose(Mb, invert(Ma))

    def reference_of(self, fp, enabled_only=True):
        _sid, s = self.set_for(fp, enabled_only=enabled_only)
        return s["reference"] if s else None

    def members(self, fp, enabled_only=True):
        _sid, s = self.set_for(fp, enabled_only=enabled_only)
        if not s:
            return []
        return [s["reference"]] + list((s.get("targets") or {}).keys())
