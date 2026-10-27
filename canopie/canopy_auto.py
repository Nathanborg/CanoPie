"""Auto Canopy: delineate every tree crown in an image region, the way a user would with the
Smart Canopy brush -- zoom so crowns are a workable size, find where crowns are, put a handful of
clicks in each, and let ``canopy_tracer.select_canopy`` outline it, every crown cut against the
ones already found so that none overlap.

The "simulated user" is explicit in :class:`AutoParams`:

* zoom      ``r_work``: crowns are worked on at this radius (px), whatever the input resolution;
            the input's typical crown radius is measured from the image (scale-normalised LoG),
            or given (``crown_radius_px``).
* finding   ``channel`` / ``seed_sigma`` / ``seed_distance`` / ``seed_strength`` / ``shadow_pct``:
            crown centres are maxima of a blob filter at about the crown's scale, apart by at
            least a fraction of a crown, and not in shadow (the gaps of a closed canopy).
* clicking  ``clicks`` / ``spread`` / ``brush`` / ``grow``: a sunflower pattern of clicks around
            each centre (skipping shadow and crowns already found), brush and grow as the user's
            (the hand-tuned optimum: ~10 clicks, brush 1/6 of the radius, grow 4).
* accepting ``area_min`` / ``area_max`` / ``solidity_min``: what is not crown-shaped is dropped.
* passes    ``residual_passes``: the canopy left unclaimed is seeded again (where the first pass
            found no centre); ``refine``: every crown is cut again with all its neighbours known.

Defaults are optimised on hand-drawn BCI crowns (``scripts/optimise_auto_canopy.py``).  No Qt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .canopy_tracer import CanopyParams, CanopyResult, _largest_contour, select_canopy

_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


@dataclass
class AutoParams:
    # Defaults: scripts/optimise_auto_canopy.py on hand-drawn BCI crowns (8 dense blocks of ~55 crowns;
    # 150-config search on 5, then the 3 held out).  With the crown size known (as the app takes it from
    # the crowns outlined in the image): crown F1 0.60 held out (0.63 / 0.58 / 0.58), 0.45 on the tuning
    # blocks, mean IoU of matched crowns 0.71, ~20 s per 1500 x 1500 px block.  With the size measured from
    # the pixels instead, 0.26 -- hence the size from the user's crowns.  A re-tune with the size known
    # scored higher on the tuning blocks (0.49) but lower held out (0.52): overfitted, not used.
    r_work: float = 41.426         # crown radius at working resolution (px): the "zoom"
    crown_radius_px: float = 0.0  # typical crown radius in the input (0 = measure it from the image)
    radius_scale: float = 1.0     # x that radius
    channel: str = "green"        # "grey" | "exg" | "green": what crowns stand out on
    seed_sigma: float = 1.119     # blob filter scale, x the crown radius / sqrt(2)
    scale_min: float = 0.954      # each centre's own radius: the strongest blob scale in
    scale_max: float = 2.173      #   [scale_min, scale_max] x the typical radius (crowns differ ~3x)
    big_bias: float = 1.174       # order: response x (radius / typical)^big_bias -- big crowns first,
                                  #   so their leaf clusters are not taken for crowns of their own
    seed_distance: float = 1.305   # crown centres at least this x the larger one's radius apart
    seed_strength: float = 0.0457  # blob response >= this x its 99th percentile
    shadow_pct: float = 25.055     # no centres or clicks on the darkest this % (gaps, shade)
    clicks: int = 8
    spread: float = 0.837         # clicks within this x the radius of the centre
    brush: float = 0.218          # brush radius, x the crown radius
    grow: float = 5.799            # x the brush (CanopyParams.grow)
    area_min: float = 0.332       # accepted crown area, x pi R^2 ...
    area_max: float = 2.626
    solidity_min: float = 0.764   # area / convex hull area
    residual_passes: int = 2
    residual_min: float = 0.85    # a residual centre needs unclaimed canopy this x R around it
    refine: bool = True
    skip_claimed: float = 0.5     # a centre whose disc (half its radius) is this much claimed is not cut
    cut_px: int = 120             # (one-shot mode) each crown's graph cut at this working size
    cut_iterations: int = 2
    # How the user works, crown by crown: zoomed in on it, clicking in rounds until it sits right.
    # On BCI (true centres + sizes, 5 tuning blocks) the best of 23 settings scored crown F1 0.80 vs 0.79
    # for the one-shot cut -- within noise, at 4x the time -- so it is off by default; Learn Crowns can
    # still choose it (SEARCH_SPACE), and params_from_sessions starts from it.
    progressive: bool = False     # False: the one-shot sunflower of `clicks` within `spread` (above)
    r_cut: float = 64.0           # each crown is cut from the full-resolution pixels, zoomed so its
                                  #   radius is this many px (by hand the cut sees ~48)
    clicks_first: int = 8         # round 1: this many clicks within `first_spread` x R of the centre
    first_spread: float = 0.7
    clicks_round: int = 6         # every next round: this many inside the outline so far, as far from
    rounds: int = 3               #   the earlier clicks as possible; until the outline stops changing
    edge_keep: float = 1.0        # a new click keeps this x the brush from the outline's edge
    # Each centre's own radius from SIZE_MODEL_BCI (or the project's adapted model).  Given the true centres
    # it lifts held-out BCI crown F1 0.66 -> 0.71; finding the centres too, the defaults score 0.55 with it vs
    # 0.57 without, and the best re-tune with it 0.60 at 6x the time (within noise) -- so off by default,
    # but in Learn Crowns' search (with the size model adapted to the user's crowns).
    per_crown_size: bool = False


@dataclass
class AutoResult:
    crowns: List[CanopyResult]    # polygons in input px (pixel centres at integers)
    crown_radius_px: float        # the typical radius used (input px)
    seeds: np.ndarray             # Nx2 input px, every centre tried
    note: str = ""


def _channel(rgb: np.ndarray, name: str) -> np.ndarray:
    f = rgb.astype(np.float32)
    if f.ndim == 2:
        return f
    if name == "exg":
        return 2.0 * f[..., 1] - f[..., 0] - f[..., 2]
    if name == "green":
        return np.ascontiguousarray(f[..., 1])
    return f[..., :3] @ np.float32([0.299, 0.587, 0.114])


def _norm01(v: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    s = v[valid] if valid is not None and valid.any() else v
    sub = s[::4] if s.size > 400_000 else s
    lo, hi = np.percentile(sub, [1, 99])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _log(f: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalised LoG, bright blobs positive."""
    g = cv2.GaussianBlur(f, (0, 0), sigma)
    return -(sigma ** 2) * cv2.Laplacian(g, cv2.CV_32F, ksize=1)


def estimate_crown_radius(rgb: np.ndarray, channel: str = "grey") -> float:
    """Typical crown radius (input px): the scale at which bright blobs respond most strongly."""
    H, W = rgb.shape[:2]
    d = max(1.0, max(H, W) / 600.0)
    small = cv2.resize(_channel(rgb, channel), (max(8, int(W / d)), max(8, int(H / d))), interpolation=cv2.INTER_AREA)
    small = _norm01(small)
    sig = 1.5
    best, best_s = -1.0, 1.5
    while sig < min(small.shape) / 6.0:
        r = _log(small, sig)
        v = float(np.mean(np.sort(r.ravel())[-max(10, r.size // 50):]))   # the strongest 2 %
        if v > best:
            best, best_s = v, sig
        sig *= 1.2
    return float(best_s * np.sqrt(2.0) * d)


def _sunflower(n: int, radius: float) -> np.ndarray:
    k = np.arange(n) + 0.5
    rr = radius * np.sqrt(k / n)
    return np.c_[rr * np.cos(k * _GOLDEN), rr * np.sin(k * _GOLDEN)]


def _solidity(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    a = abs(cv2.contourArea(poly.astype(np.float32)))
    h = abs(cv2.contourArea(cv2.convexHull(poly.astype(np.float32))))
    return a / h if h > 0 else 0.0


# ------------------------------------------------------------------------------
# Each crown's own size, from what is around its centre
# ------------------------------------------------------------------------------

SIZE_SCALES = (0.5, 0.7, 1.0, 1.4, 2.0, 2.8)    # blob strength at these x the typical radius
SIZE_RINGS = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)    # circles sampled around the centre, x the typical radius


def _size_maps(work: np.ndarray, R: float, p: "AutoParams") -> dict:
    """Maps the size features read, on the shrunk image the centres are found on (typical radius R px)."""
    chan = _norm01(_channel(work, p.channel))
    grey = cv2.GaussianBlur(_channel(work, "grey"), (0, 0), max(1.0, 0.15 * R))
    lit = grey > np.percentile(grey, p.shadow_pct)
    k = max(3, int(round(0.4 * R)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    logs = np.stack([_log(chan, max(1.0, p.seed_sigma * s * R / np.sqrt(2.0))) for s in SIZE_SCALES])
    logs /= max(float(np.percentile(logs.max(0), 99)), 1e-6)
    logs = np.stack([cv2.dilate(l, se) for l in logs])          # the best within ~0.2 R of the centre
    g = cv2.GaussianBlur(_channel(work, "grey"), (0, 0), max(1.0, 0.1 * R))
    gmag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    gmag /= max(float(np.percentile(gmag, 95)), 1e-6)
    dt = cv2.distanceTransform(lit.astype(np.uint8), cv2.DIST_L2, 5) / max(R, 1e-6)
    return dict(logs=logs, gmag=gmag, dt=cv2.dilate(dt, se), lit=lit.astype(np.float32),
                grey=_norm01(grey), R=R, shape=work.shape[:2])


def size_features(maps: dict, xy: np.ndarray) -> np.ndarray:
    """Per centre (rows x, y in the maps' px): blob strength by scale, room to the nearest shadow, and
    how lit / how edgy / how bright the rings around it are.  Scale-free (all in typical radii)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    R = maps["R"]
    h, w = maps["shape"]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    logs = maps["logs"][:, yi, xi].T                            # n x scales
    cols = [logs, np.log2(np.asarray(SIZE_SCALES))[np.argmax(logs, 1)][:, None], maps["dt"][yi, xi][:, None]]
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for rho in SIZE_RINGS:
        px = xy[:, 0, None] + rho * R * np.cos(ang)[None, :]
        py = xy[:, 1, None] + rho * R * np.sin(ang)[None, :]
        for name in ("lit", "gmag", "grey"):
            v = _bilinear_clip(maps[name], px, py)
            cols.append(v.mean(1)[:, None])
    return np.hstack(cols).astype(np.float32)


# Ridge regression of log(crown radius / typical radius) on size_features, fitted on 707 of the user's
# corrected BCI crowns (5 tuning blocks + 6 training blocks, the latter shifted onto the image where the
# shift was verified; scripts/optimise_auto_canopy.py).  On the 193 crowns of the 3 held-out blocks: median
# error x1.16 (one size for all: x1.39; strongest blob scale: x1.35), 65 % within 25 % of the true size,
# correlation 0.75 (blob scale 0.12).  The room to the nearest shadow carries most of it.  Learn Crowns
# re-fits it on the user's crowns (fit_size_model).
SIZE_MODEL_BCI = dict(
    alpha=162.38, intercept=0.021971,
    mean=(0.47018, 0.46594, 0.38252, 0.25836, 0.16007, 0.16787, -0.21989, 0.80302, 0.90211, 0.36712, 0.68169,
          0.83521, 0.41662, 0.63274, 0.77072, 0.43864, 0.5912, 0.72466, 0.43422, 0.56122, 0.72074, 0.4136, 0.56119,
          0.72901, 0.40837, 0.56761),
    scale=(0.33803, 0.36082, 0.35951, 0.33279, 0.27644, 0.21871, 0.81124, 0.42117, 0.19418, 0.11931, 0.14879,
           0.2099, 0.12378, 0.14577, 0.21062, 0.11006, 0.1374, 0.17261, 0.087226, 0.10679, 0.14187, 0.07766,
           0.086207, 0.1337, 0.071621, 0.080869),
    coef=(-0.037437, -0.00761, -0.0049505, -0.0025548, -0.0019956, -0.0237, 0.038264, 0.083375, 0.0058061,
          -0.020215, 0.013684, 0.030298, -0.046113, 0.022172, 0.021176, -0.053001, 0.01266, 0.015747, -0.0035956,
          -0.0058332, 0.010827, 0.01281, -0.026191, 0.0059614, -0.0029105, -0.020989),
)
SIZE_RANGE = (0.4, 3.0)                          # predicted radius clipped to this x the typical one


def fit_size_model(X: np.ndarray, y: np.ndarray, alpha: float = SIZE_MODEL_BCI["alpha"]) -> dict:
    """Ridge on standardised size_features -> log(radius / typical); numpy only, so Learn Crowns can
    re-fit it on the user's own crowns."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yc = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + float(alpha) * np.eye(Z.shape[1]), Z.T @ (y - yc))
    return dict(alpha=float(alpha), intercept=yc, mean=tuple(map(float, mu)), scale=tuple(map(float, sd)),
                coef=tuple(map(float, w)))


def adapt_size_model(X: np.ndarray, y: np.ndarray, prior: dict = SIZE_MODEL_BCI,
                     alpha: Optional[float] = None) -> dict:
    """The size model re-fitted on the user's crowns, pulled towards ``prior`` (ridge around the prior's
    coefficients, in its standardisation): with a few crowns it stays near the prior, with many it
    follows them.  ``alpha`` defaults to the prior's."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    a = float(prior["alpha"] if alpha is None else alpha)
    Z = (X - np.asarray(prior["mean"])) / np.asarray(prior["scale"])
    w0 = np.asarray(prior["coef"], np.float64)
    zm, ym = Z.mean(0), float(y.mean())
    Zc = Z - zm
    w = np.linalg.solve(Zc.T @ Zc + a * np.eye(Z.shape[1]), Zc.T @ (y - ym) + a * w0)
    return dict(prior, alpha=a, coef=tuple(map(float, w)), intercept=float(ym - zm @ w), crowns=int(len(y)))


def _as_rgb8(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB from RGB / single band / 16-bit input (0.5-99.5 % stretch)."""
    img = np.asarray(rgb)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = np.ascontiguousarray(img[..., :3])
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    return img


def size_samples(rgb: np.ndarray, truth: Sequence[np.ndarray], params: Optional["AutoParams"] = None):
    """(size_features at each annotated crown's centre, log(radius / typical radius), typical radius) --
    what fit_size_model / adapt_size_model learn from.  ``truth``: crown polygons in px (centres at integers);
    the typical radius is their median, as delineate_all is given it in the app."""
    p = params or AutoParams()
    img = _as_rgb8(rgb)
    H, W = img.shape[:2]
    q = [np.asarray(t, np.float64) for t in truth if len(t) >= 3]
    radii = np.array([np.sqrt(_poly_area(t) / np.pi) for t in q])
    R_typ = float(np.median(radii)) * float(p.radius_scale)
    f = min(1.0, float(p.r_work) / max(R_typ, 1e-6))
    ww, wh = max(16, int(round(W * f))), max(16, int(round(H * f)))
    fx, fy = ww / float(W), wh / float(H)
    work = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA) if f < 1.0 else img.copy()
    maps = _size_maps(work, R_typ * fx, p)
    c = np.array([t.mean(0) for t in q])
    X = size_features(maps, np.c_[(c[:, 0] + 0.5) * fx - 0.5, (c[:, 1] + 0.5) * fy - 0.5])
    return X, np.log(radii / R_typ), R_typ


def predict_size(model: dict, X: np.ndarray) -> np.ndarray:
    """Predicted radius / typical radius per row of size_features (clipped to SIZE_RANGE)."""
    Z = (np.asarray(X, np.float64) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return np.clip(np.exp(Z @ np.asarray(model["coef"]) + model["intercept"]), *SIZE_RANGE)


def _bilinear_clip(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(xs, 0, w - 1.001)
    y = np.clip(ys, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def _scale_seeds(chan: np.ndarray, R: float, p: "AutoParams", thr_frac: float, allowed: np.ndarray,
                 size_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """Crown centres with their own radius: maxima over position and scale of the scale-normalised
    LoG (the radius from ``size_fn`` when given: px per centre); a centre inside a stronger-ranked one's
    crown (seed_distance x the larger radius) is dropped.  Rows (x, y, priority, radius), best first."""
    scales = R * np.geomspace(max(0.2, p.scale_min), max(p.scale_min, p.scale_max), 6)
    stack = np.stack([_log(chan, max(1.0, p.seed_sigma * s / np.sqrt(2.0))) for s in scales])
    best = stack.max(0)
    arg = stack.argmax(0)
    thr = thr_frac * float(np.percentile(best, 99))
    cand = _local_maxima(best, max(2.0, p.seed_distance * scales[0]), thr, allowed)
    if not len(cand):
        return np.zeros((0, 4))
    xi, yi = cand[:, 0].astype(int), cand[:, 1].astype(int)
    rad = size_fn(cand[:, :2]) if size_fn is not None else scales[arg[yi, xi]]
    pri = cand[:, 2] * (rad / R) ** float(p.big_bias)
    order = np.argsort(-pri)
    keep: List[int] = []
    for i in order:
        if all(np.hypot(cand[i, 0] - cand[j, 0], cand[i, 1] - cand[j, 1]) >= p.seed_distance * max(rad[i], rad[j])
               for j in keep):
            keep.append(i)
    return np.c_[cand[keep, :2], pri[keep], rad[keep]]


def _local_maxima(resp: np.ndarray, dist: float, thr: float, allowed: np.ndarray) -> np.ndarray:
    k = max(3, int(round(2 * dist)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peak = (resp >= cv2.dilate(resp, se)) & (resp > thr) & allowed
    # a plateau is one candidate (its first pixel), so the loop below runs over candidates, not pixels
    n, lab = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((0, 3))
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    _, first = np.unique(flat[idx], return_index=True)
    ys, xs = np.divmod(idx[first], resp.shape[1])
    order = np.argsort(-resp[ys, xs])
    out: List[Tuple[float, float, float]] = []
    taken = np.zeros(resp.shape, np.uint8)
    for i in order:                                            # plateaus: one centre each
        x, y = int(xs[i]), int(ys[i])
        if taken[y, x]:
            continue
        out.append((float(x), float(y), float(resp[y, x])))
        cv2.circle(taken, (x, y), max(1, int(dist)), 1, -1)
    return np.array(out).reshape(-1, 3)


def delineate_all(
    rgb: np.ndarray,
    params: Optional[AutoParams] = None,
    stop_mask: Optional[np.ndarray] = None,
    region: Optional[np.ndarray] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    seeds: Optional[np.ndarray] = None,
    size_model: Optional[dict] = None,
) -> AutoResult:
    """Every crown in ``rgb`` (HxWx3 uint8 RGB, or HxW).  ``stop_mask``: crowns outlined already
    (kept, never overlapped).  ``region`` (HxW bool): only crowns centred inside it are made.
    ``progress(done, total)`` and ``cancel()`` let a worker thread report and stop.  ``seeds``
    (rows x, y[, radius] in input px) replaces finding the centres (e.g. known crowns).

    Centres are found on a copy shrunk to ``r_work``; each crown is then cut from the full-resolution
    pixels, zoomed to ``r_cut`` (progressive clicks, see AutoParams), and claimed at full resolution."""
    p = params or AutoParams()
    img = _as_rgb8(rgb)
    H, W = img.shape[:2]

    R_in = float(p.crown_radius_px) if p.crown_radius_px > 0 else estimate_crown_radius(img, p.channel)
    R_in *= float(p.radius_scale)
    f = min(1.0, float(p.r_work) / max(R_in, 1e-6))            # finding: never upsample
    ww, wh = max(16, int(round(W * f))), max(16, int(round(H * f)))
    fx, fy = ww / float(W), wh / float(H)
    work = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA) if f < 1.0 else img.copy()
    R = R_in * fx

    def shrink(mask):
        m = mask.astype(np.uint8)
        return (cv2.resize(m, (ww, wh), interpolation=cv2.INTER_NEAREST) if f < 1.0 else m) > 0

    claimed = stop_mask.astype(bool).copy() if stop_mask is not None else np.zeros((H, W), bool)   # full res
    reg = region.astype(bool) if region is not None else None
    reg_w = shrink(reg) if reg is not None else None
    grey_w = cv2.GaussianBlur(_channel(work, "grey"), (0, 0), max(1.0, 0.15 * R))
    grey_thr = float(np.percentile(grey_w, p.shadow_pct))
    lit_w = grey_w > grey_thr                                    # not shadow / gap
    chan = _norm01(_channel(work, p.channel))
    size_fn = None
    if p.per_crown_size:                                        # each centre's own radius (work px)
        model = size_model or SIZE_MODEL_BCI
        maps = _size_maps(work, R, p)
        size_fn = lambda xy: R * predict_size(model, size_features(maps, xy))
    crowns: List[dict] = []                                      # clicks (input px), R, box, m (crop mask)
    tried: List[Tuple[float, float]] = []

    def allowed_seeds():
        a = lit_w & ~shrink(claimed)
        return a & reg_w if reg_w is not None else a

    def area_ok(area, Ri):
        return p.area_min * np.pi * Ri * Ri <= area <= p.area_max * np.pi * Ri * Ri

    def cut(cx, cy, Ri, stop_full, clicks_in=None, own=None):
        """(clicks in input px, box, crop mask) of the crown at (cx, cy) input px, or None.
        ``own`` = (box, mask) of this crown's previous cut: not a neighbour (refine)."""
        half = Ri * (1.25 + p.brush * (1.0 + p.grow))
        x0, x1 = max(0, int(np.floor(cx - half))), min(W, int(np.ceil(cx + half)) + 1)
        y0, y1 = max(0, int(np.floor(cy - half))), min(H, int(np.ceil(cy + half)) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        stop_c = stop_full[y0:y1, x0:x1]
        if own is not None:                                     # this crown itself is not in the way
            (a0, b0, a1, b1), om = own
            stop_c = stop_c.copy()
            ix0, iy0, ix1, iy1 = max(x0, a0), max(y0, b0), min(x1, a1), min(y1, b1)
            if ix1 > ix0 and iy1 > iy0:
                stop_c[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] &= ~om[iy0 - b0:iy1 - b0, ix0 - a0:ix1 - a0]
        if p.progressive or clicks_in is not None:
            g = min(2.0, float(p.r_cut) / max(Ri, 1e-6))          # the crown's own zoom
        else:
            g = min(1.0, float(p.cut_px) / (2.0 * half))
        lw, lh = max(8, int(round((x1 - x0) * g))), max(8, int(round((y1 - y0) * g)))
        sub = img[y0:y1, x0:x1]
        L = cv2.resize(sub, (lw, lh), interpolation=cv2.INTER_AREA if g < 1.0 else cv2.INTER_LINEAR)
        gx, gy = lw / float(x1 - x0), lh / float(y1 - y0)
        stL = cv2.resize(stop_c.astype(np.uint8), (lw, lh), interpolation=cv2.INTER_NEAREST) > 0
        greyL = cv2.GaussianBlur(_channel(L, "grey"), (0, 0), max(1.0, 0.15 * Ri * gx))
        okL = (greyL > grey_thr) & ~stL
        rL = Ri * gx
        brush = max(1.0, p.brush * rL)
        tp = CanopyParams(grow=p.grow, work_px=max(lw, lh) + 2, iterations=int(p.cut_iterations))

        def to_local(q):
            q = np.asarray(q, np.float64).reshape(-1, 2)
            return np.c_[(q[:, 0] - x0 + 0.5) * gx - 0.5, (q[:, 1] - y0 + 0.5) * gy - 0.5]

        def to_input(q):
            return np.c_[(q[:, 0] + 0.5) / gx - 0.5 + x0, (q[:, 1] + 0.5) / gy - 0.5 + y0]

        def inside(q):
            return q[(q[:, 0] >= 0) & (q[:, 0] <= lw - 1) & (q[:, 1] >= 0) & (q[:, 1] <= lh - 1)]

        def usable(q):
            q = inside(q)
            if not len(q):
                return q
            return q[okL[np.round(q[:, 1]).astype(int), np.round(q[:, 0]).astype(int)]]

        def run_cut(q):
            return select_canopy(L, [(c[None, :], brush, False) for c in q], tp, stop_mask=stL)

        c0 = to_local([[cx, cy]])[0]
        if clicks_in is not None:                               # the same clicks again (refine)
            pts = usable(to_local(clicks_in))
            if not len(pts):
                return None
            res = run_cut(pts)
        elif not p.progressive:                                  # one shot: a sunflower round the centre
            pts = inside(c0 + _sunflower(int(p.clicks), p.spread * rL))
            ok = usable(pts)
            pts = ok if len(ok) >= max(2, len(pts) // 3) else pts
            if not len(pts):
                return None
            res = run_cut(pts)
        else:                                                   # in rounds, like the user
            pts = usable(c0 + _sunflower(max(1, int(p.clicks_first)), p.first_spread * rL))
            if not len(pts):
                return None
            res = run_cut(pts)
            prev = res.mask((lh, lw)) > 0
            for _ in range(max(0, int(p.rounds) - 1)):
                if not prev.any():
                    break
                edge = cv2.distanceTransform(prev.astype(np.uint8), cv2.DIST_L2, 3)
                cand = (edge >= max(1.0, p.edge_keep * brush)) & okL
                if not cand.any():
                    break
                marks = np.ones((lh, lw), np.uint8)
                for q in pts:
                    marks[int(round(q[1])), int(round(q[0]))] = 0
                new = []
                for _ in range(max(1, int(p.clicks_round))):
                    far = cv2.distanceTransform(marks, cv2.DIST_L2, 3)
                    far[~cand] = 0
                    k = int(np.argmax(far))
                    if far.flat[k] < brush:                     # the crown is covered already
                        break
                    y, x = divmod(k, lw)
                    new.append((float(x), float(y)))
                    marks[y, x] = 0
                if not new:
                    break
                pts = np.vstack([pts, np.array(new)])
                res2 = run_cut(pts)
                m2 = res2.mask((lh, lw)) > 0
                same = (m2 & prev).sum() / max(1, (m2 | prev).sum())
                res, prev = res2, m2
                if same >= 0.97:                                # it sits: no more clicks
                    break
        if res is None or len(res.polygon) < 3 or _solidity(res.polygon) < p.solidity_min:
            return None
        m = cv2.resize((res.mask((lh, lw)) * 255).astype(np.uint8), (x1 - x0, y1 - y0),
                       interpolation=cv2.INTER_LINEAR) > 127
        m &= ~stop_c
        if not area_ok(m.sum(), Ri):
            return None
        return to_input(pts), (x0, y0, x1, y1), m

    def run(rows, done0, total):
        n = 0
        for x, y, _, Ri_w in rows:
            if cancel is not None and cancel():
                return False
            n += 1
            if progress is not None:
                progress(done0 + n, total)
            cx, cy = (x + 0.5) / fx - 0.5, (y + 0.5) / fy - 0.5
            Ri = Ri_w / fx
            xi, yi = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
            if claimed[yi, xi] or (reg is not None and not reg[yi, xi]):
                continue
            if p.skip_claimed < 1.0:                           # mostly inside crowns found already: no cut
                rr = max(1, int(0.5 * Ri))
                if claimed[max(0, yi - rr):yi + rr + 1, max(0, xi - rr):xi + rr + 1].mean() >= p.skip_claimed:
                    continue
            tried.append((cx, cy))
            out = cut(cx, cy, Ri, claimed)
            if out is None:
                continue
            clicks, box, m = out
            x0, y0, x1, y1 = box
            claimed[y0:y1, x0:x1] |= m
            crowns.append(dict(c=(cx, cy), clicks=clicks, R=Ri, box=box, m=m))
        return True

    if seeds is not None:
        sd_in = np.asarray(seeds, np.float64).reshape(len(seeds), -1)
        xy_w = np.c_[(sd_in[:, 0] + 0.5) * fx - 0.5, (sd_in[:, 1] + 0.5) * fy - 0.5]
        if sd_in.shape[1] > 2:
            rad = sd_in[:, 2] * fx
        else:
            rad = size_fn(xy_w) if size_fn is not None else np.full(len(sd_in), R)
        rows = np.c_[xy_w, rad, rad]
        rows = rows[(rows[:, 0] >= 0) & (rows[:, 0] <= ww - 1) & (rows[:, 1] >= 0) & (rows[:, 1] <= wh - 1)]
        rows = rows[np.argsort(-rows[:, 3])]
    else:
        rows = _scale_seeds(chan, R, p, float(p.seed_strength), allowed_seeds(), size_fn)
    total = len(rows)
    alive = run(rows, 0, total)
    for _ in range(int(p.residual_passes) if alive else 0):
        free = allowed_seeds().astype(np.uint8)
        dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        extra = _local_maxima(dt, p.seed_distance * R * p.scale_min, p.residual_min * R, free > 0)
        if not len(extra):
            break
        if size_fn is not None:                                 # the size model, capped by the room left
            rad = np.minimum(size_fn(extra[:, :2]), np.maximum(extra[:, 2], p.scale_min * R) * 1.5)
        else:
            rad = np.clip(extra[:, 2], p.scale_min * R, p.scale_max * R)   # the room left says how big
        extra = np.c_[extra[:, :3], rad][np.argsort(-rad)]
        total += len(extra)
        if not run(extra, total - len(extra), total):
            alive = False
            break

    if p.refine and alive and len(crowns) > 1:                  # every crown again, all neighbours known
        for i, cr in enumerate(crowns):
            if cancel is not None and cancel():
                break
            x0, y0, x1, y1 = cr["box"]
            out = cut(cr["c"][0], cr["c"][1], cr["R"], claimed, clicks_in=cr["clicks"], own=(cr["box"], cr["m"]))
            if out is None:
                continue
            _, box2, m2 = out
            claimed[y0:y1, x0:x1] &= ~cr["m"]
            a0, b0, a1, b1 = box2
            claimed[b0:b1, a0:a1] |= m2
            crowns[i] = dict(cr, box=box2, m=m2)

    out: List[CanopyResult] = []
    for cr in crowns:
        x0, y0, x1, y1 = cr["box"]
        c = _largest_contour(cr["m"])
        if c is None or len(c) < 3:
            continue
        poly = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), 0.75, True).reshape(-1, 2).astype(np.float64)
        if len(poly) < 3:
            continue
        poly += (x0, y0)
        if reg is not None:
            q = np.clip(np.round(poly.mean(0)).astype(int), 0, [W - 1, H - 1])
            if not reg[q[1], q[0]]:
                continue
        out.append(CanopyResult(poly, float(cr["m"].sum()), 1, f"{len(cr['clicks'])} clicks"))
    sd = np.array(tried).reshape(-1, 2)
    note = (f"{len(out)} crowns (typical radius {R_in:.0f} px, found at {R:.0f} px, "
            + (f"cut at {p.r_cut:.0f} px" if p.progressive else "one-shot cut")
            + f", {len(sd)} centres tried)" + ("" if alive else ", cancelled"))
    return AutoResult(out, R_in, sd, note)


# ------------------------------------------------------------------------------
# Scoring against annotated crowns, and learning AutoParams from them
# ------------------------------------------------------------------------------

def _poly_area(q: np.ndarray) -> float:
    q = np.asarray(q, np.float64)
    return 0.5 * abs(float(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1))))


def _raster(shape, poly) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    q = np.asarray(poly, np.float64)
    cv2.fillPoly(m, [np.round(q * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
    return m > 0


def score_crowns(shape, preds: Sequence[np.ndarray], truth: Sequence[np.ndarray], margin: float = 0.0) -> dict:
    """Predicted vs annotated crowns (polygons in the same px, pixel centres at integers), matched one to
    one at IoU >= 0.5: crown F1 (``f1``), mean IoU of the matches (``miou``), pixel F1 (``pix``).

    Only crowns centred ``margin`` px or more inside the image are scored, and a prediction lying < 20 %
    on annotated crowns is not counted: what the annotator did not outline is not an error."""
    from scipy.optimize import linear_sum_assignment
    H, W = shape[:2]
    inner = lambda c: margin <= c[0] < W - margin and margin <= c[1] < H - margin
    G = [_raster(shape, g) for g in truth]
    gu = np.zeros((H, W), bool)
    for g in G:
        gu |= g
    gi = [i for i, g in enumerate(truth) if inner(np.mean(g, 0))]
    P = [_raster(shape, q) for q in preds]
    pi = [i for i, q in enumerate(preds) if inner(np.mean(q, 0)) and P[i].any() and (P[i] & gu).sum() >= 0.2 * P[i].sum()]
    iou = np.zeros((len(pi), len(gi)))
    for a, i in enumerate(pi):
        ys, xs = np.nonzero(P[i])
        for b, j in enumerate(gi):
            inter = int(G[j][ys, xs].sum())
            if inter:
                iou[a, b] = inter / float((P[i] | G[j]).sum())
    tp, ious = 0, []
    if iou.size:
        ra, cb = linear_sum_assignment(-iou)
        for a, b in zip(ra, cb):
            if iou[a, b] >= 0.5:
                tp += 1
                ious.append(iou[a, b])
    pu = np.zeros((H, W), bool)
    for i in pi:
        pu |= P[i]
    box = np.zeros((H, W), bool)
    mg = int(margin)
    box[mg:H - mg or H, mg:W - mg or W] = True
    gb, pb = gu & box, pu & box
    return dict(f1=2.0 * tp / max(1, len(pi) + len(gi)), miou=float(np.mean(ious)) if ious else 0.0,
                pix=2.0 * float((gb & pb).sum()) / max(1, int(gb.sum() + pb.sum())),
                n_pred=len(pi), n_gt=len(gi))


# What a search may change, and over what range
SEARCH_SPACE = {
    # radius_scale fine-tunes a KNOWN crown size (from the user's crowns): the image-measured size is too
    # unreliable to tune around (BCI blocks: 8-73 px measured for 58-79 px crowns)
    "r_work": ("float", 16, 45), "radius_scale": ("float", 0.7, 1.4), "channel": ("choice", ["grey", "exg", "green"]),
    "seed_sigma": ("float", 0.4, 1.2), "scale_min": ("float", 0.4, 1.0), "scale_max": ("float", 1.2, 3.0),
    "big_bias": ("float", 0.0, 3.0), "seed_distance": ("float", 0.6, 1.6), "seed_strength": ("log", 0.02, 0.4),
    "shadow_pct": ("float", 5, 40), "clicks": ("int", 5, 16), "spread": ("float", 0.3, 0.9),
    "brush": ("float", 0.1, 0.3), "grow": ("float", 2.0, 6.0), "area_min": ("float", 0.1, 0.5),
    "area_max": ("float", 1.5, 4.0), "solidity_min": ("float", 0.5, 0.9), "residual_passes": ("int", 0, 2),
    "residual_min": ("float", 0.3, 0.9), "refine": ("choice", [False, True]),
    "skip_claimed": ("float", 0.2, 1.0), "cut_px": ("choice", [80, 120, 160]),
    "progressive": ("choice", [False, True]), "r_cut": ("float", 32, 72), "clicks_first": ("int", 3, 8),
    "first_spread": ("float", 0.3, 0.8), "clicks_round": ("int", 2, 6), "rounds": ("int", 2, 5),
    "per_crown_size": ("choice", [False, True]),
}


def sample_params(rng, space=SEARCH_SPACE) -> dict:
    c = {}
    for k, s in space.items():
        if s[0] == "float":
            c[k] = round(rng.uniform(s[1], s[2]), 3)
        elif s[0] == "log":
            c[k] = round(float(np.exp(rng.uniform(np.log(s[1]), np.log(s[2])))), 4)
        elif s[0] == "int":
            c[k] = rng.randint(s[1], s[2])
        else:
            c[k] = rng.choice(s[1])
    return c


def params_dict(p: AutoParams) -> dict:
    return {k: getattr(p, k) for k in SEARCH_SPACE}


def params_from_sessions(sessions: Sequence[dict], base: Optional[AutoParams] = None,
                         min_sessions: int = 5) -> Tuple[AutoParams, dict]:
    """AutoParams that click the way the user did: from recorded Smart Canopy crowns
    (ImageViewer._smart_canopy_record -- strokes and outline in scene units, with the scales).

    Per crown: how many paint strokes, how far from the crown's centre the first ones were (x its
    radius), the brush and grow relative to the crown, and the zoom the graph cut actually worked at
    (crown radius in its px: the brush's 160 px working region over the source px it covered).
    Medians over the crowns go into ``base``; with fewer than ``min_sessions`` crowns it is returned
    as is.  Returns (params, the measured profile)."""
    import math
    base = base or AutoParams()
    rows = []
    for s in sessions:
        try:
            poly = np.asarray(s["polygon"], np.float64)
            R = math.sqrt(_poly_area(poly) / math.pi)
            paint = [st for st in s["strokes"] if not st.get("erase")]
            if R <= 0 or not paint:
                continue
            c = poly.mean(0)
            clicks = np.array([np.mean(np.asarray(st["pts"], np.float64), 0) for st in paint])
            d = np.hypot(*(clicks - c).T) / R
            r = float(np.median([st["r"] for st in paint]))
            grow = float(s.get("grow", base.grow))
            allp = np.vstack([np.asarray(st["pts"], np.float64) for st in paint])
            side = float(np.ptp(allp, axis=0).max()) + 2.0 * (r + grow * r) + 4.0     # scene units
            side_src = side * float(s.get("source_per_scene", 1.0))
            r_cut = R * float(s.get("source_per_scene", 1.0)) * min(1.0, 160.0 / max(side_src, 1.0))
            rows.append(dict(n=len(paint), first=float(np.max(d[:3])), brush=r / R, grow=grow, r_cut=r_cut,
                             screen_R=R * float(s.get("screen_per_scene", 1.0))))
        except Exception:
            continue
    prof = {k: float(np.median([r_[k] for r_ in rows])) for k in ("n", "first", "brush", "grow", "r_cut", "screen_R")} \
        if rows else {}
    prof["crowns"] = len(rows)
    if len(rows) < min_sessions:
        return base, prof
    n = max(1, int(round(prof["n"])))
    first = min(3, n)
    per_round = 3
    p = replace(base, clicks_first=first, first_spread=float(np.clip(prof["first"], 0.1, 0.9)),
                clicks_round=per_round, rounds=1 + int(math.ceil(max(0, n - first) / per_round)),
                brush=float(np.clip(prof["brush"], 0.05, 0.5)), grow=float(np.clip(prof["grow"], 0.0, 8.0)),
                r_cut=float(np.clip(prof["r_cut"], 16.0, 120.0)), clicks=n, progressive=True)
    return p, prof


def learn_params(
    blocks: Sequence[Tuple[np.ndarray, Sequence[np.ndarray]]],
    start: Optional[AutoParams] = None,
    n_random: int = 40,
    workers: int = 4,
    seed: int = 0,
    progress: Optional[Callable[[int, int, float], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    refine: bool = True,
    size_model: Optional[dict] = None,
    cap_s: float = 60.0,
) -> Tuple[AutoParams, dict, dict]:
    """AutoParams fitted to annotated crowns: ``blocks`` = [(rgb, [crown polygons in its px]), ...].

    Random search around nothing (``n_random`` configurations, plus ``start``), then one parameter at a
    time around the best.  Threads (GrabCut releases the GIL), so it runs inside the app.  Returns
    (best params, its score, the score of ``start``); ``progress(done, total, best_f1)``."""
    import random
    from concurrent.futures import ThreadPoolExecutor

    start = start or AutoParams()
    rng = random.Random(seed)
    margin = [0.0 for _ in blocks]

    def evaluate(cfg):
        rs = []
        for (img, truth), mg in zip(blocks, margin):
            # a setting too slow to use is cut off after cap_s and scored on what it found by then
            t_end = time.monotonic() + float(cap_s)
            stop = lambda: (cancel is not None and cancel()) or time.monotonic() > t_end
            res = delineate_all(img, replace(start, **cfg), cancel=stop, size_model=size_model)
            rs.append(score_crowns(img.shape, [c.polygon for c in res.crowns], truth, mg))
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

    base = params_dict(start)
    cfgs = [base] + [sample_params(rng) for _ in range(max(0, int(n_random)))]
    refine_n = sum(1 if s[0] != "choice" else len(s[1]) - 1 for s in SEARCH_SPACE.values()) * 2 if refine else 0
    total = len(cfgs) + refine_n
    done, best_f1 = 0, -1.0
    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        scores = []
        for s in ex.map(evaluate, cfgs):
            scores.append(s)
            done += 1
            best_f1 = max(best_f1, s["f1"])
            if progress is not None:
                progress(done, total, best_f1)
            if cancel is not None and cancel():
                break
        start_score = scores[0]
        best, bs = max(zip(cfgs, scores), key=lambda t: t[1]["f1"])
        for k, s in (SEARCH_SPACE.items() if refine else ()):
            if cancel is not None and cancel():
                break
            if s[0] == "choice":
                opts = [v for v in s[1] if v != best[k]]
            elif s[0] == "int":
                opts = sorted({max(s[1], best[k] - 2), min(s[2], best[k] + 2)} - {best[k]})
            else:
                opts = [round(best[k] * m, 4) for m in (0.75, 1.3)]
            cs = [dict(best, **{k: o}) for o in opts]
            for c, s_ in zip(cs, ex.map(evaluate, cs)):
                if s_["f1"] > bs["f1"] + 0.003:
                    best, bs = c, s_
            done += 2
            if progress is not None:
                progress(min(done, total), total, bs["f1"])
    return replace(start, **best), bs, start_score


__all__ = ["AutoParams", "AutoResult", "delineate_all", "estimate_crown_radius", "score_crowns",
           "learn_params", "params_from_sessions", "SEARCH_SPACE", "sample_params", "params_dict",
           "SIZE_MODEL_BCI", "size_samples", "fit_size_model", "adapt_size_model", "predict_size"]
