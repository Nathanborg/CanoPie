"""Smart canopy brush: brush strokes over a tree crown become its outline.

Brush strokes are seeds, as in RoosterFly's smart brush (``exp_realism/
root_tracer.py``): what the user paints is crown, what they erase is not, and
the outline snaps to the crown's own edges around the paint.  Roots are thin
ridges on soil, so RoosterFly traces them along a crest; crowns in a closed
canopy touch their neighbours, and the edge between two crowns is a change of
colour / texture and a shadow, not a drop to a background level.  So here the
seeds drive a graph cut (GrabCut: colour models of crown and surroundings plus
an edge-aware smoothness term), the way a "quick selection" brush works:

1. Seeds.  The painted band (stroke x brush radius) is crown for sure.
   Erased bands, polygons outlined already (``stop_mask``) and everything
   beyond a reach around the paint are not crown for sure.  The reach grows
   with the brush and the stroke: a click reaches further than a stroke,
   which already says how big the crown is.
2. Cut.  GrabCut on a blurred CIELAB copy at a working resolution of a few
   hundred px -- the blur leaves crown-level colour, not leaf-level texture --
   decides the band in between.
3. Shape.  The part connected to the paint is kept, holes filled, and its
   outline smoothed along the contour (crowns are rounded; a cut follows
   every leaf), never shrinking off the paint.

Tuned on annotated crowns of four CanoPie projects (BCI orthomosaic COG,
1 m airborne RGB, phenocam, MicaSense radiance band).  No Qt: ``ImageViewer``
hands in the displayed pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Stroke = Tuple[np.ndarray, float, bool]        # (Nx2 points (x, y) image px, brush radius px, erase)


@dataclass
class CanopyParams:
    # Not crown beyond this x the brush radius from the paint (the user's control: 0 = exactly the
    # paint; the viewer's - / + change it live).  Optimised for ~10 clicks inside a crown on 80 BCI
    # crowns: brush ~1/6 of the crown radius x grow 4 -> F1 0.83 on 40 held-out ones (0.86 at 15 clicks).
    grow: float = 4.0
    probable: float = 0.5         # inside this share of the reach the cut starts from "probably crown"
    # on the annotated crowns: blur 1 -> 2 raised a click's IoU 0.57 -> 0.60, 200 -> 160 px halved the time
    work_px: int = 160            # working resolution: the region's longer side
    blur: float = 2.0             # Gaussian blur at working resolution (px): crown colour, not leaves
    iterations: int = 3
    contour_sigma: float = 0.02   # outline smoothing, x the contour's length (0 = off)
    max_radius_px: float = 0.0    # cap on the reach (0 = none); the viewer passes ~ what is on screen


@dataclass
class CanopyResult:
    polygon: np.ndarray           # Nx2 float (x, y), image px (pixel centres at integers); empty if none
    area_px: float = 0.0
    pieces: int = 0               # separate parts the paint selected (joined into the one polygon)
    note: str = ""

    def mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """HxW uint8 {0, 1} of the outline."""
        m = np.zeros(shape[:2], np.uint8)
        if len(self.polygon) >= 3:
            cv2.fillPoly(m, [np.round(self.polygon * 16).astype(np.int32).reshape(-1, 1, 2)], 1,
                         lineType=cv2.LINE_8, shift=4)
        return m


def _band(shape, strokes: Sequence[Tuple[np.ndarray, float]]) -> np.ndarray:
    """The painted area of ``strokes`` (points, radius) on an HxW grid."""
    m = np.zeros(shape[:2], np.uint8)
    for pts, r in strokes:
        ip = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32).reshape(-1, 1, 2)
        rr = max(1, int(round(r * 16)))
        if len(ip) == 1:
            cv2.circle(m, tuple(int(v) for v in ip[0, 0]), rr, 1, -1, shift=4)
        else:
            cv2.polylines(m, [ip], False, 1, thickness=max(1, int(round(2 * r))), lineType=cv2.LINE_8, shift=4)
            for p in (ip[0, 0], ip[-1, 0]):                   # round ends
                cv2.circle(m, (int(p[0]), int(p[1])), rr, 1, -1, shift=4)
    return m > 0


def _keep_touching(sel: np.ndarray, seed: np.ndarray) -> Tuple[np.ndarray, int]:
    """Components of ``sel`` that touch ``seed``, holes filled; and how many there are."""
    n, lab = cv2.connectedComponents(sel.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[seed & (lab > 0)])
    if len(keep) == 0:
        return np.zeros_like(sel, bool), 0
    out = np.isin(lab, keep)
    n2, lab2 = cv2.connectedComponents((~out).astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])
    out |= (lab2 > 0) & ~np.isin(lab2, border)
    return out, len(keep)


def _join_pieces(sel: np.ndarray, limit: float) -> Tuple[np.ndarray, int]:
    """One shape from the pieces the paint selected: they are all the same crown (separate clicks
    that did not grow into each other), so close the gaps -- with the smallest disc that joins them,
    up to ``limit`` px, which keeps the outline rounded.  Returns it and how many pieces there were."""
    n, _ = cv2.connectedComponents(sel.astype(np.uint8), connectivity=8)
    pieces = n - 1
    if pieces <= 1:
        return sel, max(pieces, 0)
    k = 2
    while k <= max(2, limit):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        pad = k + 1
        big = np.pad(sel.astype(np.uint8), pad)
        closed = cv2.morphologyEx(big, cv2.MORPH_CLOSE, se)[pad:-pad, pad:-pad] > 0
        if cv2.connectedComponents(closed.astype(np.uint8), connectivity=8)[0] - 1 <= 1:
            return closed | sel, pieces
        k = int(np.ceil(k * 1.5))
    return sel, pieces


def _largest_contour(sel: np.ndarray) -> Optional[np.ndarray]:
    cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def _smooth_closed(c: np.ndarray, sigma: float) -> np.ndarray:
    n = len(c)
    if n < 12 or sigma <= 0:
        return c
    r = min(int(np.ceil(3 * sigma)), n // 2 - 1)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2.0 * sigma * sigma))
    k /= k.sum()
    return np.stack([np.convolve(np.pad(c[:, j], r, mode="wrap"), k, mode="valid") for j in range(2)], 1)


def select_canopy(
    rgb: np.ndarray,
    strokes: Sequence[Stroke],
    params: Optional[CanopyParams] = None,
    stop_mask: Optional[np.ndarray] = None,
) -> CanopyResult:
    """The crown ``strokes`` select on ``rgb`` (HxWx3 uint8 RGB, HxWx1 or HxW).

    ``strokes``: (points Nx2 (x, y) in image px with pixel centres at integers, brush radius px, erase);
    a single point is a click.  ``stop_mask`` (HxW): crowns outlined already -- never selected, not
    even under the paint (so crowns never overlap); erased bands likewise."""
    p = params or CanopyParams()
    img = np.asarray(rgb)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = np.ascontiguousarray(img[..., :3])
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    H, W = img.shape[:2]
    paint = [(np.asarray(s[0], np.float64).reshape(-1, 2), float(s[1])) for s in strokes if not s[2]]
    erase = [(np.asarray(s[0], np.float64).reshape(-1, 2), float(s[1])) for s in strokes if s[2]]
    if not paint:
        return CanopyResult(np.zeros((0, 2)), note="nothing painted")

    pts = np.vstack([q for q, _ in paint])
    r = max(r_ for _, r_ in paint)
    # how far past the paint the crown may grow: the brush's own size x the user's grow factor (it used
    # to scale with the stroke's length as well, and a stroke across a closed canopy then took neighbours)
    reach = max(0.0, float(p.grow)) * r
    if p.max_radius_px > 0:
        reach = min(reach, max(p.max_radius_px, 2.0 * r))

    # region: the paint + the reach, at a working resolution
    x0 = int(max(0, np.floor(pts[:, 0].min() - r - reach - 2)))
    x1 = int(min(W, np.ceil(pts[:, 0].max() + r + reach + 3)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - r - reach - 2)))
    y1 = int(min(H, np.ceil(pts[:, 1].max() + r + reach + 3)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return CanopyResult(np.zeros((0, 2)), note="outside the image")
    sub = img[y0:y1, x0:x1]
    f = min(1.0, float(p.work_px) / max(sub.shape[:2]))
    ws = (max(8, int(round(sub.shape[1] * f))), max(8, int(round(sub.shape[0] * f))))
    s = cv2.resize(sub, ws, interpolation=cv2.INTER_AREA) if f < 1.0 else sub.copy()
    if p.blur > 0:
        s = cv2.GaussianBlur(s, (0, 0), p.blur)
    s = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)

    def to_work(q):
        return (q - (x0, y0) + 0.5) * f - 0.5

    fg = _band(s.shape, [(to_work(q), max(0.75, r_ * f)) for q, r_ in paint])
    d = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 3)
    m = np.full(d.shape, cv2.GC_PR_BGD, np.uint8)
    m[d <= p.probable * reach * f] = cv2.GC_PR_FGD
    m[d > reach * f] = cv2.GC_BGD
    hard = np.zeros(d.shape, bool)                              # never crown: neighbours, erased bands
    if stop_mask is not None:
        st = stop_mask[y0:y1, x0:x1].astype(np.uint8)
        hard |= (cv2.resize(st, ws, interpolation=cv2.INTER_NEAREST) if f < 1.0 else st) > 0
    if erase:
        hard |= _band(s.shape, [(to_work(q), max(0.75, r_ * f)) for q, r_ in erase])
    # a brush overlapping a crown outlined already, or an erased band, does not take it: crowns never overlap
    fg &= ~hard
    if not fg.any():
        return CanopyResult(np.zeros((0, 2)), note="the paint is all on crowns outlined already / erased")
    m[hard] = cv2.GC_BGD
    m[fg] = cv2.GC_FGD
    note = ""
    if (m == cv2.GC_BGD).any() and ((m == cv2.GC_PR_FGD) | (m == cv2.GC_PR_BGD)).any():
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(np.ascontiguousarray(s), m, None, bgd, fgd, int(p.iterations), cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:                                  # e.g. a flat region: one colour cluster
            note = f" (cut skipped: {str(e).splitlines()[-1][:60]})"
            m[(m == cv2.GC_PR_FGD) | (m == cv2.GC_PR_BGD)] = cv2.GC_BGD
    sel = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    sel, _ = _keep_touching(sel & ~hard, fg)
    sel, pieces = _join_pieces(sel, 0.5 * max(sel.shape))
    if pieces > 1:
        sel, _ = _keep_touching(sel & ~hard, fg)                # fill what joining enclosed -- never a neighbour
    sel = (sel & ~hard) | fg                                    # (hole filling may enclose one: not crown either)

    c = _largest_contour(sel)
    if c is None:
        return CanopyResult(np.zeros((0, 2)), note="nothing selected" + note)
    if p.contour_sigma > 0:
        c = _smooth_closed(c, max(1.0, p.contour_sigma * len(c)))
        sm = np.zeros(sel.shape, np.uint8)
        cv2.fillPoly(sm, [np.round(c * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
        sel2, _ = _keep_touching(((sm > 0) & ~hard) | fg, fg)   # never off the paint, never on a neighbour
        c2 = _largest_contour((sel2 & ~hard) | fg)
        c = c2 if c2 is not None else c
    # back to image px, simplified to about a working pixel
    c = (c + 0.5) / f - 0.5 + (x0, y0)
    approx = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), 0.5 / f, True).reshape(-1, 2)
    poly = approx.astype(np.float64) if len(approx) >= 8 else c
    poly[:, 0] = np.clip(poly[:, 0], -0.5, W - 0.5)
    poly[:, 1] = np.clip(poly[:, 1], -0.5, H - 0.5)
    area = 0.5 * abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))))
    rad = np.sqrt(area / np.pi)
    note = (f"crown ~{2 * rad:.0f} px across ({len(paint)} stroke{'s' * (len(paint) != 1)}"
            + (f", {len(erase)} erased" if erase else "") + ")"
            + (f", {pieces} separate parts joined" if pieces > 1 else "") + note)
    return CanopyResult(poly, area, pieces, note)


__all__ = ["CanopyParams", "CanopyResult", "Stroke", "select_canopy"]
