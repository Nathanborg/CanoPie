import numpy as np
import cv2
import math
import logging
from PyQt5 import QtCore, QtGui


def _base_valid_mask(image, nodata_values=None):
    """
    Data-based validity (uint8 255=valid, 0=invalid).

    A pixel is valid if AT LEAST ONE channel carries usable data, where "usable"
    means finite AND not equal to a declared NoData value.

    Taking NoData into account is essential, not cosmetic. In the AVIRIS
    prediction stacks the science bands mark NoData with NaN but the ancillary
    bands (angles, WV_AOT, MASK) use -9999, which is perfectly finite. Judging
    validity on finiteness alone therefore called 100% of pixels valid on images
    that are 93.7% NoData, and random shapes were scattered over empty space.
    With the NoData values applied the same images report 6.3% valid, which is
    the true footprint.
    """
    arr = image if image.ndim == 3 else image[..., None]

    numeric_nd = []
    for v in (nodata_values or []):
        try:
            nv = float(v)
        except (TypeError, ValueError):
            continue                      # expressions are handled by the .ax mask
        if nv == nv:                      # NaN is covered by the isfinite test
            numeric_nd.append(nv)

    is_float = np.issubdtype(arr.dtype, np.floating)
    usable = np.isfinite(arr) if is_float else np.ones(arr.shape, dtype=bool)

    if numeric_nd:
        # A declared NoData value is authoritative: everything else is data.
        # The old `arr > 0` guess for integer rasters silently threw away class
        # 0 -- in the liana predictions (0=no_liana, 1=liana, 255=NoData) that
        # discarded most of the actually-valid area.
        af = arr.astype(np.float32, copy=False)
        for nv in numeric_nd:
            a = abs(nv)
            # Same tolerance policy as utils.build_nodata_mask.
            tol = a * 0.001 if a > 100 else 0.01
            usable &= ~(np.abs(af - nv) < tol)
    elif not is_float:
        # No NoData declared: fall back to the historical heuristic.
        usable &= arr > 0

    # Prefer pixels usable in EVERY sampled band. Extraction needs all of the
    # primary bands at once -- the point path drops a sample outright if R, G or
    # B is NoData -- so placing shapes where only some bands carry data produces
    # shapes that look valid but yield no CSV row. The science bands here have
    # slightly different footprints (P(liana) 6.3% vs NDVI/NIR 6.5%), and that
    # difference alone was enough to lose samples.
    #
    # Bands that are 100% fill are excluded from that "every band" test. They
    # say nothing about where data exists, but they would make the strict rule
    # match nothing and silently drop us to the far looser "any band" fallback.
    # That is exactly what happened between the two callers: the file worker
    # passes 3 science bands (strict applies), while a live viewer passes the
    # whole 15-band stack, where the all-fill WV_AOT/MASK planes forced the
    # fallback and admitted pixels that only an angle band covered -- so the
    # same image produced different placements depending on whether it was
    # open on screen.
    band_has_data = usable.any(axis=(0, 1))
    if band_has_data.any():
        strict = np.all(usable[..., band_has_data], axis=2)
        if strict.any():
            return strict.astype(np.uint8) * 255

    # Nothing survives the strict rule (or every band is fill): fall back to
    # "usable in at least one band" rather than leaving the user with nothing.
    return np.any(usable, axis=2).astype(np.uint8) * 255


def _build_valid_mask(image, nodata_mask, nodata_values=None):
    """
    Build a uint8 valid-pixel mask (255=valid, 0=invalid).

    Combines data-based validity (`_base_valid_mask`, which treats a pixel as
    valid if ANY channel is finite/non-zero) with an optional explicit NoData
    mask from `.ax` nodata_values.

    IMPORTANT: `build_nodata_mask` flags a pixel as NoData if ANY channel is
    NaN/Inf. For multi-band stacks with an all-NaN band (e.g. AVIRIS
    `*_prob_*_stack.tif`) that marks EVERY pixel as NoData. If applying the
    NoData mask would remove all pixels, we ignore it and fall back to the
    data-based mask instead of generating nothing.
    """
    H, W = image.shape[:2]
    base = _base_valid_mask(image, nodata_values)

    if nodata_mask is not None and np.asarray(nodata_mask).shape[:2] == (H, W):
        nd = np.asarray(nodata_mask, dtype=bool)
        combined = base.copy()
        combined[nd] = 0
        if int(cv2.countNonZero(combined)) > 0:
            return combined
        logging.warning("[random_shapes] NoData mask excluded every pixel "
                        "(likely an all-NaN band); ignoring it and using "
                        "data-based validity instead.")

    return base


def _px_dims(params, gsd_m_per_px):
    """Resolve user-unit params to pixel dimensions. Returns a dict of px values."""
    if gsd_m_per_px and gsd_m_per_px > 0:
        scale = 1.0 / gsd_m_per_px   # user units = meters -> pixels
    else:
        scale = 1.0                   # user units = pixels
    return {
        "diameter": float(params.get("diameter", 1.0)) * scale,
        "width":    float(params.get("width",    1.0)) * scale,
        "height":   float(params.get("height",   1.0)) * scale,
        "border":   float(params.get("min_dist_border", 0.0)) * scale,
        "spacing":  float(params.get("min_dist_shapes", 0.0)) * scale,
    }


class _SpacingIndex:
    """Uniform grid for 'is this point at least `min_dist` from all accepted?'.

    The spacing check used to compare each candidate against every accepted
    point with a fresh numpy array, which is O(accepted) per candidate and
    rebuilt a growing array each time. Bucketing by a cell of min_dist/sqrt(2)
    means only the 5x5 neighbouring cells can hold a conflicting point, so the
    test is effectively O(1) and the whole pass gets several times faster --
    which matters because a tight spacing forces the sampler to walk a large
    fraction of the valid pixels before it can conclude no more will fit.
    """

    __slots__ = ("_d2", "_cell", "_grid", "_enabled")

    def __init__(self, min_dist_sq):
        self._d2 = float(min_dist_sq)
        self._enabled = self._d2 > 0
        self._cell = math.sqrt(self._d2) / math.sqrt(2.0) if self._enabled else 1.0
        if self._cell <= 0:
            self._cell = 1.0
        self._grid = {}

    def accepts(self, x, y):
        if not self._enabled:
            return True
        cell = self._cell
        cx, cy = int(x // cell), int(y // cell)
        grid = self._grid
        for a in range(cx - 2, cx + 3):
            for b in range(cy - 2, cy + 3):
                bucket = grid.get((a, b))
                if not bucket:
                    continue
                for qx, qy in bucket:
                    dx = qx - x
                    dy = qy - y
                    if dx * dx + dy * dy < self._d2:
                        return False
        return True

    def add(self, x, y):
        if not self._enabled:
            return
        cell = self._cell
        key = (int(x // cell), int(y // cell))
        self._grid.setdefault(key, []).append((x, y))


def _dart_throw(sample_batch, count, min_dist_sq, max_attempts):
    """
    Generic rejection sampler with a spacing constraint.

    `sample_batch(n)` must return two arrays (xs, ys) of n candidate centres.
    Returns a list of accepted (x, y) tuples (<= count).
    """
    accepted = []
    index = _SpacingIndex(min_dist_sq)
    drawn = 0
    while len(accepted) < count and drawn < max_attempts:
        n = int(min(4096, max_attempts - drawn))
        xs, ys = sample_batch(n)
        drawn += n
        for px, py in zip(xs.tolist(), ys.tolist()):
            if len(accepted) >= count:
                break
            px = float(px); py = float(py)
            if not index.accepts(px, py):
                continue
            accepted.append((px, py))
            index.add(px, py)
    return accepted


def _stratified_centers(valid_mask, count, min_dist_sq, rng, max_tries_per_visit=4096):
    """Round-robin sample centres across the disconnected valid patches.

    Uniform sampling over valid pixels is area-proportional: a patch holding
    75% of the valid pixels legitimately receives ~75% of the shapes. That is
    the statistically unbiased default, but on a fragmented footprint it reads
    as "all the shapes are in one corner". This spreads the requested count
    across the separate patches instead, so a small patch gets comparable
    representation to a large one.

    Patches are visited in turn, one shape each pass, until the count is met or
    every patch is exhausted -- so patches that can only hold a couple of shapes
    (because they are tiny, or because spacing rejects the rest) simply drop out
    and the remainder is shared among the others rather than being lost.

    Returns a list of (x, y), or None when there is nothing to stratify (a
    single patch), letting the caller use the normal uniform path.
    """
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
    if n_cc <= 2:                      # background + at most one patch
        return None

    ys_all, xs_all = np.where(valid_mask > 0)
    if ys_all.size == 0:
        return None
    labs = labels[ys_all, xs_all]

    # Group pixel indices by patch in one pass rather than scanning the mask per
    # patch (images here can carry dozens of patches over ~10^5 valid pixels).
    order = np.argsort(labs, kind="stable")
    sorted_labs = labs[order]
    uniq, starts = np.unique(sorted_labs, return_index=True)
    starts = list(starts) + [sorted_labs.size]

    # Largest patch first, so with a count too small to reach everything the
    # shapes still favour the substantial patches over single-pixel specks.
    areas = {int(u): int(stats[int(u), cv2.CC_STAT_AREA]) for u in uniq}
    grouped = []
    for k, u in enumerate(uniq):
        idx = order[starts[k]:starts[k + 1]]
        idx = idx.copy()
        rng.shuffle(idx)
        grouped.append((areas.get(int(u), idx.size), list(idx)))
    grouped.sort(key=lambda t: -t[0])
    pools = [g[1] for g in grouped]

    accepted = []
    index = _SpacingIndex(min_dist_sq)
    active = [p for p in pools if p]
    while len(accepted) < count and active:
        made_progress = False
        for pool in list(active):
            if len(accepted) >= count:
                break
            tries = 0
            while pool and tries < max_tries_per_visit:
                tries += 1
                i = pool.pop()
                px, py = float(xs_all[i]), float(ys_all[i])
                if not index.accepts(px, py):
                    continue
                accepted.append((px, py))
                index.add(px, py)
                made_progress = True
                break
            if not pool:
                active.remove(pool)
        if not made_progress:
            break                      # spacing blocks every remaining pixel
    return accepted


def _shape_footprint_px(shape_type, dims):
    """
    Half-width/half-height a shape actually occupies around its centre.

    THIS IS THE PIECE THE CONTAINMENT LOGIC BELOW WAS MISSING: placement only ever
    validated the CENTRE point against the border margin / valid mask, never the
    shape's own extent. A Circle with diameter=80px placed with its centre 1px
    inside a valid patch still sticks ~39px into the invalid area on every side —
    "weird shapes" straddling the NoData boundary is exactly what that produces,
    and it is invisible in the fast (non-restricted) path only because there is no
    visible valid/invalid boundary there to straddle.

    Returns (half_w, half_h) in pixels:
      - Point:     (0, 0)      — no area, nothing to contain
      - Circle:    (r, r)      — isotropic, r = diameter / 2
      - Rectangle: (w/2, h/2)  — independent per axis
    """
    if shape_type == "Circle":
        r = dims["diameter"] / 2.0
        return r, r
    if shape_type == "Rectangle":
        return dims["width"] / 2.0, dims["height"] / 2.0
    return 0.0, 0.0   # Point


def _sample_centers(params, dims, H, W, image=None, nodata_mask=None, nodata_values=None):
    """
    Choose shape centres. Returns list of (x, y) in image pixel coords.

    Fast path (default, `restrict_valid` is False): pick centres uniformly inside
    the image rectangle (inset by the border distance AND the shape's own half-size,
    so the shape itself — not just its centre — stays on-canvas). This needs only
    the image WIDTH/HEIGHT — no pixel decode, no mask — which is what makes file /
    GeoTIFF processing fast and parallelisable.

    Restricted path (`restrict_valid` is True and an `image` array is given):
    keep the ENTIRE shape inside the valid (non-NaN / non-NoData) area, using a
    shape-aware erosion (see _shape_footprint_px) — not just its centre.
    """
    count       = int(params["count"])
    min_dist_sq = dims["spacing"] ** 2
    border_px   = int(math.ceil(dims["border"]))
    shape_type  = str(params.get("shape_type", "Point"))
    half_w, half_h = _shape_footprint_px(shape_type, dims)
    inset_x = border_px + int(math.ceil(half_w))
    inset_y = border_px + int(math.ceil(half_h))
    restrict    = bool(params.get("restrict_valid", False)) and image is not None
    rng = np.random.default_rng()

    def _sample_rect(ix, iy):
        """Uniformly sample `count` centres in the frame, inset by (ix, iy) px."""
        x0, y0 = ix, iy
        x1, y1 = W - ix, H - iy
        if x1 <= x0 or y1 <= y0:          # inset too large -> use full frame
            x0, y0, x1, y1 = 0, 0, W, H
        if min_dist_sq <= 0:
            xs = rng.uniform(x0, x1, size=count)
            ys = rng.uniform(y0, y1, size=count)
            return list(zip(xs.tolist(), ys.tolist()))
        return _dart_throw(
            lambda n: (rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)),
            count, min_dist_sq, max_attempts=max(count * 50, 20000),
        )

    # ---------------------------------------------------------------- #
    # Fast path: rectangle only, no pixels read.
    # ---------------------------------------------------------------- #
    if not restrict:
        return _sample_rect(inset_x, inset_y)

    # ---------------------------------------------------------------- #
    # Restricted path: erode the valid mask by (border + shape half-size) so a
    # surviving centre guarantees the WHOLE shape lands in the valid area.
    #
    # Kernel shape matters for exactness:
    #   - Rectangle needs a RECTANGULAR structuring element sized to its own
    #     width/height. An elliptical kernel is inscribed inside that box, so its
    #     corners would under-erode — a centre could pass while the rectangle's
    #     corners still land on invalid pixels.
    #   - Circle/Point are isotropic, so an elliptical (here: circular, since
    #     inset_x == inset_y for both) kernel is exact.
    # ---------------------------------------------------------------- #
    valid_mask = _build_valid_mask(image, nodata_mask, nodata_values)

    # A Point has zero footprint, so with no border it gets no erosion at all and
    # a centre sampled as a float can round onto an invalid neighbour. Erode by a
    # single pixel so the rounded coordinate is guaranteed valid too.
    if inset_x <= 0 and inset_y <= 0:
        inset_x = inset_y = 1

    if (inset_x > 0 or inset_y > 0) and int(cv2.countNonZero(valid_mask)) > 0:
        base_kw, base_kh = inset_x * 2 + 1, inset_y * 2 + 1
        if base_kw > W or base_kh > H:
            logging.warning(f"[random_shapes] Shape footprint + border ({inset_x}x{inset_y}px) "
                            f"is too large for the {W}x{H} image; ignoring containment margin "
                            f"(shapes may extend past the valid area).")
        else:
            morph = cv2.MORPH_RECT if shape_type == "Rectangle" else cv2.MORPH_ELLIPSE
            # getStructuringElement's ellipse is a rasterised approximation that
            # sits slightly inside the true disc, and shape vertices are rounded
            # to integer pixels. One extra pixel of erosion makes containment
            # hold for the whole outline rather than ~90% of it.
            ellipse_pad = 2 if morph == cv2.MORPH_ELLIPSE else 0

            # Progressive erosion, not all-or-nothing. Some of these prediction
            # tiles have a valid footprint of only a few hundred scattered
            # pixels split across a handful of small patches (a genuinely tiny
            # fraction of the frame -- see the with_angles.tif reproduction:
            # 966 of 3.4M pixels). If the full margin doesn't survive erosion
            # ANYWHERE, jumping straight to "no margin at all" scatters shapes
            # across the WHOLE frame with zero containment guarantee -- which
            # produced the paradox of a LARGER border margin giving a WORSE
            # (wider, uncontained) result than a smaller one, at the exact
            # threshold where the largest patch stopped fitting the margin.
            # Relaxing the margin in steps keeps shapes fully inside real data
            # whenever any margin at all still fits, and only drops to
            # centre-only containment when even a token margin does not.
            eroded_mask = None
            used_fraction = 0.0
            for frac in (1.0, 0.75, 0.5, 0.25, 0.1):
                ix = max(0, int(round(inset_x * frac)))
                iy = max(0, int(round(inset_y * frac)))
                kw = min(ix * 2 + 1 + ellipse_pad, W)
                kh = min(iy * 2 + 1 + ellipse_pad, H)
                if kw < 1 or kh < 1:
                    continue
                kernel = cv2.getStructuringElement(morph, (kw, kh))
                candidate = cv2.erode(valid_mask, kernel, iterations=1)
                if int(cv2.countNonZero(candidate)) > 0:
                    eroded_mask = candidate
                    used_fraction = frac
                    break

            if eroded_mask is not None:
                valid_mask = eroded_mask
                if used_fraction < 1.0:
                    logging.warning(
                        "[random_shapes] Full containment margin (%dx%d px) left no "
                        "valid centres in this image; relaxed to %.0f%% (%dx%d px) so "
                        "shapes still land fully inside valid data, just closer to its "
                        "edge." % (inset_x, inset_y, used_fraction * 100,
                                  int(inset_x * used_fraction), int(inset_y * used_fraction)))
            else:
                logging.warning("[random_shapes] Even a minimal containment margin left no "
                                f"valid centres (shape footprint {2*half_w:.1f}x{2*half_h:.1f}px "
                                f"vs. the available valid area); falling back to "
                                "centre-only containment (shapes may extend past the valid area).")

    n_valid = int(cv2.countNonZero(valid_mask))
    if n_valid == 0:
        # Nothing usable in restricted mode — don't leave the user empty-handed.
        logging.warning(
            "[random_shapes] No valid pixels for restricted placement "
            f"(dtype={getattr(image, 'dtype', None)}, shape={getattr(image, 'shape', None)}, "
            f"had_nodata_mask={nodata_mask is not None}); falling back to full frame.")
        # _sample_rect takes SEPARATE x/y insets; passing one argument raised
        # TypeError, so this "don't leave the user empty-handed" fallback
        # crashed instead of falling back. Hit whenever restricted placement
        # finds no valid pixels at all -- i.e. exactly the heavily-masked
        # rasters this option exists for.
        return _sample_rect(border_px, border_px)

    # Optional: even coverage across the separate valid patches instead of
    # uniform-over-area. Runs on the SAME eroded mask, so full-shape containment
    # is unchanged -- only which valid pixels get chosen differs.
    if bool(params.get("stratify", False)):
        stratified = _stratified_centers(valid_mask, count, min_dist_sq, rng)
        if stratified:
            logging.info("[random_shapes] Stratified placement: %d shape(s) spread "
                         "across the separate valid areas.", len(stratified))
            return stratified
        if stratified is not None:
            logging.warning("[random_shapes] Stratified placement produced nothing; "
                            "falling back to uniform sampling.")
        # stratified is None -> only one patch, nothing to stratify: fall through.

    ENUM_CAP = 3_000_000
    if n_valid <= ENUM_CAP:
        ys_all, xs_all = np.where(valid_mask > 0)
        if min_dist_sq <= 0:
            sel = rng.choice(n_valid, size=min(count, n_valid), replace=False)
            return [(float(xs_all[i]), float(ys_all[i])) for i in sel]
        accepted = []
        index = _SpacingIndex(min_dist_sq)
        for i in rng.permutation(n_valid):
            if len(accepted) >= count:
                break
            px, py = float(xs_all[i]), float(ys_all[i])
            if not index.accepts(px, py):
                continue
            accepted.append((px, py))
            index.add(px, py)
        if len(accepted) < count:
            logging.info("[random_shapes] Placed %d of %d requested: with a %.1f px "
                         "minimum separation the valid area cannot hold more.",
                         len(accepted), count, math.sqrt(min_dist_sq))
        return accepted

    # Huge, dense valid area -> rejection sample against the mask.
    def _mask_batch(n):
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)
        ok = valid_mask[ys, xs] > 0
        return xs[ok].astype(np.float64), ys[ok].astype(np.float64)

    return _dart_throw(_mask_batch, count, max(min_dist_sq, 1e-9),
                       max_attempts=max(count * 50, 20000))


def _center_points(cx, cy, shape_type, dims):
    """Return a shape as a plain list of [x, y] point pairs (closed for areas)."""
    if shape_type == "Point":
        return [[cx, cy]]

    if shape_type == "Circle":
        radius = dims["diameter"] / 2.0
        N = 32
        pts = [[cx + radius * math.cos(2.0 * math.pi * k / N),
                cy + radius * math.sin(2.0 * math.pi * k / N)] for k in range(N)]
        pts.append(list(pts[0]))   # close
        return pts

    if shape_type == "Rectangle":
        hw, hh = dims["width"] / 2.0, dims["height"] / 2.0
        return [[cx - hw, cy - hh], [cx + hw, cy - hh],
                [cx + hw, cy + hh], [cx - hw, cy + hh], [cx - hw, cy - hh]]

    logging.warning(f"[random_shapes] Unknown shape_type '{shape_type}'; skipping.")
    return None


def generate_random_shape_pointlists(params, H, W, gsd_m_per_px=None,
                                     image=None, nodata_mask=None,
                                     nodata_values=None):
    """
    Qt-free shape generator for use on background/worker threads.

    Returns a list of shapes, each a list of [x, y] pairs in IMAGE pixel coords.
    Only WIDTH/HEIGHT are required (fast path); pass `image` only when
    `params['restrict_valid']` is set and you need NoData exclusion.
    """
    if not H or not W or H <= 0 or W <= 0:
        return []

    shape_type = params["shape_type"]
    dims = _px_dims(params, gsd_m_per_px)
    centers = _sample_centers(params, dims, int(H), int(W), image, nodata_mask,
                              nodata_values)
    if not centers:
        return []

    out = []
    for (cx, cy) in centers:
        pts = _center_points(float(cx), float(cy), shape_type, dims)
        if pts:
            out.append(pts)
    logging.debug(f"[random_shapes] {len(out)}/{params.get('count')} {shape_type} "
                  f"(restrict={bool(params.get('restrict_valid'))}, {W}x{H})")
    return out


def generate_random_shapes(image, nodata_mask, params, gsd_m_per_px=None, image_shape=None,
                           nodata_values=None):
    """
    Generate random shapes as a list of QtGui.QPolygonF in IMAGE pixel coordinates
    (used by the live-viewer path). The caller converts to scene coordinates before
    adding to the viewer.

    Provide `image` for the restricted (valid-area) path, or just `image_shape`
    (H, W) for the fast shape-only path.
    """
    if image is not None:
        H, W = image.shape[:2]
    elif image_shape is not None:
        H, W = image_shape[:2]
    else:
        return []

    pointlists = generate_random_shape_pointlists(
        params, H, W, gsd_m_per_px, image=image, nodata_mask=nodata_mask,
        nodata_values=nodata_values)

    polygons = []
    for pts in pointlists:
        poly = QtGui.QPolygonF()
        for (x, y) in pts:
            poly.append(QtCore.QPointF(x, y))
        polygons.append(poly)
    return polygons
