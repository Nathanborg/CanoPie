import numpy as np
import cv2
import math
import logging
from PyQt5 import QtCore, QtGui


def _build_valid_mask(image, nodata_mask):
    """
    Build a uint8 valid-pixel mask (255=valid, 0=invalid).

    Priority:
    1. If `nodata_mask` is provided (from .ax nodata_values), use it directly.
    2. Float images: valid = all channels are finite (NaN/Inf exclusion only).
    3. Integer images: valid = at least one channel > 0 (excludes black borders).

    The result is then optionally eroded by dist_border_px in the caller.
    """
    H, W = image.shape[:2]

    if nodata_mask is not None and nodata_mask.shape[:2] == (H, W):
        return (~np.asarray(nodata_mask, dtype=bool)).astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.floating):
        if image.ndim == 3:
            finite = np.all(np.isfinite(image), axis=2)
        else:
            finite = np.isfinite(image)
        return finite.astype(np.uint8) * 255

    # Integer / uint8 / uint16: exclude pure-black pixels (stitch borders).
    if image.ndim == 3:
        return np.any(image > 0, axis=2).astype(np.uint8) * 255
    return (image > 0).astype(np.uint8) * 255


def generate_random_shapes(image, nodata_mask, params, gsd_m_per_px=None):
    """
    Generate random shapes strictly within the valid (non-NaN / non-NoData) area.

    Returns a list of QtGui.QPolygonF in IMAGE pixel coordinates.
    Each polygon represents one shape; the caller is responsible for converting
    to scene coordinates before adding to the viewer.

    Parameters
    ----------
    image        : np.ndarray  – The displayed/modified image (H, W) or (H, W, C)
    nodata_mask  : np.ndarray or None – True where pixels are invalid (NoData/NaN)
    params       : dict from RandomShapesDialog.get_parameters()
    gsd_m_per_px : float or None – meters per pixel (GSD).
                   When provided, dimension/distance params are in meters and are
                   converted to pixels using this value.
    """
    if image is None:
        return []

    H, W = image.shape[:2]

    # ---- Unit conversion -------------------------------------------------- #
    shape_type   = params["shape_type"]
    count        = int(params["count"])

    # scale: pixels per user-unit
    if gsd_m_per_px and gsd_m_per_px > 0:
        scale = 1.0 / gsd_m_per_px   # user units = meters → pixels
    else:
        scale = 1.0                   # user units = pixels

    diameter_px      = float(params.get("diameter", 1.0)) * scale
    width_px         = float(params.get("width",    1.0)) * scale
    height_px        = float(params.get("height",   1.0)) * scale
    dist_border_px   = float(params.get("min_dist_border", 0.0)) * scale
    dist_shapes_px   = float(params.get("min_dist_shapes", 0.0)) * scale

    # ---- Valid mask ------------------------------------------------------- #
    valid_mask = _build_valid_mask(image, nodata_mask)

    # Erode by border distance so shapes can't land near the valid-area edge
    border_px = int(math.ceil(dist_border_px))
    if border_px > 0:
        ksize = border_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

    # Get all valid pixel coordinates as (y, x) pairs
    valid_yx = np.column_stack(np.where(valid_mask > 0))
    if len(valid_yx) == 0:
        logging.warning("[random_shapes] No valid pixels after mask/erosion; nothing generated.")
        return []

    # ---- Dart-throwing (rejection sampling) ------------------------------- #
    np.random.shuffle(valid_yx)
    min_dist_sq = dist_shapes_px ** 2

    accepted = []   # list of (x, y)
    accepted_arr = np.empty((0, 2), dtype=np.float64)

    for yx in valid_yx:
        if len(accepted) >= count:
            break
        py, px = float(yx[0]), float(yx[1])

        if min_dist_sq > 0 and len(accepted) > 0:
            dx = accepted_arr[:, 0] - px
            dy = accepted_arr[:, 1] - py
            if np.any(dx * dx + dy * dy < min_dist_sq):
                continue

        accepted.append((px, py))
        accepted_arr = np.vstack([accepted_arr, [[px, py]]])

    if not accepted:
        logging.warning("[random_shapes] Dart-throwing produced 0 accepted points.")
        return []

    # ---- Build QPolygonF objects ------------------------------------------ #
    polygons = []
    for (cx, cy) in accepted:
        poly = QtGui.QPolygonF()

        if shape_type == "Point":
            poly.append(QtCore.QPointF(cx, cy))

        elif shape_type == "Circle":
            radius = diameter_px / 2.0
            N = 32
            for k in range(N):
                angle = 2.0 * math.pi * k / N
                poly.append(QtCore.QPointF(cx + radius * math.cos(angle),
                                            cy + radius * math.sin(angle)))
            poly.append(poly.first())   # close

        elif shape_type == "Rectangle":
            hw, hh = width_px / 2.0, height_px / 2.0
            poly.append(QtCore.QPointF(cx - hw, cy - hh))
            poly.append(QtCore.QPointF(cx + hw, cy - hh))
            poly.append(QtCore.QPointF(cx + hw, cy + hh))
            poly.append(QtCore.QPointF(cx - hw, cy + hh))
            poly.append(poly.first())   # close

        else:
            logging.warning(f"[random_shapes] Unknown shape_type '{shape_type}'; skipping.")
            continue

        polygons.append(poly)

    logging.info(f"[random_shapes] Generated {len(polygons)}/{count} shapes "
                 f"(type={shape_type}, border={dist_border_px:.1f}px, "
                 f"spacing={dist_shapes_px:.1f}px, valid_px={len(valid_yx)})")
    return polygons
