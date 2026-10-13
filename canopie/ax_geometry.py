"""
Exact geometry of an image's `.ax` crop / rotate / resize chain.

ONE authority for "where does original-raster pixel (x, y) end up in the
exported (post-.ax, full-resolution) frame, and how big is that frame". It
mirrors ProjectTab._apply_ax_to_raw step for step -- the same crop/rotate
order decision, the same integer crop offsets (round), np.rot90's index
mapping, the same three resize modes -- so the frame it predicts is the frame
CSV export, ML export and ML training actually sample.

Coordinates are CONTINUOUS pixel-edge coordinates: pixel i covers [i, i + 1),
its centre is i + 0.5. Every step is an exact affine map in that convention
(cv2.resize maps pixel edges linearly; np.rot90 is a quarter turn of the grid).

Not modelled: registration (a band-alignment warp applied before crop; it does
not change the frame's size), histogram matching and band math (pixel values
only).

Why this exists: points were drawn with a marker sized from the RAW FILE
dimensions over the displayed pixmap -- 28 x 32 export pixels per "pixel" on
a cropped + resized BCI mosaic -- and could not be snapped to the pixel the
CSV reads, because nothing knew the export frame's exact size.
"""
import math

import numpy as np

__all__ = ["frame_plan", "frame_hw", "forward", "inverse", "forward_np", "inverse_np",
           "export_pixel_center"]


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def frame_plan(ax, raw_w, raw_h):
    """Return (ops, (W, H)): the geometric steps `_apply_ax_to_raw` performs on
    a raw_w x raw_h raster, and the size of the resulting frame.

    ops: list of ("rotate", k, w_before, h_before) | ("crop", xi, yi, xe, ye) |
    ("resize", sx, sy, new_w, new_h), in application order (only the first
    elements are used to map points; the rest let a replay cut and size the
    array exactly the same way)."""
    ax = ax if isinstance(ax, dict) else {}
    W, H = int(raw_w), int(raw_h)
    ops = []

    rot = _int(ax.get("rotate", 0)) % 360
    rotate_enabled = ax.get("rotate_enabled", True)
    crop_rect = ax.get("crop_rect") or None
    crop_ref = ax.get("crop_rect_ref_size") or None
    crop_enabled = ax.get("crop_enabled", True)
    resize = ax.get("resize") or None
    resize_enabled = ax.get("resize_enabled", True)

    # -- crop/rotate order: identical decision to _apply_ax_to_raw ----------
    rotated_w, rotated_h = (H, W) if rot in (90, 270) else (W, H)
    do_rotate_first = True
    if crop_rect and rot in (90, 180, 270):
        if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
            rw_, rh_ = _int(crop_ref.get("w", 0)), _int(crop_ref.get("h", 0))
            if rw_ > 0 and rh_ > 0:
                if (rw_, rh_) == (W, H):
                    do_rotate_first = False
                elif (rw_, rh_) == (rotated_w, rotated_h):
                    do_rotate_first = True

    def _rotate():
        nonlocal W, H
        if not rotate_enabled or not rot or rot not in (90, 180, 270):
            return
        ops.append(("rotate", rot, W, H))
        if rot in (90, 270):
            W, H = H, W

    def _crop():
        nonlocal W, H
        if not crop_enabled or not (isinstance(crop_rect, dict) and crop_rect):
            return
        if W <= 0 or H <= 0:
            return
        if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
            refW = max(1, _int(crop_ref.get("w") or W, W))
            refH = max(1, _int(crop_ref.get("h") or H, H))
        else:
            refW, refH = W, H
        x = _int(crop_rect.get("x", 0)); y = _int(crop_rect.get("y", 0))
        rw = _int(crop_rect.get("width", W), W); rh = _int(crop_rect.get("height", H), H)
        sx = W / float(max(1, refW)); sy = H / float(max(1, refH))
        xi = max(0, int(round(x * sx))); yi = max(0, int(round(y * sy)))
        xe = min(W, xi + max(1, int(round(rw * sx))))
        ye = min(H, yi + max(1, int(round(rh * sy))))
        ops.append(("crop", xi, yi, xe, ye))
        W, H = max(0, xe - xi), max(0, ye - yi)

    if do_rotate_first:
        _rotate(); _crop()
    else:
        _crop(); _rotate()

    # -- resize: the three modes of _apply_ax_to_raw._do_resize -------------
    if resize_enabled and isinstance(resize, dict) and resize and W > 0 and H > 0:
        w0, h0 = W, H
        new_w = new_h = None
        if "px_w" in resize or "px_h" in resize:
            tw = _int(resize.get("px_w", 0)); th = _int(resize.get("px_h", 0))
            if tw > 0 and th > 0:
                new_w, new_h = tw, th
            elif tw > 0:
                s = tw / float(w0); new_w, new_h = tw, max(1, int(round(h0 * s)))
            elif th > 0:
                s = th / float(h0); new_h, new_w = th, max(1, int(round(w0 * s)))
        elif "scale" in resize:
            s = float(resize.get("scale", 100.0)) / 100.0
            new_w, new_h = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
        else:
            pw = float(resize.get("width", 100.0)) / 100.0
            ph = float(resize.get("height", 100.0)) / 100.0
            new_w, new_h = max(1, int(round(w0 * pw))), max(1, int(round(h0 * ph)))
        if new_w and new_h:
            ops.append(("resize", new_w / float(w0), new_h / float(h0), new_w, new_h))
            W, H = new_w, new_h
    return ops, (W, H)


def frame_hw(ax, raw_w, raw_h):
    """(H, W) of the exported frame."""
    _, (W, H) = frame_plan(ax, raw_w, raw_h)
    return H, W


def preview_px_resize(ax, raw_w, raw_h, w0, h0):
    """A pixel resize (`px_w` / `px_h`: pixels of the FULL frame) applied to a
    DECIMATED preview whose pre-resize frame is w0 x h0: the size the export's
    resize FACTOR gives it, or None when there is no pixel resize or w0 x h0 is
    the full frame itself (then the pixel size applies as is). Without this the
    preview was blown back up to the full export size (BUG-060)."""
    r = (ax or {}).get("resize") if isinstance(ax, dict) else None
    if not (isinstance(r, dict) and ("px_w" in r or "px_h" in r)):
        return None
    ops, _ = frame_plan(ax, raw_w, raw_h)
    op = next((o for o in ops if o[0] == "resize"), None)
    if op is None:
        return None
    pre_w, pre_h = op[3] / float(op[1]), op[4] / float(op[2])
    if w0 >= pre_w - 1 and h0 >= pre_h - 1:
        return None
    return max(1, int(round(w0 * float(op[1])))), max(1, int(round(h0 * float(op[2]))))


def _rot_fwd(k, w, h, x, y):
    # np.rot90(img, k=3) for 90 (clockwise); k=2 for 180; k=1 for 270.
    if k == 90:
        return h - y, x
    if k == 180:
        return w - x, h - y
    if k == 270:
        return y, w - x
    return x, y


def _rot_inv(k, w, h, x, y):
    # (w, h) are the dims BEFORE the rotation
    if k == 90:
        return y, h - x
    if k == 180:
        return w - x, h - y
    if k == 270:
        return w - y, x
    return x, y


def forward(ops, pts):
    """Original-raster continuous coords -> exported-frame coords."""
    out = []
    for (x, y) in pts:
        x, y = float(x), float(y)
        for op in ops:
            if op[0] == "rotate":
                x, y = _rot_fwd(op[1], op[2], op[3], x, y)
            elif op[0] == "crop":
                x, y = x - op[1], y - op[2]
            elif op[0] == "resize":
                x, y = x * op[1], y * op[2]
        out.append((x, y))
    return out


def inverse(ops, pts):
    """Exported-frame continuous coords -> original-raster coords."""
    out = []
    for (x, y) in pts:
        x, y = float(x), float(y)
        for op in reversed(ops):
            if op[0] == "rotate":
                x, y = _rot_inv(op[1], op[2], op[3], x, y)
            elif op[0] == "crop":
                x, y = x + op[1], y + op[2]
            elif op[0] == "resize":
                x, y = x / op[1], y / op[2]
        out.append((x, y))
    return out


def _apply_np(ops, pts, fwd):
    a = np.array(pts, dtype=np.float64).reshape(-1, 2)
    x, y = a[:, 0], a[:, 1]
    for op in (ops if fwd else reversed(ops)):
        if op[0] == "rotate":
            k, w, h = op[1], op[2], op[3]
            if fwd:
                if k == 90:
                    x, y = h - y, x
                elif k == 180:
                    x, y = w - x, h - y
                elif k == 270:
                    x, y = y, w - x
            else:
                if k == 90:
                    x, y = y, h - x
                elif k == 180:
                    x, y = w - x, h - y
                elif k == 270:
                    x, y = w - y, x
        elif op[0] == "crop":
            x, y = (x - op[1], y - op[2]) if fwd else (x + op[1], y + op[2])
        elif op[0] == "resize":
            x, y = (x * op[1], y * op[2]) if fwd else (x / op[1], y / op[2])
    return np.column_stack((x, y))


def forward_np(ops, pts):
    """`forward` on an (N, 2) array -- the same maps, for whole crown outlines."""
    return _apply_np(ops, pts, True)


def inverse_np(ops, pts):
    """`inverse` on an (N, 2) array."""
    return _apply_np(ops, pts, False)


def export_pixel_center(x, y):
    """Centre of the exported-frame pixel containing continuous (x, y). The
    CSV (floor) and ML (int) readers land on this pixel with half a pixel of
    margin either way, so float round-trips can never tip it into a neighbour."""
    return math.floor(x) + 0.5, math.floor(y) + 0.5
