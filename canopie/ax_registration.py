"""
Registration, ONE definition for every consumer (BUG-016, 2026-10-06).

The editor computes the matrix (pystackreg, an inverse map in pixel-INDEX
coordinates) between images that have already been through the .ax geometry --
crop, rotate AND resize -- and stores the size of that frame as `calc_shape`.

It used to be applied four different ways: the viewer and the editor warped the
RAW frame with only the crop translation and a scale (wrong as soon as a
rotation is involved), the analysis replay warped the processed frame, and the
image export (_apply_ax_to_raw) applied it twice. Every consumer now warps the
RAW frame (so pixels from outside the crop still come in -- no empty borders)
with the matrix conjugated exactly through the .ax geometry:

    M_raw = A^-1 . M_frame . A

where A maps raw coordinates to exported-frame coordinates (ax_geometry's ops:
rotate / crop / resize, all affine) and M_frame is the stored matrix rescaled
from `calc_shape` to the frame this .ax produces. The conjugation is done in
pixel-EDGE coordinates (where crop, rotate and resize are exact linear maps,
matching cv2.resize's pixel-centre convention) and converted back to index
coordinates for cv2.

For an array that is a scaled copy of the raw frame (the viewer's decimated
preview), pass its own size: the plan is built for that size, exactly as the
viewer's replay crops and resizes that array.
"""
import numpy as np

from . import ax_geometry as G

__all__ = ["frame_matrix", "raw_frame_matrix", "warp", "registration_active"]


def _T(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _S(sx, sy):
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def _op_matrix(op):
    """The ax_geometry op as a 3x3 on pixel-edge coordinates (see
    ax_geometry._rot_fwd for the rotation conventions)."""
    kind = op[0]
    if kind == "rotate":
        k, w, h = op[1], float(op[2]), float(op[3])
        if k == 90:                                   # (x, y) -> (h - y, x)
            return np.array([[0.0, -1.0, h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        if k == 180:                                  # (x, y) -> (w - x, h - y)
            return np.array([[-1.0, 0.0, w], [0.0, -1.0, h], [0.0, 0.0, 1.0]])
        if k == 270:                                  # (x, y) -> (y, w - x)
            return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w], [0.0, 0.0, 1.0]])
        return np.eye(3)
    if kind == "crop":
        return _T(-float(op[1]), -float(op[2]))
    if kind == "resize":
        return _S(float(op[1]), float(op[2]))
    return np.eye(3)


def registration_active(ax):
    reg = (ax or {}).get("registration") if isinstance(ax, dict) else None
    return bool(isinstance(reg, dict) and reg.get("enabled") and reg.get("matrix") is not None
                and not reg.get("is_reference", False))


def frame_matrix(reg):
    """(3x3 index-coordinate inverse map in the `calc_shape` frame, perspective)
    or (None, False) when registration is off, the image is the reference, or
    the matrix is unusable."""
    if not (isinstance(reg, dict) and reg.get("enabled") and reg.get("matrix") is not None):
        return None, False
    if reg.get("is_reference", False):
        return None, False
    try:
        t = np.array(reg["matrix"], dtype=np.float64)
    except Exception:
        return None, False
    if t.shape == (4, 4):
        t3 = np.eye(3)
        t3[0:2, 0:2] = t[0:2, 0:2]
        t3[0:2, 2] = t[0:2, 3]
        t = t3
    elif t.shape != (3, 3):
        return None, False
    persp = ("bilinear" in str(reg.get("mode", "")).lower()) or not np.allclose(t[2, :2], 0)
    return t, persp


def raw_frame_matrix(ax, arr_w, arr_h):
    """(M, perspective): the cv2 WARP_INVERSE_MAP matrix (index coordinates)
    for an arr_w x arr_h RAW-frame array, or (None, False)."""
    ax = ax if isinstance(ax, dict) else {}
    reg = ax.get("registration")
    t, persp = frame_matrix(reg)
    if t is None:
        return None, False
    ops, (W, H) = G.frame_plan(ax, int(arr_w), int(arr_h))
    if W <= 0 or H <= 0:
        return None, False
    m = _T(0.5, 0.5) @ t @ _T(-0.5, -0.5)            # index -> edge coordinates
    calc = reg.get("calc_shape")
    try:
        cw, ch = float(calc[0]), float(calc[1])
    except Exception:
        cw = ch = 0.0
    if cw > 0 and ch > 0 and (abs(cw - W) > 1e-9 or abs(ch - H) > 1e-9):
        sc = _S(W / cw, H / ch)                       # calc frame -> this frame
        m = sc @ m @ np.linalg.inv(sc)
    a = np.eye(3)
    for op in ops:
        a = _op_matrix(op) @ a
    m = np.linalg.inv(a) @ m @ a
    m = _T(-0.5, -0.5) @ m @ _T(0.5, 0.5)             # edge -> index coordinates
    return m, persp


def warp(img, m, perspective, border=np.nan):
    """Warp every band of an (H, W[, C]) array with the inverse map `m`, as
    float32; pixels that come from outside the raster get `border` (NaN:
    NoData). NaN in the input propagates (a pixel interpolated from NoData is
    NoData)."""
    import cv2
    a = np.asarray(img, dtype=np.float32)
    two_d = a.ndim == 2
    a3 = a[..., None] if two_d else a
    h, w = a3.shape[:2]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    out = np.empty(a3.shape, dtype=np.float32)
    for k in range(a3.shape[2]):
        band = np.ascontiguousarray(a3[..., k])
        if perspective:
            out[..., k] = cv2.warpPerspective(band, m, (w, h), flags=flags,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))
        else:
            out[..., k] = cv2.warpAffine(band, m[:2, :], (w, h), flags=flags,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))
    return out[..., 0] if two_d else out
