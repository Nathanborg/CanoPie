import os
import sys
import json
import tempfile
import subprocess
import logging
import numpy as np
import cv2
import re

# Windows subprocess flag to hide console window
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

STRETCH_LOW_P = 0.5
STRETCH_HIGH_P = 99.5
STRETCH_PER_CHANNEL = True
STRETCH_CLIP = True
STRETCH_SAMPLE_MAX = 250

def _dims_after_rot(w0: int, h0: int, rot: int):
    """
    Return (width, height) after rotating a rectangle of size (w0, h0) by rot degrees.
    Only multiples of 90 are supported.
    """
    r = (int(rot) // 90) % 4
    return (w0, h0) if r in (0, 2) else (h0, w0)

def _rect_after_rot(rect, ref_w, ref_h, rot):
    """
    Rotate an axis‑aligned rectangle within an image of size (ref_w, ref_h).
    Returns a new dictionary with keys x,y,width,height describing the rotated
    bounding box.  Only multiples of 90 degrees are supported.
    """
    x = int(rect.get("x", 0)); y = int(rect.get("y", 0))
    w = int(rect.get("width", 0)); h = int(rect.get("height", 0))
    if w <= 0 or h <= 0:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    r = (int(rot) // 90) % 4
    def rot90(p):  return (ref_h - 1 - p[1], p[0])
    def rot180(p): return (ref_w - 1 - p[0], ref_h - 1 - p[1])
    def rot270(p): return (p[1], ref_w - 1 - p[0])
    if r == 1:  pts2, new_w, new_h = [rot90(p)  for p in pts], ref_h, ref_w
    elif r == 2:pts2, new_w, new_h = [rot180(p) for p in pts], ref_w, ref_h
    elif r == 3:pts2, new_w, new_h = [rot270(p) for p in pts], ref_h, ref_w
    else:       pts2, new_w, new_h = pts, ref_w, ref_h
    xs, ys = [p[0] for p in pts2], [p[1] for p in pts2]
    xx0, xx1 = max(0, min(xs)), min(new_w,  max(xs))
    yy0, yy1 = max(0, min(ys)), min(new_h,  max(ys))
    return {"x": int(xx0), "y": int(yy0), "width": int(max(0, xx1-xx0)), "height": int(max(0, yy1-yy0))}

def _scale_rect(rect, from_w, from_h, to_w, to_h):
    """
    Scale a rectangle from one coordinate system (from_w, from_h) to another (to_w, to_h).
    Returns a new rectangle dictionary.  Clamps values to bounds.
    """
    if from_w <= 0 or from_h <= 0:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    sx, sy = to_w/float(from_w), to_h/float(from_h)
    x = int(round(rect.get("x", 0)      * sx))
    y = int(round(rect.get("y", 0)      * sy))
    w = int(round(rect.get("width", 0)  * sx))
    h = int(round(rect.get("height", 0) * sy))
    # clamp to bounds
    x = max(0, min(x, to_w))
    y = max(0, min(y, to_h))
    w = max(0, min(w, to_w - x))
    h = max(0, min(h, to_h - y))
    return {"x": x, "y": y, "width": w, "height": h}

def _infer_crop_basis(ax, raw_w, raw_h, rot):
    """
    Determine whether a crop rectangle is specified relative to the raw image (pre‑rotate)
    or the rotated image (after_rotate).  Returns either 'pre_rotate' or 'after_rotate'.
    """
    basis = str(ax.get("crop_rect_basis", "")).strip().lower()
    if basis in ("pre_rotate", "after_rotate"):
        return basis
    ref = ax.get("crop_rect_ref_size")
    wr, hr = _dims_after_rot(raw_w, raw_h, rot)
    if isinstance(ref, dict) and "w" in ref and "h" in ref:
        rw, rh = int(ref.get("w", raw_w)), int(ref.get("h", raw_h))
        if (rw, rh) == (wr, hr):
            return "after_rotate"
        if (rw, rh) == (raw_w, raw_h):
            return "pre_rotate"
    return "after_rotate"

def _rotate_point_in_rect(x, y, w, h, rot):
    """
    Rotate a point (x,y) within a rectangle of size (w,h) by rot degrees (multiples of 90).
    Returns the new (x',y') coordinates within the rotated rectangle.
    """
    r = (int(rot) // 90) % 4
    if r == 1:  return (h - 1 - y, x)        # 90 CW
    if r == 2:  return (w - 1 - x, h - 1 - y)  # 180
    if r == 3:  return (y, w - 1 - x)        # 270 CW
    return (x, y)

def resize_safe(img, new_w, new_h, interp=cv2.INTER_LINEAR):
    """
    Robust resize for 2D and HxWxC images with ANY number of channels.
    Falls back to per-channel resize when cn > 4 or when OpenCV's fast path fails.
    Preserves dtype.
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    if h == 0 or w == 0 or (new_w == w and new_h == h):
        return img

    # 2D (single band)
    if img.ndim == 2:
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 3D (multi-band)
    c = img.shape[2]
    try:
        # Fast path for <=4 channels
        if c <= 4:
            return cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Per-channel path for cn > 4
        out = np.empty((new_h, new_w, c), dtype=img.dtype)
        for i in range(c):
            out[..., i] = cv2.resize(img[..., i], (new_w, new_h), interpolation=interp)
        return out

    except Exception as e:
        # Absolute fallback (e.g., if OpenCV still complains for exotic dtypes)
        logging.warning(f"resize_safe: per-channel fallback due to error: {e}")
        out = np.empty((new_h, new_w, c), dtype=img.dtype)
        for i in range(c):
            out[..., i] = cv2.resize(img[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        return out

def _nanpct(a, p, axis=None):
    """
    NaN-aware percentile that never propagates NaN.

    np.nanpercentile emits a RuntimeWarning and returns NaN for an all-NaN slice,
    which would then blank the display. Fall back to 0.0 for those entries.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    with np.errstate(all="ignore"):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = np.nanpercentile(a, p, axis=axis)
        except Exception:
            out = np.percentile(np.nan_to_num(a), p, axis=axis)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_for_display(
    img,
    low_p=STRETCH_LOW_P,
    high_p=STRETCH_HIGH_P,
    per_channel=STRETCH_PER_CHANNEL,
    clip=STRETCH_CLIP,
    sample_max=STRETCH_SAMPLE_MAX,
    *,
    input_is_rgb=None,   # None=assume raw cv2 BGR, True=already RGB (post-aux/tifffile), False=BGR
    return_bgr=True      # True -> return BGR for Qt fast path; False -> return RGB
):
    """
    Normalize an image for display by stretching pixel values between low_p and high_p percentiles.
    Supports grayscale and RGB images and works on any numeric dtype.

    Parameters
    ----------
    input_is_rgb : {None, bool}
        For ≤3-channel inputs: if True, treat as RGB (e.g., post-apply_aux_modifications).
        If False, treat as BGR (e.g., raw cv2.imread). If None, defaults to BGR.
    return_bgr : bool
        If True, return BGR (ideal for QImage::Format_BGR888). If False, return RGB.

    Returns
    -------
    disp : uint8 (H,W) or (H,W,3)
        8-bit image suitable for display (max 3 channels), with channel order per return_bgr.
    """
    import numpy as np
    import cv2

    if img is None:
        return None

    # --- Fast path: already uint8 image (we'll only correct channel order later) ---
    if isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim in (2, 3):
        disp = img.copy()
    else:
        # Convert to float32 for stretching.
        # NaN/Inf are PRESERVED here (this used to nan_to_num() first, turning every
        # NaN into a real 0.0 that then skewed the percentiles below). The percentile
        # calls are nan-aware, and the final np.clip + uint8 cast maps any remaining
        # NaN to 0, so non-finite pixels still render without polluting the statistics.
        x = np.asarray(img).astype(np.float32, copy=False)

        def _sample(a):
            h, w = a.shape[:2]
            m = max(h, w)
            if m <= sample_max:
                return a
            s = sample_max / float(m)
            return cv2.resize(
                a,
                (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                interpolation=cv2.INTER_AREA,
            )

        if x.ndim == 2:
            s = _sample(x)
            lo = _nanpct(s, low_p)
            hi = _nanpct(s, high_p)
            n = (x - lo) / max(hi - lo, 1e-12) if hi > lo else np.full_like(x, 0.5, dtype=np.float32)
            if clip:
                n = np.clip(n, 0.0, 1.0)
            n = np.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0)
            disp = (n * 255.0).astype(np.uint8)

        elif x.ndim == 3:
            C = x.shape[2]
            use = x[:, :, :max(1, min(C, 3))]  # take up to 3 channels for preview
            s = _sample(use)
            # Always stretch multi-channel imagery PER CHANNEL. Pooling all bands into a
            # single distribution (the old `per_channel`-gated else branch) blows out the
            # brightest band and crushes the others on multi-band images; grayscale (1 band)
            # is unaffected. `per_channel` now governs only Absolute per-band ranges elsewhere.
            if use.ndim == 3 and use.shape[2] > 1:
                flat = s.reshape(-1, use.shape[2])
                lo = _nanpct(flat, low_p, axis=0)
                hi = _nanpct(flat, high_p, axis=0)
                scale = np.maximum(hi - lo, 1e-12)
                n = (use - lo.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
            else:
                lo = _nanpct(s, low_p)
                hi = _nanpct(s, high_p)
                n = (use - lo) / max(hi - lo, 1e-12) if hi > lo else np.full_like(use, 0.5, dtype=np.float32)
            if clip:
                n = np.clip(n, 0.0, 1.0)
            n = np.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0)
            disp = (n * 255.0).astype(np.uint8)

            # If only 1 or 2 channels, pad to 3 for display
            if disp.ndim == 3:
                if disp.shape[2] == 1:
                    disp = np.repeat(disp, 3, axis=2)
                elif disp.shape[2] == 2:
                    disp = np.concatenate([disp, disp[:, :, :1]], axis=2)
        else:
            return None

    # --- Keep to max 3 channels ---
    if disp.ndim == 3 and disp.shape[2] > 3:
        disp = disp[:, :, :3].copy()

    # --- Channel-order fix (only for 3-channel images) ---
    if disp.ndim == 3 and disp.shape[2] == 3:
        # Default assumption: raw cv2 → BGR input
        rgb_in = False if input_is_rgb is None else bool(input_is_rgb)

        # We want to return BGR for Qt fast path by default
        if return_bgr:
            # If input is RGB, flip to BGR; if input already BGR, keep as-is
            if rgb_in:
                disp = disp[:, :, ::-1].copy()
        else:
            # Caller wants RGB back; if input is BGR, flip; if RGB, keep as-is
            if not rgb_in:
                disp = disp[:, :, ::-1].copy()

    return disp

# utils.py
def _sample_for_stats(arr, sample_max=STRETCH_SAMPLE_MAX):
    """
    Downsample so the largest dim ≤ sample_max. Works for 2D or HxWxC (any C).
    """
    import numpy as np
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= sample_max:
        return arr

    scale = float(sample_max) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use the stack-safe resizer you already ship
    return resize_safe(arr, new_w, new_h, interp=cv2.INTER_AREA)

def process_band_expression(image, expr):
    """
    Evaluate a band expression like 'b1+b2/2' on an image and return a normalised uint8 result.
    This function supports only references to b1..bN and no arbitrary names.
    """
    import re
    if image is None or not expr:
        return image
    bands = re.findall(r'b(\d+)', expr)
    unique_bands = sorted(set(bands), key=lambda x: int(x))
    band_mapping = {}
    if image.ndim == 2:
        band_mapping['b1'] = image.astype(np.float32)
    elif image.ndim == 3:
        for b in unique_bands:
            band_index = int(b) - 1
            band_mapping[f'b{b}'] = image[:, :, band_index].astype(np.float32)
    allowed_names = band_mapping
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of '{name}' is not allowed.")
    result = eval(code, {"__builtins__": {}}, allowed_names)
    if isinstance(result, np.ndarray):
        if result.ndim == 2:
            if result.min() == result.max():
                return np.full(result.shape, 128, dtype=np.uint8)
            else:
                return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif result.ndim == 3:
            if result.min() == result.max():
                return np.full(result.shape, 128, dtype=np.uint8)
            else:
                return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            return image
    else:
        scalar_value = np.clip(result, 0, 255)
        return np.full(image.shape[:2], scalar_value, dtype=np.uint8)



_COMPARISON_RE = re.compile(
    r'(\b(?:b\d+|\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\b)\s*'
    r'(==|!=|<=|>=|<|>)\s*'
    r'(\b(?:b\d+|\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\b)'
)



def process_band_expression_float(image, expr):
    """
    Back-compat wrapper: evaluate band expression on FLOAT32 and return float32.
    Supports functions (mean, sum, std, ...), global reducers, safe '/', logicals, etc.
    """
    import numpy as np
    if image is None or not expr:
        return image
    return eval_band_expression(np.asarray(image, dtype=np.float32), expr)

def _eval_band_expression_float(self, img, expr):
    """
    Back-compat wrapper used by ProjectTab: same as process_band_expression_float.
    """
    import numpy as np
    if img is None or not expr:
        return None
    return eval_band_expression(np.asarray(img, dtype=np.float32), expr)

def get_exif_data_exiftool_multiple(filepaths):
    """
    Extract EXIF metadata from multiple image files using the command‑line exiftool.
    Returns a dictionary mapping absolute filepaths to their EXIF data dictionaries.
    If exiftool is not available on the system this function returns an empty dict.
    """
    if not filepaths:
        return {}
    # Default to using the 'exiftool' command on the system PATH
    exiftool_cmd = os.environ.get("EXIFTOOL_PATH", "exiftool")
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', newline='') as tmp_file:
            for filepath in filepaths:
                tmp_file.write(f"{filepath}\n")
            tmp_file_path = tmp_file.name
        command = [exiftool_cmd, '-j', '-@', tmp_file_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True,
                                creationflags=_SUBPROCESS_FLAGS)
        exif_json = json.loads(result.stdout)
        exif_dict = {os.path.abspath(item['SourceFile']): item for item in exif_json}
    except Exception as e:
        # Swallow errors rather than raising; EXIF extraction is optional
        logging.warning(f"Error extracting EXIF data with exiftool: {e}")
        exif_dict = {}
    finally:
        try:
            os.remove(tmp_file_path)
        except Exception as cleanup_error:
            logging.warning(f"Error removing temporary file {tmp_file_path}: {cleanup_error}")
    return exif_dict

def calculate_exg(red, green, blue):
    """
    Excess Green Index (ExG): emphasises green vegetation.
    """
    return (2 * green) - (red + blue)

def calculate_gcc(red, green, blue):
    """
    Green Chromatic Coordinate (GCC): green divided by total RGB, with divide‑by‑zero protection.
    """
    denominator = red + green + blue
    # Avoid division by zero
    denominator = denominator.copy()
    denominator[denominator == 0] = 1
    return green / denominator

def calculate_bcc(red, green, blue):
    """
    Blue Chromatic Coordinate (BCC): blue divided by total RGB, with divide‑by‑zero protection.
    """
    denominator = red + green + blue
    denominator = denominator.copy()
    denominator[denominator == 0] = 1
    return blue / denominator

def calculate_gbd(green, blue):
    """
    Green‑Blue Difference (GBD): emphasises the difference between green and blue bands.
    """
    denominator = green + blue
    denominator = denominator.copy()
    denominator[denominator == 0] = 1
    return (green - blue) / denominator

def calculate_wdx(red, green, blue):
    """
    Weighted Difference Index (WDX): emphasises shadows or drought stress.
    """
    return (2 * blue) + red - (2 * green)

def calculate_shd(red, green, blue):
    """
    Simple Sum Index (SHD): sums red, green and blue bands; used as a simple brightness measure.
    """
    return red + green + blue
# utils.py
import re, ast
import numpy as np

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant,
    # arithmetic
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    # unary
    ast.Invert, ast.UAdd, ast.USub,
    # boolean / bitwise
    ast.And, ast.Or, ast.BitAnd, ast.BitOr,
    # comparisons
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

def _ensure_bool(arr):
    a = np.asarray(arr)
    if a.dtype == bool:
        return a
    raise TypeError("Logical operators (&, |, not/~) must combine comparisons (boolean arrays). "
                    "Add parentheses or make each side a comparison, e.g. (b1<133) & (b2>323).")
 

import re, ast
import numpy as np

def normalize_band_expr(expr: str) -> str:
    s = expr or ""
    s = re.sub(r'\bAND\b', '&', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOR\b',  '|', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNOT\b', '~', s, flags=re.IGNORECASE)
    return s

def eval_band_expression(image: np.ndarray, expr: str) -> np.ndarray:
    """
    Evaluate a band expression on an image and return float32 HxW.
    Supports + - * /, comparisons, &,|,~, and these functions:
      sum, mean/avg, min, max, std, median, clip(x,lo,hi), where(cond,a,b), abs, sqrt, log, exp.
    Single-arg reducers (e.g., mean(b1)) are GLOBAL over the band (scalar),
    multi-arg reducers (e.g., mean(b1,b2)) are PIXELWISE across args.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Empty image.")
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Empty expression.")

    import numpy as _np
    x = _np.nan_to_num(_np.asarray(image, dtype=_np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    H, W = x.shape[:2]
    C = 1 if x.ndim == 2 else x.shape[2]

    # Map bands b1..bN
    mapping = {'b1': x} if C == 1 else {f"b{i+1}": x[:, :, i] for i in range(C)}
    mapping.update({k.upper(): v for k, v in mapping.items()})

    # Helpers
    def _to_arr(v): return _np.asarray(v, dtype=_np.float32)
    def _stack(args): return _np.stack([_to_arr(a) for a in args], axis=0)
    def _global_reduce(fn, a): return float(fn(_to_arr(a)))

    # Elementwise + reducers
    def _psum(*args):   return _np.add.reduce([_to_arr(a) for a in args])
    def _pmean(*args):  return _psum(*args) / max(len(args), 1)
    def _pmin(*args):   return _np.minimum.reduce([_to_arr(a) for a in args])
    def _pmax(*args):   return _np.maximum.reduce([_to_arr(a) for a in args])
    def _pstd(*args):   return _np.std(_stack(args), axis=0)
    def _pmedian(*args):return _np.median(_stack(args), axis=0)

    def _SUM(*args):    return _global_reduce(_np.sum,    args[0]) if len(args)==1 else _psum(*args)
    def _MEAN(*args):   return _global_reduce(_np.mean,   args[0]) if len(args)==1 else _pmean(*args)
    def _MIN(*args):    return _global_reduce(_np.min,    args[0]) if len(args)==1 else _pmin(*args)
    def _MAX(*args):    return _global_reduce(_np.max,    args[0]) if len(args)==1 else _pmax(*args)
    def _STD(*args):    return _global_reduce(_np.std,    args[0]) if len(args)==1 else _pstd(*args)
    def _MEDIAN(*args): return _global_reduce(_np.median, args[0]) if len(args)==1 else _pmedian(*args)

    allowed_funcs = {
        # global (1 arg) or pixelwise (2+)
        "sum": _SUM, "mean": _MEAN, "avg": _MEAN,
        "min": _MIN, "max": _MAX, "std": _STD, "median": _MEDIAN,
        # explicit pixelwise variants if you ever need them
        "psum": _psum, "pmean": _pmean, "pmin": _pmin, "pmax": _pmax, "pstd": _pstd, "pmedian": _pmedian,
        # elementwise utilities
        "clip":   lambda x, lo, hi: _np.clip(_to_arr(x), float(lo), float(hi)),
        "where":  lambda c, a, b: _np.where(_np.asarray(c, dtype=bool), _to_arr(a), _to_arr(b)),
        "abs":    lambda x: _np.abs(_to_arr(x)),
        "sqrt":   lambda x: _np.sqrt(_np.maximum(_to_arr(x), 0.0)),
        "log":    lambda x: _np.log(_np.maximum(_to_arr(x), 1e-12)),
        "exp":    lambda x: _np.exp(_to_arr(x)),
    }
    allowed_funcs.update({k.upper(): v for k, v in allowed_funcs.items()})

    # Safe divide: returns 0.0 when denominator is zero or non-finite
    def safe_div(a, b):
        a = _to_arr(a); b = _to_arr(b)
        out = _np.zeros(_np.broadcast(a, b).shape, dtype=_np.float32)
        valid = (b != 0) & _np.isfinite(a) & _np.isfinite(b)
        _np.divide(a, b, out=out, where=valid)
        return out

    # AST transform: whitelist names/calls; / -> safe_div; &,|,~ -> np.logical_*
    expr_norm = normalize_band_expr(expr)

    if not re.fullmatch(r"[0-9eE\.\s\+\-\*/\(\)<>!=&|~A-Za-z_,]+", expr_norm):
        raise ValueError("Disallowed characters in band expression.")

    class _X(ast.NodeTransformer):
        def __init__(self, band_keys, func_keys):
            self._bands = set(band_keys)
            self._funcs = set(func_keys)

        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Div):
                return ast.Call(func=ast.Name(id="safe_div", ctx=ast.Load()),
                                args=[node.left, node.right], keywords=[])
            if isinstance(node.op, ast.BitAnd):
                return ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                                   attr="logical_and", ctx=ast.Load()),
                                args=[node.left, node.right], keywords=[])
            if isinstance(node.op, ast.BitOr):
                return ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                                   attr="logical_or", ctx=ast.Load()),
                                args=[node.left, node.right], keywords=[])
            return node

        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Invert):
                return ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                                   attr="logical_not", ctx=ast.Load()),
                                args=[node.operand], keywords=[])
            return node

        def visit_Call(self, node):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._funcs:
                raise SyntaxError("Allowed functions: " + ", ".join(sorted(self._funcs)))
            if node.keywords:
                raise SyntaxError("Keyword arguments not allowed in band functions.")
            node.args = [self.visit(a) for a in node.args]
            return node

        def visit_Name(self, node):
            if node.id in self._bands or node.id in self._funcs or node.id in ("np","safe_div"):
                return node
            raise NameError(f"Use only b1..b{len(self._bands)} or allowed functions.")

    tree = ast.parse(expr_norm, mode="eval")
    tree = _X(mapping.keys() | {k.upper() for k in mapping.keys()}, allowed_funcs.keys()).visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<band-expr>", "eval")

    res = eval(code, {"__builtins__": {}, "np": np, "safe_div": safe_div, **allowed_funcs}, mapping)

    # ndarray → float32; bool → 0/1; scalar → broadcast
    if isinstance(res, np.ndarray):
        out = res.astype(np.float32, copy=False) if res.dtype != np.bool_ else res.astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.full((H, W), float(res), dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# Classification feature-stack primitives
#
# Shared by ImageEditorDialog._make_feature_stack_for_model (the live
# "Classify" button) and ProjectTab._make_feature_stack_for_model (render/
# preview + the .ax "appended classification band" replay in
# _apply_ax_to_raw/apply_aux_modifications). Both callers keep their own
# wrapper body -- base-feature-name resolution (case-sensitivity differs
# between them), error convention (one returns (None,(0,0)), the other
# raises), and project_tab.py's copy's extra NaN/Inf sanitization pass that
# the editor's copy currently lacks -- deliberately NOT unified here, since
# those are real behavioral differences relied on by each caller. Only the
# part that has to be fast and memory-safe is shared.
#
# THE BUG THIS FIXES ("classify using the pkl files is too slow" / "a very
# big file will take forever and crash the program"): both callers used to
# build a spatial-window (window_size > 1) feature matrix with a pure Python
# `for y: for x: for dr: for dc: for base_name: ... arr[py,px] ...` loop --
# O(H*W*window_size^2*n_features) Python-level scalar operations touching
# every pixel individually (minutes on a modest image, unusable on a large
# one), AND unconditionally allocated the WHOLE feature array up front with
# no memory budget (H*W*F*4 bytes -- ~10.8 GB for a 10000x10000/3-band/
# window=3 image), a straightforward OOM crash. process_polygon never has
# this problem because it's always bounded to one polygon's ROI via
# LazyChannels.read_window's own memory-budgeted lazy decode
# (raster_reader.py's max_materialize / _DEFAULT_CACHE_BYTES) -- there is no
# small ROI to exploit here (the user wants the WHOLE image classified), so
# the fix instead (a) vectorizes the window construction and (b) tiles by a
# row-band memory budget, mirroring that same "bound peak memory, decode in
# pieces past a threshold" principle.
# =============================================================================

def build_windowed_feature_cube(padded_arrays, base_names, window_size, out_h, out_w):
    """Vectorized replacement for the old per-pixel window-feature loop.

    `padded_arrays` maps each name in `base_names` to its 2D array, already
    padded by `window_size // 2` on every side with `mode='edge'` (same
    padding both callers already applied before the old loop). `out_h`/
    `out_w` are the UNPADDED image dimensions.

    Returns an (out_h, out_w, F) float32 array, F = window_size**2 *
    len(base_names), with columns in the EXACT order the old loop produced:
    (dr, dc) outer (row-major over [-half, half]), `base_names` inner. This
    is what makes it a drop-in replacement -- a model trained against the
    old column order still gets the same feature at the same index.

    Equivalence: the old loop read `padded[name][y+half+dr, x+half+dc]` for
    every pixel (y, x). For one fixed (dr, dc), that is exactly the slice
    `padded[name][half+dr : half+dr+out_h, half+dc : half+dc+out_w]` --
    every (y, x) at once. So this replaces H*W Python-level reads per
    (dr, dc, name) with one C-level array copy, without changing a single
    output value.
    """
    half = window_size // 2
    F = (window_size * window_size) * len(base_names)
    cube = np.empty((out_h, out_w, F), dtype=np.float32)
    feat_idx = 0
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r0 = half + dr
            c0 = half + dc
            for name in base_names:
                cube[:, :, feat_idx] = padded_arrays[name][r0:r0 + out_h, c0:c0 + out_w]
                feat_idx += 1
    return cube


#: Row-band tiling stays correct for any window_size this codebase offers
#: (1, 3, or 5 per the Image Editor's own UI) -- the halo is always small.
CLASSIFY_MAX_WINDOW_SIZE = 5


def iter_classification_row_tiles(h, w, n_features, budget_bytes, half, dtype_bytes=4):
    """Yield (row0, row1, halo_row0, halo_row1) row-band tiles whose feature
    matrix (tile_rows * w * n_features * dtype_bytes) fits `budget_bytes`.

    `half = window_size // 2`. `halo_row0`/`halo_row1` extend the tile by up
    to `half` REAL rows on each side (clamped at the true image edges) so a
    windowed feature builder given `arr[halo_row0:halo_row1]` sees genuine
    neighbor rows at every interior tile seam -- not the tile's own edge-pad,
    which would incorrectly replicate an interior row as if it were the
    image boundary. The caller discards the halo rows after building/
    predicting on the padded tile, keeping only `[row0-halo_row0 :
    row0-halo_row0+(row1-row0)]` of the result.

    If the whole image already fits the budget, yields exactly one tile
    covering everything (`(0, h, 0, h)`) -- the common case is a plain,
    single-shot pass, identical in behavior (and byte-for-byte output) to
    processing it unbounded.
    """
    if h <= 0 or w <= 0:
        return
    total_bytes = h * w * n_features * dtype_bytes
    if total_bytes <= budget_bytes or n_features <= 0:
        yield (0, h, 0, h)
        return

    per_row_bytes = max(1, w * n_features * dtype_bytes)
    tile_rows = max(1, budget_bytes // per_row_bytes)
    row0 = 0
    while row0 < h:
        row1 = min(h, row0 + tile_rows)
        halo_row0 = max(0, row0 - half)
        halo_row1 = min(h, row1 + half)
        yield (row0, row1, halo_row0, halo_row1)
        row0 = row1


def iter_tiled_feature_matrices(img, feat_names, expressions, window_size,
                                 base_feature_names, budget_bytes, builder_fn):
    """Bound peak memory for `builder_fn` (either class's
    `_make_feature_stack_for_model`) regardless of image size.

    Estimates the full feature matrix's byte size up front; if it fits
    `budget_bytes`, calls `builder_fn` ONCE on the whole image (identical,
    single-shot behavior to before this fix existed). If it doesn't, slices
    `img` into row-band tiles via `iter_classification_row_tiles` (each with
    a `half`-row halo of REAL neighbor rows) and calls `builder_fn`
    separately on each tile slice, discarding the halo rows from each
    result before yielding just the tile's own core rows.

    This never needs `builder_fn` itself to know about tiling: calling it on
    `img[halo_row0:halo_row1]` and keeping only rows
    `[row0-halo_row0 : row0-halo_row0+(row1-row0)]` of its output is
    mathematically identical to what it would have computed for those same
    rows given the WHOLE image -- the halo rows supply genuine neighbor
    context at interior tile seams, so `builder_fn`'s own edge-padding is
    only ever exercised at the image's true boundary, exactly as it would be
    without tiling.

    Yields `(row0, row1, X_tile, w)` per tile, `X_tile` shaped
    `((row1-row0)*w, F)` -- or `(row0, row1, None, w)` if `builder_fn`
    signaled failure for that tile (its own `(None, (0, 0))` convention),
    which callers should treat exactly like today's single-call failure.
    """
    h, w = img.shape[0], img.shape[1]
    n_features = len(feat_names)
    half = (window_size or 1) // 2
    total_bytes = h * w * n_features * 4  # float32

    if total_bytes <= budget_bytes:
        X, (out_h, out_w) = builder_fn(img, feat_names, expressions, window_size,
                                       base_feature_names)
        yield (0, h, X, w)
        return

    for row0, row1, halo_row0, halo_row1 in iter_classification_row_tiles(
            h, w, n_features, budget_bytes, half):
        tile_img = img[halo_row0:halo_row1]
        X_tile, (tile_h, tile_w) = builder_fn(tile_img, feat_names, expressions,
                                              window_size, base_feature_names)
        if X_tile is None:
            yield (row0, row1, None, w)
            continue
        core_lo = row0 - halo_row0
        core_hi = core_lo + (row1 - row0)
        X_core = X_tile.reshape(halo_row1 - halo_row0, w, -1)[core_lo:core_hi]
        yield (row0, row1, X_core.reshape(-1, X_core.shape[-1]), w)


# =============================================================================
# Feature normalization -- bundle-recorded, applied identically at train time
# and at every classification/predict call site that later loads the bundle.
#
# THE PROBLEM THIS SOLVES: `.pkl` model bundles never carried any normalization
# state. The only scaling anywhere in the app was an sklearn `Pipeline`-internal
# `StandardScaler` private to the LogisticRegression/SVM model types (invisible
# to every other consumer, and to every OTHER of the 8 selectable model types).
# A user wanting L2 or z-score normalization for e.g. RandomForest had no way to
# apply it consistently between training and the ~6 independent places the app
# later re-predicts from a saved bundle (process_polygon's CSV export,
# ProjectTab's classification replay/appended-band paths, the Image Editor's
# live "Classify" button, MachineLearningManager's thumbnail/segmentation
# export). These two functions are the single shared implementation every one
# of those sites calls, keyed off a `bundle["normalization"]` dict of the shape
# {"method": "none"|"l2"|"zscore", "mean": [...]|None, "scale": [...]|None}.
# =============================================================================

def fit_normalization(X, method):
    """Compute the normalization_cfg to store in a model bundle.

    X: (N, F) float32/float64 training feature matrix (the array that will
    actually be fit on -- see machine_learning_manager.py's train_models,
    which calls this on X_train, i.e. AFTER any augmented rows have been
    added, so the fitted stats reflect what the model actually trains under).
    method: 'none' | 'l2' | 'zscore'.

    'l2' needs no fitted parameters (Normalizer(norm='l2') semantics are
    entirely per-row, recomputed fresh from whatever X apply_normalization is
    later given). 'zscore' fits per-feature mean/std here, once, and both must
    be stored in the bundle so every later predict call applies the SAME
    fitted transform rather than re-fitting on its own (usually much smaller,
    single-image) feature matrix -- refitting per call would make predictions
    drift from what the model was actually trained on.

    Returns a dict matching the bundle's "normalization" key shape exactly;
    always all three keys present so callers never need `"mean" in cfg`-style
    membership checks, only `cfg["method"]`.
    """
    method = (method or "none").lower()
    if method not in ("none", "l2", "zscore"):
        raise ValueError(f"Unknown normalization method: {method!r}")

    if method != "zscore":
        return {"method": method, "mean": None, "scale": None}

    Xf = np.asarray(X, dtype=np.float64)
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0)
    # A constant feature column has std==0; dividing by it would produce
    # inf/nan for every sample. Floor it to 1.0 (mirrors sklearn's
    # StandardScaler _handle_zeros_in_scale) so a constant column normalizes
    # to a constant 0.0 instead of blowing up.
    scale = np.where(scale < 1e-12, 1.0, scale)
    return {"method": "zscore", "mean": mean.tolist(), "scale": scale.tolist()}


def apply_normalization(X, normalization_cfg):
    """Transform a feature matrix per a bundle's "normalization" config.

    X: (N, F) or (F,) array (a single row is accepted too -- several of the
    predict call sites this feeds slice a single sample out of a batch before
    calling this). Returns a NEW float32 array; never mutates X in place, since
    several callers keep using the pre-normalization X afterward (e.g. for
    label mapping keyed by row index).

    normalization_cfg: the dict fit_normalization returns (or a plain-dict
    equivalent read back from a pickled bundle, or None/{} for "no bundle
    entry at all" -- treated identically to {"method": "none"}, so every call
    site can pass `bundle.get("normalization")` directly without a None-guard).
    """
    Xin = np.asarray(X, dtype=np.float32)
    cfg = normalization_cfg or {"method": "none"}
    method = (cfg.get("method") or "none").lower()

    if method == "none":
        return Xin.copy() if Xin is X else Xin

    single_row = (Xin.ndim == 1)
    X2 = Xin[None, :] if single_row else Xin

    if method == "l2":
        norms = np.linalg.norm(X2, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        out = (X2 / norms).astype(np.float32, copy=False)
    elif method == "zscore":
        mean = cfg.get("mean")
        scale = cfg.get("scale")
        if mean is None or scale is None:
            raise ValueError(
                "normalization_cfg method='zscore' but 'mean'/'scale' are "
                "missing -- the bundle is malformed or predates this feature.")
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if X2.shape[-1] != mean.shape[0]:
            raise ValueError(
                f"Feature count mismatch for zscore normalization: X has "
                f"{X2.shape[-1]} columns but the bundle's fitted mean/scale "
                f"have {mean.shape[0]} -- this model was trained on a "
                f"different feature set than the one being predicted with.")
        out = ((X2 - mean) / scale).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return out[0] if single_row else out


# =============================================================================
# NoData Utilities - Support both numeric literals and boolean expressions
# =============================================================================

# Pattern for threshold expressions: b1<123, b2>=50, B3>100, etc.
# Equality must be written `==`; a bare `=` is deliberately NOT accepted.
_NODATA_EXPR_RE = re.compile(
    r'^([bB]\d+)\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)$'
)

def parse_nodata_text(text: str):
    """
    Parse comma-separated NoData values from user input.
    Supports:
      - Numeric literals: 0, -9999, 255
      - Boolean expressions: b1<123, b2>=50, B3>100
    
    Returns list of (float | str), where strings are threshold expressions.
    """
    if not text or not text.strip():
        return []
    
    # Remove curly braces if present (allow {-9999, 0} format)
    text = text.strip().strip('{}')
    
    result = []
    seen = set()
    
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        
        # Check if it's a threshold expression like b1<123
        if _NODATA_EXPR_RE.match(part):
            # Normalize case: b1 -> b1, B1 -> b1
            normalized = re.sub(r'^[bB]', 'b', part)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
            continue
        
        # Try parsing as numeric literal
        try:
            if '.' not in part and 'e' not in part.lower():
                val = int(part)
            else:
                val = float(part)
            if val not in seen:
                seen.add(val)
                result.append(val)
        except ValueError:
            logging.warning(f"Could not parse NoData value: {part}")
    
    return result


def build_nodata_mask(img, nd_vals, *, bgr_input=True, include_nonfinite=False):
    """
    Build boolean mask where True = NoData pixel.

    Supports:
      - Numeric literals: Match any channel within tolerance
      - Threshold expressions: b1<123 evaluates on specific band

    Parameters
    ----------
    img : ndarray (H, W) or (H, W, C)
        Image array (any dtype, converted to float32 internally).
    nd_vals : list
        List of numeric values or expression strings from parse_nodata_text.
    bgr_input : bool
        If True, input is BGR (OpenCV default): b1=Red(ch2), b2=Green(ch1), b3=Blue(ch0).
        If False, input is RGB: b1=ch0, b2=ch1, b3=ch2.
    include_nonfinite : bool
        If True, build the mask even when `nd_vals` is empty, so that NaN/Inf pixels
        alone are reported as NoData.

        Historically this function returned None whenever `nd_vals` was empty, which
        meant the NaN/Inf pass at the end (labelled "always masked") never ran unless
        the user had already typed an explicit NoData value. Float GeoTIFFs whose
        borders are NaN were therefore never auto-detected, and those NaNs leaked into
        the contrast-stretch statistics. Stretch/statistics callers should pass True;
        callers that USE the mask to restore or overwrite pixel values must keep the
        default (False) so their behaviour is unchanged.

    Returns
    -------
    mask : ndarray (H, W) of bool, or None if no valid values
    """
    if img is None:
        return None
    if not nd_vals and not include_nonfinite:
        return None
    if not nd_vals:
        # Nonfinite-only request: integer imagery can never hold NaN/Inf, so skip the
        # full-image scan entirely and keep the cheap `None` fast path.
        try:
            if not np.issubdtype(np.asarray(img).dtype, np.floating):
                return None
        except Exception:
            pass

    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    H, W = x.shape[:2]
    C = x.shape[2] if x.ndim == 3 else 1
    
    mask = np.zeros((H, W), dtype=bool)
    
    # Build band index mapping (b1, b2, b3... -> channel index)
    # For BGR input (OpenCV): b1=Red=ch2, b2=Green=ch1, b3=Blue(ch0)
    # For RGB input: b1=ch0, b2=ch1, b3=ch2
    def _get_channel_idx(band_num):
        """Convert band number (1-based) to channel index."""
        if C == 1:
            return 0  # Single channel - all band references go to channel 0
        if C == 2:
            # 2-channel: b1->0, b2->1, b3+ out of bounds
            return band_num - 1 if band_num <= 2 else band_num - 1
        if bgr_input and C == 3 and band_num <= 3:
            # BGR (3-channel only): b1->2 (Red), b2->1 (Green), b3->0 (Blue)
            return 2 - (band_num - 1)  # b1->2, b2->1, b3->0
        else:
            # RGB or multispectral (C>3) or C==3 with RGB input: direct indexing
            return band_num - 1
    
    for v in nd_vals:
        if isinstance(v, str):
            # Threshold expression: parse and evaluate
            m = _NODATA_EXPR_RE.match(v)
            if m:
                band_name, op, threshold = m.groups()
                band_num = int(band_name[1:])  # b1 -> 1, b2 -> 2, etc.
                ch_idx = _get_channel_idx(band_num)
                
                if ch_idx >= C:
                    logging.warning(f"NoData expression {v}: band {band_num} exceeds image channels ({C})")
                    continue
                
                ch = x[..., ch_idx]
                threshold_val = float(threshold)
                
                # Apply comparison operator
                if op == '<':
                    mask |= (ch < threshold_val)
                elif op == '<=':
                    mask |= (ch <= threshold_val)
                elif op == '>':
                    mask |= (ch > threshold_val)
                elif op == '>=':
                    mask |= (ch >= threshold_val)
                elif op == '==':
                    mask |= np.isclose(ch, threshold_val, rtol=0.0, atol=1e-6)
                elif op == '!=':
                    mask |= ~np.isclose(ch, threshold_val, rtol=0.0, atol=1e-6)
        else:
            # Numeric literal: match any channel within tolerance
            try:
                fv = float(v)
                abs_fv = abs(fv)
                # Use appropriate tolerance based on value magnitude
                if abs_fv > 1e+30:
                    tol = abs_fv * 0.01
                elif abs_fv > 1e+10:
                    tol = abs_fv * 0.001
                elif abs_fv > 100:
                    tol = abs_fv * 0.001
                else:
                    tol = 0.01
                # ONE call across every channel at once, instead of a Python
                # loop calling subtract/abs/compare once per channel (C calls
                # each). Equivalent output (NaN still compares False either
                # way -- see the isclose comment this replaced), but far fewer
                # ufunc dispatches: on real multi-band prediction stacks
                # (15 bands, ~44% NaN) the per-channel loop measured highly
                # inconsistent -- 70ms in a clean process but 5-6x that
                # (400+ms) reached the SAME array through the app's actual
                # call chain, for reasons that didn't trace to array layout,
                # NaN bit pattern, or system load (all checked and ruled out
                # -- see task notes). Cutting dispatch count from 2*C to 2
                # measurably helped in BOTH cases and is the safer fix when
                # the exact mechanism can't be pinned down.
                mask |= (np.abs(x - fv) <= tol).any(axis=2)
            except Exception:
                pass

    # Also check for NaN/Inf (always masked). One vectorized pass across all
    # channels at once (see the numeric-literal comment above for why this
    # replaced a per-channel loop).
    if np.issubdtype(x.dtype, np.floating):
        mask |= (~np.isfinite(x)).any(axis=2)

    return mask


# ---------------------------------------------------------------------------
# Per-polygon `properties` -> export columns
# ---------------------------------------------------------------------------
#: Every CSV column derived from a polygon's arbitrary `properties` dict is
#: prefixed with this. `properties` keys come from an imported shapefile's DBF
#: (truncated to 10 chars by the format) or from the viewer's "Edit
#: properties" dialog -- i.e. entirely user-controlled, and different from one
#: project to the next. Without a prefix a DBF field named `Mean`, `x` or
#: `group_name` would silently overwrite a computed statistic column in one of
#: the four export paths that read this. Prefixing is injective, so two
#: distinct keys can never collide with each other either.
#:
#: NOTE the shapefile/DBF export path (shapefile_io.py's json_polygons_to_features)
#: does NOT use this prefix, on purpose: DBF field names are capped at 10
#: characters, so `prop_` would leave 5 usable, and that path already has its
#: own collision guard (`if k in props: continue`) plus its own pinned test
#: (canopie/qc/test_shapefile_batch_export.py). This asymmetry is deliberate.
POLYGON_PROPERTY_COLUMN_PREFIX = 'prop_'


def polygon_property_column(key):
    """Column name for one raw property key. CR/LF/TAB are replaced with a
    space so a pasted multi-line key cannot break a CSV row's framing."""
    s = str(key)
    for ch in ('\r', '\n', '\t'):
        s = s.replace(ch, ' ')
    return POLYGON_PROPERTY_COLUMN_PREFIX + s


def polygon_property_value(v):
    """Coerce one property value to something every writer here can emit.
    Scalars pass through unchanged; None becomes '' so a missing/blank cell
    reads the same as an absent key; numpy scalars are unwrapped; anything
    else is stringified rather than raising mid-export."""
    if v is None:
        return ''
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return str(v)


def collect_polygon_property_keys(polygons):
    """SORTED union of raw property keys across `polygons`.

    `polygons` is any iterable of polygon dicts. Sorted (not first-seen) so
    the header is byte-identical between runs regardless of group-selection
    order or dict insertion order -- the same reproducibility contract
    order_csv_columns keeps for its own alphabetical tail.

    A polygon with no `properties` key at all -- every hand-drawn one -- (or
    a non-dict `properties`) contributes nothing and is never an error.
    """
    keys = set()
    for p in polygons:
        if not isinstance(p, dict):
            continue
        props = p.get('properties')
        if isinstance(props, dict):
            keys.update(str(k) for k in props.keys())
    return sorted(keys)


def iter_polygons_by_group(polygons_by_group):
    """Flatten {group: {filepath: polygon_dict}} to an iterable of polygon
    dicts -- the shape machine_learning_manager.py holds its snapshots in."""
    for file_map in (polygons_by_group or {}).values():
        if isinstance(file_map, dict):
            for poly in file_map.values():
                yield poly


def polygon_property_cells(polygon_dict, keys=None):
    """{'prop_<key>': value} for ONE polygon, for dict-shaped (DictWriter) rows.

    keys=None  -> only the keys this polygon actually has.
    keys=[...] -> every listed key is present; ones this polygon lacks get ''.
    """
    props = polygon_dict.get('properties') if isinstance(polygon_dict, dict) else None
    if not isinstance(props, dict):
        props = {}
    if keys is None:
        return {polygon_property_column(k): polygon_property_value(v)
                for k, v in props.items()}
    return {polygon_property_column(k): polygon_property_value(props.get(k, ''))
            for k in keys}


def polygon_property_values(polygon_dict, keys):
    """FIXED-ORDER list of values aligned to `keys`, for list-shaped
    (csv.writer) rows. Missing keys -> ''. Intended to be computed ONCE per
    polygon and spliced onto every row/tile that polygon produces, so a
    polygon yielding many rows (pixels) or many tiles never re-derives it."""
    props = polygon_dict.get('properties') if isinstance(polygon_dict, dict) else None
    if not isinstance(props, dict):
        props = {}
    return [polygon_property_value(props.get(k, '')) for k in keys]


__all__ = [
    'STRETCH_LOW_P',
    'STRETCH_HIGH_P',
    'STRETCH_PER_CHANNEL',
    'STRETCH_CLIP',
    'STRETCH_SAMPLE_MAX',
    '_dims_after_rot',
    '_rect_after_rot',
    '_scale_rect',
    '_infer_crop_basis',
    '_rotate_point_in_rect',
    'resize_safe',
    '_nanpct',
    '_normalize_for_display',
    '_sample_for_stats',
    'process_band_expression_float',
    'process_band_expression',
    'get_exif_data_exiftool_multiple',
    'calculate_exg',
    'calculate_gcc',
    'calculate_bcc',
    'calculate_gbd',
    'calculate_wdx',
    'calculate_shd',
    'parse_nodata_text',
    'build_nodata_mask',
    'build_windowed_feature_cube',
    'iter_classification_row_tiles',
    'iter_tiled_feature_matrices',
    'CLASSIFY_MAX_WINDOW_SIZE',
    'fit_normalization',
    'apply_normalization',
    'POLYGON_PROPERTY_COLUMN_PREFIX',
    'polygon_property_column',
    'polygon_property_value',
    'collect_polygon_property_keys',
    'iter_polygons_by_group',
    'polygon_property_cells',
    'polygon_property_values',
]