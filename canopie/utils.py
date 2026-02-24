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
        # Convert to float32 for stretching, replace NaNs/Infs
        x = np.nan_to_num(np.asarray(img).astype(np.float32, copy=False))

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
            lo = np.percentile(s, low_p)
            hi = np.percentile(s, high_p)
            n = (x - lo) / max(hi - lo, 1e-12) if hi > lo else np.full_like(x, 0.5, dtype=np.float32)
            if clip:
                n = np.clip(n, 0.0, 1.0)
            disp = (n * 255.0).astype(np.uint8)

        elif x.ndim == 3:
            C = x.shape[2]
            use = x[:, :, :max(1, min(C, 3))]  # take up to 3 channels for preview
            s = _sample(use)
            if per_channel and use.ndim == 3 and use.shape[2] > 1:
                flat = s.reshape(-1, use.shape[2])
                lo = np.percentile(flat, low_p, axis=0)
                hi = np.percentile(flat, high_p, axis=0)
                scale = np.maximum(hi - lo, 1e-12)
                n = (use - lo.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
            else:
                lo = np.percentile(s, low_p)
                hi = np.percentile(s, high_p)
                n = (use - lo) / max(hi - lo, 1e-12) if hi > lo else np.full_like(use, 0.5, dtype=np.float32)
            if clip:
                n = np.clip(n, 0.0, 1.0)
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

    # Safe divide
    def safe_div(a, b):
        a = _to_arr(a); b = _to_arr(b)
        out = _np.zeros(_np.broadcast(a, b).shape, dtype=_np.float32)
        _np.divide(a, b, out=out, where=(b != 0))
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
# NoData Utilities - Support both numeric literals and boolean expressions
# =============================================================================

# Pattern for threshold expressions: b1<123, b2>=50, B3>100, etc.
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


def build_nodata_mask(img, nd_vals, *, bgr_input=True):
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
    
    Returns
    -------
    mask : ndarray (H, W) of bool, or None if no valid values
    """
    if not nd_vals or img is None:
        return None
    
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
                for c in range(C):
                    ch = x[..., c]
                    mask |= np.isclose(ch, fv, rtol=0.0, atol=tol)
            except Exception:
                pass
    
    # Also check for NaN/Inf (always masked)
    for c in range(C):
        ch = x[..., c]
        mask |= np.isnan(ch)
        mask |= np.isinf(ch)
    
    return mask


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
]