import os
import json
import tempfile
import subprocess
import logging
import numpy as np
import cv2


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

def resize_safe(arr, new_w, new_h, interp=cv2.INTER_LINEAR):
    """
    Resize an array to new_w x new_h.  Supports arbitrary channel counts.
    Preserves dtype and handles 2D or HxWxC arrays.  Unknown ranks are returned unchanged.
    """
    if arr is None:
        return None
    if arr.ndim == 2:
        return cv2.resize(arr, (new_w, new_h), interpolation=interp)
    if arr.ndim == 3:
        h, w, c = arr.shape
        if c <= 4:
            return cv2.resize(arr, (new_w, new_h), interpolation=interp)
        # per‑channel path
        out = np.empty((new_h, new_w, c), dtype=arr.dtype)
        for i in range(c):
            out[..., i] = cv2.resize(arr[..., i], (new_w, new_h), interpolation=interp)
        return out
    # Unknown rank; return as‑is
    return arr

def _normalize_for_display(img, low_p=STRETCH_LOW_P, high_p=STRETCH_HIGH_P,
                           per_channel=STRETCH_PER_CHANNEL, clip=STRETCH_CLIP,
                           sample_max=STRETCH_SAMPLE_MAX):
    """
    Normalise an image for display by stretching its pixel values between low_p and high_p percentiles.
    Supports grayscale and RGB images and works on any numeric dtype.
    Returns an 8‑bit image suitable for display (max 3 channels).
    """
    if img is None:
        return None

    if img.dtype == np.uint8 and img.ndim in (2, 3):
        disp = img
    else:
        x = np.nan_to_num(img.astype(np.float32, copy=False))
        def _sample(a):
            h, w = a.shape[:2]
            m = max(h, w)
            if m <= sample_max:
                return a
            s = sample_max / float(m)
            return cv2.resize(a, (max(1, int(round(w*s))), max(1, int(round(h*s)))), interpolation=cv2.INTER_AREA)

        if x.ndim == 2:
            s = _sample(x)
            lo = np.percentile(s, low_p)
            hi = np.percentile(s, high_p)
            n = np.full_like(x, 0.5, dtype=np.float32) if hi <= lo else (x - lo) / max(hi - lo, 1e-12)
            if clip:
                n = np.clip(n, 0.0, 1.0)
            disp = (n * 255.0).astype(np.uint8)

        elif x.ndim == 3:
            C = x.shape[2]
            use = x[:, :, :max(1, min(C, 3))]  # take up to 3 channels for preview
            s = _sample(use)
            if per_channel:
                flat = s.reshape(-1, use.shape[2])
                lo = np.percentile(flat, low_p, axis=0)
                hi = np.percentile(flat, high_p, axis=0)
                scale = np.maximum(hi - lo, 1e-12)
                n = (use - lo.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
            else:
                lo = np.percentile(s, low_p)
                hi = np.percentile(s, high_p)
                n = np.full_like(use, 0.5, dtype=np.float32) if hi <= lo else (use - lo) / max(hi - lo, 1e-12)
            if clip:
                n = np.clip(n, 0.0, 1.0)
            disp = (n * 255.0).astype(np.uint8)
            # If only 1 or 2 channels, pad to 3 for display
            if disp.shape[2] == 1:
                disp = np.repeat(disp, 3, axis=2)
            elif disp.shape[2] == 2:
                disp = np.concatenate([disp, disp[:, :, :1]], axis=2)
        else:
            return None

    # Ensure max 3 channels for QImage
    if disp.ndim == 3 and disp.shape[2] > 3:
        disp = disp[:, :, :3].copy()
    return disp

def _sample_for_stats(arr, sample_max=STRETCH_SAMPLE_MAX):
    """
    Return a downsampled copy of the input array such that its largest dimension does not
    exceed sample_max.  This is used to compute statistics on large images.
    """
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= sample_max:
        return arr
    scale = sample_max / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

def process_band_expression_float(image, expr):
    """
    Evaluate a 'b1..bN' expression on FLOAT32.
    Only 'b1..bN' names are allowed (case‑insensitive).  If the expression
    requests bands not present, the original image is returned unchanged.
    Division is rewritten to safe_div(x, y) to avoid division by zero.
    NaN/Inf values are sanitised to 0.  Returns a 2D float32 array on success.
    """
    import re, ast
    if image is None or not expr:
        return image
    # Work in float32 and sanitise any preexisting NaN/Inf
    x = np.nan_to_num(np.asarray(image, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    # Determine band count
    if x.ndim == 2:
        H, W = x.shape
        C = 1
    elif x.ndim == 3:
        H, W, C = x.shape
    else:
        return image
    # Case‑insensitive band references
    expr_norm = str(expr).strip()
    # Bands requested by the expression (case‑insensitive)
    req = sorted({int(b) for b in re.findall(r'\bb(\d+)\b', expr_norm, flags=re.IGNORECASE)})
    if any(b > C for b in req):
        logging.warning(
            "Band expression '%s' requests b%d but image has only %d band(s); skipping.",
            expr, max(req) if req else 1, C
        )
        return image
    # Build mapping {b1: plane0, ...}
    if C == 1:
        mapping = {'b1': x}
    else:
        mapping = {f"b{i+1}": x[:, :, i] for i in range(C)}
    # Also allow upper‑case references transparently by mirroring keys
    mapping.update({k.upper(): v for k, v in mapping.items()})
    # Safe divider (no warnings; 0 where denom==0)
    def safe_div(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        out = np.zeros(np.broadcast(a, b).shape, dtype=np.float32)
        np.divide(a, b, out=out, where=(b != 0))
        return out
    # AST parsing with LRU caching on the function object
    if not hasattr(process_band_expression_float, "_compiled_cache"):
        process_band_expression_float._compiled_cache = {}
    cache = process_band_expression_float._compiled_cache
    compiled = cache.get(expr_norm)
    if compiled is None:
        class _DivFix(ast.NodeTransformer):
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                if isinstance(node.op, ast.Div):
                    return ast.Call(func=ast.Name(id="safe_div", ctx=ast.Load()),
                                    args=[node.left, node.right],
                                    keywords=[])
                return node
        # Parse
        try:
            tree = ast.parse(expr_norm, mode="eval")
        except SyntaxError as e:
            logging.warning(f"Band expression syntax error ('{expr}'): {e}")
            return image
        # Security/validation: only allow a small set of nodes
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
            ast.USub, ast.UAdd, ast.Load, ast.Name, ast.Constant, ast.Tuple, ast.List,
            ast.Call  # but only safe_div after transform
        )
        class _Validator(ast.NodeVisitor):
            def __init__(self):
                self.ok = True
            def generic_visit(self, node):
                if not isinstance(node, allowed_nodes):
                    self.ok = False
                super().generic_visit(node)
        v = _Validator()
        v.visit(tree)
        if not v.ok:
            logging.warning("Illegal syntax/nodes in band expression '%s'; skipping.", expr)
            return image
        # Replace divisions
        tree = _DivFix().visit(tree)
        ast.fix_missing_locations(tree)
        # Collect names and ensure they are permitted identifiers
        class _Names(ast.NodeVisitor):
            def __init__(self):
                self.names = set()
            def visit_Name(self, node):
                self.names.add(node.id)
        names = _Names()
        names.visit(tree)
        for name in names.names:
            if name not in mapping and name != "safe_div":
                logging.warning("Illegal name '%s' in band expr '%s' (float); skipping index.", name, expr)
                return image
        compiled = compile(tree, "<string>", "eval")
        cache[expr_norm] = compiled
    # --- Evaluate safely --------------------------------------------------------
    try:
        with np.errstate(all='ignore'):
            res = eval(compiled, {"__builtins__": {}, "safe_div": safe_div}, mapping)
    except Exception as e:
        logging.warning(f"Band expression failed ('{expr}'): {e}")
        return image
    # Normalise output type/shape to 2D float32
    if not isinstance(res, np.ndarray):
        res = np.full((H, W), float(res), dtype=np.float32)
    else:
        res = np.asarray(res, dtype=np.float32)
        if res.ndim == 3 and res.shape[-1] == 1:
            res = res[..., 0]
        elif res.ndim != 2:
            # Defensive: reduce unexpected ranks
            res = np.mean(res, axis=tuple(range(res.ndim - 2)), dtype=np.float32)
    # Final sanitise
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return res

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
        result = subprocess.run(command, capture_output=True, text=True, check=True)
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
]