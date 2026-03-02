
import cv2
import numpy as np
import logging
import re
from .utils import resize_safe, build_nodata_mask, eval_band_expression

logger = logging.getLogger(__name__)

def apply_pipeline(image, modifications, mask_polygon_raw_data=None):
    """
    Apply all image modifications (crop, rotate, hist, resize, band) to 'image'.
    
    Args:
        image (np.ndarray): The source image (H,W,C) or (H,W). BGR if 3-channel.
        modifications (dict): The .ax modifications dictionary.
        mask_polygon_raw_data (list): Optional. List of (points, ref_size) from project data 
                                      for mask polygons. If None, uses local points if any.
    
    Returns:
        np.ndarray: The processed image.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image

    mods = modifications or {}
    
    # --- 1. Enabled Flags ---
    rotate_enabled = mods.get("rotate_enabled", True)
    crop_enabled = mods.get("crop_enabled", True)
    hist_enabled = mods.get("hist_enabled", True)
    resize_enabled = mods.get("resize_enabled", True)
    band_enabled = mods.get("band_enabled", True)
    nodata_enabled = mods.get("nodata_enabled", True)
    
    nodata_values = list(mods.get("nodata_values", []) or []) if nodata_enabled else []

    # Mask Polygons
    mask_polygon_cfg = mods.get("mask_polygon", {}) or {}
    mask_polygon_enabled = bool(mask_polygon_cfg.get("enabled", False)) if isinstance(mask_polygon_cfg, dict) else False
    
    # If raw data not provided, check for legacy points
    mask_polygon_points_list = []
    if not mask_polygon_raw_data:
        legacy_points = mask_polygon_cfg.get("points", [])
        if legacy_points and len(legacy_points) >= 3:
            mask_polygon_points_list = [legacy_points]
            
    # --- 2. Crop & Rotate ---
    # Determine order
    rot = int(mods.get("rotate", 0)) if "rotate" in mods else 0
    rot = ((rot % 360) + 360) % 360
    
    result = image.copy()
    raw_h, raw_w = result.shape[:2]
    rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
    
    crop_rect_data = mods.get("crop_rect") if crop_enabled else None
    crop_ref = mods.get("crop_rect_ref_size") or {}
    
    do_rotate_first = True
    if crop_rect_data and rot in (90, 180, 270):
        ref_w = int(crop_ref.get("w", 0)) or 0
        ref_h = int(crop_ref.get("h", 0)) or 0
        if ref_w > 0 and ref_h > 0:
            if (ref_w, ref_h) == (raw_w, raw_h):
                do_rotate_first = False
            elif (ref_w, ref_h) == (rotated_w, rotated_h):
                do_rotate_first = True

    if do_rotate_first:
        result = _apply_rotate(result, rot, rotate_enabled)
        result = _apply_crop(result, mods, crop_enabled)
    else:
        result = _apply_crop(result, mods, crop_enabled)
        result = _apply_rotate(result, rot, rotate_enabled)
        
    if result is None or result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0:
        return image # Return original if processed is empty (or return None?)

    # --- 3. Histogram Matching ---
    deferred_hist_block = None
    hcfg_now = (mods.get("hist_match", None)) if hist_enabled else None
    
    # Check for deferred resize (optimization)
    # Assume default optimization is TRUE if not specified (safe default)
    is_shrink = False
    
    if resize_enabled and "resize" in mods:
        # Calculate new dims
        h0, w0 = result.shape[:2]
        new_w, new_h = _calc_resize_dims(w0, h0, mods["resize"])
        if new_w < w0 or new_h < h0:
            is_shrink = True
            
    if hcfg_now and is_shrink:
        deferred_hist_block = hcfg_now
        hcfg_now = None
        
    if hcfg_now:
        result = _apply_hist_match(result, hcfg_now, nodata_values, 
                                   mask_polygon_raw_data, mask_polygon_points_list, mask_polygon_enabled)

    # --- 4. Resize ---
    if resize_enabled and "resize" in mods:
        result = _apply_resize(result, mods["resize"], nodata_values)

    # --- 5. Deferred Histogram ---
    if deferred_hist_block:
        result = _apply_hist_match(result, deferred_hist_block, nodata_values, 
                                   mask_polygon_raw_data, mask_polygon_points_list, mask_polygon_enabled)

    # --- 6. Band Expression ---
    if band_enabled and "band_expression" in mods:
        expr = (mods.get("band_expression") or "").strip()
        if expr:
            result = _process_band_expression(result, expr, nodata_values)

    return result


def _apply_rotate(img, rot, enabled):
    if not enabled or rot not in (90, 180, 270):
        return img
    
    d = int(rot) % 360
    if d == 90:
        out = np.rot90(img, -1)
    elif d == 180:
        out = np.rot90(img, 2)
    elif d == 270:
        out = np.rot90(img, 1)
    else:
        return img
    return np.ascontiguousarray(out)


def _apply_crop(img, mods, enabled):
    if not enabled or "crop_rect" not in mods:
        return img
        
    crop_rect = mods["crop_rect"]
    x = int(crop_rect.get("x", 0))
    y = int(crop_rect.get("y", 0))
    w = int(crop_rect.get("width", 0))
    h = int(crop_rect.get("height", 0))
    
    H, W = img.shape[:2]
    
    ref = mods.get("crop_rect_ref_size") or {}
    ref_w = int(ref.get("w", W)) or W
    ref_h = int(ref.get("h", H)) or H
    
    if ref_w != W or ref_h != H:
        sx = W / float(ref_w)
        sy = H / float(ref_h)
        x = int(round(x * sx)); y = int(round(y * sy))
        w = int(round(w * sx)); h = int(round(h * sy))
        
    x0 = max(0, min(x, W))
    y0 = max(0, min(y, H))
    x1 = max(0, min(x + w, W))
    y1 = max(0, min(y + h, H))
    
    if x1 > x0 and y1 > y0:
        return img[y0:y1, x0:x1]
    return img


def _calc_resize_dims(w0, h0, resize_info):
    new_w, new_h = w0, h0
    if "px_w" in resize_info or "px_h" in resize_info:
        tw = int(resize_info.get("px_w", 0))
        th = int(resize_info.get("px_h", 0))
        if tw > 0 and th > 0:
            new_w, new_h = tw, th
        elif tw > 0:
            s = tw / float(w0)
            new_w, new_h = tw, max(1, int(round(h0 * s)))
        elif th > 0:
            s = th / float(h0)
            new_h, new_w = th, max(1, int(round(w0 * s)))
    elif "scale" in resize_info:
        scale = max(1, int(resize_info["scale"]))
        new_w = max(1, int(round(w0 * (scale / 100.0))))
        new_h = max(1, int(round(h0 * (scale / 100.0))))
    else:
        pct_w = int(resize_info.get("width", 100))
        pct_h = int(resize_info.get("height", 100))
        new_w = max(1, int(round(w0 * (pct_w / 100.0))))
        new_h = max(1, int(round(h0 * (pct_h / 100.0))))
    return new_w, new_h


def _apply_resize(img, resize_info, nodata_values):
    h0, w0 = img.shape[:2]
    new_w, new_h = _calc_resize_dims(w0, h0, resize_info)
    
    if new_w == w0 and new_h == h0:
        return img
        
    sw = new_w / float(w0); sh = new_h / float(h0)
    if sw < 1.0 or sh < 1.0:
        interp = cv2.INTER_AREA
    elif max(sw, sh) < 2.0:
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_CUBIC

    # NoData handling
    nd_restore_val = None
    for nv in (nodata_values or []):
        if not isinstance(nv, str):
            try:
                nd_restore_val = float(nv)
                break
            except (ValueError, TypeError):
                pass
                
    if nodata_values and nd_restore_val is not None:
        try:
            numeric_nd_vals = [v for v in nodata_values if not isinstance(v, str)]
            if numeric_nd_vals:
                nd_mask = build_nodata_mask(img, numeric_nd_vals, bgr_input=True)
            else:
                nd_mask = None
                
            if nd_mask is not None and nd_mask.any():
                work = img.astype(np.float32, copy=True)
                if work.ndim == 2:
                    work[nd_mask] = np.nan
                else:
                    for c in range(work.shape[2]):
                        work[..., c][nd_mask] = np.nan
                        
                work = resize_safe(work, new_w, new_h, interp)
                
                nan_mask = np.isnan(work)
                if nan_mask.any():
                    work[nan_mask] = nd_restore_val
                return work
        except Exception:
            pass
            
    return resize_safe(img, new_w, new_h, interp)


def _process_band_expression(image, expr, nodata_values):
    try:
        effective_expr = expr
        # BGR -> RGB remapping for 3-channel
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            effective_expr = re.sub(r'\bb1\b', '__B1_TEMP__', effective_expr)
            effective_expr = re.sub(r'\bb3\b', 'b1', effective_expr)
            effective_expr = re.sub(r'__B1_TEMP__', 'b3', effective_expr)
            
        nodata_mask = None
        if nodata_values:
            nodata_mask = build_nodata_mask(image, nodata_values, bgr_input=True)
            
        res = eval_band_expression(image, effective_expr)
        
        if nodata_mask is not None and isinstance(res, np.ndarray):
            if res.shape[:2] == nodata_mask.shape:
                if res.ndim == image.ndim:
                    if res.ndim == 2:
                         res = np.where(nodata_mask, image.astype(res.dtype), res)
                    else:
                        for c in range(min(res.shape[2], image.shape[2] if image.ndim == 3 else 1)):
                            img_ch = image[..., c] if image.ndim == 3 else image
                            res[..., c] = np.where(nodata_mask, img_ch.astype(res.dtype), res[..., c])
                            
        # Normalize to BGR if 3-ch
        if isinstance(res, np.ndarray) and res.ndim == 3 and res.shape[2] == 3:
             res = res[:, :, ::-1].copy()
             
        return res
    except Exception as e:
        logger.error(f"Error processing band expression '{expr}': {e}")
        return image


def _scale_polygon_points(raw_data, target_h, target_w):
    scaled_list = []
    for points, ref_size in raw_data:
        ref_w = ref_size.get('w', 0) or 0
        ref_h = ref_size.get('h', 0) or 0
        if ref_w > 0 and ref_h > 0 and (ref_w != target_w or ref_h != target_h):
            scale_x = target_w / float(ref_w)
            scale_y = target_h / float(ref_h)
            scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]
            scaled_list.append(scaled_points)
        else:
            scaled_list.append(points)
    return scaled_list


def _build_combined_polygon_mask(shape, polygon_points_list):
    if not polygon_points_list:
        return None
    H, W = shape
    combined_mask = np.zeros((H, W), dtype=bool)
    for points in polygon_points_list:
        if points and len(points) >= 3:
            pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            combined_mask |= (mask > 0)
    return combined_mask if combined_mask.any() else None


def _apply_hist_match(img, hcfg, nodata_values, mask_poly_raw, mask_poly_points, mask_poly_enabled):
    mode = (hcfg.get("mode") or "meanstd").lower()
    if not hcfg or mode in ("none", ""):
        return img
        
    x = img.astype(np.float32, copy=False)
    if x.ndim == 2:
        x = x[..., None]
    C = x.shape[2]
    
    # Build combined mask
    combined_mask = None
    if nodata_values:
        combined_mask = build_nodata_mask(x, nodata_values, bgr_input=True)
        
    if mask_poly_enabled:
        poly_points = _scale_polygon_points(mask_poly_raw, x.shape[0], x.shape[1]) if mask_poly_raw else mask_poly_points
        if poly_points:
            poly_mask = _build_combined_polygon_mask(x.shape[:2], poly_points)
            if poly_mask is not None:
                if combined_mask is None:
                    combined_mask = poly_mask
                else:
                    combined_mask = combined_mask | poly_mask

    def _safe_std(a):
        s = float(np.nanstd(a))
        return s if s > 1e-12 else 1.0

    if mode == "meanstd":
        stats = (hcfg.get("ref_stats") or [])
        for c in range(min(C, len(stats))):
            ch = x[..., c]
            if combined_mask is not None:
                ch_masked = np.where(combined_mask, np.nan, ch)
                mu_t = float(np.nanmean(ch_masked))
                sd_t = _safe_std(ch_masked)
            else:
                mu_t = float(np.nanmean(ch))
                sd_t = _safe_std(ch)
            mu_r = float(stats[c].get("mean", 0.0))
            sd_r = float(stats[c].get("std", 1.0))
            new_vals = (ch - mu_t) * (sd_r / sd_t) + mu_r
            if combined_mask is not None:
                x[..., c] = np.where(combined_mask, ch, new_vals)
            else:
                x[..., c] = new_vals
                
    else: # CDF
        ref = hcfg.get("ref_cdf", {}) or {}
        per = ref.get("per_band") or []
        bins = 2048
        for c in range(min(C, len(per))):
            lut = per[c] or {}
            x_n = np.asarray(lut.get("x") or [0.0, 1.0], dtype=np.float32)
            y_n = np.asarray(lut.get("y") or [0.0, 1.0], dtype=np.float32)
            lo = float(lut.get("lo", 0.0)); hi = float(lut.get("hi", 1.0))
            if hi <= lo: hi = lo + 1.0
            ch = x[..., c]
            
            valid_mask = ~combined_mask if combined_mask is not None else None
            z = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            idx = np.clip((z * (bins - 1)).astype(np.int32), 0, bins - 1)
            
            if valid_mask is not None:
                hist = np.bincount(idx[valid_mask].ravel(), minlength=bins).astype(np.float32)
            else:
                hist = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
                
            cdf_src = np.cumsum(hist)
            total = float(cdf_src[-1]) if cdf_src.size else 1.0
            cdf_src /= max(total, 1.0)
            
            xprime_norm = np.interp(cdf_src, y_n, x_n).astype(np.float32)
            new_vals = xprime_norm[idx] * (hi - lo) + lo
            
            if combined_mask is not None:
                x[..., c] = np.where(combined_mask, ch, new_vals)
            else:
                x[..., c] = new_vals

    return x[..., 0] if img.ndim == 2 else x
