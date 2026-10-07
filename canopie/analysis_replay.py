"""
The .ax replay for ANALYSIS -- what the CSV export, the ML manager (export and
training) and Inspect read. ``ProjectTab._apply_ax_to_raw`` stays the replay for
IMAGE export (thumbnails, project images), whose NoData fill is a file-format
concern.

The returned image is float32, (H, W, C), and:

  * holds the file's bands in FILE order (b1 first, whatever order the loader
    produced) -- consumers never need a channel order again;
  * marks NoData as NaN PER BAND (canopie.nodata: the per-band rule), so
    every consumer derives the same mask with ``~np.isfinite``;
  * appends the derived bands, named in a BandLayout:
    the band expression, the classification label, then the appended bands --
    the order _apply_ax_to_raw uses, so ``b5`` in an appended expression still
    means what it meant;
  * applies registration ONCE, on the raw frame, with the matrix conjugated
    through the .ax geometry (canopie.ax_registration -- the viewer, the
    editor and the image export use the same matrix). NoData is NaN before the
    warp, so a pixel interpolated from NoData is NoData; pixels from outside
    the raster are NoData.

When the image has no NoData at all, the pixel values are exactly
_apply_ax_to_raw's: same geometry (canopie.ax_geometry, proven equal), the
resize and histogram match in the raw dtype, then float32. With NoData, a band
is NaN wherever its footprint touches NoData (the viewer's rule), instead of
the fill value being averaged into its neighbours.
"""
import json
import logging
import os

import numpy as np

from . import ax_geometry as G
from .band_layout import BandLayout
from .nodata import band_mask, expression_source_bands, rules_from_values

__all__ = ["replay_for_analysis", "layout_for_ax", "classify_label_band"]

_HIST_MAX_SAMPLES = 250
_ROT_K = {90: 3, 180: 2, 270: 1}          # np.rot90 k for a clockwise turn


def _as_hwc(raw):
    img = np.asarray(raw)
    if img.ndim == 2:
        return img[..., None]
    if img.ndim == 3:
        d0, d1, d2 = img.shape
        if d0 <= 32 and d1 >= 32 and d2 >= 32 and d0 < min(d1, d2) // 2:
            img = np.moveaxis(img, 0, -1)
        return img
    img = np.squeeze(img)
    if img.ndim == 2:
        return img[..., None]
    return img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])


def _mask_polygon_records(pt, ax, filepath, all_polygons):
    """(points, image_ref_size) of the .ax's enabled mask polygons -- the file on
    disk first, then memory; names match a group key or a label."""
    cfg = ax.get("mask_polygon") or {}
    if not (isinstance(cfg, dict) and cfg.get("enabled", False)):
        return []
    names = list(cfg.get("names") or ([cfg["name"]] if cfg.get("name") else []))
    out = []
    legacy = cfg.get("points") or []
    if legacy and len(legacy) >= 3:
        out.append((legacy, {}))
    if not (names and filepath):
        return out
    pf = getattr(pt, "project_folder", None)
    pdir = os.path.join(pf, "polygons") if pf else None
    base = os.path.splitext(os.path.basename(filepath))[0]
    found = set()
    if pdir and os.path.isdir(pdir):
        for nm in names:
            p = os.path.join(pdir, f"{nm}_{base}_polygons.json")
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    pts = d.get("points") or []
                    if len(pts) >= 3:
                        out.append((pts, d.get("image_ref_size") or {}))
                        found.add(nm)
                except Exception as e:
                    logging.debug("[analysis_replay] mask polygon %s unreadable: %s", p, e)
    polys = all_polygons if all_polygons is not None else getattr(pt, "all_polygons", None)
    if polys and len(found) < len(names):
        nfp = os.path.normcase(os.path.normpath(filepath))
        for group, fmap in polys.items():
            if not isinstance(fmap, dict):
                continue
            for key, rec in fmap.items():
                if not isinstance(rec, dict) or os.path.normcase(os.path.normpath(key)) != nfp:
                    continue
                label = dict.get(rec, "name", group)
                if (group in names or label in names) and group not in found and label not in found:
                    pts = rec.get("points") or []
                    if len(pts) >= 3:
                        out.append((pts, dict.get(rec, "image_ref_size") or {}))
    return out


def _polygon_mask(records, H, W):
    """Pixels inside any mask polygon, in the CURRENT frame (records hold
    fractions of it: points / image_ref_size)."""
    import cv2
    if not records:
        return None
    m = np.zeros((H, W), np.uint8)
    for pts, ref in records:
        try:
            a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            rw = float((ref or {}).get("w") or 0) if isinstance(ref, dict) else 0.0
            rh = float((ref or {}).get("h") or 0) if isinstance(ref, dict) else 0.0
            if rw > 0 and rh > 0:
                a = a * (W / rw, H / rh)
            cv2.fillPoly(m, [np.round(a).astype(np.int32).reshape(-1, 1, 2)], 1)
        except Exception as e:
            logging.debug("[analysis_replay] mask polygon skipped: %s", e)
    return m.astype(bool) if m.any() else None


def _with_nan_where_sources_are(res, stack, sources):
    """A derived band is NoData wherever any band it reads is."""
    bad = ~np.isfinite(res)
    for s in sources:
        if 0 <= s < stack.shape[2]:
            bad |= ~np.isfinite(stack[..., s])
    if bad.any():
        res = res.copy()
        res[bad] = np.nan
    return res


def _eval_expression(pt, stack, expr):
    try:
        res = pt._eval_band_expression(np.asarray(stack, dtype=np.float32), expr)
    except Exception:
        from .utils import eval_band_expression
        res = eval_band_expression(np.asarray(stack, dtype=np.float32), expr)
    if res is None:
        return None
    res = np.asarray(res, dtype=np.float32)
    if res.ndim == 3:
        res = res[..., -1]
    return res


def classify_label_band(pt, stack, bundle):
    """Label band (H, W) float32 for a FILE-ordered (H, W, C) stack: class
    index, NaN where unclassified -- the rows training would have dropped
    (utils.classification_invalid_rows). The ONE classifier: the export, the
    viewer and appended classification bands all use it. `pt` may be None (a
    caller without a tab: the builder does not use it)."""
    return _classify(pt, stack, bundle)


def _classify(pt, stack, bundle):
    """See classify_label_band."""
    import functools
    from .utils import iter_tiled_feature_matrices, classification_invalid_rows
    if pt is None:
        from .project_tab import ProjectTab as _PT
        _make = functools.partial(_PT._make_feature_stack_for_model, None)
        _lock_owner = _PT
    else:
        _make = pt._make_feature_stack_for_model
        _lock_owner = pt
    try:
        from .project_tab import _CLASSIFY_FEATURE_BUDGET_BYTES as budget
    except Exception:
        budget = 512 * 1024 * 1024
    model = bundle["model"]
    feat_names = list(bundle.get("feature_names") or [])
    expressions = bundle.get("expressions", []) or []
    window = int(bundle.get("window_size", 1) or 1)
    base_names = bundle.get("base_feature_names") or feat_names
    labels = list(bundle.get("label_names") or list(getattr(model, "classes_", [])))
    idx_map = {lab: i for i, lab in enumerate(labels)}
    H, W = stack.shape[:2]
    out = np.full((H * W,), np.nan, np.float32)
    builder = functools.partial(_make,
                                normalization_cfg=bundle.get("normalization"),
                                src_channel_order="rgb", nodata_values=None)
    lock = getattr(_lock_owner, "_sklearn_predict_lock", None)
    for r0, r1, X, w in iter_tiled_feature_matrices(stack, feat_names, expressions, window,
                                                    base_names, budget, builder):
        if X is None:
            raise RuntimeError("could not build the model's feature stack")
        bad = classification_invalid_rows(X, r1 - r0, w, window_size=1, nodata_values=None)
        if window > 1:
            # window edges of the WHOLE image, not of the tile
            from .utils import window_edge_rows
            edge = window_edge_rows(H, W, window).reshape(H, W)[r0:r1].reshape(-1)
            bad |= edge
        good = np.flatnonzero(~bad)
        if good.size:
            chunk = 200_000
            preds = []
            for i in range(0, good.size, chunk):
                rows = X[good[i:i + chunk]]
                if lock is not None:
                    with lock:
                        preds.append(model.predict(rows))
                else:
                    preds.append(model.predict(rows))
            p = np.concatenate(preds)
            mapped = np.array([idx_map.get(v, np.nan) for v in p.tolist()], dtype=np.float32)
            out[r0 * w + good] = mapped
    return out.reshape(H, W)


def _band_expression_of(ax):
    be = ax.get("band_expression")
    if isinstance(be, dict):
        return str(be.get("expression") or "").strip()
    return be.strip() if isinstance(be, str) else ""


def _classification_on(ax):
    c = ax.get("classification") or {}
    return (isinstance(c, dict) and str(c.get("mode", "")).lower() == "sklearn"
            and bool(c.get("enabled", False)))


def layout_for_ax(ax, n_file):
    """The band layout replay_for_analysis gives an image with `n_file` file
    bands under this .ax -- names only, no pixels. None when a band depends on
    a model (classification), whose presence is only known at replay time."""
    ax = ax if isinstance(ax, dict) else {}
    layout = BandLayout(n_file)
    expr = _band_expression_of(ax)
    if ax.get("band_enabled", True) and expr:
        layout.add_expression(expr)
    if _classification_on(ax):
        return None
    for info in (ax.get("appended_bands") or []):
        if not isinstance(info, dict):
            continue
        kind = info.get("type", "")
        if kind in ("boolean_expression", "band_expression"):
            e = str(info.get("expression") or "").strip()
            if e:
                layout.add_expression(e, kind="appended")
        elif kind == "classification":
            return None
    return layout


def replay_for_analysis(pt, raw_img, ax, *, filepath=None, src_channel_order=None,
                        all_polygons=None, extra_nodata_values=None):
    """Returns ``(img, layout)``; see the module docstring.

    `extra_nodata_values`: NoData numbers / `bN op x` rules excluded IN
    ADDITION to the image's own (the CSV export dialog's field). They mark
    pixels NaN like the image's own rule, but do not change the histogram
    match's statistics (those stay the image's own, as the viewer has them)."""
    import cv2
    from .utils import to_rgb_band_order

    ax = ax if isinstance(ax, dict) else {}
    if raw_img is None:
        return None, BandLayout(0)
    img = _as_hwc(raw_img)
    img = np.ascontiguousarray(to_rgb_band_order(img, str(src_channel_order or "bgr").lower()))
    H0, W0, n_file = img.shape
    layout = BandLayout(n_file)

    # ---- NoData rule -------------------------------------------------------
    nodata_on = ax.get("nodata_enabled", True)
    if not nodata_on:
        values = []
    elif filepath and hasattr(pt, "effective_nodata_values"):
        values = pt.effective_nodata_values(filepath, ax)
    else:
        values = list(ax.get("nodata_values") or [])
    for v in (extra_nodata_values or []):
        if v not in values:
            values = list(values) + [v]
    rules = rules_from_values(values)

    # ---- registration: once, on the raw frame (canopie.ax_registration) ----
    # NoData of the RAW values first, as NaN, so the warp never blends a fill
    # value into its neighbours; from here on NoData is carried as NaN.
    from .ax_registration import raw_frame_matrix, warp as _reg_warp
    reg_m, reg_persp = raw_frame_matrix(ax, W0, H0)
    if reg_m is not None:
        try:
            f = img.astype(np.float32, copy=True)
            if rules:
                for k in range(n_file):
                    m = band_mask(f[..., k], k, rules)
                    if m.any():
                        f[..., k][m] = np.nan
            img = _reg_warp(f, reg_m, reg_persp)
            del f
            rules = None            # NoData is NaN now; never re-match warped values
        except Exception as e:
            logging.warning("[analysis_replay] registration failed for %s: %s",
                            os.path.basename(filepath or "?"), e)
            reg_m = None

    # ---- crop / rotate (the export's own decisions, via ax_geometry) -------
    ops, (W_out, H_out) = G.frame_plan(ax, W0, H0)
    resize_op = None
    for op in ops:
        if op[0] == "rotate":
            img = np.rot90(img, k=_ROT_K[op[1]])
        elif op[0] == "crop":
            img = img[op[2]:op[4], op[1]:op[3]]
        elif op[0] == "resize":
            resize_op = op
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]

    # ---- per-band NoData of the RAW values (before hist changes them) ------
    masks = None
    if rules or reg_m is not None:
        masks = []
        for k in range(n_file):
            m = band_mask(img[..., k], k, rules)
            masks.append(np.packbits(m) if m.any() else None)
        if not any(m is not None for m in masks):
            masks = None
    has_nan = np.issubdtype(img.dtype, np.floating)

    hm = ax.get("hist_match")
    hist_on = bool(ax.get("hist_enabled", True) and isinstance(hm, dict) and hm)
    shrink = bool(resize_op and (resize_op[3] < w or resize_op[4] < h))
    mask_recs = _mask_polygon_records(pt, ax, filepath, all_polygons)

    def _hist(arr):
        fn = getattr(type(pt), "_apply_hist_match", None)
        if not callable(fn):
            return arr
        hh, ww = arr.shape[:2]
        pts = []
        for p, ref in mask_recs:
            try:
                a = np.asarray(p, dtype=np.float64).reshape(-1, 2)
                rw = float((ref or {}).get("w") or 0); rh = float((ref or {}).get("h") or 0)
                if rw > 0 and rh > 0:
                    a = a * (ww / rw, hh / rh)
                pts.append([tuple(v) for v in a.tolist()])
            except Exception:
                continue
        try:
            from .project_tab import hist_nodata_values
            nd = hist_nodata_values(ax, filepath)
        except Exception:
            nd = list(values)
        nan_m = None
        if np.issubdtype(arr.dtype, np.floating):
            nan_m = ~np.isfinite(arr)
            if not nan_m.any():
                nan_m = None
        try:
            out = fn(arr, {"hist_match": hm}, fast=True, max_samples=_HIST_MAX_SAMPLES,
                     nodata_values=nd, mask_polygon_points=pts,
                     mask_polygon_enabled=bool(mask_recs), src_channel_order="rgb")
        except Exception as e:
            logging.warning("[analysis_replay] histogram match failed: %s", e)
            return arr
        if nan_m is not None:                     # NoData stays NoData
            out = np.asarray(out, dtype=np.float32)
            if out.shape == nan_m.shape:
                out[nan_m] = np.nan
        return out

    if hist_on and not shrink:
        img = _hist(img)

    interp = None
    if resize_op is not None:
        tw, th = int(resize_op[3]), int(resize_op[4])
        interp = (cv2.INTER_AREA if (tw < w or th < h)
                  else (cv2.INTER_LINEAR if max(tw / float(w), th / float(h)) < 2.0 else cv2.INTER_CUBIC))

    def _resize(a):
        if resize_op is None:
            return a
        tw, th = int(resize_op[3]), int(resize_op[4])
        if a.shape[2] <= 4:
            r = cv2.resize(a, (tw, th), interpolation=interp)
            return r[..., None] if r.ndim == 2 else r
        return np.stack([cv2.resize(a[..., k], (tw, th), interpolation=interp)
                         for k in range(a.shape[2])], axis=2)

    if masks is None:
        # exactly _apply_ax_to_raw's arithmetic: resize (and the deferred hist)
        # in the raw dtype, float32 afterwards
        img = _resize(img)
        if hist_on and shrink:
            img = _hist(img)
        img = img.astype(np.float32, copy=False)
    else:
        f = img.astype(np.float32, copy=True)
        for k, packed in enumerate(masks):
            if packed is not None:
                m = np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)
                f[..., k][m] = np.nan
        img = _resize(f)
        del f
        if hist_on and shrink:
            img = np.asarray(_hist(img), dtype=np.float32)
        has_nan = True
    img = np.ascontiguousarray(img, dtype=np.float32)
    H, W = img.shape[:2]

    # ---- mask polygons: excluded from every band --------------------------
    pm = _polygon_mask(mask_recs, H, W)
    if pm is not None:
        img[pm] = np.nan

    # ---- derived bands -----------------------------------------------------
    expr = _band_expression_of(ax)
    if ax.get("band_enabled", True) and expr:
        res = _eval_expression(pt, img, expr)
        if res is not None and res.shape == (H, W):
            res = _with_nan_where_sources_are(res, img, expression_source_bands(expr))
            img = np.concatenate([img, res[..., None]], axis=2)
            layout.add_expression(expr)

    cblock = ax.get("classification") or {}
    if (isinstance(cblock, dict) and str(cblock.get("mode", "")).lower() == "sklearn"
            and cblock.get("enabled", False)):
        bundle = pt._get_sklearn_bundle() if hasattr(pt, "_get_sklearn_bundle") else None
        if isinstance(bundle, dict) and "model" in bundle:
            try:
                lab = _classify(pt, img, bundle)
                img = np.concatenate([img, lab[..., None]], axis=2)
                layout.add_derived("Label", "label", range(n_file))
            except Exception as e:
                logging.warning("[analysis_replay] classification skipped for %s: %s",
                                os.path.basename(filepath or "?"), e)
        else:
            logging.warning("[analysis_replay] %s: classification is on but no model is loaded",
                            os.path.basename(filepath or "?"))

    for info in (ax.get("appended_bands") or []):
        if not isinstance(info, dict):
            continue
        kind = info.get("type", "")
        try:
            if kind in ("boolean_expression", "band_expression"):
                e = str(info.get("expression") or "").strip()
                if not e:
                    continue
                res = _eval_expression(pt, img, e)
                if res is None or res.shape != (H, W):
                    continue
                res = _with_nan_where_sources_are(res, img, expression_source_bands(e))
                img = np.concatenate([img, res[..., None]], axis=2)
                layout.add_expression(e, kind="appended")
            elif kind == "classification":
                bundle = pt._get_sklearn_bundle() if hasattr(pt, "_get_sklearn_bundle") else None
                if not (isinstance(bundle, dict) and "model" in bundle):
                    if hasattr(pt, "_record_appended_band_error"):
                        pt._record_appended_band_error(filepath, "classification", info.get("index", 0),
                                                       "no sklearn model loaded in this session")
                    continue
                lab = _classify(pt, img, bundle)
                img = np.concatenate([img, lab[..., None]], axis=2)
                layout.add_derived(info.get("model_name") or "classification", "appended", range(n_file))
        except Exception as e:
            logging.warning("[analysis_replay] appended %s band skipped: %s", kind, e)
            if hasattr(pt, "_record_appended_band_error"):
                pt._record_appended_band_error(filepath, kind, info.get("index", 0), e)

    return np.ascontiguousarray(img, dtype=np.float32), layout
