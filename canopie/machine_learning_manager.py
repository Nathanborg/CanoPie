"""Auto-generated module extracted from original CanoPie code."""


import sys
import os
import re
import json
import math
import time
import glob
import csv
import shutil
import tempfile
import subprocess
import logging
import webbrowser
import threading
import pickle
import concurrent.futures
import copy

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from functools import partial

import numpy as np
import cv2
import exifread
import folium
from shapely import geometry
from shapely.geometry import MultiPolygon
from geopy.distance import geodesic
from scipy.spatial import KDTree

# --- Qt (PyQt5)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, pyqtSlot, QSettings, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QDialog, QFileDialog, QMessageBox,
    QToolBar, QAction, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit, QSlider,
    QLabel, QPushButton, QListWidget, QInputDialog,
    QSizePolicy, QStyle, QShortcut, QAbstractSlider
)

import sip

# Prevent logging from writing to files by removing filename and file handlers
_original_basicConfig = logging.basicConfig

def _no_file_basicConfig(*args, **kwargs):
    kwargs.pop('filename', None)
    handlers = kwargs.get('handlers')
    if handlers:
        # filter out FileHandler instances
        kwargs['handlers'] = [h for h in handlers if not isinstance(h, logging.FileHandler)]
    return _original_basicConfig(*args, **kwargs)
logging.basicConfig = _no_file_basicConfig

from .utils import *
from .loaders import ImageProcessor, ImageLoaderWorker
from .image_data import ImageData

class MachineLearningManager(QtWidgets.QDialog):
    """
    Viewer-independent CSV/mask exporter:
      • Re-applies .ax (crop → resize → band_expression) on RAW every time.
      • No histogram / CLAHE for export (display-only).
      • Pixel values are the real magnitudes (no 0–255 re-scaling).
      • Coordinates map via ProjectTab.scene_to_image_coords if viewer exists;
        else uses stored pixmap_size; else assumed already in image coords.
    """
    def __init__(self, parent=None):
        super(MachineLearningManager, self).__init__(parent)
        self.setWindowTitle("MachineLearning Manager")
        self.resize(600, 400)

        self.project_folder = getattr(parent, "project_folder", None)
        self.parent_tab = parent

        # Layout + toolbar
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.toolbar = QToolBar("MachineLearning Manager Toolbar")
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.main_layout.addWidget(self.toolbar)

        self.exportCsvAct = QAction("Export CSV Data", self)
        self.exportCsvAct.setIcon(QIcon.fromTheme("document-save"))
        self.exportCsvAct.setStatusTip("Export polygon data to CSV for ML training")
        self.exportCsvAct.triggered.connect(self.export_csv_data)

        self.thumbnailsAct = QAction("Generate Thumbnails", self)
        self.thumbnailsAct.setIcon(QIcon.fromTheme("insert-image"))
        self.thumbnailsAct.setStatusTip("Generate cropped thumbnails of each polygon")
        self.thumbnailsAct.triggered.connect(self.generate_thumbnails)

        self.segmentationAct = QAction("Generate Segmentation Masks", self)
        self.segmentationAct.setIcon(QIcon.fromTheme("view-list-icons"))
        self.segmentationAct.setStatusTip("Generate grayscale segmentation masks for polygons")
        self.segmentationAct.triggered.connect(self.generate_segmentation_images)

        self.closeAct = QAction("Close", self)
        self.closeAct.setIcon(QIcon.fromTheme("window-close"))
        self.closeAct.setStatusTip("Close this dialog")
        self.closeAct.triggered.connect(self.close)

        self.toolbar.addAction(self.exportCsvAct)
        self.toolbar.addAction(self.thumbnailsAct)
        self.toolbar.addAction(self.segmentationAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.closeAct)

        self.info_label = QLabel("Select polygon groups from the list:")
        self.main_layout.addWidget(self.info_label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.main_layout.addWidget(self.list_widget)

        self.populate_polygon_groups()


    def _ensure_hwc(self, arr):
        """Return 2D or HxWxC. Accepts (bands,H,W) and (pages,H,W[,samples]) stacks."""
        import numpy as np
        if arr is None:
            return None
        a = np.asarray(arr)

        if a.ndim == 2:
            return a

        if a.ndim == 3:
            # Treat axis 0 as bands if it's plausibly the smallest dimension (typical CHW)
            B, H, W = a.shape
            if B <= min(H, W) and H > 32 and W > 32:
                return np.moveaxis(a, 0, 2).copy()         # (B,H,W) -> (H,W,B)
            return a                                        # already (H,W,C)

        if a.ndim == 4:
            # (pages, H, W, samples) -> flatten pages into channels: (H,W, P*samples)
            P, H, W, S = a.shape
            a = a.reshape(P, H, W, S)
            return np.concatenate([a[i] for i in range(P)], axis=2).copy()

        return a


    def _ax_candidates(self, filepath):
        base = os.path.splitext(os.path.basename(filepath))[0] + ".ax"
        cand = []
        if self.project_folder:
            cand.append(os.path.join(self.project_folder, base))
        cand.append(os.path.join(os.path.dirname(filepath), base))
        return cand

    def _load_ax_mods(self, filepath):
        """Load first .ax found (project folder first, then alongside image)."""
        for mf in self._ax_candidates(filepath):
            try:
                if os.path.exists(mf):
                    with open(mf, "r", encoding="utf-8") as f:
                        return json.load(f) or {}
            except Exception as e:
                logging.error(f"Failed to read AX {mf}: {e}")
        return {}


    def _eval_band_expression(self, img_float, expr):
        """
        Tolerant eval for export:
          - If expr references unknown names or bands > available, return None (caller skips appending).
          - Silences numpy warnings; returns float32 2D when ok.
        """
        import numpy as np, logging
        if not expr:
            return None

        x = np.nan_to_num(img_float.astype(np.float32, copy=False))
        C = 1 if x.ndim == 2 else (x.shape[2] if x.ndim == 3 else 0)
        if C == 0:
            return None

        # Build mapping
        mapping = {'b1': x} if x.ndim == 2 else {f"b{i+1}": x[:, :, i] for i in range(C)}

        code = compile(expr, "<expr>", "eval")
        for name in code.co_names:
            if name not in mapping:
                logging.warning("Illegal name '%s' in band expr '%s' (export); skipping index.", name, expr)
                return None

        # Quick band-number guard (e.g., asks for b4 but only 1 band)
        import re
        req = sorted({int(b) for b in re.findall(r'b(\d+)', expr)})
        if any(b > C for b in req):
            logging.warning("Expr '%s' requests b%d but only %d band(s) available; skipping index.",
                            expr, max(req), C)
            return None

        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            res = eval(code, {"__builtins__": {}}, mapping)

        if not isinstance(res, np.ndarray):
            res = np.full(x.shape[:2], float(res), dtype=np.float32)
        else:
            if res.ndim == 3:
                res = np.mean(res.astype(np.float32, copy=False), axis=2)
            res = np.nan_to_num(res.astype(np.float32, copy=False),
                                nan=0.0, posinf=0.0, neginf=0.0)
        return res

    def _apply_ax_to_raw(self, raw_img, ax):
        """
        Order-agnostic, stack-safe replay of scientific steps:

          • Honors optional ax["op_order"] = ["rotate","crop","resize","band_expression"] (any permutation)
          • If order deviates from the editor's typical "after-rotate" basis, the crop rect
            is remapped to the correct basis so extracted coordinates remain consistent.
          • For >4ch TIFF stacks, still works via tifffile path + ensure HWC before cv2 ops.

        Returns (float32 image in HxWxC, C)
        """
        if raw_img is None:
            return None, 0

        img = self._ensure_hwc(raw_img)

        # --- Gather AX parameters ----------------------------------------------------
        try:
            rot = int(ax.get("rotate", 0)) % 360
        except Exception:
            rot = 0

        crop_rect = ax.get("crop_rect") or None
        crop_ref = ax.get("crop_rect_ref_size") or None
        resize = ax.get("resize") or None
        expr   = (ax.get("band_expression") or "").strip()

        op_order = ax.get("op_order")
        if not (isinstance(op_order, (list, tuple)) and all(isinstance(s, str) for s in op_order)):
            op_order = ["rotate", "crop", "resize", "band_expression"]


        raw_h, raw_w = img.shape[:2]

        crop_basis = None
        if crop_rect:
            crop_basis = _infer_crop_basis(ax, raw_w, raw_h, rot)  # "after_rotate" or "pre_rotate"

        # --- Execute ops in declared order -------------------------------------------
        # Helper lambdas keep code readable
        def _do_rotate():
            nonlocal img
            if rot in (90, 180, 270):
                try:
                    if rot == 90:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    else:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                except Exception as e:
                    logging.warning(f"Rotation failed ({rot} deg): {e}")

        def _do_crop():
            nonlocal img
            if not crop_rect:
                return
            Hc, Wc = img.shape[:2]

            # Figure out which frame the saved rect lives in (pre- or post-rotate of RAW)
            # and convert it to the CURRENT pre-crop frame.
            rect_to_apply = dict(crop_rect)

            # Saved ref dims for the rectangle’s frame (if provided)
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w_saved, ref_h_saved = int(crop_ref.get("w", raw_w)), int(crop_ref.get("h", raw_h))
            else:
                # infer from basis
                if crop_basis == "after_rotate":
                    ref_w_saved, ref_h_saved = _dims_after_rot(raw_w, raw_h, rot)
                else:
                    ref_w_saved, ref_h_saved = (raw_w, raw_h)

            # Determine whether rotate has ALREADY been applied in this order:
            # i.e., was "rotate" listed before "crop"?
            rotate_applied = False
            if "rotate" in op_order and "crop" in op_order:
                rotate_applied = (op_order.index("rotate") < op_order.index("crop"))

            # If the saved rect is AFTER-ROTATE but rotate is NOT applied yet,
            if crop_basis == "after_rotate" and not rotate_applied and rot in (90, 180, 270):
                rect_to_apply = _rect_after_rot(rect_to_apply, ref_w_saved, ref_h_saved, (360 - rot) % 360)
                ref_w_use, ref_h_use = (raw_w, raw_h)
            # If the saved rect is PRE-ROTATE but rotate IS already applied,
            # rotate the rect into the rotated basis.
            elif crop_basis == "pre_rotate" and rotate_applied and rot in (90, 180, 270):
                rect_to_apply = _rect_after_rot(rect_to_apply, raw_w, raw_h, rot)
                ref_w_use, ref_h_use = _dims_after_rot(raw_w, raw_h, rot)
            else:
                ref_w_use, ref_h_use = (ref_w_saved, ref_h_saved)

            # Finally, scale the rect from its saved reference size to the CURRENT image size
            rect_scaled = _scale_rect(rect_to_apply, ref_w_use, ref_h_use, Wc, Hc)

            x0 = rect_scaled["x"]
            y0 = rect_scaled["y"]
            w  = rect_scaled["width"]
            h  = rect_scaled["height"]
            if w > 0 and h > 0:
                img = img[y0:y0 + h, x0:x0 + w]
            else:
                logging.warning("Crop rect empty/out of bounds after remapping; skipping crop.")

        def _do_resize():
            nonlocal img
            if not resize:
                return
            h0, w0 = img.shape[:2]
            if "scale" in resize:
                s = float(resize["scale"]) / 100.0
                new_w = max(1, int(round(w0 * s)))
                new_h = max(1, int(round(h0 * s)))
            else:
                pw = float(resize.get("width", 100)) / 100.0
                ph = float(resize.get("height", 100)) / 100.0
                new_w = max(1, int(round(w0 * pw)))
                new_h = max(1, int(round(h0 * ph)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        def _do_band_expr():
            nonlocal img
            if not expr:
                return
            # Always compute indices on float32 and append as a new band
            x = img.astype(np.float32, copy=False)
            idx = self._eval_band_expression(x, expr)
            if idx is not None:
                img = np.dstack([x, idx.astype(np.float32, copy=False)])
            else:
                img = x

        # Dispatch
        ops_map = {
            "rotate": _do_rotate,
            "crop": _do_crop,
            "resize": _do_resize,
            "band_expression": _do_band_expr,
        }
        for op in op_order:
            fn = ops_map.get(op.strip().lower())
            if fn:
                fn()

        # Ensure float32 output (scientific values) and report channel count
        img = img.astype(np.float32, copy=False)
        C = img.shape[2] if img.ndim == 3 else 1
        return img, C


    def _load_raw_image(self, filepath):
        """
        Load raw image with full band support (stack-safe):
        - For .tif/.tiff: try tifffile first (full stack), then ImageData, then Pillow page stack, last OpenCV.
        - For everything else: try ImageData, then OpenCV.
        Always return 2D or HxWxC (channels last).
        """
        import os, logging, numpy as np

        def _count_channels(a):
            if a is None:
                return 0
            a = np.asarray(a)
            if a.ndim == 2:
                return 1
            if a.ndim == 3:
                return a.shape[2]        # HWC
            if a.ndim == 4:
                return a.shape[0] * a.shape[3]  # pages*samples
            return 1

        ext = os.path.splitext(filepath)[1].lower()

        # --- TIFF path: tifffile FIRST (full stacks), then ImageData, then Pillow, then OpenCV
        if ext in (".tif", ".tiff"):
            best = None; best_c = 0

            # 1) tifffile
            try:
                import tifffile as tiff
                arr_tt = tiff.imread(filepath)
                arr_tt = self._ensure_hwc(arr_tt)
                c_tt = _count_channels(arr_tt)
                if c_tt > best_c:
                    best, best_c = arr_tt, c_tt
            except Exception as e:
                logging.warning(f"tifffile failed on {filepath}: {e}")

            # 2) ImageData 
            try:
                mode = getattr(self.parent_tab, "mode", None)
                data = ImageData(filepath, mode=mode) if mode else ImageData(filepath)
                arr_id = self._ensure_hwc(data.image)
                c_id = _count_channels(arr_id)
                if c_id > best_c:
                    best, best_c = arr_id, c_id
            except Exception:
                pass

            # 3) Pillow page stack
            if best is None or best_c <= 1:
                try:
                    from PIL import Image
                    frames = []
                    with Image.open(filepath) as im:
                        while True:
                            frames.append(np.array(im))
                            try:
                                im.seek(im.tell() + 1)
                            except EOFError:
                                break
                    if frames:
                        arr_pil = np.stack(frames, axis=0)  # (pages,H,W) or (pages,H,W,C)
                        arr_pil = self._ensure_hwc(arr_pil)
                        c_pil = _count_channels(arr_pil)
                        if c_pil > best_c:
                            best, best_c = arr_pil, c_pil
                except Exception as ee:
                    logging.warning(f"Pillow fallback failed on {filepath}: {ee}")

            # 4) OpenCV (last-resort; won’t read >4ch TIFFs)
            if best is None:
                import cv2
                arr_cv = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                best = self._ensure_hwc(arr_cv)

            try:
                logging.info(f"[ML TIFF] {os.path.basename(filepath)} -> shape={np.asarray(best).shape}")
            except Exception:
                pass
            return best

        # --- Non-TIFF path: ImageData then OpenCV
        try:
            mode = getattr(self.parent_tab, "mode", None)
            data = ImageData(filepath, mode=mode) if mode else ImageData(filepath)
            return self._ensure_hwc(data.image)
        except Exception:
            pass

        import cv2
        arr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        return self._ensure_hwc(arr)


    def _get_export_image(self, filepath):
        """
        Deterministically produce the export image:
        RAW + .ax scientific steps (crop/resize/expression), no histogram/CLAHE.
        """
        raw = self._load_raw_image(filepath)
        ax = self._load_ax_mods(filepath)
        img, C = self._apply_ax_to_raw(raw, ax)
        return img, C


    def _channels_in_export_order(self, img):
        """
        Return a list of 2D float32 arrays (each C-contiguous) in export order.
        If exactly 3 channels, map OpenCV BGR->RGB; if >3, keep natural order (b1..bC).
        Ensures dtype=float32 and C-contiguous memory for downstream OpenCV ops.
        """
        import numpy as np

        if img.ndim == 3:
            C = img.shape[2]
            if C == 3:
                chans = [img[:, :, 2], img[:, :, 1], img[:, :, 0]]  # R,G,B from B,G,R
            else:
                chans = [img[:, :, i] for i in range(C)]            # multispectral: b1..bC
        else:
            chans = [img]  # single-band

        out = []
        for ch in chans:
            a = ch
            # dtype -> float32
            if a.dtype != np.float32:
                a = a.astype(np.float32, copy=False)
            # guarantee C-contiguous (NumPy 2.x: avoid order='C',copy=False combo)
            if not a.flags.c_contiguous:
                a = np.ascontiguousarray(a)
            out.append(a)
        return out


    def _map_points_scene_to_image(self, filepath, points, img_shape, polygon_data=None):
        """
        Map scene coords -> image pixel coords without requiring the viewer.
        Priority:
          1) If viewer exists: use ProjectTab.scene_to_image_coords(viewer, ...)
          2) Else if polygon_data has 'pixmap_size': scale by (img_w/pixmap_w, img_h/pixmap_h)
          3) Else: assume points are already image coords.
        """
        # 1) Try live viewer mapping
        viewer = getattr(self.parent_tab, "get_viewer_by_filepath", lambda _p: None)(filepath)
        if viewer is not None:
            out = []
            for (x, y) in points:
                q = self.parent_tab.scene_to_image_coords(viewer, QtCore.QPointF(x, y))
                out.append((int(round(q.x())), int(round(q.y()))))
            return out

        # 2) Offline mapping using stored pixmap size
        if polygon_data:
            pm = polygon_data.get("pixmap_size") or polygon_data.get("pixmap")  # support either key
            if isinstance(pm, (list, tuple)) and len(pm) == 2:
                pix_w, pix_h = float(pm[0]), float(pm[1])
                img_h, img_w = img_shape[0], img_shape[1]
                sx = (img_w / pix_w) if pix_w > 0 else 1.0
                sy = (img_h / pix_h) if pix_h > 0 else 1.0
                return [(int(round(x * sx)), int(round(y * sy))) for (x, y) in points]

        # 3) Fallback: treat as already in image coords
        return [(int(round(x)), int(round(y))) for (x, y) in points]


    def populate_polygon_groups(self):
        self.list_widget.clear()
        if not hasattr(self.parent(), 'all_polygons'):
            logging.warning("Parent does not have 'all_polygons'. Cannot populate polygon groups.")
            return
        for group_name in sorted(self.parent().all_polygons.keys()):
            self.list_widget.addItem(QtWidgets.QListWidgetItem(group_name))

    def get_selected_groups(self):
        return [item.text() for item in self.list_widget.selectedItems()]
 
    def _snapshot_polygons_by_group(self, groups):
        """
        Take a deep copy of parent's all_polygons for just the requested groups.
        This makes ML exports immune to any UI clears/deletes that happen later.
        """
        src = getattr(self.parent_tab, "all_polygons", {}) or {}
        snap = {}
        for g in groups:
            if g in src:
                try:
                    snap[g] = copy.deepcopy(src[g])
                except Exception as e:
                    logging.warning(f"Snapshot failed for group '{g}': {e}; using shallow copy.")
                    snap[g] = dict(src[g])
            else:
                logging.debug(f"Group '{g}' not present at snapshot time.")
        return snap
        
    def export_csv_data(self):
        """
        Export REAL pixel values after RAW + .ax (crop/resize/expr) re-application.
        Snapshot polygons up front to avoid races with UI deletes/clears.
        """
        try:
            # ---- 1) Pick groups ----------------------------------------------------
            selected_groups = self.get_selected_groups()
            if not selected_groups:
                QtWidgets.QMessageBox.warning(self, "No Groups Selected",
                                              "Please select one or more polygon groups.")
                return

            # Take a stable snapshot BEFORE any dialogs or long work
            polygons_by_group = self._snapshot_polygons_by_group(selected_groups)

            # ---- 2) Extraction mode ------------------------------------------------
            items = ["Average Pixel Value", "All Pixel Values", "All Pixels with Surrounding Window"]
            extraction_mode, ok = QInputDialog.getItem(
                self, "Pixel Extraction Mode", "Select the pixel extraction mode:",
                items, 0, False
            )
            if not ok:
                return

            window_size = None
            if extraction_mode == "All Pixels with Surrounding Window":
                w_items = ["3x3", "5x5"]
                w_choice, ok = QInputDialog.getItem(
                    self, "Window Size Selection", "Select the window size:",
                    w_items, 0, False
                )
                if not ok:
                    return
                window_size = int(w_choice.split("x")[0])

            # ---- 3) Destination file ----------------------------------------------
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV Data", "", "CSV Files (*.csv)"
            )
            if not save_path:
                return

            # ---- 4) Collect all (group, filepath) entries from the SNAPSHOT -------
            all_entries = []
            for g in selected_groups:
                for fp in polygons_by_group.get(g, {}).keys():
                    all_entries.append((g, fp))
            if not all_entries:
                QtWidgets.QMessageBox.warning(self, "No Files",
                                              "No files found in the selected groups.")
                return

            import numpy as np, csv, cv2, os, logging

            # ---- helpers -----------------------------------------------------------
            def make_hashable(val):
                if isinstance(val, list):
                    return tuple(make_hashable(x) for x in val)
                if isinstance(val, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
                return val

            def _in_bounds(W, H, xi, yi):
                return (0 <= xi < W) and (0 <= yi < H)

            def _align_point_values(any_rgb, max_extras, max_small, chans, xi, yi):
                vals = []
                if any_rgb:
                    # RGB first
                    rgb = [np.nan, np.nan, np.nan]
                    if len(chans) >= 1 and _in_bounds(chans[0].shape[1], chans[0].shape[0], xi, yi):
                        rgb[0] = float(chans[0][yi, xi])
                    if len(chans) >= 2 and _in_bounds(chans[1].shape[1], chans[1].shape[0], xi, yi):
                        rgb[1] = float(chans[1][yi, xi])
                    if len(chans) >= 3 and _in_bounds(chans[2].shape[1], chans[2].shape[0], xi, yi):
                        rgb[2] = float(chans[2][yi, xi])
                    vals.extend(rgb)
                    # extra bands (4..N)
                    actual_extras = max(0, len(chans) - 3)
                    for i in range(max_extras):
                        idx = 3 + i
                        if i < actual_extras:
                            ch = chans[idx]
                            vals.append(float(ch[yi, xi]) if _in_bounds(ch.shape[1], ch.shape[0], xi, yi) else np.nan)
                        else:
                            vals.append(np.nan)
                else:
                    # 1..max_small channels
                    for i in range(max_small):
                        if i < len(chans):
                            ch = chans[i]
                            vals.append(float(ch[yi, xi]) if _in_bounds(ch.shape[1], ch.shape[0], xi, yi) else np.nan)
                        else:
                            vals.append(np.nan)
                return vals

            def _align_window_values(any_rgb, max_extras, max_small, chans, xi, yi, k):
                pad = k // 2
                padded = [cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT) for ch in chans]

                def wflat(ch):
                    cy = yi + pad
                    cx = xi + pad
                    win = ch[cy - pad: cy + pad + 1, cx - pad: cx + pad + 1]
                    return win.reshape(-1).tolist()

                out = []
                if any_rgb:
                    # 3 RGB windows
                    for i in range(3):
                        out += wflat(padded[i]) if i < len(padded) else [np.nan] * (k * k)
                    # extra bands
                    actual_extras = max(0, len(padded) - 3)
                    used = min(actual_extras, max_extras)
                    for i in range(used):
                        out += wflat(padded[3 + i])
                    if max_extras > used:
                        out += [np.nan] * ((max_extras - used) * k * k)
                else:
                    for i in range(max_small):
                        out += wflat(padded[i]) if i < len(padded) else [np.nan] * (k * k)
                return out

            def _poly_mask(pts, H, W):
                mask = np.zeros((H, W), dtype=np.uint8)
                if len(pts) == 1:
                    xi, yi = pts[0]
                    if 0 <= yi < H and 0 <= xi < W:
                        mask[yi, xi] = 255
                elif len(pts) >= 3:
                    arr = np.array([pts], dtype=np.int32)
                    cv2.fillPoly(mask, arr, 255)
                return mask

            # Detect export channel layout across files
            any_rgb = False
            max_extras = 0
            max_small = 0

            for _g, fp in all_entries:
                img, C = self._get_export_image(fp)
                if img is None or C <= 0:
                    continue
                if C >= 3:
                    any_rgb = True
                    max_extras = max(max_extras, C - 3)
                else:
                    max_small = max(max_small, C)

            # Headers (kept identical to your original logic)
            if extraction_mode == "Average Pixel Value":
                prefix = ["group_name", "image_file"]
            else:
                prefix = ["group_name", "image_file", "x", "y"]

            if any_rgb:
                channel_names = ["R", "G", "B"] + [f"band_{i}" for i in range(4, 4 + max_extras)]
            else:
                channel_names = [f"channel_{i}" for i in range(1, max_small + 1)]

            if extraction_mode == "All Pixels with Surrounding Window":
                header = list(prefix)
                for nm in channel_names:
                    for w in range(window_size * window_size):
                        header.append(f"{nm}_w{w}")
            elif extraction_mode == "Average Pixel Value":
                header = list(prefix) + [f"mean_{nm}" for nm in channel_names]
            else:
                header = list(prefix) + channel_names

            # ---- 6) Extract rows ----------------------------------------------------
            all_rows = []
            progress = QtWidgets.QProgressDialog("Extracting pixels…", "Cancel", 0, len(all_entries), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)

            for idx, (group_name, filepath) in enumerate(all_entries, start=1):
                if progress.wasCanceled():
                    break
                progress.setValue(idx)

                # READ FROM SNAPSHOT, not from live state
                polydata = polygons_by_group.get(group_name, {}).get(filepath, {})
                raw_pts = polydata.get("points", [])
                poly_type = str(polydata.get("type", "")).lower()

                img, _C = self._get_export_image(filepath)
                if img is None:
                    logging.warning(f"Could not load/export image for {filepath}")
                    continue

                H, W = img.shape[:2]
                chans = self._channels_in_export_order(img)

                # ---------- POINT SHAPES: treat each point independently ----------
                if poly_type == "point":
                    # Coerce raw points into a flat list of (x, y)
                    pts_raw = []
                    try:
                        if isinstance(raw_pts, dict) and 'x' in raw_pts and 'y' in raw_pts:
                            pts_raw = [(raw_pts['x'], raw_pts['y'])]
                        elif isinstance(raw_pts, (list, tuple)):
                            if len(raw_pts) > 0 and isinstance(raw_pts[0], dict) and 'x' in raw_pts[0] and 'y' in raw_pts[0]:
                                pts_raw = [(p['x'], p['y']) for p in raw_pts]
                            elif len(raw_pts) > 0 and isinstance(raw_pts[0], (list, tuple)) and len(raw_pts[0]) == 2 \
                                 and not isinstance(raw_pts[0][0], (list, tuple)):
                                pts_raw = [(p[0], p[1]) for p in raw_pts]
                            else:
                                # Fallback: parse, then treat each pair as an individual point
                                parsed = self.parse_polygon_points(raw_pts)
                                if parsed:
                                    pts_raw = [(x, y) for (x, y) in parsed]
                    except Exception as e:
                        logging.warning(f"Could not parse point list for {filepath}: {e}")
                        pts_raw = []

                    if not pts_raw:
                        logging.warning(f"No valid points found for {filepath}")
                        continue

                    pts_img_list = self._map_points_scene_to_image(filepath, pts_raw, img.shape, polygon_data=polydata)

                    for (xi, yi) in pts_img_list:
                        if not _in_bounds(W, H, xi, yi):
                            continue

                        if extraction_mode == "Average Pixel Value":
                            # one row per point with per-channel value at that pixel
                            if any_rgb:
                                vals = []
                                for i in range(3):
                                    vals.append(float(chans[i][yi, xi]) if i < len(chans) else np.nan)
                                for i in range(max_extras):
                                    b = 3 + i
                                    vals.append(float(chans[b][yi, xi]) if b < len(chans) else np.nan)
                            else:
                                vals = [float(chans[i][yi, xi]) if i < len(chans) else np.nan
                                        for i in range(max_small)]
                            row = [group_name, filepath] + vals
                            all_rows.append(dict(zip(header, row)))

                        elif extraction_mode == "All Pixel Values":
                            vals = _align_point_values(any_rgb, max_extras, max_small, chans, xi, yi)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))

                        else:  # "All Pixels with Surrounding Window"
                            k = window_size
                            vals = _align_window_values(any_rgb, max_extras, max_small, chans, xi, yi, k)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))

                    # Done with this file's point-shape entry
                    continue

                # ---------- POLYGONS (original behavior) ----------
                polys = self._normalize_to_polygons(raw_pts)
                if not polys:
                    logging.warning(f"Unable to parse points/polygons for file {filepath}")
                    continue

                for poly in polys:
                    pts_img = self._map_points_scene_to_image(filepath, poly, img.shape, polygon_data=polydata)
                    if not pts_img:
                        continue

                    if extraction_mode == "Average Pixel Value":
                        if len(pts_img) == 1:
                            xi, yi = pts_img[0]
                            if not _in_bounds(W, H, xi, yi):
                                continue
                            if any_rgb:
                                vals = []
                                for i in range(3):
                                    vals.append(float(chans[i][yi, xi]) if i < len(chans) else np.nan)
                                for i in range(max_extras):
                                    b = 3 + i
                                    vals.append(float(chans[b][yi, xi]) if b < len(chans) else np.nan)
                            else:
                                vals = [float(chans[i][yi, xi]) if i < len(chans) else np.nan
                                        for i in range(max_small)]
                            row = [group_name, filepath] + vals
                            all_rows.append(dict(zip(header, row)))
                        else:
                            mask = _poly_mask(pts_img, H, W)
                            means = []
                            if any_rgb:
                                for i in range(3):
                                    vals = chans[i][mask == 255] if i < len(chans) else np.array([])
                                    means.append(float(vals.mean()) if vals.size else np.nan)
                                for i in range(max_extras):
                                    b = 3 + i
                                    vals = chans[b][mask == 255] if b < len(chans) else np.array([])
                                    means.append(float(vals.mean()) if vals.size else np.nan)
                            else:
                                for i in range(max_small):
                                    vals = chans[i][mask == 255] if i < len(chans) else np.array([])
                                    means.append(float(vals.mean()) if vals.size else np.nan)
                            row = [group_name, filepath] + means
                            all_rows.append(dict(zip(header, row)))

                    elif extraction_mode == "All Pixel Values":
                        if len(pts_img) == 1:
                            xi, yi = pts_img[0]
                            if not _in_bounds(W, H, xi, yi):
                                continue
                            vals = _align_point_values(any_rgb, max_extras, max_small, chans, xi, yi)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))
                        else:
                            mask = _poly_mask(pts_img, H, W)
                            ys, xs = np.where(mask == 255)
                            for yi, xi in zip(ys, xs):
                                vals = _align_point_values(any_rgb, max_extras, max_small, chans, xi, yi)
                                row = [group_name, filepath, xi, yi] + vals
                                all_rows.append(dict(zip(header, row)))

                    else:  # window
                        k = window_size
                        if len(pts_img) == 1:
                            xi, yi = pts_img[0]
                            if not _in_bounds(W, H, xi, yi):
                                continue
                            vals = _align_window_values(any_rgb, max_extras, max_small, chans, xi, yi, k)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))
                        else:
                            mask = _poly_mask(pts_img, H, W)
                            ys, xs = np.where(mask == 255)
                            for yi, xi in zip(ys, xs):
                                vals = _align_window_values(any_rgb, max_extras, max_small, chans, xi, yi, k)
                                row = [group_name, filepath, xi, yi] + vals
                                all_rows.append(dict(zip(header, row)))

            progress.setValue(len(all_entries))

            # ---- 7) Dedupe + write --------------------------------------------------
            unique_rows, seen = [], set()
            for row in all_rows:
                key = tuple(sorted((k, make_hashable(v)) for k, v in row.items()))
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(row)

            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(unique_rows)

            QtWidgets.QMessageBox.information(self, "Export Complete", f"CSV exported to:\n{save_path}")

        except Exception as e:
            import traceback, logging
            logging.exception("CSV export failed")
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


    def _image_type_for(self, filepath, img):
        """
        Try to match ProjectTab's is_rgb / is_thermal flags when can,
        else infer from channel count.
        """
        img_type = "multispectral"
        try:
            viewer = getattr(self.parent_tab, "get_viewer_by_filepath", lambda _p: None)(filepath)
            if viewer and getattr(viewer, "image_data", None):
                if getattr(viewer.image_data, "is_thermal", False):
                    return "thermal"
                if getattr(viewer.image_data, "is_rgb", False):
                    return "rgb"
        except Exception:
            pass
        # Heuristic fallback
        if img is not None and img.ndim == 3 and img.shape[2] >= 3:
            img_type = "rgb"
        return img_type


    def _as_8bit_bgr(self, arr):
        """
        Convert any 2D/3D array (possibly uint16/float) to an 8-bit BGR image safe for JPEG.
        """
        import numpy as np, cv2
        if arr is None:
            return None
        v = arr
        if v.dtype == np.uint8:
            # If 2D, promote to BGR for drawing
            return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) if v.ndim == 2 else v

        if v.ndim == 2:
            v = cv2.normalize(v.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        else:
            chans = []
            for c in range(v.shape[2]):
                ch = v[:, :, c]
                ch8 = cv2.normalize(ch.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                chans.append(ch8)
            # Assume v is in “export order”: R,G,B,(extras...) -> convert to BGR for OpenCV drawing
            if len(chans) >= 3:
                vis = cv2.merge([chans[2], chans[1], chans[0]])  # B,G,R
            else:
                vis = cv2.merge(chans)
            # If <3 channels, OpenCV still handles 1/2 channels poorly for color; ensure 3 ch
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            elif vis.shape[2] == 1:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            return vis
    
    def generate_thumbnails(self):
        """
        Thumbnails from the same export image (RAW + .ax scientific steps),
        cropped around each polygon with the same zoom-out parameters used in ProjectTab.
        Uses a snapshot of polygons to avoid concurrent wipes.
        """
        selected_groups = self.get_selected_groups()
        if not selected_groups:
            QtWidgets.QMessageBox.warning(self, "No Groups Selected", "Please select one or more polygon groups.")
            return

        # Snapshot BEFORE dialogs/IO
        polygons_by_group = self._snapshot_polygons_by_group(selected_groups)

        save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder for Thumbnails")
        if not save_dir:
            return

        # Match ProjectTab parameters
        zoom_factor = 1.4           # same zoom-out factor
        thumbnail_size = (200, 200) # same output size
        thickness = 2               # same outline thickness

        import os, numpy as np, cv2, logging

        # Helper to get name builder
        build_name = getattr(self.parent_tab, "construct_thumbnail_name", None)

        for group_name in selected_groups:
            file_polygons = polygons_by_group.get(group_name, {})
            for filepath, pdata in file_polygons.items():
                raw_pts = pdata.get("points", [])
                if not raw_pts:
                    continue

                # Make a list of polygons: [[(x,y),...], [(x,y),...], ...]
                if isinstance(raw_pts[0], (list, tuple)) and len(raw_pts[0]) == 2 and not isinstance(raw_pts[0][0], (list, tuple)):
                    polys = [raw_pts]  # single polygon/point
                elif isinstance(raw_pts[0], (list, tuple)) and len(raw_pts[0]) == 2 and isinstance(raw_pts[0][0], (list, tuple)):
                    polys = raw_pts     # already list of polygons
                else:
                    parsed = self.parse_polygon_points(raw_pts)
                    polys = [parsed] if parsed else []
                if not polys:
                    continue

                # Export image (includes rotate/crop/resize/expr)
                img, _ = self._get_export_image(filepath)
                if img is None:
                    logging.warning(f"Could not load image at {filepath}")
                    continue
                H, W = img.shape[:2]

                # Determine image type for color
                image_type = self._image_type_for(filepath, img)
                if image_type == "rgb":
                    color = (255, 0, 0)     # red
                elif image_type == "thermal":
                    color = (0, 255, 255)   # yellow
                else:
                    color = (0, 255, 0)     # green

                # Prepare an 8-bit BGR preview source for drawing/saving
                vis_full = self._as_8bit_bgr(img)

                # Loop over polygons in this file
                for idx, poly in enumerate(polys, start=1):
                    parsed = self.parse_polygon_points(poly)
                    if not parsed:
                        continue

                    # Map to export-image pixel coords (viewer-independent)
                    pts_img = self._map_points_scene_to_image(filepath, parsed, img.shape, polygon_data=pdata)
                    if not pts_img:
                        continue

                    # Handle the single-point case by giving it a tiny bbox first
                    if len(pts_img) == 1:
                        xi, yi = pts_img[0]
                        xi = max(0, min(W-1, int(round(xi))))
                        yi = max(0, min(H-1, int(round(yi))))
                        x, y, w, h = xi, yi, 1, 1
                    else:
                        pts_np = np.array(pts_img, dtype=np.int32)
                        x, y, w, h = cv2.boundingRect(pts_np)

                    # Zoom-out bbox
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    new_w = int(round(w * zoom_factor))
                    new_h = int(round(h * zoom_factor))
                    new_w = max(16, new_w)
                    new_h = max(16, new_h)

                    x0 = int(round(cx - new_w / 2.0))
                    y0 = int(round(cy - new_h / 2.0))
                    x0 = max(0, min(x0, W - new_w))
                    y0 = max(0, min(y0, H - new_h))

                    crop = vis_full[y0:y0+new_h, x0:x0+new_w].copy()
                    if crop is None or crop.size == 0:
                        continue

                    adj = []
                    for (px, py) in pts_img:
                        adj.append([int(round(px - x0)), int(round(py - y0))])
                    adj_np = np.array([adj], dtype=np.int32)
                    if len(adj) >= 3:
                        cv2.polylines(crop, [adj_np], isClosed=True, color=color, thickness=thickness)
                    else:
                        (px, py) = adj[0]
                        cv2.rectangle(crop, (px-2, py-2), (px+2, py+2), color, thickness=thickness)

                    thumb = cv2.resize(crop, thumbnail_size, interpolation=cv2.INTER_AREA)

                    # Build output filename
                    if callable(build_name):
                        base_name = build_name(group_name, filepath, image_type=image_type)
                        if len(polys) > 1:
                            root, ext = os.path.splitext(base_name)
                            out_name = f"{root}_p{idx}{ext}"
                        else:
                            out_name = base_name
                    else:
                        stem = os.path.splitext(os.path.basename(filepath))[0]
                        suffix = f"_p{idx}" if len(polys) > 1 else ""
                        out_name = f"{group_name}_{stem}{suffix}.jpg"

                    out_path = os.path.join(save_dir, out_name)
                    ok = cv2.imwrite(out_path, thumb)
                    if not ok:
                        logging.error(f"Failed to write thumbnail: {out_path}")

        QtWidgets.QMessageBox.information(self, "Thumbnails Generated", f"Saved in {save_dir}")

    def _normalize_to_polygons(self, raw_pts):
        """
        Return a list of polygons (each polygon = list[(x,y)]).
        Accepts:
          - single polygon as list[(x,y)] or list[{'x','y'}]
          - list of polygons [[(x,y)...], [(x,y)...], ...]
          - single point [(x,y)] or {'x':...,'y':...}
        """
        def _looks_like_polygon(seq):
            return isinstance(seq, (list, tuple)) and len(seq) > 0 and \
                   all(isinstance(p, (list, tuple)) and len(p) == 2 for p in seq)

        if not raw_pts:
            return []

        # list of polygons?
        if isinstance(raw_pts, (list, tuple)) and len(raw_pts) > 0 and _looks_like_polygon(raw_pts[0]):
            polys = []
            for p in raw_pts:
                parsed = self.parse_polygon_points(p)
                if parsed:
                    polys.append(parsed)
            return polys

        # single polygon/point
        parsed = self.parse_polygon_points(raw_pts)
        return [parsed] if parsed else []

    def generate_segmentation_images(self):
        """
        Segmentation masks (16-bit) using the same export image geometry (RAW + .ax).
        Uses a snapshot of polygons to avoid concurrent wipes.
        """
        selected_groups = self.get_selected_groups()
        if not selected_groups:
            QtWidgets.QMessageBox.warning(self, "No Groups Selected", "Please select one or more polygon groups.")
            return

        # Snapshot BEFORE dialogs/IO
        polygons_by_group = self._snapshot_polygons_by_group(selected_groups)

        save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder for Segmentation Masks and CSV")
        if not save_dir:
            return

        import os, numpy as np, cv2, logging, csv

        filepath_to_polygons = {}
        for g in selected_groups:
            for fp, pdata in polygons_by_group.get(g, {}).items():
                if fp not in filepath_to_polygons:
                    filepath_to_polygons[fp] = []
                pts = pdata.get('points', [])
                if not pts:
                    continue
                # normalize to list-of-polygons
                if isinstance(pts[0][0], list):
                    for poly in pts:
                        filepath_to_polygons[fp].append((g, poly))
                else:
                    filepath_to_polygons[fp].append((g, pts))

        csv_path = os.path.join(save_dir, "segmentation_labels.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                W = csv.writer(csvfile)
                W.writerow(['object_id', 'group_name', 'image_file', 'polygon_index'])

                for filepath, polys in filepath_to_polygons.items():
                    img, _C = self._get_export_image(filepath)
                    if img is None:
                        logging.warning(f"Could not open image at {filepath}")
                        continue
                    H, Wimg = img.shape[:2]
                    mask = np.zeros((H, Wimg), dtype=np.uint16)

                    label = 1
                    poly_idx = 1
                    for group_name, poly_points in polys:
                        parsed = self.parse_polygon_points(poly_points)
                        if not parsed:
                            logging.warning(f"Invalid polygon in {filepath}")
                            continue
                        # map without viewer, using SNAPSHOT payload
                        pdata = polygons_by_group.get(group_name, {}).get(filepath, {})
                        pts_img = self._map_points_scene_to_image(filepath, parsed, img.shape, polygon_data=pdata)

                        # --- NEW: collapse decorated "point" shapes to a single pixel
                        is_point_shape = str(pdata.get("type", "")).lower() == "point"
                        if is_point_shape and pts_img:
                            cx = int(round(sum(x for x, _ in pts_img) / len(pts_img)))
                            cy = int(round(sum(y for _, y in pts_img) / len(pts_img)))
                            pts_img = [(cx, cy)]

                        if len(pts_img) == 1:
                            x, y = pts_img[0]
                            if 0 <= y < H and 0 <= x < Wimg:
                                mask[y, x] = label
                        else:
                            arr = np.array([pts_img], dtype=np.int32)
                            cv2.fillPoly(mask, arr, int(label))

                        W.writerow([label, group_name, filepath, poly_idx])
                        label += 1
                        poly_idx += 1

                    out = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_mask.png")
                    if not cv2.imwrite(out, mask):
                        logging.error(f"Failed to write mask {out}")
        except Exception as e:
            logging.critical(f"Failed to write segmentation CSV/masks: {e}")
            QMessageBox.critical(self, "Export Failed", f"Error exporting segmentation:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Done", f"Masks + CSV saved to:\n{save_dir}")


    def parse_polygon_points(self, poly_points):
        """Parse polygon points into a list of [x, y] lists."""
        parsed_points = []
        try:
            if isinstance(poly_points, list):
                if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly_points):
                    parsed_points = [list(p) for p in poly_points]
                elif all(isinstance(p, dict) and 'x' in p and 'y' in p for p in poly_points):
                    parsed_points = [[p['x'], p['y']] for p in poly_points]
                else:
                    return None
            elif isinstance(poly_points, dict):
                if 'x' in poly_points and 'y' in poly_points:
                    parsed_points = [[poly_points['x'], poly_points['y']]]
                else:
                    return None
            else:
                return None
            return parsed_points
        except Exception as e:
            logging.error(f"Error parsing polygon points: {e}")
            return None

    def get_group_name(self, filepath, selected_groups):
        for group_name in selected_groups:
            if filepath in self.parent_tab.all_polygons.get(group_name, {}):
                return group_name
        return "Unknown"
   

class RootOffsetDialog(QDialog):
    # Define a custom signal that emits the new offset value
    offset_changed = pyqtSignal(int)

    def __init__(self, current_offset, parent=None):
        super(RootOffsetDialog, self).__init__(parent)
        self.setWindowTitle("Set Root Offset")
        self.resize(400, 150)

        # Ensure the dialog is non-modal
        self.setModal(False)

        # Main Layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Instruction Label
        instruction_label = QLabel("Adjust root offset using the slider or the buttons below:")
        main_layout.addWidget(instruction_label)

        # Horizontal Layout for Slider and Buttons
        slider_layout = QHBoxLayout()
        main_layout.addLayout(slider_layout)

        # Previous Button
        self.prev_button = QPushButton("Previous")
        self.prev_button.setToolTip("Decrease offset by 1")
        self.prev_button.clicked.connect(self.decrement_offset)
        slider_layout.addWidget(self.prev_button)

        # Slider Configuration
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-1000, 1000)
        self.offset_slider.setValue(current_offset)
        self.offset_slider.setTickInterval(20)
        self.offset_slider.setTickPosition(QSlider.TicksBelow)
        self.offset_slider.setSingleStep(1)
        slider_layout.addWidget(self.offset_slider)

        # Next Button
        self.next_button = QPushButton("Next")
        self.next_button.setToolTip("Increase offset by 1")
        self.next_button.clicked.connect(self.increment_offset)
        slider_layout.addWidget(self.next_button)

        # Display Current Value
        self.value_label = QLabel(f"Current Offset: {current_offset}")
        main_layout.addWidget(self.value_label)

        # Connect the slider's valueChanged signal to update the label and button states
        self.offset_slider.valueChanged.connect(self.on_slider_value_changed)

        # Connect the slider's sliderReleased signal to emit the offset_changed signal
        self.offset_slider.sliderReleased.connect(self.on_slider_released)

        # Initialize button states based on current_offset
        self.update_button_states(current_offset)

        # Initialize Debounce Timer for Button Clicks Only
        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(300)  # 300 ms debounce interval
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.emit_offset_changed)

        # Store the latest offset value to emit after debounce
        self.latest_offset = current_offset

        # Click Counter for Button Clicks
        self.click_count = 0
        self.max_clicks = 3  # Maximum allowed rapid clicks

    def on_slider_value_changed(self, value):
        self.value_label.setText(f"Current Offset: {value}")
        self.update_button_states(value)
        self.latest_offset = value

    def on_slider_released(self):
        value = self.offset_slider.value()
        self.latest_offset = value
        self.offset_changed.emit(value)
        self.click_count = 0

    def emit_offset_changed(self):
        self.offset_changed.emit(self.latest_offset)
        self.click_count = 0

    def increment_offset(self):
        current_value = self.offset_slider.value()
        if current_value < self.offset_slider.maximum():
            new_value = current_value + 1
            self.offset_slider.setValue(new_value)
            self.latest_offset = new_value
            self.handle_button_click()

    def decrement_offset(self):
        current_value = self.offset_slider.value()
        if current_value > self.offset_slider.minimum():
            new_value = current_value - 1
            self.offset_slider.setValue(new_value)
            self.latest_offset = new_value
            self.handle_button_click()

    def handle_button_click(self):
        self.click_count += 1
        if self.click_count > self.max_clicks:
            return
        self.debounce_timer.start()

    def update_button_states(self, current_value):
        self.prev_button.setEnabled(current_value > self.offset_slider.minimum())
        self.next_button.setEnabled(current_value < self.offset_slider.maximum())


class AnalysisOptionsDialog(QtWidgets.QDialog):
    """
    Export / Analysis options dialog.

    Adds a Band-math section where users can define indices with expressions
    using b1..bN (b1=Red, b2=Green, b3=Blue, extras continue as b4...).
    Parses either JSON or simple lines like:  GCC=b2/(b1+b2+b3)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export / Analysis Options")
        self.resize(560, 540)

        vbox = QtWidgets.QVBoxLayout(self)

        # --- Statistics ---
        stats_group = QtWidgets.QGroupBox("Statistics to compute")
        st_layout   = QtWidgets.QVBoxLayout(stats_group)

        self.chk_mean   = QtWidgets.QCheckBox("Mean")
        self.chk_median = QtWidgets.QCheckBox("Median")
        self.chk_std    = QtWidgets.QCheckBox("Standard deviation")
        self.chk_quant  = QtWidgets.QCheckBox("Quantiles (comma list)")
        self.le_quant   = QtWidgets.QLineEdit("5,25,50,75,95")  # accepts 0–1 or 0–100

        # Defaults
        self.chk_mean.setChecked(True)
        self.chk_median.setChecked(True)
        self.chk_std.setChecked(True)
        self.chk_quant.setChecked(True)
        self.le_quant.setEnabled(True)
        self.chk_quant.toggled.connect(self.le_quant.setEnabled)

        for w in (self.chk_mean, self.chk_median, self.chk_std, self.chk_quant, self.le_quant):
            st_layout.addWidget(w)
        vbox.addWidget(stats_group)

        # --- Polygon / Model Options ---
        extra_group  = QtWidgets.QGroupBox("Polygon & Model Options")
        extra_layout = QtWidgets.QGridLayout(extra_group)

        self.chk_shrink = QtWidgets.QCheckBox("Shrink / Swell polygons")
        self.sp_factor  = QtWidgets.QDoubleSpinBox()
        self.sp_factor.setRange(0.0, 1.0)
        self.sp_factor.setSingleStep(0.01)
        self.sp_factor.setValue(0.07)
        self.cmb_dir    = QtWidgets.QComboBox()
        self.cmb_dir.addItems(["Shrink", "Swell"])

        self.chk_shrink.setChecked(True)
        self.sp_factor.setEnabled(True)
        self.cmb_dir.setEnabled(True)
        self.chk_shrink.toggled.connect(lambda b: (self.sp_factor.setEnabled(b), self.cmb_dir.setEnabled(b)))

        self.chk_rf = QtWidgets.QCheckBox("Use Random-Forest classification")
        self.chk_rf.setChecked(False)

        self.chk_export_mod_polys = QtWidgets.QCheckBox("Export modified polygons JSON")
        self.chk_export_mod_polys.setChecked(False)

        self.chk_exif = QtWidgets.QCheckBox("Include EXIF metadata")
        self.chk_exif.setChecked(False)

        extra_layout.addWidget(self.chk_shrink, 0, 0, 1, 2)
        extra_layout.addWidget(QtWidgets.QLabel("Factor:"), 1, 0)
        extra_layout.addWidget(self.sp_factor, 1, 1)
        extra_layout.addWidget(self.cmb_dir,   2, 0, 1, 2)
        extra_layout.addWidget(self.chk_rf,    3, 0, 1, 2)
        extra_layout.addWidget(self.chk_export_mod_polys, 4, 0, 1, 2)
        extra_layout.addWidget(self.chk_exif,  5, 0, 1, 2)

        vbox.addWidget(extra_group)

        # --- Band-math indices ---
        DEFAULT_BANDMATH_TEXT = (
            '{\n'
            '  "sum": "b1 + b2 + b3",\n'
            '  "GCC": "b2 / (b1 + b2 + b3)",\n'
            '  "EXG": "2*b2 - (b1 + b3)",\n'
            '  "RCC": "b1 / (b1 + b2 + b3)",\n'
            '  "BCC": "b3 / (b1 + b2 + b3)",\n'
            '  "WDX_2": "(2*b3) + b1 - (2*b2)",\n'
            '  "WDX": "b3 + 2*b1 - b2",\n'
            '  "WDX_3": "b3 + 2*b1 - 2*b2"\n'
            '}'
        )


        bm_group  = QtWidgets.QGroupBox("Band-math indices (b1 = Red, b2 = Green, b3 = Blue)")
        bm_layout = QtWidgets.QVBoxLayout(bm_group)

        self.chk_bandmath = QtWidgets.QCheckBox("Compute these indices")
        self.chk_bandmath.setChecked(False)

        self.te_bandmath = QtWidgets.QPlainTextEdit()
        self.te_bandmath.setPlaceholderText(
            'JSON or lines like:  GCC=b2/(b1+b2+b3), EXG=2*b2-(b1+b3)\n'
            'b1..bN refer to export-order channels (RGB -> b1,b2,b3, then extras).'
        )
        self.te_bandmath.setMinimumHeight(160)
        self.te_bandmath.setPlainText(DEFAULT_BANDMATH_TEXT)

        bm_layout.addWidget(self.chk_bandmath)
        bm_layout.addWidget(self.te_bandmath)
        vbox.addWidget(bm_group)

        # --- OK / Cancel ---
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

    @staticmethod
    def _parse_bandmath(text: str) -> dict:
        """Parse JSON or simple 'name=expr' / 'name:expr' lines into a dict."""
        import json, re
        text = (text or "").strip()
        if not text:
            return {}
        # Try JSON
        try:
            d = json.loads(text)
            return {str(k): str(v) for k, v in d.items()}
        except Exception:
            pass
        # Fallback: split by newlines/commas into name=expr or name:expr
        out = {}
        parts = re.split(r'[\n,]+', text)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if '=' in p:
                k, v = p.split('=', 1)
            elif ':' in p:
                k, v = p.split(':', 1)
            else:
                continue
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
        return out

    def get_options(self):
        """Collect dialog options into a dict used by the export pipeline."""
        # Parse quantiles safely
        quantiles = []
        if self.chk_quant.isChecked():
            txt = (self.le_quant.text() or "").strip()
            if txt:
                for token in txt.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        quantiles.append(float(token))
                    except ValueError:
                        # ignore invalid entries
                        continue

        opts = {
            "stats": {
                "mean":      self.chk_mean.isChecked(),
                "median":    self.chk_median.isChecked(),
                "std":       self.chk_std.isChecked(),
                "quantiles": quantiles,
            },
            "band_expression": None,  # kept for compatibility with older code paths
            "shrink": {
                "enabled": self.chk_shrink.isChecked(),
                "factor":  self.sp_factor.value(),
                "swell":   (self.cmb_dir.currentIndex() == 1),
            },
            "use_random_forest":        self.chk_rf.isChecked(),
            "export_modified_polygons": self.chk_export_mod_polys.isChecked(),
            "include_exif":             self.chk_exif.isChecked(),

            # NEW: user-defined indices to compute during export
            "band_math": {
                "enabled":  self.chk_bandmath.isChecked(),
                "formulas": self._parse_bandmath(self.te_bandmath.toPlainText()),
            },
        }
        return opts

