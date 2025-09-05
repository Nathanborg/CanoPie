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
from .image_viewer import EditablePolygonItem, EditablePointItem

class ImageEditorDialog(QDialog):
    # signal to let parent optionally react when group/all-group mods are saved
    modificationsAppliedToGroup = QtCore.pyqtSignal(str)

    # -------------------- configurable stretch knobs --------------------
    STRETCH_LOW_P = 0.5
    STRETCH_HIGH_P = 99.5
    STRETCH_PER_CHANNEL = True
    STRETCH_CLIP = True
    STRETCH_SAMPLE_MAX = 250   # compute percentiles on a downsampled copy for speed

    def __init__(self, parent=None, image_data=None, image_filepath=""):
        super().__init__(parent)
        self.setWindowTitle("Edit Image Viewer - Crop")

        # project folder (so .ax files go there if available)
        self.project_folder = getattr(parent, "project_folder", None)

        # determine filepath
        self.image_filepath = image_filepath or (
            image_data.filepath if (hasattr(image_data, "filepath") and image_data.filepath) else ""
        )

        # ALWAYS anchor editor to the raw on-disk image to avoid double-scaling from viewer
        raw = self._load_raw_image()
        if raw is not None:
            self.base_image = raw
        else:
            self.base_image = image_data.copy() if isinstance(image_data, np.ndarray) else None

        self.original_image = self.base_image.copy() if self.base_image is not None else None
        self.display_image_data = self.normalize_image_for_display(self.original_image)

        # state
        self.crop_rect = None                   # QtCore.QRect while dragging
        self.last_crop_rect = None              # last rect chosen (in *view* coords at time of selection)
        self.last_crop_ref_size = None          # (w,h) of the image on which last_crop_rect was defined *after inverse resize*
        self.modifications = {}                 # dict persisted to .ax
        self.last_band_float_result = None

        # ui
        self.init_ui()
        self.setMinimumSize(800, 600)
        self.fit_to_window = True
        self.zoom = 1.0

        # load mods if any
        modifications = self.load_modifications_from_file()
        if modifications:
            self.modifications.update(modifications)
            if "orig_size" not in self.modifications:
                sh = self._raw_shape()
                if sh:
                    h, w = sh[:2]
                    c = 1 if len(sh) == 2 else int(sh[2])
                    self.modifications["orig_size"] = {"h": int(h), "w": int(w), "c": int(c)}
                    self.modifications["anchor_to_original"] = True

        # apply current mods (or none) to raw base
        self.apply_modifications(self.modifications)

    # ---------- helpers for .ax paths ----------
    def _auto_refresh_after_reset(self, root_name=None):
        """
        Centralized refresh after any reset:
          1) Try parent.refresh_all_groups()
          2) Else try parent.refresh_viewer(root_name=...)
          3) Else  try parent.load_image_group(root_name)
        Ensures to make ONE refresh call.
        """
        parent = self.parent()
        try:
            if parent is None:
                return
            if hasattr(parent, "refresh_all_groups"):
                parent.refresh_all_groups()
                return
            if hasattr(parent, "refresh_viewer"):
                parent.refresh_viewer(root_name=root_name)
                return
            if root_name and hasattr(parent, "load_image_group"):
                parent.load_image_group(root_name)
        except Exception as e:
            logging.debug(f"Refresh after reset failed: {e}")

    # --- SAME-FOLDER FILTERS (Windows/Linux safe) ---
    def _same_dir_as_current(self, other_path: str) -> bool:
        import os
        try:
            cur = os.path.normcase(os.path.abspath(os.path.dirname(self.image_filepath or "")))
            oth = os.path.normcase(os.path.abspath(os.path.dirname(other_path or "")))
            return cur == oth
        except Exception:
            return False

    def _filter_same_dir(self, paths):
        return [p for p in (paths or []) if p and self._same_dir_as_current(p)]

    def _ax_path_for(self, image_path: str) -> str:
        folder = self.project_folder if (self.project_folder and self.project_folder.strip()) else os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(folder, base_name + ".ax")

    def _write_ax(self, image_path, modifications):
        mod_filename = self._ax_path_for(image_path)
        try:
            existing = {}
            if os.path.exists(mod_filename):
                try:
                    with open(mod_filename, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing.update(modifications)
            with open(mod_filename, "w") as f:
                json.dump(existing, f, indent=4)
            logging.info(f"Modifications saved to {mod_filename}")
        except Exception as e:
            logging.error(f"Failed to save modifications to {mod_filename}: {e}")

    def _load_raw_image(self):
        """
        Load the *on-disk* original image.
        - For TIFFs that are stacks (>3 bands) or multi-page, force tifffile.
        - Coerce CxHxW -> HxWxC when needed so the rest of the editor sees HWC.
        Falls back to cv2 for everything else.
        """
        import os, logging, numpy as np, cv2
        if not self.image_filepath:
            return None

        ext = os.path.splitext(self.image_filepath)[1].lower()

        def _tifffile_is_stack(path):
            try:
                import tifffile as tiff
                with tiff.TiffFile(path) as tf:
                    n_pages = len(tf.pages)
                    series = tf.series[0] if tf.series else None
                    axes   = getattr(series, "axes", "") or ""
                    shape  = getattr(series, "shape", ()) or ()
                    spp    = getattr(tf.pages[0], "samplesperpixel", 1) or 1

                    band_count = spp
                    if axes and shape:
                        if 'C' in axes:
                            band_count = int(shape[axes.index('C')])
                        elif 'S' in axes:
                            band_count = int(shape[axes.index('S')])
                        elif len(shape) == 3:
                            if shape[0] <= 32 and shape[1] >= 32 and shape[2] >= 32:
                                band_count = int(shape[0])
                            elif shape[2] <= 32:
                                band_count = int(shape[2])

                    return (band_count > 3) or (n_pages > 1)
            except Exception as e:
                logging.debug(f"[Editor] TIFF preflight failed: {e}")
                return False

        def _tifffile_read_HWC(path):
            import tifffile as tiff, numpy as np
            arr = tiff.imread(path)
            arr = np.squeeze(arr)
            # If channel-first (C,H,W) with small C, move to H,W,C
            if arr.ndim == 3 and arr.shape[0] <= 32 and arr.shape[1] >= 32 and arr.shape[2] >= 32:
                arr = np.moveaxis(arr, 0, -1)
            return arr

        # Prefer tifffile for real stacks / multi-page TIFFs or TIFFs with >3 samples per pixel
        if ext in (".tif", ".tiff") and _tifffile_is_stack(self.image_filepath):
            try:
                arr = _tifffile_read_HWC(self.image_filepath)
                if arr is not None and getattr(arr, "size", 0) > 0:
                    return arr
            except Exception as e:
                logging.warning(f"[Editor] tifffile load failed, falling back to cv2: {e}")

        # Fallback: regular images (or simple TIFFs)
        try:
            img = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
            return img
        except Exception as e:
            logging.debug(f"_load_raw_image cv2 fallback failed: {e}")
            return None

    # ---------- rect extractor (QRect or dict) ----------
    def _extract_crop_xywh(self, crop_rect_obj):
        """
        Accepts a QtCore.QRect or a dict {'x','y','width','height'} and returns (x,y,w,h) ints.
        Returns None if invalid.
        """
        if crop_rect_obj is None:
            return None
        try:
            if isinstance(crop_rect_obj, QtCore.QRect):
                return int(crop_rect_obj.x()), int(crop_rect_obj.y()), int(crop_rect_obj.width()), int(crop_rect_obj.height())
            if isinstance(crop_rect_obj, dict):
                x = int(crop_rect_obj.get("x", 0))
                y = int(crop_rect_obj.get("y", 0))
                w = int(crop_rect_obj.get("width", 0))
                h = int(crop_rect_obj.get("height", 0))
                return x, y, w, h
        except Exception as e:
            logging.debug(f"_extract_crop_xywh: {e}")
        return None

    def save_modifications_to_file(self):
        """
        Save modification parameters into .ax and return a refresh hint:
          {"scope": "all"|"group"|"single", "root_name": <str or None>}

        Behavior:
        - For both 'Apply to group' and 'Apply to all groups',only target files that
          live in the SAME OS DIRECTORY as the image currently being edited.
        - If the parent’s group mapping doesn’t include that directory,  fall back to
          scanning that directory on disk (common image extensions).
        """
        import os, json, logging

        if not self.image_filepath:
            logging.error("No image_filepath set. Cannot save modifications.")
            return None

        # ---------- helpers ----------
        def _norm_dir(p: str) -> str:
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(os.path.dirname(p))))
            except Exception:
                return ""

        def _flatten_known_files():
            """Return a de-duped list of paths from all known groups (strings only)."""
            parent = self.parent()
            out = []
            if parent is not None and hasattr(parent, "multispectral_image_data_groups"):
                try:
                    for group_files in parent.multispectral_image_data_groups.values():
                        if isinstance(group_files, (list, tuple)):
                            for fp in group_files:
                                if isinstance(fp, str) and fp:
                                    out.append(fp)
                except Exception as e:
                    logging.debug(f"_flatten_known_files failed: {e}")
            # de-dupe, preserve order
            return [fp for fp in dict.fromkeys(out)]

        def _group_files_for_current_root():
            """Files only from the current root/group (strings only)."""
            parent = self.parent()
            out = []
            root_name = None
            if parent is not None and hasattr(parent, "get_current_root_name"):
                try:
                    root_name = parent.get_current_root_name()
                except Exception:
                    root_name = None
            if root_name and hasattr(parent, "multispectral_image_data_groups"):
                try:
                    arr = parent.multispectral_image_data_groups.get(root_name, [])
                    if isinstance(arr, (list, tuple)):
                        out = [fp for fp in arr if isinstance(fp, str) and fp]
                except Exception as e:
                    logging.debug(f"_group_files_for_current_root failed: {e}")
            return [fp for fp in dict.fromkeys(out)], root_name

        def _same_dir_files(candidates, target_dir_norm):
            out = []
            for fp in candidates or []:
                try:
                    if _norm_dir(fp) == target_dir_norm:
                        out.append(fp)
                except Exception:
                    pass
            # de-dupe
            return [fp for fp in dict.fromkeys(out)]

        def _scan_folder_on_disk(target_dir_norm):
            """Fallback: list images in the directory on disk."""
            try:
                d = target_dir_norm
                if not d:
                    return []
                folder = d  # normcase/normpath is already fine for Windows
                exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                out = []
                for name in os.listdir(folder):
                    fp = os.path.join(folder, name)
                    if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in exts:
                        out.append(fp)
                return [fp for fp in dict.fromkeys(out)]
            except Exception as e:
                logging.debug(f"_scan_folder_on_disk failed: {e}")
            return []

        ui_mods = {}

        # rotation
        if "rotate" in self.modifications:
            try:
                ui_mods["rotate"] = int(self.modifications.get("rotate", 0)) % 360
            except Exception:
                ui_mods["rotate"] = 0

        # crop (prefer precise ref size captured at selection time)
        if "crop_rect" not in self.modifications and (self.last_crop_rect is not None or self.crop_rect is not None):
            xywh = self._extract_crop_xywh(self.last_crop_rect or self.crop_rect)
            if xywh:
                x, y, w, h = xywh
                ui_mods["crop_rect"] = {"x": x, "y": y, "width": w, "height": h}
                if self.last_crop_ref_size:
                    rw, rh = self.last_crop_ref_size
                    ui_mods["crop_rect_ref_size"] = {"w": int(rw), "h": int(rh)}
                else:
                    if self.original_image is not None and getattr(self.original_image, "shape", None):
                        Hc, Wc = self.original_image.shape[:2]
                        ui_mods["crop_rect_ref_size"] = {"w": int(Wc), "h": int(Hc)}

        # resize (PROPORTIONAL-ONLY: read single Scale % box)
        text = self.resize_input.text().strip() if hasattr(self, "resize_input") else ""
        if text:
            try:
                ui_mods["resize"] = {"scale": int(text)}
            except ValueError:
                pass

        # band expression
        if hasattr(self, "band_input"):
            expr = self.band_input.text().strip()
            if expr:
                ui_mods["band_expression"] = expr

        # ensure anchor metadata exists
        if "orig_size" not in self.modifications and self.base_image is not None:
            H0, W0 = self.base_image.shape[:2]
            C0 = 1 if self.base_image.ndim == 2 else int(self.base_image.shape[2])
            ui_mods["orig_size"] = {"h": int(H0), "w": int(W0), "c": int(C0)}
            ui_mods["anchor_to_original"] = True

        # merge: only add missing keys
        for k, v in ui_mods.items():
            if k not in self.modifications:
                self.modifications[k] = v

        modifications = dict(self.modifications)
        parent = self.parent()
        cur_dir_norm = _norm_dir(self.image_filepath)

        # ---------- choose targets by scope (then restrict to same folder) ----------
        # ALL GROUPS path pool
        all_known = _flatten_known_files()

        # CURRENT GROUP path pool
        group_known, root_name = _group_files_for_current_root()

        def _targets_for_scope(scope: str):
            if scope == "all":
                pool = all_known
            elif scope == "group":
                pool = group_known
            else:
                pool = []
            same_dir = _same_dir_files(pool, cur_dir_norm)
            if not same_dir:
                # fallback: scan directory on disk if group mapping didn’t include it
                same_dir = _scan_folder_on_disk(cur_dir_norm)
            # always include current file
            if self.image_filepath and self.image_filepath not in same_dir:
                same_dir.append(self.image_filepath)
            return [fp for fp in dict.fromkeys(same_dir)]

        # ======== ALL GROUPS (same-folder) ========
        if getattr(self, "apply_all_groups_checkbox", None) and self.apply_all_groups_checkbox.isChecked():
            files = _targets_for_scope("all")
            for fp in files:
                self._write_ax(fp, modifications)
            logging.info(
                "Applied modifications to %d file(s) across ALL groups (dir=%s).",
                len(files), cur_dir_norm
            )

            try:
                if parent is not None and hasattr(parent, "get_current_root_name"):
                    rn = parent.get_current_root_name()
                    if hasattr(parent, "refresh_viewer"):
                        parent.refresh_viewer(root_name=rn)
                    elif hasattr(parent, "load_image_group") and rn:
                        parent.load_image_group(rn)
            except Exception as e:
                logging.debug(f"Viewer refresh after all-groups apply failed: {e}")

            try:
                self.modificationsAppliedToGroup.emit("")
            except Exception:
                pass

            return {"scope": "all", "root_name": None}

        # ======== CURRENT GROUP (same-folder) ========
        if getattr(self, "global_mods_checkbox", None) and self.global_mods_checkbox.isChecked():
            files = _targets_for_scope("group")
            for fp in files:
                self._write_ax(fp, modifications)
            logging.info(
                "Applied modifications to %d file(s) in group '%s' (dir=%s).",
                len(files), root_name or "unknown", cur_dir_norm
            )

            try:
                if parent is not None:
                    if hasattr(parent, "refresh_viewer"):
                        parent.refresh_viewer(root_name=root_name)
                    elif hasattr(parent, "load_image_group") and root_name:
                        parent.load_image_group(root_name)
            except Exception as e:
                logging.debug(f"Viewer refresh after group-apply failed: {e}")

            try:
                self.modificationsAppliedToGroup.emit(root_name or "")
            except Exception:
                pass

            return {"scope": "group", "root_name": root_name}

        # ======== SINGLE IMAGE ========
        self._write_ax(self.image_filepath, modifications)

        root_name = None
        try:
            if parent is not None and hasattr(parent, "get_current_root_name"):
                root_name = parent.get_current_root_name()
        except Exception:
            root_name = None

        return {"scope": "single", "root_name": root_name}

    def load_modifications_from_file(self):
        if not self.image_filepath:
            return None
        mod_filename = self._ax_path_for(self.image_filepath)
        logging.debug(f"Looking for modifications in: {mod_filename}")
        if os.path.exists(mod_filename):
            try:
                with open(mod_filename, "r") as f:
                    modifications = json.load(f)
                logging.info(f"Modifications loaded from {mod_filename}")
                return modifications
            except Exception as e:
                logging.error(f"Failed to load modifications from {mod_filename}: {e}")
        return None


    # ---------- apply pipeline ----------
    def apply_all_modifications_to_image(self, image, modifications):
        """
        Order: rotate(90° steps) -> crop -> resize -> band expression.
        Rotation/crop/resize operate on native dtype. Band expression returns float32 (scientific).
        """
        if image is None or getattr(image, "size", 0) == 0:
            logging.error("apply_all_modifications_to_image: empty source image.")
            return image

        result = image.copy()

        # ---- rotation (90-degree steps) ---- (NumPy so >4 channels are fine)
        def _rotate_90s_numpy(arr, deg):
            import numpy as np
            d = int(deg) % 360
            if d == 0 or arr is None or getattr(arr, "size", 0) == 0:
                return arr
            if d == 90:      # clockwise
                out = np.rot90(arr, -1)
            elif d == 180:
                out = np.rot90(arr, 2)
            elif d == 270:   # counter-clockwise
                out = np.rot90(arr, 1)
            else:
                return arr
            return np.ascontiguousarray(out)

        try:
            rot = int(modifications.get("rotate", 0)) if "rotate" in modifications else 0
            rot = ((rot % 360) + 360) % 360
        except Exception:
            rot = 0

        if rot in (90, 180, 270):
            try:
                result = _rotate_90s_numpy(result, rot)
                logging.debug(f"Applied rotation: {rot} degrees (NumPy).")
            except Exception as e:
                logging.warning(f"Rotation failed ({rot} deg) via NumPy: {e}")

        # ---- crop (scale rect from ref-size -> current source) ----
        if "crop_rect" in modifications:
            xywh = self._extract_crop_xywh(modifications["crop_rect"])
            if xywh:
                x, y, w, h = xywh
                H, W = result.shape[:2]

                ref = modifications.get("crop_rect_ref_size") or {}
                ref_w = int(ref.get("w", W)) or W
                ref_h = int(ref.get("h", H)) or H
                if ref_w != W or ref_h != H:
                    sx = W / float(ref_w)
                    sy = H / float(ref_h)
                    x = int(round(x * sx))
                    y = int(round(y * sy))
                    w = int(round(w * sx))
                    h = int(round(h * sy))

                x0 = max(0, min(x, W))
                y0 = max(0, min(y, H))
                x1 = max(0, min(x + w, W))
                y1 = max(0, min(y + h, H))
                if x1 > x0 and y1 > y0:
                    result = result[y0:y1, x0:x1]
                    logging.debug(f"Applied crop modification (scaled): ({x0},{y0})-({x1},{y1})")
                else:
                    logging.warning("Crop rect out of bounds/empty after scaling; skipping crop.")

        # early exit if cropped to nothing
        if result is None or result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0:
            logging.error("Result is empty after crop; skipping further modifications.")
            return image

        # ---- resize (adaptive interpolation for crisp preview) ----
        if "resize" in modifications:
            resize_info = modifications["resize"]
            h0, w0 = result.shape[:2]
            if h0 > 0 and w0 > 0:
                if "scale" in resize_info:
                    scale = max(1, int(resize_info["scale"]))
                    new_w = max(1, int(w0 * scale / 100.0))
                    new_h = max(1, int(h0 * scale / 100.0))
                else:
                    pct_w = int(resize_info.get("width", 100))
                    pct_h = int(resize_info.get("height", 100))
                    new_w = max(1, int(w0 * pct_w / 100.0))
                    new_h = max(1, int(h0 * pct_h / 100.0))

                if new_w != w0 or new_h != h0:
                    sw = new_w / float(w0)
                    sh = new_h / float(h0)
                    if sw < 1.0 or sh < 1.0:
                        interp = cv2.INTER_AREA        # best for downscale
                    elif max(sw, sh) < 2.0:
                        interp = cv2.INTER_LINEAR      # mild upscale
                    else:
                        interp = cv2.INTER_CUBIC       # larger upscales
                    result = resize_safe(result, new_w, new_h, interp)
                    logging.debug(f"Applied resize modification to {new_w}x{new_h} (interp={interp}).")
            else:
                logging.warning("Resize skipped: source has zero dimension.")

        # ---- band expression (returns FLOAT scientific; preview handled later) ----
        if "band_expression" in modifications:
            expr = modifications["band_expression"]
            if expr:
                result = self.process_band_expression(result, expr)
                logging.debug("Applied band expression modification.")

        return result



    def apply_modifications(self, modifications):
        if self.base_image is None:
            return
        mod_image = self.apply_all_modifications_to_image(self.base_image, modifications)
        self.original_image = mod_image  # keep scientific dtype here
        self.display_image_data = self.normalize_image_for_display(mod_image)
        self.display_image(self.display_image_data)



    def _sample_for_stats(self, arr):
        """Return a downsampled view for percentile stats to keep UI responsive."""
        h, w = arr.shape[:2]
        m = max(h, w)
        if m <= self.STRETCH_SAMPLE_MAX:
            return arr
        scale = self.STRETCH_SAMPLE_MAX / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return resize_safe(arr, new_w, new_h, cv2.INTER_AREA)  # <-- changed

    # ---------- band expression ----------
    def process_band_expression(self, image, expr):
        """
        Evaluate band expression and return FLOAT32 (scientific) array.
        DO NOT quantize here; display stretch happens in normalize_image_for_display.
        """
        bands = re.findall(r'b(\d+)', expr)
        unique_bands = sorted(set(bands), key=lambda x: int(x))
        try:
            if image.ndim == 2:
                band_mapping = {'b1': image.astype(np.float32, copy=False)}
            elif image.ndim == 3:
                band_mapping = {}
                num_bands = image.shape[2]
                for b in unique_bands:
                    bi = int(b) - 1
                    if bi < 0 or bi >= num_bands:
                        raise ValueError(f"Band b{b} out of range for image with {num_bands} channels.")
                    band_mapping[f'b{b}'] = image[:, :, bi].astype(np.float32, copy=False)
            else:
                logging.error("Unsupported image format for band expression.")
                return image

            code = compile(expr, "<string>", "eval")
            for name in code.co_names:
                if name not in band_mapping:
                    raise NameError(f"Use of '{name}' is not allowed. Only b1..bn.")

            res = eval(code, {"__builtins__": {}}, band_mapping)

            if not isinstance(res, np.ndarray):
                res = np.full(image.shape[:2], float(res), dtype=np.float32)
            else:
                res = res.astype(np.float32, copy=False)

            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            self.last_band_float_result = res.copy()

            # Return FLOAT so inspector & exporters see real values
            return res

        except Exception as e:
            logging.error(f"Error processing band expression '{expr}': {e}")
            return image

    def init_ui(self):
        from PyQt5 import QtGui, QtWidgets, QtCore
        from PyQt5.QtWidgets import QToolButton

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- image area ---
        self.image_label = QtWidgets.QLabel("No Image Loaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setScaledContents(False)  # do NOT let label auto-scale
        try:
            self.image_label.setAttribute(QtCore.Qt.WA_HighDpiPixmaps, True)
        except AttributeError:
            pass

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setBackgroundRole(QtGui.QPalette.Dark)
        self.scroll_area.setWidget(self.image_label)

        self.display_image(self.display_image_data)

        main_layout.addWidget(self.scroll_area)

        # crop rubber band
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.image_label)
        self.origin = QtCore.QPoint()
        self.crop_rect = None

        # ========== compact controls bar (single row) ==========
        controls_bar = QtWidgets.QHBoxLayout()
        controls_bar.setContentsMargins(0, 0, 0, 0)
        controls_bar.setSpacing(10)

        def _tight_row(*widgets):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for wid in widgets:
                h.addWidget(wid)
            return w

        # --- Scale (%) + tiny "apply" ---
        scale_label = QtWidgets.QLabel("Scale %:")
        self.resize_input = QtWidgets.QLineEdit()
        self.resize_input.setPlaceholderText("1–100")
        self.resize_input.setValidator(QtGui.QIntValidator(1, 100, self))
        self.resize_input.setFixedWidth(70)

        scale_apply_btn = QtWidgets.QPushButton("apply")
        scale_apply_btn.setFixedHeight(27)
        scale_apply_btn.setFixedWidth(48)
        scale_apply_btn.setToolTip("Apply proportional resize")
        scale_apply_btn.clicked.connect(self.on_resize_input_entered)
        # pressing Enter in the box triggers the same apply
        self.resize_input.returnPressed.connect(scale_apply_btn.click)

        controls_bar.addWidget(_tight_row(scale_label, self.resize_input, scale_apply_btn))

        # --- Band expression (label + input + button) ---
        band_label = QtWidgets.QLabel("Band:")
        self.band_input = QtWidgets.QLineEdit()
        self.band_input.setPlaceholderText("b1 + b2 / b4")
        self.band_input.setMinimumWidth(180)
        self.band_input.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.band_apply_button = QtWidgets.QPushButton("Apply")
        self.band_apply_button.setFixedHeight(27)
        self.band_apply_button.setToolTip("Apply band expression")
        self.band_apply_button.clicked.connect(self.apply_band_expression)

        controls_bar.addWidget(_tight_row(band_label, self.band_input, self.band_apply_button))

        # --- Rotate (label + two small tool buttons) ---
        rot_label = QtWidgets.QLabel("Rotate 90°:")
        self.rotate_left_btn  = QToolButton()
        self.rotate_right_btn = QToolButton()
        self.rotate_left_btn.setText("↶")
        self.rotate_right_btn.setText("↷")
        self.rotate_left_btn.setToolTip("Rotate 90° counter-clockwise")
        self.rotate_right_btn.setToolTip("Rotate 90° clockwise")
        self.rotate_left_btn.setFixedSize(28, 28)
        self.rotate_right_btn.setFixedSize(28, 28)
        self.rotate_left_btn.clicked.connect(self.on_rotate_left)
        self.rotate_right_btn.clicked.connect(self.on_rotate_right)

        controls_bar.addWidget(_tight_row(rot_label, self.rotate_left_btn, self.rotate_right_btn))

        # push everything left; keeps row compact
        controls_bar.addStretch(1)
        main_layout.addLayout(controls_bar)

        # --- buttons row (unchanged) ---
        buttons_layout = QtWidgets.QHBoxLayout()
        apply_all_button = QtWidgets.QPushButton("Apply All Changes")
        apply_all_button.clicked.connect(self.apply_all_changes)

        reset_img_button = QtWidgets.QPushButton("Reset Image")
        reset_img_button.setToolTip("Delete modifications (.ax) for this image and restore it.")
        reset_img_button.clicked.connect(self.on_reset_image)

        reset_group_button = QtWidgets.QPushButton("Reset root")
        reset_group_button.setToolTip("Delete modifications (.ax) for all images in the current group and refresh.")
        reset_group_button.clicked.connect(self.on_reset_group)

        reset_all_button = QtWidgets.QPushButton("Reset All")
        reset_all_button.setToolTip("Delete ALL .ax under the project folder and refresh once.")
        reset_all_button.clicked.connect(self.on_reset_all_groups)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        for b in (apply_all_button, reset_img_button, reset_group_button, reset_all_button, cancel_button):
            buttons_layout.addWidget(b)
        main_layout.addLayout(buttons_layout)

        # --- scope checkboxes (unchanged) ---
        self.global_mods_checkbox = QtWidgets.QCheckBox("Apply modifications to all images at this root for this folder group")
        self.global_mods_checkbox.setToolTip("Write the same .ax modifications to all images in the current root.")
        self.global_mods_checkbox.toggled.connect(self._on_group_apply_toggled)
        main_layout.addWidget(self.global_mods_checkbox)

        self.apply_all_groups_checkbox = QtWidgets.QCheckBox("Apply modifications to all roots for this folder group")
        self.apply_all_groups_checkbox.setToolTip("Write the same .ax modifications to every image in every group.")
        self.apply_all_groups_checkbox.toggled.connect(self._on_all_groups_toggled)
        main_layout.addWidget(self.apply_all_groups_checkbox)

        self.setLayout(main_layout)

        # mouse events for cropping
        self.image_label.mousePressEvent = self.image_mouse_press_event
        self.image_label.mouseMoveEvent = self.image_mouse_move_event
        self.image_label.mouseReleaseEvent = self.image_mouse_release_event

        # optional keyboard shortcuts for rotation
        try:
            QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+["), self, self.on_rotate_left)
            QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+]"), self, self.on_rotate_right)
        except Exception:
            pass

    def _on_all_groups_toggled(self, checked: bool):
        if checked:
            self.global_mods_checkbox.setChecked(False)
            self.global_mods_checkbox.setEnabled(False)
        else:
            self.global_mods_checkbox.setEnabled(True)

    def _on_group_apply_toggled(self, checked: bool):
        if checked:
            self.apply_all_groups_checkbox.blockSignals(True)
            self.apply_all_groups_checkbox.setChecked(False)
            self.apply_all_groups_checkbox.blockSignals(False)

    # ---------- lightweight delete helpers / resets ----------
    def _delete_ax_in_dir(self, base_dir: str) -> int:
        """Stream-delete *.ax files under base_dir. Returns count deleted."""
        deleted = 0
        try:
            for root, _dirs, files in os.walk(base_dir):
                for name in files:
                    if name.lower().endswith(".ax"):
                        ax_path = os.path.join(root, name)
                        try:
                            os.remove(ax_path)
                            deleted += 1
                        except Exception as e:
                            logging.debug(f"Could not delete {ax_path}: {e}")
        except Exception as e:
            logging.error(f"Walking '{base_dir}' failed: {e}")
        return deleted

    def on_reset_image(self):
        """Delete .ax for THIS image and reset view (no popups)."""
        self.modifications = {}
        self.last_band_float_result = None
        self.crop_rect = None
        self.last_crop_rect = None
        self.last_crop_ref_size = None

        mod_filename = self._ax_path_for(self.image_filepath)
        logging.debug(f"Resetting image modifications. Looking for file: {mod_filename}")
        if os.path.exists(mod_filename):
            try:
                os.remove(mod_filename)
                logging.info(f"Modifications file {mod_filename} deleted.")
            except Exception as e:
                logging.error(f"Failed to delete modifications file {mod_filename}: {e}")

        raw = self._load_raw_image()
        if raw is not None:
            self.base_image = raw
            self.original_image = self.base_image.copy()
            self.display_image_data = self.normalize_image_for_display(self.original_image)
            self.display_image(self.display_image_data)

        if hasattr(self, 'resize_input'): self.resize_input.clear()
        if hasattr(self, 'resize_width_input'): self.resize_width_input.clear()
        if hasattr(self, 'resize_height_input'): self.resize_height_input.clear()
        if hasattr(self, 'band_input'): self.band_input.clear()

        root_name = None
        parent = self.parent()
        if parent is not None and hasattr(parent, "get_current_root_name"):
            try:
                root_name = parent.get_current_root_name()
            except Exception:
                root_name = None
        self._auto_refresh_after_reset(root_name=root_name)
        logging.info("Image modifications reset.")

    def on_reset_group(self):
        """Delete .ax for all images in current group and auto-refresh (no popups)."""
        parent = self.parent()
        root_name = None
        files = []

        if parent is not None and hasattr(parent, "get_current_root_name"):
            root_name = parent.get_current_root_name()

        if root_name and hasattr(parent, "multispectral_image_data_groups"):
            files = list(parent.multispectral_image_data_groups.get(root_name, []))

        if not files and self.image_filepath:
            files = [self.image_filepath]

        deleted = 0
        for fp in files:
            try:
                mod_path = self._ax_path_for(fp)
                if os.path.exists(mod_path):
                    os.remove(mod_path)
                    deleted += 1
            except Exception as e:
                logging.error(f"Failed to delete modifications file for {fp}: {e}")

        # reset this editor view
        self.modifications = {}
        self.last_band_float_result = None
        self.crop_rect = None
        self.last_crop_rect = None
        self.last_crop_ref_size = None

        raw = self._load_raw_image()
        if raw is not None:
            self.base_image = raw
            self.original_image = self.base_image.copy()
            self.display_image_data = self.normalize_image_for_display(self.original_image)
            self.display_image(self.display_image_data)

        if hasattr(self, 'resize_input'): self.resize_input.clear()
        if hasattr(self, 'resize_width_input'): self.resize_width_input.clear()
        if hasattr(self, 'resize_height_input'): self.resize_height_input.clear()
        if hasattr(self, 'band_input'): self.band_input.clear()

        self._auto_refresh_after_reset(root_name=root_name)
        logging.info(f"Reset Group: removed {deleted} .ax file(s) for root '{root_name or 'unknown'}'.")

    def on_reset_all_groups(self):
        """Delete every .ax under the project folder (or current image folder) and refresh once. (No popups)"""
        base_dir = None
        if self.project_folder and self.project_folder.strip():
            base_dir = self.project_folder
        elif self.image_filepath:
            base_dir = os.path.dirname(self.image_filepath)

        if not base_dir or not os.path.isdir(base_dir):
            logging.warning("Reset All: couldn't locate a base folder to clean.")
            return

        deleted = self._delete_ax_in_dir(base_dir)

        # reset this editor view cheaply
        self.modifications = {}
        self.last_band_float_result = None
        self.crop_rect = None
        self.last_crop_rect = None
        self.last_crop_ref_size = None

        raw = self._load_raw_image()
        if raw is not None:
            self.base_image = raw
            self.original_image = self.base_image.copy()
            self.display_image_data = self.normalize_image_for_display(self.original_image)
            self.display_image(self.display_image_data)

        if hasattr(self, 'resize_input'): self.resize_input.clear()
        if hasattr(self, 'resize_width_input'): self.resize_width_input.clear()
        if hasattr(self, 'resize_height_input'): self.resize_height_input.clear()
        if hasattr(self, 'band_input'): self.band_input.clear()

        # unified auto-refresh
        self._auto_refresh_after_reset(root_name=None)
        logging.info(f"Reset All: removed {deleted} .ax file(s) under '{base_dir}'.")

    # ---------- misc ui handlers ----------
    def on_aspect_ratio_toggled(self, state):
        if state == QtCore.Qt.Checked:
            self.independent_widget.hide()
            logging.debug("Aspect ratio kept during resizing.")
        else:
            self.independent_widget.show()
            logging.debug("Aspect ratio not kept during resizing.")


    def _raw_shape(self):
        """
        Return the true on-disk image shape. Uses tifffile metadata for TIFFs
        so stacks report correct channel/page dimensions.
        """
        import os, logging, cv2
        if not self.image_filepath:
            return self.base_image.shape if self.base_image is not None else None

        ext = os.path.splitext(self.image_filepath)[1].lower()
        if ext in (".tif", ".tiff"):
            try:
                import tifffile as tiff
                with tiff.TiffFile(self.image_filepath) as tf:
                    series = tf.series[0] if tf.series else None
                    if series and series.shape:
                        # Normalize to HxWxC when possible
                        axes  = series.axes or ""
                        shape = list(series.shape)
                        if 'C' in axes:
                            # map to H,W,C
                            h = shape[axes.index('Y')] if 'Y' in axes else shape[-2]
                            w = shape[axes.index('X')] if 'X' in axes else shape[-1]
                            c = shape[axes.index('C')]
                            return (int(h), int(w), int(c))
                        # fallback: return raw series shape
                        return tuple(int(x) for x in series.shape)
            except Exception as e:
                logging.debug(f"_raw_shape tifffile meta failed: {e}")

        # Non-TIFF or fallback
        try:
            img = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.shape
        except Exception:
            pass

        return self.base_image.shape if self.base_image is not None else None


    def current_display_scale_percent(self):
        """Return current preview scale as a percentage (None if unknown)."""
        img = self.display_image_data
        if img is None:
            return None
        h, w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else (None, None)
        if not h or not w:
            return None
        if getattr(self, "fit_to_window", False):
            vp = self.scroll_area.viewport().size()
            s = min(vp.width()/w, vp.height()/h) if w and h else 1.0
        else:
            s = float(getattr(self, "zoom", 1.0))
        return round(s * 100.0, 1)


    def display_image(self, image):
        """Crisp preview: 1:1 by default; only resample with nearest when zooming or fitting."""
        if image is None:
            self.image_label.setText("No Image Loaded")
            self.image_label.setToolTip("")
            logging.warning("No image to display.")
            return
        try:
            disp = image
            if (disp.dtype != np.uint8) or (disp.ndim not in (2, 3)):
                disp = self.normalize_image_for_display(disp)
                if disp is None:
                    self.image_label.setText("Normalization Failed.")
                    return

            # Ensure 2D or 3-channel for QImage
            if disp.ndim == 3 and disp.shape[2] != 3:
                disp = disp[:, :, :3].copy()

            # QImage needs contiguous data; make it so
            disp = np.ascontiguousarray(disp)

            # Build QImage safely (avoid memoryview by using sip.voidptr)
            if disp.ndim == 2:
                h, w = disp.shape
                try:
                    qimg = QtGui.QImage(sip.voidptr(disp.ctypes.data), w, h, disp.strides[0],
                                        QtGui.QImage.Format_Grayscale8)
                except TypeError:
                    buf = disp.tobytes()
                    qimg = QtGui.QImage(buf, w, h, disp.strides[0], QtGui.QImage.Format_Grayscale8)
            else:
                h, w, _ = disp.shape
                fmt_bgr = getattr(QtGui.QImage, "Format_BGR888", None)
                if fmt_bgr is not None:
                    try:
                        qimg = QtGui.QImage(sip.voidptr(disp.ctypes.data), w, h, disp.strides[0], fmt_bgr)
                    except TypeError:
                        buf = disp.tobytes()
                        qimg = QtGui.QImage(buf, w, h, disp.strides[0], fmt_bgr)
                else:
                    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    disp_rgb = np.ascontiguousarray(disp_rgb)
                    try:
                        qimg = QtGui.QImage(sip.voidptr(disp_rgb.ctypes.data), w, h, disp_rgb.strides[0],
                                            QtGui.QImage.Format_RGB888)
                    except TypeError:
                        buf = disp_rgb.tobytes()
                        qimg = QtGui.QImage(buf, w, h, disp_rgb.strides[0], QtGui.QImage.Format_RGB888)

            qimg = qimg.copy()  # detach so Qt owns the data
            pix = QtGui.QPixmap.fromImage(qimg)

            # ----- scaling policy (unchanged) -----
            fit = bool(getattr(self, "fit_to_window", False))
            z = float(getattr(self, "zoom", 1.0))

            if fit:
                vp = self.scroll_area.viewport().size()
                shown = pix.scaled(vp, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            else:
                if z != 1.0:
                    shown = pix.scaled(int(pix.width()*z), int(pix.height()*z),
                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
                else:
                    shown = pix

            self.image_label.setPixmap(shown)
            self.image_label.resize(shown.size())

            sc = self.current_display_scale_percent()
            if sc is not None:
                self.setWindowTitle(f"Edit Image Viewer - {sc}%")

            logging.debug("Image displayed (crisp).")
        except Exception as e:
            self.image_label.setText("Error Displaying Image")
            logging.error(f"Error during image display: {e}")



    def resizeEvent(self, event):
        # Only recompute fitting when in fit-to-window mode
        if getattr(self, "fit_to_window", True):
            self.display_image(self.display_image_data)
        super().resizeEvent(event)

    def set_fit_to_window(self, enabled: bool):
        """Toggle fit-to-window vs 1:1/zoom rendering."""
        self.fit_to_window = bool(enabled)
        # Reset zoom when switching to fit mode for predictability
        if self.fit_to_window:
            self.zoom = 1.0
        self.display_image(self.display_image_data)

    def set_zoom(self, factor: float):
        """Set a zoom factor (e.g., 0.5, 1.0, 2.0). Automatically switches to 1:1/zoom mode."""
        self.fit_to_window = False
        try:
            f = float(factor)
        except Exception:
            f = 1.0
        self.zoom = max(0.05, f)
        self.display_image(self.display_image_data)

    # ---------- crop interactions ----------
    def _inverse_resize_for_rect(self, x, y, w, h, view_w, view_h):
        """
        If a resize is in the current modifications, convert a rect defined on the *resized view*
        back to the pre-resize coordinate system so the pipeline (crop -> resize) stays aligned.
        Returns (x2, y2, w2, h2, ref_w, ref_h) where ref_* is the pre-resize image size.
        If no resize is present, returns the inputs and (view_w, view_h).
        """
        info = self.modifications.get("resize", None)
        if not info:
            return x, y, w, h, int(view_w), int(view_h)

        if "scale" in info:
            s = max(1, int(info["scale"])) / 100.0
            sx_inv = 1.0 / s
            sy_inv = 1.0 / s
        else:
            pct_w = max(1, int(info.get("width", 100))) / 100.0
            pct_h = max(1, int(info.get("height", 100))) / 100.0
            sx_inv = 1.0 / pct_w
            sy_inv = 1.0 / pct_h

        pre_w = int(round(view_w * sx_inv))
        pre_h = int(round(view_h * sy_inv))

        x2 = int(round(x * sx_inv))
        y2 = int(round(y * sy_inv))
        w2 = int(round(w * sx_inv))
        h2 = int(round(h * sy_inv))

        # clamp to pre-resize bounds
        x2 = max(0, min(x2, max(0, pre_w - 1)))
        y2 = max(0, min(y2, max(0, pre_h - 1)))
        if x2 + w2 > pre_w: w2 = max(0, pre_w - x2)
        if y2 + h2 > pre_h: h2 = max(0, pre_h - y2)

        return x2, y2, w2, h2, pre_w, pre_h

    def apply_crop(self):
        """
        Compose the new crop with any existing crop so that always store ONE
        crop_rect anchored to the base image *after rotation*.
        """
        if self.crop_rect is None or self.original_image is None or self.original_image.size == 0:
            logging.warning("Crop attempted without a selection or with empty image.")
            return

        # 1) Rect currently expressed on the *displayed* image (after old crop+resize)
        view_h, view_w = self.original_image.shape[:2]
        x = int(self.crop_rect.x()); y = int(self.crop_rect.y())
        w = int(self.crop_rect.width()); h = int(self.crop_rect.height())

        # 2) Undo the current resize so rect is relative to the image *after previous crop*, pre-resize
        x2, y2, w2, h2, ref_w_after_prev_crop, ref_h_after_prev_crop = \
            self._inverse_resize_for_rect(x, y, w, h, view_w, view_h)

        # 3) Determine the base-image "source" size after rotation (crop is defined in this space)
        if self.base_image is None:
            logging.warning("No base_image; cannot anchor crop.")
            return
        H0, W0 = self.base_image.shape[:2]
        rot = 0
        try:
            rot = int(self.modifications.get("rotate", 0)) % 360
        except Exception:
            pass
        source_w, source_h = (H0, W0) if rot in (90, 270) else (W0, H0)

        # 4) If a previous crop exists, map it to source size and add offsets (compose crops)
        combined_x, combined_y, combined_w, combined_h = x2, y2, w2, h2
        if "crop_rect" in self.modifications:
            prev = self.modifications.get("crop_rect", {})
            pref = self.modifications.get("crop_rect_ref_size", {"w": source_w, "h": source_h})
            pw = max(1, int(pref.get("w", source_w)))
            ph = max(1, int(pref.get("h", source_h)))
            sx = source_w / float(pw)
            sy = source_h / float(ph)

            ax = int(round(prev.get("x", 0) * sx))
            ay = int(round(prev.get("y", 0) * sy))
            aw = int(round(prev.get("width", 0) * sx))
            ah = int(round(prev.get("height", 0) * sy))

            # compose: new rect inside previous crop -> absolute in source space
            combined_x = ax + x2
            combined_y = ay + y2
            combined_w = w2
            combined_h = h2

        # 5) Clamp to source bounds
        x0 = max(0, min(combined_x, source_w))
        y0 = max(0, min(combined_y, source_h))
        x1 = max(0, min(combined_x + combined_w, source_w))
        y1 = max(0, min(combined_y + combined_h, source_h))
        if x1 <= x0 or y1 <= y0:
            logging.warning("Composed crop is empty/out of bounds; ignoring.")
            self.crop_rect = None
            return

        # 6) Store a single crop anchored to the source (post-rotation) space
        self.modifications["crop_rect"] = {
            "x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)
        }
        self.modifications["crop_rect_ref_size"] = {"w": int(source_w), "h": int(source_h)}
        self.last_crop_rect = QtCore.QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        self.last_crop_ref_size = (int(source_w), int(source_h))

        # Re-render from base for a stable preview
        self.reapply_modifications()
        logging.info(f"Crop applied (source coords): x={x0}, y={y0}, w={x1-x0}, h={y1-y0}, ref={source_w}x{source_h}")
        self.crop_rect = None

    def image_mouse_press_event(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubber_band.show()
            logging.debug("Mouse pressed for cropping.")

    def image_mouse_move_event(self, event):
        if not self.origin.isNull():
            rect = QtCore.QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)
            logging.debug("Mouse moved for cropping.")

    def image_mouse_release_event(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()
            if rect.width() > 10 and rect.height() > 10:
                pixmap = self.image_label.pixmap()
                if pixmap is not None and self.original_image is not None:
                    sp_size = pixmap.size()
                    lbl_size = self.image_label.size()
                    ratio_w = self.original_image.shape[1] / max(1, sp_size.width())
                    ratio_h = self.original_image.shape[0] / max(1, sp_size.height())
                    x_offset = max((lbl_size.width() - sp_size.width()) // 2, 0)
                    y_offset = max((lbl_size.height() - sp_size.height()) // 2, 0)
                    x = rect.x() - x_offset
                    y = rect.y() - y_offset
                    w = rect.width()
                    h = rect.height()
                    x = max(int(x * ratio_w), 0)
                    y = max(int(y * ratio_h), 0)
                    w = min(int(w * ratio_w), self.original_image.shape[1] - x)
                    h = min(int(h * ratio_h), self.original_image.shape[0] - y)
                    self.crop_rect = QtCore.QRect(x, y, w, h)
                    logging.debug(f"Crop rectangle (view coords): {self.crop_rect}")
                    self.apply_crop()
            else:
                logging.debug("Crop selection too small; ignored.")
            self.origin = QtCore.QPoint()

    def on_resize_input_entered(self):
        """Apply proportional resize 1–100% (aspect ratio always preserved)."""
        text = self.resize_input.text().strip()
        if not text:
            return
        try:
            scale = int(text)
        except ValueError:
            logging.warning("Resize skipped: not an integer.")
            return

        # clamp to the same range as the validator
        if not (1 <= scale <= 100):
            logging.warning("Resize percentage must be between 1 and 100.")
            return

        self.modifications["resize"] = {"scale": scale}
        self.reapply_modifications()
        logging.info(f"Image resized to {scale}% of original size.")

    def apply_resize(self, scale_factor_width, scale_factor_height=None):
        """
        Kept for API compatibility called this directly.
        Applies in self.modifications and re-renders instead of mutating arrays.
        """
        if scale_factor_height is None:
            if scale_factor_width <= 0:
                logging.error("Scale factor must be positive.")
                return
            self.modifications["resize"] = {"scale": int(scale_factor_width)}
        else:
            if scale_factor_width <= 0 or scale_factor_height <= 0:
                logging.error("Scale factors must be positive.")
                return
            self.modifications["resize"] = {"width": int(scale_factor_width), "height": int(scale_factor_height)}
        self.reapply_modifications()
        logging.debug("Resizing applied via modifications dict.")

    # ---------- rotation handlers ----------
    def _get_rotation_degrees(self):
        try:
            return int(self.modifications.get("rotate", 0)) % 360
        except Exception:
            return 0

    def on_rotate_left(self):
        deg = (self._get_rotation_degrees() - 90) % 360
        self.modifications["rotate"] = deg
        self.reapply_modifications()

    def on_rotate_right(self):
        deg = (self._get_rotation_degrees() + 90) % 360
        self.modifications["rotate"] = deg
        self.reapply_modifications()

    # ---------- finalize ----------
    def apply_all_changes(self):
        """Save .ax (with chosen scope) and close. Also trigger a centralized refresh like Reset All when needed."""
        hint = self.save_modifications_to_file()  # may be None on error
        if hint and hint.get("scope") == "single":
            self._auto_refresh_after_reset(root_name=hint.get("root_name"))
        self.accept()
        logging.info("All changes applied, modifications saved, and dialog accepted.")

    def get_modified_image(self):
        return self.original_image

    def apply_band_expression(self):
        expression = self.band_input.text().strip()
        if not expression or self.original_image is None:
            logging.warning("Band expression skipped (empty or no image).")
            return

        bands = re.findall(r'b(\d+)', expression)
        unique_bands = sorted(set(bands), key=lambda x: int(x))
        if self.original_image.ndim == 2:
            if any(int(b) != 1 for b in unique_bands):
                logging.warning("Grayscale image only has one band (b1).")
                return
        elif self.original_image.ndim == 3:
            num_bands = self.original_image.shape[2]
            for b in unique_bands:
                if int(b) < 1 or int(b) > num_bands:
                    logging.warning(f"Band b{b} does not exist in the image.")
                    return
        else:
            logging.warning("Unsupported image format for band expression.")
            return

        self.modifications["band_expression"] = expression
        self.reapply_modifications()
        logging.info("Band expression applied and image updated.")

    # ---------- display normalization ----------
    def _percentiles_from_sample(self, arr_flat):
        """Compute low/high percentiles from a 1D sample."""
        lo = np.percentile(arr_flat, self.STRETCH_LOW_P)
        hi = np.percentile(arr_flat, self.STRETCH_HIGH_P)
        return float(lo), float(hi)

    def _sample_for_stats(self, arr):
        """Return a downsampled view for percentile stats to keep UI responsive."""
        h, w = arr.shape[:2]
        m = max(h, w)
        if m <= self.STRETCH_SAMPLE_MAX:
            return arr
        scale = self.STRETCH_SAMPLE_MAX / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def normalize_image_for_display(self, image):
        """
        Normalize to 8-bit *for display only* using percentile stretch.
        - 2–98% (configurable)
        - per-channel for color (configurable)
        - works with float/uint16/uint8
        """
        if image is None:
            logging.warning("No image provided for normalization.")
            return None
        try:
            if image.dtype == np.uint8:
                # Already display-ready
                norm_img = image
            else:
                res = image.astype(np.float32, copy=False)
                res = np.nan_to_num(res)

                if res.ndim == 2:  # grayscale
                    sample = self._sample_for_stats(res)
                    lo, hi = self._percentiles_from_sample(sample.reshape(-1))
                    if hi <= lo:
                        norm = np.full(res.shape, 0.5, dtype=np.float32)  # flat mid-gray
                    else:
                        norm = (res - lo) / max(hi - lo, 1e-12)
                    if self.STRETCH_CLIP:
                        norm = np.clip(norm, 0.0, 1.0)
                    norm_img = (norm * 255.0).astype(np.uint8)

                elif res.ndim == 3:  # color or multispectral
                    H, W, C = res.shape
                    # choose channels for preview
                    if C >= 3:
                        x = res[:, :, :3]  # preview first 3 channels
                    else:
                        x = res  # C==1 handled above; C==2, show first 2 + duplicate third
                    sample = self._sample_for_stats(x)

                    if self.STRETCH_PER_CHANNEL:
                        # per-channel percentiles
                        flat = sample.reshape(-1, x.shape[2])
                        lo = np.percentile(flat, self.STRETCH_LOW_P, axis=0)
                        hi = np.percentile(flat, self.STRETCH_HIGH_P, axis=0)
                        scale = np.maximum(hi - lo, 1e-12)
                        norm = (x - lo.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
                    else:
                        lo, hi = self._percentiles_from_sample(sample.reshape(-1))
                        if hi <= lo:
                            norm = np.full(x.shape, 0.5, dtype=np.float32)
                        else:
                            norm = (x - lo) / max(hi - lo, 1e-12)

                    if self.STRETCH_CLIP:
                        norm = np.clip(norm, 0.0, 1.0)

                    norm_img = (norm * 255.0).astype(np.uint8)

                    # If fewer than 3 channels, pad to 3 for preview
                    if norm_img.ndim == 3 and norm_img.shape[2] == 2:
                        third = norm_img[:, :, :1]
                        norm_img = np.concatenate([norm_img, third], axis=2)

                else:
                    logging.error("Unsupported Image Format in normalization.")
                    return None

            logging.debug(f"Normalized Image: Shape {norm_img.shape}, Dtype {norm_img.dtype}")
            return norm_img

        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            return None

    def reapply_modifications(self):
        if self.base_image is None:
            return
        mod_image = self.apply_all_modifications_to_image(self.base_image, self.modifications)
        self.original_image = mod_image
        self.display_image_data = self.normalize_image_for_display(mod_image)
        self.display_image(self.display_image_data)

