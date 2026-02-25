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
from .utils import process_band_expression_float, normalize_band_expr
from .utils import eval_band_expression 

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

# Try importing pystackreg
try:
    from pystackreg import StackReg
    HAS_PYSTACKREG = True
except ImportError:
    HAS_PYSTACKREG = False


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        
        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toggle_button.toggled.connect(self._on_toggled)

        self.content_area = QtWidgets.QWidget()
        self.content_area.setVisible(False)
        
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)
        
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)

    def _on_toggled(self, checked):
        self.toggle_button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content_area.setVisible(checked)

class ImageEditorDialog(QDialog):
    # signal to let parent optionally react when group/all-group mods are saved
    modificationsAppliedToGroup = QtCore.pyqtSignal(str)

    # -------------------- configurable stretch knobs --------------------
    STRETCH_LOW_P = 0.5
    STRETCH_HIGH_P = 99.5
    STRETCH_PER_CHANNEL = True
    STRETCH_CLIP = True
    STRETCH_SAMPLE_MAX = 250   # compute percentiles on a downsampled copy for speed
    HIST_MATCH_FAST = True                 # master toggle
    HIST_MATCH_AFTER_RESIZE_IF_SHRINK = True
    HIST_SAMPLE_MAX = 300                 # longest side used to ESTIMATE source CDF
    HIST_BINS_8U = 256
    HIST_BINS_16U = 4096                   # 12-bit quantization
    HIST_BINS_FLOAT = 2048



    def __init__(self, parent=None, image_data=None, image_filepath=""):
        import os, numpy as np, logging
        super().__init__(parent)
        self.setWindowTitle("Edit Image Viewer - Crop")

        # project folder (so .ax files go there if available)
        self.project_folder = getattr(parent, "project_folder", None)

        # determine filepath
        self.image_filepath = image_filepath or (
            image_data.filepath if (hasattr(image_data, "filepath") and image_data.filepath) else ""
        )

        # resolve raw image (cache → raw_image_cache → export cache → disk)
        parent = self.parent()
        raw = None
        
        # 1. Try legacy _raw_cache first
        try:
            cache = getattr(parent, "_raw_cache", None)
            if cache:
                key = os.path.normcase(os.path.abspath(self.image_filepath))
                raw = cache.get(key)
        except Exception:
            pass
        
        # 2. Try new _raw_image_cache (PERFORMANCE: skips disk I/O on editor re-open)
        if raw is None:
            try:
                cache = getattr(parent, "_raw_image_cache", None)
                lock = getattr(parent, "_raw_image_cache_lock", None)
                if cache is not None and lock is not None and self.image_filepath:
                    fp_norm = os.path.normcase(os.path.abspath(self.image_filepath))
                    file_mtime = os.path.getmtime(self.image_filepath)
                    cache_key = (fp_norm, file_mtime)
                    with lock:
                        if cache_key in cache:
                            raw = cache[cache_key]
                            logging.debug(f"[Editor] Raw image from cache: {os.path.basename(self.image_filepath)}")
            except Exception:
                pass
        
        # 3. Try _export_cache
        if raw is None:
            try:
                cache = getattr(parent, "_export_cache", None)
                lock  = getattr(parent, "_export_cache_lock", None)
                if cache and lock:
                    with lock:
                        tup = cache.get(self.image_filepath)
                        if tup and tup[0] is not None:
                            raw = tup[0]
            except Exception:
                pass
        
        # 4. Load from disk and store in cache for future opens
        if raw is None:
            raw = self._load_raw_image()
            # Store in parent's raw image cache for future editor opens
            try:
                if raw is not None and parent is not None and self.image_filepath:
                    cache = getattr(parent, "_raw_image_cache", None)
                    lock = getattr(parent, "_raw_image_cache_lock", None)
                    if cache is not None and lock is not None:
                        fp_norm = os.path.normcase(os.path.abspath(self.image_filepath))
                        file_mtime = os.path.getmtime(self.image_filepath)
                        cache_key = (fp_norm, file_mtime)
                        with lock:
                            # LRU eviction
                            max_size = getattr(parent, "_raw_image_cache_max", 50)
                            if len(cache) >= max_size:
                                oldest = next(iter(cache))
                                del cache[oldest]
                            cache[cache_key] = raw
                            logging.debug(f"[Editor] Raw image cached: {os.path.basename(self.image_filepath)} (cache size: {len(cache)})")
            except Exception:
                pass

        self.base_image = (raw if raw is not None
                           else (image_data.copy() if isinstance(image_data, np.ndarray) else None))

        # If launched from a viewer that already has the edited pixels in memory,
        # keep them as a fast preview for instant dialog open (we still keep base_image=RAW).
        self._fast_open_preview = None
        try:
            if raw is not None and isinstance(image_data, np.ndarray) and getattr(image_data, "size", 0) > 0:
                self._fast_open_preview = image_data
        except Exception:
            self._fast_open_preview = None

        # initial state
        self.original_image = self.base_image if self.base_image is not None else None
        self.display_image_data = None
        self.crop_rect = None
        self.last_crop_rect = None
        self.last_crop_ref_size = None
        self.modifications = {}
        self.last_band_float_result = None  # For band expressions
        self._classification_result = None   # Separate storage for classification

        # load mods (BEFORE building UI so init_ui can reflect them)
        loaded = self.load_modifications_from_file()
        if loaded:
            self.modifications.update(loaded)
            if "orig_size" not in self.modifications:
                sh = self._raw_shape()
                if sh:
                    h, w = sh[:2]
                    c = 1 if len(sh) == 2 else int(sh[2])
                    self.modifications["orig_size"] = {"h": int(h), "w": int(w), "c": int(c)}
                    self.modifications["anchor_to_original"] = True
        self._ax_loaded_on_open = bool(loaded)
        # NOTE: do NOT pop 'classification' here; we want it to persist.

        # ui
        self.init_ui()
        self.setMinimumSize(800, 600)
        self.fit_to_window = True
        self.zoom = 1.0

        # sync histogram combo with loaded mods (now that widgets exist)
        self._sync_hist_combo_from_mods()

        # cache for histogram CDF mappings
        self._hist_cache = {}  # {(mode, ref_hash, dtype_tag, bins): (xs, xprime)}
        self._cls_snapshot = None  # tracks geometry when cls was computed

    def _sync_hist_combo_from_mods(self):
        idx = 0
        try:
            hm = (self.modifications or {}).get("hist_match", None)
            mode = ""
            if isinstance(hm, dict):
                mode = (hm.get("mode") or "").lower()
            elif hm is None:
                mode = "none"
            if mode == "meanstd":
                idx = 1
            elif mode == "cdf":
                idx = 2
            else:
                idx = 0
        except Exception:
            idx = 0
        if hasattr(self, "hist_mode_combo"):
            self.hist_mode_combo.blockSignals(True)
            self.hist_mode_combo.setCurrentIndex(idx)
            self.hist_mode_combo.blockSignals(False)

    def _ref_hash_hist(self, block):
        """Hash the reference payload (mean/std or CDF) so we can cache mappings."""
        import hashlib, json
        mode = (block.get("mode") or "meanstd").lower()
        if mode == "meanstd":
            key = ["meanstd", block.get("bands"), block.get("ref_stats")]
        else:
            key = ["cdf", block.get("bands"), (block.get("ref_cdf") or {}).get("per_band")]
        s = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8", "ignore")
        return hashlib.sha1(s).hexdigest()

    def _cdf_from_hist(self, flat, lo, hi, bins):
        """Build (xs, cdf) from histogram instead of sorting every pixel."""
        import numpy as np
        if hi <= lo:
            hi = lo + 1.0
        hist, edges = np.histogram(flat, bins=int(bins), range=(lo, hi))
        cdf = np.cumsum(hist, dtype=np.float32) / max(1, flat.size)
        xs = 0.5 * (edges[:-1] + edges[1:])  # bin centers
        return xs.astype(np.float32, copy=False), cdf.astype(np.float32, copy=False)

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

    def _get_ax_candidates(self, image_path: str) -> list:
        """Return all potential .ax file paths (Project Folder and Sidecar) for an image."""
        candidates = []
        # 1. Project-level path
        if self.project_folder and self.project_folder.strip():
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            candidates.append(os.path.join(self.project_folder, base_name + ".ax"))
        
        # 2. Sidecar path (always check)
        candidates.append(os.path.splitext(image_path)[0] + ".ax")
        
        # Unique list, preserving order
        return list(dict.fromkeys(candidates))

    def _parse_nodata_values(self, text: str):
        """
        Parse comma-separated NoData values from user input.
        Supports numeric literals AND boolean expressions (b1<123, b2>189, etc.).
        Returns list of (float | str), where strings are threshold expressions.
        """
        from .utils import parse_nodata_text
        return parse_nodata_text(text)

    def _on_nodata_enabled_changed(self, state):
        """Handle NoData enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"NoData masking {'enabled' if enabled else 'disabled'}")
        # Trigger preview update
        self.reapply_modifications()

    def _on_nodata_input_changed(self):
        """Handle NoData input text change (when user presses Enter or focus is lost)."""
        nodata_text = self.nodata_input.text().strip()
        nodata_values = self._parse_nodata_values(nodata_text)
        logging.info(f"[NoData Apply] NoData values changed: {nodata_values}")
        # Update modifications and trigger preview
        self.modifications["nodata_values"] = nodata_values
        self.modifications["nodata_enabled"] = True  # Ensure enabled when applying
        logging.info(f"[NoData Apply] Modifications updated, calling reapply_modifications()")
        self.reapply_modifications()

    def _on_nodata_pick_clicked(self):
        """Toggle NoData picker mode."""
        if not hasattr(self, '_nodata_picker_active'):
            self._nodata_picker_active = False
        
        self._nodata_picker_active = not self._nodata_picker_active
        self.nodata_pick_btn.setChecked(self._nodata_picker_active)
        
        if self._nodata_picker_active:
            # Enter picker mode - change cursor and show message
            self.image_label.setCursor(QtCore.Qt.CrossCursor)
            self.nodata_pick_btn.setStyleSheet("background-color: #ffcccc;")
            # Store that cropping is temporarily disabled
            self._crop_disabled_for_pick = True
            logging.info("NoData picker mode: Click on pixels to add their values as NoData. Click 'Pick' again or Escape to exit.")
        else:
            # Exit picker mode
            self._exit_nodata_picker_mode()

    def _exit_nodata_picker_mode(self):
        """Exit NoData picker mode and restore normal behavior."""
        self._nodata_picker_active = False
        self._crop_disabled_for_pick = False
        self.nodata_pick_btn.setChecked(False)
        self.image_label.setCursor(QtCore.Qt.ArrowCursor)
        self.nodata_pick_btn.setStyleSheet("")
        logging.info("NoData picker mode disabled.")

    # ==================== Modification Enabled Handlers ====================
    
    def _on_resize_enabled_changed(self, state):
        """Handle Resize enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Resize {'enabled' if enabled else 'disabled'}")
        self.reapply_modifications()

    def _on_rotate_enabled_changed(self, state):
        """Handle Rotate enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Rotate {'enabled' if enabled else 'disabled'}")
        self.reapply_modifications()

    def _on_crop_enabled_changed(self, state):
        """Handle Crop enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Crop {'enabled' if enabled else 'disabled'}")
        self.reapply_modifications()

    def _on_reg_mode_changed(self, index):
        self.reapply_modifications()

    def _on_reg_enabled_changed(self, state):
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Registration {'enabled' if enabled else 'disabled'} (Mode: {self.reg_mode_combo.currentText()})")
        self.reapply_modifications()

    def _on_band_enabled_changed(self, state):
        """Handle Band expression enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Band expression {'enabled' if enabled else 'disabled'}")
        self.reapply_modifications()
        # FIX: Update append button state when band enabled checkbox changes
        self._update_append_button_state()

    def _on_hist_enabled_changed(self, state):
        """Handle Histogram enabled checkbox state change."""
        enabled = (state == QtCore.Qt.Checked)
        logging.info(f"Histogram normalization {'enabled' if enabled else 'disabled'}")
        self.reapply_modifications()

    def _calculate_result_size_from_mods(self, mods):
        """
        Calculate the expected result image size from modifications WITHOUT processing the image.
        This is a lightweight calculation for UI display purposes.
        
        Order of operations (matching apply_all_modifications_to_image):
        1. Start with original size
        2. Apply crop (if enabled)
        3. Apply rotation (swaps W/H for 90/270)
        4. Apply resize (if enabled)
        
        Returns: (width, height) or (None, None) if cannot determine
        """
        try:
            # Get original size
            orig = mods.get("orig_size") or {}
            w = orig.get("w")
            h = orig.get("h")
            
            # Fallback to base_image if orig_size not stored
            if not w or not h:
                if self.base_image is not None and hasattr(self.base_image, "shape"):
                    h, w = self.base_image.shape[:2]
                else:
                    return None, None
            
            w, h = int(w), int(h)
            
            # Check enabled flags
            crop_enabled = mods.get("crop_enabled", True)
            rotate_enabled = mods.get("rotate_enabled", True)
            resize_enabled = mods.get("resize_enabled", True)
            
            # Get rotation value
            rot = 0
            if rotate_enabled:
                try:
                    rot = int(mods.get("rotate", 0)) % 360
                except:
                    rot = 0
            
            # Determine crop/rotate order based on crop_rect_ref_size
            # (Same logic as apply_all_modifications_to_image)
            crop_rect = mods.get("crop_rect") if crop_enabled else None
            crop_ref = mods.get("crop_rect_ref_size") or {}
            
            do_rotate_first = True  # Default
            if crop_rect and rot in (90, 180, 270):
                ref_w = int(crop_ref.get("w", 0)) or 0
                ref_h = int(crop_ref.get("h", 0)) or 0
                rotated_w, rotated_h = (h, w) if rot in (90, 270) else (w, h)
                if ref_w > 0 and ref_h > 0:
                    if (ref_w, ref_h) == (w, h):
                        do_rotate_first = False  # Crop before rotate
                    elif (ref_w, ref_h) == (rotated_w, rotated_h):
                        do_rotate_first = True   # Rotate before crop
            
            # Apply operations in correct order
            if do_rotate_first:
                # Rotate first
                if rot in (90, 270):
                    w, h = h, w
                # Then crop
                if crop_rect and crop_enabled:
                    cw = int(crop_rect.get("width", 0)) or 0
                    ch = int(crop_rect.get("height", 0)) or 0
                    if cw > 0 and ch > 0:
                        # Scale crop to current dimensions if ref differs
                        ref_w = int(crop_ref.get("w", w)) or w
                        ref_h = int(crop_ref.get("h", h)) or h
                        if ref_w != w or ref_h != h:
                            cw = int(round(cw * w / float(ref_w)))
                            ch = int(round(ch * h / float(ref_h)))
                        w, h = cw, ch
            else:
                # Crop first
                if crop_rect and crop_enabled:
                    cw = int(crop_rect.get("width", 0)) or 0
                    ch = int(crop_rect.get("height", 0)) or 0
                    if cw > 0 and ch > 0:
                        ref_w = int(crop_ref.get("w", w)) or w
                        ref_h = int(crop_ref.get("h", h)) or h
                        if ref_w != w or ref_h != h:
                            cw = int(round(cw * w / float(ref_w)))
                            ch = int(round(ch * h / float(ref_h)))
                        w, h = cw, ch
                # Then rotate
                if rot in (90, 270):
                    w, h = h, w
            
            # Apply resize (if enabled)
            if resize_enabled:
                resize = mods.get("resize") or {}
                if resize:
                    if "px_w" in resize and "px_h" in resize:
                        pw = int(resize.get("px_w", 0))
                        ph = int(resize.get("px_h", 0))
                        if pw > 0 and ph > 0:
                            w, h = pw, ph
                    elif "scale" in resize:
                        scale = max(1, int(resize.get("scale", 100))) / 100.0
                        w = int(round(w * scale))
                        h = int(round(h * scale))
                    elif "width" in resize and "height" in resize:
                        pct_w = int(resize.get("width", 100)) / 100.0
                        pct_h = int(resize.get("height", 100)) / 100.0
                        w = int(round(w * pct_w))
                        h = int(round(h * pct_h))
            
            return w, h
            
        except Exception as e:
            logging.debug(f"_calculate_result_size_from_mods failed: {e}")
            return None, None

    def _sync_ui_from_modifications(self):
        """
        Sync all UI controls from self.modifications.
        Called after loading .ax file to populate UI with cached values.
        """
        mods = self.modifications or {}
        
        # --- NoData ---
        try:
            nd_vals = mods.get("nodata_values", [])
            if hasattr(self, "nodata_input"):
                if nd_vals:
                    self.nodata_input.setText(", ".join(str(v) for v in nd_vals))
                else:
                    self.nodata_input.clear()
            nd_enabled = mods.get("nodata_enabled", True)
            if hasattr(self, "nodata_enabled_checkbox"):
                self.nodata_enabled_checkbox.setChecked(nd_enabled)
        except Exception:
            pass
        
        # --- Resize ---
        try:
            resize = mods.get("resize", {})
            if hasattr(self, "resize_input"):
                scale = resize.get("scale")
                if scale:
                    self.resize_input.setText(str(scale))
                else:
                    self.resize_input.clear()
            
            if hasattr(self, "resize_width_input") and hasattr(self, "resize_height_input"):
                # CRITICAL FIX: Calculate expected result size from modifications
                # This ensures we show the size AFTER crop/rotate/resize, not the original
                target_w, target_h = self._calculate_result_size_from_mods(mods)

                if target_w: self.resize_width_input.setText(str(target_w))
                else:        self.resize_width_input.clear()
                if target_h: self.resize_height_input.setText(str(target_h))
                else:        self.resize_height_input.clear()

            resize_enabled = mods.get("resize_enabled", True)
            if hasattr(self, "resize_enabled_checkbox"):
                self.resize_enabled_checkbox.setChecked(resize_enabled)
        except Exception:
            pass
        
        # --- Rotate ---
        try:
            # rotate = mods.get("rotate", 0) # Rotate value is stored but not directly shown in a textbox
            rotate_enabled = mods.get("rotate_enabled", True)
            if hasattr(self, "rotate_enabled_checkbox"):
                self.rotate_enabled_checkbox.setChecked(rotate_enabled)
        except Exception:
            pass
        
        # --- Crop ---
        try:
            crop_enabled = mods.get("crop_enabled", True)
            if hasattr(self, "crop_enabled_checkbox"):
                self.crop_enabled_checkbox.setChecked(crop_enabled)
        except Exception:
            pass
        
        # --- Band Expression ---
        try:
            band_expr = mods.get("band_expression", "")
            if hasattr(self, "band_input"):
                if band_expr:
                    self.band_input.setText(str(band_expr))
                else:
                    self.band_input.clear()
            band_enabled = mods.get("band_enabled", True)
            if hasattr(self, "band_enabled_checkbox"):
                self.band_enabled_checkbox.setChecked(band_enabled)
        except Exception:
            pass
        
        # --- Histogram ---
        try:
            # hm = mods.get("hist_match", {})
            hist_enabled = mods.get("hist_enabled", True)
            if hasattr(self, "hist_enabled_checkbox"):
                self.hist_enabled_checkbox.setChecked(hist_enabled)
            
            # Sync combo index
            # Default to "Fast (Mean/Std)" [index 1] or "Original" [index 0]? 
            # Original code in init defaults to 0 if missing.
            mode = ""
            hm = mods.get("hist_match")
            if isinstance(hm, dict):
                mode = (hm.get("mode") or "").lower()
            elif hm is None:
                mode = "none"
            
            idx = 0
            if mode == "meanstd": idx = 1
            elif mode == "cdf":   idx = 2
            
            if hasattr(self, "hist_mode_combo"):
                self.hist_mode_combo.blockSignals(True)
                self.hist_mode_combo.setCurrentIndex(idx)
                self.hist_mode_combo.blockSignals(False)

        except Exception:
            pass
        
        # --- Registration ---
        try:
             hm = mods.get("registration", {})
             enabled = hm.get("enabled", False)
             if HAS_PYSTACKREG and hasattr(self, "reg_enabled_checkbox"):
                 self.reg_enabled_checkbox.setChecked(enabled)
                 if enabled:
                     mode = hm.get("mode", "None")
                     if hasattr(self, "reg_mode_combo"):
                         self.reg_mode_combo.setCurrentText(mode)
        except Exception:
             pass

        # --- Classification ---
        try:
            clf = mods.get("classification", {})
            if bool(clf.get("enabled", False)) and hasattr(self, "use_sklearn_checkbox"):
                self.use_sklearn_checkbox.setChecked(True)
                if hasattr(self, "classify_btn"):
                    self.classify_btn.setEnabled(True)
        except Exception:
            pass

    def _pick_nodata_at_pos(self, pos):
        """
        Pick NoData value from pixel at given position in the image label.
        Adds the pixel value(s) to the nodata_input field.
        """
        if self.original_image is None:
            return
        
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        
        # Calculate the offset (image is centered in label)
        lbl_size = self.image_label.size()
        pm_size = pixmap.size()
        x_offset = max((lbl_size.width() - pm_size.width()) // 2, 0)
        y_offset = max((lbl_size.height() - pm_size.height()) // 2, 0)
        
        # Get position relative to pixmap
        px_x = pos.x() - x_offset
        px_y = pos.y() - y_offset
        
        # Check bounds
        if px_x < 0 or px_y < 0 or px_x >= pm_size.width() or px_y >= pm_size.height():
            return
        
        # Scale to original image coordinates
        # Use the currently modified image if available
        img = getattr(self, 'display_image_data', None)
        if img is None:
            img = self.original_image
        if img is None:
            return
        
        img_h, img_w = img.shape[:2]
        ratio_x = img_w / max(1, pm_size.width())
        ratio_y = img_h / max(1, pm_size.height())
        
        orig_x = int(px_x * ratio_x)
        orig_y = int(px_y * ratio_y)
        
        # Clamp to image bounds
        orig_x = max(0, min(orig_x, img_w - 1))
        orig_y = max(0, min(orig_y, img_h - 1))
        
        # Get pixel value(s)
        try:
            if img.ndim == 2:
                pixel_val = img[orig_y, orig_x]
                new_vals = [pixel_val]
            else:
                # For multi-channel, get all channel values
                pixel_vals = img[orig_y, orig_x]
                # Use the first channel value as representative, or average
                # For NoData, typically all channels have the same value
                new_vals = list(set(pixel_vals.flatten().tolist()))
            
            # Get existing values
            existing_text = self.nodata_input.text().strip()
            existing_vals = self._parse_nodata_values(existing_text)
            
            # Add new values (avoid duplicates)
            added = []
            for v in new_vals:
                # Round to reasonable precision for floats
                if isinstance(v, float):
                    v = round(v, 6)
                if v not in existing_vals:
                    existing_vals.append(v)
                    added.append(v)
            
            if added:
                # Update the input field
                self.nodata_input.setText(", ".join(str(v) for v in existing_vals))
                logging.info(f"Added NoData value(s): {added} at pixel ({orig_x}, {orig_y})")
                
                # Make sure checkbox is checked
                self.nodata_enabled_checkbox.setChecked(True)
            else:
                logging.info(f"Value(s) {new_vals} already in NoData list")
                
        except Exception as e:
            logging.error(f"Failed to pick NoData value: {e}")

    def _get_effective_nodata_values(self):
        """
        Get the NoData values to use, considering the enabled checkbox.
        Returns empty list if NoData is disabled.
        """
        if not hasattr(self, 'nodata_enabled_checkbox') or not self.nodata_enabled_checkbox.isChecked():
            return []
        
        if hasattr(self, 'nodata_input'):
            return self._parse_nodata_values(self.nodata_input.text().strip())
        return []

    # ==================== MASK POLYGON METHODS ====================
    
    def _get_all_polygon_names_in_project(self):
        """Get all unique polygon names across all images in the project."""
        import os
        parent = self.parent()
        if parent is None or not hasattr(parent, 'all_polygons'):
            return set()
        
        all_names = set()
        all_polygons = getattr(parent, 'all_polygons', {}) or {}
        
        for group_name, file_map in all_polygons.items():
            for stored_fp, poly_data in file_map.items():
                if isinstance(poly_data, dict):
                    poly_name = poly_data.get('name', group_name)
                    points = poly_data.get('points', [])
                    if points and len(points) >= 3:
                        all_names.add(poly_name)
                else:
                    all_names.add(group_name)
        
        return all_names
    
    def _get_polygon_names_for_current_image(self):
        """Get polygon names available for the current image."""
        import os
        parent = self.parent()
        if parent is None or not hasattr(parent, 'all_polygons'):
            return set()
        
        fp = getattr(self, 'image_filepath', None)
        if not fp:
            return set()
        
        fp_norm = os.path.normpath(fp).lower()
        names = set()
        all_polygons = getattr(parent, 'all_polygons', {}) or {}
        
        for group_name, file_map in all_polygons.items():
            for stored_fp, poly_data in file_map.items():
                stored_fp_norm = os.path.normpath(stored_fp).lower() if stored_fp else ""
                if stored_fp_norm == fp_norm:
                    if isinstance(poly_data, dict):
                        poly_name = poly_data.get('name', group_name)
                        points = poly_data.get('points', [])
                        if points and len(points) >= 3:
                            names.add(poly_name)
        
        return names
    
    def _populate_mask_polygon_menu(self):
        """Populate the mask polygon menu with checkable polygon names."""
        if not hasattr(self, 'mask_polygon_menu'):
            return
        
        # Get currently selected names from modifications
        mods = self.modifications or {}
        mask_poly = mods.get('mask_polygon', {}) or {}
        selected_names = set(mask_poly.get('names', []) if isinstance(mask_poly, dict) else [])
        
        # Clear menu
        self.mask_polygon_menu.clear()
        
        # Get all unique polygon names in project (for Apply to Root/All Roots)
        all_names = self._get_all_polygon_names_in_project()
        current_image_names = self._get_polygon_names_for_current_image()
        
        if not all_names:
            action = self.mask_polygon_menu.addAction("(No polygons found)")
            action.setEnabled(False)
            self._update_mask_polygon_count_label()
            return
        
        # Add "Select All" and "Clear All" options
        select_all_action = self.mask_polygon_menu.addAction("✓ Select All")
        select_all_action.triggered.connect(self._on_mask_polygon_select_all)
        clear_all_action = self.mask_polygon_menu.addAction("✗ Clear All")
        clear_all_action.triggered.connect(self._on_mask_polygon_clear_all)
        self.mask_polygon_menu.addSeparator()
        
        # Add checkable actions for each polygon name
        for name in sorted(all_names, key=str.lower):
            action = self.mask_polygon_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name in selected_names)
            # Mark if available on current image
            if name not in current_image_names:
                action.setText(f"{name} (other images)")
            action.triggered.connect(lambda checked, n=name: self._on_mask_polygon_toggled(n, checked))
        
        self._update_mask_polygon_count_label()
    
    def _update_mask_polygon_count_label(self):
        """Update the label showing how many polygons are selected."""
        if not hasattr(self, 'mask_polygon_count_label'):
            return
        
        mods = self.modifications or {}
        mask_poly = mods.get('mask_polygon', {}) or {}
        names = mask_poly.get('names', []) if isinstance(mask_poly, dict) else []
        count = len(names)
        
        if count == 0:
            self.mask_polygon_count_label.setText("(0 selected)")
            self.mask_polygon_count_label.setStyleSheet("color: gray; font-size: 10px;")
        else:
            self.mask_polygon_count_label.setText(f"({count} selected)")
            self.mask_polygon_count_label.setStyleSheet("color: green; font-size: 10px;")
    
    def _on_mask_polygon_toggled(self, name, checked):
        """Handle toggling a polygon name in the menu."""
        mods = self.modifications
        mask_poly = mods.get('mask_polygon', {}) or {}
        if not isinstance(mask_poly, dict):
            mask_poly = {}
        
        names = list(mask_poly.get('names', []))
        
        if checked and name not in names:
            names.append(name)
        elif not checked and name in names:
            names.remove(name)
        
        # Update checkbox state BEFORE setting mods (so enabled reflects correct state)
        if names and not self.mask_polygon_enabled_checkbox.isChecked():
            self.mask_polygon_enabled_checkbox.blockSignals(True)  # Prevent double reapply
            self.mask_polygon_enabled_checkbox.setChecked(True)
            self.mask_polygon_enabled_checkbox.blockSignals(False)
        
        enabled = len(names) > 0 and self.mask_polygon_enabled_checkbox.isChecked()
        mods['mask_polygon'] = {
            'enabled': enabled,
            'names': names
        }
        
        # Clear caches when mask selection changes - this forces recalculation of stats
        try:
            parent = self.parent()
            if parent:
                if hasattr(parent, "_scene_stats_cache") and parent._scene_stats_cache:
                    parent._scene_stats_cache.clear()
                if hasattr(parent, "_export_cache") and parent._export_cache:
                    parent._export_cache.clear()
        except Exception:
            pass
        
        self._update_mask_polygon_count_label()
        self._update_viewer_polygon_mask_status(names if enabled else [])
        self.reapply_modifications()
    
    def _on_mask_polygon_select_all(self):
        """Select all polygon names."""
        all_names = list(self._get_all_polygon_names_in_project())
        
        self.modifications['mask_polygon'] = {
            'enabled': True,
            'names': all_names
        }
        self.mask_polygon_enabled_checkbox.setChecked(True)
        
        # Update menu checkboxes
        for action in self.mask_polygon_menu.actions():
            if action.isCheckable():
                action.setChecked(True)
        
        # Clear caches
        try:
            parent = self.parent()
            if parent:
                if hasattr(parent, "_scene_stats_cache") and parent._scene_stats_cache:
                    parent._scene_stats_cache.clear()
                if hasattr(parent, "_export_cache") and parent._export_cache:
                    parent._export_cache.clear()
        except Exception:
            pass
        
        self._update_mask_polygon_count_label()
        self._update_viewer_polygon_mask_status(all_names)
        self.reapply_modifications()
    
    def _on_mask_polygon_clear_all(self):
        """Clear all polygon selections."""
        self.modifications['mask_polygon'] = {
            'enabled': False,
            'names': []
        }
        self.mask_polygon_enabled_checkbox.setChecked(False)
        
        # Update menu checkboxes
        for action in self.mask_polygon_menu.actions():
            if action.isCheckable():
                action.setChecked(False)
        
        # Clear caches
        try:
            parent = self.parent()
            if parent:
                if hasattr(parent, "_scene_stats_cache") and parent._scene_stats_cache:
                    parent._scene_stats_cache.clear()
                if hasattr(parent, "_export_cache") and parent._export_cache:
                    parent._export_cache.clear()
        except Exception:
            pass
        
        self._update_mask_polygon_count_label()
        self._update_viewer_polygon_mask_status([])
        self.reapply_modifications()

    def _update_viewer_polygon_mask_status(self, mask_names):
        """Update the bound viewer's polygon mask status."""
        try:
            viewer = getattr(self, 'bound_viewer', None) or getattr(self, 'viewer', None)
            if viewer and hasattr(viewer, 'update_polygon_mask_status'):
                viewer.update_polygon_mask_status(mask_names)
        except Exception as e:
            logging.debug(f"[_update_viewer_polygon_mask_status] Error: {e}")

    def _sync_mask_polygon_from_mods(self):
        """Load mask polygon settings from modifications."""
        if not hasattr(self, 'mask_polygon_menu') or not hasattr(self, 'mask_polygon_enabled_checkbox'):
            return
        
        mods = self.modifications or {}
        mask_poly = mods.get('mask_polygon', {})
        
        if mask_poly and isinstance(mask_poly, dict):
            enabled = mask_poly.get('enabled', False)
            names = mask_poly.get('names', [])
            
            # Handle legacy single-name format
            if not names and mask_poly.get('name'):
                names = [mask_poly.get('name')]
                # Migrate to new format
                mods['mask_polygon'] = {'enabled': enabled, 'names': names}
            
            self.mask_polygon_enabled_checkbox.setChecked(enabled)
            
            # Update menu checkboxes
            for action in self.mask_polygon_menu.actions():
                if action.isCheckable():
                    # Extract name from action text (remove " (other images)" suffix)
                    action_name = action.text().replace(" (other images)", "")
                    action.setChecked(action_name in names)
        else:
            self.mask_polygon_enabled_checkbox.setChecked(False)
        
        self._update_mask_polygon_count_label()

    def _on_mask_polygon_enabled_changed(self, state):
        """Handle mask polygon enabled checkbox change."""
        enabled = bool(state)
        
        mods = self.modifications
        mask_poly = mods.get('mask_polygon', {}) or {}
        if isinstance(mask_poly, dict):
            mask_poly['enabled'] = enabled
            mods['mask_polygon'] = mask_poly
            
            # Update viewer polygon mask status
            if enabled:
                names = mask_poly.get('names', []) or []
                self._update_viewer_polygon_mask_status(names)
            else:
                self._update_viewer_polygon_mask_status([])
        
        # Clear caches when mask is toggled - this forces recalculation of stats
        try:
            parent = self.parent()
            if parent:
                if hasattr(parent, "_scene_stats_cache") and parent._scene_stats_cache:
                    parent._scene_stats_cache.clear()
                    logging.debug("[ImageEditorDialog] Cleared scene_stats_cache after mask toggle")
                if hasattr(parent, "_export_cache") and parent._export_cache:
                    parent._export_cache.clear()
                    logging.debug("[ImageEditorDialog] Cleared export_cache after mask toggle")
        except Exception as e:
            logging.debug(f"[ImageEditorDialog] Failed to clear caches: {e}")
        
        self.reapply_modifications()

    def _get_mask_polygon_points_for_image(self, filepath=None, include_ref_size=False, names=None):
        """
        Get combined mask polygon points for a specific image.
        Looks up polygon points by name from all_polygons.

        Args:
            filepath: Optional filepath to look up (defaults to self.image_filepath)
            include_ref_size: If True, returns list of (points, ref_size_dict) tuples
                             If False (default), returns list of point lists
            names: Optional list of polygon names to look up. If provided, skips
                   checkbox/modifications checks and uses these names directly.

        Returns:
            If include_ref_size=False: list of polygon point lists
            If include_ref_size=True: list of (points, {'w': ref_w, 'h': ref_h}) tuples
        """
        import os, json

        # Determine target filepath
        if filepath is None:
            filepath = getattr(self, "image_filepath", None)
        if not filepath:
            return []

        # If names not explicitly provided, read from UI/modifications
        if names is None:
            if not hasattr(self, "mask_polygon_enabled_checkbox"):
                return []
            if not self.mask_polygon_enabled_checkbox.isChecked():
                return []

            mask_poly = getattr(self, "modifications", {}).get("mask_polygon", {})
            if not mask_poly or not isinstance(mask_poly, dict):
                return []
            if not mask_poly.get("enabled", False):
                return []

            names = mask_poly.get("names", [])

        # Normalize names list
        try:
            names_list = list(names) if names else []
        except Exception:
            names_list = []

        if not names_list:
            return []

        # Parent tab provides polygons and project folder
        parent_tab = self.parent()
        if parent_tab is None:
            return []

        # Flush any dirty polygon saves so disk + memory are consistent
        try:
            if hasattr(parent_tab, "_flush_dirty_polygons"):
                parent_tab._flush_dirty_polygons()
        except Exception:
            pass

        fp_norm = os.path.normpath(str(filepath)).lower()

        result = []
        seen = set()

        def _add(name_key, pts, ref_size):
            if not pts or len(pts) < 3:
                return
            if name_key in seen:
                return
            seen.add(name_key)
            if include_ref_size:
                result.append((pts, ref_size or {}))
            else:
                result.append(pts)

        # ------------------------------------------------------------
        # Prefer in-memory polygons first (instant UI updates),
        # then fall back to disk for any names not found in memory.
        # ------------------------------------------------------------
        all_polygons = getattr(parent_tab, "all_polygons", None) or {}

        # Memory lookup
        try:
            for group_name, file_map in (all_polygons or {}).items():
                if not isinstance(file_map, dict):
                    continue
                for stored_fp, poly_data in file_map.items():
                    if os.path.normpath(str(stored_fp)).lower() != fp_norm:
                        continue
                    if not isinstance(poly_data, dict):
                        continue
                    poly_name = poly_data.get("name", group_name)
                    name_key = None
                    if poly_name in names_list:
                        name_key = poly_name
                    elif group_name in names_list:
                        name_key = group_name
                    if not name_key:
                        continue
                    pts = poly_data.get("points", [])
                    ref = (
                        poly_data.get("image_ref_size", {})
                        or poly_data.get("pixmap_size", {})
                        or poly_data.get("ref_size", {})
                        or {}
                    )
                    _add(name_key, pts, ref)
        except Exception:
            pass

        # Disk lookup (only for missing names)
        try:
            project_folder = getattr(parent_tab, "project_folder", None)
            if project_folder:
                polygons_dir = os.path.join(project_folder, "polygons")
                if os.path.isdir(polygons_dir):
                    image_base = os.path.splitext(os.path.basename(str(filepath)))[0]
                    for mask_name in names_list:
                        if mask_name in seen:
                            continue
                        json_path = os.path.join(polygons_dir, f"{mask_name}_{image_base}_polygons.json")
                        if not os.path.isfile(json_path):
                            continue
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                poly_data = json.load(f)
                            if isinstance(poly_data, dict):
                                pts = poly_data.get("points", [])
                                ref = poly_data.get("image_ref_size", {}) or {}
                                _add(mask_name, pts, ref)
                            elif isinstance(poly_data, list):
                                # legacy list of points
                                _add(mask_name, poly_data, {})
                        except Exception:
                            continue
        except Exception:
            pass

        return result

    @staticmethod
    def _build_polygon_mask(shape, points):
        """
        Build a boolean mask from polygon points.
        
        Args:
            shape: (H, W) or (H, W, C) - shape of the image
            points: list of (x, y) tuples defining the polygon
            
        Returns:
            np.ndarray: Boolean mask where True = inside polygon (masked/excluded)
        """
        import numpy as np
        import cv2
        
        if not points or len(points) < 3:
            return None
        
        H = shape[0]
        W = shape[1]
        
        # Convert points to numpy array for cv2.fillPoly
        pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Return boolean mask (True = inside polygon = masked)
        return mask > 0

    @staticmethod
    def _build_combined_polygon_mask(shape, polygon_points_list):
        """
        Build a combined boolean mask from multiple polygon point lists.
        
        Args:
            shape: (H, W) or (H, W, C) - shape of the image
            polygon_points_list: list of polygon point lists, each is list of (x, y) tuples
            
        Returns:
            np.ndarray: Boolean mask where True = inside any polygon (masked/excluded)
        """
        import numpy as np
        import cv2
        
        if not polygon_points_list:
            return None
        
        H = shape[0]
        W = shape[1]
        
        combined_mask = np.zeros((H, W), dtype=bool)
        
        for points in polygon_points_list:
            if points and len(points) >= 3:
                pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                combined_mask |= (mask > 0)
        
        return combined_mask if combined_mask.any() else None


    def _write_ax(self, image_path, modifications, quiet=False):
        mod_filename = self._ax_path_for(image_path)
        try:
            existing = {}
            if os.path.exists(mod_filename):
                try:
                    with open(mod_filename, "r", encoding="utf-8") as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}

            # ----- controlled deletions -----
            deletes = []
            if isinstance(modifications, dict):
                deletes = list(modifications.get("_delete", []))
                # also treat explicit None as "delete" for safety
                for k, v in list(modifications.items()):
                    if v is None:
                        deletes.append(k)
                # de-dupe and strip control key from what we write
                deletes = [k for k in dict.fromkeys(deletes)]
                if "_delete" in modifications:
                    modifications = {k: v for k, v in modifications.items() if k != "_delete"}
            for k in deletes:
                existing.pop(k, None)

            # ----- normal merge -----
            existing.update(modifications if isinstance(modifications, dict) else {})

            with open(mod_filename, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4)
            
            if not quiet:
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

        # Try cv2 for regular images (or simple TIFFs)
        try:
            img = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except Exception as e:
            logging.debug(f"_load_raw_image cv2 failed: {e}")

        # FALLBACK: For TIF files where cv2 failed, try tifffile anyway
        if ext in (".tif", ".tiff"):
            try:
                logging.debug(f"[Editor] cv2 returned None for TIFF, trying tifffile as fallback")
                arr = _tifffile_read_HWC(self.image_filepath)
                if arr is not None and getattr(arr, "size", 0) > 0:
                    return arr
            except Exception as e:
                logging.warning(f"[Editor] tifffile fallback also failed: {e}")

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

    def save_modifications_to_file(self, progress_dialog=None):
        """
        Save modification parameters into .ax and return a refresh hint:
          {"scope": "all"|"group"|"single", "root_name": <str or None>}

        Behavior:
        - For both 'Apply to group' and 'Apply to all groups', only target files that
          live in the SAME OS DIRECTORY as the image currently being edited.
        - If the parent’s group mapping doesn’t include that directory, fall back to
          scanning that directory on disk (common image extensions).
        """
        import os, json, logging
        from PyQt5 import QtWidgets

        # CRITICAL: If classification is enabled, ensure model is synced to class attribute
        # so that apply_aux_modifications (a @staticmethod) can find it during refresh
        try:
            cblock = (self.modifications or {}).get("classification") or {}
            if isinstance(cblock, dict) and bool(cblock.get("enabled", False)):
                parent = self.parent()
                bundle = getattr(parent, "random_forest_model", None)
                if isinstance(bundle, dict) and "model" in bundle:
                    if getattr(type(parent), "shared_random_forest_model", None) is None:
                        setattr(type(parent), "shared_random_forest_model", bundle)
                        logging.info("[sklearn] save_modifications_to_file: synced model to class attribute before save")
        except Exception as e:
            logging.debug(f"[sklearn] Model sync in save_modifications_to_file failed: {e}")

        def _update_progress(label, value, maximum=None):
            if progress_dialog:
                try:
                    if maximum is not None:
                        progress_dialog.setMaximum(maximum)
                    progress_dialog.setValue(value)
                    progress_dialog.setLabelText(label)
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass

        if not self.image_filepath:
            logging.error("No image_filepath set. Cannot save modifications.")
            return None

        # ---------- helpers ----------
        def _norm_dir(p: str) -> str:
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(os.path.dirname(p))))
            except Exception:
                return ""

        def _norm_path(p: str) -> str:
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(p)))
            except Exception:
                return ""

        def _detect_folder_group():
            """
            DUAL FOLDER FIX: Detect which folder group the current image belongs to.
            Returns: (folder_group_attr, folder_groups_dict) or (None, None)
            """
            parent = self.parent()
            if parent is None:
                return None, None
            
            cur_fp_norm = _norm_path(self.image_filepath)
            
            # Check multispectral first
            if hasattr(parent, "multispectral_image_data_groups"):
                groups = getattr(parent, "multispectral_image_data_groups", None) or {}
                for root_files in groups.values():
                    if isinstance(root_files, (list, tuple)):
                        for fp in root_files:
                            if isinstance(fp, str) and _norm_path(fp) == cur_fp_norm:
                                return "multispectral_image_data_groups", groups
            
            # Check thermal_rgb second
            if hasattr(parent, "thermal_rgb_image_data_groups"):
                groups = getattr(parent, "thermal_rgb_image_data_groups", None) or {}
                for root_files in groups.values():
                    if isinstance(root_files, (list, tuple)):
                        for fp in root_files:
                            if isinstance(fp, str) and _norm_path(fp) == cur_fp_norm:
                                return "thermal_rgb_image_data_groups", groups
            
            return None, None

        def _flatten_known_files(folder_groups):
            """Get all files from the specified folder group across all roots."""
            out = []
            if folder_groups:
                for group_files in folder_groups.values():
                    if isinstance(group_files, (list, tuple)):
                        for fp in group_files:
                            if isinstance(fp, str) and fp:
                                out.append(fp)
            return [fp for fp in dict.fromkeys(out)]

        def _group_files_for_current_root(folder_groups):
            """
            Get files from the specified folder group for the current root only.
            
            CRITICAL FIX: Find which root contains the current image by searching,
            rather than assuming index correspondence between root name lists.
            The root names in multispectral vs thermal_rgb may be completely different.
            """
            parent = self.parent()
            out = []
            root_name = None
            
            if parent is None or not folder_groups:
                return [], None
            
            cur_fp_norm = _norm_path(self.image_filepath)
            
            # Find which root (key) contains the current image
            for rn, files in folder_groups.items():
                if isinstance(files, (list, tuple)):
                    for fp in files:
                        if isinstance(fp, str) and _norm_path(fp) == cur_fp_norm:
                            root_name = rn
                            break
                if root_name:
                    break
            
            if root_name:
                arr = folder_groups.get(root_name, [])
                if isinstance(arr, (list, tuple)):
                    out = [fp for fp in arr if isinstance(fp, str) and fp]
                logging.debug(f"_group_files_for_current_root: found root_name={root_name} with {len(out)} files")
            else:
                logging.debug(f"_group_files_for_current_root: could not find root for {os.path.basename(self.image_filepath)}")
            
            return [fp for fp in dict.fromkeys(out)], root_name

        def _scan_folder_on_disk(target_dir_norm):
            """Fallback: scan directory on disk (only used when folder group detection fails)."""
            try:
                if not target_dir_norm:
                    return []
                folder = target_dir_norm
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

        def _pre_resize_shape():
            mods_wo = dict(self.modifications)
            mods_wo.pop("resize", None)
            img_pre = self.apply_all_modifications_to_image(self.base_image, mods_wo)
            if img_pre is not None and getattr(img_pre, "size", 0) > 0:
                hh, ww = img_pre.shape[:2]
                return int(ww), int(hh)
            return None, None

        ui_mods = {}

        # rotation
        if "rotate" in self.modifications:
            try:
                ui_mods["rotate"] = int(self.modifications.get("rotate", 0)) % 360
            except Exception:
                ui_mods["rotate"] = 0

        # crop (prefer precise ref size captured at selection time)
        if self.last_crop_rect is not None or self.crop_rect is not None:
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

        # --- RESIZE (pixel W×H preferred; else percentage) ---
        px_w_txt = self.resize_width_input.text().strip() if hasattr(self, "resize_width_input") else ""
        px_h_txt = self.resize_height_input.text().strip() if hasattr(self, "resize_height_input") else ""
        pct_txt  = self.resize_input.text().strip() if hasattr(self, "resize_input") else ""

        # Get the "natural" size (after crop/rotate but before resize)
        # This is the size the textboxes display when auto-filled
        pre_resize_w, pre_resize_h = _pre_resize_shape()

        if px_w_txt and px_h_txt:
            try:
                tw = int(px_w_txt); th = int(px_h_txt)
                if tw > 0 and th > 0:
                    # CRITICAL FIX: Only add resize if target differs from natural size
                    # The textboxes show current size - only treat as resize target if user changed them
                    if pre_resize_w and pre_resize_h and tw == pre_resize_w and th == pre_resize_h:
                        # Textbox matches current size - no resize needed
                        pass
                    else:
                        ui_mods["resize"] = {"px_w": tw, "px_h": th}
                        if pre_resize_w and pre_resize_h:
                            ui_mods["resize_ref_size"] = {"w": pre_resize_w, "h": pre_resize_h}
            except ValueError:
                pass
        elif pct_txt:
            try:
                sc = int(pct_txt)
                # Only apply percentage resize if not 100%
                if sc != 100:
                    ui_mods["resize"] = {"scale": sc}
                    if pre_resize_w and pre_resize_h:
                        ui_mods["resize_ref_size"] = {"w": pre_resize_w, "h": pre_resize_h}
            except ValueError:
                pass

        # ==================== SYNC ALL ENABLED STATES FROM UI ====================
        
        # NoData values - ALWAYS update from UI
        if hasattr(self, "nodata_input"):
            nodata_text = self.nodata_input.text().strip()
            nodata_values = self._parse_nodata_values(nodata_text)
            self.modifications["nodata_values"] = nodata_values if nodata_values else []
        if hasattr(self, "nodata_enabled_checkbox"):
            self.modifications["nodata_enabled"] = self.nodata_enabled_checkbox.isChecked()

        # Resize enabled
        if hasattr(self, "resize_enabled_checkbox"):
            self.modifications["resize_enabled"] = self.resize_enabled_checkbox.isChecked()

        # Rotate enabled
        if hasattr(self, "rotate_enabled_checkbox"):
            self.modifications["rotate_enabled"] = self.rotate_enabled_checkbox.isChecked()

        # Crop enabled
        if hasattr(self, "crop_enabled_checkbox"):
            self.modifications["crop_enabled"] = self.crop_enabled_checkbox.isChecked()

        # Band expression - ALWAYS update from UI
        if hasattr(self, "band_input"):
            expr = self.band_input.text().strip()
            self.modifications["band_expression"] = expr  # Always save, even if empty
        if hasattr(self, "band_enabled_checkbox"):
            self.modifications["band_enabled"] = self.band_enabled_checkbox.isChecked()

        # Histogram enabled
        if hasattr(self, "hist_enabled_checkbox"):
            self.modifications["hist_enabled"] = self.hist_enabled_checkbox.isChecked()

        # ==================== END ENABLED STATES ====================

        # ensure anchor metadata exists
        if "orig_size" not in self.modifications and self.base_image is not None:
            H0, W0 = self.base_image.shape[:2]
            C0 = 1 if self.base_image.ndim == 2 else int(self.base_image.shape[2])
            ui_mods["orig_size"] = {"h": int(H0), "w": int(W0), "c": int(C0)}
            ui_mods["anchor_to_original"] = True

        # merge: ALWAYS OVERWRITE with UI values. The UI is the source of truth when hitting 'Save'.
        # Previously this used "if k not in self.modifications", which caused new UI crops
        # to be silently ignored and Registration to be run on ghost crops.
        for k, v in ui_mods.items():
            self.modifications[k] = v

        # --------- FINAL MODS TO PERSIST (keep 'classification' so it can persist) ----------
        mods = dict(self.modifications)

        # If histogram set to "None", explicitly clear in targets
        hist_disable = False
        hm = mods.get("hist_match")
        if isinstance(hm, dict):
            m = (hm.get("mode") or "").lower()
            if m in ("none", "", None):
                hist_disable = True
        try:
            if hasattr(self, "hist_mode_combo"):
                txt = (self.hist_mode_combo.currentText() or "").strip().lower()
                if "none" in txt:
                    hist_disable = True
        except Exception:
            pass
        if hist_disable:
            mods["hist_match"] = None

        modifications = mods

        parent = self.parent()
        cur_dir_norm = _norm_dir(self.image_filepath)

        # ---------- DUAL FOLDER FIX: detect which folder group this image belongs to ----------
        folder_group_attr, folder_groups = _detect_folder_group()
        logging.debug(f"save_modifications_to_file: folder_group={folder_group_attr} for {os.path.basename(self.image_filepath)}")

        # ---------- choose targets by scope ----------
        # CRITICAL: Use only the detected folder group, NOT both groups
        all_known = _flatten_known_files(folder_groups) if folder_groups else []
        group_known, root_name = _group_files_for_current_root(folder_groups) if folder_groups else ([], None)
        logging.debug(f"save_modifications_to_file: root_name={root_name}, group_known has {len(group_known)} files")

        def _targets_for_scope(scope: str):
            if scope == "all":
                pool = all_known
            elif scope == "group":
                pool = group_known
            else:
                pool = []
            
            # DUAL FOLDER FIX: If we have proper folder group detection, use the pool directly
            # Only fallback to disk scan if detection failed (pool is empty AND no folder_groups)
            if pool:
                result = list(pool)
            elif folder_groups is None:
                # No folder group detected - fallback to scanning disk
                result = _scan_folder_on_disk(cur_dir_norm)
            else:
                # Folder group detected but pool is empty (shouldn't happen normally)
                result = []
            
            # Always ensure current file is included
            if self.image_filepath and self.image_filepath not in result:
                result.append(self.image_filepath)
            return [fp for fp in dict.fromkeys(result)]

        # Determine scope first to prevent single-file saves from wiping existing registration matrices
        scope = "single"
        if getattr(self, "apply_all_groups_checkbox", None) and self.apply_all_groups_checkbox.isChecked():
            scope = "all"
        elif getattr(self, "global_mods_checkbox", None) and self.global_mods_checkbox.isChecked():
            scope = "group"

        # ==================== REGISTRATION SETUP ====================
        sr = None
        ref_img_gray = None
        reg_mode_str = "None"
        is_reg_active = False

        # TEMPORARILY DISABLED BY USER REQUEST
        # if HAS_PYSTACKREG and "registration" in modifications:
        if False:
             reg_conf = modifications["registration"]
             if reg_conf.get("enabled", False):
                 reg_mode_str = reg_conf.get("mode", "Rigid Body")
                 # Init StackReg
                 try:
                     # FIND TRUE REFERENCE IMAGE (Always scan entire group, even if saving a single file)
                     true_ref_fp = self.image_filepath
                     search_pool = group_known if group_known else _scan_folder_on_disk(cur_dir_norm)
                     for fp in search_pool:
                         ax_path = fp + ".ax"
                         if os.path.exists(ax_path):
                             try:
                                 import json
                                 with open(ax_path, "r") as f:
                                     ax_data = json.load(f)
                                     reg = ax_data.get("registration", {})
                                     if reg.get("enabled", False) and reg.get("is_reference", False):
                                         true_ref_fp = fp
                                         break
                             except Exception: pass
                             
                     # Load true reference image
                     if true_ref_fp == self.image_filepath and self.base_image is not None:
                         ref_raw = self.base_image.copy()
                     else:
                         ref_raw = cv2.imread(true_ref_fp, cv2.IMREAD_UNCHANGED)
                         if ref_raw is None:
                             try:
                                 stream = open(true_ref_fp, "rb")
                                 bytes = bytearray(stream.read())
                                 numpyarray = np.asarray(bytes, dtype=np.uint8)
                                 ref_raw = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                             except Exception: pass

                     if ref_raw is not None:
                         # 1. Apply spatial pipeline to get the exact frame the user sees
                         mods_for_reg = modifications.copy()
                         mods_for_reg["registration"] = {"enabled": False}  # Prevent infinite recursion / double reg
                         mods_for_reg["hist_enabled"] = False               # Spatially irrelevant
                         mods_for_reg["band_enabled"] = False               # Spatially irrelevant
                         mods_for_reg["classification"] = {"enabled": False}
                         
                         ref_processed = self.apply_all_modifications_to_image(ref_raw, mods_for_reg)
                         logging.info(f"[REG CALC] Reference processed: raw={ref_raw.shape[:2]}, processed={ref_processed.shape[:2]}, crop_rect={mods_for_reg.get('crop_rect')}, crop_enabled={mods_for_reg.get('crop_enabled', 'NOT SET')}")
                         
                         if ref_processed.ndim == 3:
                             # Use simple mean for grayscale
                             ref_img_gray = np.mean(ref_processed, axis=2).astype(np.float32)
                         else:
                             ref_img_gray = ref_processed.astype(np.float32)
                             
                         is_reg_active = True
                         logging.info(f"Batch Registration Initialized: {reg_mode_str} (Reference: {os.path.basename(true_ref_fp)}, Size: {ref_img_gray.shape[1]}x{ref_img_gray.shape[0]})")
                 except Exception as e:
                     logging.error(f"Registration Init Failed: {e}")
                     is_reg_active = False

        def _prepare_and_write(fp, base_mods):
            """Helper to calculate specific mods (registration) for a file and write .ax"""
            final_mods = base_mods # Default to shared mods
            
            # If this file IS the true reference file, ensure it is marked as Reference
            is_ref_file = (os.path.normpath(fp) == os.path.normpath(true_ref_fp)) if is_reg_active else False
            
            if is_reg_active and ref_img_gray is not None:
                if is_ref_file:
                     # Reference: Identity transform
                     final_mods = base_mods.copy()
                     final_mods["registration"] = {
                         "enabled": True, "mode": reg_mode_str, 
                         "matrix": None, "is_reference": True
                     }
                else:
                     # Target: Calculate transform
                     try:
                         # Read image (using CV2 for speed)
                         # Use cv2.IMREAD_UNCHANGED to get raw depth
                         img_mov = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                         if img_mov is None:
                             # Try decoding with numpy if path has unicode (Windows fix)
                             stream = open(fp, "rb")
                             bytes = bytearray(stream.read())
                             numpyarray = np.asarray(bytes, dtype=np.uint8)
                             img_mov = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                         
                         if img_mov is None:
                             raise ValueError("Could not read image")
                         
                         # Apply exact same Crop, Rotate, Resize to TARGET raw image BEFORE registration
                         # We use base_mods because this relies on the batch processing logic: the user's 
                         # current UI settings MUST strictly carry over to every image being batched.
                         mods_for_reg = base_mods.copy()
                         mods_for_reg["registration"] = {"enabled": False}
                         mods_for_reg["hist_enabled"] = False
                         mods_for_reg["band_enabled"] = False
                         mods_for_reg["classification"] = {"enabled": False}
                         
                         logging.info(f"[REG CALC] BEFORE processing target {os.path.basename(fp)}: raw={img_mov.shape[:2]}, crop_rect={mods_for_reg.get('crop_rect')}, crop_enabled={mods_for_reg.get('crop_enabled', 'NOT SET')}")
                         tgt_processed = self.apply_all_modifications_to_image(img_mov, mods_for_reg)
                         logging.info(f"[REG CALC] AFTER processing target {os.path.basename(fp)}: processed={tgt_processed.shape[:2]}")
                         
                         if tgt_processed.ndim == 3:
                             img_mov_gray_reg = np.mean(tgt_processed, axis=2).astype(np.float32)
                         else:
                             img_mov_gray_reg = tgt_processed.astype(np.float32)
                             
                         orig_h, orig_w = img_mov_gray_reg.shape[:2]
                         target_h, target_w = ref_img_gray.shape[:2]

                         # MATHEMATICAL FIX:
                         # 1. pystackreg REQUIRES arrays to be exactly the same shape.
                         # 2. If the user cropped the images differently, they have different bounding boxes.
                         # 3. We absolutely MUST pad them from the TOP-LEFT (0,0) so their pixel coordinates
                         #    map exactly 1:1 with cv2.warpAffine, preventing arbitrary translation offsets!
                         
                         max_h = max(orig_h, target_h)
                         max_w = max(orig_w, target_w)
                         
                         reg_tgt = img_mov_gray_reg
                         if (orig_h, orig_w) != (max_h, max_w):
                             pad_tgt = np.zeros((max_h, max_w), dtype=np.float32)
                             pad_tgt[:orig_h, :orig_w] = img_mov_gray_reg
                             reg_tgt = pad_tgt
                             
                         reg_ref = ref_img_gray
                         if (target_h, target_w) != (max_h, max_w):
                             pad_ref = np.zeros((max_h, max_w), dtype=np.float32)
                             pad_ref[:target_h, :target_w] = ref_img_gray
                             reg_ref = pad_ref
                             
                         # Register (Create thread-local StackReg instance to avoid GIL/state locking)
                         mode_map = {
                            "Translation": StackReg.TRANSLATION,
                            "Rigid Body": StackReg.RIGID_BODY,
                            "Scaled Rotation": StackReg.SCALED_ROTATION,
                            "Affine": StackReg.AFFINE,
                            "Bilinear": StackReg.BILINEAR
                         }
                         sr_mode = mode_map.get(reg_mode_str, StackReg.RIGID_BODY)
                         local_sr = StackReg(sr_mode)
                         
                         # tmat maps reg_tgt to reg_ref
                         tmat = local_sr.register(reg_ref, reg_tgt)
                         
                         final_mods = base_mods.copy()
                         final_mods["registration"] = {
                             "enabled": True, "mode": reg_mode_str,
                             "matrix": tmat.tolist(),
                             "calc_shape": [orig_w, orig_h], # Crucial: Save Target dimensions to correctly scale Editor vs Export
                             "is_reference": False
                         }
                         logging.info(f"[REG CALC] file={os.path.basename(fp)}, ref_gray=({target_w},{target_h}), tgt_gray=({orig_w},{orig_h}), calc_shape=[{orig_w},{orig_h}], tmat tx={tmat[0,2]:.2f} ty={tmat[1,2]:.2f}")
                     except Exception as e:
                         # Fallback
                         final_mods = base_mods.copy()
                         final_mods["registration"] = {"enabled": False, "error": str(e)}
            
            # Write
            # If strict reference, maybe we force it? No, just write.
            self._write_ax(fp, final_mods, quiet=True)


        # ======== ALL GROUPS (same-folder) ========
        # ======== ALL GROUPS (same-folder) ========
        if getattr(self, "apply_all_groups_checkbox", None) and self.apply_all_groups_checkbox.isChecked():
            files = _targets_for_scope("all")
            total = len(files)
            _update_progress(f"Applying to all roots (0/{total})...", 0, total)
            
            # Parallelize I/O
            import concurrent.futures
            completed = 0
            # Throttle UI updates to avoid freezing main thread
            update_stride = max(1, total // 100)  # update every 1% or at least every 1
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, total + 1)) as executor:
                # Use _prepare_and_write helper
                futures = {executor.submit(_prepare_and_write, fp, modifications): fp for fp in files}
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    # Update progress only periodically
                    if completed == total or completed % update_stride == 0:
                        _update_progress(f"Applying to all roots ({completed}/{total})...", completed, total)
                    try:
                        future.result() 
                    except Exception as e:
                         logging.error(f"Failed to write .ax for {futures[future]}: {e}")

            # CRITICAL: Invalidate caches for all modified files
            try:
                if parent is not None and hasattr(parent, "invalidate_caches_for_file"):
                    for fp in files:
                        parent.invalidate_caches_for_file(fp)
            except Exception:
                pass

            logging.info(
                "Applied modifications to %d file(s) across ALL groups (dir=%s).",
                len(files), cur_dir_norm
            )
            return {"scope": "all", "root_name": None}

        # ======== CURRENT GROUP (same-folder) ========
        # ======== CURRENT GROUP (same-folder) ========
        if getattr(self, "global_mods_checkbox", None) and self.global_mods_checkbox.isChecked():
            files = _targets_for_scope("group")
            total = len(files)
            _update_progress(f"Applying to group (0/{total})...", 0, total)
            
            # Parallelize I/O
            import concurrent.futures
            completed = 0
            update_stride = max(1, total // 100)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, total + 1)) as executor:
                futures = {executor.submit(_prepare_and_write, fp, modifications): fp for fp in files}
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed == total or completed % update_stride == 0:
                        _update_progress(f"Applying to group ({completed}/{total})...", completed, total)
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Failed to write .ax for {futures[future]}: {e}")

            # CRITICAL: Invalidate caches for all modified files
            try:
                if parent is not None and hasattr(parent, "invalidate_caches_for_file"):
                    for fp in files:
                        parent.invalidate_caches_for_file(fp)
            except Exception:
                pass

            logging.info(
                "Applied modifications to %d file(s) in group '%s' (dir=%s).",
                len(files), root_name or "unknown", cur_dir_norm
            )
            # CRITICAL: Return the MULTISPECTRAL root name for refresh, not the folder-specific root
            # refresh_viewer expects multispectral_root_names, so we must use get_current_root_name()
            ms_root_name = None
            try:
                if parent is not None and hasattr(parent, "get_current_root_name"):
                    ms_root_name = parent.get_current_root_name()
            except Exception:
                pass
            return {"scope": "group", "root_name": ms_root_name}

        # ======== SINGLE IMAGE ========
        # Handled as single call to _prepare_and_write (reusing logic creates consistency)
        _prepare_and_write(self.image_filepath, modifications)
        # self._write_ax(self.image_filepath, modifications)
        
        # CRITICAL: Invalidate caches after saving so viewer refresh shows updated image
        try:
            if parent is not None and hasattr(parent, "invalidate_caches_for_file"):
                parent.invalidate_caches_for_file(self.image_filepath)
        except Exception:
            pass

        root_name = None
        try:
            if parent is not None and hasattr(parent, "get_current_root_name"):
                root_name = parent.get_current_root_name()
        except Exception:
            root_name = None

        return {"scope": "single", "root_name": root_name}

 
    def apply_all_changes(self):
        """Save .ax and close. Leave refresh to ProjectTab to avoid double refresh."""
        from PyQt5 import QtWidgets, QtCore
        
        # Check if we're applying to multiple files
        apply_all = getattr(self, "apply_all_groups_checkbox", None) and self.apply_all_groups_checkbox.isChecked()
        apply_group = getattr(self, "global_mods_checkbox", None) and self.global_mods_checkbox.isChecked()
        
        if apply_all or apply_group:
            # Show progress dialog for bulk operations
            progress = QtWidgets.QProgressDialog("Applying changes...", None, 0, 0, self)
            progress.setWindowTitle("Applying Changes")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QtWidgets.QApplication.processEvents()
            
            try:
                self._last_apply_hint = self.save_modifications_to_file(progress_dialog=progress)
            finally:
                progress.close()
        else:
            self._last_apply_hint = self.save_modifications_to_file()
        
        self.accept()
        logging.info("All changes applied, modifications saved, and dialog accepted.")

    def load_modifications_from_file(self):
        import os, json, logging

        if not self.image_filepath:
            return None

        # Fix: Check both Project Folder and Sidecar locations, picking the newest one.
        # This matches ProjectTab._get_export_image behavior.
        
        candidates = []
        
        # 1. Project-level path (if project folder is set)
        if self.project_folder and self.project_folder.strip():
            base_name = os.path.splitext(os.path.basename(self.image_filepath))[0]
            p_path = os.path.join(self.project_folder, base_name + ".ax")
            candidates.append(p_path)
            
        # 2. Sidecar path (next to image)
        # Note: _ax_path_for defaults to this if project_folder is missing, 
        # but we explicit check it here to cover the case where project_folder IS set 
        # but the valid .ax file is still a sidecar (e.g. from before project setup).
        sidecar_path = os.path.splitext(self.image_filepath)[0] + ".ax"
        candidates.append(sidecar_path)
        
        # Remove duplicates
        candidates = list(dict.fromkeys(candidates))
        
        best_path = None
        best_mtime = -1.0
        
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    mt = os.path.getmtime(cand)
                    if mt > best_mtime:
                        best_mtime = mt
                        best_path = cand
                except Exception:
                    pass
        
        if best_path:
            logging.debug(f"Looking for modifications in: {best_path}")
            try:
                with open(best_path, "r", encoding="utf-8") as f:
                    modifications = json.load(f) or {}
                # NOTE: do NOT strip 'classification' here; we want it to persist.
                logging.info(f"Modifications loaded from {best_path}")
                return modifications
            except Exception as e:
                logging.error(f"Failed to load modifications from {best_path}: {e}")

        return None




    def apply_all_modifications_to_image(self, image, modifications, progress_callback=None):
        """
        Apply all image modifications in a deterministic order.
        
        Order: crop/rotate (order depends on crop_rect_ref_size) -> hist_match -> resize -> band expression.
        
        The crop/rotate order is determined by crop_rect_ref_size:
        - If crop was drawn BEFORE rotating, crop is applied first
        - If crop was drawn AFTER rotating, rotate is applied first (default)
        
        Rotation/crop/resize keep native dtype; hist/expr compute in float32.
        (If HIST_MATCH_AFTER_RESIZE_IF_SHRINK is True and target is smaller,
         hist match is deferred until after resize for speed.)
        
        NoData values from modifications["nodata_values"] are respected:
        - Excluded from histogram statistics calculation
        - Preserved unchanged through histogram normalization
        - Preserved through resize operations (mask is resized with NEAREST)
        - Excluded from band expression calculations (preserved as original)
        
        Each modification can be individually enabled/disabled via *_enabled flags.
        
        progress_callback: optional callable(step_name, step_num, total_steps) for progress updates
        """
        import cv2, numpy as np, logging

        def _update_progress(step_name, step_num, total_steps=6):
            if progress_callback:
                try:
                    progress_callback(step_name, step_num, total_steps)
                except Exception:
                    pass

        _update_progress("Preparing image...", 0)

        if image is None or getattr(image, "size", 0) == 0:
            logging.error("apply_all_modifications_to_image: empty source image.")
            return image

        mods = modifications or {}
        _has_mods = any(k in mods for k in ("rotate","crop_rect","hist_match","resize","band_expression","registration"))
        result = image.copy() if _has_mods else image
        
        # ==================== APPLY REGISTRATION BEFORE CROP ====================
        # The matrix was calculated on CROPPED images for accuracy, but we apply it
        # to the RAW image and translate the matrix from crop-space to raw-space.
        # This allows the warp to pull pixels from outside the crop boundary.
        reg_cfg = mods.get("registration") or {}
        # TEMPORARILY DISABLED BY USER REQUEST
        # if isinstance(reg_cfg, dict) and reg_cfg.get("enabled") and reg_cfg.get("matrix") is not None:
        if False:
             if not reg_cfg.get("is_reference", False):
                  try:
                      import cv2
                      mat_list = reg_cfg["matrix"]
                      tmat = np.array(mat_list, dtype=np.float64)
                      mode_str = str(reg_cfg.get("mode", "")).lower()
                      
                      h, w = result.shape[:2]
                      
                      # Convert 4x4 matrix to 3x3 for OpenCV
                      if tmat.shape == (4, 4):
                           tmat_3x3 = np.zeros((3, 3), dtype=np.float64)
                           tmat_3x3[0:2, 0:2] = tmat[0:2, 0:2]
                           tmat_3x3[0:2, 2] = tmat[0:2, 3]
                           tmat_3x3[2, :] = [0, 0, 1]
                           tmat = tmat_3x3
                      elif tmat.shape != (3, 3):
                           raise ValueError(f"Matrix shape {tmat.shape} not supported by OpenCV warp")
                      
                      # --- 1. RESOLVE SCALE INVARIANCE ---
                      expected_w, expected_h = w, h
                      crop_enabled_flag = mods.get("crop_enabled", True)
                      _crop_rect_data = mods.get("crop_rect") if crop_enabled_flag else None
                      _crop_ref_data = mods.get("crop_rect_ref_size") or {}
                      cx, cy = 0, 0
                      
                      if crop_enabled_flag and isinstance(_crop_rect_data, dict) and _crop_rect_data:
                          xywh = self._extract_crop_xywh(_crop_rect_data)
                          if xywh:
                              cx, cy, cw, ch = xywh
                              if isinstance(_crop_ref_data, dict) and "w" in _crop_ref_data and "h" in _crop_ref_data:
                                  refW = max(1, int(_crop_ref_data.get("w") or w))
                                  refH = max(1, int(_crop_ref_data.get("h") or h))
                                  if refW != w or refH != h:
                                      sx_c = w / float(refW)
                                      sy_c = h / float(refH)
                                      expected_w = int(round(cw * sx_c))
                                      expected_h = int(round(ch * sy_c))
                                      cx = int(round(cx * sx_c))
                                      cy = int(round(cy * sy_c))
                                  else:
                                      expected_w, expected_h = cw, ch
                              else:
                                  expected_w, expected_h = cw, ch

                      calc_shape = reg_cfg.get("calc_shape")
                      if calc_shape is not None and len(calc_shape) == 2:
                           calc_w, calc_h = calc_shape
                           if calc_w > 0 and calc_h > 0 and (calc_w, calc_h) != (expected_w, expected_h):
                                S_x = expected_w / float(calc_w)
                                S_y = expected_h / float(calc_h)
                                S = np.array([[S_x, 0, 0], [0, S_y, 0], [0, 0, 1]], dtype=np.float64)
                                S_inv = np.array([[1/S_x, 0, 0], [0, 1/S_y, 0], [0, 0, 1]], dtype=np.float64)
                                tmat = S @ tmat @ S_inv
                      
                      # --- 2. COORDINATE TRANSLATION: crop-space -> raw-space ---
                      if cx != 0 or cy != 0:
                          T_crop = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
                          T_crop_inv = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
                          tmat = T_crop @ tmat @ T_crop_inv
                      
                      is_perspective = "bilinear" in mode_str or not np.allclose(tmat[2, :2], 0)
                      flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                      
                      C = 1 if result.ndim == 2 else result.shape[2]
                      
                      def _warp_block_editor(block):
                          if is_perspective:
                              return cv2.warpPerspective(block, tmat, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                          else:
                              return cv2.warpAffine(block, tmat[:2, :], (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                      
                      if C <= 4:
                          result = _warp_block_editor(result)
                          if C == 1 and result.ndim == 2:
                              result = result[..., None]
                      else:
                          warped_bands = []
                          for c in range(C):
                              warped_bands.append(_warp_block_editor(result[..., c]))
                          result = np.stack(warped_bands, axis=2)
                          
                  except Exception as e:
                      logging.warning(f"Registration application failed in editor preview: {e}")

        # ==================== EXTRACT ENABLED FLAGS ====================
        # Default to True if not specified (backwards compatible)
        rotate_enabled = mods.get("rotate_enabled", True)
        crop_enabled = mods.get("crop_enabled", True)
        hist_enabled = mods.get("hist_enabled", True)
        resize_enabled = mods.get("resize_enabled", True)
        band_enabled = mods.get("band_enabled", True)
        nodata_enabled = mods.get("nodata_enabled", True)
        
        # Extract NoData values to mask throughout operations
        if nodata_enabled:
            nodata_values = list(mods.get("nodata_values", []) or [])
        else:
            nodata_values = []

        # Extract mask polygon settings (names-based lookup)
        mask_polygon_cfg = mods.get("mask_polygon", {}) or {}
        mask_polygon_enabled = bool(mask_polygon_cfg.get("enabled", False)) if isinstance(mask_polygon_cfg, dict) else False
        mask_polygon_names = mask_polygon_cfg.get("names", []) if isinstance(mask_polygon_cfg, dict) else []
        
        # Look up polygon points by name for the current image (with ref sizes for scaling)
        mask_polygon_raw_data = []  # List of (points, ref_size) tuples
        mask_polygon_points_list = []
        if mask_polygon_enabled and mask_polygon_names:
            # Pass names directly to bypass checkbox state check
            if hasattr(self, '_get_mask_polygon_points_for_image'):
                mask_polygon_raw_data = self._get_mask_polygon_points_for_image(include_ref_size=True, names=mask_polygon_names)
                # For immediate use (before any transforms), use points directly
                mask_polygon_points_list = [pts for pts, _ in mask_polygon_raw_data]
            else:
                # Legacy: use points directly if stored (for backward compatibility)
                legacy_points = mask_polygon_cfg.get("points", [])
                if legacy_points and len(legacy_points) >= 3:
                    mask_polygon_points_list = [legacy_points]

        # --- helper to scale polygon points to target dimensions ---
        def _scale_polygon_points(raw_data, target_h, target_w):
            """Scale polygon points from their ref_size to target dimensions."""
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

        # --- helper to build NoData mask (uses shared utility supporting expressions) ---
        def _build_nodata_mask(img, nd_vals):
            """Build boolean mask where True = NoData pixel. Supports both numeric literals and threshold expressions."""
            from .utils import build_nodata_mask as _shared_build_nodata_mask
            return _shared_build_nodata_mask(img, nd_vals, bgr_input=True)

        # --- helper to build combined mask (nodata + polygons) ---
        def _build_combined_mask(img, nd_vals, poly_points_list, poly_enabled):
            """Build combined mask from NoData values and multiple polygon masks."""
            mask = None
            
            # Build NoData mask
            if nd_vals:
                mask = _build_nodata_mask(img, nd_vals)
            
            # Build combined polygon mask from all polygons
            if poly_enabled and poly_points_list:
                H, W = img.shape[:2]
                poly_mask = ImageEditorDialog._build_combined_polygon_mask((H, W), poly_points_list)
                if poly_mask is not None:
                    if mask is None:
                        mask = poly_mask
                    else:
                        mask = mask | poly_mask  # Combine: either nodata OR inside polygon
            
            return mask

        # --- tiny helper for hist-match (prefer parent implementation) ---
        def _apply_hist_local(img, mods, nodata_vals=None, poly_points_list=None, poly_enabled=False):
            parent = self.parent()
            if parent is not None and hasattr(parent, "_apply_hist_match"):
                try:
                    # Pass nodata_values and mask_polygon if parent supports it
                    return parent._apply_hist_match(img, mods, nodata_values=nodata_vals,
                                                     mask_polygon_points=poly_points_list,
                                                     mask_polygon_enabled=poly_enabled)
                except TypeError:
                    # Fallback: try without polygon mask
                    try:
                        return parent._apply_hist_match(img, mods, nodata_values=nodata_vals)
                    except TypeError:
                        try:
                            return parent._apply_hist_match(img, mods)
                        except Exception:
                            pass
                except Exception:
                    pass
            # fallback (identical semantics; float32) - OPTIMIZED with LUT
            hcfg = (mods or {}).get("hist_match")
            if not hcfg:
                return img
            x = img.astype(np.float32, copy=False)
            if x.ndim == 2:
                x = x[..., None]
            C = x.shape[2]
            mode = (hcfg.get("mode") or "meanstd").lower()

            # Build combined mask (NoData + polygons) using shared utility
            combined_mask = None
            if nodata_vals:
                from .utils import build_nodata_mask as _shared_build_nodata_mask
                combined_mask = _shared_build_nodata_mask(x, nodata_vals, bgr_input=True)
            
            # Add polygon masks
            if poly_enabled and poly_points_list:
                poly_mask = ImageEditorDialog._build_combined_polygon_mask(x.shape[:2], poly_points_list)
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
                    # Apply combined mask for stats calculation
                    if combined_mask is not None:
                        ch_masked = np.where(combined_mask, np.nan, ch)
                        mu_t = float(np.nanmean(ch_masked))
                        sd_t = _safe_std(ch_masked)
                    else:
                        mu_t = float(np.nanmean(ch))
                        sd_t = _safe_std(ch)
                    mu_r = float(stats[c].get("mean", 0.0))
                    sd_r = float(stats[c].get("std", 1.0))
                    # Apply normalization, preserve masked values
                    new_vals = (ch - mu_t) * (sd_r / sd_t) + mu_r
                    if combined_mask is not None:
                        x[..., c] = np.where(combined_mask, ch, new_vals)
                    else:
                        x[..., c] = new_vals
            else:
                # FIX: Use fast LUT-based CDF matching instead of slow argsort
                ref = hcfg.get("ref_cdf", {}) or {}
                per = ref.get("per_band") or []
                bins = 2048  # number of bins for histogram
                for c in range(min(C, len(per))):
                    lut = per[c] or {}
                    x_n = np.asarray(lut.get("x") or [0.0, 1.0], dtype=np.float32)
                    y    = np.asarray(lut.get("y") or [0.0, 1.0], dtype=np.float32)
                    lo   = float(lut.get("lo", 0.0)); hi = float(lut.get("hi", 1.0))
                    if hi <= lo: hi = lo + 1.0
                    ch = x[..., c]
                    # For CDF matching, exclude masked pixels (nodata + polygons) from histogram
                    valid_mask = ~combined_mask if combined_mask is not None else None
                    
                    # Normalize to [0,1]
                    z = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
                    # Build source CDF using histogram (O(N) vs O(N log N) for argsort)
                    idx = np.clip((z * (bins - 1)).astype(np.int32), 0, bins - 1)
                    
                    if valid_mask is not None:
                        # Only count valid pixels in histogram
                        hist = np.bincount(idx[valid_mask].ravel(), minlength=bins).astype(np.float32)
                    else:
                        hist = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
                    
                    cdf_src = np.cumsum(hist)
                    total = float(cdf_src[-1]) if cdf_src.size else 1.0
                    cdf_src /= max(total, 1.0)
                    # Map source CDF probabilities to reference values
                    xprime_norm = np.interp(cdf_src, y, x_n).astype(np.float32)
                    # Apply LUT, preserve masked pixels (nodata + polygons)
                    new_vals = xprime_norm[idx] * (hi - lo) + lo
                    if combined_mask is not None:
                        x[..., c] = np.where(combined_mask, ch, new_vals)
                    else:
                        x[..., c] = new_vals
            return x[..., 0] if image.ndim == 2 else x

        # ---- rotation (90 steps) ----
        def _rotate_90s_numpy(arr, deg):
            d = int(deg) % 360
            if d == 0 or arr is None or getattr(arr, "size", 0) == 0:
                return arr
            if d == 90:
                out = np.rot90(arr, -1)
            elif d == 180:
                out = np.rot90(arr, 2)
            elif d == 270:
                out = np.rot90(arr, 1)
            else:
                return arr
            return np.ascontiguousarray(out)

        try:
            rot = int(modifications.get("rotate", 0)) if "rotate" in modifications else 0
            rot = ((rot % 360) + 360) % 360
        except Exception:
            rot = 0

        # ---- Determine crop/rotate order based on crop_rect_ref_size ----
        # crop_rect_ref_size tells us which reference frame the crop was drawn in:
        # - If matches ORIGINAL dims → user cropped BEFORE rotating → do CROP first
        # - If matches ROTATED dims → user rotated BEFORE cropping → do ROTATE first
        raw_h, raw_w = result.shape[:2]
        rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
        
        crop_rect_data = modifications.get("crop_rect") if crop_enabled else None
        crop_ref = modifications.get("crop_rect_ref_size") or {}
        
        do_rotate_first = True  # Default: rotate first (standard flow)
        if crop_rect_data and rot in (90, 180, 270):
            ref_w = int(crop_ref.get("w", 0)) or 0
            ref_h = int(crop_ref.get("h", 0)) or 0
            if ref_w > 0 and ref_h > 0:
                # If ref matches original dims → crop was drawn BEFORE rotate
                if (ref_w, ref_h) == (raw_w, raw_h):
                    do_rotate_first = False
                # If ref matches rotated dims → crop was drawn AFTER rotate
                elif (ref_w, ref_h) == (rotated_w, rotated_h):
                    do_rotate_first = True

        def _do_crop():
            nonlocal result
            if not crop_enabled or "crop_rect" not in modifications:
                return
            xywh = self._extract_crop_xywh(modifications["crop_rect"])
            if not xywh:
                return
            x, y, w, h = xywh
            H, W = result.shape[:2]

            ref = modifications.get("crop_rect_ref_size") or {}
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
                result = result[y0:y1, x0:x1]
                logging.debug(f"Applied crop modification (scaled): ({x0},{y0})-({x1},{y1})")
            else:
                logging.warning("Crop rect out of bounds/empty after scaling; skipping crop.")

        def _do_rotate():
            nonlocal result
            if not rotate_enabled or rot not in (90, 180, 270):
                return
            try:
                result = _rotate_90s_numpy(result, rot)
                logging.debug(f"Applied rotation: {rot} degrees (NumPy).")
            except Exception as e:
                logging.warning(f"Rotation failed ({rot} deg) via NumPy: {e}")

        # Execute in the correct order based on when crop was drawn
        _update_progress("Applying crop/rotate...", 1)
        if do_rotate_first:
            _do_rotate()
            _do_crop()
        else:
            _do_crop()
            _do_rotate()

        if result is None or result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0:
            logging.error("Result is empty after crop/rotate; skipping further modifications.")
            return image


        # ---- histogram matching (optionally deferred for shrink) ----
        deferred_hist_block = None
        hcfg_now = (modifications or {}).get("hist_match", None) if hist_enabled else None

        if hcfg_now and getattr(self, "HIST_MATCH_AFTER_RESIZE_IF_SHRINK", False) and resize_enabled and ("resize" in modifications):
            # Peek at resize target to see if it's a shrink
            h0, w0 = result.shape[:2]
            resize_info = modifications["resize"]
            new_w, new_h = w0, h0
            if h0 > 0 and w0 > 0:
                if "px_w" in resize_info or "px_h" in resize_info:
                    tw = int(resize_info.get("px_w", 0)) or 0
                    th = int(resize_info.get("px_h", 0)) or 0
                    if tw > 0 and th > 0:
                        new_w, new_h = tw, th
                    elif tw > 0:
                        s = tw / float(w0)
                        new_w, new_h = tw, max(1, int(round(h0 * s)))
                    elif th > 0:
                        s = th / float(h0)
                        new_h, new_w = th, max(1, int(round(w0 * s)))
                    # else: keep new_w, new_h = w0, h0
                elif "scale" in resize_info:
                    s = max(1, int(resize_info["scale"])) / 100.0
                    new_w = max(1, int(round(w0 * s)))
                    new_h = max(1, int(round(h0 * s)))
                else:
                    pct_w = int(resize_info.get("width", 100))
                    pct_h = int(resize_info.get("height", 100))
                    new_w = max(1, int(round(w0 * (pct_w / 100.0))))
                    new_h = max(1, int(round(h0 * (pct_h / 100.0))))
            if (new_w < w0) or (new_h < h0):
                deferred_hist_block = hcfg_now
                hcfg_now = None  # skip now; will apply after resize

        if hcfg_now:
            _update_progress("Applying histogram matching...", 2)
            try:
                # Scale polygon points to match current image dimensions
                cur_h, cur_w = result.shape[:2]
                scaled_poly_points = _scale_polygon_points(mask_polygon_raw_data, cur_h, cur_w) if mask_polygon_raw_data else mask_polygon_points_list
                result = _apply_hist_local(result, {"hist_match": hcfg_now}, nodata_values,
                                          scaled_poly_points, mask_polygon_enabled)
            except Exception as e:
                logging.warning(f"Histogram match failed: {e}")

        # ---- resize ----
        _update_progress("Applying resize...", 3)
        if resize_enabled and "resize" in modifications:
            resize_info = modifications["resize"]
            h0, w0 = result.shape[:2]
            if h0 > 0 and w0 > 0:
                new_w, new_h = w0, h0
                if "px_w" in resize_info or "px_h" in resize_info:
                    tw = int(resize_info.get("px_w", 0))
                    th = int(resize_info.get("px_h", 0))
                    if tw > 0 and th > 0:
                        new_w, new_h = tw, th
                    elif tw > 0:
                        s = tw / float(w0)
                        new_w = tw; new_h = max(1, int(round(h0 * s)))
                    elif th > 0:
                        s = th / float(h0)
                        new_h = th; new_w = max(1, int(round(w0 * s)))
                elif "scale" in resize_info:
                    scale = max(1, int(resize_info["scale"]))
                    new_w = max(1, int(round(w0 * (scale / 100.0))))
                    new_h = max(1, int(round(h0 * (scale / 100.0))))
                else:
                    pct_w = int(resize_info.get("width", 100))
                    pct_h = int(resize_info.get("height", 100))
                    new_w = max(1, int(round(w0 * (pct_w / 100.0))))
                    new_h = max(1, int(round(h0 * (pct_h / 100.0))))
                if new_w != w0 or new_h != h0:
                    sw = new_w / float(w0); sh = new_h / float(h0)
                    if sw < 1.0 or sh < 1.0:
                        interp = cv2.INTER_AREA
                    elif max(sw, sh) < 2.0:
                        interp = cv2.INTER_LINEAR
                    else:
                        interp = cv2.INTER_CUBIC
                    
                    # Handle NoData during resize using NaN propagation:
                    # 1. Replace NoData with NaN before resize
                    # 2. Resize - any pixel touching NaN becomes NaN
                    # 3. Replace NaN with NoData value after resize
                    # IMPORTANT: Only use numeric NoData values for restoration (not expression strings)
                    nd_restore_val = None
                    for nv in (nodata_values or []):
                        if not isinstance(nv, str):  # Skip expression strings
                            try:
                                nd_restore_val = float(nv)
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if nodata_values and nd_restore_val is not None:
                        # We have numeric NoData - use NaN propagation with proper restoration
                        try:
                            # CRITICAL: Only use NUMERIC NoData values for resize masking
                            # Expression-based NoData (b1>143) should be evaluated AFTER resize
                            numeric_nd_vals = [v for v in nodata_values if not isinstance(v, str)]
                            if numeric_nd_vals:
                                nd_mask = _build_nodata_mask(result, numeric_nd_vals)
                            else:
                                nd_mask = None
                            
                            if nd_mask is not None and nd_mask.any():
                                # Convert to float32 and replace NoData with NaN
                                work = result.astype(np.float32, copy=True)
                                if work.ndim == 2:
                                    work[nd_mask] = np.nan
                                else:
                                    for c in range(work.shape[2]):
                                        work[..., c][nd_mask] = np.nan
                                
                                # Resize - NaN propagates through interpolation
                                work = resize_safe(work, new_w, new_h, interp)
                                
                                # Replace NaN with NoData value
                                nan_mask = np.isnan(work)
                                if nan_mask.any():
                                    work[nan_mask] = nd_restore_val
                                
                                result = work
                            else:
                                result = resize_safe(result, new_w, new_h, interp)
                        except Exception as e:
                            logging.warning(f"NoData-aware resize failed: {e}; falling back to normal resize")
                            result = resize_safe(result, new_w, new_h, interp)
                    else:
                        # No numeric NoData value - just resize normally
                        # Expression-based NoData (b1>143) is evaluated AFTER resize
                        # Do NOT apply expression masks during resize as this corrupts pixel values to 0
                        result = resize_safe(result, new_w, new_h, interp)
                    
                    logging.debug(f"Applied resize modification to {new_w}x{new_h} (interp={interp}).")
            else:
                logging.warning("Resize skipped: source has zero dimension.")

        # (Registration is now applied BEFORE crop — see above)

        # ---- deferred histogram matching (after shrink) ----
        if deferred_hist_block:
            _update_progress("Applying histogram matching...", 4)
            try:
                # Scale polygon points to match current (resized) image dimensions
                cur_h, cur_w = result.shape[:2]
                scaled_poly_points = _scale_polygon_points(mask_polygon_raw_data, cur_h, cur_w) if mask_polygon_raw_data else mask_polygon_points_list
                result = _apply_hist_local(result, {"hist_match": deferred_hist_block}, nodata_values,
                                          scaled_poly_points, mask_polygon_enabled)
            except Exception as e:
                logging.warning(f"Histogram match (deferred) failed: {e}")

        # ---- band expression (float32 scientific) ----
        _update_progress("Applying band expression...", 5)
        if band_enabled and "band_expression" in modifications:
            expr = (modifications.get("band_expression") or "").strip()
            if expr:
                result = self.process_band_expression(result, expr, nodata_values=nodata_values)
                logging.debug("Applied band expression modification.")

        # Store final COMBINED mask (NoData + polygons) for display normalization to use
        # This ensures viewer shows same mask as process_polygon (built on same image state)
        # BUT: Skip if classification is enabled (result would be indices, not pixel values)
        classification_enabled = False
        try:
            cblock = (modifications or {}).get("classification") or {}
            if isinstance(cblock, dict) and str(cblock.get("mode", "")).lower() == "sklearn" and bool(cblock.get("enabled", False)):
                classification_enabled = True
        except Exception:
            pass
        
        if not classification_enabled:
            try:
                # Build combined mask (NoData + polygon masks)
                # Scale polygon points to final result dimensions
                final_h, final_w = result.shape[:2]
                final_poly_points = _scale_polygon_points(mask_polygon_raw_data, final_h, final_w) if mask_polygon_raw_data else mask_polygon_points_list
                
                # Use the combined mask builder
                self._display_nodata_mask = _build_combined_mask(
                    result, 
                    nodata_values if nodata_enabled else [], 
                    final_poly_points, 
                    mask_polygon_enabled
                )
                
                # Log what was included
                mask_info = []
                if nodata_enabled and nodata_values:
                    mask_info.append(f"nodata={nodata_values}")
                if mask_polygon_enabled and final_poly_points:
                    mask_info.append(f"polygons={len(final_poly_points)}")
                
                if self._display_nodata_mask is not None:
                    logging.info(f"[apply_all_modifications_to_image] Stored combined mask for display ({self._display_nodata_mask.sum()} masked pixels, {', '.join(mask_info) if mask_info else 'none'})")
                else:
                    logging.info(f"[apply_all_modifications_to_image] No display mask built (no NoData or polygons enabled)")
            except Exception as e:
                self._display_nodata_mask = None
                logging.warning(f"[apply_all_modifications_to_image] Failed to build display mask: {e}")
        else:
            self._display_nodata_mask = None
            logging.info(f"[apply_all_modifications_to_image] Display mask NOT built: classification_enabled={classification_enabled}")

        _update_progress("Complete", 6)
        return result


    def apply_modifications(self, modifications):
        if self.base_image is None:
            return
        mod_image = self.apply_all_modifications_to_image(self.base_image, modifications)
        self.original_image = mod_image          # keep scientific pixels (float/uint16/etc.)
        self.display_image_data = mod_image      # defer normalization to display_image()
        
        # FIX: Update size inputs to reflect actual result dimensions
        if mod_image is not None and getattr(mod_image, "size", 0) > 0:
            h, w = mod_image.shape[:2]
            if hasattr(self, "resize_width_input"):
                self.resize_width_input.setText(str(w))
            if hasattr(self, "resize_height_input"):
                self.resize_height_input.setText(str(h))

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

    
    def _safe_eval_band_expr(expr: str, names: dict):
        """
        Safe eval for band expressions.
        Supports: + - * /, unary -, comparisons (< <= > >= == !=),
                  boolean ops (& | ~) and their word forms (AND/OR/NOT), parentheses.
        """
        import re, ast
        # normalize word booleans
        expr_norm = re.sub(r'\bAND\b', '&', expr, flags=re.IGNORECASE)
        expr_norm = re.sub(r'\bOR\b',  '|', expr_norm, flags=re.IGNORECASE)
        expr_norm = re.sub(r'\bNOT\b', '~', expr_norm, flags=re.IGNORECASE)

        expr_norm = normalize_band_expr(expr_norm)

        # parse and whitelist nodes
        tree = ast.parse(expr_norm, mode="eval")
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
            ast.BitAnd, ast.BitOr, ast.Invert,  # &, |, ~
            ast.And, ast.Or, ast.Not,           # (rarely seen post-normalization)
            ast.Load, ast.Name, ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise SyntaxError(f"Illegal syntax/nodes in band expression '{expr}'")
            if isinstance(node, ast.Call):
                raise SyntaxError("Function calls not allowed")
            if isinstance(node, ast.Name) and node.id not in names:
                raise NameError(f"Use of '{node.id}' is not allowed. Only b1..bn.")
        code = compile(ast.fix_missing_locations(tree), "<band-expr>", "eval")
        return eval(code, {"__builtins__": {}}, names)
   
    
    
    def process_band_expression(self, image, expr, nodata_values=None):
        import logging, numpy as np, re
        try:
            # FIX: Remap band references for 3-channel BGR images
            # Image data is BGR but user sees RGB in display (due to BGR→RGB conversion)
            # So when user types b1, they expect Red (what they see), but data has Blue at index 0
            # Remap b1↔b3 so expression matches what user visually sees
            effective_expr = expr
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
                # BGR→RGB remapping: b1↔b3 (b2 stays the same)
                effective_expr = re.sub(r'\bb1\b', '__B1_TEMP__', effective_expr)
                effective_expr = re.sub(r'\bb3\b', 'b1', effective_expr)
                effective_expr = re.sub(r'__B1_TEMP__', 'b3', effective_expr)
                if effective_expr != expr:
                    logging.info(f"[process_band_expression] Remapped BGR→RGB: '{expr}' → '{effective_expr}'")
            
            # CRITICAL FIX: Build nodata mask using shared utility that supports expressions
            # This ensures expressions like "b1 > 143" create a SPATIAL mask from the
            # ORIGINAL multi-channel image, applied to ALL bands at those (x,y) locations
            nodata_mask = None
            if nodata_values and isinstance(image, np.ndarray):
                from .utils import build_nodata_mask as _shared_build_nodata_mask
                # Build mask from ORIGINAL image (before any band expression transform)
                # bgr_input=True because image_editor works with BGR data
                nodata_mask = _shared_build_nodata_mask(image, nodata_values, bgr_input=True)
            
            res = eval_band_expression(image, effective_expr)

            # Restore nodata values in result (preserve original pixel values where nodata)
            if nodata_mask is not None and isinstance(res, np.ndarray):
                if res.shape[:2] == nodata_mask.shape:
                    # Preserve original values where nodata
                    if res.ndim == image.ndim:
                        if res.ndim == 2:
                            res = np.where(nodata_mask, image.astype(res.dtype), res)
                        else:
                            for c in range(min(res.shape[2], image.shape[2] if image.ndim == 3 else 1)):
                                img_ch = image[..., c] if image.ndim == 3 else image
                                res[..., c] = np.where(nodata_mask, img_ch.astype(res.dtype), res[..., c])

            # ⚠️ Normalize channel order for the editor: keep 3-ch arrays in BGR
            if isinstance(res, np.ndarray) and res.ndim == 3 and res.shape[2] == 3:
                # res coming from the util is RGB — flip to BGR so display_image's
                # BGR→RGB step produces correct colors.
                res = res[:, :, ::-1].copy()

            self.last_band_float_result = res.copy()
            return res
        except Exception as e:
            logging.error(f"Error processing band expression '{expr}': {e}")
            return image


    def init_ui(self):
        from PyQt5 import QtGui, QtWidgets, QtCore
        from PyQt5.QtWidgets import QToolButton

        # --- global compact style tweaks (tight layout, BIGGER labels/checks) ---
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
            QComboBox { padding: 0px 4px; min-height: 20px; font-size: 12px; }
            QLineEdit { padding: 1px 3px; }
            QPushButton { padding: 2px 6px; min-height: 20px; }
            QToolButton { padding: 0px; }
            
            QCheckBox { font-size: 13px; }
            QCheckBox::indicator:checked {
                background-color: #FFD700; 
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked {
                background-color: white; 
                border: 1px solid gray;
            }
            QToolButton.collapsible-toggle {
                color: #006400; /* Dark Green text/arrow */
                font-weight: bold;
                font-size: 11pt;
            }
        """)

        # MAIN LAYOUT: HORIZONTAL (Image Left | Sidebar Right)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ==========================================================
        # LEFT: Image Area (Takes all space)
        # ==========================================================
        self.image_label = QtWidgets.QLabel("No Image Loaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setScaledContents(False)  # do NOT let label auto-scale
        try:
            self.image_label.setAttribute(QtCore.Qt.WA_HighDpiPixmaps, True)
        except AttributeError:
            pass
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setBackgroundRole(QtGui.QPalette.Dark)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setViewportMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.image_label)

        # Add image area to main layout with stretch=1
        main_layout.addWidget(self.scroll_area, 1)

        # crop rubber band
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.image_label)
        self.origin = QtCore.QPoint()
        self.crop_rect = None
        
        # ==========================================================
        # RIGHT: Sidebar (Fixed Width)
        # ==========================================================
        sidebar_container = QtWidgets.QWidget()
        sidebar_container.setFixedWidth(320)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)
        
        # --- Control Panel (Scrollable content inside Sidebar) ---
        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Allow it to expand vertically
        control_scroll.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        
        control_container = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 4, 0) # Right margin for scrollbar
        control_layout.setSpacing(10)
        
        control_scroll.setWidget(control_container)
        sidebar_layout.addWidget(control_scroll)

        # ========== helpers ==========
        def _tight_row(*widgets):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            for wid in widgets:
                h.addWidget(wid)
            return w

        def _fixw(widget, w):
            widget.setFixedWidth(w)
            return widget

        # ==========================================================
        # GROUP 1: Geometry Options (Scale, Size, Rotate, Crop)
        # ==========================================================
        geo_group = CollapsibleBox("Geometry Options")
        geo_group.toggle_button.setProperty("class", "collapsible-toggle")
        geo_layout = geo_group.content_layout
        
        # -- 1a. Scale / Size --
        row_scale = QtWidgets.QHBoxLayout()
        scale_label = QtWidgets.QLabel("Scale %:")
        
        self.resize_input = QtWidgets.QLineEdit()
        self.resize_input.setPlaceholderText("5–100")
        self.resize_input.setValidator(QtGui.QIntValidator(1, 100, self))
        _fixw(self.resize_input, 50)
        
        scale_apply_btn = QtWidgets.QPushButton("apply")
        scale_apply_btn.setFixedHeight(22)
        scale_apply_btn.setToolTip("Apply proportional resize")
        scale_apply_btn.clicked.connect(self.on_resize_input_entered)
        self.resize_input.returnPressed.connect(scale_apply_btn.click)
        
        self.resize_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.resize_enabled_checkbox.setToolTip("Enable/disable resize scaling")
        self.resize_enabled_checkbox.setChecked(True)
        self.resize_enabled_checkbox.stateChanged.connect(self._on_resize_enabled_changed)
        
        row_scale.addWidget(scale_label)
        row_scale.addWidget(self.resize_input)
        row_scale.addWidget(scale_apply_btn)
        row_scale.addStretch()
        row_scale.addWidget(self.resize_enabled_checkbox)
        
        geo_layout.addLayout(row_scale)

        # -- 1b. Size (px) --
        row_px = QtWidgets.QHBoxLayout()
        px_label = QtWidgets.QLabel("Size (px):")
        self.resize_width_input = QtWidgets.QLineEdit()
        self.resize_width_input.setPlaceholderText("W")
        self.resize_width_input.setValidator(QtGui.QIntValidator(1, 20000, self))
        _fixw(self.resize_width_input, 50)
        
        x_sep = QtWidgets.QLabel("×")
        self.resize_height_input = QtWidgets.QLineEdit()
        self.resize_height_input.setPlaceholderText("H")
        self.resize_height_input.setValidator(QtGui.QIntValidator(1, 20000, self))
        _fixw(self.resize_height_input, 50)
        
        px_apply_btn = QtWidgets.QPushButton("Apply")
        px_apply_btn.setFixedHeight(22)
        px_apply_btn.setToolTip("Resize to exact pixel size (W×H)")
        px_apply_btn.clicked.connect(self.on_resize_pixels_entered)
        self.resize_width_input.returnPressed.connect(px_apply_btn.click)
        self.resize_height_input.returnPressed.connect(px_apply_btn.click)
        
        row_px.addWidget(px_label)
        row_px.addWidget(self.resize_width_input)
        row_px.addWidget(x_sep)
        row_px.addWidget(self.resize_height_input)
        row_px.addWidget(px_apply_btn)
        row_px.addStretch()
        
        geo_layout.addLayout(row_px)
        
        # -- 1b. Rotate / Crop --
        row_rot = QtWidgets.QHBoxLayout()
        rot_label = QtWidgets.QLabel("Rotate:")
        
        self.rotate_left_btn  = QToolButton()
        self.rotate_right_btn = QToolButton()
        self.rotate_left_btn.setText("↶")
        self.rotate_right_btn.setText("↷")
        self.rotate_left_btn.setFixedSize(22, 22)
        self.rotate_right_btn.setFixedSize(22, 22)
        self.rotate_left_btn.clicked.connect(self.on_rotate_left)
        self.rotate_right_btn.clicked.connect(self.on_rotate_right)
        
        self.rotate_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.rotate_enabled_checkbox.setChecked(True)
        self.rotate_enabled_checkbox.stateChanged.connect(self._on_rotate_enabled_changed)
        
        row_rot.addWidget(rot_label)
        row_rot.addWidget(self.rotate_left_btn)
        row_rot.addWidget(self.rotate_right_btn)
        row_rot.addStretch()
        row_rot.addWidget(self.rotate_enabled_checkbox)
        
        geo_layout.addLayout(row_rot)
        
        # New Row for Crop
        row_crop = QtWidgets.QHBoxLayout()
        crop_label = QtWidgets.QLabel("Crop:")
        self.crop_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.crop_enabled_checkbox.setToolTip("Enable/disable crop rectangle - Draw a box on the image to crop.")
        self.crop_enabled_checkbox.setChecked(True)
        self.crop_enabled_checkbox.stateChanged.connect(self._on_crop_enabled_changed)
        
        row_crop.addWidget(crop_label)
        row_crop.addStretch()
        row_crop.addWidget(self.crop_enabled_checkbox)
        
        geo_layout.addLayout(row_crop)
        
        geo_layout.addLayout(row_crop)
        
        control_layout.addWidget(geo_group)

        # ==========================================================
        # GROUP 1.5: Registration (New) - TEMPORARILY DISABLED
        # ==========================================================
        if False:
            reg_group = CollapsibleBox("Registration")
            reg_group.toggle_button.setProperty("class", "collapsible-toggle")
            reg_layout = reg_group.content_layout
            
            if not HAS_PYSTACKREG:
                warn_lbl = QtWidgets.QLabel("pystackreg library missing.\nPlease pip install pystackreg.")
                warn_lbl.setStyleSheet("color: red; font-weight: bold;")
                reg_layout.addWidget(warn_lbl)
            
            row_reg = QtWidgets.QHBoxLayout()
            reg_label = QtWidgets.QLabel("Mode:")
            
            self.reg_mode_combo = QtWidgets.QComboBox()
            self.reg_mode_combo.addItems([
                "None", "Translation", "Rigid Body", 
                "Scaled Rotation", "Affine", "Bilinear"
            ])
            if not HAS_PYSTACKREG:
                self.reg_mode_combo.setEnabled(False)
            self.reg_mode_combo.currentIndexChanged.connect(self._on_reg_mode_changed)
            
            self.reg_enabled_checkbox = QtWidgets.QCheckBox("Use")
            self.reg_enabled_checkbox.setToolTip("Align images to the current image (Reference) when clicking 'Apply All'.")
            self.reg_enabled_checkbox.setChecked(False)
            if not HAS_PYSTACKREG:
                self.reg_enabled_checkbox.setEnabled(False)
            self.reg_enabled_checkbox.stateChanged.connect(self._on_reg_enabled_changed)
            
            row_reg.addWidget(reg_label)
            row_reg.addWidget(self.reg_mode_combo, 1)
            row_reg.addWidget(self.reg_enabled_checkbox)
            
            reg_layout.addLayout(row_reg)
            
            # Info label
            reg_info = QtWidgets.QLabel("Apply All to register others to this image.")
            reg_info.setStyleSheet("color: gray; font-style: italic;")
            reg_layout.addWidget(reg_info)
            
            control_layout.addWidget(reg_group)

        # ==========================================================
        # GROUP 2: Masking (Now includes NoData)
        # ==========================================================
        mask_group = CollapsibleBox("Masking")
        mask_group.toggle_button.setProperty("class", "collapsible-toggle")
        mask_layout = mask_group.content_layout
        
        # -- 2a. NoData (Moved here) --
        row_nodata = QtWidgets.QHBoxLayout()
        nodata_label = QtWidgets.QLabel("NoData:")
        
        self.nodata_input = QtWidgets.QLineEdit()
        self.nodata_input.setPlaceholderText("-9999, 0, b1<123")
        self.nodata_input.setToolTip("Comma-separated list of values or expressions to ignore.")
        _fixw(self.nodata_input, 110)
        self.nodata_input.editingFinished.connect(self._on_nodata_input_changed)
        
        self.nodata_pick_btn = QtWidgets.QPushButton("Pick")
        self.nodata_pick_btn.setFixedSize(40, 22)
        self.nodata_pick_btn.setCheckable(True)
        self.nodata_pick_btn.clicked.connect(self._on_nodata_pick_clicked)
        
        self.nodata_apply_btn = QtWidgets.QPushButton("Apply")
        self.nodata_apply_btn.setFixedSize(45, 22)
        self.nodata_apply_btn.clicked.connect(self._on_nodata_input_changed)
        
        self.nodata_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.nodata_enabled_checkbox.setChecked(True)
        self.nodata_enabled_checkbox.stateChanged.connect(self._on_nodata_enabled_changed)
        
        row_nodata.addWidget(nodata_label)
        row_nodata.addWidget(self.nodata_input)
        row_nodata.addWidget(self.nodata_pick_btn)
        row_nodata.addWidget(self.nodata_apply_btn)
        row_nodata.addStretch()
        row_nodata.addWidget(self.nodata_enabled_checkbox)
        mask_layout.addLayout(row_nodata)
        
        # -- 2b. Mask Polygons --
        row_mask = QtWidgets.QHBoxLayout()
        mask_poly_label = QtWidgets.QLabel("Polygons:")
        
        self.mask_polygon_btn = QtWidgets.QToolButton()
        self.mask_polygon_btn.setText("Select...")
        self.mask_polygon_btn.setMinimumWidth(80)
        self.mask_polygon_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.mask_polygon_menu = QtWidgets.QMenu(self.mask_polygon_btn)
        self.mask_polygon_btn.setMenu(self.mask_polygon_menu)
        self._populate_mask_polygon_menu()
        
        self.mask_polygon_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.mask_polygon_enabled_checkbox.setChecked(False)
        self.mask_polygon_enabled_checkbox.stateChanged.connect(self._on_mask_polygon_enabled_changed)
        
        self.mask_polygon_refresh_btn = QtWidgets.QPushButton("↻")
        self.mask_polygon_refresh_btn.setFixedSize(22, 22)
        self.mask_polygon_refresh_btn.clicked.connect(self._populate_mask_polygon_menu)
        
        self.mask_polygon_count_label = QtWidgets.QLabel("(0)")
        self.mask_polygon_count_label.setStyleSheet("color: gray; font-size: 10px;")
        
        row_mask.addWidget(mask_poly_label)
        row_mask.addWidget(self.mask_polygon_btn)
        row_mask.addWidget(self.mask_polygon_refresh_btn)
        row_mask.addWidget(self.mask_polygon_count_label)
        row_mask.addStretch()
        row_mask.addWidget(self.mask_polygon_enabled_checkbox)
        
        mask_layout.addLayout(row_mask)
        control_layout.addWidget(mask_group)
        self._sync_mask_polygon_from_mods()

        # ==========================================================
        # GROUP 3: Image Operations (Renamed from Pixel Ops)
        # ==========================================================
        img_ops_group = CollapsibleBox("Image Operations")
        img_ops_group.toggle_button.setProperty("class", "collapsible-toggle")
        img_ops_layout = img_ops_group.content_layout
        
        # -- 3a. Band Expression --
        row_band = QtWidgets.QHBoxLayout()
        band_label = QtWidgets.QLabel("Band Expr:")
        
        self.band_input = QtWidgets.QLineEdit()
        self.band_input.setPlaceholderText("(b1>150) & (b2<200)")
        self.band_input.setMinimumWidth(140)
        
        self.band_apply_button = QtWidgets.QPushButton("Apply")
        self.band_apply_button.setFixedHeight(22)
        self.band_apply_button.clicked.connect(self.apply_band_expression)
        
        self.band_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.band_enabled_checkbox.setChecked(True)
        self.band_enabled_checkbox.stateChanged.connect(self._on_band_enabled_changed)
        
        row_band.addWidget(band_label)
        row_band.addWidget(self.band_input)
        row_band.addWidget(self.band_apply_button)
        row_band.addStretch()
        row_band.addWidget(self.band_enabled_checkbox)
        img_ops_layout.addLayout(row_band)
        
        # -- 3b. Histogram --
        row_hist = QtWidgets.QHBoxLayout()
        hist_label = QtWidgets.QLabel("Hist Norm:")
        
        self.hist_mode_combo = QtWidgets.QComboBox()
        self.hist_mode_combo.addItems(["None", "Mean/Std", "CDF"])
        self._sync_hist_combo_from_mods()
        _fixw(self.hist_mode_combo, 100)
        
        self.hist_btn = QtWidgets.QPushButton("Calc")
        self.hist_btn.clicked.connect(self.on_hist_match_clicked)
        self.hist_btn.setEnabled(self.original_image is not None and getattr(self.original_image, "size", 0) > 0)
        
        self.hist_enabled_checkbox = QtWidgets.QCheckBox("Use")
        self.hist_enabled_checkbox.setChecked(True)
        self.hist_enabled_checkbox.stateChanged.connect(self._on_hist_enabled_changed)
        
        row_hist.addWidget(hist_label)
        row_hist.addWidget(self.hist_mode_combo)
        row_hist.addWidget(self.hist_btn)
        row_hist.addStretch()
        row_hist.addWidget(self.hist_enabled_checkbox)
        img_ops_layout.addLayout(row_hist)
        
        control_layout.addWidget(img_ops_group)

        # ==========================================================
        # GROUP 4: Classification
        # ==========================================================
        cls_group = CollapsibleBox("Classification")
        cls_group.toggle_button.setProperty("class", "collapsible-toggle")
        
        self.use_sklearn_checkbox = QtWidgets.QCheckBox("Use scikit-learn")
        self.use_sklearn_checkbox.toggled.connect(self._on_use_sklearn_toggled)
        
        self.classify_btn = QtWidgets.QPushButton("Classify")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.on_run_classification_clicked)
        
        self.append_band_btn = QtWidgets.QPushButton("+ Append Band")
        self.append_band_btn.setToolTip("Append current result as extra band")
        self.append_band_btn.setEnabled(False)
        self.append_band_btn.clicked.connect(self._on_append_band_clicked)
        
        row_cls = QtWidgets.QHBoxLayout()
        row_cls.addWidget(self.use_sklearn_checkbox)
        row_cls.addWidget(self.classify_btn)
        row_cls.addWidget(self.append_band_btn)
        row_cls.addStretch()
        
        cls_group.content_layout.addLayout(row_cls)
        control_layout.addWidget(cls_group)
        
        # Reflect saved state
        try:
            clf = (self.modifications or {}).get("classification") or {}
            if bool(clf.get("enabled", False)):
                self.use_sklearn_checkbox.setChecked(True)
                self.classify_btn.setEnabled(True)
                self.append_band_btn.setEnabled(True)
        except Exception:
            pass

        # ==========================================================
        # GROUP 5: .ax JSON Editor
        # ==========================================================
        ax_group = CollapsibleBox("Edit .ax file (JSON)")
        ax_group.toggle_button.setProperty("class", "collapsible-toggle")
        
        self.ax_json_edit = QtWidgets.QPlainTextEdit()
        self.ax_json_edit.setPlaceholderText("JSON content will appear here...")
        self.ax_json_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.ax_json_edit.setMinimumHeight(120)
        self.ax_json_edit.setMaximumHeight(200)
        # Use monospace font for JSON
        json_font = QtGui.QFont("Consolas", 8)
        self.ax_json_edit.setFont(json_font)
        
        ax_content_layout = ax_group.content_layout
        ax_content_layout.addWidget(self.ax_json_edit)
        
        # Buttons for JSON editing
        ax_btn_layout = QtWidgets.QHBoxLayout()
        ax_btn_layout.setContentsMargins(0, 0, 0, 0)
        ax_btn_layout.setSpacing(4)
        
        self.ax_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.ax_refresh_btn.setToolTip("Reload JSON from the current .ax file on disk")
        self.ax_refresh_btn.setFixedWidth(70)
        self.ax_refresh_btn.clicked.connect(self._on_ax_refresh_clicked)
        
        self.ax_apply_btn = QtWidgets.QPushButton("Apply from JSON")
        self.ax_apply_btn.setToolTip("Parse the JSON and update all UI controls to match")
        self.ax_apply_btn.setFixedWidth(100)
        self.ax_apply_btn.clicked.connect(self._on_ax_apply_json_clicked)
        
        ax_btn_layout.addWidget(self.ax_refresh_btn)
        ax_btn_layout.addWidget(self.ax_apply_btn)
        ax_btn_layout.addStretch()
        
        ax_content_layout.addLayout(ax_btn_layout)
        
        control_layout.addWidget(ax_group)
        self._update_ax_json_display()
        
        # Toggle visibility when checked
        ax_group.toggle_button.toggled.connect(self._on_ax_group_toggled)

        # Spacer at bottom of controls
        control_layout.addStretch()

        # Add Bottom Controls to Sidebar (Scope + Action Buttons)
        # --- Scope Checkboxes ---
        self.global_mods_checkbox = QtWidgets.QCheckBox("Apply modifications to all images at this root")
        self.apply_all_groups_checkbox = QtWidgets.QCheckBox("Apply modifications to all roots")
        self.global_mods_checkbox.toggled.connect(self._on_group_apply_toggled)
        self.apply_all_groups_checkbox.toggled.connect(self._on_all_groups_toggled)
        
        scope_layout = QtWidgets.QVBoxLayout()
        scope_layout.addWidget(self.global_mods_checkbox)
        scope_layout.addWidget(self.apply_all_groups_checkbox)
        sidebar_layout.addLayout(scope_layout)
        
        sidebar_layout.addSpacing(6)

        # --- Action Buttons ---
        btns_layout = QtWidgets.QGridLayout()
        btns_layout.setSpacing(4)
        
        apply_all_button = QtWidgets.QPushButton("Apply All Changes")
        apply_all_button.clicked.connect(self.apply_all_changes)
        apply_all_button.setStyleSheet("font-weight: bold; background-color: #006400; color: white;")
        
        reset_img_button = QtWidgets.QPushButton("Reset Image")
        reset_img_button.clicked.connect(self.on_reset_image)
        
        reset_group_button = QtWidgets.QPushButton("Reset Root")
        reset_group_button.clicked.connect(self.on_reset_group)
        
        reset_all_button = QtWidgets.QPushButton("Reset All")
        reset_all_button.clicked.connect(self.on_reset_all_groups)
        
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        # Grid layout for buttons (2 columns)
        btns_layout.addWidget(apply_all_button, 0, 0, 1, 2) # Span top
        btns_layout.addWidget(reset_img_button, 1, 0)
        btns_layout.addWidget(reset_group_button, 1, 1)
        btns_layout.addWidget(reset_all_button, 2, 0)
        btns_layout.addWidget(cancel_button, 2, 1)
        
        for i in range(btns_layout.count()):
            w = btns_layout.itemAt(i).widget()
            if w: w.setMinimumHeight(28)
            
        sidebar_layout.addLayout(btns_layout)

        # Add Sidebar to Main Layout
        main_layout.addWidget(sidebar_container)

        # Defaults: Expand Geometry and Masking
        geo_group.toggle_button.setChecked(True)
        mask_group.toggle_button.setChecked(True)
        img_ops_group.toggle_button.setChecked(True)
        
        self.resize(900, 600) # Compact default size
        
        # Sync UI from modifications (load cached values from .ax file)
        self._sync_ui_from_modifications()
        # Ensure append button state is correct after syncing
        self._update_append_button_state()

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
        
        # Sync UI from modifications (load cached values from .ax file)
        self._sync_ui_from_modifications()
        # Ensure append button state is correct after syncing
        self._update_append_button_state()

    def _sample_hwC(self, arr, max_side=None, max_samples=None, seed=42):
        import numpy as np
        if arr is None or getattr(arr, "size", 0) == 0:
            return arr
        if arr.ndim == 2:
            arr = arr[..., None]
        H, W, C = arr.shape
        max_side = int(max_side if max_side is not None else getattr(self, "HIST_SAMPLE_MAX", 1200))
        max_samples = int(max_samples if max_samples is not None else getattr(self, "HIST_MAX_SAMPLES", 250_000))
        s = max(1, int(np.sqrt((H * W) / float(max(1, max_side * max_side)))))
        samp = arr if s == 1 else arr[::s, ::s, :]
        n = samp.shape[0] * samp.shape[1]
        if n > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_samples, replace=False)
            samp = samp.reshape(-1, C)[idx]     # [K,C]
        return samp
        
    def on_hist_match_clicked(self):
        """
        Build/clear histogram-match parameters for the *current edit session only*.
        Does NOT write .ax or refresh project viewers here. Persist happens in
        'Apply All Changes'.
        """
        from PyQt5 import QtWidgets
        import numpy as np, logging
        import cv2

        # 1) Which method?
        mode_txt = None
        if hasattr(self, "hist_mode_combo") and self.hist_mode_combo is not None:
            try:
                mode_txt = (self.hist_mode_combo.currentText() or "").strip().lower()
            except Exception:
                mode_txt = ""
        mode = "none"
        if "mean/std" in mode_txt:
            mode = "meanstd"
        elif "cdf" in mode_txt:
            mode = "cdf"
        else:
            mode = "none"

        # 2) If None: clear pending hist_match and preview
        if mode == "none":
            if "hist_match" in self.modifications:
                self.modifications.pop("hist_match", None)
            logging.info("Histogram normalization disabled for this edit session (preview updated only).")
            self.reapply_modifications()
            return

        # 3) Reference = current edited pixels in editor (scientific)
        ref = getattr(self, "original_image", None)
        if ref is None or getattr(ref, "size", 0) == 0:
            QtWidgets.QMessageBox.warning(self, "No image", "Open an image in the editor first.")
            return
        ref_f = ref.astype(np.float32, copy=False)
        if ref_f.ndim == 2:
            ref_f = ref_f[..., None]
        C = int(ref_f.shape[2])
        H, W = ref_f.shape[:2]

        # 3b) Build combined mask (nodata + mask polygons) for reference stats
        combined_mask = None
        
        # Get nodata values
        mods = self.modifications or {}
        nodata_enabled = mods.get("nodata_enabled", True)
        nodata_values = []
        if nodata_enabled:
            nodata_values = list(mods.get("nodata_values", []) or [])
        
        # Build nodata mask using shared utility (supports expressions like b1>143)
        if nodata_values:
            from .utils import build_nodata_mask as _shared_build_nodata_mask
            # CRITICAL: Use shared build_nodata_mask which supports both numeric values
            # AND expression-based NoData like "b1>143". This ensures the histogram
            # stats are calculated excluding all NoData pixels.
            combined_mask = _shared_build_nodata_mask(ref_f, nodata_values, bgr_input=True)
            if combined_mask is not None:
                logging.info(f"[on_hist_match_clicked] Built NoData mask with {combined_mask.sum()} masked pixels using shared utility")
        
        # Get mask polygon settings
        mask_polygon_cfg = mods.get("mask_polygon", {}) or {}
        mask_polygon_enabled = bool(mask_polygon_cfg.get("enabled", False)) if isinstance(mask_polygon_cfg, dict) else False
        mask_polygon_names = mask_polygon_cfg.get("names", []) if isinstance(mask_polygon_cfg, dict) else []
        
        logging.info(f"[on_hist_match_clicked] mask_polygon_enabled={mask_polygon_enabled}, names={mask_polygon_names}")
        
        # Build polygon mask
        if mask_polygon_enabled and mask_polygon_names:
            # Get polygon points - try direct lookup from parent.all_polygons
            parent = self.parent()
            all_polygons = getattr(parent, 'all_polygons', {}) if parent else {}
            fp = getattr(self, 'image_filepath', None)
            
            logging.info(f"[on_hist_match_clicked] image_filepath={fp}")
            logging.info(f"[on_hist_match_clicked] all_polygons groups: {list(all_polygons.keys()) if all_polygons else 'None'}")
            
            if fp and all_polygons:
                fp_norm = os.path.normpath(fp).lower()
                poly_mask = np.zeros((H, W), dtype=bool)
                found_count = 0
                
                for group_name, file_map in all_polygons.items():
                    if not isinstance(file_map, dict):
                        logging.debug(f"[on_hist_match_clicked] group '{group_name}' is not a dict, skipping")
                        continue
                    
                    # Log all filepaths in this group
                    logging.debug(f"[on_hist_match_clicked] group '{group_name}' has {len(file_map)} files")
                    
                    for stored_fp, poly_data in file_map.items():
                        stored_fp_norm = os.path.normpath(stored_fp).lower() if stored_fp else ""
                        
                        # Check for path match
                        paths_match = (stored_fp_norm == fp_norm)
                        if not paths_match:
                            # Try basename match as fallback
                            if os.path.basename(stored_fp_norm) == os.path.basename(fp_norm):
                                paths_match = True
                                logging.debug(f"[on_hist_match_clicked] Basename match: {os.path.basename(stored_fp_norm)}")
                        
                        if paths_match:
                            if isinstance(poly_data, dict):
                                poly_name = poly_data.get('name', group_name)
                                logging.info(f"[on_hist_match_clicked] Found polygon '{poly_name}' for this image (looking for: {mask_polygon_names})")
                                if poly_name in mask_polygon_names:
                                    points = poly_data.get('points', [])
                                    if points and len(points) >= 3:
                                        found_count += 1
                                        # Scale points if needed
                                        ref_size = poly_data.get('image_ref_size', {}) or {}
                                        ref_w = ref_size.get('w', 0) or 0
                                        ref_h = ref_size.get('h', 0) or 0
                                        if ref_w > 0 and ref_h > 0 and (ref_w != W or ref_h != H):
                                            scale_x = W / float(ref_w)
                                            scale_y = H / float(ref_h)
                                            scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]
                                            logging.debug(f"[on_hist_match_clicked] Scaled polygon from {ref_w}x{ref_h} to {W}x{H}")
                                        else:
                                            scaled_points = points
                                        
                                        pts = np.array([[int(round(x)), int(round(y))] for x, y in scaled_points], dtype=np.int32)
                                        pts = pts.reshape((-1, 1, 2))
                                        mask_temp = np.zeros((H, W), dtype=np.uint8)
                                        cv2.fillPoly(mask_temp, [pts], 255)
                                        poly_mask |= (mask_temp > 0)
                                        logging.info(f"[on_hist_match_clicked] Added polygon '{poly_name}' with {len(points)} points, mask now has {poly_mask.sum()} pixels")
                                    else:
                                        logging.warning(f"[on_hist_match_clicked] Polygon '{poly_name}' has insufficient points: {len(points) if points else 0}")
                                else:
                                    logging.debug(f"[on_hist_match_clicked] Polygon '{poly_name}' not in mask_polygon_names")
                
                logging.info(f"[on_hist_match_clicked] Found {found_count} matching mask polygons, total masked pixels: {poly_mask.sum()}")
                
                if poly_mask.any():
                    if combined_mask is None:
                        combined_mask = poly_mask
                    else:
                        combined_mask = combined_mask | poly_mask
                    logging.info(f"[on_hist_match_clicked] Combined mask has {combined_mask.sum()} pixels ({100*combined_mask.sum()/(H*W):.1f}%)")
            else:
                if not fp:
                    logging.warning("[on_hist_match_clicked] No image_filepath set!")
                if not all_polygons:
                    logging.warning("[on_hist_match_clicked] all_polygons is empty or None!")
        else:
            logging.info("[on_hist_match_clicked] Mask polygon not enabled or no names selected")

        # 4) Sample for speed (with mask support)
        def _sample_hwC_masked(arr, mask=None, max_side=None, max_samples=None, seed=42):
            """Sample pixels from array, excluding masked pixels."""
            if arr is None or getattr(arr, "size", 0) == 0:
                return arr
            if arr.ndim == 2:
                arr = arr[..., None]
            H, W, Cc = arr.shape
            max_side = int(max_side if max_side is not None else getattr(self, "HIST_SAMPLE_MAX", 1200))
            max_samples = int(max_samples if max_samples is not None else getattr(self, "HIST_MAX_SAMPLES", 250_000))
            
            if mask is not None and mask.any():
                # Flatten and filter out masked pixels
                arr_flat = arr.reshape(-1, Cc)  # [H*W, C]
                mask_flat = mask.ravel()  # [H*W]
                valid_pixels = arr_flat[~mask_flat]  # Only unmasked pixels
                
                logging.info(f"[_sample_hwC_masked] Total pixels: {arr_flat.shape[0]}, masked: {mask_flat.sum()}, valid: {valid_pixels.shape[0]}")
                
                if valid_pixels.size == 0:
                    logging.warning("[_sample_hwC_masked] All pixels are masked!")
                    return arr_flat[:1]  # Return at least one pixel to avoid crash
                
                n = valid_pixels.shape[0]
                if n > max_samples:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(n, size=max_samples, replace=False)
                    valid_pixels = valid_pixels[idx]
                return valid_pixels  # [K, C]
            else:
                logging.info(f"[_sample_hwC_masked] No mask applied, using all pixels")
                # No mask - use original sampling logic
                s = max(1, int(np.sqrt((H * W) / float(max(1, max_side * max_side)))))
                samp = arr if s == 1 else arr[::s, ::s, :]
                n = samp.shape[0] * samp.shape[1]
                if n > max_samples:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(n, size=max_samples, replace=False)
                    samp = samp.reshape(-1, Cc)[idx]  # [K,C]
                return samp

        ref_samp = _sample_hwC_masked(ref_f, mask=combined_mask,
                                       max_side=int(getattr(self, "HIST_SAMPLE_MAX", 1200)),
                                       max_samples=int(getattr(self, "HIST_MAX_SAMPLES", 250_000)), seed=42)
        is_flat = (ref_samp.ndim == 2)

        # 5) Build payload (Mean/Std or CDF)
        payload = {"mode": mode, "bands": C}

        if mode == "meanstd":
            if is_flat:
                mu = np.nanmean(ref_samp, axis=0).astype(np.float32, copy=False)
                sd = np.nanstd( ref_samp, axis=0).astype(np.float32, copy=False)
            else:
                mu = np.nanmean(ref_samp, axis=(0, 1)).astype(np.float32, copy=False)
                sd = np.nanstd( ref_samp, axis=(0, 1)).astype(np.float32, copy=False)
            sd = np.where(np.isfinite(sd) & (sd >= 1e-6), sd, 1.0)
            payload["ref_stats"] = [{"mean": float(mu[i]), "std": float(sd[i])} for i in range(C)]
            logging.info(f"[on_hist_match_clicked] Calculated ref_stats from {ref_samp.shape[0]} samples: mean={mu.tolist()}, std={sd.tolist()}")

        elif mode == "cdf":
            per_band = []
            ref_use = ref_samp if is_flat else ref_samp.reshape(-1, ref_samp.shape[2])
            for c in range(C):
                ch = ref_use[:, c]
                ch = ch[np.isfinite(ch)]
                if ch.size == 0:
                    per_band.append({"x": [0.0, 1.0], "y": [0.0, 1.0], "lo": 0.0, "hi": 1.0})
                    continue
                lo = float(np.nanpercentile(ch, 0.01))
                hi = float(np.nanpercentile(ch, 99.99))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = 0.0, 1.0
                bins = 1024
                hist, edges = np.histogram(ch, bins=bins, range=(lo, hi))
                cdf = np.cumsum(hist).astype(np.float32)
                cdf /= float(ch.size if ch.size else 1)
                x_centers = 0.5 * (edges[:-1] + edges[1:])
                x_n = ((x_centers - lo) / max(1e-12, (hi - lo))).astype(np.float32)
                per_band.append({"x": x_n.tolist(), "y": cdf.tolist(), "lo": float(lo), "hi": float(hi)})
            payload["ref_cdf"] = {"per_band": per_band}

        # 6) Store *pending* edit only; do not write files or refresh tabs here
        self.modifications["hist_match"] = payload
        logging.info(f"Histogram matching set to '{mode}' (stats computed). Will persist on 'Apply All Changes'.")

        # NOTE: Do NOT call reapply_modifications() here!
        # "Calculate" computes reference stats FROM this image for use on OTHER images.
        # Applying histogram matching to the reference image itself makes no sense.
    def _apply_hist_match(self, img, block):
        import numpy as np
        import cv2

        if (img is None) or (getattr(img, "size", 0) == 0) or (not isinstance(block, dict)):
            return img

        mode = (block.get("mode") or "meanstd").lower()
        if mode == "none":  # <-- allow explicit "no normalization"
            return img

        bands_cfg = int(block.get("bands", 0))
        src_dtype = img.dtype

        # Helper: sample stride so we don't scan full-res to estimate stats/CDF
        def _sample_stride(h, w, max_side):
            return max(1, int(np.sqrt((h * w) / float(max(1, max_side * max_side)))))

        # --- FAST PATH: mean/std with LUTs for integers ---
        if mode == "meanstd":
            ref = block.get("ref_stats") or []
            # Normalize to HxWxC for uniform logic below
            if img.ndim == 2:
                H, W, C = img.shape[0], img.shape[1], 1
            else:
                H, W, C = img.shape
            bands = min(bands_cfg if bands_cfg > 0 else C, C)
            C2 = min(bands, len(ref))
            if C2 == 0:
                return img

            # Sample target stats (cheap)
            max_side = int(getattr(self, "HIST_SAMPLE_MAX", 1200))
            s = _sample_stride(H, W, max_side)

            if np.issubdtype(src_dtype, np.integer):
                # Compute mu/sd from a sample casted to float (no need for nan-ops on ints)
                if img.ndim == 2:
                    samp = img[::s, ::s].astype(np.float32, copy=False)[..., None]
                else:
                    samp = img[::s, ::s, :C2].astype(np.float32, copy=False)

                mu_t = np.mean(samp, axis=(0, 1)).astype(np.float32, copy=False).reshape(-1)
                sd_t = np.std( samp, axis=(0, 1)).astype(np.float32, copy=False).reshape(-1)
                sd_t = np.where(sd_t < 1e-6, 1.0, sd_t)

                mu_r = np.array([float(ref[i].get("mean", 0.0)) for i in range(C2)], dtype=np.float32)
                sd_r = np.array([float(ref[i].get("std",  1.0)) for i in range(C2)], dtype=np.float32)

                # uint8: exact LUT 0..255
                if src_dtype == np.uint8:
                    xs = np.arange(256, dtype=np.float32)
                    luts = []
                    for c in range(C2):
                        a = float(sd_r[c] / sd_t[c]); b = float(mu_r[c] - a * mu_t[c])
                        lut_u8 = np.clip(a * xs + b, 0, 255).astype(np.uint8)
                        luts.append(lut_u8)

                    out = img.copy()
                    if out.ndim == 2:
                        out[:] = luts[0][out]
                    else:
                        for c in range(C2):
                            out[..., c] = luts[c][out[..., c]]
                    return out

                # uint16: 12-bit quantized LUT (fast and small)
                elif src_dtype == np.uint16:
                    qbins = int(getattr(self, "HIST_BINS_16U", 4096))
                    step  = 65536.0 / qbins
                    xs    = (np.arange(qbins, dtype=np.float32) + 0.5) * step  # centers
                    luts = []
                    for c in range(C2):
                        a = float(sd_r[c] / sd_t[c]); b = float(mu_r[c] - a * mu_t[c])
                        lut_u16 = np.clip(a * xs + b, 0.0, 65535.0).astype(np.uint16)
                        luts.append(lut_u16)

                    out = img.copy()
                    if out.ndim == 2:
                        qidx = np.clip((out.astype(np.float32) * (1.0 / step)).astype(np.int32), 0, qbins - 1)
                        out[:] = luts[0][qidx]
                    else:
                        for c in range(C2):
                            qidx = np.clip((out[..., c].astype(np.float32) * (1.0 / step)).astype(np.int32), 0, qbins - 1)
                            out[..., c] = luts[c][qidx]
                    return out

                # Other integer types: fall back to float math
                # (rare in images; keeps behavior identical)
                # Continue to float path below.

            # --- float path (and non-u8/u16 ints fallback) ---
            f = img.astype(np.float32, copy=False)
            single_channel = False
            if f.ndim == 2:
                f = f[..., None]
                single_channel = True
            # recompute sample on float view
            s = _sample_stride(f.shape[0], f.shape[1], int(getattr(self, "HIST_SAMPLE_MAX", 1200)))
            samp = f[:, :, :C2] if s == 1 else f[::s, ::s, :C2]
            mu_t = np.nanmean(samp, axis=(0, 1)).astype(np.float32, copy=False)
            sd_t = np.nanstd( samp, axis=(0, 1)).astype(np.float32, copy=False)
            sd_t = np.where(sd_t < 1e-6, 1.0, sd_t)

            mu_r = np.array([float(ref[i].get("mean", 0.0)) for i in range(C2)], dtype=np.float32)
            sd_r = np.array([float(ref[i].get("std",  1.0)) for i in range(C2)], dtype=np.float32)

            gain = (sd_r / sd_t).reshape(1, 1, C2)
            x = f[:, :, :C2]
            x -= mu_t.reshape(1, 1, C2)
            x *= gain
            x += mu_r.reshape(1, 1, C2)

            out = f[..., 0] if single_channel else f
            if np.issubdtype(src_dtype, np.integer):
                info = np.iinfo(src_dtype)
                out = np.clip(out, info.min, info.max).astype(src_dtype, copy=False)
            return out

        # --- CDF path (unchanged, but convert lazily) ---
        # Convert once here for CDF logic
        f = img.astype(np.float32, copy=False)
        single_channel = False
        if f.ndim == 2:
            f = f[..., None]
            single_channel = True

        H, W, C = f.shape
        bands = min(bands_cfg if bands_cfg > 0 else C, C)

        # small, safe sampler
        def _sample_for_hist_channel(ch):
            max_side = int(getattr(self, "HIST_SAMPLE_MAX", 1200))
            h, w = ch.shape[:2]
            if h == 0 or w == 0:
                return ch
            stride = _sample_stride(h, w, max_side)
            return ch if stride == 1 else ch[::stride, ::stride]

        refcdf = (block.get("ref_cdf") or {}).get("per_band") or []
        ref_hash = self._ref_hash_hist(block)

        dtype_tag = "float"
        bins_float = getattr(self, "HIST_BINS_FLOAT", 2048)
        bins_8u   = getattr(self, "HIST_BINS_8U", 256)
        bins_16u  = getattr(self, "HIST_BINS_16U", 4096)

        for c in range(min(bands, len(refcdf))):
            ch = f[..., c]
            info = refcdf[c]
            lo = float(info.get("lo", 0.0)); hi = float(info.get("hi", 1.0))
            x_ref_n = np.asarray(info.get("x", [0.0, 1.0]), dtype=np.float32)
            y_ref   = np.asarray(info.get("y", [0.0, 1.0]), dtype=np.float32)
            hi = hi if hi > lo else (lo + 1.0)
            x_ref   = x_ref_n * (hi - lo) + lo

            # Cache key (per band, per dtype/binning policy)
            if np.issubdtype(src_dtype, np.integer):
                if src_dtype == np.uint8:
                    dtype_tag = "u8"
                    cache_key = ("cdf", ref_hash, dtype_tag, bins_8u, c)
                else:
                    dtype_tag = "u16"
                    cache_key = ("cdf", ref_hash, dtype_tag, bins_16u, c)
            else:
                dtype_tag = "float"
                cache_key = ("cdf", ref_hash, dtype_tag, bins_float, c)

            xs = xprime = None
            if cache_key in getattr(self, "_hist_cache", {}):
                xs, xprime = self._hist_cache[cache_key]
            else:
                ch_s = _sample_for_hist_channel(ch)
                flat = ch_s.reshape(-1)
                flat = flat[~np.isnan(flat)]
                if flat.size == 0:
                    continue

                if dtype_tag == "u8":
                    xs_src = np.arange(256, dtype=np.float32)
                    hist = np.bincount(ch_s.astype(np.uint8, copy=False).ravel(), minlength=256).astype(np.float32)
                    cdf_src = np.cumsum(hist) / max(1.0, float(hist.sum()))
                    xprime_at_p = np.interp(cdf_src, y_ref, x_ref)
                    lut = np.interp(xs_src, xs_src, xprime_at_p).astype(np.float32)
                    xs, xprime = xs_src, lut

                elif dtype_tag == "u16":
                    qbins = int(bins_16u)
                    step = 65536.0 / qbins
                    qidx = np.clip((ch_s.astype(np.float32) * (1.0 / step)).astype(np.int32), 0, qbins - 1)
                    hist = np.bincount(qidx.ravel(), minlength=qbins).astype(np.float32)
                    cdf_src = np.cumsum(hist) / max(1.0, float(hist.sum()))
                    xs_src = (np.arange(qbins, dtype=np.float32) + 0.5) * step
                    xprime_at_p = np.interp(cdf_src, y_ref, x_ref)
                    lut = np.interp(xs_src, xs_src, xprime_at_p).astype(np.float32)
                    xs, xprime = xs_src, lut

                else:
                    hist, edges = np.histogram(flat, bins=int(bins_float), range=(lo, hi))
                    cdf_src = np.cumsum(hist).astype(np.float32)
                    cdf_src /= float(flat.size if flat.size else 1.0)
                    xs_src = 0.5 * (edges[:-1] + edges[1:])
                    xprime = np.interp(cdf_src, y_ref, x_ref).astype(np.float32)
                    xs = xs_src.astype(np.float32)

                if not hasattr(self, "_hist_cache"):
                    self._hist_cache = {}
                self._hist_cache[cache_key] = (xs, xprime)

            # Apply mapping to full channel
            if dtype_tag == "u8":
                ch_u8 = np.clip(ch, 0, 255).astype(np.uint8, copy=False)
                ch[:] = xprime[ch_u8]
            elif dtype_tag == "u16":
                qbins = int(bins_16u)
                step = 65536.0 / qbins
                qidx_full = np.clip((ch * (1.0 / step)).astype(np.int32), 0, qbins - 1)
                ch[:] = xprime[qidx_full]
            else:
                ch[:] = np.interp(ch, xs, xprime, left=xprime[0], right=xprime[-1]).astype(np.float32)

        out = f[..., 0] if single_channel else f
        if np.issubdtype(src_dtype, np.integer):
            info = np.iinfo(src_dtype)
            out = np.clip(out, info.min, info.max).astype(src_dtype, copy=False)
        return out

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
    def _delete_ax_in_dir(self, base_dir: str, progress_dialog=None) -> int:
        """Stream-delete *.ax files under base_dir using multithreading. Returns count deleted."""
        from PyQt5 import QtWidgets
        import concurrent.futures
        
        deleted = 0
        try:
            # First, collect all .ax files
            ax_files = []
            for root, _dirs, files in os.walk(base_dir):
                for name in files:
                    if name.lower().endswith(".ax"):
                        ax_files.append(os.path.join(root, name))
            
            total = len(ax_files)
            if not ax_files:
                return 0

            if progress_dialog:
                progress_dialog.setMaximum(total)
                progress_dialog.setLabelText(f"Deleting .ax files (0/{total})...")
                progress_dialog.setValue(0)
                QtWidgets.QApplication.processEvents()
            
            # Helper for deletion
            def _remove(p):
                try:
                    os.remove(p)
                    return True
                except Exception as e:
                    logging.debug(f"Could not delete {p}: {e}")
                    return False

            # Delete with progress updates using ThreadPool
            # Limit workers to avoid system strain, but typically I/O bound
            completed = 0
            update_stride = max(1, total // 100)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, total + 1)) as executor:
                futures = {executor.submit(_remove, dp): dp for dp in ax_files}
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    if future.result():
                        deleted += 1
                    
                    if progress_dialog:
                        # Throttle updates
                        if completed == total or completed % update_stride == 0:
                            progress_dialog.setValue(completed)
                            progress_dialog.setLabelText(f"Deleting .ax files ({completed}/{total})...")
                            QtWidgets.QApplication.processEvents()
                    
        except Exception as e:
            logging.error(f"Walking '{base_dir}' failed: {e}")
        return deleted

    def on_reset_image(self):
        """Delete .ax for THIS image and reset view (no popups)."""
        self.modifications = {}
        self.last_band_float_result = None
        self._classification_result = None
        self.crop_rect = None
        self.last_crop_rect = None
        self.last_crop_ref_size = None

        # ZOMBIE FIX: Delete ALL candidate .ax files (Project Folder + Sidecar)
        candidates = self._get_ax_candidates(self.image_filepath)
        logging.debug(f"Resetting image modifications. Candidates: {candidates}")
        
        for mod_filename in candidates:
            if os.path.exists(mod_filename):
                try:
                    os.remove(mod_filename)
                    logging.info(f"Modifications file {mod_filename} deleted.")
                except Exception as e:
                    logging.error(f"Failed to delete modifications file {mod_filename}: {e}")

        # FIX: Invalidate parent caches ONCE after deletion loop
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, "invalidate_caches_for_file"):
                parent.invalidate_caches_for_file(self.image_filepath)
        except Exception:
            pass

        # Try to reload raw image from disk
        raw = self._load_raw_image()
        
        # CRITICAL FIX: If _load_raw_image() returns None (can happen for some TIFs),
        # fall back to using the existing base_image which was loaded at dialog init.
        if raw is not None:
            self.base_image = raw
        elif self.base_image is None:
            logging.warning("[Reset] No raw image available - cannot reset display")
            return
        else:
            logging.debug("[Reset] _load_raw_image returned None, using existing base_image")
        
        # Always update display with the (possibly unchanged) base_image
        self.original_image = self.base_image.copy()
        self.display_image_data = self.normalize_image_for_display(self.original_image)
        self.display_image(self.display_image_data)

        # Centralized UI reset
        self._sync_ui_from_modifications()

        root_name = None
        parent = self.parent()
        if parent is not None and hasattr(parent, "get_current_root_name"):
            try:
                root_name = parent.get_current_root_name()
            except Exception:
                root_name = None
        #self._auto_refresh_after_reset(root_name=root_name)
        logging.info("Image modifications reset.")

    def on_reset_group(self):
        """Delete .ax for all images in current group and auto-refresh (no popups)."""
        import os
        from PyQt5 import QtWidgets, QtCore
        
        parent = self.parent()
        files = []
        root_name = None

        if parent is None:
            if self.image_filepath:
                files = [self.image_filepath]
            return

        # DUAL FOLDER FIX: Detect which folder group this image belongs to
        def _norm_path(p):
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(p)))
            except Exception:
                return ""
        
        folder_groups = None
        cur_fp_norm = _norm_path(self.image_filepath) if self.image_filepath else ""
        
        # Check multispectral first
        if hasattr(parent, "multispectral_image_data_groups"):
            groups = getattr(parent, "multispectral_image_data_groups", None) or {}
            for rn, root_files in groups.items():
                if isinstance(root_files, (list, tuple)):
                    for fp in root_files:
                        if isinstance(fp, str) and _norm_path(fp) == cur_fp_norm:
                            folder_groups = groups
                            root_name = rn  # Found the root that contains this image
                            break
                if folder_groups:
                    break
        
        # Check thermal_rgb second
        if folder_groups is None and hasattr(parent, "thermal_rgb_image_data_groups"):
            groups = getattr(parent, "thermal_rgb_image_data_groups", None) or {}
            for rn, root_files in groups.items():
                if isinstance(root_files, (list, tuple)):
                    for fp in root_files:
                        if isinstance(fp, str) and _norm_path(fp) == cur_fp_norm:
                            folder_groups = groups
                            root_name = rn  # Found the root that contains this image
                            break
                if folder_groups:
                    break

        # Get files for current root from the correct folder group
        if root_name and folder_groups:
            files = list(folder_groups.get(root_name, []))

        if not files and self.image_filepath:
            files = [self.image_filepath]

        # Show progress dialog for multiple files
        progress = None
        total = len(files)
        if total > 1:
            progress = QtWidgets.QProgressDialog(f"Resetting group (0/{total})...", None, 0, total, self)
            progress.setWindowTitle("Reset Group")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QtWidgets.QApplication.processEvents()

        try:
            deleted = 0
            for i, fp in enumerate(files):
                try:
                    # ZOMBIE FIX: Delete all candidates for this file
                    candidates = self._get_ax_candidates(fp)
                    for mod_path in candidates:
                        if os.path.exists(mod_path):
                            os.remove(mod_path)
                            deleted += 1
                    
                    # FIX: Invalidate parent caches
                    if parent is not None and hasattr(parent, "invalidate_caches_for_file"):
                        parent.invalidate_caches_for_file(fp)
                except Exception as e:
                    logging.error(f"Failed to delete modifications for {fp}: {e}")
                
                if progress:
                    progress.setValue(i + 1)
                    progress.setLabelText(f"Resetting group ({i+1}/{total})...")
                    QtWidgets.QApplication.processEvents()

            if progress:
                progress.setLabelText("Reloading image...")
                QtWidgets.QApplication.processEvents()

            # reset this editor view
            self.modifications = {}
            self.last_band_float_result = None
            self._classification_result = None
            self.crop_rect = None
            self.last_crop_rect = None
            self.last_crop_ref_size = None

            # Try to reload raw image from disk
            raw = self._load_raw_image()
            
            # CRITICAL FIX: If _load_raw_image() returns None (can happen for some TIFs),
            # fall back to using the existing base_image which was loaded at dialog init.
            if raw is not None:
                self.base_image = raw
            elif self.base_image is None:
                logging.warning("[Reset Group] No raw image available - cannot reset display")
            else:
                logging.debug("[Reset Group] _load_raw_image returned None, using existing base_image")
            
            # Always update display if we have a base_image
            if self.base_image is not None:
                self.original_image = self.base_image.copy()
                self.display_image_data = self.normalize_image_for_display(self.original_image)
                self.display_image(self.display_image_data)

            # Centralized UI reset (handles NoData, checkboxes etc)
            self._sync_ui_from_modifications()

            # CRITICAL: Use multispectral root name for refresh (refresh_viewer expects it)
            ms_root_name = None
            try:
                if parent is not None and hasattr(parent, "get_current_root_name"):
                    ms_root_name = parent.get_current_root_name()
            except Exception:
                pass
            
            self._auto_refresh_after_reset(root_name=ms_root_name)
            logging.info(f"Reset Group: removed {deleted} .ax file(s) for root '{root_name or 'unknown'}'.")
        finally:
            if progress:
                progress.close()

    def on_reset_all_groups(self):
        """Delete every .ax under the project folder (or current image folder) and refresh once. (No popups)"""
        from PyQt5 import QtWidgets, QtCore
        
        base_dir = None
        if self.project_folder and self.project_folder.strip():
            base_dir = self.project_folder
        elif self.image_filepath:
            base_dir = os.path.dirname(self.image_filepath)

        if not base_dir or not os.path.isdir(base_dir):
            logging.warning("Reset All: couldn't locate a base folder to clean.")
            return

        # Show progress dialog
        progress = QtWidgets.QProgressDialog("Resetting all groups...", None, 0, 0, self)
        progress.setWindowTitle("Reset All")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            deleted = self._delete_ax_in_dir(base_dir, progress_dialog=progress)

            progress.setLabelText("Reloading image...")
            QtWidgets.QApplication.processEvents()

            # reset this editor view cheaply
            self.modifications = {}
            self.last_band_float_result = None
            self._classification_result = None
            self.crop_rect = None
            self.last_crop_rect = None
            self.last_crop_ref_size = None

            # Try to reload raw image from disk
            raw = self._load_raw_image()
            
            # CRITICAL FIX: If _load_raw_image() returns None (can happen for some TIFs),
            # fall back to using the existing base_image which was loaded at dialog init.
            if raw is not None:
                self.base_image = raw
            elif self.base_image is None:
                logging.warning("[Reset All] No raw image available - cannot reset display")
            else:
                logging.debug("[Reset All] _load_raw_image returned None, using existing base_image")
            
            # Always update display if we have a base_image
            if self.base_image is not None:
                self.original_image = self.base_image.copy()
                self.display_image_data = self.normalize_image_for_display(self.original_image)
                self.display_image(self.display_image_data)

            # Centralized UI reset
            self._sync_ui_from_modifications()

            # unified auto-refresh
            self._auto_refresh_after_reset(root_name=None)
            logging.info(f"Reset All: removed {deleted} .ax file(s) under '{base_dir}'.")
        finally:
            progress.close()

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



    def display_image(self, image_or_pixmap):
        """
        Displays an image, fitting it to the window or applying zoom.
        Handles pre-rendered QPixmaps (already display-ready) and raw NumPy arrays.
        NumPy 3-channel images are converted from BGR→RGB for UI only.
        """
        from PyQt5 import QtGui, QtCore
        import numpy as np, sip, logging

        if image_or_pixmap is None:
            self.image_label.setText("No Image Loaded")
            logging.warning("No image provided to display.")
            return

        try:
            # --- Determine the full-resolution pixmap ---
            if isinstance(image_or_pixmap, QtGui.QPixmap):
                # Assume already display-ready (e.g., from convert_cv_to_pixmap)
                pix = image_or_pixmap
            else:
                disp = image_or_pixmap

                # Ensure displayable uint8
                if getattr(disp, "dtype", None) != np.uint8 or getattr(disp, "ndim", None) not in (2, 3):
                    disp = self.normalize_image_for_display(disp)
                if disp is None:
                    self.image_label.setText("Normalization Failed.")
                    return

                disp = np.ascontiguousarray(disp)

                # Collapse (H,W,1) to (H,W) so QImage uses Format_Grayscale8
                if disp.ndim == 3 and disp.shape[2] == 1:
                    disp = disp.reshape(disp.shape[0], disp.shape[1])

                # Build QImage with proper format (convert BGR→RGB here only)
                if disp.ndim == 2:
                    h, w = disp.shape
                    qimg = QtGui.QImage(
                        sip.voidptr(disp.ctypes.data), w, h, disp.strides[0],
                        QtGui.QImage.Format_Grayscale8
                    )
                else:
                    h, w, _ = disp.shape
                    disp_rgb = disp[:, :, ::-1].copy()  # BGR -> RGB
                    qimg = QtGui.QImage(
                        sip.voidptr(disp_rgb.ctypes.data), w, h, disp_rgb.strides[0],
                        QtGui.QImage.Format_RGB888
                    )

                pix = QtGui.QPixmap.fromImage(qimg.copy())

            if pix.isNull():
                self.image_label.setText("Failed to create Pixmap.")
                return

            # --- Apply scaling (fit-to-window or zoom) ---
            fit = bool(getattr(self, "fit_to_window", True))
            z = float(getattr(self, "zoom", 1.0))

            if fit:
                vp = self.scroll_area.viewport().size()
                shown = pix.scaled(vp, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            else:
                if z != 1.0:
                    shown = pix.scaled(
                        int(pix.width() * z), int(pix.height() * z),
                        QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
                    )
                else:
                    shown = pix

            self.image_label.setPixmap(shown)
            self.image_label.resize(shown.size())

            sc = self.current_display_scale_percent()
            if sc is not None:
                self.setWindowTitle(f"Edit Image Viewer - {sc:.1f}%")

        except Exception as e:
            self.image_label.setText("Error Displaying Image")
            logging.error(f"Error during image display: {e}", exc_info=True)

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
        Map a rect drawn on the *resized* image back to pre-resize coordinates.

        Supports:
          • {"scale": pct}
          • {"width": pct_w, "height": pct_h}  # legacy percent pair
          • {"px_w": W, "px_h": H}             # absolute pixel target
        Uses optional self.modifications["resize_ref_size"] = {"w": pre_w, "h": pre_h}
        for exact inversion in the px case.
        """
        info = self.modifications.get("resize", None)
        if not info:
            return x, y, w, h, int(view_w), int(view_h)

        # Default: assume current view is the resized result.
        # Derive inverse scale factors.
        sx_inv = sy_inv = 1.0
        pre_w = int(view_w)
        pre_h = int(view_h)

        ref = self.modifications.get("resize_ref_size", {}) or {}
        ref_w = int(ref.get("w", view_w)) if view_w else 0
        ref_h = int(ref.get("h", view_h)) if view_h else 0

        if "px_w" in info or "px_h" in info:
            # Absolute pixels
            tgt_w = int(info.get("px_w", view_w)) or int(view_w)
            tgt_h = int(info.get("px_h", view_h)) or int(view_h)
            if tgt_w <= 0 or tgt_h <= 0:
                return x, y, w, h, int(view_w), int(view_h)
            # Inverse scales = (pre / post)
            pre_w = ref_w or int(round(view_w * 1.0))  # fallback if no ref size
            pre_h = ref_h or int(round(view_h * 1.0))
            sx_inv = (pre_w / float(tgt_w)) if tgt_w else 1.0
            sy_inv = (pre_h / float(tgt_h)) if tgt_h else 1.0

        elif "scale" in info:
            s = max(1, int(info["scale"])) / 100.0
            sx_inv = 1.0 / s
            sy_inv = 1.0 / s
            pre_w = int(round(view_w * sx_inv))
            pre_h = int(round(view_h * sy_inv))

        else:
            # legacy % pair
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

        return x2, y2, w2, h2, int(pre_w), int(pre_h)

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
            # Check if NoData picker mode is active
            if getattr(self, '_nodata_picker_active', False):
                self._pick_nodata_at_pos(event.pos())
                return  # Don't start crop selection in picker mode
            
            self.origin = event.pos()
            self.rubber_band.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubber_band.show()
            logging.debug("Mouse pressed for cropping.")

    def image_mouse_move_event(self, event):
        # Don't show rubber band in picker mode
        if getattr(self, '_nodata_picker_active', False):
            return
        if not self.origin.isNull():
            rect = QtCore.QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)
            logging.debug("Mouse moved for cropping.")

    def image_mouse_release_event(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Don't process crop in picker mode
            if getattr(self, '_nodata_picker_active', False) or getattr(self, '_crop_disabled_for_pick', False):
                self.rubber_band.hide()
                self.origin = QtCore.QPoint()
                return
            
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

    def keyPressEvent(self, event):
        """Handle key presses - Escape exits NoData picker mode."""
        if event.key() == QtCore.Qt.Key_Escape:
            if getattr(self, '_nodata_picker_active', False):
                self._exit_nodata_picker_mode()
                event.accept()
                return
        super().keyPressEvent(event)

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

    def on_resize_pixels_entered(self):
        """Resize to exact (W,H) pixels. Stores resize_ref_size for precise crop inverse-mapping."""
        w_txt = self.resize_width_input.text().strip() if hasattr(self, "resize_width_input") else ""
        h_txt = self.resize_height_input.text().strip() if hasattr(self, "resize_height_input") else ""
        if not (w_txt and h_txt):
            logging.warning("Pixel resize skipped: both W and H are required.")
            return

        try:
            target_w = int(w_txt)
            target_h = int(h_txt)
        except ValueError:
            logging.warning("Pixel resize skipped: invalid integers.")
            return
        if target_w <= 0 or target_h <= 0:
            logging.warning("Pixel resize skipped: W and H must be positive.")
            return

        # Compute the image size *before* resize in the current pipeline so we can invert cropping later.
        mods_wo_resize = dict(self.modifications)
        mods_wo_resize.pop("resize", None)
        pre_img = self.apply_all_modifications_to_image(self.base_image, mods_wo_resize)
        if pre_img is None or pre_img.size == 0:
            logging.warning("Pixel resize skipped: could not compute pre-resize image.")
            return
        pre_h, pre_w = pre_img.shape[:2]

        self.modifications["resize"] = {"px_w": int(target_w), "px_h": int(target_h)}
        self.modifications["resize_ref_size"] = {"w": int(pre_w), "h": int(pre_h)}

        self.reapply_modifications()
        logging.info(f"Image resized to {target_w}×{target_h} px (ref {pre_w}×{pre_h}).")

    
    
    
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


    def get_modified_image(self):
        """
        Return the modified image. If classification is enabled and has a result,
        append the classification band to the image stack (like boolean math does).
        This ensures the viewer can display it with proper auto-stretch.
        """
        import numpy as np
        import logging
        
        base_img = self.original_image
        if base_img is None:
            logging.warning("get_modified_image: base_img is None")
            return None
        
        # Check if classification is enabled and we have a result
        cblock = (self.modifications or {}).get("classification") or {}
        cls_enabled = isinstance(cblock, dict) and bool(cblock.get("enabled", False))
        cls_result = getattr(self, "last_band_float_result", None)
        
        logging.info(f"get_modified_image: cls_enabled={cls_enabled}, cls_result is None: {cls_result is None}")
        logging.info(f"get_modified_image: base_img shape={base_img.shape}, dtype={base_img.dtype}")
        
        if cls_enabled and cls_result is not None and isinstance(cls_result, np.ndarray) and cls_result.size > 0:
            logging.info(f"get_modified_image: cls_result shape={cls_result.shape}, dtype={cls_result.dtype}")
            logging.info(f"get_modified_image: cls_result min={np.min(cls_result)}, max={np.max(cls_result)}")
            # Append classification result as a new band (like boolean math does)
            try:
                # Ensure base image is 3D
                if base_img.ndim == 2:
                    base_3d = base_img[:, :, np.newaxis]
                else:
                    base_3d = base_img
                
                # Ensure classification result matches base dimensions
                cls_h, cls_w = cls_result.shape[:2]
                base_h, base_w = base_3d.shape[:2]
                
                logging.info(f"get_modified_image: base dims=({base_h},{base_w}), cls dims=({cls_h},{cls_w})")
                
                if cls_h == base_h and cls_w == base_w:
                    # Ensure cls_result is 2D for stacking
                    cls_2d = np.squeeze(cls_result)
                    if cls_2d.ndim > 2:
                        cls_2d = cls_2d[..., 0]
                    
                    # Return only the classification result (as a single band)
                    # This ensures consistency with apply_aux_modifications and refresh behavior
                    # The viewer will handle this as a 1-band image with its own auto-stretch
                    logging.info("get_modified_image: returning classification result only (no append)")
                    return cls_2d.astype(base_3d.dtype, copy=False)
                else:
                    # Dimension mismatch - just return classification result alone
                    # (as 3D so viewer can handle it)
                    logging.warning("get_modified_image: dimension mismatch, returning cls_result only")
                    return cls_result[:, :, np.newaxis] if cls_result.ndim == 2 else cls_result
            except Exception as e:
                logging.warning(f"get_modified_image: failed to append classification: {e}")
                # Fall back to just classification result
                return cls_result[:, :, np.newaxis] if cls_result.ndim == 2 else cls_result
        
        logging.info("get_modified_image: returning base_img (no classification)")
        return base_img

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
        # Update append button state after band expression changes
        self._update_append_button_state()
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
        - NoData pixels (NaN or matching nodata_values) are colored magenta
        """
        if image is None:
            logging.warning("No image provided for normalization.")
            return None
        try:
            if image.dtype == np.uint8:
                # Already display-ready
                norm_img = image
            else:
                res = image.astype(np.float32, copy=True)  # Use copy to preserve original
                
                # Use stored NoData mask from apply_all_modifications_to_image for consistency
                # This ensures viewer shows same mask as process_polygon
                nd_mask = getattr(self, '_display_nodata_mask', None)
                
                # Also include NaN/Inf in mask (in case mask wasn't stored or is stale)
                nan_inf_mask = np.isnan(res) | np.isinf(res)
                if res.ndim == 3:
                    nan_inf_mask = nan_inf_mask.any(axis=2)
                
                if nd_mask is None:
                    nd_mask = nan_inf_mask
                else:
                    # Ensure shape matches (in case image was resized after mask was built)
                    if nd_mask.shape[:2] == res.shape[:2]:
                        nd_mask = nd_mask | nan_inf_mask
                    else:
                        nd_mask = nan_inf_mask
                        logging.debug(f"[normalize_image_for_display] Stored mask shape {nd_mask.shape} != image shape {res.shape[:2]}, using NaN/Inf only")
                
                # Replace NaN/Inf for processing
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
                    
                    # Only convert grayscale to RGB if we have NoData pixels to color
                    has_nodata = nd_mask is not None and nd_mask.any()
                    if has_nodata:
                        norm_img = np.stack([norm_img, norm_img, norm_img], axis=2)

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
                
                # Color NoData pixels magenta (255, 0, 255) in BGR order for visibility
                # Only apply if we have a 3-channel image (either naturally or converted from grayscale)
                if nd_mask is not None and nd_mask.any() and norm_img.ndim == 3 and norm_img.shape[2] >= 3:
                    norm_img[nd_mask, 0] = 255  # Blue channel
                    norm_img[nd_mask, 1] = 0    # Green channel
                    norm_img[nd_mask, 2] = 255  # Red channel
                    logging.debug(f"[normalize_image_for_display] Colored {nd_mask.sum()} NoData pixels magenta")

            logging.debug(f"Normalized Image: Shape {norm_img.shape}, Dtype {norm_img.dtype}")
            return norm_img

        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            return None


    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_initial_render_done", False):
            self._initial_render_done = True
            # Show progress dialog during initial render for large images
            from PyQt5 import QtWidgets, QtCore
            progress = QtWidgets.QProgressDialog("Loading image editor...", None, 0, 10, self)
            progress.setWindowTitle("Processing Image")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(100)  # Only show if takes > 100ms
            progress.setValue(0)
            QtWidgets.QApplication.processEvents()
            try:
                # Check if classification should auto-run on open
                cblock = (self.modifications or {}).get("classification") or {}
                cls_enabled = bool(cblock.get("enabled", False))
                
                # Set flag to prevent timer scheduling during initial render
                # (we handle classification directly below)
                self._skip_cls_timer = cls_enabled
                
                # Fast-open path: if the dialog was opened from an already-rendered ProjectTab viewer
                # AND an .ax exists, reuse the viewer pixels for the first display so the dialog appears instantly.
                fast_preview_used = False
                preview = getattr(self, "_fast_open_preview", None)
                if getattr(self, "_ax_loaded_on_open", False) and preview is not None:
                    try:
                        import numpy as np, logging
                        if isinstance(preview, np.ndarray) and getattr(preview, "size", 0) > 0:
                            # This is what the user is already seeing in the ProjectTab viewer,
                            # so it is safe to show immediately.
                            self.original_image = preview
                            self.display_image_data = preview

                            # Render using the same stretch pipeline as the main viewer when possible
                            pm = None
                            try:
                                parent = self.parent()
                                v = getattr(self, "viewer", None) or getattr(self, "bound_viewer", None)
                                if parent is not None and hasattr(parent, "_render_with_viewer_stretch") and v is not None:
                                    pm = parent._render_with_viewer_stretch(
                                        preview, v,
                                        prefer_last_band=bool(getattr(v, "preview_prefers_index", False))
                                    )
                            except Exception:
                                pm = None

                            progress.setLabelText("Rendering preview...")
                            progress.setValue(8)
                            QtWidgets.QApplication.processEvents()

                            if pm is not None:
                                self.display_image(pm)
                            else:
                                self.display_image(preview)

                            try:
                                self._update_ax_json_display()
                            except Exception:
                                pass

                            fast_preview_used = True
                    except Exception as e:
                        logging.debug(f"Fast preview on open failed, falling back to full apply: {e}")

                # If we couldn't use a fast preview, apply modifications and show the base image
                if not fast_preview_used:
                    self.reapply_modifications(progress_dialog=progress)

                # If classification is enabled, run it now (not via timer)
                if cls_enabled and hasattr(self, "use_sklearn_checkbox") and self.use_sklearn_checkbox.isChecked():
                    progress.setLabelText("Running classification...")
                    progress.setValue(5)
                    QtWidgets.QApplication.processEvents()
                    try:
                        # Check if model is available before running
                        bundle = self._resolve_sklearn_model_bundle()
                        if bundle and "model" in bundle:
                            self.run_sklearn_classification()
                            self._persist_classification_enabled(True)
                    except Exception as e:
                        import logging
                        logging.debug(f"Auto-classification on dialog open failed: {e}")
            finally:
                self._skip_cls_timer = False  # Reset flag
                try:
                    progress.close()
                except Exception:
                    pass

    def _pixmap_with_ax_stretch(self, cv_img):
        """
        Build a QPixmap from a cv_img (numpy) using the *current* stretch/display
        selected by the user (viewer/root/project), as recorded in .ax or parent.
        - Reuses parent.ProjectTab's _render_with_viewer_stretch logic if present.
        - Otherwise, applies a local minimal reimplementation using percentiles/σ/abs.
        """
        from PyQt5 import QtGui
        import numpy as np

        parent = self.parent()
        # Prefer using the canonical viewer renderer when available
        if parent is not None and hasattr(parent, "_render_with_viewer_stretch"):
            # Fake a thin "viewer-like" object to pass current band choices if needed
            class _Shim:  # minimal attributes the renderer might read
                pass
            viewer_like = _Shim()
            # Try to mirror any display knobs recorded in your .ax stretch block
            ax_path = self._ax_path_for(self.image_filepath)
            try:
                import json, os
                if os.path.exists(ax_path):
                    with open(ax_path, "r", encoding="utf-8") as f:
                        ax = json.load(f) or {}
                else:
                    ax = {}
            except Exception:
                ax = {}

            # Optional: pass band picks (display_mode, display_band, r/g/b) if renderer expects them
            st = ax.get("stretch") or {}
            viewer_like.display_mode = st.get("display_mode", "auto")
            viewer_like.display_band = st.get("display_band", None)
            viewer_like.r_band = st.get("r_band", None)
            viewer_like.g_band = st.get("g_band", None)
            viewer_like.b_band = st.get("b_band", None)

            # Pass the display NoData mask so unsaved changes are reflected
            mask_to_pass = getattr(self, "_display_nodata_mask", None)
            pm = parent._render_with_viewer_stretch(cv_img, viewer_like, mask=mask_to_pass)
            return pm

        # ---- Fallback (no parent renderer): local minimal viewer-like stretch ----
        img = cv_img
        if img.ndim == 2:
            img = img[..., None]

        # Try reading the last saved stretch from .ax
        try:
            import json, os
            ax = {}
            ax_path = self._ax_path_for(self.image_filepath)
            if os.path.exists(ax_path):
                with open(ax_path, "r", encoding="utf-8") as f:
                    ax = json.load(f) or {}
            st = ax.get("stretch") or {}
        except Exception:
            st = {}

        mode = (st.get("mode") or "percentile").lower()
        per_ch = bool(st.get("per_channel", True))
        clip = bool(st.get("clip", True))
        disp_mode = (st.get("display_mode") or "auto").lower()
        disp_band = st.get("display_band", None)
        r_sel, g_sel, b_sel = st.get("r_band"), st.get("g_band"), st.get("b_band")

        import numpy as np
        x = img.astype(np.float32, copy=False)
        H, W, C = x.shape

        def _percentile_bounds(a, lo, hi):
            if a.size == 0:
                return 0.0, 1.0
            return (np.nanpercentile(a, lo), np.nanpercentile(a, hi))

        def _stretch_to_u8(y, lo, hi):
            # normalise and clip to [0,255]
            denom = (hi - lo) if (hi > lo) else 1.0
            z = (y - lo) / denom
            if clip:
                z = np.clip(z, 0.0, 1.0)
            return (z * 255.0).astype(np.uint8, copy=False)

        # Compute min/max per policy
        if mode == "absolute":
            lo = st.get("min_val", None)
            hi = st.get("max_val", None)
            if lo is None or hi is None:
                # fallback to robust percentiles
                lo, hi = _percentile_bounds(x, 0.5, 99.5)
            if per_ch and C > 1:
                lo = np.asarray([lo] * C, np.float32) if np.isscalar(lo) else np.asarray(lo, np.float32)
                hi = np.asarray([hi] * C, np.float32) if np.isscalar(hi) else np.asarray(hi, np.float32)
            y = x.copy()
            if per_ch and C > 1:
                for c in range(C):
                    y[..., c] = _stretch_to_u8(x[..., c], float(lo[c]), float(hi[c]))
            else:
                y = _stretch_to_u8(x, float(lo), float(hi))
        elif mode == "stddev":
            k = float(st.get("k_sigma", 1.0))
            if per_ch and C > 1:
                mu = np.nanmean(x.reshape(-1, C), axis=0)
                sd = np.nanstd(x.reshape(-1, C), axis=0)
                lo = mu - k * sd; hi = mu + k * sd
                y = x.copy()
                for c in range(C):
                    y[..., c] = _stretch_to_u8(x[..., c], float(lo[c]), float(hi[c]))
            else:
                mu = float(np.nanmean(x))
                sd = float(np.nanstd(x))
                y = _stretch_to_u8(x, mu - k * sd, mu + k * sd)
        else:  # "percentile"
            lp = float(st.get("low_p", 0.5)); hp = float(st.get("high_p", 99.5))
            if per_ch and C > 1:
                y = x.copy()
                for c in range(C):
                    lo, hi = _percentile_bounds(x[..., c], lp, hp)
                    y[..., c] = _stretch_to_u8(x[..., c], lo, hi)
            else:
                lo, hi = _percentile_bounds(x, lp, hp)
                y = _stretch_to_u8(x, lo, hi)

        # choose bands for display
        if disp_mode == "single" and disp_band is not None and 0 <= int(disp_band) < C:
            disp = y[..., int(disp_band)]
            disp = disp.reshape(H, W)  # grayscale
            chs = 1
        elif disp_mode == "rgb" and C >= 3 and all(v is not None for v in (r_sel, g_sel, b_sel)):
            r = int(r_sel); g = int(g_sel); b = int(b_sel)
            r = np.clip(r, 0, C - 1); g = np.clip(g, 0, C - 1); b = np.clip(b, 0, C - 1)
            disp = np.stack([y[..., b], y[..., g], y[..., r]], axis=-1)  # BGR for QImage fast path
            chs = 3
        else:
            # auto: grayscale if C==1, else first 3 channels BGR
            if C == 1:
                disp = y[..., 0]; disp = disp.reshape(H, W); chs = 1
            else:
                dC = min(3, C)
                # QImage Format_BGR888 expects BGR order
                if dC == 3:
                    disp = np.stack([y[..., 0], y[..., 1], y[..., 2]], axis=-1)
                else:
                    # C==2 -> pad a third channel
                    pad = np.zeros((H, W, 3), dtype=np.uint8)
                    pad[..., :dC] = y[..., :dC]
                    disp = pad
                chs = 3

        # Build QPixmap (same as your viewer path)
        return _pixmap_from_uint8(disp)

    def reapply_modifications(self, progress_dialog=None):
        """
        Re-run the editor pipeline and display.
        If classification preview is enabled:
          • Show the cached class map ONLY when its geometry snapshot matches
            the current edited image geometry.
          • Otherwise, invalidate the cache and (optionally) auto re-run the classifier.
        
        progress_dialog: optional QProgressDialog to show progress during processing.
        """
        import logging, re, copy
        import numpy as np
        from PyQt5 import QtCore, QtWidgets

        def _update_progress(label, value, maximum=10):
            if progress_dialog:
                try:
                    progress_dialog.setMaximum(maximum)
                    progress_dialog.setValue(value)
                    progress_dialog.setLabelText(label)
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass

        _update_progress("Syncing UI settings...", 0)

        # ==================== SYNC ALL ENABLED FLAGS FROM UI ====================
        # NoData
        if hasattr(self, "nodata_input"):
            nodata_text = self.nodata_input.text().strip()
            self.modifications["nodata_values"] = self._parse_nodata_values(nodata_text)
        if hasattr(self, "nodata_enabled_checkbox"):
            self.modifications["nodata_enabled"] = self.nodata_enabled_checkbox.isChecked()
        
        # Mask Polygon (names-based, synced from menu checkboxes)
        if hasattr(self, "mask_polygon_enabled_checkbox"):
            enabled = self.mask_polygon_enabled_checkbox.isChecked()
            # Get selected names from the menu
            names = []
            if hasattr(self, "mask_polygon_menu"):
                for action in self.mask_polygon_menu.actions():
                    if action.isCheckable() and action.isChecked():
                        # Extract name from action text (remove " (other images)" suffix)
                        action_name = action.text().replace(" (other images)", "")
                        names.append(action_name)
            self.modifications["mask_polygon"] = {
                'enabled': enabled and len(names) > 0,
                'names': names
            }
        
        # Resize
        if hasattr(self, "resize_enabled_checkbox"):
            self.modifications["resize_enabled"] = self.resize_enabled_checkbox.isChecked()
        
        # Rotate
        if hasattr(self, "rotate_enabled_checkbox"):
            self.modifications["rotate_enabled"] = self.rotate_enabled_checkbox.isChecked()
        
        # Crop
        if hasattr(self, "crop_enabled_checkbox"):
            self.modifications["crop_enabled"] = self.crop_enabled_checkbox.isChecked()
        
        # Band expression
        if hasattr(self, "band_input"):
            self.modifications["band_expression"] = self.band_input.text().strip()
        if hasattr(self, "band_enabled_checkbox"):
            self.modifications["band_enabled"] = self.band_enabled_checkbox.isChecked()
        
        # Histogram
        if hasattr(self, "hist_enabled_checkbox"):
            self.modifications["hist_enabled"] = self.hist_enabled_checkbox.isChecked()
            
        # Registration
        if hasattr(self, "reg_enabled_checkbox") and hasattr(self, "reg_mode_combo"):
            existing_reg = self.modifications.get("registration", {})
            self.modifications["registration"] = {
                "enabled": self.reg_enabled_checkbox.isChecked(),
                "mode": self.reg_mode_combo.currentText(),
                "matrix": existing_reg.get("matrix"),
                "is_reference": existing_reg.get("is_reference", False)
            }
        # ==================== END SYNC ====================

        parent = self.parent()
        if not hasattr(self, "_cls_snapshot"):
            self._cls_snapshot = None

        # 1) Run the editor pipeline (crop→rotate→hist→resize→band_expr)
        _update_progress("Processing image modifications...", 1)
        
        # Create progress callback for apply_all_modifications_to_image
        def _mod_progress(step_name, step_num, total_steps):
            # Map step progress (0-6) to overall progress (1-7 out of 10)
            overall_step = 1 + step_num
            _update_progress(step_name, overall_step)
        
        try:
            mod_image = self.apply_all_modifications_to_image(
                self.base_image, self.modifications, progress_callback=_mod_progress)
            self.original_image = mod_image
        except Exception as e:
            logging.warning(f"reapply_modifications: pipeline failed: {e}")
            mod_image = getattr(self, "original_image", None)

        if mod_image is None or getattr(mod_image, "size", 0) == 0:
            logging.warning("reapply_modifications: no image to display.")
            return

        Ht, Wt = mod_image.shape[:2]
        
        # FIX: Update size textboxes to reflect current modified image dimensions
        # This ensures crop/rotate/resize operations update the displayed size
        if hasattr(self, "resize_width_input"):
            self.resize_width_input.setText(str(Wt))
        if hasattr(self, "resize_height_input"):
            self.resize_height_input.setText(str(Ht))
        
        geom_now = {
            "shape": (Ht, Wt),
            "rotate": int(self.modifications.get("rotate", 0)) % 360,
            "crop_rect": copy.deepcopy(self.modifications.get("crop_rect")),
            "crop_rect_ref_size": copy.deepcopy(self.modifications.get("crop_rect_ref_size")),
            "resize": copy.deepcopy(self.modifications.get("resize")),
        }

        # 2) Classification preview (no resizing fallback!)
        cblock = (self.modifications or {}).get("classification") or {}
        ui_wants_cls = bool(getattr(self, "use_sklearn_checkbox", None)
                            and self.use_sklearn_checkbox.isChecked())
        cls_enabled = bool(cblock.get("enabled", False)) and ui_wants_cls
        cls = getattr(self, "last_band_float_result", None)
        cls_pending = False  # Track if classification was just triggered

        def _geom_matches(a, b):
            if not a or not b:
                return False
            # Compare all geometry-affecting fields including output shape
            return (
                a.get("shape") == b.get("shape") and
                a.get("rotate") == b.get("rotate") and
                a.get("resize") == b.get("resize") and
                a.get("crop_rect") == b.get("crop_rect") and
                a.get("crop_rect_ref_size") == b.get("crop_rect_ref_size")
            )

        if cls_enabled:
            # If we don't have a class map yet OR it's stale → invalidate & optionally re-run
            if (not isinstance(cls, np.ndarray)) or (not cls.size) or (not _geom_matches(self._cls_snapshot, geom_now)):
                # Invalidate stale cache
                self.last_band_float_result = None
                self._classification_result = None
                self._cls_snapshot = None
                # Show normal image while we (optionally) kick off a reclass
                # (Avoid blocking UI here.)
                cls_pending = True  # Mark that classification is pending
                # Only schedule timer if NOT during initial render (showEvent handles that case)
                if not getattr(self, "_skip_cls_timer", False):
                    QtCore.QTimer.singleShot(0, self.on_run_classification_clicked)
            else:
                # Geometry matches → show the class map AS-IS (no resizing).
                try:
                    vis = np.squeeze(cls).astype(np.float32, copy=False)
                except Exception:
                    vis = cls
                self.display_image_data = vis
                pm = None
                try:
                    pm = self._pixmap_with_ax_stretch(vis)   # will honor .ax "stretch" (percentile/stddev/absolute etc.)
                except Exception:
                    pm = None

                if pm is not None:
                    _update_progress("Displaying classification...", 9)
                    self.display_image(pm)                   # QPixmap fast path (matches main viewer)
                    _update_progress("Complete", 10)
                else:
                    # safe fallback: percentile → 8-bit for display
                    _update_progress("Displaying classification...", 9)
                    self.display_image(self.normalize_image_for_display(vis))
                    _update_progress("Complete", 10)
                return


        # 3) Decide whether to prefer the LAST band in the preview
        prefer_last = False
        # Only prefer last band if classification actually completed, NOT if it's pending
        if cls_enabled and not cls_pending:
            # If classification is enabled and completed, prefer last band for index-like previews
            prefer_last = True
        elif not cls_enabled:
            be = (self.modifications or {}).get("band_expression")
            if isinstance(be, dict):
                mode = str(be.get("mode") or be.get("output") or "append").lower()
                if mode != "replace":
                    prefer_last = True
                expr = str(be.get("expr") or be.get("expression") or "")
                if expr and re.search(r"(==|!=|<=|>=|<|>|&|\||~)", expr):
                    prefer_last = True
            elif isinstance(be, str):
                expr = be.strip()
                if expr:
                    prefer_last = True
                    if re.search(r"(==|!=|<=|>=|<|>|&|\||~)", expr):
                        prefer_last = True

        # --- FORCE GRAYSCALE BEFORE CALLING ANY RENDERER ---
        # If the edited image is single-band (2-D), show it as grayscale.
        if isinstance(mod_image, np.ndarray) and mod_image.ndim == 2:
            self.display_image_data = mod_image
            _update_progress("Displaying image...", 9)
            self.display_image(mod_image)
            _update_progress("Complete", 10)
            return

        # If we prefer the last band (e.g., boolean/float band expression),
        # show that band explicitly as grayscale to avoid RGB expansion.
        if prefer_last and isinstance(mod_image, np.ndarray) and mod_image.ndim == 3:
            last = np.squeeze(mod_image[..., -1])
            if last.ndim == 3 and last.shape[2] == 1:
                last = last.reshape(last.shape[0], last.shape[1])
            self.display_image_data = last
            _update_progress("Displaying image...", 9)
            self.display_image(last)
            _update_progress("Complete", 10)
            return

        # Hint the main viewer that index/last-band view is wanted
        try:
            v = getattr(self, "bound_viewer", None) or getattr(self, "viewer", None)
            if v is not None:
                setattr(v, "preview_prefers_index", bool(prefer_last))
        except Exception:
            pass

        # Persist what we are displaying so resizes/repaints stay consistent
        self.display_image_data = mod_image

        # 4) Prefer the project's renderer if available (so preview matches the main viewer)
        _update_progress("Rendering image...", 8)
        try:
            v = getattr(self, "bound_viewer", None) or getattr(self, "viewer", None)
            if parent is not None and hasattr(parent, "_render_with_viewer_stretch") and v is not None:
                # Pass local display mask so preview reflects unsaved NoData changes
                mask_to_pass = getattr(self, "_display_nodata_mask", None)
                logging.info(f"[reapply_modifications] Passing mask to renderer: {mask_to_pass.sum() if mask_to_pass is not None and hasattr(mask_to_pass, 'sum') else 'None'} masked pixels")
                pm = parent._render_with_viewer_stretch(mod_image, v, prefer_last_band=prefer_last, mask=mask_to_pass)
                if pm is not None:
                    _update_progress("Displaying image...", 9)
                    self.display_image(pm)   # QPixmap fast path
                    _update_progress("Complete", 10)
                    return
        except Exception as e:
            logging.debug(f"reapply_modifications: parent renderer failed, falling back: {e}")

        # 5) Fallback: try dialog’s own stretch helper
        try:
            pm = self._pixmap_with_ax_stretch(mod_image)
            if pm is not None:
                _update_progress("Displaying image...", 9)
                self.display_image(pm)
                _update_progress("Complete", 10)
                return
        except Exception as e:
            logging.debug(f"_pixmap_with_ax_stretch failed: {e}")

        # 6) Final safety: show whatever we have
        _update_progress("Displaying image...", 9)
        self.display_image(mod_image)
        _update_progress("Complete", 10)
        
        # 7) Update JSON text box if visible
        self._update_ax_json_display()


    def _clear_classification_preview(self, persist: bool = False):
        """
        Forget any classification artifacts immediately:
          • drop cached class map and geometry snapshot
          • remove 'classification' from pending modifications
          • optionally delete it from the .ax on disk
        """
        try:
            self.last_band_float_result = None
            self._classification_result = None
        except Exception:
            pass
        try:
            self._cls_snapshot = None
        except Exception:
            pass

        # Strip from pending edits
        try:
            if isinstance(self.modifications, dict) and "classification" in self.modifications:
                self.modifications.pop("classification", None)
        except Exception:
            pass

        # Optionally write a deletion to the .ax now (so it's gone even before "Apply All")
        if persist and getattr(self, "image_filepath", None):
            try:
                self._write_ax(self.image_filepath, {"_delete": ["classification"]})
            except Exception:
                pass

    def _on_use_sklearn_toggled(self, checked: bool):
        self.classify_btn.setEnabled(bool(checked))
        # Update append button state
        self._update_append_button_state()

        # Persist UI intent into pending modifications (so "Apply All" writes it)
        self._persist_classification_enabled(bool(checked))

        if not checked:
            # HARD RESET of any classification preview/residue
            self._clear_classification_preview(persist=True)

            # Ensure we redraw the "normal" image (no class preview)
            # By clearing display_image_data we force a fresh pipeline render.
            try:
                self.display_image_data = None
            except Exception:
                pass

            self.reapply_modifications()
        else:
            # If turning ON, we just enable the button; user runs "Classify" explicitly.
            # (No implicit auto-run here.)
            pass

    def _update_append_button_state(self):
        """Enable/disable the append button based on what can be appended."""
        can_append = False
        try:
            mods = self.modifications or {}

            # Classification enabled?
            cblock = mods.get("classification") or {}
            if isinstance(cblock, dict) and bool(cblock.get("enabled", False)):
                can_append = True

            # Any band expression present (boolean OR numeric)?
            # FIX: Also check band_enabled flag (similar to classification.enabled)
            be = mods.get("band_expression")
            band_enabled = mods.get("band_enabled", True)  # Default True for backwards compat
            expr = ""
            if be:
                if isinstance(be, dict):
                    expr = str(be.get("expression") or "").strip()
                elif isinstance(be, str):
                    expr = be.strip()

            if expr and band_enabled:
                can_append = True

        except Exception:
            pass

        if hasattr(self, "append_band_btn"):
            self.append_band_btn.setEnabled(can_append)

    def _on_append_band_clicked(self):

        """
        Append the current boolean expression or classification as a permanent extra band.
        The band is stored in the .ax file and becomes available for all processing functions.
        After appending, the classification/expression is cleared and the viewer returns to
        showing the original image. User can then run another classification or enter another
        expression to append more bands.
        """
        import logging
        import numpy as np
        from PyQt5 import QtWidgets

        mods = dict(self.modifications or {})

        # Determine what to append (classification takes priority if both are active)
        append_type = None
        append_expr = None

        # Check for classification first (priority over boolean expression)
        cblock = mods.get("classification") or {}
        if isinstance(cblock, dict) and bool(cblock.get("enabled", False)):
            # Classification result is stored in _classification_result (separate from band expressions)
            cls_result = getattr(self, "_classification_result", None)
            has_cls_result = (
                cls_result is not None
                and isinstance(cls_result, np.ndarray)
                and cls_result.size > 0
            )
            logging.info(f"Append: classification enabled={True}, has_cls_result={has_cls_result}, result shape={cls_result.shape if has_cls_result else 'N/A'}")
            if has_cls_result:
                append_type = "classification"
                logging.info("Append: Found classification result to append")

        # Check for band expression (if no classification)
        if append_type is None:
            be = mods.get("band_expression")
            expr = ""
            if be:
                if isinstance(be, dict):
                    expr = str(be.get("expression") or "").strip()
                elif isinstance(be, str):
                    expr = be.strip()

            if expr:
                import re
                is_bool = bool(re.search(r"(==|!=|<=|>=|<|>|&|\||~|AND|OR|NOT)", expr, re.IGNORECASE))
                append_type = "boolean_expression" if is_bool else "band_expression"
                append_expr = expr
                logging.info(f"Append: Found {append_type} to append: {expr}")

        if append_type is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to Append",
                "No boolean expression or classification to append.\n"
                "First run a classification or enter a boolean band expression."
            )
            return

        # Get existing appended bands
        appended_bands = list(mods.get("appended_bands") or [])

        # Create new band entry
        new_band = {
            "type": append_type,
            "index": len(appended_bands) + 1,
        }

        if append_type in ("boolean_expression", "band_expression"):
            new_band["expression"] = append_expr
            logging.info(f"Appending {append_type} band #{new_band['index']}: {append_expr}")
        elif append_type == "classification":
            # Store classification model reference and label names
            new_band["model_name"] = cblock.get("model_name", "sklearn_model")
            new_band["label_names"] = cblock.get("label_names", [])
            logging.info(f"Appending classification band #{new_band['index']} from model: {new_band['model_name']}")

        # Add to appended bands
        appended_bands.append(new_band)
        mods["appended_bands"] = appended_bands

        # Clear the source (expression or classification) from current preview
        if append_type in ("boolean_expression", "band_expression"):
            # Clear band expression
            mods["band_expression"] = None
            # Clear the band expression text field
            if hasattr(self, "band_input"):
                self.band_input.blockSignals(True)
                self.band_input.clear()
                self.band_input.blockSignals(False)
            # Uncheck the band enabled checkbox
            if hasattr(self, "band_enabled_checkbox"):
                self.band_enabled_checkbox.blockSignals(True)
                self.band_enabled_checkbox.setChecked(False)
                self.band_enabled_checkbox.blockSignals(False)

        elif append_type == "classification":
            # Clear classification from active modifications
            mods["classification"] = None
            # Uncheck the checkbox
            if hasattr(self, "use_sklearn_checkbox"):
                self.use_sklearn_checkbox.blockSignals(True)
                self.use_sklearn_checkbox.setChecked(False)
                self.use_sklearn_checkbox.blockSignals(False)
            if hasattr(self, "classify_btn"):
                self.classify_btn.setEnabled(False)

        # Clear the computed result so viewer shows original image
        self.last_band_float_result = None
        self._classification_result = None
        self._cls_snapshot = None

        # Update modifications
        self.modifications = mods

        # Save to .ax file
        try:
            self.save_modifications_to_file()
        except Exception as e:
            logging.warning(f"Failed to save appended band to .ax: {e}")

        # Update append button state (will be disabled since we cleared the source)
        self._update_append_button_state()

        # Refresh the preview to show original image (without classification/expression overlay)
        try:
            self.display_image_data = None
            self.reapply_modifications()
        except Exception:
            pass

        # Notify user
        band_count = len(appended_bands)
        QtWidgets.QMessageBox.information(
            self, "Band Appended",
            f"The {append_type.replace('_', ' ')} has been appended as band #{band_count}.\n"
            f"Total appended bands: {band_count}\n\n"
            "This band will be available for:\n"
            "  • ML training\n"
            "  • CSV export\n"
            "  • Statistics\n"
            "  • Image export\n\n"
            "You can run another classification or expression to add more bands."
        )


    def on_run_classification_clicked(self):
        if not self.use_sklearn_checkbox.isChecked():
            from PyQt5 import QtWidgets
            QtWidgets.QMessageBox.information(self, "Classification",
                                              "Enable 'Use scikit-learn classification' first.")
            return
        try:
            self.run_sklearn_classification()
            # NEW: keep the flag persisted once classification has run
            self._persist_classification_enabled(True)
            # Update append button state after classification runs
            self._update_append_button_state()
        except Exception as e:
            import logging
            logging.exception("Classification failed")
            from PyQt5 import QtWidgets
            QtWidgets.QMessageBox.critical(self, "Classification Error", str(e))


# ---------- sklearn classification: core (FAST, multi-core) ----------
    def run_sklearn_classification(self):
        """
        Ultra-fast per-pixel classification:
          • Uses model's internal n_jobs when available.
          • Else threads over big chunks, with threadpoolctl limiting BLAS.
          • Maps labels→indices per-chunk (no giant object array).
        Records a geometry snapshot so future edits can detect staleness.
        """
        import os, time, copy
        import numpy as np
        from PyQt5 import QtWidgets, QtCore
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # optional: avoid nested BLAS threads when we parallelize outside
        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            threadpool_limits = None

        if not hasattr(self, "_cls_snapshot"):
            self._cls_snapshot = None

        # Make sure we're classifying the currently edited pixels WITH modifications applied
        # This ensures histogram normalization is applied before classification
        if self.base_image is None or getattr(self.base_image, "size", 0) == 0:
            QtWidgets.QMessageBox.warning(self, "No Image", "Open an image first.")
            return
        
        # Apply current modifications (including histogram normalization) to get the correct image
        img = self.apply_all_modifications_to_image(self.base_image, self.modifications)
        if img is None or getattr(img, "size", 0) == 0:
            QtWidgets.QMessageBox.warning(self, "No Image", "Could not process image with current modifications.")
            return

        bundle = self._resolve_sklearn_model_bundle()
        if not bundle or "model" not in bundle:
            raise RuntimeError("No valid scikit-learn model bundle was provided/selected.")
        model = bundle["model"]
        feat_names  = list(bundle.get("feature_names") or [])
        label_names = list(bundle.get("label_names") or getattr(getattr(model, "classes_", []), "tolist", lambda: [])())
        expressions = bundle.get("expressions", [])  # Get custom expressions from bundle
        window_size = bundle.get("window_size", 1)  # Get spatial window size (default 1)
        base_feature_names = bundle.get("base_feature_names", feat_names)  # Base names without window suffix
        if not feat_names:
            raise RuntimeError("Model bundle missing 'feature_names'.")

        # Log model info
        import logging
        logging.info(f"[image_editor] Classification using {len(feat_names)} features, window={window_size}×{window_size}")
        if expressions:
            logging.info(f"[image_editor] Model has {len(expressions)} custom expressions: {[n for n,_ in expressions]}")

        # Build features (H*W, F) - pass expressions and window info for custom band math
        X, (H, W) = self._make_feature_stack_for_model(
            img, feat_names, 
            expressions=expressions,
            window_size=window_size,
            base_feature_names=base_feature_names
        )
        if X is None:
            raise RuntimeError("Could not build feature stack for the model. "
                             "Check that image has required bands and all expressions are valid.")
        total = H * W
        if total == 0:
            QtWidgets.QMessageBox.warning(self, "Empty Image", "Image has zero pixels after processing.")
            return
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Chunk size tuned for throughput while keeping progress smooth
        cores = max(1, (os.cpu_count() or 1))
        chunk = max(400_000, min(2_000_000, total // max(cores, 4) or 400_000))

        progress = QtWidgets.QProgressDialog("Classifying image...", "Cancel", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        # ---- fast label→index mapping (vectorized, per-chunk) ----
        classes_arr = np.asarray(getattr(model, "classes_", []))
        def map_to_idx(y):
            y_arr = np.asarray(y)
            if classes_arr.size:
                ca = classes_arr
                # dtype harmonization for searchsorted
                if y_arr.dtype != ca.dtype:
                    try:
                        y_arr = y_arr.astype(ca.dtype, copy=False)
                    except Exception:
                        y_arr = y_arr.astype(object)
                        ca = ca.astype(object)
                return np.searchsorted(ca, y_arr).astype(np.int32, copy=False)
            # fallback: stable mapping on the fly
            uniq = np.unique(y_arr.astype(object))
            return np.searchsorted(uniq, y_arr.astype(object)).astype(np.int32, copy=False)

        pred_flat = np.empty(total, dtype=np.int32)

        # Throttle progress updates (avoid UI churn)
        last_tick = 0.0
        def tick(v):
            nonlocal last_tick
            now = time.monotonic()
            if (now - last_tick) >= 0.05 or v == total:
                progress.setValue(v)
                QtWidgets.QApplication.processEvents()
                last_tick = now

        # Prefer model's internal parallelism if it supports n_jobs
        use_internal = False
        try:
            params = getattr(model, "get_params", lambda **k: {})()
            if "n_jobs" in params:
                try:
                    model.set_params(n_jobs=-1)
                except Exception:
                    pass
                use_internal = True
        except Exception:
            pass

        if use_internal:
            i = 0
            while i < total:
                if progress.wasCanceled():
                    progress.close(); return
                j = min(i + chunk, total)
                y = model.predict(X[i:j])
                pred_flat[i:j] = map_to_idx(y)
                i = j
                tick(i)
        else:
            # our own threading; prevent nested BLAS threads if possible
            if threadpool_limits:
                def predict_fn(xslice):
                    with threadpool_limits(limits=1):
                        return model.predict(xslice)
            else:
                def predict_fn(xslice):
                    return model.predict(xslice)

            ranges = []
            i = 0
            while i < total:
                j = min(i + chunk, total)
                ranges.append((i, j))
                i = j

            max_workers = max(1, cores)
            ok = True
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                it = iter(ranges)
                futures = {}

                # warm queue
                for _ in range(min(max_workers * 2, len(ranges))):
                    a, b = next(it, (None, None))
                    if a is None: break
                    futures[ex.submit(predict_fn, X[a:b])] = (a, b)

                from concurrent.futures import as_completed
                while futures:
                    if progress.wasCanceled():
                        ok = False; break
                    fut = next(as_completed(futures))
                    a, b = futures.pop(fut)
                    try:
                        y = fut.result()
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "Classification Error", str(e))
                        ok = False
                        break
                    pred_flat[a:b] = map_to_idx(y)
                    tick(b)

                    # keep queue full
                    a2, b2 = next(it, (None, None))
                    if a2 is not None:
                        futures[ex.submit(predict_fn, X[a2:b2])] = (a2, b2)

            progress.close()
            if not ok:
                return

        progress.setValue(total); progress.close()

        # reshape to HxW float32 for grayscale last-band preview
        pred_idx = pred_flat.reshape(H, W).astype(np.float32, copy=False)

        # Store result in separate classification attribute (not shared with band expressions)
        self._classification_result = pred_idx.copy()
        # Also keep in last_band_float_result for display compatibility
        self.last_band_float_result = pred_idx.copy()
        self.modifications["classification"] = {
            "mode": "sklearn",
            "enabled": True,
            "label_names": label_names
        }

        # Record the geometry snapshot that produced this class map
        self._cls_snapshot = {
            "shape": (H, W),
            "rotate": int(self.modifications.get("rotate", 0)) % 360,
            "crop_rect": copy.deepcopy(self.modifications.get("crop_rect")),
            "crop_rect_ref_size": copy.deepcopy(self.modifications.get("crop_rect_ref_size")),
            "resize": copy.deepcopy(self.modifications.get("resize")),
        }

        # Refresh view (will show class map without resizing)
        self.reapply_modifications()

        
        #self.reapply_modifications()
    # ---------- sklearn classification: model resolution ----------
    def _resolve_sklearn_model_bundle(self):
            """
            Try (in order):
              1) parent tab's 'random_forest_model' (or any sklearn bundle you stored there)
              2) parent tab's CLASS attribute 'shared_random_forest_model'
              3) ask user to pick a .pkl; cache to both places for future use.
            Returns the bundle dict or None.
            """
            import pickle, os
            from PyQt5 import QtWidgets, QtCore, QtGui

            parent = self.parent()
            # 1) per-tab instance attribute
            bundle = getattr(parent, "random_forest_model", None)
            if self._is_valid_bundle(bundle):
                # CRITICAL: Also sync to class attribute so @staticmethod apply_aux_modifications can find it
                try:
                    if getattr(type(parent), "shared_random_forest_model", None) is None:
                        setattr(type(parent), "shared_random_forest_model", bundle)
                        logging.info("[sklearn] _resolve_sklearn_model_bundle: synced instance model to class attribute")
                except Exception:
                    pass
                return bundle

            # 2) shared on the tab class (avoid importing ProjectTab to prevent circular import)
            shared = getattr(type(parent), "shared_random_forest_model", None) if parent else None
            if self._is_valid_bundle(shared):
                # also cache on instance for convenience
                try:
                    setattr(parent, "random_forest_model", shared)
                except Exception:
                    pass
                return shared

            # 3) prompt user
            options = QtWidgets.QFileDialog.Options()
            # FIX: Removed DontUseNativeDialog to prevent Windows freezing on drive scan
            # options |= QtWidgets.QFileDialog.DontUseNativeDialog 

            model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select scikit-learn model bundle", "",
                "Pickle Files (*.pkl);;All Files (*)", options=options)
            
            if not model_path:
                return None

            # FIX: Set WaitCursor so user knows app is busy loading the large pickle file
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtWidgets.QApplication.processEvents() # Force UI refresh before blocking I/O

            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
            except Exception as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{e}")
                return None
            finally:
                # Ensure cursor is restored even if successful
                QtWidgets.QApplication.restoreOverrideCursor()

            if not self._is_valid_bundle(bundle):
                QtWidgets.QMessageBox.critical(self, "Invalid Bundle",
                                               "Selected file is not a valid CanoPie model bundle.")
                return None

            # cache both (per-instance and shared)
            try:
                setattr(parent, "random_forest_model", bundle)
            except Exception:
                pass
            try:
                setattr(type(parent), "shared_random_forest_model", bundle)
            except Exception:
                pass

            return bundle
    def _is_valid_bundle(self, b):
        if not isinstance(b, dict):
            return False
        if "model" not in b:
            return False
        m = b["model"]
        return hasattr(m, "predict")

        # ---------- sklearn classification: features ----------
    def _make_feature_stack_for_model(self, img, feature_names, expressions=None, window_size=1, base_feature_names=None):
        """
        Build features in the SAME order used at training time and return:
            X_flat: (H*W, F) float32
            (H, W): original image height/width
        Order rules:
          • If img has exactly 3 channels (OpenCV BGR), remap to RGB → [R,G,B].
          • If img has >3 channels, keep natural order as b1..bC, where b1=R, b2=G, b3=B.
          • Extras map to band_4, band_5, …
          • If expressions are provided, compute them and add to features.
          • If window_size > 1, extract spatial neighborhoods around each pixel.
        
        Args:
            img: Input image (HxW or HxWxC)
            feature_names: List of feature names the model expects (may include window suffixes)
            expressions: Optional list of (name, expr_str) tuples for custom band expressions
            window_size: Spatial context window (1, 3, or 5). Default 1 = center pixel only.
            base_feature_names: Base feature names without window suffixes (needed for window > 1)
        """
        import numpy as np
        import re
        import logging

        if img is None:
            return None, (0, 0)

        a = np.asarray(img)
        if a.ndim == 2:
            chans = [a.astype(np.float32, copy=False)]
        elif a.ndim == 3:
            C = a.shape[2]
            if C == 3:
                # OpenCV is BGR → flip to RGB to match training
                chans = [a[:, :, 2], a[:, :, 1], a[:, :, 0]]  # R, G, B
            else:
                # Multispectral stack: keep natural order b1..bC
                chans = [a[:, :, i] for i in range(C)]
            chans = [np.ascontiguousarray(c.astype(np.float32, copy=False)) for c in chans]
        else:
            return None, (0, 0)

        H, W = chans[0].shape
        
        # Build base image for expression evaluation (RGB order)
        if len(chans) >= 3:
            base_rgb = np.dstack(chans)  # HxWxC in RGB order
        else:
            base_rgb = chans[0][..., None] if len(chans) == 1 else np.dstack(chans)

        # Pre-compute expression images if model has custom expressions
        expr_images = {}  # name -> HxW array
        if expressions:
            try:
                from .utils import eval_band_expression
            except ImportError:
                eval_band_expression = None
            
            def _normalize_expr(expr: str) -> str:
                s = re.sub(r'\bAND\b', '&', expr, flags=re.IGNORECASE)
                s = re.sub(r'\bOR\b',  '|', s,   flags=re.IGNORECASE)
                s = re.sub(r'\bNOT\b', '~', s,   flags=re.IGNORECASE)
                return s
            
            for expr_name, expr_str in expressions:
                try:
                    expr_norm = _normalize_expr(expr_str)
                    if eval_band_expression is not None:
                        result = eval_band_expression(base_rgb, expr_norm)
                    else:
                        # Fallback: try parent's method
                        parent = self.parent()
                        fn = getattr(parent, "_eval_band_expression", None)
                        if fn:
                            result = fn(base_rgb, expr_norm)
                        else:
                            logging.warning(f"[image_editor] No expression evaluator for '{expr_name}'")
                            result = None
                    
                    if result is not None:
                        if result.ndim == 3:
                            result = result[:, :, -1]  # Take last channel
                        expr_images[expr_name] = result.astype(np.float32, copy=False)
                        logging.debug(f"[image_editor] Computed expression '{expr_name}'")
                except Exception as e:
                    logging.warning(f"[image_editor] Failed to compute expression '{expr_name}': {e}")

        # Helper to get base feature value at a position
        def _get_base_feature_value(name, y, x):
            """Get a single feature value at position (y, x)."""
            if name == "red_channel":
                if len(chans) < 3:
                    return None
                return chans[0][y, x]
            elif name == "green_channel":
                if len(chans) < 3:
                    return None
                return chans[1][y, x]
            elif name == "blue_channel":
                if len(chans) < 3:
                    return None
                return chans[2][y, x]
            elif name.startswith("band_"):
                try:
                    k = int(name.split("_", 1)[1])
                    idx = k - 1
                except Exception:
                    return None
                if 0 <= idx < len(chans):
                    return chans[idx][y, x]
                return None
            elif name in expr_images:
                return expr_images[name][y, x]
            return None

        # Determine base feature names for window extraction
        # For window_size > 1: base_feature_names contains names without window suffixes
        # For window_size == 1: base_feature_names should be used if available (has correct model names)
        if base_feature_names:
            # Use provided base names (these have correct model names like "red_channel")
            base_names = list(base_feature_names)
        else:
            # Fallback: use feature_names (for old models without base_feature_names)
            base_names = list(feature_names)

        half = window_size // 2
        F = len(feature_names)  # Total features including window positions
        
        logging.info(f"[image_editor] Building features: window={window_size}x{window_size}, "
                    f"base_features={len(base_names)}, total_features={F}")

        # For window_size=1, use simple cube approach
        # Use base_names which has the correct model feature names
        if window_size == 1:
            cube = np.zeros((H, W, F), dtype=np.float32)
            for i, name in enumerate(base_names):
                if name == "red_channel":
                    if len(chans) < 3:
                        return None, (0, 0)
                    cube[:, :, i] = chans[0]
                elif name == "green_channel":
                    if len(chans) < 3:
                        return None, (0, 0)
                    cube[:, :, i] = chans[1]
                elif name == "blue_channel":
                    if len(chans) < 3:
                        return None, (0, 0)
                    cube[:, :, i] = chans[2]
                elif name.startswith("band_"):
                    try:
                        k = int(name.split("_", 1)[1])
                        idx = k - 1
                    except Exception:
                        return None, (0, 0)
                    if 0 <= idx < len(chans):
                        cube[:, :, i] = chans[idx]
                    else:
                        return None, (0, 0)
                elif name in expr_images:
                    cube[:, :, i] = expr_images[name]
                else:
                    logging.warning(f"[image_editor] Unknown feature name: {name}")
                    return None, (0, 0)
            
            X_flat = cube.reshape(H * W, F)
            return X_flat, (H, W)
        
        # For window_size > 1, pad image and extract neighborhoods
        # Pad channels and expression images
        padded_chans = [np.pad(ch, half, mode='edge') for ch in chans]
        padded_exprs = {name: np.pad(img, half, mode='edge') 
                       for name, img in expr_images.items()}
        
        # Build feature array: iterate over each pixel, then window positions, then base features
        X_flat = np.zeros((H * W, F), dtype=np.float32)
        
        for y in range(H):
            for x in range(W):
                flat_idx = y * W + x
                feat_idx = 0
                
                # Window positions in row-major order
                for dr in range(-half, half + 1):
                    for dc in range(-half, half + 1):
                        py = y + half + dr  # Offset by padding
                        px = x + half + dc
                        
                        # Extract each base feature at this window position
                        for base_name in base_names:
                            if base_name == "red_channel":
                                if len(padded_chans) < 3:
                                    return None, (0, 0)
                                val = padded_chans[0][py, px]
                            elif base_name == "green_channel":
                                if len(padded_chans) < 3:
                                    return None, (0, 0)
                                val = padded_chans[1][py, px]
                            elif base_name == "blue_channel":
                                if len(padded_chans) < 3:
                                    return None, (0, 0)
                                val = padded_chans[2][py, px]
                            elif base_name.startswith("band_"):
                                try:
                                    k = int(base_name.split("_", 1)[1])
                                    idx = k - 1
                                except Exception:
                                    return None, (0, 0)
                                if 0 <= idx < len(padded_chans):
                                    val = padded_chans[idx][py, px]
                                else:
                                    return None, (0, 0)
                            elif base_name in padded_exprs:
                                val = padded_exprs[base_name][py, px]
                            else:
                                logging.warning(f"[image_editor] Unknown base feature: {base_name}")
                                return None, (0, 0)
                            
                            X_flat[flat_idx, feat_idx] = val
                            feat_idx += 1
        
        return X_flat, (H, W)


    def _persist_classification_enabled(self, enabled: bool):
        """
        Merge the classification state into self.modifications so save_modifications_to_file()
        writes it to the .ax.  When disabled, set to None so _write_ax() deletes the key.
        """
        mods = dict(self.modifications or {})
        if enabled:
            block = dict(mods.get("classification") or {})
            block.update({"mode": "sklearn", "enabled": True})
            mods["classification"] = block
        else:
            # Setting to None lets _write_ax() treat it as a delete across scopes.
            mods["classification"] = None
        self.modifications = mods

    # ==================== .ax JSON Editor Methods ====================
    
    def _on_ax_group_toggled(self, checked):
        """Handle expand/collapse of the JSON editor group box."""
        # Show/hide the content container widget
        if hasattr(self, 'ax_content_widget'):
            self.ax_content_widget.setVisible(checked)
        
        # Adjust the group box size and policy
        if hasattr(self, 'ax_group'):
            if checked:
                # When expanded, allow vertical growth
                self.ax_group.setMinimumHeight(0)
                self.ax_group.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
                self.ax_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
            else:
                # When collapsed, fix to title height only
                self.ax_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                self.ax_group.setMinimumHeight(20)
                self.ax_group.setMaximumHeight(25)
        
        if checked:
            # When expanded, update the JSON display
            self._update_ax_json_display()
    
    def _update_ax_json_display(self):
        """Update the JSON text box with current modifications."""
        if not hasattr(self, 'ax_json_edit'):
            return
        
        # Only update if the group box is expanded
        if hasattr(self, 'ax_group') and not self.ax_group.isChecked():
            return  # Don't update if collapsed
        
        try:
            import json
            # Format JSON nicely
            json_str = json.dumps(self.modifications, indent=4, default=str)
            
            # Only update if content changed to avoid cursor jump
            current_text = self.ax_json_edit.toPlainText()
            if current_text != json_str:
                # Save cursor position
                cursor = self.ax_json_edit.textCursor()
                pos = cursor.position()
                
                self.ax_json_edit.setPlainText(json_str)
                
                # Restore cursor position if possible
                if pos <= len(json_str):
                    cursor.setPosition(pos)
                    self.ax_json_edit.setTextCursor(cursor)
        except Exception as e:
            logging.warning(f"[_update_ax_json_display] Error: {e}")
    
    def _on_ax_refresh_clicked(self):
        """Reload JSON from the .ax file on disk."""
        try:
            import json
            ax_path = self._ax_path_for(self.image_filepath)
            if ax_path and os.path.exists(ax_path):
                with open(ax_path, "r", encoding="utf-8") as f:
                    disk_mods = json.load(f) or {}
                json_str = json.dumps(disk_mods, indent=4, default=str)
                self.ax_json_edit.setPlainText(json_str)
                logging.info(f"[_on_ax_refresh_clicked] Loaded JSON from {ax_path}")
            else:
                QtWidgets.QMessageBox.information(
                    self, "No .ax file", 
                    f"No .ax file found for this image.\nPath: {ax_path}"
                )
        except Exception as e:
            logging.error(f"[_on_ax_refresh_clicked] Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load .ax file:\n{e}")
    
    def _on_ax_apply_json_clicked(self):
        """Parse JSON from text box and update UI controls."""
        try:
            import json
            json_str = self.ax_json_edit.toPlainText().strip()
            if not json_str:
                QtWidgets.QMessageBox.warning(self, "Empty JSON", "The JSON text box is empty.")
                return
            
            try:
                new_mods = json.loads(json_str)
            except json.JSONDecodeError as je:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid JSON", 
                    f"Failed to parse JSON:\n{je}\n\nPlease check the syntax."
                )
                return
            
            if not isinstance(new_mods, dict):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid JSON", 
                    "JSON must be a dictionary/object at the top level."
                )
                return
            
            # Update modifications
            self.modifications = new_mods
            
            # Sync all UI controls from the new modifications
            self._sync_ui_from_modifications()

            # Sync other UI elements that _sync_ui_from_modifications might miss
            self._sync_hist_combo_from_mods()
            self._sync_mask_polygon_from_mods()
            self._populate_mask_polygon_menu()

            # Reapply to see the changes
            self.reapply_modifications()

            # Ensure append button state is correct
            self._update_append_button_state()

            logging.info("[_on_ax_apply_json_clicked] Applied JSON to UI")
            
        except Exception as e:
            logging.error(f"[_on_ax_apply_json_clicked] Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to apply JSON:\n{e}")