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

# Import necessary classes from our package
from .machine_learning_manager import MachineLearningManager
from .loaders import ImageProcessor

from .utils import *

class PolygonManager(QtWidgets.QDialog):
    clear_all_polygons_signal = QtCore.pyqtSignal()  # Signal to clear all polygons across all images
    edit_group_signal = QtCore.pyqtSignal(str)       # Signal to initiate editing of a specific group
    polygons_visibility_changed = QtCore.pyqtSignal(bool)  # Signal when Show Polys checkbox is toggled

    def __init__(self, parent=None):
        super(PolygonManager, self).__init__(parent)
        self.setWindowTitle(f"{getattr(self.parent(), 'project_name', 'Project')}")
        
        # Remove standard Help (?) button from title bar
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.layout = QtWidgets.QVBoxLayout(self)

        # === Show Polygons & Labels Checkboxes (at the very top) ===
        self.top_controls_layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.top_controls_layout)

        self.show_polys_checkbox = QtWidgets.QCheckBox("Show Polys")
        self.show_polys_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background-color: #FFD700; 
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked {
                background-color: white; 
                border: 1px solid gray;
            }
        """)
        self.show_polys_checkbox.setChecked(True)  # Default: polygons visible
        self.show_polys_checkbox.setToolTip("Toggle visibility of all polygons in viewers")
        self.show_polys_checkbox.stateChanged.connect(self._on_show_polys_changed)
        self.top_controls_layout.addWidget(self.show_polys_checkbox)

        # Show Labels Checkbox
        self.show_labels_checkbox = QtWidgets.QCheckBox("Show Labels")
        self.show_labels_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background-color: #FFD700; 
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked {
                background-color: white; 
                border: 1px solid gray;
            }
        """)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.setToolTip("Toggle visibility of polygon labels")
        self.show_labels_checkbox.stateChanged.connect(self._on_show_labels_changed)
        self.top_controls_layout.addWidget(self.show_labels_checkbox)
        self.top_controls_layout.addStretch()

        # === List ===
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.list_widget)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        # === Import buttons ===
        self.import_button = QtWidgets.QPushButton("Import Polygons")
        self.import_button.setStyleSheet("background:#2e7d32; color:white; font-weight:600; padding:8px; border-radius:4px;")
        self.layout.addWidget(self.import_button)
        self.import_button.clicked.connect(self.import_polygons)

        # Old: one-button nearest 
        self.find_nearest_button = QtWidgets.QPushButton("Find Closest Images (Pick Polygons)")
        self.find_nearest_button.setToolTip(
            "Uses the folder currently open in the viewer. You will be prompted to select the polygon JSON files."
        )
        self.layout.addWidget(self.find_nearest_button)
        self.find_nearest_button.clicked.connect(self.on_find_nearest_clicked)

        # === Correction block 
        self.correction_group = QtWidgets.QGroupBox("Correct Polygons")
        self.correction_layout = QtWidgets.QVBoxLayout()
        self.correction_group.setLayout(self.correction_layout)
        self.layout.addWidget(self.correction_group)

        # Select JSONS_DIR
        self.select_jsons_dir_button = QtWidgets.QPushButton("Select JSONs Directory")
        self.correction_layout.addWidget(self.select_jsons_dir_button)
        self.select_jsons_dir_button.clicked.connect(self.select_jsons_dir)
        self.jsons_dir_label = QtWidgets.QLabel("No directory selected.")
        self.correction_layout.addWidget(self.jsons_dir_label)

        # Select POLYGONS_DIR
        self.select_polygons_dir_button = QtWidgets.QPushButton("Select Polygons Directory")
        self.correction_layout.addWidget(self.select_polygons_dir_button)
        self.select_polygons_dir_button.clicked.connect(self.select_polygons_dir)
        self.polygons_dir_label = QtWidgets.QLabel("No directory selected.")
        self.correction_layout.addWidget(self.polygons_dir_label)

        # Select CORRECTED_DIR
        self.select_corrected_dir_button = QtWidgets.QPushButton("Select Corrected Polygons Directory")
        self.correction_layout.addWidget(self.select_corrected_dir_button)
        self.select_corrected_dir_button.clicked.connect(self.select_corrected_dir)
        self.corrected_dir_label = QtWidgets.QLabel("No directory selected.")
        self.correction_layout.addWidget(self.corrected_dir_label)

        # Correct Polygons
        self.correct_polygons_button = QtWidgets.QPushButton("Correct Polygons")
        self.correction_layout.addWidget(self.correct_polygons_button)
        self.correct_polygons_button.clicked.connect(self.correct_polygons)

        self.import_project_button = QtWidgets.QPushButton("Import Polygons from Project")
        self.import_project_button.setStyleSheet("background:#2e7d32; color:white; font-weight:600; padding:8px; border-radius:4px;")
        self.import_project_button.setToolTip(
            "Pick a folder with polygon JSONs; I'll find nearest images, correct & fan-out, then import them."
        )
        self.layout.addWidget(self.import_project_button)
        self.import_project_button.clicked.connect(self.on_import_polygons_from_project)

        # Hide the old, now-redundant controls 
        self.find_nearest_button.setVisible(False)
        self.correction_group.setVisible(False)

        # === Misc buttons ===
        self.clear_all_button = QtWidgets.QPushButton("Clear All Groups")
        self.clear_all_button.setStyleSheet("background:#424242; color:white; font-weight:600; padding:8px; border-radius:4px;")
        self.layout.addWidget(self.clear_all_button)
        self.clear_all_button.clicked.connect(self.on_clear_all_polygons)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setStyleSheet("background:#424242; color:white; font-weight:600; padding:8px; border-radius:4px;")
        self.layout.addWidget(self.close_button)
        self.close_button.clicked.connect(self.close)

        # === State ===
        self.jsons_dir = ""
        self.polygons_dir = ""
        self.corrected_dir = ""
        self.current_root = None
        self.current_root_filepaths = set()

        # Dangerous delete-all (project)
        self.delete_all_polygons_button = QtWidgets.QPushButton("Delete ALL Polygons (Project)")
        self.delete_all_polygons_button.setToolTip(
            "Permanently deletes every *_polygons.json in the project and clears overlays."
        )
        self.delete_all_polygons_button.setStyleSheet("background:#c62828; color:white; font-weight:600;")
        self.layout.addWidget(self.delete_all_polygons_button)
        self.delete_all_polygons_button.clicked.connect(self.on_delete_all_polygons)
        self.list_widget.itemClicked.connect(self.on_item_clicked_select_in_viewers)
        
    def _on_show_labels_changed(self):
        """
        Called when 'Show Labels' checkbox is toggled.
        Updates label visibility in all viewers.
        """
        checked = self.show_labels_checkbox.isChecked()
        for vw_widget, viewer in self._iter_viewers():
            if hasattr(viewer, "set_labels_visible"):
                viewer.set_labels_visible(checked)

    def _select_group_in_viewers(self, group_name: str, additive: bool = False, center: bool = True):
        """
        Select only the polygons named `group_name` in all viewers.
        If additive is False, clear any previous polygon selections first.
        If center is True, perform "Zoom to Polygon" logic (Fit + Padding)
        and then ZOOM OUT by 10% (scale 0.9) as requested.
        """
        if not group_name:
            return
            
        # First pass: select the polygon items
        for vw_widget, viewer in self._iter_viewers():
            # Clear previous selections (only polygon items) unless additive
            if not additive and hasattr(viewer, "get_all_polygons"):
                try:
                    for it in viewer.get_all_polygons():
                        try:
                            it.setSelected(False)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Select the requested group
            if hasattr(viewer, "get_all_polygons"):
                for it in viewer.get_all_polygons():
                    if getattr(it, "name", None) == group_name:
                        try:
                            it.setSelected(True)
                        except Exception:
                            pass
        
        # Second pass: "Zoom to Polygon" + 10% Zoom Out
        if center:
            for vw_widget, viewer in self._iter_viewers():
                r = self._get_polygon_scene_rect(vw_widget, viewer, group_name)
                logging.debug(f"_select_group_in_viewers: group={group_name}, rect={r}")
                if r and r.isValid():
                    try:
                        # 1. Enforce Minimum Size (same as zoom_to_groups)
                        min_dim = 100.0
                        if r.width() < min_dim:
                            cx = r.center().x()
                            r.setLeft(cx - min_dim/2)
                            r.setRight(cx + min_dim/2)
                        if r.height() < min_dim:
                            cy = r.center().y()
                            r.setTop(cy - min_dim/2)
                            r.setBottom(cy + min_dim/2)

                        # 2. Standard Padding (same as zoom_to_groups: 20%)
                        pad_x = max(50.0, r.width() * 0.20)
                        pad_y = max(50.0, r.height() * 0.20)
                        
                        view_rect = r.adjusted(-pad_x, -pad_y, pad_x, pad_y)
                        
                        # 3. Fit in View (The "Zoom to Polygon" step)
                        viewer.fitInView(view_rect, QtCore.Qt.KeepAspectRatio)
                        viewer.centerOn(view_rect.center())
                        
                        # 4. Zoom Out 10% ("redies the zoom 10%")
                        # scale(0.9, 0.9) shrinks the view matrix, effectively zooming OUT
                        viewer.scale(0.9, 0.9)
                        
                        viewer.setFocus()
                    except Exception as e:
                        logging.debug(f"_select_group_in_viewers: fit+scale failed: {e}")

    def _get_polygon_scene_rect(self, viewer_widget, viewer, group_name):
        """
        Compute the BOUNDING RECT of the polygon geometry in scene coordinates.
        
        Uses robust coordinate mapping (via _ZoomBar logic) to handle:
        - Resize (scaling)
        - Crop (translation + scaling)
        - Rotation
        
        Returns a QRectF covering the polygon geometry (NOT the label).
        """
        # Get viewer info
        idata = viewer_widget.get("image_data") if isinstance(viewer_widget, dict) else None
        fp = getattr(idata, "filepath", None) if idata is not None else None
        ax_config = getattr(idata, "ax_config", {}) if idata else {}
        
        if not fp:
            return None
        
        # Get stored polygon data
        parent = self.parent()
        payload = (parent.all_polygons.get(group_name, {}) or {}).get(fp, {})
        pts = payload.get("points") or []
        
        if not pts:
            return None
        
        # Get stored reference size
        ref_size = payload.get("image_ref_size") or {}
        ref_w = int(ref_size.get("w") or 0)
        ref_h = int(ref_size.get("h") or 0)
        
        # Fallback to raw shape if ref size missing
        if not ref_w or not ref_h:
            if idata and hasattr(idata, 'raw_shape') and idata.raw_shape:
                ref_h, ref_w = idata.raw_shape[:2]
        
        # Get current Scene dimensions (this is what we map TO)
        scene_w, scene_h = 100, 100
        try:
            sr = viewer.sceneRect()
            scene_w, scene_h = sr.width(), sr.height()
        except Exception:
            pass
            
        scene_pts = []
        
        # Import mapper
        try:
            from .image_viewer import _ZoomBar
            
            for (x, y) in pts:
                if ref_w > 0 and ref_h > 0:
                    # Normalize raw (0..1)
                    nx = float(x) / float(ref_w)
                    ny = float(y) / float(ref_h)
                    
                    # Map to Scene Normalized (0..1) handling Crop/Resize/Rot
                    nsx, nsy = _ZoomBar._map_point_raw_to_scene(
                        nx, ny, ref_w, ref_h, scene_w, scene_h, ax_config
                    )
                    
                    # Map to Absolute Scene
                    sx = nsx * scene_w
                    sy = nsy * scene_h
                    scene_pts.append(QtCore.QPointF(sx, sy))
                else:
                    # No ref dims? Just assume direct mapping?
                    scene_pts.append(QtCore.QPointF(x, y))

        except ImportError:
            # Fallback to simple scaling if _ZoomBar unavailable
            logging.warning("Could not import _ZoomBar for polygon rect")
            # Get pixmap dims
            pixitem = getattr(viewer, '_image', None)
            if pixitem and pixitem.pixmap() and not pixitem.pixmap().isNull():
                vw, vh = pixitem.pixmap().width(), pixitem.pixmap().height()
                if ref_w > 0 and ref_h > 0:
                    sx = float(vw) / float(ref_w)
                    sy = float(vh) / float(ref_h)
                    for (x, y) in pts:
                        scene_pts.append(QtCore.QPointF(x * sx, y * sy))
        
        if scene_pts:
            poly = QtGui.QPolygonF(scene_pts)
            return poly.boundingRect()
        
        return None

    def on_item_clicked_select_in_viewers(self, item: QtWidgets.QListWidgetItem):
        """
        Single click in the list -> select ONLY that polygon in viewers,
        unless Ctrl/Shift are held (additive selection).
        """
        group_name = item.data(QtCore.Qt.UserRole)
        mods = QtWidgets.QApplication.keyboardModifiers()
        additive = bool(mods & (Qt.ControlModifier | Qt.ShiftModifier))
        self._select_group_in_viewers(group_name, additive=additive)

    def _infer_image_type_from_group(self, group_name: str, default: str = "RGB") -> str:
        """
        Infer image type from a free-form group name, robustly.
        Returns "RGB" or "thermal".
        """
        import re
        g = (group_name or "").lower()
        if "rgb" in g:
            return "RGB"
        if "thermal" in g:
            return "thermal"
        # treat a standalone token "ir" (underscores or non-word separators) as thermal
        if re.search(r'(^|[\W_])ir($|[\W_])', g):
            return "thermal"
        return default

    def import_polygons_from_files(self, file_paths):
        """
        Import a provided list of *_polygons.json files with viewer-aware scaling.
        Persists polygons in IMAGE coords at the target's EFFECTIVE size (post-.ax)
        and draws via target->viewer->scene mapping.
        """
        import os, re, json, logging
        import numpy as np
        from PyQt5 import QtCore, QtGui
        logger = logging.getLogger(__name__)
        logger.info(f"Starting direct import for {len(file_paths)} corrected polygon file(s).")

        if not file_paths:
            logger.warning("No files provided to import_polygons_from_files().")
            return

        parent = self.parent()

        # -------- helpers --------
        def _all_viewers_for_filepath(fp):
            out = []
            viewer_widgets = getattr(parent, 'viewer_widgets', []) or []
            # Normalize the search path for comparison
            fp_norm = os.path.normpath(fp).lower() if fp else ""
            logger.info(f"[_all_viewers_for_filepath] Looking for fp={fp} (normalized: {fp_norm})")
            logger.info(f"[_all_viewers_for_filepath] Total viewer_widgets: {len(viewer_widgets)}")
            for i, w in enumerate(viewer_widgets):
                idata = w.get('image_data')
                idata_fp = getattr(idata, 'filepath', None) if idata is not None else None
                # Normalize widget path for comparison
                idata_fp_norm = os.path.normpath(idata_fp).lower() if idata_fp else ""
                logger.info(f"[_all_viewers_for_filepath] Widget {i}: filepath={idata_fp} (normalized: {idata_fp_norm})")
                if idata_fp_norm and idata_fp_norm == fp_norm:
                    v = w.get('viewer')
                    if v:
                        out.append(v)
                        logger.info(f"[_all_viewers_for_filepath] Found matching viewer at widget {i}")
            v = parent.get_viewer_by_filepath(fp)
            if v and v not in out:
                out.append(v)
                logger.info(f"[_all_viewers_for_filepath] Found via get_viewer_by_filepath")
            logger.info(f"[_all_viewers_for_filepath] Total viewers found: {len(out)}")
            return out

        def _eff_size(fp):
            """
            Try in order:
              1) parent's .ax-aware fast size
              2) any open viewer image size
              3) disk header (cv2.imdecode)
              4) unknown -> (None, None)
            """
            # 1) post-.ax fast path
            try:
                if hasattr(parent, "_size_after_ax_fast_from_file"):
                    h, w = parent._size_after_ax_fast_from_file(fp)
                    if h and w:
                        return int(h), int(w)
            except Exception:
                pass
            # 2) viewer image size
            vlist = _all_viewers_for_filepath(fp)
            if vlist:
                try:
                    img = getattr(vlist[0], "image_data", None)
                    if img is not None and getattr(img, "image", None) is not None:
                        h, w = img.image.shape[:2]
                        return int(h), int(w)
                except Exception:
                    pass
            # 3) disk header
            try:
                import cv2
                im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if im is not None:
                    h, w = im.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            # 4) give up
            return (None, None)

        def _viewer_basis_hw(viewer, fallback_hw):
            # Prefer parent helper; else viewer image; else fallback
            try:
                if hasattr(parent, "_viewer_basis_hw"):
                    return parent._viewer_basis_hw(viewer, fallback_hw=fallback_hw)
            except Exception:
                pass
            try:
                img = getattr(viewer, "image_data", None)
                if img is not None and getattr(img, "image", None) is not None:
                    h, w = img.image.shape[:2]
                    return (int(h), int(w))
            except Exception:
                pass
            th, tw = fallback_hw
            return (int(th or 0), int(tw or 0))

        def _scene_pts_for_viewer(viewer, img_pts, img_size_hw=None):
            """
            Convert image coordinates to scene coordinates.
            img_pts: list of (x, y) tuples in IMAGE coordinates
            img_size_hw: (height, width) of the image basis for img_pts, or None to use viewer.image_data
            """
            if viewer is None or not hasattr(viewer, '_image') or viewer._image is None:
                return [QtCore.QPointF(x, y) for (x, y) in img_pts]
            
            pixitem = viewer._image
            pixmap = pixitem.pixmap()
            if pixmap is None or pixmap.isNull():
                return [QtCore.QPointF(x, y) for (x, y) in img_pts]
            
            pw, ph = max(1, pixmap.width()), max(1, pixmap.height())
            
            # Determine the image basis size
            if img_size_hw and img_size_hw[0] and img_size_hw[1]:
                img_h, img_w = img_size_hw
            elif hasattr(viewer, 'image_data') and viewer.image_data is not None:
                img = getattr(viewer.image_data, 'image', None)
                if img is not None:
                    img_h, img_w = img.shape[:2]
                else:
                    img_h, img_w = ph, pw
            else:
                img_h, img_w = ph, pw
            
            result = []
            for (x, y) in img_pts:
                # Scale from image coords to pixmap coords
                x_pix = float(x) * (pw / float(img_w))
                y_pix = float(y) * (ph / float(img_h))
                # Map pixmap coords to scene coords
                scene_pt = pixitem.mapToScene(QtCore.QPointF(x_pix, y_pix))
                result.append(scene_pt)
            return result

        # id → root_name
        try:
            id_to_root = parent.id_to_root
        except AttributeError:
            logger.critical("Internal error: 'id_to_root' attribute not found in MainWindow.")
            return

        # Collect all known image paths
        all_image_filepaths = []
        for _gn, fps in getattr(parent, "multispectral_image_data_groups", {}).items():
            all_image_filepaths.extend(fps or [])
        for _gn, fps in getattr(parent, "thermal_rgb_image_data_groups", {}).items():
            all_image_filepaths.extend(fps or [])
        possible_exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']

        # Optional root mapping
        root_mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'root_mapping.json')
        try:
            with open(root_mapping_path, 'r', encoding='utf-8') as rm_file:
                root_mapping = json.load(rm_file)
        except Exception:
            root_mapping = {}

        # Canonical index
        canon_to_paths = {}
        for fp in all_image_filepaths:
            base = os.path.splitext(os.path.basename(fp))[0].lower()
            c = self._canonicalize_base(base)
            canon_to_paths.setdefault(c, []).append(fp)

        def _search_image(filename_no_ext: str):
            q = (filename_no_ext or "").lower()
            # exact
            for ext in possible_exts:
                expected = q + ext
                for fp in all_image_filepaths:
                    if os.path.basename(fp).lower() == expected:
                        return os.path.abspath(fp)
            # canonical
            cand = canon_to_paths.get(self._canonicalize_base(q), [])
            if cand:
                for fp in cand:
                    if q in os.path.splitext(os.path.basename(fp))[0].lower():
                        return os.path.abspath(fp)
                return os.path.abspath(cand[0])
            return None

        imported = 0
        for idx, file_path in enumerate(file_paths, start=1):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    polygon_data = json.load(f)
            except Exception as e:
                logger.error(f"[{idx}] Failed to read '{file_path}': {e}")
                continue

            pts_in = polygon_data.get('points', []) or []
            json_name = (polygon_data.get('name') or '').strip()
            root_raw = polygon_data.get('root', '')
            coordinates = polygon_data.get('coordinates', {})
            coord_space = (polygon_data.get('coord_space') or 'image').lower()
            image_ref = polygon_data.get('image_ref_size') or {}

            if not pts_in:
                logger.warning(f"[{idx}] Skipping invalid polygon (no points): {file_path}")
                continue

            # normalize root from JSON
            if isinstance(root_raw, int):
                root_str = str(root_raw)
            elif isinstance(root_raw, str):
                root_str = root_raw.strip()
            else:
                root_str = ""
            try:
                root_num = int(root_str)
            except Exception:
                root_num = None

            # Verify root exists in current project, fall back to current root if not
            root_name = id_to_root.get(root_num) if root_num is not None else None
            if not root_name:
                fb_root_id, fb_root_name, _ = self._fallback_current_root_and_first_viewer()
                logger.warning(f"[{idx}] Unknown root '{root_raw}'. Falling back to current '{fb_root_name}' (id {fb_root_id}).")
                root_str = fb_root_id
                root_num = int(fb_root_id) if fb_root_id else 0
                root_name = fb_root_name
            
            logger.info(f"[{idx}] Using root: {root_str} (name: {root_name})")

            # parse filename: <group>_<imgbase>_polygons.json
            base_name = os.path.basename(file_path)
            m = re.match(r'^(?P<group>.*?)_(?P<img>.*?)_polygons\.json$', base_name, re.IGNORECASE)
            if not m:
                logger.warning(f"[{idx}] Unexpected filename: '{base_name}'. Skipping.")
                continue
            group_name = m.group('group')
            imgbase = m.group('img')

            # force polygon name -> group
            name = group_name if group_name else (json_name or "polygon")

            # resolve image
            image_fp = _search_image(imgbase)
            logger.info(f"[{idx}] Searching for image base '{imgbase}' -> found: {image_fp}")
            
            if not image_fp and root_mapping:
                # last-chance map by type
                b = base_name.lower()
                if '8b' in b:
                    img_type = 'RGB'
                elif 'ir' in b:
                    img_type = 'thermal'
                else:
                    img_type = self._infer_image_type_from_group(group_name, default='RGB')
                entry = root_mapping.get(str(root_num)) if root_num is not None else None
                if isinstance(entry, dict):
                    mapped = entry.get(img_type)
                    if isinstance(mapped, str) and mapped:
                        image_fp = _search_image(os.path.splitext(mapped)[0])
                        logger.info(f"[{idx}] Root mapping fallback for {img_type} -> {image_fp}")

            if not image_fp:
                fb_root_id, fb_root_name, fb_fp = self._fallback_current_root_and_first_viewer()
                image_fp = fb_fp
                logger.info(f"[{idx}] No image found, using fallback viewer filepath: {image_fp}")
                if not image_fp:
                    logger.warning(f"[{idx}] No resolvable image & no viewer; skipping '{base_name}'.")
                    continue
                root_str = fb_root_id  # align

            # ---------- scale to TARGET EFFECTIVE basis ----------
            th, tw = _eff_size(image_fp)

            # Try adopting a live viewer basis if unknown
            if not (th and tw):
                vlist_for_fp = _all_viewers_for_filepath(image_fp)
                if vlist_for_fp:
                    try:
                        vimg = getattr(vlist_for_fp[0], "image_data", None)
                        if vimg is not None and getattr(vimg, "image", None) is not None:
                            th, tw = vimg.image.shape[:2]
                    except Exception:
                        pass

            if not (th and tw):
                logger.warning(f"[{idx}] Could not determine effective size for '{image_fp}'. Falling back at draw time.")
                th, tw = 0, 0

            # Convert to IMAGE coords if needed
            pts_image = pts_in
            if coord_space == "scene" and hasattr(parent, "_map_points_scene_to_image"):
                try:
                    size_hint = (th or 0, tw or 0, 1)
                    pts_image = parent._map_points_scene_to_image(image_fp, pts_in, size_hint, polygon_data=polygon_data)
                except Exception:
                    pts_image = pts_in

            # Scale from source image_ref_size to target effective basis
            src_w = int(image_ref.get('w') or 0)
            src_h = int(image_ref.get('h') or 0)
            if src_w > 0 and src_h > 0 and (tw and th) and (src_w != tw or src_h != th):
                sx, sy = float(tw) / float(src_w), float(th) / float(src_h)
                pts_image = [(float(x) * sx, float(y) * sy) for (x, y) in pts_image]
            else:
                pts_image = [(float(x), float(y)) for (x, y) in pts_image]

            # ---------- persist (IMAGE coords @ target effective) ----------
            parent.all_polygons.setdefault(group_name, {})
            parent.all_polygons[group_name][image_fp] = {
                'points': pts_image,
                'coord_space': 'image',
                'image_ref_size': {'w': int(tw or 0), 'h': int(th or 0)},
                'name': name,
                'root': root_str,
                'coordinates': coordinates,
                'type': polygon_data.get('type', 'polygon'),
            }
            # Mark this polygon as dirty for incremental save
            if hasattr(parent, '_mark_polygon_dirty'):
                parent._mark_polygon_dirty(group_name, image_fp)
            logger.info(f"[ImportPolygons] Persisted polygon '{name}' for file: {image_fp}")
            logger.info(f"[ImportPolygons] Points: {pts_image[:3]}... (total {len(pts_image)} points)")
            logger.info(f"[ImportPolygons] Image ref size: w={tw}, h={th}")

            # ---------- draw on ALL viewers for that filepath ----------
            viewers_found = _all_viewers_for_filepath(image_fp)
            logger.info(f"[ImportPolygons] Found {len(viewers_found)} viewers for file: {image_fp}")
            
            for viewer in viewers_found:
                vh, vw = _viewer_basis_hw(viewer, fallback_hw=(th, tw))
                logger.info(f"[ImportPolygons] Viewer basis: vw={vw}, vh={vh}")

                # adopt viewer basis if effective unknown
                tw_eff, th_eff = (tw, th)
                if not (tw_eff and th_eff) and (vw and vh):
                    tw_eff, th_eff = vw, vh

                # remove existing same-name polygon (avoid dups)
                if hasattr(viewer, "get_all_polygons"):
                    for item in list(viewer.get_all_polygons()):
                        if getattr(item, "name", None) == name:
                            try:
                                viewer._scene.removeItem(item)
                            except Exception:
                                pass

                # Convert IMAGE coords (in effective basis) to SCENE coords
                # Pass the effective image size so scaling is correct
                scene_pts = _scene_pts_for_viewer(viewer, pts_image, img_size_hw=(th_eff, tw_eff))
                logger.info(f"[ImportPolygons] Scene points: {[(p.x(), p.y()) for p in scene_pts[:3]]}... (total {len(scene_pts)})")
                
                qpoly = QtGui.QPolygonF(scene_pts)
                if hasattr(viewer, "add_polygon_to_scene"):
                    viewer.add_polygon_to_scene(qpoly, name)
                    logger.info(f"[ImportPolygons] Added polygon '{name}' to scene via add_polygon_to_scene")
                else:
                    viewer.add_polygon(qpoly, name)
                    logger.info(f"[ImportPolygons] Added polygon '{name}' to scene via add_polygon")

            imported += 1

        # persist + refresh UI using fast incremental save
        try:
            if hasattr(parent, 'save_incremental'):
                parent.save_incremental()
            else:
                parent.save_polygons_to_json()
        except Exception as e:
            logger.error(f"Saving polygons failed: {e}")
        try:
            parent.update_polygon_manager()
        except Exception as e:
            logger.error(f"Updating Polygon Manager failed: {e}")

        QtWidgets.QMessageBox.information(self, "Import Complete",
            f"Imported {imported} corrected polygon file(s) with scaling.")


    def _fallback_current_root_and_first_viewer(self):
        """
        Returns a tuple (root_id_str, root_name, target_filepath) using:
          • current PolygonManager.current_root if available
          • else infer from first viewer’s image
          • else try ProjectTab current root name/index
        Chooses the *first viewer’s* image as target filepath.
        """
        import os

        parent = self.parent()
        root_name = None
        target_fp = None

        # First viewer’s image (preferred target for the user’s request)
        try:
            vw0 = (getattr(parent, "viewer_widgets", []) or [])[0]
            idata = vw0.get("image_data") if isinstance(vw0, dict) else None
            if idata is not None:
                target_fp = getattr(idata, "filepath", None)
        except Exception:
            target_fp = None

        # 1) prefer PolygonManager’s current_root if set
        if getattr(self, "current_root", None):
            root_name = self.current_root

        # 2) if no current_root, infer root_name from the first viewer’s filepath
        if not root_name and target_fp:
            def _find_root_name_containing(fp):
                for dname in ("image_data_groups", "multispectral_image_data_groups", "thermal_rgb_image_data_groups"):
                    d = getattr(parent, dname, None)
                    if isinstance(d, dict):
                        for rn, fps in d.items():
                            if fp in (fps or []):
                                return rn
                return None
            root_name = _find_root_name_containing(target_fp)

        # 3) fallback to ProjectTab’s currently selected multispectral root if exposed
        if not root_name:
            try:
                idx = getattr(parent, "current_root_index", None)
                if idx is not None:
                    root_name = (getattr(parent, "multispectral_root_names", []) or [])[idx]
            except Exception:
                root_name = None

       
        if not target_fp and root_name:
            for dname in ("image_data_groups", "multispectral_image_data_groups", "thermal_rgb_image_data_groups"):
                d = getattr(parent, dname, None)
                if isinstance(d, dict) and root_name in d and d[root_name]:
                    target_fp = d[root_name][0]
                    break

        # Map root_name -> root_id
        root_id_str = "0"
        try:
            id_to_root = getattr(parent, "id_to_root", {}) or {}
            rootname_to_id = {v: k for k, v in id_to_root.items()}
            rid = rootname_to_id.get(root_name)
            if rid is not None:
                root_id_str = str(rid)
        except Exception:
            pass

        return root_id_str, root_name, target_fp


    def on_import_polygons_from_project(self):
        """
        One-click pipeline:
          - Ask for a folder with source polygon JSONs
          - Build KD-tree (warming cache across both folders)
          - Compute nearest for each JSON -> write *_results.json into project/jsons/…
          - Correct/fan-out into <project>/imported_polygons
          - Import the corrected polygons into the app
        """
        import os, sys, json
        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(self, "Unavailable", "No parent ProjectTab.")
            return

        project_folder = getattr(parent, "project_folder", None)
        if not project_folder:
            QtWidgets.QMessageBox.warning(self, "Project Folder Missing", "Project folder is not set.")
            return

        images_folder = getattr(parent, "current_folder_path", None)
        if not images_folder or not os.path.isdir(images_folder):
            QtWidgets.QMessageBox.warning(
                self, "Images Folder Missing",
                "Open a folder in the viewer first (so I can build the KD-tree)."
            )
            return

        # Ask for the directory that contains the SOURCE polygon JSONs
        start_dir = os.path.join(project_folder, "polygons")
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")
        polygons_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder with Polygon JSONs (source)", start_dir
        )
        if not polygons_folder:
            return

        # Collect input polygon JSONs (skip any *_results.json sitting there)
        src_jsons = [
            os.path.join(polygons_folder, f)
            for f in os.listdir(polygons_folder)
            if f.lower().endswith(".json") and not f.lower().endswith("_results.json")
        ]
        if not src_jsons:
            QtWidgets.QMessageBox.warning(self, "No JSON Files",
                                          "No input polygon JSON files found in the selected folder.")
            return

        # Resolve exiftool path
        xp = getattr(parent, "exiftool_path", None) or os.environ.get("EXIFTOOL_PATH")
        if xp and os.path.isdir(xp):
            xp = os.path.join(xp, "exiftool.exe" if sys.platform.startswith("win") else "exiftool")
        print(f"[ImportFromProject] Using exiftool at: {xp!r}")

        # Build KD-tree (warm cache with ALL project images across both folders)
        try:
            image_processor = ImageProcessor(
                exiftool_path=xp,
                images_folder=images_folder,
                json_folder=polygons_folder,   # not used for selection anymore
                project_folder=project_folder,
                batch_size=100,
                parent_widget=self,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Init Error", f"Could not initialize the ImageProcessor:\n{e}")
            return

        try:
            self._warm_cache_with_all_project_images(image_processor)
        except Exception as e:
            print(f"[ImportFromProject] Warm cache failed (continuing): {e}")

        tree, filenames, coordinates = image_processor.run()
        if not tree:
            QtWidgets.QMessageBox.warning(
                self, "Processing Failed",
                "KD-tree was not built. Check the opened images folder and EXIF metadata."
            )
            return
        print(f"[ImportFromProject] KD-tree covers {len(filenames)} images from "
              f"{len(set(os.path.dirname(f) for f in filenames))} folder(s).")

        # Where to write *_results.json
        jsons_output_folder = getattr(image_processor, "jsons_output_folder", None) or os.path.join(project_folder, "jsons")
        os.makedirs(jsons_output_folder, exist_ok=True)

        poly_base = os.path.basename(os.path.normpath(polygons_folder))
        if poly_base.lower() == "polygons":
            source_project_name = os.path.basename(os.path.dirname(os.path.normpath(polygons_folder)))
        else:
            source_project_name = poly_base
        results_dir = os.path.join(jsons_output_folder, f"JSONs_nearest_distance_Poly_{source_project_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Get target project name
        target_project_name = os.path.basename(os.path.normpath(project_folder)) if project_folder else "unknown"

        # Compute nearest and write results
        done, failed = 0, 0
        for json_file in src_jsons:
            try:
                self.process_single_json_with_processor(json_file, tree, filenames, coordinates, results_dir)
                done += 1
            except Exception as e:
                print(f"[ImportFromProject] Failed nearest for: {json_file} -> {e}")
                failed += 1

        # Prepare correction dirs/fields
        self.jsons_dir = results_dir
        self.polygons_dir = polygons_folder
        self.corrected_dir = os.path.join(project_folder, "imported_polygons")
        os.makedirs(self.corrected_dir, exist_ok=True)

        # Run correction/fan-out with project names for metadata
        try:
            corrected_files = self.run_correction_process(
                source_project=source_project_name,
                target_project=target_project_name
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Correction Error", f"Correction failed:\n{e}")
            return

        if not corrected_files:
            QtWidgets.QMessageBox.warning(self, "Nothing to Import",
                                          "No corrected polygons were created.")
            return

        self.import_polygons_from_files(corrected_files)

    def collect_results_from_json(self, json_file_path, tree, filenames, coordinates):
        """
        Same logic as process_single_json_with_processor, but returns the results list
        instead of writing a <name>_results.json file.
        """
        import os, sys, json

        print(f"\nCollecting results for JSON file: {json_file_path}")
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {json_file_path}: {e}")
            return []

        polygons = data.get('polygons') or data.get('points') or data.get('coordinates')
        if not polygons:
            print(f"No polygons or points found in JSON file: {json_file_path}", file=sys.stderr)
            return []

        target_coords_list = []
        if 'coordinates' in data:
            target = data['coordinates']
            lat = target.get('latitude'); lon = target.get('longitude')
            if lat is not None and lon is not None:
                try:
                    target_coords_list.append((float(lat), float(lon)))
                except (TypeError, ValueError) as e:
                    print(f"Invalid coordinate values in {json_file_path}: {e}", file=sys.stderr)
        elif 'polygons' in data:
            for idx, polygon in enumerate(data['polygons'], 1):
                if not polygon:
                    continue
                try:
                    avg_lat = sum(pt[0] for pt in polygon) / len(polygon)
                    avg_lon = sum(pt[1] for pt in polygon) / len(polygon)
                    target_coords_list.append((float(avg_lat), float(avg_lon)))
                except Exception as e:
                    print(f"Invalid polygon format at index {idx} in {json_file_path}: {e}", file=sys.stderr)
        elif 'points' in data:
            for idx, pt in enumerate(data['points'], 1):
                try:
                    target_coords_list.append((float(pt[0]), float(pt[1])))
                except Exception as e:
                    print(f"Invalid point values at index {idx} in {json_file_path}: {e}", file=sys.stderr)

        if not target_coords_list:
            print(f"No valid target coordinates to process in JSON file: {json_file_path}", file=sys.stderr)
            return []

        results = []
        print(f"Finding closest images for {len(target_coords_list)} targets in {json_file_path}...")
        for idx, target_coords in enumerate(target_coords_list, 1):
            try:
                closest_image, distance = self.find_nearest_images(target_coords, tree, filenames, coordinates)
                results.append({
                    'target_index': idx,
                    'closest_image': closest_image,
                    'distance_meters': distance,
                    'target_coordinates': {
                        'latitude': target_coords[0],
                        'longitude': target_coords[1],
                    }
                })
            except Exception as e:
                print(f"Error finding nearest image for target {idx} in {json_file_path}: {e}", file=sys.stderr)
                continue

        return results

    def find_nearest_images(self, polygon_coords, tree, filenames, coordinates):
        distance, index = tree.query(polygon_coords)
        closest_filename = filenames[index]
        closest_coord = coordinates[index]
        distance_meters = geodesic(polygon_coords, closest_coord).meters
        return closest_filename, distance_meters

    def _warm_cache_with_all_project_images(self, image_processor):
        import os
        parent = self.parent()

        def gather_files(group_dict):
            files = []
            if isinstance(group_dict, dict):
                for fps in group_dict.values():
                    if fps:
                        files.extend(fps)
            return files

        all_files = []
        # NEW: include unified image_data_groups as well
        all_files += gather_files(getattr(parent, "image_data_groups", {}))
        all_files += gather_files(getattr(parent, "multispectral_image_data_groups", {}))
        all_files += gather_files(getattr(parent, "thermal_rgb_image_data_groups", {}))
        if not all_files:
            print("[warm] No project images found across group dicts.")
            return

        selected = image_processor.select_unique_roots_per_dir(all_files)

        # Merge into existing cache only if missing
        cache = []
        if os.path.isfile(image_processor.cache_file):
            cache = image_processor.load_cache()

        cached = {os.path.normpath(it.get('filename', '')) for it in cache}
        missing = [p for p in selected if os.path.normpath(p) not in cached]
        print(f"[warm] candidates={len(selected)}  cached={len(cached)}  missing={len(missing)}")
        if not missing:
            # still print coverage
            folders = {os.path.dirname(os.path.normpath(it.get('filename',''))) for it in cache if it.get('filename')}
            print(f"[warm] cache already covers {len(folders)} folder(s).")
            return

        added = image_processor.batch_extract_gps_with_exiftool(missing)
        merged = {os.path.normpath(it['filename']): it for it in cache if it.get('filename')}
        for it in added:
            merged[os.path.normpath(it['filename'])] = it
        image_processor.save_cache(list(merged.values()))

        # coverage log
        folders = {os.path.dirname(p) for p in merged.keys()}
        print(f"[warm] cache now covers {len(folders)} folder(s).")

    def on_find_nearest_clicked(self):
        """
        Flow:
          1) images_folder := folder currently open in the viewer (parent.current_folder_path)
          2) ask user to select a *folder* that contains polygon JSONs
          3) warm GPS cache with ALL project images (both folders)
          4) build KD-tree from cache
          5) process all input JSONs in that folder (skip *_results.json)
             -> write <name>_results.json into <project>/jsons/JSONs_nearest_distance_Poly_<source_project>
        """
        import os, sys, json

        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(self, "Unavailable", "No parent ProjectTab.")
            return

        project_folder = getattr(parent, "project_folder", None)
        if not project_folder:
            QtWidgets.QMessageBox.warning(self, "Project Folder Missing", "Project folder is not set.")
            return

        images_folder = getattr(parent, "current_folder_path", None)
        if not images_folder or not os.path.isdir(images_folder):
            QtWidgets.QMessageBox.warning(
                self, "Images Folder Missing",
                "Open a folder in the viewer first (so I can use it to build the KD-tree)."
            )
            return

        # Prompt for polygons folder
        start_dir = os.path.join(project_folder, "polygons")
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")
        polygons_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Polygons Folder", start_dir
        )
        if not polygons_folder:
            QtWidgets.QMessageBox.information(self, "Nothing Selected", "No folder selected.")
            return

        # Gather input JSONs (skip *_results.json)
        json_files = [
            os.path.join(polygons_folder, f)
            for f in os.listdir(polygons_folder)
            if f.lower().endswith(".json") and not f.lower().endswith("_results.json")
        ]
        if not json_files:
            QtWidgets.QMessageBox.warning(self, "No JSON Files",
                                          "No input JSON files found in the selected folder.")
            return

        # Resolve exiftool path
        xp = getattr(parent, "exiftool_path", None) or os.environ.get("EXIFTOOL_PATH")
        if xp and os.path.isdir(xp):
            xp = os.path.join(xp, "exiftool.exe" if sys.platform.startswith("win") else "exiftool")
        print(f"[PolygonManager] Using exiftool at: {xp!r}")

        # Build KD-tree (after warming cache with *all* project images)
        try:
            image_processor = ImageProcessor(
                exiftool_path=xp,
                images_folder=images_folder,
                json_folder=polygons_folder,   # not used for selection anymore, but OK to pass
                project_folder=project_folder,
                batch_size=100,
                parent_widget=self,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Init Error", f"Could not initialize the ImageProcessor:\n{e}")
            return

        # >>> Warm the GPS cache with both folders before building the tree <<<
        try:
            self._warm_cache_with_all_project_images(image_processor)
        except Exception as e:
            print(f"[PolygonManager] Warm cache failed (continuing): {e}")

        tree, filenames, coordinates = image_processor.run()
        if not tree:
            QtWidgets.QMessageBox.warning(
                self, "Processing Failed",
                "KD-tree was not built. Check the opened images folder and EXIF metadata."
            )
            return
        print(f"[Nearest] KD-tree covers {len(filenames)} images from "
              f"{len(set(os.path.dirname(f) for f in filenames))} folder(s).")

        # Output folder
        jsons_output_folder = getattr(image_processor, "jsons_output_folder", None)
        if not jsons_output_folder:
            jsons_output_folder = os.path.join(project_folder, "jsons")
        os.makedirs(jsons_output_folder, exist_ok=True)

        # Name the RESULTS folder after the polygons project/folder
        poly_base = os.path.basename(os.path.normpath(polygons_folder))
        if poly_base.lower() == "polygons":
            source_project_name = os.path.basename(os.path.dirname(os.path.normpath(polygons_folder)))
        else:
            source_project_name = poly_base

        results_dir = os.path.join(jsons_output_folder, f"JSONs_nearest_distance_Poly_{source_project_name}")
        os.makedirs(results_dir, exist_ok=True)

        # Process each input JSON and save per-file results INTO that folder
        done, failed = 0, 0
        for json_file in json_files:
            try:
                self.process_single_json_with_processor(json_file, tree, filenames, coordinates, results_dir)
                done += 1
            except Exception as e:
                print(f"[PolygonManager] Failed: {json_file} -> {e}")
                failed += 1

        QtWidgets.QMessageBox.information(
            self, "Nearest Search Complete",
            f"Processed {done} file(s){' with ' + str(failed) + ' failure(s)' if failed else ''}.\n"
            f"Results in:\n{results_dir}"
        )

    def on_delete_all_polygons(self):
        """
        Permanently deletes ALL project polygon files and clears every viewer & in-memory structure.
        Uses UndoStack for reversibility.
        """
        import os, logging
        from canopie.project_tab import DeletePolygonCommand

        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete ALL Polygons",
            "This will delete ALL saved polygon files for the entire project. This action can be Undone. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        parent = self.parent()
        if not parent:
            return

        parent.undo_stack.beginMacro("Delete ALL Polygons")
        try:
             # Iterate over a copy of group names since we might remove them
             groups = list(parent.all_polygons.keys())
             for group_name in groups:
                 # Iterate over copy of filepaths
                 filepaths = list(parent.all_polygons[group_name].keys())
                 for filepath in filepaths:
                     cmd = DeletePolygonCommand(parent, group_name, filepath)
                     parent.undo_stack.push(cmd)
                     
             # Note: We do NOT delete orphaned files (files not in memory) to keep Undo consistent.
             
        finally:
             parent.undo_stack.endMacro()

        logging.info("[PolygonManager] Deleted all polygons via UndoStack.")

  

    def update_title(self):
        # Assuming self has a setWindowTitle method or a label for title
        self.setWindowTitle(f"Polygon Manager - {self.parent().project_name}")

    def select_jsons_dir(self):
        """
        Opens a dialog for the user to select the JSONs directory.
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select JSONs Directory")
        if directory:
            self.jsons_dir = directory
            self.jsons_dir_label.setText(directory)

    def select_polygons_dir(self):
        """
        Opens a dialog for the user to select the Polygons directory.
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Polygons Directory")
        if directory:
            self.polygons_dir = directory
            self.polygons_dir_label.setText(directory)

    def select_corrected_dir(self):
        """
        Opens a dialog for the user to select the Corrected Polygons directory.
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Corrected Polygons Directory")
        if directory:
            self.corrected_dir = directory
            self.corrected_dir_label.setText(directory)

    def correct_polygons(self):
        """
        Corrects polygons by matching them with the closest images and updating their 'root' field.
        """
        if not self.jsons_dir:
            QtWidgets.QMessageBox.warning(self, "Input Required", "Please select the JSONs directory.")
            return
        if not self.polygons_dir:
            QtWidgets.QMessageBox.warning(self, "Input Required", "Please select the Polygons directory.")
            return
        if not self.corrected_dir:
            QtWidgets.QMessageBox.warning(self, "Input Required", "Please select the Corrected Polygons directory.")
            return

        # Proceed with the correction process
        try:
            self.run_correction_process()
            QtWidgets.QMessageBox.information(self, "Correction Complete", "Polygons have been corrected successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during correction:\n{e}")

    def _canonicalize_base(self, base: str) -> str:
        """
        Make a suffix-tolerant key for a base name.
        Strips rightmost tokens that are common modality/channel markers.
        """
        KNOWN_SUFFIXES = {
            "radiance", "reflectance", "radiometric",
            "ir", "rgb", "thermal", "ms", "nir", "red", "green", "blue", "gray", "grey",
            "pan", "panchromatic",
            "undist", "undistorted", "ortho", "orthophoto", "orthomosaic",
            "sr", "dn", "index", "8b"
        }
        toks = base.lower().split("_")
        # strip trailing known suffix tokens (one or many)
        while toks and toks[-1] in KNOWN_SUFFIXES:
            toks.pop()
        return "_".join(toks) if toks else base.lower()

    def _build_src_polygon_index(self, polygons_dir: str):
        """
        Return two dicts:
          exact_idx[(group_lower, base_lower)] -> fullpath
          canon_idx[(group_lower, canonical_base)] -> fullpath
        """
        import os, re
        exact_idx = {}
        canon_idx = {}
        rx = re.compile(r'^(?P<group>.+?)_(?P<base>.+?)_polygons\.json$', re.IGNORECASE)

        for name in os.listdir(polygons_dir):
            if not name.lower().endswith("_polygons.json"):
                continue
            m = rx.match(name)
            if not m:
                continue
            g = m.group("group").lower()
            b = m.group("base").lower()
            fp = os.path.join(polygons_dir, name)

            exact_idx[(g, b)] = fp
            canon_b = self._canonicalize_base(b)
            canon_idx[(g, canon_b)] = fp
        return exact_idx, canon_idx

    def _resolve_src_polygon_path(self, group: str, src_base: str, exact_idx, canon_idx):
        """
        Try to find the source polygon path for (group, src_base) using:
          1) exact match
          2) canonicalized match
        Returns path or None.
        """
        g = (group or "").lower()
        b = (src_base or "").lower()

        # 1) exact
        p = exact_idx.get((g, b))
        if p:
            return p

        # 2) canonical
        cb = self._canonicalize_base(b)
        p = canon_idx.get((g, cb))
        if p:
            return p

        return None

    def run_correction_process(self, source_project=None, target_project=None):
        """
        Correct polygons by retargeting each original polygon JSON to ALL images
        in the same root as the chosen closest image (fan-out across dual folders).

        - Recursively scans self.jsons_dir for *_polygons_results.json
        - Finds the source polygon JSON from self.polygons_dir (tolerant name matching)
        - Determines the root-id from the chosen closest image
        - Loads root_mapping.json and replicates the polygon to EACH filename listed
          for that root (folder1.files, folder2.files, and optional RGB/thermal).
        - Writes corrected polygons into self.corrected_dir:
            <group>_<dest_base>_polygons.json
        - Sets: poly["name"] = <group>, poly["root"] = <root-id as string>
        - Adds: poly["import_metadata"] with source/target project info and distance
        - Returns: list of created file paths
        """
        import os, re, json, logging

        JSONS_DIR = self.jsons_dir
        POLYGONS_DIR = self.polygons_dir
        CORRECTED_DIR = self.corrected_dir
        os.makedirs(CORRECTED_DIR, exist_ok=True)
        
        # Track files created in this run
        created_files = []

        # ---- Load root_mapping.json (same convention as import_polygons) ----
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            rm_path = os.path.join(here, 'root_mapping.json')
            with open(rm_path, 'r', encoding='utf-8') as f:
                root_mapping = json.load(f)
        except Exception as e:
            logging.warning(f"[CorrectPolygons] root_mapping.json not available ({e}). "
                            f"Fan-out will be limited to the chosen closest image only.")
            root_mapping = {}

        # ---- Build base(lower) -> root_id(str) using ALL groups (multi-folder aware) ----
        base_to_rootid = {}
        parent = self.parent()
        try:
            id_to_root = getattr(parent, "id_to_root", {}) or {}   # {int_id: root_name}
            rootname_to_id = {root_name: root_id for root_id, root_name in id_to_root.items()}

            def _collect_groups():
                for attr in ("image_data_groups", "multispectral_image_data_groups", "thermal_rgb_image_data_groups"):
                    g = getattr(parent, attr, None)
                    if isinstance(g, dict) and g:
                        yield g

            for group_map in _collect_groups():
                for root_name, filepaths in group_map.items():
                    root_id = rootname_to_id.get(root_name)
                    if root_id is None:
                        continue
                    for fp in (filepaths or []):
                        base = os.path.splitext(os.path.basename(fp))[0].lower()
                        prev = base_to_rootid.get(base)
                        if prev and str(prev) != str(root_id):
                            logging.debug(f"[CorrectPolygons] base collision '{base}': {prev} -> {root_id}")
                        base_to_rootid[base] = str(root_id)
        except Exception as e:
            logging.debug(f"[CorrectPolygons] Could not build base->root map: {e}")
            base_to_rootid = {}

        # ---- Helpers for source polygon indexing & filename handling ----
        exact_idx, canon_idx = self._build_src_polygon_index(POLYGONS_DIR)
        logging.info(f"[CorrectPolygons] Indexed {len(exact_idx)} polygon files (exact), "
                     f"{len(canon_idx)} (canonical).")

        def _file_base(path_or_name: str) -> str:
            return os.path.splitext(os.path.basename(path_or_name))[0]

        # ---- Collect *_polygons_results.json recursively ----
        results_files = []
        for r, _dirs, files in os.walk(JSONS_DIR):
            for f in files:
                if f.lower().endswith("_polygons_results.json"):
                    results_files.append(os.path.join(r, f))

        if not results_files:
            raise RuntimeError(
                f"No *_polygons_results.json found under '{JSONS_DIR}'. "
                "Select the folder that contains your results (or its parent)."
            )

        # group & base are taken from the results filename: <group>_<base>_polygons_results.json
        results_name_re = re.compile(r'^(?P<group>.+?)_(?P<base>.+?)_polygons_results\.json$', re.IGNORECASE)

        processed = 0
        skipped = 0
        missing_src = 0
        fanout_count = 0

        for results_path in results_files:
            fname = os.path.basename(results_path)
            m = results_name_re.match(fname)
            if not m:
                logging.warning(f"[CorrectPolygons] Skipping unexpected results filename: {fname}")
                skipped += 1
                continue

            group = m.group("group")
            src_base = m.group("base")

            # Load results
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                if not isinstance(results, list) or not results:
                    logging.warning(f"[CorrectPolygons] Empty/invalid results: {fname}")
                    skipped += 1
                    continue
            except Exception as e:
                logging.error(f"[CorrectPolygons] Failed to read results {fname}: {e}")
                skipped += 1
                continue

            # Pick best (min distance)
            try:
                best = min(results, key=lambda r: r.get("distance_meters", float("inf")))
                closest_image = best.get("closest_image")
                if not closest_image:
                    raise ValueError("closest_image missing")
            except Exception as e:
                logging.error(f"[CorrectPolygons] Could not choose best entry in {fname}: {e}")
                skipped += 1
                continue

            src_polygon_path = self._resolve_src_polygon_path(group, src_base, exact_idx, canon_idx)
            if not src_polygon_path:
                missing_src += 1
                logging.error(f"[CorrectPolygons] Source polygon not found for '{group}_{src_base}' "
                              f"(results: {fname})")
                continue

            # Load the source polygon object
            try:
                with open(src_polygon_path, "r", encoding="utf-8") as f:
                    poly = json.load(f)
                if not isinstance(poly, dict):
                    logging.error(f"[CorrectPolygons] Polygon JSON is not an object: {src_polygon_path}")
                    skipped += 1
                    continue
            except Exception as e:
                logging.error(f"[CorrectPolygons] Failed to read polygon {src_polygon_path}: {e}")
                skipped += 1
                continue

            # Determine the root-id of the chosen closest image, using its base
            target_base = _file_base(closest_image)
            target_base_norm = target_base.lower()
            root_id = base_to_rootid.get(target_base_norm)

            if not root_id:
                orig_root = poly.get("root", "")
                root_id = str(orig_root) if orig_root is not None else ""
                logging.debug(f"[CorrectPolygons] No root-id for base '{target_base_norm}'. "
                              f"Falling back to polygon's original root '{root_id}'.")

            # Prepare fan-out list from root_mapping
            dest_bases = []

            entry = root_mapping.get(str(root_id)) if root_id != "" else None
            if isinstance(entry, dict):
                # folder1.files
                try:
                    f1 = entry.get("folder1", {}) or {}
                    for name in (f1.get("files") or []):
                        if name:
                            dest_bases.append(_file_base(name))
                except Exception:
                    pass

                # folder2.files
                try:
                    f2 = entry.get("folder2", {}) or {}
                    for name in (f2.get("files") or []):
                        if name:
                            dest_bases.append(_file_base(name))
                except Exception:
                    pass

                # Optional singletons (RGB / thermal)
                for k in ("RGB", "thermal"):
                    val = entry.get(k)
                    if isinstance(val, str) and val:
                        dest_bases.append(_file_base(val))

            # Always include the directly chosen closest image
            dest_bases.append(target_base)

            # Deduplicate, preserve order
            seen = set()
            uniq_dest_bases = []
            for b in dest_bases:
                bl = (b or "").lower()
                if bl and bl not in seen:
                    seen.add(bl)
                    uniq_dest_bases.append(b)

            poly["name"] = group
            poly["root"] = str(root_id) if root_id is not None else ""
            
            # Add import metadata with distance and source/target info
            distance_meters = best.get("distance_meters", None)
            target_coords = best.get("target_coordinates", {})
            poly["import_metadata"] = {
                "source_project": source_project or "unknown",
                "source_polygon": group,
                "source_image": src_base,
                "target_project": target_project or "unknown",
                "target_image": target_base,
                "distance_meters": distance_meters,
                "source_coordinates": target_coords,  # The original polygon's coordinates
                "closest_image_path": closest_image,
            }

            wrote_here = 0
            for dest_base in uniq_dest_bases:
                out_name = f"{group}_{dest_base}_polygons.json"
                out_path = os.path.join(CORRECTED_DIR, out_name)
                
                # Update target_image in metadata for each destination
                poly_copy = copy.deepcopy(poly)
                poly_copy["import_metadata"]["target_image"] = dest_base
                
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(poly_copy, f, indent=4)
                    logging.info(f"[CorrectPolygons] Wrote: {out_path}")
                    wrote_here += 1
                    created_files.append(out_path)  # Track created file
                except Exception as e:
                    logging.error(f"[CorrectPolygons] Failed to write {out_path}: {e}")
                    skipped += 1

            if wrote_here > 0:
                processed += 1
                fanout_count += (wrote_here - 1)  # extra beyond the 'closest_image' itself

        logging.info(f"[CorrectPolygons] Done. Results files processed: {processed}, "
                     f"skipped: {skipped}, missing source: {missing_src}, "
                     f"extra fan-out files created: {fanout_count}")
        
        # Return the list of files created in this run
        return created_files

    def _collect_group_dicts(self):
        """
        Return every dict that maps root_name -> [filepaths] in the project.
        include the unified ('image_data_groups') and the split ones
        ('multispectral_image_data_groups', 'thermal_rgb_image_data_groups')
        so dual-folder projects are covered.
        """
        groups = []
        p = self.parent()
        for attr in ("image_data_groups",
                     "multispectral_image_data_groups",
                     "thermal_rgb_image_data_groups"):
            d = getattr(p, attr, None)
            if isinstance(d, dict) and d:
                groups.append(d)
        return groups

    def _build_base_to_rootid(self):
        """
        Build: { image_base(lowercase): root_id(str) } from ALL groups.
        Uses parent.id_to_root (int_id -> root_name) to invert back to IDs.
        """
        import os, logging
        p = self.parent()
        id_to_root = getattr(p, "id_to_root", {}) or {}
        # invert: root_name -> root_id
        rootname_to_id = {root_name: root_id for root_id, root_name in id_to_root.items()}

        base_to_rootid = {}
        for group_dict in self._collect_group_dicts():
            for root_name, filepaths in group_dict.items():
                root_id = rootname_to_id.get(root_name)
                if root_id is None:
                    logging.debug(f"[CorrectPolygons] No root_id for root_name '{root_name}'")
                    continue
                for fp in (filepaths or []):
                    base = os.path.splitext(os.path.basename(fp))[0].lower()
                    # last one wins if duplicates; log if a collision maps to a different id
                    if base in base_to_rootid and base_to_rootid[base] != str(root_id):
                        logging.debug(f"[CorrectPolygons] base collision '{base}' "
                                      f"{base_to_rootid[base]} -> {root_id}")
                    base_to_rootid[base] = str(root_id)
        return base_to_rootid

    def select_multispectral_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Multispectral Folder")
        if folder:
            self.multispectral_folder = folder
            self.multispectral_folder_label.setText(folder)

    def select_json_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select JSON Folder")
        if folder:
            self.json_folder = folder
            self.json_folder_label.setText(folder)

    def process_images(self):
        if not self.multispectral_folder:
            QtWidgets.QMessageBox.warning(self, "Input Required", "Please select the multispectral folder.")
            return
        if not self.json_folder:
            QtWidgets.QMessageBox.warning(self, "Input Required", "Please select the JSON folder.")
            return
        if not self.parent().project_folder:
            QtWidgets.QMessageBox.warning(self, "Project Folder Missing", "Project folder is not set.")
            return

        # --- Resolve exiftool path robustly (mirrors ImageProcessor logic) ---
        xp = getattr(self.parent(), "exiftool_path", None) or os.environ.get("EXIFTOOL_PATH")
        # If a folder was passed, append the executable name
        if xp and os.path.isdir(xp):
            xp = os.path.join(xp, "exiftool.exe" if sys.platform.startswith("win") else "exiftool")
        print(f"[PolygonManager] Using exiftool at: {xp!r}")

        project_folder = self.parent().project_folder
        image_processor = ImageProcessor(
            exiftool_path=xp,
            images_folder=self.multispectral_folder,
            json_folder=self.json_folder,
            project_folder=project_folder,
            batch_size=100,
            parent_widget=self,  # so message boxes have a QWidget parent
        )

        # Run processing
        tree, filenames, coordinates = image_processor.run()
        if not tree:
            QtWidgets.QMessageBox.warning(self, "Processing Failed", "Image processing did not complete successfully.")
            return

        # Process only “input” JSONs (skip any *_results.json produced earlier)
        json_files = [
            os.path.join(self.json_folder, f)
            for f in os.listdir(self.json_folder)
            if f.lower().endswith('.json') and not f.lower().endswith('_results.json')
        ]
        if not json_files:
            QtWidgets.QMessageBox.warning(self, "No JSON Files", "No input JSON files found in the selected JSON folder.")
            return

        jsons_output_folder = image_processor.jsons_output_folder
        os.makedirs(jsons_output_folder, exist_ok=True)

        for json_file in json_files:
            self.parent().process_single_json_with_processor(json_file, tree, filenames, coordinates, jsons_output_folder)

        QtWidgets.QMessageBox.information(self, "Processing Complete",
                                          f"All JSON files processed. Results saved in '{jsons_output_folder}'.")

    def process_single_json_with_processor(self, json_file_path, tree, filenames, coordinates, output_folder):
        """
        Processes a single JSON file using the provided KD-Tree and coordinates.
        Finds the closest image for each target coordinate in the JSON file and saves the result in the 'jsons' folder.
        """
        print(f"\nProcessing JSON file: {json_file_path}")
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {json_file_path}: {e}")
            return

        # Extract target coordinates
        polygons = data.get('polygons') or data.get('points') or data.get('coordinates')
        if not polygons:
            print(f"No polygons or points found in JSON file: {json_file_path}", file=sys.stderr)
            return

        target_coords_list = []
        if 'coordinates' in data:
            target = data['coordinates']
            lat = target.get('latitude')
            lon = target.get('longitude')
            if lat is not None and lon is not None:
                try:
                    target_coords_list.append((float(lat), float(lon)))
                except (TypeError, ValueError) as e:
                    print(f"Invalid coordinate values in {json_file_path}: {e}", file=sys.stderr)
            else:
                print(f"Missing latitude or longitude in {json_file_path}. Skipping this entry.", file=sys.stderr)
        elif 'polygons' in data:
            for idx, polygon in enumerate(data['polygons'], 1):
                if not polygon:
                    print(f"Empty polygon at index {idx} in {json_file_path}. Skipping.", file=sys.stderr)
                    continue
                try:
                    avg_lat = sum(point[0] for point in polygon) / len(polygon)
                    avg_lon = sum(point[1] for point in polygon) / len(polygon)
                    if avg_lat is not None and avg_lon is not None:
                        target_coords_list.append((float(avg_lat), float(avg_lon)))
                    else:
                        print(f"Missing coordinates in polygon at index {idx} in {json_file_path}. Skipping.", file=sys.stderr)
                except (TypeError, IndexError, ValueError) as e:
                    print(f"Invalid polygon format at index {idx} in {json_file_path}: {e}. Skipping.", file=sys.stderr)
                    continue
        elif 'points' in data:
            for idx, point in enumerate(data['points'], 1):
                lat = point[0] if len(point) > 0 else None
                lon = point[1] if len(point) > 1 else None
                if lat is not None and lon is not None:
                    try:
                        target_coords_list.append((float(lat), float(lon)))
                    except (TypeError, ValueError) as e:
                        print(f"Invalid point values at index {idx} in {json_file_path}: {e}. Skipping.", file=sys.stderr)
                else:
                    print(f"Missing latitude or longitude in point at index {idx} in {json_file_path}. Skipping.", file=sys.stderr)

        if not target_coords_list:
            print(f"No valid target coordinates to process in JSON file: {json_file_path}", file=sys.stderr)
            return

        results = []

        # Find closest images
        print(f"Finding closest images for {len(target_coords_list)} target coordinates in {json_file_path}...")
        for idx, target_coords in enumerate(target_coords_list, 1):
            try:
                closest_image, distance = self.find_nearest_images(target_coords, tree, filenames, coordinates)
                result = {
                    'target_index': idx,
                    'closest_image': closest_image,
                    'distance_meters': distance,
                    'target_coordinates': {
                        'latitude': target_coords[0],
                        'longitude': target_coords[1]
                    }
                }
                results.append(result)
                print(f"Target {idx}: Closest image is '{closest_image}' at {distance:.2f} meters.")
            except Exception as e:
                print(f"Error finding nearest image for target {idx} in {json_file_path}: {e}", file=sys.stderr)
                continue

        # Save results to a JSON file in 'jsons' folder
        json_filename = os.path.splitext(os.path.basename(json_file_path))[0]
        output_file_path = os.path.join(output_folder, f"{json_filename}_results.json")
        try:
            with open(output_file_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to: {output_file_path}")
        except Exception as e:
            print(f"Failed to save results to {output_file_path}: {e}")

    def natural_sort_key(self, s):
        """
        Generates a sorting key that considers numerical values within strings.
        Splits the input string into a list of strings and integers.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def set_polygons(self, polygon_groups):
        """
        PERFORMANCE OPTIMIZED: Only rebuild list if groups actually changed.
        Caches group names and builds reverse index for fast root-based selection.
        """
        # Build set of current group names
        new_groups = set(polygon_groups.keys()) if polygon_groups else set()
        old_groups = getattr(self, '_cached_group_names', None)
        
        # Only rebuild list widget if groups changed
        if old_groups is None or new_groups != old_groups:
            self.list_widget.clear()
            for group_name in sorted(polygon_groups.keys(), key=self.natural_sort_key):
                display_name = group_name or "Unnamed Group"
                item = QtWidgets.QListWidgetItem(display_name)
                item.setData(QtCore.Qt.UserRole, group_name)
                self.list_widget.addItem(item)
            self._cached_group_names = new_groups
            # Invalidate the reverse index since groups changed
            self._groups_by_filepath = None
        
        # After setting polygons, update the selection based on current_root
        self.update_selection_based_on_root()

    def set_current_root(self, current_root, image_data_groups):
        """
        Sets the current root and updates the selection in the list_widget.
        :param current_root: The name of the current root.
        :param image_data_groups: Dictionary mapping root names to their image filepaths.
        """
        self.current_root = current_root
        self.current_root_filepaths = set(image_data_groups.get(current_root, []))
        self.update_selection_based_on_root()
    
    def _build_reverse_index(self):
        """Build reverse index: filepath -> set of group names (for fast selection lookup)."""
        self._groups_by_filepath = {}
        parent = self.parent()
        if parent and hasattr(parent, 'all_polygons') and parent.all_polygons:
            for group_name, file_dict in parent.all_polygons.items():
                if isinstance(file_dict, dict):
                    for fp in file_dict.keys():
                        if fp not in self._groups_by_filepath:
                            self._groups_by_filepath[fp] = set()
                        self._groups_by_filepath[fp].add(group_name)

    def update_selection_based_on_root(self):
        """
        PERFORMANCE OPTIMIZED: Uses reverse index for O(files_in_root) lookups.
        Automatically selects polygons associated with the current root.
        """
        if not self.current_root:
            return
        
        # Lazily build reverse index if needed
        if not hasattr(self, '_groups_by_filepath') or self._groups_by_filepath is None:
            self._build_reverse_index()
        
        groups_by_fp = self._groups_by_filepath
        
        if groups_by_fp is not None and hasattr(self, 'current_root_filepaths'):
            # FAST PATH: O(files_in_root) instead of O(all_groups × all_files)
            # Find all groups that have polygons on files in current root
            groups_in_root = set()
            for fp in self.current_root_filepaths:
                if fp in groups_by_fp:
                    groups_in_root.update(groups_by_fp[fp])
            
            # Now select only items whose group is in groups_in_root
            self.list_widget.clearSelection()
            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                group_name = item.data(QtCore.Qt.UserRole)
                if group_name in groups_in_root:
                    item.setSelected(True)
        else:
            # FALLBACK: Original O(N²) behavior if index build failed
            self.list_widget.clearSelection()
            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                group_name = item.data(QtCore.Qt.UserRole)
                group_polygons = self.parent().all_polygons.get(group_name, {})
                associated = any(fp in self.current_root_filepaths for fp in group_polygons.keys())
                if associated:
                    item.setSelected(True)


    def get_selected_polygon_groups(self):
            selected_items = self.list_widget.selectedItems()
            return [item.data(QtCore.Qt.UserRole) for item in selected_items]

    def show_context_menu(self, position):
        item = self.list_widget.itemAt(position)
        if not item:
            return

        clicked_group = item.data(QtCore.Qt.UserRole)
        selected_groups = self.get_selected_polygon_groups()
        target_groups = selected_groups if clicked_group in selected_groups else [clicked_group]

        menu = QtWidgets.QMenu(self)
       
        zoom_action = menu.addAction("Zoom to Polygon Points")
        menu.addSeparator()
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")
        copy_action = menu.addAction("Copy")
        move_action = menu.addAction("Move")
        copy_to_viewers_action = menu.addAction("Copy to Viewers (This Root)")
        delete_all_action = menu.addAction("Delete All Polygons")
        menu.addSeparator()
        export_csv_action = menu.addAction("Export CSV (ML)")
        thumbs_action     = menu.addAction("Generate Thumbnails (ML)")
        masks_action      = menu.addAction("Generate Segmentation Masks (ML)")
            # NEW: stats
        menu.addSeparator()
        stats_action = menu.addAction("Polygon Stats…")  # NEW

        action = menu.exec_(self.list_widget.viewport().mapToGlobal(position))
        if action == edit_action:
            self.edit_selected_polygons(target_groups)
        elif action == delete_action:
            self.delete_selected_polygons(target_groups)
        elif action == copy_action:
            self.copy_selected_polygons(target_groups)
        elif action == move_action:
            self.move_selected_polygons(target_groups)
        elif action == copy_to_viewers_action:
            self.copy_selected_polygons_to_current_root_viewers(target_groups)
        elif action == delete_all_action:
            self.delete_all_polygons_for_groups(target_groups)
        elif action == zoom_action:
            self.zoom_to_groups(target_groups)
        elif action == export_csv_action:
            self._mlm_invoke(target_groups, kind="csv")
        elif action == thumbs_action:
            self._mlm_invoke(target_groups, kind="thumbs")
        elif action == masks_action:
            self._mlm_invoke(target_groups, kind="masks")
        elif action == stats_action: 
            # If multiple items are selected, use them; otherwise use the clicked item.
            groups = self.get_selected_polygon_groups() or target_groups
            self.show_polygon_stats(groups)
            
    def show_polygon_stats(self, groups=None):
            """
            Show counts based on the loaded project data (all_polygons) in memory.
            Calculates:
              - Total Polygons vs Point Sets
              - For Point Sets: counts total individual dots
              - Breakdowns per Root and per Group
            """
            import os
            from collections import Counter

            parent = self.parent()
            # Access the in-memory dictionary loaded from project.json
            all_polygons = getattr(parent, "all_polygons", {})
            if not all_polygons:
                 QtWidgets.QMessageBox.information(self, "Polygon Stats", "No polygons found in project data.")
                 return

            # Resolve root id -> name map if available
            id_to_root = getattr(parent, "id_to_root", {}) or {}

            def _root_label(root_raw):
                """Helper to make root IDs readable (e.g. '0 (RGB)')"""
                if root_raw is None or root_raw == "":
                    return "No Root ID"
                try:
                    # Handle int/string conversions
                    rid = int(float(str(root_raw).strip()))
                    rname = id_to_root.get(rid)
                    return f"{rid} ({rname})" if rname else str(rid)
                except Exception:
                    return str(root_raw).strip()

            # --- Aggregation Logic ---
            per_root = Counter()
            per_group = Counter()
            root_examples = {} # Store a filepath example for each root
            
            stats = {
                'total_objects': 0,
                'polygons': 0,
                'point_sets': 0,
                'total_individual_dots': 0
            }
            
            groups_set = set(groups) if groups else set()

            # Iterate: Group Name -> { FilePath: PolygonData }
            for group_name, file_map in all_polygons.items():
                # Filter if specific groups were requested
                if groups_set and group_name not in groups_set:
                    continue
                
                if not isinstance(file_map, dict):
                    continue
                    
                # Iterate individual objects
                for filepath, poly_data in file_map.items():
                    if not isinstance(poly_data, dict):
                        continue
                    
                    stats['total_objects'] += 1
                    per_group[group_name] += 1

                    # Check Type: Polygon or Point?
                    # Default to 'polygon' if key is missing
                    obj_type = poly_data.get('type', 'polygon')
                    points_list = poly_data.get('points', [])

                    if obj_type == 'point':
                        stats['point_sets'] += 1
                        # Count how many [x,y] coordinates are in this point set
                        num_dots = len(points_list)
                        stats['total_individual_dots'] += num_dots
                    else:
                        stats['polygons'] += 1

                    # Root Stats
                    root_val = poly_data.get('root')
                    label = _root_label(root_val)
                    per_root[label] += 1
                    
                    # Capture an example filename for this root if we don't have one
                    if label not in root_examples:
                        root_examples[label] = [os.path.basename(filepath)]
                    elif len(root_examples[label]) < 2:
                        root_examples[label].append(os.path.basename(filepath))

            # --- Formatting the Report ---
            distinct_roots = len(per_root)
            scope = f"selected groups ({len(groups_set)})" if groups_set else "ENTIRE PROJECT"

            def _format_counter(cntr, title):
                lines = [title]
                if not cntr:
                    lines.append("  (none)")
                    return "\n".join(lines)
                
                # Sort by Count (descending), then Name
                items = sorted(cntr.items(), key=lambda kv: (-kv[1], kv[0]))
                width = max((len(str(k)) for k, _ in items), default=0)
                
                for k, v in items:
                    # Add filename examples for Root stats
                    ex_str = ""
                    if cntr is per_root:
                        ex = root_examples.get(k, [])
                        if ex:
                            short_ex = [e[:30]+"..." if len(e)>30 else e for e in ex]
                            ex_str = f"   (e.g. {', '.join(short_ex)})"
                    
                    lines.append(f"  {str(k).ljust(width)} : {v}{ex_str}")
                return "\n".join(lines)

            report = []
            report.append(f"Source: Project Data (Memory)")
            report.append(f"Scope:  {scope}")
            report.append("-" * 40)
            report.append(f"TOTAL OBJECTS:      {stats['total_objects']}")
            report.append(f"  - Polygons:       {stats['polygons']}")
            report.append(f"  - Point Sets:     {stats['point_sets']}")
            if stats['point_sets'] > 0:
                report.append(f"    (containing {stats['total_individual_dots']} individual dots)")
            report.append("-" * 40)
            report.append(f"Active Roots:       {distinct_roots}")
            report.append("")
            report.append(_format_counter(per_root, "--- Counts by Root ID ---"))
            report.append("")
            report.append(_format_counter(per_group, "--- Counts by Group ---"))

            text = "\n".join(report)

            # --- Display Dialog ---
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Project Polygon Stats")
            dlg.resize(700, 500)
            lay = QtWidgets.QVBoxLayout(dlg)

            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setFont(QtGui.QFont("Courier New", 10))
            edit.setPlainText(text)
            lay.addWidget(edit)

            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.reject)
            lay.addWidget(btns)

            dlg.exec_()
    def _iter_viewers(self):
        """
        Yield (viewer_widget_dict, viewer) for all known viewers.
        Expects parent.viewer_widgets to be a list of dicts with 'viewer' and 'image_data'.
        """
        parent = self.parent()
        for w in getattr(parent, "viewer_widgets", []) or []:
            viewer = w.get("viewer") if isinstance(w, dict) else None
            if viewer is not None:
                yield (w, viewer)

    def _on_show_polys_changed(self, state):
        """Handle Show Polys checkbox state change."""
        visible = (state == QtCore.Qt.Checked)
        # Emit signal for ProjectTab to handle
        self.polygons_visibility_changed.emit(visible)
        # Also directly update all current viewers
        self._set_all_polygons_visible(visible)

    def _set_all_polygons_visible(self, visible):
        """Set visibility of all polygons in all viewers."""
        for _, viewer in self._iter_viewers():
            if hasattr(viewer, "set_polygons_visible"):
                viewer.set_polygons_visible(visible)

    def are_polygons_visible(self):
        """Return current state of Show Polys checkbox."""
        return self.show_polys_checkbox.isChecked()

    def _scene_rect_for_group_on_viewer(self, viewer_widget, viewer, group_name):
        """
        DEPRECATED: Use _get_polygon_scene_rect instead.
        This is kept for backwards compatibility - redirects to the new implementation.
        """
        return self._get_polygon_scene_rect(viewer_widget, viewer, group_name)

    def zoom_to_groups(self, groups):
        """
        Zoom every open viewer to frame the polygons/points belonging to the given groups.
        Shows the polygon with surrounding image context for better orientation.
        """
        if not groups:
            return
        for vw_widget, viewer in self._iter_viewers():
            combined = None
            for g in groups:
                r = self._get_polygon_scene_rect(vw_widget, viewer, g)
                logging.info(f"zoom_to_groups: group={g}, rect={r}")
                if r and r.isValid():
                    combined = r if combined is None else combined.united(r)

            if combined and combined.isValid():
                logging.info(f"zoom_to_groups: combined rect center=({combined.center().x():.1f}, {combined.center().y():.1f})")
                # Add extra context padding (show more of the surrounding image)
                # Use 20% of the rect size as additional padding
                pad_x = max(50.0, combined.width() * 0.20)
                pad_y = max(50.0, combined.height() * 0.20)
                view_rect = combined.adjusted(-pad_x, -pad_y, pad_x, pad_y)
                
                # Handle single-point or very small polygons - ensure minimum view size
                min_size = 100.0
                if view_rect.width() < min_size:
                    expand_x = (min_size - view_rect.width()) / 2
                    view_rect.adjust(-expand_x, 0, expand_x, 0)
                if view_rect.height() < min_size:
                    expand_y = (min_size - view_rect.height()) / 2
                    view_rect.adjust(0, -expand_y, 0, expand_y)
                
                try:
                    viewer.fitInView(view_rect, QtCore.Qt.KeepAspectRatio)
                    viewer.setFocus()
                except Exception:
                    # Fallback: ensureVisible
                    try:
                        viewer.ensureVisible(combined, 50, 50)
                        viewer.centerOn(combined.center())
                    except Exception:
                        pass


    def _mlm_invoke(self, groups, kind: str):
        """
        Headless bridge to MachineLearningManager  can reuse its UI/logic
        without opening the ML dialog window.
        - groups: list[str] of group names to operate on
        - kind: 'csv' | 'thumbs' | 'masks'
        """
        
        try:
            mlm = MachineLearningManager(parent=self.parent())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "MachineLearning Manager",
                                           f"Could not initialize ML Manager:\n{e}")
            return

        # Override selection source so ML Manager operations run on PolygonManager's selection
        mlm.get_selected_groups = (lambda g=list(groups): g)

        # Optional: don’t ever show its dialog
        mlm.show = lambda: None

        # Fire the desired action. These methods already show their own dialogs
        # (mode pickers, folder pickers, save dialogs) and use the shared parent.
        try:
            if kind == "csv":
                mlm.export_csv_data()
            elif kind == "thumbs":
                mlm.generate_thumbnails()
            elif kind == "masks":
                mlm.generate_segmentation_images()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Operation Failed", str(e))


    def delete_all_polygons_for_groups(self, group_names):
        """
        Bulk 'delete all polygons' across multiple groups (project-wide per group).
        """
        for group_name in list(group_names or []):
            self.delete_all_polygons_for_group(group_name)
                    # Persist project.json too (no recompute)
        #try:
            #if hasattr(self.parent(), "save_project_quick"):
              #  self.parent().save_project_quick(skip_recompute=True)
       # except Exception as e:
            #logging.debug(f"Quick-save after group delete-all failed: {e}")

    def copy_selected_polygons_to_current_root_viewers(self, groups=None):
        """
        Copy ONLY the selected groups' polygons from the inferred SOURCE root
        into the CURRENT/TARGET root, and show them immediately in the viewers.
        """
        from collections import Counter
        import inspect

        parent = self.parent()
        if not parent:
            return

        # 1) resolve selected groups
        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return
        groups = list(dict.fromkeys(groups))  # preserve order, unique

        # 2) resolve target (current) ms-root
        target_root = getattr(parent, "get_current_root_name", lambda: None)()
        if not target_root:
            print("No current root selected; aborting copy.")
            return

        # helper: map a filepath -> ms-root name
        def _ms_root_for_fp(fp: str):
            for r, files in (getattr(parent, "multispectral_image_data_groups", {}) or {}).items():
                try:
                    if fp in files:
                        return r
                except Exception:
                    pass
            th_groups = getattr(parent, "thermal_rgb_image_data_groups", {}) or {}
            th_names  = getattr(parent, "thermal_rgb_root_names", []) or []
            ms_names  = getattr(parent, "multispectral_root_names", []) or []
            off       = int(getattr(parent, "root_offset", 0) or 0)
            for th_root, files in th_groups.items():
                try:
                    if fp in files:
                        if th_root in th_names and ms_names:
                            th_idx = th_names.index(th_root)
                            ms_idx = th_idx - off
                            if 0 <= ms_idx < len(ms_names):
                                return ms_names[ms_idx]
                except Exception:
                    pass
            return None

        # 3) infer SOURCE ms-root from selected groups’ existing polygons
        all_polys = getattr(parent, "all_polygons", {}) or {}
        source_roots = []
        for g in groups:
            mapping = all_polys.get(g, {}) or {}
            for fp in mapping.keys():
                r = _ms_root_for_fp(fp)
                if r:
                    source_roots.append(r)

        if not source_roots:
            print("Could not infer a source root from the selected groups; aborting.")
            return

        source_root = Counter(source_roots).most_common(1)[0][0]
        if source_root == target_root:
            print(f"Source root equals target root ('{target_root}'); nothing to do.")
            return

        # keep only groups with polygons under the chosen source_root
        filtered_groups = []
        for g in groups:
            mapping = all_polys.get(g, {}) or {}
            if any(_ms_root_for_fp(fp) == source_root for fp in mapping.keys()):
                filtered_groups.append(g)
        if not filtered_groups:
            print("Selected groups have no polygons in the inferred source root; aborting.")
            return
        groups = filtered_groups

        # helper: paired thermal root for a given ms root
        def _paired_thermal_root(ms_root):
            try:
                ms_names = getattr(parent, "multispectral_root_names", []) or []
                th_names = getattr(parent, "thermal_rgb_root_names", []) or []
                off      = int(getattr(parent, "root_offset", 0) or 0)
                ms_idx   = ms_names.index(ms_root)
                th_idx   = ms_idx + off
                if 0 <= th_idx < len(th_names):
                    return th_names[th_idx]
            except Exception:
                pass
            return None

        # 4) snapshot pre-existing target-root entries so we can prune ONLY new extras
        tgt_ms_files = set((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(target_root, []))
        tgt_th_root  = _paired_thermal_root(target_root)
        tgt_th_files = set((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(tgt_th_root, []))
        tgt_all_files = tgt_ms_files | tgt_th_files

        pre_existing = set()
        for g_name, mapping in (getattr(parent, "all_polygons", {}) or {}).items():
            for fp in mapping.keys():
                if fp in tgt_all_files:
                    pre_existing.add((g_name, fp))

        # 5) see if parent.copy_polygons_between_roots supports a whitelist param
        groups_param_name = None
        try:
            sig = inspect.signature(parent.copy_polygons_between_roots)
            for candidate in ("groups", "restrict_groups", "group_names"):
                if candidate in sig.parameters:
                    groups_param_name = candidate
                    break
        except Exception:
            pass

        # IMPORTANT: update viewers immediately only when we can filter by groups.
        defer_viewer_update = (groups_param_name is None)

        kwargs = dict(
            source_root=source_root,
            target_root=target_root,
            broadcast_if_ambiguous=True,
            rescale=True,
            defer_viewer_update=defer_viewer_update,
            defer_save=True,
        )

        if groups_param_name:
            kwargs[groups_param_name] = list(groups)
            parent.copy_polygons_between_roots(**kwargs)
        else:
            # fallback: copy-all (deferred), we will prune then repaint
            parent.copy_polygons_between_roots(**kwargs)

        # 6) prune: if we had to copy-all, remove NEW polygons for groups not selected
        if not groups_param_name:
            all_polys = getattr(parent, "all_polygons", {}) or {}
            selected_set = set(groups)
            for g_name, mapping in list(all_polys.items()):
                if g_name in selected_set:
                    continue
                for fp in list(mapping.keys()):
                    if fp in tgt_all_files and (g_name, fp) not in pre_existing:
                        mapping.pop(fp, None)
                if not mapping:
                    all_polys.pop(g_name, None)

        # 7) save if possible
        if getattr(parent, "project_folder", "") and hasattr(parent, "save_polygons_to_json"):
            parent.save_polygons_to_json(root_name=target_root)
        else:
            print("Skipping polygon save: project not saved yet.")

        # 8) LIGHT repaint — no root switch, no full refresh:
        #    immediately draw overlays into any already-open viewers for the current root.
        try:
            if hasattr(parent, "_redraw_polys_for_root"):
                parent._redraw_polys_for_root(target_root)
            # keep the manager list in sync
            if hasattr(parent, "update_polygon_manager"):
                parent.update_polygon_manager()
                  # keep user on whatever root is currently visible
                current_root = parent.get_current_root_name() if hasattr(parent, "get_current_root_name") else None
                parent.refresh_viewer(root_name=current_root)
        except Exception:
            # make best-effort to nudge a repaint even if helper is missing
            viewers = None
            for attr in ("viewer_widgets", "open_viewers", "viewers", "image_viewers"):
                viewers = getattr(parent, attr, None)
                if viewers:
                    break
            if viewers:
                for vdict in (viewers if isinstance(viewers, list) else []):
                    v = vdict.get("viewer") if isinstance(vdict, dict) else vdict
                    if not v:
                        continue
                    try:
                        if hasattr(v, "scene") and v.scene():
                            v.scene().update()
                        v.update()
                    except Exception:
                        pass


    def delete_all_polygons_for_group(self, group_name: str):
        """
        Deletes EVERY polygon associated with the given group name across the whole project.
        Uses UndoStack.
        """
        import os, logging
        from canopie.project_tab import DeletePolygonCommand
        
        parent = self.parent()
        if not parent: return

        # Nothing to do if group absent
        group_map = parent.all_polygons.get(group_name)
        if not group_map:
            return

        parent.undo_stack.beginMacro(f"Delete Group '{group_name}'")
        try:
            filepaths = list(group_map.keys())
            for filepath in filepaths:
                cmd = DeletePolygonCommand(parent, group_name, filepath)
                parent.undo_stack.push(cmd)
        finally:
            parent.undo_stack.endMacro()

        logging.info(f"[PolygonManager] Deleted group '{group_name}' via UndoStack.")

    def delete_selected_polygons(self, groups=None):
        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return  # Do nothing if no selection

        parent = self.parent()
        if not parent: return
        
        from canopie.project_tab import DeletePolygonCommand

        # Determine the polygons to delete
        groups_to_delete = set(groups)
        
        parent.undo_stack.beginMacro("Delete Selected Polygons")
        try:
            for group_name in groups_to_delete:
                if group_name not in parent.all_polygons:
                    continue
                # Iterate over all filepaths associated with the group
                filepaths = list(parent.all_polygons[group_name].keys())
                for filepath in filepaths:
                    if filepath not in self.current_root_filepaths:
                        continue  # Skip if not in current root
                    
                    cmd = DeletePolygonCommand(parent, group_name, filepath)
                    parent.undo_stack.push(cmd)
        finally:
            parent.undo_stack.endMacro()
        
        # UI update is handled by commands


    def edit_selected_polygons(self, groups=None):
        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return  # Do nothing if no selection

        # Determine the polygons to edit
        groups_to_edit = set(groups)

        # Determine the polygons directory
        polygons_dir = os.path.join(
            self.parent().project_folder, 'polygons') if self.parent().project_folder else os.path.join(os.getcwd(), 'polygons')

        for group_name in groups_to_edit:
            if group_name not in self.parent().all_polygons:
                continue
            # Iterate over all filepaths associated with the group
            for filepath in list(self.parent().all_polygons[group_name].keys()):
                if filepath not in self.current_root_filepaths:
                    continue  # Skip if not in current root

                viewer = self.parent().get_viewer_by_filepath(filepath)
                if viewer:
                    if hasattr(viewer, "get_all_polygons"):
                        for item in list(viewer.get_all_polygons()):
                            if getattr(item, "name", None) == group_name:
                                try:
                                    viewer._scene.removeItem(item)
                                except Exception:
                                    pass
                                break  # Assuming one polygon per group per image

                # Determine the JSON file path
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                polygon_filename = f"{group_name}_{base_filename}_polygons.json"
                polygon_filepath = os.path.join(polygons_dir, polygon_filename)

                # Delete the JSON file if it exists
                if os.path.exists(polygon_filepath):
                    try:
                        os.remove(polygon_filepath)
                        print(f"Deleted polygon file: {polygon_filepath}")
                    except Exception as e:
                        print(f"Failed to delete polygon file {polygon_filepath}: {e}")

                # Remove the polygon from the in-memory data structure
                self.parent().all_polygons[group_name].pop(filepath, None)

        # If the group has no more polygons, remove it from the dictionary
        for group_name in list(self.parent().all_polygons.keys()):
            if not self.parent().all_polygons[group_name]:
                self.parent().all_polygons.pop(group_name, None)

        self.parent().update_polygon_manager()

        # Emit signal to start editing (automatic drawing)
        for group_name in groups_to_edit:
            self.edit_group_signal.emit(group_name)

        print("Edit mode activated. Please redraw the selected polygons.")

    def save_polygons_to_json(self, root_name=None):
        """
        Delegate saving to ProjectTab's canonical implementation.
        """
        parent = self.parent()
        if hasattr(parent, "save_polygons_to_json"):
            parent.save_polygons_to_json(root_name=root_name)

    def copy_selected_polygons(self, groups=None):
        """
        Copy selected polygon groups from the CURRENT root to a user-chosen TARGET root,
        using ProjectTab.copy_polygons_between_roots for robust pairing & rescaling.
        Only the SELECTED groups remain in the target (others are pruned).
        """
        from collections import Counter
        import os

        parent = self.parent()
        if not parent:
            return

        # --- selection ---
        groups = groups or self.get_selected_polygon_groups()
        if not groups:
            return

        # --- choose target root (robust list) ---
        roots_list = getattr(parent, "root_names", None) \
                  or getattr(parent, "multispectral_root_names", None) \
                  or list((getattr(parent, "image_data_groups", {}) or {}).keys())
        if not roots_list:
            return

        target_root, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Target Root", "Choose the target root to copy polygons to:",
            roots_list, 0, False
        )
        if not ok or not target_root:
            return

        # --- resolve SOURCE root (prefer PolygonManager.current_root / ProjectTab current) ---
        source_root = getattr(self, "current_root", None) \
                   or (parent.get_current_root_name() if hasattr(parent, "get_current_root_name") else None)

        # If still unknown, infer most common MS root from selected groups' filepaths.
        def _ms_root_for_fp(fp: str):
            for r, files in (getattr(parent, "multispectral_image_data_groups", {}) or {}).items():
                if fp in (files or []):
                    return r
            # infer from thermal via offset
            th_groups = getattr(parent, "thermal_rgb_image_data_groups", {}) or {}
            th_names  = getattr(parent, "thermal_rgb_root_names", []) or []
            ms_names  = getattr(parent, "multispectral_root_names", []) or []
            off       = int(getattr(parent, "root_offset", 0) or 0)
            for th_root, files in th_groups.items():
                if fp in (files or []):
                    if th_root in th_names and ms_names:
                        th_idx = th_names.index(th_root)
                        ms_idx = th_idx - off
                        if 0 <= ms_idx < len(ms_names):
                            return ms_names[ms_idx]
            return None

        if not source_root:
            src_roots = []
            for g in groups:
                for fp in (parent.all_polygons.get(g, {}) or {}).keys():
                    r = _ms_root_for_fp(fp)
                    if r:
                        src_roots.append(r)
            if not src_roots:
                QtWidgets.QMessageBox.information(self, "Copy Polygons", "Could not infer a source root from the selected groups.")
                return
            source_root = Counter(src_roots).most_common(1)[0][0]

        if source_root == target_root:
            QtWidgets.QMessageBox.information(self, "Copy Polygons", "Source and target roots are the same.")
            return

        # --- helpers to gather target-file sets for pruning ---
        def _paired_thermal_root(ms_root):
            try:
                ms_names = getattr(parent, "multispectral_root_names", []) or []
                th_names = getattr(parent, "thermal_rgb_root_names", []) or []
                off      = int(getattr(parent, "root_offset", 0) or 0)
                ms_idx   = ms_names.index(ms_root)
                th_idx   = ms_idx + off
                if 0 <= th_idx < len(th_names):
                    return th_names[th_idx]
            except Exception:
                pass
            return None

        tgt_ms_files = set((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(target_root, []) or [])
        tgt_th_root  = _paired_thermal_root(target_root)
        tgt_th_files = set((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(tgt_th_root, []) or [])
        tgt_all_files = tgt_ms_files | tgt_th_files

        # Snapshot target entries that existed BEFORE the copy (so we don't prune them)
        pre_existing = set()
        for g_name, mapping in (getattr(parent, "all_polygons", {}) or {}).items():
            for fp in (mapping or {}).keys():
                if fp in tgt_all_files:
                    pre_existing.add((g_name, fp))

        # --- do the copy via ProjectTab API ---
        parent.copy_polygons_between_roots(
            source_root,
            target_root,
            broadcast_if_ambiguous=True,
            rescale=True,
            defer_viewer_update=False,
            defer_save=True,   # we'll save after pruning
        )

        # --- prune: keep ONLY selected groups among newly copied target entries ---
        all_polys = getattr(parent, "all_polygons", {}) or {}
        for g_name, mapping in list(all_polys.items()):
            if g_name in groups:
                continue
            for fp in list(mapping.keys()):
                if fp in tgt_all_files and (g_name, fp) not in pre_existing:
                    mapping.pop(fp, None)
            if not mapping:
                all_polys.pop(g_name, None)

        # --- save/update ---
        if getattr(parent, "project_folder", "") and hasattr(parent, "save_polygons_to_json"):
            parent.save_polygons_to_json(root_name=target_root)
        if hasattr(parent, "update_polygon_manager"):
            parent.update_polygon_manager()

    def move_selected_polygons(self, groups=None):
        """
        Move selected polygon groups FROM the CURRENT root TO a user-chosen TARGET root.
        Internally: copy via ProjectTab.copy_polygons_between_roots, prune to selected groups,
        then remove those groups from the SOURCE root (memory + on-disk JSONs).
        """
        import os
        from collections import Counter

        parent = self.parent()
        if not parent:
            return

        # --- selection ---
        groups = groups or self.get_selected_polygon_groups()
        if not groups:
            return

        # --- choose target root ---
        roots_list = getattr(parent, "root_names", None) \
                  or getattr(parent, "multispectral_root_names", None) \
                  or list((getattr(parent, "image_data_groups", {}) or {}).keys())
        if not roots_list:
            return
        target_root, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Target Root", "Choose the target root to move polygons to:",
            roots_list, 0, False
        )
        if not ok or not target_root:
            return

        # --- source root = current ---
        source_root = getattr(self, "current_root", None) \
                   or (parent.get_current_root_name() if hasattr(parent, "get_current_root_name") else None)
        if not source_root:
            QtWidgets.QMessageBox.information(self, "Move Polygons", "No current root selected.")
            return
        if source_root == target_root:
            QtWidgets.QMessageBox.information(self, "Move Polygons", "Source and target roots are the same.")
            return

        # --- helpers to collect file sets for source/target ---
        def _paired_thermal_root(ms_root):
            try:
                ms_names = getattr(parent, "multispectral_root_names", []) or []
                th_names = getattr(parent, "thermal_rgb_root_names", []) or []
                off      = int(getattr(parent, "root_offset", 0) or 0)
                ms_idx   = ms_names.index(ms_root)
                th_idx   = ms_idx + off
                if 0 <= th_idx < len(th_names):
                    return th_names[th_idx]
            except Exception:
                pass
        src_ms_files = set((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(source_root, []) or [])
        src_th_root  = _paired_thermal_root(source_root)
        src_th_files = set((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(src_th_root, []) or [])
        src_all_files = src_ms_files | src_th_files

        tgt_ms_files = set((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(target_root, []) or [])
        tgt_th_root  = _paired_thermal_root(target_root)
        tgt_th_files = set((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(tgt_th_root, []) or [])
        tgt_all_files = tgt_ms_files | tgt_th_files

        # Snapshot pre-existing target entries (to preserve)
        pre_existing = set()
        for g_name, mapping in (getattr(parent, "all_polygons", {}) or {}).items():
            for fp in (mapping or {}).keys():
                if fp in tgt_all_files:
                    pre_existing.add((g_name, fp))

        # --- copy via ProjectTab API ---
        parent.copy_polygons_between_roots(
            source_root,
            target_root,
            broadcast_if_ambiguous=True,
            rescale=True,
            defer_viewer_update=False,
            defer_save=True,   # save after we prune & delete src
        )

        # --- prune target: keep only selected groups among newly added entries ---
        all_polys = getattr(parent, "all_polygons", {}) or {}
        for g_name, mapping in list(all_polys.items()):
            if g_name in groups:
                continue
            for fp in list(mapping.keys()):
                if fp in tgt_all_files and (g_name, fp) not in pre_existing:
                    mapping.pop(fp, None)
            if not mapping:
                all_polys.pop(g_name, None)

        # --- remove selected groups from SOURCE root (memory + on-disk files) ---
        # Resolve polygons directory
        polygons_dir = os.path.join(parent.project_folder, "polygons") if getattr(parent, "project_folder", None) \
                       else os.path.join(os.getcwd(), "polygons")

        for g_name in list(groups):
            mapping = (all_polys.get(g_name, {}) or {})
            # collect the filepaths to drop (those in source root)
            to_drop = [fp for fp in list(mapping.keys()) if fp in src_all_files]
            if not to_drop:
                continue

            # drop from memory
            for fp in to_drop:
                mapping.pop(fp, None)

            # delete per-image JSONs on disk for these (avoid stale files)
            if os.path.isdir(polygons_dir):
                for fp in to_drop:
                    base = os.path.splitext(os.path.basename(fp))[0]
                    json_path = os.path.join(polygons_dir, f"{g_name}_{base}_polygons.json")
                    try:
                        if os.path.exists(json_path):
                            os.remove(json_path)
                    except Exception:
                        pass

            # clean empty group
            if not mapping:
                all_polys.pop(g_name, None)

        # --- save/update ---
        if getattr(parent, "project_folder", "") and hasattr(parent, "save_polygons_to_json"):
            parent.save_polygons_to_json(root_name=target_root)
        if hasattr(parent, "refresh_viewer"):
            try:
                # keep user on whatever root is currently visible
                current_root = parent.get_current_root_name() if hasattr(parent, "get_current_root_name") else None
                parent.refresh_viewer(root_name=current_root)
                # If you prefer to refresh the destination too, uncomment:
                # parent.refresh_viewer(root_name=target_root)
            except TypeError:
                # in case refresh_viewer has no root_name param
                parent.refresh_viewer()

    
    
    
    def setup_logging():
        log_directory = os.path.join(os.path.expanduser("~"), "polygon_import_logs")
        os.makedirs(log_directory, exist_ok=True)  # Ensure the log directory exists

        log_file = os.path.join(log_directory, "import_polygons.log")

        # Configure the root logger
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # Log to a file, append mode
                logging.StreamHandler()  # Also log to console
            ]
        )


    def import_polygons(self):
        """
        Import user-picked polygon JSONs and apply them to the CURRENT viewers (both sides in dual mode).
        Behavior:
          • Force polygon 'name' = <group> (from filename).
          • Force polygon 'root' = current root (id), regardless of what's in the file.
          • Target images = images currently shown in viewers that belong to the current root.
            Fallback: all images in that root (MS + paired Thermal by offset).
          • Rescale using source image_ref_size -> target effective basis, then viewer-basis for drawing.
        """
        import os, re, json, logging, numpy as np
        from PyQt5 import QtCore, QtGui
        logger = logging.getLogger(__name__)

        parent = self.parent()
        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import Polygons", os.path.expanduser("~"), "JSON Files (*.json)"
        )
        if not file_paths:
            return

        # ---- helpers -----------------------------------------------------------
        def _current_root_name():
            if hasattr(parent, "get_current_root_name"):
                return parent.get_current_root_name()
            # fallback to ms root by index
            try:
                idx = getattr(parent, "current_root_index", None)
                if idx is not None:
                    return (getattr(parent, "multispectral_root_names", []) or [])[idx]
            except Exception:
                pass
            return None

        def _paired_thermal_root(ms_root):
            try:
                ms_names = getattr(parent, "multispectral_root_names", []) or []
                th_names = getattr(parent, "thermal_rgb_root_names", []) or []
                off      = int(getattr(parent, "root_offset", 0) or 0)
                ms_idx   = ms_names.index(ms_root)
                th_idx   = ms_idx + off
                if 0 <= th_idx < len(th_names):
                    return th_names[th_idx]
            except Exception:
                pass
            return None

        def _effective_size(fp):
            # 1) post-.ax
            try:
                if hasattr(parent, "_size_after_ax_fast_from_file"):
                    h, w = parent._size_after_ax_fast_from_file(fp)
                    if h and w: return int(h), int(w)
            except Exception:
                pass
            # 2) live viewer
            v = parent.get_viewer_by_filepath(fp)
            if v and getattr(v, "image_data", None) and getattr(v.image_data, "image", None) is not None:
                h, w = v.image_data.image.shape[:2]
                return int(h), int(w)
            # 3) disk header
            try:
                import cv2
                im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if im is not None:
                    h, w = im.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            return (None, None)

        def _viewer_hw(fp, fallback):
            v = parent.get_viewer_by_filepath(fp)
            if v and getattr(v, "image_data", None) and getattr(v.image_data, "image", None) is not None:
                h, w = v.image_data.image.shape[:2]
                return int(h), int(w)
            th, tw = fallback
            return int(th or 0), int(tw or 0)

        def _open_viewer_filepaths_for_root(ms_root):
            """All *currently shown* image filepaths that belong to ms_root (and its paired thermal)."""
            show = []
            th_root = _paired_thermal_root(ms_root)
            ms_files = set((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(ms_root, []) or [])
            th_files = set((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(th_root, []) or [])
            valid = ms_files | th_files
            for w in getattr(parent, "viewer_widgets", []) or []:
                idata = w.get("image_data") if isinstance(w, dict) else None
                fp = getattr(idata, "filepath", None) if idata is not None else None
                if fp and fp in valid:
                    show.append(fp)
            return list(dict.fromkeys(show))  # unique, keep order

        def _all_filepaths_for_root(ms_root):
            th_root = _paired_thermal_root(ms_root)
            ms_files = list((getattr(parent, "multispectral_image_data_groups", {}) or {}).get(ms_root, []) or [])
            th_files = list((getattr(parent, "thermal_rgb_image_data_groups", {}) or {}).get(th_root, []) or [])
            return ms_files + th_files

        # ---- resolve current root + its id ------------------------------------
        ms_root = _current_root_name()
        if not ms_root:
            QtWidgets.QMessageBox.information(self, "Import Polygons", "No current root selected.")
            return
        id_to_root = getattr(parent, "id_to_root", {}) or {}
        rootname_to_id = {v: k for k, v in id_to_root.items()}
        root_id_str = str(rootname_to_id.get(ms_root, 0))

        # ---- decide targets: prefer what the user is *currently looking at* ----
        target_filepaths = _open_viewer_filepaths_for_root(ms_root)
        if not target_filepaths:
            # fallback to *all* images in that root (MS + Thermal)
            target_filepaths = _all_filepaths_for_root(ms_root)
            if not target_filepaths:
                QtWidgets.QMessageBox.information(self, "Import Polygons", f"No images found in root '{ms_root}'.")
                return

        # ---- process each picked polygon file ---------------------------------
        imported = 0
        rx = re.compile(r'^(?P<group>.*?)_(?P<imgbase>.*?)_polygons\.json$', re.IGNORECASE)

        for idx, file_path in enumerate(file_paths, start=1):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(f"[{idx}] JSON read error '{file_path}': {e}")
                continue

            pts_in      = data.get('points', []) or []
            coord_space = (data.get('coord_space') or 'image').lower()
            image_ref   = data.get('image_ref_size') or {}
            coordinates = data.get('coordinates', {})  # may be absent (your case)

            if not pts_in:
                logging.warning(f"[{idx}] Skipping (no points): {os.path.basename(file_path)}")
                continue

            # Enforce viewer-friendly 'name' = <group> (from filename)
            base = os.path.basename(file_path)
            m = rx.match(base)
            if not m:
                logging.warning(f"[{idx}] Unexpected filename (needs <group>_<base>_polygons.json): {base}")
                continue
            group_name = m.group('group')
            name = group_name

            # Force root id to CURRENT root
            root_str = root_id_str

            # For each target file (every currently visible image in this root) …
            for image_fp in target_filepaths:
                # 1) effective basis
                th, tw = _effective_size(image_fp)
                # if unknown, borrow viewer basis later
                if not (th and tw):
                    th, tw = 0, 0

                # 2) map scene->image if needed (rare here, but keep parity with folder importer)
                pts_image = pts_in
                if coord_space == "scene" and hasattr(parent, "_map_points_scene_to_image"):
                    try:
                        size_hint = (th or 0, tw or 0, 1)
                        pts_image = parent._map_points_scene_to_image(image_fp, pts_in, size_hint, polygon_data=data)
                    except Exception:
                        pts_image = pts_in

                # 3) rescale from source ref -> target effective
                src_w = int(image_ref.get('w') or 0)
                src_h = int(image_ref.get('h') or 0)
                if src_w > 0 and src_h > 0 and (tw and th):
                    sx, sy = float(tw) / float(src_w), float(th) / float(src_h)
                    pts_image = [(float(x) * sx, float(y) * sy) for (x, y) in pts_image]
                else:
                    pts_image = [(float(x), float(y)) for (x, y) in pts_image]

                # 4) persist (IMAGE coords @ effective)
                parent.all_polygons.setdefault(group_name, {})
                parent.all_polygons[group_name][image_fp] = {
                    'points': pts_image,
                    'coord_space': 'image',
                    'image_ref_size': {'w': int(tw or 0), 'h': int(th or 0)},
                    'name': name,
                    'root': root_str,
                    'coordinates': coordinates,
                    'type': data.get('type', 'polygon'),
                }
                # Mark this polygon as dirty for incremental save
                if hasattr(parent, '_mark_polygon_dirty'):
                    parent._mark_polygon_dirty(group_name, image_fp)

                # 5) draw (map image coords to scene coords via pixmap)
                viewer = parent.get_viewer_by_filepath(image_fp)
                if viewer:
                    vh, vw = _viewer_hw(image_fp, (th, tw))
                    tw_eff, th_eff = (tw or vw, th or vh)  # adopt viewer if effective unknown

                    # de-dup by name on this viewer
                    if hasattr(viewer, "get_all_polygons"):
                        for item in list(viewer.get_all_polygons()):
                            if getattr(item, "name", None) == name:
                                try:
                                    viewer._scene.removeItem(item)
                                except Exception:
                                    pass

                    # Map image coords to scene coords through the pixmap
                    scene_pts = []
                    pixitem = getattr(viewer, '_image', None)
                    if pixitem is not None:
                        pixmap = pixitem.pixmap()
                        if pixmap and not pixmap.isNull():
                            pw, ph = max(1, pixmap.width()), max(1, pixmap.height())
                            img_h, img_w = th_eff or ph, tw_eff or pw
                            for (x, y) in pts_image:
                                x_pix = float(x) * (pw / float(img_w))
                                y_pix = float(y) * (ph / float(img_h))
                                scene_pt = pixitem.mapToScene(QtCore.QPointF(x_pix, y_pix))
                                scene_pts.append(scene_pt)
                    
                    if not scene_pts:
                        # Fallback: use image coords directly
                        scene_pts = [QtCore.QPointF(x, y) for (x, y) in pts_image]
                    
                    qpoly = QtGui.QPolygonF(scene_pts)
                    if hasattr(viewer, "add_polygon_to_scene"):
                        viewer.add_polygon_to_scene(qpoly, name)
                    else:
                        viewer.add_polygon(qpoly, name)

            imported += 1

        # Save & refresh UI using fast incremental save
        try:
            if hasattr(parent, 'save_incremental'):
                parent.save_incremental()
            else:
                parent.save_polygons_to_json(root_name=ms_root)
        except Exception as e:
            logging.error(f"Saving polygons failed: {e}")
        try:
            parent.update_polygon_manager()
        except Exception as e:
            logging.error(f"Updating Polygon Manager failed: {e}")

        QtWidgets.QMessageBox.information(self, "Import Complete",
            f"Imported {imported} polygon file(s) onto the current viewers of root '{ms_root}'.")

    def on_clear_all_polygons(self):
        # Perform clearing without confirmation pop-up
        self.clear_all_polygons_signal.emit()  # Emit the signal

    def on_item_double_clicked(self, item):
        group_name = item.data(QtCore.Qt.UserRole)
        print(f"[PolygonManager] Double-clicked group: {group_name}")
        if group_name:
            parent = self.parent()
            # Debug: show what filepaths exist for this group
            if hasattr(parent, 'all_polygons') and group_name in parent.all_polygons:
                fps = list(parent.all_polygons[group_name].keys())
                print(f"[PolygonManager] Group '{group_name}' has {len(fps)} filepath(s):")
                for fp in fps:
                    root_data = parent.all_polygons[group_name][fp].get('root', '?')
                    root_via_fp = parent.get_root_by_filepath(fp) if hasattr(parent, 'get_root_by_filepath') else '(no method)'
                    print(f"  -> {fp}  root_field={root_data}  root_via_filepath={root_via_fp}")
            
            # Switch to the root group associated with the group_name
            parent.switch_to_group(group_name)
            # Use proper zoom_to_groups for zoom and fit
            self.zoom_to_groups([group_name])
            # Select the polygons (without re-centering since zoom already did it)
            self._select_group_in_viewers(group_name, additive=False, center=False)
        else:
            pass  # Handle cases where group_name is invalid if necessary
