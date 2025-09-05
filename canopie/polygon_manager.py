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

# Import necessary classes from our package
from .machine_learning_manager import MachineLearningManager
from .loaders import ImageProcessor

from .utils import *

class PolygonManager(QtWidgets.QDialog):
    clear_all_polygons_signal = QtCore.pyqtSignal()  # Signal to clear all polygons across all images
    edit_group_signal = QtCore.pyqtSignal(str)       # Signal to initiate editing of a specific group

    def __init__(self, parent=None):
        super(PolygonManager, self).__init__(parent)
        self.setWindowTitle(f"{getattr(self.parent(), 'project_name', 'Project')}")
        self.layout = QtWidgets.QVBoxLayout(self)

        # === List ===
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.list_widget)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        # === Import buttons ===
        self.import_button = QtWidgets.QPushButton("Import Polygons")
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
        self.layout.addWidget(self.clear_all_button)
        self.clear_all_button.clicked.connect(self.on_clear_all_polygons)

        self.close_button = QtWidgets.QPushButton("Close")
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
        Persists polygons in IMAGE coords at the target's EFFECTIVE size
        (post-.ax) and draws via target->viewer->scene mapping.
        """
        import os, re, json, logging
        from PyQt5 import QtCore, QtGui
        logger = logging.getLogger(__name__)
        logger.info(f"Starting direct import for {len(file_paths)} corrected polygon file(s).")

        if not file_paths:
            logger.warning("No files provided to import_polygons_from_files().")
            return

        # ---------- parent helpers / wrappers ----------
        parent = self.parent()

        def _all_viewers_for_filepath(fp):
            out = []
            for w in getattr(parent, 'viewer_widgets', []) or []:
                idata = w.get('image_data')
                if idata is not None and getattr(idata, 'filepath', None) == fp:
                    out.append(w.get('viewer'))
            v = parent.get_viewer_by_filepath(fp)
            if v and v not in out:
                out.append(v)
            return [x for x in out if x]

        def _eff_size(fp):
            # Prefer parent's fast .ax-aware size if available
            try:
                if hasattr(parent, "_size_after_ax_fast_from_file"):
                    h, w = parent._size_after_ax_fast_from_file(fp)
                    if h and w:
                        return int(h), int(w)
            except Exception:
                pass
            # fallback to a live viewer image size
            vlist = _all_viewers_for_filepath(fp)
            if vlist:
                try:
                    img = getattr(vlist[0], "image_data", None)
                    if img is not None and getattr(img, "image", None) is not None:
                        h, w = img.image.shape[:2]
                        return int(h), int(w)
                except Exception:
                    pass
            return (None, None)

        def _viewer_basis_hw(viewer, fallback_hw):
            # Use parent's helper if present, else fallback to viewer image or provided
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

        def _scene_pts_for_viewer(viewer, img_pts):
            # If parent can map image->scene, use it; else assume viewer.add_polygon accepts image coords
            if hasattr(parent, "image_to_scene_coords"):
                return [ parent.image_to_scene_coords(viewer, QtCore.QPointF(x, y)) for (x, y) in img_pts ]
            return [ QtCore.QPointF(x, y) for (x, y) in img_pts ]

        try:
            id_to_root = parent.id_to_root
        except AttributeError:
            logger.critical("Internal error: 'id_to_root' attribute not found in MainWindow.")
            return

        all_image_filepaths = []
        for _gn, fps in getattr(parent, "multispectral_image_data_groups", {}).items():
            all_image_filepaths.extend(fps or [])
        for _gn, fps in getattr(parent, "thermal_rgb_image_data_groups", {}).items():
            all_image_filepaths.extend(fps or [])

        possible_exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']

        root_mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'root_mapping.json')
        try:
            with open(root_mapping_path, 'r', encoding='utf-8') as rm_file:
                root_mapping = json.load(rm_file)
        except Exception:
            root_mapping = {}

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

            # normalize root
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

            root_name = id_to_root.get(root_num) if root_num is not None else None
            if not root_name:
                fb_root_id, fb_root_name, _ = self._fallback_current_root_and_first_viewer()
                logger.warning(f"[{idx}] Unknown root '{root_raw}'. Falling back to current '{fb_root_name}' (id {fb_root_id}).")
                root_str = fb_root_id
                try:
                    root_num = int(fb_root_id)
                except Exception:
                    root_num = 0
                root_name = fb_root_name

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
            if not image_fp and root_mapping:
                # last-chance map by type
                b = base_name.lower()
                if '8b' in b:
                    img_type = 'RGB'
                elif 'ir' in b:
                    img_type = 'thermal'
                else:
                    img_type = self._infer_image_type_from_group(group_name, default='RGB')
                entry = root_mapping.get(str(root_num))
                if isinstance(entry, dict):
                    mapped = entry.get(img_type)
                    if isinstance(mapped, str) and mapped:
                        image_fp = _search_image(os.path.splitext(mapped)[0])

            if not image_fp:
                fb_root_id, fb_root_name, fb_fp = self._fallback_current_root_and_first_viewer()
                image_fp = fb_fp
                if not image_fp:
                    logger.warning(f"[{idx}] No resolvable image & no viewer; skipping '{base_name}'.")
                    continue
                root_str = fb_root_id  # align

            # ---------- scale to TARGET EFFECTIVE basis ----------
            th, tw = _eff_size(image_fp)
            # fall back to a viewer if needed
            if not (th and tw):
                vlist = _all_viewers_for_filepath(image_fp)
                if vlist:
                    try:
                        img = getattr(vlist[0], "image_data", None)
                        if img is not None and getattr(img, "image", None) is not None:
                            th, tw = img.image.shape[:2]
                    except Exception:
                        pass
            if not (th and tw):
                logger.warning(f"[{idx}] Could not determine effective size for '{image_fp}'. Using raw points.")
                th, tw = 0, 0

            # Convert incoming to IMAGE coords first if stored as scene
            pts_image = pts_in
            if coord_space == "scene" and hasattr(parent, "_map_points_scene_to_image"):
                try:
                    pts_image = parent._map_points_scene_to_image(image_fp, pts_in, (th, tw, 1), polygon_data=polygon_data)
                except Exception:
                    pts_image = pts_in  # best effort

            # If source had a different image_ref_size, scale to target effective
            src_w = int(image_ref.get('w') or 0)
            src_h = int(image_ref.get('h') or 0)
            if src_w > 0 and src_h > 0 and tw and th and (src_w != tw or src_h != th):
                sx, sy = float(tw) / float(src_w), float(th) / float(src_h)
                pts_image = [(float(x) * sx, float(y) * sy) for (x, y) in pts_image]
            else:
                # If no ref size given, assume already in target basis; just cast to float
                pts_image = [(float(x), float(y)) for (x, y) in pts_image]

            # ---------- persist (IMAGE coords @ target effective) ----------
            self.parent().all_polygons.setdefault(group_name, {})
            self.parent().all_polygons[group_name][image_fp] = {
                'points': pts_image,
                'coord_space': 'image',
                'image_ref_size': {'w': int(tw or 0), 'h': int(th or 0)},
                'name': name,
                'root': root_str,
                'coordinates': coordinates,
                'type': polygon_data.get('type', 'polygon'),
            }

            # ---------- draw on ALL viewers for that filepath ----------
            for viewer in _all_viewers_for_filepath(image_fp):
                vh, vw = _viewer_basis_hw(viewer, fallback_hw=(th, tw))
                pts_view = pts_image
                if all((tw, th, vw, vh)) and (tw != vw or th != vh):
                    sx_v, sy_v = float(vw) / float(tw), float(vh) / float(th)
                    pts_view = [(x * sx_v, y * sy_v) for (x, y) in pts_image]

                # remove existing same-name polygon (avoid dups)
                if hasattr(viewer, "get_all_polygons"):
                    for item in list(viewer.get_all_polygons()):
                        if getattr(item, "name", None) == name:
                            try:
                                viewer._scene.removeItem(item)
                            except Exception:
                                pass

                scene_pts = _scene_pts_for_viewer(viewer, pts_view)
                qpoly = QtGui.QPolygonF(scene_pts)
                if hasattr(viewer, "add_polygon_to_scene"):
                    viewer.add_polygon_to_scene(qpoly, name)
                else:
                    viewer.add_polygon(qpoly, name)

            imported += 1

        # persist + refresh
        try:
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

        # Run correction/fan-out
        try:
            self.run_correction_process()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Correction Error", f"Correction failed:\n{e}")
            return

        corrected_files = [
            os.path.join(self.corrected_dir, f)
            for f in os.listdir(self.corrected_dir)
            if f.lower().endswith("_polygons.json")
        ]
        if not corrected_files:
            QtWidgets.QMessageBox.warning(self, "Nothing to Import",
                                          "No corrected polygons were created in 'imported_polygons'.")
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
        Looks in <project_folder>/polygons (or ./polygons if no project folder).
        """
        import os, logging

        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete ALL Polygons",
            "This will permanently delete ALL saved polygon files for the entire project and remove overlays from every image. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        # Resolve polygons directory
        if getattr(self.parent(), "project_folder", None):
            polygons_dir = os.path.join(self.parent().project_folder, "polygons")
        else:
            polygons_dir = os.path.join(os.getcwd(), "polygons")

        deleted = 0
        if os.path.isdir(polygons_dir):
            for name in os.listdir(polygons_dir):
                # only wipe files created by this app’s convention
                if name.lower().endswith("_polygons.json"):
                    fp = os.path.join(polygons_dir, name)
                    try:
                        os.remove(fp)
                        deleted += 1
                    except Exception as e:
                        logging.error(f"Failed to delete polygon file {fp}: {e}")
        else:
            logging.info(f"Polygons directory not found: {polygons_dir}")

        # Clear overlays from all viewers
        try:
            for widget in getattr(self.parent(), "viewer_widgets", []):
                viewer = widget.get("viewer") if isinstance(widget, dict) else None
                if viewer and hasattr(viewer, "clear_polygons"):
                    viewer.clear_polygons()
        except Exception as e:
            logging.debug(f"Clearing viewer overlays failed: {e}")

        # Clear in-memory structures
        try:
            if hasattr(self.parent(), "all_polygons"):
                self.parent().all_polygons.clear()
        except Exception as e:
            logging.debug(f"Clearing in-memory polygons failed: {e}")

        # Refresh manager UI (and any other dependent views)
        try:
            if hasattr(self.parent(), "update_polygon_manager"):
                self.parent().update_polygon_manager()
            else:
                self.set_polygons({})
        except Exception as e:
            logging.debug(f"Refreshing polygon manager failed: {e}")

        QtWidgets.QMessageBox.information(
            self, "Delete ALL Polygons", f"Deleted {deleted} file(s). All polygon overlays cleared."
        )
                # Quick-save the project state (no recompute)
        try:
            if hasattr(self.parent(), "save_project_quick"):
                self.parent().save_project_quick(skip_recompute=True)
        except Exception as e:
            logging.debug(f"Quick-save after delete-all failed: {e}")
  

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

    def run_correction_process(self):
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
        """
        import os, re, json, logging

        JSONS_DIR = self.jsons_dir
        POLYGONS_DIR = self.polygons_dir
        CORRECTED_DIR = self.corrected_dir
        os.makedirs(CORRECTED_DIR, exist_ok=True)

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

            wrote_here = 0
            for dest_base in uniq_dest_bases:
                out_name = f"{group}_{dest_base}_polygons.json"
                out_path = os.path.join(CORRECTED_DIR, out_name)
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(poly, f, indent=4)
                    logging.info(f"[CorrectPolygons] Wrote: {out_path}")
                    wrote_here += 1
                except Exception as e:
                    logging.error(f"[CorrectPolygons] Failed to write {out_path}: {e}")
                    skipped += 1

            if wrote_here > 0:
                processed += 1
                fanout_count += (wrote_here - 1)  # extra beyond the 'closest_image' itself

        logging.info(f"[CorrectPolygons] Done. Results files processed: {processed}, "
                     f"skipped: {skipped}, missing source: {missing_src}, "
                     f"extra fan-out files created: {fanout_count}")

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
        self.list_widget.clear()
        for group_name in sorted(polygon_groups.keys(), key=self.natural_sort_key):
            display_name = group_name or "Unnamed Group"
            item = QtWidgets.QListWidgetItem(display_name)
            # Store the group name in UserRole
            item.setData(QtCore.Qt.UserRole, group_name)
            self.list_widget.addItem(item)
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

    def update_selection_based_on_root(self):
        """
        Automatically selects polygons associated with the current root.
        """
        if not self.current_root:
            return
        self.list_widget.clearSelection()
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            group_name = item.data(QtCore.Qt.UserRole)
            # Check if the group has any polygons in the current root
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
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")
        copy_action = menu.addAction("Copy")
        move_action = menu.addAction("Move")
        copy_to_viewers_action = menu.addAction("Copy to Viewers (This Root)")
        delete_all_action = menu.addAction("Delete All Polygons")

        # NEW: Zoom action
        menu.addSeparator()
        zoom_action = menu.addAction("Zoom to Polygon Points")

        menu.addSeparator()
        export_csv_action = menu.addAction("Export CSV (ML)")
        thumbs_action     = menu.addAction("Generate Thumbnails (ML)")
        masks_action      = menu.addAction("Generate Segmentation Masks (ML)")

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
            # NEW: perform zoom
            self.zoom_to_groups(target_groups)
        elif action == export_csv_action:
            self._mlm_invoke(target_groups, kind="csv")
        elif action == thumbs_action:
            self._mlm_invoke(target_groups, kind="thumbs")
        elif action == masks_action:
            self._mlm_invoke(target_groups, kind="masks")

    # ---------------- NEW: Zoom helpers ----------------

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

    def _scene_rect_for_group_on_viewer(self, viewer_widget, viewer, group_name):
        """
        Compute a QRectF in scene coordinates covering all items/points for group_name
        on the specified viewer. Falls back to stored points if items are not present.
        """
        # 1) Try live scene items
        combined_rect = None
        if hasattr(viewer, "get_all_polygons"):
            for it in viewer.get_all_polygons():
                if getattr(it, "name", None) == group_name:
                    try:
                        r = it.mapToScene(it.boundingRect()).boundingRect()
                    except Exception:
                        # fallback if mapToScene fails
                        r = it.boundingRect()
                    combined_rect = r if combined_rect is None else combined_rect.united(r)

        # 2) Fall back to stored points for this viewer's image (if available)
        if combined_rect is None:
            # try to get current image filepath from the viewer widget dict
            idata = viewer_widget.get("image_data") if isinstance(viewer_widget, dict) else None
            fp = getattr(idata, "filepath", None) if idata is not None else None
            if fp:
                payload = (self.parent().all_polygons.get(group_name, {}) or {}).get(fp, {})
                pts = payload.get("points") or []
                if pts:
                    # Map to scene if parent provides a mapper, else treat as scene coords
                    if hasattr(self.parent(), "image_to_scene_coords"):
                        pts_scene = [ self.parent().image_to_scene_coords(viewer, QtCore.QPointF(x, y)) for (x, y) in pts ]
                    else:
                        pts_scene = [ QtCore.QPointF(x, y) for (x, y) in pts ]
                    poly = QtGui.QPolygonF(pts_scene)
                    combined_rect = poly.boundingRect()

        # 3) pad a little so the view is tight but comfortable
        if combined_rect and combined_rect.isValid():
            pad = 30.0
            combined_rect = combined_rect.adjusted(-pad, -pad, pad, pad)
        return combined_rect

    def zoom_to_groups(self, groups):
        """
        Zoom every open viewer to tightly frame the polygons/points belonging to the given groups.
        Uses fitInView, then applies a small extra zoom for a closer look.
        """
        if not groups:
            return
        for vw_widget, viewer in self._iter_viewers():
            combined = None
            for g in groups:
                r = self._scene_rect_for_group_on_viewer(vw_widget, viewer, g)
                if r and r.isValid():
                    combined = r if combined is None else combined.united(r)

            if combined and combined.isValid() and combined.width() > 0 and combined.height() > 0:
                # Prefer fitInView if available (typical QGraphicsView method)
                try:
                    viewer.fitInView(combined, QtCore.Qt.KeepAspectRatio)
                    # Tiny additional zoom-in (20%) for a closer look
                    try:
                        viewer.scale(1.2, 1.2)
                    except Exception:
                        pass
                except Exception:
                    # Fallback: ensureVisible
                    try:
                        viewer.ensureVisible(combined, 20, 20)
                    except Exception:
                        pass
                try:
                    viewer.centerOn(combined.center())
                    viewer.setFocus()
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
        try:
            if hasattr(self.parent(), "save_project_quick"):
                self.parent().save_project_quick(skip_recompute=True)
        except Exception as e:
            logging.debug(f"Quick-save after group delete-all failed: {e}")


    def copy_selected_polygons_to_current_root_viewers(self, groups=None):
        """
        For each selected group, find any polygon instance *within the current root*
        and copy that polygon to **every** viewer (image) in the current root.
        If a group has no instance in the current root, falls back to any instance of that group.
        """
        from PyQt5 import QtGui, QtCore
        import os

        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups or not self.current_root_filepaths:
            return

        for group_name in groups:
            group_map = self.parent().all_polygons.get(group_name, {})
            if not group_map:
                continue

            # 1) Try to find a source polygon from the current root
            source_points = None
            source_payload = None
            for fp, payload in group_map.items():
                if fp in self.current_root_filepaths:
                    pts = payload.get('points')
                    if pts:
                        source_points = pts
                        source_payload = payload
                        break

            # 2) Fallback: any polygon for the group
            if source_points is None:
                for _fp, payload in group_map.items():
                    pts = payload.get('points')
                    if pts:
                        source_points = pts
                        source_payload = payload
                        break

            if source_points is None:
                continue  # nothing usable

            # Build a canonical payload copy
            to_store = {
                'points': source_points,
                'name': source_payload.get('name', group_name),
            }
            # preserve optional fields if present
            if 'root' in source_payload:
                to_store['root'] = source_payload['root']
            if 'coordinates' in source_payload:
                to_store['coordinates'] = source_payload['coordinates']

            # 3) Copy to all viewers (images) in the current root
            for target_fp in list(self.current_root_filepaths):
                # Write/overwrite in memory
                if group_name not in self.parent().all_polygons:
                    self.parent().all_polygons[group_name] = {}
                self.parent().all_polygons[group_name][target_fp] = dict(to_store)

                # Update viewer: remove existing same-name polygon (avoid duplicates), then add fresh
                viewer = self.parent().get_viewer_by_filepath(target_fp)
                if viewer:
                    if hasattr(viewer, "get_all_polygons"):
                        for item in list(viewer.get_all_polygons()):
                            if getattr(item, "name", None) == group_name:
                                try:
                                    viewer._scene.removeItem(item)
                                except Exception:
                                    pass
                    polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in source_points])
                    viewer.add_polygon(polygon, group_name)

        # Persist & refresh
        if hasattr(self.parent(), "save_polygons_to_json"):
            self.parent().save_polygons_to_json()
        if hasattr(self.parent(), "update_polygon_manager"):
            self.parent().update_polygon_manager()

    def delete_all_polygons_for_group(self, group_name: str):
        """
        Deletes EVERY polygon associated with the given group name across the whole project:
          - removes overlays from all ImageViewers
          - deletes *_polygons.json files for that group
          - clears from in-memory self.parent().all_polygons[group_name]
          - refreshes the manager UI
        No confirmation popups here by design.
        """
        import os, logging

        # Resolve polygons directory
        if getattr(self.parent(), "project_folder", None):
            polygons_dir = os.path.join(self.parent().project_folder, "polygons")
        else:
            polygons_dir = os.path.join(os.getcwd(), "polygons")

        # Nothing to do if group absent
        group_map = self.parent().all_polygons.get(group_name)
        if not group_map:
            logging.info(f"[PolygonManager] No polygons stored for group '{group_name}'. Nothing to delete.")
            return

        # 1) Remove overlays from all viewers for this group, on every filepath
        for filepath in list(group_map.keys()):
            viewer = self.parent().get_viewer_by_filepath(filepath)
            if viewer and hasattr(viewer, "get_all_polygons"):
                for item in list(viewer.get_all_polygons()):
                    if getattr(item, "name", None) == group_name:
                        try:
                            viewer._scene.removeItem(item)
                        except Exception:
                            pass

        # 2) Delete JSON files for this group across ALL images
        deleted_files = 0
        if os.path.isdir(polygons_dir):
            for filepath in list(group_map.keys()):
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                polygon_filename = f"{group_name}_{base_filename}_polygons.json"
                polygon_filepath = os.path.join(polygons_dir, polygon_filename)
                if os.path.exists(polygon_filepath):
                    try:
                        os.remove(polygon_filepath)
                        deleted_files += 1
                    except Exception as e:
                        logging.error(f"[PolygonManager] Failed to delete polygon file {polygon_filepath}: {e}")
        else:
            logging.debug(f"[PolygonManager] Polygons directory not found: {polygons_dir}")

        # 3) Clear from in-memory map
        try:
            del self.parent().all_polygons[group_name]
        except Exception:
            self.parent().all_polygons.pop(group_name, None)

        # 4) Refresh UI 
        try:
            if hasattr(self.parent(), "save_polygons_to_json"):
                self.parent().save_polygons_to_json()
        except Exception as e:
            logging.debug(f"[PolygonManager] save_polygons_to_json failed after delete-all: {e}")

        try:
            if hasattr(self.parent(), "update_polygon_manager"):
                self.parent().update_polygon_manager()
            else:
                self.set_polygons(self.parent().all_polygons)
        except Exception as e:
            logging.debug(f"[PolygonManager] update_polygon_manager failed after delete-all: {e}")

        logging.info(f"[PolygonManager] Deleted all polygons for group '{group_name}'. Files removed: {deleted_files}")

    def delete_selected_polygons(self, groups=None):
        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return  # Do nothing if no selection

        # Determine the polygons to delete
        groups_to_delete = set(groups)

        # Determine the polygons directory
        polygons_dir = os.path.join(
            self.parent().project_folder, 'polygons') if self.parent().project_folder else os.path.join(os.getcwd(), 'polygons')

        for group_name in groups_to_delete:
            if group_name not in self.parent().all_polygons:
                continue
            # Iterate over all filepaths associated with the group
            for filepath in list(self.parent().all_polygons[group_name].keys()):
                if filepath not in self.current_root_filepaths:
                    continue  # Skip if not in current root

                viewer = self.parent().get_viewer_by_filepath(filepath)
                if viewer:
                    # Remove the polygon from the viewer
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
            if not self.parent().all_polygons.get(group_name):
                self.parent().all_polygons.pop(group_name, None)

        # Refresh UI
        self.parent().update_polygon_manager()

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
        Copy selected polygon groups to a target root (chosen by user).
        Preserves 'root' and 'coordinates' if present in the source payload.
        Works with multi-select.
        """
        from PyQt5 import QtGui, QtCore
        import os

        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return  # Do nothing if no selection

        # Select target root from existing roots
        target_root, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Target Root", "Choose the target root to copy polygons to:",
            self.parent().root_names, 0, False
        )
        if not ok or not target_root:
            return

        target_filepaths = self.parent().image_data_groups.get(target_root) or []
        if not target_filepaths:
            return  # No target images, skip

        for group_name in groups:
            group_polygons = self.parent().all_polygons.get(group_name, {})
            if not group_polygons:
                continue

            # Take any source payload (first available)
            try:
                sample_payload = next(iter(group_polygons.values()))
            except StopIteration:
                continue

            points = sample_payload.get('points')
            if not points:
                continue

            to_store = {
                'points': points,
                'name': group_name,
            }
            if 'root' in sample_payload:
                to_store['root'] = sample_payload['root']
            if 'coordinates' in sample_payload:
                to_store['coordinates'] = sample_payload['coordinates']

            for target_filepath in target_filepaths:
                # Clone in memory
                if group_name not in self.parent().all_polygons:
                    self.parent().all_polygons[group_name] = {}
                self.parent().all_polygons[group_name][target_filepath] = dict(to_store)

                # Update ImageViewer (replace existing with same name)
                viewer = self.parent().get_viewer_by_filepath(target_filepath)
                if viewer:
                    if hasattr(viewer, "get_all_polygons"):
                        for item in list(viewer.get_all_polygons()):
                            if getattr(item, "name", None) == group_name:
                                try:
                                    viewer._scene.removeItem(item)
                                except Exception:
                                    pass
                    polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
                    viewer.add_polygon(polygon, group_name)

        # Save changes
        if hasattr(self.parent(), "save_polygons_to_json"):
            self.parent().save_polygons_to_json()
        if hasattr(self.parent(), "update_polygon_manager"):
            self.parent().update_polygon_manager()

    def move_selected_polygons(self, groups=None):
        """
        Move selected polygon groups from the current root to another root.
        Respects multi-select.
        """
        from PyQt5 import QtGui, QtCore
        import os

        if groups is None:
            groups = self.get_selected_polygon_groups()
        if not groups:
            return  # Do nothing if no selection

        # Select target root from existing roots
        target_root, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Target Root", "Choose the target root to move polygons to:",
            self.parent().root_names, 0, False
        )
        if not ok or not target_root:
            return

        polygons_dir = os.path.join(
            self.parent().project_folder, 'polygons') if self.parent().project_folder else os.path.join(os.getcwd(), 'polygons')

        target_filepaths = self.parent().image_data_groups.get(target_root) or []

        for group_name in groups:
            group_polygons = self.parent().all_polygons.get(group_name, {}).copy()
            if not group_polygons:
                continue

            for filepath, polygon_data in list(group_polygons.items()):
                if filepath not in self.current_root_filepaths:
                    continue  # Skip if not in current root

                # Remove the polygon from the source viewer
                viewer = self.parent().get_viewer_by_filepath(filepath)
                if viewer and hasattr(viewer, "get_all_polygons"):
                    for item in list(viewer.get_all_polygons()):
                        if getattr(item, "name", None) == group_name:
                            try:
                                viewer._scene.removeItem(item)
                            except Exception:
                                pass
                            break  # Assuming one polygon per group per image

                # Delete the source JSON file
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                polygon_filename = f"{group_name}_{base_filename}_polygons.json"
                polygon_filepath = os.path.join(polygons_dir, polygon_filename)
                if os.path.exists(polygon_filepath):
                    try:
                        os.remove(polygon_filepath)
                        print(f"Deleted source polygon file: {polygon_filepath}")
                    except Exception as e:
                        print(f"Failed to delete source polygon file {polygon_filepath}: {e}")

                # Remove the polygon from the in-memory data structure
                self.parent().all_polygons[group_name].pop(filepath, None)

                # Add the polygon to each target image
                for target_filepath in target_filepaths:
               
                    if group_name not in self.parent().all_polygons:
                        self.parent().all_polygons[group_name] = {}
                    self.parent().all_polygons[group_name][target_filepath] = {
                        'points': polygon_data.get('points', []),
                        'name': group_name,
                        **({'root': polygon_data['root']} if 'root' in polygon_data else {}),
                        **({'coordinates': polygon_data['coordinates']} if 'coordinates' in polygon_data else {}),
                    }

                    target_viewer = self.parent().get_viewer_by_filepath(target_filepath)
                    if target_viewer:
                        if hasattr(target_viewer, "get_all_polygons"):
                            for item in list(target_viewer.get_all_polygons()):
                                if getattr(item, "name", None) == group_name:
                                    try:
                                        target_viewer._scene.removeItem(item)
                                    except Exception:
                                        pass
                        pts = polygon_data.get('points', [])
                        polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts])
                        target_viewer.add_polygon(polygon, group_name)

        # Save changes
        if hasattr(self.parent(), "save_polygons_to_json"):
            self.parent().save_polygons_to_json()
        if hasattr(self.parent(), "update_polygon_manager"):
            self.parent().update_polygon_manager()

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
        Imports polygons from user-picked JSON files and adds them to the viewers.
        Fallbacks:
          - Unknown root -> use current root (and write that root into payload)
          - Unresolvable image -> first viewer’s image
          - Polygon 'name' forced to the group name (from filename) for viewer compatibility
        """
        import os, re, json, logging
        logger = logging.getLogger(__name__)
        logger.info("Starting polygon import process.")

        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import Polygons", os.path.expanduser("~"), "JSON Files (*.json)"
        )
        logger.info(f"Selected {len(file_paths)} JSON file(s) for import.")
        if not file_paths:
            return

        # id -> root_name map
        try:
            id_to_root = self.parent().id_to_root
        except AttributeError:
            logger.critical("Internal error: 'id_to_root' attribute not found in MainWindow.")
            return

        # all images across groups
        all_image_filepaths = []
        for _group_name, filepaths in getattr(self.parent(), "multispectral_image_data_groups", {}).items():
            all_image_filepaths.extend(filepaths or [])
        for _group_name, filepaths in getattr(self.parent(), "thermal_rgb_image_data_groups", {}).items():
            all_image_filepaths.extend(filepaths or [])
        possible_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']

        # optional root_mapping
        root_mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'root_mapping.json')
        try:
            with open(root_mapping_path, 'r', encoding='utf-8') as rm_file:
                root_mapping = json.load(rm_file)
        except Exception:
            root_mapping = {}

        # canonical base -> [paths]
        canon_to_paths = {}
        for fp in all_image_filepaths:
            base = os.path.splitext(os.path.basename(fp))[0].lower()
            c = self._canonicalize_base(base)
            canon_to_paths.setdefault(c, []).append(fp)

        def search_image(filename_no_ext: str):
            """Try exact basename (with known extensions), then canonicalized match."""
            fn_lower = (filename_no_ext or "").lower()
            # exact
            for ext in possible_extensions:
                expected_filename = fn_lower + ext
                for fp in all_image_filepaths:
                    if os.path.basename(fp).lower() == expected_filename:
                        return os.path.abspath(fp)
            # canonical
            canon_q = self._canonicalize_base(fn_lower)
            candidates = canon_to_paths.get(canon_q, [])
            if candidates:
                for fp in candidates:
                    if fn_lower in os.path.splitext(os.path.basename(fp))[0].lower():
                        return os.path.abspath(fp)
                return os.path.abspath(candidates[0])
            return None

        imported = 0
        for idx, file_path in enumerate(file_paths, start=1):
            logger.info(f"Processing file {idx}/{len(file_paths)}: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    polygon_data = json.load(f)
            except Exception as e:
                logger.error(f"JSON read error '{file_path}': {e}")
                continue

            points = polygon_data.get('points', [])
            json_name = (polygon_data.get('name') or '').strip()
            root_number_raw = polygon_data.get('root', '')
            coordinates = polygon_data.get('coordinates', {})

            if not points:
                logger.warning(f"Skipping invalid polygon (missing points): {file_path}")
                continue

            # Normalize/parse root
            if isinstance(root_number_raw, int):
                root_number_str = str(root_number_raw)
            elif isinstance(root_number_raw, str):
                root_number_str = root_number_raw.strip()
            else:
                root_number_str = ""

            try:
                root_number = int(root_number_str)
            except Exception:
                root_number = None

            root_name = id_to_root.get(root_number) if root_number is not None else None
            if not root_name:
                fb_root_id, fb_root_name, _fb_fp = self._fallback_current_root_and_first_viewer()
                logger.warning(f"No root_name for root '{root_number_raw}' in {file_path}. "
                               f"Falling back to current root '{fb_root_name}' (id {fb_root_id}).")
                root_number_str = fb_root_id
                try:
                    root_number = int(fb_root_id)
                except Exception:
                    root_number = 0
                root_name = fb_root_name

            # "<group>_<imagebase>_polygons.json"
            base_filename = os.path.basename(file_path)
            match = re.match(r'^(?P<group>.*?)_(?P<imgbase>.*?)_polygons\.json$', base_filename, re.IGNORECASE)
            if not match:
                logger.warning(f"Filename '{base_filename}' does not match expected pattern. Skipping.")
                continue

            group_name = match.group('group')
            imgbase = match.group('imgbase')

            # Force the polygon name to group name (viewer expects this)
            name = group_name
            if json_name and json_name != group_name:
                logger.debug(f"Overriding polygon 'name' from '{json_name}' -> '{group_name}' for viewer consistency.")

            # Resolve image (as-is, then try adding/removing root prefix, then mapping)
            image_filepath = search_image(imgbase)

            if not image_filepath:
                if imagebase := imgbase:
                    # try removing root prefix
                    if root_name and imagebase.startswith(root_name + "_"):
                        cand = imagebase[len(root_name) + 1:]
                        image_filepath = search_image(cand)
                    # try adding root prefix
                    if not image_filepath and root_name:
                        image_filepath = search_image(f"{root_name}_{imagebase}")

            if not image_filepath and root_mapping:
                base_lower = base_filename.lower()
                if '8b' in base_lower:
                    image_type = 'RGB'
                elif 'ir' in base_lower:
                    image_type = 'thermal'
                else:
                    image_type = self._infer_image_type_from_group(group_name, default='RGB')

                mapping_entry = root_mapping.get(root_number_str)
                if isinstance(mapping_entry, dict):
                    mapped_filename = mapping_entry.get(image_type)
                    if isinstance(mapped_filename, str) and mapped_filename:
                        image_filepath = search_image(os.path.splitext(mapped_filename)[0])

            # Final fallback to first viewer
            if not image_filepath:
                fb_root_id, fb_root_name, fb_target_fp = self._fallback_current_root_and_first_viewer()
                image_filepath = fb_target_fp
                if not image_filepath:
                    logger.warning(f"Associated image for '{base_filename}' not found and no viewer available. Skipping.")
                    continue
                # align root to fallback
                root_number_str = fb_root_id

            # Store
            if group_name not in self.parent().all_polygons:
                self.parent().all_polygons[group_name] = {}
            self.parent().all_polygons[group_name][image_filepath] = {
                'points': points,
                'name': name,                 # forced group name
                'root': root_number_str,      # corrected if needed
                'coordinates': coordinates
            }
            logger.info(f"Added polygon '{name}' to group '{group_name}' for image '{image_filepath}'.")

            # Draw on viewer
            viewer = self.parent().get_viewer_by_filepath(image_filepath)
            if viewer:
                from PyQt5 import QtGui, QtCore
                polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
                # remove any existing polygon with same name to avoid duplicates
                if hasattr(viewer, "get_all_polygons"):
                    for item in list(viewer.get_all_polygons()):
                        if getattr(item, "name", None) == name:
                            try:
                                viewer._scene.removeItem(item)
                            except Exception:
                                pass
                            break
                viewer.add_polygon(polygon, name)
            else:
                logger.warning(f"Viewer for image '{image_filepath}' not found. Skipping overlay.")

            imported += 1

        # Save & refresh
        logger.info("Saving all polygons to JSON files.")
        try:
            self.parent().save_polygons_to_json()
        except Exception as e:
            logger.exception(f"Failed to save polygons to JSON: {e}")

        logger.info("Updating the Polygon Manager UI.")
        try:
            self.parent().update_polygon_manager()
        except Exception as e:
            logger.exception(f"Failed to update Polygon Manager UI: {e}")

        logger.info("Completed importing polygons from selected files.")


    def on_clear_all_polygons(self):
        # Perform clearing without confirmation pop-up
        self.clear_all_polygons_signal.emit()  # Emit the signal

    def on_item_double_clicked(self, item):
        group_name = item.data(QtCore.Qt.UserRole)
        if group_name:
            # Switch to the root group associated with the group_name
            self.parent().switch_to_group(group_name)
            # Then, select and center the polygons
            for widget in self.parent().viewer_widgets:
                viewer = widget['viewer']
                for polygon_item in viewer.get_all_polygons():
                    if polygon_item.name == group_name:
                        polygon_item.setSelected(True)
                        viewer.centerOn(polygon_item)
                        viewer.setFocus()
        else:
            pass  # Handle cases where group_name is invalid if necessary

