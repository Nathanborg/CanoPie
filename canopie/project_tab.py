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

from .image_viewer import ImageViewer, EditablePolygonItem, EditablePointItem
from .image_editor_dialog import ImageEditorDialog
from .polygon_manager import PolygonManager
from .machine_learning_manager import MachineLearningManager, AnalysisOptionsDialog, RootOffsetDialog
from .image_data import ImageData
from .loaders import _LoaderSignals, _ImageLoadRunnable, ImageProcessor, ImageLoaderWorker
from .utils import *

class ProjectTab(QtWidgets.QWidget):
    def __init__(self, project_name, tab_widget, parent=None, exiftool_path=None):
        super().__init__(parent)
        self.project_name = project_name
        self.tab_widget = tab_widget
        self.setup_logging()
        self.init_data_structures()
        self.init_ui()
        self.exiftool_path = exiftool_path  # <-- store it
        self.current_folder_path = None
        self.thermal_rgb_folder_path = None
        self._load_busy = False
        self._load_token = 0
        self._load_stop_event = None
        self._pending_root = None
        self.setAcceptDrops(True)

        


        if not self.exiftool_path:
            settings = QSettings("YourOrg", "YourApp")
            saved = settings.value("exiftool_path", type=str)
            if saved:
                self.exiftool_path = saved
                logging.info(f"[{self.project_name}] exiftool path restored from settings: {saved}")


    def dragEnterEvent(self, event):
        import os
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore(); return
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        for u in md.urls():
            p = u.toLocalFile()
            if os.path.isdir(p) or os.path.splitext(p.lower())[1] in exts:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        import os
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if not urls:
            event.ignore(); return

        first = urls[0].toLocalFile()
        folder = first if os.path.isdir(first) else os.path.dirname(first)

        # no dialogs: always load with batch_size=10 and replace current project
        self.open_rgb_folder_from_path(folder, batch_size=10)
        event.acceptProposedAction()

    def _set_nav_enabled(self, enabled: bool):
        for btn_name in ("btn_next", "btn_prev", "btn_first", "btn_last"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                try:
                    btn.setEnabled(enabled)
                    # extra safety: avoid press-and-hold auto-fire
                    if hasattr(btn, "setAutoRepeat"):
                        btn.setAutoRepeat(False)
                except Exception:
                    pass

    def _imagedata_or_fallback(self, filepath):
        """
        Prefer ImageData(...) for normal images, but for TIFFs that are actually stacks
        (multi-page or >3 bands), force a tifffile load so they open as true stacks.

        Returns an object with .filepath and .image (like ImageData).
        """
        import os, logging, numpy as np

        # Lightweight shim that looks like ImageData
        class _Lite:
            __slots__ = ("filepath", "image")
            def __init__(self, fp, im):
                self.filepath = fp
                self.image = im

        def _tifffile_is_stack(path):
            """Quick metadata probe: does this TIFF represent a stack (>3 bands or >3 pages)?"""
            try:
                import tifffile as tiff
                with tiff.TiffFile(path) as tf:
                    n_pages = len(tf.pages)
                    # Prefer series info if available
                    series = tf.series[0] if tf.series else None
                    axes   = getattr(series, "axes", "") or ""
                    shape  = getattr(series, "shape", ()) or ()
                    spp    = getattr(tf.pages[0], "samplesperpixel", 1) or 1

                    # Channel/band count heuristic
                    band_count = spp
                    if axes and shape:
                        # If 'C' or 'S' axis present, use that for band count
                        if 'C' in axes:
                            band_count = int(shape[axes.index('C')])
                        elif 'S' in axes:
                            band_count = int(shape[axes.index('S')])
                        elif len(shape) == 3:
                            # Fallbacks: infer which dim is channels when axes unknown
                            if shape[0] <= 32 and shape[1] >= 32 and shape[2] >= 32:
                                band_count = int(shape[0])
                            elif shape[2] <= 32:
                                band_count = int(shape[2])

                    # Treat as stack if either: many bands or many uniform pages
                    return (band_count > 3) or (n_pages > 3)
            except Exception as e:
                logging.debug(f"TIFF preflight failed on '{path}': {e}")
                return False

        def _tifffile_read_as_HWC(path):
            """Read with tifffile and coerce to HxWxC when it looks like CxHxW."""
            import tifffile as tiff
            arr = tiff.imread(path)
            arr = np.squeeze(arr)

            # If it's channel-first (C,H,W) with small C, move channels last
            if arr.ndim == 3 and arr.shape[0] <= 32 and arr.shape[1] >= 32 and arr.shape[2] >= 32:
                arr = np.moveaxis(arr, 0, -1)
            return arr

        ext = os.path.splitext(filepath)[1].lower()

        # --- 1) TIFF preflight: if it's a real stack, go straight to tifffile ---
        if ext in (".tif", ".tiff") and _tifffile_is_stack(filepath):
            try:
                arr = _tifffile_read_as_HWC(filepath)
                if arr is not None and getattr(arr, "size", 0) > 0:
                    self._last_loader = "tifffile-preflight"
                    return _Lite(filepath, arr)
            except Exception as e:
                logging.warning(f"tifffile stack load failed for '{filepath}', will try ImageData: {e}")

        # --- 2) Try ImageData (cv/PIL path) ---
        try:
            imgd = ImageData(filepath, mode=getattr(self, "mode", None))
            img = getattr(imgd, "image", None)

            if img is not None and getattr(img, "size", 0) > 0:
                # If a TIFF slipped through as 4-ch via cv (e.g., BGRA) but metadata says stack, override.
                if ext in (".tif", ".tiff") and isinstance(img, np.ndarray):
                    try:
                        if img.ndim == 3 and img.shape[2] in (4, 5, 6, 7, 8):  # cv read with >3 channels
                            # Confirm with metadata; if it's truly a stack, reload via tifffile.
                            if _tifffile_is_stack(filepath):
                                arr = _tifffile_read_as_HWC(filepath)
                                if arr is not None and getattr(arr, "size", 0) > 0:
                                    self._last_loader = "tifffile-override-cv"
                                    return _Lite(filepath, arr)
                    except Exception as e:
                        logging.debug(f"cv override check failed for '{filepath}': {e}")

                # Normal path: keep ImageData result
                self._last_loader = "imagedata"
                return imgd

            raise ValueError("ImageData returned empty image")

        except Exception:
            # --- 3) Final fallback (existing raw loader; should already prefer tifffile for stacks) ---
            arr = self._load_raw_image(filepath)
            if arr is None or getattr(arr, "size", 0) == 0:
                raise
            self._last_loader = "fallback"
            return _Lite(filepath, arr)


    def _ax_op_order(self, ax: dict):
        """
        Returns the list of ops to apply, in order, limited to {'rotate','crop','resize'}.
        Falls back to ['rotate','crop','resize'] if not provided.
        Accepts variations like 'rotation', 'res', etc.
        """
        if not isinstance(ax, dict):
            return ["rotate", "crop", "resize"]
        raw = ax.get("op_order") or ax.get("ops") or ["rotate", "crop", "resize"]
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for s in (raw or []):
            t = str(s).lower()
            if "rot" in t:   out.append("rotate")
            elif "crop" in t: out.append("crop")
            elif "siz" in t or "res" in t: out.append("resize")
        return [op for op in out if op in ("rotate", "crop", "resize")] or ["rotate", "crop", "resize"]

      
    def _get_exif_data_with_optional_path(self, files):
        """
        Return {abs_normcase_path: {tag: value}}.
        Try helper w/ exiftool_path, then helper w/o, then subprocess with self.exiftool_path,
        and finally prompt once for exiftool.exe if still missing.
        """
        import os, logging

        if not files:
            return {}

        # Dedup + normalize inputs first
        files = [os.path.abspath(f) for f in files]
        files = list(dict.fromkeys(files))  # preserve order, drop dupes

        def _normkey(p):  # single canonical form used everywhere
            return os.path.normcase(os.path.abspath(p))

        # 1) helper with kwarg
        try:
            d = get_exif_data_exiftool_multiple(files, exiftool_path=self.exiftool_path)
            if d:
                return {_normkey(k): v for k, v in d.items()}
        except TypeError:
            logging.debug("Helper doesn’t accept exiftool_path; trying without it.")
        except Exception as e:
            logging.warning("Helper with exiftool_path failed: %s", e)

        # 2) helper without kwarg
        try:
            d = get_exif_data_exiftool_multiple(files)
            if d:
                return {_normkey(k): v for k, v in d.items()}
        except Exception as e:
            logging.warning("Helper without exiftool_path failed: %s", e)

        # 3) subprocess fallback (prefer self.exiftool_path)
        try:
            d = self._extract_exif_via_subprocess(files, exe_path=self.exiftool_path)
            if d:
                return {_normkey(k): v for k, v in d.items()}
        except FileNotFoundError:
            # Ask once
            from PyQt5 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Locate exiftool.exe", "", "Executable (exiftool*.exe);;All Files (*)"
            )
            if path:
                self.set_exiftool_path(path)
                try:
                    d = self._extract_exif_via_subprocess(files, exe_path=path)
                    if d:
                        return {_normkey(k): v for k, v in d.items()}
                except Exception as e:
                    logging.exception("Direct exiftool call failed after picking path: %s", e)
        except Exception as e:
            logging.exception("Subprocess EXIF fallback failed: %s", e)

        logging.error("EXIF extraction failed; returning empty tags for %d files.", len(files))
        return {_normkey(p): {} for p in files}

    def _extract_exif_via_subprocess(self, files, exe_path=None):
        """
        Run exiftool directly with a given executable path.
        - Chunks input to avoid Windows command-line length limits.
        - Returns a merged dict keyed by absolute file path.
        """
        import os, json, shutil, subprocess

        if not files:
            return {}

        exe = exe_path or self.exiftool_path or "exiftool"
        # Accept a directory pointing to the exe folder
        if os.path.isdir(exe):
            exe = os.path.join(exe, "exiftool.exe")

        # Resolve availability
        if not os.path.isfile(exe) and shutil.which(exe) is None:
            raise FileNotFoundError(f"exiftool not found at '{exe}'")

        # Sanity check & log version (helps diagnose 32/64-bit or wrong binary)
        try:
            ver_proc = subprocess.run(
                [exe, "-ver"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            ver = ver_proc.stdout.decode("utf-8", errors="ignore").strip()
            logging.info("Using exiftool %s at %s", ver, exe)
        except Exception as e:
            logging.error("exiftool sanity check failed: %s", e)
            raise

        results = {}
        BATCH = 150  # keep args manageable on Windows

        for i in range(0, len(files), BATCH):
            batch = files[i:i+BATCH]
            batch_abs = [os.path.abspath(p) for p in batch]

            cmd = [
                exe,
                "-j",                       # JSON output
                "-n",                       # numeric values
                "-api", "largefilesupport=1",
                "-charset", "filename=utf8",
                "-q", "-q",                 # be quiet so JSON stays clean on STDOUT
            ] + batch_abs

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                raw_out = proc.stdout.decode("utf-8", errors="ignore")
                if proc.stderr:
                    # Keep it for debugging, but don’t contaminate JSON
                    logging.debug(
                        "exiftool warnings (suppressed): %s",
                        proc.stderr.decode("utf-8", errors="ignore")[:500]
                    )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or b"").decode("utf-8", errors="ignore")
                logging.error(
                    "exiftool failed on batch starting %s: %s",
                    os.path.basename(batch_abs[0]) if batch_abs else "<none>",
                    err
                )
                continue

            try:
                arr = json.loads(raw_out)
            except Exception as e:
                logging.error(
                    "Failed to parse exiftool JSON (first 400 chars): %r",
                    raw_out[:400]
                )
                continue

            for rec in arr:
                src = rec.get("SourceFile") or rec.get("FileName")
                if not src:
                    continue
                key = os.path.abspath(src)
                results[key] = {k: v for k, v in rec.items() if k != "SourceFile"}

        logging.info(
            "Extracted EXIF for %d of %d files via exiftool subprocess.",
            len(results), len(files)
        )
        return results

    def init_data_structures(self):
        # Initialize data structures
        self.image_data_groups = defaultdict(list)
        self.root_names = []

        self.multispectral_image_data_groups = defaultdict(list)
        self.thermal_rgb_image_data_groups = defaultdict(list)
        self.multispectral_root_names = ['Multi1', 'Multi2', 'Multi3']  # Example data
        self.thermal_rgb_root_names = ['Thermal1', 'Thermal2', 'Thermal3']  # Example data
        self.current_root_index = 0
        self.root_offset = 0  # Offset between multispectral roots and thermal/RGB roots
        self.root_id_mapping = {}
        self.mode = 'dual_folder'
        self.sync_enabled = True  # Start with synchronization enabled
        self.root_coordinates = {}
        self.all_polygons = defaultdict(dict)  # Key: logical_name, Value: dict mapping filepath to polygon data
        self.viewer_widgets = []
        self.image_loader_thread = None
        self.image_loader_worker = None
        self.images_to_load = 0
        self.images_loaded = 0
        self.project_folder = ""  # Path to the project folder
        self.root_offset_dialog = None
        self.current_drawing_mode = "polygon"  # Default mode is polygon


    def init_ui(self):
        # Layout for the tab
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Project Name Label
        self.project_label = QLabel(self.project_name)
        self.project_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.project_label.setAlignment(QtCore.Qt.AlignCenter)
        self.project_label.setStyleSheet("color: blue;")
        self.layout.addWidget(self.project_label)

        # Image Grid Layout
        self.image_grid_layout = QtWidgets.QGridLayout()
        self.layout.addLayout(self.image_grid_layout)

        # Initialize toolbar and actions
        self.create_actions()
        self.create_toolbar()

        # Initialize Polygon Manager
        self.polygon_manager = PolygonManager(self)  # Ensure PolygonManager is defined
        self.polygon_manager.list_widget.itemClicked.connect(self.select_polygon_from_manager)
        self.polygon_manager.clear_all_polygons_signal.connect(self.clean_all_polygons)
        self.polygon_manager.edit_group_signal.connect(self.start_editing_group)
        self.inspection_on = False
  


    def create_actions(self):
        import os, sys, logging
        from PyQt5 import QtWidgets
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QIcon, QKeySequence
        from PyQt5.QtWidgets import QStyle

        # Define all actions specific to this tab
        self.openAct = QtWidgets.QAction("&Open Folder", self)
        self.openAct.setShortcut("Ctrl+O")
        self.openAct.triggered.connect(self.open_folder)

        self.openRGBAct = QtWidgets.QAction("&Open RGB Folder", self)
        self.openRGBAct.triggered.connect(self.open_rgb_folder)

        self.exitAct = QtWidgets.QAction("E&xit", self)
        self.exitAct.triggered.connect(self.close_parent)

        self.saveAct = QtWidgets.QAction("&Save Current CSV", self)
        self.saveAct.setShortcut("Ctrl+S")
        self.saveAct.triggered.connect(self.save_polygons)

        self.saveAllAct = QtWidgets.QAction("&Extract All CSV", self)
        self.saveAllAct.triggered.connect(self.save_all_polygons)

        self.nextAct = QtWidgets.QAction("Next", self)
        self.nextAct.setShortcut(QKeySequence(Qt.Key_Right))
        self.nextAct.triggered.connect(self.next_group)

        self.prevAct = QtWidgets.QAction("Previous", self)
        self.prevAct.setShortcut(QKeySequence(Qt.Key_Left))
        self.prevAct.triggered.connect(self.prev_group)

        self.syncAct = QtWidgets.QAction("Sync On", self, checkable=True)
        self.syncAct.setChecked(True)
        self.syncAct.setShortcut(QKeySequence("O"))                 
        self.syncAct.triggered.connect(self.toggle_sync)

        self.cleanAllAct = QtWidgets.QAction("Clean root Polys", self)
        self.cleanAllAct.setShortcut(QKeySequence("Ctrl+X"))        
        self.cleanAllAct.triggered.connect(self.clean_all_polygons)

        self.goToRootAct = QtWidgets.QAction("Go", self)
        self.goToRootAct.triggered.connect(self.go_to_root)

        self.polygonManagerAct = QtWidgets.QAction("Polygon Manager", self)
        self.polygonManagerAct.setShortcut(QKeySequence("P"))       
        self.polygonManagerAct.triggered.connect(self.show_polygon_manager)

        self.copyMetadataAct = QtWidgets.QAction("Copy Metadata", self)
        self.copyMetadataAct.triggered.connect(self.copy_metadata_action)

        self.openImageFoldersAct = QtWidgets.QAction("Open Image Folders", self)
        self.openImageFoldersAct.triggered.connect(self.open_image_folders)

        # Project save/load
        self.saveProjectAct = QtWidgets.QAction("Save Project", self)
        self.saveProjectAct.setShortcut("Ctrl+Shift+S")
        self.saveProjectAct.triggered.connect(self.save_project)

        self.loadProjectAct = QtWidgets.QAction("Load Project", self)
        self.loadProjectAct.setShortcut("Ctrl+L")
        self.loadProjectAct.triggered.connect(self.load_project)

        # Quick Save
        self.saveQuickAct = QtWidgets.QAction("Save", self)
        self.saveQuickAct.setShortcut("Ctrl+Q")
        self.saveQuickAct.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.saveQuickAct.triggered.connect(self.save_project_quick)

        # A / D shortcuts
        self.leftKeyAct = QtWidgets.QAction(self)
        self.leftKeyAct.setShortcut(QKeySequence("A"))
        self.leftKeyAct.triggered.connect(self.prev_group)
        self.addAction(self.leftKeyAct)

        self.rightKeyAct = QtWidgets.QAction(self)
        self.rightKeyAct.setShortcut(QKeySequence("D"))
        self.rightKeyAct.triggered.connect(self.next_group)
        self.addAction(self.rightKeyAct)

        # Copy polygons (menu-only actions)
        self.copyPolygonsNextAct = QtWidgets.QAction("Copy Polys Next", self)
        self.copyPolygonsNextAct.setShortcut("Alt+Right")
        self.copyPolygonsNextAct.triggered.connect(self.copy_polygons_to_next)

        self.copyPolygonsPrevAct = QtWidgets.QAction("Copy Polys Previous", self)
        self.copyPolygonsPrevAct.setShortcut("Alt+Left")
        self.copyPolygonsPrevAct.triggered.connect(self.copy_polygons_to_previous)

        self.copyPolygonsAllRootsAct = QtWidgets.QAction("Copy Polys → All Roots", self)
        self.copyPolygonsAllRootsAct.setShortcut("Ctrl+Shift+A")
        self.copyPolygonsAllRootsAct.triggered.connect(self.copy_polygons_to_all_roots)

        self.copyPolygonsNextTabAct = QtWidgets.QAction("Copy Polygons to Next Tab", self)
        self.copyPolygonsNextTabAct.setShortcut("Alt+3")
        self.copyPolygonsNextTabAct.triggered.connect(self.copy_polygons_to_next_tab)

        # --- Show Map (with icon fallback) ---
        self.showMapAct = QtWidgets.QAction("Show Map", self)
        self.showMapAct.setShortcut("Ctrl+M")
        icon_name = "Map_1.png"
        candidates = [
            os.path.join(os.getcwd(), icon_name),
            os.path.join(os.path.dirname(sys.argv[0]), icon_name),
            os.path.join(os.path.dirname(__file__), icon_name),
        ]
        icon_path = next((p for p in candidates if os.path.exists(p)), None)
        if icon_path:
            self.showMapAct.setIcon(QIcon(icon_path))
        else:
            self.showMapAct.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        self.showMapAct.triggered.connect(self.show_map)

        # Thumbnails
        self.saveAllThumbnailsAct = QtWidgets.QAction("Save All Thumbnails", self, triggered=self.save_all_thumbnails)

        # Root offset
        self.setRootOffsetAct = QtWidgets.QAction("Set Root Offset", self)
        self.setRootOffsetAct.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.setRootOffsetAct.triggered.connect(self.open_root_offset_dialog)

        # Inspect
        self.inspectAct = QtWidgets.QAction("Inspect", self, checkable=True)
        self.inspectAct.setShortcut(QKeySequence("I"))              
        self.inspectAct.triggered.connect(self.toggle_inspection_mode)

        # ML Manager
        self.machineLearningManagerAct = QtWidgets.QAction("MachineLearning Manager", self)
        self.machineLearningManagerAct.setIcon(QIcon.fromTheme("document-open"))
        self.machineLearningManagerAct.setShortcut("Ctrl+Shift+M")
        self.machineLearningManagerAct.triggered.connect(self.show_machine_learning_manager)

        # Refresh viewer
        self.refreshViewerAct = QtWidgets.QAction("Refresh Viewer", self)
        self.refreshViewerAct.setShortcut("E")
        self.refreshViewerAct.triggered.connect(self.refresh_viewer)

        # ---- Zoom All (Z) — use QAction with Application scope (reliable even inside line edits) ----
        self.zoomAllAct = QtWidgets.QAction("Zoom All (fit)", self)
        self.zoomAllAct.setShortcut(QKeySequence("Z"))
        self.zoomAllAct.setShortcutContext(Qt.ApplicationShortcut)      # <— key bit
        self.zoomAllAct.triggered.connect(self.zoom_all_viewers_out)
        self.addAction(self.zoomAllAct)                                  # register on this widget

        # Tooltips that show ONLY shortcuts on hover
        def _shortcuts_only(act: QtWidgets.QAction):
            sc = act.shortcut().toString() if act.shortcut() else ""
            act.setToolTip(sc)

        for a in [
            self.openAct, self.openRGBAct, self.exitAct, self.saveAct, self.saveAllAct,
            self.nextAct, self.prevAct, self.syncAct, self.cleanAllAct, self.goToRootAct,
            self.polygonManagerAct, self.copyMetadataAct, self.openImageFoldersAct,
            self.saveProjectAct, self.loadProjectAct, self.saveQuickAct,
            self.copyPolygonsNextAct, self.copyPolygonsPrevAct,
            self.copyPolygonsAllRootsAct, self.copyPolygonsNextTabAct,
            self.showMapAct, self.saveAllThumbnailsAct, self.setRootOffsetAct,
            self.inspectAct, self.machineLearningManagerAct, self.refreshViewerAct,
            self.zoomAllAct
        ]:
            _shortcuts_only(a)

        # keep a reference to the helper for use in create_toolbar (optional)
        self._shortcuts_only = _shortcuts_only

    
    def zoom_all_viewers_out(self):
        """
        Zoom every ImageViewer in this tab to 'fit whole image'.
        Finds viewers even if they're subclassed or registered elsewhere.
        """
        import logging
        viewers = []

        try:
            if hasattr(self, "viewer_widgets") and self.viewer_widgets:
                viewers.extend([v for v in self.viewer_widgets if v is not None])
        except Exception as e:
            logging.debug(f"zoom_all_viewers_out: registry probe failed: {e}")

        # 2) Fallback: any QGraphicsView child that has zoom_out_to_fit()
        try:
            for gv in self.findChildren(QtWidgets.QGraphicsView):
                if hasattr(gv, "zoom_out_to_fit"):
                    viewers.append(gv)
        except Exception as e:
            logging.debug(f"zoom_all_viewers_out: child search failed: {e}")

        # De-dup while preserving order
        seen = set()
        uniq = []
        for v in viewers:
            if id(v) not in seen:
                seen.add(id(v))
                uniq.append(v)
        viewers = uniq

        logging.info("zoom_all_viewers_out triggered; viewers=%d", len(viewers))

        if not viewers:
            return

        for v in viewers:
            try:
                # Skip empty viewers
                if hasattr(v, "has_image") and not v.has_image():
                    continue
                v.zoom_out_to_fit()
            except Exception as e:
                logging.debug(f"zoom_all_viewers_out: failed on {type(v).__name__}: {e}")

    def create_toolbar(self):
        from PyQt5 import QtWidgets, QtCore
        from PyQt5.QtCore import QSize
        from PyQt5.QtWidgets import QToolBar, QSlider, QLabel, QLineEdit, QWidget, QSizePolicy

        def _make_menu_button(text: str, menu: QtWidgets.QMenu) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setMenu(menu)
            btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)   # shows arrow
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)       # text; keep default arrow
            scs = [a.shortcut().toString() for a in menu.actions() if a.shortcut()]
            btn.setToolTip("\n".join([s for s in scs if s]))           # shortcuts-only tooltip
            return btn

        # Toolbar
        self.toolbar = QToolBar(f"{self.project_name} Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.layout.addWidget(self.toolbar)

        # Nav + housekeeping
        self.toolbar.addAction(self.prevAct)
        self.toolbar.addAction(self.nextAct)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.syncAct)
        self.toolbar.addAction(self.cleanAllAct)

        # Copy Polygons dropdown (with arrow)
        copy_menu = QtWidgets.QMenu(self)
        copy_menu.addAction(self.copyPolygonsNextAct)
        copy_menu.addAction(self.copyPolygonsPrevAct)
        copy_menu.addSeparator()
        copy_menu.addAction(self.copyPolygonsAllRootsAct)
        copy_menu.addAction(self.copyPolygonsNextTabAct)
        self.toolbar.addWidget(_make_menu_button("Copy Polygons", copy_menu))

        # Polygon manager + tools
        self.toolbar.addAction(self.polygonManagerAct)
        

        # Drawing mode toggle
        self.toggleDrawingModeAct = QtWidgets.QAction("Drawing Mode: Polygon", self, checkable=True)
        self.toggleDrawingModeAct.setChecked(False)
        self.toggleDrawingModeAct.setShortcut(QtGui.QKeySequence("Alt+D"))  
        self.toggleDrawingModeAct.triggered.connect(self.toggle_drawing_mode)
        self.toolbar.addAction(self.toggleDrawingModeAct)
        try:
            self._shortcuts_only(self.toggleDrawingModeAct)
        except Exception:
            sc = self.toggleDrawingModeAct.shortcut().toString()
            self.toggleDrawingModeAct.setToolTip(sc)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.saveQuickAct)
        self.toolbar.addSeparator()

        # Root number input + go (numeric IDs)
        self.root_number_input = QLineEdit()
        self.root_number_input.setPlaceholderText("Enter root number (e.g. '1')")
        self.root_number_input.returnPressed.connect(self.go_to_root)
        self.toolbar.addWidget(self.root_number_input)
        self.toolbar.addAction(self.goToRootAct)

        # Modules dropdown (now includes Open Image Folders)
        modules_menu = QtWidgets.QMenu(self)
        modules_menu.addAction(self.showMapAct)
        modules_menu.addAction(self.machineLearningManagerAct)
        modules_menu.addSeparator()
        modules_menu.addAction(self.setRootOffsetAct)
        modules_menu.addAction(self.openImageFoldersAct)              # ✅ moved into Modules
        self.toolbar.addWidget(_make_menu_button("Modules", modules_menu))

        # Slider (indices 0..N-1), label shows ONLY numeric root ID
        self.group_slider = QSlider(QtCore.Qt.Horizontal)
        self.group_slider.setMinimum(0)
        self.group_slider.setMaximum(max(0, len(self.multispectral_root_names) - 1))
        self.group_slider.setValue(self.current_root_index)
        self.group_slider.setSingleStep(1)
        self.group_slider.setTickPosition(QSlider.TicksBelow)
        self.group_slider.setTickInterval(1)
        self.group_slider.setTracking(True)
        self.group_slider.valueChanged.connect(self._on_slider_value_changed)
        self.group_slider.sliderReleased.connect(self.slider_released)
        self.group_slider.actionTriggered.connect(self._on_slider_action)
        self.toolbar.addWidget(self.group_slider)

        self.slider_label = QLabel(str(self._root_id_for_index(self.current_root_index)))
        self.toolbar.addWidget(self.slider_label)
        self.toolbar.addSeparator()
        # Remaining actions
        self.toolbar.addAction(self.inspectAct)
        
        self.toolbar.addAction(self.refreshViewerAct)

        # Sync label once
        self.update_slider_label(self.group_slider.value())



    def _on_slider_action(self, action: int):
        """
        Called for groove clicks (page steps), keyboard nudges, etc.
        For single-clicks the handle isn't 'down', so jump immediately.
        Avoid spamming loads during an active drag (SliderMove while down).
        """
        if not hasattr(self, "group_slider"):
            return

        # Always keep the label in sync
        val = self.group_slider.value()
        self.update_slider_label(val)

        # If it's a page/single step (typical for groove-click), jump now.
        # Skip SliderMove while dragging; that case is handled by sliderReleased.
        if action in (
            QAbstractSlider.SliderSingleStepAdd,
            QAbstractSlider.SliderSingleStepSub,
            QAbstractSlider.SliderPageStepAdd,
            QAbstractSlider.SliderPageStepSub,
            QAbstractSlider.SliderToMinimum,
            QAbstractSlider.SliderToMaximum,
        ):
            self._jump_to_index(val)



    def _on_slider_value_changed(self, value: int):
        """Always refresh the label; if this was a click (not a drag), jump now."""
        self.update_slider_label(value)
        # If the user clicked the groove (not dragging), sliderDown is False -> jump immediately
        if hasattr(self, "group_slider") and not self.group_slider.isSliderDown():
            self._jump_to_index(value)

    def slider_released(self):
        """When the user finishes a drag, jump to the selected index."""
        if hasattr(self, "group_slider"):
            self._jump_to_index(self.group_slider.value())

    def _jump_to_index(self, idx: int):
        """Centralized 'go to index' used by the slider and other UI."""
        total = len(self.multispectral_root_names)
        if total == 0:
            return
        idx = max(0, min(idx, total - 1))

        if idx == getattr(self, "current_root_index", 0):
            return

        # Save polygons for current root (best-effort)
        try:
            if 0 <= self.current_root_index < total:
                cur_root = self.multispectral_root_names[self.current_root_index]
                self.save_polygons_to_json(root_name=cur_root)
        except Exception:
            pass

        # Switch
        self.current_root_index = idx

        # Keep slider/label in sync without causing re-entrant signals
        if hasattr(self, "group_slider"):
            self.group_slider.blockSignals(True)
            self.group_slider.setValue(idx)
            self.group_slider.blockSignals(False)
        self.update_slider_label(idx)

        # Load the new group
        try:
            self.load_image_group(self.multispectral_root_names[idx])
        except Exception as e:
            print(f"Failed to load image group at index {idx}: {e}")

    def _refresh_root_offset_ui(self):
        """Keep the Modules menu 'Root Offset: N' line in sync with current value."""
        if hasattr(self, "rootOffsetInfoAct") and self.rootOffsetInfoAct:
            self.rootOffsetInfoAct.setText(f"Root Offset: {self.root_offset}")



    def _rotate_point_in_rect(self, x, y, w, h, rot):
        """Rotate a point (x,y) in a WxH image by rot∈{0,90,180,270} clockwise."""
        r = (int(rot) // 90) % 4
        if r == 0:  return (x, y)
        if r == 1:  return (h - 1 - y, x)
        if r == 2:  return (w - 1 - x, h - 1 - y)
        # r == 3
        return (y, w - 1 - x)


    def _points_replay_ax(self, pts, W, H, mods, clamp=True):
        """
        Replays .ax (crop/resize/expr) on raw-space points.
        If clamp=True, returns ints clamped to [0..W-1]/[0..H-1] (good for drawing).
        If clamp=False, returns floats, UNCLAMPED (good for CSV sampling).
        """
        P = [(float(x), float(y)) for (x, y) in pts]
        # ... apply crop/resize steps, keeping floats ...

        if clamp:
            xmax, ymax = max(0, W - 1), max(0, H - 1)
            return [(0 if x < 0 else (xmax if x > xmax else int(round(x))),
                     0 if y < 0 else (ymax if y > ymax else int(round(y)))) for (x, y) in P]
        else:
            return P


    def _points_through_mods(self, pts_xy, mods, src_shape_hw):
        """
        Map points from RAW image coords -> final edited image coords
        using the same order as the image pipeline: rotate -> crop -> resize.
        pts_xy: iterable of (x,y) in RAW coords (pre-mods)
        src_shape_hw: (H, W) of the RAW image
        returns: (mapped_points, final_shape_hw)
        """
        import math

        H, W = int(src_shape_hw[0]), int(src_shape_hw[1])
        P = [(float(x), float(y)) for (x, y) in pts_xy]

        # --- 1) rotate (90° steps, same orientation as image path) ---
        rot = 0
        try:
            rot = int(mods.get("rotate", 0)) % 360
        except Exception:
            rot = 0

        def _rot90_cw(x, y, W, H):  # 90° clockwise
            # numpy rot90(arr, -1) used in image path; this matches that
            return (W - 1 - y, x)

        def _rot180(x, y, W, H):
            return (W - 1 - x, H - 1 - y)

        def _rot270_cw(x, y, W, H):  # 270° cw == 90° ccw
            return (y, H - 1 - x)

        if rot in (90, 180, 270):
            P = [(_rot90_cw(x, y, W, H) if rot == 90 else
                  _rot180(x, y, W, H)   if rot == 180 else
                  _rot270_cw(x, y, W, H)) for (x, y) in P]
            # swap W/H for 90/270
            if rot in (90, 270):
                W, H = H, W

        # --- 2) crop (scale rect from ref_size -> current W,H, then translate) ---
        xywh = None
        if isinstance(mods.get("crop_rect"), (dict, list, tuple)):
            # reuse  existing crop parser if present; otherwise extract here
            try:
                cr = mods["crop_rect"]
                if isinstance(cr, dict):
                    xywh = (float(cr.get("x", 0)), float(cr.get("y", 0)),
                            float(cr.get("w", 0)), float(cr.get("h", 0)))
                else:
                    # [x,y,w,h] or (x,y,w,h)
                    xywh = (float(cr[0]), float(cr[1]), float(cr[2]), float(cr[3]))
            except Exception:
                xywh = None

        if xywh:
            cx, cy, cw, ch = xywh
            ref = mods.get("crop_rect_ref_size") or {}
            ref_w = int(ref.get("w", W)) or W
            ref_h = int(ref.get("h", H)) or H
            sx = float(W) / float(ref_w)
            sy = float(H) / float(ref_h)

            # scale crop rect from its reference basis to current
            cxs = int(round(cx * sx))
            cys = int(round(cy * sy))
            cws = int(round(cw * sx))
            chs = int(round(ch * sy))

            # translate points into the cropped frame
            P = [(x - cxs, y - cys) for (x, y) in P]
            W, H = cws, chs

        # --- 3) resize (percentage OR absolute w/h) ---
        rz = mods.get("resize")
        if isinstance(rz, dict):
            if "scale" in rz:
                s = max(0.0, float(rz.get("scale", 100))) / 100.0
                rx = ry = s
                W2 = max(1, int(round(W * rx)))
                H2 = max(1, int(round(H * ry)))
            else:
                tw = int(rz.get("width", 0))
                th = int(rz.get("height", 0))
                if tw and th:
                    rx, ry = float(tw) / float(W), float(th) / float(H)
                    W2, H2 = tw, th
                elif tw:  # preserve aspect
                    rx = float(tw) / float(W)
                    W2 = tw
                    H2 = max(1, int(round(H * rx)))
                    ry = float(H2) / float(H)
                elif th:
                    ry = float(th) / float(H)
                    H2 = th
                    W2 = max(1, int(round(W * ry)))
                    rx = float(W2) / float(W)
                else:
                    rx = ry = 1.0
                    W2, H2 = W, H

            if rx != 1.0 or ry != 1.0:
                P = [(x * rx, y * ry) for (x, y) in P]
                W, H = W2, H2

        # return ints (like sampling code expects) plus final shape
        mapped = [(int(round(x)), int(round(y))) for (x, y) in P]
        return mapped, (H, W)


    def _points_to_export_frame(self, filepath, pts, polygon_data, export_shape):
        """
        Map saved shape points into the *export* image frame used for CSV sampling.
        Returns floats; NO clamping. Callers must round/clip as needed.
        """
        import logging
        H, W = int(export_shape[0]), int(export_shape[1])
        if not pts:
            return []

        coord_space = (polygon_data or {}).get("coord_space", "scene")

        if coord_space == "image":
            # Points saved in image/pixmap pixels. Rescale to current export.
            src_sz = polygon_data.get("image_size") or polygon_data.get("pixmap_size")
            if isinstance(src_sz, (list, tuple)) and len(src_sz) == 2 and all(src_sz):
                pw, ph = float(src_sz[0]), float(src_sz[1])  # (W, H)
                sx, sy = (W / max(1.0, pw)), (H / max(1.0, ph))
                mapped = [(float(x) * sx, float(y) * sy) for (x, y) in pts]
            else:
                mapped = [(float(x), float(y)) for (x, y) in pts]

            # If saved before a crop/resize edit, rescale can land OOB; that’s fine.
            # Log once for visibility but do NOT clamp here.
            oob = [(x, y) for (x, y) in mapped if x < 0 or y < 0 or x >= W or y >= H]
            if oob:
                logging.warning(
                    "Mapped %d point(s) outside bounds for '%s' (image→export rescale; "
                    "saved image_size may predate current .ax edits). Example: %s",
                    len(oob), filepath, oob[0]
                )

        elif coord_space == "raw":
            # Replay .ax on raw coordinates, but do not clamp/round here.
            try:
                raw = self._load_raw_image(filepath)
                if raw is not None:
                    raw_h, raw_w = raw.shape[:2]
                    ax = self._load_ax_mods(filepath) or {}
                    mapped = self._points_replay_ax(pts, raw_w, raw_h, ax, clamp=False)
                else:
                    mapped = [(float(x), float(y)) for (x, y) in pts]
            except Exception:
                mapped = [(float(x), float(y)) for (x, y) in pts]

        else:
            # Legacy: scene → image
            mapped = self._map_points_scene_to_image(filepath, pts, (H, W, 1), polygon_data=polygon_data)
            # Ensure floats; _map_points_scene_to_image may already round but does not clamp.
            mapped = [(float(x), float(y)) for (x, y) in mapped]

        return mapped


    def _root_keys_ordered(self):
        """
        Return ordered list of root IDs ('1','2','3',...). Uses root_mapping_dict if present,
        otherwise falls back to 1..N from multispectral_root_names.
        """
        rm = getattr(self, "root_mapping_dict", None) or getattr(self, "root_mapping", None)
        if isinstance(rm, dict) and rm:
            try:
                return sorted(rm.keys(), key=lambda k: int(k))
            except Exception:
                return list(rm.keys())
        n = len(self.multispectral_root_names) if getattr(self, "multispectral_root_names", None) else 0
        return [str(i + 1) for i in range(n)]

    def _root_number_for_index(self, idx: int) -> str:
        keys = self._root_keys_ordered()
        return keys[idx] if 0 <= idx < len(keys) else str(idx + 1)

    def _index_for_root_id(self, rid: int):
        """
        Map a numeric root ID -> multispectral index.
        Prefer id_to_root mapping built by map_matching_roots(); fallback to rid-1.
       """
        # Fast path via id_to_root 
        root_name = None
        try:
            root_name = (getattr(self, "id_to_root", {}) or {}).get(int(rid))
        except Exception:
            root_name = None

        if root_name and root_name in self.multispectral_root_names:
            return self.multispectral_root_names.index(root_name)

        # Fallback: 1-based -> 0-based, if in range
        idx = int(rid) - 1
        if 0 <= idx < len(self.multispectral_root_names):
            return idx
        return None

    def update_slider_label(self, value: int):
        # label shows ONLY the number; do NOT change current_root_index here
        self.slider_label.setText(self._root_number_for_index(value))



    def _ax_path_for_fp(self, fp):
        import os
        pf = getattr(self, "project_folder", None)
        base = os.path.splitext(os.path.basename(fp))[0] + ".ax"
        return (os.path.join(os.fspath(pf), base) if pf and pf.strip()
                else os.path.splitext(fp)[0] + ".ax")

    def _load_ax_json(self, fp):
        import os, json
        axp = self._ax_path_for_fp(fp)
        try:
            if os.path.exists(axp):
                with open(axp, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _raw_hw_quick(self, fp):
        import os
        ext = os.path.splitext(fp)[1].lower()
        if ext in (".tif", ".tiff"):
            try:
                import tifffile as tiff
                with tiff.TiffFile(fp) as tf:
                    s = tf.series[0]
                    axes = s.axes or ""
                    shape = s.shape
                    if "Y" in axes and "X" in axes:
                        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
                    return int(shape[-2]), int(shape[-1])
            except Exception:
                pass
        try:
            from PIL import Image
            with Image.open(fp) as im:
                w, h = im.size
                return h, w
        except Exception:
            pass
        try:
            import cv2
            im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if im is not None:
                return im.shape[:2]
        except Exception:
            pass
        return (None, None)

    def _size_after_ax_fast(self, fp):
        """Simulate rotate→crop→resize on dimensions only; no image decode."""
        h0, w0 = self._raw_hw_quick(fp)
        if not h0 or not w0:
            return (None, None)
        ax = self._load_ax_json(fp)

        # op order
        try:
            order = self._ax_op_order(ax)
        except Exception:
            raw = ax.get("op_order") or ax.get("ops") or ["rotate","crop","resize"]
            if isinstance(raw, str): raw = [raw]
            order = []
            for s in (raw or []):
                t = str(s).lower()
                if "rot" in t: order.append("rotate")
                elif "crop" in t: order.append("crop")
                elif "siz" in t or "res" in t: order.append("resize")
            if not order: order = ["rotate","crop","resize"]

        # params
        try: rot = int(ax.get("rotate", 0)) % 360
        except Exception: rot = 0
        rect  = ax.get("crop_rect") or None
        cref  = ax.get("crop_rect_ref_size") or None
        rz    = ax.get("resize") or None

        def dims_after_rot(w, h, r):
            r = (int(r)//90) % 4
            return (w, h) if r in (0,2) else (h, w)

        W, H = w0, h0
        for op in order:
            o = str(op).lower()
            if o == "rotate" and rot in (90,180,270):
                if rot in (90,270): W, H = H, W

            elif o == "crop" and rect:
                if isinstance(cref, dict) and "w" in cref and "h" in cref:
                    refW = int(cref.get("w") or W); refH = int(cref.get("h") or H)
                else:
                    # assume rect saved in post-rotate space if rotate precedes crop
                    refW, refH = dims_after_rot(w0, h0, rot) if ("rotate" in order and order.index("rotate") < order.index("crop")) else (w0, h0)

                rw = int(rect.get("width", 0)); rh = int(rect.get("height", 0))
                sx = W / float(max(1, refW)); sy = H / float(max(1, refH))
                W = max(1, int(round(rw * sx)))
                H = max(1, int(round(rh * sy)))

            elif o == "resize" and isinstance(rz, dict) and (W>0 and H>0):
                if "scale" in rz:
                    s = max(1, int(rz.get("scale", 100))) / 100.0
                    W = max(1, int(round(W * s))); H = max(1, int(round(H * s)))
                else:
                    W = max(1, int(round(W * (int(rz.get("width", 100)) / 100.0))))
                    H = max(1, int(round(H * (int(rz.get("height", 100)) / 100.0))))
        return (H, W)


    def handle_image_loaded(self, image_data):
        # Create ImageViewer and associated widgets
        viewer = ImageViewer()
        viewer.polygon_drawn.connect(self.on_polygon_drawn)
        viewer.polygon_changed.connect(self.on_polygon_modified)

        # Create label for the image
        label = QtWidgets.QLabel(os.path.basename(image_data.filepath))
        label.setAlignment(QtCore.Qt.AlignCenter)

        # Create "Clean Vector" button
        clean_button = QtWidgets.QPushButton("Clean Vector")
        clean_button.clicked.connect(partial(self.clean_all_polygons, viewer))
        clean_button.clicked.connect(partial(self.delete_polygons_for_viewer, viewer))

        # Create container widget with vertical layout
        container = QtWidgets.QWidget()
        v_layout = QtWidgets.QVBoxLayout(container)
        v_layout.addWidget(label)
        v_layout.addWidget(viewer)
        v_layout.addWidget(clean_button)

        # Add to viewer_widgets list
        self.viewer_widgets.append({
            'container': container,
            'viewer': viewer,
            'image_data': image_data,
            'band_index': len(self.viewer_widgets) + 1  # Adjust as needed
        })

        # Determine grid position
        row, col = divmod(len(self.viewer_widgets) - 1, 5)  # Assuming 5 columns
        self.image_grid_layout.addWidget(container, row, col)

        # --- detect per-file band expression (controls preview only) ---
        has_expr = False
        try:
            ax = self._load_ax_json(image_data.filepath)
            has_expr = bool((ax.get("band_expression") or "").strip())
        except Exception:
            has_expr = False

        # keep a flag on the viewer 
        viewer.preview_prefers_index = has_expr

        # Convert image to QPixmap with normalization (preview may prefer last band)
        pixmap = self.convert_cv_to_pixmap(image_data.image, prefer_last_band=has_expr)
        if pixmap is not None:
            viewer.set_image(pixmap)
            self._wire_viewer_for_inspection(viewer)

            # Attach image data for statistics (full stack incl. expression kept here)
            viewer.image_data = image_data

            # Load existing polygons if any
            self.load_polygons(viewer, image_data)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not display image: {image_data.filepath}")


    def _viewer_basis_hw(self, viewer, *, fallback_hw=None):
        """
        Return (H,W) for the *viewer image basis* that image_to_scene_coords expects.
        Order of tries:
          1) viewer.image_data.image.shape
          2) viewer.pixmap_item.pixmap().height/width
          3) fallback_hw (e.g., target-effective)
        """
        # 1) numpy image on the viewer
        try:
            idata = getattr(viewer, "image_data", None)
            if idata is not None and getattr(idata, "image", None) is not None:
                h, w = idata.image.shape[:2]
                return (int(h), int(w))
        except Exception:
            pass

        # 2) actual displayed pixmap
        try:
            # common names: pixmap_item, _pixmap_item
            pitem = getattr(viewer, "pixmap_item", None) or getattr(viewer, "_pixmap_item", None)
            if pitem is not None:
                pm = pitem.pixmap()
                if not pm.isNull():
                    return (int(pm.height()), int(pm.width()))
        except Exception:
            pass

        # 3) fallback
        if fallback_hw and all(fallback_hw):
            return (int(fallback_hw[0]), int(fallback_hw[1]))
        return (None, None)


   
    def _apply_band_multipliers(self, img, mods):
        """
        Apply per-band multipliers to a numpy image if provided in .ax.
        Accepts:
          mods["band_multipliers"] as list [m1, m2, ...] or dict {"1": m1, "2": m2, ...}
          (also accepts "band_gains" as an alias)
        Returns float32 image with multipliers applied (does NOT normalize/clip).
        """
        import numpy as np
        gains = mods.get("band_multipliers") or mods.get("band_gains")
        if gains is None or img is None:
            return img

        x = img.astype(np.float32, copy=False)
        if x.ndim == 2:
            # single band
            g = float(gains[0] if isinstance(gains, (list, tuple)) else gains.get("1", 1.0))
            return x * g

        # multi-band
        if isinstance(gains, dict):
            for k, v in gains.items():
                try:
                    ch = int(k) - 1
                    if 0 <= ch < x.shape[2]:
                        x[..., ch] *= float(v)
                except Exception:
                    pass
        else:
            arr = np.asarray(gains, dtype=np.float32)
            n = min(x.shape[2], arr.size)
            for ch in range(n):
                x[..., ch] *= float(arr[ch])
        return x


    def load_polygons(self, viewer, image_data):
        """
        Load polygons for this image_data into the given viewer.
        - Only load polygons whose 'root' matches this image's root (prevents cross-root leakage).
        - If polygons are in IMAGE coords, rescale from their saved 'image_ref_size'
          to the viewer's image basis (what image_to_scene_coords expects), with a
          robust fallback to the file's effective size after .ax.
        - Heuristics for legacy saves: auto-interpret pixel-like 'scene' points as 'image';
          skip polygons that clearly belong to another resolution if no 'root' is present.
        """
        import os, glob, json
        from PyQt5 import QtCore, QtGui

        if not self.project_folder:
            return

        polygons_dir = os.path.join(self.project_folder, 'polygons')
        if not os.path.exists(polygons_dir):
            return

        filename = os.path.basename(image_data.filepath)
        pattern = f"*_{os.path.splitext(filename)[0]}_polygons.json"
        group_files = glob.glob(os.path.join(polygons_dir, pattern))

        # ---- helpers -------------------------------------------------------------
        def _ax_sidecar_for(fp: str) -> str:
            sidecar = os.path.splitext(fp)[0] + ".ax"
            if os.path.exists(sidecar):
                return sidecar
            folder = (self.project_folder or "").strip() or os.path.dirname(fp)
            return os.path.join(folder, os.path.splitext(os.path.basename(fp))[0] + ".ax")

        def _effective_hw_for_file(fp: str):
            """
            Best-effort (H,W) after .ax (rotate/crop/resize), no pixel ops.
            Prefers self._size_after_ax_fast if present; falls back to orig_size in .ax;
            then numpy image size; then viewer basis.
            """
            # (1) Fast helper if class provides it
            if hasattr(self, "_size_after_ax_fast"):
                try:
                    h, w = self._size_after_ax_fast(fp)
                    if h and w:
                        return int(h), int(w)
                except Exception:
                    pass

            # (2) Parse sidecar lightly for orig_size
            try:
                axp = _ax_sidecar_for(fp)
                if os.path.exists(axp):
                    with open(axp, "r", encoding="utf-8") as f:
                        ax = json.load(f) or {}
                    o = ax.get("orig_size") or {}
                    H = int(o.get("h") or 0)
                    W = int(o.get("w") or 0)
                    if H and W:
                        return H, W
            except Exception:
                pass

            # (3) Numpy image size
            try:
                if getattr(image_data, "image", None) is not None:
                    hh, ww = image_data.image.shape[:2]
                    return int(hh), int(ww)
            except Exception:
                pass

            # (4) Viewer basis last
            try:
                vh_, vw_ = self._viewer_basis_hw(viewer, fallback_hw=None)
                if vh_ and vw_:
                    return int(vh_), int(vw_)
            except Exception:
                pass

            return (None, None)

        def _root_of_filepath(fp: str):
            """Find the logical root name that owns this filepath."""
            for root, files in (getattr(self, "multispectral_image_data_groups", {}) or {}).items():
                if fp in files:
                    return root
            for root, files in (getattr(self, "thermal_rgb_image_data_groups", {}) or {}).items():
                if fp in files:
                    return root
            return None

        def _as_wh_pair(ps):
            """Accept dict {'w','h'} or [w,h]/(w,h) and return (w,h) ints or (0,0)."""
            if isinstance(ps, dict):
                w = int(ps.get("w") or 0)
                h = int(ps.get("h") or 0)
                return w, h
            if isinstance(ps, (list, tuple)) and len(ps) >= 2:
                try:
                    w = int(ps[0] or 0)
                    h = int(ps[1] or 0)
                    return w, h
                except Exception:
                    return (0, 0)
            return (0, 0)

        def _far_rel(a, b, tol=0.15):
            if not a or not b:
                return False
            return abs(a - b) / float(max(a, b)) > tol

        # This image's owning root id (used to filter)
        owning_root_name = _root_of_filepath(image_data.filepath)
        owning_root_id = None
        if owning_root_name and hasattr(self, "root_id_mapping"):
            owning_root_id = str(self.root_id_mapping.get(owning_root_name, ""))

        # The basis that image_to_scene_coords expects; fall back to effective HW
        eff_h, eff_w = _effective_hw_for_file(image_data.filepath)
        try:
            vh, vw = self._viewer_basis_hw(viewer, fallback_hw=(eff_h, eff_w))
        except Exception:
            vh, vw = (eff_h, eff_w)

        # ---- load & draw ---------------------------------------------------------
        viewer.blockSignals(True)
        try:
            for polygon_filepath in group_files:
                try:
                    with open(polygon_filepath, 'r', encoding='utf-8') as f:
                        polygon_data = json.load(f)

                    # HARD FILTER by saved root
                    saved_root = polygon_data.get('root')
                    if owning_root_id and saved_root is not None and str(saved_root) != owning_root_id:
                        continue  # skip polygons from other folder/group/resolution

                    group_name = os.path.basename(polygon_filepath).split('_')[0]
                    self.all_polygons.setdefault(group_name, {})
                    # Mirror on-disk data in memory (do NOT mutate points/basis here)
                    self.all_polygons[group_name][image_data.filepath] = polygon_data

                    pts = polygon_data.get('points') or []
                    name = polygon_data.get('name', '')
                    poly_type = (polygon_data.get('type') or 'polygon').lower()
                    coord_space = (polygon_data.get('coord_space') or 'scene').lower()

                    # FROM-basis candidates
                    ref = polygon_data.get('image_ref_size') or {}
                    ref_w, ref_h = _as_wh_pair(ref)

                    # Accept older 'pixmap_size' as fallback if present
                    if not (ref_w and ref_h):
                        ps = polygon_data.get('pixmap_size')
                        pw, ph = _as_wh_pair(ps)
                        if pw and ph:
                            ref_w, ref_h = pw, ph

                    # Last resort FROM-basis: this file's effective size
                    if not (ref_w and ref_h) and eff_w and eff_h:
                        ref_w, ref_h = eff_w, eff_h

                    # --- heuristic guards/auto-upgrade for legacy saves ---
                    # If there's no saved_root and the declared ref size is far from the current effective size,
                    # assume it belongs to another resolution folder and skip.
                    if saved_root is None and (ref_w and ref_h and eff_w and eff_h) and \
                       (_far_rel(ref_w, eff_w) or _far_rel(ref_h, eff_h)):
                        continue

                    # If 'scene' but points look like pixel coords (>>1), treat as image.
                    if coord_space == "scene" and pts:
                        try:
                            maxx = max(float(x) for (x, _) in pts)
                            maxy = max(float(y) for (_, y) in pts)
                        except Exception:
                            maxx = maxy = 0.0
                        if maxx > 4.0 and maxy > 4.0:
                            coord_space = "image"
                            if not (ref_w and ref_h) and eff_w and eff_h:
                                ref_w, ref_h = eff_w, eff_h

                    # TO-basis (what image_to_scene_coords expects for THIS viewer)
                    to_w, to_h = vw, vh

                    # Compose scene points
                    scene_pts = []

                    if coord_space == "image":
                        # Scale from saved image basis -> viewer image basis
                        if (ref_w and ref_h and to_w and to_h) and (ref_w != to_w or ref_h != to_h):
                            sx = float(to_w) / float(ref_w)
                            sy = float(to_h) / float(ref_h)
                            pts_adj = [(float(x) * sx, float(y) * sy) for (x, y) in pts]
                        else:
                            pts_adj = [(float(x), float(y)) for (x, y) in pts]

                        # image (viewer basis) -> scene
                        for (x, y) in pts_adj:
                            sp = self.image_to_scene_coords(viewer, QtCore.QPointF(x, y))
                            scene_pts.append(sp)

                    elif coord_space == "norm_pixmap":
                        # Already in viewer space
                        for (x, y) in pts:
                            scene_pts.append(QtCore.QPointF(float(x), float(y)))

                    else:
                        # Legacy true-scene basis: draw as-is
                        for (x, y) in pts:
                            scene_pts.append(QtCore.QPointF(float(x), float(y)))

                    qpoly = QtGui.QPolygonF(scene_pts)

                    # Add to scene with correct type
                    if poly_type == "point":
                        if hasattr(viewer, "add_point_to_scene"):
                            viewer.add_point_to_scene(qpoly, name)
                        elif hasattr(viewer, "add_point"):
                            viewer.add_point(qpoly, name)
                        else:
                            viewer.add_polygon(qpoly, name)
                    else:
                        if hasattr(viewer, "add_polygon_to_scene"):
                            viewer.add_polygon_to_scene(qpoly, name)
                        else:
                            viewer.add_polygon(qpoly, name)

                    # Optional debug:
                    # print(f"[poly-load] {os.path.basename(image_data.filepath)}: ref=({ref_w},{ref_h}) -> to=({to_w},{to_h}) [{coord_space}]")

                except Exception as e:
                    print(f"Could not load polygons from {polygon_filepath}: {e}")
        finally:
            viewer.blockSignals(False)


    def copy_polygons_between_roots(
        self,
        source_root,
        target_root,
        *,
        broadcast_if_ambiguous=True,
        rescale=True,
        defer_viewer_update=False,
        defer_save=False,
    ):
        """
        Copy polygons from source_root → target_root *within the same folder type*.

        Stack-safe + resolution-safe:
          • Uses post-.ax *effective image size* with a sanity check against the observed file header.
          • If filename suffix pairing is ambiguous, optionally BROADCAST to all files
            in that folder type (MS→all MS targets; TRGB→all TRGB targets).
          • Polygons persisted in IMAGE coords *with* 'image_ref_size' of the TARGET.
          • Draw path rescales from target effective basis -> live viewer image basis.

        New:
          • defer_viewer_update: when True, skips drawing into viewers (much faster in bulk).
          • defer_save: when True, skips save_polygons_to_json here (save once after the bulk).
        """
        import os, re, json
        from PyQt5 import QtCore, QtGui

        # ---------- filename helpers ----------
        def _suffix_after_root(filename):
            base = os.path.basename(filename)
            m = re.match(r'^(IMG_\d+)(.*)$', base, flags=re.IGNORECASE)
            return m.group(2) if m else base

        def _paired_thermal_root(ms_root):
            if self.mode != 'dual_folder':
                return None
            try:
                ms_idx = self.multispectral_root_names.index(ms_root)
            except ValueError:
                return None
            th_idx = ms_idx + int(self.root_offset)
            if 0 <= th_idx < len(self.thermal_rgb_root_names):
                return self.thermal_rgb_root_names[th_idx]
            return None

        def _all_viewers_for_filepath(fp):
            out = []
            for w in getattr(self, 'viewer_widgets', []) or []:
                idata = w.get('image_data')
                if idata is not None and getattr(idata, 'filepath', None) == fp:
                    v = w.get('viewer')
                    if v and v not in out:
                        out.append(v)
            v = self.get_viewer_by_filepath(fp)
            if v and v not in out:
                out.append(v)
            return out

        def _first_viewer_basis_for(fp):
            """Return (h,w) of the first live viewer that shows fp, else (None,None)."""
            vs = _all_viewers_for_filepath(fp)
            if not vs:
                return (None, None)
            try:
                return self._viewer_basis_hw(vs[0], fallback_hw=None)
            except Exception:
                return (None, None)

        # ---------- .ax + dims helpers (FAST, no pixel ops) ----------
        def _ax_path_for(fp: str) -> str:
            # PREFER SIDECAR .ax NEXT TO THE IMAGE; project-level only as fallback.
            sidecar = os.path.splitext(fp)[0] + ".ax"
            if os.path.exists(sidecar):
                return sidecar
            folder = (self.project_folder or "").strip() or os.path.dirname(fp)
            project_ax = os.path.join(folder, os.path.splitext(os.path.basename(fp))[0] + ".ax")
            return project_ax

        def _parse_op_order(ax: dict):
            raw = ax.get("op_order") or ax.get("ops") or ["rotate", "crop", "resize"]
            if isinstance(raw, str):
                raw = [raw]
            out = []
            for s in (raw or []):
                t = str(s).lower()
                if "rot" in t: out.append("rotate")
                elif "crop" in t: out.append("crop")
                elif "siz" in t or "res" in t: out.append("resize")
                elif "band" in t: out.append("band_expression")
            return [op for op in out if op in ("rotate","crop","resize","band_expression")] or ["rotate","crop","resize"]

        def _clampi(v, lo, hi):
            return max(lo, min(hi, int(v)))

        # --- observed header probe (prefers *real* image dims if available) ---
        def _probe_image_hw(fp):
            """Best-effort (H,W) from a live viewer, exported image helper, or file header."""
            # 1) live viewer already holding this image?
            try:
                for w in getattr(self, 'viewer_widgets', []) or []:
                    idata = w.get('image_data')
                    if idata is not None and getattr(idata, 'filepath', None) == fp:
                        img = getattr(idata, 'image', None)
                        if img is not None:
                            h, w = img.shape[:2]
                            return int(h), int(w)
            except Exception:
                pass
            try:
                img, _ = self._get_export_image(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            # 3) fallback to header via OpenCV (no PIL dependency)
            try:
                import numpy as np, cv2
                data = np.fromfile(fp, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            return (None, None)

        def _size_after_ax_fast_from_file(fp):
            """
            Return (H,W) AFTER rotate/crop/resize (simulated from .ax), but if that
            disagrees notably with the observed header, trust the observed header.
            """
            axH = axW = None
            try:
                ax_path = _ax_path_for(fp)
                if os.path.exists(ax_path):
                    with open(ax_path, "r", encoding="utf-8") as f:
                        ax = json.load(f) or {}
                    # base dims
                    o = ax.get("orig_size") or {}
                    H0 = int(o.get("h") or 0); W0 = int(o.get("w") or 0)
                    if not (H0 and W0):
                        obsH, obsW = _probe_image_hw(fp)
                        if obsH and obsW:
                            H0, W0 = obsH, obsW
                    if not (H0 and W0):
                        return (None, None)

                    # parse ops
                    try: rot = int(ax.get("rotate", 0)) % 360
                    except Exception: rot = 0
                    crop     = ax.get("crop_rect") or None
                    crop_ref = ax.get("crop_rect_ref_size") or None
                    resize   = ax.get("resize") or None
                    op_order = _parse_op_order(ax)

                    # simulate dims
                    curH, curW = int(H0), int(W0)

                    def do_rotate():
                        nonlocal curH, curW
                        if rot in (90, 270):
                            curH, curW = curW, curH

                    def do_crop():
                        nonlocal curH, curW
                        if not crop: return
                        refW = int((crop_ref or {}).get("w") or curW)
                        refH = int((crop_ref or {}).get("h") or curH)
                        rw = max(0, int(crop.get("width" , 0)))
                        rh = max(0, int(crop.get("height", 0)))
                        if refW <= 0 or refH <= 0: return
                        newW = _clampi(round(rw * (curW / float(refW))), 1, curW)
                        newH = _clampi(round(rh * (curH / float(refH))), 1, curH)
                        curW, curH = int(newW), int(newH)

                    def do_resize():
                        nonlocal curH, curW
                        if not isinstance(resize, dict) or not resize: return
                        if "scale" in resize:
                            s = max(1, int(resize.get("scale", 100))) / 100.0
                            curW = max(1, int(round(curW * s)))
                            curH = max(1, int(round(curH * s)))
                        else:
                            pw = max(1, int(resize.get("width", 100))) / 100.0
                            ph = max(1, int(resize.get("height", 100))) / 100.0
                            curW = max(1, int(round(curW * pw)))
                            curH = max(1, int(round(curH * ph)))

                    ops = {"rotate": do_rotate, "crop": do_crop, "resize": do_resize}
                    for op in op_order:
                        f = ops.get(op)
                        if f: f()
                    axH, axW = int(curH), int(curW)
            except Exception:
                axH = axW = None

            # observed header of the current file
            obsH, obsW = _probe_image_hw(fp)

            def _far(a, b, tol=0.05):
                if not a or not b: return False
                return abs(a - b) / float(max(a, b)) > tol

            if axH and axW and obsH and obsW:
                # disagree notably? trust observed/baked file
                if _far(axH, obsH) or _far(axW, obsW):
                    return (obsH, obsW)
                return (axH, axW)

            if axH and axW:  return (axH, axW)
            if obsH and obsW: return (obsH, obsW)
            return (None, None)

        # ---------- effective-size cache (robust: ax path + mtime + size, and image mtime+size) ----------
        size_cache = getattr(self, "_copy_size_cache", None)
        if size_cache is None:
            size_cache = self._copy_size_cache = {}

        def _eff_size(fp):
            import os
            axp = _ax_path_for(fp)
            try:
                ax_mtime = os.path.getmtime(axp) if axp and os.path.exists(axp) else None
                ax_size  = os.path.getsize(axp)  if axp and os.path.exists(axp) else None
            except Exception:
                ax_mtime = ax_size = None
            try:
                im_mtime = os.path.getmtime(fp)
                im_size  = os.path.getsize(fp)
            except Exception:
                im_mtime = im_size = None

            ax_id = (axp, ax_mtime, ax_size)
            im_id = (im_mtime, im_size)

            key = ("eff4", fp)
            rec = size_cache.get(key)  # (h, w, ax_id, im_id)
            if rec and rec[2] == ax_id and rec[3] == im_id:
                return rec[0], rec[1]

            h, w = _size_after_ax_fast_from_file(fp)
            size_cache[key] = (h, w, ax_id, im_id)
            return (h, w)

        def _best_targets_for_src(src_fp, candidates, index_map):
            """
            Choose target file(s) for a given source file:
              1) Exact suffix match -> that one.
              2) Size-aware (only if rescale=True): same (H,W) AFTER .ax -> those ...
              3) Positional fallback using index_map.
              4) Nearest-size single best match (optionally broadcast to other equal-sized files).
            """
            if not candidates:
                return []

            # 1) suffix match (IMG_#### + suffix)
            sfx = _suffix_after_root(src_fp)
            matched = [t for t in candidates if os.path.basename(t).endswith(sfx)]
            if len(matched) == 1:
                return matched

            # FAST PATH: no rescale → skip size checks
            if not rescale:
                pos = index_map.get(src_fp, None)
                if pos is not None and 0 <= pos < len(candidates):
                    return [candidates[pos]]
                return [candidates[0]]

            # 2) size-aware match
            sh, sw = _eff_size(src_fp)
            if sh and sw:
                same_size = [t for t in candidates if _eff_size(t) == (sh, sw)]
                if len(same_size) == 1:
                    return same_size
                if len(same_size) > 1:
                    if broadcast_if_ambiguous:
                        return same_size
                    pos = index_map.get(src_fp, None)
                    if pos is not None and 0 <= pos < len(candidates):
                        cand = candidates[pos]
                        if _eff_size(cand) == (sh, sw):
                            return [cand]
                    return [same_size[0]]

            # 3) positional fallback
            pos = index_map.get(src_fp, None)
            if pos is not None and 0 <= pos < len(candidates):
                return [candidates[pos]]

            # 4) nearest-size (kept for rescale=True)
            if sh and sw:
                def _diff(t):
                    th, tw = _eff_size(t)
                    return abs((th or 0) - sh) + abs((tw or 0) - sw)
                best = min(candidates, key=_diff)
                if broadcast_if_ambiguous:
                    bh, bw = _eff_size(best)
                    equal_best = [t for t in candidates if _eff_size(t) == (bh, bw)]
                    if len(equal_best) > 1:
                        return equal_best
                return [best]

            return [candidates[0]]

        # ---------- build source/target sets ----------
        src_ms_files = list(self.multispectral_image_data_groups.get(source_root, []))
        src_th_root  = _paired_thermal_root(source_root)
        src_th_files = list(self.thermal_rgb_image_data_groups.get(src_th_root, [])) if src_th_root else []

        tgt_ms_files = list(self.multispectral_image_data_groups.get(target_root, []))
        tgt_th_root  = _paired_thermal_root(target_root)
        tgt_th_files = list(self.thermal_rgb_image_data_groups.get(tgt_th_root, [])) if tgt_th_root else []

        if (not src_ms_files and not src_th_files) or (not tgt_ms_files and not tgt_th_files):
            print(f"[copy_polygons_between_roots] Nothing to copy: "
                  f"src(ms={len(src_ms_files)},th={len(src_th_files)}), "
                  f"tgt(ms={len(tgt_ms_files)},th={len(tgt_th_files)})")
            return

        src_ms_index = {fp: i for i, fp in enumerate(src_ms_files)}
        src_th_index = {fp: i for i, fp in enumerate(src_th_files)}

        # ---------- copy ----------
        copies_made = 0
        for group_name, polygons_by_file in list(self.all_polygons.items()):
            src_items = [(fp, pd) for fp, pd in polygons_by_file.items()
                         if fp in src_ms_files or fp in src_th_files]
            if not src_items:
                continue

            for src_fp, poly_data in src_items:
                if src_fp in src_ms_files:
                    target_candidates = tgt_ms_files
                    index_map = src_ms_index
                else:
                    target_candidates = tgt_th_files
                    index_map = src_th_index
                if not target_candidates:
                    continue

                target_fps = _best_targets_for_src(src_fp, target_candidates, index_map)
                if not target_fps:
                    continue

                # source effective size (post-.ax) once per source file
                sh, sw = _eff_size(src_fp)

                # normalize to IMAGE coords for source, and remember the COORDINATE BASIS of img_points
                pts_any     = poly_data.get("points", []) or []
                shape_type  = poly_data.get("type", "polygon")
                name        = poly_data.get("name", group_name)
                coord_space = (poly_data.get("coord_space", "scene") or "scene").lower()

                pts_basis_w = pts_basis_h = None  # width/height of the system that img_points are in

                if coord_space == "image":
                    img_points = [(float(x), float(y)) for (x, y) in pts_any]
                    ref = poly_data.get("image_ref_size") or {}
                    ref_w = int(ref.get('w') or 0)
                    ref_h = int(ref.get('h') or 0)

                    if ref_w and ref_h and sw and sh and (ref_w != sw or ref_h != sh):
                        # bring into SOURCE EFFECTIVE basis
                        sx_src = float(sw) / float(ref_w)
                        sy_src = float(sh) / float(ref_h)
                        img_points = [(x * sx_src, y * sy_src) for (x, y) in img_points]
                        pts_basis_w, pts_basis_h = sw, sh
                    else:
                        # still in whatever ref they were saved in
                        if ref_w and ref_h:
                            pts_basis_w, pts_basis_h = ref_w, ref_h
                        else:
                            # last resort: use source effective or a live viewer basis
                            if sw and sh:
                                pts_basis_w, pts_basis_h = sw, sh
                            else:
                                vh, vw = _first_viewer_basis_for(src_fp)
                                pts_basis_h, pts_basis_w = vh, vw
                else:
                    # legacy scene: try mapping using source effective basis
                    if sh and sw and hasattr(self, "_map_points_scene_to_image"):
                        img_points = self._map_points_scene_to_image(
                            src_fp, pts_any, (sh, sw, 1), polygon_data=poly_data
                        )
                        img_points = [(float(x), float(y)) for (x, y) in img_points]
                        pts_basis_w, pts_basis_h = sw, sh
                    else:
                        # fallback: treat numbers as image coords and guess a basis
                        img_points = [(float(x), float(y)) for (x, y) in pts_any]
                        if sw and sh:
                            pts_basis_w, pts_basis_h = sw, sh
                        else:
                            ref = poly_data.get("image_ref_size") or {}
                            ref_w = int(ref.get('w') or 0)
                            ref_h = int(ref.get('h') or 0)
                            if ref_w and ref_h:
                                pts_basis_w, pts_basis_h = ref_w, ref_h
                            else:
                                vh, vw = _first_viewer_basis_for(src_fp)
                                pts_basis_h, pts_basis_w = vh, vw
                    coord_space = "image"

                # ensure usable basis; otherwise no-op scale
                if not (pts_basis_w and pts_basis_h):
                    pts_basis_w = sw or 0
                    pts_basis_h = sh or 0

                # scale into each target's EFFECTIVE basis
                for tgt_fp in target_fps:
                    if rescale:
                        th, tw = _eff_size(tgt_fp)
                        if not (tw and th):
                            vh, vw = _first_viewer_basis_for(tgt_fp)
                            if vh and vw:
                                th, tw = vh, vw
                            else:
                                oh, ow = _probe_image_hw(tgt_fp)
                                if oh and ow:
                                    th, tw = oh, ow

                        scaled_points = img_points
                        if (pts_basis_w and pts_basis_h) and (tw and th) and \
                           (pts_basis_w != tw or pts_basis_h != th):
                            sx = float(tw) / float(pts_basis_w)
                            sy = float(th) / float(pts_basis_h)
                            scaled_points = [(x * sx, y * sy) for (x, y) in img_points]

                        # persist with TARGET basis
                        self.all_polygons.setdefault(group_name, {})
                        self.all_polygons[group_name][tgt_fp] = {
                            'points': [(float(x), float(y)) for (x, y) in scaled_points],
                            'coord_space': 'image',
                            'image_ref_size': {'w': int(tw or 0), 'h': int(th or 0)},
                            'name': name,
                            'root': self.root_id_mapping.get(target_root, "0"),
                            'type': shape_type
                        }

                        # draw: target-eff -> viewer basis (optional for speed)
                        if not defer_viewer_update:
                            for tv in _all_viewers_for_filepath(tgt_fp):
                                vh, vw = self._viewer_basis_hw(tv, fallback_hw=(th, tw))
                                pts_for_viewer = scaled_points
                                if (tw and th and vw and vh) and (tw != vw or th != vh):
                                    sx_v, sy_v = float(vw)/float(tw), float(vh)/float(th)
                                    pts_for_viewer = [(x * sx_v, y * sy_v) for (x, y) in scaled_points]

                                scene_pts = [ self.image_to_scene_coords(tv, QtCore.QPointF(x, y))
                                              for (x, y) in pts_for_viewer ]
                                qpoly = QtGui.QPolygonF(scene_pts)
                                if shape_type == "point":
                                    (tv.add_point_to_scene if hasattr(tv,"add_point_to_scene") else tv.add_point)(qpoly, name)
                                else:
                                    (tv.add_polygon_to_scene if hasattr(tv,"add_polygon_to_scene") else tv.add_polygon)(qpoly, name)

                    else:
                        # -------- NO-RESCALE FAST PATH --------
                        ref_w, ref_h = int(pts_basis_w or 0), int(pts_basis_h or 0)
                        scaled_points = img_points  # 1:1

                        # persist using the **SOURCE basis**
                        self.all_polygons.setdefault(group_name, {})
                        self.all_polygons[group_name][tgt_fp] = {
                            'points': [(float(x), float(y)) for (x, y) in scaled_points],
                            'coord_space': 'image',
                            'image_ref_size': {'w': ref_w, 'h': ref_h},
                            'name': name,
                            'root': self.root_id_mapping.get(target_root, "0"),
                            'type': shape_type
                        }

                        # draw: source basis -> target viewer basis (optional for speed)
                        if not defer_viewer_update:
                            for tv in _all_viewers_for_filepath(tgt_fp):
                                vh, vw = self._viewer_basis_hw(tv, fallback_hw=(ref_h, ref_w))
                                pts_for_viewer = scaled_points
                                if (ref_w and ref_h and vw and vh) and (ref_w != vw or ref_h != vh):
                                    sx_v, sy_v = float(vw)/float(ref_w), float(vh)/float(ref_h)
                                    pts_for_viewer = [(x * sx_v, y * sy_v) for (x, y) in scaled_points]

                                scene_pts = [ self.image_to_scene_coords(tv, QtCore.QPointF(x, y))
                                              for (x, y) in pts_for_viewer ]
                                qpoly = QtGui.QPolygonF(scene_pts)
                                if shape_type == "point":
                                    (tv.add_point_to_scene if hasattr(tv,"add_point_to_scene") else tv.add_point)(qpoly, name)
                                else:
                                    (tv.add_polygon_to_scene if hasattr(tv,"add_polygon_to_scene") else tv.add_polygon)(qpoly, name)

                    copies_made += 1

        # Persist + refresh (gated for speed in bulk)
        if not defer_save:
            self.save_polygons_to_json(root_name=target_root)
        if not defer_viewer_update:
            self.update_polygon_manager()

        print(f"[copy_polygons_between_roots] Copied {copies_made} polygon-set(s) "
              f"{source_root} → {target_root} (MS:{len(tgt_ms_files)}, TRGB:{len(tgt_th_files)}; "
              f"rescale={rescale}, defer_viewer_update={defer_viewer_update}, defer_save={defer_save}).")

    def copy_polygons_to_next_tab(self):
        """
        Copy polygons from this ProjectTab to the NEXT ProjectTab (tab+1),
        using post-.ax effective sizes as the canonical image basis.

        Behavior:
          • scene -> source-image(viewer basis) -> source-image(EFFECTIVE basis) ->
            target-image(EFFECTIVE basis) -> target-scene(viewer basis)
          • Persist polygons in IMAGE coords with image_ref_size = TARGET effective size
          • Robust to per-tab .ax rotations/resizes and differing viewer pixmap sizes
        """
        import logging
        from functools import lru_cache
        from PyQt5 import QtWidgets, QtGui, QtCore

        parent = self.tab_widget
        if not isinstance(parent, QtWidgets.QTabWidget):
            logging.error("Provided tab_widget is not a QTabWidget. Cannot copy polygons to next tab.")
            QtWidgets.QMessageBox.warning(self, "Copy Error", "Invalid tab manager. Cannot copy polygons.")
            return

        current_index = parent.indexOf(self)
        if current_index == -1:
            logging.error("Current tab not found in the QTabWidget.")
            QtWidgets.QMessageBox.warning(self, "Copy Error", "Current tab not found.")
            return

        if current_index >= parent.count() - 1:
            QtWidgets.QMessageBox.warning(self, "Copy Error", "Already at the last tab. No next tab to copy polygons to.")
            logging.warning("Attempted to copy polygons but already at the last tab.")
            return

        next_tab = parent.widget(current_index + 1)
        # Allow ProjectTab subclassing; just require expected attrs
        if not hasattr(next_tab, "viewer_widgets") or not hasattr(next_tab, "all_polygons"):
            logging.error("Next tab is not a compatible ProjectTab.")
            QtWidgets.QMessageBox.warning(self, "Copy Error", "The next tab is not a valid ProjectTab.")
            return

        source_viewers = self.viewer_widgets
        target_viewers = next_tab.viewer_widgets
        if not source_viewers or not target_viewers:
            QtWidgets.QMessageBox.information(self, "No Viewers", "One of the tabs has no viewers to copy.")
            logging.info("Copy aborted: empty viewers list in source or target.")
            return

        pair_count = min(len(source_viewers), len(target_viewers))
        if pair_count == 0:
            QtWidgets.QMessageBox.information(self, "Nothing To Copy", "No matching viewer pairs were found.")
            logging.info("Copy aborted: pair_count == 0.")
            return

        def _has_any_polygons(vwidgets):
            for i in range(pair_count):
                vw = vwidgets[i]['viewer']
                try:
                    if vw.get_all_polygons():
                        return True
                except Exception:
                    pass
            return False

        if not _has_any_polygons(source_viewers):
            QtWidgets.QMessageBox.information(self, "No Polygons", "There are no polygons to copy in the paired source viewers.")
            logging.info("No polygons found to copy in paired source viewers.")
            return

        if len(source_viewers) != len(target_viewers):
            logging.info(
                "Viewer count mismatch: source=%d, target=%d. Proceeding to copy first %d pairs in order.",
                len(source_viewers), len(target_viewers), pair_count
            )

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Copy",
            (
                f"Copy polygons from '{self.project_name}' to '{next_tab.project_name}'?\n\n"
                f"Source viewers: {len(source_viewers)}\n"
                f"Target viewers: {len(target_viewers)}\n"
                f"Pairs to copy (in order): {pair_count}"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.No:
            logging.info("User canceled the polygon copying operation.")
            return

        # -------- helpers --------

        @lru_cache(maxsize=512)
        def _eff_size_src(fp):
            """(H,W) AFTER .ax for SOURCE, resolved from *this* tab first."""
            h = w = None

            if hasattr(self, "_size_after_ax_fast"):
                try:
                    h, w = self._size_after_ax_fast(fp)
                except Exception:
                    h = w = None

            if not h or not w:
                for wdict in (self.viewer_widgets or []):
                    idata = wdict.get("image_data")
                    if idata is not None and getattr(idata, "filepath", None) == fp:
                        try:
                            hh, ww = idata.image.shape[:2]
                            h, w = int(hh), int(ww)
                            break
                        except Exception:
                            pass

            if not h or not w:
                getimg = getattr(self, "_get_export_image", None)
                if callable(getimg):
                    try:
                        img, _ = getimg(fp)
                        if img is not None:
                            h, w = img.shape[:2]
                    except Exception:
                        h = w = None

            return (h, w)

        @lru_cache(maxsize=512)
        def _eff_size_tgt(fp):
            """(H,W) AFTER .ax for TARGET, resolved from *next_tab* first."""
            h = w = None

            if hasattr(next_tab, "_size_after_ax_fast"):
                try:
                    h, w = next_tab._size_after_ax_fast(fp)
                except Exception:
                    h = w = None

            if not h or not w:
                for wdict in (getattr(next_tab, "viewer_widgets", []) or []):
                    idata = wdict.get("image_data")
                    if idata is not None and getattr(idata, "filepath", None) == fp:
                        try:
                            hh, ww = idata.image.shape[:2]
                            h, w = int(hh), int(ww)
                            break
                        except Exception:
                            pass

            if not h or not w:
                getimg = getattr(next_tab, "_get_export_image", None)
                if callable(getimg):
                    try:
                        img, _ = getimg(fp)
                        if img is not None:
                            h, w = img.shape[:2]
                    except Exception:
                        h = w = None

            if not h or not w:
                getimg = getattr(self, "_get_export_image", None)
                if callable(getimg):
                    try:
                        img, _ = getimg(fp)
                        if img is not None:
                            h, w = img.shape[:2]
                    except Exception:
                        h = w = None

            return (h, w)

        def _basis(viewer, *, fallback_hw=None):
            """
            Viewer image basis (H,W) that image_to_scene_coords expects.
            Uses self._viewer_basis_hw if present; otherwise tries image_data.image,
            then the displayed QPixmap; finally falls back to fallback_hw.
            """
            if hasattr(self, "_viewer_basis_hw"):
                try:
                    return self._viewer_basis_hw(viewer, fallback_hw=fallback_hw)
                except Exception:
                    pass

            try:
                idata = getattr(viewer, "image_data", None)
                if idata is not None and getattr(idata, "image", None) is not None:
                    h, w = idata.image.shape[:2]
                    return (int(h), int(w))
            except Exception:
                pass
            try:
                pitem = getattr(viewer, "pixmap_item", None) or getattr(viewer, "_pixmap_item", None)
                if pitem is not None:
                    pm = pitem.pixmap()
                    if not pm.isNull():
                        return (int(pm.height()), int(pm.width()))
            except Exception:
                pass
            if fallback_hw and all(fallback_hw):
                return (int(fallback_hw[0]), int(fallback_hw[1]))
            return (None, None)

        def _infer_shape_type(poly_item, default="polygon"):
            val = getattr(poly_item, "shape_type", None)
            if isinstance(val, str):
                return val
            is_point_flag = getattr(poly_item, "is_point", None)
            if isinstance(is_point_flag, bool) and is_point_flag:
                return "point"
            try:
                qpoly = getattr(poly_item, "polygon", None)
                if qpoly is not None and int(qpoly.count()) == 1:
                    return "point"
            except Exception:
                pass
            return default

        copied = 0

        for i in range(pair_count):
            source_widget = source_viewers[i]
            target_widget = target_viewers[i]

            source_viewer = source_widget['viewer']
            target_viewer = target_widget['viewer']

            # Filepaths
            src_fp = source_widget.get('image_data').filepath if source_widget.get('image_data') else None
            tgt_fp = target_widget.get('image_data').filepath if target_widget.get('image_data') else None
            if not src_fp or not tgt_fp:
                continue

            # Effective sizes (AFTER .ax) — resolved per tab
            sh, sw = _eff_size_src(src_fp)
            th, tw = _eff_size_tgt(tgt_fp)

            # Viewer bases (use robust basis → may fall back to effective)
            svh, svw = _basis(source_viewer, fallback_hw=(sh, sw))
            tvh, tvw = _basis(target_viewer, fallback_hw=(th, tw))

            try:
                polygons = source_viewer.get_all_polygons()
            except Exception as e:
                logging.error("get_all_polygons failed for source viewer %d: %s", i, e)
                continue
            if not polygons:
                continue

            for polygon_item in polygons:
                # Source scene points
                try:
                    src_scene_points = [p for p in polygon_item.polygon]
                except Exception:
                    src_scene_points = []
                    qpoly = getattr(polygon_item, "polygon", None)
                    if qpoly is not None:
                        for k in range(qpoly.count()):
                            pt = qpoly.at(k)
                            src_scene_points.append(pt)

                # scene -> image (source viewer basis)
                src_img_pts_view = []
                for pt in src_scene_points:
                    try:
                        ip = self.scene_to_image_coords(source_viewer, QtCore.QPointF(pt.x(), pt.y()))
                    except Exception:
                        ip = QtCore.QPointF(pt.x(), pt.y())
                    src_img_pts_view.append(ip)

                # source viewer basis -> SOURCE EFFECTIVE basis
                src_img_pts_eff = []
                if svw and svh and sw and sh and (svw != sw or svh != sh):
                    sx0 = float(sw) / float(svw)
                    sy0 = float(sh) / float(svh)
                    for p in src_img_pts_view:
                        src_img_pts_eff.append((p.x() * sx0, p.y() * sy0))
                else:
                    src_img_pts_eff = [(p.x(), p.y()) for p in src_img_pts_view]

                # SOURCE EFFECTIVE -> TARGET EFFECTIVE basis (with viewer-basis fallback)
                tgt_img_pts_eff = []
                if sw and sh and tw and th and (sw != tw or sh != th):
                    sx = float(tw) / float(sw)
                    sy = float(th) / float(sh)
                elif svw and svh and tvw and tvh and (svw != tvw or svh != tvh):
                    sx = float(tvw) / float(svw)
                    sy = float(tvh) / float(svh)
                else:
                    sx = sy = 1.0

                for (x, y) in src_img_pts_eff:
                    tgt_img_pts_eff.append((x * sx, y * sy))

                # Persist in next_tab backing store — IMAGE coords with target reference size
                group_name = getattr(polygon_item, "name", "polygon")
                shape_type = _infer_shape_type(polygon_item, default="polygon")

                if group_name not in next_tab.all_polygons:
                    next_tab.all_polygons[group_name] = {}

                src_group = self.all_polygons.get(group_name, {})
                src_meta = src_group.get(src_fp, {}) if src_fp else {}

                ref_w = int(tw or tvw or 0)
                ref_h = int(th or tvh or 0)

                next_tab.all_polygons[group_name][tgt_fp] = {
                    'points': [(float(x), float(y)) for (x, y) in tgt_img_pts_eff],
                    'coord_space': 'image',
                    'image_ref_size': {'w': ref_w, 'h': ref_h},
                    'name': str(group_name),
                    'root': str(src_meta.get('root', '0')),
                    'type': str(shape_type),
                    'coordinates': {"latitude": None, "longitude": None}
                }

                # TARGET EFFECTIVE -> target viewer basis -> scene (draw)
                pts_for_view = tgt_img_pts_eff
                if tw and th and tvw and tvh and (tw != tvw or th != tvh):
                    sxv = float(tvw) / float(tw)
                    syv = float(tvh) / float(th)
                    pts_for_view = [(x * sxv, y * syv) for (x, y) in tgt_img_pts_eff]

                scene_pts = []
                for (x, y) in pts_for_view:
                    try:
                        sp = self.image_to_scene_coords(target_viewer, QtCore.QPointF(x, y))
                    except Exception:
                        sp = QtCore.QPointF(x, y)
                    scene_pts.append(sp)

                qpoly = QtGui.QPolygonF(scene_pts)
                if shape_type == "point":
                    if hasattr(target_viewer, "add_point_to_scene"):
                        target_viewer.add_point_to_scene(qpoly, group_name)
                    elif hasattr(target_viewer, "add_point"):
                        target_viewer.add_point(qpoly, group_name)
                    else:
                        target_viewer.add_polygon(qpoly, group_name)
                else:
                    if hasattr(target_viewer, "add_polygon_to_scene"):
                        target_viewer.add_polygon_to_scene(qpoly, group_name)
                    else:
                        target_viewer.add_polygon(qpoly, group_name)

                copied += 1

        # Save + refresh next tab
        try:
            target_root = None
            if hasattr(next_tab, "multispectral_root_names") and hasattr(next_tab, "current_root_index"):
                try:
                    target_root = next_tab.multispectral_root_names[next_tab.current_root_index]
                except Exception:
                    target_root = None

            if hasattr(next_tab, "save_polygons_to_json"):
                next_tab.save_polygons_to_json(root_name=target_root)
        except Exception as e:
            logging.error("Failed to save polygons to JSON in target tab '%s': %s", getattr(next_tab, "project_name", "?"), e)
            QtWidgets.QMessageBox.critical(self, "Save Error", f"An error occurred while saving polygons to JSON:\n{e}")
            return

        try:
            if hasattr(next_tab, "refresh_viewer"):
                if target_root:
                    next_tab.refresh_viewer(root_name=target_root)
                else:
                    next_tab.refresh_viewer()
        except Exception as e:
            logging.error("Failed to refresh viewer in target tab '%s': %s", getattr(next_tab, "project_name", "?"), e)
            QtWidgets.QMessageBox.warning(self, "Refresh Warning", f"An error occurred while refreshing the viewer in target tab:\n{e}")

        logging.info("Copied %d polygon set(s) to next tab with effective-size + viewer-basis semantics.", copied)




    def copy_polygons_to_all_roots(self):
        """
        Much faster bulk copy:
          - Cancelable progress dialog with ETA
          - Defers per-target viewer updates and saves
          - Freezes repaints during the batch
          - Optional prewarm of effective-size cache when rescaling
          - One final refresh + quick save
        """
        from PyQt5 import QtWidgets, QtCore
        import logging, time

        # current root
        try:
            current_root = self.get_current_root_name() if hasattr(self, "get_current_root_name") else None
        except Exception:
            current_root = None
        if not current_root:
            QtWidgets.QMessageBox.warning(self, "No Active Root", "Could not determine the current root.")
            return

        ms_roots = set(self.multispectral_image_data_groups.keys()) if getattr(self, "multispectral_image_data_groups", None) else set()
        th_roots = set(self.thermal_rgb_image_data_groups.keys())     if getattr(self, "thermal_rgb_image_data_groups", None) else set()
        all_roots = sorted(ms_roots | th_roots)

        targets = [r for r in all_roots if r != current_root]
        if not targets:
            QtWidgets.QMessageBox.information(self, "Nothing To Copy", "There are no other roots to copy to.")
            return

        # ensure pixmap_size tagging from visible viewers
        try:
            for vw in getattr(self, "viewer_widgets", []) or []:
                viewer = vw.get("viewer", None)
                if viewer is not None and hasattr(self, "update_current_polygons_pixmap_size"):
                    self.update_current_polygons_pixmap_size(viewer)
        except Exception as e:
            logging.debug(f"Could not tag pixmap_size on visible viewers: {e}")

        # confirm
        resp = QtWidgets.QMessageBox.question(
            self, "Copy Polygons to All Roots",
            f"Copy polygons from root '{current_root}' to {len(targets)} other root(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        # fast vs rescale
        res = QtWidgets.QMessageBox.question(
            self, "Copy mode",
            "Use FAST copy (no rescale)?\n\n"
            "Yes — Fast (recommended when all roots share the same resolution)\n"
            "No  — Rescale per target (slower; use if resolutions differ)\n\n"
            "Cancel to abort.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        if res == QtWidgets.QMessageBox.Cancel:
            return
        do_rescale = (res == QtWidgets.QMessageBox.No)  # Yes=fast(no rescale), No=rescale

        # Optional prewarm when rescaling (skips heavy size IO inside the loop)
        if do_rescale:
            try:
                all_files = []
                for g in (self.multispectral_image_data_groups or {}).values():
                    all_files.extend(list(g))
                for g in (self.thermal_rgb_image_data_groups or {}).values():
                    all_files.extend(list(g))
                # dedupe
                all_files = list({f for f in all_files})
                self._prewarm_eff_size_cache(all_files, parent=self, label="Analyzing image sizes…")
            except Exception as e:
                logging.debug(f"Prewarm skipped: {e}")

        # cancelable progress
        progress = QtWidgets.QProgressDialog("Preparing…", "Cancel", 0, len(targets), self)
        progress.setWindowTitle("Copying Polygons")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        app = QtWidgets.QApplication.instance()

        # freeze repaints
        frozen = []
        try:
            if hasattr(self, "setUpdatesEnabled"):
                self.setUpdatesEnabled(False); frozen.append(self)
            for vw in (getattr(self, "viewer_widgets", []) or []):
                w = vw.get("viewer", None) if isinstance(vw, dict) else None
                if w is not None and hasattr(w, "setUpdatesEnabled"):
                    w.setUpdatesEnabled(False); frozen.append(w)
        except Exception:
            pass
        try:
            app.setOverrideCursor(QtCore.Qt.WaitCursor)
        except Exception:
            pass

        # bulk loop (defer heavy stuff)
        successes = failures = 0
        t0 = time.perf_counter()
        try:
            for i, tgt in enumerate(targets, 1):
                if progress.wasCanceled():
                    break

                done = i - 1
                elapsed = max(1e-6, time.perf_counter() - t0)
                eta = "" if done == 0 else f"  •  ETA ~ {int((len(targets)-done) * (elapsed/done))} s"
                progress.setLabelText(f"Copying {current_root} → {tgt} ({i}/{len(targets)})…{eta}")
                progress.setValue(i - 1)
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)

                try:
                    self.copy_polygons_between_roots(
                        current_root, tgt,
                        rescale=do_rescale,
                        defer_viewer_update=True,    # <<< speed
                        defer_save=True              # <<< speed
                    )
                    successes += 1
                except Exception as e:
                    logging.error(f"Copy polygons {current_root} -> {tgt} failed: {e}")
                    failures += 1

                progress.setValue(i)
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        finally:
            for w in frozen:
                try: w.setUpdatesEnabled(True)
                except Exception: pass
            try: app.restoreOverrideCursor()
            except Exception: pass

        if progress.wasCanceled():
            progress.cancel()
            QtWidgets.QMessageBox.information(self, "Copy Cancelled",
                f"Stopped after {successes} success(es), {failures} failure(s).")
            return

        progress.setLabelText("Finalizing…")
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)

        # one save + one refresh
        try:
            if hasattr(self, "save_project_quick"):
                self.save_project_quick(skip_recompute=True)
            elif hasattr(self, "save_polygons_to_json"):
                self.save_polygons_to_json()
        except Exception as e:
            logging.debug(f"Quick-save after bulk copy failed: {e}")

        try:
            if hasattr(self, "update_polygon_manager"):
                self.update_polygon_manager()
            if hasattr(self, "refresh_viewer"):
                self.refresh_viewer(root_name=current_root)
        except Exception:
            pass

        progress.close()
        if failures == 0:
            QtWidgets.QMessageBox.information(self, "Copy Complete",
                f"Polygons copied from '{current_root}' to {successes} root(s).")
        else:
            QtWidgets.QMessageBox.warning(self, "Copy Finished with Errors",
                f"Copied to {successes} root(s); {failures} failed. See log for details.")


    def _prewarm_eff_size_cache(self, filepaths, parent=None, label="Prewarming…"):
        """
        Fill self._copy_size_cache ('eff4' entries) for filepaths to avoid repeated
        .ax/header IO in copy loops. Safe to call even if cache already exists.
        """
        import os, json
        from PyQt5 import QtWidgets, QtCore

        # local copies of helpers to match the ones in copy_polygons_between_roots
        def _ax_path_for(fp: str) -> str:
            sidecar = os.path.splitext(fp)[0] + ".ax"
            if os.path.exists(sidecar):
                return sidecar
            folder = (self.project_folder or "").strip() or os.path.dirname(fp)
            return os.path.join(folder, os.path.splitext(os.path.basename(fp))[0] + ".ax")

        def _probe_image_hw(fp):
            try:
                for w in getattr(self, 'viewer_widgets', []) or []:
                    idata = w.get('image_data')
                    if idata is not None and getattr(idata, 'filepath', None) == fp:
                        img = getattr(idata, 'image', None)
                        if img is not None:
                            h, w = img.shape[:2]
                            return int(h), int(w)
            except Exception:
                pass
            try:
                img, _ = self._get_export_image(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            try:
                import numpy as np, cv2
                data = np.fromfile(fp, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    return int(h), int(w)
            except Exception:
                pass
            return (None, None)

        def _parse_op_order(ax: dict):
            raw = ax.get("op_order") or ax.get("ops") or ["rotate", "crop", "resize"]
            if isinstance(raw, str):
                raw = [raw]
            out = []
            for s in (raw or []):
                t = str(s).lower()
                if "rot" in t: out.append("rotate")
                elif "crop" in t: out.append("crop")
                elif "siz" in t or "res" in t: out.append("resize")
                elif "band" in t: out.append("band_expression")
            return [op for op in out if op in ("rotate","crop","resize","band_expression")] or ["rotate","crop","resize"]

        def _size_after_ax_fast_from_file(fp):
            axH = axW = None
            try:
                ax_path = _ax_path_for(fp)
                if os.path.exists(ax_path):
                    with open(ax_path, "r", encoding="utf-8") as f:
                        ax = json.load(f) or {}
                    o = ax.get("orig_size") or {}
                    H0 = int(o.get("h") or 0); W0 = int(o.get("w") or 0)
                    if not (H0 and W0):
                        obsH, obsW = _probe_image_hw(fp)
                        if obsH and obsW:
                            H0, W0 = obsH, obsW
                    if not (H0 and W0):
                        return (None, None)

                    try: rot = int(ax.get("rotate", 0)) % 360
                    except Exception: rot = 0
                    crop     = ax.get("crop_rect") or None
                    crop_ref = ax.get("crop_rect_ref_size") or None
                    resize   = ax.get("resize") or None
                    op_order = _parse_op_order(ax)

                    curH, curW = int(H0), int(W0)
                    if rot in (90, 270): curH, curW = curW, curH
                    if crop:
                        refW = int((crop_ref or {}).get("w") or curW)
                        refH = int((crop_ref or {}).get("h") or curH)
                        if refW and refH:
                            rw = max(0, int(crop.get("width" , 0)))
                            rh = max(0, int(crop.get("height", 0)))
                            newW = max(1, int(round(rw * (curW / float(refW)))))
                            newH = max(1, int(round(rh * (curH / float(refH)))))
                            curW, curH = int(newW), int(newH)
                    if isinstance(resize, dict) and resize:
                        if "scale" in resize:
                            s = max(1, int(resize.get("scale", 100))) / 100.0
                            curW = max(1, int(round(curW * s)))
                            curH = max(1, int(round(curH * s)))
                        else:
                            pw = max(1, int(resize.get("width", 100))) / 100.0
                            ph = max(1, int(resize.get("height", 100))) / 100.0
                            curW = max(1, int(round(curW * pw)))
                            curH = max(1, int(round(curH * ph)))
                    axH, axW = int(curH), int(curW)
            except Exception:
                axH = axW = None

            obsH, obsW = _probe_image_hw(fp)
            def _far(a, b, tol=0.05):
                if not a or not b: return False
                return abs(a - b) / float(max(a, b)) > tol
            if axH and axW and obsH and obsW:
                if _far(axH, obsH) or _far(axW, obsW):
                    return (obsH, obsW)
                return (axH, axW)
            if axH and axW:  return (axH, axW)
            if obsH and obsW: return (obsH, obsW)
            return (None, None)

        # cache structure reused by copy_polygons_between_roots
        if getattr(self, "_copy_size_cache", None) is None:
            self._copy_size_cache = {}

        prog = None
        if parent:
            prog = QtWidgets.QProgressDialog(label, "Skip", 0, len(filepaths), parent)
            prog.setWindowModality(QtCore.Qt.WindowModal)
            prog.setMinimumDuration(0)
            prog.show()

        import os
        cache = self._copy_size_cache
        for i, fp in enumerate(filepaths, 1):
            if prog and prog.wasCanceled(): break

            axp = _ax_path_for(fp)
            try:
                ax_mtime = os.path.getmtime(axp) if axp and os.path.exists(axp) else None
                ax_size  = os.path.getsize(axp)  if axp and os.path.exists(axp) else None
            except Exception:
                ax_mtime = ax_size = None
            try:
                im_mtime = os.path.getmtime(fp)
                im_size  = os.path.getsize(fp)
            except Exception:
                im_mtime = im_size = None

            ax_id = (axp, ax_mtime, ax_size)
            im_id = (im_mtime, im_size)
            key = ("eff4", fp)

            rec = cache.get(key)
            if not (rec and rec[2] == ax_id and rec[3] == im_id):
                h, w = _size_after_ax_fast_from_file(fp)
                cache[key] = (h, w, ax_id, im_id)

            if prog:
                prog.setValue(i)
                QtWidgets.QApplication.processEvents()
        if prog:
            prog.close()





    def copy_polygons_to_previous(self):
        if self.current_root_index > 0:
            current_root = self.multispectral_root_names[self.current_root_index]
            prev_root = self.multispectral_root_names[self.current_root_index - 1]

            self.copy_polygons_between_roots(current_root, prev_root)
            # Removed duplicate call to copy_polygons_between_roots

            self.save_polygons_to_json(root_name=prev_root)

            # Refresh only the previous root's viewer
            #self.refresh_viewer(root_name=prev_root)

            logging.info(f"Copied polygons from '{current_root}' to '{prev_root}' and refreshed the viewer.")
        else:
            QMessageBox.information(self, "Operation Not Allowed", "Already at the first root group.")
            logging.info("Attempted to copy polygons to previous root but already at the first root group.")


     
    def copy_polygons_to_next(self):
        if self.current_root_index < len(self.multispectral_root_names) - 1:
            current_root = self.multispectral_root_names[self.current_root_index]
            next_root = self.multispectral_root_names[self.current_root_index + 1]

            self.copy_polygons_between_roots(current_root, next_root)
            # Removed duplicate call to copy_polygons_between_roots

            self.save_polygons_to_json(root_name=next_root)

            # Refresh only the next root's viewer
            #self.refresh_viewer(root_name=next_root)

            logging.info(f"Copied polygons from '{current_root}' to '{next_root}' and refreshed the viewer.")
        else:
            QMessageBox.information(self, "Operation Not Allowed", "Already at the last root group.")
            logging.info("Attempted to copy polygons to next root but already at the last root group.")



    def update_current_polygons_pixmap_size(self, viewer):
        """
        Store the display pixmap size alongside saved shapes for robust offline mapping.
        """
        if viewer is None or viewer._image is None:
            return
        pixmap = viewer._image.pixmap()
        pm_size = (pixmap.width(), pixmap.height())
        filepath = getattr(getattr(viewer, "image_data", None), "filepath", None)
        if not filepath:
            return
        for group_name, files in self.all_polygons.items():
            if filepath in files:
                files[filepath]["pixmap_size"] = list(pm_size)
               
                try:
                    exp_img, _meta = self._get_export_image(filepath)
                    if exp_img is not None:
                        h, w = exp_img.shape[:2]
                        files[filepath]["image_size"] = [w, h]   # (W,H) to match how scale above
                except Exception:
                    pass




    def _points_scene_to_norm_pixmap(self, viewer, pts_scene):
        """
        Map scene points -> normalized pixmap coords (u,v in [0,1]) using the viewer's pixmap item.
        """
        from PyQt5 import QtCore
        if viewer is None or getattr(viewer, "_image", None) is None:
            return []
        pm = viewer._image.pixmap()
        pw, ph = float(max(1, pm.width())), float(max(1, pm.height()))
        out = []
        for (x, y) in pts_scene:
            p_item = viewer._image.mapFromScene(QtCore.QPointF(float(x), float(y)))
            u = p_item.x() / pw
            v = p_item.y() / ph
            out.append((u, v))
        return out


    def _points_norm_pixmap_to_scene(self, viewer, uv_list):
        """
        Map normalized pixmap coords (u,v) -> scene points using the viewer's pixmap item.
        """
        from PyQt5 import QtCore
        if viewer is None or getattr(viewer, "_image", None) is None:
            return []
        pm = viewer._image.pixmap()
        pw, ph = float(max(1, pm.width())), float(max(1, pm.height()))
        out = []
        for (u, v) in uv_list:
            p_item = QtCore.QPointF(float(u) * pw, float(v) * ph)
            p_scene = viewer._image.mapToScene(p_item)
            out.append((float(p_scene.x()), float(p_scene.y())))
        return out


    def _with_root_loaded(self, root_name):
        """
        Context manager that ensures `root_name` is displayed (viewers exist),
        then restores the previous root afterward.
        """
        import contextlib
        @contextlib.contextmanager
        def _mgr():
            prev = self.get_current_root_name()
            if prev and prev != root_name:
                try:
                    self.save_polygons_to_json(root_name=prev)
                except Exception:
                    pass
                self.load_image_group(root_name)
            try:
                yield
            finally:
                cur = self.get_current_root_name()
                if prev and cur != prev:
                    try:
                        self.save_polygons_to_json(root_name=root_name)
                    except Exception:
                        pass
                    self.load_image_group(prev)
        return _mgr()



    def add_polygon_to_other_images(self, source_viewer, polygon, logical_name="", action="copy", shape_type="polygon"):
        """
        Adds or moves polygons/points to other images in the same root.
        Adjusts the polygon/point coordinates based on image sizes.
        
        Parameters:
          source_viewer: The viewer where the shape was drawn.
          polygon: A QPolygonF containing the shape's points.
          logical_name: The logical name for the shape.
          action: (Optional) Action type, default "copy".
          shape_type: "polygon" or "point" to determine which type of shape to add.
        """
        # Get the root name for the source viewer's image.
        source_filepath = source_viewer.image_data.filepath
        root_name = self.get_root_by_filepath(source_filepath)
        if not root_name:
            return

        # Get all image filepaths in the same root.
        all_filepaths_in_root = self.multispectral_image_data_groups.get(root_name, []) + \
                                  self.thermal_rgb_image_data_groups.get(root_name, [])

        # Get source image dimensions.
        source_image = source_viewer.image_data.image
        source_height, source_width = source_image.shape[:2]

        # Determine root number.
        try:
            root_number = str(self.root_names.index(root_name) + 1)
        except ValueError:
            root_number = "0"

        for filepath in all_filepaths_in_root:
            if filepath == source_filepath:
                continue  # Skip the source image

            viewer = self.get_viewer_by_filepath(filepath)
            if viewer:
                # Get target image dimensions.
                target_image = viewer.image_data.image
                target_height, target_width = target_image.shape[:2]

                # Calculate scaling factors.
                scale_x = target_width / source_width
                scale_y = target_height / source_height

                # Adjust polygon/point coordinates for the target image.
                adjusted_polygon = QtGui.QPolygonF()
                for point in polygon:
                    adjusted_x = point.x() * scale_x
                    adjusted_y = point.y() * scale_y
                    adjusted_polygon.append(QtCore.QPointF(adjusted_x, adjusted_y))

                # Determine if the target image is RGB 
                is_rgb = False
                if len(target_image.shape) == 3 and target_image.shape[2] == 3:
                    is_rgb = True

                # Copy the adjusted shape to the viewer based on shape type.
                if shape_type == "point":
                    viewer.add_point_to_scene(adjusted_polygon, logical_name)
                else:
                    viewer.add_polygon_to_scene(adjusted_polygon, logical_name)

                # Update the all_polygons structure with the adjusted points, root number, and shape type.
                if logical_name not in self.all_polygons:
                    self.all_polygons[logical_name] = {}
                self.all_polygons[logical_name][filepath] = {
                    'points': [(point.x(), point.y()) for point in adjusted_polygon],
                    'name': logical_name,
                    'root': root_number,
                    'type': shape_type  # Added to record whether it's a polygon or point.
                }

    @staticmethod
    def apply_aux_modifications(image_filepath, image, project_folder=None, global_mode=False):
        """
        Apply edits recorded in the .ax using the SAME order the editor saved:
        supports any permutation of rotate/crop/resize (+ optional band_expression).
        Keeps scientific magnitudes; no histogram/CLAHE here.
        """
        import os, json, logging, cv2, numpy as np

        # ---- locate .ax ------------------------------------------------------------
        if global_mode:
            mod_filename = (
                os.path.join(project_folder, "global.ax")
                if project_folder and project_folder.strip()
                else os.path.join(os.path.dirname(image_filepath), "global.ax")
            )
        else:
            base = os.path.splitext(os.path.basename(image_filepath))[0] + ".ax"
            mod_filename = (
                os.path.join(project_folder, base)
                if project_folder and project_folder.strip()
                else os.path.splitext(image_filepath)[0] + ".ax"
            )

        if not os.path.exists(mod_filename) or image is None or getattr(image, "size", 0) == 0:
            return image

        try:
            with open(mod_filename, "r", encoding="utf-8") as f:
                ax = json.load(f) or {}
        except Exception as e:
            logging.error(f"Failed to load aux modifications from {mod_filename}: {e}")
            return image

        # Work on a copy; keep dtype 
        result = np.array(image, copy=True)

        # ---- parse params ----------------------------------------------------------
        try:
            rot = int(ax.get("rotate", 0)) % 360
        except Exception:
            rot = 0
        crop_rect = ax.get("crop_rect") or None
        crop_ref  = ax.get("crop_rect_ref_size") or None
        resize    = ax.get("resize") or None
        expr      = (ax.get("band_expression") or "").strip()

        try:
            op_order = self._ax_op_order(ax)  # ['rotate','crop','resize'] filtered/ordered per ax
        except Exception:
            raw = ax.get("op_order") or ax.get("ops") or ["rotate", "crop", "resize"]
            if isinstance(raw, str): raw = [raw]
            op_order = []
            for s in (raw or []):
                t = str(s).lower()
                if "rot" in t: op_order.append("rotate")
                elif "crop" in t: op_order.append("crop")
                elif "siz" in t or "res" in t: op_order.append("resize")
            if not op_order:
                op_order = ["rotate", "crop", "resize"]

        # Helpers reused below 
        def _dims_after_rot(w0, h0, r):
            r = (int(r) // 90) % 4
            return (w0, h0) if r in (0, 2) else (h0, w0)

     
        # Determine how the crop rect was authored (RAW vs. after-rotate)
        raw_h, raw_w = result.shape[:2]
        crop_basis = _infer_crop_basis(ax, raw_w, raw_h, rot) if crop_rect else None

        # ---- ops mirroring the points pipeline ------------------------------------
        def _do_rotate():
            nonlocal result
            if rot in (90, 180, 270):
                try:
                    # NumPy rotation is channel-agnostic (HWC, any C)
                    # k: -1 = 90° CW, 2 = 180°, 1 = 90° CCW
                    k = {90: -1, 180: 2, 270: 1}[rot]
                    result = np.ascontiguousarray(np.rot90(result, k))
                    logging.debug(f"Applied rotation {rot}° (NumPy).")
                except Exception as e:
                    logging.warning(f"Rotation failed ({rot}°) via NumPy: {e}")


        def _do_crop():
            nonlocal result
            if not crop_rect:
                return

            Hc, Wc = result.shape[:2]

            # Saved ref dims (what frame the rect numbers are relative to)
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w_saved, ref_h_saved = int(crop_ref.get("w", raw_w)) or raw_w, int(crop_ref.get("h", raw_h)) or raw_h
            else:
                # If the UI saved after a rotate, the natural saved basis is rotated raw
                if crop_basis == "after_rotate":
                    ref_w_saved, ref_h_saved = _dims_after_rot(raw_w, raw_h, rot)
                else:
                    ref_w_saved, ref_h_saved = raw_w, raw_h

            rotate_applied = ("rotate" in op_order and "crop" in op_order and
                              op_order.index("rotate") < op_order.index("crop") and rot in (90, 180, 270))

            rect_to_apply = dict(crop_rect)
            if crop_basis == "after_rotate" and not rotate_applied and rot in (90, 180, 270):
                # Un-rotate the saved rect back to RAW basis
                rect_to_apply = _rect_after_rot(rect_to_apply, ref_w_saved, ref_h_saved, (360 - rot) % 360)
                ref_w_use, ref_h_use = raw_w, raw_h
            elif crop_basis == "pre_rotate" and rotate_applied and rot in (90, 180, 270):
                # Rotate the saved rect into the already-rotated basis
                rect_to_apply = _rect_after_rot(rect_to_apply, raw_w, raw_h, rot)
                ref_w_use, ref_h_use = _dims_after_rot(raw_w, raw_h, rot)
            else:
                ref_w_use, ref_h_use = ref_w_saved, ref_h_saved

            scaled = _scale_rect(rect_to_apply, ref_w_use, ref_h_use, Wc, Hc)
            x0, y0, ww, hh = scaled["x"], scaled["y"], scaled["width"], scaled["height"]

            if ww > 0 and hh > 0:
                result = result[y0:y0+hh, x0:x0+ww]
                logging.debug(f"Applied crop to {ww}x{hh} at ({x0},{y0}) in {Wc}x{Hc} frame (ref={ref_w_use}x{ref_h_use}).")
            else:
                logging.warning("Crop rect empty/out of bounds after remapping; skipping crop.")

        def _do_resize():
            nonlocal result
            if not isinstance(resize, dict) or not resize:
                return
            h0, w0 = result.shape[:2]
            if h0 <= 0 or w0 <= 0:
                return
            if "scale" in resize:
                s = float(resize.get("scale", 100.0)) / 100.0
                new_w = max(1, int(round(w0 * s)))
                new_h = max(1, int(round(h0 * s)))
            else:
                pw = float(resize.get("width", 100.0)) / 100.0
                ph = float(resize.get("height", 100.0)) / 100.0
                new_w = max(1, int(round(w0 * pw)))
                new_h = max(1, int(round(h0 * ph)))
            if new_w == w0 and new_h == h0:
                return
            sw = new_w / float(w0); sh = new_h / float(h0)
            if sw < 1.0 or sh < 1.0:
                interp = cv2.INTER_AREA
            elif max(sw, sh) < 2.0:
                interp = cv2.INTER_LINEAR
            else:
                interp = cv2.INTER_CUBIC
            result = resize_safe(result, new_w, new_h, interp)
            logging.debug(f"Applied resize to {new_w}x{new_h} (interp={interp}).")

        def _do_band_expr():
            nonlocal result
            expr_str = (ax.get("band_expression") or "").strip()
            if not expr_str:
                return
            try:
                res = process_band_expression_float(result, expr_str)
                if isinstance(res, np.ndarray):
                    # v94 behavior: show the expression as the image
                    result = res.astype(np.float32, copy=False)

                else:
                    result = res
            except Exception as e:
                logging.warning(f"Band expression failed ('{expr_str}'): {e}")

        # Execute in the exact order requested
        ops = {"rotate": _do_rotate, "crop": _do_crop, "resize": _do_resize}
        for op in op_order:
            if op in ops: ops[op]()
        # Expression (if any) always last
        _do_band_expr()

        return result


    def process_band_expression(image, expr):
        bands = re.findall(r'b(\d+)', expr)
        unique_bands = sorted(set(bands), key=lambda x: int(x))
        band_mapping = {}
        if len(image.shape) == 2:
            band_mapping['b1'] = image.astype(np.float32)
        elif len(image.shape) == 3:
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
                
    def load_image_group(self, root_name):
        """
        Re-entrant safe, cancelable multi-threaded loader (no loading bar).
        - Only one batch is active at a time ("single flight").
        - Rapid "next" presses coalesce; only the LAST requested root loads.
        - Old/stale results are ignored via a request token.
        """
        from PyQt5 import QtCore, QtWidgets
        import os, threading

        if getattr(self, "_load_busy", False):
            self._pending_root = root_name
            try:
                if self._load_stop_event is not None:
                    self._load_stop_event.set()
            except Exception:
                pass
            return

        # Fresh batch
        self._load_busy = True
        self._set_nav_enabled(False)
        self._pending_root = None

        # Token for this invocation; stale batches are ignored.
        self._load_token += 1
        tok = self._load_token

        # New cooperative cancel flag for this batch.
        stop_event = threading.Event()
        self._load_stop_event = stop_event

        # --- Resolve roots exactly as before ---
        try:
            multispectral_index = self.multispectral_root_names.index(root_name)
        except ValueError:
            print(f"Root name '{root_name}' not found in multispectral_root_names.")
            # mark idle and re-enable nav
            self._load_busy = False
            self._set_nav_enabled(True)
            return

        if self.mode == 'dual_folder':
            thermal_rgb_index = multispectral_index + self.root_offset
            if 0 <= thermal_rgb_index < len(self.thermal_rgb_root_names):
                thermal_rgb_root_name = self.thermal_rgb_root_names[thermal_rgb_index]
                thermal_rgb_image_paths = self.thermal_rgb_image_data_groups.get(thermal_rgb_root_name, [])
            else:
                thermal_rgb_image_paths = []
        else:
            thermal_rgb_image_paths = []

        ms_paths = sorted(self.multispectral_image_data_groups.get(root_name, []))
        all_paths = list(ms_paths)
        if self.mode == 'dual_folder':
            all_paths += sorted(thermal_rgb_image_paths)

        if not all_paths:
            # Still current?
            if tok == self._load_token:
                QtWidgets.QMessageBox.information(self, "No Images", "No images were found for this root.")
            self._load_busy = False
            self._set_nav_enabled(True)
            # pick up any queued request
            if self._pending_root and tok == self._load_token:
                nxt = self._pending_root
                self._pending_root = None
                QtCore.QTimer.singleShot(0, lambda: self.load_image_group(nxt))
            return

        if self.mode == 'dual_folder':
            self.image_data_groups[root_name] = ms_paths + list(thermal_rgb_image_paths)
        else:
            self.image_data_groups[root_name] = ms_paths

        # --- Thread pool config ---
        pool = QtCore.QThreadPool.globalInstance()
        try:
            max_workers = min(32, (os.cpu_count() or 4) + 4)  # IO-bound
            pool.setMaxThreadCount(max_workers)
        except Exception:
            pass

        # --- Prepare result holders ---
        results = {}           # index -> ImageData
        errors  = []           # (index, message)
        done_count = 0
        total = len(all_paths)

        indexed_paths = list(enumerate(all_paths))

        # Read once (thread-safe)
        gm = getattr(self, "global_mods_checkbox", None)
        global_mods = bool(gm.isChecked()) if gm else False

        # --- Slots (run on main thread) ---
        def on_result(idx, imgd):
            if tok != self._load_token:   # stale
                return
            results[idx] = imgd

        def on_error(idx, msg):
            if tok != self._load_token:   # stale
                return
            errors.append((idx, msg))

        def on_done_one():
            nonlocal done_count
            if tok != self._load_token:   # stale
                return
            done_count += 1
            if done_count >= total:
                finalize()

        finalized = {"done": False}
        def finalize():
            if finalized["done"]:
                return
            finalized["done"] = True

            # If a newer batch started, ignore this one entirely.
            if tok != self._load_token:
                return

            # Deterministic order
            ordered = [results[i] for i in sorted(results.keys())]

            # If not canceled, update UI
            if not stop_event.is_set():
                if not ordered and errors:
                    QtWidgets.QMessageBox.warning(
                        self, "Load Failed",
                        "All loads failed:\n" + "\n".join(m for _, m in errors[:10]) +
                        ("\n..." if len(errors) > 10 else "")
                    )
                else:
                    self.current_root_id = self.root_id_mapping.get(root_name, 0)
                    self.display_image_group(ordered, root_name)
                    self.polygon_manager.set_current_root(root_name, self.image_data_groups)
                    self._rewire_all_viewers_for_inspection()
                    if errors:
                        QtWidgets.QMessageBox.warning(
                            self, "Some Images Failed",
                            "\n".join(m for _, m in errors[:15]) + ("\n..." if len(errors) > 15 else "")
                        )

            # Mark idle and re-enable nav
            self._load_busy = False
            self._set_nav_enabled(True)

            # If user queued another root while loading, start it now (last one wins)
            if self._pending_root:
                nxt = self._pending_root
                self._pending_root = None
                QtCore.QTimer.singleShot(0, lambda: self.load_image_group(nxt))

        # --- Enqueue tasks ---
        for idx, fp in indexed_paths:
            worker = _ImageLoadRunnable(
                tab_ref=self,
                filepath=fp,
                index=idx,
                project_folder=self.project_folder,
                global_mods=global_mods,
                stop_event=stop_event
            )
            worker.signals.result.connect(on_result)
            worker.signals.error.connect(on_error)
            worker.signals.done_one.connect(on_done_one)
            pool.start(worker)



    def display_image_group(self, image_data_list, root_name):
        # Stop repaints while rebuilding the grid
        self.setUpdatesEnabled(False)

        # Clear previous viewers and labels
        for widget in self.viewer_widgets:
            self.image_grid_layout.removeWidget(widget['container'])
            widget['container'].deleteLater()
        self.viewer_widgets = []

        # Determine grid size
        num_images = len(image_data_list)
        grid_cols = 5  # Adjust columns as needed
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        positions = [(i, j) for i in range(grid_rows) for j in range(grid_cols)]
        positions = positions[:num_images]

        # Build all widgets with updates still disabled
        local_viewers = []
        for idx, (position, image_data) in enumerate(zip(positions, image_data_list)):
            label = QLabel(os.path.basename(image_data.filepath))
            label.setAlignment(Qt.AlignCenter)
            label.setMaximumHeight(20)

            viewer = ImageViewer()
            self._wire_viewer_for_inspection(viewer)
            viewer.drawing_mode = self.current_drawing_mode
            viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            viewer.polygon_drawn.connect(self.on_polygon_drawn)
            viewer.polygon_changed.connect(self.on_polygon_modified)
            viewer.editing_finished.connect(self.reload_current_root)

            clean_button = QPushButton("Clean Vector")
            clean_button.setFixedSize(70, 25)
            clean_button.setStyleSheet("QPushButton { padding: 1px 5px; font-size: 9px; }")
            edit_button = QPushButton("Edit Image Viewer")
            edit_button.setFixedSize(130, 25)
            edit_button.setStyleSheet("QPushButton { padding: 1px 5px; font-size: 9px; }")
            clean_button.clicked.connect(partial(self.clean_polygons, viewer))
            clean_button.clicked.connect(partial(self.delete_polygons_for_viewer, viewer))
            edit_button.clicked.connect(partial(self.edit_image_viewer, viewer))

            buttons_layout = QHBoxLayout()
            buttons_layout.setSpacing(2)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.addWidget(clean_button)
            buttons_layout.addWidget(edit_button)

            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.setSpacing(1)
            v_layout.setContentsMargins(1, 1, 1, 1)
            v_layout.addWidget(label)
            v_layout.addWidget(viewer)
            v_layout.addLayout(buttons_layout)

            vw_rec = {
                'container': container,
                'viewer': viewer,
                'image_data': image_data,
                'band_index': idx + 1
            }
            local_viewers.append(vw_rec)
            self.image_grid_layout.addWidget(container, position[0], position[1])

        # Swap in the new list at once
        self.viewer_widgets = local_viewers

        # Now set images + polygons after all widgets exist
        for rec in self.viewer_widgets:
            viewer = rec['viewer']
            image_data = rec['image_data']
            pixmap = self.convert_cv_to_pixmap(image_data.image)
            if pixmap is not None:
                # IMPORTANT: assign image_data BEFORE set_image
                viewer.image_data = image_data
                viewer.set_image(pixmap)
                # EXPLICITLY load polygons from the tab (don’t rely on viewer.parent())
                self.load_polygons(viewer, image_data)
            else:
                QMessageBox.warning(self, "Error", f"Could not display image: {image_data.filepath}")

        self.update_polygon_manager()
        self._rewire_all_viewers_for_inspection()
        self.setUpdatesEnabled(True)

    def edit_image_viewer(self, viewer):
        """
        Open ImageEditorDialog for the current viewer image, update the viewer if
        the dialog is accepted. No QMessageBox pop-ups on success/cancel/errors.
        """
        import logging
        from PyQt5 import QtWidgets

        # 1) Guard: need an image to edit
        if not getattr(viewer, "image_data", None) or viewer.image_data.image is None:
            logging.warning("edit_image_viewer: no image to edit in this viewer.")
            return

        # 2) Build editor anchored to the on-disk file (prevents double-scaling)
        original_image = viewer.image_data.image.copy()
        filepath = getattr(viewer.image_data, "filepath", "") or ""

        try:
            editor = ImageEditorDialog(self, image_data=original_image, image_filepath=filepath)
            # Pass project folder so .ax files go to the right place
            editor.project_folder = getattr(self, "project_folder", None)
        except Exception as e:
            logging.exception(f"edit_image_viewer: failed to create editor: {e}")
            return

        # 3) Execute dialog (modal). Do nothing if cancelled.
        try:
            result = editor.exec_()
        except Exception as e:
            logging.exception(f"edit_image_viewer: editor.exec_() failed: {e}")
            return

        if result != QtWidgets.QDialog.Accepted:
            # Silent cancel: no popups
            logging.info("edit_image_viewer: editing cancelled by user.")
            return

        # 4) Accepted → fetch modified image and update viewer (no popups)
        try:
            modified_image = editor.get_modified_image()
            if modified_image is None:
                logging.warning("edit_image_viewer: editor returned no modified image.")
                return

            viewer.image_data.image = modified_image
            pixmap = self.convert_cv_to_pixmap(modified_image)
            if pixmap is None:
                logging.error("edit_image_viewer: convert_cv_to_pixmap failed; viewer not updated.")
                return

            # Clear overlays that no longer align after edits (safe if method exists)
            try:
                viewer.clear_polygons()
            except Exception:
                pass

            viewer.set_image(pixmap)

            try:
                if hasattr(self, "refresh_viewer"):
                    root = self.get_current_root_name() if hasattr(self, "get_current_root_name") else None
                    self.refresh_viewer(root_name=root)
            except Exception:
                pass

            logging.info("edit_image_viewer: image updated from editor.")
        except Exception as e:
            logging.exception(f"edit_image_viewer: failed to apply modified image: {e}")
            return


    def toggle_drawing_mode(self):
        if self.toggleDrawingModeAct.isChecked():
            new_mode = "point"
            self.toggleDrawingModeAct.setText("Drawing Mode: Point")
        else:
            new_mode = "polygon"
            self.toggleDrawingModeAct.setText("Drawing Mode: Polygon")
        
        # Update the central drawing mode property.
        self.current_drawing_mode = new_mode

        # Update all currently active viewers.
        for item in self.viewer_widgets:
            if isinstance(item, dict) and "viewer" in item:
                item["viewer"].drawing_mode = new_mode
            else:
                item.drawing_mode = new_mode


                
    def open_folder(self):
        """
        Always operate in 'dual_folder' mode.
        - First folder: required (multispectral)
        - Second folder: optional; if canceled, set a fake folder, skip matching, and
          use multispectral roots to drive the UI.
        """
        import os, logging
        # Reset data
        self.reset_data_structures()

        # --- 1) First folder (required) ---
        options = QtWidgets.QFileDialog.Options()
        initial_dir = os.path.expanduser("~")
        multispectral_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select first image Folder", initial_dir, options=options
        )
        if not multispectral_folder:
            return  # user canceled entirely

        self.current_folder_path = multispectral_folder
        self.load_multispectral_images_from_folder(multispectral_folder)

        # --- 2) Second folder (optional) ---
        thermal_rgb_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select second image Folder (Cancel to skip)", initial_dir, options=options
        )

        self.mode = "dual_folder"

        if thermal_rgb_folder:
            # Normal dual-folder path
            self.thermal_rgb_folder_path = thermal_rgb_folder
            try:
                self.load_thermal_rgb_images_from_folder(thermal_rgb_folder)
                try:
                    self.matching_roots_dialog()
                except Exception as e:
                    logging.debug(f"matching_roots_dialog() skipped/failed: {e}")
            except Exception as e:
                logging.error(f"Failed loading second folder: {e}")
           
            root_names = getattr(self, "root_names", None) or self.multispectral_root_names
        else:
            # User skipped second folder: synthesize a fake one and avoid matching
            self.thermal_rgb_folder_path = os.path.join(
                getattr(self, "project_folder", os.getcwd()), "_FAKE_SECOND_FOLDER_"
            )
            # Do NOT call load_thermal_rgb_images_from_folder()
            # Ensure dual-folder structures exist but are empty
            self.thermal_rgb_image_data_groups = {}
            self.thermal_root_names = []
            # Present a unified root list so downstream code has something to use
            self.root_names = list(self.multispectral_root_names)
            # Optional flag if other code needs to know this is synthetic
            self._dual_folder_fake_second = True

            root_names = self.root_names  # drives the UI from multispectral roots

        # --- 3) Update slider + load first group ---
        if not root_names:
            logging.info("No roots found after opening folder(s).")
            QtWidgets.QMessageBox.warning(
                self, "No Images Found",
                "No root names were found. Please ensure the selected folder(s) contain valid images."
            )
            self.group_slider.setMaximum(0)
            self.group_slider.setValue(0)
            return

        # Clamp index and apply limits
        if not hasattr(self, "current_root_index") or self.current_root_index is None:
            self.current_root_index = 0
        self.current_root_index = max(0, min(self.current_root_index, len(root_names) - 1))
        self.group_slider.setMaximum(max(0, len(root_names) - 1))
        self.group_slider.setValue(self.current_root_index)

        # Load the first group
        try:
            self.load_image_group(root_names[self.current_root_index])
        except Exception as e:
            logging.error(f"Error loading image group: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Loading Error",
                "Failed to load image group. Please ensure the selected folder(s) contain valid images."
            )

    def open_rgb_folder(self):
        """
        Allows the user to select an RGB images folder, prompts for batch size,
        groups images into fake roots based on the batch size, and updates the application state accordingly.
        """
        import os
        from collections import defaultdict

        # Prompt to select RGB images folder
        options = QtWidgets.QFileDialog.Options()
        initial_dir = os.path.expanduser("~")
        rgb_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select RGB Images Folder", initial_dir, options=options)
        if not rgb_folder:
            return  # User canceled the dialog

        # Prompt the user for batch size
        batch_size, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Batch Size",
            "Enter the number of images per root group:",
            value=10,  # Default value
            min=1,     # Minimum value
            max=1000,  # Maximum value, adjust as needed
            step=1
        )
        if not ok:
            return  # User canceled the dialog
        self.batch_size = batch_size  # <- store for save/load

        # Reset data structures relevant to RGB-only mode
        self.reset_data_structures()

        # Clear the image grid layout
        self.clear_image_grid()

        # Reset the polygon manager
        self.polygon_manager.list_widget.clear()

        # Set the RGB folder path
        self.current_folder_path = rgb_folder

        # Load RGB image file paths
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')  # Add more if needed
        all_images = [
            os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder)
            if f.lower().endswith(image_extensions)
        ]
        all_images.sort()  # Optional: sort the images

        if not all_images:
            QtWidgets.QMessageBox.warning(
                self, "No Images Found",
                "No images were found in the selected folder. Please select a folder with valid images."
            )
            return

        # Group images into batches based on the batch size
        self.multispectral_root_names = []
        self.multispectral_image_data_groups = defaultdict(list)
        total_images = len(all_images)
        num_roots = (total_images + batch_size - 1) // batch_size  # Ceiling division

        for i in range(num_roots):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, total_images)
            batch_images = all_images[start_index:end_index]
            root_name = f"Root{i + 1}"  # Create fake root names like Root1, Root2, ...
            self.multispectral_root_names.append(root_name)
            self.multispectral_image_data_groups[root_name].extend(batch_images)

        # Keep mirrors consistent
        self.root_names = self.multispectral_root_names.copy()
        self.root_id_mapping = {name: idx + 1 for idx, name in enumerate(self.multispectral_root_names)}

        self.mode = 'rgb_only'
        self.current_root_index = 0  # start at the first batch

        # Update the slider maximum after grouping images
        self.group_slider.blockSignals(True)
        self.group_slider.setMaximum(len(self.multispectral_root_names) - 1)
        self.group_slider.setValue(self.current_root_index)
        self.group_slider.blockSignals(False)

        # Update the slider label
        if self.multispectral_root_names:
            self.slider_label.setText(f"Root: {self.multispectral_root_names[self.current_root_index]}")
        else:
            self.slider_label.setText("Root: N/A")

        # Logging (optional)
        logging.info(f"Grouped {len(all_images)} images into {num_roots} roots.")
        for root in self.multispectral_root_names:
            logging.debug(f"{root}: {len(self.multispectral_image_data_groups[root])} images.")

        # Load the first image group
        if self.multispectral_root_names:
            try:
                self.load_image_group(self.multispectral_root_names[self.current_root_index])
            except IndexError as e:
                print(f"Error loading image group: {e}")
                QtWidgets.QMessageBox.critical(
                    self, "Loading Error",
                    "Failed to load image group. Please ensure that the selected folder contains valid images."
                )
        else:
            print("No images found after grouping. Please check the selected folder and batch size.")
            QtWidgets.QMessageBox.warning(
                self, "No Image Groups",
                "No image groups were created. Please ensure that the selected folder contains valid images and the batch size is appropriate."
            )


    def open_rgb_folder_from_path(self, rgb_folder, batch_size=None):
        import os
        from collections import defaultdict
        import logging
        from PyQt5 import QtWidgets

        if not rgb_folder or not os.path.isdir(rgb_folder):
            QtWidgets.QMessageBox.warning(self, "Folder Not Found",
                                          "Dragged path is not a folder. Please drop an image or a folder.")
            return

        # fixed batch size = 10, no prompt
        if batch_size is None:
            batch_size = 3
        self.batch_size = batch_size

        # Reset state (same reset you do in your Open RGB flow)
        self.reset_data_structures()
        self.clear_image_grid()
        self.polygon_manager.list_widget.clear()
        self.current_folder_path = rgb_folder

        # Scan images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_images = [os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder)
                      if f.lower().endswith(image_extensions)]
        all_images.sort()

        if not all_images:
            QtWidgets.QMessageBox.warning(self, "No Images Found",
                                          "No images were found in the dropped folder.")
            return

        # Batch into fake roots: Root1, Root2, ...
        self.multispectral_root_names = []
        self.multispectral_image_data_groups = defaultdict(list)
        total_images = len(all_images)
        num_roots = (total_images + batch_size - 1) // batch_size
        for i in range(num_roots):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, total_images)
            batch_images = all_images[start_index:end_index]
            root_name = f"Root{i + 1}"
            self.multispectral_root_names.append(root_name)
            self.multispectral_image_data_groups[root_name].extend(batch_images)

        # Keep mirrors consistent + set mode and slider, then load first group
        self.root_names = self.multispectral_root_names.copy()
        self.root_id_mapping = {name: idx + 1 for idx, name in enumerate(self.multispectral_root_names)}
        self.mode = 'rgb_only'
        self.current_root_index = 0
        self.group_slider.blockSignals(True)
        self.group_slider.setMaximum(len(self.multispectral_root_names) - 1)
        self.group_slider.setValue(self.current_root_index)
        self.group_slider.blockSignals(False)

        try:
            self.load_image_group(self.root_names[self.current_root_index])
        except Exception as e:
            logging.error(f"Error loading image group: {e}")
            QtWidgets.QMessageBox.critical(self, "Loading Error",
                                           "Failed to load image group. Please ensure the folder contains valid images.")


    def _thumbname_cached(self, obj_name, filepath, image_type="RGB"):
        """
        Fast, cached wrapper around construct_thumbnail_name(...).
        If self._suppress_thumbs is True, returns a cheap deterministic name
        without creating/saving anything.
        """
        import os
        key = (obj_name, os.path.normcase(os.path.abspath(filepath)), image_type)
        cache = getattr(self, "_thumb_name_cache", None)
        if cache is None:
            cache = self._thumb_name_cache = {}

        if key in cache:
            return cache[key]

        if getattr(self, "_suppress_thumbs", False):
            # cheap, deterministic filename; *no disk I/O*
            stem = os.path.splitext(os.path.basename(filepath))[0]
            name = f"{obj_name}_{stem}_{image_type}.jpg"
        else:
            name = self.construct_thumbnail_name(obj_name, filepath, image_type=image_type)

        cache[key] = name
        return name
    

    def _build_exif_map_with_progress(self, paths, *, parent=None, chunk_size=128, timeout_per_file=2.5):
        """
        Read EXIF for 'paths' in chunks with a progress dialog, robust logging, and timeouts.
        Returns a dict: {abs_norm_path: {tag:value, ...}}. Sets self._exif_last_method to:
          'exiftool' | 'missing' | 'cancelled' | 'error'.
        """
        import os, json, time, shutil, subprocess, logging
        from PyQt5 import QtWidgets, QtCore

        def _normkey(p):  
            return os.path.normcase(os.path.abspath(p))

        total = len(paths)
        exe = getattr(self, "exiftool_path", None) or "exiftool"
        have = bool(shutil.which(exe))
        raw_map = {}
        errors = 0

        if not have:
            self._exif_last_method = "missing"
            logging.info("[exif] exiftool not found; skipping EXIF for %d files", total)
            return raw_map

        # UI
        dlg = QtWidgets.QProgressDialog("Reading EXIF…", "Cancel", 0, total, parent or self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.show()

        logging.info("[exif] start: files=%d exe=%r chunk_size=%d timeout_per_file=%.2fs",
                     total, exe, chunk_size, timeout_per_file)
        t0 = time.perf_counter()
        done = 0

        for start in range(0, total, chunk_size):
            if dlg.wasCanceled():
                self._exif_last_method = "cancelled"
                logging.warning("[exif] user cancelled at %d/%d", done, total)
                return {}

            chunk = paths[start:start+chunk_size]
            # Reasonable timeout scales with chunk length
            timeout_s = max(5.0, timeout_per_file * max(1, len(chunk)))
            cmd = [exe, "-json", "-n", "-fast2"] + chunk

            logging.debug("[exif] run exiftool on chunk %d..%d (%d files); timeout=%.1fs",
                          start+1, start+len(chunk), len(chunk), timeout_s)
            try:
                cp = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=timeout_s, check=False
                )
                if cp.returncode != 0:
                    logging.warning("[exif] non-zero exit (%d) on chunk %d..%d. stderr(head): %s",
                                    cp.returncode, start+1, start+len(chunk),
                                    (cp.stderr or b"")[:200].decode("utf-8", "ignore"))
                try:
                    data = json.loads(cp.stdout.decode("utf-8", "ignore") or "[]")
                except Exception as e:
                    data = []
                    errors += len(chunk)
                    logging.error("[exif] JSON parse failed on chunk %d..%d: %s",
                                  start+1, start+len(chunk), e)

            except subprocess.TimeoutExpired:
                data = []
                errors += len(chunk)
                logging.error("[exif] TIMEOUT on chunk %d..%d (skipping %d files)",
                              start+1, start+len(chunk), len(chunk))
            except Exception as e:
                data = []
                errors += len(chunk)
                logging.error("[exif] fatal error on chunk %d..%d: %s",
                              start+1, start+len(chunk), e)

            # Map by SourceFile (if present)
            for entry in (data or []):
                sf = entry.get("SourceFile")
                if not sf:
                    continue
                k = _normkey(sf)
                # Keep everything except SourceFile
                raw_map[k] = {kk: vv for kk, vv in entry.items() if kk != "SourceFile"}

            done += len(chunk)
            # UI tick
            dlg.setLabelText(f"Reading EXIF… {done}/{total}")
            dlg.setValue(done)
            QtWidgets.QApplication.processEvents()

            # Periodic throughput log
            if done == total or (done % (chunk_size*4) == 0):
                dt = time.perf_counter() - t0
                rate = (done/dt) if dt > 0 else 0.0
                logging.info("[exif] progress: %d/%d (%.1f%%)  throughput=%.1f files/s",
                             done, total, 100.0*done/max(1,total), rate)

        dt = time.perf_counter() - t0
        rate = (total/dt) if dt > 0 else 0.0
        self._exif_last_method = "exiftool"
        logging.info("[exif] done: files=%d ok=%d errors=%d  time=%.2fs  rate=%.1f files/s",
                     total, len(raw_map), errors, dt, rate)
        return raw_map


    def _safe_cell(v):
        """Sanitize a value for CSV: strip control chars, remove active delimiter,
        and ONLY Excel-guard non-numeric strings that start with = + - @."""
        import re
        try:
            import numpy as _np
            _np_ok = True
        except Exception:
            _np_ok = False

        def _is_numeric_scalar(x):
            # Python numbers or NumPy scalar (not array)
            if isinstance(x, (int, float)):
                return True
            if _np_ok:
                import numpy as _np
                if isinstance(x, (_np.integer, _np.floating)):  # NumPy scalars
                    return True
            return False

        def _is_numeric_string(s: str) -> bool:
            s = s.strip()
            # Plain number: ±digits[.digits][e±digits]
            return bool(re.match(r'^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?$', s))

        if v is None:
            return ""

        # Containers
        if isinstance(v, (list, tuple, set)):
            return " ".join(_safe_cell(x) for x in v if x is not None)
        if isinstance(v, dict):
            return " ".join(f"{_safe_cell(k)}:{_safe_cell(val)}" for k, val in sorted(v.items()))
        if isinstance(v, (bytes, bytearray)):
            return f"[{len(v)} bytes]"

        # NumPy arrays
        if _np_ok and isinstance(v, _np.ndarray):
            if v.ndim == 0:  # scalar array
                try:
                    return str(v.item())
                except Exception:
                    return str(v)
            flat = v.ravel()
            if flat.size > 64:
                return f"[{flat.size} values]"
            return " ".join(_safe_cell(x) for x in flat.tolist())

        # Numeric scalars → plain string (no Excel guard)
        if _is_numeric_scalar(v):
            return str(v)

        # Everything else → string sanitize
        s = str(v).replace("\r", " ").replace("\n", " ").replace("\t", " ")

        # remove active delimiter so naive split works
        delim = getattr(self, "_active_csv_delimiter", ",")
        if delim and delim in s:
            s = s.replace(delim, " ")

        # Excel-injection guard ONLY for *non-numeric* strings
        s_stripped = s.lstrip()
        if s_stripped and s_stripped[0] in "=+-@" and not _is_numeric_string(s_stripped):
            s = "'" + s
        return s


    def process_polygon(self, group_name, filepath, polygon_dict, exif_data_dict, exif_tags, model_loaded, opts=None):
        """
        Deterministic CSV extraction on RAW + .ax (crop → resize → expr), independent of viewer.
        Coordinates are mapped using viewer if available, else using stored pixmap_size, else raw.
        """
        import os, logging
        import numpy as np
        import cv2

        # Avoid OpenCV oversubscription under ThreadPoolExecutor
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        data_rows = []
        modified_polygons = []

        def make_hashable(val):
            if isinstance(val, list):
                return tuple(make_hashable(x) for x in val)
            elif isinstance(val, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
            else:
                return val

        exif_key = os.path.normcase(os.path.abspath(filepath))

        # --- CSV cell sanitizer + EXIF attach helper -------------------------------
        def _safe_cell(v):
            try:
                import numpy as _np
            except Exception:
                _np = None

            if v is None:
                return ""
            if isinstance(v, (list, tuple, set)):
                return " ".join(_safe_cell(x) for x in v if x is not None)
            if _np is not None and isinstance(v, _np.ndarray):
                if v.ndim == 0:
                    try:
                        return str(v.item())
                    except Exception:
                        return str(v)
                flat = v.ravel()
                if flat.size > 64:
                    return f"[{flat.size} values]"
                return " ".join(str(x) for x in flat.tolist())
            if isinstance(v, (bytes, bytearray)):
                return f"[{len(v)} bytes]"
            s = str(v).replace("\r", " ").replace("\n", " ").replace("\t", " ")
            # remove active delimiter from data so naive split works
            delim = getattr(self, "_active_csv_delimiter", ",")
            if delim and delim in s:
                s = s.replace(delim, " ")
            # Excel injection guard
            if s and s[0] in "=+-@":
                s = "'" + s
            return s

        def _attach_exif(row):
            exif_data = exif_data_dict.get(exif_key, {}) or {}
            for tag in exif_tags:
                row[tag] = _safe_cell(exif_data.get(tag, ""))
            return row
        # --------------------------------------------------------------------------

        # --- LOG (once) how/if EXIF was obtained for this run ---------------------
        try:
            if not getattr(self, "_exif_usage_logged_once", False):
                method = getattr(self, "_exif_last_method", "unknown")
                exif_enabled = bool(exif_tags) and (exif_tags != ['FakePath'])
                logging.info(
                    "[process_polygon:init] exif_enabled=%s; method=%s; exiftool_path=%r; tags=%s; file=%s",
                    exif_enabled, method, getattr(self, "exiftool_path", None), exif_tags, filepath
                )
                self._exif_usage_logged_once = True
        except Exception:
            pass
        # --------------------------------------------------------------------------

        stats_cfg   = (opts or {}).get('stats', {}) if opts is not None else {}
        shrink_opt  = (opts or {}).get('shrink', {}) if opts is not None else {}
        bm_opts     = (opts or {}).get('band_math', {}) if opts is not None else {}
        bm_enabled  = bool(bm_opts.get('enabled', False))
        bm_formulas = dict(bm_opts.get('formulas', {}) or {}) if bm_enabled else {}

        rf_enabled = bool(
            model_loaded and hasattr(self, "random_forest_model")
            and self.random_forest_model is not None and "model" in self.random_forest_model
        )

        def _calc_stats(array_1d):
            out = {}
            qs = stats_cfg.get('quantiles', []) or []
            if array_1d is None or (hasattr(array_1d, "size") and array_1d.size == 0):
                if stats_cfg.get('mean'):   out['Mean'] = None
                if stats_cfg.get('median'): out['Median'] = None
                if stats_cfg.get('std'):    out['Standard Deviation'] = None
                for q in qs:
                    q_str = (str(q).rstrip('0').rstrip('.') if isinstance(q, float) else str(q))
                    out[f'Q{q_str}'] = None
                return out

            orig_is_int = np.issubdtype(array_1d.dtype, np.integer)
            a = np.asarray(array_1d, dtype=np.float64)
            try:
                a_min = float(np.nanmin(a)); a_max = float(np.nanmax(a))
            except Exception:
                a_min, a_max = float(a.min()), float(a.max())
            looks_beyond_8bit = (a_max - a_min) > 255.0 or a_max > 255.0

            if stats_cfg.get('mean'):   out['Mean']   = float(np.mean(a))
            if stats_cfg.get('median'): out['Median'] = float(np.median(a))
            if stats_cfg.get('std'):    out['Standard Deviation'] = float(np.std(a, ddof=0))

            q_norm = []
            for q in qs:
                try: qf = float(q)
                except Exception: continue
                if 0.0 <= qf <= 1.0: qf *= 100.0
                qf = 0.0 if qf < 0.0 else (100.0 if qf > 100.0 else qf)
                q_norm.append((q, qf))

            for q_orig, q_for_np in q_norm:
                try:
                    try:
                        q_val = np.percentile(a, q_for_np, method='nearest')
                    except TypeError:
                        q_val = np.percentile(a, q_for_np, interpolation='nearest')
                except Exception:
                    q_val = np.nan
                if orig_is_int or looks_beyond_8bit:
                    q_val = int(round(q_val)) if np.isfinite(q_val) else None
                else:
                    q_val = float(q_val) if np.isfinite(q_val) else None
                q_str = (str(q_orig).rstrip('0').rstrip('.') if isinstance(q_orig, float) else str(q_orig))
                out[f'Q{q_str}'] = q_val
            return out

        # ===================== SPEED TWEAKS START (cache + I/O gate) =====================
        cache = getattr(self, "_export_cache", None)
        cache_lock = getattr(self, "_export_cache_lock", None)
        cached_tuple = None
        if cache is not None and cache_lock is not None:
            try:
                with cache_lock:
                    cached_tuple = cache.get(filepath)
            except Exception:
                cached_tuple = None

        if cached_tuple is not None:
            img, chans, is_rgb = cached_tuple
            if img is None:
                logging.warning(f"Could not build export image for {filepath}")
                return data_rows, modified_polygons
            if img.ndim == 2:
                img = img[..., None]
            H, W = img.shape[:2]
        else:
            gate = getattr(self, "_export_load_gate", None)
            if gate is not None:
                with gate:
                    img, _C = self._get_export_image(filepath)
            else:
                img, _C = self._get_export_image(filepath)
            if img is None:
                logging.warning(f"Could not build export image for {filepath}")
                return data_rows, modified_polygons
            if img.ndim == 2:
                img = img[..., None]
            H, W = img.shape[:2]

            # Channels in export order: R,G,B,(extras…)
            chans = self._channels_in_export_order(img)
            # Fallback: split by last axis if needed
            if len(chans) == 1 and img.ndim == 3 and img.shape[2] > 1:
                chans = [img[..., i] for i in range(img.shape[2])]
            # Sanity: force each channel to H×W
            fixed = []
            for i, ch in enumerate(chans):
                ch = np.asarray(ch)
                ch = np.squeeze(ch)
                if ch.ndim == 3 and ch.shape[-1] == 1: ch = ch[..., 0]
                if ch.ndim == 3 and ch.shape[0] == 1 and ch.shape[1:] == (H, W): ch = ch[0]
                if ch.shape != (H, W):
                    raise ValueError(f"Incompatible channel shape at index {i}: {ch.shape} vs {(H, W)}")
                fixed.append(ch)
            chans = fixed
            is_rgb = len(chans) >= 3

            # Save into per-run cache
            if cache is not None and cache_lock is not None:
                try:
                    with cache_lock:
                        cache[filepath] = (img, chans, is_rgb)
                except Exception:
                    pass
        # ====================== SPEED TWEAKS END (cache + I/O gate) ======================

        # b1..bN mapping in export order
        bmap_arrays = {f"b{i+1}": chans[i] for i in range(len(chans))}

        # Compile band expressions once per call
        expr_code_cache = {}
        def _eval_band_expr(expr, xi=None, yi=None):
            code = expr_code_cache.get(expr)
            if code is None:
                code = compile(expr, "<expr>", "eval")
                for name in code.co_names:
                    if name not in bmap_arrays:
                        maxb = len(bmap_arrays)
                        raise NameError(f"Use only b1..b{maxb} in band expression")
                expr_code_cache[expr] = code

            env = {"__builtins__": {}}
            if xi is None or yi is None:
                local_map = {k: v.astype(np.float32, copy=False) for k, v in bmap_arrays.items()}
                with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                    res = eval(code, env, local_map)
                return np.nan_to_num(np.asarray(res, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                local_map = {k: float(v[yi, xi]) for k, v in bmap_arrays.items()}
                with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                    res = eval(code, env, local_map)
                try: res = float(res)
                except Exception: res = float(np.nan)
                return 0.0 if not np.isfinite(res) else res

        # Scene stats (once per image)
        scene_stats = {}
        if is_rgb:
            for channel_name, ch in zip(["R", "G", "B"], chans[:3]):
                a = ch.astype(float, copy=False)
                scene_stats[channel_name] = {
                    "Scene Mean": float(np.mean(a)),
                    "Scene Median": float(np.median(a)),
                    "Scene Standard Deviation": float(np.std(a)),
                }
            for i_extra, ch in enumerate(chans[3:], start=4):
                a = ch.astype(float, copy=False)
                scene_stats[f"band_{i_extra}"] = {
                    "Scene Mean": float(np.mean(a)),
                    "Scene Median": float(np.median(a)),
                    "Scene Standard Deviation": float(np.std(a)),
                }
        else:
            a = img.astype(float, copy=False)
            channel_type = "Gray" if img.ndim == 2 else "Other"
            scene_stats[channel_type] = {
                "Scene Mean": float(np.mean(a)),
                "Scene Median": float(np.median(a)),
                "Scene Standard Deviation": float(np.std(a)),
            }

        if bm_formulas:
            for fname, expr in bm_formulas.items():
                try:
                    aa = _eval_band_expr(expr).astype(float, copy=False)
                    scene_stats[fname] = {
                        "Scene Mean": float(np.mean(aa)),
                        "Scene Median": float(np.median(aa)),
                        "Scene Standard Deviation": float(np.std(aa)),
                    }
                except Exception as e:
                    logging.warning(f"Band-math scene stats skipped ({fname}='{expr}') for '{filepath}': {e}")

        extraction_type = polygon_dict.get("type", "polygon")

        # --------------------- POINT extraction ---------------------
        if extraction_type == "point":
            parsed_points = polygon_dict.get("points", [])
            if not polygon_dict.get("name"):
                logging.warning(f"Point in group '{group_name}' for '{filepath}' has no name. Skipping.")
                return data_rows, modified_polygons

            name = polygon_dict.get("name", "")
            root_name = self.get_root_by_filepath(filepath)
            root_id = self.root_id_mapping.get(root_name, "N/A")

            pts_img = self._points_to_export_frame(filepath, parsed_points, polygon_dict, img.shape)
            pts_img = [(int(round(x)), int(round(y))) for (x, y) in pts_img]

            for idx, (xi, yi) in enumerate(pts_img):
                if not (0 <= xi < W and 0 <= yi < H):
                    logging.warning(f"Point {(xi, yi)} out of bounds for '{filepath}'")
                    continue

                if is_rgb:
                    red_channel   = np.array([chans[0][yi, xi]], dtype=np.float32)
                    green_channel = np.array([chans[1][yi, xi]], dtype=np.float32)
                    blue_channel  = np.array([chans[2][yi, xi]], dtype=np.float32)

                if rf_enabled and is_rgb:
                    try:
                        model_classes = self.random_forest_model["model"].classes_
                        class_percentages = {f"Class {cls} %": None for cls in model_classes}
                        model_feature_names = self.random_forest_model.get(
                            "feature_names", ["red_channel", "green_channel", "blue_channel"]
                        )
                        features_map = {
                            "red_channel": red_channel,
                            "green_channel": green_channel,
                            "blue_channel": blue_channel,
                        }
                        additional = ["exg", "gcc", "bcc", "gbd", "wdx", "shd"]
                        if any(feat in model_feature_names for feat in additional):
                            features_map["exg"] = calculate_exg(red_channel, green_channel, blue_channel)
                            features_map["gcc"] = calculate_gcc(red_channel, green_channel, blue_channel)
                            features_map["bcc"] = calculate_bcc(red_channel, green_channel, blue_channel)
                            features_map["gbd"] = calculate_gbd(green_channel, blue_channel)
                            features_map["wdx"] = calculate_wdx(red_channel, green_channel, blue_channel)
                            features_map["shd"] = calculate_shd(red_channel, green_channel, blue_channel)
                        X = np.column_stack([features_map[feat] for feat in model_feature_names])
                        if X.size == 0:
                            class_percentages = {f"Class {cls} %": 0.0 for cls in model_classes}
                        else:
                            predictions = self.random_forest_model["model"].predict(X)
                            u, cts = np.unique(predictions, return_counts=True)
                            total = cts.sum()
                            for cls in model_classes: class_percentages[f"Class {cls} %"] = 0.0
                            for cls, cnt in zip(u, cts):
                                class_percentages[f"Class {cls} %"] = (cnt / total) * 100
                    except Exception as e:
                        logging.error(f"RF prediction error for '{filepath}', point {idx}: {e}")
                        class_percentages = {}
                else:
                    class_percentages = {}

                if is_rgb:
                    for channel_name, ch in zip(["R","G","B"], chans[:3]):
                        v = np.array([ch[yi, xi]], dtype=np.float32)
                        channel_stats = _calc_stats(v)
                        row = {
                            "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                            "Root ID": root_id,
                            "Root Folder": os.path.dirname(filepath),
                            "File Name": os.path.basename(filepath),
                            "Object ID": f"{name}_point_{idx}",   # fixed
                            "Channel": channel_name,
                            "Pixel Count": 1,
                        }
                        row.update(channel_stats)
                        sc = scene_stats.get(channel_name, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                        row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                        row.update(class_percentages)
                        _attach_exif(row)
                        data_rows.append(row)

                    if len(chans) > 3:
                        for i_extra, ch in enumerate(chans[3:], start=4):
                            channel_name = f"band_{i_extra}"
                            v = np.array([ch[yi, xi]], dtype=np.float32)
                            channel_stats = _calc_stats(v)
                            row = {
                                "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                                "Root ID": root_id,
                                "Root Folder": os.path.dirname(filepath),
                                "File Name": os.path.basename(filepath),
                                "Object ID": f"{name}_point_{idx}",
                                "Channel": channel_name,
                                "Pixel Count": 1,
                            }
                            row.update(channel_stats)
                            sc = scene_stats.get(channel_name, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                            row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                            row.update(class_percentages)
                            _attach_exif(row)
                            data_rows.append(row)
                else:
                    pix = img[yi, xi]
                    pixs = np.array([pix]).reshape(-1) if np.isscalar(pix) else np.array(pix).reshape(-1)
                    point_stats = _calc_stats(pixs)
                    channel_type = "Gray" if img.ndim == 2 else "Other"
                    row = {
                        "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                        "Root ID": root_id,
                        "Root Folder": os.path.dirname(filepath),
                        "File Name": os.path.basename(filepath),
                        "Object ID": f"{name}_point_{idx}",
                        "Channel": channel_type,
                        "Pixel Count": 1,
                    }
                    row.update(point_stats)
                    sc = scene_stats.get(channel_type, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                    row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                    _attach_exif(row)
                    data_rows.append(row)

                if bm_formulas:
                    for fname, expr in bm_formulas.items():
                        try:
                            val = _eval_band_expr(expr, xi=xi, yi=yi)
                            v = np.array([val], dtype=np.float32)
                            channel_stats = _calc_stats(v)
                            row = {
                                "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                                "Root ID": root_id,
                                "Root Folder": os.path.dirname(filepath),
                                "File Name": os.path.basename(filepath),
                                "Object ID": f"{name}_point_{idx}",
                                "Channel": fname,
                                "Pixel Count": 1,
                            }
                            row.update(channel_stats)
                            sc = scene_stats.get(fname, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                            row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                            if rf_enabled:
                                model_classes = self.random_forest_model["model"].classes_
                                row.update({f"Class {cls} %": None for cls in model_classes})
                            _attach_exif(row)
                            data_rows.append(row)
                        except Exception as e:
                            logging.warning(f"Band-math point eval skipped ({fname}='{expr}') for '{filepath}': {e}")

        # ------------------- POLYGON extraction -------------------
        else:
            if not polygon_dict.get("name"):
                logging.warning(f"Polygon in group '{group_name}' for '{filepath}' has no name. Skipping.")
                return data_rows, modified_polygons

            original_points = polygon_dict["points"]
            if shrink_opt.get('enabled'):
                factor = float(shrink_opt.get('factor', 0.07))
                swell  = bool(shrink_opt.get('swell', False))
                mod_points = self.shrink_or_swell_shapely_polygon(original_points, factor=factor, swell=swell)
            else:
                mod_points = original_points

            modified_polygons.append({
                "group_name": group_name,
                "filepath": filepath,
                "object_id": polygon_dict.get("name", ""),
                "modified_points": mod_points
            })

            name = polygon_dict.get("name", "")
            root_name = self.get_root_by_filepath(filepath)
            root_id = self.root_id_mapping.get(root_name, "N/A")

            pts_img = self._points_to_export_frame(filepath, mod_points, polygon_dict, img.shape)
            pts_img = [(int(round(x)), int(round(y))) for (x, y) in pts_img]

            import numpy as np, cv2
            mask = np.zeros((H, W), dtype=np.uint8)
            if len(pts_img) == 1:
                xi, yi = pts_img[0]
                if 0 <= yi < H and 0 <= xi < W:
                    mask[yi, xi] = 255
            elif len(pts_img) >= 3:
                pts = np.array(pts_img, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

            mask_bool = mask.astype(bool)
            num_pixels = int(mask_bool.sum())
            if num_pixels == 0:
                logging.warning(f"No pixels found within polygon for '{filepath}'. Skipping.")
                return data_rows, modified_polygons

            if is_rgb:
                red_channel   = chans[0][mask_bool]
                green_channel = chans[1][mask_bool]
                blue_channel  = chans[2][mask_bool]

                channel_triplets = {
                    "R": red_channel.astype(float, copy=False),
                    "G": green_channel.astype(float, copy=False),
                    "B": blue_channel.astype(float, copy=False),
                }

                # RF (optional)
                if rf_enabled:
                    try:
                        model_classes = self.random_forest_model["model"].classes_
                        class_percentages = {f"Class {cls} %": None for cls in model_classes}
                        model_feature_names = self.random_forest_model.get(
                            "feature_names", ["red_channel", "green_channel", "blue_channel"]
                        )
                        features_map = {
                            "red_channel": red_channel.astype(np.float32, copy=False),
                            "green_channel": green_channel.astype(np.float32, copy=False),
                            "blue_channel": blue_channel.astype(np.float32, copy=False),
                        }
                        additional = ["exg", "gcc", "bcc", "gbd", "wdx", "shd"]
                        if any(feat in model_feature_names for feat in additional):
                            features_map["exg"] = calculate_exg(features_map["red_channel"], features_map["green_channel"], features_map["blue_channel"])
                            features_map["gcc"] = calculate_gcc(features_map["red_channel"], features_map["green_channel"])
                            features_map["bcc"] = calculate_bcc(features_map["red_channel"], features_map["green_channel"])
                            features_map["gbd"] = calculate_gbd(features_map["green_channel"], features_map["blue_channel"])
                            features_map["wdx"] = calculate_wdx(features_map["red_channel"], features_map["green_channel"])
                            features_map["shd"] = calculate_shd(features_map["red_channel"], features_map["green_channel"], features_map["blue_channel"])
                        X = np.column_stack([features_map[feat] for feat in model_feature_names])
                        if X.size == 0:
                            class_percentages = {f"Class {cls} %": 0.0 for cls in model_classes}
                        else:
                            predictions = self.random_forest_model["model"].predict(X)
                            u, cts = np.unique(predictions, return_counts=True)
                            total = cts.sum()
                            for cls in model_classes: class_percentages[f"Class {cls} %"] = 0.0
                            for cls, cnt in zip(u, cts):
                                class_percentages[f"Class {cls} %"] = (cnt / total) * 100
                    except Exception as e:
                        logging.error(f"RF prediction error for '{filepath}', polygon '{name}': {e}")
                        class_percentages = {}
                else:
                    class_percentages = {}

                for channel_name in ["R", "G", "B"]:
                    channel_pixels = channel_triplets[channel_name]
                    if channel_pixels.size == 0:
                        continue
                    channel_stats = _calc_stats(channel_pixels)
                    row = {
                        "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                        "Root ID": root_id,
                        "Root Folder": os.path.dirname(filepath),
                        "File Name": os.path.basename(filepath),
                        "Object ID": name,
                        "Channel": channel_name,
                        "Pixel Count": num_pixels,
                    }
                    row.update(channel_stats)
                    sc = scene_stats.get(channel_name, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                    row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                    row.update(class_percentages)
                    _attach_exif(row)
                    data_rows.append(row)

                if len(chans) > 3:
                    for i_extra, ch in enumerate(chans[3:], start=4):
                        channel_name = f"band_{i_extra}"
                        channel_pixels = ch[mask_bool].astype(float, copy=False)
                        if channel_pixels.size == 0:
                            continue
                        channel_stats = _calc_stats(channel_pixels)
                        row = {
                            "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                            "Root ID": root_id,
                            "Root Folder": os.path.dirname(filepath),
                            "File Name": os.path.basename(filepath),
                            "Object ID": name,
                            "Channel": channel_name,
                            "Pixel Count": num_pixels,
                        }
                        row.update(channel_stats)
                        sc = scene_stats.get(channel_name, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                        row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                        row.update(class_percentages)
                        _attach_exif(row)
                        data_rows.append(row)

                if bm_formulas:
                    for fname, expr in bm_formulas.items():
                        try:
                            arr = _eval_band_expr(expr)
                            vals = arr[mask_bool].astype(float, copy=False)
                            channel_stats = _calc_stats(vals)
                            row = {
                                "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                                "Root ID": root_id,
                                "Root Folder": os.path.dirname(filepath),
                                "File Name": os.path.basename(filepath),
                                "Object ID": name,
                                "Channel": fname,
                                "Pixel Count": num_pixels,
                            }
                            row.update(channel_stats)
                            sc = scene_stats.get(fname, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                            row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                            row.update(class_percentages)
                            _attach_exif(row)
                            data_rows.append(row)
                        except Exception as e:
                            logging.warning(f"Band-math polygon eval skipped ({fname}='{expr}') for '{filepath}': {e}")

            else:
                pixels = img[mask_bool]
                if pixels.size == 0:
                    logging.warning(f"No pixels under mask for '{filepath}'.")
                    return data_rows, modified_polygons

                poly_stats = _calc_stats(pixels.reshape(-1))
                channel_type = "Gray" if img.ndim == 2 else "Other"
                row = {
                    "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                    "Root ID": root_id,
                    "Root Folder": os.path.dirname(filepath),
                    "File Name": os.path.basename(filepath),
                    "Object ID": name,
                    "Channel": channel_type,
                    "Pixel Count": int(pixels.size if img.ndim == 2 else pixels.shape[0]),
                }
                row.update(poly_stats)
                sc = scene_stats.get(channel_type, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                _attach_exif(row)
                data_rows.append(row)

                if bm_formulas:
                    for fname, expr in bm_formulas.items():
                        try:
                            arr = _eval_band_expr(expr)
                            vals = arr[mask_bool].astype(float, copy=False)
                            channel_stats = _calc_stats(vals)
                            row = {
                                "Project": os.path.basename(os.path.normpath(self.project_folder)) if self.project_folder else "N/A",
                                "Root ID": root_id,
                                "Root Folder": os.path.dirname(filepath),
                                "File Name": os.path.basename(filepath),
                                "Object ID": name,
                                "Channel": fname,
                                "Pixel Count": int(vals.size),
                            }
                            row.update(channel_stats)
                            sc = scene_stats.get(fname, {"Scene Mean": None, "Scene Median": None, "Scene Standard Deviation": None})
                            row["Scene Mean"] = sc["Scene Mean"]; row["Scene Median"] = sc["Scene Median"]; row["Scene Standard Deviation"] = sc["Scene Standard Deviation"]
                            _attach_exif(row)
                            data_rows.append(row)
                        except Exception as e:
                            logging.warning(f"Band-math polygon (non-RGB) eval skipped ({fname}='{expr}') for '{filepath}': {e}")

        # --- Dedupe ---
        unique_data_rows, seen = [], set()
        for row in data_rows:
            row_key = tuple(sorted((k, make_hashable(v)) for k, v in row.items()))
            if row_key not in seen:
                seen.add(row_key)
                unique_data_rows.append(row)
        return unique_data_rows, modified_polygons

    
    def save_polygons_to_csv(self):
        """
        Saves all polygon data to a CSV file using multithreading for faster processing.

        - EXIF sanitization + strict delimiter guard (no commas inside cells when delimiter=',').
        - No thumbnails.
        - Dialog options decide stats, shrink/swell, RF, EXIF, export modified polygons.
        - Auto-suffixes filenames if exists.
        - Writes CSV with QUOTE_NONE; UTF-8-BOM for Excel friendliness.
        """
        import os, time, csv, random, string, concurrent.futures, functools, logging, threading
        from PyQt5 import QtWidgets, QtCore

        # ----------------- helpers -----------------
        def _next_available_path(base_path: str) -> str:
            base_dir = os.path.dirname(base_path)
            stem, ext = os.path.splitext(os.path.basename(base_path))
            candidate = os.path.join(base_dir, f"{stem}{ext}")
            i = 1
            while os.path.exists(candidate):
                candidate = os.path.join(base_dir, f"{stem}_{i}{ext}")
                i += 1
            return candidate

        def _fake_path_token(n=8) -> str:
            return "FAKE:/" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

        def _sanitize_for_csv(val, delim=','):
            """Plain string with no CR/LF/TAB and no active delimiter.
            Excel-guard ONLY non-numeric strings that start with = + - @."""
            import re
            try:
                import numpy as np
                _np_ok = True
            except Exception:
                _np_ok = False

            def _is_numeric_scalar(x):
                if isinstance(x, (int, float)):
                    return True
                if _np_ok:
                    import numpy as np
                    if isinstance(x, (np.integer, np.floating)):
                        return True
                return False

            def _is_numeric_string(s: str) -> bool:
                s = s.strip()
                return bool(re.match(r'^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?$', s))

            if val is None:
                s = ""
            elif isinstance(val, (list, tuple, set)):
                s = " ".join(_sanitize_for_csv(v, delim) for v in val if v is not None)
            elif isinstance(val, dict):
                s = " ".join(f"{_sanitize_for_csv(k, delim)}:{_sanitize_for_csv(v, delim)}"
                             for k, v in sorted(val.items()))
            elif isinstance(val, (bytes, bytearray)):
                s = f"[{len(val)} bytes]"
            elif _np_ok and isinstance(val, np.ndarray):
                a = np.ravel(val)
                if a.size > 64:
                    s = f"[{a.size} values]"
                else:
                    s = " ".join(_sanitize_for_csv(x, delim) for x in a.tolist())
            elif _is_numeric_scalar(val):
                # numeric scalar: write as-is; no Excel guard
                s = str(val)
            else:
                s = str(val)

            # normalize whitespace + remove delimiter
            s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            if delim and delim in s:
                s = s.replace(delim, " ")

            # Excel guard ONLY for non-numeric strings
            s_stripped = s.lstrip()
            if s_stripped and s_stripped[0] in "=+-@" and not _is_numeric_string(s_stripped):
                s = "'" + s
            return s


        def _audit_row_for_delim(safe_row: dict, delim: str, where: str):
            """Guarantee no cell contains the delimiter; replace+log if any do."""
            fixed = {}
            for k, v in safe_row.items():
                vv = v
                if isinstance(vv, str) and delim and delim in vv:
                    vv = vv.replace(delim, " ")
                    logging.warning(f"[csv-guard:{where}] Delimiter found in field '{k}' → replaced.")
                fixed[k] = vv
            return fixed
        # ------------------------------------------

        # -- show the export/options dialog --
        dlg = AnalysisOptionsDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return  # user cancelled

        self.analysis_options = dlg.get_options()
        opts = self.analysis_options
        include_exif_opt = bool(opts.get("include_exif", True))
        use_rf           = bool(opts.get("use_random_forest", True))
        export_mods      = bool(opts.get("export_modified_polygons", True))

        # Choose CSV delimiter (default ',');
        csv_delimiter = (opts or {}).get('csv_delimiter', ',')
        # expose to workers (process_polygon) so their EXIF sanitizer uses the same delimiter
        self._active_csv_delimiter = csv_delimiter

        # ====== SPEED TWEAKS INIT (per-run cache + I/O gate) ======
        try:
            self._export_cache = {}
            self._export_cache_lock = threading.Lock()
            if not hasattr(self, "_export_load_gate") or not isinstance(getattr(self, "_export_load_gate"), threading.BoundedSemaphore):
                self._export_load_gate = threading.BoundedSemaphore(value=2)
        except Exception:
            pass
        # ===========================================================

        total_start_time = time.perf_counter()
        logging.basicConfig(filename='save_polygons_to_csv.log', level=logging.DEBUG,
                            format='%(asctime)s:%(levelname)s:%(message)s')

        # Any polygons to save?
        if not self.all_polygons:
            from PyQt5 import QtWidgets as _QtW
            _QtW.QMessageBox.warning(self, "No Polygons", "There are no polygons to save.")
            return

        # Determine final CSV path (unique)
        if getattr(self, 'project_folder', None):
            exports_dir = os.path.join(self.project_folder, 'exports')
            os.makedirs(exports_dir, exist_ok=True)
            default_name = f'exported_polygons_{self.project_name}.csv'
            save_path = _next_available_path(os.path.join(exports_dir, default_name))
            project_name = os.path.basename(os.path.normpath(self.project_folder))
        else:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            chosen_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save All CSV", os.path.expanduser("~"), "CSV Files (*.csv)", options=options)
            if not chosen_path:
                return
            save_path = _next_available_path(chosen_path)
            project_name = "N/A"

        def _normkey(p): return os.path.normcase(os.path.abspath(p))
        filepaths_with_polygons = []
        for group_polygons in self.all_polygons.values():
            for fp in group_polygons.keys():
                filepaths_with_polygons.append(fp)
        filepaths_with_polygons = list({_normkey(p) for p in filepaths_with_polygons})

        # Build EXIF map / tags per option
        if include_exif_opt:
            raw_exif = self._get_exif_data_with_optional_path(filepaths_with_polygons)
            raw_exif = {_normkey(k): v for k, v in (raw_exif or {}).items()}
            sanitized_exif, sanitized_tags = self._sanitize_exif_map_for_csv(raw_exif, filepaths_with_polygons)
            if sanitized_tags:
                exif_tags = sanitized_tags[:]
                exif_data_dict = sanitized_exif
            else:
                exif_tags = ['FakePath']
                exif_data_dict = {fp: {'FakePath': _fake_path_token()} for fp in filepaths_with_polygons}
        else:
            exif_tags = ['FakePath']
            exif_data_dict = {fp: {'FakePath': _fake_path_token()} for fp in filepaths_with_polygons}

        # Deduplicate EXIF tag headers while preserving order; ensure strings
        seen = set()
        exif_tags = [str(t) for t in exif_tags if not (t in seen or seen.add(t))]

        # RF model decision: dialog decides
        if use_rf:
            if hasattr(ProjectTab, 'shared_random_forest_model') and ProjectTab.shared_random_forest_model is not None:
                self.random_forest_model = ProjectTab.shared_random_forest_model
                model_loaded = True
            else:
                try:
                    model_loaded = bool(self.load_random_forest_model())
                except Exception:
                    model_loaded = False
        else:
            model_loaded = False

        # CSV columns
        fieldnames_raw = [
            'Project', 'Root ID', 'Root Folder', 'File Name', 'Object ID', 'Channel'
        ] + exif_tags + [
            'Pixel Count', 'Mean', 'Median', 'Standard Deviation',
            'Scene Mean', 'Scene Median', 'Scene Standard Deviation'
        ]

        # Quantile columns
        stats_cfg = opts.get('stats', {}) if opts is not None else {}
        for q in (stats_cfg.get('quantiles', []) or []):
            lbl = (str(q).rstrip('0').rstrip('.') if isinstance(q, float) else str(q))
            col = f'Q{lbl}'
            if col not in fieldnames_raw:
                fieldnames_raw.append(col)

        # RF class columns only if model actually loaded
        if model_loaded and getattr(self, "random_forest_model", None) is not None and "model" in self.random_forest_model:
            model_classes = self.random_forest_model['model'].classes_
            rgb_class_fields = [f'Class {cls} %' for cls in model_classes]
            fieldnames_raw.extend(rgb_class_fields)

        # Build work list
        polygons_to_process = []
        for group_name, polygons_data in self.all_polygons.items():
            for filepath, polygon_dict in polygons_data.items():
                polygons_to_process.append((group_name, filepath, polygon_dict))

        data_rows = []
        modified_polygons = []

        # Progress & parallel
        progress_dialog = QtWidgets.QProgressDialog("Saving polygons...", "Cancel", 0, len(polygons_to_process), self)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.show()

        cancelled = False
        max_workers = min(32, (os.cpu_count() or 4) + 4)

        worker = functools.partial(
            self.process_polygon,
            exif_data_dict=exif_data_dict,
            exif_tags=exif_tags,
            model_loaded=model_loaded,
            opts=opts,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, g, fp, pd) for (g, fp, pd) in polygons_to_process]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if progress_dialog.wasCanceled():
                    cancelled = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    row, mod_polygon = future.result()
                    data_rows.extend(row)
                    if export_mods:
                        modified_polygons.extend(mod_polygon)
                except Exception as e:
                    logging.error(f"Error processing a polygon: {e}")
                progress_dialog.setValue(i + 1)
                QtWidgets.QApplication.processEvents()

        progress_dialog.setValue(len(polygons_to_process))
        if cancelled:
            return

        # Sanitize headers (and remember mapping raw->safe)
        fieldnames_safe = [_sanitize_for_csv(h, csv_delimiter) for h in fieldnames_raw]
        key_map = dict(zip(fieldnames_raw, fieldnames_safe))

        # Write CSV (UTF-8 BOM for Excel)
        try:
            with open(save_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames_safe,
                    restval="",
                    extrasaction="ignore",
                    delimiter=csv_delimiter,
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    lineterminator="\n",
                )

                # header: double-sanitize + audit just in case
                header_safe = {safe: _sanitize_for_csv(safe, csv_delimiter) for safe in fieldnames_safe}
                header_safe = _audit_row_for_delim(header_safe, csv_delimiter, "header")
                writer.writeheader()  # DictWriter ignores provided dict for header, uses fieldnames

                # rows
                for row in data_rows:
                    safe_row = {}
                    for raw_key, safe_key in key_map.items():
                        safe_val = _sanitize_for_csv(row.get(raw_key, ""), csv_delimiter)
                        safe_row[safe_key] = safe_val
                    # final guard
                    safe_row = _audit_row_for_delim(safe_row, csv_delimiter, "row")
                    writer.writerow(safe_row)

            logging.info(f"Data successfully saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save CSV to {save_path}: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save data to CSV:\n{e}")
            return

        # Post-write audit: verify each line has exactly N-1 delimiters
        try:
            expected = len(fieldnames_safe) - 1
            bad_lines = 0
            with open(save_path, 'r', encoding='utf-8-sig', newline='') as fh:
                for ln, line in enumerate(fh, 1):
                    cnt = line.rstrip("\n").count(csv_delimiter)
                    if cnt != expected:
                        bad_lines += 1
                        logging.warning(f"[csv-guard:file] Line {ln} has {cnt} delimiters; expected {expected}.")
            if bad_lines == 0:
                logging.info("[csv-guard:file] Delimiter audit passed.")
            else:
                logging.warning(f"[csv-guard:file] Delimiter audit found {bad_lines} problematic lines.")
        except Exception as e:
            logging.warning(f"[csv-guard:file] Post-write audit skipped: {e}")

        # Save modified polygons JSON only if requested
        if export_mods:
            json_base = os.path.splitext(save_path)[0]
            json_save_path = _next_available_path(f"{json_base}_modified_polygons.json")
            try:
                with open(json_save_path, 'w', encoding='utf-8') as jsonfile:
                    import json
                    json.dump(modified_polygons, jsonfile, indent=4)
                logging.info(f"Modified polygons successfully saved to {json_save_path}")
            except Exception as e:
                logging.error(f"Failed to save modified polygons to {json_save_path}: {e}")
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save modified polygons to JSON:\n{e}")
        else:
            logging.info("Skipping modified polygons JSON export per user option.")

        total_duration = time.perf_counter() - total_start_time
        logging.info(f"Total time to execute save_polygons_to_csv: {total_duration:.2f} seconds")

        try:
            self._export_cache.clear()
        except Exception:
            pass

    def _collect_all_image_paths_for_exif(self):
        """
        Return a sorted, de-duplicated list of every image path known to the project.
        Pulls from multispectral / thermal_rgb groups if present.
        """
        import os

        paths = []

        def _add_from_group(d):
            if isinstance(d, dict):
                for lst in d.values():
                    if isinstance(lst, (list, tuple)):
                        paths.extend([p for p in lst if p])

        _add_from_group(getattr(self, "multispectral_image_data_groups", None))
        _add_from_group(getattr(self, "thermal_rgb_image_data_groups", None))
        _add_from_group(getattr(self, "rgb_image_data_groups", None))  # if you have one

        # also mirror-group in case it exists
        _add_from_group(getattr(self, "image_data_groups", None))

        # unique + normalized + sorted
        normed = {os.path.normpath(p) for p in paths if isinstance(p, str)}
        return sorted(normed)


    def extract_exif_to_csv(self):
        """
        Export EXIF for *all* project images into a CSV.
        - Same CSV style as polygon export (delimiter guard, QUOTE_NONE, UTF-8 BOM).
        - Cancelable progress dialog.
        - Disables navigation via _set_nav_enabled while running.
        """
        from PyQt5 import QtWidgets, QtCore
        import os, csv, logging, random, string

        # ---------- small helpers ----------
        def _next_available_path(base_path: str) -> str:
            base_dir = os.path.dirname(base_path)
            stem, ext = os.path.splitext(os.path.basename(base_path))
            candidate = os.path.join(base_dir, f"{stem}{ext}")
            i = 1
            while os.path.exists(candidate):
                candidate = os.path.join(base_dir, f"{stem}_{i}{ext}")
                i += 1
            return candidate

        def _fake_path_token(n=8) -> str:
            return "FAKE:/" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

        # Excel-safe sanitizer that does NOT prefix numeric strings like "-49.98"
        def _sanitize_for_csv(val, delim=','):
            """Plain string with no CR/LF/TAB and no active delimiter.
            Excel-guard ONLY non-numeric strings that start with = + - @."""
            import re
            try:
                import numpy as np
                _np_ok = True
            except Exception:
                _np_ok = False

            def _is_numeric_scalar(x):
                if isinstance(x, (int, float)):
                    return True
                if _np_ok:
                    import numpy as np
                    if isinstance(x, (np.integer, np.floating)):
                        return True
                return False

            def _is_numeric_string(s: str) -> bool:
                s = s.strip()
                return bool(re.match(r'^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?$', s))

            if val is None:
                s = ""
            elif isinstance(val, (list, tuple, set)):
                s = " ".join(_sanitize_for_csv(v, delim) for v in val if v is not None)
            elif isinstance(val, dict):
                s = " ".join(f"{_sanitize_for_csv(k, delim)}:{_sanitize_for_csv(v, delim)}"
                             for k, v in sorted(val.items()))
            elif isinstance(val, (bytes, bytearray)):
                s = f"[{len(val)} bytes]"
            elif _np_ok and isinstance(val, np.ndarray):
                a = val.ravel()
                if a.size > 64:
                    s = f"[{a.size} values]"
                else:
                    s = " ".join(_sanitize_for_csv(x, delim) for x in a.tolist())
            elif _is_numeric_scalar(val):
                s = str(val)
            else:
                s = str(val)

            s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            if delim and delim in s:
                s = s.replace(delim, " ")
            s_stripped = s.lstrip()
            if s_stripped and s_stripped[0] in "=+-@" and not _is_numeric_string(s_stripped):
                s = "'" + s
            return s

        def _audit_row_for_delim(safe_row: dict, delim: str, where: str):
            fixed = {}
            for k, v in safe_row.items():
                vv = v
                if isinstance(vv, str) and delim and delim in vv:
                    vv = vv.replace(delim, " ")
                    logging.warning(f"[csv-guard:{where}] Delimiter found in field '{k}' → replaced.")
                fixed[k] = vv
            return fixed
        # -----------------------------------

        # Gather every image in the project
        filepaths = self._collect_all_image_paths_for_exif()
        if not filepaths:
            QtWidgets.QMessageBox.information(self, "No Images", "No images were found in the project.")
            return

        # Ask where to save
        if getattr(self, 'project_folder', None):
            exports_dir = os.path.join(self.project_folder, 'exports')
            os.makedirs(exports_dir, exist_ok=True)
            default_name = f'exif_export_{self.project_name}.csv' if getattr(self, "project_name", None) else 'exif_export.csv'
            save_path = _next_available_path(os.path.join(exports_dir, default_name))
            project_name = os.path.basename(os.path.normpath(self.project_folder))
        else:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            chosen_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save EXIF CSV", os.path.expanduser("~"), "CSV Files (*.csv)", options=options)
            if not chosen_path:
                return
            save_path = _next_available_path(chosen_path)
            project_name = "N/A"

        # Pick CSV delimiter (match your polygon export default, or read from self.analysis_options if present)
        csv_delimiter = getattr(self, "_active_csv_delimiter", ",")
        self._active_csv_delimiter = csv_delimiter  # ensure helpers use same delimiter

        # Disable nav while working
        try:
            if hasattr(self, "_set_nav_enabled"):
                self._set_nav_enabled(False)
        except Exception:
            pass

        # Fetch EXIF for all paths (reuses your same helpers)
        def _normkey(p): 
            import os
            return os.path.normcase(os.path.abspath(p))

        normed_paths = [_normkey(p) for p in filepaths]
        try:
            raw_exif = self._get_exif_data_with_optional_path(normed_paths) or {}
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "EXIF Error", f"Failed to read EXIF from files:\n{e}")
            try:
                if hasattr(self, "_set_nav_enabled"):
                    self._set_nav_enabled(True)
            except Exception:
                pass
            return

        # Sanitize EXIF map & tags (reuse your helper)
        try:
            sanitized_exif, sanitized_tags = self._sanitize_exif_map_for_csv(raw_exif, normed_paths)
        except Exception:
            # Fallback: if your helper is unavailable, build a minimal sanitized view
            sanitized_exif, sanitized_tags = {}, []
            for fp, kv in (raw_exif or {}).items():
                if not isinstance(kv, dict):
                    continue
                for k in kv.keys():
                    if k not in sanitized_tags:
                        sanitized_tags.append(str(k))
                sanitized_exif[_normkey(fp)] = {str(k): str(v) for k, v in kv.items()}

        if not sanitized_tags:
            # Still write something: produce a fake path column so CSV isn't empty after file info
            sanitized_tags = ['FakePath']
            sanitized_exif = {p: {'FakePath': _fake_path_token()} for p in normed_paths}

        # Deduplicate tags while preserving order
        seen = set()
        sanitized_tags = [t for t in sanitized_tags if not (t in seen or seen.add(t))]
        sanitized_tags = [str(t) for t in sanitized_tags]

        # Build CSV header
        fieldnames_raw = ['Project', 'Root ID', 'Root Folder', 'File Name'] + sanitized_tags

        # Open CSV & progress dialog
        progress = QtWidgets.QProgressDialog("Exporting EXIF…", "Cancel", 0, len(normed_paths), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Sanitized headers and mapping
        fieldnames_safe = [_sanitize_for_csv(h, csv_delimiter) for h in fieldnames_raw]
        key_map = dict(zip(fieldnames_raw, fieldnames_safe))

        # Write CSV (UTF-8 BOM, QUOTE_NONE)
        try:
            with open(save_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames_safe,
                    restval="",
                    extrasaction="ignore",
                    delimiter=csv_delimiter,
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    lineterminator="\n",
                )

                # Header (DictWriter uses fieldnames, but we still audit)
                header_safe = {safe: _sanitize_for_csv(safe, csv_delimiter) for safe in fieldnames_safe}
                header_safe = _audit_row_for_delim(header_safe, csv_delimiter, "header")
                writer.writeheader()

                # Rows
                for i, fp_abs in enumerate(normed_paths):
                    if progress.wasCanceled():
                        break

                    # original path form for display columns
                    fp_display = filepaths[i]
                    root_name = self.get_root_by_filepath(fp_display) if hasattr(self, "get_root_by_filepath") else ""
                    root_id = self.root_id_mapping.get(root_name, "N/A") if hasattr(self, "root_id_mapping") else "N/A"

                    exif_row = sanitized_exif.get(fp_abs, {})
                    safe_row = {
                        key_map['Project']: _sanitize_for_csv(project_name, csv_delimiter),
                        key_map['Root ID']: _sanitize_for_csv(root_id, csv_delimiter),
                        key_map['Root Folder']: _sanitize_for_csv(os.path.dirname(fp_display), csv_delimiter),
                        key_map['File Name']: _sanitize_for_csv(os.path.basename(fp_display), csv_delimiter),
                    }

                    # fill EXIF columns (missing→"")
                    for tag in sanitized_tags:
                        safe_row[key_map[tag]] = _sanitize_for_csv(exif_row.get(tag, ""), csv_delimiter)

                    # final audit and write
                    safe_row = _audit_row_for_delim(safe_row, csv_delimiter, "row")
                    writer.writerow(safe_row)

                    progress.setValue(i + 1)
                    QtWidgets.QApplication.processEvents()

                progress.setValue(len(normed_paths))

            logging.info(f"EXIF exported to {save_path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save EXIF CSV:\n{e}")
            return
        finally:
            try:
                if hasattr(self, "_set_nav_enabled"):
                    self._set_nav_enabled(True)
            except Exception:
                pass


    def _is_gdalish_tag(self, tag: str) -> bool:
        """
        Returns True if an EXIF tag name is GDAL/GeoTIFF-ish and should be excluded from CSV.
        Examples: GDALMetadata, GDALNoData, ModelTiePoint, ModelTransformation, PixelScale,
                  GeoKeyDirectory, GeoDoubleParams, GeoAsciiParams, TIFFTAG_GDAL_*, etc.
        """
        if not isinstance(tag, str):
            return False
        t = tag.strip()
        if not t:
            return False
        # Fast exacts
        if t in {
            "GDALMetadata", "GDALNoData",
            "ModelTiePoint", "ModelTransformation", "ModelTransformationMatrix",
            "PixelScale", "ModelPixelScale",
            "GeoKeyDirectory", "GeoDoubleParams", "GeoAsciiParams",
            "TIFFTAG_GDAL_METADATA", "TIFFTAG_GDAL_NODATA",
            "GdalMetadata", "GdalNoData",
        }:
            return True
        # Common prefixes
        lowers = t.lower()
        if lowers.startswith("tifftag_gdal"):
            return True
        if lowers.startswith("geo") and ("key" in lowers or "params" in lowers):
            return True
        if "tiepoint" in lowers or "transform" in lowers or "pixelscale" in lowers:
            return True
        return False


    def _is_gdalish_value(self, value) -> bool:
        """
        Returns True if an EXIF field value looks like a huge GDAL/XML blob or
        a massive numeric list that makes CSV ugly.
        """
        # XML-ish / multi-line GDAL metadata
        try:
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return False
                if s.startswith("<GDALMetadata") or "<GDALMetadata" in s:
                    return True
                if s.startswith("<xml") or s.startswith("<?xml"):
                    return True
                # Heuristic: very long or very multi-line strings → treat as GDAL-ish
                if len(s) > 1024 or s.count("\n") > 5:
                    return True
                # STATISTICS_* smell inside strings
                if "STATISTICS_MAXIMUM" in s or "STATISTICS_MEAN" in s or "STATISTICS_STDDEV" in s:
                    return True
            elif isinstance(value, (list, tuple)):
                # Enormous numeric arrays → likely georeferencing / per-band stats
                if len(value) > 16:
                    return True
            elif hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
                # Unknown iterables: if long, skip
                try:
                    val_list = list(value)
                    if len(val_list) > 16:
                        return True
                except Exception:
                    pass
        except Exception:
            return False
        return False


    def _sanitize_exif_record(self, record: dict) -> dict:
        """
        Remove GDAL/GeoTIFF-ish keys/values and coerce remaining values to CSV-friendly scalars.
        - Drops keys matched by _is_gdalish_tag(...)
        - Drops values matched by _is_gdalish_value(...)
        - Joins short lists/tuples with spaces (e.g. '16 16 16 16')
        """
        if not isinstance(record, dict):
            return {}
        out = {}
        for k, v in record.items():
            if self._is_gdalish_tag(k):
                continue
            if self._is_gdalish_value(v):
                continue
            # Coerce to a compact scalar/text
            try:
                if isinstance(v, (list, tuple)):
                    # Keep short lists only
                    if len(v) <= 8:
                        out[str(k)] = " ".join(str(x) for x in v)
                    else:
                        continue
                elif isinstance(v, (bytes, bytearray)):
                    # avoid raw bytes
                    continue
                else:
                    # prefer one-liners
                    s = str(v)
                    if s.count("\n") > 0:
                        s = s.replace("\r", " ").replace("\n", " ").strip()
                    out[str(k)] = s
            except Exception:
                # If coercion fails, just skip this tag
                continue
        return out


    def _sanitize_exif_map_for_csv(self, exif_map: dict, file_list_normed: list) -> (dict, list):
        """
        Sanitize a whole EXIF map (per file).
        Returns (sanitized_exif_map, sorted_tag_list).
        If, AFTER sanitization, there are no tags at all across all files, returns ({}, []).
        """
        clean = {}
        tag_set = set()
        for fp in file_list_normed:
            rec = exif_map.get(fp, {}) or {}
            srec = self._sanitize_exif_record(rec)
            clean[fp] = srec
            tag_set.update(srec.keys())
        tags = sorted(tag_set)
        return clean if tags else {}, tags


    def _ax_candidates(self, filepath):
        base = os.path.splitext(os.path.basename(filepath))[0] + ".ax"
        cand = []
        if getattr(self, "project_folder", None):
            cand.append(os.path.join(self.project_folder, base))
        cand.append(os.path.join(os.path.dirname(filepath), base))
        return cand

    def _load_ax_mods(self, filepath):
        for mf in self._ax_candidates(filepath):
            try:
                if os.path.exists(mf):
                    with open(mf, "r", encoding="utf-8") as f:
                        return json.load(f) or {}
            except Exception as e:
                logging.error(f"Failed to read AX {mf}: {e}")
        return {}

    def _eval_band_expression(self, img_float, expr):
        if not expr:
            return None
        if img_float.ndim == 2:
            mapping = {'b1': img_float.astype(np.float32, copy=False)}
        else:
            mapping = {f"b{i+1}": img_float[:, :, i].astype(np.float32, copy=False)
                       for i in range(img_float.shape[2])}
        code = compile(expr, "<expr>", "eval")
        for name in code.co_names:
            if name not in mapping:
                maxb = 1 if img_float.ndim == 2 else img_float.shape[2]
                raise NameError(f"Use only b1..b{maxb} in band expression")
        res = eval(code, {"__builtins__": {}}, mapping)
        if isinstance(res, np.ndarray):
            res = res.astype(np.float32, copy=False)
        else:
            res = np.full(img_float.shape[:2], float(res), dtype=np.float32)
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)



    def _apply_ax_to_raw(self, raw_img, ax):
        """
        Deterministic, stack-safe replay (viewer-independent):

          • Honors optional ax["op_order"]=["rotate","crop","resize","band_expression"]
          • If the crop rect was saved in a different basis than the order user runs now,
            it’s remapped so pixels/coords stay consistent.
          • Keeps scientific magnitudes (float32); no histogram/CLAHE.

        Returns (img_float32, channels)
        """
        import numpy as np, cv2, logging
        if raw_img is None:
            return None, 0

        img = self._ensure_hwc(raw_img)

        # AX params
        try: rot = int(ax.get("rotate", 0)) % 360
        except Exception: rot = 0
        crop_rect = ax.get("crop_rect") or None
        crop_ref  = ax.get("crop_rect_ref_size") or None
        resize    = ax.get("resize") or None
        expr      = (ax.get("band_expression") or "").strip()

        op_order = ax.get("op_order")
        if not (isinstance(op_order, (list, tuple)) and all(isinstance(s, str) for s in op_order)):
            op_order = ["rotate", "crop", "resize", "band_expression"]

        raw_h, raw_w = img.shape[:2]
        crop_basis = _infer_crop_basis(ax, raw_w, raw_h, rot) if crop_rect else None

        def _do_rotate():
            nonlocal img
            if rot in (90, 180, 270):
                try:
                    # np.rot90: k=-1 = 90° CW, k=2 = 180°, k=1 = 90° CCW
                    k = {90: -1, 180: 2, 270: 1}[rot]
                    img = np.ascontiguousarray(np.rot90(img, k))
                except Exception as e:
                    logging.warning(f"Rotation failed ({rot} deg) via NumPy: {e}")


        def _do_crop():
            nonlocal img
            if not crop_rect:
                return
            Hc, Wc = img.shape[:2]

            # saved ref dims for the rectangle’s frame
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w_saved, ref_h_saved = int(crop_ref.get("w", raw_w)), int(crop_ref.get("h", raw_h))
            else:
                if crop_basis == "after_rotate":
                    ref_w_saved, ref_h_saved = _dims_after_rot(raw_w, raw_h, rot)
                else:
                    ref_w_saved, ref_h_saved = raw_w, raw_h

            rotate_applied = ("rotate" in op_order and "crop" in op_order and
                              op_order.index("rotate") < op_order.index("crop"))

            rect_to_apply = dict(crop_rect)
            if crop_basis == "after_rotate" and not rotate_applied and rot in (90,180,270):
                # un-rotate rect back to RAW basis
                rect_to_apply = _rect_after_rot(rect_to_apply, ref_w_saved, ref_h_saved, (360 - rot) % 360)
                ref_w_use, ref_h_use = raw_w, raw_h
            elif crop_basis == "pre_rotate" and rotate_applied and rot in (90,180,270):
                # rotate rect into rotated basis
                rect_to_apply = _rect_after_rot(rect_to_apply, raw_w, raw_h, rot)
                ref_w_use, ref_h_use = _dims_after_rot(raw_w, raw_h, rot)
            else:
                ref_w_use, ref_h_use = ref_w_saved, ref_h_saved

            rect_scaled = _scale_rect(rect_to_apply, ref_w_use, ref_h_use, Wc, Hc)
            x0, y0 = rect_scaled["x"], rect_scaled["y"]
            w,  h  = rect_scaled["width"], rect_scaled["height"]
            if w > 0 and h > 0:
                img = img[y0:y0+h, x0:x0+w]
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
            x = img.astype(np.float32, copy=False)
            idx = self._eval_band_expression(x, expr)
            img = np.dstack([x, idx.astype(np.float32, copy=False)]) if idx is not None else x

        ops = {
            "rotate": _do_rotate,
            "crop": _do_crop,
            "resize": _do_resize,
            "band_expression": _do_band_expr,
        }
        for op in op_order:
            f = ops.get(op.strip().lower())
            if f: f()

        img = img.astype(np.float32, copy=False)
        C = img.shape[2] if img.ndim == 3 else 1
        return img, C


    def _load_raw_image(self, filepath):
        """Load raw image similarly to the viewer pipeline, but without display normalization."""
        try:
            mode = getattr(self, "mode", None)
            data = ImageData(filepath, mode=mode) if mode else ImageData(filepath)
            img = data.image
            self._last_loader = "imagedata"
            return img
        except Exception:
            pass

        # 2) If it's a TIFF, use tifffile so 5+ channel stacks load correctly
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".tif", ".tiff"):
            try:
                import tifffile as tiff
                arr = tiff.imread(filepath)  # handles 16-bit, 32-bit, N-channel, etc.

                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    # If first dim looks like channel count (small) and the other two look like image dims, move axis
                    if arr.shape[0] <= 16 and arr.shape[1] >= 32 and arr.shape[2] >= 32:
                        arr = np.moveaxis(arr, 0, -1)  # C,H,W -> H,W,C
                elif arr.ndim > 3:
                    # Best-effort squeeze + put channels last if first dim is small
                    while arr.ndim > 3 and arr.shape[0] == 1:
                        arr = np.squeeze(arr, axis=0)
                    if arr.ndim == 3 and arr.shape[0] <= 16:
                        arr = np.moveaxis(arr, 0, -1)

                self._last_loader = "tifffile"
                return arr
            except ImportError:
                logging.error("tifffile is required to read 5+ channel TIFFs. pip install tifffile")
            except Exception as e:
                logging.warning(f"tifffile load failed for '{filepath}': {e}")

        # 3) Fallback to OpenCV (works for 1/3/4 channels)
        self._last_loader = "cv2"
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


    def _load_image_generic(self, filepath):
        """
        Generic HWC loader for non-TIFF files (and a fallback for TIFF).
        Returns a NumPy array (H×W×C or H×W), dtype preserved when possible.
        """
        import os, logging, numpy as np
        filepath = os.fspath(filepath)  # handle pathlib.Path

        # Try PIL first (handles PNG/JPEG/… and many 16-bit cases)
        try:
            from PIL import Image
            im = Image.open(filepath)
            im.load()
            # normalize paletted images to RGBA
            if im.mode == "P":
                im = im.convert("RGBA")
            arr = np.array(im)
            return arr
        except Exception as e:
            logging.debug(f"PIL load failed for '{filepath}': {e}")

        # Try OpenCV (robust on Windows paths via imdecode)
        try:
            import cv2
            data = np.fromfile(filepath, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            # BGR->RGB for 3-channel
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[..., ::-1]
            return img
        except Exception as e:
            logging.error(f"Generic load failed for '{filepath}': {e}")
            return None


    def _get_export_image(self, filepath):
        """
        Fast, deterministic export image:
        - Super-fast path for plain RGB images when there is NO .ax sidecar.
        - TIF/TIFF read via tifffile to preserve bands/pages.
        - Returns (img, meta) where img is HxWxC (C>=1), no display normalization.
        """
        import os, logging
        import numpy as np
        import cv2

        filepath = os.fspath(filepath)
        if not hasattr(self, "_export_image_cache"):
            self._export_image_cache = {}

        def _ax_mtime(fp):
            pf = getattr(self, "project_folder", None)
            base_ax = os.path.splitext(os.path.basename(fp))[0] + ".ax"
            candidates = []
            if pf:
                candidates.append(os.path.join(os.fspath(pf), base_ax))
            candidates.append(os.path.splitext(fp)[0] + ".ax")
            mt = 0.0
            for c in candidates:
                try:
                    if os.path.exists(c):
                        mt = max(mt, os.path.getmtime(c))
                except Exception:
                    pass
            return mt

        try:
            img_mtime = os.path.getmtime(filepath)
        except Exception:
            img_mtime = 0.0
        axmt = _ax_mtime(filepath)
        cache_key = ("export", os.path.abspath(filepath), img_mtime, axmt)

        if cache_key in self._export_image_cache:
            return self._export_image_cache[cache_key]

        ext = os.path.splitext(filepath)[1].lower()

        # -------- SUPER FAST PATH: plain images with no .ax --------
        if axmt == 0.0 and ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, {}
            # ensure HWC
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[..., None]
            elif img.ndim == 3 and img.shape[0] <= 32 and img.shape[0] not in (img.shape[1], img.shape[2]):
                img = np.moveaxis(img, 0, -1)
            img = np.ascontiguousarray(img)
            H, W = img.shape[:2]
            C = img.shape[2] if img.ndim == 3 else 1
            out = (img, {"H": H, "W": W, "C": C})
            self._export_image_cache[cache_key] = out
            return out

        # -------- TIFF path or anything needing .ax --------
        img = None
        use_tifffile = False
        if ext in (".tif", ".tiff"):
            try:
                import tifffile as tiff
                use_tifffile = True
            except Exception:
                use_tifffile = False

        if use_tifffile:
            try:
                a = np.asarray(tiff.imread(filepath))
                if a.ndim == 2:
                    img = a[..., None]
                elif a.ndim == 3:
                    if a.shape[0] <= 32 and a.shape[0] not in (a.shape[1], a.shape[2]):
                        img = np.moveaxis(a, 0, -1)
                    else:
                        img = a
                else:
                    flat = int(np.prod(a.shape[:-2]))
                    img = np.moveaxis(a.reshape(flat, a.shape[-2], a.shape[-1]), 0, -1)
            except Exception as e:
                logging.warning(f"tifffile read failed for '{filepath}': {e}")

        if img is None:
            # generic fallback (avoid heavy EXIF autorotate paths here)
            try:
                import cv2
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            except Exception:
                img = None
            if img is None:
                return None, {}
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[..., None]
            elif img.ndim == 3 and img.shape[0] <= 32 and img.shape[0] not in (img.shape[1], img.shape[2]):
                img = np.moveaxis(img, 0, -1)

        img = np.ascontiguousarray(img)

        # Apply .ax only if it actually exists
        if axmt > 0.0:
            try:
                if "apply_aux_modifications" in globals():
                    img = apply_aux_modifications(
                        filepath, img,
                        project_folder=os.fspath(getattr(self, "project_folder", "")) or None,
                        global_mode=False,
                    )
                else:
                    try:
                        img = self.apply_aux_modifications(
                            filepath, img,
                            project_folder=getattr(self, "project_folder", None),
                            global_mode=False,
                        )
                    except TypeError:
                        img = self.apply_aux_modifications(
                            filepath, img,
                            project_folder=getattr(self, "project_folder", None),
                        )
                if img.ndim == 2:
                    img = img[..., None]
                elif img.ndim == 3 and img.shape[0] <= 32 and img.shape[0] not in (img.shape[1], img.shape[2]):
                    img = np.moveaxis(img, 0, -1)
                img = np.ascontiguousarray(img)
            except Exception as e:
                logging.warning(f"apply_aux_modifications failed for '{filepath}': {e}")

        H, W = img.shape[:2]
        C = img.shape[2] if img.ndim == 3 else 1
        out = (img, {"H": H, "W": W, "C": C})
        self._export_image_cache[cache_key] = out
        return out


    def _channels_in_export_order(self, img):
        """
        Returns a list [R, G, B, band_4, band_5, ...] where each entry is HxW.
        For C<3, it returns [band_1, band_2, ...] in natural order.
        If C>=3, assumes first 3 channels are BGR if the image came via OpenCV.
        For multispectral where the first 3 aren’t literal BGR, this still
        produces a stable order and preserves all bands.
        """
        import numpy as np

        a = np.asarray(img)
        if a.ndim == 2:
            return [a]
        if a.ndim != 3:
            raise ValueError(f"_channels_in_export_order expects HxWxC; got shape {a.shape}")

        H, W, C = a.shape
        # If someone hands us CHW by mistake, fix it defensively.
        if C <= 2 and a.shape[0] <= 32 and a.shape[0] not in (H, W):
            a = np.moveaxis(a, 0, -1)
            H, W, C = a.shape

        chans = []
        if C >= 3:
            # Most OpenCV reads are BGR; flip to RGB for first three
            chans = [a[:, :, 2], a[:, :, 1], a[:, :, 0]]
            for i in range(3, C):
                chans.append(a[:, :, i])
        else:
            # C == 1 or 2
            for i in range(C):
                chans.append(a[:, :, i])

        # Ensure pure HxW arrays
        return [np.ascontiguousarray(c) for c in chans]


    def _map_points_scene_to_image(self, filepath, pts_any, img_shape, *, polygon_data=None):
        """
        INSPECTION-STYLE mapping (scene -> pixmap -> image):
          • If points were saved in image coords, return as-is.
          • If a live viewer exists, use its _image item's mapFromScene and the
            pixmap size to scale into the *actual* post-aux image size.
          • If no viewer, but saved a pixmap_size with the shape, use that.
          • Otherwise, pass through (best effort).

        
        """
        from PyQt5 import QtCore

        if not pts_any or img_shape is None:
            return []

        H = int(img_shape[0])
        W = int(img_shape[1])

        # 1) Already in image pixels?
        coord_space = (polygon_data or {}).get("coord_space", "scene")
        if coord_space == "image":
            return [(float(x), float(y)) for (x, y) in pts_any]

        # 2) Preferred: live viewer → map scene→pixmap, then scale to image size
        v = self.get_viewer_by_filepath(filepath)
        if v is not None and getattr(v, "_image", None) is not None:
            pm = v._image.pixmap()
            pw = float(max(1, pm.width()))
            ph = float(max(1, pm.height()))
            out = []
            for (sx, sy) in pts_any:
                p_item = v._image.mapFromScene(QtCore.QPointF(float(sx), float(sy)))
                xi = p_item.x() * (W / pw)
                yi = p_item.y() * (H / ph)
                out.append((float(xi), float(yi)))
            return out

        # 3) Headless export: scale using saved pixmap_size if present
        pm_size = (polygon_data or {}).get("pixmap_size")
        if isinstance(pm_size, (list, tuple)) and len(pm_size) == 2:
            pw = float(max(1, pm_size[0]))
            ph = float(max(1, pm_size[1]))
            sx = W / pw
            sy = H / ph
            return [(float(x) * sx, float(y) * sy) for (x, y) in pts_any]

        # 4) Last resort: assume they’re already image pixels
        return [(float(x), float(y)) for (x, y) in pts_any]


    def _paired_trgb_root(self, ms_root: str):
        """Return the paired Thermal/RGB root name for a given MS root (or None)."""
        if self.mode != 'dual_folder':
            return None
        try:
            i = self.multispectral_root_names.index(ms_root)
        except ValueError:
            return None
        j = i + int(self.root_offset)
        return self.thermal_rgb_root_names[j] if 0 <= j < len(self.thermal_rgb_root_names) else None

    def _paired_ms_root(self, trgb_root: str):
        """Return the paired MS root name for a given Thermal/RGB root (or None)."""
        if self.mode != 'dual_folder':
            return None
        try:
            j = self.thermal_rgb_root_names.index(trgb_root)
        except ValueError:
            return None
        i = j - int(self.root_offset)
        return self.multispectral_root_names[i] if 0 <= i < len(self.multispectral_root_names) else None

    def _gps_from_exif_tags(self, tags: dict):
        """
        Extract a signed (lat, lon) from common exiftool keys (already numeric with -n).
        """
        lat = (tags.get('GPSLatitude') or tags.get('Composite:GPSLatitude') or
               tags.get('XMP:GPSLatitude') or tags.get('Latitude'))
        lon = (tags.get('GPSLongitude') or tags.get('Composite:GPSLongitude') or
               tags.get('XMP:GPSLongitude') or tags.get('Longitude'))
        lat_ref = tags.get('GPSLatitudeRef')
        lon_ref = tags.get('GPSLongitudeRef')
        try:
            lat = float(lat); lon = float(lon)
        except Exception:
            return None, None
        # If refs exist, force correct sign
        if isinstance(lat_ref, str) and lat_ref.upper() == 'S':
            lat = -abs(lat)
        if isinstance(lon_ref, str) and lon_ref.upper() == 'W':
            lon = -abs(lon)
        return lat, lon

    def _first_gps_from_files(self, files: list):
        """
        Return {'latitude': ..., 'longitude': ...} from the first file that has GPS,
        reading EXIF once for this batch.
        """
        import os
        if not files:
            return None
        norm = lambda p: os.path.normcase(os.path.abspath(p))
        exif_map = self._get_exif_data_with_optional_path(files)
        for fp in files:
            tags = exif_map.get(norm(fp), {}) or {}
            lat, lon = self._gps_from_exif_tags(tags)
            if lat is not None and lon is not None:
                return {'latitude': float(lat), 'longitude': float(lon)}
        return None


    def build_root_coordinates_map(self):
        """
        Compute one (lat,lon) per logical root using either MS or its paired TRGB files.
        Mirror coordinates into both roots so saved polygon JSONs always get coordinates.
        """
        import logging
        self.root_coordinates = {}

        # Walk all MS roots, attach their paired TRGB file lists
        for ms_root in self.multispectral_root_names:
            ms_files   = list(self.multispectral_image_data_groups.get(ms_root, []))
            trgb_root  = self._paired_trgb_root(ms_root)
            trgb_files = list(self.thermal_rgb_image_data_groups.get(trgb_root, [])) if trgb_root else []

            # Prefer any file with valid GPS from either side
            coords = self._first_gps_from_files(ms_files + trgb_files)
            if coords:
                self.root_coordinates[ms_root] = coords
                if trgb_root:
                    self.root_coordinates[trgb_root] = coords

        logging.info("[build_root_coordinates_map] Built for %d roots.", len(self.root_coordinates))


    def open_root_offset_dialog(self):
        """
        Opens a dialog allowing the user to set a new root offset via a slider.
        """
        # Check if the dialog is already open and visible
        if self.root_offset_dialog is not None and self.root_offset_dialog.isVisible():
            self.root_offset_dialog.raise_()
            self.root_offset_dialog.activateWindow()
            return

        # Create a new instance of the dialog and store the reference
        self.root_offset_dialog = RootOffsetDialog(current_offset=self.root_offset, parent=self)

        # Connect the dialog's offset_changed signal to the set_root_offset method
        self.root_offset_dialog.offset_changed.connect(self.set_root_offset)

        # Optionally, connect the dialog's finished signal to handle dialog closure
        self.root_offset_dialog.finished.connect(self.on_root_offset_dialog_closed)

        # Show the dialog non-modally
        self.root_offset_dialog.show()

    def on_root_offset_dialog_closed(self, result):
        """
        Handles the closure of the RootOffsetDialog.
        """
        print("RootOffsetDialog closed with result:", result)
        self.root_offset_dialog = None  # Clear the reference
            

    def set_root_offset(self, new_offset):
        """
        Sets the new root offset and updates the application state accordingly.
        :param new_offset: The new integer value for root_offset.
        """
        if not isinstance(new_offset, int):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Root offset must be an integer.")
            return

        # Validate the new_offset
        if self.mode == 'dual_folder':
            min_offset = -len(self.thermal_rgb_root_names) + 1
            max_offset = len(self.thermal_rgb_root_names) - 1
            if not (min_offset <= new_offset <= max_offset):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Offset",
                    f"Root offset must be between {min_offset} and {max_offset} for dual_folder mode."
                )
                return
        elif self.mode == 'rgb_only':
            # Define validation rules for rgb_only mode if necessary
            pass

        # Proceed with setting the new offset
        old_offset = self.root_offset
        self.root_offset = new_offset

        logging.info(f"Root offset changed from {old_offset} to {new_offset}.")

        # Update the Root Offset Label
        #self.root_offset_label.setText(f"Root Offset: {self.root_offset}")

        # Remap the roots with the new offset
        self.map_matching_roots()
        
        #self.build_root_coordinates_map()


        # Update the slider's maximum value after remapping
        if self.multispectral_root_names:
            self.group_slider.setMaximum(len(self.multispectral_root_names) - 1)
        else:
            self.group_slider.setMaximum(0)  # Default to 0 if list is empty

        # Reload the current image group
        if self.multispectral_root_names:
            try:
                self.load_image_group(self.multispectral_root_names[self.current_root_index])
                self.update_slider_label(self.current_root_index)
            except IndexError as e:
                print(f"Error loading image group after root offset change: {e}")
                QtWidgets.QMessageBox.critical(
                    self, "Loading Error",
                    "Failed to load image group after changing root offset. Please ensure that the new offset is valid."
                )
        else:
            print("No multispectral roots found after changing root offset.")
            QtWidgets.QMessageBox.warning(
                self, "No Images",
                "No image root names were found after changing root offset. Please ensure that the selected folders contain valid images."
            )


    def filter_polygons_by_bounds(self, polygons_data, new_width, new_height):
        """
        Removes polygons that have any vertex outside [0, new_width) x [0, new_height).
        Returns a filtered list of polygons that all fit within the image.
        """
        filtered = []
        for poly_data in polygons_data:
            points = poly_data['points']
            inside = True
            for pt in points:
                if not (0 <= pt.x() < new_width and 0 <= pt.y() < new_height):
                    inside = False
                    break
            if inside:
                filtered.append(poly_data)
        return filtered

    def set_project_folder(self, folder_path):
        """
        Sets the project folder path for this ProjectTab.
        :param folder_path: The path to the project folder.
        """
        self.project_folder = folder_path
        logging.info(f"ProjectTab '{self.project_name}' set to use project folder: {self.project_folder}")
        
        # Implement additional logic as needed, such as updating file paths,
        # loading existing data from the project folder, etc.
        self.load_existing_data()     
        

               
    def handle_tab_switch_request(self, project_name):
        """
        Handles requests from the PolygonManager to switch tabs.
        """
        # Emit a signal to notify MainWindow. Define a new signal in ProjectTab.
        self.parentWidget().parentWidget().switch_to_specific_tab(project_name)

    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.project_name}_log.log',
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        logging.info(f"Initialized ProjectTab for '{self.project_name}'.")
    def close_parent(self):
        """
        Closes the current tab by instructing the parent QTabWidget to remove this tab.
        """
        parent = self.parent()
        if isinstance(parent, QtWidgets.QTabWidget):
            index = parent.indexOf(self)
            if index != -1:
                parent.removeTab(index)
                logging.info(f"Closed tab: {self.project_name}")
        else:
            logging.error("Parent is not a QTabWidget. Cannot close the tab.")


    def get_polygons_in_image_coords(self, viewer):
        """
        Extracts all polygons from the viewer, converting scene coords to image coords.
        Returns a list of dictionaries with polygon names and their points in image coordinates.
        """
        polygons_data = []
        for item in viewer.get_all_polygons():
            if isinstance(item, EditablePolygonItem):
                scene_poly = item.polygon
                name = item.name

                # Convert each point from scene to image coordinate
                image_points = []
                for p in scene_poly:
                    image_point = self.scene_to_image_coords(viewer, p)
                    image_points.append(image_point)

                polygons_data.append({
                    'name': name,
                    'points': image_points
                })
        return polygons_data

    def scene_to_image_coords(self, viewer, scene_point):
        """
        Map scene -> image pixels using the pixmap item’s own transform.
        Accounts for offsets, zoom, and any view transforms.
        """
        if viewer is None or viewer.image_data is None or viewer._image is None:
            return scene_point

        pixitem = viewer._image
        img_h, img_w = viewer.image_data.image.shape[:2]
        pixmap = pixitem.pixmap()
        pw, ph = max(1, pixmap.width()), max(1, pixmap.height())

        # scene -> item (pixmap) coordinates
        item_pt = pixitem.mapFromScene(scene_point)
        x_img = float(item_pt.x()) * (img_w / float(pw))
        y_img = float(item_pt.y()) * (img_h / float(ph))
        return QtCore.QPointF(x_img, y_img)

    def image_to_scene_coords(self, viewer, image_point):
        """
        Map image pixels -> scene using the pixmap item’s transform.
        """
        if viewer is None or viewer.image_data is None or viewer._image is None:
            return image_point

        pixitem = viewer._image
        img_h, img_w = viewer.image_data.image.shape[:2]
        pixmap = pixitem.pixmap()
        pw, ph = max(1, pixmap.width()), max(1, pixmap.height())

        x_pix = float(image_point.x()) * (pw / float(img_w))
        y_pix = float(image_point.y()) * (ph / float(img_h))
        return pixitem.mapToScene(QtCore.QPointF(x_pix, y_pix))

    def show_machine_learning_manager(self):
        ml_manager = MachineLearningManager(self)
        ml_manager.exec_()
    def update_all_polygons(self):
        # Store polygons grouped by logical names
        for group_name, file_polygons in self.all_polygons.items():
            for filepath, polygon_data in file_polygons.items():
                # Update polygon points
                viewer = self.get_viewer_by_filepath(filepath)
                if viewer:
                    for item in viewer.get_all_polygons():
                        if item.name == group_name:
                            # Use 'polygon' attribute if available; otherwise, use 'points'
                            if hasattr(item, 'polygon'):
                                poly = item.polygon
                            elif hasattr(item, 'points'):
                                poly = item.points
                            else:
                                continue  # Skip if neither attribute exists

                            transformed_polygon = item.mapToScene(poly)
                            points = [(point.x(), point.y()) for point in transformed_polygon]
                            self.all_polygons[group_name][filepath]['points'] = points


    
 

    def show_polygon_manager(self):
            all_polygon_groups = self.all_polygons
            self.polygon_manager.set_polygons(all_polygon_groups)
            self.polygon_manager.show()

    def update_polygon_manager(self):
        if self.polygon_manager.isVisible():
            self.polygon_manager.set_polygons(self.all_polygons)
            # Also update current root in PolygonManager
            current_root = self.root_names[self.current_root_index] if self.root_names else None
            self.polygon_manager.set_current_root(current_root, self.image_data_groups)

    def select_polygon_from_manager(self, list_item):
        group_name = list_item.data(QtCore.Qt.UserRole)
        if group_name:
            # Iterate through all polygons to find and select them
            for widget in self.viewer_widgets:
                viewer = widget['viewer']
                for item in viewer.get_all_polygons():
                    if item.name == group_name:
                        item.setSelected(True)
                        viewer.centerOn(item)
            # Removed pop-up
        else:
            pass  # Handle cases where group_name is invalid if necessary


    def prev_group(self):
        """Go to the previous root (single hop, centralized)."""
        if not self.multispectral_root_names:
            return
        target = self.current_root_index - 1
        if target < 0:
            print("Already at the first root group.")
            return
        self._jump_to_index(target)


    def next_group(self):
        """Go to the next root (single hop, centralized)."""
        if not self.multispectral_root_names:
            return
        target = self.current_root_index + 1
        if target >= len(self.multispectral_root_names):
            print("Already at the last root group.")
            return
        self._jump_to_index(target)

    

   
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
            
            
            
            
    def build_kdtree(self, gps_data):
        coordinates = [(item['latitude'], item['longitude']) for item in gps_data]
        filenames = [item['filename'] for item in gps_data]
        tree = KDTree(coordinates)
        return tree, filenames, coordinates

    def find_nearest_images(self, polygon_coords, tree, filenames, coordinates):
        distance, index = tree.query(polygon_coords)
        closest_filename = filenames[index]
        closest_coord = coordinates[index]
        distance_meters = geodesic(polygon_coords, closest_coord).meters
        return closest_filename, distance_meters
        
            
 
    def save_polygons(self):
        # Save polygons for current group to CSV
        self.save_polygons_to_csv() 


    def get_polygons_missing_coordinates(self):
        """
        Return (group_name, filepath) for polygons that truly need geo coords
        but don't have them.  skip polygons whose coord_space is not 'geo'.
        """
        missing = []
        for group_name, polygons in (self.all_polygons or {}).items():
            for filepath, polygon_data in (polygons or {}).items():
                pd = polygon_data or {}
                coord_space = (pd.get("coord_space") or "").lower()
                if coord_space and coord_space != "geo":
                    # Image-space polygon: do not require lat/lon
                    continue
                coords = pd.get("coordinates")
                if not isinstance(coords, dict):
                    missing.append((group_name, filepath))
                    continue
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                # treat empty strings as missing too
                if lat in (None, "") or lon in (None, ""):
                    missing.append((group_name, filepath))
        return missing

   
    def open_image_folders(self):
        """
        Opens the currently used image folders (multispectral and thermal/RGB) in the file explorer.
        Ignores the synthetic/fake second folder created when the user skipped it.
        """
        import os, platform, subprocess
        from PyQt5 import QtWidgets

        FAKE_SENTINEL = "_FAKE_SECOND_FOLDER_"

        folders_to_open = []

        # Multispectral folder
        ms = getattr(self, "current_folder_path", "")
        if ms and os.path.isdir(ms):
            folders_to_open.append(ms)

        # Thermal/RGB folder — skip if fake or flagged
        trgb = getattr(self, "thermal_rgb_folder_path", "")
        is_fake = (
            getattr(self, "_dual_folder_fake_second", False)
            or (trgb and os.path.basename(os.path.normpath(trgb)) == FAKE_SENTINEL)
        )
        if trgb and not is_fake and os.path.isdir(trgb):
            folders_to_open.append(trgb)

        if not folders_to_open:
            QtWidgets.QMessageBox.warning(self, "No Folders", "No image folders are currently open.")
            return

        for folder in folders_to_open:
            try:
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Folder Open Error",
                                              f"Could not open folder '{folder}': {e}")

# ------------------- Data Management Methods -------------------

    def reset_data_structures(self):
        """
        Resets all relevant data structures to their initial state.
        """
        self.all_polygons.clear()
        self.project_folder = ""
        self.current_folder_path = ""
        self.thermal_rgb_folder_path = ""
        self.current_root_index = 0
        self.root_offset = 0
        self.viewer_widgets = []
        self.image_data_groups.clear()
        self.root_names.clear()
        self.multispectral_image_data_groups.clear()
        self.thermal_rgb_image_data_groups.clear()
        self.multispectral_root_names.clear()
        self.thermal_rgb_root_names.clear()

    def clear_image_grid(self):
        """
        Clears all widgets from the image grid layout.
        """
        while self.image_grid_layout.count():
            item = self.image_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def matching_roots_dialog(self):
        """
        Implements a dialog that allows the user to select matching roots from both multispectral and thermal/RGB sets.
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set Matching Roots")
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel("Select matching roots for Multispectral and Thermal/RGB images")
        layout.addWidget(label)

        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        multispectral_list = QtWidgets.QListWidget()
        multispectral_list.addItems(self.multispectral_root_names)
        multispectral_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        multispectral_list_label = QtWidgets.QLabel("Multispectral Roots")
        v_layout1 = QtWidgets.QVBoxLayout()
        v_layout1.addWidget(multispectral_list_label)
        v_layout1.addWidget(multispectral_list)
        h_layout.addLayout(v_layout1)

        thermal_rgb_list = QtWidgets.QListWidget()
        thermal_rgb_list.addItems(self.thermal_rgb_root_names)
        thermal_rgb_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        thermal_rgb_list_label = QtWidgets.QLabel("Thermal/RGB Roots")
        v_layout2 = QtWidgets.QVBoxLayout()
        v_layout2.addWidget(thermal_rgb_list_label)
        v_layout2.addWidget(thermal_rgb_list)
        h_layout.addLayout(v_layout2)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            multispectral_selected = multispectral_list.currentRow()
            thermal_rgb_selected = thermal_rgb_list.currentRow()

            if multispectral_selected == -1 or thermal_rgb_selected == -1:
                QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select roots from both lists.")
                return

            # Calculate offset
            self.root_offset = thermal_rgb_selected - multispectral_selected

            # Set current_root_index to multispectral_selected
            self.current_root_index = multispectral_selected

            # Map matching roots
            self.map_matching_roots()

            # Load the images
            self.load_image_group(self.multispectral_root_names[self.current_root_index])

        else:
            # User canceled
            pass
    def map_matching_roots(self):
        """
        Build a root mapping that records *all images* loaded per root, divided
        into folder1 (multispectral) and folder2 (thermal/RGB).still keep
        root_id mapping intact.

        root_mapping_dict format (example):
        {
          "1": {
            "folder1": {"dir": "/path/multi/IMG_0114", "files": ["IMG_0114_0_radiance.tif","IMG_0114_1_radiance.tif"]},
            "folder2": {"dir": "/path/trgb/IMG_0114",  "files": ["IMG_0114_rgb.jpg","IMG_0114_thermal.tif"]},
            # Optional legacy convenience keys to avoid breaking old code:
            "RGB": "IMG_0114_rgb.jpg",
            "thermal": "IMG_0114_thermal.tif"
          },
          ...
        }
        """
        import os
        logger = logging.getLogger(__name__)
        logger.info("Mapping matching roots and generating root_mapping.json.")

        def _dir_or_none(paths):
            return os.path.dirname(paths[0]) if paths else None

        def _basenames(paths):
            return [os.path.basename(p) for p in paths]

        # Reset mappings
        self.root_id_mapping = {}
        self.id_to_root = {}
        self.root_mapping_dict = {}

        if self.mode == 'dual_folder':
            # Map each multispectral root to a root_id, and align a thermal/RGB root via offset
            for ms_idx, ms_root in enumerate(self.multispectral_root_names):
                tr_idx = ms_idx + self.root_offset
                tr_root = self.thermal_rgb_root_names[tr_idx] if 0 <= tr_idx < len(self.thermal_rgb_root_names) else None

                # Assign a unique root_id (1-based)
                root_id = ms_idx + 1
                self.root_id_mapping[ms_root] = root_id
                self.id_to_root[root_id] = ms_root
                if tr_root:
                    self.root_id_mapping[tr_root] = root_id  # same id for the matched pair

                # Gather paths
                ms_paths = self.multispectral_image_data_groups.get(ms_root, [])
                tr_paths = self.thermal_rgb_image_data_groups.get(tr_root, []) if tr_root else []

                # Build entry
                entry = {
                    "folder1": {
                        "dir": _dir_or_none(ms_paths),
                        "files": _basenames(ms_paths),
                    },
                    "folder2": {
                        "dir": _dir_or_none(tr_paths),
                        "files": _basenames(tr_paths),
                    }
                }

                # Legacy convenience keys (kept to avoid breaking code that expects RGB/thermal)
                # Use first two files (if any) from folder2 as RGB/thermal best guesses.
                if entry["folder2"]["files"]:
                    entry["RGB"] = entry["folder2"]["files"][0]
                    entry["thermal"] = entry["folder2"]["files"][1] if len(entry["folder2"]["files"]) > 1 else None
                else:
                    entry["RGB"] = None
                    entry["thermal"] = None

                self.root_mapping_dict[str(root_id)] = entry
                logger.debug(f"Root {root_id}: folder1={len(entry['folder1']['files'])} files, "
                             f"folder2={len(entry['folder2']['files'])} files")

            # Handle extra thermal/RGB roots that don’t align to a multispectral one via offset
       
            last_mapped = len(self.multispectral_root_names) + self.root_offset
            if last_mapped < len(self.thermal_rgb_root_names):
                for i, tr_root in enumerate(self.thermal_rgb_root_names[last_mapped:], start=1):
                    root_id = len(self.multispectral_root_names) + i
                    self.root_id_mapping[tr_root] = root_id
                    self.id_to_root[root_id] = tr_root

                    tr_paths = self.thermal_rgb_image_data_groups.get(tr_root, [])
                    entry = {
                        "folder1": {"dir": None, "files": []},
                        "folder2": {"dir": _dir_or_none(tr_paths), "files": _basenames(tr_paths)},
                    }
                    entry["RGB"] = entry["folder2"]["files"][0] if entry["folder2"]["files"] else None
                    entry["thermal"] = entry["folder2"]["files"][1] if len(entry["folder2"]["files"]) > 1 else None

                    self.root_mapping_dict[str(root_id)] = entry
                    logger.debug(f"Extra thermal root '{tr_root}' → Root ID {root_id}")

        elif self.mode == 'rgb_only':
            # Only folder1 is populated; folder2 stays empty
            for idx, root_name in enumerate(self.multispectral_root_names):
                root_id = idx + 1
                self.root_id_mapping[root_name] = root_id
                self.id_to_root[root_id] = root_name

                ms_paths = self.multispectral_image_data_groups.get(root_name, [])
                entry = {
                    "folder1": {"dir": _dir_or_none(ms_paths), "files": _basenames(ms_paths)},
                    "folder2": {"dir": None, "files": []},
                    "RGB": None,
                    "thermal": None,
                }
                self.root_mapping_dict[str(root_id)] = entry
                logger.debug(f"RGB-only root '{root_name}' → Root ID {root_id} ({len(ms_paths)} files)")

        else:
            logger.warning(f"Unknown mode '{self.mode}'. Root mapping not performed.")
            QtWidgets.QMessageBox.warning(
                self,
                "Unknown Mode",
                f"The application is in an unknown mode: '{self.mode}'. Root mapping was not performed."
            )
            return

        # Persist
        self.save_root_mapping_json()
        logger.info("Completed root ID mapping and saved root_mapping.json.")




    def save_root_mapping_json(self, mapping_file_path=None):
        """
        Saves self.root_mapping_dict to JSON.
        Schema per root_id (string key):
          {
            "folder1": {"dir": <str or null>, "files": [<basename>, ...]},
            "folder2": {"dir": <str or null>, "files": [<basename>, ...]},
            "RGB": <basename or null>,        # legacy convenience key
            "thermal": <basename or null>     # legacy convenience key
          }
        """
        logger = logging.getLogger(__name__)
        if not mapping_file_path:
            mapping_file_path = (os.path.join(self.project_folder, 'root_mapping.json')
                                 if self.project_folder else os.path.join(os.getcwd(), 'root_mapping.json'))

        try:
            with open(mapping_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.root_mapping_dict, f, indent=4)
            logger.info(f"Saved root mapping to {mapping_file_path}")
        except Exception as e:
            logger.error(f"Failed to save root mapping to {mapping_file_path}: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                "Save Mapping Error",
                f"Failed to save root mapping to {mapping_file_path}:\n{e}"
            )



    def _root_id_for_index(self, idx: int) -> int:
        """
        Return the numeric root ID for a multispectral index.
        Uses self.root_id_mapping / self.id_to_root built by map_matching_roots().
        Falls back to 1-based index if mapping isn't ready.
        """
        try:
            name = self.multispectral_root_names[idx]
        except Exception:
            return idx + 1
        rid = getattr(self, "root_id_mapping", {}).get(name)
        return int(rid) if rid is not None else (idx + 1)


    def _index_for_root_id(self, rid: int):
        """Return multispectral index for a numeric root ID (prefer MS names)."""
        try:
            rid = int(rid)
        except Exception:
            return None

        ms_list = getattr(self, "multispectral_root_names", []) or []

        # Fast path via id_to_root
        name = getattr(self, "id_to_root", {}).get(rid)
        if name in ms_list:
            return ms_list.index(name)

        # Otherwise scan root_id_mapping for any MS name with this rid
        rim = getattr(self, "root_id_mapping", {})
        for i, ms_name in enumerate(ms_list):
            if rim.get(ms_name) == rid:
                return i
        return None

    # ------------------- Image Loading Methods -------------------

    def load_multispectral_images_from_folder(self, folder_path):
        self.multispectral_image_data_groups.clear()
        self.multispectral_root_names.clear()

        # Collect all image files in the folder
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]

        # Group images by root name
        for filepath in image_files:
            filename = os.path.basename(filepath)
            root_name = self.extract_root_name(filename)
            self.multispectral_image_data_groups[root_name].append(os.path.abspath(filepath))

        # Sort root names
        self.multispectral_root_names = sorted(self.multispectral_image_data_groups.keys())

        # Synchronize self.root_names with multispectral_root_names
        self.root_names = self.multispectral_root_names.copy()

    def load_thermal_rgb_images_from_folder(self, folder_path):
        if self.mode != 'dual_folder':
            # Do not load Thermal/RGB images in rgb_only mode
            return

        self.thermal_rgb_image_data_groups.clear()
        self.thermal_rgb_root_names.clear()

        # Collect all image files in the folder
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]

        # Group images by root name
        for filepath in image_files:
            filename = os.path.basename(filepath)
            root_name = self.extract_root_name(filename)
            self.thermal_rgb_image_data_groups[root_name].append(os.path.abspath(filepath))

        # Sort root names
        self.thermal_rgb_root_names = sorted(self.thermal_rgb_image_data_groups.keys())

    def extract_root_name(self, filename):
        """
        Extracts the root name from the filename using regex or fallback methods.
        Example:
        - 'IMG_0001_0_radiance.tif' -> 'IMG_0001'
        """
        match = re.match(r'^(IMG_\d+)', filename)
        if match:
            return match.group(1)
        else:
            # Fallback: extract root by removing the last two underscores and everything after
            parts = filename.split('_')
            if len(parts) >= 3:
                root_name = '_'.join(parts[:2])  # e.g., 'IMG_0001' from 'IMG_0001_0_radiance'
            elif len(parts) > 1:
                root_name = '_'.join(parts[:-2])
            else:
                root_name = os.path.splitext(filename)[0]
            return root_name
  
   

    def convert_cv_to_pixmap(self, cv_img, prefer_last_band: bool = False):
        from PyQt5 import QtGui
        import cv2, numpy as np, sip

        if cv_img is None:
            return None

        # --- choose what to PREVIEW (display only) ---
        # keep the full stack in image_data.image (hover/CSV still see all bands incl. expression)
        if cv_img.ndim == 3:
            C = cv_img.shape[2]
            if prefer_last_band and C > 3:
                preview = cv_img[..., -1]            # show expression plane as grayscale
            else:
                preview = cv_img[..., :min(3, C)]    # original behavior for stacks/RGB
        else:
            preview = cv_img

        disp = _normalize_for_display(preview)
        if disp is None:
            return None

        # Ensure contiguous for QImage
        disp = np.ascontiguousarray(disp)

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
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                try:
                    qimg = QtGui.QImage(sip.voidptr(rgb.ctypes.data), w, h, rgb.strides[0],
                                        QtGui.QImage.Format_RGB888)
                except TypeError:
                    buf = rgb.tobytes()
                    qimg = QtGui.QImage(buf, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap.fromImage(qimg.copy())


 # ------------------- Mapping and Loading Methods -------------------
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

    def run_correction_process(self):
        """
        Runs the polygon correction process as per the provided script.
        """
        # Configure logging
        log_file = os.path.join(self.corrected_dir, 'rename_polygons.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        JSONS_DIR = self.jsons_dir
        POLYGONS_DIR = self.polygons_dir
        CORRECTED_DIR = self.corrected_dir

        # Create corrected directory if it doesn't exist
        os.makedirs(CORRECTED_DIR, exist_ok=True)

        # Regular expression patterns
        POLYGON_FILENAME_REGEX = re.compile(r'^([a-zA-Z0-9]+_)IMG_\d+_(\d+)_radiance_polygons\.json$')
        CLOSEST_IMAGE_REGEX = re.compile(r'^(IMG_\d+)_(\d+)_(radiance)$')

        def extract_prefix_and_index(polygon_filename):
            match = POLYGON_FILENAME_REGEX.match(polygon_filename)
            if match:
                prefix = match.group(1)  # e.g., 'a1_'
                polygon_index = match.group(2)  # e.g., '0'
                return prefix, polygon_index
            else:
                return None, None

        def replace_image_index(closest_image_base, polygon_index):
            match = CLOSEST_IMAGE_REGEX.match(closest_image_base)
            if match:
                img_prefix = match.group(1)  # e.g., 'IMG_0113'
                suffix = match.group(3)  # e.g., 'radiance'
                new_base = f"{img_prefix}_{polygon_index}_{suffix}"
                return new_base
            else:
                return None

        # Iterate over each JSON file in the jsons directory
        for json_filename in os.listdir(JSONS_DIR):
            if not json_filename.endswith('_polygons_results.json'):
                continue  # Skip files that do not match the pattern

            json_path = os.path.join(JSONS_DIR, json_filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from file {json_filename}: {e}")
                print(f"Error decoding JSON from file {json_filename}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error reading file {json_filename}: {e}")
                print(f"Unexpected error reading file {json_filename}: {e}")
                continue

            if not isinstance(data, list) or not data:
                logging.warning(f"No data found in {json_filename}")
                print(f"No data found in {json_filename}")
                continue

            # Determine the corresponding original polygon file in the polygons directory
            if json_filename.endswith('_polygons_results.json'):
                base_polygon_filename = json_filename.replace('_polygons_results.json', '_polygons.json')
            else:
                logging.warning(f"Filename {json_filename} does not follow the expected pattern.")
                print(f"Filename {json_filename} does not follow the expected pattern.")
                continue

            original_polygon_path = os.path.join(POLYGONS_DIR, base_polygon_filename)

            if not os.path.exists(original_polygon_path):
                logging.error(f"Polygon file {base_polygon_filename} does not exist for JSON file {json_filename}")
                print(f"Polygon file {base_polygon_filename} does not exist for JSON file {json_filename}")
                continue

            # Extract prefix and polygon index from the polygon filename
            prefix, polygon_index = extract_prefix_and_index(base_polygon_filename)
            if prefix is None or polygon_index is None:
                logging.error(f"Cannot extract prefix and index from polygon filename {base_polygon_filename}")
                print(f"Cannot extract prefix and index from polygon filename {base_polygon_filename}")
                continue

            # Process each entry in the JSON array
            for entry in data:
                closest_image = entry.get('closest_image')

                if not closest_image:
                    logging.warning(f"No 'closest_image' found in {json_filename} for polygon index {polygon_index}")
                    print(f"No 'closest_image' found in {json_filename} for polygon index {polygon_index}")
                    continue

                # Extract the base name without extension from closest_image
                closest_image_base = os.path.splitext(closest_image)[0]  # Removes '.tif'

                # Replace the index in closest_image_base with polygon_index
                new_image_base = replace_image_index(closest_image_base, polygon_index)
                if not new_image_base:
                    logging.error(f"Failed to replace index in closest_image_base '{closest_image_base}' with polygon_index '{polygon_index}'")
                    print(f"Failed to replace index in closest_image_base '{closest_image_base}' with polygon_index '{polygon_index}'")
                    continue

                # Construct the new polygon filename with prefix
                new_polygon_filename = f"{prefix}{new_image_base}_polygons.json"

                # Destination path in the corrected directory with the new unique name
                corrected_polygon_path = os.path.join(CORRECTED_DIR, new_polygon_filename)

                try:
                    shutil.copy2(original_polygon_path, corrected_polygon_path)
                    logging.info(f"Copied and renamed {base_polygon_filename} to {new_polygon_filename}")
                    print(f"Copied and renamed {base_polygon_filename} to {new_polygon_filename}")
                except Exception as e:
                    logging.error(f"Failed to copy {base_polygon_filename} to {new_polygon_filename}: {e}")
                    print(f"Failed to copy {base_polygon_filename} to {new_polygon_filename}: {e}")
                    continue

                # Now, update the 'root' field in the copied polygon JSON file
                try:
                    with open(corrected_polygon_path, 'r') as f:
                        polygon_data = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from copied file {new_polygon_filename}: {e}")
                    print(f"Error decoding JSON from copied file {new_polygon_filename}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error reading copied file {new_polygon_filename}: {e}")
                    print(f"Unexpected error reading copied file {new_polygon_filename}: {e}")
                    continue

                # Extract the number from the new filename to update 'root'
                # Example new_polygon_filename: a1_IMG_0113_0_radiance_polygons.json
              
                match = re.search(r'IMG_(\d+)_\d+_radiance', new_image_base)
                if match:
                    extracted_number_str = match.group(1)  # '0113'
                    try:
                        extracted_number = int(extracted_number_str)
                        new_root = str(extracted_number + 1)  # 113 + 1 = 114
                    except ValueError:
                        logging.error(f"Invalid number extracted from filename {new_polygon_filename}: {extracted_number_str}")
                        print(f"Invalid number extracted from filename {new_polygon_filename}: {extracted_number_str}")
                        continue
                else:
                    logging.error(f"Failed to extract number from new_image_base '{new_image_base}' in filename {new_polygon_filename}")
                    print(f"Failed to extract number from new_image_base '{new_image_base}' in filename {new_polygon_filename}")
                    continue

                # Update the 'root' field in the polygon JSON data
                if isinstance(polygon_data, dict):  # Ensure it's a dictionary
                    if 'root' in polygon_data:
                        original_root = polygon_data['root']
                        polygon_data['root'] = new_root
                        logging.info(f"Updated 'root' from {original_root} to {new_root} in {new_polygon_filename}")
                        print(f"Updated 'root' from {original_root} to {new_root} in {new_polygon_filename}")
                    else:
                        # If 'root' field does not exist, add it
                        polygon_data['root'] = new_root
                        logging.info(f"Added 'root': {new_root} in {new_polygon_filename}")
                        print(f"Added 'root': {new_root} in {new_polygon_filename}")

                    # Write the updated JSON back to the copied file
                    try:
                        with open(corrected_polygon_path, 'w') as f:
                            json.dump(polygon_data, f, indent=4)
                        logging.info(f"Successfully updated 'root' in {new_polygon_filename}")
                        print(f"Successfully updated 'root' in {new_polygon_filename}")
                    except Exception as e:
                        logging.error(f"Failed to write updated JSON to {new_polygon_filename}: {e}")
                        print(f"Failed to write updated JSON to {new_polygon_filename}: {e}")
                        continue
                else:
                    logging.error(f"Polygon data in {new_polygon_filename} is not a dictionary.")
                    print(f"Polygon data in {new_polygon_filename} is not a dictionary.")
                    continue 

    # ------------------- Helper Methods -------------------

    def get_viewer_by_filepath(self, filepath):
        for widget in self.viewer_widgets:
            if widget['image_data'].filepath == filepath:
                return widget['viewer']
        return None

    def get_root_by_filepath(self, filepath):
        """
        Retrieves the root name associated with a given filepath.
        """
        for root, paths in self.image_data_groups.items():
            if filepath in paths:
                return root
        return None

    def get_current_root_name(self):
        if 0 <= self.current_root_index < len(self.multispectral_root_names):
            return self.multispectral_root_names[self.current_root_index]
        return None


    def slider_released(self):
        # Get the mapped numeric root ID for the current slider position
        rid_str = self._root_number_for_index(self.group_slider.value())
        try:
            rid = int(rid_str)
        except ValueError:
            rid = self.group_slider.value() + 1  # sensible fallback
        self.go_to_root_id(rid)

    

    def update_progress(self, image_data):
        self.images_loaded += 1
        self.progress_dialog.setValue(self.images_loaded)
        if self.progress_dialog.wasCanceled():
            # If user cancels, stop the worker
            if self.image_loader_worker:
                self.image_loader_worker.stop()

    def on_all_images_loaded(self):
        # Restore the cursor and close the progress dialog
        QtWidgets.QApplication.restoreOverrideCursor()
        self.progress_dialog.close()
        print("All images have been loaded.")

    def cancel_image_loading(self):
        if self.image_loader_worker:
            self.image_loader_worker.stop()
        print("Image loading has been canceled by the user.")            


    def save_polygons_to_json(self, root_name: str = None):
        """
        Write polygons to `<project>/polygons/{group}_{imageBase}_polygons.json`
        with reliable coordinates for BOTH MS and Thermal/RGB roots in dual-folder mode.

        If root_name is provided, only polygons associated with that logical root are saved.
        """
        import os, json, logging

        if not getattr(self, "project_folder", None):
            polygons_dir = os.path.join(os.getcwd(), "polygons")
        else:
            polygons_dir = os.path.join(self.project_folder, "polygons")
        os.makedirs(polygons_dir, exist_ok=True)

        # Helper: normalize filepath -> root name (fallback if method missing)
        def _root_for_fp(fp: str):
            if hasattr(self, "get_root_by_filepath"):
                return self.get_root_by_filepath(fp)
            # Fallback: scan groups (slow but safe)
            for rn, paths in self.multispectral_image_data_groups.items():
                if fp in paths:
                    return rn
            for rn, paths in getattr(self, "thermal_rgb_image_data_groups", {}).items():
                if fp in paths:
                    return rn
            return None

        # Helper: coordinate lookup with mirroring & EXIF fallback
        def _coords_for_root(image_root: str, filepath: str):
            # 1) Root map
            coords = self.root_coordinates.get(image_root)
            if coords:
                return coords
            # 2) Paired root
            if self.mode == 'dual_folder':
                paired = self._paired_ms_root(image_root) or self._paired_trgb_root(image_root)
                if paired:
                    coords = self.root_coordinates.get(paired)
                    if coords:
                        return coords
            # 3) EXIF fallback from this file; cache under root
            one = self._first_gps_from_files([filepath])
            if one:
                self.root_coordinates[image_root] = one
                return one
            # 4) Nothing available
            return {'latitude': None, 'longitude': None}

        # Build the subset to save
        if root_name:
            # Save only polygons tied to this logical root's filepaths (MS + paired TRGB)
            filepaths_in_root = set(self.image_data_groups.get(root_name, []))
            to_save = {
                g: {fp: pd for fp, pd in files.items() if fp in filepaths_in_root}
                for g, files in self.all_polygons.items()
            }
            # Drop empty groups
            to_save = {g: files for g, files in to_save.items() if files}
        else:
            to_save = self.all_polygons

        # Write each polygon JSON
        for group_name, files in to_save.items():
            for filepath, polygon_data in files.items():
                try:
                    image_root = _root_for_fp(filepath) or ""
                    root_id    = str(self.root_id_mapping.get(image_root, "0"))

                    # Decide coordinates (prefer any already on the polygon)
                    poly_coords = None
                    if isinstance(polygon_data, dict):
                        c = polygon_data.get('coordinates')
                        if isinstance(c, dict) and 'latitude' in c and 'longitude' in c:
                            poly_coords = c

                    coords = poly_coords or _coords_for_root(image_root, filepath)

                    # Shape metadata
                    poly_type    = (polygon_data.get('type') if isinstance(polygon_data, dict) else None) or 'polygon'
                    coord_space  = (polygon_data.get('coord_space') if isinstance(polygon_data, dict) else None) or 'scene'
                    points_any   = (polygon_data.get('points') if isinstance(polygon_data, dict) else None) or []

                    # Construct output
                    data_to_save = {
                        'points': points_any,
                        'name':   (polygon_data.get('name') if isinstance(polygon_data, dict) else None) or group_name,
                        'root':   root_id,
                        'coordinates': coords,
                        'type':   poly_type,
                        'coord_space': coord_space
                    }

                    base_filename     = os.path.splitext(os.path.basename(filepath))[0]
                    polygon_filename  = f"{group_name}_{base_filename}_polygons.json"
                    polygon_filepath  = os.path.join(polygons_dir, polygon_filename)

                    with open(polygon_filepath, 'w', encoding='utf-8') as f:
                        json.dump(data_to_save, f, indent=4)

                    logging.info("Saved %s", polygon_filepath)
                except Exception as e:
                    logging.error("Failed to save polygon for file '%s': %s", filepath, e)


    def compute_root_coordinates(self, root_name=None):
        """
        Computes and stores GPS coordinates for roots.
        If root_name is specified, computes for that root only.
        Otherwise, computes for all roots.
        """
        if root_name:
            roots_to_compute = [root_name]
        else:
            roots_to_compute = self.root_names

        for root in roots_to_compute:
            # Get image filepaths from both multispectral and thermal/RGB groups
            image_filepaths = self.multispectral_image_data_groups.get(root, []) + \
                              self.thermal_rgb_image_data_groups.get(root, [])
            coords_found = False
            for filepath in image_filepaths:
                coords = self.get_gps_coordinates(filepath)
                if coords['latitude'] is not None and coords['longitude'] is not None:
                    self.root_coordinates[root] = coords
                    logging.debug(f"Root '{root}' GPS coordinates: ({coords['latitude']}, {coords['longitude']}) from image '{filepath}'")
                    coords_found = True
                    break  # Stop after finding the first image with valid GPS data
            if not coords_found:
                self.root_coordinates[root] = {'latitude': None, 'longitude': None}
                logging.debug(f"Root '{root}' does not have any images with GPS coordinates.")
        logging.info(f"Completed recomputing root coordinates: {self.root_coordinates}")

    def get_gps_coordinates(self, filepath):
        """
        Extract GPS coordinates from an image file using exifread.
        Returns a dictionary with 'latitude' and 'longitude' in decimal degrees,
        or {'latitude': None, 'longitude': None} if not found.
        """
        try:
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f, details=False)
           
            gps_latitude = tags.get('GPS GPSLatitude')
            gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
            gps_longitude = tags.get('GPS GPSLongitude')
            gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
           
            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degrees(gps_latitude)
                if gps_latitude_ref.values != 'N':
                    lat = -lat
                lon = self.convert_to_degrees(gps_longitude)
                if gps_longitude_ref.values != 'E':
                    lon = -lon
                return {'latitude': lat, 'longitude': lon}
            else:
                return {'latitude': None, 'longitude': None}
        except Exception as e:
            print(f"Error extracting GPS data from {filepath}: {e}")
            return {'latitude': None, 'longitude': None} 

    def convert_to_degrees(self, value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degrees in float format.
        """
        try:
            d = float(value.values[0].num) / float(value.values[0].den)
            m = float(value.values[1].num) / float(value.values[1].den)
            s = float(value.values[2].num) / float(value.values[2].den)
            return d + (m / 60.0) + (s / 3600.0)
        except (AttributeError, IndexError, ZeroDivisionError) as e:
            print(f"Error converting GPS to degrees: {e}")
            return None

  
    # ------------------- Polygon Management Methods -------------------
    def on_polygon_drawn(self, polygon_item):
        sender_viewer = self.sender()
        image_id = os.path.splitext(os.path.basename(sender_viewer.image_data.filepath))[0]

        # Editing existing group?
        if sender_viewer.pending_group_name:
            logical_name = sender_viewer.pending_group_name
            sender_viewer.pending_group_name = None
        else:
            logical_name, ok = QtWidgets.QInputDialog.getText(
                self, "Logical Object Name", "Enter logical name for the polygon (e.g., 'tree1'):")
            if not (ok and logical_name.strip()):
                sender_viewer._scene.removeItem(polygon_item)
                print("Unnamed polygon was not saved.")
                return
            logical_name = logical_name.strip()

        # Assign the name
        polygon_item.name = logical_name

        # Root number for this image
        root_name = self.get_root_by_filepath(sender_viewer.image_data.filepath)
        if root_name:
            try:
                root_number = str(self.root_names.index(root_name) + 1)
            except ValueError:
                root_number = "0"
        else:
            root_number = "0"

        # Determine geometry + type
        if hasattr(polygon_item, "polygon"):
            scene_poly = polygon_item.polygon
            shape_type = "polygon"
        elif hasattr(polygon_item, "points"):
            scene_poly = polygon_item.points
            shape_type = "point"
        else:
            scene_poly = QtGui.QPolygonF()
            shape_type = "polygon"

        # Convert SCENE -> IMAGE and save image pixels as source of truth
        img_points = []
        for p in scene_poly:
            ip = self.scene_to_image_coords(sender_viewer, p)
            img_points.append((float(ip.x()), float(ip.y())))

        if logical_name not in self.all_polygons:
            self.all_polygons[logical_name] = {}

        self.all_polygons[logical_name][sender_viewer.image_data.filepath] = {
            'points': img_points,           # image pixels!
            'coord_space': 'image',
            'name': logical_name,
            'root': root_number,
            'type': shape_type
        }

        # Sync to other images in same root (uses image->scene at target)
        if self.sync_enabled:
            self.add_polygon_to_other_images(sender_viewer, scene_poly, logical_name, action="copy", shape_type=shape_type)

        # Persist + UI updates
        self.save_polygons_to_json(root_name=self.get_current_root_name())
        self.update_polygon_manager()
        self.generate_thumbnail(logical_name, sender_viewer.image_data)

    def on_polygon_modified(self):
        # A polygon was modified
        self.update_all_polygons()
        self.save_polygons_to_json(root_name=self.get_current_root_name())


    def toggle_sync(self):
        self.sync_enabled = self.syncAct.isChecked()
        if self.sync_enabled:
            self.syncAct.setText("Sync On")
        else:
            self.syncAct.setText("Sync Off")
            
            


    def go_to_root(self):
        from PyQt5 import QtWidgets
        text = self.root_number_input.text().strip()
        if not text:
            return
        try:
            rid = int(text)
        except ValueError:
            QtWidgets.QMessageBox.information(self, "Go to root", f"Please enter a number (got '{text}').")
            return
        self.go_to_root_id(rid)

    def go_to_root_id(self, rid: int):
        idx = self._index_for_root_id(rid)
        if idx is None:
            print(f"No multispectral root for root ID {rid}.")
            return
        self._jump_to_index(idx)    

        # Save current polygons before leaving
        if 0 <= self.current_root_index < len(self.multispectral_root_names):
            try:
                cur_root = self.multispectral_root_names[self.current_root_index]
                self.save_polygons_to_json(root_name=cur_root)
            except Exception:
                pass

        # Jump
        self.current_root_index = idx

        # Update slider without re-triggering sliderReleased
        block = self.group_slider.blockSignals(True)
        self.group_slider.setValue(idx)
        self.group_slider.blockSignals(block)

        # Keep label in sync
        self.update_slider_label(idx)

        # Load the new group
        self.load_image_group(self.multispectral_root_names[idx])

  
    def save_polygon_group_to_file(self, group_name, filepath, polygon_data):
        """
        Saves polygon data to a JSON file, including the 'root' number and 'coordinates' field.
        """
        if not self.project_folder:
            # If no project folder is set, polygons are saved in 'polygons' directory in current working directory
            polygons_dir = os.path.join(os.getcwd(), 'polygons')
        else:
            # Save polygons inside the project folder
            polygons_dir = os.path.join(self.project_folder, 'polygons')
        os.makedirs(polygons_dir, exist_ok=True)

        # Save polygons to a JSON file named after the group and image file
        filename = os.path.basename(filepath)
        polygon_filename = f"{group_name}_{os.path.splitext(filename)[0]}_polygons.json"
        polygon_filepath = os.path.join(polygons_dir, polygon_filename)

        try:
            # Determine the root number based on the image's group using root_id_mapping
            root_name = self.get_root_by_filepath(filepath)
            if root_name:
                root_id = self.root_id_mapping.get(root_name, 0)
                root_number = str(root_id)
            else:
                root_number = "0"  # Default value if root_name is None

            # Get coordinates for the root
            coordinates = self.root_coordinates.get(root_name, {'latitude': None, 'longitude': None})

            # Add 'root' and 'coordinates' field to polygon_data
            polygon_data_with_extra = polygon_data.copy()
            polygon_data_with_extra['root'] = root_number
            polygon_data_with_extra['coordinates'] = coordinates  # Add coordinates

            with open(polygon_filepath, 'w', encoding='utf-8') as f:
                json.dump(polygon_data_with_extra, f, indent=4)
            print(f"Saved polygon to {polygon_filepath}")
        except Exception as e:
            # Removed pop-up to prevent annoying sounds
            print(f"Could not save polygons for {filename}: {e}")

    def construct_thumbnail_name(self, logical_name, filepath, image_type='multispectral'):
        """
        Constructs the thumbnail filename based on the naming convention:
        - RGB: {logical_name}_{root_id}_RGB.jpg
        - Thermal: {logical_name}_{root_id}_IR.jpg
        - Multispectral: {logical_name}_{root_id}_{band_index}.jpg
        """
        root_name = self.get_root_by_filepath(filepath)
        if root_name:
            root_id = self.root_id_mapping.get(root_name, 0)
        else:
            root_id = 0  # Default value if root_name is None

        # Determine band index relative to its root group
        if root_name and root_name in self.image_data_groups:
            try:
                # Sort the images within the root to ensure consistent indexing
                sorted_images = sorted(self.image_data_groups[root_name])
                band_index = sorted_images.index(filepath) + 1  # Starting from 1
                suffix = f"_{band_index}" if image_type.lower() == 'multispectral' else ""
            except ValueError:
                band_index = 0  # Default value if filepath is not found in its root
                suffix = ""
        else:
            band_index = 0  # Default value if root_name is None or not found
            suffix = ""

        # Determine the suffix based on image_type
        if image_type.lower() == 'rgb':
            suffix = "_RGB"
        elif image_type.lower() == 'thermal':
            suffix = "_IR"
        elif image_type.lower() == 'multispectral' and band_index > 0:
            suffix = f"_{band_index}"
        else:
            suffix = ""

        # Incorporate the base filename to ensure uniqueness
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        thumbnail_filename = f"{logical_name}_{root_id}_{base_filename}{suffix}.jpg"
        print(f"Constructed Thumbnail Name: {thumbnail_filename}")  # Debugging Statement
        return thumbnail_filename


    def shrink_or_swell_shapely_polygon(self, polygon_points, factor=0.07, swell=False):
        ''' 
        Returns a list of points representing the shapely polygon 
        which is smaller or bigger by the passed factor.
        If swell = True, then it returns a bigger polygon, else smaller.
        '''
        # Create a shapely polygon from the points
        my_polygon = geometry.Polygon(polygon_points)

        # Use the passed factor
        shrink_factor = 0.07

        xs = list(my_polygon.exterior.coords.xy[0])
        ys = list(my_polygon.exterior.coords.xy[1])
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)
        center = geometry.Point(x_center, y_center)
        min_corner = geometry.Point(min(xs), min(ys))
        shrink_distance = center.distance(min_corner) * shrink_factor

        if swell:
            my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
        else:
            my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

        # Check if the polygon is valid after buffering
        if not my_polygon_resized.is_valid:
            logging.warning("Resized polygon is invalid. Returning original polygon.")
            return polygon_points

        if isinstance(my_polygon_resized, MultiPolygon):
            # Choose the polygon with the largest area
            my_polygon_resized = max(my_polygon_resized.geoms, key=lambda p: p.area)


        # Extract the coordinates and return as a list of points
        return list(my_polygon_resized.exterior.coords)





    def save_all_polygons(self):
        # Alias to save_polygons_to_csv
        self.save_polygons_to_csv()

    def copy_metadata_action(self):
        """
        Copies metadata from selected source images to target images.
        Users need to select source and target images via file dialogs.
        """
        # Select source images
        source_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Source Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not source_files:
            return  # Do nothing if no selection

        # Select target images
        target_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Target Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not target_files:
            return  # Do nothing if no selection

        if len(source_files) != len(target_files):
            print("Source and target selections do not match in number.")
            return  # Do nothing if counts do not match

        # Perform the metadata copy without confirmation pop-up
        copy_metadata_from_multiple_sources_parallel(source_files, target_files, exiftool_path, max_workers=8)
        # Removed pop-up to prevent annoying sounds
        print("Metadata has been successfully copied.")

    def clean_polygons(self, viewer):
            viewer.clear_polygons()
            # Update all_polygons dictionary
            self.update_all_polygons()
            self.save_polygons_to_json(root_name=self.get_current_root_name())
            self.update_polygon_manager()  
    def delete_polygons_for_viewer(self, viewer):
        """
        Deletes all polygons associated with the specified viewer's image.
        """
        # Get the filepath of the image from the viewer
        filepath = viewer.image_data.filepath
        print(f"Deleting polygons for image: {filepath}")
        
        # Identify all groups that have polygons for this filepath
        groups_with_polygons = [
            group_name for group_name, polygons in self.all_polygons.items()
            if filepath in polygons
        ]
        print(f"Groups to delete polygons from: {groups_with_polygons}")
        
        # Remove polygons from in-memory data structure
        for group_name in groups_with_polygons:
            del self.all_polygons[group_name][filepath]
            print(f"Removed polygon '{group_name}' from '{filepath}'")
            
            # If the group has no more polygons, remove the group
            if not self.all_polygons[group_name]:
                del self.all_polygons[group_name]
                print(f"Removed empty group '{group_name}'")
        
        # Clear polygons from the viewer visually
        viewer.clear_polygons()
        print(f"Cleared polygons from viewer for image: {filepath}")
        
        # Determine the polygons directory path
        polygons_dir = os.path.join(
            self.project_folder, 'polygons'
        ) if self.project_folder else os.path.join(os.getcwd(), 'polygons')
        
        # Delete the corresponding JSON file for each group
        for group_name in groups_with_polygons:
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            polygon_filename = f"{group_name}_{base_filename}_polygons.json"
            polygon_filepath = os.path.join(polygons_dir, polygon_filename)
            
            if os.path.exists(polygon_filepath):
                try:
                    os.remove(polygon_filepath)
                    print(f"Deleted polygon JSON file: {polygon_filepath}")
                except Exception as e:
                    print(f"Failed to delete polygon JSON file {polygon_filepath}: {e}")
            else:
                print(f"Polygon JSON file does not exist: {polygon_filepath}")
        
        # Save the updated polygons data to JSON to persist changes
        # Ensure that multimspec_root_names exists and is not empty
        if hasattr(self, 'multispectral_root_names') and self.multispectral_root_names:
            # Ensure current_root_index is within bounds
            if 0 <= self.current_root_index < len(self.multispectral_root_names):
                current_root = self.multispectral_root_names[self.current_root_index]
                current_filepaths = set(self.image_data_groups[current_root])
                self.save_polygons_to_json(root_name=current_root)
            else:
                print(f"current_root_index {self.current_root_index} is out of range for multispectral_root_names.")
                # Handle the out-of-range index
                if self.multispectral_root_names:
                    # Reset to first root or another valid index
                    self.current_root_index = 0
                    current_root = self.multispectral_root_names[self.current_root_index]
                    current_filepaths = set(self.image_data_groups[current_root])
                    self.save_polygons_to_json(root_name=current_root)
                    print(f"Reset current_root_index to 0 and saved polygons for '{current_root}'.")
                else:
                    print("No multispectral_root_names available to save polygons.")
        else:
            print("multispectral_root_names is empty or not defined. Skipping save_polygons_to_json.")
        
        # Update the Polygon Manager UI to reflect changes
        self.update_polygon_manager()
        print("Updated Polygon Manager UI.")
        print("Saved updated polygons to JSON.")






    def clean_all_polygons(self):
            """
            Clears polygon groups associated with the currently displayed root image by:
            - Clearing polygons from the current viewers.
            - Deleting polygon JSON files for the current root from the 'polygons' directory within the project folder.
            - Removing the polygons from the in-memory 'all_polygons' data structure.
            """
            active_roots = self.root_names or self.multispectral_root_names

            if not active_roots:
                print("No roots available – nothing to clean.")
                return

          

            current_root = active_roots[self.current_root_index]
            current_filepaths = set(self.image_data_groups[current_root])

            # Identify which groups have polygons for the current root
            groups_to_delete = set()
            for group_name, polygons in list(self.all_polygons.items()):
                # Check if any filepath in this group belongs to the current root
                if any(fp in current_filepaths for fp in polygons.keys()):
                    groups_to_delete.add(group_name)

            # Remove polygons from in-memory data structure
            for group_name in groups_to_delete:
                for filepath in list(self.all_polygons[group_name].keys()):
                    if filepath in current_filepaths:
                        del self.all_polygons[group_name][filepath]
                # If no polygons remain in this group, remove the group entirely
                if not self.all_polygons[group_name]:
                    del self.all_polygons[group_name]

            # Clear polygons from viewers that display images of the current root
            for widget in self.viewer_widgets:
                viewer = widget['viewer']
                filepath = widget['image_data'].filepath
                if filepath in current_filepaths:
                    viewer.clear_polygons()

            # Determine polygon directory path
            polygons_dir = os.path.join(self.project_folder, 'polygons') if self.project_folder else os.path.join(os.getcwd(), 'polygons')

            # Delete JSON files associated with the current root
            if os.path.exists(polygons_dir):
                try:
                    for group_name in groups_to_delete:
                        for filepath in current_filepaths:
                            base_filename = os.path.splitext(os.path.basename(filepath))[0]
                            polygon_filename = f"{group_name}_{base_filename}_polygons.json"
                            polygon_filepath = os.path.join(polygons_dir, polygon_filename)
                            if os.path.exists(polygon_filepath):
                                os.remove(polygon_filepath)
                except Exception as e:
                    print(f"Could not delete polygon files: {e}")

            # Save the updated polygons data to JSON to persist the current state
            self.save_polygons_to_json(root_name=current_root)

            # Refresh the polygon manager UI to reflect the changes
            self.update_polygon_manager()

            print(f"All polygons related to root '{current_root}' have been cleared, data structures updated, and files removed.")
            
            
            
    def get_current_root_name(self):
        if self.multispectral_root_names and 0 <= self.current_root_index < len(self.multispectral_root_names):
            return self.multispectral_root_names[self.current_root_index]
        return None
    
    
    
    def save_project(self):
        """
        Saves the current project state to a JSON file within the selected project folder.
        Includes multispectral and thermal/RGB folder paths, root offset, all polygons,
        current root index, root geographical coordinates, root_mapping, and batch options.
        """
        import os, json

        if self.mode == 'dual_folder':
            if not self.current_folder_path or not self.thermal_rgb_folder_path:
                print("Multispectral or Thermal/RGB folder is not set. Please open folders first.")
                QtWidgets.QMessageBox.warning(
                    self, "Folders Not Set",
                    "Please open both Multispectral and Thermal/RGB folders before saving the project."
                )
                return
        elif self.mode == 'rgb_only':
            if not self.current_folder_path:
                print("RGB folder is not set. Please open a folder first.")
                QtWidgets.QMessageBox.warning(
                    self, "Folder Not Set",
                    "Please open an RGB folder before saving the project."
                )
                return

        # Prompt user to select a project folder
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        project_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Project Folder", os.path.expanduser("~"), options=options)
        if not project_folder:
            return  # User canceled the dialog

        self.project_folder = project_folder  # Set the project folder path

        # Update Project Name Based on Folder Name
        project_name = os.path.basename(os.path.normpath(project_folder))
        self.update_project_name(project_name)

        # Compute Root Coordinates Before Saving
        self.compute_root_coordinates()

        # Map Matching Roots and Generate Mapping
        self.map_matching_roots()

        # Prepare project data
        project_data = {
            "project_name": self.project_name,
            "mode": self.mode,
            "root_offset": self.root_offset,
            "all_polygons": {k: v for k, v in self.all_polygons.items()},
            "current_root_index": self.current_root_index,
            "root_coordinates": self.root_coordinates,
            "root_mapping": getattr(self, "root_mapping_dict", {})  # Include root_mapping if present
        }

        # Save folder paths and batch options based on mode.
        if self.mode == 'dual_folder':
            project_data["multispectral_folder_path"] = self.current_folder_path
            project_data["thermal_rgb_folder_path"] = self.thermal_rgb_folder_path
        elif self.mode == 'rgb_only':
            project_data["rgb_folder_path"] = self.current_folder_path
            # Persist batch options and groups so load_project doesn't rescan/lose batches
            project_data["batch_size"] = int(getattr(self, "batch_size", 10) or 10)
            project_data["multispectral_root_names"] = list(getattr(self, "multispectral_root_names", []))
            project_data["multispectral_image_data_groups"] = dict(getattr(self, "multispectral_image_data_groups", {}))

        # Save project.json inside the project folder
        project_json_path = os.path.join(project_folder, 'project.json')
        try:
            with open(project_json_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4)
            print(f"Project JSON saved to {project_json_path}")
        except Exception as e:
            print(f"Failed to save project JSON: {e}")
            return

        # Save all polygons to the 'polygons' subdirectory within the project folder
        self.save_polygons_to_json()

        print(f"Project saved to {self.project_folder}")
        QtWidgets.QMessageBox.information(
            self, "Save Successful", f"Project saved to {self.project_folder}"
        )

    def load_project(self, project_folder=None):
        """
        Loads a project state from a JSON file within the selected project folder.
        On dual-folder projects it DOES NOT prompt for matching; it uses the saved
        root_offset and rebuilds mapping via map_matching_roots().
        If the second folder is missing/invalid/fake, it is skipped and MS roots
        drive the UI (still in dual_folder mode).
        """
        import os, json, logging
        from collections import defaultdict

        # 1) Pick folder
        if project_folder is None:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            project_folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Project Folder", os.path.expanduser("~"), options=options)
            if not project_folder:
                return

        project_json_path = os.path.join(project_folder, 'project.json')
        if not os.path.exists(project_json_path):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Project Folder",
                "The selected folder does not contain a 'project.json' file."
            )
            return

        try:
            with open(project_json_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load the project file:\n{e}")
            return

        required_keys = {"all_polygons", "current_root_index", "root_offset", "root_coordinates"}
        if not required_keys.issubset(project_data.keys()):
            QtWidgets.QMessageBox.critical(self, "Corrupted Project",
                                           "The project data is incomplete or corrupted.")
            return

        # 2) Restore basic state (no prompts)
        self.root_offset = project_data.get("root_offset", 0)
        project_name = os.path.basename(os.path.normpath(project_folder))
        self.update_project_name(project_name)
        self.project_folder = project_folder

        # Determine mode (do not prompt)
        if "mode" in project_data:
            self.mode = project_data["mode"]
        else:
            has_ms  = bool(project_data.get("multispectral_folder_path"))
            has_tr  = bool(project_data.get("thermal_rgb_folder_path"))
            has_rgb = bool(project_data.get("rgb_folder_path"))
            self.mode = 'dual_folder' if (has_ms and has_tr) else ('rgb_only' if has_rgb else 'dual_folder')

        # 3) Load folders based on mode — NO matching dialog anywhere
        if self.mode == 'dual_folder':
            ms_path = project_data.get("multispectral_folder_path", "")
            tr_path = project_data.get("thermal_rgb_folder_path", "")

            if not ms_path or not os.path.exists(ms_path):
                QtWidgets.QMessageBox.warning(self, "Folder Not Found",
                                              f"The multispectral folder path '{ms_path}' does not exist.")
                return

            # Always load multispectral
            self.current_folder_path = ms_path
            self.load_multispectral_images_from_folder(self.current_folder_path)

            # Decide if second folder is usable; if not, skip cleanly
            FAKE_SENTINEL = "_FAKE_SECOND_FOLDER_"
            second_invalid = (
                not tr_path
                or not os.path.exists(tr_path)
                or os.path.basename(os.path.normpath(tr_path)) == FAKE_SENTINEL
            )

            if second_invalid:
                # Keep dual-folder structures valid but empty; drive UI from MS
                self.thermal_rgb_folder_path = os.path.join(self.project_folder, FAKE_SENTINEL)
                self.thermal_rgb_image_data_groups = {}
                self.thermal_rgb_root_names = []
                self._dual_folder_fake_second = True
                self.root_names = list(self.multispectral_root_names)
            else:
                self._dual_folder_fake_second = False
                self.thermal_rgb_folder_path = tr_path
                self.load_thermal_rgb_images_from_folder(self.thermal_rgb_folder_path)
              
                self.root_names = list(self.multispectral_root_names)

        elif self.mode == 'rgb_only':
            rgb_path = project_data.get("rgb_folder_path", "")
            if not rgb_path or not os.path.exists(rgb_path):
                QtWidgets.QMessageBox.warning(self, "Folder Not Found",
                                              f"The RGB folder path '{rgb_path}' does not exist.")
                return

            self.current_folder_path = rgb_path
            saved_batch = project_data.get("batch_size", None)
            self.batch_size = int(saved_batch) if isinstance(saved_batch, (int, float, str)) and str(saved_batch).isdigit() else saved_batch
            if not self.batch_size or self.batch_size <= 0:
                self.batch_size = int(getattr(self, "batch_size", 0) or 0)

            saved_names  = project_data.get("multispectral_root_names") or []
            saved_groups = project_data.get("multispectral_image_data_groups") or {}
            if saved_names and saved_groups:
                self.multispectral_root_names = list(saved_names)
                from collections import defaultdict as _dd
                self.multispectral_image_data_groups = _dd(list)
                for root, files in saved_groups.items():
                    fixed = []
                    for p in files:
                        pp = p if os.path.isabs(p) else os.path.join(self.current_folder_path, p)
                        if not os.path.exists(pp):
                            cand = os.path.join(self.current_folder_path, os.path.basename(pp))
                            if os.path.exists(cand):
                                pp = cand
                        fixed.append(pp)
                    self.multispectral_image_data_groups[root] = fixed
                if not self.batch_size:
                    try:
                        first_len = len(next(iter(self.multispectral_image_data_groups.values())))
                        self.batch_size = first_len if first_len > 0 else 10
                    except StopIteration:
                        self.batch_size = 10
            else:
                exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                all_images = sorted(
                    os.path.join(self.current_folder_path, f)
                    for f in os.listdir(self.current_folder_path)
                    if f.lower().endswith(exts)
                )
                b = self.batch_size if (isinstance(self.batch_size, int) and self.batch_size > 0) else 10
                self.batch_size = b
                self.multispectral_root_names = []
                from collections import defaultdict as _dd
                self.multispectral_image_data_groups = _dd(list)
                for i in range(0, len(all_images), b):
                    root = f"Root{i//b + 1}"
                    self.multispectral_root_names.append(root)
                    self.multispectral_image_data_groups[root] = all_images[i:i+b]

            self.root_names = list(self.multispectral_root_names)
            self.root_id_mapping = {name: i + 1 for i, name in enumerate(self.multispectral_root_names)}

        else:
            QtWidgets.QMessageBox.warning(self, "Unknown Mode",
                                          f"The project data contains an unknown mode: '{self.mode}'.")
            return

        # 4) Build root_id mapping strictly from saved root_offset (no UI)
        self.map_matching_roots()

        # 5) Restore polygons & state
        self.all_polygons       = defaultdict(dict, project_data["all_polygons"])
        self.root_coordinates   = project_data.get("root_coordinates", {})
        self.current_root_index = project_data.get("current_root_index", 0)

        # Clamp index
        if not (0 <= self.current_root_index < len(self.multispectral_root_names)):
            self.current_root_index = 0

        # 6) Slider + label (numeric if helper exists)
        self.group_slider.blockSignals(True)
        self.group_slider.setMaximum(len(self.multispectral_root_names) - 1 if self.multispectral_root_names else 0)
        self.group_slider.setValue(self.current_root_index)
        self.group_slider.blockSignals(False)

        try:
            # numeric label only
            self.slider_label.setText(self._root_number_for_index(self.current_root_index))
        except Exception:
            # fallback to name if helper not present
            if self.multispectral_root_names:
                self.slider_label.setText(f"Root: {self.multispectral_root_names[self.current_root_index]}")
            else:
                self.slider_label.setText("Root: N/A")

        # 7) Load the current group (no extra prompts)
        if self.multispectral_root_names:
            cur_root = self.multispectral_root_names[self.current_root_index]
            self.load_image_group(cur_root)

        # 8) Update polygon manager
        self.update_polygon_manager()

        print(f"Project loaded from {self.project_folder}")


    def save_project_quick(self, *, skip_recompute: bool = False):
        """
        Saves the current project to the existing project folder without prompting the user.
        If no project folder is set, behaves like 'Save Project' by prompting the user to select one.

        If skip_recompute=True, this will NOT prompt to recompute missing coordinates and
        will NOT recompute coordinates (silent quick save).

        FIX: Persist batch_size and grouped lists in rgb_only mode, just like save_project(),
        so load_project() won't fall back to batch_size=10.
        """
        import os, json

        if not self.project_folder:
            self.save_project()
            return

        if not skip_recompute:
            missing_polygons = self.get_polygons_missing_coordinates()
            if missing_polygons:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Missing Coordinates Detected",
                    f"{len(missing_polygons)} polygons do not have geographical coordinates assigned. "
                    "Would you like to recompute coordinates for these polygons before saving?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self.compute_root_coordinates()
                    self.generate_all_thumbnails()
                    QtWidgets.QMessageBox.information(
                        self,
                        "Coordinates Recomputed",
                        "Geographical coordinates have been recomputed for polygons missing them, and thumbnails have been updated."
                    )

        # If no coordinates in memory, try to load existing ones from project.json
        if not getattr(self, "root_coordinates", None):
            project_json_path = os.path.join(self.project_folder, 'project.json')
            if os.path.exists(project_json_path):
                try:
                    with open(project_json_path, 'r', encoding='utf-8') as f:
                        existing_project_data = json.load(f)
                    self.root_coordinates = existing_project_data.get("root_coordinates", {})
                except Exception as e:
                    print(f"Failed to load existing root coordinates: {e}")

        # Prepare project data (mirror save_project)
        project_data = {
            "project_name": self.project_name,
            "mode": self.mode,
            "root_offset": self.root_offset,
            "all_polygons": {k: v for k, v in self.all_polygons.items()},
            "current_root_index": self.current_root_index,
            "root_coordinates": self.root_coordinates,
            "root_mapping": getattr(self, "root_mapping_dict", {})
        }

        if self.mode == 'dual_folder':
            project_data["multispectral_folder_path"] = self.current_folder_path
            project_data["thermal_rgb_folder_path"] = self.thermal_rgb_folder_path
        elif self.mode == 'rgb_only':
            project_data["rgb_folder_path"] = self.current_folder_path
            project_data["batch_size"] = int(getattr(self, "batch_size", 10) or 10)
            project_data["multispectral_root_names"] = list(getattr(self, "multispectral_root_names", []))
            project_data["multispectral_image_data_groups"] = dict(getattr(self, "multispectral_image_data_groups", {}))

        # Save project.json
        project_json_path = os.path.join(self.project_folder, 'project.json')
        try:
            with open(project_json_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4)
            print(f"Project JSON saved to {project_json_path}")
        except Exception as e:
            print(f"Failed to save project JSON: {e}")
            return

        # Save (now-empty or reduced) polygon files
        self.save_polygons_to_json()

        print(f"Project saved to {self.project_folder}")



    def show_polygon_manager(self):
        all_polygon_groups = self.all_polygons
        self.polygon_manager.set_polygons(all_polygon_groups)
        self.polygon_manager.show()

    def update_polygon_manager(self):
        if self.polygon_manager.isVisible():
            self.polygon_manager.set_polygons(self.all_polygons)
            # Also update current root in PolygonManager
            current_root = self.root_names[self.current_root_index] if self.root_names else None
            self.polygon_manager.set_current_root(current_root, self.image_data_groups)

    def select_polygon_from_manager(self, list_item):
        group_name = list_item.data(QtCore.Qt.UserRole)
        if group_name:
            # Iterate through all polygons to find and select them
            for widget in self.viewer_widgets:
                viewer = widget['viewer']
                for item in viewer.get_all_polygons():
                    if item.name == group_name:
                        item.setSelected(True)
                        viewer.centerOn(item)
            # Removed pop-up
        else:
            pass  # Handle cases where group_name is invalid if necessary

    def refresh_viewer(self, root_name=None):
        """
        Refreshes the viewer for a specific root by reloading images.
        If no root_name is provided, refreshes the current root.
        """
        if root_name:
            # Find the index of the specified root_name
            try:
                root_index = self.multispectral_root_names.index(root_name)
            except ValueError:
                QMessageBox.warning(self, "Invalid Root Name", f"Root '{root_name}' does not exist.")
                logging.warning(f"Attempted to refresh viewer for non-existent root '{root_name}'.")
                return
        else:
            # Use the current root index
            root_index = self.current_root_index

        #if not self.project_folder:
            #QMessageBox.warning(self, "No Project Folder", "Project folder is not set.")
           # logging.warning("Attempted to refresh viewers without a project folder set.")
            #return

        if root_index < 0 or root_index >= len(self.multispectral_root_names):
            QMessageBox.warning(self, "Invalid Root Index", "Specified root index is out of bounds.")
            logging.warning("Specified root index is out of bounds.")
            return

        current_root_name = self.multispectral_root_names[root_index]
        logging.info(f"Refreshing viewer for project '{self.project_name}' and root '{current_root_name}'.")

        try:
            self.load_image_group(current_root_name)
            self.save_polygons_to_json(root_name=current_root_name)
            self.update_polygon_manager()  # Ensure Polygon Manager UI is updated
            self._rewire_all_viewers_for_inspection()
    

            logging.info(f"Completed refreshing viewer for root '{current_root_name}'.")
        except Exception as e:
            logging.error(f"Error during refresh_viewer for root '{current_root_name}': {e}")
            # Use a non-critical message box to avoid system sounds
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Refresh Error")
            msg_box.setText(f"An error occurred while refreshing the viewer for root '{current_root_name}':\n{e}")
            msg_box.setIcon(QMessageBox.NoIcon)  # Remove the icon to prevent sound
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()




     
    # **New Method to Switch to a Specific Group**
    def switch_to_group(self, group_name):
        """
        Switches the current view to the root group that contains polygons of the specified group_name,
        using the 'root' index directly from the polygon data.
        """
        print(f"Attempting to switch to group: {group_name}")
       
        if group_name not in self.all_polygons:
            print(f"Group name '{group_name}' does not exist in all_polygons.")
            return

        # Iterate over filepaths associated with the group_name
        for filepath, polygon_data in self.all_polygons[group_name].items():
            # Get the 'root' index from the polygon data
            root_index = int(polygon_data.get('root', 0)) - 1  # Convert to zero-based index
           
            if 0 <= root_index < len(self.root_names):
                root_name = self.root_names[root_index]
                print(f"Found group '{group_name}' in root '{root_name}' at index {root_index}.")
                self.current_root_index = root_index
                self.load_image_group(root_name)
                return

        # If not found, print a message
        print(f"No root group found containing polygons of group '{group_name}'.")


    # **Helper Method to Get Root by Filepath**
    def get_root_by_filepath(self, filepath):
        # Check in multispectral image groups
        for root_name, filepaths in self.multispectral_image_data_groups.items():
            if filepath in filepaths:
                return root_name
        # Check in thermal/RGB image groups
        for root_name, filepaths in self.thermal_rgb_image_data_groups.items():
            if filepath in filepaths:
                return root_name
        return None




    # **Added: Slot to Handle Editing Initiation**
    def start_editing_group(self, group_name):
        """
        Initiates the drawing mode for a specific group across all ImageViewers.
        """
        for widget in self.viewer_widgets:
            viewer = widget['viewer']
            if self.sync_enabled:
                viewer.start_drawing_with_group_name(group_name)
            else:
           
                viewer.start_drawing_with_group_name(group_name)

    def reload_current_root(self):
        """
        Reloads the current root image group to reflect updated polygons.
        """
        if not self.root_names:
            return
        current_root = self.root_names[self.current_root_index]
        self.load_image_group(current_root)
        print(f"Reloaded root image group: {current_root}")
    
    

    from concurrent.futures import ThreadPoolExecutor

    def generate_thumbnail(self, logical_name, image_data):
        """
        Generate and save a low-res JPG thumbnail for a given polygon.
        Works for RGB, thermal, and multispectral stacks (including >4 channels).
        """
        import os, cv2, numpy as np

        # --- robust access to path & pixel data (Lite-safe) ---
        filepath = getattr(image_data, "filepath", "")
        image = getattr(image_data, "image", None)
        if image is None:
            print(f"Failed to load image for thumbnail: {filepath}")
            return

        # If TIFF stack came channels-first (C,H,W), convert to H,W,C
        if image.ndim == 3 and image.shape[0] <= 16 and image.shape[0] < min(image.shape[1], image.shape[2]):
            # likely channels-first
            image = np.transpose(image, (1, 2, 0))

        H, W = image.shape[:2]

        # --- safe flags (Lite-friendly) ---
        is_rgb = bool(getattr(image_data, "is_rgb", False))
        is_thermal = bool(getattr(image_data, "is_thermal", False))
        if not (is_rgb or is_thermal):
            # Minimal inference
            name_upper = os.path.splitext(os.path.basename(filepath))[0].upper()
            if image.ndim == 3 and image.shape[2] == 3 and image.dtype != np.uint16:
                is_rgb = True
            elif image.dtype == np.uint16 and (name_upper.endswith("_IR") or "THERM" in name_upper):
                is_thermal = True

        # --- root/band info if available (kept as-is; harmless if missing) ---
        root_name = self.get_root_by_filepath(filepath) if hasattr(self, "get_root_by_filepath") else None
        root_id = self.root_id_mapping.get(root_name, 0) if hasattr(self, "root_id_mapping") and root_name else 0
        band_index = 0
        if root_name and hasattr(self, "image_data_groups") and root_name in self.image_data_groups:
            try:
                sorted_images = sorted(self.image_data_groups[root_name])
                band_index = sorted_images.index(filepath) + 1
            except Exception:
                band_index = 0

        # --- fetch polygon points stored in IMAGE coordinates ---
        poly_entry = self.all_polygons.get(logical_name, {}).get(filepath, {})
        points = poly_entry.get("points", [])
        if not points:
            print(f"No polygon points found for {logical_name} in {filepath}")
            return

        # Ensure proper shape for OpenCV (N,1,2) int32
        pts = np.asarray(points, dtype=np.float32)
        pts_i32 = pts.astype(np.int32).reshape(-1, 1, 2)

        # --- bounding rect with small zoom-out ---
        x, y, w, h = cv2.boundingRect(pts_i32)
        zoom = 1.4
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = int(round(w * zoom))
        new_h = int(round(h * zoom))

        # clamp size to image bounds
        new_w = max(1, min(new_w, W))
        new_h = max(1, min(new_h, H))

        x_new = int(round(cx - new_w / 2.0))
        y_new = int(round(cy - new_h / 2.0))
        x_new = max(0, min(x_new, W - new_w))
        y_new = max(0, min(y_new, H - new_h))

        cropped = image[y_new:y_new + new_h, x_new:x_new + new_w]
        if cropped.size == 0:
            print(f"Empty crop for {logical_name} in {filepath}")
            return

        # --- helpers (local, surgical) ---
        def _norm_to_u8(a):
            a = a.astype(np.float32, copy=False)
            mn = float(a.min())
            mx = float(a.max())
            if mx > mn:
                return np.clip(((a - mn) * (255.0 / (mx - mn))).round(), 0, 255).astype(np.uint8)
            return np.zeros_like(a, dtype=np.uint8)

        def _to_bgr_preview(arr):
            """
            Return a 3-channel uint8 BGR image suitable for drawing/saving, from:
            - 1-channel
            - 3-channel
            - 4-channel (drops alpha)
            - >4 channels (takes first 3 bands as quicklook)
            Handles uint16 by normalizing per channel.
            """
            if arr.ndim == 2:
                return cv2.cvtColor(_norm_to_u8(arr), cv2.COLOR_GRAY2BGR)

            C = arr.shape[2]
            # reduce to exactly 3 channels for OpenCV drawing
            if C == 3:
                out = arr
            elif C >= 4:
                out = arr[:, :, :3]  # drop alpha or extra bands
            else:  # C == 2 (rare) -> pad third channel with first
                out = np.dstack([arr, arr[:, :, :1]])

            if out.dtype == np.uint16:
                # per-channel normalization
                chans = [ _norm_to_u8(out[:, :, i]) for i in range(3) ]
                out8 = cv2.merge(chans)
            elif out.dtype != np.uint8:
                out8 = _norm_to_u8(out)  # fallback
            else:
                out8 = out

            return np.ascontiguousarray(out8)

        # --- make BGR preview and draw polygon ---
        cropped_bgr = _to_bgr_preview(cropped)

        adjusted = (pts - np.array([x_new, y_new], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        color = (255, 0, 0) if is_rgb else ((0, 255, 255) if is_thermal else (0, 255, 0))
        thickness = 2

        cv2.polylines(cropped_bgr, [adjusted], isClosed=True, color=color, thickness=thickness)

        # --- resize to thumbnail, save to project/thumbnails ---
        thumbnail = cv2.resize(cropped_bgr, (200, 200), interpolation=cv2.INTER_AREA)

        if getattr(self, "project_folder", None):
            thumbnails_dir = os.path.join(self.project_folder, "thumbnails")
        else:
            thumbnails_dir = os.path.join(os.getcwd(), "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)

        image_type = "rgb" if is_rgb else ("thermal" if is_thermal else "multispectral")
        if hasattr(self, "construct_thumbnail_name"):
            fname = self.construct_thumbnail_name(logical_name, filepath, image_type=image_type)
        else:
            stem = os.path.splitext(os.path.basename(filepath))[0]
            fname = f"{logical_name}__{stem}__{image_type}__b{band_index}_r{root_id}.jpg"

        outpath = os.path.join(thumbnails_dir, fname)
        try:
            cv2.imwrite(outpath, thumbnail)
            print(f"Thumbnail saved: {outpath}")
        except Exception as e:
            print(f"Failed to save thumbnail for {logical_name} in {filepath}: {e}")


    def generate_all_thumbnails(self):
        """
        Generates and saves thumbnails for all polygons.
        Utilizes multithreading for efficiency.
        """
        print("Starting thumbnail generation for all polygons...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for logical_name, polygons in self.all_polygons.items():
                for filepath in polygons.keys():
                    futures.append(executor.submit(self.process_thumbnail, logical_name, filepath))
            for future in futures:
                try:
                    future.result()  # wait for tasks
                except Exception as e:
                    print(f"Error in thumbnail generation thread: {e}")
        print("Completed thumbnail generation for all polygons.")


    def process_thumbnail(self, logical_name, filepath):
        """
        Processes a single thumbnail generation task.
        """
        import cv2
        print(f"Generating thumbnail for Logical Name: {logical_name}, Filepath: {filepath}")

        # try to reuse an already loaded viewer image, else load from disk
        viewer = self.get_viewer_by_filepath(filepath) if hasattr(self, "get_viewer_by_filepath") else None
        if viewer and hasattr(viewer, "image_data") and getattr(viewer.image_data, "image", None) is not None:
            image_data = viewer.image_data
        else:
            try:
                image_data = ImageData(filepath, mode=getattr(self, "mode", "dual_folder"))
                print(f"Loaded image data for {filepath} outside of viewers.")
            except Exception as e:
                print(f"Failed to load image data for {filepath}: {e}")
                return

        # generate thumbnail
        self.generate_thumbnail(logical_name, image_data)


    def save_all_thumbnails(self):
        """
        Saves thumbnails for all polygons by iterating through all polygon groups.
        """
        self.generate_all_thumbnails()
        QtWidgets.QMessageBox.information(self, "Thumbnails Saved", "All thumbnails have been successfully saved.")
        print("All thumbnails have been successfully saved.")




    def load_random_forest_model(self):
        """
        Loads the Random Forest model to be used for RGB image predictions.
        """
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        # Check if the model is already loaded
        if hasattr(self, 'random_forest_model'):
            return True  # Model is already loaded

        while True:
            # Prompt the user to select the Random Forest model file
            model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Random Forest Model File", os.path.expanduser("~"), "Pickle Files (*.pkl)")

            if not model_path:
                # User clicked cancel; ask if they want to proceed without the model
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Model Not Selected",
                    "Random Forest model was not selected. Do you want to proceed without the model?\n"
                    "Class percentages will not be calculated for RGB images.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:
                    return False  # Proceed without the model
                else:
                    continue  # Prompt again

            try:
                self.random_forest_model = joblib.load(model_path)
                return True  # Indicate success
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Model Load Error", f"Failed to load Random Forest model:\n{e}")
                # Ask if the user wants to try loading again
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Load Error",
                    "Failed to load the model. Do you want to try again?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes)
                if reply == QtWidgets.QMessageBox.No:
                    return False  # Proceed without the model            

   
    def update_project_name(self, new_name):
                """
                Updates the project name displayed in the QLabel and the window title.
                :param new_name: The new name for the project.
                """
                self.project_name = new_name
                self.project_label.setText(self.project_name)
                self.setWindowTitle(f"{self.project_name} - Multispectral and Thermal Image Analyzer")
               
                # **Optional: Update Polygon Manager's Title if Needed**
                if hasattr(self, 'polygon_manager') and self.polygon_manager.isVisible():
                    self.polygon_manager.update_title()
    def copy_metadata_action(self):
        """
        Copies metadata from selected source images to target images.
        Users need to select source and target images via file dialogs.
        """
        # Select source images
        source_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Source Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not source_files:
            return  # Do nothing if no selection

        # Select target images
        target_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Target Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not target_files:
            return  # Do nothing if no selection

        if len(source_files) != len(target_files):
            print("Source and target selections do not match in number.")
            return  # Do nothing if counts do not match

        # Perform the metadata copy without confirmation pop-up
        copy_metadata_from_multiple_sources_parallel(source_files, target_files, self.exiftool_path, max_workers=8)
        # Removed pop-up to prevent annoying sounds
        print("Metadata has been successfully copied.")


    def save_project_file(self, project_file):
        """
        Implements the logic to save the project to a file.
        """
        try:
            project_data = {
                'project_name': self.project_name,
                'mode': self.mode,
                'root_offset': self.root_offset,
                'multispectral_image_data_groups': self.multispectral_image_data_groups,
                'thermal_rgb_image_data_groups': self.thermal_rgb_image_data_groups,
                'root_mapping_dict': self.root_mapping_dict,
                'all_polygons': self.all_polygons,
                # Add other necessary data
            }
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=4)
            logging.info(f"Project saved to {project_file}.")
            print(f"Project saved to {project_file}.")
        except Exception as e:
            logging.error(f"Failed to save project to {project_file}: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Save Project Error",
                f"Failed to save project to {project_file}:\n{e}"
            )

    def load_project_file(self, project_file):
        """
        Implements the logic to load the project from a file.
        """
        try:
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            self.project_name = project_data.get('project_name', "Untitled Project")
            self.mode = project_data.get('mode', 'dual_folder')
            self.root_offset = project_data.get('root_offset', 0)
            self.multispectral_image_data_groups = defaultdict(list, project_data.get('multispectral_image_data_groups', {}))
            self.thermal_rgb_image_data_groups = defaultdict(list, project_data.get('thermal_rgb_image_data_groups', {}))
            self.root_mapping_dict = project_data.get('root_mapping_dict', {})
            self.all_polygons = defaultdict(dict, project_data.get('all_polygons', {}))
            # Load other necessary data

            # Update UI elements
            self.project_label.setText(self.project_name)
            self.setWindowTitle(f"{self.project_name} - Multispectral and Thermal Image Analyzer")

            # Map matching roots
            self.map_matching_roots()

            # Load the first image group
            if self.multispectral_root_names:
                try:
                    self.load_image_group(self.multispectral_root_names[self.current_root_index])
                except IndexError as e:
                    print(f"Error loading image group: {e}")
                    QtWidgets.QMessageBox.critical(
                        self, "Loading Error",
                        "Failed to load image group from project file."
                    )
            else:
                print("No multispectral roots found in project file.")
                QtWidgets.QMessageBox.warning(
                    self, "No Multispectral Images",
                    "No multispectral root names were found in the project file."
                )

            logging.info(f"Project loaded from {project_file}.")
            print(f"Project loaded from {project_file}.")

        except Exception as e:
            logging.error(f"Failed to load project from {project_file}: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Load Project Error",
                f"Failed to load project from {project_file}:\n{e}"
            )

    def save_all_polygons(self):
        """
        Saves all polygons to JSON files.
        """
        for root_name in self.multispectral_root_names:
            self.save_polygons_to_json(root_name=root_name)
        QtWidgets.QMessageBox.information(self, "Save All", "All polygons have been saved.")
        logging.info("All polygons have been saved.")


    # ------------------- Exiftool Path Management -------------------

    def set_exiftool_path(self, path: str):
        """Store and verify the exiftool path, persist it for future sessions."""
        import os, subprocess, logging
        from PyQt5.QtCore import QSettings
        from PyQt5 import QtWidgets

        exe = os.path.abspath(path)
        if os.path.isdir(exe):
            for name in ("exiftool.exe", "exiftool(-k).exe"):
                cand = os.path.join(exe, name)
                if os.path.isfile(cand):
                    exe = cand
                    break

        try:
            out = subprocess.check_output([exe, "-ver"], stderr=subprocess.STDOUT)
            ver = out.decode(errors="ignore").strip()
            self.exiftool_path = exe
            logging.info(f"[{self.project_name}] exiftool path set to: {exe} (ver {ver})")
            QSettings("YourOrg", "YourApp").setValue("exiftool_path", exe)
        except OSError as e:
            logging.error(f"Selected exiftool can’t run: {e}")
            QSettings("YourOrg", "YourApp").remove("exiftool_path")
            QtWidgets.QMessageBox.critical(
                self, "ExifTool error",
                "The selected exiftool cannot run on this Windows build.\n"
                "Please choose the 64-bit standalone exiftool.exe."
            )

    def show_map(self):
        """
        Creates a Folium map with satellite view using Esri's World Imagery tiles,
        adds markers from root_coordinates, and saves it as '<project_name>_root_coordinates_map.html',
        then opens it in the default web browser.
        """
        if not self.root_coordinates:
            QtWidgets.QMessageBox.warning(self, "No Coordinates", "No coordinates found to plot on the map.")
            return

        # Extract valid coordinates
        valid_coords = {k: v for k, v in self.root_coordinates.items() if v.get('latitude') and v.get('longitude')}

        if not valid_coords:
            QtWidgets.QMessageBox.warning(self, "Invalid Coordinates", "No valid latitude and longitude found in root_coordinates.")
            return

        # Calculate the average latitude and longitude for centering the map
        avg_lat = sum(coord['latitude'] for coord in valid_coords.values()) / len(valid_coords)
        avg_lon = sum(coord['longitude'] for coord in valid_coords.values()) / len(valid_coords)

        # Create a Folium map centered at the average coordinates with no default tiles
        folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles=None)

        # Add Esri Satellite Tile Layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            control=True
        ).add_to(folium_map)

        # Add markers for each root
        for root_name, coord in valid_coords.items():
            folium.Marker(
                location=[coord['latitude'], coord['longitude']],
                popup=root_name,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(folium_map)

        # Add Layer Control to toggle between Esri Satellite and OpenStreetMap
        folium.LayerControl().add_to(folium_map)

        # Sanitize the project name for filename
        project_name_clean = self.sanitize_project_name(self.project_name)

        # Define the map filename with project name
        map_filename = f"{project_name_clean}_root_coordinates_map.html"

        # Define the directory to save the map
        maps_dir = os.path.join(tempfile.gettempdir(), "multispectral_maps")
        os.makedirs(maps_dir, exist_ok=True)

        # Full path to save the map
        map_path = os.path.join(maps_dir, map_filename)

        # Save the map
        folium_map.save(map_path)

        # Append a timestamp to prevent caching
        timestamp = int(time.time())
        map_url = f'file://{map_path}?t={timestamp}'

        try:
            # Open the map in the default web browser after a short delay
            QtCore.QTimer.singleShot(5, lambda: webbrowser.open(map_url))
            logging.info(f"Map saved to {map_path} and opened in the default browser.")
            print(f"Map saved to {map_path} and opened in the default browser.")
        except Exception as e:
            logging.error(f"Failed to generate or open the map: {e}")
            QtWidgets.QMessageBox.critical(self, "Map Error", f"An error occurred while generating the map:\n{e}")

    def sanitize_project_name(self, project_name):
        """
        Sanitizes the project name to make it safe for use in filenames.
        Replaces any character that is not alphanumeric, space, underscore, or hyphen with an underscore.
        """
        return re.sub(r'[^\w\- ]', '_', project_name).replace(' ', '_')

    def closeEvent(self, event):
        """
        Override the closeEvent to clean up temporary map files.
        """
        maps_dir = os.path.join(tempfile.gettempdir(), "multispectral_maps")
        if os.path.exists(maps_dir):
            try:
                shutil.rmtree(maps_dir)
                print("Temporary map files cleaned up.")
            except Exception as e:
                print(f"Error cleaning up temporary map files: {e}")
        event.accept()

   

    def construct_thumbnail_name(self, logical_name, filepath, image_type='multispectral'):
        """
        Constructs the thumbnail filename based on the naming convention:
        - RGB: {logical_name}_{root_id}_RGB.jpg
        - Thermal: {logical_name}_{root_id}_IR.jpg
        - Multispectral: {logical_name}_{root_id}_{band_index}.jpg
        """
        root_name = self.get_root_by_filepath(filepath)
        if root_name:
            root_id = self.root_id_mapping.get(root_name, 0)
        else:
            root_id = 0  # Default value if root_name is None

        # Determine band index relative to its root group
        if root_name and root_name in self.image_data_groups:
            try:
                # Sort the images within the root to ensure consistent indexing
                sorted_images = sorted(self.image_data_groups[root_name])
                band_index = sorted_images.index(filepath) + 1  # Starting from 1
                suffix = f"_{band_index}" if image_type.lower() == 'multispectral' else ""
            except ValueError:
                band_index = 0  # Default value if filepath is not found in its root
                suffix = ""
        else:
            band_index = 0  # Default value if root_name is None or not found
            suffix = ""

        # Determine the suffix based on image_type
        if image_type.lower() == 'rgb':
            suffix = "_RGB"
        elif image_type.lower() == 'thermal':
            suffix = "_IR"
        elif image_type.lower() == 'multispectral' and band_index > 0:
            suffix = f"_{band_index}"
        else:
            suffix = ""

        # Incorporate the base filename to ensure uniqueness
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        thumbnail_filename = f"{logical_name}_{root_id}_{base_filename}{suffix}.jpg"
        logging.debug(f"Constructed Thumbnail Name: {thumbnail_filename}")  # Debugging Statement
        return thumbnail_filename

    
    def _wire_viewer_for_inspection(self, viewer):
        """Make a single viewer honor the current Inspect toggle and (re)connect the signal."""
        checked = getattr(self, "inspectAct", None).isChecked() if hasattr(self, "inspectAct") else False
        viewer.set_inspection_mode(checked)
        # idempotent connection
        try:
            viewer.pixel_clicked.disconnect(self.display_pixel_value)
        except TypeError:
            pass
        if checked:
            viewer.pixel_clicked.connect(self.display_pixel_value)

    def _rewire_all_viewers_for_inspection(self):
        for w in self.viewer_widgets:
            v = w.get("viewer") if isinstance(w, dict) else None
            if v:
                self._wire_viewer_for_inspection(v)

    def toggle_inspection_mode(self, checked):
        self._rewire_all_viewers_for_inspection()

        sb = getattr(self.window(), "statusBar", None)
        if callable(sb):
            sb = self.window().statusBar()
            if checked:
                sb.showMessage("Inspection mode: Click on an image to get pixel values.")
            else:
                sb.clearMessage()


    def display_pixel_value(self, point, payload):
        """
        Displays real pixel values in the status bar.
        Accepts:
          - payload = None (clicked outside image)
          - payload = tuple/list of numbers (values only)
          - payload = dict with keys {'values': sequence, 'names': sequence[str]} from the viewer
        """
        sb = getattr(self.window(), "statusBar", None)
        if not callable(sb):
            return
        sb = self.window().statusBar()

        if payload is None:
            sb.showMessage(f"Pixel at ({int(point.x())}, {int(point.y())}): outside image")
            return

        # Normalize payload into (names, values)
        names = None
        values = None

        if isinstance(payload, dict):
            values = tuple(payload.get("values", ()))
            names  = tuple(payload.get("names", ())) if payload.get("names") else None
        else:
            # assume iterable of numbers
            try:
                values = tuple(payload)
            except TypeError:
                values = ()

        # Try to get channel names from the emitting viewer if not provided
        if names is None:
            sender = getattr(self, "sender", None)
            viewer = sender() if callable(sender) else None
            chn = getattr(viewer, "channel_names", None)
            if isinstance(chn, (list, tuple)) and len(chn) == len(values):
                names = tuple(chn)

        # Fallback name scheme
        if names is None:
            if len(values) == 3:
                names = ("R", "G", "B")
            else:
                names = tuple([f"b{i+1}" for i in range(len(values))])

        # Format clean numeric output
        def fmt(v):
            # print integers as-is; floats with up to 6 sig figs
            if isinstance(v, (int, np.integer)):
                return f"{int(v)}"
            try:
                fv = float(v)
                return f"{fv:.6g}"
            except Exception:
                return str(v)

        parts = [f"{n}={fmt(v)}" for n, v in zip(names, values)]
        message = f"Pixel at ({int(point.x())}, {int(point.y())}): " + ", ".join(parts)
        sb.showMessage(message)
           
    def _ax_path_for(self, image_path: str) -> str:
        parent = self.parent()
        project_folder = getattr(parent, "project_folder", None) if parent else None
        folder = project_folder if (project_folder and project_folder.strip()) else os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(folder, base_name + ".ax")


    def _eval_band_expression_float(self, img, expr):
        """
        Evaluate band expression on img (float32) and return 2D float32.
        Allowed symbols: b1..bN, no builtins.
        """
        if img is None or not expr:
            return None
        if img.ndim == 2:
            mapping = {'b1': img.astype(np.float32, copy=False)}
        elif img.ndim == 3:
            mapping = {f"b{i+1}": img[:, :, i].astype(np.float32, copy=False) for i in range(img.shape[2])}
        else:
            return None

        code = compile(expr, "<expr>", "eval")
        for name in code.co_names:
            if name not in mapping:
                maxb = img.shape[2] if img.ndim == 3 else 1
                raise NameError(f"Use only b1..b{maxb} in band expression")

        res = eval(code, {"__builtins__": {}}, mapping)
        if isinstance(res, np.ndarray):
            res = res.astype(np.float32, copy=False)
        else:
            res = np.full(img.shape[:2], float(res), dtype=np.float32)
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


    def _apply_ax_mods_and_optional_index(self, base_image, image_path):
        """
        Apply crop/resize from .ax (if any) to base_image, keep float32 scale,
        and append an 'index' channel (float32) if band_expression is present.
        Returns (img_float32, channel_names_list)
        """
        img = base_image.copy()
        # Load modifications
        mods = {}
        try:
            axp = self._ax_path_for(image_path)
            if os.path.exists(axp):
                with open(axp, "r", encoding="utf-8") as f:
                    mods = json.load(f)
        except Exception as e:
            logging.debug(f"Failed to read .ax: {e}")

        # Crop
        rect = mods.get("crop_rect")
        if rect:
            x = int(rect.get("x", 0)); y = int(rect.get("y", 0))
            w = int(rect.get("width", 0)); h = int(rect.get("height", 0))
            img = img[max(0,y):max(0,y)+max(0,h), max(0,x):max(0,x)+max(0,w)]

        # Resize
        if "resize" in mods:
            info = mods["resize"]
            h0, w0 = img.shape[:2]
            if "scale" in info:
                scale = int(info["scale"])
                new_w = max(1, int(w0 * scale / 100.0))
                new_h = max(1, int(h0 * scale / 100.0))
            else:
                pct_w = int(info.get("width", 100))
                pct_h = int(info.get("height", 100))
                new_w = max(1, int(w0 * pct_w / 100.0))
                new_h = max(1, int(h0 * pct_h / 100.0))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Real numeric scale
        img = img.astype(np.float32, copy=False)

        # Channel names (R,G,B if >=3 then extras)
        ch_names = []
        if img.ndim == 3:
            C = img.shape[2]
            if C >= 3:
                ch_names = ["R", "G", "B"] + [f"band_{i}" for i in range(4, C+1)]
                # reorder to R,G,B,... (OpenCV typically loads BGR)
                img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]] + ([img[:, :, i] for i in range(3, C)] if C > 3 else []))
            else:
                ch_names = [f"b{i+1}" for i in range(C)]
        else:
            ch_names = ["b1"]

        # Optional index
        expr = (mods.get("band_expression") or "").strip()
        if expr:
            try:
                idx = self._eval_band_expression_float(img, expr)
                if idx is not None:
                    img = np.dstack([img, idx.astype(np.float32, copy=False)])
                    ch_names.append("index")
            except Exception as e:
                logging.debug(f"Index eval failed: {e}")

        return img, ch_names


    def _get_numeric_base_image(self):
        """
        Try to get the original numeric image the viewer is showing.
        Prefers self.image (if set). Falls back to reading from disk path.
        """
        if hasattr(self, "image") and isinstance(self.image, np.ndarray):
            return self.image
        path = getattr(self, "filepath", None) or getattr(self, "image_path", None)
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            return img
        return None
            
      
            

   
    def get_image_data_by_filepath(self, filepath):
        """
        Retrieves the ImageData instance associated with a given filepath.
        """
        for widget in self.viewer_widgets:
            if widget['image_data'].filepath == filepath:
                return widget['image_data']
        return None

  
    def _setup_roots_from_single_folder_with_batch(self, folder_path: str):
        """
        Ask for batch size and build fake roots from a single folder (RGB or stack).
        Leaves dual_folder vs rgb_only UI to user, but will switch to rgb_only
        when the second folder is not configured so the viewer behaves correctly.
        """
        import os, logging
        from collections import defaultdict
        from PyQt5 import QtWidgets

        if not folder_path or not os.path.isdir(folder_path):
            return False  # nothing to do

        # Ask for batch size
        batch_size, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Batch Size",
            "Enter the number of images per root group:",
            value=getattr(self, "batch_size", 10) or 10,
            min=1,
            max=1000,
            step=1
        )
        if not ok:
            return False

        self.batch_size = batch_size

        if hasattr(self, "reset_data_structures"):
            self.reset_data_structures()
        if hasattr(self, "clear_image_grid"):
            self.clear_image_grid()
        if hasattr(self, "polygon_manager") and hasattr(self.polygon_manager, "list_widget"):
            self.polygon_manager.list_widget.clear()

        # Collect images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]
        all_images.sort()

        if not all_images:
            QtWidgets.QMessageBox.warning(self, "No Images Found",
                                          "No images were found in the selected folder.")
            return False

        # Build batches -> fake roots
        from collections import defaultdict
        self.multispectral_root_names = []
        self.multispectral_image_data_groups = defaultdict(list)

        total_images = len(all_images)
        num_roots = (total_images + batch_size - 1) // batch_size

        for i in range(num_roots):
            start = i * batch_size
            end   = min(start + batch_size, total_images)
            root  = f"Root{i+1}"
            self.multispectral_root_names.append(root)
            self.multispectral_image_data_groups[root].extend(all_images[start:end])

        # Mirrors / ids
        self.root_names = self.multispectral_root_names.copy()
        self.root_id_mapping = {name: idx + 1 for idx, name in enumerate(self.multispectral_root_names)}
        self.current_root_index = 0

        if not getattr(self, "thermal_rgb_folder_path", ""):
            self.mode = "rgb_only"

        # Update slider/label if present
        if hasattr(self, "group_slider"):
            self.group_slider.blockSignals(True)
            self.group_slider.setMaximum(max(0, len(self.multispectral_root_names) - 1))
            self.group_slider.setValue(self.current_root_index)
            self.group_slider.blockSignals(False)

        if hasattr(self, "slider_label"):
            if self.multispectral_root_names:
                self.slider_label.setText(f"Root: {self.multispectral_root_names[self.current_root_index]}")
            else:
                self.slider_label.setText("Root: N/A")

        logging.info(f"Grouped {total_images} images into {num_roots} roots from {folder_path}.")
        return True

    def _setup_roots_from_single_folder_with_batch_SAFE(self, folder_path: str, old_rgb_folder: str = ""):
        """
        Safe single-folder (RGB/stack) builder:
          • Prompts for batch size and groups images into "Root1", "Root2", ...
          • Preserves project name and polygons.
          • Remaps polygon keys (filepaths) into the new folder by matching basenames.
          • Keeps current_folder_path set (prevents 'RGB folder is not set').

        Returns True on success, False if user cancels or folder invalid.
        """
        import os, logging, copy
        from collections import defaultdict
        from PyQt5 import QtWidgets

        # 1) Validate folder
        if not folder_path or not os.path.isdir(folder_path):
            logging.warning("SAFE builder: invalid folder_path")
            return False

        # 2) Ask for batch size (reuse any existing)
        try:
            cur_bs = int(getattr(self, "batch_size", 10) or 10)
        except Exception:
            cur_bs = 10
        batch_size, ok = QtWidgets.QInputDialog.getInt(
            self, "Batch Size", "Enter the number of images per root group:",
            value=cur_bs, min=1, max=10000, step=1
        )
        if not ok:
            return False
        self.batch_size = int(batch_size)

        saved_polys = copy.deepcopy(getattr(self, "all_polygons", {}))
        saved_project_folder = getattr(self, "project_folder", None)
        saved_current_root_index = int(getattr(self, "current_root_index", 0) or 0)

        # 4) Clear ONLY image-group structures (do NOT nuke polygons/paths)
        try:
            if hasattr(self, "viewer_widgets"): self.viewer_widgets = []
            if hasattr(self, "image_data_groups"): self.image_data_groups.clear()
            if hasattr(self, "multispectral_image_data_groups"): self.multispectral_image_data_groups.clear()
            if hasattr(self, "thermal_rgb_image_data_groups"): self.thermal_rgb_image_data_groups.clear()
            if hasattr(self, "multispectral_root_names"): self.multispectral_root_names.clear()
            if hasattr(self, "thermal_rgb_root_names"): self.thermal_rgb_root_names.clear()
            if hasattr(self, "root_names"): self.root_names.clear()
        except Exception:
            pass
        if hasattr(self, "clear_image_grid"):
            try: self.clear_image_grid()
            except Exception: pass
        if hasattr(self, "polygon_manager") and hasattr(self.polygon_manager, "list_widget"):
            try: self.polygon_manager.list_widget.clear()
            except Exception: pass

        # 5) Collect images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF')
        all_images = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if any(f.endswith(ext) for ext in image_extensions)
        ]
        if not all_images:
            QtWidgets.QMessageBox.warning(self, "No Images Found", "No images were found in the selected folder.")
            return False

        # 6) Build batches as roots
        from math import ceil
        total_images = len(all_images)
        num_roots = ceil(total_images / self.batch_size)

        from collections import defaultdict as _dd
        self.multispectral_root_names = []
        self.thermal_rgb_root_names = []
        self.multispectral_image_data_groups = _dd(list)
        self.thermal_rgb_image_data_groups = _dd(list)

        for i in range(num_roots):
            start = i * self.batch_size
            end   = min(start + self.batch_size, total_images)
            root  = f"Root{i+1}"
            imgs  = all_images[start:end]
            self.multispectral_root_names.append(root)
            self.thermal_rgb_root_names.append(root)
            self.multispectral_image_data_groups[root].extend(imgs)
            self.thermal_rgb_image_data_groups[root].extend(imgs)

        self.root_names = list(self.multispectral_root_names)
        self.root_id_mapping = {name: idx + 1 for idx, name in enumerate(self.multispectral_root_names)}
        self.current_root_index = 0 if self.multispectral_root_names else -1

        # 7) Update mode/paths
        self.current_folder_path = folder_path
        self.thermal_rgb_folder_path = ""
        self.mode = "rgb_only"

        # 8) UI slider/label
        try:
            if hasattr(self, "group_slider"):
                self.group_slider.blockSignals(True)
                self.group_slider.setMaximum(max(0, len(self.multispectral_root_names) - 1))
                self.group_slider.setValue(max(0, self.current_root_index))
                self.group_slider.blockSignals(False)
            if hasattr(self, "slider_label"):
                self.slider_label.setText(
                    f"Root: {self.multispectral_root_names[self.current_root_index]}" if self.multispectral_root_names
                    else "Root: N/A"
                )
        except Exception:
            pass

        # 9) Remap polygon filepaths by basename into the new folder
        # Structure: all_polygons[group][filepath] = polygon_data
        try:
            new_polys = {}
            new_base = os.path.normpath(folder_path)
            for group, by_fp in (saved_polys or {}).items():
                new_polys[group] = {}
                for old_fp, pdata in (by_fp or {}).items():
                    base = os.path.basename(old_fp)
                    candidate = os.path.join(new_base, base)
                    new_polys[group][candidate] = pdata
            self.all_polygons = new_polys
        except Exception as e:
            logging.warning(f"Polygon remap failed, keeping original polygons: {e}")
            self.all_polygons = saved_polys

        # 10) Restore project folder reference and previous root selection (if still valid)
        self.project_folder = saved_project_folder
        if 0 <= saved_current_root_index < len(self.multispectral_root_names):
            self.current_root_index = saved_current_root_index

        logging.info(f"[SAFE] Grouped {total_images} images into {num_roots} roots from {folder_path}.")
        return True

    # --- Compatibility shim: if other parts call the old name, route them here ---
    def _setup_roots_from_single_folder_with_batch(self, folder_path: str):
        return self._setup_roots_from_single_folder_with_batch_SAFE(folder_path)

    def change_folders_path(self):
        """
        Change project folder path(s) safely, rebuild roots when needed, and refresh the viewer.

        Rules
          - dual_folder:
              * If BOTH sides already configured -> open TWO pickers (each cancellable independently).
              * If ONLY ONE side configured -> open ONE picker for that side (no prompt to add the other).
              * If NONE configured -> open ONE picker for the FIRST (multispectral) side.
            After picking:
              * If BOTH folders exist -> use dual loaders to (re)index, then refresh.
              * If ONLY ONE folder exists -> prompt for batch size, build roots from that one, then refresh.

          - rgb_only:
              * Open ONE picker for the RGB folder, then prompt for batch size and build roots.
              * Preserve project name and polygons; remap polygon filepaths by basename.
        """
        import os, logging
        from PyQt5 import QtWidgets
        from PyQt5.QtWidgets import QMessageBox

        def _existing_dir(p: str) -> bool:
            return bool(p) and isinstance(p, str) and os.path.isdir(p)

        def _pick_dir(title: str, start_dir: str = "") -> str:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            start = start_dir if _existing_dir(start_dir) else os.path.expanduser("~")
            path = QtWidgets.QFileDialog.getExistingDirectory(self, title, start, options=options)
            return path or ""

        current_ms   = getattr(self, "current_folder_path", "") or ""
        current_trgb = getattr(self, "thermal_rgb_folder_path", "") or ""

        mode = getattr(self, "mode", "rgb_only")
        if mode == "dual_folder":
            has_ms   = _existing_dir(current_ms)
            has_trgb = _existing_dir(current_trgb)

            new_ms   = current_ms
            new_trgb = current_trgb
            changed = False

            if has_ms and has_trgb:
                picked_ms = _pick_dir("Select new FIRST folder (multispectral)", start_dir=current_ms)
                if picked_ms and picked_ms != current_ms:
                    new_ms = picked_ms; changed = True
                picked_trgb = _pick_dir("Select new SECOND folder (thermal/RGB)", start_dir=current_trgb)
                if picked_trgb and picked_trgb != current_trgb:
                    new_trgb = picked_trgb; changed = True

            elif has_ms and not has_trgb:
                picked_ms = _pick_dir("Select new FIRST folder (multispectral)", start_dir=current_ms)
                if picked_ms and picked_ms != current_ms:
                    new_ms = picked_ms; changed = True

            elif has_trgb and not has_ms:
                picked_trgb = _pick_dir("Select new SECOND folder (thermal/RGB)", start_dir=current_trgb)
                if picked_trgb and picked_trgb != current_trgb:
                    new_trgb = picked_trgb; changed = True

            else:
                picked_ms = _pick_dir("Select FIRST folder (multispectral)")
                if picked_ms and picked_ms != current_ms:
                    new_ms = picked_ms; changed = True

            # Commit
            self.current_folder_path = new_ms
            self.thermal_rgb_folder_path = new_trgb

            if not changed:
                return

            ms_ok   = _existing_dir(self.current_folder_path)
            trgb_ok = _existing_dir(self.thermal_rgb_folder_path)

            try:
                if ms_ok and trgb_ok:
                    # Re-index both sides (dual-folder)
                    if hasattr(self, "load_multispectral_images_from_folder"):
                        self.load_multispectral_images_from_folder(self.current_folder_path)
                    if hasattr(self, "load_thermal_rgb_images_from_folder"):
                        self.load_thermal_rgb_images_from_folder(self.thermal_rgb_folder_path)
                else:
                    # Single-folder case -> behave like rgb_only builder but preserve polygons
                    one = self.current_folder_path if ms_ok else (self.thermal_rgb_folder_path if trgb_ok else "")
                    if one:
                        ok = self._setup_roots_from_single_folder_with_batch_SAFE(
                            one, old_rgb_folder=(current_ms or current_trgb)
                        )
                        if not ok:
                            return
                        # lock in the one folder + clear second path
                        self.current_folder_path = one
                        self.thermal_rgb_folder_path = ""
                        self.mode = "rgb_only"

                self.refresh_viewer()
            except Exception as e:
                logging.error(f"Folder change reload failed: {e}")
                mb = QMessageBox(self); mb.setWindowTitle("Refresh Error")
                mb.setText(f"An error occurred while reloading after folder change:\n{e}")
                mb.setIcon(QMessageBox.NoIcon); mb.setStandardButtons(QMessageBox.Ok); mb.exec_()

        elif mode == "rgb_only":
            picked = _pick_dir("Select new RGB folder", start_dir=current_ms)
            if not picked or picked == current_ms:
                return
            try:
                ok = self._setup_roots_from_single_folder_with_batch_SAFE(picked, old_rgb_folder=current_ms)
                if not ok:
                    return
                # ensure paths/mode are consistent (avoid “RGB folder is not set” on save)
                self.current_folder_path = picked
                self.thermal_rgb_folder_path = ""
                self.mode = "rgb_only"
                self.refresh_viewer()
            except Exception as e:
                import logging
                logging.error(f"refresh_viewer() failed after changing RGB folder: {e}")
                mb = QMessageBox(self); mb.setWindowTitle("Refresh Error")
                mb.setText(f"An error occurred while refreshing the viewer:\n{e}")
                mb.setIcon(QMessageBox.NoIcon); mb.setStandardButtons(QMessageBox.Ok); mb.exec_()
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown Mode", f"The project is in an unknown mode: '{self.mode}'.")


    def copy_metadata_action(self):
        """
        Copies metadata from selected source images to target images.
        Users need to select source and target images via file dialogs.
        """
        # Select source images
        source_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Source Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not source_files:
            return  # Do nothing if no selection

        # Select target images
        target_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Target Images", os.path.expanduser("~"), "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)")
        if not target_files:
            return  # Do nothing if no selection

        if len(source_files) != len(target_files):
            print("Source and target selections do not match in number.")
            return  # Do nothing if counts do not match

        # Perform the metadata copy without confirmation pop-up
        copy_metadata_from_multiple_sources_parallel(source_files, target_files, self.exiftool_path, max_workers=8)
        # Removed pop-up to prevent annoying sounds
        print("Metadata has been successfully copied.")
     
