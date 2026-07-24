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
        
        self.trainAct = QAction("Train Model(s)", self)
        self.trainAct.setIcon(QIcon.fromTheme("system-run"))
        self.trainAct.setStatusTip("Train scikit-learn model(s) from selected groups")
        self.trainAct.triggered.connect(self.train_models)
        
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
        self.toolbar.addAction(self.trainAct)         

        self.toolbar.addSeparator()
        self.toolbar.addAction(self.closeAct)

        self.info_label = QLabel("Select polygon groups from the list:")
        self.main_layout.addWidget(self.info_label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.main_layout.addWidget(self.list_widget)

        self.populate_polygon_groups()
    
    
    
    def train_models(self):
        """
        Train one or more sklearn classifiers from the currently selected groups.
        Each group name becomes a class label. Samples pixels inside polygons/points.
        Saves .pkl bundles into <project_folder>/Machine_learning_models/.

        Features used (ORDER EXACTLY):
          - 'red_channel', 'green_channel', 'blue_channel'
          - 'band_4','band_5',... (only if the stack has >3 bands; appended in order)
        """
        import os, pickle, logging, numpy as np
        from datetime import datetime
        from PyQt5 import QtWidgets, QtCore

        # ---- 0) sanity: groups ----
        selected_groups = self.get_selected_groups()
        if len(selected_groups) < 2:
            QtWidgets.QMessageBox.warning(self, "Need ≥ 2 Classes",
                                          "Select at least two groups to serve as class labels.")
            return

        # Pre-import sklearn in background while user interacts with dialogs
        # This hides the ~1-3 second sklearn import time
        def _preload_sklearn():
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
            except Exception:
                pass
        
        import threading
        _sklearn_thread = threading.Thread(target=_preload_sklearn, daemon=True)
        _sklearn_thread.start()

        # Stable snapshot (avoids races with UI changes)
        polygons_by_group = self._snapshot_polygons_by_group(selected_groups)

        # Collate (group, filepath) pairs
        entries = []
        for g in selected_groups:
            for fp in polygons_by_group.get(g, {}).keys():
                entries.append((g, fp))
        if not entries:
            QtWidgets.QMessageBox.warning(self, "No Data", "No files in the selected groups.")
            return

        # ---- 0.5) Detect available bands across all files (with caching) ----
        max_bands = 0
        _image_cache = {}  # Cache: filepath -> (img, C) to avoid loading twice
        
        # Show progress during band detection (this can be slow for many large images)
        band_progress = QtWidgets.QProgressDialog("Detecting image bands...", "Cancel", 0, len(entries), self)
        band_progress.setWindowModality(QtCore.Qt.WindowModal)
        band_progress.setMinimumDuration(100)
        
        for idx, (_g, fp) in enumerate(entries):
            if band_progress.wasCanceled():
                band_progress.close()
                return
            band_progress.setValue(idx)
            
            img, _C = self._get_export_image(fp)
            if img is None:
                continue
            C = 1 if img.ndim == 2 else img.shape[2]
            max_bands = max(max_bands, C)
            
            # Cache the loaded image for reuse during training (avoid loading twice)
            _image_cache[fp] = (img, C)
        
        band_progress.close()

        if max_bands < 1:
            QtWidgets.QMessageBox.warning(self, "No Valid Images",
                                          "Could not detect any valid image bands.")
            return

        # Build default band names (RGB first, then band_4, band_5, etc.)
        all_band_names = []
        if max_bands >= 1:
            all_band_names.append("Red (b1)")
        if max_bands >= 2:
            all_band_names.append("Green (b2)")
        if max_bands >= 3:
            all_band_names.append("Blue (b3)")
        for i in range(3, max_bands):
            all_band_names.append(f"Band {i+1} (b{i+1})")

        # ---- 0.6) Band Selection Dialog ----
        class _BandSelector(QtWidgets.QDialog):
            """Dialog to select and reorder bands for ML training, with custom band expressions."""
            
            # Default band expressions (vegetation indices and custom math)
            DEFAULT_EXPRESSIONS = '''{
    "boolean1": "b1 > 150",
    "boolean2": "(b1 > 150) & (b2 > 165)",
    "boolean3": "(b2 / (b1 + b2 + b3)) > 0.41",
    "sum": "b1 + b2 + b3",
    "GCC": "b2 / (b1 + b2 + b3)",
    "EXG": "2*b2 - (b1 + b3)",
    "RCC": "b3 / (b1 + b2 + b3)",
    "BCC": "b1 / (b1 + b2 + b3)",
    "WDX_2": "(2*b1) + b3 - (2*b2)",
    "WDX": "b1 + 2*b1 - b2",
    "WDX_3": "b1 + 2*b3 - 2*b2"
}'''
            
            def __init__(self, band_names, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Select Training Bands")
                self.resize(550, 650)
                self.band_names = list(band_names)
                self.selected_order = []
                self.custom_expressions = {}  # name -> expression

                layout = QtWidgets.QVBoxLayout(self)

                # Instructions
                info = QtWidgets.QLabel(
                    "Select bands to use for training and their order.\n"
                    "Use checkboxes to include/exclude bands.\n"
                    "Use ↑/↓ buttons to change the order of selected bands."
                )
                info.setWordWrap(True)
                layout.addWidget(info)

                # Horizontal layout for list and buttons
                h_layout = QtWidgets.QHBoxLayout()

                # Band list with checkboxes
                self.band_list = QtWidgets.QListWidget()
                self.band_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
                self.band_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

                for i, name in enumerate(self.band_names):
                    item = QtWidgets.QListWidgetItem(name)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    # Default: select first 3 bands (RGB) if available
                    if i < 3:
                        item.setCheckState(QtCore.Qt.Checked)
                    else:
                        item.setCheckState(QtCore.Qt.Unchecked)
                    item.setData(QtCore.Qt.UserRole, i)  # Store original band index
                    item.setData(QtCore.Qt.UserRole + 1, "band")  # Mark as regular band
                    self.band_list.addItem(item)

                h_layout.addWidget(self.band_list, stretch=1)

                # Move buttons
                btn_layout = QtWidgets.QVBoxLayout()
                btn_layout.addStretch()

                self.up_btn = QtWidgets.QPushButton("↑ Move Up")
                self.up_btn.clicked.connect(self._move_up)
                btn_layout.addWidget(self.up_btn)

                self.down_btn = QtWidgets.QPushButton("↓ Move Down")
                self.down_btn.clicked.connect(self._move_down)
                btn_layout.addWidget(self.down_btn)

                btn_layout.addSpacing(20)

                self.select_all_btn = QtWidgets.QPushButton("Select All")
                self.select_all_btn.clicked.connect(self._select_all)
                btn_layout.addWidget(self.select_all_btn)

                self.select_none_btn = QtWidgets.QPushButton("Select None")
                self.select_none_btn.clicked.connect(self._select_none)
                btn_layout.addWidget(self.select_none_btn)

                self.reset_btn = QtWidgets.QPushButton("Reset Order")
                self.reset_btn.clicked.connect(self._reset_order)
                btn_layout.addWidget(self.reset_btn)

                btn_layout.addStretch()
                h_layout.addLayout(btn_layout)

                layout.addLayout(h_layout)

                # Preview of selected bands in order
                self.preview_label = QtWidgets.QLabel("Selected bands (in order): ")
                self.preview_label.setWordWrap(True)
                layout.addWidget(self.preview_label)

                # ---- Custom Band Expressions Section ----
                layout.addSpacing(10)
                expr_group = QtWidgets.QGroupBox("Custom Band Expressions (Vegetation Indices)")
                expr_layout = QtWidgets.QVBoxLayout(expr_group)

                # Checkbox to enable custom expressions
                self.expr_checkbox = QtWidgets.QCheckBox("Enable custom band expressions for training")
                self.expr_checkbox.setChecked(False)
                self.expr_checkbox.stateChanged.connect(self._on_expr_checkbox_changed)
                expr_layout.addWidget(self.expr_checkbox)

                # Help text
                expr_help = QtWidgets.QLabel(
                    "JSON format: {\"name\": \"expression\", ...}\n"
                    "Use b1, b2, b3, ... for band references. Supports +, -, *, /, (), &, |, >, <, comparisons."
                )
                expr_help.setStyleSheet("color: gray; font-size: 10px;")
                expr_help.setWordWrap(True)
                expr_layout.addWidget(expr_help)

                # Text edit for expressions
                self.expr_edit = QtWidgets.QPlainTextEdit()
                self.expr_edit.setPlainText(self.DEFAULT_EXPRESSIONS)
                self.expr_edit.setMinimumHeight(120)
                self.expr_edit.setEnabled(False)  # Disabled until checkbox is checked
                expr_layout.addWidget(self.expr_edit)

                # Add expressions button
                self.add_expr_btn = QtWidgets.QPushButton("Add Expressions to Band List")
                self.add_expr_btn.setEnabled(False)
                self.add_expr_btn.clicked.connect(self._add_expressions_to_list)
                expr_layout.addWidget(self.add_expr_btn)

                layout.addWidget(expr_group)

                # ---- Spatial Context (Window) Section ----
                layout.addSpacing(10)
                window_group = QtWidgets.QGroupBox("Spatial Context (Neighborhood Window)")
                window_layout = QtWidgets.QVBoxLayout(window_group)

                window_help = QtWidgets.QLabel(
                    "Use surrounding pixels as additional features.\n"
                    "• 1×1: Center pixel only (default, fastest)\n"
                    "• 3×3: 9 pixels × bands = more features\n"
                    "• 5×5: 25 pixels × bands = most context (slower training)"
                )
                window_help.setStyleSheet("color: gray; font-size: 10px;")
                window_help.setWordWrap(True)
                window_layout.addWidget(window_help)

                window_row = QtWidgets.QHBoxLayout()
                window_row.addWidget(QtWidgets.QLabel("Window size:"))
                
                self.window_combo = QtWidgets.QComboBox()
                self.window_combo.addItem("1×1 (center pixel only)", 1)
                self.window_combo.addItem("3×3 (9 pixels)", 3)
                self.window_combo.addItem("5×5 (25 pixels)", 5)
                self.window_combo.setCurrentIndex(0)  # Default: 1×1
                self.window_combo.currentIndexChanged.connect(self._on_window_changed)
                window_row.addWidget(self.window_combo)
                window_row.addStretch()
                
                window_layout.addLayout(window_row)

                # Feature count estimate label
                self.window_feature_label = QtWidgets.QLabel("")
                self.window_feature_label.setStyleSheet("font-style: italic;")
                window_layout.addWidget(self.window_feature_label)

                layout.addWidget(window_group)

                # Dialog buttons
                bb = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                )
                bb.accepted.connect(self.accept)
                bb.rejected.connect(self.reject)
                layout.addWidget(bb)

                # Connect item changes to preview update
                self.band_list.model().dataChanged.connect(self._update_preview)
                self.band_list.model().rowsMoved.connect(self._update_preview)
                self._update_preview()

            def _on_expr_checkbox_changed(self, state):
                enabled = (state == QtCore.Qt.Checked)
                self.expr_edit.setEnabled(enabled)
                self.add_expr_btn.setEnabled(enabled)
                if enabled:
                    # Auto-add expressions when checkbox is enabled
                    self._add_expressions_to_list()
                else:
                    # Remove expression items when unchecked
                    self._remove_expression_items()

            def _remove_expression_items(self):
                """Remove all expression items from the list."""
                items_to_remove = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.data(QtCore.Qt.UserRole + 1) == "expression":
                        items_to_remove.append(i)
                
                for i in reversed(items_to_remove):
                    self.band_list.takeItem(i)
                
                self.custom_expressions.clear()
                self._update_preview()

            def _add_expressions_to_list(self):
                """Parse JSON and add expressions to the band list."""
                import json
                
                # Remove existing expression items first
                self._remove_expression_items()
                
                if not self.expr_checkbox.isChecked():
                    return
                
                try:
                    text = self.expr_edit.toPlainText().strip()
                    if not text:
                        return
                    
                    expressions = json.loads(text)
                    if not isinstance(expressions, dict):
                        QtWidgets.QMessageBox.warning(self, "Invalid Format",
                            "Expressions must be a JSON object: {\"name\": \"expression\", ...}")
                        return
                    
                    # Add each expression as a new band
                    for name, expr in expressions.items():
                        if not isinstance(name, str) or not isinstance(expr, str):
                            continue
                        
                        # Create list item for this expression
                        display_name = f"[EXPR] {name}: {expr[:25]}{'...' if len(expr) > 25 else ''}"
                        item = QtWidgets.QListWidgetItem(display_name)
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        item.setCheckState(QtCore.Qt.Checked)  # Auto-select new expressions
                        item.setData(QtCore.Qt.UserRole, name)  # Store expression name
                        item.setData(QtCore.Qt.UserRole + 1, "expression")  # Mark as expression
                        item.setData(QtCore.Qt.UserRole + 2, expr)  # Store the actual expression
                        item.setBackground(QtGui.QColor(230, 255, 230))  # Light green background
                        self.band_list.addItem(item)
                        
                        self.custom_expressions[name] = expr
                    
                    logging.info(f"[train] Added {len(self.custom_expressions)} custom band expressions")
                    
                except json.JSONDecodeError as e:
                    QtWidgets.QMessageBox.warning(self, "Invalid JSON",
                        f"Could not parse expressions:\n{e}")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Error",
                        f"Failed to add expressions:\n{e}")
                
                self._update_preview()

            def _move_up(self):
                row = self.band_list.currentRow()
                if row > 0:
                    item = self.band_list.takeItem(row)
                    self.band_list.insertItem(row - 1, item)
                    self.band_list.setCurrentRow(row - 1)
                    self._update_preview()

            def _move_down(self):
                row = self.band_list.currentRow()
                if row < self.band_list.count() - 1:
                    item = self.band_list.takeItem(row)
                    self.band_list.insertItem(row + 1, item)
                    self.band_list.setCurrentRow(row + 1)
                    self._update_preview()

            def _select_all(self):
                for i in range(self.band_list.count()):
                    self.band_list.item(i).setCheckState(QtCore.Qt.Checked)
                self._update_preview()

            def _select_none(self):
                for i in range(self.band_list.count()):
                    self.band_list.item(i).setCheckState(QtCore.Qt.Unchecked)
                self._update_preview()

            def _reset_order(self):
                # Re-sort: regular bands by original index, then expressions
                band_items = []
                expr_items = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    item_type = item.data(QtCore.Qt.UserRole + 1)
                    if item_type == "expression":
                        expr_items.append((
                            item.data(QtCore.Qt.UserRole),  # name
                            item.text(),
                            item.checkState(),
                            item.data(QtCore.Qt.UserRole + 2)  # expression
                        ))
                    else:
                        band_items.append((
                            item.data(QtCore.Qt.UserRole),  # original index
                            item.text(),
                            item.checkState()
                        ))
                
                # Sort bands by original index
                band_items.sort(key=lambda x: x[0])
                
                self.band_list.clear()
                
                # Add regular bands first
                for orig_idx, name, check_state in band_items:
                    item = QtWidgets.QListWidgetItem(name)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(check_state)
                    item.setData(QtCore.Qt.UserRole, orig_idx)
                    item.setData(QtCore.Qt.UserRole + 1, "band")
                    self.band_list.addItem(item)
                
                # Add expressions after
                for name, display_text, check_state, expr in expr_items:
                    item = QtWidgets.QListWidgetItem(display_text)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(check_state)
                    item.setData(QtCore.Qt.UserRole, name)
                    item.setData(QtCore.Qt.UserRole + 1, "expression")
                    item.setData(QtCore.Qt.UserRole + 2, expr)
                    item.setBackground(QtGui.QColor(230, 255, 230))
                    self.band_list.addItem(item)
                
                self._update_preview()

            def _update_preview(self):
                selected = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        item_type = item.data(QtCore.Qt.UserRole + 1)
                        if item_type == "expression":
                            selected.append(f"[{item.data(QtCore.Qt.UserRole)}]")
                        else:
                            selected.append(item.text())
                if selected:
                    self.preview_label.setText(f"Selected features ({len(selected)}): {', '.join(selected)}")
                else:
                    self.preview_label.setText("Selected features (in order): (none)")
                
                # Also update window feature count
                if hasattr(self, 'window_feature_label'):
                    self._update_window_feature_count()

            def get_selected_band_indices(self):
                """Return list of original band indices for regular bands (in order)."""
                result = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        item_type = item.data(QtCore.Qt.UserRole + 1)
                        if item_type == "band":
                            orig_idx = item.data(QtCore.Qt.UserRole)
                            result.append(orig_idx)
                return result

            def get_selected_band_names(self):
                """Return list of band names for regular bands (in order)."""
                result = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        item_type = item.data(QtCore.Qt.UserRole + 1)
                        if item_type == "band":
                            result.append(item.text())
                return result

            def get_selected_expressions(self):
                """Return list of (name, expression) tuples for custom expressions (in order)."""
                result = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        item_type = item.data(QtCore.Qt.UserRole + 1)
                        if item_type == "expression":
                            name = item.data(QtCore.Qt.UserRole)
                            expr = item.data(QtCore.Qt.UserRole + 2)
                            result.append((name, expr))
                return result

            def get_all_feature_names(self):
                """Return list of all feature names in order (bands + expressions)."""
                result = []
                for i in range(self.band_list.count()):
                    item = self.band_list.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        item_type = item.data(QtCore.Qt.UserRole + 1)
                        if item_type == "expression":
                            result.append(item.data(QtCore.Qt.UserRole))  # expression name
                        else:
                            result.append(item.text())  # band name
                return result

            def get_window_size(self):
                """Return the selected spatial window size (1, 3, or 5)."""
                return self.window_combo.currentData()

            def _on_window_changed(self, index):
                """Update feature count estimate when window size changes."""
                self._update_window_feature_count()

            def _update_window_feature_count(self):
                """Update the estimated feature count label based on selections and window."""
                n_base_features = len(self.get_all_feature_names())
                window_size = self.get_window_size()
                n_pixels = window_size * window_size
                total_features = n_base_features * n_pixels
                
                if window_size == 1:
                    self.window_feature_label.setText(
                        f"Total features per pixel: {total_features}")
                else:
                    self.window_feature_label.setText(
                        f"Total features per pixel: {n_base_features} bands × {n_pixels} pixels = {total_features}")

            def _update_preview_and_window(self):
                """Update both preview and window feature count."""
                self._update_preview_internal()
                self._update_window_feature_count()

        # Show band selector dialog
        band_selector = _BandSelector(all_band_names, self)
        if band_selector.exec_() != QtWidgets.QDialog.Accepted:
            return

        selected_band_indices = band_selector.get_selected_band_indices()
        selected_band_names = band_selector.get_selected_band_names()
        selected_expressions = band_selector.get_selected_expressions()  # [(name, expr), ...]
        all_feature_names = band_selector.get_all_feature_names()  # Combined ordered list (base names)
        window_size = band_selector.get_window_size()  # 1, 3, or 5

        if not selected_band_indices and not selected_expressions:
            QtWidgets.QMessageBox.warning(self, "No Features Selected",
                                          "You must select at least one band or expression for training.")
            return

        logging.info(f"[train] Selected {len(selected_band_indices)} bands + {len(selected_expressions)} expressions")
        logging.info(f"[train] Band indices: {selected_band_indices}")
        logging.info(f"[train] Window size: {window_size}×{window_size}")
        if selected_expressions:
            logging.info(f"[train] Expressions: {[name for name, _ in selected_expressions]}")

        # ---- Helper function for window feature names ----
        def generate_window_feature_names(base_names, win_size):
            """
            Generate feature names for spatial window.
            For win_size=1: returns base_names unchanged
            For win_size=3: base_name_r-1c-1, base_name_r-1c0, ... base_name_r+1c+1
            Order: row-major (top-left to bottom-right), then by band
            """
            if win_size == 1:
                return list(base_names)
            
            half = win_size // 2
            result = []
            # Iterate positions in row-major order
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    suffix = f"_r{dr:+d}c{dc:+d}"  # e.g., _r-1c+1
                    for base_name in base_names:
                        result.append(f"{base_name}{suffix}")
            return result

        # Generate full feature names including window positions
        feature_names_with_window = generate_window_feature_names(all_feature_names, window_size)
        num_features = len(feature_names_with_window)
        
        logging.info(f"[train] Total features: {num_features} ({len(all_feature_names)} base × {window_size*window_size} positions)")

        logging.info(f"[train] Selected bands for training: {selected_band_names} (indices: {selected_band_indices})")

        # ---- 1) sampling: pixels per polygon (per group/file) ----
        samples_ok = False
        while not samples_ok:
            n, ok = QtWidgets.QInputDialog.getInt(
                self, "Sampling", "Pixels per polygon (per file):", 1000, 10, 200000, 100
            )
            if not ok:
                return
            samples_per_poly = int(n)
            samples_ok = samples_per_poly > 0

        # ---- 2) choose models ----
        # detect optional libs
        xgb_avail = False
        lgbm_avail = False
        try:
            from xgboost import XGBClassifier  # type: ignore
            xgb_avail = True
        except Exception:
            XGBClassifier = None  # type: ignore
        try:
            from lightgbm import LGBMClassifier  # type: ignore
            lgbm_avail = True
        except Exception:
            LGBMClassifier = None  # type: ignore

        class _ModelChooser(QtWidgets.QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Select Models to Train")
                self.resize(360, 360)
                v = QtWidgets.QVBoxLayout(self)
                self.list = QtWidgets.QListWidget()
                self.list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
                base_names = [
                    "RandomForest", "ExtraTrees", "GradientBoosting",
                    "LogisticRegression", "SVM", "GaussianNB"
                ]
                if xgb_avail: base_names.append("XGBoost")
                if lgbm_avail: base_names.append("LightGBM")
                for name in base_names:
                    self.list.addItem(QtWidgets.QListWidgetItem(name))
                v.addWidget(self.list)
                bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                v.addWidget(bb)
                bb.accepted.connect(self.accept)
                bb.rejected.connect(self.reject)
            def selected(self):
                return [i.text() for i in self.list.selectedItems()]

        chooser = _ModelChooser(self)
        if chooser.exec_() != QtWidgets.QDialog.Accepted:
            return
        model_names = chooser.selected()
        if not model_names:
            QtWidgets.QMessageBox.information(self, "Cancelled", "No models selected.")
            return

        # n_estimators (default for tree ensembles when no tuning)
        n_estimators, ok = QtWidgets.QInputDialog.getInt(
            self, "Trees", "n_estimators (RF/ExtraTrees/GB, default when no tuning):", 200, 10, 2000, 10
        )
        if not ok:
            return
        n_estimators = int(n_estimators)

        # ---- 3) Ask about hyperparameter optimization ----
        opt_mode, ok = QtWidgets.QInputDialog.getItem(
            self, "Hyperparameter optimization",
            "Optimize model parameters?",
            ["None (fit defaults)", "Faster (RandomizedSearchCV)", "Slower (GridSearchCV)"],
            0, False
        )
        if not ok:
            return
        use_grid = (opt_mode == "Slower (GridSearchCV)")
        use_rand = (opt_mode == "Faster (RandomizedSearchCV)")
        # fixed, sensible defaults (no extra dialogs)
        cv_folds = 5
        rand_iter = 30

        # ---- 4) Build feature names from selected bands ----
        # Map band indices to feature names
        def _band_idx_to_feature_name(idx):
            if idx == 0:
                return "red_channel"
            elif idx == 1:
                return "green_channel"
            elif idx == 2:
                return "blue_channel"
            else:
                return f"band_{idx + 1}"

        # Build base feature names list: first bands, then expressions
        # These are the "base" feature names before window expansion
        base_feature_names = [_band_idx_to_feature_name(i) for i in selected_band_indices]
        base_feature_names.extend([name for name, _ in selected_expressions])
        
        # feature_names_with_window was already computed above when window_size was obtained
        # It includes window position suffixes if window_size > 1
        feature_names = feature_names_with_window
        # num_features was already set above
        
        logging.info(f"[train] Base feature names: {base_feature_names}")
        logging.info(f"[train] Feature names for model ({num_features} total, window={window_size}×{window_size})")
        if window_size > 1:
            logging.info(f"[train] First 10 feature names: {feature_names[:10]}...")

        # ---- 5) Build dataset (X,y) by sampling pixels using SELECTED bands ----
        X_rows, y_rows = [], []
        dropped_nan = 0
        dropped_missing_band = 0

        progress = QtWidgets.QProgressDialog("Sampling training pixels…", "Cancel", 0, len(entries), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        rng = np.random.default_rng(12345)

        for idx, (group_name, filepath) in enumerate(entries, start=1):
            if progress.wasCanceled():
                progress.close()
                return
            progress.setValue(idx)

            # Use cached image if available (from band detection), otherwise load
            if filepath in _image_cache:
                img, _ = _image_cache[filepath]
            else:
                img, _ = self._get_export_image(filepath)
            
            if img is None:
                logging.warning(f"[train] Could not load {filepath}")
                continue
            H, W = img.shape[:2]
            chans = self._channels_in_export_order(img)

            # Load NoData values from .ax file for this image
            nodata_values = []
            mask_polygon_enabled = False
            mask_polygon_names = []
            mask_polygon_points_list = []
            try:
                import json
                base = os.path.splitext(os.path.basename(filepath))[0] + ".ax"
                pf = getattr(self, "project_folder", None)
                ax_candidates = []
                if pf:
                    ax_candidates.append(os.path.join(os.fspath(pf), base))
                    ax_candidates.append(os.path.join(os.fspath(pf), "global.ax"))  # Also check global.ax
                ax_candidates.append(os.path.join(os.path.dirname(filepath), base))
                ax_candidates.append(os.path.join(os.path.dirname(filepath), "global.ax"))
                for axp in ax_candidates:
                    if os.path.exists(axp):
                        with open(axp, "r", encoding="utf-8") as f:
                            ax = json.load(f) or {}
                        # Check if nodata is enabled (default True)
                        if ax.get("nodata_enabled", True):
                            nodata_values = list(ax.get("nodata_values", []) or [])
                            if nodata_values:
                                logging.info(f"[train] Loaded nodata_values={nodata_values} from {axp}")
                        # Load mask_polygon (names-based)
                        mp_cfg = ax.get("mask_polygon", {}) or {}
                        if isinstance(mp_cfg, dict):
                            mask_polygon_enabled = bool(mp_cfg.get("enabled", False))
                            mask_polygon_names = mp_cfg.get("names", []) or []
                            # Legacy: support old single-name format
                            if not mask_polygon_names and mp_cfg.get("name"):
                                mask_polygon_names = [mp_cfg.get("name")]
                            # Legacy: support old points format
                            legacy_points = mp_cfg.get("points", []) or []
                            if legacy_points and len(legacy_points) >= 3:
                                mask_polygon_points_list.append(legacy_points)
                            if mask_polygon_enabled and (mask_polygon_names or mask_polygon_points_list):
                                logging.debug(f"[train] Loaded mask_polygon from {axp}")
                        break
            except Exception as e:
                logging.debug(f"[train] Could not load .ax for {filepath}: {e}")

            # Look up polygon points by name from all_polygons
            if mask_polygon_enabled and mask_polygon_names:
                all_polygons = getattr(self, "all_polygons", None) or getattr(self.parent(), "all_polygons", None) if hasattr(self, "parent") and callable(self.parent) else None
                if all_polygons:
                    fp_norm = os.path.normpath(filepath).lower()
                    for group_name, file_map in all_polygons.items():
                        for stored_fp, poly_data in file_map.items():
                            stored_fp_norm = os.path.normpath(stored_fp).lower() if stored_fp else ""
                            if stored_fp_norm == fp_norm:
                                if isinstance(poly_data, dict):
                                    poly_name = poly_data.get('name', group_name)
                                    if poly_name in mask_polygon_names:
                                        points = poly_data.get('points', [])
                                        if points and len(points) >= 3:
                                            # Scale points if stored at different resolution
                                            ref_size = poly_data.get('image_ref_size', {}) or {}
                                            ref_w = ref_size.get('w', 0) or 0
                                            ref_h = ref_size.get('h', 0) or 0
                                            if ref_w > 0 and ref_h > 0 and (ref_w != W or ref_h != H):
                                                scale_x = W / float(ref_w)
                                                scale_y = H / float(ref_h)
                                                scaled_points = [(x * scale_x, y * scale_y) for (x, y) in points]
                                                mask_polygon_points_list.append(scaled_points)
                                                logging.debug(f"[train] Scaled mask polygon '{poly_name}' from {ref_w}x{ref_h} to {W}x{H}")
                                            else:
                                                mask_polygon_points_list.append(points)

            # Build combined polygon mask for this image
            poly_mask = None
            if mask_polygon_enabled and mask_polygon_points_list:
                poly_mask = np.zeros((H, W), dtype=bool)
                for poly_pts in mask_polygon_points_list:
                    if poly_pts and len(poly_pts) >= 3:
                        pts = np.array([[int(round(x)), int(round(y))] for x, y in poly_pts], dtype=np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        mask_temp = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask_temp, [pts], 255)
                        poly_mask |= (mask_temp > 0)
                if poly_mask.any():
                    logging.debug(f"[train] Built combined polygon mask with {poly_mask.sum()} masked pixels")
                else:
                    poly_mask = None

            # Check if this image has all required bands
            max_required_band = max(selected_band_indices) if selected_band_indices else -1
            if max_required_band >= 0 and len(chans) <= max_required_band:
                logging.warning(f"[train] Image {filepath} has only {len(chans)} bands, "
                               f"but band index {max_required_band} is required. Skipping.")
                continue

            # Extract only the selected bands in the specified order
            selected_chans = []
            for band_idx in selected_band_indices:
                if band_idx < len(chans):
                    selected_chans.append(chans[band_idx].astype(np.float32, copy=False))
                else:
                    selected_chans.append(None)

            # Pre-compute expression values for this image (more efficient than per-pixel)
            expr_images = []  # List of 2D arrays, one per expression
            if selected_expressions:
                try:
                    from .utils import eval_band_expression
                except ImportError:
                    eval_band_expression = None
                
                # Prepare image for expression evaluation (needs HxWxC float32)
                img_for_expr = img.astype(np.float32, copy=False)
                
                for expr_name, expr_str in selected_expressions:
                    try:
                        if eval_band_expression is not None:
                            result = eval_band_expression(img_for_expr, expr_str)
                        else:
                            # Fallback: use parent's eval method if available
                            fn = getattr(self, "_eval_band_expression", None)
                            if fn:
                                result = fn(img_for_expr, expr_str)
                            else:
                                logging.warning(f"[train] No expression evaluator available for '{expr_name}'")
                                result = None
                        
                        # Result should be 2D (HxW) - take last channel if 3D
                        if result is not None:
                            if result.ndim == 3:
                                result = result[:, :, -1]  # Take last channel
                            expr_images.append(result.astype(np.float32, copy=False))
                            logging.debug(f"[train] Computed expression '{expr_name}' for {os.path.basename(filepath)}")
                        else:
                            expr_images.append(None)
                            logging.warning(f"[train] Expression '{expr_name}' returned None for {filepath}")
                    except Exception as e:
                        logging.warning(f"[train] Failed to compute expression '{expr_name}' for {filepath}: {e}")
                        expr_images.append(None)

            # iterate polygons/points inside this file
            file_polys = polygons_by_group.get(group_name, {}).get(filepath, {})
            raw_pts = file_polys.get("points", [])
            if not raw_pts:
                continue

            # Check if this is point data vs polygon data
            extraction_type = (file_polys.get("type") or "polygon").lower()
            is_point_extraction = (extraction_type == "point")

            polys = self._normalize_to_polygons(raw_pts)
            if not polys:
                continue

            # For each polygon: sample pixels and collect selected bands + expressions
            for poly in polys:
                pts_img = self._map_points_scene_to_image(filepath, poly, img.shape, polygon_data=file_polys)
                if not pts_img:
                    continue

                mask = np.zeros((H, W), dtype=np.uint8)
                
                # Handle based on extraction type and number of points
                if is_point_extraction or len(pts_img) == 1:
                    # Point mode: each coordinate is a separate point
                    for (xi, yi) in pts_img:
                        xi, yi = int(round(xi)), int(round(yi))
                        if 0 <= yi < H and 0 <= xi < W:
                            mask[yi, xi] = 255
                elif len(pts_img) == 2:
                    # Two points - treat as two individual points (not a line)
                    for (xi, yi) in pts_img:
                        xi, yi = int(round(xi)), int(round(yi))
                        if 0 <= yi < H and 0 <= xi < W:
                            mask[yi, xi] = 255
                elif len(pts_img) >= 3:
                    # Polygon - fill
                    arr = np.array([pts_img], dtype=np.int32)
                    cv2.fillPoly(mask, arr, 255)

                mask_bool = mask.astype(bool)
                n_pix = int(mask_bool.sum())
                if n_pix == 0:
                    continue

                ys, xs = np.where(mask_bool)
                # Only subsample if this is a true polygon with many pixels
                # For points, use all pixels (no subsampling)
                is_point_data = is_point_extraction or len(pts_img) <= 2
                if not is_point_data and n_pix > samples_per_poly:
                    sel = rng.choice(n_pix, size=samples_per_poly, replace=False)
                    ys = ys[sel]; xs = xs[sel]

                for (yy, xx) in zip(ys, xs):
                    # Skip pixels inside the mask polygon
                    if poly_mask is not None and poly_mask[yy, xx]:
                        dropped_nan += 1  # Count as dropped (masked)
                        continue
                    
                    # For window-based extraction, skip edge pixels where window would go out of bounds
                    half = window_size // 2
                    if yy - half < 0 or yy + half >= H or xx - half < 0 or xx + half >= W:
                        if window_size > 1:
                            dropped_nan += 1  # Can't use edge pixels with windows
                            continue
                    
                    row = []
                    ok_row = True
                    
                    # Window-based feature extraction
                    # Order: iterate positions (row-major), then bands/expressions at each position
                    for dr in range(-half, half + 1):
                        for dc in range(-half, half + 1):
                            py, px = yy + dr, xx + dc
                            
                            # First: collect band values at this window position
                            for ch in selected_chans:
                                if ch is None:
                                    ok_row = False
                                    dropped_missing_band += 1
                                    break
                                val = float(ch[py, px])
                                if not np.isfinite(val):
                                    ok_row = False
                                    dropped_nan += 1
                                    break
                                # Skip NoData values
                                if nodata_values:
                                    is_nodata = False
                                    for nd in nodata_values:
                                        try:
                                            nd_val = float(nd)
                                            abs_nd = abs(nd_val)
                                            # Use appropriate tolerance based on value magnitude
                                            if abs_nd > 1e+30:
                                                tol = abs_nd * 0.01  # 1% for extreme values
                                            elif abs_nd > 1e+10:
                                                tol = abs_nd * 0.001  # 0.1% for very large values
                                            elif abs_nd > 100:
                                                tol = abs_nd * 0.001  # 0.1% for large values
                                            else:
                                                tol = 0.01  # Absolute for small values
                                            if abs(val - nd_val) < tol:
                                                is_nodata = True
                                                break
                                        except Exception:
                                            pass
                                    if is_nodata:
                                        ok_row = False
                                        dropped_nan += 1  # Count as dropped
                                        break
                                row.append(val)
                            
                            if not ok_row:
                                break
                            
                            # Second: collect expression values at this window position
                            if expr_images:
                                for expr_img in expr_images:
                                    if expr_img is None:
                                        ok_row = False
                                        dropped_missing_band += 1
                                        break
                                    val = float(expr_img[py, px])
                                    if not np.isfinite(val):
                                        ok_row = False
                                        dropped_nan += 1
                                        break
                                    row.append(val)
                            
                            if not ok_row:
                                break
                        
                        if not ok_row:
                            break
                    
                    if ok_row and len(row) == num_features:
                        X_rows.append(row)
                        y_rows.append(group_name)

        progress.setValue(len(entries))
        progress.close()
        
        # Free memory from image cache (no longer needed)
        _image_cache.clear()

        if not X_rows:
            QtWidgets.QMessageBox.warning(self, "No Samples",
                                          "No valid training pixels could be collected.")
            return

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=object)

        # ---- 7) Fit models (+ optional tuning) + write reports ----
        saved = []
        last_saved_bundle = None
        last_saved_path = None
        try:
            from sklearn.ensemble import (
                RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
            from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
            from sklearn.base import clone
            import sklearn
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Missing scikit-learn",
                                           f"scikit-learn is required:\n{e}")
            return

        # builder for estimators (returns estimator object and a flag whether it's a pipeline)
        def _build_estimator(name: str):
            if name == "RandomForest":
                return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42), False
            if name == "ExtraTrees":
                return ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42), False
            if name == "GradientBoosting":
                return GradientBoostingClassifier(n_estimators=n_estimators, random_state=42), False
            if name == "LogisticRegression":
                est = Pipeline([("scaler", StandardScaler()),
                                ("clf", LogisticRegression(max_iter=1000, n_jobs=None, solver="lbfgs"))])
                return est, True
            if name == "SVM":
                est = Pipeline([("scaler", StandardScaler()),
                                ("clf", SVC(probability=True))])
                return est, True
            if name == "GaussianNB":
                return GaussianNB(), False
            if name == "XGBoost" and xgb_avail:
                # sensible defaults; label-wise mlogloss
                est = XGBClassifier(
                    objective="multi:softprob", eval_metric="mlogloss",
                    tree_method="hist", n_estimators=n_estimators, random_state=42, verbosity=0
                )
                return est, False
            if name == "LightGBM" and lgbm_avail:
                est = LGBMClassifier(
                    n_estimators=n_estimators, random_state=42
                )
                return est, False
            raise ValueError(name)

        # search spaces (grid + distributions)
        from scipy.stats import loguniform
        def _spaces(name: str, is_pipeline: bool):
            p = "clf__" if is_pipeline else ""
            grid, dist = {}, {}
            if name in ("RandomForest", "ExtraTrees"):
                grid = {
                    f"{p}n_estimators": [100, 200, 400],
                    f"{p}max_depth": [None, 10, 20, 40],
                    f"{p}min_samples_split": [2, 5, 10],
                    f"{p}min_samples_leaf": [1, 2, 4],
                    f"{p}max_features": ["sqrt", "log2", None],
                }
                dist = {
                    f"{p}n_estimators": [100, 150, 200, 300, 400, 600],
                    f"{p}max_depth": [None, 8, 12, 20, 32, 48],
                    f"{p}min_samples_split": [2, 3, 5, 7, 10],
                    f"{p}min_samples_leaf": [1, 2, 3, 4],
                    f"{p}max_features": ["sqrt", "log2", None],
                }
            elif name == "GradientBoosting":
                grid = {
                    f"{p}n_estimators": [100, 200, 400],
                    f"{p}learning_rate": [0.05, 0.1, 0.2],
                    f"{p}max_depth": [2, 3, 5],
                    f"{p}subsample": [0.8, 1.0],
                    f"{p}max_features": ["sqrt", "log2", None],
                }
                dist = {
                    f"{p}n_estimators": [100, 150, 200, 300, 400, 600],
                    f"{p}learning_rate": loguniform(1e-3, 3e-1),
                    f"{p}max_depth": [2, 3, 4, 5, 6],
                    f"{p}subsample": [0.7, 0.8, 0.9, 1.0],
                    f"{p}max_features": ["sqrt", "log2", None],
                }
            elif name == "LogisticRegression":
                grid = {
                    f"{p}C": [0.1, 1.0, 3.0, 10.0],
                    f"{p}penalty": ["l2"],
                    f"{p}solver": ["lbfgs"],
                }
                dist = {
                    f"{p}C": loguniform(1e-2, 1e2),
                }
            elif name == "SVM":
                grid = {
                    f"{p}kernel": ["rbf", "linear"],
                    f"{p}C": [0.5, 1.0, 3.0, 10.0],
                    f"{p}gamma": ["scale", "auto"],
                }
                dist = {
                    f"{p}kernel": ["rbf", "linear"],
                    f"{p}C": loguniform(1e-2, 1e2),
                    f"{p}gamma": ["scale", "auto"],
                }
            elif name == "GaussianNB":
                grid = {f"{p}var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}
                dist = {f"{p}var_smoothing": loguniform(1e-10, 1e-6)}
            elif name == "XGBoost" and xgb_avail:
                grid = {
                    f"{p}n_estimators": [200, 400, 800],
                    f"{p}max_depth": [3, 5, 7],
                    f"{p}learning_rate": [0.05, 0.1, 0.2],
                    f"{p}subsample": [0.7, 0.9, 1.0],
                    f"{p}colsample_bytree": [0.7, 0.9, 1.0],
                }
                dist = {
                    f"{p}n_estimators": [200, 300, 400, 600, 800],
                    f"{p}max_depth": [2, 3, 4, 5, 6, 7, 8],
                    f"{p}learning_rate": loguniform(1e-3, 3e-1),
                    f"{p}subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    f"{p}colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                }
            elif name == "LightGBM" and lgbm_avail:
                grid = {
                    f"{p}n_estimators": [200, 400, 800],
                    f"{p}num_leaves": [31, 63, 127],
                    f"{p}max_depth": [-1, 10, 20, 40],
                    f"{p}learning_rate": [0.05, 0.1, 0.2],
                    f"{p}subsample": [0.7, 0.9, 1.0],
                    f"{p}colsample_bytree": [0.7, 0.9, 1.0],
                }
                dist = {
                    f"{p}n_estimators": [200, 300, 400, 600, 800],
                    f"{p}num_leaves": [31, 63, 127, 255],
                    f"{p}max_depth": [-1, 8, 12, 20, 32, 48],
                    f"{p}learning_rate": loguniform(1e-3, 3e-1),
                    f"{p}subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    f"{p}colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                }
            return grid, dist

        out_dir = os.path.join(self.project_folder or "", "Machine_learning_models")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot Create Folder",
                                           f"Failed to create:\n{out_dir}\n\n{e}")
            return

        # optional: channel friendly names
        def _try_get_channel_names():
            try:
                get_names = getattr(self.parent_tab, "channel_names_for_filepath", None)
                if callable(get_names) and entries:
                    return get_names(entries[0][1])
            except Exception:
                pass
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels_sorted = sorted(list(set(map(str, y))))

        # can we stratify? (>=2 samples per class)
        can_holdout = True
        try:
            _, counts = np.unique(y, return_counts=True)
            if np.any(counts < 2) or len(labels_sorted) < 2:
                can_holdout = False
        except Exception:
            can_holdout = False

        # storage to potentially build a stack later
        tuned_models_for_stack = []   # list of (safe_name, tuned_estimator)
        tuned_meta_info = {}          # name -> dict(meta about tuning for txt/stack)

        # ==== loop over selected models ====
        for name in model_names:
            try:
                est, is_pipe = _build_estimator(name)
                grid, dist = _spaces(name, is_pipe)

                optimization = "None"
                best_params = {}
                best_cv_score = None
                tried = 0

                estimator_for_eval = est

                if use_grid and grid:
                    try:
                        search = GridSearchCV(
                            est, param_grid=grid, cv=cv_folds, n_jobs=-1,
                            scoring="balanced_accuracy", refit=True, verbose=0
                        )
                        search.fit(X, y)
                        estimator_for_eval = search.best_estimator_
                        best_params = dict(search.best_params_)
                        best_cv_score = float(search.best_score_)
                        tried = len(search.cv_results_.get("params", []))
                        optimization = "GridSearchCV"
                    except Exception as e:
                        logging.warning(f"[train] GridSearchCV failed for {name}: {e}")

                elif use_rand and dist:
                    try:
                        search = RandomizedSearchCV(
                            est, param_distributions=dist, n_iter=rand_iter, cv=cv_folds, n_jobs=-1,
                            scoring="balanced_accuracy", random_state=42, refit=True, verbose=0
                        )
                        search.fit(X, y)
                        estimator_for_eval = search.best_estimator_
                        best_params = dict(search.best_params_)
                        best_cv_score = float(search.best_score_)
                        tried = len(search.cv_results_.get("params", []))
                        optimization = "RandomizedSearchCV"
                    except Exception as e:
                        logging.warning(f"[train] RandomizedSearchCV failed for {name}: {e}")

                # If no tuning happened (or failed), just fit defaults for evaluation and saving
                if optimization == "None":
                    estimator_for_eval.fit(X, y)

                # ---- holdout metrics for the report ----
                acc = None; bacc = None; report = "N/A"; cm = None
                if can_holdout:
                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=0.20, random_state=42, stratify=y
                        )
                        clf_eval = clone(estimator_for_eval)
                        clf_eval.fit(X_tr, y_tr)
                        y_pred = clf_eval.predict(X_te)
                        cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
                        acc = float((y_pred == y_te).mean())
                        bacc = float(balanced_accuracy_score(y_te, y_pred))
                        report = classification_report(
                            y_te, y_pred, labels=labels_sorted, zero_division=0
                        )
                    except Exception as e:
                        logging.warning(f"[train] Holdout metrics skipped for {name}: {e}")

                # ---- final model on ALL data (already tuned if optimization ran) ----
                final_model = estimator_for_eval
                final_model.fit(X, y)

                bundle = {
                    "model": final_model,
                    "feature_names": list(feature_names),
                    "label_names": labels_sorted,
                    "band_indices": list(selected_band_indices),  # Store which bands were used (in order)
                    "expressions": [(name, expr) for name, expr in selected_expressions],  # Store custom expressions
                    "window_size": window_size,  # Spatial context window (1, 3, or 5)
                    "base_feature_names": list(base_feature_names),  # Feature names without window suffix
                    "meta": {
                        "created": ts,
                        "notes": f"Trained in CanoPie ML Manager from groups: {', '.join(selected_groups)}",
                        "sklearn_version": getattr(sklearn, "__version__", "unknown"),
                        "n_samples": int(X.shape[0]),
                        "n_features": int(len(feature_names)),
                        "optimization": optimization,
                        "best_cv_score": best_cv_score,
                        "best_params": best_params,
                        "cv_folds": cv_folds if optimization != "None" else None,
                        "tried_candidates": tried if optimization != "None" else None,
                        "band_selection": list(selected_band_indices),  # Also in meta for reports
                        "n_expressions": len(selected_expressions),  # Number of custom expressions
                        "window_size": window_size,  # Also in meta for reports
                    }
                }

                fname = f"{name}_{ts}_classes{len(labels_sorted)}_feat{len(feature_names)}.pkl"
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "wb") as f:
                    pickle.dump(bundle, f)
                saved.append(fpath)
                last_saved_bundle = bundle
                last_saved_path = fpath

                # record for potential stacking
                safe_name = name.lower().replace(" ", "")
                tuned_models_for_stack.append((safe_name, clone(final_model)))  # clone => clean fit inside Stacking
                tuned_meta_info[name] = {
                    "optimization": optimization,
                    "best_cv_score": best_cv_score,
                    "best_params": best_params,
                }

                # ---- sidecar TXT ----
                channel_names = _try_get_channel_names()
                _, cls_counts = np.unique(y, return_counts=True)
                dist_lines = [f"  {lab}: {cnt}" for lab, cnt in zip(labels_sorted, cls_counts)]
                txt_path = os.path.splitext(fpath)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as rf:
                    rf.write(f"Model: {name}\n")
                    rf.write(f"Saved: {fpath}\n")
                    rf.write(f"Created: {ts}\n")
                    rf.write(f"scikit-learn: {getattr(sklearn, '__version__', 'unknown')}\n\n")

                    rf.write(f"Classes (order used below): {labels_sorted}\n")
                    rf.write("Class distribution (all samples):\n")
                    rf.write("\n".join(dist_lines) + "\n\n")

                    rf.write("Selected bands for training (in order):\n")
                    for i, (bi, fnm) in enumerate(zip(selected_band_indices, feature_names[:len(selected_band_indices)]), 1):
                        rf.write(f"  {i:02d}. Band index {bi} -> {fnm}\n")
                    rf.write("\n")

                    # Write expression info if any
                    if selected_expressions:
                        rf.write("Custom band expressions used:\n")
                        for expr_name, expr_str in selected_expressions:
                            rf.write(f"  {expr_name}: {expr_str}\n")
                        rf.write("\n")

                    # Write spatial window info
                    rf.write(f"Spatial window: {window_size}×{window_size}")
                    if window_size > 1:
                        rf.write(f" ({window_size * window_size} pixels per sample)")
                    rf.write("\n\n")

                    rf.write("Feature names in order:\n")
                    for i, fnm in enumerate(feature_names, 1):
                        rf.write(f"  {i:02d}. {fnm}\n")
                    if channel_names:
                        rf.write("\nDetected channel names for example file:\n")
                        for i, nm in enumerate(channel_names, 1):
                            rf.write(f"  ch{i}: {nm}\n")
                    rf.write("\n")

                    # tuning info
                    rf.write(f"Optimization: {optimization}\n")
                    if optimization != "None":
                        rf.write(f"  CV folds: {cv_folds}\n")
                        rf.write(f"  Candidates tried: {tried}\n")
                        if best_cv_score is not None:
                            rf.write(f"  Best CV balanced accuracy: {best_cv_score:.6f}\n")
                        rf.write("  Best params:\n")
                        for k, v in (best_params or {}).items():
                            rf.write(f"    {k}: {v}\n")
                        rf.write("\n")

                    if acc is not None:
                        rf.write("Holdout (20%) evaluation:\n")
                        rf.write(f"  Accuracy: {acc:.4f}\n")
                        rf.write(f"  Balanced Accuracy: {bacc:.4f}\n\n")
                        rf.write("Classification report:\n")
                        rf.write(report + "\n")
                        if cm is not None:
                            rf.write("Confusion matrix (rows=true, cols=pred; label order shown above):\n")
                            for row in cm:
                                rf.write("  " + " ".join(f"{int(x):5d}" for x in row) + "\n")
                    else:
                        rf.write("Holdout evaluation: not computed (insufficient per-class samples).\n")

                    # feature importances when available
                    try:
                        importances = getattr(final_model, "feature_importances_", None)
                        if importances is not None:
                            rf.write("\nFeature importances:\n")
                            for fn, im in sorted(zip(feature_names, importances), key=lambda t: -t[1]):
                                rf.write(f"  {fn:>12s} : {im:.6f}\n")
                    except Exception:
                        pass

            except Exception as e:
                logging.exception(f"Training {name} failed")
                QtWidgets.QMessageBox.critical(self, "Training Failed",
                                               f"{name} failed:\n{e}")
                # keep going for other models

        # ==== Optional: Stacking ====
        stacked_saved = None
        if len(tuned_models_for_stack) >= 2:
            reply = QtWidgets.QMessageBox.question(
                self, "Stack selected models?",
                "You selected multiple models.\n\n"
                "Do you want to build a stacked ensemble (StackingClassifier) that combines them and save it as one model?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    # meta-learner
                    meta = LogisticRegression(max_iter=1000, solver="lbfgs")
                    # StackingClassifier clones incoming estimators => tuned params preserved, refit inside stack
                    stack = StackingClassifier(
                        estimators=tuned_models_for_stack,
                        final_estimator=meta,
                        stack_method="auto",
                        passthrough=False,
                        cv=cv_folds,
                    )

                    # evaluate (optional holdout)
                    acc = None; bacc = None; report = "N/A"; cm = None
                    if can_holdout:
                        try:
                            from sklearn.model_selection import train_test_split
                            X_tr, X_te, y_tr, y_te = train_test_split(
                                X, y, test_size=0.20, random_state=42, stratify=y
                            )
                            stack_eval = clone(stack)
                            stack_eval.fit(X_tr, y_tr)
                            y_pred = stack_eval.predict(X_te)
                            from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
                            cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
                            acc = float((y_pred == y_te).mean())
                            bacc = float(balanced_accuracy_score(y_te, y_pred))
                            report = classification_report(
                                y_te, y_pred, labels=labels_sorted, zero_division=0
                            )
                        except Exception as e:
                            logging.warning(f"[train] Holdout metrics skipped for Stacked: {e}")

                    # fit on ALL data
                    stack.fit(X, y)

                    # save stacked bundle
                    base_list = [{"name": name, **tuned_meta_info.get(name, {})} for name, _ in
                                 [(n.upper() if n.islower() else n, est) for (n, est) in tuned_models_for_stack]]

                    stack_bundle = {
                        "model": stack,
                        "feature_names": list(feature_names),
                        "label_names": labels_sorted,
                        "band_indices": list(selected_band_indices),  # Store which bands were used (in order)
                        "expressions": [(name, expr) for name, expr in selected_expressions],  # Store custom expressions
                        "window_size": window_size,  # Spatial context window (1, 3, or 5)
                        "base_feature_names": list(base_feature_names),  # Feature names without window suffix
                        "meta": {
                            "created": ts,
                            "notes": f"STACKED model from: {', '.join([n for (n, _) in tuned_models_for_stack])}",
                            "sklearn_version": getattr(sklearn, "__version__", "unknown"),
                            "n_samples": int(X.shape[0]),
                            "n_features": int(len(feature_names)),
                            "stacking": True,
                            "cv_folds": cv_folds,
                            "base_models": base_list,
                            "final_estimator": "LogisticRegression(max_iter=1000, solver='lbfgs')",
                            "band_selection": list(selected_band_indices),  # Also in meta for reports
                            "n_expressions": len(selected_expressions),  # Number of custom expressions
                            "window_size": window_size,  # Also in meta for reports
                        }
                    }

                    fname = f"Stacked_{'+'.join([n for (n, _) in tuned_models_for_stack])}_{ts}_classes{len(labels_sorted)}_feat{len(feature_names)}.pkl"
                    # sanitize filename (avoid overly long/special chars)
                    fname = fname.replace('/', '-').replace('\\', '-')
                    fpath = os.path.join(out_dir, fname)
                    with open(fpath, "wb") as f:
                        pickle.dump(stack_bundle, f)
                    stacked_saved = fpath
                    saved.append(fpath)
                    last_saved_bundle = stack_bundle
                    last_saved_path = fpath

                    # sidecar TXT for stacked
                    channel_names = _try_get_channel_names()
                    _, cls_counts = np.unique(y, return_counts=True)
                    dist_lines = [f"  {lab}: {cnt}" for lab, cnt in zip(labels_sorted, cls_counts)]
                    txt_path = os.path.splitext(fpath)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as rf:
                        rf.write("Model: STACKED (StackingClassifier)\n")
                        rf.write(f"Saved: {fpath}\n")
                        rf.write(f"Created: {ts}\n")
                        rf.write(f"scikit-learn: {getattr(sklearn, '__version__', 'unknown')}\n\n")

                        rf.write("Base models (with tuning summary):\n")
                        for (safe_name, _) in tuned_models_for_stack:
                            disp = safe_name  # already safe lower-case name
                            meta_i = tuned_meta_info.get(disp.capitalize(), None) or tuned_meta_info.get(disp, {})
                            rf.write(f"  - {disp}\n")
                            opt = meta_i.get("optimization", "None")
                            rf.write(f"      optimization: {opt}\n")
                            if meta_i.get("best_cv_score") is not None:
                                rf.write(f"      best_cv_bal_acc: {meta_i['best_cv_score']:.6f}\n")
                            bp = meta_i.get("best_params") or {}
                            if bp:
                                rf.write("      best_params:\n")
                                for k, v in bp.items():
                                    rf.write(f"        {k}: {v}\n")
                        rf.write(f"\nFinal estimator: LogisticRegression(max_iter=1000, solver='lbfgs')\n\n")

                        rf.write(f"Classes (order used below): {labels_sorted}\n")
                        rf.write("Class distribution (all samples):\n")
                        rf.write("\n".join(dist_lines) + "\n\n")

                        # Write expression info if any
                        if selected_expressions:
                            rf.write("Custom band expressions used:\n")
                            for expr_name, expr_str in selected_expressions:
                                rf.write(f"  {expr_name}: {expr_str}\n")
                            rf.write("\n")

                        # Write spatial window info
                        rf.write(f"Spatial window: {window_size}×{window_size}")
                        if window_size > 1:
                            rf.write(f" ({window_size * window_size} pixels per sample)")
                        rf.write("\n\n")

                        rf.write("Feature names in order:\n")
                        for i, fnm in enumerate(feature_names, 1):
                            rf.write(f"  {i:02d}. {fnm}\n")
                        if channel_names:
                            rf.write("\nDetected channel names for example file:\n")
                            for i, nm in enumerate(channel_names, 1):
                                rf.write(f"  ch{i}: {nm}\n")
                        rf.write("\n")

                        if acc is not None:
                            rf.write("Holdout (20%) evaluation:\n")
                            rf.write(f"  Accuracy: {acc:.4f}\n")
                            rf.write(f"  Balanced Accuracy: {bacc:.4f}\n\n")
                            rf.write("Classification report:\n")
                            rf.write(report + "\n")
                            if cm is not None:
                                rf.write("Confusion matrix (rows=true, cols=pred; label order shown above):\n")
                                for row in cm:
                                    rf.write("  " + " ".join(f"{int(x):5d}" for x in row) + "\n")
                        else:
                            rf.write("Holdout evaluation: not computed (insufficient per-class samples).\n")

                except Exception as e:
                    logging.exception("Stacked model training failed")
                    QtWidgets.QMessageBox.critical(self, "Stacking Failed", f"Could not build stacked model:\n{e}")

        if not saved:
            QtWidgets.QMessageBox.warning(self, "No Models Saved",
                                          "Training did not produce any saved models.")
            return

        # ---- 7.99) Make the newly trained model immediately available program-wide ----
        # This ensures ImageEditorDialog/ProjectTab/ML exports can append the sklearn label band
        # and produce pixel values consistent with what the user sees after classification.
        try:
            if isinstance(last_saved_bundle, dict) and 'model' in last_saved_bundle:
                # cache in ML manager too (handy for debugging / future UI)
                try:
                    self.last_trained_model_bundle = last_saved_bundle
                    self.last_trained_model_path = last_saved_path
                except Exception:
                    pass
                pt = getattr(self, 'parent_tab', None)
                if pt is not None:
                    try:
                        setattr(pt, 'random_forest_model', last_saved_bundle)
                    except Exception:
                        pass
                    try:
                        setattr(type(pt), 'shared_random_forest_model', last_saved_bundle)
                    except Exception:
                        pass
        except Exception:
            pass

        msg = "Saved model bundle(s):\n\n" + "\n".join(saved)
        if stacked_saved:
            msg += "\n\n(Stacked ensemble included.)"
        if dropped_nan or dropped_missing_band:
            skip_total = dropped_nan + dropped_missing_band
            msg += f"\n\nNote: {skip_total} sampled pixels were skipped"
            details = []
            if dropped_nan:
                details.append(f"{dropped_nan} NaN/Inf")
            if dropped_missing_band:
                details.append(f"{dropped_missing_band} missing bands")
            if details:
                msg += f" ({', '.join(details)})."
        QtWidgets.QMessageBox.information(self, "Training Complete", msg)

    def _rotate_any_channels(self, img, rot):
        """Rotate 2D or HxWxC arrays; supports C > 4 using NumPy."""
        import numpy as np, cv2
        if rot not in (90, 180, 270):
            return img
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 4):
            if rot == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if rot == 180: return cv2.rotate(img, cv2.ROTATE_180)
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # C > 4 → per-channel via rot90 (CCW): 90° CW == 3×CCW
        k = {90: 3, 180: 2, 270: 1}[rot]
        return np.stack([np.rot90(img[:, :, i], k=k) for i in range(img.shape[2])], axis=2)

    def _resize_any_channels(self, img, new_w, new_h, interpolation):
        """Resize 2D or HxWxC arrays; supports C > 4 via per-channel loop."""
        import numpy as np, cv2
        if img.ndim == 2:
            return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        C = img.shape[2]
        if C <= 4:
            return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        chans = [cv2.resize(img[:, :, i], (new_w, new_h), interpolation=interpolation) for i in range(C)]
        return np.stack(chans, axis=2)

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
        import os, re

        cand = []

        # 1) Prefer ProjectTab's own resolver (exactly what the viewer uses)
        pt = getattr(self, "parent_tab", None)
        _ax_path_for = getattr(pt, "_ax_path_for", None)
        if callable(_ax_path_for):
            try:
                p = _ax_path_for(filepath)
                if p:
                    cand.append(p)
            except Exception:
                pass

        # 2) Robust fallbacks (mirror viewer's naming and simple base.ax)
        base_ax_regex = re.sub(
            r"\.(tif|tiff|bmp|png|jpg|jpeg)($|\.|_)",
            r".ax\2",
            os.path.basename(str(filepath)),
            flags=re.I,
        )
        if self.project_folder:
            cand.append(os.path.join(self.project_folder, base_ax_regex))
            cand.append(os.path.join(self.project_folder, os.path.splitext(os.path.basename(filepath))[0] + ".ax"))
        cand.append(os.path.join(os.path.dirname(filepath), base_ax_regex))
        cand.append(os.path.join(os.path.dirname(filepath), os.path.splitext(os.path.basename(filepath))[0] + ".ax"))

        # Keep order; _load_ax_mods will check existence
        return cand

    def _load_ax_mods(self, filepath):
        """Load first .ax found (viewer path first, then project folder, then alongside image)."""
        import os, json, logging
        for mf in self._ax_candidates(filepath):
            try:
                if mf and os.path.exists(mf):
                    with open(mf, "r", encoding="utf-8") as f:
                        return json.load(f) or {}
            except Exception as e:
                logging.error(f"Failed to read AX {mf}: {e}")
        return {}


        def _load_ax_mods(self, filepath):
            """Load first .ax found (viewer path first, then project folder, then alongside image)."""
            import os, json, logging
            for mf in self._ax_candidates(filepath):
                try:
                    if mf and os.path.exists(mf):
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
          - PRESERVES NaNs/Infs so downstream stats/CSV can reflect missing data.
        """
        import numpy as np, logging, re
        if not expr:
            return None

        x = img_float.astype(np.float32, copy=False)  # keep NaNs; no nan_to_num here
        C = 1 if x.ndim == 2 else (x.shape[2] if x.ndim == 3 else 0)
        if C == 0:
            return None

        mapping = {'b1': x} if x.ndim == 2 else {f"b{i+1}": x[:, :, i] for i in range(C)}

        code = compile(expr, "<expr>", "eval")
        for name in code.co_names:
            if name not in mapping:
                logging.warning("Illegal name '%s' in band expr '%s' (export); skipping index.", name, expr)
                return None

        req = sorted({int(b) for b in re.findall(r'b(\d+)', expr)})
        if any(b > C for b in req):
            logging.warning("Expr '%s' requests b%d but only %d band(s) available; skipping index.",
                            expr, max(req), C)
            return None

        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            res = eval(code, {"__builtins__": {}}, mapping)

        if not isinstance(res, np.ndarray):
            try:
                r = float(res)
            except Exception:
                r = float('nan')
            return np.full(x.shape[:2], r, dtype=np.float32)

        if res.ndim == 3:
            res = np.mean(res.astype(np.float32, copy=False), axis=2)
        return res.astype(np.float32, copy=False)  # keep NaNs; no nan_to_num

    def _apply_ax_to_raw(self, raw_img, ax, force_hist=True):
        """
        Order-agnostic, stack-safe replay of scientific steps:

          • Honors optional ax["op_order"] = ["crop","rotate","resize","band_expression"] (any permutation)
          • Inserts (or repositions) a single 'hist' stage *after crop* when ax has 'hist_match'.
          • Supports BOTH percent resize (scale/width/height) and absolute-pixel resize (px_w/px_h)
          • If the crop rect was saved in a different basis than the order user runs now,
            the rect is remapped so pixels/coords stay consistent.
          • For >4ch stacks, resize/rotate are applied per-channel where needed.

        Returns (float32 image in HxWxC, C)
        """
        import numpy as np, cv2, logging

        if raw_img is None:
            return None, 0

        # ---- helpers ----
        def _dims_after_rot(w, h, deg):
            return (h, w) if (deg % 360) in (90, 270) else (w, h)

        def _rect_after_rot(rect, src_w, src_h, deg):
            x = int(rect.get("x", 0)); y = int(rect.get("y", 0))
            w = int(rect.get("width", 0)); h = int(rect.get("height", 0))
            d = int(deg) % 360
            if d == 0:
                return {"x": x, "y": y, "width": w, "height": h}
            if d == 90:   # CW
                nx = src_h - (y + h); ny = x
                return {"x": nx, "y": ny, "width": h, "height": w}
            if d == 180:
                nx = src_w - (x + w); ny = src_h - (y + h)
                return {"x": nx, "y": ny, "width": w, "height": h}
            if d == 270:  # CW
                nx = y; ny = src_w - (x + w)
                return {"x": nx, "y": ny, "width": h, "height": w}
            return {"x": x, "y": y, "width": w, "height": h}

        def _scale_rect(rect, ref_w, ref_h, to_w, to_h):
            sx = float(to_w) / float(max(1, ref_w))
            sy = float(to_h) / float(max(1, ref_h))
            x = int(round(int(rect.get("x", 0)) * sx))
            y = int(round(int(rect.get("y", 0)) * sy))
            w = int(round(int(rect.get("width", 0)) * sx))
            h = int(round(int(rect.get("height", 0)) * sy))
            x = max(0, min(x, max(0, to_w - 1)))
            y = max(0, min(y, max(0, to_h - 1)))
            if x + w > to_w: w = max(0, to_w - x)
            if y + h > to_h: h = max(0, to_h - y)
            return {"x": x, "y": y, "width": w, "height": h}

        def _infer_crop_basis(ax_dict, raw_w, raw_h, rot_deg):
            ref = (ax_dict or {}).get("crop_rect_ref_size") or {}
            try:
                rw = int(ref.get("w") or 0); rh = int(ref.get("h") or 0)
            except Exception:
                rw = rh = 0
            aw, ah = _dims_after_rot(raw_w, raw_h, rot_deg)
            return "after_rotate" if (rw and rh and (rw, rh) == (aw, ah)) else "pre_rotate"

        def _rotate_any_channels(img, rot):
            if rot not in (90, 180, 270):
                return img
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 4):
                if rot == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if rot == 180: return cv2.rotate(img, cv2.ROTATE_180)
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            k = {90: 3, 180: 2, 270: 1}[rot]  # rot90 is CCW: 90° CW == 3×CCW
            return np.stack([np.rot90(img[:, :, i], k=k) for i in range(img.shape[2])], axis=2)

        def _resize_any_channels(img, new_w, new_h, interpolation):
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 4):
                return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            chans = [cv2.resize(img[:, :, i], (new_w, new_h), interpolation=interpolation) for i in range(img.shape[2])]
            return np.stack(chans, axis=2)

        # Prefer the local scientific matcher (float-preserving); fallback to parent if needed
        def _apply_hist_local(img, mods, nodata_vals=None, poly_points_list=None, poly_enabled=False):
            FAST = True
            MAX_SAMPLES = 1000  # same as ProjectTab.HIST_MAX_SAMPLES
            # Prefer local implementation (if present) with fast/sampled args
            try:
                return self._apply_hist_match(img, mods, FAST, MAX_SAMPLES, nodata_values=nodata_vals,
                                              mask_polygon_points=poly_points_list,
                                              mask_polygon_enabled=poly_enabled)
            except TypeError:
                # Try without mask_polygon
                try:
                    return self._apply_hist_match(img, mods, FAST, MAX_SAMPLES, nodata_values=nodata_vals)
                except TypeError:
                    return self._apply_hist_match(img, mods)
                except Exception:
                    pass
            # Fallback to ProjectTab’s implementation
            parent = getattr(self, "parent_tab", None) or getattr(self, "parent", lambda: None)()
            if parent is not None and hasattr(parent, "_apply_hist_match"):
                try:
                    return parent._apply_hist_match(img, mods, FAST, MAX_SAMPLES, nodata_values=nodata_vals)
                except TypeError:
                    return parent._apply_hist_match(img, mods)
                except Exception:
                    pass
            logging.warning("Histogram matching not applied (no working implementation found).")
            return img


        # ---- start from HWC ----
        img = self._ensure_hwc(raw_img)

        # AX params
        try: rot = int(ax.get("rotate", 0)) % 360
        except Exception: rot = 0
        crop_rect = ax.get("crop_rect") or None
        crop_ref  = ax.get("crop_rect_ref_size") or None
        resize    = ax.get("resize") or None
        expr      = (ax.get("band_expression") or "").strip()
        hist_cfg  = (ax.get("hist_match") or None)
        nodata_values = []
        if ax.get("nodata_enabled", True):
            nodata_values = list(ax.get("nodata_values", []) or [])
            if nodata_values:
                logging.info(f"[_apply_ax_modifications] Using nodata_values={nodata_values}")

        # Mask polygon - names-based lookup
        mask_polygon_cfg = ax.get("mask_polygon", {}) or {}
        mask_polygon_enabled = bool(mask_polygon_cfg.get("enabled", False)) if isinstance(mask_polygon_cfg, dict) else False
        mask_polygon_names = mask_polygon_cfg.get("names", []) if isinstance(mask_polygon_cfg, dict) else []
        # Legacy: support old single-name format
        if not mask_polygon_names and mask_polygon_cfg.get("name"):
            mask_polygon_names = [mask_polygon_cfg.get("name")]
        # Legacy: support old points format
        mask_polygon_points_list = []
        legacy_points = mask_polygon_cfg.get("points", []) if isinstance(mask_polygon_cfg, dict) else []
        if legacy_points and len(legacy_points) >= 3:
            mask_polygon_points_list.append(legacy_points)

        # ---- parse enabled flags (default True for backward compatibility) ----
        rotate_enabled = ax.get("rotate_enabled", True)
        crop_enabled = ax.get("crop_enabled", True)
        hist_enabled = ax.get("hist_enabled", True)
        resize_enabled = ax.get("resize_enabled", True)
        band_enabled = ax.get("band_enabled", True)
        
        # --- REGISTRATION ---
        def _do_registration():
            nonlocal img
            reg_cfg = ax.get("registration") or {}
            if isinstance(reg_cfg, dict) and reg_cfg.get("enabled") and reg_cfg.get("matrix") is not None:
                try:
                    import pystackreg
                    mat_list = reg_cfg["matrix"]
                    tmat = np.array(mat_list, dtype=np.float64)
                    
                    # Determine mode from string or infer
                    mode_str = str(reg_cfg.get("mode", "")).lower()
                    
                    was_uint8 = (img.dtype == np.uint8)
                    was_int = np.issubdtype(img.dtype, np.integer)
                    
                    # OPTIMIZATION: Use OpenCV for applying the transform
                    try:
                        import cv2
                        h, w = img.shape[:2]
                        
                        if tmat.shape == (4, 4):
                             tmat_3x3 = np.zeros((3, 3), dtype=np.float64)
                             tmat_3x3[0:2, 0:2] = tmat[0:2, 0:2]
                             tmat_3x3[0:2, 2] = tmat[0:2, 3]
                             tmat_3x3[2, :] = [0, 0, 1]
                             tmat = tmat_3x3
                        elif tmat.shape != (3, 3):
                             raise ValueError(f"Matrix shape {tmat.shape} not supported by OpenCV warp")
                        
                        is_perspective = "bilinear" in mode_str or not np.allclose(tmat[2, :2], 0)
                        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                        
                        # Scale Invariance: Maps normalized StackReg matrix to native runtime bounds
                        # This is ONLY triggered if the rendering pipeline (e.g., Export) is executing 
                        # at a different literal pixel resolution than the Editor Preview where it was calculated.
                        # It does NOT trigger simply because Target Crop Box != Reference Crop Box.
                        calc_shape = reg_cfg.get("calc_shape")
                        if calc_shape is not None and len(calc_shape) == 2:
                             calc_w, calc_h = calc_shape
                             if (h, w) != (calc_h, calc_w) and w > 0 and h > 0:
                                  S_x = calc_w / float(w)
                                  S_y = calc_h / float(h)
                                  S = np.array([[S_x, 0, 0], [0, S_y, 0], [0, 0, 1]], dtype=np.float64)
                                  S_inv = np.array([[1/S_x, 0, 0], [0, 1/S_y, 0], [0, 0, 1]], dtype=np.float64)
                                  tmat = S_inv @ tmat @ S
                        
                        C_out = 1 if img.ndim == 2 else img.shape[2]
                        
                        def _warp_block_viewer(block):
                            if is_perspective:
                                return cv2.warpPerspective(block, tmat, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            else:
                                return cv2.warpAffine(block, tmat[:2, :], (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        
                        if C_out <= 4:
                            img = _warp_block_viewer(img)
                            if C_out == 1 and img.ndim == 2:
                                img = img[..., None]
                        else:
                            warped_bands = []
                            for c in range(C_out):
                                warped_bands.append(_warp_block_viewer(img[..., c]))
                            img = np.stack(warped_bands, axis=2)
                            
                    except Exception as cv_e:
                        logging.warning(f"OpenCV warp failed in ML pipeline, falling back to pystackreg: {cv_e}")
                        import pystackreg
                        sr_mode = pystackreg.StackReg.RIGID_BODY # default
                        if "translation" in mode_str: sr_mode = pystackreg.StackReg.TRANSLATION
                        elif "scaled" in mode_str: sr_mode = pystackreg.StackReg.SCALED_ROTATION
                        elif "affine" in mode_str: sr_mode = pystackreg.StackReg.AFFINE
                        elif "bilinear" in mode_str: sr_mode = pystackreg.StackReg.BILINEAR
                        
                        sr = pystackreg.StackReg(sr_mode)
                        
                        def _apply_reg_2d(plane):
                            f_plane = plane.astype(np.float32)
                            return sr.transform(f_plane, tmat=tmat)
    
                        if img.ndim == 2:
                            img = _apply_reg_2d(img)
                        elif img.ndim == 3:
                             bands = []
                             for c in range(img.shape[2]):
                                 bands.append(_apply_reg_2d(img[..., c]))
                             img = np.stack(bands, axis=2)
    
                    # Restore dtype if needed
                    if was_int:
                        if was_uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        else:
                            info = np.iinfo(raw_img.dtype)
                            img = np.clip(img, info.min, info.max).astype(raw_img.dtype)
                            
                except ImportError:
                    logging.warning("Skipping registration: pystackreg not installed.")
                except Exception as e:
                    logging.warning(f"Registration application failed in ML pipeline: {e}")

        # ---------- Determine crop/rotate order based on crop_rect_ref_size ----------
        # crop_rect_ref_size tells us which reference frame the crop was drawn in:
        # - If matches ORIGINAL dims → user cropped BEFORE rotating → do CROP first
        # - If matches ROTATED dims → user rotated BEFORE cropping → do ROTATE first
        raw_h, raw_w = img.shape[:2]
        rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
        
        do_rotate_first = True  # Default: rotate first
        if crop_rect and rot in (90, 180, 270):
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w = int(crop_ref.get("w", 0)) or 0
                ref_h = int(crop_ref.get("h", 0)) or 0
                if ref_w > 0 and ref_h > 0:
                    # If ref matches original dims → crop was drawn BEFORE rotate
                    if (ref_w, ref_h) == (raw_w, raw_h):
                        do_rotate_first = False
                    # If ref matches rotated dims → crop was drawn AFTER rotate
                    elif (ref_w, ref_h) == (rotated_w, rotated_h):
                        do_rotate_first = True

        # ---- ops ----
        def _do_rotate():
            nonlocal img
            if not rotate_enabled:
                return
            if rot in (90, 180, 270):
                try:
                    img = _rotate_any_channels(img, rot)
                except Exception as e:
                    logging.warning(f"Rotation failed ({rot} deg): {e}")

        def _do_crop():
            nonlocal img
            if not crop_enabled:
                return
            if not crop_rect:
                return
            H, W = img.shape[:2]
            if H <= 0 or W <= 0:
                return

            # Get reference size for scaling
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w = int(crop_ref.get("w") or W)
                ref_h = int(crop_ref.get("h") or H)
            else:
                ref_w, ref_h = W, H

            x = int(crop_rect.get("x", 0))
            y = int(crop_rect.get("y", 0))
            w = int(crop_rect.get("width", W))
            h = int(crop_rect.get("height", H))

            # Scale if reference size differs from current size
            if ref_w != W or ref_h != H:
                sx = W / float(max(1, ref_w))
                sy = H / float(max(1, ref_h))
                x = int(round(x * sx))
                y = int(round(y * sy))
                w = int(round(w * sx))
                h = int(round(h * sy))

            x0 = max(0, min(x, W))
            y0 = max(0, min(y, H))
            x1 = max(0, min(x + w, W))
            y1 = max(0, min(y + h, H))
            
            if x1 > x0 and y1 > y0:
                img = img[y0:y1, x0:x1]
            else:
                logging.warning("Crop rect empty/out of bounds; skipping crop.")

        # Uses shared utility supporting expressions
        def _build_nodata_mask(arr, nd_vals):
            """Build boolean mask where True = NoData pixel. Supports both numeric literals and threshold expressions."""
            from .utils import build_nodata_mask as _shared_build_nodata_mask
            return _shared_build_nodata_mask(arr, nd_vals, bgr_input=True)

        def _do_hist():
            """Apply histogram matching now (explicit stage)."""
            nonlocal img
            if not hist_enabled:
                return
            if not hist_cfg:
                return
            try:
                img = _apply_hist_local(img, {"hist_match": hist_cfg}, nodata_values,
                                       mask_polygon_points_list, mask_polygon_enabled)
            except Exception as e:
                logging.warning(f"Histogram matching failed: {e}")

        def _do_resize():
            nonlocal img
            if not resize_enabled:
                return
            if not isinstance(resize, dict) or not resize:
                return
            h0, w0 = img.shape[:2]
            if h0 <= 0 or w0 <= 0:
                return

            # Absolute pixels
            if ("px_w" in resize) or ("px_h" in resize):
                tw = int(resize.get("px_w", 0) or 0)
                th = int(resize.get("px_h", 0) or 0)
                if tw > 0 and th > 0:
                    new_w, new_h = tw, th
                elif tw > 0:
                    s = tw / float(w0); new_w = tw; new_h = max(1, int(round(h0 * s)))
                elif th > 0:
                    s = th / float(h0); new_h = th; new_w = max(1, int(round(w0 * s)))
                else:
                    new_w, new_h = w0, h0
            # Percent scale
            elif "scale" in resize:
                s = float(resize.get("scale", 100.0)) / 100.0
                new_w = max(1, int(round(w0 * s)))
                new_h = max(1, int(round(h0 * s)))
            # Percent width/height (legacy)
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

            try:
                # Handle NoData during resize using NaN propagation
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
                    nd_mask = _build_nodata_mask(img, nodata_values)
                    if nd_mask is not None and nd_mask.any():
                        # Convert to float32 and replace NoData with NaN
                        work = img.astype(np.float32, copy=True)
                        if work.ndim == 2:
                            work[nd_mask] = np.nan
                        else:
                            for c in range(work.shape[2]):
                                work[..., c][nd_mask] = np.nan
                        
                        # Resize - NaN propagates through interpolation
                        work = _resize_any_channels(work, new_w, new_h, interp)
                        
                        # Replace NaN with NoData value
                        nan_mask = np.isnan(work)
                        if nan_mask.any():
                            work[nan_mask] = nd_restore_val
                        
                        img = work
                    else:
                        img = _resize_any_channels(img, new_w, new_h, interp)
                elif nodata_values:
                    # Expression-only NoData - still apply mask but use 0.0 for restore
                    nd_mask = _build_nodata_mask(img, nodata_values)
                    if nd_mask is not None and nd_mask.any():
                        work = img.astype(np.float32, copy=True)
                        if work.ndim == 2:
                            work[nd_mask] = np.nan
                        else:
                            for c in range(work.shape[2]):
                                work[..., c][nd_mask] = np.nan
                        work = _resize_any_channels(work, new_w, new_h, interp)
                        nan_mask = np.isnan(work)
                        if nan_mask.any():
                            work[nan_mask] = 0.0  # Use 0 as fallback NoData value
                        img = work
                    else:
                        img = _resize_any_channels(img, new_w, new_h, interp)
                else:
                    img = _resize_any_channels(img, new_w, new_h, interp)
            except Exception as e:
                logging.warning(f"Resize failed to {new_w}x{new_h}: {e}")

        def _do_band_expr():
            nonlocal img
            if not band_enabled:
                return
            if not expr:
                return
            x = img.astype(np.float32, copy=False)
            
            # FIX: Remap band references for 3-channel BGR images
            # User sees image as RGB (display converts BGR→RGB), so when they type b1
            # they expect it to refer to Red (what they see as channel 1).
            # But data is BGR, so b1 would be Blue without remapping.
            # Swap b1↔b3 so expression matches visual expectation.
            effective_expr = expr
            C = x.shape[2] if x.ndim == 3 else 1
            if C == 3:
                import re
                # BGR→RGB remapping: b1↔b3 (b2 stays the same)
                effective_expr = re.sub(r'\bb1\b', '__B1_TEMP__', effective_expr)
                effective_expr = re.sub(r'\bb3\b', 'b1', effective_expr)
                effective_expr = re.sub(r'__B1_TEMP__', 'b3', effective_expr)
                if effective_expr != expr:
                    logging.info(f"[_do_band_expr] Remapped BGR→RGB: '{expr}' → '{effective_expr}'")
            
            idx = self._eval_band_expression(x, effective_expr)
            img = np.dstack([x, idx.astype(np.float32, copy=False)]) if idx is not None else x

        # Execute crop/rotate in the correct order based on crop_rect_ref_size
        if do_rotate_first:
            _do_rotate()
            _do_crop()
        else:
            _do_crop()
            _do_rotate()
        
        # Check if resize is a shrink - if so, defer histogram to AFTER resize
        # This matches the editor's HIST_MATCH_AFTER_RESIZE_IF_SHRINK = True behavior
        defer_hist = False
        if hist_enabled and hist_cfg and resize_enabled and isinstance(resize, dict) and resize:
            h0, w0 = img.shape[:2]
            if h0 > 0 and w0 > 0:
                if ("px_w" in resize) or ("px_h" in resize):
                    tw = int(resize.get("px_w", 0) or 0)
                    th = int(resize.get("px_h", 0) or 0)
                    if tw > 0 and th > 0:
                        new_w, new_h = tw, th
                    elif tw > 0:
                        s = tw / float(w0)
                        new_w, new_h = tw, max(1, int(round(h0 * s)))
                    elif th > 0:
                        s = th / float(h0)
                        new_h, new_w = th, max(1, int(round(w0 * s)))
                    else:
                        new_w, new_h = w0, h0
                elif "scale" in resize:
                    s = float(resize.get("scale", 100.0)) / 100.0
                    new_w, new_h = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
                else:
                    pw = float(resize.get("width", 100.0)) / 100.0
                    ph = float(resize.get("height", 100.0)) / 100.0
                    new_w, new_h = max(1, int(round(w0 * pw))), max(1, int(round(h0 * ph)))
                # If shrinking, defer histogram to after resize
                if new_w < w0 or new_h < h0:
                    defer_hist = True
        
        # Apply histogram BEFORE resize (unless deferred for shrink)
        if not defer_hist:
            _do_hist()
        
        _do_resize()
        
        # Execute Registration AFTER Crop and Resize so multi-sensor normalized dimensions align perfectly.
        # The stored matrix was computed natively on the post-resized coordinate space.
        _do_registration()
        
        # Apply deferred histogram AFTER resize (for shrink case)
        if defer_hist:
            _do_hist()
        
        _do_band_expr()

        # --- classification band unchanged (safe no-op if not configured) ---
        try:
            cblock = (ax.get("classification") or {})
            enabled = (
                isinstance(cblock, dict)
                and str(cblock.get("mode", "")).lower() == "sklearn"
                and bool(cblock.get("enabled", False))
            )
            if enabled and img is not None:
                bundle = None
                pt = getattr(self, "parent_tab", None)
                if pt is not None:
                    getb = getattr(pt, "_get_sklearn_bundle", None)
                    if callable(getb):
                        try: bundle = getb()
                        except Exception: bundle = None
                    if not (isinstance(bundle, dict) and "model" in bundle):
                        for attr in ("shared_model_bundle","model_bundle","random_forest_model","shared_random_forest_model"):
                            b = getattr(pt, attr, None)
                            if isinstance(b, dict) and "model" in b:
                                bundle = b; break
                if isinstance(bundle, dict) and "model" in bundle:
                    chans = self._channels_in_export_order(img)
                    cls_idx = self._predict_class_map_from_bundle(chans, bundle)
                    if cls_idx is not None and cls_idx.shape[:2] == img.shape[:2]:
                        if img.ndim == 2: img = img[..., None]
                        cls_band = cls_idx.astype(np.float32, copy=False)[..., None]
                        img = np.concatenate([img, cls_band], axis=2)
        except Exception:
            pass

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

        arr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        return self._ensure_hwc(arr)

    def _apply_hist_match(self, img, mods):
        """
        Apply histogram normalization described by mods['hist_match'] to img.
        Supports:
          - mode='meanstd' with 'ref_stats' [{'mean':..,'std':..}, ...]
          - mode='cdf' with 'ref_cdf': {'per_band':[{'x':..,'y':..,'lo':..,'hi':..}, ...]}
        Returns float32 image; preserves NaNs.
        """
        import numpy as np

        hcfg = (mods or {}).get("hist_match")
        if not hcfg:
            return img

        x = img.astype(np.float32, copy=False)
        if x.ndim == 2:
            x = x[..., None]
        C = x.shape[2]

        mode = str(hcfg.get("mode", "meanstd")).lower()

        def _safe_std(a):
            s = float(np.nanstd(a))
            return s if s > 1e-12 else 1.0

        if mode == "meanstd":
            stats = list(hcfg.get("ref_stats") or [])
            # Apply per channel; ignore extra channels beyond provided stats
            for c in range(min(C, len(stats))):
                ch   = x[..., c]
                mu_r = float(stats[c].get("mean", 0.0))
                sd_r = float(stats[c].get("std",  1.0))
                mu_t = float(np.nanmean(ch))
                sd_t = _safe_std(ch)
                x[..., c] = (ch - mu_t) * (sd_r / sd_t) + mu_r
        else:
            # CDF-based matching (kept for completeness)
            ref = hcfg.get("ref_cdf", {}) or {}
            per = ref.get("per_band") or []
            for c in range(min(C, len(per))):
                lut = per[c] or {}
                x_n = np.asarray(lut.get("x")  or [0.0, 1.0], dtype=np.float32)
                y   = np.asarray(lut.get("y")  or [0.0, 1.0], dtype=np.float32)
                lo  = float(lut.get("lo", 0.0))
                hi  = float(lut.get("hi", 1.0))
                if not np.isfinite(lo): lo = 0.0
                if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0

                ch = x[..., c]
                z  = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
                flat = z.reshape(-1)
                rk_idx = np.argsort(flat)
                ranks  = np.empty_like(flat, dtype=np.float32)
                ranks[rk_idx] = np.linspace(0.0, 1.0, flat.size, endpoint=True)
                ranks = ranks.reshape(z.shape)
                z2 = np.interp(ranks, y, x_n)
                x[..., c] = z2 * (hi - lo) + lo

        return x[..., 0] if img.ndim == 2 else x

    def _get_export_image(self, filepath):
        """
        Deterministically produce the export image used by ML Manager (export_csv + training).

        **Critical for consistency across CanoPie:**
        If the parent ProjectTab provides `_get_export_image()`, delegate to it so ML Manager uses
        the exact same RAW + .ax replay path as `process_polygon` (crop/rotate/hist_match/resize/band_expr
        and optional sklearn classification label band).

        Fallback: use ML Manager's local RAW+.ax replay.
        """
        # Prefer the ProjectTab export pipeline (keeps ML exports identical to process_polygon)
        pt = getattr(self, "parent_tab", None)
        get_pt = getattr(pt, "_get_export_image", None) if pt is not None else None
        if callable(get_pt):
            try:
                out = get_pt(filepath)
                # ProjectTab returns (img, meta) where meta is a dict with 'C'
                if isinstance(out, tuple) and len(out) == 2:
                    img, meta = out
                    if isinstance(meta, dict):
                        try:
                            C = int(meta.get("C", (img.shape[2] if getattr(img, "ndim", 0) == 3 else 1)))
                        except Exception:
                            C = (img.shape[2] if getattr(img, "ndim", 0) == 3 else 1)
                        return img, C
                    # Some older variants returned (img, C)
                    try:
                        return img, int(meta)
                    except Exception:
                        return img, (img.shape[2] if getattr(img, "ndim", 0) == 3 else 1)
                # Unexpected shape: fall back
            except Exception:
                pass

        # Fallback: local RAW + .ax replay
        raw = self._load_raw_image(filepath)
        ax  = self._load_ax_mods(filepath)
        img, C = self._apply_ax_to_raw(raw, ax, force_hist=True)
        return img, C

    def _get_geometry_only_image(self, filepath):
        """
        Produce export image with ONLY geometric operations (crop, rotate, resize).
        
        This is used for segmentation masks where we need the final image dimensions
        to match where polygons were drawn, but we don't want histogram matching,
        band expressions, or classification bands that would alter the mask alignment.
        
        Returns (float32 image in HxWxC, C) - same as _get_export_image but geometry-only.
        """
        raw = self._load_raw_image(filepath)
        ax  = self._load_ax_mods(filepath)
        img, C = self._apply_ax_geometry_only(raw, ax)
        return img, C

    def _apply_ax_geometry_only(self, raw_img, ax):
        """
        Apply ONLY geometric operations from .ax: crop, rotate, resize.
        
        Skips: histogram matching, band expressions, classification.
        This ensures the output image has the same dimensions as when polygons were drawn
        but without any operations that could misalign the mask.
        
        Returns (float32 image in HxWxC, C)
        """
        import numpy as np, cv2, logging

        if raw_img is None:
            return None, 0

        # ---- helpers (same as _apply_ax_to_raw) ----
        def _dims_after_rot(w, h, deg):
            return (h, w) if (deg % 360) in (90, 270) else (w, h)

        def _rect_after_rot(rect, src_w, src_h, deg):
            x = int(rect.get("x", 0)); y = int(rect.get("y", 0))
            w = int(rect.get("width", 0)); h = int(rect.get("height", 0))
            d = int(deg) % 360
            if d == 0:
                return {"x": x, "y": y, "width": w, "height": h}
            if d == 90:   # CW
                nx = src_h - (y + h); ny = x
                return {"x": nx, "y": ny, "width": h, "height": w}
            if d == 180:
                nx = src_w - (x + w); ny = src_h - (y + h)
                return {"x": nx, "y": ny, "width": w, "height": h}
            if d == 270:  # CW
                nx = y; ny = src_w - (x + w)
                return {"x": nx, "y": ny, "width": h, "height": w}
            return {"x": x, "y": y, "width": w, "height": h}

        def _scale_rect(rect, ref_w, ref_h, to_w, to_h):
            sx = float(to_w) / float(max(1, ref_w))
            sy = float(to_h) / float(max(1, ref_h))
            x = int(round(int(rect.get("x", 0)) * sx))
            y = int(round(int(rect.get("y", 0)) * sy))
            w = int(round(int(rect.get("width", 0)) * sx))
            h = int(round(int(rect.get("height", 0)) * sy))
            x = max(0, min(x, max(0, to_w - 1)))
            y = max(0, min(y, max(0, to_h - 1)))
            if x + w > to_w: w = max(0, to_w - x)
            if y + h > to_h: h = max(0, to_h - y)
            return {"x": x, "y": y, "width": w, "height": h}

        def _infer_crop_basis(ax_dict, raw_w, raw_h, rot_deg):
            ref = (ax_dict or {}).get("crop_rect_ref_size") or {}
            try:
                rw = int(ref.get("w") or 0); rh = int(ref.get("h") or 0)
            except Exception:
                rw = rh = 0
            aw, ah = _dims_after_rot(raw_w, raw_h, rot_deg)
            return "after_rotate" if (rw and rh and (rw, rh) == (aw, ah)) else "pre_rotate"

        def _rotate_any_channels(img, rot):
            if rot not in (90, 180, 270):
                return img
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 4):
                if rot == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if rot == 180: return cv2.rotate(img, cv2.ROTATE_180)
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            k = {90: 3, 180: 2, 270: 1}[rot]  # rot90 is CCW: 90° CW == 3×CCW
            return np.stack([np.rot90(img[:, :, i], k=k) for i in range(img.shape[2])], axis=2)

        def _resize_any_channels(img, new_w, new_h, interpolation):
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 4):
                return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            chans = [cv2.resize(img[:, :, i], (new_w, new_h), interpolation=interpolation) for i in range(img.shape[2])]
            return np.stack(chans, axis=2)

        # ---- start from HWC ----
        img = self._ensure_hwc(raw_img)

        # AX params - ONLY geometric ones
        try: rot = int(ax.get("rotate", 0)) % 360
        except Exception: rot = 0
        crop_rect = ax.get("crop_rect") or None
        crop_ref  = ax.get("crop_rect_ref_size") or None
        resize    = ax.get("resize") or None
        nodata_values = []
        if ax.get("nodata_enabled", True):
            nodata_values = list(ax.get("nodata_values", []) or [])
            if nodata_values:
                logging.info(f"[_apply_ax_geometry_only] Using nodata_values={nodata_values}")

        # ---- parse enabled flags (default True for backward compatibility) ----
        rotate_enabled = ax.get("rotate_enabled", True)
        crop_enabled = ax.get("crop_enabled", True)
        resize_enabled = ax.get("resize_enabled", True)

        # Uses shared utility supporting expressions
        def _build_nodata_mask(arr, nd_vals):
            """Build boolean mask where True = NoData pixel. Supports both numeric literals and threshold expressions."""
            from .utils import build_nodata_mask as _shared_build_nodata_mask
            return _shared_build_nodata_mask(arr, nd_vals, bgr_input=True)

        # op order from ax - ONLY geometric operations
        # Use dynamic order detection based on crop_rect_ref_size
        raw_h, raw_w = img.shape[:2]
        rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
        
        do_rotate_first = True  # Default: rotate first
        if crop_rect and rot in (90, 180, 270):
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w = int(crop_ref.get("w", 0)) or 0
                ref_h = int(crop_ref.get("h", 0)) or 0
                if ref_w > 0 and ref_h > 0:
                    # If ref matches original dims → crop was drawn BEFORE rotate
                    if (ref_w, ref_h) == (raw_w, raw_h):
                        do_rotate_first = False
                    # If ref matches rotated dims → crop was drawn AFTER rotate
                    elif (ref_w, ref_h) == (rotated_w, rotated_h):
                        do_rotate_first = True

        # ---- ops ----
        def _do_rotate():
            nonlocal img
            if not rotate_enabled:
                return
            if rot in (90, 180, 270):
                try:
                    img = _rotate_any_channels(img, rot)
                except Exception as e:
                    logging.warning(f"Rotation failed ({rot} deg): {e}")

        def _do_crop():
            nonlocal img
            if not crop_enabled:
                return
            if not crop_rect:
                return
            H, W = img.shape[:2]
            if H <= 0 or W <= 0:
                return

            # Get reference size for scaling
            if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                ref_w = int(crop_ref.get("w") or W)
                ref_h = int(crop_ref.get("h") or H)
            else:
                ref_w, ref_h = W, H

            x = int(crop_rect.get("x", 0))
            y = int(crop_rect.get("y", 0))
            w = int(crop_rect.get("width", W))
            h = int(crop_rect.get("height", H))

            # Scale if reference size differs from current size
            if ref_w != W or ref_h != H:
                sx = W / float(max(1, ref_w))
                sy = H / float(max(1, ref_h))
                x = int(round(x * sx))
                y = int(round(y * sy))
                w = int(round(w * sx))
                h = int(round(h * sy))

            x0 = max(0, min(x, W))
            y0 = max(0, min(y, H))
            x1 = max(0, min(x + w, W))
            y1 = max(0, min(y + h, H))
            
            if x1 > x0 and y1 > y0:
                img = img[y0:y1, x0:x1]
            else:
                logging.warning("Crop rect empty/out of bounds; skipping crop.")

        def _do_resize():
            nonlocal img
            if not resize_enabled:
                return
            if not isinstance(resize, dict) or not resize:
                return
            h0, w0 = img.shape[:2]
            if h0 <= 0 or w0 <= 0:
                return

            # Absolute pixels
            if ("px_w" in resize) or ("px_h" in resize):
                tw = int(resize.get("px_w", 0) or 0)
                th = int(resize.get("px_h", 0) or 0)
                if tw > 0 and th > 0:
                    new_w, new_h = tw, th
                elif tw > 0:
                    s = tw / float(w0); new_w = tw; new_h = max(1, int(round(h0 * s)))
                elif th > 0:
                    s = th / float(h0); new_h = th; new_w = max(1, int(round(w0 * s)))
                else:
                    new_w, new_h = w0, h0
            # Percent scale
            elif "scale" in resize:
                s = float(resize.get("scale", 100.0)) / 100.0
                new_w = max(1, int(round(w0 * s)))
                new_h = max(1, int(round(h0 * s)))
            # Percent width/height (legacy)
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

            try:
                # Handle NoData during resize using NaN propagation
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
                    nd_mask = _build_nodata_mask(img, nodata_values)
                    if nd_mask is not None and nd_mask.any():
                        # Convert to float32 and replace NoData with NaN
                        work = img.astype(np.float32, copy=True)
                        if work.ndim == 2:
                            work[nd_mask] = np.nan
                        else:
                            for c in range(work.shape[2]):
                                work[..., c][nd_mask] = np.nan
                        
                        # Resize - NaN propagates through interpolation
                        work = _resize_any_channels(work, new_w, new_h, interp)
                        
                        # Replace NaN with NoData value
                        nan_mask = np.isnan(work)
                        if nan_mask.any():
                            work[nan_mask] = nd_restore_val
                        
                        img = work
                    else:
                        img = _resize_any_channels(img, new_w, new_h, interp)
                elif nodata_values:
                    # Expression-only NoData - still apply mask but use 0.0 for restore
                    nd_mask = _build_nodata_mask(img, nodata_values)
                    if nd_mask is not None and nd_mask.any():
                        work = img.astype(np.float32, copy=True)
                        if work.ndim == 2:
                            work[nd_mask] = np.nan
                        else:
                            for c in range(work.shape[2]):
                                work[..., c][nd_mask] = np.nan
                        work = _resize_any_channels(work, new_w, new_h, interp)
                        nan_mask = np.isnan(work)
                        if nan_mask.any():
                            work[nan_mask] = 0.0  # Use 0 as fallback NoData value
                        img = work
                    else:
                        img = _resize_any_channels(img, new_w, new_h, interp)
                else:
                    img = _resize_any_channels(img, new_w, new_h, interp)
            except Exception as e:
                logging.warning(f"Resize failed to {new_w}x{new_h}: {e}")

        # Execute crop/rotate in the correct order based on crop_rect_ref_size
        if do_rotate_first:
            _do_rotate()
            _do_crop()
        else:
            _do_crop()
            _do_rotate()
        
        # Always do resize after crop/rotate
        _do_resize()

        img = img.astype(np.float32, copy=False)
        C = img.shape[2] if img.ndim == 3 else 1
        return img, C




    def _compute_bandmath_layers(self, chans, formulas):
        """
        chans: list of 2D float32 arrays in export order (R,G,B,band_4..)
        formulas: dict name->expr using b1..bN
        Returns (names, layers[list of 2D float32 arrays])
        """
        import numpy as np, re
        names, layers = [], []
        if not formulas:
            return names, layers

        # Build HxWxC float stack for eval (preserve NaNs)
        if len(chans) == 1:
            x = chans[0].astype(np.float32, copy=False)
        else:
            x = np.dstack([c.astype(np.float32, copy=False) for c in chans])

        C = 1 if x.ndim == 2 else x.shape[2]
        for nm, expr in formulas.items():
            try:
                # reuse your evaluator semantics
                if x.ndim == 2:
                    # b1 only
                    res = self._eval_band_expression(x, expr.replace("b2", "b9999").replace("b3","b9999"))
                else:
                    res = self._eval_band_expression(x, expr)
                if res is None:
                    continue
                # if boolean, cast → {0.0,1.0}
                if res.dtype == np.bool_:
                    res = res.astype(np.float32, copy=False)
                else:
                    res = res.astype(np.float32, copy=False)
                names.append(str(nm))
                layers.append(res)
            except Exception:
                # skip bad formulas instead of crashing export
                continue
        return names, layers


    def _predict_class_map_from_bundle(self, chans, bundle, nodata_values=None, mask_polygon_mask=None):
        """
        chans: list of 2D float32 in export order (R,G,B,band_4..)
        bundle: dict saved by your train_models() with keys 'model','feature_names','label_names'
        nodata_values: list of values to treat as NoData (these pixels are not classified)
        mask_polygon_mask: optional boolean mask (H, W) where True = masked (not classified)
        
        Returns 2D float32 array of class ids (0..K-1), with NoData pixels set to the first
        nodata_value (or NaN if no nodata_values provided but pixel has NaN/Inf).
        Returns None on failure.
        """
        import numpy as np, pickle

        if not bundle or ("model" not in bundle) or ("feature_names" not in bundle):
            return None

        H, W = chans[0].shape
        feat_names = list(bundle.get("feature_names") or [])
        model = bundle["model"]
        label_names = list(bundle.get("label_names") or [])

        # Map feature name -> channel matrix
        def _feat_to_mat(fn):
            fn = str(fn)
            if fn == "red_channel"   and len(chans) >= 1: return chans[0]
            if fn == "green_channel" and len(chans) >= 2: return chans[1]
            if fn == "blue_channel"  and len(chans) >= 3: return chans[2]
            if fn.startswith("band_"):
                try:
                    bi = int(fn.split("_", 1)[1])  # band_4 -> 4
                    idx = bi - 1                   # 1-based -> 0-based
                    if 0 <= idx < len(chans):
                        return chans[idx]
                except Exception:
                    return None
            return None

        mats = []
        for fn in feat_names:
            m = _feat_to_mat(fn)
            if m is None:
                return None
            mats.append(m.astype(np.float32, copy=False))

        # Build combined mask (NoData + polygon) using shared utility
        combined_mask = np.zeros((H, W), dtype=bool)
        
        # Add NoData mask using shared function (supports expressions like b1<123)
        if nodata_values and len(chans) > 0:
            from .utils import build_nodata_mask as _shared_build_nodata_mask
            # Build full image from channels to pass to shared function
            full_img = np.stack([chans[i].astype(np.float32) for i in range(len(chans))], axis=-1) if len(chans) > 1 else chans[0][..., None]
            nodata_mask = _shared_build_nodata_mask(full_img, nodata_values, bgr_input=True)
            if nodata_mask is not None:
                combined_mask |= nodata_mask
        
        # Add polygon mask
        if mask_polygon_mask is not None:
            combined_mask |= mask_polygon_mask

        X = np.stack([m.reshape(-1) for m in mats], axis=1)  # (H*W, F)
        valid = ~combined_mask.reshape(-1)  # Use combined mask
        
        if not np.any(valid):
            return None

        # Use float32 to allow NoData values
        # Initialize with first nodata value (or 0 if none)
        nodata_fill = float(nodata_values[0]) if nodata_values else np.nan
        y_all = np.full((H * W,), nodata_fill, dtype=np.float32)
        
        try:
            y_pred = model.predict(X[valid])
        except Exception:
            return None

        # Convert strings -> numeric ids if needed
        # FIX: Use label_names from bundle, or fallback to model.classes_ (like project_tab.py)
        if y_pred.dtype.kind in ("U", "S", "O"):
            # Build idx_map: prefer label_names, fallback to model.classes_
            if label_names:
                idx_map = {lbl: i for i, lbl in enumerate(label_names)}
            else:
                # Fallback to model.classes_ (this is what project_tab.py does)
                classes = list(getattr(model, "classes_", []))
                idx_map = {lbl: i for i, lbl in enumerate(classes)}
            y_num = np.array([idx_map.get(val, 0) for val in y_pred], dtype=np.float32)
        else:
            y_num = y_pred.astype(np.float32, copy=False)

        # Only set valid pixels - NoData pixels keep their nodata_fill value
        y_all[valid] = y_num
        return y_all.reshape(H, W)

    def _channels_in_export_order(self, img):
        """
        Return a list of 2D float32 arrays (each C-contiguous) in export order.
        If exactly 3 channels, map OpenCV BGR->RGB; if >3, swap first 3 BGR->RGB
        and keep remaining bands as-is. Ensures dtype=float32 and C-contiguous memory.
        """
        import numpy as np

        if img.ndim == 3:
            C = img.shape[2]
            if C == 3:
                chans = [img[:, :, 2], img[:, :, 1], img[:, :, 0]]  # R,G,B from B,G,R
            elif C > 3:
                # FIX: First 3 channels are BGR from OpenCV, swap to RGB
                # Then append remaining bands (band_expression result, label band, etc.) as-is
                chans = [img[:, :, 2], img[:, :, 1], img[:, :, 0]]  # BGR -> RGB for first 3
                for i in range(3, C):
                    chans.append(img[:, :, i])  # Keep additional bands in order
            else:
                chans = [img[:, :, i] for i in range(C)]  # C == 1 or 2
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
        Map polygon coords -> image pixel coords for the given export image shape.
        
        Priority:
          0) NEW: If polygon_data has coord_space='image' with image_ref_size, scale from ref to target.
             This is the primary path for modern polygons stored after geometric operations.
          1) If viewer exists and coords are scene-based: map scene->item (pixmap) and scale to image size.
          2) Else if polygon_data has 'pixmap_size' (legacy): scale by (img_w/pixmap_w, img_h/pixmap_h).
          3) Else: assume points are already in target image coords (floor+clamp).
        """
        from PyQt5 import QtCore

        if not points or img_shape is None:
            return []

        H = int(img_shape[0])
        W = int(img_shape[1])

        # 0) NEW: Handle coord_space='image' with image_ref_size (modern polygon format)
        # Polygons are now stored in image coordinates with a reference size that represents
        # the effective image size after .ax geometric operations. We just need to scale
        # from that reference size to the actual export image size.
        if polygon_data:
            coord_space = (polygon_data.get('coord_space') or '').lower()
            if coord_space == 'image':
                ref = polygon_data.get('image_ref_size') or {}
                ref_w = int(ref.get('w') or 0)
                ref_h = int(ref.get('h') or 0)
                
                if ref_w > 0 and ref_h > 0:
                    # Scale from reference size to export image size
                    sx = float(W) / float(ref_w)
                    sy = float(H) / float(ref_h)
                    
                    out = []
                    for (x, y) in points:
                        xi = int(float(x) * sx)
                        yi = int(float(y) * sy)
                        # Clamp to valid range
                        xi = max(0, min(xi, W - 1))
                        yi = max(0, min(yi, H - 1))
                        out.append((xi, yi))
                    return out
                else:
                    # Reference size not set but coord_space is 'image' - assume already correct
                    out = []
                    for (x, y) in points:
                        xi = int(float(x))
                        yi = int(float(y))
                        xi = max(0, min(xi, W - 1))
                        yi = max(0, min(yi, H - 1))
                        out.append((xi, yi))
                    return out

        # 1) Try live viewer mapping (scene -> item pixmap -> image pixels)
        # This path is used when coord_space is 'scene' or not specified
        viewer = getattr(self.parent_tab, "get_viewer_by_filepath", lambda _p: None)(filepath)
        if viewer is not None and getattr(viewer, "_image", None) is not None:
            pm = viewer._image.pixmap()
            pw = float(max(1, pm.width()))
            ph = float(max(1, pm.height()))
            sx = W / pw
            sy = H / ph

            out = []
            for (sx_scene, sy_scene) in points:
                p_item = viewer._image.mapFromScene(QtCore.QPointF(float(sx_scene), float(sy_scene)))
                xi = int(p_item.x() * sx)  # floor (no rounding)
                yi = int(p_item.y() * sy)
                if xi < 0: xi = 0
                if yi < 0: yi = 0
                if xi >= W: xi = W - 1
                if yi >= H: yi = H - 1
                out.append((xi, yi))
            return out

        # If a viewer exists but no _image item, fall back to its helper but still floor+clamp.
        if viewer is not None:
            out = []
            for (x, y) in points:
                q = self.parent_tab.scene_to_image_coords(viewer, QtCore.QPointF(float(x), float(y)))
                xi = int(q.x())
                yi = int(q.y())
                if xi < 0: xi = 0
                if yi < 0: yi = 0
                if xi >= W: xi = W - 1
                if yi >= H: yi = H - 1
                out.append((xi, yi))
            return out

        # 2) Offline mapping using stored pixmap size in polygon payload (legacy support)
        if polygon_data:
            pm = polygon_data.get("pixmap_size") or polygon_data.get("pixmap")  # support either key
            if isinstance(pm, (list, tuple)) and len(pm) == 2:
                pix_w = float(pm[0]) if pm[0] else 1.0
                pix_h = float(pm[1]) if pm[1] else 1.0
                sx = W / pix_w
                sy = H / pix_h
                out = []
                for (x, y) in points:
                    xi = int(float(x) * sx)  # floor (no rounding)
                    yi = int(float(y) * sy)
                    if xi < 0: xi = 0
                    if yi < 0: yi = 0
                    if xi >= W: xi = W - 1
                    if yi >= H: yi = H - 1
                    out.append((xi, yi))
                return out

        # 3) Fallback: treat as already in image coords (floor+clamp)
        out = []
        for (x, y) in points:
            xi = int(float(x))
            yi = int(float(y))
            if xi < 0: xi = 0
            if yi < 0: yi = 0
            if xi >= W: xi = W - 1
            if yi >= H: yi = H - 1
            out.append((xi, yi))
        return out


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
                
                # ---------- Load NoData values from .ax file ----------
                nodata_values = []
                try:
                    import json as json_mod
                    base = os.path.splitext(os.path.basename(filepath))[0] + ".ax"
                    pf = getattr(self, "project_folder", None)
                    ax_candidates = []
                    if pf:
                        ax_candidates.append(os.path.join(os.fspath(pf), base))
                        ax_candidates.append(os.path.join(os.fspath(pf), "global.ax"))
                    ax_candidates.append(os.path.join(os.path.dirname(filepath), base))
                    ax_candidates.append(os.path.join(os.path.dirname(filepath), "global.ax"))
                    for axp in ax_candidates:
                        if os.path.exists(axp):
                            with open(axp, "r", encoding="utf-8") as f:
                                ax_data = json_mod.load(f) or {}
                            if ax_data.get("nodata_enabled", True):
                                nodata_values = list(ax_data.get("nodata_values", []) or [])
                                if nodata_values:
                                    logging.info(f"[export_csv_data] Loaded nodata_values={nodata_values} from {axp}")
                            break
                except Exception as e:
                    logging.debug(f"[export_csv_data] Could not load .ax for {filepath}: {e}")
                
                # ---------- Build NoData mask ----------
                def _is_nodata_pixel(val, nd_vals):
                    """Check if a pixel value matches any NoData value."""
                    if not nd_vals:
                        return False
                    if not np.isfinite(val):
                        return True
                    for nd in nd_vals:
                        try:
                            nd_val = float(nd)
                            abs_nd = abs(nd_val)
                            if abs_nd > 1e+30:
                                tol = abs_nd * 0.01
                            elif abs_nd > 1e+10:
                                tol = abs_nd * 0.001
                            elif abs_nd > 100:
                                tol = abs_nd * 0.001
                            else:
                                tol = 0.01
                            if abs(val - nd_val) < tol:
                                return True
                        except Exception:
                            pass
                    return False
                
                def _build_nodata_mask_2d(ch, nd_vals):
                    """Build boolean mask where True = NoData pixel. Uses shared utility supporting expressions."""
                    from .utils import build_nodata_mask as _shared_build_nodata_mask
                    # build_nodata_mask expects HxW or HxWxC; for 2D we'll rely on it handling 2D arrays
                    return _shared_build_nodata_mask(ch, nd_vals, bgr_input=True) or np.zeros(ch.shape, dtype=bool)
                
                # Build combined NoData mask (any channel has NoData)
                nd_mask = np.zeros((H, W), dtype=bool)
                if nodata_values:
                    # CRITICAL FIX: Prefer the tracked mask from ProjectTab's _apply_ax_to_raw
                    # over rebuilding from transformed pixels. Boolean expressions like "b1<50"
                    # will evaluate differently on interpolated pixels after resize.
                    pt = getattr(self, "parent_tab", None)
                    tracked_mask = getattr(pt, "_last_export_nodata_mask", None) if pt else None
                    tracked_fp = getattr(pt, "_last_export_nodata_filepath", None) if pt else None
                    fp_key = os.path.normcase(os.path.abspath(filepath))
                    
                    if (tracked_mask is not None and tracked_fp and 
                        tracked_mask.shape == (H, W) and
                        os.path.normcase(os.path.abspath(tracked_fp)) == fp_key):
                        # Use the tracked mask - it was built on original pixels and transformed correctly
                        nd_mask = tracked_mask.copy()
                        logging.info(f"[export_csv_data] Using tracked NoData mask from ProjectTab ({nd_mask.sum()} masked pixels)")
                    else:
                        # Fallback: rebuild mask from transformed image (for non-boolean expressions or if tracking failed)
                        for ch in chans:
                            nd_mask |= _build_nodata_mask_2d(ch, nodata_values)
                        nd_count = nd_mask.sum()
                        if nd_count > 0:
                            logging.info(f"[export_csv_data] Rebuilt NoData mask with {nd_count} masked pixels for {os.path.basename(filepath)}")

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
                        
                        # Skip NoData pixels
                        if nodata_values and nd_mask[yi, xi]:
                            logging.debug(f"[export_csv_data] Skipping NoData pixel at ({xi}, {yi})")
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
                            # Skip NoData pixel
                            if nodata_values and nd_mask[yi, xi]:
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
                            # Combine with NoData mask: valid pixels are inside polygon AND not NoData
                            if nodata_values:
                                valid_mask = (mask == 255) & (~nd_mask)
                            else:
                                valid_mask = (mask == 255)
                            means = []
                            if any_rgb:
                                for i in range(3):
                                    vals = chans[i][valid_mask] if i < len(chans) else np.array([])
                                    means.append(float(np.nanmean(vals)) if vals.size else np.nan)
                                for i in range(max_extras):
                                    b = 3 + i
                                    vals = chans[b][valid_mask] if b < len(chans) else np.array([])
                                    means.append(float(np.nanmean(vals)) if vals.size else np.nan)
                            else:
                                for i in range(max_small):
                                    vals = chans[i][valid_mask] if i < len(chans) else np.array([])
                                    means.append(float(np.nanmean(vals)) if vals.size else np.nan)

                            row = [group_name, filepath] + means
                            all_rows.append(dict(zip(header, row)))

                    elif extraction_mode == "All Pixel Values":
                        if len(pts_img) == 1:
                            xi, yi = pts_img[0]
                            if not _in_bounds(W, H, xi, yi):
                                continue
                            # Skip NoData pixel
                            if nodata_values and nd_mask[yi, xi]:
                                continue
                            vals = _align_point_values(any_rgb, max_extras, max_small, chans, xi, yi)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))
                        else:
                            mask = _poly_mask(pts_img, H, W)
                            # Combine with NoData mask
                            if nodata_values:
                                valid_mask = (mask == 255) & (~nd_mask)
                            else:
                                valid_mask = (mask == 255)
                            ys, xs = np.where(valid_mask)
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
                            # Skip NoData pixel
                            if nodata_values and nd_mask[yi, xi]:
                                continue
                            vals = _align_window_values(any_rgb, max_extras, max_small, chans, xi, yi, k)
                            row = [group_name, filepath, xi, yi] + vals
                            all_rows.append(dict(zip(header, row)))
                        else:
                            mask = _poly_mask(pts_img, H, W)
                            # Combine with NoData mask
                            if nodata_values:
                                valid_mask = (mask == 255) & (~nd_mask)
                            else:
                                valid_mask = (mask == 255)
                            ys, xs = np.where(valid_mask)
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
          - list of individual points [[x1,y1], [x2,y2], ...] - each becomes a 1-point polygon
        """
        def _looks_like_polygon(seq):
            """Check if seq looks like a polygon (list of (x,y) pairs where each pair is a list/tuple)"""
            return isinstance(seq, (list, tuple)) and len(seq) > 0 and \
                   all(isinstance(p, (list, tuple)) and len(p) == 2 for p in seq)
        
        def _looks_like_dict_points(seq):
            """Check if seq is a list of dict points [{'x':..,'y':..}, ...]"""
            return isinstance(seq, (list, tuple)) and len(seq) > 0 and \
                   all(isinstance(p, dict) and 'x' in p and 'y' in p for p in seq)

        if not raw_pts:
            return []

        # Case 1: list of polygons [[(x,y)...], [(x,y)...], ...]
        # Check if first element looks like a polygon (list of coordinate pairs)
        if isinstance(raw_pts, (list, tuple)) and len(raw_pts) > 0 and _looks_like_polygon(raw_pts[0]):
            polys = []
            for p in raw_pts:
                parsed = self.parse_polygon_points(p)
                if parsed:
                    polys.append(parsed)
            return polys

        # Case 2: list of dict points [{'x':..,'y':..}, ...] - treat each as individual point
        if _looks_like_dict_points(raw_pts):
            # Each dict is a single point -> each becomes a 1-point "polygon"
            polys = []
            for pt in raw_pts:
                polys.append([[pt['x'], pt['y']]])
            return polys

        # Case 3: list of coordinate pairs [[x1,y1], [x2,y2], ...]
        # This is ambiguous - could be one polygon OR multiple individual points
        # Heuristic: if there are many "vertices" that are far apart, treat as individual points
        if _looks_like_polygon(raw_pts):
            # Check if this looks like individual points (scattered) vs a polygon (connected)
            # If only 1-2 points, definitely individual points
            if len(raw_pts) <= 2:
                return [[[p[0], p[1]]] for p in raw_pts]  # Each point becomes its own polygon
            
            # For 3+ points: try to detect if it's meant to be a polygon or individual points
            # A true polygon typically has vertices close together forming a connected shape
            # Individual points are typically scattered (training data points)
            # 
            # Heuristic: Use the point structure stored in the JSON
            # If it came from polygon drawing, it's likely a polygon
            # If it came from point clicking, each point should be separate
            #
            # For now, treat as a single polygon (original behavior for backward compatibility)
            # The caller should use a different structure for individual points
            parsed = self.parse_polygon_points(raw_pts)
            return [parsed] if parsed else []

        # Case 4: single dict point {'x':..,'y':..}
        if isinstance(raw_pts, dict) and 'x' in raw_pts and 'y' in raw_pts:
            return [[[raw_pts['x'], raw_pts['y']]]]

        # Fallback: try to parse as single polygon
        parsed = self.parse_polygon_points(raw_pts)
        return [parsed] if parsed else []

    def generate_segmentation_images(self):
        """
        Segmentation masks (16-bit) using ONLY geometric .ax operations (rotate/crop/resize).
        
        Uses geometry-only export to ensure mask dimensions match where polygons were drawn,
        without histogram matching or band expressions that could cause misalignment.
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
                    # FIX: Use geometry-only export to get correct dimensions
                    # This ensures the mask matches the image dimensions when polygons were drawn,
                    # without histogram matching or band expressions that could cause misalignment.
                    img, _C = self._get_geometry_only_image(filepath)
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
        self.resize(560, 600)

        # Apply custom styling for "Dark Green" ticks/arrows and "Gold" highlights
        # Note: "Gold tick" logic applied to checkboxes.
        # "Arrows dark green" applied to ToolButtons via color.
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QCheckBox::indicator:checked {
                background-color: #FFD700; 
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked {
                background-color: white; 
                border: 1px solid gray;
            }
            QToolButton {
                color: #006400; /* Dark Green text/arrow */
                font-weight: bold;
                font-size: 11pt;
            }
            QGroupBox {
                font-weight: bold;
                color: #006400;
            }
        """)

        # Main layout: Scroll area to handle expanded groups
        main_layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(5, 5, 5, 5)
        
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # --- Statistics (Collapsible) ---
        stats_group = CollapsibleBox("Statistics to compute")
        st_layout   = stats_group.content_layout

        self.chk_mean   = QtWidgets.QCheckBox("Mean")
        self.chk_median = QtWidgets.QCheckBox("Median")
        self.chk_std    = QtWidgets.QCheckBox("Standard deviation")
        self.chk_quant  = QtWidgets.QCheckBox("Quantiles (comma list)")
        self.le_quant   = QtWidgets.QLineEdit("5,25,50,75,95")          
        
        self.chk_count01 = QtWidgets.QCheckBox("Count 0/1 (if array is binary or classification masks 0,1,2,3 and so on)")
        self.chk_count01.setChecked(False)

        # NEW: Scene-wide statistics checkboxes
        self.chk_scene_mean   = QtWidgets.QCheckBox("Scene Mean (Full Image)")
        self.chk_scene_median = QtWidgets.QCheckBox("Scene Median (Full Image)")
        self.chk_scene_std    = QtWidgets.QCheckBox("Scene Std Dev (Full Image)")
        
        self.chk_scene_mean.setChecked(True)
        self.chk_scene_median.setChecked(True)
        self.chk_scene_std.setChecked(True)

        # Defaults
        self.chk_mean.setChecked(True)
        self.chk_median.setChecked(True)
        self.chk_std.setChecked(True)
        self.chk_quant.setChecked(True)
        self.le_quant.setEnabled(True)
        self.chk_quant.toggled.connect(self.le_quant.setEnabled)

        for w in (self.chk_mean, self.chk_median, self.chk_std, self.chk_quant, self.le_quant, self.chk_count01,
                  self.chk_scene_mean, self.chk_scene_median, self.chk_scene_std):
            st_layout.addWidget(w)
            
        # By default, expand stats
        stats_group.toggle_button.setChecked(True)
        vbox.addWidget(stats_group)

        # --- Polygon Options (Collapsible) ---
        extra_group  = CollapsibleBox("Polygon Options")
        extra_layout = QtWidgets.QGridLayout()
        extra_group.content_layout.addLayout(extra_layout)

        self.chk_shrink = QtWidgets.QCheckBox("Shrink / Swell polygons")
        self.sp_factor  = QtWidgets.QDoubleSpinBox()
        self.sp_factor.setRange(0.0, 1.0)
        self.sp_factor.setSingleStep(0.01)
        self.sp_factor.setValue(0.07)
        self.cmb_dir    = QtWidgets.QComboBox()
        self.cmb_dir.addItems(["Shrink", "Swell"])

        self.chk_shrink.setChecked(False)
        self.sp_factor.setEnabled(True)
        self.cmb_dir.setEnabled(True)
        self.chk_shrink.toggled.connect(lambda b: (self.sp_factor.setEnabled(b), self.cmb_dir.setEnabled(b)))

        self.chk_export_mod_polys = QtWidgets.QCheckBox("Export modified polygons JSON")
        self.chk_export_mod_polys.setChecked(False)

        self.chk_exif = QtWidgets.QCheckBox("Include EXIF metadata")
        self.chk_exif.setChecked(False)

        extra_layout.addWidget(self.chk_shrink, 0, 0, 1, 2)
        extra_layout.addWidget(QtWidgets.QLabel("Factor:"), 1, 0)
        extra_layout.addWidget(self.sp_factor, 1, 1)
        extra_layout.addWidget(self.cmb_dir,   2, 0, 1, 2)
        extra_layout.addWidget(self.chk_export_mod_polys, 3, 0, 1, 2)
        extra_layout.addWidget(self.chk_exif,  4, 0, 1, 2)

        vbox.addWidget(extra_group)

        # --- Image Options (Collapsible) ---
        # Was: Channel-math indices
        # Merged NoData UI + RF Checkbox + BandMath here
        img_opts_group  = CollapsibleBox("Image Options")
        img_opts_layout = img_opts_group.content_layout

        # 1. NoData UI (Moved here)
        nodata_frame = QtWidgets.QGroupBox("NoData masking")
        nodata_layout = QtWidgets.QGridLayout(nodata_frame)
        
        self.chk_nodata = QtWidgets.QCheckBox("Mask these NoData values in all stats & band-math")
        self.chk_nodata.setChecked(False)

        self.le_nodata = QtWidgets.QLineEdit()
        self.le_nodata.setPlaceholderText("{-9999, -32768}  or  -9999, -32768")
        self.le_nodata.setEnabled(False)
        self.chk_nodata.toggled.connect(self.le_nodata.setEnabled)
        
        # Try to auto-detect nodata values
        self._try_prepopulate_nodata(parent)

        nodata_layout.addWidget(self.chk_nodata, 0, 0, 1, 2)
        nodata_layout.addWidget(QtWidgets.QLabel("Values:"), 1, 0)
        nodata_layout.addWidget(self.le_nodata,             1, 1)
        hint_label = QtWidgets.QLabel("(Auto-detected from .ax files if available)")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        nodata_layout.addWidget(hint_label, 2, 0, 1, 2)
        
        img_opts_layout.addWidget(nodata_frame)

        # 2. sklearn classification Checkbox (Positioned ABOVE sklearn as requested "put what is under No makisng group aboce sklearn classication")
        # Wait, user said: "put what is under No makisng group aboce sklearn classication"
        # I interpret this as: NoData > Sklearn > BandMath
        
        self.chk_rf = QtWidgets.QCheckBox("use_scikit-learn pkl model classification")
        self.chk_rf.setChecked(False)
        img_opts_layout.addWidget(self.chk_rf)
        
        # 3. Band Math UI
        DEFAULT_BANDMATH_TEXT = (
            '{\n'
            '  "boolean1": "b1 >150",\n'
            '  "boolean2": "(b1 >150) & (b2>165)",\n'
            '  "boolean3": "(b2 / (b1 + b2 + b3))>0.41",\n'
            '  "sum": "b1 + b2 + b3",\n'
            '  "GCC": "b2 / (b1 + b2 + b3)",\n'
            '  "EXG": "2*b2 - (b1 + b3)",\n'
            '  "RCC": "b3 / (b1 + b2 + b3)",\n'
            '  "BCC": "b1 / (b1 + b2 + b3)",\n'
            '  "WDX_2": "(2*b1) + b3 - (2*b2)",\n'
            '  "WDX": "b1 + 2*b1 - b2",\n'
            '  "WDX_3": "b1 + 2*b3 - 2*b2"\n'
            '}'
        )
        
        bm_frame = QtWidgets.QGroupBox("Channel-math indices")
        bm_layout = QtWidgets.QVBoxLayout(bm_frame)
        
        self.chk_bandmath = QtWidgets.QCheckBox("Compute these indices")
        self.chk_bandmath.setChecked(False)

        self.te_bandmath = QtWidgets.QPlainTextEdit()
        self.te_bandmath.setPlaceholderText(
            'JSON or lines like:  GCC=b2/(b1+b2+b3), EXG=2*b2-(b1+b3)\n'
            'b1..bN refer to export-order channels (RGB -> b1,b2,b3, then extras).'
        )
        self.te_bandmath.setMinimumHeight(120)
        self.te_bandmath.setPlainText(DEFAULT_BANDMATH_TEXT)

        bm_layout.addWidget(self.chk_bandmath)
        bm_layout.addWidget(self.te_bandmath)
        
        img_opts_layout.addWidget(bm_frame)
        
        # Expand Image Options group by default
        img_opts_group.toggle_button.setChecked(True)
        vbox.addWidget(img_opts_group)

        # --- Saving Options (Collapsible) ---
        save_group = CollapsibleBox("Saving Options")
        save_layout = QtWidgets.QGridLayout()
        save_group.content_layout.addLayout(save_layout)
        
        self.chk_custom_save = QtWidgets.QCheckBox("Use Custom Export Path")
        self.chk_custom_save.setToolTip("If checked, all CSVs will be saved to the specified folder instead of the project default.")
        self.chk_custom_save.setChecked(False)
        
        self.le_save_path = QtWidgets.QLineEdit()
        self.le_save_path.setPlaceholderText("Select folder...")
        self.le_save_path.setEnabled(False)
        
        self.btn_browse_save = QtWidgets.QPushButton("Browse...")
        self.btn_browse_save.setEnabled(False)
        self.btn_browse_save.clicked.connect(self._on_browse_save_path)
        
        self.chk_custom_save.toggled.connect(self.le_save_path.setEnabled)
        self.chk_custom_save.toggled.connect(self.btn_browse_save.setEnabled)
        
        save_layout.addWidget(self.chk_custom_save, 0, 0, 1, 2)
        save_layout.addWidget(QtWidgets.QLabel("Save Path:"), 1, 0)
        save_layout.addWidget(self.le_save_path, 1, 1)
        save_layout.addWidget(self.btn_browse_save, 1, 2)
        
        vbox.addWidget(save_group)

        # Spacer to push everything up
        vbox.addStretch()

        # --- OK / Cancel ---
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        
        # Add buttons to main_layout (outside scroll)
        main_layout.addWidget(btns)

    def _on_browse_save_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder", "", options=options)
        if folder:
            self.le_save_path.setText(folder)

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

    def _try_prepopulate_nodata(self, parent):
        """
        Try to auto-detect nodata values from .ax files in the project.
        If found, pre-populate the nodata field and check the checkbox.
        """
        import os, json, logging
        try:
            project_folder = getattr(parent, 'project_folder', None) if parent else None
            if not project_folder or not os.path.isdir(project_folder):
                return
            
            # Look for any .ax file with nodata_values
            # PERFORMANCE FIX: Limit scan to first 50 files to prevent freezing on large datasets
            nodata_vals = None
            scanned_count = 0
            
            # Using scantir is faster than listdir for large directories
            with os.scandir(project_folder) as it:
                for entry in it:
                    if scanned_count > 50:
                        break
                    
                    if entry.is_file() and entry.name.lower().endswith('.ax'):
                        scanned_count += 1
                        try:
                            with open(entry.path, 'r', encoding='utf-8') as f:
                                ax = json.load(f) or {}
                            if ax.get('nodata_enabled', True):
                                nd = ax.get('nodata_values', [])
                                if nd:
                                    nodata_vals = nd
                                    logging.info(f"[AnalysisOptionsDialog] Auto-detected nodata_values={nd} from {entry.name}")
                                    break
                        except Exception:
                            continue
            
            if nodata_vals:
                # Pre-populate the field and check the checkbox
                self.le_nodata.setText(", ".join(str(v) for v in nodata_vals))
                self.chk_nodata.setChecked(True)
                self.le_nodata.setEnabled(True)
        except Exception as e:
            logging.debug(f"[AnalysisOptionsDialog] Failed to auto-detect nodata: {e}")

    @staticmethod
    def _parse_nodata_text(text: str):
        """
        Parse NoData values from text input.
        Supports numeric literals AND boolean expressions (b1<123, b2>189, etc.).
        """
        from .utils import parse_nodata_text
        return parse_nodata_text(text)

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
                        continue

        # NEW: NoData values (empty list when feature disabled)
        nodata_values = []
        if self.chk_nodata.isChecked():
            nodata_values = self._parse_nodata_text(self.le_nodata.text().strip())
            logging.info(f"[AnalysisOptionsDialog.get_options] NoData checkbox checked, parsed values: {nodata_values}")
        else:
            logging.info(f"[AnalysisOptionsDialog.get_options] NoData checkbox NOT checked - nodata_values will be empty (will fall back to .ax file)")

        # NEW: Custom Save Path
        custom_save_path = None
        if self.chk_custom_save.isChecked():
            path_cand = self.le_save_path.text().strip()
            if path_cand:
                custom_save_path = path_cand

        opts = {
            "stats": {
                "mean":      self.chk_mean.isChecked(),
                "median":    self.chk_median.isChecked(),
                "std":       self.chk_std.isChecked(),
                "quantiles": quantiles,
                "quantiles": quantiles,
                "count01":   self.chk_count01.isChecked(),
                "scene_mean":   self.chk_scene_mean.isChecked(),
                "scene_median": self.chk_scene_median.isChecked(),
                "scene_std":    self.chk_scene_std.isChecked(),
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

            # user-defined indices to compute during export
            "band_math": {
                "enabled":  self.chk_bandmath.isChecked(),
                "formulas": self._parse_bandmath(self.te_bandmath.toPlainText()),
            },

            # NEW: feed to processing (no overhead if empty)
            "nodata_values": nodata_values,
            
            # NEW: Custom Export Path
            "custom_save_path": custom_save_path,
        }
        return opts