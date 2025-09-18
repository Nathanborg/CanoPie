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

class EditablePolygonItem(QtWidgets.QGraphicsObject):
    polygon_modified = QtCore.pyqtSignal()

    def __init__(self, polygon, name="", is_rgb=False, parent=None):
        super(EditablePolygonItem, self).__init__(parent)
        self.polygon = polygon
        self.name = name
        self.is_rgb = is_rgb  # Determines polygon appearance

        # Set flags to make the polygon selectable and movable
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable |
            QtWidgets.QGraphicsItem.ItemIsMovable |
            QtWidgets.QGraphicsItem.ItemSendsGeometryChanges |
            QtWidgets.QGraphicsItem.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)

        # Enable caching to improve rendering performance
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)

        # Flag to track if the polygon is currently being moved
        self.is_moving = False

        # Define additional space for the label
        self.label_offset = QtCore.QPointF(10, -10)  # Adjust as needed for label position

    def boundingRect(self):
        # Original bounding rect of the polygon
        rect = self.polygon.boundingRect()

        if self.name:
            # Choose font size based on whether it's RGB
            font = QtGui.QFont()
            if self.is_rgb:
                font.setPointSize(25)  # Larger font for RGB
            else:
                font.setPointSize(15)  # Default font size

            metrics = QtGui.QFontMetrics(font)
            label_width = metrics.horizontalAdvance(self.name)
            label_height = metrics.height()

            # Define a rectangle for the label above and to the right of the polygon
            label_rect = QtCore.QRectF(
                rect.topRight() + self.label_offset,
                QtCore.QSizeF(label_width, label_height)
            )

            # Unite the label rect with the polygon rect to ensure both are visible
            rect = rect.united(label_rect)

        margin = 10
        rect.adjust(-margin, -margin, margin, margin)

        return rect

    def shape(self):
        """
        Override the shape method to return only the polygon's shape for interaction purposes.
        """
        path = QtGui.QPainterPath()
        path.addPolygon(self.polygon)
        return path
    def paint(self, painter, option, widget=None):
        # --- current zoom scale ---
        t = painter.worldTransform()
        scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

        # ---- stroke width scaling ----
        desired_px = 2.0
        min_px, max_px = 1.0, 6.0
        width_scene = desired_px / scale
        width_scene = max(min_px / scale, min(max_px / scale, width_scene))

        pen = QtGui.QPen(QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue)
        if self.isUnderMouse() or self.isSelected():
            pen.setColor(QtCore.Qt.magenta)
            width_scene *= 1.25
        pen.setWidthF(width_scene)

        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.transparent)
        painter.drawPolygon(self.polygon)

        # ---- zoom + resolution aware label ----
        if self.name and not self.is_moving:
            # get largest pixmap dimensions from the scene
            img_w = img_h = None
            scn = self.scene()
            if scn:
                biggest_area = -1
                for it in scn.items():
                    if isinstance(it, QtWidgets.QGraphicsPixmapItem):
                        pm = it.pixmap()
                        if not pm.isNull():
                            w, h = pm.width(), pm.height()
                            area = w * h
                            if area > biggest_area:
                                img_w, img_h = w, h
                                biggest_area = area

            # resolution boost for large images
            res_boost = 1.0
            if img_w and img_h:
                long_side = max(img_w, img_h)
                res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))

            # final pixel size
            base_px = 40 if self.is_rgb else 32  # bigger base
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            font = QtGui.QFont()
            font.setPixelSize(px)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtCore.Qt.red))

            metrics = QtGui.QFontMetrics(font)
            text_w = metrics.horizontalAdvance(self.name)
            text_h = metrics.height()

            bbox = self.polygon.boundingRect()
            pos = bbox.topRight() + self.label_offset

            br = self.boundingRect()
            if pos.x() + text_w > br.right():
                pos.setX(br.right() - text_w - 5)
            if pos.y() - text_h < br.top():
                pos.setY(bbox.top() + text_h + 5)

            painter.drawText(pos, self.name)


    def hoverEnterEvent(self, event):
        """
        Change cursor to pointing hand when hovering over the polygon.
        """
        self.setCursor(QtCore.Qt.PointingHandCursor)
        super(EditablePolygonItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """
        Revert cursor back to default when not hovering over the polygon.
        """
        self.setCursor(QtCore.Qt.ArrowCursor)
        super(EditablePolygonItem, self).hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_moving = True  # Start tracking movement
        super(EditablePolygonItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.is_moving:
                self.is_moving = False  # Movement finished
                self.polygon_modified.emit()  # Emit signal once after movement
        super(EditablePolygonItem, self).mouseReleaseEvent(event)

    def switch_to_tab(self, tab_name):
        """
        Switches to the tab with the given name.
        """
        for index in range(self.tab_widget.count()):
            if self.tab_widget.tabText(index) == tab_name:
                self.tab_widget.setCurrentIndex(index)
                self.status.showMessage(f"Switched to {tab_name}", 2000)  # Optional status message
                logging.info(f"Switched to tab: {tab_name}")
                return
        QtWidgets.QMessageBox.warning(self, "Tab Not Found", f"No tab named '{tab_name}' was found.")
        logging.warning(f"Attempted to switch to non-existent tab: {tab_name}")


class EditablePointItem(QtWidgets.QGraphicsObject):
    point_modified = QtCore.pyqtSignal()

    def __init__(self, points, name="", is_rgb=False, parent=None,
                 pixmap_item=None, points_are_pixmap_local=False):
        super(EditablePointItem, self).__init__(parent)
   
        self.points = points
        self.name = name
        self.is_rgb = is_rgb
        self.pixmap_item = pixmap_item
        self.points_are_pixmap_local = points_are_pixmap_local

        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable |
            QtWidgets.QGraphicsItem.ItemIsMovable |
            QtWidgets.QGraphicsItem.ItemSendsGeometryChanges |
            QtWidgets.QGraphicsItem.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        self.is_moving = False
        self.label_offset = QtCore.QPointF(10, -10)

    def _pixmap_pos(self):
        return self.pixmap_item.pos() if self.pixmap_item is not None else QtCore.QPointF(0, 0)

    def _scene_xy(self, p):
        """Return integer scene coords for a stored point `p`."""
        if self.points_are_pixmap_local:
            off = self._pixmap_pos()
            return int(off.x() + p.x()), int(off.y() + p.y())
        else:
            return int(p.x()), int(p.y())

    def boundingRect(self):
        if self.points.isEmpty():
            return QtCore.QRectF()

        xs, ys = [], []
        for p in self.points:
            sx, sy = self._scene_xy(p)
            xs.append(sx); ys.append(sy)
        rect = QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

        # Always give a minimum rect (so a lone point isn't too small)
        if rect.width() < 10 and rect.height() < 10:
            rect = rect.adjusted(-20, -20, 20, 20)

        rect.adjust(-5, -5, 5, 5)
        return rect


    def shape(self):
        path = QtGui.QPainterPath()
        if self.points.isEmpty():
            return path
        for p in self.points:
            sx, sy = self._scene_xy(p)
            path.addEllipse(sx, sy, 6, 6)  # bigger than a 1×1 rect
        return path

    
    def paint(self, painter, option, widget=None):
        # draw tiny points
        color = QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        for p in self.points:
            sx, sy = self._scene_xy(p)
            painter.drawRect(QtCore.QRectF(sx, sy, 1, 1))

        # ---- zoom + resolution aware label ----
        if self.name and not self.is_moving:
            t = painter.worldTransform()
            scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

            img_w = img_h = None
            if self.pixmap_item is not None:
                pm = self.pixmap_item.pixmap()
                if not pm.isNull():
                    img_w, img_h = pm.width(), pm.height()

            res_boost = 1.0
            if img_w and img_h:
                long_side = max(img_w, img_h)
                res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))

            base_px = 40 if self.is_rgb else 32
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            font = QtGui.QFont()
            font.setPixelSize(px)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtCore.Qt.red))

            metrics = QtGui.QFontMetrics(font)
            text_width = metrics.horizontalAdvance(self.name)
            text_height = metrics.height()

            xs, ys = [], []
            for p in self.points:
                sx, sy = self._scene_xy(p)
                xs.append(sx); ys.append(sy)
            rx, ry = min(xs), min(ys)
            text_position = QtCore.QPointF(rx, ry) + self.label_offset

            br = self.boundingRect()
            if text_position.x() + text_width > br.right():
                text_position.setX(br.right() - text_width - 5)
            if text_position.y() - text_height < br.top():
                text_position.setY(br.top() + text_height + 5)

            painter.drawText(text_position, self.name)

    def hoverEnterEvent(self, event):
        self.setCursor(QtCore.Qt.PointingHandCursor)
        super(EditablePointItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(QtCore.Qt.ArrowCursor)
        super(EditablePointItem, self).hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_moving = True
        super(EditablePointItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.is_moving:
                self.is_moving = False
                self.point_modified.emit()
        super(EditablePointItem, self).mouseReleaseEvent(event)


class ImageViewer(QtWidgets.QGraphicsView):
    polygon_drawn = QtCore.pyqtSignal(object)      # Signal to notify when a polygon/point item is drawn
    polygon_changed = QtCore.pyqtSignal()          # Signal to notify when any polygon is modified
    editing_finished = QtCore.pyqtSignal()         # Signal to notify when editing is finished
    pixel_clicked = QtCore.pyqtSignal(QtCore.QPointF, object)
  # Emits the point and pixel value (e.g., RGB or grayscale)

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._image = None

        # For drawing
        self.drawing = False
        self.currentPolygon = QtGui.QPolygonF()
        self.polygons = []

        # Temporary drawing item for visual feedback (will be a polygon item in polygon mode,
        # or a path item (showing circles) in point mode)
        self.temp_drawing_item = None

        # Enable panning when not drawing
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Flag to prevent recursive polygon_drawn signals
        self.programmatically_adding_polygon = False

        # Pending Group Name for Editing
        self.pending_group_name = None

        # Flag to indicate if editing is via Polygon Manager
        self.is_editing_group = False

        # Flags to track mouse buttons
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.last_pan_point = QtCore.QPoint()
        self.inspection_mode = False

        # Original Image Data Reference
        self.image_data = None

        # New attribute: drawing_mode (either "polygon" or "point")
        self.drawing_mode = "polygon"
        self.setMouseTracking(True)          # get move events with no buttons pressed
        self._last_hover_pixel = None        # (x0, y0) in base image coords
        self._mods_cache_source = None       # cache for .ax file
        self._mods_cache = None
        self.preview_prefers_index = False  # show the expression (last band) when True
    
    def zoom_out_to_fit(self):
        """
        Snap to most zoomed-out (fit whole PIXMAP), recentre, and keep _zoom in sync.
        Uses the item overload of fitInView to avoid sceneRect pitfalls.
        """
        if not self.has_image() or self._image is None or self._image.pixmap().isNull():
            return

        # Reset any transform/pan
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.resetTransform()

        # Fit the actual pixmap item (not sceneRect, which can be stale/empty)
        try:
            self.fitInView(self._image, QtCore.Qt.KeepAspectRatio)  # <— item overload
        except Exception:
            # Ultra-safe fallback to the pixmap’s rect in scene coordinates
            r = self._image.mapToScene(self._image.boundingRect()).boundingRect()
            if r.isValid() and r.width() > 0 and r.height() > 0:
                self.fitInView(r, QtCore.Qt.KeepAspectRatio)

        # Recentre on the item (in case fit math + scrollbars left us offset)
        try:
            self.centerOn(self._image)
        except Exception:
            pass

        # Keep wheel-zoom bookkeeping coherent
        self._zoom = 0

        # Normalize any temporary drawing pen width (~2 px on screen)
        if self.drawing and self.temp_drawing_item:
            scale = self.get_current_scale_factor()
            pen = self.temp_drawing_item.pen()
            pen.setWidthF(2.0 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)

        # Make sure the viewport repaints immediately
        self.viewport().update()

    def _load_ax_mods(self, image_path):
        
        if image_path == self._mods_cache_source:
            return self._mods_cache or {}
        import os, json, logging
        mods = {}
        if image_path:
            ax_name = os.path.splitext(os.path.basename(image_path))[0] + ".ax"
            candidates = []
            parent = self.parent()
            proj_folder = getattr(parent, "project_folder", None) if parent else None
            if proj_folder:
                candidates.append(os.path.join(proj_folder, ax_name))
            candidates.append(os.path.join(os.path.dirname(image_path), ax_name))
            for mfp in candidates:
                if os.path.exists(mfp):
                    try:
                        with open(mfp, "r", encoding="utf-8") as f:
                            mods = json.load(f)
                        break
                    except Exception as e:
                        logging.debug(f"Failed to read mods {mfp}: {e}")
        self._mods_cache_source = image_path
        self._mods_cache = mods
        return mods

    def _inspect_at_scene_point(self, scene_pt):
        """
        Exactly the same logic on left-click in inspection mode,
        but as a reusable function (returns True if it emitted, else False).
        """
        import numpy as np, cv2, logging, os

        img_data = getattr(self, "image_data", None)
        base_img = None
        if img_data is not None and getattr(img_data, "image", None) is not None:
            base_img = img_data.image
        if base_img is None or self._image is None or self._image.pixmap() is None:
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        # map scene -> pixmap local
        pixitem = self._image
        pixmap = pixitem.pixmap()
        px, py = pixitem.pos().x(), pixitem.pos().y()
        local_x = scene_pt.x() - px
        local_y = scene_pt.y() - py
        if not (0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height()):
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        # map pixmap local -> base image coords
        H0, W0 = base_img.shape[0], base_img.shape[1]
        scale_x = W0 / float(max(1, pixmap.width()))
        scale_y = H0 / float(max(1, pixmap.height()))
        x0 = int(local_x * scale_x)
        y0 = int(local_y * scale_y)
        if not (0 <= x0 < W0 and 0 <= y0 < H0):
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        # read & cache .ax
        image_path = getattr(img_data, "filepath", None) or getattr(self, "image_path", None)
        mods = self._load_ax_mods(image_path)
        expr = (mods.get("band_expression") or "").strip() if mods else ""

        # apply crop->resize on a working copy
        img_mod = base_img
        cx = cy = 0
        cw, ch = W0, H0

        rect = mods.get("crop_rect") or {}
        try:
            cx = int(rect.get("x", 0)); cy = int(rect.get("y", 0))
            cw = int(rect.get("width",  W0)); ch = int(rect.get("height", H0))
        except Exception:
            cx = cy = 0; cw = W0; ch = H0
        cx = max(0, min(cx, W0)); cy = max(0, min(cy, H0))
        cw = max(0, min(cw, W0 - cx)); ch = max(0, min(ch, H0 - cy))
        if cw > 0 and ch > 0:
            img_mod = img_mod[cy:cy+ch, cx:cx+cw]
        else:
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        info = mods.get("resize", None)
        if info:
            h1, w1 = img_mod.shape[:2]
            if "scale" in info:
                s = int(info["scale"])
                new_w = max(1, int(w1 * s / 100.0))
                new_h = max(1, int(h1 * s / 100.0))
            else:
                pct_w = int(info.get("width",  100))
                pct_h = int(info.get("height", 100))
                new_w = max(1, int(w1 * pct_w / 100.0))
                new_h = max(1, int(h1 * pct_h / 100.0))
            if new_w != w1 or new_h != h1:
                img_mod = resize_safe(img_mod, new_w, new_h, cv2.INTER_AREA)  # <-- changed


        # compute modified-image coords (x0,y0)->crop->resize
        x_c = x0 - cx; y_c = y0 - cy
        xm = x_c; ym = y_c
        if info:
            prev_w = cw if cw > 0 else img_mod.shape[1]
            prev_h = ch if ch > 0 else img_mod.shape[0]
            new_h, new_w = img_mod.shape[:2]
            if prev_w > 0 and prev_h > 0:
                xm = int(round(x_c * (new_w / float(prev_w))))
                ym = int(round(y_c * (new_h / float(prev_h))))

        Hm, Wm = img_mod.shape[0], img_mod.shape[1]
        if not (0 <= xm < Wm and 0 <= ym < Hm):
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        # build channel names, evaluate expression 
        if img_mod.ndim == 2:
            ch_names = ["b1"]
        else:
            ch_names = [f"b{i+1}" for i in range(img_mod.shape[2])]

        img_mod = img_mod.astype(np.float32, copy=False)
        if expr:
            try:
                if img_mod.ndim == 2:
                    mapping = {'b1': img_mod}
                else:
                    mapping = {f"b{i+1}": img_mod[:, :, i] for i in range(img_mod.shape[2])}
                code = compile(expr, "<expr>", "eval")
                for name in code.co_names:
                    if name not in mapping:
                        raise NameError(f"Use only {', '.join(mapping.keys())} in expression")
                idx_res = eval(code, {"__builtins__": {}}, mapping)
                if isinstance(idx_res, np.ndarray):
                    idx_res = np.nan_to_num(idx_res.astype(np.float32, copy=False),
                                            nan=0.0, posinf=0.0, neginf=0.0)
                    img_mod = np.dstack([img_mod, idx_res]) if img_mod.ndim == 3 else np.dstack([img_mod[..., None], idx_res])
                else:
                    idx_plane = np.full((Hm, Wm), float(idx_res), dtype=np.float32)
                    img_mod = np.dstack([img_mod, idx_plane]) if img_mod.ndim == 3 else np.dstack([img_mod[..., None], idx_plane])
                ch_names.append("index")
            except Exception as e:
                logging.debug(f"Index eval error at ({x0},{y0}) expr='{expr}': {e}")

        if img_mod.ndim == 2:
            vals = [float(img_mod[ym, xm])]
        else:
            C = img_mod.shape[2]
            vals = [float(img_mod[ym, xm, c]) for c in range(C)]

        payload = {"values": vals, "names": ch_names}
        try:
            self.pixel_clicked.emit(scene_pt, payload)
        except TypeError:
            self.pixel_clicked.emit(scene_pt, tuple(vals))
        return True


# helper inside ImageViewer
    def _scene_to_pix_local_int(self, p: QtCore.QPointF) -> QtCore.QPointF:
        """Map a scene point to *pixmap-local* integer pixel (top-left) using rounding."""
        if not self._image:
            return QtCore.QPointF(int(round(p.x())), int(round(p.y())))
        off = self._image.pos() 
        lx = int(round(p.x() - off.x()))
        ly = int(round(p.y() - off.y()))
        # clamp to pixmap bounds
        pm = self._image.pixmap()
        lx = max(0, min(lx, pm.width()  - 1))
        ly = max(0, min(ly, pm.height() - 1))
        return QtCore.QPointF(lx, ly)

    def get_current_scale_factor(self):
        """
        Calculates the current scale factor based on the view's transformation matrix.
        Assumes uniform scaling (scale_x == scale_y).
        """
        transform = self.transform()
        return math.sqrt(transform.m11() ** 2 + transform.m12() ** 2)

    def set_inspection_mode(self, enabled):
        """
        Enables or disables inspection mode.
        """
        self.inspection_mode = enabled
        if enabled:
            self.setCursor(QtCore.Qt.CrossCursor)  # Change cursor to crosshair
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)  # Disable panning
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)  # Revert cursor
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)  # Enable panning

    def has_image(self):
        return not self._empty

    def fit_to_window(self):
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.scale(1, 1)

 

    def set_image(self, pixmap):
        import sip
        if getattr(self, "_scene", None) is None or sip.isdeleted(self._scene):
            self._scene = QtWidgets.QGraphicsScene(self)
            self.setScene(self._scene)

        if getattr(self, "_image", None) is not None:
            try:
                if self._image.scene() is None:
                    self._image = None
            except RuntimeError:
                self._image = None

        self._zoom = 0
        self._empty = False
        try:
            self._scene.clear()
        except RuntimeError:
            self._scene = QtWidgets.QGraphicsScene(self)
            self.setScene(self._scene)

        self._image = self._scene.addPixmap(pixmap)
        self.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.fit_to_window()

        # Clear lingering polygon/point items; keep pixmap
        self.clear_polygons()

        # NOTE: polygons are now loaded explicitly by the tab after set_image()
        # to avoid relying on the viewer's immediate parent widget having load_polygons.



    def wheelEvent(self, event):
        if self.has_image():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fit_to_window()
            else:
                self._zoom = 0

            # Update temp_drawing_item's pen width if drawing
            if self.drawing and self.temp_drawing_item:
                scale = self.get_current_scale_factor()
                desired_screen_width = 2  # Desired pen width in screen pixels
                pen_width = desired_screen_width / scale
                pen = self.temp_drawing_item.pen()
                pen.setWidthF(pen_width)
                self.temp_drawing_item.setPen(pen)
        else:
            super(ImageViewer, self).wheelEvent(event)

    def mousePressEvent(self, event):
        # === INSPECTION MODE (left click) =========================================
        if self.inspection_mode and event.button() == QtCore.Qt.LeftButton:
            if self._inspect_at_scene_point(self.mapToScene(event.pos())):
                event.accept()
                return
            import os, json, logging
            import numpy as np
            import cv2

            point = self.mapToScene(event.pos())

            # Need a visible pixmap and numeric image to inspect
            img_data = getattr(self, "image_data", None)
            base_img = None
            if img_data is not None and getattr(img_data, "image", None) is not None:
                base_img = img_data.image

            if base_img is None or self._image is None or self._image.pixmap() is None:
                try:
                    self.pixel_clicked.emit(point, None)
                except TypeError:
                    self.pixel_clicked.emit(point, tuple())
                event.accept()
                return

            # --- Map scene -> pixmap local -> base image coords (uses pixmap's on-scene offset) ---
            pixitem = self._image
            pixmap = pixitem.pixmap()
            px, py = pixitem.pos().x(), pixitem.pos().y()
            local_x = point.x() - px
            local_y = point.y() - py
            if not (0 <= local_x < pixmap.width() and 0 <= local_y < pixmap.height()):
                try:
                    self.pixel_clicked.emit(point, None)
                except TypeError:
                    self.pixel_clicked.emit(point, tuple())
                event.accept()
                return

            H0, W0 = base_img.shape[0], base_img.shape[1]
            scale_x = W0 / float(max(1, pixmap.width()))
            scale_y = H0 / float(max(1, pixmap.height()))
            x0 = int(local_x * scale_x)
            y0 = int(local_y * scale_y)
            if not (0 <= x0 < W0 and 0 <= y0 < H0):
                try:
                    self.pixel_clicked.emit(point, None)
                except TypeError:
                    self.pixel_clicked.emit(point, tuple())
                event.accept()
                return

            # --- Read .ax (crop/resize/band_expression) for THIS image, if any ---
            image_path = getattr(img_data, "filepath", None) or getattr(self, "image_path", None)
            mods = {}
            expr = ""
            if image_path:
                ax_name = os.path.splitext(os.path.basename(image_path))[0] + ".ax"
                candidates = []
                parent = self.parent()
                proj_folder = getattr(parent, "project_folder", None) if parent else None
                if proj_folder:
                    candidates.append(os.path.join(proj_folder, ax_name))
                candidates.append(os.path.join(os.path.dirname(image_path), ax_name))
                for mfp in candidates:
                    if os.path.exists(mfp):
                        try:
                            with open(mfp, "r", encoding="utf-8") as f:
                                mods = json.load(f)
                            break
                        except Exception as e:
                            logging.debug(f"Failed to read mods {mfp}: {e}")
            expr = (mods.get("band_expression") or "").strip() if mods else ""

            # --- Apply crop->resize to a WORKING copy (float32), then evaluate index (float) ---
            img_mod = base_img
            cx = cy = 0
            cw, ch = W0, H0

            # crop
            rect = mods.get("crop_rect") or {}
            try:
                cx = int(rect.get("x", 0)); cy = int(rect.get("y", 0))
                cw = int(rect.get("width",  W0)); ch = int(rect.get("height", H0))
            except Exception:
                cx = cy = 0; cw = W0; ch = H0
            # clamp crop
            cx = max(0, min(cx, W0))
            cy = max(0, min(cy, H0))
            cw = max(0, min(cw, W0 - cx))
            ch = max(0, min(ch, H0 - cy))
            if cw > 0 and ch > 0:
                img_mod = img_mod[cy:cy+ch, cx:cx+cw]
            else:
                img_mod = None

            if img_mod is None or img_mod.size == 0:
                # crop removed everything -> nothing to show
                try:
                    self.pixel_clicked.emit(point, None)
                except TypeError:
                    self.pixel_clicked.emit(point, tuple())
                event.accept()
                return

            # resize
            info = mods.get("resize", None)
            if info:
                h1, w1 = img_mod.shape[:2]
                if "scale" in info:
                    s = int(info["scale"])
                    new_w = max(1, int(w1 * s / 100.0))
                    new_h = max(1, int(h1 * s / 100.0))
                else:
                    pct_w = int(info.get("width",  100))
                    pct_h = int(info.get("height", 100))
                    new_w = max(1, int(w1 * pct_w / 100.0))
                    new_h = max(1, int(h1 * pct_h / 100.0))
                if new_w != w1 or new_h != h1:
                    img_mod = cv2.resize(img_mod, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # scientific dtype
            img_mod = img_mod.astype(np.float32, copy=False)

            # --- Build channel names & (optionally) evaluate index on the modified stack ---
            # keep native order b1..bN for consistency with expressions
            if img_mod.ndim == 2:
                ch_names = ["b1"]
            else:
                ch_names = [f"b{i+1}" for i in range(img_mod.shape[2])]

            # compute modified-image coordinates of the clicked pixel: (x0,y0) -> crop -> resize
            x_c = x0 - cx
            y_c = y0 - cy
            xm = x_c
            ym = y_c
            if info:
                prev_w = cw if cw > 0 else img_mod.shape[1]
                prev_h = ch if ch > 0 else img_mod.shape[0]
                new_h, new_w = img_mod.shape[:2]
                if prev_w > 0 and prev_h > 0:
                    xm = int(round(x_c * (new_w / float(prev_w))))
                    ym = int(round(y_c * (new_h / float(prev_h))))

            # clamp to bounds
            Hm, Wm = img_mod.shape[0], img_mod.shape[1]
            if not (0 <= xm < Wm and 0 <= ym < Hm):
                try:
                    self.pixel_clicked.emit(point, None)
                except TypeError:
                    self.pixel_clicked.emit(point, tuple())
                event.accept()
                return

            # Evaluate index on the modified stack (full 2D then sample) so divisions are true floats
            if expr:
                try:
                    if img_mod.ndim == 2:
                        mapping = {'b1': img_mod}
                    else:
                        mapping = {f"b{i+1}": img_mod[:, :, i] for i in range(img_mod.shape[2])}
                    code = compile(expr, "<expr>", "eval")
                    for name in code.co_names:
                        if name not in mapping:
                            raise NameError(f"Use only {', '.join(mapping.keys())} in expression")
                    idx_res = eval(code, {"__builtins__": {}}, mapping)
                    if isinstance(idx_res, np.ndarray):
                        idx_res = np.nan_to_num(idx_res.astype(np.float32, copy=False),
                                                nan=0.0, posinf=0.0, neginf=0.0)
                        # append as last channel for consistent sampling
                        img_mod = np.dstack([img_mod, idx_res]) if img_mod.ndim == 3 else np.dstack([img_mod[..., None], idx_res])
                    else:
                        # scalar expression (unlikely) -> broadcast
                        idx_plane = np.full((Hm, Wm), float(idx_res), dtype=np.float32)
                        img_mod = np.dstack([img_mod, idx_plane]) if img_mod.ndim == 3 else np.dstack([img_mod[..., None], idx_plane])
                    ch_names.append("index")
                except Exception as e:
                    logging.debug(f"Index eval error at ({x0},{y0}) expr='{expr}': {e}")
                    # still proceed with base bands only

            # --- Sample *modified* stack at (xm,ym) and emit real values -----------------
            if img_mod.ndim == 2:
                vals = [float(img_mod[ym, xm])]
            else:
                C = img_mod.shape[2]
                vals = [float(img_mod[ym, xm, c]) for c in range(C)]

            payload = {"values": vals, "names": ch_names}
            try:
                self.pixel_clicked.emit(point, payload)
            except TypeError:
                self.pixel_clicked.emit(point, tuple(vals))

            event.accept()
            return

        # === DRAWING / PAN / CONTROLS =============================================
        if event.button() == QtCore.Qt.LeftButton and self.has_image():
            # Drawing mode (polygon or point) activated with Shift modifier or if already drawing
            if (event.modifiers() & QtCore.Qt.ShiftModifier) or self.drawing:
                if not self.drawing:
                    self.drawing = True
                    self.setDragMode(QtWidgets.QGraphicsView.NoDrag)  # Disable panning
                    point = self.mapToScene(event.pos())
                    self.currentPolygon = QtGui.QPolygonF()
                    self.currentPolygon.append(point)
                    self.lastPoint = point
                    # Create a temporary drawing item based on the drawing_mode
                    if self.drawing_mode == "polygon":
                        self.temp_drawing_item = QtWidgets.QGraphicsPolygonItem()
                        pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
                    else:  # "point" mode
                        self.temp_drawing_item = QtWidgets.QGraphicsPathItem()
                        pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
                    scale = self.get_current_scale_factor()
                    desired_screen_width = 2
                    pen_width = desired_screen_width / scale
                    pen.setWidthF(pen_width)
                    pen.setColor(QtCore.Qt.red)
                    self.temp_drawing_item.setPen(pen)
                    brush = QtGui.QBrush(QtCore.Qt.transparent)
                    self.temp_drawing_item.setBrush(brush)
                    self._scene.addItem(self.temp_drawing_item)
                    self.update_temp_drawing()
                    event.accept()
                else:
                    point = self.mapToScene(event.pos())
                    self.currentPolygon.append(point)
                    self.lastPoint = point
                    self.update_temp_drawing()
                    event.accept()
                self.left_button_pressed = True
            else:
                super(ImageViewer, self).mousePressEvent(event)
        elif event.button() == QtCore.Qt.MiddleButton and self.has_image():
            self.middle_button_pressed = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.last_pan_point = event.pos()
            event.accept()
        elif event.button() == QtCore.Qt.RightButton and self.drawing:
            self.finish_polygon()
            event.accept()
        else:
            super(ImageViewer, self).mousePressEvent(event)

    def get_pixel_value(self, x, y):
        try:
            if len(self.image_data.image.shape) == 3:
                b, g, r = self.image_data.image[y, x]
                return (r, g, b)
            elif len(self.image_data.image.shape) == 2:
                gray = self.image_data.image[y, x]
                return (gray,)
            else:
                return None
        except IndexError:
            return None


    def _scene_to_pix_local_int(self, p: QtCore.QPointF) -> QtCore.QPointF:
        """Scene point -> pixmap-local integer pixel (top-left)."""
        if not self._image:
            return QtCore.QPointF(int(math.floor(p.x())), int(math.floor(p.y())))
        lp = self._image.mapFromScene(p)            # robust against any pixmap offset/transform
        lx = int(math.floor(lp.x()))
        ly = int(math.floor(lp.y()))
        pm = self._image.pixmap()
        lx = max(0, min(lx, pm.width()  - 1))       # clamp
        ly = max(0, min(ly, pm.height() - 1))
        return QtCore.QPointF(lx, ly)
        
    
    def mouseMoveEvent(self, event):
        # --- ALWAYS-ON HOVER INSPECT (non-intrusive) ---
        if self.has_image():
            sp = self.mapToScene(event.pos())

            # quick bounds check in pixmap space to avoid work off-image
            pixitem = self._image
            pixmap = pixitem.pixmap()
            px, py = pixitem.pos().x(), pixitem.pos().y()
            lx = sp.x() - px; ly = sp.y() - py
            if 0 <= lx < pixmap.width() and 0 <= ly < pixmap.height():
                img_data = getattr(self, "image_data", None)
                base_img = getattr(img_data, "image", None) if img_data is not None else None
                if base_img is not None:
                    H0, W0 = base_img.shape[0], base_img.shape[1]
                    scale_x = W0 / float(max(1, pixmap.width()))
                    scale_y = H0 / float(max(1, pixmap.height()))
                    x0 = int(lx * scale_x); y0 = int(ly * scale_y)

                    # Only emit if  entered a new integer pixel; skip if panning/drawing buttons are down
                    if not (self.left_button_pressed or self.middle_button_pressed):
                        if self._last_hover_pixel != (x0, y0):
                            self._last_hover_pixel = (x0, y0)
                            # emit values (same mapping / .ax handling as clicks)
                            self._inspect_at_scene_point(sp)

        
        
        
        if self.middle_button_pressed and self.has_image():
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        elif self.drawing and self.has_image():
            point = self.mapToScene(event.pos())
            if self.left_button_pressed:
                min_distance = 15 / self.get_current_scale_factor()
                if self.lastPoint is None or QtCore.QLineF(self.lastPoint, point).length() > min_distance:
                    self.currentPolygon.append(point)
                    self.lastPoint = point
                    self.update_temp_drawing()
            else:
                if self.drawing_mode == "polygon":
                    tempPolygon = QtGui.QPolygonF(self.currentPolygon)
                    tempPolygon.append(point)
                    if isinstance(self.temp_drawing_item, QtWidgets.QGraphicsPolygonItem):
                        self.temp_drawing_item.setPolygon(tempPolygon)
                else:
                    tempPolygon = QtGui.QPolygonF(self.currentPolygon)
                    tempPolygon.append(point)
                    path = QtGui.QPainterPath()
                    scale = self.get_current_scale_factor()
                    radius = 5 / scale
                    for p in tempPolygon:
                        path.addEllipse(p, radius, radius)
                    if isinstance(self.temp_drawing_item, QtWidgets.QGraphicsPathItem):
                        self.temp_drawing_item.setPath(path)
            event.accept()
        else:
            super(ImageViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.left_button_pressed = False
        elif event.button() == QtCore.Qt.MiddleButton and self.middle_button_pressed:
            self.middle_button_pressed = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
        super(ImageViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drawing and self.has_image():
            self.finish_polygon()
            event.accept()
        else:
            super(ImageViewer, self).mouseDoubleClickEvent(event)

    def _find_project_owner(self):
        """
        Walk up parents to find the object that owns project_folder + all_polygons.
        Returns that owner or None.
        """
        w = self
        # Avoid sip.isdeleted surprises; stop if any parent disappears
        while w is not None:
            try:
                if hasattr(w, "project_folder") and hasattr(w, "all_polygons"):
                    return w
            except Exception:
                pass
            w = w.parent()
        return None


    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            selected = list(self._scene.selectedItems())
            if not selected:
                event.accept()
                return

            # We delete only the first selected item (keeps your original behavior)
            item = selected[0]
            if isinstance(item, (EditablePolygonItem, EditablePointItem)):
                group_name = getattr(item, "name", "") or ""
                fp = getattr(getattr(self, "image_data", None), "filepath", None)

                # 1) remove from scene + viewer memory
                try:
                    self._scene.removeItem(item)
                except Exception:
                    pass
                self.polygons = [p for p in self.polygons if p.get('item') is not item]

                # 2) remove JSON + parent map (robust owner lookup)
                owner = self._find_project_owner()
                if owner and fp and group_name:
                    try:
                        import os, logging
                        polygons_dir = os.path.join(owner.project_folder, "polygons")
                        base = os.path.splitext(os.path.basename(fp))[0]
                        json_path = os.path.join(polygons_dir, f"{group_name}_{base}_polygons.json")

                        # Delete file on disk (if present)
                        if os.path.exists(json_path):
                            os.remove(json_path)

                        # Prune in-memory map
                        ap = getattr(owner, "all_polygons", None)
                        if isinstance(ap, dict):
                            mapping = ap.get(group_name, {})
                            mapping.pop(fp, None)
                            if not mapping:
                                ap.pop(group_name, None)

                        # Persist + refresh UI
                        if hasattr(owner, "save_polygons_to_json"):
                            try:
                                owner.save_polygons_to_json()
                            except Exception:
                                pass
                        if hasattr(owner, "update_polygon_manager"):
                            try:
                                owner.update_polygon_manager()
                            except Exception:
                                pass
                    except Exception as e:
                        logging.error(f"Delete polygon JSON failed for '{group_name}': {e}")

                # 3) notify
                self.polygon_changed.emit()
                event.accept()
                return

            # Not a polygon/point item -> fall through
            event.accept()
        else:
            super(ImageViewer, self).keyPressEvent(event)

    def add_polygon_to_scene(self, polygon, name=""):
        is_rgb = False
        if hasattr(self, 'image_data') and self.image_data.image is not None:
            if len(self.image_data.image.shape) == 3 and self.image_data.image.shape[2] == 3:
                is_rgb = True
        pen_thickness = 3
        polygon_item = EditablePolygonItem(polygon, name, is_rgb)
        self._scene.addItem(polygon_item)
        polygon_item.polygon_modified.connect(self.on_polygon_modified)
        self.polygons.append({'polygon': polygon, 'name': name, 'item': polygon_item, 'type': 'polygon'})
        return polygon_item

    def add_point_to_scene(self, points, name=""):
        is_rgb = False
        if hasattr(self, 'image_data') and self.image_data is not None and getattr(self.image_data, "image", None) is not None:
            img = self.image_data.image
            if len(img.shape) == 3 and img.shape[2] == 3:
                is_rgb = True

        # Convert incoming scene points -> pixmap-local integer pixels (FLOOR)
        pix_local = QtGui.QPolygonF()
        for p in points:
            pix_local.append(self._scene_to_pix_local_int(p))

        point_item = EditablePointItem(
            pix_local, name, is_rgb,
            pixmap_item=self._image,
            points_are_pixmap_local=True
        )
        self._scene.addItem(point_item)
        point_item.point_modified.connect(self.on_polygon_modified)

        self.polygons.append({
            'points': points,          # keep original scene points for compatibility
            'points_pix': pix_local,   # exact integer pixmap pixels (use these for sampling/CSV)
            'name': name,
            'item': point_item,
            'type': 'point'
        })
        return point_item



    def finish_polygon(self):
        if self.drawing:
            self.drawing = False
            self.left_button_pressed = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            if self.temp_drawing_item:
                self._scene.removeItem(self.temp_drawing_item)
                self.temp_drawing_item = None
            group_name = self.pending_group_name if self.pending_group_name else ""
            if self.drawing_mode == "polygon":
                if len(self.currentPolygon) > 2:
                    polygon_item = self.add_polygon_to_scene(self.currentPolygon, group_name)
                    if not self.programmatically_adding_polygon:
                        self.polygon_drawn.emit(polygon_item)
            else:  # "point" mode
                if len(self.currentPolygon) >= 1:
                    point_item = self.add_point_to_scene(self.currentPolygon, group_name)
                    if not self.programmatically_adding_polygon:
                        self.polygon_drawn.emit(point_item)
            self.pending_group_name = None
            self.currentPolygon = QtGui.QPolygonF()
            if self.is_editing_group:
                self.editing_finished.emit()
                self.is_editing_group = False

    def update_temp_drawing(self):
        if not self.temp_drawing_item:
            return
        self.temp_drawing_item.setPos(0, 0)
        self.temp_drawing_item.setTransform(QtGui.QTransform())

        scale = self.get_current_scale_factor()
        if self.drawing_mode == "polygon":
            self.temp_drawing_item.setPolygon(self.currentPolygon)
            pen = self.temp_drawing_item.pen(); pen.setWidthF(2 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)
            return

        # point mode: draw 1x1 rects at exact pixmap pixels (FLOOR)
        path = QtGui.QPainterPath()
        off = self._image.pos() if self._image else QtCore.QPointF(0, 0)
        for p in self.currentPolygon:                 # scene points while drawing
            lp = self._image.mapFromScene(p)
            rx = int(math.floor(lp.x()))
            ry = int(math.floor(lp.y()))
            path.addRect(off.x() + rx, off.y() + ry, 1, 1)
        self.temp_drawing_item.setPath(path)

        pen = self.temp_drawing_item.pen(); pen.setWidthF(2 / max(1e-6, scale))
        self.temp_drawing_item.setPen(pen)


    def on_polygon_modified(self):
        self.polygon_changed.emit()

    def add_polygon(self, polygon, name=""):
        self.programmatically_adding_polygon = True
        polygon_item = self.add_polygon_to_scene(polygon, name)
        if name:
            self.polygons.append({'polygon': polygon, 'name': name, 'item': polygon_item, 'type': 'polygon'})
        self.programmatically_adding_polygon = False

    def get_all_polygons(self):
        return [item for item in self._scene.items() if isinstance(item, EditablePolygonItem) or isinstance(item, EditablePointItem)]

    def clear_polygons(self):
        for item in self.get_all_polygons():
            self._scene.removeItem(item)
        self.polygons = []
        self.currentPolygon = QtGui.QPolygonF()
        self.drawing = False
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.last_pan_point = QtCore.QPoint()
        self.pending_group_name = None

    def start_drawing_with_group_name(self, group_name):
        self.pending_group_name = group_name
        self.drawing = True
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setFocus()
        self.currentPolygon = QtGui.QPolygonF()
        self.lastPoint = None
        if self.drawing_mode == "polygon":
            self.temp_drawing_item = QtWidgets.QGraphicsPolygonItem()
            pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
        else:
            self.temp_drawing_item = QtWidgets.QGraphicsPathItem()
            pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
        scale = self.get_current_scale_factor()
        desired_screen_width = 2
        pen_width = desired_screen_width / scale
        pen.setWidthF(pen_width)
        pen.setColor(QtCore.Qt.red)
        self.temp_drawing_item.setPen(pen)
        brush = QtGui.QBrush(QtCore.Qt.transparent)
        self.temp_drawing_item.setBrush(brush)
        self._scene.addItem(self.temp_drawing_item)
        self.update_temp_drawing()
        logging.info(f"Started drawing new shape for group '{group_name}'.")

