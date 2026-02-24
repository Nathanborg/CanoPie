# image_viewer.py
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
        kwargs['handlers'] = [h for h in handlers if not isinstance(h, logging.FileHandler)]
    return _original_basicConfig(*args, **kwargs)
logging.basicConfig = _no_file_basicConfig

from .utils import *

# -------------------------------------------------------------------
# Small, movable on-image overlay button (hidden by default; shows on hover)
# -------------------------------------------------------------------
class OverlayButton(QtWidgets.QGraphicsObject):
    """
    Tiny on-image movable button for tools. Constant on-screen size, draggable,
    fades on hover, and emits clicked()/dragged(pos) signals.

    - Hidden by default. Caller should show it only when mouse is over the image.
    - ItemIgnoresTransformations keeps size constant while zooming.
    - Parent this item to the pixmap item so it moves with the image.
    """
    clicked = QtCore.pyqtSignal()
    dragged = QtCore.pyqtSignal(QtCore.QPointF)

    def __init__(self, size=28, tooltip="Tools", parent=None):
        super().__init__(parent)
        self.size = int(size)
        self.setToolTip(tooltip)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setZValue(1e7)

        self._hover = False
        self._press_pos = None
        self._moved = False

    def boundingRect(self):
        s = float(self.size)
        return QtCore.QRectF(0.0, 0.0, s, s)

    def paint(self, p, option, widget=None):
        r = self.boundingRect()
        # background (subtle, semi-transparent)
        base = QtGui.QColor(0, 0, 0, 90 if not self._hover else 150)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(base)
        p.drawRoundedRect(r, 8, 8)

        # icon (three dots)
        pen = QtGui.QPen(QtCore.Qt.white)
        pen.setWidthF(2.0)
        p.setPen(pen)
        cx = r.center().x()
        cy = r.center().y()
        d = 3.5
        for dx in (-8, 0, 8):
            p.drawEllipse(QtCore.QPointF(cx + dx, cy), d, d)

    # --- hover for subtle emphasis ---
    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(e)

    # --- drag/click handling ---
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_pos = e.pos()
            self._moved = False
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press_pos is not None:
            delta = e.pos() - self._press_pos
            if delta.manhattanLength() >= 2:
                self._moved = True
            self.setPos(self.pos() + delta)
            self.dragged.emit(self.pos())
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._press_pos is None:
            super().mouseReleaseEvent(e); return
        if not self._moved:
            self.clicked.emit()
        self._press_pos = None
        self._moved = False
        e.accept()


# -------------------------------------------------------------------
# Vertex Handle for polygon vertex editing
# -------------------------------------------------------------------
class VertexHandle(QtWidgets.QGraphicsObject):
    """
    A draggable handle for editing polygon vertices.
    Appears as a black square with red border, scales with zoom.
    """
    moved = QtCore.pyqtSignal(int, QtCore.QPointF)  # index, new_position
    
    def __init__(self, index, position, viewer, parent=None):
        super(VertexHandle, self).__init__(parent)
        self.index = index
        self.viewer = viewer
        self.setPos(position)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable |
            QtWidgets.QGraphicsItem.ItemIsSelectable |
            QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.SizeAllCursor)
        self.setZValue(1e7)  # On top of everything
        self._hover = False
        self._base_size = 12  # Base size in pixels
        
    def boundingRect(self):
        # Get current scale to make handle constant screen size
        size = self._get_screen_size()
        return QtCore.QRectF(-size/2, -size/2, size, size)
    
    def _get_screen_size(self):
        """Get the size in scene coordinates to appear as constant screen pixels."""
        try:
            views = self.scene().views() if self.scene() else []
            if views:
                view = views[0]
                # Get the current transform scale
                t = view.transform()
                scale = (t.m11()**2 + t.m12()**2) ** 0.5
                if scale > 0:
                    return self._base_size / scale
        except Exception:
            pass
        return self._base_size
    
    def paint(self, painter, option, widget=None):
        size = self._get_screen_size()
        half = size / 2
        rect = QtCore.QRectF(-half, -half, size, size)
        
        # Black fill with red border (or orange when hovered/selected)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        if self._hover or self.isSelected():
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 0)))  # Yellow/orange
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2 / self._get_scale()))
        else:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))  # Black
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2 / self._get_scale()))
        
        painter.drawRect(rect)
        
    def _get_scale(self):
        try:
            views = self.scene().views() if self.scene() else []
            if views:
                t = views[0].transform()
                return (t.m11()**2 + t.m12()**2) ** 0.5
        except Exception:
            pass
        return 1.0
    
    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)
    
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            # Notify viewer of the move
            try:
                if self.viewer and hasattr(self.viewer, 'on_vertex_moved'):
                    self.viewer.on_vertex_moved(self.index, value)
            except Exception as e:
                logging.debug(f"[VertexHandle] Error notifying viewer: {e}")
        return super().itemChange(change, value)
    
    def mousePressEvent(self, event):
        # PERFORMANCE: Notify viewer that vertex is being dragged
        if event.button() == QtCore.Qt.LeftButton:
            try:
                if self.viewer and hasattr(self.viewer, '_begin_item_drag'):
                    self.viewer._begin_item_drag()
                if self.viewer and hasattr(self.viewer, '_item_being_dragged'):
                    self.viewer._item_being_dragged = True
            except Exception:
                pass
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        # PERFORMANCE: Notify viewer that drag ended
        if event.button() == QtCore.Qt.LeftButton:
            try:
                if self.viewer and hasattr(self.viewer, '_end_item_drag'):
                    self.viewer._end_item_drag()
                if self.viewer and hasattr(self.viewer, '_item_being_dragged'):
                    self.viewer._item_being_dragged = False
            except Exception:
                pass
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        # Double-click to finish editing
        try:
            if self.viewer and hasattr(self.viewer, 'finish_vertex_editing'):
                self.viewer.finish_vertex_editing()
        except Exception as e:
            logging.debug(f"[VertexHandle] Error finishing vertex editing: {e}")
        event.accept()


# -------------------------------------------------------------------
# Editable overlay items
# -------------------------------------------------------------------
class EditablePolygonItem(QtWidgets.QGraphicsObject):
    polygon_modified = QtCore.pyqtSignal()

    def __init__(self, polygon, name="", is_rgb=False, parent=None, is_mask_polygon=False):
        super(EditablePolygonItem, self).__init__(parent)
        self.polygon = polygon
        self.name = name
        self.is_rgb = is_rgb  # Determines polygon appearance
        self.is_mask_polygon = is_mask_polygon  # If True, draw with solid fill

        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable |
            QtWidgets.QGraphicsItem.ItemIsMovable |
            QtWidgets.QGraphicsItem.ItemSendsGeometryChanges |
            QtWidgets.QGraphicsItem.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        # No caching - cache invalidation on zoom causes more overhead than it saves
        self.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
        self.is_moving = False
        self.show_label = True  # Default visible
        self._hover_showing_label = False # Temp visibility on hover
        self.label_offset = QtCore.QPointF(10, -10)
        
        # Cache for performance - avoid recalculating on every paint
        self._cached_img_size = None
        self._cached_res_boost = 1.0

    def _get_image_size(self):
        """Get image dimensions, cached to avoid scene iteration on every paint."""
        scn = self.scene()
        if not scn:
            return None, None
        
        # Return cached value if available
        if self._cached_img_size is not None:
            return self._cached_img_size
        
        # Find the pixmap item (usually just one)
        for it in scn.items():
            if isinstance(it, QtWidgets.QGraphicsPixmapItem):
                pm = it.pixmap()
                if not pm.isNull():
                    self._cached_img_size = (pm.width(), pm.height())
                    # Calculate res_boost once
                    long_side = max(self._cached_img_size)
                    self._cached_res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))
                    return self._cached_img_size
        return None, None

    def boundingRect(self):
        rect = self.polygon.boundingRect()
        # During active dragging, use tight bounding rect (no label area)
        # This prevents ghost labels since label isn't drawn during drag
        if self.is_moving:
            margin = 10
            rect.adjust(-margin, -margin, margin, margin)
            return rect
        # Normal case: include label area
        if self.name:
            # Make bounding rect large enough for label at any zoom level
            # Label can be quite large when zoomed out, so use generous estimate
            label_width = len(self.name) * 50 + 100  # generous estimate
            label_height = 250  # generous for large fonts when zoomed out
            label_rect = QtCore.QRectF(
                rect.topRight() + self.label_offset,
                QtCore.QSizeF(label_width, label_height)
            )
            rect = rect.united(label_rect)
        margin = 50  # larger margin to ensure clean repaints
        rect.adjust(-margin, -margin, margin, margin)
        return rect

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.polygon)
        return path

    def itemChange(self, change, value):
        """Notify scene of geometry changes to ensure proper repainting."""
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            # During active dragging, skip expensive scene updates - Qt handles basic redraw
            # The full cleanup happens in mouseReleaseEvent when dragging ends
            if not self.is_moving:
                self.prepareGeometryChange()
        elif change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            # Only do expensive cleanup when NOT actively dragging
            # During drag, Qt's default redraw is sufficient
            if not self.is_moving:
                scene = self.scene()
                if scene:
                    br = self.boundingRect()
                    scene_rect = self.mapRectToScene(br.adjusted(-200, -200, 200, 200))
                    scene.update(scene_rect)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        # --- current zoom scale ---
        t = painter.worldTransform()
        scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

        # ---- stroke width scaling ----
        desired_px = 2.0
        min_px, max_px = 1.0, 6.0
        width_scene = desired_px / scale
        width_scene = max(min_px / scale, min(max_px / scale, width_scene))

        # Determine colors based on mask polygon status
        if self.is_mask_polygon:
            # Mask polygons: orange/yellow with semi-transparent fill
            base_color = QtGui.QColor(255, 165, 0)  # Orange
            fill_color = QtGui.QColor(255, 165, 0, 80)  # Semi-transparent orange
        else:
            base_color = QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue
            fill_color = QtCore.Qt.transparent
        
        pen = QtGui.QPen(base_color)
        if self.isUnderMouse() or self.isSelected():
            pen.setColor(QtCore.Qt.magenta)
            if self.is_mask_polygon:
                fill_color = QtGui.QColor(255, 0, 255, 100)  # Magenta fill when selected
            width_scene *= 1.25
        pen.setWidthF(width_scene)

        painter.setPen(pen)
        painter.setBrush(fill_color)
        painter.drawPolygon(self.polygon)

        # ---- zoom + resolution aware label ----
        # Skip label during active dragging to prevent ghost marks
        should_show = getattr(self, "show_label", True) or getattr(self, "_hover_showing_label", False)
        if self.name and not self.is_moving and should_show:
            # Use cached image size
            img_w, img_h = self._get_image_size()
            res_boost = self._cached_res_boost

            base_px = 40 if self.is_rgb else 32
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            font = QtGui.QFont()
            font.setPixelSize(px)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtCore.Qt.red))

            # Simple positioning without extra boundingRect call
            bbox = self.polygon.boundingRect()
            pos = bbox.topRight() + self.label_offset

            painter.drawText(pos, self.name)

    def hoverEnterEvent(self, event):
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._hover_showing_label = True
        self.update()
        super(EditablePolygonItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(QtCore.Qt.ArrowCursor)
        self._hover_showing_label = False
        self.update()
        super(EditablePolygonItem, self).hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # IMPORTANT: call BEFORE is_moving changes boundingRect()
            self.prepareGeometryChange()
            
            self.is_moving = True
            self.setSelected(True)
            
            # Capture start positions for ALL selected items (Batch Move support)
            # This ensures that if we drag a group, we know where everyone started.
            try:
                scene = self.scene()
                if scene:
                    # We check type(self) to ensure we only capture our known polygon items
                    for item in scene.selectedItems():
                        if isinstance(item, type(self)):
                            item._undo_start_pos = item.pos()
            except Exception:
                pass
            
            # Ensure self is captured (fallback)
            self._undo_start_pos = self.pos()

            # PERFORMANCE: Notify viewer that item is being dragged
            # This skips expensive hover inspection during movement
            try:
                v = self.scene().views()[0]
                if v:
                    v.setFocus(QtCore.Qt.MouseFocusReason)
                    if hasattr(v, '_begin_item_drag'):
                        v._begin_item_drag()
                    if hasattr(v, '_item_being_dragged'):
                        v._item_being_dragged = True
            except Exception:
                pass
        super(EditablePolygonItem, self).mousePressEvent(event)


    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.is_moving:
                # IMPORTANT: call BEFORE is_moving changes boundingRect()
                self.prepareGeometryChange()

                self.is_moving = False

                # Update only the polygon's area, not the entire scene
                try:
                    scene = self.scene()
                    if scene:
                        # Use targeted update instead of full scene invalidation
                        br = self.boundingRect()
                        scene_rect = self.mapRectToScene(br.adjusted(-100, -100, 100, 100))
                        scene.invalidate(scene_rect, QtWidgets.QGraphicsScene.ItemLayer)
                        for view in scene.views():
                            # PERFORMANCE: Notify viewer that drag ended
                            if hasattr(view, '_end_item_drag'):
                                view._end_item_drag()
                            if hasattr(view, '_item_being_dragged'):
                                view._item_being_dragged = False
                except Exception:
                    pass

                # Check for undoable move (Batch Support)
                moved_via_undo = False
                try:
                    scene = self.scene()
                    v = scene.views()[0]
                    if hasattr(v, 'handle_undoable_move_batch'):
                        changes = []
                        # Iterate all selected items to find who moved
                        # This handles multi-selection drag
                        for item in scene.selectedItems():
                            if isinstance(item, type(self)) and hasattr(item, '_undo_start_pos'):
                                if item.pos() != item._undo_start_pos:
                                    changes.append((item, item._undo_start_pos, item.pos()))
                        
                        if changes:
                             if v.handle_undoable_move_batch(changes):
                                 moved_via_undo = True
                except Exception:
                    pass

                if not moved_via_undo:
                    self.polygon_modified.emit()
        super(EditablePolygonItem, self).mouseReleaseEvent(event)


    def switch_to_tab(self, tab_name):
        for index in range(self.tab_widget.count()):
            if self.tab_widget.tabText(index) == tab_name:
                self.tab_widget.setCurrentIndex(index)
                self.status.showMessage(f"Switched to {tab_name}", 2000)
                logging.info(f"Switched to tab: {tab_name}")
                return
        QtWidgets.QMessageBox.warning(self, "Tab Not Found", f"No tab named '{tab_name}' was found.")
        logging.warning(f"Attempted to switch to non-existent tab: {tab_name}")

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu()
        a_copy     = menu.addAction("Copy polygon")
        a_repl     = menu.addAction("Replicate to all viewers")
        a_edit     = menu.addAction("Edit this polygon")
        a_edit_all = menu.addAction("Edit all polygons in this group")
        menu.addSeparator()
        a_edit_vertices = menu.addAction("Edit vertices")
        menu.addSeparator()
        a_delete   = menu.addAction("Delete this polygon")
        a_delete_all = menu.addAction("Delete all polygons in this group")

        chosen = menu.exec_(QtGui.QCursor.pos())

        v = self.scene().views()[0] if self.scene().views() else None
        if not v:
            return

        if chosen == a_copy and hasattr(v, "copy_specific_items"):
            v.copy_specific_items([self])

        elif chosen == a_repl and hasattr(v, "replicate_toviewer"):
            selected = [it for it in self.scene().selectedItems()
                        if isinstance(it, (EditablePolygonItem, EditablePointItem))]
            v.replicate_toviewer(selected or [self])

        elif chosen == a_edit and hasattr(v, "edit_single_polygon"):
            v.edit_single_polygon(self, start_redraw=True)

        elif chosen == a_edit_all and hasattr(v, "edit_all_polygons_in_group"):
            v.edit_all_polygons_in_group(self, start_redraw=True, respect_sync=True)

        elif chosen == a_edit_vertices and hasattr(v, "start_vertex_editing"):
            v.start_vertex_editing(self)

        elif chosen == a_delete and hasattr(v, "delete_polygon_for_this_file"):
            v.delete_polygon_for_this_file(self)

        elif chosen == a_delete_all and hasattr(v, "delete_all_polygons_in_group"):
            v.delete_all_polygons_in_group(self)


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
        # No caching - simpler and avoids cache invalidation overhead
        self.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
        self.is_moving = False
        self.show_label = True  # Default visible
        self._hover_showing_label = False
        self.label_offset = QtCore.QPointF(10, -10)
        
        # Cache for res_boost calculation
        self._cached_res_boost = 1.0
        self._cached_img_size = None

    def _pixmap_pos(self):
        return self.pixmap_item.pos() if self.pixmap_item is not None else QtCore.QPointF(0, 0)

    def _scene_xy(self, p):
        """Return integer scene coords for a stored point `p`."""
        if self.points_are_pixmap_local:
            off = self._pixmap_pos()
            return int(off.x() + p.x()), int(off.y() + p.y())
        else:
            return int(p.x()), int(p.y())

    def _get_res_boost(self):
        """Get resolution boost factor, cached."""
        if self.pixmap_item is not None and self._cached_img_size is None:
            pm = self.pixmap_item.pixmap()
            if not pm.isNull():
                self._cached_img_size = (pm.width(), pm.height())
                long_side = max(self._cached_img_size)
                self._cached_res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))
        return self._cached_res_boost

    def boundingRect(self):
        if self.points.isEmpty():
            return QtCore.QRectF()

        xs, ys = [], []
        for p in self.points:
            sx, sy = self._scene_xy(p)
            xs.append(sx); ys.append(sy)
        rect = QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

        if rect.width() < 10 and rect.height() < 10:
            rect = rect.adjusted(-20, -20, 20, 20)

        # During active dragging, use tight bounding rect (no label area)
        # This prevents ghost labels since label isn't drawn during drag
        if self.is_moving:
            margin = 10
            rect.adjust(-margin, -margin, margin, margin)
            return rect

        # Add generous space for label at any zoom level
        if self.name:
            label_width = len(self.name) * 50 + 100
            label_height = 250
            label_rect = QtCore.QRectF(
                rect.topLeft() + self.label_offset,
                QtCore.QSizeF(label_width, label_height)
            )
            rect = rect.united(label_rect)

        margin = 50  # larger margin to ensure clean repaints
        rect.adjust(-margin, -margin, margin, margin)
        return rect

    def shape(self):
        path = QtGui.QPainterPath()
        if self.points.isEmpty():
            return path
        for p in self.points:
            sx, sy = self._scene_xy(p)
            path.addEllipse(sx, sy, 6, 6)
        return path

    def itemChange(self, change, value):
        """Notify scene of geometry changes to ensure proper repainting."""
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            # During active dragging, skip expensive scene updates - Qt handles basic redraw
            # The full cleanup happens in mouseReleaseEvent when dragging ends
            if not self.is_moving:
                self.prepareGeometryChange()
        elif change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            # Only do expensive cleanup when NOT actively dragging
            # During drag, Qt's default redraw is sufficient
            if not self.is_moving:
                scene = self.scene()
                if scene:
                    br = self.boundingRect()
                    scene_rect = self.mapRectToScene(br.adjusted(-200, -200, 200, 200))
                    scene.update(scene_rect)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        color = QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        for p in self.points:
            sx, sy = self._scene_xy(p)
            painter.drawRect(QtCore.QRectF(sx, sy, 1, 1))

        # Skip label during active dragging to prevent ghost marks
        should_show = getattr(self, "show_label", True) or getattr(self, "_hover_showing_label", False)
        if self.name and not self.is_moving and should_show:
            t = painter.worldTransform()
            scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

            res_boost = self._get_res_boost()

            base_px = 40 if self.is_rgb else 32
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            font = QtGui.QFont()
            font.setPixelSize(px)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtCore.Qt.red))

            # Simple positioning - get bounds once
            xs, ys = [], []
            for p in self.points:
                sx, sy = self._scene_xy(p)
                xs.append(sx); ys.append(sy)
            rx, ry = min(xs), min(ys)
            text_position = QtCore.QPointF(rx, ry) + self.label_offset

            painter.drawText(text_position, self.name)

    def hoverEnterEvent(self, event):
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._hover_showing_label = True
        self.update()
        super(EditablePointItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(QtCore.Qt.ArrowCursor)
        self._hover_showing_label = False
        self.update()
        super(EditablePointItem, self).hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # IMPORTANT: call BEFORE is_moving changes boundingRect()
            self.prepareGeometryChange()

            self.is_moving = True
            self.setSelected(True)

            # PERFORMANCE: Notify viewer that item is being dragged
            try:
                v = self.scene().views()[0]
                if v:
                    v.setFocus(QtCore.Qt.MouseFocusReason)
                    if hasattr(v, '_begin_item_drag'):
                        v._begin_item_drag()
                    if hasattr(v, '_item_being_dragged'):
                        v._item_being_dragged = True
            except Exception:
                pass
        super(EditablePointItem, self).mousePressEvent(event)


    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.is_moving:
                # IMPORTANT: call BEFORE is_moving changes boundingRect()
                self.prepareGeometryChange()

                self.is_moving = False

                # Update only the point item's area, not the entire scene
                try:
                    scene = self.scene()
                    if scene:
                        # Use targeted update instead of full scene invalidation
                        br = self.boundingRect()
                        scene_rect = self.mapRectToScene(br.adjusted(-100, -100, 100, 100))
                        scene.invalidate(scene_rect, QtWidgets.QGraphicsScene.ItemLayer)
                        for view in scene.views():
                            # PERFORMANCE: Notify viewer that drag ended
                            if hasattr(view, '_end_item_drag'):
                                view._end_item_drag()
                            if hasattr(view, '_item_being_dragged'):
                                view._item_being_dragged = False
                except Exception:
                    pass

                self.point_modified.emit()
        super(EditablePointItem, self).mouseReleaseEvent(event)


    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu()
        a_copy = menu.addAction("Copy points")
        a_repl = menu.addAction("Replicate to all viewers")
        menu.addSeparator()
        a_delete   = menu.addAction("Delete this polygon")
        a_delete_all = menu.addAction("Delete all polygons in this group")

        chosen = menu.exec_(QtGui.QCursor.pos())

        v = self.scene().views()[0] if self.scene().views() else None
        if not v:
            return

        if chosen == a_copy and hasattr(v, "copy_specific_items"):
            v.copy_specific_items([self])

        elif chosen == a_repl and hasattr(v, "replicate_toviewer"):
            selected = [it for it in self.scene().selectedItems()
                        if isinstance(it, (EditablePolygonItem, EditablePointItem))]
            v.replicate_toviewer(selected or [self])

        elif chosen == a_delete and hasattr(v, "delete_polygon_for_this_file"):
            v.delete_polygon_for_this_file(self)

        elif chosen == a_delete_all and hasattr(v, "delete_all_polygons_in_group"):
            v.delete_all_polygons_in_group(self)


# -------------------------------------------------------------------
# Main ImageViewer
# -------------------------------------------------------------------
class ImageViewer(QtWidgets.QGraphicsView):
    polygon_drawn = QtCore.pyqtSignal(object)
    polygon_changed = QtCore.pyqtSignal()
    editing_finished = QtCore.pyqtSignal()
    pixel_clicked = QtCore.pyqtSignal(QtCore.QPointF, object)
    editing_cancelled = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self._labels_visible = True
        self._zoom = 0
    

        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        # Use BSP tree for faster item lookup in large scenes
        self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)
        self.setScene(self._scene)
        self._image = None

        # === PERFORMANCE OPTIMIZATIONS FOR LARGE IMAGES ===
        # Use minimal viewport updates - only repaint what changed
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        # Don't expand update regions for antialiasing (saves significant work)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        # Skip painter state save/restore overhead
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        # Cache the background (the image) for faster repaints
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        # Disable antialiasing on the view for speed (polygons have their own)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)

        # --- Ghost-label fix during drag ---
        # MinimalViewportUpdate + cached background can leave text trails when moving many items.
        # We temporarily switch to FullViewportUpdate while any item is dragged, then restore.
        self._normal_viewport_update_mode = self.viewportUpdateMode()
        self._normal_cache_mode = self.cacheMode()
        self._drag_viewport_update_mode = QtWidgets.QGraphicsView.FullViewportUpdate
        self._drag_cache_mode = QtWidgets.QGraphicsView.CacheNone
        self._drag_active_count = 0

        # For drawing
        self.drawing = False
        self.currentPolygon = QtGui.QPolygonF()
        self.polygons = []
        self._rb_dragging = False
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)

        self.temp_drawing_item = None

        # Panning / focus
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Flags for drawing/interaction
        self.programmatically_adding_polygon = False
        self.pending_group_name = None
        self.is_editing_group = False
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self._is_panning = False  # Track active panning to skip expensive hover operations
        self.last_pan_point = QtCore.QPoint()
        self.inspection_mode = False

        # Original Image Data Reference
        self.image_data = None

        # Drawing mode: "polygon", "point", "rectangle", "circle", "random_points"
        self.drawing_mode = "polygon"
        self.setMouseTracking(True)
        self._last_hover_pixel = None
        self._mods_cache_source = None
        self._mods_cache = None
        self.preview_prefers_index = False
        self.sync_enabled = True
        self._sync_restore_slots = []
        self._local_edit_active = False
        self._last_pan_sync_time = 0
        
        # For rectangle/circle drawing
        self._shape_start_point = None
        self._shape_end_point = None
        
        # For vertex editing
        self._vertex_editing_item = None  # The polygon being edited
        self._vertex_handles = []  # List of VertexHandle objects
        self._vertex_resampled = False  # True if vertices were resampled for performance
        self._original_polygon = None  # Original polygon if resampled
        
        # PERFORMANCE: Track when any polygon/point item is being dragged
        # to skip expensive hover inspection operations during movement
        self._item_being_dragged = False
        
        # Polygon visibility state (respects PolygonManager's "Show Polys" checkbox)
        self._polygons_visible = True
        
        # PERFORMANCE: Cached values for huge images
        self._cached_pixmap_size = None  # (width, height) of current pixmap
        self._polygon_item_count = 0  # Number of polygon/point items for fast panning check
        self._hover_inspect_enabled = True  # Set to False to disable hover pixel inspection

        # For rectangle zoom mode (right-click drag to zoom to rectangle)
        self._rect_zoom_mode = False
        self._rect_zoom_start = None
        self._rect_zoom_item = None

        # --- Overlay tool button (DISABLED - use zoom bar instead) ---
        self.overlay_enabled = False         # Disabled: was interfering with drawing
        self._overlay_btn = None
        self._overlay_default_pos = QtCore.QPointF(14, 14)
        self._hover_hide_timer = QtCore.QTimer(self)
        self._hover_hide_timer.setInterval(600)
        self._hover_hide_timer.setSingleShot(True)
        self._hover_hide_timer.timeout.connect(lambda: self._set_overlay_visible(False, immediate=True))

        # --- Attach zoom bar (deferred to ensure viewport exists) ---
        self._zoombar = None
        QtCore.QTimer.singleShot(0, self._attach_zoom_bar_deferred)





    def _attach_zoom_bar_deferred(self):
        """Attach the zoom bar after the widget is fully initialized."""
        try:
            attach_zoom_bar(self)
        except Exception as e:
            logging.debug(f"[ImageViewer] Failed to attach zoom bar: {e}")

    # ---------- Drag repaint helpers ----------
    def _begin_item_drag(self):
        """Temporarily switch to a repaint mode that avoids label 'ink' trails."""
        try:
            self._drag_active_count = getattr(self, "_drag_active_count", 0) + 1
            if self._drag_active_count == 1:
                # Save current modes (in case caller changed them elsewhere)
                self._normal_viewport_update_mode = self.viewportUpdateMode()
                self._normal_cache_mode = self.cacheMode()
                self.setViewportUpdateMode(self._drag_viewport_update_mode)
                self.setCacheMode(self._drag_cache_mode)
                # Force a full redraw right away to clear any stale painted text
                scn = self.scene()
                if scn:
                    scn.invalidate(scn.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
        except Exception:
            pass

    def _end_item_drag(self):
        """Restore normal repaint mode after dragging ends."""
        try:
            cnt = getattr(self, "_drag_active_count", 0)
            self._drag_active_count = max(0, cnt - 1)
            if self._drag_active_count == 0:
                self.setViewportUpdateMode(getattr(self, "_normal_viewport_update_mode", QtWidgets.QGraphicsView.MinimalViewportUpdate))
                self.setCacheMode(getattr(self, "_normal_cache_mode", QtWidgets.QGraphicsView.CacheBackground))
                scn = self.scene()
                if scn:
                    scn.invalidate(scn.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
        except Exception:
            pass

    # ---------- Overlay helpers ----------
    def _ensure_overlay(self):
        """Create the overlay button as a child of the pixmap so it follows the image."""
        if not self.overlay_enabled or self._image is None:
            return
        if self._overlay_btn is None:
            self._overlay_btn = OverlayButton(size=28, tooltip="Tools", parent=None)
            self._overlay_btn.setParentItem(self._image)
            self._overlay_btn.setPos(self._overlay_default_pos)
            self._overlay_btn.setVisible(False)
            self._overlay_btn.clicked.connect(self._on_overlay_clicked)
            self._overlay_btn.dragged.connect(self._on_overlay_dragged)

    def _set_overlay_visible(self, visible: bool, *, immediate=False):
        if not self._overlay_btn:
            return
        if visible:
            self._overlay_btn.setVisible(True)
            self._overlay_btn.setOpacity(1.0)
            self._hover_hide_timer.stop()
        else:
            if immediate:
                self._overlay_btn.setVisible(False)
            else:
                self._hover_hide_timer.start()

    def _on_overlay_clicked(self):
        # Placeholder action: toggle inspection mode
        try:
            self.set_inspection_mode(not self.inspection_mode)
        except Exception:
            pass

    def _on_overlay_dragged(self, pos: QtCore.QPointF):
        # Clamp within the pixmap bounds so it never escapes the image.
        if not self._image:
            return
        pm = self._image.pixmap()
        w, h = pm.width(), pm.height()
        s = float(self._overlay_btn.size if self._overlay_btn else 28.0)
        x = max(0.0, min(pos.x(), w - s))
        y = max(0.0, min(pos.y(), h - s))
        if self._overlay_btn:
            self._overlay_btn.setPos(QtCore.QPointF(x, y))

    # ---------- General helpers ----------
    def update_pixmap_only(self, pixmap):
        """
        Replaces the current pixmap without clearing polygons or other scene items.
        This is used for visual updates like stretching, where the underlying geometry is the same.
        """
        if self._image is not None and pixmap is not None and not pixmap.isNull():
            self._image.setPixmap(pixmap)
            self.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.viewport().update()
        elif self._image is None:
            self.set_image(pixmap)

    def show_preview_array(self, arr_uint8):
        """
        Convenience: take a uint8 array (H,W), (H,W,1), (H,W,2), (H,W,3), or (H,W,4)
        and update pixmap in place.
        
        - 2D or (H,W,1): Grayscale
        - (H,W,2): Pad with zeros to make 3-channel RGB (e.g., Gray + Classification)
        - (H,W,3): RGB
        - (H,W,4): Convert BGRA to RGB (discard alpha) or use ARGB format
        """
        if arr_uint8 is None or arr_uint8.size == 0:
            return
        
        h, w = arr_uint8.shape[:2]
        
        if arr_uint8.ndim == 2:
            # Grayscale 2D
            fmt = QtGui.QImage.Format_Grayscale8
            arr = np.ascontiguousarray(arr_uint8)
            qimg = QtGui.QImage(arr.data, w, h, w, fmt)
        elif arr_uint8.ndim == 3:
            c = arr_uint8.shape[2]
            
            if c == 1:
                # Single channel 3D -> treat as grayscale
                arr = np.ascontiguousarray(arr_uint8[:, :, 0])
                fmt = QtGui.QImage.Format_Grayscale8
                qimg = QtGui.QImage(arr.data, w, h, w, fmt)
            elif c == 2:
                # 2-channel (e.g., Grayscale + Classification) -> pad to 3-channel
                zeros = np.zeros((h, w, 1), dtype=np.uint8)
                arr = np.concatenate((arr_uint8, zeros), axis=2)
                arr = np.ascontiguousarray(arr)
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            elif c == 3:
                # Standard 3-channel RGB
                arr = np.ascontiguousarray(arr_uint8)
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            elif c == 4:
                # 4-channel (e.g., RGB + Classification or BGRA) -> convert to RGB
                # Take first 3 channels as RGB
                arr = np.ascontiguousarray(arr_uint8[:, :, :3])
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            else:
                # More than 4 channels -> take first 3 for display
                arr = np.ascontiguousarray(arr_uint8[:, :, :3])
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        else:
            # Unexpected ndim, try to handle gracefully
            logging.warning(f"show_preview_array: unexpected array ndim={arr_uint8.ndim}")
            return
        
        self.update_pixmap_only(QtGui.QPixmap.fromImage(qimg))


    def get_selected_polygons(self):
        items = []
        for it in self._scene.selectedItems():
            if isinstance(it, EditablePolygonItem):
                items.append(it)
        return items

    def zoom_out_to_fit(self):
        """
        Snap to most zoomed-out (fit whole PIXMAP), recentre, and keep _zoom in sync.
        """
        if not self.has_image() or self._image is None or self._image.pixmap().isNull():
            return
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._deactivate_scene_padding()
        self.resetTransform()
        try:
            self.fitInView(self._image, QtCore.Qt.KeepAspectRatio)
        except Exception:
            r = self._image.mapToScene(self._image.boundingRect()).boundingRect()
            if r.isValid() and r.width() > 0 and r.height() > 0:
                self.fitInView(r, QtCore.Qt.KeepAspectRatio)
        try:
            self.centerOn(self._image)
        except Exception:
            pass
        self._zoom = 0

        if self.drawing and self.temp_drawing_item:
            scale = self.get_current_scale_factor()
            pen = self.temp_drawing_item.pen()
            pen.setWidthF(2.0 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)
        self.viewport().update()

    def _load_ax_mods(self, image_path):
        """Load .ax modifications with mtime-based cache invalidation."""
        if image_path:
            ax_name = os.path.splitext(os.path.basename(image_path))[0] + ".ax"
            candidates = []
            
            # CRITICAL FIX: Use _find_project_owner to walk up parent chain
            # self.parent() only returns the immediate container, not the ProjectTab
            owner = self._find_project_owner() if hasattr(self, '_find_project_owner') else None
            proj_folder = getattr(owner, "project_folder", None) if owner else None
            
            if proj_folder:
                candidates.append(os.path.join(proj_folder, ax_name))
            candidates.append(os.path.join(os.path.dirname(image_path), ax_name))
            
            # Find the actual .ax file path
            ax_path = None
            for mfp in candidates:
                if os.path.exists(mfp):
                    ax_path = mfp
                    break
            
            if ax_path:
                try:
                    mtime = os.path.getmtime(ax_path)
                    # Check if cache is still valid (same path AND same mtime)
                    cached_path = getattr(self, "_mods_cache_source", None)
                    cached_mtime = getattr(self, "_mods_cache_mtime", None)
                    if cached_path == image_path and cached_mtime == mtime:
                        return self._mods_cache or {}
                    
                    # Read fresh from disk
                    with open(ax_path, "r", encoding="utf-8") as f:
                        mods = json.load(f)
                    self._mods_cache_source = image_path
                    self._mods_cache_mtime = mtime
                    self._mods_cache = mods
                    logging.debug(f"[_load_ax_mods] Loaded fresh .ax from {ax_path}, nodata_values={mods.get('nodata_values')}")
                    return mods
                except Exception as e:
                    logging.debug(f"Failed to read mods {ax_path}: {e}")
        
        # No .ax file found - clear cache
        self._mods_cache_source = image_path
        self._mods_cache_mtime = None
        self._mods_cache = {}
        return {}

    def _inspect_at_scene_point(self, scene_pt):
        """
        Read pixel values from the viewer's image_data.image at the given scene point.
        
        IMPORTANT: image_data.image is ALREADY modified (crop/rotate/hist/resize applied).
        We should NOT apply these transforms again. We only need to:
        1. Map scene coordinates to image pixel coordinates
        2. Read pixel values directly
        3. Optionally compute band expression for index value
        """
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

        # map pixmap local -> image_data.image coords
        # image_data.image is ALREADY modified (crop/rotate/hist/resize applied)
        # so we map directly to it without offset adjustments
        H, W = base_img.shape[:2]
        scale_x = W / float(max(1, pixmap.width()))
        scale_y = H / float(max(1, pixmap.height()))
        xm = int(local_x * scale_x)
        ym = int(local_y * scale_y)
        if not (0 <= xm < W and 0 <= ym < H):
            try: self.pixel_clicked.emit(scene_pt, None)
            except TypeError: self.pixel_clicked.emit(scene_pt, tuple())
            return False

        # channel names
        if base_img.ndim == 2:
            ch_names = ["b1"]
        else:
            ch_names = [f"b{i+1}" for i in range(base_img.shape[2])]

        # Read pixel values directly from the already-modified image
        img_mod = base_img.astype(np.float32, copy=False)
        
        # Load NoData values from .ax file
        image_path = getattr(img_data, "filepath", None) or getattr(self, "image_path", None)
        mods = self._load_ax_mods(image_path)
        
        nodata_values = []
        try:
            nodata_enabled = mods.get("nodata_enabled", True) if mods else True
            if nodata_enabled:
                nodata_values = list(mods.get("nodata_values", []) or []) if mods else []
        except Exception:
            pass
        
        # DEBUG: Log NoData values being used
        if nodata_values:
            logging.info(f"[Inspector] NoData values from .ax: {nodata_values} for {os.path.basename(image_path) if image_path else 'None'}")
        else:
            logging.debug(f"[Inspector] No nodata_values found in mods for {image_path}. nodata_enabled={mods.get('nodata_enabled') if mods else 'N/A'}, mods keys: {list(mods.keys()) if mods else 'None'}")
        
        # Only apply band expression (to compute index value) - NOT crop/rotate/hist/resize
        expr = (mods.get("band_expression") or "").strip() if mods else ""
        band_enabled = mods.get("band_enabled", True) if mods else True
        
        if expr and band_enabled:
            try:
                if img_mod.ndim == 2:
                    mapping = {'b1': img_mod}
                else:
                    # CRITICAL FIX: image_data.image is BGR from cv2.imread, but user sees RGB
                    # in the viewer. When user types b1, they mean Red (what they see).
                    # BGR channel order: 0=Blue, 1=Green, 2=Red
                    # RGB mapping (what user sees): b1=Red, b2=Green, b3=Blue
                    # So for 3-channel images: b1→channel2(Red), b2→channel1(Green), b3→channel0(Blue)
                    C = img_mod.shape[2]
                    if C == 3:
                        # BGR→RGB semantic mapping
                        mapping = {
                            'b1': img_mod[:, :, 2],  # b1 = Red (user sees as channel 1)
                            'b2': img_mod[:, :, 1],  # b2 = Green (channel 2)
                            'b3': img_mod[:, :, 0],  # b3 = Blue (channel 3)
                        }
                    elif C > 3:
                        # For >3 channels (e.g., BGR + appended bands):
                        # First 3 are BGR, additional bands stay in order
                        mapping = {
                            'b1': img_mod[:, :, 2],  # Red
                            'b2': img_mod[:, :, 1],  # Green
                            'b3': img_mod[:, :, 0],  # Blue
                        }
                        for i in range(3, C):
                            mapping[f'b{i+1}'] = img_mod[:, :, i]
                    else:
                        # 1 or 2 channels - no remapping needed
                        mapping = {f"b{i+1}": img_mod[:, :, i] for i in range(C)}
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
                    idx_plane = np.full((H, W), float(idx_res), dtype=np.float32)
                    img_mod = np.dstack([img_mod, idx_plane]) if img_mod.ndim == 3 else np.dstack([img_mod[..., None], idx_plane])
                ch_names.append("index")
            except Exception as e:
                logging.debug(f"Index eval error at ({xm},{ym}) expr='{expr}': {e}")

        if img_mod.ndim == 2:
            vals = [float(img_mod[ym, xm])]
        else:
            C = img_mod.shape[2]
            # CRITICAL FIX: Extract values in RGB semantic order to match channel names
            # img_mod is BGR from cv2, but ch_names are ["b1", "b2", "b3", ...]
            # where b1=Red, b2=Green, b3=Blue semantically
            if C == 3:
                # BGR → show as RGB order: Red(2), Green(1), Blue(0)
                vals = [
                    float(img_mod[ym, xm, 2]),  # b1 = Red
                    float(img_mod[ym, xm, 1]),  # b2 = Green
                    float(img_mod[ym, xm, 0]),  # b3 = Blue
                ]
            elif C > 3:
                # First 3 are BGR, additional bands stay in order
                vals = [
                    float(img_mod[ym, xm, 2]),  # b1 = Red
                    float(img_mod[ym, xm, 1]),  # b2 = Green
                    float(img_mod[ym, xm, 0]),  # b3 = Blue
                ]
                for i in range(3, C):
                    vals.append(float(img_mod[ym, xm, i]))
            else:
                # 1 or 2 channels - no remapping
                vals = [float(img_mod[ym, xm, c]) for c in range(C)]

        # Check if this pixel is NoData using shared utility (consistent with process_polygon)
        # This ensures grayscale, stacked, and resized images are handled correctly
        is_nodata = False
        if nodata_values:
            try:
                from canopie.utils import build_nodata_mask as _shared_build_nodata_mask
                
                # Build NoData mask for the full image (or use cached if available)
                # For performance, we only evaluate the single pixel, not the whole image
                # We'll re-implement the pixel-level check using the same logic as build_nodata_mask
                import re
                _NODATA_EXPR_RE = re.compile(r'^([bB]\d+)\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)$')
                
                # Prepare the image for evaluation (consistent with build_nodata_mask)
                x = img_mod
                if x.ndim == 2:
                    C = 1
                else:
                    C = x.shape[2]
                
                # Channel mapping function (consistent with build_nodata_mask)
                def _get_channel_idx(band_num):
                    """Convert band number (1-based) to channel index."""
                    if C == 1:
                        return 0  # Single channel - all band references go to channel 0
                    if C == 2:
                        return band_num - 1 if band_num <= 2 else band_num - 1
                    # NOTE: vals is now in RGB semantic order (b1=Red, b2=Green, b3=Blue)
                    # after our BGR→RGB remapping above, so direct indexing works correctly
                    return band_num - 1
                
                for nd_val in nodata_values:
                    if isinstance(nd_val, str):
                        # Handle expression strings like "b1<50"
                        m = _NODATA_EXPR_RE.match(nd_val)
                        if m:
                            band_name, op, threshold = m.groups()
                            band_num = int(band_name[1:])  # b1 -> 1
                            ch_idx = _get_channel_idx(band_num)
                            
                            if ch_idx >= C:
                                logging.debug(f"[Inspector] NoData expression {nd_val}: band {band_num} exceeds image channels ({C})")
                                continue
                            
                            # Get the band value
                            if C == 1:
                                band_val = vals[0]  # Grayscale: always use the single channel
                            else:
                                band_val = vals[ch_idx] if ch_idx < len(vals) else None
                            
                            if band_val is None:
                                continue
                                
                            threshold_val = float(threshold)
                            
                            # Check threshold condition
                            if op == '<' and band_val < threshold_val:
                                is_nodata = True
                            elif op == '<=' and band_val <= threshold_val:
                                is_nodata = True
                            elif op == '>' and band_val > threshold_val:
                                is_nodata = True
                            elif op == '>=' and band_val >= threshold_val:
                                is_nodata = True
                            elif op == '==' and abs(band_val - threshold_val) < 1e-6:
                                is_nodata = True
                            elif op == '!=' and abs(band_val - threshold_val) >= 1e-6:
                                is_nodata = True
                            
                            if is_nodata:
                                logging.debug(f"[Inspector] NoData match: expression '{nd_val}' matched for b{band_num}(ch{ch_idx})={band_val}")
                                break
                    else:
                        # Handle numeric literal
                        try:
                            fv = float(nd_val)
                            abs_fv = abs(fv)
                            # Use appropriate tolerance based on value magnitude (consistent with build_nodata_mask)
                            if abs_fv > 1e+30:
                                tol = abs_fv * 0.01
                            elif abs_fv > 1e+10:
                                tol = abs_fv * 0.001
                            elif abs_fv > 100:
                                tol = abs_fv * 0.001
                            else:
                                tol = 0.01
                            for v in vals:
                                if abs(v - fv) < tol:
                                    is_nodata = True
                                    logging.debug(f"[Inspector] NoData match: pixel value {v} matches nodata {fv}")
                                    break
                                # Also check for NaN/Inf
                                if not math.isfinite(v):
                                    is_nodata = True
                                    logging.debug(f"[Inspector] NoData match: pixel value {v} is NaN/Inf")
                                    break
                            if is_nodata:
                                break
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                logging.debug(f"[Inspector] NoData check failed: {e}")

        payload = {"values": vals, "names": ch_names, "is_nodata": is_nodata}
        try:
            self.pixel_clicked.emit(scene_pt, payload)
        except TypeError:
            self.pixel_clicked.emit(scene_pt, tuple(vals))
        return True

    def _scene_to_pix_local_int(self, p: QtCore.QPointF) -> QtCore.QPointF:
        """Map a scene point to *pixmap-local* integer pixel using floor (pixel that contains the point)."""
        if not self._image:
            return QtCore.QPointF(int(math.floor(p.x())), int(math.floor(p.y())))
        off = self._image.pos()
        # Use floor: point at (5.9, 5.9) is inside pixel (5, 5), not (6, 6)
        lx = int(math.floor(p.x() - off.x()))
        ly = int(math.floor(p.y() - off.y()))
        pm = self._image.pixmap()
        lx = max(0, min(lx, pm.width()  - 1))
        ly = max(0, min(ly, pm.height() - 1))
        return QtCore.QPointF(lx, ly)

    def get_current_scale_factor(self):
        transform = self.transform()
        return math.sqrt(transform.m11() ** 2 + transform.m12() ** 2)

    def set_inspection_mode(self, enabled):
        self.inspection_mode = enabled
        if enabled:
            self.setCursor(QtCore.Qt.CrossCursor)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def set_rect_zoom_mode(self, enabled):
        """Enable/disable rectangle zoom mode (right-click drag to zoom)."""
        self._rect_zoom_mode = enabled
        # Keep normal arrow cursor - cross cursor only shown during drag
        self.setCursor(QtCore.Qt.ArrowCursor)
        if not enabled:
            # Clean up any pending rectangle zoom
            if self._rect_zoom_item is not None:
                try:
                    self._scene.removeItem(self._rect_zoom_item)
                except Exception:
                    pass
                self._rect_zoom_item = None
            self._rect_zoom_start = None

    def has_image(self):
        return not self._empty

    def fit_to_window(self):
        # Fit the IMAGE item to the view, ignoring the extra scene padding
        if self._image:
            self.fitInView(self._image, QtCore.Qt.KeepAspectRatio)
        else:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _get_image_rect(self):
        """Get the actual image bounding rect in scene coordinates."""
        if self._image is not None and not sip.isdeleted(self._image):
            return self._image.boundingRect()
        return QtCore.QRectF()

    _SCENE_PAD_FRACTION = 0.20

    def _activate_scene_padding(self):
        img_rect = self._get_image_rect()
        if not img_rect.isNull() and not img_rect.isEmpty():
            p = self._SCENE_PAD_FRACTION
            self.setSceneRect(img_rect.adjusted(
                -img_rect.width() * p, -img_rect.height() * p,
                 img_rect.width() * p,  img_rect.height() * p))

    def _deactivate_scene_padding(self):
        img_rect = self._get_image_rect()
        if not img_rect.isNull() and not img_rect.isEmpty():
            self.setSceneRect(img_rect)

    def set_image(self, pixmap, defer_fit=False):
        if getattr(self, "_scene", None) is None or sip.isdeleted(self._scene):
            self._scene = QtWidgets.QGraphicsScene(self)
            self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)
            self.setScene(self._scene)

        self._zoom = 0
        self._empty = False
        
        # Reset background cache before content update
        self.resetCachedContent()
        
        # PERFORMANCE: For standard drone images (>5MP), use NoIndex.
        # BSP tree overhead (construction + maintenance) is significant and unnecessary
        # for scenes with <1000 items. Standard usage here has 1 pixmap + ~50 polygons.
        # NoIndex is strictly faster for loading and interaction in this case.
        total_pixels = pixmap.width() * pixmap.height()
        if total_pixels > 5_000_000:  # 5 megapixels (was 50MP)
            self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
            self._hover_inspect_enabled = True # Keep enabled, just use NoIndex
        else:
            self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)
            self._hover_inspect_enabled = True

        # PERFORMANCE: Reuse existing pixmap item if possible instead of scene.clear()
        # scene.clear() destroys all items, which is expensive.
        # We manually clear polygons/misc items but keep the heavy pixmap item.
        self.clear_polygons()
        
        # Remove rect zoom item if exists
        if self._rect_zoom_item:
            self._remove_polygon_item_safely(self._rect_zoom_item)
            self._rect_zoom_item = None
            
        # Check if we can reuse the existing image item
        reuse_item = False
        if getattr(self, "_image", None) is not None:
            try:
                if not sip.isdeleted(self._image) and self._image.scene() is self._scene:
                    reuse_item = True
            except Exception:
                pass
        
        if reuse_item:
            # Reuse the item - just swap the pixmap data (much faster)
            self._image.setPixmap(pixmap)
            self._image.setPos(0, 0)
            self._image.setTransform(QtGui.QTransform())
            self._image.setVisible(True)
        else:
            # Fallback: full clear and create
            self._scene.clear()
            self._image = self._scene.addPixmap(pixmap)
            # Ensure overlay button is recreated since clear() destroyed it
            self._overlay_btn = None
        
        # PERFORMANCE: Cache pixmap dimensions to avoid repeated access
        self._cached_pixmap_size = (pixmap.width(), pixmap.height())

        # Use ALWAYS-ON PADDING to prevent oscillation
        # Set scene rect to image rect + % padding on all sides
        rect = QtCore.QRectF(pixmap.rect())
        pad = self._SCENE_PAD_FRACTION or 0.20
        self.setSceneRect(rect.adjusted(
            -rect.width() * pad, -rect.height() * pad,
             rect.width() * pad,  rect.height() * pad
        ))
        
        # Check if zoom should be fixed across roots
        if not defer_fit:
            # ALWAYS fit to window first (user requirement: "Always Keep Fit")
            self.fit_to_window()
            
            # If sync is enabled, UPDATE the fixed zoom state to match this new fit
            # This ensures we start with a full view, and other viewers will sync to THIS level
            if _ZoomBar._zoom_sync_enabled:
                 try:
                     cur = self.current_zoom_factor() if hasattr(self, "current_zoom_factor") else None
                     if cur:
                         _ZoomBar._fixed_zoom = cur
                         # Store center (0.5, 0.5 for fit)
                         _ZoomBar._fixed_center_norm = (0.5, 0.5)
                         
                         # Update our own zoom bar to show the correct level
                         zb = getattr(self, "_zoombar", None)
                         if zb:
                             zb._set_slider_from_zoom(cur)
                             zb._update_label(cur)
                 except Exception:
                     pass

        # Ensure overlay is ready (recreates if needed) and hidden
        self._ensure_overlay()
        self._set_overlay_visible(False, immediate=True)
    
    def _apply_fixed_zoom(self):
        """Apply the fixed zoom level and scroll position from _ZoomBar."""
        if _ZoomBar._fixed_zoom is None:
            return
        z = _ZoomBar._fixed_zoom
        center_norm = _ZoomBar._fixed_center_norm
        h_ratio = _ZoomBar._fixed_hscroll
        v_ratio = _ZoomBar._fixed_vscroll
        
        _ZoomBar._applying_fixed_zoom = True
        self._suppress_sync = True
        try:
            self.resetTransform()
            self.scale(z, z)
            
            # Prefer normalized center if available
            if center_norm:
                rx, ry = center_norm
                img_rect = self._get_image_rect()
                if not img_rect.isEmpty():
                    x = img_rect.left() + rx * img_rect.width()
                    y = img_rect.top() + ry * img_rect.height()
                    self.centerOn(x, y)
            elif h_ratio is not None and v_ratio is not None:
                try:
                    hs = self.horizontalScrollBar()
                    vs = self.verticalScrollBar()
                    h_range = hs.maximum() - hs.minimum()
                    v_range = vs.maximum() - vs.minimum()
                    if h_range > 0:
                        hs.setValue(hs.minimum() + int(h_ratio * h_range))
                    if v_range > 0:
                        vs.setValue(vs.minimum() + int(v_ratio * v_range))
                except Exception:
                    pass
            zb = getattr(self, "_zoombar", None)
            if zb:
                zb._set_slider_from_zoom(z)
                zb._update_label(z)
        finally:
            self._suppress_sync = False
            _ZoomBar._applying_fixed_zoom = False

    def reapply_fixed_zoom_if_enabled(self):
        """
        Re-apply fixed zoom after layout is complete.
        
        Logic:
        1. If Sync is enabled AND a fixed zoom exists (persistence): Apply it.
        2. If Sync is enabled but NO fixed zoom (initial load): Fit to Window, then set fixed zoom.
        3. If Sync is disabled: Fit to Window (standard behavior).
        """
        if _ZoomBar._zoom_sync_enabled:
            if _ZoomBar._fixed_zoom is not None:
                # PERSISTENCE: Restore the previous fixed zoom/center
                self._apply_fixed_zoom()
            else:
                # INITIAL LOAD: Fit to window, then adopt this as the new fixed state
                self.fit_to_window()
                try:
                    cur = self.current_zoom_factor() if hasattr(self, "current_zoom_factor") else None
                    if cur:
                        _ZoomBar._fixed_zoom = cur
                        _ZoomBar._fixed_center_norm = (0.5, 0.5)
                        
                        # Update our own zoom bar
                        zb = getattr(self, "_zoombar", None)
                        if zb:
                            zb._set_slider_from_zoom(cur)
                            zb._update_label(cur)
                        
                        # EXPLICITLY sync this new state to other viewers
                        # (since we disabled the implicit sync in attach_zoom_bar)
                        if zb:
                             zb._sync_zoom_to_all_viewers(cur)
                except Exception:
                    pass
        else:
             # Standard behavior when sync is off: Fit to Window
             self.fit_to_window()

    def wheelEvent(self, event):
        if not self._image:
            super(ImageViewer, self).wheelEvent(event)
            return
        
        # Prevent re-entrancy during zoom
        if getattr(self, '_zooming', False):
            event.accept()
            return
        
        # Throttle wheel events for performance on huge images
        import time
        current_time = time.time() * 1000  # ms
        last_wheel = getattr(self, '_last_wheel_time', 0)
        if current_time - last_wheel < 25:  # Max ~40fps for wheel zoom
            event.accept()
            return
        self._last_wheel_time = current_time
        
        self._zooming = True
        
        try:
            # Get current zoom factor directly from transform matrix
            tr = self.transform()
            current_zoom = tr.m11()
            
            # Calculate the minimum zoom (fit to window) - use CACHED dimensions
            vp = self.viewport()
            cached_size = getattr(self, '_cached_pixmap_size', None)
            if vp and cached_size:
                pw, ph = cached_size
                if pw > 0 and ph > 0:
                    fit_zoom = min(vp.width() / pw, vp.height() / ph) * 0.98
                else:
                    fit_zoom = 0.01
            else:
                fit_zoom = 0.01
            
            # Calculate new zoom factor
            if event.angleDelta().y() > 0:
                new_zoom = current_zoom * 1.25
            else:
                new_zoom = current_zoom * 0.8
            
            # Clamp zoom: min is fit_zoom, max is 50x
            new_zoom = max(fit_zoom, min(50.0, new_zoom))
            
            # Calculate relative scale required
            scale_factor = new_zoom / current_zoom
            
            # Update zoom counter based on whether we're above fit level
            if new_zoom > fit_zoom * 1.01:
                self._zoom = max(1, int((new_zoom / fit_zoom - 1) * 5))
            else:
                self._zoom = 0
            
            # PERFORMANCE: Use native Qt anchoring instead of manual calculation
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
            self.scale(scale_factor, scale_factor)
            
            # Restore anchor immediately so subsequent operations use ViewCenter
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            
            # (Padding logic removed: always-on padding handled in set_image)
            
            # Update fixed state for synced viewers
            if _ZoomBar._zoom_sync_enabled and not _ZoomBar._applying_fixed_zoom:
                _ZoomBar._fixed_zoom = new_zoom
                _ZoomBar._store_scroll_position(self)

            # Update temp drawing pen width if drawing
            if self.drawing and self.temp_drawing_item:
                pen = self.temp_drawing_item.pen()
                pen.setWidthF(2.0 / new_zoom)
                self.temp_drawing_item.setPen(pen)
            
            # Sync zoom to other viewers (debounced)
            zb = getattr(self, "_zoombar", None)
            if zb and _ZoomBar._zoom_sync_enabled and not _ZoomBar._syncing:
                if not hasattr(self, '_wheel_sync_timer'):
                    self._wheel_sync_timer = QtCore.QTimer(self)
                    self._wheel_sync_timer.setSingleShot(True)
                    self._wheel_sync_timer.setInterval(100)  # Reduced to 100ms
                    def _do_sync():
                        try:
                            cur = self.current_zoom_factor() if hasattr(self, "current_zoom_factor") else None
                            if cur and not _ZoomBar._syncing:
                                zb._sync_zoom_to_all_viewers(cur)
                        except Exception:
                            pass
                    self._wheel_sync_timer.timeout.connect(_do_sync)
                self._wheel_sync_timer.start()
            
            event.accept()
        finally:
            self._zooming = False

    def mousePressEvent(self, event):
        self.setFocus(QtCore.Qt.MouseFocusReason)

        # CTRL + Left: start rubber-band multi-selection (when not drawing/inspecting)
        if (event.button() == QtCore.Qt.LeftButton
                and not self.drawing
                and not self.inspection_mode
                and (event.modifiers() & QtCore.Qt.ControlModifier)):
            self._rb_dragging = True
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            super(ImageViewer, self).mousePressEvent(event)
            return

        # INSPECTION MODE (left click)
        if self.inspection_mode and event.button() == QtCore.Qt.LeftButton:
            if self._inspect_at_scene_point(self.mapToScene(event.pos())):
                event.accept()
                return
            # Fallback (should rarely trigger)
            try:
                self.pixel_clicked.emit(self.mapToScene(event.pos()), None)
            except TypeError:
                self.pixel_clicked.emit(self.mapToScene(event.pos()), tuple())
            event.accept()
            return

        # DRAWING / PAN / CONTROLS
        if event.button() == QtCore.Qt.LeftButton and self.has_image():
            if (event.modifiers() & QtCore.Qt.ShiftModifier) or self.drawing:
                if not self.drawing:
                    self.drawing = True
                    self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
                    point = self.mapToScene(event.pos())
                    self.currentPolygon = QtGui.QPolygonF()
                    self.currentPolygon.append(point)
                    self.lastPoint = point
                    
                    # Handle random points mode - prompt for count immediately
                    if self.drawing_mode == "random_points":
                        self._generate_random_points()
                        self.drawing = False
                        event.accept()
                        return
                    
                    # For rectangle/circle, store start point
                    if self.drawing_mode in ("rectangle", "circle"):
                        self._shape_start_point = point
                        self._shape_end_point = point
                    
                    if self.drawing_mode in ("polygon", "rectangle", "circle"):
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
                    event.accept()
                else:
                    point = self.mapToScene(event.pos())
                    
                    # For rectangle/circle, a second click finishes the shape
                    if self.drawing_mode in ("rectangle", "circle"):
                        self._shape_end_point = point
                        self.update_temp_drawing()
                        self.finish_polygon()
                        event.accept()
                        return
                    
                    self.currentPolygon.append(point)
                    self.lastPoint = point
                    self.update_temp_drawing()
                    event.accept()
                self.left_button_pressed = True
            else:
                # Left-click: check if clicking on background (for panning) or item (for selection)
                # PERFORMANCE: Fast path - if we have few polygon items, use simpler detection
                polygon_count = getattr(self, '_polygon_item_count', 0)
                
                if polygon_count == 0:
                    # No polygons - always start panning (no need for items lookup)
                    self._is_panning = True
                    self.last_pan_point = event.pos()
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                    event.accept()
                else:
                    # Have polygons - need to check if clicking on one
                    scene_pos = self.mapToScene(event.pos())
                    items_at_pos = self._scene.items(scene_pos)
                    # Filter out the pixmap - we only care about interactive items like polygons
                    clickable_items = [it for it in items_at_pos 
                                       if it != self._image and not isinstance(it, QtWidgets.QGraphicsPixmapItem)]
                    
                    if clickable_items:
                        # Clicked on an item - let Qt handle selection/dragging normally
                        super(ImageViewer, self).mousePressEvent(event)
                    else:
                        # Clicked on background - use fast manual panning
                        self._is_panning = True
                        self.last_pan_point = event.pos()
                        self.setCursor(QtCore.Qt.ClosedHandCursor)
                        event.accept()

        elif event.button() == QtCore.Qt.MiddleButton and self.has_image():
            self.middle_button_pressed = True
            self._is_panning = True  # Middle-button panning
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.last_pan_point = event.pos()
            event.accept()

        elif event.button() == QtCore.Qt.RightButton:
            if self.drawing:
                self.finish_polygon()
                event.accept()
            elif self._rect_zoom_mode and self.has_image():
                # Check if clicking on a polygon/point item - if so, show context menu instead
                scene_pos = self.mapToScene(event.pos())
                items_at_pos = self._scene.items(scene_pos)
                polygon_items = [it for it in items_at_pos 
                                 if isinstance(it, (EditablePolygonItem, EditablePointItem))]
                
                if polygon_items:
                    # Clicked on a polygon/point - let context menu show
                    super(ImageViewer, self).mousePressEvent(event)
                else:
                    # Start rectangle zoom selection
                    self._rect_zoom_start = self.mapToScene(event.pos())
                    # Create temporary rectangle item
                    self._rect_zoom_item = QtWidgets.QGraphicsRectItem()
                    pen = QtGui.QPen(QtGui.QColor(0, 120, 215), 2, QtCore.Qt.DashLine)
                    pen.setCosmetic(True)  # Constant screen width
                    self._rect_zoom_item.setPen(pen)
                    brush = QtGui.QBrush(QtGui.QColor(0, 120, 215, 40))
                    self._rect_zoom_item.setBrush(brush)
                    self._scene.addItem(self._rect_zoom_item)
                    self.setCursor(QtCore.Qt.CrossCursor)
                    event.accept()
            else:
                super(ImageViewer, self).mousePressEvent(event)

        else:
            super(ImageViewer, self).mousePressEvent(event)

    def get_pixel_value(self, x, y):
        try:
            if self.image_data is None or self.image_data.image is None:
                return None
            if len(self.image_data.image.shape) == 3:
                b, g, r = self.image_data.image[y, x]
                return (r, g, b)
            elif len(self.image_data.image.shape) == 2:
                gray = self.image_data.image[y, x]
                return (gray,)
            else:
                return None
        except (IndexError, AttributeError):
            return None

    def mouseMoveEvent(self, event):
        # --- FAST PATH: Skip all expensive operations during active panning ---
        if self._is_panning:
            # Use fast manual scrollbar updates for both left and middle button panning
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            
            # Sync to other viewers if enabled (throttled)
            if _ZoomBar._zoom_sync_enabled:
                 current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
                 last_sync = getattr(self, '_last_pan_sync_time', 0)
                 if current_time - last_sync > 50:  # Max 20fps sync
                     _ZoomBar.update_fixed_center(self)
                     self._last_pan_sync_time = current_time
                     
            event.accept()
            return
        
        # --- FAST PATH: Skip expensive operations when dragging a polygon/point item ---
        if getattr(self, '_item_being_dragged', False):
            # Let Qt handle the item drag, skip all hover inspection
            super(ImageViewer, self).mouseMoveEvent(event)
            return
        
        # --- FAST PATH: Skip hover inspection during active drawing ---
        # Drawing mode already handles its own preview updates
        if self.drawing and self.has_image():
            point = self.mapToScene(event.pos())
            if self.left_button_pressed:
                # For rectangle/circle, update end point during drag
                if self.drawing_mode in ("rectangle", "circle"):
                    self._shape_end_point = point
                    self.update_temp_drawing()
                else:
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
                elif self.drawing_mode in ("rectangle", "circle"):
                    # Preview shape while hovering (after first click)
                    if self._shape_start_point is not None:
                        self._shape_end_point = point
                        self.update_temp_drawing()
                else:
                    # For point mode, snap to pixel coordinates using floor
                    path = QtGui.QPainterPath()
                    off = self._image.pos() if self._image else QtCore.QPointF(0, 0)
                    
                    # Draw already-clicked points as pixel rects
                    for p in self.currentPolygon:
                        lp = self._image.mapFromScene(p)
                        rx = int(math.floor(lp.x()))
                        ry = int(math.floor(lp.y()))
                        path.addRect(off.x() + rx, off.y() + ry, 1, 1)
                    
                    # Draw hover point as pixel rect too
                    lp = self._image.mapFromScene(point)
                    rx = int(math.floor(lp.x()))
                    ry = int(math.floor(lp.y()))
                    path.addRect(off.x() + rx, off.y() + ry, 1, 1)
                    
                    if isinstance(self.temp_drawing_item, QtWidgets.QGraphicsPathItem):
                        self.temp_drawing_item.setPath(path)
            event.accept()
            return
        
        # --- Rectangle zoom drag ---
        if self._rect_zoom_start is not None and self._rect_zoom_item is not None:
            current_pos = self.mapToScene(event.pos())
            rect = QtCore.QRectF(self._rect_zoom_start, current_pos).normalized()
            self._rect_zoom_item.setRect(rect)
            event.accept()
            return
        
        # --- Hover inspect & overlay show/hide (only when NOT in any active mode) ---
        # PERFORMANCE: Skip hover inspection entirely for huge images unless explicitly enabled
        cached_size = getattr(self, '_cached_pixmap_size', None)
        hover_enabled = getattr(self, '_hover_inspect_enabled', True)
        if self.has_image() and cached_size:
            sp = self.mapToScene(event.pos())

            pixitem = self._image
            px, py = pixitem.pos().x(), pixitem.pos().y()
            lx = sp.x() - px; ly = sp.y() - py
            pw, ph = cached_size
            if 0 <= lx < pw and 0 <= ly < ph:
                # show overlay while mouse is over the image
                if self.overlay_enabled:
                    self._set_overlay_visible(True)

                # PERFORMANCE: Only do expensive inspect if enabled and not pressing buttons
                if hover_enabled and not (self.left_button_pressed or self.middle_button_pressed):
                    img_data = getattr(self, "image_data", None)
                    base_img = getattr(img_data, "image", None) if img_data is not None else None
                    if base_img is not None:
                        H0, W0 = base_img.shape[0], base_img.shape[1]
                        scale_x = W0 / float(max(1, pw))
                        scale_y = H0 / float(max(1, ph))
                        x0 = int(lx * scale_x); y0 = int(ly * scale_y)
                        if self._last_hover_pixel != (x0, y0):
                            self._last_hover_pixel = (x0, y0)
                            self._inspect_at_scene_point(sp)
            else:
                # schedule hide after leaving image bounds
                self._hover_hide_timer.start()

        super(ImageViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rb_dragging and event.button() == QtCore.Qt.LeftButton:
            self._rb_dragging = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            event.accept()
            return

        # Rectangle zoom completion
        if event.button() == QtCore.Qt.RightButton and self._rect_zoom_start is not None:
            if self._rect_zoom_item is not None:
                rect = self._rect_zoom_item.rect()
                # Remove the temporary rectangle
                self._scene.removeItem(self._rect_zoom_item)
                self._rect_zoom_item = None
                self._rect_zoom_start = None
                self.setCursor(QtCore.Qt.ArrowCursor)
                
                # Only zoom if rectangle is large enough (at least 10x10 pixels in scene)
                if rect.width() > 10 and rect.height() > 10:
                    self._activate_scene_padding()
                    # Zoom to fit the selected rectangle
                    self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
                    self._zoom = max(1, self._zoom)  # Ensure zoom counter is positive
                    
                    # Get the current zoom factor after fitInView
                    cur_zoom = None
                    zb = getattr(self, "_zoombar", None)
                    if zb:
                        try:
                            cur_zoom = self.current_zoom_factor() if hasattr(self, "current_zoom_factor") else None
                            if cur_zoom:
                                zb._block = True
                                zb._set_slider_from_zoom(cur_zoom)
                                zb._update_label(cur_zoom)
                                zb._block = False
                        except Exception:
                            pass
                    
                    # Sync to all viewers if zoom sync is enabled
                    if _ZoomBar._zoom_sync_enabled and zb and not _ZoomBar._applying_fixed_zoom:
                        try:
                            if cur_zoom:
                                _ZoomBar._fixed_zoom = cur_zoom
                                _ZoomBar._store_scroll_position(self)
                                
                                # Sync to all other viewers
                                if not _ZoomBar._syncing:
                                    zb._sync_zoom_to_all_viewers(cur_zoom)
                        except Exception:
                            pass
                
                event.accept()
                return

        if event.button() == QtCore.Qt.LeftButton:
            self.left_button_pressed = False
            if self._is_panning:
                self._is_panning = False  # End panning
                self.setCursor(QtCore.Qt.ArrowCursor)
            
            # Finish rectangle/circle on mouse release (drag-to-draw)
            if self.drawing and self.drawing_mode in ("rectangle", "circle"):
                point = self.mapToScene(event.pos())
                self._shape_end_point = point
                self.finish_polygon()
                event.accept()
                return
            
            # Update fixed pan position after drag
            if _ZoomBar._zoom_sync_enabled and not _ZoomBar._applying_fixed_zoom:
                _ZoomBar._store_scroll_position(self)
                _ZoomBar.update_fixed_center(self)

        elif event.button() == QtCore.Qt.MiddleButton and self.middle_button_pressed:
            self.middle_button_pressed = False
            self._is_panning = False  # End panning
            self.setCursor(QtCore.Qt.ArrowCursor)
            # Update fixed pan position after middle-button drag
            if _ZoomBar._zoom_sync_enabled and not _ZoomBar._applying_fixed_zoom:
                _ZoomBar._store_scroll_position(self)
                _ZoomBar.update_fixed_center(self)
            event.accept()

        super(ImageViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drawing and self.has_image():
            self.finish_polygon()
            event.accept()
        else:
            super(ImageViewer, self).mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        # hide overlay immediately when cursor leaves the view
        self._set_overlay_visible(False, immediate=True)
        super(ImageViewer, self).leaveEvent(event)

    def _find_project_owner(self):
        """
        Walk up parents to find the object that owns project_folder + all_polygons.
        """
        w = self
        while w is not None:
            try:
                if hasattr(w, "project_folder") and hasattr(w, "all_polygons"):
                    return w
            except Exception:
                pass
            w = w.parent()
        return None

    def keyPressEvent(self, event):
        k = event.key()
        t = event.text() or ""

        # --- Escape: Cancel drawing OR finish vertex editing ---
        if k == QtCore.Qt.Key_Escape:
            if self._vertex_editing_item:
                self.finish_vertex_editing()
                event.accept()
                return
            elif self.drawing:
                # Cancel current drawing
                self.drawing = False
                self.left_button_pressed = False
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
                if self.temp_drawing_item:
                    try:
                        self._scene.removeItem(self.temp_drawing_item)
                    except Exception:
                        pass
                    self.temp_drawing_item = None
                self.currentPolygon = QtGui.QPolygonF()
                self._shape_start_point = None
                self._shape_end_point = None
                event.accept()
                return
        
        # --- Enter/Return: Finish vertex editing ---
        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self._vertex_editing_item:
                self.finish_vertex_editing()
                event.accept()
                return

        MINUS_TEXTS = ("-", "_", "−", "–", "—")
        PLUS_TEXTS  = ("+",)
        is_plus  = (k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal)) or (t in PLUS_TEXTS)
        is_minus = (k in (QtCore.Qt.Key_Minus, getattr(QtCore.Qt, "Key_Underscore", QtCore.Qt.Key_Minus))) or (t in MINUS_TEXTS)
        if not is_minus and hasattr(QtCore.Qt, "Key_Subtract") and k == QtCore.Qt.Key_Subtract:
            is_minus = True

        def _has_viewer_poly_selection():
            try:
                return bool(self.get_selected_polygons())
            except Exception:
                return False

        def _resize(delta_sign):
            owner = self._find_project_owner()
            if owner and hasattr(owner, "resize_selected_polygons"):
                step = getattr(owner, "polygon_resize_step", 0.07)
                try:
                    owner.resize_selected_polygons(delta_sign * float(step))
                except Exception:
                    pass

        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            event.accept(); return
        if event.matches(QKeySequence.Paste):
            self.paste_geometry()
            event.accept(); return

        # --- Z key: zoom out to fit (no modifiers) ---
        if k == QtCore.Qt.Key_Z and not (event.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)):
            try:
                self.zoom_out_to_fit()
                # Sync and show zoom bar briefly
                if getattr(self, "_zoombar", None):
                    cur = self.current_zoom_factor() if hasattr(self, "current_zoom_factor") else 1.0
                    self._zoombar.set_zoom(cur)
                    self._zoombar.show_briefly()
            except Exception as e:
                logging.debug(f"[ImageViewer] Z key zoom failed: {e}")
            event.accept()
            return

        if is_plus and _has_viewer_poly_selection():
            _resize(+1.0); event.accept(); return
        if is_minus and _has_viewer_poly_selection():
            _resize(-1.0); event.accept(); return

        # Delete/Backspace: delete ONLY the selected item's group for THIS file
        if k in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            selected = list(self._scene.selectedItems())
            if not selected:
                event.accept(); return
            item = selected[0]

            try:
                valid_item = isinstance(item, (EditablePolygonItem, EditablePointItem))
            except Exception:
                valid_item = hasattr(item, "name")
            if not valid_item:
                event.accept(); return
            
            # Delegate deletion to ProjectTab (Owner) for Undo/Redo support
            logging.info("[ImageViewer] Delete Key Event. Finding owner...")
            owner = self._find_project_owner()
            logging.info(f"[ImageViewer] Owner: {owner}")
            fp_view = getattr(getattr(self, "image_data", None), "filepath", None)
            
            # Attempt Undo-Capable Deletion
            if owner and hasattr(owner, "delete_polygon_command") and fp_view:
                 try:
                     logging.info(f"[ImageViewer] Invoking owner.delete_polygon_command for {fp_view}")
                     owner.delete_polygon_command(item, fp_view)
                     event.accept()
                     return
                 except Exception as e:
                     logging.error(f"[ImageViewer] Undo-command deletion failed: {e}")
            else:
                 logging.warning(f"[ImageViewer] Cannot use Command. Owner={owner}, HasCmd={hasattr(owner, 'delete_polygon_command') if owner else 'N/A'}, FP={fp_view}")
            
            # FALLBACK: Legacy Destructive Deletion
            # Used if owner not found, command missing, or command failed.
            logging.warning("[ImageViewer] Using legacy destructive deletion.")
            item_label = (getattr(item, "name", "") or "").strip()
            
            if not item_label or not fp_view:
                self._remove_polygon_item_safely(item)
                event.accept(); return

            if not owner:
                self._remove_polygon_item_safely(item)
                event.accept(); return

            def _norm(p):
                try:    return os.path.normcase(os.path.abspath(p or ""))
                except: return p or ""

            ap = getattr(owner, "all_polygons", {}) or {}
            candidates = [g for g, m in ap.items() if isinstance(m, dict) and any(_norm(k) == _norm(fp_view) for k in m)]
            polygons_dir = (os.path.join(owner.project_folder, "polygons")
                            if getattr(owner, "project_folder", None)
                            else os.path.join(os.getcwd(), "polygons"))
            base_view = os.path.splitext(os.path.basename(fp_view))[0]

            canonical_group = None
            for g in candidates:
                if (g or "").strip().lower() == item_label.lower():
                    canonical_group = g; break
            if not canonical_group:
                for g in candidates:
                    jp = os.path.join(polygons_dir, f"{g}_{base_view}_polygons.json")
                    try:
                        if os.path.exists(jp):
                            with open(jp, "r", encoding="utf-8") as f:
                                js = json.load(f)
                            if (js.get("name", "") or "").strip().lower() == item_label.lower():
                                canonical_group = g; break
                    except Exception:
                        pass
            if not canonical_group:
                canonical_group = candidates[0] if len(candidates) == 1 else item_label

            gmap = ap.get(canonical_group)
            if not isinstance(gmap, dict):
                gmap = {}
            stored_key = None
            for k2 in list(gmap.keys()):
                if _norm(k2) == _norm(fp_view):
                    stored_key = k2; break
            if stored_key is None:
                stored_key = fp_view
            base_for_json = os.path.splitext(os.path.basename(stored_key))[0]

            # Remove visuals
            self._remove_polygon_item_safely(item)
            try:
                entries = list(self.get_all_polygons()) if hasattr(self, "get_all_polygons") else list(self._scene.items())
                for it in list(entries):
                    if (getattr(it, "name", "") or "").strip() == item_label:
                        self._remove_polygon_item_safely(it)
                self.polygons = [p for p in self.polygons if (p.get('name') or '').strip() != item_label]
            except Exception:
                pass
            
            # Force FULL scene invalidation to clear ghost labels
            try:
                self._scene.invalidate(self._scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
            except Exception:
                pass

            # Delete per-file JSON
            json_path = os.path.join(polygons_dir, f"{canonical_group}_{base_for_json}_polygons.json")
            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
                    logging.info(f"[ImageViewer] Deleted polygon file: {json_path}")
            except Exception as e:
                logging.error(f"[ImageViewer] Failed to delete polygon file {json_path}: {e}")

            # Prune in memory - directly modify ap[canonical_group]
            try:
                if canonical_group in ap and isinstance(ap[canonical_group], dict):
                    ap[canonical_group].pop(stored_key, None)
                    for k3 in list(ap[canonical_group].keys()):
                        if os.path.splitext(os.path.basename(k3))[0] == base_for_json:
                            ap[canonical_group].pop(k3, None)
                    if not ap[canonical_group]:
                        del ap[canonical_group]
            except Exception as e:
                logging.debug(f"[ImageViewer] In-memory prune failed: {e}")

            # Update mask config
            try:
                if hasattr(owner, "_remove_polygon_from_mask_config"):
                    owner._remove_polygon_from_mask_config(fp_view, item_label)
                if hasattr(owner, "update_polygon_manager"):
                    owner.update_polygon_manager()
            except Exception:
                pass

        super(ImageViewer, self).keyPressEvent(event)

    def add_polygon_to_scene(self, polygon, name="", is_mask_polygon=False):
        is_rgb = False
        if hasattr(self, 'image_data') and self.image_data is not None and self.image_data.image is not None:
            if len(self.image_data.image.shape) == 3 and self.image_data.image.shape[2] == 3:
                is_rgb = True
        polygon_item = EditablePolygonItem(polygon, name, is_rgb, is_mask_polygon=is_mask_polygon)
        self._scene.addItem(polygon_item)
        polygon_item.polygon_modified.connect(self.on_polygon_modified)
        self.polygons.append({'polygon': polygon, 'name': name, 'item': polygon_item, 'type': 'polygon', 'is_mask_polygon': is_mask_polygon})
        # PERFORMANCE: Update polygon count for fast panning detection
        self._polygon_item_count = len(self.polygons)
        # Respect current visibility state
        if not self.are_polygons_visible():
            polygon_item.setVisible(False)
        
        # Respect label visibility state
        if hasattr(self, "_labels_visible") and not self._labels_visible:
            polygon_item.show_label = False
            
        return polygon_item

    def add_point_to_scene(self, points, name=""):
        is_rgb = False
        if hasattr(self, 'image_data') and self.image_data is not None and getattr(self.image_data, "image", None) is not None:
            img = self.image_data.image
            if len(img.shape) == 3 and img.shape[2] == 3:
                is_rgb = True

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
            'points': points,          # original scene points
            'points_pix': pix_local,   # integer pixmap pixels
            'name': name,
            'item': point_item,
            'type': 'point'
        })
        # PERFORMANCE: Update polygon count for fast panning detection
        self._polygon_item_count = len(self.polygons)
        # Respect current visibility state
        if not self.are_polygons_visible():
            point_item.setVisible(False)
            
        # Respect label visibility state
        if hasattr(self, "_labels_visible") and not self._labels_visible:
            point_item.show_label = False
            
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
            
            # Rectangle mode - create a 4-point polygon
            if self.drawing_mode == "rectangle":
                if self._shape_start_point and self._shape_end_point:
                    x1, y1 = self._shape_start_point.x(), self._shape_start_point.y()
                    x2, y2 = self._shape_end_point.x(), self._shape_end_point.y()
                    rect_poly = QtGui.QPolygonF([
                        QtCore.QPointF(x1, y1),
                        QtCore.QPointF(x2, y1),
                        QtCore.QPointF(x2, y2),
                        QtCore.QPointF(x1, y2)
                    ])
                    polygon_item = self.add_polygon_to_scene(rect_poly, group_name)
                    if not self.programmatically_adding_polygon:
                        self.polygon_drawn.emit(polygon_item)
                self._shape_start_point = None
                self._shape_end_point = None
            
            # Circle mode - create a polygon approximating a circle
            elif self.drawing_mode == "circle":
                if self._shape_start_point and self._shape_end_point:
                    cx, cy = self._shape_start_point.x(), self._shape_start_point.y()
                    ex, ey = self._shape_end_point.x(), self._shape_end_point.y()
                    radius = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                    if radius > 1:  # Only create if radius is meaningful
                        # Use 32-64 segments based on radius for smooth appearance
                        num_segments = min(64, max(32, int(radius / 3)))
                        circle_poly = QtGui.QPolygonF()
                        for i in range(num_segments):
                            angle = 2 * math.pi * i / num_segments
                            x = cx + radius * math.cos(angle)
                            y = cy + radius * math.sin(angle)
                            circle_poly.append(QtCore.QPointF(x, y))
                        polygon_item = self.add_polygon_to_scene(circle_poly, group_name)
                        if not self.programmatically_adding_polygon:
                            self.polygon_drawn.emit(polygon_item)
                self._shape_start_point = None
                self._shape_end_point = None
            
            # Polygon mode
            elif self.drawing_mode == "polygon":
                if len(self.currentPolygon) > 2:
                    polygon_item = self.add_polygon_to_scene(self.currentPolygon, group_name)
                    if not self.programmatically_adding_polygon:
                        self.polygon_drawn.emit(polygon_item)
            
            # Point mode
            else:
                if len(self.currentPolygon) >= 1:
                    point_item = self.add_point_to_scene(self.currentPolygon, group_name)
                    if not self.programmatically_adding_polygon:
                        self.polygon_drawn.emit(point_item)

            self.pending_group_name = None
            self.currentPolygon = QtGui.QPolygonF()

            if self.is_editing_group:
                self.editing_finished.emit()
                self.is_editing_group = False

            try:
                if getattr(self, "_sync_depth", 0) > 0 or getattr(self, "_local_edit_active", False):
                    self._pop_local_sync()
            except Exception:
                pass

    def update_temp_drawing(self):
        if not self.temp_drawing_item:
            return
        self.temp_drawing_item.setPos(0, 0)
        self.temp_drawing_item.setTransform(QtGui.QTransform())

        scale = self.get_current_scale_factor()
        
        # Rectangle mode preview
        if self.drawing_mode == "rectangle":
            if self._shape_start_point and self._shape_end_point:
                x1, y1 = self._shape_start_point.x(), self._shape_start_point.y()
                x2, y2 = self._shape_end_point.x(), self._shape_end_point.y()
                rect_poly = QtGui.QPolygonF([
                    QtCore.QPointF(x1, y1),
                    QtCore.QPointF(x2, y1),
                    QtCore.QPointF(x2, y2),
                    QtCore.QPointF(x1, y2),
                    QtCore.QPointF(x1, y1)  # Close the rectangle
                ])
                if isinstance(self.temp_drawing_item, QtWidgets.QGraphicsPolygonItem):
                    self.temp_drawing_item.setPolygon(rect_poly)
            pen = self.temp_drawing_item.pen()
            pen.setWidthF(2 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)
            return
        
        # Circle mode preview
        if self.drawing_mode == "circle":
            if self._shape_start_point and self._shape_end_point:
                cx, cy = self._shape_start_point.x(), self._shape_start_point.y()
                ex, ey = self._shape_end_point.x(), self._shape_end_point.y()
                radius = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                # Create circle as polygon with reasonable number of segments
                num_segments = min(64, max(24, int(radius / 3)))
                circle_poly = QtGui.QPolygonF()
                for i in range(num_segments + 1):
                    angle = 2 * math.pi * i / num_segments
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    circle_poly.append(QtCore.QPointF(x, y))
                if isinstance(self.temp_drawing_item, QtWidgets.QGraphicsPolygonItem):
                    self.temp_drawing_item.setPolygon(circle_poly)
            pen = self.temp_drawing_item.pen()
            pen.setWidthF(2 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)
            return
        
        # Polygon mode
        if self.drawing_mode == "polygon":
            self.temp_drawing_item.setPolygon(self.currentPolygon)
            pen = self.temp_drawing_item.pen(); pen.setWidthF(2 / max(1e-6, scale))
            self.temp_drawing_item.setPen(pen)
            return

        # Point mode
        path = QtGui.QPainterPath()
        off = self._image.pos() if self._image else QtCore.QPointF(0, 0)
        for p in self.currentPolygon:
            lp = self._image.mapFromScene(p)
            # Use floor: point at (5.9, 5.9) is inside pixel (5, 5)
            rx = int(math.floor(lp.x()))
            ry = int(math.floor(lp.y()))
            path.addRect(off.x() + rx, off.y() + ry, 1, 1)
        self.temp_drawing_item.setPath(path)

        pen = self.temp_drawing_item.pen(); pen.setWidthF(2 / max(1e-6, scale))
        self.temp_drawing_item.setPen(pen)

    def _generate_random_points(self):
        """Generate random points across the image after asking the user for the count."""
        import random
        
        # Get image dimensions
        if not self._image or not self._image.pixmap():
            QtWidgets.QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        
        pm = self._image.pixmap()
        img_w, img_h = pm.width(), pm.height()
        
        # Ask user for number of points
        num_points, ok = QtWidgets.QInputDialog.getInt(
            self, "Random Points",
            "How many random points to generate?",
            value=10, min=1, max=10000, step=1
        )
        
        if not ok:
            return
        
        # Generate random points
        group_name = self.pending_group_name if self.pending_group_name else "random_points"
        
        # Get pixmap position offset
        off = self._image.pos()
        
        # Create points as a QPolygonF
        random_polygon = QtGui.QPolygonF()
        for _ in range(num_points):
            x = random.uniform(0, img_w - 1)
            y = random.uniform(0, img_h - 1)
            # Convert to scene coordinates
            scene_x = off.x() + x
            scene_y = off.y() + y
            random_polygon.append(QtCore.QPointF(scene_x, scene_y))
        
        # Add as point item (individual points)
        self.currentPolygon = random_polygon
        point_item = self.add_point_to_scene(random_polygon, group_name)
        if not self.programmatically_adding_polygon:
            self.polygon_drawn.emit(point_item)
        
        self.pending_group_name = None
        self.currentPolygon = QtGui.QPolygonF()
        logging.info(f"[ImageViewer] Generated {num_points} random points for group '{group_name}'")

    def handle_undoable_move_batch(self, changes):
        """
        Delegates a batch of polygon moves to the ProjectTab's undo stack as a Macro.
        Args:
            changes: list of tuples (item, start_pos, end_pos)
        Returns:
            True if successful.
        """
        owner = self._find_project_owner()
        if owner and hasattr(owner, 'modify_polygon_command') and hasattr(owner, 'undo_stack'):
             owner.undo_stack.beginMacro("Move Polygons")
             try:
                 for item, start_pos, end_pos in changes:
                     delta = end_pos - start_pos
                     new_points_scene = item.mapToScene(item.polygon)
                     
                     old_points_scene = QtGui.QPolygonF()
                     for p in new_points_scene:
                         old_points_scene.append(p - delta)
                     
                     owner.modify_polygon_command(item, old_points_scene, new_points_scene, viewer=self)
             finally:
                 owner.undo_stack.endMacro()
             return True
        return False

    def on_polygon_modified(self):
        try:
            self._last_modified_item = self.sender()
        except Exception:
            self._last_modified_item = None
        self.polygon_changed.emit()

    def add_polygon(self, polygon, name=""):
        self.programmatically_adding_polygon = True
        polygon_item = self.add_polygon_to_scene(polygon, name)
        if name:
            self.polygons.append({'polygon': polygon, 'name': name, 'item': polygon_item, 'type': 'polygon'})
        self.programmatically_adding_polygon = False

    def get_all_polygons(self):
        return [item for item in self._scene.items() if isinstance(item, EditablePolygonItem) or isinstance(item, EditablePointItem)]

    def set_polygons_visible(self, visible):
        """
        Show or hide all polygon items in this viewer.
        
        Args:
            visible: True to show polygons, False to hide them.
        """
        for item in self.get_all_polygons():
            try:
                item.setVisible(visible)
            except Exception:
                pass
        # Store the visibility state so new polygons can respect it
        self._polygons_visible = visible
    
    def set_labels_visible(self, visible):
        """
        Show or hide labels for all polygon/point items.
        """
        self._labels_visible = visible
        for item in self.get_all_polygons():
            try:
                item.show_label = visible
                item.update()
            except Exception:
                pass

    def are_polygons_visible(self):
        """Return current polygon visibility state (default True)."""
        return getattr(self, "_polygons_visible", True)

    def update_polygon_mask_status(self, mask_polygon_names):
        """
        Update the is_mask_polygon property for all polygons based on the provided names.
        
        Args:
            mask_polygon_names: Set/list of polygon names that should be treated as mask polygons.
                               Pass empty set/list to clear all mask status.
        """
        mask_names = set(mask_polygon_names) if mask_polygon_names else set()
        
        for item in self.get_all_polygons():
            if isinstance(item, EditablePolygonItem):
                poly_name = getattr(item, 'name', '') or ''
                new_is_mask = poly_name in mask_names
                if item.is_mask_polygon != new_is_mask:
                    item.is_mask_polygon = new_is_mask
                    item.update()  # Trigger repaint

    def clear_polygons(self):
        # Stop any vertex editing first
        self.stop_vertex_editing()
        
        # Force prepareGeometryChange on all items first
        all_items = self.get_all_polygons()
        for item in all_items:
            try:
                item.prepareGeometryChange()
            except Exception:
                pass
        
        # Remove all items
        for item in all_items:
            try:
                self._scene.removeItem(item)
            except Exception:
                pass
        
        self.polygons = []
        # PERFORMANCE: Reset polygon count for fast panning detection
        self._polygon_item_count = 0
        self.currentPolygon = QtGui.QPolygonF()
        self.drawing = False
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.last_pan_point = QtCore.QPoint()
        self.pending_group_name = None
        
        # Only invalidate if we had items to remove (skip for huge images with no polygons)
        if all_items:
            try:
                self._scene.invalidate(self._scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
            except Exception:
                pass

    def _remove_polygon_item_safely(self, item):
        """
        Safely remove a polygon/point item from scene with proper ghost label cleanup.
        Call this instead of self._scene.removeItem(item) directly.
        """
        if item is None:
            return
        try:
            # Prepare for geometry change
            item.prepareGeometryChange()
            
            # Remove from scene
            if item.scene() is self._scene:
                self._scene.removeItem(item)
            
            # Force full scene invalidation to clear any ghost labels
            self._scene.invalidate(self._scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
            self.viewport().update()
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Vertex Editing Methods
    # -------------------------------------------------------------------
    def start_vertex_editing(self, polygon_item):
        """
        Start editing vertices of a polygon. Creates draggable handles at each vertex.
        For performance, limits to max 100 handles (resamples if needed).
        """
        if not isinstance(polygon_item, EditablePolygonItem):
            return
        
        # Verify item is valid and in scene
        try:
            if polygon_item.scene() is None:
                logging.warning("[ImageViewer] Cannot edit vertices: polygon not in scene")
                return
        except RuntimeError:
            logging.warning("[ImageViewer] Cannot edit vertices: polygon was deleted")
            return
        
        # Stop any existing vertex editing
        self.stop_vertex_editing()
        
        self._vertex_editing_item = polygon_item
        self._vertex_handles = []
        self._vertex_resampled = False
        self._original_polygon = None
        
        # Get polygon points (in item coordinates)
        try:
            poly = polygon_item.polygon
            num_points = poly.count()
        except Exception as e:
            logging.warning(f"[ImageViewer] Cannot edit vertices: error accessing polygon: {e}")
            self._vertex_editing_item = None
            return
        
        if num_points == 0:
            logging.warning("[ImageViewer] Cannot edit vertices: polygon has 0 points")
            self._vertex_editing_item = None
            return
        
        # Performance protection: max 100 handles
        MAX_HANDLES = 100
        
        try:
            if num_points > MAX_HANDLES:
                # Resample to MAX_HANDLES points
                self._vertex_resampled = True
                self._original_polygon = QtGui.QPolygonF(poly)  # Save original
                
                # Calculate step to get evenly spaced points
                step = num_points / MAX_HANDLES
                indices = [int(i * step) for i in range(MAX_HANDLES)]
                
                # Create handles only for resampled points
                for handle_idx, poly_idx in enumerate(indices):
                    pt = poly.at(poly_idx)
                    # Map point to scene coordinates
                    scene_pt = polygon_item.mapToScene(pt)
                    handle = VertexHandle(handle_idx, scene_pt, self)
                    handle._poly_index = poly_idx  # Store actual polygon index
                    self._scene.addItem(handle)
                    self._vertex_handles.append(handle)
                
                logging.info(f"[ImageViewer] Vertex editing: resampled {num_points} vertices to {MAX_HANDLES} handles")
            else:
                # Create handle for each vertex
                for i in range(num_points):
                    pt = poly.at(i)
                    # Map point to scene coordinates
                    scene_pt = polygon_item.mapToScene(pt)
                    handle = VertexHandle(i, scene_pt, self)
                    handle._poly_index = i
                    self._scene.addItem(handle)
                    self._vertex_handles.append(handle)
                
                logging.info(f"[ImageViewer] Vertex editing: created {num_points} handles")
            
            # Make polygon non-movable while editing vertices
            polygon_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            
        except Exception as e:
            logging.error(f"[ImageViewer] Error creating vertex handles: {e}")
            # Clean up any handles that were created
            self.stop_vertex_editing()
        
    def stop_vertex_editing(self):
        """Remove all vertex handles and stop editing."""
        # CRITICAL: Stop any pending throttled updates FIRST to prevent race conditions
        try:
            if getattr(self, '_vertex_update_timer', None):
                self._vertex_update_timer.stop()
            self._pending_vertex_update = None
        except Exception:
            pass
        
        # Remove all handles from scene
        for handle in self._vertex_handles:
            try:
                if handle is not None and handle.scene() is self._scene:
                    self._scene.removeItem(handle)
            except RuntimeError:
                # Handle was already deleted
                pass
            except Exception as e:
                logging.debug(f"[ImageViewer] Error removing vertex handle: {e}")
        
        # Restore movability on the polygon
        if self._vertex_editing_item is not None:
            try:
                # Check if item is still valid
                if self._vertex_editing_item.scene() is not None:
                    self._vertex_editing_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
            except RuntimeError:
                # Item was deleted
                pass
            except Exception as e:
                logging.debug(f"[ImageViewer] Error restoring polygon movability: {e}")
        
        # Clear all references
        self._vertex_handles = []
        self._vertex_editing_item = None
        self._vertex_resampled = False
        self._original_polygon = None
        
    def on_vertex_moved(self, handle_index, new_scene_pos):
        """Called when a vertex handle is moved. Updates the polygon.
        
        PERFORMANCE: Throttled to 60 FPS max to prevent crashes on large polygons.
        """
        if not self._vertex_editing_item:
            return
        
        # PERFORMANCE: Throttle updates to 60 FPS max during drag
        import time
        now = time.time()
        min_interval = 1.0 / 60  # 60 FPS
        last_update = getattr(self, '_last_vertex_update_time', 0)
        
        if now - last_update < min_interval:
            # Store pending update, defer until next frame
            self._pending_vertex_update = (handle_index, new_scene_pos)
            if not getattr(self, '_vertex_update_timer', None):
                self._vertex_update_timer = QtCore.QTimer()
                self._vertex_update_timer.setSingleShot(True)
                self._vertex_update_timer.timeout.connect(self._apply_pending_vertex_update)
            if not self._vertex_update_timer.isActive():
                self._vertex_update_timer.start(int(min_interval * 1000))
            return
        
        self._last_vertex_update_time = now
        self._apply_vertex_update(handle_index, new_scene_pos)
    
    def _apply_pending_vertex_update(self):
        """Apply deferred vertex update from throttling."""
        if hasattr(self, '_pending_vertex_update') and self._pending_vertex_update:
            handle_index, new_scene_pos = self._pending_vertex_update
            self._pending_vertex_update = None
            self._apply_vertex_update(handle_index, new_scene_pos)
    
    def _apply_vertex_update(self, handle_index, new_scene_pos):
        """Actually update the polygon vertex. Called by on_vertex_moved after throttle check."""
        if not self._vertex_editing_item:
            return
        
        poly_item = self._vertex_editing_item
        
        # Verify the item is still valid
        try:
            if poly_item.scene() is None:
                logging.warning("[ImageViewer] Vertex editing: polygon item no longer in scene")
                self.stop_vertex_editing()
                return
        except RuntimeError:
            # Item was deleted
            self.stop_vertex_editing()
            return
        
        # Map scene position back to item coordinates
        try:
            item_pos = poly_item.mapFromScene(new_scene_pos)
        except Exception as e:
            logging.warning(f"[ImageViewer] Vertex editing: mapFromScene failed: {e}")
            return
        
        # Validate coordinates are finite
        if not (math.isfinite(item_pos.x()) and math.isfinite(item_pos.y())):
            logging.warning("[ImageViewer] Vertex editing: invalid coordinates (NaN/Inf)")
            return
        
        # Get the actual polygon index from the handle
        if handle_index < len(self._vertex_handles):
            handle = self._vertex_handles[handle_index]
            poly_index = getattr(handle, '_poly_index', handle_index)
        else:
            poly_index = handle_index
        
        # Update the polygon vertex
        poly = poly_item.polygon
        if 0 <= poly_index < poly.count():
            # Create new polygon with updated point
            new_poly = QtGui.QPolygonF()
            for i in range(poly.count()):
                if i == poly_index:
                    new_poly.append(item_pos)
                else:
                    new_poly.append(poly.at(i))
            
            poly_item.prepareGeometryChange()
            poly_item.polygon = new_poly
            poly_item.update()
            
    def finish_vertex_editing(self):
        """Finish vertex editing and emit modification signal."""
        # Store reference to the item before stopping
        edited_item = self._vertex_editing_item
        
        # FIRST: Stop vertex editing (remove handles, restore state)
        # This must happen BEFORE emitting the signal to ensure clean scene state
        self.stop_vertex_editing()
        
        # THEN: Emit modification signal so the polygon gets saved
        if edited_item:
            try:
                # Verify the item is still valid and in a scene
                if edited_item.scene() is not None:
                    edited_item.polygon_modified.emit()
            except RuntimeError:
                # Item may have been deleted
                logging.warning("[ImageViewer] Vertex editing: item was deleted before signal could be emitted")
            except Exception as e:
                logging.warning(f"[ImageViewer] Vertex editing: error emitting signal: {e}")
        
        logging.info("[ImageViewer] Vertex editing finished")

    def delete_polygon_for_this_file(self, item):
        """
        Delete a single polygon via Undo Command (delegated to owner).
        """
        item_label = (getattr(item, "name", "") or "").strip()
        if not item_label:
            return

        fp_view = self.get_viewer_filepath()
        if not fp_view:
            return

        owner = self._find_project_owner()
        if owner and hasattr(owner, "delete_polygon_command"):
            try:
                owner.delete_polygon_command(item, fp_view)
                return
            except Exception as e:
                logging.error(f"[ImageViewer] Undo deletion failed: {e}")
        
        # Fallback: remove from scene to prevent confusion if command fails
        self._remove_polygon_item_safely(item)

    def delete_all_polygons_in_group(self, item):
        """
        Delete ALL polygons with the same group name via Undo Command (delegated to owner).
        """
        group_name = (getattr(item, "name", "") or "").strip()
        if not group_name:
            return
        
        owner = self._find_project_owner()
        if owner and hasattr(owner, "delete_group_command"):
            try:
                owner.delete_group_command(group_name)
                return
            except Exception as e:
                logging.error(f"[ImageViewer] Undo group deletion failed: {e}")
        else:
             logging.warning(f"[ImageViewer] Cannot delete group {group_name}: owner or command missing.")


    def _norm(p: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(p or ""))
        except Exception:
            return p or ""

    def get_viewer_filepath(self) -> str:
        return getattr(getattr(self, "image_data", None), "filepath", None) or ""

    def start_drawing_with_group_name(self, group_name, *, broadcast=False, target_fp=None):
        """
        Enter drawing mode for a specific group in THIS viewer.
        If broadcast=True, emit a targeted edit signal (group_name, target_fp or this viewer's filepath),
        but only when global sync is enabled.
        """
        self.pending_group_name = group_name
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setFocus()
        self.currentPolygon = QtGui.QPolygonF()
        self.lastPoint = None
        self.is_editing_group = False
        
        # Reset shape drawing state
        self._shape_start_point = None
        self._shape_end_point = None
        
        # Handle random points mode specially - generate immediately
        if getattr(self, "drawing_mode", "polygon") == "random_points":
            self.drawing = True
            self._generate_random_points()
            self.drawing = False
            return

        self.drawing = True

        try:
            if getattr(self, "temp_drawing_item", None) is not None:
                if self.temp_drawing_item.scene() is self._scene:
                    self._scene.removeItem(self.temp_drawing_item)
        except Exception:
            pass

        mode = getattr(self, "drawing_mode", "polygon")
        if mode in ("polygon", "rectangle", "circle"):
            self.temp_drawing_item = QtWidgets.QGraphicsPolygonItem()
        else:
            self.temp_drawing_item = QtWidgets.QGraphicsPathItem()

        pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
        scale = getattr(self, "get_current_scale_factor", lambda: 1.0)()
        desired_screen_width = 2.0
        pen.setWidthF(max(0.5, desired_screen_width / (scale or 1.0)))
        pen.setColor(QtCore.Qt.red)
        self.temp_drawing_item.setPen(pen)
        self.temp_drawing_item.setBrush(QtGui.QBrush(QtCore.Qt.transparent))
        try:
            self.temp_drawing_item.setZValue(1e6)
        except Exception:
            pass

        self._scene.addItem(self.temp_drawing_item)
        if hasattr(self, "update_temp_drawing"):
            self.update_temp_drawing()

        logging.info(f"[ImageViewer] Started drawing new shape (mode: {mode}) for group '{group_name}'.")

        if broadcast:
            owner = getattr(self, "_find_project_owner", lambda: None)()
            if owner and getattr(owner, "sync_enabled", True):
                try:
                    my_fp = self._norm(target_fp or self.get_viewer_filepath())
                    owner.edit_group_signal.emit(group_name, my_fp)
                except Exception:
                    pass

    @QtCore.pyqtSlot(str, str)  # (group_name, target_fp)
    def on_edit_group_signal(self, group_name, target_fp):
        my_fp = self._norm(self.get_viewer_filepath())
        if target_fp and self._norm(target_fp) != my_fp:
            return
        self.start_drawing_with_group_name(group_name, broadcast=False)

    # ---- App-wide geometry clipboard format ----
    _CLIP_FMT = "application/x-imgviewer-geom"
    _INTERNAL_GEOM_CLIP = None

    def _put_geom_on_clipboard(self, payload_bytes: bytes, payload_text: str):
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(payload_text)
        md = QtCore.QMimeData()
        md.setData(self._CLIP_FMT, payload_bytes)
        md.setText(payload_text)
        cb.setMimeData(md)

        ok = False
        md2 = cb.mimeData()
        if md2:
            ok = md2.hasFormat(self._CLIP_FMT) or (md2.hasText() and md2.text() == payload_text)
        if not ok:
            ImageViewer._INTERNAL_GEOM_CLIP = payload_bytes

    def copy_selection(self):
        items = [it for it in self._scene.selectedItems()
                 if isinstance(it, (EditablePolygonItem, EditablePointItem))]
        if not items:
            return
        payload = [p for p in (self._serialize_item(it) for it in items) if p]
        if not payload:
            return
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        txt = raw.decode("utf-8")
        self._put_geom_on_clipboard(raw, txt)

    def copy_specific_items(self, items):
        items = [it for it in items if isinstance(it, (EditablePolygonItem, EditablePointItem))]
        if not items:
            return
        payload = [p for p in (self._serialize_item(it) for it in items) if p]
        if not payload:
            return
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        txt = raw.decode("utf-8")
        self._put_geom_on_clipboard(raw, txt)

    def _serialize_item(self, item):
        """Return a JSON-serializable dict for a polygon/point item in *pixmap coords*."""
        pm = self._image.pixmap() if self._image else None
        src_w = pm.width() if pm else 0
        src_h = pm.height() if pm else 0

        if isinstance(item, EditablePolygonItem):
            pts = [(p.x(), p.y()) for p in item.polygon]  # scene==pixmap coords
            kind = "polygon"
        elif isinstance(item, EditablePointItem):
            pts = [(p.x(), p.y()) for p in item.points]
            kind = "point"
        else:
            return None

        return {
            "type": kind,
            "name": getattr(item, "name", "") or "",
            "points": pts,
            "src_w": src_w,
            "src_h": src_h,
        }

    def paste_geometry(self, drop_at: QtCore.QPointF = None):
        cb = QtWidgets.QApplication.clipboard()
        md = cb.mimeData()
        raw = None

        if md and md.hasFormat(self._CLIP_FMT):
            raw = bytes(md.data(self._CLIP_FMT))
        elif md and md.hasText():
            raw = md.text().encode("utf-8")

        if (not raw) and ImageViewer._INTERNAL_GEOM_CLIP:
            raw = ImageViewer._INTERNAL_GEOM_CLIP

        if not raw:
            return

        try:
            items = json.loads(raw.decode("utf-8"))
        except Exception:
            return

        if not self._image or self._image.pixmap().isNull():
            return

        pm = self._image.pixmap()
        tgt_w = pm.width()
        tgt_h = pm.height()

        for it in items:
            try:
                pts = it.get("points") or []
                if not pts:
                    continue

                src_w = max(1, int(it.get("src_w") or tgt_w))
                src_h = max(1, int(it.get("src_h") or tgt_h))
                sx = tgt_w / float(src_w)
                sy = tgt_h / float(src_h)

                qpts = [QtCore.QPointF(float(x) * sx, float(y) * sy) for (x, y) in pts]

                if drop_at and qpts:
                    cx = sum(p.x() for p in qpts) / len(qpts)
                    cy = sum(p.y() for p in qpts) / len(qpts)
                    delta = QtCore.QPointF(drop_at.x() - cx, drop_at.y() - cy)
                    qpts = [p + delta for p in qpts]

                name = (it.get("name") or "").strip()

                if (it.get("type") == "polygon") and len(qpts) >= 3:
                    new_item = self.add_polygon_to_scene(QtGui.QPolygonF(qpts), name)
                    try:
                        self._save_pasted_polygon_for_this_file(new_item)
                    except Exception as e:
                        logging.debug(f"[ImageViewer] paste_geometry: save skipped: {e}")
                elif (it.get("type") == "point") and len(qpts) >= 1:
                    self.add_point_to_scene(QtGui.QPolygonF(qpts), name)
            except Exception as e:
                logging.debug(f"[ImageViewer] paste_geometry: failed to add item: {e}")

        try:
            self.polygon_changed.emit()
        except Exception:
            pass

    def contextMenuEvent(self, event):
        # Check if clicking on a polygon/point item - always allow their context menus
        if any(isinstance(i, (EditablePolygonItem, EditablePointItem)) for i in self.items(event.pos())):
            return super().contextMenuEvent(event)
        
        # Skip background context menu when in rectangle zoom mode
        if self._rect_zoom_mode:
            event.accept()  # Accept to prevent propagation
            return

        menu = QtWidgets.QMenu(self)
        act_paste = menu.addAction("Paste geometry here")
        chosen = menu.exec_(event.globalPos())

        if chosen == act_paste:
            self.paste_geometry(self.mapToScene(event.pos()))

    def _save_pasted_polygon_for_this_file(self, poly_item):
        try:
            owner = self._find_project_owner()
            img_data = getattr(self, "image_data", None)
            fp_view = getattr(img_data, "filepath", None)
            if not owner or not getattr(owner, "project_folder", None) or not fp_view:
                return

            group_name = (getattr(poly_item, "name", "") or "Unnamed").strip()
            polygons_dir = os.path.join(owner.project_folder, "polygons")
            os.makedirs(polygons_dir, exist_ok=True)

            base_for_json = os.path.splitext(os.path.basename(fp_view))[0]
            json_path = os.path.join(polygons_dir, f"{group_name}_{base_for_json}_polygons.json")

            # CRITICAL FIX: Use image.shape dimensions (not pixmap) for consistency with load_polygons
            # load_polygons uses viewer.image_data.image.shape[:2] as the basis
            img_arr = getattr(img_data, "image", None)
            if img_arr is not None:
                img_h, img_w = img_arr.shape[:2]
            else:
                # Fallback to pixmap if no image array
                pm = self._image.pixmap() if self._image else None
                img_w = int(pm.width()) if pm else 0
                img_h = int(pm.height()) if pm else 0

            # CRITICAL FIX: Convert scene coordinates to image coordinates
            # poly_item.polygon is in scene coordinates, we need image coordinates
            pm = self._image.pixmap() if self._image else None
            pm_w = int(pm.width()) if pm else img_w
            pm_h = int(pm.height()) if pm else img_h
            
            pts = []
            for p in poly_item.polygon:
                # Scene coords -> pixmap coords (via mapFromScene on pixmap item)
                if self._image:
                    local_pt = self._image.mapFromScene(p)
                    px_x, px_y = local_pt.x(), local_pt.y()
                else:
                    px_x, px_y = p.x(), p.y()
                
                # Pixmap coords -> image coords
                if pm_w > 0 and pm_h > 0:
                    img_x = px_x * (img_w / float(pm_w))
                    img_y = px_y * (img_h / float(pm_h))
                else:
                    img_x, img_y = px_x, px_y
                    
                pts.append((float(img_x), float(img_y)))

            # FIX: Get the correct root ID for THIS target file
            root_id = "0"
            try:
                if hasattr(owner, "get_root_by_filepath") and hasattr(owner, "root_id_mapping"):
                    root_name = owner.get_root_by_filepath(fp_view)
                    if root_name and owner.root_id_mapping:
                        root_id = str(owner.root_id_mapping.get(root_name, "0"))
            except Exception:
                pass

            payload = {
                "name": group_name,
                "file": fp_view,
                "type": "polygon",
                "coord_space": "image",
                "image_ref_size": {"w": img_w, "h": img_h},
                "root": root_id,
                "points": pts
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            if not hasattr(owner, "all_polygons") or owner.all_polygons is None:
                owner.all_polygons = {}

            grp = owner.all_polygons.get(group_name)
            if not isinstance(grp, dict):
                grp = {}
                owner.all_polygons[group_name] = grp

            grp[fp_view] = {**payload, "json_path": json_path}
            
            # CRITICAL FIX: Update polygon index so load_polygons can find this entry
            if hasattr(owner, "_add_to_polygon_index"):
                owner._add_to_polygon_index(group_name, fp_view)
            
            # FIX: Mark polygon as dirty for incremental save
            if hasattr(owner, "_mark_polygon_dirty"):
                owner._mark_polygon_dirty(group_name, fp_view)

            if hasattr(owner, "update_polygon_manager"):
                owner.update_polygon_manager()
            try:
                self.polygon_changed.emit()
            except Exception:
                pass

            logging.info(f"[ImageViewer] Saved pasted polygon to {json_path}")
        except Exception as e:
            logging.error(f"[ImageViewer] Failed to save pasted polygon: {e}")

    def replicate_toviewer(self, items=None, also_save=True):
        try:
            if items is None:
                items = [it for it in self._scene.selectedItems()
                         if isinstance(it, (EditablePolygonItem, EditablePointItem))]
            if not items:
                return

            payload = []
            for it in items:
                p = self._serialize_item(it)
                if p:
                    payload.append(p)
            if not payload:
                return

            # find targets
            targets = []
            owner = self._find_project_owner()
            if owner:
                try:
                    targets = [v for v in owner.findChildren(ImageViewer) if v is not self]
                except Exception:
                    targets = []
            if not targets:
                try:
                    targets = [w for w in QtWidgets.QApplication.allWidgets()
                               if isinstance(w, ImageViewer) and w is not self]
                except Exception:
                    targets = []
            if not targets:
                return

            for v in targets:
                try:
                    pm = getattr(getattr(v, "_image", None), "pixmap", lambda: None)()
                    if not pm or pm.isNull():
                        continue
                    tgt_w, tgt_h = pm.width(), pm.height()

                    for it in payload:
                        try:
                            pts = it.get("points") or []
                            if not pts:
                                continue
                            src_w = max(1, int(it.get("src_w") or tgt_w))
                            src_h = max(1, int(it.get("src_h") or tgt_h))
                            sx = tgt_w / float(src_w)
                            sy = tgt_h / float(src_h)
                            qpts = [QtCore.QPointF(float(x)*sx, float(y)*sy) for (x, y) in pts]
                            name = (it.get("name") or "").strip()

                            if (it.get("type") == "polygon") and len(qpts) >= 3:
                                new_item = v.add_polygon_to_scene(QtGui.QPolygonF(qpts), name)
                                if also_save and hasattr(v, "_save_pasted_polygon_for_this_file"):
                                    try:
                                        v._save_pasted_polygon_for_this_file(new_item)
                                    except Exception as e:
                                        logging.debug(f"[ImageViewer] replicate_toviewer save skipped: {e}")
                            elif (it.get("type") == "point") and len(qpts) >= 1:
                                v.add_point_to_scene(QtGui.QPolygonF(qpts), name)
                        except Exception as e:
                            logging.debug(f"[ImageViewer] replicate_toviewer add failed: {e}")

                    try:
                        v.polygon_changed.emit()
                    except Exception:
                        pass

                except Exception as e:
                    logging.debug(f"[ImageViewer] replicate_toviewer target failed: {e}")

        except Exception as e:
            logging.error(f"[ImageViewer] replicate_toviewer failed: {e}")

    def _push_local_sync_off(self, *, gate_viewer=False):
        """Temporarily gate cross-viewer sync. Re-entrant-safe."""
        owner = getattr(self, "_find_project_owner", lambda: None)()

        if not hasattr(self, "_sync_depth"):
            self._sync_depth = 0
        if not hasattr(self, "_sync_restore_slots"):
            self._sync_restore_slots = []

        if self._sync_depth > 0:
            self._sync_depth += 1
            self._local_edit_active = True
            return

        slots = []
        if owner is not None and hasattr(owner, "sync_enabled"):
            slots.append((owner, "sync_enabled", getattr(owner, "sync_enabled", True)))
            setattr(owner, "sync_enabled", False)

        if gate_viewer:
            slots.append((self, "sync_enabled", getattr(self, "sync_enabled", True)))
            self.sync_enabled = False

        self._sync_restore_slots = slots
        self._sync_depth = 1
        self._local_edit_active = True

    def _pop_local_sync(self):
        """Restore sync when depth returns to zero. No-throw."""
        if not hasattr(self, "_sync_depth"):
            self._sync_depth = 0
        if self._sync_depth > 1:
            self._sync_depth -= 1
            return

        for obj, attr, prev in (self._sync_restore_slots or []):
            try:
                setattr(obj, attr, prev)
            except Exception:
                pass
        self._sync_restore_slots = []
        self._sync_depth = 0
        self._local_edit_active = False

    def edit_single_polygon(self, item, *, start_redraw=True):
        """
        Remove exactly this polygon's (group, THIS-file) JSON and in-memory entry,
        remove the visual from THIS viewer, then optionally start local redraw for the same group.
        """
        def _norm(p):
            try:    return os.path.normcase(os.path.abspath(p or ""))
            except: return p or ""

        owner = getattr(self, "_find_project_owner", lambda: None)()
        self._push_local_sync_off(gate_viewer=False)

        try:
            item_label = (getattr(item, "name", "") or "").strip()
            fp_view    = self.get_viewer_filepath()
            if not item_label or not fp_view:
                self._pop_local_sync()
                return

            ap = getattr(owner, "all_polygons", None) if owner else None
            if not isinstance(ap, dict):
                ap = {}

            polygons_dir  = (os.path.join(owner.project_folder, "polygons")
                             if (owner and getattr(owner, "project_folder", None))
                             else os.path.join(os.getcwd(), "polygons"))
            base_for_json = os.path.splitext(os.path.basename(fp_view))[0]
            group         = item_label

            gmap = ap.get(group)
            if not isinstance(gmap, dict):
                gmap = {}

            # 1) Remove ONLY this graphics item in THIS viewer
            try:
                if getattr(item, "scene", None) and item.scene() is getattr(self, "_scene", None):
                    try:
                        item.prepareGeometryChange()
                        self._scene.removeItem(item)
                    except Exception:
                        pass
                try:
                    self.polygons = [
                        p for p in getattr(self, "polygons", [])
                        if not (
                            (p.get("name", "").strip() == group) and
                            (_norm(p.get("filepath", "")) == _norm(fp_view))
                        )
                    ]
                except Exception:
                    pass
                # Force FULL scene invalidation to clear ghost labels
                try:
                    self._scene.invalidate(self._scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                    self.viewport().update()
                except Exception:
                    pass
            except Exception:
                pass

            # 2) Delete JUST this (group,file) JSON
            json_path = os.path.join(polygons_dir, f"{group}_{base_for_json}_polygons.json")
            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
                    logging.info(f"[ImageViewer] (edit) Deleted polygon file: {json_path}")
            except Exception as e:
                logging.error(f"[ImageViewer] (edit) Failed to delete polygon file {json_path}: {e}")

            # 3) Prune memory ONLY for this exact filepath key
            try:
                key_to_pop = None
                for k2 in list(gmap.keys()):
                    if _norm(k2) == _norm(fp_view):
                        key_to_pop = k2
                        break
                if key_to_pop is not None:
                    gmap.pop(key_to_pop, None)
                if not gmap and ap.get(group) is gmap:
                    ap.pop(group, None)
            except Exception as e:
                logging.debug(f"[ImageViewer] (edit) In-memory prune failed: {e}")

            # 4) Refresh polygon manager UI
            try:
                if owner and hasattr(owner, "update_polygon_manager"):
                    owner.update_polygon_manager()
            except Exception:
                pass

            # 5) Start local redraw; restore sync after finish
            if start_redraw:
                try:
                    try: self.editing_finished.disconnect(self._pop_local_sync)
                    except Exception: pass
                    try: self.editing_cancelled.disconnect(self._pop_local_sync)
                    except Exception: pass
                    try: self.destroyed.disconnect(self._pop_local_sync)
                    except Exception: pass

                    try:
                        self.editing_finished.connect(self._pop_local_sync, QtCore.Qt.UniqueConnection)
                    except Exception:
                        self.editing_finished.connect(self._pop_local_sync)

                    try:
                        self.editing_cancelled.connect(self._pop_local_sync, QtCore.Qt.UniqueConnection)
                    except Exception:
                        self.editing_cancelled.connect(self._pop_local_sync)

                    try:
                        self.destroyed.connect(self._pop_local_sync, QtCore.Qt.UniqueConnection)
                    except Exception:
                        self.destroyed.connect(self._pop_local_sync)

                except Exception:
                    pass

                self.start_drawing_with_group_name(group, broadcast=False)

        except Exception as e:
            logging.error(f"[ImageViewer] (edit_single_polygon) unexpected error: {e}")
            self._pop_local_sync()

    def _remove_group_from_viewer_instance(self, viewer, group):
        """UI-only purge of a group's overlays in a specific viewer."""
        try:
            sc = getattr(viewer, "_scene", None)
            if sc is not None:
                editable_types = tuple(t for t in (
                    globals().get("EditablePolygonItem"),
                    globals().get("EditablePointItem"),
                ) if t is not None)

                def _is_target(it):
                    if editable_types:
                        ok = isinstance(it, editable_types)
                    else:
                        ok = hasattr(it, "name")
                    return ok and ((getattr(it, "name", "") or "").strip() == group)

                for it in list(sc.items()):
                    if _is_target(it):
                        try:
                            it.prepareGeometryChange()
                            sc.removeItem(it)
                        except Exception:
                            pass
                
                # Force FULL scene invalidation to clear ghost labels
                try:
                    sc.invalidate(sc.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                except Exception:
                    pass

            try:
                viewer.polygons = [
                    p for p in (getattr(viewer, "polygons", []) or [])
                    if (p.get("name", "") or "").strip() != group
                ]
            except Exception:
                pass

            try:
                viewer.viewport().update()
            except Exception:
                pass

        except Exception:
            pass

    def edit_all_polygons_in_group(self, item, *, start_redraw=True, respect_sync=True):
        """
        Remove ALL polygons for this item's group in THIS viewer (UI + JSON + in-memory for THIS file),
        then optionally start local redraw for that group. Also UI-purges the same-named overlays
        from ALL other ImageViewer instances (name-only match, case-insensitive).
        """
        def _norm(p):
            try:
                return os.path.normcase(os.path.abspath(p or ""))
            except Exception:
                return p or ""

        owner = getattr(self, "_find_project_owner", lambda: None)()
        group_raw = (getattr(item, "name", "") or "")
        group = group_raw.strip()
        group_ci = group.lower()
        fp_view = self.get_viewer_filepath()
        if not group or not fp_view:
            return

        gated = False
        if not respect_sync:
            try:
                self._push_local_sync_off(gate_viewer=False)
                gated = True
            except Exception:
                gated = False

        try:
            ap = getattr(owner, "all_polygons", None) if owner else None
            if not isinstance(ap, dict):
                ap = {}
            gmap = ap.get(group)
            if not isinstance(gmap, dict):
                gmap = {}

            try:
                scene = getattr(self, "_scene", None)
                if scene is not None:
                    def _is_target(it):
                        nm = (getattr(it, "name", "") or "").strip().lower()
                        return bool(nm) and nm == group_ci
                    for it in [i for i in scene.items() if _is_target(i)]:
                        try:
                            it.prepareGeometryChange()
                            scene.removeItem(it)
                        except Exception:
                            pass
                    # Force FULL scene invalidation to clear ghost labels
                    try:
                        scene.invalidate(scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                        self.viewport().update()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                self.polygons = [
                    p for p in (getattr(self, "polygons", []) or [])
                    if not ((p.get("name", "") or "").strip().lower() == group_ci
                            and (_norm(p.get("filepath", "")) == _norm(fp_view)))
                ]
            except Exception:
                pass

            try:
                if owner and getattr(owner, "project_folder", None):
                    polygons_dir = os.path.join(owner.project_folder, "polygons")
                else:
                    polygons_dir = os.path.join(os.getcwd(), "polygons")
                base_for_json = os.path.splitext(os.path.basename(fp_view))[0]
                json_path = os.path.join(polygons_dir, f"{group}_{base_for_json}_polygons.json")
                if os.path.exists(json_path):
                    os.remove(json_path)
                    logging.info(f"[ImageViewer] (edit-all) Deleted polygon file: {json_path}")
            except Exception as e:
                logging.error(f"[ImageViewer] (edit-all) Failed to delete polygon file: {e}")

            try:
                key_to_pop = None
                for k2 in list(gmap.keys()):
                    if _norm(k2) == _norm(fp_view):
                        key_to_pop = k2; break
                if key_to_pop is not None:
                    gmap.pop(key_to_pop, None)
                if not gmap and ap.get(group) is gmap:
                    ap.pop(group, None)
            except Exception as e:
                logging.debug(f"[ImageViewer] (edit-all) In-memory prune failed: {e}")

            try:
                if owner and hasattr(owner, "update_polygon_manager"):
                    owner.update_polygon_manager()
            except Exception:
                pass

            # UI-only purge in other viewers
            try:
                viewers = set()
                try:
                    if owner:
                        viewers.update(owner.findChildren(ImageViewer))
                except Exception:
                    pass
                try:
                    viewers.update(
                        w for w in QtWidgets.QApplication.allWidgets()
                        if isinstance(w, ImageViewer)
                    )
                except Exception:
                    pass
                try:
                    for tlw in QtWidgets.QApplication.topLevelWidgets():
                        if isinstance(tlw, ImageViewer):
                            viewers.add(tlw)
                        viewers.update(tlw.findChildren(ImageViewer))
                except Exception:
                    pass

                viewers.discard(self)

                for v in list(viewers):
                    try:
                        sc = getattr(v, "_scene", None) or v.scene()
                        if sc is None:
                            continue

                        def _is_target_v(it):
                            nm = (getattr(it, "name", "") or "").strip().lower()
                            return bool(nm) and nm == group_ci

                        for it in [i for i in sc.items() if _is_target_v(i)]:
                            try:
                                it.prepareGeometryChange()
                                sc.removeItem(it)
                            except Exception:
                                pass

                        try:
                            v.polygons = [
                                p for p in (getattr(v, "polygons", []) or [])
                                if (p.get("name", "") or "").strip().lower() != group_ci
                            ]
                        except Exception:
                            pass

                        # Force FULL scene invalidation to clear ghost labels
                        try:
                            sc.invalidate(sc.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                            v.viewport().update()
                        except Exception:
                            pass

                        try:
                            v.polygon_changed.emit()
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            if start_redraw:
                if gated:
                    try:
                        self.editing_finished.disconnect(self._pop_local_sync)
                    except Exception:
                        pass
                    try:
                        self.editing_finished.connect(self._pop_local_sync, QtCore.Qt.UniqueConnection)
                    except Exception:
                        try:
                            self.editing_finished.connect(self._pop_local_sync)
                        except Exception:
                            pass

                try:
                    self.start_drawing_with_group_name(group, broadcast=False)
                except Exception as e:
                    logging.debug(f"[ImageViewer] (edit-all) Failed to start local redraw: {e}")
                    if gated:
                        self._pop_local_sync()

        except Exception as e:
            if gated:
                self._pop_local_sync()
            logging.error(f"[ImageViewer] (edit_all_polygons_in_group) unexpected error: {e}")


# --- ZoomBar overlay for ImageViewer -----------------------------------------
class _ZoomBar(QtWidgets.QFrame):
    """
    Lightweight overlay widget (Fix  –  slider  +  |  100%  Fit) that floats over the
    ImageViewer viewport. Auto-hides when not interacting.
    
    The "Fix" button synchronizes zoom levels across all viewers in the project.
    """
    zoomChanged = QtCore.pyqtSignal(float)  # emits new absolute zoom factor (e.g., 1.0 = 100%)
    
    # Class-level state for zoom/pan sync across all viewers
    _zoom_sync_enabled = True
    _syncing = False  # Prevent recursive sync
    _applying_fixed_zoom = False  # Prevent center updates while applying fixed zoom
    _navigation_lock = False  # CRITICAL: Prevent anchor updates during entire navigation sequence
    _navigation_lock_timer = None  # Timer to release the lock
    _viewer_cache = None  # Cache for viewers list
    _viewer_cache_time = 0  # Timestamp of last cache update
    
    @classmethod
    def invalidate_viewer_cache(cls):
        """Call this when viewers are added or removed to force cache refresh."""
        cls._viewer_cache = None
        cls._viewer_cache_time = 0
    
    # SIMPLIFIED: Just store zoom and normalized center directly
    _fixed_zoom = None  # The fixed zoom level
    _fixed_center_norm = None  # Normalized center (x, y) 0.0-1.0 relative to image rect
    
    # Deprecated/Legacy fields (kept briefly to avoid immediate breakages if referenced elsewhere, but unused by new sync)
    _fixed_hscroll = None  
    _fixed_vscroll = None
    
    # Legacy fields (kept for compatibility but not used)
    _fixed_center = None
    _fixed_center_raw = None
    _fixed_raw_dims = None
    
    @classmethod
    def _get_viewer_ax(cls, viewer):
        """Get the .ax config for a viewer, if available."""
        try:
            # Try to get from viewer's image_data
            if hasattr(viewer, 'image_data') and viewer.image_data:
                if hasattr(viewer.image_data, 'ax_config'):
                    return viewer.image_data.ax_config or {}
            # Try to get from project_tab
            pt = cls._get_project_tab(viewer)
            if pt and hasattr(pt, '_load_ax_for'):
                filepath = getattr(viewer.image_data, 'filepath', None) if hasattr(viewer, 'image_data') else None
                if filepath:
                    return pt._load_ax_for(filepath) or {}
        except Exception:
            pass
        return {}
    
    @classmethod
    def _get_project_tab(cls, viewer):
        """Get the ProjectTab parent of a viewer, if any."""
        try:
            # Walk up the widget tree to find ProjectTab
            parent = viewer.parent()
            while parent is not None:
                # Check if this is a ProjectTab (by duck typing)
                if hasattr(parent, 'all_polygons') and hasattr(parent, '_load_ax_for'):
                    return parent
                parent = parent.parent() if hasattr(parent, 'parent') else None
        except Exception:
            pass
        return None
    
    @classmethod
    def _get_raw_dims(cls, viewer):
        """Get the RAW image dimensions (before .ax transforms) for a viewer."""
        try:
            if hasattr(viewer, 'image_data') and viewer.image_data:
                data = viewer.image_data
                # Support object attribute
                if hasattr(data, 'raw_shape') and data.raw_shape:
                    rs = data.raw_shape
                    return (rs[1], rs[0])  # (w, h)
                
                # Support dict key (just in case)
                if isinstance(data, dict):
                    rs = data.get('raw_shape')
                    if rs: return (rs[1], rs[0])

                # Fall back to original_shape
                if hasattr(data, 'original_shape') and data.original_shape:
                    os = data.original_shape
                    return (os[1], os[0])
                if isinstance(data, dict):
                    os = data.get('original_shape')
                    if os: return (os[1], os[0])

            # Fall back to scene rect (current dims, may be transformed)
            scene = viewer.scene()
            if scene:
                sr = scene.sceneRect()
                return (int(sr.width()), int(sr.height()))
        except Exception:
            pass
        return None
    
    @classmethod
    def _map_point_raw_to_scene(cls, raw_rx, raw_ry, raw_w, raw_h, scene_w, scene_h, ax):
        """
        Map a normalized point from RAW space to SCENE space through .ax transforms.
        
        Args:
            raw_rx, raw_ry: Normalized coordinates in raw image (0-1 range)
            raw_w, raw_h: Raw image dimensions
            scene_w, scene_h: Current scene dimensions
            ax: The .ax configuration dict
        
        Returns:
            (scene_rx, scene_ry): Normalized coordinates in scene space (0-1 range)
        """
        if not ax:
            return (raw_rx, raw_ry)  # No transforms, coordinates are the same
        
        try:
            # Convert normalized to absolute raw coordinates
            x, y = raw_rx * raw_w, raw_ry * raw_h
            
            # Get transformation parameters
            rot = int(ax.get("rotate", 0) or 0) % 360
            crop_rect = ax.get("crop_rect") or None
            crop_ref = ax.get("crop_rect_ref_size") or None
            resize = ax.get("resize") or None
            
            # Determine operation order (mirrors _apply_ax_to_raw logic)
            do_rotate_first = True
            if crop_rect and rot in (90, 180, 270):
                if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                    ref_w = int(crop_ref.get("w", 0)) or 0
                    ref_h = int(crop_ref.get("h", 0)) or 0
                    rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
                    if ref_w > 0 and ref_h > 0:
                        if (ref_w, ref_h) == (raw_w, raw_h):
                            do_rotate_first = False
                        elif (ref_w, ref_h) == (rotated_w, rotated_h):
                            do_rotate_first = True
            
            # Current working dimensions
            cur_w, cur_h = raw_w, raw_h
            
            def apply_rotate():
                nonlocal x, y, cur_w, cur_h
                if rot == 90:
                    x, y = cur_h - y, x
                    cur_w, cur_h = cur_h, cur_w
                elif rot == 180:
                    x, y = cur_w - x, cur_h - y
                elif rot == 270:
                    x, y = y, cur_w - x
                    cur_w, cur_h = cur_h, cur_w
            
            def apply_crop():
                nonlocal x, y, cur_w, cur_h
                if not isinstance(crop_rect, dict) or not crop_rect:
                    return
                # Get crop parameters
                if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                    refW = max(1, int(crop_ref.get("w") or cur_w))
                    refH = max(1, int(crop_ref.get("h") or cur_h))
                else:
                    refW, refH = cur_w, cur_h
                cx = int(crop_rect.get("x", 0))
                cy = int(crop_rect.get("y", 0))
                cw = int(crop_rect.get("width", cur_w))
                ch = int(crop_rect.get("height", cur_h))
                # Scale crop to current dims
                sx = cur_w / float(max(1, refW))
                sy = cur_h / float(max(1, refH))
                cx_scaled = cx * sx
                cy_scaled = cy * sy
                cw_scaled = max(1, cw * sx)
                ch_scaled = max(1, ch * sy)
                # Offset point
                x = x - cx_scaled
                y = y - cy_scaled
                cur_w, cur_h = cw_scaled, ch_scaled
            
            def apply_resize():
                nonlocal x, y, cur_w, cur_h
                if not isinstance(resize, dict) or not resize:
                    return
                old_w, old_h = cur_w, cur_h
                # Calculate new dimensions
                if "px_w" in resize or "px_h" in resize:
                    tw = int(resize.get("px_w", 0) or 0)
                    th = int(resize.get("px_h", 0) or 0)
                    if tw > 0 and th > 0:
                        new_w, new_h = tw, th
                    elif tw > 0:
                        s = tw / float(old_w)
                        new_w, new_h = tw, max(1, int(round(old_h * s)))
                    elif th > 0:
                        s = th / float(old_h)
                        new_h, new_w = th, max(1, int(round(old_w * s)))
                    else:
                        return
                elif "scale" in resize:
                    s = float(resize.get("scale", 100.0)) / 100.0
                    new_w = max(1, int(round(old_w * s)))
                    new_h = max(1, int(round(old_h * s)))
                else:
                    pw = float(resize.get("width", 100.0)) / 100.0
                    ph = float(resize.get("height", 100.0)) / 100.0
                    new_w = max(1, int(round(old_w * pw)))
                    new_h = max(1, int(round(old_h * ph)))
                # Scale point
                x = x * (new_w / float(max(1, old_w)))
                y = y * (new_h / float(max(1, old_h)))
                cur_w, cur_h = new_w, new_h
            
            # Apply transforms in order
            if do_rotate_first:
                if rot: apply_rotate()
                apply_crop()
            else:
                apply_crop()
                if rot: apply_rotate()
            apply_resize()
            
            # Convert back to normalized coordinates
            # Convert back to normalized coordinates
            # FIX: Use actual scene dimensions if available to perform correct denormalization
            # in _apply_fixed_zoom. Fallback to calculated if scene dims not provided.
            final_w = scene_w if (scene_w is not None and scene_w > 0) else max(1, cur_w)
            final_h = scene_h if (scene_h is not None and scene_h > 0) else max(1, cur_h)
            
            scene_rx = x / float(final_w)
            scene_ry = y / float(final_h)
            
            # Don't clamp - let centerOn handle edge cases naturally
            # Clamping to [0,1] can cause visible shifts at corners
            
            return (scene_rx, scene_ry)
        except Exception:
            return (raw_rx, raw_ry)  # Fall back to original on error
    
    @classmethod
    def _map_point_scene_to_raw(cls, scene_rx, scene_ry, scene_w, scene_h, raw_w, raw_h, ax):
        """
        Map a normalized point from SCENE space back to RAW space (inverse .ax transforms).
        
        Args:
            scene_rx, scene_ry: Normalized coordinates in scene (0-1 range)
            scene_w, scene_h: Current scene dimensions
            raw_w, raw_h: Raw image dimensions
            ax: The .ax configuration dict
        
        Returns:
            (raw_rx, raw_ry): Normalized coordinates in raw image space (0-1 range)
        """
        if not ax:
            return (scene_rx, scene_ry)
        
        try:
            # Convert normalized scene to absolute scene coords
            x, y = scene_rx * scene_w, scene_ry * scene_h
            
            # Get transformation parameters
            rot = int(ax.get("rotate", 0) or 0) % 360
            crop_rect = ax.get("crop_rect") or None
            crop_ref = ax.get("crop_rect_ref_size") or None
            resize = ax.get("resize") or None
            
            # Determine operation order
            do_rotate_first = True
            if crop_rect and rot in (90, 180, 270):
                if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                    ref_w = int(crop_ref.get("w", 0)) or 0
                    ref_h = int(crop_ref.get("h", 0)) or 0
                    rotated_w, rotated_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
                    if ref_w > 0 and ref_h > 0:
                        if (ref_w, ref_h) == (raw_w, raw_h):
                            do_rotate_first = False
            
            # Work backwards from scene dimensions
            cur_w, cur_h = scene_w, scene_h
            
            # Calculate what dims would be BEFORE resize
            def calc_pre_resize_dims():
                if not isinstance(resize, dict) or not resize:
                    return cur_w, cur_h
                # We need to figure out what dims were before resize
                # This is complex - we need to reverse the resize calculation
                # For now, approximate using raw dims and transforms
                if do_rotate_first:
                    # After rotate, before crop
                    rw, rh = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
                    # After crop
                    if isinstance(crop_rect, dict) and crop_rect:
                        if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                            refW = max(1, int(crop_ref.get("w") or rw))
                            refH = max(1, int(crop_ref.get("h") or rh))
                        else:
                            refW, refH = rw, rh
                        sx = rw / float(max(1, refW))
                        sy = rh / float(max(1, refH))
                        cw = int(crop_rect.get("width", rw)) * sx
                        ch = int(crop_rect.get("height", rh)) * sy
                        return max(1, cw), max(1, ch)
                    return rw, rh
                else:
                    # After crop, before rotate
                    if isinstance(crop_rect, dict) and crop_rect:
                        if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                            refW = max(1, int(crop_ref.get("w") or raw_w))
                            refH = max(1, int(crop_ref.get("h") or raw_h))
                        else:
                            refW, refH = raw_w, raw_h
                        sx = raw_w / float(max(1, refW))
                        sy = raw_h / float(max(1, refH))
                        cw = int(crop_rect.get("width", raw_w)) * sx
                        ch = int(crop_rect.get("height", raw_h)) * sy
                        # After rotate
                        if rot in (90, 270):
                            return max(1, ch), max(1, cw)
                        return max(1, cw), max(1, ch)
                    # Just rotate
                    if rot in (90, 270):
                        return raw_h, raw_w
                    return raw_w, raw_h
            
            def inverse_resize():
                nonlocal x, y, cur_w, cur_h
                if not isinstance(resize, dict) or not resize:
                    return
                pre_w, pre_h = calc_pre_resize_dims()
                # Scale point back
                x = x * (pre_w / float(max(1, cur_w)))
                y = y * (pre_h / float(max(1, cur_h)))
                cur_w, cur_h = pre_w, pre_h
            
            def inverse_crop():
                nonlocal x, y, cur_w, cur_h
                if not isinstance(crop_rect, dict) or not crop_rect:
                    return
                # Get what dimensions were before crop (depends on rotate order)
                if do_rotate_first:
                    pre_w, pre_h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
                else:
                    pre_w, pre_h = raw_w, raw_h
                # Get crop parameters
                if isinstance(crop_ref, dict) and "w" in crop_ref and "h" in crop_ref:
                    refW = max(1, int(crop_ref.get("w") or pre_w))
                    refH = max(1, int(crop_ref.get("h") or pre_h))
                else:
                    refW, refH = pre_w, pre_h
                cx = int(crop_rect.get("x", 0))
                cy = int(crop_rect.get("y", 0))
                # Scale crop offset to pre-crop dims
                sx = pre_w / float(max(1, refW))
                sy = pre_h / float(max(1, refH))
                cx_scaled = cx * sx
                cy_scaled = cy * sy
                # Add back crop offset
                x = x + cx_scaled
                y = y + cy_scaled
                cur_w, cur_h = pre_w, pre_h
            
            def inverse_rotate():
                nonlocal x, y, cur_w, cur_h
                if rot == 90:
                    x, y = y, cur_w - x
                    cur_w, cur_h = cur_h, cur_w
                elif rot == 180:
                    x, y = cur_w - x, cur_h - y
                elif rot == 270:
                    x, y = cur_h - y, x
                    cur_w, cur_h = cur_h, cur_w
            
            # Apply inverse transforms in REVERSE order
            inverse_resize()
            if do_rotate_first:
                inverse_crop()
                if rot: inverse_rotate()
            else:
                if rot: inverse_rotate()
                inverse_crop()
            
            # Convert to normalized raw coordinates
            raw_rx = x / float(max(1, raw_w))
            raw_ry = y / float(max(1, raw_h))
            
            # Clamp to valid range
            raw_rx = max(0.0, min(1.0, raw_rx))
            raw_ry = max(0.0, min(1.0, raw_ry))
            
            return (raw_rx, raw_ry)
        except Exception:
            return (scene_rx, scene_ry)

    def __init__(self, parent_view, *, min_zoom=0.05, max_zoom=20.0):
        super().__init__(parent_view.viewport())
        self.setObjectName("_ZoomBar")
        self._view = parent_view
        self._min_zoom = float(min_zoom)
        self._max_zoom = float(max_zoom)
        self._block = False

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setStyleSheet("""
            QFrame#_ZoomBar {
                background: rgba(245, 245, 245, 220);
                border: 1px solid rgba(180, 180, 180, 200);
                border-radius: 6px;
            }
            QToolButton { 
                color: black; 
                background: transparent;
                border: none;
                padding: 2px 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 30);
                border-radius: 3px;
            }
            QToolButton:pressed {
                background: rgba(0, 0, 0, 50);
            }
            QToolButton:checked {
                background: rgba(100, 149, 237, 150);
                border-radius: 3px;
            }
            QLabel { 
                color: black; 
                font-size: 10px;
            }
            QSlider::groove:horizontal { 
                height: 3px; 
                background: rgba(0, 0, 0, 80); 
                border-radius: 1px;
            }
            QSlider::handle:horizontal { 
                width: 10px; 
                height: 10px;
                background: #555; 
                border-radius: 5px; 
                margin: -4px 0; 
            }
            QSlider::handle:horizontal:hover {
                background: #333;
            }
        """)

        # --- UI (smaller buttons)
        # Fix button - synchronize zoom across all viewers
        self._btn_fix = QtWidgets.QToolButton(self)
        self._btn_fix.setText("Fix")
        self._btn_fix.setCheckable(True)
        self._btn_fix.setChecked(_ZoomBar._zoom_sync_enabled)
        self._btn_fix.setToolTip(
            "Fix zoom level across all viewers in the project.\n"
            "When enabled, changing zoom in any viewer syncs all others."
        )
        
        self._btn_minus = QtWidgets.QToolButton(self); self._btn_minus.setText("−")
        self._btn_plus  = QtWidgets.QToolButton(self); self._btn_plus.setText("+")
        self._btn_100   = QtWidgets.QToolButton(self); self._btn_100.setText("100%")
        self._btn_fit   = QtWidgets.QToolButton(self); self._btn_fit.setText("Fit")
        
        # Make buttons smaller
        for btn in [self._btn_fix, self._btn_minus, self._btn_plus, self._btn_100, self._btn_fit]:
            btn.setFixedHeight(20)
            btn.setMinimumWidth(24)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self._slider.setRange(0, 1000)   # log-scale mapping
        self._slider.setFixedWidth(100)  # smaller slider
        self._slider.setFixedHeight(16)
        
        self._lbl = QtWidgets.QLabel("100%")
        self._lbl.setMinimumWidth(36)
        self._lbl.setAlignment(QtCore.Qt.AlignCenter)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)  # smaller margins
        lay.setSpacing(3)  # tighter spacing
        lay.addWidget(self._btn_fix)
        lay.addSpacing(2)
        lay.addWidget(self._btn_minus)
        lay.addWidget(self._slider)
        lay.addWidget(self._btn_plus)
        lay.addSpacing(4)
        lay.addWidget(self._lbl)
        lay.addWidget(self._btn_100)
        lay.addWidget(self._btn_fit)

        # Initialize with ACTUAL zoom level from parent view
        # The view is likely already fitted or set up by the time this is called
        current_zoom = 1.0
        try:
            if hasattr(self._view, "current_zoom_factor"):
                current_zoom = self._view.current_zoom_factor()
        except:
            pass

        self._set_slider_from_zoom(current_zoom)
        self._update_label(current_zoom)

        # --- Auto-hide timer ---
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(1500)  # hide after 1.5 seconds of no interaction
        self._hide_timer.timeout.connect(self._do_hide)
        
        # Start hidden
        self.hide()
        
        # place immediately
        self.reposition()

        # signals
        self._btn_fix.clicked.connect(self._on_fix_clicked)
        self._btn_minus.clicked.connect(lambda: self._nudge(-1))
        self._btn_plus.clicked.connect(lambda: self._nudge(+1))
        self._btn_100.clicked.connect(lambda: self.set_zoom(1.0, emit=True))
        self._btn_fit.clicked.connect(self._fit_clicked)
        self._slider.valueChanged.connect(self._on_slider_changed)
        
        # Reset hide timer on any interaction
        self._slider.sliderPressed.connect(self._reset_hide_timer)
        self._slider.sliderReleased.connect(self._start_hide_timer)

    # ---------- Fix button / zoom + pan sync ----------
    def _on_fix_clicked(self):
        """Toggle zoom/pan sync across all viewers."""
        _ZoomBar._zoom_sync_enabled = self._btn_fix.isChecked()
        
        # Update all other zoom bars' Fix button state
        self._sync_fix_button_state()
        
        if _ZoomBar._zoom_sync_enabled:
            # When enabling, capture current zoom and scroll positions
            try:
                _ZoomBar._fixed_zoom = self._view.current_zoom_factor()
                self._store_scroll_position(self._view)
                
                # Sync all viewers to this zoom level
                self._sync_zoom_to_all_viewers(_ZoomBar._fixed_zoom)
            except Exception:
                _ZoomBar._fixed_zoom = 1.0
                _ZoomBar._fixed_hscroll = 0.5
                _ZoomBar._fixed_vscroll = 0.5
            logging.info(f"View fixed at {_ZoomBar._fixed_zoom*100:.0f}% zoom - will persist across root changes")
        else:
            # Clear fixed state when disabling
            _ZoomBar._fixed_zoom = None
            _ZoomBar._fixed_hscroll = None
            _ZoomBar._fixed_vscroll = None
            _ZoomBar._fixed_center = None
            _ZoomBar._fixed_center_raw = None
            _ZoomBar._fixed_raw_dims = None
            logging.info("View sync disabled")
    
    @staticmethod
    def _get_view_center_ratio(viewer):
        """Get the center of the visible area as a ratio (0-1) of the IMAGE rect (not scene)."""
        try:
            # 1. Get the Image Item's bounding rect (the content)
            img_rect = viewer._get_image_rect()
            if img_rect.isEmpty():
                return (0.5, 0.5)
                
            # 2. Get the center of the viewport in Scene coordinates
            #    (map the center pixel of the widget to the scene)
            vp_center = viewer.viewport().rect().center()
            scene_center = viewer.mapToScene(vp_center)
            
            # 3. Calculate position relative to the Image Rect
            #    (0,0) = top-left of image, (1,1) = bottom-right
            rx = (scene_center.x() - img_rect.left()) / img_rect.width()
            ry = (scene_center.y() - img_rect.top()) / img_rect.height()
            
            return (rx, ry)
        except Exception:
            return (0.5, 0.5)

    @classmethod
    def _store_scroll_position(cls, viewer):
        """Store the current view center as a normalized ratio (0-1)."""
        if cls._applying_fixed_zoom:
            return  # Don't update while applying
            
        try:
            # Store normalized center instead of scroll bar positions
            cls._fixed_center_norm = cls._get_view_center_ratio(viewer)
            
            # Legacy fields (just in case)
            cls._fixed_hscroll = 0.5
            cls._fixed_vscroll = 0.5
            
            logging.debug(f"[_store_scroll_position] Stored center: {cls._fixed_center_norm}")
            
        except Exception as e:
            logging.debug(f"[_store_scroll_position] Failed: {e}")
    
    @classmethod
    def update_fixed_center(cls, source_viewer):
        """Update the fixed scroll positions and sync to all other viewers."""
        if not cls._zoom_sync_enabled:
            return
        if cls._syncing:
            return
        if cls._applying_fixed_zoom:
            return
        
        try:
            # Sync pan to all other viewers immediately using normalized center
            cls._syncing = True
            try:
                # 1. Store the new center from source
                cls._store_scroll_position(source_viewer)
                center_norm = cls._fixed_center_norm
                
                if not center_norm:
                    return

                rx, ry = center_norm
                
                # 2. Apply to all other viewers
                for viewer in cls._get_all_viewers():
                    if viewer == source_viewer:
                        continue
                    try:
                        img_rect = viewer._get_image_rect()
                        if not img_rect.isEmpty():
                            x = img_rect.left() + rx * img_rect.width()
                            y = img_rect.top() + ry * img_rect.height()
                            viewer.centerOn(x, y)
                    except Exception:
                        pass
            finally:
                cls._syncing = False
        except Exception:
            pass
    
    def _sync_fix_button_state(self):
        """Sync the Fix button checked state across all zoom bars."""
        try:
            for viewer in self._get_all_viewers():
                zb = getattr(viewer, "_zoombar", None)
                if zb and zb != self and hasattr(zb, "_btn_fix"):
                    zb._btn_fix.setChecked(_ZoomBar._zoom_sync_enabled)
        except Exception:
            pass
    
    # Short-lived cache for _get_all_viewers (avoids repeated widget tree walks)
    _viewers_cache = None
    _viewers_cache_time = 0
    _VIEWERS_CACHE_TTL_MS = 500  # 500ms TTL

    @staticmethod
    def _get_all_viewers():
        """Get all ImageViewer instances in the application (cached 500ms)."""
        import sip
        
        now = time.time() * 1000
        if (_ZoomBar._viewers_cache is not None
                and now - _ZoomBar._viewers_cache_time < _ZoomBar._VIEWERS_CACHE_TTL_MS):
            # Validate cached entries are still alive
            valid = {v for v in _ZoomBar._viewers_cache if not sip.isdeleted(v)}
            if valid:
                return valid
        
        viewers = set()
        try:
            # Prefer the parent's viewer_widgets if available (much faster)
            for tlw in QtWidgets.QApplication.topLevelWidgets():
                if sip.isdeleted(tlw): continue
                
                # Look for ProjectTab-like objects that have viewer_widgets
                if hasattr(tlw, 'viewer_widgets'):
                    for wdict in (tlw.viewer_widgets or []):
                        v = wdict.get('viewer') if isinstance(wdict, dict) else None
                        if v is not None and isinstance(v, ImageViewer) and not sip.isdeleted(v):
                            viewers.add(v)
                # Also check for tabs in tab widgets
                if hasattr(tlw, 'findChildren'):
                    try:
                        for tab_widget in tlw.findChildren(QtWidgets.QTabWidget):
                            if sip.isdeleted(tab_widget): continue
                            for i in range(tab_widget.count()):
                                tab = tab_widget.widget(i)
                                if tab and not sip.isdeleted(tab) and hasattr(tab, 'viewer_widgets'):
                                    for wdict in (tab.viewer_widgets or []):
                                        v = wdict.get('viewer') if isinstance(wdict, dict) else None
                                        if v is not None and isinstance(v, ImageViewer) and not sip.isdeleted(v):
                                            viewers.add(v)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Fallback: direct search (only if we found nothing)
        if not viewers:
            try:
                for tlw in QtWidgets.QApplication.topLevelWidgets():
                    if sip.isdeleted(tlw): continue
                    if isinstance(tlw, ImageViewer):
                        viewers.add(tlw)
                    try:
                        viewers.update([v for v in tlw.findChildren(ImageViewer) if not sip.isdeleted(v)])
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Update cache
        _ZoomBar._viewers_cache = viewers
        _ZoomBar._viewers_cache_time = now
        return viewers
    
    def _sync_zoom_to_all_viewers(self, zoom_factor):
        """Sync the given zoom factor and scroll positions to all viewers."""
        if _ZoomBar._syncing:
            return  # Prevent recursive sync
        if _ZoomBar._applying_fixed_zoom:
            return
        
        # Update the fixed zoom and center
        if _ZoomBar._zoom_sync_enabled:
            _ZoomBar._fixed_zoom = zoom_factor
            _ZoomBar._store_scroll_position(self._view)
            
        center_norm = _ZoomBar._fixed_center_norm
        
        _ZoomBar._syncing = True
        try:
            viewers = self._get_all_viewers()
            
            # Batch updates: disable updates on all viewers first
            import sip
            valid_viewers = [v for v in viewers if not sip.isdeleted(v)]
            
            for viewer in valid_viewers:
                if viewer != self._view:
                    try:
                        # Safeguard against potential C++ object deletion issues
                        if not sip.isdeleted(viewer):
                            viewer.setUpdatesEnabled(False)
                    except Exception:
                        pass
            
            # Apply zoom and center to all viewers
            for viewer in valid_viewers:
                if viewer == self._view:
                    continue  # Skip self
                try:
                    if sip.isdeleted(viewer): continue
                    
                    # Suppress sync timer to prevent infinite ping-pong loops
                    viewer._suppress_sync = True
                    try:
                        # Apply zoom
                        if hasattr(viewer, "set_zoom_factor"):
                            viewer.set_zoom_factor(zoom_factor, anchor=QtWidgets.QGraphicsView.AnchorViewCenter)
                        
                        # Apply center
                        if center_norm:
                            rx, ry = center_norm
                            # Validate coordinates to prevent NaN/Inf crashes
                            if not (math.isfinite(rx) and math.isfinite(ry)):
                                continue
                                
                            img_rect = viewer._get_image_rect()
                            if not img_rect.isEmpty():
                                x = img_rect.left() + rx * img_rect.width()
                                y = img_rect.top() + ry * img_rect.height()
                                
                                # Validate final coordinates
                                if math.isfinite(x) and math.isfinite(y):
                                    viewer.centerOn(x, y)
                        
                        # Update the zoom bar UI if present
                        zb = getattr(viewer, "_zoombar", None)
                        if zb and not sip.isdeleted(zb):
                            zb._set_slider_from_zoom(zoom_factor)
                            zb._update_label(zoom_factor)
                    finally:
                        if not sip.isdeleted(viewer):
                            viewer._suppress_sync = False
                except Exception:
                    pass
            
            # Re-enable updates
            for viewer in valid_viewers:
                if viewer != self._view:
                    try:
                        if not sip.isdeleted(viewer):
                            viewer.setUpdatesEnabled(True)
                    except Exception:
                        pass
        finally:
            _ZoomBar._syncing = False
    
    @classmethod
    def apply_fixed_zoom_to_viewer(cls, viewer):
        """
        Apply the fixed zoom level and scroll positions to a viewer after image load.
        Called from set_image when Fix is enabled.
        """
        if not cls._zoom_sync_enabled or cls._fixed_zoom is None:
            return False
        
        if cls._syncing:
            return False
        
        try:
            # Apply zoom
            if hasattr(viewer, "set_zoom_factor"):
                viewer.set_zoom_factor(cls._fixed_zoom, anchor=QtWidgets.QGraphicsView.AnchorViewCenter)
            
            # Process events so scroll bar ranges update after zoom
            QtWidgets.QApplication.processEvents()
            
            # Apply scroll positions (using normalized center)
            if cls._fixed_center_norm:
                rx, ry = cls._fixed_center_norm
                img_rect = viewer._get_image_rect()
                if not img_rect.isEmpty():
                    x = img_rect.left() + rx * img_rect.width()
                    y = img_rect.top() + ry * img_rect.height()
                    viewer.centerOn(x, y)
            elif cls._fixed_hscroll is not None and cls._fixed_vscroll is not None:
                # Fallback to legacy scroll ratio if center not available
                hs = viewer.horizontalScrollBar()
                vs = viewer.verticalScrollBar()
                
                h_range = hs.maximum() - hs.minimum()
                v_range = vs.maximum() - vs.minimum()
                
                if h_range > 0:
                    hs.setValue(hs.minimum() + int(cls._fixed_hscroll * h_range))
                if v_range > 0:
                    vs.setValue(vs.minimum() + int(cls._fixed_vscroll * v_range))
            
            # Update zoom bar UI
            zb = getattr(viewer, "_zoombar", None)
            if zb:
                zb._set_slider_from_zoom(cls._fixed_zoom)
                zb._update_label(cls._fixed_zoom)
            return True
        except Exception as e:
            logging.debug(f"Failed to apply fixed zoom: {e}")
            return False

    # ---------- auto-hide ----------
    def _do_hide(self):
        self.hide()
    
    def _start_hide_timer(self):
        self._hide_timer.start()
    
    def _reset_hide_timer(self):
        self._hide_timer.stop()
    
    def show_briefly(self):
        """Show the zoom bar and start the auto-hide timer."""
        if not self.isVisible():
            self.reposition()
        self.show()
        self._start_hide_timer()
    
    def enterEvent(self, event):
        """Stop hiding when mouse enters, change cursor to pointer."""
        self._hide_timer.stop()
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Start hide timer when mouse leaves, restore cursor."""
        self._start_hide_timer()
        self.unsetCursor()
        super().leaveEvent(event)

    # ---------- placement ----------
    def reposition(self):
        vp = self._view.viewport()
        if not vp:
            return
        m = 8
        self.adjustSize()
        s = self.sizeHint()
        x = max(m, vp.width() - s.width() - m)
        y = max(m, vp.height() - s.height() - m)
        self.setGeometry(x, y, s.width(), s.height())

    # ---------- zoom mapping ----------
    @staticmethod
    def _zoom_to_slider(z, zmin, zmax):
        z = max(zmin, min(zmax, float(z)))
        if zmin <= 0:
            zmin = 0.01
        t = (math.log(z) - math.log(zmin)) / (math.log(zmax) - math.log(zmin))
        return int(round(1000.0 * max(0.0, min(1.0, t))))

    @staticmethod
    def _slider_to_zoom(pos, zmin, zmax):
        pos = max(0, min(1000, int(pos)))
        t = pos / 1000.0
        return math.exp((1.0 - t) * math.log(zmin) + t * math.log(zmax))

    def _set_slider_from_zoom(self, z):
        self._block = True
        self._slider.setValue(self._zoom_to_slider(z, self._min_zoom, self._max_zoom))
        self._block = False

    def _on_slider_changed(self, v):
        if self._block:
            return
        z = self._slider_to_zoom(v, self._min_zoom, self._max_zoom)
        self._update_label(z)
        self.zoomChanged.emit(z)
        # Sync to other viewers if enabled
        if _ZoomBar._zoom_sync_enabled:
            self._sync_zoom_to_all_viewers(z)

    def _update_label(self, z):
        self._lbl.setText(f"{z*100.0:0.0f}%")

    def _nudge(self, step):
        self._slider.setValue(self._slider.value() + (30 * step))

    def _fit_clicked(self):
        try:
            self._view.zoom_out_to_fit()
            cur = self._view.current_zoom_factor()
        except Exception:
            cur = 1.0
        self._set_slider_from_zoom(cur)
        self._update_label(cur)
        # Sync to other viewers if enabled
        if _ZoomBar._zoom_sync_enabled:
            self._sync_zoom_to_all_viewers(cur)

    # ---------- external API ----------
    def set_zoom(self, z, *, emit=False, sync=True):
        z = max(self._min_zoom, min(self._max_zoom, float(z)))
        self._set_slider_from_zoom(z)
        self._update_label(z)
        if emit:
            self.zoomChanged.emit(z)
        # Sync to other viewers if enabled
        if sync and _ZoomBar._zoom_sync_enabled:
            self._sync_zoom_to_all_viewers(z)


# ---- installation helper (non-invasive): attach to an existing ImageViewer ----
def attach_zoom_bar(viewer):
    """
    Installs a _ZoomBar on top of `viewer` and wires it to:
      - mouse wheel zoom (shows bar briefly)
      - programmatic zoom (fit_to_window / setTransform)
      - manual slider changes (adjust view to absolute zoom)
    The bar auto-hides after 1.5 seconds of no interaction.
    """
    if getattr(viewer, "_zoombar", None) is not None:
        return viewer._zoombar

    zb = _ZoomBar(viewer)
    viewer._zoombar = zb

    def current_zoom_factor():
        tr = viewer.transform()
        return max(1e-6, math.hypot(tr.m11(), tr.m12()))
    viewer.current_zoom_factor = current_zoom_factor

    def set_zoom_factor(z, anchor=QtWidgets.QGraphicsView.AnchorUnderMouse):
        z = max(0.01, min(50.0, float(z)))
        prev_anchor = viewer.transformationAnchor()
        viewer.setTransformationAnchor(anchor)
        viewer.resetTransform()
        viewer.scale(z, z)
        viewer.setTransformationAnchor(prev_anchor)
        if getattr(viewer, "_zoombar", None):
            viewer._zoombar.set_zoom(z)
    viewer.set_zoom_factor = set_zoom_factor

    zb.zoomChanged.connect(lambda z: viewer.set_zoom_factor(z, anchor=QtWidgets.QGraphicsView.AnchorViewCenter))

    old_resize = viewer.resizeEvent
    def _resized(ev):
        try:
            zb.reposition()
        except Exception:
            pass
        if callable(old_resize):
            old_resize(ev)
    viewer.resizeEvent = _resized

    old_wheel = viewer.wheelEvent
    def _wheel(ev):
        if callable(old_wheel):
            old_wheel(ev)
        # Update zoom bar UI immediately (lightweight operation)
        try:
            cur = viewer.current_zoom_factor()
            zb._block = True  # Prevent slider from triggering zoom changes
            zb._set_slider_from_zoom(cur)
            zb._update_label(cur)
            zb._block = False
            zb.show_briefly()
        except Exception:
            pass
    viewer.wheelEvent = _wheel

    viewer._zoom_sync_timer = QtCore.QTimer(viewer)
    viewer._zoom_sync_timer.setSingleShot(True)
    viewer._zoom_sync_timer.setInterval(0)

    viewer._suppress_sync = False
    
    def _sync_bar():
        try:
            # Only update the local UI, do NOT broadcast sync (avoid loops)
            zb.set_zoom(viewer.current_zoom_factor(), sync=False)
        except Exception:
            pass

    viewer._zoom_sync_timer.timeout.connect(_sync_bar)

    old_setTransform = viewer.setTransform
    def _setTransform(*args, **kwargs):
        r = old_setTransform(*args, **kwargs)
        if not getattr(viewer, "_suppress_sync", False):
            viewer._zoom_sync_timer.start()
        return r
    viewer.setTransform = _setTransform

    zb.reposition()
    zb.hide()  # Start hidden, will show on interaction
    return zb
