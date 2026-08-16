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
# Scene stacking order (QGraphicsItem Z values)
# -------------------------------------------------------------------
# These were all implicitly 0 and resolved by INSERTION ORDER, which worked
# only as long as nothing was ever inserted between the base image and the
# annotations. The high-resolution zoom overlay is exactly such an item: it is
# added later, is opaque, and sits at 0.5 -- so with polygons left at the
# default 0 it covered every polygon and point the moment a zoom refined the
# view. The symptom was "polygons vanish when I zoom in and come back when I
# zoom out", because zooming back out clears the overlay.
#
# Making the order explicit removes the dependence on insertion order.
IMAGE_Z          = 0.0     # base (preview) pixmap
HIGHRES_TILE_Z   = 0.5     # sharpened viewport tile -- above the image...
POLYGON_Z        = 1.0     # ...and BELOW every annotation
TEMP_DRAWING_Z   = 1e6     # in-progress rubber-band polygon
LABEL_Z          = 1e7     # name labels, always on top


class PolygonPropertiesDialog(QtWidgets.QDialog):
    """Property (key/value) editor for a single polygon/point.

    Populates from `parent.all_polygons[group][filepath]['properties']`
    (see ImageViewer.edit_polygon_properties) and hands back a plain dict on
    accept. Purely metadata: no repaint or geometry work happens here, so
    there is nothing performance-sensitive about this dialog.
    """

    def __init__(self, properties=None, title="Edit Properties", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(420, 360)

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Property", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        for k, v in (properties or {}).items():
            self._append_row(str(k), "" if v is None else str(v))

        btn_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add Property")
        self.remove_btn = QtWidgets.QPushButton("Remove Selected")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(lambda: self._append_row("", ""))
        self.remove_btn.clicked.connect(self._remove_selected)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_row(self, key, value):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    @staticmethod
    def _coerce(value_str):
        """Cast a string back to int/float where it round-trips cleanly, else
        keep it as a string. Prevents numeric DBF columns being silently
        coerced to text on export (shapefile_io.write_dbf infers column type
        from the Python type of the first non-null value it sees)."""
        s = value_str.strip()
        if s == "":
            return ""
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return value_str

    def get_properties(self):
        """Return the edited {key: value} dict, empty keys stripped."""
        out = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            key = (key_item.text().strip() if key_item else "")
            if not key:
                continue
            val = (val_item.text() if val_item else "")
            out[key] = self._coerce(val)
        return out


# -------------------------------------------------------------------
# Editable overlay items
# -------------------------------------------------------------------
class EditablePolygonItem(QtWidgets.QGraphicsObject):
    polygon_modified = QtCore.pyqtSignal()

    # -- Label suppression threshold --
    # When the view scale falls below this value labels are hidden unless
    # the item is hovered or selected.  This avoids the rasterisation cost
    # of drawing thousands of labels when zoomed out.
    _LABEL_SCALE_THRESHOLD = 0.35

    def __init__(self, polygon, name="", is_rgb=False, parent=None, is_mask_polygon=False):
        super(EditablePolygonItem, self).__init__(parent)
        self._polygon = polygon
        self.name = name
        self.is_rgb = is_rgb  # Determines polygon appearance
        self.is_mask_polygon = is_mask_polygon  # If True, draw with solid fill

        # ---- Filtering / coloring (PolygonManager "Filter & Color") ----
        #: Arbitrary imported/user attributes for this polygon, mirrored from
        #: all_polygons[group][filepath]['properties']. Populated by whoever
        #: draws/loads the item; never required to be non-empty.
        self.properties = {}
        #: QColor override for the outline, or None to use the default
        #: red(RGB)/blue(multispectral) pen. Set by PolygonManager's filter
        #: rules; cleared by "Clear Filters".
        self.current_color = None
        #: True when a filter rule hid this polygon. Distinct from Qt's own
        #: isVisible(): the LOD batch path (PolygonTileItem) and the "Show
        #: Polys" checkbox both toggle visibility too, and a filtered-out
        #: polygon must stay hidden through either of those without losing
        #: track of WHY, so it can come back when the filter is cleared.
        self._filtered_hidden = False

        # ---- Locking (PolygonManager "Lock Polygons") ----
        #: When True, mousePressEvent blocks drag-start and offers to unlock.
        #: New items inherit the owning viewer's global_polygons_locked at
        #: creation time (see ImageViewer.add_polygon_to_scene).
        self.is_locked = False

        # Explicitly above the high-res zoom overlay -- see the Z constants at
        # the top of this module. Left at the default 0 these were covered by
        # the overlay as soon as a zoom refined the view.
        self.setZValue(POLYGON_Z)

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
        #: True when this item was drawn from a COARSE pyramid level rather
        #: than the polygon's real coordinates (see polygon_lod). Such an item
        #: is a PICTURE of the polygon, not the polygon: its vertices must
        #: never be written back to storage, or the decimated outline would
        #: replace the real one. `ProjectTab.update_all_polygons` skips these,
        #: and any attempt to drag one upgrades it to full geometry first.
        self.is_lod_geometry = False
        # Backing fields for the show_label / _hover_showing_label PROPERTIES
        # below. They must exist before _invalidate_geometry_cache() runs, and
        # must be set directly (not via the properties) here, because the
        # geometry cache does not exist yet at this point.
        # Labels default to OFF.
        #
        # A label makes the item's boundingRect reserve `len(name)*50 + 100` by
        # 250 scene units for text that may never be drawn (see
        # _invalidate_geometry_cache). On real crown names that is a ~1150x350
        # box around a 40x40 polygon, so Qt's culling hands us far more items
        # than are actually on screen AND each one rasterises text. Measured on
        # 10000 generated circles with realistic names, ms/frame:
        #
        #     zoom   labels ON   labels OFF
        #     0.25       25.7        28.4
        #     0.40       51.2        10.0     <-- 5.1x
        #     0.70       27.1         9.4     <-- 2.9x
        #     1.00       18.9         6.3     <-- 3.0x
        #
        # ...and at zoom 0.25 labels-on painted 967 items where labels-off
        # painted 696: 39% of the work was for polygons that were not visible.
        # Turning labels on is a deliberate act ("Show Labels" in the polygon
        # manager), not something a user with 10000 polygons should pay for by
        # default.
        self._show_label = False
        self._hover_label = False # Temp visibility on hover
        self.label_offset = QtCore.QPointF(10, -10)
        
        # Cache for performance - avoid recalculating on every paint
        self._cached_img_size = None
        self._cached_res_boost = 1.0

        # ---- Geometry cache (invalidated via polygon property setter) ----
        self._cached_brect = None          # full bounding rect (with label)
        self._cached_brect_drag = None     # tight bounding rect (no label)
        self._cached_shape = None          # QPainterPath from addPolygon

        # ---- Paint state cache (avoid per-frame allocation) ----
        self._pen_base = None
        self._pen_highlight = None
        self._pen_label = QtGui.QPen(QtCore.Qt.red)
        self._cached_font = None
        self._cached_font_px = -1

        # Pre-compute cached geometry
        self._invalidate_geometry_cache()

    # ---- label-visibility properties ----
    # These are PROPERTIES rather than plain attributes because boundingRect
    # now depends on them (see _invalidate_geometry_cache): the label area is
    # only reserved when a label can actually be painted. Anything that flips
    # these must therefore go through prepareGeometryChange(), or Qt keeps
    # using the stale rect and leaves label artifacts behind. Callers assign
    # `item.show_label = ...` directly all over this file and in project_tab,
    # so making it a property keeps every one of those correct without
    # touching them.
    @property
    def show_label(self):
        return self._show_label

    @show_label.setter
    def show_label(self, value):
        value = bool(value)
        if value == getattr(self, "_show_label", None):
            return
        self.prepareGeometryChange()
        self._show_label = value
        self._update_bounding_rect()

    @property
    def _hover_showing_label(self):
        return self._hover_label

    @_hover_showing_label.setter
    def _hover_showing_label(self, value):
        value = bool(value)
        if value == getattr(self, "_hover_label", None):
            return
        self.prepareGeometryChange()
        self._hover_label = value
        self._update_bounding_rect()

    # ---- polygon property: invalidate caches when geometry changes ----
    @property
    def polygon(self):
        return self._polygon

    @polygon.setter
    def polygon(self, new_poly):
        self.prepareGeometryChange()
        self._polygon = new_poly
        self._invalidate_geometry_cache()

    #: Below this vertex count, decimation cannot pay for itself.
    _SIMPLIFY_MIN_POINTS = 64

    def _display_polygon(self, scale):
        """A vertex-DECIMATED copy of the polygon, for PAINTING ONLY.

        This is what QGIS does by default (QgsVectorSimplifyMethod): drop
        vertices that land within ~one device pixel of the previous one, since
        they cannot be distinguished on screen anyway.

        Measured on the real project -- 2333 crown polygons carrying
        1,010,806 vertices (mean 433 each, max 2312) -- drawing every vertex at
        every zoom cost:

            zoom 0.02   555 ms/frame ( 1.8 FPS)   ->  95% of vertices dropped
            zoom 0.05   228 ms/frame ( 4.4 FPS)   ->  89% dropped
            zoom 0.15    40 ms/frame (25.0 FPS)   ->  73% dropped

        i.e. 2.2x-3.7x faster panning for pixel-identical output.

        `self._polygon` itself is NEVER modified: statistics, export, hit
        testing and shape() all keep the exact geometry. Only the painted path
        is reduced.
        """
        pts = self._pts_np
        if pts is None or len(pts) < self._SIMPLIFY_MIN_POINTS or scale <= 0:
            return self._polygon

        # EARLY-OUT, before any maths.
        #
        # The expensive part of decimating is NOT the numpy pass -- it is
        # rebuilding a QPolygonF from a Python list of QPointF. And that cost
        # is paid on FIRST PAINT OF EACH ITEM, which during a pan is almost
        # every paint: instrumenting a 30-frame pan measured 1556 rebuilds
        # against only 608 cache hits, because panning continuously brings
        # previously-unseen polygons into view. A naive "decimate whenever
        # anything can be dropped" rule therefore made zoomed-in panning
        # SLOWER (10.0 -> 20.5 ms/frame at zoom 0.40).
        #
        # `_outline_extent * scale` is roughly the polygon's outline length in
        # device pixels; the multiplier says how much denser than that the
        # vertices must be before decimating pays for the rebuild.
        #
        # It was 4x, chosen when a thick pen made every vertex expensive to
        # stroke. Moving to a 1px pen (see paint()) cut per-vertex draw cost by
        # ~14x, so the break-even moved with it -- at 4x, decimation became a
        # net LOSS when zoomed in. Measured on the BCI crown map, ms/frame:
        #
        #     zoom    no LOD    x4     x16    x64
        #     0.02      86.8   55.1   52.8   60.7
        #     0.05      46.5   40.4   35.5   43.1
        #     0.15       8.5   18.7   12.9    7.8    <-- x4/x16 lose here
        #
        # 32x keeps most of the overview win without the zoomed-in penalty.
        if len(pts) <= 32.0 * self._outline_extent * scale:
            return self._polygon

        # One device pixel expressed in scene units. Bucketed to powers of two
        # so ordinary zooming reuses a cached result instead of rebuilding on
        # every wheel notch.
        tol = 1.0 / scale
        try:
            bucket = int(math.floor(math.log2(tol)))
        except (ValueError, OverflowError):
            return self._polygon

        cached = self._simplified_cache
        if cached is not None and cached[0] == bucket:
            return cached[1]

        tol_b = 2.0 ** bucket
        try:
            import numpy as _np
            q = _np.floor(pts / tol_b)
            keep = _np.empty(len(q), dtype=bool)
            keep[0] = True
            keep[1:] = (q[1:] != q[:-1]).any(axis=1)
            keep[-1] = True
            reduced = pts[keep]
            if len(reduced) < 4 or len(reduced) >= len(pts):
                simplified = self._polygon
            else:
                simplified = QtGui.QPolygonF(
                    [QtCore.QPointF(float(x), float(y)) for x, y in reduced])
        except Exception:
            simplified = self._polygon

        self._simplified_cache = (bucket, simplified)
        return simplified

    def _invalidate_geometry_cache(self):
        """Recompute cached boundingRect and shape from the current polygon."""
        poly = self._polygon
        raw = poly.boundingRect()

        # Vertex array backing _display_polygon, plus its cache. Built once per
        # geometry change rather than per paint.
        self._simplified_cache = None
        self._pts_np = None
        self._outline_extent = raw.width() + raw.height()
        try:
            n = poly.size()
            if n >= self._SIMPLIFY_MIN_POINTS:
                import numpy as _np
                arr = _np.empty((n, 2), dtype=_np.float64)
                for i in range(n):
                    p = poly.at(i)
                    arr[i, 0] = p.x()
                    arr[i, 1] = p.y()
                self._pts_np = arr
        except Exception:
            self._pts_np = None

        self._update_bounding_rect()

        # Cached shape (QPainterPath)
        path = QtGui.QPainterPath()
        path.addPolygon(poly)
        self._cached_shape = path

        # Also invalidate pen cache (color depends on is_rgb/mask)
        self._pen_base = None

    def _update_bounding_rect(self):
        """Recompute only the bounding rect (label visibility change)."""
        raw = self._polygon.boundingRect()
        margin_tight = 10
        self._cached_brect_drag = raw.adjusted(-margin_tight, -margin_tight,
                                               margin_tight, margin_tight)

        full = QtCore.QRectF(raw)
        if self.name and (self._show_label or self._hover_showing_label):
            label_width = len(self.name) * 50 + 100
            label_height = 250
            label_rect = QtCore.QRectF(
                full.topRight() + self.label_offset,
                QtCore.QSizeF(label_width, label_height)
            )
            full = full.united(label_rect)
        margin = 50
        self._cached_brect = full.adjusted(-margin, -margin, margin, margin)

    def _get_image_size(self):
        """Get image dimensions, cached to avoid scene iteration on every paint."""
        scn = self.scene()
        if not scn:
            return None, None
        
        # Return cached value if available
        if self._cached_img_size is not None:
            return self._cached_img_size
        
        # Prefer the main viewer base image item
        views = scn.views() if scn else []
        if views and hasattr(views[0], '_image') and views[0]._image is not None:
            try:
                pm = views[0]._image.pixmap()
                if not pm.isNull():
                    self._cached_img_size = (pm.width(), pm.height())
                    long_side = max(self._cached_img_size)
                    self._cached_res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))
                    return self._cached_img_size
            except Exception:
                pass

        # Fallback: Find base pixmap item (zValue == 0)
        for it in scn.items():
            if isinstance(it, QtWidgets.QGraphicsPixmapItem) and getattr(it, 'zValue', lambda: 0)() == 0:
                pm = it.pixmap()
                if not pm.isNull():
                    self._cached_img_size = (pm.width(), pm.height())
                    # Calculate res_boost once
                    long_side = max(self._cached_img_size)
                    self._cached_res_boost = min(3.0, max(1.0, (long_side / 2048.0) ** 0.5))
                    return self._cached_img_size
        return None, None

    def boundingRect(self):
        # During active dragging, return the FULL cached rect (not the tight
        # one).  Keeping the bounding rect stable prevents Qt from thinking
        # the item has moved outside its old rect, which would force
        # FullViewportUpdate to erase label ghosts.  Since labels are not
        # drawn during drag the extra area is harmless.
        if self.is_moving:
            return self._cached_brect if self._cached_brect else self._cached_brect_drag
        return self._cached_brect if self._cached_brect else QtCore.QRectF()

    def shape(self):
        if self._cached_shape is not None:
            return self._cached_shape
        path = QtGui.QPainterPath()
        path.addPolygon(self._polygon)
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

    def _ensure_pens(self):
        """Lazily build cached QPen objects.  Called once or after geometry invalidation."""
        if self._pen_base is not None:
            return
        if self.current_color is not None:
            # A filter rule assigned an explicit color -- it wins over both
            # the mask-polygon orange and the default red/blue, so the user's
            # rule is always visibly in effect.
            self._pen_base = QtGui.QPen(self.current_color)
            self._fill_base = (QtGui.QColor(self.current_color.red(), self.current_color.green(),
                                             self.current_color.blue(), 80)
                                if self.is_mask_polygon else QtCore.Qt.transparent)
            self._pen_highlight = QtGui.QPen(QtCore.Qt.magenta)
            self._fill_highlight = (QtGui.QColor(255, 0, 255, 100) if self.is_mask_polygon
                                     else QtCore.Qt.transparent)
        elif self.is_mask_polygon:
            self._pen_base = QtGui.QPen(QtGui.QColor(255, 165, 0))
            self._fill_base = QtGui.QColor(255, 165, 0, 80)
            self._pen_highlight = QtGui.QPen(QtCore.Qt.magenta)
            self._fill_highlight = QtGui.QColor(255, 0, 255, 100)
        else:
            base_color = QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue
            self._pen_base = QtGui.QPen(base_color)
            self._fill_base = QtCore.Qt.transparent
            self._pen_highlight = QtGui.QPen(QtCore.Qt.magenta)
            self._fill_highlight = QtCore.Qt.transparent

    def set_filter_style(self, visible, color=None):
        """Apply a PolygonManager filter/color rule result to this item.

        `visible=False` hides the polygon and remembers WHY (see
        `_filtered_hidden`) so "Show Polys" and LOD batching do not
        accidentally reveal it again; `color=None` restores the default
        red/blue outline.
        """
        self._filtered_hidden = not bool(visible)
        self.current_color = color
        self._pen_base = None  # clear the pen cache -- see _ensure_pens
        try:
            scn = self.scene()
            v = scn.views()[0] if scn and scn.views() else None
        except Exception:
            v = None
        want_visible = bool(visible) and (v is None or v.are_polygons_visible())
        self.setVisible(want_visible)
        self.update()

    def paint(self, painter, option, widget=None):
        # --- current zoom scale ---
        t = painter.worldTransform()
        scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

        # ---- stroke width: stay on Qt's THIN-LINE fast path ----
        #
        # THE dominant paint cost, and it is neither the geometry nor the
        # number of items.
        #
        # Qt draws pens of width <= 1 with a fast line routine. At width >= 2 it
        # falls back to the full stroker, which builds join/cap outline geometry
        # for every segment. That boundary is a CLIFF, measured drawing the real
        # BCI crown map (3504 polygons, 144,798 vertices after decimation)
        # straight onto a pixmap:
        #
        #     width 0 (hairline)      :   5.9 ms
        #     width 1                 :   5.5 ms
        #     width 2                 :  81.2 ms      <-- 14x
        #     width 3                 :  93.0 ms
        #     width 2/scale (=100)    :  84.4 ms      (the old code)
        #
        # Cosmetic vs scene-units barely matters; the WIDTH does. The old code
        # set `2.0 / scale` in scene units, which is far past the cliff at every
        # zoom.
        #
        # So: a 1-pixel cosmetic outline by default. Cosmetic means "width in
        # DEVICE pixels", which is exactly what the old division was emulating,
        # so the line keeps a constant on-screen thickness at any zoom -- and at
        # overview zoom a 1px outline is what you want anyway, since 2px strokes
        # on thousands of small crowns just merge into mush.
        #
        # Highlighted/selected items DO get the thicker 2px stroke: there are
        # only ever a handful of them, so paying the stroker there is free.
        self._ensure_pens()
        highlighted = self.isUnderMouse() or self.isSelected()
        if highlighted:
            pen = self._pen_highlight
            fill_color = self._fill_highlight if self.is_mask_polygon else self._fill_base
            width_px = 2.0
        else:
            pen = self._pen_base
            fill_color = self._fill_base
            width_px = 1.0
        pen.setCosmetic(True)
        pen.setWidthF(width_px)

        painter.setPen(pen)
        painter.setBrush(fill_color)
        # Zoom-aware vertex decimation -- see _display_polygon. Pixel-identical
        # output, 2.2x-3.7x faster on the real 1.01M-vertex project.
        painter.drawPolygon(self._display_polygon(scale))

        # ---- zoom + resolution aware label ----
        # Skip label during active dragging to prevent ghost marks.
        # Smart suppression: skip label rasterisation when zoomed out
        # (scale < _LABEL_SCALE_THRESHOLD) unless the item is hovered or
        # selected.  This dramatically reduces paint time at low zoom.
        should_show = self.show_label or self._hover_showing_label
        if self.name and not self.is_moving and should_show:
            # Smart zoom suppression
            if scale < self._LABEL_SCALE_THRESHOLD and not highlighted:
                return

            # Use cached image size
            img_w, img_h = self._get_image_size()
            res_boost = self._cached_res_boost

            base_px = 40 if self.is_rgb else 32
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            # Reuse cached QFont when pixel size hasn't changed
            if px != self._cached_font_px:
                font = QtGui.QFont()
                font.setPixelSize(px)
                self._cached_font = font
                self._cached_font_px = px
            painter.setFont(self._cached_font)
            painter.setPen(self._pen_label)

            # Simple positioning without extra boundingRect call
            bbox = self._polygon.boundingRect()
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
            # Locked polygons refuse to start a drag. Checked here -- at the
            # very top, before selection/undo bookkeeping -- rather than
            # inside a running drag, because interrupting an ALREADY-active
            # Qt drag is what causes the item to glitch/snap.
            if getattr(self, "is_locked", False) and not _prompt_unlock_click(self):
                event.ignore()
                return

            # An item drawn from a coarse pyramid level must be upgraded to its
            # real coordinates BEFORE any edit begins -- otherwise the user
            # would be dragging a decimated outline and that outline is what
            # would get saved.
            if getattr(self, "is_lod_geometry", False):
                try:
                    for v in (self.scene().views() if self.scene() else []):
                        if hasattr(v, "request_full_geometry"):
                            v.request_full_geometry(self)
                            break
                except Exception as e:
                    logging.debug("[polygon_lod] full-geometry upgrade failed: %s", e)
                if getattr(self, "is_lod_geometry", False):
                    # Still coarse -- refuse to move rather than corrupt it.
                    logging.warning(
                        "[polygon_lod] '%s' could not be upgraded to full "
                        "geometry; not editable this session", self.name)
                    event.ignore()
                    return

            # IMPORTANT: call BEFORE is_moving changes boundingRect()
            self.prepareGeometryChange()

            self.is_moving = True
            
            # Only deselect others if:
            #   - This item is NOT already selected (i.e. a fresh click on a new polygon), AND
            #   - Ctrl is NOT held (Ctrl = intentional multi-select)
            # This preserves Ctrl-built groups during drag: the user clicks an already-selected
            # item to start the drag without Ctrl, and we must NOT clear the group.
            try:
                scene = self.scene()
                if scene:
                    mods = event.modifiers()
                    if not self.isSelected() and not (mods & QtCore.Qt.ControlModifier):
                        for item in scene.selectedItems():
                            if item is not self:
                                item.setSelected(False)
            except Exception:
                pass
            
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
        a_props = menu.addAction("Edit properties")
        a_lock = menu.addAction("Unlock this polygon" if getattr(self, "is_locked", False)
                                 else "Lock this polygon")
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

        elif chosen == a_props and hasattr(v, "edit_polygon_properties"):
            v.edit_polygon_properties(self)

        elif chosen == a_lock:
            # Per-polygon toggle -- independent of PolygonManager's global
            # "Lock Polygons" checkbox, which sets every polygon at once.
            # This is the individual counterpart: lock/unlock just this one.
            self.is_locked = not getattr(self, "is_locked", False)

        elif chosen == a_delete and hasattr(v, "delete_polygon_for_this_file"):
            v.delete_polygon_for_this_file(self)

        elif chosen == a_delete_all and hasattr(v, "delete_all_polygons_in_group"):
            v.delete_all_polygons_in_group(self)


def _prompt_unlock_click(item):
    """Shared lock-click handler for EditablePolygonItem / EditablePointItem.

    Returns True if the click should proceed as a normal press (item is now
    unlocked one way or another), False if it must be swallowed.
    """
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Polygon Locked",
                                 f"'{getattr(item, 'name', '') or 'This polygon'}' is locked.",
                                 parent=None)
    btn_this = box.addButton("Unlock This", QtWidgets.QMessageBox.AcceptRole)
    btn_group = box.addButton("Unlock Group", QtWidgets.QMessageBox.AcceptRole)
    box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    box.exec_()
    clicked = box.clickedButton()

    if clicked is btn_this:
        item.is_locked = False
        return True
    if clicked is btn_group:
        try:
            scn = item.scene()
            v = scn.views()[0] if scn and scn.views() else None
            if v is not None and hasattr(v, "unlock_group"):
                v.unlock_group(getattr(item, "name", None))
        except Exception:
            logging.debug("[lock] unlock_group failed", exc_info=True)
        return not getattr(item, "is_locked", False)
    return False


def image_points_to_scene(pixitem, img_pts, img_size_hw=None):
    """Map IMAGE-pixel points to scene coords as a QPolygonF.

    The version this replaces called `pixitem.mapToScene(QPointF(x, y))` once
    per VERTEX -- a Python->C++ round trip each time, and a shapefile import of
    the BCI crown map pushes ~2.45 M of them through here.

    Building the ring in pixmap-local space and handing Qt the whole polygon
    uses the mapToScene(QPolygonF) overload instead: one call per ring rather
    than one per vertex. Measured over 3504 BCI-scale rings / 2,450,286
    vertices:

        per-vertex mapToScene   1.99 s
        whole-polygon overload  1.23 s      1.6x, bit-identical output
        numpy affine            3.15 s      SLOWER -- np.asarray on a Python
                                            list-of-lists costs more than the
                                            Qt calls it removes
        inline affine in Python 1.08 s      fastest, but only correct for a
                                            pure scale+translate; it drops
                                            m12/m21, so a rotated pixmap item
                                            would place every polygon wrong

    The remaining cost is constructing one QPointF per vertex, which no variant
    avoids -- that is what `is_lod` geometry is for.
    """
    if img_pts is None or len(img_pts) == 0:
        return QtGui.QPolygonF()

    if pixitem is not None:
        pm = pixitem.pixmap()
        if pm is not None and not pm.isNull():
            pw, ph = max(1, pm.width()), max(1, pm.height())
            h_eff, w_eff = img_size_hw if img_size_hw else (ph, pw)
            img_h, img_w = h_eff or ph, w_eff or pw
            sx = pw / float(img_w)
            sy = ph / float(img_h)
            local = QtGui.QPolygonF(
                [QtCore.QPointF(float(x) * sx, float(y) * sy) for x, y in img_pts])
            return pixitem.mapToScene(local)

    return QtGui.QPolygonF(
        [QtCore.QPointF(float(x), float(y)) for x, y in img_pts])


class PolygonTileItem(QtWidgets.QGraphicsItem):
    """One spatial TILE of many polygons, drawn in a single paint() call.

    THE PROBLEM THIS SOLVES
    -----------------------
    One QGraphicsItem per polygon means one PYTHON paint() call per polygon per
    frame, plus Qt's per-item machinery (bounding-rect test, style option,
    transform setup). Measured on the real BCI crown map -- 3504 polygons,
    2,142,685 vertices -- that floor is ~50 us per item, i.e. ~175 ms/frame
    (6 FPS) no matter how few vertices each item actually draws. Vertex
    decimation cannot help: the cost is per ITEM, not per vertex.

    WHY TILES AND NOT ONE BIG ITEM
    ------------------------------
    A single item holding every polygon was tried first and is WORSE when
    zoomed in -- 10x worse -- because its bounding rect covers the whole scene,
    so Qt can never cull it and repaints everything for any viewport change.
    Tiles keep a tight rect each, so only the handful under the viewport paint.
    That is the difference between naive batching and what QGIS actually does.

    Measured, same data, tiles + a 1px pen:

        zoom    per-item     tiled
        0.02    157.1 ms    15.4 ms     (10.2x -- 6 FPS -> 65 FPS)
        0.05     84.9 ms    13.4 ms
        0.15     14.9 ms     4.5 ms
        0.40     11.1 ms     4.4 ms

    Faster at EVERY zoom, which the single-item version was not.

    LEVELS are built LAZILY. Pre-building all of them for 3504 polygons cost
    ~20 s up front; almost all of it was for levels the user never looks at.
    """

    #: tolerance for level L is 2**L raster pixels, as in polygon_lod
    _LEVELS = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def __init__(self, rect, polygons, is_rgb=True):
        super().__init__()
        self._rect = rect
        # list[(np.ndarray(N,2), QColor-or-None)]. None means "use the
        # default red(RGB)/blue(multispectral) outline" -- kept as None
        # rather than resolving it here so the overwhelmingly common case
        # (no filter/color rule active) still hashes to a single dict entry
        # below instead of comparing QColor objects.
        self._polygons = polygons
        self._cache = {}                   # level -> {color_key: (QPainterPath, QColor)}
        self._is_rgb = is_rgb
        self._default_color = QtGui.QColor(QtCore.Qt.red if is_rgb else QtCore.Qt.blue)
        self.setZValue(POLYGON_Z)
        # Purely decorative: hit-testing and editing go through the real
        # EditablePolygonItem that gets promoted on click.
        self.setAcceptHoverEvents(False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setCacheMode(QtWidgets.QGraphicsItem.NoCache)

    def boundingRect(self):
        return self._rect

    def _level_for(self, scale):
        best = 0
        for lv in self._LEVELS:
            if (2.0 ** lv) * scale <= 1.0 and lv > best:
                best = lv
        return best

    def _path_for(self, level):
        """Return {color_key: (QPainterPath, QColor)} for this LOD level.

        Grouped by color rather than one path for the whole tile: a
        PolygonManager color rule can put different polygons in the same
        tile under different outline colors, and QPainterPath only supports
        one pen per drawPath() call. With no active color rule every polygon
        shares the `None` -> default-color group, so this degrades to the
        original single-path/single-drawPath behavior.
        """
        group = self._cache.get(level)
        if group is not None:
            return group
        from .polygon_lod import decimate
        group = {}
        tol = 2.0 ** level
        for pts, color in self._polygons:
            d = decimate(pts, tol) if level > 0 else pts
            c = color if color is not None else self._default_color
            key = c.rgba()
            entry = group.get(key)
            if entry is None:
                entry = (QtGui.QPainterPath(), c)
                group[key] = entry
            entry[0].addPolygon(QtGui.QPolygonF(
                [QtCore.QPointF(float(x), float(y)) for x, y in d]))
            entry[0].closeSubpath()
        self._cache[level] = group
        return group

    def paint(self, painter, option, widget=None):
        t = painter.worldTransform()
        scale = (t.m11() ** 2 + t.m12() ** 2) ** 0.5 or 1.0
        painter.setBrush(QtCore.Qt.NoBrush)
        for path, color in self._path_for(self._level_for(scale)).values():
            pen = QtGui.QPen(color)
            # Width 1, cosmetic: Qt's thin-line fast path. Anything >= 2
            # invokes the full stroker and costs ~14x more -- see
            # EditablePolygonItem.paint.
            pen.setCosmetic(True)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawPath(path)


class EditablePointItem(QtWidgets.QGraphicsObject):
    point_modified = QtCore.pyqtSignal()

    # Same threshold as EditablePolygonItem
    _LABEL_SCALE_THRESHOLD = 0.35

    def __init__(self, points, name="", is_rgb=False, parent=None,
                 pixmap_item=None, points_are_pixmap_local=False):
        super(EditablePointItem, self).__init__(parent)

        self.points = points
        self.name = name
        self.is_rgb = is_rgb
        self.pixmap_item = pixmap_item
        self.points_are_pixmap_local = points_are_pixmap_local

        # Same filtering/locking state as EditablePolygonItem -- see its
        # __init__ for the rationale.
        self.properties = {}
        self.current_color = None
        self._filtered_hidden = False
        self.is_locked = False

        # Same stacking rule as EditablePolygonItem.
        self.setZValue(POLYGON_Z)

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
        # Backing fields for the show_label / _hover_showing_label properties
        # (boundingRect depends on them -- see _invalidate_point_cache). Set
        # directly here because the geometry cache does not exist yet.
        self._show_label = False   # see EditablePolygonItem.__init__
        self._hover_label = False
        self.label_offset = QtCore.QPointF(10, -10)

        # Cache for res_boost calculation
        self._cached_res_boost = 1.0
        self._cached_img_size = None

        # ---- Geometry cache ----
        self._cached_brect = None
        self._cached_brect_drag = None
        self._cached_shape = None

        # ---- Paint state cache ----
        self._pen_label = QtGui.QPen(QtCore.Qt.red)
        self._cached_font = None
        self._cached_font_px = -1
        self._brush_color = QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue

        # Pre-compute cached geometry
        self._invalidate_point_cache()

    def _pixmap_pos(self):
        return self.pixmap_item.pos() if self.pixmap_item is not None else QtCore.QPointF(0, 0)

    def _scene_xy(self, p):
        """Return integer scene coords for a stored point `p`."""
        if hasattr(p, 'x'):
            px, py = p.x(), p.y()
        else:
            px, py = p[0], p[1]
        if self.points_are_pixmap_local:
            off = self._pixmap_pos()
            return int(off.x() + px), int(off.y() + py)
        else:
            return int(px), int(py)

    # ---- label-visibility properties (boundingRect depends on them) ----
    # Mirrors EditablePolygonItem: flipping either must go through
    # prepareGeometryChange(), or the scene keeps the stale rect and the label
    # ghosts. Callers assign `item.show_label = ...` directly, so a property
    # keeps every existing call site correct untouched.
    @property
    def show_label(self):
        return self._show_label

    @show_label.setter
    def show_label(self, value):
        value = bool(value)
        if value == getattr(self, "_show_label", None):
            return
        self.prepareGeometryChange()
        self._show_label = value
        self._update_bounding_rect()

    @property
    def _hover_showing_label(self):
        return self._hover_label

    @_hover_showing_label.setter
    def _hover_showing_label(self, value):
        value = bool(value)
        if value == getattr(self, "_hover_label", None):
            return
        self.prepareGeometryChange()
        self._hover_label = value
        self._update_bounding_rect()

    def _invalidate_point_cache(self):
        """Recompute cached boundingRect and shape from current points."""
        self._update_bounding_rect()

        pts = self.points
        is_empty = pts.isEmpty() if hasattr(pts, 'isEmpty') else (len(pts) == 0)
        if is_empty:
            self._cached_shape = QtGui.QPainterPath()
            return

        path = QtGui.QPainterPath()
        for p in self.points:
            sx, sy = self._scene_xy(p)
            path.addEllipse(sx, sy, 6, 6)
        self._cached_shape = path

    def _update_bounding_rect(self):
        pts = self.points
        is_empty = pts.isEmpty() if hasattr(pts, 'isEmpty') else (len(pts) == 0)
        if is_empty:
            self._cached_brect = QtCore.QRectF()
            self._cached_brect_drag = QtCore.QRectF()
            return

        xs, ys = [], []
        for p in self.points:
            sx, sy = self._scene_xy(p)
            xs.append(sx); ys.append(sy)
        rect = QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

        if rect.width() < 10 and rect.height() < 10:
            rect = rect.adjusted(-20, -20, 20, 20)

        self._cached_brect_drag = rect.adjusted(-10, -10, 10, 10)

        full = QtCore.QRectF(rect)
        if self.name and (self._show_label or self._hover_showing_label):
            label_width = len(self.name) * 50 + 100
            label_height = 250
            label_rect = QtCore.QRectF(
                full.topLeft() + self.label_offset,
                QtCore.QSizeF(label_width, label_height)
            )
            full = full.united(label_rect)
        self._cached_brect = full.adjusted(-50, -50, 50, 50)

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
        # Keep stable during drag (same rationale as EditablePolygonItem)
        if self.is_moving:
            return self._cached_brect if self._cached_brect else self._cached_brect_drag
        return self._cached_brect if self._cached_brect else QtCore.QRectF()

    def shape(self):
        if self._cached_shape is not None:
            return self._cached_shape
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

    def set_filter_style(self, visible, color=None):
        """Apply a PolygonManager filter/color rule result to this point item."""
        self._filtered_hidden = not bool(visible)
        self.current_color = color
        self._brush_color = color if color is not None else (
            QtCore.Qt.red if self.is_rgb else QtCore.Qt.blue)
        try:
            scn = self.scene()
            v = scn.views()[0] if scn and scn.views() else None
        except Exception:
            v = None
        want_visible = bool(visible) and (v is None or v.are_polygons_visible())
        self.setVisible(want_visible)
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(self._brush_color)
        for p in self.points:
            sx, sy = self._scene_xy(p)
            painter.drawRect(QtCore.QRectF(sx, sy, 1, 1))

        # Skip label during active dragging to prevent ghost marks.
        # Smart suppression: same as EditablePolygonItem.
        should_show = self.show_label or self._hover_showing_label
        if self.name and not self.is_moving and should_show:
            t = painter.worldTransform()
            scale = (t.m11()**2 + t.m12()**2) ** 0.5 or 1.0

            highlighted = self.isUnderMouse() or self.isSelected()
            if scale < self._LABEL_SCALE_THRESHOLD and not highlighted:
                return

            res_boost = self._get_res_boost()

            base_px = 40 if self.is_rgb else 32
            px = (base_px * res_boost) / scale
            px = int(max(14, min(220, round(px))))

            # Reuse cached QFont when pixel size hasn't changed
            if px != self._cached_font_px:
                font = QtGui.QFont()
                font.setPixelSize(px)
                self._cached_font = font
                self._cached_font_px = px
            painter.setFont(self._cached_font)
            painter.setPen(self._pen_label)

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
            if getattr(self, "is_locked", False) and not _prompt_unlock_click(self):
                event.ignore()
                return

            # IMPORTANT: call BEFORE is_moving changes boundingRect()
            self.prepareGeometryChange()

            self.is_moving = True

            # Only deselect others if this item is NOT already selected AND Ctrl is NOT held.
            try:
                scene = self.scene()
                if scene:
                    mods = event.modifiers()
                    if not self.isSelected() and not (mods & QtCore.Qt.ControlModifier):
                        for item in scene.selectedItems():
                            if item is not self:
                                item.setSelected(False)
            except Exception:
                pass

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
        a_props = menu.addAction("Edit properties")
        a_lock = menu.addAction("Unlock this polygon" if getattr(self, "is_locked", False)
                                 else "Lock this polygon")
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

        elif chosen == a_props and hasattr(v, "edit_polygon_properties"):
            v.edit_polygon_properties(self)

        elif chosen == a_lock:
            self.is_locked = not getattr(self, "is_locked", False)

        elif chosen == a_delete and hasattr(v, "delete_polygon_for_this_file"):
            v.delete_polygon_for_this_file(self)

        elif chosen == a_delete_all and hasattr(v, "delete_all_polygons_in_group"):
            v.delete_all_polygons_in_group(self)


def _norm_style_key_fp(fp):
    """Canonical form of an `all_polygons[group][filepath]`-style path key.

    Same formula as `ProjectTab._poly_index_lookup` (normpath THEN lower) --
    the two must agree, since both exist to bridge the same split: one
    project routinely holds two spellings of the same file's path
    (project.json/viewer form vs. shapefile-import's `os.path.normpath`
    form). Used by `ImageViewer.apply_polygon_style_map` so a filter/color
    style map keyed in one spelling still matches a viewer whose
    `image_data.filepath` is the other.
    """
    try:
        return os.path.normpath(str(fp)).lower() if fp else ""
    except Exception:
        return str(fp).lower()


# -------------------------------------------------------------------
# Main ImageViewer
# -------------------------------------------------------------------
class ImageViewer(QtWidgets.QGraphicsView):
    polygon_drawn = QtCore.pyqtSignal(object)
    polygon_changed = QtCore.pyqtSignal()
    editing_finished = QtCore.pyqtSignal()
    pixel_clicked = QtCore.pyqtSignal(QtCore.QPointF, object)
    editing_cancelled = QtCore.pyqtSignal()
    # Emits the 0-based FILE band index the user clicked in the band-selector bar.
    band_selected = QtCore.pyqtSignal(int)
    # Emits stretch parameters when the user adjusts the stretch slider
    stretch_applied = QtCore.pyqtSignal(object)

    # Carries a finished high-resolution viewport tile back to the GUI thread.
    #
    # The tile is produced on a ThreadPoolExecutor worker (see
    # ProjectTab._request_highres_viewport_region). It CANNOT be applied from
    # there: update_highres_overlay mutates the QGraphicsScene, which is only
    # legal on the GUI thread. The previous attempt used
    # QTimer.singleShot(0, callable) from the worker, but a QTimer takes the
    # affinity of the thread that creates it and that worker has no Qt event
    # loop -- so it never fired and every finished tile was silently dropped
    # (the region really was read and rendered; it just never reached the
    # scene, so zooming showed no error and no sharpening).
    #
    # A signal is the right primitive: emitting across threads queues the call
    # onto the receiver's thread automatically, with no event-loop assumptions
    # about the sender.
    highres_tile_ready = QtCore.pyqtSignal(object, float, float, float, float, object)

    # Global overlay toggle state across all viewers
    overlays_muted = False

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self._labels_visible = False   # see EditablePolygonItem.__init__
        self._zoom = 0
    

        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        # Use BSP tree for faster item lookup in large scenes
        self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)
        self._scene.setBspTreeDepth(4)  # Prevent excessive subdivisions for large coords
        self.setScene(self._scene)
        self._image = None

        # === PERFORMANCE OPTIMIZATIONS FOR LARGE IMAGES ===
        # Use minimal viewport updates - only repaint what changed
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        # Don't expand update regions for antialiasing (saves significant work)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        # Skip painter state save/restore overhead
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        # NO viewport cache.
        #
        # This used to be CacheBackground, with the comment "cache the
        # background (the image) for faster repaints". That premise is wrong:
        # CacheBackground caches what `drawBackground()` paints, and this class
        # neither overrides drawBackground nor sets a backgroundBrush -- the
        # image is a QGraphicsPixmapItem, i.e. an ITEM, which the background
        # cache never touches. So it cached an empty background while still
        # allocating a viewport-sized pixmap and re-rendering it every time the
        # transform changed, which is exactly what a wheel zoom does on every
        # tick.
        #
        # Measured on an 8000x6000 image at 1200x900 viewport, median of 5
        # interleaved runs (ms per wheel tick / per pan step, lower is better):
        #
        #                        CacheBackground   CacheNone
        #     zoom, 0 polygons        19.20           14.27    1.35x
        #     zoom, 1500 polygons     57.66           53.83    1.07x
        #     pan,  0 polygons         6.76            5.92    1.14x
        #     pan,  1500 polygons     60.19           44.67    1.35x
        #
        # Faster in every case, so there is no zoom/pan trade-off to balance.
        self.setCacheMode(QtWidgets.QGraphicsView.CacheNone)
        # Disable antialiasing on the view for speed (polygons have their own)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)

        # --- Drag repaint mode ---
        # With stable boundingRects (items keep the same bounding rect during
        # drag), SmartViewportUpdate is sufficient to avoid ghost labels and
        # is MUCH cheaper than FullViewportUpdate.  The cache mode is left
        # alone during drag for the same reason it is CacheNone above: there
        # is no background to cache, so switching to CacheBackground here only
        # bought a viewport-sized pixmap allocation.
        self._normal_viewport_update_mode = self.viewportUpdateMode()
        self._normal_cache_mode = self.cacheMode()
        self._drag_viewport_update_mode = QtWidgets.QGraphicsView.SmartViewportUpdate
        self._drag_cache_mode = QtWidgets.QGraphicsView.CacheNone
        self._drag_active_count = 0

        # For drawing
        self._drawing = False
        self.currentPolygon = QtGui.QPolygonF()
        self.polygons = []
        self._rb_dragging = False
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        self.temp_drawing_item = None

        if not hasattr(ImageViewer, "overlays_muted"):
            ImageViewer.overlays_muted = False
        self._overlay_toggle_btn = _OverlayToggleButton(self)

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
        self._image_data = None

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

        # --- Double-Buffered High-Resolution Overlay System ---
        self._highres_front_item = None
        self._highres_back_item = None
        self._highres_enabled = False
        self._highres_request_callback = None
        # Worker threads emit this; Qt queues it onto THIS (the GUI) thread,
        # which is the only thread allowed to touch the QGraphicsScene.
        self.highres_tile_ready.connect(self._on_highres_tile_ready)

        # For rectangle zoom mode (right-click drag to zoom to rectangle)
        self._rect_zoom_mode = False
        self._rect_zoom_start = None
        self._rect_zoom_item = None

        # --- Overlay tool button (DISABLED - use zoom bar instead) ---
        self.overlay_enabled = False         # Disabled: was interfering with drawing
        self._overlay_btn = None
        self._overlay_default_pos = QtCore.QPointF(14, 14)
        self._hover_hide_timer = QtCore.QTimer(self)
        self._hover_hide_timer.setInterval(5000)
        self._hover_hide_timer.setSingleShot(True)
        self._hover_hide_timer.timeout.connect(lambda: self._set_overlay_visible(False, immediate=True))

        # --- Attach zoom bar (deferred to ensure viewport exists) ---
        self._zoombar = None
        QtCore.QTimer.singleShot(0, self._attach_zoom_bar_deferred)

        # --- Attach band-selector bar (same deferral, kept as a separate
        # scheduled call so either overlay can be disabled independently) ---
        self._bandbar = None
        QtCore.QTimer.singleShot(0, self._attach_band_bar_deferred)

        # --- Attach top stretch bar ---
        self._stretchbar = None
        QtCore.QTimer.singleShot(0, self._attach_stretch_bar_deferred)





    def _attach_zoom_bar_deferred(self):
        """Attach the zoom bar after the widget is fully initialized."""
        try:
            attach_zoom_bar(self)
        except Exception as e:
            logging.debug(f"[ImageViewer] Failed to attach zoom bar: {e}")

    def _attach_band_bar_deferred(self):
        """Attach the band-selector bar after the widget is fully initialized."""
        try:
            attach_band_bar(self)
        except Exception as e:
            logging.debug(f"[ImageViewer] Failed to attach band bar: {e}")

    def _attach_stretch_bar_deferred(self):
        """Attach the top stretch bar after the widget is fully initialized."""
        try:
            attach_stretch_bar(self)
        except Exception as e:
            logging.debug(f"[ImageViewer] Failed to attach stretch bar: {e}")

    # ------------------------------------------------------------------
    # image_data / drawing as properties (not plain attributes)
    #
    # Both used to be bare instance attributes mutated from many call sites
    # (image_data reassigned from four places in project_tab.py; drawing set
    # true/false from ~8 places in this class). Converting them to properties
    # gives one choke point to refresh the band-selector bar whenever the
    # displayed image changes, and to hide it the instant drawing starts,
    # without touching any of those existing call sites.
    # ------------------------------------------------------------------

    @property
    def image_data(self):
        return self._image_data

    @image_data.setter
    def image_data(self, value):
        self._image_data = value
        try:
            self._refresh_band_bar()
        except Exception as e:
            logging.debug(f"[ImageViewer] band bar refresh failed: {e}")
        try:
            self._refresh_stretch_bar()
        except Exception as e:
            logging.debug(f"[ImageViewer] stretch bar refresh failed: {e}")

    @property
    def drawing(self):
        return self._drawing

    @drawing.setter
    def drawing(self, value):
        value = bool(value)
        turning_on = value and not self._drawing
        self._drawing = value
        if turning_on:
            bb = getattr(self, "_bandbar", None)
            if bb is not None:
                try:
                    bb.hide_immediately()
                except Exception:
                    pass
            sb = getattr(self, "_stretchbar", None)
            if sb is not None:
                try:
                    sb.hide_immediately()
                except Exception:
                    pass
            zb = getattr(self, "_zoombar", None)
            if zb is not None:
                try:
                    zb.hide_immediately()
                except Exception:
                    pass

    def _refresh_stretch_bar(self):
        """Re-seed the stretch bar's data range for whatever image is now loaded.

        This method was CALLED from two places -- the `image_data` setter above
        and `attach_stretch_bar()` -- but never actually defined. Both call
        sites wrap the call in a bare try/except that logs at debug level, so
        the resulting AttributeError was swallowed on every image load and the
        bar's Min/Max readout sat at its "-" placeholder with the sliders
        pinned at their construction defaults (0/1000 over a 0-255 assumed
        range) until some unrelated path happened to call refresh_range().
        That is why the min/max values never appeared.

        Mirrors _refresh_band_bar: a pure read of data already on image_data,
        with the same "no image -> hide" behaviour.
        """
        sb = getattr(self, "_stretchbar", None)
        if sb is None:
            return
        idata = self._image_data
        img = getattr(idata, "image", None) if idata is not None else None
        if idata is None or img is None:
            sb.hide_immediately()
            return
        try:
            sb.refresh_range()
            sb.reposition()
        except Exception as e:
            logging.debug(f"[ImageViewer] stretch bar range refresh failed: {e}")

    def _refresh_band_bar(self):
        """Repopulate the band-selector bar for whatever image is now loaded.

        Mirrors the same preview-vs-plain-image branching ImageStretchDialog
        already uses (project_tab.py's __init__ and open_stretch_dialog) --
        duplicated here rather than called back into ProjectTab, since this is
        a pure read of data already sitting on image_data with no side effects,
        and image_viewer.py has no existing import of project_tab.py.
        """
        bb = getattr(self, "_bandbar", None)
        if bb is None:
            return
        idata = self._image_data
        img = getattr(idata, "image", None) if idata is not None else None
        if idata is None or img is None:
            bb.hide_immediately()
            return

        profile = getattr(idata, "profile", None)
        preview_bands = getattr(idata, "preview_bands", None)

        if profile is not None and preview_bands:
            names = list(getattr(profile, "band_names", None) or [])
            count = int(profile.count)
            band_names = [names[i] if i < len(names) and names[i] else None
                          for i in range(count)]
        else:
            count = int(img.shape[2]) if hasattr(img, "ndim") and img.ndim == 3 else 1
            band_names = None

        # If a single band is currently displayed, highlight its button. Bar
        # buttons are indexed by FILE band, so a position within the resident
        # preview array must be mapped back through preview_bands first.
        active_band = None
        sp = getattr(self, "stretch_params", None)
        if (sp is not None and str(getattr(sp, "display_mode", "")).lower() == "single"
                and getattr(sp, "display_band", None) is not None):
            pos = int(sp.display_band)
            if profile is not None and preview_bands:
                if 0 <= pos < len(preview_bands):
                    active_band = int(preview_bands[pos])
            else:
                active_band = pos

        # Leading "composite" button reflects whichever full-image stretch mode
        # (auto / rgb) is the actual configured one -- while a single band is
        # being viewed, that mode lives on the stashed _composite_stretch_params
        # (see ProjectTab._on_band_bar_clicked), since stretch_params itself has
        # been overridden to "single" for the duration of the single-band view.
        composite_active = active_band is None
        composite_mode = "auto"
        if sp is not None and str(getattr(sp, "display_mode", "")).lower() != "single":
            composite_mode = str(getattr(sp, "display_mode", "auto") or "auto").lower()
        else:
            comp = getattr(self, "_composite_stretch_params", None)
            if comp is not None:
                composite_mode = str(getattr(comp, "display_mode", "auto") or "auto").lower()
        composite_label = "RGB" if composite_mode == "rgb" else "Auto"

        try:
            bb.populate(band_names, count, active_band=active_band,
                        composite_label=composite_label, composite_active=composite_active)
            bb.reposition()
            zb = getattr(self, "_zoombar", None)
            if zb is not None:
                zb.reposition()
        except Exception as e:
            logging.debug(f"[ImageViewer] band bar populate failed: {e}")

    # ---------- Drag repaint helpers ----------
    def _begin_item_drag(self):
        """Switch to SmartViewportUpdate during drag for clean repaints."""
        try:
            if getattr(self, "_bandbar", None) is not None:
                self._bandbar.hide_immediately()
            if getattr(self, "_stretchbar", None) is not None:
                self._stretchbar.hide_immediately()
            if getattr(self, "_zoombar", None) is not None:
                self._zoombar.hide_immediately()
                
            self._drag_active_count = getattr(self, "_drag_active_count", 0) + 1
            if self._drag_active_count == 1:
                # Save current modes (in case caller changed them elsewhere)
                self._normal_viewport_update_mode = self.viewportUpdateMode()
                self._normal_cache_mode = self.cacheMode()
                self.setViewportUpdateMode(self._drag_viewport_update_mode)
                self.setCacheMode(self._drag_cache_mode)
                # Lightweight viewport refresh (no full scene invalidation needed
                # since SmartViewportUpdate + stable boundingRects handle repaints)
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
                self.setCacheMode(getattr(self, "_normal_cache_mode", QtWidgets.QGraphicsView.CacheNone))
                scn = self.scene()
                if scn:
                    scn.invalidate(scn.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
                
                if getattr(self, "_bandbar", None) is not None:
                    self._bandbar.show_briefly()
                if getattr(self, "_stretchbar", None) is not None:
                    self._stretchbar.show_briefly()
                if getattr(self, "_zoombar", None) is not None:
                    self._zoombar.show_briefly()
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
            
            # Auto-refresh stretch bar so sliders match the newly applied stretch
            sb = getattr(self, "_stretchbar", None)
            if sb is not None:
                try:
                    sb.refresh_range()
                except Exception:
                    pass
        elif self._image is None:
            self.set_image(pixmap)

    def show_preview_array(self, arr_uint8, channel_order="rgb"):
        """
        Convenience: take a uint8 array (H,W), (H,W,1), (H,W,2), (H,W,3), or (H,W,4)
        and update pixmap in place.

        - 2D or (H,W,1): Grayscale
        - (H,W,2): Pad with zeros to make 3-channel RGB (e.g., Gray + Classification)
        - (H,W,3): RGB
        - (H,W,4): Convert BGRA to RGB (discard alpha) or use ARGB format

        channel_order: "rgb" (default, the historical assumption) or "bgr". The caller
        must state the TRUE order of the array — this used to be hard-coded to RGB, so
        cv2-loaded (BGR) images had red and blue swapped whenever the stretch dialog
        rendered them, while the refresh path (Format_BGR888) drew them correctly.
        """
        if arr_uint8 is None or arr_uint8.size == 0:
            return

        h, w = arr_uint8.shape[:2]

        # Pick the QImage format that matches the array's actual channel order, so no
        # byte-level reordering (and no risk of a double swap) is needed.
        _fmt3 = QtGui.QImage.Format_RGB888
        if str(channel_order).lower() == "bgr" and getattr(QtGui.QImage, "Format_BGR888", None) is not None:
            _fmt3 = QtGui.QImage.Format_BGR888

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
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, _fmt3)
            elif c == 3:
                # Standard 3-channel colour, in `channel_order`
                arr = np.ascontiguousarray(arr_uint8)
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, _fmt3)
            elif c == 4:
                # 4-channel (e.g., RGB + Classification or BGRA) -> drop alpha/extra
                arr = np.ascontiguousarray(arr_uint8[:, :, :3])
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, _fmt3)
            else:
                # More than 4 channels -> take first 3 for display
                arr = np.ascontiguousarray(arr_uint8[:, :, :3])
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, _fmt3)
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
        self._trigger_highres_update()

    def _load_ax_mods(self, image_path):
        """Load .ax modifications with mtime-based cache invalidation.

        Throttled: filesystem stat calls happen at most once per second.
        Between checks the cached value is returned immediately.
        """
        import time as _time

        now = _time.monotonic()
        last_check = getattr(self, "_ax_last_check_time", 0.0)

        # Fast path: if we checked less than 1 second ago, return cache
        if (now - last_check) < 1.0 and getattr(self, "_mods_cache_source", None) == image_path:
            return self._mods_cache or {}

        self._ax_last_check_time = now

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

        # Whether channel 0/2 need swapping to read as R/…/B. Only true when the
        # array is genuinely BGR (cv2-loaded, <=3-band small files). Multi-band
        # TIFFs loaded via the tifffile-preflight stack path (see
        # ProjectTab._imagedata_or_fallback) are tagged channel_order="rgb" and
        # already sit in native band order -- swapping them here used to silently
        # mislabel band 1 as band 3 and vice versa for every such file, since this
        # was the one consumer of image_data that never consulted channel_order
        # (the on-screen display and CSV export already do -- see
        # ProjectTab._render_with_viewer_stretch / _channels_in_export_order).
        _is_bgr = str(getattr(img_data, "channel_order", "bgr") or "bgr").lower() == "bgr"

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
                    # image_data.image is BGR from cv2.imread ONLY when _is_bgr
                    # (channel_order == "bgr"); the user always types b1/b2/b3
                    # meaning what they see as R/G/B in the viewer, so only swap
                    # when the underlying array is genuinely in BGR order.
                    C = img_mod.shape[2]
                    if C == 3:
                        if _is_bgr:
                            mapping = {
                                'b1': img_mod[:, :, 2],  # b1 = Red (user sees as channel 1)
                                'b2': img_mod[:, :, 1],  # b2 = Green (channel 2)
                                'b3': img_mod[:, :, 0],  # b3 = Blue (channel 3)
                            }
                        else:
                            mapping = {
                                'b1': img_mod[:, :, 0],
                                'b2': img_mod[:, :, 1],
                                'b3': img_mod[:, :, 2],
                            }
                    elif C > 3:
                        # First 3 channels only swap when genuinely BGR; additional
                        # bands stay in native order either way.
                        if _is_bgr:
                            mapping = {
                                'b1': img_mod[:, :, 2],  # Red
                                'b2': img_mod[:, :, 1],  # Green
                                'b3': img_mod[:, :, 0],  # Blue
                            }
                        else:
                            mapping = {
                                'b1': img_mod[:, :, 0],
                                'b2': img_mod[:, :, 1],
                                'b3': img_mod[:, :, 2],
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
                    # Keep NaN/Inf as-is instead of nan_to_num'ing them to 0.0.
                    # An index computed over NoData inputs (a NaN-filled border,
                    # or a 0/0 division inside e.g. NDVI) is not "0" -- reporting
                    # it as a hard 0 was indistinguishable from a genuine zero
                    # index. Non-finite now falls through to the NoData check
                    # below and reads back as "NoData", matching what the viewer
                    # renders for the same pixel.
                    idx_res = idx_res.astype(np.float32, copy=False)
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
            # Extract values in the semantic order matching ch_names ("b1"=R,
            # "b2"=G, "b3"=B). Only swap [2,1,0] when the array is genuinely BGR
            # (_is_bgr) -- tifffile-loaded multi-band stacks (channel_order="rgb")
            # already sit in native band order and must NOT be swapped, or band 1
            # and band 3's values get reported under each other's label.
            if C == 3:
                if _is_bgr:
                    vals = [
                        float(img_mod[ym, xm, 2]),  # b1 = Red
                        float(img_mod[ym, xm, 1]),  # b2 = Green
                        float(img_mod[ym, xm, 0]),  # b3 = Blue
                    ]
                else:
                    vals = [float(img_mod[ym, xm, c]) for c in range(3)]
            elif C > 3:
                if _is_bgr:
                    vals = [
                        float(img_mod[ym, xm, 2]),  # b1 = Red
                        float(img_mod[ym, xm, 1]),  # b2 = Green
                        float(img_mod[ym, xm, 0]),  # b3 = Blue
                    ]
                else:
                    vals = [float(img_mod[ym, xm, c]) for c in range(3)]
                for i in range(3, C):
                    vals.append(float(img_mod[ym, xm, i]))
            else:
                # 1 or 2 channels - no remapping
                vals = [float(img_mod[ym, xm, c]) for c in range(C)]

        # Check NoData status. This used to be "if ANY channel matches ANY
        # NoData value, the WHOLE pixel is NoData" -- which is correct for a
        # boolean EXPRESSION (e.g. "b1>182" is deliberately a whole-pixel
        # exclusion rule, same as process_polygon), but is wrong for a plain
        # numeric literal like -9999 checked across every channel.
        #
        # Multi-band prediction stacks carry ancillary planes (viewing/solar
        # geometry, water-vapor/AOT, an internal MASK band) that are declared
        # -9999 across the ENTIRE image, everywhere, including exactly where
        # the science bands the user is looking at (P(liana), NDVI, reflectance)
        # hold perfectly good values. Any file small enough to load in full
        # (under the viewer's preview threshold) then had every single click
        # report "NoData" in the status bar, no matter where you clicked --
        # because the always-invalid ancillary bands matched -9999, and the
        # per-pixel readout only ever reported the OR of every channel.
        #
        # NoData is now evaluated PER CHANNEL for numeric literals -- matching
        # the per-band masks used by process_polygon and random_shapes for the
        # same reason -- while expressions keep the original whole-pixel
        # semantics, since those are an intentional "exclude this pixel" rule.
        is_nodata = False           # whole-pixel, from an expression only
        channel_is_nodata = [False] * len(vals)

        # Non-finite (NaN / +-Inf) is intrinsically NoData -- it is not a
        # sentinel that has to be declared anywhere. Every other subsystem
        # already treats it that way unconditionally: utils.build_nodata_mask
        # ORs `~np.isfinite` in as its final pass, _per_band_nodata_masks seeds
        # its mask with `~np.isfinite`, and the polygon statistics go through
        # nanmean/nanmedian. The Inspector was the one place that gated it,
        # running the check only inside `if numeric_vals:` -- itself nested
        # inside `if nodata_values:`. So on a raster with no .ax sidecar, or one
        # whose .ax declares only expressions, a NaN pixel read back as an
        # ordinary value: the viewer showed a masked hole while clicking it
        # reported a number. Float rasters that mark their fill as NaN rather
        # than -9999 (the science bands of a prediction stack, typically) hit
        # this on every single click.
        for _i, _v in enumerate(vals):
            if not math.isfinite(_v):
                channel_is_nodata[_i] = True

        if nodata_values:
            try:
                import re
                _NODATA_EXPR_RE = re.compile(r'^([bB]\d+)\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)$')

                C = 1 if img_mod.ndim == 2 else img_mod.shape[2]

                def _get_channel_idx(band_num):
                    # vals/ch_names are already in RGB-semantic order (see above),
                    # so direct 1-based -> 0-based indexing works for all C.
                    return band_num - 1

                numeric_vals = []
                for nd_val in nodata_values:
                    if isinstance(nd_val, str):
                        m = _NODATA_EXPR_RE.match(nd_val)
                        if not m:
                            continue
                        band_name, op, threshold = m.groups()
                        ch_idx = _get_channel_idx(int(band_name[1:]))
                        if ch_idx >= C or ch_idx >= len(vals):
                            logging.debug(f"[Inspector] NoData expression {nd_val}: band exceeds image channels ({C})")
                            continue
                        band_val = vals[ch_idx]
                        threshold_val = float(threshold)
                        matched = (
                            (op == '<' and band_val < threshold_val) or
                            (op == '<=' and band_val <= threshold_val) or
                            (op == '>' and band_val > threshold_val) or
                            (op == '>=' and band_val >= threshold_val) or
                            (op == '==' and abs(band_val - threshold_val) < 1e-6) or
                            (op == '!=' and abs(band_val - threshold_val) >= 1e-6)
                        )
                        if matched:
                            is_nodata = True
                            logging.debug(f"[Inspector] NoData match: expression '{nd_val}' matched, ch{ch_idx}={band_val}")
                    else:
                        try:
                            numeric_vals.append(float(nd_val))
                        except (ValueError, TypeError):
                            pass

                if numeric_vals:
                    for i, v in enumerate(vals):
                        if not math.isfinite(v):
                            channel_is_nodata[i] = True
                            continue
                        for fv in numeric_vals:
                            abs_fv = abs(fv)
                            if abs_fv > 1e+30:
                                tol = abs_fv * 0.01
                            elif abs_fv > 1e+10:
                                tol = abs_fv * 0.001
                            elif abs_fv > 100:
                                tol = abs_fv * 0.001
                            else:
                                tol = 0.01
                            if abs(v - fv) < tol:
                                channel_is_nodata[i] = True
                                break
            except Exception as e:
                logging.debug(f"[Inspector] NoData check failed: {e}")

        payload = {"values": vals, "names": ch_names, "is_nodata": is_nodata,
                   "channel_nodata": channel_is_nodata}
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

    def scale(self, sx, sy):
        super().scale(sx, sy)
        self._trigger_highres_update()

    def fit_to_window(self):
        # Fit the IMAGE item to the view, ignoring the extra scene padding
        if self._image:
            self.fitInView(self._image, QtCore.Qt.KeepAspectRatio)
        else:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._trigger_highres_update()

    def smart_zoom_to_scene_rect(self, rect):
        """"Bring this scene rect into view" -- PANS ONLY, never changes the
        current zoom level. Used by:
          - PolygonManager._select_group_in_viewers (click a group in the
            list) -- ALWAYS, for both polygons and points. Clicking between
            rows in a list should behave like scrolling, not re-fitting.
          - PolygonManager.zoom_to_groups ("Zoom to Polygon" / double-click)
            -- only when the target is POINT-LIKE (near-zero area). A
            zero-area target has no natural "fit" scale (any zoom "fits" a
            single point), so forcing one was the actual source of the
            "zoom is too high on a COG" complaint this method exists to
            fix. A real polygon with actual extent goes through
            smart_fit_to_scene_rect below instead, which genuinely zooms.

        rect: bounding rect of the polygon/point in SCENE coordinates. May
        legitimately be zero-width/zero-height -- centerOn handles that
        fine, no expansion needed since no fit is computed. Only
        `rect is None` is treated as "nothing to pan to".
        """
        try:
            if rect is None:
                return
            if self._image is None or sip.isdeleted(self._image):
                return
            self.centerOn(rect.center())
            self.setFocus()
        except Exception as e:
            logging.debug(f"[smart_zoom_to_scene_rect] failed: {e}")

    def smart_fit_to_scene_rect(
        self, rect, native_hw=None, *,
        max_native_scale=2.0, padding_fraction=0.20, settle_back=0.9,
    ):
        """Actually ZOOM to frame this scene rect -- used only by
        PolygonManager.zoom_to_groups ("Zoom to Polygon" / double-click)
        when the combined target has real extent (a genuine polygon, not a
        point -- see smart_zoom_to_scene_rect above for the point case,
        and _select_group_in_viewers, which never calls this method at
        all: a plain list click is pan-only, always).

        One fitInView, one native-resolution-aware scale clamp, one
        settle-back, one final centerOn -- replaces the old hardcoded
        absolute scene-unit constants (min_dim=100.0, pad=max(50.0, ...))
        that broke down across image resolutions/COG pyramid levels, with
        no cap on how far fitInView could zoom in.

        rect: bounding rect of the polygon in SCENE coordinates. Expected
        to have real (non-near-zero) extent -- callers route point-like
        rects to smart_zoom_to_scene_rect instead. Only `rect is None` is
        treated as "nothing to fit".

        native_hw: (H, W) of the FULL NATIVE raster (from
        ProjectTab.polygon_basis_hw), used to cap zoom at max_native_scale
        (default 200%) of NATIVE resolution -- deliberately NOT scale
        relative to whatever pixmap happens to be cached right now
        (self.transform().m11() alone), since on a COG that cached pixmap
        is often a decimated preview, not the native raster. Pass None
        when native size can't be resolved (e.g. an unsaved/synthetic
        image with no file to probe); the cap is then skipped and
        padding/settle-back alone apply.
        """
        try:
            if rect is None:
                return
            if self._image is None or sip.isdeleted(self._image):
                return

            cached = getattr(self, "_cached_pixmap_size", None)
            cached_pw, cached_ph = cached if cached else (0, 0)

            # 1) Relative padding (percentage of the rect's own size, not a
            # fixed scene-unit constant), plus a SMALL, purely-defensive
            # floor so a near-zero-area rect still gets a sane nonzero
            # starting size for fitInView instead of the old fixed
            # min_dim=100.0.
            #
            # This floor is NOT where the native-resolution cap is
            # enforced -- that's step 3 below, an exact post-fitInView
            # clamp. An earlier version of this floor tried to PRE-empt the
            # cap here too, sized off the viewport (vp_w*cached_pw /
            # (max_native_scale*native_w)) -- but whenever the current
            # pixmap is already close to native resolution (any plain,
            # non-COG image: no decimated-preview gap, cached_pw≈native_w),
            # that formula collapses to roughly "half the viewport size in
            # scene units", disconnected from the polygon's/image's own
            # size. Verified this is a code-quality simplification, NOT an
            # independent bug: floor-expansion is always symmetric around
            # the rect's own center (never moves centerOn's target), and
            # the exact clamp in step 3 corrects the final scale to
            # max_native_scale regardless of what the floor produced --
            # the real "zoom lands outside the image" bug was in
            # PolygonManager._get_polygon_scene_rect's own coordinate
            # mapping (see canopie/qc/test_polygon_zoom.py's module
            # docstring), not here. This floor only needs to keep
            # fitInView from choking on a literal 0x0 input.
            pad_x = rect.width() * padding_fraction
            pad_y = rect.height() * padding_fraction
            view_rect = rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

            floor_w = floor_h = 0.0
            if cached_pw and cached_ph:
                floor_w = floor_h = 0.02 * max(cached_pw, cached_ph)

            if floor_w and view_rect.width() < floor_w:
                cx = view_rect.center().x()
                view_rect.setLeft(cx - floor_w / 2.0)
                view_rect.setRight(cx + floor_w / 2.0)
            if floor_h and view_rect.height() < floor_h:
                cy = view_rect.center().y()
                view_rect.setTop(cy - floor_h / 2.0)
                view_rect.setBottom(cy + floor_h / 2.0)

            # 2) The one and only fitInView call. Deliberately no centerOn
            # here even though fitInView centers internally -- step 5 below
            # is the single authoritative center call, once everything
            # (including the clamp/settle-back scale adjustments) is final.
            self.fitInView(view_rect, QtCore.Qt.KeepAspectRatio)

            # 3) Clamp to the native-resolution cap if fitInView overshot it.
            if native_hw and native_hw[0] and native_hw[1] and cached_pw:
                native_h, native_w = native_hw
                cur = self.transform().m11()
                screen_px_per_native_px = cur * (cached_pw / float(native_w))
                if screen_px_per_native_px > max_native_scale:
                    clamp_factor = max_native_scale / screen_px_per_native_px
                    self.scale(clamp_factor, clamp_factor)

            # 4) Settle back 10%.
            self.scale(settle_back, settle_back)

            # 5) Center exactly once, as the final step -- fixes the
            # jitter/offset from the old fitInView -> centerOn -> scale
            # ordering, where centering happened BEFORE the trailing scale.
            self.centerOn(view_rect.center())
            self.setFocus()
        except Exception as e:
            logging.debug(f"[smart_fit_to_scene_rect] failed: {e}")

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
        self._clear_highres_overlay()
        
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

        # Refresh the stretch bar range so its data bounds and slider labels instantly
        # reflect the new image data (e.g. after Image Editor changes the raw array).
        sb = getattr(self, "_stretchbar", None)
        if sb is not None:
            try:
                sb.refresh_range()
            except Exception as e:
                logging.debug(f"[ImageViewer] stretch bar auto-refresh failed: {e}")
    
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
            self._trigger_highres_update()
        finally:
            self._suppress_sync = False
            _ZoomBar._applying_fixed_zoom = False

    # --- High-Resolution Dynamic Viewport Region Loading ---
    def enable_highres_viewport(self, callback=None, cancel_callback=None):
        self._highres_request_callback = callback
        self._highres_cancel_callback = cancel_callback
        self._highres_enabled = True
        self.cancel_pending_highres_requests()

    def disable_highres_viewport(self):
        self.cancel_pending_highres_requests()
        self._highres_enabled = False
        self._highres_request_callback = None
        self._highres_cancel_callback = None
        self._clear_highres_overlay()

    def cancel_pending_highres_requests(self):
        """Cancel any pending or in-flight highres tile requests."""
        self._highres_request_id = getattr(self, "_highres_request_id", 0) + 1
        timer = getattr(self, "_highres_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        cancel_cb = getattr(self, "_highres_cancel_callback", None)
        if cancel_cb is not None and callable(cancel_cb):
            try:
                cancel_cb(self, self._highres_request_id)
            except Exception as e:
                logging.debug(f"[ImageViewer] highres cancel callback failed: {e}")

    def _clear_highres_overlay(self):
        """Remove both front and back buffer items from the scene and reset all handles."""
        scene = getattr(self, "_scene", None)
        for attr in ("_highres_front_item", "_highres_back_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    if not sip.isdeleted(item) and item.scene() is scene:
                        scene.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._highres_item = None

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._trigger_highres_update()

    def fitInView(self, *args, **kwargs):
        super().fitInView(*args, **kwargs)
        self._trigger_highres_update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._trigger_highres_update()

    def setTransform(self, transform, combine=False):
        super().setTransform(transform, combine)
        self._trigger_highres_update()

    def resetTransform(self):
        super().resetTransform()
        self._trigger_highres_update()

    def centerOn(self, *args, **kwargs):
        super().centerOn(*args, **kwargs)
        self._trigger_highres_update()

    def _trigger_highres_update(self):
        if not getattr(self, "_highres_enabled", False):
            return
        self.cancel_pending_highres_requests()
        timer = getattr(self, "_highres_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(200)  # 200 ms debounce
            timer.timeout.connect(self._on_highres_timer_timeout)
            self._highres_timer = timer
        timer.start()

    def _on_highres_timer_timeout(self):
        if not getattr(self, "_highres_enabled", False):
            return
        current_req_id = getattr(self, "_highres_request_id", 0)
        scale = self.transform().m11()
        vp = self.viewport()
        cached_size = getattr(self, '_cached_pixmap_size', None)
        if vp and cached_size and cached_size[0] > 0 and cached_size[1] > 0:
            fit_scale = min(vp.width() / float(cached_size[0]), vp.height() / float(cached_size[1]))
            if scale <= fit_scale * 1.05:
                self._clear_highres_overlay()
                return
        elif scale <= 1.05:
            self._clear_highres_overlay()
            return

        vp_rect = self.viewport().rect()
        scene_rect = self.mapToScene(vp_rect).boundingRect()
        cb = getattr(self, "_highres_request_callback", None)
        if cb is not None and callable(cb):
            try:
                import inspect
                sig = inspect.signature(cb)
                if len(sig.parameters) >= 3:
                    cb(self, scene_rect, current_req_id)
                else:
                    cb(self, scene_rect)
            except Exception:
                cb(self, scene_rect)

    def _on_highres_tile_ready(self, pixmap, scene_pos_x, scene_pos_y,
                               scale_x, scale_y, request_id):
        """GUI-thread landing point for a tile produced on a worker thread.

        Connected to `highres_tile_ready` in __init__, so Qt queues the call
        here from whatever thread emitted it. See that signal's comment for
        why the previous QTimer.singleShot approach silently dropped tiles.
        """
        try:
            self.update_highres_overlay(pixmap, scene_pos_x, scene_pos_y,
                                        scale_x, scale_y, request_id=request_id)
        except Exception as e:
            logging.debug(f"[ImageViewer] high-res tile apply failed: {e}")

    def update_highres_overlay(self, pixmap, scene_pos_x, scene_pos_y, scale_x, scale_y, request_id=None):
        """Double-buffered overlay swap.

        On each call:
          - The current *back* item (hidden) is promoted to *front* and filled
            with the new pixmap.
          - The current *front* item is demoted to *back* and hidden.
        This guarantees at most 2 QGraphicsPixmapItem allocations for the lifetime
        of the viewer and produces zero flicker.

        Backward-compat: ``self._highres_item`` always points to the front item.
        """
        if not getattr(self, "_highres_enabled", False) or pixmap is None or pixmap.isNull():
            return
        if request_id is not None and request_id != getattr(self, "_highres_request_id", 0):
            logging.debug(
                "[ImageViewer] Ignoring stale highres overlay (req %s != current %s)",
                request_id, self._highres_request_id,
            )
            return
        scene = getattr(self, "_scene", None)
        if scene is None or sip.isdeleted(scene):
            return

        def _item_valid(it):
            """True if the item belongs to the current scene and has not been deleted."""
            if it is None:
                return False
            try:
                return not sip.isdeleted(it) and it.scene() is scene
            except Exception:
                return False

        front = getattr(self, "_highres_front_item", None)
        back  = getattr(self, "_highres_back_item", None)

        if not _item_valid(front) and not _item_valid(back):
            # ── First call: create the front item from scratch ──────────────
            new_front = scene.addPixmap(pixmap)
            new_front.setZValue(HIGHRES_TILE_Z)
            new_front.setPos(scene_pos_x, scene_pos_y)
            new_front.setTransform(QtGui.QTransform().scale(scale_x, scale_y))
            new_front.setVisible(True)
            self._highres_front_item = new_front
            self._highres_back_item  = None
        else:
            # ── Subsequent calls: swap buffers ──────────────────────────────
            # The *back* buffer becomes the new *front* (reusing its scene item).
            # The current *front* is demoted to *back* and hidden.

            if _item_valid(back):
                # Promote back → front
                new_front = back
                new_front.setPixmap(pixmap)
                new_front.setPos(scene_pos_x, scene_pos_y)
                new_front.setTransform(QtGui.QTransform().scale(scale_x, scale_y))
                new_front.setVisible(True)
                new_front.setZValue(HIGHRES_TILE_Z)
            else:
                # Back was invalid/missing: allocate a second item
                new_front = scene.addPixmap(pixmap)
                new_front.setZValue(HIGHRES_TILE_Z)
                new_front.setPos(scene_pos_x, scene_pos_y)
                new_front.setTransform(QtGui.QTransform().scale(scale_x, scale_y))
                new_front.setVisible(True)

            # Demote the old front → back (hide it)
            new_back = front if _item_valid(front) else None
            if new_back is not None:
                new_back.setVisible(False)

            self._highres_front_item = new_front
            self._highres_back_item  = new_back

        # Backward-compat alias
        self._highres_item = self._highres_front_item

        self._highres_front_item.update()
        if self.viewport():
            self.viewport().update()

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

        delta = event.angleDelta().y()
        if not delta:
            event.accept()
            return

        # COALESCE, never discard. This used to return early (dropping the
        # event) whenever two wheel events arrived within 25 ms, and applied a
        # fixed 1.25x step to whichever ones survived. High-resolution wheels
        # and trackpads emit an event every few milliseconds, so most of the
        # user's scrolling was thrown away and the zoom appeared to stall --
        # while the zoom bar, which applies exactly what it is given, stayed
        # responsive. Accumulating the angle instead means no input is lost:
        # the throttle now only limits how often we REPAINT, not how much
        # scrolling counts.
        self._wheel_accum = getattr(self, '_wheel_accum', 0.0) + float(delta)
        event.accept()

        import time
        now = time.monotonic() * 1000.0
        last_wheel = getattr(self, '_last_wheel_time', 0.0)

        # Anchor the zoom to the point under the cursor, captured ONCE per
        # gesture. Qt's AnchorUnderMouse consults its own idea of the cursor
        # position when scale() runs, which is stale when the zoom is applied
        # from the coalescing timer rather than inside the event -- and because
        # scrollbars are integer-valued, re-deriving the anchor on every step
        # let rounding errors accumulate, so the image slid away from the
        # pointer during a long scroll. Pinning one scene point for the whole
        # gesture means the error cannot compound: every step re-solves against
        # the same reference.
        gesture_gap = now - getattr(self, '_wheel_gesture_time', 0.0)
        try:
            pos = event.pos()
            if gesture_gap > 250.0 or getattr(self, '_wheel_anchor_scene', None) is None:
                self._wheel_anchor_vp = QtCore.QPointF(pos)
                self._wheel_anchor_scene = self.mapToScene(pos)
            elif (QtCore.QPointF(pos) - self._wheel_anchor_vp).manhattanLength() > 2.0:
                # Cursor moved to a new spot mid-scroll: re-anchor there.
                self._wheel_anchor_vp = QtCore.QPointF(pos)
                self._wheel_anchor_scene = self.mapToScene(pos)
        except Exception:
            self._wheel_anchor_scene = None
        self._wheel_gesture_time = now

        if getattr(self, '_zooming', False):
            return          # a pending apply will consume the accumulated delta

        if now - last_wheel < 25.0:
            timer = getattr(self, '_wheel_coalesce_timer', None)
            if timer is None:
                timer = QtCore.QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(25)
                timer.timeout.connect(self._apply_wheel_zoom)
                self._wheel_coalesce_timer = timer
            if not timer.isActive():
                timer.start()
            return

        self._apply_wheel_zoom()

    def _apply_wheel_zoom(self):
        """Apply (and clear) the accumulated wheel delta as a single zoom step."""
        if not self._image:
            self._wheel_accum = 0.0
            return
        if getattr(self, '_zooming', False):
            return

        accum = getattr(self, '_wheel_accum', 0.0)
        self._wheel_accum = 0.0
        if not accum:
            return

        import time
        self._last_wheel_time = time.monotonic() * 1000.0
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
            
            # Zoom proportionally to how far the wheel actually turned. One
            # standard notch is 120 units and still gives exactly 1.25x, but a
            # high-resolution wheel sending many small deltas now sums to the
            # same total instead of either over-zooming (a full step per tiny
            # event) or under-zooming (events dropped by the old throttle).
            steps = accum / 120.0
            new_zoom = current_zoom * (1.25 ** steps)

            # Clamp zoom: min is fit_zoom, max is 50x
            new_zoom = max(fit_zoom, min(50.0, new_zoom))
            
            # Calculate relative scale required
            scale_factor = new_zoom / current_zoom
            
            # Update zoom counter based on whether we're above fit level
            if new_zoom > fit_zoom * 1.01:
                self._zoom = max(1, int((new_zoom / fit_zoom - 1) * 5))
            else:
                self._zoom = 0
            
            # Scale with NO anchor, then put the gesture's anchor scene point
            # back under the cursor ourselves. Solving it explicitly against a
            # fixed scene reference keeps the point under the pointer exactly,
            # instead of drifting as integer scrollbar rounding accumulates.
            anchor_scene = getattr(self, '_wheel_anchor_scene', None)
            anchor_vp = getattr(self, '_wheel_anchor_vp', None)
            prev_anchor = self.transformationAnchor()
            if anchor_scene is not None and anchor_vp is not None:
                self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
                self.scale(scale_factor, scale_factor)
                try:
                    now_vp = self.mapFromScene(anchor_scene)
                    dx = float(now_vp.x()) - anchor_vp.x()
                    dy = float(now_vp.y()) - anchor_vp.y()
                    if dx or dy:
                        hb = self.horizontalScrollBar()
                        vb = self.verticalScrollBar()
                        hb.setValue(int(round(hb.value() + dx)))
                        vb.setValue(int(round(vb.value() + dy)))
                except Exception:
                    pass
            else:
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
            
            # Refresh the zoom bar HERE. attach_zoom_bar's wheel wrapper reads
            # the zoom immediately after the event returns, which was correct
            # when zooming happened inline -- but the coalescing timer applies
            # it later, so that read saw the pre-zoom value and the bar lagged
            # behind the view. QGraphicsView.scale() is C++ and bypasses the
            # patched setTransform, so nothing else refreshes it either.
            zb = getattr(self, "_zoombar", None)
            if zb is not None:
                try:
                    cur = self.current_zoom_factor()
                    zb._block = True
                    zb._set_slider_from_zoom(cur)
                    zb._update_label(cur)
                    zb._block = False
                except Exception:
                    pass

            # Sync zoom to other viewers (debounced)
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
        finally:
            self._zooming = False
            self._trigger_highres_update()

        # More wheel input arrived while this step was being applied: schedule
        # it rather than leaving the view short of where the user scrolled to.
        if getattr(self, '_wheel_accum', 0.0):
            timer = getattr(self, '_wheel_coalesce_timer', None)
            if timer is None:
                timer = QtCore.QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(25)
                timer.timeout.connect(self._apply_wheel_zoom)
                self._wheel_coalesce_timer = timer
            if not timer.isActive():
                timer.start()

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
                    
                    # Hide overlay bars immediately when drawing starts
                    if getattr(self, "_bandbar", None) is not None:
                        self._bandbar.hide_immediately()
                    if getattr(self, "_stretchbar", None) is not None:
                        self._stretchbar.hide_immediately()
                    if getattr(self, "_zoombar", None) is not None:
                        self._zoombar.hide_immediately()
                        
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
                    # Cosmetic: 2 DEVICE pixels at any zoom, so the width never
                    # has to be recomputed when the user zooms mid-draw.
                    pen.setCosmetic(True)
                    pen.setWidthF(2.0)
                    pen.setColor(QtCore.Qt.red)
                    self.temp_drawing_item.setPen(pen)
                    brush = QtGui.QBrush(QtCore.Qt.transparent)
                    self.temp_drawing_item.setBrush(brush)
                    # WITHOUT this the item defaults to z=0, which is BELOW the
                    # sharpened COG viewport tile at HIGHRES_TILE_Z (0.5) -- so
                    # on a COG, zoomed in far enough for the tile to exist, the
                    # tile painted straight over the shape being drawn and
                    # nothing appeared. Plain images have no tile, which is why
                    # it only ever reproduced on a COG at high resolution.
                    self.temp_drawing_item.setZValue(TEMP_DRAWING_Z)
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
                    # Above the high-res zoom overlay, same as the polygon
                    # rubber band -- otherwise the selection rectangle is
                    # painted over by a refined tile mid-drag and the user
                    # cannot see what they are selecting.
                    self._rect_zoom_item.setZValue(TEMP_DRAWING_Z)
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

    def request_full_geometry(self, item):
        """Ask the owning ProjectTab to swap a coarse (pyramid) item for its
        real coordinates. Called just before an edit begins -- see
        EditablePolygonItem.mousePressEvent."""
        cb = getattr(self, "_full_geometry_callback", None)
        if callable(cb):
            try:
                cb(self, item)
            except Exception as e:
                logging.debug("[polygon_lod] full-geometry callback failed: %s", e)

    def set_full_geometry_callback(self, cb):
        self._full_geometry_callback = cb

    def add_polygon_to_scene(self, polygon, name="", is_mask_polygon=False, is_lod=False):
        is_rgb = False
        if hasattr(self, 'image_data') and self.image_data is not None and self.image_data.image is not None:
            if len(self.image_data.image.shape) == 3 and self.image_data.image.shape[2] == 3:
                is_rgb = True
        polygon_item = EditablePolygonItem(polygon, name, is_rgb, is_mask_polygon=is_mask_polygon)
        polygon_item.is_lod_geometry = bool(is_lod)
        polygon_item.is_locked = self.global_polygons_locked
        self._scene.addItem(polygon_item)
        polygon_item.polygon_modified.connect(self.on_polygon_modified)
        self.polygons.append({'polygon': polygon, 'name': name, 'item': polygon_item, 'type': 'polygon', 'is_mask_polygon': is_mask_polygon})
        # PERFORMANCE: Update polygon count for fast panning detection
        self._polygon_item_count = len(self.polygons)
        # The tiles no longer describe the scene. Drop them; paintEvent will
        # rebuild once the caller stops adding (import adds thousands in a row,
        # so rebuilding per-add would be quadratic).
        if getattr(self, "_batch_tiles", None):
            self.clear_polygon_batch()
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
        point_item.is_locked = self.global_polygons_locked
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
        """
        Generate random points across the image after asking the user for the count.

        Each point becomes its OWN item with its OWN unique group name (one
        all_polygons[group][filepath] entry per point) -- exactly like manually clicking one
        point at a time. Previously all N points were bundled into a single EditablePointItem
        with ONE polygon_drawn emission, so they all landed in ONE all_polygons entry: moving
        or deleting any point moved/deleted the whole batch, and Polygon Manager showed one
        row for all of them. Emitting once per point (via the existing on_polygon_drawn path)
        makes each point individually selectable, movable, and deletable, and gives it its own
        row in Polygon Manager -- the same mechanism used by manual drawing.
        """
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

        base_name = self.pending_group_name if self.pending_group_name else "random_point"

        # Get pixmap position offset
        off = self._image.pos()

        created = 0
        for i in range(num_points):
            x = random.uniform(0, img_w - 1)
            y = random.uniform(0, img_h - 1)
            # Convert to scene coordinates
            scene_x = off.x() + x
            scene_y = off.y() + y

            single_point = QtGui.QPolygonF()
            single_point.append(QtCore.QPointF(scene_x, scene_y))

            # Unique per-point group name so on_polygon_drawn creates a SEPARATE
            # all_polygons entry for this point (consumed immediately below, same as a
            # manual click would set it once per drawn shape).
            self.pending_group_name = f"{base_name}_{i + 1}"

            self.currentPolygon = single_point
            point_item = self.add_point_to_scene(single_point, self.pending_group_name)
            if not self.programmatically_adding_polygon:
                self.polygon_drawn.emit(point_item)
            created += 1

        self.pending_group_name = None
        self.currentPolygon = QtGui.QPolygonF()
        logging.info(
            f"[ImageViewer] Generated {created} individual random points "
            f"(groups '{base_name}_1'..'{base_name}_{created}')"
        )

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
             # Taking this branch SUPPRESSES polygon_modified (see
             # EditablePolygonItem.mouseReleaseEvent's moved_via_undo), which is
             # what would otherwise have dropped the LOD tiles. Without this the
             # tiles keep the pre-drag geometry and the polygon appears to snap
             # back to its original position as soon as the user zooms out.
             self.invalidate_polygon_batch()
             return True
        return False

    def on_polygon_modified(self):
        try:
            self._last_modified_item = self.sender()
        except Exception:
            self._last_modified_item = None
        # A vertex moved, so the tiles now describe geometry that no longer
        # exists. Editing only happens zoomed in (where the tiles are hidden),
        # so dropping them here is free and guarantees the user never zooms out
        # onto a stale outline.
        self.invalidate_polygon_batch()
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
        # Store the visibility state so new polygons can respect it
        self._polygons_visible = visible
        if getattr(self, "_batch_tiles", None):
            # Let the batch decide WHICH representation is shown; this call
            # only decides WHETHER anything is.
            self._set_batch_active(getattr(self, "_batch_active", False))
            return
        for item in self.get_all_polygons():
            try:
                # A filter rule's hide stays in effect even when polygons are
                # globally re-shown -- only "Clear Filters" restores it.
                item.setVisible(visible and not getattr(item, "_filtered_hidden", False))
            except Exception:
                pass

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

    # ------------------------------------------------------------------
    # Polygon locking ("Lock Polygons" in PolygonManager)
    # ------------------------------------------------------------------
    #: Persisted on the viewer (not just applied to current items) so that
    #: polygons spawned later -- by drawing, import, zoom/pan re-render,
    #: replication to this viewer -- inherit the correct lock state instead
    #: of defaulting to unlocked. See add_polygon_to_scene/add_point_to_scene.
    global_polygons_locked = False

    def set_polygons_locked(self, locked: bool):
        """Lock or unlock every polygon/point currently in this viewer."""
        self.global_polygons_locked = bool(locked)
        for item in self.get_all_polygons():
            item.is_locked = self.global_polygons_locked

    def unlock_group(self, group_name):
        """Unlock only the polygons/points named `group_name` in this viewer."""
        if not group_name:
            return
        for item in self.get_all_polygons():
            if getattr(item, "name", None) == group_name:
                item.is_locked = False

    # ------------------------------------------------------------------
    # Filter & Color (PolygonManager "Filter & Color Polygons")
    # ------------------------------------------------------------------
    def apply_polygon_style_map(self, style_map, key_for_item=None):
        """Apply a `{(group_name, filepath): {'visible': bool, 'color': QColor|None}}`
        mapping (built off-thread by PolygonManager's filter worker) to every
        polygon/point currently in this viewer, then keep the LOD tile batch
        (if active) consistent with the new visibility.

        `key_for_item` lets a caller override key resolution; by default the
        key is `(item.name, this viewer's current image filepath)`, matching
        how `all_polygons[group][filepath]` is addressed elsewhere.

        The default lookup is spelling-NORMALIZED, not exact. One project
        routinely holds two spellings of the same file's path -- project.json
        / this viewer's own `image_data.filepath` use forward slashes, while
        a shapefile import stores `os.path.normpath` (backslashes on
        Windows); see `ProjectTab._poly_index_lookup`'s docstring, which
        documents this exact split with measured numbers on a real project.
        `style_map`'s keys come from PolygonFilterWorker's snapshot of
        `all_polygons`, i.e. the STORAGE spelling. An exact-tuple lookup
        against the viewer's spelling therefore missed on every single
        shapefile-imported polygon: `style_map.get(key)` returned None for
        all of them, and the `style is None` branch below actively RESETS
        each item to its default appearance -- so on exactly the projects
        that motivated this feature, the filter looked like it did nothing
        at all, regardless of whether any rule matched correctly.
        """
        fp = getattr(getattr(self, "image_data", None), "filepath", None)
        touched_batch = False
        by_name = None
        if key_for_item is None:
            # Build once per call: normalize style_map's keys down to
            # {name: style} for entries whose filepath matches THIS viewer's
            # image, under the same normpath-then-lower formula
            # _poly_index_lookup uses, so either spelling matches.
            want = _norm_style_key_fp(fp)
            by_name = {}
            for (nm, k), style in style_map.items():
                if _norm_style_key_fp(k) == want:
                    by_name[nm] = style
        for item in self.get_all_polygons():
            if key_for_item is not None:
                style = style_map.get(key_for_item(item))
            else:
                style = by_name.get(getattr(item, "name", None))
            if style is None:
                # No rule matched -- restore the default appearance rather
                # than leaving a stale filter result in place.
                if getattr(item, "current_color", None) is not None or getattr(item, "_filtered_hidden", False):
                    item.set_filter_style(True, None)
                    touched_batch = True
                continue
            item.set_filter_style(style.get("visible", True), style.get("color"))
            touched_batch = True

        if touched_batch and getattr(self, "_batch_tiles", None):
            # Lazy invalidate, not an eager rebuild: rebuild_polygon_batch()
            # is O(polygons x vertices) and was measured costing multiple
            # seconds on real projects (3500+ polygons), running synchronously
            # on the GUI thread on every single "Apply" click -- this was the
            # other half of "filtering should be instant."
            #
            # This does not reintroduce the stale-tile problem the eager
            # rebuild guarded against: invalidate_polygon_batch() drops the
            # tiles via clear_polygon_batch(), whose `was_active` branch falls
            # back to per-item visibility that already respects the NEW
            # _filtered_hidden state each item was just given above (this
            # loop runs set_filter_style on every item BEFORE this block).
            # So between this click and the next zoom crossing the batch
            # threshold, the user sees the correct per-item state, not a
            # stale tile -- and paintEvent's own staleness check rebuilds
            # correctly (and lazily) once batching is needed again.
            self.invalidate_polygon_batch()

    def clear_polygon_style(self):
        """Undo every filter/color rule in this viewer -- restores default
        red/blue outlines and full visibility."""
        self.apply_polygon_style_map({})

    # ------------------------------------------------------------------
    # Properties editor (right-click "Edit properties")
    # ------------------------------------------------------------------
    def edit_polygon_properties(self, item):
        """Open PolygonPropertiesDialog for `item` and persist edits back to
        `all_polygons[group][filepath]['properties']`."""
        owner = self._find_project_owner()
        if owner is None:
            QtWidgets.QMessageBox.warning(self, "Unavailable", "No owning project found.")
            return

        group = getattr(item, "name", None) or ""
        fp = getattr(getattr(self, "image_data", None), "filepath", None)
        if not group or not fp:
            QtWidgets.QMessageBox.warning(self, "Unavailable",
                                           "This polygon is not yet associated with a saved image/group.")
            return

        # all_polygons is keyed by filepath STRING, and one project routinely
        # holds two spellings of the same file: the viewer/project.json form
        # ('C:/Users/x/y.tif', forward slashes) and the shapefile-import form
        # ('C:\\Users\\x\\y.tif', os.path.normpath). An exact .get(fp) matches
        # only the former, so on a project whose polygons came from a shapefile
        # import EVERY "Edit properties" failed with "Could not find this
        # polygon's stored data" -- the polygon is plainly on screen, but it is
        # filed under the other spelling. _poly_index_lookup returns the UNION
        # of both indices for exactly this reason; see its docstring.
        polys_for_group = owner.all_polygons.get(group, {}) or {}
        entry = polys_for_group.get(fp)
        storage_key = fp
        if entry is None:
            try:
                for gn, key in owner._poly_index_lookup(fp):
                    if gn == group and key in polys_for_group:
                        entry, storage_key = polys_for_group[key], key
                        break
            except Exception:
                logging.debug("[properties] index lookup failed", exc_info=True)
        if entry is None:
            QtWidgets.QMessageBox.warning(self, "Unavailable",
                                           "Could not find this polygon's stored data.")
            return

        props = dict(entry.get('properties') or {})

        dlg = PolygonPropertiesDialog(props, title=f"Properties — {group}", parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        new_props = dlg.get_properties()
        entry['properties'] = new_props
        item.properties = new_props

        try:
            if hasattr(owner, "_mark_polygon_dirty"):
                # The key the entry actually lives under -- not necessarily the
                # viewer's spelling of it (see the lookup above).
                owner._mark_polygon_dirty(group, storage_key)
        except Exception:
            logging.debug("[properties] _mark_polygon_dirty failed", exc_info=True)

        try:
            # ProjectTab has no request_save_polygons method (never has --
            # confirmed via grep) and never will unless someone adds one by
            # accident: this used to probe for it via hasattr() first, which
            # always failed, so every save silently fell through to
            # save_incremental() anyway. That fallback is now unconditional.
            if hasattr(owner, "save_incremental"):
                owner.save_incremental()
        except Exception:
            logging.exception("[properties] saving polygon properties failed")

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
        self.clear_polygon_batch()

        # INVALIDATE THE LOAD RECORD.
        #
        # ProjectTab.load_polygons stamps `_loaded_polygon_names` with what it
        # drew, and update_all_polygons uses it to decide which polygons a
        # missing scene item may be read as "the user deleted this". Once the
        # scene has been emptied that record describes a load that no longer
        # exists -- every name in it is now absent, so the purge would treat
        # ALL of them as deletions.
        #
        # set_image() calls this method, so it runs on every refresh: the
        # shapefile-derived polygons were purged from memory in the window
        # between the clear and load_polygons repopulating the scene, while a
        # just-drawn polygon (not in the stale record) survived. Dropping the
        # record here makes the purge refuse until a fresh load re-establishes
        # it, which is the same safe default it already applies when no record
        # was ever taken.
        self._loaded_polygon_names = None

        # Only invalidate if we had items to remove (skip for huge images with no polygons)
        if all_items:
            try:
                self._scene.invalidate(self._scene.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
                self.viewport().update()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Batched polygon rendering (tiles)
    # ------------------------------------------------------------------
    # Above _BATCH_MIN_POLYGONS items, and below _BATCH_SCALE_THRESHOLD zoom,
    # the per-polygon QGraphicsItems are HIDDEN and a handful of PolygonTileItem
    # objects draw the same geometry instead. See PolygonTileItem for the
    # measurements.
    #
    # The items are hidden, NOT removed. That is deliberate: get_all_polygons()
    # walks the scene, and ProjectTab.update_all_polygons() deletes from
    # self.all_polygons any polygon it cannot find a scene item for. Removing
    # the items to batch them would therefore silently DESTROY the user's
    # polygons on the next autosave. Hiding costs nothing at paint time (Qt
    # skips invisible items entirely) and keeps every existing code path --
    # save, export, selection, mask status -- working unchanged.
    #
    # Below the threshold a crown is ~2 screen pixels, so losing hover/selection
    # there costs the user nothing; zoom in and the real items come straight
    # back.

    _BATCH_MIN_POLYGONS = 800

    # Fallback only; the live value comes from _batch_scale_threshold().
    _BATCH_SCALE_THRESHOLD = 0.25

    # How many per-item polygons we are willing to paint in one frame.
    # Measured at 30000 polygons (ms/frame vs items actually painted):
    #     2025 items -> 35.9 ms      303 items -> 13.0 ms
    #      821 items -> 18.7 ms      169 items ->  8.9 ms
    # so ~250 is the most that fits a 60 FPS budget.
    _BATCH_MAX_LIVE_ITEMS = 250

    # Never batch at or above this zoom: at 1:1 the user is unambiguously
    # editing, and hover/selection matter more than frame rate.
    _BATCH_SCALE_CAP = 1.0

    # Polygons per tile. Small enough that a zoomed-in viewport touching one
    # tile does not pay for hundreds of off-screen polygons; large enough that
    # a zoomed-out frame is still only a few dozen drawPath calls.
    _BATCH_TILE_TARGET = 200

    # Defaults live on the class so every read is safe even for a viewer that
    # never crosses the threshold. _batch_tiles is an immutable () so that any
    # append that skips clear_polygon_batch() fails loudly instead of sharing
    # one list across every viewer.
    _batch_tiles = ()
    _batch_active = False
    _batch_built_count = -1
    _batch_sync_queued = False

    def _batch_scale_threshold(self):
        """Zoom below which tiles draw instead of the per-polygon items.

        A FIXED threshold cannot be right for both a 3000-polygon project and a
        30000-polygon one: what actually costs time is how many items land in
        the viewport, and that is set by polygon DENSITY, not by zoom.

            items_in_view ~= density * viewport_scene_area
                          =  (N / scene_area) * (vw * vh) / scale**2

        Solving for the scale at which that equals _BATCH_MAX_LIVE_ITEMS gives
        the switch-over point. Denser projects then keep tiles to a higher zoom
        automatically, which is what "load as many features as you like without
        losing viewer speed" actually requires.
        """
        try:
            n = len(self._batch_source_items())
            rect = self._scene.sceneRect()
            vp = self.viewport()
            area = rect.width() * rect.height()
            if n <= 0 or area <= 0 or vp is None:
                return self._BATCH_SCALE_THRESHOLD
            density = n / float(area)
            scale = math.sqrt(density * vp.width() * vp.height()
                              / float(self._BATCH_MAX_LIVE_ITEMS))
            return max(0.02, min(self._BATCH_SCALE_CAP, scale))
        except Exception:
            return self._BATCH_SCALE_THRESHOLD

    def _batch_source_items(self):
        out = []
        for rec in getattr(self, "polygons", []):
            if rec.get('type') != 'polygon':
                continue
            it = rec.get('item')
            if it is not None:
                out.append(it)
        return out

    def clear_polygon_batch(self):
        """Drop the tiles and hand rendering back to the per-polygon items."""
        was_active = getattr(self, "_batch_active", False)
        for t in getattr(self, "_batch_tiles", []):
            try:
                self._scene.removeItem(t)
            except Exception:
                pass
        self._batch_tiles = []
        self._batch_active = False
        if was_active:
            # The items were hidden on our behalf; nothing is drawing them now.
            # A polygon a filter rule hid stays hidden -- the batch toggle
            # must not un-hide it just because it is no longer tiled.
            visible = self.are_polygons_visible()
            for it in self._batch_source_items():
                it.setVisible(visible and not getattr(it, "_filtered_hidden", False))

    def invalidate_polygon_batch(self):
        """Drop the LOD tiles AND force paintEvent to rebuild them.

        Clearing alone is not enough: paintEvent's staleness check compares
        `_batch_built_count` against len(self.polygons), which is UNCHANGED by
        a move (no polygon was added or removed), so the tiles would simply be
        rebuilt from the same stale snapshot -- or not rebuilt at all. Stamping
        -1 guarantees the next paint re-reads the live geometry.

        Must be called by every path that changes a polygon's geometry or
        position. `on_polygon_modified` is NOT such a path on its own: when a
        drag is committed through the undo stack,
        `EditablePolygonItem.mouseReleaseEvent` sets `moved_via_undo` and
        deliberately does not emit `polygon_modified`, so nothing here ever
        ran. Zoomed in the user saw the real (correct) item; zoomed out the
        tiles took over and redrew the polygon at its PRE-DRAG position --
        the "polygons jump back when I zoom out" report.
        """
        if getattr(self, "_batch_tiles", None):
            self.clear_polygon_batch()
        self._batch_built_count = -1

    def rebuild_polygon_batch(self):
        """(Re)build the tile items from the current polygon items."""
        import numpy as _np
        self.clear_polygon_batch()
        items = self._batch_source_items()
        if len(items) < self._BATCH_MIN_POLYGONS:
            return
        # Stamp the count up front: if the build bails out below (degenerate
        # geometry), we must not re-attempt it on every single paint. Counts
        # ALL records, not just polygons, so it matches the cheap staleness
        # test in paintEvent (which must not walk the list).
        self._batch_built_count = len(getattr(self, "polygons", ()))

        is_rgb = False
        img = getattr(getattr(self, 'image_data', None), 'image', None)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
            is_rgb = True

        # Collect geometry once, as arrays, with each polygon's centroid.
        geoms, cxs, cys, colors = [], [], [], []
        for it in items:
            # A polygon a filter rule hid must not reappear the moment the
            # LOD batch takes over -- exclude it from the tile entirely
            # rather than relying on a visibility flag the tile does not
            # have per-polygon.
            if getattr(it, "_filtered_hidden", False):
                continue
            # Reuse the array the item already keeps for decimation. Rebuilding
            # it here from the QPolygonF costs a Python loop over every vertex
            # -- 4.46 M of them on the BCI crown map, measured at 4.3 s.
            arr = getattr(it, "_pts_np", None)
            if arr is None:
                poly = it.polygon
                if poly is None or len(poly) < 2:
                    continue
                arr = _np.array([[p.x(), p.y()] for p in poly], dtype=_np.float64)
            if len(arr) < 2:
                continue

            # ITEM coordinates -> SCENE coordinates.
            #
            # Both sources above are the item's LOCAL geometry. Dragging a
            # polygon does not rewrite that geometry: Qt moves the item's
            # pos() and itemChange deliberately leaves the points alone. So a
            # tile built straight from them draws every dragged polygon at the
            # position it had BEFORE the drag -- and because tiles only take
            # over when zoomed out, the polygon appeared to snap back to its
            # original place the moment you zoomed out.
            #
            # sceneTransform() covers pos() and any item transform. It is the
            # identity for every untouched polygon, so this costs nothing in
            # the common case.
            t = it.sceneTransform()
            if not t.isIdentity():
                m = _np.array([[t.m11(), t.m12()], [t.m21(), t.m22()]])
                arr = arr @ m + _np.array([t.dx(), t.dy()])

            geoms.append(arr)
            colors.append(getattr(it, "current_color", None))
            cxs.append(arr[:, 0].mean())
            cys.append(arr[:, 1].mean())
        if not geoms:
            return

        cxs = _np.asarray(cxs)
        cys = _np.asarray(cys)
        # Tile GRANULARITY has to follow the polygon count, not be fixed.
        #
        # A tile is all-or-nothing: if any part of it is on screen, every
        # polygon in it is drawn. With a fixed 8x8 grid and 30000 polygons each
        # tile holds ~470 polygons, so a zoomed-in viewport touching one tile
        # paid for all 470 -- measured 30.0 ms/frame at zoom 0.40, WORSE than
        # the 18.7 ms the per-item path cost. Sizing the grid so a tile holds
        # ~_BATCH_TILE_TARGET polygons keeps that bounded at any count.
        nx = ny = int(max(2, min(32, math.ceil(
            math.sqrt(len(geoms) / float(self._BATCH_TILE_TARGET))))))
        x0, x1 = float(cxs.min()), float(cxs.max())
        y0, y1 = float(cys.min()), float(cys.max())
        dx = (x1 - x0) / nx or 1.0
        dy = (y1 - y0) / ny or 1.0
        ix = _np.clip(((cxs - x0) / dx).astype(int), 0, nx - 1)
        iy = _np.clip(((cys - y0) / dy).astype(int), 0, ny - 1)

        buckets = {}
        for k, (gx, gy) in enumerate(zip(ix, iy)):
            buckets.setdefault((int(gx), int(gy)), []).append((geoms[k], colors[k]))

        for polys in buckets.values():
            # TIGHT rect from the real geometry, not the tile grid -- the rect
            # is what lets Qt cull, and a loose one is how the naive
            # single-item version ended up 10x slower when zoomed in.
            minx = min(float(p[:, 0].min()) for p, _c in polys)
            maxx = max(float(p[:, 0].max()) for p, _c in polys)
            miny = min(float(p[:, 1].min()) for p, _c in polys)
            maxy = max(float(p[:, 1].max()) for p, _c in polys)
            rect = QtCore.QRectF(minx, miny, maxx - minx, maxy - miny)
            tile = PolygonTileItem(rect, polys, is_rgb=is_rgb)
            tile.setVisible(False)
            self._scene.addItem(tile)
            self._batch_tiles.append(tile)

    def _sync_polygon_batch(self):
        """Switch between per-item and tiled rendering for the current zoom."""
        items = self._batch_source_items()
        if len(items) < self._BATCH_MIN_POLYGONS:
            # DROP the tiles, do not merely hide them. Hiding leaves geometry
            # in the scene that no longer corresponds to any polygon, and the
            # next thing that shows them paints deleted outlines -- which is
            # exactly what "remnants after Delete All" was.
            if getattr(self, "_batch_tiles", None):
                self.clear_polygon_batch()
            self._batch_built_count = -1
            return

        # Rebuild if the tiles were dropped, or if polygons were removed behind
        # our back (several code paths rebuild self.polygons by list
        # comprehension rather than going through a remove method).
        if not getattr(self, "_batch_tiles", None) or \
                getattr(self, "_batch_built_count", -1) != len(getattr(self, "polygons", ())):
            self.rebuild_polygon_batch()
            if not self._batch_tiles:
                return

        scale = self.transform().m11() or 1.0
        want = scale < self._batch_scale_threshold()
        if want != getattr(self, "_batch_active", False):
            self._set_batch_active(want, items)

    def _set_batch_active(self, active, items=None):
        if items is None:
            items = self._batch_source_items()
        visible = self.are_polygons_visible()
        for t in getattr(self, "_batch_tiles", []):
            t.setVisible(bool(active) and visible)
        for it in items:
            # A filter-hidden polygon is excluded from the tile geometry
            # (see rebuild_polygon_batch), so it must also stay hidden in the
            # per-item representation -- otherwise it would flicker back into
            # view every time zoom crosses the batch threshold.
            it.setVisible((not active) and visible and not getattr(it, "_filtered_hidden", False))
        self._batch_active = bool(active)

    def paintEvent(self, event):
        # The universal zoom hook. setTransform() is not virtual in Qt, so
        # QGraphicsView.scale()/fitInView() bypass the Python override -- but
        # nothing bypasses a repaint. Toggling is deferred to the event loop
        # because changing item visibility during a paint is not allowed.
        n = len(getattr(self, "polygons", ()))
        if getattr(self, "_batch_tiles", None) or n >= self._BATCH_MIN_POLYGONS:
            scale = self.transform().m11() or 1.0
            want = scale < self._batch_scale_threshold()
            # Staleness must NOT be gated on n >= _BATCH_MIN_POLYGONS. Deleting
            # every polygon takes n to 0, which under that gate reported "not
            # stale" -- so no resync was queued and the tiles carried on
            # painting the polygons that had just been deleted.
            stale = (getattr(self, "_batch_tiles", None)
                     and getattr(self, "_batch_built_count", -1) != n)
            if (want != getattr(self, "_batch_active", False) or stale) and \
                    not getattr(self, "_batch_sync_queued", False):
                self._batch_sync_queued = True
                QtCore.QTimer.singleShot(0, self._deferred_batch_sync)
        super().paintEvent(event)

    def _deferred_batch_sync(self):
        self._batch_sync_queued = False
        try:
            self._sync_polygon_batch()
        except Exception:
            logging.exception("polygon batch sync failed")

    def _remove_polygon_item_safely(self, item, defer_repaint=False):
        """
        Safely remove a polygon/point item from scene with proper ghost label cleanup.
        Call this instead of self._scene.removeItem(item) directly.

        Parameters
        ----------
        defer_repaint : bool
            If True, skip per-item scene invalidation and viewport update
            (caller is responsible for a single repaint after the batch).
        """
        if item is None:
            return
        try:
            # Remove from scene (prepareGeometryChange is unnecessary right
            # before removeItem -- it forces a redundant index update on an
            # item that is about to be detached from the scene).
            if item.scene() is self._scene:
                self._scene.removeItem(item)
            
            if not defer_repaint:
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
            self.temp_drawing_item.setZValue(TEMP_DRAWING_Z)
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
        """UI-only purge of a group's overlays in a specific viewer.

        Batches scene item removals: temporarily switches to NoIndex and
        disables viewport updates to avoid per-item BSP rebalancing and
        repaints, then performs a single scene invalidation at the end.
        """
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

                # Collect items first, then batch-remove with indexing disabled
                items_to_remove = [it for it in list(sc.items()) if _is_target(it)]
                if items_to_remove:
                    old_idx = sc.itemIndexMethod()
                    sc.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
                    try:
                        for it in items_to_remove:
                            try:
                                sc.removeItem(it)
                            except Exception:
                                pass
                    finally:
                        sc.setItemIndexMethod(old_idx)

                # Single scene invalidation after all removals
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

# --- Overlay Toggle Button ---------------------------------------------------
class _OverlayToggleButton(QtWidgets.QToolButton):
    """
    Small button in the top-left of the viewer to toggle all overlays
    (Band, Stretch, Zoom) on or off globally across all viewers.
    """
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.setFixedSize(36, 36)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.update_icon()
        
        self.setStyleSheet("""
            QToolButton {
                background: rgba(0, 0, 0, 0.4);
                color: rgba(255, 255, 255, 0.8);
                border-radius: 6px;
                font-size: 20px;
                font-weight: bold;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 0.9);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
        """)
        self.clicked.connect(self.toggle_overlays)
        self.reposition()
        self.show()

        # Wire up resize tracking to stay anchored bottom-left
        old_resize = viewer.resizeEvent
        def _resized(ev):
            try:
                self.reposition()
            except Exception:
                pass
            if callable(old_resize):
                old_resize(ev)
        viewer.resizeEvent = _resized

    def reposition(self):
        vp_geom = self.viewer.viewport().geometry()
        margin = 15
        x = vp_geom.x() + margin
        y = max(vp_geom.y(), vp_geom.bottom() - self.height() - margin + 1)
        self.move(x, y)

    def update_icon(self):
        muted = getattr(ImageViewer, "overlays_muted", False)
        # 👁 for visible, ✕ for hidden
        self.setText("👁" if not muted else "✕")
        self.setToolTip("Hide all UI Overlays" if not muted else "Show all UI Overlays")
        
    def toggle_overlays(self):
        muted = getattr(ImageViewer, "overlays_muted", False)
        ImageViewer.overlays_muted = not muted
        
        # Sync button state and overlay visibility across all viewers
        try:
            from PyQt5.QtWidgets import QApplication
            all_viewers = set()
            for w in QApplication.topLevelWidgets():
                if isinstance(w, ImageViewer):
                    all_viewers.add(w)
                all_viewers.update(w.findChildren(ImageViewer))
            
            # Fallback to self.viewer if topLevelWidgets is empty or doesn't include it
            if hasattr(self, "viewer") and self.viewer:
                all_viewers.add(self.viewer)
                
            for v in all_viewers:
                if hasattr(v, "_overlay_toggle_btn"):
                    v._overlay_toggle_btn.update_icon()
                if ImageViewer.overlays_muted:
                    if getattr(v, "_bandbar", None): v._bandbar.hide_immediately()
                    if getattr(v, "_stretchbar", None): v._stretchbar.hide_immediately()
                    if getattr(v, "_zoombar", None): v._zoombar.hide_immediately()
        except Exception:
            pass

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
        self._hide_timer.setInterval(5000)  # hide after 5 seconds of no interaction
        self._hide_timer.timeout.connect(self._do_hide)
        
        # Start hidden
        self.hide()
        
        # place immediately
        self.reposition()

        # Catch viewport resizes that don't go through the outer QGraphicsView's
        # own resizeEvent -- e.g. scrollbars appearing/disappearing as the zoom
        # level crosses the fit threshold shrinks/grows the viewport directly.
        # installEventFilter works here (unlike monkey-patching viewport()'s
        # resizeEvent) because the viewport is a plain C++-created QWidget with
        # no Python-overridable virtual table; the FILTER (this bar, a genuine
        # Python QFrame subclass) is what needs the override, not the target.
        parent_view.viewport().installEventFilter(self)

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
        if getattr(ImageViewer, "overlays_muted", False): return
        
        self.reposition()
        self.show()
        self._start_hide_timer()

    def hide_immediately(self):
        """Hide with no delay."""
        self._hide_timer.stop()
        self.hide()
    
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

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            try:
                self.reposition()
            except Exception:
                pass
        return False

    # ---------- placement ----------
    def reposition(self):
        if getattr(self, "_repositioning", False):
            return
        self._repositioning = True
        try:
            vp = self._view.viewport()
            if not vp:
                return
            m = 8
            s = self.sizeHint()
            x = max(m, vp.width() - s.width() - m)
            y = max(m, vp.height() - s.height() - m)

            bb = getattr(self._view, "_bandbar", None)
            if bb is not None:
                try:
                    if getattr(bb, "_buttons_by_band", None):
                        bb_h = bb.sizeHint().height()
                        y = max(m, vp.height() - bb_h - m - s.height() - 4)
                except Exception:
                    pass

            new_geom = QtCore.QRect(x, y, s.width(), s.height())
            if self.geometry() != new_geom:
                self.setGeometry(new_geom)
        finally:
            self._repositioning = False

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


# --- BandBar overlay for ImageViewer ------------------------------------------
class _BandBar(QtWidgets.QFrame):
    """
    Lightweight overlay widget: a horizontally scrollable strip of small buttons,
    one per band, that floats over the ImageViewer viewport. Auto-hides when not
    interacting (same pattern as _ZoomBar), and hides immediately when drawing
    starts (see ImageViewer.drawing's setter).

    Clicking a button emits ImageViewer.band_selected(file_band_index); it does
    not directly touch image_data or the pixmap itself -- ProjectTab connects to
    that signal (in display_image_group) and does the actual reload/render, since
    that needs project-level context (root_name, the .ax sidecar, the stretch
    render pipeline) this class has no business knowing about.
    """

    def __init__(self, parent_view):
        super().__init__(parent_view.viewport())
        self.setObjectName("_BandBar")
        self._view = parent_view
        self._buttons_by_band = {}   # file band index -> QToolButton

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setStyleSheet("""
            QFrame#_BandBar {
                background: rgba(245, 245, 245, 220);
                border: 1px solid rgba(180, 180, 180, 200);
                border-radius: 6px;
            }
            QToolButton {
                color: black;
                background: transparent;
                border: none;
                padding: 3px 6px;
                font-size: 13px;
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
            QToolButton#_BandBarComposite {
                border: 1px solid rgba(120, 120, 120, 160);
                border-radius: 3px;
            }
            QFrame#_BandBarSep {
                background: rgba(150, 150, 150, 160);
                max-width: 1px;
                min-width: 1px;
            }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:horizontal {
                height: 8px;
                background: transparent;
            }
            QScrollBar::handle:horizontal {
                background: rgba(0, 0, 0, 90);
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(0)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._scroll.setFixedHeight(42)

        self._inner = QtWidgets.QWidget()
        self._row = QtWidgets.QHBoxLayout(self._inner)
        self._row.setContentsMargins(3, 3, 3, 3)
        self._row.setSpacing(3)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        # Leading "composite" button: restores whatever the actual configured
        # stretch is (Auto full-band composite, or an RGB composite) -- i.e.
        # undoes single-band viewing. Persists across populate() calls; only
        # its text/checked state is updated (unlike the numbered band buttons,
        # which are rebuilt from scratch per image).
        self._composite_btn = QtWidgets.QToolButton(self._inner)
        self._composite_btn.setObjectName("_BandBarComposite")
        self._composite_btn.setText("Auto")
        self._composite_btn.setToolTip(
            "Return to the full composite stretch (Auto/RGB) configured in the Stretch dialog.")
        self._composite_btn.setCheckable(True)
        self._composite_btn.setFixedHeight(34)
        self._composite_btn.setMinimumWidth(52)
        self._composite_btn.clicked.connect(lambda checked: self._on_button_clicked(-1))
        self._group.addButton(self._composite_btn)
        self._row.addWidget(self._composite_btn)

        sep = QtWidgets.QFrame(self._inner)
        sep.setObjectName("_BandBarSep")
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFixedHeight(24)
        self._row.addWidget(sep)
        self._row.addSpacing(3)

        self._row.addStretch(1)   # keeps buttons left-aligned when the strip is wider than its content

        # --- Auto-hide timer (same 5s convention as _ZoomBar) ---
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(5000)
        self._hide_timer.timeout.connect(self._do_hide)

        self.hide()
        self.reposition()

        # Catch viewport resizes that don't go through the outer QGraphicsView's
        # own resizeEvent -- see the matching comment in _ZoomBar.__init__.
        parent_view.viewport().installEventFilter(self)

    # ---------- population ----------
    # Row layout is fixed as: [composite_btn][separator][spacing][...band btns...][stretch]
    # -- the first 3 items persist across calls; only the band buttons (and the
    # composite button's text/checked state) are rebuilt/updated here.
    _LEADING_ITEMS = 3

    def populate(self, band_names, count, active_band=None,
                 composite_label="Auto", composite_active=False):
        """Rebuild the numbered-band buttons for `count` bands (0-based indices)."""
        self.setUpdatesEnabled(False)
        try:
            self._composite_btn.setText(composite_label)
            self._composite_btn.setChecked(bool(composite_active))

            # Clear existing numbered buttons only -- composite/separator/spacing persist.
            for btn in list(self._buttons_by_band.values()):
                self._group.removeButton(btn)
                btn.setParent(None)
                btn.deleteLater()
            self._buttons_by_band = {}
            while self._row.count() > self._LEADING_ITEMS + 1:   # +1 keeps the trailing stretch
                item = self._row.takeAt(self._LEADING_ITEMS)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

            count = max(0, int(count or 0))
            for i in range(count):
                nm = band_names[i] if (band_names and i < len(band_names) and band_names[i]) else None
                btn = QtWidgets.QToolButton(self._inner)
                btn.setText(str(i + 1))
                btn.setToolTip(nm if nm else f"Band {i + 1}")
                btn.setCheckable(True)
                btn.setFixedSize(42, 34)
                btn.clicked.connect(lambda checked, band_idx=i: self._on_button_clicked(band_idx))
                self._row.insertWidget(self._row.count() - 1, btn)
                self._group.addButton(btn)
                self._buttons_by_band[i] = btn

            if active_band is not None:
                btn = self._buttons_by_band.get(int(active_band))
                if btn is not None:
                    btn.setChecked(True)

            self.adjustSize()
        finally:
            self.setUpdatesEnabled(True)

    def _on_button_clicked(self, band_idx):
        """band_idx == -1 is the sentinel for the leading composite button."""
        try:
            self._view.band_selected.emit(int(band_idx))
        except Exception as e:
            logging.debug(f"[_BandBar] band_selected emit failed: {e}")
        # Clicking is itself an interaction -- keep the bar up a while longer.
        self.show_briefly()

    # ---------- show/hide ----------
    def _do_hide(self):
        self.hide()

    def _start_hide_timer(self):
        self._hide_timer.start()

    def _reset_hide_timer(self):
        self._hide_timer.stop()

    def show_briefly(self):
        """Show the band bar and start the auto-hide timer."""
        if getattr(ImageViewer, "overlays_muted", False): return

        if not self._buttons_by_band:
            return   # nothing to show (no image loaded yet)
        self.reposition()
        self.show()
        self._start_hide_timer()

    def hide_immediately(self):
        """Hide with no delay -- used when the user starts drawing."""
        self._hide_timer.stop()
        self.hide()

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

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            try:
                self.reposition()
            except Exception:
                pass
        return False

    # ---------- placement ----------
    def reposition(self):
        if getattr(self, "_repositioning", False):
            return
        self._repositioning = True
        try:
            vp = self._view.viewport()
            if not vp:
                return
            m = 8
            s = self.sizeHint()
            target_w = max(s.width(), min(480, vp.width() - 2 * m))
            bar_w = max(s.width(), min(target_w, vp.width() - 2 * m))
            x = max(m, (vp.width() - bar_w) // 2)
            y = max(m, vp.height() - s.height() - m)
            new_geom = QtCore.QRect(x, y, bar_w, s.height())
            if self.geometry() != new_geom:
                self.setGeometry(new_geom)
        finally:
            self._repositioning = False


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
        if hasattr(viewer, "_trigger_highres_update"):
            viewer._trigger_highres_update()
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
            if not getattr(viewer, "drawing", False):
                zb.show_briefly()
        except Exception:
            pass
        try:
            if getattr(viewer, "_bandbar", None) and not getattr(viewer, "drawing", False):
                viewer._bandbar.show_briefly()
        except Exception:
            pass
        try:
            if getattr(viewer, "_stretchbar", None) and not getattr(viewer, "drawing", False):
                viewer._stretchbar.show_briefly()
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


# ---- installation helper (non-invasive): attach a _BandBar to an existing ImageViewer ----
def attach_band_bar(viewer):
    """
    Installs a _BandBar on top of `viewer`. Population happens separately via
    ImageViewer._refresh_band_bar (called from the image_data property setter)
    -- this function only creates the widget and wires resize/repositioning.
    Showing it is driven by attach_zoom_bar's own wheelEvent wrapper (see the
    guarded viewer._bandbar.show_briefly() call added there), so wheel-zoom
    remains the single shared trigger for both overlay bars.
    """
    if getattr(viewer, "_bandbar", None) is not None:
        return viewer._bandbar

    bb = _BandBar(viewer)
    viewer._bandbar = bb

    old_resize = viewer.resizeEvent
    def _resized(ev):
        try:
            bb.reposition()
        except Exception:
            pass
        if callable(old_resize):
            old_resize(ev)
    viewer.resizeEvent = _resized

    bb.reposition()
    bb.hide()  # Start hidden, will show on interaction

    # Now that the band bar exists, let the zoom bar make room above it.
    zb = getattr(viewer, "_zoombar", None)
    if zb is not None:
        try:
            zb.reposition()
        except Exception:
            pass

    # If an image was already assigned before this deferred attach ran,
    # populate immediately instead of waiting for the next image_data set.
    try:
        viewer._refresh_band_bar()
    except Exception:
        pass

    return bb


def _make_stretch_params(**kwargs):
    """Helper to instantiate _StretchParams from project_tab or fallback container."""
    try:
        from .project_tab import _StretchParams
        return _StretchParams(**kwargs)
    except Exception:
        class _Params:
            def __init__(self, **kw):
                self.mode = kw.get("mode", "percentile")
                self.low_p = float(kw.get("low_p", 0.5))
                self.high_p = float(kw.get("high_p", 99.5))
                self.k_sigma = float(kw.get("k_sigma", 1.0))
                self.min_val = kw.get("min_val", None)
                self.max_val = kw.get("max_val", None)
                self.per_channel = bool(kw.get("per_channel", True))
                self.clip = bool(kw.get("clip", True))
                self.scope = kw.get("scope", "viewer")
                self.display_band = kw.get("display_band", None)
                self.display_mode = kw.get("display_mode", "auto")
                self.r_band = kw.get("r_band", None)
                self.g_band = kw.get("g_band", None)
                self.b_band = kw.get("b_band", None)
        return _Params(**kwargs)


# --- StretchBar overlay for ImageViewer ----------------------------------------
class _StretchBar(QtWidgets.QFrame):
    """
    Lightweight overlay widget: docked to the top edge of the ImageViewer viewport.
    Allows the user to adjust contrast stretch using the absolute data range of
    the image via Min/Max sliders.
    Auto-hides when not interacting, matching _BandBar and _ZoomBar.
    """

    def __init__(self, parent_view):
        super().__init__(parent_view.viewport())
        self.setObjectName("_StretchBar")
        self._view = parent_view
        self._data_min = 0.0
        self._data_max = 255.0
        self._block_signals = False

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setStyleSheet("""
            QFrame#_StretchBar {
                background: rgba(245, 245, 245, 220);
                border: 1px solid rgba(180, 180, 180, 200);
                border-radius: 6px;
            }
            QLabel {
                color: black;
                font-size: 11px;
                font-weight: bold;
            }
            QToolButton {
                color: black;
                background: transparent;
                border: 1px solid rgba(120, 120, 120, 160);
                border-radius: 3px;
                padding: 1px 5px;
                font-size: 11px;
                font-weight: bold;
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 30);
            }
            QToolButton:pressed {
                background: rgba(0, 0, 0, 50);
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(180, 180, 180, 200);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4682B4;
                border: 1px solid #2B547E;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #1E90FF;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Stretch:", self)
        layout.addWidget(title)

        lbl_min = QtWidgets.QLabel("Min", self)
        layout.addWidget(lbl_min)
        self._slider_min = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self._slider_min.setRange(0, 1000)
        self._slider_min.setValue(0)
        self._slider_min.setMinimumWidth(20)
        self._slider_min.setMaximumWidth(75)
        layout.addWidget(self._slider_min)

        lbl_max = QtWidgets.QLabel("Max", self)
        layout.addWidget(lbl_max)
        self._slider_max = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self._slider_max.setRange(0, 1000)
        self._slider_max.setValue(1000)
        self._slider_max.setMinimumWidth(20)
        self._slider_max.setMaximumWidth(75)
        layout.addWidget(self._slider_max)

        self._lbl_vals = QtWidgets.QLabel("–", self)
        self._lbl_vals.setMinimumWidth(40)
        self._lbl_vals.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self._lbl_vals)

        self._btn_auto = QtWidgets.QToolButton(self)
        self._btn_auto.setText("Auto")
        self._btn_auto.setToolTip("Reset to default auto percentile stretch")
        self._btn_auto.clicked.connect(self._on_auto_clicked)
        layout.addWidget(self._btn_auto)

        self._slider_min.valueChanged.connect(self._on_slider_changed)
        self._slider_max.valueChanged.connect(self._on_slider_changed)
        self._slider_min.sliderReleased.connect(self._apply_stretch)
        self._slider_max.sliderReleased.connect(self._apply_stretch)

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(5000)
        self._hide_timer.timeout.connect(self._do_hide)

        self.hide()
        self.reposition()
        parent_view.viewport().installEventFilter(self)

    def _current_band_key(self):
        """Identity of the band context the bar is currently showing.

        ("single", <FILE band index>) while one band is displayed, else
        ("composite",). The FILE band index is used rather than the channel
        position inside the resident array because a preview-loaded cube swaps
        its single resident channel out on every band switch (see
        ProjectTab._on_band_bar_clicked -> set_viewer_preview_image), so
        position 0 means a different band each time and caching on position
        would collide across bands.
        """
        view = self._view
        sp = getattr(view, "stretch_params", None)
        pos = None
        if (sp is not None
                and str(getattr(sp, "display_mode", "")).lower() == "single"
                and getattr(sp, "display_band", None) is not None):
            try:
                pos = int(sp.display_band)
            except (TypeError, ValueError):
                pos = None
        if pos is None:
            return ("composite",)

        idata = getattr(view, "image_data", None)
        preview_bands = getattr(idata, "preview_bands", None) if idata is not None else None
        if preview_bands:
            try:
                if 0 <= pos < len(preview_bands):
                    return ("single", int(preview_bands[pos]))
            except (TypeError, ValueError):
                pass
        return ("single", pos)

    def _nodata_values(self):
        """NoData literals for the current file: in-memory ax_config, else .ax."""
        idata = getattr(self._view, "image_data", None)
        try:
            ax_cfg = getattr(idata, "ax_config", None) or {}
            if ax_cfg.get("nodata_enabled", True):
                vals = list(ax_cfg.get("nodata_values", []) or [])
                if vals:
                    return vals
        except Exception:
            pass
        try:
            fp = getattr(idata, "filepath", None)
            if fp:
                ax_path = os.path.splitext(fp)[0] + ".ax"
                if os.path.exists(ax_path):
                    with open(ax_path, "r", encoding="utf-8") as f:
                        ax_cfg = json.load(f) or {}
                    if ax_cfg.get("nodata_enabled", True):
                        return list(ax_cfg.get("nodata_values", []) or [])
        except Exception:
            pass
        return []

    def _stats_slice_for_key(self, img, key):
        """The pixels whose min/max the bar should report for band context `key`."""
        if key[0] == "single" and getattr(img, "ndim", 0) == 3:
            # The resident array is what is on screen, so the channel POSITION
            # indexes it -- not the file band index carried in the cache key.
            sp = getattr(self._view, "stretch_params", None)
            try:
                pos = int(getattr(sp, "display_band", 0) or 0)
            except (TypeError, ValueError):
                pos = 0
            if 0 <= pos < img.shape[2]:
                return img[:, :, pos]
        try:
            from .project_tab import _preview_take3
            return _preview_take3(img, prefer_last_band=False)
        except Exception:
            return img

    def _compute_data_range(self, arr):
        """Min/max over `arr`, sampled and NoData-filtered exactly the way
        ImageStretchDialog computes the range it displays -- same k=400 stride
        sampler, same .ax NoData exclusion -- so the bar's bounds always agree
        with the Stretch Viewer's. Returns None when nothing is usable."""
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim >= 2:
            H, W = a.shape[:2]
            stride = max(1, int(np.sqrt((H * W) / 400.0)))
            s = a[::stride, ::stride, ...] if a.ndim == 3 else a[::stride, ::stride]
        else:
            s = a
        if s.size == 0:
            return None

        finite_mask = np.isfinite(s)
        for nd in self._nodata_values():
            try:
                nd_val = float(nd)
            except (ValueError, TypeError):
                continue        # an expression rule, not a value to drop here
            if not np.isnan(nd_val):
                finite_mask &= (s != nd_val)

        if np.any(finite_mask):
            return float(np.nanmin(s[finite_mask])), float(np.nanmax(s[finite_mask]))
        if np.any(np.isfinite(s)):
            return float(np.nanmin(s)), float(np.nanmax(s))
        return None

    def _range_cache(self):
        """Per-band range cache for the file currently loaded in the viewer.

        Reset whenever the viewer moves to a different file -- viewers are
        reused as the user pages through roots, so a cache keyed only by band
        would hand the next file the previous file's bounds."""
        view = self._view
        idata = getattr(view, "image_data", None)
        fp = getattr(idata, "filepath", None) if idata is not None else None
        cache = getattr(view, "_stretch_data_ranges", None)
        if not isinstance(cache, dict) or getattr(view, "_stretch_data_ranges_fp", None) != fp:
            cache = {}
            view._stretch_data_ranges = cache
            view._stretch_data_ranges_fp = fp
        return cache

    def seed_range(self, vmin, vmax):
        """Adopt a data range computed elsewhere (the Stretch Viewer dialog's
        _real_min/_real_max) for the band context currently on screen, so the
        bar and the dialog never disagree about the bounds they are showing."""
        try:
            vmin, vmax = float(vmin), float(vmax)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return
        self._range_cache()[self._current_band_key()] = (vmin, vmax)
        self.refresh_range()

    def refresh_range(self):
        """Set the slider bounds to the data range of the band CURRENTLY on screen.

        The bounds used to come from one pair of `_stretch_data_min` /
        `_stretch_data_max` attributes cached on the viewer, computed once over
        `_preview_take3` (the first three channels) and then reused
        unconditionally for the rest of the session. Switching bands with the
        band-selector bar therefore left the Min/Max readout on the previous
        band's range -- and on a cube whose band 1 is reflectance (0-1) while
        band 11 is a sensor azimuth angle (0-360), the sliders were mapping
        over an interval that had nothing to do with the pixels on screen.
        The range is now derived from, and cached per, the displayed band.
        """
        idata = getattr(self._view, "image_data", None)
        img = getattr(idata, "image", None) if idata is not None else None
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return

        key = self._current_band_key()
        cache = self._range_cache()
        rng = cache.get(key)
        if rng is None:
            try:
                rng = self._compute_data_range(self._stats_slice_for_key(img, key))
            except Exception as e:
                logging.debug(f"[_StretchBar] data range computation failed: {e}")
                rng = None
            if rng is None:
                rng = (0.0, 255.0)
            cache[key] = rng

        self._data_min, self._data_max = float(rng[0]), float(rng[1])
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1.0

        # --- position sliders to match current stretch_params ---
        sp = getattr(self._view, "stretch_params", None)
        b_min = self._slider_min.blockSignals(True)
        b_max = self._slider_max.blockSignals(True)
        try:
            if sp is not None and getattr(sp, "mode", "") == "absolute":
                # Use current absolute bounds (prefer band_mins/band_maxs for per-ch)
                bm = getattr(sp, "band_mins", None)
                bx = getattr(sp, "band_maxs", None)
                if bm and bx:
                    c_min = float(min(bm))
                    c_max = float(max(bx))
                elif sp.min_val is not None and sp.max_val is not None:
                    c_min = float(sp.min_val)
                    c_max = float(sp.max_val)
                else:
                    c_min, c_max = self._data_min, self._data_max
            else:
                # Non-absolute stretch (auto/percentile): sliders show full data range
                c_min, c_max = self._data_min, self._data_max

            range_len = max(1e-6, self._data_max - self._data_min)
            pos_min = int(round(max(0.0, min(1.0, (c_min - self._data_min) / range_len)) * 1000))
            pos_max = int(round(max(0.0, min(1.0, (c_max - self._data_min) / range_len)) * 1000))
            self._slider_min.setValue(pos_min)
            self._slider_max.setValue(pos_max)
            self._lbl_vals.setText(f"{self._format_val(c_min)} – {self._format_val(c_max)}")
        finally:
            self._slider_min.blockSignals(b_min)
            self._slider_max.blockSignals(b_max)

    def _pos_to_val(self, pos):
        frac = pos / 1000.0
        return self._data_min + frac * (self._data_max - self._data_min)

    def _format_val(self, val):
        try:
            img = getattr(getattr(self._view, "image_data", None), "image", None)
            if img is not None and img.dtype.kind in ('i', 'u'):
                return f"{int(round(val))}"
        except Exception:
            pass
        if getattr(self, "_data_max", 0) - getattr(self, "_data_min", 0) > 255:
            return f"{int(round(val))}"
        return f"{val:.4g}"

    def _on_slider_changed(self):
        pmin = self._slider_min.value()
        pmax = self._slider_max.value()
        if pmin > pmax:
            b_min = self._slider_min.blockSignals(True)
            b_max = self._slider_max.blockSignals(True)
            try:
                if self.sender() == self._slider_min:
                    self._slider_max.setValue(pmin)
                    pmax = pmin
                else:
                    self._slider_min.setValue(pmax)
                    pmin = pmax
            finally:
                self._slider_min.blockSignals(b_min)
                self._slider_max.blockSignals(b_max)

        c_min = self._pos_to_val(pmin)
        c_max = self._pos_to_val(pmax)
        # Formatted dynamically to prevent UI overlap for integers/large floats
        self._lbl_vals.setText(f"{self._format_val(c_min)} – {self._format_val(c_max)}")
        self.show_briefly()

    def _apply_stretch(self):
        """Apply absolute stretch — preserves all display_mode/band settings from the
        current stretch_params so the bar never changes which band/composite is shown."""
        pmin = self._slider_min.value()
        pmax = self._slider_max.value()
        c_min = self._pos_to_val(pmin)
        c_max = self._pos_to_val(pmax)

        old_sp = getattr(self._view, "stretch_params", None)
        # Carry over every display setting — only override the stretch bounds.
        disp_mode = getattr(old_sp, "display_mode", "auto") if old_sp else "auto"
        disp_band = getattr(old_sp, "display_band", None) if old_sp else None
        r_band = getattr(old_sp, "r_band", None) if old_sp else None
        g_band = getattr(old_sp, "g_band", None) if old_sp else None
        b_band = getattr(old_sp, "b_band", None) if old_sp else None
        per_channel = bool(getattr(old_sp, "per_channel", True)) if old_sp else True
        clip = bool(getattr(old_sp, "clip", True)) if old_sp else True

        # Scale any existing band_mins/band_maxs proportionally so per-channel
        # rendering still honours the individual-band ranges.
        band_mins = None
        band_maxs = None
        old_bm = getattr(old_sp, "band_mins", None) if old_sp else None
        old_bx = getattr(old_sp, "band_maxs", None) if old_sp else None
        if old_bm and old_bx and len(old_bm) == len(old_bx):
            old_lo = min(old_bm)
            old_hi = max(old_bx)
            old_range = max(1e-9, old_hi - old_lo)
            new_range = max(1e-9, c_max - c_min)
            scale = new_range / old_range
            band_mins = [c_min + (v - old_lo) * scale for v in old_bm]
            band_maxs = [c_min + (v - old_lo) * scale for v in old_bx]

        params = _make_stretch_params(
            mode="absolute",
            min_val=c_min,
            max_val=c_max,
            per_channel=per_channel,
            clip=clip,
            scope="viewer",
            display_mode=disp_mode,
            display_band=disp_band,
            r_band=r_band,
            g_band=g_band,
            b_band=b_band,
        )
        if band_mins is not None:
            params.band_mins = band_mins
            params.band_maxs = band_maxs

        self._view.stretch_params = params
        try:
            self._view.stretch_applied.emit(params)
        except Exception as e:
            logging.debug(f"[_StretchBar] stretch_applied emit failed: {e}")

    def _on_auto_clicked(self):
        """Restore the saved stretch (file → root → project) without wiping band selection.
        Emits None so the ProjectTab re-runs its normal stretch lookup chain."""
        # Clear only the in-memory override; ProjectTab._on_stretch_apply(None) will
        # fall back through file/root/project saved stretch, preserving band selection.
        self._view.stretch_params = None
        self.refresh_range()
        try:
            self._view.stretch_applied.emit(None)
        except Exception as e:
            logging.debug(f"[_StretchBar] stretch_applied auto emit failed: {e}")
        self.show_briefly()

    def _do_hide(self):
        self.hide()

    def _start_hide_timer(self):
        self._hide_timer.start()

    def show_briefly(self):
        if getattr(ImageViewer, "overlays_muted", False): return
        
        idata = getattr(self._view, "image_data", None)
        if idata is None or getattr(idata, "image", None) is None:
            return
        self.reposition()
        self.show()
        self._start_hide_timer()

    def hide_immediately(self):
        self._hide_timer.stop()
        self.hide()

    def enterEvent(self, event):
        self._hide_timer.stop()
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_hide_timer()
        self.unsetCursor()
        super().leaveEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            try:
                self.reposition()
            except Exception:
                pass
        return False

    def reposition(self):
        if getattr(self, "_repositioning", False):
            return
        self._repositioning = True
        try:
            vp = self._view.viewport()
            if not vp:
                return
            m = 8
            s = self.sizeHint()
            target_w = max(s.width(), min(480, vp.width() - 2 * m))
            bar_w = max(s.width(), min(target_w, vp.width() - 2 * m))
            x = max(m, (vp.width() - bar_w) // 2)
            y = m
            new_geom = QtCore.QRect(x, y, bar_w, s.height())
            if self.geometry() != new_geom:
                self.setGeometry(new_geom)
        finally:
            self._repositioning = False


def attach_stretch_bar(viewer):
    """
    Installs a _StretchBar on top of `viewer`. Docked to the top edge of the viewport.
    """
    if getattr(viewer, "_stretchbar", None) is not None:
        return viewer._stretchbar

    sb = _StretchBar(viewer)
    viewer._stretchbar = sb

    old_resize = viewer.resizeEvent
    def _resized(ev):
        try:
            sb.reposition()
        except Exception:
            pass
        if callable(old_resize):
            old_resize(ev)
    viewer.resizeEvent = _resized

    sb.reposition()
    sb.hide()

    try:
        viewer._refresh_stretch_bar()
    except Exception:
        pass

    return sb

