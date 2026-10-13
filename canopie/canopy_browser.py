"""
Canopy browser: the same crown side by side across many images.

One tile per image -- the source first, then every image the crown was found
in (best score first) -- in pages of cols x rows with a bar to move through
them, so 100 images across roots / projects stay cheap: only the visible page
is read, and each tile holds a small window, never the whole image.

The outline in a tile IS the record: dragging a vertex (Edit outlines on)
updates the stored polygon of that image (current project only) and its grid
viewer; for a result not saved yet it updates what Save will write. "Sync
zoom" keeps the same ground extent (metres) and offset from the crown in every
tile, whatever each image's resolution.

A tile can be ROTATED (its buttons, or Ctrl + wheel): only the view turns --
the image file and its .ax are untouched. The scene stays in the image's own
(post-.ax) pixels, so a crown dragged or reshaped in a rotated tile moves in
those pixels and the record lands where it was put on the image.
"""
import logging
import math
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import canopy_finder as CF
from . import canopy_finder_dialog as D


def _qpixmap(u8):
    a = np.ascontiguousarray(u8[:, :, :3] if u8.shape[2] >= 3 else np.repeat(u8[:, :, :1], 3, axis=2))
    h, w = a.shape[:2]
    img = QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img.copy())


class _Handle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, tile, idx, x, y, r):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.tile, self.idx = tile, idx
        self.setPos(x, y)
        self.setBrush(QtGui.QBrush(QtGui.QColor("#ffeb3b")))
        pen = QtGui.QPen(QtGui.QColor("#000"), 0)
        self.setPen(pen)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
                      | QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(10)
        self.setCursor(QtCore.Qt.SizeAllCursor)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.tile._handle_moved(self.idx, self.pos())
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.tile._edit_finished()


class _Outline(QtWidgets.QGraphicsPolygonItem):
    """The crown in a tile: drag inside it to move the whole crown (the tile pans
    when the drag starts outside it). It used to move only with "Edit outlines"
    ticked -- otherwise the drag panned -- which read as "can't move polygons"."""

    def __init__(self, tile, poly, pen):
        super().__init__(poly)
        self.tile = tile
        self.setPen(pen)

    def set_movable(self, on):
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, bool(on))
        self.setBrush(QtGui.QBrush(QtGui.QColor(118, 255, 3, 30)) if on else QtGui.QBrush(QtCore.Qt.NoBrush))
        self.setCursor(QtCore.Qt.SizeAllCursor if on else QtCore.Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        d = self.pos()
        if abs(d.x()) > 1e-9 or abs(d.y()) > 1e-9:
            self.tile._outline_moved(d.x(), d.y())


class CanopyTile(QtWidgets.QGraphicsView):
    view_changed = QtCore.pyqtSignal(object)
    edited = QtCore.pyqtSignal(object)

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QColor("#202020"))
        self._syncing = False
        self._handles = []
        self.poly_item = None
        self._fitted = False

    # ---- rotation of the view (the image is not touched) ----
    def view_scale(self):
        """View pixels per scene pixel, whatever the rotation (m11 is s * cos)."""
        t = self.transform()
        return math.sqrt(abs(t.m11() * t.m22() - t.m12() * t.m21())) or 1.0

    def view_rotation(self):
        return float(self.entry.get("view_rot", 0.0) or 0.0)

    def set_view_rotation(self, deg):
        """Turn the view to `deg` (clockwise on screen) about the view centre."""
        deg = float(deg) % 360.0
        c = self.mapToScene(self.viewport().rect().center())
        s = self.view_scale()
        self.setTransform(QtGui.QTransform().rotate(deg).scale(s, s))
        self.centerOn(c)
        self.entry["view_rot"] = deg
        self.view_changed.emit(self)

    def rotate_by(self, ddeg):
        self.set_view_rotation(self.view_rotation() + ddeg)

    # entry: fp, x0, y0, pixmap, pts (N,2 raw), gsd
    def load(self, pixmap):
        sc = self.scene()
        sc.clear()
        self._handles = []
        sc.addPixmap(pixmap)
        sc.setSceneRect(QtCore.QRectF(pixmap.rect()))
        pen = QtGui.QPen(QtGui.QColor("#00e5ff" if self.entry.get("kind") == "source" else "#76ff03"), 0)
        pen.setCosmetic(True); pen.setWidthF(2.0)
        self.poly_item = _Outline(self, self._local_poly(), pen)
        sc.addItem(self.poly_item)
        self.poly_item.setZValue(5)
        self.poly_item.set_movable(True)          # drag inside the crown moves it; outside pans
        self._fitted = False
        if self.view_rotation():
            self.setTransform(QtGui.QTransform().rotate(self.view_rotation()))
        QtCore.QTimer.singleShot(0, self.fit)

    def _local_pts(self):
        return np.asarray(self.entry["pts"], np.float64) - (self.entry["x0"], self.entry["y0"])

    def _local_poly(self):
        return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self._local_pts()])

    def fit(self):
        if self.scene() is None or self.poly_item is None:
            return
        r = self.poly_item.boundingRect()
        pad = max(r.width(), r.height()) * 0.6 + 10
        self.fitInView(r.adjusted(-pad, -pad, pad, pad), QtCore.Qt.KeepAspectRatio)
        self._fitted = True

    def set_editable(self, on):
        sc = self.scene()
        for h in self._handles:
            sc.removeItem(h)
        self._handles = []
        if on and self.poly_item is not None:
            for i, (x, y) in enumerate(self._local_pts()):
                h = _Handle(self, i, x, y, 4)
                sc.addItem(h)
                self._handles.append(h)

    def _handle_moved(self, idx, pos):
        pts = np.asarray(self.entry["pts"], np.float64).copy()
        pts[idx] = (pos.x() + self.entry["x0"], pos.y() + self.entry["y0"])
        self.entry["pts"] = pts
        if self.poly_item is not None:
            self.poly_item.setPolygon(self._local_poly())

    def _outline_moved(self, dx, dy):
        """The whole outline was dragged by (dx, dy) tile pixels (= raw pixels)."""
        self.entry["pts"] = np.asarray(self.entry["pts"], np.float64) + (dx, dy)
        self.poly_item.setPos(0, 0)
        self.poly_item.setPolygon(self._local_poly())
        editing = bool(self._handles)
        self.set_editable(editing)
        self.edited.emit(self)

    def _edit_finished(self):
        self.edited.emit(self)

    # zoom / pan
    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            self.rotate_by(5.0 if ev.angleDelta().y() > 0 else -5.0)
            return
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self.view_changed.emit(self)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.view_changed.emit(self)

    def mouseDoubleClickEvent(self, ev):
        self.fit()
        self.view_changed.emit(self)

    def ground_state(self):
        """(metres per view pixel, view-centre offset from the crown centroid in metres)."""
        s = self.view_scale()
        g = self.entry["gsd"]
        c = self.mapToScene(self.viewport().rect().center())
        cx, cy = self._local_pts().mean(axis=0)
        return g / s, ((c.x() - cx) * g, (c.y() - cy) * g)

    def apply_ground_state(self, mpp, off):
        g = self.entry["gsd"]
        s = g / max(1e-9, mpp)
        cur = self.view_scale()
        self._syncing = True
        try:
            self.scale(s / cur, s / cur)
            cx, cy = self._local_pts().mean(axis=0)
            self.centerOn(cx + off[0] / g, cy + off[1] / g)
        finally:
            self._syncing = False


class CanopyBrowser(QtWidgets.QWidget):
    def __init__(self, finder_dialog):
        super().__init__(finder_dialog, QtCore.Qt.Window)
        self.dlg = finder_dialog
        self.pt = finder_dialog.pt
        self.setWindowTitle("Canopy browser")
        self.resize(1300, 820)
        self.setStyleSheet(D.STYLE)
        self.ctxs, self.results = {}, []
        self.entries = []
        self.visible = []
        self.page = 0
        self._pix_cache = {}
        self.tiles = []
        self._build_ui()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.crown = QtWidgets.QComboBox()
        self.crown.setToolTip("Which crown's images to show")
        self.crown.currentIndexChanged.connect(lambda _i: self._crown_changed())
        self.title = QtWidgets.QLabel()
        top.addWidget(QtWidgets.QLabel("Crown:")); top.addWidget(self.crown); top.addWidget(self.title, 1)
        lay.addLayout(top)
        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setSpacing(6)
        lay.addWidget(self.grid_host, 1)
        bar = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.page_lbl = QtWidgets.QLabel()
        self.cols = QtWidgets.QSpinBox(); self.cols.setRange(1, 8); self.cols.setValue(3); self.cols.setPrefix("cols ")
        self.rows = QtWidgets.QSpinBox(); self.rows.setRange(1, 6); self.rows.setValue(2); self.rows.setPrefix("rows ")
        self.sync = QtWidgets.QCheckBox("Sync zoom"); self.sync.setChecked(True)
        self.edit = QtWidgets.QCheckBox("Edit vertices")
        self.edit.setToolTip("Show the vertex handles (drag one to reshape). Drag inside an outline "
                             "to move it, outside to pan; the wheel zooms, Ctrl + wheel rotates the "
                             "view, a double-click refits.")
        self.save_all_btn = QtWidgets.QPushButton("Save all")
        self.save_all_btn.setToolTip("Save every outline not saved yet on this project's images")
        self.save_all_btn.clicked.connect(self._save_all)
        self.export_btn = QtWidgets.QPushButton("Export this crown…")
        self.export_btn.setToolTip("Registration squares / aligned crown stack of the crown shown "
                                   "(the dialog's export options)")
        self.export_btn.clicked.connect(lambda: self.dlg.export(groups=[self.group]))
        self.filter = QtWidgets.QComboBox()
        for label, key in (("All", "all"), ("Found", "found"), ("Low score", "low_score"),
                           ("Saved", "saved"), ("Not saved", "unsaved")):
            self.filter.addItem(label, key)
        self.prev_btn.clicked.connect(lambda: self.set_page(self.page - 1))
        self.next_btn.clicked.connect(lambda: self.set_page(self.page + 1))
        self.slider.valueChanged.connect(self.set_page)
        self.cols.valueChanged.connect(lambda _v: self.set_page(0, force=True))
        self.rows.valueChanged.connect(lambda _v: self.set_page(0, force=True))
        self.filter.currentIndexChanged.connect(lambda _i: self._apply_filter())
        self.edit.toggled.connect(lambda on: [t.set_editable(on) for t, _w in self.tiles])
        for w in (self.prev_btn, self.slider, self.next_btn, self.page_lbl, self.cols, self.rows,
                  self.filter, self.sync, self.edit, self.save_all_btn, self.export_btn):
            bar.addWidget(w, 1 if w is self.slider else 0)
        lay.addLayout(bar)

    # ---------------- entries ----------------
    def set_entries(self, ctxs, results, src_rec=None, group=None):
        """`ctxs`: {crown group: SourceContext} (or one SourceContext)."""
        if not isinstance(ctxs, dict):
            ctxs = {ctxs.group: ctxs}
        self.ctxs, self.results = dict(ctxs), results
        self.crown.blockSignals(True)
        self.crown.clear()
        for g in self.ctxs:
            n = sum(1 for r, t, _s in results if r.has_polygon and t.get("group", g) == g)
            self.crown.addItem(f"{g}  ({n})", g)
        want = group if group in self.ctxs else next(iter(self.ctxs))
        self.crown.setCurrentIndex(max(0, self.crown.findData(want)))
        self.crown.blockSignals(False)
        self._show_crown(want)

    def _crown_changed(self):
        g = self.crown.currentData()
        if g is not None and g in self.ctxs:
            self._show_crown(g)

    def _show_crown(self, g):
        ctx = self.ctxs[g]
        results = self.results
        self.ctx = ctx
        self.group = ctx.group
        r0, c0, r1, c1 = ctx.bbox
        src = {"kind": "source", "fp": ctx.fp, "label": "SOURCE", "pts": np.asarray(ctx.raw_pts),
               "x0": c0, "y0": r0, "x1": c1, "y1": r1, "gsd": float(min(ctx.gsd)), "row": None,
               "external": False, "status": "source", "score": None, "ax": ctx.ax or {}}
        out = [src]
        rows = sorted(((i, r, t, s) for i, (r, t, s) in enumerate(results)
                       if r.has_polygon and t.get("group", self.group) == self.group),
                      key=lambda e: -(e[1].score if e[1].score is not None else -2))
        for i, r, t, s in rows:
            x0, y0, x1, y1 = r.square
            m = int(0.15 * max(x1 - x0, y1 - y0))
            tg = r.extra.get("frame") or CF.frame_for(r.target_fp, fetch_missing=False, ax=t.get("ax") or {})
            try:
                g = float(min(CF.gsd_m(tg, *np.asarray(r.raw_pts).mean(axis=0))))
            except Exception:
                g = float(min(ctx.gsd))
            out.append({"kind": "target", "fp": r.target_fp, "label": t.get("root", ""),
                        "pts": np.asarray(r.raw_pts, np.float64), "x0": x0 - m, "y0": y0 - m,
                        "x1": x1 + m, "y1": y1 + m, "gsd": g, "row": i, "external": bool(t.get("external")),
                        "status": r.status, "score": r.score, "ax": t.get("ax") or {}})
        self.entries = out
        self.title.setText(f"source {os.path.basename(ctx.fp)} — {len(out) - 1} image(s) with the crown")
        self._apply_filter()

    def _saved_state(self, e):
        if e["kind"] == "source":
            return "saved"
        return self.dlg.results[e["row"]][2]

    def _apply_filter(self):
        key = self.filter.currentData()
        vis = []
        for e in self.entries:
            st = self._saved_state(e)
            if e["kind"] == "source" or key == "all" or (key in ("found", "low_score") and e["status"] == key) \
                    or (key == "saved" and st in ("saved", "replaced", "browser")) \
                    or (key == "unsaved" and st not in ("saved", "replaced", "browser")):
                vis.append(e)
        self.visible = vis
        self.set_page(0, force=True)

    def per_page(self):
        return self.cols.value() * self.rows.value()

    def n_pages(self):
        return max(1, int(math.ceil(len(self.visible) / float(self.per_page()))))

    def focus_file(self, fp):
        n = D._norm(fp)
        for i, e in enumerate(self.visible):
            if D._norm(e["fp"]) == n:
                self.set_page(i // self.per_page())
                return

    def set_page(self, p, force=False):
        p = max(0, min(int(p), self.n_pages() - 1))
        if p == self.page and not force and self.tiles:
            return
        self.page = p
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_pages() - 1); self.slider.setValue(p)
        self.slider.blockSignals(False)
        self.page_lbl.setText(f"page {p + 1} / {self.n_pages()}  ({len(self.visible)} tiles)")
        self.prev_btn.setEnabled(p > 0); self.next_btn.setEnabled(p < self.n_pages() - 1)
        self._build_page()

    def _pixmap_for(self, e):
        key = (D._norm(e["fp"]), e["x0"], e["y0"], e["x1"], e["y1"])
        pm = self._pix_cache.get(key)
        if pm is None:
            # the post-.ax frame: oriented, cropped and stretched-from like the viewer
            arr, v, _f, _o = self.dlg.frames.read(e["fp"], e.get("ax") or {}, e["x0"], e["y0"], e["x1"], e["y1"],
                                                  self.ctx.bands)
            pm = _qpixmap(CF.to_uint8(arr, v))
            if len(self._pix_cache) > 60:
                self._pix_cache.clear()
            self._pix_cache[key] = pm
        return pm

    def _build_page(self):
        for t, w in self.tiles:
            w.setParent(None)
            w.deleteLater()
        self.tiles = []
        cols = self.cols.value()
        start = self.page * self.per_page()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for k, e in enumerate(self.visible[start:start + self.per_page()]):
                box = QtWidgets.QFrame()
                box.setFrameShape(QtWidgets.QFrame.StyledPanel)
                v = QtWidgets.QVBoxLayout(box)
                v.setContentsMargins(3, 3, 3, 3)
                v.setSpacing(2)
                head = QtWidgets.QLabel(self._head_text(e))
                head.setToolTip(e["fp"])
                v.addWidget(head)
                tile = CanopyTile(e)
                try:
                    tile.load(self._pixmap_for(e))
                except Exception as ex:
                    logging.exception("[canopy_browser] %s", e["fp"])
                    head.setText(head.text() + f"  <span style='color:#c62828'>read failed: {ex}</span>")
                tile.view_changed.connect(self._on_view_changed)
                tile.edited.connect(self._on_edited)
                v.addWidget(tile, 1)
                btns = QtWidgets.QHBoxLayout()
                b_save = QtWidgets.QPushButton("Save")
                b_rej = QtWidgets.QPushButton("Reject")
                b_grid = QtWidgets.QPushButton("Show in grid")
                st = self._saved_state(e)
                b_save.setEnabled(e["kind"] == "target" and not e["external"]
                                  and st not in ("saved", "replaced", "browser"))
                b_rej.setEnabled(e["kind"] == "target")
                b_grid.setEnabled(not e["external"])
                b_save.clicked.connect(lambda _c=False, e=e: self._save(e))
                b_rej.clicked.connect(lambda _c=False, e=e: self._reject(e))
                b_grid.clicked.connect(lambda _c=False, e=e: self._show_in_grid(e))
                b_rl = QtWidgets.QPushButton("⟲")
                b_rr = QtWidgets.QPushButton("⟳")
                for b, d, tip in ((b_rl, -90.0, "anticlockwise"), (b_rr, 90.0, "clockwise")):
                    b.setFixedWidth(30)
                    b.setToolTip(f"Rotate this view 90° {tip} (Ctrl + wheel: 5°). The image file is not "
                                 f"changed; a crown moved here lands where you put it on the image.")
                    b.clicked.connect(lambda _c=False, t=tile, d=d: t.rotate_by(d))
                for b in (b_save, b_rej, b_grid, b_rl, b_rr):
                    btns.addWidget(b)
                v.addLayout(btns)
                self.grid.addWidget(box, k // cols, k % cols)
                self.tiles.append((tile, box))
                if self.edit.isChecked():
                    tile.set_editable(True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _head_text(self, e):
        name = os.path.basename(e["fp"])
        if e["kind"] == "source":
            return f"<b>{name}</b> <span style='color:#00838f'>source</span>"
        sc = "" if e["score"] is None else f" · score {e['score']:.2f}"
        st = self._saved_state(e)
        ext = " · other project" if e["external"] else ""
        return (f"<b>{name}</b> · {e['label']} · {D.STATUS_TEXT.get(e['status'], e['status'])}{sc}"
                f"{' · ' + st if st and st != 'external' else ''}{ext}")

    # ---------------- interaction ----------------
    def _on_view_changed(self, tile):
        if not self.sync.isChecked() or tile._syncing:
            return
        mpp, off = tile.ground_state()
        for t, _w in self.tiles:
            if t is not tile:
                t.apply_ground_state(mpp, off)

    def _record_space(self, e):
        tg = CF.frame_for(e["fp"], fetch_missing=False, ax=e.get("ax") or {})
        return tg.to_record(e["pts"])

    def _on_edited(self, tile):
        e = tile.entry
        if e["external"]:
            self._update_result(e)
            return
        pts, ref, _inside = self._record_space(e)
        st = self._saved_state(e)
        if st in ("saved", "replaced", "browser"):
            if D.update_record_points(self.pt, self.group, e["fp"], pts, ref):
                D.refresh_after_save(self.pt, [e["fp"]])
        if e["kind"] == "target":
            self._update_result(e, pts, ref)

    def _update_result(self, e, pts=None, ref=None):
        r = self.dlg.results[e["row"]][0]
        r.raw_pts = np.asarray(e["pts"], np.float64)
        if pts is not None:
            r.record_points, r.image_ref_size = pts, ref

    def refresh_page(self):
        self._build_page()

    def result_changed(self, row):
        """The dialog moved a result (dragged in a grid viewer): follow it."""
        r = self.dlg.results[row][0]
        for e in self.entries:
            if e.get("row") == row and r.raw_pts is not None:
                e["pts"] = np.asarray(r.raw_pts, np.float64)
        self._build_page()

    def _save_all(self):
        self.dlg.save_all()
        self._build_page()

    def _save(self, e):
        pts, ref, inside = self._record_space(e)
        self._update_result(e, pts, ref)
        st = D.save_found_record(self.pt, self.group, e["fp"], pts, ref, getattr(self.ctx, "shape_type", "polygon"),
                                 replace=False)
        if st == "exists":
            ans = QtWidgets.QMessageBox.question(
                self, "Canopy browser", f"'{self.group}' already exists on {os.path.basename(e['fp'])}.\n"
                                        "Replace it (a zip backup is written first)?")
            if ans != QtWidgets.QMessageBox.Yes:
                return
            st = D.save_found_record(self.pt, self.group, e["fp"], pts, ref,
                                     getattr(self.ctx, "shape_type", "polygon"), replace=True)
        if st in ("saved", "replaced"):
            self.dlg.results[e["row"]][2] = st
            self.dlg._update_row(e["row"])
            D.refresh_after_save(self.pt, [e["fp"]])
        self._build_page()

    def _reject(self, e):
        self.visible = [x for x in self.visible if x is not e]
        self.entries = [x for x in self.entries if x is not e]
        self.set_page(self.page, force=True)

    def _show_in_grid(self, e):
        pts, ref, _inside = self._record_space(e)
        D.show_in_grid(self.pt, e["fp"], pts, ref)
