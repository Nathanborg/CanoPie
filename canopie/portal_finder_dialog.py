"""
Find data in an ArcGIS Hub / open-data portal (default: the STRI data portal).

Two modes, one window:

* ``rasters`` (from File > Add Remote Images): layers CanoPie can stream, and
  rasters that only download whole. Streamable ones go into the links box;
  for download-only ones the user is asked before anything is downloaded.
* ``vectors`` (from the Polygon Manager): feature layers, shapefiles,
  GeoPackages and GeoJSON, imported as polygons.

Searching runs on a worker thread (the portal answers in a few seconds), so
the window never blocks. The work itself is canopie.remote_portal.
"""

import concurrent.futures
import logging
import os

from PyQt5 import QtCore, QtWidgets

from . import remote_portal as rp

_SETTINGS = ("CanoPie", "CanoPie")
PORTAL_SETTING = "data_portal_url"

_HOW = {"stream": "Stream", "download": "Download", "vector": "Import shapes"}


class PortalFinderDialog(QtWidgets.QDialog):
    """``exec_()`` then ``selected``: the chosen items (remote_portal.search dicts)."""

    def __init__(self, parent=None, mode="rasters"):
        super().__init__(parent)
        self.mode = mode
        self.selected = []
        self._org = None
        self._items = []
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="canopie-portal")
        self._future = None
        self.setWindowTitle("Find data in a portal" + (" -- layers to import" if mode == "vectors" else ""))
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(900, 600)

        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Portal"))
        self.portal_edit = QtWidgets.QLineEdit(
            QtCore.QSettings(*_SETTINGS).value(PORTAL_SETTING, rp.DEFAULT_PORTAL, type=str) or rp.DEFAULT_PORTAL)
        self.portal_edit.setToolTip("An ArcGIS Hub / open-data site, e.g. the STRI data portal")
        row.addWidget(self.portal_edit, 1)
        lay.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search, e.g. BCI 50ha, LiDAR, forest cover, plots ... (empty: everything)")
        self.search_edit.returnPressed.connect(self.run_search)
        row.addWidget(self.search_edit, 1)
        self.kind_combo = QtWidgets.QComboBox()
        if mode == "vectors":
            self.kind_combo.addItem("Vector layers", ("vector",))
        else:
            self.kind_combo.addItem("Rasters CanoPie can stream", ("stream",))
            self.kind_combo.addItem("Rasters that only download", ("download",))
            self.kind_combo.addItem("All rasters", ("stream", "download"))
        row.addWidget(self.kind_combo)
        self.search_btn = QtWidgets.QPushButton("Search")
        self.search_btn.clicked.connect(self.run_search)
        row.addWidget(self.search_btn)
        lay.addLayout(row)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Title", "Type", "How", "Owner"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.itemDoubleClicked.connect(lambda *_: self._accept())
        lay.addWidget(self.table, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        buttons = QtWidgets.QDialogButtonBox()
        self.add_btn = buttons.addButton("Add selected" if mode == "rasters" else "Import selected",
                                         QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        QtCore.QTimer.singleShot(0, self.run_search)

    # -- searching (off the GUI thread) -------------------------------------

    def run_search(self):
        if self._future is not None and not self._future.done():
            return
        portal = self.portal_edit.text().strip() or rp.DEFAULT_PORTAL
        text = self.search_edit.text().strip()
        kinds = self.kind_combo.currentData()
        org = self._org if self._org and self._org[0] == portal else None
        self.status.setText("Searching {} ...".format(portal))
        self.search_btn.setEnabled(False)

        def job():
            o = org or (portal,) + rp.org_for_site(portal)
            return o, rp.search(o[1], text=text, kinds=kinds, num=1000)

        self._future = self._pool.submit(job)
        self._timer.start()

    def _poll(self):
        if self._future is None or not self._future.done():
            return
        self._timer.stop()
        self.search_btn.setEnabled(True)
        try:
            self._org, items = self._future.result()
        except Exception as e:
            self.status.setText("Search failed: {}".format(e))
            return
        QtCore.QSettings(*_SETTINGS).setValue(PORTAL_SETTING, self._org[0])
        self.show_items(items)
        self.status.setText("{} item(s) in {}. Select one or more{}.".format(
            len(items), self._org[2],
            " (Ctrl/Shift for several)" if items else ""))

    def show_items(self, items):
        self._items = list(items)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._items))
        for r, it in enumerate(self._items):
            cells = [it["title"], it["type"], _HOW.get(it["kind"], it["kind"]), it["owner"]]
            for c, text in enumerate(cells):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setData(QtCore.Qt.UserRole, r)
                if it.get("snippet"):
                    cell.setToolTip(it["snippet"])
                self.table.setItem(r, c, cell)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

    def _accept(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        idx = [self.table.item(r, 0).data(QtCore.Qt.UserRole) for r in rows]
        self.selected = [self._items[i] for i in idx if i is not None]
        if not self.selected:
            self.status.setText("Select at least one item.")
            return
        self.accept()

    def done(self, r):
        self._timer.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().done(r)


def run_with_progress(parent, title, fn):
    """Run ``fn(progress, cancelled)`` on a worker thread behind a progress
    dialog; returns its result (raises its exception). ``progress(done, total)``
    may be called from the worker."""
    import time
    dlg = QtWidgets.QProgressDialog(title, "Cancel", 0, 0, parent)
    dlg.setWindowModality(QtCore.Qt.WindowModal)
    dlg.setMinimumDuration(200)
    state = {"done": 0, "total": None}

    def progress(done, total):
        state["done"], state["total"] = done, total

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn, progress, dlg.wasCanceled)
    try:
        while not fut.done():
            QtWidgets.QApplication.processEvents()
            d, t = state["done"], state["total"]
            if t:
                dlg.setMaximum(100)
                dlg.setValue(int(100 * d / max(1, t)))
                dlg.setLabelText("{}  {:.1f} of {:.1f} MB".format(title, d / 1e6, t / 1e6))
            elif d:
                dlg.setLabelText("{}  {:,}".format(title, d))
            time.sleep(0.05)
        return fut.result()
    finally:
        dlg.close()
        pool.shutdown(wait=False)


def fetch_download_items(parent, items, folder):
    """Ask, then download download-only raster items into ``folder``.
    Returns the raster files to open."""
    out = []
    for it in items:
        size = rp.download_size(it)
        what = "{:.1f} MB".format(size / 1e6) if size else "unknown size"
        ans = QtWidgets.QMessageBox.question(
            parent, "Download only",
            "'{}' cannot be streamed: it is only available as a download ({}).\n\n"
            "Download it into\n{}\nand open it locally?".format(it["title"], what, folder))
        if ans != QtWidgets.QMessageBox.Yes:
            continue
        try:
            files = run_with_progress(parent, "Downloading {}".format(it["title"]),
                                      lambda p, c, it=it: rp.download(it, folder, p, c))
        except Exception as e:
            QtWidgets.QMessageBox.warning(parent, "Download failed", "{}: {}".format(it["title"], e))
            continue
        rasters = [f for f in files if f.lower().endswith((".tif", ".tiff", ".jpg", ".png", ".img"))]
        if not rasters:
            QtWidgets.QMessageBox.information(
                parent, "Downloaded", "{} downloaded to {}, but it holds no raster CanoPie opens "
                "(.tif / .jpg / .png).".format(it["title"], folder))
        out += rasters
    return out


def vectors_to_shapefiles(parent, items, folder):
    """Convert chosen vector items to shapefiles under ``folder`` (with a
    progress dialog). Returns the .shp paths; failures are reported."""
    import re
    shps, failed = [], []
    for it in items:
        sub = os.path.join(folder, re.sub(r"[^\w-]+", "_", it["title"]).strip("_") or it["id"])
        try:
            shps += run_with_progress(parent, "Fetching {}".format(it["title"]),
                                      lambda p, c, it=it, sub=sub: rp.vector_to_shapefiles(it, sub, p, c))
        except Exception as e:
            logging.warning("[portal] %s: %s", it["title"], e)
            failed.append("{}: {}".format(it["title"], e))
    if failed:
        QtWidgets.QMessageBox.warning(parent, "Some layers were not imported", "\n".join(failed[:10]))
    return shps
