import os
import logging
import threading
import queue
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thumbnail loader — persistent worker thread with generation handling
# ---------------------------------------------------------------------------

_THUMB_SIZE = 120


class _ThumbLoader(QtCore.QObject):
    """
    Loads thumbnails using a thread pool (4 workers).
    Workers produce QImage (thread-safe); main thread converts to QPixmap.
    Generation system cancels stale requests.
    """
    thumb_ready = QtCore.pyqtSignal(str, QtGui.QPixmap, int)  # path, pixmap, generation

    _N_WORKERS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_cache = {}       # path -> QPixmap (persists across generations)
        self._generation = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Shared queue consumed by multiple workers
        self._queue = queue.Queue()
        self._threads = []
        for _ in range(self._N_WORKERS):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._threads.append(t)

        # Internal signal: worker finished loading a QImage (thread-safe)
        # We convert QImage→QPixmap on the main thread via this signal.
        self._qimage_ready = _QImageBridge()
        self._qimage_ready.ready.connect(self._on_qimage_ready)

    def reset(self):
        """Bump generation — all queued items become stale."""
        with self._lock:
            self._generation += 1

    def is_cached(self, path):
        return path in self._img_cache

    def request(self, paths):
        """Queue paths for loading under the current generation."""
        with self._lock:
            gen = self._generation
        for p in paths:
            if p in self._img_cache:
                self.thumb_ready.emit(p, self._img_cache[p], gen)
            else:
                self._queue.put((p, gen))

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                path, gen = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                current_gen = self._generation
            if gen != current_gen:
                self._queue.task_done()
                continue

            if path in self._img_cache:
                self._qimage_ready.ready.emit(path, None, gen)
                self._queue.task_done()
                continue

            qimg = self._load_image(path)
            self._qimage_ready.ready.emit(path, qimg, gen)
            self._queue.task_done()

    @QtCore.pyqtSlot(str, object, int)
    def _on_qimage_ready(self, path, qimg, gen):
        """Runs on main thread — converts QImage to QPixmap and emits."""
        with self._lock:
            if gen != self._generation:
                return

        # If already cached (race between workers), just emit cache
        if path in self._img_cache:
            self.thumb_ready.emit(path, self._img_cache[path], gen)
            return

        if qimg is None:
            return

        pm = QtGui.QPixmap.fromImage(qimg)
        self._img_cache[path] = pm
        self.thumb_ready.emit(path, pm, gen)

    @staticmethod
    def _load_image(path):
        import cv2
        import numpy as np
        cv2.setNumThreads(0)

        img = None
        # Strategy 1: cv2.imread (fast, handles most formats)
        try:
            if path.isascii():
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

        # Strategy 2: numpy buffer + imdecode (handles unicode paths)
        if img is None:
            try:
                buf = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            except Exception:
                pass

        # Strategy 3: PIL/Pillow fallback (handles exotic formats)
        if img is None:
            try:
                from PIL import Image
                pil_img = Image.open(path)
                pil_img.load()
                arr = np.array(pil_img)
                if arr.ndim == 2:
                    img = arr  # grayscale
                elif arr.ndim == 3:
                    if arr.shape[2] == 4:
                        img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    elif arr.shape[2] == 3:
                        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        img = arr[:, :, :3]
            except Exception:
                pass

        if img is None:
            return None

        try:
            # Multi-band stack: take first 3 bands
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]

            # Convert non-uint8 (16-bit TIFFs etc.) with percentile stretch
            if img.dtype != np.uint8:
                arr = img.astype(np.float32)
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    lo, hi = np.percentile(finite, [1, 99])
                    if hi <= lo:
                        hi = lo + 1.0
                    arr = (arr - lo) / (hi - lo) * 255.0
                else:
                    arr = np.zeros_like(img, dtype=np.float32)
                img = np.clip(arr, 0, 255).astype(np.uint8)

            # Grayscale -> BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize to thumbnail
            h, w = img.shape[:2]
            if max(h, w) > _THUMB_SIZE:
                s = _THUMB_SIZE / max(h, w)
                img = cv2.resize(img, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)

            # BGR -> RGB for QImage
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, w, h, w * 3,
                                QtGui.QImage.Format_RGB888).copy()
            return qimg
        except Exception:
            return None

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)


class _QImageBridge(QtCore.QObject):
    """Bridge signal to move QImage from worker threads to main thread."""
    ready = QtCore.pyqtSignal(str, object, int)  # path, QImage|None, generation


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

class SimilarityFilterDialog(QtWidgets.QDialog):
    """
    Dialog to filter images based on similarity to 1-3 reference images.
    Includes a live thumbnail preview of images that pass the threshold.

    Thumbnail strategy: ALL passing items are inserted as lightweight
    placeholders (text only, no pixmap) so the scrollbar is correct.
    Only the ~12 items visible in the viewport get their thumbnails
    loaded.  Scrolling triggers loading of newly-visible items.
    """

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Similar Images")
        self.resize(1300, 750)

        self.all_paths = image_paths
        self.reference_paths = []
        self.results = []           # [(path, score), ...]
        self.threshold = 0.85

        self.current_generation = 0
        self._thumb_loader = _ThumbLoader(self)
        self._thumb_loader.thumb_ready.connect(self._on_thumb_ready)

        self._thumb_item_map = {}   # path -> QListWidgetItem
        self._passing = []          # filtered results for current threshold

        # Timer for mouse-wheel scrolling (no sliderReleased for wheel)
        self._viewport_timer = QtCore.QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(150)
        self._viewport_timer.timeout.connect(self._load_visible)

        self._thresh_line = None
        self._thresh_legend = None
        self._prev_stats_color = None

        self._init_ui()

    def closeEvent(self, event):
        self._thumb_loader.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        main_h = QtWidgets.QHBoxLayout()

        # ========= LEFT PANEL (reference + histogram) =========
        left = QtWidgets.QVBoxLayout()

        ref_group = QtWidgets.QGroupBox("1. Select Reference Images (Max 3)")
        ref_lay = QtWidgets.QVBoxLayout(ref_group)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        for p in self.all_paths:
            item = QtWidgets.QListWidgetItem(os.path.basename(p))
            item.setData(QtCore.Qt.UserRole, p)
            self.list_widget.addItem(item)
        self.list_widget.itemSelectionChanged.connect(
            self._on_selection_changed)

        ref_lay.addWidget(QtWidgets.QLabel(
            "Select 1-3 images to serve as 'good' references."))
        ref_lay.addWidget(self.list_widget)
        left.addWidget(ref_group, stretch=3)

        self.btn_calculate = QtWidgets.QPushButton(
            "2. Calculate Similarity")
        self.btn_calculate.clicked.connect(self._run_calculation)
        self.btn_calculate.setEnabled(False)
        left.addWidget(self.btn_calculate)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        hist_group = QtWidgets.QGroupBox("3. Score Distribution")
        hist_lay = QtWidgets.QVBoxLayout(hist_group)

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        hist_lay.addWidget(self.canvas)

        th_lay = QtWidgets.QHBoxLayout()
        th_lay.addWidget(QtWidgets.QLabel("Threshold:"))

        self.slider_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(int(self.threshold * 100))
        self.slider_thresh.valueChanged.connect(self._on_threshold_changed)
        self.slider_thresh.sliderReleased.connect(self._on_slider_released)
        th_lay.addWidget(self.slider_thresh)

        self.lbl_thresh_val = QtWidgets.QLabel(f"{self.threshold:.2f}")
        th_lay.addWidget(self.lbl_thresh_val)

        self.lbl_stats = QtWidgets.QLabel("Selected: 0 / 0")
        th_lay.addWidget(self.lbl_stats)
        hist_lay.addLayout(th_lay)

        left.addWidget(hist_group, stretch=4)
        main_h.addLayout(left, stretch=5)

        # ========= RIGHT PANEL (thumbnail preview) =========
        right_group = QtWidgets.QGroupBox(
            "Preview — Images Above Threshold")
        right_lay = QtWidgets.QVBoxLayout(right_group)

        self.thumb_widget = QtWidgets.QListWidget()
        self.thumb_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.thumb_widget.setIconSize(
            QtCore.QSize(_THUMB_SIZE, _THUMB_SIZE))
        self.thumb_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumb_widget.setMovement(QtWidgets.QListView.Static)
        self.thumb_widget.setSpacing(6)
        self.thumb_widget.setUniformItemSizes(True)
        self.thumb_widget.setWordWrap(True)
        self.thumb_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.thumb_widget.verticalScrollBar().valueChanged.connect(
            self._on_thumb_scroll)
        # When user drags scrollbar and releases → load visible
        self.thumb_widget.verticalScrollBar().sliderReleased.connect(
            self._load_visible)
        right_lay.addWidget(self.thumb_widget)

        self.lbl_preview_count = QtWidgets.QLabel("")
        self.lbl_preview_count.setAlignment(QtCore.Qt.AlignCenter)
        right_lay.addWidget(self.lbl_preview_count)

        main_h.addWidget(right_group, stretch=4)

        # ========= BOTTOM BUTTONS =========
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QtWidgets.QDialogButtonBox.Ok).setText(
            "Apply Filter (Keep Selected)")

        outer.addLayout(main_h, stretch=1)
        outer.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Reference selection
    # ------------------------------------------------------------------
    def _on_selection_changed(self):
        selected = self.list_widget.selectedItems()
        self.reference_paths = [
            item.data(QtCore.Qt.UserRole) for item in selected]
        if len(self.reference_paths) > 3:
            self.btn_calculate.setEnabled(False)
            self.btn_calculate.setText("Max 3 References Allowed")
        else:
            self.btn_calculate.setEnabled(len(self.reference_paths) > 0)
            self.btn_calculate.setText("2. Calculate Similarity")

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------
    def _run_calculation(self):
        if not self.reference_paths:
            return

        from .phenocam_filter import PhenocamFilter, batch_extract_features

        self.progress_bar.setVisible(True)
        self.btn_calculate.setEnabled(False)
        self.list_widget.setEnabled(False)
        self.slider_thresh.setEnabled(False)
        self.results = []

        try:
            total_imgs = len(self.all_paths)
            self.progress_bar.setRange(0, total_imgs)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Extracting Features... %p%")

            def on_progress(done, total):
                self.progress_bar.setValue(done)
                QtWidgets.QApplication.processEvents()

            all_features = batch_extract_features(
                self.all_paths, resize_max=256,
                progress_callback=on_progress)

            if not all_features:
                QtWidgets.QMessageBox.warning(
                    self, "Extraction Failed",
                    "Could not extract features from any images.")
                return

            final_scores = {}
            for i, ref_p in enumerate(self.reference_paths):
                self.progress_bar.setFormat(
                    f"Scoring against Ref {i + 1}... %p%")
                QtWidgets.QApplication.processEvents()
                try:
                    pf = PhenocamFilter(
                        reference_path=ref_p, resize_max=256)
                    rankings = pf.rank_from_features(all_features)
                    for path, score in rankings:
                        if path not in final_scores:
                            final_scores[path] = score
                        else:
                            final_scores[path] = max(
                                final_scores[path], score)
                except Exception as e:
                    logger.error(
                        f"Failed to score against reference {ref_p}: {e}")

            self.results = sorted(
                final_scores.items(),
                key=lambda x: x[1], reverse=True)
            self._draw_histogram()
            self._update_stats()
            self._rebuild_placeholders()

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"An error occurred during calculation:\n{e}")
        finally:
            self.progress_bar.setVisible(False)
            self.btn_calculate.setEnabled(True)
            self.list_widget.setEnabled(True)
            self.slider_thresh.setEnabled(True)

    # ------------------------------------------------------------------
    # Threshold (drag = cheap updates only, release = rebuild)
    # ------------------------------------------------------------------
    def _on_threshold_changed(self, val):
        self.threshold = val / 100.0
        self.lbl_thresh_val.setText(f"{self.threshold:.2f}")
        self._move_threshold_line()
        self._update_stats()

    def _on_slider_released(self):
        self._rebuild_placeholders()

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------
    def _draw_histogram(self):
        self.figure.clear()
        self._thresh_line = None
        self._thresh_legend = None

        if not self.results:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        scores = [s for _, s in self.results]
        ax.hist(scores, bins=30, edgecolor='black', alpha=0.7,
                color='steelblue')
        self._thresh_line = ax.axvline(
            x=self.threshold, color='red', linestyle='--',
            linewidth=2, label=f'Threshold={self.threshold:.2f}')
        ax.set_title("Similarity Score Distribution")
        ax.set_xlabel("Score (1.0 = Identical)")
        ax.set_ylabel("Count")
        self._thresh_legend = ax.legend()
        self.canvas.draw()

    def _move_threshold_line(self):
        if self._thresh_line is None:
            return
        self._thresh_line.set_xdata([self.threshold, self.threshold])
        if self._thresh_legend is not None:
            try:
                texts = self._thresh_legend.get_texts()
                if texts:
                    texts[0].set_text(
                        f'Threshold={self.threshold:.2f}')
            except Exception:
                pass
        self.canvas.draw_idle()

    def _update_stats(self):
        if not self.results:
            return
        n_total = len(self.results)
        n_sel = sum(1 for _, s in self.results if s >= self.threshold)
        self.lbl_stats.setText(
            f"Selected: {n_sel} / {n_total} images")
        color = "red" if n_sel == 0 else "green"
        if color != self._prev_stats_color:
            self.lbl_stats.setStyleSheet(
                f"color: {color}; font-weight: bold")
            self._prev_stats_color = color

    # ------------------------------------------------------------------
    # Thumbnail preview — VIEWPORT-ONLY loading
    # ------------------------------------------------------------------
    def _rebuild_placeholders(self):
        """
        Insert ALL passing items as lightweight placeholders instantly.
        Scrollbar reflects true count.  Only visible items get loaded.
        """
        self._thumb_loader.reset()
        self.current_generation = self._thumb_loader._generation
        self._thumb_item_map = {}

        self.thumb_widget.setUpdatesEnabled(False)
        self.thumb_widget.clear()

        if not self.results:
            self._passing = []
            self.thumb_widget.setUpdatesEnabled(True)
            self.lbl_preview_count.setText("")
            return

        self._passing = [
            (p, s) for p, s in self.results if s >= self.threshold]

        # Insert all placeholders (text only — no pixmap loading)
        placeholder = QtGui.QIcon()
        item_size = QtCore.QSize(
            _THUMB_SIZE + 16, _THUMB_SIZE + 36)
        for path, score in self._passing:
            label = f"{os.path.basename(path)}\n{score:.2f}"
            item = QtWidgets.QListWidgetItem(placeholder, label)
            item.setData(QtCore.Qt.UserRole, path)
            item.setSizeHint(item_size)
            self.thumb_widget.addItem(item)
            self._thumb_item_map[path] = item

        self.thumb_widget.setUpdatesEnabled(True)

        self.lbl_preview_count.setText(
            f"Showing {len(self._passing)} / {len(self.results)} images")

        # Load only the visible ones (after layout settles)
        QtCore.QTimer.singleShot(0, self._load_visible)

    def _on_thumb_scroll(self, _value):
        """Scroll event — only react to wheel scrolling, not drag."""
        sb = self.thumb_widget.verticalScrollBar()
        if sb.isSliderDown():
            # User is dragging — do nothing until they release
            return
        # Mouse wheel or programmatic scroll — debounce then load
        self._viewport_timer.start()

    def _load_visible(self):
        """
        Find items visible in the viewport using scroll position math
        and request their thumbnails.  The loader's internal cache
        ensures already-loaded images are returned instantly.
        """
        n_items = self.thumb_widget.count()
        if n_items == 0 or not self._passing:
            return

        vp = self.thumb_widget.viewport()
        vp_w = vp.width()
        vp_h = vp.height()
        if vp_w <= 0 or vp_h <= 0:
            return

        # Compute grid layout from known item sizes
        item_w = _THUMB_SIZE + 16 + 6   # sizeHint width + spacing
        item_h = _THUMB_SIZE + 36 + 6   # sizeHint height + spacing
        cols = max(1, vp_w // item_w)

        sb = self.thumb_widget.verticalScrollBar()
        scroll_top = sb.value()
        scroll_bot = scroll_top + vp_h

        # Which rows are visible?
        first_row = max(0, scroll_top // item_h - 1)
        last_row = scroll_bot // item_h + 2   # +2 buffer rows

        first_idx = first_row * cols
        last_idx = min((last_row + 1) * cols, n_items)

        # Also add a buffer of extra items below
        last_idx = min(last_idx + cols * 2, n_items)

        paths_to_load = []
        for i in range(first_idx, last_idx):
            item = self.thumb_widget.item(i)
            if item is None:
                continue
            path = item.data(QtCore.Qt.UserRole)
            if path:
                paths_to_load.append(path)

        if paths_to_load:
            self._thumb_loader.request(paths_to_load)

    @QtCore.pyqtSlot(str, QtGui.QPixmap, int)
    def _on_thumb_ready(self, path, pixmap, gen):
        """Thumbnail arrived — update its item if generation matches."""
        if gen != self.current_generation:
            return
        item = self._thumb_item_map.get(path)
        if item is None:
            return
        item.setIcon(QtGui.QIcon(pixmap))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_selected_paths(self):
        """Returns list of paths that met the threshold."""
        return [p for p, s in self.results if s >= self.threshold]
