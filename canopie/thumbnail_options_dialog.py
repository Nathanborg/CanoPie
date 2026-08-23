"""Options for MachineLearningManager.generate_thumbnails.

Before this dialog existed, generate_thumbnails asked only for an output folder
and then applied three hardcoded constants -- zoom_factor 1.4, a 200x200 output
and a 2 px outline -- writing one annotated JPEG per polygon. That is a fine
default for eyeballing crowns and a poor one for producing training data, where
you want a specific resolution, usually no annotation burnt into the pixels, and
often a grid of fixed-size patches rather than one variable-shaped crop.

The defaults here reproduce the old behaviour exactly, so pressing Generate
without touching anything writes what it always did.
"""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QCheckBox, QSpinBox, QDoubleSpinBox,
                             QDialogButtonBox)

from .image_editor_dialog import CollapsibleBox


#: The values generate_thumbnails hardcoded before this dialog existed. Every
#: default below is one of these, which is what makes "press Generate" a no-op
#: change; canopie/qc/test_thumbnail_options.py asserts they stay in step.
LEGACY_THUMBNAIL_SIZE = (200, 200)
LEGACY_ZOOM_FACTOR = 1.4
LEGACY_OUTLINE_THICKNESS = 2


class ThumbnailOptionsDialog(QDialog):
    """Output size, outline, and DL tiling options for thumbnail generation."""

    def __init__(self, polygons=None, parent=None):
        """`polygons` is [(points, W, H), ...] for the live tile estimate.

        Optional: without it the dialog still works, it just cannot say how
        many tiles a run will produce.
        """
        super().__init__(parent)
        self._polygons = list(polygons or [])
        self.setWindowTitle("Thumbnail Options")
        self.setMinimumWidth(380)
        self.setup_ui()
        self.apply_style()
        self.update_estimate()

    # ------------------------------------------------------------------
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # -- Output Size ------------------------------------------------
        size_box = CollapsibleBox("Output Size")
        row_wh = QHBoxLayout()
        row_wh.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 4096)
        self.width_spin.setValue(LEGACY_THUMBNAIL_SIZE[0])
        row_wh.addWidget(self.width_spin)
        row_wh.addSpacing(8)
        row_wh.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 4096)
        self.height_spin.setValue(LEGACY_THUMBNAIL_SIZE[1])
        row_wh.addWidget(self.height_spin)
        row_wh.addStretch()
        size_box.content_layout.addLayout(row_wh)

        row_zoom = QHBoxLayout()
        row_zoom.addWidget(QLabel("Crop padding:"))
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.0, 5.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setDecimals(2)
        self.zoom_spin.setValue(LEGACY_ZOOM_FACTOR)
        self.zoom_spin.setToolTip(
            "How far the crop extends beyond the polygon's bounding box.\n"
            "1.0 = the bounding box itself. With tiling on this also decides\n"
            "how many tiles each polygon yields.")
        row_zoom.addWidget(self.zoom_spin)
        row_zoom.addStretch()
        size_box.content_layout.addLayout(row_zoom)
        layout.addWidget(size_box)

        # -- Annotation -------------------------------------------------
        annot_box = CollapsibleBox("Annotation")
        self.outline_cb = QCheckBox("Draw polygon outline")
        self.outline_cb.setChecked(True)
        self.outline_cb.setToolTip(
            "Off writes clean imagery with nothing burnt into the pixels,\n"
            "which is usually what a training set wants.")
        annot_box.content_layout.addWidget(self.outline_cb)
        layout.addWidget(annot_box)

        # -- Tiling -----------------------------------------------------
        tile_box = CollapsibleBox("Tiling")
        self.tile_cb = QCheckBox("Split each crop into tiles")
        self.tile_cb.setChecked(False)
        self.tile_cb.setToolTip(
            "Cut each polygon's crop into a grid of non-overlapping squares,\n"
            "one file per tile, instead of a single resized thumbnail.")
        self.tile_cb.toggled.connect(self.on_tiling_toggled)
        tile_box.content_layout.addWidget(self.tile_cb)

        row_tile = QHBoxLayout()
        row_tile.addWidget(QLabel("Tile size:"))
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(32, 2048)
        self.tile_spin.setValue(256)
        self.tile_spin.setEnabled(False)
        self.tile_spin.setToolTip(
            "Tiles are this many pixels square. The crop is grown to a whole\n"
            "number of tiles so no polygon is dropped for being too small.")
        row_tile.addWidget(self.tile_spin)
        row_tile.addStretch()
        tile_box.content_layout.addLayout(row_tile)

        self.inside_cb = QCheckBox("Only squares that fit inside the polygon")
        self.inside_cb.setChecked(False)
        self.inside_cb.setEnabled(False)
        self.inside_cb.setToolTip(
            "Keep a tile only when the whole square lies inside the outline.\n"
            "Tiles that straddle the edge, and any polygon narrower than one\n"
            "tile, produce nothing -- watch the estimate below.")
        self.inside_cb.toggled.connect(self._on_inside_toggled)
        tile_box.content_layout.addWidget(self.inside_cb)

        self.allow_na_cb = QCheckBox("Allow NA (keep partial tiles, mask what's outside)")
        self.allow_na_cb.setChecked(False)
        self.allow_na_cb.setEnabled(False)
        self.allow_na_cb.setToolTip(
            "Keep every tile that touches the polygon at all, instead of only\n"
            "ones fully inside it. The tile image's pixels are left exactly as\n"
            "read; a same-named '<tile>_mask.png' is written alongside it\n"
            "(255 = inside the polygon, 0 = NA) so a training loader can mask\n"
            "the loss or ignore those pixels. Tiles with NO overlap at all are\n"
            "still dropped -- an all-NA tile carries no signal.\n"
            "This is the other answer to what 'Only squares that fit inside'\n"
            "answers, so only one of the two can be on at a time.")
        self.allow_na_cb.toggled.connect(self._on_allow_na_toggled)
        tile_box.content_layout.addWidget(self.allow_na_cb)

        self.estimate_lbl = QLabel("")
        self.estimate_lbl.setWordWrap(True)
        self.estimate_lbl.setStyleSheet("color: #006400; font-size: 12px;")
        tile_box.content_layout.addWidget(self.estimate_lbl)

        # The estimate moves with every input that changes the grid. Debounced,
        # because with "inside only" on it rasterises the sampled polygons --
        # ~150 ms on the real 3504-crown project, which held down a spin-box
        # arrow would queue up faster than it can be served.
        self.tile_spin.valueChanged.connect(self._schedule_estimate)
        self.zoom_spin.valueChanged.connect(self._schedule_estimate)

        layout.addWidget(tile_box)

        # Sections open by default -- CollapsibleBox starts collapsed, which
        # would hide every option behind three clicks.
        for box in (size_box, annot_box, tile_box):
            box.toggle_button.setChecked(True)

        # -- Buttons ----------------------------------------------------
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        ok_btn = self.buttonBox.button(QDialogButtonBox.Ok)
        ok_btn.setText("Generate")
        ok_btn.setDefault(True)
        layout.addWidget(self.buttonBox)

    # ------------------------------------------------------------------
    def apply_style(self):
        # CanoPie styling, same palette as the shapefile import dialog.
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
            QSpinBox, QDoubleSpinBox {
                padding: 0px 4px;
                min-height: 20px;
                font-size: 12px;
            }
            QPushButton {
                padding: 2px 6px;
                min-height: 20px;
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                border-radius: 3px;
                border: 1px solid #1b5e20;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QToolButton { padding: 0px; }

            QCheckBox { font-size: 13px; }
            QCheckBox::indicator:checked {
                background-color: #FFD700;
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 1px solid gray;
            }
            QToolButton.collapsible-toggle {
                color: #006400; /* Dark Green text/arrow */
                font-weight: bold;
                font-size: 11pt;
            }
        """)

    # ------------------------------------------------------------------
    def on_tiling_toggled(self, checked):
        self.tile_spin.setEnabled(checked)
        self.inside_cb.setEnabled(checked)
        self.allow_na_cb.setEnabled(checked)
        # Tiles are tile_size square by construction, so an output width/height
        # would simply not be honoured -- disable rather than silently ignore.
        self.width_spin.setEnabled(not checked)
        self.height_spin.setEnabled(not checked)
        self._schedule_estimate()

    def _on_inside_toggled(self, checked):
        """'Only squares inside' and 'Allow NA' are two different answers to
        the same question (what to do with a tile that straddles the edge) --
        they don't compose, so picking one turns the other off."""
        if checked and self.allow_na_cb.isChecked():
            self.allow_na_cb.setChecked(False)
        self._schedule_estimate()

    def _on_allow_na_toggled(self, checked):
        if checked and self.inside_cb.isChecked():
            self.inside_cb.setChecked(False)
        self._schedule_estimate()

    def _schedule_estimate(self, *_):
        """Coalesce rapid control changes into one recompute."""
        timer = getattr(self, "_est_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.update_estimate)
            self._est_timer = timer
        timer.start(120)

    def update_estimate(self, *_):
        """Refresh the live tile-count readout.

        Deliberately shows how many polygons yield NOTHING as well as the tile
        total. With "only squares inside" on, anything narrower than one tile
        is silently dropped, and on the real BCI crowns that reaches two thirds
        of them at tile=512 -- a number the user should see while choosing the
        size, not discover afterwards in a short output folder.
        """
        if not hasattr(self, "estimate_lbl"):
            return
        if not self._polygons:
            self.estimate_lbl.setText("")
            return
        if not self.tile_cb.isChecked():
            n = len(self._polygons)
            self.estimate_lbl.setText(f"{n:,} polygon(s) -> {n:,} thumbnail(s).")
            return

        # Lazy import: machine_learning_manager imports this module for its
        # LEGACY_* constants, so a module-level import here would be circular.
        from .machine_learning_manager import estimate_tile_yield
        try:
            est = estimate_tile_yield(self._polygons,
                                      int(self.tile_spin.value()),
                                      float(self.zoom_spin.value()),
                                      bool(self.inside_cb.isChecked()),
                                      bool(self.allow_na_cb.isChecked()))
        except Exception as e:
            self.estimate_lbl.setText(f"(estimate unavailable: {e})")
            return

        approx = "~" if est['sampled'] < est['total'] else ""
        parts = [f"{approx}{est['tiles']:,} tile(s) from {est['total']:,} polygon(s)"]
        if self.inside_cb.isChecked():
            parts.append(f"{approx}{est['grid']:,} without the inside-only filter")
        elif self.allow_na_cb.isChecked():
            parts.append(f"{approx}{est['grid']:,} without the NA filter, "
                         f"each kept tile gets a companion mask")
        if est['empty']:
            pct = 100.0 * est['empty'] / max(1, est['total'])
            parts.append(f"{approx}{est['empty']:,} polygon(s) ({pct:.0f}%) yield none")
        if est['sampled'] < est['total']:
            parts.append(f"estimated from {est['sampled']:,} sampled")
        self.estimate_lbl.setText("  |  ".join(parts) + ".")

    def get_options(self):
        """Plain dict of the chosen options (see module docstring for defaults)."""
        return {
            'thumbnail_size': (int(self.width_spin.value()),
                               int(self.height_spin.value())),
            'zoom_factor': float(self.zoom_spin.value()),
            'draw_outline': bool(self.outline_cb.isChecked()),
            'tile_enabled': bool(self.tile_cb.isChecked()),
            'tile_size': int(self.tile_spin.value()),
            'tiles_inside_only': bool(self.inside_cb.isChecked()),
            'allow_na': bool(self.allow_na_cb.isChecked()),
        }
