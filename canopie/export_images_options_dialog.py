"""Options for ProjectTab.export_project_images.

Before this dialog existed, export_project_images asked only two yes/no
QMessageBox questions (copy EXIF? run in background?) and then always wrote
classic TIFF, with the band order decided implicitly by whichever save
branch a given source/settings combination happened to hit -- which is what
let a real R/B swap bug hide in the less-common (but far from rare, since
any .ax with a band expression / appended bands / classification enabled
triggers it) branch for years. See test_export_images_band_order.py for the
regression this closes.

"Keep original (recommended)" is the default and is the fix on its own --
it just uses whatever channel order ProjectImagesExportWorker/export_project_
images' loaders actually tracked for each source. "Force RGB"/"Force BGR"
are an escape hatch for a downstream tool that insists on one interpretation
regardless of source.
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QCheckBox, QComboBox, QDialogButtonBox)

from .image_editor_dialog import CollapsibleBox


class ExportImagesOptionsDialog(QDialog):
    """Format, band order, EXIF, and background-run options for the
    "Export Project Images" action -- replaces the old pair of QMessageBox
    prompts with one dialog in CanoPie's standard style."""

    def __init__(self, n_images=0, parent=None):
        super().__init__(parent)
        self._n_images = int(n_images or 0)
        self.setWindowTitle("Export Images Options")
        self.setMinimumWidth(380)
        self.setup_ui()
        self.apply_style()

    # ------------------------------------------------------------------
    def setup_ui(self):
        layout = QVBoxLayout(self)

        if self._n_images:
            hdr = QLabel(f"Export {self._n_images:,} image(s) with .ax transformations.")
            hdr.setWordWrap(True)
            layout.addWidget(hdr)

        # -- Format -------------------------------------------------------
        format_box = CollapsibleBox("Format")
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("File format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("TIFF (.tif)", "tif")
        self.format_combo.addItem("JPEG (.jpg)", "jpg")
        self.format_combo.addItem("PNG (.png)", "png")
        self.format_combo.setCurrentIndex(0)  # TIFF -- matches current behavior
        self.format_combo.setToolTip(
            "A .jpg/.jpeg SOURCE always exports as .jpg regardless of this\n"
            "choice (unchanged from before) -- this picks the format for\n"
            "every other source.")
        row_fmt.addWidget(self.format_combo)
        row_fmt.addStretch()
        format_box.content_layout.addLayout(row_fmt)
        layout.addWidget(format_box)

        # -- Band Order -----------------------------------------------------
        band_box = CollapsibleBox("Band Order")
        row_band = QHBoxLayout()
        row_band.addWidget(QLabel("Band order:"))
        self.band_order_combo = QComboBox()
        self.band_order_combo.addItem("Keep original (recommended)", "keep")
        self.band_order_combo.addItem("Force RGB", "rgb")
        self.band_order_combo.addItem("Force BGR", "bgr")
        self.band_order_combo.setCurrentIndex(0)
        self.band_order_combo.setToolTip(
            "Keep original: use each source's own true channel order (the\n"
            "fix for exported images coming out with red/blue swapped).\n"
            "Force RGB/BGR: reinterpret every source as that order regardless\n"
            "of what it actually is -- only for a downstream tool that\n"
            "insists on one convention.")
        row_band.addWidget(self.band_order_combo)
        row_band.addStretch()
        band_box.content_layout.addLayout(row_band)
        layout.addWidget(band_box)

        # -- Metadata -------------------------------------------------------
        meta_box = CollapsibleBox("Metadata")
        self.copy_exif_cb = QCheckBox("Copy EXIF metadata to exported images")
        self.copy_exif_cb.setChecked(False)  # matches the old QMessageBox default (No)
        self.copy_exif_cb.setToolTip(
            "Requires ExifTool on PATH. Skipped for any file written as\n"
            "BigTIFF (ExifTool cannot write that container).")
        meta_box.content_layout.addWidget(self.copy_exif_cb)
        layout.addWidget(meta_box)

        # -- Run --------------------------------------------------------
        run_box = CollapsibleBox("Run")
        self.background_cb = QCheckBox("Run in background (continue using the app)")
        self.background_cb.setChecked(True)  # matches the old QMessageBox default (Yes)
        run_box.content_layout.addWidget(self.background_cb)
        layout.addWidget(run_box)

        # Sections open by default -- CollapsibleBox starts collapsed, which
        # would hide every option behind clicks.
        for box in (format_box, band_box, meta_box, run_box):
            box.toggle_button.setChecked(True)

        # -- Buttons ----------------------------------------------------
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        ok_btn = self.buttonBox.button(QDialogButtonBox.Ok)
        ok_btn.setText("Export")
        ok_btn.setDefault(True)
        layout.addWidget(self.buttonBox)

    # ------------------------------------------------------------------
    def apply_style(self):
        # Same palette as the thumbnail/shapefile-import dialogs.
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
            QComboBox {
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
    def get_options(self):
        """Plain dict of the chosen options."""
        return {
            'format': self.format_combo.currentData(),
            'band_order': self.band_order_combo.currentData(),
            'copy_exif': bool(self.copy_exif_cb.isChecked()),
            'background': bool(self.background_cb.isChecked()),
        }
