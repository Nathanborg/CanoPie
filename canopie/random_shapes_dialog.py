from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSettings
import logging

class RandomShapesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, is_georeferenced=False, gsd=None,
                 root_images=None, current_image=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Random Shapes")
        self.is_georeferenced = is_georeferenced
        self.gsd = gsd
        # Images of the current root, so "Current Image Only" can be pointed at a
        # specific one rather than defaulting to whichever viewer came first.
        self.root_images = list(root_images or [])
        self.current_image = current_image

        self.resize(350, 300)
        self._init_ui()
        self._load_settings()
        self._on_shape_changed()
        self._on_scope_changed()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info Label
        unit_text = "Meters" if self.is_georeferenced else "Pixels"
        info_label = QtWidgets.QLabel(f"<b>Units:</b> Inputs below are in <b>{unit_text}</b>.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        form_layout = QtWidgets.QFormLayout()
        
        # Shape Type
        self.shape_type_cb = QtWidgets.QComboBox()
        self.shape_type_cb.addItems(["Point", "Circle", "Rectangle"])
        self.shape_type_cb.currentTextChanged.connect(self._on_shape_changed)
        form_layout.addRow("Shape Type:", self.shape_type_cb)
        
        # Count
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 100000)
        self.count_spin.setValue(10)
        form_layout.addRow("Count per Image:", self.count_spin)
        
        # Dimensions: Diameter
        self.diameter_spin = QtWidgets.QDoubleSpinBox()
        self.diameter_spin.setRange(0.001, 100000)
        self.diameter_spin.setDecimals(3)
        self.diameter_spin.setValue(1.0)
        self.diameter_row = form_layout.addRow("Diameter:", self.diameter_spin)
        
        # Dimensions: Width/Height
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0.001, 100000)
        self.width_spin.setDecimals(3)
        self.width_spin.setValue(1.0)
        self.width_row = form_layout.addRow("Width:", self.width_spin)
        
        self.height_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin.setRange(0.001, 100000)
        self.height_spin.setDecimals(3)
        self.height_spin.setValue(1.0)
        self.height_row = form_layout.addRow("Height:", self.height_spin)
        
        # Spatial Rules
        self.min_dist_border_spin = QtWidgets.QDoubleSpinBox()
        self.min_dist_border_spin.setRange(0.0, 100000)
        self.min_dist_border_spin.setDecimals(3)
        self.min_dist_border_spin.setValue(0.0)
        form_layout.addRow("Min Dist from Valid Edge:", self.min_dist_border_spin)
        
        self.min_dist_shapes_spin = QtWidgets.QDoubleSpinBox()
        self.min_dist_shapes_spin.setRange(0.0, 100000)
        self.min_dist_shapes_spin.setDecimals(3)
        self.min_dist_shapes_spin.setValue(0.0)
        form_layout.addRow("Min Dist Between Shapes:", self.min_dist_shapes_spin)
        
        # Batch Scope
        self.scope_cb = QtWidgets.QComboBox()
        self.scope_cb.addItems(["Current Image Only", "All Images in Current Root", "All Images in Project"])
        self.scope_cb.currentTextChanged.connect(self._on_scope_changed)
        form_layout.addRow("Target Scope:", self.scope_cb)

        # Which image of the root receives the shapes (Current Image Only).
        # Without this the target was implicit -- whichever viewer the code
        # happened to find first -- which is not necessarily the one on screen.
        self.target_image_cb = QtWidgets.QComboBox()
        self.target_image_cb.setToolTip(
            "Which image of the current root receives the shapes.\n"
            "Only used when Target Scope is 'Current Image Only'.")
        import os as _os
        for fp in self.root_images:
            self.target_image_cb.addItem(_os.path.basename(fp), fp)
        if self.current_image:
            idx = self.target_image_cb.findData(self.current_image)
            if idx >= 0:
                self.target_image_cb.setCurrentIndex(idx)
        self.target_image_row_label = QtWidgets.QLabel("Image:")
        form_layout.addRow(self.target_image_row_label, self.target_image_cb)

        # Restrict to valid (non-NoData) area.
        # OFF (default) = fast: shapes are placed from image width/height only,
        # no pixel decode, and files are processed in parallel.
        # ON = accurate for irregular GeoTIFF footprints, but must read every
        # pixel to build the NoData/NaN mask (much slower).
        self.restrict_valid_cb = QtWidgets.QCheckBox("Keep shapes inside valid area (reads pixels, slower)")
        self.restrict_valid_cb.setChecked(False)
        form_layout.addRow("", self.restrict_valid_cb)

        # Sampling design within the valid area.
        # OFF = uniform over valid pixels: every valid pixel is equally likely, so
        #       shapes land in proportion to each patch's AREA. Statistically the
        #       unbiased default, but on a footprint split into unequal patches it
        #       looks "concentrated" -- a patch holding 75% of the valid pixels
        #       legitimately receives ~75% of the shapes.
        # ON  = round-robin across the separate valid patches, so a small patch
        #       gets comparable representation to a large one.
        self.stratify_cb = QtWidgets.QCheckBox("Spread evenly across separate valid areas")
        self.stratify_cb.setChecked(False)
        self.stratify_cb.setToolTip(
            "Off: shapes are placed uniformly over the valid area, so a large patch\n"
            "receives proportionally more shapes than a small one (unbiased sampling).\n\n"
            "On: shapes are distributed round-robin across the disconnected valid\n"
            "patches, giving small patches comparable representation.\n\n"
            "Only applies when 'Keep shapes inside valid area' is enabled.")
        form_layout.addRow("", self.stratify_cb)

        # Stratification is meaningless without a valid-area mask to split.
        self.restrict_valid_cb.toggled.connect(self.stratify_cb.setEnabled)
        self.stratify_cb.setEnabled(self.restrict_valid_cb.isChecked())

        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _on_shape_changed(self):
        shape = self.shape_type_cb.currentText()
        is_circle = shape == "Circle"
        is_rect = shape == "Rectangle"
        
        # In QFormLayout, we can access the label and field to show/hide
        form_layout = self.layout().itemAt(1)
        if isinstance(form_layout, QtWidgets.QFormLayout):
            for i in range(form_layout.rowCount()):
                label_item = form_layout.itemAt(i, QtWidgets.QFormLayout.LabelRole)
                field_item = form_layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
                if not label_item or not field_item:
                    continue
                lbl_w = label_item.widget()
                fld_w = field_item.widget()
                
                if fld_w == self.diameter_spin:
                    if lbl_w: lbl_w.setVisible(is_circle)
                    fld_w.setVisible(is_circle)
                elif fld_w in (self.width_spin, self.height_spin):
                    if lbl_w: lbl_w.setVisible(is_rect)
                    fld_w.setVisible(is_rect)

    def _on_scope_changed(self, *_):
        """The image picker only applies to the single-image scope."""
        single = self.scope_cb.currentText() == "Current Image Only"
        has_choices = self.target_image_cb.count() > 0
        show = single and has_choices
        self.target_image_cb.setVisible(show)
        self.target_image_row_label.setVisible(show)

    def _load_settings(self):
        try:
            settings = QSettings("CanoPie", "RandomShapesModule")
            idx = self.shape_type_cb.findText(settings.value("shape_type", "Point"))
            if idx >= 0: self.shape_type_cb.setCurrentIndex(idx)
            
            self.count_spin.setValue(int(settings.value("count", 10)))
            self.diameter_spin.setValue(float(settings.value("diameter", 1.0)))
            self.width_spin.setValue(float(settings.value("width", 1.0)))
            self.height_spin.setValue(float(settings.value("height", 1.0)))
            self.min_dist_border_spin.setValue(float(settings.value("min_dist_border", 0.0)))
            self.min_dist_shapes_spin.setValue(float(settings.value("min_dist_shapes", 0.0)))
            
            scope_idx = self.scope_cb.findText(settings.value("scope", "Current Image Only"))
            if scope_idx >= 0: self.scope_cb.setCurrentIndex(scope_idx)

            self.restrict_valid_cb.setChecked(
                settings.value("restrict_valid", False, type=bool))
            self.stratify_cb.setChecked(
                settings.value("stratify", False, type=bool))
            self.stratify_cb.setEnabled(self.restrict_valid_cb.isChecked())
        except Exception as e:
            logging.warning(f"Failed to load RandomShapes settings: {e}")
        
    def _save_settings(self):
        try:
            settings = QSettings("CanoPie", "RandomShapesModule")
            settings.setValue("shape_type", self.shape_type_cb.currentText())
            settings.setValue("count", self.count_spin.value())
            settings.setValue("diameter", self.diameter_spin.value())
            settings.setValue("width", self.width_spin.value())
            settings.setValue("height", self.height_spin.value())
            settings.setValue("min_dist_border", self.min_dist_border_spin.value())
            settings.setValue("min_dist_shapes", self.min_dist_shapes_spin.value())
            settings.setValue("scope", self.scope_cb.currentText())
            settings.setValue("restrict_valid", self.restrict_valid_cb.isChecked())
            settings.setValue("stratify", self.stratify_cb.isChecked())
        except Exception as e:
            logging.warning(f"Failed to save RandomShapes settings: {e}")
        
    def accept(self):
        self._save_settings()
        super().accept()
        
    def get_parameters(self):
        return {
            "shape_type": self.shape_type_cb.currentText(),
            "count": self.count_spin.value(),
            "diameter": self.diameter_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "min_dist_border": self.min_dist_border_spin.value(),
            "min_dist_shapes": self.min_dist_shapes_spin.value(),
            "scope": self.scope_cb.currentText(),
            # Only meaningful for the single-image scope; None lets the caller
            # keep its own (focused-viewer) choice.
            "target_image": (self.target_image_cb.currentData()
                             if (self.scope_cb.currentText() == "Current Image Only"
                                 and self.target_image_cb.count() > 0) else None),
            "restrict_valid": self.restrict_valid_cb.isChecked(),
            # Only meaningful alongside restrict_valid; the generator ignores it
            # otherwise (there is no valid-area mask to split into patches).
            "stratify": self.stratify_cb.isChecked() and self.restrict_valid_cb.isChecked()
        }
