from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSettings
import logging

class RandomShapesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, is_georeferenced=False, gsd=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Random Shapes")
        self.is_georeferenced = is_georeferenced
        self.gsd = gsd
        
        self.resize(350, 300)
        self._init_ui()
        self._load_settings()
        self._on_shape_changed()

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
        form_layout.addRow("Target Scope:", self.scope_cb)
        
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
            "scope": self.scope_cb.currentText()
        }
