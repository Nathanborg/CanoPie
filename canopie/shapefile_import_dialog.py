import os
import shapefile
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QListWidget, QListWidgetItem,
                             QCheckBox, QDoubleSpinBox, QProgressBar, QGroupBox,
                             QFormLayout, QDialogButtonBox, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

def inspect_shapefile_schema(shp_paths):
    """
    Reads the first shapefile from the list to extract field names and types
    from its DBF file. Returns a list of field names.
    """
    if not shp_paths:
        return []
    
    first_shp = shp_paths[0]
    fields = []
    
    try:
        # We only need the DBF to get the fields
        with shapefile.Reader(first_shp) as sf:
            # fields[0] is the DeletionFlag, skip it
            if len(sf.fields) > 1:
                # Field format: (name, type, size, decimal)
                fields = [f[0] for f in sf.fields[1:]]
    except Exception as e:
        print(f"Error inspecting shapefile schema: {e}")
    
    return fields


class ShapefileImportDialog(QDialog):
    """
    Dialog to configure shapefile import options, including field mapping,
    attribute filtering, and geometry simplification.
    """
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.fields = fields
        self.setWindowTitle("Import Shapefile Options")
        self.setMinimumWidth(400)
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # -- Attribute Selection Group --
        attr_group = QGroupBox("Attributes to Import")
        attr_layout = QVBoxLayout()
        
        # Select All / None buttons
        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_none = QPushButton("Select None")
        self.btn_select_all.clicked.connect(self.select_all_attributes)
        self.btn_select_none.clicked.connect(self.select_no_attributes)
        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_select_none)
        btn_layout.addStretch()
        attr_layout.addLayout(btn_layout)

        # Attributes list
        self.attr_list = QListWidget()
        for field in self.fields:
            item = QListWidgetItem(field)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked) # Keep all by default
            self.attr_list.addItem(item)
        attr_layout.addWidget(self.attr_list)
        attr_group.setLayout(attr_layout)
        layout.addWidget(attr_group)

        # -- Mapping Group --
        map_group = QGroupBox("Field Mapping")
        map_layout = QFormLayout()

        self.name_combo = QComboBox()
        self.name_combo.addItem("Auto (Default)")
        self.name_combo.addItems(self.fields)
        map_layout.addRow("Name / ID Field:", self.name_combo)

        self.group_combo = QComboBox()
        self.group_combo.addItem("None")
        self.group_combo.addItems(self.fields)
        map_layout.addRow("Group Field:", self.group_combo)

        map_group.setLayout(map_layout)
        layout.addWidget(map_group)

        # -- Geometry Group --
        geom_group = QGroupBox("Geometry Processing")
        geom_layout = QVBoxLayout()

        self.simplify_cb = QCheckBox("Simplify Polygons (Reduce Vertices)")
        self.simplify_cb.toggled.connect(self.on_simplify_toggled)
        geom_layout.addWidget(self.simplify_cb)

        simplify_opts_layout = QHBoxLayout()
        simplify_opts_layout.addWidget(QLabel("Tolerance:"))
        self.simplify_tol = QDoubleSpinBox()
        self.simplify_tol.setRange(0.0001, 1000.0)
        self.simplify_tol.setDecimals(4)
        self.simplify_tol.setValue(1.0)
        self.simplify_tol.setEnabled(False)
        self.simplify_tol.setToolTip("Higher values remove more vertices. Uses Douglas-Peucker algorithm.")
        simplify_opts_layout.addWidget(self.simplify_tol)
        simplify_opts_layout.addStretch()
        geom_layout.addLayout(simplify_opts_layout)

        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)

        # -- Buttons --
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        # Style Ok button
        ok_btn = self.buttonBox.button(QDialogButtonBox.Ok)
        ok_btn.setText("Import")
        ok_btn.setDefault(True)

        layout.addWidget(self.buttonBox)

    def apply_style(self):
        # CanoPie styling copied from image editor
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
            QComboBox { padding: 0px 4px; min-height: 20px; font-size: 12px; }
            QLineEdit { padding: 1px 3px; }
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
            
            QListWidget::item:selected {
                background-color: #006400;
            }
        """)

    def on_simplify_toggled(self, checked):
        self.simplify_tol.setEnabled(checked)

    def select_all_attributes(self):
        for i in range(self.attr_list.count()):
            self.attr_list.item(i).setCheckState(Qt.Checked)

    def select_no_attributes(self):
        for i in range(self.attr_list.count()):
            self.attr_list.item(i).setCheckState(Qt.Unchecked)

    def get_options(self):
        """
        Returns a dictionary of the selected options.
        """
        selected_attrs = []
        for i in range(self.attr_list.count()):
            item = self.attr_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_attrs.append(item.text())

        name_field = self.name_combo.currentText()
        if name_field == "Auto (Default)":
            name_field = None

        group_field = self.group_combo.currentText()
        if group_field == "None":
            group_field = None

        simplify = self.simplify_cb.isChecked()
        tolerance = self.simplify_tol.value() if simplify else None

        return {
            'selected_properties': selected_attrs,
            'name_field': name_field,
            'group_field': group_field,
            'simplify_tolerance': tolerance
        }


class ShapefileImportProgressDialog(QDialog):
    """
    Enhanced progress dialog for shapefile importing with dual progress bars
    (files and features) and cancellation support.
    """
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importing Shapefiles")
        self.setFixedSize(450, 150)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        # Prevent user from closing window without explicit cancel (optional, but good for robustness)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_status = QLabel("Initializing import...")
        self.lbl_status.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.lbl_status)

        self.lbl_stats = QLabel("Processed: 0 features | Errors: 0")
        layout.addWidget(self.lbl_stats)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel Import")
        self.btn_cancel.clicked.connect(self.on_cancel)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)

    def apply_style(self):
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
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
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #006400; /* Green */
                width: 1px;
            }
        """)

    def on_cancel(self):
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Canceling...")
        self.cancel_requested.emit()

    def update_progress(self, percent, processed_count, error_count, message=""):
        self.progress_bar.setValue(percent)
        self.lbl_stats.setText(f"Processed: {processed_count} features | Errors: {error_count}")
        if message:
            self.lbl_status.setText(message)
        # Ensure UI repaints to avoid "stucking"
        QApplication.processEvents()

