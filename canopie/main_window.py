import os, sys, json, logging, tempfile, subprocess
from functools import partial
import pickle

# --- third-party
import numpy as np
import cv2
import exifread
import folium
from shapely import geometry
from shapely.geometry import MultiPolygon
from geopy.distance import geodesic
from scipy.spatial import KDTree
import sip

# --- Qt (PyQt5)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, pyqtSlot, QSettings, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QMessageBox,
    QToolBar, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QSlider,
    QLabel, QPushButton, QListWidget, QInputDialog, QSizePolicy, QStyle,
    QShortcut, QAbstractSlider
)
import csv  # NEW

# --- your modules
from .project_tab import ProjectTab

# (optional) keep this: prevent basicConfig from adding FileHandlers
_original_basicConfig = logging.basicConfig
def _no_file_basicConfig(*args, **kwargs):
    kwargs.pop('filename', None)
    handlers = kwargs.get('handlers')
    if handlers:
        from logging import FileHandler
        kwargs['handlers'] = [h for h in handlers if not isinstance(h, FileHandler)]
    return _original_basicConfig(*args, **kwargs)
logging.basicConfig = _no_file_basicConfig

def setup_logging_light(app_name="CanoPie", level=logging.INFO):
    # Don’t try to print handler errors to missing stderr in frozen apps
    logging.raiseExceptions = False

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any auto-added/duplicate handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler (always exists)
    from logging.handlers import RotatingFileHandler
    logdir = os.path.join(tempfile.gettempdir(), app_name)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "app.log")

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")

    fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)

    # Optional console stream: only add if a *real* stream exists
    real_stderr = getattr(sys, "__stderr__", None) or getattr(sys, "stderr", None)
    if real_stderr and hasattr(real_stderr, "write"):
        sh = logging.StreamHandler(real_stderr)
        sh.setFormatter(fmt)
        sh.setLevel(level)
        root.addHandler(sh)

    # Log uncaught exceptions to file
    def _excepthook(etype, value, tb):
        logging.getLogger("uncaught").exception("Uncaught exception", exc_info=(etype, value, tb))
    sys.excepthook = _excepthook

    return root, logfile


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.status = self.statusBar()

        # set up logging here (instance scope)
        self.logger, self.logfile = setup_logging_light()

        # Thread-safe status bar logging handler using Qt signals
        class _StatusBarSignals(QObject):
            message = pyqtSignal(str, int)
        
        class _StatusBarHandler(logging.Handler):
            def __init__(inner_self, main_window):
                super().__init__()
                inner_self._signals = _StatusBarSignals()
                inner_self._main_window = main_window
                # Connect signal to slot (this runs in main thread during __init__)
                inner_self._signals.message.connect(inner_self._show_message)
            
            def _show_message(inner_self, msg, timeout):
                """Slot that runs in main thread - safe to access Qt widgets."""
                try:
                    if hasattr(inner_self._main_window, "status"):
                        inner_self._main_window.status.showMessage(msg, timeout)
                except Exception:
                    pass
            
            def emit(inner_self, record):
                """Called from any thread - emits signal instead of direct Qt access."""
                try:
                    if record.levelno >= logging.INFO:
                        # Emit signal - Qt will queue this for main thread
                        inner_self._signals.message.emit(record.getMessage(), 4000)
                except Exception:
                    pass
        
        self._status_handler = _StatusBarHandler(self)
        logging.getLogger().addHandler(self._status_handler)

        logging.info("CanoPie started")
        logging.info(f"Log file: {self.logfile}")

    def init_ui(self):
        self.setWindowTitle("CanoPie V_1_0_0")
        self.setGeometry(100, 100, 1200, 800)

        # Create the QTabWidget for managing tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Add a default tab
        self.add_new_tab("Project 1")

        # Set up the menu bar
        self.setup_menu()

        # Set up the main toolbar
        self.setup_main_toolbar()

        # Add corner buttons (Undo/Redo)
        self.setup_tab_corner_widget()

        # Add tab-switching shortcuts
        self.add_tab_shortcuts()
        
        # Enable movable tabs
        self.tab_widget.setMovable(True)
    def open_log_file(self):
        path = getattr(self, "logfile", None)
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Logs", "No log file found yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            logging.info(f"Opened log file: {path}")
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            QMessageBox.critical(self, "Logs", f"Failed to open log file:\n{e}")

    def setup_menu(self):
        # Create a menu bar
        menu_bar = self.menuBar()

        # Create the File menu
        file_menu = menu_bar.addMenu("File")
        
        # Create Edit menu (inserted after File)
        edit_menu = menu_bar.addMenu("Edit")
        
        # Undo Action
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setStatusTip("Undo last polygon action")
        undo_action.triggered.connect(self.undo_action_triggered)
        edit_menu.addAction(undo_action)
        self.undo_action = undo_action  # Keep reference
        
        # Redo Action
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.setStatusTip("Redo last polygon action")
        redo_action.triggered.connect(self.redo_action_triggered)
        edit_menu.addAction(redo_action)
        self.redo_action = redo_action  # Keep reference

        # Existing File menu actions
        new_tab_action = QAction("New Tab", self)
        new_tab_action.setShortcut("Ctrl+N")
        new_tab_action.triggered.connect(self.add_new_tab)
        file_menu.addAction(new_tab_action)

        close_tab_action = QAction("Close Current Tab", self)
        close_tab_action.setShortcut("Ctrl+W")
        close_tab_action.triggered.connect(self.close_current_tab)
        file_menu.addAction(close_tab_action)

        file_menu.addSeparator()

        # Existing actions for opening, loading, saving projects
        open_folder_action = QAction("Open multispectral/multiple Folders", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
         
        open_rgb_action = QAction("Open RGB/stack Folder", self)
        open_rgb_action.setShortcut("Ctrl+R")
        open_rgb_action.setStatusTip("Open a folder containing RGB/stacked images")
        open_rgb_action.triggered.connect(self.open_rgb_folder)
        file_menu.addAction(open_rgb_action)
        
        open_rgb_picker_action = QAction("Open Image files (Select Files…)", self)
        open_rgb_picker_action.setShortcut("Ctrl+Shift+R")
        open_rgb_picker_action.setStatusTip("Select specific images with preview and choose images-per-root")
        open_rgb_picker_action.triggered.connect(self.open_rgb_picker)
        file_menu.addAction(open_rgb_picker_action)
        
        file_menu.addSeparator()

        load_project_action = QAction("Load Project", self)
        load_project_action.setShortcut("Ctrl+L")
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)
        
       


        # New: Load Multiple Projects
        load_multiple_projects_action = QAction("Load Multiple Projects", self)
        load_multiple_projects_action.setShortcut("Ctrl+M")
        load_multiple_projects_action.setStatusTip("Load all projects from a folder, each in a new tab")
        load_multiple_projects_action.triggered.connect(self.load_all_projects_from_folder)
        file_menu.addAction(load_multiple_projects_action)


        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut("Ctrl+Shift+S")
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
            # New: Change Folders Path
        change_folders_path_action = QAction("Change Folders Path", self)
        change_folders_path_action.setShortcut("Ctrl+Shift+F")
        change_folders_path_action.setStatusTip("Change the folder paths for the current project")
        change_folders_path_action.triggered.connect(self.change_folders_path)
        file_menu.addAction(change_folders_path_action)

        # New: Change Batch Size
        change_batch_size_action = QAction("Change Batch Size", self)
        change_batch_size_action.setShortcut("Ctrl+Shift+B")
        change_batch_size_action.setStatusTip("Change the batch size for the current project")
        change_batch_size_action.triggered.connect(self.change_batch_size)
        file_menu.addAction(change_batch_size_action)
        
        # New: Filter Similar Images
        filter_similar_action = QAction("Filter Similar Images...", self)
        filter_similar_action.setStatusTip("Filter images by similarity to reference images")
        filter_similar_action.triggered.connect(self.filter_similar_images_current_tab)
        file_menu.addAction(filter_similar_action)
        
        file_menu.addSeparator()

        save_current_csv_action = QAction("Extract polygons CSV from Current Project", self)
        save_current_csv_action.setShortcut("Ctrl+Alt+S")
        save_current_csv_action.setStatusTip("Save all polygon data to CSV for the current project")
        save_current_csv_action.triggered.connect(self.save_all_csv_current_tab)
        file_menu.addAction(save_current_csv_action)

        save_all_csv_action = QAction("Extract polygons CSV from All Projects", self)
        save_all_csv_action.setShortcut("Ctrl+Shift+A")
        save_all_csv_action.setStatusTip("Save all polygon data to CSV for all open projects")
        save_all_csv_action.triggered.connect(self.save_all_csv_all_projects)
        file_menu.addAction(save_all_csv_action)
        
        
        exif_current_action = QAction("Extract all images EXIF from Current Project", self)
        exif_current_action.setShortcut("Ctrl+E")
        exif_current_action.setStatusTip("Export EXIF for all images in the current project")
        exif_current_action.triggered.connect(self.save_exif_csv_current_tab)
        file_menu.addAction(exif_current_action)

        exif_all_action = QAction("Extract all images EXIF from All Projects", self)
        exif_all_action.setShortcut("Ctrl+Shift+E")
        exif_all_action.setStatusTip("Export EXIF for all images in every open project (same delimiter)")
        exif_all_action.triggered.connect(self.save_exif_csv_all_projects)
        file_menu.addAction(exif_all_action)


        file_menu.addSeparator()

        # New Action: Set exiftool Path
        set_exiftool_path_act = QAction("Set exiftool Path", self)
        set_exiftool_path_act.triggered.connect(self.set_exiftool_path)
        file_menu.addAction(set_exiftool_path_act)
    
        file_menu.addSeparator()
       

    
        
        # New Action: Save All Thumbnails
        save_all_thumbnails_action = QAction("Save All Thumbnails", self)
        save_all_thumbnails_action.setShortcut("Ctrl+T")
        save_all_thumbnails_action.setStatusTip("Save thumbnails for all polygons in the current project")
        save_all_thumbnails_action.triggered.connect(self.save_all_thumbnails_current_tab)
        file_menu.addAction(save_all_thumbnails_action)
        # New Action: Export Project Images (below "Save All Thumbnails")
        export_images_action = QAction("Export Project Images", self)
        export_images_action.setShortcut("Ctrl+Shift+X")
        export_images_action.setStatusTip("Export current project's images with .ax applied and copy EXIF")
        export_images_action.triggered.connect(self.export_project_images_current_tab)
        file_menu.addAction(export_images_action)

        
        file_menu.addSeparator()

        # Exit Action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        file_menu.addSeparator()

        view_menu = menu_bar.addMenu("View")
        open_log_action = QAction("Open Log File", self)
        open_log_action.setStatusTip("Open the current log file in your system editor")
        open_log_action.setShortcut("F12")
        open_log_action.triggered.connect(self.open_log_file)
        view_menu.addAction(open_log_action)
    
        # NEW: CSV Viewer
        csv_viewer_action = QAction("CSV Viewer…", self)  # NEW
        csv_viewer_action.setShortcut("Ctrl+Shift+V")     # NEW
        csv_viewer_action.setStatusTip("Open and preview a CSV file")  # NEW
        csv_viewer_action.triggered.connect(self.open_csv_viewer)      # NEW
        view_menu.addAction(csv_viewer_action)           # NEW



    def open_rgb_picker(self):
        current_tab = self.get_current_tab()
        if not current_tab:
            QtWidgets.QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")
            return
        try:
            if hasattr(current_tab, "open_rgb_images_with_picker"):
                current_tab.open_rgb_images_with_picker()
            elif hasattr(current_tab, "open_rgb_files_quick"):
                # Fallback to quick multi-file chooser if the full picker isn't present
                current_tab.open_rgb_files_quick()
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Not Implemented",
                    "This tab does not implement 'open_rgb_images_with_picker' or 'open_rgb_files_quick'."
                )
        except Exception as e:
            logging.exception("RGB picker failed")
            QtWidgets.QMessageBox.critical(
                self, "Open Error",
                f"Failed to open selected images:\n{e}"
            )

       
    def export_project_images_current_tab(self):
        current_tab = self.get_current_tab()
        if not current_tab:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")
            return
        try:
            current_tab.export_project_images()
        except Exception as e:
            logging.exception("Export failed")
            QMessageBox.critical(self, "Export Error", f"Failed to export images:\n{e}")
       
       
       
       
    def save_all_thumbnails_current_tab(self):
        current_tab = self.get_current_tab()
        if current_tab:
            try:
                result = current_tab.save_all_thumbnails()
                # If result is None, export is running in background - Export Manager shows progress
                # If result is a path, foreground export completed
                if result is None:
                    # Background mode - Export Manager handles notification
                    logging.info(f"Thumbnail export started in background for project '{current_tab.project_name}'.")
                elif result:
                    # Foreground mode completed
                    QMessageBox.information(
                        self,
                        "Thumbnails Saved",
                        f"All thumbnails have been successfully saved for project '{current_tab.project_name}'."
                    )
                    logging.info(f"Thumbnails saved for project '{current_tab.project_name}'.")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save thumbnails for project '{current_tab.project_name}':\n{e}"
                )
                logging.error(f"Failed to save thumbnails for project '{current_tab.project_name}': {e}")
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")        
        
    def change_folders_path(self):
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.change_folders_path()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")

    def change_batch_size(self):
        current_tab = self.get_current_tab()
        if current_tab:
            if hasattr(current_tab, "change_batch_size"):
                current_tab.change_batch_size()
            else:
                QMessageBox.warning(self, "Not Supported", "Change Batch Size is not supported by this tab.")
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")

    def filter_similar_images_current_tab(self):
        current_tab = self.get_current_tab()
        if current_tab:
            if hasattr(current_tab, "filter_similar_images"):
                current_tab.filter_similar_images(parent_window=self)
            else:
                QMessageBox.warning(self, "Not Supported", "Filter Similar is not supported by this tab.")
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")
     
    def open_rgb_folder(self):
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.open_rgb_folder()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")

    def switch_to_specific_tab(self, project_name):
        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if hasattr(tab, 'project_name') and tab.project_name == project_name:
                self.tab_widget.setCurrentIndex(index)
                logging.info(f"Switched to tab: {project_name}")
                return
        logging.warning(f"Project tab '{project_name}' not found.")
        QMessageBox.warning(self, "Tab Not Found", f"No tab named '{project_name}' was found.")        
        

    def setup_main_toolbar(self):
        main_toolbar = QToolBar("Main Toolbar")
        main_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(main_toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_toolbar.addWidget(spacer)

        add_tab_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        add_tab_action = QAction(add_tab_icon, "Add Tab", self)
        add_tab_action.setToolTip("Add a new tab")
        add_tab_action.triggered.connect(self.add_new_tab)
        main_toolbar.addAction(add_tab_action)

        close_tab_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
        close_tab_action = QAction(close_tab_icon, "Close Tab", self)
        close_tab_action.setToolTip("Close the current tab")
        close_tab_action.triggered.connect(self.close_current_tab)
        main_toolbar.addAction(close_tab_action)
        
        open_project_folder_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)
        open_project_folder_action = QAction(open_project_folder_icon, "Open Project Folder", self)
        open_project_folder_action.setToolTip("Open the current project folder in the system's file explorer")
        open_project_folder_action.triggered.connect(self.open_project_folder_in_explorer)
        main_toolbar.addAction(open_project_folder_action)
        
        # New: Random Forest Model Loading Icon
        rf_icon = self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        load_rf_action = QAction(rf_icon, "Load Random Forest Model", self)
        load_rf_action.setToolTip("Load scikit-learn learn image classification")
        load_rf_action.triggered.connect(self.load_random_forest_model)
        main_toolbar.addAction(load_rf_action)

    def open_project_folder_in_explorer(self):
        current_tab = self.get_current_tab()
        if current_tab and hasattr(current_tab, 'project_folder') and current_tab.project_folder:
            try:
                path = current_tab.project_folder
                if os.name == "nt":
                    os.startfile(path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
                logging.info(f"Opened project folder: {path}")
            except Exception as e:
                logging.error(f"Failed to open project folder '{current_tab.project_folder}': {e}")
                QMessageBox.critical(self, "Open Folder Error", f"Failed to open project folder:\n{e}")
        else:
            logging.warning("Project folder is not set for the current tab.")
            QMessageBox.warning(self, "No Project Folder", "The project folder is not set for the current project tab.")

    def load_random_forest_model(self):
        """
        Prompts the user to select a Random Forest model file (e.g., a pickle file),
        loads the model, and sets it as the shared model for all project tabs.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        model_file, _ = QFileDialog.getOpenFileName(
            self, "Select scikit-learn Model", "", "Pickle Files (*.pkl);;All Files (*)", options=options)
        if model_file:
            try:
                with open(model_file, 'rb') as f:
                    loaded_model = pickle.load(f)
                # Set the loaded model as the shared model for all tabs
                ProjectTab.shared_random_forest_model = loaded_model
                # Update each open tab's random forest model
                for index in range(self.tab_widget.count()):
                    tab = self.tab_widget.widget(index)
                    if hasattr(tab, 'random_forest_model'):
                        tab.random_forest_model = loaded_model
                QMessageBox.information(self, "Model Loaded", "Random Forest model loaded successfully.")
                logging.info("Random Forest model loaded and set as shared model.")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load Random Forest model:\n{e}")
                logging.error(f"Failed to load Random Forest model: {e}")

    def setup_tab_corner_widget(self):
        corner_widget = QWidget()
        layout = QHBoxLayout(corner_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.undo_btn = QPushButton()
        self.undo_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_action_triggered)
        
        self.redo_btn = QPushButton()
        self.redo_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo_action_triggered)
        
        layout.addWidget(self.undo_btn)
        layout.addWidget(self.redo_btn)
        
        self.tab_widget.setCornerWidget(corner_widget, Qt.TopRightCorner)

    def undo_action_triggered(self):
        current_tab = self.get_current_tab()
        if current_tab and hasattr(current_tab, "undo"):
            current_tab.undo()
        else:
            self.statusBar().showMessage("Nothing to undo or undo not supported here.", 2000)

    def redo_action_triggered(self):
        current_tab = self.get_current_tab()
        if current_tab and hasattr(current_tab, "redo"):
            current_tab.redo()
        else:
            self.statusBar().showMessage("Nothing to redo or redo not supported here.", 2000)

    def add_new_tab(self, project_name=None):
        project_number = self.tab_widget.count() + 1
        project_name = project_name or f"Project {project_number}"
        new_tab = ProjectTab(project_name, self.tab_widget)
        self.tab_widget.addTab(new_tab, project_name)
        self.tab_widget.setCurrentWidget(new_tab)
        logging.info(f"Added new tab: {project_name}")
    
    def close_current_tab(self):
        current_index = self.tab_widget.currentIndex()
        if current_index != -1:
            tab = self.tab_widget.widget(current_index)
            project_name = getattr(tab, 'project_name', "Untitled Project")
            reply = QMessageBox.question(
                self,
                "Save Project",
                f"Do you want to save the project '{project_name}' before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                if hasattr(tab, 'save_project_quick') and callable(getattr(tab, 'save_project_quick')):
                    tab.save_project_quick()
                else:
                    QMessageBox.warning(self, "Save Not Available", f"Save functionality is not available for project '{project_name}'.")
            elif reply == QMessageBox.Cancel:
                return
            self.tab_widget.removeTab(current_index)
            logging.info(f"Closed tab: {project_name}")

    def get_current_tab(self):
        current_widget = self.tab_widget.currentWidget()
        if hasattr(current_widget, 'project_name'):
            return current_widget
        return None

    def open_folder(self):
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.open_folder()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")

    def load_project(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        project_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Project Folder", os.path.expanduser("~"), options=options)
        if not project_folder:
            return

        project_json_path = os.path.join(project_folder, 'project.json')
        if not os.path.exists(project_json_path):
            QMessageBox.warning(
                self, "Invalid Project Folder",
                "The selected folder does not contain a 'project.json' file.")
            return

        try:
            with open(project_json_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load the project file:\n{e}")
            return

        required_keys = {"all_polygons", "current_root_index", "root_offset", "root_coordinates"}
        if not required_keys.issubset(project_data.keys()):
            QMessageBox.critical(
                self, "Corrupted Project",
                "The project data is incomplete or corrupted.")
            return

        # For simplicity, update the project name based on folder name
        project_name = os.path.basename(os.path.normpath(project_folder))
        if hasattr(self, 'update_project_name'):
            self.update_project_name(project_name)

        self.add_new_tab(project_name)
        current_tab = self.get_current_tab()
        try:
            current_tab.load_project(project_folder)
            logging.info(f"Project loaded from {project_folder}")
        except Exception as e:
            logging.error(f"Failed to load project from {project_folder}: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")

    def save_project(self):
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.save_project()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")

    def save_all_projects(self):
        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if hasattr(tab, 'project_name'):
                project_file = f"{tab.project_name}.proj"
                if hasattr(tab, 'save_project_file'):
                    tab.save_project_file(project_file)
        QMessageBox.information(self, "Save All", "All projects have been saved.")
        logging.info("All projects have been saved.")
        
    def save_all_csv_current_tab(self):
        current_tab = self.get_current_tab()
        if current_tab:
            try:
                saved_path = current_tab.save_polygons_to_csv()  # <-- must return path on success, None on cancel/background
                if saved_path is None:
                    # User cancelled OR background export started - Export Manager shows progress
                    # Don't show any popup here
                    logging.info(f"CSV export cancelled or started in background for project '{current_tab.project_name}'.")
                    return
                # Foreground export completed successfully
                QMessageBox.information(self, "Save Successful",
                                        f"CSV saved for project '{current_tab.project_name}'.\n\n{saved_path}")
                logging.info(f"CSV saved for project '{current_tab.project_name}' at: {saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error",
                                     f"Failed to save CSV for project '{current_tab.project_name}':\n{e}")
                logging.error(f"Failed to save CSV for project '{current_tab.project_name}': {e}")
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab to save CSV.")
            logging.warning("Attempted to save CSV, but no ProjectTab is currently selected.")

            
    def save_all_csv_all_projects(self):
        if self.tab_widget.count() == 0:
            QMessageBox.information(self, "No Projects", "There are no open projects to save.")
            logging.info("Save All CSV triggered, but no projects are open.")
            return

        # 1. Ask for options ONCE using the first valid project tab as context
        #    This allows auto-detecting NoData from the first project's .ax files
        first_tab = None
        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if hasattr(tab, 'project_name'):
                first_tab = tab
                break
        
        if not first_tab:
            return

        # Local import to avoid circular dependency
        from .machine_learning_manager import AnalysisOptionsDialog
        
        dlg = AnalysisOptionsDialog(first_tab)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            logging.info("Save All CSV cancelled by user in options dialog.")
            return  # User cancelled the global options

        common_options = dlg.get_options()
        logging.info(f"Save All CSV: Using common options: {common_options.keys()}")

        # 2. Ask for Processing mode options ONCE
        # Use helper on the first tab
        if hasattr(first_tab, "get_export_processing_options"):
             processing_params = first_tab.get_export_processing_options()
             if processing_params is None:
                 logging.info("Save All CSV cancelled by user in export processing dialog.")
                 return
             common_options['processing_params'] = processing_params

        success_projects = []   # (project_name, path)
        failed_projects = []    # (project_name, error)
        background_or_cancelled = [] # [project_name] - either background mode or user cancelled

        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if not hasattr(tab, 'project_name'):
                continue
            try:
                # Pass the common options to suppress per-project dialog
                saved_path = tab.save_polygons_to_csv(options=common_options)
                
                if saved_path:
                    # Foreground export completed successfully
                    success_projects.append((tab.project_name, saved_path))
                    logging.info(f"CSV saved for project '{tab.project_name}' at: {saved_path}")
                else:
                    # None means either user cancelled (not possible here since we passed options) 
                    # OR background export started
                    background_or_cancelled.append(tab.project_name)
                    logging.info(f"CSV export started (background) for project '{tab.project_name}'.")
            except Exception as e:
                failed_projects.append((tab.project_name, str(e)))
                logging.error(f"Failed to save CSV for project '{tab.project_name}': {e}")

        # Build user message - only show popup for foreground completions
        parts = []
        if success_projects:
            parts.append("CSV saved for the following projects:")
            for name, path in success_projects:
                parts.append(f"  • {name}: {path}")
        if failed_projects:
            parts.append("\nFailed to save CSV for:")
            for name, err in failed_projects:
                parts.append(f"  • {name}: {err}")
        
        # Only show cancel message if ALL exports were cancelled (no success, no fails, no background)
        if background_or_cancelled and not success_projects and not failed_projects:
            # All exports running in background likely
            logging.info("All CSV exports running in background.")
            return

        if success_projects and not failed_projects:
            QMessageBox.information(self, "Save All Successful", "\n".join(parts))
        elif success_projects and failed_projects:
            QMessageBox.warning(self, "Partial Save", "\n".join(parts))
        elif failed_projects:
            QMessageBox.critical(self, "Save All Failed", "\n".join(parts))

    
    def save_exif_csv_current_tab(self):
        """
        Export EXIF-only CSV for the current tab.
        Uses that tab's _active_csv_delimiter (or ',' if unset).
        """
        current_tab = self.get_current_tab()
        if not current_tab:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab.")
            return

        try:
            # Ensure delimiter exists on the tab
            if not hasattr(current_tab, "_active_csv_delimiter") or not current_tab._active_csv_delimiter:
                current_tab._active_csv_delimiter = ","
            current_tab.extract_exif_to_csv()
            QMessageBox.information(self, "EXIF Export", f"EXIF CSV exported for '{current_tab.project_name}'.")
            logging.info(f"EXIF CSV exported for project '{current_tab.project_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export EXIF CSV:\n{e}")
            logging.error(f"Failed to export EXIF CSV for '{current_tab.project_name}': {e}")

    def save_exif_csv_all_projects(self):
        """
        Export EXIF-only CSV for ALL open projects.
        Forces the SAME delimiter across all tabs:
        - Uses the first tab's _active_csv_delimiter if present,
          otherwise defaults to ','.
        """
        if self.tab_widget.count() == 0:
            QMessageBox.information(self, "No Projects", "There are no open projects to export.")
            return

        # Determine a single delimiter to use everywhere
        unified_delim = ","
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, "_active_csv_delimiter") and tab._active_csv_delimiter:
                unified_delim = tab._active_csv_delimiter
                break

        success, failed = [], []
        run_bg = None
        
        # Determine total images to export for the prompt
        total_images = 0
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, "_collect_all_image_paths_for_exif"):
                paths = tab._collect_all_image_paths_for_exif()
                if paths:
                    total_images += len(paths)

        if total_images > 0:
            reply = QMessageBox.question(
                self, "Run in Background?",
                f"Export EXIF data from {total_images} images across all projects.\n\n"
                "Run in background to continue using the app?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Cancel:
                return
            run_bg = (reply == QMessageBox.Yes)

        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if not hasattr(tab, "project_name"):
                continue
            try:
                tab._active_csv_delimiter = unified_delim  # enforce same delimiter
                tab.extract_exif_to_csv(run_background=run_bg)
                success.append(tab.project_name)
                logging.info(f"EXIF CSV exported for project '{tab.project_name}'.")
            except Exception as e:
                failed.append((tab.project_name, str(e)))
                logging.error(f"Failed EXIF export for '{tab.project_name}': {e}")

        # Report
        if success and not failed:
            QMessageBox.information(self, "EXIF Export", "EXIF CSV exported for all projects.")
        elif success and failed:
            msg = "EXIF CSV exported for:\n- " + "\n- ".join(success) + "\n\nFailures:\n"
            msg += "\n".join(f"- {p}: {err}" for p, err in failed)
            QMessageBox.warning(self, "Partial EXIF Export", msg)
        else:
            QMessageBox.critical(self, "EXIF Export Failed", "Could not export EXIF CSV for any project.")
    
    
    def set_exiftool_path(self):
        """
        Resolve an exiftool path (auto-detect if possible), then set it on every ProjectTab.
        """
        path = self._resolve_exiftool_path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Exiftool Not Found",
                                          "Could not locate exiftool. Please install it or pick it manually.")
            return

        self.exiftool_path = path
        QtWidgets.QMessageBox.information(self, "Exiftool Path Set",
                                          f"Exiftool path set to:\n{path}")

        # Push to all tabs that support it
        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if hasattr(tab, "set_exiftool_path"):
                tab.set_exiftool_path(path)
# Still inside MainWindow
    def _resolve_exiftool_path(self):
        """
        Try common strategies to find exiftool:
        1) Use self.exiftool_path if valid
        2) Use PATH (shutil.which)
        3) Ask the user to pick the executable (with Windows-friendly filter)
        4) Let the user pick a folder and auto-detect inside it
        Returns: absolute path string or None
        """
        import os, sys, shutil, glob
        from PyQt5 import QtWidgets

        # 1) Existing and valid?
        if getattr(self, "exiftool_path", None) and os.path.isfile(self.exiftool_path):
            return self.exiftool_path

        # 2) PATH lookup
        candidates = ["exiftool.exe", "exiftool", "exiftool.pl", "exiftool(-k).exe"]
        for name in candidates:
            found = shutil.which(name)
            if found:
                return os.path.abspath(found)

        # 3) Direct file pick
        if sys.platform.startswith("win"):
            filter_str = "exiftool executables (exiftool*.exe);;Executables (*.exe);;All files (*.*)"
        else:
            filter_str = "All files (*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Locate exiftool", "", filter_str
        )
        if file_path and os.path.isfile(file_path):
            return os.path.abspath(file_path)

        # 4) Folder pick + auto-detect inside
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder that contains exiftool")
        if folder:
            patterns = ["exiftool*.exe", "exiftool", "exiftool.pl"]
            for pat in patterns:
                matches = glob.glob(os.path.join(folder, pat))
                if matches:
                    return os.path.abspath(matches[0])

        return None
                

    def add_tab_shortcuts(self):
        self.shortcut_prev_tab = QShortcut(QKeySequence("Alt+1"), self)
        self.shortcut_prev_tab.activated.connect(self.switch_to_previous_tab)
        self.shortcut_next_tab = QShortcut(QKeySequence("Alt+2"), self)
        self.shortcut_next_tab.activated.connect(self.switch_to_next_tab)

    def switch_to_previous_tab(self):
        current_index = self.tab_widget.currentIndex()
        if current_index > 0:
            self.tab_widget.setCurrentIndex(current_index - 1)
            logging.info(f"Switched to previous tab: {self.tab_widget.tabText(current_index - 1)}")
        else:
            logging.info("Already at the first tab. Cannot switch to previous.")

    def switch_to_next_tab(self):
        current_index = self.tab_widget.currentIndex()
        if current_index < self.tab_widget.count() - 1:
            self.tab_widget.setCurrentIndex(current_index + 1)
            logging.info(f"Switched to next tab: {self.tab_widget.tabText(current_index + 1)}")
        else:
            logging.info("Already at the last tab. Cannot switch to next.")
    
    def open_csv_viewer(self):
        import os
        start_dir = ""
        tab = self.get_current_tab()
        if tab and getattr(tab, "project_folder", None):
            cand = os.path.join(tab.project_folder, "exports")
            if os.path.isdir(cand):
                start_dir = cand
            else:
                start_dir = tab.project_folder  # fallback

        dlg = CSVViewerDialog(self, start_dir=start_dir)
        dlg.exec_()
        

    def load_all_projects_from_folder(self):
        """
        Prompts the user to select a folder containing multiple project folders.
        Each subfolder with a 'project.json' file is loaded as a new project tab.
        """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder Containing Projects", os.path.expanduser("~"), options=options)
        if not folder_path:
            return
        
        for entry in os.listdir(folder_path):
            project_folder = os.path.join(folder_path, entry)
            if os.path.isdir(project_folder):
                project_json_path = os.path.join(project_folder, "project.json")
                if os.path.exists(project_json_path):
                    self.add_new_tab(project_name=entry)
                    current_tab = self.get_current_tab()
                    try:
                        current_tab.load_project(project_folder)
                        logging.info(f"Loaded project from {project_folder} into tab '{entry}'")
                    except Exception as e:
                        logging.error(f"Failed to load project from {project_folder}: {e}")
                        QMessageBox.warning(self, "Load Error", f"Failed to load project from {project_folder}:\n{e}")
                else:
                    logging.info(f"Folder '{project_folder}' does not contain a project.json file.")

# === NEW: Lightweight CSV Viewer (no external deps) ===========================
class CsvTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=None, header=None, parent=None):
        super().__init__(parent)
        self._data = data or []      # list[list[str]]
        self._header = header or []  # list[str]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if not self._data:
            return len(self._header)
        return max(len(r) for r in self._data)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None
        r, c = index.row(), index.column()
        try:
            return str(self._data[r][c])
        except Exception:
            return ""

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and self._header:
            if 0 <= section < len(self._header):
                return self._header[section]
        return super().headerData(section, orientation, role)

    def load_from_csv(self, filepath, delimiter=None, has_header=True, encoding_list=("utf-8-sig","utf-8","latin-1")):
        rows, header = [], []
        # Try encodings in order
        last_err = None
        for enc in encoding_list:
            try:
                with open(filepath, "r", encoding=enc, newline="") as fh:
                    sample = fh.read(4096)
                    fh.seek(0)
                    # Auto-detect delimiter if not specified
                    if not delimiter:
                        try:
                            sniffer = csv.Sniffer()
                            dialect = sniffer.sniff(sample, delimiters=[",",";","\t","|"])
                            delimiter = dialect.delimiter
                        except Exception:
                            delimiter = ","
                    reader = csv.reader(fh, delimiter=delimiter)
                    if has_header:
                        try:
                            header = next(reader, [])
                        except StopIteration:
                            header = []
                    rows = [row for row in reader]
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err

        self.beginResetModel()
        self._data = rows
        self._header = header
        self.endResetModel()



class CSVViewerDialog(QtWidgets.QDialog):
    BIG_FILE_BYTES = 2 * 1024 * 1024     # 25 MB threshold (tweak if you want)
    SAMPLE_ROWS    = 1000                  # how many data rows to show when big
    SNIFF_BYTES    = 256 * 1024           # read up to 256 KB to sniff dialect

    def __init__(self, parent=None, start_dir=""):
        super().__init__(parent)
        from PyQt5 import QtGui  # for QKeySequence in shortcuts

        self.start_dir = start_dir or ""
        self.setWindowTitle("CSV Viewer")
        self.resize(900, 600)

        # --- Top bar (path + buttons) ---
        top = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select a CSV…")
        browse_btn = QtWidgets.QPushButton("Browse…")
        load_btn   = QtWidgets.QPushButton("Load")

        top.addWidget(self.path_edit, 1)
        top.addWidget(browse_btn)
        top.addWidget(load_btn)

        # --- Notice (sampling / status) ---
        self.notice = QtWidgets.QLabel("")                 # shows sampling notice
        self.notice.setStyleSheet("color:#b58900;")        # subtle amber
        self.notice.setWordWrap(True)

        # --- Table ---
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)

        # Enable copy via shortcuts + context menu
        QtWidgets.QShortcut(QtGui.QKeySequence.Copy, self.table,
                            activated=self._copy_selection_to_clipboard)
        QtWidgets.QShortcut(QtGui.QKeySequence.SelectAll, self.table,
                            activated=self.table.selectAll)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), self.table,
                            activated=self._copy_all_to_clipboard)

        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)

        # --- Layout ---
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.notice)
        lay.addWidget(self.table, 1)

        # --- Signals ---
        browse_btn.clicked.connect(self._on_browse)
        load_btn.clicked.connect(self._on_load)

    # ---------- copy helpers ----------
    def _range_to_tsv(self, r0, r1, c0, c1, include_header=False):
        """Return TSV text for a rectangular block [r0..r1], [c0..c1] (inclusive)."""
        parts = []
        if include_header:
            headers = []
            for c in range(c0, c1 + 1):
                hitem = self.table.horizontalHeaderItem(c)
                headers.append("" if hitem is None else hitem.text())
            parts.append("\t".join(headers))

        for r in range(r0, r1 + 1):
            row_vals = []
            for c in range(c0, c1 + 1):
                it = self.table.item(r, c)
                row_vals.append("" if it is None else it.text())
            parts.append("\t".join(row_vals))
        return "\n".join(parts)

    def _copy_selection_to_clipboard(self):
        """Copy the current selection as TSV. If multiple blocks, copy each block separated by a blank line."""
        rngs = self.table.selectedRanges()
        if not rngs:
            return
        chunks = []
        # If user selected a contiguous block, include header row to be nice
        include_header = (len(rngs) == 1)
        for r in rngs:
            text = self._range_to_tsv(
                r.topRow(), r.bottomRow(), r.leftColumn(), r.rightColumn(),
                include_header=include_header
            )
            chunks.append(text)
        QtWidgets.QApplication.clipboard().setText("\n\n".join(chunks))

    def _copy_all_to_clipboard(self):
        """Copy the entire table (visible data) as TSV with header."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        if rows == 0 or cols == 0:
            QtWidgets.QApplication.clipboard().clear()
            return
        text = self._range_to_tsv(0, rows - 1, 0, cols - 1, include_header=True)
        QtWidgets.QApplication.clipboard().setText(text)

    def _show_table_menu(self, pos):
        menu = QtWidgets.QMenu(self.table)
        act_copy = menu.addAction("Copy")
        act_copy_all = menu.addAction("Copy All (with header)")
        act = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if act == act_copy:
            self._copy_selection_to_clipboard()
        elif act == act_copy_all:
            self._copy_all_to_clipboard()

    # ---------- file I/O helpers ----------
    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select CSV", self.start_dir or "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)
            self._on_load()

    def _on_load(self):
        path = self.path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "CSV", "Please choose a CSV file.")
            return
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.critical(self, "CSV", "File does not exist.")
            return
        try:
            headers, rows, sampled, approx_rows = self._load_csv_streaming(path)
            self._populate_table(headers, rows)
            if sampled:
                msg = f"Showing {len(rows):,} sampled rows (of a large file"
                if approx_rows is not None:
                    msg += f", ~{approx_rows:,} total"
                msg += ")."
                self.notice.setText(msg)
            else:
                self.notice.setText("")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"Failed to load CSV:\n{e}")

    def _load_csv_streaming(self, path):
        """
        Returns: (headers: list[str], rows: list[list[str]],
                  sampled: bool, approx_rows: Optional[int])
        - For big files (> BIG_FILE_BYTES), do reservoir sampling of SAMPLE_ROWS data rows.
        - Always preserves the header row.
        """
        fsize = os.path.getsize(path)
        sampled_mode = fsize > self.BIG_FILE_BYTES

        # --- sniff encoding & delimiter from a small chunk ---
        # Try UTF-8-sig first; if that fails, fallback to latin-1 to avoid crashes
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
        last_err = None
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, newline="") as fh:
                    sample = fh.read(self.SNIFF_BYTES)
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                    except Exception:
                        dialect = csv.excel
                    # Also detect header if possible (optional; we still treat first row as header)
                    fh.seek(0)
                    reader = csv.reader(fh, dialect)
                    # Read header
                    header = next(reader, [])
                    if sampled_mode:
                        # Reservoir sampling of data rows (keep header)
                        k = self.SAMPLE_ROWS
                        reservoir = []
                        n = 0
                        for row in reader:
                            n += 1
                            if len(reservoir) < k:
                                reservoir.append(row)
                            else:
                                # replace with decreasing probability
                                j = np.random.randint(0, n)
                                if j < k:
                                    reservoir[j] = row
                        approx_rows = None
                        # Try to estimate total rows cheaply if file has \n counts
                        # (not precise for quoted newlines, but fine for a hint)
                        try:
                            # Count newline bytes quickly (binary mode)
                            with open(path, "rb") as fb:
                                approx_rows = fb.read().count(b"\n")
                                if approx_rows > 0:
                                    approx_rows -= 1  # minus header (roughly)
                        except Exception:
                            pass
                        return header, reservoir, True, approx_rows
                    else:
                        # Read whole file
                        rows = list(reader)
                        return header, rows, False, len(rows)
            except Exception as e:
                last_err = e
                continue
        # If all encodings failed:
        raise last_err or RuntimeError("Unable to read file with common encodings")

    # ---------- table ----------
    def _populate_table(self, headers, rows):
        self.table.clear()
        cols = max(len(headers), max((len(r) for r in rows), default=0))
        self.table.setColumnCount(cols)
        self.table.setRowCount(len(rows))
        if headers:
            self.table.setHorizontalHeaderLabels([str(h) for h in headers] + [""] * (cols - len(headers)))

        for r, row in enumerate(rows):
            for c in range(cols):
                val = row[c] if c < len(row) else ""
                item = QtWidgets.QTableWidgetItem(str(val))
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()


# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
