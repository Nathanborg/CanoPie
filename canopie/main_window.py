"""Auto-generated module extracted from original CanoPie code."""


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
        # filter out FileHandler instances
        kwargs['handlers'] = [h for h in handlers if not isinstance(h, logging.FileHandler)]
    return _original_basicConfig(*args, **kwargs)
logging.basicConfig = _no_file_basicConfig

from .project_tab import ProjectTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.status = self.statusBar()

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

        # Add tab-switching shortcuts
        self.add_tab_shortcuts()
        
        # Enable movable tabs
        self.tab_widget.setMovable(True)

    def setup_menu(self):
        # Create a menu bar
        menu_bar = self.menuBar()

        # Create the File menu
        file_menu = menu_bar.addMenu("File")

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
        open_folder_action = QAction("Open multispectral/multiple Folder", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
         
        open_rgb_action = QAction("Open RGB/stack", self)
        open_rgb_action.setShortcut("Ctrl+R")
        open_rgb_action.setStatusTip("Open a folder containing RGB images")
        open_rgb_action.triggered.connect(self.open_rgb_folder)
        file_menu.addAction(open_rgb_action)
        
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
                current_tab.save_all_thumbnails()
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
        load_rf_action.setToolTip("Load the Random Forest model for RGB image classification")
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
            self, "Select Random Forest Model", "", "Pickle Files (*.pkl);;All Files (*)", options=options)
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
                current_tab.save_polygons_to_csv()
                QMessageBox.information(self, "Save Successful", f"CSV saved for project '{current_tab.project_name}'.")
                logging.info(f"CSV saved for project '{current_tab.project_name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save CSV for project '{current_tab.project_name}':\n{e}")
                logging.error(f"Failed to save CSV for project '{current_tab.project_name}': {e}")
        else:
            QMessageBox.warning(self, "No Tab Selected", "Please select a project tab to save CSV.")
            logging.warning("Attempted to save CSV, but no ProjectTab is currently selected.")
            
    def save_all_csv_all_projects(self):
        if self.tab_widget.count() == 0:
            QMessageBox.information(self, "No Projects", "There are no open projects to save.")
            logging.info("Save All CSV triggered, but no projects are open.")
            return

        success_projects = []
        failed_projects = []

        for index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(index)
            if hasattr(tab, 'project_name'):
                try:
                    tab.save_polygons_to_csv()
                    success_projects.append(tab.project_name)
                    logging.info(f"CSV saved for project '{tab.project_name}'.")
                except Exception as e:
                    failed_projects.append((tab.project_name, str(e)))
                    logging.error(f"Failed to save CSV for project '{tab.project_name}': {e}")

        message = ""
        if success_projects:
            message += "CSV saved for the following projects:\n" + "\n".join(success_projects) + "\n\n"
        if failed_projects:
            message += "Failed to save CSV for the following projects:\n"
            for project, error in failed_projects:
                message += f"- {project}: {error}\n"

        if success_projects and not failed_projects:
            QMessageBox.information(self, "Save All Successful", "CSV files saved for all projects successfully.")
        elif success_projects and failed_projects:
            QMessageBox.warning(self, "Partial Save", message)
        elif failed_projects:
            QMessageBox.critical(self, "Save All Failed", message)
        
    
    
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
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if not hasattr(tab, "project_name"):
                continue
            try:
                tab._active_csv_delimiter = unified_delim  # enforce same delimiter
                tab.extract_exif_to_csv()
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


# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
