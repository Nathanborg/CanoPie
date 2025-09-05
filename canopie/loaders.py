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

from .utils import *

class _LoaderSignals(QtCore.QObject):
    result   = QtCore.pyqtSignal(int, object)  # index, ImageData
    error    = QtCore.pyqtSignal(int, str)     # index, message
    done_one = QtCore.pyqtSignal()             # one task finished


class _ImageLoadRunnable(QtCore.QRunnable):
    """
    Loads one image (and applies aux mods) in a worker thread.
    Emits signals when finished. No UI calls here!
    """
    def __init__(self, tab_ref, filepath, index, project_folder, global_mods, stop_event):
        super().__init__()
        self.setAutoDelete(True)
        self._tab      = tab_ref          # reference to ProjectTab (read-only)
        self._filepath = filepath
        self._index    = index
        self._proj     = project_folder
        self._gmods    = bool(global_mods)
        self._stop     = stop_event
        self.signals   = _LoaderSignals()

    @QtCore.pyqtSlot()
    def run(self):
        # If cancel requested before starting, just count down and go.
        if self._stop.is_set():
            self.signals.done_one.emit()
            return

        try:
            import os

            # Keep OpenCV from oversubscribing threads on its own.
            try:
                import cv2
                cv2.setNumThreads(0)
            except Exception:
                pass

            # 1) Load or fallback
            imgd = self._tab._imagedata_or_fallback(self._filepath)

            if self._stop.is_set():
                self.signals.done_one.emit()
                return

            # 2) Apply aux mods (no UI here)
            #    FIX: pass the actual setting for global_mode
            imgd.image = self._tab.__class__.apply_aux_modifications(
                self._filepath, imgd.image, self._proj, global_mode=self._gmods
            )

            if self._stop.is_set():
                self.signals.done_one.emit()
                return

            # 3) Success
            self.signals.result.emit(self._index, imgd)

        except Exception as e:
            # Make sure 'os' is imported; else this path would fail.
            try:
                base = os.path.basename(self._filepath)
            except Exception:
                base = str(self._filepath)
            self.signals.error.emit(self._index, f"{base}: {e}")

        finally:
            self.signals.done_one.emit()


class ImageProcessor:
    def __init__(self, exiftool_path, images_folder, json_folder, project_folder, batch_size=100, parent_widget=None):
        self.exiftool_path = exiftool_path
        self.images_folder = images_folder
        self.json_folder = json_folder
        self.project_folder = project_folder
        self.batch_size = batch_size
        self.cache_file = os.path.join(self.project_folder, 'gps_cache.json')
        self.jsons_output_folder = os.path.join(self.project_folder, 'jsons')
        self.parent_widget = parent_widget  # <-- so message boxes can use a QWidget parent

        os.makedirs(self.jsons_output_folder, exist_ok=True)

    def _msgbox(self, kind, title, text):
        """kind: 'critical' | 'warning' | 'information'."""
        try:
            from PyQt5 import QtWidgets
            parent = self._parent_widget if isinstance(self._parent_widget, QtWidgets.QWidget) else None
            getattr(QtWidgets.QMessageBox, kind)(parent, title, text)
        except Exception:
            # headless-safe fallback
            print(f"[{kind.upper()}] {title}: {text}")



    def select_unique_roots_per_dir(self, filepaths):
        """
        Return one representative file per (directory, root) pair.
        This ensures dual-folder projects (e.g., RGB/JPG vs Multispectral)
        contribute at least one image per root from *each* folder.

        filepaths: iterable of absolute paths to images.
        """
        import os
        selected = []
        seen = set()

        # Normalize and make deterministic
        for fp in sorted({os.path.normpath(p) for p in filepaths}):
            root = (self.extract_root(os.path.basename(fp)) or "").lower()
            d = os.path.normpath(os.path.dirname(fp)).lower()
            key = (d, root)
            if key not in seen:
                seen.add(key)
                selected.append(fp)
        return selected



    def split_into_batches(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def extract_root(self, filename):
        name, ext = os.path.splitext(filename)
        parts = name.split('_')

        if len(parts) < 2:
            return name  # If there are fewer than 2 parts, the whole name is the root

        if parts[0] == 'IMG':
            # For filenames like IMG_0113_8_radiance, root is IMG_0113
            root = '_'.join(parts[:2])
        else:
            # For filenames like 20221212_103001_750_IR, root is 20221212_103001_750
            root = '_'.join(parts[:3])

        return root

    def select_unique_roots(self, image_files):
        root_dict = {}
        for file in image_files:
            root = self.extract_root(os.path.basename(file))
            if root not in root_dict:
                root_dict[root] = file  # Select the first occurrence per root
        selected_files = list(root_dict.values())
        return selected_files


    def _resolve_exiftool_path(self):
        """
        Return a valid exiftool executable path or None.
        Tries, in order:
          - self.exiftool_path (file or directory)
          - ENV var EXIFTOOL_PATH (file or directory)
          - siblings: exiftool(-k).exe / exiftool.exe if a folder was given
          - PATH via shutil.which
        """
        import shutil

        def _expand_candidate(p):
            # If it's a folder, consider exe names inside it
            if p and os.path.isdir(p):
                names = ["exiftool.exe", "exiftool(-k).exe", "exiftool"]
                return [os.path.join(p, n) for n in names]
            return [p] if p else []

        candidates = []
        # 1) explicit arg
        candidates += _expand_candidate(getattr(self, "exiftool_path", None))
        # 2) env var
        candidates += _expand_candidate(os.environ.get("EXIFTOOL_PATH"))
        # 3) direct file siblings if explicit was a file: also try the folder's standard names
        p = getattr(self, "exiftool_path", None)
        if p and os.path.isfile(p):
            d = os.path.dirname(p)
            candidates += [os.path.join(d, "exiftool.exe"),
                           os.path.join(d, "exiftool(-k).exe"),
                           os.path.join(d, "exiftool")]

        # 4) PATH
        for name in ("exiftool.exe", "exiftool(-k).exe", "exiftool"):
            w = shutil.which(name)
            if w:
                candidates.append(w)

        # Dedup while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)

        for c in uniq:
            if os.path.isfile(c):
                return c
        return None

    def _sanity_check_exiftool(self, exe_path: str) -> bool:
        """Return True if `<exe_path> -ver` works."""
        try:
            out = subprocess.run([exe_path, "-ver"], text=True, capture_output=True, check=True)
            print(f"[ImageProcessor] exiftool -ver -> {out.stdout.strip()}")
            return True
        except Exception as e:
            print(f"[ImageProcessor] exiftool version check failed: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print(e.stderr)
            return False


    def batch_extract_gps_with_exiftool(self, selected_files):
        """
        Extract GPS for a list of files using exiftool (in batches).
        Returns a LIST of dicts: [{'filename': <abs path>, 'latitude': <float>, 'longitude': <float>}, ...]
        """
        import sys

        # Resolve and validate exiftool
        exiftool = self._resolve_exiftool_path()
        print(f"[ImageProcessor] Resolved exiftool -> {exiftool!r}")
        if not exiftool or not self._sanity_check_exiftool(exiftool):
            msg = "ExifTool not found. Set a valid 'exiftool_path' or add it to PATH."
   
            try:
                from PyQt5 import QtWidgets
                if self.parent_widget is not None:
                    QtWidgets.QMessageBox.critical(self.parent_widget, "ExifTool not found", msg)
                else:
                    print(msg, file=sys.stderr)
            except Exception:
                print(msg, file=sys.stderr)
            return []

        # Filter to existing files and normalize paths
        files = [os.path.normpath(p) for p in selected_files if os.path.isfile(p)]
        if not files:
            return []

        batch_size = getattr(self, "batch_size", 80)
        results = []

        # Windows: run without opening a console window
        creationflags = 0
        if sys.platform.startswith("win"):
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except Exception:
                creationflags = 0

        def _fmt_cmd(args):
            return " ".join([f'"{a}"' if " " in a else a for a in args])

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            cmd = [exiftool, "-json", "-n", "-GPSLatitude", "-GPSLongitude", "-filename"] + batch
            print("ExifTool command:\n" + _fmt_cmd(cmd))

            try:
                r = subprocess.run(cmd, text=True, capture_output=True, check=True, creationflags=creationflags)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "").strip()
                print("ExifTool stderr:\n", err, file=sys.stderr)
                continue
            except FileNotFoundError:
                print("ExifTool binary disappeared or is invalid.", file=sys.stderr)
                return []

            try:
                payload = json.loads(r.stdout)
            except json.JSONDecodeError:
                print("Failed to parse ExifTool JSON. First 1000 chars:\n" + r.stdout[:1000], file=sys.stderr)
                continue

            for item in payload:
                src = item.get("SourceFile") or item.get("FileName") or item.get("Filename")
                lat = item.get("GPSLatitude")
                lon = item.get("GPSLongitude")
                if src and (lat is not None) and (lon is not None):
                    results.append({
                        "filename": os.path.normpath(src),
                        "latitude": float(lat),
                        "longitude": float(lon),
                    })

        return results



    def save_cache(self, gps_data):
        """Persist a LIST of {'filename','latitude','longitude'} items."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(gps_data, f, indent=2)


    def load_cache(self):
        """Load cache; accept old dict format and normalize to list of dicts."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # old format: {filename: (lat,lon)}
            out = []
            for fn, pair in data.items():
                try:
                    lat, lon = pair
                except Exception:
                    continue
                out.append({"filename": os.path.normpath(fn), "latitude": float(lat), "longitude": float(lon)})
            return out
        return data if isinstance(data, list) else []


    def build_kdtree(self, gps_data):
        """
        gps_data: list of {'filename','latitude','longitude'} OR old dict format.
        Returns (KDTree, filenames_list, coordinates_list)
        """
        from scipy.spatial import KDTree

        items = []
        if isinstance(gps_data, dict):
            for fn, (lat, lon) in gps_data.items():
                items.append({"filename": os.path.normpath(fn), "latitude": float(lat), "longitude": float(lon)})
        elif isinstance(gps_data, list):
            items = gps_data
        else:
            items = []

        if not items:
            return (None, [], [])

        coordinates = [(it['latitude'], it['longitude']) for it in items]
        filenames = [it['filename'] for it in items]
        tree = KDTree(coordinates)
        return tree, filenames, coordinates


    def _normalize_gps_data(self, data):
        """
        Accepts various historical formats and normalizes to:
          [{'filename': <basename>, 'latitude': float, 'longitude': float}, ...]
        """
        rows = []

        def _add(filename, lat, lon):
            if filename and lat is not None and lon is not None:
                try:
                    rows.append({
                        "filename": os.path.basename(str(filename)),
                        "latitude": float(lat),
                        "longitude": float(lon),
                    })
                except Exception:
                    pass

        # list formats
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lat = item.get("latitude") or item.get("lat") or item.get("GPSLatitude")
                    lon = item.get("longitude") or item.get("lon") or item.get("GPSLongitude")
                    fn  = (item.get("filename") or item.get("file") or
                           os.path.basename(item.get("SourceFile","")) or item.get("name"))
                    _add(fn, lat, lon)
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    fn, lat, lon = item[0], item[1], item[2]
                    _add(fn, lat, lon)

        # dict formats  (old cache: {abs_path: (lat, lon)})
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    lat = v.get("latitude") or v.get("lat")
                    lon = v.get("longitude") or v.get("lon")
                    _add(k, lat, lon)
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    _add(k, v[0], v[1])

        # de-dup by filename, keep first
        seen, uniq = set(), []
        for r in rows:
            fn = r["filename"]
            if fn in seen:
                continue
            seen.add(fn)
            uniq.append(r)
        return uniq


    def find_nearest_images(self, polygon_coords, tree, filenames, coordinates):
        distance, index = tree.query(polygon_coords)
        closest_filename = filenames[index]
        closest_coord = coordinates[index]
        distance_meters = geodesic(polygon_coords, closest_coord).meters
        return closest_filename, distance_meters
    def process_images(self):
        """Scan folder, pick unique roots, extract GPS via exiftool. Returns a LIST of dicts (filename/lat/lon)."""
        import sys
        supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif')

        try:
            all_files = os.listdir(self.images_folder)
        except FileNotFoundError:
            print(f"Error: The folder '{self.images_folder}' does not exist.", file=sys.stderr)
            return []
        except PermissionError:
            print(f"Error: Permission denied for folder '{self.images_folder}'.", file=sys.stderr)
            return []

        print(f"Listing files in: {self.images_folder}")
        for f in all_files:
            print(f"Found file: {f}")

        image_files = [
            os.path.join(self.images_folder, f)
            for f in all_files
            if f.lower().endswith(supported_extensions)
        ]

        print("\nSupported image files found:")
        for file in image_files:
            print(file)

        if not image_files:
            print(f"No image files found in: {self.images_folder}", file=sys.stderr)
            return []

        # Select unique roots
        print("\nSelecting unique images based on roots...")
        selected_files = self.select_unique_roots(image_files)
        print("\nSelected image files with unique roots:")
        for file in selected_files:
            print(file)

        # Extract GPS data from selected files
        gps_data = self.batch_extract_gps_with_exiftool(selected_files)
        if gps_data:
            self.save_cache(gps_data)
            print(f"\nGPS data extracted and saved to {self.cache_file}.")
        else:
            print("No GPS data extracted. Exiting.", file=sys.stderr)
            return []

        return gps_data


    def run(self):
        import sys, os
        supported = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

        # Load cache if any
        gps_data = self.load_cache() if os.path.isfile(self.cache_file) else []

        # Drop stale/invalid rows
        cleaned = []
        for it in gps_data:
            fn = os.path.normpath(it.get('filename',''))
            lat = it.get('latitude'); lon = it.get('longitude')
            if fn and os.path.isfile(fn) and (lat is not None) and (lon is not None):
                cleaned.append({'filename': fn, 'latitude': float(lat), 'longitude': float(lon)})
        if len(cleaned) != len(gps_data):
            gps_data = cleaned
            self.save_cache(gps_data)

        try:
            folder_files = [os.path.join(self.images_folder, f)
                            for f in os.listdir(self.images_folder)
                            if f.lower().endswith(supported)]
        except Exception:
            folder_files = []

        selected = self.select_unique_roots(folder_files)
        cached_set = {os.path.normpath(it.get('filename','')) for it in gps_data}
        missing = [p for p in selected if os.path.normpath(p) not in cached_set]
        if missing:
            added = self.batch_extract_gps_with_exiftool(missing)
            merged = {os.path.normpath(it['filename']): it for it in gps_data if it.get('filename')}
            for it in added:
                merged[os.path.normpath(it['filename'])] = it
            gps_data = list(merged.values())
            self.save_cache(gps_data)

        if not gps_data:
            print("No GPS data found in images.", file=sys.stderr)
            return (None, [], [])

        folders = sorted({os.path.dirname(os.path.normpath(it['filename'])) for it in gps_data})
        print(f"[ImageProcessor] KD-tree input: {len(gps_data)} items from {len(folders)} folder(s).")
        return self.build_kdtree(gps_data)




    def get_gps_coordinates(filepath):
        """
        Extract GPS coordinates from an image file using exifread.
        Returns a dictionary with 'latitude' and 'longitude' in decimal degrees,
        or {'latitude': None, 'longitude': None} if not found.
        """
        try:
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f, details=False)
           
            gps_latitude = tags.get('GPS GPSLatitude')
            gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
            gps_longitude = tags.get('GPS GPSLongitude')
            gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
           
            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = convert_to_degrees(gps_latitude)
                if gps_latitude_ref.values != 'N':
                    lat = -lat
                lon = convert_to_degrees(gps_longitude)
                if gps_longitude_ref.values != 'E':
                    lon = -lon
                return {'latitude': lat, 'longitude': lon}
            else:
                return {'latitude': None, 'longitude': None}
        except Exception as e:
            print(f"Error extracting GPS data from {filepath}: {e}")
            return {'latitude': None, 'longitude': None}

    def convert_to_degrees(value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degrees in float format.
        """
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)

    def extract_root_name(filename):
        parts = filename.split('_')
        if len(parts) >= 2:
            root_name = '_'.join(parts[:2])  # Extract 'IMG_0001' from 'IMG_0001_1.tif'
        else:
            root_name = os.path.splitext(filename)[0]
        return root_name


    def get_exif_data_exiftool_multiple(filepaths):
        """
        Extracts EXIF data from multiple images using exiftool.
        Returns a dictionary mapping absolute filepaths to their EXIF data.
        """
        if not filepaths:
            return {}

        # **Modified: Use a temporary file to pass filepaths to exiftool**
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', newline='') as tmp_file:
                for filepath in filepaths:
                    tmp_file.write(f"{filepath}\n")
                tmp_file_path = tmp_file.name

            # Construct the command using the -@ option
            command = [exiftool_path, '-j', '-@', tmp_file_path]

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            exif_json = json.loads(result.stdout)
            exif_dict = {os.path.abspath(item['SourceFile']): item for item in exif_json}
        except Exception as e:
            print(f"Error extracting EXIF data with exiftool: {e}")
            exif_dict = {}
        finally:
            # **Modified: Clean up the temporary file**
            try:
                os.remove(tmp_file_path)
            except Exception as cleanup_error:
                print(f"Error removing temporary file {tmp_file_path}: {cleanup_error}")

        return exif_dict


    def copy_metadata(source, target, exiftool_path):
        """
        Copies metadata from the source image to the target image using exiftool.
        """
        command = [
            exiftool_path,
            '-TagsFromFile', source,
            '-all:all>all:all',
            '-xmp',
            '-overwrite_original',
            target
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.returncode == 0:
                print(f"Processed: Source: {source}, Target: {target}")
            else:
                print(f"Error processing {source} to {target}: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {source} to {target}: {e.stderr}")
        except Exception as e:
            print(f"Exception occurred while processing {source} to {target}: {e}")
    def extract_tree_number(filename):
        """
        Extracts the tree number or common identifier from the filename.
        Example:
        - Filename: 'tree155_IMG_0888_0_radiance_polygons.tif'
        - Extracted Tree Number: 'tree155'
        """
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        if parts:
            tree_number = parts[0]  # Assuming the tree number is the first part
            return tree_number
        else:
            return base_name  # Fallback to the base name if splitting fails
    def map_images_by_tree_number(self):
        """
        Creates a mapping from tree numbers to image filepaths.
        Returns two dictionaries:
        - tree_to_multispectral: {tree_number: [multispectral_filepaths]}
        - tree_to_thermal_rgb: {tree_number: [thermal_rgb_filepaths]}
        """
        tree_to_multispectral = defaultdict(list)
        tree_to_thermal_rgb = defaultdict(list)
        
        # Map multispectral images
        for root_name, filepaths in self.multispectral_image_data_groups.items():
            for filepath in filepaths:
                filename = os.path.basename(filepath)
                tree_number = extract_tree_number(filename)
                tree_to_multispectral[tree_number].append(filepath)
        
        # Map thermal/RGB images
        for root_name, filepaths in self.thermal_rgb_image_data_groups.items():
            for filepath in filepaths:
                filename = os.path.basename(filepath)
                tree_number = extract_tree_number(filename)
                tree_to_thermal_rgb[tree_number].append(filepath)
        
        return tree_to_multispectral, tree_to_thermal_rgb


    def copy_metadata_from_multiple_sources_parallel(sources, targets, exiftool_path, max_workers=8):
        """
        Copies metadata from multiple source images to target images in parallel.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            args = ((source, target, exiftool_path) for source, target in zip(sources, targets))
            executor.map(lambda p: copy_metadata(*p), args)
    def extract_root_name(filename):
        parts = filename.split('_')
        if len(parts) >= 2:
            root_name = '_'.join(parts[:2])  # Extract 'IMG_0001' from 'IMG_0001_1.tif'
        else:
            root_name = os.path.splitext(filename)[0]
        return root_name


class ImageLoaderWorker(QObject):
    image_loaded = pyqtSignal(object)  # Emitted when an image is loaded
    finished = pyqtSignal()           # Emitted when all images are loaded
    error = pyqtSignal(str)            # Emitted when an error occurs

    def __init__(self, filepaths):
        super().__init__()
        self.filepaths = filepaths
        self._is_running = True

    @pyqtSlot()
    def run(self):
        for filepath in self.filepaths:
            if not self._is_running:
                print("Image loading has been stopped.")
                break
            try:
                 # Load and process the image
                tab = self.parent() if hasattr(self, "parent") else None
                if tab and hasattr(tab, "_imagedata_or_fallback"):
                    image_data = tab._imagedata_or_fallback(filepath)
                else:
                    image_data = ImageData(filepath, mode=self.mod)  # fallback if no tab
            except Exception as e:
                error_message = f"Failed to load image {filepath}: {e}"
                print(error_message)
                self.error.emit(error_message)  # Emit error signal
        self.finished.emit()

    @pyqtSlot()
    def stop(self):
        self._is_running = False

