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

class ImageData:
    def __init__(self, filepath, mode='dual_folder'):
        self.filepath = filepath
        self.image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Could not read image file: {filepath}")

        self.is_rgb = False
        self.is_thermal = False
        self.is_multispectral = False

        filename = os.path.basename(filepath)
        name_without_ext, ext = os.path.splitext(filename)
        name_upper = name_without_ext.upper()

        if len(self.image.shape) == 3:
            if self.image.shape[2] == 3:
                if self.image.dtype == np.uint16:
                    if name_upper.endswith('_IR'):
                        self.is_thermal = True
                    else:
                        self.is_rgb = True
                else:
                    self.is_rgb = True
        elif len(self.image.shape) == 2:
            if self.image.dtype == np.uint16:
                if name_upper.endswith('_IR'):
                    self.is_thermal = True
                else:
                    self.is_multispectral = True
            else:
                self.is_multispectral = True

        # Check mode before rotating
        if (self.is_rgb or self.is_thermal) and mode == 'dual_folder':
            self.image = self.image
            print(f"Rotated image: {filepath} by 180 degrees.")

