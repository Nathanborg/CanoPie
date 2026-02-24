"""Package initializer for CanoPie modules."""

from .loaders import _LoaderSignals
from .loaders import _ImageLoadRunnable
from .loaders import ImageProcessor
from .loaders import ImageLoaderWorker
from .machine_learning_manager import MachineLearningManager
from .machine_learning_manager import RootOffsetDialog
from .machine_learning_manager import AnalysisOptionsDialog
from .image_viewer import EditablePolygonItem
from .image_viewer import EditablePointItem
from .image_viewer import ImageViewer
from .image_editor_dialog import ImageEditorDialog
from .image_data import ImageData
from .polygon_manager import PolygonManager
from .project_tab import ProjectTab
from .main_window import MainWindow

# High-performance computation module (optional)
try:
    from .performance import (
        FastBandMathEngine,
        FastSklearnPredictor,
        fast_safe_predict,
        fast_eval_band_expression,
        fast_polygon_mask,
        fast_extract_roi_pixels,
        fast_stats,
        get_band_math_engine,
        BatchPolygonProcessor,
        numexpr_available,
        numba_available,
    )
    _PERF_EXPORTS = [
        "FastBandMathEngine", "FastSklearnPredictor", "fast_safe_predict",
        "fast_eval_band_expression", "fast_polygon_mask", "fast_extract_roi_pixels",
        "fast_stats", "get_band_math_engine", "BatchPolygonProcessor",
        "numexpr_available", "numba_available"
    ]
except ImportError:
    _PERF_EXPORTS = []

__all__ = [
    "_LoaderSignals", "_ImageLoadRunnable", "ImageProcessor", "ImageLoaderWorker",
    "MachineLearningManager", "RootOffsetDialog", "AnalysisOptionsDialog",
    "EditablePolygonItem", "EditablePointItem", "ImageViewer",
    "ImageEditorDialog", "ImageData", "PolygonManager", "ProjectTab", "MainWindow"
] + _PERF_EXPORTS