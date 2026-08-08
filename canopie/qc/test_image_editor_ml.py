import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from PyQt5 import QtWidgets

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.editor, pytest.mark.ml]

# Ensure path is correct (root dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from canopie.image_editor_dialog import ImageEditorDialog

# Define mock classes for Scikit-Learn models
class MockDummyClassifier:
    def __init__(self, strategy="constant", constant=0):
        self.strategy = strategy
        self.constant = constant
        
    def predict(self, X):
        return np.full(X.shape[0], self.constant)

class MockModel:
    def __init__(self):
        self.classes_ = ["Class A", "Class B"]
    
    def predict(self, X):
        # Always predict class 0
        import numpy as np
        return np.zeros(len(X), dtype=int)

def MockModelBundle():
    return {"model": MockModel(), "feature_names": ["band_1"]}

# NOTE: no local `dummy_app` fixture -- conftest.py owns a SESSION-scoped
# `qapp`. A second QApplication path is the documented native-crash risk in
# this suite, so tests take `qapp` instead.


class DummyParentTab:
    pass

@pytest.fixture
def mock_parent_tab():
    parent = DummyParentTab()
    parent.random_forest_model = MockModelBundle()
    return parent

def test_ml_classification_mapping_on_synthetic_image(qapp, mock_parent_tab):
    """
    Test that ImageEditorDialog correctly resolves the ML model bundle and generates
    a 2D classification map when classification is enabled in the mod dictionary.
    """
    # Create a 3x3 synthetic 2-band image
    artificial_image = np.array([
        [[10, 20], [30, 40], [50, 60]],
        [[70, 80], [90, 10], [20, 30]],
        [[40, 50], [60, 70], [80, 90]]
    ], dtype=np.uint8)

    dialog = ImageEditorDialog(parent=None, image_data=artificial_image, image_filepath="dummy.tif")
    
    # Configure modifications to enable classification
    dialog.modifications = {
        'classification': {'enabled': True}
    }
    
    # Mock _resolve_sklearn_model_bundle to return our dummy bundle
    def mock_resolve_bundle():
        return mock_parent_tab.random_forest_model
    dialog._resolve_sklearn_model_bundle = mock_resolve_bundle
    
    # We must explicitly set the check box for classification
    dialog.use_sklearn_checkbox = MagicMock()
    dialog.use_sklearn_checkbox.isChecked.return_value = True

    with patch.object(dialog, 'parent', return_value=mock_parent_tab):
        dialog.run_sklearn_classification()
    
    # Verify the result is generated and stored correctly
    assert hasattr(dialog, '_classification_result')
    res = dialog._classification_result
    
    # The output should be a 2D map of the same spatial shape as the input image
    assert res is not None
    assert res.shape == (3, 3)
    
    # Since our MockModel always returns 0, the map should be all 0s
    assert np.all(res == 0)

def test_hist_match_and_crop_consistency(qapp, mock_parent_tab):
    """
    Test that cropping and histogram matching work together without introducing NaNs
    and output the correct final cropped shape.
    """
    artificial_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dialog = ImageEditorDialog(parent=None, image_data=artificial_image, image_filepath="dummy2.tif")
    
    # Create a reference image (e.g. thermal or MS pair)
    ref_image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    # Set up crop rect and histogram match parameters
    dialog.modifications = {
        'crop_rect': {'x': 20, 'y': 30, 'width': 40, 'height': 50},
        'histogram_match': True
    }

    # Mock fetching the reference image
    def mock_get_reference_image(mod_type):
        if mod_type == "histogram_match":
            return ref_image
        return None
        
    dialog._get_reference_image = mock_get_reference_image
    
    with patch.object(dialog, 'parent', return_value=mock_parent_tab):
        # Apply modifications
        dialog.apply_modifications(dialog.modifications)
    
    # The returned image from display_image_data should be cropped
    final_img = dialog.display_image_data
    assert final_img is not None
    
    # The shape should be exactly height=50, width=40, 3 bands
    assert final_img.shape == (50, 40, 3)
    
    # No NaNs should be produced by histogram matching
    assert not np.isnan(final_img).any()

def test_model_class_attribute_sync(qapp, mock_parent_tab):
    """
    Test that when saving modifications, if classification is enabled, the model bundle
    is correctly hoisted to the class attribute for background rendering to access.
    """
    artificial_image = np.zeros((10, 10, 1))
    dialog = ImageEditorDialog(parent=None, image_data=artificial_image, image_filepath="dummy3.tif")
    
    dialog.modifications = {
        'classification': {'enabled': True}
    }
    
    def mock_resolve_bundle():
        return mock_parent_tab.random_forest_model
    dialog._resolve_sklearn_model_bundle = mock_resolve_bundle
    
    # Mock the accept() slot to prevent UI closing logic issues in headless tests
    dialog.accept = MagicMock()
    
    with patch.object(dialog, 'parent', return_value=mock_parent_tab):
        dialog.save_modifications_to_file()
    
    # Check if the class attribute on parent's type was updated
    parent_type = type(mock_parent_tab)
    assert hasattr(parent_type, "shared_random_forest_model")
    assert parent_type.shared_random_forest_model is mock_parent_tab.random_forest_model
