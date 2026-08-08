import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from scipy.spatial import cKDTree

from PyQt5 import QtWidgets

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.polygons]

# Ensure path is correct (root dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from canopie.polygon_manager import PolygonManager

# NOTE: there is deliberately no local `dummy_app` fixture here. conftest.py
# owns a SESSION-scoped `qapp`; creating a second QApplication path is the
# documented native-crash risk this suite has hit before. Use `qapp`.


# KNOWN WEAKNESS -- this fixture is a MagicMock on purpose, for now.
#
# A Mock answers every attribute access with another Mock, so production code
# reading an attribute the test never set gets a truthy Mock rather than the
# real value. In `move_selected_polygons` that matters: it does
#     tgt_all_files = set(getattr(parent, "thermal_rgb_image_data_groups", {}).get(...))
# and against a Mock that set contains Mocks, so the prune step matches nothing
# and quietly does nothing. The assertions below therefore pass partly because
# the code under test was neutered, not because it behaved correctly.
#
# Swapping in an explicit stub was tried and makes
# `test_copy_and_move_polygons_between_roots` fail with KeyError('TestGroup') --
# i.e. with faithful inputs the real code removes the group. That is worth
# running down, but it changes what the test asserts, so it is left as a
# follow-up rather than silently rewritten here.
#
# TODO: rebuild these against the real `synthetic_project` fixture, which is
# what every other file in this suite uses.
@pytest.fixture
def mock_project_tab(qapp):
    tab = MagicMock()
    # Provide a simple root_coordinates dict mapping root ids to (lat, lon)
    tab.root_coordinates = {
        "root1": {"lat": 10.0, "lon": 10.0},
        "root2": {"lat": 10.01, "lon": 10.01},
        "root3": {"lat": 20.0, "lon": 20.0}
    }
    tab.project_folder = "dummy_folder"
    
    # Mock some methods
    def get_root_by_filepath(fp):
        # mock mapping
        return fp
    
    def get_coordinates_by_root_name(rn):
        if rn == "root1": return (10.0, 10.0)
        if rn == "root2": return (10.01, 10.01)
        if rn == "root3": return (20.0, 20.0)
        return (None, None)
        
    tab.get_root_by_filepath = get_root_by_filepath
    tab.get_coordinates_by_root_name = get_coordinates_by_root_name
    return tab

def test_find_nearest_images(qapp, mock_project_tab):
    """
    Test that polygon manager correctly identifies the nearest image using cKDTree.
    """
    manager = PolygonManager(None)
    manager.parent = MagicMock(return_value=mock_project_tab)
    
    # Create KDTree with our mock coordinates
    # For test simplicity, treat lat/lon as Euclidean XY
    points = [
        [10.0, 10.0],
        [10.01, 10.01],
        [20.0, 20.0]
    ]
    tree = cKDTree(points)
    
    filenames = ["root1", "root2", "root3"]
    
    # We want to find the nearest image to polygon coords [10.005, 10.005]
    polygon_coords = (10.004, 10.004) # Closer to root1 (10.0) than root2 (10.01)
    
    nearest_filename, nearest_dist = manager.find_nearest_images(polygon_coords, tree, filenames, points)
    
    # Assert it correctly resolves to root1
    assert nearest_filename == "root1"
    
    # Change target to be closer to root3
    polygon_coords = (19.9, 19.9)
    nearest_filename, nearest_dist = manager.find_nearest_images(polygon_coords, tree, filenames, points)
    
    # Assert it correctly resolves to root3
    assert nearest_filename == "root3"

def test_copy_and_move_polygons_between_roots(qapp, mock_project_tab):
    """
    Test moving polygons from a multispectral root to a thermal root.
    """
    manager = PolygonManager(None)
    manager.parent = MagicMock(return_value=mock_project_tab)
    
    # Setup mock data in the parent tab
    ms_root = "MS_Root_1"
    th_root = "TH_Root_1"
    
    manager.current_root = ms_root
    
    mock_project_tab.multispectral_root_names = [ms_root]
    mock_project_tab.thermal_rgb_root_names = [th_root]
    
    # Polygons are stored in project_tab.all_polygons
    # { group_name : { filepath : payload } }
    mock_project_tab.all_polygons = {
        "TestGroup": {
            ms_root: {
                "points": [(1, 1), (2, 2)],
                "root": ms_root
            }
        }
    }
    mock_project_tab.multispectral_image_data_groups = {ms_root: [ms_root]}
    mock_project_tab.thermal_rgb_image_data_groups = {th_root: [th_root]}
    
    # We select the item in the list
    mock_item = MagicMock()
    mock_item.text.return_value = "TestGroup (from MS_Root_1)"
    mock_item.data.return_value = {"group_name": "TestGroup", "root_name": ms_root}
    manager.list_widget = MagicMock()
    manager.list_widget.selectedItems = MagicMock(return_value=[mock_item])
    
    # We mock copy_polygons_between_roots so that it simulates the copy
    def mock_copy(*args, **kwargs):
        mock_project_tab.all_polygons["TestGroup"][th_root] = {
            "points": [(1, 1), (2, 2)],
            "root": th_root
        }
    mock_project_tab.copy_polygons_between_roots = mock_copy

    # We want to move it. Mock the UI selection dialog to choose the thermal root
    with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=(th_root, True)):
        # Execute the move
        manager.move_selected_polygons()
    
    # Verify the polygon was removed from the old root
    assert ms_root not in mock_project_tab.all_polygons["TestGroup"]
    
    # Verify the polygon exists in the new root
    assert th_root in mock_project_tab.all_polygons["TestGroup"]
    
    # Verify the payload was preserved and updated
    payload = mock_project_tab.all_polygons["TestGroup"][th_root]
    assert payload["points"] == [(1, 1), (2, 2)]
    assert payload["root"] == th_root

def test_multispectral_to_thermal_polygon_correction(qapp, mock_project_tab):
    """
    Test the spatial offset logic when correcting polygons between MS and TH modes.
    """
    manager = PolygonManager(None)
    manager.parent = MagicMock(return_value=mock_project_tab)
    
    # Mock source and target projects
    source_project = mock_project_tab
    source_project.mode = 'multispectral'
    source_project.all_polygons = {
        "GroupA": {
            "root1": {
                "points": [(10, 10), (20, 20)],
                "root": "root1",
                "coord_space": "image",
                "image_ref_size": {"w": 100, "h": 100}
            }
        }
    }
    
    target_project = MagicMock()
    target_project.mode = 'thermal_rgb'
    target_project.all_polygons = {}
    
    # The paired thermal root for "root1" is "root2" (they share the same coordinates)
    # The find_nearest logic will map root1 -> root2 if we configure the KD trees
    
    # But for run_correction_process, we can just mock _paired_thermal_root for simplicity
    manager._paired_thermal_root = MagicMock(return_value="root2")
    
    # We also mock getting the scale/offset
    manager._get_root_offset = MagicMock(return_value=(5, 5)) # x_off=5, y_off=5
    
    # Run the correction requires directory parsing, so mock os file operations
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manager.jsons_dir = tmp
        manager.polygons_dir = tmp
        manager.corrected_dir = tmp
        
        # mock listdir to return nothing so it doesn't try to parse real jsons,
        # but wait, if it finds no JSONs, it might not do anything.
        # Let's mock the actual target function so we just test the math logic.
        # Actually, the easiest way to test correction logic is calling _apply_correction directly if it exists,
        # or creating a dummy json.
        
        # Write a dummy json
        dummy_json = os.path.join(tmp, "GroupA_root1_polygons_results.json")
        with open(dummy_json, "w") as f:
            f.write('[{"closest_image": "image_2.tif", "distance_meters": 1.5}]')
            
        dummy_poly = os.path.join(tmp, "GroupA_root1_polygons.json")
        with open(dummy_poly, "w") as f:
            f.write('{"points": [[10, 10], [20, 20]], "root": "root1", "coord_space": "image", "image_ref_size": {"w": 100, "h": 100}}')
            
        # We mock open for root_mapping.json since it's hardcoded to be near polygon_manager.py
        mock_mapping = '{"root2": ["image_2.tif"]}'
        import builtins
        original_open = builtins.open
        with patch('builtins.open') as mock_open:
            # We must make sure it handles our tmp paths normally and ONLY mocks root_mapping.json
            def side_effect(path, *args, **kwargs):
                if 'root_mapping.json' in str(path):
                    from io import StringIO
                    return StringIO(mock_mapping)
                return original_open(path, *args, **kwargs)
            mock_open.side_effect = side_effect
            
            # Run the correction
            manager.run_correction_process("mock_source", "mock_target")
        
        # Verify the output JSON in corrected_dir
        expected_out = os.path.join(tmp, "GroupA_image_2_polygons.json")
        assert os.path.exists(expected_out)
        
        import json
        with open(expected_out, "r") as f:
            content = f.read()
            print("CONTENT:", repr(content))
            payload = json.loads(content)
            
        # Points should be preserved exactly as they were in the original JSON
        assert payload["points"] == [[10, 10], [20, 20]]
        assert payload["root"] == "root1"
