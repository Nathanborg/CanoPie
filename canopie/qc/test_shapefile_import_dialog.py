import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from canopie.shapefile_import_dialog import ShapefileImportDialog, ShapefileImportProgressDialog

app = QApplication.instance()
if app is None:
    app = QApplication([])

class TestShapefileImportDialog(unittest.TestCase):
    def test_shapefile_import_dialog_defaults(self):
        fields = ["id", "name", "category", "height"]
        dialog = ShapefileImportDialog(fields)
        
        # Check if all fields are populated correctly in combos
        self.assertEqual(dialog.name_combo.count(), 5)  # Auto + 4 fields
        self.assertEqual(dialog.group_combo.count(), 5) # None + 4 fields
        
        # Check if all fields are in the list widget and checked by default
        self.assertEqual(dialog.attr_list.count(), 4)
        for i in range(dialog.attr_list.count()):
            self.assertEqual(dialog.attr_list.item(i).checkState(), Qt.Checked)

        opts = dialog.get_options()
        self.assertIsNone(opts['name_field'])  # 'Auto (Default)'
        self.assertIsNone(opts['group_field']) # 'None'
        self.assertIsNone(opts['simplify_tolerance']) # checkbox unchecked
        self.assertEqual(set(opts['selected_properties']), set(fields))

    def test_shapefile_import_dialog_interactions(self):
        fields = ["id", "name", "category", "height"]
        dialog = ShapefileImportDialog(fields)
        
        # Test simplify toggle
        self.assertFalse(dialog.simplify_tol.isEnabled())
        dialog.simplify_cb.setChecked(True)
        self.assertTrue(dialog.simplify_tol.isEnabled())
        dialog.simplify_tol.setValue(2.5)

        # Test combo selections
        dialog.name_combo.setCurrentText("name")
        dialog.group_combo.setCurrentText("category")

        # Test select none
        dialog.btn_select_none.click()
        opts = dialog.get_options()
        self.assertEqual(opts['selected_properties'], [])
        
        # Select specific attributes manually
        dialog.attr_list.item(0).setCheckState(Qt.Checked)
        
        opts = dialog.get_options()
        self.assertEqual(opts['name_field'], "name")
        self.assertEqual(opts['group_field'], "category")
        self.assertEqual(opts['simplify_tolerance'], 2.5)
        self.assertEqual(opts['selected_properties'], ["id"])

    def test_progress_dialog_cancellation(self):
        dialog = ShapefileImportProgressDialog()
        
        cancel_emitted = [False]
        def on_cancel():
            cancel_emitted[0] = True

        dialog.cancel_requested.connect(on_cancel)
        
        self.assertTrue(dialog.btn_cancel.isEnabled())
        dialog.btn_cancel.click()
        
        self.assertTrue(cancel_emitted[0])
        self.assertFalse(dialog.btn_cancel.isEnabled())
        self.assertEqual(dialog.btn_cancel.text(), "Canceling...")

if __name__ == '__main__':
    unittest.main()
