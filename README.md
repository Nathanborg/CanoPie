# CanoPie Split

This repository contains the CanoPie codebase split into individual modules.  Each class from the original
monolithic script has been extracted into its own file under the `canopie` package.  The main entry
point is `main.py`.  Logging has been configured globally to avoid writing any log files to disk.

## Running the application

Install dependencies listed in `requirements.txt` and run:

```bash
python main.py
```

## Structure

* `main_window.py` – contains `MainWindow` class.
* `project_tab.py` – contains `ProjectTab` class with imports of helpers and managers.
* `polygon_manager.py` – contains `PolygonManager` class.
* `machine_learning_manager.py` – contains machine learning dialogs.
* `image_viewer.py` – contains image viewing and editable polygon items.
* `image_editor_dialog.py` – contains image editing dialog.
* `image_data.py` – contains `ImageData` class.
* `loaders.py` – contains loading worker classes.
* `utils.py` – re-exports helper functions from the original script.

## Logging

A global override of `logging.basicConfig` is installed in each module to remove any `filename` parameter and
filter out `FileHandler` instances.  This ensures that the program never writes log files to the working
folder.
