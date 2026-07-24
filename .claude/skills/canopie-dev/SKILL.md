---
name: canopie-dev
description: Orient and work on the CanoPie codebase — a PyQt5 desktop app for RGB/multispectral/thermal UAV & phenocam canopy image analysis (polygon annotation, band math, vegetation indices, scikit-learn pixel classification, CSV/EXIF/thumbnail export). Use whenever editing, debugging, extending, or explaining any file under CanoPie-main/canopie/ (main_window, project_tab, image_viewer, image_editor_dialog, polygon_manager, machine_learning_manager, loaders, performance, utils, image_pipeline), the project.json / .ax file formats, the "root" navigation model, or anything about how these modules connect.
---

# Working on CanoPie

CanoPie is a **PyQt5 desktop app** for analyzing raw RGB / multispectral / thermal UAV and
phenocam imagery for forest & agriculture **canopy studies**. It avoids orthomosaics and works
directly on raw images: draw polygons/points, compute per-polygon stats and vegetation indices,
run scikit-learn pixel classification, and export CSV/EXIF/thumbnails — preserving metadata and
spectral integrity.

Project root: `C:\ConoPie_main\CanoPie-main`. Entry point: `main.py` → `MainWindow` → `ProjectTab`.
Target: Windows 10/11, Python 3.10 (conda env `canopie-env`). Not a git repo as shipped.

## Before you edit — read the architecture doc

**Always read `ARCHITECTURE.md` in the project root first.** It is the source of truth for module
connections, the "root" concept, the `.ax` edit model, band-math, the `project.json` schema, the
threading/export contracts, and known gotchas. This skill is a pointer + quick index; the doc has
the detail. If your change contradicts the doc, update the doc in the same session.

## Mental model (30-second version)

- **MainWindow** (`main_window.py`) = thin router: menus/toolbar actions delegate to
  `get_current_tab().<method>()`. Real logic lives in **ProjectTab**.
- **ProjectTab** (`project_tab.py`, ~26k lines) = the central controller. State, image grouping
  into **roots**, viewers, polygons (`all_polygons`), export workers, ML hooks, save/load.
- **ImageViewer** (`image_viewer.py`, QGraphicsView) = display + polygon/point drawing/editing.
- **ImageEditorDialog** (`image_editor_dialog.py`) = non-destructive per-image edits stored as
  `.ax` JSON sidecars (crop→rotate→hist→resize→band_expression).
- **PolygonManager** = polygon-group tree/visibility/import UI over `ProjectTab.all_polygons`.
- **MachineLearningManager** = sklearn training from polygon labels + per-pixel prediction.
- **loaders.py** = async image loading (QThreadPool/QRunnable). **performance.py** = optional
  NumExpr/Numba/joblib accelerators with NumPy fallback. **utils.py** = shared band math, veg
  indices, geometry, exiftool helpers (`from .utils import *`; add to `__all__` to export).

## Key concepts you must respect

- **Root**: the navigation unit — a synchronized group of images. `current_root_index`,
  `root_offset` (dual-folder alignment), `root_coordinates` (GPS), `root_id_mapping`.
- **Modes**: `dual_folder` (two aligned folders) vs `rgb_only` (single folder / picked files).
- **.ax files**: non-destructive edits; canonical order crop→rotate→hist/CLAHE→resize→band_expr.
- **Band math**: `b1..bN` refs, float32; `utils.eval_band_expression` (reference) and
  `performance.FastBandMathEngine` (accelerated).
- **Polygons**: `all_polygons[group][filepath] = [{points, image_ref_size, root, class}]`;
  mirrored to `*_polygons.json` and into `project.json`.

## Gotchas (don't regress these)

- **Windows Unicode paths**: image reads fall back to `np.fromfile`+`cv2.imdecode` for non-ASCII
  paths (`cv2.imread` fails). Keep this pattern for any new image loading.
- **Export return contract**: export methods return `None` when work went to a background thread,
  a path when it finished in the foreground. Callers (MainWindow) use this to decide popups.
- **Logging**: `basicConfig` is monkey-patched to strip FileHandlers; logs go to
  `%TEMP%/CanoPie/app.log`. Don't reintroduce user-dir file logging.
- **Thread safety**: deep-copy polygons before export; marshal all Qt widget updates to the main
  thread via signals.
- **Optional deps**: NumExpr/Numba/XGBoost/LightGBM are optional — guard new usage behind
  availability checks, mirroring `performance.py` and `__init__.py`.

## Finding things fast

1. Identify the user-facing label or action name in `main_window.py` (`setup_menu` / toolbar).
2. Follow the delegated `current_tab.<method>()` call into `project_tab.py`.
3. For drawing/display → `image_viewer.py`; for image edits → `image_editor_dialog.py` +
   `image_pipeline.py`; for band functions → `utils.py`; for ML → `machine_learning_manager.py`;
   for speedups → `performance.py`.

See the "Where to start for common tasks" table at the end of `ARCHITECTURE.md`.

## Running

```sh
conda activate canopie-env
pip install -r requirements.txt   # first time; needs ExifTool on PATH for EXIF features
python main.py
```
This is a GUI app — there is no test suite. Verify changes by launching and exercising the
affected workflow; you generally cannot fully validate UI behavior headlessly, so say so if you
can't run it.
