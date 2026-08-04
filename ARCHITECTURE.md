# CanoPie — Architecture & Developer Guide

> Internal developer reference. For end-user install/usage see `README.md` and
> `CanoPie_Quick_start_user_guide.docx`. This document describes what the program
> does, how the modules connect, and the core concepts you need before editing code.

## What CanoPie is

CanoPie is a **PyQt5 desktop application** for analyzing RGB, multispectral, thermal,
and multitemporal imagery from UAVs and phenocams, aimed at forest/agricultural
**canopy studies**. Instead of building orthomosaics (which resample pixels and distort
radiometry), it works **directly on raw images**: you draw polygons/points on images,
extract per-polygon pixel statistics and vegetation indices, run scikit-learn pixel
classification, and export everything to CSV — while preserving EXIF metadata and
spectral integrity.

Core libraries: **PyQt5** (GUI), **OpenCV** + **tifffile**/**imagecodecs** (image I/O),
**NumPy/SciPy** (math), **shapely** (polygon geometry), **scikit-learn** (classification),
**folium**/**geopy** (GPS mapping), **ExifTool** (external, for metadata).

Entry point: `main.py` → sets Windows AppID + icon → creates `MainWindow`.
Platform target: **Windows 10/11**, Python **3.10**, conda env `canopie-env`.

## Module map

```
main.py
  └── canopie/main_window.py      MainWindow (QMainWindow) — menus, toolbar, tabs, global actions
        └── project_tab.py        ProjectTab (QWidget) — THE central controller (~26k lines)
              ├── image_viewer.py         ImageViewer (QGraphicsView) + EditablePolygonItem / EditablePointItem
              ├── image_editor_dialog.py  ImageEditorDialog (QDialog) — per-image non-destructive edits (.ax files)
              ├── polygon_manager.py      PolygonManager (QDialog) — polygon-group tree, import/visibility
              ├── machine_learning_manager.py  MachineLearningManager + RootOffsetDialog + AnalysisOptionsDialog
              ├── loaders.py              async image loading (QThreadPool/QRunnable), EXIF batch, ImageProcessor
              ├── image_data.py           ImageData — reads one image file, classifies rgb/thermal/multispectral
              ├── performance.py          optional accelerators (NumExpr/Numba/joblib) with NumPy fallback
              ├── utils.py                shared helpers: band math, veg indices, crop/rotate geom, exiftool
              (image_pipeline.py was REMOVED — it was a dead, divergent copy of the .ax replay;
               the live implementations are ProjectTab.apply_aux_modifications + _apply_ax_to_raw)
              ├── phenocam_filter.py      filter images by similarity to a reference image
              ├── similarity_dialog.py    threaded thumbnail chooser for the similarity filter
              ├── patch_dialog.py         dev utility (not core runtime)
              └── log_window.py           in-app log viewer (QtLogHandler → QPlainTextEdit)
```

`canopie/__init__.py` re-exports the public classes and *optionally* the `performance.py`
fast functions (wrapped in try/except — the app runs without NumExpr/Numba installed).

## The control flow (how a session works)

1. **MainWindow** owns a `QTabWidget`. Each tab is a **ProjectTab** (one "project").
   Menus/toolbar actions in MainWindow almost always just delegate to
   `self.get_current_tab().<method>()` — MainWindow is a thin router; the real logic
   lives in ProjectTab.
2. **ProjectTab** loads a folder of images, groups them into **roots**, renders the
   current root into one or more **ImageViewer** panes, and lets the user draw polygons.
3. Drawing/editing happens in **ImageViewer**; polygons live in `ProjectTab.all_polygons`
   and are mirrored to disk. **PolygonManager** is the management UI over that dict.
4. **ImageEditorDialog** applies non-destructive per-image edits saved as **.ax** sidecars.
5. **MachineLearningManager** trains sklearn models from polygon labels and/or applies a
   model during CSV export. Heavy pixel math is offloaded to **performance.py** when available.
6. Export (CSV / EXIF / thumbnails / images) runs in **background QThread workers** so the
   UI stays responsive.

## Core concepts (read this before editing)

### "Root" — the fundamental navigation unit
A **root** is a logical group/batch of related images treated as one synchronized frame.
Examples: 10 consecutive multispectral bands from one capture, or one phenocam timestamp.
- `current_root_index` — 0-based slider position across all roots.
- `root_offset` — a fixed index shift used in **dual-folder mode** to align a multispectral
  root N with the corresponding thermal/RGB root N+offset (two capture sets, one sensor lags).
- `root_coordinates` — `{root_name: {lat, lon}}` from GPS EXIF, drives the folium map.
- `root_id_mapping` / `root_id` — stable numeric IDs per root, persisted so CSV rows are joinable.
Navigation (`next_group`/`prev_group`/`_jump_to_index` → `load_image_group`) swaps which
images are shown and reloads that root's polygons.

### Two loading modes
- **`dual_folder`** — two synchronized folders (e.g. multispectral + thermal/RGB) aligned via `root_offset`.
- **`rgb_only`** — a single folder or hand-picked file set (RGB or stacked bands).
`ImageData` auto-detects type from shape/dtype/filename: 3-channel → RGB (or thermal if
`_IR` suffix + uint16); single-channel → multispectral (or thermal if `_IR`).

### `.ax` files — non-destructive edits
Every image edit (crop, rotate, brightness/contrast via histogram matching, resize,
**band-math expression**, false color, registration matrix, per-pixel classification) is
stored as JSON in a sidecar `<image>.ax` (or in the project folder). Nothing is written
back to the raw image. The canonical application order is
**crop → rotate → histogram/CLAHE → resize → band_expression** (see
`ProjectTab.apply_aux_modifications` for the display path and `ProjectTab._apply_ax_to_raw`
for the export path). `.ax` edits can be applied to one image, a group, or all images.

**Histogram matching is applied EXACTLY ONCE per image**, inside those two functions.
Renderers must never re-apply it: `_render_with_viewer_stretch` receives arrays that have
already been through the replay, and a second pass there used to make the viewer disagree
with CSV/ML export (it also dropped the NoData mask). `ProjectTab._apply_hist_match` is the
single canonical implementation — the editor and `MachineLearningManager` delegate to it.

### Band-math expressions
User expressions like `(b4-b3)/(b4+b3)` (NDVI) or named indices (`GCC=b2/(b1+b2+b3)`) are
evaluated in float32. Two engines:
- `utils.eval_band_expression` / `process_band_expression` — pure-NumPy reference impl,
  supports arithmetic, comparisons, logical ops, and reducers (mean/min/max/std/median/clip/where/…).
- `performance.FastBandMathEngine` — NumExpr-accelerated with a NumPy fallback; used on hot paths.
Bands are referenced `b1..bN` (also uppercase). Prebuilt veg-index helpers live in `utils`
(`calculate_exg/gcc/bcc/gbd/wdx/shd`).

### Polygons
Stored hierarchically: `all_polygons[group_name][image_filepath] = [ {points, image_ref_size, root, class, ...}, ... ]`.
- `points` are raw-image pixel coordinates.
- `image_ref_size` records the image dimensions when drawn, so polygons remap correctly onto
  cropped/resized/rotated versions.
- `group_name` = a label/category (e.g. "canopy_mask", a species); `class` = optional per-polygon label.
On disk they are mirrored as `*_polygons.json` sidecars, and embedded in `project.json`.
There is **no in-memory undo transaction system for polygons** beyond the QUndoStack in
ProjectTab/ImageViewer + file versioning — be careful with bulk operations.

### project.json
Saved at `<project_folder>/project.json`. Key fields:
`project_name`, `mode`, folder paths (`multispectral_folder_path` / `thermal_rgb_folder_path` /
`rgb_folder_path`), `root_offset`, `current_root_index`, `batch_size`,
`multispectral_root_names`, `multispectral_image_data_groups` (root → file list),
`root_coordinates`, `root_mapping`/`root_id_mapping`, and `all_polygons`.
`MainWindow.load_project` requires at least
`{all_polygons, current_root_index, root_offset, root_coordinates}` to accept a folder.

## Threading model

- **Image loading**: `loaders.ImageLoaderWorker` (QObject) + `_ImageLoadRunnable` (QRunnable)
  on a `QThreadPool`; results delivered via Qt signals (`image_loaded`/`finished`/`error`).
  OpenCV thread count is pinned to 0 inside runnables to avoid oversubscription.
- **Export**: CSV/EXIF/thumbnail/image exports each run in dedicated worker threads defined
  inside `project_tab.py` (ExportWorker, ExifExportWorker, ThumbnailExportWorker). They emit
  progress/finished signals; ProjectTab methods often **return `None` when work went to the
  background** and a path when it completed in the foreground — respect this contract when
  wiring new callers (MainWindow relies on it to decide whether to show a popup).
- **Cross-thread safety**: polygons are deep-copied (`_snapshot_polygons_by_group`) before
  export; all Qt widget updates are marshaled to the main thread via signals (see the
  status-bar logging handler in `main_window.py`).

## Performance & caching

ProjectTab keeps several LRU caches: `_pixmap_cache`, `_imgdata_cache`, `_raw_cache`,
`_export_cache`, `_scene_stats_cache`, and ImageEditorDialog reuses the parent's raw-image
cache. `performance.py` adds `FastBandMathEngine`, `FastSklearnPredictor` (batched, multi-core
prediction), `fast_polygon_mask` (OpenCV `fillPoly`), `fast_stats`, and `BatchPolygonProcessor`.
All of it degrades gracefully to NumPy/sklearn if NumExpr/Numba aren't installed
(`numexpr_available()` / `numba_available()`).

## Machine learning

`MachineLearningManager` samples pixels inside polygon ROIs (grouped by `group_name`),
trains one or more sklearn classifiers (RandomForest and friends; XGBoost/LightGBM if
present), tunes with Grid/RandomizedSearchCV, and writes `.pkl` bundles under
`<project>/Machine_learning_models/`. Feature order is fixed:
`red, green, blue, band_4, band_5, …`. A loaded model becomes
`ProjectTab.shared_random_forest_model` (see `MainWindow.load_random_forest_model`) and is
applied per-pixel during export via `ProjectTab._classify_array` / `_apply_sklearn_classification`.
`AnalysisOptionsDialog` collects export stats + index definitions; `RootOffsetDialog` tunes
the dual-folder offset.

## Conventions & gotchas

- **ProjectTab is huge (~26k lines) and central.** Most features are methods on it. Use grep
  to locate a feature by its menu label or action name in `main_window.py`, then follow the
  delegated `current_tab.<method>()` call into `project_tab.py`.
- **MainWindow delegates almost everything** via `hasattr(tab, "…")` guards — new ProjectTab
  methods are auto-reachable if you add the matching action.
- **Windows Unicode paths**: `ImageData` falls back to `np.fromfile` + `cv2.imdecode` for
  non-ASCII paths (plain `cv2.imread` fails on them). Preserve this pattern for any new image read.
- **Logging never writes user files by default**: `basicConfig` is monkey-patched to strip
  FileHandlers; a RotatingFileHandler writes to `%TEMP%/CanoPie/app.log`. `logging.raiseExceptions`
  is disabled for frozen builds.
- **Frozen/Nuitka builds**: `main.py` resolves the logo relative to `__file__` for Nuitka
  onefile; keep asset paths robust to bundling.
- **Not a git repo** as shipped — there's no history to consult; rely on this doc + the code.
- **`from .utils import *`** is used widely; new shared helpers must be added to `utils.__all__`
  to be visible.

## Where to start for common tasks

| Task | Start here |
|------|-----------|
| Add a menu action / global command | `main_window.py` `setup_menu` / `setup_main_toolbar`, then a delegate method on `ProjectTab` |
| Change how images are grouped into roots | `ProjectTab.open_folder` / `load_*_images_from_folder`, root offset logic |
| Change polygon drawing/editing UX | `image_viewer.py` (`EditablePolygonItem`, `start_drawing_with_group_name`, mouse events) |
| Add/adjust an image edit | `image_editor_dialog.py` + `ProjectTab.apply_aux_modifications` / `_apply_ax_to_raw` + `.ax` schema |
| Add a vegetation index / band function | `utils.py` (add helper + expose in `__all__`) and/or `performance.FastBandMathEngine` |
| Change CSV/EXIF/thumbnail export | export worker classes + `save_polygons_to_csv` / `extract_exif_to_csv` / `save_all_thumbnails` in `project_tab.py` |
| ML training/prediction | `machine_learning_manager.py`, `performance.FastSklearnPredictor` |
| Project save/load format | `ProjectTab.save_project`/`load_project`, `MainWindow.load_project`, project.json keys |
| Speed up a hot pixel loop | `performance.py`; guard new deps behind availability checks |
