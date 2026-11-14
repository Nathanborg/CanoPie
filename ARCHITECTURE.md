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
              │     ├── canopy_tracer.py  Smart Canopy brush: strokes on a crown -> its outline (graph cut, no Qt)
              │     └── canopy_auto.py    Auto Canopy engine: offline only, not wired into the app (no Qt)
              ├── image_editor_dialog.py  ImageEditorDialog (QDialog) — per-image non-destructive edits (.ax files)
              ├── polygon_manager.py      PolygonManager (QDialog) — polygon-group tree, import/visibility
              │     └── annotator.py      Annotator (PolygonManager subclass) — custom per-polygon properties
              ├── polygon_bulk.py         bulk crown copies (replicate, copy to roots, import) off the GUI thread
              ├── machine_learning_manager.py  MachineLearningManager + RootOffsetDialog + AnalysisOptionsDialog
              ├── loaders.py              async image loading (QThreadPool/QRunnable), EXIF batch, ImageProcessor
              ├── image_data.py           ImageData — reads one image file, classifies rgb/thermal/multispectral
              ├── raster_reader.py        tiled/windowed TIFF + COG reads, previews, the decoded-tile cache (no Qt)
              │     └── remote_raster.py  RangeFile + .cogref pointers: a COG over HTTP Range requests (no Qt)
              ├── remote_images_dialog.py File > Add Remote Images (URLs)…: check links, write pointers, open as roots
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
   The **Annotator** (`annotator.py`, Modules > Annotator, Ctrl+Shift+P) adds user-defined
   numeric/categorical properties scoped to the project, a root or an image (optionally one
   label). VALUES live in each polygon's own `properties` under the property's name (number,
   choice, tags 'a;b', null = N/A), so the viewer's Edit properties dialog, the sidecar JSON,
   shapefile DBF and `prop_<name>` CSV columns all carry them. `project.json["annotator"]`
   holds the definitions plus a value INDEX (a cache) so completion stats never page in a
   LazyPolygonRecord; any read of a record refreshes it. Applicability is computed on demand,
   so a property covers existing and later polygons. "Include in exports" off takes the
   property out of each export's copy (attach_export_values), never off the polygon.
   The Polygon Properties panel is the left half of the Annotator window, which moves off the
   polygon a list click zooms to. Per property, "Draw" turns on a plain brush (PatchBrush, an
   event filter on that viewer's viewport): painted areas are clipped to the crown
   (PatchCanvas, OpenCV mask) and saved as polygons in record["annotator_patches"][name] --
   written to the sidecar by _sidecar_payload / save_polygons_to_json and lazy-loaded like
   `properties` (polygon_lod._EXACT_KEYS). Finishing can set a numeric value to the % drawn.
   While the Annotator is open, ←/→/A/D mean previous/next polygon to fill (the project
   window's root-stepping shortcuts are borrowed and restored on close), the marked viewer's
   strip is red while anything is pending and green when complete, and "From previous image"
   copies a property's patches from the same group's latest earlier image (transfer_patches:
   source crown bbox -> target crown bbox).
   Patches of different properties never overlap (PatchCanvas.set_blocked: the crown's other
   properties' paint is off limits). With the Annotator open a click on a locked crown only
   opens it; a drag past startDragDistance asks to unlock (ImageViewer._locked_drag).
   **Bulk copies** (Replicate to all viewers, Copy to next / previous / all roots, shapefile
   import) go through `polygon_bulk.py`: a list of jobs (one crown, one image) run by
   `BulkPolygonWorker` on a QThread, then `ProjectTab._apply_bulk_results` on the GUI thread
   (lazy records, `_draw_groups_chunked`, one `AddPolygonsCommand` undo step). A copy is placed
   by image-size ratio -- the same FRACTION of the frame, which is how records are stored -- so
   its sidecar is the source's bytes and is made by HARD LINK (~0.2 ms; a written file costs
   1-2 ms on Windows). Every sidecar writer must REPLACE the file (`polygon_lod.dump_sidecar`:
   temp file + os.replace), never write into it, or a linked crown changes on both images. A
   linked record pins its own `root` (`LazyPolygonRecord(pinned=...)`, kept in the overview).
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
**crop → rotate → histogram/CLAHE → resize → band_expression → appended_bands** (see
`ProjectTab.apply_aux_modifications` for the display path and `ProjectTab._apply_ax_to_raw`
for the export path — both are replayed by `MachineLearningManager` too via `_apply_hist_local`).
`.ax` edits can be applied to one image, a group, or all images.

**`appended_bands`**: the Image Editor's Append (+) button turns a live classification result
or boolean/numeric band expression into a *permanent extra band*, stored as
`{"type": "classification"|"boolean_expression"|"band_expression", "index", "expression"?,
"model_name"?, "label_names"?}`. Only the reference/expression is persisted — the pixel data is
regenerated on every replay. A classification entry therefore needs the ORIGINAL sklearn model
back in memory (`ProjectTab._get_sklearn_bundle()` → `self.random_forest_model` or the
class-shared `ProjectTab.shared_random_forest_model`) to re-run prediction; that state is
runtime-only and never persisted, so restarting the app (or never re-loading the `.pkl` this
session) makes every classification-type appended band silently vanish from every consumer
(CSV, ML training, image export) unless the caller checks `_record_appended_band_error`'s
counter — see the channel-order/silent-failure gotchas above.

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

**`b1` IS RED, everywhere, for every loader.** `eval_band_expression` itself is purely
positional (`b1` = array channel 0), so every caller must hand it an array whose first three
channels are already R,G,B. The one way to do that is
`utils.to_rgb_band_order(arr, channel_order)` driven by the loader's real provenance
(`ImageData.channel_order` / `_Lite.channel_order` / `ProjectTab._last_export_channel_order` /
`ImageEditorDialog._editor_channel_order` / `MachineLearningManager._current_channel_order()`).
Two idioms that used to do this and must never come back: an `if C == 3: swap` shape guess
(right for cv2, silently backwards for tifffile), and rewriting the expression STRING via a
`__B1_TEMP__` placeholder (only ever handled b1/b3, so `b4`+ on a wider stack addressed the
wrong plane). Bands 4..N are NEVER reordered — their index is their identity.
`utils.swap_b1_b3_tokens` survives for a hypothetical one-shot migration of stored expressions
only and must not appear on any live pixel path. Pinned by
`canopie/qc/test_band_convention_contract.py`, which drives the viewer replay, the export
replay, `process_polygon`, the Image Editor and the ML manager on the SAME image delivered
under BOTH provenances and requires one number.

### Polygons
Stored hierarchically: `all_polygons[group_name][image_filepath] = [ {points, image_ref_size, root, class, ...}, ... ]`.
- `points` are raw-image pixel coordinates.
- `image_ref_size` records the image dimensions when drawn, so polygons remap correctly onto
  cropped/resized/rotated versions.
- `group_name` = a label/category (e.g. "canopy_mask", a species); `class` = optional per-polygon label.
On disk they are mirrored as `*_polygons.json` sidecars, and embedded in `project.json`.

**Drawing modes** (`ProjectTab.toggle_drawing_mode`, Alt+D): polygon, point, rectangle, circle,
smart_canopy, random_points. Every non-point mode MUST be listed in
`ImageViewer._is_point_drawing_mode`, or it is silently treated as point mode. **Smart Canopy**
is a brush (as in RoosterFly): Shift+click/drag starts a session, later clicks/drags add strokes,
Ctrl+drag erases, `[`/`]` size the brush, `-`/`+` (or Alt+scroll) set how far the crown may grow
past the paint (`CanopyParams.grow` x brush radius — the only reach, so it never runs away),
Backspace undoes a stroke, Esc cancels. Defaults are optimised on BCI crowns for ~10 clicks inside
a crown (brush ~1/6 of the crown radius, grow 4: F1 0.84 held out). Existing polygons and erased
bands win over paint — crowns never overlap. Strokes are kept in scene
coordinates (`_sc_strokes`) and after each one `_trace_smart_canopy` re-selects the crown from all
of them: it reads the *displayed* pixels — the sharp COG viewport tile when it is shown, else the
pixmap — and calls `canopy_tracer.select_canopy` (a GrabCut graph cut seeded by the paint; erased
bands, polygons already in the viewer and everything beyond a reach around the paint are hard
background). The yellow preview is `_sc_sel_item`. Only **Enter** (or right-click) commits: the
outline goes through the ordinary `add_polygon_to_scene` / `polygon_drawn` path, so ProjectTab's
name dialog opens then, and it is saved, undone and exported like a hand-drawn polygon.

**Vertex editing** drags handles (at most 100 per outline, `start_vertex_editing`); a drag pulls every
vertex within `VERTEX_SOFT_RADIUS_SCREEN_PX` (`[`/`]` while editing) along with a smooth falloff —
`_vertex_drag_begin` fixes the weights at the press, `_apply_vertex_update` moves the vertices in
place (`EditablePolygonItem._set_vertex_points`) and the handles with them; Shift+drag moves one
vertex. **Grow/shrink** (`+`/`-`, `ProjectTab.resize_selected_polygons`) buffers the record's exact
points in the FULL-raster basis, maps them with `image_to_scene_coords` (never via the preview size —
on a COG that double-scaled the outline into the image corner) and records the change with
`ModifyPolygonCommand`, so it is undoable and keeps the record's other fields;
`ModifyPolygonCommand._apply_state` loads a lazy crown before writing its points and drops its stale
`lod`/`display_points`.

**Auto Canopy** (`canopy_auto.py`, every crown in a region found automatically) is **not wired into
the app** — it was removed from the toolbar after testing (held-out BCI crown F1 ~0.57–0.60 against
0.84 for Smart Canopy by hand). The engine, its tests (`canopie/qc/test_canopy_auto.py`) and the
optimiser (`scripts/optimise_auto_canopy.py`: dense blocks from a project's polygons or a crown
shapefile, `--auto-align` to shift a crown map drawn on another orthomosaic onto this one, random
search with a held-out split) remain for offline work. Diagnostic on BCI (held-out crown F1): found
centres 0.57, true centres 0.66, true centres and sizes 0.77; `SIZE_MODEL_BCI` (ridge on
`size_features`) predicts each crown's size well (correlation 0.75), but finding the centres is the
limit.

**Point items carry SUB-PIXEL precision, and must keep it.** `EditablePolygonItem.polygon`
holds genuine scene coordinates; `EditablePointItem.points` is pixmap-LOCAL instead, so a true
scene position is `item.mapToScene(QPointF(*item._scene_xy(p)))` -- BOTH steps (`_scene_xy`
converts pixmap-local -> item-local; `mapToScene` adds the `pos()` Qt's default drag
accumulated). `ImageViewer._scene_to_pix_local` used to `floor()` that storage to a whole
PIXMAP pixel. The saved coordinate is a float in the FULL-RASTER basis, so one pixmap pixel is
one raster pixel only at 1:1 -- under an `.ax` resize (measured: a 2064x1544 capture displayed
at 50x33) it is 41 x 47 RASTER pixels, and the reload quantised the user's drop position away:
24/24 dragged points moved 19-61 raster pixels on the next refresh, and a freshly drawn point
could jump a whole pixmap pixel because the load-side round trip (`stored * pm_w / ref_w`)
lands a hair below the integer it started from. Pinned by
`canopie/qc/test_point_drag_refresh_stability.py`.

**The point PREVIEW and the committed point must come from one builder.** There are two
point-preview call sites in `ImageViewer` -- `update_temp_drawing()` (after each click) and
`mouseMoveEvent()`'s hover branch (the marker that follows the cursor) -- and they used to
carry two independent copies of the same math. Making the finalize side sub-pixel updated only
the first; the hover copy kept `int(math.floor(...))`, so the marker sat on the top-left corner
of the containing PIXMAP pixel while the committed point kept its true fractional position, and
it visibly jumped the instant the point was committed ("the temporary pixel point and the final
drawn point not matching"). Measured with the real handlers on the reported image
(IMG_0402_stacked.tif, 2064x1544): 0.93 scene units = **16.6 screen px = 38 raw raster px**
under an `.ax` resize to 50x33, 0.4 screen px with no `.ax`. Both preview sites now call
`ImageViewer._point_marker_path`, which must stay the single source of that geometry --
it has to place the marker exactly where `EditablePointItem.paint()` will draw it
(`QRectF(*_scene_xy(p), 1, 1)`). Note the temp item and the point item are NOT in different
coordinate spaces (both are scene children at pos (0,0), identity transform) -- that was
measured and refuted, and is pinned so a future reparent cannot reintroduce the symptom.

**An `EditablePointItem`'s grab region is centred ON the point.** `shape()` uses
`addEllipse(QPointF(sx, sy), POINT_GRAB_RADIUS, POINT_GRAB_RADIUS)`. The four-argument
`addEllipse(sx, sy, 6, 6)` it replaces INSCRIBES the ellipse in that rect, so the region was
centred on `(sx + 3, sy + 3)` and the point sat outside its own grab region -- measured, for a
point at (40, 25) the region spanned x 40.1..45.9 / y 25.1..30.9 and `scene.itemAt(exact point)`
returned the pixmap item. That was "to move the points the pan has to be further away from the
point and not exact". Both copies of the shape geometry (the cached one in
`_invalidate_point_cache` and the fallback inside `shape()`) must stay identical. Pinned by
`canopie/qc/test_point_preview_matches_final.py`.

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
  inside `project_tab.py` (`ExportWorker` for CSV, `ExifExportWorker`, `ThumbnailExportWorker`,
  `ProjectImagesExportWorker` for "Export Project Images"). They emit progress/finished signals
  surfaced by `ExportManagerDialog` (a singleton, non-modal, multi-export status panel);
  ProjectTab methods often **return `None` when work went to the background** and a path when it
  completed in the foreground — respect this contract when wiring new callers (MainWindow relies
  on it to decide whether to show a popup). "Export Project Images" prompts via
  `export_images_options_dialog.ExportImagesOptionsDialog` (format/band-order/EXIF/background) —
  the same options dict must reach BOTH `ProjectImagesExportWorker.__init__` (background) and the
  inline `export_project_images` foreground loop, since they don't share code.
  A QThread worker returning `finished(path, ok, message)` should append any
  `_appended_band_error_count`/EXIF-copy/BigTIFF-fallback notes to `message` — that string is
  literally what `ExportManagerDialog` displays, so it's the only place those notes reach the user.
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

### Which pixels are eligible for classification (inference must match training)

`train_models` never fits on every pixel inside a labelled polygon. It DROPS a sampled pixel
when any value in its window is non-finite, when any value in its window equals a declared
numeric NoData literal, or (for `window_size > 1`) when its window would leave the image.
`process_polygon`'s CSV `Class N %` columns exclude exactly those pixels too.

Every inference path must therefore exclude them as well — a median-imputed feature, an
edge-padded neighbour and a NoData sentinel are all rows the model was never trained on, and
predicting from one makes the screen and the CSV describe different pixel populations for the
same polygon. The single definition lives in `utils`:
`classification_invalid_rows` / `mark_ineligible_rows` / `window_edge_rows` /
`nodata_match_tolerance` (the tolerance ladder is copied from
`MachineLearningManager._build_training_row`, which is what training actually uses).

Mechanics: both `_make_feature_stack_for_model` copies (ProjectTab's and
ImageEditorDialog's) set ineligible rows to **NaN** instead of imputing them — NaN survives
`iter_tiled_feature_matrices`' row slicing and `apply_normalization`, and makes a caller that
forgets to filter fail loudly in `model.predict` rather than silently classify. Callers filter
with `np.all(np.isfinite(X), axis=1)` and write `utils.UNCLASSIFIED_CLASS_INDEX` (-1) for the
rest; `_colorize_class_map` renders a negative index as 0. The hand-built classification blocks
in `_apply_ax_to_raw` / `apply_aux_modifications` keep their own `nodata_fill` convention for
the exported label band but apply the same three rules. Computing the window-edge mask INSIDE
the builder is correct under tiling: `iter_tiled_feature_matrices` supplies halo rows and
discards exactly them, so only the image's true boundary is ever marked.

**Known remaining gap** (not fixed, measured): at `window_size > 1` with a *numeric* NoData
literal, `process_polygon` excludes a pixel only when its OWN centre is NoData (plus
non-finite anywhere in the window), while training and the feature builders exclude a pixel
whose window merely TOUCHES NoData. `_apply_ax_to_raw` zeroes NoData pixels before
`process_polygon` sees them, so process_polygon cannot detect them by value; matching training
there needs its NoData mask dilated by `half`. Covered by
`canopie/qc/test_classification_pixel_eligibility.py`, which deliberately tests NoData at
`window_size == 1` and the edge/non-finite rules at `window_size == 3`.

### Training augmentation + feature normalization

`MachineLearningManager.train_models` optionally augments training pixels and/or
normalizes features before fitting, via `MLAugmentationOptionsDialog`
(`ml_augmentation_options_dialog.py`), shown as one more wizard step right before
feature names are built. Two things to know before touching any of this:

- **Two parallel row collections, not one `is_augmented` mask.**
  `X_rows_report`/`y_rows_report` are pristine and are the *only* thing the
  holdout-metrics `train_test_split` calls ever see; `X_rows_train`/`y_rows_train`
  (aliased to the same list objects when augmentation is off — zero extra cost in
  the common case) are what actually gets `.fit()`ed. `ml_augmentation.assemble_pixel_rows`
  is the policy function deciding what goes into the train collection per pixel
  ("add" appends every successful augmented variant; "replace" always emits exactly
  one row, falling back to the pristine row if every variant failed). Never merge
  these into one array + a boolean column — Replace mode can legitimately touch
  every sampled pixel, which would shrink or exhaust an `is_augmented`-filtered
  holdout pool.
- **Augment the raw image, before bands/expressions are derived from it.**
  `_channels_in_export_order` returns copies and band-expression evaluation reads
  the raw image directly — patching only the extracted channels would leave
  expressions (e.g. NDVI) computed from unaugmented pixels while raw bands look
  augmented. `MachineLearningManager._select_bands_and_expressions` is the shared
  helper both the pristine pass and every augmented variant call, so this can't
  drift.

A trained bundle can carry a `"normalization"` key —
`{"method": "none"|"l2"|"zscore", "mean": [...]|None, "scale": [...]|None}`, built by
`utils.fit_normalization` and applied by `utils.apply_normalization` — recording how
features were transformed before `.fit()`. **Every place that later loads a bundle and
calls `model.predict()` on a freshly-built feature matrix must apply this same
transform**, since `_make_feature_stack_for_model` is *not* a universal choke point
(several sites hand-build their own `X`). As of this writing there are six such sites,
each taking a `normalization_cfg` parameter or reading `bundle.get("normalization")`
directly: `project_tab.py`'s and `image_editor_dialog.py`'s own
`_make_feature_stack_for_model` (covers `_apply_sklearn_classification[_with_indices]`
and both `.ax` "appended classification band" replay blocks), the hand-built main
classification block inside both `_apply_ax_to_raw` and `apply_aux_modifications`,
`ProjectTab._safe_predict` (the choke point for `process_polygon`'s three independent
feature builders), and `MachineLearningManager._predict_class_map_from_bundle`. Adding
a 7th classification call site anywhere means adding it to this list too —
`canopie/qc/test_ml_normalization_end_to_end.py` only covers three of the six
end-to-end; check the others by inspection when you add a new one.

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
- **This IS a git repo** (`origin` → `github.com/Nathanborg/CanoPie`, branch `main`) — `git log`/
  `git blame` are real history here, use them. (Stale note from before the repo was fixed onto
  the right root — see the skill's "Repo setup" section for the incident.)
- **`from .utils import *`** is used widely; new shared helpers must be added to `utils.__all__`
  to be visible.
- **Channel order is provenance, not a format constant.** cv2 (`imread`/`imdecode`) always
  decodes to BGR; tifffile always returns the file's native order (never BGR). Code that treats
  "3-channel array" as unconditionally one or the other has repeatedly caused real R/B swaps —
  in histogram-match reference application, in `_channels_in_export_order`, in Export Project
  Images' save step, in thumbnail/segmentation-mask rendering. The fix pattern every time is the
  same: track the true order at the point of loading (`channel_order` on `ImageData`,
  `_last_export_channel_order` on `ProjectTab`, the `(img, channel_order)` tuple
  `ProjectImagesExportWorker._load_image_simple` returns) and branch on it, never assume.
- **A `try/except` that only `logging.warning()`s hides real, silent failures.** Multiple bugs
  this project has shipped were a feature working perfectly *when its precondition held* (a model
  loaded, an expression naming existing bands) and silently doing nothing otherwise, with only a
  log line nobody reads. The established fix pattern (`_record_rf_prediction_error`,
  `_record_appended_band_error`): track a count + a few sample messages on `self`, reset it at
  the start of each export run, and append a `WARNING: ...` note to that run's completion
  message. Apply this pattern to any new swallowed-exception path that affects export output.
- **Don't force `QFileDialog.DontUseNativeDialog`.** Every folder/file picker in the app
  (`main_window.py`'s Load Project/Load All Projects/model picker, `project_tab.py`'s Open
  Folder/Open RGB Folder/Save/Load Project/thumbnail-output/change-folder pickers,
  `machine_learning_manager.py`'s export-folder picker — 13 sites total) used to set this flag
  unconditionally, with no comment or documented reason at 12 of the 13 sites. Forcing Qt's own
  non-native dialog widget means every picker depends on Qt's `QFileSystemModel`/
  `QFileIconProvider` working correctly on this machine instead of the OS's native (COM
  `IFileDialog`-backed) picker; when that Qt-side path misbehaves the symptom is exactly "folders
  don't show up, and Cancel hangs the whole app" — reported across effectively every picker in the
  app at once, on plain local folders as well as cloud-synced ones, so it isn't simply "slow
  enumeration of a big/remote directory." **This was already diagnosed once before**, at exactly
  one of the 13 sites (`image_editor_dialog.py`'s "Select scikit-learn model bundle" picker),
  committed 2026-02-24 (`f93072bb`) with the comment "FIX: Removed DontUseNativeDialog to prevent
  Windows freezing on drive scan" — but the same fix was never carried over to the other 13 sites
  that shipped in that same commit, which is why the bug kept recurring. Removed the flag from all
  13 (kept the now-inert `options = QFileDialog.Options()` line and the `options=options` kwarg as
  harmless no-ops so the diff stays minimal) so every picker uses the native OS dialog. If a
  *future* picker genuinely needs Qt's own non-native dialog (custom sidebar, multi-selection UI
  beyond what `getExistingDirectory`/`getOpenFileName` support), document the reason right next to
  the flag — don't reintroduce it silently, and check this list isn't missing a new site.

### Remote COGs (`.cogref`)

A `.cogref` is a small JSON pointer (`{"canopie_cogref": 1, "url": ..., "size": ...}`, optional
`auth`) that stands in for a remote Cloud-Optimized GeoTIFF. The pointer's PATH is the image's
identity everywhere (caches, `.ax`, polygon sidecars, project.json), so `ortho.cogref` shares
`ortho.ax` with a local `ortho.tif`. `raster_reader.probe`/`open_reader` open it through
`remote_raster.RangeFile`: the header parse reads aligned 256 KB blocks (cached), tile reads fetch
exactly their bytes, adjacent tiles share a request and separate requests run in parallel. A
remote profile never uses memmap and always loads as a viewer/editor preview
(`_wants_preview`, `_load_editor_preview`), at a level whose download fits
`_REMOTE_PREVIEW_FETCH_BYTES` (`level_for_bytes`' fetch budget); the zoom tiles bring full resolution.
Size, transform and EPSG come from the remote header too (`shapefile_io._tiff_open_target`
inside `_raw_image_dims` / `get_geotiff_transform` / `epsg_from_geotiff`), so the polygon basis
is the full raster, never the preview, and the canopy finder reads `.cogref` window by window.
Data portals: `remote_portal.py` (no Qt, no GDAL) finds a Hub site's org, searches it and classifies
items as stream / download / vector, downloads (zip-slip safe) and turns feature services (GeoJSON
pages), GeoPackages (sqlite3 + Shapely WKB) and GeoJSON into shapefiles; `portal_finder_dialog.py` is
the search window (rasters from Add Remote Images, vectors from PolygonManager.on_import_from_portal,
which hands the shapefiles to on_import_shapefile(file_paths=...)). LERC image services stream as
float32 data (`TileService.data`).
Tiled ArcGIS services (`remote_tiles.py`; pointer `"kind": "arcgis-tiles"`, made from a MapServer /
ImageServer URL or an ArcGIS item / Hub page): `raster_reader.probe` defers to `remote_tiles.probe` and
`open_reader` returns `TileServiceReader`, a `TiledRasterReader` subclass whose levels are the service's
zoom levels (level 0 = finest zoom with data, cropped to the extent, 4 bands RGBA) and whose I/O fetches
`.../tile/{z}/{row}/{col}`; size/georeference helpers in shapefile_io answer from the service (EPSG:3857).
OneDrive / SharePoint share links (`onedrive_kind`, or a pointer's `"provider"`) get a resolver:
`resolve_onedrive` tries the consumer shares endpoint and `?download=1` (and `/redir` -> `/download`),
keeps the first address answering a Range request with file bytes, and `_RangeSource` renews it on
401/403/410 (its token expires). Google Drive links resolve to Drive's public download endpoint (`GDRIVE_DOWNLOAD`,
`confirm=t`; no key, file shared by link) or, when a key is set, to `files.get?alt=media`; the key comes from the app
(`remote_raster.set_default_api_key`, remembered in QSettings by the dialog and applied in
`MainWindow.__init__`) or `CANOPIE_GDRIVE_API_KEY`, is masked in every message and is never
written into a pointer. **File > Add Remote Images (URLs)…** (`remote_images_dialog.py`) runs
`remote_raster.make_pointer` per link and opens the pointers through
`ProjectTab.open_image_paths` -- the grouping into roots that File > Open Image files also uses.
CLI: `python -m canopie.remote_raster make <url> ortho.cogref`; measure with
`scripts/bench_remote_cog.py`; examples and user steps in `examples/remote_cogs/`.
Analysis: `_get_export_image` always gives a remote image the windowed `LazyAnalysis` (never a
whole read); an `.ax` step that needs the whole image (hist match, registration, classification,
up-size) is refused with that reason. `ProjectTab.iter_with_remote_prefetch` wraps the polygon
loops (`_process_file_batch`, ML `train_models` and `_CsvExportWorker` via
`MachineLearningManager._remote_prefetched`): items keep their order, and before an item whose
tiles are not in memory the tiles of it and the following items (`_REMOTE_PREFETCH_BATCH_BYTES`)
come down in one parallel round (`LazyChannels.prefetch` -> `TiledRasterReader.prefetch_rects`,
compressed bytes held in the reader's raw cache). Downloads report progress
(`remote_raster.add_progress_listener`) to the status bar's `StreamProgress`; each batch logs one
INFO line. Still refused: Export Project Images and exiftool.

## Where to start for common tasks

| Task | Start here |
|------|-----------|
| Add a menu action / global command | `main_window.py` `setup_menu` / `setup_main_toolbar`, then a delegate method on `ProjectTab` |
| Change how images are grouped into roots | `ProjectTab.open_folder` / `load_*_images_from_folder`, root offset logic |
| Change polygon drawing/editing UX | `image_viewer.py` (`EditablePolygonItem`, `start_drawing_with_group_name`, mouse events) |
| Add/adjust an image edit | `image_editor_dialog.py` + `ProjectTab.apply_aux_modifications` / `_apply_ax_to_raw` + `.ax` schema |
| Add a vegetation index / band function | `utils.py` (add helper + expose in `__all__`) and/or `performance.FastBandMathEngine` |
| Change CSV/EXIF/thumbnail export | export worker classes + `save_polygons_to_csv` / `extract_exif_to_csv` / `save_all_thumbnails` in `project_tab.py` |
| Change "Export Project Images" (image files, not CSV) | `ProjectImagesExportWorker` + `export_project_images` (both in `project_tab.py`) + `export_images_options_dialog.py` |
| ML training/prediction | `machine_learning_manager.py`, `performance.FastSklearnPredictor` |
| Project save/load format | `ProjectTab.save_project`/`load_project`, `MainWindow.load_project`, project.json keys |
| Speed up a hot pixel loop | `performance.py`; guard new deps behind availability checks |
| A drawn point/polygon looks wrong after reload, or you need to clean up already-corrupted saved coordinates | `image_viewer.py` (`EditablePointItem._scene_xy`, `on_polygon_drawn`/`on_polygon_modified` composing it with `item.mapToScene`) for the live bug; `ProjectTab.find_suspect_point_coordinates` (detect-only migration, next to `migrate_polygon_basis`) for already-saved files |
| Verify a change touching pixel values/export | `canopie/qc/` — the one permanent regression suite here, see the skill's "QC test suite" section and `python -m canopie.qc.which_tests <files>` |
