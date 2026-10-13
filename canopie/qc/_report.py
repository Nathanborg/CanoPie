"""
Feature-grouped reporting for the QC suite.

Produces a structured report at the end of every run, grouped by the CanoPie
FEATURE each test file covers rather than by filename, so the output answers
"which parts of the app are verified, and did any of them regress?" directly.

Written to `canopie/qc/last_run_report.txt` (gitignored) as well as printed,
because this environment's known native-crash flakiness can truncate terminal
output after results are computed -- the file always survives.
"""
import os
import time

# test module -> (feature name, what it actually asserts)
FEATURE_MAP = {
    "test_image_viewer": (
        "ImageViewer / image loading",
        "loader routing, band count, channel_order, raw pixel values, preview decimation"),
    "test_inspect_tool": (
        "Pixel Inspector",
        "per-band values at known coords, NoData per band (values and expressions), the band expression named by its text"),
    "test_process_polygon_csv": (
        "Main CSV export (process_polygon)",
        "per-band Mean/Median/Std/Q25/Q75/PixelCount vs ground truth, NoData per band; an all-NoData band keeps an empty row"),
    "test_ml_manager_csv_and_training": (
        "ML manager CSV export",
        "all 3 extraction modes, agreement with process_polygon, NoData handling"),
    "test_point_polygon_consistency": (
        "Point <-> polygon consistency",
        "1px-polygon == point, polygon Mean == mean of its points, cross-tool agreement"),
    "test_lazy_eager_agreement": (
        "Lazy vs eager raster reads",
        "windowed reader == whole-array reader, .ax windowability gate, crop offset"),
    "test_ml_manager_bugfix": (
        "ML manager LazyChannels fix",
        "regression guard for the .ndim crash on large rasters"),
    "test_ax_pipeline": (
        ".ax edit pipeline",
        "rotate / resize / crop / band_expression applied correctly and in order"),
    "test_stretch_rendering": (
        "Display stretch rendering",
        "percentile/stddev/absolute stretch, single-band vs RGB, raw data left untouched"),
    "test_band_bar": (
        "In-viewport band selector",
        "population, band switching, composite restore, auto-hide on draw"),
    "test_band_math": (
        "Band math / vegetation indices",
        "expression evaluation, reducers, band references, NaN handling"),
    "test_random_shapes": (
        "Random shape generation",
        "count, containment in valid area, spacing constraints, stratification"),
    "test_shapefile_export": (
        "Shapefile export",
        "geometry/attribute round-trip, pixel<->geo transforms, DBF field naming"),
    "test_project_roundtrip": (
        "Project save / load",
        "project.json round-trip, polygon persistence, root mapping"),
    "test_jpeg_compressed_tiff": (
        "JPEG-compressed TIFF / COG decoding",
        "shared JPEGTables forwarding, tile decode, real GDAL COG regression"),
    "test_expression_nodata_lazy_path": (
        "Expression-based NoData on the lazy read path",
        "boolean NoData rules (b7==0) on windowed reads, real-project regression"),
    "test_scene_stats_nodata": (
        "Scene stats under NoData",
        "Scene Mean/Median exclude fill values on the lazy path, lazy==eager"),
    "test_band_math_formula_parsing": (
        "Band-math / indices formula parsing",
        "bare comparisons (b7==0) parse and yield a CSV row with the boolean average"),
    "test_stretch_bar": (
        "In-viewport stretch bar",
        "min/max readout populates, range follows the selected band, agrees with the dialog"),
    "test_inspect_nodata_nan": (
        "Pixel Inspector / non-finite pixels",
        "NaN and Inf read as NoData without an .ax, per channel, incl. band-expression index"),
    "test_shapefile_crs": (
        "Shapefile CRS handling",
        "per-feature CRS, mixed-CRS split into separate files, pixel coords get no .prj"),
    "test_shapefile_batch_export": (
        "Shapefile export during batched/background CSV runs",
        "ExportWorker writes the shapefile, processed-subset scoping, CSV-ordered DBF fields"),
    "test_hist_match": (
        "Histogram matching",
        "diagnostic plot maths, windowing gate, viewer/CSV/ML/Inspect agreement"),
    "test_hist_match_nodata": (
        "Histogram matching under NoData",
        "declared GDAL_NODATA honoured, fill sentinel preserved, per-band masking"),
    "test_contract_band_math": (
        "CONTRACT: band-math engine equivalence",
        "FastBandMathEngine == eval_band_expression on hostile inputs; cache cannot leak across images"),
    "test_classification_pixel_eligibility": (
        "CONTRACT: viewer and CSV classify the same pixels",
        "NoData / non-finite / window-edge pixels excluded from prediction exactly as training "
        "drops them; viewer class percentages == process_polygon's Class N % columns"),
    "test_band_convention_contract": (
        "CONTRACT: b1 = Red everywhere",
        "viewer/export/process_polygon/editor/ML agree on b1 for BOTH cv2 and tifffile provenance; "
        "bands 4+ never reordered; appended bands share the same b1; no __B1_TEMP__ token rewriting"),
    "test_contract_ax_consumers": (
        "CONTRACT: .ax replays agree across consumers",
        "viewer/CSV/ML/Inspect see identical pixels per .ax op; stats match a NumPy recompute"),
    "test_perf_budgets": (
        "Speed and memory budgets",
        "lazy cube never materialised, bounded windowed reads, peak-memory and latency ceilings"),
    "test_nodata_expression_per_band": (
        "Per-band NoData expressions (b2>1) on R/G/B",
        "a rule naming any band affects THAT band's row; R/G/B no longer collapse onto band 1's mask"),
    "test_highres_viewport_threading": (
        "COG zoom refinement (high-res viewport tiles)",
        "worker-thread tiles reach the GUI via a signal, not a dead QTimer; zoom requests a region"),
    "test_highres_ax_consistency": (
        "Editor edits vs the zoom overlay (COG)",
        "crop/resize/rotate/band math are replayed on the zoom tile; hist match correctly disables it"),
    "test_highres_ax_freshness": (
        "Zoom tile uses the CURRENT edits",
        "the tile resolves image data and .ax per request, so edits are never rendered stale"),
    "test_polygon_display_simplification": (
        "Zoom-aware vertex decimation",
        "display-only simplification: 1.01M vertices drawn at screen resolution, geometry untouched"),
    "test_delete_undo_cost": (
        "Bulk delete / undo cost",
        "undo snapshots cost nothing: sidecars are moved to trash, not read or re-serialised"),
    "test_polygon_manager_scaling": (
        "Polygon manager at thousands of groups",
        "batched list rebuild + range selection; clean paths use the bulk-delete context"),
    "test_project_polygons_external": (
        "project.json does not duplicate polygons",
        "polygons/ dir is the single source of truth: 360x faster saves, no resurrected deletes"),
    "test_shapefile_import_worker": (
        "Shapefile import, end to end",
        "the worker actually runs; bbox pre-filter is CRS-correct and fails open; skips are counted"),
    "test_shapefile_import_perf": (
        "Shapefile import speed + progress bar",
        "the CRS transformer is built once per file, not per feature; progress moves per feature"),
    "test_polygon_bounding_rect_perf": (
        "Polygon boundingRect stays tight",
        "no label box is reserved when no label is drawn -- 10.6x faster panning at 6000 polygons"),
    "test_bulk_polygon_delete_perf": (
        "Bulk polygon delete is O(N), not O(N^2)",
        "manager refresh, .ax read, and scene scan happen once per batch, not once per polygon"),
    "test_shapefile_import_render": (
        "Shapefile-imported polygons render",
        "coord_space stamped 'image' not 'pixel'; empty-string root isn't a mismatch; import draw loop batches, catches errors, repaints"),
    "test_editor_refresh_preserves_view": (
        "Viewer state across Apply All Changes",
        "the post-editor refresh reaches the stretch-aware renderer instead of silently falling back"),
    "test_fix_zoom_root_navigation": (
        "Fix Zoom anchor across root navigation",
        "the pinned view center reads back exactly what centerOn set (sub-pixel), so N root changes with no panning never move it"),
    "test_classification_render_eligibility_fast_path": (
        "Classification render: eligible-rows fast path",
        "all-valid tiles skip the boolean-index copy and scatter; partly-invalid tiles still exclude NoData rows; offsets correct"),
    "test_polygon_identity_vs_label": (
        "Polygon identity is the group key, not the label",
        "reloaded crowns named by group and labelled by name; moving keeps group, name and properties; PM delete and Clean Vector remove them from the screen"),
    "test_polygon_sidecar_writes": (
        "Polygon sidecars: written once, by the C encoder",
        "one write per dirty polygon per save (no second flush write), one canonical format with properties, no pure-Python JSON encoding"),
    "test_navigation_speed_contracts": (
        "Navigation never reads or draws more than the screen shows",
        "strided reads == sliced full reads (memmap, tiles, strips); no sharp tile below 1.1 device px per preview px; no LOD swap for sub-2-px gains; exact numpy<->QPolygonF"),
    "test_lazy_point_nodata": (
        "Points on NoData give empty rows on both read paths",
        "large files (lazy windowed read) no longer export a point on NoData with the fill value as its mean -- the row is kept, empty; eager and lazy CSVs agree"),
    "test_ax_geometry_exact": (
        "Crop / rotate / resize geometry matches the export exactly",
        "frame size and point forward/inverse maps equal _apply_ax_to_raw's rounding, rotation order and resize modes on every .ax combination"),
    "test_polygons_follow_ax": (
        "Polygons and points stay on their trees when crop / rotation / resize change",
        "exact re-projection on every image type with a zip backup first, resize-only changes free, one move per image "
        "whatever its path spelling, lazy crowns stay lazy, Reset counted once, stale items never read back; editor "
        "opens without paging crowns in; the zoom tile shows exported pixels; a point drag paints each pixel once"),
    "test_post_ax_size_helpers": (
        "Copy/replicate size helpers give the exported frame size",
        "_raw_hw_quick reads the header (it raised TypeError); the post-.ax size equals the exported image's size for crop, "
        "rotation, px_w resize, disabled crop and a stale orig_size"),
    "test_editor_open_state": (
        "The image editor opens knowing its pixels",
        "a reopen served from the viewer's or the editor's cache keeps the channel order (b1 = Red on a stack); "
        "orig_size is the raster header size; the editor's expression is NoData where a band it reads is"),
    "test_lazy_analysis": (
        "Large rasters: band math and booleans computed window by window",
        "the windowed analysis image equals the whole-image replay for crop, every rotation, integer and odd down-sizes, "
        "NoData (number / expression / NaN), booleans, appended bands and mask polygons -- in one window, in odd-sized "
        "windows and in row strips; steps needing the whole image are refused; a huge full band is refused, not attempted"),
    "test_preview_px_resize": (
        "A pixel resize on a decimated preview",
        "viewer and editor apply the export's resize FACTOR to a preview (they blew it back up to the full size); "
        "the editor's resize boxes and save work in full-frame pixels and an open + save writes no resize; "
        "a full image keeps the exact pixel size"),
    "test_stretch_dialog_previews": (
        "The stretch dialog on preview-loaded rasters",
        "a preview loads the file band its stretch shows; 'all viewers' loads it into every preview and draws it; a "
        "band the viewer does not hold is saved as its file band (not band 1); the dialog opens on the saved band and "
        "keeps the saved Absolute range (it reset it to the band's data range on every open); an image without the "
        "band a root / project stretch shows keeps its own display (and its own stretch); reloaded bands keep the .ax crop"),
    "test_polygon_zoom": (
        "Polygon manager: click and Zoom to polygon",
        "a list click pans only, Zoom to polygon fits a real polygon with a native-resolution cap; the zoom rect is "
        "where the polygon is drawn -- also on a decimated COG / mosaic preview with an .ax (it drifted ~7x off)"),
    "test_canopy_finder": (
        "Find canopy in other images (GeoTIFF)",
        "a crown goes through world coordinates (any CRS / resolution) and the .ax crop into other GeoTIFFs and the "
        "exhaustive / StackReg engines find it within 1 px through a known shift + rotation; it is saved as the same "
        "group (an existing one kept unless replaced, after a zip backup), viewers zoom onto it, other projects are "
        "never written, the exported square is georeferenced, browser edits update the record; raw drone frames "
        "located by EXIF (KD-tree picks the frames nearest the crown) ortho->raw, raw->ortho, raw->raw at their own "
        "resolution with GPS / height / heading errors; phenocams in image coordinates; a COG read from the overview "
        "at the working resolution; dragged outlines and Save all; the target scan never blocks the dialog; "
        "images are matched in their post-.ax frame (crop / rotation / NoData from the .ax replay, rotated squares "
        "exported with a ModelTransformation); a GPS-only flight (no height / heading) is calibrated from its own "
        "overlap; GPS 0,0 is no GPS; several crowns in one run; the image editor's palette; GPS from the file "
        "header in threads (same values as the whole file); the image list never measures frames; no typed "
        "flight height / sensor width; a 20 % low or 25 % high ground size still found; the aligned crown "
        "stack (every target on the source crown grid, georeferenced like the source square); the EXIF cache "
        "kept per image folder and reused next session (nothing written beside the images, a changed file read "
        "again, a plain TIFF not reopened for a georeference); several folders, each with its own nearest frames; "
        "searches and flight-line measurements in worker processes (finder modules only, no Qt, below-normal "
        "priority) give the result of one search after another, also for GPS-only frames; a replaced or "
        "browser-edited crown replaces the outline drawn in the grid viewer; the dialog sections fold; a crown "
        "dragged in a rotated browser tile lands where it was put on the (unrotated) image; a mosaic with no "
        "data under the crown does not hold it; the export is only the source crop + each target's crop aligned "
        "on it (GeoTIFF or JPEG) + one metadata CSV, into the project's canopy_exports by default, rejected "
        "images left out (a re-export drops their earlier files), with each image's crown polygon as JSON (its own "
        "frame + the crop's pixels); crowns selected together in the polygon manager, from different roots, are "
        "each searched from their own image"),
    "test_polygon_sync_scaling": (
        "A synced polygon keeps its place on every image of the root",
        "drawn with sync on, the copies sit at the same fraction of each frame (as copy-to-root) and are stamped "
        "with each image's basis -- also when an image is shown resized, cropped or as a decimated preview"),
    "test_export_speed_exactness": (
        "Faster CSV export, same bytes",
        "median/quantiles from one partition keep numpy's bits; the expression evaluator never returns the caller's "
        "memory; band math after an .ax edit comes from the new image and is dropped per file; the label-band guess "
        "runs once per file; thumbnails and the augmentation preview work on windowed rasters; a whole-image replay "
        "that cannot fit is refused and reported; previews hold at most 16 Mpx; saving a polygon's size reads no pixels"),
    "test_band_bar_names": (
        "The band bar names bands like the CSV",
        "on a B,G,R-loaded image button 1 is file band 1 (Red) and selects its array position; "
        "the band expression's button carries its expression text"),
    "test_stretch_band_basis": (
        "A saved stretch names file bands (red stays red)",
        "a stretch saved on a B,G,R-loaded image draws natural colour on a file-order one; a legacy 2/1/0 entry shows "
        "file Red as red (it showed Blue); root stretches follow each image's band order; a derived band is kept by name"),
    "test_tile_stretch": (
        "Zoom tiles keep the preview's stretch; 0/1 bands stay black/white",
        "a tile maps with the bounds the preview was drawn with (both stretch engines, wiring included); "
        "a boolean band maps 0 -> black, 1 -> white even when a tile is all True; back to RGB in the stretch "
        "dialog the tiles get the RGB bounds (they kept a boolean's 0..1); bounds of another stretch are never reused"),
    "test_export_dialog_nodata": (
        "Export dialog NoData adds to each image's own",
        "values typed in the CSV export dialog are masked in addition to every image's own NoData (eager and lazy), "
        "never instead of it; the field is no longer prefilled from another image's .ax"),
    "test_inspect_export_pixel": (
        "Inspect reads the exported pixel",
        "full resolution on a decimated preview (eager and lazy), CSV band names with no duplicate index, NoData per band incl. the raster's own tag"),
    "test_ml_training_rows": (
        "ML training samples the exported pixels",
        "training rows are raw file pixels in file band order whatever another export left behind; NoData drops a row "
        "only when one of its features is NoData (eager and lazy); an expression reading NoData is NoData; rows keep "
        "their class label when the image has a mask polygon; expressions work on lazily read images"),
    "test_viewer_nodata_display": (
        "NoData on screen: white only where a shown band is NoData",
        "the display whitens a pixel when a band it shows is NaN, never for a band that is not on screen; "
        "a raster's own GDAL_NODATA tag reaches the viewer data with no .ax, NaN exactly where the export's is"),
    "test_undo_trash_and_mask_replay": (
        "Undo keeps the trash; mask polygons replay from memory",
        "the undo reload does not purge .trash (only opening a project does); a mask polygon held only in memory "
        "no longer breaks the viewer's .ax replay"),
    "test_point_pixel_snapping": (
        "A point is one export pixel: click, highlight, marker, CSV and ML agree",
        "clicks and drags snap to the export pixel centre; hover and marker cover exactly that pixel on decimated previews; points follow a crop / rotate change; ML neither clamps nor shifts edge points"),
    "test_polygon_lod_follows_zoom": (
        "Polygon detail follows the zoom (QGIS-style)",
        "exact outlines zoomed in, coarsest sub-pixel level in between, decimated-preview aware, display-only; Edit Vertices upgrades first"),
    "test_band_expression_display_persists": (
        "A band expression applied in the editor stays on screen",
        "Apply stores the expression band as the .ax display (no revert to RGB on refresh/reopen); switching band math off restores the previous stretch"),
    "test_export_and_ax_regressions": (
        "Export path + .ax durability regressions",
        "atomic .ax writes (no truncation, None deletes, concurrent merges), CSV path-spelling dedupe, classification failures surfaced, cache-clear completeness"),
    "test_point_drag_refresh_stability": (
        "A point stays where it was dropped, across refreshes",
        "EditablePointItem stored its geometry floor()ed to a whole PIXMAP pixel, but the SAVED coordinate is a float in the full-raster basis -- with an .ax resize one pixmap pixel is 41 x 47 raster pixels, so every reload quantised away the drop position (measured on the reported 5-band 2064x1544 capture: 24/24 dragged points moved 19-61 raster pixels on the next refresh, and freshly drawn ones jumped a full pixmap pixel when the load-side round trip landed at k - 1e-15)"),
    "test_point_preview_matches_final": (
        "Point drawing preview == the committed point",
        "the temp marker that follows the cursor is built by the SAME helper as the finalized point (it used to keep its own floor()ing copy, so the marker jumped up to a whole pixmap pixel -- 16.6 screen px / 38 raster px on the reported .ax-resized capture -- the instant the point was committed), plus EditablePointItem's grab region is centred ON the point instead of 3px down-right of it"),
    "test_point_scene_coordinates": (
        "Point item scene-coordinate reconstruction",
        "drawn/moved points saved via _scene_xy, not raw pixmap-local storage -- fixes points going stray on project reload"),
    "test_suspect_point_coordinates": (
        "Migration: detect stray points saved before the scene-coordinate fix",
        "filepath resolved by filename-stem match (no such key exists in the JSON body), out-of-bounds points removable, sibling-outlier points report-only via leave-one-out extent"),
    "test_export_images_band_order": (
        "Export Project Images band order (background + foreground)",
        "channel order tracked from the loader (tifffile=native, cv2=BGR) instead of assumed; the real bug was in the tifffile.imwrite branch (band expr/classification/float sources), not the cv2.imwrite branch which already round-tripped correctly -- verified empirically both ways"),
    "test_export_images_options_dialog": (
        "Export Images options dialog (format / band order / EXIF / background)",
        "replaces the old pair of QMessageBox prompts; defaults reproduce their old answers exactly; format and band order now thread through to both the background worker and the foreground path"),
    "test_vertex_edit_perf": (
        "Edit Vertices crash on high-vertex-count polygons",
        "dragging a handle rebuilt the ENTIRE polygon (3 full O(n) passes) on every throttled frame regardless of the 100-handle cap; _set_vertex_point does an O(1) in-place update instead, deferring the one full resync to drag-end -- verified with a real 20,000-vertex polygon through the actual start/apply/stop methods, and confirmed the old code blows the timing bound on revert"),
    "test_polygon_disappear_on_refresh_race": (
        "Polygons vanishing after Image Editor apply (classification and others)",
        "the debounced polygon-load pipeline bailed on a global _nav_seq counter bumped by ANY navigation/refresh anywhere, silently dropping loads for grids it never touched; both bail-outs now ask the precise question (grid identity / per-viewer filepath) instead"),
    "test_appended_band_failures": (
        "Appended bands (classification/boolean/band-expression) silently missing from exports",
        "_apply_ax_to_raw's per-band try/excepts only logged a warning on failure -- most commonly a classification band whose sklearn model isn't loaded in the current session -- so the band just vanished from CSV/image/ML output with nothing visible; now tracked and surfaced in every consumer's completion message, verified against a real project + a real trained model"),
    "test_csv_column_order": (
        "CSV column ordering",
        "canonical order shared by foreground and background exports; quantiles numeric not alphabetical"),
    "test_band_math_scene_stats_sampling": (
        "Band-math Scene Mean/Median/Std on the lazy export path",
        "no longer NaN, agrees with eager within tolerance, per-band NoData reindexed correctly"),
    "test_image_editor_ml": (
        "Image editor / ML integration",
        "classification mapping, hist-match + crop consistency, model class sync"),
    "test_image_viewer_ui": (
        "Viewer overlay toggle",
        "global overlays_muted flag, bars hide while drawing"),
    "test_label_defaults_and_mapping": (
        "Label defaults + image->scene mapping",
        "labels OFF by default (3-5x zoomed-in paint, culling not defeated); mapping bit-identical to Qt incl. rotation"),
    "test_polygon_save_basis": (
        "Saving never rebases polygons onto the viewer preview",
        "save_polygons_to_json keeps the raster basis; repeated saves do not walk polygons toward the origin; a genuinely stale basis is still repaired"),
    "test_polygon_coordinate_basis": (
        "Polygon coordinate basis (COG preview vs raster)",
        "image_ref_size is always the full raster, stable across preview levels; migration rescales and is idempotent"),
    "test_polygon_batch_layer": (
        "Tiled batch rendering above ~800 polygons",
        "items hidden not removed (no autosave purge), tight cullable tile rects, lazy LOD levels, 11x faster zoomed out"),
    "test_polygon_manager_advanced": (
        "PolygonManager advanced operations",
        "nearest-image search, copy/move between roots, multispectral->thermal correction"),
    "test_thumbnail_options": (
        "Thumbnail options dialog and DL tiling",
        "dialog defaults reproduce the old hardcoded 200x200/1.4/outline behaviour exactly; outline toggle skips drawing; every polygon yields at least one tile (crop grown, not dropped); inside-only keeps whole squares and its live estimate matches what the run writes; zero-padded row-major tile names; the crop is read once, not once per tile"),
    "test_ml_manager_cog_windowed": (
        "Thumbnails and segmentation masks stay windowed on a COG",
        "per-polygon crops instead of a whole-frame preview (eager crops byte-identical); geometry-only shape without decoding; full-res masks built in strips, pixel-exact vs a dense build"),
    "test_polygon_path_key_shadowing": (
        "Two spellings of one image path must not hide each other",
        "polygon lookups union the exact and normalised indices, so one drawn polygon cannot hide every imported one; saving finds polygons keyed under normpath; untouched pyramid records are not paged in"),
}


class QCReporter:
    def __init__(self, report_path):
        self.report_path = report_path
        self.results = []          # (module, testname, outcome, duration, longrepr)
        self.session_start = time.time()

    def record(self, report):
        if report.when != "call" and not (report.when == "setup" and report.outcome in ("failed", "skipped")):
            return
        if report.when == "setup" and report.outcome == "passed":
            return
        module = report.nodeid.split("::")[0].split("/")[-1].replace(".py", "")
        testname = report.nodeid.split("::", 1)[1] if "::" in report.nodeid else report.nodeid
        longrepr = ""
        if report.outcome == "failed":
            longrepr = str(report.longrepr)
            # keep the assertion line, drop the noise
            lines = [ln for ln in longrepr.splitlines() if ln.strip().startswith("E ")]
            longrepr = " | ".join(ln.strip()[2:].strip() for ln in lines[:3]) or longrepr[-300:]
        self.results.append((module, testname, report.outcome, report.duration, longrepr))

    def build(self):
        by_module = {}
        for module, testname, outcome, duration, longrepr in self.results:
            by_module.setdefault(module, []).append((testname, outcome, duration, longrepr))

        total = len(self.results)
        passed = sum(1 for r in self.results if r[2] == "passed")
        failed = sum(1 for r in self.results if r[2] == "failed")
        skipped = sum(1 for r in self.results if r[2] == "skipped")
        elapsed = time.time() - self.session_start

        W = 78
        out = []
        out.append("=" * W)
        out.append("CANOPIE QC REPORT".center(W))
        out.append(time.strftime("%Y-%m-%d %H:%M:%S").center(W))
        out.append("=" * W)
        out.append("")

        out.append("FEATURE COVERAGE")
        out.append("-" * W)
        for module in sorted(by_module):
            feature, detail = FEATURE_MAP.get(module, (module, ""))
            entries = by_module[module]
            p = sum(1 for e in entries if e[1] == "passed")
            f = sum(1 for e in entries if e[1] == "failed")
            s = sum(1 for e in entries if e[1] == "skipped")
            dur = sum(e[2] or 0.0 for e in entries)
            status = "FAIL" if f else ("PASS" if p else "----")
            out.append(f"[{status}] {feature}")
            out.append(f"        {p} passed"
                       + (f", {f} FAILED" if f else "")
                       + (f", {s} skipped" if s else "")
                       + f"  ({dur:.2f}s)")
            if detail:
                out.append(f"        covers: {detail}")
            for testname, outcome, _d, longrepr in entries:
                if outcome == "failed":
                    out.append(f"          FAILED: {testname}")
                    if longrepr:
                        out.append(f"                  {longrepr[:200]}")
                elif outcome == "skipped":
                    out.append(f"          skipped: {testname}")
            out.append("")

        untested = [m for m in FEATURE_MAP if m not in by_module]
        if untested:
            out.append("NOT EXERCISED IN THIS RUN")
            out.append("-" * W)
            for m in sorted(untested):
                out.append(f"  - {FEATURE_MAP[m][0]}  ({m}.py)")
            out.append("")

        out.append("=" * W)
        verdict = "ALL GREEN" if failed == 0 else f"{failed} FAILURE(S) -- SEE ABOVE"
        out.append(f"TOTAL: {total} tests | {passed} passed | {failed} failed | "
                   f"{skipped} skipped | {elapsed:.1f}s")
        out.append(f"VERDICT: {verdict}")
        out.append("=" * W)
        return "\n".join(out)

    def write(self):
        text = self.build()
        try:
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
        return text
