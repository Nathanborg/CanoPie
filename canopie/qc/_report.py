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
        "per-band values at known coords, NoData display modes, band-expression index"),
    "test_process_polygon_csv": (
        "Main CSV export (process_polygon)",
        "per-band Mean/Median/Std/Q25/Q75/PixelCount vs ground truth, NoData masking"),
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
