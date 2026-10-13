# CanoPie — bug list

The single place every known bug lives. Read this at the start of every session.

**Rules**
- No fix starts without an entry here.
- Status flow: `open` → `reproduced` → `fixed-unverified` → `verified`.
- `verified` means all of: reproduced on **real data** in a fresh project folder; a QC test
  that fails without the fix (revert-verified); full suite green before and after
  (`python -m pytest canopie/qc -q`, totals in `canopie/qc/last_run_report.txt`); confirmed
  in the app by Nathan; committed.
- Severity: **high** = wrong data or lost work · **medium** = wrong display, crash with a
  workaround, broken tooling · **low** = cosmetic.

---

## Open

| ID | Title | Area | Sev | Status |
|---|---|---|---|---|
| BUG-001 | Fix Zoom: viewers other than the first drift on every root change | viewer | high | open |
| BUG-002 | Single-viewer Fix Zoom drift (sub-pixel per navigation) | viewer | high | fixed-unverified (committed 00b9ab7; awaiting Nathan's check in the app) |
| BUG-003 | `requirements.txt` omits `pyproj` and `pyshp` | packaging | high | open |
| BUG-004 | QC report says "ALL GREEN" when 0 tests ran | qc | medium | open |
| BUG-005 | `os._exit` in `pytest_unconfigure` hides pytest's summary and collection errors | qc | medium | open |
| BUG-006 | 12 methods silently shadowed by a later definition of the same name | code health | medium | open |
| BUG-007 | Boolean band expression shows after Apply, then reverts to RGB (D1); zoom tiles re-stretched / a True tile drawn black (D2/D3) | viewer/editor | high | fixed-unverified (D1 committed c7008a3; D2/D3 Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-008 | Editor second open: b1 treated as Blue; `orig_size` stamped from the preview (E1) | editor | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-009 | Band numbering differs per consumer: "band 4" = alpha in CSV/ML, = expression on screen (X1) | extraction | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-010 | Inspect reads the decimated preview, mislabels bands, duplicates the expression (X2) | extraction | medium | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-011 | Classification + band expression build different feature stacks in viewer vs export (X3) | ML | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-012 | ML lazy read builds its window from unscaled points — may read wrong pixels silently (M1) | ML | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-013 | ML `_image_cache` loses channel order — R/B swapped features on mixed roots (M2) | ML | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-014 | ML clamps polygon vertices, CSV clips — edge polygons rasterise differently (M3) | ML | medium | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-015 | NoData resolved differently by CSV, ML, Inspect and viewer (M4) | extraction | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-016 | Registration applied twice in export, and in the wrong frame under rotation (M5) | extraction | medium | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-017 | Item identity mixed up with its label: deletes / Clean Vector leave crowns on screen, moves create property-less duplicates (P1) | polygons | high | fixed-unverified (awaiting Nathan's check in the app) |
| BUG-018 | Refresh recreates cleared polygons from leftover scene items (P2) | polygons | high | not reproduced (steps 3, 3c, 3e) |
| BUG-019 | Imported polygons disappear after a refresh: load-time de-dup by LABEL drops every crown sharing a species name | polygons | high | fixed-unverified (cause found; part of the BUG-017 fix) |
| BUG-020 | Polygon writers stamp mixed bases (preview vs raw); two broken size helpers (P4) | polygons | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-021 | Undoing a bulk delete empties the whole `.trash` (P5) | polygons | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-022 | Polygons stored in the cropped frame move off their trees when the crop changes (P6) | polygons | high | fixed-unverified: records re-projected on every crop / rotation change (BUG-040 points, BUG-043 polygons) |
| BUG-023 | `apply_aux_modifications` is a staticmethod but uses `self` (NameError with mask_polygon) (S1) | viewer | medium | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-024 | Viewer shows red and blue swapped on this project (screen red = file Blue) | viewer | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-025 | CSV writes an empty `Label` row when no classification is active | extraction | low | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-026 | Re-import leaves stale `viewer.polygons` entries (5,603 for 2,802 items) that still feed the zoomed-out tiles | polygons | medium | fixed-unverified |
| BUG-027 | "Import Polygons from Project" cannot run on an orthomosaic project (needs a folder + EXIF GPS) | polygons | high | reproduced |
| BUG-028 | ML pixel export x,y are in the cropped frame, not the original image grid | ML | medium | closed by design (Nathan 2026-10-06: ML x,y stay in the export frame) |
| BUG-029 | Fix Zoom navigation test failed once in a full run (anchor drift 0.0058) | qc / viewer | medium | open (intermittent; 1 of 2 full runs on 2026-10-05, 1 of 4 on 2026-10-06) |
| BUG-030 | Shapefile import freezes the app ~95 s for 2,801 crowns: every sidecar written twice, by the pure-Python JSON encoder | polygons / io | high | fixed-unverified (105 s -> 18 s total, GUI 93 s -> 6.7 s) |
| BUG-031 | Polygon detail never follows the zoom: crowns keep the 16-unit overview outline at every zoom (`level_for_scale` unused) | polygons / viewer | high | fixed-unverified (QGIS-style refine, ortho + COG measured) |
| BUG-032 | Edit Vertices / Copy / Replicate read a pyramid item's simplified outline and save it over the real geometry | polygons | high | fixed-unverified |
| BUG-033 | High-res overlay never disabled when an image has no reader: the `disable_highres_viewport` `elif` is unreachable (`project_tab.py`, display_image_group) | viewer | low | open |
| BUG-034 | `_overview.json` keeps listing cleared polygons (2,801 entries, 0 files in C:\Nova pasta (6)); harmless (the loader checks files) but stale | polygons | low | open |
| BUG-035 | Full QC run segfaults intermittently at 95%, in `test_temp_drawing_pen_zoom.py::test_start_drawing_with_group_name_pen_survives_zoom[50.0]` (passes alone) | qc / viewer | medium | open (pre-existing: HEAD 73ab693 crashed 2/2, current tree 1/2) |
| BUG-036 | Large files (lazy read) export a POINT on NoData as a row with the fill value (9999) as its mean; small files omit it | extraction | high | fixed-unverified |
| BUG-037 | Navigation stalls (zoom/pan freeze 1-2 s) on the BCI mosaic and the COG: full-res sharp tiles at every zoom, eager LOD swaps, per-vertex polygon loops | viewer / perf | high | fixed-unverified (stalls 6.9 s -> 0.34 s per zoom sequence) |
| BUG-038 | Applying a stretch drops the view's scene padding (`update_pixmap_only` sets the scene rect to the pixmap rect) | viewer | low | open |
| BUG-039 | A point's marker and highlight covered k x k export pixels (k = export px per preview px: 18x18 on resize 50 %, 28x32 on crop + px_w) and clicks were not snapped to a pixel | points / viewer | high | fixed-unverified |
| BUG-040 | Points move off their pixel when a crop / rotate / resize is applied after drawing them (P6 for points) | points | high | fixed-unverified |
| BUG-041 | ML clamps point pixels to the image edge (an off-frame point samples the edge pixel); CSV floors and skips it | ML / points | medium | fixed-unverified |
| BUG-042 | Zoomed in, the sharp tile showed SOURCE pixels when the export is downsized: 2,500 BCI pixels under a one-pixel point on a 566 x 511 export ("bigger drawing than the pixels I have") | viewer | high | fixed-unverified |
| BUG-043 | Polygons slide off their trees when crop / rotation change after drawing or importing (P6 for polygons; points were BUG-040) | polygons | high | fixed-unverified |
| BUG-044 | Editor Reset: crowns "disappear" (view left on a black corner of the 16x larger frame, nothing loaded until the dialog closed), are not re-projected on Reset / Cancel, and the Reset refresh read old-frame items back over the records | editor / polygons | high | fixed-unverified |
| BUG-045 | Image editor took 60 s to open on the BCI project: the mask-polygon menu paged in all 2,802 crowns for their names, and every open re-decoded the 2.9 GB mosaic | editor / perf | high | fixed-unverified (60 s -> 0.4 s first open, 0.24 s after) |
| BUG-046 | Reading back sidecars the app just wrote pays the virus scanner's first-read cost (~20 ms a file: 90 s for a Reset after a crop) | polygons / perf | medium | fixed-unverified (write-through cache) |
| BUG-047 | A drag in point mode appended unsnapped, duplicated points (23 points over 10 pixels on New Folder223) | points / viewer | medium | fixed-unverified |
| BUG-048 | A band that is NaN at a point is left out of that point's CSV rows (139-band stack: 138 rows, band 19 is NaN on 99.9 % of the image) | extraction | low | fixed-unverified (Nathan 2026-10-06: keep the row, empty; real 139-band window: 139 rows, band_19 empty) |
| BUG-049 | CSV export dialog's NoData field: prefilled from the first `.ax` in the folder and REPLACING every image's own NoData | extraction | high | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-050 | ML training with a custom expression crashed on a lazily read (large) image (`LazyChannels` has no `astype`) | ML | medium | fixed-unverified (Stage 1, 2026-10-06/07; awaiting Nathan's check in the app) |
| BUG-051 | `test_highres_request_fires_after_the_burst_settles` fails intermittently when run with its file (pre-existing: also with the earlier viewer, 1 of 4) | qc / viewer | low | open (intermittent) |
| BUG-052 | A boolean / band expression on a COG or a large mosaic crashed the app: Inspect, CSV / ML export, training, thumbnails and saving a polygon read the whole raster; the viewer kept a 1.3 GB float preview | extraction / viewer | high | fixed-unverified (2026-10-07; awaiting Nathan's check in the app) |
| BUG-053 | Band-math rows of a re-export after an `.ax` edit came from the OLD image (background export; cache keyed by file only, never dropped -- also kept every file's whole-frame arrays) | extraction | high | fixed-unverified (2026-10-07; awaiting Nathan's check in the app) |
| BUG-054 | Polygon thumbnails silently skipped on large (windowed) rasters: the image has no `.size` and was taken for an empty one | thumbnails | medium | fixed-unverified (2026-10-07) |
| BUG-055 | ML augmentation preview put a whole large raster in memory | ML | medium | fixed-unverified (2026-10-07) |
| BUG-056 | An image refused for memory (whole-image `.ax` step) exported no rows while the export said "(0 errors)" | extraction | medium | fixed-unverified (2026-10-07) |
| BUG-057 | `test_a_4326_source_needs_no_transformer` failed once in a full run: it built two CRS objects and relied on Python reusing the first one's address for the `id()` cache key (test defect; the import holds one CRS) | qc | low | fixed-unverified (2026-10-07; test holds one CRS) |
| BUG-058 | Zoomed in after a boolean and back to RGB (stretch dialog), the image turns white / red / blue: every zoom tile mapped R, G, B with the boolean's 0..1 | viewer | high | fixed-unverified (2026-10-07; awaiting Nathan's check in the app) |
| BUG-059 | A polygon drawn with sync on landed at the wrong place (and was SAVED there) on the other images of the root whenever the image drawn on is shown resized / cropped / as a decimated preview: scaled by displayed array sizes instead of each image's basis | polygons | high | fixed-unverified (2026-10-07; awaiting Nathan's check in the app) |
| BUG-060 | The viewer replays a pixel resize (`px_w`/`px_h`) at its absolute size on a DECIMATED preview: New Folder213's 3.3 m mosaic preview (1/8, 2438 x 2186) is enlarged back to 19499 x 17481 float (1.36 GB + a 341 Mpx pixmap) -- defeats the 16 Mpx cap, a crash risk with more bands. The `.ax` itself (`px` = raw size, `resize_ref_size` = the preview) was written by the editor on the preview | viewer / editor | high | fixed-unverified (2026-10-09; awaiting Nathan's check in the app) |
| BUG-061 | Stretch dialog on Nova pasta (10): every viewer black, "all viewers" not applied, the band saved as band 1, and the saved Absolute range reset on every open | viewer / stretch | high | fixed-unverified (2026-10-07; awaiting Nathan's check in the app) |
| BUG-062 | Single-band stretch drawn as a 0/1 boolean (all white) whenever the band's first column is all 0 -- an orthomosaic's black border: a band change in Percentile gave a binary picture, Percentile "stopped working" after Absolute | viewer / stretch | high | fixed-unverified (2026-10-07; my D3 code of 2026-10-06) |
| BUG-063 | Polygon manager "Zoom to polygon" drifted off the crown on large rasters (Nova pasta (5)): the zoom rect pushed the vertices through the .ax again and divided by the PREVIEW's size -- ~7x too far from the origin | polygons / viewer | high | fixed-unverified (2026-10-09; awaiting Nathan's check in the app) |
| BUG-064 | An all-viewers / root stretch naming a band an image does not have (thermal band 4 on a 3-band RGB) drew that image's band 1 with the thermal range; it now keeps its own display (Nathan, 2026-10-09) | viewer / stretch | medium | fixed-unverified (2026-10-09) |
| BUG-065 | Bands reloaded for a preview (stretch dialog band pick, band bar, all-viewers band load) came back RAW: a cropped / rotated / resized image was shown uncropped, its polygons off the image | viewer / stretch | high | fixed-unverified (2026-10-09) |
| ENV-001 | `canopie-env` has no Python interpreter | environment | — | open |

### COG boolean crash, windowed band math, Stage 3 speed (2026-10-07)
Nathan: "For the cogs boolean is crashing the program also do the speed work" -- "the program
has to be fast to all cases types of data and never crash" -- "use the same cog philosophy for
band math and boolean".
- **BUG-052.** `canopie/lazy_analysis.py`: the analysis image of a large raster is computed
  window by window like the COG tiles -- crop, rotation, down-size (area average), per-band
  NoData, mask polygons, band expression and appended expressions per window, equal to the
  whole-image replay (`test_lazy_analysis`, 45 cases). Steps that need the whole image
  (histogram matching, registration, classification, an up-size) still read it whole, but
  only when it fits: otherwise the image is refused with the reason (`_get_export_image`
  memory guard, 75 % of free RAM). The viewer preview holds at most 16 Mpx (was 512 MB of
  raw bytes: 9439 x 8516 -> 1.3 GB as float). Saving a polygon's size reads the header, not
  the export image. Training reads each polygon's window. Measured on the BCI COG (48031 x
  50101 x 4, 17.9 GiB, `b1>100`): no whole read anywhere, Inspect 0.1 s, CSV 86 ms/crown, ML
  15 crowns 4.8 s; on a copy of pasta (5): Apply 26 s -> 5.2 s.
- **BUG-053.** `_bm_expr_cache` entries are tied to the export key (file, `.ax` mtime,
  NoData); the background worker drops each file's entry after the file. Reproduced on the
  BCI copy: export, crop moved by (60, 30), export again in the same tab -> 220 cells of the
  110 band-math rows differed from a fresh tab (sum3 Mean 275.9 vs 346.3); after the fix 0.
- **BUG-054 / BUG-055 / BUG-056.** Thumbnails test for a windowed image before `.size`; the
  augmentation preview reads the central 1024 x 1024 window; refused images are listed in
  the completion message (background CSV, ML export, training) or a warning (foreground CSV).
- **Stage 3 speed, same bytes.** Golden CSVs (the real ExportWorker on fixed folders: QC
  fixtures, BCI 3000 x 3000 window with crop + 50 % + `b1>100` + NoData, the same windowed,
  New Folder223 139 bands, the COG with 60 crowns) are byte-identical after every change.
  - median + quantiles from one partition with numpy's own arithmetic (a zero picked where
    the data holds -0.0 and +0.0 goes back to numpy; 1,872 cases bit-equal);
  - no NaN copies in the expression evaluator when nothing is non-finite (it never returns
    the caller's memory);
  - integer / 0-1 counts without `isclose` (same comparisons) and a histogram instead of
    `np.unique`;
  - the label-band guess once per file (it sampled every band for every polygon).

  Export time, A/B on two copies of the repo, alternating, best of 3 (the machine swings
  up to 2x between minutes): COG 60 crowns 8.1 -> 3.0 s (x2.7), BCI windowed 3.8 -> 1.9 s
  (x2.0), BCI 1.8 -> 1.1 s (x1.65), fixtures 0.77 -> 0.58 s (x1.3). What remains on the COG
  is the window reads (decode + NoData) and the median/quantile partition itself.
- **Editor Apply, same results.** Saving computed the pre-resize size by running the whole
  editor pipeline (band expression included) a second time -- now crop/rotation only, the
  on-screen display state untouched; the decimated all-band preview reads strided views
  instead of an `np.ix_` gather per band per strip (mosaic, 4 bands: 0.95 -> 0.29 s, equal
  arrays). Apply on the BCI mosaic copy (b1>100): 5.9-6.2 s -> 4.4 s. The full-band load
  before an expression stays (product decision in `apply_band_expression`).
- ML export and training list refused images in their completion message too.
  Test: `test_export_speed_exactness` (23 tests, each fix revert-verified on a copy of the repo).
- **BUG-058** (Nathan, Nova pasta (5): "zooming after boolean and after return to normal rgb
  stretching ... a strange looking RGB with whites and blues and reds"). The zoom tiles map with
  the preview's own bounds (D2, Stage 1), but the stretch dialog's Apply / OK redraws the preview
  with `_compute_stretched_preview` and never recorded them, so the boolean's lo 0 / hi 1 stayed
  on the viewer for the same image and every tile thresholded R, G and B at 1. Measured on a copy:
  tile 100 % of pixels 0 or 255 per channel (preview 0.4 %); after the fix 0.3 %, same mean
  colour as the preview. Fix: the dialog path records its bounds; recorded bounds carry the
  stretch they belong to (`_stretch_signature`) and are used only by a render with that stretch.
  Tests: `test_tile_stretch` (2 new, revert-verified).
- **BUG-059** (Nathan: "on drawing polygons not rescaling to lower resolution imagery on multiple
  viewers ... it is not making the maths translations as it does on copying or rescaling").
  `add_polygon_to_other_images` (sync on) converted the drawn shape into the polygon basis
  (`polygon_basis_hw`, the raster size stretched over the displayed frame) but scaled between
  images by the DISPLAYED arrays' sizes and stamped those as `image_ref_size`. Now it scales by
  the bases and stamps the target's: the same fraction of every frame, as copy-to-root. Measured
  on copies with the real `on_polygon_drawn`: New Folder223 (139-band stack shown .ax-resized
  to 118 px, climate stack 236 px) and New Folder129 (RGB JPG .ax-resized, IR TIFFs) put the
  synced corner at 5.97 / 6.45 of the frame instead of 0.30 (off the image; after save + reopen
  nothing was drawn); fixed: 0.0 error both directions, in the record, on screen, and after save
  + reopen; New Folder213 (1 m and 3.3 m mosaics, previews 1/16 and 1/8): 0.0 too.
  Test: `test_polygon_sync_scaling` (3 polygon cases + points, all 4 fail on the old code).
  Full suite after the fix: 1,675 passed, 0 failed.
- **BUG-061** (Nathan, Nova pasta (10): "not applying to the other viewers and not keeping the range
  values"). Five 4-band uint16 RGBT orthomosaics (~10400 x 5500) + an RGB; `global.ax` holds a
  project stretch of file band 4 (thermal), Absolute 7446..7546. Since today's 16 Mpx preview cap
  these images are previews -- they were loaded whole before (435 MB < 512 MB), so the 16 Mpx cap
  is what exposed it -- and four defects met:
  1. the preview's bands came from the file's `display_bands` / bands 1-3, never from the stretch
     it is drawn with: band 4 had no position, band 3 was drawn with the thermal range (black);
  2. a root / project apply loaded the stretch's bands only into the dialog's own viewer;
  3. `_stretch_to_dict` turned a band without a position into None, so the stretch was saved
     with band 1;
  4. the dialog selected the saved band by ARRAY POSITION in combos that list FILE bands, read
     combo values as positions for its range, and its last init step reset the Absolute min/max
     to the band's data range in Single + Absolute mode -- on every open, preview or not.
  Measured on a copy with the real `open_stretch_dialog` (exec replaced): before, every RGBT viewer
  black, dialog at 0..203 band 1, OK -> still black, `global.ax` band 1, reopen 0..203 / 0..214;
  after, each viewer draws band 4 mapped with the range (mean |difference| <= 0.06 / 255 against an
  independent render), dialog at 7446..7546 band 4, OK 7400..7700 all viewers, saved band 4, reopen
  7400..7700 band 4. Test: `test_stretch_dialog_previews` (5, all fail on the old code).
  Full suite after BUG-059 + BUG-061: 1,681 passed, 0 failed.
- **BUG-062** (Nathan: "when changing bands stretch should assume new number percentile (when
  absolute is on, normal percentile stops working completely)"). `_binary_stretch_bounds` (D3:
  a 0/1 band maps 0 -> black, 1 -> white) took a 2-D single-band sample's COLUMNS for channels;
  the dialog engine passes the band as a 2-D image, so one all-0 column (the orthomosaics'
  black border) made the whole band 0..1. Driven in the real dialog on a copy of Nova pasta (10)
  (shown, live apply as in the app), 12 steps -- Single / RGB, Percentile / Absolute, band changes
  between them: before, single-band Percentile wrong at every step (|pixmap - reference| ~21 / 255,
  viewer bounds 0..1); after, all 12 within 0.05 / 255. The channel count now comes from the
  bounds. Tests: `test_tile_stretch` (+3; the C0 and unit cases fail on the old code).
  Also measured: the thermal orthomosaics are 77-81 % background 0 in all four bands, with no
  NoData declared, so Percentile 0.5-99.5 of band 4 is 0..7559 (real data 7356..7585): with
  `nodata_values: [0]` in their `.ax` (copy) Percentile gives 7402..7705 and matches the reference.
  Full suite after BUG-062: 1,684 passed, 0 failed.
- **2026-10-09** (Nathan: "RGB_june keep its own display ... yes; fix bug 060; fix my polygon
  manager zoom to polygon on COGs, Nova pasta (5), the zoom is drifting"), all on copies:
  - **BUG-063.** `_get_polygon_scene_rect` (polygon manager click and Zoom to polygon) mapped a
    record through `_ZoomBar._map_point_raw_to_scene`: the .ax geometry applied AGAIN (records are
    already relative to the post-.ax frame) and the result divided by the PIXMAP's size instead of
    the frame's. Pasta (5) (28315 x 25548 mosaic, 4045 x 3650 preview, 2,802 crowns): six crowns
    spread over the mosaic landed 2,456-13,883 px off the viewport centre; now 0-8 px, stable after
    tiles settle. It now uses load_polygons' own mapping (fraction of image_ref_size x pixmap).
    Test: `test_polygon_zoom` (+2, fail on the old code).
  - **BUG-060.** px_w / px_h are full-frame pixels: on a decimated preview the viewer
    (`apply_aux_modifications`) and the editor (its pipeline, its resize boxes, `_pre_resize_shape`,
    the crop inverse) now apply the export's resize factor (`ax_geometry.preview_px_resize`); a full
    image keeps the exact pixel size. New Folder213's 3.3 m mosaic: viewer 17481 x 19499 float
    (1.36 GB, 341 Mpx pixmap) -> 2186 x 2438 (21 MB); editor the same. The `.ax` origin reproduced
    on the old code: open the editor on the preview and save with no edit -> `px 19499 x 17481,
    ref 2438 x 2186`; the new code writes nothing. Test: `test_preview_px_resize` (4).
  - **BUG-064.** `_shared_stretch_for`: a root / project stretch naming a band the image lacks is
    not used for it (resolution at load / refresh / Auto, the all-viewers apply, and the clearing of
    own stretches on OK); a viewer that took an earlier live state of the dialog goes back to its own.
    Pasta (10) copy, all-viewers thermal band 4 7400..7700: RGB_june unchanged (0 difference, with
    and without an own stretch) in the session and after reopening; the five thermal mosaics show
    band 4 (<= 0.06 / 255).
  - **BUG-065.** `reload_preview_bands` now applies the .ax as the loader does (found when
    RGB_june, cropped, came back 3746 x 2006 instead of 447 x 350).
  Tests: `test_stretch_dialog_previews` (+4), all fail on the last commit's code.
- **Final review before the commit (2026-10-07)**, the whole uncommitted tree read for bugs; fixed,
  each with a test that fails without it (revert-verified on a copy of the repo):
  - ML training on a windowed raster: a polygon whose window is over 1 GB raised MemoryError out of
    the Train slot (an abort); it is now skipped and listed in "Training Complete";
  - the background export kept every file's analysis image in `_export_image_cache` -- a windowed one
    is sized as a small object but holds up to ~0.75 GB of windows / samples; dropped per file now,
    like the foreground export;
  - a band choice from the View menu was overridden by the preview-band rule of BUG-061 (the stretch's
    bands won the reload); the choice now becomes the image's stretch;
  - the stretch dialog loaded a missing band into the VIEWER while opening, so Cancel left the viewer
    holding it under its old picture; the viewer gets it with the first Apply / OK;
  - `band_layout` documented the derived-band order wrongly (label before appended bands).

  Left as noted, not fixed: synced points are not snapped to the target's export pixel; a project-scope
  stretch reloads each preview in turn on the GUI thread; `_stretch_signature` / `_ensure_preview_holds`
  duplicate field lists / reload logic found elsewhere.
  Golden CSVs byte-identical; full suite 1,688 passed, 0 failed.
- **Full suite** in parts, after the last code change (BUG-058, 2026-10-07): a-o 1,052 passed /
  9 skipped, p-s 526 passed, the last 7 files 94 passed -- 1,672 passed, 0 failed. Earlier runs
  today: BUG-057 once (a test defect, fixed) and BUG-051 once (3/3 on rerun).
- The "test_polygon_batch_layer got 23 s slower" lead was noise: the same code ran 32.8 s and
  56.9 s back to back, the session-start code 48.1 s and 25.1 s; under cProfile they differ by 2 %.

### Stage 1 — the earlier plan's bugs, one NoData rule, one band model (2026-10-06/07)
Nathan: "full revision on export_csv and ML csv extraction and ML training ... and how they
interact with my viewer and ax files -- I need everything 100% correct" -- then "correct the
bugs from the earlier plan first". Decisions (Nathan, 2026-10-06): derived bands named by their
expression text; a band that is NoData keeps its row (empty); ML x,y in the export frame;
classification exports bands + expression + appended + label; NoData **per band, GDAL style**,
one rule everywhere. Every fix below has a QC test that fails without it (revert-verified, the
reverted file restored byte-identical).

**One analysis image** (`canopie/analysis_replay.py`, `nodata.py`, `band_layout.py`): float32,
FILE band order, NoData as NaN per band (`.ax` values + the raster's own tag), derived bands
named, registration once. The CSV, the ML export, ML training and Inspect read it; the viewer's
data equals it NaN for NaN.

- **BUG-015 (NoData).** CSV gated R/G/B by band 0, ML masked whole pixels and ignored the
  GDAL tag, the viewer held fill values whole-pixel, Inspect blanked the readout for an
  expression, the viewer blended NoData through registration. Now per band everywhere; the
  display whitens a pixel only when a band it SHOWS is NoData (`_with_display_nodata`). A
  tag-only file with no `.ax` gets the same NaN in the viewer. Ground truth regenerated
  (`per_band`; JSON only, images untouched; a regeneration with the old generator was
  byte-identical first). Tests: `test_contract_ax_consumers` (NoData x crop / rotate / resize
  both ways / hist before and after a shrink / expression / band math / appended bands),
  `test_viewer_nodata_display`, `test_hist_match_nodata`, `test_scene_stats_nodata` (flipped,
  as it was written to: explicit values are now additive).
- **BUG-048.** A band that is NoData keeps its row with empty Mean and Pixel Count 0 (points
  and polygons, eager and lazy).
- **BUG-049 (new).** The export dialog's NoData field was prefilled from whichever `.ax` it
  found first and REPLACED each image's own NoData; the full read then ignored it. Now each
  image keeps its own rule and the field adds to it (cache key includes it), on both read
  paths; no prefill. Test: `test_export_dialog_nodata`.
- **BUG-009 / BUG-025 (names, Label).** CSV / ML channels from the band layout: R, G, B,
  band_4 ..., expressions by their text; ML columns by name, not position. The label band is the
  classification's only: a guessed 0/1 band (a boolean expression) no longer replaces its own
  row by a "Label" row; "Count 0/1" still counts it on its own row. Tests:
  `test_process_polygon_csv` (2 new), `test_ml_manager_csv_and_training`. The band bar's
  buttons were array positions labelled 1..C (button 1 = file Blue on a cv2 load, the
  expression = "4"): now file band numbers, the expression by its text. Test:
  `test_band_bar_names`.
- **BUG-010 (Inspect).** Reads the EXPORT pixel under the click (`inspect_export_pixel`): the
  viewer's own data when it is the analysis image, else the analysis image (one pixel of a
  lazy read); CSV names; no second "index"; per-band NoData; the status bar shows the export
  pixel's coordinates. Test: `test_inspect_export_pixel` (decimated preview, eager and lazy).
- **BUG-012 / 013 / 014 + training.** ML export windows from the mapped points and rasterises
  with the CSV's vertex rule. Training: per-image channel order from its own export info (M2),
  NoData as NaN per band with NO literal matching against features (T3, also on lazy reads),
  an expression feature NoData where a band it reads is, polygons with the CSV's rule (M3), the
  mask-polygon loop no longer relabels rows (T1). **BUG-050 (new):** expressions on a lazy read
  crashed. Sampling moved verbatim into `_sample_training_file`. Test: `test_ml_training_rows`.
- **BUG-016 (registration).** The matrix is computed on the processed frame (crop + rotate +
  resize). The viewer/editor applied it on the raw frame with only the crop translation (off by
  up to 26 of a 0-140 range under a 90 deg rotation, measured), the analysis replay after the
  geometry, the image export twice. Now one matrix (`ax_registration.raw_frame_matrix`: exact
  conjugation through ax_geometry, edge coordinates), raw-frame warp, NoData NaN before the
  warp, used by viewer, analysis and image export (the editor's preview of it stays disabled,
  by Nathan's earlier request). Integer shifts under crop + rotation come out exact. Tests:
  `test_contract_ax_consumers` (registration cases, image export once).
- **BUG-011 (classification).** The viewer's replay had its own older feature builder and wrote
  unclassified as the first NoData number or 0 (= class 0); appended classification predicted
  NaN rows and wrote 0. Both use the export's classifier (`classify_label_band`): unclassified =
  NaN, same label per pixel as the export, the tab's own model (`model_bundle`) before the
  class-wide one. Tests: `test_classification_pixel_eligibility` (3 new),
  `test_classification_feature_stack_perf` (updated).
- **BUG-024 (red/blue).** A stretch saved positions in the loaded array (B,G,R first for cv2).
  Saved now as FILE bands (`band_basis: "file"`, derived bands by name) and mapped to each
  array when drawn; a legacy natural-colour triplet (2,1,0 / 0,1,2) reads as natural colour;
  root / project stretches follow each image's order; the expression pin stores the band by
  name. Real mosaic window: legacy 2/1/0 drew file Blue as red (corr 0.90) -> 0.9999. Test:
  `test_stretch_band_basis` (+ `test_band_expression_display_persists` updated).
- **BUG-007 D2/D3.** A zoom tile maps with the preview's own stretch bounds; a band holding only
  0 and 1 maps 0 -> black, 1 -> white in both engines. Test: `test_tile_stretch`.
- **BUG-008 (editor).** A reopen served from the viewer's or the editor's cache now carries the
  channel order (it fell back to B,G,R: b1 = Blue on a stack); the `_export_cache` is no longer
  used as a raw source (it holds the analysis image); `orig_size` = the raster header on open and
  on every Apply; editor NoData uses the real order; the editor's expression is NoData where a
  band it reads is; the expression preview honours `band_enabled`. Test: `test_editor_open_state`.
- **BUG-020 (size helpers).** `_raw_hw_quick` had no `self` (TypeError on every call); the
  copy/replicate size helpers trusted `orig_size`, simulated the geometry their own way and
  returned the RAW size after any crop. Now `post_ax_frame_hw_for_file` (ax_geometry on the
  header). Audit of the `image_ref_size` writers: each stamps the frame its points were scaled
  to, so stored fractions are exact; the remaining "mixed bases" are cosmetic. Test:
  `test_post_ax_size_helpers`.
- **BUG-021 / BUG-023** fixed earlier this session (undo keeps `.trash`; no `self` in the
  staticmethod). Test: `test_undo_trash_and_mask_replay`.

**Real data** (windows cut from the real rasters into the scratchpad; sources read-only):
BCI mosaic 3000 x 3000 RGBA around 55 real crowns of Nova pasta (5), `.ax` crop + 50 % +
`b1>100` + NoData `[0, "b1<20"]`: analysis image == an independent numpy/cv2 computation (NaN
equal, max |diff| 0); names R, G, B, band_4, b1>100; viewer data == analysis; CSV per crown
== numpy (every band, count and mean; 38 ms/crown); ML Average == CSV; Inspect == analysis on
21 pixels; colours right. New Folder223 400 x 400 x 139 window: a point gives 139 rows,
band_19 (NaN on 99.1 %) empty, the other 138 == the export pixel; a polygon 139 rows.
(The whole-mosaic export with band math was not run: 1.4 GB RAM free, it reads 2.9 GB.)

**Full suite** in parts (final run, 2026-10-07): a-o 984 passed / 9 skipped, p-s 526 passed,
the last 7 files 91 passed -- 1,601 passed, 0 failed. Intermittent in earlier runs: BUG-051
(pre-existing) and BUG-029 (1 of 4).

### Phase 0 step 1 — measured on a copy of `C:\Nova pasta (5)` (2026-10-05)
Reopened state (`.ax` holds `b1>100` + `viewer_stretch` rgb 2/1/0). Ground truth read straight
from the TIFF strips at the same pixels.
- **Geometry and values agree** between viewer and CSV for `dsa`: per-channel means
  R 81.69 / 81.75, G 101.11 / 101.16, B 55.73 / 55.77, boolean 0.386 / 0.388 (viewer = every
  3rd pixel, 14,335 px; CSV = full res, 127,746 px).
- **Viewer array** (1879, 1502, 4) float32 = file b1 (R), b2 (G), b3 (B) in true order + the
  boolean, which matches `FILE b1 > 100` (b1 = Red is correct). **No alpha.**
- **BUG-009 reproduced:** CSV channels `R, G, B, band_4` (= alpha, mean 255), `band_5`
  (= boolean, mean 0.388); viewer and Inspect call the boolean band 4.
- **BUG-010 reproduced:** Inspect returns names `b1, b2, b3, b4, index` — `b4` is the boolean,
  `index` repeats it; values are preview pixels.
- **BUG-024 reproduced:** screen red correlates 1.000 with array channel 2 (file Blue), screen
  blue 1.000 with channel 0 (file Red). Cause: the stored `viewer_stretch` r/g/b = 2/1/0 applied
  to a native-order (RGB) preview. Subtle on green canopy; bare ground shows it.
- **BUG-025 reproduced:** an extra `Label` row with empty Mean/Median per polygon.
- Viewer load 5 s; harness `p0_step1_bands.py` (session scratchpad).

### Phase 0 step 2 — boolean persistence, measured (2026-10-05)
Start: copy with `band_expression` removed, `viewer_stretch` rgb 2/1/0 kept. Editor Apply
`b1>100` driven through the real `ImageEditorDialog` (`edit_image_viewer`).

| step | on screen | in-memory stretch | `.ax` viewer_stretch |
|---|---|---|---|
| open | RGB | rgb 2/1/0 | rgb 2/1/0 |
| right after Apply | **boolean** | single, band 3 | **rgb 2/1/0** (written back) |
| zoom ×4 (high-res tile) | boolean | single, band 3 | rgb 2/1/0 |
| refresh | **RGB — reverted** | rgb 2/1/0 | rgb 2/1/0 |
| reopen | RGB | rgb 2/1/0 | rgb 2/1/0 |

- **BUG-007 reproduced** (D1): the boolean is still in the viewer array after refresh
  (1879×1502×4) — it is only hidden, because the "show it" choice is memory-only while the
  `.ax` keeps the pre-editor RGB stretch. Zoom did not trigger the revert in this sequence (D2/D3
  not reproduced yet).
- **BUG-008 narrowed:** "b1 = Blue on a second open" **refuted** — channel order `rgb` on 1st and
  same-tab 2nd open, true shape (25548, 28315, 4) found, boolean == Red > 100 on 100% of pixels.
  What holds: the stale `orig_size` {9439, 8516, 3} is never corrected by any later Apply.
- Editor numbering after band math: (1409, 1126, **5**) = file b1-b4 (alpha = b4) + expression
  as b5 — a third numbering besides viewer (expression = b4) and CSV (`band_5`). Part of BUG-009.

### Phase 0 step 3 — shapefile import + clear, WITH the current crop (2026-10-05)
BCI crown shapefile imported through the real `PolygonManager.on_import_shapefile` into a copy
of the project as it is now (crop active). Only **276** crowns fall inside the crop.
- Import finished in 26.8 s; items visible, records and files 276; **stable after refresh** —
  "disappears after import" **not reproduced** in this state.
- Clean Vector: items, `viewer.polygons`, records and files all 0, trash 276, screen back to
  the pre-import image (0.02% pixel difference); nothing returns after refresh or reopen —
  "still visible" **not reproduced** below the 800-polygon batch threshold.
- **New:** reopening the project **empties `.trash`** (276 → 0), so undo data does not survive a
  session (related to BUG-021).
- Open question: whether the pre-existing `dsa` polygon survives an import.
- Next: replay the real 10:24 state — no `.ax`, all 5,601 crowns, above the batch threshold.

**Replay of the 10:24 state (no `.ax`):** one import = **2,802** polygons (2,801 crowns + `dsa`);
tile drawing active (14 tiles, items hidden). Stable after refresh; Clean Vector cleared
everything, screen included. **Neither symptom reproduces from a single import.**
- Nathan's trash held 5,601 = 2 × 2,801 − 1 → he most likely **imported twice**. Suspect: the second
  import removes same-name scene items but leaves them in `viewer.polygons`, which feeds the
  tile drawing (`polygon_manager.py:3622-3627`) → imports "vanish" and ghosts survive Clear.
- **Double import (`p0_step3c_twice.py`, no `.ax`, 2,802 polygons):** the second import leaves
  `viewer.polygons` at **5,603** while the scene holds 2,802 and records/files 2,802; batch tiles
  go 14 → 30 (they are built from `viewer.polygons`, `image_viewer.py:5003`, so they paint the
  removed items too). Refresh resets it to 2,802. Clean Vector then clears everything; nothing
  returns after refresh or reopen. → **BUG-026**.
- Once in this run, the pre-existing `dsa` was **not drawn on open** (record + file present, 0
  scene items). 8 further opens of a fresh copy (`p0_step5_open_repeat.py`): **0/8** missing.
  Rare; not explained by the app log. Kept as an observation, not a bug entry yet.

### Phase 0 step 3d/3e — deletes on a REOPENED project (2026-10-05)
With the crop, 275 of the 276 imported crowns have record `name` ≠ group key (e.g. group
`oleifera_7114`, name `7114`). On reopen `load_polygons` names items by `name`
(`project_tab.py:8410`); the import named them by group.
- **Polygon Manager → Delete** on `oleifera_7114` after reopen: record and file removed,
  **item still in the scene and visible** (also after zoom ×4 and back); gone only after a
  refresh. → **BUG-017 confirmed** (for PM delete; `DeletePolygonCommand` matches the group key,
  `project_tab.py:3394-3410`, `_flush_bulk_delete_pending` `:22621-22625`).
- **Clean Vector** and **Clear All** after reopen: scene 0, `viewer.polygons` 0, records 0,
  files 0, also after refresh — **work** (step 3e). The "Clear still visible" report is so far
  only reproduced through PM delete.
- Same-session delete after a double import: target removed (both refs pruned by name), but
  274 stale `viewer.polygons` entries still point at items no longer in the scene (BUG-026).
- `edit_selected_polygons` (`polygon_manager.py:~4030`) also removes items directly without
  pruning `viewer.polygons`, and only the first match (`break`).

### Phase 0 step 4 — ML Manager extraction vs CSV and the TIFF (2026-10-05)
`MachineLearningManager.export_csv_data` on `dsa` (crop + `b1>100`):
- Average: R 81.750, G 101.160, B 55.765, `band_4` 255.0 (alpha), `band_5` 0.3885 — **identical
  to the CSV export** (81.75 / 101.16 / 55.77 / 255 / 0.388).
- All Pixel Values: 127,746 rows; per-column means identical. x,y match the TIFF R,G,B exactly
  for **300/300** sampled pixels when read as *cropped-frame* coordinates (raw = crop offset ×3 +
  x,y), 4/300 as raw coordinates → ML **x,y are in the cropped full-res frame**, not the original
  image's pixel grid → **BUG-028**.
- Labels `band_4` / `band_5` — same naming problem as the CSV (BUG-009).
- BUG-012 (M1 lazy window) is not exercised here: `band_expression` disables the lazy path.

### Phase 0 step 6/7 — polygon imports from another project (2026-10-05)
Source copy holds `dsa`; target copy has it removed.
- **"Import Polygons from Project"** (`on_import_polygons_from_project`, `polygon_manager.py:1724`):
  stops at once with "Images Folder Missing: Open a folder in the viewer first" — for this project
  `current_folder_path` is the TIFF's own path, not a folder. The pipeline also matches polygons to
  images by EXIF GPS (KD-tree), which a single orthomosaic does not have. → **BUG-027**.
- **"Import Polygons (JSON)"** (`import_polygons`, `:4366`): `dsa` lands in the same place as in the
  source (scene bbox identical), survives refresh and reopen. But the file is stamped
  `image_ref_size` **1502×1879** (the crop preview) where the source and every shapefile import use
  raw **28315×25548**; after refresh the in-memory record says 28315×25548 while the file still says
  1502×1879. → **BUG-020 confirmed** for this writer. (Placement is right today only because the crop
  has not changed since; see BUG-022.)

**BUG-007 … BUG-023** come from a code read on 2026-10-05 against `C:\Nova pasta (5)`; causes
with file:line are in the plan (`~/.claude/plans/noble-waddling-fairy.md`, "Findings" table).
Each is confirmed or refuted by the Phase 0 real-data measurement before any fix, and the
measured numbers are recorded here.

### BUG-007 (D1) — fix, 2026-10-05
- **Cause:** `edit_image_viewer` restored the pre-editor `viewer_stretch` ("rgb") after Apply; the
  post-Apply single-band pin lived only in memory, and `display_image_group` resolves the `.ax`
  stretch first on every later load.
- **Fix (`project_tab.py`, `edit_image_viewer`):** an Apply that sets or changes the active band
  expression stores `viewer_stretch` = single, last band, tagged `auto_expression_pin`, and keeps
  the user's stretch as `viewer_stretch_before_expression`; an Apply that switches band math off
  restores it. A stretch saved from the dialog drops the tag (`_stretch_to_dict`), so a user's own
  choice is never auto-reverted. New helper `_ax_active_expression` honours `band_enabled`; the two
  loader checks that ignored `band_enabled` (`display_image_group`, `_render_with_viewer_stretch`)
  now honour it (otherwise an RGBA image with band math off would pin the alpha band).
- **Test:** `qc/test_band_expression_display_persists.py` (2 tests), each branch revert-verified.
- **Real data** (`p0_step2_boolean.py`, fresh copy of `C:\Nova pasta (5)`): boolean on screen after
  Apply, zoom tile, refresh **and reopen** (before: RGB after refresh and reopen); `.ax` stores
  single / band 3.
- Full suite: 1409 passed / 0 failed / 3 skipped (second run; the first run hit BUG-029).
- Still open: D2/D3 (tile stretch on a mostly-True tile), BUG-009 band numbering.

### BUG-017 / BUG-019 / BUG-026 — polygon identity, 2026-10-05
- **Evidence (C:\Nova pasta (6), read-only):** the 19:54 Clear trashed **5,601** sidecars =
  2,801 crowns + **2,800 duplicates** filed under the bare tag (`107109`, `23530.0`...), each with
  **no properties** -- "properties are kept at import, but once I move a polygon they vanish".
  Later imports used the species as the name field (`ekmanii` for group `ekmanii_23`).
- **Cause:** one field did two jobs. `item.name` is the identity every delete / sync / save /
  select / Edit-properties path matches against the group key, but `load_polygons` (and
  copy-between-roots) set it to the record's label. After any refresh or reopen: PM delete
  missed the item; a moved crown was saved by `on_polygon_modified` as a new group named by its
  label, with no properties; `update_all_polygons` / `on_polygon_modified` also overwrote the
  record's name with the group key; the load-time de-dup skipped every crown after the first
  sharing a label (species) -> **BUG-019**; Clean Vector / Clear All queued the label for scene
  removal, so crowns drawn by the importer (named by group) stayed on screen.
- **Fix:** items are named by group key everywhere and carry `display_name` (label text only,
  `EditablePolygonItem.label_text`). `load_polygons`, the shapefile importer and
  copy-between-roots set it; `DeletePolygonCommand` matches and restores by group;
  `on_polygon_modified` / `update_all_polygons` keep the record's name; mask lookups accept
  either spelling. BUG-026: `ImageViewer.prune_detached_polygon_records`, called after every bare
  `removeItem` path (re-import, Edit Selected, JSON import, copy-between-roots).
- **Tests:** `qc/test_polygon_identity_vs_label.py` (4) -- each fix revert-verified.
- **Real data** (`lod6_identity_real.py`, `lod7_clear_after_import.py`, copies of (6)): a move
  keeps group / name / all 31 shapefile attributes, in memory and on disk, and creates no `6740`
  group; PM delete is gone at once; Clean Vector after reopen 2,800 -> 0 items (before the command
  fix: records 0, **2,800 items still on screen**); Clear All and Clean Vector right after an
  import, no refresh: everything 0, at every zoom.

### BUG-030 — import freeze: double writes through the pure-Python encoder, 2026-10-05
- **Profile** (2,801 crowns into a copy of (6)): 104.7 s total, **93.4 s on the GUI thread**,
  88 s of it in `json/encoder.py`. Every sidecar writer used `json.dump(..., indent=...)` (always
  the pure-Python encoder), and every file was written twice -- `save_incremental` (64 s) then the
  debounced `_flush_dirty_polygons` (24 s), which also dumped the raw record without the `lod`
  pyramid over the first write.
- **Fix:** `polygon_lod.dump_sidecar` (C encoder via `json.dumps`, compact, atomic temp+replace)
  for all seven sidecar writers and the overview; one payload builder `_sidecar_payload` shared
  by both writers; a save clears BOTH dirty sets.
- **Two more costs found after the first fix:** (1) every sidecar stored a 9-level `lod`
  pyramid -- 41% of the bytes (55 of 134 MB) and 2.5 s per import -- that nothing needs on disk
  any more (the overview and the zoom refiner decimate on the fly); it is no longer written
  (old sidecars that carry one still read fine). (2) The import's `processEvents()` every 100
  crowns repainted the viewer, and above 800 polygons each repaint rebuilt every zoomed-out tile:
  10 rebuilds, ~4 s. Viewport updates are now suspended for the add loop (as `load_polygons`
  already did) and restored with one repaint.
- **Result (copy of (6), 2,801 crowns):** **18.1 s total, 6.7 s on the GUI thread** (was 104.7 s /
  93.4 s); sidecars 86 MB (was 134 MB). Remaining GUI time is mostly file IO (4.1 s); the import
  worker thread (feature matching 4.1 s, reprojection 2.4 s) does not block the UI. Moving the
  writes off the GUI thread needs re-entrancy guards (autosave / flush timers) and a drain on
  close -- not done yet.
- **Tests:** `qc/test_polygon_sidecar_writes.py` (2) -- both revert-verified.

### BUG-031 / BUG-032 — LOD follows the zoom (QGIS-style), 2026-10-05
- **Cause:** `polygon_lod.level_for_scale` was never called; reopened crowns kept the level-4
  overview (16 record units, ~1 m on the 7 cm ortho) at every zoom. Edit Vertices, Copy and
  Replicate used that simplified outline as data (BUG-032).
- **Fix:** after a zoom/pan settles (150 ms, from `paintEvent`), `_refine_visible_polygon_lod`
  swaps the crowns IN VIEW (scene BSP query) to the coarsest level <= 1 device px, or to the exact
  geometry. Device px per record unit = view scale x pixmap/record basis, so a 1/16 COG preview is
  handled like a full-resolution image. Full geometry still on disk is read on an 8-thread pool
  (Windows `open()` measured ~5.5 ms per fresh sidecar); 40 ms GUI budget per pass, progressive
  passes. Display only (`is_lod_geometry` kept, record untouched, never written back). Records
  already in memory (right after an import) are drawn from the overview too.
  `ensure_full_geometry` guards Edit Vertices / Copy / Replicate.
- **Real data:** ortho (2,801 crowns): 1:4 -> 205 crowns at level 2 in passes <= 38 ms; 1:1 and
  2:1 -> every crown in view exact (33,928 / 11,653 vertices); repaint 6-20 ms. COG twin (3,504
  crowns, 50101x48031, preview 3001x3131): 1:16 overview kept for 1,428 crowns; 1:4 -> level 2 in
  passes <= 26 ms; 1:1 and 2:1 exact.
- **Tests:** `qc/test_polygon_lod_follows_zoom.py` (3) -- refine and the Edit-Vertices guard
  revert-verified.

### BUG-042 .. BUG-048 — round 2 on points, polygons and the editor, 2026-10-06
Nathan: "after cropping and rescale I still see bigger drawing than the pixels I actually have
(C:\Nova pasta (5)); polygons are disappearing on resetting the image editor dialog; opening
the image editor dialog still too slow; C:\New Folder223: I drew like 1 pixel after rescaling
and it is returning 138 -- these are COGs, make sure it works in all other types of images".
Approved: polygons follow the crop like points, tested on backup copies.

- **BUG-042 (marker vs visible pixels).** The zoom tile replayed resize as "absorbed by the
  display scale": it read and drew source-resolution pixels whatever the export's resolution.
  On pasta (5) (566 x 511 export of the 28,315 px mosaic) one exported pixel = 50 x 50 source
  pixels, so the (correct) one-pixel marker sat over 2,500 visible pixels. **Fix:** when the
  preview already is the export's resolution no tile is requested; otherwise the request is
  widened to whole exported pixels and the window is area-averaged onto the export grid
  (`_area_resample`, cv2.INTER_AREA's rule, valid mid-cell; band math after it, like the
  export). **Measured** (copy, crop + 50 %): one tile pixel = 0.33333 display units = one
  exported pixel, origin on the grid, tile data vs `_get_export_image` mean |diff| 0.0000 (max 1,
  uint8 rounding), marker = that pixel; current pasta (5) `.ax`: no tile.
- **BUG-043 (polygons follow crop / rotation).** `_reproject_after_ax_change` now moves every
  record (polygons and points) of each image whose crop / rotation changed, exactly
  (ax_geometry, numpy), after zipping the affected sidecars + overview to
  `polygons/.reproject_backup/` (newest 5 per image; no backup, no rewrite). A change of
  resize alone is skipped (records are fractions of the frame). Lazy crowns are transformed
  file to file and stay lazy. Found while testing: the image is keyed under two path
  spellings on the BCI project, and a per-spelling snapshot moved every crown twice
  (18,036 px off) -- one entry per image now.
- **BUG-044 (Reset).** Reset Image / Root / All call `_after_editor_ax_write` (re-project)
  before refreshing; Cancel after a Reset re-projects too; the snapshot is updated so OK does
  not move them again. `update_all_polygons` skips a viewer whose items still show the old
  frame (`_ax_frame_stale`, cleared by `load_polygons`) -- it read them back over the
  re-projected records. Reset Image refreshes only its viewer, and the view carries the same
  ground at the same ground scale through the change (`_carry_view_through_geometry`).
  Measured on a copy without Fix Zoom: raw (12538, 11873) at 0.5 raw px per screen px before
  Reset, (12538, 11872) at 0.5 after. With Fix Zoom on (pasta (5) has it) Fix Zoom places the
  view as before: the whole image fitted, crowns drawn.
- **BUG-045 (editor open).** `_get_all_polygon_names_in_project` / `..._for_current_image`
  use `polygon_lod.point_count` (no sidecar read); previews go through
  `raster_reader.read_preview` (shared with the viewer load and band changes, keyed on the
  source file, copies handed out). A decimated preview is no longer put in
  `_raw_image_cache` (served from there, the editor lost its channel order: part of BUG-008).
- **BUG-046.** Windows' scanner reads every new file on its first open (500 just-written
  sidecars: 9.7 s, 0.1 s twenty seconds later). Bulk writers keep the text they wrote
  (`dump_sidecar(..., remember=True)`), served while the file's mtime + size are unchanged;
  the backup zips that text. Reset after a crop: 90 s -> 19 s on a fresh copy.
- **BUG-047.** Point mode: every added point (click or drag) snaps to its exported pixel and
  a pixel the shape already has is skipped.
- **BUG-048 / "returning 138"** (New Folder223, measured on a copy): the drawn point exports
  138 rows with Pixel Count 1 -- one row per band; the stack has 139 bands and band 19 is
  NaN at that pixel (and on 99.9 % of the image), so its row is omitted. Not changed.
- **Timings** (2,802 BCI crowns, fresh copies -- every first read pays the scanner):
  crop applied through the editor 29 s, Reset 19 s; floor ~11 s on this machine
  (JSON parse 1.5 s + encode 3.6 s for 3.9 M floats, plus file IO). Resize-only changes: 0 s.
- **Tests:** `qc/test_polygons_follow_ax.py` (38; 6 image types x 4 geometry changes), every
  fix revert-verified (12 reverts); `qc/test_point_pixel_snapping.py` isolated from the shared
  project folder. Full suite: 1,529 passed / 0 failed / 9 skipped (run as test_a..test_o,
  test_p..test_s, and the last four files one by one -- see BUG-035).
- **Real data again:** pasta (6) copy, a point + 2,801 crowns: crop -> 2,802 records moved, the
  point's CSV pixel maps back to raw (11609, 12001) = the clicked pixel; then resize 50 % ->
  0 records rewritten, point still in its 2 x 2 block.

### BUG-039 / BUG-040 / BUG-041 — points: one pixel, follow the crop, ML agrees, 2026-10-06
- **Report:** "points completely buggy on rescaled and cropped images, not rescaling to the new
  image size"; wanted back: click a pixel -> that pixel highlights -> that pixel is drawn.
- **Measured** (`pt1_points.py`, `pt2_points_follow.py`, copies of `C:\Nova pasta (5)` / `(6)`):
  the marker covered 18x18 export pixels (resize 50 %) and 28x32 (crop + px_w 1000), not one; CSV
  and ML read the same pixel as each other, but clicks were not snapped, so a click near a pixel
  edge was ambiguous. A crop applied after drawing moved a point from raw (11609, 12001) to
  (11719, 12462).
- **Causes:**
  - BUG-039: 5ec209c (2026-09-02) drew the marker `raw_to_pixmap_scale` preview units wide (the
    ratio used as a width, so ratio^2 pixels), and that ratio came from the raw file size, not
    the export frame, so crop and resize were ignored.
  - BUG-040: point records are stored in the post-`.ax` frame stretched to the raw size (P6), so a
    later crop / rotate / resize reads the same numbers in a different frame.
  - BUG-041: ML mapped points with `_map_points_scene_to_image` and clamped to the image.
- **Fix:**
  - `canopie/ax_geometry.py` (new): the frame plan of `_apply_ax_to_raw` (rotate-first decision,
    crop rounding, rot90 k, every resize mode) and forward / inverse point maps.
  - `ProjectTab.export_frame_hw` (cached on the `.ax` path, mtime and size); the viewer's
    `_raw_to_pixmap_scale` is now export px per preview px.
  - Viewer: point clicks snap to the export pixel centre, hover highlights that pixel, the marker
    is the cell that contains the point (1/k preview units), and a drag snaps on release.
  - `edit_image_viewer` snapshots each point record's `.ax` geometry before the editor and, once
    it is accepted, re-projects the point records (stored -> old frame -> raw -> new frame).
  - `load_polygons` skips drawing a point outside the frame instead of clamping and resaving it.
  - ML `_point_pixels`: the same floor as the CSV, no clamp, for both export and training.
- **After (same harnesses):** marker exactly 1.00 x 1.00 export px in every setup; CSV pixel = ML
  pixel = marker pixel; crop applied after drawing -> CSV pixel maps back to raw (11609, 12001) =
  the clicked pixel; then resize 50 % -> the 2x2 raw block that contains it.
- **Not covered:** polygons (same P6 frame problem, BUG-022): re-projecting them on a crop change
  rewrites every crown file of that image, so waiting on Nathan's go-ahead. `ax_geometry` does not
  model registration, histogram match or band math (documented in the module).
- **Tests:** `qc/test_ax_geometry_exact.py` (41) and `qc/test_point_pixel_snapping.py` (7), every
  fix revert-verified; `qc/test_point_marker_size.py` corrected to one export pixel.

### BUG-037 — navigation stalls, 2026-10-05
- **Report:** "still too slow to navigate the same cog", mainly after applying the stretch (change
  bands). Measured with real wheel/drag input and a 5 ms heartbeat (`nav1.py`) on copies of
  C:\Nova pasta (6) (BCI mosaic: 28315x25548 uncompressed, ONE ROW PER STRIP, no overviews) and the
  COG twin. The stretch itself made little difference; navigation was slow in general.
  Before: 8 zoom notches -> 6.9 s of GUI stalls (longest 1.5 s), 12-step pan 1.8 s, pan at 1.3x
  3.1 s; sharp detail ~3 s after input. COG similar (6.3 s / 4.5 s).
- **Causes / fixes:**
  1. Sharp viewport tiles were requested at ANY zoom above 1.05x fit, though the 1/3-res preview
     out-resolves the screen up to zoom 1.0; with no overviews each tile was read, stretched and
     pixmapped at FULL resolution (8,900x5,500 for a 1,600x1,000 view: LUT 230 ms + pixmap 190 ms,
     the pixmap built on the worker holding Python's lock). Now: no request until > 1.1 device px per
     preview px (`_HIGHRES_MIN_DEVICE_PX_PER_PREVIEW_PX`), none at all when the preview is full-res;
     reads at screen resolution (`TiledRasterReader.read_window(step=)`, strided on memmap, tiles and
     strips -- decodes only the sampled strips); tiles cross threads as QImage
     (`_qimage_from_uint8`), pixmapped on the GUI thread.
  2. LOD refiner swapped crowns for sub-2-px gains (one sidecar read per crown in view, 8 parser
     threads starving the GUI): now hysteresis (`_LOD_MAX_ERROR_PX` = 2), cached per-item scale,
     2 reader threads.
  3. Per-vertex Python loops: `qpolygonf_from_xy` / `xy_from_qpolygonf` fill/read QPolygonF from
     numpy (0.012 vs 1.41 ms per 1,200 vertices; self-checked at import) in tile paths, item caches,
     refine swaps, import drawing and load_polygons. Per-paint `_batch_source_items()` count and
     `isUnderMouse()` per crown replaced by a count and a hover flag.
- **After** (same harness): 8 zoom notches 0.34 s of stalls (longest 0.14 s), pan 0.27 s, pan at 1.3x
  0; COG 0.24 s / 0 / 0.32 s; sharp detail 0.25-0.55 s after input; input handled in real time.
- **Tests:** `qc/test_navigation_speed_contracts.py` (strided == sliced reads on every TIFF fixture
  and generated strip files; tile gate; refine hysteresis; exact numpy<->QPolygonF) -- gate and
  hysteresis revert-verified.
- Not done: the one-off band change still costs ~4 s (preview re-read) + ~2.5 s (stretch apply).

### BUG-036 — points on NoData exported on the lazy path, 2026-10-05
- Found as an order-dependent failure of `test_process_polygon_csv` (pre-existing: fails on HEAD
  too). The export cache is keyed by (path, mtime) only, so a lazy-path entry left by
  `test_lazy_eager_agreement` reached the eager test -- and the lazy path exports a point on NoData
  with the fill value: probe on ax_crop_nodata_source, eager {} vs lazy {R,G,B,band_4: 9999.0}.
  Large files (the BCI mosaic, any COG) always take the lazy path, so their CSVs carried 9999 rows.
- **Fix:** the point branch evaluates the same `_per_band_nodata_masks` rule on each point's pixel
  (the polygon branch already did, via `_build_roi_nodata`) behind the existing `nd_masks[bi][yi, xi]`
  interface. **Test:** `qc/test_lazy_point_nodata.py` (eager + lazy) -- revert-verified; clears the
  export cache on exit.

### BUG-035 — native crash late in the full QC run
- 2026-10-05, five full runs logged with `-v`: every crash is inside
  `test_temp_drawing_pen_zoom.py::test_start_drawing_with_group_name_pen_survives_zoom[50.0]`
  (exit 139, ~95%). A/B: the committed HEAD 73ab693 crashed 2 of 2 runs, the current tree 1 of 2
  (the other: 1,418 passed / 0 failed). The file passes alone 3 of 3. Needs the accumulated state
  of a full run -- likely the many viewers/scenes earlier tests leave for teardown (see the
  viewer_factory docstring on segfaults with many live viewers). Earlier "flaky native crash"
  sightings at 44% / 93% are probably the same family. Not investigated further yet.
- 2026-10-06: now also crashes with ONLY the last four files (test_temp_drawing_pen_zoom,
  test_thumbnail_options, test_vertex_edit_perf, test_wheel_zoom_repaint): HEAD 73ab693 3 of 4
  runs, working tree 4 of 4; each file alone passes. The stack is in ImageViewer's wheel-sync
  `_do_sync` timer, which only exists while the class-level Fix Zoom sync flag is on -- an
  earlier test switches it on and nothing resets it. Test isolation, not product code; open.

### BUG-029 — Fix Zoom navigation test failed once in a full run
- Full run with the BUG-007 fix, machine loaded by a concurrent real-data harness: the anchor
  walked (0.27, 0.68) → (0.2752, 0.6826) over 12 root changes, in uneven steps of ~6-20 device px.
- Not reproduced in: a second full run; 10 single-file runs under 12-process CPU load; 2 runs of
  every test file up to and including it, under load.
- The steps are far larger than the sub-pixel `QRect.center()` error BUG-002 fixed. Hypothesis: a
  late signal or timer in ANOTHER live viewer (left over from an earlier test) reaches one of the
  anchor writers that store from their own viewer — `_ZoomBar._store_scroll_position` at
  `image_viewer.py:3596` (zoom), `:3989` (wheel release), `_sync_zoom_to_all_viewers` (`:7265`).
  That would also be a route for BUG-001 in the app. Unconfirmed.

### BUG-001 — Fix Zoom: viewers other than the first drift on every root change
- **Report (2026-10-05):** with Fix Zoom on and several same-size viewers, the first viewer
  keeps the pinned position; the others drift further off with every next/previous root.
- **Where:** `ProjectTab._jump_to_index` captures the anchor from `viewer_widgets[0]` only
  (`project_tab.py:7315-7328`) → `_ZoomBar.update_fixed_center` immediately `centerOn`s the
  other viewers (`image_viewer.py:7140-7177`) → later `_reapply_fixed_zoom_after_layout`
  (`project_tab.py:25645`) → `reapply_fixed_zoom_if_enabled` → `_apply_fixed_zoom`.
- **Candidate causes (to confirm by measurement):** ordering of the early `centerOn` vs the
  post-layout reapply and viewport resizes; stale 500 ms `_get_all_viewers` cache; extra writers
  of the shared anchor; high-res overlay changing `_get_image_rect()`.
- **Existing test gap:** `test_fix_zoom_root_navigation.py` builds a single viewer only.
- **2026-10-05, synthetic reproduction (does NOT reproduce):** real `ProjectTab`, rgb_only,
  12 roots × 3 same-size PNGs (2400×1800), Fix on viewer 0, zoom 2.35, pan (0.27, 0.68),
  `next_group` ×12. Current working tree: **0 px drift in all three viewers**. Old (committed)
  `_get_view_center_ratio`: **all three drift together**, ~1.6 px per navigation (20 px after
  12) — not the reported "first stays, others drift" pattern either.
- **What the trace shows:** every viewer is **rebuilt** on each root change and positioned by
  `_apply_fixed_zoom` from the single shared anchor; the anchor is not rewritten during
  navigation. So progressive drift in non-first viewers needs per-viewer state that survives
  the rebuild (a per-file saved view, a viewer-reuse path), positioning from something other
  than the anchor, or a changing `_get_image_rect()` (high-res overlay on large rasters).
- **Next:** reproduce on Nathan's real project with his exact navigation (button / arrow
  keys / slider) and which viewer he pans in.
- Harness: `fixzoom_multiviewer_harness.py` (session scratchpad; promote to QC with the fix).

### BUG-002 — Single-viewer Fix Zoom drift (sub-pixel per navigation)
- **Cause:** `_get_view_center_ratio` read the center back with integer-truncated
  `QRect.center()` while `centerOn` sets it with the exact half-width; the capture runs on every
  navigation, so half a device pixel was re-baked each time.
- **Fix (uncommitted, earlier session):** average the mapped viewport corners
  (`image_viewer.py` `_ZoomBar._get_view_center_ratio`).
- **Test:** `canopie/qc/test_fix_zoom_root_navigation.py` (untracked; not yet in `_report.py`).
- **Remaining for verified:** commit; Nathan's confirmation in the app (together with BUG-001).

### BUG-003 — `requirements.txt` omits `pyproj` and `pyshp`
- **Effect:** a clean install from `requirements.txt` cannot import
  `canopie/shapefile_import_dialog.py` (`import shapefile`) or `canopie/spatial_index.py`
  (`import pyproj`), and `shapefile_io.py` / `project_tab.py:26605` lose CRS transforms.
  Two QC modules fail to collect (`test_shapefile_import.py`, `test_shapefile_import_dialog.py`).
- **Found:** 2026-10-05, collecting the suite in base miniconda (Python 3.12).
- **Fix:** add both to `requirements.txt` (pin versions after checking the working env).

### BUG-004 — QC report says "ALL GREEN" when 0 tests ran
- **Effect:** a run that collected nothing (here: two import errors aborted collection) wrote
  `TOTAL: 0 tests | 0 passed | 0 failed | 0 skipped` and `VERDICT: ALL GREEN`.
- **Where:** `canopie/qc/_report.py` (`QCReporter.write`).
- **Fix:** 0 tests, or a non-zero pytest exit status, must never produce a green verdict.

### BUG-005 — `os._exit` in `pytest_unconfigure` hides pytest's own summary
- **Effect:** `canopie/qc/conftest.py:97` calls `os._exit()` before pytest prints its summary,
  so collection errors are never shown (exit code 2 with no message), and when stdout is
  redirected to a file nothing is flushed — the log comes out empty.
- **Context:** the hard exit is a deliberate guard against a native access-violation crash at
  shutdown (see the conftest comment); the fix must keep that guard.
- **Fix:** flush `sys.stdout`/`sys.stderr` and print collection errors before `os._exit`, or
  move the hard exit after pytest's terminal summary.

### BUG-006 — 12 methods silently shadowed by a later definition of the same name
Python keeps only the **last** `def` of a name in a class; earlier ones are dead code, and any
fix made to them never runs. Found by AST scan of the live modules (2026-10-05). Each needs:
which copy is live, whether the dead one holds logic the live one lacks, then delete one.

| Method | Defined at | Live |
|---|---|---|
| `ProjectTab._remove_polygon_from_mask_config` | 4057, 27377 | 27377 |
| `ProjectTab._ax_path_for` | 6347, 30874 | 30874 |
| `ProjectTab.eventFilter` | 7266, 30734 | 30734 — the slider-tooltip version at 7266 (with the `print("DEBUG ...")` lines) never runs |
| `ProjectTab._index_for_root_id` | 7617, 23289 | 23289 |
| `ProjectTab._ax_path_for_fp` | 7837, 7843 | 7843 |
| `ProjectTab.get_root_by_filepath` | 25986, 29995 | 29995 |
| `ProjectTab.get_current_root_name` | 26022, 27942 | 27942 |
| `ProjectTab.construct_thumbnail_name` | 27640, 30660 | 30660 |
| `ProjectTab.shrink_or_swell_shapely_polygon` | 27684, 32257 | 32257 |
| `ProjectTab.copy_metadata_action` | 27726, 30396, 31792 | 31792 |
| `ProjectTab._setup_roots_from_single_folder_with_batch` | 31085, 31361 | 31361 |
| `ImageProcessor.extract_root_name` (loaders.py) | 621, 734 | 734 |

(The 7 same-name pairs in `image_viewer.py` are 4 lines apart — very likely `@property` getter
plus setter, which is legitimate; to confirm.)

### ENV-001 — `canopie-env` has no Python interpreter
- `C:\Users\natha\miniconda3\envs\canopie-env` holds pip-installed libraries (all built for
  cp310) but no `python.exe` and an empty `conda-meta`. Its pip launchers still point at
  `canopie-env\python.exe`, so the interpreter was removed after the packages were installed.
- Current working interpreter: `C:\Users\natha\miniconda3\python.exe` (base, Python 3.12;
  pytest, pyproj and pyshp added 2026-10-05). Repair option:
  `conda install -n canopie-env python=3.10`.

---

## Nathan's known bugs (to be added)

_List them here as they come up; each becomes a BUG-### entry above._

---

## Verified fixes since August (history)

Recorded so a recurrence is recognised instead of rediscovered. Details are in each commit.

| Commit | Date | Fix |
|---|---|---|
| 825b8c1 | 2026-08-05 | 7 data-integrity bugs in pixel extraction; QC regression suite added |
| a8a0f97 | 2026-08-05 | Stretch bar overlay vs stretch dialog data range |
| ff05b83 | 2026-08-06 | Shapefile export: mixed-CRS corruption, batched-export drop, .prj encoding |
| f42f111 | 2026-08-06 | Histogram matching: diagnostic plots, NoData corruption, silent mismatches |
| ac07261 | 2026-08-08 | COG zoom refinement ignored editor edits |
| 7beeeb9 | 2026-08-08 | COG zoom, shapefile import, polygon delete/pan/open at scale (perf) |
| 6342f8a | 2026-08-09 | Refresh trashing polygons; coordinate basis; batch rendering; drawing on COG |
| 4c3c62b | 2026-08-09 | Editor: duplicate `geo_layout.addLayout(row_crop)` |
| f64c684 | 2026-08-09 | Saving rebased every polygon onto the viewer preview |
| 3a0cba9 | 2026-08-09 | Refresh purged every polygon the last load drew |
| 6e94931 | 2026-08-09 | Dragged polygon snapped back to its old position |
| 0643129 | 2026-08-11 | One drawn polygon hid every imported one |
| d1cfcd7 | 2026-08-11 | ML thumbnails / segmentation masks read the whole raster |
| 6bdef1b | 2026-08-14 | Editor render and CSV export on multi-band rasters (+ lock/filter/colour) |
| 9e0b929 | 2026-08-15 | Polygon properties into every export path; accumulated polygon/perf/ML fixes |
| 9781913 | 2026-08-16 | Polygon zoom-to-scene-rect mapping; GPS auto-assignment on save |
| b7f8a60 | 2026-08-16 | Reverted broken data_rows pre-allocation in `save_polygons_to_csv` |
| cc9d700 | 2026-08-16 | Redundant NaN-scan in `process_polygon` stats (perf) |
| d241bdc | 2026-08-16 | CSV export: frozen pre-dialog wait; threading mode losing rows |
| 1ed6c23 | 2026-08-16 | Image editor Apply corrupting other .ax files; band expressions |
| 5802909 | 2026-08-16 | CSV export: dropped Lat/Long columns; `_valid_count` leak |
| cc3d73d | 2026-08-23 | Export band order; appended-band export failures; .ax/thumbnail/editor fixes |
| 9be5f9a | 2026-08-23 | Classify perf/crash; hist-match mode drift; stale point/polygon coords |
| f33e2ce | 2026-08-31 | b1=Red made consistent app-wide; viewer-vs-CSV classification divergences |
| 5ec209c | 2026-09-02 | Quadratic point extraction; point markers under decimation; `tile_bytes()` |
