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
| BUG-007 | Boolean band expression shows after Apply, then reverts to RGB (D1; D2/D3 still open) | viewer/editor | high | fixed-unverified (D1 committed c7008a3; awaiting Nathan's check in the app) |
| BUG-008 | Editor second open: b1 treated as Blue; `orig_size` stamped from the preview (E1) | editor | high | open |
| BUG-009 | Band numbering differs per consumer: "band 4" = alpha in CSV/ML, = expression on screen (X1) | extraction | high | reproduced |
| BUG-010 | Inspect reads the decimated preview, mislabels bands, duplicates the expression (X2) | extraction | medium | open |
| BUG-011 | Classification + band expression build different feature stacks in viewer vs export (X3) | ML | high | open |
| BUG-012 | ML lazy read builds its window from unscaled points — may read wrong pixels silently (M1) | ML | high | open |
| BUG-013 | ML `_image_cache` loses channel order — R/B swapped features on mixed roots (M2) | ML | high | open |
| BUG-014 | ML clamps polygon vertices, CSV clips — edge polygons rasterise differently (M3) | ML | medium | open |
| BUG-015 | NoData resolved differently by CSV, ML, Inspect and viewer (M4) | extraction | high | open |
| BUG-016 | Registration may be applied twice in export (M5) | extraction | medium | open |
| BUG-017 | PM Delete leaves the polygon visible after a reopen: items named by record name, deletes match the group key (P1) | polygons | high | reproduced (PM delete; Clean Vector / Clear All work) |
| BUG-018 | Refresh recreates cleared polygons from leftover scene items (P2) | polygons | high | not reproduced (steps 3, 3c, 3e) |
| BUG-019 | Imported polygons disappear: drawn directly, skip the root filter (P3) | polygons | high | not reproduced (one root; steps 3, 3c, 7) |
| BUG-020 | Polygon writers stamp mixed bases (preview vs raw); two broken size helpers (P4) | polygons | high | reproduced (JSON import stamps 1502×1879) |
| BUG-021 | Undoing a bulk delete empties the whole `.trash` (P5) | polygons | high | open |
| BUG-022 | Polygons stored in the cropped frame move off their trees when the crop changes (P6) | polygons | high | open (design; decided: store in raw pixels) |
| BUG-023 | `apply_aux_modifications` is a staticmethod but uses `self` (NameError with mask_polygon) (S1) | viewer | medium | open |
| BUG-024 | Viewer shows red and blue swapped on this project (screen red = file Blue) | viewer | high | reproduced |
| BUG-025 | CSV writes an empty `Label` row when no classification is active | extraction | low | reproduced |
| BUG-026 | Re-import leaves stale `viewer.polygons` entries (5,603 for 2,802 items) that still feed the zoomed-out tiles | polygons | medium | reproduced |
| BUG-027 | "Import Polygons from Project" cannot run on an orthomosaic project (needs a folder + EXIF GPS) | polygons | high | reproduced |
| BUG-028 | ML pixel export x,y are in the cropped frame, not the original image grid | ML | medium | reproduced |
| BUG-029 | Fix Zoom navigation test failed once in a full run (anchor drift 0.0058) | qc / viewer | medium | open (intermittent; 1 of 2 full runs) |
| ENV-001 | `canopie-env` has no Python interpreter | environment | — | open |

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
