# Changelog

All notable changes to CanoPie. Versions follow `pyproject.toml`.

## 1.2.0 - 2026-11-12

### Added
- **Annotator** (Modules menu): custom numeric, categorical and tag properties per crown, scoped to
  the project, a root or an image and optionally to certain labels; values live in the polygon's
  properties and reach CSV / shapefile exports (each property can be left out of exports). A red /
  green strip on each viewer shows what is still to fill; arrows or A / D step to the next crown;
  brush-drawn patches inside a crown give a percentage cover, never overlap between properties, are
  saved with the polygon and can be copied from the previous image.
- **Polygon Stats** for the selected polygons (viewer and Polygon Manager menus); the former stats
  window is now *Polygon Info*. Export Manager moved to the Modules menu.
- **Polygon & label appearance** (Configuration menu, and the gear beside Lock Polygons): label size
  fixed on screen or growing with zoom, colours, outline width.
- **Filter & Color**: property drop-down with hidden area / perimeter / last_modified, value
  suggestions, an attribute index cached on disk (fast re-filtering); after Apply the Polygon
  Manager and ML lists show only the matching polygons.
- Polygons record a `last_modified` date when edited.

### Changed
- The Machine Learning Manager no longer blocks the main window; its list and the Polygon Manager's
  follow clicks in the viewers, and clicking a row (or the arrow keys) moves the viewers.
- COG orthomosaics zoom in to their real pixels.

### Fixed
- Pan stuck after cancelling the locked-polygon prompt; the prompt now appears when dragging a
  locked polygon.
- `+` / `-` on polygons imported from shapefiles, a moved polygon snapping back after resizing, and
  resize lag.
- Selecting crowns while zoomed far out, and deleted polygons left drawn in the zoomed-out tiles.

## 1.1.0 - 2026-10-29

First tagged release.

### Added
- **Find canopy in other images**: register a drawn crown into other orthomosaics, raw UAV frames
  (located by EXIF GPS) and folders, with a results table and a Canopy browser to adjust or reject
  each match.
- **Export crowns**: the aligned crop of every date at each image's own resolution, plus the crown as
  CanoPie polygon files; the export folder opens as a CanoPie project.
- **Smart Canopy brush**: paint over a crown and its outline snaps to the crown edge.
- Packaging (`pip install .`, the `canopie` command), MIT license, tutorial with a generated example
  project, CONTRIBUTING guide, and CI on Windows and Linux.
- Window-by-window analysis for large rasters, one NoData rule, native OS file dialogs.

### Fixed
- Zoom-sync crash, stale band-math cache, `+` / `-` resize on COGs, smooth vertex dragging.
- Python 3.10 byte-budgeted cache deadlock.
- Earlier CSV-export, band-order, image-editor and polygon-coordinate fixes (see `BUGS.md` and the git
  history).

## Before 1.1.0

Development history from September 2025 is in the git log.
