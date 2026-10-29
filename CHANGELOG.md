# Changelog

All notable changes to CanoPie. Versions follow `pyproject.toml`.

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
