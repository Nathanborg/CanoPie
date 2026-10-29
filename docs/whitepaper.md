# CanoPie: following individual tree crowns through raw drone, orthomosaic and phenocam imagery

**White paper -- version 1.1.0, October 2026**

**Nathan B. Gonçalves** ([ORCID 0000-0001-9068-6904](https://orcid.org/0000-0001-9068-6904))
and **Scott C. Stark** ([ORCID 0000-0002-1579-1648](https://orcid.org/0000-0002-1579-1648))

Software: <https://github.com/Nathanborg/CanoPie> ·
DOI: [10.5281/zenodo.23772111](https://doi.org/10.5281/zenodo.23772111) ·
License: MIT

> CanoPie was designed and developed by Nathan B. Gonçalves during a PhD in the
> Department of Forestry and the Program in Ecology, Evolution, and Behavior at
> **Michigan State University**, advised by Scott C. Stark, to analyse the drone and
> phenocam imagery of the dissertation research in the Amazon. Nathan B. Gonçalves is now
> a postdoctoral researcher at the **Smithsonian Tropical Research Institute (STRI)**, where CanoPie continues to
> be developed.

---

## 1. The problem

Drones and fixed cameras now photograph forest canopies at centimetre resolution, every
week if needed, in RGB, multispectral and thermal bands. For ecology, the interesting
unit in those images is usually not the landscape but the **individual tree crown**:
when does this tree flush new leaves, how does its canopy temperature respond to
drought, does it lose its leaves in the dry season while its neighbour does not?

The standard path from drone images to such numbers goes through structure-from-motion
photogrammetry: hundreds of overlapping frames are stitched into one orthomosaic per
flight, and crowns are then outlined on the mosaics. That path has real costs:

- **Radiometry.** Orthorectification resamples pixels, and overlap blending averages
  several frames of the same tree -- seen under different sensor and sun angles -- into
  one value. Seam lines, exposure jumps and smoothing follow (Kapil et al., 2023; Ahn et al., 2025).
  Multispectral and thermal imagery, with band-specific calibration and low native
  resolution, suffer most.
- **Metadata.** Per-frame information -- timestamp, irradiance from the downwelling light
  sensor, viewing geometry -- is lost in the stitched product.
- **Effort.** Photogrammetry is computationally heavy, storage-hungry, often commercial,
  and fails on low-texture imagery such as thermal frames of a closed canopy.
- **Time series.** Mosaics of different dates rarely line up: GPS error and
  flight-to-flight differences shift them by metres, so every date has to be
  co-registered again before the same tree can be compared.

CanoPie came out of exactly these problems. Over 14 months we flew repeated missions
with a MicaSense dual camera (10 bands) and a FLIR Duo Pro R (thermal + RGB) over the
same 100 ha of Amazon forest. The dataset was huge, orthomosaicking compromised both the
thermal and the spectral data, spatial consistency between months was poor, and thermal
imagery was especially hard because of its low contrast and the constant illumination
changes caused by clouds (Gonçalves et al., 2024). Getting reliable numbers for individual trees
through time was the bottleneck of the research, so we built a tool for it.

## 2. What CanoPie is

CanoPie is a desktop application, written entirely in Python (PyQt5, OpenCV, NumPy,
tifffile, scikit-learn), for loading, annotating, editing and measuring multitemporal
canopy imagery: **raw drone frames, orthomosaics and phenocam images alike**, without
requiring any of them to be stitched. It is free and open source (MIT), runs on an
ordinary laptop, and ships as a standalone Windows app that needs no Python installation.

![Several sensors, with different bands and resolutions, loaded and annotated in CanoPie.](https://github.com/user-attachments/assets/02aac60c-c71c-42b0-8d3c-acd9f4233f0d)

Its main pieces:

- **Projects and roots.** Images are grouped into *roots* -- the bands of one capture,
  the sensors of one flight, the dates of one site, one phenocam timestamp -- shown side
  by side in synchronized viewers. Two camera folders (e.g. multispectral and thermal)
  can be paired capture by capture.
- **Annotation.** Polygons, rectangles, circles and points, grouped by name (one group
  per tree), synchronized across the images of a root, managed in a polygon manager,
  and imported from or exported to shapefiles. The **Smart Canopy brush** outlines a
  crown from one brush stroke (Section 4).
- **Non-destructive editing.** Crop, rotation, resizing, histogram matching between
  dates, NoData rules and masks, band math, vegetation indices and boolean masks, stored
  as small `.ax` sidecar files and never written into the images.
- **Data extraction.** Per-polygon statistics of every band and index, written to CSV
  together with each image's EXIF metadata and timestamp.
- **Machine learning.** Pixel classification with scikit-learn models trained from
  labelled polygons; export of pixels, thumbnails and segmentation masks for
  deep-learning datasets.
- **Large rasters.** Orthomosaics and Cloud-Optimized GeoTIFFs of several gigabytes are
  read window by window from their overviews.
- **Phenocam series.** Frames taken in fog, rain or glare are filtered out by similarity
  to reference frames.
- **Finding a crown in other images** (Section 3): a crown outlined once is located in
  other orthomosaics, raw frames and phenocam images, and exported as aligned crops.

## 3. Finding the same crown everywhere -- without photogrammetry

The central idea of CanoPie is that, to follow a tree, you do not need to rebuild the
whole scene. You need to know **where that one crown is in each image**. CanoPie finds
it from information every drone image already carries, and then refines the position
by matching the crown's own pixels.

**Step 1 -- a first guess from the image's own metadata.**

- A GeoTIFF (orthomosaic) has a georeference: the crown's map coordinates are projected
  into it, whatever its coordinate system or pixel size.
- A raw frame is placed with a simple camera model built from its EXIF alone: the GPS
  position, the flight altitude above ground, the focal length and sensor size, which
  give the ground pixel size (GSD = altitude × sensor width / (focal length × image
  width)), and the heading. The frame is treated as a nadir view of flat ground.
- Some cameras record GPS but no usable heading or scale (the FLIR Duo Pro R and the
  MicaSense, for example). For those, CanoPie **calibrates the camera from the flight
  itself**: consecutive frames of a flight line overlap, so registering each frame to
  the next gives a pixel shift, and a least-squares fit of the accumulated pixel shifts
  against the GPS displacements gives the frame's metres per pixel and heading.

**Step 2 -- refine by matching the crown.** Around the first guess, the crown's patch is
searched for in the target image with masked normalized cross-correlation over
rotation, translation and scale (or with StackReg (Thévenaz et al., 1998) for a fast
rigid fit). The search window is generous enough for real GPS and heading errors:
±15 m and ±5° between GeoTIFFs; ±40 m, ±20° (or the full circle when the heading is
unknown) and a scale of 0.7--1.4 for raw frames.

The result: a crown drawn once on an orthomosaic is found on the orthomosaics of other
dates, in the individual raw RGB, multispectral and thermal frames of other flights --
each at its own resolution, rotation and sensor -- and back from a raw frame to a mosaic.
**No structure-from-motion, no camera poses, no point cloud and no elevation model are
needed.** Each match comes with a correlation score and its offset from where the
metadata put it, which doubles as a measure of the georeference error.

Every match can be reviewed in a canopy browser (move, reshape or reject an outline),
saved as the same polygon group on every image, and exported as aligned crops: one crop
per date over the same ground and orientation, each at its own image's native
resolution, plus a metadata CSV and a ready-to-open CanoPie project.

### How this differs from existing approaches

| Approach | Needs | Unit of analysis |
|---|---|---|
| Photogrammetry (Metashape, Pix4D, OpenDroneMap (OpenDroneMap, 2020)) | overlapping frames, SfM, often a DSM | stitched mosaic |
| Projecting annotations back to frames (e.g. via SfM camera poses and a mesh) | a full photogrammetric reconstruction | raw frame |
| Image-to-image co-registration (e.g. AROSICS) | two georeferenced rasters | whole image |
| Phenocam tools (phenopix (Filippa et al., 2016), PhenoCam (Sonnentag et al., 2012)) | a fixed camera | region of interest of one camera |
| **CanoPie** | **EXIF of each frame, or a GeoTIFF** | **one crown, across mosaics, raw frames and dates** |

To our knowledge, no other open tool locates a single crown across orthomosaics and raw
frames of different sensors and dates from EXIF metadata alone. A systematic benchmark
against these baselines is planned.

## 4. The Smart Canopy brush

Outlining hundreds of crowns by hand, vertex by vertex, is slow. With the Smart Canopy
brush the user paints a short stroke over a crown, and the outline snaps to the crown's
own edges. In a closed canopy, crowns touch their neighbours: the edge between two
crowns is a change of colour and texture and a shadow, not a drop to bare ground. So the
brush strokes are used as seeds for a graph cut (GrabCut on a blurred CIELAB copy of the
image, at a working resolution of a few hundred pixels): what is painted is crown for
sure, what is erased or already outlined is not, and the cut decides the band in
between. The part connected to the paint is kept, holes are filled and the outline is
smoothed, never shrinking off the paint. The brush was tuned on annotated crowns from an
orthomosaic COG, 1 m airborne RGB, phenocam frames and a MicaSense radiance band.

## 5. Design principles

**Edits never touch the images.** Each image's edits are stored in a small `.ax` JSON
sidecar and replayed in a fixed order by every consumer -- the viewer, the CSV and
machine-learning exports, the pixel inspector. A number exported from CanoPie is
computed from exactly the pixels the user sees, and the original radiometry is always
recoverable.

**Polygons follow their objects.** Polygons are stored per image as JSON, relative to
the edited frame, so they stay on their tree when an edit changes the geometry; the
group name links a tree across images and dates in every export.

**Frame-level analysis.** CanoPie deliberately does not orthorectify. Analyses stay in
each frame's own geometry, which preserves radiometry and metadata; the price is that
locating objects in raw frames relies on GPS and a simple camera model, refined by
image matching, rather than on a terrain model.

**Built for ordinary laptops.** Images load asynchronously, exports run in background
workers, band math uses NumExpr when available, polygons are drawn at simplified levels
of detail when zoomed out, and rasters too large for memory are processed window by
window.

**Tested.** An automated quality-control suite of more than 1,700 tests checks the edit
replays, exports, band math, classification, polygon handling and crown matching on
synthetic scenes with known answers, on Windows and Linux, on every change. A tutorial
walks through every feature on an example project generated by one script.

## 6. Use so far

CanoPie was developed for, and used in, the 14-month MicaSense and FLIR survey over
100 ha of Amazon forest described above, where it follows individual crowns across
dates. The resulting crown time series support the analysis of leaf demography,
spectral variability and productivity across seasons in Nathan B. Gonçalves's doctoral
dissertation at Michigan State University (Gonçalves, 2025).

## 7. Limitations and next steps

- Raw frames are modelled as nadir views of flat ground. Oblique frames or steep terrain
  rely more on the image-matching step to correct the first guess.
- Registration needs the crown to be recognisable between images; a crown that changed
  completely (e.g. fell) scores low and is flagged for review rather than silently
  accepted.
- The standalone app is built for Windows; on macOS and Linux CanoPie runs from source.
- Planned: a published benchmark of the crown registration against photogrammetry-based
  and co-registration baselines, and more sensors calibrated out of the box.

## 8. Getting and citing CanoPie

- Windows app: `CanoPie-<version>-windows.zip` from the
  [latest release](https://github.com/Nathanborg/CanoPie/releases/latest).
- From source: `pip install .` in a clone of the repository (Python 3.10+).
- Tutorial: [docs/tutorial.md](tutorial.md).

Please cite:

> Gonçalves, N. B., & Stark, S. C. (2026). *CanoPie: A Graphical User Interface for
> Multitemporal Canopy Imagery Data Analysis* (Version 1.1.0) [Computer software].
> Zenodo. <https://doi.org/10.5281/zenodo.23772111>

## AI usage

Generative AI (Anthropic Claude, through Claude Code) was used during development to
write and refactor parts of the code -- including the crown-registration module,
performance work and bug fixes -- and the automated tests, and to help edit this text.
All AI-assisted code was reviewed by the authors and verified with the test suite and on
real datasets; the design decisions are the authors'.

## Acknowledgements

N.B. Gonçalves and S.C. Stark were supported by NSF DEB awards no. 1950080 and 1754357.
N.B. Gonçalves was also financed by the Michigan State University Mericle Memorial Fund.
We gratefully acknowledge the financial support of Simons Foundation award 429440, which
supported N.B. Gonçalves.

## References

- Ahn, H., Kim, C., Lim, S., Jin, C., Kim, J., & Choi, C. (2025). Minimizing
  seam lines in UAV multispectral image mosaics utilizing irradiance, vignette, and
  BRDF. *Remote Sensing*, 17(1), 151. <https://doi.org/10.3390/rs17010151>
- Filippa, G., Cremonese, E., Migliavacca, M., Galvagno, M.,
  Forkel, M., Wingate, L., Tomelleri, E., Morra di Cella, U., & Richardson, A. D.
  (2016). Phenopix: A R package for image-based vegetation phenology. *Agricultural and
  Forest Meteorology*, 220, 141--150. <https://doi.org/10.1016/j.agrformet.2016.01.006>
- Gonçalves, N. B., Nelson, B. W., Simonetti, A., & Stark, S. C.
  (2024). Cloud shadow and uneven illumination detection and correction using the U-net
  architecture in near-surface images of complex forest canopies. *ISPRS Archives*,
  XLVIII-3-2024, 191--196. <https://doi.org/10.5194/isprs-archives-XLVIII-3-2024-191-2024>
- Gonçalves, N. B. (2025). *Leaf Demography, Spectral
  Variability, and Hydrological-topographic Controls on Amazon Forest Productivity
  Across Seasons and Under Climate Stress* [Doctoral dissertation, Michigan State
  University].
- Kapil, R., Castilla, G., Marvasti-Zadeh, S. M., Goodsman, D., Erbilgin,
  N., & Ray, N. (2023). Orthomosaicking thermal drone images of forests via
  simultaneously acquired RGB images. *Remote Sensing*, 15(10), 2653.
  <https://doi.org/10.3390/rs15102653>
- OpenDroneMap Authors (2020). ODM -- A command line toolkit to generate
  maps, point clouds, 3D models and DEMs from drone, balloon or kite images.
  <https://github.com/OpenDroneMap/ODM>
- Sonnentag, O., Hufkens, K., Teshera-Sterne, C., Young, A. M.,
  Friedl, M., Braswell, B. H., Milliman, T., O'Keefe, J., & Richardson, A. D. (2012).
  Digital repeat photography for phenological research in forest ecosystems.
  *Agricultural and Forest Meteorology*, 152, 159--177.
  <https://doi.org/10.1016/j.agrformet.2011.09.009>
- Thévenaz, P., Ruttimann, U. E., & Unser, M. (1998). A pyramid
  approach to subpixel registration based on intensity. *IEEE Transactions on Image
  Processing*, 7(1), 27--41. <https://doi.org/10.1109/83.650848>
