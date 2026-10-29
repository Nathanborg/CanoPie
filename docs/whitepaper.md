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
second if needed, in RGB, multispectral and thermal bands: phenocams on towers record
the same canopy every few minutes for years, drones fly the same forest week after
week, and both feed orthomosaics of whole landscapes. For ecology, the interesting
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
- **Disconnected tools.** Phenocam series, raw drone frames and orthomosaics are
  usually analysed in different software, so the same tree seen by a tower camera, a
  drone and a mosaic is rarely measured in one place.

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
- **Image editor** (Section 5): non-destructive crop, rotation, resizing, histogram
  matching between dates, NoData rules and polygon masks, band math, vegetation indices,
  boolean masks and classification, stored as small `.ax` sidecar files and never
  written into the images.
- **Data extraction.** Per-polygon statistics of every band and index, written to CSV
  together with each image's EXIF metadata and timestamp.
- **Machine Learning Manager** (Section 6): pixel classification with scikit-learn, and
  export of pixels, thumbnails and segmentation masks for deep-learning training.
- **Orthomosaics as Cloud-Optimized GeoTIFFs.** Mosaics of several gigabytes -- including
  Cloud-Optimized GeoTIFFs (COGs) and hyperspectral cubes with hundreds of bands -- open
  in seconds: CanoPie reads only the tiles on screen, from the overview level that
  matches the zoom, instead of loading the whole raster into memory. Statistics, scene
  means and exports on these rasters are computed window by window.
- **Phenocams.** A phenocam series is a project like any other: each frame is a root,
  a crown drawn once with *Sync* follows the whole series, and the CSV gives its
  greenness (e.g. green chromatic coordinate) frame by frame. Frames taken in fog, rain
  or with the sun in the lens are filtered out by their similarity to reference frames.
- **Finding a crown in other images** (Section 3): a crown outlined once is located in
  other orthomosaics, raw drone frames and phenocam images, and exported as aligned
  crops.

## 3. Finding the same crown everywhere -- without photogrammetry

The central idea of CanoPie is that, to follow a tree, you do not need to rebuild the
whole scene. You need to know **where that one crown is in each image**. CanoPie finds
it from information every drone image already carries, and then refines the position
by matching the crown's own pixels.

The same search works between any of the kinds of imagery a canopy study collects:

| From → to | How the crown is first placed |
|---|---|
| **Orthomosaic → orthomosaic** (other dates, other coordinate systems, other pixel sizes, COGs) | the GeoTIFF georeference of both |
| **Orthomosaic → raw drone frames** (RGB, multispectral, thermal) | the frames' EXIF GPS, altitude, camera and heading |
| **Raw frame → orthomosaic** | the frame's camera model, projected into the mosaic |
| **Raw frame → raw frames** of another flight or sensor | both frames' camera models |
| **Phenocam → phenocam** frames | image coordinates (a fixed camera) |

Images can come from the current project, from every root of it, or from any folders or
other projects -- one folder per flight or date -- so a single crown can be followed
through a whole archive of mosaics, flights and camera series.

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

## 5. The image editor

Comparing a tree across dates and sensors usually needs some preparation of the images:
cropping, correcting brightness between flights, masking clouds or gaps, computing an
index. In most workflows each of these steps writes a new copy of the imagery. In
CanoPie, the **image editor** records them instead: every edit is saved in a small
`.ax` JSON file next to the image and replayed, in a fixed order, by the viewer, the
CSV export, the machine-learning exports and the image export alike. The original
file is never rewritten, so the original radiometry is always one click away, and
every number exported is computed from exactly the pixels on screen.

An edit can be applied to one image, to every image of a root, or to the whole project:

- **Geometry**: crop, rotation and resizing.
- **NoData**: values or expressions to ignore (e.g. `0, b1<123`), and **polygon masks**
  that leave out the pixels of chosen groups -- a cloud, a gap, a tower.
- **Histogram matching** (mean/std or full CDF) of every date to a reference image,
  which removes much of the light difference between flights or phenocam frames before
  they are compared.
- **Band math**: any expression of the bands -- NDVI, GCC, ExG, a thermal threshold --
  shown live; **boolean** expressions such as `(b4-b1)/(b4+b1) > 0.6` give masks.
- **Classification** with a trained model (Section 6).
- **Appended bands**: the result of an expression or a classification can be kept as an
  extra band that every export then sees as data.

## 6. The Machine Learning Manager: from labelled crowns to training data

Polygons drawn in CanoPie are labelled data. The **Machine Learning Manager** turns them
into models and into training sets for other tools.

**Pixel classification.** Each polygon group is a class (e.g. `leaf`, `bare`,
`understory`). The manager samples pixels from the polygons -- the bands in any order,
vegetation-index expressions, and an optional 3×3 or 5×5 neighbourhood window -- trains
scikit-learn models (RandomForest, ExtraTrees, GradientBoosting, LogisticRegression,
SVM, GaussianNB, and XGBoost or LightGBM when installed, optionally tuned and stacked),
and saves each model with a report: classes, features, holdout accuracy, confusion
matrix and feature importances. The model then classifies whole images in the editor
and adds class counts per polygon to the CSV.

**Exporting for deep learning.** For models trained elsewhere (U-Net, Mask R-CNN,
vision transformers), the manager exports:

- **thumbnails** -- an image crop around every polygon, for classification datasets;
- **segmentation masks** -- image and mask pairs, cut into fixed-size tiles (e.g. 256 or
  512 px) for segmentation networks, with a CSV index of every pair;
- **pixel tables** -- the values inside each polygon, as CSV;

all with the image editor's edits applied, so the training data match what was seen
and checked on screen. Training-time **augmentation** (brightness changes over the
whole image or patch by patch, and synthetic shadows and uneven illumination) teaches
models not to depend on the light of one flight -- a frequent failure in forest imagery
under moving clouds.

**Raw frames multiply the samples.** Because CanoPie works on raw frames, a crown is
not one sample per flight, as on an orthomosaic, but one per frame that sees it --
often 5 to 20 overlapping frames, each from a different view angle, sun geometry and
position in the image. With *Find canopy in other images*, a crown labelled once is
placed on all of them automatically. Every one of those views becomes a training
example, so the same field effort yields many times more samples, with the natural
variation in viewing geometry and illumination that a model meets in practice and that
an orthomosaic averages away.

## 7. Design principles

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

## 8. Use so far

CanoPie was developed for, and used in, the 14-month MicaSense and FLIR survey over
100 ha of Amazon forest described above, where it follows individual crowns across
dates. The resulting crown time series support the analysis of leaf demography,
spectral variability and productivity across seasons in Nathan B. Gonçalves's doctoral
dissertation at Michigan State University (Gonçalves, 2025).

## 9. Limitations and next steps

- Raw frames are modelled as nadir views of flat ground. Oblique frames or steep terrain
  rely more on the image-matching step to correct the first guess.
- Registration needs the crown to be recognisable between images; a crown that changed
  completely (e.g. fell) scores low and is flagged for review rather than silently
  accepted.
- The standalone app is built for Windows; on macOS and Linux CanoPie runs from source.
- Planned: a published benchmark of the crown registration against photogrammetry-based
  and co-registration baselines, and more sensors calibrated out of the box.

## 10. Getting and citing CanoPie

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
