---
title: 'CanoPie: A Graphical User Interface for Multitemporal Canopy Imagery Data Analysis'
tags:
  - Python
  - remote sensing
  - UAV
  - phenocam
  - thermal imagery
  - multispectral imagery
  - forest ecology
  - time series
authors:
  - name: Nathan B. Gonçalves
    # orcid: 0000-0000-0000-0000   # TODO: add the ORCID
    corresponding: true
    affiliation: "1, 2"
  - name: Scott C. Stark
    # orcid: 0000-0000-0000-0000   # TODO: add the ORCID
    affiliation: "1, 2"
affiliations:
  - name: Department of Forestry, College of Agriculture and Natural Resources, Michigan State University, East Lansing, MI, USA
    index: 1
  - name: Program in Ecology, Evolution, and Behavior, Michigan State University, East Lansing, MI, USA
    index: 2
date: 13 October 2026
bibliography: paper.bib
---

# Summary

Monitoring forest and agricultural canopies is critical for understanding
ecosystem function, crop productivity, and responses to environmental change
[@asner1998biophysical]. UAVs and near-surface remote sensing platforms now
provide extensive high-resolution imagery [@sonnentag2012digital; @almeida2016;
@simonetti2023], but analyzing these data often requires photogrammetric
pipelines that are computationally expensive, storage-heavy, often commercial,
and that can cause radiometric distortions [@kapil2023; @ahn2025]. A number of
tools exist for image stitching and orthorectification, but they pose a
technical barrier and, even when successful, the resulting orthomosaics can
alter spectral integrity in ways that complicate multispectral or thermal
analyses. Open-source tools that enable streamlined analysis of raw,
non-orthorectified imagery, while preserving radiometry and metadata, improve
the accessibility and reproducibility of canopy monitoring for a broader
community of researchers. CanoPie is such a tool: a Python graphical user
interface to load, edit, annotate, and extract statistics from multitemporal
canopy imagery -- raw UAV frames, phenocam images, and orthomosaics alike --
and to follow individual tree crowns across images and dates.

# Statement of need

CanoPie is an entirely Python-based GUI for canopy imagery analysis, focused
primarily but not exclusively on UAV and phenocam imagery. It relies on
libraries such as OpenCV, tifffile, and PyQt5. It loads and displays the image
formats most common for multispectral, thermal, and RGB UAV-derived data
(single and multi-band TIFF stacks, Cloud-Optimized GeoTIFFs, JPG, PNG).
CanoPie is intended for researchers in forest- and agriculture-related fields
who work with multitemporal imagery and do not want to rely on photogrammetry
software and processes. The standard path to analysis of UAV-derived images
routes through structure-from-motion (SfM) photogrammetry and orthomosaic
generation. These steps are computationally heavy, storage-intensive, and not
designed to preserve the per-image radiometry or metadata critical for
multispectral and thermal science.

Conventional orthomosaicking discards view geometry and associated metadata,
such as irradiance recorded by downwelling light sensors, during stitching:
multiple frames of the same object (e.g., a tree observed under different
sensor and sun angles) are blended into a single pixel. Orthorectification,
resampling, and overlap blending also introduce artifacts, exposure
discontinuities, and spectral distortion. These issues are especially
consequential for multispectral and thermal data, where band-specific
calibration and lower native resolution make mosaics more prone to radiometric
inconsistency. In addition, imperfect GPS and flight-to-flight variability can
degrade temporal co-registration, requiring further georeferencing.

Many canopy-monitoring tasks instead require rapid, frame-level access to
imagery without sacrificing metadata (EXIF, irradiance, timestamp, viewing
geometry). When the objective is to follow individual trees or plots over
time, the user may only need raw or radiance/reflectance-corrected frames
(e.g., via the MicaSense image-processing utilities) to extract statistics from
individual objects. CanoPie targets this use case: multitemporal data
extraction directly from original frames or their corrected versions, without
orthomosaic generation. Image metadata are extracted with ExifTool and kept
with every extracted statistic, so that radiometry, sun--sensor geometry, and
temporal information remain available for quantitative analysis.

CanoPie was first designed to analyze a multitemporal dataset from a MicaSense
dual camera (10 bands) and a FLIR Duo Pro R (thermal + RGB) camera, collected
during surveys in the Amazon (\autoref{fig:overview}). Over 14 months we flew
repeated missions over the same 100 ha area; the dataset was massive, and
orthomosaicking introduced artifacts that compromised both thermal and
spectral integrity. Spatial consistency between months was a further
challenge, and thermal imagery proved especially difficult because of its low
contrast and the constant illumination changes caused by cloud cover
[@goncalves2024]. It became very difficult to obtain reliable data on
individual trees through time. CanoPie was built to follow individual trees
through time after correcting the raw images to radiance, so that a single
project CSV tracks the statistics of those trees across dates while retaining
all image metadata.

In practice, a CanoPie project provides:

- **Projects and roots**: images grouped into *roots* -- the bands of one
  capture, the sensors of one flight, or one phenocam timestamp -- shown side by
  side in synchronized viewers; two folders (e.g., multispectral and thermal/RGB)
  can be paired with an offset, and GPS positions are shown on a map.
- **Polygon and point annotation**: groups of polygons (e.g., one per tree)
  drawn once and synchronized across the images of a root, managed in a polygon
  manager, imported from and exported to shapefiles, and drawn at interactive
  speed for thousands of crowns.
- **Non-destructive editing**: crop, rotation, resizing, histogram matching,
  band registration, NoData rules, band math and vegetation indices (e.g., NDVI,
  GCC, ExG), boolean masks, and derived bands, applied to one image, a root, or
  the whole project without altering the original files.
- **Data extraction**: per-polygon statistics of every band and index, written
  to CSV together with each image's EXIF metadata and timestamp, so the same
  trees can be compared across dates and sensors.
- **Machine learning**: pixel classification with scikit-learn models trained
  from labelled polygons, and export of thumbnails and segmentation masks for
  deep-learning datasets.
- **Large rasters and image selection**: orthomosaics and Cloud-Optimized
  GeoTIFFs are read window by window, and phenocam series can be filtered by
  similarity to a reference image.
- **Crown matching across images**: a crown drawn once can be located in other
  orthomosaics, raw UAV frames (from EXIF GPS), or phenocam images, registered
  with StackReg [@thevenaz1998pyramid], saved as the same polygon group, and
  exported as aligned crops.

![Multiple sensors, with different bands and resolutions, loaded and annotated in CanoPie.\label{fig:overview}](figures/canopie_overview.png)

# State of the field

SfM packages -- commercial (Agisoft Metashape, Pix4D) or open source
(OpenDroneMap, @odm2020) -- produce orthomosaics, point clouds, and elevation
models; their unit of analysis is the stitched mosaic, not the original frame.
GIS software such as QGIS extracts zonal statistics from georeferenced
rasters, but it does not manage the same objects across hundreds of raw,
non-georeferenced frames and dates. Phenocam tools, such as the phenopix R
package [@filippa2016phenopix] and the PhenoCam processing approaches
[@sonnentag2012digital], extract greenness time series from regions of interest
of fixed cameras, but they do not handle UAV frames or multispectral and
thermal stacks. Radiometric libraries such as the MicaSense image-processing
utilities correct frames to radiance or reflectance but provide no
multitemporal annotation or extraction. CanoPie complements rather than
replaces these tools: it reads their outputs (corrected frames, orthomosaics,
phenocam images) and adds the missing piece -- a single interactive project in
which the same crowns are drawn, edited, and measured across
sensors, roots (groups of co-registered images), and dates. Contributing this
to an existing package was not practical, as none of them are built around
frame-level, multitemporal polygon management with non-destructive edits.

# Software design

CanoPie is a PyQt5 application organized around a project controller that
loads images into roots, renders them in synchronized viewers, and owns the
polygon store; dialogs (image editor, polygon manager, machine-learning
manager, crown finder) operate on that shared state. Three design choices shape
the software.

First, edits are never written into the images. Each image's edits -- crop,
rotation, resize, histogram matching, band registration, NoData rules, band
expressions, classification -- are stored in a small `.ax` sidecar and
replayed in a fixed order by every consumer: the viewer, the CSV and
machine-learning exports, and the pixel inspector. A statistic exported from
CanoPie is therefore computed from exactly the pixels the user sees, and the
original radiometry can always be recovered.

Second, polygons are stored per image as JSON records relative to the edited
frame, so a polygon stays on its object when an edit changes the image
geometry, and the same group name links a tree across images and dates in the
exported CSV.

Third, CanoPie favours responsiveness on ordinary laptops over raw throughput.
Images load asynchronously, exports run in background workers, band math uses
NumExpr when available, polygons are drawn from simplified levels of detail
when zoomed out, and rasters too large for memory are processed window by
window from their overviews instead of as whole images. The trade-off is added
code complexity in the replay paths, which is controlled by keeping one
canonical implementation of each step. CanoPie deliberately does not
orthorectify: analyses stay in each frame's own geometry, which preserves
radiometry but means that locating objects in raw frames relies on GPS and
simple camera models rather than on a terrain model.

CanoPie is installed with `pip` and ships with an automated quality-control
suite (`canopie/qc`, more than 1,700 pytest tests with a feature-level report)
covering the edit replays, exports, band math, classification, polygon
handling, and crown matching on synthetic scenes with known answers; it runs in
continuous integration on Windows and Linux. A tutorial (`docs/tutorial.md`)
walks through every feature on an example project generated by one script.

# Research impact statement

<!-- TODO (authors): state REALIZED impact only: the publications, theses, and
projects that used CanoPie, and its users / groups. JOSS asks for evidence, not
aspirations. -->

CanoPie was developed for, and used in, a 14-month multitemporal survey of
MicaSense multispectral and FLIR Duo Pro R thermal imagery over 100 ha of
Amazon forest, where it is used to follow individual crowns across dates.
<!-- TODO: add the studies / datasets / groups that use it (e.g., Panama / Barro
Colorado Island work) with citations. -->

# AI usage disclosure

<!-- TODO (authors): confirm and complete -- JOSS requires tools, versions,
where they were used, and confirmation of human review. -->

Generative AI (Anthropic Claude, model Claude Opus 5.5, through Claude Code)
was used during development to write and refactor parts of the code --
including the crown-registration module, performance work, and bug fixes --
and the automated tests, and to help edit this manuscript. All AI-assisted
code was reviewed by the authors, verified with the automated test suite and
on real datasets, and the core design decisions were made by the authors.

# Acknowledgements

<!-- TODO (authors): funding sources and grant numbers; collaborators. -->

# References
