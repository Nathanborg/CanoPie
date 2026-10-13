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
  - image registration
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

Beyond polygon-based extraction, CanoPie now **finds the same crown in other
images**. A crown outlined once is located, saved under the same polygon
group, and exported in every other image that holds it, across:

- orthomosaics of different dates, coordinate systems, and resolutions
  (through their GeoTIFF georeference);
- orthomosaics and raw UAV frames, in both directions: raw frames are located
  from EXIF GPS and a camera model, and GPS-only cameras without
  height-above-ground or heading (e.g., FLIR Duo Pro R, MicaSense) are
  calibrated from the overlap of consecutive frames of their own flight line;
- raw frames of different flights and dates;
- phenocam images, matched in image coordinates.

Candidate images are chosen by a KD-tree on their GPS positions; the crown is
then registered with the StackReg (TurboReg) rigid-body method
[@thevenaz1998pyramid] or an exhaustive rotation and shift search scored by
normalized cross-correlation. The result is exported as aligned crown crops on
a common grid (GeoTIFF or JPEG), one polygon file per image, and a metadata
CSV -- a ready-made time series of the same tree.

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
which the same crowns are drawn, edited, registered, and measured across
sensors, roots (groups of co-registered images), and dates. Contributing this
to an existing package was not practical, as none of them are built around
frame-level, multitemporal polygon management with non-destructive edits.

# Software design

A CanoPie project groups images into *roots*: sets shown side by side in
synchronized viewers (e.g., the bands, sensors, or dates of one site). Edits
-- crop, rotation, resize, band math, NoData rules, histogram matching, band
registration, classification -- are never written into the images; they are
stored in a small `.ax` sidecar per image and replayed identically by the
viewer, the CSV and machine-learning exports, and the pixel inspector, so the
statistics a user exports are those of the pixels they see. Polygons are
stored per image as JSON records relative to the edited frame, so they stay on
their object when an edit changes the geometry.

Large rasters (orthomosaics of tens of gigapixels, COGs) are read window by
window from their overviews, with the `.ax` replay computed per window rather
than on the whole image; this trades some code complexity for analyses that
fit in memory on a laptop. The crown finder is a Qt-free module: each image is
described by a georeference (GeoTIFF, an EXIF camera model, or its own pixel
frame) composed with its `.ax` geometry, and matching happens on a working grid
at the coarser of the two resolutions, so a 2 cm raw frame is never compared
pixel by pixel with a 15 cm mosaic. CanoPie does not orthorectify: raw-frame
locations rely on GPS and a nadir camera model, and registration solves for a
rigid transform plus scale, which suits individual crowns but not
terrain-induced distortion over large areas. Because the registration library
holds Python's global interpreter lock, searches and flight-line calibrations
run in worker processes at below-normal priority, keeping the interface
responsive. EXIF data are read from file headers and cached per image folder,
so a folder of thousands of frames is indexed once.

CanoPie ships with an automated quality-control suite (`canopie/qc`, more than
1,700 pytest tests with a feature-level report) covering geometry replays,
exports, polygon handling, and registration on synthetic scenes with known
answers.

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
