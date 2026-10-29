<img width="456" height="135" alt="logo" src="https://github.com/user-attachments/assets/8cf3f529-817b-45cb-b9ce-0315d12cea37" />

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23772111.svg)](https://doi.org/10.5281/zenodo.23772111)
[![Latest release](https://img.shields.io/github/v/release/Nathanborg/CanoPie)](https://github.com/Nathanborg/CanoPie/releases/latest)

**DOI:** [10.5281/zenodo.23772111](https://doi.org/10.5281/zenodo.23772111) ·
**Zenodo:** [record](https://zenodo.org/records/23772112) ·
**Download:** [latest release](https://github.com/Nathanborg/CanoPie/releases/latest) ·
**Docs:** [tutorial](docs/tutorial.md), [white paper](docs/whitepaper.md)

CanoPie was developed by Nathan B. Gonçalves during a PhD at Michigan State University,
with Scott C. Stark, and continues to be developed at the Smithsonian Tropical Research Institute (STRI), where
Nathan B. Gonçalves is now a postdoctoral researcher.

## Project Description
CanoPie is an image analysis tool for RGB, multispectral, and multitemporal data, designed for forest and agricultural canopy studies. With an intuitive PyQt5-based GUI, it enables researchers to process UAV and phenocam imagery (multispectral, thermal, and RGB), draw and manage polygons for data extraction, compute image statistics, and retrieve metadata directly via ExifTool,  all without relying on heavy photogrammetry pipelines.
Canopie's core functionality relies on libraries such as OpenCV, tiffile, and PyQt5.

Unlike orthomosaics, which can distort radiometry and demand large storage and computing resources, CanoPie provides fast, direct access to raw image data while preserving metadata and maintaining spectral integrity.
<img width="800" height="800" alt="Canopie_fig1" src="https://github.com/user-attachments/assets/02aac60c-c71c-42b0-8d3c-acd9f4233f0d" />

## Why CanoPie?
Orthomosaics are useful for mapping when you need , but they come at a cost:
-   *Require resampling and interpolation, which alters pixel values.
-   *Blend across overlaps in the stiching process, often smoothing or distorting radiometry.
-   *Add artifacts, shadows, or inconsistencies due to geometry and illumination differences.
-   *Can be computationally intensive and storage-heavy.
-   *For many ecological and forestry applications — such as monitoring specific trees, crown patches, or hundreds of individuals over time — orthomosaics are not necessary. -   *CanoPie provides a faster, lighter, and more spectrally faithful alternative.

## Getting Started

**New to CanoPie?** The [tutorial](docs/tutorial.md) walks through every feature on a small example project
you generate in one command (`python examples/make_example_project.py`).

### Download (Windows, no Python needed)
Download `CanoPie-<version>-windows.zip` from the [latest release](https://github.com/Nathanborg/CanoPie/releases/latest),
unzip it anywhere and run `CanoPie.exe`. The optional accelerators (numba) and the XGBoost / LightGBM
classifiers are left out of this build; install from source (below) if you need them.

### Prerequisites
Before you begin, ensure you have the following installed:
*   Python 3.10
*   The required libraries listed in `requirements.txt`
*   ExifTool by Phil Harvey if you need exif data information from imagery
### Platform support
CanoPie has been developed and tested on **Windows 10/11**.  
The test suite also runs on Linux (Ubuntu) in continuous integration; macOS should work but is not tested.  

### Installation and Usage (Conda)

1. **Install Miniconda/Conda (if not already installed)**  
   Download and install Miniconda for your operating system from:  
   [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)  

2. **Create and activate environment**
```sh
conda create --name canopie-env python=3.10 -y
conda activate canopie-env
```
3. **Clone the repository** (if you haven't already) or download the raw file (on the top) and extract from zipfile
anywhere in your computer:
```sh
git clone https://github.com/Nathanborg/CanoPie.git
```
4. **Navigate to the project directory (Folder where CanoPie was stored and extracted where main.py is stored)**
```sh
cd CanoPie
```
5. **Install dependencies**
```sh
pip install -r requirements.txt
```
6. **Run CanoPie**
```sh
python main.py
```
Or install it as a package, which adds a `canopie` command:
```sh
pip install .
canopie
```

### Following a crown across images (Find canopy in other images)
Right-click a polygon (in a viewer or in the Polygon Manager; several polygons can be selected) and choose
**Find canopy in other images…**. CanoPie locates the same crown in:
- orthomosaics of other dates, coordinate systems and resolutions (GeoTIFF georeference);
- raw UAV frames, from their EXIF GPS and a camera model -- GPS-only cameras such as the FLIR Duo Pro R and
  MicaSense are calibrated from the overlap of consecutive frames of their flight line -- and back from a raw
  frame to an orthomosaic or to another flight's frames;
- phenocam images, matched in image coordinates.

Images can come from the current root, every root of the project, or other folders / projects (one or several,
each folder treated as one flight). The crown is registered with StackReg (rigid body) or an exhaustive
rotation and shift search, saved as the same polygon group on each image, and can be reviewed (moved, reshaped,
rejected) in the canopy browser. **Export crowns** writes, per crown, the source crop and every image's crop
over the same ground and orientation, each at its own image's original resolution (GeoTIFF or JPEG), a metadata
CSV, and the crown on every crop as CanoPie polygon files: each crown's folder (by default in `canopy_exports`
inside the project) opens in CanoPie with **File > Load Project**.

### Running the tests
```sh
pip install -r requirements-dev.txt
python -m pytest canopie/qc -q -p no:cacheprovider
```
A feature-level report is written to `canopie/qc/last_run_report.txt`. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Contributing and support
Bug reports, questions and pull requests are welcome through the
[issue tracker](https://github.com/Nathanborg/CanoPie/issues); see [CONTRIBUTING.md](CONTRIBUTING.md).

### License
CanoPie is released under the [MIT License](LICENSE).

### How to cite
If you use CanoPie in your research, please cite it:

Gonçalves, N. B., & Stark, S. C. (2026). *CanoPie: A Graphical User Interface for Multitemporal Canopy Imagery
Data Analysis* (Version 1.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23772111

This DOI always resolves to the latest version; each release also has its own DOI on
[Zenodo](https://doi.org/10.5281/zenodo.23772111) (1.1.0: `10.5281/zenodo.23772112`). GitHub's
"Cite this repository" button (from `CITATION.cff`) gives the same reference in APA and BibTeX.

### Project status
CanoPie is an early-stage collaborative project intirelly python-based developed by and for the research community. 
As a project in active development, it welcomes contributions from researchers, developers, 
and users interested in advancing open-source tools for environmental and agricultural monitoring.







