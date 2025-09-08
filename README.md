<img width="456" height="135" alt="logo" src="https://github.com/user-attachments/assets/8cf3f529-817b-45cb-b9ce-0315d12cea37" />

## Project Description
CanoPie is an image analysis tool for RGB, multispectral, and multitemporal data, designed for forest and agricultural canopy studies. With an intuitive PyQt5-based GUI, it enables researchers to process UAV and phenocam imagery (multispectral, thermal, and RGB), draw and manage polygons for data extraction, compute image statistics, and retrieve metadata directly via ExifTool,  all without relying on heavy photogrammetry pipelines.

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

### Prerequisites
Before you begin, ensure you have the following installed:
*   Python 3.1
*   The required libraries listed in `requirements.txt`

### Platform support
CanoPie has been developed and tested on **Windows 10/11**.  
It should run on other operating systems (Linux, macOS) as long as Python 3.10+ and all dependencies are correctly installed, but cross-platform support has not been fully tested yet.  

### Installation and usage Python
1.  **Clone the repository** (if you haven't already):
    ```sh
    git clone https://github.com/Nathanborg/CanoPie.git
    ```
2.  **Navigate to the project directory**:
    ```sh
    cd CanoPie
    ```
3.  **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
4.  **To run the main application**:
    ```sh
    python main.py
    ```
### Installation and Usage (Conda)

1. **Install Miniconda/Conda (if not already installed)**  
   Download and install Miniconda for your operating system from:  
   [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)  

2. **Create and activate environment**
```sh
conda create --name canopie-env python=3.10 -y
conda activate canopie-env```
```
3. **Navigate to the project directory**
```sh
cd CanoPie
```
4. **Install dependencies**
```sh
pip install -r requirements.txt
```
4. **Run CanoPie**
```sh
python main.py
```

### Project status
CanoPie is an early-stage collaborative project intirelly python-based developed by and for the research community. 
As a project in active development, it welcomes contributions from researchers, developers, 
and users interested in advancing open-source tools for environmental and agricultural monitoring.







