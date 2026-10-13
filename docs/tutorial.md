# CanoPie tutorial

A tour of CanoPie on a small example project that you generate on your own computer
(synthetic data, no download), so every step gives the same result for everyone. It
takes about 30 minutes; each section can also be read on its own.

1. [Install](#1-install)
2. [Make the example project](#2-make-the-example-project)
3. [Open a project, roots and viewers](#3-open-a-project-roots-and-viewers)
4. [Open images from one or two folders](#4-open-images-from-one-or-two-folders)
5. [Draw and manage polygons](#5-draw-and-manage-polygons)
6. [The image editor: non-destructive edits, histogram matching, band and boolean math](#6-the-image-editor)
7. [Extract statistics to CSV](#7-extract-statistics-to-csv)
8. [Machine learning](#8-machine-learning)
9. [Phenocam series: filter similar images](#9-phenocam-series-filter-similar-images)
10. [Follow a crown through time](#10-follow-a-crown-through-time)
11. [Run the tests](#11-run-the-tests)

## 1. Install

```sh
conda create --name canopie-env python=3.10 -y
conda activate canopie-env
git clone https://github.com/Nathanborg/CanoPie.git
cd CanoPie
pip install -r requirements.txt      # or: pip install .   (adds a `canopie` command)
```

[ExifTool](https://exiftool.org) is optional: it is only needed to put the images'
EXIF metadata in the exports (**File > Set exiftool Path**).

## 2. Make the example project

```sh
python examples/make_example_project.py canopie_example
```

It writes, in `canopie_example/`:

| Folder | What |
|---|---|
| `mosaics/` | Three dates (2024-02-10, 05-12, 08-20) of a 4-band orthomosaic -- bands R, G, B, NIR; 10 cm; UTM 21S -- of the same 90 m patch of canopy. Each date's georeference is off by a few metres, as flights are, and one tree (`tree_3`) is bare on the last date. |
| `phenocam/` | Five RGB frames of a fixed camera over the canopy, in changing light. |
| `flight/multispectral/`, `flight/thermal/` | Raw frames of a drone flight, as two cameras write them: four captures of four single-band frames (`IMG_0001_1.tif` ... `IMG_0004_4.tif` = R, G, B, NIR) and one thermal frame per capture, numbered by its own camera (`IMG_0101_IR.tif` ...). |
| `project.json`, `polygons/` | A CanoPie project with two roots (`Mosaics`, `Phenocam`), five crowns (`tree_1` ... `tree_5`) drawn on the first date, and three training classes (`leaf`, `bare`, `understory`). |

## 3. Open a project, roots and viewers

```sh
python main.py        # or: canopie
```

**File > Load Project** and choose `canopie_example`. A project is a set of *roots*: a
root is a group of images shown together in synchronized viewers -- the bands of one
capture, the sensors of one flight, the dates of one site, or one phenocam frame. Here
the first root, `Mosaics`, shows the three dates side by side. **Next** / **Previous**
(or the slider) move between roots; **Zoom All** and **Rectangle Zoom** zoom every
viewer of the root together.

- **View > Select Display Bands…** chooses the bands shown (e.g. 4, 1, 2 for false
  colour). **Stretch…** sets the contrast for one image, a root or the whole project.
  Display settings never change the files.
- **Inspect** shows every band's value under the cursor.
- **Show Map** places the roots on a map when the images carry GPS.
- **File > Save Project** writes `project.json`; polygons are saved as you go.

## 4. Open images from one or two folders

Projects start from image folders:

- **File > Open RGB/stack Folder** -- one folder of RGB images or multi-band stacks;
  you are asked how many images make one root (**File > Change Batch Size** changes it
  later).
- **File > Open Image files (Select Files…)** -- hand-picked files.
- **File > Open multispectral/multiple Folders** -- two cameras flown together. Try it
  on the example: choose `flight/multispectral` as the first folder and
  `flight/thermal` as the second. Frames are grouped into roots by their capture name
  (`IMG_0001_1.tif` ... `IMG_0001_4.tif` are root `IMG_0001`), and **Set Matching
  Roots** asks which thermal capture matches which multispectral one -- pick `IMG_0001`
  and `IMG_0101`, since each camera numbers its own frames. Each root then shows the
  four bands and the thermal frame of one capture. **Set Root Offset** changes the
  pairing later.

## 5. Draw and manage polygons

Back in the example project (File > Load Project): with **Drawing Mode: Polygon**,
click the vertices of a crown and close it; you are asked for its name (its *group*).
With **Sync On**, a polygon drawn in one viewer is placed at the same position on every
image of the root. Right-click a polygon to copy it to all viewers, edit its vertices
or properties, or delete it; **Copy Polys → All Roots** copies the current root's
polygons to every root.

The **Polygon Manager** lists every group: click one to find it, right-click to zoom,
edit, copy or move it, show statistics, import or export shapefiles (`.shp`), or export
the selected groups' pixels, thumbnails and segmentation masks for machine learning
(section 8).

## 6. The image editor

**Edit Image** opens the editor on a viewer's image. Every edit is saved in a small
`.ax` file next to the image and replayed by the viewer, the CSV and the
machine-learning exports alike -- the image file is never rewritten, so the original
radiometry is always there. The *Current Scope* sets where an edit goes (*Single image*,
*all images at this root*, or *all roots*); **Apply All Changes** applies it, **Reset
Image / Root / All** removes it.

- **Geometry**: crop (draw a box), rotate, resize (by percent or to an exact size).
- **NoData**: values or expressions to ignore, e.g. `0, b1<123`; **Pick** takes one
  from the image.
- **Histogram matching** (*Hist Norm*): *Mean/Std* or *CDF* match an image's
  distribution to a reference image -- open the editor on the reference (e.g. the
  2024-02-10 mosaic), choose the mode, press **Calc**, and apply it to the root: the
  later dates are brought to the reference's brightness, which removes the light
  differences between flights before comparing them. **Show Histogram** compares the
  two distributions.
- **Band registration** (*Mode*: Translation, Rigid Body, Scaled Rotation, Affine,
  Bilinear; tick **Use**): aligns the bands of a multi-camera capture (e.g. the
  MicaSense bands of `flight/`) to the current image when you **Apply All**.
- **Band math** (*Band Expr*): any expression of the bands `b1 ... bN` (`b1` = red,
  `b2` = green, `b3` = blue, `b4` = NIR here). `(b4-b1)/(b4+b1)` shows NDVI;
  **boolean** expressions such as `(b4-b1)/(b4+b1) > 0.6` or `(b1>150) & (b2<200)` show
  a mask. **+ Append Band** keeps the result as an extra band that every export sees.
- **Classify** (with *Use scikit-learn*) shows a trained model's classes (section 8).
- The JSON panel shows the `.ax` itself; **Apply from JSON** applies an edited one.

## 7. Extract statistics to CSV

**File > Extract polygons CSV from Current Project** (or *from All Projects*). The
options dialog chooses:

- the statistics: mean, median, standard deviation, quantiles, and the same over the
  whole image (*Scene Mean / Median / Std*);
- *Channel-math indices*: formulas computed per polygon, e.g. `NDVI = (b4-b1)/(b4+b1)`;
- *Shrink / Swell polygons* (a buffer, by a factor) and *Export modified polygons JSON*;
- *Include EXIF metadata* (needs ExifTool), *NoData masking*, a trained
  *scikit-learn pkl model* (adds class counts per polygon), a custom path, and *Also
  export as Shapefile*;

then how to run it (one file after another, in threads, or in the background while you
keep working). The CSV has one row per polygon, image and band (or index), with the
group (*Object ID*), file, root, centroid, pixel count and statistics. On the example,
with the NDVI formula, `tree_3`'s NDVI mean is **0.83** on the first two dates and
**0.42** on the last one, when it has lost its leaves -- once the crown is on every
date (section 10). **File > Extract all images EXIF** exports the metadata of every
image.

## 8. Machine learning

Open the **MachineLearning Manager** and tick the groups `leaf`, `bare` and
`understory` -- each group is a class. **Train Model(s)** then asks:

1. the training bands, in order, plus optional index expressions and a neighbourhood
   window (1×1, 3×3, 5×5 pixels);
2. how many pixels to sample per polygon;
3. which models to train: RandomForest, ExtraTrees, GradientBoosting,
   LogisticRegression, SVM, GaussianNB (XGBoost / LightGBM when installed);
4. the number of trees, an optional hyperparameter search, and data augmentation.

Each model is saved in `canopie_example/Machine_learning_models/` with a report:
classes, features, holdout accuracy, confusion matrix and feature importances. On the
example, a RandomForest on the default bands (R, G, B) reaches about **0.86** accuracy
(`bare` is never confused); add band 4 (NIR) and an NDVI expression in step 1 and train
again to see what the near-infrared adds. The model then classifies pixels in the image
editor (**Classify**) and per polygon in the CSV export.

The manager also writes training data for other tools: **Export CSV Data** (the pixel
values inside each polygon), **Generate Thumbnails** (an image around every polygon) and
**Generate Segmentation Masks** (image + mask pairs).

## 9. Phenocam series: filter similar images

Go to the `Phenocam` root. Long phenocam series hold frames taken in fog, rain or with
the sun in the lens. **File > Filter Similar Images...** scores every frame against one
to three reference frames you pick (*Threshold* sets how similar they must be), shows the
distribution of scores, and **Apply Filter (Keep Selected)** keeps the frames above the
threshold. Draw a crown on a frame with **Sync On** and export the CSV: the indices
(e.g. green chromatic coordinate, `b2/(b1+b2+b3)`) follow the canopy through the series.

## 10. Follow a crown through time

The five crowns were drawn on the first date only. To put them on the other two dates:

1. In the **Polygon Manager**, select `tree_1` ... `tree_5`, right-click and choose
   **Find canopy in other images…**.
2. Keep *Search in: Viewers of this root* (the two other dates are listed and ticked)
   and press **Find**.

Each crown is registered on each date and saved there under the same name. The table
shows the score and how far the crown was from where the georeference put it: about
**3.1 m east, 1.7 m south** on 2024-05-12 and **2.4 m west, 2.2 m north** on 2024-08-20
-- exactly the errors the example was built with. Scores are about 0.94; `tree_3` on
2024-08-20 scores lower (about 0.56): it has lost its leaves, yet it is found.

The same search works between orthomosaics and raw drone frames (located by their EXIF
GPS), between flights, in other projects or folders, and in phenocam frames. The
**Canopy browser** pages through each crown on every date: drag an outline to move it,
tick *Edit vertices* to reshape it, rotate a tile (only the view turns), or **Reject** a
date. **Export crowns** writes, per crown, the crop of every date aligned on the first
one (GeoTIFF or JPEG), each date's polygon (JSON) and a metadata CSV, by default into
`canopie_example/canopy_exports/`.

## 11. Run the tests

```sh
pip install -r requirements-dev.txt
python -m pytest canopie/qc -q -p no:cacheprovider
```

A report by feature is written to `canopie/qc/last_run_report.txt`.
