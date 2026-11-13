# Remote images (`.cogref`)

CanoPie can show a Cloud-Optimized GeoTIFF (COG) that lives on a web server or in
Google Drive without downloading it. It reads only the tiles on screen, plus the
file's header (a few hundred KB).

A remote image is represented by a small `.cogref` text file: the *pointer*.
CanoPie treats the pointer like the image itself. Its `.ax` edits, polygons and
place in `project.json` all go by the pointer's path. If you name the pointer
like an image you already worked on locally (`plot_A.tif` → `plot_A.cogref`), it
reuses that image's `.ax` and polygons.

## Try it with the examples in this folder

| Pointer | What it is |
|---|---|
| `sentinel2_TCI_deflate.cogref` | Sentinel-2 true colour, 10,980² px, DEFLATE, 1024-px tiles (235 MB) |
| `drone_ortho_jpeg_39199x30003.cogref` | drone orthomosaic from OpenAerialMap, JPEG, 512-px tiles (252 MB) |
| `drone_ortho_jpeg_42489x92594.cogref` | larger OpenAerialMap drone orthomosaic, JPEG (556 MB) |

All three are public files. **File > Open RGB/stack Folder** (Ctrl+R) on this
folder shows them as roots. With "images per root" set to 3, one root shows all
three side by side.

## Opening your own

**File > Add Remote Images (URLs)…** (Ctrl+Shift+U):

1. Paste the links, one per line: `https://…/ortho.tif`, or a Google Drive share link.
2. Choose where the pointer files go, and how many images each root shows.
3. Google Drive links also need an API key (below).
4. **Check and Open**. For each link CanoPie checks the file, writes `<name>.cogref` and
   reports its size and pyramid levels. A file that is not tiled, or has no overviews,
   is flagged as slow. Then CanoPie opens the files as roots, as **Open Image files** does.

From a terminal, in the CanoPie folder:

```
python -m canopie.remote_raster make "https://…/ortho.tif" D:\remote\ortho.cogref
python -m canopie.remote_raster info D:\remote\ortho.cogref
```

## Google Drive

**The file:** in Drive, right-click → Share → General access → **Anyone with the
link** (Viewer) → Copy link. Paste that link into the dialog. A Google Workspace
account (e.g. a university) may not allow link sharing. That is an admin setting.

**The API key** is free for this use:

1. Open <https://console.cloud.google.com/> and sign in.
2. Create a project (project picker at the top → New project).
3. **APIs & Services → Library** → search "Google Drive API" → **Enable**.
4. **APIs & Services → Credentials → Create credentials → API key**. Copy the key (it starts `AIza…`).
5. Recommended: edit the key → **API restrictions → Restrict key → Google Drive API**.

Paste the key into the dialog and tick **Remember on this computer**. CanoPie keeps
it in its own settings and loads it at start-up. It is never written into a
`.cogref`, so pointer files can be shared with a project. The other option is the
environment variable `CANOPIE_GDRIVE_API_KEY`.

**First test without your own data:** upload one of the public files above to a
test folder in your Drive. Its download link is the `url` in the matching
`.cogref`. Share the copy by link and open it with the dialog. You can then
compare it with the same file opened from S3.

## Make your files COGs (and smaller)

Speed depends on the bytes per tile. A DEFLATE orthomosaic like BCI averages
~0.9 MB per 512-px tile, so the first preview is ~38 MB. JPEG is ~10× smaller
(the example drone orthos are 25-40 KB per tile). With GDAL 3.1 or later:

```
gdal_translate ortho.tif ortho_cog.tif -of COG -co COMPRESS=JPEG -co QUALITY=90 -co BLOCKSIZE=512 -co OVERVIEWS=AUTO
```

JPEG takes 1 or 3 bands. If the 4th band of an RGBA orthomosaic is alpha, add
`-b 1 -b 2 -b 3 -mask 4`. JPEG is lossy: keep the original for pixel statistics,
or use `COMPRESS=DEFLATE -co PREDICTOR=2` for a lossless COG.

## What does not work on remote images yet

CSV/ML export, Inspect, Export Project Images and EXIF. They say so instead of
reading the whole file. Viewing, zooming, drawing polygons (stored in
full-resolution pixels), the image editor and the canopy finder work.
