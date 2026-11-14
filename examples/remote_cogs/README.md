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
3. Google Drive: optionally an API key (below).
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

**No key is needed** for a file shared by link. CanoPie then reads it from Drive's
public download address, which honours byte ranges. If the file is not shared,
CanoPie reports the Google sign-in page it got instead.

**An API key is optional.** With one, CanoPie uses the official Drive API, which is
documented and steadier if many people open the same file. The key is free:

1. Open <https://console.cloud.google.com/> and sign in.
2. Create a project (project picker at the top → New project).
3. **APIs & Services → Library** → search "Google Drive API" → **Enable**.
4. **APIs & Services → Credentials → Create credentials → API key**. Copy the key (it starts `AIza…`).
5. Recommended: edit the key → **API restrictions → Restrict key → Google Drive API**.

Paste the key into the dialog and tick **Remember on this computer**. CanoPie keeps
it in its own settings and loads it at start-up. It is never written into a
`.cogref`, so pointer files can be shared with a project. The other option is the
environment variable `CANOPIE_GDRIVE_API_KEY`.

**Large hyperspectral cubes:** a cube stored band-interleaved by pixel (every tile
holds all bands) costs whole tiles even when 3 bands are shown. A cube stored
band by band costs only the bands on screen. Measured on a 298-band AVIRIS-3 cube
(1,799 x 1,809, float32, ZSTD), in an 800 x 600 viewer with a Drive-like delay:

| Layout | Open | Zoom to 1:1 | Pan |
|---|---|---|---|
| pixel-interleaved (as delivered) | 18.9 MB, 1/8-resolution preview | 546 MB | 471 MB |
| band-interleaved | 12.9 MB, **full-resolution** preview | nothing | nothing |

The band-interleaved copy held identical pixels and was ~17% larger (1.33 → 1.56 GB).

**First test without your own data:** upload one of the public files above to a
test folder in your Drive. Its download link is the `url` in the matching
`.cogref`. Share the copy by link and open it with the dialog. You can then
compare it with the same file opened from S3.

## OneDrive and SharePoint

Paste a OneDrive share link (`1drv.ms/…`, `onedrive.live.com/…`) or a work/school
one (`…-my.sharepoint.com/:i:/…`) into the dialog. The file must be shared as
**Anyone with the link** (Can view). CanoPie turns the link into the file's
direct download address and reuses it. When that address expires, CanoPie
fetches a new one and carries on. The temporary token in it is never written
into the pointer. If your organisation does not allow link sharing, CanoPie
says so; reading such a file would need a sign-in, which CanoPie does not do.

So far this is tested against a stand-in of OneDrive's behaviour, not against
real OneDrive. The speed per request is not measured yet.

## Why Google Drive is slow, and what helps

Measured from this machine (November 2026) on the BCI orthomosaic:

- Drive takes **about 1 s to answer each request**, however small. After that it
  sends ~5 MB/s per stream. S3 answered in ~0.3 s at ~8 MB/s.
- Tiles fetched one after another therefore manage 0.6 MB/s. CanoPie merges
  neighbouring tiles into one request and runs separate requests in parallel.
  On a remote file it reads the coarsest pyramid level that still fills the
  screen, and it drops zoom requests you have already moved past.
- BCI is 8-bit data stored as 16-bit, DEFLATE without a predictor:
  0.94 MB per 512-px tile, so a 1:1 view (about 12 tiles) is ~11 MB. The same
  tiles as 8-bit JPEG would be ~0.10 MB; the example drone orthos are 25-40 KB per tile.

Hosting options:

| Where | Per-request delay | Cost for an 11 GB orthomosaic |
|---|---|---|
| Google Drive (link, or API key) | ~1 s | free (the Drive API has no charge, only request quotas) |
| Cloudflare R2 (public bucket) | not measured | 10 GB/month free, then ~$0.015/GB-month; no download fees |
| Amazon S3 / Google Cloud Storage | ~0.3 s (S3) | ~$0.02-0.03/GB-month, plus ~$0.09-0.12/GB downloaded beyond free allowances |

Prices are list prices at the time of writing; check the providers' pages.
Viewing reads tens of MB per session, so download fees stay at cents.

## What works on remote images

Viewing and zooming, with a "Streaming … %" counter in the status bar until the
tiles land and a line in the log for each download. Also: drawing polygons
(stored in full-resolution pixels), the image editor, the canopy finder,
Inspect, CSV export with statistics, and ML training and ML CSV export.

Analysis reads only the tiles under your polygons. Before a run of polygons,
CanoPie downloads the tiles of the next ones in one parallel round. Measured with
300 scattered crowns on a 39,199 x 30,003 orthomosaic behind a Drive-like delay:
24 s, against 325 s one polygon at a time, with identical rows.

Not available on remote images: Export Project Images (it would download the
whole file), EXIF (exiftool needs a local file), and `.ax` edits that need the
whole image (histogram matching, registration, classification, an up-size).
Each says so with the reason.
