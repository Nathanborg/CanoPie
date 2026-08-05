"""
Lazy, tile-level raster reading for CanoPie.

Why this exists
---------------
Every raster read in CanoPie used to go through a single full-array call
(``tifffile.imread``) and then slice the result in RAM. For AVIRIS-class
hyperspectral mosaics (hundreds of float32 bands) that is not merely slow, it
is impossible: a 284-band 1778x1921 float32 scene needs 3.88 GB resident, and
the CHW->HWC conversion that follows needs another 3.61 GB.

This module reads **only the TIFF tiles that intersect the requested window**,
which is what makes QGIS fast on the same files. It is GDAL-free and adds no
new dependencies: ``tifffile`` and ``imagecodecs`` are already required by
CanoPie, and between them they handle Deflate, ZSTD, the TIFF floating-point
predictor, planar configuration and partial edge tiles.

Design notes
------------
* ``TiffPage.decode(raw, index)`` is used rather than a hand-rolled
  decompress+reshape. It is the only thing that gets predictor 3 and the
  various codecs right, and it was verified byte-identical to a manual
  zlib decode on the reference imagery.
* ``imagecodecs`` releases the GIL, so a plain ``ThreadPoolExecutor`` scales
  nearly linearly on tile decode. Worker count is *interleave-aware*: one tile
  of a band-interleaved (planar=1) 298-band file is 78 MB uncompressed, so
  running 12 of those concurrently is slower than running 2.
* The tile cache is bounded by **bytes**, not tile count. A count-bounded
  cache reached 2.68 GB and raised MemoryError on the reference file.

This module deliberately has no Qt dependency so it can be exercised headlessly
(see ``selftest_against_full``).
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import tifffile
    _TIFFFILE_AVAILABLE = True
except Exception:  # pragma: no cover - tifffile is a hard requirement in practice
    tifffile = None
    _TIFFFILE_AVAILABLE = False


__all__ = [
    "RasterProfile",
    "TiledRasterReader",
    "LazyChannels",
    "probe",
    "open_reader",
    "clear_reader_cache",
    "ensure_hwc",
    "axes_for",
    "selftest_against_full",
]


# Band-first axis labels as tifffile reports them in ``series.axes``.
_BAND_FIRST_AXES = ("S", "C", "I", "Z", "Q")


def axes_for(path):
    """``series[0].axes`` for a TIFF, or ``None``. Header-only, so it is cheap."""
    if not _TIFFFILE_AVAILABLE:
        return None
    try:
        if os.path.splitext(str(path))[1].lower() not in (".tif", ".tiff"):
            return None
        with tifffile.TiffFile(path) as tf:
            return (tf.series[0].axes or "").upper() or None
    except Exception:
        return None


def ensure_hwc(arr, axes=None):
    """Normalise an array to H x W x C.

    CanoPie previously carried four different versions of this, with band-count
    ceilings of 32, 64, and "<= min(H, W)". A 284-band ``axes='SYX'`` scene
    failed the 32 and 64 tests, so the cube was left as (284, 1778, 1921) and
    then read as H=284, W=1778, C=1921 -- bands silently became rows, which
    corrupted CSV export on every file with more than 64 bands.

    Pass ``axes`` (from :func:`axes_for` or ``RasterProfile.axes``) whenever the
    array came from a TIFF; the shape heuristic is only a fallback for OpenCV /
    PIL arrays, which are always channel-last anyway.

    Returns a view where possible -- callers that need contiguity should say so,
    rather than paying for a copy of a multi-GB cube here.
    """
    if arr is None:
        return None
    a = np.asarray(arr)

    if a.ndim == 2:
        return a[..., None]

    if a.ndim == 3:
        if axes:
            ax = str(axes).upper()
            if len(ax) >= 3 and ax[0] in _BAND_FIRST_AXES:
                return np.moveaxis(a, 0, -1)
            return a
        d0, d1, d2 = a.shape
        # No axes metadata: only call it band-first when the leading axis is
        # smaller than *both* spatial axes. (H, W, 3) is unaffected because
        # H < 3 is false; (284, 1778, 1921) is correctly transposed.
        if d0 < d1 and d0 < d2:
            return np.moveaxis(a, 0, -1)
        return a

    if a.ndim == 4:
        # (pages, H, W, samples) -> fold pages into channels.
        pages = a.shape[0]
        if a.shape[-1] in (1, 3, 4):
            return np.concatenate([a[i] for i in range(pages)], axis=2)
        return np.moveaxis(a.reshape(pages, a.shape[-2], a.shape[-1]), 0, -1)

    return a


# Strategy names. 'full' means "fall back to the legacy whole-array read".
STRATEGY_MEMMAP = "memmap"
STRATEGY_TILES_BSQ = "tiles_bsq"
STRATEGY_TILES_BIP = "tiles_bip"
STRATEGY_STRIPS = "strips"

#: TIFF Compression tag values that mean "JPEG" in one of its variants:
#: 6 = old-style JPEG, 7 = JPEG (the one GDAL writes for COGs),
#: 34892 = lossy DNG JPEG, 33007 = Alias/JPEG. Segments in these files need
#: the page's shared jpegtables/jpegheader handed to decode() -- see
#: _level_geometry. Value set matches tifffile's own TiffPage.segments().
_JPEG_COMPRESSIONS = frozenset({6, 7, 34892, 33007})
STRATEGY_FULL = "full"

_WINDOWABLE_STRATEGIES = (
    STRATEGY_MEMMAP,
    STRATEGY_TILES_BSQ,
    STRATEGY_TILES_BIP,
    STRATEGY_STRIPS,
)

# Default ceiling for decoded tile bytes held in memory, across all readers.
# Sized so a whole polygon's worth of tiles across a few hundred bands fits and
# can be reused by the next polygon: 256x256 float32 tiles x 284 bands x 4 tiles
# is ~291 MB, and a budget below that turns every polygon into a full re-decode.
_DEFAULT_CACHE_BYTES = 640 * 1024 * 1024


def _cpu_workers(default=12):
    try:
        n = os.cpu_count() or default
    except Exception:
        n = default
    return max(1, min(default, n))


# How much decoded tile data may be in flight at once. Tiles are transient, but
# running 12 workers over 78 MB band-interleaved tiles thrashes: 4 workers on
# those was slower than 1 (8.4 s). Concurrency therefore follows tile size.
_INFLIGHT_TILE_BYTES = 192 * 1024 * 1024


def workers_for_tile_bytes(tile_bytes, cap=None):
    """Thread count for decoding tiles of the given decoded size."""
    cap = cap or _cpu_workers()
    if not tile_bytes or tile_bytes <= 0:
        return cap
    return max(1, min(cap, _INFLIGHT_TILE_BYTES // int(tile_bytes) or 1))


# --------------------------------------------------------------------------
# Profile (header-only probe)
# --------------------------------------------------------------------------

class RasterProfile:
    """Everything we can learn about a raster without decoding a single pixel.

    Probing costs 0.02-0.08 s even on a file with 16,688 tile offsets, so this
    is cheap enough to call on every image in a folder.
    """

    __slots__ = (
        "path", "width", "height", "count", "dtype", "axes",
        "tiled", "tile_w", "tile_h", "planar", "compression", "predictor",
        "n_levels", "level_shapes", "band_names", "wavelengths", "nodata",
        "epsg", "transform", "strategy", "channel_origin",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    @property
    def is_windowable(self):
        return self.strategy in _WINDOWABLE_STRATEGIES

    @property
    def is_band_interleaved(self):
        """True for planar=1 (BIP/contig): one tile holds every band."""
        return self.planar == 1

    @property
    def bytes_if_fully_loaded(self):
        try:
            return int(self.width) * int(self.height) * int(self.count) * np.dtype(self.dtype).itemsize
        except Exception:
            return 0

    def workers(self):
        """Tile-decode concurrency for level 0. Prefer TiledRasterReader.workers_for."""
        return workers_for_tile_bytes(self.tile_bytes(), _cpu_workers())

    def tile_bytes(self, level=0):
        """Decoded size of one tile at level 0 (see reader for other levels)."""
        try:
            tw = self.tile_w or self.width
            th = self.tile_h or self.height
            per_pixel = self.count if self.planar == 1 else 1
            return int(tw) * int(th) * int(per_pixel) * np.dtype(self.dtype).itemsize
        except Exception:
            return 0

    def describe(self):
        return (
            "{name}: {w}x{h}x{c} {dt} | strategy={s} planar={p} "
            "tiled={t} tile={tw}x{th} comp={comp} pred={pred} levels={lv}".format(
                name=os.path.basename(self.path or "?"),
                w=self.width, h=self.height, c=self.count, dt=self.dtype,
                s=self.strategy, p=self.planar, t=self.tiled,
                tw=self.tile_w, th=self.tile_h,
                comp=self.compression, pred=self.predictor, lv=self.n_levels,
            )
        )

    def __repr__(self):
        return "<RasterProfile {}>".format(self.describe())


_BAND_DESC_RE = re.compile(r'sample="(\d+)"[^>]*role="description">([^<]*)<')
_WAVELENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)


def _parse_band_metadata(page, count):
    """Pull per-band names (and wavelengths, where present) from GDAL_METADATA.

    On the reference imagery this yields 284/284 band names, and on the COG
    variant 284 of 298 names carry an explicit wavelength (``RFL_388.17nm``).
    """
    names = [None] * count
    try:
        tag = page.tags.get("GDAL_METADATA")
        if tag is not None and tag.value:
            for idx_s, desc in _BAND_DESC_RE.findall(tag.value):
                i = int(idx_s)
                if 0 <= i < count:
                    names[i] = desc.strip()
    except Exception as exc:
        logging.debug("GDAL_METADATA parse failed: %s", exc)

    wavelengths = [None] * count
    for i, nm in enumerate(names):
        if not nm:
            continue
        m = _WAVELENGTH_RE.search(nm)
        if m:
            try:
                wavelengths[i] = float(m.group(1))
            except ValueError:
                pass
    return names, wavelengths


def _parse_nodata(page):
    try:
        tag = page.tags.get("GDAL_NODATA")
        if tag is None or tag.value is None:
            return None
        raw = str(tag.value).strip()
        if raw.lower() in ("nan", "-nan", "+nan"):
            return float("nan")
        return float(raw)
    except Exception:
        return None


def _parse_geo(page):
    """Lightweight EPSG + affine transform straight off the open page.

    ``shapefile_io`` remains the authority for shapefile export; this is here so
    callers holding a profile do not have to reopen the file. Only the common
    ModelPixelScale + ModelTiepoint form is handled (same subset shapefile_io
    supports).
    """
    epsg = None
    transform = None
    try:
        gk = page.tags.get("GeoKeyDirectoryTag")
        if gk is not None and gk.value:
            keys = list(gk.value)
            # Header is 4 shorts, then 4-short entries: id, location, count, value.
            for i in range(4, len(keys) - 3, 4):
                key_id, location, _count, value = keys[i:i + 4]
                if location != 0:
                    continue
                if key_id == 3072:          # ProjectedCSTypeGeoKey wins
                    epsg = int(value)
                    break
                if key_id == 2048 and epsg is None:   # GeographicTypeGeoKey
                    epsg = int(value)
    except Exception as exc:
        logging.debug("GeoKeyDirectoryTag parse failed: %s", exc)

    try:
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is not None and tie_tag is not None:
            sx, sy = float(scale_tag.value[0]), float(scale_tag.value[1])
            tie = list(tie_tag.value)
            ox, oy = float(tie[3]), float(tie[4])
            transform = [ox, sx, 0.0, oy, 0.0, -sy]
    except Exception as exc:
        logging.debug("Model*Tag parse failed: %s", exc)

    return epsg, transform


def _choose_strategy(page, tiled, planar):
    try:
        if page.is_memmappable:
            return STRATEGY_MEMMAP
    except Exception:
        pass
    if tiled:
        return STRATEGY_TILES_BIP if planar == 1 else STRATEGY_TILES_BSQ
    try:
        if page.rowsperstrip and len(page.dataoffsets) > 1:
            return STRATEGY_STRIPS
    except Exception:
        pass
    return STRATEGY_FULL


def probe(path):
    """Read headers only and decide how this file should be read.

    Returns ``None`` when the file is not a TIFF we can plan for; callers should
    then use their existing full-read path.
    """
    if not _TIFFFILE_AVAILABLE:
        return None
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in (".tif", ".tiff"):
        return None

    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            page = tf.pages[0]

            axes = (series.axes or "").upper()
            shape = tuple(int(v) for v in series.shape)

            width = int(page.imagewidth)
            height = int(page.imagelength)
            count = int(page.samplesperpixel or 1)
            # A multi-page stack (one band per page) reports spp=1 per page.
            if count == 1 and len(tf.pages) > 1 and len(shape) == 3:
                count = len(tf.pages)

            planar = int(page.planarconfig) if page.planarconfig else 1
            tiled = bool(page.is_tiled)
            tile_w = int(page.tilewidth) if tiled else 0
            tile_h = int(page.tilelength) if tiled else 0

            level_shapes = []
            try:
                level_shapes = [tuple(int(v) for v in lv.shape) for lv in series.levels]
            except Exception:
                level_shapes = [shape]

            band_names, wavelengths = _parse_band_metadata(page, count)
            nodata = _parse_nodata(page)
            epsg, transform = _parse_geo(page)

            strategy = _choose_strategy(page, tiled, planar)
            # Multi-page band stacks do not fit the single-page segment math below.
            if len(tf.pages) > 1 and count == len(tf.pages) and strategy != STRATEGY_MEMMAP:
                strategy = STRATEGY_FULL

            return RasterProfile(
                path=os.path.abspath(str(path)),
                width=width, height=height, count=count,
                dtype=np.dtype(page.dtype), axes=axes,
                tiled=tiled, tile_w=tile_w, tile_h=tile_h,
                planar=planar,
                compression=int(page.compression),
                predictor=int(page.predictor) if page.predictor else 1,
                n_levels=len(level_shapes), level_shapes=level_shapes,
                band_names=band_names, wavelengths=wavelengths, nodata=nodata,
                epsg=epsg, transform=transform,
                strategy=strategy,
                # tifffile hands back native band order; cv2 would hand back BGR.
                channel_origin="tifffile",
            )
    except Exception as exc:
        logging.debug("probe failed for %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------
# Byte-budgeted tile cache
# --------------------------------------------------------------------------

class _TileCache:
    """LRU over decoded tiles, evicted on total bytes.

    Shared by all readers so that the budget is global rather than per-file.
    """

    def __init__(self, budget_bytes=_DEFAULT_CACHE_BYTES):
        self._budget = int(budget_bytes)
        self._data = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            arr = self._data.get(key)
            if arr is not None:
                self._data.move_to_end(key)
            return arr

    def put(self, key, arr):
        nbytes = int(getattr(arr, "nbytes", 0))
        # A single tile larger than the whole budget is never worth caching.
        if nbytes > self._budget:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= int(self._data[key].nbytes)
            self._data[key] = arr
            self._data.move_to_end(key)
            self._bytes += nbytes
            while self._bytes > self._budget and self._data:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= int(getattr(evicted, "nbytes", 0))

    def drop_path(self, path):
        with self._lock:
            for key in [k for k in self._data if k[0] == path]:
                self._bytes -= int(getattr(self._data.pop(key), "nbytes", 0))

    def clear(self):
        with self._lock:
            self._data.clear()
            self._bytes = 0

    @property
    def nbytes(self):
        with self._lock:
            return self._bytes

    @property
    def budget(self):
        return self._budget

    def set_budget(self, budget_bytes):
        with self._lock:
            self._budget = int(budget_bytes)


_TILE_CACHE = _TileCache()


def set_cache_budget(budget_bytes):
    _TILE_CACHE.set_budget(budget_bytes)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

class TiledRasterReader:
    """Windowed reads against one TIFF, using its native tile/strip grid."""

    def __init__(self, profile):
        if not _TIFFFILE_AVAILABLE:
            raise RuntimeError("tifffile is required for TiledRasterReader")
        self.profile = profile
        self.path = profile.path
        self._tf = tifffile.TiffFile(self.path)
        self._series = self._tf.series[0]
        self._local = threading.local()
        self._lock = threading.Lock()
        self._tpool = None
        # Thread-local handles are unreachable from close(), so keep a registry.
        self._handles = []

        self._levels = []
        try:
            level_pages = [lv.pages[0] for lv in self._series.levels]
        except Exception:
            level_pages = [self._tf.pages[0]]
        for page in level_pages:
            self._levels.append(self._level_geometry(page))

        # Build each decoder once, up front: TiffPage.decode is a cached
        # property, and racing its construction across threads is not worth it.
        for lv in self._levels:
            _ = lv["page"].decode

        self._memmap = None
        if profile.strategy == STRATEGY_MEMMAP:
            try:
                self._memmap = tifffile.memmap(self.path)
            except Exception as exc:
                logging.debug("memmap failed for %s: %s", self.path, exc)
                self._memmap = None

    # -- geometry ---------------------------------------------------------

    @staticmethod
    def _level_geometry(page):
        width = int(page.imagewidth)
        height = int(page.imagelength)
        count = int(page.samplesperpixel or 1)
        planar = int(page.planarconfig) if page.planarconfig else 1
        if page.is_tiled:
            tw, th = int(page.tilewidth), int(page.tilelength)
        else:
            # Treat each strip as a full-width tile.
            tw = width
            th = int(page.rowsperstrip) or height
            th = min(th, height) if th > 0 else height
        across = max(1, math.ceil(width / tw))
        down = max(1, math.ceil(height / th))
        # Concurrency is per level: an overview's tiles are much smaller than
        # level 0's, so a band-interleaved file that wants 2 workers at full
        # resolution can use all of them one level down.
        tile_bytes = tw * th * (count if planar == 1 else 1) * np.dtype(page.dtype).itemsize

        # JPEG-in-TIFF keeps its quantization/Huffman tables ONCE in the
        # JPEGTables tag (and, for some encoders, a shared jpegheader) rather
        # than repeating them in every tile -- which is exactly what GDAL
        # writes for a JPEG-compressed COG. Passing the raw tile bytes to
        # decode() without those tables fails with
        # "Quantization table 0x00 was not defined", because the tile alone is
        # only entropy-coded scan data and genuinely cannot be decoded on its
        # own. Mirrors tifffile's own TiffPage.segments():
        #     if compression in {6, 7, 34892, 33007}:   # JPEG variants
        #         decodeargs['jpegtables'] = page.jpegtables
        #         decodeargs['jpegheader'] = page.jpegheader
        # Resolved once per level here (they are per-page constants) rather
        # than per tile. Note tifffile's own writer often stores the tables
        # inline instead, leaving page.jpegtables None -- so a locally-written
        # JPEG TIFF may decode fine while a real GDAL COG does not, which is
        # why this went unnoticed.
        decode_kwargs = {}
        try:
            compression = int(getattr(page, "compression", 0) or 0)
        except (TypeError, ValueError):
            compression = 0
        if compression in _JPEG_COMPRESSIONS:
            decode_kwargs["jpegtables"] = getattr(page, "jpegtables", None)
            decode_kwargs["jpegheader"] = getattr(page, "jpegheader", None)

        return {
            "page": page, "width": width, "height": height, "count": count,
            "planar": planar, "tw": tw, "th": th,
            "across": across, "down": down, "per_plane": across * down,
            "offsets": page.dataoffsets, "counts": page.databytecounts,
            "tile_bytes": tile_bytes,
            "workers": workers_for_tile_bytes(tile_bytes),
            "decode_kwargs": decode_kwargs,
        }

    def level_shape(self, level=0):
        lv = self._levels[level]
        return lv["height"], lv["width"], lv["count"]

    def overview_for(self, max_dim):
        """Smallest pyramid level whose longest side is still >= ``max_dim``."""
        best = 0
        for i, lv in enumerate(self._levels):
            if max(lv["width"], lv["height"]) >= max_dim:
                best = i
            else:
                break
        return best

    def level_for_bytes(self, n_bands, max_bytes, decode_budget=None, target_dim=2400):
        """Highest-resolution level that satisfies both output and decode limits.

        Two separate costs have to be respected:

        * **output** - a 7-band 19499x17250 mosaic is only 3 bands on screen but
          that is still 4.04 GB at full resolution.
        * **decode** - on a band-interleaved (planar=1) file one tile carries
          *every* band, so asking for 3 of 298 bands still decompresses the whole
          cube. Reading a 298-band BIP COG at level 0 took 94 s for a 39 MB
          result; counting the decode cost picks a pyramid level instead.

        Falls back to the smallest level; the caller can decimate from there.
        """
        itemsize = np.dtype(self.profile.dtype).itemsize
        if decode_budget is None:
            decode_budget = max_bytes * 2

        fits = []
        for i, lv in enumerate(self._levels):
            pixels = lv["width"] * lv["height"]
            out_bytes = pixels * n_bands * itemsize
            # planar=1 decodes every band whether or not it is wanted.
            decode_bands = lv["count"] if lv["planar"] == 1 else n_bands
            decode_bytes = pixels * decode_bands * itemsize
            if out_bytes <= max_bytes and decode_bytes <= decode_budget:
                fits.append(i)
        if not fits:
            return len(self._levels) - 1

        # Among the levels that fit, take the SMALLEST one still larger than a
        # screen. Decoding a 4875 px pyramid level to show it on a ~2000 px
        # canvas is wasted work -- on the coarse mosaic that was 588 MB of
        # inflate versus 147 MB one level down, for no visible difference.
        if target_dim:
            for i in reversed(fits):        # levels run large -> small
                lv = self._levels[i]
                if max(lv["width"], lv["height"]) >= target_dim:
                    return i
        return fits[0]

    def step_for_bytes(self, n_bands, max_bytes, level):
        """Integer decimation factor needed at ``level`` to fit ``max_bytes``."""
        lv = self._levels[level]
        itemsize = np.dtype(self.profile.dtype).itemsize
        need = lv["width"] * lv["height"] * n_bands * itemsize
        if need <= max_bytes or max_bytes <= 0:
            return 1
        return int(math.ceil(math.sqrt(need / float(max_bytes))))

    def _segment_index(self, lv, band, trow, tcol):
        spatial = trow * lv["across"] + tcol
        if lv["planar"] == 2:
            return band * lv["per_plane"] + spatial
        return spatial

    def _spatial_tiles(self, lv, x0, y0, x1, y1):
        tc0, tc1 = x0 // lv["tw"], (x1 - 1) // lv["tw"]
        tr0, tr1 = y0 // lv["th"], (y1 - 1) // lv["th"]
        return [(tr, tc) for tr in range(tr0, tr1 + 1) for tc in range(tc0, tc1 + 1)]

    # -- I/O --------------------------------------------------------------

    def _pool(self):
        """One long-lived pool per reader.

        Creating a ThreadPoolExecutor per call is not free: band-major
        iteration over 284 bands spun up 284 pools and spent 2.1 s where the
        equivalent single window read took 0.19 s.
        """
        pool = getattr(self, "_tpool", None)
        if pool is None:
            with self._lock:
                pool = getattr(self, "_tpool", None)
                if pool is None:
                    # Sized for the widest level so every level can use its own
                    # concurrency; _decode_many caps the chunk count per call.
                    pool = ThreadPoolExecutor(
                        max_workers=max(lv["workers"] for lv in self._levels),
                        thread_name_prefix="canopie-tile")
                    self._tpool = pool
        return pool

    def _handle(self):
        """One file handle per thread; reused across tiles."""
        fh = getattr(self._local, "fh", None)
        if fh is None or fh.closed:
            fh = open(self.path, "rb", buffering=0)
            self._local.fh = fh
            with self._lock:
                self._handles.append(fh)
        return fh

    def _decode_segment(self, level, seg_index):
        key = (self.path, level, seg_index)
        cached = _TILE_CACHE.get(key)
        if cached is not None:
            return cached

        arr = self._decode_uncached(level, seg_index)
        _TILE_CACHE.put(key, arr)
        return arr

    def _decode_uncached(self, level, seg_index):
        lv = self._levels[level]
        offset = lv["offsets"][seg_index]
        nbytes = lv["counts"][seg_index]
        if nbytes == 0:          # sparse tile: TIFF allows an empty segment
            shape = (lv["th"], lv["tw"]) if lv["planar"] == 2 else (lv["th"], lv["tw"], lv["count"])
            arr = np.zeros(shape, dtype=self.profile.dtype)
        else:
            fh = self._handle()
            fh.seek(offset)
            raw = fh.read(nbytes)
            data = lv["page"].decode(raw, seg_index, **lv.get("decode_kwargs", {}))[0]
            arr = np.squeeze(np.asarray(data))
            # Squeeze can flatten a legitimate single-band trailing axis.
            if lv["planar"] == 1 and arr.ndim == 2 and lv["count"] > 1:
                arr = arr[..., None]
        return arr

    def _decode_many(self, level, seg_indices, workers):
        """Decode segments, sorted by file offset, across a thread pool.

        The returned dict is the caller's working set and holds strong
        references, so it is never at the mercy of cache eviction. Feeding the
        working set *through* the LRU instead made a 291 MB window thrash
        against a 256 MB budget and re-decode tiles serially (2.7 s vs 0.19 s).
        The shared cache is populated opportunistically on top of that.
        """
        lv = self._levels[level]
        wanted = list(dict.fromkeys(seg_indices))
        result = {}
        todo = []
        for i in wanted:
            hit = _TILE_CACHE.get((self.path, level, i))
            if hit is not None:
                result[i] = hit
            else:
                todo.append(i)
        if not todo:
            return result

        # Sequential file order keeps the OS/Drive readahead useful.
        todo.sort(key=lambda i: lv["offsets"][i])
        lock = threading.Lock()

        def run(chunk):
            local = [(i, self._decode_uncached(level, i)) for i in chunk]
            with lock:
                result.update(local)

        if workers <= 1 or len(todo) == 1:
            run(todo)
        else:
            nw = min(workers, len(todo))
            chunks = [c for c in (todo[k::nw] for k in range(nw)) if c]
            list(self._pool().map(run, chunks))

        # Only cache a working set that can actually survive in the cache. A
        # window bigger than the budget evicts itself as it is inserted, so every
        # put runs the eviction loop and nothing is retained (0.76 s vs 0.37 s
        # for not caching at all). Anything that fits is worth keeping: one
        # 36-px polygon over 284 bands is ~291 MB of tiles, and the next polygon
        # nearby reuses them, so the threshold must not be stricter than this.
        fresh_bytes = sum(int(result[i].nbytes) for i in todo)
        if fresh_bytes <= _TILE_CACHE.budget:
            for i in todo:
                _TILE_CACHE.put((self.path, level, i), result[i])
        return result

    # -- public reads -----------------------------------------------------

    def read_window(self, x0, y0, x1, y1, bands=None, level=0):
        """Read ``[y0:y1, x0:x1]`` for ``bands``. Returns (h, w, C)."""
        lv = self._levels[level]
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(lv["width"], int(x1)); y1 = min(lv["height"], int(y1))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 0), dtype=self.profile.dtype)

        if bands is None:
            bands = range(lv["count"])
        bands = [int(b) for b in bands]

        if self._memmap is not None:
            return self._read_window_memmap(x0, y0, x1, y1, bands)

        h, w = y1 - y0, x1 - x0
        # Assemble band-major: out[bi] is a contiguous 2D block, so each blit is
        # a straight memcpy. Filling an (h, w, C) array through out[:, :, bi]
        # instead means a strided scatter with a C-element stride, which cost
        # 0.94 s versus 0.19 s for a 300x300x284 window.
        out = np.empty((len(bands), h, w), dtype=self.profile.dtype)
        tiles = self._spatial_tiles(lv, x0, y0, x1, y1)
        workers = lv["workers"]

        # The overlap geometry is identical for every band, so compute the slice
        # pairs once instead of per (band, tile). With 284 bands that removed
        # ~22k redundant bound calculations per 20 polygons.
        geom = []
        for (tr, tc) in tiles:
            ty0, tx0 = tr * lv["th"], tc * lv["tw"]
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + lv["th"])
            sx0, sx1 = max(x0, tx0), min(x1, tx0 + lv["tw"])
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            geom.append((tr, tc,
                         slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0),
                         slice(sy0 - ty0, sy1 - ty0), slice(sx0 - tx0, sx1 - tx0)))

        seg_index = self._segment_index
        if lv["planar"] == 2:
            seg = [seg_index(lv, b, tr, tc) for b in bands for (tr, tc) in tiles]
            decoded = self._decode_many(level, seg, workers)
            for bi, b in enumerate(bands):
                dst = out[bi]
                for (tr, tc, dy, dx, sy, sx) in geom:
                    dst[dy, dx] = decoded[seg_index(lv, b, tr, tc)][sy, sx]
        else:
            seg = [seg_index(lv, 0, tr, tc) for (tr, tc) in tiles]
            decoded = self._decode_many(level, seg, workers)
            for (tr, tc, dy, dx, sy, sx) in geom:
                arr = decoded[seg_index(lv, 0, tr, tc)]
                patch = arr[sy, sx]
                for bi, b in enumerate(bands):
                    out[bi][dy, dx] = patch[..., b] if patch.ndim == 3 else patch
        # A view, not a copy. Note that cube[:, :, i] on the result is still
        # contiguous, which is what the per-band consumers downstream want.
        return np.moveaxis(out, 0, -1)

    @staticmethod
    def _blit(dst, tile, lv, trow, tcol, x0, y0, x1, y1):
        """Copy the overlap of one (possibly padded edge) tile into ``dst``."""
        ty0, tx0 = trow * lv["th"], tcol * lv["tw"]
        sy0, sy1 = max(y0, ty0), min(y1, ty0 + lv["th"])
        sx0, sx1 = max(x0, tx0), min(x1, tx0 + lv["tw"])
        if sy1 <= sy0 or sx1 <= sx0:
            return
        dst[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
            tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]

    def _read_window_memmap(self, x0, y0, x1, y1, bands):
        mm = self._memmap
        axes = (self.profile.axes or "").upper()
        if mm.ndim == 2:
            return np.asarray(mm[y0:y1, x0:x1])[..., None]
        if axes.startswith(("S", "C", "I", "Z")):        # (bands, H, W)
            sub = mm[:, y0:y1, x0:x1][bands, :, :]
            return np.ascontiguousarray(np.moveaxis(sub, 0, -1))
        return np.ascontiguousarray(mm[y0:y1, x0:x1, :][:, :, bands])

    def read_bands_full(self, bands, level=0, step=1):
        """Whole frame, selected bands. Used for viewer composites.

        ``step`` > 1 decimates while assembling, one tile at a time, so peak
        memory is the (small) output plus a single tile rather than the whole
        level. That is what makes a 19499x17250 scene displayable when it has
        no usable pyramid.
        """
        lv = self._levels[level]
        if step <= 1:
            return self.read_window(0, 0, lv["width"], lv["height"], bands=bands, level=level)

        bands = [int(b) for b in bands]
        step = int(step)
        out_h = (lv["height"] + step - 1) // step
        out_w = (lv["width"] + step - 1) // step
        out = np.empty((len(bands), out_h, out_w), dtype=self.profile.dtype)
        workers = lv["workers"]

        # Walk tile rows so only one row of tiles is resident at a time.
        for tr in range(lv["down"]):
            ty0 = tr * lv["th"]
            ty1 = min(lv["height"], ty0 + lv["th"])
            # First sampled row at or after ty0.
            sy0 = ((ty0 + step - 1) // step) * step
            if sy0 >= ty1:
                continue
            rows = np.arange(sy0, ty1, step)
            if lv["planar"] == 2:
                seg = [self._segment_index(lv, b, tr, tc)
                       for b in bands for tc in range(lv["across"])]
            else:
                seg = [self._segment_index(lv, 0, tr, tc) for tc in range(lv["across"])]
            decoded = self._decode_many(level, seg, workers)
            for tc in range(lv["across"]):
                tx0 = tc * lv["tw"]
                tx1 = min(lv["width"], tx0 + lv["tw"])
                sx0 = ((tx0 + step - 1) // step) * step
                if sx0 >= tx1:
                    continue
                cols = np.arange(sx0, tx1, step)
                for bi, b in enumerate(bands):
                    if lv["planar"] == 2:
                        tile = decoded[self._segment_index(lv, b, tr, tc)]
                    else:
                        t = decoded[self._segment_index(lv, 0, tr, tc)]
                        tile = t[..., b] if t.ndim == 3 else t
                    patch = tile[np.ix_(rows - ty0, cols - tx0)]
                    out[bi,
                        sy0 // step: sy0 // step + patch.shape[0],
                        sx0 // step: sx0 // step + patch.shape[1]] = patch
        return np.moveaxis(out, 0, -1)

    def iter_band_windows(self, bands, x0, y0, x1, y1, level=0):
        """Yield ``(band, (h, w) array)`` one band at a time.

        This is the bounded-memory path for whole-frame work (FullFrame
        polygons, scene statistics): peak usage is the tiles of a single band
        rather than the whole cube. Measured peak on the reference file was
        9 MB for 50 scattered polygons versus 3.88 GB for a full read.
        """
        for b in bands:
            win = self.read_window(x0, y0, x1, y1, bands=[b], level=level)
            yield b, win[:, :, 0]

    def close(self):
        pool, self._tpool = self._tpool, None
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception:
                pass
        with self._lock:
            handles, self._handles = self._handles, []
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass
        try:
            self._tf.close()
        except Exception:
            pass
        self._memmap = None
        _TILE_CACHE.drop_path(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------
# Reader cache
# --------------------------------------------------------------------------

_READERS = OrderedDict()
_READERS_LOCK = threading.Lock()
_MAX_READERS = 8


def open_reader(path, profile=None):
    """Open (or reuse) a reader. Returns ``None`` if the file is not windowable."""
    if profile is None:
        profile = probe(path)
    if profile is None or not profile.is_windowable:
        return None

    abspath = os.path.abspath(str(path))
    try:
        mtime = os.path.getmtime(abspath)
    except Exception:
        mtime = 0.0
    key = (os.path.normcase(abspath), mtime)

    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is not None:
            _READERS.move_to_end(key)
            return reader

    try:
        reader = TiledRasterReader(profile)
    except Exception as exc:
        logging.debug("open_reader failed for %s: %s", path, exc)
        return None

    with _READERS_LOCK:
        existing = _READERS.get(key)
        if existing is not None:
            reader.close()
            return existing
        _READERS[key] = reader
        _READERS.move_to_end(key)
        while len(_READERS) > _MAX_READERS:
            _, old = _READERS.popitem(last=False)
            old.close()
    return reader


def clear_reader_cache():
    with _READERS_LOCK:
        readers = list(_READERS.values())
        _READERS.clear()
    for r in readers:
        r.close()
    _TILE_CACHE.clear()


# --------------------------------------------------------------------------
# LazyChannels — drop-in for the `chans` list in ProjectTab.process_polygon
# --------------------------------------------------------------------------

class _LazyWindowList:
    """List of per-band 2D windows, decoded on access rather than up front.

    Only used for windows too large to hold as one cube. Keeps a small ring of
    recently touched bands so that ``roi_slices[0]``, ``[1]``, ``[2]`` in a row
    (the RGB triplet) does not re-decode.
    """

    _KEEP = 4

    def __init__(self, reader, order, x0, y0, x1, y1, level):
        self._reader = reader
        self._order = list(order)
        self._box = (x0, y0, x1, y1)
        self._level = level
        self._recent = OrderedDict()

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        for i in range(len(self._order)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._order)))]
        if index < 0:
            index += len(self._order)
        hit = self._recent.get(index)
        if hit is not None:
            self._recent.move_to_end(index)
            return hit
        x0, y0, x1, y1 = self._box
        arr = self._reader.read_window(
            x0, y0, x1, y1, bands=[self._order[index]], level=self._level)[:, :, 0]
        self._recent[index] = arr
        while len(self._recent) > self._KEEP:
            self._recent.popitem(last=False)
        return arr


class OffsetBand:
    """A window of one band that is still indexed in FULL-image coordinates.

    Point-sampling code (notably the ML manager) walks ``chans[b][yi, xi]`` with
    image coordinates for every point and every band. Against a LazyChannels
    that would decode a whole band per access. Reading the points' bounding box
    once and wrapping each band here keeps all that indexing code unchanged
    while touching only the tiles the points actually fall in.

    ``shape`` deliberately reports the full image size, because callers use it
    for bounds checks.
    """

    __slots__ = ("_a", "_x0", "_y0", "shape", "dtype")

    def __init__(self, arr, x0, y0, full_h, full_w):
        self._a = arr
        self._x0 = int(x0)
        self._y0 = int(y0)
        self.shape = (int(full_h), int(full_w))
        self.dtype = arr.dtype

    @property
    def local(self):
        """The underlying window array."""
        return self._a

    @property
    def origin(self):
        return (self._x0, self._y0)

    def take_yx(self, ys, xs):
        """Gather the outer product of row indices ``ys`` and column ``xs``."""
        ly = [int(y) - self._y0 for y in ys]
        lx = [int(x) - self._x0 for x in xs]
        return self._a[np.ix_(ly, lx)]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            ky, kx = key
            if isinstance(ky, slice) or isinstance(kx, slice):
                ys = slice((ky.start or 0) - self._y0,
                           None if ky.stop is None else ky.stop - self._y0,
                           ky.step) if isinstance(ky, slice) else ky - self._y0
                xs = slice((kx.start or 0) - self._x0,
                           None if kx.stop is None else kx.stop - self._x0,
                           kx.step) if isinstance(kx, slice) else kx - self._x0
                return self._a[ys, xs]
            return self._a[int(ky) - self._y0, int(kx) - self._x0]
        raise TypeError("OffsetBand supports [y, x] indexing only, got %r" % (key,))

    def __array__(self, dtype=None):
        # Guard: converting to an array would silently lose the offset, so make
        # that a loud failure rather than a wrong-coordinates bug.
        raise TypeError(
            "OffsetBand cannot be converted to an array; it holds a window at "
            "origin %r of a %r image. Use .local for the window itself."
            % (self.origin, self.shape))


class LazyChannels:
    """Behaves like ``list[np.ndarray]`` of H x W bands, but reads on demand.

    ``ProjectTab.process_polygon`` does ``[ch[y0:y1, x0:x1] for ch in chans]``.
    That still works here (via ``__getitem__`` returning a real band), but the
    caller should prefer :meth:`read_window`, which fetches every band's tiles
    for the bbox in one threaded pass instead of one pass per band.

    ``order`` maps output channel index -> file band index, so the existing
    "first three channels are RGB" export convention can be expressed without
    reordering pixels after the fact.
    """

    def __init__(self, reader, order=None, level=0, origin=(0, 0), size=None):
        self._reader = reader
        self._level = level
        h, w, c = reader.level_shape(level)
        # `origin`/`size` express a crop in full-resolution pixels. Applying it
        # here rather than slicing afterwards means a cropped raster only ever
        # reads the tiles inside the crop.
        self._ox = max(0, int(origin[0]))
        self._oy = max(0, int(origin[1]))
        if size:
            self.width = max(1, min(int(size[0]), w - self._ox))
            self.height = max(1, min(int(size[1]), h - self._oy))
        else:
            self.width, self.height = w - self._ox, h - self._oy
        self._order = list(order) if order is not None else list(range(c))
        self.shape = (self.height, self.width, len(self._order))

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        for i in range(len(self._order)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._order)))]
        band = self._order[index]
        return self._reader.read_window(
            self._ox, self._oy, self._ox + self.width, self._oy + self.height,
            bands=[band], level=self._level)[:, :, 0]

    def read_window(self, x0, y0, x1, y1, max_materialize=_DEFAULT_CACHE_BYTES):
        """All channels for one bbox, as a list of 2D arrays.

        Normal polygons are small, so every band is fetched in a single
        threaded pass and returned as real arrays. A whole-frame window on a
        284-band cube would be 3.88 GB, so past ``max_materialize`` this returns
        a lazy list that decodes one band at a time instead -- consumers that
        loop band-by-band (scene stats, FullFrame polygons) then stay bounded.
        """
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self.width, x1); y1 = min(self.height, y1)
        h = max(0, y1 - y0)
        w = max(0, x1 - x0)
        # Shift into full-resolution coordinates for the reader.
        fx0, fy0 = x0 + self._ox, y0 + self._oy
        fx1, fy1 = x1 + self._ox, y1 + self._oy
        nbytes = h * w * len(self._order) * np.dtype(self._reader.profile.dtype).itemsize
        if nbytes <= max_materialize:
            cube = self._reader.read_window(
                fx0, fy0, fx1, fy1, bands=self._order, level=self._level)
            return [cube[:, :, i] for i in range(cube.shape[2])]
        return _LazyWindowList(self._reader, self._order, fx0, fy0, fx1, fy1, self._level)

    def read_window_offset(self, x0, y0, x1, y1):
        """Like :meth:`read_window`, but each band keeps full-image indexing.

        Returns a list of :class:`OffsetBand`, so existing ``ch[yi, xi]`` code
        works untouched while only the window is actually read.
        """
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.width, int(x1)); y1 = min(self.height, int(y1))
        cube = self._reader.read_window(
            x0 + self._ox, y0 + self._oy, x1 + self._ox, y1 + self._oy,
            bands=self._order, level=self._level)
        # Indices stay in this LazyChannels' (possibly cropped) frame.
        return [OffsetBand(cube[:, :, i], x0, y0, self.height, self.width)
                for i in range(cube.shape[2])]

    def iter_bands(self, x0=None, y0=None, x1=None, y1=None):
        """Bounded-memory band-major iteration over a window (default: full frame)."""
        x0 = 0 if x0 is None else x0
        y0 = 0 if y0 is None else y0
        x1 = self.width if x1 is None else x1
        y1 = self.height if y1 is None else y1
        return self._reader.iter_band_windows(
            self._order, x0 + self._ox, y0 + self._oy,
            x1 + self._ox, y1 + self._oy, level=self._level)


# --------------------------------------------------------------------------
# Headless self-test
# --------------------------------------------------------------------------

def selftest_against_full(path, verbose=True):
    """Assert windowed reads match a full ``tifffile.imread`` of the same file.

    Exercises an interior window, a partial edge tile (bottom-right corner) and
    the last band. Only usable on files small enough to load whole -- use the
    small ``_WV_AOT.tif`` / ``_MASK.tif`` variants for routine checks.
    """
    profile = probe(path)
    if profile is None:
        raise AssertionError("probe() returned None for %s" % path)
    if verbose:
        print(profile.describe())
    if not profile.is_windowable:
        print("  strategy=full -> nothing to verify")
        return True

    reader = open_reader(path, profile)
    if reader is None:
        raise AssertionError("open_reader() returned None for %s" % path)

    full = np.asarray(tifffile.imread(path))
    if full.ndim == 2:
        full = full[..., None]
    elif (profile.axes or "").startswith(("S", "C", "I", "Z")):
        full = np.moveaxis(full, 0, -1)

    H, W, C = profile.height, profile.width, profile.count
    checks = [
        ("interior", (W // 4, H // 4, W // 4 + 40, H // 4 + 40)),
        ("edge-corner", (max(0, W - 37), max(0, H - 29), W, H)),
        ("full-width row band", (0, H // 2, W, min(H, H // 2 + 3))),
    ]
    band_sets = [None, [C - 1], [0]] if C > 1 else [None]

    for label, (x0, y0, x1, y1) in checks:
        for bands in band_sets:
            got = reader.read_window(x0, y0, x1, y1, bands=bands)
            idx = range(C) if bands is None else bands
            want = full[y0:y1, x0:x1][:, :, list(idx)]
            if got.shape != want.shape:
                raise AssertionError(
                    "%s bands=%s shape %s != %s" % (label, bands, got.shape, want.shape))
            if not np.array_equal(np.nan_to_num(got, nan=-12345.0),
                                  np.nan_to_num(want, nan=-12345.0)):
                bad = int(np.sum(np.nan_to_num(got, nan=-12345.0) !=
                                 np.nan_to_num(want, nan=-12345.0)))
                raise AssertionError(
                    "%s bands=%s mismatch in %d of %d values"
                    % (label, bands, bad, want.size))
            if verbose:
                print("  ok  %-20s bands=%-8s %s" % (label, bands, got.shape))
    print("PASS %s" % os.path.basename(path))
    return True
