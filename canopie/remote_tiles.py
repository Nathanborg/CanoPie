"""
Remote imagery from an ArcGIS tiled map/image service, read like a COG.

A tiled ArcGIS service (``.../MapServer`` or ``.../ImageServer`` with
``tileInfo``; what an ArcGIS Hub / open-data page such as the STRI data portal
publishes) is not a GeoTIFF: it is the image pre-cut into 256 x 256 JPEG/PNG
tiles at every zoom level of a Web Mercator pyramid, fetched as
``.../tile/{level}/{row}/{col}``. That maps one to one onto CanoPie's tiled
reader -- zoom levels are pyramid levels, service tiles are tiles -- so
:class:`TileServiceReader` subclasses :class:`raster_reader.TiledRasterReader`
and replaces only the I/O. Everything above it (previews, zoom tiles, the
streaming percentage, Inspect, CSV and ML through LazyAnalysis, batched
prefetch) is unchanged.

The image CanoPie sees: level 0 is the finest zoom that has data, cropped to
the layer's extent, 4 bands (R, G, B, alpha; alpha 0 where the service has no
tile). Georeferenced in the service's CRS (Web Mercator, EPSG:3857, for ArcGIS
Online). It is the rendered picture -- 8-bit, usually JPEG -- not the source
sensor values.

Pointer (a ``.cogref``)::

    {"canopie_cogref": 1, "kind": "arcgis-tiles",
     "url": "https://tiles.arcgis.com/.../MapServer", "name": "...", "zmax": 21}
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict

import numpy as np

from . import remote_raster as rm

__all__ = ["looks_like_tile_service", "is_tile_pointer", "resolve_service_url",
           "TileService", "TileServiceReader", "probe", "make_pointer", "georef"]

KIND = "arcgis-tiles"
#: ArcGIS Online item metadata; a module constant so the QC suite can use a
#: local stand-in.
ARCGIS_ITEM_URL = "https://www.arcgis.com/sharing/rest/content/items/{}?f=json"
#: Compressed tiles held ahead of use (prefetch), per reader.
RAW_CACHE_BYTES = 256 * 1024 * 1024
#: Smallest pyramid level kept: the coarsest whose longer side is still >= this.
MIN_LEVEL_DIM = 384
#: Before any tile has been seen, its compressed size is assumed to be this.
DEFAULT_TILE_BYTES = 24 * 1024

_SERVICE_RE = re.compile(r"(?i)^(https?://.+?/rest/services/.+?/(?:MapServer|ImageServer))")
_ITEM_RE = re.compile(r"(?i)\b([0-9a-f]{32})\b")


# --------------------------------------------------------------------------
# Recognising and resolving links
# --------------------------------------------------------------------------

def looks_like_tile_service(url):
    """True for an ArcGIS REST MapServer/ImageServer URL, or an ArcGIS Online /
    Hub / open-data page (an item id in an arcgis.com address)."""
    url = str(url).strip()
    if _SERVICE_RE.match(url):
        return True
    host = rm.urlsplit(url).netloc.lower()
    return bool(_ITEM_RE.search(url)) and ("arcgis.com" in host or "arcgis" in host)


def _get_json(url, what):
    try:
        resp = rm._session().get(url, timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT))
    except Exception as exc:
        raise rm.RemoteError("{}: request failed: {}: {}".format(what, type(exc).__name__, rm.redact(exc))) from None
    if resp.status_code != 200:
        raise rm.RemoteError("{}: HTTP {}".format(what, resp.status_code))
    try:
        data = resp.json()
    except ValueError:
        raise rm.RemoteError("{}: the answer is not JSON".format(what)) from None
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise rm.RemoteError("{}: {}".format(what, rm.redact(err.get("message") or err)))
    return data


def resolve_service_url(url):
    """``(service_url, title)`` for a service URL or an ArcGIS item / Hub page."""
    url = str(url).strip()
    m = _SERVICE_RE.match(url)
    if m:
        return m.group(1), None
    im = _ITEM_RE.search(url)
    if not im:
        raise rm.RemoteError("not an ArcGIS service or item link: {}".format(rm.redact(url)))
    item = _get_json(ARCGIS_ITEM_URL.format(im.group(1)), "ArcGIS item " + im.group(1))
    svc = item.get("url") or ""
    m = _SERVICE_RE.match(svc)
    if not m:
        raise rm.RemoteError(
            "ArcGIS item '{}' is a {} without a tiled map or image service CanoPie can "
            "stream.".format(item.get("title") or im.group(1), item.get("type") or "item"))
    return m.group(1), item.get("title")


def _safe_name(title):
    return re.sub(r"[^\w.-]+", "_", str(title or "")).strip("_") or "arcgis_tiles"


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------

class TileService:
    """Tiling scheme and extent of one tiled ArcGIS service."""

    def __init__(self, url, zmax=None):
        self.url = url.rstrip("/")
        meta = _get_json(self.url + "?f=json", self.url.rsplit("/rest/services/", 1)[-1])
        ti = meta.get("tileInfo") or {}
        if not ti.get("lods"):
            raise rm.RemoteError("{}: not a tiled (cached) service -- it has no tile "
                                 "pyramid to stream".format(self.url))
        self.name = meta.get("documentInfo", {}).get("Title") or meta.get("name") \
            or self.url.rstrip("/").split("/")[-2]
        self.copyright = meta.get("copyrightText") or ""
        self.tile = int(ti.get("cols") or 256)
        if int(ti.get("rows") or self.tile) != self.tile:
            raise rm.RemoteError("{}: non-square tiles are not supported".format(self.url))
        self.ox = float(ti["origin"]["x"])
        self.oy = float(ti["origin"]["y"])
        self.res = {int(l["level"]): float(l["resolution"]) for l in ti["lods"]}
        sr = ti.get("spatialReference") or meta.get("spatialReference") or {}
        epsg = sr.get("latestWkid") or sr.get("wkid")
        self.epsg = 3857 if epsg in (102100, 102113, 900913) else (int(epsg) if epsg else None)
        ext = meta.get("fullExtent") or meta.get("initialExtent") or {}
        self.extent = (float(ext["xmin"]), float(ext["ymin"]), float(ext["xmax"]), float(ext["ymax"]))
        self.zmax = int(zmax) if zmax is not None else self._deepest_level_with_data()

    def tile_url(self, z, row, col):
        return "{}/tile/{}/{}/{}".format(self.url, int(z), int(row), int(col))

    def global_px(self, z, x, y):
        """Map coordinates -> global pixel at zoom ``z`` (float)."""
        r = self.res[z]
        return (x - self.ox) / r, (self.oy - y) / r

    def _has_tile(self, z, row, col):
        try:
            resp = rm._session().get(self.tile_url(z, row, col), timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT))
            ok = resp.status_code == 200 and resp.content[:4] in (
                b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xdb", b"\x89PNG") or (
                resp.status_code == 200 and resp.content[:2] == b"\xff\xd8")
            return ok
        except Exception:
            return False

    def _deepest_level_with_data(self):
        """Services list zoom levels past their data (the STRI orthophoto
        lists 24, has tiles to 21): the finest level with a tile at the
        extent's centre or quarter points."""
        x0, y0, x1, y1 = self.extent
        pts = [((x0 + x1) / 2, (y0 + y1) / 2), ((3 * x0 + x1) / 4, (3 * y0 + y1) / 4),
               ((x0 + 3 * x1) / 4, (y0 + 3 * y1) / 4)]
        for z in sorted(self.res, reverse=True):
            for (x, y) in pts:
                gx, gy = self.global_px(z, x, y)
                if self._has_tile(z, int(gy // self.tile), int(gx // self.tile)):
                    return z
        raise rm.RemoteError("{}: no tile answered at any level".format(self.url))

    def geometry(self):
        """Per level k (zoom zmax - k): (zoom, origin_x, origin_y, width, height)
        in that zoom's global pixels; level 0 is the finest zoom with data."""
        x0, y0, x1, y1 = self.extent
        gx0, gy0 = self.global_px(self.zmax, x0, y1)
        gx1, gy1 = self.global_px(self.zmax, x1, y0)
        ox0, oy0 = int(math.floor(gx0)), int(math.floor(gy0))
        w0, h0 = int(math.ceil(gx1)) - ox0, int(math.ceil(gy1)) - oy0
        levels = []
        z = self.zmax
        while z in self.res:
            k = self.zmax - z
            f = 2 ** k
            ox, oy = ox0 // f, oy0 // f
            w = max(1, int(math.ceil((ox0 + w0) / f)) - ox)
            h = max(1, int(math.ceil((oy0 + h0) / f)) - oy)
            levels.append((z, ox, oy, w, h))
            if max(w, h) <= MIN_LEVEL_DIM:
                break
            z -= 1
        return levels


# --------------------------------------------------------------------------
# Pointers
# --------------------------------------------------------------------------

_KIND_CACHE = {}


def is_tile_pointer(path):
    """True when ``path`` is a .cogref pointing at a tile service."""
    if not rm.is_cogref(path):
        return False
    ap = os.path.abspath(os.fspath(path))
    try:
        key = (os.path.normcase(ap), os.stat(ap).st_mtime_ns)
    except OSError:
        return False
    hit = _KIND_CACHE.get(key)
    if hit is None:
        try:
            hit = rm.read_pointer(ap).get("kind") == KIND
        except rm.RemoteError:
            hit = False
        _KIND_CACHE[key] = hit
    return hit


_SERVICES = OrderedDict()
_SERVICES_LOCK = threading.Lock()


def _service_for(path):
    ap = os.path.abspath(os.fspath(path))
    key = (os.path.normcase(ap), os.stat(ap).st_mtime_ns)
    with _SERVICES_LOCK:
        svc = _SERVICES.get(key)
    if svc is None:
        ptr = rm.read_pointer(ap)
        svc = TileService(ptr["url"], zmax=ptr.get("zmax"))
        with _SERVICES_LOCK:
            svc = _SERVICES.setdefault(key, svc)
            while len(_SERVICES) > 16:
                _SERVICES.popitem(last=False)
    return svc


def _profile_from(path, svc):
    from .raster_reader import RasterProfile, STRATEGY_TILES_BIP
    geo = svc.geometry()
    z, ox, oy, w, h = geo[0]
    r = svc.res[z]
    return RasterProfile(
        path=os.path.abspath(os.fspath(path)), width=w, height=h, count=4,
        dtype=np.dtype(np.uint8), axes="YXS", tiled=True, tile_w=svc.tile, tile_h=svc.tile,
        planar=1, compression=7, predictor=1, n_levels=len(geo),
        level_shapes=[(lh, lw, 4) for (_z, _x, _y, lw, lh) in geo],
        band_names=["red", "green", "blue", "alpha"], wavelengths=[None] * 4, nodata=None,
        epsg=svc.epsg, transform=[svc.ox + ox * r, r, 0.0, svc.oy - oy * r, 0.0, -r],
        strategy=STRATEGY_TILES_BIP, channel_origin="tifffile", remote=True,
        source=svc.url, service=svc)


def probe(path):
    """RasterProfile of a tile-service pointer (raster_reader.probe defers here)."""
    return _profile_from(path, _service_for(path))


def georef(path):
    """``(transform, epsg, (height, width))`` of a tile-service pointer."""
    p = probe(path)
    return list(p.transform), p.epsg, (int(p.height), int(p.width))


def make_pointer(url, folder):
    """Check a service / item link and write ``<name>.cogref`` in ``folder``
    (the remote_raster.make_pointer counterpart; same info dict)."""
    svc_url, title = resolve_service_url(url)
    svc = TileService(svc_url)
    name = _safe_name(title or svc.name)
    os.makedirs(folder, exist_ok=True)
    n, stem = 1, name
    while True:
        path = os.path.join(folder, stem + ("" if n == 1 else "_{}".format(n)) + rm.COGREF_EXT)
        if not os.path.exists(path):
            break
        try:
            if rm.read_pointer(path).get("url") == svc_url:
                break
        except rm.RemoteError:
            pass
        n += 1
    rm.write_pointer(path, svc_url, kind=KIND, name=name, zmax=svc.zmax,
                     **({"credit": svc.copyright} if svc.copyright else {}))
    geo = svc.geometry()
    z, _ox, _oy, w, h = geo[0]
    return {"url": svc_url, "name": name, "size": None, "width": w, "height": h, "bands": 4,
            "dtype": "uint8", "tiled": True, "levels": len(geo), "compression": 7,
            "provider": KIND, "warning": None, "path": path,
            "pixel_m": svc.res[z], "zoom": z, "credit": svc.copyright,
            "requests": 0, "bytes_fetched": 0}


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------

class _TileStats:
    """What raster_reader expects in ``reader._remote`` for statistics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_stats()

    def add(self, nbytes, seconds):
        with self._lock:
            self._s["requests"] += 1
            self._s["bytes_fetched"] += int(nbytes)
            self._s["http_seconds"] += float(seconds)

    def stats(self):
        with self._lock:
            return dict(self._s)

    def reset_stats(self):
        with getattr(self, "_lock", threading.Lock()):
            self._s = {"requests": 0, "bytes_fetched": 0, "cache_hits": 0, "http_seconds": 0.0,
                       "size": None, "cache_bytes": 0}

    def close(self):
        pass


def _decode_tile(data, tile):
    """RGBA uint8 (tile, tile, 4) from a JPEG or PNG tile; None if undecodable."""
    import imagecodecs
    try:
        if data[:4] == b"\x89PNG":
            a = imagecodecs.png_decode(data)
        else:
            a = imagecodecs.jpeg8_decode(data)
    except Exception:
        try:
            a = imagecodecs.imread(data)
        except Exception:
            return None
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2)
    elif a.shape[2] > 4:
        a = a[..., :4]
    a = a.astype(np.uint8, copy=False)
    if a.shape[:2] != (tile, tile):
        out = np.zeros((tile, tile, 4), np.uint8)
        out[:a.shape[0], :a.shape[1]] = a[:tile, :tile]
        a = out
    return a


from .raster_reader import TiledRasterReader, _TILE_CACHE  # noqa: E402


class TileServiceReader(TiledRasterReader):
    """A tiled ArcGIS service as a TiledRasterReader: same levels, same
    read_window / read_bands_full / prefetch API, tiles fetched by URL."""

    def __init__(self, profile):                 # noqa: D401 -- no TIFF to parse
        self.profile = profile
        self.path = profile.path
        self._svc = profile.service if getattr(profile, "service", None) is not None \
            else _service_for(profile.path)
        self._remote = _TileStats()
        self._label = os.path.basename(self.path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._tpool = None
        self._handles = []
        self._memmap = None
        self._raw = OrderedDict()
        self._raw_bytes = 0
        self._raw_lock = threading.Lock()
        self._avg = float(DEFAULT_TILE_BYTES)
        T = self._svc.tile
        self._geo = self._svc.geometry()
        self._levels = []
        for (z, ox, oy, w, h) in self._geo:
            tx0, ty0 = ox // T, oy // T
            across = (ox + w - 1) // T - tx0 + 1
            down = (oy + h - 1) // T - ty0 + 1
            self._levels.append({
                "page": None, "width": w, "height": h, "count": 4, "planar": 1,
                "tw": T, "th": T, "across": across, "down": down, "per_plane": across * down,
                "offsets": None, "counts": [DEFAULT_TILE_BYTES] * (across * down),
                "tile_bytes": T * T * 4, "workers": 8, "decode_kwargs": {},
                "z": z, "ox": ox, "oy": oy, "tx0": tx0, "ty0": ty0})

    # -- tiles --------------------------------------------------------------

    def _tile_keys(self, level, x0, y0, x1, y1):
        lv = self._levels[level]
        T = lv["tw"]
        gx0, gy0 = lv["ox"] + x0, lv["oy"] + y0
        gx1, gy1 = lv["ox"] + x1, lv["oy"] + y1
        return [(ty, tx) for ty in range(gy0 // T, (gy1 - 1) // T + 1)
                for tx in range(gx0 // T, (gx1 - 1) // T + 1)]

    def segments_for_rects(self, rects, level=0, bands=None):
        lv = self._levels[level]
        out = []
        for (x0, y0, x1, y1) in rects:
            x0 = max(0, int(x0)); y0 = max(0, int(y0))
            x1 = min(lv["width"], int(math.ceil(x1))); y1 = min(lv["height"], int(math.ceil(y1)))
            if x1 > x0 and y1 > y0:
                out.extend(self._tile_keys(level, x0, y0, x1, y1))
        return list(dict.fromkeys(out))

    def _have(self, level, key):
        if _TILE_CACHE.get((self.path, level, key)) is not None:
            return True
        with self._raw_lock:
            return (level, key) in self._raw

    def _download(self, level, keys):
        """Fetch tiles ``keys`` ((row, col) at this level's zoom) in parallel.
        Returns {key: bytes or None (no tile there)}."""
        z = self._levels[level]["z"]
        keys = list(keys)
        if not keys:
            return {}
        progress = rm._Progress(self._label, int(self._avg * len(keys)))
        t0 = time.perf_counter()

        def one(key):
            ts = time.perf_counter()
            try:
                resp = rm._session().get(self._svc.tile_url(z, key[0], key[1]),
                                         timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT))
                data = resp.content if resp.status_code == 200 else None
            except Exception as exc:
                logging.debug("[tiles] %s %s: %s", self._label, key, exc)
                data = None
            self._remote.add(len(data or b""), time.perf_counter() - ts)
            progress.add(len(data or b"") or int(self._avg))
            return key, data

        if len(keys) == 1:
            got = [one(keys[0])]
        else:
            got = list(rm._io_pool().map(one, keys))
        progress.finish()
        sizes = [len(d) for _k, d in got if d]
        if sizes:
            self._avg = 0.8 * self._avg + 0.2 * (sum(sizes) / len(sizes))
        logging.info("[remote] %s: streamed %.1f MB in %d tile(s), %.1f s", self._label,
                     sum(sizes) / 1e6, len(keys), time.perf_counter() - t0)
        return dict(got)

    def _tiles(self, level, keys):
        """Decoded (T, T, 4) arrays for ``keys``: cache, prefetched bytes,
        then one parallel download for the rest."""
        T = self._levels[level]["tw"]
        out, todo = {}, []
        for k in dict.fromkeys(keys):
            hit = _TILE_CACHE.get((self.path, level, k))
            if hit is not None:
                out[k] = hit
                continue
            with self._raw_lock:
                raw = self._raw.pop((level, k), None)
                if raw is not None:
                    self._raw_bytes -= len(raw)
            if raw is not None:
                out[k] = raw
            else:
                todo.append(k)
        fetched = self._download(level, todo)
        for k in list(out) + todo:
            v = out.get(k) if k not in fetched else fetched[k]
            if isinstance(v, np.ndarray):
                continue
            arr = _decode_tile(v, T) if v else None
            if arr is None:
                arr = np.zeros((T, T, 4), np.uint8)          # no tile: alpha 0
            out[k] = arr
            _TILE_CACHE.put((self.path, level, k), arr)
        return out

    # -- prefetch (the TiledRasterReader API) --------------------------------

    def level_fetch_bytes(self, level, n_bands):
        lv = self._levels[level]
        return int(lv["across"] * lv["down"] * self._avg)

    def fetch_bytes_for_rects(self, rects, level=0, bands=None):
        return int(sum(self._avg for k in self.segments_for_rects(rects, level)
                       if not self._have(level, k)))

    def prefetch_rects(self, rects, level=0, bands=None):
        todo = [k for k in self.segments_for_rects(rects, level) if not self._have(level, k)]
        got = self._download(level, todo)
        n = 0
        with self._raw_lock:
            for k, data in got.items():
                self._raw[(level, k)] = data or b""
                self._raw_bytes += len(data or b"")
                n += len(data or b"")
            while self._raw_bytes > RAW_CACHE_BYTES and len(self._raw) > 1:
                _k, old = self._raw.popitem(last=False)
                self._raw_bytes -= len(old)
        return n

    # -- reads ------------------------------------------------------------

    def read_window(self, x0, y0, x1, y1, bands=None, level=0, step=1):
        lv = self._levels[level]
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(lv["width"], int(x1)); y1 = min(lv["height"], int(y1))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 0), np.uint8)
        bands = list(range(4)) if bands is None else [int(b) for b in bands]
        step = max(1, int(step or 1))
        T = lv["tw"]
        rows = np.arange(y0, y1, step) + lv["oy"]
        cols = np.arange(x0, x1, step) + lv["ox"]
        out = np.empty((len(rows), len(cols), len(bands)), np.uint8)
        trows, tcols = rows // T, cols // T
        keys = [(int(ty), int(tx)) for ty in np.unique(trows) for tx in np.unique(tcols)]
        tiles = self._tiles(level, keys)
        for ty in np.unique(trows):
            ri = np.nonzero(trows == ty)[0]
            ry = rows[ri] - ty * T
            for tx in np.unique(tcols):
                ci = np.nonzero(tcols == tx)[0]
                cx = cols[ci] - tx * T
                t = tiles[(int(ty), int(tx))]
                if step == 1:
                    patch = t[int(ry[0]):int(ry[-1]) + 1, int(cx[0]):int(cx[-1]) + 1]
                else:
                    patch = t[np.ix_(ry, cx)]
                out[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1] = patch[..., bands]
        return out

    def read_bands_full(self, bands, level=0, step=1):
        lv = self._levels[level]
        return self.read_window(0, 0, lv["width"], lv["height"], bands=bands, level=level, step=step)

    def close(self):
        with self._raw_lock:
            self._raw.clear()
            self._raw_bytes = 0
        _TILE_CACHE.drop_path(self.path)
