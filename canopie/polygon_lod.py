"""
Polygon pyramids -- the COG overview idea applied to vector geometry.

WHY THIS EXISTS
---------------
Measured on a real project (2333 tree crowns traced at full raster
resolution):

    1,010,806 vertices   mean 433/polygon, max 2312
    90.9 MB of polygon JSON
    ~5 s just to parse it at project open

Almost none of that detail is visible. At an overview zoom a 433-vertex crown
covers a handful of screen pixels, so the painter is handed roughly 400
vertices it cannot draw distinguishably. A COG solves the identical problem for
rasters by storing decimated overviews alongside the full-resolution data and
reading only the level that matches the current zoom. This module does that for
polygons.

Measured pyramid on that same project:

    level  tol(px)      vertices   % of full   size    parse
        3        8       241,642      23.9%   5.25 MB  0.100 s
        4       16       137,565      13.6%   2.99 MB  0.057 s
        5       32        74,429       7.4%   1.62 MB  0.026 s

So a single overview file holding a coarse level for EVERY polygon parses in
57 ms instead of ~5 s -- and that is enough to draw the whole-mosaic view.

THE INVARIANT
-------------
Levels are DISPLAY-ONLY. Statistics, CSV export, ML extraction, shapefile
export and hit-testing must always use the full coordinates. `LazyPolygonRecord`
enforces this by construction: it holds only the coarse level until something
asks for 'points', at which moment it loads the real file. Nothing that needs
exact geometry can accidentally get a decimated answer.
"""
import copy
import json
import logging
import os
import threading
from collections import OrderedDict

import numpy as np

#: Consolidated coarse-geometry file, written next to the per-polygon files.
OVERVIEW_NAME = "_overview.json"

#: Level written into OVERVIEW_NAME. 16 px is the sweet spot from the table
#: above: 14% of the vertices, 3 MB, and still smooth at whole-mosaic zoom.
OVERVIEW_LEVEL = 4

#: Levels are tolerance = 2**level raster pixels, exactly like COG overviews
#: halve the resolution each step.
MAX_LEVEL = 8

#: Below this a polygon is already cheap; a pyramid would cost more than it saves.
MIN_POINTS_FOR_LOD = 64

#: Never decimate a ring below this -- fewer points stops being a polygon.
MIN_RING_POINTS = 4

#: Coordinates are rounded to this many decimals when stored. On a 0.07 m/px
#: raster, 2 decimals is sub-millimetre -- far below anything the geometry
#: means -- while `25369.88464643981` (the current full repr) costs 3x the
#: bytes for noise.
COORD_DECIMALS = 2


def decimate(points, tol):
    """Drop vertices that land within `tol` of the previous kept vertex.

    Grid-quantisation rather than Douglas-Peucker: fully vectorised, O(n), and
    visually equivalent at the scale it is used (a vertex moved by <1 device
    pixel cannot be seen). Douglas-Peucker gives slightly better shapes for the
    same vertex count but is far slower, and this runs over a million vertices.

    Always keeps the first and last vertex so the ring stays closed.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 8 or tol <= 0:
        return pts
    q = np.floor(pts / float(tol))
    keep = np.empty(len(q), dtype=bool)
    keep[0] = True
    keep[1:] = (q[1:] != q[:-1]).any(axis=1)
    keep[-1] = True
    out = pts[keep]
    return out if len(out) >= MIN_RING_POINTS else pts[:MIN_RING_POINTS]


def build_pyramid(points, max_level=MAX_LEVEL):
    """Return ``{level: [[x, y], ...]}`` for a polygon's vertex list.

    Level L uses a tolerance of ``2**L`` raster pixels. Levels are emitted only
    while they actually shrink the ring; once decimation stops helping (or the
    ring hits MIN_RING_POINTS) further levels would be identical, so they are
    dropped rather than stored redundantly.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < MIN_POINTS_FOR_LOD:
        return {}

    pyramid = {}
    prev_n = len(pts)
    for level in range(max_level + 1):
        red = decimate(pts, 2.0 ** level)
        n = len(red)
        if n >= prev_n:
            continue                     # this level buys nothing
        pyramid[str(level)] = np.round(red, COORD_DECIMALS).tolist()
        prev_n = n
        if n <= MIN_RING_POINTS:
            break                        # cannot get coarser
    return pyramid


def level_for_scale(scale, available_levels):
    """Pick the coarsest stored level whose tolerance is still under a device
    pixel at `scale`, or None to use the full geometry.

    `scale` is scene-units -> device-pixels. A level's tolerance of ``2**L``
    scene units spans ``2**L * scale`` device pixels; anything at or below 1 is
    invisible, so the best level is the largest L with ``2**L * scale <= 1``.
    """
    if not available_levels or scale <= 0:
        return None
    best = None
    for lv in available_levels:
        try:
            l = int(lv)
        except (TypeError, ValueError):
            continue
        if (2.0 ** l) * scale <= 1.0:
            if best is None or l > best:
                best = l
    return best


# ---------------------------------------------------------------------------
# Per-polygon sidecar files
# ---------------------------------------------------------------------------
def dump_sidecar(path, data, remember=False):
    """Write one polygon sidecar (`<group>_<image>_polygons.json`).

    remember=True keeps the written text in the write-through cache (see
    _remember_text) -- for bulk writers whose files this process reads back.

    Every sidecar writer goes through here. They used
    ``json.dump(data, f, indent=...)``, and that combination can never use
    CPython's C encoder: ``json.dump`` streams through ``iterencode`` (the
    pure-Python encoder), and any ``indent`` disables the C one besides.
    Importing 2,801 BCI crowns (14 M coordinate floats) spent 88 of 93 GUI
    seconds in ``encoder.py``. ``json.dumps`` with compact separators takes
    the C encoder in one shot -- same JSON, read back identically by
    ``json.load`` -- at a fraction of the time and about a third of the bytes.

    Written to a temporary file and moved over the target, so a crash or a
    full disk mid-write can no longer leave a truncated sidecar (the old
    writers opened the real file with "w" and streamed into it).
    """
    text = json.dumps(data, separators=(',', ':'))
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            if remember:
                _remember_text(path, text)
            return
        except PermissionError:
            # Windows: the target is briefly held open (indexer, antivirus,
            # a reader thread). Back off a little and retry.
            if attempt == 3:
                break
            import time as _time
            _time.sleep(0.05 * (attempt + 1))
    # Last resort: write in place rather than lose the save.
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    try:
        os.remove(tmp)
    except OSError:
        pass


def read_sidecar(path):
    """Parse one sidecar; None (logged) when unreadable. Safe off the GUI thread:
    it touches nothing but the file (and the write-through cache below)."""
    hit = _recall_written(path)
    if hit is not None:
        return hit
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error("[polygon_lod] could not load full geometry from %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Write-through cache of sidecars this session wrote in bulk
# ---------------------------------------------------------------------------
# Windows' real-time scanner reads every freshly written file on its first
# open: 500 just-rewritten BCI sidecars took 9.7 s to read back, 0.1 s twenty
# seconds later. After re-projecting 2,802 crowns for a new crop, the next
# Reset (its backup zip included), CSV export or zoom refine paid that again --
# 90 s for one Reset -- to read back what this process itself had just written.
# So bulk writers keep the exact text they wrote, keyed on the file's
# (mtime, size) right after the write; any other change to the file makes the
# entry miss and the file is read as usual.
_WRITTEN_BUDGET_BYTES = 256 * 1024 * 1024
_WRITTEN = OrderedDict()
_WRITTEN_LOCK = threading.Lock()
_WRITTEN_TOTAL = [0]          # bytes of text held, kept as a running sum


def _written_key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _remember_text(path, text):
    try:
        st = os.stat(path)
    except OSError:
        return
    key = _written_key(path)
    with _WRITTEN_LOCK:
        old = _WRITTEN.pop(key, None)
        if old is not None:
            _WRITTEN_TOTAL[0] -= len(old[2])
        _WRITTEN[key] = (st.st_mtime_ns, st.st_size, text)
        _WRITTEN_TOTAL[0] += len(text)
        while _WRITTEN_TOTAL[0] > _WRITTEN_BUDGET_BYTES and len(_WRITTEN) > 1:
            _, old = _WRITTEN.popitem(last=False)
            _WRITTEN_TOTAL[0] -= len(old[2])


def cached_text(path):
    """The text this process last wrote to `path` with remember=True, if the
    file is still exactly that (same mtime and size); else None."""
    key = _written_key(path)
    with _WRITTEN_LOCK:
        hit = _WRITTEN.get(key)
    if hit is None:
        return None
    mtime_ns, size, text = hit
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_mtime_ns != mtime_ns or st.st_size != size:
        with _WRITTEN_LOCK:
            old = _WRITTEN.pop(key, None)
            if old is not None:
                _WRITTEN_TOTAL[0] -= len(old[2])
        return None
    return text


def _recall_written(path):
    text = cached_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def forget_written():
    with _WRITTEN_LOCK:
        _WRITTEN.clear()
        _WRITTEN_TOTAL[0] = 0


# ---------------------------------------------------------------------------
# The consolidated overview file
# ---------------------------------------------------------------------------
def overview_path(polygons_dir):
    return os.path.join(polygons_dir, OVERVIEW_NAME)


def write_overview(polygons_dir, records, level=OVERVIEW_LEVEL):
    """Write the coarse-geometry index every project open reads.

    `records` is an iterable of ``(group, filepath, polygon_data)``. Only the
    handful of fields the VIEWER needs are stored -- never the full
    coordinates, never `properties`, which is what keeps this file ~3 MB where
    the polygon dir is 91 MB.
    """
    entries = []
    for group, filepath, data in records:
        if not isinstance(data, dict):
            continue
        pts = data.get('points') or []
        lod = data.get('lod') or {}
        coarse = lod.get(str(level))
        if coarse is None:
            # Fall back to the coarsest level we do have, else decimate now.
            if lod:
                try:
                    best = max(int(k) for k in lod)
                    coarse = lod[str(best)]
                except ValueError:
                    coarse = None
            if coarse is None and pts:
                coarse = np.round(decimate(pts, 2.0 ** level),
                                  COORD_DECIMALS).tolist()
        entries.append({
            'g': group,
            'f': filepath,
            'n': data.get('name', group),
            't': data.get('type', 'polygon'),
            'r': data.get('root', ''),
            'cs': data.get('coord_space', 'image'),
            'rs': data.get('image_ref_size') or {},
            'p': coarse or [],
            'np': len(pts),
        })

    tmp = overview_path(polygons_dir) + ".tmp"
    # json.dumps (C encoder), not json.dump (always pure Python) -- see dump_sidecar.
    text = json.dumps({'version': 1, 'level': level, 'entries': entries},
                      separators=(',', ':'))
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, overview_path(polygons_dir))
    logging.info("[polygon_lod] overview: %d entries, %.2f MB, level %d",
                 len(entries),
                 os.path.getsize(overview_path(polygons_dir)) / 1e6, level)
    return len(entries)


def read_overview(polygons_dir):
    """Return ``(level, entries)`` or ``(None, None)`` when absent/unusable."""
    path = overview_path(polygons_dir)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            blob = json.load(f)
        if not isinstance(blob, dict) or blob.get('version') != 1:
            return None, None
        return blob.get('level', OVERVIEW_LEVEL), blob.get('entries') or []
    except Exception as e:
        logging.warning("[polygon_lod] unreadable overview %s: %s", path, e)
        return None, None


# ---------------------------------------------------------------------------
# Lazy record
# ---------------------------------------------------------------------------
class LazyPolygonRecord(dict):
    """A polygon whose FULL coordinates are read from disk on first real use.

    Behaves as an ordinary dict carrying the viewer-facing metadata plus
    `display_points` (the coarse level). The moment anything asks for a key
    that implies exact geometry -- 'points' above all -- the backing file is
    read and merged in.

    That indirection is the safety mechanism for the whole feature: statistics,
    CSV export and ML extraction all reach geometry through `['points']`, so
    they cannot be served decimated data by accident. There is no list of call
    sites to keep in sync.
    """

    #: Keys that must never be answered from the coarse overview.
    _EXACT_KEYS = frozenset({'points', 'coordinates', 'properties', 'lod'})

    def __init__(self, meta, source_path):
        super().__init__(meta)
        self._source_path = source_path
        self._materialised = False

    @property
    def is_materialised(self):
        return self._materialised

    @property
    def source_path(self):
        return self._source_path

    def _materialise(self):
        if self._materialised:
            return
        self._materialised = True          # set first: a failed read must not loop
        full = read_sidecar(self._source_path)
        if full is not None:
            self._merge_full(full)

    def materialise_from(self, full):
        """Materialise from an already-parsed sidecar (read off the GUI thread
        by the LOD refiner) -- no IO here. A no-op if anything materialised the
        record in the meantime: the first full read wins, and the record may
        have been edited since."""
        if self._materialised:
            return
        self._materialised = True
        if full is not None:
            self._merge_full(full)

    def _merge_full(self, full):
        if isinstance(full, dict):
            display = dict.get(self, 'display_points')
            dict.update(self, full)
            if display is not None:
                dict.__setitem__(self, 'display_points', display)

    def __getitem__(self, key):
        if key in self._EXACT_KEYS and not self._materialised:
            self._materialise()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self._EXACT_KEYS and not self._materialised:
            self._materialise()
        return dict.get(self, key, default)

    def __contains__(self, key):
        if key in self._EXACT_KEYS and not self._materialised:
            self._materialise()
        return dict.__contains__(self, key)

    # Any wholesale read of the record (iteration, json.dump, copy, **kwargs)
    # must see the real thing, or a save would persist the coarse geometry
    # over the full one.
    def _full(self):
        self._materialise()
        return self

    def keys(self):
        return dict.keys(self._full())

    def values(self):
        return dict.values(self._full())

    def items(self):
        return dict.items(self._full())

    def __iter__(self):
        return dict.__iter__(self._full())

    def copy(self):
        return dict(self._full())


def point_count(rec):
    """Vertex count of a polygon record WITHOUT reading a lazy record's file.

    Listing names (the editor's mask-polygon menu) used `len(rec.get('points'))`,
    which materialises every LazyPolygonRecord: 2,802 sidecar opens, 55 s of the
    60 s it took to open the image editor on the BCI project. The overview
    already stores the count ('np' -> 'full_point_count').
    """
    if isinstance(rec, LazyPolygonRecord) and not rec.is_materialised:
        n = dict.get(rec, 'full_point_count')
        if n is not None:
            try:
                return int(n)
            except (TypeError, ValueError):
                pass
    pts = rec.get('points') if isinstance(rec, dict) else None
    return len(pts) if pts else 0
