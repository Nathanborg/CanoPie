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
import json
import logging
import os

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
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'level': level, 'entries': entries},
                  f, separators=(',', ':'))
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

    def _materialise(self):
        if self._materialised:
            return
        self._materialised = True          # set first: a failed read must not loop
        try:
            with open(self._source_path, 'r', encoding='utf-8') as f:
                full = json.load(f)
        except Exception as e:
            logging.error("[polygon_lod] could not load full geometry from %s: %s",
                          self._source_path, e)
            return
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
