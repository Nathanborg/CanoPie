"""
Where a NON-georeferenced image looks at the ground, for the canopy finder.

Raw drone frames carry their position in EXIF (DJI also writes the camera
heading and the height above take-off in XMP). From that, a nadir camera model
gives each frame an approximate EPSG:4326 transform -- the model of
shapefile_io.estimate_transform_from_exif (GSD = height * sensor width /
(focal * image width), image centre at the GPS point), but on NUMERIC EXIF
(exiftool -n; without it DJI values are DMS / "8.8 mm" strings) and with the
heading turning the image the way DJI means it (clockwise from north: a frame
shot heading east has its top to the east and its right to the south).

Picking frames: the same scipy KDTree as loaders.ImageProcessor.build_kdtree,
built on local metres instead of raw degrees and queried for the k nearest to
the CROWN (not to the source image's GPS), plus the notebook's optional
the frames of its own collection.

Positions come first from what CanoPie already stored (gps_cache.json, the
project's EXIF GPS cache), then from ONE batched exiftool call; full EXIF
(height / focal / heading) is read only for the few frames actually used.
Everything is cached in <current project>/canopy_exif_cache.json (another
project's folder is never written).
"""
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time

import numpy as np

__all__ = ["set_cache_folder", "set_shared_cache_dir", "shared_cache_dir", "georef_flag", "put_georef_flag",
           "get_meta", "fetch", "positions", "nearest",
           "exif_transform", "image_size", "set_exiftool"]

#: bump when what is read from EXIF changes: older cache files are read again
CACHE_VERSION = 3
_LOCK = threading.RLock()
_MEM = {}                 # normcase path -> meta dict
_CALIB = {}               # collection key -> {"gsd_m", "yaw_offset", ...} or {"failed": reason}
_GEO = {}                 # normcase path -> [stamp, has a GeoTIFF georeference]
_CACHE_PATH = None
_EXIFTOOL = None
_FLAGS = 0x08000000 if os.name == "nt" else 0       # CREATE_NO_WINDOW

_TAGS = ["GPSLatitude", "GPSLongitude", "GPSAltitude", "RelativeAltitude", "FocalLength",
         "FocalLengthIn35mmFormat", "FocalPlaneXResolution", "FocalPlaneResolutionUnit",
         "GimbalYawDegree", "FlightYawDegree", "GPSImgDirection", "GPSTrack", "Yaw",
         "ImageWidth", "ImageHeight", "ExifImageWidth", "ExifImageHeight", "Model"]
_GPS_TAGS = ["GPSLatitude", "GPSLongitude"]


def _key(path):
    return os.path.normcase(os.path.abspath(str(path)))


_DIRSTAT = {}            # normcase folder -> (checked at, folder mtime, {name: [mtime, size]})
_DIRSTAT_LOCK = threading.Lock()


def _stamp(path):
    """[mtime, size] of a file, from its folder's listing (os.scandir).

    Not os.stat: on Windows a stat OPENS the file and the antivirus checks it --
    measured 37 ms per JPG of a FLIR folder, so validating the cache of 1,343
    frames cost ~50 s before a single byte of EXIF was read. A folder listing
    returns every file's time and size without opening any of them."""
    d, name = os.path.split(os.path.abspath(str(path)))
    nd = os.path.normcase(d)
    now = time.monotonic()
    with _DIRSTAT_LOCK:
        hit = _DIRSTAT.get(nd)
    if hit is not None and now - hit[0] < 2.0 and name.lower() in hit[2]:
        return hit[2][name.lower()]
    if hit is not None and name.lower() not in hit[2]:
        hit = None                                  # a file new since the listing: list again
    try:
        dm = os.stat(d).st_mtime                    # a folder: no antivirus scan
    except OSError:
        return None
    if hit is None or hit[1] != dm:
        entries = {}
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_file(follow_symlinks=False):
                            st = e.stat(follow_symlinks=False)
                            entries[e.name.lower()] = [round(st.st_mtime, 3), int(st.st_size)]
                    except OSError:
                        continue
        except OSError:
            return None
        hit = (now, dm, entries)
    else:
        hit = (now, dm, hit[2])
    with _DIRSTAT_LOCK:
        _DIRSTAT[nd] = hit
    return hit[2].get(name.lower())


def set_exiftool(path):
    global _EXIFTOOL
    _EXIFTOOL = path or None


def _exiftool():
    for cand in (_EXIFTOOL, os.environ.get("EXIFTOOL_PATH"), "exiftool"):
        if not cand:
            continue
        exe = os.path.join(cand, "exiftool.exe") if os.path.isdir(cand) else cand
        if os.path.isfile(exe) or shutil.which(exe):
            return exe
    return None


# ---------------------------------------------------------------- cache files
# One small JSON per IMAGE FOLDER, kept in the user's profile
# (%LOCALAPPDATA%\CanoPie\exif_cache), shared by every project: a folder of
# frames searched again -- from this project or another -- is not read again.
# Nothing is written into the image folders themselves (they may be on Google
# Drive or read-only). It holds each file's EXIF and the ground size / heading
# measured from its flight line.
_SHARED_DIR = None
_LOADED = set()                # normcase folders whose cache file is in memory
_DIRTY_DIRS = set()


def set_shared_cache_dir(path):
    """Where the per-folder cache files live (tests point it at a temp dir)."""
    global _SHARED_DIR
    with _LOCK:
        _SHARED_DIR = path
        _LOADED.clear()
        _DIRTY_DIRS.clear()


def shared_cache_dir():
    if _SHARED_DIR:
        return _SHARED_DIR
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "CanoPie", "exif_cache")


def _folder_of_key(key):
    if key.startswith("frame|"):
        return os.path.dirname(key[len("frame|"):])
    if "|" in key:
        return key.split("|", 1)[0]
    return os.path.dirname(key)


def _folder_file(nd):
    import hashlib
    h = hashlib.sha1(nd.encode("utf-8", "replace")).hexdigest()[:16]
    tail = "".join(c if c.isalnum() else "_" for c in os.path.basename(nd))[-40:]
    return os.path.join(shared_cache_dir(), f"{h}_{tail}.json")


def _ensure_loaded(nd):
    """Bring a folder's cache file into memory (once)."""
    if not nd or nd in _LOADED:
        return
    with _LOCK:
        if nd in _LOADED:
            return
        _LOADED.add(nd)
        p = _folder_file(nd)
        if not os.path.isfile(p):
            return
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            logging.warning("[canopy_meta] cache %s unreadable: %s", p, e)
            return
        if int(d.get("version", 0)) < CACHE_VERSION:
            return                                   # read with older rules: read the files again
        for k, v in (d.get("files") or {}).items():
            _MEM.setdefault(k, v)
        for k, v in (d.get("collections") or {}).items():
            _CALIB.setdefault(k, v)
        for k, v in (d.get("georef") or {}).items():
            _GEO.setdefault(k, v)


def set_cache_folder(project_folder):
    """Kept for callers. The cache is per image folder now (see above); an older
    <project>/canopy_exif_cache.json of this version is still read."""
    global _CACHE_PATH
    with _LOCK:
        p = os.path.join(project_folder, "canopy_exif_cache.json") if project_folder else None
        if p == _CACHE_PATH:
            return
        _CACHE_PATH = p
        if p and os.path.isfile(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                if int(d.get("version", 0)) >= CACHE_VERSION:
                    for k, v in (d.get("files") or {}).items():
                        _MEM.setdefault(k, v)
                        _DIRTY_DIRS.add(os.path.dirname(k))
                    for k, v in (d.get("collections") or {}).items():
                        _CALIB.setdefault(k, v)
            except Exception as e:
                logging.warning("[canopy_meta] cache %s unreadable: %s", p, e)


def save():
    """Write the cache file of every folder that changed."""
    with _LOCK:
        dirty = sorted(_DIRTY_DIRS)
        _DIRTY_DIRS.clear()
        if not dirty:
            return
        by_dir = {nd: ({}, {}, {}) for nd in dirty}
        for k, v in _MEM.items():
            t = by_dir.get(os.path.dirname(k))
            if t is not None:
                t[0][k] = v
        for k, v in _CALIB.items():
            t = by_dir.get(_folder_of_key(k))
            if t is not None:
                t[1][k] = v
        for k, v in _GEO.items():
            t = by_dir.get(os.path.dirname(k))
            if t is not None:
                t[2][k] = v
    try:
        os.makedirs(shared_cache_dir(), exist_ok=True)
    except OSError as e:
        logging.warning("[canopy_meta] no cache folder: %s", e)
        return
    for nd, (files, calib, geo) in by_dir.items():
        p = _folder_file(nd)
        try:
            tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"     # several writers at once
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": CACHE_VERSION, "folder": nd, "files": files, "collections": calib,
                           "georef": geo}, f)
            os.replace(tmp, p)
        except Exception as e:
            logging.warning("[canopy_meta] cache %s not saved: %s", p, e)


def georef_flag(path):
    """True / False when it is known whether this (unchanged) TIFF carries a
    GeoTIFF georeference, else None. Spares re-opening ~1,000 IR TIFFs of a
    raw-frame folder on every scan (on Google Drive each open is a download)."""
    k = _key(path)
    _ensure_loaded(os.path.dirname(k))
    with _LOCK:
        v = _GEO.get(k)
    if v and v[0] is not None and v[0] == _stamp(path):
        return bool(v[1])
    return None


def put_georef_flag(path, has_georef):
    k = _key(path)
    nd = os.path.dirname(k)
    _ensure_loaded(nd)
    st = _stamp(path)
    with _LOCK:
        if _GEO.get(k) != [st, bool(has_georef)]:
            _GEO[k] = [st, bool(has_georef)]
            _DIRTY_DIRS.add(nd)


def _put(path, meta):
    k = _key(path)
    nd = os.path.dirname(k)
    _ensure_loaded(nd)
    with _LOCK:
        meta = dict(meta)
        meta["stamp"] = _stamp(path)
        _MEM[k] = meta
        _DIRTY_DIRS.add(nd)


def get_meta(path, full=True):
    """Cached meta of `path` (None when unknown or stale). full=True also needs
    the camera fields (a positions-only entry does not count)."""
    k = _key(path)
    _ensure_loaded(os.path.dirname(k))
    with _LOCK:
        m = _MEM.get(k)
    if not m or m.get("stamp") != _stamp(path):
        return None
    if full and not m.get("full"):
        return None
    return m


# ---------------------------------------------------------------- reading EXIF
def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().split()[0].replace("+", ""))
    except (ValueError, IndexError):
        return None


def _from_exiftool(d):
    lat, lon = _num(d.get("GPSLatitude")), _num(d.get("GPSLongitude"))
    # Only DJI's gimbal / flight yaw is the image heading. A FLIR Duo Pro R's
    # XMP-Camera:Yaw is an IMU Euler angle of a camera pitched -84 deg (yaw -43
    # where the image heading measured from the overlap is ~95): near nadir the
    # Euler yaw is no heading at all, and trusting it put crowns 40 m off.
    yaw = None
    for k in ("GimbalYawDegree", "FlightYawDegree"):
        if _num(d.get(k)) is not None:
            yaw = _num(d.get(k)); break
    fpx, unit = _num(d.get("FocalPlaneXResolution")), _num(d.get("FocalPlaneResolutionUnit"))
    return {"lat": lat, "lon": lon, "alt_abs": _num(d.get("GPSAltitude")),
            "alt_rel": _num(d.get("RelativeAltitude")), "focal": _num(d.get("FocalLength")),
            "focal35": _num(d.get("FocalLengthIn35mmFormat")),
            "fp_xres": fpx, "fp_unit": unit, "yaw": yaw, "track": _num(d.get("GPSTrack")),
            "w": _num(d.get("ImageWidth") or d.get("ExifImageWidth")),
            "h": _num(d.get("ImageHeight") or d.get("ExifImageHeight")), "model": d.get("Model")}


def _run_exiftool(paths, tags, progress=None, cancel=None):
    exe = _exiftool()
    if exe is None:
        raise FileNotFoundError("exiftool not found")
    out = {}
    B = 150
    args_t = ["-" + t for t in tags]
    for i in range(0, len(paths), B):
        if cancel is not None and cancel():
            break
        batch = [os.path.abspath(p) for p in paths[i:i + B]]
        cmd = [exe, "-j", "-n", "-fast", "-api", "largefilesupport=1", "-charset", "filename=utf8",
               "-q", "-q"] + args_t + batch
        try:
            pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
            items = json.loads(pr.stdout.decode("utf-8", errors="ignore") or "[]")
        except Exception as e:
            logging.warning("[canopy_meta] exiftool batch failed: %s", e)
            items = []
        for it in items:
            src = it.get("SourceFile")
            if src:
                out[_key(src)] = it
        if progress is not None:
            progress(min(len(paths), i + B), len(paths))
    return out


_XMP_RE = {k: re.compile(rb'(?:drone-dji:)?' + k.encode() + rb'\s*=\s*"([+\-0-9.eE]+)"')
           for k in ("RelativeAltitude", "GimbalYawDegree", "FlightYawDegree")}


_HEAD_BYTES = 64 * 1024        # a JPG's EXIF + DJI XMP sit in its first segments
_WORKERS = 16                  # reading is I/O bound (Google Drive: ~40 ms per file each)


def _meta_from_tags(tags, head):
    def _ratio(v):
        try:
            return float(v.num) / float(v.den or 1)
        except Exception:
            return float(v)

    def _dms(tag, ref):
        t = tags.get(tag)
        if not t:
            return None
        d, m, s_ = (_ratio(x) for x in t.values[:3])
        val = d + m / 60.0 + s_ / 3600.0
        r = tags.get(ref)
        return -val if r is not None and str(r.values).upper() in ("S", "W") else val

    def _one(tag):
        # cameras put these in the EXIF sub-IFD or in IFD0 ("Image ..."): FLIR Duo Pro R uses IFD0
        for name in (tag, tag.replace("EXIF ", "Image ", 1)):
            t = tags.get(name)
            if t is None:
                continue
            try:
                return _ratio(t.values[0])
            except Exception:
                continue
        return None
    meta = {"lat": _dms("GPS GPSLatitude", "GPS GPSLatitudeRef"),
            "lon": _dms("GPS GPSLongitude", "GPS GPSLongitudeRef"),
            "alt_abs": _one("GPS GPSAltitude"), "focal": _one("EXIF FocalLength"),
            "focal35": _one("EXIF FocalLengthIn35mmFilm"), "fp_xres": _one("EXIF FocalPlaneXResolution"),
            "fp_unit": _one("EXIF FocalPlaneResolutionUnit"), "track": _one("GPS GPSTrack"),
            "model": str(tags.get("Image Model")) if tags.get("Image Model") is not None else None,
            "alt_rel": None, "yaw": None, "w": None, "h": None}
    # DJI's XMP only: three regex scans of a 64 KB header were 5 ms per file
    # (held under the GIL, so threads did not hide them) on FLIR frames that
    # never carry these tags
    xmp = head if head and b"drone-dji" in head else b""
    for k, rx in _XMP_RE.items():
        m = rx.search(xmp) if xmp else None
        if m:
            v = float(m.group(1))
            if k == "RelativeAltitude":
                meta["alt_rel"] = v
            elif meta["yaw"] is None:
                meta["yaw"] = v
    return meta


def _head_meta(path):
    """EXIF of one image from as few bytes as possible.

    A JPG: its first 64 KB only (EXIF, GPS, DJI XMP live in the first segments).
    exiftool -fast2 reads much further into a FLIR Duo Pro R JPG (it carries the
    thermal data in its own segments): over Google Drive that was ~0.35 s per
    frame -- 14 min for one collection -- against ~6 ms here, in threads.
    A TIFF (MicaSense, FLIR IR): exifread on the file itself, which seeks to the IFDs."""
    import io
    import exifread
    low = str(path).lower()
    if low.endswith((".jpg", ".jpeg", ".png")):
        with open(path, "rb") as f:
            head = f.read(_HEAD_BYTES)
        tags = exifread.process_file(io.BytesIO(head), details=False)
        if not tags.get("GPS GPSLatitude") and len(head) == _HEAD_BYTES:
            with open(path, "rb") as f:                   # a big APP segment first: read further once
                head = f.read(4 * _HEAD_BYTES)
            tags = exifread.process_file(io.BytesIO(head), details=False)
        return _meta_from_tags(tags, head)
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
        f.seek(0)
        head = f.read(_HEAD_BYTES)
    return _meta_from_tags(tags, head)


_from_exifread = _head_meta          # round-2 name


def _no_null_island(meta):
    """GPS written as 0, 0 (a phenocam's empty GPS block) is no GPS."""
    try:
        if meta.get("lat") is not None and abs(float(meta["lat"])) < 1e-9 and abs(float(meta.get("lon") or 0)) < 1e-9:
            meta["lat"] = meta["lon"] = None
    except (TypeError, ValueError):
        meta["lat"] = meta["lon"] = None
    return meta


def fetch(paths, full=True, progress=None, cancel=None):
    """Read and cache EXIF for `paths` without a valid cache entry: the file
    headers, 16 at a time. exiftool only for what the headers did not give a GPS
    position (a camera writing its EXIF somewhere unusual). `full` is kept for
    callers; one read gives every field the finder uses."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [p for p in dict.fromkeys(paths) if get_meta(p, full=False) is None]
    if not todo:
        return
    n = len(todo)
    done = 0
    missing = []

    def one(p):
        try:
            return p, _head_meta(p)
        except Exception as e:
            logging.debug("[canopy_meta] no EXIF in the header of %s: %s", p, e)
            return p, None
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for p, meta in ex.map(one, todo):
            if cancel is not None and cancel():
                break
            done += 1
            if meta is None or meta.get("lat") is None:
                missing.append((p, meta))
            else:
                meta["full"] = True
                _put(p, _no_null_island(meta))
            if progress is not None and (done % 50 == 0 or done == n):
                progress(done, n)
    if missing and _exiftool() is not None and not (cancel is not None and cancel()):
        got = {}
        try:
            got = _run_exiftool([p for p, _m in missing], _TAGS, None, cancel)
        except Exception as e:
            logging.debug("[canopy_meta] exiftool fallback: %s", e)
        for p, meta in missing:
            d = got.get(_key(p))
            if d is not None:
                meta = _from_exiftool(d)
            meta = meta or {"lat": None, "lon": None}
            meta["full"] = True
            _put(p, _no_null_island(meta))
    else:
        for p, meta in missing:
            meta = meta or {"lat": None, "lon": None}
            meta["full"] = True
            _put(p, _no_null_island(meta))
    save()


def positions(paths, project_folder=None, gps_exif_cache=None, progress=None, cancel=None):
    """{path: (lat, lon)} for the paths with a known position."""
    out, missing = {}, []
    stored = {}
    if project_folder:
        gp = os.path.join(project_folder, "gps_cache.json")
        if os.path.isfile(gp):
            try:
                data = json.load(open(gp, encoding="utf-8"))
                items = data if isinstance(data, list) else [
                    {"filename": k, "latitude": v[0], "longitude": v[1]} for k, v in data.items()]
                for it in items:
                    stored[_key(it["filename"])] = (float(it["latitude"]), float(it["longitude"]))
            except Exception as e:
                logging.debug("[canopy_meta] gps_cache.json: %s", e)
    for p in paths:
        m = get_meta(p, full=False)
        if m and m.get("lat") is None and "lat" in m:
            continue                                    # known: no GPS
        if m and m.get("lat") is not None and m.get("lon") is not None:
            out[p] = (m["lat"], m["lon"])
            continue
        hit = stored.get(_key(p))
        if hit is None and gps_exif_cache:
            try:
                c = gps_exif_cache.get((_key(p), os.path.getmtime(p)))
                if isinstance(c, dict) and c.get("latitude") is not None:
                    hit = (float(c["latitude"]), float(c["longitude"]))
            except OSError:
                pass
        if hit is not None and not (abs(hit[0]) < 1e-9 and abs(hit[1]) < 1e-9):
            out[p] = hit
        else:
            missing.append(p)
    if missing and not (cancel is not None and cancel()):
        fetch(missing, full=False, progress=progress, cancel=cancel)
        for p in missing:
            m = get_meta(p, full=False)
            if m and m.get("lat") is not None and m.get("lon") is not None:
                out[p] = (m["lat"], m["lon"])
    return out


# ---------------------------------------------------------------- selection
def _local_m(lat0, lon0, lat, lon):
    kx = 111320.0 * math.cos(math.radians(lat0))
    return (np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * 110540.0


def nearest(pos, lat, lon, k=5, max_dist_m=None, exclude=()):
    """The k frames nearest to (lat, lon): [(path, distance_m)], nearest first."""
    from scipy.spatial import KDTree
    ex = {_key(e) for e in exclude}
    items = [(p, c) for p, c in pos.items() if _key(p) not in ex]
    if not items:
        return []
    lats = np.array([c[0] for _p, c in items]); lons = np.array([c[1] for _p, c in items])
    x, y = _local_m(lat, lon, lats, lons)
    tree = KDTree(np.column_stack((x, y)))
    kk = min(int(k), len(items))
    d, idx = tree.query([0.0, 0.0], k=kk)
    d, idx = np.atleast_1d(d), np.atleast_1d(idx)
    out = []
    for dist, i in zip(d, idx):
        if max_dist_m is not None and dist > max_dist_m:
            continue
        out.append((items[int(i)][0], float(dist)))
    return out


# ---------------------------------------------------------------- collections
_DIR_CACHE = {}


def _pattern(path):
    """File name with its digits blanked: 20230117_153006_928_8b.JPG and
    20230117_153007_929_8b.JPG are one collection; ..._IR.TIFF is another;
    IMG_0113_1_radiance.tif and IMG_0114_1_radiance.tif share band 1."""
    stem, ext = os.path.splitext(os.path.basename(path))
    # runs of 3+ digits (date, time, capture number) blanked; a lone band digit stays
    return re.sub(r"\d{3,}", "#", stem) + ext.lower()


def collection_members(path):
    """The frames of the same camera / band in the same folder, in name (= time) order."""
    d = os.path.dirname(os.path.abspath(path))
    pat = _pattern(path)
    try:
        st = os.stat(d).st_mtime
    except OSError:
        st = None
    hit = _DIR_CACHE.get(d)
    if hit is None or hit[0] != st:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            names = []
        groups = {}
        for n in names:
            groups.setdefault(_pattern(n), []).append(os.path.join(d, n))
        hit = (st, groups)
        _DIR_CACHE[d] = hit
    return hit[1].get(pat, [os.path.abspath(path)])


def collection_key(path):
    meta = get_meta(path, full=False) or {}
    return "|".join([os.path.normcase(os.path.dirname(os.path.abspath(path))), _pattern(path),
                     str(meta.get("model") or ""), str(image_size(path) or "")])


def get_calibration(key):
    _ensure_loaded(_folder_of_key(key))
    with _LOCK:
        return _CALIB.get(key)


def put_calibration(key, calib):
    nd = _folder_of_key(key)
    _ensure_loaded(nd)
    with _LOCK:
        _CALIB[key] = dict(calib)
        _DIRTY_DIRS.add(nd)


def bearing_deg(lat0, lon0, lat1, lon1):
    """Compass bearing (degrees clockwise from north) from point 0 to point 1."""
    dx, dy = _local_m(lat0, lon0, lat1, lon1)
    return math.degrees(math.atan2(float(dx), float(dy))) % 360.0


def motion_heading(path, members=None):
    """Direction the camera moved at this frame: from the previous to the next
    frame of its collection (their GPS), else its GPS track."""
    members = members or collection_members(path)
    try:
        i = [os.path.normcase(m) for m in members].index(os.path.normcase(os.path.abspath(path)))
    except ValueError:
        i = -1
    if i >= 0:
        nb = [members[j] for j in (i - 1, i + 1) if 0 <= j < len(members)]
        pos = positions([path] + nb)
        a = pos.get(members[i - 1]) if i - 1 >= 0 else None
        b = pos.get(members[i + 1]) if i + 1 < len(members) else None
        c = pos.get(path)
        p0, p1 = (a or c), (b or c)
        if p0 and p1 and p0 != p1:
            dx, dy = _local_m(p0[0], p0[1], p1[0], p1[1])
            if math.hypot(float(dx), float(dy)) > 0.5:
                return bearing_deg(p0[0], p0[1], p1[0], p1[1])
    m = get_meta(path, full=False) or {}
    return m.get("track")


# ---------------------------------------------------------------- camera model
_SIZE_CACHE = {}


def image_size(path):
    """(w, h) from the file header (no decode)."""
    st = _stamp(path)
    k = _key(path)
    hit = _SIZE_CACHE.get(k)
    if hit and hit[0] == st:
        return hit[1]
    wh = None
    if str(path).lower().endswith((".tif", ".tiff")):
        try:
            from .shapefile_io import _raw_image_dims
            h, w = _raw_image_dims(path)
            if h and w:
                wh = (int(w), int(h))
        except Exception:
            wh = None
    if wh is None:
        try:
            from PIL import Image
            with Image.open(path) as im:
                wh = tuple(int(v) for v in im.size)
        except Exception:
            wh = None
    _SIZE_CACHE[k] = (st, wh)
    return wh


def _plausible(sw, focal):
    """A sensor width giving a horizontal field of view of 10-130 degrees."""
    if not (sw and focal):
        return False
    fov = 2.0 * math.degrees(math.atan(sw / (2.0 * float(focal))))
    return 10.0 <= fov <= 130.0


def sensor_width_mm(meta, w):
    """Sensor width from FocalPlaneXResolution, else the 35 mm equivalent focal
    length, else `default_mm`. Implausible values are skipped: a FLIR Duo Pro R
    writes 10.88 px/mm for its 4000 px RGB frame, i.e. a 368 mm "sensor"."""
    focal = meta.get("focal")
    if meta.get("fp_xres") and w:
        try:
            unit = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(meta.get("fp_unit") or 2), 25.4)
            sw = w / float(meta["fp_xres"]) * unit
            if _plausible(sw, focal):
                return sw
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if focal and meta.get("focal35"):
        sw = float(focal) / float(meta["focal35"]) * 36.0
        if _plausible(sw, focal):
            return sw
    return None


def exif_transform(meta, w, h, calib=None, heading=None):
    """(GDAL transform in EPSG:4326, info) of a nadir frame, or (None, reason).

    A frame with a full camera model in its EXIF (DJI: RelativeAltitude, focal
    length, sensor size, Gimbal / Flight yaw) uses it. Anything else -- FLIR Duo
    Pro R, MicaSense: GPSAltitude is above sea level, no heading -- needs `calib`,
    the ground size and heading measured from the frame's own flight line
    (canopy_finder.calibrate_frame); a typed flight height or sensor width was
    dropped (2026-10-12): it did not change where crowns were found."""
    if not meta or meta.get("lat") is None or meta.get("lon") is None:
        return None, "no GPS"
    focal = meta.get("focal")
    sw = sensor_width_mm(meta, w)
    agl = meta.get("alt_rel")
    camera_ok = bool(focal and sw and agl is not None and agl > 0)
    if calib and calib.get("gsd_m") and (not camera_ok or agl is None or meta.get("yaw") is None):
        # the collection's own overlap measured the ground size and the heading
        gsd = float(calib["gsd_m"])
        if meta.get("yaw") is not None:
            yaw, src = float(meta["yaw"]), "EXIF heading"
        elif calib.get("yaw_deg") is not None:
            yaw, src = float(calib["yaw_deg"]), "heading measured against the next frame"
        else:
            hd = heading if heading is not None else meta.get("track")
            yaw = (float(hd or 0.0) + float(calib.get("yaw_offset") or 0.0)) % 360.0
            src = "heading from the flight direction"
        return _transform(meta, w, h, gsd, yaw), {
            "gsd_m": gsd, "agl_m": None, "yaw_known": False,
            "note": f"ground size measured from the overlap with the next frame ({src})"}
    if not focal or not sw:
        return None, "no focal length / sensor size"
    notes = []
    yaw_known = meta.get("yaw") is not None
    if not yaw_known and meta.get("track") is not None:
        meta = dict(meta, yaw=meta["track"])
        notes.append("no heading in EXIF (GPS track used)")
    if agl is None or agl <= 0:
        return None, "no height above ground (measured from the overlap instead)"
    gsd = agl * sw / (float(focal) * float(w))                     # m / px
    if meta.get("yaw") is None:
        notes.append("no heading in EXIF")
    return _transform(meta, w, h, gsd, float(meta.get("yaw") or 0.0)), {
        "gsd_m": gsd, "agl_m": agl, "note": "; ".join(notes), "yaw_known": yaw_known}


def _transform(meta, w, h, gsd, yaw_deg):
    """Nadir frame centred on its GPS point, `gsd` m / px, top of the frame
    towards `yaw_deg` (clockwise from north)."""
    lat, lon = float(meta["lat"]), float(meta["lon"])
    dlat = gsd / 110540.0
    dlon = gsd / (111320.0 * math.cos(math.radians(lat)))
    y = math.radians(float(yaw_deg))
    # column +1 -> ground (cos y, -sin y) * gsd ; row +1 -> (-sin y, -cos y) * gsd   (E, N)
    t1, t2 = dlon * math.cos(y), -dlon * math.sin(y)
    t4, t5 = -dlat * math.sin(y), -dlat * math.cos(y)
    cx, cy = w / 2.0, h / 2.0
    t0 = lon - (t1 * cx + t2 * cy)
    t3 = lat - (t4 * cx + t5 * cy)
    return (t0, t1, t2, t3, t4, t5)
