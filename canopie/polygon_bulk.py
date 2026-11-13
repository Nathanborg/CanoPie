"""Copy many polygons to other images fast: record level, off the GUI thread.

"Replicate to all viewers" used to copy crowns item by item on the GUI
thread. For each crown it read the full outline from disk, built Python
point lists, wrote an indented JSON file, mapped every vertex through Qt, and
refreshed the Polygon Manager and every viewer's records. That was fine for a
few crowns. With 3,511 imported crowns (2.1 M vertices) it froze CanoPie.

Here the copy is a list of *jobs*: one crown, one target image. A worker
thread reads the source outline, scales it (with its drawn patches) to the
target image, and writes the target sidecar with the C JSON encoder. It sends
back a few hundred bytes per crown: the coarse ring and the metadata a
`polygon_lod.LazyPolygonRecord` needs. The GUI thread installs those lazy
records and draws them in chunks.

Placement is by image-size ratio. Points are kept as the same fractions of
the image frame (records are stored that way, in units of their
`image_ref_size`), only re-expressed in the target's pixel basis. Everything
else travels with the outline: label, shapefile attributes, Annotator values
and patches, mask flag.

Pure functions first (no Qt, unit-testable), then the QThread worker.
"""
import copy
import os
import time

import numpy as np
from PyQt5 import QtCore

from . import polygon_lod

#: Keys that describe the in-memory record, never a sidecar.
_TRANSIENT_KEYS = frozenset({
    "json_path", "display_points", "display_level", "full_point_count",
    "group", "lod", "pixmap_size",
})


def record_basis_wh(record, fallback_wh=None):
    """(w, h) of the pixel basis a record's points are expressed in."""
    if (dict.get(record, "coord_space", "image") or "image").lower() == "norm_pixmap":
        return (1.0, 1.0)
    ref = dict.get(record, "image_ref_size") or {}
    try:
        w, h = float(ref.get("w") or 0), float(ref.get("h") or 0)
    except (TypeError, ValueError, AttributeError):
        w = h = 0.0
    if w > 0 and h > 0:
        return (w, h)
    return fallback_wh


def _scaled(pts, sx, sy):
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if sx != 1.0 or sy != 1.0:
        a = a * (sx, sy)
    return a


def _scale_patches(patches, sx, sy, dst_wh):
    """Annotator patches (annotator.PATCHES_KEY): rings in the crown's basis."""
    out = {}
    for prop, entry in (patches or {}).items():
        if not isinstance(entry, dict):
            continue
        e = copy.deepcopy(entry)
        polys = []
        for rings in entry.get("polygons") or []:
            polys.append([_scaled(r, sx, sy).round(3).tolist() for r in rings])
        e["polygons"] = polys
        for k in ("area_px", "crown_area_px"):
            if isinstance(e.get(k), (int, float)):
                e[k] = float(e[k]) * sx * sy
        if dst_wh:
            e["basis"] = {"w": dst_wh[0], "h": dst_wh[1]}
        out[prop] = e
    return out


def copy_record(full, src_wh, dst_wh, stamp=None):
    """The whole record re-expressed in the target image's basis.

    `full` is the source record WITH its exact points (a parsed sidecar or an
    in-memory dict). `src_wh` is the basis its points are in (normally
    `record_basis_wh(full)`), and `dst_wh` is the target image's (w, h) in
    full-resolution pixels. Returns a new dict and never mutates `full`.
    """
    out = {}
    for k, v in full.items():
        if k in _TRANSIENT_KEYS or k in ("points", "annotator_patches"):
            continue
        out[k] = copy.deepcopy(v)
    space = (full.get("coord_space") or "image").lower()
    if space == "pixel":
        space = "image"
    out["coord_space"] = space
    pts = full.get("points") or []
    if space == "norm_pixmap" or not (src_wh and dst_wh):
        sx = sy = 1.0
        out["points"] = _scaled(pts, 1.0, 1.0).tolist() if len(pts) else []
        if space != "norm_pixmap" and dst_wh:
            out["image_ref_size"] = {"w": dst_wh[0], "h": dst_wh[1]}
    else:
        sx = float(dst_wh[0]) / float(src_wh[0])
        sy = float(dst_wh[1]) / float(src_wh[1])
        out["points"] = _scaled(pts, sx, sy).tolist() if len(pts) else []
        out["image_ref_size"] = {"w": dst_wh[0], "h": dst_wh[1]}
    patches = full.get("annotator_patches")
    if patches:
        out["annotator_patches"] = _scale_patches(
            patches, sx, sy, dst_wh if space != "norm_pixmap" else None)
    out["last_modified"] = stamp or time.strftime("%Y-%m-%d %H:%M:%S")
    return out


def coarse_points(pts, kind="polygon"):
    """The overview ring: what the viewers draw and `_overview.json` stores."""
    if not len(pts) or str(kind or "polygon").lower() == "point":
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2).tolist() if len(pts) else []
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(a) >= polygon_lod.MIN_POINTS_FOR_LOD:
        a = polygon_lod.decimate(a, 2.0 ** polygon_lod.OVERVIEW_LEVEL)
    return np.round(a, polygon_lod.COORD_DECIMALS).tolist()


def sidecar_path(polygons_dir, group, filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(polygons_dir, f"{group}_{base}_polygons.json")


def sidecar_payload(rec, group, root_id, fallback_coords=None):
    """What a sidecar holds. Same fields as ProjectTab._sidecar_payload, but
    without its per-vertex Python loop."""
    pts = rec.get("points") or []
    coords = rec.get("coordinates")
    if not (isinstance(coords, dict) and "latitude" in coords and "longitude" in coords):
        coords = fallback_coords
    data = {
        "points": np.asarray(pts, dtype=np.float64).reshape(-1, 2).tolist() if len(pts) else [],
        "name": rec.get("name", group),
        "root": root_id,
        "coordinates": coords,
        "type": rec.get("type", "polygon"),
        "coord_space": rec.get("coord_space", "image"),
        "image_ref_size": rec.get("image_ref_size", {}),
        "properties": rec.get("properties") or {},
    }
    if rec.get("is_mask_polygon"):
        data["is_mask_polygon"] = True
    if rec.get("annotator_patches"):
        data["annotator_patches"] = rec["annotator_patches"]
    if rec.get("last_modified"):
        data["last_modified"] = rec["last_modified"]
    return data


def lazy_meta(payload, group):
    """The metadata of a LazyPolygonRecord for a sidecar just written (the same
    fields ProjectTab._load_polygons_from_dir builds from the overview)."""
    pts = payload.get("points") or []
    return {
        "name": payload.get("name", group),
        "group": group,
        "root": payload.get("root", ""),
        "type": payload.get("type", "polygon"),
        "coord_space": payload.get("coord_space", "image"),
        "image_ref_size": payload.get("image_ref_size") or {},
        "display_points": coarse_points(pts, payload.get("type")),
        "display_level": polygon_lod.OVERVIEW_LEVEL,
        "full_point_count": len(pts),
    }


def trash_aside(path, tag):
    """Move an existing sidecar into polygons/.trash (same volume: a rename,
    no read) so undo can put it back. Returns the trash path or None."""
    if not os.path.exists(path):
        return None
    trash_dir = os.path.join(os.path.dirname(path), ".trash")
    os.makedirs(trash_dir, exist_ok=True)
    dst = os.path.join(trash_dir, f"{tag}_{os.path.basename(path)}")
    os.replace(path, dst)
    return dst


def link_meta(rec, group, root_id):
    """LazyPolygonRecord metadata for a crown copied by hard link: the source
    record's own (same file, same numbers), with the target's root. No IO --
    the coarse ring is the source's."""
    dp = dict.get(rec, "display_points")
    n = dict.get(rec, "full_point_count")
    if dp is None or n is None:
        pts = rec.get("points") or []
        dp = coarse_points(pts, dict.get(rec, "type", "polygon"))
        n = len(pts)
    return {
        "name": dict.get(rec, "name", group),
        "group": group,
        "root": root_id,
        "type": dict.get(rec, "type", "polygon"),
        "coord_space": dict.get(rec, "coord_space", "image"),
        "image_ref_size": dict.get(rec, "image_ref_size") or {},
        "display_points": dp,
        "display_level": dict.get(rec, "display_level", polygon_lod.OVERVIEW_LEVEL),
        "full_point_count": n,
    }


def run_job(job, stamp=None):
    """One crown onto one image. Safe off the GUI thread: only files and the
    job's own data. Returns (meta, trash_path, linked) or raises.

    A copy is placed by image-size ratio, i.e. at the same FRACTION of the
    image frame -- and records are stored as fractions (points in units of
    their image_ref_size). So the copy's sidecar is byte-for-byte the
    source's, and the fast path is a HARD LINK to it: a directory entry, no
    data read or written, ~0.2 ms (3,506 BCI crowns: 0.7 s, against 7-11 s
    writing copies; the file system and the virus scanner charge ~1-2 ms per
    NEW file). Every sidecar writer replaces the file (dump_sidecar: temp file
    + os.replace), so editing either crown later gives it its own file and the
    other one keeps the shared content.

    job keys:
      group, dst_fp, sidecar      -- what and where
      link_from, meta             -- hard-link this sidecar; meta for the record
      src                         -- else: the source record WITH points, or
      src_path                    -- the sidecar to read them from
      src_wh                      -- basis of the source points (w, h) fallback
      dst_wh                      -- target basis (w, h)
      root_id, coords             -- target's root id / fallback GPS
      replace                     -- an existing target sidecar is moved to trash
      tag                         -- unique trash prefix
    """
    trash = None
    if job.get("replace"):
        trash = trash_aside(job["sidecar"], job.get("tag", "bulk"))
    link_from = job.get("link_from")
    if link_from:
        try:
            os.link(link_from, job["sidecar"])
            return job["meta"], trash, True
        except OSError:
            pass    # FAT / exFAT, another volume, 1023-link limit: write a copy
    full = job.get("src")
    if full is None:
        path = job.get("src_path") or link_from
        full = polygon_lod.read_sidecar(path)
        if not isinstance(full, dict):
            raise IOError("could not read %s" % path)
    src_wh = record_basis_wh(full, job.get("src_wh"))
    rec = copy_record(full, src_wh, job.get("dst_wh"), stamp=stamp)
    payload = sidecar_payload(rec, job["group"], job.get("root_id", "0"), job.get("coords"))
    polygon_lod.dump_sidecar(job["sidecar"], payload, remember=True)
    return lazy_meta(payload, job["group"]), trash, False


def _run_chunk(jobs, indices, stamp):
    """Process-pool entry point: run a few jobs, report (index, meta, trash, error)."""
    out = []
    for i, job in zip(indices, jobs):
        try:
            meta, trash, linked = run_job(job, stamp=stamp)
            out.append((i, dict(meta, _linked=linked), trash, None))
        except Exception as e:
            out.append((i, None, None, str(e) or type(e).__name__))
    return out


class BulkPolygonWorker(QtCore.QObject):
    """Runs `run_job` for every job on a QThread (ShapefileImportWorker pattern).

    progress(done, total); finished(results, errors), where each result is
    (job_index, meta, trash_path) and each error is (job_index, message).
    Cancel stops before the next job. What is done is kept and reported, so
    the GUI side can install it, or undo it.
    """
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(list, list)

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    #: Parallel file IO. Opening a freshly written file costs ~30 ms on Windows
    #: (real-time scanning) -- 3,506 crowns one after another took 109 s,
    #: almost all of it waiting on open(). Waiting releases the GIL, so a pool
    #: overlaps it (save_incremental uses 16 for the same reason).
    IO_THREADS = 16

    #: From this many jobs on, use CanoPie's process pool: the work is JSON
    #: parsing / writing, which holds the GIL, so threads cannot share it out
    #: (3,506 BCI crowns: 13 s on threads; processes use every core).
    PROCESS_MIN_JOBS = 300
    CHUNK = 40

    @QtCore.pyqtSlot()
    def run(self):
        n = len(self.jobs)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        n_write = sum(1 for j in self.jobs if not j.get("link_from"))
        if n_write >= self.PROCESS_MIN_JOBS:
            try:
                return self._run_processes(stamp)
            except Exception as e:      # no pool (frozen edge cases): threads
                import logging
                logging.warning("[polygon_bulk] process pool unavailable (%s); using threads", e)
        return self._run_threads(stamp)

    #: Chunks in flight at once. Creating and renaming files serialises in the
    #: file system (and the virus scanner): on the BCI copy, 11 processes spent
    #: 44 s summed in write+rename against 4 s for one, and 4 in flight was
    #: the fastest (4.5 s for 3,506 crowns, against 10 s in one process).
    MAX_INFLIGHT = 4

    def _run_processes(self, stamp):
        import concurrent.futures as cf
        from .performance import get_cpu_executor
        ex = get_cpu_executor()
        n = len(self.jobs)
        pending = [list(range(i, min(n, i + self.CHUNK))) for i in range(0, n, self.CHUNK)]
        inflight = {}
        results, errors = [], []
        done = 0
        t_last = 0.0
        broken = False

        def _submit():
            while pending and len(inflight) < self.MAX_INFLIGHT and not self._cancel and not broken:
                c = pending.pop(0)
                inflight[ex.submit(_run_chunk, [self.jobs[i] for i in c], c, stamp)] = c

        _submit()
        while inflight or (pending and broken and not self._cancel):
            outs = []
            if inflight:
                finished, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
                for f in finished:
                    c = inflight.pop(f)
                    try:
                        outs.append(f.result())
                    except Exception as e:
                        # A dead pool (BrokenProcessPool) must not lose the
                        # copy: the rest runs here, in this worker thread.
                        if not broken:
                            import logging
                            logging.warning("[polygon_bulk] process pool failed (%s); "
                                            "finishing in this thread", e)
                            broken = True
                            try:
                                from . import performance
                                performance._cpu_executor = None
                            except Exception:
                                pass
                        pending.insert(0, c)
            elif pending:
                c = pending.pop(0)
                outs.append(_run_chunk([self.jobs[i] for i in c], c, stamp))
            _submit()
            for out in outs:
                for i, meta, trash, err in out:
                    done += 1
                    if err is None:
                        results.append((i, meta, trash))
                    else:
                        errors.append((i, err))
            now = time.perf_counter()
            if now - t_last > 0.1:
                t_last = now
                self.progress.emit(done, n)
        self.progress.emit(done, n)
        self.finished.emit(results, errors)

    def _run_threads(self, stamp):
        import concurrent.futures
        results, errors = [], []
        n = len(self.jobs)

        def _one(i):
            if self._cancel:
                return i, None, None, "cancelled"
            try:
                meta, trash, linked = run_job(self.jobs[i], stamp=stamp)
                return i, dict(meta, _linked=linked), trash, None
            except Exception as e:      # one bad crown never stops the batch
                return i, None, None, str(e) or type(e).__name__

        t_last = 0.0
        done = 0
        # Links are metadata operations the file system serialises anyway:
        # a few threads (1 thread: 0.67 s for 3,506; 8 threads: 0.97 s).
        n_threads = self.IO_THREADS if any(not j.get("link_from") for j in self.jobs) else 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            for i, meta, trash, err in ex.map(_one, range(n)):
                if err == "cancelled":
                    continue
                done += 1
                if err is None:
                    results.append((i, meta, trash))
                else:
                    errors.append((i, err))
                now = time.perf_counter()
                if now - t_last > 0.1:
                    t_last = now
                    self.progress.emit(done, n)
        self.progress.emit(done, n)
        self.finished.emit(results, errors)
