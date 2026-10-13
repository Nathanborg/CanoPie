"""
"Find canopy in other images…" -- the dialog, its worker, and how results are
applied to the project (saved records, zoom, temporary outlines).

The search itself is canopie/canopy_finder.py (no Qt). This module only:
  * collects the targets (this root's viewers, every root of the project, or
    the GeoTIFFs of another project / folder),
  * runs canopy_finder.find_in_target on a QThread (one target after another;
    no nested pools -- see ExportWorker in project_tab.py for why),
  * saves found crowns as records of the SAME group on current-project images
    (an existing record of that group is only replaced when asked, after a zip
    backup), zooms the live viewers onto them, exports registration squares,
  * opens the canopy browser (canopie/canopy_browser.py).
Images of another project or folder are never written to.
"""
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from . import canopy_finder as CF


def search_workers(requested=None):
    """Images searched at once. Default: a third of the logical cores, at most
    6 -- the rest stay free for the app and the system. Each search is CPU and
    memory bound: on a 12-thread laptop (i5-1335U: 2 fast + 8 efficiency cores,
    15 W) 4 at once gave 3.4x the speed of one and 6 at once only 3.1x."""
    n = os.cpu_count() or 2
    if requested:
        return max(1, min(int(requested), n))
    return max(1, min(6, n // 3))


def _low_priority():
    """Below-normal priority for the calling search thread (Windows)."""
    CF.lower_priority()

STATUS_TEXT = {
    "found": "found", "low_score": "low score", "geo": "geo only",
    "outside_footprint": "not in this image", "outside_frame": "outside the .ax crop",
    "no_georef": "not georeferenced", "error": "error",
}
SAVABLE = ("found", "geo")          # saved automatically; "low score" only from the table / browser
#: what "another project or folder" lists (raw drone frames are JPG / TIFF, phenocams JPG)
IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")

#: the image editor's look (image_editor_dialog.init_ui): compact, yellow checks with a dark-green
#: border, dark-green headings and primary button
STYLE = """
    QWidget { font-size: 11px; }
    QLabel { font-size: 12px; }
    QComboBox { padding: 0px 4px; min-height: 18px; font-size: 12px; }
    QPushButton { padding: 1px 6px; min-height: 18px; }
    QPushButton#primary { font-weight: bold; background-color: #006400; color: white; }
    QPushButton#primary:disabled { background-color: #9bb59b; color: #eeeeee; }
    QCheckBox, QRadioButton { font-size: 12px; }
    QCheckBox::indicator:checked, QListView::indicator:checked, QTableView::indicator:checked {
        background-color: #FFD700; border: 1px solid #006400; }
    QCheckBox::indicator:unchecked, QListView::indicator:unchecked, QTableView::indicator:unchecked {
        background-color: white; border: 1px solid gray; }
    QRadioButton::indicator:checked { background-color: #FFD700; border: 1px solid #006400; border-radius: 6px; }
    QRadioButton::indicator:unchecked { background-color: white; border: 1px solid gray; border-radius: 6px; }
    QGroupBox { border: 1px solid #006400; border-radius: 4px; margin-top: 8px; padding-top: 2px; }
    QGroupBox::indicator { width: 0px; height: 0px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #006400;
                       font-weight: bold; font-size: 12px; }
    QProgressBar { border: 1px solid #006400; border-radius: 3px; text-align: center; }
    QProgressBar::chunk { background-color: #006400; }
    QHeaderView::section { background-color: #e3eee3; color: #006400; font-weight: bold;
                           border: 0px; border-right: 1px solid #b8d3b8; padding: 2px 4px; }
    QListWidget::item:selected, QTableWidget::item:selected { background-color: #c5dcc5; color: black; }
"""


# ════════════════════════════════════════════════════════════════════════
# Project helpers
# ════════════════════════════════════════════════════════════════════════
def _norm(fp):
    return os.path.normcase(os.path.normpath(str(fp))) if fp else ""


def existing_record(pt, group, fp):
    """(stored key, record) of `group` on image `fp` (either path spelling)."""
    gmap = (getattr(pt, "all_polygons", {}) or {}).get(group) or {}
    if fp in gmap:
        return fp, gmap[fp]
    try:
        for g, key in pt._poly_index_lookup(fp):
            if g == group and key in gmap:
                return key, gmap[key]
    except Exception:
        pass
    n = _norm(fp)
    for key, rec in gmap.items():
        if _norm(key) == n:
            return key, rec
    return None, None


def project_files(pt):
    """[(filepath, root_name)] of every image of the project, root order."""
    out, seen = [], set()
    for attr in ("multispectral_image_data_groups", "thermal_rgb_image_data_groups"):
        groups = getattr(pt, attr, None) or {}
        names = list(getattr(pt, "multispectral_root_names", None) or []) if attr.startswith("multi") \
            else list(getattr(pt, "thermal_rgb_root_names", None) or [])
        for rn in names + [k for k in groups if k not in names]:
            for fp in groups.get(rn, []) or []:
                if fp and _norm(fp) not in seen:
                    seen.add(_norm(fp))
                    out.append((fp, rn))
    return out


def current_root_files(pt):
    out = []
    for w in getattr(pt, "viewer_widgets", []) or []:
        idata = w.get("image_data") if isinstance(w, dict) else None
        fp = getattr(idata, "filepath", None)
        if fp:
            out.append(fp)
    return out


def live_viewer_for(pt, fp):
    n = _norm(fp)
    for w in getattr(pt, "viewer_widgets", []) or []:
        v = w.get("viewer") if isinstance(w, dict) else None
        idata = getattr(v, "image_data", None) if v is not None else None
        if v is not None and _norm(getattr(idata, "filepath", None)) == n:
            return v
    return None


def image_folders_under(parent, max_depth=4):
    """Every folder under `parent` (itself included) that holds images."""
    out = []
    base = os.path.normpath(parent)
    for root, dirs, files in os.walk(base):
        depth = os.path.relpath(root, base).count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if any(f.lower().endswith(IMAGE_EXTS) for f in files):
            out.append(root)
    return sorted(out)


def external_files(path):
    """[(filepath, label, project_folder or None)] of a project folder (its
    project.json file lists) or of a plain folder (its TIFF / JPG / PNG images)."""
    import json
    pj = os.path.join(path, "project.json")
    if os.path.isfile(pj):
        try:
            d = json.load(open(pj, encoding="utf-8"))
        except Exception:
            d = {}
        out, seen = [], set()
        for key in ("multispectral_image_data_groups", "thermal_rgb_image_data_groups"):
            for rn, fps in (d.get(key) or {}).items():
                for fp in fps or []:
                    if fp and str(fp).lower().endswith(IMAGE_EXTS) and _norm(fp) not in seen \
                            and os.path.exists(fp):
                        seen.add(_norm(fp))
                        out.append((fp, f"{os.path.basename(path)} / {rn}", path))
        return out
    out = []
    try:
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(IMAGE_EXTS):
                out.append((os.path.join(path, name), os.path.basename(path), None))
    except OSError:
        pass
    return out


def ax_for(pt, fp, ext_project=None):
    """The .ax of a target: the current project's lookup, or the other
    project's <folder>/<stem>.ax, or none for a plain folder."""
    import json
    if ext_project:
        p = os.path.join(ext_project, os.path.splitext(os.path.basename(fp))[0] + ".ax")
        try:
            return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        except Exception:
            return {}
    try:
        return pt._load_ax_mods(fp) or {}
    except Exception:
        return {}


def match_bands(ax, n_bands):
    """The bands the viewer displays (file-band numbers, 0-based), else the first 3."""
    b = (ax or {}).get("display_bands")
    if isinstance(b, (list, tuple)) and b:
        try:
            bb = [int(x) for x in b if 0 <= int(x) < n_bands]
            if bb:
                return bb[:3]
        except (TypeError, ValueError):
            pass
    return list(range(min(3, n_bands)))


def record_scene_polygon(viewer, points, ref):
    """Record-space vertices -> scene QPolygonF, the same map load_polygons
    draws with (fraction of image_ref_size of the displayed pixmap)."""
    pixitem = getattr(viewer, "_image", None)
    if pixitem is None:
        return None
    cached = getattr(viewer, "_cached_pixmap_size", None)
    if cached and cached[0] and cached[1]:
        pw, ph = cached
    else:
        pm = pixitem.pixmap()
        if pm is None or pm.isNull():
            return None
        pw, ph = pm.width(), pm.height()
    rw, rh = float(ref.get("w") or 0), float(ref.get("h") or 0)
    if not (rw and rh):
        return None
    return QtGui.QPolygonF([pixitem.mapToScene(QtCore.QPointF(float(x) / rw * pw, float(y) / rh * ph))
                            for x, y in points])


def zoom_viewer_to(pt, viewer, points, ref):
    poly = record_scene_polygon(viewer, points, ref)
    if poly is None or poly.isEmpty():
        return False
    rect = poly.boundingRect()
    try:
        native = pt.polygon_basis_hw(viewer)
    except Exception:
        native = None
    if rect.width() < 1.0 and rect.height() < 1.0:
        viewer.smart_zoom_to_scene_rect(rect)
    else:
        viewer.smart_fit_to_scene_rect(rect, native)
    return True


def drop_viewer_group(pt, fp, group):
    """Take `group`'s outline off the live viewer of `fp`, so the redraw draws
    its new record. ProjectTab.load_polygons skips a group the viewer already
    shows (its guard against duplicates), so a replaced or edited crown kept
    its OLD outline on screen until the root was reloaded. The group also
    leaves the viewer's load record: an item missing from it must not read as
    "the user deleted this" (update_all_polygons) before the redraw."""
    v = live_viewer_for(pt, fp)
    if v is None:
        return 0
    try:
        gone = [it for it in v.get_all_polygons() if (getattr(it, "name", "") or "") == group]
    except Exception:
        return 0
    if not gone:
        return 0
    if getattr(v, "_vertex_editing_item", None) in gone:
        try:
            v.stop_vertex_editing()
        except Exception:
            pass
    for it in gone:
        try:
            v._remove_polygon_item_safely(it, defer_repaint=True)
        except Exception:
            pass
    try:
        v.prune_detached_polygon_records()           # its records and the zoomed-out tiles
    except Exception:
        pass
    loaded = getattr(v, "_loaded_polygon_names", None)
    if isinstance(loaded, set):
        loaded.discard(group)
    try:
        v.viewport().update()
    except Exception:
        pass
    return len(gone)


def save_found_record(pt, group, fp, points, ref, shape_type="polygon", replace=False):
    """Write a found crown as `group` on image `fp`. Returns "saved",
    "replaced" or "exists" (kept, replace not asked)."""
    key, old = existing_record(pt, group, fp)
    if old is not None and not replace:
        return "exists"
    if old is not None:
        # zip backup of the sidecar being replaced (polygons/.reproject_backup)
        try:
            pt._backup_sidecars_before_reproject(key, [(group, old)])
        except Exception as e:
            logging.error("[canopy_finder] backup failed for %s: %s -- not replaced", fp, e)
            return "exists"
    store_key = key if key is not None else fp
    rec = {
        "points": [(float(x), float(y)) for x, y in points],
        "coord_space": "image",
        "image_ref_size": {"w": int(ref["w"]), "h": int(ref["h"])},
        "name": group,
        "root": pt._sidecar_root_id(store_key) if hasattr(pt, "_sidecar_root_id") else "0",
        "type": shape_type or "polygon",
    }
    if old is not None:
        try:
            props = old.get("properties")
            if props:
                rec["properties"] = dict(props)
        except Exception:
            pass
    pt.all_polygons.setdefault(group, {})[store_key] = rec
    try:
        pt._add_to_polygon_index(group, store_key)
    except Exception:
        pass
    pt._mark_polygon_dirty(group, store_key)
    if old is not None:
        drop_viewer_group(pt, fp, group)
    return "replaced" if old is not None else "saved"


def update_record_points(pt, group, fp, points, ref):
    """An outline edited in the browser: update the stored record in place."""
    key, old = existing_record(pt, group, fp)
    if old is None:
        return False
    old["points"] = [(float(x), float(y)) for x, y in points]
    old["image_ref_size"] = {"w": int(ref["w"]), "h": int(ref["h"])}
    old["coord_space"] = "image"
    try:
        dict.pop(old, "display_points", None)
        dict.pop(old, "lod", None)
    except Exception:
        pass
    pt._mark_polygon_dirty(group, key)
    drop_viewer_group(pt, fp, group)
    return True


def refresh_after_save(pt, fps):
    try:
        pt._flush_dirty_polygons()
    except Exception:
        logging.debug("[canopy_finder] flush failed", exc_info=True)
    try:
        pt._write_polygon_overview()
    except Exception:
        logging.debug("[canopy_finder] overview rewrite failed", exc_info=True)
    try:
        pt.update_polygon_manager()
    except Exception:
        pass
    for fp in fps:
        v = live_viewer_for(pt, fp)
        if v is not None and getattr(v, "image_data", None) is not None:
            try:
                pt._schedule_polygons_for_viewer(v, v.image_data, delay_ms=0)
            except Exception:
                logging.debug("[canopy_finder] redraw failed", exc_info=True)


# ════════════════════════════════════════════════════════════════════════
# Which images to search (no Qt)
# ════════════════════════════════════════════════════════════════════════
MAX_LISTED_UNTICKED = 300


def classify_targets(src_fp, src_rec, src_ax, targets, opts, project_folder=None, gps_exif_cache=None,
                     progress=None, cancel=None):
    """[(target, state text, tick)], summary, source kind.

    * GeoTIFFs: ticked when the crown falls inside them.
    * Images with GPS in EXIF (raw drone frames): the k nearest to the CROWN by
      KD-tree, ticked; the others are only counted. Their footprint is checked
      during the search, not here: measuring each frame's ground size decodes its
      flight line, which made this list take minutes.
    * Images without any location: ticked when the source has none either
      (phenocam: matched in image coordinates)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import canopy_finder_meta as M
    cancel = cancel or (lambda: False)
    ax_of = opts.get("ax_of") or (lambda t: t.get("ax") or {})
    if progress is not None:
        progress(0, 1, "locating the source crown")
    sgeo = CF.frame_for(src_fp, ax=src_ax or {})
    if sgeo is None:
        return [], "the source image cannot be read", None
    raw_pts = sgeo.from_record(src_rec)
    centre = raw_pts.mean(axis=0)
    latlon = CF.crown_latlon(sgeo, centre) if sgeo.located else None
    n = len(targets)
    out, geo_t, rest = [], [], []

    def georef_of(t):
        fp = t["fp"]
        if not str(fp).lower().endswith((".tif", ".tiff")):
            return t, None
        if M.georef_flag(fp) is False:              # known (cache, file unchanged): a plain TIFF
            return t, None
        g = CF.georef(fp)
        M.put_georef_flag(fp, g is not None)
        return t, (CF.compose_ax(g, ax_of(t)) if g is not None else None)
    # TIFF headers in threads: a raw-frame folder on Google Drive holds ~1,000 IR TIFFs
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (t, g) in enumerate(ex.map(georef_of, targets)):
            if cancel():
                return [], "", sgeo.kind
            if progress is not None and i % 100 == 0:
                progress(i, n, "checking georeference")
            (geo_t if g is not None else rest).append((t, g))
    for t, g in geo_t:
        if not sgeo.located:
            out.append((t, "georeferenced (the source is not located)", False))
            continue
        try:
            cx, cy = CF.map_px(sgeo, g, [centre])[0]
            ok = 0 <= cx <= g.raw_w and 0 <= cy <= g.raw_h
            out.append((t, "GeoTIFF · covers the crown" if ok else "GeoTIFF · crown outside", ok))
        except Exception as e:
            out.append((t, f"GeoTIFF · ? ({e})", True))
    summary = []
    if rest:
        pos = M.positions([t["fp"] for t, _g in rest], project_folder, gps_exif_cache,
                          progress=(lambda i, m: progress(i, m, "reading GPS")) if progress else None,
                          cancel=cancel)
        gps_t = [t for t, _g in rest if t["fp"] in pos]
        nolo = [t for t, _g in rest if t["fp"] not in pos]
        if gps_t and sgeo.located and latlon is not None:
            # each folder is one flight (or one station / date): its own k nearest
            by_dir = {}
            for t in gps_t:
                by_dir.setdefault(os.path.normcase(os.path.dirname(os.path.abspath(t["fp"]))), []).append(t)
            listed, empty = 0, 0
            for nd, ts in by_dir.items():
                sub = {t["fp"]: pos[t["fp"]] for t in ts}
                near = M.nearest(sub, latlon[0], latlon[1], k=int(opts.get("k", 5)),
                                 max_dist_m=opts.get("max_dist_m", 300.0))
                by_fp = {t["fp"]: t for t in ts}
                tag = f"[{os.path.basename(nd)}] " if len(by_dir) > 1 else ""
                for k, (p, d) in enumerate(near):
                    out.append((by_fp[p], f"{tag}raw frame, GPS {d:.0f} m from the crown (nearest #{k + 1})", True))
                listed += len(near)
                empty += 0 if near else 1
            others = len(gps_t) - listed
            if len(by_dir) > 1:
                summary.append(f"{len(by_dir)} folders" + (f" ({empty} without a frame near the crown)" if empty
                                                          else ""))
            if others > 0:
                summary.append(f"{others} other image(s) with GPS, farther from the crown (not listed)")
        elif gps_t:
            summary.append(f"{len(gps_t)} image(s) with GPS not listed (the source is not located)")
        for k, t in enumerate(nolo):
            if not sgeo.located:
                out.append((t, "no location · same camera frame", True))
            elif k < MAX_LISTED_UNTICKED:
                out.append((t, "no location (the source is located)", False))
        if sgeo.located and len(nolo) > MAX_LISTED_UNTICKED:
            summary.append(f"{len(nolo) - MAX_LISTED_UNTICKED} more image(s) without location not listed")
    M.save()
    if progress is not None:
        progress(n, n, "done")
    return out, " · ".join(summary), sgeo.kind


class TargetScanWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object, str, object)

    def __init__(self, src_fp, src_rec, src_ax, targets, opts, project_folder, gps_exif_cache, parent=None):
        super().__init__(parent)
        self.args = (src_fp, src_rec, src_ax, targets, opts, project_folder, gps_exif_cache)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            items, summary, kind = classify_targets(*self.args, progress=self.progress.emit,
                                                    cancel=lambda: self._cancelled)
        except Exception as e:
            logging.exception("[canopy_finder] target scan failed")
            items, summary, kind = [], f"checking failed: {e}", None
        if not self._cancelled:
            self.done.emit(items, summary, kind)


# ════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════
class CanopyFindWorker(QtCore.QThread):
    """One or several crowns of one source image. The first crown's targets are
    the ticked list; another crown elsewhere in the image gets its OWN nearest
    raw frames / covering GeoTIFFs (the same check, `scan`, run here)."""
    progress = QtCore.pyqtSignal(int, int, str)
    source_ready = QtCore.pyqtSignal(object)
    result = QtCore.pyqtSignal(object, object)       # (Result, target dict incl. "group")
    failed = QtCore.pyqtSignal(str)

    def __init__(self, source_fp, jobs, source_ax, cfg, scan=None, parent=None):
        super().__init__(parent)
        self.source_fp, self.jobs, self.source_ax = source_fp, jobs, source_ax
        self.cfg, self.scan = dict(cfg), scan
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        work = []
        for job in self.jobs:
            if self._cancelled:
                break
            g = job["group"]
            sfp = job.get("source_fp") or self.source_fp
            sax = job.get("source_ax") if job.get("source_fp") else self.source_ax
            try:
                ctx = CF.prepare_source(sfp, job["rec"], sax, dict(self.cfg, bands=job.get("bands",
                                                                                          self.cfg.get("bands"))), g)
            except Exception as e:
                self.failed.emit(f"Crown '{g}': {e}")
                continue
            self.source_ready.emit(ctx)
            targets = job.get("targets")
            if targets is None and self.scan is not None:
                self.progress.emit(0, 1, f"'{g}': choosing images")
                scan = dict(self.scan)
                if _norm(sfp) != _norm(self.source_fp):
                    # a crown on another image: the dialog's images, plus the dialog's
                    # source image, minus its own
                    extra = [{"fp": self.source_fp, "root": scan.get("source_root", ""), "project": None,
                              "external": False}]
                    scan["targets"] = [t for t in list(scan.get("targets") or []) + extra
                                       if _norm(t["fp"]) != _norm(sfp)]
                scan.pop("source_root", None)
                try:
                    items, _summary, _kind = classify_targets(
                        sfp, job["rec"], sax, cancel=lambda: self._cancelled, **scan)
                except Exception as e:
                    self.failed.emit(f"Crown '{g}': {e}")
                    continue
                targets = []
                for t, _state, ok in items:
                    if ok:
                        t = dict(t)
                        t["ax"] = self.scan["opts"]["ax_of"](t)
                        targets.append(t)
            for t in targets or []:
                work.append((ctx, dict(t, group=g)))
        n = len(work)
        workers = search_workers(self.cfg.get("workers"))
        done = [0]
        lock = threading.Lock()
        self.used_processes = 0

        def report(r, t, how):
            with lock:
                done[0] += 1
                k = done[0]
            self.progress.emit(k, n, f"{t['group']} · {os.path.basename(t['fp'])} · {workers} {how}")
            self.result.emit(r, t)

        def one(item):
            ctx, t = item
            _low_priority()
            if self._cancelled:
                return None
            r = CF.find_in_target(ctx, t["fp"], t.get("ax"), self.cfg, cancel=lambda: self._cancelled)
            report(r, t, "in parallel")
            return r
        # `workers` searches at once, each single-threaded inside (OpenCV's own pool
        # off), at below-normal priority: the other cores stay free for the app.
        # Images read straight from their file go to worker processes (no shared
        # GIL); the ones that need the project's .ax replay run in threads here.
        frames = CF.frames_of(self.cfg)
        procs = workers > 1 and self.cfg.get("processes", True) and CF.processes_available()
        in_proc = [bool(procs) and CF.search_in_process_ok(frames, t["fp"], t.get("ax")) for _c, t in work]
        to_proc = [w for w, k in zip(work, in_proc) if k]
        rest = [w for w, k in zip(work, in_proc) if not k]
        old_cv = cv2.getNumThreads()
        cv2.setNumThreads(1)
        pool = None
        if len(to_proc) > 1:
            try:
                pool = CF.ProcessSearch(workers)
            except Exception as e:
                logging.warning("[canopy_finder] worker processes unavailable (%s): threads used", e)
        try:
            # flight lines are measured with their pair registrations in the processes
            self._locate(work, workers, pool.pair_shifts if pool is not None else None)
            if self._cancelled:
                return
            if pool is not None:
                from . import canopy_finder_meta as _M
                _M.save()                        # what the searches read in the workers
                rest += self._run_in_processes(pool, to_proc, workers, report)
            else:
                rest += to_proc
            if rest and not self._cancelled:
                self.progress.emit(done[0], n, f"searching {len(rest)} image(s), {workers} in parallel")
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(one, rest))
        finally:
            if pool is not None:
                pool.close(cancel=self._cancelled)
            cv2.setNumThreads(old_cv)
            from . import canopy_finder_meta as _M
            _M.save()
        self.progress.emit(n, n, "")

    def _locate(self, work, workers, pair_shifts=None):
        """Locate every target before any search. A GPS-only frame borrows the
        ground size of the first frame of its flight line that was measured, so
        the frames of one folder are located one after another in list order
        (nearest first) -- the same answer with 1 or 12 workers -- and folders in
        parallel. The measurements are cached (the worker processes read them)."""
        by_dir = {}
        for _c, t in work:
            by_dir.setdefault(os.path.normcase(os.path.dirname(os.path.abspath(t["fp"]))), []).append(t)
        n, done = len(work), [0]
        lock = threading.Lock()
        self.progress.emit(0, n, f"locating {n} image(s)")

        def folder(ts):
            _low_priority()
            for t in ts:
                if self._cancelled:
                    return
                try:
                    with CF.pair_shifts_with(pair_shifts):
                        CF.frame_for(t["fp"], ax=t.get("ax") or {})
                except Exception:
                    pass                         # the search reports it
                with lock:
                    done[0] += 1
                    k = done[0]
                self.progress.emit(k, n, f"locating {os.path.basename(t['fp'])}")
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(by_dir)))) as ex:
            list(ex.map(folder, by_dir.values()))

    def _run_in_processes(self, pool, items, workers, report):
        """Search `items` in the worker processes; returns what is left to run
        here (a pool that broke)."""
        from concurrent.futures import FIRST_COMPLETED, wait
        n = len(items)
        futs, left = {}, []
        for item in items:
            ctx, t = item
            try:
                futs[pool.search(ctx, t["fp"], t.get("ax"), self.cfg)] = item
            except Exception as e:
                logging.warning("[canopy_finder] worker processes failed (%s): threads used", e)
                left.append(item)
        self.used_processes = min(workers, n)
        self.progress.emit(0, n, f"searching {n} image(s), {self.used_processes} processes")
        pending = set(futs)
        while pending:
            if self._cancelled:
                return []
            fin, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for fut in fin:
                try:
                    r = fut.result()
                except Exception as e:               # a broken pool: the rest run here
                    logging.warning("[canopy_finder] worker process failed (%s): run here", e)
                    left.append(futs[fut])
                    continue
                report(r, futs[fut][1], "processes")
        return left


class TempOutline(QtWidgets.QGraphicsPolygonItem):
    """A found crown not saved yet, drawn dashed in a grid viewer. Drag it to
    move it; right-click to save it."""

    def __init__(self, poly, dialog, viewer, row, status):
        super().__init__(poly)
        self.dialog, self.viewer, self.row = dialog, viewer, row
        self._canopy_temp = True
        col = QtGui.QColor("#ffeb3b" if status in ("found", "geo") else "#ff9800")
        pen = QtGui.QPen(col, 0, QtCore.Qt.DashLine)
        pen.setCosmetic(True); pen.setWidthF(2.0)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 235, 59, 25)))
        self.setZValue(1e6)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setCursor(QtCore.Qt.SizeAllCursor)
        self.setToolTip(f"Found crown ({STATUS_TEXT.get(status, status)}) — not saved.\n"
                        "Drag to move it; right-click to save it.")

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        try:
            self.dialog.temp_outline_moved(self)
        except RuntimeError:
            pass

    def contextMenuEvent(self, ev):
        m = QtWidgets.QMenu()
        a_save = m.addAction("Save this outline")
        a_repl = m.addAction("Save, replacing the existing one")
        ch = m.exec_(ev.screenPos())
        if ch in (a_save, a_repl) and self.row is not None:
            self.dialog.temp_outline_moved(self)
            st = self.dialog.save_row(self.row, replace=(ch == a_repl))
            if st == "exists":
                g = self.dialog.results[self.row][1].get("group", self.dialog.group)
                QtWidgets.QMessageBox.information(None, "Find canopy", f"'{g}' already exists on that image (kept).")
        ev.accept()


# ════════════════════════════════════════════════════════════════════════
# Dialog
# ════════════════════════════════════════════════════════════════════════
#: where the folded state of the sections is kept (tests use their own)
SETTINGS_ID = ("CanoPie", "CanoPie")


class Section(QtWidgets.QGroupBox):
    """A dialog section that folds to its title: click "▾ Title" / "▸ Title".
    Its controls live in `body`; the folded state is remembered."""

    def __init__(self, title, key=None, parent=None):
        super().__init__(parent)
        self._title = title
        self._key = f"canopy_finder/folded/{key or title}"
        self.body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(self.body)
        self.setCheckable(True)
        try:
            folded = QtCore.QSettings(*SETTINGS_ID).value(self._key, False, type=bool)
        except Exception:
            folded = False
        self.setChecked(not folded)
        self._fold(not folded, save=False)
        self.toggled.connect(self._fold)
        self.setToolTip("Click the title to fold / unfold this section")

    def _fold(self, open_, save=True):
        self.body.setVisible(bool(open_))
        self.setTitle(("▾ " if open_ else "▸ ") + self._title)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred if open_ else QtWidgets.QSizePolicy.Maximum)
        if save:
            try:
                QtCore.QSettings(*SETTINGS_ID).setValue(self._key, not open_)
            except Exception:
                pass

    def is_open(self):
        return self.isChecked()

    def set_open(self, on):
        self.setChecked(bool(on))


class CanopyFinderDialog(QtWidgets.QDialog):
    """Non-modal; one per ProjectTab (ProjectTab.open_canopy_finder)."""

    COLS = ("Polygon", "Image", "Root / source", "Status", "Score", "Offset (m)", "Rotation (°)", "Saved",
            "Time (s)")

    def __init__(self, project_tab, group, source_fp):
        super().__init__(project_tab)
        self.pt = project_tab
        self.setWindowTitle("Find canopy in other images")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setModal(False)
        self.resize(920, 640)
        self.setStyleSheet(STYLE)
        self.worker = None
        self.ctx = None
        self.ctxs = {}
        self.results = []            # [(Result, target dict, saved_state)]
        self._temp_items = []        # (viewer, item) dashed outlines of unsaved results
        self.browser = None
        self._crown_src = {}             # crowns on OTHER images than the source (polygon manager)
        self._ext_path = None            # (one folder; kept for callers -- see _ext_paths)
        self._ext_paths = []
        self._src_rec = None
        self._saved_fps = []
        self.scan_worker = None
        self._scan_cache = {}
        self.frames = CF.Frames(project_tab)
        self.src_kind = None
        from . import canopy_finder_meta as M
        M.set_cache_folder(getattr(project_tab, "project_folder", None))
        M.set_exiftool(getattr(project_tab, "exiftool_path", None))
        self._build_ui()
        self.set_source(group, source_fp)

    # ---------------- UI ----------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        src_box = self.sec_crowns = Section("Crowns")
        sf = QtWidgets.QGridLayout(src_box.body)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.currentIndexChanged.connect(lambda _i: self._source_changed())
        sf.addWidget(QtWidgets.QLabel("Source image:"), 0, 0)
        sf.addWidget(self.source_combo, 0, 1, 1, 3)
        self.group_filter = QtWidgets.QLineEdit()
        self.group_filter.setPlaceholderText("filter polygons…")
        self.group_filter.textChanged.connect(self._filter_groups)
        b_gall = QtWidgets.QPushButton("All"); b_gnone = QtWidgets.QPushButton("None")
        b_gall.clicked.connect(lambda: self._tick_groups(True))
        b_gnone.clicked.connect(lambda: self._tick_groups(False))
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setMaximumHeight(84)
        self.group_list.setToolTip("Tick one or several crowns of the source image: each is searched for. "
                                   "The highlighted one drives the image list below.")
        self.group_list.itemChanged.connect(lambda _it: self._groups_changed())
        self.group_list.currentItemChanged.connect(lambda cur, _prev: self._current_group_changed(cur))
        sf.addWidget(QtWidgets.QLabel("Polygons:"), 1, 0, QtCore.Qt.AlignTop)
        sf.addWidget(self.group_list, 1, 1, 2, 1)
        sf.addWidget(self.group_filter, 1, 2, 1, 2)
        sf.addWidget(b_gall, 2, 2); sf.addWidget(b_gnone, 2, 3)
        sf.setColumnStretch(1, 1)
        lay.addWidget(src_box)

        tgt_box = self.sec_targets = Section("Search in")
        tl = QtWidgets.QVBoxLayout(tgt_box.body)
        row = QtWidgets.QHBoxLayout()
        self.rb_root = QtWidgets.QRadioButton("Viewers of this root")
        self.rb_project = QtWidgets.QRadioButton("All roots of this project")
        self.rb_ext = QtWidgets.QRadioButton("Other folders / projects:")
        self.rb_root.setChecked(True)
        self.ext_btn = QtWidgets.QPushButton("Add folders…")
        self.ext_btn.setToolTip("One or several folders of images (or project folders); each folder is one "
                                "flight: its nearest frames are searched")
        self.ext_sub_btn = QtWidgets.QPushButton("Add subfolders of…")
        self.ext_sub_btn.setToolTip("Every folder holding images under a parent folder (e.g. one per flight)")
        self.ext_rm_btn = QtWidgets.QPushButton("Remove")
        for w in (self.rb_root, self.rb_project, self.rb_ext, self.ext_btn, self.ext_sub_btn, self.ext_rm_btn):
            row.addWidget(w)
        row.addStretch(1)
        tl.addLayout(row)
        self.ext_list = QtWidgets.QListWidget()
        self.ext_list.setMaximumHeight(50)
        self.ext_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ext_list.setVisible(False)
        tl.addWidget(self.ext_list)
        for rb in (self.rb_root, self.rb_project, self.rb_ext):
            rb.toggled.connect(lambda on: on and self._refresh_targets())
        self.rb_ext.toggled.connect(lambda on: self.ext_list.setVisible(on or bool(self._ext_paths)))
        self.ext_btn.clicked.connect(self._browse_external)
        self.ext_sub_btn.clicked.connect(self._browse_subfolders)
        self.ext_rm_btn.clicked.connect(self._remove_folders)
        self.target_list = QtWidgets.QListWidget()
        self.target_list.setMaximumHeight(120)
        tl.addWidget(self.target_list)
        sel = QtWidgets.QHBoxLayout()
        b_all = QtWidgets.QPushButton("Tick all covering")
        b_none = QtWidgets.QPushButton("Untick all")
        b_all.clicked.connect(lambda: self._tick_all(True))
        b_none.clicked.connect(lambda: self._tick_all(False))
        sel.addWidget(b_all); sel.addWidget(b_none); sel.addStretch(1)
        tl.addLayout(sel)
        lay.addWidget(tgt_box)

        opt = QtWidgets.QHBoxLayout()
        set_box = self.sec_reg = Section("Registration")
        f = QtWidgets.QFormLayout(set_box.body)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("StackReg rigid (fast)", "stackreg")
        self.method.addItem("Exhaustive (rotation + shift search)", "exhaustive")
        self.method.addItem("Geo only (no registration)", "geo")
        def _dspin(lo, hi, val, dec=1, prefix="", suffix="", step=None):
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(lo, hi); sp.setDecimals(dec); sp.setValue(val)
            if prefix:
                sp.setPrefix(prefix)
            if suffix:
                sp.setSuffix(suffix)
            if step:
                sp.setSingleStep(step)
            return sp
        D = CF.DEFAULTS
        # per kind of image pair: GeoTIFF <-> GeoTIFF / a raw frame located by EXIF / no location
        self.radius = _dspin(0, 500, D["search_radius_m"], suffix=" m")
        self.angle = _dspin(0, 180, D["angle_range"], prefix="± ", suffix="°")
        self.radius_exif = _dspin(0, 1000, D["search_radius_exif_m"], suffix=" m")
        self.angle_exif = _dspin(0, 180, D["angle_range_exif"], prefix="± ", suffix="°")
        self.scale_lo = _dspin(0.3, 1.0, D["scale_min_exif"], dec=2, step=0.05)
        self.scale_hi = _dspin(1.0, 3.0, D["scale_max_exif"], dec=2, step=0.05)
        self.scale_exif = QtWidgets.QWidget()
        _sl = QtWidgets.QHBoxLayout(self.scale_exif); _sl.setContentsMargins(0, 0, 0, 0)
        _sl.addWidget(self.scale_lo); _sl.addWidget(QtWidgets.QLabel("–")); _sl.addWidget(self.scale_hi)
        self.radius_pct = _dspin(0, 100, D["search_radius_pct"], suffix=" % of the image")
        self.angle_pixel = _dspin(0, 180, D["angle_range_pixel"], prefix="± ", suffix="°")
        grid = QtWidgets.QGridLayout()
        for c, h in enumerate(("", "search radius", "rotation", "scale")):
            grid.addWidget(QtWidgets.QLabel(f"<i>{h}</i>"), 0, c)
        grid.addWidget(QtWidgets.QLabel("GeoTIFF ↔ GeoTIFF"), 1, 0)
        grid.addWidget(self.radius, 1, 1); grid.addWidget(self.angle, 1, 2)
        grid.addWidget(QtWidgets.QLabel("raw frames (EXIF GPS)"), 2, 0)
        grid.addWidget(self.radius_exif, 2, 1); grid.addWidget(self.angle_exif, 2, 2)
        grid.addWidget(self.scale_exif, 2, 3)
        grid.addWidget(QtWidgets.QLabel("no location (phenocam)"), 3, 0)
        grid.addWidget(self.radius_pct, 3, 1); grid.addWidget(self.angle_pixel, 3, 2)
        self.step = _dspin(0.1, 30, D["angle_step"], suffix="°")
        self.min_score = _dspin(-1, 1, D["min_ncc"], dec=2, step=0.05)
        self.k_near = QtWidgets.QSpinBox(); self.k_near.setRange(1, 50); self.k_near.setValue(5)
        kn = QtWidgets.QHBoxLayout(); kn.addWidget(self.k_near); kn.addWidget(QtWidgets.QLabel("per folder"))
        kn.addStretch(1)
        self.workers = QtWidgets.QSpinBox(); self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(search_workers())
        self.workers.setToolTip(f"Images searched at once, in separate worker processes "
                                f"({os.cpu_count()} logical cores here). The default leaves most cores to "
                                f"the app and the system; the searches run at below-normal priority.")
        f.addRow("Method:", self.method)
        f.addRow(grid)
        for w in (self.step, self.min_score, self.workers):
            w.setMaximumWidth(90)
        self.k_near.setMaximumWidth(60)
        for w in (self.radius, self.angle, self.radius_exif, self.angle_exif, self.radius_pct, self.angle_pixel):
            w.setMaximumWidth(150)
        g2 = QtWidgets.QGridLayout()
        g2.setHorizontalSpacing(6); g2.setVerticalSpacing(2)
        g2.addWidget(QtWidgets.QLabel("Rotation step:"), 0, 0); g2.addWidget(self.step, 0, 1)
        g2.addWidget(QtWidgets.QLabel("Min score (NCC):"), 0, 2); g2.addWidget(self.min_score, 0, 3)
        lk = QtWidgets.QLabel("Nearest raw frames:")
        lk.setToolTip("KD-tree on the GPS of the frames: the k nearest the crown in each folder")
        g2.addWidget(lk, 1, 0); g2.addLayout(kn, 1, 1)
        g2.addWidget(QtWidgets.QLabel("Parallel searches:"), 1, 2); g2.addWidget(self.workers, 1, 3)
        g2.setColumnStretch(4, 1)
        f.addRow(g2)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.k_near.valueChanged.connect(lambda _v: self._refresh_targets())
        opt.addWidget(set_box)

        act_box = self.sec_found = Section("When found")
        al = QtWidgets.QVBoxLayout(act_box.body)
        self.cb_save = QtWidgets.QCheckBox("Save found polygons to this project's images (same group)")
        self.cb_replace = QtWidgets.QCheckBox("Replace an existing polygon of this group (zip backup first)")
        self.cb_zoom = QtWidgets.QCheckBox("Zoom the viewers to the crown")
        self.cb_browser = QtWidgets.QCheckBox("Open the canopy browser")
        self.cb_save.setChecked(True); self.cb_zoom.setChecked(True); self.cb_browser.setChecked(True)
        for w in (self.cb_save, self.cb_replace, self.cb_zoom, self.cb_browser):
            al.addWidget(w)
        al.addWidget(QtWidgets.QLabel("<i>Images of another project / folder are never written to.</i>"))
        al.addStretch(1)
        opt.addWidget(act_box, 1)
        opt.setAlignment(set_box, QtCore.Qt.AlignTop)
        opt.setAlignment(act_box, QtCore.Qt.AlignTop)
        lay.addLayout(opt)

        run = QtWidgets.QHBoxLayout()
        self.btn_find = QtWidgets.QPushButton("Find")
        self.btn_find.setObjectName("primary")
        self.btn_find.setDefault(True)
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.progress = QtWidgets.QProgressBar(); self.progress.setTextVisible(True)
        self.btn_find.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        run.addWidget(self.btn_find); run.addWidget(self.btn_stop); run.addWidget(self.progress, 1)
        lay.addLayout(run)

        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda r, _c: self.show_result(r))
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.table.setMinimumHeight(140)
        self.table.verticalHeader().setDefaultSectionSize(20)
        lay.addWidget(self.table, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.ex_format = QtWidgets.QComboBox()
        for key, label in CF.EXPORT_FORMATS.items():
            self.ex_format.addItem(label, key)
        self.ex_format.setToolTip("The crop of the crown in the source and in every image it was found in, "
                                  "all aligned on the source crop (same grid: they stack), plus one metadata CSV")
        self.ex_dir = QtWidgets.QLineEdit(self.default_export_dir())
        self.ex_dir.setPlaceholderText("export folder")
        self.ex_dir.setToolTip("Default: canopy_exports inside this project. One sub-folder per crown.")
        self.ex_dir_btn = QtWidgets.QPushButton("…")
        self.ex_dir_btn.setFixedWidth(28)
        self.ex_dir_btn.setToolTip("Choose another folder")
        self.ex_dir_btn.clicked.connect(self._browse_export_dir)
        self.btn_export = QtWidgets.QPushButton("Export crowns")
        self.btn_export.setToolTip("Aligned crops of the source and targets + metadata CSV; rejected results "
                                   "are left out")
        self.btn_export.clicked.connect(lambda: self.export())
        self.btn_save_all = QtWidgets.QPushButton("Save all")
        self.btn_save_all.setToolTip("Save every outline not saved yet (found, low score, geo) "
                                     "on this project's images")
        self.btn_save_all.clicked.connect(lambda: self.save_all())
        self.btn_browser = QtWidgets.QPushButton("Canopy browser")
        self.btn_browser.clicked.connect(self.open_browser)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(QtWidgets.QLabel("Export:"))
        bottom.addWidget(self.ex_format)
        bottom.addWidget(self.ex_dir, 1)
        bottom.addWidget(self.ex_dir_btn)
        bottom.addWidget(self.btn_export)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_save_all); bottom.addWidget(self.btn_browser); bottom.addWidget(close)
        lay.addLayout(bottom)
        for sec in (self.sec_crowns, self.sec_targets, self.sec_reg, self.sec_found):
            bl = sec.body.layout()
            bl.setContentsMargins(4, 2, 4, 2)
            if isinstance(bl, QtWidgets.QFormLayout):
                bl.setVerticalSpacing(2)
            elif isinstance(bl, QtWidgets.QGridLayout):
                bl.setVerticalSpacing(3)
            else:
                bl.setSpacing(3)
        grid.setVerticalSpacing(2)
        self._set_result_buttons()

    def _method_changed(self):
        exh = self.method.currentData() == "exhaustive"
        geo = self.method.currentData() == "geo"
        for w in (self.angle, self.angle_exif, self.angle_pixel, self.step):
            w.setEnabled(exh)
        for w in (self.radius, self.radius_exif, self.radius_pct, self.scale_lo, self.scale_hi, self.min_score):
            w.setEnabled(not geo)

    def default_export_dir(self):
        pf = getattr(self.pt, "project_folder", None)
        return os.path.join(pf, "canopy_exports") if pf else ""

    def _browse_export_dir(self):
        start = self.ex_dir.text().strip() or self.default_export_dir()
        while start and not os.path.isdir(start):
            start = os.path.dirname(start) if os.path.dirname(start) != start else ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export the crowns into…", start)
        if d:
            self.ex_dir.setText(os.path.normpath(d))

    # ---------------- reject ----------------
    def is_rejected(self, row):
        return bool(self.results[row][1].get("rejected"))

    def set_rejected(self, row, on=True):
        """Leave a result out of Save all, the export and the browser (or take it
        back). A polygon already saved stays saved."""
        self.results[row][1]["rejected"] = bool(on)
        self._update_row(row)
        if on:
            self._drop_temp_outline(row)
        if self.browser is not None and self.browser.isVisible():
            self.browser.result_changed(row)

    def _drop_temp_outline(self, row):
        keep = []
        for v, it in self._temp_items:
            if getattr(it, "row", None) == row:
                try:
                    v.scene().removeItem(it)
                except Exception:
                    pass
            else:
                keep.append((v, it))
        self._temp_items = keep

    def _set_result_buttons(self):
        has = any(r.has_polygon and not t.get("rejected") for r, t, _s in self.results)
        self.btn_export.setEnabled(has)
        self.btn_save_all.setEnabled(any(r.has_polygon and not t.get("external") and not t.get("rejected")
                                         and s not in ("saved", "replaced") for r, t, s in self.results))
        self.btn_browser.setEnabled(bool(self.results) and self.ctx is not None)

    # ---------------- source / targets ----------------
    def set_source(self, group, source_fp):
        """(Re)aim the dialog at a crown: the source combo lists the images holding
        its group, the polygon list the crowns of the chosen image (this one ticked)."""
        self.group = group
        self._wanted_groups = {group}
        self._crown_src = {}
        files = []
        gmap = (self.pt.all_polygons or {}).get(group) or {}
        seen = set()
        for key in gmap:
            if _norm(key) not in seen:
                seen.add(_norm(key))
                files.append(key)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for fp in files:          # (how each is located is reported once chosen, off the UI thread)
            self.source_combo.addItem(os.path.basename(fp), fp)
        idx = next((i for i, fp in enumerate(files) if _norm(fp) == _norm(source_fp)), 0)
        self.source_combo.setCurrentIndex(idx)
        self.source_combo.blockSignals(False)
        self._fill_groups()
        self._refresh_targets()

    def source_fp(self):
        return self.source_combo.currentData()

    def set_crowns(self, crowns):
        """Several crowns, each on its own image (any root): the first one's image
        is the source; the others are listed as "crown  [image]" and each is
        searched from its own image. The search covers every root."""
        crowns = [(g, fp) for g, fp in crowns if g is not None and fp]
        if not crowns:
            return
        self.set_source(crowns[0][0], crowns[0][1])
        sfp = self.source_fp()
        self._wanted_groups = {g for g, _fp in crowns}
        self._crown_src = {g: fp for g, fp in crowns if _norm(fp) != _norm(sfp)}
        self._fill_groups()
        if self._crown_src and not self.rb_ext.isChecked():
            self.rb_project.setChecked(True)
        self._refresh_targets()

    def crown_source(self, group):
        """The image a crown is searched from."""
        return self._crown_src.get(group) or self.source_fp()

    # ---------------- the polygon list ----------------
    def _groups_on(self, fp):
        try:
            gs = {g for g, _k in self.pt._poly_index_lookup(fp)}
        except Exception:
            gs = {g for g, m in (self.pt.all_polygons or {}).items() if isinstance(m, dict)
                  and any(_norm(k) == _norm(fp) for k in m)}
        return sorted(gs, key=lambda g: (g != self.group, str(g).lower()))

    def _fill_groups(self):
        sfp = self.source_fp()
        self.group_list.blockSignals(True)
        self.group_list.clear()
        cur = None
        here = self._groups_on(sfp) if sfp else []
        for g in here + [g for g in self._crown_src if g not in here]:
            elsewhere = g in self._crown_src and g not in here
            it = QtWidgets.QListWidgetItem(f"{g}  [{os.path.basename(self._crown_src[g])}]" if elsewhere else str(g))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if g in self._wanted_groups else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, g)
            if elsewhere:
                it.setToolTip(f"On {self._crown_src[g]}: searched from that image")
            self.group_list.addItem(it)
            if g == self.group:
                cur = it
        if cur is not None:
            self.group_list.setCurrentItem(cur)
        self.group_list.blockSignals(False)
        self._filter_groups(self.group_filter.text())

    def _filter_groups(self, text):
        t = (text or "").lower()
        for i in range(self.group_list.count()):
            it = self.group_list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def _tick_groups(self, on):
        self.group_list.blockSignals(True)
        for i in range(self.group_list.count()):
            it = self.group_list.item(i)
            if not it.isHidden():
                it.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.group_list.blockSignals(False)
        self._groups_changed()

    def selected_groups(self):
        out = []
        for i in range(self.group_list.count()):
            it = self.group_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.append(it.data(QtCore.Qt.UserRole))
        return out

    def _groups_changed(self):
        sel = self.selected_groups()
        self._wanted_groups = set(sel)
        if sel and self.group not in sel:
            self._set_current_group(sel[0])
        self.btn_find.setEnabled(bool(sel) and self.src_kind is not None)

    def _current_group_changed(self, item):
        if item is not None and item.data(QtCore.Qt.UserRole) != self.group:
            self._set_current_group(item.data(QtCore.Qt.UserRole))

    def _set_current_group(self, g):
        self.group = g
        self._refresh_targets()

    def _source_changed(self):
        self._fill_groups()
        if self.group not in self._groups_on(self.source_fp()):
            sel = self.selected_groups()
            if sel:
                self.group = sel[0]
        self._refresh_targets()

    def _browse_external(self):
        """Pick one or several folders (Ctrl / Shift click in the folder list)."""
        dlg = QtWidgets.QFileDialog(self, "Folders of images (GeoTIFFs, raw frames, phenocam) or project folders")
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)     # the native one picks one folder
        if self._ext_paths:
            dlg.setDirectory(os.path.dirname(self._ext_paths[-1]))
        for view in dlg.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        if dlg.exec_():
            self.add_folders(dlg.selectedFiles())

    def _browse_subfolders(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Parent folder (its image folders are added)",
                                                       os.path.dirname(self._ext_paths[-1]) if self._ext_paths else "")
        if d:
            self.add_folders(image_folders_under(d))

    def _remove_folders(self):
        rm = {it.text() for it in self.ext_list.selectedItems()} or set(self._ext_paths[-1:])
        self.add_folders([], remove=rm)

    def add_folders(self, folders, remove=()):
        """Set the folder list (scope 'Other folders / projects') and rescan."""
        paths = [p for p in self._ext_paths if p not in set(remove)]
        for f in folders:
            f = os.path.normpath(f)
            if os.path.isdir(f) and f not in paths:
                paths.append(f)
        self._ext_paths = paths
        self._ext_path = paths[0] if paths else None
        self.ext_list.clear()
        self.ext_list.addItems(paths)
        self.ext_list.setVisible(bool(paths))
        if self.rb_ext.isChecked():
            self._refresh_targets()
        else:
            self.rb_ext.setChecked(True)

    def _candidate_targets(self):
        sfp = self.source_fp()
        out = []
        if self.rb_ext.isChecked():
            paths = self._ext_paths or ([self._ext_path] if self._ext_path else [])
            seen = set()
            for folder in paths:
                for fp, label, proj in external_files(folder):
                    if _norm(fp) != _norm(sfp) and _norm(fp) not in seen:
                        seen.add(_norm(fp))
                        out.append({"fp": fp, "root": label, "project": proj, "external": True})
            return out
        roots = {_norm(fp): rn for fp, rn in project_files(self.pt)}
        if self.rb_project.isChecked():
            files = [fp for fp, _rn in project_files(self.pt)]
        else:
            files = current_root_files(self.pt)
        for fp in files:
            if _norm(fp) != _norm(sfp):
                out.append({"fp": fp, "root": roots.get(_norm(fp), ""), "project": None, "external": False})
        return out

    def _scope_key(self):
        scope = "ext" if self.rb_ext.isChecked() else ("project" if self.rb_project.isChecked() else "root")
        _k, rec = existing_record(self.pt, self.group, self.source_fp()) if self.source_fp() else (None, None)
        pts = tuple(map(tuple, (rec or {}).get("points") or [])) if rec is not None else ()
        ext = tuple(self._ext_paths or ([self._ext_path] if self._ext_path else []))
        return (scope, ext if scope == "ext" else None, _norm(self.source_fp()), self.group,
                hash(pts), self.k_near.value(),
                tuple(_norm(f) for f in current_root_files(self.pt)) if scope == "root" else None)

    def _refresh_targets(self):
        """Classify the targets OFF the UI thread (a project with many roots means
        thousands of files): GeoTIFF footprints, GPS positions + KD-tree for raw
        frames, plain frames for phenocams. Cached per scope."""
        sfp = self.source_fp()
        if not sfp:
            self.target_list.clear()
            return
        key = self._scope_key()
        cache = getattr(self, "_scan_cache", None)
        if cache is None:
            cache = self._scan_cache = {}
        if key in cache:
            self._fill_targets(*cache[key])
            return
        _k, src_rec = existing_record(self.pt, self.group, sfp)
        if src_rec is None:
            self.target_list.clear()
            self.btn_find.setEnabled(False)
            return
        old = getattr(self, "scan_worker", None)
        if old is not None and old.isRunning():
            old.cancel()
        self.target_list.clear()
        self.target_list.addItem("checking the images…")
        self.btn_find.setEnabled(False)
        rec = {k: src_rec.get(k) for k in ("points", "coord_space", "image_ref_size", "type")
               if src_rec.get(k) is not None}
        w = TargetScanWorker(sfp, rec, ax_for(self.pt, sfp), self._candidate_targets(), self._scan_opts(),
                             getattr(self.pt, "project_folder", None),
                             getattr(self.pt, "_gps_exif_cache", None), self)
        w.progress.connect(lambda i, n, msg: (self.progress.setRange(0, max(1, n)), self.progress.setValue(i),
                                              self.progress.setFormat(f"{msg} {i} / {n}")))
        w.done.connect(lambda items, summary, src_kind, k=key, ww=w: self._on_scan_done(k, ww, items, summary,
                                                                                       src_kind))
        self.scan_worker = w
        w.start()

    def _scan_opts(self):
        return {"k": self.k_near.value(),
                "ax_of": lambda t, pt=self.pt: ax_for(pt, t["fp"], t.get("project"))}

    def _on_scan_done(self, key, worker, items, summary, src_kind):
        if worker is not getattr(self, "scan_worker", None):
            return                       # a newer scan replaced it
        self._scan_cache[key] = (items, summary, src_kind)
        self._fill_targets(items, summary, src_kind)

    def _fill_targets(self, items, summary, src_kind):
        self.target_list.setUpdatesEnabled(False)
        self.target_list.clear()
        for t, state, ok in items:
            if not t["external"]:
                _k, ex = existing_record(self.pt, self.group, t["fp"])
                if ex is not None:
                    state += f" · has '{self.group}'"
            it = QtWidgets.QListWidgetItem(f"{os.path.basename(t['fp'])}   [{t['root']}]   — {state}")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if ok else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, t)
            it.setData(QtCore.Qt.UserRole + 1, ok)
            self.target_list.addItem(it)
        if summary:
            it = QtWidgets.QListWidgetItem(summary)
            it.setFlags(QtCore.Qt.NoItemFlags)
            self.target_list.addItem(it)
        self.target_list.setUpdatesEnabled(True)
        self.src_kind = src_kind
        self.btn_find.setEnabled(src_kind is not None and bool(self.selected_groups() or [self.group]))
        self.progress.setRange(0, 1); self.progress.setValue(0)
        self.progress.setFormat({"geotiff": "source: GeoTIFF", "exif": "source: raw frame located by EXIF",
                                 "pixel": "source: not located (matched in image coordinates)"}.get(
            src_kind, "the source image cannot be read"))

    def _tick_all(self, on):
        for i in range(self.target_list.count()):
            it = self.target_list.item(i)
            if not it.data(QtCore.Qt.UserRole):
                continue
            it.setCheckState(QtCore.Qt.Checked if (on and it.data(QtCore.Qt.UserRole + 1)) else QtCore.Qt.Unchecked)

    def cfg(self):
        return {"method": self.method.currentData(), "search_radius_m": self.radius.value(),
                "angle_range": self.angle.value(), "angle_step": self.step.value(),
                "search_radius_exif_m": self.radius_exif.value(), "angle_range_exif": self.angle_exif.value(),
                "scale_min_exif": self.scale_lo.value(), "scale_max_exif": self.scale_hi.value(),
                "search_radius_pct": self.radius_pct.value(),
                "angle_range_pixel": self.angle_pixel.value(),
                "refine_step": min(CF.DEFAULTS["refine_step"], self.step.value()),
                "min_ncc": self.min_score.value(), "frames": self.frames, "workers": self.workers.value()}

    # ---------------- run ----------------
    def start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        sfp = self.source_fp()
        _k, src_rec = existing_record(self.pt, self.group, sfp)
        if src_rec is None:
            QtWidgets.QMessageBox.warning(self, "Find canopy", "The source polygon was not found.")
            return
        targets = []
        for i in range(self.target_list.count()):
            it = self.target_list.item(i)
            if it.checkState() == QtCore.Qt.Checked and it.data(QtCore.Qt.UserRole):
                t = dict(it.data(QtCore.Qt.UserRole))
                t["ax"] = ax_for(self.pt, t["fp"], t.get("project"))
                targets.append(t)
        if not targets:
            QtWidgets.QMessageBox.information(self, "Find canopy", "Tick at least one image to search.")
            return
        src_ax = ax_for(self.pt, sfp)
        cfg = self.cfg()
        cfg["bands"] = match_bands(src_ax, CF.file_band_count(sfp))

        def _plain(rec):            # a lazy record pages its points in on .get
            return {k: rec.get(k) for k in ("points", "coord_space", "image_ref_size", "type", "name")
                    if rec.get(k) is not None}
        groups = self.selected_groups() or [self.group]
        if self.group not in groups:
            self.group = groups[0]
        jobs = [{"group": self.group, "rec": _plain(src_rec), "targets": targets}]
        for g in groups:
            if g == self.group:
                continue
            gfp = self.crown_source(g)
            _k, r_ = existing_record(self.pt, g, gfp)
            if r_ is None:
                continue
            job = {"group": g, "rec": _plain(r_), "targets": None}
            if _norm(gfp) != _norm(sfp):
                gax = ax_for(self.pt, gfp)
                job.update(source_fp=gfp, source_ax=gax, bands=match_bands(gax, CF.file_band_count(gfp)))
            jobs.append(job)
        try:
            src_root = self.pt.get_root_by_filepath(sfp) or ""
        except Exception:
            src_root = ""
        scan = {"targets": self._candidate_targets(), "opts": self._scan_opts(),
                "project_folder": getattr(self.pt, "project_folder", None),
                "gps_exif_cache": getattr(self.pt, "_gps_exif_cache", None), "source_root": src_root}
        self._clear_temp_items()
        if self.browser is not None:
            # its tiles index the rows about to be replaced
            self.browser.hide()
            self.browser.entries, self.browser.visible = [], []
        self.results = []
        self.table.setRowCount(0)
        self._saved_fps = []
        self.ctxs = {}
        self.ctx = None
        self._src_rec = jobs[0]["rec"]
        self._src_recs = {j["group"]: j["rec"] for j in jobs}
        self.worker = CanopyFindWorker(sfp, jobs, src_ax, cfg, scan, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.source_ready.connect(self._on_source)
        self.worker.result.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.btn_find.setEnabled(False); self.btn_stop.setEnabled(True)
        self.progress.setRange(0, len(targets)); self.progress.setValue(0)
        self._t0 = time.perf_counter()
        self.worker.start()

    def stop(self):
        if self.worker is not None:
            self.worker.cancel()

    def _on_progress(self, i, n, name):
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)
        self.progress.setFormat(f"{i} / {n}  {name}" if name else f"{i} / {n}")

    def _on_source(self, ctx):
        self.ctxs[ctx.group] = ctx
        if self.ctx is None or ctx.group == self.group:
            self.ctx = ctx

    def ctx_of(self, row):
        g = self.results[row][1].get("group", self.group)
        return self.ctxs.get(g, self.ctx)

    def group_of(self, row):
        return self.results[row][1].get("group", self.group)

    def _on_failed(self, msg):
        QtWidgets.QMessageBox.warning(self, "Find canopy", msg)

    def _on_result(self, r, t):
        saved = ""
        g = t.get("group", self.group)
        if r.has_polygon and not t.get("external") and self.cb_save.isChecked() and r.status in SAVABLE:
            saved = save_found_record(self.pt, g, t["fp"], r.record_points, r.image_ref_size,
                                      getattr(self.ctxs.get(g, self.ctx), "shape_type", "polygon"),
                                      self.cb_replace.isChecked())
            if saved in ("saved", "replaced"):
                self._saved_fps.append(t["fp"])
        elif t.get("external"):
            saved = "external"
        self.results.append([r, t, saved])
        self._add_row(len(self.results) - 1)
        self._set_result_buttons()

    def _add_row(self, i):
        r, t, saved = self.results[i]
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = (str(t.get("group", self.group)), os.path.basename(r.target_fp), t.get("root", ""),
                STATUS_TEXT.get(r.status, r.status)
                + (f" — {r.message}" if r.message and r.status == "error" else ""),
                "" if r.score is None else f"{r.score:.3f}",
                f"{r.offset_m:.2f}" if r.has_polygon else "", f"{r.angle:+.1f}" if r.has_polygon else "",
                saved, f"{r.seconds:.2f}")
        for c, v in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(v)
            if c == 3:
                col = {"found": "#2e7d32", "geo": "#1565c0", "low_score": "#ef6c00"}.get(r.status, "#c62828")
                it.setForeground(QtGui.QBrush(QtGui.QColor(col)))
            self.table.setItem(row, c, it)

    def _update_row(self, i):
        r, t, saved = self.results[i]
        it = self.table.item(i, 7)
        if it is not None:
            it.setText("rejected" if t.get("rejected") else saved)
        grey = QtGui.QColor("#9e9e9e")
        for c in range(self.table.columnCount()):
            cell = self.table.item(i, c)
            if cell is None:
                continue
            f = cell.font()
            f.setStrikeOut(bool(t.get("rejected")))
            cell.setFont(f)
            if t.get("rejected"):
                if cell.data(QtCore.Qt.UserRole + 7) is None:
                    cell.setData(QtCore.Qt.UserRole + 7, cell.foreground())
                cell.setForeground(grey)
            elif cell.data(QtCore.Qt.UserRole + 7) is not None:
                cell.setForeground(cell.data(QtCore.Qt.UserRole + 7))
                cell.setData(QtCore.Qt.UserRole + 7, None)
        self._set_result_buttons()

    def _on_finished(self):
        self.btn_find.setEnabled(True); self.btn_stop.setEnabled(False)
        n_found = sum(1 for r, _t, _s in self.results if r.status in ("found", "geo"))
        self.progress.setFormat(f"{n_found} of {len(self.results)} found in "
                                f"{time.perf_counter() - getattr(self, '_t0', time.perf_counter()):.1f} s")
        if getattr(self, "_saved_fps", None):
            refresh_after_save(self.pt, self._saved_fps)
        if self.cb_zoom.isChecked():
            QtCore.QTimer.singleShot(150, self.zoom_all)
        if self.cb_browser.isChecked() and self.ctx is not None and self.results:
            self.open_browser()
        self._set_result_buttons()

    # ---------------- results -> viewers ----------------
    def _result_record_space(self, r):
        return r.record_points, r.image_ref_size

    def zoom_all(self):
        """Zoom every live viewer showing the source or a result onto the crown;
        draw a dashed, draggable outline for results that were not saved."""
        self._clear_temp_items()
        sfp = self.source_fp()
        v = live_viewer_for(self.pt, sfp)
        if v is not None and self._src_rec is not None:
            zoom_viewer_to(self.pt, v, self._src_rec.get("points") or [], self._src_rec.get("image_ref_size") or {})
        for i, (r, t, saved) in enumerate(self.results):
            if not r.has_polygon or t.get("external") or t.get("rejected"):
                continue
            v = live_viewer_for(self.pt, r.target_fp)
            if v is None:
                continue
            pts, ref = self._result_record_space(r)
            zoom_viewer_to(self.pt, v, pts, ref)
            if saved not in ("saved", "replaced", "browser"):
                self._add_temp_outline(v, pts, ref, r.status, i)

    def _add_temp_outline(self, viewer, pts, ref, status, row=None):
        poly = record_scene_polygon(viewer, pts, ref)
        if poly is None:
            return
        item = TempOutline(poly, self, viewer, row, status)
        try:
            viewer.scene().addItem(item)
            self._temp_items.append((viewer, item))
            viewer._has_canopy_temp = True
        except Exception:
            pass

    def temp_outline_moved(self, item):
        """A dashed outline was dragged in a grid viewer: the result follows it."""
        row, viewer = item.row, item.viewer
        if row is None or not (0 <= row < len(self.results)):
            return
        r, t, _s = self.results[row]
        scene_poly = item.mapToScene(item.polygon())
        pts = []
        for k in range(scene_poly.count()):
            ip = self.pt.scene_to_image_coords(viewer, scene_poly.at(k))
            pts.append((float(ip.x()), float(ip.y())))
        h, w = self.pt.polygon_basis_hw(viewer)
        if not (h and w):
            return
        r.record_points, r.image_ref_size = pts, {"w": int(w), "h": int(h)}
        try:
            g = CF.frame_for(r.target_fp, fetch_missing=False, ax=t.get("ax") or ax_for(self.pt, r.target_fp))
            r.raw_pts = g.from_record({"points": pts, "coord_space": "image", "image_ref_size": r.image_ref_size})
        except Exception:
            logging.debug("[canopy_finder] moved outline -> raw failed", exc_info=True)
        r.extra["moved"] = True
        if self.browser is not None and self.browser.isVisible():
            self.browser.result_changed(row)

    def _clear_temp_items(self):
        for v, item in self._temp_items:
            try:
                v._has_canopy_temp = False
            except RuntimeError:
                pass
            try:
                sc = item.scene()
                if sc is not None:
                    sc.removeItem(item)
            except RuntimeError:
                pass
        self._temp_items = []

    def show_result(self, row):
        """Double-click: zoom to it if it is on screen, else go to its root."""
        if not (0 <= row < len(self.results)):
            return
        r, t, _saved = self.results[row]
        if not r.has_polygon:
            return
        if t.get("external"):
            self.open_browser(focus_fp=r.target_fp)
            return
        show_in_grid(self.pt, r.target_fp, r.record_points, r.image_ref_size)

    def save_row(self, row, replace=None):
        r, t, saved = self.results[row]
        if not r.has_polygon or t.get("external") or saved in ("saved", "replaced"):
            return saved
        st = save_found_record(self.pt, self.group_of(row), t["fp"], r.record_points, r.image_ref_size,
                               getattr(self.ctx_of(row), "shape_type", "polygon"),
                               self.cb_replace.isChecked() if replace is None else replace)
        self.results[row][2] = st
        self._update_row(row)
        if st in ("saved", "replaced"):
            refresh_after_save(self.pt, [t["fp"]])
            self.zoom_all()
        return st

    def save_all(self, ask=True):
        """Save every unsaved result with an outline on a current-project image
        (found, low score or geo). Images already holding the group are replaced
        only after one question (zip backup first). Returns the number saved."""
        rows = [i for i, (r, t, s) in enumerate(self.results)
                if r.has_polygon and not t.get("external") and not t.get("rejected")
                and s not in ("saved", "replaced")]
        if not rows:
            if ask:
                QtWidgets.QMessageBox.information(self, "Save all", "Nothing left to save.")
            return 0
        exists = [i for i in rows
                  if existing_record(self.pt, self.group_of(i), self.results[i][1]["fp"])[1] is not None]
        replace = False
        if exists and ask:
            ans = QtWidgets.QMessageBox.question(
                self, "Save all",
                f"{len(exists)} of the {len(rows)} outline(s) are for images that already hold that polygon.\n"
                "Replace them too (a zip backup is written first)?\n\n"
                "No = save only the others.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.No)
            if ans == QtWidgets.QMessageBox.Cancel:
                return 0
            replace = ans == QtWidgets.QMessageBox.Yes
        fps, n = [], 0
        for i in rows:
            r, t, _s = self.results[i]
            st = save_found_record(self.pt, self.group_of(i), t["fp"], r.record_points, r.image_ref_size,
                                   getattr(self.ctx_of(i), "shape_type", "polygon"), replace)
            self.results[i][2] = st
            self._update_row(i)
            if st in ("saved", "replaced"):
                fps.append(t["fp"]); n += 1
        if fps:
            refresh_after_save(self.pt, fps)
            self.zoom_all()
        self._set_result_buttons()
        if self.browser is not None and self.browser.isVisible():
            self.browser.refresh_page()
        return n

    def _table_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        r, t, saved = self.results[row]
        m = QtWidgets.QMenu(self)
        a_show = m.addAction("Show / zoom")
        a_save = m.addAction("Save this polygon")
        a_repl = m.addAction("Save, replacing the existing one")
        m.addSeparator()
        a_rej = m.addAction("Restore (back into Save all and the export)" if t.get("rejected")
                            else "Reject (leave out of Save all and the export)")
        ok = r.has_polygon and not t.get("external") and not t.get("rejected") and saved not in ("saved", "replaced")
        a_save.setEnabled(ok); a_repl.setEnabled(ok)
        ch = m.exec_(self.table.viewport().mapToGlobal(pos))
        if ch == a_rej:
            self.set_rejected(row, not t.get("rejected"))
        elif ch == a_show:
            self.show_result(row)
        elif ch == a_save:
            if self.save_row(row) == "exists":
                QtWidgets.QMessageBox.information(self, "Find canopy",
                                                  f"'{self.group_of(row)}' already exists on that image (kept).")
        elif ch == a_repl:
            self.save_row(row, replace=True)

    # ---------------- export / browser ----------------
    def export(self, groups=None, folder=None, fmt=None, quiet=False):
        """Per crown, into <folder>/<crown>_<source>/: the source crop and the
        crop of every target it was found in (rejected ones left out), aligned on
        the source crop, as GeoTIFF or JPEG, + <crown>_metadata.csv. Default
        folder: canopy_exports inside the project. Returns the crown folders."""
        if not self.results or not self.ctxs:
            return []
        d = folder or self.ex_dir.text().strip()
        if not d:
            self._browse_export_dir()
            d = self.ex_dir.text().strip()
            if not d:
                return []
        fmt = fmt or self.ex_format.currentData() or "tif"
        outs, n_img = [], 0
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            os.makedirs(d, exist_ok=True)
            for g, ctx in self.ctxs.items():
                if groups is not None and g not in groups:
                    continue
                mine = [(r, t, s_) for r, t, s_ in self.results
                        if t.get("group", self.group) == g and not t.get("rejected")]
                out, files = CF.export_crowns(ctx, [r for r, _t, _s in mine], d, fmt=fmt, frames=self.frames,
                                              saved={r.target_fp: s_ for r, _t, s_ in mine})
                outs.append(out)
                n_img += len(files) - 1
        except Exception as e:
            logging.exception("[canopy_finder] export failed")
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Export", f"Export failed: {e}")
            return outs
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not quiet:
            QtWidgets.QMessageBox.information(
                self, "Export", f"{len(outs)} crown(s): {n_img} aligned crop(s) + metadata CSV in\n"
                                + "\n".join(outs))
        return outs

    def open_browser(self, focus_fp=None):
        if self.ctx is None:
            return
        from .canopy_browser import CanopyBrowser
        if self.browser is None:
            self.browser = CanopyBrowser(self)
        self.browser.set_entries(self.ctxs or {self.ctx.group: self.ctx}, self.results, group=self.group)
        self.browser.show(); self.browser.raise_(); self.browser.activateWindow()
        if focus_fp:
            self.browser.focus_file(focus_fp)

    def closeEvent(self, ev):
        self.stop()
        if self.scan_worker is not None:
            self.scan_worker.cancel()
            self.scan_worker.wait(3000)
        if self.worker is not None:
            self.worker.wait(3000)
        self._clear_temp_items()
        super().closeEvent(ev)


def show_in_grid(pt, fp, points, ref, _tries=0):
    """Zoom the grid viewer showing `fp` onto the crown, loading its root first."""
    v = live_viewer_for(pt, fp)
    if v is not None and getattr(v, "_image", None) is not None:
        zoom_viewer_to(pt, v, points, ref)
        return
    if _tries == 0:
        try:
            rn = pt.get_root_by_filepath(fp)
            rid = (getattr(pt, "root_id_mapping", {}) or {}).get(rn)
            if rid is not None:
                pt.go_to_root_id(int(rid))
            elif rn in (pt.multispectral_root_names or []):
                pt._jump_to_index(pt.multispectral_root_names.index(rn))
        except Exception:
            logging.debug("[canopy_finder] navigation failed", exc_info=True)
    if _tries < 60:     # the root loads asynchronously: look again for up to ~15 s
        QtCore.QTimer.singleShot(250, lambda: show_in_grid(pt, fp, points, ref, _tries + 1))
