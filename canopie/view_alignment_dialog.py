"""
The window of viewer alignment (view_alignment.py): spots on the reference
image (picked automatically, clicked, or the user's own polygons), matched in
the other images of the root or of every root with the canopy finder's
worker; one fit per image (local by default: it follows each mosaic's
distortions), the canopy browser to correct spots, re-fit, and apply.

Reuses, unchanged in their logic:
  canopy_finder_dialog.CanopyFindWorker  -- the matching (threads / processes,
                                            COG overviews, the georeference prior)
  canopy_browser.CanopyBrowser           -- the tiles where a spot that was not
                                            found is dragged into place
This window provides what the browser asks of its parent (pt, frames, results,
_update_row, _drop_temp_outline, save_all, export); saving is hidden there
(`browser_allows_save = False`): spots are not crowns.
"""
import logging
import os
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import canopy_finder as CF
from . import canopy_finder_dialog as D
from . import view_alignment as VA

__all__ = ["AlignmentDialog", "ClickSpotsMode"]


def _affine_from_3(P, Q):
    """The 2x3 affine taking the three points P to Q."""
    X = np.column_stack((np.asarray(P, np.float64), np.ones(3)))
    return np.linalg.solve(X, np.asarray(Q, np.float64)).T


class AlignmentDialog(QtWidgets.QDialog):
    browser_allows_save = False

    def __init__(self, pt, ref_viewer, target_fps=None, spot_scene_pts=None, cfg=None):
        super().__init__(pt, QtCore.Qt.Window)
        self.setWindowTitle("Align viewers")
        self.setStyleSheet(D.STYLE)
        self.pt = pt
        self.ref_viewer = ref_viewer
        self.ref_fp = ref_viewer.image_data.filepath
        self.ref_ax = (pt._load_ax_mods(self.ref_fp) or {}) if hasattr(pt, "_load_ax_mods") else {}
        self.ref_geo = CF.frame_for(self.ref_fp, ax=self.ref_ax)
        self.frames = CF.Frames(pt)
        self.targets = []
        self._explicit_targets = target_fps is not None
        if target_fps is not None:
            self._set_targets(target_fps)
        self._clicker = None
        self.spot_scene_pts = None if spot_scene_pts is None else np.asarray(spot_scene_pts, np.float64)
        self.extra_cfg = dict(cfg or {})
        self.results = []           # [Result, target dict, saved state] -- the browser's model
        self.ctxs = {}              # spot group -> SourceContext
        self.spots = []             # [{"group", "rec", "center"}]
        self.fits = {}              # target fp -> fit dict
        self.worker = None
        self.browser = None
        self._prior = None          # {fp: 2x3 export ref -> export target}
        self._build_ui()
        self.resize(860, 420)
        self._update_info()

    def _set_targets(self, fps):
        self.targets = []
        for f in fps:
            ax = (self.pt._load_ax_mods(f) or {}) if hasattr(self.pt, "_load_ax_mods") else {}
            root = None
            try:
                root = self.pt.get_root_by_filepath(f)
            except Exception:
                root = None
            self.targets.append({"fp": f, "ax": ax, "root": root or os.path.basename(f), "project": None,
                                 "external": False})

    def _scope_targets(self):
        if self._explicit_targets:
            return [t["fp"] for t in self.targets]
        return self.pt._view_align_target_fps(self.ref_fp, all_roots=self.scope.currentData() == "all")

    def set_source(self, kind):
        i = self.source.findData(kind)
        if i >= 0:
            self.source.setCurrentIndex(i)

    def _n_polygons(self):
        try:
            return len(self.pt._view_align_polygon_records(self.ref_fp))
        except Exception:
            return 0

    def _update_info(self):
        try:
            n = len(self._scope_targets())
        except Exception:
            n = len(self.targets)
        self.info.setText(f"Reference: <b>{os.path.basename(self.ref_fp)}</b> — {n} image(s) to align. "
                          "Choose the spots and press <b>Match</b>.")

    # ---------------- UI ----------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row0 = QtWidgets.QHBoxLayout()
        self.scope = QtWidgets.QComboBox()
        self.scope.addItem("This root", "root")
        self.scope.addItem("All roots", "all")
        self.scope.setToolTip("Align the other images of this root, or every image of the project, "
                              "to this image")
        self.source = QtWidgets.QComboBox()
        self.source.addItem("Automatic", "auto")
        self.source.addItem("Click them on the image", "click")
        self.source.addItem("My polygons on this image", "polygons")
        self.source.setToolTip("Where the spots come from. Your polygons: each crown drawn on this "
                               "image is matched (the most spread ones, up to the number of spots)")
        self.n_spots = QtWidgets.QSpinBox()
        self.n_spots.setRange(3, 60)
        self.n_spots.setValue(12)
        self.n_spots.setToolTip("More spots let the local fit follow the distortions of each mosaic")
        self.fit_mode = QtWidgets.QComboBox()
        self.fit_mode.addItem("Local (follows distortions)", "local")
        self.fit_mode.addItem("Global (one shift, rotation, scale)", "global")
        self.fit_mode.setToolTip("Local: each place takes its shift, rotation and scale from the nearby "
                                 "spots. Global: one similarity for the whole image")
        row0.addWidget(QtWidgets.QLabel("Images")); row0.addWidget(self.scope)
        row0.addSpacing(12)
        row0.addWidget(QtWidgets.QLabel("Spots")); row0.addWidget(self.source); row0.addWidget(self.n_spots)
        row0.addSpacing(12)
        row0.addWidget(QtWidgets.QLabel("Fit")); row0.addWidget(self.fit_mode)
        row0.addStretch(1)
        lay.addLayout(row0)
        self.scope.currentIndexChanged.connect(lambda _i: self._update_info())
        self.fit_mode.currentIndexChanged.connect(lambda _i: self.fits and self.refit())
        row = QtWidgets.QHBoxLayout()
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setRange(2.0, 150.0); self.radius.setValue(40.0); self.radius.setSuffix(" m")
        self.radius.setToolTip("How far from the georeferenced position a spot is searched")
        self.method = QtWidgets.QComboBox()
        self.method.addItem("StackReg (fast)", "stackreg")
        self.method.addItem("Exhaustive (rotation search)", "exhaustive")
        row.addWidget(QtWidgets.QLabel("Search radius")); row.addWidget(self.radius)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Matching")); row.addWidget(self.method)
        row.addStretch(1)
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        lay.addWidget(self.progress)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Image", "Spots used", "Fit error", "Between spots",
                                              "Scale", "Rotation", "Status"])
        self.table.horizontalHeaderItem(3).setToolTip(
            "Each spot predicted from the others: the accuracy to expect between spots")
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        btns = QtWidgets.QHBoxLayout()
        self.b_match = QtWidgets.QPushButton("Match")
        self.b_match.setToolTip("Pick the spots and search them in the images")
        btns.addWidget(self.b_match)
        self.b_match.clicked.connect(self._on_match)
        self.b_browser = QtWidgets.QPushButton("Show matches…")
        self.b_browser.setToolTip("The spots in every image: drag a spot that was not found to its place, "
                                  "reject a wrong one, then Re-fit")
        self.b_refit = QtWidgets.QPushButton("Re-fit")
        self.b_rematch = QtWidgets.QPushButton("Re-match with this fit")
        self.b_rematch.setToolTip("Search the spots again starting from the current fit (useful when "
                                  "the georeference is far off and you corrected a few spots by hand)")
        self.b_apply = QtWidgets.QPushButton("Apply")
        self.b_apply.setDefault(True)
        self.b_close = QtWidgets.QPushButton("Close")
        for b in (self.b_browser, self.b_refit, self.b_rematch):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.b_apply); btns.addWidget(self.b_close)
        lay.addLayout(btns)
        self.b_browser.clicked.connect(self.open_browser)
        self.b_refit.clicked.connect(self.refit)
        self.b_rematch.clicked.connect(lambda: self.start(use_fit_prior=True))
        self.b_apply.clicked.connect(self.apply)
        self.b_close.clicked.connect(self.close)
        self._set_busy(False)

    def _set_busy(self, busy):
        for b in (self.b_browser, self.b_refit, self.b_rematch, self.b_apply, self.b_match):
            b.setEnabled(not busy)
        for w in (self.scope, self.source, self.n_spots):
            w.setEnabled(not busy)
        self.b_rematch.setEnabled(not busy and bool(self.fits))
        self.b_apply.setEnabled(not busy and bool(self.fits))

    # ---------------- spots ----------------
    def _export_wh(self):
        return int(self.ref_geo.raw_w), int(self.ref_geo.raw_h)

    def _raw_wh(self, geo):
        return int(geo.ref_w or geo.raw_w), int(geo.ref_h or geo.raw_h)

    def _gsd(self):
        g = self.ref_geo
        if not g.located:
            return None
        W, H = self._export_wh()
        try:
            return float(min(CF.gsd_m(g, W / 2.0, H / 2.0)))
        except Exception:
            return None

    def _overlap_region(self, shape):
        """(H, W) bool on the reference preview: where every target has ground."""
        ph, pw = shape[:2]
        W, H = self._export_wh()
        sx, sy = W / float(pw), H / float(ph)
        masks = []
        for t in self.targets:
            try:
                tg = CF.frame_for(t["fp"], ax=t["ax"])
                if tg is None or not tg.located or not self.ref_geo.located:
                    continue
                c = np.array([[0, 0], [tg.raw_w, 0], [tg.raw_w, tg.raw_h], [0, tg.raw_h]], np.float64)
                q = CF.map_px(tg, self.ref_geo, c) / (sx, sy)
                m = np.zeros((ph, pw), np.uint8)
                cv2.fillPoly(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 1)
                masks.append(m.astype(bool))
            except Exception as e:
                logging.debug(f"[view alignment] footprint of {t['fp']} failed: {e}")
        if not masks:
            return None
        inter = np.logical_and.reduce(masks)
        if inter.mean() > 0.05:
            return inter
        return np.logical_or.reduce(masks)

    def _polygon_spots(self, n):
        """Spots from the polygons drawn on the reference: the canopy finder's
        own use (a crown matched in another image), up to `n`, spread out."""
        recs = self.pt._view_align_polygon_records(self.ref_fp)
        items = []
        for g, rec in recs:
            try:
                pts = rec.get("points") or []
                r = {"points": [tuple(map(float, p[:2])) for p in pts],
                     "coord_space": rec.get("coord_space", "image"),
                     "image_ref_size": rec.get("image_ref_size"), "type": "polygon"}
                c = np.asarray(self.ref_geo.from_record(r), np.float64).reshape(-1, 2).mean(0)
            except Exception:
                continue
            if np.all(np.isfinite(c)):
                items.append((g, r, c))
        if not items:
            return []
        C = np.array([c for _g, _r, c in items])
        pick = [int(np.argmin(((C - C.mean(0)) ** 2).sum(1)))]
        while len(pick) < min(n, len(items)):
            d = np.sqrt(((C[:, None] - C[pick][None]) ** 2).sum(-1)).min(1)
            d[pick] = -1
            pick.append(int(np.argmax(d)))
        return [{"group": f"spot {k + 1}: {items[i][0]}", "center": tuple(items[i][2]), "rec": items[i][1]}
                for k, i in enumerate(pick)]

    def _make_spots(self):
        W, H = self._export_wh()
        side = VA.spot_side_px(W, H, self._gsd())
        n = int(self.n_spots.value())
        centers = []
        if self.source.currentData() == "polygons" and self.spot_scene_pts is None:
            return self._polygon_spots(n)
        if self.spot_scene_pts is not None and len(self.spot_scene_pts):
            rec_pts = self.pt._scene_to_image_xy(self.ref_viewer, self.spot_scene_pts)
            bw, bh = self.ref_geo.basis
            centers = list(self.ref_geo.from_record({"points": rec_pts.tolist(), "coord_space": "image",
                                                     "image_ref_size": {"w": bw, "h": bh}}))
        else:
            prev = getattr(self.ref_viewer.image_data, "image", None)
            if prev is None:
                return []
            ph, pw = prev.shape[:2]
            region = self._overlap_region(prev.shape)
            if n > 8:
                # many spots: smaller squares, so neighbours stay apart
                side = max(VA.SPOT_MIN_PX, min(side, 0.5 * min(W, H) / (n ** 0.5)))
            cp = VA.pick_spots(prev, None, n=n, side=side * pw / float(W), region=region)
            centers = [(x * W / float(pw), y * H / float(ph)) for x, y in cp]
        spots = []
        for k, (cx, cy) in enumerate(centers):
            h = side / 2.0
            sq = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])
            sq[:, 0] = np.clip(sq[:, 0], 0, W - 1)
            sq[:, 1] = np.clip(sq[:, 1], 0, H - 1)
            pts, ref, _inside = self.ref_geo.to_record(sq)
            spots.append({"group": f"spot {k + 1}", "center": (cx, cy),
                          "rec": {"points": pts, "coord_space": "image", "image_ref_size": ref,
                                  "type": "polygon"}})
        return spots

    # ---------------- matching ----------------
    def _cfg(self):
        cfg = {"method": self.method.currentData(), "search_radius_m": float(self.radius.value()),
               "angle_range": 5.0, "search_radius_pct": 25.0, "angle_range_pixel": 5.0,
               "buffer_ratio": 0.1, "min_buffer": 10, "context_ring_ratio": 0.3,
               "max_context_ring": 1500, "min_ncc": 0.2}
        cfg.update(self.extra_cfg)
        return cfg

    def _fit_prior(self, fp, center=None):
        """export ref -> export target, from the current fit (or None): the
        fit's local affine around `center` (export px of the reference) when
        given, else over the whole frame."""
        f = self.fits.get(fp)
        t = next((x for x in self.targets if x["fp"] == fp), None)
        if not f or t is None:
            return None
        tg = CF.frame_for(fp, ax=t["ax"])
        W, H = self._export_wh()
        if center is not None:
            d = max(8.0, 0.05 * min(W, H))
            cx, cy = center
            P = np.array([[cx, cy], [cx + d, cy], [cx, cy + d]], np.float64)
        else:
            P = np.array([[0.0, 0.0], [W, 0.0], [0.0, H]])
        rw, rh = self._raw_wh(self.ref_geo)
        tw, th = self._raw_wh(tg)
        raw = VA.export_to_raw(P, self.ref_ax, rw, rh)
        Q = VA.raw_to_export(VA.warp_forward(f.get("warp") or {"M": f["M"]}, raw), t["ax"], tw, th)
        return _affine_from_3(P, Q)

    def _on_match(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.spots, self.fits = [], {}
        if not self._explicit_targets:
            self._set_targets(self._scope_targets())
        if not self.targets:
            self.info.setText("There is no other image to align.")
            return
        if self.source.currentData() == "click":
            self.spot_scene_pts = None
            self._start_clicks()
            return
        if self.source.currentData() != "click":
            self.spot_scene_pts = None
        if self.source.currentData() == "polygons" and not self._n_polygons():
            self.info.setText("No polygon is drawn on this image: draw a few crowns spread over it, "
                              "or use automatic spots.")
            return
        self.start()

    def _start_clicks(self):
        def _done(pts):
            self._clicker = None
            self.spot_scene_pts = np.asarray(pts, np.float64)
            self.spots = []
            self.show(); self.raise_()
            self.start()
        if self._clicker is not None:
            self._clicker._cleanup()
        self._clicker = ClickSpotsMode(self.ref_viewer, on_done=_done, n=int(self.n_spots.value()))
        self.info.setText(f"Click up to {int(self.n_spots.value())} spots on "
                          f"<b>{os.path.basename(self.ref_fp)}</b> (Enter when done).")

    def start(self, use_fit_prior=False):
        if self.worker is not None and self.worker.isRunning():
            return
        prior = {}
        if use_fit_prior:
            for t in self.targets:
                if self._fit_prior(t["fp"]) is not None:
                    prior[t["fp"]] = True
        if not self.targets and not self._explicit_targets:
            self._set_targets(self._scope_targets())
        if not self.spots:
            try:
                self.spots = self._make_spots()
            except Exception as e:
                logging.exception("[view alignment] spot picking failed")
                self.info.setText(f"Could not pick spots: {e}")
                return
        if len(self.spots) < 2:
            self.info.setText("Not enough textured spots with data on this image to align "
                              "(try 'Align with spots I click').")
            return
        self.results, self.ctxs = [], {}
        jobs = []
        for s in self.spots:
            ts = []
            for t in self.targets:
                tt = dict(t)
                if t["fp"] in prior:
                    A = self._fit_prior(t["fp"], center=s.get("center"))
                    if A is not None:
                        tt["cfg"] = {"prior_affine": A.tolist()}
                ts.append(tt)
            jobs.append({"group": s["group"], "rec": s["rec"], "targets": ts})
        self.info.setText(f"Reference: <b>{os.path.basename(self.ref_fp)}</b> — matching "
                          f"{len(self.spots)} spots in {len(self.targets)} image(s)"
                          + (" from the current fit" if prior else "") + "…")
        self.progress.setRange(0, max(1, len(jobs) * len(self.targets)))
        self.progress.setValue(0)
        self.worker = D.CanopyFindWorker(self.ref_fp, jobs, self.ref_ax, self._cfg(), scan=None, parent=self)
        self.worker.source_ready.connect(self._on_source)
        self.worker.result.connect(self._on_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(lambda msg: logging.warning(f"[view alignment] {msg}"))
        self.worker.finished.connect(self._on_finished)
        self._set_busy(True)
        self.worker.start()

    def _on_source(self, ctx):
        self.ctxs[ctx.group] = ctx

    def _on_progress(self, k, n, msg):
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(k)
        self.progress.setFormat(f"{k} / {n}  {msg}")

    def _on_result(self, r, t):
        t = dict(t)
        t["_orig"] = None if r.raw_pts is None else np.array(r.raw_pts, np.float64, copy=True)
        self.results.append([r, t, ""])

    def _on_finished(self):
        self._set_busy(False)
        self.progress.setFormat("done")
        self.refit()
        self._predict_missing()
        self.refit()
        if self.browser is not None:
            try:
                self.browser.set_entries(self.ctxs, self.results)
            except Exception:
                pass

    # ---------------- fit ----------------
    def _row(self, group, fp):
        for i, (r, t, _s) in enumerate(self.results):
            if t.get("group") == group and D._norm(t["fp"]) == D._norm(fp):
                return i
        return None

    def _spot_pairs(self, t):
        """Matched spots of one target, in raw file pixels."""
        tg = CF.frame_for(t["fp"], ax=t["ax"])
        rw, rh = self._raw_wh(self.ref_geo)
        tw, th = self._raw_wh(tg)
        out = []
        for s in self.spots:
            ctx = self.ctxs.get(s["group"])
            i = self._row(s["group"], t["fp"])
            entry = {"group": s["group"], "src": None, "dst": None, "status": "missing", "score": None}
            if ctx is None or i is None:
                out.append(entry)
                continue
            r, tt, _st = self.results[i]
            entry["status"], entry["score"] = r.status, r.score
            entry["src"] = VA.export_to_raw(ctx.raw_pts, self.ref_ax, rw, rh)
            edited = (r.raw_pts is not None and tt.get("_orig") is not None
                      and not np.allclose(np.asarray(r.raw_pts), tt["_orig"], atol=0.25))
            usable = (not tt.get("rejected") and r.raw_pts is not None
                      and (r.status in ("found", "low_score") or edited)
                      and not (tt.get("predicted") and not edited))
            if usable:
                entry["dst"] = VA.export_to_raw(r.raw_pts, t["ax"], tw, th)
                if edited:
                    entry["weight"], entry["status"] = 2.0, "edited"
                elif r.status == "found":
                    entry["weight"] = max(0.3, float(r.score or 0.3))
                else:
                    entry["weight"] = 0.3
            out.append(entry)
        return out, tg

    def refit(self):
        self.fits = {}
        self.table.setRowCount(len(self.targets))
        for row, t in enumerate(self.targets):
            pairs, tg = self._spot_pairs(t)
            fit = VA.fit_alignment(pairs, mode=self.fit_mode.currentData(), min_spots=2)
            name = QtWidgets.QTableWidgetItem(os.path.basename(t["fp"]))
            name.setToolTip(t["fp"])
            self.table.setItem(row, 0, name)
            n_ok = sum(1 for p in pairs if p["dst"] is not None)
            if fit is None:
                for c, txt in enumerate(["0 / %d" % len(pairs), "", "", "", "", "not enough spots found"], 1):
                    self.table.setItem(row, c, QtWidgets.QTableWidgetItem(txt))
                continue
            s, ang, _tx, _ty = VA.similarity_params(fit["M"])
            rms_m = loo_m = None
            try:
                base = tg.base or tg
                if base.located:
                    tw, th = self._raw_wh(tg)
                    g = float(min(CF.gsd_m(base, tw / 2.0, th / 2.0)))
                    rms_m = fit["rms_px"] * g
                    loo_m = fit["loo_px"] * g if np.isfinite(fit.get("loo_px", np.nan)) else None
            except Exception:
                rms_m = loo_m = None
            fit["rms_m"] = rms_m
            fit["loo_m"] = loo_m
            fit["pairs"] = pairs
            self.fits[t["fp"]] = fit
            warn = []
            if fit["n_used"] < 3:
                warn.append("only %d spots: check it" % fit["n_used"])
            if fit["reflected"]:
                warn.append("mirrored?")
            loo = fit.get("loo_px", float("nan"))
            vals = [f"{fit['n_used']} / {len(pairs)}  ({n_ok} matched)",
                    f"{fit['rms_px']:.1f} px" + (f" ({rms_m:.2f} m)" if rms_m is not None else ""),
                    (f"{loo:.1f} px" + (f" ({loo_m:.2f} m)" if loo_m is not None else ""))
                    if np.isfinite(loo) else "",
                    f"{s:.4f}", f"{ang:+.2f}°", "; ".join(warn) or fit.get("mode", "ok")]
            for c, txt in enumerate(vals, 1):
                self.table.setItem(row, c, QtWidgets.QTableWidgetItem(txt))
        self._set_busy(False)

    def _predict_missing(self):
        """Spots not found get an outline where the fit (else the
        georeference) puts them, so they can be dragged into place in the
        browser. They count in the fit only once moved."""
        for t in self.targets:
            tg = CF.frame_for(t["fp"], ax=t["ax"])
            rw, rh = self._raw_wh(self.ref_geo)
            tw, th = self._raw_wh(tg)
            fit = self.fits.get(t["fp"])
            for s in self.spots:
                ctx = self.ctxs.get(s["group"])
                i = self._row(s["group"], t["fp"])
                if ctx is None or i is None:
                    continue
                r, tt, _st = self.results[i]
                if r.has_polygon:
                    continue
                pred = None
                if fit is not None:
                    raw = VA.warp_forward(fit.get("warp") or {"M": fit["M"]},
                                          VA.export_to_raw(ctx.raw_pts, self.ref_ax, rw, rh))
                    pred = VA.raw_to_export(raw, t["ax"], tw, th)
                elif r.geo_pts is not None:
                    pred = np.asarray(r.geo_pts, np.float64)
                if pred is None:
                    continue
                r.status = "geo"
                r.raw_pts = pred
                x0, y0 = np.floor(pred.min(axis=0))
                x1, y1 = np.ceil(pred.max(axis=0))
                r.square = (int(max(0, x0)), int(max(0, y0)), int(min(tg.raw_w, x1)), int(min(tg.raw_h, y1)))
                r.extra["frame"] = tg
                tt["predicted"] = True
                tt["_orig"] = np.array(pred, copy=True)

    # ---------------- browser hooks ----------------
    def open_browser(self):
        if not self.ctxs:
            QtWidgets.QMessageBox.information(self, "Align viewers", "No spots were matched yet.")
            return
        from .canopy_browser import CanopyBrowser
        if self.browser is None:
            self.browser = CanopyBrowser(self)
            self.browser.setWindowTitle("Alignment spots")
        self.browser.set_entries(self.ctxs, self.results)
        self.browser.show()
        self.browser.raise_()

    def _update_row(self, i):
        pass

    def _drop_temp_outline(self, row):
        pass

    def save_all(self, ask=True):
        pass

    def export(self, *a, **k):
        pass

    # ---------------- apply ----------------
    def apply(self):
        if self.browser is not None and self.browser.isVisible():
            self.refit()
        if not self.fits:
            return
        targets = {}
        for fp, f in self.fits.items():
            spots = []
            for p in f.get("pairs") or []:
                spots.append({"group": p["group"], "status": p["status"], "score": p["score"],
                              "src": None if p["src"] is None else np.asarray(p["src"]).round(3).tolist(),
                              "dst": None if p["dst"] is None else np.asarray(p["dst"]).round(3).tolist()})
            used = f.get("used") or []
            for k, sp in enumerate(spots):
                sp["used"] = bool(used[k]) if k < len(used) else False
            loo = f.get("loo_px", float("nan"))
            targets[fp] = {"M": np.asarray(f["M"]).tolist(), "local": f.get("local"),
                           "mode": f.get("mode", "global"), "rms_px": float(f["rms_px"]),
                           "rms_m": f.get("rms_m"), "loo_px": float(loo) if np.isfinite(loo) else None,
                           "loo_m": f.get("loo_m"), "n_used": int(f["n_used"]), "spots": spots,
                           "method": self.method.currentData(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        missing = [os.path.basename(t["fp"]) for t in self.targets if t["fp"] not in self.fits]
        root = self.pt.get_root_by_filepath(self.ref_fp) if hasattr(self.pt, "get_root_by_filepath") else None
        if not self._explicit_targets and self.scope.currentData() == "all":
            root = "all roots"
        self.pt.apply_view_alignment(self.ref_fp, targets, root=root)
        msg = f"Aligned {len(targets)} image(s) to {os.path.basename(self.ref_fp)}."
        if missing:
            msg += "\nNot aligned (too few spots): " + ", ".join(missing)
        self.info.setText(msg.replace("\n", "<br>"))

    def closeEvent(self, ev):
        if self._clicker is not None:
            self._clicker._cleanup()
            self._clicker = None
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        super().closeEvent(ev)


class ClickSpotsMode(QtCore.QObject):
    """Collect spots clicked on a viewer: left click adds one (a red cross),
    Backspace removes the last, Enter accepts (3 or more), Esc cancels. The
    5th click accepts by itself. `on_done((N, 2) scene points)`."""

    def __init__(self, viewer, on_done, n=5):
        super().__init__(viewer)
        self.viewer, self.on_done, self.n = viewer, on_done, int(n)
        self.pts, self.items = [], []
        self.label = QtWidgets.QLabel(viewer.viewport())
        self.label.setStyleSheet("QLabel { background: rgba(0,0,0,170); color: white; padding: 4px 8px;"
                                 " border-radius: 4px; font-size: 11px; }")
        self._update_label()
        self.label.move(10, 10)
        self.label.show()
        viewer.viewport().installEventFilter(self)
        viewer.installEventFilter(self)
        viewer.setFocus()

    def _update_label(self):
        self.label.setText(f"Align viewers: click {self.n} spots with clear structure "
                           f"({len(self.pts)} / {self.n}).  Enter: done (3+)  Backspace: undo  Esc: cancel")
        self.label.adjustSize()

    def _cross(self, p):
        path = QtGui.QPainterPath()
        r = 9.0
        path.moveTo(-r, 0); path.lineTo(r, 0)
        path.moveTo(0, -r); path.lineTo(0, r)
        it = QtWidgets.QGraphicsPathItem(path)
        pen = QtGui.QPen(QtGui.QColor(255, 40, 40))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        it.setPen(pen)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        it.setZValue(1e6)
        it.setPos(p)
        self.viewer.scene().addItem(it)
        return it

    def eventFilter(self, obj, ev):
        t = ev.type()
        if obj is self.viewer.viewport() and t in (QtCore.QEvent.MouseButtonPress,
                                                   QtCore.QEvent.MouseButtonRelease) \
                and ev.button() == QtCore.Qt.LeftButton:
            if t == QtCore.QEvent.MouseButtonPress:
                p = self.viewer.mapToScene(ev.pos())
                self.pts.append((p.x(), p.y()))
                self.items.append(self._cross(p))
                self._update_label()
                if len(self.pts) >= self.n:
                    QtCore.QTimer.singleShot(0, self._finish)
            return True
        if t == QtCore.QEvent.KeyPress:
            k = ev.key()
            if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if len(self.pts) >= 3:
                    self._finish()
                return True
            if k == QtCore.Qt.Key_Backspace:
                if self.pts:
                    self.pts.pop()
                    it = self.items.pop()
                    self.viewer.scene().removeItem(it)
                    self._update_label()
                return True
            if k == QtCore.Qt.Key_Escape:
                self._cleanup()
                return True
        return False

    def _cleanup(self):
        try:
            self.viewer.viewport().removeEventFilter(self)
            self.viewer.removeEventFilter(self)
            for it in self.items:
                if it.scene() is not None:
                    it.scene().removeItem(it)
            self.items = []
            self.label.hide()
            self.label.deleteLater()
        except Exception:
            pass

    def _finish(self):
        pts = np.asarray(self.pts, np.float64)
        self._cleanup()
        if len(pts) >= 2 and callable(self.on_done):
            self.on_done(pts)
