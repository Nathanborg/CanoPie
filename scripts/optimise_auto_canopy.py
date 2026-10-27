"""Optimise Auto Canopy (canopie/canopy_auto.py) against hand-drawn crowns of a CanoPie project.

    python scripts/optimise_auto_canopy.py build  --project C:/BCI_canopie --out <dir>
    python scripts/optimise_auto_canopy.py search --out <dir> [--configs 150] [--workers 10]
    python scripts/optimise_auto_canopy.py test   --out <dir> [--params best.json]

build   finds blocks (3000 x 3000 image px) where the project's polygons cover >= 60 % of the
        area -- neighbourhoods of 50-80 crowns annotated nearly completely -- picks 8 spread over
        the image (5 to tune on, 3 held out), and caches each as the viewer shows it (0.5-99.5 %
        stretch, at half resolution) with its crowns.
search  random search over canopy_auto.SEARCH_SPACE with successive halving (2 tuning blocks ->
        all 5 for the best quarter), then one parameter at a time around the best.  Every
        configuration's score is printed as soon as it is known (and logged); a run is cut off
        after --cap seconds -- too slow to use interactively, so it scores what it found by then.
        Writes best.json.
test    scores the defaults and best.json on the held-out blocks, with overlays.

The score is canopy_auto.score_crowns (crown F1 at IoU >= 0.5, the in-app Learn Crowns uses the
same), with a margin of one crown radius around each block.
"""
import argparse, glob, json, os, random, sys, time
from dataclasses import fields
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BLOCK = 3000
SCALE = 0.5            # cache at half resolution (BCI crowns: radius ~60 px there)
CAP = 60.0


# ------------------------------------------------------------------------------ build
def build(project, out, n_tune=5, n_test=3):
    import cv2
    from canopie import raster_reader as RR
    from canopie.canopy_auto import _poly_area
    os.makedirs(out, exist_ok=True)
    ov = json.load(open(os.path.join(project, "polygons", "_overview.json"), encoding="utf-8"))["entries"]
    ov = [e for e in ov if e.get("t") == "polygon" and len(e.get("p", [])) >= 3]
    img_path = ov[0]["f"]
    C = np.array([np.mean(e["p"], 0) for e in ov]); A = np.array([_poly_area(np.array(e["p"], float)) for e in ov])
    W0, H0 = ov[0]["rs"]["w"], ov[0]["rs"]["h"]
    cov = {}
    for (x, y), a in zip(C, A):
        k = (int(x // BLOCK), int(y // BLOCK)); cov[k] = cov.get(k, 0.0) + a / BLOCK ** 2
    dense = sorted([k for k, v in cov.items() if v >= 0.6], key=lambda k: -cov[k])
    pick = [dense[0]]                                            # spread over the image: farthest first
    while len(pick) < min(n_tune + n_test, len(dense)):
        d = [min(np.hypot(k[0] - q[0], k[1] - q[1]) for q in pick) for k in dense]
        pick.append(dense[int(np.argmax(d))])
    w = _BlockWriter(project, ov, C, W0, H0)
    for i, (bx, by) in enumerate(pick):
        role = "tune" if i < n_tune else "test"
        n = w.write(os.path.join(out, f"{role}_{i}.npz"), bx, by)
        print(f"{role} block {i}: x{bx * BLOCK} y{by * BLOCK}  cover {cov[(bx, by)]:.2f}  {n} crowns", flush=True)


class _BlockWriter:
    """Caches one BLOCK x BLOCK block of the project image as the viewer shows it, with its crowns."""

    def __init__(self, project, ov, C, W0, H0, image=None, polys=None):
        from canopie import raster_reader as RR
        self.project, self.ov, self.C, self.W0, self.H0 = project, ov, C, W0, H0
        self.polys = polys                                       # given polygons (px) instead of project files
        self.img_path = image or ov[0]["f"]
        self.r = RR.open_reader(self.img_path)
        pr = self.r.profile
        small = np.asarray(self.r.read_window(0, 0, 10 ** 9, 10 ** 9, level=int(pr.n_levels or 1) - 1))[..., :3]
        self.lo, self.hi = np.percentile(small[small > 0], [0.5, 99.5])
        self.stem = os.path.splitext(os.path.basename(self.img_path))[0]

    def write(self, path, bx, by, shift=(0.0, 0.0)):
        disp, gts, origin = self.render(bx, by)
        gts = [(np.asarray(q) + shift).tolist() for q in gts]
        np.savez_compressed(path, img=disp, gt=np.array(gts, dtype=object), origin=origin, scale=SCALE,
                            shift=np.asarray(shift, float))
        return len(gts)

    def render(self, bx, by):
        """(the block as displayed at SCALE, its crowns in block px (pixel-corner convention), origin)."""
        import cv2
        x0, y0 = bx * BLOCK, by * BLOCK
        x1, y1 = min(self.W0, x0 + BLOCK), min(self.H0, y0 + BLOCK)
        win = np.asarray(self.r.read_window(x0, y0, x1, y1))[..., :3].astype(np.float32)
        disp = np.clip((win - self.lo) * (255.0 / (self.hi - self.lo)), 0, 255).astype(np.uint8)
        disp = cv2.resize(disp, (int(round((x1 - x0) * SCALE)), int(round((y1 - y0) * SCALE))), interpolation=cv2.INTER_AREA)
        gts = []
        for i, (e, (cx, cy)) in enumerate(zip(self.ov, self.C)):
            if x0 - 400 <= cx < x1 + 400 and y0 - 400 <= cy < y1 + 400:
                if self.polys is not None:
                    pts = np.asarray(self.polys[i], float)
                else:
                    fp = os.path.join(self.project, "polygons", f"{e['g']}_{self.stem}_polygons.json")
                    pts = np.array(json.load(open(fp, encoding="utf-8"))["points"], float) if os.path.exists(fp) else np.array(e["p"], float)
                gts.append(((pts - (x0, y0)) * SCALE).tolist())          # pixel-corner convention
        return disp, gts, np.array([x0, y0])


# ------------------------------------------------------------------------------ aligning a crown map
def edge_shift(img, gts, rng=90, step=3):
    """(score, dx, dy): the shift (block px) that puts the crown outlines on the darkest pixels -- the
    shadow gaps of a closed canopy run between crowns, so an aligned crown map has its edges on them."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.0)
    dark = 1.0 - (g - g.min()) / max(float(np.ptp(g)), 1e-6)
    H, W = dark.shape
    edge = np.zeros((H + 2 * rng, W + 2 * rng), np.uint8)
    for q in gts:
        cv2.polylines(edge, [np.round(np.asarray(q) + rng).astype(np.int32)], True, 1, 3)
    best = (-1.0, 0, 0)
    for dy in range(-rng, rng + 1, step):
        for dx in range(-rng, rng + 1, step):
            e = edge[rng - dy:rng - dy + H, rng - dx:rng - dx + W] > 0
            if e.sum() >= 100:
                s = float(dark[e].mean())
                if s > best[0]:
                    best = (s, dx, dy)
    return best


def _given_crowns_f1(img, gts):
    """Auto Canopy's crown F1 when given the crowns' true centres and sizes: high (~0.8) only where the
    crown map lines up with the image -- so it checks a shift."""
    from dataclasses import replace
    from canopie.canopy_auto import AutoParams, delineate_all, score_crowns, _poly_area
    g = [np.asarray(q, float) - 0.5 for q in gts]
    radii = np.array([np.sqrt(_poly_area(q) / np.pi) for q in g])
    R = float(np.median(radii))
    res = delineate_all(img, replace(AutoParams(), crown_radius_px=R), seeds=np.c_[[q.mean(0) for q in g], radii])
    return score_crowns(img.shape, [c.polygon for c in res.crowns], g, margin=R)["f1"]


def _align_job(a):
    """One block: its shift onto the image (coarse, then +-6 px at 1 px) and the check it is right."""
    import cv2
    cv2.setNumThreads(1)
    key, disp, gts = a
    gts = [np.asarray(q, float) for q in gts]
    _, dx, dy = edge_shift(disp, gts)
    _, ddx, ddy = edge_shift(disp, [q + (dx, dy) for q in gts], rng=6, step=1)
    dx, dy = dx + ddx, dy + ddy
    return key, (float(dx), float(dy)), _given_crowns_f1(disp, gts), _given_crowns_f1(disp, [q + (dx, dy) for q in gts])


def build_shp(shp, image, out, n_tune=5, n_test=3, min_cover=0.5, auto_align=False, workers=8,
              min_f1=0.65, min_gain=0.15):
    """Blocks from a reference crown shapefile (e.g. a crown census) on a georeferenced image: the
    polygons go to image px through the image's GeoTIFF tie point / pixel scale (same CRS assumed).
    Every crown counts (a complete census needs no minimum cover beyond ``min_cover`` of the block):
    n_test held-out blocks spread over the area, n_tune next, every other block for training.

    ``auto_align``: a crown map drawn on another orthomosaic is offset from this one by metres, and not
    by the same amount everywhere.  Per block, the shift that puts the crown edges on the shadow gaps
    (edge_shift) is checked by Auto Canopy given the true centres and sizes: the block is kept, shifted,
    only if that rises to >= min_f1 and by >= min_gain (or is already >= min_f1 unshifted)."""
    import shapefile, tifffile
    from canopie.canopy_auto import _poly_area
    with tifffile.TiffFile(image) as t:
        tags = {k: v.value for k, v in t.pages[0].tags.items()}
        H0, W0 = t.pages[0].shape[:2]
    sx, sy = tags[33550][0], tags[33550][1]
    tx, ty = tags[33922][3], tags[33922][4]
    r = shapefile.Reader(shp)
    polys, skipped = [], 0
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        if str(rec.as_dict().get("plot_statu", "")).strip().lower().startswith("duplicate"):
            skipped += 1
            continue
        pts = np.asarray(sh.points, np.float64)
        parts = list(sh.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        rings = [q for q in rings if len(q) >= 3]
        if not rings:                                            # an empty shape
            skipped += 1
            continue
        ring = max(rings, key=lambda q: _poly_area(q))          # the outer ring of the largest part
        if len(ring) < 3:
            continue
        polys.append(np.c_[(ring[:, 0] - tx) / sx, (ty - ring[:, 1]) / sy])   # px, pixel-corner convention
    C = np.array([q.mean(0) for q in polys]); A = np.array([_poly_area(q) for q in polys])
    cov = {}
    for (x, y), a in zip(C, A):
        k = (int(x // BLOCK), int(y // BLOCK)); cov[k] = cov.get(k, 0.0) + a / BLOCK ** 2
    full = sorted([k for k, v in cov.items() if v >= min_cover], key=lambda k: -cov[k])
    ov = [{"p": q} for q in polys]
    w = _BlockWriter(None, ov, C, W0, H0, image=image, polys=polys)
    shifts = {k: (0.0, 0.0) for k in full}
    if auto_align:
        renders = {k: w.render(*k) for k in full}
        with Pool(workers) as pool:
            res = pool.map(_align_job, [(k, renders[k][0], renders[k][1]) for k in full])
        kept = []
        for k, (dx, dy), f0, f1 in res:
            ok_shift = f1 >= min_f1 and f1 - f0 >= min_gain
            ok_as_is = f0 >= min_f1 and f0 >= f1
            verdict = "shifted" if ok_shift and not ok_as_is else ("as is" if ok_as_is else "left out")
            print(f"  block x{k[0] * BLOCK} y{k[1] * BLOCK}: shift ({dx / SCALE:+.0f}, {dy / SCALE:+.0f}) full px, "
                  f"given-crowns F1 {f0:.2f} -> {f1:.2f}: {verdict}", flush=True)
            if verdict == "shifted":
                shifts[k] = (dx, dy)
            if verdict != "left out":
                kept.append(k)
        full = [k for k in full if k in kept]
    test = [full[0]]                                             # held out: spread over the area
    while len(test) < n_test:
        d = [min(np.hypot(k[0] - q[0], k[1] - q[1]) for q in test) for k in full]
        test.append(full[int(np.argmax(d))])
    rest = [k for k in full if k not in test]
    tune = []
    while len(tune) < n_tune and rest:
        d = [min(np.hypot(k[0] - q[0], k[1] - q[1]) for q in test + tune) for k in rest]
        tune.append(rest.pop(int(np.argmax(d))))
    os.makedirs(out, exist_ok=True)
    for role, keys in (("test", test), ("tune", tune), ("train", rest)):
        for i, (bx, by) in enumerate(keys):
            m = w.write(os.path.join(out, f"{role}_{i}.npz"), bx, by, shift=shifts[(bx, by)])
            print(f"{role} block {i}: x{bx * BLOCK} y{by * BLOCK}  cover {cov[(bx, by)]:.2f}  {m} crowns", flush=True)
    print(f"{len(polys)} crowns used ({skipped} duplicate tags or empty shapes left out)")


def build_train(project, out, n=12, min_cover=0.3):
    """Extra blocks for fitting the crown-size model: partly annotated is enough (every annotated crown
    is a sample), never one of the existing blocks or next to a held-out one."""
    from canopie.canopy_auto import _poly_area
    ov = json.load(open(os.path.join(project, "polygons", "_overview.json"), encoding="utf-8"))["entries"]
    ov = [e for e in ov if e.get("t") == "polygon" and len(e.get("p", [])) >= 3]
    C = np.array([np.mean(e["p"], 0) for e in ov]); A = np.array([_poly_area(np.array(e["p"], float)) for e in ov])
    W0, H0 = ov[0]["rs"]["w"], ov[0]["rs"]["h"]
    cov = {}
    for (x, y), a in zip(C, A):
        k = (int(x // BLOCK), int(y // BLOCK)); cov[k] = cov.get(k, 0.0) + a / BLOCK ** 2
    used, held = set(), set()
    for f in glob.glob(os.path.join(out, "*_*.npz")):
        o = np.load(f, allow_pickle=True)["origin"]
        k = (int(o[0]) // BLOCK, int(o[1]) // BLOCK)
        used.add(k)
        if os.path.basename(f).startswith("test_"):
            held.add(k)
    ok = [k for k, v in cov.items() if v >= min_cover and k not in used
          and all(max(abs(k[0] - h[0]), abs(k[1] - h[1])) > 1 for h in held)]
    ok.sort(key=lambda k: -cov[k])
    w = _BlockWriter(project, ov, C, W0, H0)
    for i, (bx, by) in enumerate(ok[:n]):
        m = w.write(os.path.join(out, f"train_{i}.npz"), bx, by)
        print(f"train block {i}: x{bx * BLOCK} y{by * BLOCK}  cover {cov[(bx, by)]:.2f}  {m} crowns", flush=True)


# ------------------------------------------------------------------------------ score one run
_cache = {}


def run_job(job):
    idx, path, cfg = job
    import cv2
    cv2.setNumThreads(1)
    from canopie.canopy_auto import AutoParams, _poly_area, delineate_all, score_crowns
    if path not in _cache:
        z = np.load(path, allow_pickle=True)
        gts = [np.array(g, float) - 0.5 for g in z["gt"]]               # -> pixel centres at integers
        _cache[path] = (z["img"], gts, float(np.median([np.sqrt(_poly_area(g) / np.pi) for g in gts])))
    img, gts, Rg = _cache[path]
    if os.environ.get("AUTO_CANOPY_KNOWN_SIZE") == "1":            # as in the app: size from the user's crowns
        cfg = dict(cfg, crown_radius_px=Rg)
    t0 = time.time()
    res = delineate_all(img, AutoParams(**cfg), cancel=lambda: time.time() - t0 > CAP)
    s = score_crowns(img.shape, [c.polygon for c in res.crowns], gts, margin=Rg)
    s.update(secs=time.time() - t0, R_est=res.crown_radius_px, R_gt=Rg, cut=float("cancelled" in res.note))
    return idx, s


def evaluate(pool, blocks, cfgs, say=None, tag=""):
    """Mean score per config; each printed as soon as all its blocks are done."""
    jobs = [(i, b, c) for i, c in enumerate(cfgs) for b in blocks]
    got = {i: [] for i in range(len(cfgs))}
    out = [None] * len(cfgs)
    for i, s in pool.imap_unordered(run_job, jobs):
        got[i].append(s)
        if len(got[i]) == len(blocks):
            out[i] = {k: float(np.mean([r[k] for r in got[i]])) for k in got[i][0]}
            if say is not None:
                say(line(f"{tag}{i}", cfgs[i], out[i]))
    return out


def line(tag, cfg, s):
    return (f"{tag:22s} crownF1 {s['f1']:.3f}  mIoU {s['miou']:.3f}  pixF1 {s['pix']:.3f}  pred {s['n_pred']:.0f}/gt {s['n_gt']:.0f}"
            f"  R est/gt {s['R_est']:.0f}/{s['R_gt']:.0f}  {s['secs']:.0f}s{' CUT' if s.get('cut') else ''}   {json.dumps(cfg)}")


# ------------------------------------------------------------------------------ search / test
def _defaults():
    from canopie.canopy_auto import AutoParams, params_dict
    return params_dict(AutoParams())


def search(out, n_configs, workers, seed=0, start=None):
    from canopie.canopy_auto import SEARCH_SPACE, sample_params
    tune = sorted(glob.glob(os.path.join(out, "tune_*.npz")))
    rng = random.Random(seed)
    first = dict(_defaults(), **(json.load(open(start)) if start else {}))
    cfgs = [first] + [sample_params(rng) for _ in range(n_configs - 1)]
    log = open(os.path.join(out, "search.log"), "a", encoding="utf-8")

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    t0 = time.time()
    with Pool(workers) as pool:
        say(f"== stage A: {len(cfgs)} configs on 2 tuning blocks (A0 = defaults)")
        sa = evaluate(pool, tune[:2], cfgs, say, "A")
        top = [c for _, c in sorted(zip([s["f1"] for s in sa], cfgs), key=lambda t: -t[0])[:max(4, len(cfgs) // 4)]]
        say(f"== stage B: best {len(top)} on all {len(tune)} tuning blocks  ({time.time() - t0:.0f}s)")
        sb = evaluate(pool, tune, top, say, "B")
        best, bs = max(zip(top, sb), key=lambda t: t[1]["f1"])
        say(f"== stage C: one parameter at a time around the best (F1 {bs['f1']:.3f})  ({time.time() - t0:.0f}s)")
        for k, s in SEARCH_SPACE.items():
            if s[0] == "choice":
                opts = [v for v in s[1] if v != best[k]]
            elif s[0] == "int":
                opts = sorted({max(s[1], best[k] - 2), min(s[2], best[k] + 2)} - {best[k]})
            else:
                opts = [round(best[k] * m, 4) for m in (0.75, 1.3)]
            if not opts:
                continue
            cs = [dict(best, **{k: o}) for o in opts]
            for c, s_ in zip(cs, evaluate(pool, tune, cs, say, f"C {k} ")):
                if s_["f1"] > bs["f1"] + 0.003:
                    best, bs = c, s_
            json.dump(best, open(os.path.join(out, "best.json"), "w"), indent=1)
        say(f"== best on tuning blocks: crown F1 {bs['f1']:.3f}  ({time.time() - t0:.0f}s)\n{json.dumps(best)}")
    json.dump(best, open(os.path.join(out, "best.json"), "w"), indent=1)


def test(out, params, workers):
    import cv2
    from canopie.canopy_auto import AutoParams, delineate_all
    blocks = sorted(glob.glob(os.path.join(out, "test_*.npz")))
    best = json.load(open(params or os.path.join(out, "best.json")))
    with Pool(workers) as pool:
        for tag, cfg in (("defaults", _defaults()), ("best", best)):
            per = [s for _, s in sorted(pool.map(run_job, [(i, b, cfg) for i, b in enumerate(blocks)]))]
            for b, s in zip(blocks, per):
                print(line(f"{tag} {os.path.basename(b)}", {}, s), flush=True)
            print(line(f"{tag} MEAN", cfg, {k: float(np.mean([s[k] for s in per])) for k in per[0]}), flush=True)
    for b in blocks:                                             # overlays of the best
        z = np.load(b, allow_pickle=True)
        img = z["img"]; res = delineate_all(img, AutoParams(**best))
        o = img[..., ::-1].copy()
        for g in z["gt"]:
            cv2.polylines(o, [np.round(np.array(g, float) - 0.5).astype(np.int32)], True, (0, 255, 0), 2)
        for c in res.crowns:
            cv2.polylines(o, [np.round(c.polygon).astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out, os.path.basename(b).replace(".npz", "_overlay.jpg")), o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build-train", "build-shp", "search", "test"])
    ap.add_argument("--project"); ap.add_argument("--out", required=True)
    ap.add_argument("--shp", help="build-shp: reference crown shapefile"); ap.add_argument("--image", help="build-shp: its image")
    ap.add_argument("--auto-align", action="store_true", help="build-shp: shift each block's crowns onto the image "
                                                               "(shadow gaps), keep only blocks where that is verified")
    ap.add_argument("--configs", type=int, default=150); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cap", type=float, default=CAP); ap.add_argument("--params")
    ap.add_argument("--start", help="json of params to start from (A0)")
    ap.add_argument("--known-size", action="store_true",
                    help="give each block its annotated median crown radius, as the app does from the user's crowns")
    a = ap.parse_args()
    CAP = a.cap
    if a.known_size:
        os.environ["AUTO_CANOPY_KNOWN_SIZE"] = "1"                  # inherited by the worker processes
    if a.cmd == "build":
        build(a.project, a.out)
    elif a.cmd == "build-train":
        build_train(a.project, a.out)
    elif a.cmd == "build-shp":
        build_shp(a.shp, a.image, a.out, auto_align=a.auto_align, workers=a.workers)
    elif a.cmd == "search":
        search(a.out, a.configs, a.workers, start=a.start)
    else:
        test(a.out, a.params, a.workers)
