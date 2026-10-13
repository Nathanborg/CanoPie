"""Build a small CanoPie example project to try the software in a few minutes.

    python examples/make_example_project.py [output folder]     (default: ./canopie_example)

It writes synthetic but structured data -- no download, no license question:

* mosaics/  three dates of a 4-band (R, G, B, NIR) georeferenced orthomosaic of
  the same patch of canopy (UTM 21S, 10 cm). Each date is shifted by a few
  metres (as flights are), the light changes, and one tree ("tree_3") drops
  its leaves by the last date;
* phenocam/ five RGB frames of a fixed camera over the same canopy, under
  changing illumination;
* flight/multispectral + flight/thermal  raw frames of a drone flight over
  the patch, as two cameras write them: four captures of four single-band
  frames (IMG_0001_1.tif ... IMG_0004_4.tif: R, G, B, NIR) and one thermal
  frame per capture, numbered by its own camera (IMG_0101_IR.tif ...) -- for
  "Open multispectral/multiple Folders";
* project.json + polygons/ a CanoPie project with two roots ("Mosaics",
  "Phenocam"), five crowns drawn on the first date, and three training
  classes for the MachineLearning Manager ("leaf", "bare", "understory").

Then start CanoPie (python main.py), File > Open Project, and pick the folder.
docs/tutorial.md walks through it.
"""
import json
import math
import os
import sys

import numpy as np

GSD = 0.10                         # m / px
N = 900                            # mosaic size (90 m)
ORIGIN = (500000.0, 9700000.0)     # UTM 21S
TREES = [  # name, centre (px), radius (px), leaf NIR / red / green / blue of the crown (0-1)
    ("tree_1", (250, 260), 60, (0.62, 0.06, 0.14, 0.05)),
    ("tree_2", (560, 230), 75, (0.55, 0.08, 0.16, 0.06)),
    ("tree_3", (420, 520), 70, (0.66, 0.05, 0.13, 0.05)),      # deciduous: bare on date 3
    ("tree_4", (690, 600), 55, (0.50, 0.07, 0.15, 0.06)),
    ("tree_5", (220, 680), 65, (0.58, 0.06, 0.15, 0.05)),
]
DATES = [("2024-02-10", (0.0, 0.0), 1.00), ("2024-05-12", (3.1, -1.7), 0.85), ("2024-08-20", (-2.4, 2.2), 1.10)]


def _canopy(n, seed, date_index, extra=60):
    """(n+2*extra)^2 x 4 float reflectance of the whole scene (the same ground
    on every date); the trees at TREES (+extra)."""
    rng = np.random.default_rng(seed)
    m = n + 2 * extra
    yy, xx = np.mgrid[0:m, 0:m].astype(np.float32)
    # understory / background crowns: smooth noise
    base = np.zeros((m, m), np.float32)
    for s, a in ((40, 0.5), (14, 0.3), (5, 0.2)):
        g = rng.normal(size=(m // s + 3, m // s + 3)).astype(np.float32)
        import cv2
        base += a * cv2.resize(g, (m, m), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    # leaf clumps and shadows (what makes real crowns registrable)
    clump = np.zeros((m, m), np.float32)
    for s, a in ((3, 0.6), (8, 0.4)):
        g = rng.normal(size=(m // s + 3, m // s + 3)).astype(np.float32)
        clump += a * cv2.resize(g, (m, m), interpolation=cv2.INTER_CUBIC)
    clump = 0.7 + 0.3 * np.tanh(clump)
    refl = np.stack([0.05 + 0.03 * base, 0.10 + 0.05 * base, 0.04 + 0.02 * base, 0.35 + 0.25 * base], axis=2)
    refl *= clump[..., None]
    for i, (name, (cx, cy), r, (nir, red, green, blue)) in enumerate(TREES):
        cx, cy = cx + extra, cy + extra
        d = np.hypot(xx - cx, yy - cy) / r
        w = np.clip(1.2 - d, 0, 1)[..., None]
        leaf = np.array([red, green, blue, nir], np.float32)
        if name == "tree_3" and date_index == 2:
            leaf = np.array([0.16, 0.14, 0.11, 0.24], np.float32)   # bare branches: brown, low NIR
        refl = refl * (1 - w) + leaf * clump[..., None] * w
    return refl


def _write_geotiff(path, arr, x0, y0, gsd, epsg=32721):
    import tifffile
    gk = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg]
    tifffile.imwrite(path, arr, photometric="minisblack", planarconfig="contig", compression="deflate",
                     extratags=[(33922, "d", 6, (0, 0, 0, float(x0), float(y0), 0), True),
                                (33550, "d", 3, (gsd, gsd, 0), True),
                                (34735, "H", len(gk), tuple(gk), True)])


def main(out):
    import cv2
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "mosaics"), exist_ok=True)
    os.makedirs(os.path.join(out, "phenocam"), exist_ok=True)
    os.makedirs(os.path.join(out, "polygons"), exist_ok=True)
    mosaics = []
    for k, (date, (dx_m, dy_m), light) in enumerate(DATES):
        # each flight's georeference is off by a few metres: the file says the
        # patch starts at ORIGIN, but the ground in it is shifted by (dx_m east,
        # dy_m north) -- the whole scene, as a GPS error does
        sx, sy = int(round(dx_m / GSD)), int(round(-dy_m / GSD))
        refl = _canopy(N, seed=7, date_index=k)[60 - sy:60 - sy + N, 60 - sx:60 - sx + N]
        dn = np.clip(refl * light * 40000 + np.random.default_rng(k).normal(0, 60, refl.shape), 0, 65535)
        p = os.path.join(out, "mosaics", f"mosaic_RGBN_{date}.tif")
        _write_geotiff(p, dn.astype(np.uint16), ORIGIN[0], ORIGIN[1], GSD)
        mosaics.append(p)
    # a drone flight: 4 captures along the patch, 4 single-band frames + 1 thermal frame each
    os.makedirs(os.path.join(out, "flight", "multispectral"), exist_ok=True)
    os.makedirs(os.path.join(out, "flight", "thermal"), exist_ok=True)
    import tifffile
    scene = _canopy(N, seed=7, date_index=0)
    for k in range(4):
        x0 = 60 + 120 * k
        win = scene[200:200 + 480, x0:x0 + 640]                       # 64 m x 48 m footprint
        win = cv2.resize(win, (640, 480), interpolation=cv2.INTER_AREA)
        for b in range(4):
            dn = np.clip(win[:, :, b] * 30000 + np.random.default_rng(10 * k + b).normal(0, 40, win.shape[:2]),
                         0, 65535).astype(np.uint16)
            tifffile.imwrite(os.path.join(out, "flight", "multispectral", f"IMG_{k + 1:04d}_{b + 1}.tif"), dn)
        ndvi = (win[:, :, 3] - win[:, :, 0]) / (win[:, :, 3] + win[:, :, 0] + 1e-6)
        kelvin = 300.0 + 6.0 * (0.8 - ndvi)                            # leafy crowns are cooler
        tifffile.imwrite(os.path.join(out, "flight", "thermal", f"IMG_{k + 101:04d}_IR.tif"),
                         np.clip(kelvin * 100, 0, 65535).astype(np.uint16))        # centi-kelvin
    frames = []
    full = _canopy(N, seed=7, date_index=0)[60:60 + N, 60:60 + N]
    rgb = np.clip(full[:, :, [0, 1, 2]] * 900, 0, 255)
    for k in range(5):
        light = 0.7 + 0.12 * k
        tilt = np.linspace(1.15 - 0.08 * k, 0.85 + 0.08 * k, N, dtype=np.float32)[None, :, None]   # sun side
        img = np.clip(rgb * light * tilt, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA)
        p = os.path.join(out, "phenocam", f"phenocam_2024-03-{10 + k:02d}_1200.jpg")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append(p)
    # crowns drawn on the first date (image pixels of the mosaic)
    polys = {}
    a = np.linspace(0, 2 * math.pi, 25)[:-1]
    for name, (cx, cy), r, _leaf in TREES:
        pts = np.column_stack((cx + 0.8 * r * np.cos(a), cy + 0.8 * r * np.sin(a))).round(2).tolist()
        rec = {"points": pts, "coord_space": "image", "image_ref_size": {"w": N, "h": N},
               "name": name, "root": "1", "type": "polygon"}
        polys[name] = {mosaics[0].replace("\\", "/"): rec}
        stem = os.path.splitext(os.path.basename(mosaics[0]))[0]
        with open(os.path.join(out, "polygons", f"{name}_{stem}_polygons.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
    # training classes for the MachineLearning Manager (one polygon per class and image)
    def circle(cx, cy, r, n=16):
        b = np.linspace(0, 2 * math.pi, n + 1)[:-1]
        return np.column_stack((cx + r * np.cos(b), cy + r * np.sin(b))).round(2).tolist()
    sx3, sy3 = int(round(DATES[2][1][0] / GSD)), int(round(-DATES[2][1][1] / GSD))
    train = [("leaf", 0, circle(250, 260, 30)), ("leaf", 1, circle(560 + 31, 230 + 17, 35)),
             ("bare", 2, circle(420 + sx3, 520 + sy3, 30)), ("understory", 0, circle(405, 110, 22)),
             ("understory", 2, circle(405 + sx3, 110 + sy3, 22))]
    for name, k, pts in train:
        rec = {"points": pts, "coord_space": "image", "image_ref_size": {"w": N, "h": N},
               "name": name, "root": "1", "type": "polygon"}
        polys.setdefault(name, {})[mosaics[k].replace("\\", "/")] = rec
        stem = os.path.splitext(os.path.basename(mosaics[k]))[0]
        with open(os.path.join(out, "polygons", f"{name}_{stem}_polygons.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
    project = {
        "project_name": "CanoPie example", "mode": "rgb_only", "root_offset": 0, "current_root_index": 0,
        "root_coordinates": {}, "rgb_folder_path": out, "batch_size": 5,
        "multispectral_root_names": ["Mosaics", "Phenocam"],
        "multispectral_image_data_groups": {"Mosaics": [p.replace("\\", "/") for p in mosaics],
                                            "Phenocam": [p.replace("\\", "/") for p in frames]},
        "all_polygons": polys,
    }
    with open(os.path.join(out, "project.json"), "w", encoding="utf-8") as f:
        json.dump(project, f, indent=1)
    print(f"Example project written to {out}")
    print("  start CanoPie (python main.py), File > Open Project, and choose that folder")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "canopie_example")
