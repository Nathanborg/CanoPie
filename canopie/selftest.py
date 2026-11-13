"""Check that a CanoPie installation -- above all the frozen Windows exe -- has
everything it loads at runtime.

    CanoPie.exe --selftest [REPORT.txt]
    python -m canopie --selftest [REPORT.txt]

A freezer (Nuitka) only bundles what it can see imported. Packages that load
code or data by NAME at runtime are missed silently: imagecodecs' codecs (the
exe could not open LZW GeoTIFFs), PIL's image plugins, matplotlib's data,
pyproj's proj.db, folium's templates, worker processes. Each check below runs
the real thing on a tiny input; the report says OK or the error. Exit code 0
only if every check passed.
"""
import os
import sys
import tempfile
import time
import traceback

import numpy as np


# A FIXED list, never a scan of the folder: a source checkout can hold scratch
# scripts that run code on import (and the exe has no folder to scan).
# test_selftest.py keeps this list in step with the package.
CANOPIE_MODULES = [
    "analysis_replay", "annotator", "app", "ax_geometry", "ax_registration", "band_layout",
    "canopy_auto", "canopy_browser", "canopy_finder", "canopy_finder_dialog",
    "canopy_finder_meta", "canopy_tracer", "export_images_options_dialog",
    "image_data", "image_editor_dialog", "image_viewer", "lazy_analysis",
    "loaders", "log_window", "machine_learning_manager", "main_window",
    "ml_augmentation", "ml_augmentation_options_dialog", "nodata", "patch_dialog",
    "performance", "phenocam_filter", "polygon_bulk", "polygon_lod", "polygon_manager",
    "project_tab", "random_shapes_dialog", "random_shapes_generator",
    "raster_reader", "shapefile_import_dialog", "shapefile_io",
    "similarity_dialog", "spatial_index", "thumbnail_options_dialog", "utils",
]


def _canopie_modules():
    return ["canopie." + m for m in CANOPIE_MODULES]


# Third-party modules CanoPie imports (many only inside functions, so a crash
# shows up only when that feature is first used).
_THIRD_PARTY = [
    "PIL", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets", "cv2", "exifread",
    "folium", "geopy.distance", "joblib", "matplotlib.backends.backend_qt5agg",
    "matplotlib.figure", "matplotlib.pyplot", "numpy", "pyproj",
    "pystackreg", "scipy.optimize", "scipy.spatial", "scipy.stats", "scipy.ndimage",
    "shapefile", "shapely.geometry", "shapely.ops", "shapely.strtree", "sklearn.base",
    "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
    "sklearn.model_selection", "sklearn.naive_bayes", "sklearn.pipeline",
    "sklearn.preprocessing", "sklearn.svm", "sklearn.neighbors", "sklearn.tree",
    "sklearn.cluster", "sklearn.decomposition", "threadpoolctl", "tifffile",
    "imagecodecs",
]
# Optional: CanoPie falls back without them (left out of the exe on purpose).
_OPTIONAL = ["numba", "numexpr", "xgboost", "lightgbm", "psutil"]


def _square(x):
    return x * x


def _checks(tmp):
    import importlib

    def imports(names):
        def run():
            bad = []
            for n in names:
                try:
                    importlib.import_module(n)
                except Exception as e:
                    bad.append("%s (%s: %s)" % (n, type(e).__name__, e))
            if bad:
                raise ImportError("; ".join(bad))
            return "%d modules" % len(names)
        return run

    yield "import canopie modules", imports(_canopie_modules())
    yield "import third-party modules", imports(_THIRD_PARTY)

    def optional():
        have = []
        for n in _OPTIONAL:
            try:
                importlib.import_module(n)
                have.append(n)
            except Exception:
                pass
        return "present: %s (all optional)" % (", ".join(have) or "none")
    yield "optional accelerators", optional

    def tiff_codecs():
        import tifffile
        a8 = (np.arange(64 * 64 * 3) % 251).astype(np.uint8).reshape(64, 64, 3)
        f32 = np.random.default_rng(0).random((64, 64, 6), dtype=np.float32)
        u16 = (np.arange(64 * 64 * 6) % 60000).astype(np.uint16).reshape(64, 64, 6)
        cases = [
            ("lzw", f32, {}), ("lzw+horizontal predictor", u16, {"predictor": 2}),
            ("lzw+float predictor", f32, {"predictor": 3}),
            ("deflate", f32, {}), ("adobe_deflate", u16, {}), ("zstd", f32, {}),
            ("lzma", u16, {}), ("packbits", a8, {}), ("jpeg", a8, {}),
            ("webp", a8, {}), ("lerc", f32, {}), ("jpeg2000", a8, {}), ("png", a8, {}),
        ]
        bad, ok = [], []
        for name, arr, kw in cases:
            comp = name.split("+")[0]
            fp = os.path.join(tmp, "c_%s.tif" % len(ok + bad))
            try:
                tifffile.imwrite(fp, arr, tile=(32, 32), compression=comp,
                                 photometric="rgb" if arr.shape[2] == 3 else "minisblack",
                                 planarconfig="contig", **kw)
                back = tifffile.imread(fp)
                if back.shape != arr.shape:
                    raise ValueError("shape %s" % (back.shape,))
                if comp not in ("jpeg", "webp", "jpeg2000", "lerc") and not np.array_equal(back, arr):
                    raise ValueError("pixels differ")
                ok.append(name)
            except Exception as e:
                bad.append("%s (%s)" % (name, e))
        if bad:
            raise RuntimeError("failed: " + "; ".join(bad))
        return ", ".join(ok)
    yield "TIFF compressions (tifffile + imagecodecs)", tiff_codecs

    def raster_reader_window():
        import tifffile
        from canopie import raster_reader
        fp = os.path.join(tmp, "cube.tif")
        cube = np.random.default_rng(1).random((300, 260, 12), dtype=np.float32)
        tifffile.imwrite(fp, cube, tile=(64, 64), compression="lzw",
                         photometric="minisblack", planarconfig="contig")
        prof = raster_reader.probe(fp)
        rd = raster_reader.open_reader(fp, prof)
        win = rd.read_window(10, 20, 50, 40, [0, 5, 11]) if hasattr(rd, "read_window") else None
        prev = raster_reader.read_preview(rd, fp, [0, 1, 2], level=0, step=4)
        if prev.shape[2] != 3:
            raise ValueError("preview shape %s" % (prev.shape,))
        return "%s, preview %s%s" % (prof.strategy, prev.shape, "" if win is None else ", window ok")
    yield "lazy windowed read of an LZW cube", raster_reader_window

    def cv2_formats():
        import cv2
        img = (np.arange(40 * 30 * 3) % 255).astype(np.uint8).reshape(40, 30, 3)
        out = []
        for ext in (".jpg", ".png", ".tif", ".bmp", ".webp"):
            fp = os.path.join(tmp, "cv" + ext)
            if not cv2.imwrite(fp, img):
                raise RuntimeError("cv2 cannot write %s" % ext)
            if cv2.imread(fp, cv2.IMREAD_UNCHANGED) is None:
                raise RuntimeError("cv2 cannot read %s" % ext)
            out.append(ext)
        return " ".join(out)
    yield "OpenCV image formats", cv2_formats

    def pil_formats():
        from PIL import Image
        Image.init()
        img = Image.new("RGB", (30, 20), (10, 120, 30))
        out = []
        for ext in (".jpg", ".png", ".tif", ".bmp", ".webp", ".gif"):
            fp = os.path.join(tmp, "pil" + ext)
            img.save(fp)
            with Image.open(fp) as im:
                im.load()
            out.append(ext)
        return " ".join(out)
    yield "PIL image plugins", pil_formats

    def exif():
        import exifread
        from PIL import Image
        fp = os.path.join(tmp, "exif.jpg")
        im = Image.new("RGB", (8, 8))
        ex = Image.Exif()
        ex[0x010F] = "CanoPieTest"          # Make
        im.save(fp, exif=ex)
        with open(fp, "rb") as f:
            tags = exifread.process_file(f, details=False)
        if "Image Make" not in tags:
            raise RuntimeError("exifread found no Make tag")
        return "Make=%s" % tags["Image Make"]
    yield "EXIF reading", exif

    def qt_images():
        from PyQt5 import QtGui, QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["selftest"])
        fmts = {bytes(f).decode() for f in QtGui.QImageReader.supportedImageFormats()}
        need = {"jpg", "png", "tif", "bmp"}
        if not need <= fmts:
            raise RuntimeError("Qt cannot read: %s" % ", ".join(sorted(need - fmts)))
        return "Qt reads " + " ".join(sorted(fmts))
    yield "Qt image format plugins", qt_images

    def mpl():
        import matplotlib
        matplotlib.use("Qt5Agg", force=True)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        fig = Figure(figsize=(2, 2))
        FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [1, 0]); ax.set_title("t"); ax.hist(np.random.rand(50))
        fp = os.path.join(tmp, "plot.png")
        fig.savefig(fp)
        return "Qt5Agg figure saved (%d bytes)" % os.path.getsize(fp)
    yield "matplotlib (Qt canvas, fonts, savefig)", mpl

    def proj():
        import pyproj
        t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
        x, y = t.transform(-79.85, 9.15)
        crs = pyproj.CRS.from_epsg(32617)
        if not (600000 < x < 700000):
            raise ValueError("odd transform %s" % x)
        return "%s, x=%.0f (proj data: %s)" % (crs.name, x, pyproj.datadir.get_data_dir())
    yield "pyproj CRS database", proj

    def shapes():
        import shapefile
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
        a, b = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        u = unary_union([a, b])
        STRtree([a, b]).query(a)
        base = os.path.join(tmp, "shp")
        w = shapefile.Writer(base, shapeType=shapefile.POLYGON)
        w.field("name", "C")
        w.poly([list(a.exterior.coords)]); w.record("a"); w.close()
        r = shapefile.Reader(base)
        n = len(r.shapes()); r.close()
        return "union area %.1f, %d shape written/read" % (u.area, n)
    yield "shapely + shapefile", shapes

    def sk():
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        rng = np.random.default_rng(0)
        X = rng.random((80, 4)); y = (X[:, 0] > 0.5).astype(int)
        for m in (RandomForestClassifier(n_estimators=10, n_jobs=-1), ExtraTreesClassifier(n_estimators=10),
                  make_pipeline(StandardScaler(), SVC()), LogisticRegression(), GaussianNB()):
            m.fit(X, y); m.predict(X)
        fp = os.path.join(tmp, "m.joblib")
        joblib.dump(m, fp); joblib.load(fp)
        return "5 models fit/predict, joblib dump/load"
    yield "scikit-learn models (threads)", sk

    def sk_procs():
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X = rng.random((60, 3)); y = (X[:, 0] > 0.5).astype(int)
        g = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=2, n_jobs=2)
        g.fit(X, y)
        return "GridSearchCV n_jobs=2 best C=%s" % g.best_params_["C"]
    yield "scikit-learn search in worker processes (n_jobs)", sk_procs

    def procpool():
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
            r = list(ex.map(_square, [2, 3], timeout=120))
        if r != [4, 9]:
            raise ValueError(r)
        return "spawned workers returned %s" % r
    yield "process pool (performance.get_cpu_executor)", procpool

    def sci():
        from scipy import ndimage, optimize, spatial, stats
        ndimage.gaussian_filter(np.ones((10, 10)), 1)
        optimize.minimize_scalar(lambda v: (v - 2) ** 2)
        spatial.cKDTree(np.random.rand(20, 2)).query([0.5, 0.5])
        stats.pearsonr(np.arange(10), np.arange(10) ** 2)
        return "ndimage, optimize, spatial, stats"
    yield "scipy", sci

    def reg():
        from pystackreg import StackReg
        a = np.zeros((40, 40)); a[10:20, 10:20] = 1
        b = np.roll(a, 3, axis=1)
        StackReg(StackReg.TRANSLATION).register(a, b)
        return "translation registered"
    yield "pystackreg", reg

    def fol():
        import folium
        m = folium.Map(location=[9.15, -79.85], zoom_start=12, tiles=None)
        folium.Marker([9.15, -79.85]).add_to(m)
        fp = os.path.join(tmp, "map.html")
        m.save(fp)
        return "map saved (%d bytes)" % os.path.getsize(fp)
    yield "folium map export", fol

    def misc():
        from geopy.distance import geodesic
        d = geodesic((9.15, -79.85), (9.16, -79.85)).meters
        return "geodesic %.0f m" % d
    yield "geopy", misc


def run(report_path=None):
    lines = ["CanoPie self-test", "python %s, frozen=%s, executable=%s" % (
        sys.version.split()[0],
        bool(getattr(sys, "frozen", False) or "__compiled__" in globals()),
        sys.executable), ""]
    failed = 0
    with tempfile.TemporaryDirectory(prefix="canopie_selftest_", ignore_cleanup_errors=True) as tmp:
        for name, fn in _checks(tmp):
            t0 = time.time()
            try:
                detail = fn()
                lines.append("OK    %-48s %s  (%.1fs)" % (name, detail, time.time() - t0))
            except Exception as e:
                failed += 1
                lines.append("FAIL  %-48s %s: %s" % (name, type(e).__name__, e))
                lines.extend("        " + l for l in traceback.format_exc().splitlines()[-6:])
    lines += ["", "%d check(s) failed" % failed if failed else "all checks passed"]
    text = "\n".join(lines) + "\n"
    if report_path is None:
        report_path = os.path.join(tempfile.gettempdir(), "CanoPie", "selftest.txt")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        print(text)
        print("report:", report_path)
    except Exception:
        pass                                    # no console in the windowed exe
    return 1 if failed else 0
