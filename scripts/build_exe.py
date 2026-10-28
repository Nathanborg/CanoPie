"""Build a standalone CanoPie (no Python needed on the user's machine) with Nuitka.

    python scripts/build_exe.py [--out DIR] [--jobs N]

Writes DIR/main.dist/ (run CanoPie.exe) and DIR/CanoPie-<version>-windows.zip.
Optional accelerators and model back-ends (numba, xgboost, lightgbm) are left
out: CanoPie falls back without them, and they would double the size.

Build from a clean checkout (`git archive HEAD`): `--include-package=canopie`
takes every .py in the folder, scratch copies included. On Windows, set
NUITKA_CACHE_DIR to a short path (e.g. C:\\nc): Nuitka unpacks its MinGW there,
and under the Microsoft Store Python the default cache is deep enough to cut
off the compiler's headers at the 260-character path limit.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from canopie._version import __version__  # noqa: E402

# never bundled: optional back-ends, test suites, dev tools
LEAVE_OUT = [
    "numba", "llvmlite", "xgboost", "lightgbm", "torch", "tensorflow",
    "pytest", "_pytest", "IPython", "jupyter", "notebook", "tkinter",
    "canopie.qc", "numpy.tests", "scipy.tests", "sklearn.tests",
    "matplotlib.tests", "pandas", "setuptools", "pip",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    icon = os.path.join(a.out, "canopie.ico")
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    Image.open(os.path.join(ROOT, "logo.PNG")).save(
        icon, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    cmd = [
        sys.executable, "-m", "nuitka", os.path.join(ROOT, "main.py"),
        "--mode=standalone",
        "--enable-plugin=pyqt5",
        "--include-qt-plugins=sensible,styles",
        "--include-package=canopie",
        "--include-package-data=folium,branca,pyproj",
        "--include-data-files=%s=logo.PNG" % os.path.join(ROOT, "logo.PNG"),
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-unittest-mode=nofollow",
        "--noinclude-IPython-mode=nofollow",
        "--noinclude-dlls=cv2/opencv_videoio_ffmpeg*",    # 29 MB of video codecs CanoPie never reads
        "--windows-console-mode=disable",
        "--windows-icon-from-ico=%s" % icon,
        "--output-filename=CanoPie.exe",
        "--company-name=CanoPie",
        "--product-name=CanoPie",
        "--file-version=%s" % __version__,
        "--product-version=%s" % __version__,
        "--assume-yes-for-downloads",
        "--jobs=%d" % a.jobs,
        "--output-dir=%s" % a.out,
    ] + ["--nofollow-import-to=%s" % m for m in LEAVE_OUT]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)

    dist = os.path.join(a.out, "main.dist")
    for name in ("README.md", "LICENSE", "CHANGELOG.md"):
        shutil.copy(os.path.join(ROOT, name), dist)
    zip_base = os.path.join(a.out, "CanoPie-%s-windows" % __version__)
    shutil.make_archive(zip_base, "zip", dist)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(dist) for f in fs)
    print("folder: %.0f MB   zip: %.0f MB" % (size / 1e6, os.path.getsize(zip_base + ".zip") / 1e6))


if __name__ == "__main__":
    main()
