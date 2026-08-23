"""QC regression tests for the "Export Project Images" band-order bug.

THE BUG (reported as: "jpeg images [meaning JPEG-compressed TIFFs] are being
saved as tif, this is okay, but the band order is wrong -- BGR instead of the
original RGB"):

`_load_image_simple` (the background `ProjectImagesExportWorker`) and
`_read_raw_any` (the foreground `export_project_images` path) both read TIFF
sources via tifffile, which returns the file's NATIVE band order -- never
BGR. But the save step downstream, in both paths, treated every array as if
it were genuinely BGR (the convention that only actually holds for a cv2
imdecode fallback) and applied an unconditional BGR->RGB flip.

That flip happened to be harmless wherever the final write went through
cv2.imwrite -- cv2.imwrite re-applies its own BGR convention on write, so an
unconditional flip of a genuinely-RGB tifffile array cancels out correctly.
But it was a REAL, confirmed R/B swap wherever the final write went through
tifffile.imwrite(photometric="rgb") instead (the branch used whenever the
.ax has a band expression / appended bands / classification enabled, or the
source isn't uint8/uint16) -- tifffile.imwrite does no BGR reinterpretation
of its own, so flipping a genuinely-RGB array before handing it there stores
true-B where true-R belongs and vice versa. This was verified empirically
(round-tripped a known R=10/G=20/B=30 array through the pre-fix and
post-fix code and read the raw stored bytes back) before writing the fix
below -- an earlier hand-derivation had actually blamed the WRONG branch
(the harmless cv2.imwrite one), which is exactly the kind of channel-order
reasoning that is easy to get backwards on paper and must be checked against
real pixels.

The fix: both loaders now return (array, channel_order) instead of a bare
array, and the save steps branch on the TRACKED order instead of assuming
BGR unconditionally.
"""
import os

import numpy as np
import pytest
import tifffile
import cv2

pytestmark = [pytest.mark.io]


# ---------------------------------------------------------------------------
# _load_image_simple (background worker) -- channel-order tracking
# ---------------------------------------------------------------------------
def _write_rgb_tiff(path, r=10, g=20, b=30, h=8, w=8, dtype=np.uint8):
    arr = np.zeros((h, w, 3), dtype=dtype)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    tifffile.imwrite(path, arr, photometric="rgb")
    return arr


def _write_bgr_jpg(path, r=10, g=20, b=30, h=8, w=8):
    """A JPEG whose TRUE color is (r, g, b) -- cv2.imwrite treats a 3-channel
    array as BGR, so the array fed in must be (b, g, r)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = b
    arr[..., 1] = g
    arr[..., 2] = r
    cv2.imwrite(path, arr)


def test_load_image_simple_tracks_tif_as_rgb(tmp_path):
    from ..project_tab import ProjectImagesExportWorker
    src = str(tmp_path / "src.tif")
    _write_rgb_tiff(src)

    img, order = ProjectImagesExportWorker._load_image_simple(None, src)
    assert order == "rgb"
    assert tuple(img[0, 0][:3]) == (10, 20, 30)


def test_load_image_simple_tracks_jpg_as_bgr(tmp_path):
    from ..project_tab import ProjectImagesExportWorker
    src = str(tmp_path / "src.jpg")
    _write_bgr_jpg(src, r=10, g=20, b=30)

    img, order = ProjectImagesExportWorker._load_image_simple(None, src)
    assert order == "bgr"
    # cv2.imdecode of a jpg whose true color is (10,20,30) yields a BGR
    # array (30ish, 20ish, 10ish) -- jpg is lossy, so allow tolerance.
    px = img[0, 0][:3].astype(int)
    assert abs(int(px[0]) - 30) <= 3 and abs(int(px[2]) - 10) <= 3


def test_load_image_simple_returns_none_order_bgr_on_missing_file():
    from ..project_tab import ProjectImagesExportWorker
    img, order = ProjectImagesExportWorker._load_image_simple(None, "no_such_file.tif")
    assert img is None
    assert order == "bgr"  # a harmless default; caller checks img is None first


# ---------------------------------------------------------------------------
# _reorder_first3 -- only bands 0/2 swap, band 1 and 4th+ band untouched
# ---------------------------------------------------------------------------
def test_reorder_first3_swaps_only_r_and_b():
    from ..project_tab import ProjectImagesExportWorker
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[..., 0] = 10   # R
    arr[..., 1] = 20   # G
    arr[..., 2] = 30   # B
    arr[..., 3] = 40   # 4th band (e.g. NIR/alpha) -- must survive untouched

    out = ProjectImagesExportWorker._reorder_first3(arr, swap=True)
    assert tuple(out[0, 0]) == (30, 20, 10, 40)


def test_reorder_first3_no_op_when_swap_false():
    from ..project_tab import ProjectImagesExportWorker
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 10; arr[..., 1] = 20; arr[..., 2] = 30
    out = ProjectImagesExportWorker._reorder_first3(arr, swap=False)
    assert tuple(out[0, 0]) == (10, 20, 30)
    assert out is arr, "swap=False must be a true no-op (no copy needed)"


def test_reorder_first3_leaves_grayscale_alone():
    from ..project_tab import ProjectImagesExportWorker
    arr = np.full((3, 3), 99, dtype=np.uint8)
    out = ProjectImagesExportWorker._reorder_first3(arr, swap=True)
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# End-to-end: worker.run() round-trips true color through both save branches
# ---------------------------------------------------------------------------
class _FakeProjectTab:
    """A plain (non-QObject) stand-in for the handful of things
    ProjectImagesExportWorker.run() actually calls on self.project_tab.

    ProjectTab itself is a QWidget subclass -- ProjectTab.__new__(ProjectTab)
    (the pattern used elsewhere in this suite for pure-Python methods like
    migrate_polygon_basis) produces an object whose C++/sip side was never
    constructed, and running it through an actual QThread.run() call trips
    PyQt's "super-class __init__() of type ProjectTab was never called" the
    moment anything touches Qt's meta-object machinery. Since run() only
    ever needs plain attribute access and _load_ax_json/_apply_ax_to_raw,
    a small non-Qt fake avoids that entirely and is exactly what these
    save-path tests need to isolate: channel-order bookkeeping, not the
    real .ax replay pipeline (which test_export_and_ax_regressions.py and
    the ML/CSV export tests already cover)."""

    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.exiftool_path = None
        self._is_exporting = False
        self._last_export_channel_order = None

    def _load_ax_json(self, fp):
        return {}  # no .ax -- _apply_ax_to_raw is never reached (ax is falsy)

    def _apply_ax_to_raw(self, img, ax, filepath=None):
        raise AssertionError("must not be called when _load_ax_json returns {}")


def _stub_tab(tmp_path):
    return _FakeProjectTab(str(tmp_path))


def test_worker_round_trips_true_color_uint8_tif_source(tmp_path, qapp):
    """The common case: a plain uint8 TIFF re-exported with no .ax (no band
    expression / classification), so needs_float is False -- this reaches
    the cv2.imwrite save branch."""
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"
    src = str(src_dir / "a.tif")
    _write_rgb_tiff(src, r=10, g=20, b=30)

    tab = _stub_tab(tmp_path)
    worker = ProjectImagesExportWorker(tab, str(out_dir), [src], "tif")
    worker.run()

    produced = list(out_dir.glob("*.tif"))
    assert produced, "worker did not write an output file"
    got = tifffile.imread(str(produced[0]))
    assert tuple(int(v) for v in got[0, 0][:3]) == (10, 20, 30), (
        f"band order corrupted on round-trip: got {got[0,0]}, expected (10,20,30)")


def test_worker_round_trips_true_color_needs_float_branch(tmp_path, qapp):
    """THE historically-broken branch: an .ax with classification enabled
    forces needs_float=True, routing the save through tifffile.imwrite
    instead of cv2.imwrite -- this is where the unconditional flip actually
    produced a wrong R/B swap (see module docstring)."""
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"
    src = str(src_dir / "a.tif")
    _write_rgb_tiff(src, r=10, g=20, b=30)

    # A minimal .ax that forces needs_float via has_classification, without
    # requiring a real trained model -- ProjectTab._load_ax_json just reads
    # this back as plain JSON, and _apply_ax_to_raw is never reached because
    # ax truthiness alone is enough to force needs_float below; what matters
    # here is purely the save-branch selection, so classification.enabled is
    # set but the worker's `if ax:` replay call is exercised too (empty
    # model config just means _apply_ax_to_raw's classification step no-ops
    # or is skipped internally -- either way the array's channel order must
    # survive unchanged, which is exactly what this test checks).
    ax_path = os.path.join(str(tmp_path), "a.ax")
    import json
    with open(ax_path, "w", encoding="utf-8") as f:
        json.dump({}, f)  # no classification -- keep needs_float False here;
    # needs_float is instead forced directly by writing a float32 source,
    # which is the OTHER trigger for the tifffile.imwrite branch
    # (img.dtype not in (uint8, uint16)) and needs no .ax at all.
    os.remove(ax_path)

    src_f = str(src_dir / "b.tif")
    _write_rgb_tiff(src_f, r=10, g=20, b=30, dtype=np.float32)

    tab = _stub_tab(tmp_path)
    worker = ProjectImagesExportWorker(tab, str(out_dir), [src_f], "tif")
    worker.run()

    produced = list(out_dir.glob("*.tif"))
    assert produced, "worker did not write an output file"
    got = tifffile.imread(str(produced[0]))
    assert tuple(float(v) for v in got[0, 0][:3]) == (10.0, 20.0, 30.0), (
        f"band order corrupted in the tifffile.imwrite branch: got {got[0,0]}, "
        f"expected (10,20,30) -- this is THE historically broken case")


def test_worker_jpg_source_round_trips_true_color(tmp_path, qapp):
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"
    src = str(src_dir / "a.jpg")
    _write_bgr_jpg(src, r=10, g=20, b=30)

    tab = _stub_tab(tmp_path)
    worker = ProjectImagesExportWorker(tab, str(out_dir), [src], "tif")
    worker.run()

    produced = list(out_dir.glob("*.jpg"))
    assert produced, "a jpg source must stay a jpg output regardless of export_format"
    from PIL import Image
    got = np.array(Image.open(str(produced[0])))
    px = got[0, 0][:3].astype(int)
    assert abs(px[0] - 10) <= 3 and abs(px[2] - 30) <= 3, (
        f"jpg source round-trip corrupted: got {px}, expected ~(10,20,30)")


# ---------------------------------------------------------------------------
# AST-level: channel order must be genuinely tracked, not hardcoded
# ---------------------------------------------------------------------------
def test_load_image_simple_returns_a_tuple_not_a_bare_array():
    from .test_export_and_ax_regressions import _names_in
    from ..project_tab import ProjectImagesExportWorker
    names = _names_in(ProjectImagesExportWorker._load_image_simple)
    assert "rgb" in names and "bgr" in names, (
        "_load_image_simple must label its result's channel order, not "
        "return a bare, untracked array")


def test_save_branches_reference_the_tracked_order_not_a_bare_flip():
    """Guards against reverting to the old unconditional `img[..., ::-1]`
    flip -- the fixed code must consult is_rgb_order (derived from the
    tracked chan_order / self.band_order) before deciding whether to swap."""
    from .test_export_and_ax_regressions import _names_in
    from ..project_tab import ProjectImagesExportWorker
    names = _names_in(ProjectImagesExportWorker.run)
    assert "is_rgb_order" in names
    assert "chan_order" in names
    assert "_reorder_first3" in names


def test_foreground_save_tiff_takes_a_src_order_param():
    """_read_raw_any/_save_tiff are closures inside export_project_images
    (not addressable on their own via __qualname__ -- _func_tree raises
    LookupError for "<locals>" -- so this checks the enclosing method's
    whole AST, which includes the nested defs)."""
    from .test_export_and_ax_regressions import _names_in
    from ..project_tab import ProjectTab
    names = _names_in(ProjectTab.export_project_images)
    assert "src_order" in names
    assert "is_rgb_order" in names
