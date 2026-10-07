"""
CONTRACT: `b1` means RED in every consumer, for every loader provenance.

WHY THIS FILE EXISTS
--------------------
CanoPie evaluates band-math expressions (`b1`, `(b4-b1)/(b4+b1)`, `b1>150`, a
model's stored `expressions`) in at least five genuinely separate code paths:

    viewer replay      ProjectTab.apply_aux_modifications
    export replay      ProjectTab._apply_ax_to_raw (via _get_export_image)
    CSV statistics     ProjectTab.process_polygon's band-math columns
    Image Editor       ImageEditorDialog.process_band_expression
                       / apply_all_modifications_to_image
    ML manager         MachineLearningManager._select_bands_and_expressions

Each of them used to decide "which array plane is b1" from the SHAPE of the
array (`if C == 3: swap`), or by rewriting the expression STRING
(`b1` <-> `b3` via a `__B1_TEMP__` placeholder). Both idioms are wrong for the
same reason: **channel order is provenance, not a format constant**. cv2 always
decodes BGR; tifffile always returns the file's native order. A shape guess is
right for one loader and silently backwards for the other, and the string
rewrite additionally never touched `b4`+, so wide stacks addressed the wrong
planes.

The fix routes every one of those sites through
`utils.to_rgb_band_order(arr, channel_order)`, driven by the loader that
actually produced the pixels. The rest of the QC suite is green but predates
that fix and would NOT catch a future site drifting back to `C == 3`: nothing
else here drives two provenances of the SAME logical image through every
consumer and compares the numbers.

WHAT MAKES THESE TESTS DECISIVE
-------------------------------
The synthetic image's three channels occupy **disjoint value ranges**
(R in [200,219], G in [100,119], B in [30,49]), and bands 4-6 of the stack sit
at [1000,1019] / [2000,2019] / [3000,3019]. So a single number is enough to
name the physical band it came from: no symmetric/grayscale image, no
"passes either way" arithmetic. A `b1` that resolved to Blue would land in
[30,49] and fail loudly.

Every assertion here is made against output produced by the REAL production
function -- no reimplementation of the band math, no assertions on source text
(except the explicitly-supplementary AST guards at the bottom, which are
themselves verified against a deliberately-broken module).
"""
import ast
import json
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.contract, pytest.mark.extraction]


# ---------------------------------------------------------------------------
# The fixture image: three channels in DISJOINT value ranges.
# ---------------------------------------------------------------------------
H = W = 32

#: (low, high) inclusive value range of each logical channel.
RED_RANGE = (200, 219)
GREEN_RANGE = (100, 119)
BLUE_RANGE = (30, 49)

#: base value per band for the 6-band stack (band 1..6); spread is +0..19.
STACK_BASES = [200, 100, 30, 1000, 2000, 3000]


def _channel_planes():
    """R, G, B as (H, W) uint8 arrays with disjoint value ranges.

    Spatially varying on purpose: a constant image would let a broken consumer
    agree with a correct one by accident on any statistic that happens to be
    shape-invariant.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    r = (RED_RANGE[0] + ((xx + yy) % 20)).astype(np.uint8)
    g = (GREEN_RANGE[0] + ((2 * xx + yy) % 20)).astype(np.uint8)
    b = (BLUE_RANGE[0] + ((xx + 3 * yy) % 20)).astype(np.uint8)
    return r, g, b


def _stack_planes():
    """Six uint16 planes, each in its own disjoint range (see STACK_BASES)."""
    yy, xx = np.mgrid[0:H, 0:W]
    return [(base + ((xx * (i + 1) + yy) % 20)).astype(np.uint16)
            for i, base in enumerate(STACK_BASES)]


def _in_range(value, rng, what):
    lo, hi = rng
    assert lo <= float(value) <= hi, (
        f"{what}: got {value}, which is NOT in {rng}. The disjoint fixture "
        f"ranges are R{RED_RANGE} G{GREEN_RANGE} B{BLUE_RANGE} -- so this "
        f"number names a different physical band than the one required.")


# ---------------------------------------------------------------------------
# A real project holding the same logical image under BOTH loader provenances
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bc_project(qapp, tmp_path_factory):
    """One real ProjectTab over three files:

      pair_native.tif  written by tifffile in R,G,B order
                       -> export loads it with tifffile  -> channel_order "rgb"
                       -> viewer/editor load it with cv2 -> channel_order "bgr"
      pair_cv2.png     the SAME logical image, written so cv2 reads it back
                       -> every path loads it with cv2   -> channel_order "bgr"
      stack6.tif       6 bands, always tifffile          -> "rgb"

    The first two are pixel-for-pixel the same picture; they exist purely so
    the same expression can be run under both provenances and compared.
    """
    import cv2
    import tifffile
    from PyQt5 import QtWidgets

    from ..project_tab import ProjectTab

    root = tmp_path_factory.mktemp("band_convention")
    images = root / "images"
    images.mkdir()

    r, g, b = _channel_planes()
    rgb = np.stack([r, g, b], axis=-1)

    tif = str(images / "pair_native.tif")
    png = str(images / "pair_cv2.png")
    tifffile.imwrite(tif, rgb, photometric="rgb")
    # cv2.imwrite consumes a BGR array, so hand it the reversed view; the FILE
    # then holds the identical logical picture as pair_native.tif.
    cv2.imwrite(png, rgb[..., ::-1])

    stack = np.stack(_stack_planes(), axis=-1)
    st = str(images / "stack6.tif")
    tifffile.imwrite(st, stack, photometric="minisblack", planarconfig="contig")

    # A polygon covering the WHOLE frame, so process_polygon's Mean is directly
    # comparable to a whole-array mean. (Asserted, not assumed -- the tests
    # check Pixel Count == H*W before relying on it.)
    pts = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
    all_polygons = {}
    groups = {}
    root_names = []
    for i, fp in enumerate((tif, png, st)):
        gname = "poly__" + os.path.basename(fp)
        all_polygons[gname] = {fp: {
            "points": [[float(x), float(y)] for (x, y) in pts],
            "coord_space": "image",
            "image_ref_size": {"w": W, "h": H},
            "name": gname, "root": str(i + 1), "type": "polygon",
        }}
        rn = f"root_{i}"
        root_names.append(rn)
        groups[rn] = [fp]

    project_folder = root / "project"
    project_folder.mkdir()
    with open(project_folder / "project.json", "w", encoding="utf-8") as f:
        json.dump({
            "project_name": "band_convention", "mode": "rgb_only",
            "root_offset": 0, "all_polygons": all_polygons,
            "current_root_index": 0, "root_coordinates": {},
            "rgb_folder_path": str(images), "batch_size": 1,
            "multispectral_root_names": root_names,
            "multispectral_image_data_groups": groups,
        }, f)

    pt = ProjectTab("band_convention", QtWidgets.QTabWidget())
    pt.load_project(str(project_folder))

    yield {"tab": pt, "tif": tif, "png": png, "stack": st,
           "red": r, "green": g, "blue": b, "stack_planes": _stack_planes()}

    # Stop background timers before interpreter shutdown (see conftest's
    # synthetic_project fixture for why).
    for attr in ("_poly_idle_timer", "_autosave_timer"):
        t = getattr(pt, attr, None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass


def _clear_caches(pt):
    """Every cache keyed by filepath -- a stale entry from a previous .ax would
    otherwise satisfy the next assertion with the wrong pixels."""
    for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                 "_bm_expr_cache", "_file_nodata_cache", "_per_band_nd_cache",
                 "_master_nd_cache", "_imgdata_cache", "_raw_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            try:
                c.clear()
            except Exception:
                pass


def _set_ax(pt, fp, ax):
    with open(pt._ax_path_for(fp), "w", encoding="utf-8") as f:
        json.dump(ax, f)
    _clear_caches(pt)


def _clear_ax(pt, fp):
    try:
        p = pt._ax_path_for(fp)
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass
    _clear_caches(pt)


# ---------------------------------------------------------------------------
# The individual consumers, each driven through its REAL entry point.
# Every one returns the 2-D result of evaluating `expr`.
# ---------------------------------------------------------------------------
def _viewer_expression(pt, fp, expr):
    """ProjectTab.apply_aux_modifications -- the display replay.

    Loads via the same call loaders.py makes (`_imagedata_or_fallback`), so the
    channel_order fed in is the one the real viewer would see, then replays the
    .ax. The expression lands as the trailing band.
    """
    from ..project_tab import ProjectTab

    _set_ax(pt, fp, {"band_expression": expr, "band_enabled": True})
    try:
        lite = pt._imagedata_or_fallback(fp)
        out = ProjectTab.apply_aux_modifications(
            fp, lite.image, pt.project_folder, global_mode=False,
            src_channel_order=lite.channel_order)
        out = np.asarray(out)
        assert out.ndim == 3, f"viewer replay returned {out.shape}, expected HxWxC"
        return np.asarray(out[..., -1], dtype=np.float64), lite.channel_order
    finally:
        _clear_ax(pt, fp)


def _export_expression(pt, fp, expr):
    """_get_export_image -> _apply_ax_to_raw -> _channels_in_export_order:
    the CSV/statistics replay. The expression lands as the trailing channel."""
    _set_ax(pt, fp, {"band_expression": expr, "band_enabled": True})
    try:
        img, _info = pt._get_export_image(fp)
        order = getattr(pt, "_last_export_channel_order", None)
        chans = pt._channels_in_export_order(img)
        return np.asarray(chans[-1], dtype=np.float64), order
    finally:
        _clear_ax(pt, fp)


def _process_polygon_expression_mean(pt, fp, expr):
    """ProjectTab.process_polygon's band-math column -- what the CSV actually
    contains. Returns (mean, pixel_count, rows_by_channel)."""
    group = "poly__" + os.path.basename(fp)
    poly = pt.all_polygons[group][fp]
    rows, _ = pt.process_polygon(
        group, fp, poly, {}, [], False,
        opts={"stats": {"mean": True},
              "band_math": {"enabled": True, "formulas": {"__expr__": expr}}})
    by_channel = {r.get("Channel"): r for r in rows if isinstance(r, dict)}
    assert "__expr__" in by_channel, (
        f"process_polygon produced no band-math row for {expr!r}; rows: "
        f"{sorted(k for k in by_channel if k)}")
    row = by_channel["__expr__"]
    return float(row["Mean"]), int(row.get("Pixel Count") or 0), by_channel


def _editor_expression(fp, expr):
    """ImageEditorDialog.process_band_expression, fed by the editor's OWN
    loader (`_load_raw_image`), which is what records `_editor_channel_order`.

    Uses `__new__` deliberately: the real __init__ builds a whole QDialog UI,
    and both methods here are pure-Python (they touch no Qt meta-object
    machinery), so this exercises the genuine code path -- the same pattern
    test_image_editor_apply_and_band_expr.py already relies on.
    """
    from ..image_editor_dialog import ImageEditorDialog

    ed = ImageEditorDialog.__new__(ImageEditorDialog)
    ed.image_filepath = fp
    arr = ImageEditorDialog._load_raw_image(ed)
    assert arr is not None, f"editor could not load {fp}"
    order = getattr(ed, "_editor_channel_order", None)
    out = np.asarray(ImageEditorDialog.process_band_expression(
        ed, arr.astype(np.float32), expr))
    assert getattr(ed, "_last_band_expr_error", None) in (None, ""), (
        f"editor expression {expr!r} failed: {ed._last_band_expr_error}")
    assert out.ndim == 3, f"editor returned {out.shape}"
    return np.asarray(out[..., -1], dtype=np.float64), order


def _editor_apply_all_expression(fp, expr):
    """The other editor entry point: the full modification pipeline."""
    from ..image_editor_dialog import ImageEditorDialog

    ed = ImageEditorDialog.__new__(ImageEditorDialog)
    ed.image_filepath = fp
    arr = ImageEditorDialog._load_raw_image(ed)
    out = np.asarray(ImageEditorDialog.apply_all_modifications_to_image(
        ed, arr.astype(np.float32),
        {"band_expression": expr, "band_enabled": True}))
    assert out.ndim == 3, f"editor apply_all returned {out.shape}"
    return np.asarray(out[..., -1], dtype=np.float64)


def _ml_expression(mgr, fp, expr):
    """MachineLearningManager._select_bands_and_expressions -- the shared
    helper BOTH the pristine and augmented training passes call, and the place
    a model's stored `expressions` become feature columns.

    Also returns the "red_channel" band feature from the SAME call, because the
    real invariant is that the two agree: an expression saying `b1` and the
    feature column named red must be the same physical plane.
    """
    img, _info = mgr._get_export_image(fp)
    chans = mgr._channels_in_export_order(img)
    sel, exprs = mgr._select_bands_and_expressions(
        np.asarray(img).astype(np.float32), chans, [0, 1, 2], [("e", expr)])
    assert exprs and exprs[0] is not None, (
        f"ML manager returned no result for {expr!r}")
    return (np.asarray(exprs[0], dtype=np.float64),
            np.asarray(sel[0], dtype=np.float64),
            mgr._current_channel_order())


# ---------------------------------------------------------------------------
# 1. THE core cross-consumer contract
# ---------------------------------------------------------------------------
@pytest.mark.ml
@pytest.mark.editor
@pytest.mark.parametrize("which", ["tif", "png"])
def test_b1_is_red_in_every_consumer(bc_project, ml_manager_factory, which):
    """Every consumer, one image, one expression: `b1` must be RED everywhere.

    `pair_native.tif` and `pair_cv2.png` are the SAME picture; between them the
    five consumers span both provenances (the .tif is tifffile/"rgb" on the
    export path but cv2/"bgr" on the viewer and editor paths -- exactly the
    mismatch that used to make the screen and the CSV disagree).
    """
    pt = bc_project["tab"]
    fp = bc_project[which]
    red = bc_project["red"].astype(np.float64)

    viewer, viewer_order = _viewer_expression(pt, fp, "b1")
    export, export_order = _export_expression(pt, fp, "b1")
    editor, editor_order = _editor_expression(fp, "b1")
    editor_all = _editor_apply_all_expression(fp, "b1")

    mgr = ml_manager_factory(pt)
    ml_expr, ml_red_feature, ml_order = _ml_expression(mgr, fp, "b1")

    pp_mean, pp_count, pp_rows = _process_polygon_expression_mean(pt, fp, "b1")

    arrays = {
        "viewer (apply_aux_modifications)": viewer,
        "export (_apply_ax_to_raw)": export,
        "editor (process_band_expression)": editor,
        "editor (apply_all_modifications_to_image)": editor_all,
        "ML (_select_bands_and_expressions)": ml_expr,
        "ML band feature 'red_channel'": ml_red_feature,
    }

    # (a) every array consumer is EXACTLY the red plane
    for name, arr in arrays.items():
        assert arr.shape == red.shape, f"{name}: shape {arr.shape} != {red.shape}"
        assert np.array_equal(arr, red), (
            f"{name}: `b1` is not Red. Sample value {arr[0, 0]} vs Red "
            f"{red[0, 0]} / Blue {bc_project['blue'][0, 0]}. Orders seen: "
            f"viewer={viewer_order} export={export_order} editor={editor_order} "
            f"ml={ml_order}")

    # (b) they agree with EACH OTHER pairwise (stated separately so a future
    #     failure names the two consumers that drifted, not just "not red")
    names = list(arrays)
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        assert np.array_equal(arrays[a], arrays[b]), (
            f"{a} and {b} disagree about `b1`")

    # (c) the CSV number agrees too
    assert pp_count == H * W, (
        f"whole-frame polygon covered {pp_count} of {H * W} pixels; the Mean "
        "comparison below assumes it covers all of them")
    assert pp_mean == pytest.approx(float(red.mean()), abs=1e-6), (
        f"process_polygon's band-math Mean for `b1` is {pp_mean}, but Red's "
        f"mean is {red.mean()} (Blue's is {bc_project['blue'].mean()})")

    # (d) and it is unambiguously in RED's range, not Blue's or Green's
    _in_range(pp_mean, RED_RANGE, "process_polygon `b1` Mean")
    _in_range(float(viewer.mean()), RED_RANGE, "viewer `b1`")
    _in_range(float(export.mean()), RED_RANGE, "export `b1`")

    # (e) process_polygon's own R column must be that same plane -- an
    #     expression and the channel it names cannot disagree inside one CSV row
    assert float(pp_rows["R"]["Mean"]) == pytest.approx(pp_mean, abs=1e-6), (
        "process_polygon's `b1` column and its R column disagree")


@pytest.mark.editor
@pytest.mark.parametrize("which", ["tif", "png"])
def test_b3_is_blue_in_every_consumer(bc_project, which):
    """The mirror image of the test above -- and the thing that makes it
    non-vacuous. If `b1` and `b3` produced the same number, (a) above could
    pass with the mapping fully inverted."""
    pt = bc_project["tab"]
    fp = bc_project[which]
    blue = bc_project["blue"].astype(np.float64)

    viewer, _ = _viewer_expression(pt, fp, "b3")
    export, _ = _export_expression(pt, fp, "b3")
    editor, _ = _editor_expression(fp, "b3")
    pp_mean, _c, _rows = _process_polygon_expression_mean(pt, fp, "b3")

    for name, arr in (("viewer", viewer), ("export", export), ("editor", editor)):
        assert np.array_equal(arr, blue), f"{name}: `b3` is not Blue"
    _in_range(pp_mean, BLUE_RANGE, "process_polygon `b3` Mean")

    assert not np.array_equal(blue, bc_project["red"].astype(np.float64)), (
        "fixture is degenerate: Red and Blue are identical, so nothing above "
        "could distinguish them")


def test_the_two_provenances_are_the_same_picture(bc_project):
    """Guards the whole file: pair_native.tif and pair_cv2.png must genuinely
    be the same logical image, or 'both provenances agree' means nothing."""
    import cv2
    import tifffile

    from_tif = tifffile.imread(bc_project["tif"])
    from_png_bgr = cv2.imread(bc_project["png"], cv2.IMREAD_UNCHANGED)
    assert np.array_equal(from_tif, from_png_bgr[..., ::-1]), (
        "the .tif and .png fixtures do not hold the same picture")
    # ... and the two files really are read back in OPPOSITE array orders,
    # which is the entire point of the pair.
    assert not np.array_equal(from_tif, from_png_bgr), (
        "cv2 and tifffile returned the same array order for this pair -- the "
        "dual-provenance coverage in this file is not actually exercised")


@pytest.mark.ml
@pytest.mark.editor
def test_both_provenances_yield_the_same_b1(bc_project, ml_manager_factory):
    """THE bug this change set fixed, stated directly: the same picture
    delivered by cv2 and by tifffile must give the SAME `b1`."""
    pt = bc_project["tab"]
    mgr = ml_manager_factory(pt)

    for consumer, call in (
        ("viewer", lambda fp: _viewer_expression(pt, fp, "b1")[0]),
        ("export", lambda fp: _export_expression(pt, fp, "b1")[0]),
        ("editor", lambda fp: _editor_expression(fp, "b1")[0]),
        ("ml", lambda fp: _ml_expression(mgr, fp, "b1")[0]),
    ):
        a = call(bc_project["tif"])
        b = call(bc_project["png"])
        assert np.array_equal(a, b), (
            f"{consumer}: the tifffile-loaded and cv2-loaded copies of the "
            f"SAME image disagree about `b1` ({a[0, 0]} vs {b[0, 0]})")

    # And the two files really do come out of their decoders in different
    # channel orders (tifffile R,G,B; cv2 B,G,R), so the agreement above was
    # earned rather than trivial. (The export normalises every image to file
    # band order now, so its own `_last_export_channel_order` is "rgb" for
    # both and no longer shows the provenance.)
    import cv2
    import tifffile
    t = np.asarray(tifffile.imread(bc_project["tif"]))
    c = cv2.imread(bc_project["png"], cv2.IMREAD_UNCHANGED)
    assert t.shape == c.shape and t.ndim == 3 and t.shape[2] == 3
    assert np.array_equal(t[..., 0], c[..., 2]) and not np.array_equal(t[..., 0], c[..., 0]), (
        "the tifffile and cv2 copies do not arrive in different channel "
        "orders -- the dual-provenance coverage is not exercised")


# ---------------------------------------------------------------------------
# 2. Bands 4+ are NEVER reordered, and b4/b5 address NATIVE bands
# ---------------------------------------------------------------------------
@pytest.mark.ml
@pytest.mark.editor
@pytest.mark.parametrize("band", [1, 2, 3, 4, 5, 6])
def test_stack_band_tokens_address_native_bands(bc_project, ml_manager_factory, band):
    """On a 6-band stack, `bN` must resolve to file band N for every consumer.

    The removed `__B1_TEMP__` string-swap idiom only ever rewrote b1/b3, so
    bands 4+ silently addressed whatever plane happened to sit there after an
    array-level swap elsewhere. Each band here occupies its own disjoint range,
    so a mis-addressed band is unmistakable.
    """
    pt = bc_project["tab"]
    fp = bc_project["stack"]
    expected = bc_project["stack_planes"][band - 1].astype(np.float64)
    expr = f"b{band}"

    viewer, _ = _viewer_expression(pt, fp, expr)
    export, _ = _export_expression(pt, fp, expr)
    editor, _ = _editor_expression(fp, expr)

    mgr = ml_manager_factory(pt)
    ml_expr, _sel, _o = _ml_expression(mgr, fp, expr)

    for name, arr in (("viewer", viewer), ("export", export),
                      ("editor", editor), ("ml", ml_expr)):
        assert np.array_equal(arr, expected), (
            f"{name}: {expr} on the 6-band stack gave {arr[0, 0]}, expected "
            f"file band {band} = {expected[0, 0]}")

    pp_mean, pp_count, _rows = _process_polygon_expression_mean(pt, fp, expr)
    assert pp_count == H * W
    assert pp_mean == pytest.approx(float(expected.mean()), abs=1e-6), (
        f"process_polygon: {expr} Mean {pp_mean} != file band {band} mean "
        f"{expected.mean()}")


@pytest.mark.parametrize("band", [4, 5, 6])
def test_ancillary_bands_survive_a_bgr_first_three_stack(bc_project, band):
    """A >3-band array whose FIRST THREE channels are BGR.

    This is the case the old code got wrong in both directions at once: the
    `C == 3` shape guess refused to swap it (so b1 stayed Blue), while the
    string rewrite would have swapped b1/b3 and left b4+ pointing at the wrong
    plane. Both replays are driven with an explicit `src_channel_order`, which
    is how ThumbnailExportWorker / export_project_images call them.
    """
    from ..project_tab import ProjectTab

    pt = bc_project["tab"]
    fp = bc_project["stack"]
    planes = bc_project["stack_planes"]

    native = np.stack(planes, axis=-1).astype(np.float32)
    bgr_first3 = native.copy()
    bgr_first3[..., [0, 2]] = native[..., [2, 0]]

    expr = f"b{band}"
    ax = {"band_expression": expr, "band_enabled": True}
    _set_ax(pt, fp, ax)
    try:
        expected = planes[band - 1].astype(np.float64)

        # viewer replay, both provenances of the same logical stack
        v_native = np.asarray(ProjectTab.apply_aux_modifications(
            fp, native, pt.project_folder, global_mode=False,
            src_channel_order="rgb"))[..., -1].astype(np.float64)
        v_bgr = np.asarray(ProjectTab.apply_aux_modifications(
            fp, bgr_first3, pt.project_folder, global_mode=False,
            src_channel_order="bgr"))[..., -1].astype(np.float64)

        # export replay, both provenances
        e_native = np.asarray(pt._apply_ax_to_raw(
            native, ax, filepath=fp, src_channel_order="rgb")[0]
        )[..., -1].astype(np.float64)
        e_bgr = np.asarray(pt._apply_ax_to_raw(
            bgr_first3, ax, filepath=fp, src_channel_order="bgr")[0]
        )[..., -1].astype(np.float64)

        for name, arr in (("viewer/rgb", v_native), ("viewer/bgr", v_bgr),
                          ("export/rgb", e_native), ("export/bgr", e_bgr)):
            assert np.array_equal(arr, expected), (
                f"{name}: {expr} gave {arr[0, 0]}, expected band {band} = "
                f"{expected[0, 0]} -- an ancillary band was reordered")
    finally:
        _clear_ax(pt, fp)


def test_bgr_first_three_stack_still_resolves_b1_to_red(bc_project):
    """Companion to the test above for the first three bands: a genuinely-BGR
    wide stack must still give b1=Red / b3=Blue. The old `C == 3` guess left
    these alone entirely, so `b1` stayed Blue on any stack with 4+ bands."""
    from ..project_tab import ProjectTab

    pt = bc_project["tab"]
    fp = bc_project["stack"]
    planes = bc_project["stack_planes"]
    native = np.stack(planes, axis=-1).astype(np.float32)
    bgr_first3 = native.copy()
    bgr_first3[..., [0, 2]] = native[..., [2, 0]]

    for expr, plane_idx in (("b1", 0), ("b3", 2)):
        ax = {"band_expression": expr, "band_enabled": True}
        _set_ax(pt, fp, ax)
        try:
            expected = planes[plane_idx].astype(np.float64)
            got = np.asarray(ProjectTab.apply_aux_modifications(
                fp, bgr_first3, pt.project_folder, global_mode=False,
                src_channel_order="bgr"))[..., -1].astype(np.float64)
            assert np.array_equal(got, expected), (
                f"BGR 6-band stack: {expr} gave {got[0, 0]}, expected "
                f"{expected[0, 0]}")
            got_exp = np.asarray(pt._apply_ax_to_raw(
                bgr_first3, ax, filepath=fp, src_channel_order="bgr")[0]
            )[..., -1].astype(np.float64)
            assert np.array_equal(got_exp, expected), (
                f"BGR 6-band stack, export replay: {expr} gave {got_exp[0, 0]}, "
                f"expected {expected[0, 0]}")
        finally:
            _clear_ax(pt, fp)


def test_to_rgb_band_order_never_moves_bands_beyond_the_third():
    """Unit-level statement of the same rule, so a regression in the shared
    helper is attributed to the helper rather than to six consumers at once."""
    from ..utils import to_rgb_band_order

    a = np.arange(2 * 2 * 7, dtype=np.float32).reshape(2, 2, 7)
    out = np.asarray(to_rgb_band_order(a, "bgr"))
    assert np.array_equal(out[..., 0], a[..., 2])
    assert np.array_equal(out[..., 1], a[..., 1])
    assert np.array_equal(out[..., 2], a[..., 0])
    for c in range(3, 7):
        assert np.array_equal(out[..., c], a[..., c]), f"band {c + 1} was moved"

    assert np.array_equal(np.asarray(to_rgb_band_order(a, "rgb")), a)
    # 2-D / narrow arrays must not raise -- several call sites hand these in.
    for shape in ((4, 4), (4, 4, 1), (4, 4, 2)):
        small = np.zeros(shape, dtype=np.float32)
        assert np.asarray(to_rgb_band_order(small, "bgr")).shape == shape


# ---------------------------------------------------------------------------
# 3. Appended bands must share the SAME b1 namespace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["tif", "png"])
def test_appended_bands_use_the_same_b1_as_the_main_expression(bc_project, which):
    """`appended_bands` entries previously applied NO remap while the .ax's own
    `band_expression` did -- two expressions in the SAME sidecar, evaluated a
    few lines apart in the SAME function, disagreeing about `b1`.

    The boolean entry is chosen so it splits the frame only if `b1` is Red:
    Red is in [200,219] and everything else is below 150, so `b1>150` is
    all-True for Red and all-False for Green or Blue.
    """
    from ..project_tab import ProjectTab

    pt = bc_project["tab"]
    fp = bc_project[which]
    red = bc_project["red"].astype(np.float64)

    ax = {
        "band_expression": "b1", "band_enabled": True,
        "appended_bands": [
            {"type": "band_expression", "index": 4, "expression": "b1"},
            {"type": "boolean_expression", "index": 5, "expression": "b1>150"},
        ],
    }
    _set_ax(pt, fp, ax)
    try:
        lite = pt._imagedata_or_fallback(fp)
        viewer = np.asarray(ProjectTab.apply_aux_modifications(
            fp, lite.image, pt.project_folder, global_mode=False,
            src_channel_order=lite.channel_order)).astype(np.float64)

        img, _i = pt._get_export_image(fp)
        chans = [np.asarray(c, dtype=np.float64)
                 for c in pt._channels_in_export_order(img)]

        assert viewer.shape[2] == 6, (
            f"viewer replay produced {viewer.shape[2]} bands; expected 3 base "
            "+ band_expression + 2 appended")
        assert len(chans) == 6, f"export produced {len(chans)} channels, expected 6"

        # band index 3 = the .ax's own band_expression, 4 = appended numeric,
        # 5 = appended boolean.
        for label, arr in (("viewer main expr", viewer[..., 3]),
                           ("viewer appended band_expression", viewer[..., 4]),
                           ("export main expr", chans[3]),
                           ("export appended band_expression", chans[4])):
            assert np.array_equal(arr, red), (
                f"{label}: `b1` gave {arr[0, 0]}, Red is {red[0, 0]}")

        for label, arr in (("viewer appended boolean_expression", viewer[..., 5]),
                           ("export appended boolean_expression", chans[5])):
            assert float(arr.min()) == 1.0 and float(arr.max()) == 1.0, (
                f"{label}: `b1>150` is {arr.min()}..{arr.max()}; with b1=Red "
                "(>=200) it must be all-True. All-False means b1 resolved to "
                "Green or Blue.")

        # viewer vs export, band by band -- the two replays are separate code.
        for i in (3, 4, 5):
            assert np.array_equal(viewer[..., i], chans[i]), (
                f"viewer and export disagree on derived band index {i}")
    finally:
        _clear_ax(pt, fp)


def test_appended_boolean_expression_is_decisive(bc_project):
    """Anti-vacuum guard for the test above: `b3>150` (Blue, max 49) must be
    all-FALSE, proving the threshold really does discriminate."""
    from ..project_tab import ProjectTab

    pt = bc_project["tab"]
    fp = bc_project["tif"]
    _set_ax(pt, fp, {"appended_bands": [
        {"type": "boolean_expression", "index": 3, "expression": "b3>150"}]})
    try:
        lite = pt._imagedata_or_fallback(fp)
        out = np.asarray(ProjectTab.apply_aux_modifications(
            fp, lite.image, pt.project_folder, global_mode=False,
            src_channel_order=lite.channel_order))
        band = np.asarray(out[..., -1], dtype=np.float64)
        assert float(band.max()) == 0.0, (
            f"`b3>150` is {band.min()}..{band.max()}; Blue tops out at "
            f"{BLUE_RANGE[1]}, so it must be all-False -- b3 is not Blue here")
    finally:
        _clear_ax(pt, fp)


# ---------------------------------------------------------------------------
# 4. Structural guards (SUPPLEMENT to the behavioural tests above)
# ---------------------------------------------------------------------------
# These are AST-based, never string-in-source, and each is verified against a
# deliberately-broken module so it cannot silently stop discriminating.

#: modules that actually run in the app. The `*_A.py` / `csv_test_*.py` scratch
#: files sitting in this package are dead developer copies that nothing
#: imports, and are deliberately NOT scanned.
LIVE_PIXEL_MODULES = [
    "project_tab.py", "image_editor_dialog.py", "machine_learning_manager.py",
    "utils.py", "image_viewer.py", "performance.py", "raster_reader.py",
    "loaders.py", "image_data.py",
]

_REMAP_CALL_NAMES = {"to_rgb_band_order", "_to_rgb_band_order", "_to_rgb"}


def _canopie_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _call_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def find_expression_string_rewrites(source):
    """AST scan for the removed string-rewrite idiom.

    Returns (placeholder_lines, swap_call_lines). A `__B1_TEMP__` string
    literal ANYWHERE, or a call to `swap_b1_b3_tokens` (the migration-only
    helper, which must never appear on a live pixel path), is a reintroduction
    of the band-token rewriting this contract replaced.
    """
    tree = ast.parse(source)
    placeholders, swaps = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "B1_TEMP" in node.value:
            placeholders.append(getattr(node, "lineno", -1))
        elif isinstance(node, ast.Call) and _call_name(node) == "swap_b1_b3_tokens":
            swaps.append(getattr(node, "lineno", -1))
    return placeholders, swaps


def function_calls_remap(source, class_name, func_name):
    """True when `class_name.func_name` contains a call to a to_rgb_band_order
    alias. Returns None when the function is not found at all."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and sub.name == func_name:
                return any(isinstance(n, ast.Call)
                           and _call_name(n) in _REMAP_CALL_NAMES
                           for n in ast.walk(sub))
    return None


@pytest.mark.parametrize("module", LIVE_PIXEL_MODULES)
def test_no_live_module_rewrites_band_tokens(module):
    path = os.path.join(_canopie_dir(), module)
    if not os.path.exists(path):
        pytest.skip(f"{module} not present in this build")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    placeholders, swaps = find_expression_string_rewrites(src)
    assert not placeholders, (
        f"{module}: `__B1_TEMP__` band-token rewriting reintroduced at line(s) "
        f"{placeholders}. Reorder the ARRAY via utils.to_rgb_band_order instead "
        "-- string rewriting only ever handled b1/b3 and silently corrupted b4+.")
    assert not swaps, (
        f"{module}: calls utils.swap_b1_b3_tokens at line(s) {swaps}. That "
        "helper exists only for a one-shot migration of stored expressions and "
        "must never appear on a live pixel path.")


def test_the_band_token_scanner_actually_detects_the_old_idiom():
    """Self-check: run the scanner over a deliberately-broken module so a
    future refactor cannot leave it passing against everything."""
    broken = (
        "import re\n"
        "from .utils import swap_b1_b3_tokens\n"
        "def evaluate(img, expr, C):\n"
        "    if C == 3:\n"
        "        expr = re.sub(r'\\bb1\\b', '__B1_TEMP__', expr)\n"
        "        expr = re.sub(r'\\bb3\\b', 'b1', expr)\n"
        "        expr = re.sub(r'__B1_TEMP__', 'b3', expr)\n"
        "        expr = swap_b1_b3_tokens(expr)\n"
        "    return expr\n"
    )
    placeholders, swaps = find_expression_string_rewrites(broken)
    assert placeholders, "scanner missed the __B1_TEMP__ placeholder"
    assert swaps, "scanner missed the swap_b1_b3_tokens call"

    clean = ("from .utils import to_rgb_band_order\n"
             "def evaluate(img, expr, order):\n"
             "    return to_rgb_band_order(img, order)\n")
    assert find_expression_string_rewrites(clean) == ([], [])


@pytest.mark.parametrize("module,cls,func", [
    ("project_tab.py", "ProjectTab", "apply_aux_modifications"),
    ("project_tab.py", "ProjectTab", "_apply_ax_to_raw"),
    ("image_editor_dialog.py", "ImageEditorDialog", "process_band_expression"),
    ("machine_learning_manager.py", "MachineLearningManager",
     "_select_bands_and_expressions"),
])
def test_b1_resolving_functions_route_through_the_shared_helper(module, cls, func):
    """Supplementary: each function that decides what `b1` means must call the
    ONE provenance-aware helper, not re-derive the order locally."""
    path = os.path.join(_canopie_dir(), module)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    found = function_calls_remap(src, cls, func)
    assert found is not None, f"{cls}.{func} not found in {module}"
    assert found, (
        f"{cls}.{func} no longer calls to_rgb_band_order/_to_rgb -- it is "
        "deciding band order some other way, which is how the C == 3 shape "
        "guess got in.")


def test_the_remap_call_detector_actually_discriminates():
    """Self-check for the detector above, on hand-written good/bad modules."""
    good = ("class K:\n"
            "    def f(self, img, order):\n"
            "        return _to_rgb(img, order)\n")
    bad = ("class K:\n"
           "    def f(self, img, order):\n"
           "        if img.shape[2] == 3:\n"
           "            img = img[..., ::-1]\n"
           "        return img\n")
    assert function_calls_remap(good, "K", "f") is True
    assert function_calls_remap(bad, "K", "f") is False
    assert function_calls_remap(good, "K", "missing") is None
