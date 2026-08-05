"""
Generates the QC synthetic fixture TIFFs (+ paired .ax sidecars) and their
committed ground truth (JSON + NPZ), from the specs in fixtures_manifest.py.

Run directly to (re)generate everything:
    python -m canopie.qc.generate_fixtures [--force]

Conventions confirmed empirically against tifffile + canopie.raster_reader
before this module was written (see the throwaway verification script used
during development):
  - Tile width/height must each be a multiple of 16 (TIFF spec).
  - planarconfig="contig" (BIP) needs the array as (H, W, C).
  - planarconfig="separate" (BSQ) needs the array as (C, H, W) -- passing
    (H, W, C) here silently mislabels axes (tifffile reads shape literally
    against the requested planar layout), which raster_reader.py then
    faithfully (and correctly) reports as a wrong-looking profile. This is
    a fixture-generation concern, not a raster_reader.py bug.
  - GDAL_NODATA is TIFF tag 42113, GDAL_METADATA is 42112, both ASCII text;
    both are only ever read from page 0 by raster_reader.probe(), so they
    only need to be attached to the main (full-resolution) IFD.
  - A COG-style overview pyramid is written via TiffWriter(subifds=N) on the
    main write, followed by N more tif.write(..., subfiletype=1) calls.

process_polygon's NoData masking (see project_tab.py) is asymmetric by
design: for images with 3+ bands, channels 0/1/2 (R/G/B) are always gated by
band 0's own NoData status ("master_nodata_mask = nd_roi[0]"), while bands
index >= 3 are gated by their OWN band's NoData status. This holds
regardless of whether the NoData source was an auto-detected GDAL_NODATA tag
(genuinely per-band masks) or explicit .ax nodata_values (a single any-band
master mask) -- the RGB-vs-extra-bands split happens downstream of that.
MachineLearningManager, by contrast, uniformly excludes a pixel from every
band the instant ANY band matches NoData. Ground truth stores both rules'
resulting stats per band, so fixtures with no real NoData naturally produce
identical numbers under both (a useful cross-check in itself), and fixtures
4/6/7 (which do carry real NoData) produce the documented divergence.
"""
import argparse
import json
import os

import numpy as np
import tifffile

from .fixtures_manifest import (
    FIXTURES,
    FIXTURE_IMAGES_DIR,
    GROUND_TRUTH_DIR,
    raw_value,
    fixture_format,
    fixture_image_path,
    fixture_ax_path,
    ground_truth_json_path,
    ground_truth_npz_path,
)

GDAL_NODATA_TAG = 42113
GDAL_METADATA_TAG = 42112


# ---------------------------------------------------------------------------
# Raw pixel array construction
# ---------------------------------------------------------------------------
def build_raw_array(spec):
    """(H, W, C) float64 array per the deterministic formula, with fill_bands/
    nodata_stamp applied. This IS the ground truth for the un-edited file --
    exactly what a correct reader must report before any .ax transform."""
    H, W, C = spec["height"], spec["width"], spec["bands"]
    modulus = spec["modulus"]
    seed = spec["seed"]

    bands_idx = np.arange(C).reshape(C, 1, 1)
    rows_idx = np.arange(H).reshape(1, H, 1)
    cols_idx = np.arange(W).reshape(1, 1, W)
    raw = (seed * 131 + bands_idx * 977 + rows_idx * 31 + cols_idx * 7) % modulus
    arr = np.transpose(raw, (1, 2, 0)).astype(np.float64)  # (C,H,W) -> (H,W,C)

    # Sanity-check the vectorized form against the scalar formula at a few points.
    for (b, r, c) in [(0, 0, 0), (min(1, C - 1), min(2, H - 1), min(3, W - 1))]:
        assert arr[r, c, b] == raw_value(seed, b, r, c, modulus)

    nodata_value = spec.get("nodata_value")
    if nodata_value is not None:
        for b in spec.get("fill_bands", []):
            arr[:, :, b] = nodata_value
        for (b, r, c) in spec.get("nodata_stamp", []):
            arr[r, c, b] = nodata_value

    return arr


def _gdal_metadata_xml(band_names, count):
    if not band_names:
        return None
    items = []
    for i in range(count):
        nm = band_names.get(i)
        if nm:
            items.append(f'<Item name="DESCRIPTION" sample="{i}" role="description">{nm}</Item>')
    if not items:
        return None
    return "<GDALMetadata>" + "".join(items) + "</GDALMetadata>"


def _downsample(arr_hwc, factor=2):
    return arr_hwc[::factor, ::factor, :]


def write_cv2_image(spec, arr_hwc):
    """Write a PNG/JPEG via cv2, which is what CanoPie will read them back with.

    cv2.imwrite interprets a 3-channel array as BGR, so the array is reversed
    on the way out; that makes file-band 0 == our formula's band 0 == what a
    viewer calls Red, keeping "native file band order" meaning the same thing
    across every fixture regardless of format.
    """
    import cv2

    dtype = np.dtype(spec["dtype"])
    arr_cast = arr_hwc.astype(dtype)
    path = fixture_image_path(spec["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if spec["bands"] == 1:
        out = arr_cast[:, :, 0]
    elif spec["bands"] == 3:
        out = arr_cast[:, :, ::-1]  # native RGB -> cv2's BGR slots
    else:
        raise ValueError(f"{spec['name']}: cv2 formats only support 1 or 3 bands, got {spec['bands']}")

    params = []
    if fixture_format(spec) == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    ok = cv2.imwrite(path, out, params)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    return path


def _array_as_stored(spec):
    """Read the fixture back off disk and return it in NATIVE FILE band order.

    Ground truth is computed from this, not from the generator formula, so
    lossy formats (JPEG) are handled correctly and losslessly-stored formats
    are verified to actually round-trip.
    """
    import cv2

    path = fixture_image_path(spec["name"])
    if fixture_format(spec) == "tiff":
        import tifffile
        from ..raster_reader import ensure_hwc
        with tifffile.TiffFile(path) as tf:
            axes = (tf.series[0].axes or "").upper()
            arr = ensure_hwc(np.squeeze(tf.asarray()), axes=axes)
        return np.asarray(arr, dtype=np.float64)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {path}")
    if img.ndim == 2:
        img = img[:, :, None]
    elif img.shape[2] == 3:
        img = img[:, :, ::-1]  # cv2's BGR -> native RGB
    return np.asarray(img, dtype=np.float64)


def write_tiff(spec, arr_hwc):
    """Write the fixture TIFF exactly per its spec, casting to the fixture dtype."""
    dtype = np.dtype(spec["dtype"])
    arr_cast = arr_hwc.astype(dtype)

    # photometric="rgb" for ANY 3-sample file, regardless of dtype -- verified
    # empirically that cv2.imread collapses a photometric="minisblack" 3-sample
    # TIFF down to a 2D (single-channel) array, silently losing 2 of 3 bands.
    # Only 3-band fixtures need this; see fixture 3/4's notes in
    # fixtures_manifest.py for why BSQ was moved off the 3-band fixture.
    photometric = "rgb" if spec["bands"] == 3 else "minisblack"
    extratags = []
    if spec.get("nodata_value") is not None:
        extratags.append((GDAL_NODATA_TAG, "s", 0, repr(float(spec["nodata_value"])), False))
    meta_xml = _gdal_metadata_xml(spec.get("band_names"), spec["bands"])
    if meta_xml:
        extratags.append((GDAL_METADATA_TAG, "s", 0, meta_xml, False))

    path = fixture_image_path(spec["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_kwargs = dict(photometric=photometric, planarconfig=spec["planarconfig"])
    if spec["tiled"]:
        write_kwargs["tile"] = spec["tile"]

    if spec["planarconfig"] == "separate":
        write_arr = np.ascontiguousarray(np.transpose(arr_cast, (2, 0, 1)))  # -> (C,H,W)
    else:
        write_arr = np.ascontiguousarray(arr_cast)  # (H,W,C)

    n_overviews = int(spec.get("overview_levels", 0) or 0)
    if n_overviews > 0:
        with tifffile.TiffWriter(path) as tif:
            tif.write(write_arr, subifds=n_overviews, extratags=extratags, **write_kwargs)
            level_arr = arr_cast
            for _ in range(n_overviews):
                level_arr = _downsample(level_arr, 2)
                lv_write = (np.ascontiguousarray(np.transpose(level_arr, (2, 0, 1)))
                            if spec["planarconfig"] == "separate"
                            else np.ascontiguousarray(level_arr))
                lv_kwargs = dict(write_kwargs)
                if spec["tiled"]:
                    # Overview levels may be smaller than one tile; tifffile
                    # requires the level itself be a whole number of tiles or
                    # it auto-pads -- untiled is simplest and still parses
                    # fine as an additional pyramid level.
                    th, tw = level_arr.shape[0], level_arr.shape[1]
                    if th < spec["tile"][1] or tw < spec["tile"][0]:
                        lv_kwargs.pop("tile", None)
                tif.write(lv_write, subfiletype=1, **lv_kwargs)
    else:
        tifffile.imwrite(path, write_arr, extratags=extratags, **write_kwargs)

    return path


def write_ax(spec):
    ax = spec.get("ax")
    if not ax:
        return None
    path = fixture_ax_path(spec["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ax, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Ground-truth computation (plain numpy only -- never calls into canopie).
# ---------------------------------------------------------------------------
def _rasterize_polygon_mask(points, H, W):
    """Match process_polygon's own rasterization exactly enough for ground
    truth purposes: cv2.fillPoly on the polygon's local bounding box."""
    import cv2
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(min(xs))); y0 = max(0, int(min(ys)))
    x1 = min(W - 1, int(max(xs))); y1 = min(H - 1, int(max(ys)))
    x1 += 1; y1 += 1
    mask = np.zeros((H, W), dtype=bool)
    if len(points) >= 3:
        roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        pts = np.array([[int(round(x - x0)), int(round(y - y0))] for (x, y) in points], dtype=np.int32)
        import cv2 as _cv2
        _cv2.fillPoly(roi, [pts.reshape((-1, 1, 2))], 255)
        mask[y0:y1, x0:x1] = roi != 0
    elif len(points) == 1:
        xi, yi = int(points[0][0]), int(points[0][1])
        if 0 <= xi < W and 0 <= yi < H:
            mask[yi, xi] = True
    return mask


def _stat_block(values_1d):
    """Mirrors what ProjectTab._calc_stats ACTUALLY computes in practice.

    _calc_stats unconditionally delegates to performance.fast_stats whenever
    the performance module is importable at all (project_tab.py:11903:
    `if _PERF_MODULE_AVAILABLE and array_1d is not None and ... size > 0`,
    no size threshold, no config toggle) -- and _PERF_MODULE_AVAILABLE only
    requires performance.py itself to import, which needs nothing beyond
    numpy (its numba/numexpr accelerators are individually try/excepted
    inside that module, not required for import). So this is not a rare
    "if the fast path happens to be installed" case -- it is what runs in
    essentially every real install, including this one (confirmed directly:
    fast_stats' Median is float(np.median(...)) -- NumPy's own even-length
    average-of-two-middle-values definition -- and its Q{n} quantiles are
    float(np.percentile(..., q)) with NumPy's *default* ('linear') interpolation.
    This differs from _calc_stats' own plain-Python fallback (only reached
    when performance.py fails to import at all), which instead uses
    np.nanpercentile(..., method='nearest') for both. Ground truth mirrors
    the fast_stats path since that's what real runs actually hit; a plain-
    numpy-only environment without performance.py at all would need
    different ground truth for Median/Q25/Q75 specifically (Mean/Std/Count
    are identical either way)."""
    n = int(values_1d.size)
    if n == 0:
        return {"mean": None, "median": None, "std": None, "q25": None, "q75": None, "count": 0}
    return {
        "mean": float(np.mean(values_1d)),
        "median": float(np.median(values_1d)),
        "std": float(np.std(values_1d, ddof=0)),
        "q25": float(np.percentile(values_1d, 25)),
        "q75": float(np.percentile(values_1d, 75)),
        "count": n,
    }


def _nodata_equal(arr_band, nodata_value, tol=1e-3):
    if nodata_value is None:
        return np.zeros(arr_band.shape, dtype=bool)
    return np.isclose(arr_band, nodata_value, atol=max(tol, abs(nodata_value) * 1e-4))


def compute_polygon_ground_truth(arr_hwc, poly_mask, nodata_value, ax_nodata_value=None):
    """Per-band stats under EACH consumer's real masking rule.

    The two consumers do NOT see the same NoData values, which matters more
    than the (also real) difference in how they combine them across bands:

    - `process_polygon` resolves NoData via ProjectTab.effective_nodata_values,
      which merges the .ax `nodata_values` with the raster's own GDAL_NODATA
      TAG. It then applies them asymmetrically: bands 0/1/2 are gated by band
      0's own NoData status (whenever the file has 3+ bands), while bands >= 3
      are gated by their own band.

    - `MachineLearningManager.export_csv_data` reads `nodata_values` ONLY from
      the .ax sidecar (machine_learning_manager.py's own ax_candidates loop) --
      it never consults GDAL_NODATA. It then applies them whole-pixel: ANY
      band matching invalidates that pixel for every band.

    Consequence, verified empirically against these fixtures: for a raster
    that declares its fill value ONLY via the GDAL_NODATA tag and has no .ax
    (fixtures 4 and 6), ML-manager CSV export applies NO masking at all and
    averages the raw fill values into its means -- e.g. mean_R = -2034.5 on
    nodata_fragmented_multiband, where the -9999 holes drag the mean far below
    any real pixel value, versus process_polygon's correctly-masked 620.33.
    That is current behavior, encoded here as the regression baseline; see the
    suite's README/notes for it being flagged rather than silently fixed.

    `no_mask` is also emitted so a test can assert the unmasked value directly
    when that IS the expected result.
    """
    H, W, C = arr_hwc.shape
    pixel_count_total = int(poly_mask.sum())

    band_nodata = np.zeros((C, H, W), dtype=bool)
    if nodata_value is not None:
        for b in range(C):
            band_nodata[b] = _nodata_equal(arr_hwc[:, :, b], nodata_value)

    # ML manager sees ONLY .ax-declared NoData, never the GDAL_NODATA tag.
    ml_band_nodata = np.zeros((C, H, W), dtype=bool)
    if ax_nodata_value is not None:
        for b in range(C):
            ml_band_nodata[b] = _nodata_equal(arr_hwc[:, :, b], ax_nodata_value)
    any_band_invalid = np.any(ml_band_nodata, axis=0)

    out = {"pixel_count_total": pixel_count_total, "bands": {}}
    for b in range(C):
        if C >= 3 and b < 3:
            pp_invalid = band_nodata[0] if nodata_value is not None else np.zeros((H, W), dtype=bool)
        else:
            pp_invalid = band_nodata[b] if nodata_value is not None else np.zeros((H, W), dtype=bool)
        pp_valid_mask = poly_mask & (~pp_invalid)
        ml_valid_mask = poly_mask & (~any_band_invalid)

        out["bands"][str(b)] = {
            "process_polygon": _stat_block(arr_hwc[:, :, b][pp_valid_mask]),
            "ml_manager": _stat_block(arr_hwc[:, :, b][ml_valid_mask]),
            "no_mask": _stat_block(arr_hwc[:, :, b][poly_mask]),
        }
    return out


def compute_point_values(arr_hwc, x, y):
    H, W, C = arr_hwc.shape
    xi, yi = int(x), int(y)
    if not (0 <= xi < W and 0 <= yi < H):
        return None
    return [float(arr_hwc[yi, xi, b]) for b in range(C)]


# ---------------------------------------------------------------------------
# Per-fixture ground truth assembly
# ---------------------------------------------------------------------------
def _crop_array(arr_hwc, crop_rect):
    x0, y0 = crop_rect["x"], crop_rect["y"]
    w, h = crop_rect["width"], crop_rect["height"]
    return arr_hwc[y0:y0 + h, x0:x0 + w, :]


def build_ground_truth(spec, raw_arr_hwc):
    """raw_arr_hwc is the array as written to disk (pre-.ax). For fixtures
    with an .ax, the ground truth reflects the POST-edit array, since that is
    what process_polygon/Inspect/ImageViewer all actually read -- see
    ProjectTab._get_export_image / _apply_ax_to_raw."""
    ax = spec.get("ax")
    working = raw_arr_hwc
    extra = {}

    if ax and "crop_rect" in ax:
        working = _crop_array(raw_arr_hwc, ax["crop_rect"])
        extra["crop_rect"] = ax["crop_rect"]

    if ax and "band_expression" in ax:
        # b{i+1} = file band i (0-based), no BGR remap for band counts != 3
        # (see project_tab.py _do_band_expr: the b1<->b3 swap only triggers
        # when the array has exactly 3 channels -- confirmed by direct read).
        C = working.shape[2]
        mapping = {f"b{i+1}": working[:, :, i] for i in range(C)}
        expr = ax["band_expression"]
        # Deliberately reimplemented with plain numpy (not utils.eval_band_expression)
        # to avoid a tautological ground truth.
        b1, b4 = mapping["b1"], mapping["b4"]
        assert expr.replace(" ", "") == "(b4-b1)/(b4+b1)", f"generator only knows this one expression: {expr}"
        with np.errstate(divide="ignore", invalid="ignore"):
            index_band = np.where((b4 + b1) != 0, (b4 - b1) / (b4 + b1), 0.0)
        working = np.concatenate([working, index_band[..., None]], axis=2)
        extra["band_expression_appended_index"] = spec["bands"]  # new band's index

    nodata_value = spec.get("nodata_value")

    points_gt = {}
    for p in spec["points"]:
        vals = compute_point_values(working, p["x"], p["y"])
        points_gt[p["name"]] = {"x": p["x"], "y": p["y"], "values": vals}

    poly = spec["polygon"]
    H, W = working.shape[:2]
    poly_mask = _rasterize_polygon_mask(poly["points"], H, W)
    # The .ax-declared NoData (if any) is the ONLY NoData MachineLearningManager
    # ever sees -- see compute_polygon_ground_truth's docstring.
    _ax_nd_list = (ax or {}).get("nodata_values") or []
    ax_nodata_value = None
    for _v in _ax_nd_list:
        try:
            ax_nodata_value = float(_v)
            break
        except (TypeError, ValueError):
            continue  # string expressions like "b1>182" aren't numeric fills

    poly_gt = compute_polygon_ground_truth(working, poly_mask, nodata_value,
                                           ax_nodata_value=ax_nodata_value)
    poly_gt["name"] = poly["name"]
    poly_gt["points"] = poly["points"]

    return {
        "name": spec["name"],
        "dtype": spec["dtype"],
        "raw_shape": list(raw_arr_hwc.shape),
        "working_shape": list(working.shape),
        "seed": spec["seed"],
        "modulus": spec["modulus"],
        "nodata_value": nodata_value,
        "format": fixture_format(spec),
        # _get_export_image's _read_raw_any prefers tifffile for .tif/.tiff
        # (native "rgb" order); any other extension skips straight to
        # cv2.imread, which genuinely is BGR.
        "export_channel_order": "rgb" if fixture_format(spec) == "tiff" else "bgr",
        # _imagedata_or_fallback (ImageViewer/Inspect) routes through tifffile
        # only when _tifffile_is_stack is true (>3 bands or >3 pages); anything
        # else goes through ImageData/cv2.imread, which returns "bgr".
        # Confirmed by direct read of _tifffile_is_stack (project_tab.py) --
        # the TIFF fixtures were deliberately designed (see fixtures 3/4's
        # notes) so every <=3-band one uses photometric="rgb" + BIP, which
        # cv2 reads correctly.
        "viewer_channel_order": "rgb" if spec["bands"] > 3 else "bgr",
        "points": points_gt,
        "polygon": poly_gt,
        "degenerate_point_name": spec["degenerate_point_name"],
        **extra,
    }


# ---------------------------------------------------------------------------
# Self-check against raster_reader.py -- catches format mistakes immediately.
# ---------------------------------------------------------------------------
def _self_check(spec, raw_arr_hwc, tif_path):
    if fixture_format(spec) != "tiff":
        # raster_reader is TIFF-only by design; for cv2 formats the meaningful
        # check is that the file decodes at all with the expected geometry --
        # exact values are asserted separately via _array_as_stored, which is
        # what ground truth is built from (and which, for JPEG, legitimately
        # differs from the formula).
        stored = _array_as_stored(spec)
        assert stored.shape[:2] == (spec["height"], spec["width"]), \
            f"{spec['name']}: decoded dims {stored.shape[:2]} != {(spec['height'], spec['width'])}"
        assert stored.shape[2] == spec["bands"], \
            f"{spec['name']}: decoded band count {stored.shape[2]} != {spec['bands']}"
        if fixture_format(spec) == "png":
            # PNG is lossless -- it MUST round-trip the formula exactly, and if
            # it ever doesn't, that's a real bug in how the fixture is written.
            assert np.array_equal(stored, raw_arr_hwc), \
                f"{spec['name']}: lossless PNG did not round-trip exactly"
        return

    from ..raster_reader import probe, open_reader

    profile = probe(tif_path)
    assert profile is not None, f"raster_reader.probe() failed to read {tif_path}"
    assert profile.count == spec["bands"], f"{spec['name']}: band count {profile.count} != {spec['bands']}"
    assert profile.width == spec["width"] and profile.height == spec["height"], \
        f"{spec['name']}: dims {profile.width}x{profile.height} != {spec['width']}x{spec['height']}"
    assert profile.tiled == spec["tiled"], f"{spec['name']}: tiled={profile.tiled} != {spec['tiled']}"

    reader = open_reader(tif_path, profile)
    window = reader.read_window(0, 0, spec["width"], spec["height"])
    window = np.asarray(window, dtype=np.float64)
    if spec["dtype"].startswith("float"):
        ok = np.allclose(window, raw_arr_hwc, atol=1e-3)
    else:
        ok = np.array_equal(window.astype(np.dtype(spec["dtype"])), raw_arr_hwc.astype(np.dtype(spec["dtype"])))
    assert ok, f"{spec['name']}: raster_reader round-trip does not match the array that was written"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_one(spec, force=False):
    tif_path = fixture_image_path(spec["name"])
    gt_json_path = ground_truth_json_path(spec["name"])
    gt_npz_path = ground_truth_npz_path(spec["name"])

    if (not force) and os.path.exists(tif_path) and os.path.exists(gt_json_path):
        return False  # already present, nothing to do

    raw_arr = build_raw_array(spec)
    if fixture_format(spec) == "tiff":
        write_tiff(spec, raw_arr)
    else:
        write_cv2_image(spec, raw_arr)
    write_ax(spec)
    _self_check(spec, raw_arr, tif_path)

    # Ground truth comes from what is ACTUALLY STORED on disk, not from the
    # generator formula. Identical for lossless formats (asserted in
    # _self_check), and the only correct choice for lossy JPEG.
    stored_arr = _array_as_stored(spec)
    gt = build_ground_truth(spec, stored_arr)

    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    np.savez_compressed(gt_npz_path, raw=stored_arr.astype(np.float32))
    return True


def generate_all(force=False):
    os.makedirs(FIXTURE_IMAGES_DIR, exist_ok=True)
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    generated = []
    for spec in FIXTURES:
        if generate_one(spec, force=force):
            generated.append(spec["name"])
    return generated


def fixtures_missing():
    """True if any fixture image or ground-truth file is absent."""
    for spec in FIXTURES:
        if not os.path.exists(fixture_image_path(spec["name"])):
            return True
        if not os.path.exists(ground_truth_json_path(spec["name"])):
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate all fixtures even if present.")
    args = parser.parse_args()
    made = generate_all(force=args.force)
    print(f"Generated {len(made)} fixture(s): {made}" if made else "All fixtures already present, nothing to do.")
