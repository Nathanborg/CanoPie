"""
Single source of truth for the QC synthetic fixture matrix.

Both `generate_fixtures.py` (writes the .tif/.ax files and computes ground
truth) and every `test_*.py` module import from here -- fixture specs,
polygon/point definitions, and the deterministic pixel formula all live in
exactly one place so the generator and the tests can never silently drift
apart.

See the plan at the top of this package's docstring intent: 8 small,
tifffile-written fixtures covering bit depth (8/12-in-16/16), band count
(3/8/200), tiling structure (untiled/tiled-BSQ/tiled-BIP+overview), a
fragmented NoData footprint, and two `.ax`-edited variants (crop+nodata,
band_expression).
"""
import os

QC_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_IMAGES_DIR = os.path.join(QC_DIR, "fixture_images")
GROUND_TRUTH_DIR = os.path.join(QC_DIR, "ground_truth")


# ---------------------------------------------------------------------------
# Deterministic pixel formula -- used ONLY by generate_fixtures.py, never by
# the tests themselves (tests assert against the committed ground truth).
# ---------------------------------------------------------------------------
def raw_value(seed, band, row, col, modulus):
    """A simple, hand-verifiable formula: distinct per (seed, band, row, col)."""
    return (seed * 131 + band * 977 + row * 31 + col * 7) % modulus


# ---------------------------------------------------------------------------
# Fixture specs
# ---------------------------------------------------------------------------
# Each entry:
#   name          -- file stem (also the ground-truth file stem)
#   bands, height, width, dtype (numpy dtype name)
#   modulus       -- value range the formula wraps into (matches dtype range,
#                    or a restricted sub-range for the 12-in-16 fixture)
#   seed          -- distinct per fixture
#   tiled         -- bool; False => tifffile writes strips
#   tile          -- (tile_w, tile_h) when tiled
#   planarconfig  -- "contig" (BIP) or "separate" (BSQ)
#   overview_levels -- int, extra reduced-resolution IFDs to append (COG-style)
#   nodata_value  -- numeric GDAL_NODATA value, or None
#   fill_bands    -- band indices that are 100% nodata_value (ancillary planes)
#   nodata_stamp  -- explicit [(band, row, col), ...] cells forced to nodata_value,
#                    on top of any fill_bands, for deterministic per-pixel checks
#   band_names    -- optional {band_index: "name"} for GDAL_METADATA coverage
#   ax            -- optional dict written as the paired .ax sidecar
#   points        -- list of {"name","x","y"} spot points (image pixel coords,
#                    used both for "point" polygon_dicts and as spot-checks)
#   polygon       -- {"name","points":[(x,y),...]} one small interior polygon
#   degenerate_point_name -- which entry in `points` also gets a co-located
#                    1-vertex "polygon" (for point-vs-1px-polygon consistency)

FIXTURES = [
    # ---- 1: baseline 8-bit RGB, untiled -----------------------------------
    {
        "name": "rgb_8bit_untiled",
        "bands": 3, "height": 64, "width": 64, "dtype": "uint8",
        "modulus": 256, "seed": 101,
        "tiled": False, "tile": None, "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 4, "y": 4},
            {"name": "p_center", "x": 32, "y": 32},
            {"name": "p_br", "x": 59, "y": 59},
        ],
        "polygon": {"name": "poly_interior", "points": [(10, 10), (25, 10), (25, 25), (10, 25)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 2: 16-bit, tiled BIP, +1 overview (COG-style) --------------------
    {
        "name": "rgb_16bit_tiled_bip_cog",
        "bands": 3, "height": 96, "width": 96, "dtype": "uint16",
        "modulus": 65536, "seed": 202,
        "tiled": True, "tile": (32, 32), "planarconfig": "contig",
        "overview_levels": 1,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 5, "y": 5},
            {"name": "p_center", "x": 48, "y": 48},
            {"name": "p_br", "x": 90, "y": 90},
        ],
        "polygon": {"name": "poly_interior", "points": [(20, 20), (45, 20), (45, 45), (20, 45)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 3: 12-in-16-bit, tiled BIP ----------------------------------------
    # NOTE: BIP (not BSQ) is deliberate here -- cv2.imread cannot read
    # planarconfig="separate" (BSQ) TIFFs at all for <=3-band files (verified
    # empirically: it silently returns a wrong, single-repeated-band array),
    # and _tifffile_is_stack's >3-band heuristic means a 3-band file never
    # gets routed to the (BSQ-safe) tifffile path for display. BSQ coverage
    # lives on fixture 4 instead (8 bands -- always tifffile-routed, so BSQ
    # is read correctly there for both the viewer and export paths).
    {
        "name": "rgb_12in16_tiled_bip",
        "bands": 3, "height": 64, "width": 64, "dtype": "uint16",
        "modulus": 4096, "seed": 303,
        "tiled": True, "tile": (32, 32), "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 3, "y": 3},
            {"name": "p_center", "x": 32, "y": 32},
            {"name": "p_br", "x": 60, "y": 60},
        ],
        "polygon": {"name": "poly_interior", "points": [(8, 8), (20, 8), (20, 20), (8, 20)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 4: 8-band float32, tiled BSQ, band 7 (0-idx) 100% fill -----------
    # BSQ (planarconfig="separate") lives here rather than on a 3-band RGB
    # fixture -- see fixture 3's note above for why. 8 bands always routes
    # through tifffile (never cv2) for both the viewer and export paths, so
    # BSQ is read correctly here.
    {
        "name": "multiband_8band_ancillary",
        "bands": 8, "height": 48, "width": 48, "dtype": "float32",
        "modulus": 1000, "seed": 404,
        "tiled": True, "tile": (16, 16), "planarconfig": "separate",
        "overview_levels": 0,
        "nodata_value": -9999.0, "fill_bands": [7], "nodata_stamp": [],
        "band_names": {i: f"synband_{i+1}" for i in range(8)},
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 3, "y": 3},
            {"name": "p_center", "x": 24, "y": 24},
            {"name": "p_br", "x": 44, "y": 44},
        ],
        "polygon": {"name": "poly_interior", "points": [(10, 10), (22, 10), (22, 22), (10, 22)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 5: 200-band float32, tiled BIP (lazy/eager agreement fixture) ----
    {
        "name": "hyperspectral_200band",
        "bands": 200, "height": 32, "width": 32, "dtype": "float32",
        "modulus": 1000, "seed": 505,
        "tiled": True, "tile": (16, 16), "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": -9999.0, "fill_bands": [], "nodata_stamp": [],
        "band_names": {0: "band_001_400nm", 50: "band_051_550nm", 199: "band_200_900nm"},
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 2, "y": 2},
            {"name": "p_center", "x": 16, "y": 16},
            {"name": "p_br", "x": 29, "y": 29},
        ],
        "polygon": {"name": "poly_interior", "points": [(6, 6), (14, 6), (14, 14), (6, 14)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 6: 8-band float32, fragmented NoData footprint -------------------
    {
        "name": "nodata_fragmented_multiband",
        "bands": 8, "height": 40, "width": 40, "dtype": "float32",
        "modulus": 1000, "seed": 606,
        "tiled": True, "tile": (16, 16), "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": -9999.0, "fill_bands": [],
        # Two disconnected 10x10 holes: rows/cols [5,15) and [25,35).
        # Stamped across ALL bands at every (row,col) in each hole -- a genuine
        # invalid-area footprint, not a single all-fill ancillary band.
        "nodata_stamp": [
            (b, r, c)
            for b in range(8)
            for r in range(5, 15) for c in range(5, 15)
        ] + [
            (b, r, c)
            for b in range(8)
            for r in range(25, 35) for c in range(25, 35)
        ],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_valid", "x": 2, "y": 2},
            {"name": "p_in_hole_a", "x": 10, "y": 10},
            {"name": "p_in_hole_b", "x": 30, "y": 30},
        ],
        # Straddles hole A (rows/cols 5-14): polygon spans rows/cols 0-19,
        # so it covers both valid pixels and the entire hole A.
        "polygon": {"name": "poly_straddle_hole_a", "points": [(0, 0), (19, 0), (19, 19), (0, 19)]},
        "degenerate_point_name": "p_valid",
    },
    # ---- 7: .ax crop + nodata (stays lazy-eligible) ------------------------
    {
        "name": "ax_crop_nodata_source",
        "bands": 4, "height": 80, "width": 80, "dtype": "uint16",
        "modulus": 65536, "seed": 707,
        "tiled": True, "tile": (16, 16), "planarconfig": "contig",
        "overview_levels": 0,
        # NoData value stamped only within the post-crop region (rows/cols 20-59),
        # at two known post-crop-relative cells, so ground truth for the CROPPED
        # image can assert exact exclusion. (row/col below are RAW/pre-crop coords.)
        "nodata_value": 9999.0, "fill_bands": [],
        "nodata_stamp": [(b, 25, 25) for b in range(4)] + [(b, 45, 45) for b in range(4)],
        "band_names": None,
        "ax": {
            "crop_rect": {"x": 20, "y": 20, "width": 40, "height": 40},
            "crop_rect_ref_size": {"w": 80, "h": 80},
            "crop_enabled": True,
            "nodata_values": [9999],
            "nodata_enabled": True,
        },
        # Points given in the CROPPED (post-.ax) frame, since that's the frame
        # process_polygon/_get_export_image actually operate on for this file.
        "points": [
            {"name": "p_tl", "x": 2, "y": 2},
            {"name": "p_center", "x": 20, "y": 20},
            {"name": "p_at_nodata_1", "x": 5, "y": 5},    # raw (25,25) -> cropped (5,5)
            {"name": "p_at_nodata_2", "x": 25, "y": 25},  # raw (45,45) -> cropped (25,25)
        ],
        "polygon": {"name": "poly_interior", "points": [(8, 8), (16, 8), (16, 16), (8, 16)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 8: .ax band_expression (blocks windowing) -------------------------
    {
        "name": "ax_band_expression_source",
        "bands": 4, "height": 40, "width": 40, "dtype": "uint16",
        "modulus": 65536, "seed": 808,
        "tiled": False, "tile": None, "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": {0: "Red", 1: "Green", 2: "Blue", 3: "NIR"},
        "ax": {
            # NDVI-style: (NIR - Red) / (NIR + Red) = (b4 - b1) / (b4 + b1)
            "band_expression": "(b4-b1)/(b4+b1)",
            "band_enabled": True,
        },
        "points": [
            {"name": "p_tl", "x": 2, "y": 2},
            {"name": "p_center", "x": 20, "y": 20},
            {"name": "p_br", "x": 37, "y": 37},
        ],
        "polygon": {"name": "poly_interior", "points": [(8, 8), (18, 8), (18, 18), (8, 18)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 9: 8-bit RGB PNG (lossless, non-TIFF) ----------------------------
    # Exercises the OTHER branch of _get_export_image._read_raw_any: a
    # non-.tif extension skips tifffile entirely and goes to cv2.imread, which
    # really does return BGR -- so this is the fixture that proves the
    # channel_order fix still applies the [2,1,0] swap where it IS correct,
    # complementing the TIFF fixtures that prove it is NOT applied where it
    # would be wrong. Also the only format here that a phenocam/consumer
    # camera workflow would actually produce.
    {
        "name": "rgb_8bit_png",
        "format": "png",
        "bands": 3, "height": 48, "width": 48, "dtype": "uint8",
        "modulus": 256, "seed": 909,
        "tiled": False, "tile": None, "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 3, "y": 3},
            {"name": "p_center", "x": 24, "y": 24},
            {"name": "p_br", "x": 44, "y": 44},
        ],
        "polygon": {"name": "poly_interior", "points": [(10, 10), (22, 10), (22, 22), (10, 22)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 10: 8-bit RGB JPEG (LOSSY) ---------------------------------------
    # JPEG does not round-trip exactly, by design. Ground truth for every
    # non-TIFF fixture is therefore taken from the DECODED file rather than
    # from the generator formula (see generate_fixtures._array_as_stored), so
    # this fixture still asserts exact equality against real stored values --
    # it just can't assume those equal the formula. That makes it a genuine
    # regression guard for "did our decode path change?" without being
    # brittle about JPEG's own quantization.
    {
        "name": "rgb_8bit_jpeg",
        "format": "jpeg",
        "bands": 3, "height": 48, "width": 48, "dtype": "uint8",
        "modulus": 256, "seed": 1010,
        "tiled": False, "tile": None, "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 3, "y": 3},
            {"name": "p_center", "x": 24, "y": 24},
            {"name": "p_br", "x": 44, "y": 44},
        ],
        "polygon": {"name": "poly_interior", "points": [(10, 10), (22, 10), (22, 22), (10, 22)]},
        "degenerate_point_name": "p_center",
    },
    # ---- 11: single-band 8-bit grayscale PNG ------------------------------
    # Fills a real coverage gap: every other fixture has >= 3 bands, so the
    # C == 1 code paths were entirely untested -- and they are genuinely
    # separate branches (Inspect's `ch_names = ["b1"]` / `img_mod.ndim == 2`
    # case, _channels_in_export_order's `a.ndim == 2 -> return [a]`,
    # process_polygon's non-RGB branch with its "Gray" channel label, and
    # export_csv_data's `channel_N` naming instead of R/G/B).
    {
        "name": "gray_8bit_png",
        "format": "png",
        "bands": 1, "height": 40, "width": 40, "dtype": "uint8",
        "modulus": 256, "seed": 1111,
        "tiled": False, "tile": None, "planarconfig": "contig",
        "overview_levels": 0,
        "nodata_value": None, "fill_bands": [], "nodata_stamp": [],
        "band_names": None,
        "ax": None,
        "points": [
            {"name": "p_tl", "x": 3, "y": 3},
            {"name": "p_center", "x": 20, "y": 20},
            {"name": "p_br", "x": 36, "y": 36},
        ],
        "polygon": {"name": "poly_interior", "points": [(8, 8), (20, 8), (20, 20), (8, 20)]},
        "degenerate_point_name": "p_center",
    },
]


def get_fixture(name):
    for f in FIXTURES:
        if f["name"] == name:
            return f
    raise KeyError(f"No fixture named {name!r}")


#: file extension per fixture "format" (specs default to "tiff" when absent)
FORMAT_EXT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}


def fixture_format(spec_or_name):
    spec = get_fixture(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    return spec.get("format", "tiff")


def fixture_image_path(name):
    ext = FORMAT_EXT[fixture_format(name)]
    return os.path.join(FIXTURE_IMAGES_DIR, f"{name}{ext}")


def fixture_ax_path(name):
    return os.path.join(FIXTURE_IMAGES_DIR, f"{name}.ax")


def ground_truth_json_path(name):
    return os.path.join(GROUND_TRUTH_DIR, f"{name}.json")


def ground_truth_npz_path(name):
    return os.path.join(GROUND_TRUTH_DIR, f"{name}.npz")
