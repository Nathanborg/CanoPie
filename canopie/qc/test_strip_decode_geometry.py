"""
QC regression tests for STRIP/TILE decode geometry in raster_reader.

THE BUG THIS PINS (found on a real user project, C:\\PCA_climate_soil_panama,
reading G:\\Meu Drive\\GEE_Exports\\pca_scores_Panama_soil_terrain.tif --
2031x2350x20 float32, LZW, untiled, planarconfig=1, **rowsperstrip=1**):

    could not broadcast input array from shape (1,20) into shape (1,2350)

``TiffPage.decode`` always returns four axes, ``(depth, height, width,
samples)``. ``_decode_uncached`` normalised that with a blind ``np.squeeze``,
which drops EVERY length-1 axis -- including a geometry axis that is
legitimately 1. With ``rowsperstrip == 1`` a strip decodes as
``(1, 1, W, S)``, so squeeze produced ``(W, S)``: the WIDTH axis silently
took the ROW axis's place. The follow-up ``arr[..., None]`` then made it
``(W, S, 1)``, and ``read_window`` sliced rows out of the width axis --
raising the error above for EVERY windowed read of such a file.

WHY IT MATTERED BEYOND THE EXCEPTION: callers catch that failure and fall
back to full-resolution reads. On a 20-band Google-Drive-hosted raster that
turned a 0.06 s sampled read into a multi-second full read on the GUI
thread, which is what made "opening the Image Editor dialog" feel frozen.
The exception itself was only logged as a warning, so the symptom the user
actually reported was slowness, not an error.

WHY THE EXISTING FIXTURES MISSED IT: every fixture in this suite is either
tiled (16x16/32x32/64x64) or has a multi-row strip, so no fixture had a
geometry axis of length 1 and the squeeze always happened to be harmless.
One row per strip is not exotic -- it is what GDAL and Google Earth Engine
routinely emit for GeoTIFF exports.
"""
import numpy as np
import pytest
import tifffile

from ..raster_reader import (
    TiledRasterReader,
    STRATEGY_STRIPS,
    clear_reader_cache,
    open_reader,
    probe,
)

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
# raster_reader.py underpins every consumer, so changes here must also run
# with `-m "extraction or contract"`.
pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _write_row_per_strip(path, h=9, w=23, bands=5, seed=0):
    """A multiband, contig (planar=1), LZW, untiled TIFF with ONE ROW PER
    STRIP -- the exact layout that reproduced the bug."""
    rng = np.random.default_rng(seed)
    arr = (rng.uniform(0, 1000, size=(h, w, bands))).astype(np.float32)
    arr[0, :, :] = -9999.0          # a fill row, so NoData handling is exercised too
    tifffile.imwrite(str(path), arr, planarconfig="contig", rowsperstrip=1,
                     compression="zlib")
    return arr


def _write_single_band_row_per_strip(path, h=7, w=13, seed=1):
    rng = np.random.default_rng(seed)
    arr = (rng.uniform(0, 1000, size=(h, w))).astype(np.float32)
    tifffile.imwrite(str(path), arr, rowsperstrip=1, compression="zlib")
    return arr


# ---------------------------------------------------------------------------
# Segment geometry -- the root cause, asserted directly
# ---------------------------------------------------------------------------
def test_row_per_strip_segment_keeps_its_row_axis(tmp_path):
    """THE root cause. A decoded segment must be (h, w, samples) for planar=1,
    with the row axis intact even when it is length 1."""
    fp = tmp_path / "rowstrip.tif"
    _write_row_per_strip(fp)

    clear_reader_cache()
    profile = probe(fp)
    assert profile is not None and profile.strategy == STRATEGY_STRIPS, (
        f"fixture did not take the strip path (strategy={getattr(profile,'strategy',None)})")

    reader = open_reader(fp, profile)
    lv = reader._levels[0]
    assert lv["th"] == 1, "fixture is not one row per strip"

    seg = reader._decode_uncached(0, 0)
    assert seg.shape == (lv["th"], lv["tw"], lv["count"]), (
        f"segment shape {seg.shape} -- expected (th, tw, count) = "
        f"({lv['th']}, {lv['tw']}, {lv['count']}). A blind np.squeeze drops the "
        "length-1 row axis and the width axis takes its place.")


def test_normalise_segment_shapes():
    """Unit-level: the decode-shape normaliser, across the layouts tifffile
    actually produces (verified empirically against every fixture in this
    suite plus the real 20-band export)."""
    contig = {"planar": 1}
    separate = {"planar": 2}

    # planar=1: (depth, h, w, samples) -> (h, w, samples)
    got = TiledRasterReader._normalise_segment(np.zeros((1, 1, 2350, 20)), contig)
    assert got.shape == (1, 2350, 20), got.shape
    got = TiledRasterReader._normalise_segment(np.zeros((1, 16, 16, 200)), contig)
    assert got.shape == (16, 16, 200), got.shape
    # planar=1, single sample: the sample axis is kept (read_window indexes it)
    got = TiledRasterReader._normalise_segment(np.zeros((1, 64, 64, 1)), contig)
    assert got.shape == (64, 64, 1), got.shape

    # planar=2: one plane per segment -> (h, w)
    got = TiledRasterReader._normalise_segment(np.zeros((1, 16, 16, 1)), separate)
    assert got.shape == (16, 16), got.shape
    got = TiledRasterReader._normalise_segment(np.zeros((1, 1, 2350, 1)), separate)
    assert got.shape == (1, 2350), got.shape


# ---------------------------------------------------------------------------
# End to end: windowed reads must equal a full tifffile read
# ---------------------------------------------------------------------------
def test_row_per_strip_windows_match_full_read(tmp_path):
    """THE reported failure, end to end. Every one of these windows raised
    'could not broadcast input array from shape (1,S) into shape (1,w)'
    before the fix."""
    fp = tmp_path / "rowstrip_e2e.tif"
    arr = _write_row_per_strip(fp, h=9, w=23, bands=5)

    clear_reader_cache()
    reader = open_reader(fp, probe(fp))
    assert reader is not None

    cases = [
        (0, 0, 23, 1, None),        # a single row -- the degenerate case
        (0, 0, 5, 3, [0, 1, 2]),
        (7, 2, 20, 8, [0, 4]),
        (0, 0, 23, 9, None),        # whole frame
        (20, 6, 23, 9, [4]),        # bottom-right corner
    ]
    for x0, y0, x1, y1, bands in cases:
        got = reader.read_window(x0, y0, x1, y1, bands=bands)
        idx = list(range(arr.shape[2])) if bands is None else bands
        ref = arr[y0:y1, x0:x1][:, :, idx]
        assert got.shape == ref.shape, f"{(x0,y0,x1,y1,bands)}: {got.shape} != {ref.shape}"
        assert np.array_equal(got, ref), f"{(x0,y0,x1,y1,bands)}: values differ"


def test_single_band_row_per_strip_round_trips(tmp_path):
    fp = tmp_path / "rowstrip_1band.tif"
    arr = _write_single_band_row_per_strip(fp)

    clear_reader_cache()
    reader = open_reader(fp, probe(fp))
    assert reader is not None

    got = reader.read_window(0, 0, arr.shape[1], arr.shape[0])
    assert got.shape == (arr.shape[0], arr.shape[1], 1), got.shape
    assert np.array_equal(got[:, :, 0], arr)


def test_row_per_strip_sample_bands_does_not_fall_back(tmp_path):
    """`LazyChannels.sample_bands` is what process_polygon calls for per-band
    scene statistics. Its failure is only WARNED about and then silently
    replaced by full-resolution reads, so a regression here shows up to the
    user as unexplained slowness rather than an error."""
    from ..raster_reader import LazyChannels

    fp = tmp_path / "rowstrip_sample.tif"
    arr = _write_row_per_strip(fp, h=40, w=64, bands=6)

    clear_reader_cache()
    reader = open_reader(fp, probe(fp))
    chans = LazyChannels(reader, level=0)

    sample = np.asarray(
        chans.sample_bands(list(range(len(chans))),
                           max_bytes=8 * 1024 * 1024,
                           decode_budget=16 * 1024 * 1024),
        dtype=np.float32)

    assert sample.ndim == 3 and sample.shape[2] == arr.shape[2], (
        f"sample_bands returned {sample.shape} for a {arr.shape} raster")
    assert sample.size > 0


def test_declared_nodata_survives_a_windowed_read(tmp_path):
    """The fill row must come back as the sentinel, in the right place --
    proof the row axis was not transposed with the width axis."""
    fp = tmp_path / "rowstrip_nodata.tif"
    arr = _write_row_per_strip(fp, h=9, w=23, bands=5)
    assert np.all(arr[0] == -9999.0), "fixture lost its fill row"

    clear_reader_cache()
    profile = probe(fp)
    reader = open_reader(fp, profile)

    got = reader.read_window(0, 0, 23, 9)
    assert np.all(got[0] == -9999.0), "fill row is not row 0 after the read"
    assert not np.any(got[1:] == -9999.0), "fill leaked into other rows"
