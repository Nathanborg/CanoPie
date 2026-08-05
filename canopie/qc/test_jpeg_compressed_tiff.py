"""
QC regression tests for JPEG-COMPRESSED TIFFs (compression tag 6/7/34892/33007).

THE BUG THIS PINS (found by a user running a real GDAL COG, NOT by this
suite's original fixtures):

    imagecodecs.Jpeg8Error: Quantization table 0x00 was not defined

A JPEG-compressed TIFF stores its quantization/Huffman tables ONCE in the
TIFF `JPEGTables` tag rather than repeating them in every tile. Each tile
therefore holds only entropy-coded scan data and genuinely cannot be decoded
on its own. TiffPage.decode() accepts `jpegtables`/`jpegheader` for exactly
this; raster_reader's tile decoder never passed them, so EVERY read of a
JPEG-compressed COG raised the error above -- viewer, Inspect, CSV export,
ML export alike.

WHY THE ORIGINAL FIXTURES MISSED IT: this suite had a standalone `.jpg`
fixture, but JPEG-as-a-file and JPEG-as-a-TIFF-codec are unrelated code
paths. Worse, tifffile's own writer stores JPEG tables INLINE per tile and
leaves the JPEGTables tag unset, so a JPEG TIFF written by tifffile decodes
fine either way -- a synthetic fixture literally cannot reproduce this
without hand-built bytes. These tests therefore assert the MECHANISM (that
the shared tables are resolved and forwarded), which is checkable
synthetically, and additionally run end-to-end against a real GDAL COG when
one is present on this machine.
"""
import os

import numpy as np
import pytest
import tifffile

from ..raster_reader import _JPEG_COMPRESSIONS, TiledRasterReader, probe, open_reader, clear_reader_cache

# A real GDAL-written JPEG COG, if this machine has one. Not required: the
# mechanism tests below stand on their own.
REAL_JPEG_COG = r"C:\Users\natha\Downloads\aligned_COG\aligned_COG\P0194882_aligned.cog.tif"
_has_real_cog = os.path.exists(REAL_JPEG_COG)


@pytest.fixture()
def jpeg_tiff(tmp_path):
    """A tiled, JPEG-compressed TIFF. Note tifffile inlines the tables here
    (page.jpegtables is None), so this file decodes with or without the fix --
    it guards against the fix BREAKING the inline case, which is the other
    half of getting this right."""
    path = tmp_path / "jpeg_tiles.tif"
    rng = np.random.default_rng(7)
    arr = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(str(path), arr, photometric="rgb", planarconfig="contig",
                     tile=(64, 64), compression="jpeg")
    return str(path), arr


def test_jpeg_compression_ids_cover_the_real_world_tag():
    """GDAL writes compression=7 for JPEG COGs. If that ever drops out of the
    set, the tables stop being forwarded and the original crash returns."""
    assert 7 in _JPEG_COMPRESSIONS, "compression 7 (JPEG) must be recognized"
    assert 6 in _JPEG_COMPRESSIONS, "compression 6 (old-style JPEG) must be recognized"


def test_decode_kwargs_populated_for_jpeg_pages(jpeg_tiff):
    """THE MECHANISM: for a JPEG-compressed page the per-level decode kwargs
    must carry jpegtables/jpegheader through to TiffPage.decode(). Their VALUE
    may legitimately be None (tifffile inlines tables), but the keys must be
    present -- that is what makes a shared-table file work."""
    path, _ = jpeg_tiff
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        assert int(page.compression) in _JPEG_COMPRESSIONS, "fixture is not JPEG-compressed"
        geom = TiledRasterReader._level_geometry(page)

    kwargs = geom.get("decode_kwargs")
    assert kwargs is not None, "level geometry carries no decode_kwargs at all"
    assert "jpegtables" in kwargs, (
        "jpegtables not forwarded -- a COG with shared tables will fail with "
        "'Quantization table 0x00 was not defined'")
    assert "jpegheader" in kwargs, "jpegheader not forwarded"


def test_decode_kwargs_empty_for_non_jpeg_pages(tmp_path):
    """Non-JPEG codecs must NOT be handed JPEG-only kwargs; deflate/raw pages
    would reject them."""
    path = tmp_path / "deflate.tif"
    arr = (np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
    tifffile.imwrite(str(path), arr, photometric="rgb", planarconfig="contig",
                     tile=(32, 32), compression="deflate")
    with tifffile.TiffFile(str(path)) as tf:
        geom = TiledRasterReader._level_geometry(tf.pages[0])
    assert not geom.get("decode_kwargs"), (
        f"non-JPEG page got JPEG decode kwargs: {geom.get('decode_kwargs')}")


def test_jpeg_tiff_decodes_and_is_close_to_source(jpeg_tiff):
    """End-to-end read of a JPEG-compressed TIFF through the real reader.
    JPEG is lossy, so this asserts closeness rather than equality -- the point
    is that it decodes at all and returns plausible pixels."""
    path, arr = jpeg_tiff
    clear_reader_cache()
    profile = probe(path)
    assert profile is not None
    reader = open_reader(path, profile)

    window = np.asarray(reader.read_window(0, 0, 64, 64))
    assert window.shape == (64, 64, 3), f"unexpected window shape {window.shape}"
    assert window.dtype == np.uint8

    ref = arr[0:64, 0:64, :].astype(np.float32)
    got = window.astype(np.float32)
    # Random noise + chroma subsampling is the worst case for JPEG; a loose
    # bound still proves we decoded real image data rather than garbage.
    assert abs(float(got.mean()) - float(ref.mean())) < 40.0, (
        f"decoded mean {got.mean():.1f} implausibly far from source {ref.mean():.1f}")


def test_jpeg_tiff_matches_tifffile_reference(jpeg_tiff):
    """Our windowed read must agree EXACTLY with tifffile's own full-page
    decode of the same region -- same decoder, same tables, so any difference
    is our tiling/assembly being wrong."""
    path, _ = jpeg_tiff
    clear_reader_cache()
    reader = open_reader(path, probe(path))
    window = np.asarray(reader.read_window(32, 16, 96, 80))

    with tifffile.TiffFile(path) as tf:
        full = tf.pages[0].asarray()
    ref = full[16:80, 32:96, :]

    assert np.array_equal(window, ref), (
        "windowed JPEG read does not match tifffile's reference decode")


@pytest.mark.skipif(not _has_real_cog,
                    reason="no real GDAL JPEG COG on this machine")
def test_real_gdal_jpeg_cog_decodes():
    """THE ACTUAL REPORTED FAILURE, against the actual file.

    This COG has compression=7 with a populated 142-byte JPEGTables tag --
    the shared-table layout tifffile's writer cannot produce, and the exact
    configuration that raised 'Quantization table 0x00 was not defined' for
    every read before the fix."""
    clear_reader_cache()
    profile = probe(REAL_JPEG_COG)
    assert profile is not None

    with tifffile.TiffFile(REAL_JPEG_COG) as tf:
        page = tf.pages[0]
        assert int(page.compression) in _JPEG_COMPRESSIONS
        assert page.jpegtables is not None, (
            "this file no longer uses shared JPEGTables -- it can no longer "
            "reproduce the original bug, so pick a different reference COG")

    reader = open_reader(REAL_JPEG_COG, profile)
    window = np.asarray(reader.read_window(1000, 1000, 1064, 1064))

    assert window.shape == (64, 64, 3)
    assert window.dtype == np.uint8
    assert int(window.max()) > 0, "decoded an all-zero window -- decode silently failed"
