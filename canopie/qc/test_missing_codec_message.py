"""
A TIFF whose compression codec is missing must be reported as that, not as RAM.

The Windows exe bundled none of imagecodecs' codec modules, so the windowed
preview of an LZW-compressed GEE export failed with "could not import name
'lzw_decode' from 'imagecodecs'". The loader then fell back to a full read,
whose size guard said "probe failed ... refusing full in-memory load to avoid
exhausting RAM. Convert to a tiled/COG TIFF" -- for a file that was already
tiled and read fine from source.
"""
import numpy as np
import pytest
import tifffile


def test_missing_codec_is_named_instead_of_ram(synthetic_project, tmp_path, monkeypatch):
    from .. import project_tab, raster_reader

    fp = str(tmp_path / "gee_stack.tif")
    tifffile.imwrite(fp, np.random.rand(64, 64, 8).astype(np.float32),
                     tile=(16, 16), compression="lzw", photometric="minisblack",
                     planarconfig="contig")

    # Make this small stack count as "oversized", as the 2.6 GB export was.
    monkeypatch.setattr(project_tab, "_PREVIEW_LOAD_THRESHOLD_BYTES", 1000)

    def _no_lzw(*a, **k):
        raise ImportError("could not import name 'lzw_decode' from 'imagecodecs'")
    monkeypatch.setattr(raster_reader, "read_preview", _no_lzw)

    with pytest.raises(RuntimeError) as exc:
        synthetic_project._imagedata_or_fallback(fp)
    msg = str(exc.value)
    assert "lzw_decode" in msg and "imagecodecs" in msg
    assert "exhausting RAM" not in msg


def test_ram_guard_unchanged_without_a_codec_error(synthetic_project, tmp_path, monkeypatch):
    from .. import project_tab, raster_reader

    fp = str(tmp_path / "stack.tif")
    tifffile.imwrite(fp, np.random.rand(64, 64, 8).astype(np.float32),
                     tile=(16, 16), photometric="minisblack", planarconfig="contig")
    monkeypatch.setattr(project_tab, "_PREVIEW_LOAD_THRESHOLD_BYTES", 1000)

    def _broken(*a, **k):
        raise ValueError("some other preview failure")
    monkeypatch.setattr(raster_reader, "read_preview", _broken)

    with pytest.raises(MemoryError):
        synthetic_project._imagedata_or_fallback(fp)
