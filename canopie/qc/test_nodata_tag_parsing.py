"""
QC regression tests for GDAL_NODATA tag parsing (raster_reader._parse_nodata).

THE BUG THIS PINS
------------------
``_parse_nodata`` used to do::

    raw = str(tag.value).strip()
    return float(raw)

which is correct only when ``tifffile`` hands back the tag as a clean
``str``. On files where the GDAL_NODATA (TIFF tag 42113) ASCII tag was
written -- or later re-tagged -- with a mismatched TIFF field type, tifffile
instead returns:

  * ``bytes``, often null-terminated (e.g. ``b'-9999\\x00'``) when the tag
    was typed BYTE instead of ASCII, or
  * a ``tuple`` of character codes (e.g. ``(45, 57, 57, 57, 57, 0)``) when
    it was typed SHORT/LONG instead of ASCII.

``str(b'-9999\\x00')`` is the Python *repr* of the bytes object --
``"b'-9999\\x00'"`` -- and ``str((45, 57, ...))`` is the repr of the tuple.
Neither parses as a float, so ``float(raw)`` raised ``ValueError``, which the
bare ``except Exception:`` in ``_parse_nodata`` silently swallowed, returning
``None``. That disabled NoData masking end-to-end (viewer, stats, export) for
every image carrying one of these encodings -- observed on real project
imagery under ``C:\\New Folder201``.

WHY SYNTHETIC FIXTURES CAN REPRODUCE THIS (unlike the JPEG-COG bug in
test_jpeg_compressed_tiff.py): the mistyped-tag encoding is just a TIFF field
type byte, which ``tifffile.imwrite(..., extratags=[(42113, 'B', ...)])``
controls directly -- no external GDAL write path is needed to build a file
that reproduces the exact ``tag.value`` shapes seen in the field. Verified
empirically against tifffile 2025.8.28 before writing these assertions (see
task notes); if a future tifffile version changes how it exposes a
mistyped-type tag, these tests will fail loudly rather than silently agree
with a regression.
"""
import os

import numpy as np
import pytest
import tifffile

from ..raster_reader import _parse_nodata, probe

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
# raster_reader.py is one of the four "underpins everything" modules, so any
# change here must also be run with `-m "extraction or contract"`.
pytestmark = [pytest.mark.io, pytest.mark.extraction]

_TAG_ID = 42113  # GDAL_NODATA

# A real project file that hit this bug in the field, if this machine has it
# mounted (Google Drive virtual FS -- not always available). Not required:
# the synthetic tests below stand on their own, exactly like
# test_jpeg_compressed_tiff.py's REAL_JPEG_COG pattern.
REAL_NODATA_FILE = r"G:\Meu Drive\GEE_Exports\pca_scores_Panama_climate.tif"
_has_real_file = os.path.exists(REAL_NODATA_FILE)


def _write_tif_with_nodata_tag(path, tag_type, count, value):
    """Write a tiny TIFF whose GDAL_NODATA tag uses a specific TIFF field
    type -- this is what actually determines whether tifffile hands
    ``_parse_nodata`` a str, bytes, or tuple."""
    arr = np.zeros((8, 8), dtype=np.float32)
    tifffile.imwrite(str(path), arr, extratags=[(_TAG_ID, tag_type, count, value, False)])


def _nodata_via_page(path):
    with tifffile.TiffFile(str(path)) as tf:
        return _parse_nodata(tf.pages[0])


# ---------------------------------------------------------------------------
# Unit tests against _parse_nodata directly, one per tag encoding.
# ---------------------------------------------------------------------------

def test_ascii_tag_clean_string_still_works(tmp_path):
    """Regression guard: the common case (tifffile decodes ASCII to str)
    must keep working after the fix."""
    p = tmp_path / "ascii.tif"
    _write_tif_with_nodata_tag(p, "s", 0, "-9999\x00")
    assert _nodata_via_page(p) == -9999.0


def test_byte_tag_with_null_terminator(tmp_path):
    """THE reported failure mode: GDAL_NODATA mistyped as BYTE, null-terminated."""
    p = tmp_path / "byte_null.tif"
    raw = b"-9999\x00"
    _write_tif_with_nodata_tag(p, "B", len(raw), raw)
    assert _nodata_via_page(p) == -9999.0


def test_byte_tag_without_null_terminator(tmp_path):
    p = tmp_path / "byte_no_null.tif"
    raw = b"-9999"
    _write_tif_with_nodata_tag(p, "B", len(raw), raw)
    assert _nodata_via_page(p) == -9999.0


def test_short_tag_tuple_of_char_codes(tmp_path):
    """GDAL_NODATA mistyped as SHORT array -- tifffile hands back a tuple of
    ints (character codes), not bytes or str."""
    p = tmp_path / "short_tuple.tif"
    codes = tuple(ord(c) for c in "-9999\x00")
    _write_tif_with_nodata_tag(p, "H", len(codes), codes)
    assert _nodata_via_page(p) == -9999.0


def test_decimal_value_survives_every_encoding(tmp_path):
    for tag_type, count, value in [
        ("s", 0, "-9999.5\x00"),
        ("B", 7, b"-9999.5"),
        ("H", 8, tuple(ord(c) for c in "-9999.5\x00")),
    ]:
        p = tmp_path / f"decimal_{tag_type}.tif"
        _write_tif_with_nodata_tag(p, tag_type, count, value)
        assert _nodata_via_page(p) == pytest.approx(-9999.5), (
            f"tag_type={tag_type!r} lost precision or failed to parse")


@pytest.mark.parametrize("tag_type,count,value", [
    ("s", 0, "nan\x00"),
    ("B", 4, b"nan\x00"),
    ("H", 4, tuple(ord(c) for c in "nan\x00")),
])
def test_nan_literal_across_encodings(tmp_path, tag_type, count, value):
    p = tmp_path / f"nan_{tag_type}.tif"
    _write_tif_with_nodata_tag(p, tag_type, count, value)
    result = _nodata_via_page(p)
    assert result is not None and np.isnan(result)


def test_missing_tag_returns_none(tmp_path):
    p = tmp_path / "no_tag.tif"
    tifffile.imwrite(str(p), np.zeros((8, 8), dtype=np.float32))
    assert _nodata_via_page(p) is None


def test_garbage_value_returns_none_not_raises(tmp_path):
    """A tag that decodes to non-numeric text must fail SAFE (None), not
    propagate an exception into probe()."""
    p = tmp_path / "garbage.tif"
    _write_tif_with_nodata_tag(p, "s", 0, "not-a-number\x00")
    assert _nodata_via_page(p) is None


def test_numeric_tag_id_fallback_when_name_lookup_misses():
    """Some tifffile tag-map implementations may not expose the ASCII name
    'GDAL_NODATA'; the numeric TIFF tag ID (42113) must still resolve it.

    Exercised against a minimal fake page/tags object rather than a real
    TiffFile, since the installed tifffile version *does* register the name
    (see module docstring) and so cannot be coaxed into omitting it -- this
    directly tests the fallback branch of _parse_nodata itself.
    """
    class _FakeTag:
        value = "-9999"

    class _FakeTags(dict):
        def get(self, key, default=None):
            # Only resolvable by the numeric TIFF tag ID, never by name --
            # simulates the lookup miss the fallback exists for.
            if key == _TAG_ID:
                return _FakeTag()
            return default

    class _FakePage:
        tags = _FakeTags()

    assert _parse_nodata(_FakePage()) == -9999.0


def test_no_tag_at_all_via_fallback_returns_none():
    class _FakeTags(dict):
        def get(self, key, default=None):
            return default

    class _FakePage:
        tags = _FakeTags()

    assert _parse_nodata(_FakePage()) is None


# ---------------------------------------------------------------------------
# Integration: the fix must flow through probe(), the public entry point
# every other subsystem (viewer, stats, export) actually calls.
# ---------------------------------------------------------------------------

def test_probe_recovers_nodata_from_mistyped_byte_tag(tmp_path):
    p = tmp_path / "profile_byte.tif"
    _write_tif_with_nodata_tag(p, "B", 6, b"-9999\x00")
    profile = probe(p)
    assert profile is not None
    assert profile.nodata == -9999.0


def test_probe_recovers_nodata_from_mistyped_short_tag(tmp_path):
    p = tmp_path / "profile_short.tif"
    codes = tuple(ord(c) for c in "-9999\x00")
    _write_tif_with_nodata_tag(p, "H", len(codes), codes)
    profile = probe(p)
    assert profile is not None
    assert profile.nodata == -9999.0


@pytest.mark.skipif(not _has_real_file,
                    reason="Google Drive mount with the real project file is not available on this machine")
def test_real_project_file_nodata_is_recovered():
    """THE ACTUAL REPORTED FAILURE, against the real file referenced by
    C:\\New Folder201\\project.json (rgb_folder_path)."""
    profile = probe(REAL_NODATA_FILE)
    assert profile is not None
    assert profile.nodata is not None, (
        "GDAL_NODATA failed to parse on the real reference file -- "
        "the exact regression this test suite exists to catch")
    assert profile.nodata == pytest.approx(-9999.0)
