"""
QC tests for shapefile export (shapefile_io).

Shapefiles leave CanoPie and get opened in QGIS/ArcGIS, so a malformed
geometry, a truncated DBF field name or a wrong pixel->geo transform is a
silent data-integrity problem for downstream analysis rather than a visible
crash. These tests read the written bytes back rather than trusting the
writer's return value.
"""
import os
import struct

import numpy as np
import pytest

from ..shapefile_io import (
    write_shapefile, write_dbf, _dbf_safe_fieldname,
    pixel_to_geo, geo_to_pixel, wkt_for_epsg,
)

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.io]

SHAPE_TYPE_POLYGON = 5


def _square(x, y, size=10):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]


def _feature(points, **props):
    """The real feature schema: 'geometry' (ring points) + 'properties' (attrs).
    NOTE: write_shapefile's output_path EXCLUDES the extension -- it appends
    .shp/.shx/.dbf/.prj itself."""
    return {"geometry": points, "properties": dict(props)}


def _read_shp_header(path):
    with open(path, "rb") as f:
        head = f.read(100)
    file_code = struct.unpack(">i", head[0:4])[0]
    version = struct.unpack("<i", head[28:32])[0]
    shape_type = struct.unpack("<i", head[32:36])[0]
    bbox = struct.unpack("<4d", head[36:68])
    return file_code, version, shape_type, bbox


def test_writes_the_three_mandatory_sidecar_files(tmp_path):
    """A shapefile is only valid as a .shp + .shx + .dbf set -- writing the
    geometry alone produces something most GIS tools refuse to open."""
    stem = tmp_path / "shapes"
    write_shapefile([_feature(_square(0, 0), name="a")], str(stem),
                    crs_wkt=wkt_for_epsg(4326), shape_type=SHAPE_TYPE_POLYGON)

    for ext in (".shp", ".shx", ".dbf"):
        p = tmp_path / ("shapes" + ext)
        assert p.exists(), f"missing mandatory sidecar {ext}"
        assert p.stat().st_size > 0, f"{ext} was written but is empty"


def test_shp_header_is_well_formed(tmp_path):
    """File code 9994 and version 1000 are fixed by the ESRI spec; a wrong
    value here is what makes a reader reject the file outright."""
    stem = tmp_path / "hdr"
    write_shapefile([_feature(_square(2, 3))], str(stem),
                    shape_type=SHAPE_TYPE_POLYGON)

    file_code, version, shape_type, bbox = _read_shp_header(str(stem) + ".shp")
    assert file_code == 9994, f"bad .shp file code {file_code} (expected 9994)"
    assert version == 1000, f"bad .shp version {version} (expected 1000)"
    assert shape_type == SHAPE_TYPE_POLYGON, f"header shape type {shape_type}"

    xmin, ymin, xmax, ymax = bbox
    assert xmin <= 2 and ymin <= 3, f"bbox min {(xmin, ymin)} excludes the geometry"
    assert xmax >= 12 and ymax >= 13, f"bbox max {(xmax, ymax)} excludes the geometry"


def test_prj_written_when_crs_supplied(tmp_path):
    stem = tmp_path / "withcrs"
    write_shapefile([_feature(_square(0, 0))], str(stem),
                    crs_wkt=wkt_for_epsg(4326), shape_type=SHAPE_TYPE_POLYGON)
    prj = tmp_path / "withcrs.prj"
    assert prj.exists(), ".prj not written despite a CRS being supplied"
    # pyproj (when installed) emits WKT2, whose keyword is GEOGCRS rather than
    # WKT1's GEOGCS -- accept either instead of pinning one library's output.
    txt = prj.read_text(encoding="utf-8", errors="replace").upper()
    assert "GEOGCRS" in txt or "GEOGCS" in txt, f"unexpected .prj content: {txt[:80]!r}"


def test_multiple_features_all_recorded(tmp_path):
    """Record count lives in the DBF header; under-counting silently drops
    polygons from the export."""
    stem = tmp_path / "many"
    features = [_feature(_square(i * 20, 0), idx=i) for i in range(5)]
    write_shapefile(features, str(stem), shape_type=SHAPE_TYPE_POLYGON)

    with open(str(tmp_path / "many.dbf"), "rb") as f:
        header = f.read(32)
    n_records = struct.unpack("<I", header[4:8])[0]
    assert n_records == 5, f"DBF reports {n_records} records, expected 5"


def test_dbf_field_names_are_sanitized_and_unique():
    """DBF caps field names at 10 chars. Long attribute names must be
    truncated AND de-duplicated, or two different columns collapse into one
    and data is silently lost."""
    used = set()
    a = _dbf_safe_fieldname("a_very_long_attribute_name", used)
    used.add(a.upper())
    b = _dbf_safe_fieldname("a_very_long_attribute_name_2", used)

    assert len(a) <= 10, f"field name {a!r} exceeds the 10-char DBF limit"
    assert len(b) <= 10, f"field name {b!r} exceeds the 10-char DBF limit"
    assert a.upper() != b.upper(), (
        f"two long names collapsed to the same field ({a!r} / {b!r}) -- "
        "one column's data would overwrite the other")


def test_dbf_roundtrip_preserves_attribute_values(tmp_path):
    """Attribute values must survive to disk -- this is the table half of the
    export that downstream analysis actually joins on."""
    dbf = tmp_path / "attrs.dbf"
    features = [
        {"properties": {"name": "alpha", "value": 12.5}},
        {"properties": {"name": "beta", "value": -3.25}},
    ]
    write_dbf(str(dbf), features)

    raw = dbf.read_bytes()
    text = raw.decode("latin-1")
    assert "alpha" in text and "beta" in text, "attribute strings missing from the DBF"
    assert "12.5" in text and "-3.25" in text, "numeric attributes missing from the DBF"


def test_pixel_to_geo_and_back_is_identity():
    """The two transforms are used on opposite ends of import/export; if they
    are not exact inverses, coordinates drift every round trip."""
    # (originX, pixelWidth, rotX, originY, rotY, pixelHeight) GDAL-style
    transform = (500000.0, 0.5, 0.0, 4000000.0, 0.0, -0.5)
    pixels = [(0, 0), (10, 20), (123.5, 77.25)]

    geo = pixel_to_geo(pixels, transform)
    back = geo_to_pixel(geo, transform)

    for (px, py), (bx, by) in zip(pixels, back):
        assert abs(px - bx) < 1e-6, f"x drifted {px} -> {bx}"
        assert abs(py - by) < 1e-6, f"y drifted {py} -> {by}"


def test_pixel_to_geo_applies_the_transform_correctly():
    """Spot-check against hand-computed values, so an inverted sign or a
    swapped axis can't hide behind a self-consistent round trip."""
    transform = (100.0, 2.0, 0.0, 900.0, 0.0, -2.0)
    (gx, gy), = pixel_to_geo([(3, 4)], transform)
    assert abs(gx - (100.0 + 3 * 2.0)) < 1e-9, f"got x={gx}"
    assert abs(gy - (900.0 - 4 * 2.0)) < 1e-9, f"got y={gy} (north-up should decrease)"


def test_empty_feature_list_does_not_produce_a_corrupt_file(tmp_path):
    """Exporting with nothing selected must not emit a half-written file that
    a GIS will choke on."""
    stem = tmp_path / "empty"
    try:
        write_shapefile([], str(stem), shape_type=SHAPE_TYPE_POLYGON)
    except Exception:
        return  # refusing outright is an acceptable outcome
    shp = tmp_path / "empty.shp"
    if shp.exists():
        file_code, version, _st, _bbox = _read_shp_header(str(shp))
        assert file_code == 9994 and version == 1000, (
            "an empty export produced a structurally invalid .shp")


def test_wkt_for_known_epsg_is_returned():
    wkt = wkt_for_epsg(4326)
    assert wkt, "no WKT returned for EPSG:4326"
    up = wkt.upper()
    assert "GEOGCRS" in up or "GEOGCS" in up, f"unexpected WKT for EPSG:4326: {wkt[:80]!r}"
