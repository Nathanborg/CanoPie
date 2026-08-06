"""
QC regression tests for coordinate-reference-system handling in shapefile export.

THE BUG THIS PINS (reported as: "when I load in QGIS the shapefile is in a very
different coordinate system"):

`json_polygons_to_features` returned a SINGLE `crs_wkt` -- the first one it
happened to encounter -- and every caller wrote every feature into one .shp
with that one .prj. But an ESRI shapefile is single-CRS by definition, and
CanoPie derives coordinates three different ways:

    GeoTIFF tags        -> the raster's own CRS, usually UTM METRES
    EXIF GPS estimate   -> WGS84 DEGREES  (estimate_transform_from_exif)
    no georeferencing   -> RAW PIXEL indices, passed straight through

So a project mixing UTM zones put every feature outside the first zone hundreds
of kilometres away; an un-georeferenced raster contributed vertices like
(10, 10) into a file labelled UTM, landing them near the projection origin (for
UTM north, in the ocean off West Africa); and an EXIF-estimated file mixed
degrees into metres. All three look identical to the user.

Features now carry their own `crs_wkt`, and `write_feature_collection` splits
into one file per (CRS, geometry type) -- the same reason Point and Polygon
were already split.
"""
import os

import numpy as np
import pytest
import tifffile

from ..shapefile_io import (crs_label, group_features_by_crs,
                            json_polygons_to_features, write_feature_collection)


def _write_geotiff(path, epsg, origin_x, origin_y, scale=1.0, w=64, h=64):
    """Minimal but real GeoTIFF: ModelTiepoint + ModelPixelScale + a
    GeoKeyDirectory declaring ProjectedCSTypeGeoKey (3072)."""
    gk = [1, 1, 0, 1, 3072, 0, 1, int(epsg)]
    extratags = [
        (33922, 'd', 6, (0.0, 0.0, 0.0, float(origin_x), float(origin_y), 0.0), True),
        (33550, 'd', 3, (float(scale), float(scale), 0.0), True),
        (34735, 'H', len(gk), tuple(gk), True),
    ]
    tifffile.imwrite(path, np.zeros((h, w), dtype=np.uint8), extratags=extratags)
    return path


def _write_plain_tiff(path, w=64, h=64):
    tifffile.imwrite(path, np.zeros((h, w), dtype=np.uint8))
    return path


def _shape(fp, kind="polygon"):
    pts = ([(10, 10), (20, 10), (20, 20), (10, 20)] if kind == "polygon"
           else [(10, 10), (20, 20)])
    return {"points": pts, "name": os.path.basename(fp), "root": "r",
            "type": kind, "coord_space": "image",
            "image_ref_size": {"w": 64, "h": 64}}


@pytest.fixture
def mixed_crs_project(tmp_path):
    """Three rasters: UTM 20N, UTM 21N, and one with no georeferencing."""
    a = _write_geotiff(str(tmp_path / "zone20.tif"), 32620, 600000.0, 1000000.0)
    b = _write_geotiff(str(tmp_path / "zone21.tif"), 32621, 300000.0, 9000000.0)
    c = _write_plain_tiff(str(tmp_path / "plain.tif"))
    all_polygons = {"g20": {a: _shape(a)},
                    "g21": {b: _shape(b)},
                    "gnone": {c: _shape(c)}}
    return all_polygons, str(tmp_path), (a, b, c)


# ---------------------------------------------------------------------------
# Per-feature CRS
# ---------------------------------------------------------------------------
def test_each_feature_carries_its_own_crs(mixed_crs_project):
    """THE root fix: a feature has to know which CRS its coordinates are in.
    Only one CRS was tracked for the whole collection before."""
    all_polygons, folder, (a, b, c) = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)

    by_file = {f['properties']['filename']: f for f in features}
    assert "crs_wkt" in by_file["zone20.tif"], "features do not carry crs_wkt"
    assert "20N" in (by_file["zone20.tif"]["crs_wkt"] or "")
    assert "21N" in (by_file["zone21.tif"]["crs_wkt"] or "")
    assert by_file["plain.tif"]["crs_wkt"] is None, (
        "an un-georeferenced feature must not claim a CRS -- its coordinates "
        "are raw pixel indices")


def test_ungeoreferenced_features_really_are_pixel_coordinates(mixed_crs_project):
    """Establishes WHY mixing is fatal: these vertices are (10, 10), which in a
    UTM file would sit at the projection origin, ~5000 km from the data."""
    all_polygons, folder, (a, b, c) = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)
    plain = next(f for f in features if f['properties']['filename'] == "plain.tif")

    x, y = plain['geometry'][0]
    assert abs(x) < 100 and abs(y) < 100, f"expected pixel coords, got ({x}, {y})"

    utm = next(f for f in features if f['properties']['filename'] == "zone20.tif")
    ux, _uy = utm['geometry'][0]
    assert ux > 100000, "UTM feature should be in metres -- fixture is wrong"


# ---------------------------------------------------------------------------
# Grouping / splitting
# ---------------------------------------------------------------------------
def test_mixed_crs_groups_separately(mixed_crs_project):
    all_polygons, folder, _ = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)
    groups = group_features_by_crs(features)

    assert len(groups) == 3, f"expected 3 CRS groups, got {list(groups)}"
    assert "pixel" in groups, "un-georeferenced features need their own group"


def test_mixed_crs_is_reported_as_a_warning(mixed_crs_project):
    """Silent mislocation is the whole problem -- the user must be told."""
    all_polygons, folder, _ = mixed_crs_project
    _features, _crs, warns = json_polygons_to_features(all_polygons, folder)

    joined = " ".join(warns).lower()
    assert "more than one coordinate reference system" in joined, (
        f"no mixed-CRS warning was emitted; warnings were: {warns}")


def test_mixed_crs_writes_one_file_per_crs(mixed_crs_project, tmp_path):
    """THE regression: three CRSs must not share one .shp."""
    all_polygons, folder, _ = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)

    stem = str(tmp_path / "out" / "export")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    written, _notes = write_feature_collection(features, stem)

    assert len(written) == 3, f"expected 3 shapefiles, got {written}"
    for p in written:
        assert os.path.exists(p), f"{p} was not written"


def test_each_prj_matches_the_features_in_its_own_file(mixed_crs_project, tmp_path):
    """The .prj must describe the coordinates actually in that .shp -- writing
    zone 20N's .prj beside zone 21N's metres is exactly the reported bug."""
    all_polygons, folder, _ = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)

    stem = str(tmp_path / "export")
    written, _notes = write_feature_collection(features, stem)

    zone21 = [p for p in written if "32621" in p]
    assert zone21, f"zone 21 got no dedicated file: {written}"
    prj = os.path.splitext(zone21[0])[0] + ".prj"
    assert os.path.exists(prj)
    assert "21N" in open(prj, encoding="utf-8").read()


def test_pixel_coordinate_file_gets_no_prj(mixed_crs_project, tmp_path):
    """Without a .prj a GIS ASKS instead of silently placing pixel indices on a
    projected map -- strictly better than a confident wrong answer."""
    all_polygons, folder, _ = mixed_crs_project
    features, _crs, _warns = json_polygons_to_features(all_polygons, folder)

    stem = str(tmp_path / "export")
    written, notes = write_feature_collection(features, stem)

    pixel = [p for p in written if "pixel" in os.path.basename(p)]
    assert pixel, f"pixel-space features got no dedicated file: {written}"
    assert not os.path.exists(os.path.splitext(pixel[0])[0] + ".prj"), (
        "a pixel-coordinate shapefile must not claim a projection")
    assert any("RAW PIXEL" in n for n in notes), (
        f"the pixel-coordinate caveat was not surfaced: {notes}")


# ---------------------------------------------------------------------------
# The common single-CRS case must be unchanged
# ---------------------------------------------------------------------------
def test_single_crs_keeps_the_plain_filename(tmp_path):
    """Splitting must not rename the output for everyone who was fine before."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    b = _write_geotiff(str(tmp_path / "b.tif"), 32620, 601000.0, 1000000.0)
    all_polygons = {"ga": {a: _shape(a)}, "gb": {b: _shape(b)}}

    features, _crs, warns = json_polygons_to_features(all_polygons, str(tmp_path))
    stem = str(tmp_path / "export")
    written, _notes = write_feature_collection(features, stem)

    assert written == [stem + ".shp"], f"unexpected outputs: {written}"
    assert not any("more than one coordinate" in w.lower() for w in warns)


def test_point_and_polygon_still_split_within_one_crs(tmp_path):
    """The pre-existing geometry split must survive the CRS split."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32620, 600000.0, 1000000.0)
    all_polygons = {"poly": {a: _shape(a, "polygon")},
                    "pts": {a: _shape(a, "point")}}

    features, _crs, _warns = json_polygons_to_features(all_polygons, str(tmp_path))
    stem = str(tmp_path / "export")
    written, _notes = write_feature_collection(features, stem)

    names = sorted(os.path.basename(p) for p in written)
    assert names == ["export_points.shp", "export_polygons.shp"], names


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def test_crs_label_is_filename_safe_and_distinct():
    a = crs_label('PROJCS["WGS_1984_UTM_Zone_20N",UNIT["Meter",1.0]]')
    b = crs_label('PROJCS["WGS_1984_UTM_Zone_21N",UNIT["Meter",1.0]]')

    assert a != b, "two different CRSs must not share a filename label"
    for lbl in (a, b):
        assert lbl and all(ch.isalnum() or ch == "_" for ch in lbl), lbl


def test_crs_label_for_no_crs_is_pixel():
    assert crs_label(None) == "pixel"


# ---------------------------------------------------------------------------
# .prj encoding
# ---------------------------------------------------------------------------
def test_prj_is_pure_ascii(tmp_path):
    """The .prj was written with a bare `open(path, 'w')`, i.e. in the machine's
    ANSI codepage. A pyproj WKT carries a degree sign in its AREA description
    ("Between 60 deg W and 54 deg W"), so the same export produced different
    BYTES on differently-configured machines, and the file was not the plain
    ASCII a .prj is expected to be. Caught when reading one back as UTF-8 blew
    up on 0xB0."""
    a = _write_geotiff(str(tmp_path / "a.tif"), 32621, 300000.0, 9000000.0)
    features, _crs, _warns = json_polygons_to_features(
        {"g": {a: _shape(a)}}, str(tmp_path))

    stem = str(tmp_path / "export")
    written, _notes = write_feature_collection(features, stem)
    prj_path = os.path.splitext(written[0])[0] + ".prj"

    raw = open(prj_path, "rb").read()
    raw.decode("ascii")            # must not raise
    assert b"\xb0" not in raw, "a cp1252 degree sign leaked into the .prj"
    assert raw.decode("ascii").startswith(("PROJCS", "PROJCRS", "GEOGCS", "GEOGCRS"))


def test_prj_is_wkt1_not_wkt2(tmp_path):
    """A shapefile's .prj holds WKT1 in ESRI's flavour.

    `pyproj.CRS.to_wkt()` defaults to WKT2_2019, and that is what was being
    written: a 1.6 KB string whose datum is expressed as `ENSEMBLE[...]`, a node
    that does not exist in WKT1. ArcGIS cannot read it at all and older
    GDAL/QGIS reject it and fall back to the project CRS -- which presents
    exactly as "my shapefile opens in a completely different coordinate
    system", the reported symptom, even when only ONE CRS is involved.
    """
    from ..shapefile_io import wkt_for_epsg

    wkt = wkt_for_epsg(32617)
    assert wkt, "no WKT produced for a common UTM zone"
    assert wkt.startswith("PROJCS["), (
        f"a .prj must be WKT1 (PROJCS[...]), got: {wkt[:60]}")
    assert "ENSEMBLE" not in wkt, (
        "ENSEMBLE[...] is WKT2-only and is unreadable in a .prj")

    a = _write_geotiff(str(tmp_path / "a.tif"), 32617, 647606.0, 1014471.0)
    features, _crs, _warns = json_polygons_to_features(
        {"g": {a: _shape(a)}}, str(tmp_path))
    written, _notes = write_feature_collection(features, str(tmp_path / "export"))
    on_disk = open(os.path.splitext(written[0])[0] + ".prj", encoding="ascii").read()

    assert on_disk.startswith("PROJCS["), on_disk[:80]
    assert "ENSEMBLE" not in on_disk
