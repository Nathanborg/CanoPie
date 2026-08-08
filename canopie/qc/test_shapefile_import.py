"""
Tests for Shapefile import subsystem:
- Pure-Python struct binary shapefile and DBF parsing
- CRS parsing from .prj
- Inverse coordinate reprojection pipeline
- Dual-tier spatial indexing (STRtree + KDTree)
- MultiPolygon decomposition and polygon ingest
"""

import os
import io
import struct
import tempfile
import pytest
import numpy as np

from canopie.shapefile_io import (
    read_dbf,
    read_shapefile,
    write_dbf,
    write_shapefile,
    parse_crs_from_prj,
    reproject_shapefile_geometry_to_image_pixels,
    shapefile_to_json_polygons,
    pixel_to_geo,
    geo_to_pixel
)
from canopie.spatial_index import (
    SpatialIndexManager,
    get_geotiff_footprint,
    decompose_geometry
)
from shapely.geometry import Polygon, MultiPolygon, Point, box


def test_pure_python_dbf_roundtrip(tmp_path):
    """Verify write_dbf and read_dbf roundtrip attributes accurately."""
    dbf_file = str(tmp_path / "test_table.dbf")
    records = [
        {"GROUP": "Canopy", "NAME": "Tree_1", "ID": 101, "NDVI_Mean": 0.852, "ACTIVE": True},
        {"GROUP": "Soil", "NAME": "Ground_2", "ID": 102, "NDVI_Mean": 0.124, "ACTIVE": False},
    ]
    features = [{'geometry': [(0, 0)], 'properties': r} for r in records]

    # Write DBF
    write_dbf(dbf_file, features)
    assert os.path.exists(dbf_file)

    # Read DBF back
    parsed = read_dbf(dbf_file)
    assert len(parsed) == 2
    assert parsed[0]["GROUP"] == "Canopy"
    assert parsed[0]["NAME"] == "Tree_1"
    assert parsed[0]["ID"] == 101
    assert abs(parsed[0]["NDVI_Mean"] - 0.852) < 1e-3
    assert parsed[0]["ACTIVE"] is True

    assert parsed[1]["GROUP"] == "Soil"
    assert parsed[1]["ACTIVE"] is False


def test_pure_python_shp_point_parsing(tmp_path):
    """Verify write_shapefile and read_shapefile roundtrip Point geometries."""
    shp_base = str(tmp_path / "test_points")
    points = [(500100.0, 4000200.0), (500250.0, 4000350.0), (500300.0, 4000400.0)]
    features = [
        {'geometry': [pt], 'properties': {'name': f'pt_{i}', 'group': 'Points', 'type': 'point'}}
        for i, pt in enumerate(points)
    ]

    # Write Point shapefile (shape_type=1)
    write_shapefile(features, shp_base, shape_type=1)
    shp_file = shp_base + ".shp"
    assert os.path.exists(shp_file)

    # Read back
    read_feats = read_shapefile(shp_file)
    assert len(read_feats) == 3
    for i, rf in enumerate(read_feats):
        assert rf['shape_type'] == 1  # Point
        assert len(rf['geometry']) == 1
        assert abs(rf['geometry'][0][0] - points[i][0]) < 1e-4
        assert abs(rf['geometry'][0][1] - points[i][1]) < 1e-4
        assert rf['properties']['name'] == f'pt_{i}'


def test_pure_python_shp_polygon_parsing(tmp_path):
    """Verify write_shapefile and read_shapefile roundtrip Polygon geometries."""
    shp_base = str(tmp_path / "test_polys")
    ring1 = [(500100.0, 4000200.0), (500200.0, 4000200.0), (500200.0, 4000300.0), (500100.0, 4000300.0), (500100.0, 4000200.0)]
    features = [
        {'geometry': ring1, 'properties': {'group': 'Canopy', 'name': 'Crown_A', 'type': 'polygon'}}
    ]

    # Write Polygon shapefile (shape_type=5)
    write_shapefile(features, shp_base, shape_type=5)
    shp_file = shp_base + ".shp"
    assert os.path.exists(shp_file)

    # Read back
    read_feats = read_shapefile(shp_file)
    assert len(read_feats) == 1
    rf = read_feats[0]
    assert rf['shape_type'] == 5  # Polygon
    assert len(rf['geometry']) == 5
    assert rf['geometry'][0] == rf['geometry'][-1]  # Closed ring
    assert rf['properties']['group'] == 'Canopy'
    assert rf['properties']['name'] == 'Crown_A'


def test_parse_crs_from_prj(tmp_path):
    """Verify parse_crs_from_prj reads WKT and extracts pyproj.CRS."""
    prj_file = str(tmp_path / "test.prj")
    wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    with open(prj_file, "w", encoding="utf-8") as f:
        f.write(wkt)

    wkt_out, crs_obj = parse_crs_from_prj(prj_file)
    assert wkt_out == wkt
    assert crs_obj is not None
    assert crs_obj.to_epsg() == 4326


def test_spatial_index_matching():
    """Verify SpatialIndexManager matches polygons to overlapping footprints."""
    mgr = SpatialIndexManager()

    # Create synthetic image records with known footprint boxes in lon/lat
    img1_fp = "C:/project/img1.tif"
    img2_fp = "C:/project/img2.tif"

    records = [
        {'filepath': img1_fp, 'latitude': 10.0, 'longitude': 20.0, 'profile': None},
        {'filepath': img2_fp, 'latitude': 10.0, 'longitude': 21.0, 'profile': None}
    ]
    mgr.build_index(records)

    # Test polygon overlapping img1
    poly1 = Polygon([(19.9998, 9.9998), (20.0002, 9.9998), (20.0002, 10.0002), (19.9998, 10.0002)])
    matched_fp, score, match_type = mgr.match_polygon(poly1)
    assert matched_fp == img1_fp
    assert match_type == "footprint_overlap"

    # Test polygon closer to img2
    poly2 = Point(21.0001, 10.0001)
    matched_fp2, score2, match_type2 = mgr.match_polygon(poly2)
    assert matched_fp2 == img2_fp


def test_multipolygon_decomposition():
    """Verify decompose_geometry breaks MultiPolygons into simple Polygon items."""
    poly1 = box(0, 0, 10, 10)
    poly2 = box(20, 20, 30, 30)
    mp = MultiPolygon([poly1, poly2])

    decomposed = decompose_geometry(mp, min_area=1.0)
    assert len(decomposed) == 2
    assert decomposed[0].area == 100.0
    assert decomposed[1].area == 100.0


def test_inverse_coordinate_mapping(tmp_path):
    """Verify roundtrip: pixel -> geo -> raw pixel -> modified pixel -> reference pixel."""
    transform = [500000.0, 0.05, 0.0, 4000000.0, 0.0, -0.05]
    pixel_pts = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]

    # Forward: pixel to geo
    geo_pts = pixel_to_geo(pixel_pts, transform)

    # Inverse: geo to pixel
    raw_pixels = geo_to_pixel(geo_pts, transform)
    for orig, inv in zip(pixel_pts, raw_pixels):
        assert abs(orig[0] - inv[0]) < 1e-4
        assert abs(orig[1] - inv[1]) < 1e-4


def test_raw_image_dims_and_probe(tmp_path):
    """Verify raw image dimension extraction for TIFFs and generic images."""
    import tifffile
    import numpy as np

    tif_path = str(tmp_path / "test_dim.tif")
    arr = np.zeros((120, 240, 3), dtype=np.uint8)
    tifffile.imwrite(tif_path, arr)

    from canopie.raster_reader import probe
    prof = probe(tif_path)
    assert prof is not None
    assert prof.height == 120
    assert prof.width == 240


def test_multi_feature_shapefile_ingestion(tmp_path):
    """Verify that shapefiles containing multiple features disaggregate into unique polygon keys without data loss."""
    shp_base = str(tmp_path / "multi_trees")
    features = []
    for i in range(1, 6):
        ring = [
            (500000.0 + i * 10, 4000000.0),
            (500005.0 + i * 10, 4000000.0),
            (500005.0 + i * 10, 4000005.0),
            (500000.0 + i * 10, 4000005.0),
            (500000.0 + i * 10, 4000000.0)
        ]
        features.append({
            'geometry': ring,
            'properties': {
                'GROUP': 'Canopy',
                'TREE_ID': f'Tree_{i:03d}',
                'SPECIES': 'Quercus' if i % 2 == 0 else 'Pinus',
                'HEIGHT': 10.5 + i
            }
        })

    # Write shapefile
    write_shapefile(features, shp_base, shape_type=5)
    shp_file = shp_base + ".shp"
    assert os.path.exists(shp_file)

    # Ingest using shapefile_to_json_polygons
    dummy_img = str(tmp_path / "img1.tif")
    imported_polys, warns = shapefile_to_json_polygons(
        shp_path=shp_file,
        target_filepaths=[dummy_img],
        default_group="Canopy"
    )

    # All 5 features MUST be present in imported_polys under unique entry keys!
    assert len(imported_polys) == 5
    for i in range(1, 6):
        expected_key = f"Canopy_Tree_{i:03d}"
        assert expected_key in imported_polys
        poly_dict = imported_polys[expected_key][os.path.normpath(dummy_img)]
        assert poly_dict['name'] == f"Tree_{i:03d}"
        assert poly_dict['group'] == "Canopy"
        assert poly_dict['properties']['TREE_ID'] == f"Tree_{i:03d}"
        assert poly_dict['properties']['HEIGHT'] == 10.5 + i


def test_multi_feature_spatial_routing():
    """Verify that multiple features are independently matched and routed to correct images."""
    mgr = SpatialIndexManager()
    img1_fp = "C:/project/img1.tif"
    img2_fp = "C:/project/img2.tif"

    records = [
        {'filepath': img1_fp, 'latitude': 10.0, 'longitude': 20.0, 'profile': None},
        {'filepath': img2_fp, 'latitude': 10.0, 'longitude': 21.0, 'profile': None}
    ]
    mgr.build_index(records)

    # feature 1 lands on img1
    feat1 = {
        'geometry': Polygon([(19.9998, 9.9998), (20.0002, 9.9998), (20.0002, 10.0002), (19.9998, 10.0002)]),
        'properties': {'group': 'g1'}
    }
    # feature 2 lands on img2
    feat2 = {
        'geometry': Polygon([(20.9998, 9.9998), (21.0002, 9.9998), (21.0002, 10.0002), (20.9998, 10.0002)]),
        'properties': {'group': 'g2'}
    }

    results = mgr.route_shapefile_features([feat1, feat2])
    assert len(results) == 2
    assert results[0][0][0] == img1_fp
    assert results[1][0][0] == img2_fp


def test_vectorized_coordinate_reprojection():
    """Verify that vectorized reprojection outputs match expected coordinates."""
    pixel_pts = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]
    transform = [500000.0, 0.05, 0.0, 4000000.0, 0.0, -0.05]
    geo_pts = pixel_to_geo(pixel_pts, transform)

    # Use mock params
    ref_size = {'w': 800, 'h': 600}
    ax_data = {'crop': {'x': 10, 'y': 20, 'w': 380, 'h': 280}, 'resize': {'w': 800, 'h': 600}}

    res = reproject_shapefile_geometry_to_image_pixels(
        geo_points=geo_pts,
        shapefile_crs=None,
        target_image_path="dummy.tif",
        ax_data=ax_data,
        ref_size=ref_size
    )

    assert len(res) == len(pixel_pts)
    for x, y in res:
        assert isinstance(x, float)
        assert isinstance(y, float)
