"""
Spatial indexing and footprint matching engine for CanoPie.
Provides dual-tier spatial search (STRtree over image bounding footprints
and KDTree over camera GPS centers) to match vector geometries (Shapefiles)
to project raster images.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point, box, shape
from shapely.strtree import STRtree
from scipy.spatial import KDTree
from geopy.distance import geodesic

import pyproj
from .shapefile_io import get_geotiff_transform, estimate_transform_from_exif, _raw_image_dims


def get_geotiff_footprint(profile_or_filepath, target_epsg: int = 4326) -> Optional[Polygon]:
    """
    Extracts the bounding polygon footprint for a GeoTIFF image in target EPSG (default WGS84).
    """
    transform = None
    epsg = None
    W, H = None, None

    if isinstance(profile_or_filepath, str):
        filepath = profile_or_filepath
        raw_h, raw_w = _raw_image_dims(filepath)
        W, H = raw_w, raw_h
        transform, crs_wkt = get_geotiff_transform(filepath)
        if not transform and W and H:
            transform, crs_wkt = estimate_transform_from_exif(filepath, W, H)
        if crs_wkt:
            try:
                crs_obj = pyproj.CRS.from_user_input(crs_wkt)
                epsg = crs_obj.to_epsg()
            except Exception:
                epsg = None
    else:
        profile = profile_or_filepath
        W, H = profile.width, profile.height
        transform = getattr(profile, 'transform', None)
        epsg = getattr(profile, 'epsg', None)

    if not transform or not W or not H or W <= 0 or H <= 0:
        return None

    # 4 corners in pixel space: (0,0), (W,0), (W,H), (0,H)
    pixel_corners = [
        (0.0, 0.0),
        (float(W), 0.0),
        (float(W), float(H)),
        (0.0, float(H))
    ]

    c, a, b, f, d, e = transform[:6]
    ground_corners = []
    for P, L in pixel_corners:
        X_g = c + a * P + b * L
        Y_g = f + d * P + e * L
        ground_corners.append((X_g, Y_g))

    poly_native = Polygon(ground_corners)

    # Reproject to target_epsg if needed and known
    if epsg and epsg != target_epsg:
        try:
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{epsg}", f"EPSG:{target_epsg}", always_xy=True
            )
            reprojected = [transformer.transform(x, y) for x, y in poly_native.exterior.coords]
            return Polygon(reprojected)
        except Exception as exc:
            logging.debug("Footprint reprojection failed for EPSG:%s: %s", epsg, exc)

    return poly_native


def decompose_geometry(geom, min_area: float = 0.0) -> List[Polygon]:
    """
    Recursively decomposes MultiPolygons and GeometryCollections into a flat
    list of simple Shapely Polygon objects, filtering out non-polygonal artifacts.
    """
    polygons = []
    if geom is None or geom.is_empty:
        return polygons

    gtype = geom.geom_type
    if gtype == 'Polygon':
        if min_area <= 0.0 or geom.area >= min_area:
            polygons.append(geom)
    elif gtype == 'MultiPolygon':
        for poly in geom.geoms:
            polygons.extend(decompose_geometry(poly, min_area=min_area))
    elif gtype == 'GeometryCollection':
        for part in geom.geoms:
            polygons.extend(decompose_geometry(part, min_area=min_area))

    return polygons


class SpatialIndexManager:
    """
    Dual-Tier Spatial Index Manager for matching Shapefile polygons to project images.
    - Tier 1: STRtree over image footprint bounding polygons (computes geometric overlap).
    - Tier 2: KDTree over camera GPS center positions (finds closest camera node).
    """

    def __init__(self):
        self.image_records: List[Dict[str, Any]] = []
        self.footprint_polygons: List[Polygon] = []
        self.str_tree: Optional[STRtree] = None
        self.kd_tree: Optional[KDTree] = None
        self.kd_coordinates: List[Tuple[float, float]] = []
        # Cache of {crs cache key: transformer-or-None} for the source-CRS ->
        # EPSG:4326 hop in match_feature_geometry. Every feature in a shapefile
        # shares one CRS, so this is built at most once per import -- see the
        # comment there for why that matters so much.
        self._wgs84_transformer_cache: Dict[Any, Any] = {}

    @staticmethod
    def _crs_cache_key(shapefile_crs) -> Any:
        """A cheap, stable key for a CRS input.

        Deliberately avoids `str(crs)` for pyproj CRS objects: that serialises
        the full WKT (kilobytes) on every call, which is itself measurable when
        done per feature. `id()` is safe here because the caller holds one CRS
        object for the whole import, and the cache lives only as long as this
        manager.
        """
        if isinstance(shapefile_crs, str):
            return shapefile_crs
        return ('id', id(shapefile_crs))

    def _get_wgs84_transformer(self, shapefile_crs):
        """Return a cached source-CRS -> EPSG:4326 transformer (or None when the
        source already IS 4326 and no transform is needed).

        THE HOT PATH. `match_feature_geometry` used to call
        `pyproj.CRS.from_user_input()` and `pyproj.Transformer.from_crs()`
        inline, so both ran once PER FEATURE. Building a PROJ transformation
        pipeline is expensive -- it hits the PROJ database and searches for
        candidate operations/grids -- and profiling the real worker on 2000
        synthetic features measured 26.1 s of a 29.1 s total (90%) inside
        `Transformer.__init__` alone, i.e. ~13 ms per feature purely to
        reconstruct an object identical to the previous one. Extrapolated to a
        6000-feature shapefile that is ~80 s of pure waste, which is the
        difference between "takes forever" and QGIS-like instant.
        """
        key = self._crs_cache_key(shapefile_crs)
        if key in self._wgs84_transformer_cache:
            return self._wgs84_transformer_cache[key]

        transformer = None
        try:
            import pyproj
            src_crs = pyproj.CRS.from_user_input(shapefile_crs)
            if src_crs.to_epsg() != 4326:
                transformer = pyproj.Transformer.from_crs(
                    src_crs, "EPSG:4326", always_xy=True)
        except Exception as e:
            logging.debug("Failed to build WGS84 transformer: %s", e)
            transformer = None

        self._wgs84_transformer_cache[key] = transformer
        return transformer

    def build_index(self, image_records: List[Dict[str, Any]]):
        """
        Builds Tier 1 (STRtree) and Tier 2 (KDTree) indices for a list of image records.
        Each record should contain 'filepath' and optional 'latitude', 'longitude', 'profile'.
        """
        self.image_records.clear()
        self.footprint_polygons.clear()
        self.kd_coordinates.clear()

        for rec in image_records:
            fp_path = rec['filepath']
            profile = rec.get('profile')
            center_lat = rec.get('latitude')
            center_lon = rec.get('longitude')

            footprint = None
            try:
                footprint = get_geotiff_footprint(profile or fp_path, target_epsg=4326)
            except Exception:
                footprint = None

            # Fallback footprint from EXIF point coordinates
            if footprint is None and center_lat is not None and center_lon is not None:
                delta = 0.0005  # ~55 meters bounding box
                footprint = Polygon([
                    (center_lon - delta, center_lat - delta),
                    (center_lon + delta, center_lat - delta),
                    (center_lon + delta, center_lat + delta),
                    (center_lon - delta, center_lat + delta)
                ])

            if footprint is not None:
                rec_idx = len(self.image_records)
                self.image_records.append({
                    'index': rec_idx,
                    'filepath': fp_path,
                    'footprint': footprint,
                    'center': (center_lat, center_lon) if (center_lat is not None and center_lon is not None) else None,
                    'profile': profile
                })
                self.footprint_polygons.append(footprint)

            if center_lat is not None and center_lon is not None:
                self.kd_coordinates.append((center_lat, center_lon))

        # Build Tier 1 STRtree
        if self.footprint_polygons:
            self.str_tree = STRtree(self.footprint_polygons)

        # Build Tier 2 KDTree
        if self.kd_coordinates:
            self.kd_tree = KDTree(self.kd_coordinates)

    def match_polygon(self, poly_geom_or_coords: Union[Polygon, List[Tuple[float, float]]], min_overlap_ratio: float = 0.01) -> Tuple[Optional[str], float, str]:
        """
        Matches a Shapefile polygon to the best target image.

        Returns
        -------
        tuple of (best_filepath, score, match_type)
        """
        matches = self.match_feature_geometry(poly_geom_or_coords, min_overlap_ratio=min_overlap_ratio, match_all_overlapping=False)
        if matches:
            return matches[0]
        return None, 0.0, "unmatched"

    def match_feature_geometry(
        self,
        poly_geom_or_coords: Union[Polygon, Point, List[Tuple[float, float]], Any],
        shapefile_crs: Any = None,
        min_overlap_ratio: float = 0.01,
        match_all_overlapping: bool = False
    ) -> List[Tuple[str, float, str]]:
        """
        Independently matches a single vector feature geometry to target image(s).
        Reprojects to EPSG:4326 if shapefile_crs is provided.
        """
        if not self.image_records:
            return []

        if isinstance(poly_geom_or_coords, list):
            if len(poly_geom_or_coords) < 3:
                if len(poly_geom_or_coords) == 1:
                    poly_geom = Point(poly_geom_or_coords[0][0], poly_geom_or_coords[0][1])
                else:
                    poly_geom = Polygon(poly_geom_or_coords)
            else:
                poly_geom = Polygon(poly_geom_or_coords)
        else:
            poly_geom = poly_geom_or_coords

        if poly_geom is None or poly_geom.is_empty:
            return []

        geom_wgs84 = poly_geom
        if shapefile_crs:
            try:
                # Cached: building the CRS + Transformer here inline made this
                # ~13 ms per feature (90% of total import time). See
                # _get_wgs84_transformer.
                transformer = self._get_wgs84_transformer(shapefile_crs)
                if transformer is not None:
                    from shapely.ops import transform
                    geom_wgs84 = transform(transformer.transform, poly_geom)
            except Exception as e:
                logging.debug("Failed to reproject feature geometry to EPSG:4326: %s", e)

        matches = []

        # Tier 1: Footprint Spatial Containment / Overlap
        if self.str_tree is not None:
            candidate_indices = self.str_tree.query(geom_wgs84)
            if hasattr(candidate_indices, '__len__') and len(candidate_indices) > 0:
                poly_area = geom_wgs84.area
                for idx in candidate_indices:
                    footprint = self.footprint_polygons[idx]
                    try:
                        inter = geom_wgs84.intersection(footprint)
                        inter_area = inter.area
                        ioa = (inter_area / poly_area) if poly_area > 0 else (1.0 if not inter.is_empty else 0.0)
                        if ioa >= min_overlap_ratio:
                            matches.append((self.image_records[idx]['filepath'], ioa, "footprint_overlap"))
                    except Exception:
                        pass

        # Tier 2: Point KDTree distance fallback
        if not matches and self.kd_tree is not None and self.kd_coordinates:
            centroid = geom_wgs84.centroid
            poly_center = (centroid.y, centroid.x)  # (lat, lon)

            dist, idx = self.kd_tree.query(poly_center)
            if idx < len(self.image_records):
                closest_rec = self.image_records[idx]
                if closest_rec['center']:
                    dist_meters = geodesic(poly_center, closest_rec['center']).meters
                    matches.append((closest_rec['filepath'], dist_meters, "kdtree_distance"))

        # Default fallback to first image
        if not matches and self.image_records:
            matches.append((self.image_records[0]['filepath'], 0.0, "fallback_first"))

        if match_all_overlapping and len(matches) > 1:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches
        elif matches:
            best_match = max(matches, key=lambda x: x[1])
            return [best_match]
        return []

    def route_shapefile_features(
        self,
        features: List[Dict[str, Any]],
        shapefile_crs: Any = None,
        min_overlap_ratio: float = 0.01,
        match_all_overlapping: bool = False
    ) -> Dict[int, List[Tuple[str, float, str]]]:
        """
        Batch spatial router for multi-feature shapefiles.
        Returns dict mapping feature_index -> List of (filepath, score, match_type).
        """
        routing_results = {}
        for idx, feat in enumerate(features):
            geom = feat.get('geometry')
            if not geom:
                continue
            matches = self.match_feature_geometry(
                poly_geom_or_coords=geom,
                shapefile_crs=shapefile_crs,
                min_overlap_ratio=min_overlap_ratio,
                match_all_overlapping=match_all_overlapping
            )
            routing_results[idx] = matches
        return routing_results
