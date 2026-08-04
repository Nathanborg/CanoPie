import os
import struct
import math
import json
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np
import tifffile
import geopy.distance
from canopie.utils import get_exif_data_exiftool_multiple

WGS84_WKT = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
EPSG_WKT = {
    4326: WGS84_WKT,
}

_WGS84_GEOGCS = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                 'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                 'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


def wkt_for_epsg(code):
    """
    Best-effort WKT for an EPSG code, for writing the ``.prj`` sidecar.

    Without a ``.prj`` a GIS has to guess the CRS: a UTM shapefile (metres, e.g.
    500000/4000000) opened against a lat/lon project is placed absurdly far from
    the raster it came from. The original table only held EPSG:4326, so EVERY
    projected raster exported with no ``.prj`` at all.

    Order: pyproj if installed (authoritative, covers everything) -> the explicit
    table -> formulaic WGS84/UTM (EPSG 326xx north, 327xx south), which covers the
    overwhelming majority of drone and airborne products. Returns None if unknown,
    in which case the caller should warn rather than write a guessed projection.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None

    try:
        import pyproj
        return pyproj.CRS.from_epsg(code).to_wkt()
    except Exception:
        pass

    if code in EPSG_WKT:
        return EPSG_WKT[code]

    # WGS84 / UTM: 326zz = zone zz North, 327zz = zone zz South.
    zone = None
    north = True
    if 32601 <= code <= 32660:
        zone, north = code - 32600, True
    elif 32701 <= code <= 32760:
        zone, north = code - 32700, False
    if zone is not None:
        central_meridian = zone * 6 - 183
        false_northing = 0.0 if north else 10000000.0
        hemi = 'N' if north else 'S'
        return (f'PROJCS["WGS_1984_UTM_Zone_{zone}{hemi}",{_WGS84_GEOGCS},'
                'PROJECTION["Transverse_Mercator"],'
                'PARAMETER["False_Easting",500000.0],'
                f'PARAMETER["False_Northing",{false_northing}],'
                f'PARAMETER["Central_Meridian",{float(central_meridian)}],'
                'PARAMETER["Scale_Factor",0.9996],'
                'PARAMETER["Latitude_Of_Origin",0.0],'
                'UNIT["Meter",1.0]]')
    return None


def epsg_from_geotiff(filepath):
    """Return the EPSG code declared in a GeoTIFF's GeoKeyDirectory, or None."""
    try:
        with tifffile.TiffFile(filepath) as tif:
            gk = tif.pages[0].tags.get('GeoKeyDirectoryTag')
            if not gk:
                return None
            keys = gk.value
            # ProjectedCSTypeGeoKey (3072) wins over GeographicTypeGeoKey (2048).
            found = None
            for i in range(4, len(keys), 4):
                key_id, _loc, _count, value_offset = keys[i:i + 4]
                if key_id == 3072 and value_offset:
                    return int(value_offset)
                if key_id == 2048 and value_offset and found is None:
                    found = int(value_offset)
            return found
    except Exception:
        return None


def _dbf_safe_fieldname(name, used_upper):
    """
    Produce a DBF-legal field name (<=10 chars, ASCII) that doesn't collide with an
    already-used name.

    DBF's 10-character limit means real statistic column names ("Green_Mean",
    "NDVI_Mean", "Class 0 %"...) truncate to the same prefix easily. The previous
    writer truncated with no de-duplication at all, so two different CSV columns
    could silently collide into one DBF field. Collisions here are resolved with a
    numeric suffix that still respects the 10-char limit; the caller is expected to
    keep a name_map (original -> dbf name) and write it out as a legend file so the
    short names stay traceable back to the source CSV columns.
    """
    base = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name)) or "FIELD"
    base = base[:10]
    candidate = base
    n = 1
    while candidate.upper() in used_upper:
        suffix = str(n)
        candidate = base[:max(1, 10 - len(suffix))] + suffix
        n += 1
    used_upper.add(candidate.upper())
    return candidate


def write_dbf(dbf_path, features, legend_path=None):
    """
    Write a DBF file using struct.

    Schema is derived from the UNION of every feature's property keys, not just the
    first feature's — once per-shape statistics are embedded, later features
    legitimately carry keys earlier ones lack (e.g. RF classification columns only
    exist when a model was used; multispectral band counts vary by image). Deriving
    the schema from feature 0 alone silently dropped every such column.

    `legend_path`, if given, gets a small "dbf_field,original_name" CSV so a field
    truncated/renamed for the 10-char DBF limit stays traceable to its real name.
    """
    if not features:
        return

    # 1) Ordered union of property keys across ALL features.
    all_keys = []
    seen_keys = set()
    for feat in features:
        for k in feat['properties'].keys():
            if k not in seen_keys:
                seen_keys.add(k)
                all_keys.append(k)

    # 2) Infer type + width by scanning every feature's value for that key — not
    #    just the first feature, which could easily have an unrepresentative value
    #    (None, or an int where later rows are float).
    used_upper = set()
    fields = []          # (original_key, dbf_name, type, length, dec)
    name_map = {}         # original_key -> dbf_name (for the legend)
    for k in all_keys:
        values = [feat['properties'].get(k) for feat in features]
        non_none = [v for v in values if v is not None]

        dbf_name = _dbf_safe_fieldname(k, used_upper)
        name_map[k] = dbf_name

        if non_none and all(isinstance(v, bool) for v in non_none):
            fields.append((k, dbf_name, 'L', 1, 0))
        elif non_none and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none):
            is_int = all(float(v).is_integer() for v in non_none)
            if is_int:
                width = max([len(str(int(v))) for v in non_none] + [1])
                fields.append((k, dbf_name, 'N', min(18, max(3, width + 1)), 0))
            else:
                fields.append((k, dbf_name, 'F', 19, 6))
        else:
            width = max([len(str(v)) for v in non_none] + [1]) if non_none else 1
            fields.append((k, dbf_name, 'C', min(254, max(10, width)), 0))

    num_records = len(features)
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(f[3] for f in fields)

    with open(dbf_path, 'wb') as f:
        # DBF Header
        now = datetime.now()
        f.write(struct.pack("<BBB", 3, now.year - 1900, now.month))
        f.write(struct.pack("<B", now.day))
        f.write(struct.pack("<i", num_records))
        f.write(struct.pack("<HH", header_length, record_length))
        f.write(b'\x00' * 20)

        # Field descriptors
        for _orig, dbf_name, typ, length, dec in fields:
            name_bytes = dbf_name.encode('ascii', 'ignore').ljust(11, b'\x00')
            f.write(name_bytes)
            f.write(typ.encode('ascii'))
            f.write(struct.pack("<i", 0))
            f.write(struct.pack("<BB", length, dec))
            f.write(b'\x00' * 14)

        f.write(b'\x0D')

        # Records
        for feature in features:
            props = feature['properties']
            f.write(b' ')  # Valid record marker
            for orig_key, _dbf_name, typ, length, dec in fields:
                val = props.get(orig_key)
                if val is None:
                    val_str = ''.ljust(length)
                elif typ == 'L':
                    val_str = ('T' if val else 'F')
                elif typ == 'N':
                    try:
                        val_str = str(int(round(float(val)))).rjust(length)
                    except (TypeError, ValueError):
                        val_str = ''.rjust(length)
                elif typ == 'F':
                    try:
                        val_str = f"{float(val):.{dec}f}".rjust(length)
                    except (TypeError, ValueError):
                        val_str = ''.rjust(length)
                else:
                    val_str = str(val).encode('ascii', 'ignore').decode('ascii').ljust(length)
                val_str = val_str[:length]
                f.write(val_str.encode('ascii', 'ignore').ljust(length, b' '))

        f.write(b'\x1A')

    if legend_path and any(k != v for k, v in name_map.items()):
        try:
            with open(legend_path, 'w', encoding='utf-8') as lf:
                lf.write("dbf_field,original_name\n")
                for orig, short in name_map.items():
                    safe_orig = str(orig).replace('"', '""')
                    lf.write(f'{short},"{safe_orig}"\n')
        except Exception as e:
            logging.warning(f"Failed to write field-name legend: {e}")


def write_shapefile(features, output_path, crs_wkt=None, shape_type=5, legend_path=None):
    """
    Write an ESRI Shapefile using struct. output_path should not include the extension.

    shape_type: 1 = Point, 5 = Polygon. A single .shp is homogeneous — ESRI's format
    declares one shape type in its header for the whole file. Callers with a mix of
    point and polygon features must call this twice, once per type (see
    polygon_manager.py's shapefile export, which splits on `properties['type']`).

    For shape_type == 1, each feature's geometry must be a single [(x, y)] pair.
    For shape_type == 5, each feature's geometry is the polygon ring's points.
    """
    if not features:
        logging.warning("No features to write.")
        return

    # Keep exactly the features that have usable geometry, and reuse THIS SAME
    # filtered list for shp, shx, AND dbf. Previously shp/shx silently skipped
    # empty-geometry features while write_dbf() still got the FULL, unfiltered
    # list — after any skip, every subsequent DBF row then described the WRONG
    # geometry, since GIS readers match dbf row i to shp record i by position.
    written = [f for f in features if f.get('geometry')]
    if not written:
        logging.warning("No features with geometry to write.")
        return

    shp_path = f"{output_path}.shp"
    shx_path = f"{output_path}.shx"
    dbf_path = f"{output_path}.dbf"
    prj_path = f"{output_path}.prj"

    if crs_wkt:
        with open(prj_path, 'w') as f:
            f.write(crs_wkt)

    # Calculate bounding box
    all_x = []
    all_y = []
    for feat in written:
        for x, y in feat['geometry']:
            all_x.append(x)
            all_y.append(y)

    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0

    # Shapefile writing
    shp_records = []
    shx_records = []

    # Pre-build shp records to calculate file length
    offset = 50  # 100 bytes = 50 16-bit words
    for i, feature in enumerate(written):
        geom = feature['geometry']

        if shape_type == 1:
            # Point (Type 1): 4 (type) + 16 (x, y) = 20 bytes, no bbox/parts.
            x, y = geom[0]
            content_bytes = 4 + 16
            content_words = content_bytes // 2
            rec_hdr = struct.pack(">ii", i + 1, content_words)
            content = struct.pack("<i", 1)
            content += struct.pack("<2d", x, y)
        else:
            # Ensure closure
            geom = list(geom)
            if geom[0] != geom[-1]:
                geom = geom + [geom[0]]

            num_points = len(geom)
            num_parts = 1

            pts_x = [p[0] for p in geom]
            pts_y = [p[1] for p in geom]
            f_min_x, f_max_x = min(pts_x), max(pts_x)
            f_min_y, f_max_y = min(pts_y), max(pts_y)

            # Content length in 16-bit words:
            # Polygon (Type 5):
            # 4 (type) + 32 (box) + 4 (num parts) + 4 (num points) + 4*num_parts (parts) + 16*num_points (points)
            content_bytes = 44 + 4 * num_parts + 16 * num_points
            content_words = content_bytes // 2

            # Record Header (8 bytes = 4 words)
            rec_hdr = struct.pack(">ii", i + 1, content_words)

            # Record Content
            content = struct.pack("<i", 5)
            content += struct.pack("<4d", f_min_x, f_min_y, f_max_x, f_max_y)
            content += struct.pack("<ii", num_parts, num_points)
            content += struct.pack("<i", 0)  # part 0 index
            for x, y in geom:
                content += struct.pack("<2d", x, y)

        shp_records.append(rec_hdr + content)
        shx_records.append(struct.pack(">ii", offset, content_words))

        offset += 4 + content_words

    file_length_words = offset

    # SHP Header
    shp_header = struct.pack(">i20xi", 9994, file_length_words)
    shp_header += struct.pack("<i", 1000)
    shp_header += struct.pack("<i", shape_type)
    shp_header += struct.pack("<8d", min_x, min_y, max_x, max_y, 0.0, 0.0, 0.0, 0.0)

    with open(shp_path, 'wb') as f:
        f.write(shp_header)
        for rec in shp_records:
            f.write(rec)

    # SHX Header
    shx_file_length_words = 50 + 4 * len(shx_records)
    shx_header = struct.pack(">i20xi", 9994, shx_file_length_words)
    shx_header += struct.pack("<i", 1000)
    shx_header += struct.pack("<i", shape_type)
    shx_header += struct.pack("<8d", min_x, min_y, max_x, max_y, 0.0, 0.0, 0.0, 0.0)

    with open(shx_path, 'wb') as f:
        f.write(shx_header)
        for rec in shx_records:
            f.write(rec)

    # DBF writing — same `written` list as shp/shx, so record N always describes
    # the correct geometry.
    try:
        write_dbf(dbf_path, written, legend_path=legend_path)
    except Exception as e:
        logging.warning(f"Error writing DBF file: {e}")


def get_geotiff_transform(filepath):
    """
    Extract affine transform from GeoTIFF using tifffile.TiffFile.
    """
    try:
        with tifffile.TiffFile(filepath) as tif:
            tags = tif.pages[0].tags
            
            model_tiepoint = tags.get('ModelTiepointTag')
            model_pixel_scale = tags.get('ModelPixelScaleTag')
            
            if model_tiepoint and model_pixel_scale:
                tp = model_tiepoint.value
                scale = model_pixel_scale.value
                
                I, J, K, X, Y, Z = tp[0], tp[1], tp[2], tp[3], tp[4], tp[5]
                scale_X, scale_Y, scale_Z = scale[0], scale[1], scale[2]
                
                origin_x = X - I * scale_X
                origin_y = Y + J * scale_Y
                
                transform_list = [origin_x, scale_X, 0, origin_y, 0, -scale_Y]
                
                # Resolve the CRS through wkt_for_epsg (pyproj / table / UTM formula)
                # instead of the old EPSG:4326-only lookup, which left every
                # projected raster without a .prj sidecar.
                crs_wkt = None
                geo_key_dir = tags.get('GeoKeyDirectoryTag')
                if geo_key_dir:
                    keys = geo_key_dir.value
                    epsg_geog = None
                    for i in range(4, len(keys), 4):
                        key_id, tiff_tag_location, count, value_offset = keys[i:i+4]
                        if key_id == 3072 and value_offset:      # ProjectedCSTypeGeoKey
                            crs_wkt = wkt_for_epsg(value_offset)
                            if crs_wkt:
                                break
                        elif key_id == 2048 and value_offset and epsg_geog is None:
                            epsg_geog = value_offset             # GeographicTypeGeoKey
                    if crs_wkt is None and epsg_geog is not None:
                        crs_wkt = wkt_for_epsg(epsg_geog)

                return transform_list, crs_wkt
    except Exception as e:
        logging.warning(f"Failed to read GeoTIFF transform for {filepath}: {e}")
        
    return None, None


def pixel_to_geo(points, transform):
    """
    Convert list of (col, row) pixel coordinates to (x_geo, y_geo).
    """
    geo_points = []
    for col, row in points:
        x_geo = transform[0] + col * transform[1] + row * transform[2]
        y_geo = transform[3] + col * transform[4] + row * transform[5]
        geo_points.append((x_geo, y_geo))
    return geo_points


def geo_to_pixel(points, transform):
    """
    Convert list of (x_geo, y_geo) geographic coordinates to (col, row) pixel coordinates.
    """
    det = transform[1] * transform[5] - transform[2] * transform[4]
    if det == 0:
        return []
    
    inv_transform = [
        (transform[0] * transform[5] - transform[3] * transform[2]) / -det,
        transform[5] / det,
        -transform[2] / det,
        (transform[3] * transform[1] - transform[0] * transform[4]) / -det,
        -transform[4] / det,
        transform[1] / det
    ]
    
    pixel_points = []
    for x, y in points:
        col = inv_transform[0] + x * inv_transform[1] + y * inv_transform[2]
        row = inv_transform[3] + x * inv_transform[4] + y * inv_transform[5]
        pixel_points.append((col, row))
    return pixel_points


def _convert_exifread_gps(value, ref):
    if not value or not ref: return None
    try:
        d = float(value.values[0].num) / max(1, float(value.values[0].den))
        m = float(value.values[1].num) / max(1, float(value.values[1].den))
        s = float(value.values[2].num) / max(1, float(value.values[2].den))
        deg = d + (m / 60.0) + (s / 3600.0)
        if ref.values in ('S', 'W'):
            deg = -deg
        return deg
    except Exception:
        return None

def estimate_transform_from_exif(filepath, image_w, image_h):
    """
    Estimate affine transform for non-georeferenced images with EXIF GPS data.

    Uses GPS lat/lon as image centre, altitude + focal length + sensor width
    to compute GSD, and flight yaw for rotation.

    Returns (transform_list, WGS84_WKT) or (None, None) if data is insufficient.
    """
    try:
        lat = lon = alt = focal_length = sensor_width = yaw = None
        
        # 1. Try exiftool first (best coverage for DJI drone metadata)
        exif_dict = get_exif_data_exiftool_multiple([filepath])
        if exif_dict:
            abs_fp = os.path.abspath(filepath)
            exif = exif_dict.get(abs_fp) or next(iter(exif_dict.values()), None)
            if exif:
                lat = exif.get('GPSLatitude')
                lon = exif.get('GPSLongitude')
                alt = exif.get('RelativeAltitude') or exif.get('GPSAltitude')
                focal_length = exif.get('FocalLength')
                sensor_width = exif.get('SensorWidth')
                if not sensor_width:
                    fl35 = exif.get('FocalLengthIn35mmFormat')
                    if fl35 and focal_length:
                        sensor_width = (float(focal_length) / float(fl35)) * 36.0
                yaw = exif.get('FlightYawDegree') or exif.get('GimbalYawDegree')
                
        # 2. Pure-Python Fallback (exifread + XMP regex) if exiftool failed/missing
        if lat is None or lon is None or alt is None:
            import exifread, re
            logging.getLogger("exifread").setLevel(logging.ERROR)
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
            lat = _convert_exifread_gps(tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef'))
            lon = _convert_exifread_gps(tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef'))
            
            alt_tag = tags.get('GPS GPSAltitude')
            if alt_tag:
                try:
                    alt = float(alt_tag.values[0].num) / max(1, float(alt_tag.values[0].den))
                except Exception: pass
                
            fl_tag = tags.get('EXIF FocalLength')
            if fl_tag:
                try:
                    focal_length = float(fl_tag.values[0].num) / max(1, float(fl_tag.values[0].den))
                except Exception: pass
                
            fl35_tag = tags.get('EXIF FocalLengthIn35mmFilm')
            if fl35_tag and focal_length:
                try:
                    sensor_width = (float(focal_length) / float(fl35_tag.values[0])) * 36.0
                except Exception: pass
                
            # Regex scan for DJI XMP Yaw
            try:
                with open(filepath, 'rb') as f:
                    content = f.read(100000) # XMP is usually near the start
                match = re.search(br'FlightYawDegree="([^"]+)"', content) or re.search(br'GimbalYawDegree="([^"]+)"', content)
                if match:
                    yaw = float(match.group(1))
            except Exception: pass

        if lat is None or lon is None or alt is None:
            return None, None

        # Ensure numeric
        lat, lon, alt = float(lat), float(lon), float(alt)
        if alt <= 0:
            alt = 50.0 # Fallback default altitude in meters if <= 0

        if focal_length is None or sensor_width is None:
            return None, None
            
        focal_length = float(focal_length)
        sensor_width = float(sensor_width)
        if focal_length <= 0 or sensor_width <= 0:
            return None, None

        yaw = float(yaw) if yaw is not None else 0.0

        # GSD in metres/pixel
        gsd = (alt * sensor_width) / (focal_length * image_w)

        # Degrees per pixel
        deg_lat_per_pixel = gsd / 111320.0
        deg_lon_per_pixel = gsd / (111320.0 * math.cos(math.radians(lat)))

        yaw_rad = math.radians(yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Affine coefficients (pixel → geo, Y positive up but row increases down)
        a = deg_lon_per_pixel * cos_y
        b = deg_lon_per_pixel * sin_y
        d = deg_lat_per_pixel * sin_y
        e = -deg_lat_per_pixel * cos_y

        # Origin so that image centre maps to (lon, lat)
        cx, cy = image_w / 2.0, image_h / 2.0
        origin_x = lon - (a * cx + b * cy)
        origin_y = lat - (d * cx + e * cy)

        transform = [origin_x, a, b, origin_y, d, e]
        return transform, WGS84_WKT

    except Exception as e:
        logging.warning(f"EXIF estimation failed for {filepath}: {e}")
        return None, None


def get_ax_transform_matrix(ax, raw_w, raw_h):
    """
    Returns (M, out_w, out_h) where M is a 3x3 matrix mapping RAW pixel coordinates
    to MODIFIED (.ax) pixel coordinates.
    """
    M = np.eye(3, dtype=np.float64)
    if not ax or not isinstance(ax, dict):
        return M, raw_w, raw_h

    cur_w, cur_h = float(raw_w), float(raw_h)

    # 1. Rotate & Crop Order Detection
    rot = int(ax.get("rotate", 0)) if ax.get("rotate_enabled", True) else 0
    if rot not in (90, 180, 270):
        rot = 0

    crop_enabled = bool(ax.get("crop_enabled", True))
    crop_rect_raw = ax.get("crop_rect")
    crop_ref_raw = ax.get("crop_rect_ref_size")

    def _parse_ref(ref_obj, default_w, default_h):
        if isinstance(ref_obj, dict):
            w = ref_obj.get("w") or ref_obj.get("width") or default_w
            h = ref_obj.get("h") or ref_obj.get("height") or default_h
            return float(w), float(h)
        elif isinstance(ref_obj, (list, tuple)) and len(ref_obj) >= 2:
            return float(ref_obj[0]), float(ref_obj[1])
        return float(default_w), float(default_h)

    def _parse_crop(crop_obj):
        if isinstance(crop_obj, dict):
            x = crop_obj.get("x", 0)
            y = crop_obj.get("y", 0)
            w = crop_obj.get("width") if "width" in crop_obj else crop_obj.get("w")
            h = crop_obj.get("height") if "height" in crop_obj else crop_obj.get("h")
            if w is not None and h is not None:
                return float(x), float(y), float(w), float(h)
        elif isinstance(crop_obj, (list, tuple)) and len(crop_obj) >= 4:
            return float(crop_obj[0]), float(crop_obj[1]), float(crop_obj[2]), float(crop_obj[3])
        return None

    rotated_w, rotated_h = (cur_h, cur_w) if rot in (90, 270) else (cur_w, cur_h)

    do_rotate_first = True  # Default
    if crop_rect_raw and rot:
        # Prefer the app's explicit signal when present (matches utils._infer_crop_basis):
        # "pre_rotate" => crop was drawn before rotate (crop first); "after_rotate" => rotate first.
        basis = str(ax.get("crop_rect_basis", "")).strip().lower()
        if basis == "pre_rotate":
            do_rotate_first = False
        elif basis == "after_rotate":
            do_rotate_first = True
        else:
            ref_w, ref_h = _parse_ref(crop_ref_raw, cur_w, cur_h)
            if (int(round(ref_w)), int(round(ref_h))) == (int(round(cur_w)), int(round(cur_h))):
                do_rotate_first = False
            elif (int(round(ref_w)), int(round(ref_h))) == (int(round(rotated_w)), int(round(rotated_h))):
                do_rotate_first = True

    def apply_rotate():
        nonlocal M, cur_w, cur_h
        if rot == 90:
            R = np.array([[0.0, -1.0, cur_h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            cur_w, cur_h = cur_h, cur_w
        elif rot == 180:
            R = np.array([[-1.0, 0.0, cur_w - 1.0], [0.0, -1.0, cur_h - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        elif rot == 270:
            R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, cur_w - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            cur_w, cur_h = cur_h, cur_w
        else:
            R = np.eye(3, dtype=np.float64)
        M = R @ M

    def apply_crop():
        nonlocal M, cur_w, cur_h
        if not crop_enabled or not crop_rect_raw:
            return
        parsed = _parse_crop(crop_rect_raw)
        if not parsed:
            return
        cx, cy, cw, ch = parsed
        ref_w, ref_h = _parse_ref(crop_ref_raw, cur_w, cur_h)
        ref_w = max(1.0, ref_w)
        ref_h = max(1.0, ref_h)

        sx = cur_w / ref_w
        sy = cur_h / ref_h

        x = cx * sx
        y = cy * sy
        w = cw * sx
        h = ch * sy

        C = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]], dtype=np.float64)
        M = C @ M
        cur_w, cur_h = max(1.0, w), max(1.0, h)

    if do_rotate_first:
        apply_rotate()
        apply_crop()
    else:
        apply_crop()
        apply_rotate()

    # 3. Resize
    resize_enabled = bool(ax.get("resize_enabled", True))
    res = ax.get("resize")
    if resize_enabled and isinstance(res, dict) and res:
        new_w = new_h = None
        if "px_w" in res or "px_h" in res:
            tw = res.get("px_w")
            th = res.get("px_h")
            if tw and th:
                new_w, new_h = float(tw), float(th)
            elif tw:
                s = float(tw) / cur_w
                new_w, new_h = float(tw), cur_h * s
            elif th:
                s = float(th) / cur_h
                new_w, new_h = cur_w * s, float(th)
        elif "scale" in res:
            sc = float(res["scale"]) / 100.0
            if sc > 0:
                new_w, new_h = cur_w * sc, cur_h * sc
        elif "width" in res or "height" in res:
            # CanoPie stores resize "width"/"height" as PERCENTAGES (see
            # apply_aux_modifications and image_editor_dialog: pct = value/100.0), NOT
            # absolute pixels. Treating them as pixels produced a wildly wrong scale.
            pw = float(res.get("width", 100.0)) / 100.0
            ph = float(res.get("height", 100.0)) / 100.0
            if pw > 0 and ph > 0:
                new_w, new_h = cur_w * pw, cur_h * ph

        if new_w and new_h and new_w > 0 and new_h > 0:
            sx = new_w / cur_w
            sy = new_h / cur_h
            if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                M = S @ M
                cur_w, cur_h = new_w, new_h

    return M, int(round(cur_w)), int(round(cur_h))


def _raw_image_dims(filepath):
    """
    Return (raw_h, raw_w) for an image on disk, or (None, None).

    BAND-FIRST SAFE. `tif.pages[0].shape[:2]` is WRONG for planar-separate /
    band-first rasters: an AVIRIS-style stack with 7 bands has
    ``series[0].shape == (7, H, W)`` and ``axes == 'SYX'``, so pages[0].shape[:2]
    yields ``(7, H)`` — i.e. a "height" of 7. Downstream that made
    ``out_h`` 7 and scaled every polygon's Y by ``7/ref_h`` (~0.004), collapsing
    exported shapes into a sliver at the top of the image.

    Resolve Y/X through the series axes (same approach as
    ``ProjectTab._image_dims_only``), falling back to the page shape and then cv2.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            with tifffile.TiffFile(filepath) as tif:
                ser = tif.series[0]
                shape = tuple(ser.shape)
                axes = (getattr(ser, "axes", "") or "")
                if 'Y' in axes and 'X' in axes and len(axes) == len(shape):
                    return int(shape[axes.index('Y')]), int(shape[axes.index('X')])
                pshape = tif.pages[0].shape
                if len(pshape) >= 3 and pshape[0] <= 4 and pshape[1] > 4:
                    # Band-first single page, e.g. (C, H, W).
                    return int(pshape[1]), int(pshape[2])
                if len(pshape) >= 2:
                    return int(pshape[0]), int(pshape[1])
        except Exception:
            pass
    try:
        import cv2
        img = cv2.imread(filepath)
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    except Exception:
        pass
    return None, None


def _sanitize_dbf_value(v):
    """
    Coerce a process_polygon stat value into something write_dbf can type (bool/int/
    float/str/None) — CSV rows can carry numpy scalars, dicts (e.g. class-count
    breakdowns), arrays, etc. that DBF has no native representation for.
    """
    import numpy as _np
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, _np.integer):
        return int(v)
    if isinstance(v, _np.floating):
        return float(v)
    if isinstance(v, dict):
        return " ".join(f"{k}:{val}" for k, val in v.items())
    if isinstance(v, (list, tuple, set, _np.ndarray)):
        try:
            seq = list(v)
        except Exception:
            return str(v)
        return f"[{len(seq)} values]" if len(seq) > 20 else " ".join(str(x) for x in seq)
    return str(v)


def json_polygons_to_features(all_polygons, project_folder,
                               get_image_filepath_fn=None, get_gps_fn=None,
                               get_ax_path_fn=None, stats_lookup=None):
    """
    Convert CanoPie's ``all_polygons`` dict to a flat feature list for
    ``write_shapefile()``.

    Parameters
    ----------
    all_polygons : dict
        Structure: all_polygons[group_name][filepath] = polygon_data_dict
        Each polygon_data_dict has: points, name, root, type, coord_space,
        image_ref_size, coordinates.
    project_folder : str
        Project folder path. Used both for resolving relative image paths and, when
        ``get_ax_path_fn`` is not given, for locating the per-image ``.ax`` sidecar
        (CanoPie writes ``.ax`` files into the project folder, NOT next to the image).
    get_image_filepath_fn : callable, optional
        Not used currently; reserved for future image path resolution.
    get_gps_fn : callable, optional
        Not used currently; reserved for future GPS lookup.
    get_ax_path_fn : callable, optional
        ``filepath -> ax_path`` resolver (pass ``ProjectTab._ax_path_for``). This is the
        single source of truth for where the ``.ax`` lives (project folder if saved,
        else a temp cache, else a legacy image-sidecar). Without it the crop/rotate/resize
        modifications cannot be undone and shapes are misplaced for edited images.
    stats_lookup : dict, optional
        ``(group_name, filepath) -> list[row_dict]`` — the SAME per-shape statistics
        CSV export computes (see ``ProjectTab.compute_shapefile_stats``, which calls
        ``process_polygon`` directly, the canonical implementation CSV export itself
        uses). When given:
          - a 'polygon' shape's single row (if any) is merged into its feature's
            properties;
          - a 'point' shape's list of rows (there is one row PER independently
            measured point) each becomes its OWN Point feature, located at that
            row's own "Centroid X"/"Centroid Y" — which is already in the same
            export/.ax-frame pixel space this function maps through M_inv, so no
            separate bounds/mask re-checking is needed;
          - a shape whose key maps to an EMPTY row list (process_polygon found
            nothing exportable — e.g. fully masked, or out of image bounds) is
            SKIPPED entirely, so the shapefile never carries a stats-less feature
            that the CSV wouldn't have a row for either.
        When ``None``, stats aren't attached (a shapefile with just identity
        properties is written — including the point-type fix below, which is not
        gated on stats being requested).

    Returns
    -------
    (features_list, crs_wkt, warnings_list)
    """
    features_list = []
    warnings_list = []
    global_crs = None

    def _resolve_ax_path(image_fp):
        """Find the .ax for this image, mirroring ProjectTab._ax_path_for.
        Order: explicit resolver -> project_folder/<base>.ax -> image-sidecar."""
        base = os.path.splitext(os.path.basename(image_fp))[0] + ".ax"
        candidates = []
        if callable(get_ax_path_fn):
            try:
                p = get_ax_path_fn(image_fp)
                if p:
                    candidates.append(p)
            except Exception:
                pass
        if project_folder:
            candidates.append(os.path.join(project_folder, base))
        candidates.append(os.path.splitext(image_fp)[0] + ".ax")  # legacy sidecar
        for c in candidates:
            try:
                if c and os.path.exists(c):
                    return c
            except Exception:
                continue
        return candidates[0] if candidates else (os.path.splitext(image_fp)[0] + ".ax")

    # Cache transforms per image filepath to avoid re-reading the same raster
    _transform_cache = {}

    try:
        for group_name, by_filepath in all_polygons.items():
            if not isinstance(by_filepath, dict):
                continue

            for filepath, polygon_data in by_filepath.items():
                if not isinstance(polygon_data, dict):
                    continue

                points = polygon_data.get('points', [])
                if not points:
                    continue

                ref_size = polygon_data.get('image_ref_size', {})
                ref_w = float(ref_size.get('w', 0) or ref_size.get('width', 0))
                ref_h = float(ref_size.get('h', 0) or ref_size.get('height', 0))
                if ref_w <= 0 or ref_h <= 0:
                    warnings_list.append(
                        f"Missing image_ref_size for polygon "
                        f"'{group_name}' on '{os.path.basename(filepath)}' — skipped."
                    )
                    continue

                # --- get / cache transform for this image ------------------
                norm_fp = os.path.normpath(filepath)
                if norm_fp not in _transform_cache:
                    transform, crs = None, None

                    # Get RAW dims for ax transform (band-first safe — see
                    # _raw_image_dims; using pages[0].shape[:2] here reported a
                    # height of 7 for 7-band planar stacks and flattened every
                    # exported shape).
                    raw_w = raw_h = None
                    if os.path.isfile(norm_fp):
                        raw_h, raw_w = _raw_image_dims(norm_fp)

                    if os.path.isfile(norm_fp):
                        transform, crs = get_geotiff_transform(norm_fp)

                    georef_method = "geotiff" if transform else None

                    if not transform and os.path.isfile(norm_fp):
                        if raw_w and raw_h:
                            transform, crs = estimate_transform_from_exif(norm_fp, raw_w, raw_h)
                        else:
                            transform, crs = estimate_transform_from_exif(norm_fp, ref_w, ref_h)
                        if transform:
                            georef_method = "exif"

                    if transform and not crs:
                        # A .prj cannot be written without a CRS, and a GIS then has
                        # to guess — which silently mislocates projected (metre) data.
                        # Never let that pass unreported.
                        _epsg = epsg_from_geotiff(norm_fp) if os.path.isfile(norm_fp) else None
                        if _epsg:
                            warnings_list.append(
                                f"'{os.path.basename(filepath)}': CRS EPSG:{_epsg} could not be "
                                "converted to WKT, so no .prj was written — assign EPSG:"
                                f"{_epsg} manually in your GIS (installing 'pyproj' fixes this)."
                            )
                        else:
                            warnings_list.append(
                                f"'{os.path.basename(filepath)}': georeferenced but the CRS is "
                                "undeclared, so no .prj was written — set the layer CRS manually."
                            )

                    if not transform:
                        georef_method = "none"
                        warnings_list.append(
                            f"No georeferencing for "
                            f"'{os.path.basename(filepath)}' — using pixel coords."
                        )
                        
                    # Handle .ax modifications for correct pixel coordinates.
                    # NOTE: CanoPie stores .ax in the PROJECT FOLDER (not next to the image),
                    # so resolve via _resolve_ax_path — otherwise crop/resize/rotate is never
                    # undone and shapes land in the wrong place for edited images.
                    ax_path = _resolve_ax_path(norm_fp)
                    M_inv = np.eye(3, dtype=np.float64)
                    out_w = out_h = None
                    if raw_w and raw_h and os.path.exists(ax_path):
                        try:
                            import json
                            with open(ax_path, 'r', encoding='utf-8') as f:
                                ax_data = json.load(f)
                            M, out_w, out_h = get_ax_transform_matrix(ax_data, raw_w, raw_h)
                            M_inv = np.linalg.inv(M)
                        except Exception as e:
                            logging.warning(f"Failed to apply .ax transform for {norm_fp}: {e}")

                    _transform_cache[norm_fp] = (transform, crs, georef_method, M_inv, out_w, out_h)
                else:
                    transform, crs, georef_method, M_inv, out_w, out_h = _transform_cache[norm_fp]

                if crs and not global_crs:
                    global_crs = crs

                extraction_type = str(polygon_data.get('type', 'polygon')).lower()

                # Statistics are STRICTLY ADDITIVE. Geometry export must never depend on
                # them succeeding — an earlier version skipped any shape whose stats
                # lookup came back empty, which silently produced an EMPTY shapefile
                # whenever stats computation failed or returned nothing.
                rows = list((stats_lookup or {}).get((group_name, filepath)) or [])

                root_val = polygon_data.get('root', '')
                coords = polygon_data.get('coordinates', {}) or {}

                def _base_props(name_val, type_val):
                    p = OrderedDict()
                    p['group'] = str(group_name)[:254]
                    p['name'] = str(name_val)[:254]
                    p['root'] = str(root_val)[:50]
                    p['filename'] = str(os.path.basename(filepath))[:254]
                    p['type'] = str(type_val)[:50]
                    p['georef'] = str(georef_method)[:20]
                    p['gps_lat'] = float(coords.get('latitude') or 0.0)
                    p['gps_lon'] = float(coords.get('longitude') or 0.0)
                    return p

                def _stored_to_export(p):
                    """
                    Stored point -> .ax/export pixel frame. This is the ONE coordinate
                    derivation used for every geometry in this function (points and
                    polygon rings alike).

                    Do NOT substitute a CSV row's "Centroid X/Y" here: those are produced
                    by a different code path (`_points_to_export_frame` against the actual
                    export image shape) and are floored to whole pixels, whereas out_w/out_h
                    come from this module's own independent `get_ax_transform_matrix`.
                    Mixing the two put points in the wrong place.
                    """
                    x, y = float(p[0]), float(p[1])
                    if ref_w > 0 and ref_h > 0 and out_w and out_h and \
                            (abs(ref_w - out_w) > 1.0 or abs(ref_h - out_h) > 1.0):
                        x *= (out_w / ref_w)
                        y *= (out_h / ref_h)
                    return x, y

                # ---------------------------------------------------------- #
                # POINT shapes: one stored polygon_data entry can hold MANY
                # independently-measured points (a "point set"). Each is its own
                # shape with its own location and — where matchable — its own CSV
                # row. The original code strung ALL of a point set's coordinates
                # into a single closed "polygon" ring, fabricating a shape with no
                # relation to what was actually drawn. Geometry always comes from
                # the stored points, exactly as the polygon path does.
                # ---------------------------------------------------------- #
                if extraction_type == 'point':
                    export_pts = [_stored_to_export(p) for p in points]

                    # Attach stats per point where we can match confidently.
                    # process_polygon emits rows in order but skips masked /
                    # out-of-bounds points, so the counts can legitimately differ.
                    row_for_idx = {}
                    if rows:
                        if len(rows) == len(export_pts):
                            row_for_idx = dict(enumerate(rows))
                        else:
                            used = set()
                            for r in rows:
                                rx, ry = r.get('Centroid X'), r.get('Centroid Y')
                                if rx is None or ry is None:
                                    continue
                                best_i, best_d = None, None
                                for i, (ex, ey) in enumerate(export_pts):
                                    if i in used:
                                        continue
                                    d = (ex - float(rx)) ** 2 + (ey - float(ry)) ** 2
                                    if best_d is None or d < best_d:
                                        best_i, best_d = i, d
                                # Centroids are floored to whole pixels, so allow ~2px.
                                if best_i is not None and best_d is not None and best_d <= 4.0:
                                    row_for_idx[best_i] = r
                                    used.add(best_i)
                            if len(row_for_idx) < len(rows):
                                warnings_list.append(
                                    f"Point set '{polygon_data.get('name', group_name)}' on "
                                    f"'{os.path.basename(filepath)}': matched "
                                    f"{len(row_for_idx)}/{len(rows)} statistic rows to points."
                                )

                    for i, (ex, ey) in enumerate(export_pts):
                        pt_mod = np.array([ex, ey, 1.0], dtype=np.float64)
                        pt_raw = M_inv @ pt_mod
                        px, py = float(pt_raw[0]), float(pt_raw[1])
                        geo_pt = pixel_to_geo([(px, py)], transform)[0] if transform else (px, py)

                        props = _base_props(polygon_data.get('name', group_name), 'point')
                        r = row_for_idx.get(i)
                        if r:
                            for k, v in r.items():
                                if k in ('Centroid X', 'Centroid Y') or k in props:
                                    continue
                                props[k] = _sanitize_dbf_value(v)
                        features_list.append({'geometry': [geo_pt], 'properties': props})
                    continue

                # ---------------------------------------------------------- #
                # POLYGON shapes: unchanged ring geometry, plus its stats row
                # (normally exactly one) merged into the properties.
                # ---------------------------------------------------------- #
                pixel_pts = []
                for p in points:
                    x, y = _stored_to_export(p)
                    pt_mod = np.array([x, y, 1.0], dtype=np.float64)
                    pt_raw = M_inv @ pt_mod
                    pixel_pts.append((float(pt_raw[0]), float(pt_raw[1])))

                if transform:
                    geo_points = pixel_to_geo(pixel_pts, transform)
                else:
                    # Keep raw pixel coordinates
                    geo_points = pixel_pts

                # Ensure polygon ring is closed
                if geo_points and geo_points[0] != geo_points[-1]:
                    geo_points.append(geo_points[0])

                props = _base_props(polygon_data.get('name', group_name), extraction_type)
                if rows:
                    if len(rows) > 1:
                        warnings_list.append(
                            f"Polygon '{props['name']}' produced {len(rows)} stat rows; "
                            "using the first."
                        )
                    for k, v in rows[0].items():
                        if k in props:
                            continue
                        props[k] = _sanitize_dbf_value(v)

                features_list.append({
                    'geometry': geo_points,
                    'properties': props,
                })

    except Exception as e:
        logging.warning(f"Error converting polygons to features: {e}")
        warnings_list.append(f"Conversion error: {e}")

    return features_list, global_crs, warnings_list

