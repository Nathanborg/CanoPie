"""
Find data in an ArcGIS Hub / open-data portal (the STRI data portal, or any
other) and bring it into CanoPie: rasters by streaming where possible, by
download (asked first) otherwise, and vector layers as polygons.

No Qt and no GDAL here. Searching uses the public ArcGIS REST search; vector
layers become ESRI shapefiles (pyshp, Shapely) that CanoPie's own shapefile
import then places on the images:

* a feature layer (Feature Service): queried as GeoJSON in WGS 84, page by page;
* a zipped shapefile item: downloaded and unpacked;
* a GeoPackage (.gpkg): read with Python's sqlite3 -- GeoPackage geometries are
  WKB behind a small header, decoded by Shapely;
* GeoJSON.

Each item is classified:

``stream``    a tiled map / image service, or a GeoTIFF item whose download
              address honours HTTP Range (remote_raster / remote_tiles open it);
``download``  a raster that only downloads whole (an image collection zip);
``vector``    something whose shapes CanoPie can import.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import zipfile

from . import remote_raster as rm

__all__ = ["DEFAULT_PORTAL", "org_for_site", "search", "item_link", "data_url", "download",
           "vector_to_shapefiles", "geojson_to_shapefiles", "gpkg_to_shapefiles"]

DEFAULT_PORTAL = "https://stridata-si.opendata.arcgis.com"
#: ArcGIS endpoints as module constants, so the QC suite can use a local stand-in.
HUB_DOMAIN_URL = "https://hub.arcgis.com/api/v3/domains/{}"
SEARCH_URL = "https://www.arcgis.com/sharing/rest/search"
ITEM_DATA_URL = "https://www.arcgis.com/sharing/rest/content/items/{}/data"
ITEM_PAGE_URL = "https://www.arcgis.com/home/item.html?id={}"

_RASTER_TYPES = ("Map Service", "Image Service", "Image", "Image Collection")
_VECTOR_TYPES = ("Feature Service", "Shapefile", "GeoPackage", "GeoJson")


def _get_json(url, what, params=None):
    try:
        resp = rm._session().get(url, params=params, timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT))
    except Exception as exc:
        raise rm.RemoteError("{}: request failed: {}".format(what, type(exc).__name__)) from None
    if resp.status_code != 200:
        raise rm.RemoteError("{}: HTTP {}".format(what, resp.status_code))
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise rm.RemoteError("{}: {}".format(what, data["error"].get("message") or data["error"]))
    return data


# --------------------------------------------------------------------------
# Portal and search
# --------------------------------------------------------------------------

def org_for_site(site):
    """``(org_id, title)`` of a Hub site URL (e.g. the STRI data portal).
    An ArcGIS organisation id given directly is returned as is."""
    site = str(site or DEFAULT_PORTAL).strip()
    if re.fullmatch(r"[A-Za-z0-9]{10,20}", site):
        return site, site
    host = rm.urlsplit(site if "://" in site else "https://" + site).netloc
    d = _get_json(HUB_DOMAIN_URL.format(host), "data portal " + host)
    org = d.get("orgId")
    if not org:
        raise rm.RemoteError("{} is not an ArcGIS Hub site".format(host))
    return org, d.get("siteTitle") or host


def _classify(it):
    typ = it.get("type") or ""
    name = str(it.get("name") or "")
    url = it.get("url") or ""
    low = name.lower()
    if typ in ("Map Service", "Image Service") and re.search(r"/(MapServer|ImageServer)/?$", url):
        return "stream"
    if typ == "Image" and low.endswith((".tif", ".tiff")):
        return "stream"                       # the item's download address honours Range
    if typ == "Image Collection" or (typ == "Image" and low.endswith(".zip")):
        return "download"
    if typ == "Feature Service" and "/FeatureServer" in url:
        return "vector"
    if typ in ("Shapefile", "GeoPackage", "GeoJson"):
        return "vector"
    return None                               # pictures, logos, web maps ...


def search(org_id, text="", kinds=("stream", "download", "vector"), num=100, start=1):
    """Items of the organisation CanoPie can use, as dicts: id, title, type,
    kind, url, name, snippet, owner. ``text`` narrows the search."""
    types = [t for t in (_RASTER_TYPES if {"stream", "download"} & set(kinds) else ())] + \
            [t for t in (_VECTOR_TYPES if "vector" in kinds else ())]
    q = "orgid:{} AND ({})".format(org_id, " OR ".join('type:"{}"'.format(t) for t in types))
    if text and text.strip():
        q = "({}) AND {}".format(text.strip(), q)
    out, start = [], int(start)
    while True:
        d = _get_json(SEARCH_URL, "portal search", params={
            "q": q, "num": min(100, int(num)), "start": start, "sortField": "title",
            "sortOrder": "asc", "f": "json"})
        for it in d.get("results") or []:
            kind = _classify(it)
            if kind in kinds:
                out.append({"id": it["id"], "title": (it.get("title") or "").strip(), "type": it.get("type"),
                            "kind": kind, "url": it.get("url") or "", "name": it.get("name") or "",
                            "snippet": it.get("snippet") or "", "owner": it.get("owner") or ""})
        nxt = d.get("nextStart", -1)
        if nxt in (-1, None, 0) or len(out) >= num:
            break
        start = int(nxt)
    return out


def data_url(item):
    return ITEM_DATA_URL.format(item["id"])


def item_link(item):
    """What to paste into Add Remote Images for a ``stream`` item."""
    if item["type"] == "Image":
        return data_url(item)                 # a GeoTIFF: stream it by byte ranges
    return item.get("url") or ITEM_PAGE_URL.format(item["id"])


def download_size(item):
    """Bytes of an item's download, or None if the server does not say."""
    try:
        r = rm._session().get(data_url(item), headers={"Range": "bytes=0-0"}, stream=True,
                              timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT))
        m = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range") or "")
        size = int(m.group(1)) if m else (int(r.headers["Content-Length"])
                                           if r.status_code == 200 and r.headers.get("Content-Length") else None)
        r.close()
        return size
    except Exception:
        return None


def download(item, folder, progress=None, cancelled=None):
    """Download an item's file into ``folder`` (zips unpacked next to it).
    ``progress(done, total)``; ``cancelled()`` stops it. Returns the files."""
    os.makedirs(folder, exist_ok=True)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", item.get("name") or item["id"]) or item["id"]
    dest = os.path.join(folder, name)
    tmp = dest + ".part"
    with rm._session().get(data_url(item), stream=True, timeout=(rm.CONNECT_TIMEOUT, rm.READ_TIMEOUT)) as r:
        if r.status_code != 200:
            raise rm.RemoteError("{}: download failed (HTTP {})".format(item.get("title"), r.status_code))
        total = int(r.headers.get("Content-Length") or 0) or None
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if cancelled is not None and cancelled():
                    fh.close()
                    os.remove(tmp)
                    raise rm.RemoteError("download cancelled")
                fh.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    os.replace(tmp, dest)
    files = [dest]
    if zipfile.is_zipfile(dest):
        sub = os.path.splitext(dest)[0]
        with zipfile.ZipFile(dest) as z:
            for m in z.infolist():
                target = os.path.realpath(os.path.join(sub, m.filename))
                if not target.startswith(os.path.realpath(sub) + os.sep):
                    continue                  # a path that escapes the folder
                z.extract(m, sub)
        files = [os.path.join(dp, f) for dp, _d, fs in os.walk(sub) for f in fs]
    return files


# --------------------------------------------------------------------------
# Vectors -> shapefiles
# --------------------------------------------------------------------------

def _field_names(keys):
    out, used = {}, set()
    for k in keys:
        base = re.sub(r"[^A-Za-z0-9_]", "_", str(k))[:10] or "field"
        name, n = base, 1
        while name.upper() in used:
            suffix = str(n)
            name = base[:10 - len(suffix)] + suffix
            n += 1
        used.add(name.upper())
        out[k] = name
    return out


def _write_shapefile(path, shapes, props, prj_wkt):
    """Write (shapely geometry, properties) of one geometry family. Returns path."""
    import shapefile
    from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, MultiLineString
    from shapely.geometry.polygon import orient
    first = shapes[0]
    if isinstance(first, (Polygon, MultiPolygon)):
        kind = shapefile.POLYGON
    elif isinstance(first, (Point, MultiPoint)):
        kind = shapefile.POINT if all(isinstance(s, Point) for s in shapes) else shapefile.MULTIPOINT
    else:
        kind = shapefile.POLYLINE
    keys = []
    for p in props:
        for k in p:
            if k not in keys:
                keys.append(k)
    names = _field_names(keys)
    w = shapefile.Writer(path, shapeType=kind)
    numeric = {}
    for k in keys:
        vals = [p.get(k) for p in props if p.get(k) is not None]
        is_num = bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        numeric[k] = is_num
        if is_num:
            w.field(names[k], "N", 19, 0 if all(float(v).is_integer() for v in vals) else 6)
        else:
            size = max([len(str(v)) for v in vals] + [1])
            w.field(names[k], "C", min(254, size))
    if not keys:
        w.field("ID", "N", 10, 0)
    for i, (g, p) in enumerate(zip(shapes, props)):
        if kind == shapefile.POLYGON:
            rings = []
            for poly in (g.geoms if isinstance(g, MultiPolygon) else [g]):
                poly = orient(poly, sign=-1.0)            # shapefile: outer rings clockwise
                rings.append([list(c) for c in poly.exterior.coords])
                rings.extend([list(c) for c in r.coords] for r in poly.interiors)
            w.poly(rings)
        elif kind == shapefile.POINT:
            w.point(g.x, g.y)
        elif kind == shapefile.MULTIPOINT:
            w.multipoint([[pt.x, pt.y] for pt in (g.geoms if hasattr(g, "geoms") else [g])])
        else:
            parts = g.geoms if isinstance(g, MultiLineString) else [g]
            w.line([[list(c) for c in part.coords] for part in parts])
        if keys:
            w.record(*[(p.get(k) if numeric[k] else ("" if p.get(k) is None else str(p.get(k))[:254]))
                       for k in keys])
        else:
            w.record(i + 1)
    w.close()
    if prj_wkt:
        with open(os.path.splitext(path)[0] + ".prj", "w", encoding="utf-8") as fh:
            fh.write(prj_wkt)
    return os.path.splitext(path)[0] + ".shp"


def _family(g):
    t = g.geom_type
    return "poly" if "Polygon" in t else ("point" if "Point" in t else "line")


def _write_families(base, geoms, props, prj_wkt):
    """One shapefile per geometry family (a shapefile holds a single type)."""
    fams = {}
    for g, p in zip(geoms, props):
        if g is None or g.is_empty:
            continue
        fams.setdefault(_family(g), ([], []))
        fams[_family(g)][0].append(g)
        fams[_family(g)][1].append(p)
    out = []
    for fam, (gs, ps) in fams.items():
        path = base if len(fams) == 1 else "{}_{}".format(base, fam)
        out.append(_write_shapefile(path, gs, ps, prj_wkt))
    return out


def geojson_to_shapefiles(geojson, base, epsg=4326):
    """GeoJSON (dict, or a .geojson / .json path) -> shapefiles at ``base``."""
    from shapely.geometry import shape
    from .shapefile_io import wkt_for_epsg
    if isinstance(geojson, (str, os.PathLike)):
        with open(geojson, encoding="utf-8") as fh:
            geojson = json.load(fh)
    feats = geojson.get("features") if geojson.get("type") == "FeatureCollection" else [geojson]
    geoms, props = [], []
    for f in feats or []:
        if not f or not f.get("geometry"):
            continue
        geoms.append(shape(f["geometry"]))
        props.append(dict(f.get("properties") or {}))
    if not geoms:
        raise rm.RemoteError("no shapes in that GeoJSON")
    return _write_families(base, geoms, props, wkt_for_epsg(epsg))


def _gpkg_wkb(blob):
    """The WKB inside a GeoPackage geometry blob (None for an empty geometry)."""
    if not blob or bytes(blob[:2]) != b"GP":
        return bytes(blob) if blob else None
    flags = blob[3]
    if flags & 0x10:
        return None
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get((flags >> 1) & 0x07, 0)
    return bytes(blob[8 + env:])


def gpkg_to_shapefiles(path, folder):
    """Every feature table of a GeoPackage -> shapefiles in ``folder``."""
    from shapely import wkb
    os.makedirs(folder, exist_ok=True)
    out = []
    con = sqlite3.connect("file:{}?mode=ro".format(os.path.abspath(path)), uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")]
        for t in tables:
            col, srs = con.execute(
                "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?", (t,)).fetchone()
            row = con.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = ?", (srs,)).fetchone()
            prj = row[0] if row and row[0] and row[0] != "undefined" else None
            cur = con.execute('SELECT * FROM "{}"'.format(t.replace('"', '""')))
            cols = [c[0] for c in cur.description]
            gi = cols.index(col)
            geoms, props = [], []
            for rec in cur:
                w = _gpkg_wkb(rec[gi])
                if w is None:
                    continue
                geoms.append(wkb.loads(w))
                props.append({c: v for c, v in zip(cols, rec)
                              if c != col and not isinstance(v, (bytes, memoryview))})
            if geoms:
                safe = re.sub(r"[^\w-]+", "_", t)
                out += _write_families(os.path.join(folder, safe), geoms, props, prj)
    finally:
        con.close()
    if not out:
        raise rm.RemoteError("{}: no feature tables with shapes".format(os.path.basename(path)))
    return out


def _feature_layers(url):
    """[(layer_url, name)] for a FeatureServer URL (or one layer's URL)."""
    url = url.rstrip("/")
    if re.search(r"/FeatureServer/\d+$", url):
        meta = _get_json(url + "?f=json", "feature layer")
        return [(url, meta.get("name") or "layer")]
    meta = _get_json(url + "?f=json", "feature service")
    return [("{}/{}".format(url, l["id"]), l.get("name") or str(l["id"]))
            for l in meta.get("layers") or []]


def feature_service_to_shapefiles(url, folder, progress=None):
    """Every layer of a feature service -> shapefiles in ``folder`` (WGS 84)."""
    os.makedirs(folder, exist_ok=True)
    out = []
    for layer_url, name in _feature_layers(url):
        feats, offset = [], 0
        while True:
            page = _get_json(layer_url + "/query", "feature layer " + name, params={
                "where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                "resultOffset": offset, "resultRecordCount": 2000})
            got = page.get("features") or []
            feats += got
            if progress is not None:
                progress(len(feats), None)
            if not got or not (page.get("exceededTransferLimit")
                               or (page.get("properties") or {}).get("exceededTransferLimit")):
                break
            offset += len(got)
        if feats:
            safe = re.sub(r"[^\w-]+", "_", name)
            out += geojson_to_shapefiles({"type": "FeatureCollection", "features": feats},
                                         os.path.join(folder, safe), epsg=4326)
    if not out:
        raise rm.RemoteError("no features in that service")
    return out


def vector_to_shapefiles(source, folder, progress=None, cancelled=None):
    """Shapefiles for a portal vector item (dict from :func:`search`) or a
    local .shp / .gpkg / .geojson / .json file."""
    if isinstance(source, dict):
        if source["type"] == "Feature Service":
            return feature_service_to_shapefiles(source["url"], folder, progress)
        files = download(source, folder, progress, cancelled)
        shps = [f for f in files if f.lower().endswith(".shp")]
        if shps:
            return shps
        out = []
        for f in files:
            out += vector_to_shapefiles(f, folder) if f.lower().endswith(
                (".gpkg", ".geojson", ".json")) else []
        if not out:
            raise rm.RemoteError("{}: nothing CanoPie can import inside".format(source.get("title")))
        return out
    path = os.fspath(source)
    low = path.lower()
    if low.endswith(".shp"):
        return [path]
    if low.endswith(".gpkg"):
        return gpkg_to_shapefiles(path, folder)
    if low.endswith((".geojson", ".json")):
        base = os.path.join(folder, re.sub(r"[^\w-]+", "_", os.path.splitext(os.path.basename(path))[0]))
        os.makedirs(folder, exist_ok=True)
        return geojson_to_shapefiles(path, base)
    raise rm.RemoteError("{}: not a shapefile, GeoPackage or GeoJSON".format(os.path.basename(path)))
