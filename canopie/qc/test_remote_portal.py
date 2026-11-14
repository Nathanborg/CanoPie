"""
Data portals (ArcGIS Hub / open data, e.g. the STRI data portal):
canopie.remote_portal -- searching, classifying items, downloading, and turning
vector layers (feature services, GeoPackages, GeoJSON) into shapefiles that
CanoPie's shapefile import reads. No GDAL; everything runs against local
stand-ins.
"""
import json
import os
import re
import sqlite3
import struct
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from .. import remote_portal as rp

pytestmark = [pytest.mark.io]

shapefile = pytest.importorskip("shapefile")


# --------------------------------------------------------------------------
# A portal stand-in: Hub domain lookup, search, item data, a feature service
# --------------------------------------------------------------------------

ORG = "OrgId1234567890"
FEATURES = [{"type": "Feature", "properties": {"plot": "P{}".format(i), "area_m2": 25.0 * i},
             "geometry": {"type": "Polygon", "coordinates": [[[-79.85 + i * 1e-4, 9.15], [-79.8499 + i * 1e-4, 9.15],
                                                              [-79.8499 + i * 1e-4, 9.1501], [-79.85 + i * 1e-4, 9.15]]]}}
            for i in range(5)]
ITEMS = [
    {"id": "a" * 32, "title": "BCI 50ha ortho 2020", "type": "Map Service", "name": "o",
     "url": "https://tiles.arcgis.com/tiles/x/arcgis/rest/services/BCI/MapServer"},
    {"id": "b" * 32, "title": "BCNM DTM", "type": "Image Service", "name": "d",
     "url": "https://tiledimageservices2.arcgis.com/x/arcgis/rest/services/DTM/ImageServer"},
    {"id": "c" * 32, "title": "Canal basin DEM", "type": "Image", "name": "dem30.tif", "url": None},
    {"id": "d" * 32, "title": "Trail map", "type": "Image", "name": "trails.jpg", "url": None},
    {"id": "e" * 32, "title": "Panama DEM 30m", "type": "Image Collection", "name": "dem.zip", "url": None},
    {"id": "f" * 32, "title": "50ha plot polygons", "type": "Feature Service", "name": "p", "url": None},
    {"id": "1" * 32, "title": "Plots shapefile", "type": "Shapefile", "name": "plots.zip", "url": None},
]


class _Portal(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.server
        parts = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if parts.path.startswith("/domains/"):
            return self._send(200, {"orgId": ORG, "siteTitle": "Test portal"})
        if parts.path == "/search":
            assert "orgid:" + ORG in q["q"]
            items = [dict(it) for it in ITEMS]
            for it in items:
                if it["type"] == "Feature Service":
                    it["url"] = srv.base + "fs/FeatureServer"
            start = int(q.get("start", 1)) - 1
            page = items[start:start + 4]                     # two pages
            nxt = start + 5 if start + 4 < len(items) else -1
            return self._send(200, {"results": page, "nextStart": nxt, "total": len(items)})
        m = re.fullmatch(r"/items/(\w+)/data", parts.path)
        if m:
            return self._send(200, srv.files[m.group(1)], "application/octet-stream")
        if parts.path == "/fs/FeatureServer":
            return self._send(200, {"layers": [{"id": 0, "name": "50ha plot polygons"}]})
        if parts.path == "/fs/FeatureServer/0/query":
            off, n = int(q.get("resultOffset", 0)), int(q.get("resultRecordCount", 2000))
            n = min(n, 2)                                      # the service pages by 2
            page = FEATURES[off:off + n]
            return self._send(200, {"type": "FeatureCollection", "features": page,
                                    "exceededTransferLimit": off + n < len(FEATURES)})
        return self._send(404, b"", "text/plain")


@pytest.fixture(scope="module")
def portal():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Portal)
    srv.daemon_threads = True
    srv.base = "http://127.0.0.1:{}/".format(srv.server_address[1])
    srv.files = {}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def endpoints(portal, monkeypatch):
    monkeypatch.setattr(rp, "HUB_DOMAIN_URL", portal.base + "domains/{}")
    monkeypatch.setattr(rp, "SEARCH_URL", portal.base + "search")
    monkeypatch.setattr(rp, "ITEM_DATA_URL", portal.base + "items/{}/data")
    return portal


def test_a_hub_site_is_searched_and_its_items_sorted(endpoints):
    org, title = rp.org_for_site("https://test-si.opendata.arcgis.com")
    assert (org, title) == (ORG, "Test portal")
    items = rp.search(org, num=1000)
    kinds = {it["title"]: it["kind"] for it in items}
    assert kinds == {"BCI 50ha ortho 2020": "stream", "BCNM DTM": "stream", "Canal basin DEM": "stream",
                     "Panama DEM 30m": "download", "50ha plot polygons": "vector",
                     "Plots shapefile": "vector"}, "a trail-map JPG is not a raster layer"
    by = {it["title"]: it for it in items}
    assert rp.item_link(by["BCI 50ha ortho 2020"]).endswith("/MapServer")
    assert rp.item_link(by["Canal basin DEM"]) == endpoints.base + "items/{}/data".format("c" * 32), \
        "a GeoTIFF item streams from its download address"
    assert [it["title"] for it in rp.search(org, kinds=("vector",))] == ["50ha plot polygons", "Plots shapefile"]


def test_a_feature_service_becomes_a_shapefile_across_pages(endpoints, tmp_path):
    items = {it["title"]: it for it in rp.search(ORG)}
    shps = rp.vector_to_shapefiles(items["50ha plot polygons"], str(tmp_path))
    assert len(shps) == 1
    r = shapefile.Reader(shps[0])
    assert r.shapeType == shapefile.POLYGON and len(r) == len(FEATURES), "every page was fetched"
    assert [rec["plot"] for rec in r.records()] == ["P{}".format(i) for i in range(5)]
    assert [rec["area_m2"] for rec in r.records()] == [25.0 * i for i in range(5)]
    assert r.shape(2).points[0] == pytest.approx([-79.85 + 2e-4, 9.15])
    prj = open(shps[0][:-4] + ".prj").read()
    assert "WGS" in prj and "84" in prj


def test_a_zipped_shapefile_item_is_downloaded_and_unpacked_safely(endpoints, tmp_path):
    import io
    src = tmp_path / "src"
    src.mkdir()
    w = shapefile.Writer(str(src / "plots"), shapeType=shapefile.POINT)
    w.field("name", "C")
    w.point(1.0, 2.0)
    w.record("a")
    w.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for ext in ("shp", "shx", "dbf"):
            z.write(str(src / "plots.{}".format(ext)), "plots.{}".format(ext))
        z.writestr("../escape.txt", "outside")                 # must not land outside
    endpoints.files["1" * 32] = buf.getvalue()
    items = {it["title"]: it for it in rp.search(ORG)}
    out = tmp_path / "out"
    shps = rp.vector_to_shapefiles(items["Plots shapefile"], str(out))
    assert len(shps) == 1 and shapefile.Reader(shps[0]).record(0)["name"] == "a"
    assert not (out / "escape.txt").exists() and not (tmp_path / "escape.txt").exists()


def _gp_blob(wkb, srs_id=4326):
    """A GeoPackage geometry blob: magic, version, flags (little endian, no
    envelope), srs id, then WKB."""
    return b"GP" + bytes([0, 0b00000001]) + struct.pack("<i", srs_id) + wkb


def test_a_geopackage_is_read_without_gdal(tmp_path):
    from shapely.geometry import Polygon, Point
    path = str(tmp_path / "plots.gpkg")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY, organization TEXT,
            organization_coordsys_id INTEGER, definition TEXT, description TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT, geometry_type_name TEXT,
            srs_id INTEGER, z INTEGER, m INTEGER);
        CREATE TABLE crowns (fid INTEGER PRIMARY KEY, geom BLOB, species TEXT, dbh REAL);
        CREATE TABLE trees (fid INTEGER PRIMARY KEY, geom BLOB, tag INTEGER);
    """)
    con.execute("INSERT INTO gpkg_spatial_ref_sys VALUES ('UTM 17N', 32617, 'EPSG', 32617, ?, '')",
                ('PROJCS["WGS 84 / UTM zone 17N"]',))
    con.execute("INSERT INTO gpkg_contents VALUES ('crowns', 'features', 'crowns')")
    con.execute("INSERT INTO gpkg_contents VALUES ('trees', 'features', 'trees')")
    con.execute("INSERT INTO gpkg_geometry_columns VALUES ('crowns', 'geom', 'POLYGON', 32617, 0, 0)")
    con.execute("INSERT INTO gpkg_geometry_columns VALUES ('trees', 'geom', 'POINT', 32617, 0, 0)")
    holed = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]])
    con.execute("INSERT INTO crowns VALUES (1, ?, 'Ficus', 81.5)", (_gp_blob(holed.wkb, 32617),))
    con.execute("INSERT INTO crowns VALUES (2, ?, 'Ceiba', 120.0)",
                (_gp_blob(Polygon([(20, 0), (25, 0), (25, 5)]).wkb, 32617),))
    con.execute("INSERT INTO trees VALUES (1, ?, 7)", (_gp_blob(Point(3, 4).wkb, 32617),))
    con.commit()
    con.close()

    shps = sorted(rp.vector_to_shapefiles(path, str(tmp_path / "out")))
    assert [os.path.basename(s) for s in shps] == ["crowns.shp", "trees.shp"]
    crowns = shapefile.Reader(shps[0])
    assert crowns.shapeType == shapefile.POLYGON and len(crowns) == 2
    assert [(r["species"], r["dbh"]) for r in crowns.records()] == [("Ficus", 81.5), ("Ceiba", 120.0)]
    assert len(crowns.shape(0).parts) == 2, "the hole was kept as a second ring"
    from shapely.geometry import shape
    assert shape(crowns.shape(0).__geo_interface__).area == pytest.approx(holed.area)
    assert "UTM zone 17N" in open(shps[0][:-4] + ".prj").read()
    trees = shapefile.Reader(shps[1])
    assert trees.shapeType == shapefile.POINT and list(trees.shape(0).points[0]) == [3.0, 4.0]


def test_geojson_with_mixed_geometry_splits_into_one_shapefile_per_type(tmp_path):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"n": 1},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]],
                                                               [[[2, 2], [3, 2], [3, 3], [2, 2]]]]}},
        {"type": "Feature", "properties": {"n": 2}, "geometry": {"type": "Point", "coordinates": [5, 5]}}]}
    path = tmp_path / "mixed.geojson"
    path.write_text(json.dumps(gj))
    shps = sorted(rp.vector_to_shapefiles(str(path), str(tmp_path / "out")))
    kinds = sorted(shapefile.Reader(s).shapeType for s in shps)
    assert kinds == sorted([shapefile.POLYGON, shapefile.POINT])
    poly = [s for s in shps if shapefile.Reader(s).shapeType == shapefile.POLYGON][0]
    assert len(shapefile.Reader(poly).shape(0).parts) == 2


# --------------------------------------------------------------------------
# Where the user meets it: Import Shapefile asks, Add Remote Images lists vectors
# --------------------------------------------------------------------------

def test_import_shapefile_asks_file_or_portal(synthetic_project, monkeypatch):
    from PyQt5 import QtWidgets
    pm = synthetic_project.polygon_manager
    calls = []
    monkeypatch.setattr(pm, "on_import_from_portal", lambda: calls.append("portal"))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (calls.append("file") or [], "")))
    for answer, want in (("portal", ["portal"]), ("file", ["file"]), (None, [])):
        calls.clear()
        monkeypatch.setattr(pm, "_ask_import_source", lambda a=answer: a)
        pm.on_import_shapefile()
        assert calls == want, (answer, calls)
    assert not hasattr(pm, "import_portal_button"), "the extra button is gone"


def test_add_remote_images_lists_vectors_and_accepts_them_alone(qapp, monkeypatch):
    from PyQt5 import QtCore
    from .. import portal_finder_dialog as pfd
    from .. import remote_images_dialog as rid
    monkeypatch.setattr(QtCore.QTimer, "singleShot", staticmethod(lambda *a, **k: None))   # no search
    f = pfd.PortalFinderDialog(None, mode="rasters")
    try:
        kinds = [f.kind_combo.itemData(i) for i in range(f.kind_combo.count())]
        assert ("vector",) in kinds and ("stream", "download", "vector") in kinds
    finally:
        f.deleteLater()
    monkeypatch.setattr(rid, "_SETTINGS", ("CanoPie-QC", "portal-vectors-test"))
    d = rid.RemoteImagesDialog(None)
    try:
        d.vector_items = [{"id": "f" * 32, "title": "50ha plot polygons", "type": "Feature Service",
                           "kind": "vector", "url": "x", "name": "p", "snippet": "", "owner": ""}]
        d._update_extra_label()
        assert "imported as polygons" in d.local_label.text()
        d._check_and_open()
        assert d.result() == d.Accepted and d.result_paths == []
    finally:
        QtCore.QSettings("CanoPie-QC", "portal-vectors-test").clear()
        d.deleteLater()
