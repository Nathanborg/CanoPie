"""
Tiled ArcGIS services (MapServer / ImageServer, and their ArcGIS Hub pages)
streamed as remote images: canopie.remote_tiles.

A local stand-in serves the service metadata, an ArcGIS item and PNG tiles whose
pixels are a function of their global position, so every value CanoPie reads
can be checked exactly. Tiles exist down to zoom 17 only (the service lists 20
levels), as the real STRI orthophoto lists 24 levels and has tiles to 21.
"""
import json
import math
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from .. import raster_reader as rr
from .. import remote_raster as rm
from .. import remote_tiles as rt

pytestmark = [pytest.mark.io]

R0 = 156543.03392804097                # Web Mercator resolution at zoom 0 (256 px tiles)
ORIGIN = (-20037508.342787, 20037508.342787)
ZDATA = 17                             # deepest zoom with tiles
EXTENT = (-8889642.0, 1022828.0, -8889642.0 + 2400.0, 1022828.0 + 1550.0)
ITEM = "554f6a11e80944cabf7fc29e2d08408a"


def pattern(gx, gy):
    """RGBA of global pixels (gx, gy) at any zoom."""
    gx = np.asarray(gx, np.int64)
    gy = np.asarray(gy, np.int64)
    return np.stack([gx % 251, gy % 241, (3 * gx + gy) % 256,
                     np.full(np.broadcast(gx, gy).shape, 255)], axis=-1).astype(np.uint8)


DTM_Z = 4                              # 1 m at zoom 4
DTM_ORIGIN = (620914.0, 1020672.0)     # UTM 17N, the tile grid's corner
DTM_MISSING = (1, 2)                   # a tile the service does not have


def dtm_values(z, gx, gy):
    """Elevation-like float32 at global pixel (gx, gy) of zoom z, with an
    ArcGIS NoData sentinel in one corner."""
    v = (40.0 + 0.5 * gx + 0.25 * gy + 0.125 * (DTM_Z - z)).astype(np.float32)
    v[(gx < 20) & (gy < 20)] = -3.4028234663852886e38
    return v


class _Arcgis(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        import imagecodecs
        srv = self.server
        parts = urlsplit(self.path)
        with srv.lock:
            srv.paths.append(parts.path)
        if parts.path.startswith("/sharing/rest/content/items/"):
            body = json.dumps({"title": "Test orthophoto 2021", "type": "Map Service",
                               "url": srv.base + "arcgis/rest/services/Test/MapServer"}).encode()
            return self._send(200, body, "application/json")
        m = re.fullmatch(r"/arcgis/rest/services/Test/MapServer/tile/(\d+)/(\d+)/(\d+)", parts.path)
        if m:
            z, row, col = (int(v) for v in m.groups())
            if z > ZDATA:
                return self._send(404, b"<html>no tile</html>", "text/html")
            jj, ii = np.mgrid[0:256, 0:256]
            tile = pattern(col * 256 + ii, row * 256 + jj)
            return self._send(200, imagecodecs.png_encode(tile), "image/png")
        m = re.fullmatch(r"/arcgis/rest/services/Dtm/ImageServer/tile/(\d+)/(\d+)/(\d+)", parts.path)
        if m:
            z, row, col = (int(v) for v in m.groups())
            if z > DTM_Z or (row, col) == DTM_MISSING:
                return self._send(404, b"<html>no tile</html>", "text/html")
            jj, ii = np.mgrid[0:256, 0:256]
            vals = dtm_values(z, col * 256 + ii, row * 256 + jj)
            return self._send(200, imagecodecs.lerc_encode(vals, 0.0), "application/octet-stream")
        if parts.path == "/arcgis/rest/services/Dtm/ImageServer":
            meta = {"name": "Dtm", "bandCount": 1, "pixelType": "F32", "bandNames": ["Band_1"],
                    "tileInfo": {"rows": 256, "cols": 256, "format": "LERC2D",
                                 "origin": {"x": DTM_ORIGIN[0], "y": DTM_ORIGIN[1]},
                                 "spatialReference": {"wkid": 32617, "latestWkid": 32617},
                                 "lods": [{"level": z, "resolution": 2.0 ** (DTM_Z - z)} for z in range(DTM_Z + 1)]},
                    "extent": {"xmin": DTM_ORIGIN[0], "ymin": DTM_ORIGIN[1] - 700, "xmax": DTM_ORIGIN[0] + 900,
                               "ymax": DTM_ORIGIN[1], "spatialReference": {"wkid": 32617}},
                    "fullExtent": {"xmin": DTM_ORIGIN[0], "ymin": DTM_ORIGIN[1] - 700, "xmax": DTM_ORIGIN[0] + 900,
                                   "ymax": DTM_ORIGIN[1], "spatialReference": {"wkid": 32617}}}
            return self._send(200, json.dumps(meta).encode(), "application/json")
        if parts.path == "/arcgis/rest/services/Test/MapServer":
            meta = {"name": "Test", "copyrightText": "Test lab 2021",
                    "tileInfo": {"rows": 256, "cols": 256, "format": "MIXED",
                                 "origin": {"x": ORIGIN[0], "y": ORIGIN[1]},
                                 "spatialReference": {"wkid": 102100, "latestWkid": 3857},
                                 "lods": [{"level": z, "resolution": R0 / 2 ** z} for z in range(21)]},
                    "fullExtent": {"xmin": EXTENT[0], "ymin": EXTENT[1], "xmax": EXTENT[2], "ymax": EXTENT[3],
                                   "spatialReference": {"wkid": 102100}}}
            return self._send(200, json.dumps(meta).encode(), "application/json")
        return self._send(404, b"", "text/plain")


@pytest.fixture(scope="module")
def arcgis():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Arcgis)
    srv.daemon_threads = True
    srv.lock = threading.Lock()
    srv.paths = []
    srv.base = "http://127.0.0.1:{}/".format(srv.server_address[1])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture(autouse=True)
def _clean():
    rr.clear_reader_cache()
    rt._SERVICES.clear()
    yield
    rr.clear_reader_cache()
    rt._SERVICES.clear()


@pytest.fixture()
def pointer(arcgis, tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "ARCGIS_ITEM_URL", arcgis.base + "sharing/rest/content/items/{}?f=json")
    hub = "https://test-si.opendata.arcgis.com/maps/{}/about".format(ITEM)
    # the hub host is not ours: the item id is what is used
    info = rm.make_pointer(hub, str(tmp_path))
    return info


def _origin_at(z):
    r = R0 / 2 ** z
    return (int(math.floor((EXTENT[0] - ORIGIN[0]) / r)), int(math.floor((ORIGIN[1] - EXTENT[3]) / r)))


def test_service_item_and_hub_links_are_recognised():
    assert rt.looks_like_tile_service("https://tiles.arcgis.com/tiles/x/arcgis/rest/services/BCI/MapServer")
    assert rt.looks_like_tile_service("https://h.arcgis.com/arcgis/rest/services/A/ImageServer/tile/3/1/2")
    assert rt.looks_like_tile_service("https://stridata-si.opendata.arcgis.com/maps/{}/about".format(ITEM))
    assert rt.looks_like_tile_service("https://www.arcgis.com/home/item.html?id=" + ITEM)
    assert not rt.looks_like_tile_service("https://example.org/ortho.tif")
    assert not rt.looks_like_tile_service("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view")


def test_a_hub_page_becomes_a_tile_pointer(pointer, arcgis):
    assert pointer["provider"] == rt.KIND and pointer["zoom"] == ZDATA
    ptr = rm.read_pointer(pointer["path"])
    assert ptr["kind"] == rt.KIND and ptr["zmax"] == ZDATA
    assert ptr["url"] == arcgis.base + "arcgis/rest/services/Test/MapServer"
    assert os.path.basename(pointer["path"]) == "Test_orthophoto_2021.cogref"
    assert rt.is_tile_pointer(pointer["path"])


def test_pixels_at_every_level_are_the_services_pixels(pointer):
    prof = rr.probe(pointer["path"])
    assert prof is not None and prof.remote and prof.epsg == 3857 and prof.count == 4
    r = rr.open_reader(pointer["path"], prof)
    assert r.is_remote and isinstance(r, rt.TileServiceReader)
    ox, oy = _origin_at(ZDATA)
    H, W = r.level_shape(0)[:2]
    assert (W, H) == (prof.width, prof.height)
    for win in ((0, 0, W, H), (300, 200, 900, 700), (W - 40, H - 30, W, H)):
        x0, y0, x1, y1 = win
        jj, ii = np.mgrid[y0:y1, x0:x1]
        assert np.array_equal(r.read_window(*win), pattern(ox + ii, oy + jj)), win
    # a coarser level is the coarser zoom, from its own origin
    lv = r._levels[2]
    jj, ii = np.mgrid[0:lv["height"], 0:lv["width"]]
    assert np.array_equal(r.read_bands_full([0, 1, 2, 3], level=2),
                          pattern(lv["ox"] + ii, lv["oy"] + jj))
    # strided reads equal slicing
    a = r.read_window(10, 20, 700, 600, bands=[0, 2], level=0, step=3)
    b = r.read_window(10, 20, 700, 600, bands=[0, 2], level=0)[::3, ::3]
    assert np.array_equal(a, b)


def test_georeference_and_size_come_from_the_service(pointer):
    from ..shapefile_io import get_geotiff_transform, epsg_from_geotiff, _raw_image_dims
    prof = rr.probe(pointer["path"])
    tr, wkt = get_geotiff_transform(pointer["path"])
    ox, oy = _origin_at(ZDATA)
    r = R0 / 2 ** ZDATA
    assert tr == pytest.approx([ORIGIN[0] + ox * r, r, 0.0, ORIGIN[1] - oy * r, 0.0, -r])
    assert wkt and epsg_from_geotiff(pointer["path"]) == 3857
    assert _raw_image_dims(pointer["path"]) == (prof.height, prof.width)


def test_prefetch_then_reads_make_no_requests(pointer, arcgis):
    r = rr.open_reader(pointer["path"])
    rects = [(100, 100, 160, 140), (1200, 900, 1260, 960)]
    assert r.fetch_bytes_for_rects(rects) > 0
    r.prefetch_rects(rects)
    assert r.fetch_bytes_for_rects(rects) == 0
    n = r.remote_stats()["requests"]
    for (x0, y0, x1, y1) in rects:
        r.read_window(x0, y0, x1, y1)
    assert r.remote_stats()["requests"] == n


@pytest.mark.extraction
def test_canopie_opens_and_analyses_a_tile_service(pointer, synthetic_project):
    """Viewer preview, full-resolution polygon basis, CSV statistics over a
    polygon -- the values equal the service's pixels."""
    pt = synthetic_project
    ref = pointer["path"]
    lite = pt._imagedata_or_fallback(ref)
    prof = lite.profile
    assert lite.reader.is_remote and pt.polygon_basis_hw(None, image_data=lite) == (prof.height, prof.width)
    x0, y0, x1, y1 = 400, 300, 459, 340
    pd = {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "coord_space": "image",
          "image_ref_size": {"w": prof.width, "h": prof.height}, "name": "crown", "type": "polygon"}
    for attr in ("_export_image_cache", "_export_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()
    rows, _ = pt.process_polygon("crown", ref, pd, {}, [], False, opts={"stats": {"mean": True}})
    by = {r.get("Channel"): r for r in rows}
    ox, oy = _origin_at(ZDATA)
    jj, ii = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    want = pattern(ox + ii, oy + jj).reshape(-1, 4).astype(np.float64)
    assert by["R"]["Pixel Count"] == len(want)
    for k, ch in enumerate(("R", "G", "B")):
        assert by[ch]["Mean"] == pytest.approx(want[:, k].mean(), abs=1e-6)



# --------------------------------------------------------------------------
# Data tiles (LERC): an elevation model in its own UTM grid
# --------------------------------------------------------------------------

def test_a_lerc_image_service_streams_its_float_values(arcgis, tmp_path):
    info = rm.make_pointer(arcgis.base + "arcgis/rest/services/Dtm/ImageServer", str(tmp_path))
    assert info["data"] and info["bands"] == 1 and info["dtype"] == "float32" and info["zoom"] == DTM_Z
    prof = rr.probe(info["path"])
    assert prof.count == 1 and prof.dtype == np.float32 and prof.epsg == 32617
    assert prof.transform == pytest.approx([DTM_ORIGIN[0], 1.0, 0.0, DTM_ORIGIN[1], 0.0, -1.0])
    assert (prof.width, prof.height) == (900, 700)
    r = rr.open_reader(info["path"], prof)
    w = r.read_window(0, 0, 900, 700, level=0)[..., 0]
    jj, ii = np.mgrid[0:700, 0:900]
    want = dtm_values(DTM_Z, ii, jj)
    sentinel = np.abs(want) > 1e37
    missing = (jj // 256 == DTM_MISSING[0]) & (ii // 256 == DTM_MISSING[1])
    assert np.isnan(w[sentinel]).all(), "the NoData sentinel must read as NaN"
    assert np.isnan(w[missing]).all(), "a tile the service does not have must read as NaN"
    ok = ~sentinel & ~missing
    assert np.array_equal(w[ok], want[ok])
    # a coarser level is that zoom's own values
    lv = r._levels[1]
    jj, ii = np.mgrid[0:lv["height"], 0:lv["width"]]
    w1 = r.read_bands_full([0], level=1)[..., 0]
    want1 = dtm_values(DTM_Z - 1, lv["ox"] + ii, lv["oy"] + jj)
    ok1 = (np.abs(want1) < 1e37) & ~((jj // 256 == DTM_MISSING[0]) & (ii // 256 == DTM_MISSING[1]))
    assert np.array_equal(w1[ok1], want1[ok1])
