"""
Remote COGs: CanoPie reads a GeoTIFF over HTTP Range requests, via a .cogref
pointer, and gets exactly the bytes and pixels a local read gets.

A local http.server thread (with Range support, which the stock
SimpleHTTPRequestHandler lacks) serves fixture TIFFs written here:

* ``pyr.tif`` -- 4-band uint8, 256x256 DEFLATE tiles with partial edge tiles,
  and two overview levels: the COG case.
* ``strips.tif`` -- 3-band uint16, uncompressed, planar=separate strips. Local
  probe() maps it (memmap); a remote one must use strips instead.

Every remote result is compared with the local file read the same way.
"""
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
import tifffile

from .. import raster_reader as rr
from .. import remote_raster as rm

pytestmark = [pytest.mark.io]


# --------------------------------------------------------------------------
# A Range-capable static server
# --------------------------------------------------------------------------

class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive, like a real host

    def log_message(self, *args):          # quiet
        pass

    def _json(self, code, obj):
        import json
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drive(self, parts):
        """Just enough of the Drive v3 files API: metadata and alt=media."""
        from urllib.parse import parse_qs
        srv = self.server
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        file_id = parts.path.rsplit("/", 1)[-1]
        if q.get("key") != srv.drive_key:
            return self._json(400, {"error": {"code": 400, "message":
                                              "API key not valid. Please pass a valid API key."}})
        if file_id not in srv.drive_files:
            return self._json(404, {"error": {"code": 404,
                                              "message": "File not found: {}.".format(file_id)}})
        local, name = srv.drive_files[file_id]
        if q.get("alt") == "media":
            return local                      # served below, with Range
        return self._json(200, {"name": name, "size": str(os.path.getsize(local)),
                                "mimeType": "image/tiff"})

    def _download(self, parts):
        """Drive's public download endpoint: shared files stream (with Range
        and a file name); a restricted one redirects to a sign-in page."""
        from urllib.parse import parse_qs
        srv = self.server
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        fid = q.get("id", "")
        if fid in srv.drive_private:
            self.send_response(302)
            self.send_header("Location", srv.base + "accounts.google.com/v3/signin/identifier")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if fid not in srv.drive_files or q.get("confirm") != "t":
            body = b"<html><title>Google Drive - Virus scan warning</title></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        local, name = srv.drive_files[fid]
        self._disposition = 'attachment; filename="{}"'.format(name)
        return local

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _onedrive(self, parts):
        """Share page, ?download=1 and the shares endpoint redirect to a blob
        address with a temporary token; a private file goes to sign-in."""
        import base64
        from urllib.parse import parse_qs
        srv = self.server
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        path = parts.path
        with srv.lock:
            srv.paths.append(path)
        if path.startswith("/od/api/shares/"):
            tok = path.split("/")[4][2:]
            share = base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4)).decode()
            name = urlsplit(share).path.rsplit("/", 1)[-1]
            if name in srv.od_private:
                return self._json(401, {"error": {"code": "unauthenticated"}})
            return self._redirect(srv.base + "od/blob/{}?tempauth={}".format(name, srv.od_token))
        if path.startswith("/od/share/"):
            name = path.rsplit("/", 1)[-1]
            if q.get("download") == "1":
                if name in srv.od_private:
                    return self._redirect(srv.base + "accounts.google.com/../login.microsoftonline.com")
                return self._redirect(srv.base + "od/blob/{}?tempauth={}".format(name, srv.od_token))
            body = b"<html><title>OneDrive</title></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        if path.startswith("/od/blob/"):
            if q.get("tempauth") != srv.od_token:
                return self._json(401, {"error": {"code": "expired"}})
            name = path.rsplit("/", 1)[-1]
            self._disposition = 'attachment; filename="{}"'.format(name)
            return os.path.join(srv.root, name)
        return None

    def do_GET(self):
        srv = self.server
        parts = urlsplit(self.path)
        drive_path = None
        self._disposition = None
        if parts.path.startswith("/od/"):
            drive_path = self._onedrive(parts)
            if drive_path is None:
                return
        if parts.path.startswith("/accounts.google.com/"):
            body = b"<html><title>Sign in - Google Accounts</title></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parts.path == "/download":
            drive_path = self._download(parts)
            if drive_path is None:
                return
        if parts.path.startswith("/drive/v3/files/"):
            drive_path = self._drive(parts)
            if drive_path is None:
                return
        if "forbidden" in parts.path:
            body = b'{"error": {"code": 403, "message": "The request is missing a valid API key."}}'
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = drive_path or os.path.join(srv.root, unquote(parts.path.lstrip("/")))
        if not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        with srv.lock:
            srv.log.append(rng)
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng or "")
        try:
            with open(path, "rb") as fh:
                if m and not srv.ignore_range:
                    start = int(m.group(1))
                    end = min(size - 1, int(m.group(2)) if m.group(2) else size - 1)
                    if start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */{}".format(size))
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    fh.seek(start)
                    body = fh.read(end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
                else:
                    body = fh.read()
                    self.send_response(200)
                self.send_header("Content-Type", "image/tiff")
                if self._disposition:
                    self.send_header("Content-Disposition", self._disposition)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                with srv.lock:
                    srv.bytes_sent += len(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def _start_server(root, ignore_range=False):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    srv.daemon_threads = True
    srv.root = str(root)
    srv.ignore_range = ignore_range
    srv.lock = threading.Lock()
    srv.log = []
    srv.bytes_sent = 0
    srv.drive_key = "test-key"
    srv.drive_files = {}
    srv.drive_private = set()
    srv.paths = []
    srv.od_token = "t1"
    srv.od_private = set()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.base = "http://127.0.0.1:{}/".format(srv.server_address[1])
    return srv


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# GeoTIFF tags for level 0: 5 cm pixels, UTM 17N (EPSG:32617, BCI's zone)
_GEO_EXTRATAGS = [
    (33550, "d", 3, (0.05, 0.05, 0.0), True),                            # ModelPixelScale
    (33922, "d", 6, (0.0, 0.0, 0.0, 626000.0, 1012000.0, 0.0), True),   # ModelTiepoint
    (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32617), True),
]


def _write_pyramid(path, data, tile=256):
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, tile=(tile, tile), compression="deflate", photometric="minisblack",
                 planarconfig="contig", subifds=2, extratags=_GEO_EXTRATAGS)
        tw.write(data[::2, ::2], tile=(tile, tile), compression="deflate",
                 photometric="minisblack", planarconfig="contig", subfiletype=1)
        tw.write(data[::4, ::4], tile=(tile, tile), compression="deflate",
                 photometric="minisblack", planarconfig="contig", subfiletype=1)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote_cog")
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:1000, 0:1300]
    pyr = np.stack([(xx * 7 + yy) % 251, (yy * 3) % 241, (xx ^ yy) % 256,
                    rng.integers(0, 256, size=xx.shape)], axis=-1).astype(np.uint8)
    _write_pyramid(str(root / "pyr.tif"), pyr)

    strips = rng.integers(0, 60000, size=(3, 300, 410)).astype(np.uint16)
    tifffile.imwrite(str(root / "strips.tif"), strips, photometric="minisblack",
                     planarconfig="separate", rowsperstrip=16)

    srv = _start_server(root)
    refs = {}
    for name in ("pyr", "strips"):
        local = str(root / (name + ".tif"))
        ref = str(root / (name + ".cogref"))
        rm.write_pointer(ref, srv.base + name + ".tif", size=os.path.getsize(local))
        refs[name] = (local, ref)
    yield {"root": root, "srv": srv, "refs": refs, "pyr": pyr}
    srv.shutdown()
    srv.server_close()
    rr.clear_reader_cache()
    rm.clear_sources()


@pytest.fixture(autouse=True)
def _fresh_remote_state():
    rm.clear_sources()
    rr.clear_reader_cache()
    yield
    rr.clear_reader_cache()
    rm.clear_sources()


# --------------------------------------------------------------------------
# RangeFile
# --------------------------------------------------------------------------

def test_rangefile_reads_equal_local_reads(served):
    local, _ = served["refs"]["pyr"]
    blob = open(local, "rb").read()
    size = len(blob)
    # small blocks, so header-sized reads cross several of them
    rf = rm.RangeFile(served["srv"].base + "pyr.tif", block_size=4096, cache_bytes=64 * 1024)
    assert rf.size == size
    rng = np.random.default_rng(0)
    cases = [(0, 0), (0, 1), (0, 8), (4090, 20), (4096, 4096), (100, 70_000),
             (size - 10, 10), (size - 10, 100), (size, 5), (size + 50, 5)]
    cases += [(int(o), int(n)) for o, n in zip(rng.integers(0, size, 60),
                                               rng.integers(1, 20_000, 60))]
    with open(local, "rb") as fh:
        for off, n in cases:
            fh.seek(off)
            rf.seek(off)
            assert rf.read(n) == fh.read(n), (off, n)
            assert rf.tell() == fh.tell(), (off, n)
            assert rf.pread(off, n) == blob[off:off + n], (off, n)
        # whence, read-to-end, readinto
        assert rf.seek(-100, os.SEEK_END) == fh.seek(-100, os.SEEK_END)
        assert rf.read() == fh.read()
        rf.seek(1000)
        rf.seek(500, os.SEEK_CUR)
        buf = bytearray(3000)
        assert rf.readinto(buf) == 3000 and bytes(buf) == blob[1500:4500]
    with pytest.raises(OSError):
        rf.seek(-1)
    with pytest.raises(OSError):
        rf.fileno()


def test_rangefile_is_thread_safe(served):
    """Many threads on one handle (pread) and on clones (seek+read), with a
    tiny cache so blocks are evicted and re-fetched while others wait."""
    local, _ = served["refs"]["pyr"]
    blob = open(local, "rb").read()
    rf = rm.RangeFile(served["srv"].base + "pyr.tif", block_size=2048, cache_bytes=16 * 1024)
    errors = []

    def work(seed):
        try:
            r = np.random.default_rng(seed)
            mine = rf.clone()
            for _ in range(150):
                off = int(r.integers(0, len(blob)))
                n = int(r.integers(1, 9000))
                if r.random() < 0.5:
                    got = rf.pread(off, n)
                else:
                    mine.seek(off)
                    got = mine.read(n)
                if got != blob[off:off + n]:
                    errors.append((seed, off, n))
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors[:5]


def test_read_ranges_merges_neighbours_and_matches(served):
    local, _ = served["refs"]["pyr"]
    blob = open(local, "rb").read()
    rf = rm.RangeFile(served["srv"].base + "pyr.tif")
    rf.reset_stats()
    # three runs of adjacent ranges, far apart, given out of order
    ranges = [(400_000, 300_000), (0, 1000), (1000, 2000), (3000, 500),
              (200_000, 50_000), (250_000, 60_000), (5, 10)]
    ranges = [(o, n) for o, n in ranges if o < len(blob)]
    got = rf.read_ranges(ranges)
    assert got == [blob[o:o + n] for o, n in ranges]
    # the run at 0 is inside block 0, cached when the size was probed; the
    # other two runs are fetched exactly -- no rounding up to whole blocks
    st = rf.stats()
    assert st["requests"] == 2, st
    assert st["bytes_fetched"] == 110_000 + 300_000, st


# --------------------------------------------------------------------------
# probe / reader through a .cogref
# --------------------------------------------------------------------------

_SAME_FIELDS = ("width", "height", "count", "dtype", "axes", "tiled", "tile_w", "tile_h",
                "planar", "compression", "predictor", "n_levels", "level_shapes",
                "nodata", "epsg", "transform")


@pytest.mark.parametrize("name", ["pyr", "strips"])
def test_probe_remote_matches_local(served, name):
    local, ref = served["refs"][name]
    lp, rp = rr.probe(local), rr.probe(ref)
    assert lp is not None and rp is not None, rr.last_probe_error(ref)
    for f in _SAME_FIELDS:
        assert getattr(lp, f) == getattr(rp, f), f
    assert rp.remote is True and lp.remote is False
    assert rp.path == os.path.abspath(ref)          # identity is the pointer's path
    assert rp.source == served["srv"].base + name + ".tif"
    assert rp.is_windowable
    assert rp.strategy != rr.STRATEGY_MEMMAP
    if name == "strips":
        assert lp.strategy == rr.STRATEGY_MEMMAP, "fixture no longer exercises the memmap switch"
        assert rp.strategy == rr.STRATEGY_STRIPS


def _windows(h, w):
    return [(0, 0, w, h), (10, 20, 300, 270), (250, 250, 520, 513), (w - 37, h - 41, w, h),
            (511, 0, 513, h), (0, 255, w, 257)]


@pytest.mark.parametrize("name", ["pyr", "strips"])
def test_remote_reads_equal_local_reads(served, name):
    local, ref = served["refs"][name]
    lr = rr.open_reader(local)
    rrd = rr.open_reader(ref)
    assert rrd is not None and rrd.is_remote and not lr.is_remote
    assert rrd._memmap is None
    for level in range(len(lr._levels)):
        h, w, c = lr.level_shape(level)
        assert rrd.level_shape(level) == (h, w, c)
        bands = list(range(c))
        for (x0, y0, x1, y1) in _windows(h, w):
            for step in (1, 3):
                a = lr.read_window(x0, y0, x1, y1, bands=bands, level=level, step=step)
                b = rrd.read_window(x0, y0, x1, y1, bands=bands, level=level, step=step)
                assert a.dtype == b.dtype and np.array_equal(a, b), (level, x0, y0, x1, y1, step)
        for step in (1, 2, 5):
            sub = bands[::-1][:3]
            a = lr.read_bands_full(sub, level=level, step=step)
            b = rrd.read_bands_full(sub, level=level, step=step)
            assert np.array_equal(a, b), (level, step)


def test_pyramid_fixture_matches_written_pixels(served):
    """The remote reader returns the pixels that were written, not merely the
    same thing the local reader returns."""
    _, ref = served["refs"]["pyr"]
    rrd = rr.open_reader(ref)
    pyr = served["pyr"]
    assert np.array_equal(rrd.read_window(0, 0, 1300, 1000), pyr)
    assert np.array_equal(rrd.read_window(0, 0, 325, 250, level=2), pyr[::4, ::4])


def test_remote_open_and_window_fetch_only_what_they_need(served):
    local, ref = served["refs"]["pyr"]
    size = os.path.getsize(local)
    prof = rr.probe(ref)
    after_probe = rm.open_pointer(ref).stats()
    # tifffile writes the IFDs after the pixel data (GDAL's COG layout puts
    # them all in the first block), so the header here takes a few blocks
    assert after_probe["requests"] <= 4, after_probe
    assert after_probe["bytes_fetched"] <= 4 * rm.BLOCK_SIZE < size / 2, after_probe
    rrd = rr.open_reader(ref, prof)
    st = rrd.remote_stats()
    # probe and reader share one source: the header comes over once
    assert st["requests"] == after_probe["requests"], (st, after_probe)
    rrd._remote.reset_stats()
    rrd.read_window(300, 300, 400, 400, level=0)        # inside one 256x256 tile
    st = rrd.remote_stats()
    lv = rrd._levels[0]
    seg = rrd._segment_index(lv, 0, 1, 1)
    assert st["requests"] == 1, st
    assert st["bytes_fetched"] == int(lv["counts"][seg]), "a tile read must fetch exactly the tile"
    rrd._remote.reset_stats()
    rrd.read_window(300, 300, 400, 400, level=0)        # decoded tile is cached
    assert rrd.remote_stats()["requests"] == 0


# --------------------------------------------------------------------------
# Refusals and messages
# --------------------------------------------------------------------------

def test_server_without_range_support_is_refused(served):
    srv = _start_server(served["root"], ignore_range=True)
    try:
        with pytest.raises(rm.RemoteError, match="Range"):
            rm.RangeFile(srv.base + "pyr.tif")
    finally:
        srv.shutdown()
        srv.server_close()


def test_pointer_size_mismatch_is_refused(served, tmp_path):
    local, _ = served["refs"]["pyr"]
    ref = str(tmp_path / "pyr.cogref")
    rm.write_pointer(ref, served["srv"].base + "pyr.tif", size=os.path.getsize(local) + 1)
    assert rr.probe(ref) is None
    assert "pointer says" in (rr.last_probe_error(ref) or "")


def test_missing_remote_file_reports_http_status(served, tmp_path):
    ref = str(tmp_path / "gone.cogref")
    rm.write_pointer(ref, served["srv"].base + "no_such.tif")
    assert rr.probe(ref) is None
    assert "HTTP 404" in rr.last_probe_error(ref)


def test_api_key_never_appears_in_errors_or_names(served, tmp_path):
    secret = "AIzaSECRETsecret123"
    url = served["srv"].base + "forbidden/x.tif?alt=media&key=" + secret
    with pytest.raises(rm.RemoteError) as ei:
        rm.RangeFile(url)
    assert "403" in str(ei.value) and secret not in str(ei.value)
    assert secret not in rm.redact(url)
    ref = str(tmp_path / "x.cogref")
    rm.write_pointer(ref, url)
    assert rr.probe(ref) is None
    assert secret not in rr.last_probe_error(ref)


def test_pointer_roundtrip_and_url_kinds(tmp_path, monkeypatch):
    ref = str(tmp_path / "ortho.cogref")
    rm.write_pointer(ref, "https://example.org/ortho.tif", size=123,
                     auth={"api_key_env": "MY_KEY"})
    ptr = rm.read_pointer(ref)
    assert ptr["url"] == "https://example.org/ortho.tif" and ptr["size"] == 123
    assert ptr["auth"] == {"api_key_env": "MY_KEY"}
    assert rm.is_cogref(ref) and rm.is_cogref("A.COGREF") and not rm.is_cogref("a.tif")
    assert rm.resolve_pointer({"url": "https://example.org/a.tif"}) == ("https://example.org/a.tif", {})

    fid = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    for link in ("gdrive://" + fid,
                 "https://drive.google.com/file/d/{}/view?usp=sharing".format(fid),
                 "https://drive.google.com/open?id=" + fid,
                 "https://drive.google.com/uc?id={}&export=download".format(fid),
                 "https://www.googleapis.com/drive/v3/files/{}?alt=media".format(fid)):
        assert rm.gdrive_file_id(link) == fid, link
    assert rm.gdrive_file_id("https://example.org/file/d/x") is None

    monkeypatch.delenv(rm.GDRIVE_KEY_ENV, raising=False)
    monkeypatch.setattr(rm, "_DEFAULT_API_KEY", None)
    # no key: Drive's public download endpoint (files shared by link)
    assert rm.resolve_pointer({"url": "gdrive://" + fid}) == (
        "{}?id={}&export=download&confirm=t".format(rm.GDRIVE_DOWNLOAD, fid), {})
    assert rm.gdrive_file_id(
        "https://drive.usercontent.google.com/download?id={}&export=download".format(fid)) == fid
    monkeypatch.setenv(rm.GDRIVE_KEY_ENV, "k123")
    url, headers = rm.resolve_pointer({"url": "https://drive.google.com/file/d/{}/view".format(fid)})
    assert url == ("https://www.googleapis.com/drive/v3/files/{}?alt=media"
                   "&supportsAllDrives=true&key=k123".format(fid))
    assert headers == {} and "k123" not in rm.redact(url)
    monkeypatch.setenv("MY_KEY", "other")
    url, _ = rm.resolve_pointer({"url": "gdrive://" + fid, "auth": {"api_key_env": "MY_KEY"}})
    assert url.endswith("key=other")
    monkeypatch.setenv("MY_TOKEN", "tok")
    url, headers = rm.resolve_pointer({"url": "gdrive://" + fid,
                                       "auth": {"bearer_token_env": "MY_TOKEN"}})
    assert "key=" not in url and headers == {"Authorization": "Bearer tok"}
    with pytest.raises(rm.RemoteError, match="scheme"):
        rm.resolve_pointer({"url": "file:///C:/ortho.tif"})

    bad = str(tmp_path / "bad.cogref")
    with open(bad, "w") as fh:
        fh.write('{"size": 5}')
    with pytest.raises(rm.RemoteError, match="url"):
        rm.read_pointer(bad)


def test_imagedata_refuses_a_pointer_clearly(served):
    from ..image_data import ImageData
    _, ref = served["refs"]["pyr"]
    with pytest.raises(ValueError, match="remote image"):
        ImageData(ref)


# --------------------------------------------------------------------------
# The project's loader
# --------------------------------------------------------------------------

def test_project_tab_opens_a_cogref_as_a_windowed_preview(served, synthetic_project):
    pt = synthetic_project
    local, ref = served["refs"]["pyr"]
    lite = pt._imagedata_or_fallback(ref)
    assert lite.reader is not None and lite.reader.is_remote
    assert lite.profile.remote and lite.channel_order == "rgb"
    assert lite.full_shape == (1000, 1300, 4)
    # 1300 x 1000 fits the preview budget at full resolution: the preview is
    # the file's own level-0 pixels for the display bands
    assert lite.display_scale == 1.0
    want = served["pyr"][..., list(lite.preview_bands)]
    assert np.array_equal(lite.image, want)

    # analysis reads it window by window; EXIF still says no
    img, info = pt._get_export_image(ref)
    assert img is not None and info.get("lazy"), info
    from ..project_tab import _drop_remote_for_exif
    assert _drop_remote_for_exif([ref, local]) == [local]


def test_project_tab_explains_an_unreachable_pointer(served, synthetic_project, tmp_path):
    ref = str(tmp_path / "gone.cogref")
    rm.write_pointer(ref, served["srv"].base + "no_such.tif")
    with pytest.raises(RuntimeError, match=r"cannot open the remote image: .*HTTP 404"):
        synthetic_project._imagedata_or_fallback(ref)


@pytest.mark.editor
def test_image_editor_opens_a_cogref_from_its_preview(served, synthetic_project, qapp):
    """The editor only writes .ax parameters, so a preview from the tiled
    reader is all it needs; the whole-file loaders cannot read a pointer."""
    from ..image_editor_dialog import ImageEditorDialog
    _, ref = served["refs"]["pyr"]
    dlg = ImageEditorDialog(synthetic_project, image_data=None, image_filepath=ref)
    try:
        assert dlg.base_image is not None
        assert dlg.base_image.shape[:2] == (1000, 1300)
        assert dlg._editor_channel_order == "rgb"
        assert np.array_equal(dlg.base_image, served["pyr"][..., list(dlg._editor_bands)])
        assert dlg._header_size() == {"h": 1000, "w": 1300, "c": 4}
        assert dlg._raw_shape() == (1000, 1300, 4)
    finally:
        dlg.close()
        dlg.deleteLater()


# --------------------------------------------------------------------------
# Size and georeference: the full-resolution frame, never the preview's
# --------------------------------------------------------------------------

def test_size_and_georeference_come_from_the_remote_header(served):
    from ..shapefile_io import _raw_image_dims, get_geotiff_transform, epsg_from_geotiff
    local, ref = served["refs"]["pyr"]
    assert _raw_image_dims(ref) == (1000, 1300)
    tr, wkt = get_geotiff_transform(ref)
    assert tr is not None and (tr, wkt) == get_geotiff_transform(local)
    assert tr == [626000.0, 0.05, 0, 1012000.0, 0, -0.05]
    assert epsg_from_geotiff(ref) == epsg_from_geotiff(local) == 32617


def test_polygon_basis_of_a_decimated_remote_preview_is_full_resolution(
        served, synthetic_project, monkeypatch):
    """Polygons are stored in full-raster pixels. On a remote image the size
    used to come back unknown, and the basis fell back to the PREVIEW's
    shape, so crowns drawn on it would have been stored 4x too small."""
    from .. import project_tab as ptm
    monkeypatch.setattr(ptm, "_PREVIEW_LOAD_THRESHOLD_BYTES", 600_000)   # -> level 2 preview
    pt = synthetic_project
    _, ref = served["refs"]["pyr"]
    lite = pt._imagedata_or_fallback(ref)
    assert lite.image.shape[:2] == (250, 325) and lite.display_scale == 4.0
    assert pt.polygon_basis_hw(None, image_data=lite) == (1000, 1300)
    assert pt._image_dims_only(ref) == (1000, 1300)


def test_canopy_finder_reads_a_cogref_window_by_window(served):
    from .. import canopy_finder as CF
    local, ref = served["refs"]["pyr"]
    g = CF.georef(ref)
    assert g is not None and (g.raw_w, g.raw_h) == (1300, 1000) and g.epsg == 32617
    assert CF.file_band_count(ref) == 4
    a, real_a = CF.read_raw_window(local, 1200, 900, 1400, 1100)      # runs off the edge
    b, real_b = CF.read_raw_window(ref, 1200, 900, 1400, 1100)
    assert np.array_equal(a, b) and np.array_equal(real_a, real_b)
    with pytest.raises(ValueError, match="not available for remote images"):
        CF._decode(ref)


# --------------------------------------------------------------------------
# File > Add Remote Images: checking links, writing pointers, opening roots
# --------------------------------------------------------------------------

DRIVE_ID = "1PyrFixtureFileIdForQC0123456789"
PRIVATE_ID = "1RestrictedFileIdForQC000000000"


@pytest.fixture()
def fake_drive(served, monkeypatch):
    """The test server answers as the Drive API for one shared file."""
    srv = served["srv"]
    local, _ = served["refs"]["pyr"]
    srv.drive_files = {DRIVE_ID: (local, "plot A ortho.tif")}
    srv.drive_private = {PRIVATE_ID}
    monkeypatch.setattr(rm, "GDRIVE_API", srv.base + "drive/v3/files")
    monkeypatch.setattr(rm, "GDRIVE_DOWNLOAD", srv.base + "download")
    monkeypatch.setattr(rm, "_DEFAULT_API_KEY", None)
    monkeypatch.delenv(rm.GDRIVE_KEY_ENV, raising=False)
    return srv


def test_make_pointer_names_reuses_and_never_clobbers(served, tmp_path):
    srv = served["srv"]
    info = rm.make_pointer(srv.base + "pyr.tif", str(tmp_path))
    assert info["path"] == str(tmp_path / "pyr.cogref")
    assert (info["width"], info["height"], info["bands"], info["levels"]) == (1300, 1000, 4, 3)
    assert info["warning"] is None and info["tiled"]
    ptr = rm.read_pointer(info["path"])
    assert ptr["url"] == srv.base + "pyr.tif" and ptr["size"] == info["size"]
    # the same link again: the same pointer
    assert rm.make_pointer(srv.base + "pyr.tif", str(tmp_path))["path"] == info["path"]
    # another file with the same name: a new pointer, the old one untouched
    sub = served["root"] / "sub"
    sub.mkdir(exist_ok=True)
    import shutil
    shutil.copy(served["refs"]["strips"][0], sub / "pyr.tif")
    other = rm.make_pointer(srv.base + "sub/pyr.tif", str(tmp_path))
    assert other["path"] == str(tmp_path / "pyr_2.cogref")
    assert rm.read_pointer(info["path"])["url"] == srv.base + "pyr.tif"
    # a stripped, overview-less file is flagged as slow
    assert other["warning"] and "not tiled" in other["warning"]


def test_a_drive_link_end_to_end(fake_drive, served, tmp_path):
    share = "https://drive.google.com/file/d/{}/view?usp=sharing".format(DRIVE_ID)
    local, _ = served["refs"]["pyr"]

    # no key: a file shared by link opens through the public download endpoint,
    # named from its Content-Disposition
    keyless = rm.make_pointer(share, str(tmp_path / "keyless"))
    assert os.path.basename(keyless["path"]) == "plot A ortho.cogref"
    a, b = rr.open_reader(local), rr.open_reader(keyless["path"])
    assert b is not None and b.is_remote, rr.last_probe_error(keyless["path"])
    assert np.array_equal(a.read_window(10, 20, 600, 500), b.read_window(10, 20, 600, 500))
    # a restricted file: Google's sign-in page, said as such
    with pytest.raises(rm.RemoteError, match="sign-in page.*Anyone with the link"):
        rm.make_pointer("https://drive.google.com/open?id=" + PRIVATE_ID, str(tmp_path))

    with pytest.raises(rm.RemoteError, match="API key not valid.*check the API key"):
        rm.make_pointer(share, str(tmp_path), api_key="wrong")
    with pytest.raises(rm.RemoteError, match="File not found.*Anyone with the link"):
        rm.make_pointer("gdrive://1NoSuchFileIdForQC000000000", str(tmp_path), api_key="test-key")

    info = rm.make_pointer(share, str(tmp_path), api_key="test-key")
    assert info["path"] == str(tmp_path / "plot A ortho.cogref"), "named after the Drive file"
    text = open(info["path"], encoding="utf-8").read()
    assert "test-key" not in text, "the API key must never be written into a pointer"

    # opening the pointer later uses the app's key (set from the settings)
    rm.set_default_api_key("test-key")
    a, b = rr.open_reader(local), rr.open_reader(info["path"])
    assert b is not None and b.is_remote, rr.last_probe_error(info["path"])
    assert np.array_equal(a.read_window(100, 200, 700, 900), b.read_window(100, 200, 700, 900))
    assert "test-key" not in (rr.probe(info["path"]).source or "")


def test_parse_urls_keeps_order_and_skips_comments():
    from ..remote_images_dialog import parse_urls
    text = "\n  https://a/x.tif \n# a note\n\nhttps://b/y.tif\nhttps://a/x.tif\n"
    assert parse_urls(text) == ["https://a/x.tif", "https://b/y.tif"]


@pytest.fixture()
def qc_settings(monkeypatch):
    """Keep the dialog's QSettings away from the user's real CanoPie settings."""
    from PyQt5 import QtCore
    from .. import remote_images_dialog as rid
    monkeypatch.setattr(rid, "_SETTINGS", ("CanoPie-QC", "remote-images-test"))
    yield rid
    QtCore.QSettings("CanoPie-QC", "remote-images-test").clear()


def test_dialog_checks_links_writes_pointers_and_reports_failures(
        fake_drive, served, tmp_path, qapp, qc_settings):
    rid = qc_settings
    srv = served["srv"]
    dlg = rid.RemoteImagesDialog(None, batch_size=2)
    try:
        dlg.urls_edit.setPlainText("\n".join([
            "https://drive.google.com/file/d/{}/view".format(DRIVE_ID),
            srv.base + "strips.tif",
            srv.base + "missing.tif",
        ]))
        dlg.folder_edit.setText(str(tmp_path / "pointers"))
        dlg.key_edit.setText("test-key")
        dlg.remember.setChecked(False)
        dlg._check_and_open()          # QMessageBox.question -> Yes (conftest)
        assert dlg.result() == dlg.Accepted
        assert [os.path.basename(p) for p in dlg.result_paths] == [
            "plot A ortho.cogref", "strips.cogref"], "link order, failures left out"
        assert dlg.batch_size == 2
        failed = [r for r in dlg.results if r[1] is None]
        assert len(failed) == 1 and "HTTP 404" in failed[0][2]
        assert dlg.results_list.count() == 3
        assert rm.default_api_key() == "test-key"
        from PyQt5 import QtCore
        assert not QtCore.QSettings(*rid._SETTINGS).value(rid.KEY_SETTING, ""), \
            "the key was remembered although 'Remember' was off"
    finally:
        dlg.deleteLater()


def test_open_image_paths_groups_remote_images_into_roots(served, qapp, tmp_path):
    """Three pointers, two per root: Remote1 shows two remote viewers."""
    import time
    from PyQt5 import QtWidgets
    from ..project_tab import ProjectTab
    paths = [served["refs"]["strips"][1], served["refs"]["pyr"][1]]
    third = str(tmp_path / "third.cogref")
    rm.write_pointer(third, served["srv"].base + "pyr.tif")
    paths.append(third)

    tabs = QtWidgets.QTabWidget()
    pt = ProjectTab("remote-qc", tabs)
    try:
        pt.open_image_paths(paths, 2, root_prefix="Remote", sort=False)
        assert pt.root_names == ["Remote1", "Remote2"]
        assert pt.multispectral_image_data_groups["Remote1"] == paths[:2]
        assert pt.multispectral_image_data_groups["Remote2"] == paths[2:]
        end = time.time() + 30

        def ready():
            return len(pt.viewer_widgets) == 2 and all(
                getattr(getattr(w["viewer"], "image_data", None), "reader", None) is not None
                for w in pt.viewer_widgets)

        while time.time() < end and not ready():
            qapp.processEvents()
            time.sleep(0.02)
        assert ready(), "the two viewers of Remote1 did not load"
        shown = [w["viewer"].image_data.filepath for w in pt.viewer_widgets]
        assert sorted(map(os.path.normcase, shown)) == sorted(map(os.path.normcase, paths[:2]))
        assert all(w["viewer"].image_data.reader.is_remote for w in pt.viewer_widgets)
    finally:
        for attr in ("_poly_idle_timer", "_autosave_timer"):
            t = getattr(pt, attr, None)
            if t is not None:
                t.stop()
        pt.deleteLater()
        tabs.deleteLater()
        qapp.processEvents()


def test_remote_preview_level_respects_the_download_budget(served, monkeypatch):
    """A remote preview picks a coarser level when the finer one would be a
    big download (a 298-band pixel-interleaved cube: 299 MB vs 19 MB); a
    local file is unaffected."""
    local, ref = served["refs"]["pyr"]
    lr, rrd = rr.open_reader(local), rr.open_reader(ref)
    sizes = [rrd.level_fetch_bytes(i, 3) for i in range(3)]
    assert sizes[0] > sizes[1] > sizes[2] > 0
    assert sizes == [lr.level_fetch_bytes(i, 3) for i in range(3)]
    big = 1 << 40
    assert lr.level_for_bytes(3, big, big, target_dim=None) == 0
    assert rrd.level_for_bytes(3, big, big, target_dim=None) == 0      # all fit 48 MB
    monkeypatch.setattr(rr, "_REMOTE_PREVIEW_FETCH_BYTES", (sizes[0] + sizes[1]) // 2)
    assert rrd.level_for_bytes(3, big, big, target_dim=None) == 1
    assert lr.level_for_bytes(3, big, big, target_dim=None) == 0, "local files have no download budget"
    monkeypatch.setattr(rr, "_REMOTE_PREVIEW_FETCH_BYTES", 1)
    assert rrd.level_for_bytes(3, big, big, target_dim=None) == 2, "nothing fits: the smallest level"


# --------------------------------------------------------------------------
# Zoom tiles on a remote file
# --------------------------------------------------------------------------

def _zoom_request(pt, fp, monkeypatch, request_id=None, current_id=None):
    """Ask for the zoom tile of a 1200 x 750 raw window shown in a 400 x 250
    viewport (a 1:3 zoom), the preview being level 2 of the fixture. Returns
    the levels the reader was asked for."""
    import time
    import types
    from PyQt5 import QtCore
    prof = rr.probe(fp)
    reader = rr.open_reader(fp, prof)
    disp = np.zeros((250, 325, 3), dtype=np.uint8)
    image_data = types.SimpleNamespace(
        filepath=fp, image=disp, reader=reader, profile=prof, preview_bands=[0, 1, 2],
        full_shape=(prof.height, prof.width, prof.count), display_scale=4.0)
    seen = []
    original = type(reader).read_window

    def _spy(self, x0, y0, x1, y1, bands=None, level=0, step=1):
        seen.append(level)
        return original(self, x0, y0, x1, y1, bands=bands, level=level, step=step)

    monkeypatch.setattr(type(reader), "read_window", _spy)

    class _VP:
        def width(self): return 400
        def height(self): return 250

    class _T:
        def m11(self): return 400 / 300.0

    viewer = types.SimpleNamespace(viewport=lambda: _VP(), transform=lambda: _T(),
                                   devicePixelRatioF=lambda: 1.0, image_data=image_data,
                                   _highres_request_id=current_id)
    rect = QtCore.QRectF(12.5, 31.25, 300.0, 187.5)        # raw 50..1250 x 125..875
    pt._request_highres_viewport_region(viewer, image_data, rect, request_id=request_id, ax={})
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not seen and request_id == current_id:
        time.sleep(0.02)
    time.sleep(0.3)
    return seen


def test_remote_zoom_tile_reads_the_coarsest_level_with_enough_pixels(
        served, synthetic_project, monkeypatch):
    """A 1:3 zoom in a landscape viewport: the local rule keeps level 0 (the
    window is not twice the viewport's longer side in height) and samples it;
    a remote file reads level 1, which still has 600 x 375 pixels for the
    400 x 250 on screen -- a quarter of the tiles."""
    local, ref = served["refs"]["pyr"]
    assert _zoom_request(synthetic_project, local, monkeypatch) == [0]
    assert _zoom_request(synthetic_project, ref, monkeypatch) == [1]


def test_a_superseded_zoom_request_never_reads(served, synthetic_project, monkeypatch):
    _, ref = served["refs"]["pyr"]
    assert _zoom_request(synthetic_project, ref, monkeypatch, request_id=1, current_id=2) == []
    assert _zoom_request(synthetic_project, ref, monkeypatch, request_id=3, current_id=3) == [1]



# --------------------------------------------------------------------------
# Analysis on remote images: CSV, Inspect, ML -- the same numbers as local
# --------------------------------------------------------------------------

ANALYSIS_FIXTURES = ["hyperspectral_200band", "ax_crop_nodata_source", "rgb_16bit_tiled_bip_cog"]
STATS_COLS = ("Mean", "Median", "Standard Deviation", "Q25", "Q75", "Pixel Count")


@pytest.fixture(scope="module")
def fixture_server(fixtures_ready):
    from .fixtures_manifest import FIXTURE_IMAGES_DIR
    srv = _start_server(FIXTURE_IMAGES_DIR)
    yield srv
    srv.shutdown()
    srv.server_close()


def _remote_twin(fixture_server, name, tmp_path):
    """A .cogref for a QC fixture, served over HTTP, with its .ax beside it."""
    import shutil
    from .fixtures_manifest import fixture_image_path
    fp = fixture_image_path(name)
    stem = os.path.splitext(os.path.basename(fp))[0]
    ref = str(tmp_path / (stem + ".cogref"))
    rm.write_pointer(ref, fixture_server.base + os.path.basename(fp), size=os.path.getsize(fp))
    ax = os.path.join(os.path.dirname(fp), stem + ".ax")
    if os.path.exists(ax):
        shutil.copy(ax, str(tmp_path / (stem + ".ax")))
    return fp, ref


def _clear_analysis_caches(pt):
    for attr in ("_export_image_cache", "_export_cache", "_per_band_nd_cache", "_master_nd_cache",
                 "_scene_stats_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def _csv_rows(pt, name, fp):
    from .fixtures_manifest import get_fixture
    from .project_builder import polygon_group_name
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = pt.all_polygons[group][fixture_image_path_of(name)]
    rows, _ = pt.process_polygon(group, fp, poly, {}, [], False,
                                 opts={"stats": {"mean": True, "median": True, "std": True,
                                                 "quantiles": [25, 75]}})
    return {r.get("Channel"): r for r in rows if isinstance(r, dict)}


def fixture_image_path_of(name):
    from .fixtures_manifest import fixture_image_path
    return fixture_image_path(name)


@pytest.mark.extraction
@pytest.mark.parametrize("name", ANALYSIS_FIXTURES)
def test_csv_statistics_of_a_remote_image_equal_the_local_ones(
        name, synthetic_project, fixture_server, tmp_path, monkeypatch):
    """Same polygon, same file: read from disk and streamed over HTTP. The
    remote file always takes the windowed path; it must give the numbers the
    local file gives on its own path (whole read for these small files)."""
    pt = synthetic_project
    fp, ref = _remote_twin(fixture_server, name, tmp_path)
    _clear_analysis_caches(pt)
    local = _csv_rows(pt, name, fp)
    _clear_analysis_caches(pt)
    remote = _csv_rows(pt, name, ref)
    assert local and set(local) == set(remote), (set(local), set(remote))
    for ch in local:
        for col in STATS_COLS:
            if col in local[ch]:
                assert remote[ch].get(col) == pytest.approx(local[ch][col], rel=1e-6, abs=1e-6), \
                    f"{name}/{ch} {col}: local {local[ch][col]} remote {remote[ch].get(col)}"
    img, info = pt._get_export_image(ref)
    assert info.get("lazy") and hasattr(img, "prefetch"), "a remote image must be read window by window"


@pytest.mark.extraction
def test_the_analysis_image_inspect_reads_is_the_same(synthetic_project, fixture_server, tmp_path, monkeypatch):
    """Inspect, CSV and ML read _get_export_image: a remote image's windows
    equal the local file's windowed analysis image, band for band."""
    from .. import project_tab as ptm
    pt = synthetic_project
    fp, ref = _remote_twin(fixture_server, "ax_crop_nodata_source", tmp_path)
    monkeypatch.setattr(ptm, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)        # local lazy too
    _clear_analysis_caches(pt)
    a, ia = pt._get_export_image(fp)
    _clear_analysis_caches(pt)
    b, ib = pt._get_export_image(ref)
    assert ia["layout"].names() == ib["layout"].names() and (ia["H"], ia["W"]) == (ib["H"], ib["W"])
    for win in ((0, 0, ia["W"], ia["H"]), (3, 5, 17, 21)):
        wa, wb = a.read_window(*win), b.read_window(*win)
        for k, (x, y) in enumerate(zip(wa, wb)):
            assert np.array_equal(x, y, equal_nan=True), f"band {k} differs in window {win}"


@pytest.mark.ml
def test_ml_training_rows_from_a_remote_image_equal_the_local_ones(
        synthetic_project, ml_manager_factory, fixture_server, tmp_path):
    from .fixtures_manifest import get_fixture
    from .project_builder import polygon_group_name
    name = "rgb_16bit_tiled_bip_cog"
    pt = synthetic_project
    fp, ref = _remote_twin(fixture_server, name, tmp_path)
    spec = get_fixture(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    mgr = ml_manager_factory(pt)
    pbg = mgr._snapshot_polygons_by_group([group])
    pbg_remote = {group: {ref: pbg[group][fp]}}

    def sample(path, polys):
        _clear_analysis_caches(pt)
        img, _c = mgr._get_export_image(path)
        assert img is not None, path
        xr, yr, *_rest = mgr._sample_training_file(
            group, path, img, polys, selected_band_indices=[0, 1, 2], selected_expressions=[],
            window_size=1, samples_per_poly=10 ** 9, num_features=3,
            augmentation_cfg={"enabled": False}, augmentation_enabled=False,
            n_variants=1, rng=np.random.default_rng(0))
        return np.asarray(xr, dtype=np.float64).reshape(-1, 3), list(yr)

    xl, yl = sample(fp, pbg)
    xr_, yr_ = sample(ref, pbg_remote)
    assert len(xl) > 0 and xl.shape == xr_.shape and yl == yr_
    assert np.array_equal(xl, xr_)
    # the manager's entry loop goes through the prefetcher, unchanged in order
    entries = [(group, ref), (group, fp), (group, ref)]
    assert list(mgr._remote_prefetched(entries, {group: {ref: pbg[group][fp], fp: pbg[group][fp]}})) == entries


def test_csv_export_prefetches_a_files_polygons_in_one_round(
        served, synthetic_project, tmp_path, monkeypatch):
    """Polygons across a remote image: iter_with_remote_prefetch downloads
    every tile under them before the first is processed, so processing them
    costs no further requests -- and yields them in their original order."""
    pt = synthetic_project
    _, ref = served["refs"]["pyr"]
    polys = []
    for k, (x, y) in enumerate([(100, 100), (900, 150), (300, 800), (1100, 900), (600, 500)]):
        polys.append(("g{}".format(k), {
            "points": [[x, y], [x + 60, y], [x + 60, y + 40], [x, y + 40]],
            "coord_space": "image", "image_ref_size": {"w": 1300, "h": 1000},
            "name": "g{}".format(k), "type": "polygon"}))
    _clear_analysis_caches(pt)
    img, info = pt._get_export_image(ref)
    assert info.get("lazy")
    reader = img._reader
    reader._remote.reset_stats()
    out = list(pt.iter_with_remote_prefetch(polys, lambda it: (ref, it[1])))
    assert out == polys
    after_prefetch = reader.remote_stats()["requests"]
    assert after_prefetch >= 1
    for _g, pd in polys:
        pts = pt._points_to_export_frame(ref, pd["points"], pd, img.shape)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        img.read_window(int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)
    assert reader.remote_stats()["requests"] == after_prefetch, \
        "reading the polygons after the prefetch went to the network again"


def test_a_remote_image_whose_ax_needs_the_whole_image_is_refused_with_the_reason(
        served, synthetic_project, tmp_path):
    import json
    import shutil
    pt = synthetic_project
    _, ref = served["refs"]["pyr"]
    ref2 = str(tmp_path / "pyr.cogref")
    shutil.copy(ref, ref2)
    with open(str(tmp_path / "pyr.ax"), "w") as fh:
        json.dump({"hist_enabled": True, "hist_match": {"enabled": True, "reference": "x.tif"}}, fh)
    _clear_analysis_caches(pt)
    img, info = pt._get_export_image(ref2)
    assert img is None and "needs the whole image" in info.get("error", "")


def test_download_progress_reaches_100_percent_per_batch(served):
    seen = []

    def listener(bid, label, done, total):
        seen.append((bid, label, done, total))

    rm.add_progress_listener(listener)
    try:
        rf = rm.RangeFile(served["srv"].base + "pyr.tif", name="pyr")
        rf.read_ranges([(300_000, 400_000), (2_000_000, 300_000)])
    finally:
        rm.remove_progress_listener(listener)
    assert seen, "no progress was reported"
    ids = {b for b, *_ in seen}
    assert len(ids) == 1
    dones = [d for _b, _l, d, _t in seen]
    assert dones[0] == 0 and dones == sorted(dones)
    assert seen[-1][2] == seen[-1][3] == 700_000 and seen[-1][1] == "pyr"


def test_status_bar_shows_streaming_percent_until_the_tiles_land(qapp):
    from PyQt5 import QtWidgets
    from ..remote_images_dialog import install_stream_progress
    win = QtWidgets.QMainWindow()
    w = install_stream_progress(win.statusBar())
    try:
        w._on_changed(1, "ortho.cogref", 0, 4_000_000)
        assert not w.isHidden() and w.bar.value() == 0 and "Streaming ortho.cogref" in w.label.text()
        w._on_changed(2, "ortho.cogref", 1_000_000, 4_000_000)       # a second viewer
        w._on_changed(1, "ortho.cogref", 3_000_000, 4_000_000)
        assert w.bar.value() == 50 and "4.0 MB of 8.0 MB" in w.label.text()
        w._on_changed(1, "ortho.cogref", 4_000_000, 4_000_000)
        w._on_changed(2, "ortho.cogref", 4_000_000, 4_000_000)
        assert w.bar.value() == 100 and "streamed" in w.label.text() and w._hide.isActive()
    finally:
        rm.remove_progress_listener(w.report)
        win.deleteLater()



# --------------------------------------------------------------------------
# OneDrive / SharePoint links
# --------------------------------------------------------------------------

def test_onedrive_and_sharepoint_links_are_recognised():
    assert rm.onedrive_kind("https://1drv.ms/i/s!AbCdEf") == "onedrive"
    assert rm.onedrive_kind("https://onedrive.live.com/?redeem=aHR0cHM6") == "onedrive"
    assert rm.onedrive_kind("https://contoso-my.sharepoint.com/:i:/g/personal/x/EaBc") == "sharepoint"
    assert rm.onedrive_kind("https://example.org/ortho.tif") is None
    assert rm.onedrive_kind("http://127.0.0.1:1/od/share/x.tif", provider="sharepoint") == "sharepoint"
    assert rm.gdrive_file_id("https://1drv.ms/i/s!AbCdEf") is None


def _od_count(srv, prefix):
    with srv.lock:
        return sum(1 for p in srv.paths if p.startswith(prefix))


def test_a_sharepoint_link_streams_from_its_direct_address(served, tmp_path):
    srv = served["srv"]
    local, _ = served["refs"]["pyr"]
    srv.od_token = "t1"
    share = srv.base + "od/share/pyr.tif"
    info = rm.make_pointer(share, str(tmp_path), provider="sharepoint")
    assert os.path.basename(info["path"]) == "pyr.cogref" and info["provider"] == "sharepoint"
    assert rm.read_pointer(info["path"])["provider"] == "sharepoint"
    assert "t1" not in open(info["path"], encoding="utf-8").read(), "a temporary token was saved"
    rm.clear_sources()
    a, b = rr.open_reader(local), rr.open_reader(info["path"])
    assert b is not None, rr.last_probe_error(info["path"])
    shares = _od_count(srv, "/od/share/")
    for win in ((0, 0, 1300, 1000), (700, 300, 1200, 900)):
        assert np.array_equal(a.read_window(*win), b.read_window(*win))
    assert _od_count(srv, "/od/share/") == shares, \
        "tile reads went through the share link (a redirect per request) instead of the direct address"


def test_an_expired_download_address_is_renewed(served, tmp_path):
    srv = served["srv"]
    local, _ = served["refs"]["pyr"]
    srv.od_token = "t1"
    ref = str(tmp_path / "pyr.cogref")
    rm.write_pointer(ref, srv.base + "od/share/pyr.tif", provider="sharepoint")
    b = rr.open_reader(ref)
    assert b is not None, rr.last_probe_error(ref)
    srv.od_token = "t2"                                     # the old address now answers 401
    a = rr.open_reader(local)
    assert np.array_equal(a.read_window(0, 0, 640, 480, level=1), b.read_window(0, 0, 640, 480, level=1))
    assert "tempauth=t2" in b._remote._src.url


def test_a_personal_onedrive_link_uses_the_shares_endpoint(served, tmp_path, monkeypatch):
    srv = served["srv"]
    local, _ = served["refs"]["pyr"]
    srv.od_token = "t1"
    monkeypatch.setattr(rm, "ONEDRIVE_API", srv.base + "od/api/shares")
    before = _od_count(srv, "/od/api/")
    info = rm.make_pointer(srv.base + "od/share/pyr.tif", str(tmp_path), provider="onedrive")
    assert _od_count(srv, "/od/api/") > before
    rm.clear_sources()
    a, b = rr.open_reader(local), rr.open_reader(info["path"])
    assert np.array_equal(a.read_window(10, 10, 900, 700), b.read_window(10, 10, 900, 700))


def test_a_onedrive_file_that_is_not_shared_says_so(served, tmp_path, monkeypatch):
    srv = served["srv"]
    srv.od_private = {"secret.tif"}
    monkeypatch.setattr(rm, "ONEDRIVE_API", srv.base + "od/api/shares")
    try:
        with pytest.raises(rm.RemoteError, match="Anyone with the link") as ei:
            rm.make_pointer(srv.base + "od/share/secret.tif", str(tmp_path), provider="onedrive")
        assert "tempauth" not in str(ei.value)
    finally:
        srv.od_private = set()


def test_onedrive_tokens_are_masked():
    assert "abc123" not in rm.redact("https://x.sharepoint.com/download.aspx?UniqueId=1&tempauth=abc123")
    assert "zzz" not in rm.redact("https://onedrive.live.com/download?resid=1&authkey=zzz")


# --------------------------------------------------------------------------
# Hover inspect never waits for the network on the GUI thread
# --------------------------------------------------------------------------

@pytest.mark.viewer
def test_hover_inspect_on_a_remote_image_answers_at_once_then_exactly(
        served, synthetic_project, qapp, monkeypatch):
    """Hover inspect runs on every mouse move. On a remote image whose
    full-resolution tile is not in memory it used to read it over the network
    on the GUI thread (~1 s per pixel on Drive): the window froze while
    panning. Now the preview's value comes back at once, and the exact value
    follows when the tile has downloaded in the background."""
    import time
    from PyQt5 import QtCore
    from .. import project_tab as ptm
    monkeypatch.setattr(ptm, "_PREVIEW_LOAD_THRESHOLD_BYTES", 600_000)    # a level-2 preview
    pt = synthetic_project
    local, ref = served["refs"]["pyr"]
    lite = pt._imagedata_or_fallback(ref)
    assert lite.display_scale == 4.0

    class _Viewer(QtCore.QObject):
        pixel_clicked = QtCore.pyqtSignal(object, object)

    v = _Viewer()
    v.image_data = lite
    got = []
    v.pixel_clicked.connect(lambda pt_, payload: got.append(payload))
    for attr in ("_export_image_cache", "_export_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()

    src = rm.open_pointer(ref)
    ex, ey = 1001, 777
    t0 = time.perf_counter()
    before = src.stats()["requests"]
    first = pt.inspect_export_pixel(v, ex, ey)
    took = time.perf_counter() - t0
    assert first is not None and first.get("approximate"), first
    assert took < 0.5 and src.stats()["requests"] == before, \
        "the hover readout waited for the network on the GUI thread"
    assert all(n.endswith("~") for n in first["names"])

    deadline = time.monotonic() + 15
    while not got and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert got, "the exact value never arrived"
    exact = got[-1]
    assert not exact.get("approximate") and exact["pixel"] == (ex, ey)
    assert exact["values"][:4] == [float(x) for x in served["pyr"][ey, ex, :4]]
    # the tile is in memory now: the next readout there is exact at once
    again = pt.inspect_export_pixel(v, ex + 1, ey)
    assert again is not None and not again.get("approximate")
