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

    def do_GET(self):
        srv = self.server
        parts = urlsplit(self.path)
        if "forbidden" in parts.path:
            body = b'{"error": {"code": 403, "message": "The request is missing a valid API key."}}'
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = os.path.join(srv.root, unquote(parts.path.lstrip("/")))
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
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.base = "http://127.0.0.1:{}/".format(srv.server_address[1])
    return srv


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _write_pyramid(path, data, tile=256):
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, tile=(tile, tile), compression="deflate", photometric="minisblack",
                 planarconfig="contig", subifds=2)
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
    with pytest.raises(rm.RemoteError, match=rm.GDRIVE_KEY_ENV):
        rm.resolve_pointer({"url": "gdrive://" + fid})
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

    # analysis/export and EXIF say no, with the reason
    img, info = pt._get_export_image(ref)
    assert img is None and "not available for remote images" in info["error"]
    from ..project_tab import _drop_remote_for_exif
    assert _drop_remote_for_exif([ref, local]) == [local]


def test_project_tab_explains_an_unreachable_pointer(served, synthetic_project, tmp_path):
    ref = str(tmp_path / "gone.cogref")
    rm.write_pointer(ref, served["srv"].base + "no_such.tif")
    with pytest.raises(RuntimeError, match=r"cannot open the remote image: .*HTTP 404"):
        synthetic_project._imagedata_or_fallback(ref)
