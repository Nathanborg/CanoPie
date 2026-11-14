"""
Remote Cloud-Optimized GeoTIFFs: read only the bytes CanoPie needs, over HTTP.

Why this exists
---------------
Orthomosaics run to 11 GB. Google Drive for desktop serves random reads of an
online-only file, but then pulls the WHOLE file into its local cache, so it
does not keep the file off the disk. A COG does not need that: its header
lists the byte offset of every tile, so a reader can fetch just the tiles a
view needs with HTTP ``Range`` requests.

Pieces
------
* :class:`RangeFile` -- a thread-safe, seekable, read-only file object over
  one URL. Small reads (the TIFF header parse makes hundreds) go through an
  LRU cache of aligned blocks; tile-sized reads are fetched exactly, because
  decoded tiles are cached by ``raster_reader`` anyway. :meth:`read_ranges`
  merges nearby ranges and fetches them in parallel -- one request per run of
  adjacent tiles, not one per tile.
* The ``.cogref`` pointer file -- a small JSON file that stands in for the
  image on disk, so everything in CanoPie that keys on the image path (sidecar
  names, ``.ax`` files, the image lists, project.json) keeps working::

      {"canopie_cogref": 1,
       "url": "https://example.org/ortho.tif",
       "size": 11234567890}

  Name it like the image it replaces (``ortho.tif`` -> ``ortho.cogref``) and it
  shares that image's ``ortho.ax`` and polygon files.
* URL kinds: any HTTPS server that honours ``Range`` (a static host, a public
  S3/GCS bucket), and Google Drive files shared "anyone with the link", read
  through the Drive API with an API key (``files.get?alt=media``). The key is
  never written into the code: it comes from the pointer file or from the
  environment variable named there (default ``CANOPIE_GDRIVE_API_KEY``), and
  it is masked in every name, message and log line.

Not covered: OAuth sign-in, EXIF (exiftool needs a local file), and exporting
from a remote image -- those paths raise :func:`not_supported_message`.

No Qt here, and ``requests`` is imported only when a remote file is opened.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = [
    "COGREF_EXT",
    "GDRIVE_KEY_ENV",
    "RangeFile",
    "RemoteError",
    "is_cogref",
    "read_pointer",
    "write_pointer",
    "resolve_pointer",
    "gdrive_file_id",
    "open_pointer",
    "clear_sources",
    "set_default_api_key",
    "onedrive_kind",
    "resolve_onedrive",
    "add_progress_listener",
    "remove_progress_listener",
    "default_api_key",
    "gdrive_metadata",
    "inspect_url",
    "make_pointer",
    "pointer_stem",
    "redact",
    "not_supported_message",
]

COGREF_EXT = ".cogref"
POINTER_VERSION = 1
GDRIVE_KEY_ENV = "CANOPIE_GDRIVE_API_KEY"

#: Small reads are served from aligned blocks of this size. The header of the
#: BCI COG (8 levels, ~9k tiles at level 0) is ~0.2 MB, so one or two blocks
#: hold all of it.
BLOCK_SIZE = 256 * 1024
#: Per-file block-cache budget. Tiles bypass it (see RangeFile), so this only
#: has to hold the header and the odd small read.
CACHE_BYTES = 16 * 1024 * 1024
#: read_ranges merges two ranges when the gap between them is at most this.
#: A request costs a round trip (~20-150 ms); 64 KB costs ~10 ms at 50 Mbit/s.
MERGE_GAP = 64 * 1024
#: ...and never builds one request larger than this.
MAX_SPAN = 16 * 1024 * 1024
#: Concurrent HTTP requests for read_ranges, across all remote files.
HTTP_WORKERS = 16
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60


class RemoteError(IOError):
    """A remote read failed. The message never contains an API key."""


class _Expired(RemoteError):
    """A temporary download address (OneDrive / SharePoint) stopped working."""


# --------------------------------------------------------------------------
# Secrets stay out of names, messages and logs
# --------------------------------------------------------------------------

_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:key|access_token|token|sig|signature|tempauth|authkey|download_token)=)[^&#\s'\"]+")


def redact(text):
    """``text`` with API keys / tokens in URL query strings masked."""
    return _SECRET_QUERY_RE.sub(r"\1***", str(text))


def not_supported_message(what, path):
    """The one wording every unsupported remote operation uses."""
    return ("{name}: {what} is not available for remote images (.cogref) in this "
            "version of CanoPie. Download the file to work with it locally."
            .format(name=os.path.basename(str(path)), what=what))


# --------------------------------------------------------------------------
# Pointer files
# --------------------------------------------------------------------------

def is_cogref(path):
    """True when ``path`` names a ``.cogref`` pointer (by extension only)."""
    try:
        return os.fspath(path).lower().endswith(COGREF_EXT)
    except TypeError:
        return False


def read_pointer(path):
    """Load and validate a ``.cogref`` file. Returns the dict."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            ptr = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RemoteError("{}: not a readable .cogref pointer ({})".format(
            os.path.basename(str(path)), exc)) from None
    if not isinstance(ptr, dict) or not isinstance(ptr.get("url"), str) or not ptr["url"].strip():
        raise RemoteError("{}: a .cogref pointer needs a \"url\"".format(os.path.basename(str(path))))
    size = ptr.get("size")
    if size is not None and (not isinstance(size, int) or size <= 0):
        raise RemoteError("{}: \"size\" must be a positive integer".format(os.path.basename(str(path))))
    auth = ptr.get("auth")
    if auth is not None and not isinstance(auth, dict):
        raise RemoteError("{}: \"auth\" must be an object".format(os.path.basename(str(path))))
    return ptr


def write_pointer(path, url, *, size=None, auth=None, **extra):
    """Write a ``.cogref`` pointer atomically. ``auth`` is an optional dict:
    ``{"api_key_env": "NAME"}`` (Drive key from that environment variable),
    ``{"api_key": "..."}`` (key stored in the pointer -- it then travels with
    the project folder), or ``{"bearer_token_env": "NAME"}`` (an
    ``Authorization: Bearer`` token from the environment)."""
    ptr = {"canopie_cogref": POINTER_VERSION, "url": str(url)}
    if size is not None:
        ptr["size"] = int(size)
    if auth:
        ptr["auth"] = dict(auth)
    ptr.update(extra)
    path = os.fspath(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ptr, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return ptr


_GDRIVE_PATTERNS = (
    re.compile(r"^gdrive://([A-Za-z0-9_-]{10,})"),
    re.compile(r"^https?://drive\.google\.com/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"^https?://www\.googleapis\.com/drive/v3/files/([A-Za-z0-9_-]{10,})"),
)


def gdrive_file_id(url):
    """The Drive file id in a ``gdrive://ID`` URL, a share link
    (``drive.google.com/file/d/ID/view``, ``.../open?id=ID``, ``.../uc?id=ID``)
    or an API URL. ``None`` for anything else."""
    url = str(url).strip()
    for pat in _GDRIVE_PATTERNS:
        m = pat.match(url)
        if m:
            return m.group(1)
    parts = urlsplit(url)
    if parts.netloc.lower() in ("drive.google.com", "docs.google.com",
                                "drive.usercontent.google.com"):
        ids = parse_qs(parts.query).get("id")
        if ids and re.fullmatch(r"[A-Za-z0-9_-]{10,}", ids[0]):
            return ids[0]
    return None


def _env(name):
    val = os.environ.get(name) if name else None
    return val.strip() if val and val.strip() else None


#: Drive API base. A module constant so the QC suite can point it at a local
#: test server that behaves like the Drive API.
GDRIVE_API = "https://www.googleapis.com/drive/v3/files"
#: Drive's public download endpoint: a file shared "Anyone with the link"
#: downloads from here without any key, and it honours Range. ``confirm=t``
#: skips the "can't scan this file for viruses" page Drive puts in front of
#: files over ~100 MB. Not a documented API (the key route is), so it is the
#: fallback when no key is set.
GDRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download"

# The key the app was given (File > Add Remote Images, remembered in the
# user's settings). Never written into pointer files, which travel with the
# project; see set_default_api_key.
_DEFAULT_API_KEY = None


def set_default_api_key(key):
    """The Google Drive API key to use when a pointer names none. Set by the
    GUI from its settings; ``None`` clears it."""
    global _DEFAULT_API_KEY
    _DEFAULT_API_KEY = (str(key).strip() or None) if key else None


def default_api_key():
    return _DEFAULT_API_KEY


def _drive_key(auth, url, api_key=None):
    """Key for a Drive request: the pointer's own, the caller's, the app's,
    the environment's, or one already in the URL -- in that order."""
    key = (auth.get("api_key") or api_key or _DEFAULT_API_KEY
           or _env(auth.get("api_key_env") or GDRIVE_KEY_ENV))
    if not key:
        existing = parse_qs(urlsplit(url).query).get("key")
        key = existing[0] if existing else None
    return key


# --------------------------------------------------------------------------
# OneDrive (personal) and SharePoint / OneDrive for work or school
# --------------------------------------------------------------------------

#: The consumer OneDrive "shares" endpoint; a module constant so the QC suite
#: can point it at a local stand-in.
ONEDRIVE_API = "https://api.onedrive.com/v1.0/shares"
_ONEDRIVE_HOSTS = ("1drv.ms", "onedrive.live.com", "api.onedrive.com")


def onedrive_kind(url, provider=None):
    """``"onedrive"`` (personal), ``"sharepoint"`` (work or school) or None.
    ``provider`` (a pointer's own "provider" field) wins over the host name."""
    if provider in ("onedrive", "sharepoint"):
        return provider
    host = urlsplit(str(url).strip()).netloc.lower().split(":")[0]
    if host in _ONEDRIVE_HOSTS or host.endswith(".onedrive.live.com"):
        return "onedrive"
    if host.endswith(".sharepoint.com"):
        return "sharepoint"
    return None


def _share_token(url):
    """Microsoft's encoding of a sharing URL for the shares endpoint."""
    return "u!" + base64.urlsafe_b64encode(str(url).encode("utf-8")).decode("ascii").rstrip("=")


def _with_query(url, **params):
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in params]
    q += list(params.items())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


def _onedrive_candidates(url, kind, headers):
    """Addresses that may serve the file itself, most direct first. A share
    link opens a web page; these are the forms that download instead."""
    if kind == "onedrive":
        yield "{}/{}/root/content".format(ONEDRIVE_API, _share_token(url))
    yield _with_query(url, download="1")
    if kind == "onedrive":
        # an older-style link resolves to onedrive.live.com/redir?...; its
        # /download twin serves the file
        try:
            r = _session().get(url, headers=dict(headers or {}), stream=True,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True)
            final = r.url
            r.close()
            if "/redir" in final:
                yield final.replace("/redir", "/download", 1)
        except Exception:
            pass


def resolve_onedrive(url, kind, headers=None):
    """The direct, Range-capable address of a OneDrive / SharePoint share
    link: the first candidate that answers a 1-byte Range request with 206
    and file bytes (not a web page). Its final address after redirects is
    returned, so later reads skip the redirect -- it carries a temporary
    token, which _RangeSource renews when it expires."""
    tried = []
    for cand in _onedrive_candidates(url, kind, headers):
        try:
            hdrs = dict(headers or {})
            hdrs["Range"] = "bytes=0-0"
            r = _session().get(cand, headers=hdrs, stream=True,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True)
            ctype = (r.headers.get("Content-Type") or "").lower()
            ok = r.status_code == 206 and "text/html" not in ctype
            final = r.url
            r.close()
            if ok:
                return final
            tried.append("HTTP {}{}".format(r.status_code, " (a web page)" if "text/html" in ctype else ""))
        except Exception as exc:
            tried.append(type(exc).__name__)
    raise RemoteError(
        "{}: no direct download address for this {} link ({}). Share the file as "
        "'Anyone with the link' (Can view); a work or school account may not allow "
        "that.".format(_name_from_url(url), "OneDrive" if kind == "onedrive" else "SharePoint",
                       "; ".join(tried) or "nothing answered"))


def resolve_pointer(ptr, api_key=None):
    """``(url, headers)`` to fetch, from a pointer dict.

    Google Drive links become ``files.get?alt=media`` API URLs when there is
    a key: ``auth.api_key``, else ``api_key``, else the app's key
    (:func:`set_default_api_key`), else the environment variable
    ``auth.api_key_env`` (default ``CANOPIE_GDRIVE_API_KEY``).
    ``auth.bearer_token_env`` sends an ``Authorization: Bearer`` header
    instead, for a token made elsewhere. With neither, the file is read from
    Drive's public download endpoint (``GDRIVE_DOWNLOAD``), which needs the
    file shared as "Anyone with the link" and no key at all.
    """
    url = str(ptr["url"]).strip()
    auth = ptr.get("auth") or {}
    headers = {}
    token = _env(auth.get("bearer_token_env"))
    if auth.get("bearer_token_env") and not token:
        raise RemoteError("the pointer asks for a bearer token in environment variable "
                          "{}, which is not set".format(auth["bearer_token_env"]))
    if token:
        headers["Authorization"] = "Bearer " + token

    file_id = gdrive_file_id(url)
    if file_id is not None:
        api = "{}/{}?alt=media&supportsAllDrives=true".format(GDRIVE_API, file_id)
        if not token:
            key = _drive_key(auth, url, api_key)
            if not key:
                return ("{}?id={}&export=download&confirm=t".format(GDRIVE_DOWNLOAD, file_id),
                        headers)
            api += "&key=" + key
        return api, headers

    scheme = urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise RemoteError("unsupported URL scheme {!r} (use https:// or a Google Drive link)"
                          .format(scheme or url))
    return url, headers


# --------------------------------------------------------------------------
# Download progress (for the status bar)
# --------------------------------------------------------------------------

_PROGRESS_LISTENERS = []


def add_progress_listener(fn):
    """``fn(batch_id, label, done_bytes, total_bytes)`` is called while a batch
    of remote reads downloads (from worker threads; at most every 0.1 s, and
    once at 100 %). ``batch_id`` tells concurrent downloads apart. The GUI
    marshals it to the status bar."""
    if fn not in _PROGRESS_LISTENERS:
        _PROGRESS_LISTENERS.append(fn)


def remove_progress_listener(fn):
    try:
        _PROGRESS_LISTENERS.remove(fn)
    except ValueError:
        pass


class _Progress:
    """Bytes received for one batch of reads, reported to the listeners."""

    def __init__(self, label, total):
        self.label, self.total, self.done = label, max(1, int(total)), 0
        self._lock = threading.Lock()
        self._last = 0.0
        self._emit(0)

    def add(self, n):
        with self._lock:
            self.done = min(self.total, self.done + int(n))
            now = time.perf_counter()
            if now - self._last < 0.1 and self.done < self.total:
                return
            self._last = now
            done = self.done
        self._emit(done)

    def finish(self):
        with self._lock:
            self.done = self.total
        self._emit(self.total)

    def _emit(self, done):
        for fn in list(_PROGRESS_LISTENERS):
            try:
                fn(id(self), self.label, done, self.total)
            except Exception:
                pass


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

_SESSION = None
_SESSION_LOCK = threading.Lock()
_IO_POOL = None


def _session():
    """One keep-alive session for every remote file. urllib3's pool is
    thread-safe; the session itself is never mutated after creation."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                try:
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                except ImportError as exc:      # pragma: no cover - requests ships with folium
                    raise RemoteError("remote images need the 'requests' package: {}".format(exc))
                s = requests.Session()
                retry = Retry(total=3, connect=3, read=2, backoff_factor=0.4,
                              status_forcelist=(429, 500, 502, 503, 504),
                              allowed_methods=frozenset(["GET", "HEAD"]),
                              raise_on_status=False)
                adapter = HTTPAdapter(pool_connections=8, pool_maxsize=HTTP_WORKERS * 2,
                                      max_retries=retry)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                s.headers["User-Agent"] = "CanoPie-RangeFile/1"
                s.headers["Accept-Encoding"] = "identity"    # byte offsets must be raw bytes
                _SESSION = s
    return _SESSION


def _io_pool():
    global _IO_POOL
    if _IO_POOL is None:
        with _SESSION_LOCK:
            if _IO_POOL is None:
                _IO_POOL = ThreadPoolExecutor(max_workers=HTTP_WORKERS,
                                              thread_name_prefix="canopie-http")
    return _IO_POOL


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


class _RangeSource:
    """Shared state behind one URL: size, block cache, statistics."""

    def __init__(self, url, *, headers=None, size=None, label=None,
                 block_size=BLOCK_SIZE, cache_bytes=CACHE_BYTES, resolver=None):
        # resolver: a callable giving the real download address (OneDrive /
        # SharePoint). Called now, and again when that address expires.
        self._resolver = resolver
        self._resolve_lock = threading.Lock()
        self.label = label or redact(url)
        self.url = resolver() if resolver is not None else url
        self.headers = dict(headers or {})
        self.block_size = int(block_size)
        self.cache_budget = int(cache_bytes)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._pending = {}
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "bytes_fetched": 0, "cache_hits": 0,
                      "http_seconds": 0.0}
        self.first_headers = {}
        self.size = self._discover_size()
        if size is not None and int(size) != self.size:
            raise RemoteError(
                "{}: the remote file is {:,} bytes but the pointer says {:,} -- it was "
                "replaced or the link points elsewhere. Polygons are stored in this "
                "file's pixel coordinates, so CanoPie will not open a different file "
                "under the same name.".format(self.label, self.size, int(size)))

    # -- HTTP -------------------------------------------------------------

    def _get(self, start, end, total_known=True, progress=None):
        """:meth:`_get_once`, renewing an expired download address once."""
        used = self.url
        try:
            return self._get_once(start, end, total_known, progress)
        except _Expired:
            with self._resolve_lock:
                if self.url == used:                   # nobody renewed it meanwhile
                    self.url = self._resolver()
            return self._get_once(start, end, total_known, progress)

    def _get_once(self, start, end, total_known=True, progress=None):
        """Bytes ``[start, end]`` (inclusive) with one Range request.

        Returns ``(data, total_size)``. Refuses a 200 response: a server that
        ignores Range would otherwise stream the whole file into memory."""
        hdrs = dict(self.headers)
        hdrs["Range"] = "bytes={}-{}".format(int(start), int(end))
        t0 = time.perf_counter()
        try:
            resp = _session().get(self.url, headers=hdrs, stream=True,
                                  timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except Exception as exc:
            raise RemoteError("{}: request failed: {}: {}".format(
                self.label, type(exc).__name__, redact(exc))) from None
        try:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if resp.status_code in (200, 206) and ("text/html" in ctype or any(
                    "accounts.google.com" in (h.headers.get("Location") or "")
                    for h in resp.history)):
                # A web page instead of the file -- typically Google's sign-in
                # page for a Drive file that is not shared publicly.
                signin = "accounts.google.com" in (resp.url or "") or any(
                    "accounts.google.com" in (h.headers.get("Location") or "") for h in resp.history)
                try:
                    page = resp.raw.read(8192, decode_content=True).decode("utf-8", "replace")
                except Exception:
                    page = ""
                tm = re.search(r"<title>(.*?)</title>", page, re.S | re.I)
                title = " ".join(tm.group(1).split()) if tm else ""
                if signin:
                    what, fix = " (a Google sign-in page)", (
                        "The file is not shared publicly: in Drive, Share > General access > "
                        "'Anyone with the link' (Viewer).")
                elif "quota" in title.lower():
                    what, fix = " ({})".format(title), (
                        "Google limits downloads of a popular public file for a while; "
                        "try later, or use an API key.")
                else:
                    what, fix = (" ({})".format(title) if title else ""), (
                        "Use a direct link to the .tif itself.")
                raise RemoteError("{}: the link returned a web page{} instead of the file. {}"
                                  .format(self.label, what, fix))
            if resp.status_code == 200:
                raise RemoteError(
                    "{}: the server ignored the byte-range request (HTTP 200 instead of 206), "
                    "so it would send the whole file. Host the COG somewhere that supports "
                    "HTTP Range requests.".format(self.label))
            if resp.status_code in (401, 403, 410) and self._resolver is not None:
                raise _Expired("{}: download address expired (HTTP {})".format(
                    self.label, resp.status_code))
            if resp.status_code != 206:
                try:
                    body = resp.raw.read(600, decode_content=True) or b""
                except Exception:
                    body = b""
                snippet = redact(body.decode("utf-8", "replace")).strip().replace("\n", " ")
                raise RemoteError("{}: HTTP {} {}{}".format(
                    self.label, resp.status_code, resp.reason or "",
                    (": " + snippet[:300]) if snippet else ""))
            m = _CONTENT_RANGE_RE.match(resp.headers.get("Content-Range", ""))
            if not m or int(m.group(1)) != int(start):
                raise RemoteError("{}: bad Content-Range {!r} for bytes {}-{}".format(
                    self.label, resp.headers.get("Content-Range"), start, end))
            total = None if m.group(3) == "*" else int(m.group(3))
            if start == 0:
                self.first_headers = dict(resp.headers)
            if progress is None:
                data = resp.content
            else:
                parts = []
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        parts.append(chunk)
                        progress.add(len(chunk))
                data = b"".join(parts)
        except RemoteError:
            raise
        except Exception as exc:
            raise RemoteError("{}: read failed: {}: {}".format(
                self.label, type(exc).__name__, redact(exc))) from None
        finally:
            resp.close()
        want = int(m.group(2)) - int(m.group(1)) + 1
        if len(data) != want:
            raise RemoteError("{}: short read: {} of {} bytes at offset {}".format(
                self.label, len(data), want, start))
        dt = time.perf_counter() - t0
        with self._lock:
            self.stats["requests"] += 1
            self.stats["bytes_fetched"] += len(data)
            self.stats["http_seconds"] += dt
        return data, total

    def _discover_size(self):
        """One request for the first block: it gives the total size (from
        Content-Range) and the start of the TIFF header, which is cached."""
        data, total = self._get(0, self.block_size - 1)
        if total is None:
            raise RemoteError("{}: the server did not report the file size".format(self.label))
        if total > 0 and data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise RemoteError("{}: not a TIFF (starts with {!r}). For Google Drive, check the "
                              "file is shared as 'Anyone with the link'.".format(self.label, data[:8]))
        if len(data) == self.block_size or len(data) == total:
            self._store({0: data})
        return int(total)

    # -- block cache ------------------------------------------------------

    def _store(self, blocks):
        with self._lock:
            for b, data in blocks.items():
                old = self._cache.pop(b, None)
                if old is not None:
                    self._cache_bytes -= len(old)
                self._cache[b] = data
                self._cache_bytes += len(data)
            while self._cache_bytes > self.cache_budget and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= len(old)

    def _block_span(self, first, last):
        bs = self.block_size
        return first * bs, min(self.size, (last + 1) * bs) - 1

    def _blocks(self, indices):
        """``{block: bytes}`` for ``indices``. Misses are fetched as runs of
        consecutive blocks; a block another thread is already fetching is
        waited for instead of being requested twice."""
        have, mine, waits = {}, [], []
        with self._lock:
            for b in indices:
                data = self._cache.get(b)
                if data is not None:
                    self._cache.move_to_end(b)
                    self.stats["cache_hits"] += 1
                    have[b] = data
                elif b in self._pending:
                    waits.append((b, self._pending[b]))
                else:
                    self._pending[b] = threading.Event()
                    mine.append(b)
        try:
            run = []
            for b in mine + [None]:
                if run and (b is None or b != run[-1] + 1):
                    start, end = self._block_span(run[0], run[-1])
                    data, _ = self._get(start, end)
                    fetched = {r: data[k * self.block_size:(k + 1) * self.block_size]
                               for k, r in enumerate(run)}
                    have.update(fetched)
                    self._store(fetched)
                    run = []
                if b is not None:
                    run.append(b)
        finally:
            with self._lock:
                for b in mine:
                    ev = self._pending.pop(b, None)
                    if ev is not None:
                        ev.set()
        for b, ev in waits:
            ev.wait(CONNECT_TIMEOUT + READ_TIMEOUT)
            with self._lock:
                data = self._cache.get(b)
            if data is None:           # evicted already, or that fetch failed
                start, end = self._block_span(b, b)
                data, _ = self._get(start, end)
            have[b] = data
        return have

    # -- reads ------------------------------------------------------------

    def _from_cache(self, offset, end):
        """Bytes ``[offset, end)`` if every block covering them is cached."""
        bs = self.block_size
        first, last = offset // bs, (end - 1) // bs
        with self._lock:
            blocks = [self._cache.get(b) for b in range(first, last + 1)]
            if any(b is None for b in blocks):
                return None
            for b in range(first, last + 1):
                self._cache.move_to_end(b)
            self.stats["cache_hits"] += len(blocks)
        s = offset - first * bs
        return b"".join(blocks)[s:s + (end - offset)]

    def pread(self, offset, n, cached=True, progress=None):
        """``n`` bytes at ``offset``.

        ``cached=True`` (the file API, i.e. the header parse): served from
        aligned blocks, fetching missing ones. ``cached=False`` (tile data):
        exactly these bytes in one request, unless the cache already holds
        them -- a 37 KB JPEG tile must not cost a 256 KB block. A read of a
        block or more is always fetched exactly."""
        offset = int(offset)
        if offset < 0:
            raise ValueError("negative offset")
        if n is None or n < 0:
            n = self.size - offset
        end = min(self.size, offset + int(n))
        if end <= offset:
            return b""
        if not cached or end - offset >= self.block_size:
            # Tile data: decoded tiles are cached by raster_reader, so the
            # compressed bytes are not kept here. Small overview tiles that
            # sit in the header's blocks (GDAL writes them right after the
            # IFDs) are still served from those blocks.
            data = self._from_cache(offset, end)
            if data is None:
                data, _ = self._get(offset, end - 1, progress=progress)
            elif progress is not None:
                progress.add(len(data))
            return data
        bs = self.block_size
        first, last = offset // bs, (end - 1) // bs
        blocks = self._blocks(range(first, last + 1))
        buf = b"".join(blocks[b] for b in range(first, last + 1))
        s = offset - first * bs
        return buf[s:s + (end - offset)]

    def read_ranges(self, ranges, merge_gap=MERGE_GAP, max_span=MAX_SPAN):
        order = sorted((int(off), int(n), i) for i, (off, n) in enumerate(ranges))
        spans = []                     # [start, end_excl, [(i, off, n), ...]]
        for off, n, i in order:
            if n <= 0:
                continue
            if spans and off - spans[-1][1] <= merge_gap \
                    and max(spans[-1][1], off + n) - spans[-1][0] <= max_span:
                spans[-1][1] = max(spans[-1][1], off + n)
                spans[-1][2].append((i, off, n))
            else:
                spans.append([off, off + n, [(i, off, n)]])
        out = [b""] * len(ranges)
        total = sum(sp[1] - sp[0] for sp in spans)
        progress = _Progress(self.label, total) if spans else None
        t0 = time.perf_counter()

        def fetch(span):
            return span, self.pread(span[0], span[1] - span[0], cached=False, progress=progress)

        if len(spans) <= 1:
            done = [fetch(s) for s in spans]
        else:
            done = list(_io_pool().map(fetch, spans))
        if progress is not None:
            progress.finish()
            logging.info("[remote] %s: streamed %.1f MB in %d request(s), %.1f s",
                         self.label, total / 1e6, len(spans), time.perf_counter() - t0)
        for (start, _end, parts), data in done:
            for i, off, n in parts:
                out[i] = data[off - start:off - start + n]
        return out


# --------------------------------------------------------------------------
# File object
# --------------------------------------------------------------------------

class RangeFile:
    """A read-only, seekable file object over an HTTP URL.

    ``seek``/``tell``/``read``/``readinto`` behave like a local binary file,
    which is all ``tifffile.TiffFile`` needs. Each call is atomic, so one
    handle can be shared between threads; for concurrent reads prefer
    :meth:`pread` (no shared position) or one :meth:`clone` per thread -- the
    clones share the size probe, block cache, statistics and connection pool.
    """

    mode = "rb"

    def __init__(self, url=None, *, headers=None, size=None, name=None,
                 block_size=BLOCK_SIZE, cache_bytes=CACHE_BYTES, _source=None):
        if _source is None:
            if not url:
                raise ValueError("RangeFile needs a URL")
            _source = _RangeSource(url, headers=headers, size=size, label=name,
                                   block_size=block_size, cache_bytes=cache_bytes)
        self._src = _source
        self.name = name or _source.label
        self._pos = 0
        self._closed = False
        self._lock = threading.Lock()

    # -- file protocol ----------------------------------------------------

    @property
    def size(self):
        return self._src.size

    @property
    def url(self):
        """The URL with any API key masked (safe to show or log)."""
        return redact(self._src.url)

    @property
    def closed(self):
        return self._closed

    def _check(self):
        if self._closed:
            raise ValueError("I/O operation on closed RangeFile")

    def seek(self, offset, whence=io.SEEK_SET):
        self._check()
        with self._lock:
            if whence == io.SEEK_SET:
                pos = int(offset)
            elif whence == io.SEEK_CUR:
                pos = self._pos + int(offset)
            elif whence == io.SEEK_END:
                pos = self._src.size + int(offset)
            else:
                raise ValueError("invalid whence {}".format(whence))
            if pos < 0:
                raise OSError("negative seek position {}".format(pos))
            self._pos = pos
            return pos

    def tell(self):
        self._check()
        return self._pos

    def read(self, n=-1):
        self._check()
        with self._lock:
            data = self._src.pread(self._pos, n)
            self._pos += len(data)
            return data

    def readinto(self, buffer):
        mv = memoryview(buffer).cast("B")
        data = self.read(len(mv))
        mv[:len(data)] = data
        return len(data)

    def pread(self, offset, n, cached=True):
        """``n`` bytes at ``offset`` without touching the file position.
        ``cached=False`` fetches exactly these bytes (for tile data)."""
        self._check()
        return self._src.pread(offset, n, cached=cached)

    def read_ranges(self, ranges):
        """Bytes for each ``(offset, length)``, in order. Ranges closer than
        ``MERGE_GAP`` share a request; separate requests run in parallel."""
        self._check()
        return self._src.read_ranges(list(ranges))

    def clone(self):
        """A new handle (own position) on the same source and cache."""
        return RangeFile(name=self.name, _source=self._src)

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("a RangeFile has no file descriptor")

    def flush(self):
        pass

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- diagnostics ------------------------------------------------------

    def stats(self):
        """Counters shared by every clone of this source."""
        with self._src._lock:
            out = dict(self._src.stats)
            out["cache_bytes"] = self._src._cache_bytes
        out["size"] = self._src.size
        return out

    def reset_stats(self):
        with self._src._lock:
            for k in self._src.stats:
                self._src.stats[k] = 0.0 if isinstance(self._src.stats[k], float) else 0

    def __repr__(self):
        return "<RangeFile {} size={}>".format(self.name, self._src.size)


# --------------------------------------------------------------------------
# One source per pointer file, shared by probe() and every reader
# --------------------------------------------------------------------------

_SOURCES = OrderedDict()
_SOURCES_LOCK = threading.Lock()
_MAX_SOURCES = 16


def _source_for(ptr, label, api_key=None):
    """The _RangeSource for a pointer dict: resolved URL, headers, and for a
    OneDrive / SharePoint link a resolver that finds and renews the direct
    download address."""
    url, headers = resolve_pointer(ptr, api_key=api_key)
    kind = onedrive_kind(ptr["url"], ptr.get("provider"))
    resolver = None
    if kind is not None:
        share = str(ptr["url"]).strip()

        def resolver():
            return resolve_onedrive(share, kind, headers)
    return _RangeSource(url, headers=headers, size=ptr.get("size"), label=label, resolver=resolver)


def open_pointer(path):
    """A :class:`RangeFile` for a ``.cogref`` pointer.

    The underlying source (size, block cache, statistics) is shared per
    pointer file and its modification time, so ``probe()`` and the reader
    opened right after it parse the header from the same cached blocks.
    """
    abspath = os.path.abspath(os.fspath(path))
    try:
        mtime = os.stat(abspath).st_mtime_ns
    except OSError as exc:
        raise RemoteError("{}: {}".format(os.path.basename(abspath), exc)) from None
    key = (os.path.normcase(abspath), mtime)
    with _SOURCES_LOCK:
        src = _SOURCES.get(key)
        if src is not None:
            _SOURCES.move_to_end(key)
    if src is None:
        ptr = read_pointer(abspath)
        src = _source_for(ptr, os.path.basename(abspath))
        with _SOURCES_LOCK:
            src = _SOURCES.setdefault(key, src)
            _SOURCES.move_to_end(key)
            while len(_SOURCES) > _MAX_SOURCES:
                _SOURCES.popitem(last=False)
    return RangeFile(name=abspath, _source=src)


def clear_sources():
    with _SOURCES_LOCK:
        _SOURCES.clear()


# --------------------------------------------------------------------------
# Checking a link and writing its pointer (File > Add Remote Images)
# --------------------------------------------------------------------------

def _drive_error(resp):
    """The message in a Drive API error response, without secrets."""
    try:
        msg = resp.json().get("error", {}).get("message") or ""
    except Exception:
        msg = resp.text[:300] if resp.text else ""
    msg = redact(msg).strip()
    hint = ""
    if resp.status_code == 404:
        hint = " -- check the link, and that the file is shared as 'Anyone with the link'"
    elif resp.status_code in (400, 403) and "key" in msg.lower():
        hint = " -- check the API key, and that the Google Drive API is enabled for its project"
    return "Google Drive: HTTP {} {}{}".format(resp.status_code, msg, hint)


def gdrive_metadata(file_id, api_key=None, headers=None):
    """``{"name", "size", "mimeType"}`` of a Drive file, through the API."""
    params = {"fields": "name,size,mimeType", "supportsAllDrives": "true"}
    if api_key:
        params["key"] = api_key
    try:
        resp = _session().get("{}/{}".format(GDRIVE_API, file_id), params=params,
                              headers=dict(headers or {}),
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as exc:
        raise RemoteError("Google Drive: request failed: {}: {}".format(
            type(exc).__name__, redact(exc))) from None
    if resp.status_code != 200:
        raise RemoteError(_drive_error(resp))
    return resp.json()


def _disposition_name(headers):
    """The file name in a Content-Disposition header, or None."""
    from urllib.parse import unquote
    cd = (headers or {}).get("Content-Disposition") or ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
    if m:
        return unquote(m.group(1).strip().strip('"'))
    m = re.search(r'filename="?([^";]+)"?', cd, re.I)
    return m.group(1).strip() if m else None


def _name_from_url(url):
    from urllib.parse import unquote
    name = unquote(os.path.basename(urlsplit(url).path)) or "remote"
    return name


def inspect_url(url, api_key=None, provider=None):
    """Check a link before CanoPie uses it: reachable, Range-capable, a TIFF,
    and whether it is a real COG. Reads the name (from the Drive API for a
    Drive link), the size and the header -- nothing else.

    Returns a dict: url, name, size, width, height, bands, dtype, tiled,
    levels, compression, warning (None or why it will be slow), requests,
    bytes_fetched."""
    url = str(url).strip()
    real, headers = resolve_pointer({"url": url}, api_key=api_key)
    kind = onedrive_kind(url, provider)
    file_id = gdrive_file_id(url)
    keyless = file_id is not None and real.startswith(GDRIVE_DOWNLOAD)
    if file_id is not None and not keyless:
        key = None if "Authorization" in headers else _drive_key({}, url, api_key)
        name = gdrive_metadata(file_id, api_key=key, headers=headers).get("name") or (
            "gdrive_" + file_id[:10])
    elif keyless:
        name = "gdrive_" + file_id[:10]
    else:
        name = _name_from_url(url)
    ptr = {"url": url}
    if provider:
        ptr["provider"] = provider
    rf = RangeFile(name=name, _source=_source_for(ptr, name, api_key=api_key))
    if keyless or kind is not None:
        name = _disposition_name(rf._src.first_headers) or name
    import tifffile
    with tifffile.TiffFile(rf) as tf:
        page = tf.pages[0]
        try:
            levels = len(tf.series[0].levels)
        except Exception:
            levels = 1
        info = {
            "url": url, "name": name, "size": rf.size,
            "width": int(page.imagewidth), "height": int(page.imagelength),
            "bands": int(page.samplesperpixel or 1), "dtype": str(page.dtype),
            "tiled": bool(page.is_tiled), "levels": int(levels),
            "compression": int(page.compression),
            "provider": kind,
        }
    warning = None
    if not info["tiled"]:
        warning = ("not tiled: every view reads whole rows of the image -- convert it to a "
                   "Cloud-Optimized GeoTIFF")
    elif info["levels"] < 2 and max(info["width"], info["height"]) > 4096:
        warning = ("no overviews: the first preview reads the full-resolution image -- "
                   "convert it to a Cloud-Optimized GeoTIFF")
    info["warning"] = warning
    st = rf.stats()
    info["requests"], info["bytes_fetched"] = st["requests"], st["bytes_fetched"]
    return info


_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def pointer_stem(name):
    """A file-name stem for a remote image called ``name``."""
    stem = str(name)
    if stem.lower().endswith((".tif", ".tiff")):
        stem = os.path.splitext(stem)[0]
    stem = _UNSAFE_NAME.sub("_", stem).strip(" .")
    return (stem or "remote")[:120]


def make_pointer(url, folder, api_key=None, provider=None):
    """Check ``url`` (:func:`inspect_url`) and write ``<name>.cogref`` in
    ``folder``. A pointer already there for the same link is reused; one for
    another link keeps its name and the new one gets ``_2``, ``_3``...
    The API key is never written. Returns the info dict plus ``path``."""
    info = inspect_url(url, api_key=api_key, provider=provider)
    os.makedirs(folder, exist_ok=True)
    stem = pointer_stem(info["name"])
    n = 1
    while True:
        path = os.path.join(folder, stem + ("" if n == 1 else "_{}".format(n)) + COGREF_EXT)
        if not os.path.exists(path):
            break
        try:
            old = read_pointer(path)
            if str(old.get("url", "")).strip() == info["url"]:
                break
        except RemoteError:
            pass
        n += 1
    extra = {"provider": provider} if provider else {}
    write_pointer(path, info["url"], size=info["size"], name=info["name"], **extra)
    info["path"] = path
    return info


# --------------------------------------------------------------------------
# Command line: make and check pointer files
# --------------------------------------------------------------------------

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m canopie.remote_raster",
        description="Create or check a .cogref pointer to a remote Cloud-Optimized GeoTIFF.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write a .cogref for a URL (checks it first)")
    mk.add_argument("url", help="https URL, or a Google Drive share link / gdrive://FILE_ID")
    mk.add_argument("out", help="pointer to write, e.g. ortho.cogref (name it like the image)")
    mk.add_argument("--api-key-env", default=None,
                    help="environment variable holding the Drive API key "
                         "(default {})".format(GDRIVE_KEY_ENV))
    mk.add_argument("--bearer-token-env", default=None,
                    help="environment variable holding an OAuth bearer token")
    info = sub.add_parser("info", help="open a .cogref and print what CanoPie sees")
    info.add_argument("pointer")
    args = ap.parse_args(argv)

    from . import raster_reader
    if args.cmd == "make":
        out = args.out if is_cogref(args.out) else args.out + COGREF_EXT
        auth = {}
        if args.api_key_env:
            auth["api_key_env"] = args.api_key_env
        if args.bearer_token_env:
            auth["bearer_token_env"] = args.bearer_token_env
        url, headers = resolve_pointer({"url": args.url, "auth": auth})
        rf = RangeFile(url, headers=headers, name=redact(url))
        import tifffile
        with tifffile.TiffFile(rf) as tf:
            page = tf.pages[0]
            levels = len(tf.series[0].levels)
            print("{}x{} px, {} band(s), {}, tiled={}, levels={}, compression={}".format(
                page.imagewidth, page.imagelength, page.samplesperpixel, page.dtype,
                page.is_tiled, levels, page.compression))
            if not page.is_tiled or levels < 2:
                print("warning: not a tiled COG with overviews -- CanoPie will have to read "
                      "much more of it than it would from a COG")
        write_pointer(out, args.url, size=rf.size, auth=auth or None)
        print("wrote {} ({:,} bytes remote; {} request(s), {:,} bytes read to check it)".format(
            out, rf.size, rf.stats()["requests"], rf.stats()["bytes_fetched"]))
        return 0

    prof = raster_reader.probe(args.pointer)
    if prof is None:
        print("could not open: {}".format(raster_reader.last_probe_error(args.pointer)))
        return 1
    print(prof.describe())
    print("source:", prof.source)
    print("io:", open_pointer(args.pointer).stats())
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(_main())
