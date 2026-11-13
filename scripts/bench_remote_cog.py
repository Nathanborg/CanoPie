"""Time and count the bytes CanoPie reads from a COG: remote (.cogref) vs local.

    python scripts/bench_remote_cog.py --pointer ortho.cogref [--local ortho.tif] [--json out.json]

Replays what the app does when it shows an image, through the same
raster_reader calls:

open     probe() + open_reader(): the TIFF header.
preview  the viewer's first image -- display bands at the pyramid level and
         decimation _imagedata_or_fallback picks (same budget constants).
views    a sequence of zoom tiles for a 1600 x 1000 viewport: 1:1, pans, a jump,
         1:2 and 2x zoomed out, with the level/step rule of the viewer's
         high-res tile (ProjectTab, "Determine best pyramid level" /
         "Read at SCREEN resolution"). Decoded tiles stay cached between
         views, as in the app, so a pan only fetches the tiles it uncovers.

Remote numbers come from the RangeFile's own counters (requests, bytes on the
wire, seconds in HTTP). Local runs report time only; the OS file cache is
warm after the first run, which favours local.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from canopie import raster_reader as rr          # noqa: E402
from canopie import remote_raster as rm          # noqa: E402

VIEWPORT = (1600, 1000)


def _app_constants():
    """The viewer's preview budget, from project_tab without building a tab."""
    from canopie import project_tab as pt
    return pt._preview_budget_bytes, pt._PREVIEW_DECODE_BUDGET_BYTES


def _zoom_tile_plan(reader, x0, y0, x1, y1, vp=VIEWPORT):
    """Level and step for a full-resolution window, as the viewer's zoom tile
    chooses them (project_tab.py, the high-res viewport fetch)."""
    full_h, full_w, _ = reader.level_shape(0)
    vp_w, vp_h = vp
    target = max(vp_w, vp_h)
    win_w, win_h = x1 - x0, y1 - y0
    level = 0
    if len(reader._levels) > 1 and win_w >= target * 2 and win_h >= target * 2:
        for i, lv in enumerate(reader._levels):
            sx, sy = lv["width"] / float(full_w), lv["height"] / float(full_h)
            if win_w * sx >= target and win_h * sy >= target:
                level = i
            else:
                break
    lv = reader._levels[level]
    sx, sy = lv["width"] / float(full_w), lv["height"] / float(full_h)
    lx0, ly0 = int(x0 * sx), int(y0 * sy)
    lx1, ly1 = min(lv["width"], int(-(-x1 * sx // 1))), min(lv["height"], int(-(-y1 * sy // 1)))
    step = max(1, int(min((lx1 - lx0) / float(vp_w), (ly1 - ly0) / float(vp_h))))
    return level, step, (lx0, ly0, lx1, ly1)


def _views(w, h):
    vw, vh = VIEWPORT
    cx, cy = w // 2, h // 2
    out = [("1:1 centre", cx - vw // 2, cy - vh // 2, 1),
           ("pan right 800", cx + 800 - vw // 2, cy - vh // 2, 1),
           ("pan down 500", cx + 800 - vw // 2, cy + 500 - vh // 2, 1),
           ("1:1 jump to corner", w // 8, h // 8, 1),
           ("1:2 (zoom out)", w // 8, h // 8, 2),
           ("1:4", cx - 2 * vw, cy - 2 * vh, 4),
           ("1:8", max(0, cx - 4 * vw), max(0, cy - 4 * vh), 8)]
    views = []
    for name, x0, y0, k in out:
        x0 = max(0, min(w - vw * k, x0))
        y0 = max(0, min(h - vh * k, y0))
        views.append((name, x0, y0, min(w, x0 + vw * k), min(h, y0 + vh * k)))
    return views


def run(path, bands=None):
    rr.clear_reader_cache()
    rm.clear_sources()
    preview_budget, decode_budget = _app_constants()
    res = {"path": os.path.basename(path), "remote": rm.is_cogref(path), "steps": []}

    def snap():
        if not rm.is_cogref(path):
            return None
        return rm.open_pointer(path).stats()

    def record(name, t0, before, **extra):
        dt = time.perf_counter() - t0
        after = snap()
        row = {"view": name, "seconds": round(dt, 3), **extra}
        if after is not None:
            row["requests"] = after["requests"] - (before["requests"] if before else 0)
            row["bytes"] = after["bytes_fetched"] - (before["bytes_fetched"] if before else 0)
        res["steps"].append(row)
        return row

    t0 = time.perf_counter()
    prof = rr.probe(path)
    if prof is None:
        raise SystemExit("cannot open {}: {}".format(path, rr.last_probe_error(path)))
    reader = rr.open_reader(path, prof)
    record("open (header)", t0, None)
    res["profile"] = prof.describe()
    res["levels"] = [reader.level_shape(i) for i in range(len(reader._levels))]

    bands = bands or list(range(min(3, prof.count)))
    before = snap()
    t0 = time.perf_counter()
    budget = preview_budget(prof, len(bands))
    level = reader.level_for_bytes(len(bands), budget, decode_budget=decode_budget)
    step = reader.step_for_bytes(len(bands), budget, level)
    arr = reader.read_bands_full(bands, level=level, step=step)
    record("first preview", t0, before, level=level, step=step, shape=list(arr.shape))

    h, w, _ = reader.level_shape(0)
    for name, x0, y0, x1, y1 in _views(w, h):
        level, step, (lx0, ly0, lx1, ly1) = _zoom_tile_plan(reader, x0, y0, x1, y1)
        before = snap()
        t0 = time.perf_counter()
        arr = reader.read_window(lx0, ly0, lx1, ly1, bands=bands, level=level, step=step)
        lv = reader._levels[level]
        tiles = len(reader._spatial_tiles(lv, lx0, ly0, lx1, ly1))
        record(name, t0, before, level=level, step=step, tiles=tiles, shape=list(arr.shape))
    rr.clear_reader_cache()
    return res


def _table(res):
    lines = ["{} ({})".format(res["path"], "remote" if res["remote"] else "local"),
             "  " + res["profile"]]
    for r in res["steps"]:
        extra = ""
        if "level" in r:
            extra = "level {} step {}".format(r["level"], r["step"])
            if "tiles" in r:
                extra += ", {} tiles".format(r["tiles"])
        net = ""
        if "bytes" in r:
            net = "{:>4} req {:>8.2f} MB".format(r["requests"], r["bytes"] / 1e6)
        lines.append("  {:<20} {:>7.2f} s  {:<22} {}".format(r["view"], r["seconds"], net, extra))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pointer", required=True, help=".cogref pointer to the remote COG")
    ap.add_argument("--local", help="the same COG on disk, for comparison")
    ap.add_argument("--repeat", type=int, default=1, help="runs of each (fresh caches each run)")
    ap.add_argument("--json", help="write all results here")
    args = ap.parse_args(argv)

    results = []
    for _ in range(max(1, args.repeat)):
        results.append(run(args.pointer))
        print(_table(results[-1]))
        if args.local:
            results.append(run(args.local))
            print(_table(results[-1]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
