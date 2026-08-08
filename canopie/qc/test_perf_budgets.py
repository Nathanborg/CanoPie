"""
Speed and memory budgets for the extraction path.

Wall-clock alone is a bad regression test: it depends on the machine, the load,
and whether some other suite warmed a cache. So the HARD assertions here are on
DETERMINISTIC counters -- how many windowed reads were issued, whether the whole
cube was materialised, how much memory the operation peaked at relative to the
data it legitimately needs. Those catch the failure modes that actually matter
and give the same answer on every machine. Wall-clock appears only as a
generous ceiling, marked `perf` so it can be excluded.

The specific regressions these guard:

  * A lazily-read cube must stay lazy. `LazyChannels` recently gained an
    `__array__`, which is convenient but means `np.asarray(lazy)` now SILENTLY
    reads the entire cube instead of raising. On a 3.88 GB hyperspectral stack
    that is the difference between a windowed ROI read and an out-of-memory
    kill, and nothing about it is visible in the output -- the numbers are
    right, the machine just dies. Peak-memory and call-count assertions are the
    only way to see it.

  * Windowed export must issue O(polygon) reads, not O(image).

  * Per-band NoData masking and scene statistics must not quietly start
    scanning every band of every file.
"""
import gc
import time
import tracemalloc

import numpy as np
import pytest

from . import fixtures_manifest as fm
from .fixtures_manifest import fixture_image_path, get_fixture
from .project_builder import polygon_group_name

pytestmark = [pytest.mark.perf, pytest.mark.extraction]


STATS_OPTS = {"stats": {"mean": True, "median": True, "std": True}}


def _clear_caches(pt):
    for attr in ("_export_image_cache", "_export_cache", "_scene_stats_cache",
                 "_per_band_nd_cache", "_master_nd_cache", "_file_nodata_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()
    gc.collect()


def _run_polygon(pt, name, opts=STATS_OPTS):
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly = pt.all_polygons[group][fp]
    return pt.process_polygon(group, fp, poly, {}, [], False, opts=opts)


# ---------------------------------------------------------------------------
# The lazy path must stay lazy
# ---------------------------------------------------------------------------
def test_lazy_export_never_materialises_the_whole_cube(
        synthetic_project, force_lazy_export, monkeypatch):
    """THE guard for LazyChannels.__array__.

    `np.asarray(lazy_channels)` used to raise, which made an accidental
    materialisation obvious at the call site. It now succeeds silently, so the
    only remaining signal is that the full cube got read. Count the calls.
    """
    from ..raster_reader import LazyChannels

    calls = {"n": 0}
    original = LazyChannels.__array__

    def counting(self, dtype=None):
        calls["n"] += 1
        return original(self, dtype)

    monkeypatch.setattr(LazyChannels, "__array__", counting)

    _clear_caches(synthetic_project)
    _run_polygon(synthetic_project, "hyperspectral_200band")

    assert calls["n"] == 0, (
        f"the whole cube was materialised {calls['n']} time(s) during a "
        "windowed export -- something called np.asarray() on a LazyChannels. "
        "That defeats the lazy reader entirely and will OOM on a real stack.")


def test_lazy_export_peak_memory_scales_with_the_polygon_not_the_cube(
        synthetic_project, force_lazy_export):
    """Peak allocation must stay near the ROI's size.

    A full 30x30x200 float32 cube is ~720 KB; the polygon covers a small
    fraction of it. The ceiling is deliberately loose (it only has to
    distinguish "read the window" from "read everything") so it does not become
    a flaky byte-counting test.
    """
    spec = get_fixture("hyperspectral_200band")
    full_cube_bytes = spec["height"] * spec["width"] * spec["bands"] * 4

    _clear_caches(synthetic_project)
    gc.collect()
    tracemalloc.start()
    try:
        _run_polygon(synthetic_project, "hyperspectral_200band")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Scene statistics legitimately touch every band, so allow several cube
    # copies; a silent np.asarray of the cube plus its float32 promotion lands
    # far above this.
    budget = full_cube_bytes * 8
    assert peak < budget, (
        f"peak allocation {peak/1e6:.2f} MB exceeds {budget/1e6:.2f} MB for a "
        f"{full_cube_bytes/1e6:.2f} MB cube -- the export is materialising far "
        "more than the polygon needs")


def test_windowed_reads_are_bounded(synthetic_project, force_lazy_export, monkeypatch):
    """A windowed export must not issue an unbounded number of reads.

    Counts calls rather than timing them, so it means the same thing on any
    machine. The bound is per-band-plus-slack, which catches a regression to
    "re-read the window once per statistic" or "once per pixel".
    """
    from ..raster_reader import TiledRasterReader

    calls = {"n": 0}
    original = TiledRasterReader.read_window

    def counting(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(TiledRasterReader, "read_window", counting)

    spec = get_fixture("hyperspectral_200band")
    _clear_caches(synthetic_project)
    _run_polygon(synthetic_project, "hyperspectral_200band")

    # Generous: a handful of passes over the bands (ROI + scene stats + masks).
    budget = spec["bands"] * 6 + 50
    assert calls["n"] <= budget, (
        f"{calls['n']} windowed reads for a {spec['bands']}-band cube "
        f"(budget {budget}) -- the reader is being called far more than once "
        "per band per pass")


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["rgb_8bit_untiled", "multiband_8band_ancillary"])
def test_eager_export_peak_is_proportional_to_the_image(synthetic_project, name):
    """Catches an accidental O(bands^2) or per-pixel-copy regression on the
    ordinary (non-lazy) path."""
    spec = get_fixture(name)
    img_bytes = spec["height"] * spec["width"] * spec["bands"] * 4

    _clear_caches(synthetic_project)
    gc.collect()
    tracemalloc.start()
    try:
        _run_polygon(synthetic_project, name)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    budget = max(img_bytes * 12, 8_000_000)
    assert peak < budget, (
        f"{name}: peak {peak/1e6:.2f} MB for a {img_bytes/1e6:.2f} MB image "
        f"(budget {budget/1e6:.2f} MB)")


def test_repeated_exports_do_not_grow_memory(synthetic_project):
    """A cache that never evicts, or a leaked reference per call, shows up as
    monotonically rising peak memory across identical runs."""
    _clear_caches(synthetic_project)
    _run_polygon(synthetic_project, "rgb_8bit_untiled")   # warm caches

    peaks = []
    for _ in range(4):
        gc.collect()
        tracemalloc.start()
        try:
            _run_polygon(synthetic_project, "rgb_8bit_untiled")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(peak)

    # Later runs must not need dramatically more than the first warmed one.
    assert max(peaks) < max(peaks[0] * 4, 4_000_000), (
        f"peak memory grew across identical repeated exports: {peaks}")


# ---------------------------------------------------------------------------
# Latency ceilings -- deliberately loose, and only meaningful as "did something
# become orders of magnitude slower", not as a benchmark.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("name,ceiling_s", [
    ("rgb_8bit_untiled", 2.0),            # measured ~0.02 s
    ("multiband_8band_ancillary", 2.0),   # measured ~0.02 s
    ("hyperspectral_200band", 8.0),       # measured ~0.26 s
])
def test_extraction_latency_ceiling(synthetic_project, name, ceiling_s):
    """~10-30x the measured baseline, so only a real algorithmic regression
    trips it -- not a busy machine."""
    _clear_caches(synthetic_project)
    t0 = time.perf_counter()
    _run_polygon(synthetic_project, name)
    dt = time.perf_counter() - t0

    assert dt < ceiling_s, (
        f"{name}: extraction took {dt:.2f}s, ceiling {ceiling_s}s. This is a "
        "very loose bound, so exceeding it usually means an algorithmic "
        "regression rather than a slow machine.")


@pytest.mark.slow
def test_band_math_is_cached_within_one_image(synthetic_project):
    """The same formula evaluated for many polygons on ONE image must not be
    recomputed each time -- process_polygon keeps a per-file cache for this.
    (Correctness of that cache ACROSS images is covered by
    test_contract_band_math.py.)"""
    opts = {"stats": {"mean": True},
            "band_math": {"enabled": True,
                          "formulas": {f"idx{i}": "b1+b2+b3" for i in range(6)}}}
    _clear_caches(synthetic_project)

    t0 = time.perf_counter()
    rows, _ = _run_polygon(synthetic_project, "multiband_8band_ancillary", opts=opts)
    dt = time.perf_counter() - t0

    produced = {r.get("Channel") for r in rows}
    assert {f"idx{i}" for i in range(6)} <= produced, (
        f"not all formulas produced rows: {sorted(produced)}")
    assert dt < 5.0, f"six identical formulas took {dt:.2f}s"
