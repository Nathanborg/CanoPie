"""The foreground CSV writer must not silently drop row data.

Found by comparing two real exports of the same project
(C:\\New Folder210\\exports): identical row counts (40,801 each) but
15.9 MB vs 14.1 MB. The difference was entirely in the COLUMNS.

Two independent defects, opposite in direction:

1. MISSING COLUMNS -- `Lat`, `Long` and `Label Band Index` are written into
   the row dicts by process_polygon and are part of the canonical column order
   (`_CSV_COLUMN_ORDER_HINT`), but the foreground writer's STATIC
   `fieldnames_raw` omitted them. Its DictWriter uses
   `extrasaction="ignore"`, so those keys were dropped without a warning --
   a foreground export lost its GPS coordinates while a background export of
   the same project kept them.

2. LEAKED PRIVATE COLUMN -- `_valid_count` is process_polygon's internal
   per-channel pixel count, normally popped by the caller. Two call sites did
   not pop it, so the key survived into the row dict; the background writer
   derives its columns from the row keys, so it shipped a literal
   `_valid_count` column to users.

Both are pinned here, in both directions, so the two writers cannot drift
apart again.
"""
import inspect
import re

import pytest

from canopie import project_tab as pt_mod
from canopie.project_tab import ProjectTab

pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _save_csv_src():
    return inspect.getsource(ProjectTab.save_polygons_to_csv)


def _process_polygon_src():
    return inspect.getsource(ProjectTab.process_polygon)


# ---------------------------------------------------------------------------
# 1. Columns the rows carry must survive the static header
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("column", ["Lat", "Long", "Label Band Index"])
def test_row_keys_are_present_in_the_static_header(column):
    """THE regression test for the silently-dropped GPS columns.

    process_polygon emits these keys; the foreground header must list them or
    extrasaction="ignore" discards them without a word.
    """
    src = _save_csv_src()
    head = src[:src.index("# Conditionally add Scene Stats columns")]
    # Judge the CODE, not the prose: the comment above fieldnames_raw names
    # these columns on purpose, and matching it would make this test pass even
    # with the entries removed from the list.
    head = "\n".join(
        line for line in head.splitlines() if not line.lstrip().startswith("#"))

    assert f"'{column}'" in head, (
        f"{column!r} is written into rows by process_polygon but is missing "
        f"from the foreground static fieldnames, so DictWriter("
        f"extrasaction='ignore') drops it silently")


def test_emitted_row_keys_are_a_subset_of_the_canonical_order_hint():
    """Anything a row can carry should have a defined position, otherwise the
    two writers order/keep columns differently."""
    hint = set(pt_mod._CSV_COLUMN_ORDER_HINT)
    for column in ("Lat", "Long", "Label Band Index", "Centroid X", "Pixel Count"):
        assert column in hint, f"{column!r} has no canonical position"


# ---------------------------------------------------------------------------
# 2. Private bookkeeping keys must never become columns
# ---------------------------------------------------------------------------
def test_every_calc_stats_call_site_strips_the_private_count():
    """`_valid_count` is popped by the caller by convention. A site that forgets
    ships it to users as a column -- which is exactly what happened."""
    src = _process_polygon_src()
    # Exclude the definition itself -- only real invocations count.
    calls = len(re.findall(r"(?<!def )_calc_stats\(", src))
    pops = len(re.findall(r"pop\('_valid_count'", src))

    assert pops >= calls, (
        f"{calls} _calc_stats() call sites but only {pops} strip "
        f"'_valid_count' -- the unstripped ones leak a private column into "
        f"any export whose columns come from the row keys")


def test_export_worker_never_emits_underscore_prefixed_columns():
    """Defence in depth: even if a call site forgets, the writer that derives
    columns from row keys must not publish private ones."""
    src = inspect.getsource(pt_mod.ExportWorker.run)
    assert 'startswith("_")' in src, (
        "the background writer builds its column set from raw row keys with no "
        "filter, so any internal key that escapes becomes a user-visible column")


def test_underscore_filter_actually_removes_the_key():
    """Behavioural check of the filter's logic, independent of where it lives."""
    row = {"Mean": 1.0, "_valid_count": 42, "Lat": -9.0}
    cleaned = {k: v for k, v in row.items()
               if not (isinstance(k, str) and k.startswith("_"))}
    assert "_valid_count" not in cleaned
    assert cleaned == {"Mean": 1.0, "Lat": -9.0}, "the filter dropped real data"
