"""
CSV column ordering must be canonical and identical on both export paths.

WHY THIS MATTERS (found from a real exported row a user was reading):

`ExportWorker` (the "run in background" path) wrote `sorted(keys_union)` --
plain ALPHABETICAL order -- while the foreground path wrote `fieldnames_raw`,
a hand-built logical order. So the same project produced two DIFFERENT column
layouts depending on a checkbox, and the background one interleaved unrelated
columns:

    ... Mean | Median | Object ID | Pixel Count | Project | Q25 | Q5 | Q50 ...
    ... Scene Mean | Scene Median | Scene Standard Deviation | Standard Deviation ...

Two things there invite a misread: `Q25` sorts before `Q5`, and `Standard
Deviation` ends up separated from `Mean`/`Median` by the three Scene columns.
A user comparing a value against the wrong header is then very easy -- and on a
row where a legitimate outlier had made Mean much larger than Median, the
natural first assumption was a column-alignment bug rather than the data.

`order_csv_columns` is now the single source of truth for both paths.
"""
import pytest

from ..project_tab import order_csv_columns

pytestmark = [pytest.mark.extraction, pytest.mark.io]


# Exactly the header from the reported CSV.
REPORTED = [
    "Band Name", "Centroid X", "Centroid Y", "Channel", "Counts", "FakePath",
    "File Name", "Label Band Index", "Lat", "Long", "Mean", "Median",
    "Object ID", "Pixel Count", "Project", "Q25", "Q5", "Q50", "Q75", "Q95",
    "Root Folder", "Root ID", "Scene Mean", "Scene Median",
    "Scene Standard Deviation", "Standard Deviation", "counts_0", "counts_1",
]


def test_no_column_is_lost_or_duplicated():
    """Ordering must be a pure permutation -- never drop or invent a column."""
    out = order_csv_columns(REPORTED)
    assert sorted(out) == sorted(REPORTED)
    assert len(out) == len(set(out)) == len(REPORTED)


def test_quantiles_are_in_numeric_not_alphabetical_order():
    """THE readability regression: alphabetically `Q25` precedes `Q5`."""
    out = order_csv_columns(REPORTED)
    qs = [c for c in out if c.startswith("Q")]
    assert qs == ["Q5", "Q25", "Q50", "Q75", "Q95"], qs


def test_statistics_are_grouped_together():
    """Mean/Median/Standard Deviation must be adjacent -- alphabetical order
    pushed `Standard Deviation` past the three Scene columns."""
    out = order_csv_columns(REPORTED)
    i_mean, i_med, i_std = (out.index("Mean"), out.index("Median"),
                            out.index("Standard Deviation"))
    assert i_med == i_mean + 1, out
    assert i_std == i_med + 1, out


def test_scene_columns_follow_the_polygon_statistics():
    """Scene* is context for the row's own statistics, so it belongs after
    them -- and never interleaved with them."""
    out = order_csv_columns(REPORTED)
    assert out.index("Scene Mean") > out.index("Standard Deviation")
    assert out.index("Scene Median") == out.index("Scene Mean") + 1
    assert out.index("Scene Standard Deviation") == out.index("Scene Median") + 1


def test_identity_columns_come_first():
    out = order_csv_columns(REPORTED)
    head = out[:5]
    assert head == ["Project", "Root ID", "Root Folder", "File Name", "Object ID"], head


def test_unknown_columns_keep_a_stable_alphabetical_tail():
    """EXIF tags / Class_* / formula names have no canonical position, but the
    header must not reorder between runs."""
    extra = REPORTED + ["ZZ_custom", "AA_custom", "NDVI"]
    a = order_csv_columns(extra)
    b = order_csv_columns(list(reversed(extra)))
    assert a == b, "column order is not deterministic w.r.t. input order"
    tail = [c for c in a if c in ("ZZ_custom", "AA_custom", "NDVI")]
    assert tail == sorted(tail), tail


def test_both_export_paths_use_the_same_ordering_function():
    """The foreground and background CSVs must not diverge again."""
    import ast
    import inspect
    import textwrap
    from ..project_tab import ExportWorker

    tree = ast.parse(textwrap.dedent(inspect.getsource(ExportWorker.run)))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "order_csv_columns" in called, (
        "ExportWorker.run must order its CSV columns with order_csv_columns, "
        "not sorted() -- otherwise the background export is alphabetical while "
        "the foreground one is logical")
