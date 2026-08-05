"""
QC regression tests for the "compute these indices" band-math parser
(AnalysisOptionsDialog._parse_bandmath) and the CSV row it produces.

THE BUG THIS PINS (reported against a real 0-3 categorical "leaf_age" band,
where `b7==0` means "fraction of pixels in class FLUSH"):

Typing a bare comparison such as `b7==0` into the indices box produced NO ROW
in the exported CSV, with no visible error. The parser split each line on the
FIRST `=`, so:

    "b7==0"        -> name "b7",        expression "=0"     <-- invalid Python
    "flush: b7==0" -> name "flush: b7", expression "=0"     <-- ':' never reached

`=0` then raised inside process_polygon's band-math evaluation, which catches
everything into a `logging.warning("Band-math polygon eval skipped ...")` --
so the row silently vanished rather than surfacing the failure. Only the
fully-qualified `flush=b7==0` form happened to work.

Equality must be written `==`; a single `=` keeps its documented `name=expr`
meaning and is NOT treated as a comparison.
"""
import numpy as np
import pytest

from ..machine_learning_manager import AnalysisOptionsDialog

parse = AnalysisOptionsDialog._parse_bandmath


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def test_bare_comparison_is_self_named():
    """THE regression: a bare comparison must survive intact and become its
    own column name, instead of being split into name 'b7' / expr '=0'."""
    assert parse("b7==0") == {"b7==0": "b7==0"}


@pytest.mark.parametrize("expr", ["b7==0", "b7!=0", "b7>=1", "b7<=2", "b7>1", "b7<3"])
def test_all_bare_comparison_operators_survive(expr):
    """Every comparison operator containing '=' was vulnerable to the naive
    split; '>' and '<' are included to prove they were never broken and stay
    that way."""
    assert parse(expr) == {expr: expr}, f"{expr} was mangled by the parser"


def test_named_comparison_with_colon():
    """The ':' form was unreachable for any expression containing '=', because
    the '=' branch was tested first and matched inside the expression."""
    assert parse("flush: b7==0") == {"flush": "b7==0"}


def test_named_comparison_with_equals():
    """The one form that always worked -- must keep working."""
    assert parse("flush=b7==0") == {"flush": "b7==0"}


def test_single_equals_keeps_name_expr_meaning():
    """A single '=' is the name/expression separator, NOT equality. `b7=0`
    therefore defines a column named 'b7' holding the constant 0. Equality
    must be spelled '=='."""
    assert parse("b7=0") == {"b7": "0"}


def test_arithmetic_index_unaffected():
    assert parse("ndvi=(b4-b1)/(b4+b1)") == {"ndvi": "(b4-b1)/(b4+b1)"}


def test_multiple_formulas_comma_separated():
    assert parse("a=b7==0, b=b7==1") == {"a": "b7==0", "b": "b7==1"}


def test_multiple_formulas_newline_separated():
    assert parse("a=b7==0\nb=b7==2") == {"a": "b7==0", "b": "b7==2"}


def test_json_form_still_supported():
    assert parse('{"flush": "b7==0"}') == {"flush": "b7==0"}


def test_json_with_trailing_comma_is_repaired():
    """SECOND REPORTED FAILURE, verbatim from the user's indices box.

    A trailing comma before `}` is what people actually type (and what most
    editors leave behind after deleting a line), but it is invalid JSON. The
    parser fell through to the line-based fallback, which then produced:
        {'{': '{', '"boolean1"': '"b1 >150"', '"boolean2"': '"b7 == 2"', '}': '}'}
    -- braces became bogus formulas, and the quotes survived so each
    "expression" was a Python STRING LITERAL rather than an expression. Those
    all raised inside process_polygon's band-math eval and were swallowed into
    a warning, so NO boolean row reached the CSV."""
    text = '{\n  "boolean1": "b1 >150","boolean2": "b7 == 2",\n}'
    assert parse(text) == {"boolean1": "b1 >150", "boolean2": "b7 == 2"}


def test_quotes_are_stripped_in_fallback_parser():
    """A quoted expression must not survive as a quoted string -- eval would
    return the literal text instead of a pixel array."""
    assert parse('"flush": "b7==0"') == {"flush": "b7==0"}


def test_stray_braces_are_not_formulas():
    assert parse("{\nb7==0\n}") == {"b7==0": "b7==0"}


@pytest.mark.parametrize("expr", [
    "where(b1>50,1,0)",
    "clip(b1,0,50)",
    "mean(b1,b2,b3)",
])
def test_multi_argument_functions_are_not_split_on_their_commas(expr):
    """Entries are comma-separated, but a comma INSIDE parentheses belongs to
    the function call. Splitting on every comma tore these apart -- and these
    are documented band-math functions, so this was broken for anyone using
    them regardless of the comparison bug."""
    assert parse(f"idx={expr}") == {"idx": expr}


def test_empty_input():
    assert parse("") == {}
    assert parse("   ") == {}


# ---------------------------------------------------------------------------
# End-to-end: the parsed formula must actually yield a CSV row whose Mean is
# the boolean average (fraction of matching pixels).
# ---------------------------------------------------------------------------
def test_bare_comparison_produces_a_row_with_boolean_average(synthetic_project):
    """What the user actually asked for: a row whose Mean is the fraction of
    pixels satisfying the comparison, computed independently from the
    ground-truth array."""
    from .fixtures_manifest import fixture_image_path, get_fixture
    from .generate_fixtures import _rasterize_polygon_mask
    from .project_builder import polygon_group_name
    from ._helpers import load_raw_npz, assert_close

    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    raw = load_raw_npz(name)
    poly_mask = _rasterize_polygon_mask(
        spec["polygon"]["points"], spec["height"], spec["width"])
    band0_vals = raw[:, :, 0][poly_mask]
    # Pick a threshold that genuinely splits the region, so the expected
    # fraction is strictly between 0 and 1 (a 0.0 or 1.0 result could pass
    # even if the expression were being ignored).
    threshold = float(np.median(band0_vals))
    expected_fraction = float((band0_vals < threshold).mean())
    assert 0.0 < expected_fraction < 1.0

    user_text = f"b1<{threshold:g}"
    formulas = parse(user_text)
    assert formulas == {user_text: user_text}, f"parser mangled {user_text!r}: {formulas}"

    opts = {"stats": {"mean": True},
            "band_math": {"enabled": True, "formulas": formulas}}
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=opts)

    row = next((r for r in rows if r.get("Channel") == user_text), None)
    assert row is not None, (
        f"no CSV row for {user_text!r} -- rows present: "
        f"{[r.get('Channel') for r in rows]}")
    assert_close(row["Mean"], expected_fraction, tol=1e-6,
                 msg=f"{user_text} boolean average")
    assert row.get("Band Name") == user_text, f"expected Band Name to be {user_text!r}, got {row.get('Band Name')!r}"


def test_band_math_expression_in_band_name_row(synthetic_project):
    """Verifies that the math expression appears in the 'Band Name' field of CSV
    rows for both named formulas (e.g. 'boolean1': 'b1 > 150') and bare formulas."""
    from .fixtures_manifest import fixture_image_path, get_fixture
    from .project_builder import polygon_group_name

    name = "multiband_8band_ancillary"
    spec = get_fixture(name)
    fp = fixture_image_path(name)
    group = polygon_group_name(name, spec["polygon"]["name"])
    poly_dict = synthetic_project.all_polygons[group][fp]

    formulas = {
        "boolean1": "b1 > 150",
        "GCC": "b2 / (b1 + b2 + b3)",
        "b1 > 100": "b1 > 100"
    }

    opts = {
        "stats": {"mean": True},
        "band_math": {"enabled": True, "formulas": formulas}
    }
    rows, _ = synthetic_project.process_polygon(
        group, fp, poly_dict, {}, [], False, opts=opts
    )

    by_channel = {r.get("Channel"): r for r in rows if isinstance(r, dict)}

    assert "boolean1" in by_channel
    assert by_channel["boolean1"].get("Band Name") == "b1 > 150"

    assert "GCC" in by_channel
    assert by_channel["GCC"].get("Band Name") == "b2 / (b1 + b2 + b3)"

    assert "b1 > 100" in by_channel
    assert by_channel["b1 > 100"].get("Band Name") == "b1 > 100"

