"""Regression guards for the export path (process_polygon / CSV export) and
for .ax sidecar durability.

Every test here pins a bug that was diagnosed and fixed against a real
project. They are grouped by the symptom the user actually reported, because
that is what makes a regression recognisable when it comes back.


1. ".ax edits silently disappear"  -- ATOMIC WRITES
   Both writers (`ProjectTab._write_json_safely`, despite its name, and
   `ImageEditorDialog._write_ax`) did a plain `open(path, "w")`, which
   TRUNCATES the file to zero bytes before writing a single byte of new
   content. Any interruption in that window leaves a truncated/empty .ax --
   and EVERY reader in the codebase swallows a parse failure and substitutes
   `{}` (`_read_json_silently`; the `except: existing = {}` in `_write_ax`).
   So a half-written .ax never raises: it silently discards every edit made
   to that image, and the next save persists that empty state as the truth.
   `_merge_write_ax`'s docstring claimed "re-write atomically" the whole time.

2. "the same image is exported twice"  -- PATH-SPELLING DEDUPE
   `all_polygons[group]` is keyed by the literal filepath STRING, and one
   project routinely holds two spellings of the same file (the viewer's
   forward-slash form vs `os.path.normpath`'s backslash form from shapefile
   import -- see test_polygon_path_key_shadowing.py for the original
   incident). Rendering already handles this with a two-tier exact+normalised
   lookup, but the CSV export's job-list builder iterated the raw dict, so a
   polygon stored under both spellings was processed and exported TWICE:
   identical "File Name" (basename hides the spelling), identical Object ID,
   every channel row duplicated.

3. "Class NPV % / Class flush % are blank"  -- SILENT CLASSIFICATION FAILURE
   process_polygon catches every RF/classification prediction failure
   internally and only logging.error()s it, so it never reaches
   ExportWorker's `errors` counter (that only counts exceptions escaping
   process_polygon entirely -- deliberately, so one bad polygon cannot abort
   a whole export). Result: a CSV whose "Class X %" headers exist (built
   upfront from model.classes_) but whose cells are all blank, while the
   completion dialog still says "(0 errors)".

4. "other roots aren't classified until I refresh"  -- CACHE COMPLETENESS
   `invalidate_caches_for_file` is the canonical "drop every per-file cache"
   function, but it missed `_export_image_cache` -- the one its own sibling
   call sites call the "biggest memory user".
"""
import ast
import functools
import inspect
import json
import os
import textwrap

import numpy as np
import pytest

from ..image_editor_dialog import ImageEditorDialog
from ..project_tab import ProjectTab
from ..raster_reader import ByteBudgetedLRUDict, estimate_cache_entry_bytes

pytestmark = [pytest.mark.extraction, pytest.mark.io]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _module_tree(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read())


def _func_tree(func):
    """AST of `func`, located from its MODULE rather than via
    textwrap.dedent(inspect.getsource(func)).

    dedent is not usable here: project_tab.py contains column-0 comments
    *inside* indented method bodies (e.g. the `# fp_key must be the first
    element ...` line inside process_polygon), which makes the common
    leading-whitespace prefix an empty string. dedent then strips nothing,
    the `def` stays indented, and ast.parse raises IndentationError. Walking
    the module tree by __qualname__ sidesteps that entirely.
    """
    tree = _module_tree(inspect.getsourcefile(func))
    node = tree
    for part in func.__qualname__.split("."):
        if part == "<locals>":
            raise LookupError(f"{func.__qualname__} is a closure; not addressable")
        nxt = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and child.name == part:
                nxt = child
                break
        if nxt is None:
            raise LookupError(f"could not locate {func.__qualname__} in the module AST")
        node = nxt
    return node


def _calls_in(func, name):
    """True when `func`'s body contains a LIVE call to `name`.

    Deliberately AST-based, not a substring test: the bug being pinned in
    `on_reset_image` was a call that had been COMMENTED OUT, which any
    `"..." in src` check passes happily. Only a real Call node counts.
    """
    for node in ast.walk(_func_tree(func)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == name:
            return True
        if isinstance(f, ast.Name) and f.id == name:
            return True
    return False


def _names_in(func):
    """Every attribute/name/string-literal identifier in `func`'s body, from
    the AST -- so identifiers that appear ONLY inside comments are NOT
    counted (a comment mentioning a cache name must not satisfy a test that
    the cache is actually cleared)."""
    out = set()
    for node in ast.walk(_func_tree(func)):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
        elif isinstance(node, ast.keyword) and node.arg:
            # Keyword-argument NAMES are ast.keyword.arg, not Name nodes, so
            # they are invisible to the branches above. Without this, a test
            # asserting that a caller passes `src_channel_order=...` fails even
            # though the call is right there.
            out.add(node.arg)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


class _StubTab:
    """A ProjectTab stand-in exposing only what a unit under test touches.

    Explicit stub rather than MagicMock on purpose: a MagicMock satisfies
    every `hasattr`/attribute access, which silently neuters exactly the
    `if hasattr(...)` branches these tests exist to exercise.
    """

    def __init__(self):
        self.calls = []
        self.project_folder = ""

    def invalidate_caches_for_file(self, fp):
        self.calls.append(("invalidate_caches_for_file", fp))

    def get_current_root_name(self):
        self.calls.append(("get_current_root_name", None))
        return "Root1"

    def refresh_viewer(self, root_name=None):
        self.calls.append(("refresh_viewer", root_name))


# ===========================================================================
# 1. .ax durability -- atomic writes
# ===========================================================================

def _bare_editor(project_folder):
    """An ImageEditorDialog shell for testing _write_ax without Qt init.

    ImageEditorDialog.__new__ skips QDialog.__init__, so attributes must be
    planted directly in __dict__ (reading an absent one raises RuntimeError
    from sip rather than AttributeError).
    """
    ed = ImageEditorDialog.__new__(ImageEditorDialog)
    ed.__dict__["project_folder"] = str(project_folder)
    return ed


def test_write_json_safely_is_atomic(tmp_path):
    """THE regression: a write must never leave the target truncated.

    Asserted structurally (os.replace, not a bare open-for-write on the
    destination) because the failure mode is a crash mid-write, which cannot
    be provoked deterministically from a unit test.
    """
    src = textwrap.dedent(inspect.getsource(ProjectTab._write_json_safely))
    assert "os.replace" in _names_in(ProjectTab._write_json_safely) or "replace" in _names_in(
        ProjectTab._write_json_safely), (
        "_write_json_safely no longer writes via os.replace(); a plain "
        "open(path,'w') truncates the .ax to zero bytes before writing, and "
        "every reader silently turns a truncated .ax into {} -- i.e. total, "
        "silent loss of the user's edits for that image")

    tree = ast.parse(src)
    opened_targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                if "w" in str(node.args[1].value):
                    opened_targets.append(node.args[0])
    assert opened_targets, "expected at least one open(...,'w') for the temp file"
    for tgt in opened_targets:
        assert not (isinstance(tgt, ast.Name) and tgt.id == "path"), (
            "_write_json_safely opens the DESTINATION path for writing -- that "
            "truncates it in place; it must write a temp file and os.replace()")


def test_write_json_safely_round_trips_and_leaves_no_temp(tmp_path):
    """Functional half: the atomic path must still actually write, must
    overwrite cleanly, and must not litter .tmp files next to the .ax."""
    pt = ProjectTab.__new__(ProjectTab)
    target = tmp_path / "IMG_0001.ax"

    pt._write_json_safely(str(target), {"crop_rect": {"x": 1, "y": 2}})
    assert json.loads(target.read_text(encoding="utf-8"))["crop_rect"]["x"] == 1

    pt._write_json_safely(str(target), {"crop_rect": {"x": 9, "y": 9}})
    assert json.loads(target.read_text(encoding="utf-8"))["crop_rect"]["x"] == 9

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files were left behind: {leftovers}"


def test_write_ax_is_atomic_and_merges(tmp_path):
    """_write_ax must merge into the existing .ax AND replace it atomically.

    The merge behaviour is what makes the truncation bug so damaging: the
    function READS the existing file first, so a previously-truncated .ax is
    read as {} and the merge silently starts from nothing.
    """
    ed = _bare_editor(tmp_path)
    img = str(tmp_path / "IMG_0001.tif")

    ed._write_ax(img, {"rotate": 90}, quiet=True)
    ax_path = tmp_path / "IMG_0001.ax"
    assert json.loads(ax_path.read_text(encoding="utf-8")) == {"rotate": 90}

    # merge, not replace
    ed._write_ax(img, {"crop_rect": {"x": 5}}, quiet=True)
    data = json.loads(ax_path.read_text(encoding="utf-8"))
    assert data["rotate"] == 90, "existing keys must survive a merge"
    assert data["crop_rect"]["x"] == 5

    # None means delete
    ed._write_ax(img, {"rotate": None}, quiet=True)
    data = json.loads(ax_path.read_text(encoding="utf-8"))
    assert "rotate" not in data, "a None value must delete the key"
    assert data["crop_rect"]["x"] == 5, "unrelated keys must not be dropped"

    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    src = textwrap.dedent(inspect.getsource(ImageEditorDialog._write_ax))
    assert "os.replace" in src or "replace" in _names_in(ImageEditorDialog._write_ax), (
        "_write_ax must os.replace() a temp file, never truncate the .ax in place")


def test_write_ax_never_leaves_a_partial_file_when_serialisation_fails(tmp_path):
    """If the payload cannot be serialised, the PREVIOUS .ax must survive
    intact -- the whole point of writing to a temp file first."""
    ed = _bare_editor(tmp_path)
    img = str(tmp_path / "IMG_0002.tif")
    ax_path = tmp_path / "IMG_0002.ax"

    ed._write_ax(img, {"rotate": 180}, quiet=True)
    good = ax_path.read_text(encoding="utf-8")

    class _Unserialisable:
        pass

    # _write_ax swallows its own exceptions (by design -- a failed .ax write
    # must not abort a batch), so this must not raise here either.
    ed._write_ax(img, {"bad": _Unserialisable()}, quiet=True)

    assert ax_path.read_text(encoding="utf-8") == good, (
        "a failed write corrupted/truncated the existing .ax instead of "
        "leaving it untouched")
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == [], (
        "a failed write left its temp file behind")


def test_write_ax_serialises_concurrent_writers(tmp_path):
    """_ax_path_for keys on project_folder + BASENAME, so two images with the
    same filename in different folders map to the SAME .ax -- and
    save_modifications_to_file fans _prepare_and_write out over a 32-worker
    pool. Without a lock, two read-modify-write cycles interleave and one
    merge is silently lost."""
    import threading

    ed = _bare_editor(tmp_path)
    img = str(tmp_path / "SHARED.tif")
    ed._write_ax(img, {"seed": True}, quiet=True)

    n = 24
    barrier = threading.Barrier(n)

    def writer(i):
        barrier.wait()
        ed._write_ax(img, {f"k{i}": i}, quiet=True)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads((tmp_path / "SHARED.ax").read_text(encoding="utf-8"))
    missing = [f"k{i}" for i in range(n) if f"k{i}" not in data]
    assert not missing, (
        f"{len(missing)} concurrent merges were lost ({missing[:5]}...) -- "
        "_write_ax's read-modify-write is not serialised")
    assert data.get("seed") is True


# ===========================================================================
# 2. duplicate export rows -- path-spelling dedupe
# ===========================================================================

def test_two_spellings_of_one_path_collapse_to_one_dedupe_key():
    """Establishes the premise the exporter's dedupe relies on: the two
    spellings that legitimately coexist in all_polygons are DIFFERENT dict
    keys but must normalise to ONE key."""
    raw = os.path.join("C:" + os.sep, "imgs", ".", "IMG_0020.tif")
    norm = os.path.normpath(raw)
    assert raw != norm, "fixture no longer produces two distinct spellings"

    k1 = os.path.normpath(raw).lower()
    k2 = os.path.normpath(norm).lower()
    assert k1 == k2, "normalisation must collapse the two spellings"

    # a naive dict keyed on the literal strings holds BOTH -> the duplication
    naive = {raw: 1, norm: 1}
    assert len(naive) == 2, (
        "the premise of the duplicate-rows bug: all_polygons holds the same "
        "image under two keys")


def test_csv_export_job_list_dedupes_by_normalised_path():
    """THE regression: save_polygons_to_csv's job-list builder must not queue
    the same (group, image) twice just because all_polygons holds two
    spellings of the image's path."""
    tree = _func_tree(ProjectTab.save_polygons_to_csv)

    # Find the statement that appends to polygons_to_process.
    appends = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute)
               and n.func.attr == "append"
               and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "polygons_to_process"]
    assert appends, "polygons_to_process.append(...) not found -- test needs updating"

    names = _names_in(ProjectTab.save_polygons_to_csv)
    assert "normpath" in names, (
        "the CSV export job list is built without normalising the filepath, so "
        "an image stored under two path spellings is exported twice (duplicate "
        "rows for the same File Name / Object ID)")

    # The dedupe must be a real guard (a set membership test), not just a call
    # to normpath somewhere unrelated.
    has_seen_set = any(
        isinstance(n, ast.Compare)
        and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops)
        for n in ast.walk(tree))
    assert has_seen_set, (
        "no membership guard found near the job-list build -- the dedupe must "
        "skip a (group, normalised path) that was already queued")


# ===========================================================================
# 3. silent classification failure must be surfaced
# ===========================================================================

def test_process_polygon_records_rf_prediction_errors():
    """Both classification failure handlers must report, not just log.

    Without this the CSV ships blank "Class X %" cells while the export's
    completion dialog reports "(0 errors)", which is what made the original
    report so hard to diagnose.
    """
    assert hasattr(ProjectTab, "_record_rf_prediction_error"), (
        "the helper that makes classification failures visible is gone")

    assert _calls_in(ProjectTab.process_polygon, "_record_rf_prediction_error"), (
        "process_polygon swallows RF/classification prediction failures without "
        "recording them, so every 'Class X %' cell can be blank while the "
        "export still reports zero errors")

    # Both handlers -- the point-mode batch predict and the area-mode
    # per-polygon predict -- must record, not just one of them.
    recording_handlers = 0
    for node in ast.walk(_func_tree(ProjectTab.process_polygon)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "_record_rf_prediction_error"):
                recording_handlers += 1
                break
    assert recording_handlers >= 2, (
        f"only {recording_handlers} classification except-handler(s) record the "
        "failure; both the point-mode and area-mode predict paths must")


def test_record_rf_prediction_error_accumulates_and_caps_samples():
    """The recorder must count every failure but keep only a bounded number of
    sample messages (an export can have thousands of polygons)."""
    pt = ProjectTab.__new__(ProjectTab)
    # ProjectTab.__new__ skips QWidget.__init__, so reading an attribute that
    # is not already in __dict__ raises RuntimeError from sip -- which even
    # getattr(..., default) cannot absorb. _record_rf_prediction_error probes
    # exactly those, and swallows its own exceptions by design, so without
    # seeding them here it would silently no-op and the test would assert
    # nothing. (Same reason test_polygon_path_key_shadowing seeds _bare_tab.)
    pt._rf_export_error_count = 0
    pt._rf_export_error_samples = []

    for i in range(50):
        pt._record_rf_prediction_error(f"C:/imgs/IMG_{i}.tif", f"grp{i}", ValueError(f"boom{i}"))

    assert pt._rf_export_error_count == 50, "every failure must be counted"
    assert len(pt._rf_export_error_samples) <= 5, (
        "sample messages must be capped so a big export cannot balloon memory")
    assert "boom0" in pt._rf_export_error_samples[0], (
        "the sample must carry the underlying exception text -- that string is "
        "the whole diagnostic value")


def test_export_worker_surfaces_classification_failures():
    """ExportWorker.run must fold the recorded count into its completion
    message; otherwise the failures stay invisible to the user."""
    from ..project_tab import ExportWorker

    names = _names_in(ExportWorker.run)
    assert "_rf_export_error_count" in names, (
        "ExportWorker.run does not read the classification-failure count, so a "
        "run where every classification failed still reports '(0 errors)'")


# ===========================================================================
# 4. cache-invalidation completeness
# ===========================================================================

# Every per-file cache invalidate_caches_for_file is responsible for.
# _export_image_cache is the one that was missing -- and it is the largest.
_PER_FILE_CACHES = (
    "_export_cache",
    "_scene_stats_cache",
    "_raw_cache",
    "_nodata_mask_by_filepath",
    "_ax_json_cache",
    "_imgdata_cache",
    "_pixmap_cache",
    "_raw_image_cache",
    "_export_image_cache",
)


def test_invalidate_caches_for_file_covers_every_per_file_cache():
    """THE regression: this function is the canonical 'drop everything cached
    for this file' call, and it silently skipped _export_image_cache -- the
    cache its own sibling call sites label the biggest memory user. A stale
    entry there is why other roots kept rendering pre-classification pixels
    until the user manually refreshed."""
    names = _names_in(ProjectTab.invalidate_caches_for_file)
    missing = [c for c in _PER_FILE_CACHES if c not in names]
    assert not missing, (
        f"invalidate_caches_for_file does not clear {missing} -- stale entries "
        "there survive an edit and are served as if fresh")


def test_bulk_scope_cache_clear_covers_every_per_file_cache():
    """The 'apply to group / all roots' path clears caches wholesale rather
    than per-file; it had the identical omission."""
    names = _names_in(ProjectTab.edit_image_viewer)
    missing = [c for c in _PER_FILE_CACHES if c not in names]
    assert not missing, (
        f"the group/all-scope cache clear in edit_image_viewer misses {missing}")


# ===========================================================================
# 5. Reset Image must tell the viewer to re-read
# ===========================================================================

def test_on_reset_image_refreshes_the_viewer():
    """THE regression, and it was a single commented-out line.

    on_reset_image deletes the .ax and invalidates the caches, but was the
    only one of the three reset variants that never told ProjectTab to
    re-acquire the file -- so the viewer kept displaying (and holding) stale,
    possibly full-resolution pixels, and never re-entered the lazy COG path.
    Checked via AST because the bug WAS a comment: a substring test passes on
    the broken code.
    """
    assert _calls_in(ImageEditorDialog.on_reset_image, "_auto_refresh_after_reset"), (
        "on_reset_image does not call _auto_refresh_after_reset -- after a "
        "single-image reset the main viewer never re-reads the file")


def test_all_three_reset_variants_refresh_the_viewer():
    """The single-image variant is the one that regressed, but all three must
    behave the same way."""
    for fn in (ImageEditorDialog.on_reset_image,
               ImageEditorDialog.on_reset_group,
               ImageEditorDialog.on_reset_all_groups):
        assert _calls_in(fn, "_auto_refresh_after_reset"), (
            f"{fn.__name__} does not refresh the viewer after resetting")


def test_reset_image_verifies_the_ax_was_actually_deleted():
    """A locked/permission-denied .ax silently kept its windowing-blocking
    keys alive, so the file stayed on the slow full-read path while the UI
    claimed it had been reset."""
    names = _names_in(ImageEditorDialog.on_reset_image)
    assert "exists" in names, (
        "on_reset_image does not re-check os.path.exists after deleting the "
        ".ax candidates, so a failed delete stays silent")


# ===========================================================================
# 6. classification must reach every target .ax
# ===========================================================================

def test_classification_flag_is_broadcast_to_group_and_all_targets():
    """`classification` lives in _PER_IMAGE_AX_KEYS (a blanket guard against
    copying per-image state), but the persisted block is only
    {"mode","enabled","label_names"} -- an instruction to run the shared model
    against THIS image's own bands, never a pixel array. Stripping it meant
    'apply to all roots' silently disabled classification everywhere except
    the image open in the editor, so it must be explicitly re-added."""
    from .. import image_editor_dialog as ied

    assert "classification" in ied._PER_IMAGE_AX_KEYS, (
        "the blanket per-image guard was weakened; the documented corruption "
        "incident (viewer_stretch/orig_size copied between images) relies on it")

    tree = _func_tree(ImageEditorDialog.save_modifications_to_file)

    # After the per-image strip, the classification key must be put back.
    readds = [n for n in ast.walk(tree)
              if isinstance(n, ast.Subscript)
              and isinstance(n.slice, ast.Constant)
              and n.slice.value == "classification"
              and isinstance(n.ctx, ast.Store)]
    assert readds, (
        "_prepare_and_write strips 'classification' with the other per-image "
        "keys and never re-adds it, so applying to a group/all roots leaves "
        "classification enabled ONLY on the edited image")

    # And the re-add must be a MEMBERSHIP test, not a truthiness test:
    # disabling classification is expressed as {"classification": None}, which
    # is what _write_ax turns into a key delete. A truthy guard broadcasts the
    # enable but drops the disable, so "apply to all roots" could switch
    # classification on everywhere and never off again.
    membership_tests = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Compare)
        and any(isinstance(op, ast.In) for op in n.ops)
        and isinstance(n.left, ast.Constant)
        and n.left.value == "classification"
    ]
    assert membership_tests, (
        "the classification re-add is not guarded by a "
        "`'classification' in base_mods` membership test. A truthiness guard "
        "silently drops the DISABLE signal -- disabling classification is "
        "expressed as {'classification': None}, which is falsy -- so "
        "'apply to all roots' could switch classification on everywhere and "
        "then never be able to switch it back off")


def test_write_ax_none_value_deletes_the_key_entirely(tmp_path):
    """`None` means DELETE, and the key must not survive the merge.

    The delete list was computed correctly and the key WAS popped -- and then
    `existing.update(modifications)` put it straight back as `"key": null`,
    because the None-valued entries were never stripped from `modifications`.
    Readers that test truthiness (`ax.get(k) or {}`) tolerated that by luck;
    readers that test membership (`if "hist_match" in mods`) did not, and the
    nulls accumulated in the file permanently.
    """
    ed = _bare_editor(tmp_path)
    img = str(tmp_path / "IMG_0003.tif")
    ax_path = tmp_path / "IMG_0003.ax"

    ed._write_ax(img, {"classification": {"mode": "sklearn", "enabled": True},
                       "rotate": 90}, quiet=True)
    assert "classification" in json.loads(ax_path.read_text(encoding="utf-8"))

    ed._write_ax(img, {"classification": None}, quiet=True)
    data = json.loads(ax_path.read_text(encoding="utf-8"))
    assert "classification" not in data, (
        f"None must remove the key, but the .ax still holds "
        f"{data.get('classification')!r} for it")
    assert data["rotate"] == 90, "unrelated keys must be untouched"

    # explicit _delete control key must behave the same and not leak itself
    ed._write_ax(img, {"_delete": ["rotate"]}, quiet=True)
    data = json.loads(ax_path.read_text(encoding="utf-8"))
    assert "rotate" not in data
    assert "_delete" not in data, "the control key must never be persisted"


# ===========================================================================
# 6b. the two .ax replay paths must honour the same enable flags
# ===========================================================================

#: Every op in the .ax carries a separate "is it switched on" flag, written by
#: the editor's _collect_modifications. Both replay paths must consult ALL of
#: them or the viewer and the CSV silently disagree about the pixels.
_AX_ENABLE_FLAGS = (
    "hist_enabled",
    "band_enabled",
    "crop_enabled",
    "resize_enabled",
    "rotate_enabled",
    "nodata_enabled",
)


def _string_constants(func):
    return {n.value for n in ast.walk(_func_tree(func))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@pytest.mark.contract
def test_both_ax_replay_paths_honour_every_enable_flag():
    """CONTRACT: `_apply_ax_to_raw` (export/CSV) and `apply_aux_modifications`
    (viewer) are two independent replays of the same .ax. If one consults an
    enable flag and the other does not, switching an op OFF changes the
    picture but not the exported numbers (or vice versa) -- a disagreement
    that is invisible until someone cross-checks a CSV against the screen.

    This is the failure mode that already bit `_ax_is_windowable`, whose
    original version looked for an "enabled" key INSIDE each op's dict --
    which nothing writes -- so turning histogram matching off left the
    hist_match block behind and every large raster silently kept taking the
    slow whole-image path.
    """
    export_side = _string_constants(ProjectTab._apply_ax_to_raw)
    viewer_side = _string_constants(ProjectTab.apply_aux_modifications)

    mismatched = [f for f in _AX_ENABLE_FLAGS
                  if (f in export_side) != (f in viewer_side)]
    assert not mismatched, (
        f"the export and viewer .ax replays disagree about {mismatched}: one "
        "honours the flag and the other ignores it, so toggling that op "
        "changes the viewer and the CSV differently")

    missing_both = [f for f in _AX_ENABLE_FLAGS
                    if f not in export_side and f not in viewer_side]
    assert not missing_both, (
        f"neither replay path consults {missing_both} -- switching those ops "
        "off in the editor would have no effect on either the viewer or the CSV")


# ===========================================================================
# 6c. histogram matching must not silently swap R and B
# ===========================================================================
#
# THE INCIDENT (reproduced on a real project, C:\New Folder215):
# an .ax whose hist_match reference_path pointed at the SOURCE IMAGE ITSELF --
# i.e. the image was being matched against its own statistics, which must be
# an identity no-op -- came back with bands 0 and 2 exchanged:
#
#     RAW  band0 mean= 91.895   ->  after hist match  52.787
#     RAW  band1 mean=110.038   ->                   110.037
#     RAW  band2 mean= 53.003   ->                    91.898
#
# Matching a quantile curve per band against the stored reference proved the
# stored reference was in BGR order while the .ax labelled it "rgb"
# (pairing {0:2, 1:1, 2:0}, error 0.19/0.34/0.20 on the correct pairing vs
# 39/58/19 otherwise). Root cause: the label was derived from
# `self.image_data_obj` -- an attribute never assigned anywhere -- and then
# from the PARENT's `_last_export_channel_order`, a stale global describing a
# previous, unrelated export. The export side does check for a mismatch, but a
# WRONG label makes that check agree and stay silent, so every exported
# statistic was computed on swapped channels with no warning at all.


def test_hist_match_against_its_own_statistics_is_identity():
    """THE invariant: matching an image to ITS OWN statistics is a no-op.

    This is what makes the channel-order bug self-evident -- no reference
    image, no ground truth and no tolerance argument is needed. If band i's
    reference is band i's own mean/std, band i must come back unchanged; if
    the reference is paired to the wrong band, the means visibly move.
    """
    rng = np.random.default_rng(0)
    h = w = 64
    # Deliberately DISTINCT per-band distributions -- with identical bands a
    # channel swap would be undetectable and the test would pass vacuously.
    img = np.stack([
        np.clip(rng.normal(90, 50, (h, w)), 0, 255),
        np.clip(rng.normal(110, 55, (h, w)), 0, 255),
        np.clip(rng.normal(53, 40, (h, w)), 0, 255),
    ], axis=-1).astype(np.uint8)

    means = [float(img[:, :, c].mean()) for c in range(3)]
    stds = [float(img[:, :, c].std()) for c in range(3)]
    assert max(means) - min(means) > 20, "bands must differ for this test to bite"

    hm = {"mode": "meanstd", "bands": 3,
          "ref_stats": [{"mean": means[c], "std": stds[c]} for c in range(3)],
          "channel_order": "rgb"}

    out = ProjectTab._apply_hist_match(img, {"hist_match": hm},
                                       fast=True, nodata_values=[])

    for c in range(3):
        got = float(out[:, :, c].mean())
        assert abs(got - means[c]) < 2.0, (
            f"band {c} was matched to its OWN statistics (target mean "
            f"{means[c]:.2f}) but came back {got:.2f}. A large move here means "
            f"the reference is being paired with the wrong band -- the classic "
            f"R/B swap. All band means: {[round(float(out[:, :, i].mean()), 2) for i in range(3)]}")


def test_hist_match_detects_a_reversed_reference():
    """Guard on the guard: prove the identity test above would actually FAIL
    if the reference were paired in reverse, so it cannot pass vacuously."""
    rng = np.random.default_rng(0)
    h = w = 64
    img = np.stack([
        np.clip(rng.normal(90, 50, (h, w)), 0, 255),
        np.clip(rng.normal(110, 55, (h, w)), 0, 255),
        np.clip(rng.normal(53, 40, (h, w)), 0, 255),
    ], axis=-1).astype(np.uint8)

    means = [float(img[:, :, c].mean()) for c in range(3)]
    stds = [float(img[:, :, c].std()) for c in range(3)]

    # REVERSED reference -- what a BGR-measured reference does to RGB pixels.
    hm = {"mode": "meanstd", "bands": 3,
          "ref_stats": [{"mean": means[c], "std": stds[c]} for c in (2, 1, 0)]}
    out = ProjectTab._apply_hist_match(img, {"hist_match": hm},
                                       fast=True, nodata_values=[])

    assert abs(float(out[:, :, 0].mean()) - means[0]) > 20, (
        "a reversed reference did NOT move band 0 -- the identity test above "
        "would therefore pass even with the channel-order bug present")


def test_hist_reference_channel_order_is_measured_not_inherited():
    """The recorded channel_order must describe the array the reference was
    measured from, never a phantom attribute or the parent's leftover export
    state."""
    live = _names_in(ImageEditorDialog.on_hist_match_clicked)

    assert "image_data_obj" not in live, (
        "on_hist_match_clicked still reads self.image_data_obj, which is never "
        "assigned anywhere in the class -- so the lookup always fails and the "
        "channel_order label silently comes from somewhere else")
    assert "_last_export_channel_order" not in live, (
        "the hist reference's channel_order is taken from the PARENT's "
        "_last_export_channel_order -- a stale global describing whatever "
        "export ran last, not the image these statistics were measured from. "
        "A wrong label is worse than none: it makes the export-side mismatch "
        "check agree and stay silent while R and B are paired in reverse")
    assert "_editor_channel_order" in live, (
        "the label must come from the order the editor's own loader recorded")


def test_realignment_lives_in_the_one_canonical_hist_routine():
    """When the reference's recorded order disagrees with the pixels', the
    reference must be REALIGNED -- and that must happen in
    ProjectTab._apply_hist_match, not in any single caller.

    The original code only warned, reasoning that permuting a stored reference
    "would change numbers a previous export already used". Those numbers were
    computed with R and B transposed; reproducing a channel swap is not a
    virtue, and a log line nobody reads is not informed consent.

    Placing it in the canonical routine matters just as much as doing it at
    all: the CSV export (_apply_ax_to_raw), the viewer (apply_aux_modifications)
    and ML training (MachineLearningManager._apply_hist_local) all funnel
    through this one function. Fixing only the export -- which a first attempt
    at this did -- leaves the viewer and the model trained on swapped
    channels while the CSV is correct, which is worse than a uniform bug
    because the three then disagree.
    """
    live = _names_in(ProjectTab._apply_hist_match)
    assert "channel_order" in live, (
        "_apply_hist_match does not consult the reference's recorded "
        "channel_order, so a BGR reference on RGB pixels swaps R and B")
    assert "src_channel_order" in live, (
        "_apply_hist_match cannot be told which order its pixels are in, so it "
        "has no way to detect a mismatch")

    reverses = [n for n in ast.walk(_func_tree(ProjectTab._apply_hist_match))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "reversed"]
    assert reverses, (
        "no reversal of the reference bands happens on a channel-order "
        "mismatch -- the code only warns, so every consumer stays computed on "
        "swapped channels")


def test_every_hist_consumer_passes_its_channel_order_provenance():
    """Knowing HOW to realign is useless if no caller says which order its
    pixels are in. All three replay paths must pass their provenance."""
    from ..machine_learning_manager import MachineLearningManager

    assert "src_channel_order" in _names_in(ProjectTab._apply_ax_to_raw), (
        "the CSV export path does not tell _apply_hist_match which channel "
        "order its pixels are in")
    assert "src_channel_order" in _names_in(ProjectTab.apply_aux_modifications), (
        "the viewer replay path does not pass its channel-order provenance")

    ml_src = inspect.getsource(MachineLearningManager._apply_ax_ops_for_ml)         if hasattr(MachineLearningManager, "_apply_ax_ops_for_ml") else ""
    if not ml_src:
        ml_src = inspect.getsource(MachineLearningManager)
    assert "src_channel_order" in ml_src, (
        "ML training/prediction does not pass its channel-order provenance, so "
        "a model can be trained on Red/Blue-transposed features while the CSV "
        "export of the same .ax is correct")


def test_reference_realignment_only_touches_the_first_three_bands():
    """RGB<->BGR is a convention for the first three bands only; a 15-band
    stack's ancillary planes must not be permuted."""
    def realign(seq):
        n = min(3, len(seq))
        return list(reversed(seq[:n])) + list(seq[n:])

    assert realign(["R", "G", "B"]) == ["B", "G", "R"]
    assert realign(["R", "G", "B", "b4", "b5"]) == ["B", "G", "R", "b4", "b5"], (
        "bands beyond the third are not part of the RGB/BGR convention and "
        "must survive realignment untouched")
    # and it must be an involution -- applying it twice restores the original
    assert realign(realign(["R", "G", "B", "b4"])) == ["R", "G", "B", "b4"]


def test_editor_loader_records_the_channel_order_it_produced():
    """_load_raw_image returns pixels from tifffile (native/RGB) or cv2 (BGR);
    whichever ran must be recorded, because the hist reference is measured
    from exactly that array."""
    for fn in (ImageEditorDialog._load_raw_image,
               ImageEditorDialog._load_editor_preview):
        assert "_editor_channel_order" in _names_in(fn), (
            f"{fn.__name__} does not record which channel order it returned, so "
            "the hist_match reference cannot be labelled correctly")


# ===========================================================================
# 7. byte-budgeted caches
# ===========================================================================

def test_byte_budgeted_cache_is_a_real_dict_subclass():
    """Non-negotiable: call sites all over the codebase (and the QC helpers)
    gate cache clears behind `isinstance(cache, dict)`. A UserDict-based
    implementation fails that check, turning those clears into silent no-ops
    -- which is exactly how a first attempt at this fix produced stale
    cached export images and broke unrelated tests."""
    c = ByteBudgetedLRUDict(1024)
    assert isinstance(c, dict), (
        "ByteBudgetedLRUDict must subclass dict/OrderedDict -- isinstance("
        "cache, dict) guards real cache-clearing code paths")


def test_byte_budgeted_cache_evicts_on_bytes_not_count():
    c = ByteBudgetedLRUDict(1000)
    c["a"] = np.zeros(400, dtype=np.uint8)
    c["b"] = np.zeros(400, dtype=np.uint8)
    assert c.nbytes == 800
    c["c"] = np.zeros(400, dtype=np.uint8)     # 1200 > 1000 -> evict oldest
    assert c.nbytes <= 1000
    assert "a" not in c and "b" in c and "c" in c


def test_byte_budgeted_cache_is_true_lru_on_read():
    c = ByteBudgetedLRUDict(1000)
    c["a"] = np.zeros(400, dtype=np.uint8)
    c["b"] = np.zeros(400, dtype=np.uint8)
    _ = c["a"]                                  # touch 'a' -> 'b' is now oldest
    c["c"] = np.zeros(400, dtype=np.uint8)
    assert "a" in c, "a read must refresh recency (plain FIFO would drop 'a')"
    assert "b" not in c


def test_byte_budgeted_cache_keeps_byte_count_accurate_through_dict_ops():
    """pop/del/clear must all keep the running total correct, or the budget
    drifts and eviction stops working."""
    c = ByteBudgetedLRUDict(10_000)
    c["a"] = np.zeros(100, dtype=np.uint8)
    c["b"] = np.zeros(100, dtype=np.uint8)
    assert c.nbytes == 200

    c.pop("a", None)
    assert c.nbytes == 100, "pop() did not decrement the byte total"

    del c["b"]
    assert c.nbytes == 0, "__delitem__ did not decrement the byte total"

    c["x"] = np.zeros(100, dtype=np.uint8)
    c.clear()
    assert c.nbytes == 0 and len(c) == 0, "clear() did not reset the byte total"


def test_estimate_cache_entry_bytes_handles_every_cached_shape():
    """The four converted caches store different shapes; mis-sizing any of
    them silently disables the budget for that cache."""
    arr = np.zeros(100, dtype=np.float64)       # 800 bytes

    assert estimate_cache_entry_bytes(arr) == 800

    class _Lite:
        def __init__(self, image):
            self.image = image

    assert estimate_cache_entry_bytes(_Lite(arr)) == 800, (
        "ImageData/_Lite entries are sized via their .image array")

    # _export_image_cache's real value shape
    tup = (arr, {"H": 1, "W": 1, "C": 1}, None, [], "fp.tif", "rgb")
    assert estimate_cache_entry_bytes(tup) == 800

    # nothing array-like -> must still be non-zero, never silently free
    assert estimate_cache_entry_bytes("just a string") > 0


# ===========================================================================
# 8. oversized non-windowable stacks must not be fully materialised
# ===========================================================================

def test_imagedata_or_fallback_refuses_an_oversized_stack_read():
    """The lazy reader exists precisely to avoid multi-GB whole-cube reads,
    but the 'TIFF preflight' fallback called tifffile's whole-array read with
    no size check at all whenever probe() could not classify the file as
    windowable -- reintroducing the exact failure raster_reader.py was written
    to prevent. The MemoryError must also NOT fall through to section 2's
    equally unbounded cv2 read."""
    tree = _func_tree(ProjectTab._imagedata_or_fallback)

    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_tifffile_read_as_HWC"]
    assert reads, "_tifffile_read_as_HWC call not found -- test needs updating"

    gated = [n for n in reads
             if any(kw.arg == "max_bytes" for kw in n.keywords)]
    assert len(gated) == len(reads), (
        f"{len(reads) - len(gated)} of {len(reads)} _tifffile_read_as_HWC call "
        "site(s) have no max_bytes gate -- an oversized non-windowable stack "
        "is read fully into RAM there")

    reraises = [h for h in ast.walk(tree)
                if isinstance(h, ast.ExceptHandler)
                and isinstance(h.type, ast.Name)
                and h.type.id == "MemoryError"]
    assert reraises, (
        "the oversized-stack MemoryError is not caught/re-raised distinctly, so "
        "it falls through to the ImageData/cv2 branch which reads the same "
        "oversized file fully into RAM anyway")


# ===========================================================================
# 9. ML training must see the same pixels the .ax describes
# ===========================================================================

def test_ml_training_uses_the_shared_export_replay():
    """ML training/prediction must read pixels through the SAME RAW+.ax replay
    as process_polygon, not a private copy.

    A second, independently-written replay is how ML once ran a different CDF
    algorithm and silently discarded NoData -- exported numbers then depended
    on which code path happened to run. The delegation is the invariant.
    """
    from ..machine_learning_manager import MachineLearningManager

    live = _names_in(MachineLearningManager._get_export_image)
    assert "_get_export_image" in live, (
        "ML manager no longer delegates to ProjectTab._get_export_image, so "
        "training can be fed different pixels than the CSV export reports")
    assert "parent_tab" in live, (
        "the delegation must go through the owning ProjectTab")


def test_ml_hist_match_uses_the_canonical_implementation():
    """ML must not carry its own histogram-matching algorithm."""
    from ..machine_learning_manager import MachineLearningManager

    live = _names_in(MachineLearningManager._apply_hist_match)
    assert "_apply_hist_match" in live and "ProjectTab" in str(
        inspect.getsource(MachineLearningManager._apply_hist_match)), (
        "ML manager does not delegate histogram matching to "
        "ProjectTab._apply_hist_match -- a second algorithm means exported and "
        "trained-on pixels can diverge for the same .ax")


# ===========================================================================
# 10. appended bands must return the viewer to the original composite
# ===========================================================================

def test_auto_single_band_pin_is_marked_so_it_can_be_undone():
    """Appending an ML result turns it into an ordinary stacked band and
    switches classification/expression OFF, so the viewer must go back to the
    ORIGINAL composite.

    It did not: the earlier expression/classification pinned the viewer to
    display_mode='single', display_band=C-1, and the branch that sets that pin
    is guarded by `elif has_expr` -- so once the expression was gone it neither
    re-ran nor undid itself. The stale pin survived (with C now changed by the
    appended band) and auto-stretch kept showing a single band.

    The pin must therefore be MARKED as automatic, so it can be distinguished
    from a single-band view the user deliberately chose.
    """
    # These are the two places that AUTOMATICALLY pin the viewer to the last
    # band (handle_image_loaded on load, refresh_single_viewer on refresh).
    for fn in (ProjectTab.handle_image_loaded, ProjectTab.refresh_single_viewer):
        marks = [n for n in ast.walk(_func_tree(fn))
                 if isinstance(n, ast.Attribute)
                 and n.attr == "_auto_pinned_last_band"
                 and isinstance(n.ctx, ast.Store)]
        assert marks, (
            f"{fn.__name__} pins the viewer to a single band without marking the "
            "pin as automatic, so it can never be safely undone")


def test_editor_drops_a_stale_auto_pin_when_cls_and_expr_are_off():
    """THE regression: with classification and band expression both off, an
    automatic single-band pin must be released."""
    live = _names_in(ProjectTab.edit_image_viewer)
    assert "_auto_pinned_last_band" in live, (
        "edit_image_viewer never releases the automatic single-band pin, so "
        "after appending a band the viewer stays locked on one band instead of "
        "returning to the original composite")


def test_a_user_chosen_stretch_is_never_auto_dropped():
    """Guard on the fix: releasing the pin must not clobber a single-band view
    the user selected themselves."""
    # Resolving a stored/deliberate stretch must clear the automatic marker.
    src = inspect.getsource(ProjectTab.display_image_group)
    assert "_auto_pinned_last_band = False" in src, (
        "display_image_group applies a resolved stretch without clearing the "
        "automatic-pin marker, so a single-band view the user deliberately "
        "chose would later be auto-dropped as if the app had set it")
