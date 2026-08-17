"""QC for Image Editor defects found against C:\\PCA_climate_soil_panama.

Three reported symptoms, three clusters of cause:

* "viewer changes channel order and stops matching the .ax files"
  -> A3: "apply to group" / "apply to all groups" wrote the EDITED image's
     whole `.ax` -- including its `viewer_stretch` (display_mode, display_band,
     r_band/g_band/b_band, min/max), its `orig_size` and its `classification` --
     onto every other image, so each sidecar stopped describing its own raster.
  -> A2: the editor read `ax["stretch"]` while every writer writes
     `ax["viewer_stretch"]`, so it never saw the stored absolute bounds and
     always fell back to auto/percentile.

* "band expression does not correctly append band"
  -> B1: every evaluation error was swallowed and the UNMODIFIED image
     returned, which is indistinguishable from "ran and did nothing".
  -> B2: on a raster over `_EDITOR_PREVIEW_BYTES` the editor holds only the 3
     display bands; the old `need > loaded` gate let `b1+b2` evaluate against
     that preview, where the C==3 BGR remap swaps b1<->b3 -- so `b1` meant a
     different channel than at export time.
  -> B4: the band scan was lowercase-only, so `B4` found no bands, skipped both
     the load and the range check, and failed into B1's silent swallow.

* "slow to open"
  -> S3: the raw-pixel cache was dropped after every Apply even though the
     editor only writes `.ax`, so the next open re-read the raster. On this
     project the rasters sit on a Google Drive mount where a single open costs
     ~1-1.9s and `_load_raw_image` opens three times.
"""
import inspect
import re

import pytest

from canopie import image_editor_dialog as ied
from canopie.image_editor_dialog import ImageEditorDialog
from canopie.project_tab import ProjectTab

pytestmark = [pytest.mark.io]


# ---------------------------------------------------------------------------
# A3 -- per-image state must not be broadcast to other images
# ---------------------------------------------------------------------------
def test_per_image_ax_keys_are_defined():
    assert hasattr(ied, "_PER_IMAGE_AX_KEYS")
    # viewer_stretch is the one that actually reorders channels; orig_size and
    # classification are equally image-specific.
    for k in ("viewer_stretch", "orig_size", "classification"):
        assert k in ied._PER_IMAGE_AX_KEYS, f"{k} may be copied between images"


def test_apply_to_other_files_strips_per_image_state():
    """THE regression test for the reported channel-order corruption.

    `_prepare_and_write` is a closure inside save_modifications_to_file, so this
    asserts on its source: every file that is NOT the one open in the editor
    must have the per-image keys filtered out before the write.
    """
    src = inspect.getsource(ImageEditorDialog.save_modifications_to_file)

    assert "_PER_IMAGE_AX_KEYS" in src, (
        "save_modifications_to_file writes self.modifications unfiltered, so "
        "'apply to group'/'apply to all' stamps the edited image's "
        "viewer_stretch/orig_size/classification onto every other .ax")

    # The filter must be conditional on NOT being the edited image -- a blanket
    # strip would stop the edited image keeping its own display state.
    filt = src[src.index("_PER_IMAGE_AX_KEYS"):]
    head = src[:src.index("_PER_IMAGE_AX_KEYS")]
    assert "image_filepath" in head[-1200:] or "_is_edited" in head[-1200:], (
        "the per-image-key filter is not gated on 'is this the edited image?'")
    assert "not _is_edited" in head[-1200:] or "_is_edited" in filt[:200], (
        "filter does not distinguish the edited image from the other targets")


# ---------------------------------------------------------------------------
# A2 -- the editor must read the key that is actually written
# ---------------------------------------------------------------------------
def test_editor_reads_viewer_stretch_not_only_legacy_stretch():
    src = inspect.getsource(ied)
    assert 'ax.get("viewer_stretch")' in src, (
        'the editor reads only ax["stretch"]; every writer writes '
        '"viewer_stretch", so the stored absolute min/max were never applied')
    # Legacy spelling must stay as a fallback so old projects still load.
    assert 'ax.get("stretch")' in src, "legacy 'stretch' fallback was dropped"


# ---------------------------------------------------------------------------
# B1 -- a failed expression must not masquerade as success
# ---------------------------------------------------------------------------
def test_failed_band_expression_is_recorded_not_swallowed():
    src = inspect.getsource(ImageEditorDialog.process_band_expression)
    assert "_last_band_expr_error" in src, (
        "process_band_expression returns the unmodified image on error with no "
        "record of the failure -- indistinguishable from a successful no-op")


def test_band_expression_error_is_cleared_per_run():
    """A stale error from an earlier expression must not be reported against a
    later, successful one."""
    src = inspect.getsource(ImageEditorDialog.process_band_expression)
    reset = src.index("_last_band_expr_error = None")
    first_try = src.index("try:")
    assert reset < first_try, "the error flag is not reset before evaluation"


def test_failed_expression_still_returns_pixels_unchanged():
    """Behavioural: a bad expression must be non-destructive."""
    import numpy as np

    tab = ImageEditorDialog.__new__(ImageEditorDialog)
    img = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    original = img.copy()

    out = ImageEditorDialog.process_band_expression(tab, img, "b1 +* b2")

    assert np.array_equal(out, original), "a failed expression mutated the image"
    assert getattr(tab, "_last_band_expr_error", None), (
        "a failed expression left no error record")


# ---------------------------------------------------------------------------
# B2 / B4 -- band references
# ---------------------------------------------------------------------------
def test_band_scan_is_case_insensitive():
    src = inspect.getsource(ImageEditorDialog.apply_band_expression)
    m = re.search(r"re\.findall\(r'([^']+)'", src)
    assert m, "could not find the band-reference scan"
    pattern = m.group(1)
    assert re.findall(pattern, "B4 + b2") == ["4", "2"], (
        f"scan {pattern!r} misses uppercase B -- 'B4' silently skips both the "
        "band-range check and the full-stack load")


def test_expression_always_loads_all_bands():
    """THE regression test for b1 addressing the wrong channel on big rasters.

    The old gate (`need > loaded`) let an expression run against the 3-band
    preview, where the C==3 BGR remap swaps b1<->b3.
    """
    src = inspect.getsource(ImageEditorDialog.apply_band_expression)
    # Comments explain the old gate on purpose -- judge the CODE only.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))

    assert "ensure_all_bands_loaded()" in code
    assert "need > loaded" not in code, (
        "apply_band_expression still short-circuits the full-stack load, so on "
        "a raster above _EDITOR_PREVIEW_BYTES the expression evaluates against "
        "the 3-band preview and b1 means the wrong channel")
    # And the load must not be hidden behind any band-count comparison.
    assert not re.search(r"if\s+need\s*[<>=]", code), (
        "the full-stack load is still conditional on the referenced band index")


# ---------------------------------------------------------------------------
# S3 -- do not throw away raw pixels the editor never touched
# ---------------------------------------------------------------------------
def test_apply_does_not_drop_the_raw_pixel_cache():
    src = inspect.getsource(ProjectTab.edit_image_viewer)
    assert "_raw_cache.pop(" not in src, (
        "edit_image_viewer drops _raw_cache after Apply. The editor writes only "
        "the .ax sidecar, so the raw pixels cannot have changed -- and on a "
        "network-backed project the next open pays a full re-read (~3s here)")
