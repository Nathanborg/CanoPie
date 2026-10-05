"""
A band expression applied in the editor must STAY on screen (BUG-007, D1).

THE BUG THIS PINS (reported as "my boolean shows, then reverts"):

`edit_image_viewer` snapshots the `.ax` `viewer_stretch` BEFORE the editor
opens and writes it back after Apply ("protected keys"). For a viewer that was
showing an RGB composite, that snapshot is `display_mode "rgb"`. The refresh
right after Apply pins the viewer to the expression band -- but only in
memory (`refresh_single_viewer`, `viewer._auto_pinned_last_band`). Every later
load (`display_image_group`: navigation, refresh, reopen) resolves the stretch
from the `.ax` first, finds the restored "rgb", and builds the composite from
r/g/b -- the expression band is dropped from the display.

Measured on the real project (C:\\Nova pasta (5), b1>100): boolean on screen
right after Apply and after a zoom; RGB again after a refresh and after reopen;
the `.ax` kept `viewer_stretch` rgb 2/1/0 throughout.

The contract:
  * an Apply that sets or changes the expression stores a single-band view of
    the expression band in the `.ax`;
  * an Apply that removes the expression brings back the stretch the user had
    before it.
"""
import json
import os

import pytest
from PyQt5 import QtWidgets

from ..image_editor_dialog import ImageEditorDialog

pytestmark = [pytest.mark.viewer, pytest.mark.editor]

RGB_STRETCH = {"mode": "percentile", "low_p": 2.0, "high_p": 98.0,
               "display_mode": "rgb", "r_band": 0, "g_band": 1, "b_band": 2}


def _clear_caches(pt):
    for attr in ("_ax_json_cache", "_export_image_cache", "_pixmap_cache", "_ax_cache"):
        c = getattr(pt, attr, None)
        if hasattr(c, "clear"):
            c.clear()


def _read_ax(pt, fp):
    with open(pt._ax_path_for(fp), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def editor_case(synthetic_project, viewer_factory):
    from .fixtures_manifest import fixture_image_path

    pt = synthetic_project
    fp = fixture_image_path("rgb_8bit_untiled")
    axp = pt._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None
    with open(axp, "w", encoding="utf-8") as f:
        json.dump({"viewer_stretch": dict(RGB_STRETCH)}, f)
    _clear_caches(pt)

    viewer = viewer_factory()
    imgd = pt._imagedata_or_fallback(fp)
    assert imgd is not None and imgd.image is not None
    viewer.image_data = imgd
    rec = {"viewer": viewer, "image_data": imgd}
    pt.viewer_widgets = list(getattr(pt, "viewer_widgets", []) or []) + [rec]
    try:
        yield pt, viewer, fp
    finally:
        try:
            pt.viewer_widgets.remove(rec)
        except Exception:
            pass
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)
        _clear_caches(pt)


def _apply_in_editor(monkeypatch, pt, viewer, expression):
    """Open the REAL editor on `viewer`, type `expression` (or switch band math
    off when None), Apply All Changes."""
    def fake_exec(self):
        if expression is None:
            self.band_enabled_checkbox.setChecked(False)
        else:
            self.band_input.setText(expression)
            self.band_enabled_checkbox.setChecked(True)
            self.apply_band_expression()
        self.apply_all_changes()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(ImageEditorDialog, "exec_", fake_exec)
    pt.edit_image_viewer(viewer)
    _clear_caches(pt)


def test_applied_expression_is_the_stored_display(editor_case, monkeypatch):
    pt, viewer, fp = editor_case
    _apply_in_editor(monkeypatch, pt, viewer, "b1>100")

    ax = _read_ax(pt, fp)
    assert (ax.get("band_expression") or "").strip() == "b1>100", (
        f"precondition: the editor did not store the expression (.ax = {ax})")
    vs = ax.get("viewer_stretch") or {}
    C = viewer.image_data.image.shape[2]
    assert str(vs.get("display_mode")).lower() == "single" and vs.get("display_band") == C - 1, (
        "After Apply of a band expression the .ax still stores the pre-editor "
        f"display {vs.get('display_mode')!r} r/g/b={vs.get('r_band')}/{vs.get('g_band')}/"
        f"{vs.get('b_band')} -- edit_image_viewer wrote its snapshot back over it. Every "
        "later load (navigation, refresh, reopen) resolves this stretch and rebuilds the "
        "RGB composite, so the expression shows once and then reverts.")

    # The loader resolves the stretch from the .ax exactly like this.
    sp = pt._stretch_from_dict(vs)
    assert sp is not None and sp.display_mode == "single" and sp.display_band == C - 1


def test_removing_the_expression_restores_the_previous_stretch(editor_case, monkeypatch):
    pt, viewer, fp = editor_case
    _apply_in_editor(monkeypatch, pt, viewer, "b1>100")
    _apply_in_editor(monkeypatch, pt, viewer, None)

    ax = _read_ax(pt, fp)
    vs = ax.get("viewer_stretch") or {}
    assert str(vs.get("display_mode")).lower() == "rgb" and \
        (vs.get("r_band"), vs.get("g_band"), vs.get("b_band")) == (0, 1, 2), (
        "Switching band math off left the automatic expression view in place "
        f"({vs}); the user's own RGB stretch from before the expression must come back.")
    assert vs.get("low_p") == RGB_STRETCH["low_p"] and vs.get("high_p") == RGB_STRETCH["high_p"]
