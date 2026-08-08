"""
The zoom tile must use the CURRENT .ax and the CURRENT image data.

THE BUG THIS PINS (reported as "on cropping the original image is retrieved on
zooming, and the same for band math"):

`display_image_group` installed the refinement callback like this:

    def _highres_cb(v, scene_rect, req_id=None, _imgd=imgd, _ax=_ax_for_hr):
        self._request_highres_viewport_region(v, _imgd, scene_rect, ax=_ax)

Both the image data and the `.ax` are DEFAULT ARGUMENTS, i.e. bound once when
the group is displayed. Nothing re-runs `display_image_group` after an edit --
the post-editor path is `refresh_single_viewer`, which rebinds
`viewer.image_data` to a NEW object and never touches the callback.

So the moment the user applied a crop or a band expression, the tile was still
computed from the PRE-EDIT image data and the PRE-EDIT (usually empty) .ax, and
faithfully painted the original, uncropped, un-computed raster on top of the
edited preview. The geometry was self-consistent -- for the OLD image -- which
is why it looked like a sharp, correct picture of the wrong thing rather than
like a glitch.

This was latent before per-window replay landed: crop used to produce a black
tile and band math disabled the overlay outright, so the staleness never got a
chance to show. Fixing those exposed it.

THE FIX: `_highres_viewport_callback` resolves both at REQUEST time --
`viewer.image_data` for the data (kept current by refresh_single_viewer) and
`_get_cached_ax` for the .ax (keyed on the sidecar's mtime+size, so the
editor's write is picked up for free).
"""
import ast
import inspect
import textwrap
import types

import numpy as np
import pytest

from ..project_tab import ProjectTab

pytestmark = [pytest.mark.viewer, pytest.mark.contract]


def test_callback_does_not_capture_image_data_or_ax():
    """A static guard on the exact construct that caused this.

    Late-binding is the property under test, and a default argument silently
    destroys it -- so assert the installed callback captures neither.
    """
    src = textwrap.dedent(inspect.getsource(ProjectTab.display_image_group))
    tree = ast.parse(src)

    cb = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_highres_cb":
            cb = node
    assert cb is not None, "the high-res callback is no longer named _highres_cb"

    defaults = {a.arg for a in cb.args.args[-len(cb.args.defaults):]} if cb.args.defaults else set()
    leaked = defaults & {"_imgd", "_ax", "_image_data", "_ax_for_hr"}
    assert not leaked, (
        f"_highres_cb captures {sorted(leaked)} as default argument(s), freezing "
        "them at display time. refresh_single_viewer rebinds viewer.image_data "
        "after every edit, so a captured value renders the PRE-EDIT image -- "
        "the original raster painted over the cropped preview.")


def _lite(fp, image, reader=object()):
    return types.SimpleNamespace(filepath=fp, image=image, reader=reader,
                                 profile=None, preview_bands=[0, 1, 2],
                                 full_shape=(96, 96, 3), display_scale=1.0)


def test_request_uses_the_ax_on_disk_now_not_the_one_from_display_time(
        synthetic_project, monkeypatch, tmp_path):
    """THE regression: write a crop AFTER the callback is installed, and require
    the request to see it."""
    import json
    import os

    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    axp = synthetic_project._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None

    seen = {}

    def _spy(viewer, image_data, scene_rect, request_id=None, ax=None):
        seen["ax"] = ax
        seen["imgd"] = image_data

    monkeypatch.setattr(synthetic_project, "_request_highres_viewport_region", _spy)

    stale = _lite(fp, np.zeros((96, 96, 3), np.float32))
    fresh = _lite(fp, np.zeros((40, 40, 3), np.float32))    # post-crop preview
    viewer = types.SimpleNamespace(image_data=fresh,
                                   cancel_pending_highres_requests=lambda: None,
                                   _clear_highres_overlay=lambda: None)

    try:
        crop = {"crop_enabled": True,
                "crop_rect": {"x": 8, "y": 8, "width": 40, "height": 40},
                "crop_rect_ref_size": {"w": 96, "h": 96}}
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(crop, f)

        # `stale` stands in for what display_image_group captured.
        synthetic_project._highres_viewport_callback(
            viewer, None, request_id=1, fallback_image_data=stale)

        assert "ax" in seen, "no request was made"
        assert seen["ax"].get("crop_enabled") and seen["ax"].get("crop_rect"), (
            f"the request used a stale .ax ({seen['ax']!r}) -- with no crop in "
            "it the tile reads the ORIGINAL raster and paints it over the "
            "cropped preview")
        assert seen["imgd"] is fresh, (
            "the request used the image data captured at display time, whose "
            "preview shape still describes the UNCROPPED image")
    finally:
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)


def test_band_expression_written_after_display_is_picked_up(
        synthetic_project, monkeypatch):
    """Same staleness, the other reported symptom."""
    import json
    import os

    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    axp = synthetic_project._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None

    seen = {}
    monkeypatch.setattr(
        synthetic_project, "_request_highres_viewport_region",
        lambda v, i, r, request_id=None, ax=None: seen.update(ax=ax))

    imgd = _lite(fp, np.zeros((96, 96, 3), np.float32))
    viewer = types.SimpleNamespace(image_data=imgd,
                                   cancel_pending_highres_requests=lambda: None,
                                   _clear_highres_overlay=lambda: None)
    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"band_enabled": True, "band_expression": "(b2-b1)/(b2+b1)"}, f)

        synthetic_project._highres_viewport_callback(viewer, None, request_id=1)

        assert seen.get("ax", {}).get("band_expression") == "(b2-b1)/(b2+b1)", (
            "the request used a stale .ax -- the tile would show RAW bands "
            "over a preview displaying the computed index")
    finally:
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)


def test_turning_hist_match_on_drops_the_tile_without_a_redisplay(
        synthetic_project, monkeypatch):
    """The gate is re-evaluated per request, in BOTH directions.

    Deciding tileability once at display time also meant that switching
    histogram matching OFF left the overlay disabled until the group was
    re-displayed, and switching it ON left a stale sharp tile on screen.
    """
    import json
    import os

    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    axp = synthetic_project._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None

    calls = {"requests": 0, "cleared": 0}
    monkeypatch.setattr(
        synthetic_project, "_request_highres_viewport_region",
        lambda *a, **k: calls.update(requests=calls["requests"] + 1))

    imgd = _lite(fp, np.zeros((96, 96, 3), np.float32))
    viewer = types.SimpleNamespace(
        image_data=imgd,
        cancel_pending_highres_requests=lambda: None,
        _clear_highres_overlay=lambda: calls.update(cleared=calls["cleared"] + 1))

    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"hist_enabled": True,
                       "hist_match": {"mode": "meanstd", "bands": 3,
                                      "ref_stats": [{"mean": 1.0, "std": 1.0}] * 3}}, f)
        synthetic_project._highres_viewport_callback(viewer, None)
        assert calls["requests"] == 0, "a tile was requested despite hist matching"
        assert calls["cleared"] == 1, (
            "an already-drawn tile was left on screen after hist matching was "
            "switched on -- it now shows unmatched pixels")

        # ...and switching it back off must restore refinement, with no
        # re-display in between.
        with open(axp, "w", encoding="utf-8") as f:
            json.dump({"hist_enabled": False,
                       "hist_match": {"mode": "meanstd", "bands": 3,
                                      "ref_stats": [{"mean": 1.0, "std": 1.0}] * 3}}, f)
        synthetic_project._highres_viewport_callback(viewer, None)
        assert calls["requests"] == 1, (
            "the overlay stayed disabled after histogram matching was turned "
            "off -- the gate is still being decided once at display time")
    finally:
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)


def test_falls_back_when_the_viewer_has_no_image_data(synthetic_project, monkeypatch):
    """The captured value is still a legitimate FALLBACK -- it just must not
    win over a live one."""
    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    seen = {}
    monkeypatch.setattr(
        synthetic_project, "_request_highres_viewport_region",
        lambda v, i, r, request_id=None, ax=None: seen.update(imgd=i))

    fallback = _lite(fp, np.zeros((96, 96, 3), np.float32))
    viewer = types.SimpleNamespace(image_data=None,
                                   cancel_pending_highres_requests=lambda: None,
                                   _clear_highres_overlay=lambda: None)
    synthetic_project._highres_viewport_callback(
        viewer, None, fallback_image_data=fallback)
    assert seen.get("imgd") is fallback


def test_no_reader_means_no_request(synthetic_project, monkeypatch):
    """Non-windowable rasters must not reach the tile path at all."""
    from .fixtures_manifest import fixture_image_path

    calls = []
    monkeypatch.setattr(synthetic_project, "_request_highres_viewport_region",
                        lambda *a, **k: calls.append(1))
    imgd = _lite(fixture_image_path("rgb_8bit_untiled"),
                 np.zeros((96, 96, 3), np.float32), reader=None)
    viewer = types.SimpleNamespace(image_data=imgd,
                                   cancel_pending_highres_requests=lambda: None,
                                   _clear_highres_overlay=lambda: None)
    synthetic_project._highres_viewport_callback(viewer, None)
    assert not calls
