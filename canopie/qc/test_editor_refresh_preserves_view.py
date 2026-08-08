"""
Applying edits must not silently throw away the viewer's stretch.

THE BUG THIS PINS (reported as "temporary change of stretching and zoom when
image editor Apply All Changes / Reset Image is applied"):

`refresh_single_viewer` -- the function `edit_image_viewer` calls after the
editor closes -- contains

    if has_classification:
        ...
    elif has_expr and hasattr(imgd.image, "shape") and ...:

`has_expr` is never assigned anywhere in that function. The surrounding names
are `prefer_last_band` and `has_classification`; `has_expr` looks like it was
renamed and one use was missed. So on every refresh where classification is NOT
active -- i.e. essentially every Apply All Changes -- evaluating that `elif`
raises NameError.

It was invisible because the whole block sits in

    try:
        ...
        pm = self._render_with_viewer_stretch(imgd.image, viewer, ...)
    except Exception:
        pass
    if pm is None:
        pm = self.convert_cv_to_pixmap(imgd.image, prefer_last_band=prefer_last_band)

The NameError fires BEFORE `_render_with_viewer_stretch` is reached, is
swallowed by the bare `except`, and `pm` falls through to
`convert_cv_to_pixmap` -- which knows nothing about `viewer.stretch_params`.
The image still appears, so nothing looks broken; it is just rendered with a
default stretch instead of the one the user set. That is the reported
"temporary change of stretching".
"""
import inspect

import numpy as np
import pytest

from ..project_tab import ProjectTab

pytestmark = [pytest.mark.viewer, pytest.mark.editor]


def test_refresh_single_viewer_has_no_undefined_names():
    """A static guard, because the runtime one is swallowed by a bare except.

    Compiles the function and checks every name it loads is either bound
    inside it, an argument, a global, or a builtin.
    """
    src = inspect.getsource(ProjectTab.refresh_single_viewer)
    import textwrap
    import ast

    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]

    assigned = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                assigned.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not fn:
                assigned.add(node.name)
    for a in fn.args.args + fn.args.kwonlyargs:
        assigned.add(a.arg)

    import builtins
    from .. import project_tab as _pt
    known = assigned | set(dir(builtins)) | set(vars(_pt))

    loaded = {n.id for n in ast.walk(fn)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    undefined = sorted(loaded - known)

    assert not undefined, (
        f"refresh_single_viewer references undefined name(s): {undefined}. "
        "The whole render block is wrapped in a bare `except Exception: pass`, "
        "so this raises silently and the viewer falls back to "
        "convert_cv_to_pixmap -- losing the user's stretch on every "
        "Apply All Changes.")


@pytest.mark.parametrize("ax_mods", [
    pytest.param({}, id="no-edits"),
    pytest.param({"band_enabled": True, "band_expression": "b1+b2"}, id="band-math"),
    pytest.param({"crop_enabled": True,
                  "crop_rect": {"x": 4, "y": 4, "width": 40, "height": 40},
                  "crop_rect_ref_size": {"w": 96, "h": 96}}, id="crop"),
])
def test_refresh_reaches_the_stretch_aware_renderer(
        synthetic_project, viewer_factory, monkeypatch, tmp_path, ax_mods):
    """THE regression, behaviourally.

    Drives the real refresh and asserts the STRETCH-AWARE renderer is what
    produced the pixmap. Under the NameError the call never happened and the
    plain `convert_cv_to_pixmap` fallback rendered instead -- with a default
    stretch, not the user's.

    Parametrised over the edit states that reach the broken branch: the
    `elif has_expr` was evaluated whenever classification was off, so a plain
    image and a cropped one hit it just as a band-math one does.
    """
    import json
    import os

    from .fixtures_manifest import fixture_image_path

    fp = fixture_image_path("rgb_8bit_untiled")
    axp = synthetic_project._ax_path_for(fp)
    prev = open(axp, encoding="utf-8").read() if os.path.exists(axp) else None

    viewer = viewer_factory()
    imgd = synthetic_project._imagedata_or_fallback(fp)
    assert imgd is not None and imgd.image is not None
    viewer.image_data = imgd
    rec = {"viewer": viewer, "image_data": imgd}
    synthetic_project.viewer_widgets = list(
        getattr(synthetic_project, "viewer_widgets", []) or []) + [rec]

    calls = {"stretch": 0, "fallback": 0}

    real_stretch = synthetic_project._render_with_viewer_stretch
    real_fallback = synthetic_project.convert_cv_to_pixmap

    def _spy_stretch(*a, **k):
        calls["stretch"] += 1
        return real_stretch(*a, **k)

    def _spy_fallback(*a, **k):
        calls["fallback"] += 1
        return real_fallback(*a, **k)

    monkeypatch.setattr(synthetic_project, "_render_with_viewer_stretch", _spy_stretch)
    monkeypatch.setattr(synthetic_project, "convert_cv_to_pixmap", _spy_fallback)

    try:
        with open(axp, "w", encoding="utf-8") as f:
            json.dump(ax_mods, f)
        for attr in ("_ax_json_cache", "_export_image_cache", "_pixmap_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()

        synthetic_project.refresh_single_viewer(viewer, reapply_mods=True,
                                                preserve_view=True)

        assert calls["stretch"] > 0, (
            "refresh_single_viewer never reached _render_with_viewer_stretch -- "
            "the render block raised before it and the bare `except Exception: "
            "pass` hid the reason, so the viewer was repainted without the "
            "user's stretch")
    finally:
        try:
            synthetic_project.viewer_widgets.remove(rec)
        except Exception:
            pass
        if prev is not None:
            open(axp, "w", encoding="utf-8").write(prev)
        elif os.path.exists(axp):
            os.remove(axp)
        for attr in ("_ax_json_cache", "_export_image_cache", "_pixmap_cache"):
            c = getattr(synthetic_project, attr, None)
            if hasattr(c, "clear"):
                c.clear()
