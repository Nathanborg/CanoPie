"""
Regression tests pinning the MachineLearningManager / LazyChannels fix.

THE BUG (found while building this suite, fixed as part of it):
raster_reader.LazyChannels -- what _get_export_image returns for any large,
windowable raster -- exposes .shape but deliberately NOT .ndim (it is a
band-list-like, not an array). MachineLearningManager carried its own copy of
_channels_in_export_order that was missing the _is_lazy_channels guard
ProjectTab's version has, and did `if img.ndim == 3:` directly, so:

  - export_csv_data raised AttributeError, caught by its top-level try/except
    and surfaced as an "Export Failed" dialog;
  - train_models raised the SAME AttributeError even earlier, at its
    band-detection pre-pass (`C = 1 if img.ndim == 2 else img.shape[2]`), with
    no try/except anywhere around it -- a raw unhandled crash before any
    dialog opened.

Both paths were dead for exactly the files most likely to need ML export:
big hyperspectral cubes. The fix delegates to ProjectTab's already-correct
implementation and uses the band count _get_export_image already returns.

These tests fail loudly if either regression is reintroduced.
"""
import csv

import pytest
from PyQt5 import QtWidgets

from .fixtures_manifest import fixture_image_path
from .project_builder import polygon_group_name

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.ml]

LAZY_FIXTURE = "hyperspectral_200band"


def _clear_export_cache(pt):
    cache = getattr(pt, "_export_image_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _lazy_channels_for(pt, monkeypatch, name=LAZY_FIXTURE):
    from .. import project_tab as project_tab_module
    from ..raster_reader import LazyChannels

    _clear_export_cache(pt)
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)
    img, info = pt._get_export_image(fixture_image_path(name))
    assert isinstance(img, LazyChannels), (
        f"test setup failed: expected LazyChannels, got {type(img).__name__}")
    return img, info


def test_ml_channels_in_export_order_passes_lazy_through(
        synthetic_project, ml_manager_factory, monkeypatch):
    """THE regression test for the primary crash: this call raised
    AttributeError: 'LazyChannels' object has no attribute 'ndim' before the
    fix. It must now return the LazyChannels untouched -- materializing it
    would silently defeat the point of reading lazily."""
    img, _ = _lazy_channels_for(synthetic_project, monkeypatch)
    mgr = ml_manager_factory(synthetic_project)

    result = mgr._channels_in_export_order(img)
    assert result is img, (
        "expected the LazyChannels to be passed through unchanged (it already "
        f"carries export band order), got {type(result).__name__}")


def test_ml_channels_in_export_order_still_handles_plain_arrays(
        synthetic_project, ml_manager_factory):
    """The fix delegates to ProjectTab; make sure that didn't break the
    ordinary ndarray case -- 3-channel input must still come back as a list of
    2-D float32 bands."""
    import numpy as np

    mgr = ml_manager_factory(synthetic_project)
    arr = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    chans = mgr._channels_in_export_order(arr)

    assert isinstance(chans, list) and len(chans) == 3
    for ch in chans:
        assert ch.shape == (4, 4), f"expected 2-D HxW bands, got {ch.shape}"


def test_train_models_band_detection_survives_lazy(synthetic_project, monkeypatch):
    """The SECOND crash site: train_models' band-detection pre-pass used
    img.ndim before any try/except existed to catch it. Exercise that exact
    line's logic against a real LazyChannels -- the band count must come from
    _get_export_image's own return value, with no .ndim access at all."""
    img, info = _lazy_channels_for(synthetic_project, monkeypatch)

    _, c_from_helper = synthetic_project._get_export_image(
        fixture_image_path(LAZY_FIXTURE))
    band_count = int(c_from_helper["C"] if isinstance(c_from_helper, dict) else c_from_helper)

    assert band_count == 200, f"expected 200 bands, got {band_count}"
    assert band_count == len(img), "band count must agree with the LazyChannels' own length"


def test_export_csv_data_completes_on_lazy_raster(
        synthetic_project, ml_manager_factory, monkeypatch, tmp_path):
    """End-to-end: the full export that previously died with an 'Export
    Failed' dialog must now produce a real CSV with correct values.

    Note this drives the export with the threshold patched for the WHOLE call,
    so the file is read lazily throughout -- not just in the setup above."""
    from .. import project_tab as project_tab_module

    pt = synthetic_project
    name = LAZY_FIXTURE
    group = polygon_group_name(name, "poly_interior")

    _clear_export_cache(pt)
    monkeypatch.setattr(project_tab_module, "_EXPORT_LAZY_THRESHOLD_BYTES", 1)

    mgr = ml_manager_factory(pt)
    for i in range(mgr.list_widget.count()):
        item = mgr.list_widget.item(i)
        item.setSelected(item.text() == group)

    calls = {"n": 0}
    def fake_get_item(*a, **k):
        calls["n"] += 1
        return ("Average Pixel Value", True) if calls["n"] == 1 else ("3x3", True)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", staticmethod(fake_get_item))
    out_csv = tmp_path / "lazy_export.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out_csv), "")))

    mgr.export_csv_data()

    assert out_csv.exists(), (
        "export_csv_data produced no file -- it most likely hit its top-level "
        "except and showed 'Export Failed' (the pre-fix symptom)")
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected one averaged row, got {len(rows)}"

    from ._helpers import load_ground_truth, assert_close
    gt = load_ground_truth(name)
    assert_close(float(rows[0]["mean_R"]),
                 gt["polygon"]["bands"]["0"]["no_mask"]["mean"], tol=1e-1,
                 msg="lazy-read ML export mean_R")
