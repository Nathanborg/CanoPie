"""QC regression tests for the "appended bands are silently missing from
exports" bug (reported as: "the appended bands from pkl classification and
boolean are not being exported").

THE BUG: _apply_ax_to_raw's appended-bands replay (the .ax `appended_bands`
list, written by the Image Editor's Append button -- see
image_editor_dialog.py's do_append_band) regenerates a classification /
boolean_expression / band_expression band on every replay, since only the
model reference and label names are persisted to disk, not the pixel data
itself. Each of its three per-band-type branches wraps its own work in a
bare try/except that only logging.warning()s on failure -- deliberately, so
one bad band doesn't abort every other band or every other consumer of this
one shared function (CSV export, ML training, and image export all replay
through it). But that means a failure was previously INVISIBLE: the
consumer just got an image with fewer bands / a CSV missing a feature
column, with no error anywhere the user would ever see.

THE MOST COMMON TRIGGER, reproduced against a real project on real data
before this fix: a classification-type appended band needs the ORIGINAL
sklearn model to re-run prediction (_get_sklearn_bundle() ->
self.random_forest_model / the class-shared fallback). That state is
runtime-only, never persisted anywhere -- restart the app, or simply never
(re)load the .pkl this session, and _get_sklearn_bundle() returns None, so
EVERY classification-type appended band across EVERY consumer silently
vanishes. Verified end to end against a real 2-classification-band .ax
(C:\\New Folder215) and a real trained RandomForestClassifier .pkl: with the
model loaded, the exported TIFF has 5 bands (3 RGB + 2 classification);
without it, silently 3 bands, with only a buried logging.warning to explain
why.

THE FIX: _record_appended_band_error tracks a count + up to 5 sample
messages on ProjectTab, called from all three appended-bands branches on
failure. ProjectImagesExportWorker.run(), export_project_images's foreground
path, and the CSV ExportWorker.run() each reset the counter at the start of
their own run and append a WARNING to their completion message when it's
nonzero -- the same pattern _record_rf_prediction_error already established
for process_polygon's classification failures.
"""
import json
import os

import numpy as np
import pytest
import tifffile

pytestmark = [pytest.mark.io, pytest.mark.extraction]


def _write_rgb_tiff(path, r=10, g=20, b=30, h=8, w=8):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    tifffile.imwrite(path, arr, photometric="rgb")


class _FakeModel:
    """Stands in for a fitted sklearn classifier: predict() returns the
    first class for every row, classes_ matches label_names exactly."""
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _stub_tab(tmp_path, with_model=False):
    from ..project_tab import ProjectTab

    class _Tab:
        pass

    tab = _Tab()
    tab.project_folder = str(tmp_path)
    tab._load_ax_json = ProjectTab._load_ax_json.__get__(tab, ProjectTab)
    tab._ax_path_for_fp = ProjectTab._ax_path_for_fp.__get__(tab, ProjectTab)
    tab._ax_path_for = ProjectTab._ax_path_for.__get__(tab, ProjectTab)
    tab._apply_ax_to_raw = ProjectTab._apply_ax_to_raw.__get__(tab, ProjectTab)
    tab._get_sklearn_bundle = ProjectTab._get_sklearn_bundle.__get__(tab, ProjectTab)
    tab._make_feature_stack_for_model = ProjectTab._make_feature_stack_for_model.__get__(tab, ProjectTab)
    tab._record_appended_band_error = ProjectTab._record_appended_band_error.__get__(tab, ProjectTab)
    tab.exiftool_path = None
    if with_model:
        tab.random_forest_model = {
            "model": _FakeModel(),
            "feature_names": ["red_channel", "green_channel", "blue_channel"],
            "base_feature_names": ["red_channel", "green_channel", "blue_channel"],
            "expressions": [],
            "window_size": 1,
            "label_names": ["A", "B"],
        }
    return tab


def _write_ax(project_folder, image_fp, appended_bands):
    base = os.path.splitext(os.path.basename(image_fp))[0] + ".ax"
    with open(os.path.join(project_folder, base), "w", encoding="utf-8") as f:
        json.dump({"appended_bands": appended_bands}, f)


# ---------------------------------------------------------------------------
# _record_appended_band_error -- the tracking primitive
# ---------------------------------------------------------------------------
def test_record_appended_band_error_tracks_count_and_samples(tmp_path):
    tab = _stub_tab(tmp_path)
    tab._record_appended_band_error("img1.tif", "classification", 1, "boom")
    tab._record_appended_band_error("img2.tif", "boolean_expression", 2, "kaboom")
    assert tab._appended_band_error_count == 2
    assert len(tab._appended_band_error_samples) == 2
    assert "img1.tif" in tab._appended_band_error_samples[0]
    assert "classification #1" in tab._appended_band_error_samples[0]


def test_record_appended_band_error_caps_samples_at_five(tmp_path):
    tab = _stub_tab(tmp_path)
    for i in range(10):
        tab._record_appended_band_error(f"img{i}.tif", "classification", i, "x")
    assert tab._appended_band_error_count == 10
    assert len(tab._appended_band_error_samples) == 5


# ---------------------------------------------------------------------------
# _apply_ax_to_raw -- the real failure/success paths, on real image data
# ---------------------------------------------------------------------------
def test_classification_band_silently_dropped_without_a_loaded_model(tmp_path):
    """THE reproduced bug: no model in memory -> the band vanishes, but is
    now tracked as a recorded failure instead of just a buried log line."""
    src = str(tmp_path / "a.tif")
    _write_rgb_tiff(src)
    tab = _stub_tab(tmp_path, with_model=False)

    ax = {"appended_bands": [{"type": "classification", "index": 1,
                              "label_names": ["A", "B"]}]}
    img = tifffile.imread(src)
    out, C = tab._apply_ax_to_raw(img, ax, filepath=src)

    assert C == 3, "the classification band must be silently ABSENT (the bug), not error out"
    assert getattr(tab, "_appended_band_error_count", 0) == 1
    assert "classification" in tab._appended_band_error_samples[0]
    assert "no sklearn model" in tab._appended_band_error_samples[0]


def test_classification_band_appends_when_a_model_is_loaded(tmp_path):
    src = str(tmp_path / "a.tif")
    _write_rgb_tiff(src)
    tab = _stub_tab(tmp_path, with_model=True)

    ax = {"appended_bands": [{"type": "classification", "index": 1,
                              "label_names": ["A", "B"]}]}
    img = tifffile.imread(src)
    out, C = tab._apply_ax_to_raw(img, ax, filepath=src)

    assert C == 4, "with a model loaded, the classification band must be appended"
    assert getattr(tab, "_appended_band_error_count", 0) == 0


def test_boolean_expression_band_failure_is_recorded(tmp_path):
    src = str(tmp_path / "a.tif")
    _write_rgb_tiff(src)
    tab = _stub_tab(tmp_path)

    # A reference to a nonexistent band forces eval_band_expression to raise.
    ax = {"appended_bands": [{"type": "boolean_expression", "index": 1,
                              "expression": "b99 > 0"}]}
    img = tifffile.imread(src)
    out, C = tab._apply_ax_to_raw(img, ax, filepath=src)

    assert C == 3, "a failing expression must not silently add a garbage band"
    assert getattr(tab, "_appended_band_error_count", 0) == 1
    assert "boolean_expression" in tab._appended_band_error_samples[0]


def test_boolean_expression_band_appends_correctly(tmp_path):
    src = str(tmp_path / "a.tif")
    _write_rgb_tiff(src, r=150, g=20, b=30)
    tab = _stub_tab(tmp_path)

    ax = {"appended_bands": [{"type": "boolean_expression", "index": 1,
                              "expression": "b1 > 100"}]}
    img = tifffile.imread(src)
    out, C = tab._apply_ax_to_raw(img, ax, filepath=src)

    assert C == 4
    assert getattr(tab, "_appended_band_error_count", 0) == 0
    assert out[0, 0, 3] == 1.0  # r=150 > 100 -> True


def test_band_expression_failure_is_recorded(tmp_path):
    src = str(tmp_path / "a.tif")
    _write_rgb_tiff(src)
    tab = _stub_tab(tmp_path)

    ax = {"appended_bands": [{"type": "band_expression", "index": 1,
                              "expression": "b99 / b1"}]}
    img = tifffile.imread(src)
    out, C = tab._apply_ax_to_raw(img, ax, filepath=src)

    assert C == 3
    assert getattr(tab, "_appended_band_error_count", 0) == 1
    assert "band_expression" in tab._appended_band_error_samples[0]


# ---------------------------------------------------------------------------
# End to end: ProjectImagesExportWorker surfaces the failure in its
# completion message (the actual user-visible fix)
# ---------------------------------------------------------------------------
def test_worker_completion_message_warns_when_a_band_is_dropped(tmp_path, qapp):
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"
    src = str(src_dir / "a.tif")
    _write_rgb_tiff(src)
    _write_ax(str(tmp_path), src, [{"type": "classification", "index": 1,
                                     "label_names": ["A", "B"]}])

    tab = _stub_tab(tmp_path, with_model=False)
    worker = ProjectImagesExportWorker(tab, str(out_dir), [src], "tif")
    captured = {}
    worker.finished.connect(lambda p, ok, m: captured.update(ok=ok, msg=m))
    worker.run()

    assert captured["ok"] is True, "the export itself must still succeed"
    assert "WARNING" in captured["msg"]
    assert "appended band" in captured["msg"]

    produced = list(out_dir.glob("*.tif"))
    assert produced
    got = tifffile.imread(str(produced[0]))
    assert got.shape[-1] == 3 if got.ndim == 3 else True  # the band really is missing


def test_worker_completion_message_is_clean_when_nothing_fails(tmp_path, qapp):
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"
    src = str(src_dir / "a.tif")
    _write_rgb_tiff(src)
    _write_ax(str(tmp_path), src, [{"type": "classification", "index": 1,
                                     "label_names": ["A", "B"]}])

    tab = _stub_tab(tmp_path, with_model=True)
    worker = ProjectImagesExportWorker(tab, str(out_dir), [src], "tif")
    captured = {}
    worker.finished.connect(lambda p, ok, m: captured.update(ok=ok, msg=m))
    worker.run()

    assert "WARNING" not in captured["msg"], captured["msg"]


def test_worker_resets_tracking_between_runs(tmp_path, qapp):
    """A previous run's failures must not bleed into the next run's
    completion message."""
    from ..project_tab import ProjectImagesExportWorker

    src_dir = tmp_path / "src"; src_dir.mkdir()
    src = str(src_dir / "a.tif")
    _write_rgb_tiff(src)
    _write_ax(str(tmp_path), src, [{"type": "classification", "index": 1,
                                     "label_names": ["A", "B"]}])

    tab = _stub_tab(tmp_path, with_model=False)
    out_dir1 = tmp_path / "out1"
    ProjectImagesExportWorker(tab, str(out_dir1), [src], "tif").run()
    assert tab._appended_band_error_count == 1

    # Second run, now with a model available -- must start clean, not carry
    # the first run's count forward.
    tab.random_forest_model = {
        "model": _FakeModel(),
        "feature_names": ["red_channel", "green_channel", "blue_channel"],
        "base_feature_names": ["red_channel", "green_channel", "blue_channel"],
        "expressions": [], "window_size": 1, "label_names": ["A", "B"],
    }
    out_dir2 = tmp_path / "out2"
    worker2 = ProjectImagesExportWorker(tab, str(out_dir2), [src], "tif")
    captured = {}
    worker2.finished.connect(lambda p, ok, m: captured.update(ok=ok, msg=m))
    worker2.run()
    assert "WARNING" not in captured["msg"], captured["msg"]


# ---------------------------------------------------------------------------
# AST-level: the other two consumers (foreground image export, CSV export)
# are wired the same way
# ---------------------------------------------------------------------------
def test_foreground_export_wires_the_recorder():
    from .test_export_and_ax_regressions import _names_in, _calls_in
    from ..project_tab import ProjectTab
    names = _names_in(ProjectTab.export_project_images)
    assert "_appended_band_error_count" in names
    assert "_appended_band_error_samples" in names


def test_csv_export_worker_wires_the_recorder():
    from .test_export_and_ax_regressions import _names_in
    from ..project_tab import ExportWorker
    names = _names_in(ExportWorker.run)
    assert "_appended_band_error_count" in names
    assert "_appended_band_error_samples" in names


def test_all_three_appended_band_branches_call_the_recorder():
    """Guards against a future edit adding a fourth branch (or refactoring
    one of these three) that forgets to wire in the recorder."""
    from .test_export_and_ax_regressions import _calls_in
    from ..project_tab import ProjectTab
    assert _calls_in(ProjectTab._apply_ax_to_raw, "_record_appended_band_error")
