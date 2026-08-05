"""Shared comparison utilities for the QC test modules."""
import json

import numpy as np

from .fixtures_manifest import ground_truth_json_path, ground_truth_npz_path


def load_ground_truth(name):
    with open(ground_truth_json_path(name), encoding="utf-8") as f:
        return json.load(f)


def load_raw_npz(name):
    return np.load(ground_truth_npz_path(name))["raw"]


def expected_viewer_loader(spec):
    """Mirrors _tifffile_is_stack's rule (project_tab.py): >3 bands or >3
    pages routes through tifffile-preflight; otherwise ImageData/cv2."""
    return "tifffile-preflight" if spec["bands"] > 3 else "imagedata"


def normalize_to_native_order(values, channel_order, band_count):
    """`values` are per-band, in the order an array with this channel_order
    physically stores them. Returns them reordered into NATIVE FILE band
    order, matching ground truth's convention (values[i] = file band i)."""
    if channel_order == "bgr" and band_count >= 3:
        return [values[2], values[1], values[0]] + list(values[3:])
    return list(values)


def pixel_values_native_order(img, x, y, channel_order):
    """Read img[y, x, :] and return the values in NATIVE FILE band order."""
    if img.ndim == 2:
        return [float(img[y, x])]
    C = img.shape[2]
    array_order_vals = [float(img[y, x, c]) for c in range(C)]
    return normalize_to_native_order(array_order_vals, channel_order, C)


def expected_channel_names(band_count):
    """The "Channel" column labels process_polygon emits, in file-band order.

    Verified empirically per band count -- these are NOT a uniform scheme:
      - >= 3 bands: "R", "G", "B", then "band_4" ... "band_N" (1-based)
      - 1 band:     "Gray"  (NOT "band_1" -- single-band images take
                    process_polygon's separate non-RGB branch)
    2-band images take that same non-RGB branch but aren't covered by any
    fixture, so their labels are deliberately not guessed at here.
    """
    n = int(band_count)
    if n >= 3:
        return ["R", "G", "B"] + [f"band_{i}" for i in range(4, n + 1)]
    if n == 1:
        return ["Gray"]
    raise NotImplementedError(
        f"No fixture covers {n}-band images, so their Channel labels are unverified"
    )


def expected_ml_channel_names(band_count):
    """The channel column names MachineLearningManager.export_csv_data uses.

    Different from process_polygon's scheme (confirmed by direct read of its
    header construction): "R"/"G"/"B" + "band_N" when any file in the export
    has >= 3 bands, else "channel_1" ... "channel_N".
    """
    n = int(band_count)
    if n >= 3:
        return ["R", "G", "B"] + [f"band_{i}" for i in range(4, n + 1)]
    return [f"channel_{i}" for i in range(1, n + 1)]


def assert_close(a, b, tol=1e-2, msg=""):
    assert a is not None and b is not None, f"{msg}: got None (a={a}, b={b})"
    assert abs(float(a) - float(b)) <= tol, f"{msg}: {a} != {b} (tol={tol})"
