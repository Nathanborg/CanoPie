"""
ONE NoData rule for every analysis consumer -- CSV export, the ML manager's
export and training, Inspect and the viewer's data.

Per band, GDAL style (Nathan, 2026-10-06):

  * a NoData NUMBER excludes a pixel only in the band that holds that value;
  * NaN / Inf excludes it only in that band;
  * a NoData EXPRESSION (``b1<123``) excludes the band(s) it references;
  * derived bands (a band expression, appended bands, a classification label)
    are never matched against NoData numbers: they are NoData wherever any band
    they read is NoData.

Before this module there were two rules. ``utils.build_nodata_mask`` (the
normal-size path) matched every number against every channel -- derived bands
included, so ``nodata 0`` plus a boolean ``b1>100`` deleted every False pixel
from every band -- and excluded the WHOLE pixel; ``_per_band_nodata_masks``
(the large-file path) was per band. The same project gave different pixel
counts depending on the file size.

Band numbers are file band numbers: ``b1`` is the file's first band (Red for an
RGB TIFF), whatever order the array happens to be in.
"""
import re

import numpy as np

__all__ = ["NoDataRules", "rules_from_values", "tolerance", "band_mask",
           "expression_source_bands", "apply_nan_per_band"]

# Same grammar the editor writes (utils.parse_nodata_text).
_EXPR_RE = re.compile(r'^([bB])(\d+)\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)$')
_BAND_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9_])[bB](\d+)(?![A-Za-z0-9_])')


def tolerance(value):
    """How close a pixel value must be to a NoData number to count as NoData.

    One definition (it was the eager path's; the lazy path used a coarser
    two-tier version with a strict ``<``)."""
    a = abs(float(value))
    if a > 1e30:
        return a * 0.01
    if a > 100:
        return a * 0.001
    return 0.01


class NoDataRules:
    """Parsed NoData values: numbers, and per-band threshold expressions."""
    __slots__ = ("numbers", "exprs")

    def __init__(self, numbers=(), exprs=None):
        self.numbers = [float(v) for v in numbers]
        #: 0-based file band -> [(op, threshold), ...]
        self.exprs = dict(exprs or {})

    def __bool__(self):
        return bool(self.numbers or self.exprs)

    def __repr__(self):
        return f"NoDataRules(numbers={self.numbers}, exprs={self.exprs})"


def rules_from_values(values):
    """``.ax`` ``nodata_values`` (numbers and ``bN op x`` strings) -> rules.
    Anything else is ignored, as before."""
    numbers, exprs = [], {}
    for v in (values or []):
        if isinstance(v, str):
            m = _EXPR_RE.match(v.strip())
            if m:
                band0 = int(m.group(2)) - 1
                exprs.setdefault(band0, []).append((m.group(3), float(m.group(4))))
                continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            numbers.append(f)
    return NoDataRules(numbers, exprs)


def band_mask(plane, band0, rules):
    """True where FILE band ``band0`` (0-based) is NoData in ``plane``."""
    a = np.asarray(plane)
    af = a if a.dtype == np.float32 else a.astype(np.float32, copy=False)
    m = ~np.isfinite(af) if np.issubdtype(af.dtype, np.floating) else np.zeros(af.shape, bool)
    if rules is None:
        return m
    for nv in rules.numbers:
        m |= np.abs(af - nv) <= tolerance(nv)
    for op, thr in rules.exprs.get(int(band0), ()):
        if op == '<':
            m |= af < thr
        elif op == '<=':
            m |= af <= thr
        elif op == '>':
            m |= af > thr
        elif op == '>=':
            m |= af >= thr
        elif op == '==':
            m |= np.isclose(af, thr, rtol=0.0, atol=1e-6)
        elif op == '!=':
            m |= ~np.isclose(af, thr, rtol=0.0, atol=1e-6)
    return m


def expression_source_bands(expr):
    """0-based file bands a band expression reads (``b2 - b1`` -> {0, 1})."""
    if not isinstance(expr, str):
        return set()
    return {int(m.group(1)) - 1 for m in _BAND_TOKEN_RE.finditer(expr) if int(m.group(1)) >= 1}


def apply_nan_per_band(img, rules, n_file_bands=None):
    """Set NaN, in place, wherever each FILE band is NoData. ``img`` must be a
    float array (H, W, C) in FILE band order; only its first ``n_file_bands``
    channels are file bands (default: all). Returns ``img``."""
    if img.ndim == 2:
        img = img[..., None]
    n = img.shape[2] if n_file_bands is None else min(int(n_file_bands), img.shape[2])
    for k in range(n):
        plane = img[..., k]
        m = band_mask(plane, k, rules)
        if m.any():
            plane[m] = np.nan
    return img
