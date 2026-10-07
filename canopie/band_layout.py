"""
One description of the bands of an analysis image, shared by the CSV export,
the ML manager, Inspect and the viewer's band bar.

An analysis image (ProjectTab._replay_for_analysis) holds the file's bands in
FILE order -- b1 first, whatever order the loader produced -- followed by the
bands the .ax derives from them, in this order: the band expression, the
classification label, the appended bands (canopie.analysis_replay's order).
Each band has one name:

  file bands     "Gray" (1 band) | "Other", "band_2" (2 bands) |
                 "R", "G", "B", "band_4", ... (3+ bands; alpha is band_4)
  expression     its text, e.g. "b1>100"
  appended band  its expression, or the model name for a classification
  label          "Label"

The names were derived from the channel COUNT before, derived bands included:
the boolean of an RGBA file was "band_5" in the CSV, "band 4" in the viewer,
and a 2-band file with an expression came out labelled R, G, B.
"""
from .nodata import expression_source_bands

__all__ = ["BandEntry", "BandLayout", "file_band_names"]


def file_band_names(n_file):
    n = int(n_file)
    if n <= 0:
        return []
    if n == 1:
        return ["Gray"]
    if n == 2:
        return ["Other", "band_2"]
    return ["R", "G", "B"] + [f"band_{k}" for k in range(4, n + 1)]


class BandEntry:
    __slots__ = ("name", "kind", "sources", "file_index")

    def __init__(self, name, kind, sources=(), file_index=None):
        self.name = str(name)
        #: "file" | "expression" | "appended" | "label"
        self.kind = kind
        #: 0-based FILE bands this band reads (derived bands only)
        self.sources = frozenset(int(s) for s in (sources or ()))
        self.file_index = file_index

    def __repr__(self):
        return f"BandEntry({self.name!r}, {self.kind})"


class BandLayout:
    def __init__(self, n_file_bands):
        self.n_file = int(n_file_bands)
        self.entries = [BandEntry(nm, "file", (k,), k)
                        for k, nm in enumerate(file_band_names(self.n_file))]

    # -- building ----------------------------------------------------------
    def _unique(self, name):
        name = str(name).strip() or "band"
        taken = {e.name for e in self.entries}
        if name not in taken:
            return name
        k = 2
        while f"{name} #{k}" in taken:
            k += 1
        return f"{name} #{k}"

    def add_expression(self, expr, kind="expression"):
        e = BandEntry(self._unique(expr), kind, expression_source_bands(expr))
        self.entries.append(e)
        return e

    def add_derived(self, name, kind, sources):
        e = BandEntry(self._unique(name), kind, sources)
        self.entries.append(e)
        return e

    # -- reading -----------------------------------------------------------
    def __len__(self):
        return len(self.entries)

    def names(self):
        return [e.name for e in self.entries]

    def name(self, j):
        """Channel name of band j (positional fallback for out-of-range j)."""
        if 0 <= j < len(self.entries):
            return self.entries[j].name
        return f"band_{j + 1}"

    def index_of(self, name):
        for j, e in enumerate(self.entries):
            if e.name == name:
                return j
        return None

    @property
    def label_index(self):
        for j, e in enumerate(self.entries):
            if e.kind == "label":
                return j
        return None

    def is_file_band(self, j):
        return 0 <= j < len(self.entries) and self.entries[j].kind == "file"

    def __repr__(self):
        return f"BandLayout({self.names()})"
