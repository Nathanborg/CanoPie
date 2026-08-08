"""Turn a list of changed source files into the pytest command that covers them.

Usage
-----
    python -m canopie.qc.which_tests canopie/image_viewer.py canopie/utils.py
    python -m canopie.qc.which_tests $(git diff --name-only)

Why this exists: "I changed X, what do I run?" needs a mechanical answer, not a
judgement call. Guessing wrong is worse than running everything, so anything
unrecognised falls back to the FULL suite -- fail safe, never fail silent.

The four files in `CORE` map to `extraction` AND `contract` on purpose: they sit
underneath every consumer, so a change there can break a subsystem that looks
completely unrelated. `utils.eval_band_expression` and
`performance.FastBandMathEngine` are the same computation reached two different
ways, and the CSV/ML/Inspect/viewer paths all bottom out in `project_tab` and
`raster_reader`.
"""
import os
import sys

# Underpins every consumer -- always pull in the cross-module contracts too.
CORE = {
    "utils.py":          ["extraction", "contract"],
    "performance.py":    ["extraction", "contract"],
    "raster_reader.py":  ["extraction", "contract", "io"],
    "project_tab.py":    ["extraction", "contract", "io", "viewer", "ml"],
}

# Everything else: one subsystem (plus contract where it participates in a
# cross-module invariant).
MODULE_MARKERS = {
    "image_viewer.py":            ["viewer", "contract"],
    "image_editor_dialog.py":     ["editor", "extraction", "contract"],
    "machine_learning_manager.py": ["ml", "extraction", "contract"],
    "polygon_manager.py":         ["polygons", "contract"],
    "shapefile_io.py":            ["io"],
    "loaders.py":                 ["io", "viewer"],
    "image_data.py":              ["io"],
    "random_shapes_generator.py": ["polygons"],
    "random_shapes_dialog.py":    ["polygons"],
    "similarity_dialog.py":       ["viewer"],
    "phenocam_filter.py":         ["io"],
    "main_window.py":             ["viewer"],
    "log_window.py":              ["viewer"],
    "patch_dialog.py":            ["editor"],
}

ALL_MARKERS = ["extraction", "viewer", "editor", "ml", "polygons", "io",
               "contract", "perf", "slow"]


def markers_for(paths):
    """Markers covering `paths`. Returns (markers, unknown_paths, run_all)."""
    marks, unknown = set(), []
    run_all = False
    for p in paths:
        name = os.path.basename(str(p).replace("\\", "/"))
        if not name.endswith(".py"):
            continue
        # A change to the QC suite itself, its fixtures, or the pytest config
        # can affect anything -- run everything.
        norm = str(p).replace("\\", "/")
        if "/qc/" in norm or name in ("conftest.py", "pytest.ini"):
            run_all = True
            continue
        if name in CORE:
            marks.update(CORE[name])
        elif name in MODULE_MARKERS:
            marks.update(MODULE_MARKERS[name])
        else:
            unknown.append(norm)
    return marks, unknown, run_all


def command_for(paths):
    """The exact pytest command to run for `paths`."""
    marks, unknown, run_all = markers_for(paths)
    base = "python -m pytest canopie/qc -p no:cacheprovider"
    if run_all or unknown or not marks:
        return base, marks, unknown, True
    expr = " or ".join(sorted(marks))
    return f'{base} -m "{expr}"', marks, unknown, False


def main(argv):
    paths = [a for a in argv if a.strip()]
    if not paths:
        print(__doc__.strip())
        print("\nMarkers:", ", ".join(ALL_MARKERS))
        return 0

    cmd, marks, unknown, full = command_for(paths)
    if unknown:
        print("# unrecognised (running the full suite to be safe): "
              + ", ".join(unknown))
    elif full:
        print("# QC suite / config touched -- running everything")
    else:
        print("# markers: " + ", ".join(sorted(marks)))
    print(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
