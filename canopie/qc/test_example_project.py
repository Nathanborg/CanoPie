"""The tutorial's example project (examples/make_example_project.py) keeps
giving what docs/tutorial.md says it gives."""
import json
import os
import sys

import numpy as np
import pytest

from .. import canopy_finder as CF

pytestmark = [pytest.mark.io]

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples")


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    sys.path.insert(0, _EX)
    try:
        import make_example_project as E
    finally:
        sys.path.remove(_EX)
    out = E.main(str(tmp_path_factory.mktemp("canopie_example")))
    return E, out


def test_the_example_project_has_what_the_tutorial_lists(example):
    E, out = example
    proj = json.load(open(os.path.join(out, "project.json"), encoding="utf-8"))
    assert proj["multispectral_root_names"] == ["Mosaics", "Phenocam"]
    groups = proj["multispectral_image_data_groups"]
    assert len(groups["Mosaics"]) == 3 and len(groups["Phenocam"]) == 5
    assert all(os.path.isfile(p) for g in groups.values() for p in g)
    assert set(proj["all_polygons"]) == {"tree_1", "tree_2", "tree_3", "tree_4", "tree_5", "leaf", "bare",
                                         "understory"}
    ms = sorted(os.listdir(os.path.join(out, "flight", "multispectral")))
    th = sorted(os.listdir(os.path.join(out, "flight", "thermal")))
    assert len(ms) == 16 and ms[0] == "IMG_0001_1.tif" and th == [f"IMG_{k:04d}_IR.tif" for k in (101, 102, 103, 104)]
    g = CF.georef(groups["Mosaics"][0])
    assert g is not None and g.epsg == 32721 and CF.file_band_count(groups["Mosaics"][0]) == 4


@pytest.mark.parametrize("k", [1, 2])
def test_the_crowns_are_found_at_the_built_in_offsets(example, k):
    """docs/tutorial.md section 10: 3.1 m E, 1.7 m S on date 2; 2.4 m W, 2.2 m N on date 3."""
    E, out = example
    proj = json.load(open(os.path.join(out, "project.json"), encoding="utf-8"))
    src, tgt = proj["multispectral_image_data_groups"]["Mosaics"][0], proj["multispectral_image_data_groups"]["Mosaics"][k]
    dx, dy = E.DATES[k][1]
    for name in ("tree_1", "tree_3"):
        rec = proj["all_polygons"][name][src]
        ctx = CF.prepare_source(src, rec, {}, {"method": "stackreg"})
        r = CF.find_in_target(ctx, tgt, {}, {"method": "stackreg"})
        off = (np.asarray(r.raw_pts).mean(axis=0) - np.asarray(r.geo_pts).mean(axis=0)) * E.GSD
        assert r.status == "found", (name, k, r.status, r.score)
        assert abs(off[0] - dx) < 0.15 and abs(-off[1] - dy) < 0.15, (name, k, off, (dx, dy))
        if name == "tree_3" and k == 2:
            assert r.score < 0.8                      # bare on the last date: found, with a lower score
        else:
            assert r.score > 0.85


def test_tree_3_loses_its_leaves_on_the_last_date(example):
    E, out = example
    proj = json.load(open(os.path.join(out, "project.json"), encoding="utf-8"))
    mos = proj["multispectral_image_data_groups"]["Mosaics"]
    rec = proj["all_polygons"]["tree_3"][mos[0]]
    P = np.asarray(rec["points"])
    ndvi = []
    for k, fp in enumerate(mos):
        sx, sy = round(E.DATES[k][1][0] / E.GSD), round(-E.DATES[k][1][1] / E.GSD)
        x0, y0 = (P.min(axis=0) + (sx, sy)).astype(int)
        x1, y1 = (P.max(axis=0) + (sx, sy)).astype(int)
        a, _v = CF.read_raw_window(fp, x0, y0, x1, y1, bands=[0, 3])
        a = a.astype(float)
        ndvi.append(float(np.median((a[:, :, 1] - a[:, :, 0]) / (a[:, :, 1] + a[:, :, 0] + 1e-9))))
    assert ndvi[0] > 0.75 and ndvi[1] > 0.75 and ndvi[2] < 0.55, ndvi
