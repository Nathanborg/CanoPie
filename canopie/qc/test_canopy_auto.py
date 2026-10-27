"""
QC tests: the Auto Canopy engine (canopie/canopy_auto.py; offline only -- not wired into the app).

A synthetic closed canopy -- touching, dome-shaded crowns of different sizes with
dark gaps between them -- must come out crown by crown with none overlapping;
plus the size model, learning, and turning recorded clicking into settings.
"""
import numpy as np
import cv2
import pytest

from ..canopy_auto import (AutoParams, SIZE_MODEL_BCI, adapt_size_model, delineate_all, fit_size_model,
                           learn_params, params_from_sessions, predict_size, score_crowns, size_samples)

pytestmark = [pytest.mark.polygons]

H, W = 520, 640


def _canopy(seed=7):
    """RGB uint8 closed canopy + the true crown polygons (array px, pixel centres at integers)."""
    rng = np.random.default_rng(seed)
    centres, radii = [], []
    for gy in range(50, H - 30, 95):
        for gx in range(50, W - 30, 100):
            centres.append((gx + rng.uniform(-15, 15) + (45 if (gy // 95) % 2 else 0), gy + rng.uniform(-12, 12)))
            radii.append(rng.uniform(32, 50))
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    # each pixel belongs to the crown whose (radius-scaled) distance is least, if within its radius
    best = np.full((H, W), np.inf, np.float32)
    lab = np.full((H, W), -1, np.int32)
    for i, ((cx, cy), r) in enumerate(zip(centres, radii)):
        d = np.hypot(xx - cx, yy - cy) / r
        upd = (d < best) & (d <= 1.0)
        best[upd] = d[upd]
        lab[upd] = i
    img = np.zeros((H, W, 3), np.float32) + np.float32([25, 40, 20])          # shadowed gaps
    tex = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32) * 30, (0, 0), 1.8)
    for i in range(len(centres)):
        m = lab == i
        base = np.float32([rng.uniform(50, 110), rng.uniform(120, 190), rng.uniform(40, 80)])
        dome = (1.0 - 0.55 * best[m] ** 2)[:, None]
        img[m] = base * dome + tex[m][:, None]
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    truth = []
    for i in range(len(centres)):
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
            if cv2.contourArea(c.astype(np.float32)) > 400:
                truth.append(c)
    return np.clip(img, 0, 255).astype(np.uint8), truth


@pytest.fixture(scope="module")
def canopy():
    return _canopy()


def _mask(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly) * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
    return m > 0


def test_a_closed_canopy_comes_out_crown_by_crown(canopy):
    img, truth = canopy
    res = delineate_all(img, AutoParams())
    s = score_crowns(img.shape, [c.polygon for c in res.crowns], truth)
    assert s["f1"] >= 0.6, (s, res.note)
    assert 25 <= res.crown_radius_px * 1.0 <= 70, res.note


def test_crowns_never_overlap_and_keep_existing_ones(canopy):
    img, truth = canopy
    stop = _mask(truth[0]) | _mask(truth[5])
    res = delineate_all(img, AutoParams(), stop_mask=stop)
    masks = [_mask(c.polygon) for c in res.crowns]
    total = np.zeros((H, W), np.int32)
    for m in masks:
        total += m
    assert (total > 1).sum() <= 0.01 * max(1, (total > 0).sum()), "crowns overlap"
    assert (total.astype(bool) & stop).sum() <= 0.02 * stop.sum(), "an existing crown was taken"


def test_region_limits_where_crowns_are_made(canopy):
    img, _ = canopy
    region = np.zeros((H, W), bool)
    region[:, : W // 2] = True
    res = delineate_all(img, AutoParams(), region=region)
    assert res.crowns
    assert all(c.polygon[:, 0].mean() < W // 2 for c in res.crowns)


def test_cancel_stops_early(canopy):
    img, _ = canopy
    calls = []
    res = delineate_all(img, AutoParams(), progress=lambda d, t: calls.append(d), cancel=lambda: len(calls) >= 3)
    assert "cancelled" in res.note and len(res.crowns) <= 3


def test_true_centres_and_sizes_are_used_as_given(canopy):
    img, truth = canopy
    sd = np.array([np.r_[t.mean(0), np.sqrt(cv2.contourArea(t.astype(np.float32)) / np.pi)] for t in truth])
    res = delineate_all(img, AutoParams(), seeds=sd)
    assert len(res.seeds) <= len(sd) + 40                    # the given centres (+ residual passes)
    s = score_crowns(img.shape, [c.polygon for c in res.crowns], truth)
    assert s["f1"] >= 0.7, s


def test_the_size_model_tells_big_crowns_from_small_ones(canopy):
    """Fitted on half the synthetic crowns, the per-crown size beats one size for all on the other half."""
    img, truth = canopy
    X, y, _ = size_samples(img, truth)
    tr, te = np.arange(len(y)) % 2 == 0, np.arange(len(y)) % 2 == 1
    m = fit_size_model(X[tr], y[tr], alpha=10.0)
    err = np.abs(np.log(predict_size(m, X[te])) - y[te])
    assert np.median(err) < np.median(np.abs(y[te])), (np.median(err), np.median(np.abs(y[te])))


def test_adapting_the_size_model_keeps_close_to_the_prior_with_few_crowns(canopy):
    img, truth = canopy
    X, y, _ = size_samples(img, truth)
    few = adapt_size_model(X[:3], y[:3], SIZE_MODEL_BCI, alpha=1e6)
    assert np.allclose(few["coef"], SIZE_MODEL_BCI["coef"], atol=1e-3)
    many = adapt_size_model(X, y, SIZE_MODEL_BCI, alpha=1.0)
    assert not np.allclose(many["coef"], SIZE_MODEL_BCI["coef"], atol=1e-3)
    assert many["crowns"] == len(y)


def test_params_from_sessions_copy_the_users_clicking():
    """Recorded crowns -> the clicking they show: 9 clicks, the first 3 within 0.3 R, brush R/5, grow 4."""
    rng = np.random.default_rng(0)
    sessions = []
    for i in range(6):
        c, R = np.array([500.0 + 300 * i, 400.0]), 100.0
        poly = [(c + R * np.array([np.cos(a), np.sin(a)])).tolist() for a in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
        clicks = [c + 0.3 * R * np.array([np.cos(a), np.sin(a)]) for a in (0.0, 2.1, 4.2)]
        clicks += [c + rng.uniform(0.4, 0.7) * R * np.array([np.cos(a), np.sin(a)]) for a in rng.uniform(0, 6.28, 6)]
        sessions.append(dict(polygon=poly, grow=4.0, screen_per_scene=1.0, source_per_scene=1.0, raster_per_scene=1.0,
                             strokes=[dict(pts=[q.tolist()], r=20.0, erase=False) for q in clicks]))
    p, prof = params_from_sessions(sessions, AutoParams())
    assert prof["crowns"] == 6 and round(prof["n"]) == 9
    assert p.clicks_first == 3 and p.rounds == 3 and p.clicks_round == 3
    assert abs(p.first_spread - 0.3) < 0.03 and abs(p.brush - 0.2) < 0.01 and p.grow == 4.0
    few, prof2 = params_from_sessions(sessions[:2], AutoParams())
    assert few == AutoParams() and prof2["crowns"] == 2       # too few recorded crowns: unchanged


def test_learn_params_does_not_get_worse(canopy):
    img, truth = canopy
    best, score, before = learn_params([(img, truth)], AutoParams(), n_random=0, workers=2, refine=False)
    assert score["f1"] >= before["f1"]
    assert isinstance(best, AutoParams)
