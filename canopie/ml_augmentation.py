"""Training-time data augmentation for MachineLearningManager.

Two independently-toggleable stages, composed by `augment_image_for_training`
(the SINGLE orchestrator both `machine_learning_manager.py::train_models` and
`ml_augmentation_options_dialog.py`'s live preview call -- never reimplement
any of this math elsewhere; see `qc/test_ml_preview_matches_training.py`'s AST
guard, which fails the build if the preview dialog stops delegating here):

- **Brightness** (`apply_brightness_image_level` / `apply_brightness_patch_level`):
  a per-band linear rescale toward a randomly perturbed target mean/std,
  reusing the exact `(x - mu_source) * (sd_target/sd_source) + mu_target`
  formula `ProjectTab._apply_hist_match`'s meanstd mode uses
  (project_tab.py:4698-4702) -- duplicated here as `_linear_rescale` rather
  than imported, since that function's `.ax`/hist-match reference-provenance
  plumbing (stored reference stats, channel-order realignment) has no
  equivalent here: the "reference" is a fresh random draw from the image's
  OWN measured stats, not a stored file.
  - Image-level: one draw for the whole image.
  - Patch-level: the image is partitioned into a fixed-size grid of tiles,
    each independently measured and independently perturbed -- simulates
    uneven lighting across a scene, forcing a model trained on it to learn
    patterns that don't depend on absolute brightness. Deliberately computed
    ONCE per whole loaded image (a grid over the whole frame), not once per
    polygon: multiple polygons/pixels often sample the same physical image,
    and per-polygon-independent perturbation would give the same physical
    tile two different random brightnesses depending on which polygon
    happened to sample it -- not a physically plausible "one image variant."

- **Shadow / illumination** (`apply_shadow_illumination`): reimplements
  `uneven_illumination_rgb()` / `generate_noisy_images()` from
  https://github.com/Nathanborg/Cloud_shadow_correction -- a random
  linear/circular/diagonal illumination-gradient mask, Gaussian-blurred for
  feathering, applied multiplicatively per channel.

Both stages respect an optional `nodata_mask` (True = do not perturb this
pixel) so augmentation honors the same NoData/mask-polygon exclusions the
training loop's per-pixel sampling already respects -- callers must build
that mask themselves (e.g. via `utils.build_nodata_mask`) and pass it in;
nothing here re-derives it.
"""
import cv2
import numpy as np

__all__ = [
    "apply_brightness_image_level",
    "apply_brightness_patch_level",
    "apply_shadow_illumination",
    "augment_image_for_training",
    "assemble_pixel_rows",
]


def _linear_rescale(arr3, mu_source, sd_source, mu_target, sd_target, keep_mask=None):
    """Per-band linear rescale of an HxWxC array.

    out = (arr3 - mu_source) * (sd_target / sd_source) + mu_target, per band.
    mu_source/sd_source/mu_target/sd_target: length-C sequences.
    keep_mask: optional HxW bool array; True positions are restored to arr3's
    ORIGINAL (pre-rescale) values afterward -- mirrors
    project_tab.py:4716-4720's per-band-mask NoData restore in
    `_apply_hist_match`. Always HxWxC in, HxWxC out (float32); never mutates
    arr3 in place. 2D/3D squeeze-and-expand is the caller's job (each public
    function below does it once at its own boundary), so this private helper
    never has to guess.
    """
    a = np.asarray(arr3, dtype=np.float32)
    C = a.shape[2]
    mu_source = np.asarray(mu_source, dtype=np.float32).reshape(1, 1, C)
    sd_source = np.asarray(sd_source, dtype=np.float32).reshape(1, 1, C)
    mu_target = np.asarray(mu_target, dtype=np.float32).reshape(1, 1, C)
    sd_target = np.asarray(sd_target, dtype=np.float32).reshape(1, 1, C)
    sd_source_safe = np.where(sd_source < 1e-6, 1.0, sd_source)
    gain = sd_target / sd_source_safe
    out = (a - mu_source) * gain + mu_target
    if keep_mask is not None:
        out[keep_mask] = a[keep_mask]
    return out.astype(np.float32, copy=False)


def _band_stats(arr3, valid_2d):
    """Per-band (mean, std) of arr3's pixels where valid_2d is True. A band
    with zero valid pixels falls back to (0.0, 1.0) -- an inert rescale
    (mu_target lands at 0, sd_target at whatever jitter*1.0 draws) rather than
    a NaN-producing empty-slice reduction."""
    C = arr3.shape[2]
    mu = np.empty(C, dtype=np.float32)
    sd = np.empty(C, dtype=np.float32)
    for c in range(C):
        vals = arr3[..., c][valid_2d]
        if vals.size == 0:
            mu[c] = 0.0
            sd[c] = 1.0
        else:
            mu[c] = float(np.mean(vals))
            sd[c] = float(np.std(vals))
    return mu, sd


def _as_hwc(img):
    """Returns (arr3, was_2d) -- arr3 always HxWxC float32."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        return a[..., None], True
    return a, False


def apply_brightness_image_level(img, rng, nodata_mask=None, mu_jitter=0.15, sd_jitter=0.25):
    """Shift the WHOLE image's per-band mean/std to a randomly perturbed
    target, drawn as mu_target = mu_source * U(1-mu_jitter, 1+mu_jitter) (and
    similarly for sd_target/sd_jitter) -- simulates "this same scene, but
    captured under different overall exposure/lighting."

    img: HxWxC or HxW array, raw pixel values as loaded (before any band
        reordering). rng: np.random.Generator (share the caller's seeded
        instance for run-to-run reproducibility). nodata_mask: HxW bool,
        True = excluded from BOTH the source-stat measurement and the
        perturbation itself (restored to original value).
    Returns a new float32 array, same shape as img; img is never mutated.
    """
    a3, was_2d = _as_hwc(img)
    H, W, C = a3.shape
    valid = ~nodata_mask if nodata_mask is not None else np.ones((H, W), dtype=bool)

    mu_source, sd_source = _band_stats(a3, valid)
    mu_gain = rng.uniform(1.0 - mu_jitter, 1.0 + mu_jitter, size=C)
    sd_gain = rng.uniform(1.0 - sd_jitter, 1.0 + sd_jitter, size=C)
    mu_target = mu_source * mu_gain
    sd_target = np.maximum(sd_source * sd_gain, 1e-6)

    out3 = _linear_rescale(a3, mu_source, sd_source, mu_target, sd_target, keep_mask=nodata_mask)
    return out3[..., 0] if was_2d else out3


def apply_brightness_patch_level(img, rng, tile_size, nodata_mask=None, mu_jitter=0.15, sd_jitter=0.25):
    """Same contract as `apply_brightness_image_level`, but the image is
    partitioned into a fixed `tile_size` x `tile_size` grid (last row/col
    clipped at the image bounds -- a plain nested loop, NOT
    `utils.iter_classification_row_tiles`, which is a 1-D memory-budget row
    tiler built for a different purpose and would need a divergent contract
    here). Each cell independently measures its own source mean/std (over
    its own eligible pixels only) and independently draws its own target --
    so different regions of the SAME image end up with different simulated
    brightness. A cell with zero eligible pixels is left unperturbed.
    """
    a3, was_2d = _as_hwc(img)
    H, W, C = a3.shape
    valid_full = ~nodata_mask if nodata_mask is not None else np.ones((H, W), dtype=bool)

    tile_size = max(1, int(tile_size))
    out3 = a3.copy()
    for r0 in range(0, H, tile_size):
        r1 = min(H, r0 + tile_size)
        for c0 in range(0, W, tile_size):
            c1 = min(W, c0 + tile_size)
            cell_valid = valid_full[r0:r1, c0:c1]
            if not cell_valid.any():
                continue  # nothing eligible here -- leave the (already-copied) original
            cell = a3[r0:r1, c0:c1, :]
            mu_source, sd_source = _band_stats(cell, cell_valid)
            mu_gain = rng.uniform(1.0 - mu_jitter, 1.0 + mu_jitter, size=C)
            sd_gain = rng.uniform(1.0 - sd_jitter, 1.0 + sd_jitter, size=C)
            mu_target = mu_source * mu_gain
            sd_target = np.maximum(sd_source * sd_gain, 1e-6)
            cell_keep_mask = ~cell_valid
            out3[r0:r1, c0:c1, :] = _linear_rescale(
                cell, mu_source, sd_source, mu_target, sd_target, keep_mask=cell_keep_mask)

    return out3[..., 0] if was_2d else out3


def _generate_illumination_mask(h, w, rng, gradient_type, min_illum, max_illum):
    """An (h, w) float32 illumination mask spanning [min_illum, max_illum],
    via one of three geometric constructions -- mirrors
    uneven_illumination_rgb() from
    https://github.com/Nathanborg/Cloud_shadow_correction:

    - "linear": a np.linspace band between min_illum/max_illum over a
      randomly chosen column range; outside that range, clamped to the
      nearer endpoint value; the ramp direction (left-to-right or reversed)
      is also randomized.
    - "circular": radial falloff from a random center point -- max_illum at
      the center, min_illum at the farthest image corner (a vignette).
    - "diagonal": a ramp from a randomly chosen corner (max_illum at that
      corner, min_illum at the opposite corner).
    """
    if gradient_type == "linear":
        c0 = int(rng.integers(0, max(1, w // 2)))
        c1 = int(rng.integers(c0 + 1, w + 1))
        band = np.linspace(min_illum, max_illum, c1 - c0, dtype=np.float32)
        if rng.uniform() < 0.5:
            band = band[::-1].copy()
        row = np.empty(w, dtype=np.float32)
        row[:c0] = band[0]
        row[c0:c1] = band
        row[c1:] = band[-1]
        mask = np.tile(row, (h, 1))

    elif gradient_type == "circular":
        cy = float(rng.uniform(0, h))
        cx = float(rng.uniform(0, w))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        corners = np.array([[0, 0], [0, w], [h, 0], [h, w]], dtype=np.float32)
        max_dist = float(np.max(np.sqrt(((corners - [cy, cx]) ** 2).sum(axis=1))))
        max_dist = max(max_dist, 1e-6)
        norm_dist = np.clip(dist / max_dist, 0.0, 1.0)
        mask = (max_illum - (max_illum - min_illum) * norm_dist).astype(np.float32)

    elif gradient_type == "diagonal":
        corner = int(rng.integers(0, 4))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ny = yy / max(1, h - 1)
        nx = xx / max(1, w - 1)
        if corner == 0:      # top-left bright
            t = 1.0 - np.clip((ny + nx) / 2.0, 0.0, 1.0)
        elif corner == 1:    # top-right bright
            t = 1.0 - np.clip((ny + (1.0 - nx)) / 2.0, 0.0, 1.0)
        elif corner == 2:    # bottom-left bright
            t = 1.0 - np.clip(((1.0 - ny) + nx) / 2.0, 0.0, 1.0)
        else:                 # bottom-right bright
            t = 1.0 - np.clip(((1.0 - ny) + (1.0 - nx)) / 2.0, 0.0, 1.0)
        mask = (min_illum + (max_illum - min_illum) * t).astype(np.float32)

    else:
        raise ValueError(f"Unknown gradient_type: {gradient_type!r}")

    lo, hi = min(min_illum, max_illum), max(min_illum, max_illum)
    return np.clip(mask, lo, hi).astype(np.float32)


def apply_shadow_illumination(img, rng, smoothness=75, nodata_mask=None):
    """Multiplicative illumination/shadow augmentation. Draws
    max_illumination ~ U(0.7, 1.0), min_illumination ~ U(0.1, max_illumination)
    (matching the referenced repo's `generate_noisy_images`), picks one of the
    three gradient types uniformly at random, builds the mask via
    `_generate_illumination_mask`, Gaussian-blurs it for feathering
    (`cv2.GaussianBlur`, kernel forced odd -- `cv2.GaussianBlur` REQUIRES an
    odd kernel size; a caller-supplied even `smoothness` would otherwise crash
    here, not at options-dialog build time), then multiplies each channel by
    the blurred mask. NoData/masked pixels are restored to their original
    values afterward, same convention as the brightness functions.
    """
    a3, was_2d = _as_hwc(img)
    H, W, C = a3.shape

    max_illum = float(rng.uniform(0.7, 1.0))
    min_illum = float(rng.uniform(0.1, max_illum))
    gradient_type = str(rng.choice(["linear", "circular", "diagonal"]))
    mask = _generate_illumination_mask(H, W, rng, gradient_type, min_illum, max_illum)

    k = int(smoothness)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    mask_blurred = cv2.GaussianBlur(mask, (k, k), 0)

    out3 = a3.copy()
    for c in range(C):
        out3[..., c] = cv2.multiply(a3[..., c], mask_blurred)

    if nodata_mask is not None:
        out3[nodata_mask] = a3[nodata_mask]

    return out3[..., 0] if was_2d else out3.astype(np.float32, copy=False)


def augment_image_for_training(img, cfg, rng, nodata_mask=None):
    """The single augmentation entry point -- both `train_models` and the
    live preview dialog must call ONLY this, never the individual stage
    functions directly, so the two can never silently drift apart.

    cfg: the "augmentation" sub-dict of get_options() (see
    ml_augmentation_options_dialog.py), e.g.
    {"enabled": bool,
     "brightness": {"mode": "none"|"image"|"patch", "tile_size": int,
                     "mu_jitter": float, "sd_jitter": float},
     "shadow": {"enabled": bool, "smoothness": int},
     "row_policy": "add"|"replace", "n_variants": int}
    (row_policy/n_variants are read by the training loop, not by this
    function -- this function only cares about brightness/shadow).

    Applies brightness (if mode != "none") THEN shadow (if enabled), in that
    fixed order, composing sequentially when both are on -- realizes the
    "two independent axes" as independently-toggleable stages of ONE
    augmented output, not two separate augmented copies. Returns a new
    array; img is never mutated.
    """
    out = np.asarray(img, dtype=np.float32)
    cfg = cfg or {}

    b_cfg = cfg.get("brightness") or {}
    mode = (b_cfg.get("mode") or "none").lower()
    if mode == "image":
        out = apply_brightness_image_level(
            out, rng, nodata_mask=nodata_mask,
            mu_jitter=float(b_cfg.get("mu_jitter", 0.15)),
            sd_jitter=float(b_cfg.get("sd_jitter", 0.25)))
    elif mode == "patch":
        out = apply_brightness_patch_level(
            out, rng, int(b_cfg.get("tile_size", 64)), nodata_mask=nodata_mask,
            mu_jitter=float(b_cfg.get("mu_jitter", 0.15)),
            sd_jitter=float(b_cfg.get("sd_jitter", 0.25)))
    elif mode not in ("none", ""):
        raise ValueError(f"Unknown brightness mode: {mode!r}")

    s_cfg = cfg.get("shadow") or {}
    if s_cfg.get("enabled"):
        out = apply_shadow_illumination(
            out, rng, smoothness=int(s_cfg.get("smoothness", 75)), nodata_mask=nodata_mask)

    return out


def assemble_pixel_rows(pristine_ok, pristine_row, augmented_results, row_policy):
    """Decide which row(s) to add to the TRAINING set for ONE sampled pixel,
    given its pristine row and its augmented variants -- the policy at the
    heart of "add vs replace" and "never augment the report/holdout split"
    (see Decision 0.1 in the ML augmentation plan: this is intentionally
    SEPARATE from whatever goes into the report/holdout row list, which the
    caller always builds unconditionally from `pristine_row` alone, on every
    call, regardless of augmentation).

    pristine_ok: bool -- whether the pristine row itself was valid. If False,
        there is nothing to train on for this pixel and this function returns
        an empty list unconditionally (augmentation cannot rescue a pixel
        that failed for reasons unrelated to augmentation, e.g. a genuinely
        missing band).
    pristine_row: list[float] -- the pixel's un-augmented feature row.
    augmented_results: list of (ok: bool, row: list[float]) tuples, one per
        generated augmented variant (may be empty if augmentation produced
        zero variants, e.g. n_variants misconfigured to 0).
    row_policy: "add" | "replace".
        "add"     -> [pristine_row] + every row where ok was True (the
                     original pixel plus all of its successful augmented
                     siblings -- this is what grows the sample count).
        "replace" -> [the first row where ok was True], or [pristine_row] as
                     a fallback if every augmented variant failed for this
                     pixel (so a pixel is never silently dropped just because
                     its one augmented attempt happened to fail) -- always
                     exactly one row, so replace mode never changes the
                     training set's size, however many variants were tried.

    Returns a list of `row` (list[float]) to append to the training
    collection -- callers pair each with the same class label.
    """
    if not pristine_ok:
        return []
    ok_rows = [row for (ok, row) in augmented_results if ok]
    if row_policy == "replace":
        return [ok_rows[0]] if ok_rows else [pristine_row]
    # "add" (default/fallback policy)
    return [pristine_row] + ok_rows
