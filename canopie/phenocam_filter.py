"""
phenocam_filter.py - Image Similarity Filter for CanoPie
=========================================================
Filters images by illumination and homogeneity similarity
against a user-chosen reference image.

Usage:
    from phenocam_filter import PhenocamFilter, batch_extract_features

    # From file paths (multithreaded by default)
    pf = PhenocamFilter(reference_path="ref.jpg")
    results = pf.rank_from_paths(image_paths) 
    selected, rejected = pf.apply_threshold(results, threshold=0.85)

    # Advanced: Pre-compute features (useful for multi-reference)
    feats = batch_extract_features(image_paths, resize_max=400)
    results = pf.rank_from_features(feats)
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging
import concurrent.futures
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Features
# =============================================================================

@dataclass
class ImageFeatures:
    path: str
    mean_L: float = 0.0
    std_L: float = 0.0
    mean_a: float = 0.0
    mean_b: float = 0.0
    ratio_rg: float = 0.0
    ratio_rb: float = 0.0
    ratio_gb: float = 0.0
    hist_L: np.ndarray = field(default_factory=lambda: np.zeros(32))
    spatial_variance: float = 0.0
    entropy: float = 0.0
    glcm_contrast: float = 0.0
    glcm_homogeneity: float = 0.0
    glcm_energy: float = 0.0
    glcm_correlation: float = 0.0
    coeff_variation: float = 0.0

    FEATURE_NAMES = [
        'mean_L', 'std_L', 'mean_a', 'mean_b',
        'ratio_rg', 'ratio_rb', 'ratio_gb',
        'spatial_variance', 'entropy',
        'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
        'glcm_correlation', 'coeff_variation',
    ]

    _vec_cache: np.ndarray = field(default=None, repr=False, compare=False)

    def to_vector(self) -> np.ndarray:
        if self._vec_cache is not None:
            return self._vec_cache
        v = np.array([self.mean_L, self.std_L, self.mean_a, self.mean_b,
                       self.ratio_rg, self.ratio_rb, self.ratio_gb,
                       self.spatial_variance, self.entropy,
                       self.glcm_contrast, self.glcm_homogeneity,
                       self.glcm_energy, self.glcm_correlation,
                       self.coeff_variation])
        self._vec_cache = v
        return v


# =============================================================================
# GLCM (no skimage dependency) — vectorized with np.bincount
# =============================================================================

# Precomputed angle offsets: (dy, dx) for d=1, angles 0, 45, 90, 135
_GLCM_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _compute_glcm(gray: np.ndarray, levels: int = 32) -> Tuple[float, float, float, float]:
    """Compute GLCM features using np.bincount (much faster than np.add.at)."""
    quantized = np.clip((gray * (levels / 256.0)).astype(np.int32), 0, levels - 1)
    rows, cols = quantized.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    lsq = levels * levels

    for dy, dx in _GLCM_OFFSETS:
        r0, r1 = max(0, -dy), min(rows, rows - dy)
        c0, c1 = max(0, -dx), min(cols, cols - dx)
        if r0 >= r1 or c0 >= c1:
            continue
        a = quantized[r0:r1, c0:c1].ravel()
        b = quantized[r0 + dy:r1 + dy, c0 + dx:c1 + dx].ravel()
        # Flat index into levels×levels matrix
        flat = a * levels + b
        counts = np.bincount(flat, minlength=lsq)
        glcm.ravel()[:] += counts[:lsq]

    total = glcm.sum()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    glcm /= total

    # Precompute i,j grids
    i_arr = np.arange(levels, dtype=np.float64)
    # diff matrix (i - j) for each cell
    diff = i_arr[:, None] - i_arr[None, :]

    contrast = float(np.sum(glcm * diff * diff))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(diff))))
    energy = float(np.sum(glcm * glcm))

    # Marginal means and stds
    i_grid = i_arr[:, None]  # (levels, 1)
    j_grid = i_arr[None, :]  # (1, levels)
    mu_i = float(np.sum(i_grid * glcm))
    mu_j = float(np.sum(j_grid * glcm))
    si = np.sqrt(float(np.sum(glcm * (i_grid - mu_i) ** 2)))
    sj = np.sqrt(float(np.sum(glcm * (j_grid - mu_j) ** 2)))
    if si > 1e-10 and sj > 1e-10:
        correlation = float(np.sum(glcm * (i_grid - mu_i) * (j_grid - mu_j)) / (si * sj))
    else:
        correlation = 0.0

    return contrast, homogeneity, energy, correlation


# =============================================================================
# Feature extraction
# =============================================================================

def _extract(img_bgr: np.ndarray, path: str = "",
             roi: Optional[Tuple[int, int, int, int]] = None,
             resize_max: int = 800) -> ImageFeatures:
    """Core extraction from a uint8 BGR image."""

    if roi is not None:
        y0, y1, x0, x1 = roi
        img_bgr = img_bgr[y0:y1, x0:x1]

    h, w = img_bgr.shape[:2]
    if resize_max > 0 and max(h, w) > resize_max:
        s = resize_max / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s,
                             interpolation=cv2.INTER_AREA)

    # LAB for illumination features
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    # RGB channel means directly from BGR (no full conversion needed)
    eps = 1e-6
    mB = float(np.mean(img_bgr[:, :, 0])) + eps
    mG = float(np.mean(img_bgr[:, :, 1])) + eps
    mR = float(np.mean(img_bgr[:, :, 2])) + eps

    f = ImageFeatures(path=path)

    # --- Illumination ---
    f.mean_L = float(np.mean(L))
    f.std_L = float(np.std(L))
    f.mean_a = float(np.mean(a_ch.astype(np.float64)))
    f.mean_b = float(np.mean(b_ch.astype(np.float64)))

    f.ratio_rg = mR / mG
    f.ratio_rb = mR / mB
    f.ratio_gb = mG / mB

    # Single 256-bin histogram, then rebin to 32 for similarity
    L_flat = L.ravel()
    hist256 = np.bincount(L_flat, minlength=256).astype(np.float64)
    hist32 = hist256.reshape(32, 8).sum(axis=1)
    total_px = hist32.sum() + eps
    f.hist_L = hist32 / total_px

    # --- Homogeneity ---
    # Block means via cv2.resize (single C call, replaces Python loop)
    hp, wp = L.shape
    n_blocks = 8
    if hp >= n_blocks and wp >= n_blocks:
        block_means = cv2.resize(
            L, (n_blocks, n_blocks), interpolation=cv2.INTER_AREA
        ).astype(np.float64).ravel()
        f.spatial_variance = float(np.var(block_means))
        bm_mean = np.mean(block_means)
        f.coeff_variation = float(np.std(block_means) / (bm_mean + eps))

    # Entropy from 256-bin histogram (already computed)
    p = hist256 / (hist256.sum() + eps)
    mask = p > 0
    f.entropy = float(-np.sum(p[mask] * np.log2(p[mask])))

    # GLCM (now uses bincount, 32 levels instead of 64)
    f.glcm_contrast, f.glcm_homogeneity, f.glcm_energy, f.glcm_correlation = \
        _compute_glcm(L)

    return f


def extract_from_path(path: str, **kw) -> ImageFeatures:
    # Load image — handle grayscale, 16-bit, and multi-band stacks
    img = None
    path_str = str(path)
    try:
        if path_str.isascii():
            img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        else:
            stream = np.fromfile(path_str, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
    except Exception:
        pass
    if img is None:
        try:
            stream = np.fromfile(path_str, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass
    if img is None:
        raise ValueError(f"Cannot read: {path}")

    # Multi-band stack: take first 3 bands
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # Convert non-uint8 (e.g. 16-bit TIFFs) to uint8 with percentile stretch
    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo, hi = np.percentile(finite, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            arr = (arr - lo) / (hi - lo) * 255.0
        img = np.clip(arr, 0, 255).astype(np.uint8)

    # Grayscale → fake BGR (needed for LAB conversion in _extract)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return _extract(img, path=path_str, **kw)


def extract_from_array(arr: np.ndarray, path: str = "", is_bgr: bool = False,
                       **kw) -> ImageFeatures:
    if not is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return _extract(arr, path=path, **kw)


# =============================================================================
# Batch Extraction (Multithreaded)
# =============================================================================

def batch_extract_features(paths: List[str],
                           roi=None,
                           resize_max=800,
                           progress_callback=None,
                           max_workers=None) -> List[ImageFeatures]:
    """
    Extract features from multiple images in parallel.

    Parameters
    ----------
    paths : list of str
        Image file paths.
    roi : tuple, optional
        Region of interest.
    resize_max : int
        Max dimension for processing.
    progress_callback : callable(done, total), optional
        Progress reporting.
    max_workers : int, optional
        Number of threads. Defaults to CPU count + 4.

    Returns
    -------
    List[ImageFeatures] — successful extractions in input order.
    """
    if not paths:
        return []

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) + 4)

    features_map = {}
    total = len(paths)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract_from_path, p, roi=roi, resize_max=resize_max): i
            for i, p in enumerate(paths)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            path = paths[idx]
            try:
                feat = future.result()
                features_map[idx] = feat
            except Exception as e:
                logger.warning(f"Error extracting features from {path}: {e}")

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    ordered_feats = []
    for i in range(total):
        if i in features_map:
            ordered_feats.append(features_map[i])

    return ordered_feats


# =============================================================================
# Similarity
# =============================================================================

def _bhattacharyya_batch(ref_hist: np.ndarray, hists: np.ndarray) -> np.ndarray:
    """Vectorized Bhattacharyya distance for N histograms at once.
    
    ref_hist : (32,)
    hists    : (N, 32)
    Returns  : (N,) distances
    """
    bc = np.clip(np.sum(np.sqrt(ref_hist[None, :] * hists), axis=1), 1e-10, 1.0)
    return -np.log(bc)


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    bc = np.clip(np.sum(np.sqrt(h1 * h2)), 1e-10, 1.0)
    return float(-np.log(bc))


# =============================================================================
# Main filter class
# =============================================================================

class PhenocamFilter:
    """
    Filters images by similarity to a reference.

    Parameters
    ----------
    reference_path : str, optional
        Path to reference image file.
    reference_array : np.ndarray, optional
        Reference as RGB float 0-1 array.  Provide one of the two.
    roi : tuple (y0, y1, x0, x1), optional
        Crop region for analysis (e.g. to exclude sky).
    weights : dict, optional
        Per-feature weights (see DEFAULT_WEIGHTS).
    hist_weight : float
        Weight for histogram distance (0-1).  Default 0.3.
    sigma : float
        Controls score decay speed.  Lower = stricter.  Default 2.0.
    resize_max : int
        Downscale longest edge to this for speed.  Default 800.
    """

    DEFAULT_WEIGHTS = {
        'mean_L': 3.0, 'std_L': 2.0,
        'mean_a': 1.5, 'mean_b': 1.5,
        'ratio_rg': 2.0, 'ratio_rb': 1.5, 'ratio_gb': 1.5,
        'spatial_variance': 2.5, 'entropy': 1.5,
        'glcm_contrast': 1.0, 'glcm_homogeneity': 1.5,
        'glcm_energy': 1.0, 'glcm_correlation': 0.5,
        'coeff_variation': 2.0,
    }

    def __init__(self, reference_path: str = None, reference_array: np.ndarray = None,
                 roi=None, weights=None, hist_weight=0.3, sigma=2.0,
                 resize_max=800):

        if reference_path is None and reference_array is None:
            raise ValueError("Provide reference_path or reference_array")

        self.roi = roi
        self.resize_max = resize_max
        self.hist_weight = hist_weight
        self.sigma = sigma
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._w_vec = np.array([self.weights.get(n, 1.0)
                                for n in ImageFeatures.FEATURE_NAMES])

        if reference_array is not None:
            self._ref = extract_from_array(reference_array, path="<reference>",
                                           roi=roi, resize_max=resize_max)
        else:
            self._ref = extract_from_path(reference_path, roi=roi,
                                          resize_max=resize_max)
        self._ref_vec = self._ref.to_vector()
        self._ref_hist = self._ref.hist_L

    # ----- scoring (single image, kept for API compat) -----

    def _score(self, feat: ImageFeatures, means: np.ndarray,
               stds: np.ndarray) -> float:
        safe = np.where(stds > 1e-10, stds, 1.0)
        r = (self._ref_vec - means) / safe
        c = (feat.to_vector() - means) / safe
        scalar_d = np.sqrt(np.sum((r - c) ** 2 * self._w_vec))
        hist_d = _bhattacharyya(self._ref_hist, feat.hist_L)
        d = (1 - self.hist_weight) * scalar_d + self.hist_weight * hist_d
        return float(np.exp(-d / self.sigma))

    # ----- vectorized scoring (all images at once) -----

    def _score_batch(self, feats: List[ImageFeatures],
                     means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Score all features in one vectorized pass. Returns (N,) scores."""
        safe = np.where(stds > 1e-10, stds, 1.0)
        r = (self._ref_vec - means) / safe                     # (14,)

        # Build NxF matrix of normalized feature vectors
        vecs = np.array([f.to_vector() for f in feats])         # (N, 14)
        c = (vecs - means) / safe                               # (N, 14)

        # Weighted Euclidean distance (vectorized)
        diff = r - c                                            # (N, 14)
        scalar_d = np.sqrt(np.sum(diff * diff * self._w_vec, axis=1))  # (N,)

        # Histogram distances (vectorized)
        hists = np.array([f.hist_L for f in feats])             # (N, 32)
        hist_d = _bhattacharyya_batch(self._ref_hist, hists)    # (N,)

        d = (1 - self.hist_weight) * scalar_d + self.hist_weight * hist_d
        return np.exp(-d / self.sigma)

    # ----- public API -----

    def rank_from_paths(self, paths: List[str],
                        progress_callback=None) -> List[Tuple[str, float]]:
        """
        Rank image files by similarity to reference.
        Multithreaded via batch_extract_features.
        """
        feats = batch_extract_features(
            paths, 
            roi=self.roi, 
            resize_max=self.resize_max, 
            progress_callback=progress_callback
        )
        return self._rank(feats)

    def rank_from_arrays(self, images: np.ndarray,
                         labels: List[str] = None) -> List[Tuple[str, float]]:
        """
        Rank in-memory arrays (N,H,W,3) RGB float 0-1.
        """
        n = len(images)
        if labels is None:
            labels = [f"image_{i:04d}" for i in range(n)]

        feats = []
        for i in range(n):
            try:
                feats.append(extract_from_array(
                    images[i], path=labels[i],
                    roi=self.roi, resize_max=self.resize_max))
            except Exception as e:
                logger.warning(f"Skipped {labels[i]}: {e}")

        return self._rank(feats)
        
    def rank_from_features(self, feats: List[ImageFeatures]) -> List[Tuple[str, float]]:
        """
        Rank from already extracted ImageFeatures objects.
        Useful for multi-reference scenarios where extraction happens once.
        """
        return self._rank(feats)

    def _rank(self, feats: List[ImageFeatures]) -> List[Tuple[str, float]]:
        if not feats:
            return []

        # Build matrix of feature vectors (vectorized)
        vecs = np.array([f.to_vector() for f in feats])    # (N, 14)
        all_vecs = np.vstack([vecs, self._ref_vec.reshape(1, -1)])
        means = np.mean(all_vecs, axis=0)
        stds = np.std(all_vecs, axis=0)

        # Score all at once
        scores = self._score_batch(feats, means, stds)

        # Build result list
        paths = [f.path for f in feats]
        results = list(zip(paths, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def apply_threshold(results: List[Tuple[str, float]],
                        threshold: float = 0.85):
        """
        Split results into selected / rejected.

        Returns (selected, rejected) — each a list of (path, score).
        """
        selected = [(p, s) for p, s in results if s >= threshold]
        rejected = [(p, s) for p, s in results if s < threshold]
        return selected, rejected

    # ----- diagnostics -----

    def plot_diagnostics(self, results: List[Tuple[str, float]],
                         threshold: float = 0.85,
                         category_fn=None):
        """
        Show score histogram + boxplot by category.
        """
        scores = [s for _, s in results]

        if category_fn is None:
            category_fn = lambda p: "all"

        cats = [category_fn(p) for p, _ in results]
        unique_cats = sorted(set(cats))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # --- Histogram ---
        ax1.hist(scores, bins=30, edgecolor='black', alpha=0.7,
                 color='steelblue')
        ax1.axvline(x=threshold, color='red', linestyle='--',
                    linewidth=2, label=f'threshold={threshold}')
        n_sel = sum(1 for s in scores if s >= threshold)
        ax1.set_title(f'Score Distribution  '
                      f'({n_sel}/{len(scores)} selected)')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Count')
        ax1.legend()

        # --- Boxplot ---
        cat_data = {c: [] for c in unique_cats}
        for c, (_, s) in zip(cats, results):
            cat_data[c].append(s)

        data = [cat_data[c] for c in unique_cats]
        labels_bp = [f"{c}\n(n={len(cat_data[c])})" for c in unique_cats]
        bp = ax2.boxplot(data, patch_artist=True)

        palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_cats), 1)))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_xticklabels(labels_bp)
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Scores by Category')
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Print stats
        print(f"\n{'Category':<15} {'Mean':>7} {'Std':>7} "
              f"{'Min':>7} {'Max':>7} {'N':>5}")
        print("-" * 52)
        for c in unique_cats:
            s = np.array(cat_data[c])
            print(f"{c:<15} {np.mean(s):7.4f} {np.std(s):7.4f} "
                  f"{np.min(s):7.4f} {np.max(s):7.4f} {len(s):5d}")
