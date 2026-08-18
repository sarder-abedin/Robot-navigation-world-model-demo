"""
test_feature_viz_rpi.py – PCA dense-feature → RGB mapping for the V-JEPA 2 view.

Covers feature_viz.patch_features_to_rgb (shape/dtype/range, determinism, temporal
sign-alignment) and infer_patch_grid. Pure NumPy — no torch/GPU/PyQt.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import feature_viz as fv


def _feats(n=256, d=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_output_shape_dtype_range():
    rgb, basis = fv.patch_features_to_rgb(_feats(256, 32), (16, 16))
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255
    assert basis.shape == (32, 3)


def test_deterministic():
    a, _ = fv.patch_features_to_rgb(_feats(256, 32, seed=1), (16, 16))
    b, _ = fv.patch_features_to_rgb(_feats(256, 32, seed=1), (16, 16))
    assert np.array_equal(a, b)


def test_bad_shape_returns_none():
    # N must equal H*W.
    rgb, basis = fv.patch_features_to_rgb(_feats(255, 32), (16, 16), prev_basis="keep")
    assert rgb is None and basis == "keep"
    # too few feature dims for 3 components
    rgb2, _ = fv.patch_features_to_rgb(_feats(256, 2), (16, 16))
    assert rgb2 is None


def test_sign_alignment_is_stable():
    """Feeding the previous basis back keeps the colours from flipping: the same
    features processed with the returned basis reproduce the same image."""
    x = _feats(256, 32, seed=2)
    rgb1, basis1 = fv.patch_features_to_rgb(x, (16, 16))
    # Next frame = same scene → with basis1 fed back, output matches frame 1.
    rgb2, basis2 = fv.patch_features_to_rgb(x, (16, 16), prev_basis=basis1)
    assert np.array_equal(rgb1, rgb2)
    # And the aligned basis columns point the same way as the previous frame.
    for i in range(3):
        assert float(np.dot(basis2[:, i], basis1[:, i])) > 0


def test_sign_alignment_flips_negated_components():
    x = _feats(256, 32, seed=3)
    _, basis = fv.patch_features_to_rgb(x, (16, 16))
    # Give a prev_basis that is the negation → the function should flip back so
    # the result aligns (positive dot with the prev/target).
    flipped = -basis
    _, basis2 = fv.patch_features_to_rgb(x, (16, 16), prev_basis=flipped)
    for i in range(3):
        assert float(np.dot(basis2[:, i], flipped[:, i])) > 0


def test_saturation_reduces_colourfulness():
    """Lower saturation pulls colours toward luminance → smaller channel spread
    (less garish) while keeping the same shape/range and staying deterministic."""
    x = _feats(256, 32, seed=7)
    full, _ = fv.patch_features_to_rgb(x, (16, 16), saturation=1.0)
    calm, _ = fv.patch_features_to_rgb(x, (16, 16), saturation=0.3)
    assert calm.shape == full.shape and calm.dtype == np.uint8
    # per-pixel spread across R/G/B channels shrinks when desaturated
    spread_full = np.ptp(full.astype(np.int16), axis=2).mean()
    spread_calm = np.ptp(calm.astype(np.int16), axis=2).mean()
    assert spread_calm < spread_full
    # saturation=0 → (near-)grey: channels nearly equal per pixel
    grey, _ = fv.patch_features_to_rgb(x, (16, 16), saturation=0.0)
    assert np.ptp(grey.astype(np.int16), axis=2).max() <= 2


def test_infer_patch_grid():
    # vitl-fpc64-256: 256/16 = 16 → 256 spatial patches; 8192 tokens = 32 temporal.
    assert fv.infer_patch_grid(8192, 256, 16) == (16, 16)
    assert fv.infer_patch_grid(256, 256, 16) == (16, 16)   # single temporal plane
    # Not a whole number of spatial planes → None (skip the overlay).
    assert fv.infer_patch_grid(8000, 256, 16) is None
    assert fv.infer_patch_grid(0, 256, 16) is None
    assert fv.infer_patch_grid(8192, 0, 16) is None
