"""
feature_viz.py – PCA visualisation of V-JEPA 2 dense patch features.

Turns the model's per-patch encoder features into the colourful "what V-JEPA 2
sees" image from the V-JEPA 2 paper (Fig. 1): project the D-dim patch features
onto their top-3 principal components and map those to RGB, so semantically
similar patches (road, sky, a limb, a wall) share a colour.

Pure NumPy — no torch, no PyQt — so the colour mapping is unit-testable. The
world model calls this in its subprocess with the patch features; the resulting
small RGB grid is upscaled and drawn beside the camera in the HUD.
"""
from __future__ import annotations

import numpy as np


def patch_features_to_rgb(feats, grid_hw, prev_basis=None, saturation=0.65):
    """(N, D) patch features → ((H, W, 3) uint8 RGB, basis).

    N must equal H*W (the spatial patch grid). PCA's top-3 components become the
    R/G/B channels. `prev_basis` (D, 3) from the previous frame is used to
    sign-align this frame's components so the colours don't flip/flicker over
    time; the returned basis should be fed back in on the next call. On a bad
    shape or a numerical failure returns (None, prev_basis) so the caller can
    just skip the overlay for that frame.

    `saturation` (0..1) pulls the colours toward their luminance so the result
    reads as coherent regions instead of a fully-saturated rainbow — 1.0 keeps
    the raw PCA colours, lower is calmer/more legible (0.65 default).
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    x = np.asarray(feats, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != h * w or x.shape[1] < 3:
        return None, prev_basis

    xc = x - x.mean(axis=0, keepdims=True)
    try:
        # top-3 right singular vectors = principal directions of the patches
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, prev_basis
    basis = np.ascontiguousarray(vt[:3].T)          # (D, 3)

    # Temporal stability: PCA components are sign-ambiguous, which makes the
    # colours flip frame-to-frame. Align this frame's signs to the previous
    # frame's (dot < 0 → flip); on the first frame use a deterministic rule
    # (largest-magnitude element positive) so a run is reproducible.
    if prev_basis is not None and getattr(prev_basis, "shape", None) == basis.shape:
        for i in range(3):
            if float(np.dot(basis[:, i], prev_basis[:, i])) < 0:
                basis[:, i] = -basis[:, i]
    else:
        for i in range(3):
            j = int(np.argmax(np.abs(basis[:, i])))
            if basis[j, i] < 0:
                basis[:, i] = -basis[:, i]

    proj = xc @ basis                                # (N, 3)
    # Robust per-channel normalisation to [0,1] (2–98 pct clips outliers).
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    rng = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
    rgb = np.clip((proj - lo) / rng, 0.0, 1.0)       # (N, 3) float

    # Desaturate toward luminance so the map reads as coherent regions rather
    # than a fully-saturated rainbow (the raw top-3-PCA→RGB is very garish).
    s = float(np.clip(saturation, 0.0, 1.0))
    if s < 1.0:
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)   # (N,)
        rgb = lum[:, None] + s * (rgb - lum[:, None])
        rgb = np.clip(rgb, 0.0, 1.0)

    rgb = (rgb * 255.0).astype(np.uint8).reshape(h, w, 3)
    return rgb, basis


def infer_patch_grid(num_tokens: int, input_size: int, patch_size: int = 16):
    """Best-effort (rows, cols) spatial patch grid for a V-JEPA 2 token sequence.

    Tokens are temporal×spatial; the spatial side is (input_size/patch_size)². If
    the token count isn't a whole number of such spatial planes, returns None so
    the caller skips the overlay rather than reshaping garbage.
    """
    if input_size <= 0 or patch_size <= 0:
        return None
    side = input_size // patch_size
    plane = side * side
    if plane <= 0 or num_tokens <= 0 or num_tokens % plane != 0:
        return None
    return side, side
