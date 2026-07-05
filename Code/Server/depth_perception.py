"""
depth_perception.py – monocular depth → free-space + clear direction (server side).

YOLO only knows its 80 classes and V-JEPA 2 gives a single global risk scalar, so
neither can say "there is a wall 0.4 m ahead and the left is open". A monocular
depth model (Depth-Anything V2) fills that gap: it produces a per-pixel depth map
for ANY scene (walls, doors, furniture — class-agnostic), from which we derive:

  • clear_distance_m – nearest obstacle straight ahead (metric), for the governor
  • clear_direction  – LEFT / CENTER / RIGHT, whichever has the most open space,
                       giving REROUTE an actual direction to turn toward.

It complements the ultrasonic (narrow forward cone, misses angled walls) and the
YOLO label (what the obstacle is, when it's a known class).

Falls back to a stub (buffer_ready=False → the pipeline ignores it) when
transformers / the checkpoint are unavailable, matching world_model/ssv2. Run it
where the model actually loads (native GPU/MPS box) for real depth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    clear_distance_m: float = -1.0        # nearest obstacle straight ahead (m); -1 = unknown
    clear_direction: str = "CENTER"       # LEFT | CENTER | RIGHT (most open)
    region_distances_m: dict = field(default_factory=dict)  # {"LEFT":.., "CENTER":.., "RIGHT":..}
    buffer_ready: bool = False            # False when using the stub / no depth
    is_stub: bool = False


def freespace_from_depth(
    depth_m: np.ndarray,
    near_percentile: float = 15.0,
    path_band_frac: float = 0.6,
    direction_margin_m: float = 0.3,
    max_range_m: float = 5.0,
) -> DepthResult:
    """Turn a metric depth map (HxW, metres) into free-space + clear direction.

    We look at the lower `path_band_frac` of the frame (the ground path ahead),
    split it into LEFT/CENTER/RIGHT thirds, and take a low percentile of each as
    that direction's nearest-obstacle distance (robust to a few noisy pixels).
    The clear direction is the most open third; CENTER is preferred unless a side
    is at least `direction_margin_m` more open (avoids needless turning).
    """
    h, w = depth_m.shape[:2]
    y0 = int(h * (1.0 - path_band_frac))
    band = np.clip(depth_m[y0:h, :], 0.0, max_range_m)
    thirds = {
        "LEFT": band[:, : w // 3],
        "CENTER": band[:, w // 3: 2 * w // 3],
        "RIGHT": band[:, 2 * w // 3:],
    }
    dists = {}
    for name, region in thirds.items():
        vals = region[np.isfinite(region)]
        dists[name] = float(np.percentile(vals, near_percentile)) if vals.size else max_range_m

    center = dists["CENTER"]
    # Pick the most open side; only prefer it over CENTER by a clear margin.
    best_side = max(("LEFT", "RIGHT"), key=lambda s: dists[s])
    if dists[best_side] > center + direction_margin_m:
        direction = best_side
    else:
        direction = "CENTER"

    return DepthResult(
        clear_distance_m=round(center, 3),
        clear_direction=direction,
        region_distances_m={k: round(v, 3) for k, v in dists.items()},
        buffer_ready=True,
        is_stub=False,
    )


class DepthEstimator:
    def __init__(self, cfg: dict):
        d = cfg.get("depth", {}) or {}
        self._enabled = bool(d.get("enabled", True))
        self._model_id = d.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
        self._device_str = d.get("device", "auto")
        self._run_every = max(1, int(d.get("run_every_n_frames", 6)))
        self._max_range_m = float(d.get("max_range_m", 5.0))
        self._near_pct = float(d.get("near_percentile", 15.0))
        self._path_band = float(d.get("path_band_frac", 0.6))
        self._dir_margin = float(d.get("direction_margin_m", 0.3))

        self._model = None
        self._processor = None
        self._device = None
        self._call_count = 0
        self._last = DepthResult()

    def load(self) -> None:
        if not self._enabled:
            logger.info("Depth channel disabled in config – skipping")
            self._model = None
            return
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
            from device_utils import is_gpu, resolve_device
            self._device, dev_name = resolve_device(self._device_str)
            if not is_gpu(dev_name):
                # Depth on CPU is slow; run it less often.
                self._run_every = max(self._run_every, 12)
            self._processor = AutoImageProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()
            logger.info("Depth model loaded: %s on %s (every %d frames)",
                        self._model_id, dev_name, self._run_every)
        except Exception as exc:
            logger.warning("Depth model unavailable (%s) – depth channel off "
                           "(ultrasonic + V-JEPA 2 still active)", exc)
            self._model = None

    def estimate(self, frame_rgb: np.ndarray) -> DepthResult:
        """Return free-space + clear direction for the frame (cached between runs)."""
        self._call_count += 1
        if self._model is None:
            # Stub: no depth → buffer_ready False so the pipeline ignores it.
            self._last = DepthResult(is_stub=True, buffer_ready=False)
            return self._last
        if self._call_count % self._run_every != 0 and self._last.buffer_ready:
            return self._last
        try:
            depth_m = self._infer_depth_m(frame_rgb)
            self._last = freespace_from_depth(
                depth_m, self._near_pct, self._path_band, self._dir_margin, self._max_range_m,
            )
        except Exception as exc:
            logger.debug("Depth inference error: %s", exc)
            return self._last
        return self._last

    def _infer_depth_m(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch
        inputs = self._processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = self._model(**inputs).predicted_depth  # (1, H, W), metres for metric models
        return pred.squeeze(0).cpu().numpy().astype(np.float32)
