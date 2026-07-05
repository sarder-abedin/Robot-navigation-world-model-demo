"""
world_model.py – V-JEPA 2 as a predictive world model (server side).

How it works in this pipeline
──────────────────────────────
V-JEPA 2 is a Joint-Embedding Predictive Architecture that predicts future
*latent representations* without pixel reconstruction.

We exploit this by:
1. Feeding the last clip_length frames (rolling buffer) as context.
2. Masking the final prediction_horizon frames so the predictor must imagine
   the near future (0.5–1 s ahead at typical camera fps).
3. Comparing the predicted future embedding against two anchors:
   - obstacle_anchor  ← average latent of "corridor blocked" scenes
   - clear_anchor     ← average latent of "corridor free" scenes
4. The cosine-similarity difference gives predicted_risk ∈ [0,1].

Key insight: the robot can slow down *before* the blocker fully enters frame
because V-JEPA 2 sees a person entering at the edge and predicts the blocked
latent for the near future.  A purely reactive system only reacts when the
obstacle is already centered.

Fallback: when the HuggingFace model is unavailable (no internet, no large
RAM), _StubEncoder kicks in.  It still produces *meaningful* risk scores
derived from scene pixel statistics, so the full pipeline exercises correctly.

API
───
  wm = WorldModel(cfg)
  wm.load()
  wm.predict(clip: list[np.ndarray]) -> WorldModelResult
  wm.build_anchors(obstacle_frames, clear_frames)   # optional recalibration
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorldModelResult:
    predicted_risk: float = 0.0
    similarity_to_obstacle: float = 0.0
    similarity_to_clear: float = 0.0
    label: str = "UNKNOWN"
    buffer_ready: bool = False


class WorldModel:
    def __init__(self, cfg: dict):
        wm_cfg = cfg["world_model"]
        self._model_id = wm_cfg["model_id"]
        self._clip_len = cfg["camera"]["clip_length"]
        self._horizon = wm_cfg["prediction_horizon"]
        self._input_size = wm_cfg["input_size"]
        self._risk_thresh = wm_cfg["risk_similarity_threshold"]
        # Min gap between obstacle- and clear-similarity to commit to a label
        # (below it → MIXED). Relative test, robust to uncalibrated anchors.
        self._label_margin = float(wm_cfg.get("label_margin", 0.02))
        self._device_str = wm_cfg.get("device", "cpu")
        self._run_every = wm_cfg.get("run_every_n_frames", 8)
        # Path to calibrated corridor anchors (built by calibrate_anchors.py from
        # real "blocked"/"clear" frames). When set + present they replace the
        # synthetic anchors, making BLOCKED/CLEAR reflect *your* environment.
        self._anchors_path = wm_cfg.get("anchors_path", "") or ""

        self._model = None
        self._device = None
        self._obstacle_anchor: np.ndarray | None = None
        self._clear_anchor: np.ndarray | None = None
        self._call_count = 0
        self._last_result = WorldModelResult()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch  # type: ignore
        from device_utils import resolve_device
        self._device, dev_name = resolve_device(self._device_str)
        try:
            self._load_vjepa2()
            logger.info("V-JEPA 2 loaded: %s on %s", self._model_id, dev_name)
        except Exception as exc:
            logger.warning("V-JEPA 2 load failed (%s) – using stub encoder", exc)
            self._model = _StubEncoder(embed_dim=1024)
        # Prefer calibrated corridor anchors when available; else synthetic.
        if self._anchors_path and os.path.exists(self._anchors_path):
            try:
                self.load_anchors(self._anchors_path)
                logger.info("V-JEPA 2 anchors loaded from %s (calibrated)", self._anchors_path)
            except Exception as exc:
                logger.warning("Anchor load failed (%s) – using synthetic anchors", exc)
                self._init_anchors()
        else:
            if self._anchors_path:
                logger.warning("anchors_path %s not found – using synthetic anchors "
                               "(run calibrate_anchors.py for your corridor)", self._anchors_path)
            self._init_anchors()

    def save_anchors(self, path: str) -> None:
        """Persist the current obstacle/clear anchors to an .npz file."""
        if self._obstacle_anchor is None or self._clear_anchor is None:
            raise RuntimeError("anchors not built yet")
        np.savez(path, obstacle=self._obstacle_anchor, clear=self._clear_anchor)
        logger.info("Saved V-JEPA 2 anchors → %s", path)

    def load_anchors(self, path: str) -> None:
        """Load obstacle/clear anchors from an .npz file (see save_anchors)."""
        data = np.load(path)
        self._obstacle_anchor = data["obstacle"]
        self._clear_anchor = data["clear"]

    def predict(self, clip: list[np.ndarray]) -> WorldModelResult:
        """
        clip: list of clip_length uint8 RGB frames (H, W, 3), already resized.
        Returns WorldModelResult with predicted_risk in [0,1].

        Expensive inference is skipped on intermediate frames; the last result
        is returned instead to match the configured run_every_n_frames cadence.
        """
        self._call_count += 1

        if len(clip) < self._clip_len:
            return WorldModelResult(buffer_ready=False)

        if self._call_count % self._run_every != 0:
            return self._last_result

        import torch  # type: ignore

        stack = self._preprocess_clip(clip)  # (T, C, H, W)
        emb = self._predict_future(stack)

        sim_obs = float(_cosine_sim(emb, self._obstacle_anchor))
        sim_clr = float(_cosine_sim(emb, self._clear_anchor))
        predicted_risk = _sigmoid_scale(sim_obs - sim_clr)

        # Label from the RELATIVE similarity (which anchor the scene is closer to),
        # not an absolute threshold. An absolute test (sim_obs > thresh) sticks on
        # BLOCKED with the synthetic/uncalibrated anchors, because a real indoor
        # scene is cosine-close to the grey "obstacle" anchor regardless. The
        # relative test matches how predicted_risk is already computed.
        diff = sim_obs - sim_clr
        if diff > self._label_margin:
            label = "BLOCKED"
        elif diff < -self._label_margin:
            label = "CLEAR"
        else:
            label = "MIXED"

        self._last_result = WorldModelResult(
            predicted_risk=predicted_risk,
            similarity_to_obstacle=sim_obs,
            similarity_to_clear=sim_clr,
            label=label,
            buffer_ready=True,
        )
        return self._last_result

    def build_anchors(
        self,
        obstacle_frames: list[np.ndarray],
        clear_frames: list[np.ndarray],
    ) -> None:
        obs_embs = [self._embed_single(f) for f in obstacle_frames]
        clr_embs = [self._embed_single(f) for f in clear_frames]
        self._obstacle_anchor = np.mean(obs_embs, axis=0)
        self._clear_anchor = np.mean(clr_embs, axis=0)
        logger.info(
            "Anchors rebuilt: %d obstacle / %d clear frames",
            len(obstacle_frames), len(clear_frames),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_vjepa2(self) -> None:
        from transformers import AutoModel, AutoProcessor  # type: ignore
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

    def _init_anchors(self) -> None:
        """Bootstrap with synthetic frames (grey wall vs open corridor)."""
        obs = np.full((self._input_size, self._input_size, 3), 70, dtype=np.uint8)
        clr = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        s = self._input_size
        for y in range(s):
            for x in range(s):
                d = np.sqrt((x - s // 2) ** 2 + (y - s // 2) ** 2)
                v = int(max(0, 220 - d * 1.4))
                clr[y, x] = [v, v, v]
        self._obstacle_anchor = self._embed_single(obs)
        self._clear_anchor = self._embed_single(clr)
        logger.debug("Synthetic anchors initialised")

    def _preprocess_frame(self, rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 → (C,input_size,input_size) float32 ImageNet-normalised.

        Frames are resized to the model's expected resolution (input_size, e.g.
        256 for vitl-fpc64-256) — the checkpoint's patch embedding is fixed to
        that size, so feeding a different resolution corrupts the embedding.
        """
        import cv2
        if rgb.shape[0] != self._input_size or rgb.shape[1] != self._input_size:
            rgb = cv2.resize(rgb, (self._input_size, self._input_size),
                             interpolation=cv2.INTER_AREA)
        f = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        f = (f - mean) / std
        return f.transpose(2, 0, 1)

    @staticmethod
    def _sample_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
        """Evenly sample exactly n frames (the checkpoint's frames-per-clip)."""
        if len(frames) == n:
            return list(frames)
        idxs = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return [frames[i] for i in idxs]

    def _preprocess_clip(self, frames: list[np.ndarray]):
        import torch
        # V-JEPA 2 checkpoints are fixed to N frames per clip (fpc64 → 64). Sample
        # exactly that many so the temporal patch embedding matches the weights.
        frames = self._sample_frames(frames, self._clip_len)
        arr = np.stack([self._preprocess_frame(f) for f in frames], axis=0)
        return torch.from_numpy(arr).float().to(self._device)

    def _predict_future(self, clip) -> np.ndarray:
        """
        Run V-JEPA 2 context encoder with future frames masked.

        Input:  clip tensor (T, C, H, W)
        Output: predicted future embedding (D,)

        The masking forces the predictor to imagine what the scene will look
        like T→T+horizon steps ahead, which is the core of the world-model idea.
        """
        import torch

        with torch.no_grad():
            if isinstance(self._model, _StubEncoder):
                return self._model.encode(clip)

            # (1, T, C, H, W)
            pixel_values = clip.unsqueeze(0)
            T = pixel_values.shape[1]
            mask = torch.zeros(1, T, dtype=torch.bool, device=self._device)
            mask_start = max(0, T - self._horizon)
            mask[0, mask_start:] = True

            outputs = self._model(
                pixel_values_videos=pixel_values,
                bool_masked_pos=mask,
            )
            # Mean-pool sequence dim → (D,)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    def _embed_single(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(
            np.stack([self._preprocess_frame(rgb)] * self._clip_len, axis=0)
        ).float().to(self._device)
        return self._predict_future(tensor)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a.flatten(), b.flatten()) / denom, -1.0, 1.0))


def _sigmoid_scale(x: float, k: float = 5.0) -> float:
    return float(1.0 / (1.0 + np.exp(-k * x)))


class _StubEncoder:
    """
    Lightweight deterministic embedding stub for CPU-only / offline testing.

    Uses a fixed random projection of per-channel pixel statistics, giving a
    scene embedding that is *consistent* across frames (same scene → same vector)
    and therefore produces meaningful cosine similarities against the anchors.
    """

    def __init__(self, embed_dim: int = 1024):
        self._dim = embed_dim
        rng = np.random.default_rng(42)
        self._proj = rng.standard_normal((3, embed_dim)).astype(np.float32)

    def encode(self, clip) -> np.ndarray:
        # Average channel values over time and space
        import torch
        mean_px = clip.mean(dim=[0, 2, 3]).cpu().numpy()   # (C,)
        emb = mean_px @ self._proj                          # (D,)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
