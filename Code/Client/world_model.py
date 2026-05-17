"""
world_model.py – V-JEPA 2 as a predictive world model (client side).

Runs on the operator PC / laptop where GPU headroom is available.
Identical logic to Code/Server/world_model.py — split here so the client
can import it without touching the Pi server's Python path.

See Code/Server/world_model.py for full documentation.
"""

from __future__ import annotations

import logging
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
        self._device_str = wm_cfg.get("device", "cpu")
        self._run_every = wm_cfg.get("run_every_n_frames", 8)

        self._model = None
        self._device = None
        self._obstacle_anchor: np.ndarray | None = None
        self._clear_anchor: np.ndarray | None = None
        self._call_count = 0
        self._last_result = WorldModelResult()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch  # type: ignore
        self._device = torch.device(self._device_str)
        try:
            self._load_vjepa2()
            logger.info("V-JEPA 2 loaded: %s on %s", self._model_id, self._device)
        except Exception as exc:
            logger.warning("V-JEPA 2 load failed (%s) – using stub encoder", exc)
            self._model = _StubEncoder(embed_dim=1024)
        self._init_anchors()

    def predict(self, clip: list[np.ndarray]) -> WorldModelResult:
        """
        clip: list of clip_length uint8 RGB frames (H, W, 3).
        Returns WorldModelResult with predicted_risk in [0,1].
        """
        self._call_count += 1

        if len(clip) < self._clip_len:
            return WorldModelResult(buffer_ready=False)

        if self._call_count % self._run_every != 0:
            return self._last_result

        import torch  # type: ignore

        stack = self._preprocess_clip(clip)
        emb = self._predict_future(stack)

        sim_obs = float(_cosine_sim(emb, self._obstacle_anchor))
        sim_clr = float(_cosine_sim(emb, self._clear_anchor))
        predicted_risk = _sigmoid_scale(sim_obs - sim_clr)

        if sim_obs > self._risk_thresh:
            label = "BLOCKED"
        elif sim_clr > self._risk_thresh:
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

    def _preprocess_frame(self, rgb: np.ndarray) -> np.ndarray:
        f = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        f = (f - mean) / std
        return f.transpose(2, 0, 1)

    def _preprocess_clip(self, frames: list[np.ndarray]):
        import torch
        arr = np.stack([self._preprocess_frame(f) for f in frames], axis=0)
        return torch.from_numpy(arr).float().to(self._device)

    def _predict_future(self, clip) -> np.ndarray:
        import torch
        with torch.no_grad():
            if isinstance(self._model, _StubEncoder):
                return self._model.encode(clip)

            pixel_values = clip.unsqueeze(0)
            T = pixel_values.shape[1]
            mask = torch.zeros(1, T, dtype=torch.bool, device=self._device)
            mask_start = max(0, T - self._horizon)
            mask[0, mask_start:] = True

            outputs = self._model(
                pixel_values=pixel_values,
                bool_masked_pos=mask,
            )
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
    """Lightweight deterministic stub when V-JEPA 2 weights are unavailable."""

    def __init__(self, embed_dim: int = 1024):
        self._dim = embed_dim
        rng = np.random.default_rng(42)
        self._proj = rng.standard_normal((3, embed_dim)).astype(np.float32)

    def encode(self, clip) -> np.ndarray:
        import torch
        mean_px = clip.mean(dim=[0, 2, 3]).cpu().numpy()
        emb = mean_px @ self._proj
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
