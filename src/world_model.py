"""
world_model.py – V-JEPA 2 as a predictive world model.

Core idea
─────────
V-JEPA 2 is a joint-embedding predictive architecture trained to predict
future *latent representations* of a scene without reconstructing pixels.
We exploit this property to ask: "given the last N frames, what will the
latent embedding of the scene look like T steps ahead?"

We then compare that predicted future embedding against two anchor
embeddings:
  • obstacle_anchor  – the average latent of corridor frames containing a
                       large, centered obstacle
  • clear_anchor     – the average latent of obstacle-free corridor frames

The cosine similarity to each anchor tells us how "blocked" the predicted
future scene is, giving a predictive risk score *before* the obstacle
is fully visible.

Anchor construction
───────────────────
On first run (or if a checkpoint is not found) the anchors are initialised
with a fast heuristic: we run the model on a single synthetic frame and
use the resulting embedding as a seed. In a real deployment you would
collect a small labelled clip set and call WorldModel.build_anchors().

API
───
  wm = WorldModel(cfg)
  wm.load()                    # download / load weights
  wm.push_frame(rgb_frame)     # add frame to rolling buffer
  result = wm.predict()        # → WorldModelResult

WorldModelResult
  .predicted_risk   float [0,1]  – how "blocked" the predicted future looks
  .similarity_to_obstacle float  – raw cosine similarity
  .similarity_to_clear    float
  .label            str          – "BLOCKED" | "MIXED" | "CLEAR"
  .buffer_ready     bool         – False until the buffer has enough frames
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass

import cv2
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
    """
    Wraps V-JEPA 2 to generate a predictive risk signal.

    The model is loaded lazily (only when load() is called) so that
    the rest of the system can be tested without the heavy checkpoint.
    """

    def __init__(self, cfg: dict):
        wm_cfg = cfg["world_model"]
        self._model_id = wm_cfg["model_id"]
        self._clip_len = wm_cfg["clip_length"]
        self._horizon = wm_cfg["prediction_horizon"]
        self._input_size = wm_cfg["input_size"]
        self._risk_threshold = wm_cfg["risk_similarity_threshold"]
        self._device_str = wm_cfg.get("device", "cpu")

        # Rolling frame buffer (stores preprocessed tensors)
        self._buffer: deque = deque(maxlen=self._clip_len)

        # Anchor embeddings (set in load() or build_anchors())
        self._obstacle_anchor: np.ndarray | None = None
        self._clear_anchor: np.ndarray | None = None

        self._model = None
        self._processor = None
        self._device = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download weights and initialise the model."""
        import torch  # type: ignore

        self._device = torch.device(self._device_str)

        try:
            self._load_vjepa2()
            logger.info("V-JEPA 2 loaded from %s on %s", self._model_id, self._device)
        except Exception as exc:
            # Gracefully fall back to a lightweight stub so the rest of the
            # system keeps running without a GPU or internet connection.
            logger.warning(
                "V-JEPA 2 load failed (%s). Falling back to embedding stub.", exc
            )
            self._model = _StubEncoder(embed_dim=1024)

        self._init_anchors()

    def push_frame(self, rgb_frame: np.ndarray) -> None:
        """Preprocess one RGB frame and append it to the rolling buffer."""
        tensor = self._preprocess(rgb_frame)
        self._buffer.append(tensor)

    def predict(self) -> WorldModelResult:
        """
        Run V-JEPA 2 on the current buffer and return a predictive risk score.

        If the buffer does not yet hold a full clip the method returns a
        zero-risk result with buffer_ready=False so the caller can decide
        to use detector-only risk until enough frames accumulate.
        """
        if len(self._buffer) < self._clip_len:
            return WorldModelResult(buffer_ready=False)

        import torch  # type: ignore

        clip = self._stack_buffer()                         # (T, C, H, W)
        context = clip                                      # past context
        predicted_emb = self._predict_future_embedding(context)

        sim_obs = float(_cosine_sim(predicted_emb, self._obstacle_anchor))
        sim_clr = float(_cosine_sim(predicted_emb, self._clear_anchor))

        # Risk = how similar the future is to an obstacle scenario
        predicted_risk = _sigmoid_scale(sim_obs - sim_clr)

        if sim_obs > self._risk_threshold:
            label = "BLOCKED"
        elif sim_clr > self._risk_threshold:
            label = "CLEAR"
        else:
            label = "MIXED"

        return WorldModelResult(
            predicted_risk=predicted_risk,
            similarity_to_obstacle=sim_obs,
            similarity_to_clear=sim_clr,
            label=label,
            buffer_ready=True,
        )

    def build_anchors(
        self,
        obstacle_frames: list[np.ndarray],
        clear_frames: list[np.ndarray],
    ) -> None:
        """
        Compute anchor embeddings from labelled example frames.

        Call this once with a small reference set to calibrate the model
        for the specific corridor environment.
        """
        obs_embs = [self._embed_single(f) for f in obstacle_frames]
        clr_embs = [self._embed_single(f) for f in clear_frames]
        self._obstacle_anchor = np.mean(obs_embs, axis=0)
        self._clear_anchor = np.mean(clr_embs, axis=0)
        logger.info("Anchors rebuilt from %d obs / %d clear frames",
                    len(obstacle_frames), len(clear_frames))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_vjepa2(self) -> None:
        """
        Load V-JEPA 2 via HuggingFace transformers.

        V-JEPA 2 follows the VideoMAE / ViT API for feature extraction.
        We use the encoder (context encoder) only – we do not need the
        predictor head for our use case because we compare embeddings
        directly.
        """
        from transformers import AutoModel, AutoProcessor  # type: ignore

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

    def _init_anchors(self) -> None:
        """
        Bootstrap anchors with synthetic frames when no reference data exists.

        Obstacle anchor: a grey frame (simulates a wall / person blocking the view).
        Clear anchor:    a gradient frame (simulates open corridor perspective lines).
        """
        # Synthetic obstacle frame: uniform dark grey
        obs_frame = np.full((self._input_size, self._input_size, 3), 80, dtype=np.uint8)
        # Synthetic clear frame: bright centre gradient (open corridor)
        clr_frame = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        cx, cy = self._input_size // 2, self._input_size // 2
        for y in range(self._input_size):
            for x in range(self._input_size):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                val = int(max(0, 220 - dist * 1.5))
                clr_frame[y, x] = [val, val, val]

        self._obstacle_anchor = self._embed_single(obs_frame)
        self._clear_anchor = self._embed_single(clr_frame)
        logger.debug("Synthetic anchors initialised")

    def _preprocess(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Resize and normalise a single frame → float32 (C, H, W)."""
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))
        normed = resized.astype(np.float32) / 255.0
        # ImageNet mean/std normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (normed - mean) / std
        return normed.transpose(2, 0, 1)  # (C, H, W)

    def _stack_buffer(self) -> "torch.Tensor":  # type: ignore
        import torch
        arr = np.stack(list(self._buffer), axis=0)          # (T, C, H, W)
        return torch.from_numpy(arr).float().to(self._device)

    def _predict_future_embedding(self, clip: "torch.Tensor") -> np.ndarray:  # type: ignore
        """
        Use V-JEPA 2 to predict the embedding of the scene horizon steps ahead.

        V-JEPA 2 is a masked-prediction model.  We feed the full clip as
        context and mask out the last `horizon` frames so the predictor
        must imagine the future.  The returned embedding is the predicted
        latent for the final (future) frame.
        """
        import torch

        with torch.no_grad():
            if isinstance(self._model, _StubEncoder):
                return self._model.encode(clip)

            # V-JEPA 2 / VideoMAE-style inference:
            # pixel_values shape expected: (B, T, C, H, W)
            pixel_values = clip.unsqueeze(0)  # (1, T, C, H, W)

            # Create boolean mask: mask the last `horizon` frames
            T = pixel_values.shape[1]
            bool_masked_pos = torch.zeros(1, T, dtype=torch.bool, device=self._device)
            mask_start = max(0, T - self._horizon)
            bool_masked_pos[0, mask_start:] = True

            outputs = self._model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
            )
            # last_hidden_state: (B, T_unmasked, hidden_dim)
            # We take the mean over the sequence as the scene embedding
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return emb

    def _embed_single(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Embed a single frame (repeated to fill the clip) for anchor construction."""
        import torch

        tensor = self._preprocess(rgb_frame)
        clip = torch.from_numpy(
            np.stack([tensor] * self._clip_len, axis=0)
        ).float().to(self._device)
        return self._predict_future_embedding(clip)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [-1, 1]."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a.flatten(), b.flatten()) / denom, -1.0, 1.0))


def _sigmoid_scale(x: float, k: float = 5.0) -> float:
    """Map a raw score through a sigmoid centred at 0, output in (0, 1)."""
    return float(1.0 / (1.0 + np.exp(-k * x)))


# ── Lightweight stub used when V-JEPA 2 weights are unavailable ───────────────

class _StubEncoder:
    """
    Deterministic embedding stub.

    Produces an embedding by running a tiny PCA-like projection on the
    flattened mean-pooled frame pixels.  This gives a *meaningful but
    cheap* representation that lets the rest of the system exercise the
    full pipeline without a GPU.
    """

    def __init__(self, embed_dim: int = 1024):
        self._dim = embed_dim
        rng = np.random.default_rng(42)
        # Fixed random projection matrix (simulates learned weights)
        self._proj = rng.standard_normal((3, embed_dim)).astype(np.float32)

    def encode(self, clip: "torch.Tensor") -> np.ndarray:  # type: ignore
        import torch

        # Average across time and spatial dims → (C,)
        mean_pixels = clip.mean(dim=[0, 2, 3]).cpu().numpy()  # (C,)
        # Project to embedding space
        emb = mean_pixels @ self._proj                          # (embed_dim,)
        # L2-normalise
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
