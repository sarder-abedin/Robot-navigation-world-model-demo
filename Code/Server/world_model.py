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


def _should_retry_gpu(cpu_calls: int, retry_every: int) -> bool:
    """After falling back to CPU, whether to re-probe the GPU on this call.

    retry_every <= 0 means never retry (stay on CPU until the process restarts);
    otherwise retry once every `retry_every` CPU forwards."""
    return retry_every > 0 and cpu_calls >= retry_every


@dataclass
class WorldModelResult:
    predicted_risk: float = 0.0
    similarity_to_obstacle: float = 0.0
    similarity_to_clear: float = 0.0
    label: str = "UNKNOWN"
    buffer_ready: bool = False
    # Small (grid×grid×3) uint8 PCA visualisation of the dense patch features —
    # the "what V-JEPA 2 sees" image. None when feature_viz is off or unavailable.
    feature_rgb: object = None


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
        # Compute precision for the GPU forward: bf16 (default) halves the attention
        # activation memory — the fix for the 12 GiB masked-forward OOM — and ~2× the
        # speed. Ignored on CPU/MPS (kept fp32). See device_utils.resolve_dtype.
        self._precision = wm_cfg.get("precision", "bf16")
        # After a GPU OOM the forward runs on CPU; re-probe the GPU every N such
        # calls in case VRAM has since freed up (0 = never retry, stay on CPU).
        self._gpu_retry_every = int(wm_cfg.get("gpu_retry_every_calls", 30))
        self._run_every = wm_cfg.get("run_every_n_frames", 8)
        # Path to calibrated corridor anchors (built by calibrate_anchors.py from
        # real "blocked"/"clear" frames). When set + present they replace the
        # synthetic anchors, making BLOCKED/CLEAR reflect *your* environment.
        self._anchors_path = wm_cfg.get("anchors_path", "") or ""

        self._model = None
        self._device = None
        self._device_name = "cpu"    # resolved name ("cuda"/"mps"/"cpu") for autocast
        self._is_gpu = False
        self._amp_dtype = None       # autocast compute dtype (bf16/fp16/fp32)
        self._obstacle_anchor: np.ndarray | None = None
        self._clear_anchor: np.ndarray | None = None
        self._call_count = 0
        self._last_result = WorldModelResult()
        self._mask_warned = False   # log the masked-forward fallback only once
        self._oom_warned = False    # log the GPU→CPU OOM offload only once
        self._on_cpu = False        # sticky: degraded to CPU after an OOM
        self._cpu_calls = 0         # forwards done on CPU since the last GPU probe
        # Dense-feature PCA visualisation ("what V-JEPA 2 sees"): compute a small
        # RGB map of the patch features each forward, shipped to the HUD.
        self._feature_viz = bool(wm_cfg.get("feature_viz", True))
        self._pca_basis = None      # previous frame's PCA basis (temporal sign-align)
        self._last_feature_rgb = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch  # type: ignore
        from device_utils import resolve_device, is_gpu, resolve_dtype
        self._device, dev_name = resolve_device(self._device_str)
        self._device_name = dev_name
        self._is_gpu = is_gpu(dev_name)
        self._amp_dtype = resolve_dtype(self._precision, dev_name)
        try:
            self._load_vjepa2()
            logger.info("V-JEPA 2 loaded: %s on %s (%s)", self._model_id, dev_name,
                        str(self._amp_dtype).replace("torch.", "") if self._is_gpu else "fp32")
        except Exception as exc:
            hint = ""
            if "recognize" in str(exc).lower() or "vjepa" in str(exc).lower():
                hint = (" — this transformers build may predate V-JEPA 2; "
                        "upgrade it (pip install -U 'transformers>=4.53')")
            logger.warning("V-JEPA 2 load failed (%s) – using stub encoder%s", exc, hint)
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

        # Run once as soon as the clip is first ready (don't return the default
        # buffer_ready=False for the first run_every frames), then honour the
        # cadence — matches depth_perception / ssv2_model.
        if self._last_result.buffer_ready and self._call_count % self._run_every != 0:
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
            feature_rgb=self._last_feature_rgb,
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
        from transformers import AutoModel  # type: ignore
        # V-JEPA 2 does its OWN preprocessing (_preprocess_frame: resize + ImageNet
        # normalise), so the HF AutoProcessor is optional and, in fact, unused for
        # inference. Some transformers versions ship the VJEPA2 *model* but can't
        # instantiate an AutoProcessor for this repo ("Unrecognized processing
        # class …") — that must NOT sink the real encoder into the stub. Load it
        # best-effort and carry on without it.
        self._processor = None
        try:
            from transformers import AutoProcessor  # type: ignore
            self._processor = AutoProcessor.from_pretrained(self._model_id)
        except Exception as exc:
            logger.info("V-JEPA 2 processor unavailable (%s) – using built-in "
                        "preprocessing (processor is not needed for inference)", exc)
        # On a GPU, request memory-efficient (SDPA / flash) attention: it computes
        # attention without ever materialising the full N×N score matrix, which is
        # the tensor that blew up to 12 GiB in the masked forward. Best-effort — an
        # older transformers or this arch may not accept the kwarg, so fall back.
        kwargs = {}
        if self._is_gpu:
            kwargs["attn_implementation"] = "sdpa"
        try:
            self._model = AutoModel.from_pretrained(self._model_id, **kwargs)
        except (TypeError, ValueError, KeyError, ImportError) as exc:
            if kwargs:
                logger.info("V-JEPA 2: SDPA attention unsupported (%s) – default attention", exc)
                self._model = AutoModel.from_pretrained(self._model_id)
            else:
                raise
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
        from device_utils import autocast_ctx, is_oom_error

        with torch.no_grad():
            if isinstance(self._model, _StubEncoder):
                return self._model.encode(clip)

            # (1, T, C, H, W)
            pixel_values = clip.unsqueeze(0)
            T = pixel_values.shape[1]
            mask = torch.zeros(1, T, dtype=torch.bool, device=pixel_values.device)
            mask[0, max(0, T - self._horizon):] = True

            # Degraded (CPU) mode after a prior OOM: the model already lives on the
            # CPU — keep it there and run on CPU. Moving the ~1.6 GB model GPU↔CPU
            # every compute tick would cost far more than the forward itself, so we
            # DON'T bounce it back each call; instead re-probe the GPU periodically
            # in case VRAM has since freed up.
            if self._on_cpu and self._is_gpu:
                self._cpu_calls += 1
                if _should_retry_gpu(self._cpu_calls, self._gpu_retry_every):
                    self._cpu_calls = 0
                    self._model.to(self._device)
                    self._on_cpu = False
                    logger.info("V-JEPA 2: re-probing the GPU after CPU fallback")
                else:
                    return self._forward_embed(
                        self._model, pixel_values.to("cpu"), mask.to("cpu"))

            # Normal path: run on the GPU in bf16/fp16 (autocast) — half the
            # activation memory and ~2× faster. If it STILL runs out of GPU memory
            # (a spike bigger than the free budget), don't degrade the algorithm —
            # move the model to the CPU and run there (correct, slower). This runs
            # in the V-JEPA 2 subprocess, so the slow CPU forward can't stall the
            # camera loop.
            try:
                with autocast_ctx(self._device_name, self._amp_dtype):
                    return self._forward_embed(self._model, pixel_values, mask)
            except RuntimeError as exc:
                if not (self._is_gpu and is_oom_error(exc)):
                    raise
                if not self._oom_warned:
                    logger.warning(
                        "V-JEPA 2 forward ran out of GPU memory (%s) – running it on "
                        "CPU (slower). To keep it on the GPU, free VRAM (rocm-smi / "
                        "nvidia-smi shows what's using it) or set a smaller "
                        "world_model.model_id / camera.clip_length.",
                        str(exc).split(".")[0])
                    self._oom_warned = True
                torch.cuda.empty_cache()
                self._model.to("cpu")        # sticky: stay on CPU until the next re-probe
                self._on_cpu = True
                self._cpu_calls = 0
                return self._forward_embed(
                    self._model, pixel_values.to("cpu"), mask.to("cpu"))

    def _forward_embed(self, model, pixel_values, mask) -> np.ndarray:
        """One masked forward → mean-pooled (D,) embedding. Falls back to a plain
        (unmasked) encode on a signature/shape mismatch, but lets an OOM propagate
        so the caller can retry on CPU."""
        import torch
        from device_utils import is_oom_error
        try:
            outputs = model(pixel_values_videos=pixel_values, bool_masked_pos=mask)
        except (TypeError, ValueError, RuntimeError) as exc:
            if is_oom_error(exc):
                raise                       # handled by the caller's CPU offload
            if not self._mask_warned:
                logger.warning("V-JEPA 2 masked forward failed (%s) – encoding "
                               "without bool_masked_pos", exc)
                self._mask_warned = True
            outputs = model(pixel_values_videos=pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
        self._compute_feature_rgb(outputs.last_hidden_state)
        return emb

    def _compute_feature_rgb(self, last_hidden_state) -> None:
        """PCA of the encoder's patch tokens → a small RGB 'what V-JEPA 2 sees'
        grid (stored on self._last_feature_rgb). Best-effort: any failure just
        leaves it None so the HUD simply skips the overlay."""
        if not self._feature_viz:
            self._last_feature_rgb = None
            return
        try:
            from feature_viz import patch_features_to_rgb, infer_patch_grid
            hs = last_hidden_state[0].float().cpu().numpy()      # (N_tokens, D)
            cfg = getattr(self._model, "config", None)
            ps = int(getattr(cfg, "patch_size", 16) or 16)
            grid = infer_patch_grid(hs.shape[0], self._input_size, ps)
            if grid is None:
                self._last_feature_rgb = None
                return
            plane = grid[0] * grid[1]
            temporal = max(1, hs.shape[0] // plane)
            # tokens are temporal×spatial → average over time for one spatial map
            feats = hs[: temporal * plane].reshape(temporal, plane, hs.shape[1]).mean(axis=0)
            rgb, self._pca_basis = patch_features_to_rgb(feats, grid, self._pca_basis)
            self._last_feature_rgb = rgb
        except Exception as exc:
            if not getattr(self, "_featviz_warned", False):
                logger.debug("Feature-viz unavailable (%s) – skipping overlay", exc)
                self._featviz_warned = True
            self._last_feature_rgb = None

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
