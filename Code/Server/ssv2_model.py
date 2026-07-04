"""
ssv2_model.py – genuine Something-Something-V2 action recognition (server side).

Unlike temporal_action.py (a lightweight motion heuristic), this module runs a
REAL video-classification model fine-tuned on the Something-Something-V2 dataset
(VideoMAE by default) over a rolling clip of frames.

SSv2 class labels are templated phrases with a "something" placeholder, e.g.:
  "Moving something closer to something"
  "Pushing something from left to right"
  "Something falling like a rock"

We fill the "something" slot(s) with the object YOLO detected (the largest /
closest obstacle), producing a human-readable sentence like:
  "Moving person closer"        (from label "person" + template)
  "Pushing chair from left to right"

The sentence is for annotation + logging; it does NOT drive navigation (the
decision fuser still uses the fast temporal heuristic for temporal_risk).

Fallback: if transformers / the checkpoint / torch are unavailable, a stub keeps
the pipeline working and still demonstrates the YOLO-filled composition with a
low-confidence "<object> in view" placeholder (clearly logged as a stub).

API
───
  ssv2 = SSv2Recognizer(cfg)
  ssv2.load()
  ssv2.recognize(clip: list[np.ndarray], object_label: str) -> SSv2Result
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSv2Result:
    template: str = "UNKNOWN"      # raw SSv2 class label (with "something")
    sentence: str = ""            # template with "something" -> object_label
    object_label: str = ""       # YOLO object used to fill the slot
    confidence: float = 0.0
    buffer_ready: bool = False
    is_stub: bool = False


def fill_template(template: str, object_label: str) -> str:
    """Replace the SSv2 'something' placeholder(s) with the detected object.

    Preserves sentence-initial capitalisation ("Something" -> "Person"). If no
    object was detected, the template is returned unchanged ("something").
    """
    if not template:
        return ""
    obj = (object_label or "").strip()
    if not obj:
        return template
    # Capitalised placeholder at the start of the sentence.
    out = re.sub(r"^Something\b", obj[:1].upper() + obj[1:], template)
    # Any remaining lowercase "something" placeholders.
    out = re.sub(r"\bsomething\b", obj, out)
    return out


class SSv2Recognizer:
    def __init__(self, cfg: dict):
        ssv2_cfg = cfg.get("ssv2", {}) or {}
        self._enabled = ssv2_cfg.get("enabled", True)
        self._model_id = ssv2_cfg.get("model_id", "MCG-NJU/videomae-base-finetuned-ssv2")
        self._num_frames = int(ssv2_cfg.get("num_frames", 16))
        self._device_str = ssv2_cfg.get("device", "cpu")
        self._run_every = max(1, int(ssv2_cfg.get("run_every_n_frames", 16)))
        self._min_conf = float(ssv2_cfg.get("min_confidence", 0.15))

        self._model = None
        self._processor = None
        self._id2label: dict[int, str] = {}
        self._device = None
        self._call_count = 0
        self._last = SSv2Result()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self._enabled:
            logger.info("SSv2 disabled in config – skipping model load")
            self._model = None
            return
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
            )
            from device_utils import is_gpu, resolve_device
            self._device, dev_name = resolve_device(self._device_str)
            # Classify more often when a GPU is available (cheap), less on CPU.
            if is_gpu(dev_name):
                self._run_every = max(1, self._run_every // 2)
            self._processor = VideoMAEImageProcessor.from_pretrained(self._model_id)
            self._model = VideoMAEForVideoClassification.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()
            self._id2label = dict(self._model.config.id2label)
            # VideoMAE checkpoints define how many frames they expect.
            self._num_frames = int(getattr(self._model.config, "num_frames", self._num_frames))
            logger.info(
                "SSv2 model loaded: %s (%d classes, %d frames) on %s (every %d frames)",
                self._model_id, len(self._id2label), self._num_frames, dev_name, self._run_every,
            )
        except Exception as exc:
            logger.warning(
                "SSv2 model unavailable (%s) – using stub (composition still works, "
                "no real classification)", exc,
            )
            self._model = None

    def recognize(self, clip: list[np.ndarray], object_label: str) -> SSv2Result:
        """
        clip: list of RGB uint8 frames (already resized square). Needs at least
        num_frames to run; otherwise buffer_ready=False.
        object_label: the YOLO class name to fill the "something" slot.

        Heavy inference is skipped between run_every_n_frames calls; the last
        result (re-filled with the current object) is returned in between.
        """
        self._call_count += 1

        if not clip or len(clip) < self._num_frames:
            return SSv2Result(buffer_ready=False)

        # Re-use the cached classification on skipped frames, but refresh the
        # filled sentence with the current object so the annotation stays live.
        if self._model is not None and self._call_count % self._run_every != 0:
            cached = self._last
            return SSv2Result(
                template=cached.template,
                sentence=fill_template(cached.template, object_label) or cached.sentence,
                object_label=object_label,
                confidence=cached.confidence,
                buffer_ready=True,
                is_stub=cached.is_stub,
            )

        if self._model is None:
            # Stub: no real classification, but demonstrate YOLO-filled output.
            template = "something in view" if object_label else "scene is clear"
            res = SSv2Result(
                template=template,
                sentence=fill_template(template, object_label),
                object_label=object_label,
                confidence=0.0,
                buffer_ready=True,
                is_stub=True,
            )
            self._last = res
            return res

        try:
            res = self._classify(clip, object_label)
        except Exception as exc:
            logger.debug("SSv2 inference error: %s", exc)
            return self._last
        self._last = res
        return res

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify(self, clip: list[np.ndarray], object_label: str) -> SSv2Result:
        import torch  # type: ignore

        frames = self._sample_frames(clip, self._num_frames)
        inputs = self._processor(frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        template = self._id2label.get(int(idx.item()), "UNKNOWN")
        return SSv2Result(
            template=template,
            sentence=fill_template(template, object_label),
            object_label=object_label,
            confidence=float(conf.item()),
            buffer_ready=True,
            is_stub=False,
        )

    @staticmethod
    def _sample_frames(clip: list[np.ndarray], n: int) -> list[np.ndarray]:
        """Evenly sample exactly n frames from the clip (VideoMAE expects n)."""
        if len(clip) == n:
            return list(clip)
        idxs = np.linspace(0, len(clip) - 1, n).round().astype(int)
        return [clip[i] for i in idxs]
