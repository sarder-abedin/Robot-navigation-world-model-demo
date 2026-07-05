"""
calibrate_anchors.py – build V-JEPA 2 corridor anchors from real frames.

The world model scores a scene by cosine similarity to two anchor embeddings:
an "obstacle/blocked" anchor and a "clear/open" anchor. By default those anchors
are SYNTHETIC (a grey square vs a gradient), which barely matches a real corridor.
This tool replaces them with anchors built from YOUR frames so BLOCKED/CLEAR is
meaningful in your environment.

Collect a handful of images each (JPEG/PNG) — e.g. grab frames from the demo
video or snapshots from the robot:
  blocked/  – the corridor blocked (a wall/person/obstacle filling the path)
  clear/    – the corridor open (nothing in the way)

Then:
  cd Code/Server
  python calibrate_anchors.py --blocked ./blocked --clear ./clear --out anchors.npz
  # then set  world_model.anchors_path: "anchors.npz"  in config.yaml

Runs the same V-JEPA 2 encoder the server uses (falls back to the stub encoder
if the model/weights are unavailable — you'll want the real model for useful
anchors, so run this where V-JEPA 2 loads, e.g. natively on the GPU/MPS box).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_images(folder: str) -> list[np.ndarray]:
    """Load all images in a folder as RGB uint8 arrays (world_model resizes them)."""
    frames = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.splitext(p)[1].lower() not in _EXTS:
            continue
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build V-JEPA 2 corridor anchors")
    ap.add_argument("--blocked", required=True, help="folder of BLOCKED corridor images")
    ap.add_argument("--clear", required=True, help="folder of CLEAR corridor images")
    ap.add_argument("--out", default="anchors.npz", help="output .npz path")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    blocked = load_images(args.blocked)
    clear = load_images(args.clear)
    if not blocked or not clear:
        print(f"Need images in BOTH folders (blocked={len(blocked)}, clear={len(clear)})",
              file=sys.stderr)
        return 1

    from world_model import WorldModel
    wm = WorldModel(cfg)
    wm.load()
    wm.build_anchors(blocked, clear)
    wm.save_anchors(args.out)
    print(f"Built anchors from {len(blocked)} blocked + {len(clear)} clear frames → {args.out}")
    print(f"Now set  world_model.anchors_path: \"{args.out}\"  in {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
