"""
build_anchor_set.py – auto-sort a balanced blocked/clear frame set for V-JEPA 2
anchor calibration, so you skip the manual sorting.

The world model scores a scene by cosine similarity to two anchor embeddings
(blocked vs clear). Building good anchors just needs a *balanced* handful of
frames per class — this tool pulls them for you from either source:

  --run  <dir>   A logged run (navigation_log.csv + raw_frames/, or frames/).
                 Labels frames with the SAME independent signals as
                 calibrate_from_logs (YOLO risk + ultrasonic + action) — never
                 the world model itself. Prefers raw_frames/ (no HUD overlay).
  --video <path> Any video clip (e.g. assets/demo_clips/corridor.mp4). Samples
                 frames and runs YOLO to label blocked (obstacle centred/large)
                 vs clear (nothing significant ahead).

It writes a balanced, capped set to  <out>/blocked/*.jpg  +  <out>/clear/*.jpg,
then prints the calibrate_anchors.py command. With --build it also builds the
anchors.npz directly (and --apply patches world_model.anchors_path in config).

  cd Code/Server
  python build_anchor_set.py --run ../../logs_rpi/run_XXXX --out anchorset
  python build_anchor_set.py --video ../../assets/demo_clips/corridor.mp4 --out anchorset
  python build_anchor_set.py --run <dir> --out anchorset --build anchors.npz --apply config.yaml

Note: a run must have BOTH blocked and clear stretches (and raw_frames/ for clean
anchors — enable logging.save_raw_frames before the run). A video needs the real
YOLO model available (transformers/ultralytics), so run it where the server runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ── Pure labelling (video path) — testable without torch ──────────────────────

def label_by_detection(in_center: bool, closest_area: float,
                       block_area: float = 0.10, clear_area: float = 0.02):
    """Label a frame from YOLO's largest-obstacle geometry.

    blocked → an obstacle sits in the centre AND fills at least `block_area` of
    the frame; clear → nothing bigger than `clear_area` is present. The ambiguous
    middle returns None so only confident frames become anchors.
    """
    if in_center and closest_area >= block_area:
        return "blocked"
    if closest_area <= clear_area:
        return "clear"
    return None


def _even_pick(xs: list, n: int) -> list:
    """Evenly subsample xs down to n items (keeps temporal spread)."""
    if n <= 0:
        return []
    if len(xs) <= n:
        return list(xs)
    step = len(xs) / n
    return [xs[int(i * step)] for i in range(n)]


def balance(blocked: list, clear: list, per_class: int) -> tuple[list, list]:
    """Trim both classes to the same size (≤ per_class) for balanced anchors."""
    n = min(len(blocked), len(clear), per_class)
    return _even_pick(blocked, n), _even_pick(clear, n)


# ── Source: a logged run ──────────────────────────────────────────────────────

def _available_frame_idxs(src_dir: str) -> set:
    idxs = set()
    for p in glob.glob(os.path.join(src_dir, "frame_*.jpg")):
        m = re.search(r"frame_(\d+)\.jpg$", os.path.basename(p))
        if m:
            idxs.add(int(m.group(1)))
    return idxs


def frames_from_run(run_dir: str, per_class: int):
    """(blocked, clear, used_raw): balanced [(name, bgr)] from a logged run.

    Auto-labels via calibrate_from_logs (independent of the world model) and only
    keeps labelled frames that actually have a saved image on disk.
    """
    import cv2

    from calibrate_from_logs import autolabel_rows, read_rows

    rows = read_rows(run_dir)
    b_idx, c_idx = autolabel_rows(rows)

    raw_dir = os.path.join(run_dir, "raw_frames")
    used_raw = os.path.isdir(raw_dir) and _available_frame_idxs(raw_dir)
    src = raw_dir if used_raw else os.path.join(run_dir, "frames")
    have = _available_frame_idxs(src)

    b_idx = sorted(set(b_idx) & have)
    c_idx = sorted(set(c_idx) & have)
    b_idx, c_idx = balance(b_idx, c_idx, per_class)

    def load(idxs):
        out = []
        for idx in idxs:
            img = cv2.imread(os.path.join(src, f"frame_{idx:06d}.jpg"))
            if img is not None:
                out.append((f"frame_{idx:06d}.jpg", img))
        return out

    return load(b_idx), load(c_idx), bool(used_raw)


# ── Source: a video clip (runs YOLO) ──────────────────────────────────────────

def frames_from_video(video_path: str, cfg: dict, per_class: int, fps: float,
                      block_area: float, clear_area: float):
    """(blocked, clear): balanced [(name, bgr)] labelled by YOLO over the clip."""
    import cv2

    from detector import Detector

    det = Detector(cfg)
    det.load()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / max(fps, 0.1))))

    blocked, clear = [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            res = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            lab = label_by_detection(res.obstacle_in_center, res.closest_area,
                                     block_area, clear_area)
            if lab == "blocked":
                blocked.append((f"frame_{i:06d}.jpg", bgr))
            elif lab == "clear":
                clear.append((f"frame_{i:06d}.jpg", bgr))
        i += 1
    cap.release()
    return balance(blocked, clear, per_class)


# ── Output ────────────────────────────────────────────────────────────────────

def write_set(out_dir: str, blocked: list, clear: list) -> tuple[str, str]:
    import cv2
    bdir = os.path.join(out_dir, "blocked")
    cdir = os.path.join(out_dir, "clear")
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    for name, img in blocked:
        cv2.imwrite(os.path.join(bdir, name), img)
    for name, img in clear:
        cv2.imwrite(os.path.join(cdir, name), img)
    return bdir, cdir


def _build_anchors(bdir: str, cdir: str, config: str, out_npz: str,
                   apply_config: bool) -> None:
    import yaml

    from calibrate_anchors import load_images
    from world_model import WorldModel

    with open(config) as f:
        cfg = yaml.safe_load(f)
    blocked, clear = load_images(bdir), load_images(cdir)
    wm = WorldModel(cfg)
    wm.load()
    wm.build_anchors(blocked, clear)
    wm.save_anchors(out_npz)
    print(f"Built anchors from {len(blocked)} blocked + {len(clear)} clear → {out_npz}")
    if apply_config:
        from calibrate_from_logs import patch_config_block
        patch_config_block(config, "world_model", {"anchors_path": os.path.abspath(out_npz)})
        print(f"Patched world_model.anchors_path in {config} (backup: {config}.bak)")
    else:
        print(f'Now set  world_model.anchors_path: "{out_npz}"  in {config}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-sort a balanced blocked/clear "
                                             "frame set for V-JEPA 2 anchors")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="logged run dir (navigation_log.csv + frames)")
    src.add_argument("--video", help="video clip to label with YOLO")
    ap.add_argument("--out", default="anchorset", help="output dir (blocked/ + clear/)")
    ap.add_argument("--per-class", type=int, default=24, help="max frames per class")
    ap.add_argument("--fps", type=float, default=2.0, help="video sampling rate")
    ap.add_argument("--block-area", type=float, default=0.10,
                    help="video: min centred-obstacle area fraction → blocked")
    ap.add_argument("--clear-area", type=float, default=0.02,
                    help="video: max obstacle area fraction → clear")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--build", metavar="ANCHORS_NPZ", nargs="?", const="anchors.npz",
                    help="also build anchors.npz from the extracted set")
    ap.add_argument("--apply", action="store_true",
                    help="with --build: patch world_model.anchors_path in config")
    args = ap.parse_args(argv)

    if args.run:
        blocked, clear, used_raw = frames_from_run(args.run, args.per_class)
        if not used_raw and (blocked or clear):
            print("WARNING: using annotated frames/ (HUD overlays baked in). For clean "
                  "anchors, enable logging.save_raw_frames before the run.", file=sys.stderr)
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        blocked, clear = frames_from_video(args.video, cfg, args.per_class,
                                           args.fps, args.block_area, args.clear_area)

    if not blocked or not clear:
        print(f"Not enough balanced frames (blocked={len(blocked)}, clear={len(clear)}). "
              f"The source needs BOTH blocked and clear stretches.", file=sys.stderr)
        return 1

    bdir, cdir = write_set(args.out, blocked, clear)
    print(f"Wrote {len(blocked)} blocked + {len(clear)} clear frames → {args.out}/")

    if args.build:
        _build_anchors(bdir, cdir, args.config, args.build, args.apply)
    else:
        print("Next:")
        print(f"  python calibrate_anchors.py --blocked {bdir} --clear {cdir} --out anchors.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
