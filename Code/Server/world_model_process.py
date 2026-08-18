"""
world_model_process.py – run the heavy V-JEPA 2 + SSv2 inference in a SEPARATE
OS process so it can't stall the main server's camera I/O.

Why a process, not a thread: V-JEPA 2 is a ViT-L over a 64-frame clip and SSv2 is
VideoMAE over 16 frames. Their per-call Python work (resizing/normalising dozens
of frames, tensor marshalling, MPS sync points) holds the GIL for seconds at a
time. On a background *thread* that still freezes the whole interpreter — the
camera-receive thread can't drain the socket, the robot's stream backs up, and
the link eventually times out. A separate *process* has its own GIL, so the only
cost the main process pays is pickling one clip out (~tens of ms) and a small
result back — the multi-second inference no longer touches the main loop.

Protocol (one clip in flight, lock-step so the queues never back up):
    main → worker :  (clip: list[np.ndarray], object_label: str)  |  None = stop
    worker → main :  (WorldModelResult | None, SSv2Result | None)

The main-side feeder thread mostly blocks on the result queue (which releases the
GIL), so it doesn't starve I/O either. Results are latest-wins; a missed/stale
pairing is harmless.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

logger = logging.getLogger(__name__)


def _worker_main(cfg: dict, in_q, out_q) -> None:
    """Subprocess entry point: load the heavy models, then serve inference jobs.

    Runs in a fresh (spawned) interpreter, so it re-imports everything and owns
    its own GIL / MPS context. Best-effort throughout — any failure degrades to
    returning None for that model, never crashes the parent.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("world_model_process")

    import numpy as np

    wm = ssv2 = None
    try:
        from world_model import WorldModel
        wm = WorldModel(cfg)
        wm.load()
    except Exception as exc:
        log.error("World model load failed in subprocess (%s) – V-JEPA 2 disabled", exc)
    try:
        from ssv2_model import SSv2Recognizer
        ssv2 = SSv2Recognizer(cfg)
        ssv2.load()
    except Exception as exc:
        log.error("SSv2 load failed in subprocess (%s) – SSv2 disabled", exc)

    # Warm up the cold first inference HERE, off the main process entirely.
    clip_len = max(int((cfg.get("camera", {}) or {}).get("clip_length", 64)), 16)
    dummy = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(clip_len)]
    for name, fn in (("V-JEPA 2", lambda: wm and wm.predict(dummy)),
                     ("SSv2", lambda: ssv2 and ssv2.recognize(dummy, ""))):
        try:
            fn()
        except Exception as exc:
            log.debug("%s warmup skipped: %s", name, exc)
    # The whole point of this dedicated process is to run the heavy models
    # flat-out WITHOUT stalling the main loop's camera I/O — the feeder is
    # lock-step (one clip in → one result out), so the worker only ever holds
    # the newest clip and never queues up. The per-frame skip-gate
    # (run_every_n_frames) exists for the IN-PROCESS fallback, where a skipped
    # tick spares the drive loop a multi-second stall; here it would just waste
    # cycles returning stale results. So run V-JEPA 2 on EVERY clip for the
    # freshest navigation-driving risk. SSv2 is annotation-only and heavier, so
    # keep it on a small job cadence (every 2nd clip) — fresh enough for the
    # caption without halving the V-JEPA 2 update rate.
    if wm is not None:
        wm._run_every = 1
    if ssv2 is not None:
        ssv2._run_every = max(1, int(cfg.get("ssv2", {}).get("subprocess_run_every", 2)))
    log.info("World-model subprocess ready (V-JEPA 2 every clip, SSv2 every %d; warmed)",
             getattr(ssv2, "_run_every", 0) if ssv2 is not None else 0)

    while True:
        try:
            job = in_q.get()
        except (EOFError, OSError):
            break
        if job is None:
            break
        clip, label = job
        wmr = ssr = None
        try:
            if wm is not None:
                r = wm.predict(clip)
                if getattr(r, "buffer_ready", False):
                    wmr = r
        except Exception as exc:
            log.exception("world model predict error: %s", exc)
        try:
            if ssv2 is not None:
                s = ssv2.recognize(clip, label or "")
                if getattr(s, "buffer_ready", False):
                    ssr = s
        except Exception as exc:
            log.exception("ssv2 recognize error: %s", exc)
        try:
            out_q.put((wmr, ssr))
        except Exception:
            pass


class WorldModelProcess:
    """Main-process handle to the world-model subprocess (+ a feeder thread)."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._clip_len = int((cfg.get("camera", {}) or {}).get("clip_length", 64))
        self._ctx = mp.get_context("spawn")   # spawn: safe with MPS/CUDA (fork isn't)
        self._in_q = self._ctx.Queue(maxsize=1)
        self._out_q = self._ctx.Queue(maxsize=2)
        self._proc: mp.process.BaseProcess | None = None
        self._feeder: threading.Thread | None = None
        self._cam_buf = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_wm = None
        self._latest_ssv2 = None
        self._object_label = ""
        self._dead_logged = False

    def start(self, cam_buf) -> None:
        self._cam_buf = cam_buf
        self._running = True
        self._proc = self._ctx.Process(
            target=_worker_main, args=(self._cfg, self._in_q, self._out_q),
            daemon=True, name="WorldModelProc",
        )
        self._proc.start()
        self._feeder = threading.Thread(target=self._feed_loop, daemon=True, name="WMFeeder")
        self._feeder.start()
        logger.info("World-model subprocess started (pid=%s) – V-JEPA 2 + SSv2 run "
                    "off the main process", getattr(self._proc, "pid", "?"))

    def set_object_label(self, label: str) -> None:
        self._object_label = label or ""

    def latest(self):
        with self._lock:
            return self._latest_wm, self._latest_ssv2

    def _feed_loop(self) -> None:
        while self._running:
            if not (self._proc and self._proc.is_alive()):
                if self._running and not self._dead_logged:
                    logger.error("World-model subprocess is not alive – V-JEPA 2/SSv2 "
                                 "results will freeze (drive falls back to detector "
                                 "risk). Set world_model.run_in_subprocess: false to run "
                                 "it in-process instead.")
                    self._dead_logged = True
                return
            clip = self._cam_buf.get_clip() if self._cam_buf else None
            if not clip or len(clip) < self._clip_len:
                time.sleep(0.05)
                continue
            try:
                self._in_q.put((clip, self._object_label), timeout=5.0)
            except queue.Full:
                time.sleep(0.05)
                continue
            try:
                wm, ssv2 = self._out_q.get(timeout=120.0)
            except queue.Empty:
                continue
            with self._lock:
                if wm is not None:
                    self._latest_wm = wm
                if ssv2 is not None:
                    self._latest_ssv2 = ssv2
            time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        try:
            self._in_q.put(None, timeout=0.5)
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
