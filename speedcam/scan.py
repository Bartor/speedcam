"""Activity scan: where in a (long) video are there vehicles at all?

A quick pass reads about one frame a second at low resolution and counts vehicles with a small
detector. Where the file allows it, only keyframes are decoded (a GOP of up to `keyframe_max_gap_s`):
decoding is the slow part of reading 4K, and keyframe-only decoding skips almost all of it. The
counts give a timeline and suggested parts to analyse (vehicles present, padded, short gaps
merged); the analysis and the render can then skip everything else.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import av
import numpy as np

from .config import ProjectConfig, ScanSettings
from .video import _stream_fps

SCAN_FILE = "scan.json"


def keyframe_indices(path: str | Path) -> tuple[list[int], float]:
    """Frame indices of the keyframes (from the packets: no decoding) and the container fps."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        fps = _stream_fps(st)
        tb, start = float(st.time_base), st.start_time or 0
        out = []
        for pkt in c.demux(st):
            if pkt.is_keyframe and pkt.pts is not None:
                out.append(int(round((pkt.pts - start) * tb * fps)))
    return sorted(set(out)), fps


def sample_frames(path: str | Path, s: ScanSettings, size: int, reporter=None):
    """Yield (frame index, BGR frame scaled to `size` on its long side, mode): one keyframe per
    1/sample_fps seconds when keyframes come often enough (only those packets are decoded: each
    keyframe decodes on its own), else every k-th frame (then all frames are decoded)."""
    keys, fps = keyframe_indices(path)
    gaps = np.diff(keys) / fps if len(keys) > 1 else np.array([np.inf])
    keyframes_only = len(keys) > 1 and float(np.median(gaps)) <= s.keyframe_max_gap_s
    step = max(1, int(round(fps / max(s.sample_fps, 1e-6))))
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb, start = float(st.time_base), st.start_time or 0

        def small(frame):
            k = size / max(frame.width, frame.height)
            w, h = max(2, int(round(frame.width * k / 2)) * 2), max(2, int(round(frame.height * k / 2)) * 2)
            return frame.to_ndarray(format="bgr24", width=w, height=h)

        if keyframes_only:
            st.thread_type = "SLICE"  # no frame-threading delay: a keyframe packet in, its frame out
            next_i = 0
            for pkt in c.demux(st):
                if not pkt.is_keyframe or pkt.pts is None:
                    continue
                i = int(round((pkt.pts - start) * tb * fps))
                if i < next_i:
                    continue
                for frame in st.codec_context.decode(pkt):
                    next_i = i + step
                    yield i, small(frame), "keyframes"
        else:
            st.thread_type = "AUTO"
            for frame in c.decode(st):
                i = int(round((frame.pts - start) * tb * fps)) if frame.pts is not None else 0
                if i % step == 0:
                    yield i, small(frame), "sampled"


def scan(cfg: ProjectConfig, out_dir: str | Path, reporter=None) -> dict:
    from .detection import make_detector
    from .timing import probe

    s = cfg.scan
    out = Path(out_dir)
    info = probe(cfg.source)
    duration = info.nb_frames / info.container_fps
    # the analysis's detector (or the one named for the scan), on frames at its detection size
    ds = cfg.detection.model_copy(update={"half": s.half, "conf": s.conf, "imgsz": s.imgsz, "tile": 0,
                                          **({"model": s.model} if s.model else {})})
    if ds.type == "blob":  # synthetic tests: frames are smaller than the clip
        ds = ds.model_copy(update={"blob_min_area": max(1, int(ds.blob_min_area * (s.imgsz / info.width) ** 2))})
    det = make_detector(ds)
    if hasattr(det, "choose_primary"):
        # Auto: pick the model for this clip's camera angle from frames spread over all of it
        # (as the analysis does; a quiet start of a CCTV clip has nothing to judge by)
        import cv2

        from .video import FrameGrabber

        g = FrameGrabber(cfg.source)
        try:
            picks = []
            for f in np.linspace(0, info.nb_frames - 1, cfg.detection.auto_sample_frames).round():
                img = g.get(int(f))
                k = s.imgsz / max(img.shape[:2])
                picks.append(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
        finally:
            g.close()
        det.choose_primary(picks)
    t0 = time.monotonic()
    frames, counts, mode = [], [], "keyframes"
    batch: list = []

    boxes: list[np.ndarray] = []

    def flush() -> None:
        for (i, _), d in zip(batch, det.detect_batch([b for _, b in batch])):
            frames.append(i)
            counts.append(int(len(d)))
            boxes.append(np.asarray(d.xyxy, float).reshape(-1, 4))
        batch.clear()

    for i, img, mode in sample_frames(cfg.source, s, s.imgsz, reporter):
        batch.append((i, img))
        if len(batch) == 16:
            flush()
            if reporter:
                reporter.progress("scanning for vehicles", i, info.nb_frames,
                                  fps=len(frames) / max(time.monotonic() - t0, 1e-6))
                if reporter.cancelled():
                    raise InterruptedError("cancelled")
    if batch:
        flush()
    result = {"frames": frames, "counts": counts, "moving": moving_counts(boxes, s.still_iou), "mode": mode, "model": s.model, "imgsz": s.imgsz,
              "nb_frames": info.nb_frames, "container_fps": info.container_fps, "duration_s": duration,
              "elapsed_s": round(time.monotonic() - t0, 1), "created": time.time()}
    (out / SCAN_FILE).write_text(json.dumps(result), encoding="utf-8")
    if reporter:
        reporter.progress("done", 1, 1)
    return result


def moving_counts(boxes: list[np.ndarray], still_iou: float) -> list[int]:
    """Vehicles per sample that aren't standing still: a box that overlaps one in the sample
    before or after (IoU >= still_iou, ~1 s apart) is a parked or waiting vehicle. Activity means
    moving vehicles; a street of parked cars is not."""
    from .detection import _iou_matrix

    out = []
    for k, b in enumerate(boxes):
        still = np.zeros(len(b), bool)
        for j in (k - 1, k + 1):
            if 0 <= j < len(boxes) and len(b) and len(boxes[j]):
                still |= _iou_matrix(b, boxes[j]).max(1) >= still_iou
        out.append(int((~still).sum()))
    return out


def load(out_dir: str | Path) -> Optional[dict]:
    p = Path(out_dir) / SCAN_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def suggest_parts(sc: dict, s: ScanSettings) -> list[tuple[int, int]]:
    """Frame ranges [start, end) with vehicles: each sample with at least `min_count` vehicles
    stands for the time up to its neighbours, padded by `pad_s`; parts closer than `merge_s` are
    joined. In frames of the video (container rate, same as the analysis)."""
    f = np.asarray(sc["frames"], int)
    c = np.asarray(sc.get("moving") if s.moving_only and "moving" in sc else sc["counts"], int)
    n, fps = int(sc["nb_frames"]), float(sc["container_fps"])
    if not len(f):
        return []
    gap = np.diff(f)
    step = int(np.median(gap)) if len(gap) else n
    prev = np.r_[f[0], gap]  # gap to the previous sample (the first reaches back to the start)
    nxt = np.r_[gap, n - f[-1]]  # to the next one (the last reaches to the end)
    lo = f - (np.minimum(prev, 2 * step) + 1) // 2
    hi = f + (np.minimum(nxt, 2 * step) + 1) // 2
    lo[0] = 0 if f[0] <= step else lo[0]
    hi[-1] = n if n - f[-1] <= step else hi[-1]
    pad, merge = int(round(s.pad_s * fps)), int(round(s.merge_s * fps))
    parts: list[list[int]] = []
    for a, b, k in zip(lo, hi, c):
        if k < s.min_count:
            continue
        a, b = max(0, a - pad), min(n, b + pad + 1)
        if parts and a - parts[-1][1] <= merge:
            parts[-1][1] = max(parts[-1][1], b)
        else:
            parts.append([a, b])
    return [(int(a), int(b)) for a, b in parts if b > a]


def analysis_ranges(cfg: ProjectConfig, nb_frames: int) -> list[tuple[int, int]]:
    """Frame ranges the analysis covers: start/end frames, cut down to the chosen parts if any."""
    lo = max(0, cfg.start_frame)
    hi = min(cfg.end_frame or nb_frames, nb_frames)
    if not cfg.parts:
        return [(lo, hi)] if hi > lo else []
    out = []
    for a, b in sorted(cfg.parts):
        a, b = max(a, lo), min(b, hi)
        if b > a:
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
    return out
