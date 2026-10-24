"""Segmentation masks of vehicles near solid lines: the vehicle's own outline instead of its box.

A detection box (axis-aligned for most models) only bounds a vehicle; whether its wheels are on a
solid line is decided by the outline. Instance segmentation (YOLO11-seg, COCO) gives that
outline, but at ~30 ms per vehicle crop it's only run where it matters: on detections whose box
outline (violations.footprint) comes within violations.mask_gate_m of a solid line, on every
mask_every-th frame (violations.mask_requests). Each outline is stored in reference px in
masks.json, keyed by detection, so moving or adding a line later only segments the frames that
weren't needed before.

Seen from an angle a mask includes the vehicle's side and its mirrors: the part of it nearest
the camera's ground point is the wheels' side, the far part the roof projected outwards. On 0491
(camera on the line's side) the masks were 2.0-2.2 m wide for ~1.8 m cars.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .stabilization import transform_points

COCO_VEHICLES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PAD_PX = 60  # around the box, so the mask isn't cut at the crop's edge

# (crops) -> per crop a list of (mask HxW float, xyxy box in crop px, class)
Segmenter = Callable[[list[np.ndarray]], list[list[tuple[np.ndarray, np.ndarray, int]]]]


def load(out: Path, model: str) -> dict[str, list]:
    """Stored outlines {mask_key: [[x, y], ...] in reference px} ([] = segmented, no vehicle found).
    Outlines from another model don't count."""
    p = Path(out) / "masks.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("entries", {}) if data.get("model") == model else {}


def save(out: Path, model: str, entries: dict[str, list]) -> None:
    p = Path(out) / "masks.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model": model, "entries": entries}), encoding="utf-8")
    tmp.replace(p)


def yolo_segmenter(model: str) -> Segmenter:
    from ultralytics import YOLO

    from .detection import resolve_model

    net = YOLO(resolve_model(model))

    def run(crops: list[np.ndarray]) -> list[list[tuple[np.ndarray, np.ndarray, int]]]:
        out = []
        for res in net.predict(crops, imgsz=640, conf=0.25, verbose=False, retina_masks=True):
            if res.masks is None:
                out.append([])
                continue
            out.append(list(zip(res.masks.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy().astype(int))))
        return out
    return run


def outline_of(found: list[tuple[np.ndarray, np.ndarray, int]], box: tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """The vehicle mask that best matches the detection box (crop px), as a simplified polygon."""
    bx0, by0, bx1, by1 = box
    best, best_iou = None, 0.3  # below this it's another vehicle, or nothing
    for m, b, c in found:
        if int(c) not in COCO_VEHICLES:
            continue
        ix = max(0.0, min(b[2], bx1) - max(b[0], bx0)) * max(0.0, min(b[3], by1) - max(b[1], by0))
        iou = ix / max((b[2] - b[0]) * (b[3] - b[1]) + (bx1 - bx0) * (by1 - by0) - ix, 1e-9)
        if iou > best_iou:
            best, best_iou = m, iou
    if best is None:
        return None
    cs, _ = cv2.findContours((best > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float)


def sample(cfg, out: Path, requests: dict[str, tuple[int, int, np.ndarray]], transforms: dict,
           reporter=None, segment: Optional[Segmenter] = None, batch: int = 8) -> dict:
    """Segment the requested detections that have no stored outline yet; returns the store.
    `requests` from violations.mask_requests; `transforms` {"frames", "T"} (frame -> reference)."""
    from .video import FrameGrabber

    model = cfg.violations.mask_model
    entries = load(out, model)
    todo = sorted(((f, key, corners) for key, (_, f, corners) in requests.items() if key not in entries),
                  key=lambda x: x[0])
    if not todo:
        return {"entries": entries, "new": 0}
    segment = segment or yolo_segmenter(model)
    grab = FrameGrabber(cfg.source)
    T_frames, T_all = transforms["frames"], transforms["T"]
    t0 = time.monotonic()
    pending: list[tuple[str, np.ndarray, np.ndarray, tuple, np.ndarray]] = []

    def flush():
        found = segment([p[1] for p in pending])
        for (key, _, Tk, box, off), f in zip(pending, found):
            poly = outline_of(f, box)
            entries[key] = [] if poly is None else np.round(transform_points(Tk, poly + off), 1).tolist()
        pending.clear()

    try:
        for i, (f, key, corners) in enumerate(todo):
            T = T_all[int(np.clip(np.searchsorted(T_frames, f), 0, len(T_frames) - 1))] if len(T_frames) else np.eye(3)
            img = grab.get(f)
            raw = transform_points(np.linalg.inv(T), np.asarray(corners, float).reshape(4, 2))
            x0, y0 = np.maximum(raw.min(0) - PAD_PX, 0).astype(int)
            x1, y1 = np.minimum(raw.max(0) + PAD_PX, [img.shape[1], img.shape[0]]).astype(int)
            crop = np.ascontiguousarray(img[y0:y1, x0:x1])
            box = (*(raw.min(0) - [x0, y0]), *(raw.max(0) - [x0, y0]))
            pending.append((key, crop, T, box, np.array([x0, y0], float)))
            if len(pending) >= batch:
                flush()
            if reporter and (i % 10 == 0 or i == len(todo) - 1):
                reporter.progress("outlining vehicles near solid lines", i + 1, len(todo),
                                  fps=(i + 1) / max(time.monotonic() - t0, 1e-6))
                if reporter.cancelled():
                    raise InterruptedError("cancelled")
        if pending:
            flush()
    finally:
        grab.close()
        save(out, model, entries)
    return {"entries": entries, "new": len(todo)}
