"""Vehicle detection: Ultralytics YOLO (oriented or axis-aligned boxes) with optional tiling.

A simple colour-blob detector is included for synthetic validation clips.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectionSettings

CLASSES = ["car", "truck", "bus", "motorcycle"]
# DOTA (OBB models) has no car/truck split: map its vehicle classes onto ours.
ALIASES = {
    "small vehicle": "car", "large vehicle": "truck", "car": "car", "van": "car", "truck": "truck",
    "bus": "bus", "motorcycle": "motorcycle", "motorbike": "motorcycle",
}


@dataclass
class Detections:
    corners: np.ndarray  # (n, 4, 2) oriented box corners, frame px
    conf: np.ndarray  # (n,)
    cls: np.ndarray  # (n,) index into CLASSES

    @property
    def xyxy(self) -> np.ndarray:
        if len(self.corners) == 0:
            return np.zeros((0, 4))
        return np.concatenate([self.corners.min(1), self.corners.max(1)], axis=1)

    @property
    def centers(self) -> np.ndarray:
        return self.corners.mean(1) if len(self.corners) else np.zeros((0, 2))

    def __len__(self) -> int:
        return len(self.conf)

    def subset(self, mask) -> "Detections":
        return Detections(self.corners[mask], self.conf[mask], self.cls[mask])

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4, 2)), np.zeros(0), np.zeros(0, int))


def _has_quantize() -> bool:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return "quantize" in DEFAULT_CFG_DICT


def make_tiles(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = max(1, int(tile * (1 - overlap)))
        s = list(range(0, n - tile, step))
        s.append(n - tile)
        return s

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


def _nms(dets: Detections, iou: float) -> Detections:
    if len(dets) < 2:
        return dets
    xyxy = dets.xyxy
    boxes = [[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in xyxy]
    keep = cv2.dnn.NMSBoxes(boxes, dets.conf.astype(float).tolist(), 0.0, iou)
    keep = np.array(keep, int).reshape(-1)
    return dets.subset(keep)


class YoloDetector:
    def __init__(self, s: DetectionSettings):
        threads = cv2.getNumThreads()
        from ultralytics import YOLO

        # Importing Ultralytics calls cv2.setNumThreads(0) (for its DataLoader), which makes the
        # OpenCV work here (stabilization, colour conversion) single-threaded and ~5x slower.
        cv2.setNumThreads(max(threads, 1))
        self.s = s
        self.model = YOLO(s.model)
        self.obb = self.model.task == "obb"
        names = self.model.names
        wanted = set(s.classes)
        self.class_map = {
            int(i): CLASSES.index(ALIASES[n.lower()])
            for i, n in names.items()
            if n.lower() in ALIASES and ALIASES[n.lower()] in wanted
        }
        if not self.class_map:
            raise ValueError(f"Model {s.model} has no vehicle classes (names: {list(names.values())})")
        import torch

        self.device = s.device or (0 if torch.cuda.is_available() else "cpu")
        half = s.half and self.device != "cpu"
        # newer Ultralytics replaced `half` with `quantize`
        self.precision = {"quantize": 16 if half else 32} if _has_quantize() else {"half": half}

    MAX_IMAGES_PER_CALL = 32  # crops per GPU call (frames x tiles)

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[Detections]:
        """Detect each frame independently; several frames (and their tiles) share one GPU call."""
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        tiles = make_tiles(w, h, self.s.tile, self.s.tile_overlap) if self.s.tile else [(0, 0, w, h)]
        crops = [f[y0:y1, x0:x1] for f in frames for x0, y0, x1, y1 in tiles]
        results = []
        for k in range(0, len(crops), self.MAX_IMAGES_PER_CALL):
            results += self.model.predict(
                crops[k:k + self.MAX_IMAGES_PER_CALL], imgsz=self.s.imgsz, conf=self.s.conf, iou=self.s.iou,
                device=self.device, classes=list(self.class_map), verbose=False, **self.precision,
            )
        nt = len(tiles)
        return [self._collect(tiles, results[j * nt:(j + 1) * nt]) for j in range(len(frames))]

    def _collect(self, tiles, results) -> Detections:
        corners, conf, cls = [], [], []
        for (x0, y0, _, _), r in zip(tiles, results):
            if self.obb:
                if r.obb is None or len(r.obb) == 0:
                    continue
                c = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
                cf = r.obb.conf.cpu().numpy()
                cl = r.obb.cls.cpu().numpy().astype(int)
            else:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy()
                c = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
            corners.append(c + np.array([x0, y0]))
            conf.append(cf)
            cls.append(np.array([self.class_map[int(k)] for k in cl], int))
        if not corners:
            return Detections.empty()
        dets = Detections(np.concatenate(corners).astype(np.float64), np.concatenate(conf), np.concatenate(cls))
        return _nms(dets, self.s.iou) if len(tiles) > 1 else dets


class BlobDetector:
    """Detects saturated colour blobs (e.g. the moving rectangle of a synthetic clip)."""

    def __init__(self, s: DetectionSettings):
        self.s = s

    def __call__(self, frame: np.ndarray) -> Detections:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 1] >= self.s.blob_min_saturation).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners = []
        for c in contours:
            if cv2.contourArea(c) < self.s.blob_min_area:
                continue
            corners.append(cv2.boxPoints(cv2.minAreaRect(c)))
        if not corners:
            return Detections.empty()
        n = len(corners)
        return Detections(np.array(corners, np.float64), np.full(n, 0.99), np.zeros(n, int))

    def detect_batch(self, frames: list[np.ndarray]) -> list[Detections]:
        return [self(f) for f in frames]


def make_detector(s: DetectionSettings):
    return BlobDetector(s) if s.type == "blob" else YoloDetector(s)
