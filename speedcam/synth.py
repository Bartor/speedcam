"""Synthetic validation clip: a coloured rectangle at a known speed over a textured
road, with camera sway and per-frame position jitter."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Calibration, DetectionSettings, ProjectConfig, Segment
from .video import FFmpegWriter


def road_texture(w: int, h: int, m_per_px: float, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(110, 18, (h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    coarse = cv2.resize(rng.normal(0, 14, (h // 16 + 1, w // 16 + 1)).astype(np.float32), (w, h))
    img = np.clip(base + coarse, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # speckles (gravel, cracks) give the feature matcher something to hold on to
    for _ in range(int(w * h / 900)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        v = int(rng.choice([40, 200]))
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), (v, v, v), -1)
    lane = 3.5 / m_per_px
    cy = h / 2
    for k in (-1, 0, 1):
        y = int(cy + k * lane)
        dash, gap = int(3 / m_per_px), int(6 / m_per_px)
        if k == 0:
            for x in range(0, w, dash + gap):
                cv2.rectangle(img, (x, y - 2), (x + dash, y + 2), (235, 235, 235), -1)
        else:
            cv2.line(img, (0, y), (w, y), (230, 230, 230), 4)
    for y in (int(cy - 1.6 * lane), int(cy + 1.6 * lane)):
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 10)
    return img


def make_synthetic(out: str | Path, speed_kmh: float = 50.0, m_per_px: float = 0.05,
                   size: tuple[int, int] = (1280, 720), capture_fps: float = 120.0,
                   container_fps: float = 30.0, jitter_px: float = 0.5, sway_px: float = 4.0,
                   sway_deg: float = 0.3, seed: int = 0, gap_m: float = 24.0,
                   textured: bool = False) -> ProjectConfig:
    """Write the clip and return a ready-to-run ProjectConfig (blob detector, scale calibration).

    `textured` paints a fixed pattern on the car (shades of red) so optical flow can follow it."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    W, H = size
    bg = road_texture(W, H, m_per_px, rng)
    v_px = speed_kmh / 3.6 / m_per_px / capture_fps  # px per frame
    car_l, car_w = 4.5 / m_per_px, 1.8 / m_per_px
    x0 = -car_l
    n_frames = int(np.ceil((W + 2 * car_l) / v_px))
    lane_y = H / 2 + 3.5 / m_per_px / 2
    if textured:
        cl, cw = int(np.ceil(car_l)), int(np.ceil(car_w))
        shade = cv2.GaussianBlur(rng.normal(1.0, 0.35, (cw, cl)).astype(np.float32), (0, 0), 1.5)
        car_img = np.clip(np.float32([40, 40, 210]) * np.clip(shade, 0.55, 1.2)[..., None], 0, 255)

    writer = FFmpegWriter(out, W, H, container_fps, encoder="x264", quality=14)
    phase = rng.uniform(0, 2 * np.pi, 3)
    for i in range(n_frames):
        t = i / capture_fps
        scene = bg.copy()
        cx = x0 + v_px * i + rng.normal(0, jitter_px)
        cy = lane_y + rng.normal(0, jitter_px)
        if textured:
            A = np.float32([[1, 0, cx - car_l / 2], [0, 1, cy - car_w / 2]])
            car = cv2.warpAffine(car_img, A, (W, H), flags=cv2.INTER_CUBIC)
            a = cv2.warpAffine(np.ones(car_img.shape[:2], np.float32), A, (W, H))[..., None]
            scene = (scene * (1 - a) + car * a).astype(np.uint8)
        else:
            rect = cv2.boxPoints(((cx, cy), (car_l, car_w), 0.0))
            cv2.fillPoly(scene, [np.round(rect * 16).astype(np.int32)], (40, 40, 210), cv2.LINE_AA, shift=4)
        # camera sway: smooth, zero at t=0 so frame 0 is the reference
        dx = sway_px * (np.sin(2 * np.pi * 0.7 * t + phase[0]) - np.sin(phase[0]))
        dy = sway_px * (np.sin(2 * np.pi * 0.5 * t + phase[1]) - np.sin(phase[1]))
        ang = sway_deg * (np.sin(2 * np.pi * 0.3 * t + phase[2]) - np.sin(phase[2]))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        M[:, 2] += (dx, dy)
        frame = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.close()

    xa = W / 2 - gap_m / m_per_px / 2
    xb = W / 2 + gap_m / m_per_px / 2
    y0, y1 = H / 2 - 2 * 3.5 / m_per_px, H / 2 + 2 * 3.5 / m_per_px
    seg_px = 20.0 / m_per_px
    cal = Calibration(
        reference_frame=0,
        segments=[Segment(a=(100, H - 40), b=(100 + seg_px, H - 40), meters=20.0),
                  Segment(a=(W - 60, 40), b=(W - 60, 40 + 12 / m_per_px), meters=12.0)],
        line_a=((xa, y0), (xa, y1)), line_b=((xb, y0), (xb, y1)),
    )
    return ProjectConfig(source=str(out.resolve()), capture_fps=capture_fps,
                         real_duration_s=n_frames / capture_fps, calibration=cal,
                         detection=DetectionSettings(type="blob"))
