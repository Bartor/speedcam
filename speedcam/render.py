"""Annotated video export, drawn from stored tracks (no new analysis)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import calibration
from .config import ProjectConfig
from .pipeline import Reporter, resolve_capture_fps
from .speed import TrackRender, compute_speeds
from .stabilization import transform_points
from .timing import probe
from .video import FFmpegReader, FFmpegWriter

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, LINE_A, LINE_B = (255, 255, 255), (170, 170, 170), (255, 200, 0), (255, 0, 200)


def speed_color(kmh: float, limit: float) -> tuple[int, int, int]:
    if kmh <= limit:
        return (60, 200, 60)
    if kmh <= limit + 10:
        return (0, 165, 255)
    return (40, 40, 230)


def _label(img, text, org, fs, color, bg=(0, 0, 0)):
    th = max(1, int(round(fs * 2)))
    (w, h), base = cv2.getTextSize(text, FONT, fs, th)
    x, y = int(org[0]), int(org[1])
    y = max(y, h + 4)
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), bg, -1)
    cv2.putText(img, text, (x, y), FONT, fs, color, th, cv2.LINE_AA)


class Overlay:
    def __init__(self, cfg: ProjectConfig, fps: float, cal, renders: dict[int, TrackRender],
                 tracks: pd.DataFrame, transforms: dict, size: tuple[int, int], src_size: tuple[int, int]):
        self.cfg, self.fps, self.cal, self.renders = cfg, fps, cal, renders
        self.rs = cfg.render
        self.W, self.H = size
        self.S = np.diag([size[0] / src_size[0], size[1] / src_size[1], 1.0])
        self.S_inv = np.linalg.inv(self.S)
        self.H_inv = np.linalg.inv(cal.H)
        self.fs = 0.65 * self.H / 1080
        self.lw = max(1, int(round(2 * self.H / 1080)))
        self.T_frames, self.T = transforms["frames"], transforms["T"]
        corner_cols = [f"c{k}{a}" for k in range(4) for a in "xy"]
        self.by_frame = {int(f): (g["track_id"].to_numpy(), g[corner_cols].to_numpy().reshape(-1, 4, 2))
                         for f, g in tracks.groupby("frame")}
        self.trail_frames = int(round(self.rs.trail_s * fps))
        # trails in reference px
        self.trail_px = {tid: calibration.apply_h(self.H_inv, r.world) for tid, r in renders.items()}

    def T_at(self, n: int) -> np.ndarray:
        if len(self.T_frames) == 0:
            return np.eye(3)
        k = int(np.clip(np.searchsorted(self.T_frames, n), 0, len(self.T_frames) - 1))
        return self.T[k]

    def draw(self, frame: np.ndarray, n: int) -> np.ndarray:
        T = self.T_at(n)
        if self.rs.mode == "stabilized":
            M = self.S @ T @ self.S_inv
            frame = cv2.warpPerspective(frame, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
            to_disp = self.S
        else:
            to_disp = self.S @ np.linalg.inv(T)
        t = n / self.fps
        cal = self.cfg.calibration
        gap = self.cal.line_gap_m

        if cal.roi and len(cal.roi) >= 3:
            roi = transform_points(to_disp, np.array(cal.roi, float)).astype(np.int32)
            cv2.polylines(frame, [roi], True, (120, 120, 120), 1, cv2.LINE_AA)
        for name, line, color in (("A", cal.line_a, LINE_A), ("B", cal.line_b, LINE_B)):
            if not line:
                continue
            p = transform_points(to_disp, np.array(line, float))
            cv2.line(frame, tuple(p[0].astype(int)), tuple(p[1].astype(int)), color, self.lw + 1, cv2.LINE_AA)
            txt = name + (f"  {gap:.1f} m" if gap and name == "A" else "")
            _label(frame, txt, p[0] + np.array([6, -6]), self.fs, color)

        ids, corners = self.by_frame.get(n, (np.zeros(0, int), np.zeros((0, 4, 2))))
        for tid, c in zip(ids, corners):
            r = self.renders.get(int(tid))
            res = r.result if r else None
            color, text = WHITE, f"#{tid}"
            if r:
                text = f"#{tid} {r.cls}"
            if res:
                if t >= r.t_second:
                    color = speed_color(res["speed_kmh"], self.rs.speed_limit_kmh) if res["valid"] else (200, 0, 200)
                    text = f"{res['speed_kmh']:.1f} km/h  dt {res['dt_s']:.3f}s" + ("" if res["valid"] else " ?")
                elif t > r.t_first:
                    k = n - r.frames[0]
                    v = r.provisional_kmh[k] if 0 <= k < len(r.provisional_kmh) else np.nan
                    color = GREY
                    text = f"~{v:.0f} km/h" if np.isfinite(v) else f"#{tid} ..."
            pts = transform_points(to_disp, c)
            cv2.polylines(frame, [pts.astype(np.int32)], True, color, self.lw, cv2.LINE_AA)
            if r is not None:
                k = n - r.frames[0]
                lo = max(0, k - self.trail_frames)
                if k > lo:
                    trail = transform_points(to_disp, self.trail_px[int(tid)][lo:k + 1]).astype(np.int32)
                    cv2.polylines(frame, [trail], False, color, self.lw, cv2.LINE_AA)
            _label(frame, text, (pts[:, 0].min(), pts[:, 1].min() - 8), self.fs, color)

        _label(frame, f"t {t:9.3f} s   frame {n}   {self.fps:g} fps   {self.rs.mode}", (12, 12 + 30 * self.fs),
               self.fs, WHITE)
        return frame


def render(cfg: ProjectConfig, out_dir: str | Path, dst: Optional[str | Path] = None,
           reporter: Optional[Reporter] = None) -> dict:
    reporter = reporter or Reporter()
    out = Path(out_dir)
    dst = Path(dst) if dst else out / "render.mp4"
    rs = cfg.render
    info = probe(cfg.source)
    fps = resolve_capture_fps(cfg, info, reporter)
    cal = calibration.compute(cfg.calibration, (info.width, info.height))
    tracks = pd.read_parquet(out / "tracks.parquet")
    world = calibration.apply_h(cal.H, tracks[["rx", "ry"]].to_numpy(float))
    tracks["x_m"], tracks["y_m"] = world[:, 0], world[:, 1]
    _, renders = compute_speeds(tracks, fps, cal.line_a_world, cal.line_b_world, cfg.speed)
    tf = np.load(out / "transforms.npz")
    transforms = {"frames": tf["frames"], "T": tf["T"]}

    W = max(2, int(round(info.width * rs.scale / 2)) * 2)
    H = max(2, int(round(info.height * rs.scale / 2)) * 2)
    overlay = Overlay(cfg, fps, cal, renders, tracks, transforms, (W, H), (info.width, info.height))

    out_fps = rs.output_fps or fps
    ratio = rs.playback_speed * fps / out_fps  # source frames per output frame
    n_out = int(math.floor((info.nb_frames - 1) / ratio)) + 1

    reader = FFmpegReader(cfg.source, W, H, rs.lut)
    writer = FFmpegWriter(dst, W, H, out_fps, rs.encoder, rs.quality)
    j = 0
    t0 = time.monotonic()
    try:
        for n, frame in enumerate(reader):
            if j >= n_out:
                break
            while j < n_out and int(round(j * ratio)) == n:
                writer.write(overlay.draw(frame.copy(), n))
                j += 1
            if j % 10 == 0:
                reporter.progress("rendering", j, n_out, fps=j / max(time.monotonic() - t0, 1e-6))
            if reporter.cancelled():
                raise InterruptedError("cancelled")
    except BaseException:
        writer.abort()
        raise
    finally:
        reader.close()
    writer.close()
    meta = {"file": dst.name, "output_fps": out_fps, "playback_speed": rs.playback_speed, "scale": rs.scale,
            "mode": rs.mode, "frames": j, "width": W, "height": H, "codec": writer.codec,
            "capture_fps": fps}
    (out / "render.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    reporter.progress("done", 1, 1)
    return meta


def overlay_data(cfg: ProjectConfig, out_dir: str | Path) -> dict:
    """Everything a client needs to draw the tracking overlay on the proxy itself.

    Geometry is in reference px; `T_inv[k]` maps reference px -> frame px for frame `T_frames[k]`.
    """
    out = Path(out_dir)
    info = probe(cfg.source)
    fps = resolve_capture_fps(cfg, info, Reporter())
    cal = calibration.compute(cfg.calibration, (info.width, info.height))
    tracks = pd.read_parquet(out / "tracks.parquet")
    world = calibration.apply_h(cal.H, tracks[["rx", "ry"]].to_numpy(float))
    tracks["x_m"], tracks["y_m"] = world[:, 0], world[:, 1]
    _, renders = compute_speeds(tracks, fps, cal.line_a_world, cal.line_b_world, cfg.speed)
    tf = np.load(out / "transforms.npz")
    T_inv = np.linalg.inv(tf["T"]) if len(tf["T"]) else np.zeros((0, 3, 3))
    H_inv = np.linalg.inv(cal.H)
    corner_cols = [f"c{k}{a}" for k in range(4) for a in "xy"]

    def r1(a) -> list:
        return np.round(np.asarray(a, float), 1).ravel().tolist()

    track_info = {}
    for tid, r in renders.items():
        res = r.result
        track_info[str(tid)] = {
            "f0": int(r.frames[0]),
            "trail": r1(calibration.apply_h(H_inv, r.world)),
            "prov": [None if not np.isfinite(v) else round(float(v), 1) for v in r.provisional_kmh]
            if res else None,
            "cls": r.cls,
            "t_first": r.t_first, "t_second": r.t_second,
            "speed": res["speed_kmh"] if res else None, "dt": res["dt_s"] if res else None,
            "valid": res["valid"] if res else None,
        }
    return {
        "width": info.width, "height": info.height, "capture_fps": fps, "container_fps": info.container_fps,
        "line_a": cfg.calibration.line_a, "line_b": cfg.calibration.line_b, "roi": cfg.calibration.roi,
        "line_gap_m": cal.line_gap_m, "speed_limit_kmh": cfg.render.speed_limit_kmh,
        "trail_frames": int(round(cfg.render.trail_s * fps)),
        "T_frames": tf["frames"].tolist(), "T_inv": np.round(T_inv.reshape(-1, 9), 7).tolist(),
        "det_frame": tracks["frame"].astype(int).tolist(), "det_tid": tracks["track_id"].astype(int).tolist(),
        "det_corners": r1(tracks[corner_cols].to_numpy()),
        "tracks": track_info,
    }
