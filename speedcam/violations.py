"""Red-light running and solid-line crossings, from the tracks of the analysis.

Red lights: each setup has lights (circles around signal heads, reference px) and stop lines.
The light's phase on every frame comes from the colour of its bright pixels (or from typed red
intervals). A vehicle runs the red when its front reaches a stop line, in the counted direction,
while the phase is red.

Solid lines: a vehicle crossing the line anywhere along it is reported ("centre" mode: its centre
goes from one side to the other, i.e. a lane change; "wheels" mode: any part of its box gets over
the line, e.g. straddling).

Crossings are found in reference px, where the lines were drawn, on the smoothed vehicle positions
of the speed analysis. A crossing only counts once the position is `hysteresis_m` clear of the line
on both sides: the tracked position wobbles by a few tens of cm, and a vehicle driving along a
line would otherwise "cross" it back and forth.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import calibration
from .config import Calibration, ProjectConfig, RedLightSetup, ViolationSettings
from .stabilization import transform_points

STATE_NAMES = ["red", "amber", "green", "not red"]  # "not red": outside typed red intervals
UNKNOWN = -1
RED, AMBER, GREEN, NOT_RED = 0, 1, 2, 3
CORNER_COLS = [f"c{k}{a}" for k in range(4) for a in "xy"]
SAMPLES_FILE = "lights.npz"


def state_name(code: int) -> str:
    return STATE_NAMES[code] if code >= 0 else "unknown"


# -- light colour ---------------------------------------------------------------------------
def light_list(cal: Calibration) -> list[tuple[int, int]]:
    """(setup index, light index) of every light, in a fixed order (columns of the samples)."""
    return [(s, k) for s, r in enumerate(cal.red_lights) for k in range(len(r.lights))]


def lights_key(cal: Calibration, vs: ViolationSettings) -> str:
    """Identifies stored samples: changes when a light moves or the colour thresholds change."""
    src = {"ref": cal.reference_frame,
           "lights": [[(lt.px, lt.radius_px) for lt in r.lights] for r in cal.red_lights],
           "v": vs.light_min_value, "s": vs.light_min_saturation}
    return hashlib.sha1(json.dumps(src, sort_keys=True).encode()).hexdigest()


def colour_counts(frame: np.ndarray, center: tuple[float, float], radius: float,
                  vs: ViolationSettings) -> np.ndarray:
    """Number of bright, saturated (red, amber, green) pixels in a circle of `frame` (BGR)."""
    x, y, r = float(center[0]), float(center[1]), max(1.0, float(radius))
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(np.floor(x - r))), min(w, int(np.ceil(x + r)) + 1)
    y0, y1 = max(0, int(np.floor(y - r))), min(h, int(np.ceil(y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(3, np.int32)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = inside & (V >= vs.light_min_value) & (S >= vs.light_min_saturation)
    # OpenCV hue is 0-179. Traffic green is bluish (~75-95); amber ~15-30
    red = lit & ((H <= 10) | (H >= 160))
    amber = lit & (H > 10) & (H <= 35)
    green = lit & (H >= 40) & (H <= 105)
    return np.array([red.sum(), amber.sum(), green.sum()], np.int32)


class LightSampler:
    """Reads every light's colour counts on a frame; light positions are in reference px."""

    def __init__(self, cal: Calibration, vs: ViolationSettings):
        self.vs = vs
        self.idx = light_list(cal)
        self.px = np.array([cal.red_lights[s].lights[k].px for s, k in self.idx], float).reshape(-1, 2)
        self.r = np.array([cal.red_lights[s].lights[k].radius_px for s, k in self.idx], float)

    def __len__(self) -> int:
        return len(self.idx)

    def sample(self, frame: np.ndarray, T: Optional[np.ndarray] = None) -> np.ndarray:
        """`T` maps frame px -> reference px (the stabilization transform); None = identity."""
        pts = self.px if T is None else transform_points(np.linalg.inv(T), self.px)
        return np.array([colour_counts(frame, p, r, self.vs) for p, r in zip(pts, self.r)],
                        np.int32).reshape(-1, 3)


def save_samples(out: Path, key: str, frames: list[int], counts: list[np.ndarray]) -> None:
    np.savez_compressed(out / SAMPLES_FILE, key=np.array(key), frames=np.array(frames, np.int64),
                        counts=np.array(counts, np.int32) if counts else np.zeros((0, 0, 3), np.int32))


def load_samples(out: Path, key: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    p = out / SAMPLES_FILE
    if not p.exists():
        return None
    d = np.load(p)
    if str(d["key"]) != key:
        return None
    return d["frames"], d["counts"]


def needs_samples(cal: Calibration) -> bool:
    return any(r.state_source == "colour" and r.lights for r in cal.red_lights)


def sample_lights(cfg: ProjectConfig, out: Path, reporter=None) -> tuple[np.ndarray, np.ndarray]:
    """A decode-only pass over the analysed frames, for lights added or moved after the analysis."""
    from .video import iter_frames

    tf = np.load(out / "transforms.npz")
    T_frames, T_all = tf["frames"], tf["T"]
    sampler = LightSampler(cfg.calibration, cfg.violations)
    frames, counts = [], []
    if len(T_frames) and len(sampler):
        lo, hi = int(T_frames[0]), int(T_frames[-1]) + 1
        k = 0
        for i, frame in iter_frames(cfg.source, lo, hi):
            while k < len(T_frames) - 1 and T_frames[k] < i:
                k += 1
            frames.append(i)
            counts.append(sampler.sample(frame, T_all[k]))
            if reporter and len(frames) % 25 == 0:
                reporter.progress("reading traffic lights", len(frames), hi - lo)
                if reporter.cancelled():
                    raise InterruptedError("cancelled")
    save_samples(out, lights_key(cfg.calibration, cfg.violations), frames, counts)
    return np.array(frames, np.int64), np.array(counts, np.int32).reshape(len(frames), len(sampler), 3)


def colour_codes(counts: np.ndarray, fps: float, vs: ViolationSettings) -> np.ndarray:
    """Per-frame phase code from the summed colour counts of one setup's lights, (n, 3)."""
    n = len(counts)
    if n == 0:
        return np.zeros(0, int)
    raw = np.where(counts.max(1) >= vs.light_min_pixels, counts.argmax(1), UNKNOWN)
    # bridge frames where the (LED) light reads dark
    hold = int(round(vs.light_hold_s * fps))
    codes = raw.copy()
    last, since = UNKNOWN, 10 ** 9
    for i in range(n):
        if raw[i] != UNKNOWN:
            last, since = raw[i], 0
        else:
            since += 1
            if last != UNKNOWN and since <= hold:
                codes[i] = last
    # majority vote over a short window
    w = int(round(vs.light_smooth_s * fps)) | 1
    if w > 1:
        onehot = np.zeros((n, 4))
        onehot[np.arange(n), codes + 1] = 1  # column 0 = unknown
        c = np.vstack([np.zeros((1, 4)), np.cumsum(onehot, 0)])
        lo = np.clip(np.arange(n) - w // 2, 0, n)
        hi = np.clip(np.arange(n) + w // 2 + 1, 0, n)
        win = c[hi] - c[lo]
        best = win.argmax(1) - 1
        # keep the frame's own reading on ties
        own = win[np.arange(n), codes + 1]
        codes = np.where(own >= win.max(1), codes, best)
    return codes.astype(int)


def manual_codes(frames: np.ndarray, fps: float, setup: RedLightSetup) -> np.ndarray:
    t = frames / fps
    codes = np.full(len(frames), NOT_RED)
    for a, b in setup.red_intervals:
        codes[(t >= min(a, b)) & (t < max(a, b))] = RED
    return codes


def runs(frames: np.ndarray, codes: np.ndarray, fps: float) -> list[dict]:
    """Phase intervals: [{t0, t1, state}] in seconds of real time."""
    out = []
    if not len(codes):
        return out
    edges = np.nonzero(np.diff(codes))[0] + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(codes)]
    for a, b in zip(starts, ends):
        out.append({"t0": round(float(frames[a] / fps), 3),
                    "t1": round(float((frames[b - 1] + 1) / fps), 3), "state": state_name(int(codes[a]))})
    return out


@dataclass
class Phases:
    name: str
    source: str
    frames: np.ndarray
    codes: np.ndarray
    read_share: Optional[float] = None  # colour: share of frames where a colour was read

    def at(self, t: float, fps: float) -> tuple[int, Optional[float]]:
        """(phase code, seconds since this phase began) at time t."""
        if not len(self.frames):
            return UNKNOWN, None
        k = int(np.clip(np.searchsorted(self.frames, t * fps), 0, len(self.frames) - 1))
        if abs(self.frames[k] - t * fps) > 1.5:
            return UNKNOWN, None
        c = self.codes[k]
        j = k
        while j > 0 and self.codes[j - 1] == c:
            j -= 1
        # time since the phase began; None when it began before the analysed range (unknown start)
        since = None if j == 0 else float(t - self.frames[j] / fps)
        return int(c), since


def light_phases(cfg: ProjectConfig, fps: float, frame_range: np.ndarray,
                 samples: Optional[tuple[np.ndarray, np.ndarray]]) -> tuple[list[Phases], list[str]]:
    cal, vs = cfg.calibration, cfg.violations
    warnings: list[str] = []
    order = light_list(cal)
    out = []
    for s, setup in enumerate(cal.red_lights):
        if setup.state_source == "manual":
            out.append(Phases(setup.name, "manual", frame_range, manual_codes(frame_range, fps, setup)))
            continue
        cols = [j for j, (si, _) in enumerate(order) if si == s]
        if not cols:
            warnings.append(f"{setup.name}: no light marked, so its phase is unknown. Mark the light, "
                            "or type its red intervals.")
            out.append(Phases(setup.name, "colour", frame_range, np.full(len(frame_range), UNKNOWN)))
            continue
        if samples is None:
            warnings.append(f"{setup.name}: the light was moved or added after the analysis; recompute to "
                            "read its colour.")
            out.append(Phases(setup.name, "colour", frame_range, np.full(len(frame_range), UNKNOWN)))
            continue
        frames, counts = samples
        summed = counts[:, cols, :].sum(1)
        codes = colour_codes(summed, fps, vs)
        read = float((summed.max(1) >= vs.light_min_pixels).mean()) if len(summed) else 0.0
        if read < 0.5:
            warnings.append(f"{setup.name}: a colour was read on only {read:.0%} of frames. Check the circle sits "
                            "on the lit lamps (zoom in), or type the red intervals instead.")
        out.append(Phases(setup.name, "colour", frames, codes, read))
    return out, warnings


# -- crossings ------------------------------------------------------------------------------
def signed_distance(P: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance (px) of points to a polyline, positive on its right (image coordinates,
    walking from the first point to the last), and whether the nearest point is inside the
    polyline rather than past one of its ends."""
    best_d = np.full(len(P), np.inf)
    sign = np.zeros(len(P))
    beside = np.zeros(len(P), bool)
    last = len(line) - 2
    for k in range(len(line) - 1):
        a, b = line[k], line[k + 1]
        d = b - a
        L2 = float(d @ d)
        if L2 <= 0:
            continue
        u = ((P - a) @ d) / L2
        dist = np.linalg.norm(P - (a + np.clip(u, 0, 1)[:, None] * d), axis=1)
        cross = d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])
        # only the two outer ends can be "past the line"; inner vertices join two segments
        inner = np.ones(len(P), bool)
        if k == 0:
            inner &= u >= 0
        if k == last:
            inner &= u <= 1
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        sign = np.where(better, np.sign(cross), sign)
        beside = np.where(better, inner, beside)
    return sign * best_d, beside


def crossings(t: np.ndarray, P: np.ndarray, line: np.ndarray, hyst_px: float) -> list[dict]:
    """Crossings of a polyline by a trajectory P(t) (reference px), with hysteresis.

    Each: {t, side_to (+1 = onto the right side), point}."""
    if len(P) < 2 or len(line) < 2:
        return []
    d, beside = signed_distance(P, line)
    out = []
    side, i_side = 0, 0  # last side the vehicle was clearly on, and the last frame it was there
    for j in range(len(d)):
        s = 1 if d[j] >= hyst_px else -1 if d[j] <= -hyst_px else 0
        if s == 0:
            continue
        if side != 0 and s != side:
            # the zero crossing between leaving the old side and reaching the new one
            for k in range(i_side, j):
                if np.sign(d[k]) != np.sign(d[k + 1]) or d[k + 1] == 0:
                    frac = d[k] / (d[k] - d[k + 1]) if d[k] != d[k + 1] else 0.0
                    pt = P[k] + frac * (P[k + 1] - P[k])
                    if beside[k] or beside[k + 1]:
                        out.append({"t": float(t[k] + frac * (t[k + 1] - t[k])), "side_to": s, "point": pt})
                    break
        side, i_side = s, j
    return out


def _counts(direction: str, side_to: int) -> bool:
    return direction == "both" or (direction == "forward") == (side_to > 0)


def _px_per_m(H: np.ndarray, line: np.ndarray) -> float:
    return 1.0 / max(calibration.local_scale(H, tuple(np.asarray(line, float).mean(0))), 1e-9)


def _front_time(t_c: float, side_to: int, line: np.ndarray, boxes: Optional[tuple[np.ndarray, np.ndarray]],
                fps: float) -> float:
    """When the vehicle's leading box corner reached the line (before its centre did)."""
    if boxes is None:
        return t_c
    bt, corners = boxes
    sel = np.nonzero((bt >= t_c - 3.0) & (bt <= t_c + 1.0 / fps))[0]
    if len(sel) < 2:
        return t_c
    lead = np.array([(side_to * signed_distance(c, line)[0]).max() for c in corners[sel]])
    over = np.nonzero(lead >= 0)[0]
    if not len(over) or over[0] == 0:
        return t_c
    k = over[0]
    a, b = lead[k - 1], lead[k]
    frac = -a / (b - a) if b != a else 0.0
    return float(min(t_c, bt[sel[k - 1]] + frac * (bt[sel[k]] - bt[sel[k - 1]])))


def _speed_at(r, t: float, fps: float) -> Optional[float]:
    k = int(round(t * fps)) - int(r.frames[0])
    if 0 <= k < len(r.inst_kmh) and np.isfinite(r.inst_kmh[k]):
        return round(float(r.inst_kmh[k]), 1)
    return None


def _over_spans(boxes: Optional[tuple[np.ndarray, np.ndarray]], t: np.ndarray, P: np.ndarray,
                line: np.ndarray, min_px: float) -> list[tuple[float, float, float]]:
    """Time spans (t0, t1, max px over) when a corner of the vehicle's box is over the line, on
    the far side from its centre, by at least min_px."""
    if boxes is None:
        return []
    bt, corners = boxes
    centre = np.column_stack([np.interp(bt, t, P[:, 0]), np.interp(bt, t, P[:, 1])])
    dc, beside_c = signed_distance(centre, line)
    over = np.zeros(len(bt))
    for k in range(len(bt)):
        dk, bk = signed_distance(corners[k], line)
        if bk.any() or beside_c[k]:
            over[k] = max(0.0, float((-(np.sign(dc[k]) or 1.0) * dk).max()))
    on = over >= min_px
    spans, k = [], 0
    while k < len(on):
        if not on[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(on) and on[j + 1] and bt[j + 1] - bt[j] <= 0.5:
            j += 1
        spans.append((float(bt[k]), float(bt[j] + (bt[1] - bt[0] if len(bt) > 1 else 0)), float(over[k:j + 1].max())))
        k = j + 1
    return spans


def detect(cfg: ProjectConfig, cal_res, tracks: pd.DataFrame, renders: dict, fps: float,
           samples: Optional[tuple[np.ndarray, np.ndarray]], frame_range: np.ndarray) -> dict:
    """All violations of a clip. `renders` are the speed analysis' tracks (smoothed world
    positions); `tracks` the raw rows, for box corners."""
    cal, vs = cfg.calibration, cfg.violations
    H_inv = np.linalg.inv(cal_res.H)
    phases, warnings = light_phases(cfg, fps, frame_range, samples) if cal.red_lights else ([], [])

    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(tracks) and all(c in tracks for c in CORNER_COLS):
        for tid, g in tracks.groupby("track_id"):
            g = g.sort_values("frame")
            boxes[int(tid)] = (g["frame"].to_numpy() / fps, g[CORNER_COLS].to_numpy(float).reshape(-1, 4, 2))
    traj = {tid: (r.frames / fps, calibration.apply_h(H_inv, r.world)) for tid, r in renders.items()}

    red_events: list[dict] = []
    for setup, ph in zip(cal.red_lights, phases):
        for li, sl in enumerate(setup.stop_lines):
            line = np.array(sl.points, float)
            if len(line) < 2:
                continue
            hyst = vs.hysteresis_m * _px_per_m(cal_res.H, line)
            for tid, (t, P) in traj.items():
                hits = [c for c in crossings(t, P, line, hyst) if _counts(sl.direction, c["side_to"])]
                if not hits:
                    continue
                c = hits[0]
                t_front = _front_time(c["t"], c["side_to"], line, boxes.get(tid), fps)
                code, since = ph.at(t_front, fps)
                is_red = code == RED or (vs.amber_is_red and code == AMBER)
                violation = None if code == UNKNOWN else bool(is_red and (since is None or since >= vs.red_grace_s))
                r = renders[tid]
                red_events.append({
                    "type": "red_light", "name": setup.name, "line": li + 1, "track_id": int(tid), "class": r.cls,
                    "t_s": round(t_front, 3), "t_centre_s": round(c["t"], 3), "frame": int(round(t_front * fps)),
                    "light": state_name(code), "phase_for_s": None if since is None else round(since, 2),
                    "violation": violation, "speed_kmh": _speed_at(r, t_front, fps),
                })
    # a vehicle crossing two stop lines of one setup is counted once (the first)
    seen, dedup = set(), []
    for e in sorted(red_events, key=lambda e: e["t_s"]):
        if (e["name"], e["track_id"]) not in seen:
            seen.add((e["name"], e["track_id"]))
            dedup.append(e)
    red_events = dedup

    line_events: list[dict] = []
    for sl in cal.solid_lines:
        line = np.array(sl.points, float)
        if len(line) < 2:
            continue
        ppm = _px_per_m(cal_res.H, line)
        hyst = vs.hysteresis_m * ppm
        for tid, (t, P) in traj.items():
            r = renders[tid]
            cross = crossings(t, P, line, hyst)
            # wheels: also spans with any box corner over the line by overlap_m (straddling)
            spans = _over_spans(boxes.get(tid), t, P, line, vs.overlap_m * ppm) if vs.solid_mode == "wheels" else []

            def event(t0: float, how: str, span=None) -> dict:
                return {"type": "solid_line", "name": sl.name, "track_id": int(tid), "class": r.cls,
                        "t_s": round(t0, 3), "frame": int(round(t0 * fps)), "how": how,
                        "duration_s": None if span is None else round(span[1] - span[0], 2),
                        "max_over_m": None if span is None else round(span[2] / ppm, 2),
                        "speed_kmh": _speed_at(r, t0, fps)}

            # a centre crossing always counts; in wheels mode it starts when the box got over
            used = set()
            for c in cross:
                k = next((k for k, s in enumerate(spans) if s[0] - 0.5 <= c["t"] <= s[1] + 0.5), None)
                if k is not None:
                    used.add(k)
                line_events.append(event(min(c["t"], spans[k][0]) if k is not None else c["t"], "crossed",
                                         spans[k] if k is not None else None))
            for k, s in enumerate(spans):
                if k not in used and s[1] - s[0] >= vs.min_duration_s:
                    line_events.append(event(s[0], "straddled", s))
    line_events.sort(key=lambda e: e["t_s"])

    return {
        "lights": [{"name": p.name, "source": p.source, "read_share": p.read_share,
                    "phases": runs(p.frames, p.codes, fps)} for p in phases],
        "red_light": red_events,
        "solid_line": line_events,
        "n_red_light": sum(1 for e in red_events if e["violation"]),
        "n_solid_line": len(line_events),
        "warnings": warnings,
    }


def write_csv(out: Path, v: dict) -> None:
    cols = ["type", "name", "line", "track_id", "class", "t_s", "t_centre_s", "frame", "light", "phase_for_s",
            "violation", "how", "duration_s", "max_over_m", "speed_kmh"]
    rows = [{**e, "violation": e.get("violation", True)} for e in v["red_light"] + v["solid_line"]]
    pd.DataFrame(rows, columns=cols).sort_values("t_s").to_csv(out / "violations.csv", index=False)
