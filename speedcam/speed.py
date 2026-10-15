"""Per-track speed estimation from world trajectories.

Primary: arc length of the smoothed trajectory vs. time, linear regression over
the segment between line A and line B (handles lane changes and curves).
Audit: sub-frame line-crossing times, speed = chord between crossings / Δt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import SpeedSettings
from .detection import CLASSES

INVALIDATING = {"accel", "residual", "gap", "few_points"}


@dataclass
class Crossing:
    t: float
    point: np.ndarray
    count: int


@dataclass
class TrackRender:
    """Everything the overlay needs about one track."""

    frames: np.ndarray  # contiguous frame indices
    world: np.ndarray  # (n, 2) smoothed world positions
    provisional_kmh: np.ndarray  # (n,) running-fit speed between the lines, NaN elsewhere
    t_first: Optional[float] = None
    t_second: Optional[float] = None
    result: Optional[dict] = None
    cls: str = "car"
    crossed: str = ""  # which lines the smoothed track crosses: "", "A", "B", "AB"


def odd_window(n: int, poly: int, length: int) -> int:
    w = max(n, poly + 2)
    w += (w + 1) % 2
    if w > length:
        w = length if length % 2 else length - 1
    return w


def smooth(xy: np.ndarray, fps: float, s: SpeedSettings) -> np.ndarray:
    win = odd_window(int(round(s.smooth_window_s * fps)), s.smooth_poly, len(xy))
    if win <= s.smooth_poly:
        return xy.copy()
    return savgol_filter(xy, win, s.smooth_poly, axis=0, mode="interp")


def first_crossing(t: np.ndarray, P: np.ndarray, line: np.ndarray, extend: float) -> Optional[Crossing]:
    """Sub-frame crossing of a polyline P(t) with segment `line` (extended by `extend` × length)."""
    l0, l1 = line
    d = l1 - l0
    L2 = float(d @ d)
    side = d[0] * (P[:, 1] - l0[1]) - d[1] * (P[:, 0] - l0[0])
    hits: list[tuple[float, np.ndarray]] = []
    for i in np.nonzero(np.signbit(side[:-1]) != np.signbit(side[1:]))[0]:
        denom = side[i] - side[i + 1]
        frac = 0.0 if denom == 0 else side[i] / denom
        pt = P[i] + frac * (P[i + 1] - P[i])
        u = float((pt - l0) @ d) / L2
        if -extend <= u <= 1 + extend:
            hits.append((t[i] + frac * (t[i + 1] - t[i]), pt))
    if not hits:
        return None
    return Crossing(hits[0][0], hits[0][1], len(hits))


def _running_slopes(ts: np.ndarray, ss: np.ndarray, min_span: float) -> np.ndarray:
    """Least-squares slope over samples [0..i] for every i (cumulative sums)."""
    n = np.arange(1, len(ts) + 1)
    St, Ss = np.cumsum(ts), np.cumsum(ss)
    Stt, Sts = np.cumsum(ts * ts), np.cumsum(ts * ss)
    den = n * Stt - St ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * Sts - St * Ss) / den
    slope[(ts - ts[0]) < min_span] = np.nan
    return slope


def analyze_track(track_id: int, frames: np.ndarray, xy: np.ndarray, fps: float,
                  line_a: np.ndarray, line_b: np.ndarray, s: SpeedSettings,
                  cls: str = "car", mean_conf: float = 1.0) -> Optional[TrackRender]:
    order = np.argsort(frames)
    frames, xy = frames[order], xy[order]
    frames, uidx = np.unique(frames, return_index=True)
    xy = xy[uidx]
    if len(frames) < max(3, s.min_points):
        return None
    full = np.arange(frames[0], frames[-1] + 1)
    P_raw = np.column_stack([np.interp(full, frames, xy[:, 0]), np.interp(full, frames, xy[:, 1])])
    max_gap_s = float((np.diff(frames).max() - 1) / fps) if len(frames) > 1 else 0.0
    t = full / fps
    P = smooth(P_raw, fps, s)

    render = TrackRender(frames=full, world=P, provisional_kmh=np.full(len(full), np.nan), cls=cls)
    ca = first_crossing(t, P, line_a, s.line_extend)
    cb = first_crossing(t, P, line_b, s.line_extend)
    render.crossed = ("A" if ca else "") + ("B" if cb else "")
    if ca is None or cb is None or ca.t == cb.t:
        return render

    direction = "A->B" if ca.t < cb.t else "B->A"
    first, second = (ca, cb) if ca.t < cb.t else (cb, ca)
    t_in, t_out = first.t, second.t
    dt = t_out - t_in

    # Arc length on a decimated trajectory (limits jitter-induced length bias).
    k = max(1, int(round(s.arc_dt_s * fps)))
    idx = np.arange(0, len(t), k)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    td, Pd = t[idx], P[idx]
    sd = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    inside = (td > t_in) & (td < t_out)
    ts = np.concatenate([[t_in], td[inside], [t_out]])
    ss = np.interp(ts, td, sd)

    flags: list[str] = []
    v_line = float(np.linalg.norm(second.point - first.point) / dt)
    path_len = float(ss[-1] - ss[0])
    if len(ts) >= 3:
        v_arc, _ = np.polyfit(ts, ss, 1)
    else:
        v_arc = path_len / dt
        flags.append("few_points")
    # residuals against a constant-acceleration fit: track jitter, not real speed change
    accel, quad_max, rms = 0.0, 0.0, 0.0
    if len(ts) >= 4:
        q = np.polyfit(ts, ss, 2)
        accel = float(2 * q[0])
        quad_res = ss - np.polyval(q, ts)
        quad_max = float(np.abs(quad_res).max())
        rms = float(np.sqrt(np.mean(quad_res ** 2)))
    v_arc = float(v_arc)
    disagreement = (v_arc - v_line) / v_line if v_line > 0 else np.inf

    if abs(accel) > s.max_accel:
        flags.append("accel")
    if quad_max > s.max_residual_m:
        flags.append("residual")
    if max_gap_s > s.max_gap_s:
        flags.append("gap")
    if abs(disagreement) > s.max_disagreement:
        flags.append("disagree")
    if ca.count > 1 or cb.count > 1:
        flags.append("multi_cross")
    invalid = INVALIDATING | ({"disagree"} if s.disagreement_invalidates else set())
    valid = not (set(flags) & invalid)

    result = {
        "track_id": int(track_id),
        "class": cls,
        "direction": direction,
        "t_entry_s": float(t_in),
        "t_exit_s": float(t_out),
        "frame_entry": float(t_in * fps),
        "frame_exit": float(t_out * fps),
        "dt_s": float(dt),
        "path_length_m": path_len,
        "chord_m": float(np.linalg.norm(second.point - first.point)),
        "speed_kmh": v_arc * 3.6,
        "speed_line_kmh": v_line * 3.6,
        "disagreement_pct": float(disagreement * 100),
        "fit_residual_m": rms,
        "accel_mps2": accel,
        "max_gap_s": max_gap_s,
        "n_points": int(len(frames)),
        "mean_conf": float(mean_conf),
        "valid": bool(valid),
        "flags": ";".join(flags),
    }

    # Provisional (running-fit) speed for frames between the crossings.
    s_full = np.interp(t, td, sd)
    between = (t > t_in) & (t < t_out)
    if between.any():
        tt = np.concatenate([[t_in], t[between]])
        st = np.concatenate([[np.interp(t_in, td, sd)], s_full[between]])
        render.provisional_kmh[between] = _running_slopes(tt, st, 0.15)[1:] * 3.6
    render.t_first, render.t_second, render.result = t_in, t_out, result
    return render


def compute_speeds(tracks: pd.DataFrame, fps: float, line_a: np.ndarray, line_b: np.ndarray,
                   s: SpeedSettings) -> tuple[list[dict], dict[int, TrackRender]]:
    """`tracks` needs columns frame, track_id, x_m, y_m, cls, conf."""
    results: list[dict] = []
    renders: dict[int, TrackRender] = {}
    df = tracks[tracks["track_id"] >= 0]
    for tid, g in df.groupby("track_id"):
        cls = CLASSES[int(g["cls"].mode().iloc[0])]
        r = analyze_track(int(tid), g["frame"].to_numpy(), g[["x_m", "y_m"]].to_numpy(float), fps,
                          np.asarray(line_a, float), np.asarray(line_b, float), s, cls,
                          float(g["conf"].mean()))
        if r is None:
            continue
        renders[int(tid)] = r
        if r.result:
            results.append(r.result)
    results.sort(key=lambda r: r["t_entry_s"])
    return results, renders
