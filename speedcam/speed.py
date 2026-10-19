"""Per-track speed estimation from world trajectories.

Primary: arc length of the smoothed trajectory vs. time, linear regression over
the segment between line A and line B (handles lane changes and curves).
Audit: sub-frame line-crossing times, speed = chord between crossings / Δt.
Momentary: |velocity| from a local quadratic fit over a short window (default 0.5 s),
giving a speed profile between the lines with its max, min and noise level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import SpeedSettings
from .detection import CLASSES
from .path import PathGeom

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
    inst_kmh: np.ndarray  # (n,) momentary speed along the whole track
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


def momentary_speed(P: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Speed (km/h) per frame from a local quadratic fit over `window_s` (Savitzky-Golay derivative).

    Shorter windows follow speed changes faster but pass more position noise; on 4K drone
    footage with flow positions 0.5 s keeps it around +-0.2 km/h without biasing the maximum upwards.
    """
    n = odd_window(int(round(window_s * fps)), 2, len(P))
    if n <= 2:
        v = np.gradient(P, 1 / fps, axis=0)
    else:
        v = savgol_filter(P, n, 2, deriv=1, delta=1 / fps, axis=0, mode="interp")
    return np.hypot(v[:, 0], v[:, 1]) * 3.6


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


def slope_se(ts: np.ndarray, res: np.ndarray, corr_s: float) -> float:
    """Standard error (m/s) of the arc-length regression slope.

    `res` are residuals against a constant-acceleration fit: real acceleration doesn't bias the
    least-squares slope over the section (it equals the mean speed), so it isn't counted as noise.
    Samples are correlated over about `corr_s` (smoothing, box wobble), so the plain SE is
    inflated to the number of independent samples in the section.
    """
    n = len(ts)
    sxx = float(((ts - ts.mean()) ** 2).sum())
    if n < 4 or sxx <= 0:
        return 0.0
    se = np.sqrt(float(res @ res) / (n - 3) / sxx)
    n_eff = max(4.0, (ts[-1] - ts[0]) / max(corr_s, 1e-6))
    return float(se * np.sqrt(max(1.0, (n - 3) / (n_eff - 3))))


def analyze_track(track_id: int, frames: np.ndarray, xy: np.ndarray, fps: float,
                  line_a: Optional[np.ndarray], line_b: Optional[np.ndarray], s: SpeedSettings,
                  cls: str = "car", mean_conf: float = 1.0,
                  paths: Optional[list[PathGeom]] = None) -> Optional[TrackRender]:
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

    if paths:
        # measure on every path; keep the one this vehicle covers most (a vehicle is counted once,
        # also where corridors overlap at an intersection; a turning car goes to its longer road)
        tries = [_analyze_path(track_id, frames, full, t, P_raw, P, max_gap_s, fps, g, s, cls, mean_conf)
                 for g in paths]
        measured = [r for r in tries if r.result]
        if measured:
            return max(measured, key=lambda r: r.result["path_length_m"])
        return next((r for r in tries if r.crossed == "short"), tries[0])
    inst = momentary_speed(P_raw, fps, s.momentary_window_s)
    render = TrackRender(frames=full, world=P, inst_kmh=inst, cls=cls)
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
    accel, quad_max, rms, avg_se = 0.0, 0.0, 0.0, None
    if len(ts) >= 4:
        q = np.polyfit(ts, ss, 2)
        accel = float(2 * q[0])
        quad_res = ss - np.polyval(q, ts)
        quad_max = float(np.abs(quad_res).max())
        rms = float(np.sqrt(np.mean(quad_res ** 2)))
        avg_se = slope_se(ts, quad_res, max(s.smooth_window_s, s.momentary_window_s))
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
        "speed_avg_se_kmh": None if avg_se is None else avg_se * 3.6,
        "disagreement_pct": float(disagreement * 100),
        "fit_residual_m": rms,
        "accel_mps2": accel,
        "max_gap_s": max_gap_s,
        "n_points": int(len(frames)),
        "mean_conf": float(mean_conf),
        "valid": bool(valid),
        "flags": ";".join(flags),
    }

    inst_ref = momentary_speed(P_raw, fps, max(2.0, 4 * s.momentary_window_s))
    dist = np.interp(t, td, sd) - np.interp(t_in, td, sd)
    result.update(_profile(t, inst, inst_ref, dist, t_in, t_out, v_arc * 3.6, direction))
    render.t_first, render.t_second, render.result = t_in, t_out, result
    return render


def _deriv_kmh(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """d/dt of a per-frame distance (m) from a local quadratic fit over `window_s`, in km/h."""
    n = odd_window(int(round(window_s * fps)), 2, len(x))
    v = np.gradient(x, 1 / fps) if n <= 2 else savgol_filter(x, n, 2, deriv=1, delta=1 / fps, mode="interp")
    return v * 3.6


def _longest_run(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """First and last index of the longest run of True."""
    if not mask.any():
        return None
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _analyze_path(track_id, frames, full, t, P_raw, P, max_gap_s, fps, path: PathGeom, s: SpeedSettings,
                  cls: str, mean_conf: float) -> TrackRender:
    """Measure a vehicle along the path of motion over all of it that the vehicle covers."""
    s_sm, lat_sm = path.project(P)
    inside = (np.abs(lat_sm) <= path.corridor_m) & (s_sm > path.lo + 0.25) & (s_sm < path.hi - 0.25)
    render = TrackRender(frames=full, world=P, inst_kmh=np.full(len(full), np.nan), cls=cls)
    run = _longest_run(inside)
    if run is None:
        return render
    i0, i1 = run
    cover = abs(s_sm[i1] - s_sm[i0])
    if cover < min(s.path_min_cover_m, 0.5 * (path.hi - path.lo)) or i1 - i0 < 3:
        render.crossed = "short"
        return render
    render.crossed = "AB"
    sign = 1.0 if s_sm[i1] > s_sm[i0] else -1.0
    direction = "A->B" if sign > 0 else "B->A"
    # distance actually travelled along the path (curvature-corrected), increasing with travel
    u_sm = sign * path.travelled(s_sm, lat_sm)
    s_raw, lat_raw = path.project(P_raw)
    u_raw = sign * path.travelled(s_raw, lat_raw)
    t_in, t_out = float(t[i0]), float(t[i1])
    dt = t_out - t_in
    ts, us = t[i0:i1 + 1], u_sm[i0:i1 + 1]

    v_path, _ = np.polyfit(ts, us, 1)
    accel, quad_max, rms, avg_se = 0.0, 0.0, 0.0, None
    if len(ts) >= 4:
        q = np.polyfit(ts, us, 2)
        accel = float(2 * q[0])
        quad_res = us - np.polyval(q, ts)
        quad_max = float(np.abs(quad_res).max())
        rms = float(np.sqrt(np.mean(quad_res ** 2)))
        avg_se = slope_se(ts, quad_res, max(s.smooth_window_s, s.momentary_window_s))
    # audit: the vehicle's own trajectory length over the same time (lane changes make it longer)
    k = max(1, int(round(s.arc_dt_s * fps)))
    idx = np.unique(np.r_[np.arange(i0, i1 + 1, k), i1])
    arc = float(np.linalg.norm(np.diff(P[idx], axis=0), axis=1).sum())
    v_arc = arc / dt
    v_path = float(v_path)
    disagreement = (v_path - v_arc) / v_arc if v_arc > 0 else np.inf

    flags: list[str] = []
    if abs(accel) > s.max_accel:
        flags.append("accel")
    if quad_max > s.max_residual_m:
        flags.append("residual")
    if max_gap_s > s.max_gap_s:
        flags.append("gap")
    if abs(disagreement) > s.max_disagreement:
        flags.append("disagree")
    invalid = INVALIDATING | ({"disagree"} if s.disagreement_invalidates else set())
    path_len = float(us[-1] - us[0])
    result = {
        "track_id": int(track_id), "class": cls, "direction": direction,
        "t_entry_s": t_in, "t_exit_s": t_out, "frame_entry": t_in * fps, "frame_exit": t_out * fps,
        "dt_s": float(dt), "path_length_m": path_len,
        "chord_m": float(np.linalg.norm(P[i1] - P[i0])),
        "speed_kmh": v_path * 3.6,
        "speed_line_kmh": v_arc * 3.6,  # audit: trajectory length / time
        "speed_avg_se_kmh": None if avg_se is None else avg_se * 3.6,
        "disagreement_pct": float(disagreement * 100),
        "fit_residual_m": rms, "accel_mps2": accel, "max_gap_s": max_gap_s,
        "n_points": int(len(frames)), "mean_conf": float(mean_conf),
        "valid": not (set(flags) & invalid), "flags": ";".join(flags),
        "s_entry_m": float(s_sm[i0]), "s_exit_m": float(s_sm[i1]),
        "coverage_pct": float(100 * cover / max(path.hi - path.lo, 1e-9)),
        "path": path.name,
    }
    inst = _deriv_kmh(u_raw, fps, s.momentary_window_s)
    inst_ref = _deriv_kmh(u_raw, fps, max(2.0, 4 * s.momentary_window_s))
    result.update(_profile(t, inst, inst_ref, u_sm - u_sm[i0], t_in, t_out, v_path * 3.6, direction))
    render.inst_kmh = inst
    render.t_first, render.t_second, render.result = t_in, t_out, result
    return render


def _profile(t, inst, inst_ref, dist, t_in, t_out, avg_kmh: float, direction: str) -> dict:
    """Momentary speed over the measured stretch: max/min, where the max happened, noise and a
    compact profile. `inst_ref` is the same curve over a much longer window, `dist` the distance
    travelled since the start of the stretch, all per frame."""
    seg = (t >= t_in) & (t <= t_out)
    if seg.sum() < 3:
        return {"speed_max_kmh": avg_kmh, "speed_min_kmh": avg_kmh, "speed_sigma_kmh": None, "t_max_s": None,
                "d_max_m": None, "speed_trend_pct": None, "profile": None}
    v = inst[seg]
    ts = t[seg]
    d = dist[seg]
    # noise: deviation from a much smoother version of the same curve (real speed changes are slow)
    ref = inst_ref[seg]
    sigma = float(np.std(v - ref))
    k = int(np.argmax(v))
    # speed change from the first to the second half of the stretch, as a share of the average,
    # signed towards line B: shared by all vehicles it points at calibration, not driving
    half = d <= d[-1] / 2
    trend = (v[~half].mean() - v[half].mean()) / avg_kmh * 100 if half.any() and (~half).any() else None
    if trend is not None and direction == "B->A":
        trend = -trend
    step = max(1, int(np.ceil(len(v) / 120)))
    return {
        "speed_max_kmh": float(v.max()),
        "speed_min_kmh": float(v.min()),
        "speed_sigma_kmh": sigma,
        "t_max_s": float(ts[k]),
        "d_max_m": float(d[k]),
        "speed_trend_pct": None if trend is None else float(trend),
        "profile": {"t": np.round(ts[::step] - t_in, 3).tolist(), "d": np.round(d[::step], 2).tolist(),
                    "v": np.round(v[::step], 2).tolist()},
    }


def calibration_check(results: list[dict], min_vehicles: int = 5) -> dict:
    """Use the traffic to check the calibration: drivers differ, geometry doesn't.

    If momentary speed changes by the same share along A->B for every vehicle, a wrong
    perspective/scale is the likely cause, and max speeds are biased by about that much.
    """
    tr = np.array([r["speed_trend_pct"] for r in results if r["valid"] and r.get("speed_trend_pct") is not None])
    out: dict = {"n": int(len(tr))}
    if len(tr) < min_vehicles:
        out["status"] = "insufficient"
        return out
    mean, sd = float(tr.mean()), float(tr.std(ddof=1))
    se = sd / np.sqrt(len(tr))
    out.update(mean_pct=mean, sd_pct=sd)
    if abs(mean) > 3 and abs(mean) > 3 * se:
        out["status"] = "suspect"
        out["message"] = (
            f"Every vehicle's momentary speed {'rises' if mean > 0 else 'falls'} by about {abs(mean):.0f}% "
            f"from the start to the end of the measured stretch, towards B (spread {sd:.1f}%). Real drivers vary; a shared trend points at the "
            "calibration (perspective or a wrong distance). Max speeds are biased by up to that much. "
            "Re-check the measured distances, and make the polygon span the whole A-B stretch.")
    else:
        out["status"] = "ok"
    return out


def compute_speeds(tracks: pd.DataFrame, fps: float, line_a: Optional[np.ndarray], line_b: Optional[np.ndarray],
                   s: SpeedSettings, paths: Optional[list[PathGeom]] = None) -> tuple[list[dict], dict[int, TrackRender]]:
    """`tracks` needs columns frame, track_id, x_m, y_m, cls, conf. With `paths`, each vehicle is
    measured along the path it covers most and lines A/B are ignored."""
    results: list[dict] = []
    renders: dict[int, TrackRender] = {}
    df = tracks[tracks["track_id"] >= 0]
    for tid, g in df.groupby("track_id"):
        cls = CLASSES[int(g["cls"].mode().iloc[0])]
        r = analyze_track(int(tid), g["frame"].to_numpy(), g[["x_m", "y_m"]].to_numpy(float), fps,
                          None if line_a is None else np.asarray(line_a, float),
                          None if line_b is None else np.asarray(line_b, float), s, cls,
                          float(g["conf"].mean()), paths=paths)
        if r is None:
            continue
        renders[int(tid)] = r
        if r.result:
            results.append(r.result)
    results.sort(key=lambda r: r["t_entry_s"])
    return results, renders
