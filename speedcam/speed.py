"""Per-track speed estimation from world trajectories.

Primary: arc length of the smoothed trajectory between line A and line B, divided by
the time between the crossings (handles lane changes and curves).
Audit: sub-frame line-crossing times, speed = chord between crossings / Δt.
Momentary: |velocity| from a local quadratic fit over a short window (default 0.5 s),
giving a speed profile between the lines with its max, min and noise level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import SpeedSettings
from .detection import CLASSES
from .path import PathGeom

# The max speed is what matters most. It comes from the frames tracked cleanly, so a gap or a
# position jump elsewhere only makes the average over the stretch inconclusive.
INVALIDATING = {"accel", "few_points", "no_clean"}  # no usable speed at all
AVG_INCONCLUSIVE = {"residual", "gap"}  # the average is unreliable; the max still stands


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


def section_fit(ts: np.ndarray, us: np.ndarray, s: SpeedSettings) -> dict:
    """Average speed over a measured stretch (distance travelled us, m, at times ts) and its checks.

    The average is distance / time between the ends: that is what average speed means. A
    regression slope only equals it at constant acceleration; stop-and-go traffic on 0488 read up
    to 24 % low with it. The residual is the deviation from a local quadratic over
    `residual_window_s`, which follows real speed changes but not steps (ID switches, box jumps);
    its rms is the position noise behind the average's standard error."""
    dt = float(ts[-1] - ts[0])
    v = float(us[-1] - us[0]) / dt if dt > 0 else 0.0
    out = {"v": v, "accel": 0.0, "resid_max": 0.0, "rms": 0.0, "se": None}
    if len(ts) < 4 or dt <= 0:
        return out
    out["accel"] = float(2 * np.polyfit(ts, us, 2)[0])  # mean acceleration over the stretch
    # the filter needs evenly spaced samples (line mode adds the exact crossing times at the ends)
    step = float(np.median(np.diff(ts)))
    tg = np.linspace(ts[0], ts[-1], max(4, int(round(dt / step)) + 1))
    ug = np.interp(tg, ts, us)
    win = odd_window(int(round(s.residual_window_s / step)), 2, len(ug))
    e = ug - (savgol_filter(ug, win, 2, mode="interp") if win > 2 else ug)
    out["resid_max"] = float(np.abs(e).max())
    out["rms"] = float(np.sqrt(np.mean(e ** 2)))
    out["se"] = float(np.sqrt(2) * out["rms"] / dt)  # two end positions, each with that noise
    return out


def clean_frames(full: np.ndarray, frames: np.ndarray, dist: np.ndarray, fps: float, s: SpeedSettings) -> np.ndarray:
    """Per frame of the track: whether its momentary speed rests on clean tracking, i.e. no
    detection gap longer than max_gap_s and no position jump (deviation from a local constant-
    acceleration fit over residual_window_s above max_residual_m) within its speed window."""
    n = len(full)
    bad = np.zeros(n, bool)
    det = np.isin(full, frames)
    edges = np.diff(np.r_[0, (~det).astype(int), 0])
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        if (b - a) / fps > s.max_gap_s:
            bad[a:b] = True
    win = odd_window(int(round(s.residual_window_s * fps)), 2, n)
    if win > 2:
        bad |= np.abs(dist - savgol_filter(dist, win, 2, mode="interp")) > s.max_residual_m
    h = int(round(s.momentary_window_s / 2 * fps))
    if h > 0 and bad.any():
        bad = np.convolve(bad.astype(float), np.ones(2 * h + 1), mode="same") > 0
    return ~bad


def _verdict(flags: list[str], s: SpeedSettings) -> tuple[bool, bool]:
    """(max speed valid, average valid)."""
    valid = not (set(flags) & INVALIDATING)
    avg = AVG_INCONCLUSIVE | ({"disagree"} if s.disagreement_invalidates else set())
    return valid, valid and not (set(flags) & avg)


def trim_truncated(frames: np.ndarray, xy: np.ndarray, corners_w: np.ndarray, fps: float, ratio: float) -> slice:
    """Detections to keep: drops the first/last ones whose box is shorter along the direction of
    travel than `ratio` x the box 0.25-0.5 s later/earlier (a vehicle entering or leaving the
    frame, or coming out from behind something). Comparing with boxes that close in time leaves
    the slow growth of an approaching vehicle in an oblique view (~10 %/s on 0447) alone."""
    n = len(frames)
    a, b = max(1, int(round(0.25 * fps))), max(2, int(round(0.5 * fps)))
    if n < 2 * b + 3 or ratio <= 0:
        return slice(0, n)
    k = max(1, int(round(0.25 * fps)))
    d = xy[np.minimum(np.arange(n) + k, n - 1)] - xy[np.maximum(np.arange(n) - k, 0)]
    norm = np.linalg.norm(d, axis=1)
    d = np.where(norm[:, None] > 0.2, d, xy[-1] - xy[0])  # stopped: use the overall heading
    d = d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    ext = np.ptp(np.einsum("nkj,nj->nk", corners_w, d), axis=1)
    i0 = 0
    while i0 < n - b and ext[i0] < ratio * np.median(ext[i0 + a:i0 + b + 1]):
        i0 += 1
    i1 = n - 1
    while i1 > i0 + b and ext[i1] < ratio * np.median(ext[i1 - b:i1 - a + 1]):
        i1 -= 1
    return slice(i0, i1 + 1)


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
    if line_a is None or line_b is None:  # no speed geometry: violation checks only
        return render
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
    if len(ts) < 3:
        flags.append("few_points")
    fit = section_fit(ts, ss, s)
    v_arc, accel, quad_max, rms, avg_se = fit["v"], fit["accel"], fit["resid_max"], fit["rms"], fit["se"]
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
    dist_all = np.interp(t, td, sd)
    clean = clean_frames(full, frames, dist_all, fps, s)

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
    }

    inst_ref = momentary_speed(P_raw, fps, max(2.0, 4 * s.momentary_window_s))
    dist = dist_all - np.interp(t_in, td, sd)
    result.update(_profile(t, inst, inst_ref, dist, t_in, t_out, v_arc * 3.6, direction, s.momentary_window_s / 2,
                           clean, s.min_clean_s))
    if result.pop("_no_clean"):
        flags.append("no_clean")
    result["valid"], result["avg_valid"] = _verdict(flags, s)
    result["flags"] = ";".join(flags)
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

    fit = section_fit(ts, us, s)
    v_path, accel, quad_max, rms, avg_se = fit["v"], fit["accel"], fit["resid_max"], fit["rms"], fit["se"]
    # audit: the vehicle's own trajectory length over the same time (lane changes make it longer)
    k = max(1, int(round(s.arc_dt_s * fps)))
    idx = np.unique(np.r_[np.arange(i0, i1 + 1, k), i1])
    arc = float(np.linalg.norm(np.diff(P[idx], axis=0), axis=1).sum())
    v_arc = arc / dt
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
        "s_entry_m": float(s_sm[i0]), "s_exit_m": float(s_sm[i1]),
        "coverage_pct": float(100 * cover / max(path.hi - path.lo, 1e-9)),
        "path": path.name,
    }
    inst = _deriv_kmh(u_raw, fps, s.momentary_window_s)
    inst_ref = _deriv_kmh(u_raw, fps, max(2.0, 4 * s.momentary_window_s))
    clean = clean_frames(full, frames, u_sm, fps, s)
    result.update(_profile(t, inst, inst_ref, u_sm - u_sm[i0], t_in, t_out, v_path * 3.6, direction,
                           s.momentary_window_s / 2, clean, s.min_clean_s))
    if result.pop("_no_clean"):
        flags.append("no_clean")
    result["valid"], result["avg_valid"] = _verdict(flags, s)
    result["flags"] = ";".join(flags)
    render.inst_kmh = inst
    render.t_first, render.t_second, render.result = t_in, t_out, result
    return render


def _profile(t, inst, inst_ref, dist, t_in, t_out, avg_kmh: float, direction: str, half_window_s: float,
             clean: Optional[np.ndarray] = None, min_clean_s: float = 0.0) -> dict:
    """Momentary speed over the measured stretch: max/min, where the max happened, noise and a
    compact profile. `inst_ref` is the same curve over a much longer window, `dist` the distance
    travelled since the start of the stretch, all per frame of the track (`t`). Max and min only
    come from frames whose speed window (± `half_window_s`) lies inside the track: at its very ends
    the speed is extrapolated (on 0474 the first frame after a vehicle entered read 47 km/h, the
    rest of it 31-44), and only from `clean` frames (see clean_frames); fewer than `min_clean_s`
    of them sets "_no_clean"."""
    seg = (t >= t_in) & (t <= t_out)
    if seg.sum() < 3:
        return {"speed_max_kmh": avg_kmh, "speed_min_kmh": avg_kmh, "speed_sigma_kmh": None, "t_max_s": None,
                "d_max_m": None, "speed_trend_pct": None, "profile": None, "max_clean_pct": None,
                "_no_clean": False}
    v = inst[seg]
    ts = t[seg]
    d = dist[seg]
    # noise: deviation from a much smoother version of the same curve (real speed changes are slow)
    ref = inst_ref[seg]
    sigma = float(np.std(v - ref))
    inner = (ts >= t[0] + half_window_s) & (ts <= t[-1] - half_window_s)
    if inner.sum() < 3:
        inner = np.ones(len(v), bool)
    ok = clean[seg] if clean is not None else np.ones(len(v), bool)
    fps = (len(t) - 1) / max(t[-1] - t[0], 1e-9)
    no_clean = (inner & ok).sum() < max(3, min_clean_s * fps)
    if not no_clean:
        inner &= ok
    vi = np.where(inner, v, np.nan)
    k = int(np.nanargmax(vi))
    # speed change from the first to the second half of the stretch, as a share of the average,
    # signed towards line B: shared by all vehicles it points at calibration, not driving
    half = d <= d[-1] / 2
    trend = (v[~half].mean() - v[half].mean()) / avg_kmh * 100 if half.any() and (~half).any() else None
    if trend is not None and direction == "B->A":
        trend = -trend
    step = max(1, int(np.ceil(len(v) / 120)))
    return {
        "speed_max_kmh": float(vi[k]),
        "speed_min_kmh": float(np.nanmin(vi)),
        "speed_sigma_kmh": sigma,
        "t_max_s": float(ts[k]),
        "d_max_m": float(d[k]),
        "speed_trend_pct": None if trend is None else float(trend),
        "max_clean_pct": float(100 * ok.mean()),
        "_no_clean": bool(no_clean),
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
                   s: SpeedSettings, paths: Optional[list[PathGeom]] = None,
                   H: Optional[np.ndarray] = None) -> tuple[list[dict], dict[int, TrackRender]]:
    """`tracks` needs columns frame, track_id, x_m, y_m, cls, conf. With `paths`, each vehicle is
    measured along the path it covers most and lines A/B are ignored. With `H` (reference px ->
    ground m) and box corners (c0x..c3y), truncated boxes at the ends of tracks are dropped."""
    results: list[dict] = []
    renders: dict[int, TrackRender] = {}
    df = tracks[tracks["track_id"] >= 0]
    cols = [f"c{k}{a}" for k in range(4) for a in "xy"]
    trim = H is not None and all(c in df for c in cols)
    for tid, g in df.groupby("track_id"):
        cls = CLASSES[int(g["cls"].mode().iloc[0])]
        if trim:
            g = g.sort_values("frame")
            cw = cv2.perspectiveTransform(g[cols].to_numpy(float).reshape(-1, 1, 2), H).reshape(-1, 4, 2)
            g = g.iloc[trim_truncated(g["frame"].to_numpy(), g[["x_m", "y_m"]].to_numpy(float), cw, fps,
                                      s.trim_truncated_ratio)]
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
