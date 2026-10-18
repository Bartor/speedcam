"""Per-vehicle margin of error: an uncertainty budget at k = 2 (about 95 %).

Terms, each a standard uncertainty (1 sigma) in km/h, combined root-sum-square:

* calibration: the typed calibration distances (± `distance_tolerance_m`) and the clicked points
  (± `click_tolerance_px`) are perturbed `mc_samples` times, the image -> ground mapping is refitted
  each time, and the spread of each vehicle's path length gives the relative speed uncertainty
  (Monte Carlo propagation through scale / similarity / homography / polygon least squares).
  When the check distances disagree more than the declared tolerances explain (Birge ratio > 1),
  the term is inflated by that ratio.
* noise: momentary-speed noise sigma for the max; the arc-length regression's standard error for
  the A->B average.
* parallax (one-sided, not corrected): a tracked point h metres above the road, seen from a drone
  H metres up, moves H/(H-h) times faster on the ground plane than the vehicle. With the drone
  height known the bias bound is declared as an extra margin below the measured speed;
  unknown, the budget says parallax isn't covered.

Timing (capture fps) and stabilization residuals are far below these and are not budgeted.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from . import calibration
from .config import Calibration, UncertaintySettings

K = 2.0  # coverage factor: ~95 % for a normal distribution


def rss(terms: Iterable[Optional[float]], k: float = K) -> float:
    """Expanded uncertainty: k x root-sum-square of independent standard uncertainties."""
    return k * math.sqrt(sum(float(t) ** 2 for t in terms if t))


def perturb(cal: Calibration, rng: np.random.Generator, tol_m: float, tol_px: float) -> Calibration:
    """A copy of the calibration with every typed distance and clicked point randomly disturbed."""
    c = cal.model_copy(deep=True)

    def px(p):
        return (p[0] + rng.normal(0, tol_px), p[1] + rng.normal(0, tol_px)) if tol_px > 0 else p

    def metres(m):
        return m + rng.normal(0, tol_m) if tol_m > 0 else m

    for s in c.segments:
        s.a, s.b, s.meters = px(s.a), px(s.b), metres(s.meters)
    for p in c.points:
        # tolerance is the position error of the point, spread over both coordinates
        w = tol_m / math.sqrt(2)
        p.px = px(p.px)
        p.world = (p.world[0] + rng.normal(0, w), p.world[1] + rng.normal(0, w)) if w > 0 else p.world
    if c.polygon:
        c.polygon.points = [px(p) for p in c.polygon.points]
        for d in c.polygon.distances:
            d.meters = metres(d.meters)
    return c


def sample_homographies(cal: Calibration, n: int, tol_m: float, tol_px: float, seed: int = 0) -> np.ndarray:
    """(k, 3, 3) mappings refitted from `n` perturbed copies of the calibration (failed fits dropped)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        try:
            H, _, _ = calibration.fit(perturb(cal, rng, tol_m, tol_px))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if H is not None and np.all(np.isfinite(H)):
            out.append(H / H[2, 2])
    return np.array(out).reshape(-1, 3, 3)


def map_many(Hs: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Apply k mappings to m pixels at once: (k, m, 2)."""
    ph = np.column_stack([px, np.ones(len(px))])
    w = np.einsum("kij,mj->kmi", Hs, ph)
    return w[..., :2] / w[..., 2:3]


def length_rel_sigma(H0: np.ndarray, Hs: np.ndarray, px: np.ndarray) -> Optional[float]:
    """Relative spread (1 sigma) of a pixel polyline's ground length across the sampled mappings."""
    if len(Hs) < 2 or len(px) < 2:
        return None
    L0 = float(np.linalg.norm(np.diff(map_many(H0[None], px)[0], axis=0), axis=1).sum())
    if L0 <= 0:
        return None
    L = np.linalg.norm(np.diff(map_many(Hs, px), axis=1), axis=2).sum(axis=1)
    return float(np.std(L, ddof=1) / L0)


def fit_ratio(cal: Calibration, res: "calibration.CalibrationResult", tol_m: float,
              tol_px: float) -> tuple[Optional[float], int]:
    """Birge ratio sqrt(chi2 / dof) of the calibration residuals against the declared tolerances.

    Around 1: the check distances agree as well as the tolerances say. Above 1: the measurements
    or the mapping model (non-flat ground, wrong model) are worse than declared. None: no
    redundancy, so nothing checks the calibration.
    """
    px_m = tol_px * res.m_per_px
    var_d = tol_m ** 2 + 2 * px_m ** 2  # a distance between two clicked points
    var_xy = tol_m ** 2 / 2 + px_m ** 2  # one coordinate of a control point
    chi2, eqs = 0.0, 0
    for r in res.residuals:
        if r["kind"] in ("polygon", "segment"):
            chi2 += r["error_m"] ** 2 / max(var_d, 1e-12)
            eqs += 1
        elif r["kind"] == "point" and res.method in ("similarity", "homography"):
            chi2 += r["error_m"] ** 2 / max(var_xy, 1e-12)
            eqs += 2
    params = {"scale": 1, "similarity": 4, "homography": 8}.get(res.method, 0)
    if res.method == "polygon" and cal.polygon:
        params = 2 * len(cal.polygon.points) - 3
    dof = eqs - params
    if dof <= 0:
        return None, 0
    return float(math.sqrt(chi2 / dof)), dof


def _path_px(H0_inv: np.ndarray, t: np.ndarray, world: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Pixel polyline of a smoothed world track between t0 and t1 (endpoints interpolated)."""
    inside = (t > t0) & (t < t1)
    ts = np.concatenate([[t0], t[inside], [t1]])
    P = np.column_stack([np.interp(ts, t, world[:, 0]), np.interp(ts, t, world[:, 1])])
    return calibration.apply_h(H0_inv, P)


def parallax_bias(drone_height_m: Optional[float], h: float) -> Optional[float]:
    """Relative overestimate H/(H-h) - 1 of an uncorrected tracked point h above the road."""
    if not drone_height_m or drone_height_m <= h:
        return None
    return drone_height_m / (drone_height_m - h) - 1


def annotate(results: list[dict], renders: dict, cal_cfg: Calibration, cal_res, fps: float,
             us: UncertaintySettings, momentary_window_s: float = 0.5, seed: int = 0) -> dict:
    """Add the margin of error to every result in place; return the project-level summary."""
    Hs = sample_homographies(cal_cfg, us.mc_samples, us.distance_tolerance_m, us.click_tolerance_px, seed)
    H0 = cal_res.H / cal_res.H[2, 2]
    H0_inv = np.linalg.inv(H0)
    ratio, dof = fit_ratio(cal_cfg, cal_res, us.distance_tolerance_m, us.click_tolerance_px)
    infl = max(1.0, ratio) if ratio is not None else 1.0

    gap_rel = None
    if cal_cfg.line_a and cal_cfg.line_b and len(Hs) >= 2:
        la, lb = np.array(cal_cfg.line_a, float), np.array(cal_cfg.line_b, float)
        gaps = [calibration._gap_between(calibration.apply_h(H, la), calibration.apply_h(H, lb)) for H in Hs]
        g0 = calibration._gap_between(calibration.apply_h(H0, la), calibration.apply_h(H0, lb))
        gap_rel = float(np.std(gaps, ddof=1) / g0) if g0 > 0 else None

    for r in results:
        rend = renders.get(r["track_id"])
        v_avg, v_max = r["speed_kmh"], r.get("speed_max_kmh", r["speed_kmh"])
        rel_avg = rel_max = None
        if rend is not None:
            t = rend.frames / fps
            rel_avg = length_rel_sigma(H0, Hs, _path_px(H0_inv, t, rend.world, r["t_entry_s"], r["t_exit_s"]))
            if r.get("t_max_s") is not None:
                half = max(momentary_window_s / 2, 2 / fps)
                rel_max = length_rel_sigma(H0, Hs, _path_px(H0_inv, t, rend.world, r["t_max_s"] - half,
                                                            r["t_max_s"] + half))
        rel_avg = (rel_avg or 0.0) * infl
        rel_max = (rel_max if rel_max is not None else rel_avg / infl) * infl

        b = parallax_bias(us.drone_height_m, us.point_height_m.get(r["class"], us.point_height_m.get("car", 0.8)))
        terms = {}
        for key, v, noise in (("max", v_max, r.get("speed_sigma_kmh")), ("avg", v_avg, r.get("speed_avg_se_kmh"))):
            cal_u = (rel_max if key == "max" else rel_avg) * v
            u = rss([cal_u, noise])
            bias = v * b / (1 + b) if b else 0.0
            terms[key] = {"calibration": K * cal_u, "noise": K * (noise or 0.0), "parallax_bias": bias}
            r[f"speed_{key}_u_kmh"] = u
            r[f"speed_{key}_u_lower_kmh"] = u + bias  # measured minus this: lowest plausible true speed
        u_pct = 100 * max(r["speed_max_u_lower_kmh"] / max(v_max, 1e-9), r["speed_avg_u_lower_kmh"] / max(v_avg, 1e-9))
        r["u_pct"] = float(u_pct)
        r["u_cal_pct"] = float(100 * K * rel_avg)
        r["u_parallax_pct"] = None if b is None else float(100 * b)
        r["uncertainty"] = terms

    valid = [r for r in results if r["valid"]]
    return {
        "k": K,
        "settings": us.model_dump(mode="json"),
        "mc_samples": int(len(Hs)),
        "line_gap_rel_pct": None if gap_rel is None else 100 * K * gap_rel,
        "fit_ratio": ratio,
        "fit_dof": dof,
        "fit_inflation": infl,
        "parallax": "bounded" if us.drone_height_m else "unknown",
        "median_u_pct": float(np.median([r["u_pct"] for r in valid])) if valid else None,
    }
