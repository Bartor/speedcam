"""Reference-pixel → world (metres) mapping.

* >= 4 control points with world coordinates: full homography.
* 2-3 control points: similarity (rotation + uniform scale + translation).
* Distance segments only: uniform scale (valid for a nadir camera).

Segments are always reported as check residuals, whatever model is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from scipy.optimize import least_squares

from .config import Calibration, Polygon
from .path import PathGeom


@dataclass
class CalibrationResult:
    H: np.ndarray  # 3x3, reference px -> world m
    method: str
    m_per_px: float
    residuals: list[dict] = field(default_factory=list)
    rms_m: float = 0.0
    max_rel_error: float = 0.0
    line_gap_m: Optional[float] = None
    line_a_world: Optional[np.ndarray] = None
    line_b_world: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)
    fit_ok: bool = True
    path_geoms: list[PathGeom] = field(default_factory=list)
    # per path, for drawing (reference px): smooth curve, measured part, corridor edges
    paths_px: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "m_per_px": self.m_per_px,
            "cm_per_px": self.m_per_px * 100,
            "residuals": self.residuals,
            "rms_m": self.rms_m,
            "max_rel_error": self.max_rel_error,
            "line_gap_m": self.line_gap_m,
            "H": self.H.tolist(),
            "warnings": self.warnings,
            "fit_ok": self.fit_ok,
            "paths": [{"name": g.name, "length_m": g.length, "measured_m": g.hi - g.lo,
                       "px": _thin(d["px"]), "trusted_px": _thin(d["trusted_px"]),
                       "corridor_px": [_thin(e) for e in d["corridor_px"]]}
                      for g, d in zip(self.path_geoms, self.paths_px)],
        }


def _thin(px: Optional[np.ndarray], n: int = 160) -> Optional[list]:
    """At most ~n points of a pixel polyline, for drawing in the UI."""
    if px is None:
        return None
    step = max(1, int(np.ceil(len(px) / n)))
    keep = np.unique(np.r_[np.arange(0, len(px), step), len(px) - 1])
    return np.round(px[keep], 1).tolist()


# Above this, the mapping model clearly does not fit the clicked distances; analysis is refused.
MAX_REL_ERROR = 0.05


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def local_scale(H: np.ndarray, at: tuple[float, float]) -> float:
    """sqrt(|det J|) of the mapping at a pixel — metres per pixel."""
    p = np.array([at, (at[0] + 1, at[1]), (at[0], at[1] + 1)])
    w = apply_h(H, p)
    J = np.column_stack([w[1] - w[0], w[2] - w[0]])
    return float(np.sqrt(abs(np.linalg.det(J))))


def _similarity(px: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama), allowing a reflection (image y-down vs world y-up)."""
    best = None
    for flip in (False, True):
        src = px.copy()
        if flip:
            src[:, 1] *= -1
        mu_s, mu_w = src.mean(0), world.mean(0)
        S, W = src - mu_s, world - mu_w
        U, D, Vt = np.linalg.svd(W.T @ S)
        d = np.sign(np.linalg.det(U @ Vt)) or 1.0
        R = U @ np.diag([1, d]) @ Vt
        scale = (D * np.array([1, d])).sum() / (S ** 2).sum()
        t = mu_w - scale * R @ mu_s
        M = np.eye(3)
        M[:2, :2] = scale * R
        M[:2, 2] = t
        if flip:
            M = M @ np.diag([1.0, -1.0, 1.0])
        err = np.linalg.norm(apply_h(M, px) - world, axis=1).sum()
        if best is None or err < best[0]:
            best = (err, M)
    return best[1]


def _gap_between(a: np.ndarray, b: np.ndarray) -> float:
    def dist_point_line(p, l):
        d = l[1] - l[0]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        return abs(np.dot(p - l[0], n))

    return 0.5 * (dist_point_line(a.mean(0), b) + dist_point_line(b.mean(0), a))


def solve_polygon(poly: Polygon) -> tuple[np.ndarray, np.ndarray, list]:
    """Recover the flat shape of a ground polygon from its measured sides/diagonals.

    Returns (pixel points, world points in metres, used distances). The world frame is arbitrary
    (rotation/translation/reflection), which does not matter for distances and speeds.
    """
    px = np.array(poly.points, float)
    n = len(px)
    if n < 4:
        raise ValueError("A polygon needs at least 4 points")
    d = [x for x in poly.distances if x.meters > 0 and x.i != x.j and 0 <= x.i < n and 0 <= x.j < n]
    need = 2 * n - 3
    if len(d) < need:
        raise ValueError(f"A {n}-point polygon needs at least {need} measured distances to fix its shape "
                         f"(it has {len(d)}). Measure the sides and add diagonals.")
    I = np.array([x.i for x in d])
    J = np.array([x.j for x in d])
    m = np.array([x.meters for x in d])
    # start from the image shape at a rough scale; fix the gauge (point 0 and the 0->1 direction)
    s0 = float(np.median(m / np.maximum(np.linalg.norm(px[I] - px[J], axis=1), 1e-9)))
    X0 = (px - px[0]) * s0
    u = X0[1] / max(np.linalg.norm(X0[1]), 1e-9)

    def f(x):
        X = x.reshape(n, 2)
        r = np.linalg.norm(X[I] - X[J], axis=1) - m
        gauge = [X[0, 0], X[0, 1], u[0] * X[1, 1] - u[1] * X[1, 0]]
        return np.concatenate([r, gauge])

    sol = least_squares(f, X0.ravel(), x_scale="jac")
    if np.linalg.matrix_rank(sol.jac[: len(d)], tol=1e-6 * max(1.0, np.abs(sol.jac).max())) < need:
        raise ValueError("These distances don't fix the polygon's shape (it could still flex). "
                         "Add a diagonal that crosses the polygon.")
    return px, sol.x.reshape(n, 2), d


def _spread_deg(vectors: np.ndarray) -> float:
    """Largest angle between any two undirected directions, in degrees."""
    ang = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180
    diff = np.abs(ang[:, None] - ang[None, :])
    return float(np.minimum(diff, 180 - diff).max()) if len(ang) > 1 else 0.0


def fit(cal: Calibration) -> tuple[np.ndarray, str, list]:
    """The image -> ground mapping alone: (H, method, polygon distances used).

    Cheap enough to re-run a few hundred times with perturbed inputs (see uncertainty.py)."""
    pts = cal.points
    segs = cal.segments
    poly_d: list = []
    if cal.polygon and cal.polygon.points:
        poly_px, poly_world, poly_d = solve_polygon(cal.polygon)
        H, _ = cv2.findHomography(poly_px, poly_world, 0)
        method = "polygon"
    elif len(pts) >= 4:
        px = np.array([p.px for p in pts], float)
        world = np.array([p.world for p in pts], float)
        H, _ = cv2.findHomography(px, world, 0)
        method = "homography"
    elif len(pts) >= 2:
        px = np.array([p.px for p in pts], float)
        world = np.array([p.world for p in pts], float)
        H = _similarity(px, world)
        method = "similarity"
    elif segs:
        lens = np.array([np.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]) for s in segs])
        meters = np.array([s.meters for s in segs])
        if np.any(lens <= 0):
            raise ValueError("Calibration segment has zero pixel length")
        s = float((meters * lens).sum() / (lens ** 2).sum())
        H = np.diag([s, s, 1.0])
        method = "scale"
    else:
        raise ValueError("Calibration needs at least one distance segment or two control points")
    return H, method, poly_d


def compute(cal: Calibration, image_size: Optional[tuple[int, int]] = None) -> CalibrationResult:
    warnings: list[str] = []
    pts = cal.points
    segs = cal.segments
    H, method, poly_d = fit(cal)

    residuals: list[dict] = []
    for k, pd_ in enumerate(poly_d):
        wa, wb = apply_h(H, [cal.polygon.points[pd_.i], cal.polygon.points[pd_.j]])
        dd = float(np.linalg.norm(wb - wa))
        residuals.append({"kind": "polygon", "index": k, "i": pd_.i, "j": pd_.j, "measured_m": dd,
                          "given_m": pd_.meters, "error_m": dd - pd_.meters, "rel_error": (dd - pd_.meters) / pd_.meters})
    for i, p in enumerate(pts if method != "polygon" else []):
        w = apply_h(H, [p.px])[0]
        err = float(np.linalg.norm(w - np.array(p.world)))
        residuals.append({"kind": "point", "index": i, "error_m": err})
    for i, sg in enumerate(segs):
        wa, wb = apply_h(H, [sg.a, sg.b])
        d = float(np.linalg.norm(wb - wa))
        residuals.append({"kind": "segment", "index": i, "measured_m": d, "given_m": sg.meters,
                          "error_m": d - sg.meters, "rel_error": (d - sg.meters) / sg.meters})

    if method == "scale" and len(segs) == 1:
        warnings.append("Single segment: scale is exact by construction, there is no residual to check it.")
    if method == "scale" and len(segs) >= 1:
        vec = np.array([np.subtract(s.b, s.a) for s in segs], float)
        if _spread_deg(vec) < 30:
            warnings.append(
                "All distances point the same way, so the scale across them is unchecked. On a view that isn't "
                "straight down this inflates sideways motion and overestimates speed. Use Polygon or Rectangle, "
                "or add a distance across the road.")
    if method == "polygon" and len(poly_d) == 2 * len(cal.polygon.points) - 3 and not segs:
        warnings.append("No redundant distance: the polygon fits exactly, so a typo or misclick can't be detected. "
                        "Measure one more diagonal, or add a distance segment as a check.")
    if method == "similarity" and len(pts) == 2 and not segs:
        warnings.append("Two control points: similarity is exact by construction, no residual check.")

    errs = np.array([r["error_m"] for r in residuals]) if residuals else np.zeros(1)
    rels = [abs(r["rel_error"]) for r in residuals if "rel_error" in r]
    center = (image_size[0] / 2, image_size[1] / 2) if image_size else (0.0, 0.0)
    if cal.line_a:
        center = tuple(np.mean(np.array(cal.line_a + (cal.line_b or cal.line_a)), axis=0))
    elif any(p.points for p in cal.paths):
        center = tuple(np.mean(np.array([q for p in cal.paths for q in p.points], float), axis=0))
    res = CalibrationResult(
        H=H, method=method, m_per_px=local_scale(H, center), residuals=residuals,
        rms_m=float(np.sqrt(np.mean(errs ** 2))), max_rel_error=float(max(rels) if rels else 0.0),
        warnings=warnings,
    )
    if res.max_rel_error > MAX_REL_ERROR:
        res.fit_ok = False
        hint = (" A single scale only fits a camera pointing straight down; this view has perspective. "
                "Use the Polygon or Rectangle tool for a perspective correction (homography)."
                if method == "scale" else " Check the clicked points and the typed distances.")
        res.warnings.insert(0, f"Calibration does not fit: a check distance is off by {res.max_rel_error:.0%}.{hint}")
    if cal.line_a:
        res.line_a_world = apply_h(H, cal.line_a)
    if cal.line_b:
        res.line_b_world = apply_h(H, cal.line_b)
    if res.line_a_world is not None and res.line_b_world is not None:
        res.line_gap_m = float(_gap_between(res.line_a_world, res.line_b_world))
    hull = calibrated_area_world(cal, H) if cal.paths else None
    H_inv = np.linalg.inv(H)
    for mp in cal.paths:
        if len(mp.points) < 2:
            continue
        g = PathGeom.from_world_points(apply_h(H, mp.points), mp.corridor_m, name=mp.name)
        if cal.path_clip_to_calibration and hull is not None:
            g.trust_within(hull, cal.path_calibration_margin_m)
            outside = g.length - (g.hi - g.lo)
            if g.hi - g.lo <= 0:
                res.warnings.append(f"{mp.name} doesn't pass through the calibrated area, so nothing would be "
                                    "measured on it. Calibrate where that traffic is, or draw the path there.")
            elif outside > 1:
                res.warnings.append(
                    f"{mp.name}: {outside:.0f} m of its {g.length:.0f} m lies outside the calibrated area (more "
                    f"than {cal.path_calibration_margin_m:g} m from your calibration points), where the "
                    f"calibration would be extrapolated; speeds are measured on the remaining {g.hi - g.lo:.0f} m. "
                    "Extend the calibration to measure further.")
        px = apply_h(H_inv, g.pts)
        inn = (g.s >= g.lo) & (g.s <= g.hi)
        res.path_geoms.append(g)
        res.paths_px.append({"px": px, "trusted_px": px[inn] if inn.sum() >= 2 else None,
                             "corridor_px": (apply_h(H_inv, g.offset(mp.corridor_m)),
                                             apply_h(H_inv, g.offset(-mp.corridor_m)))})
    if len(res.path_geoms) == 1:
        res.line_gap_m = res.path_geoms[0].hi - res.path_geoms[0].lo  # "the measured section"
    elif res.path_geoms:
        res.line_gap_m = None
    return res


def calibrated_area_world(cal: Calibration, H: np.ndarray) -> Optional[np.ndarray]:
    """Convex outline (ground m) of everything the calibration was measured on."""
    import cv2

    px: list = []
    if cal.polygon:
        px += list(cal.polygon.points)
    px += [p.px for p in cal.points]
    for sg in cal.segments:
        px += [sg.a, sg.b]
    if len(px) < 2:
        return None
    w = apply_h(H, np.array(px, float)).astype(np.float32)
    if len(w) == 2:  # a single distance: a thin strip along it
        d = w[1] - w[0]
        n = np.array([-d[1], d[0]], np.float32) / max(float(np.linalg.norm(d)), 1e-6)
        w = np.array([w[0] + n, w[1] + n, w[1] - n, w[0] - n], np.float32)
    return cv2.convexHull(w).reshape(-1, 2)


def has_speed_geometry(cal: Calibration) -> bool:
    """Lines A and B, or a path of motion."""
    return bool((cal.line_a and cal.line_b) or any(len(p.points) >= 2 for p in cal.paths))


def has_violation_geometry(cal: Calibration) -> bool:
    """A stop line of a red-light setup, or a solid line."""
    return any(len(sl.points) >= 2 for r in cal.red_lights for sl in r.stop_lines) or         any(len(s.points) >= 2 for s in cal.solid_lines)


def has_measurement_geometry(cal: Calibration) -> bool:
    """Anything to measure: speeds (lines A/B or a path) or violations."""
    return has_speed_geometry(cal) or has_violation_geometry(cal)
