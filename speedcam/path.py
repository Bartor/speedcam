"""Path of motion: a smooth curve on the ground along which traffic is measured.

The user clicks a few points along the road (reference-frame px). They are mapped to the ground
plane and joined by a centripetal Catmull-Rom spline, then resampled evenly. Vehicle positions
are projected onto the curve: `s` is the distance along it (m, from the first clicked point) and
`lateral` the signed offset across it. Speed along the path is ds/dt, so lane changes and bends
don't inflate it, and a vehicle is measured over whatever part of the path it covers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def catmull_rom(points: np.ndarray, samples_per_segment: int = 40, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom spline through `points` (n, 2); passes through every point, no loops."""
    P = np.asarray(points, float)
    if len(P) < 3:
        return np.linspace(P[0], P[-1], samples_per_segment + 1) if len(P) == 2 else P.copy()
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])  # phantom end points continue the ends
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        t = np.linspace(t1, t2, samples_per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-1:])
    return np.vstack(out)


@dataclass
class PathGeom:
    """Dense ground-plane polyline (m) with cumulative arc length."""

    pts: np.ndarray  # (m, 2)
    s: np.ndarray  # (m,) distance from the start
    corridor_m: float  # half-width: vehicles further than this from the curve are ignored
    trusted: tuple[float, float] = (0.0, float("inf"))  # measured span of s (inside the calibrated area)
    name: str = "Path"

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def lo(self) -> float:
        return max(0.0, self.trusted[0])

    @property
    def hi(self) -> float:
        return min(self.length, self.trusted[1])

    def trust_within(self, hull_world: np.ndarray, margin_m: float) -> None:
        """Limit measuring to the longest stretch of the path within `margin_m` of a polygon."""
        import cv2

        hull = np.asarray(hull_world, np.float32).reshape(-1, 1, 2)
        ok = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -margin_m for x, y in self.pts])
        if not ok.any():
            self.trusted = (0.0, 0.0)
            return
        edges = np.diff(np.r_[0, ok.astype(int), 0])
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
        k = int(np.argmax(self.s[ends] - self.s[starts]))
        self.trusted = (float(self.s[starts[k]]), float(self.s[ends[k]]))

    @classmethod
    def from_world_points(cls, world_pts: np.ndarray, corridor_m: float, step_m: float = 0.25,
                          name: str = "Path") -> "PathGeom":
        dense = catmull_rom(world_pts)
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        even = np.arange(0.0, s[-1], step_m)
        even = np.append(even, s[-1])
        pts = np.column_stack([np.interp(even, s, dense[:, 0]), np.interp(even, s, dense[:, 1])])
        return cls(pts=pts, s=even, corridor_m=float(corridor_m), name=name)

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest point on the curve for each position: (distance along, signed lateral offset).

        Positions beyond either end project onto the end, so `s` is 0 or `length` there."""
        P = np.asarray(P, float).reshape(-1, 2)
        a, b = self.pts[:-1], self.pts[1:]
        d = b - a
        L2 = np.maximum((d ** 2).sum(1), 1e-12)
        out_s = np.empty(len(P))
        out_lat = np.empty(len(P))
        for k0 in range(0, len(P), 512):  # bounded memory: positions x segments
            Q = P[k0:k0 + 512]
            u = np.clip(((Q[:, None, :] - a[None]) * d[None]).sum(-1) / L2[None], 0, 1)
            foot = a[None] + u[..., None] * d[None]
            dist2 = ((Q[:, None, :] - foot) ** 2).sum(-1)
            j = np.argmin(dist2, axis=1)
            idx = np.arange(len(Q))
            out_s[k0:k0 + 512] = self.s[j] + u[idx, j] * np.sqrt(L2[j])
            cross = d[j, 0] * (Q[:, 1] - a[j, 1]) - d[j, 1] * (Q[:, 0] - a[j, 0])
            out_lat[k0:k0 + 512] = np.sign(cross) * np.sqrt(dist2[idx, j])
        return out_s, out_lat

    def curvature(self, s: np.ndarray, smooth_m: float = 5.0) -> np.ndarray:
        """Signed curvature (1/m, positive = bending left) at distances `s`, smoothed over `smooth_m`."""
        if len(self.pts) < 3:
            return np.zeros_like(np.asarray(s, float))
        tang = np.gradient(self.pts, axis=0)
        theta = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
        k = np.gradient(theta) / np.maximum(np.gradient(self.s), 1e-9)
        n = max(1, int(round(smooth_m / max(self.s[1] - self.s[0], 1e-9))))
        if n > 1 and len(k) > n:
            k = np.convolve(np.pad(k, n // 2, mode="edge"), np.ones(n) / n, mode="valid")[:len(self.s)]
        return np.interp(s, self.s, k)

    def travelled(self, s: np.ndarray, lateral: np.ndarray) -> np.ndarray:
        """Distance actually covered by a vehicle moving along the curve at its own lateral offset.

        On a bend, a lane `lateral` metres off the curve is (1 - curvature * lateral) times as long,
        so the distance along the curve alone would make cars in outer lanes look slow (and inner
        lanes fast) by about offset / radius. Lateral moves (lane changes) still add nothing."""
        s = np.asarray(s, float)
        lat = np.asarray(lateral, float)
        if len(s) < 2:
            return np.zeros_like(s)
        mid_s, mid_lat = (s[1:] + s[:-1]) / 2, (lat[1:] + lat[:-1]) / 2
        du = np.diff(s) * (1 - self.curvature(mid_s) * mid_lat)
        return np.concatenate([[0.0], np.cumsum(du)])

    def offset(self, dist_m: float) -> np.ndarray:
        """The curve shifted sideways by `dist_m` (positive = left of the travel direction)."""
        t = np.gradient(self.pts, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        n = np.column_stack([-t[:, 1], t[:, 0]])
        return self.pts + dist_m * n
