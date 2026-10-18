"""Vehicle positions from the vehicle's own texture (optical flow) instead of the box centre.

Detector boxes wobble by 0.2-0.4 m frame to frame (occlusion by poles and wires, appearance
changes in oblique views). The texture of the car itself moves rigidly, so tracking corners
inside the box with pyramidal Lucas-Kanade gives a much smoother frame-to-frame displacement.

Per track: features are picked inside the (shrunk) box, away from other vehicles, tracked
frame to frame with a forward-backward check, mapped to reference-frame pixels and fitted
with a RANSAC similarity. The position is propagated by that similarity and pulled towards
the box centre with a complementary filter (time constant `anchor_s`), so texture drift
can't accumulate and the long-run position is the box centre's. Features are re-seeded when
too few survive (occlusion, turning); with too few the track falls back to the box centre.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from .config import FlowSettings
from .stabilization import transform_points



@dataclass
class _Track:
    pts: np.ndarray  # (m, 2) float32 feature positions in the previous frame (frame px)
    x: np.ndarray  # (2,) filtered position, reference px
    n: int = 1  # frames since the position was anchored to a box (filter gain warm-up)
    coast: int = 0  # consecutive frames without a box
    ok: bool = False  # position came from flow in the last update (not a fresh anchor)
    boxes: deque = field(default_factory=deque)  # recent (frame, box centre ref px)

    def box_velocity(self) -> np.ndarray | None:
        """Box-centre velocity (ref px / frame) over the recent history: noisy per frame, but over
        a few tenths of a second good enough to tell the vehicle's texture from a pole or the road."""
        if len(self.boxes) < 3:
            return None
        f = np.array([b[0] for b in self.boxes], float)
        c = np.array([b[1] for b in self.boxes])
        f -= f.mean()
        return (f @ (c - c.mean(0))) / (f @ f)


def _shrink(poly: np.ndarray, k: float) -> np.ndarray:
    c = poly.mean(0)
    return c + (poly - c) * k


class FlowTracker:
    """Call `update` once per frame, in order, with every detection of that frame."""

    def __init__(self, s: FlowSettings, fps: float):
        self.s = s
        self.fps = fps
        self.alpha = min(1.0, 1.0 / max(s.anchor_s * fps, 1.0))
        self.max_coast = max(0, int(round(s.max_coast_s * fps)))
        self.tracks: dict[int, _Track] = {}
        self.prev_gray = None
        self.prev_T = None
        self.win = (s.win, s.win)
        self.hist = max(3, int(round(s.velocity_s * fps)))
        self.i = 0

    def _lk(self, g0: np.ndarray, g1: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward-backward LK on a crop around the points (pyramids of a whole 4K frame cost
        ~30 ms; a vehicle-sized crop well under one)."""
        h, w = g0.shape[:2]
        m = self.s.crop_margin_px
        x0, y0 = np.maximum(np.floor(p0.min(0)).astype(int) - m, 0)
        x1, y1 = np.minimum(np.ceil(p0.max(0)).astype(int) + m + 1, [w, h])
        c0, c1 = g0[y0:y1, x0:x1], g1[y0:y1, x0:x1]
        off = np.float32([x0, y0])
        q0 = (p0 - off).reshape(-1, 1, 2).astype(np.float32)
        kw = dict(winSize=self.win, maxLevel=self.s.levels, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                  self.s.lk_iters, self.s.lk_eps))
        q1, st1, _ = cv2.calcOpticalFlowPyrLK(c0, c1, q0, None, **kw)
        qb, st2, _ = cv2.calcOpticalFlowPyrLK(c1, c0, q1, None, **kw)
        fb = np.linalg.norm((qb - q0).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.s.fb_max_px)
        return q1.reshape(-1, 2) + off, good

    def update(self, gray: np.ndarray, T: np.ndarray, ids: np.ndarray, corners: np.ndarray,
               centers_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`corners` (n, 4, 2) frame px, `centers_ref` (n, 2) box centres in reference px.

        Returns per detection the flow-refined reference position (NaN where the box centre
        must be used) and the number of features that supported it.
        """
        ids = np.asarray(ids, int)
        n = len(ids)
        out = np.full((n, 2), np.nan)
        npts = np.zeros(n, int)
        box_of = {int(t): k for k, t in enumerate(ids) if t >= 0}
        polys = [np.asarray(c, np.float32) for c in corners]

        # 1. propagate every live track by the motion of its features
        for tr in self.tracks.values():
            tr.ok = False
        if self.prev_gray is not None and self.tracks:
            for t, tr in self.tracks.items():
                if not len(tr.pts):
                    continue
                p0 = tr.pts
                p1, g = self._lk(self.prev_gray, gray, p0)
                q0 = transform_points(self.prev_T, p0)
                q1 = transform_points(T, p1)
                v = tr.box_velocity()
                if v is not None:  # features must move with the vehicle (not a wire, pole or road)
                    dev = np.linalg.norm(q1 - q0 - v, axis=1)
                    g &= dev < max(self.s.motion_gate_px, self.s.motion_gate * float(np.hypot(*v)))
                k = box_of.get(t)
                if k is not None and g.any():  # features must still lie on this vehicle
                    poly = _shrink(polys[k], self.s.gate)
                    g &= np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                                   for x, y in p1], bool)
                self._propagate(tr, q0[g], q1[g], p1[g])
        for t in [t for t in self.tracks if t not in box_of]:
            tr = self.tracks[t]
            tr.coast += 1
            if not tr.ok or tr.coast > self.max_coast:
                del self.tracks[t]

        # 2. anchor to the box centre, emit, re-seed
        for t, k in box_of.items():
            c = np.asarray(centers_ref[k], float)
            tr = self.tracks.get(t)
            if tr is None:
                tr = self.tracks[t] = _Track(pts=np.zeros((0, 2), np.float32), x=c.copy())
            size = float(np.linalg.norm(polys[k][2] - polys[k][0]))
            if tr.ok and np.linalg.norm(tr.x - c) > self.s.max_offset * size:
                tr.ok = False  # flow has wandered off the vehicle: start over from the box
                tr.pts = np.zeros((0, 2), np.float32)
            if tr.ok:
                a = max(self.alpha, 1.0 / (tr.n + 1))
                tr.x = tr.x + a * (c - tr.x)
                tr.n += 1
                out[k] = tr.x
                npts[k] = len(tr.pts)
            else:  # new track or flow lost: restart from the box
                tr.x, tr.n = c.copy(), 1
            tr.coast = 0
            tr.boxes.append((self.i, c))
            while tr.boxes and tr.boxes[0][0] <= self.i - self.hist:
                tr.boxes.popleft()
            if len(tr.pts) < self.s.reseed_below * self.s.max_points:
                others = [polys[j] for j in range(n) if j != k]
                tr.pts = self._seed(gray, polys[k], others, tr.pts)
        self.prev_gray, self.prev_T = gray, T
        self.i += 1
        return out, npts

    def _propagate(self, tr: _Track, q0: np.ndarray, q1: np.ndarray, p1: np.ndarray) -> None:
        tr.ok = False
        if len(q0) < self.s.min_points:
            tr.pts = np.zeros((0, 2), np.float32)
            return
        M, inl = cv2.estimateAffinePartial2D(q0, q1, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.s.ransac_px, maxIters=self.s.ransac_iters,
                                             refineIters=10)
        if M is None:
            tr.pts = np.zeros((0, 2), np.float32)
            return
        inl = inl.ravel().astype(bool)
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if inl.sum() < self.s.min_points or abs(scale - 1) > 0.05:
            tr.pts = p1[inl].astype(np.float32)
            return
        tr.x = M[:, :2] @ tr.x + M[:, 2]
        tr.pts = p1[inl].astype(np.float32)
        tr.ok = True

    def _seed(self, gray: np.ndarray, poly: np.ndarray, others: list[np.ndarray],
              existing: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        x0, y0 = np.floor(poly.min(0)).astype(int)
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return existing
        off = np.array([x0, y0], np.float32)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        # stay off the box border: road and neighbours show through there
        cv2.fillPoly(mask, [np.round((_shrink(poly, self.s.inner) - off) * 16).astype(np.int32)], 255,
                     cv2.LINE_8, shift=4)
        # every LK window must lie on the vehicle: a window that straddles the box edge sees the
        # road too and under-estimates the motion
        r = self.s.win // 2 + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        for o in others:
            if (o.max(0) < [x0, y0]).any() or (o.min(0) > [x1, y1]).any():
                continue
            cv2.fillPoly(mask, [np.round((_shrink(o, 1.1) - off) * 16).astype(np.int32)], 0,
                         cv2.LINE_8, shift=4)
        side = float(np.sqrt(max(cv2.contourArea(poly), 1.0)))
        min_dist = max(3.0, side / np.sqrt(self.s.max_points) / 2)
        for p in existing:
            cv2.circle(mask, tuple(int(v) for v in np.round(p - off)), int(min_dist), 0, -1)
        need = self.s.max_points - len(existing)
        if need <= 0 or not mask.any():
            return existing
        new = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], maxCorners=need, qualityLevel=self.s.quality,
                                      minDistance=min_dist, mask=mask, blockSize=5)
        if new is None:
            return existing
        return np.concatenate([existing, new.reshape(-1, 2) + off]).astype(np.float32)


def reference_positions(df: pd.DataFrame) -> np.ndarray:
    """Reference-frame position per row: flow-refined where available, else the box centre."""
    box = df[["rx", "ry"]].to_numpy(float)
    if "frx" not in df:
        return box
    fl = df[["frx", "fry"]].to_numpy(float)
    ok = np.isfinite(fl).all(1)
    return np.where(ok[:, None], fl, box)
