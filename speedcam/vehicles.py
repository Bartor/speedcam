"""The user's corrections to the vehicles of an analysis: merged tracks, and a look at each
vehicle (a picture and its colour, to suggest a name).

A tracker that loses a vehicle (a tree, a sign, a bus in front) starts a new id when it finds
it again, so one vehicle becomes two tracks and neither may cross the whole measured section.
Merging measures them as one track. The gap between them has no positions, so the average over
the stretch is inconclusive (the "gap" flag), while the max still comes from the frames tracked
cleanly on either side of it.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .stabilization import transform_points

CORNER_COLS = [f"c{k}{a}" for k in range(4) for a in "xy"]


def merge_groups(merges: list[list[int]]) -> list[list[int]]:
    """Groups that share a track joined into one (merging 3+4, then 4+7 gives 3+4+7), each in
    the order given, empty and single ones dropped."""
    out: list[list[int]] = []
    for g in merges:
        g = [int(t) for t in dict.fromkeys(g)]
        hit = [o for o in out if set(o) & set(g)]
        for o in hit:
            out.remove(o)
        joined = list(dict.fromkeys([t for o in hit for t in o] + g))
        out.append(joined)
    return [g for g in out if len(g) > 1]


def apply_merges(df: pd.DataFrame, merges: list[list[int]]) -> pd.DataFrame:
    """Tracks of a merge group relabelled to the group's first id. Where the tracks overlap
    (the old id still coasting when the new one starts), the more confident detection is kept."""
    groups = merge_groups(merges)
    if not groups or not len(df):
        return df
    to = {int(t): g[0] for g in groups for t in g[1:]}
    df = df.copy()
    df["track_id"] = df["track_id"].map(lambda t: to.get(int(t), int(t))).astype(np.int64)
    keep = set(g[0] for g in groups)
    merged = df["track_id"].isin(keep)
    if merged.any():
        m = df[merged].sort_values("conf", ascending=False).drop_duplicates(["track_id", "frame"])
        df = pd.concat([df[~merged], m]).sort_values(["frame", "track_id"]).reset_index(drop=True)
    return df


def _ends(g: pd.DataFrame, fps: float) -> Optional[dict]:
    g = g.sort_values("frame")
    f = g["frame"].to_numpy()
    p = g[["x_m", "y_m"]].to_numpy(float)
    if len(f) < 3 or not np.isfinite(p).all():
        return None
    k = max(2, min(len(f) // 2, int(round(0.5 * fps))))  # velocity over ~0.5 s at each end

    def vel(a, b):
        dt = (f[b] - f[a]) / fps
        return (p[b] - p[a]) / dt if dt > 0 else np.zeros(2)

    return {"f0": int(f[0]), "f1": int(f[-1]), "p0": p[0], "p1": p[-1],
            "v0": vel(0, k - 1), "v1": vel(len(f) - k, len(f) - 1)}


def merge_candidates(df: pd.DataFrame, fps: float, max_gap_s: float = 3.0, overlap_s: float = 0.3,
                     relevant: Optional[set[int]] = None, min_track_s: float = 0.25) -> list[dict]:
    """Pairs of tracks that look like one vehicle split in two: the second starts within
    `max_gap_s` of the end of the first (or overlaps it briefly), heading the same way, where the
    first one would have got to at its speed. `miss_m` is how far off that is; `likely` pairs are
    close enough to suggest, the rest only shown when asked about one of them. With `relevant`
    (the tracks that reach the measured section), pairs of two other tracks are left out: merging
    them can't change a measurement, and a busy junction has plenty (queues, parked cars).

    `df` needs frame, track_id, cls, x_m, y_m (world metres)."""
    ends = {}
    for tid, g in df[df["track_id"] >= 0].groupby("track_id"):
        e = _ends(g, fps)
        if e:
            ends[int(tid)] = e
    out = []
    items = sorted(ends.items(), key=lambda kv: kv[1]["f1"])
    starts = sorted(ends.items(), key=lambda kv: kv[1]["f0"])
    for a, ea in items:
        sa = float(np.linalg.norm(ea["v1"]))
        if sa < 1.0:  # parked or waiting: anything nearby would "continue" it
            continue
        for b, eb in starts:
            if b == a:
                continue
            gap = (eb["f0"] - ea["f1"]) / fps
            if gap < -overlap_s:
                continue
            if gap > max_gap_s:
                break
            sb = float(np.linalg.norm(eb["v0"]))
            if sb < 1.0 or eb["f1"] <= ea["f1"]:
                continue
            if relevant is not None and a not in relevant and b not in relevant:
                continue
            cos = float(np.dot(ea["v1"], eb["v0"]) / (sa * sb))
            if cos < 0.7:
                continue
            g = max(gap, 0.0)
            miss = min(float(np.linalg.norm(ea["p1"] + ea["v1"] * g - eb["p0"])),
                       float(np.linalg.norm(eb["p0"] - eb["v0"] * g - ea["p1"])))
            bound = 2.5 + 0.25 * max(sa, sb) * g  # metres: a box's length, plus speed drift
            if miss > 3 * bound:
                continue
            # a sliver (a few frames of a vehicle entering the picture) isn't worth suggesting
            long_enough = min(ea["f1"] - ea["f0"], eb["f1"] - eb["f0"]) / fps >= min_track_s
            # one vehicle keeps its speed across a short gap: at most ~3 m/s² either way, plus noise
            steady = abs(sa - sb) <= 1.5 + 3.0 * g
            out.append({"a": a, "b": b, "gap_s": round(gap, 3), "miss_m": round(miss, 2),
                        "speed_kmh": round(3.6 * (sa + sb) / 2, 1), "speed_change_kmh": round(3.6 * (sb - sa), 1),
                        "likely": miss <= bound and long_enough and steady,
                        "t_a_end_s": round(ea["f1"] / fps, 3), "t_b_start_s": round(eb["f0"] / fps, 3)})
    # a track continues at most one other: keep each one's best pairing among the likely ones
    out.sort(key=lambda c: c["miss_m"])
    used_a, used_b = set(), set()
    for c in out:
        if c["likely"]:
            if c["a"] in used_a or c["b"] in used_b:
                c["likely"] = False
            else:
                used_a.add(c["a"])
                used_b.add(c["b"])
    return sorted(out, key=lambda c: c["t_a_end_s"])


# -- a look at a vehicle ------------------------------------------------------------------------
def best_view(g: pd.DataFrame) -> pd.Series:
    """The detection to picture a vehicle by: the biggest confident box in the middle of its track
    (at the ends a vehicle is often only half in the frame)."""
    g = g.sort_values("frame")
    n = len(g)
    mid = g.iloc[n // 5: max(n // 5 + 1, n - n // 5)]
    c = mid[CORNER_COLS].to_numpy(float).reshape(-1, 4, 2)
    area = np.ptp(c[:, :, 0], axis=1) * np.ptp(c[:, :, 1], axis=1)
    score = area * mid["conf"].to_numpy(float)
    return mid.iloc[int(np.argmax(score))]


def frame_quad(row: pd.Series, T: np.ndarray, scale: float) -> np.ndarray:
    """The detection's box on its own (unstabilized) frame, at `scale` of the source size.
    T maps frame px -> reference px."""
    ref = row[CORNER_COLS].to_numpy(float).reshape(4, 2)
    return transform_points(np.linalg.inv(T), ref) * scale


def crop(img: np.ndarray, quad: np.ndarray, pad: float = 0.25) -> np.ndarray:
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    h, w = img.shape[:2]
    a, b = int(max(0, x0 - px)), int(min(w, x1 + px))
    c, d = int(max(0, y0 - py)), int(min(h, y1 + py))
    return img[c:d, a:b] if b > a and d > c else img[:1, :1]


def body_colour(img: np.ndarray, quad: np.ndarray) -> tuple[int, int, int]:
    """The main colour (RGB) inside the middle of the box: the biggest of 3 colour clusters, so
    windows, shadow and road at the edges don't tint it."""
    c = quad.mean(0)
    inner = c + (quad - c) * 0.6
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [inner.astype(np.int32)], 255)
    px = img[mask > 0]
    if len(px) < 12:
        px = crop(img, quad, 0.0).reshape(-1, 3)
    if len(px) == 0:
        return (128, 128, 128)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = min(3, len(lab))
    _, labels, centres = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    big = np.bincount(labels.ravel(), minlength=k).argmax()
    bgr = cv2.cvtColor(centres[big].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def colour_name(rgb: tuple[int, int, int]) -> str:
    """A plain colour word for a car's paint, as seen from above in daylight."""
    hsv = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]) * 2, int(hsv[1]), int(hsv[2])
    if v < 55:
        return "black"
    if s < 45 or (s < 70 and v < 110):
        return "white" if v > 190 else "silver" if v > 135 else "grey" if v > 85 else "dark grey"
    if h < 15 or h >= 340:
        return "red" if v > 90 else "dark red"
    if h < 40:
        return "brown" if v < 140 else "beige" if s < 110 else "orange"
    if h < 70:
        return "beige" if s < 100 else "yellow"
    if h < 165:
        return "green"
    if h < 260:
        return "blue" if v > 70 else "dark blue"
    if h < 290:
        return "purple"
    return "pink" if v > 150 else "red"
