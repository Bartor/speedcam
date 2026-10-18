"""Annotated video export, drawn from stored tracks (no new analysis).

Presentation layer for sharing: anti-aliased text in a real font, speed pills on vehicles,
and an optional side panel that lists every vehicle's max speed as it crosses the second line,
with running totals and the latest vehicle's speed profile.
"""
from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import calibration
from .config import ProjectConfig, RenderSettings
from .pipeline import Reporter, margin_of_error, resolve_capture_fps
from .speed import TrackRender, compute_speeds
from .stabilization import transform_points
from .timing import probe
from .video import FFmpegReader, FFmpegWriter

# colours are RGB (PIL); frames are BGR (OpenCV)
WHITE = (255, 255, 255)
MUTED = (160, 166, 176)
INK = (14, 16, 20)
LINE_A, LINE_B = (0, 200, 255), (255, 60, 200)
GOOD, WARN, BAD, INVALID = (46, 204, 113), (255, 176, 32), (235, 64, 52), (150, 150, 160)


def speed_rgb(kmh: float, limit: float) -> tuple[int, int, int]:
    if kmh <= limit:
        return GOOD
    if kmh <= limit + 10:
        return WARN
    return BAD


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


# -- typography -----------------------------------------------------------------------------
def _font_candidates(kind: str) -> list[Path]:
    win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    try:
        import matplotlib

        mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    except ImportError:  # pragma: no cover
        mpl = Path(".")
    mac = Path("/System/Library/Fonts/Supplemental")
    lin = Path("/usr/share/fonts/truetype/dejavu")
    return {
        "regular": [win / "segoeui.ttf", mac / "Arial.ttf", lin / "DejaVuSans.ttf", mpl / "DejaVuSans.ttf"],
        "bold": [win / "segoeuib.ttf", mac / "Arial Bold.ttf", lin / "DejaVuSans-Bold.ttf", mpl / "DejaVuSans-Bold.ttf"],
        # numerals: a DIN-style face reads like an instrument display
        "num": [win / "bahnschrift.ttf", win / "segoeuib.ttf", mac / "Arial Bold.ttf", mpl / "DejaVuSans-Bold.ttf"],
    }[kind]


class Typography:
    def __init__(self, override: Optional[str] = None):
        self.paths = {}
        for kind in ("regular", "bold", "num"):
            cands = ([Path(override)] if override and kind == "regular" else []) + _font_candidates(kind)
            self.paths[kind] = next((p for p in cands if p.is_file()), None)

    @lru_cache(maxsize=64)
    def font(self, kind: str, size: int) -> ImageFont.FreeTypeFont:
        p = self.paths.get(kind)
        if p is None:
            return ImageFont.load_default(size=size)
        f = ImageFont.truetype(str(p), size=max(6, int(size)))
        if kind == "num" and p.name.lower() == "bahnschrift.ttf":
            try:
                f.set_variation_by_name("Bold")
            except (OSError, ValueError):
                pass
        return f

    def width(self, text: str, kind: str, size: int) -> float:
        return self.font(kind, size).getlength(text)


def _rgba_to_blend(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """RGBA image -> (premultiplied BGR float32, 1-alpha float32) for fast repeated blending."""
    a = np.asarray(img, np.float32) / 255.0
    alpha = a[..., 3:4]
    premult = a[..., [2, 1, 0]] * alpha * 255.0
    return premult, 1.0 - alpha


def _frost(frame: np.ndarray, x: int, w: int) -> None:
    """Blur the video behind the panel (frosted glass) so text stays readable over any background."""
    region = frame[:, x:x + w]
    h = region.shape[0]
    small = cv2.resize(region, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    region[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blend(frame: np.ndarray, x: int, y: int, layer: tuple[np.ndarray, np.ndarray]) -> None:
    premult, inv = layer
    h, w = inv.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - x, y0 - y
    region = frame[y0:y1, x0:x1].astype(np.float32)
    region = region * inv[sy:sy + y1 - y0, sx:sx + x1 - x0] + premult[sy:sy + y1 - y0, sx:sx + x1 - x0]
    frame[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


def _translucent(img: Image.Image, draw_fn) -> None:
    """Draw semi-transparent shapes *over* `img`. ImageDraw replaces pixels (alpha included),
    so translucent fills drawn directly would punch see-through holes into the panel."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    img.alpha_composite(layer)


class Pills:
    """Cached rounded labels: one or two lines of text on a coloured, slightly translucent pill."""

    def __init__(self, typo: Typography, scale: float):
        self.t, self.k = typo, scale

    @lru_cache(maxsize=512)
    def make(self, big: str, small: str, fill: tuple, ink: tuple, opacity: float = 0.88) -> tuple:
        k = self.k
        fb, fs = self.t.font("num", int(30 * k)), self.t.font("bold", int(15 * k))
        pad_x, pad_y, gap = int(10 * k), int(6 * k), int(6 * k)
        wb = fb.getlength(big)
        ws = fs.getlength(small) if small else 0
        asc_b, desc_b = fb.getmetrics()
        w = int(wb + (gap + ws if small else 0) + 2 * pad_x)
        h = int(asc_b + desc_b * 0.4 + 2 * pad_y)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(h * 0.32), fill=(*fill, int(255 * opacity)))
        d.text((pad_x, pad_y), big, font=fb, fill=ink, anchor="la")
        if small:
            # small unit text sits on the big text's baseline
            d.text((pad_x + wb + gap, pad_y + asc_b), small, font=fs, fill=ink, anchor="ls")
        return _rgba_to_blend(img), (w, h)


# -- side panel -----------------------------------------------------------------------------
def margin_text(v: dict, key: str) -> str:
    """'±1.1', or '+0.5 −1.4' when an uncorrected one-sided parallax bias widens the lower side."""
    u, lo = v[f"speed_{key}_u_kmh"], v.get(f"speed_{key}_u_lower_kmh", v[f"speed_{key}_u_kmh"])
    return f"±{u:.1f}" if lo - u < 0.05 else f"+{u:.1f} −{lo:.1f}"


class Panel:
    """Leaderboard-style panel: totals, latest vehicle's speed profile and the list of vehicles."""

    def __init__(self, typo: Typography, rs: RenderSettings, results: list[dict], size: tuple[int, int],
                 opaque: bool):
        self.t, self.rs = typo, rs
        self.W, self.H = size
        self.opaque = opaque
        self.k = self.H / 1080
        self.items = sorted([r for r in results if r["valid"]], key=lambda r: r["t_exit_s"])
        self._key = None
        self._layer = None

    def layer_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        done = [r for r in self.items if r["t_exit_s"] <= t]
        newest_age = t - done[-1]["t_exit_s"] if done else 99.0
        fade = round(max(0.0, 1.0 - newest_age / 2.0), 1)  # highlight of the newest row, 2 s
        key = (len(done), fade)
        if key != self._key:
            self._key = key
            self._layer = _rgba_to_blend(self._draw(done, fade))
        return self._layer

    def _draw(self, done: list[dict], fade: float) -> Image.Image:
        k, W, H, rs = self.k, self.W, self.H, self.rs
        f = self.t.font
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, H], fill=(*INK, 255 if self.opaque else 222))  # over a blurred backdrop
        edge = 0 if rs.panel == "right" else W - int(3 * k)
        d.rectangle([edge, 0, edge + int(3 * k), H], fill=(*LINE_A, 255))
        pad = int(26 * k)
        x, y = pad, int(28 * k)
        inner = W - 2 * pad
        limit = rs.speed_limit_kmh

        if rs.title:
            y = self._wrap(d, rs.title, x, y, inner, f("bold", int(34 * k)), WHITE, max_lines=2)
        if rs.subtitle:
            y = self._wrap(d, rs.subtitle, x, y + int(2 * k), inner, f("regular", int(21 * k)), MUTED, max_lines=2)
        if rs.title or rs.subtitle:
            y += int(16 * k)

        # speed limit sign (white disc, red ring) + legend
        r = int(34 * k)
        cx, cy = x + r, y + r
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255), outline=(220, 30, 30, 255),
                  width=max(2, int(r * 0.2)))
        num = f"{limit:g}"
        d.text((cx, cy + int(1 * k)), num, font=f("num", int(r * (1.0 if len(num) < 3 else 0.8))),
               fill=(15, 15, 15), anchor="mm")
        lx = cx + r + int(14 * k)
        d.text((lx, cy - int(4 * k)), "SPEED LIMIT", font=f("bold", int(15 * k)), fill=MUTED, anchor="ls")
        d.text((lx, cy + int(4 * k)), f"{num} km/h", font=f("num", int(28 * k)), fill=WHITE, anchor="lt")
        y = cy + r + int(22 * k)

        # totals
        maxes = np.array([v["speed_max_kmh"] for v in done]) if done else np.zeros(0)
        over = int((maxes > limit).sum())
        cards = [
            ("VEHICLES", f"{len(done)}", WHITE),
            ("OVER LIMIT", f"{over}" + (f"  ({over / len(done):.0%})" if done else ""), BAD if over else WHITE),
            ("TOP SPEED", f"{maxes.max():.0f} km/h" if done else "–", speed_rgb(maxes.max(), limit) if done else WHITE),
            ("85th PERCENTILE", f"{np.percentile(maxes, 85):.0f} km/h" if len(done) >= 3 else "–", WHITE),
        ]
        cw = (inner - int(10 * k)) / 2
        ch = int(70 * k)
        for i, (lab, val, col) in enumerate(cards):
            bx = x + (i % 2) * (cw + int(10 * k))
            by = y + (i // 2) * (ch + int(10 * k))
            _translucent(img, lambda o: o.rounded_rectangle([bx, by, bx + cw, by + ch], radius=int(10 * k),
                                                            fill=(255, 255, 255, 16), outline=(255, 255, 255, 30)))
            d.text((bx + int(12 * k), by + int(10 * k)), lab, font=f("bold", int(13 * k)), fill=MUTED, anchor="lt")
            d.text((bx + int(12 * k), by + ch - int(10 * k)), val, font=f("num", int(30 * k)), fill=col, anchor="lb")
        y += 2 * ch + int(10 * k) + int(24 * k)

        # latest vehicle's momentary speed along the section
        last = done[-1] if done else None
        if last and last.get("profile"):
            y = self._sparkline(img, d, last, x, y, inner, int(108 * k), limit)
            y += int(18 * k)

        # list
        title = "FASTEST" if rs.panel_sort == "fastest" else "LATEST"
        d.text((x, y), title, font=f("bold", int(14 * k)), fill=MUTED, anchor="lt")
        y += int(24 * k)
        _translucent(img, lambda o: o.line([x, y, x + inner, y], fill=(255, 255, 255, 46), width=1))
        y += int(6 * k)
        footer_h = int(46 * k) if rs.footer else int(14 * k)
        row_h = int(64 * k)
        n_rows = max(0, min(rs.panel_rows, (H - footer_h - y) // row_h))
        if len(done) > n_rows:  # leave room for the "+ N more" line
            n_rows = max(0, min(n_rows, (H - footer_h - y - int(24 * k)) // row_h))
        rows = sorted(done, key=lambda v: -v["speed_max_kmh"]) if rs.panel_sort == "fastest" else done[::-1]
        newest = done[-1]["track_id"] if done else None
        if not done:
            d.text((x, y + int(14 * k)), "Waiting for the first vehicle…", font=f("regular", int(19 * k)),
                   fill=MUTED, anchor="lt")
        for v in rows[:n_rows]:
            col = speed_rgb(v["speed_max_kmh"], limit)
            if v["track_id"] == newest and fade > 0:
                box = [x - int(8 * k), y, x + inner + int(8 * k), y + row_h - int(6 * k)]
                _translucent(img, lambda o: o.rounded_rectangle(box, radius=int(8 * k), fill=(*col, int(60 * fade))))
            d.rounded_rectangle([x, y + int(8 * k), x + int(5 * k), y + row_h - int(14 * k)], radius=int(2 * k),
                                fill=(*col, 255))
            tx = x + int(16 * k)
            d.text((tx, y + int(10 * k)), v["class"].capitalize(), font=f("bold", int(20 * k)), fill=WHITE, anchor="lt")
            mm, ss = divmod(int(v["t_exit_s"]), 60)
            d.text((tx, y + row_h - int(14 * k)), f"{v['direction'].replace('->', ' → ')}  ·  {mm}:{ss:02d}",
                   font=f("regular", int(15 * k)), fill=MUTED, anchor="lb")
            rx = x + inner
            d.text((rx, y + int(6 * k)), "km/h", font=f("bold", int(14 * k)), fill=MUTED, anchor="rt")
            unit_w = f("bold", int(14 * k)).getlength("km/h") + int(6 * k)
            d.text((rx - unit_w, y + int(2 * k)), f"{v['speed_max_kmh']:.1f}", font=f("num", int(32 * k)),
                   fill=col, anchor="rt")
            sub = f"avg {v['speed_kmh']:.1f}"
            if rs.show_margin and v.get("speed_max_u_kmh") is not None:
                sub = f"{margin_text(v, 'max')}  ·  {sub}"
            d.text((rx, y + row_h - int(14 * k)), sub, font=f("regular", int(15 * k)), fill=MUTED, anchor="rb")
            y += row_h
        if len(rows) > n_rows and n_rows:
            d.text((x, y + int(2 * k)), f"+ {len(rows) - n_rows} more", font=f("regular", int(15 * k)),
                   fill=MUTED, anchor="lt")

        if rs.footer:
            d.text((x, H - int(22 * k)), rs.footer, font=f("regular", int(15 * k)), fill=MUTED, anchor="lb")
        return img

    def _wrap(self, d, text, x, y, width, font, fill, max_lines=2) -> int:
        words, line, lines = text.split(), "", []
        for w in words:
            cand = f"{line} {w}".strip()
            if font.getlength(cand) <= width or not line:
                line = cand
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        asc, desc = font.getmetrics()
        for ln in lines[:max_lines]:
            d.text((x, y), ln, font=font, fill=fill, anchor="la")
            y += asc + desc
        return y

    def _sparkline(self, img, d, v: dict, x: int, y: int, w: int, h: int, limit: float) -> int:
        k, f = self.k, self.t.font
        d.text((x, y), "LAST VEHICLE · SPEED ALONG THE SECTION", font=f("bold", int(13 * k)), fill=MUTED, anchor="lt")
        y += int(22 * k)
        prof = v["profile"]
        dd, vv = np.array(prof["d"]), np.array(prof["v"])
        lo = min(vv.min(), limit) - 3
        hi = max(vv.max(), limit) + 3
        span_d = max(dd[-1], 1e-6)

        def P(di, vi):
            return x + di / span_d * w, y + h - (vi - lo) / (hi - lo) * h

        _translucent(img, lambda o: o.rounded_rectangle([x, y, x + w, y + h], radius=int(8 * k), fill=(255, 255, 255, 12)))
        ly = P(0, limit)[1]
        for xx in range(int(x), int(x + w), int(12 * k)):  # dashed speed-limit line
            d.line([xx, ly, min(xx + int(6 * k), x + w), ly], fill=(220, 30, 30, 200), width=max(1, int(2 * k)))
        col = speed_rgb(v["speed_max_kmh"], limit)
        pts = [P(a, b) for a, b in zip(dd, vv)]
        d.line(pts, fill=(*col, 255), width=max(2, int(3 * k)), joint="curve")
        i = int(np.argmax(vv))
        px, py = pts[i]
        rr = int(6 * k)
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(*col, 255), outline=(255, 255, 255, 255), width=2)
        lab = f"max {v['speed_max_kmh']:.1f}"
        anchor = "rb" if px > x + w * 0.6 else "lb"
        d.text((px + (-rr if anchor == "rb" else rr), py - rr), lab, font=f("bold", int(16 * k)), fill=WHITE,
               anchor=anchor)
        d.text((x + w, y + h + int(4 * k)), f"{span_d:.0f} m", font=f("regular", int(13 * k)), fill=MUTED, anchor="rt")
        d.text((x, y + h + int(4 * k)), "0 m", font=f("regular", int(13 * k)), fill=MUTED, anchor="lt")
        return y + h + int(20 * k)


# -- per-frame overlay ------------------------------------------------------------------------
CORNER_COLS = [f"c{k}{a}" for k in range(4) for a in "xy"]


def filled_boxes(tracks: pd.DataFrame, max_gap_frames: int) -> pd.DataFrame:
    """Track boxes on every frame of each track: frames where the detector missed a vehicle
    (up to `max_gap_frames` in a row) get corners interpolated from the detections either side.
    Without this, boxes and labels blink whenever detection confidence dips below the threshold."""
    parts = []
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("frame").drop_duplicates("frame")
        f = g["frame"].to_numpy()
        if len(f) < 2:
            parts.append(g[["frame", "track_id", *CORNER_COLS]])
            continue
        full = np.arange(f[0], f[-1] + 1)
        nxt = np.searchsorted(f, full)  # index of the detection at or after each frame
        prev_f = f[np.maximum(nxt - 1, 0)]
        next_f = f[np.minimum(nxt, len(f) - 1)]
        keep = (np.isin(full, f)) | (next_f - prev_f <= max_gap_frames + 1)
        full = full[keep]
        C = g[CORNER_COLS].to_numpy(float)
        cols = {c: np.interp(full, f, C[:, k]) for k, c in enumerate(CORNER_COLS)}
        parts.append(pd.DataFrame({"frame": full, "track_id": tid, **cols}))
    if not parts:
        return pd.DataFrame(columns=["frame", "track_id", *CORNER_COLS])
    return pd.concat(parts, ignore_index=True)

class Overlay:
    def __init__(self, cfg: ProjectConfig, fps: float, cal, renders: dict[int, TrackRender],
                 tracks: pd.DataFrame, transforms: dict, size: tuple[int, int], src_size: tuple[int, int],
                 typo: Optional[Typography] = None):
        self.cfg, self.fps, self.cal, self.renders = cfg, fps, cal, renders
        self.rs = cfg.render
        self.W, self.H = size
        self.S = np.diag([size[0] / src_size[0], size[1] / src_size[1], 1.0])
        self.S_inv = np.linalg.inv(self.S)
        self.H_inv = np.linalg.inv(cal.H)
        self.k = self.H / 1080
        self.lw = max(1, int(round(2 * self.k)))
        self.typo = typo or Typography(self.rs.font)
        self.pills = Pills(self.typo, self.k)
        self.T_frames, self.T = transforms["frames"], transforms["T"]
        boxes = filled_boxes(tracks, int(round(cfg.tracking.lost_buffer_s * fps)))
        self.by_frame = {int(f): (g["track_id"].to_numpy(), g[CORNER_COLS].to_numpy().reshape(-1, 4, 2))
                         for f, g in boxes.groupby("frame")}
        self.trail_frames = int(round(self.rs.trail_s * fps))
        self.trail_px = {tid: calibration.apply_h(self.H_inv, r.world) for tid, r in renders.items()}
        # where the traffic is (reference px), to put line tags on the kerb side, off the lanes
        measured = [tid for tid, r in renders.items() if r.result]
        sel = tracks[tracks["track_id"].isin(measured)] if measured else tracks
        self.traffic_ref = sel[["rx", "ry"]].to_numpy(float).mean(0) if len(sel) else np.zeros(2)

    def T_at(self, n: int) -> np.ndarray:
        if len(self.T_frames) == 0:
            return np.eye(3)
        k = int(np.clip(np.searchsorted(self.T_frames, n), 0, len(self.T_frames) - 1))
        return self.T[k]

    def _pill(self, frame, x, y, big, small, fill, ink=WHITE, opacity=0.88, center=False):
        layer, (w, h) = self.pills.make(big, small, fill, ink, opacity)
        _blend(frame, int(x - (w / 2 if center else 0)), int(y - h), layer)

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
        limit = self.rs.speed_limit_kmh

        # measurement lines, with the measured section length between them
        gap = self.cal.line_gap_m
        for name, line, color in (("A", cal.line_a, LINE_A), ("B", cal.line_b, LINE_B)):
            if not line:
                continue
            p = transform_points(to_disp, np.array(line, float))
            cv2.line(frame, tuple(p[0].astype(int)), tuple(p[1].astype(int)), bgr(color), self.lw + 1, cv2.LINE_AA)
            ref_pts = np.array(line, float)
            far = int(np.argmax(np.linalg.norm(ref_pts - self.traffic_ref, axis=1)))
            out = p[far] - p[1 - far]
            out = out / max(np.linalg.norm(out), 1e-6)
            small = f"{gap:.1f} m from A" if name == "B" and gap else ""
            _, (pw_, ph_) = self.pills.make(name, small, color, INK, 0.95)
            # step outwards along the line until the pill clears its end
            c = p[far] + out * (0.5 * abs(out[0]) * pw_ + 0.5 * abs(out[1]) * ph_ + 8 * self.k)
            self._pill(frame, c[0], c[1] + ph_ / 2, name, small, color, INK, 0.95, center=True)

        ids, corners = self.by_frame.get(n, (np.zeros(0, int), np.zeros((0, 4, 2))))
        for tid, c in zip(ids, corners):
            r = self.renders.get(int(tid))
            res = r.result if r else None
            if res is None and not self.rs.show_unmeasured:
                continue  # parked cars, trams, tracks that never cross both lines
            pts = transform_points(to_disp, c)
            color, label = (220, 220, 220), None
            if res and r.t_first is not None:
                if t >= r.t_second:
                    vmax = res.get("speed_max_kmh", res["speed_kmh"])
                    color = speed_rgb(vmax, limit) if res["valid"] else INVALID
                    label = (f"{vmax:.1f}" + ("" if res["valid"] else "?"), "km/h max", color, INK)
                elif t > r.t_first:
                    kk = n - r.frames[0]
                    v = r.inst_kmh[kk] if 0 <= kk < len(r.inst_kmh) else np.nan
                    color = WHITE
                    if np.isfinite(v):
                        label = (f"{v:.0f}", "km/h", INK, WHITE)
            cv2.polylines(frame, [pts.astype(np.int32)], True, bgr(color), self.lw, cv2.LINE_AA)
            if r is not None and res and r.t_first is not None and t > r.t_first:
                kk = n - r.frames[0]
                lo = max(0, kk - self.trail_frames)
                if kk > lo:
                    trail = transform_points(to_disp, self.trail_px[int(tid)][lo:kk + 1]).astype(np.int32)
                    cv2.polylines(frame, [trail], False, bgr(color), self.lw, cv2.LINE_AA)
            if label:
                big, small, fill, ink = label
                self._pill(frame, pts[:, 0].mean(), pts[:, 1].min() - 6 * self.k, big, small, fill, ink, center=True)

        if self.rs.show_hud:
            self._pill(frame, 12 * self.k, 44 * self.k, f"{t:.2f} s", f"frame {n} · {self.fps:g} fps", INK, WHITE, 0.7)
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
    results, renders = compute_speeds(tracks, fps, cal.line_a_world, cal.line_b_world, cfg.speed)
    if rs.panel != "none" and rs.show_margin:
        margin_of_error(cfg, results, renders, cal, fps)
    tf = np.load(out / "transforms.npz")
    transforms = {"frames": tf["frames"], "T": tf["T"]}

    W = max(2, int(round(info.width * rs.scale / 2)) * 2)
    H = max(2, int(round(info.height * rs.scale / 2)) * 2)
    typo = Typography(rs.font)
    overlay = Overlay(cfg, fps, cal, renders, tracks, transforms, (W, H), (info.width, info.height), typo)

    panel = None
    out_w = W
    if rs.panel != "none":
        pw = max(2, int(round(W * rs.panel_width / 2)) * 2)
        extend = rs.panel_layout == "extend"
        panel = Panel(typo, rs, results, (pw, H), opaque=extend)
        if extend:
            out_w = W + pw
    video_x = pw if panel and rs.panel_layout == "extend" and rs.panel == "left" else 0
    panel_x = 0
    if panel:
        panel_x = (W if rs.panel_layout == "extend" else W - pw) if rs.panel == "right" else 0

    out_fps = rs.output_fps or fps
    ratio = rs.playback_speed * fps / out_fps  # source frames per output frame
    n_out = int(math.floor((info.nb_frames - 1) / ratio)) + 1

    reader = FFmpegReader(cfg.source, W, H, rs.lut)
    writer = FFmpegWriter(dst, out_w, H, out_fps, rs.encoder, rs.quality)
    canvas = np.zeros((H, out_w, 3), np.uint8)
    j = 0
    t0 = time.monotonic()
    try:
        for n, frame in enumerate(reader):
            if j >= n_out:
                break
            while j < n_out and int(round(j * ratio)) == n:
                drawn = overlay.draw(frame.copy(), n)
                if panel is None:
                    writer.write(drawn)
                else:
                    canvas[:, video_x:video_x + W] = drawn
                    if rs.panel_layout == "overlay":
                        _frost(canvas, panel_x, panel.W)
                    _blend(canvas, panel_x, 0, panel.layer_at(n / fps))
                    writer.write(canvas)
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
            "mode": rs.mode, "frames": j, "width": out_w, "height": H, "codec": writer.codec,
            "capture_fps": fps, "panel": rs.panel, "panel_layout": rs.panel_layout,
            "video_x": video_x, "video_width": W}
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
    boxes = filled_boxes(tracks, int(round(cfg.tracking.lost_buffer_s * fps)))

    def r1(a) -> list:
        return np.round(np.asarray(a, float), 1).ravel().tolist()

    track_info = {}
    for tid, r in renders.items():
        res = r.result
        track_info[str(tid)] = {
            "f0": int(r.frames[0]),
            "trail": r1(calibration.apply_h(H_inv, r.world)),
            # momentary speed per frame (shown between the lines)
            "prov": [None if not np.isfinite(v) else round(float(v), 1) for v in r.inst_kmh] if res else None,
            "cls": r.cls,
            "t_first": r.t_first, "t_second": r.t_second,
            "speed": res["speed_kmh"] if res else None,
            "speed_max": res.get("speed_max_kmh") if res else None,
            "dt": res["dt_s"] if res else None,
            "valid": res["valid"] if res else None,
        }
    return {
        "width": info.width, "height": info.height, "capture_fps": fps, "container_fps": info.container_fps,
        "line_a": cfg.calibration.line_a, "line_b": cfg.calibration.line_b, "roi": cfg.calibration.roi,
        "line_gap_m": cal.line_gap_m, "speed_limit_kmh": cfg.render.speed_limit_kmh,
        "trail_frames": int(round(cfg.render.trail_s * fps)),
        "T_frames": tf["frames"].tolist(), "T_inv": np.round(T_inv.reshape(-1, 9), 7).tolist(),
        "det_frame": boxes["frame"].astype(int).tolist(), "det_tid": boxes["track_id"].astype(int).tolist(),
        "det_corners": r1(boxes[CORNER_COLS].to_numpy()),
        "tracks": track_info,
    }
