"""Video probing and real-time timing.

Real time is always derived as ``frame_index / capture_fps``. The container
frame rate is *not* trusted: DJI stores high-frame-rate clips as slow motion
at a lower playback rate, so container timestamps run slower than real time.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    container_fps: float
    nb_frames: int
    duration_s: float
    codec: str
    pix_fmt: str
    bit_depth: int
    color_transfer: str
    is_dji: bool
    srt_fps: Optional[float]
    suggested_capture_fps: float
    slowmo_suspect: bool
    warnings: list[str] = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    # clockwise degrees a player turns the stored frames to show them upright (display-matrix
    # side data, e.g. DJI vertical shooting). Analysis always works on the stored frames.
    display_rotation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _rate(s: str | None) -> float:
    if not s or s in ("0/0", "0"):
        return 0.0
    return float(Fraction(s))


def _ffprobe(args: list[str]) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", *args],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def sidecar_srt_fps(video_path: str | Path) -> Optional[float]:
    """Estimate the capture rate from a DJI sidecar .SRT (``DiffTime: 8ms`` per frame)."""
    p = Path(video_path)
    for cand in (p.with_suffix(".SRT"), p.with_suffix(".srt")):
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="ignore")
            diffs = [float(m) for m in re.findall(r"DiffTime\s*:\s*([\d.]+)\s*ms", text)]
            diffs = [d for d in diffs if d > 0]
            if len(diffs) >= 3:
                return 1000.0 / statistics.median(diffs)
    return None


def probe(path: str | Path) -> VideoInfo:
    path = str(path)
    data = _ffprobe(["-show_streams", "-show_format", "-select_streams", "v:0", path])
    if not data.get("streams"):
        raise ValueError(f"No video stream in {path}")
    s = data["streams"][0]
    fmt = data.get("format", {})
    tags = {**fmt.get("tags", {}), **s.get("tags", {})}

    fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
    duration = float(s.get("duration") or fmt.get("duration") or 0.0)
    nb = int(s.get("nb_frames") or 0)
    if nb <= 0:
        cnt = _ffprobe(["-count_packets", "-show_entries", "stream=nb_read_packets",
                        "-select_streams", "v:0", path])
        nb = int(cnt["streams"][0].get("nb_read_packets") or round(duration * fps))

    pix_fmt = s.get("pix_fmt", "")
    bit_depth = int(s.get("bits_per_raw_sample") or (10 if "10" in pix_fmt else 8))
    tag_text = " ".join(str(v) for v in tags.values()).lower()
    is_dji = "dji" in tag_text or Path(path).name.upper().startswith("DJI_")

    srt_fps = sidecar_srt_fps(path)
    warnings: list[str] = []
    slowmo = False
    suggested = fps
    if srt_fps and abs(srt_fps - fps) / fps > 0.1:
        slowmo = True
        suggested = round(srt_fps, 3)
        warnings.append(
            f"Sidecar SRT reports ~{srt_fps:.1f} fps but the container plays at {fps:.3f} fps: "
            f"this looks like a slow-motion clip. Real capture rate is probably {suggested:g} fps."
        )
    elif is_dji and fps <= 31:
        slowmo = True
        warnings.append(
            f"DJI clip stored at {fps:.3f} fps. If it was recorded as slow motion (e.g. 4K/100 or 4K/120), "
            "the container rate is NOT the capture rate — set capture_fps to the recorded rate."
        )
    if bit_depth > 8:
        warnings.append(f"{bit_depth}-bit source ({pix_fmt}); a browser proxy will be generated for scrubbing.")

    rot = next((sd.get("rotation") for sd in s.get("side_data_list", []) if "rotation" in sd), None)
    if rot is None and "rotate" in s.get("tags", {}):  # older muxers: a "rotate" tag, clockwise
        rot = -int(s["tags"]["rotate"])
    display_rotation = int(round(-float(rot or 0) / 90)) * 90 % 360  # side data is counter-clockwise

    return VideoInfo(
        path=path, width=int(s["width"]), height=int(s["height"]), container_fps=fps,
        nb_frames=nb, duration_s=duration, codec=s.get("codec_name", ""), pix_fmt=pix_fmt,
        bit_depth=bit_depth, color_transfer=s.get("color_transfer", ""), is_dji=is_dji,
        srt_fps=srt_fps, suggested_capture_fps=suggested, slowmo_suspect=slowmo,
        warnings=warnings, tags=tags, display_rotation=display_rotation,
    )


def frame_time(index: int | float, capture_fps: float) -> float:
    """Real time (s) of a frame. The only timing formula the pipeline uses."""
    if capture_fps <= 0:
        raise ValueError("capture_fps must be positive")
    return index / capture_fps


def check_timing(info: VideoInfo, capture_fps: float, real_duration_s: Optional[float] = None,
                 tolerance: float = 0.01) -> list[str]:
    """Cross-check capture_fps against clip metadata and an optional known real duration."""
    msgs: list[str] = []
    real = info.nb_frames / capture_fps
    if real_duration_s:
        err = abs(real - real_duration_s) / real_duration_s
        if err > tolerance:
            msgs.append(
                f"{info.nb_frames} frames / {capture_fps:g} fps = {real:.3f} s, but the known duration is "
                f"{real_duration_s:.3f} s ({err:.1%} off). Check capture_fps."
            )
    if info.srt_fps and abs(info.srt_fps - capture_fps) / capture_fps > 0.02:
        msgs.append(f"capture_fps {capture_fps:g} disagrees with sidecar SRT ({info.srt_fps:.2f} fps).")
    if abs(capture_fps - info.container_fps) / info.container_fps > 0.02:
        msgs.append(
            f"capture_fps {capture_fps:g} differs from container fps {info.container_fps:.3f}: "
            f"treating clip as slow motion ({capture_fps / info.container_fps:.2f}x). "
            f"Real duration {real:.3f} s vs container {info.duration_s:.3f} s."
        )
    return msgs
