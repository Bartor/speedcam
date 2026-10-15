"""Decoding, single-frame access, encoder selection and proxy generation."""
from __future__ import annotations

import functools
import logging
import re
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterator, Optional

import av
import numpy as np

log = logging.getLogger(__name__)


def _stream_fps(stream) -> float:
    rate = stream.average_rate or stream.guessed_rate or stream.base_rate
    return float(rate)


def iter_frames(path: str | Path, start: int = 0, end: Optional[int] = None,
                pts_check: Optional[Callable[[int, float, float], None]] = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr) sequentially. Index counts decoded frames from 0.

    `pts_check(index, pts_seconds, expected_seconds)` is called for every frame so
    the caller can sanity-check container timestamps (dropped/duplicated frames).
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = _stream_fps(stream)
        t0 = None
        for i, frame in enumerate(container.decode(stream)):
            if end is not None and i >= end:
                break
            if frame.time is not None:
                if t0 is None:
                    t0 = frame.time - i / fps
                if pts_check:
                    pts_check(i, frame.time - t0, i / fps)
            if i < start:
                continue
            yield i, frame.to_ndarray(format="bgr24")


class FrameGrabber:
    """Random access to frames of the original video by index (seek + decode forward)."""

    def __init__(self, path: str | Path, cache_size: int = 8):
        self.path = str(path)
        self._lock = threading.Lock()
        self._container = av.open(self.path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self.fps = _stream_fps(self._stream)
        self._tb = float(self._stream.time_base)
        self._start = self._stream.start_time or 0
        self._iter = None
        self._last = -10**9
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _index_of(self, frame) -> int:
        return int(round((frame.pts - self._start) * self._tb * self.fps))

    def get(self, n: int) -> np.ndarray:
        with self._lock:
            if n in self._cache:
                self._cache.move_to_end(n)
                return self._cache[n]
            if self._iter is None or not (self._last < n <= self._last + 90):
                target = int(n / self.fps / self._tb) + self._start
                self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
                self._iter = self._container.decode(self._stream)
                self._last = -10**9
            result = None
            for frame in self._iter:
                idx = self._index_of(frame)
                self._last = idx
                if idx >= n:
                    result = frame.to_ndarray(format="bgr24")
                    break
            if result is None:
                self._iter = None
                raise IndexError(f"Frame {n} not found")
            self._cache[n] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return result

    def close(self) -> None:
        self._container.close()


@functools.lru_cache(maxsize=None)
def available_encoder(preference: str = "auto") -> tuple[str, list[str]]:
    """Return (codec, extra args) for H.264, testing hardware encoders with a tiny encode."""
    candidates = {
        "nvenc": ("h264_nvenc", ["-preset", "p5", "-rc", "vbr"]),
        "videotoolbox": ("h264_videotoolbox", []),
        "x264": ("libx264", ["-preset", "medium"]),
    }
    order = [preference] if preference != "auto" else ["nvenc", "videotoolbox", "x264"]
    for key in order:
        codec, args = candidates[key]
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", codec, "-f", "null", "-"],
            capture_output=True,
        )
        if test.returncode == 0:
            return codec, args
    return candidates["x264"]


def quality_args(codec: str, quality: int) -> list[str]:
    if codec == "h264_nvenc":
        return ["-cq", str(quality), "-b:v", "0"]
    if codec == "h264_videotoolbox":
        return ["-q:v", str(max(1, 100 - quality * 2))]
    return ["-crf", str(quality)]


def run_ffmpeg_with_progress(cmd: list[str], total_frames: int,
                             progress: Optional[Callable[[int, int], None]] = None,
                             should_cancel: Optional[Callable[[], bool]] = None) -> None:
    proc = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    stderr_lines: list[str] = []
    t = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)
    t.start()
    for line in proc.stdout:
        m = re.match(r"frame=(\d+)", line)
        if m and progress:
            progress(int(m.group(1)), total_frames)
        if should_cancel and should_cancel():
            proc.kill()
            proc.wait()
            raise InterruptedError("cancelled")
    proc.wait()
    t.join(timeout=2)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + "".join(stderr_lines[-30:]))


def build_proxy(src: str | Path, dst: str | Path, total_frames: int, height: int = 1080,
                progress: Optional[Callable[[int, int], None]] = None,
                should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """8-bit H.264 proxy for browser scrubbing. Frames stay 1:1 with the source
    (same count, same container rate), so proxy frame n == source frame n."""
    codec, args = available_encoder("auto")
    with av.open(str(src)) as c:
        fps = _stream_fps(c.streams.video[0])
    gop = max(1, int(round(fps / 2)))
    cmd = ["ffmpeg", "-hide_banner", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an",
           "-vf", f"scale=-2:{height}:flags=bicubic,format=yuv420p", "-fps_mode", "passthrough",
           "-c:v", codec, *args, *quality_args(codec, 23), "-g", str(gop),
           "-movflags", "+faststart", str(dst)]
    run_ffmpeg_with_progress(cmd, total_frames, progress, should_cancel)


class FFmpegReader:
    """Sequential decode through ffmpeg (optional LUT and scaling) into BGR frames."""

    def __init__(self, src: str | Path, width: int, height: int, lut: Optional[str] = None):
        filters = []
        if lut:
            lut_path = str(Path(lut).resolve()).replace("\\", "/").replace(":", "\\:")
            filters.append(f"lut3d=file='{lut_path}'")
        filters.append(f"scale={width}:{height}:flags=bicubic")
        self.w, self.h = width, height
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an",
             "-vf", ",".join(filters), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, bufsize=width * height * 3 * 2,
        )
        self.frame_bytes = width * height * 3

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                return
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FFmpegWriter:
    def __init__(self, dst: str | Path, width: int, height: int, fps: float,
                 encoder: str = "auto", quality: int = 20):
        codec, args = available_encoder(encoder)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", codec, *args,
             *quality_args(codec, quality), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
            stdin=subprocess.PIPE,
        )
        self.codec = codec

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoder failed")

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()
