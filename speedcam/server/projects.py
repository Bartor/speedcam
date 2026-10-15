"""Project folders: project.json (+ proxy.mp4, tracks.parquet, results.*, render.mp4, cache/).

The folders are the source of truth; `speedcam.db` in the data directory indexes them for the
project library and keeps job history (see db.py).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2

from ..config import ProjectConfig
from ..timing import probe
from ..video import FrameGrabber
from .db import Database

ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")
OUTPUT_FILES = {"proxy.mp4", "render.mp4", "results.csv", "results.json", "tracks.parquet",
                "stabilization.csv", "render.json"}
SCREENS = {"import", "calibrate", "process", "results"}
FRAME_CACHE_MAX = 400  # decoded full-res frames kept on disk per project


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._grabbers: dict[str, FrameGrabber] = {}
        self.db = Database(self.root / "speedcam.db")
        self.sync()

    # -- folders -------------------------------------------------------------------
    def dir(self, pid: str) -> Path:
        if not ID_RE.match(pid):
            raise KeyError(pid)
        d = self.root / pid
        if not (d / "project.json").exists():
            raise KeyError(pid)
        return d

    def cache_dir(self, pid: str, *parts: str) -> Path:
        d = self.dir(pid).joinpath("cache", *parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create(self, source: str, name: Optional[str] = None) -> dict:
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(str(src))
        info = probe(src)
        slug = re.sub(r"[^a-z0-9]+", "-", (name or src.stem).lower()).strip("-")[:40] or "project"
        pid = f"{slug}-{uuid.uuid4().hex[:6]}"
        d = self.root / pid
        d.mkdir(parents=True)
        cfg = ProjectConfig(source=str(src), capture_fps=None)
        data = {"id": pid, "name": name or src.name, "created": time.time(),
                "info": info.to_dict(), "config": cfg.model_dump(mode="json")}
        self._write(pid, data)
        self.refresh(pid, opened=time.time(), last_screen="import")
        return data

    def _write(self, pid: str, data: dict) -> None:
        p = self.root / pid / "project.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)

    def load(self, pid: str) -> dict:
        return json.loads((self.dir(pid) / "project.json").read_text(encoding="utf-8"))

    def config(self, pid: str) -> ProjectConfig:
        return ProjectConfig.model_validate(self.load(pid)["config"])

    def save_config(self, pid: str, cfg: ProjectConfig, **extra) -> dict:
        with self._lock:
            data = self.load(pid)
            data["config"] = cfg.model_dump(mode="json")
            data.update(extra)
            self._write(pid, data)
        self.refresh(pid)
        return data

    def status(self, pid: str) -> dict:
        d = self.dir(pid)
        return {name.replace(".", "_"): (d / name).exists() for name in OUTPUT_FILES}

    # -- index (database) ------------------------------------------------------------
    def summary(self, pid: str) -> dict[str, Any]:
        """Library row for a project, derived from its folder."""
        from .. import calibration as calib

        d = self.dir(pid)
        data = self.load(pid)
        cfg = data["config"]
        info = data["info"]
        st = self.status(pid)
        row: dict[str, Any] = {
            "id": pid, "name": data["name"], "source": cfg["source"], "created": data["created"],
            "updated": (d / "project.json").stat().st_mtime,
            "source_exists": int(Path(cfg["source"]).is_file()),
            "width": info["width"], "height": info["height"], "capture_fps": cfg.get("capture_fps"),
            "duration_s": info["nb_frames"] / (cfg.get("capture_fps") or info["container_fps"]),
            "has_proxy": int(st["proxy_mp4"]), "has_results": int(st["results_json"]),
            "has_render": int(st["render_mp4"]),
            "n_vehicles": None, "n_valid": None, "mean_kmh": None, "p85_kmh": None, "analyzed": None,
        }
        try:
            res = calib.compute(ProjectConfig.model_validate(cfg).calibration)
            row["calibrated"] = int(res.fit_ok and bool(cfg["calibration"].get("line_a")) and
                                    bool(cfg["calibration"].get("line_b")))
        except ValueError:
            row["calibrated"] = 0
        if st["results_json"]:
            try:
                r = json.loads((d / "results.json").read_text(encoding="utf-8"))
                s = r.get("stats", {})
                row.update(n_vehicles=s.get("n_total"), n_valid=s.get("n_valid"), mean_kmh=s.get("mean"),
                           p85_kmh=s.get("p85"), analyzed=(d / "results.json").stat().st_mtime)
            except (OSError, ValueError):
                pass
        return row

    def refresh(self, pid: str, **extra) -> None:
        try:
            self.db.upsert_project({**self.summary(pid), **extra})
        except KeyError:
            pass

    def sync(self) -> None:
        """Bring the index in line with the folders on disk (new, changed and removed projects)."""
        on_disk = set()
        for p in self.root.glob("*/project.json"):
            pid = p.parent.name
            if not ID_RE.match(pid):
                continue
            on_disk.add(pid)
            try:
                self.refresh(pid)
            except (OSError, ValueError, KeyError):
                continue
        for pid in self.db.project_ids() - on_disk:
            self.db.delete_project(pid)

    def list(self) -> list[dict]:
        rows = self.db.projects()
        for r in rows:  # cheap and keeps "file missing" honest if a drive was unplugged
            r["source_exists"] = Path(r["source"]).is_file()
        return rows

    def touch(self, pid: str, screen: Optional[str]) -> None:
        self.dir(pid)
        fields: dict[str, Any] = {"opened": time.time()}
        if screen in SCREENS:
            fields["last_screen"] = screen
        self.db.update_project(pid, **fields)

    def rename(self, pid: str, name: str) -> None:
        with self._lock:
            data = self.load(pid)
            data["name"] = name.strip() or data["name"]
            self._write(pid, data)
        self.refresh(pid)

    def relink(self, pid: str, source: str) -> list[str]:
        """Point a project at a moved/renamed video. Returns warnings if it doesn't look like the same clip."""
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(str(src))
        info = probe(src)
        with self._lock:
            data = self.load(pid)
            old = data["info"]
            warnings = [f"{k}: was {old[k]}, now {getattr(info, k)}" for k in ("width", "height", "nb_frames")
                        if old[k] != getattr(info, k)]
            data["config"]["source"] = str(src)
            data["info"] = info.to_dict()
            self._write(pid, data)
            g = self._grabbers.pop(pid, None)
        if g:
            g.close()
        shutil.rmtree(self.dir(pid) / "cache", ignore_errors=True)
        self.refresh(pid)
        return warnings

    def delete(self, pid: str) -> None:
        """Delete the project folder (results, proxy, cache). The source video is not touched."""
        d = self.dir(pid)
        with self._lock:
            g = self._grabbers.pop(pid, None)
        if g:
            g.close()
        shutil.rmtree(d)
        self.db.delete_project(pid)

    # -- caches ----------------------------------------------------------------------
    def grabber(self, pid: str) -> FrameGrabber:
        with self._lock:
            g = self._grabbers.get(pid)
            if g is None:
                g = FrameGrabber(self.config(pid).source)
                self._grabbers[pid] = g
            return g

    def frame_file(self, pid: str, n: int, fmt: str = "jpg", max_width: Optional[int] = None) -> Path:
        """Encoded frame from the disk cache, decoding the original only on a miss."""
        fmt = "png" if fmt == "png" else "jpg"
        d = self.cache_dir(pid, "frames")
        f = d / f"{n}_{max_width or 'full'}.{fmt}"
        if f.exists():
            return f
        img = self.grabber(pid).get(n)
        if max_width and img.shape[1] > max_width:
            k = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if fmt == "png" else [cv2.IMWRITE_JPEG_QUALITY, 92]
        tmp = f.with_suffix(f".tmp.{fmt}")
        cv2.imwrite(str(tmp), img, params)
        tmp.replace(f)
        self._prune(d, FRAME_CACHE_MAX)
        return f

    @staticmethod
    def _prune(d: Path, keep: int) -> None:
        files = sorted(d.glob("*.*"), key=lambda p: p.stat().st_mtime)
        for p in files[:-keep] if len(files) > keep else []:
            p.unlink(missing_ok=True)

    def thumbnail(self, pid: str) -> Path:
        f = self.cache_dir(pid) / "thumb.jpg"
        if not f.exists():
            ref = self.config(pid).calibration.reference_frame
            src = self.frame_file(pid, ref, "jpg", 480)
            shutil.copyfile(src, f)
        return f

    def overlay(self, pid: str) -> dict:
        """Overlay data, cached until the tracks or the settings it depends on change."""
        from ..render import overlay_data

        d = self.dir(pid)
        cfg = self.config(pid)
        key_src = json.dumps({
            "tracks": (d / "tracks.parquet").stat().st_mtime,
            "cal": cfg.calibration.model_dump(mode="json"), "speed": cfg.speed.model_dump(mode="json"),
            "fps": cfg.capture_fps, "limit": cfg.render.speed_limit_kmh, "trail": cfg.render.trail_s,
        }, sort_keys=True)
        key = hashlib.sha1(key_src.encode()).hexdigest()
        f = self.cache_dir(pid) / "overlay.json"
        if f.exists():
            try:
                cached = json.loads(f.read_text(encoding="utf-8"))
                if cached.get("key") == key:
                    return cached["data"]
            except (OSError, ValueError):
                pass
        data = overlay_data(cfg, d)
        f.write_text(json.dumps({"key": key, "data": data}), encoding="utf-8")
        return data
