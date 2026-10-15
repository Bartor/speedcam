"""Project folders: project.json (+ proxy.mp4, tracks.parquet, results.*, render.mp4)."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig
from ..timing import probe
from ..video import FrameGrabber

ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")
OUTPUT_FILES = {"proxy.mp4", "render.mp4", "results.csv", "results.json", "tracks.parquet",
                "stabilization.csv", "render.json"}


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._grabbers: dict[str, FrameGrabber] = {}

    def dir(self, pid: str) -> Path:
        if not ID_RE.match(pid):
            raise KeyError(pid)
        d = self.root / pid
        if not (d / "project.json").exists():
            raise KeyError(pid)
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
            return data

    def status(self, pid: str) -> dict:
        d = self.dir(pid)
        return {name.replace(".", "_"): (d / name).exists() for name in OUTPUT_FILES}

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append({"id": data["id"], "name": data["name"], "created": data["created"],
                            "source": data["config"]["source"]})
            except (OSError, ValueError, KeyError):
                continue
        return out

    def grabber(self, pid: str) -> FrameGrabber:
        with self._lock:
            g = self._grabbers.get(pid)
            if g is None:
                g = FrameGrabber(self.config(pid).source)
                self._grabbers[pid] = g
            return g
