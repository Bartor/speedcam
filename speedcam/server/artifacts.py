"""Per-job copies of what a job produced, so earlier runs stay downloadable after a project's
results or render are overwritten. Stored in <project>/jobs/<job id>/ (deleted with the project)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# outputs each kind of job writes into the project folder
ARTIFACTS: dict[str, list[str]] = {
    "analyze": ["results.json", "results.csv", "violations.csv", "tracks.parquet", "stabilization.csv"],
    "recompute": ["results.json", "results.csv", "violations.csv"],
    "render": ["render.mp4", "render.json"],
}
SETTINGS = "settings.json"  # the project's settings when the job finished
ALLOWED = {n for names in ARTIFACTS.values() for n in names} | {SETTINGS}


def job_dir(project_dir: Path, job_id: str) -> Path:
    return Path(project_dir) / "jobs" / job_id


def archive(project_dir: Path, job_id: str, kind: str) -> None:
    """Copy a finished job's outputs (and the settings it ran with) next to its record. Copies,
    not hard links: the next run rewrites those files in place."""
    names = ARTIFACTS.get(kind)
    if not names:
        return
    src = Path(project_dir)
    dst = job_dir(src, job_id)
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        if (src / n).exists():
            shutil.copy2(src / n, dst / n)
    p = src / "project.json"
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {})
        (dst / SETTINGS).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def listing(project_dir: Path, job_id: str, kind: str) -> list[dict]:
    d = job_dir(project_dir, job_id)
    if not d.is_dir():
        return []
    order = ARTIFACTS.get(kind, []) + [SETTINGS]
    return [{"name": n, "size": (d / n).stat().st_size} for n in order if (d / n).is_file()]


def delete(project_dir: Path, job_id: str) -> None:
    shutil.rmtree(job_dir(project_dir, job_id), ignore_errors=True)
