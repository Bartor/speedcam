"""Job runner with N parallel slots. Each job runs in its own spawned worker process (GPU-bound);
progress flows back through a multiprocessing queue and is exposed over SSE. At most one job per
project runs at a time, so jobs never read files another job of the same project is writing."""
from __future__ import annotations

import base64
import multiprocessing as mp
import queue as queue_mod
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..pipeline import Reporter


class QueueReporter(Reporter):
    def __init__(self, q, cancel_event):
        self.q, self.cancel_event = q, cancel_event

    def progress(self, stage, done, total, **extra):
        self.q.put({"type": "progress", "stage": stage, "done": done, "total": total, **extra})

    def log(self, msg, level="info"):
        self.q.put({"type": "log", "level": level, "message": msg})

    def preview(self, jpeg: bytes):
        self.q.put({"type": "preview", "jpeg": base64.b64encode(jpeg).decode()})

    def cancelled(self):
        return self.cancel_event.is_set()


def run_job(kind: str, project_dir: str, q, cancel_event, params: Optional[dict] = None) -> None:
    """Worker-process entry point."""
    import json
    import logging

    from ..config import ProjectConfig

    logging.basicConfig(level=logging.INFO)
    rep = QueueReporter(q, cancel_event)
    d = Path(project_dir)
    try:
        data = json.loads((d / "project.json").read_text(encoding="utf-8"))
        cfg = ProjectConfig.model_validate(data["config"])
        if kind == "proxy":
            from ..video import build_proxy

            total = data["info"]["nb_frames"]
            tmp = d / "proxy.tmp.mp4"
            build_proxy(cfg.source, tmp, total, progress=lambda n, t: rep.progress("building proxy", n, t),
                        should_cancel=rep.cancelled)
            tmp.replace(d / "proxy.mp4")
            result: Any = {"file": "proxy.mp4"}
        elif kind == "analyze":
            from ..pipeline import analyze

            summary = analyze(cfg, d, rep)
            result = {"n_vehicles": len(summary["vehicles"]), "n_valid": sum(v["valid"] for v in summary["vehicles"])}
        elif kind == "render":
            from ..render import render

            result = render(cfg, d, None, rep)
        elif kind == "scan":
            from ..scan import scan

            sc = scan(cfg, d, rep)
            result = {"samples": len(sc["frames"]), "with_vehicles": sum(c > 0 for c in sc["counts"]),
                      "mode": sc["mode"], "elapsed_s": sc["elapsed_s"]}
        elif kind == "masks":
            # outline the vehicles near solid lines drawn after the analysis, then recompute
            from .. import calibration
            from ..pipeline import load_tracks, outline_near_lines, recompute, resolve_capture_fps
            from ..speed import compute_speeds
            from ..timing import probe

            info = probe(cfg.source)
            fps = resolve_capture_fps(cfg, info, rep)
            cal = calibration.for_measuring(cfg.calibration, (info.width, info.height))
            df = load_tracks(cfg, d, cal)
            _, renders = compute_speeds(df, fps, cal.line_a_world, cal.line_b_world, cfg.speed,
                                        paths=cal.path_geoms, H=cal.H)
            n = outline_near_lines(cfg, cal, df, renders, fps, d, rep)
            rep.progress("recomputing", 0, 1)
            summary = recompute(cfg, d)
            v = summary.get("violations") or {}
            result = {"outlined": n, "n_solid_line": v.get("n_solid_line"), "cases": len(v.get("solid_cases", []))}
        elif kind == "preview":
            from ..pipeline import analyze

            # short tracked run from the given frame; results go to a scratch subfolder
            start = int((params or {}).get("frame", 0))
            seconds = float((params or {}).get("seconds", 3.0))
            fps = cfg.capture_fps or data["info"]["suggested_capture_fps"]
            cfg.start_frame, cfg.end_frame = start, start + int(round(seconds * fps))
            cfg.parts = None  # a preview looks at exactly the frames asked for
            summary = analyze(cfg, d / "preview", rep, preview_every_s=0.25)
            result = {**summary["run"], "n_vehicles": len(summary["vehicles"]),
                      "vehicles": [{k: v[k] for k in ("track_id", "speed_kmh", "valid", "flags")}
                                   for v in summary["vehicles"]]}
        else:
            raise ValueError(f"unknown job kind {kind}")
        q.put({"type": "done", "result": result})
    except InterruptedError:
        q.put({"type": "cancelled"})
    except Exception as e:  # noqa: BLE001 - report everything to the UI
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})


@dataclass
class Job:
    id: str
    project_id: str
    kind: str
    project_dir: str
    params: dict = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: dict = field(default_factory=dict)
    preview: Optional[str] = None
    preview_seq: int = 0
    log: list = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    cancel_event: Any = None
    seq: int = 0  # bumps on any change; SSE clients poll it

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "project_id", "kind", "status", "created", "started",
                                              "finished", "progress", "result", "error")} | \
               {"log": self.log[-200:]}

    def brief(self) -> dict:
        """Status + progress only (for the global jobs indicator, polled often)."""
        return {k: getattr(self, k) for k in ("id", "project_id", "kind", "status", "created", "started", "progress")}


class JobManager:
    # jobs that load the detection model on the GPU (a batched analysis already saturates it)
    GPU_KINDS = {"analyze", "preview", "scan", "masks"}

    def __init__(self, on_change: Optional[Callable[[Job], None]] = None, slots: int = 2, gpu_slots: int = 1):
        """`on_change(job)` is called on every status transition (queued, running, finished).
        `slots`: jobs running at once. `gpu_slots`: of those, how many may run detection (each
        needs ~6-9 GB of VRAM with batching); proxy and render jobs use CPU + the video encoder."""
        self.on_change = on_change
        self.slots = max(1, int(slots))
        self.gpu_slots = max(1, int(gpu_slots))
        self.ctx = mp.get_context("spawn")
        self.jobs: dict[str, Job] = {}
        self.pending: deque[Job] = deque()
        self.cv = threading.Condition()
        self.running: dict[str, Job] = {}
        self._procs: dict[str, Any] = {}
        self._threads = [threading.Thread(target=self._loop, daemon=True, name=f"job-slot-{k}")
                         for k in range(self.slots)]
        for t in self._threads:
            t.start()

    def _notify(self, job: Job) -> None:
        if self.on_change:
            try:
                self.on_change(job)
            except Exception:  # noqa: BLE001 - bookkeeping must never break the runner
                pass

    def submit(self, project_id: str, kind: str, project_dir: Path, params: Optional[dict] = None) -> Job:
        job = Job(uuid.uuid4().hex[:12], project_id, kind, str(project_dir), params or {},
                  cancel_event=self.ctx.Event())
        with self.cv:
            self.jobs[job.id] = job
            self.pending.append(job)
            self.cv.notify_all()
        self._notify(job)
        return job

    def active(self) -> list[Job]:
        with self.cv:
            return list(self.running.values()) + list(self.pending)

    def active_for(self, project_id: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.project_id == project_id and j.status in ("queued", "running")]

    def shutdown(self, timeout: float = 3.0) -> None:
        """Server is exiting: drop queued jobs and stop running workers (they hold the GPU)."""
        with self.cv:
            while self.pending:
                job = self.pending.popleft()
                job.status, job.finished = "cancelled", time.time()
                self._notify(job)
            running = [(j, self._procs.get(j.id)) for j in self.running.values()]
        for job, _ in running:
            job.cancel_event.set()
        deadline = time.time() + timeout
        for _, proc in running:
            if proc is not None:
                proc.join(max(0.0, deadline - time.time()))
        for _, proc in running:
            if proc is not None and proc.is_alive():
                proc.kill()
                proc.join(2)

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        with self.cv:
            if job.status == "queued":
                self.pending.remove(job)
                job.status, job.finished = "cancelled", time.time()
                job.seq += 1
                self._notify(job)
            elif job.status == "running":
                job.cancel_event.set()
        return job

    def _next(self) -> Optional[Job]:
        """Oldest queued job whose project has nothing running and, for detection jobs, whose GPU
        slot is free (call with self.cv held)."""
        busy = {j.project_id for j in self.running.values()}
        gpu_free = sum(j.kind in self.GPU_KINDS for j in self.running.values()) < self.gpu_slots
        return next((j for j in self.pending
                     if j.project_id not in busy and (gpu_free or j.kind not in self.GPU_KINDS)), None)

    def _loop(self) -> None:
        while True:
            with self.cv:
                while (job := self._next()) is None:
                    self.cv.wait()
                self.pending.remove(job)
                job.status, job.started = "running", time.time()
                job.seq += 1
                self.running[job.id] = job
            self._notify(job)
            try:
                self._run(job)
            finally:
                with self.cv:
                    self.running.pop(job.id, None)
                    self._procs.pop(job.id, None)
                    self.cv.notify_all()

    def _run(self, job: Job) -> None:
        q = self.ctx.Queue()
        proc = self.ctx.Process(target=run_job, args=(job.kind, job.project_dir, q, job.cancel_event, job.params), daemon=True)
        proc.start()
        with self.cv:
            self._procs[job.id] = proc
        cancel_deadline = None
        final = None
        while True:
            try:
                ev = q.get(timeout=0.25)
            except queue_mod.Empty:
                ev = None
            if ev is not None:
                t = ev["type"]
                if t == "progress":
                    job.progress = {k: v for k, v in ev.items() if k != "type"}
                elif t == "preview":
                    job.preview = ev["jpeg"]
                    job.preview_seq += 1
                elif t == "log":
                    job.log.append({"t": time.time(), "level": ev["level"], "message": ev["message"]})
                else:
                    final = ev
                job.seq += 1
                continue
            if job.cancel_event.is_set() and cancel_deadline is None:
                cancel_deadline = time.time() + 15
            if cancel_deadline and time.time() > cancel_deadline and proc.is_alive():
                proc.kill()
            if not proc.is_alive():
                # drain anything left
                try:
                    while True:
                        ev = q.get_nowait()
                        if ev["type"] in ("done", "error", "cancelled"):
                            final = ev
                except queue_mod.Empty:
                    pass
                break
        proc.join(timeout=5)
        if final is None:
            final = {"type": "cancelled"} if job.cancel_event.is_set() else \
                {"type": "error", "message": f"worker exited with code {proc.exitcode}"}
        if final["type"] == "done":
            job.status, job.result = "done", final.get("result")
        elif final["type"] == "cancelled":
            job.status = "cancelled"
        else:
            job.status, job.error, job.traceback = "error", final.get("message"), final.get("traceback")
        job.finished = time.time()
        job.seq += 1
        self._notify(job)
