"""FastAPI app: REST + SSE API under /api, built web UI served at /."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import calibration as calib
from ..config import Calibration, ProjectConfig
from ..timing import check_timing
from .jobs import JobManager
from .projects import OUTPUT_FILES, ProjectStore

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
STATIC = Path(__file__).parent / "static"


class CreateProject(BaseModel):
    source: str
    name: Optional[str] = None


def create_app(data_dir: Path) -> FastAPI:
    store = ProjectStore(data_dir)
    store.db.mark_interrupted()

    def on_job_change(job) -> None:
        store.db.record_job(job.public())
        if job.status in ("done", "error", "cancelled") and job.kind in ("proxy", "analyze", "render"):
            store.refresh(job.project_id)

    jobs = JobManager(on_change=on_job_change)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await asyncio.to_thread(jobs.shutdown)
        store.db.close()

    app = FastAPI(title="speedcam", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.state.store, app.state.jobs = store, jobs

    def _dir(pid: str) -> Path:
        try:
            return store.dir(pid)
        except KeyError:
            raise HTTPException(404, f"project {pid} not found")

    def _project(pid: str) -> dict:
        _dir(pid)
        data = store.load(pid)
        cfg = ProjectConfig.model_validate(data["config"])
        data["status"] = store.status(pid)
        data["source_exists"] = Path(cfg.source).is_file()
        data["jobs"] = [j.public() for j in jobs.active_for(pid)]
        info = probe_cached(data)
        data["timing_warnings"] = check_timing(info, cfg.capture_fps, cfg.real_duration_s) if cfg.capture_fps else []
        try:
            data["calibration_result"] = calib.compute(cfg.calibration, (info.width, info.height)).summary()
        except ValueError as e:
            data["calibration_result"] = {"error": str(e)}
        return data

    def probe_cached(data: dict):
        from ..timing import VideoInfo

        return VideoInfo(**data["info"])

    # -- filesystem picker -------------------------------------------------------
    @app.get("/api/fs")
    def list_dir(path: Optional[str] = None):
        p = Path(path).expanduser() if path else Path.home()
        if p.is_file():
            p = p.parent
        if not p.is_dir():
            raise HTTPException(404, f"{p} is not a directory")
        entries = []
        try:
            for e in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        entries.append({"name": e.name, "path": str(e), "dir": True})
                    elif e.suffix.lower() in VIDEO_EXT:
                        entries.append({"name": e.name, "path": str(e), "dir": False, "size": e.stat().st_size})
                except OSError:
                    continue
        except PermissionError:
            raise HTTPException(403, f"cannot read {p}")
        return {"path": str(p.resolve()), "parent": str(p.resolve().parent), "entries": entries}

    # -- projects ----------------------------------------------------------------
    @app.get("/api/projects")
    def list_projects():
        return store.list()

    @app.post("/api/projects")
    def create_project(body: CreateProject):
        try:
            data = store.create(body.source, body.name)
        except FileNotFoundError as e:
            raise HTTPException(400, f"file not found: {e}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        job = jobs.submit(data["id"], "proxy", store.dir(data["id"]))
        return {**_project(data["id"]), "proxy_job": job.id}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return _project(pid)

    @app.post("/api/projects/{pid}/opened")
    def project_opened(pid: str, body: Optional[dict[str, Any]] = Body(None)):
        """Remember when a project was last opened and on which screen (to resume there)."""
        _dir(pid)
        store.touch(pid, (body or {}).get("screen"))
        return {"ok": True}

    @app.post("/api/projects/{pid}/relink")
    def relink(pid: str, body: dict[str, Any] = Body(...)):
        _dir(pid)
        try:
            warnings = store.relink(pid, str(body.get("source", "")))
        except FileNotFoundError as e:
            raise HTTPException(400, f"file not found: {e}")
        return {**_project(pid), "relink_warnings": warnings}

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        _dir(pid)
        if jobs.active_for(pid):
            raise HTTPException(409, "a job is running for this project; cancel it first")
        store.delete(pid)
        return {"deleted": pid}

    @app.get("/api/projects/{pid}/thumb.jpg")
    def thumbnail(pid: str):
        _dir(pid)
        try:
            return FileResponse(store.thumbnail(pid), media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})
        except (IndexError, EOFError, OSError, ValueError) as e:  # missing source etc.
            raise HTTPException(404, f"no thumbnail: {e}")

    @app.get("/api/projects/{pid}/jobs")
    def project_jobs(pid: str):
        _dir(pid)
        return store.db.jobs_for(pid)

    @app.patch("/api/projects/{pid}")
    def patch_project(pid: str, body: dict[str, Any] = Body(...)):
        _dir(pid)
        data = store.load(pid)
        merged = data["config"]
        for k, v in body.items():
            if k == "name":
                continue
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        try:
            cfg = ProjectConfig.model_validate(merged)
        except ValueError as e:
            raise HTTPException(422, str(e))
        store.save_config(pid, cfg)
        if "name" in body:
            store.rename(pid, str(body["name"]))
        return _project(pid)

    @app.post("/api/projects/{pid}/proxy")
    def rebuild_proxy(pid: str):
        return jobs.submit(pid, "proxy", _dir(pid)).public()

    @app.get("/api/projects/{pid}/frame/{n}")
    def get_frame(pid: str, n: int, fmt: str = "jpg", max_width: Optional[int] = None):
        _dir(pid)
        try:
            f = store.frame_file(pid, n, fmt, max_width)
        except (IndexError, EOFError):
            raise HTTPException(404, f"frame {n} not available")
        return FileResponse(f, media_type="image/png" if fmt == "png" else "image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/projects/{pid}/proxy.mp4")
    def get_proxy(pid: str):
        p = _dir(pid) / "proxy.mp4"
        if not p.exists():
            raise HTTPException(404, "proxy not built yet")
        return FileResponse(p, media_type="video/mp4")

    @app.put("/api/projects/{pid}/calibration")
    def put_calibration(pid: str, cal: Calibration):
        _dir(pid)
        cfg = store.config(pid)
        cfg.calibration = cal
        data = store.save_config(pid, cfg)
        info = data["info"]
        try:
            return {"saved": True, "result": calib.compute(cal, (info["width"], info["height"])).summary()}
        except ValueError as e:
            return {"saved": True, "result": {"error": str(e)}}

    @app.post("/api/projects/{pid}/analyze")
    def start_analyze(pid: str):
        _dir(pid)
        cfg = store.config(pid)
        if not cfg.capture_fps:
            raise HTTPException(400, "Set capture_fps first")
        if not (cfg.calibration.line_a and cfg.calibration.line_b):
            raise HTTPException(400, "Draw measurement lines A and B first")
        try:
            res = calib.compute(cfg.calibration)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not res.fit_ok:
            raise HTTPException(400, res.warnings[0])
        return jobs.submit(pid, "analyze", _dir(pid)).public()

    @app.post("/api/projects/{pid}/preview")
    def preview_run(pid: str, frame: Optional[int] = None, seconds: float = 3.0):
        """Detect + stabilize + track a few seconds and stream annotated previews, without touching results."""
        d = _dir(pid)
        cfg = store.config(pid)
        if not cfg.capture_fps:
            raise HTTPException(400, "Set capture_fps first")
        if not (cfg.calibration.line_a and cfg.calibration.line_b):
            raise HTTPException(400, "Draw measurement lines A and B first")
        n = frame if frame is not None else cfg.calibration.reference_frame
        return jobs.submit(pid, "preview", d, {"frame": n, "seconds": seconds}).public()

    @app.post("/api/projects/{pid}/recompute")
    def recompute(pid: str):
        d = _dir(pid)
        if not (d / "tracks.parquet").exists():
            raise HTTPException(400, "run analysis first")
        from ..pipeline import recompute as _recompute

        _recompute(store.config(pid), d)
        store.refresh(pid)
        return get_results(pid)

    @app.get("/api/projects/{pid}/results")
    def get_results(pid: str):
        d = _dir(pid)
        p = d / "results.json"
        if not p.exists():
            raise HTTPException(404, "no results yet")
        data = json.loads(p.read_text(encoding="utf-8"))
        r = d / "render.json"
        data["render"] = json.loads(r.read_text(encoding="utf-8")) if (d / "render.mp4").exists() and r.exists() else None
        return data

    @app.get("/api/projects/{pid}/overlay")
    def get_overlay(pid: str):
        d = _dir(pid)
        if not (d / "tracks.parquet").exists():
            raise HTTPException(404, "run analysis first")
        return store.overlay(pid)

    @app.post("/api/projects/{pid}/render")
    def start_render(pid: str, body: Optional[dict[str, Any]] = Body(None)):
        d = _dir(pid)
        if not (d / "tracks.parquet").exists():
            raise HTTPException(400, "run analysis first")
        if body:
            cfg = store.config(pid)
            try:
                cfg.render = cfg.render.model_validate({**cfg.render.model_dump(), **body})
            except ValueError as e:
                raise HTTPException(422, str(e))
            store.save_config(pid, cfg)
        return jobs.submit(pid, "render", d).public()

    @app.get("/api/projects/{pid}/files/{name}")
    def get_file(pid: str, name: str):
        if name not in OUTPUT_FILES:
            raise HTTPException(404)
        p = _dir(pid) / name
        if not p.exists():
            raise HTTPException(404, f"{name} not available")
        dl = name not in ("proxy.mp4",)
        return FileResponse(p, filename=f"{pid}-{name}" if dl else None)

    # -- jobs ----------------------------------------------------------------------
    def _job(jid: str):
        job = jobs.jobs.get(jid)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/api/jobs")
    def list_jobs():
        return [j.public() for j in sorted(jobs.jobs.values(), key=lambda j: j.created, reverse=True)]

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return _job(jid).public()

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        _job(jid)
        return jobs.cancel(jid).public()

    @app.get("/api/jobs/{jid}/events")
    async def job_events(jid: str, request: Request):
        job = _job(jid)

        def sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        async def gen():
            last_seq, last_prev, log_idx, idle = -1, 0, 0, 0
            while True:
                if await request.is_disconnected():
                    return
                idle += 1
                if job.seq != last_seq:
                    idle = 0
                    last_seq = job.seq
                    for entry in job.log[log_idx:]:
                        yield sse("log", entry)
                    log_idx = len(job.log)
                    yield sse("progress", {"status": job.status, "started": job.started, **job.progress})
                    if job.preview_seq != last_prev and job.preview:
                        last_prev = job.preview_seq
                        yield sse("preview", {"jpeg": job.preview})
                    if job.status in ("done", "error", "cancelled"):
                        yield sse("end", job.public() | {"traceback": job.traceback})
                        return
                elif idle % 40 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- web UI ------------------------------------------------------------------
    if (STATIC / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def no_ui():
            return ("<h1>speedcam API is running</h1><p>The web UI is not built. Run "
                    "<code>npm install && npm run build</code> in <code>web/</code>. API docs: "
                    "<a href='/docs'>/docs</a></p>")

    return app


DEFAULT_DATA_DIR = Path.home() / "speedcam-projects"
app = create_app(Path(os.environ.get("SPEEDCAM_DATA_DIR", DEFAULT_DATA_DIR)))
