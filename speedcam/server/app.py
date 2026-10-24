"""FastAPI app: REST + SSE API under /api, built web UI served at /."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
import uuid
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
from ..validation import Reference, evaluate
from . import artifacts
from .jobs import JobManager
from .projects import OUTPUT_FILES, ProjectStore

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
STATIC = Path(__file__).parent / "static"


class CreateProject(BaseModel):
    source: str
    name: Optional[str] = None


def create_app(data_dir: Path, job_slots: Optional[int] = None) -> FastAPI:
    store = ProjectStore(data_dir)
    store.db.mark_interrupted()

    def on_job_change(job) -> None:
        if job.status == "done":
            try:  # keep this job's outputs; later runs overwrite the project's copies
                artifacts.archive(Path(job.project_dir), job.id, job.kind)
            except OSError:
                pass
        store.db.record_job(job.public())
        if job.status in ("done", "error", "cancelled") and job.kind in ("proxy", "analyze", "render"):
            store.refresh(job.project_id)

    slots = job_slots or int(os.environ.get("SPEEDCAM_JOB_SLOTS", "2"))
    gpu_slots = int(os.environ.get("SPEEDCAM_GPU_JOBS", "1"))
    jobs = JobManager(on_change=on_job_change, slots=slots, gpu_slots=gpu_slots)

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
        d = _dir(pid)
        data = store.load(pid)
        cfg = ProjectConfig.model_validate(data["config"])
        if not data.get("stored_orientation") and Path(cfg.source).is_file():
            # Made before rotated clips were handled: its proxy may have been turned upright by
            # ffmpeg while calibration and analysis used the stored frames. Re-probe; if the clip
            # is flagged as rotated, rebuild the proxy as stored and show it with the view rotation
            # (tracks and calibration are in stored pixels already, so nothing else changes).
            from ..timing import probe

            info = probe(cfg.source)
            if info.display_rotation and cfg.rotation == 0:
                cfg.rotation = info.display_rotation
            store.save_config(pid, cfg, info=info.to_dict(), stored_orientation=True)
            if info.display_rotation and (d / "proxy.mp4").exists():
                (d / "proxy.mp4").unlink()
                jobs.submit(pid, "proxy", d)
            data = store.load(pid)
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
        used = store.by_source()  # videos that already have a project
        try:
            for e in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        entries.append({"name": e.name, "path": str(e), "dir": True})
                    elif e.suffix.lower() in VIDEO_EXT:
                        entries.append({"name": e.name, "path": str(e), "dir": False, "size": e.stat().st_size,
                                        "projects": used.get(store.source_key(e), [])})
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
        # one project per video: opening a video that already has one returns that project
        existing = store.by_source().get(store.source_key(body.source)) if body.source.strip() else None
        if existing:
            return {**_project(existing[0]["id"]), "existing": True}
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

    @app.get("/api/projects/{pid}/lights/{n}")
    def read_lights(pid: str, n: int):
        """What each red-light setup's lights show on frame n: to check the circles and thresholds."""
        import numpy as np

        from ..violations import LightSampler, light_list, state_name

        d = _dir(pid)
        cfg = store.config(pid)
        try:
            img = store.grabber(pid).get(n)
        except (IndexError, EOFError):
            raise HTTPException(404, f"frame {n} not available")
        T = None  # light circles are in reference px; follow the camera if the clip was stabilized
        if (d / "transforms.npz").exists():
            tf = np.load(d / "transforms.npz")
            k = np.nonzero(tf["frames"] == n)[0]
            if len(k):
                T = tf["T"][k[0]]
        counts = LightSampler(cfg.calibration, cfg.violations).sample(img, T)
        order = light_list(cfg.calibration)
        vs = cfg.violations
        out = []
        for s, setup in enumerate(cfg.calibration.red_lights):
            c = [counts[j].tolist() for j, (si, _) in enumerate(order) if si == s]
            total = np.sum(c, 0) if c else np.zeros(3)
            out.append({"name": setup.name, "counts": c,
                        "state": state_name(int(np.argmax(total))) if total.max() >= vs.light_min_pixels else "unknown"})
        return {"frame": n, "stabilized": T is not None, "setups": out}

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
        # no calibration or nothing to measure is fine: the clip is then only detected and tracked
        return jobs.submit(pid, "analyze", _dir(pid)).public()

    @app.post("/api/projects/{pid}/scan")
    def start_scan(pid: str):
        """Activity scan: a quick pass that finds the parts of the video with vehicles."""
        return jobs.submit(pid, "scan", _dir(pid)).public()

    @app.get("/api/projects/{pid}/scan")
    def get_scan(pid: str):
        """The scan's timeline (vehicles per sample) and the parts it suggests with the current
        padding and merging settings (see config.ScanSettings)."""
        from .. import scan as activity

        d = _dir(pid)
        sc = activity.load(d)
        if sc is None:
            return {"scan": None, "suggested": [], "parts": store.config(pid).parts}
        cfg = store.config(pid)
        return {"scan": sc, "suggested": activity.suggest_parts(sc, cfg.scan), "parts": cfg.parts}

    @app.post("/api/projects/{pid}/preview")
    def preview_run(pid: str, frame: Optional[int] = None, seconds: float = 3.0):
        """Detect + stabilize + track a few seconds and stream annotated previews, without touching results."""
        d = _dir(pid)
        cfg = store.config(pid)
        if not cfg.capture_fps:
            raise HTTPException(400, "Set capture_fps first")
        n = frame if frame is not None else cfg.calibration.reference_frame
        return jobs.submit(pid, "preview", d, {"frame": n, "seconds": seconds}).public()

    @app.post("/api/projects/{pid}/recompute")
    def recompute(pid: str):
        d = _dir(pid)
        if not (d / "tracks.parquet").exists():
            raise HTTPException(400, "run analysis first")
        from ..pipeline import recompute as _recompute

        # recorded like a job, so its results show (and stay downloadable) in the jobs history
        rec = {"id": uuid.uuid4().hex[:12], "project_id": pid, "kind": "recompute", "status": "running",
               "created": time.time(), "started": time.time()}
        try:
            summary = _recompute(store.config(pid), d)
        except Exception as e:
            store.db.record_job({**rec, "status": "error", "finished": time.time(), "error": f"{type(e).__name__}: {e}"})
            raise
        rec.update(status="done", finished=time.time(),
                   result={"n_vehicles": len(summary["vehicles"]),
                           "n_valid": sum(v["valid"] for v in summary["vehicles"])})
        artifacts.archive(d, rec["id"], "recompute")
        store.db.record_job(rec)
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

    # -- validation run: known reference speeds vs. the measured ones --------------------
    def _validation(pid: str, refs: Optional[list[Reference]] = None) -> dict:
        d = _dir(pid)
        p = d / "validation.json"
        stored = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if refs is None:
            refs = [Reference.model_validate(r) for r in stored.get("references", [])]
        res = d / "results.json"
        vehicles = json.loads(res.read_text(encoding="utf-8"))["vehicles"] if res.exists() else []
        data = {"references": [r.model_dump(mode="json") for r in refs], "updated": time.time(),
                "report": evaluate(refs, vehicles)}
        if refs or p.exists():
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(p)
        return data

    @app.get("/api/projects/{pid}/validation")
    def get_validation(pid: str):
        """Reference speeds of a validation run, with the report against the current results."""
        return _validation(pid)

    @app.put("/api/projects/{pid}/validation")
    def put_validation(pid: str, body: dict[str, Any] = Body(...)):
        try:
            refs = [Reference.model_validate(r) for r in body.get("references", [])]
        except ValueError as e:
            raise HTTPException(422, str(e))
        return _validation(pid, refs)

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

    @app.get("/api/jobs/active")
    def active_jobs():
        """Running and queued jobs across all projects, with progress (polled by the UI)."""
        names = {r["id"]: r["name"] for r in store.list()}
        return {"slots": jobs.slots, "gpu_slots": jobs.gpu_slots,
                "jobs": [j.brief() | {"project_name": names.get(j.project_id, j.project_id)} for j in jobs.active()]}

    @app.get("/api/jobs/history")
    def job_history(limit: int = 500):
        """Every job, newest first, across projects (the database keeps them across restarts),
        with live status for running ones and the files each finished job left."""
        names = {r["id"]: r["name"] for r in store.list()}
        live = {j.id: j for j in jobs.jobs.values()}
        out = []
        for r in store.db.all_jobs(limit):
            j = live.get(r["id"])
            if j is not None:
                r.update(status=j.status, started=j.started, finished=j.finished, error=j.error)
                r["progress"] = j.progress
            pdir = store.root / r["project_id"]
            r["project_name"] = names.get(r["project_id"])
            r["files"] = artifacts.listing(pdir, r["id"], r["kind"]) if pdir.is_dir() else []
            out.append(r)
        total = sum(f["size"] for r in out for f in r["files"])
        return {"jobs": out, "artifacts_bytes": total}

    def _job_record(jid: str) -> dict:
        r = store.db.job(jid)
        if not r:
            raise HTTPException(404, "job not found")
        return r

    @app.get("/api/jobs/{jid}/files/{name}")
    def job_file(jid: str, name: str):
        r = _job_record(jid)
        if name not in artifacts.ALLOWED:
            raise HTTPException(404)
        p = artifacts.job_dir(store.root / r["project_id"], jid) / name
        if not p.is_file():
            raise HTTPException(404, f"{name} not kept for this job")
        when = time.strftime("%Y%m%d-%H%M", time.localtime(r["created"] or 0))
        return FileResponse(p, filename=f"{r['project_id']}-{r['kind']}-{when}-{name}")

    @app.delete("/api/jobs/{jid}/files")
    def delete_job_files(jid: str):
        """Free the disk space a job's copies take; its record stays."""
        r = _job_record(jid)
        artifacts.delete(store.root / r["project_id"], jid)
        return {"deleted": True}

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
