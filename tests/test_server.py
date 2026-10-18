import time

import pytest
from fastapi.testclient import TestClient

from speedcam.server.app import create_app
from speedcam.synth import make_synthetic


def _wait(client, job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.5)
    raise TimeoutError(job_id)


@pytest.mark.slow
def test_full_api_flow(tmp_path):
    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=70.0, size=(640, 360), m_per_px=0.1)
    client = TestClient(create_app(tmp_path / "projects"))

    p = client.post("/api/projects", json={"source": cfg.source}).json()
    pid = p["id"]
    assert p["info"]["nb_frames"] > 0
    assert _wait(client, p["proxy_job"])["status"] == "done"

    r = client.get(f"/api/projects/{pid}/proxy.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    img = client.get(f"/api/projects/{pid}/frame/10")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    p = client.patch(f"/api/projects/{pid}", json={"capture_fps": 120, "detection": {"type": "blob"}}).json()
    assert p["config"]["capture_fps"] == 120
    assert p["config"]["detection"]["model"] == "auto"  # partial update keeps other fields

    cal = client.put(f"/api/projects/{pid}/calibration", json=cfg.calibration.model_dump(mode="json")).json()
    assert cal["result"]["line_gap_m"] == pytest.approx(24.0)

    job = client.post(f"/api/projects/{pid}/analyze").json()
    end = _wait(client, job["id"])
    assert end["status"] == "done", end
    res = client.get(f"/api/projects/{pid}/results").json()
    assert res["stats"]["n_valid"] == 1
    assert res["vehicles"][0]["speed_kmh"] == pytest.approx(70.0, rel=0.005)

    ov = client.get(f"/api/projects/{pid}/overlay").json()
    assert len(ov["det_frame"]) == len(ov["det_tid"]) == len(ov["det_corners"]) // 8 > 0
    assert any(t["speed"] for t in ov["tracks"].values())

    pv = client.post(f"/api/projects/{pid}/preview?frame=0&seconds=2").json()
    pv_end = _wait(client, pv["id"])
    assert pv_end["status"] == "done", pv_end
    assert pv_end["result"]["tracks"] >= 1

    # SSE stream of a finished job replays its end event
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as s:
        body = "".join(s.iter_text())
    assert "event: end" in body

    job = client.post(f"/api/projects/{pid}/render", json={"output_fps": 30, "scale": 1.0}).json()
    assert _wait(client, job["id"])["status"] == "done"
    assert client.get(f"/api/projects/{pid}/results").json()["render"]["output_fps"] == 30
    assert client.get(f"/api/projects/{pid}/files/results.csv").status_code == 200
    assert client.get(f"/api/projects/{pid}/files/../project.json").status_code == 404


@pytest.mark.slow
def test_library_cache_and_history(tmp_path):
    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=60.0, size=(640, 360), m_per_px=0.1)
    data_dir = tmp_path / "projects"
    client = TestClient(create_app(data_dir))
    p = client.post("/api/projects", json={"source": cfg.source}).json()
    pid = p["id"]
    assert _wait(client, p["proxy_job"])["status"] == "done"

    # library row reflects folder state; opening records the screen to resume on
    lib = client.get("/api/projects").json()
    assert lib[0]["id"] == pid and lib[0]["has_proxy"] == 1 and lib[0]["source_exists"]
    client.post(f"/api/projects/{pid}/opened", json={"screen": "calibrate"})
    assert client.get("/api/projects").json()[0]["last_screen"] == "calibrate"

    # frames and thumbnails are served from the disk cache after the first decode
    assert client.get(f"/api/projects/{pid}/frame/5").status_code == 200
    assert (data_dir / pid / "cache" / "frames" / "5_full.jpg").exists()
    assert client.get(f"/api/projects/{pid}/thumb.jpg").headers["content-type"] == "image/jpeg"

    client.patch(f"/api/projects/{pid}", json={"capture_fps": 120, "name": "Renamed", "detection": {"type": "blob"},
                                               "stabilization": {"enabled": True}})
    client.put(f"/api/projects/{pid}/calibration", json=cfg.calibration.model_dump(mode="json"))
    end = _wait(client, client.post(f"/api/projects/{pid}/analyze").json()["id"])
    assert end["status"] == "done"
    row = client.get("/api/projects").json()[0]
    assert row["name"] == "Renamed" and row["n_valid"] == 1 and row["mean_kmh"] == pytest.approx(60, rel=0.01)
    drift = client.get(f"/api/projects/{pid}/results").json()["run"]["camera_drift"]
    assert drift["max_px"] > 0.5  # the synthetic clip sways

    # overlay is cached until its inputs change
    client.get(f"/api/projects/{pid}/overlay")
    assert (data_dir / pid / "cache" / "overlay.json").exists()

    # job history survives a restart, on a new app instance over the same folder
    client2 = TestClient(create_app(data_dir))
    kinds = [j["kind"] for j in client2.get(f"/api/projects/{pid}/jobs").json()]
    assert "analyze" in kinds and "proxy" in kinds

    # relink to a copy of the clip, then delete the project (the video stays)
    import shutil

    moved = tmp_path / "moved.mp4"
    shutil.copy(cfg.source, moved)
    r = client2.post(f"/api/projects/{pid}/relink", json={"source": str(moved)}).json()
    assert r["relink_warnings"] == [] and r["config"]["source"] == str(moved.resolve())
    assert client2.delete(f"/api/projects/{pid}").status_code == 200
    assert not (data_dir / pid).exists() and moved.exists()
    assert client2.get("/api/projects").json() == []


def test_index_rebuilds_from_folders(tmp_path):
    """Deleting speedcam.db loses nothing but job history: the library is rebuilt from project folders."""
    from speedcam.server.projects import ProjectStore

    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=40.0, size=(320, 180), m_per_px=0.2)
    store = ProjectStore(tmp_path / "projects")
    pid = store.create(cfg.source)["id"]
    store.db.close()
    for f in (tmp_path / "projects").glob("speedcam.db*"):
        f.unlink()
    store2 = ProjectStore(tmp_path / "projects")
    assert [r["id"] for r in store2.list()] == [pid]


@pytest.mark.slow
def test_parallel_slots_one_job_per_project(tmp_path):
    """Two projects run at the same time; two jobs of one project never overlap."""
    a = make_synthetic(tmp_path / "a.mp4", speed_kmh=50.0, size=(640, 360), m_per_px=0.1)
    b = make_synthetic(tmp_path / "b.mp4", speed_kmh=50.0, size=(640, 360), m_per_px=0.1, seed=1)
    app = create_app(tmp_path / "projects", job_slots=2)
    client = TestClient(app)
    pa = client.post("/api/projects", json={"source": a.source}).json()
    pb = client.post("/api/projects", json={"source": b.source}).json()
    second_a = client.post(f"/api/projects/{pa['id']}/proxy").json()  # same project as pa's proxy job

    seen_parallel, seen_same_project = False, False
    t0 = time.time()
    while time.time() - t0 < 180:
        active = client.get("/api/jobs/active").json()
        assert active["slots"] == 2
        running = [j for j in active["jobs"] if j["status"] == "running"]
        projects = [j["project_id"] for j in running]
        seen_parallel |= len(set(projects)) == 2
        seen_same_project |= len(projects) != len(set(projects))
        if not active["jobs"]:
            break
        time.sleep(0.1)
    assert seen_parallel, "jobs of different projects should run at the same time"
    assert not seen_same_project, "two jobs of one project must not run at once"
    for jid in (pa["proxy_job"], pb["proxy_job"], second_a["id"]):
        assert client.get(f"/api/jobs/{jid}").json()["status"] == "done"


def test_scheduler_one_detection_job_on_gpu():
    """Only `gpu_slots` detection jobs run at once; proxy/render jobs may run alongside them."""
    import threading
    from collections import deque

    from speedcam.server.jobs import Job, JobManager

    m = JobManager.__new__(JobManager)  # scheduler state only, no worker threads
    m.slots, m.gpu_slots, m.cv = 2, 1, threading.Condition()
    m.running = {"r": Job("r", "p1", "analyze", ".")}
    m.pending = deque([Job("a", "p2", "analyze", "."), Job("b", "p1", "render", "."), Job("c", "p3", "proxy", ".")])
    assert m._next().id == "c"  # p2's analysis waits for the GPU, p1 is busy
    m.running = {}
    assert m._next().id == "a"
