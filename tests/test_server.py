import json
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

    # one project per video: the same file (any spelling of its path) opens the existing project
    import os
    again = client.post("/api/projects", json={"source": os.path.normcase(cfg.source).upper()}).json()
    assert again["id"] == pid and again["existing"]
    assert len(client.get("/api/projects").json()) == 1
    listing = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    entry = next(e for e in listing["entries"] if e["name"] == "syn.mp4")
    assert entry["projects"] == [{"id": pid, "name": p["name"]}]

    r = client.get(f"/api/projects/{pid}/proxy.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    strip = client.get(f"/api/projects/{pid}/filmstrip.jpg?count=12&height=40")  # timeline thumbnails
    assert strip.status_code == 200 and strip.headers["content-type"] == "image/jpeg"
    import cv2
    import numpy as np
    s = cv2.imdecode(np.frombuffer(strip.content, np.uint8), 1)
    assert s.shape[0] == 40 and s.shape[1] == 12 * (s.shape[1] // 12)

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
    veh = res["vehicles"][0]
    assert 0 < veh["speed_avg_u_kmh"] < 0.05 * 70 and veh["speed_max_u_kmh"] > 0
    assert res["uncertainty"]["mc_samples"] > 100 and res["uncertainty"]["parallax"] == "unknown"

    # validation run: a reference speed per vehicle, stored with the project
    assert client.get(f"/api/projects/{pid}/validation").json()["report"]["summary"]["n_matched"] == 0
    val = client.put(f"/api/projects/{pid}/validation", json={"references": [
        {"speed_kmh": 70.0, "t_s": (veh["t_entry_s"] + veh["t_exit_s"]) / 2, "source": "synthetic"}]}).json()
    row = val["report"]["rows"][0]
    assert row["track_id"] == veh["track_id"] and row["inside"]
    assert client.get(f"/api/projects/{pid}/validation").json()["references"][0]["source"] == "synthetic"
    assert client.get(f"/api/projects/{pid}/files/validation.json").status_code == 200

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

    # jobs history: every job newest first, finished ones with their own copies of their outputs
    rc = client.post(f"/api/projects/{pid}/recompute")
    assert rc.status_code == 200
    for _ in range(20):  # copies are made just after a job reports done
        hist = client.get("/api/jobs/history").json()
        by_kind = {}
        for r in hist["jobs"]:
            by_kind.setdefault(r["kind"], r)
        if all(by_kind.get(k, {}).get("files") for k in ("analyze", "render", "recompute")):
            break
        time.sleep(0.3)
    created = [r["created"] for r in hist["jobs"]]
    assert created == sorted(created, reverse=True)
    assert {"proxy", "analyze", "preview", "render", "recompute"} <= set(by_kind)
    names = {k: {f["name"] for f in r["files"]} for k, r in by_kind.items()}
    assert {"results.json", "results.csv", "tracks.parquet", "settings.json"} <= names["analyze"]
    assert {"render.mp4", "render.json"} <= names["render"] and names["preview"] == set()
    assert by_kind["recompute"]["status"] == "done" and by_kind["recompute"]["result"]["n_valid"] == 1
    assert hist["artifacts_bytes"] > 0 and by_kind["analyze"]["project_name"]
    aj = by_kind["analyze"]["id"]
    f = client.get(f"/api/jobs/{aj}/files/results.json")
    assert f.status_code == 200 and json.loads(f.content)["vehicles"][0]["speed_kmh"] == pytest.approx(70.0, rel=0.005)
    assert "analyze" in f.headers["content-disposition"]
    assert client.get(f"/api/jobs/{aj}/files/..%2Fproject.json").status_code == 404
    assert client.delete(f"/api/jobs/{aj}/files").status_code == 200
    assert client.get(f"/api/jobs/{aj}/files/results.json").status_code == 404
    assert next(r for r in client.get("/api/jobs/history").json()["jobs"] if r["id"] == aj)["files"] == []


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


@pytest.mark.slow
def test_tracking_only_then_calibrate_and_recompute(tmp_path):
    """No calibration and nothing to measure: the analysis still runs (detection + tracking only);
    calibrating afterwards and recomputing gives the speeds without analysing again."""
    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=70.0, size=(640, 360), m_per_px=0.1)
    client = TestClient(create_app(tmp_path / "projects"))
    p = client.post("/api/projects", json={"source": cfg.source}).json()
    pid = p["id"]
    assert _wait(client, p["proxy_job"])["status"] == "done"
    client.patch(f"/api/projects/{pid}", json={"capture_fps": 120, "detection": {"type": "blob"}})
    job = client.post(f"/api/projects/{pid}/analyze").json()
    end = _wait(client, job["id"])
    assert end["status"] == "done", end
    res = client.get(f"/api/projects/{pid}/results").json()
    assert res["vehicles"] == [] and res["calibration"]["method"] == "none" and res["run"]["tracks"] >= 1
    assert client.get(f"/api/projects/{pid}/overlay").status_code == 200

    client.put(f"/api/projects/{pid}/calibration", json=cfg.calibration.model_dump(mode="json"))
    res = client.post(f"/api/projects/{pid}/recompute").json()
    assert res["stats_max"]["n_valid"] == 1
    assert res["vehicles"][0]["speed_kmh"] == pytest.approx(70.0, rel=0.005)

    # Review says what changed in the setup since: a Recompute applies speed checks, a new
    # analysis the detector; the speed limit is only how results are shown
    assert res["stale"] == {"recompute": False, "analysis": False}
    client.patch(f"/api/projects/{pid}", json={"render": {"speed_limit_kmh": 30}})
    assert client.get(f"/api/projects/{pid}/results").json()["stale"] == {"recompute": False, "analysis": False}
    client.patch(f"/api/projects/{pid}", json={"speed": {"max_gap_s": 0.4}})
    assert client.get(f"/api/projects/{pid}/results").json()["stale"] == {"recompute": True, "analysis": False}
    client.patch(f"/api/projects/{pid}", json={"detection": {"conf": 0.3}})
    assert client.get(f"/api/projects/{pid}/results").json()["stale"]["analysis"] is True


@pytest.mark.slow
def test_vehicle_names_merges_looks_and_render_preview(tmp_path):
    """A car the tracker lost for a moment (two ids) is found as a likely split, merged into one
    measured vehicle (max intact, average inconclusive across the gap), named, pictured, and
    split again. The Export preview draws one frame with unsaved settings."""
    import cv2
    import numpy as np
    import pandas as pd

    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=70.0, size=(640, 360), m_per_px=0.1)
    client = TestClient(create_app(tmp_path / "projects"))
    p = client.post("/api/projects", json={"source": cfg.source}).json()
    pid = p["id"]
    assert _wait(client, p["proxy_job"])["status"] == "done"
    client.patch(f"/api/projects/{pid}", json={"capture_fps": 120, "detection": {"type": "blob"}})
    client.put(f"/api/projects/{pid}/calibration", json=cfg.calibration.model_dump(mode="json"))
    assert _wait(client, client.post(f"/api/projects/{pid}/analyze").json()["id"])["status"] == "done"
    tid = client.get(f"/api/projects/{pid}/results").json()["vehicles"][0]["track_id"]

    # break the track: 0.5 s missing (longer than speed.max_gap_s) early on, a new id after it. (The
    # synthetic section is only 1.2 s long: a gap in its middle would leave too little clean
    # tracking for a max, which is right, but not what this test is about.)
    d = tmp_path / "projects" / pid
    df = pd.read_parquet(d / "tracks.parquet")
    f = df.loc[df["track_id"] == tid, "frame"]
    mid = int(f.quantile(0.3))
    df = df[~((df["track_id"] == tid) & df["frame"].between(mid, mid + 59))].copy()
    df.loc[(df["track_id"] == tid) & (df["frame"] > mid), "track_id"] = 900
    df.to_parquet(d / "tracks.parquet", index=False)
    res = client.post(f"/api/projects/{pid}/recompute").json()
    assert not any(v["valid"] and v["track_id"] in (tid, 900) for v in res["vehicles"])

    cand = client.get(f"/api/projects/{pid}/vehicles/candidates").json()
    pair = next(c for c in cand["pairs"] if c["likely"])
    assert (pair["a"], pair["b"]) == (tid, 900) and pair["gap_s"] == pytest.approx(61 / 120, abs=0.01)
    assert cand["tracks"][str(tid)]["cls"] == "car"

    client.put(f"/api/projects/{pid}/vehicles/900", json={"name": "  red   test car "})
    res = client.post(f"/api/projects/{pid}/vehicles/merge", json={"ids": [900, tid]}).json()
    v = next(v for v in res["vehicles"] if v["track_id"] == tid)  # under the id seen first
    assert v["merged"] == [tid, 900] and v["name"] == "red test car"  # the name moves with it
    assert v["valid"] and v["speed_max_kmh"] == pytest.approx(70.0, rel=0.02)
    assert v["avg_valid"] is False and "gap" in v["flags"]  # nothing measured across the gap
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["config"]["vehicles"] == {"names": {str(tid): "red test car"}, "merges": [[tid, 900]]}
    assert "red test car" in client.get(f"/api/projects/{pid}/files/results.csv").text

    client.put(f"/api/projects/{pid}/vehicles/{tid}", json={"name": "blue van"})
    assert client.get(f"/api/projects/{pid}/results").json()["vehicles"][0]["name"] == "blue van"
    assert "blue van" in client.get(f"/api/projects/{pid}/files/results.csv").text

    look = client.get(f"/api/projects/{pid}/vehicles/{tid}/look").json()
    assert look["colour"] and look["suggestion"].endswith(" car")
    thumb = client.get(f"/api/projects/{pid}/vehicles/{tid}/thumb.jpg")
    assert thumb.status_code == 200 and cv2.imdecode(np.frombuffer(thumb.content, np.uint8), 1).size > 0

    frame = int((v["t_entry_s"] + 0.1) * 120)
    pv = client.post(f"/api/projects/{pid}/render/preview?frame={frame}", json={"panel": "none", "scale": 0.5})
    assert pv.status_code == 200 and pv.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(pv.content, np.uint8), 1)
    assert img.shape[:2] == (180, 320)
    wide = client.post(f"/api/projects/{pid}/render/preview?frame={frame}",
                       json={"panel": "right", "panel_layout": "extend", "scale": 0.5, "show_boxes": False})
    assert cv2.imdecode(np.frombuffer(wide.content, np.uint8), 1).shape[1] > 320
    assert client.get(f"/api/projects/{pid}").json()["config"]["render"]["panel"] == "right"  # not saved

    res = client.post(f"/api/projects/{pid}/vehicles/{tid}/split").json()
    assert not any(v["merged"] for v in res["vehicles"])
    assert client.get(f"/api/projects/{pid}").json()["config"]["vehicles"]["merges"] == []

    # a new analysis has new track ids: names and merges are dropped with the old ones
    client.post(f"/api/projects/{pid}/vehicles/merge", json={"ids": [tid, 900]})
    assert _wait(client, client.post(f"/api/projects/{pid}/analyze").json()["id"])["status"] == "done"
    for _ in range(20):
        if client.get(f"/api/projects/{pid}").json()["config"]["vehicles"] == {"names": {}, "merges": []}:
            break
        time.sleep(0.2)
    assert client.get(f"/api/projects/{pid}").json()["config"]["vehicles"] == {"names": {}, "merges": []}
    assert client.get(f"/api/projects/{pid}/results").json()["vehicles"][0]["name"] == ""


def test_ui_cache_headers(tmp_path):
    """index.html is always revalidated (it names the current app code); hashed assets are
    cached for good. Without this a browser kept showing an old UI after an update."""
    from speedcam.server.app import STATIC

    if not (STATIC / "index.html").exists():
        pytest.skip("web UI not built")
    client = TestClient(create_app(tmp_path / "projects"))
    r = client.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    asset = next((STATIC / "assets").glob("*.js")).name
    r = client.get(f"/assets/{asset}")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    # a revalidation of an unchanged index.html is a cheap 304
    r2 = client.get("/", headers={"If-None-Match": client.get("/").headers["etag"]})
    assert r2.status_code == 304
