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
    assert p["config"]["detection"]["model"] == "yolo11x.pt"  # partial update keeps other fields

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
