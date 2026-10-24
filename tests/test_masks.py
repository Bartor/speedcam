"""Solid lines judged by segmentation masks (masks.py, violations.mask_requests / detect)."""
import numpy as np
import pandas as pd
import pytest

from speedcam import calibration, masks, violations
from speedcam.config import ProjectConfig, SolidLine
from speedcam.speed import compute_speeds

from test_violations import FPS, _cal, _rows

PPM = 20.0  # px per metre in these tests (M_PER_PX = 0.05)


def _setup(rows, **vs):
    cal = _cal(solid_lines=[SolidLine(name="Edge", points=[(0, 300), (2000, 300)])])
    cfg = ProjectConfig(source="x.mp4", calibration=cal)
    cfg.violations = cfg.violations.model_copy(update=vs)
    df = pd.DataFrame(rows)
    res = calibration.compute(cal)
    _, renders = compute_speeds(df, FPS, res.line_a_world, res.line_b_world, cfg.speed, paths=res.path_geoms)
    return cfg, res, df, renders


def _detect(cfg, res, df, renders, store):
    return violations.detect(cfg, res, df, renders, FPS, None, np.arange(df["frame"].max() + 1), store)


def _along(tid=1, y=272.0, n=int(4 * FPS)):
    t = np.arange(n) / FPS
    return _rows(tid, 200 + 200 * t, np.full_like(t, y))  # 10 m/s along a line 1.4 m below it


def _mask_store(requests, bottom_of):
    """A rectangular 'mask' per requested detection, its lower edge at bottom_of(frame) (px)."""
    out = {}
    for key, (_, f, corners) in requests.items():
        x0, y0 = np.asarray(corners).reshape(4, 2).min(0)
        x1, _ = np.asarray(corners).reshape(4, 2).max(0)
        b = bottom_of(f)
        out[key] = [[x0, y0], [x1, y0], [x1, b], [x0, b]]
    return out


def test_only_vehicles_near_a_line_are_outlined():
    near, far = _along(1, y=272.0), _along(2, y=150.0)  # box edges 0.4 m and 6.5 m from the line
    cfg, res, df, renders = _setup(near + far)
    req = violations.mask_requests(cfg, res, df, renders, FPS)
    assert {tid for tid, _, _ in req.values()} == {1}
    assert all(f % 2 == 0 for _, f, _ in req.values())  # every 2nd frame
    cfg.violations.mask_gate_m = 0.2
    assert violations.mask_requests(cfg, res, df, renders, FPS) == {}  # the near one is 0.4 m clear
    cfg.violations.solid_masks = False
    cfg.violations.mask_gate_m = 0.5
    assert violations.mask_requests(cfg, res, df, renders, FPS) == {}


def test_masks_decide_where_the_box_cannot():
    cfg, res, df, renders = _setup(_along())
    req = violations.mask_requests(cfg, res, df, renders, FPS)

    # without masks: the box estimate is 0.4 m clear, nothing reported, but every request is missing
    v = _detect(cfg, res, df, renders, {})
    assert v["solid_line"] == [] and v["masks_missing"] == len(req)
    [c] = v["solid_cases"]
    assert c["source"] == "outline" and c["max_over_m"] == 0

    # its masks reach 0.4 m over the line for frames 30-70 (1.3 s): reported, measured by masks
    over = _mask_store(req, lambda f: 308.0 if 30 <= f <= 70 else 296.0)
    v = _detect(cfg, res, df, renders, over)
    [e] = v["solid_line"]
    assert e["how"] == "straddled" and e["measured_by"] == "mask" and e["max_over_m"] == pytest.approx(0.4, abs=0.01)
    assert e["t_s"] == pytest.approx(1.0, abs=0.1) and e["duration_s"] == pytest.approx(1.4, abs=0.15)
    [c] = v["solid_cases"]
    assert c["source"] == "mask" and c["reported"] and v["masks_missing"] == 0

    # masks clear of the line: not reported, even though the box estimate came close
    v = _detect(cfg, res, df, renders, _mask_store(req, lambda f: 296.0))
    assert v["solid_line"] == [] and v["solid_cases"][0]["max_over_m"] == 0


def test_a_single_glitched_mask_is_not_a_violation():
    cfg, res, df, renders = _setup(_along())
    req = violations.mask_requests(cfg, res, df, renders, FPS)
    glitch = min(f for _, f, _ in req.values()) + 40  # one mask bleeds 2 m over the line
    v = _detect(cfg, res, df, renders, _mask_store(req, lambda f: 340.0 if f == glitch else 296.0))
    assert v["solid_line"] == []
    assert v["solid_cases"][0]["max_over_m"] == 0  # the median of 7 masks drops it


def test_mask_over_ignores_points_past_the_line_ends():
    line = np.array([[0.0, 300.0], [100.0, 300.0]])
    inside = np.array([[10, 290], [50, 290], [50, 310], [10, 310]], float)  # 10 px over, beside the line
    past = np.array([[150, 290], [190, 290], [190, 400], [150, 400]], float)  # past its end
    assert violations.mask_over(inside, line, +1.0) == pytest.approx(10.0)
    assert violations.mask_over(past, line, +1.0) == -np.inf
    assert violations.mask_over(np.zeros((0, 2)), line, +1.0) == -np.inf


def test_sample_segments_only_new_detections_and_stores_outlines(tmp_path, monkeypatch):
    """masks.sample with a fake segmenter and frames: outlines land in masks.json in reference px."""
    cfg, res, df, renders = _setup(_along())
    req = violations.mask_requests(cfg, res, df, renders, FPS)
    calls = []

    class FakeGrabber:
        def __init__(self, path):
            pass

        def get(self, n):
            return np.zeros((720, 2400, 3), np.uint8)

        def close(self):
            pass

    def segment(crops):
        calls.append(len(crops))
        out = []
        for c in crops:
            h, w = c.shape[:2]
            m = np.zeros((h, w), np.float32)
            m[60:h - 60, 60:w - 60] = 1.0  # the vehicle fills its box
            out.append([(m, np.array([60, 60, w - 60, h - 60], float), 2)])
        return out

    import speedcam.video as video
    monkeypatch.setattr(video, "FrameGrabber", FakeGrabber)
    tf = {"frames": np.arange(200), "T": np.repeat(np.eye(3)[None], 200, axis=0)}
    got = masks.sample(cfg, tmp_path, req, tf, segment=segment)
    assert got["new"] == len(req) and sum(calls) == len(req)
    stored = masks.load(tmp_path, cfg.violations.mask_model)
    key = next(iter(req))
    pts = np.array(stored[key])
    box = np.asarray(req[key][2]).reshape(4, 2)
    assert np.allclose(pts.min(0), box.min(0), atol=1.5) and np.allclose(pts.max(0), box.max(0), atol=1.5)
    # nothing new to do the second time; another model's outlines don't count
    assert masks.sample(cfg, tmp_path, req, tf, segment=segment)["new"] == 0
    assert masks.load(tmp_path, "other-seg.pt") == {}
