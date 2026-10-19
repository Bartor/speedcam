import numpy as np
import pandas as pd

from speedcam import calibration, violations
from speedcam.config import (Calibration, ProjectConfig, RedLightSetup, Segment, SolidLine, StopLine,
                             TrafficLight)
from speedcam.speed import compute_speeds

FPS = 30.0
M_PER_PX = 0.05  # 100 px = 5 m


def _rows(tid, xs, ys, length_px=80, width_px=40, frames=None):
    """Track rows (reference px == world / M_PER_PX) with an axis-aligned box along +x."""
    n = len(xs)
    frames = np.arange(n) if frames is None else frames
    rows = []
    for f, x, y in zip(frames, xs, ys):
        c = [(x - length_px / 2, y - width_px / 2), (x + length_px / 2, y - width_px / 2),
             (x + length_px / 2, y + width_px / 2), (x - length_px / 2, y + width_px / 2)]
        rows.append({"frame": int(f), "track_id": tid, "cls": 2, "conf": 0.9, "rx": x, "ry": y,
                     "x_m": x * M_PER_PX, "y_m": y * M_PER_PX,
                     **{f"c{k}{a}": c[k][j] for k in range(4) for j, a in enumerate("xy")}})
    return rows


def _run(cal: Calibration, rows, samples=None, **vs):
    cfg = ProjectConfig(source="x.mp4", calibration=cal)
    cfg.violations = cfg.violations.model_copy(update=vs)
    df = pd.DataFrame(rows)
    res = calibration.compute(cal)
    _, renders = compute_speeds(df, FPS, res.line_a_world, res.line_b_world, cfg.speed, paths=res.path_geoms)
    frames = np.arange(df["frame"].max() + 1)
    return violations.detect(cfg, res, df, renders, FPS, samples, frames)


def _cal(**kw):
    return Calibration(segments=[Segment(a=(0, 0), b=(100, 0), meters=5.0)], **kw)


def _car_through_stop_line(tid=1, y=200.0):
    # 10 m/s = 200 px/s from x=600 to 1400; centre on the line (x=1000) at t=2.0 s, front at 1.8 s
    t = np.arange(int(4 * FPS)) / FPS
    return _rows(tid, 600 + 200 * t, np.full_like(t, y))


def test_red_light_manual_phases_direction_and_grace():
    stop = StopLine(points=[(1000, 0), (1000, 400)])  # drawn downwards: +x traffic goes right -> left side
    setup = RedLightSetup(name="Main", stop_lines=[stop], state_source="manual", red_intervals=[(1.5, 3.0)])
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line())
    [e] = v["red_light"]
    assert e["violation"] is True and e["light"] == "red"
    assert abs(e["t_centre_s"] - 2.0) < 0.05
    assert abs(e["t_s"] - 1.8) < 0.05  # the front of the box reached the line first
    assert abs(e["phase_for_s"] - 0.3) < 0.05
    assert abs(e["speed_kmh"] - 36.0) < 0.5
    assert v["n_red_light"] == 1

    # within the grace period after the change to red: not reported
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line(), red_grace_s=0.5)
    assert v["red_light"][0]["violation"] is False

    # light turns red after the vehicle entered
    late = setup.model_copy(update={"red_intervals": [(2.5, 5.0)]})
    v = _run(_cal(red_lights=[late]), _car_through_stop_line())
    assert v["red_light"][0]["violation"] is False and v["red_light"][0]["light"] == "not red"

    # only the opposite direction counts: the crossing is ignored
    fwd = setup.model_copy(update={"stop_lines": [StopLine(points=stop.points, direction="forward")]})
    assert _run(_cal(red_lights=[fwd]), _car_through_stop_line())["red_light"] == []
    rev = setup.model_copy(update={"stop_lines": [StopLine(points=stop.points, direction="reverse")]})
    assert len(_run(_cal(red_lights=[rev]), _car_through_stop_line())["red_light"]) == 1

    # passing beside the end of the stop line isn't crossing it
    assert _run(_cal(red_lights=[setup]), _car_through_stop_line(y=600))["red_light"] == []


def test_red_light_phase_from_colour_samples():
    n = int(4 * FPS)
    counts = np.zeros((n, 1, 3), np.int32)
    counts[: int(1.0 * FPS), 0, 2] = 20  # green
    counts[int(1.0 * FPS): int(1.5 * FPS), 0, 1] = 20  # amber
    counts[int(1.5 * FPS):, 0, 0] = 20  # red
    counts[int(1.6 * FPS): int(1.7 * FPS), 0, :] = 0  # LED flicker: dark frames
    counts[int(2.5 * FPS), 0, :] = [0, 0, 30]  # a single misread frame
    setup = RedLightSetup(name="Main", lights=[TrafficLight(px=(50, 50))],
                          stop_lines=[StopLine(points=[(1000, 0), (1000, 400)])])
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line(), samples=(np.arange(n), counts))
    phases = v["lights"][0]["phases"]
    assert [p["state"] for p in phases] == ["green", "amber", "red"]
    assert abs(phases[2]["t0"] - 1.5) < 0.05
    [e] = v["red_light"]
    assert e["violation"] is True and abs(e["phase_for_s"] - 0.3) < 0.05

    # amber counted as red when asked
    counts[int(1.0 * FPS):, 0, :] = [0, 20, 0]
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line(), samples=(np.arange(n), counts))
    assert v["red_light"][0]["light"] == "amber" and v["red_light"][0]["violation"] is False
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line(), samples=(np.arange(n), counts), amber_is_red=True)
    assert v["red_light"][0]["violation"] is True

    # lights moved since the analysis: phase unknown, not a violation either way
    v = _run(_cal(red_lights=[setup]), _car_through_stop_line(), samples=None)
    assert v["red_light"][0]["violation"] is None and v["warnings"]


def test_colour_counts_reads_a_lit_lamp():
    img = np.zeros((60, 60, 3), np.uint8)
    import cv2
    cv2.circle(img, (30, 30), 5, (40, 40, 255), -1)  # red lamp (BGR)
    vs = ProjectConfig(source="x").violations
    c = violations.colour_counts(img, (30, 30), 8, vs)
    assert c[0] > 50 and c[1] == 0 and c[2] == 0
    cv2.circle(img, (30, 30), 5, (180, 255, 60), -1)  # bluish traffic green
    c = violations.colour_counts(img, (30, 30), 8, vs)
    assert c[2] > 50 and c[0] == 0
    assert violations.colour_counts(img, (5, 5), 3, vs).sum() == 0  # dark housing


def test_solid_line_centre_crossing_with_hysteresis():
    line = SolidLine(name="Centre line", points=[(0, 200), (2000, 200)])
    t = np.arange(int(4 * FPS)) / FPS
    x = 200 + 200 * t
    lane_change = _rows(1, x, 170 + 60 / (1 + np.exp(-(t - 2) * 4)))  # 1.5 m to the other lane
    along = _rows(2, x, 200 + 8 * np.sin(t * 9))  # rides on the line, +-0.4 m wobble
    past_end = _rows(3, 2200 + 200 * t, 170 + 60 / (1 + np.exp(-(t - 2) * 4)))  # beyond its end
    v = _run(_cal(solid_lines=[line]), lane_change + along + past_end)
    assert [(e["track_id"], e["how"]) for e in v["solid_line"]] == [(1, "crossed")]
    assert abs(v["solid_line"][0]["t_s"] - 2.0) < 0.1

    # wheels: a car whose box overlaps the line by 0.7 m for the whole clip is straddling
    straddle = _rows(4, x, np.full_like(t, 194.0), width_px=40)  # centre 0.3 m off, half-width 1 m
    clear = _rows(5, x, np.full_like(t, 170.0), width_px=40)  # box edge 0.5 m short of the line
    v = _run(_cal(solid_lines=[line]), lane_change + straddle + clear, solid_mode="wheels")
    got = {e["track_id"]: e for e in v["solid_line"]}
    assert set(got) == {1, 4}
    assert got[1]["how"] == "crossed" and got[4]["how"] == "straddled"
    assert abs(got[4]["max_over_m"] - 0.7) < 0.05


def test_violations_only_project_needs_no_speed_lines():
    cal = _cal(solid_lines=[SolidLine(points=[(0, 200), (2000, 200)])])
    assert calibration.has_measurement_geometry(cal) and not calibration.has_speed_geometry(cal)
    t = np.arange(int(4 * FPS)) / FPS
    v = _run(cal, _rows(1, 200 + 200 * t, 170 + 60 / (1 + np.exp(-(t - 2) * 4))))
    assert len(v["solid_line"]) == 1
