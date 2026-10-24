import pytest
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
    v = _run(_cal(solid_lines=[line]), lane_change + along + past_end, solid_mode="centre")
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


def test_footprint_of_axis_aligned_box_on_a_diagonal():
    # a 4.5 x 1.8 m car (px == m here) heading 30 degrees: its axis-aligned box is ~4.8 x 3.8
    L, w, a = 4.5, 1.8, np.radians(30)
    h, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    car = np.array([h * L / 2 + n * w / 2, h * L / 2 - n * w / 2, -h * L / 2 - n * w / 2, -h * L / 2 + n * w / 2])
    x0, y0 = car.min(0)
    x1, y1 = car.max(0)
    aabb = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    fp = violations.footprint(aabb, h)
    sides = sorted(np.linalg.norm(fp - np.roll(fp, 1, axis=0), axis=1))
    assert sides[0] == pytest.approx(w, abs=0.05) and sides[-1] == pytest.approx(L, abs=0.05)
    assert np.allclose(sorted(map(tuple, np.round(fp, 3))), sorted(map(tuple, np.round(car, 3))))
    # near 45 degrees (singular) it falls back to a typical car shape that fits the box
    b = (L + w) / np.sqrt(2) / 2
    sq = np.array([[-b, -b], [b, -b], [b, b], [-b, b]])
    fp45 = violations.footprint(sq, np.array([1.0, 1.0]))
    assert np.all(fp45.min(0) >= -b - 1e-6) and np.all(fp45.max(0) <= b + 1e-6)
    sides = sorted(np.linalg.norm(fp45 - np.roll(fp45, 1, axis=0), axis=1))
    assert sides[-1] / sides[0] == pytest.approx(violations.ASPECT, rel=0.01)
    assert violations.footprint(car, h) is car  # an oriented box is kept


def test_footprint_on_the_class_width_survives_a_noisy_box():
    """The solved width divides by cos² - sin²: at 30 degrees a box a few % off (or a heading a
    few degrees off) made a 1.8 m car 0.5 m wide on 0491. On the class width it stays a car."""
    L, w, a = 4.5, 1.8, np.radians(30)
    h = np.array([np.cos(a), np.sin(a)])
    W, H = L * np.cos(a) + w * np.sin(a), L * np.sin(a) + w * np.cos(a)
    noisy = np.array([[0, 0], [W * 0.97, 0], [W * 0.97, H * 1.06], [0, H * 1.06]])
    solved = violations.footprint(noisy, h)
    sides = sorted(np.linalg.norm(solved - np.roll(solved, 1, axis=0), axis=1))
    assert abs(sides[0] - w) > 0.25 * w  # the unstable solve: a 3-6 % box error, a 30 % width error
    fp = violations.footprint(noisy, h, width_px=w)
    sides = sorted(np.linalg.norm(fp - np.roll(fp, 1, axis=0), axis=1))
    assert sides[0] == pytest.approx(w) and sides[-1] == pytest.approx(L, rel=0.08)
    exact = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    sides = sorted(np.linalg.norm(violations.footprint(exact, h, w) - np.roll(violations.footprint(exact, h, w), 1, axis=0), axis=1))
    assert sides[-1] == pytest.approx(L)


def test_diagonal_car_beside_a_line_isnt_touching_it():
    """Axis-aligned boxes of a car driving diagonally next to a line overlap it; the car doesn't."""
    line = SolidLine(name="Edge", points=[(0, 60), (2000, 1215)])  # 30 degrees, like the car
    t = np.arange(int(4 * FPS)) / FPS
    d = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    nrm = np.array([-d[1], d[0]])
    centre = np.array([200.0, 60 + 200 * np.tan(np.radians(30))]) - nrm * 40 + np.outer(200 * t, d)  # 2 m off the line (1.1 m clear)

    def rows(tid, pts, L=90, W=36):
        out = []
        for f, c in enumerate(pts):
            car = np.array([c + d * L / 2 + nrm * W / 2, c + d * L / 2 - nrm * W / 2, c - d * L / 2 - nrm * W / 2, c - d * L / 2 + nrm * W / 2])
            x0, y0 = car.min(0)
            x1, y1 = car.max(0)
            box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # what a COCO-style detector gives
            out.append({"frame": f, "track_id": tid, "cls": 0, "conf": 0.9, "rx": c[0], "ry": c[1],
                        "x_m": c[0] * M_PER_PX, "y_m": c[1] * M_PER_PX,
                        **{f"c{k}{a}": box[k][j] for k in range(4) for j, a in enumerate("xy")}})
        return out

    clear = rows(1, centre)
    on_line = rows(2, centre + nrm * 30)  # centre 0.5 m off: its side is 0.4 m over the line
    v = _run(_cal(solid_lines=[line]), clear + on_line)
    got = {e["track_id"]: e for e in v["solid_line"]}
    assert set(got) == {2} and got[2]["how"] == "straddled"
    assert got[2]["max_over_m"] == pytest.approx(0.4, abs=0.1) and got[2]["t_end_s"] > got[2]["t_s"] + 3
