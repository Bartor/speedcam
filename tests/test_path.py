"""Path of motion: measuring along a curve over whatever part of it each vehicle covers."""
import numpy as np
import pytest

from speedcam.config import MotionPath, SpeedSettings
from speedcam.path import PathGeom, catmull_rom
from speedcam.speed import analyze_track

FPS = 60.0
R0 = 60.0  # centreline radius of a left-hand bend, metres


def bend_path(corridor=6.0):
    ang = np.linspace(0, np.pi / 2, 6)
    return PathGeom.from_world_points(np.column_stack([R0 * np.cos(ang), R0 * np.sin(ang)]), corridor_m=corridor)


def drive_arc(R, v_kmh, th0, th1, noise=0.0, seed=0):
    """Positions of a car driving a circular lane of radius R from angle th0 to th1 at constant speed."""
    v = v_kmh / 3.6
    n = int(abs(th1 - th0) * R / v * FPS)
    th = th0 + np.sign(th1 - th0) * np.arange(n) * v / FPS / R
    xy = np.column_stack([R * np.cos(th), R * np.sin(th)])
    return np.arange(n), xy + np.random.default_rng(seed).normal(0, noise, xy.shape)


def test_catmull_rom_passes_through_points():
    P = np.array([[0, 0], [10, 2], [20, -1], [30, 4]], float)
    dense = catmull_rom(P)
    for p in P:
        assert np.min(np.linalg.norm(dense - p, axis=1)) < 1e-6


@pytest.mark.parametrize("lane", [-3.5, 0.0, 3.5])  # outer lane, centre, inner lane
def test_speed_along_a_bend_is_right_in_every_lane(lane):
    path = bend_path()
    frames, xy = drive_arc(R0 - lane, 50.0, -0.2, np.pi / 2 + 0.2, noise=0.03)
    r = analyze_track(1, frames, xy, FPS, None, None, SpeedSettings(), paths=[path]).result
    assert r["direction"] == "A->B"
    assert r["speed_kmh"] == pytest.approx(50.0, rel=0.005)
    assert r["speed_max_kmh"] == pytest.approx(50.0, abs=1.0)
    assert r["coverage_pct"] > 95


def test_partial_coverage_and_reverse_direction():
    path = bend_path()
    # enters halfway along the bend, drives the other way and leaves past the start
    frames, xy = drive_arc(R0 + 1.5, 70.0, np.pi / 4, -0.3)
    r = analyze_track(2, frames, xy, FPS, None, None, SpeedSettings(), paths=[path]).result
    assert r["direction"] == "B->A"
    assert r["speed_kmh"] == pytest.approx(70.0, rel=0.005)
    assert 40 < r["coverage_pct"] < 60


def test_vehicles_outside_the_corridor_or_too_short_are_not_measured():
    path = bend_path(corridor=6.0)
    frames, xy = drive_arc(R0 + 15, 50.0, -0.2, np.pi / 2)  # a parallel street, 15 m away
    assert analyze_track(3, frames, xy, FPS, None, None, SpeedSettings(), paths=[path]).result is None
    frames, xy = drive_arc(R0, 50.0, 0.3, 0.3 + 6 / R0)  # only 6 m of the path
    rend = analyze_track(4, frames, xy, FPS, None, None, SpeedSettings(), paths=[path])
    assert rend.result is None and rend.crossed == "short"


def test_lane_change_does_not_inflate_path_speed():
    path = PathGeom.from_world_points(np.array([[0.0, 0.0], [80.0, 0.0]]), corridor_m=6)
    v = 60 / 3.6
    t = np.arange(0, 80 / v, 1 / FPS)
    x = 2 + v * t
    y = 3.5 / (1 + np.exp(-(x - 40) / 3))  # one lane over, mid-path
    r = analyze_track(5, np.arange(len(t)), np.column_stack([x, y]), FPS, None, None, SpeedSettings(), paths=[path]).result
    assert r["speed_kmh"] == pytest.approx(60.0, rel=0.003)
    assert r["speed_line_kmh"] > r["speed_kmh"]  # the vehicle's own trajectory is longer


@pytest.mark.slow
def test_synthetic_clip_measured_along_a_path(tmp_path):
    from speedcam.pipeline import analyze
    from speedcam.synth import make_synthetic

    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=55.0, size=(960, 540), m_per_px=0.07)
    y = (cfg.calibration.line_a[0][1] + cfg.calibration.line_a[1][1]) / 2
    cfg.calibration.line_a = cfg.calibration.line_b = None
    cfg.calibration.paths = [MotionPath(name="Main", points=[(60.0, y), (480.0, y + 2), (900.0, y)])]
    summary = analyze(cfg, tmp_path / "out")
    v = [v for v in summary["vehicles"] if v["valid"]][0]
    assert v["speed_kmh"] == pytest.approx(55.0, rel=0.005)
    assert v["path"] == "Main" and summary["by_path"]["Main"]["avg"]["n_valid"] == 1
    # longer than the 24 m A/B section, but only where the calibration distances reach
    assert 30 < v["path_length_m"] < 50
    assert any("outside the calibrated area" in w for w in summary["calibration"]["warnings"])
    cfg.calibration.path_clip_to_calibration = False  # measure the whole visible road anyway
    v = [v for v in analyze(cfg, tmp_path / "out2")["vehicles"] if v["valid"]][0]
    assert v["path_length_m"] > 50 and v["speed_kmh"] == pytest.approx(55.0, rel=0.005)


def crossroads():
    """Two straight roads crossing at the origin: east-west and north-south, 120 m each."""
    ew = PathGeom.from_world_points(np.array([[-60.0, 0.0], [60.0, 0.0]]), corridor_m=5, name="East-West")
    ns = PathGeom.from_world_points(np.array([[0.0, -60.0], [0.0, 60.0]]), corridor_m=5, name="North-South")
    return [ew, ns]


def straight(p0, p1, v_kmh):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = int(np.linalg.norm(p1 - p0) / (v_kmh / 3.6) * FPS)
    f = np.arange(n) / max(n - 1, 1)
    return np.arange(n), p0 + f[:, None] * (p1 - p0)


def test_intersection_each_vehicle_on_its_own_road():
    paths = crossroads()
    frames, xy = straight((-58, 1.5), (58, 1.5), 40.0)  # east-west, crossing the north-south corridor
    r = analyze_track(1, frames, xy, FPS, None, None, SpeedSettings(), paths=paths).result
    assert r["path"] == "East-West" and r["speed_kmh"] == pytest.approx(40.0, rel=0.005)
    frames, xy = straight((-1.5, 58), (-1.5, -58), 30.0)  # north to south
    r = analyze_track(2, frames, xy, FPS, None, None, SpeedSettings(), paths=paths).result
    assert r["path"] == "North-South" and r["direction"] == "B->A"


def test_turning_vehicle_is_counted_once_on_its_longer_road():
    paths = crossroads()
    # 50 m westbound on East-West, then turns and drives 20 m south on North-South
    f1, a = straight((50, -1.5), (0, -1.5), 36.0)
    f2, b = straight((0, -1.5), (0, -21.5), 36.0)
    xy = np.vstack([a, b[1:]])
    rend = analyze_track(3, np.arange(len(xy)), xy, FPS, None, None, SpeedSettings(), paths=paths)
    assert rend.result["path"] == "East-West"
    assert rend.result["path_length_m"] == pytest.approx(48, abs=4)
