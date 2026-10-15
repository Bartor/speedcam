import math

import cv2
import numpy as np
import pytest

from speedcam import calibration
from speedcam.config import (Calibration, ControlPoint, Polygon, PolygonDistance, Segment, SpeedSettings,
                             StabilizationSettings)
from speedcam.speed import analyze_track, first_crossing
from speedcam.stabilization import Stabilizer, transform_points
from speedcam.stats import required_n, summarize
from speedcam.config import StatsSettings
from speedcam.synth import road_texture
from speedcam.timing import VideoInfo, check_timing, frame_time, sidecar_srt_fps


# ---------------------------------------------------------------- timing
def _info(**kw):
    base = dict(path="x.mp4", width=3840, height=2160, container_fps=30.0, nb_frames=1200, duration_s=40.0,
                codec="hevc", pix_fmt="yuv420p10le", bit_depth=10, color_transfer="", is_dji=True,
                srt_fps=None, suggested_capture_fps=30.0, slowmo_suspect=True)
    base.update(kw)
    return VideoInfo(**base)


def test_frame_time_uses_capture_fps():
    assert frame_time(120, 120.0) == pytest.approx(1.0)
    assert frame_time(150, 100.0) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        frame_time(1, 0)


def test_slowmo_clip_real_time_is_shorter_than_container():
    info = _info()
    msgs = check_timing(info, 120.0)
    assert any("slow motion" in m for m in msgs)
    # 1200 frames at 120 fps = 10 s real, container says 40 s
    assert frame_time(info.nb_frames, 120.0) == pytest.approx(10.0)


def test_known_duration_check():
    info = _info()
    assert any("known duration" in m for m in check_timing(info, 30.0, real_duration_s=10.0))
    assert not any("known duration" in m for m in check_timing(info, 120.0, real_duration_s=10.0))


def test_sidecar_srt(tmp_path):
    video = tmp_path / "DJI_0001.MP4"
    video.write_bytes(b"")
    blocks = [f"{i}\n00:00:00,000 --> 00:00:00,008\n<font>FrameCnt: {i}, DiffTime: 8ms\n" for i in range(1, 20)]
    (tmp_path / "DJI_0001.SRT").write_text("\n".join(blocks))
    assert sidecar_srt_fps(video) == pytest.approx(125.0)


# ---------------------------------------------------------------- calibration
def test_scale_from_segments():
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0),
                                Segment(a=(0, 0), b=(0, 200), meters=10.0)])
    r = calibration.compute(cal)
    assert r.method == "scale"
    assert r.m_per_px == pytest.approx(0.05)
    assert r.rms_m == pytest.approx(0.0, abs=1e-9)


def test_segment_residual_reported():
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0),
                                Segment(a=(0, 0), b=(400, 0), meters=20.4)])
    r = calibration.compute(cal)
    assert r.max_rel_error == pytest.approx(0.01, rel=0.05)


def test_homography_recovered_from_four_points():
    H_true = np.array([[0.05, 0.002, 1.0], [-0.001, 0.049, -2.0], [1e-6, 2e-6, 1.0]])
    px = [(100, 100), (3700, 120), (3650, 2000), (150, 2050), (1900, 1000)]
    world = calibration.apply_h(H_true, px)
    cal = Calibration(points=[ControlPoint(px=p, world=tuple(w)) for p, w in zip(px, world)])
    r = calibration.compute(cal)
    assert r.method == "homography"
    test = np.array([[1000.0, 1500.0], [3000.0, 500.0]])
    np.testing.assert_allclose(calibration.apply_h(r.H, test), calibration.apply_h(H_true, test), atol=1e-6)


def test_similarity_with_y_up_world():
    # world y points up (map convention) while image y points down: needs a reflection
    px = np.array([(100.0, 100.0), (500.0, 100.0), (100.0, 400.0)])
    world = np.column_stack([px[:, 0] * 0.1, -px[:, 1] * 0.1])
    cal = Calibration(points=[ControlPoint(px=tuple(p), world=tuple(w)) for p, w in zip(px, world)])
    r = calibration.compute(cal)
    assert r.method == "similarity"
    assert r.rms_m < 1e-9
    assert r.m_per_px == pytest.approx(0.1)


def test_scale_model_rejected_for_perspective_view():
    # along-road and across-road distances imply different scales: the view is oblique
    cal = Calibration(segments=[Segment(a=(1025, 362), b=(1879, 1044), meters=35.0),
                                Segment(a=(1023, 683), b=(1441, 704), meters=7.5)])
    r = calibration.compute(cal)
    assert not r.fit_ok
    assert "Rectangle" in r.warnings[0]


def test_rectangle_homography_fits():
    H_true = np.array([[0.02, 0.01, -5.0], [-0.004, 0.035, -3.0], [2e-6, 1.5e-4, 1.0]])
    world = np.array([(0, 0), (12, 0), (12, 3), (0, 3)], float)
    px = calibration.apply_h(np.linalg.inv(H_true), world)
    check_px = calibration.apply_h(np.linalg.inv(H_true), [(0, 6), (20, 6)])
    cal = Calibration(points=[ControlPoint(px=tuple(p), world=tuple(w)) for p, w in zip(px, world)],
                      segments=[Segment(a=tuple(check_px[0]), b=tuple(check_px[1]), meters=20.0)])
    r = calibration.compute(cal)
    assert r.method == "homography" and r.fit_ok
    assert r.max_rel_error < 1e-6


H_OBLIQUE = np.array([[0.02, 0.01, -5.0], [-0.004, 0.035, -3.0], [2e-6, 1.5e-4, 1.0]])


def _polygon(world, pairs, H=H_OBLIQUE, scale=1.0):
    px = calibration.apply_h(np.linalg.inv(H), world)
    d = [PolygonDistance(i=i, j=j, meters=float(np.linalg.norm(world[i] - world[j])) * scale) for i, j in pairs]
    return Calibration(polygon=Polygon(points=[tuple(p) for p in px], distances=d))


def test_polygon_recovers_perspective():
    # irregular quadrilateral on the ground: 4 sides + both diagonals (one redundant)
    world = np.array([(0, 0), (10, 1), (12, 25), (-1, 22)], float)
    r = calibration.compute(_polygon(world, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]))
    assert r.method == "polygon" and r.fit_ok
    a, b = calibration.apply_h(np.linalg.inv(H_OBLIQUE), [(2, 3), (8, 20)])
    assert np.linalg.norm(np.subtract(*calibration.apply_h(r.H, [a, b]))) == pytest.approx(np.hypot(6, 17), rel=1e-6)


def test_polygon_pentagon_least_squares():
    world = np.array([(0, 0), (8, -1), (13, 9), (6, 18), (-2, 10)], float)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (0, 3), (1, 3)]
    r = calibration.compute(_polygon(world, pairs))
    assert r.fit_ok and r.max_rel_error < 1e-6


def test_polygon_needs_diagonal():
    world = np.array([(0, 0), (10, 0), (10, 20), (0, 20)], float)
    with pytest.raises(ValueError, match="needs at least 5"):
        calibration.compute(_polygon(world, [(0, 1), (1, 2), (2, 3), (3, 0)]))


def test_polygon_typo_detected():
    world = np.array([(0, 0), (10, 1), (12, 25), (-1, 22)], float)
    cal = _polygon(world, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
    cal.polygon.distances[4].meters *= 1.2  # mistyped diagonal
    assert not calibration.compute(cal).fit_ok


def test_parallel_segments_warned():
    cal = Calibration(segments=[Segment(a=(2011, 1140), b=(2140, 1386), meters=31.7),
                                Segment(a=(1510, 1172), b=(1682, 1573), meters=51.5)])
    r = calibration.compute(cal)
    assert r.fit_ok
    assert any("point the same way" in w for w in r.warnings)


def test_line_gap():
    cal = Calibration(segments=[Segment(a=(0, 0), b=(100, 0), meters=5.0)],
                      line_a=((100, 0), (100, 500)), line_b=((500, 0), (500, 500)))
    assert calibration.compute(cal).line_gap_m == pytest.approx(20.0)


# ---------------------------------------------------------------- crossings & speed
def test_subframe_crossing_interpolation():
    t = np.array([0.0, 0.01, 0.02])
    P = np.array([[9.7, 0.0], [10.3, 0.0], [10.9, 0.0]])
    c = first_crossing(t, P, np.array([[10.0, -5.0], [10.0, 5.0]]), 0.1)
    assert c.t == pytest.approx(0.005)
    np.testing.assert_allclose(c.point, [10.0, 0.0])


def test_crossing_outside_segment_ignored():
    t = np.array([0.0, 0.01])
    P = np.array([[9.0, 20.0], [11.0, 20.0]])
    assert first_crossing(t, P, np.array([[10.0, -5.0], [10.0, 5.0]]), 0.1) is None


LINES = (np.array([[10.0, -10.0], [10.0, 10.0]]), np.array([[30.0, -10.0], [30.0, 10.0]]))


def _track(speed_kmh, fps=120.0, noise=0.0, lane_change=0.0, seed=0):
    v = speed_kmh / 3.6
    t = np.arange(0, 45 / v, 1 / fps)
    x = t * v
    y = lane_change / (1 + np.exp(-(x - 20) / 2))  # smooth lane change around the middle
    rng = np.random.default_rng(seed)
    xy = np.column_stack([x, y]) + rng.normal(0, noise, (len(t), 2))
    return np.arange(len(t)), xy


@pytest.mark.parametrize("speed", [30.0, 50.0, 90.0, 140.0])
def test_constant_speed_recovered(speed):
    frames, xy = _track(speed, noise=0.02)
    r = analyze_track(1, frames, xy, 120.0, *LINES, SpeedSettings()).result
    assert r["direction"] == "A->B"
    assert r["speed_kmh"] == pytest.approx(speed, rel=0.005)
    assert r["speed_line_kmh"] == pytest.approx(speed, rel=0.005)
    assert r["valid"]


def test_reverse_direction():
    frames, xy = _track(60.0)
    xy[:, 0] = 45 - xy[:, 0]
    r = analyze_track(1, frames, xy, 120.0, *LINES, SpeedSettings()).result
    assert r["direction"] == "B->A"
    assert r["speed_kmh"] == pytest.approx(60.0, rel=0.002)


def test_lane_change_arc_length_longer_than_chord():
    frames, xy = _track(50.0, lane_change=3.5)
    r = analyze_track(1, frames, xy, 120.0, *LINES, SpeedSettings()).result
    # true path speed is 50 km/h along x plus the lateral move; arc length captures the extra distance
    assert r["path_length_m"] > r["chord_m"]
    assert r["speed_kmh"] > r["speed_line_kmh"]


def test_id_switch_jump_rejected():
    frames, xy = _track(50.0)
    xy[len(xy) // 2:, 0] += 4.0  # identity jumps to a car 4 m ahead
    r = analyze_track(1, frames, xy, 120.0, *LINES, SpeedSettings()).result
    assert not r["valid"]
    assert "accel" in r["flags"] or "residual" in r["flags"]


def test_track_missing_a_line_has_no_result():
    frames, xy = _track(50.0)
    keep = xy[:, 0] < 25
    assert analyze_track(1, frames[keep], xy[keep], 120.0, *LINES, SpeedSettings()).result is None


def test_momentary_speed_constant():
    frames, xy = _track(72.0, noise=0.02)
    r = analyze_track(1, frames, xy, 120.0, *LINES, SpeedSettings())
    res = r.result
    assert res["speed_max_kmh"] == pytest.approx(72.0, abs=1.0)
    assert res["speed_min_kmh"] == pytest.approx(72.0, abs=1.0)
    assert res["speed_sigma_kmh"] < 0.5
    assert abs(res["speed_trend_pct"]) < 1
    assert len(res["profile"]["v"]) <= 121 and res["profile"]["d"][0] == pytest.approx(0, abs=0.5)


def test_momentary_speed_finds_peak():
    # 50 km/h, a 2 s burst up to 80 km/h in the middle of the section, then back to 50
    fps, dt = 120.0, 1 / 120.0
    t = np.arange(0, 6, dt)
    v = 50 / 3.6 + (30 / 3.6) * np.exp(-((t - 3.0) / 0.5) ** 2)
    x = np.cumsum(v) * dt - 5
    xy = np.column_stack([x, np.zeros_like(x)])
    lines = (np.array([[10.0, -10], [10.0, 10]]), np.array([[70.0, -10], [70.0, 10]]))
    res = analyze_track(1, np.arange(len(t)), xy, fps, *lines, SpeedSettings()).result
    assert res["speed_max_kmh"] == pytest.approx(80.0, abs=1.5)  # 0.5 s window slightly flattens the peak
    assert res["speed_min_kmh"] == pytest.approx(50.0, abs=1.0)
    assert res["speed_kmh"] < res["speed_max_kmh"] - 10  # the A->B average hides the peak


def test_calibration_check_flags_shared_trend():
    from speedcam.speed import calibration_check

    def veh(trend):
        return {"valid": True, "speed_trend_pct": trend}

    assert calibration_check([veh(13 + d) for d in (-2, -1, 0, 1, 2, 3)])["status"] == "suspect"
    assert calibration_check([veh(d) for d in (-6, 4, -2, 5, -3, 1)])["status"] == "ok"
    assert calibration_check([veh(10)])["status"] == "insufficient"


# ---------------------------------------------------------------- stats
def test_required_n_ite_formula():
    # S=8 km/h, K=1.96, E=2 km/h, U=1.04 -> (8*1.96/2)^2 * (2+1.04^2)/2 = 61.47*1.5408 -> 95
    assert required_n(8.0, 2.0, 0.95, "p85") == 95
    assert required_n(8.0, 2.0, 0.95, "mean") == math.ceil((8 * 1.959964 / 2) ** 2)


def test_summary_cis_contain_estimates():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 6, 300)
    s = summarize(x, StatsSettings())
    assert s["mean_ci"][0] < s["mean"] < s["mean_ci"][1]
    assert s["p85_ci"][0] <= s["p85"] <= s["p85_ci"][1]
    assert s["n_valid"] == 300


# ---------------------------------------------------------------- stabilization
@pytest.mark.parametrize("features,tol", [("klt", 0.1), ("orb", 0.3)])
def test_stabilizer_recovers_similarity(features, tol):
    rng = np.random.default_rng(3)
    ref = road_texture(1280, 720, 0.05, rng)
    M = cv2.getRotationMatrix2D((640, 360), 0.4, 1.0)
    M[:, 2] += (6.3, -4.1)
    frame = cv2.warpAffine(ref, M, (1280, 720), borderMode=cv2.BORDER_REFLECT)
    stab = Stabilizer(ref, StabilizationSettings(features=features))
    reg = stab.register(frame)
    assert reg.ok
    # frame -> ref is the inverse of the applied warp
    T_true = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    pts = np.array([[100.0, 100.0], [1100.0, 650.0], [640.0, 360.0]])
    err = np.linalg.norm(transform_points(reg.T, pts) - transform_points(T_true, pts), axis=1)
    assert err.max() < tol
