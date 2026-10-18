import math

import numpy as np
import pandas as pd
import pytest

from speedcam import calibration, uncertainty
from speedcam.config import (Calibration, ControlPoint, Polygon, PolygonDistance, Segment, SpeedSettings,
                             UncertaintySettings)
from speedcam.speed import compute_speeds
from speedcam.validation import Reference, evaluate

PATH = np.array([[0.0, 0.0], [800.0, 0.0]])  # 40 m along the road at 0.05 m/px


def _rel(cal, tol_m, tol_px, path=PATH, n=2000):
    H0 = calibration.compute(cal).H
    Hs = uncertainty.sample_homographies(cal, n, tol_m, tol_px, seed=1)
    return uncertainty.length_rel_sigma(H0 / H0[2, 2], Hs, path)


# ---------------------------------------------------------------- Monte Carlo propagation
def test_distance_error_propagates_to_relative_scale():
    # nadir view, one 20 m reference: a 20 cm error on it is a 1 % error on every speed
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0)])
    assert _rel(cal, 0.20, 0.0) == pytest.approx(0.01, rel=0.07)


def test_independent_distances_average_down():
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0),
                                Segment(a=(0, 100), b=(0, 500), meters=20.0)])
    assert _rel(cal, 0.20, 0.0) == pytest.approx(0.01 / math.sqrt(2), rel=0.07)


def test_click_error_propagates():
    # 0.5 px on each end of a 400 px reference: length sigma 0.5*sqrt(2) px
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0)])
    assert _rel(cal, 0.0, 0.5) == pytest.approx(0.5 * math.sqrt(2) / 400, rel=0.07)


def test_polygon_propagation_scales_with_tolerance():
    # 20 x 10 m rectangle seen straight down (0.05 m/px), 4 sides + both diagonals
    world = np.array([(0, 0), (20, 0), (20, 10), (0, 10)], float)
    px = world / 0.05
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    cal = Calibration(polygon=Polygon(points=[tuple(p) for p in px], distances=[
        PolygonDistance(i=i, j=j, meters=float(np.linalg.norm(world[i] - world[j]))) for i, j in pairs]))
    path = np.array([[0.0, 100.0], [400.0, 100.0]])  # 20 m along the long side
    r1, r2 = _rel(cal, 0.05, 0.0, path, 600), _rel(cal, 0.10, 0.0, path, 600)
    assert r2 == pytest.approx(2 * r1, rel=0.15)
    # redundant distances average the error below that of a single 20 m tape measurement
    assert 0.3 * 0.05 / 20 < r1 < 0.05 / 20


def test_homography_points_world_tolerance():
    world = np.array([(0, 0), (20, 0), (20, 10), (0, 10)], float)
    cal = Calibration(points=[ControlPoint(px=tuple(w / 0.05), world=tuple(w)) for w in world])
    r = _rel(cal, 0.10, 0.0, np.array([[0.0, 100.0], [400.0, 100.0]]), 600)
    assert 0.001 < r < 0.01


def test_line_gap_uncertainty_matches_scale():
    cal = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0)],
                      line_a=((200, -400), (200, 400)), line_b=((600, -400), (600, 400)))
    s = uncertainty.annotate([], {}, cal, calibration.compute(cal), 60.0,
                             UncertaintySettings(distance_tolerance_m=0.2, click_tolerance_px=0, mc_samples=2000))
    assert s["line_gap_rel_pct"] == pytest.approx(2 * 1.0, rel=0.07)  # k=2 x 1 %


# ---------------------------------------------------------------- combination
def test_rss_combination():
    assert uncertainty.rss([3.0, 4.0]) == pytest.approx(10.0)
    assert uncertainty.rss([3.0, None, 0.0, 4.0], k=1) == pytest.approx(5.0)
    assert uncertainty.rss([]) == 0.0


def test_parallax_bias_bound():
    assert uncertainty.parallax_bias(100, 0.8) == pytest.approx(100 / 99.2 - 1)
    assert uncertainty.parallax_bias(50, 1.5) == pytest.approx(0.0309, abs=1e-4)
    assert uncertainty.parallax_bias(None, 0.8) is None


def _measure(us: UncertaintySettings, cal: Calibration, speed=60.0, noise=0.05):
    res = calibration.compute(cal)
    v, fps = speed / 3.6, 60.0
    t = np.arange(0, 45 / v, 1 / fps)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(frame=np.arange(len(t)), track_id=1, x_m=t * v + rng.normal(0, noise, len(t)),
                           y_m=rng.normal(0, noise, len(t)), cls=0, conf=0.9))
    results, renders = compute_speeds(df, fps, res.line_a_world, res.line_b_world, SpeedSettings())
    summary = uncertainty.annotate(results, renders, cal, res, fps, us)
    return results[0], summary


CAL20 = Calibration(segments=[Segment(a=(0, 0), b=(400, 0), meters=20.0)],
                    line_a=((200, -400), (200, 400)), line_b=((600, -400), (600, 400)))


def test_vehicle_budget_is_rss_of_terms():
    r, s = _measure(UncertaintySettings(distance_tolerance_m=0.1, click_tolerance_px=0.0, mc_samples=1000), CAL20)
    assert r["class"] == "car"
    for key, v in (("avg", r["speed_kmh"]), ("max", r["speed_max_kmh"])):
        terms = r["uncertainty"][key]
        # 10 cm on 20 m = 0.5 % (1 sigma) of the speed, doubled for k=2
        assert terms["calibration"] == pytest.approx(2 * 0.005 * v, rel=0.08)
        assert r[f"speed_{key}_u_kmh"] == pytest.approx(math.hypot(terms["calibration"], terms["noise"]))
        assert r[f"speed_{key}_u_lower_kmh"] == r[f"speed_{key}_u_kmh"]  # no drone height: nothing one-sided
    assert r["uncertainty"]["max"]["noise"] == pytest.approx(2 * r["speed_sigma_kmh"])
    assert r["uncertainty"]["avg"]["noise"] == pytest.approx(2 * r["speed_avg_se_kmh"])
    assert s["parallax"] == "unknown" and r["u_parallax_pct"] is None
    assert not r["u_over_target"]


def test_parallax_widens_lower_side_and_flags_target():
    r, s = _measure(UncertaintySettings(drone_height_m=50.0, point_height_m={"car": 1.5}, target_pct=2.0), CAL20)
    b = 50 / 48.5 - 1
    assert r["u_parallax_pct"] == pytest.approx(100 * b)
    assert r["speed_avg_u_lower_kmh"] - r["speed_avg_u_kmh"] == pytest.approx(r["speed_kmh"] * b / (1 + b))
    assert r["u_over_target"] and s["n_over_target"] == 1


def test_check_residuals_inflate_calibration_term():
    # two references that disagree by 2 % while declared good to 1 cm: the fit ratio inflates the term
    bad = CAL20.model_copy(update={"segments": [Segment(a=(0, 0), b=(400, 0), meters=20.0),
                                                Segment(a=(0, 100), b=(400, 100), meters=20.4)]})
    us = UncertaintySettings(distance_tolerance_m=0.01, click_tolerance_px=0.0)
    r_ok, s_ok = _measure(us, CAL20)
    r_bad, s_bad = _measure(us, bad)
    assert s_ok["fit_ratio"] is None  # a single reference can't be checked
    assert s_bad["fit_ratio"] > 10 and s_bad["fit_inflation"] == s_bad["fit_ratio"]
    assert r_bad["uncertainty"]["avg"]["calibration"] > 5 * r_ok["uncertainty"]["avg"]["calibration"]


def test_avg_standard_error_grows_with_noise():
    us = UncertaintySettings(distance_tolerance_m=0.0, click_tolerance_px=0.0)
    lo, _ = _measure(us, CAL20, noise=0.02)
    hi, _ = _measure(us, CAL20, noise=0.2)
    assert hi["speed_avg_se_kmh"] > 3 * lo["speed_avg_se_kmh"]
    # the declared margin covers the actual error of the noisy average
    assert abs(hi["speed_kmh"] - 60.0) < hi["speed_avg_u_kmh"]


# ---------------------------------------------------------------- validation run
def _veh(tid, t0, speed, u=1.0, direction="A->B", valid=True):
    return {"track_id": tid, "t_entry_s": t0, "t_exit_s": t0 + 2, "direction": direction, "valid": valid,
            "flags": "", "speed_kmh": speed, "speed_max_kmh": speed + 1, "speed_avg_u_kmh": u,
            "speed_avg_u_lower_kmh": u, "speed_max_u_kmh": u, "speed_max_u_lower_kmh": u}


def test_validation_matches_by_time_and_track():
    vehicles = [_veh(1, 0.0, 50.0), _veh(2, 10.0, 60.0), _veh(3, 10.5, 70.0, direction="B->A")]
    rep = evaluate([Reference(speed_kmh=50.4, t_s=1.0), Reference(speed_kmh=68.5, t_s=11.0, direction="B->A"),
                    Reference(speed_kmh=60.0, track_id=2), Reference(speed_kmh=60.0, t_s=30.0)], vehicles)
    assert [r["track_id"] for r in rep["rows"]] == [1, 3, 2, None]
    assert rep["rows"][0]["inside"] and not rep["rows"][1]["inside"]  # 1.5 km/h off vs hypot(1, 0.5)
    assert rep["summary"]["n_matched"] == 3 and rep["summary"]["verdict"] == "too_few"


def test_validation_coverage_and_bias_verdicts():
    rng = np.random.default_rng(3)
    vehicles = [_veh(i, 5.0 * i, 50.0 + i) for i in range(30)]
    # errors consistent with the declared ±1 km/h (k=2 -> sigma 0.5): ~95 % inside
    good = [Reference(speed_kmh=v["speed_kmh"] - rng.normal(0, 0.4), track_id=v["track_id"], tolerance_kmh=0.3)
            for v in vehicles]
    s = evaluate(good, vehicles)["summary"]
    assert s["verdict"] == "consistent" and s["coverage"] >= 0.85
    # measurements 3 % high: outside the margin and flagged
    high = [Reference(speed_kmh=v["speed_kmh"] / 1.03, track_id=v["track_id"], tolerance_kmh=0.3) for v in vehicles]
    s = evaluate(high, vehicles)["summary"]
    assert s["verdict"] == "margin_too_small" and s["bias_pct"] == pytest.approx(3.0, abs=0.01)
    # a small shared bias inside the margin still shows up as a bias
    biased = [Reference(speed_kmh=v["speed_kmh"] / 1.01 + rng.normal(0, 0.05), track_id=v["track_id"],
                        tolerance_kmh=2.0) for v in vehicles]
    assert evaluate(biased, vehicles)["summary"]["verdict"] == "bias"
