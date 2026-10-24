import numpy as np
import pandas as pd
import pytest

from speedcam.vehicles import apply_merges, body_colour, colour_name, merge_candidates, merge_groups


def _track(tid, frames, fps=30.0, speed_mps=15.0, x0=0.0, y=0.0, conf=0.8):
    f = np.asarray(frames)
    return pd.DataFrame({"frame": f, "track_id": tid, "cls": 0, "conf": conf,
                         "x_m": x0 + speed_mps * f / fps, "y_m": y})


def test_merge_groups_join_chains():
    assert merge_groups([[3, 4], [4, 7], [9, 9], [1]]) == [[3, 4, 7]]
    assert merge_groups([[1, 2], [5, 6]]) == [[1, 2], [5, 6]]


def test_apply_merges_relabels_and_keeps_the_more_confident_overlap():
    a = _track(1, range(0, 40))
    b = _track(2, range(38, 80), conf=0.9)  # overlaps frames 38-39
    df = apply_merges(pd.concat([a, b]), [[1, 2]])
    assert set(df["track_id"]) == {1}
    assert not df.duplicated(["track_id", "frame"]).any()
    assert len(df) == 80
    assert df.loc[df["frame"] == 39, "conf"].iloc[0] == pytest.approx(0.9)


def test_candidates_find_a_split_vehicle_and_not_a_neighbour():
    fps = 30.0
    a = _track(1, range(0, 60), fps)
    b = _track(2, range(75, 150), fps)  # the same car after a 0.5 s gap behind a tree
    other = _track(3, range(70, 150), fps, x0=40.0)  # another car well ahead in the same lane
    across = _track(4, range(65, 150), fps, y=30.0)  # crossing traffic elsewhere
    pairs = merge_candidates(pd.concat([a, b, other, across]), fps)
    likely = [(p["a"], p["b"]) for p in pairs if p["likely"]]
    assert likely == [(1, 2)]
    best = next(p for p in pairs if p["a"] == 1 and p["b"] == 2)
    assert best["gap_s"] == pytest.approx(16 / fps, abs=1e-3)  # last frame 59, first frame 75
    assert best["miss_m"] < 0.5 and best["speed_kmh"] == pytest.approx(54.0, abs=0.5)
    # neither track reaches the measured section: merging them couldn't change a measurement
    assert not any((p["a"], p["b"]) == (1, 2) for p in merge_candidates(pd.concat([a, b, other]), fps, relevant={3}))
    assert any((p["a"], p["b"]) == (1, 2) for p in merge_candidates(pd.concat([a, b, other]), fps, relevant={2}))


def test_a_much_faster_track_is_not_a_likely_continuation():
    fps = 30.0
    slow = _track(1, range(0, 60), fps, speed_mps=5.0)
    # starts where the slow one would be after the gap, but at three times its speed
    fast = _track(2, range(75, 150), fps, speed_mps=15.0, x0=5.0 * 75 / fps - 15.0 * 75 / fps)
    pairs = merge_candidates(pd.concat([slow, fast]), fps)
    assert pairs and not pairs[0]["likely"] and pairs[0]["speed_change_kmh"] == pytest.approx(36.0, abs=1)


def test_parked_cars_are_never_candidates():
    fps = 30.0
    parked = _track(1, range(0, 60), fps, speed_mps=0.0)
    arrives = _track(2, range(70, 120), fps, speed_mps=0.2)
    assert merge_candidates(pd.concat([parked, arrives]), fps) == []


@pytest.mark.parametrize("rgb,name", [
    ((245, 245, 245), "white"), ((20, 20, 22), "black"), ((160, 162, 165), "silver"),
    ((200, 30, 30), "red"), ((30, 60, 190), "blue"), ((240, 200, 30), "yellow"), ((40, 140, 60), "green"),
])
def test_colour_names(rgb, name):
    assert colour_name(rgb) == name


def test_body_colour_ignores_the_road_around_the_car():
    img = np.full((100, 100, 3), (90, 90, 90), np.uint8)  # grey asphalt (BGR)
    img[30:70, 20:80] = (40, 40, 200)  # a red car
    img[45:55, 30:70] = (30, 30, 30)  # its dark windscreen strip
    quad = np.array([[20, 30], [80, 30], [80, 70], [20, 70]], float)
    assert colour_name(body_colour(img, quad)) == "red"
