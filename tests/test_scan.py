import json

import numpy as np
import pytest

from speedcam import scan
from speedcam.config import ProjectConfig, ScanSettings
from speedcam.synth import make_synthetic
from speedcam.video import FFmpegWriter, iter_frames, iter_ranges

EMPTY = 90  # frames of empty road before and after the car (3 s at 30 fps)


def _padded_clip(tmp_path) -> ProjectConfig:
    """The synthetic car clip with empty road before and after it, like a quiet CCTV stretch."""
    cfg = make_synthetic(tmp_path / "car.mp4", speed_kmh=70.0, size=(640, 360), m_per_px=0.1)
    frames = [f for _, f in iter_frames(cfg.source)]
    out = tmp_path / "long.mp4"
    w = FFmpegWriter(out, 640, 360, 30.0, encoder="x264", quality=14)
    for f in [frames[0]] * EMPTY + frames + [frames[-1]] * EMPTY:  # first/last frames: car off screen
        w.write(f)
    w.close()
    data = cfg.model_dump()
    data["source"] = str(out)
    return ProjectConfig.model_validate(data)


def test_iter_ranges_matches_sequential_decoding(tmp_path):
    cfg = make_synthetic(tmp_path / "car.mp4", speed_kmh=70.0, size=(640, 360), m_per_px=0.1)
    seq = {i: f for i, f in iter_frames(cfg.source)}
    got = list(iter_ranges(cfg.source, [(5, 9), (200, 203), (204, 206)]))
    assert [i for i, _ in got] == [5, 6, 7, 8, 200, 201, 202, 204, 205]
    for i, f in got:
        assert np.array_equal(f, seq[i])


def test_suggest_parts_pads_and_merges():
    sc = {"frames": list(range(0, 300, 30)), "counts": [0, 0, 1, 1, 0, 0, 0, 2, 0, 0], "nb_frames": 300,
          "container_fps": 30.0}
    s = ScanSettings(pad_s=1.0, merge_s=0.0)
    assert scan.suggest_parts(sc, s) == [(15, 136), (165, 256)]
    assert scan.suggest_parts(sc, ScanSettings(pad_s=1.0, merge_s=2.0)) == [(15, 256)]
    assert scan.suggest_parts({**sc, "counts": [0] * 10}, s) == []


@pytest.mark.slow
def test_scan_then_analyse_only_the_part_with_the_car(tmp_path):
    from speedcam.pipeline import analyze

    cfg = _padded_clip(tmp_path)
    sc = scan.scan(cfg, tmp_path)
    assert json.loads((tmp_path / scan.SCAN_FILE).read_text())["counts"] == sc["counts"]
    counts = dict(zip(sc["frames"], sc["counts"]))
    assert counts[min(counts)] == 0 and counts[max(counts)] == 0  # empty road at both ends
    parts = scan.suggest_parts(sc, cfg.scan)
    assert len(parts) == 1
    a, b = parts[0]
    assert 0 < a < EMPTY and sc["nb_frames"] - EMPTY < b < sc["nb_frames"]

    out = tmp_path / "parts"
    s = analyze(cfg.model_copy(update={"parts": parts}), out)
    assert s["run"]["frames"] == b - a
    [v] = s["vehicles"]
    assert v["valid"] and v["speed_kmh"] == pytest.approx(70.0, rel=0.005)

    # two parts: tracks restart at the jump with new ids, nothing joins across it
    mid = (a + b) // 2
    s2 = analyze(cfg.model_copy(update={"parts": [(a, mid - 40), (mid + 40, b)]}), tmp_path / "two")
    import pandas as pd

    tr = pd.read_parquet(tmp_path / "two" / "tracks.parquet")
    first = set(tr[tr.frame < mid].track_id)
    second = set(tr[tr.frame > mid].track_id)
    assert first and second and not first & second
    assert s2["run"]["frames"] == (mid - 40 - a) + (b - mid - 40)


def test_parked_vehicles_are_not_activity():
    parked = np.array([[0, 0, 10, 20.0]])
    car = lambda x: np.array([[x, 50, x + 10, 70.0]])  # noqa: E731
    boxes = [np.vstack([parked, car(0)]), np.vstack([parked, car(30)]), parked.copy(), parked.copy()]
    assert scan.moving_counts(boxes, 0.6) == [1, 1, 0, 0]
    sc = {"frames": [0, 30, 60, 90], "counts": [2, 2, 1, 1], "moving": [1, 1, 0, 0], "nb_frames": 120, "container_fps": 30.0}
    assert scan.suggest_parts(sc, ScanSettings(pad_s=0, merge_s=0)) == [(0, 46)]
    assert scan.suggest_parts(sc, ScanSettings(pad_s=0, merge_s=0, moving_only=False)) == [(0, 120)]
