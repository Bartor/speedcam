"""End-to-end validation from the brief: a moving rectangle at a known speed over a
still frame with jitter must be recovered within 0.5 %."""
import pytest

from speedcam.pipeline import analyze
from speedcam.render import render
from speedcam.synth import make_synthetic


@pytest.mark.slow
@pytest.mark.parametrize("speed,seed", [(50.0, 0), (95.0, 1)])
def test_synthetic_speed_within_half_percent(tmp_path, speed, seed):
    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=speed, seed=seed, size=(960, 540), m_per_px=0.07)
    summary = analyze(cfg, tmp_path / "out")
    valid = [v for v in summary["vehicles"] if v["valid"]]
    assert len(valid) == 1
    v = valid[0]
    assert v["speed_kmh"] == pytest.approx(speed, rel=0.005)
    assert v["speed_line_kmh"] == pytest.approx(speed, rel=0.005)
    assert summary["run"]["registration_failures"] == 0
    # container is 30 fps but real time comes from capture_fps=120
    assert v["dt_s"] == pytest.approx(24.0 / (speed / 3.6), rel=0.005)


@pytest.mark.slow
def test_synthetic_textured_car_uses_flow(tmp_path):
    import numpy as np
    import pandas as pd

    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=60.0, seed=2, size=(960, 540), m_per_px=0.07,
                         textured=True)
    summary = analyze(cfg, tmp_path / "out")
    v = [v for v in summary["vehicles"] if v["valid"]][0]
    assert v["speed_kmh"] == pytest.approx(60.0, rel=0.005)
    assert v["speed_max_kmh"] == pytest.approx(60.0, abs=0.6)
    tracks = pd.read_parquet(tmp_path / "out" / "tracks.parquet")
    between = tracks[(tracks["frame"] >= v["frame_entry"]) & (tracks["frame"] <= v["frame_exit"])]
    assert np.isfinite(between["frx"]).mean() > 0.9  # positions came from flow, not the box


@pytest.mark.slow
def test_render_produces_video(tmp_path):
    cfg = make_synthetic(tmp_path / "syn.mp4", speed_kmh=60.0, size=(640, 360), m_per_px=0.1)
    analyze(cfg, tmp_path / "out")
    cfg.render.output_fps = 30
    cfg.render.scale = 1.0
    meta = render(cfg, tmp_path / "out")
    assert (tmp_path / "out" / "render.mp4").stat().st_size > 10_000
    # 120 fps capture rendered at 30 fps real time keeps every 4th frame
    from speedcam.timing import probe

    info = probe(cfg.source)
    assert meta["frames"] == pytest.approx(info.nb_frames / 4, abs=1)
