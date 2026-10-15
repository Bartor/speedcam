"""Command line entry points: probe, init, calibrate, analyze, recompute, render, synth, serve."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import ProjectConfig, load_config, save_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Drone vehicle speed measurement")


@app.command()
def probe(video: Path):
    """Print video metadata and timing warnings."""
    from .timing import probe as _probe

    info = _probe(video)
    d = info.to_dict()
    d.pop("tags")
    typer.echo(json.dumps(d, indent=2))


@app.command()
def init(video: Path, out: Path = typer.Option(Path("project.yaml"), help="Config file to write"),
         capture_fps: Optional[float] = typer.Option(None, help="Real capture rate (slow-motion clips!)")):
    """Write a config template for a video."""
    from .timing import probe as _probe

    info = _probe(video)
    for w in info.warnings:
        typer.secho(w, fg="yellow")
    cfg = ProjectConfig(source=str(video.resolve()), capture_fps=capture_fps or info.suggested_capture_fps)
    save_config(cfg, out)
    typer.echo(f"Wrote {out}. Run `speedcam calibrate {out}` next.")


@app.command()
def calibrate(config: Path, frame: Optional[int] = typer.Option(None, help="Reference frame index")):
    """Click ground points, lines and ROI on the reference frame (OpenCV window)."""
    from . import calibrate_tool

    cfg = load_config(config)
    cfg = calibrate_tool.run(cfg, frame if frame is not None else cfg.calibration.reference_frame)
    save_config(cfg, config)
    typer.echo(f"Saved calibration to {config}")


@app.command()
def analyze(config: Path, out: Path = typer.Option(Path("out"), help="Output folder")):
    """Run detection, tracking and speed estimation."""
    from .pipeline import PrintReporter, analyze as _analyze

    summary = _analyze(load_config(config), out, PrintReporter())
    _print_summary(summary, out)


@app.command()
def recompute(config: Path, out: Path = typer.Option(Path("out"))):
    """Recompute speeds from stored tracks (after editing lines, calibration or thresholds)."""
    from .pipeline import recompute as _recompute

    _print_summary(_recompute(load_config(config), out), out)


@app.command()
def render(config: Path, out: Path = typer.Option(Path("out")), dst: Optional[Path] = None,
           mode: Optional[str] = typer.Option(None, help="stabilized | original"),
           fps: Optional[float] = typer.Option(None, help="Output fps (default: native capture fps)"),
           playback: Optional[float] = typer.Option(None, help="Playback speed, e.g. 0.25 for 4x slow motion"),
           scale: Optional[float] = None, lut: Optional[Path] = None):
    """Export the annotated video."""
    from .pipeline import PrintReporter
    from .render import render as _render

    cfg = load_config(config)
    for k, v in dict(mode=mode, output_fps=fps, playback_speed=playback, scale=scale,
                     lut=str(lut) if lut else None).items():
        if v is not None:
            setattr(cfg.render, k, v)
    meta = _render(cfg, out, dst, PrintReporter())
    typer.echo(f"Wrote {Path(out) / meta['file']} ({meta['frames']} frames, {meta['codec']})")


@app.command()
def synth(out: Path = typer.Option(Path("synthetic.mp4")), speed: float = 50.0,
          config: Path = typer.Option(Path("synthetic.yaml")), seed: int = 0):
    """Generate a synthetic validation clip with a known speed and its config."""
    from .synth import make_synthetic

    cfg = make_synthetic(out, speed_kmh=speed, seed=seed)
    save_config(cfg, config)
    typer.echo(f"Wrote {out} ({speed} km/h ground truth) and {config}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000,
          data_dir: Optional[Path] = typer.Option(None, help="Where project folders live (default ~/speedcam-projects)"),
          jobs: int = typer.Option(2, "--jobs", help="Jobs that run at the same time"),
          gpu_jobs: int = typer.Option(1, "--gpu-jobs", help="Of those, detection jobs (each ~6-9 GB of VRAM)"),
          open_browser: bool = typer.Option(True, "--open/--no-open")):
    """Start the local web app."""
    import os
    import webbrowser

    import uvicorn

    if data_dir:
        os.environ["SPEEDCAM_DATA_DIR"] = str(data_dir.resolve())
    os.environ["SPEEDCAM_JOB_SLOTS"] = str(max(1, jobs))
    os.environ["SPEEDCAM_GPU_JOBS"] = str(max(1, gpu_jobs))
    if open_browser:
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    import asyncio
    import logging

    class _DropShutdownCancels(logging.Filter):
        """Requests cut off when the grace period ends raise CancelledError: expected, not an error."""

        def filter(self, record: logging.LogRecord) -> bool:
            return not (record.exc_info and isinstance(record.exc_info[1], asyncio.CancelledError))

    logging.getLogger("uvicorn.error").addFilter(_DropShutdownCancels())
    # A paused <video> keeps its download open without reading it, so a graceful shutdown would
    # wait for it forever: drop remaining connections after a short grace period.
    uvicorn.run("speedcam.server.app:app", host=host, port=port, log_level="info", timeout_graceful_shutdown=2)


def _print_summary(summary: dict, out: Path) -> None:
    st = summary["stats"]
    typer.echo(f"\n{st['n_valid']} valid of {st['n_total']} measured vehicles")
    for v in summary["vehicles"]:
        mark = "" if v["valid"] else f"  [invalid: {v['flags']}]"
        typer.echo(f"  #{v['track_id']:<4} {v['direction']}  {v['speed_kmh']:6.2f} km/h "
                   f"(lines {v['speed_line_kmh']:6.2f})  dt {v['dt_s']:.3f}s{mark}")
    if st.get("mean") is not None:
        typer.echo(f"mean {st['mean']:.1f}  p85 {st['p85']:.1f}  p95 {st['p95']:.1f} km/h")
    typer.echo(f"Results in {out}")


if __name__ == "__main__":
    app()
