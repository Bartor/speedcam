"""Analysis pipeline: decode → register → detect → track → world trajectories → speeds."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import calibration, uncertainty, violations
from .config import ProjectConfig
from .detection import make_detector
from .flow import FlowTracker, reference_positions
from .speed import calibration_check, compute_speeds
from .stabilization import Stabilizer, transform_points
from .stats import summarize
from .timing import check_timing, probe
from .tracking import Tracker
from .video import FrameGrabber, iter_frames, rotate_frame

log = logging.getLogger(__name__)


class Reporter:
    """Progress sink. The server subclasses this to forward events over SSE."""

    def progress(self, stage: str, done: int, total: int, **extra) -> None:
        pass

    def log(self, msg: str, level: str = "info") -> None:
        getattr(log, level if level != "warning" else "warning")(msg)

    def preview(self, jpeg: bytes) -> None:
        pass

    def cancelled(self) -> bool:
        return False


class PrintReporter(Reporter):
    def __init__(self):
        self._last = 0.0

    def progress(self, stage, done, total, **extra):
        now = time.monotonic()
        if now - self._last > 1.0 or done == total:
            self._last = now
            fps = extra.get("fps")
            print(f"\r{stage}: {done}/{total}" + (f"  {fps:.1f} fps" if fps else ""), end="", flush=True)
            if done == total:
                print()

    def log(self, msg, level="info"):
        print(f"\n[{level}] {msg}")


def resolve_capture_fps(cfg: ProjectConfig, info, reporter: Reporter) -> float:
    if cfg.capture_fps:
        return float(cfg.capture_fps)
    reporter.log(f"capture_fps not set; using {info.suggested_capture_fps:g} from metadata. "
                 "Set it explicitly for slow-motion clips.", "warning")
    return float(info.suggested_capture_fps)


def _track_color(tid: int) -> tuple[int, int, int]:
    h = (tid * 47) % 180
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _preview_jpeg(frame: np.ndarray, corners: np.ndarray, ids: np.ndarray, trails: dict,
                  lines: list, text: str, width: int = 1280, rotation: int = 0) -> bytes:
    """Tracking preview: boxes coloured by track id, id labels, trails and lines A/B (all frame px)."""
    k = width / frame.shape[1]
    img = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    for name, pts, color in lines:  # lines A/B, or the path of motion and its corridor edges
        p = (pts * k).astype(np.int32)
        cv2.polylines(img, [p], False, color, 2 if name else 1, cv2.LINE_AA)
        if name:
            cv2.putText(img, name, tuple(p[0] + [4, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    for tid, pts in trails.items():
        if len(pts) > 1:
            cv2.polylines(img, [(np.array(pts) * k).astype(np.int32)], False, _track_color(tid), 2, cv2.LINE_AA)
    for c, tid in zip(corners, ids):
        p = (c * k).astype(np.int32)
        color = _track_color(int(tid)) if tid >= 0 else (160, 160, 160)
        cv2.polylines(img, [p], True, color, 2 if tid >= 0 else 1, cv2.LINE_AA)
        if tid >= 0:
            org = (int(p[:, 0].min()), int(p[:, 1].min()) - 5)
            cv2.putText(img, f"#{tid}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"#{tid}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    img = rotate_frame(img, rotation)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


class _Prefetch:
    """Run a frame iterator on a background thread, up to `maxsize` frames ahead."""

    _END = object()

    def __init__(self, it, maxsize: int):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(it,), daemon=True, name="decode")
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, it) -> None:
        try:
            for item in it:
                if not self._put(item):
                    return
            self._put(self._END)
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            self._put(e)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _chunks(it, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def analyze(cfg: ProjectConfig, out_dir: str | Path, reporter: Optional[Reporter] = None,
            preview_every_s: float = 0.5) -> dict:
    reporter = reporter or Reporter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(cfg.source)
    fps = resolve_capture_fps(cfg, info, reporter)
    timing_msgs = check_timing(info, fps, cfg.real_duration_s)
    for m in timing_msgs:
        reporter.log(m, "warning")

    cal = calibration.compute(cfg.calibration, (info.width, info.height))
    if not calibration.has_measurement_geometry(cfg.calibration):
        raise ValueError("Draw a path of motion, measurement lines A and B, a stop line or a solid line "
                         "in the calibration")
    if not cal.fit_ok:
        raise ValueError(cal.warnings[0])
    roi = np.array(cfg.calibration.roi, np.float32) if cfg.calibration.roi and len(cfg.calibration.roi) >= 3 else None

    reporter.progress("loading model", 0, 1)
    detector = make_detector(cfg.detection)
    detector_info = None
    if hasattr(detector, "choose_primary"):
        # pick the model that suits this clip's camera angle, from frames spread across it
        n_s = max(1, cfg.detection.auto_sample_frames)
        lo, hi = max(0, cfg.start_frame), min(cfg.end_frame or info.nb_frames, info.nb_frames) - 1
        g = FrameGrabber(cfg.source)
        sample = [g.get(int(f)) for f in np.linspace(lo, max(lo, hi), n_s).round()]
        g.close()
        detector_info = detector.choose_primary(sample)
        reporter.log(f"Auto detector: primary {detector_info['primary']}, fill-in {detector_info['secondary']} "
                     f"(vehicles x confidence per frame: {detector_info['scores']})")

    ref_idx = cfg.calibration.reference_frame
    grabber = FrameGrabber(cfg.source)
    ref_frame = grabber.get(ref_idx)
    grabber.close()
    stab = None
    if cfg.stabilization.enabled:
        stab = Stabilizer(ref_frame, cfg.stabilization, detector(ref_frame).xyxy)
    tracker = Tracker(cfg.tracking, fps)
    flow = FlowTracker(cfg.flow, fps) if cfg.flow.enabled else None

    start = max(0, cfg.start_frame)
    end = min(cfg.end_frame or info.nb_frames, info.nb_frames)
    total = max(1, end - start)

    pts_anomalies = []

    def pts_check(i, pts, expected):
        if abs(pts - expected) > 0.5 / info.container_fps and len(pts_anomalies) < 1000:
            pts_anomalies.append((i, pts, expected))

    rows: list[np.ndarray] = []
    T_frames, T_all, stab_rows = [], [], []
    t_start = time.monotonic()
    # how far the camera moves relative to the reference frame (frame centre, px):
    # tells whether stabilization can be skipped for this footage
    centre = np.array([[info.width / 2, info.height / 2]])
    drift: list[float] = []
    trails: dict[int, list] = {}
    trail_len = int(round(fps))  # 1 s of history
    seen_tracks: set[int] = set()
    state = {"done": 0, "n_dets": 0, "last_preview": 0.0}
    # traffic-light colours, read on every frame while it's decoded anyway
    sampler = violations.LightSampler(cfg.calibration, cfg.violations) if violations.needs_samples(cfg.calibration) else None
    light_frames: list[int] = []
    light_counts: list[np.ndarray] = []

    def per_frame(i: int, frame: np.ndarray, dets) -> None:
        """Stabilize, track and record one frame. Called strictly in frame order (tracking needs it)."""
        state["n_dets"] += len(dets)
        if stab:
            reg = stab.register(frame, dets.xyxy)
            T = reg.T
            stab_rows.append((i, reg.ok, reg.inliers, reg.source))
            if not reg.ok and (stab.n_fallback <= 20 or stab.n_fallback % 100 == 0):
                reporter.log(f"frame {i}: registration failed ({reg.inliers} inliers), "
                             f"reusing previous transform ({stab.n_fallback} so far)", "warning")
            drift.append(float(np.linalg.norm(transform_points(T, centre) - centre)))
        else:
            T = np.eye(3)
        T_frames.append(i)
        T_all.append(T)
        if sampler:
            light_frames.append(i)
            light_counts.append(sampler.sample(frame, T))

        if len(dets):
            corners_ref = transform_points(T, dets.corners)
            centers_ref = corners_ref.mean(1)
            if roi is not None:
                keep = np.array([cv2.pointPolygonTest(roi, (float(x), float(y)), False) >= 0
                                 for x, y in centers_ref], bool)
                dets, corners_ref, centers_ref = dets.subset(keep), corners_ref[keep], centers_ref[keep]
        else:
            corners_ref = np.zeros((0, 4, 2))
            centers_ref = np.zeros((0, 2))
        xyxy_ref = (np.concatenate([corners_ref.min(1), corners_ref.max(1)], 1)
                    if len(dets) else np.zeros((0, 4)))
        ids = tracker.update(xyxy_ref, dets.conf, dets.cls)
        live = set()
        for tid, c in zip(ids, dets.centers):
            if tid >= 0:
                live.add(int(tid))
                seen_tracks.add(int(tid))
                trails.setdefault(int(tid), []).append(c)
                del trails[int(tid)][:-trail_len]
        for tid in [t for t in trails if t not in live]:  # fade out trails of lost tracks
            del trails[tid][:1]
            if not trails[tid]:
                del trails[tid]
        if flow:
            flow_ref, flow_n = flow.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), T, ids, dets.corners,
                                           centers_ref)
        else:
            flow_ref, flow_n = np.full((len(dets), 2), np.nan), np.zeros(len(dets))
        if len(dets):
            pos = np.where(np.isfinite(flow_ref), flow_ref, centers_ref)
            world = calibration.apply_h(cal.H, pos)
            cf = dets.centers
            n = len(dets)
            rows.append(np.column_stack([
                np.full(n, i), ids, dets.cls, dets.conf, cf, centers_ref, world, corners_ref.reshape(n, 8),
                flow_ref, flow_n,
            ]))

        state["done"] += 1
        done = state["done"]
        now = time.monotonic()
        if done % 5 == 0 or done == total:
            reporter.progress("analyzing", done, total, fps=done / max(now - t_start, 1e-6),
                              tracks=len(seen_tracks))
        if now - state["last_preview"] > preview_every_s:
            state["last_preview"] = now
            T_inv = np.linalg.inv(T)
            if cal.paths_px:
                lines = []
                for g, d in zip(cal.path_geoms, cal.paths_px):
                    lines.append((g.name, transform_points(T_inv, d["px"]), (255, 200, 0)))
                    lines += [("", transform_points(T_inv, e), (160, 160, 160)) for e in d["corridor_px"]]
            else:
                lines = [(n, transform_points(T_inv, np.array(l, float)), col) for n, l, col in
                         (("A", cfg.calibration.line_a, (255, 200, 0)), ("B", cfg.calibration.line_b, (255, 0, 200)))
                         if l]
            text = f"frame {i}  t {i / fps:.2f} s  {len(dets)} det  {len(live)} tracked  {len(seen_tracks)} tracks so far"
            reporter.preview(_preview_jpeg(frame, dets.corners, ids, trails, lines, text, rotation=cfg.rotation))

    def track_chunk(chunk, dets_list) -> None:
        for (i, f), d in zip(chunk, dets_list):
            per_frame(i, f, d)

    # Three-stage pipeline so the GPU isn't idle while the CPU decodes and tracks:
    #   decoder thread -> batch of frames -> detection (GPU; each frame detected on its own)
    #   -> per_frame() on a single worker thread, strictly in frame order.
    batch = max(1, cfg.detection.batch)
    frames = _Prefetch(iter_frames(cfg.source, start, end, pts_check), maxsize=2 * batch)
    pending: deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="track") as post:
        try:
            for chunk in _chunks(frames, batch):
                dets_list = detector.detect_batch([f for _, f in chunk])
                pending.append(post.submit(track_chunk, chunk, dets_list))
                while len(pending) > 1:  # bound memory: one batch tracking, at most one waiting
                    pending.popleft().result()
                if reporter.cancelled():
                    raise InterruptedError("cancelled")
            while pending:
                pending.popleft().result()
        finally:
            frames.close()
            for f in pending:
                f.cancel()
    done, n_dets = state["done"], state["n_dets"]

    cols = ["frame", "track_id", "cls", "conf", "cx", "cy", "rx", "ry", "x_m", "y_m"] + \
           [f"c{k}{a}" for k in range(4) for a in "xy"] + ["frx", "fry", "flow_n"]
    df = pd.DataFrame(np.concatenate(rows) if rows else np.zeros((0, len(cols))), columns=cols)
    for c in ("frame", "track_id", "cls", "flow_n"):
        df[c] = df[c].astype(np.int64)
    df["t"] = df["frame"] / fps
    df = df[df["track_id"] >= 0].reset_index(drop=True)
    df.to_parquet(out / "tracks.parquet", index=False)
    np.savez_compressed(out / "transforms.npz", frames=np.array(T_frames, np.int64),
                        T=np.array(T_all) if T_all else np.zeros((0, 3, 3)))
    if stab_rows:
        pd.DataFrame(stab_rows, columns=["frame", "ok", "inliers", "source"]).to_csv(
            out / "stabilization.csv", index=False)
    if sampler:
        violations.save_samples(out, violations.lights_key(cfg.calibration, cfg.violations), light_frames, light_counts)

    reporter.progress("computing speeds", 0, 1)
    results, renders = compute_speeds(df, fps, cal.line_a_world, cal.line_b_world, cfg.speed,
                                           paths=cal.path_geoms)
    budget = margin_of_error(cfg, results, renders, cal, fps)
    viol = find_violations(cfg, cal, df, renders, fps, out, reporter)
    n_fail = sum(1 for r in stab_rows if not r[1])
    if pts_anomalies:
        reporter.log(f"{len(pts_anomalies)} frames have container timestamps off the expected grid "
                     "(possible dropped/duplicated frames)", "warning")

    summary = write_results(out, cfg, results, info, fps, cal, timing_msgs, budget, viol, {
        "frames": done, "registration_failures": n_fail,
        "detections_per_frame": n_dets / max(done, 1), **track_diagnostics(df, renders),
        "camera_drift": camera_drift(drift, cal.m_per_px),
        "detector": detector_info,
        "pts_anomalies": len(pts_anomalies), "elapsed_s": time.monotonic() - t_start,
    })
    reporter.progress("done", 1, 1)
    return summary


def find_violations(cfg: ProjectConfig, cal, df: pd.DataFrame, renders: dict, fps: float, out: Path,
                    reporter: Optional[Reporter] = None, sample: bool = True) -> Optional[dict]:
    """Red-light running and solid-line crossings (see violations.py); None when none are set up.
    With `sample`, lights added or moved since the analysis are read from the video (decode only)."""
    c = cfg.calibration
    if not (c.red_lights or c.solid_lines):
        return None
    samples = violations.load_samples(out, violations.lights_key(c, cfg.violations))
    if samples is None and sample and violations.needs_samples(c):
        samples = violations.sample_lights(cfg, out, reporter)
    frames = np.load(out / "transforms.npz")["frames"]
    return violations.detect(cfg, cal, df, renders, fps, samples, frames)


def margin_of_error(cfg: ProjectConfig, results: list[dict], renders: dict, cal, fps: float) -> dict:
    """Per-vehicle uncertainty budget (fills the speed_*_u_kmh fields); see uncertainty.py."""
    return uncertainty.annotate(results, renders, cfg.calibration, cal, fps, cfg.uncertainty,
                                cfg.speed.momentary_window_s)


def write_results(out: Path, cfg: ProjectConfig, results: list[dict], info, fps: float, cal,
                  timing_msgs: list[str], budget: Optional[dict], viol: Optional[dict], run: dict) -> dict:
    cols = ["track_id", "class", "path", "direction", "coverage_pct", "t_entry_s", "t_exit_s", "frame_entry", "frame_exit", "dt_s",
            "path_length_m", "chord_m", "speed_kmh", "speed_avg_u_kmh", "speed_avg_u_lower_kmh",
            "speed_max_kmh", "speed_max_u_kmh", "speed_max_u_lower_kmh", "speed_min_kmh", "speed_sigma_kmh",
            "speed_avg_se_kmh", "u_pct", "u_cal_pct", "u_parallax_pct",
            "t_max_s", "d_max_m", "speed_trend_pct", "speed_line_kmh", "disagreement_pct",
            "fit_residual_m", "accel_mps2", "max_gap_s", "n_points", "mean_conf", "valid", "flags", "violations"]
    # each measured vehicle lists its violations, e.g. "red light (Main St)"
    tags: dict[int, list[str]] = {}
    for e in (viol or {}).get("red_light", []):
        if e["violation"]:
            tags.setdefault(e["track_id"], []).append(f"red light ({e['name']})")
    for e in (viol or {}).get("solid_line", []):
        tags.setdefault(e["track_id"], []).append(f"solid line ({e['name']})")
    for r in results:
        r["violations"] = tags.get(r["track_id"], [])
    if viol:
        violations.write_csv(out, viol)
    elif (out / "violations.csv").exists():
        (out / "violations.csv").unlink()
    table = pd.DataFrame(results, columns=cols)
    table["violations"] = table["violations"].map(lambda v: "; ".join(v) if isinstance(v, list) else "")
    table.to_csv(out / "results.csv", index=False, float_format="%.4f")
    valid = [r for r in results if r["valid"]]
    summary = {
        "source": cfg.source,
        "capture_fps": fps,
        "container_fps": info.container_fps,
        "nb_frames": info.nb_frames,
        "calibration": cal.summary(),
        "timing_warnings": timing_msgs,
        "run": run,
        "stats": summarize([r["speed_kmh"] for r in valid], cfg.stats, n_total=len(results)),
        "stats_max": summarize([r["speed_max_kmh"] for r in valid], cfg.stats, n_total=len(results)),
        "by_direction": {
            d: summarize([r["speed_kmh"] for r in valid if r["direction"] == d], cfg.stats)
            for d in ("A->B", "B->A")
        },
        "by_direction_max": {
            d: summarize([r["speed_max_kmh"] for r in valid if r["direction"] == d], cfg.stats)
            for d in ("A->B", "B->A")
        },
        # one set of statistics per path of motion (e.g. per road of an intersection)
        "by_path": {
            name: {
                "avg": summarize([r["speed_kmh"] for r in valid if r.get("path") == name], cfg.stats,
                                 n_total=sum(r.get("path") == name for r in results)),
                "max": summarize([r["speed_max_kmh"] for r in valid if r.get("path") == name], cfg.stats,
                                 n_total=sum(r.get("path") == name for r in results)),
            }
            for name in [p.name for p in cfg.calibration.paths if len(p.points) >= 2]
        },
        "calibration_check": calibration_check(results),
        "uncertainty": budget,
        "violations": viol,
        "vehicles": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def recompute(cfg: ProjectConfig, out_dir: str | Path) -> dict:
    """Re-run speed estimation from stored tracks (e.g. after changing lines or thresholds)."""
    out = Path(out_dir)
    info = probe(cfg.source)
    fps = resolve_capture_fps(cfg, info, Reporter())
    cal = calibration.compute(cfg.calibration, (info.width, info.height))
    df = pd.read_parquet(out / "tracks.parquet")
    world = calibration.apply_h(cal.H, reference_positions(df))
    df["x_m"], df["y_m"] = world[:, 0], world[:, 1]
    results, renders = compute_speeds(df, fps, cal.line_a_world, cal.line_b_world, cfg.speed,
                                           paths=cal.path_geoms)
    prev = json.loads((out / "results.json").read_text(encoding="utf-8")) if (out / "results.json").exists() else {}
    run = {**prev.get("run", {}), **track_diagnostics(df, renders)}
    budget = margin_of_error(cfg, results, renders, cal, fps)
    viol = find_violations(cfg, cal, df, renders, fps, out)
    return write_results(out, cfg, results, info, fps, cal, check_timing(info, fps, cfg.real_duration_s),
                         budget, viol, run)


def camera_drift(drift_px: list[float], m_per_px: float) -> Optional[dict]:
    """Camera motion relative to the reference frame; None when stabilization was off."""
    if not drift_px:
        return None
    a = np.asarray(drift_px)
    return {"max_px": float(a.max()), "p95_px": float(np.percentile(a, 95)),
            "max_m": float(a.max() * m_per_px), "frames": len(a)}


def track_diagnostics(df: pd.DataFrame, renders: dict) -> dict:
    """Where tracks were lost: explains an empty result table."""
    crossed = [r.crossed for r in renders.values()]
    return {
        "tracks": int(df["track_id"].nunique()) if len(df) else 0,
        "tracks_long_enough": len(renders),
        "tracks_crossing_a_only": crossed.count("A"),
        "tracks_crossing_b_only": crossed.count("B"),
        "tracks_crossing_neither": crossed.count(""),
        "tracks_crossing_both": crossed.count("AB"),
        "tracks_too_short": crossed.count("short"),  # path mode: in the corridor, too little of the path
    }
