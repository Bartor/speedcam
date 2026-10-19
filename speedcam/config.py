"""Project configuration schema.

The same schema is used for `project.json` (web app) and the YAML config the CLI
accepts. All pixel coordinates are in *reference-frame* pixels of the original
video (full resolution).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

Pt = tuple[float, float]


class Segment(BaseModel):
    """Two ground points with a measured distance between them."""

    a: Pt
    b: Pt
    meters: float


class ControlPoint(BaseModel):
    """A ground point with known world coordinates (metres, any local frame)."""

    px: Pt
    world: Pt


class PolygonDistance(BaseModel):
    """Measured ground distance between polygon vertices i and j (a side or a diagonal)."""

    i: int
    j: int
    meters: float


class Polygon(BaseModel):
    """Flat ground polygon with measured sides/diagonals. Its shape is solved from the distances
    (needs >= 2N-3 of them) and gives the image -> ground homography (perspective correction)."""

    points: list[Pt] = Field(default_factory=list)
    distances: list[PolygonDistance] = Field(default_factory=list)


class MotionPath(BaseModel):
    """A curve along which traffic moves: points in reference px, in the direction called A->B."""

    name: str = "Path"
    points: list[Pt] = Field(default_factory=list)
    corridor_m: float = 6.0  # half-width of the corridor around the path, metres


class TrafficLight(BaseModel):
    """A signal head: the circle (reference px) around its lamps whose colour is read each frame."""

    px: Pt
    radius_px: float = 12.0


class StopLine(BaseModel):
    """A line (2+ points, reference px) that must not be crossed while the light is red."""

    points: list[Pt] = Field(default_factory=list)
    # which crossings count: "forward" = from the left of the line (as drawn, first point to last)
    # to its right, "reverse" the other way, "both" either
    direction: Literal["both", "forward", "reverse"] = "both"


class RedLightSetup(BaseModel):
    """One signal group: its lights (any number of heads showing the same phase) and stop lines."""

    name: str = "Light 1"
    lights: list[TrafficLight] = Field(default_factory=list)
    stop_lines: list[StopLine] = Field(default_factory=list)
    # "colour": read the phase from the lights in the video; "manual": red during red_intervals
    state_source: Literal["colour", "manual"] = "colour"
    red_intervals: list[tuple[float, float]] = Field(default_factory=list)  # seconds of real time


class SolidLine(BaseModel):
    """A solid road marking (2+ points, reference px): any vehicle crossing it is reported."""

    name: str = "Solid line 1"
    points: list[Pt] = Field(default_factory=list)


class Calibration(BaseModel):
    reference_frame: int = 0
    segments: list[Segment] = Field(default_factory=list)
    points: list[ControlPoint] = Field(default_factory=list)
    polygon: Optional[Polygon] = None
    line_a: Optional[tuple[Pt, Pt]] = None
    line_b: Optional[tuple[Pt, Pt]] = None
    roi: Optional[list[Pt]] = None
    # Paths of motion (e.g. one per road of an intersection). When any is set they replace lines
    # A/B: each vehicle is measured along the path it covers most, over all of it inside the corridor.
    paths: list["MotionPath"] = Field(default_factory=list)
    # Only measure on the part of the path inside the calibrated area (outline of the calibration
    # points, plus this margin): far from where it was measured a calibration is extrapolated.
    path_clip_to_calibration: bool = True
    path_calibration_margin_m: float = 5.0
    # violation checks (see violations.py): red-light setups and solid lines
    red_lights: list[RedLightSetup] = Field(default_factory=list)
    solid_lines: list[SolidLine] = Field(default_factory=list)


class DetectionSettings(BaseModel):
    type: Literal["yolo", "blob"] = "yolo"
    # COCO model handles oblique and nadir views; yolo11*-obb.pt (DOTA) only suits true top-down footage.
    # "auto" runs a COCO model (good for oblique views) and a DOTA oriented-box model (good for
    # straight-down views); see AutoDetector. Or name one Ultralytics model, e.g. "yolo11x.pt".
    model: str = "auto"
    # candidates for "auto": COCO (oblique), DOTA oriented boxes (straight down), and the
    # VisDrone-trained model when its weights are installed (skipped otherwise)
    auto_models: list[str] = Field(default_factory=lambda: ["yolo11x.pt", "yolo11x-obb.pt", "visdrone"])

    @field_validator("auto_models")
    @classmethod
    def _add_visdrone(cls, v: list[str]) -> list[str]:
        # projects saved before VisDrone was a candidate stored the old two-model default
        return [*v, "visdrone"] if v == ["yolo11x.pt", "yolo11x-obb.pt"] else v
    # the top-down (OBB) model only works straight down; it's preferred whenever its score reaches
    # this share of the best model's score. Measured on 10 real clips: oblique 0.00-0.22,
    # straight down 0.54
    auto_topdown_share: float = 0.35
    # fill-in model when the top-down model is primary. On the straight-down clip 0457, VisDrone
    # fill-ins broke tracks of cars under trees: 17 valid vehicles ("gap" on 15) vs 26 with COCO
    auto_topdown_fill: str = "yolo11x.pt"
    obb_imgsz: int = 1024  # DOTA models are trained at 1024; larger sizes detected far less here
    auto_sample_frames: int = 12  # frames sampled across the clip to pick the primary model
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 1280
    # 0 = run on the whole frame (resized to imgsz); otherwise tile size in source px.
    tile: int = 0
    tile_overlap: float = 0.2
    # frames detected per GPU call. Each frame is still detected on its own (no context between
    # frames); batching only keeps the GPU busy. Tracking always runs frame by frame, in order.
    batch: int = 8
    device: Optional[str] = None
    # FP16 is ~45% faster but not repeatable run to run, and moves speeds by up to ~0.2 km/h
    # (box coordinates are quantized to ~1.5-3 px at 4K). FP32 gives identical results on reruns.
    half: bool = False
    classes: list[str] = Field(default_factory=lambda: ["car", "truck", "bus", "motorcycle"])
    # "blob" detector (synthetic tests): minimum saturation and component area.
    blob_min_saturation: int = 120
    blob_min_area: int = 200


class TrackingSettings(BaseModel):
    activation_threshold: float = 0.25
    lost_buffer_s: float = 1.0
    match_threshold: float = 0.8
    min_consecutive_frames: int = 3


class FlowSettings(BaseModel):
    """Vehicle positions from optical flow on the vehicle's texture, anchored to the box centre
    (see flow.py). Off = box centres only."""

    enabled: bool = True
    # complementary filter time constant: flow gives the frame-to-frame motion, the box centre
    # the long-run position. Longer = smoother, but slower to absorb flow drift.
    anchor_s: float = 1.0
    max_points: int = 100  # features per vehicle (more = steadier fit; accuracy over speed)
    min_points: int = 6  # fewer RANSAC inliers = fall back to the box centre
    reseed_below: float = 0.6  # top up features when fewer than this share of max_points survive
    inner: float = 0.9  # features are picked inside the box shrunk by this factor (and by the LK window)
    gate: float = 1.15  # tracked features must stay inside the box grown by this factor
    quality: float = 0.01
    win: int = 15  # LK window, px
    levels: int = 3
    crop_margin_px: int = 96  # LK runs on a crop around each vehicle's features: must exceed its motion per frame
    fb_max_px: float = 0.5  # forward-backward LK error
    lk_iters: int = 100
    lk_eps: float = 0.001
    ransac_iters: int = 1000
    ransac_px: float = 0.5
    max_coast_s: float = 0.5  # keep following a vehicle on flow alone through detector gaps
    # features must move like the box centre did over the last `velocity_s` (within the larger of
    # motion_gate_px and motion_gate x speed): drops features stuck on wires, poles or the road
    velocity_s: float = 0.5
    motion_gate: float = 0.3
    motion_gate_px: float = 1.5
    max_offset: float = 0.15  # flow further than this share of the box diagonal from the box = reset


class StabilizationSettings(BaseModel):
    enabled: bool = True
    model: Literal["similarity", "affine", "homography"] = "similarity"
    # klt: track reference corners with pyramidal Lucas-Kanade (fast, sub-pixel);
    # orb/sift: descriptor matching. KLT falls back to ORB when it loses lock.
    features: Literal["klt", "orb", "sift"] = "klt"
    max_features: int = 3000
    work_width: int = 1600
    min_inliers: int = 40
    ransac_thresh: float = 1.5
    vehicle_mask_pad: float = 0.15


class SpeedSettings(BaseModel):
    smooth_window_s: float = 0.25
    smooth_poly: int = 2
    arc_dt_s: float = 0.05
    max_accel: float = 8.0
    # max deviation from a constant-acceleration fit; box-centre wobble on real footage is
    # ~0.2-0.4 m, an ID switch to another car jumps by a car length
    max_residual_m: float = 0.5
    max_disagreement: float = 0.02
    disagreement_invalidates: bool = False
    max_gap_s: float = 0.3
    min_points: int = 10
    # window for momentary speed (max/min, profiles). Shorter follows changes faster but adds noise.
    # Measured on 10 4K clips with flow positions: 0.25 s ~ +-0.28 km/h (max +0.16 km/h high),
    # 0.5 s ~ +-0.18 km/h (max unbiased, +0.02); box centres alone: +-0.57 / +-0.29.
    momentary_window_s: float = 0.5
    line_extend: float = 0.1
    # path mode: a vehicle must cover at least this much of the path (or half of it, if shorter)
    path_min_cover_m: float = 10.0


class ViolationSettings(BaseModel):
    """How red-light running and solid-line crossings are decided (see violations.py)."""

    # light colour: a pixel in a light's circle counts when it is this bright and saturated
    # (HSV, 0-255); the colour with the most such pixels is the light's state on that frame
    light_min_value: int = 150
    light_min_saturation: int = 70
    light_min_pixels: int = 3
    # LED lights flicker against the camera shutter and drop out on single frames: a frame with
    # no colour keeps the last state for up to this long
    light_hold_s: float = 1.0
    light_smooth_s: float = 0.3  # majority vote over this window removes single-frame misreads
    amber_is_red: bool = False
    # a vehicle entering this soon after the light turned red isn't reported (reaction tolerance)
    red_grace_s: float = 0.0
    # a crossing counts once the vehicle's position is this far on both sides of the line: the
    # tracked position wobbles by a few tens of cm, so a vehicle driving along a line doesn't
    # "cross" it back and forth
    hysteresis_m: float = 0.5
    # solid lines: "centre" = the vehicle's centre crosses (a lane change); "wheels" = any part of
    # its box gets over the line by overlap_m for min_duration_s (straddling); boxes are only the
    # vehicle's footprint when filmed straight down. On 0457 (straight down, no lane changes)
    # boxes of normally driven vehicles reached up to 0.29 m over the lane line, so the default
    # leaves a margin above that
    solid_mode: Literal["centre", "wheels"] = "centre"
    overlap_m: float = 0.5
    min_duration_s: float = 0.5


class RenderSettings(BaseModel):
    mode: Literal["stabilized", "original"] = "stabilized"
    output_fps: Optional[float] = None  # None = native (capture_fps)
    playback_speed: float = 1.0  # 1.0 = real time, 0.25 = 4x slow motion
    scale: float = 0.5
    lut: Optional[str] = None  # .cube file applied before drawing
    encoder: Literal["auto", "nvenc", "videotoolbox", "x264"] = "auto"
    quality: int = 20
    trail_s: float = 1.0
    label_hold_s: float = 2.5  # how long a vehicle keeps its "max" label after finishing
    speed_limit_kmh: float = 50.0
    # presentation (social media): side panel listing every vehicle's max speed as it finishes
    panel: Literal["none", "right", "left"] = "right"
    panel_layout: Literal["overlay", "extend"] = "overlay"  # extend = video + panel side by side (wider output)
    panel_width: float = 0.27  # fraction of the video width
    panel_sort: Literal["latest", "fastest"] = "latest"
    panel_rows: int = 7
    title: str = ""
    subtitle: str = ""
    footer: str = ""
    show_hud: bool = False  # technical time/frame line in the corner
    show_margin: bool = True  # each panel row shows the max speed's 95 % margin of error
    show_unmeasured: bool = False  # boxes on vehicles that never cross both lines (parked cars etc.)
    font: Optional[str] = None  # .ttf for text; default Segoe UI / DejaVu Sans


class StatsSettings(BaseModel):
    permitted_error_kmh: float = 2.0
    confidence: float = 0.95


class UncertaintySettings(BaseModel):
    """Inputs of the per-vehicle margin of error (see uncertainty.py)."""

    # standard uncertainty (1 sigma) of each typed calibration distance / control-point position:
    # tape ~0.02 m, orthophoto ~0.10 m
    distance_tolerance_m: float = 0.05
    click_tolerance_px: float = 0.5  # 1 sigma of each clicked calibration point
    # drone height above the road (m). Known: the uncorrected height-parallax bias is bounded
    # and declared. Unknown: the margin says parallax is not covered.
    drone_height_m: Optional[float] = None
    # height of the tracked box centre above the road per class (m); parallax scales speeds by H/(H-h)
    point_height_m: dict[str, float] = Field(
        default_factory=lambda: {"car": 0.8, "motorcycle": 0.7, "truck": 1.8, "bus": 1.5})
    mc_samples: int = 200


class ProjectConfig(BaseModel):
    source: str
    capture_fps: Optional[float] = None
    real_duration_s: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    calibration: Calibration = Field(default_factory=Calibration)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    flow: FlowSettings = Field(default_factory=FlowSettings)
    stabilization: StabilizationSettings = Field(default_factory=StabilizationSettings)
    speed: SpeedSettings = Field(default_factory=SpeedSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    stats: StatsSettings = Field(default_factory=StatsSettings)
    uncertainty: UncertaintySettings = Field(default_factory=UncertaintySettings)
    violations: ViolationSettings = Field(default_factory=ViolationSettings)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    cfg = ProjectConfig.model_validate(data)
    src = Path(cfg.source)
    if not src.is_absolute():
        cfg.source = str((path.parent / src).resolve())
    return cfg


def save_config(cfg: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    data = cfg.model_dump(mode="json")
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
