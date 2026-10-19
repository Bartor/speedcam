# speedcam

Measures vehicle speeds from nadir (straight-down) drone footage and exports a
per-vehicle CSV/JSON and an annotated MP4. It's a local app: a Python analysis
package with a CLI, plus a FastAPI server and a browser UI that acts as a client of
that package.

## Quick start

Needs Python 3.11+, the ffmpeg/ffprobe CLI on `PATH`, and Node.js 18+ (only to build
the UI). Then, from this folder:

```bash
speedcam            # Windows: speedcam.cmd (or double-click it); macOS/Linux: ./speedcam.sh
```

The first run takes a few minutes. It creates `.venv`, installs dependencies (with
CUDA PyTorch when an NVIDIA GPU is present), builds the web UI, then starts the app
and opens the browser. After that it starts straight away, and redoes a setup step
only when it's needed: `pyproject.toml` changed, or the UI sources are newer than
the build. No need to activate the venv.

```bash
speedcam --shortcut          # desktop shortcut (Windows)
speedcam serve --jobs 3      # any CLI command works through the launcher
speedcam analyze p.yaml --out out/
speedcam --setup-only        # install/update without starting
```

Manual setup, if you prefer:

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -e ".[dev]"
cd web && npm install && npm run build   # builds the UI into speedcam/server/static
```

Ultralytics downloads the YOLO weights the first time they are used. The default
detector is **Auto**, which works at any camera angle. It has up to three candidates:
COCO `yolo11x.pt` (strong on oblique views), DOTA `yolo11x-obb.pt` (strong straight
down, useless otherwise) and the VisDrone-trained model below, when its weights are
installed. For each clip it samples 12 frames and scores each model by the sum of its
detection confidences. The top-down model becomes the primary whenever it scores at
least 35% of the best model (`auto_topdown_share`), since it holds cars under trees
better straight down; otherwise the best scorer does. The runner-up fills in (COCO when
the top-down model leads, see below): on every frame the primary's boxes are kept, and
the second model only adds vehicles the primary missed, so each vehicle keeps a
consistent box. It costs about twice the GPU
time of one model. On ten real clips the top-down share was 0.00-0.22 on oblique
footage and 0.54 on the straight-down clip (0457), which got the top-down model plus
COCO; six oblique clips got VisDrone plus COCO, three got COCO plus VisDrone. Full
analyses with this Auto: 0455 39 valid vehicles (old Auto 36), 0446 25 (25); 0457 gets
the same two models as the old Auto (26). The same cars' speeds agree within 0.4%. Filling the
top-down model in with VisDrone instead cost 0457 nine vehicles: its extra boxes on
cars under trees broke their tracks.
Any model can still be chosen on its own on the Process screen.

**VisDrone (trained)** is an optional third detector: `yolo11x` fine-tuned on
VisDrone2019-DET (drone footage of city traffic at many heights and angles), with
`tools/train_visdrone.py`. It took 6.5 h on an RTX 3080 Ti at 1280 px, batch 2, and
scored car mAP50 0.908 (all 10 VisDrone classes: 0.619). Weights go in
`~/speedcam-models/yolo11x-visdrone.pt`. Compared with Auto on full analyses of
three real clips:

| Clip | Valid vehicles, Auto → VisDrone | Same car's speed | Analysis time |
|---|---|---|---|
| oblique (0455) | 36 → **39** | ±0.1% | 3.4 → 2.9 min |
| oblique (0446) | 25 → 25 | ±0.1% | 1.7 → 1.1 min |
| straight down, trees (0457) | **26** → 18 | ±0.3% | 3.3 → 1.8 min |

The detector doesn't change any vehicle's speed, only which vehicles get measured.
VisDrone finds more distant cars in oblique views and is faster, since it's a single
model. Straight down, cars partly under trees drop out longer, and the top-down model
in Auto holds them better. That's why Auto now picks between all three per clip.

Boxes and labels in the overlay and the render stay steady through detector gaps
of up to 1 s: missed frames are interpolated from the detections on either side.

## Web app

```bash
speedcam serve            # http://127.0.0.1:8000, projects stored in ~/speedcam-projects
```

0. **Projects:** the start page lists every project in the data folder with a
   thumbnail, its status and headline stats (valid/measured, mean, p85). Click a
   project to resume on the screen you left it. Projects can be renamed, deleted
   (the source video is never touched), or relinked when the video has moved. The
   index and job history live in `~/speedcam-projects/speedcam.db` (SQLite). The
   project folders are the source of truth, so deleting the database only loses
   job history. Decoded full-res frames, thumbnails and overlay data are cached
   under each project's `cache/` folder.
1. **Import:** pick a video on disk. Nothing is uploaded. The server builds a 1080p
   H.264 proxy for scrubbing. Confirm `capture_fps`. **DJI slow-motion clips** are
   stored at e.g. 30 fps, but were captured at 100/120 fps. Real time is always
   `frame / capture_fps`.
2. **Calibrate:** choose a reference frame. For any view that isn't straight down,
   use **Polygon**: click 4+ flat ground points, then type measured sides and
   diagonals. A polygon with N corners needs 2N−3 distances to fix its shape (a
   quadrilateral: 4 sides + 1 diagonal). The shape is solved by least squares, and
   the image → ground homography (perspective correction) comes from it. Extra
   distances act as checks and show their residuals. **Rectangle** is the shortcut
   for a known rectangle: click 4 ground corners of a known rectangle (e.g. lane width ×
   a run of lane markings) in order and type the two edge lengths. That gives a
   homography. Add distance segments as independent checks. A straight-down camera
   can use distance segments alone (uniform scale). If a check distance is off by
   more than 5%, the calibration is refused, because the model doesn't fit the view. Draw lines A and B, plus an optional ROI. Scale,
   residuals and the A–B gap update live. All clicks are made on full-resolution
   stills decoded from the original.
   **Measuring: lines A/B or paths of motion.** Lines A and B measure each vehicle
   between two crossings. With the **Path** tool you instead click points along a road
   in the direction of travel. A smooth curve on the ground is fitted, with a corridor
   (± metres) around it, and you can draw several, e.g. one per road of an
   intersection. Each vehicle is measured along the path it covers most, over all of
   it inside the corridor, so a vehicle is never counted twice. Speed is the rate of
   distance *along* the path, so lane changes don't inflate it. On bends it's
   corrected for the vehicle's lane: an outer lane 3.5 m off a 50 m-radius curve
   would otherwise read ~7% slow. By default only the part of each path inside the
   calibrated area (plus 5 m) is measured. Far from where the calibration was
   measured it's extrapolated; on a test clip that cost 12 km/h. The rest is drawn
   dashed. Results get a Path column and per-path statistics.
   **Violations: red lights and solid lines.** Under *Violations*, the **Light** tool
   marks a traffic light (a circle around its lamps) and **Stop line** draws the line
   that mustn't be crossed on red. Each red-light setup has any number of lights
   showing the same signal and any number of stop lines, and a clip can have any
   number of setups. The light's phase on every frame is read from the colour of
   the bright, saturated pixels in its circle. Dark frames (LED flicker against the
   shutter) keep the last state for up to 1 s, and single-frame misreads are voted
   out. The side panel shows what each light reads on the current frame. When the
   lamps aren't visible (e.g. filmed from behind or straight down), type the red
   intervals instead. A vehicle runs the red when the front of its box reaches a
   stop line, in the counted direction (both, or along/against the arrow), while
   the phase is red. Options: amber counts as red, and a grace period after the
   change. **Solid line** draws a road marking: every vehicle crossing it anywhere
   along it is reported. *Centre crosses* (default) catches lane changes; *Any part
   over* also catches straddling, with a box at least 0.5 m over for 0.5 s (needs
   straight-down footage, where the box is the vehicle's footprint). On 0457,
   normally driven vehicles' boxes reached 0.29 m over the lane line. A crossing
   only counts once the vehicle is 0.5 m clear on both sides, so tracking wobble
   along a line isn't reported. These checks use the stored tracks: draw or change
   them after the analysis and press **Recompute**. Lights added or moved since then
   are read from the video (decode only, no GPU). A project can have only
   violation checks, without lines A/B or paths.
3. **Process:** choose **Camera: Hovering** (stabilize every frame) or **Still**
   (skip stabilization, faster). A preview run with stabilization on measures camera
   drift against the reference frame, in px and in cm on the ground, and says whether
   skipping is safe. **Preview tracking** runs detection, stabilization and tracking on a
   few seconds and streams annotated frames: boxes coloured by track ID, trails and
   lines A/B. It also reports how many tracks crossed each line. Use it to check
   settings before the full run, which streams the same tracking preview and can be
   cancelled.
4. **Results:** the proxy video, with a live tracking overlay drawn in the browser
   (boxes, trails, momentary speed between the lines, then max and average),
   sits beside the vehicle table. No render is needed for review. Each vehicle has a
   **max** (highest momentary speed between A and B) and an **average** (A→B).
   Click a row to seek to it and plot its **speed along the section**: the momentary
   speed curve with max/min marked, a ±2σ noise band, and the average and speed-limit
   lines. Statistics and the histogram switch between max and average. They include
   mean, p85 and p95 with confidence intervals, plus the ITE sample size needed
   against the valid n. A **calibration check** warns when every vehicle's speed
   rises or falls by the same share along A→B. Drivers vary but geometry doesn't, so
   a shared trend means a calibration error that biases max speeds. If nothing was
   measured, the page says where tracks were lost. Export CSV, JSON and MP4.
   **Recompute** re-runs the speed and violation steps, without re-detecting.
   With violation checks, a **Violations** panel shows each light's phase timeline
   (red/amber/green, with the runs marked) and lists every red-light run and
   solid-line crossing with its time, vehicle and speed. Click one to watch it. The
   overlay colours stop lines and lights by the current phase and tags violating
   vehicles, and `violations.csv` lists them. The vehicle table marks them too.
5. **Render:** an annotated MP4 for sharing. Vehicles get speed pills (live
   momentary speed between the lines, then max, coloured against the limit). An
   optional side panel (left or right, over the 16:9 frame or beside it) shows:
   a title and subtitle, a speed-limit sign, running totals (vehicles, over the
   limit, top speed, p85), the latest vehicle's speed curve, and the list of
   vehicles with their max and average speeds, latest or fastest first. Stop lines
   and lights are drawn in the phase's colour, and violating vehicles get a red box
   and a RED LIGHT / SOLID LINE pill. A footer is
   optional. Text uses Segoe UI and Bahnschrift (Windows) or DejaVu Sans, or any
   `.ttf` set in `render.font`.

The API lives under `/api` (OpenAPI at `/docs`). It covers projects, frames,
`proxy.mp4` with range requests, calibration, analyze, render, results, job
events (SSE) and cancel.

For frontend development, run `npm run dev` in `web/`. Vite proxies `/api` to
port 8000.

## CLI

```bash
speedcam probe clip.mp4                       # metadata + slow-motion warnings
speedcam init clip.mp4 --out p.yaml --capture-fps 120
speedcam calibrate p.yaml --frame 0           # OpenCV click tool
speedcam analyze p.yaml --out out/
speedcam recompute p.yaml --out out/          # speeds only, from stored tracks
speedcam render p.yaml --out out/ --fps 30 --playback 0.25 --lut dlogm.cube
speedcam synth --speed 50                     # synthetic validation clip + config
```

`p.yaml` uses the same schema as the web app's `project.json` (`speedcam/config.py`).

## How it measures

| Step | Implementation |
| --- | --- |
| Timing | `frame / capture_fps`. Container PTS is checked for gaps; sidecar DJI `.SRT` `DiffTime` is used to suggest the capture rate. |
| Calibration | Polygon: shape solved from sides + diagonals, then homography. Segments only: uniform scale (warns when all segments are parallel). 2–3 control points: similarity. ≥4: homography. Every segment is reported as a check residual. |
| Stabilization | Reference corners are tracked into each frame with pyramidal Lucas–Kanade, then fitted with RANSAC (similarity by default). Vehicles are masked. Falls back to ORB vs. reference, then ORB vs. the previous frame (chained), then the last transform (logged). Frames are never warped for analysis. |
| Detection | Ultralytics YOLO: Auto (COCO, DOTA-OBB and VisDrone; a primary per clip plus fill-in) or a single model, oriented or axis-aligned boxes, optional tiling with NMS across tiles. DOTA `small/large vehicle` map to car/truck. |
| Tracking | ByteTrack (supervision) on reference-frame boxes. |
| Vehicle position | Optical flow on the vehicle's own texture (`flow.*`): corners inside each box (away from its edge and other vehicles) are tracked with forward-backward Lucas–Kanade, mapped to the reference frame and fitted with a RANSAC similarity; features that don't move like the box did over the last 0.5 s (wires, poles, road) are dropped. The position follows the flow frame to frame and is pulled back to the box centre with a 1 s complementary filter, so it can't drift; with too few features it falls back to the box centre. Stored as `frx/fry` next to the box centre `rx/ry` in `tracks.parquet`. On 10 real 4K clips this cut momentary-speed noise from ±0.57 to ±0.28 km/h at 0.25 s and ±0.29 to ±0.18 at 0.5 s, the max-speed bias at 0.5 s from +0.28 to +0.02 km/h and the arc-length fit residual from 2.1 to 1.0 cm; average speeds agree with box centres to within 0.3 km/h on most clips. Costs ~15-30 ms CPU per frame on the tracking thread. |
| Speed | Savitzky–Golay smoothing of the world trajectory. **Momentary:** |velocity| from a local quadratic fit over 0.5 s (`speed.momentary_window_s`; measured noise about ±0.18 km/h on 4K footage with flow positions, ±0.28 at 0.25 s; box centres alone ±0.29 / ±0.57), giving max/min, where the max happened, and a profile. **Average:** linear regression of arc length vs. time between the sub-frame crossings of A and B. **Audit:** chord / Δt. Flags: `disagree` (>2%), `accel` (>8 m/s²), `residual` (jumps ⇒ ID switch), `gap`, `multi_cross`. |
| Paths | Centripetal Catmull-Rom spline through the clicked points on the ground plane; positions projected to (distance along, offset across); speed = d(distance along, curvature-corrected for the offset)/dt; each vehicle on the path it covers most; measured only inside the calibrated area by default. |
| Stats | t-interval for the mean, bootstrap CIs for p85/p95, ITE `n = (S·K/E)²·(2+U²)/2`. |

Outputs in the project or `--out` folder: `tracks.parquet`, `transforms.npz`,
`stabilization.csv`, `results.csv`, `results.json`, `render.mp4`, `render.json`,
`validation.json`.

## How the margin of error is computed

Every vehicle gets a declared margin, shown in the table, the speed profile and
(optionally, in km/h) the render panel. Hover a value for its breakdown.
It is an expanded uncertainty at k = 2 (about 95 %): independent terms, each a
standard uncertainty, combined root-sum-square and doubled.

| Term | How it's computed | Typical size (k = 2) |
| --- | --- | --- |
| Calibration | Monte Carlo: the typed calibration distances (± *distance tolerance*, default 5 cm; tape ~2 cm, orthophoto ~10 cm) and every clicked calibration point (± 0.5 px) are randomly perturbed 200 times, and the mapping (scale, similarity, homography or polygon least squares) is refitted each time. The spread of each vehicle's path length over A→B (and over the 0.5 s around its max) is the relative speed uncertainty. | 0.2–1 %: 20 cm over 20 m is 1 % (1σ) |
| Check-distance fit | Redundant distances give a Birge ratio √(χ²/dof) of their residuals against the declared tolerances. Above 1 (typos, non-flat ground, wrong model), the calibration term is widened by that factor. With no redundant distance the calibration is unchecked, and the UI says so. | ×1–several |
| Noise, max | The momentary-speed noise σ (deviation from a much smoother curve). | 0.4–1 km/h |
| Noise, average | Standard error of the arc-length regression slope, from the residuals of a constant-acceleration fit (real acceleration isn't noise). It's inflated for samples correlated over 0.5 s. | 0.2–0.5 km/h |
| Height parallax | **Not corrected.** The tracked box centre sits h above the road (car ~0.8 m, bus ~1.5 m), so on the ground plane it moves H/(H−h) times faster than the vehicle. With the drone height H entered, this one-sided bias widens only the lower side: `61.3 +1.1 −1.9`. Without it, parallax isn't in the margin, and the UI says so. | +0.8 % car at 100 m, +3 % bus at 50 m |
| Timing, stabilization | Not budgeted: frame timing from `capture_fps` and the stabilization residuals are far below the terms above. | < 0.1 % |

In the app each vehicle's speed carries its margin as a share of the speed, e.g.
`54.6 (±2.4 %)`. The inputs (distance tolerance, click precision, drone height) are set per
project on the Results screen. **Apply & recompute** refreshes the margins
without re-detecting. `results.csv` and `results.json` carry `speed_max_u_kmh`,
`speed_avg_u_kmh`, the `*_u_lower_kmh` variants and the per-term breakdown.

The margin is a model of the known error sources. It can't see what it doesn't
model: a wrong `capture_fps`, a mis-measured distance beyond its tolerance, or
lens distortion. **Only a field validation proves it.** On the Results screen,
**Validation run** takes reference speeds, from a GPS logger's average over A→B
or a cruise-control speed, for specific vehicles (by track number or by time in
the clip). The app matches them to the measured vehicles and reports the bias,
the spread, and how many reference speeds fall inside the declared margin
(combined with the reference's own tolerance). About 95 % should. Fewer, with a
binomial p < 0.05, means the margin is too small. The references and the report
are stored per project in `validation.json`. Use 20+ runs at several speeds and
in both directions.

## Tests

```bash
pytest            # unit + synthetic end-to-end + API flow (~40 s)
pytest -m "not slow"
```

The synthetic test renders a rectangle at a known speed over a textured road, with
camera sway and per-frame position jitter. It stores the clip as 30 fps
"slow motion" with a 120 fps capture rate. The tool must recover the speed within
0.5%.

## Not covered yet

- A field test with GPS-logged runs, with a ±2% target. The Validation run screen
  is ready for it, but it needs real footage.
- Height-parallax correction (it is only bounded, when the drone height is known).
- Reading traffic lights from real footage: none of the test clips shows a lamp
  facing the camera, so colour reading is only tested on synthetic images and on
  a typed-phase setup over real tracks.
- Non-goals for v1: real-time processing, live feed, oblique angles, multi-drone fusion.
