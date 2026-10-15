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
is `yolo11x.pt`, the full COCO model, which works on both oblique and top-down views.
`yolo11x-obb.pt` (DOTA, oriented boxes) only suits true top-down footage: on oblique
footage it finds almost nothing.

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
3. **Process:** choose **Camera: Hovering** (stabilize every frame) or **Still**
   (skip stabilization, faster). A preview run with stabilization on measures camera
   drift against the reference frame, in px and in cm on the ground, and says whether
   skipping is safe. **Preview tracking** runs detection, stabilization and tracking on a
   few seconds and streams annotated frames: boxes coloured by track ID, trails and
   lines A/B. It also reports how many tracks crossed each line. Use it to check
   settings before the full run, which streams the same tracking preview and can be
   cancelled.
4. **Results:** the proxy video, with a live tracking overlay drawn in the browser
   (boxes, trails, provisional then final speeds), sits beside the vehicle table.
   No render is needed for review. If nothing was measured, the page says where
   tracks were lost (few detections, or crossing only one line). Clicking a row seeks to that vehicle. The page shows a histogram; mean,
   p85 and p95 with confidence intervals; and the ITE sample size needed against
   the valid n. You can export CSV, JSON and MP4. **Recompute speeds** re-runs only
   the speed step, so you can change thresholds, lines or calibration without
   re-detecting.

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
| Detection | Ultralytics YOLO, OBB or axis-aligned, with optional tiling and NMS across tiles. DOTA `small/large vehicle` map to car/truck. |
| Tracking | ByteTrack (supervision) on reference-frame boxes. |
| Speed | Savitzky–Golay smoothing of the world trajectory. **Primary:** linear regression of arc length vs. time between the sub-frame crossings of A and B. **Audit:** chord / Δt. Flags: `disagree` (>2%), `accel` (>8 m/s²), `residual` (jumps ⇒ ID switch), `gap`, `multi_cross`. |
| Stats | t-interval for the mean, bootstrap CIs for p85/p95, ITE `n = (S·K/E)²·(2+U²)/2`. |

Outputs in the project or `--out` folder: `tracks.parquet`, `transforms.npz`,
`stabilization.csv`, `results.csv`, `results.json`, `render.mp4`, `render.json`.

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

- A field test with GPS-logged runs, with a ±2% target. This needs real footage.
- Non-goals for v1: real-time processing, live feed, oblique angles, multi-drone fusion.
