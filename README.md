# speedcam

## Get it

**Windows** — in PowerShell:

```powershell
irm https://raw.githubusercontent.com/Bartor/speedcam/main/install.ps1 | iex
```

**macOS / Linux** — in a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/Bartor/speedcam/main/install.sh | sh
```

That's all. It installs Python and ffmpeg if they're missing (winget on Windows, Homebrew on
macOS, your package manager on Linux), downloads the latest release, installs the Python
packages (a few minutes and ~3 GB the first time, with CUDA PyTorch on an NVIDIA GPU) and opens
the app in your browser. Run the same command again to update; your projects are kept. The
app lives in `%LOCALAPPDATA%\speedcam` (Windows, with a desktop shortcut) or `~/speedcam`.

Measures vehicle speeds from drone footage (straight down or at an angle), and checks
red-light running and solid-line crossings. It exports a per-vehicle CSV/JSON and an
annotated MP4. Everything runs locally: a Python analysis package with a CLI, plus a
browser UI served from your own machine. No video is uploaded anywhere.

## Requirements

| | Needed | Notes |
|---|---|---|
| OS | Windows 10/11, macOS, Linux | Developed and tested on Windows 11 |
| Python | 3.11 or newer | The launcher creates its own virtual environment |
| ffmpeg + ffprobe | on `PATH` | Decoding, proxies and the rendered video |
| Node.js | 18 or newer | Optional: only to change the web UI. The launcher downloads the prebuilt UI for your version |
| GPU | NVIDIA with ~8 GB VRAM recommended | Works without one, but analysis is 10–50× slower |
| Disk | ~6 GB for the install | PyTorch with CUDA is ~3 GB; YOLO weights download on first use |

## Install by hand

If you'd rather not run the one-line installer:

1. Install **Python 3.11+** and **ffmpeg** (commands below).
2. Download **speedcam-x.y.z.zip** from the
   [latest release](https://github.com/Bartor/speedcam/releases/latest) and unzip it
   anywhere. No git or Node.js needed: the web UI comes prebuilt.
3. Double-click `speedcam.cmd` (Windows) or run `./speedcam.sh` (macOS/Linux).

To update, the launcher tells you when a new version is out: copy the new zip's
contents over the folder. Installed packages are reused, and projects live in
`~/speedcam-projects`. Developers can `git clone` the repository instead; the launcher
then builds or downloads the UI for whatever is checked out.

**Windows** (in PowerShell or a terminal):

```bash
winget install Python.Python.3.12 Gyan.FFmpeg
```

Open a new terminal so `PATH` picks them up.

**macOS**: `brew install python@3.12 ffmpeg`.

**Linux** (Debian/Ubuntu): `sudo apt install python3 python3-venv ffmpeg`.
For an NVIDIA GPU, install the driver first; the launcher picks
CUDA PyTorch when `nvidia-smi` works.

Not sure what's missing? `speedcam --check` (`speedcam.cmd --check` / `./speedcam.sh --check`)
lists each requirement, says what to install, and installs nothing.

## First run

```bash
speedcam            # Windows: speedcam.cmd (or double-click it); macOS/Linux: ./speedcam.sh
```

The first run takes a few minutes. It checks the requirements, creates `.venv`,
installs dependencies (CUDA PyTorch when an NVIDIA GPU is present), downloads the web UI
built for this version (or builds it with Node.js when you've changed it), then starts the app and opens the browser at http://127.0.0.1:8000. After that it starts
straight away, and redoes a setup step only when needed: `pyproject.toml` changed, or
the UI sources changed. No need to activate the venv. A short tutorial
opens the first time; the **?** buttons explain each part of the app.

```bash
speedcam --shortcut          # desktop shortcut (Windows)
speedcam serve --jobs 3      # any CLI command works through the launcher
speedcam analyze p.yaml --out out/
speedcam --setup-only        # install/update without starting
```

**Prebuilt web UI.** On every push to `main` that changes `web/`, a GitHub Actions
workflow (`.github/workflows/ui-release.yml`) builds the UI and attaches it to the rolling
[`ui-latest`](https://github.com/Bartor/speedcam/releases/tag/ui-latest) release as
`ui-<hash>.zip`, where the hash covers the UI sources (`speedcam --ui-hash`). The launcher
downloads the zip matching your checkout, so the UI always fits the code. With no match
(you changed the UI, or the build is still running) it builds with Node.js. Forks can point
it elsewhere with `SPEEDCAM_UI_REPO=owner/name`.

**Troubleshooting**

- *"ffmpeg not found"* right after installing it: open a new terminal (or sign out and
  in) so `PATH` updates.
- *Port 8000 in use:* `speedcam serve --port 8010`.
- *Analysis very slow:* `speedcam --check` says whether a GPU is used. With an NVIDIA GPU
  but CPU PyTorch, delete `.venv` and run `speedcam` again.
- *GPU out of memory:* lower *Frames per GPU batch* under Advanced settings on the
  Process screen.

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
scored car mAP50 0.908 (all 10 VisDrone classes: 0.619). The weights download on first use
from the `models` release into `~/speedcam-models/`. Compared with Auto on full analyses of
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
   **Jobs** (top bar) lists every job of every project, newest first: analyses,
   recomputes, renders, previews and proxies, with status, duration and what each came
   to. Each finished analysis, recompute and render keeps its own copy of what it
   produced (results JSON/CSV, violations, tracks, the MP4, and the settings it ran
   with) in `<project>/jobs/<job id>/`, so earlier runs stay downloadable after the
   project's results or video are replaced. A job's copies can be deleted to free space;
   the page shows how much they take.
1. **Import:** pick a video on disk. Each video has one project: picking a video that
   already has one opens it (the picker tags those videos, and the library marks older
   duplicates). Nothing is uploaded. The server builds a 1080p
   H.264 proxy for scrubbing. Confirm `capture_fps`. **DJI slow-motion clips** are
   stored at e.g. 30 fps, but were captured at 100/120 fps. Real time is always
   `frame / capture_fps`.
   **View rotation** (⟲ ⟳ here and on the Calibrate screen) turns the picture by 90°
   steps, e.g. for a drone shot with the road running across the frame. It only
   changes how the video is shown, previewed and rendered. Calibration, tracks and
   results stay in the original frame's pixels, so it works on analysed clips
   straight away, with nothing to re-run. A rendered MP4 is rotated as it was at
   render time (90°/270° gives a portrait video; with the side panel, widen the
   panel or put it beside the video, since its text scales to the panel's width).
   **Clips stored rotated** (a "display rotation" flag, e.g. DJI vertical shooting:
   3384×6016 stored, shown 6016×3384) are always analysed as stored, and the view
   rotation is set from the flag, so they show upright. The proxy and the render
   ignore the flag too, so picture and overlay always line up. Projects made before
   this rebuild their proxy once when opened, with nothing to re-analyse.
   **Timeline** (Calibrate, Process, Results and Export): a scrubbable view of the whole clip
   with a time ruler, frame thumbnails (hover for a bigger preview), the activity scan
   and the parts to analyse; on Results also each measured vehicle (click to jump to
   it). Drag on the bottom lane to cut a stretch out (starting over blue) or add one back
   (starting elsewhere), drag a part's edge to adjust it, or press **I** and **O** at the
   playhead for in and out points. *Use activity* takes the scan's parts, *Analyse all*
   the whole clip. The video has its own controls (frame steps, real-time readout,
   speed, full screen) and Premiere-style keys: **Space** plays and pauses, **K** / **L**
   pause and play (L again: faster), **←** / **→** step a frame (with Shift, a second),
   **Home** / **End** go to the start and end. On the timeline the wheel scrolls along it
   when zoomed in, **Alt** or **Ctrl**+wheel zooms at the cursor, **Shift**+wheel moves
   the playhead frame by frame; **+** / **−** zoom at the playhead and **\\** fits the
   whole clip. **?** (or the *Shortcuts* button) lists them all.
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
   change. **Solid line** draws a road marking (a smooth curve through the clicked points, like a path). By default (*Touches the line*) a vehicle is reported when its outline gets over the line by 0.3 m for 0.5 s, and on every lane change; *Crosses it* reports lane changes only. The outline is the oriented box, or for an axis-aligned box a rectangle of the class's usual width (car 1.8 m, truck 2.5 m, bus 2.55 m, motorcycle 0.8 m) along the direction of travel, as long as the box allows (a box around a car driving diagonally is much bigger than the car; solving the width from the box alone is unstable away from 0° and 90°). Detections cut off by the frame edge don't count. Vehicles that got over a line by less than the threshold are listed under the violations with how far, so the threshold can be judged against the footage. The video marks the vehicle and flashes the line for as long as it's on it. Known limit: in an oblique view the box includes the car's roof and side, so a car with its wheels on the line can read a few tens of cm clear (seen on 0492); lower the margin there. A lane change only counts once the vehicle is 0.5 m clear on both sides, so tracking wobble along a line isn't reported. These checks use the stored tracks: draw or change
   them after the analysis and press **Recompute**. Lights added or moved since then
   are read from the video (decode only, no GPU). A project can have only
   violation checks, without lines A/B or paths.
   **Nothing set up is fine too:** without a calibration, or with nothing to measure, the
   analysis only detects and tracks; calibrate and draw lines later and press Recompute.
3. **Process:**
   **Activity scan (long videos, CCTV):** *Scan for vehicles* reads about one frame a
   second at the detection size (1280 px), only keyframes where the file has them every
   2 s or less, with the analysis's own detector. A timeline shows moving vehicles per
   second (bright) and all of them (dim). A box that stays put between neighbouring
   samples is a parked or waiting vehicle, so a street of parked cars isn't "activity".
   *Analyse only the parts with vehicles* keeps the parts with moving vehicles, padded
   by 2 s and with gaps under 5 s joined. The analysis then seeks to each part, and
   tracking starts fresh in each one. Frame numbers and times stay those of the whole
   video. Stabilization registers every frame to the reference frame, so jumps are fine:
   it re-finds the reference after each. The render covers only those parts too, back to
   back (optional). Measured: a 57 s 4K HEVC clip scans in ~21 s, mostly decoding;
   1080p H.264 is much faster.
 choose **Camera: Hovering** (stabilize every frame) or **Still**
   (skip stabilization, faster). A preview run with stabilization on measures camera
   drift against the reference frame, in px and in cm on the ground, and says whether
   skipping is safe. **Preview tracking** runs detection, stabilization and tracking on a
   few seconds and streams annotated frames: boxes coloured by track ID, trails and
   lines A/B. It also reports how many tracks crossed each line. Use it to check
   settings before the full run, which streams the same tracking preview and can be
   cancelled.
4. **Results:** a dashboard. The proxy video, with a live tracking overlay drawn in
   the browser (boxes, trails, momentary speed between the lines, then max and average),
   has the list of measured vehicles beside it: a picture of each, its name, max and
   average speed, and marks for violations (⛔), merged tracks (⧉), an inconclusive
   average (~) or an invalid measurement (✕). No render is needed for review. Each
   vehicle has a **max** (highest momentary speed between A and B) and an **average**
   (A→B). Pick one in the list or on the timeline's vehicle lane to seek to it and plot
   its **speed along the section** below the video: the momentary speed curve with
   max/min marked, a ±2σ noise band, the average and speed-limit lines, and its
   violations marked where they happened (click a mark to watch it, or anywhere on the
   curve to jump there).
   **Vehicle card**, beside the graph: give the vehicle a name to find it again (it goes
   into the list, `results.csv`/`.json` and, optionally, the rendered video). The app
   suggests one from the vehicle's colour, read from the middle of its box where it's
   seen best, so from above it's the roof's colour ("white car", "blue truck"); makes and
   models aren't recognised. **Merge** fixes a vehicle the tracker split in two: when it
   loses a vehicle (behind a tree, a sign or a bus) it picks it up again under a new
   number, so one car becomes two tracks, neither may cover the whole section, and one
   that does can be counted twice. The card lists tracks that continue this one: starting
   up to 3 s after it ends, heading the same way, where it would have got to at its speed
   (within a box length plus drift), at a similar speed. *Possibly split vehicles* lists
   every likely pair in the clip that reaches the measured section. Merged tracks are
   measured as one under the first one's number; nothing is measured across the gap, so
   the average becomes inconclusive when the gap is longer than 0.3 s, while the max from
   the cleanly tracked frames on either side stands. *Split* undoes it; the stored tracks
   are never changed. A merge re-measures the clip (like Recompute), and a new analysis
   drops names and merges, since its track numbers are new.
   Statistics and the histogram switch between max and average. They include
   mean, p85 and p95 with confidence intervals, plus the ITE sample size needed
   against the valid n. A **calibration check** warns when every vehicle's speed
   rises or falls by the same share along A→B. Drivers vary but geometry doesn't, so
   a shared trend means a calibration error that biases max speeds. If nothing was
   measured, the page says where tracks were lost. *All measurements* folds out the
   full table with every column.
   **Recompute** re-runs the speed and violation steps, without re-detecting.
   With violation checks, a **Violations** panel shows each light's phase timeline
   (red/amber/green, with the runs marked) and lists every red-light run and
   solid-line crossing with its time, vehicle and speed. Click one to watch it. The
   overlay colours stop lines and lights by the current phase and tags violating
   vehicles, and `violations.csv` lists them.
5. **Export:** the annotated MP4 for sharing, and the data downloads (CSV, JSON,
   violations). A **preview** shows any frame exactly as the render will draw it with
   the current settings (pick the frame on the timeline); settings are kept with the
   project as you change them. **What to render:** the whole clip, the analysed parts,
   only the measured vehicles (each from 1 s before it until its label goes), only the
   violations (3 s either side), or ranges picked on the timeline's bottom lane (drag,
   or I / O). The parts are rendered back to back. **Quality:** resolution (720p up to
   the original), frame rate, playback speed (down to 0.1× slow motion), encoder
   quality, encoder (NVIDIA, Apple or x264; automatic picks a GPU one when it works),
   stabilized or as-filmed frames, and a colour LUT (`.cube`, e.g. D-Log M → Rec.709).
   **On the video:** speed labels, vehicle names, boxes, trails, paths / lines A–B,
   lights and solid lines, unmeasured vehicles and a time readout can each be switched
   off; violation labels always show. Vehicles get speed pills (live momentary speed
   between the lines, then max, coloured against the limit), above, below, left or right
   of each vehicle (as seen in the output): pick the side away from neighbouring lanes
   and lines. How long the max label stays and the trail length are set here too. An
   optional **side panel** (left or right, over the frame or beside it, in three widths)
   shows a title and subtitle, a speed-limit sign, running totals (vehicles, over the
   limit, top speed, p85), the latest vehicle's speed curve, and the list of vehicles
   (by name, if they have one) with their max and average speeds, latest or fastest
   first, and an optional footer. Stop lines and lights are drawn in the phase's colour,
   and violating vehicles get a red box and a RED LIGHT / SOLID LINE pill. Text uses
   Segoe UI and Bahnschrift (Windows) or DejaVu Sans, or any `.ttf` set in `render.font`.

The API lives under `/api` (OpenAPI at `/docs`). It covers projects, frames,
`proxy.mp4` with range requests, calibration, analyze, render (and one-frame render
previews), results, vehicle names, merges and pictures, job events (SSE) and cancel.

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
| Speed | Savitzky–Golay smoothing of the world trajectory. **Momentary:** |velocity| from a local quadratic fit over 0.5 s (`speed.momentary_window_s`; measured noise about ±0.18 km/h on 4K footage with flow positions, ±0.28 at 0.25 s; box centres alone ±0.29 / ±0.57), giving max/min, where the max happened, and a profile. **Average:** distance travelled between the sub-frame crossings of A and B (or along the path) divided by the time between them. (A regression slope read stop-and-go traffic up to 24% low.) **Audit:** chord / Δt. **Max first:** the max only uses frames tracked cleanly, i.e. no detection gap over 0.3 s and no position jump (deviation from a local constant-acceleration fit over 2 s above 0.5 m) within the 0.5 s speed window, and not the track's first or last quarter second. A gap or jump makes only the **average inconclusive** (`avg_valid` false, shown as –); the vehicle and its max still count. A vehicle is invalid only for `accel` (>8 m/s² mean), too few points, or under 0.5 s of clean tracking. **Truncated boxes:** at the start and end of a track, detections whose box is under 85% of its length 0.25–0.5 s later/earlier (a vehicle entering the frame or emerging from behind something) are dropped. On 0488 entering boxes grew 3 → 6.7 m in 0.3 s and put the position ~1 m ahead of the car. Flags: `disagree` (>2%), `accel`, `residual` (jump), `gap`, `multi_cross`, `no_clean`. On 13 real clips this took 0488 from 13 to 48 measured vehicles and 0474 from 7 to 40, with every previously valid vehicle's max unchanged except one (3.9%). |
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
| Noise, average | Two end positions, each with the position noise left after a local constant-acceleration fit over 2 s (real speed changes aren't noise), divided by the time between them. | 0.1–0.5 km/h |
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

## Publishing (maintainers)

- **A version:** set `version` in `pyproject.toml`, commit, then tag and push it:
  `git tag v0.2.0 && git push origin v0.2.0`. The *Release* workflow checks the tag matches
  `pyproject.toml`, builds the web UI, and publishes `speedcam-0.2.0.zip` (source plus
  built UI) on a release with install notes. Launchers of older versions then say a new
  one is out.
- **The web UI between versions:** every push to `main` that changes `web/` publishes
  `ui-<hash>.zip` to the `ui-latest` prerelease. Git checkouts download it instead of
  needing Node.js.
- **Model weights:** `speedcam` downloads its own models on first use from the release
  tagged `models`, and checks each file's SHA-256 (`NAMED_MODELS` in
  `speedcam/detection.py`). To publish one, create that release once (as a prerelease, so it
  never shows as the latest version) and attach the file, e.g. `yolo11x-visdrone.pt`.
  Ultralytics fetches its own weights (`yolo11x.pt`, `yolo11x-obb.pt`) by itself.

## License

Copyright (C) 2026 Bartosz Barnaba Rajczyk

speedcam is free software: you can redistribute it and/or modify it under the terms of
the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. It is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
[LICENSE](LICENSE) for the full text.

In short: use it, change it and share it freely. If you distribute it, or modify it and
run it as a service other people use over a network, make your source code available
under the same license.

Third-party parts keep their own licenses:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (detection) and the
  YOLO weights it downloads: AGPL-3.0. This is why speedcam is AGPL too.
- PyTorch (BSD-3-Clause), OpenCV (Apache-2.0), PyAV and the other Python dependencies:
  see each package.
- ffmpeg is called as a separate program (LGPL/GPL, depending on your build).
- The VisDrone-trained weights are made from the
  [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) dataset, which is
  for academic use. Check its terms before sharing weights trained on it.
  `tools/train_visdrone.py` lets anyone train their own.

