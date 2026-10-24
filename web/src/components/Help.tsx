import { useEffect, useRef, useState } from "react";

/** Short explanations behind the ? buttons. Plain words; the README has the details. */
export const HELP: Record<string, { title: string; body: string[] }> = {
  "import.timing": {
    title: "Timing (capture fps)",
    body: [
      "Speeds come from distance over time, and time is frame number ÷ capture fps, so this number must be the rate the drone actually recorded at.",
      "DJI slow-motion clips are stored at 30 fps but recorded at 100 or 120 fps: enter the recorded rate (the app suggests it when a sidecar .SRT file is next to the video).",
      "A known real duration (e.g. from the flight log) checks the rate.",
    ],
  },
  "import.rotation": {
    title: "View rotation",
    body: ["Turns the picture in 90° steps, e.g. when the road runs across a portrait shot. Only how the video is shown and rendered changes; calibration and results stay as they are, nothing needs re-running."],
  },
  "calibrate.calibration": {
    title: "Calibration",
    body: [
      "Tells the app how many metres a pixel is, so pixel motion becomes speed. Click on the reference frame; zoom in with the wheel for precise clicks.",
      "Straight-down footage: a few Distances (two points + their measured length) are enough. Any tilt: use Polygon or Rectangle: 4+ ground points with measured sides and diagonals correct the perspective.",
      "Measure distances with tape or take them from an orthophoto. Extra distances act as checks: their residuals show how well the calibration fits.",
      "Without a calibration the analysis still runs, but only detects and tracks.",
    ],
  },
  "calibrate.measure": {
    title: "Where speeds are measured",
    body: [
      "Path: click points along a road in the direction of travel (A → B). Each vehicle is measured along the curve, over all of it inside the corridor. Draw one path per road, e.g. at an intersection.",
      "Lines A and B: the older way, two lines across the road; speed is measured between the crossings. Paths replace them when both exist.",
    ],
  },
  "calibrate.redlights": {
    title: "Red lights",
    body: [
      "One setup per signal group: mark the traffic light(s) with Light and the stop line(s) with Stop line.",
      "The light's colour is read from the video on every frame. If the lamps aren't visible (filmed from behind or straight down), type when it was red instead.",
      "A vehicle runs the red when the front of its box reaches a stop line while the light is red. The arrow on a stop line shows which direction counts.",
    ],
  },
  "calibrate.solid": {
    title: "Solid lines",
    body: [
      "Draw a solid road marking (a smooth curve through your clicks). Touches the line: a vehicle whose outline gets over the line by the margin, for the set time, is reported, as is every lane change. Crosses it: lane changes only.",
      "In views at an angle a car's box includes its roof and side, so a car with its wheels on the line can look a little clear of it: lower the margin there.",
    ],
  },
  "calibrate.area": {
    title: "Region of interest (ROI)",
    body: ["Optional: a polygon outside which detections are ignored, e.g. to skip a car park or a parallel road."],
  },
  "process.analysis": {
    title: "Analysis",
    body: [
      "Detects vehicles on every frame (GPU), aligns frames to the reference (stabilization), tracks each vehicle and measures it. It takes roughly the clip's length to a few times it.",
      "After it, changing anything in Setup (scale, paths, lights, limits, tolerances) only needs Recompute in Review: no new analysis.",
    ],
  },
  "process.activity": {
    title: "Activity scan",
    body: [
      "For long videos (CCTV, long hovers): a quick pass at about one frame a second finds where vehicles move. Analysing (and rendering) only those parts saves most of the time.",
      "Parked cars don't count as activity. Padding keeps vehicles whole as they enter and leave.",
    ],
  },
  "process.camera": {
    title: "Camera: hovering or still",
    body: ["Hovering: every frame is aligned to the reference frame, which removes drone drift. Still: frames are assumed to line up (a tripod or a very steady hover); faster, but any drift becomes speed error. A preview run measures the drift and tells you whether skipping is safe."],
  },
  "process.preview": {
    title: "Preview tracking",
    body: ["Runs detection and tracking on a few seconds to check the settings: how many vehicles are found and whether they cross your lines or path. Nothing is saved."],
  },
  "process.settings": {
    title: "Detector",
    body: [
      "Auto picks the best model per clip: COCO for views at an angle, a top-down model for straight down, and the VisDrone-trained model when installed.",
      "The advanced settings rarely need changing; lower Frames per GPU batch if the GPU runs out of memory.",
    ],
  },
  "results.stats": {
    title: "Speed statistics",
    body: [
      "Max: each vehicle's highest momentary speed between the lines (or along the path), measured over 0.5 s. Average: distance over time for the whole stretch.",
      "p85 is the speed 85 % of vehicles stay under, the usual figure for speed limits. Required n says how many vehicles that figure needs to be reliable.",
    ],
  },
  "results.vehicles": {
    title: "Vehicles",
    body: [
      "One row per measured vehicle; click one to jump the video to it and plot its speed. (± %) is the vehicle's 95 % margin of error; hover a speed for its breakdown.",
      "A vehicle is valid when its max speed comes from cleanly tracked frames. A detection gap or a jump only makes the average inconclusive (–); the max still counts.",
    ],
  },
  "results.profile": {
    title: "Speed along the section",
    body: ["The vehicle's momentary speed along the stretch, with its max, min, average and the speed limit. The shaded band is ± 2 × the measured noise."],
  },
  "results.margin": {
    title: "Margin of error",
    body: [
      "How far off each speed can be (95 %), from the calibration's tolerance (Monte Carlo over your measured distances and clicks) and the tracking noise.",
      "Entering the drone's height also bounds the height parallax: vehicle tops are closer to the camera than the road, which makes speeds read slightly high.",
    ],
  },
  "results.violations": {
    title: "Violations",
    body: ["Red-light runs and solid-line crossings, with the light's phase timeline. Click a row to watch it. Change the lines or their settings in Setup → Rules, then Recompute."],
  },
  "setup.tolerances": {
    title: "Tolerances",
    body: [
      "How well the calibration inputs are known: each typed distance (tape ~2 cm, orthophoto ~10 cm) and each click. Every speed's margin of error comes from them (Monte Carlo over these tolerances) and the tracking noise.",
      "The drone height (from the flight log) bounds the height-parallax bias: vehicle tops are closer to the camera than the road, so speeds read slightly high. Without it that bias isn't in the margin.",
    ],
  },
  "setup.limit": {
    title: "Speed limit",
    body: ["Used everywhere a speed is judged: vehicles over the limit in Review, the red line on the speed graphs, and the colours in the rendered video (green up to the limit, amber up to 10 km/h over, red beyond)."],
  },
  "setup.checks": {
    title: "When a measurement counts",
    body: [
      "A vehicle's max speed comes only from cleanly tracked frames: it needs at least the clean tracking time set here. A detection gap or a jump in position makes the average over the stretch inconclusive; the max still stands.",
      "The momentary speed (the max and the graphs) is measured over a short window: shorter follows changes faster but is noisier.",
    ],
  },
  "results.vehicle": {
    title: "Vehicle",
    body: [
      "The vehicle picked in the list or on the timeline. Give it a name to find it again; it shows in the list, the downloads and (if you like) the rendered video. The suggestion comes from its colour seen from above, so it's the roof's colour.",
      "If the tracker lost it for a moment (behind a tree, a sign or a bus) and picked it up under a new number, merge the two: they're measured as one vehicle. Nothing is measured across the gap, so the average becomes inconclusive, but the max from the cleanly tracked frames on either side stands. Split undoes it.",
    ],
  },
  "results.splits": {
    title: "Possibly split vehicles",
    body: ["Tracks that start shortly after another one ended, heading the same way, where that one would have got to at its speed: likely one vehicle under two numbers. Watch the hand-over, then merge if it's the same vehicle."],
  },
  "export.preview": {
    title: "Preview",
    body: ["Any frame of the clip drawn exactly as the render will draw it, with the settings on the right: pick a frame on the timeline. The vehicle lane shows where the measured vehicles are."],
  },
  "export.range": {
    title: "What to render",
    body: [
      "The whole clip, the parts that were analysed, just the measured vehicles or the violations (each with a little lead-in), or any ranges you pick on the timeline's bottom lane.",
      "The parts are rendered back to back into one video.",
    ],
  },
  "export.quality": {
    title: "Quality",
    body: [
      "Resolution and frame rate of the video, how fast it plays (0.5× and slower are slow motion, smooth with high-frame-rate footage), and the encoder's quality: lower numbers are better and bigger.",
      "A GPU encoder (NVIDIA or Apple) is much faster than x264. The colour LUT turns flat log footage (e.g. D-Log M) into normal colours.",
    ],
  },
  "export.overlay": {
    title: "On the video",
    body: ["What's drawn over the picture. Violation labels always show. Put labels on the side of the vehicles away from neighbouring lanes, so they don't cover other cars."],
  },
  "export.panel": {
    title: "Side panel",
    body: ["A leaderboard for sharing: the speed limit, totals, the last vehicle's speed along the section and a list of every vehicle's max speed as it crosses the second line. Over the video keeps the picture's size; beside it makes the video wider."],
  },
  "results.validation": {
    title: "Validation run",
    body: ["Drive past at known speeds (a GPS logger or a calibrated speedometer), enter them here, and see whether the measured speeds fall within their margins. It's the only real proof of accuracy."],
  },
  "jobs": {
    title: "Jobs",
    body: ["Every analysis, recompute, render and scan of every project. Finished jobs keep their own copies of what they produced, so earlier runs stay downloadable. Delete a job's files to free space; the record stays."],
  },
};

/** A small "?" button that shows the explanation for `topic`. */
export default function Help({ topic }: { topic: keyof typeof HELP | string }) {
  const h = HELP[topic];
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!pos) return;
    const close = (e: Event) => { if (!(e.target as HTMLElement)?.closest?.(".help-pop, .help-btn")) setPos(null); };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setPos(null);
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    window.addEventListener("scroll", close, true);
    return () => { window.removeEventListener("mousedown", close); window.removeEventListener("keydown", esc); window.removeEventListener("scroll", close, true); };
  }, [pos]);
  if (!h) return null;
  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation(); e.preventDefault();
    if (pos) return setPos(null);
    const r = btn.current!.getBoundingClientRect();
    const w = 340;
    setPos({ x: Math.max(8, Math.min(window.innerWidth - w - 8, r.left - 12)), y: r.bottom + 6 });
  };
  return (
    <>
      <button ref={btn} type="button" className="help-btn" aria-label={`Help: ${h.title}`} title={h.title} onClick={toggle}>?</button>
      {pos && (
        <div className="help-pop" role="dialog" aria-label={h.title} style={{ left: pos.x, top: pos.y }}>
          <div className="help-title">{h.title}</div>
          {h.body.map((p, i) => <p key={i}>{p}</p>)}
        </div>
      )}
    </>
  );
}

/** Rarely touched settings, folded away; remembers whether it was open. */
export function Advanced({ id, title = "Advanced settings", children }: { id: string; title?: string; children: React.ReactNode }) {
  const key = `speedcam.advanced.${id}`;
  const [open, setOpen] = useState(() => { try { return localStorage.getItem(key) === "1"; } catch { return false; } });
  return (
    <details className="advanced" open={open}
      onToggle={(e) => { const o = (e.currentTarget as HTMLDetailsElement).open; setOpen(o); try { localStorage.setItem(key, o ? "1" : "0"); } catch { /* private mode */ } }}>
      <summary>{title}</summary>
      <div className="advanced-body">{children}</div>
    </details>
  );
}
