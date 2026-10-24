import { useEffect, useState } from "react";

const KEY = "speedcam.tutorial.v1";

const STEPS: { title: string; body: string[]; tip?: string }[] = [
  {
    title: "Welcome to speedcam",
    body: [
      "speedcam measures vehicle speeds in drone footage, and can check red-light running and solid-line crossings. Everything runs on this computer; no video leaves it.",
      "A project goes through five steps, shown at the top: Import → Setup → Analyze → Review → Export.",
    ],
    tip: "This takes a minute. You can skip it and reopen it any time from Tutorial in the top bar.",
  },
  {
    title: "1 · Import",
    body: [
      "Pick a video from the file browser on the right. Each video gets one project; opening it again takes you back to it. A small proxy is built for smooth scrubbing.",
      "Check the capture fps: speeds depend on it. DJI slow-motion clips are recorded at 100/120 fps but stored at 30.",
    ],
  },
  {
    title: "2 · Setup",
    body: [
      "Everything you set before analysing, in four stages. Scale: click two ground points and type their measured distance (straight-down footage), or use Polygon / Rectangle with measured sides for views at an angle, and say how well they're known.",
      "Measure: a Path along each road (in the direction of travel), or lines A and B. Limits & tolerances: the speed limit and when a measurement counts. Rules (optional): red lights and solid lines.",
    ],
    tip: "Measured distances matter most: 20 cm off over 20 m is already 1 %.",
  },
  {
    title: "3 · Analyze",
    body: [
      "1. Find the activity: a quick scan shows on the timeline where vehicles move. 2. Choose what to analyse: only those parts, the whole clip, or parts you pick on the timeline.",
      "3. Check and run: a preview on a few seconds checks the detector and the camera setting, then the analysis detects, tracks and measures the vehicles.",
    ],
  },
  {
    title: "4 · Review",
    body: [
      "The video with a live overlay and the list of measured vehicles beside it: pick one to see its speed along the section, with any violations marked. Name vehicles, and merge one the tracker split in two.",
      "Statistics, violations, possible splits, the full table and the validation run in the tabs below. Changed something in Setup? Press Recompute: no new analysis.",
    ],
  },
  {
    title: "5 · Export",
    body: [
      "Render an annotated MP4 to share: choose what to render on the timeline (the whole clip, only the measured vehicles, the violations), what's drawn, the side panel and the quality, with a preview of any frame as it will look.",
      "Download the results as CSV or JSON there too.",
    ],
    tip: "Every section has a ? button with a short explanation.",
  },
];

/** First-run walkthrough. Shown once unless reopened; `open`/`onClose` let the top bar reopen it. */
export default function Tutorial({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [k, setK] = useState(0);
  useEffect(() => { if (open) setK(0); }, [open]);
  useEffect(() => {
    if (!open) return;
    const on = (e: KeyboardEvent) => {
      if (e.key === "Escape") finish();
      if (e.key === "ArrowRight") setK((x) => Math.min(STEPS.length - 1, x + 1));
      if (e.key === "ArrowLeft") setK((x) => Math.max(0, x - 1));
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  });
  if (!open) return null;
  const finish = () => { try { localStorage.setItem(KEY, "done"); } catch { /* private mode */ } onClose(); };
  const s = STEPS[k];
  const last = k === STEPS.length - 1;
  return (
    <div className="tutorial-backdrop" onMouseDown={(e) => e.target === e.currentTarget && finish()}>
      <div className="tutorial" role="dialog" aria-modal="true" aria-label="Tutorial">
        <div className="tutorial-steps">{STEPS.map((_, i) => <span key={i} className={i === k ? "on" : i < k ? "done" : ""} />)}</div>
        <h2>{s.title}</h2>
        {s.body.map((p, i) => <p key={i}>{p}</p>)}
        {s.tip && <p className="tutorial-tip">{s.tip}</p>}
        <div className="row">
          <button className="ghost" onClick={finish}>{last ? "Close" : "Skip tutorial"}</button>
          <span className="grow" />
          {k > 0 && <button onClick={() => setK(k - 1)}>Back</button>}
          <button className="primary" onClick={() => (last ? finish() : setK(k + 1))}>{last ? "Get started" : "Next"}</button>
        </div>
      </div>
    </div>
  );
}

/** Whether the tutorial should open by itself (never finished or skipped before). */
export function tutorialPending(): boolean {
  try { return localStorage.getItem(KEY) !== "done"; } catch { return false; }
}
