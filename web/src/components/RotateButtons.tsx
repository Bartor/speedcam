import { useState } from "react";
import { api, Project } from "../api";

/** Rotate the view by 90° either way. Only how the video is shown and rendered: analysis results
 * stay valid, nothing needs re-running. */
export default function RotateButtons({ project, onChanged }: { project: Project; onChanged: (p: Project) => void }) {
  const [busy, setBusy] = useState(false);
  const rot = project.config.rotation ?? 0;
  const set = async (by: number) => {
    setBusy(true);
    try { onChanged(await api.patch(project.id, { rotation: (rot + by + 360) % 360 })); } finally { setBusy(false); }
  };
  const tip = "Rotate the view. Only changes how the video is shown, previewed and rendered: calibration and results stay as they are.";
  return (
    <span className="rotate-btns" title={tip}>
      <button className="ghost" disabled={busy} onClick={() => set(-90)} aria-label="Rotate 90° counter-clockwise">⟲</button>
      <span className="muted small">{rot}°</span>
      <button className="ghost" disabled={busy} onClick={() => set(90)} aria-label="Rotate 90° clockwise">⟳</button>
    </span>
  );
}
