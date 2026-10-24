import { useEffect, useMemo, useState } from "react";
import Help from "./Help";
import ProgressBar from "./ProgressBar";
import { api, Project, Results, SolidCase, useJob } from "../api";

interface Props {
  project: Project; res: Results; onResults: (r: Results) => void; onChanged: (p: Project) => void;
  onSeek?: (t: number) => void;
}

/** Spans over the threshold (samples at most 0.5 s apart), as the server finds them. */
function spans(c: SolidCase, thr: number): [number, number][] {
  const out: [number, number][] = [];
  let k = 0;
  while (k < c.t.length) {
    if (c.over_m[k] < thr) { k++; continue; }
    let j = k;
    while (j + 1 < c.t.length && c.over_m[j + 1] >= thr && c.t[j + 1] - c.t[j] <= 0.5) j++;
    out.push([c.t[k], c.t[j] + c.step_s]);
    k = j + 1;
  }
  return out;
}
const verdict = (c: SolidCase, thr: number, dur: number) => {
  const sp = spans(c, thr);
  const longest = sp.reduce((a, [x, y]) => Math.max(a, y - x), 0);
  return { reported: c.crossed_t.length > 0 || sp.some(([x, y]) => y - x >= dur), longest, crossed: c.crossed_t.length > 0, sp };
};

/** Every vehicle that came near a solid line: how far over it was, frame by frame (by its
 * segmentation masks where it has them), and whether the current threshold reports it. The
 * threshold and duration re-judge every case here before they're saved. */
export default function SolidCases({ project, res, onResults, onChanged, onSeek }: Props) {
  const v = res.violations;
  const fps = res.capture_fps;
  const [thr, setThr] = useState(v?.overlap_m ?? 0.3);
  const [dur, setDur] = useState(v?.min_duration_s ?? 0.5);
  useEffect(() => { setThr(v?.overlap_m ?? 0.3); setDur(v?.min_duration_s ?? 0.5); }, [v?.overlap_m, v?.min_duration_s]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(project.jobs.find((j) => j.kind === "masks")?.id ?? null);
  const job = useJob(jobId, () => {
    api.results(project.id).then(onResults).catch(() => {});
    api.project(project.id).then(onChanged).catch(() => {});
  });
  const running = ["running", "queued", "connecting"].includes(job.status);
  const cases = v?.solid_cases ?? [];
  const judged = useMemo(() => cases.map((c) => ({ c, ...verdict(c, thr, dur) })), [cases, thr, dur]);
  if (!v || v.overlap_m == null) return null;  // only for "touches the line"
  const now = judged.filter((x) => x.reported).length;
  const saved = cases.filter((c) => c.reported).length;
  const changed = Math.abs(thr - (v.overlap_m ?? 0)) > 1e-9 || Math.abs(dur - (v.min_duration_s ?? 0)) > 1e-9;

  const apply = async () => {
    setBusy(true);
    setErr(null);
    try {
      onChanged(await api.patch(project.id, { violations: { overlap_m: thr, min_duration_s: dur } }));
      onResults(await api.recompute(project.id));
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const outline = async () => {
    setErr(null);
    try { setJobId((await api.outline(project.id)).id); } catch (e) { setErr(String(e)); }
  };

  return (
    <div className="solid-cases">
      <div className="panel-head">
        <h3>Solid-line cases<Help topic="setup.cases" /></h3>
        <span className="tag">{cases.length}</span>
      </div>
      {(v.masks_missing ?? 0) > 0 && !running && (
        <div className="banner info small row">
          <span className="grow">{v.masks_missing} detections near a solid line have no outline yet (lines drawn or moved after the analysis):
            they're judged by the box estimate.</span>
          <button className="primary" onClick={outline} title="Segment just those (GPU, ~30 ms each), then recompute">Outline them</button>
        </div>
      )}
      {running && <ProgressBar label={job.progress.stage ?? job.status} done={job.progress.done} total={job.progress.total} />}
      {job.end?.status === "error" && <div className="banner error small">{job.end.error}</div>}
      <div className="cases-ctl">
        <label>Over the line by ≥ <b>{thr.toFixed(2)} m</b>
          <input type="range" min={0.02} max={0.8} step={0.01} value={thr} onChange={(e) => setThr(+e.target.value)} /></label>
        <label>for ≥ <b>{dur.toFixed(1)} s</b>
          <input type="range" min={0.1} max={3} step={0.1} value={dur} onChange={(e) => setDur(+e.target.value)} /></label>
        <span className="cases-sum">{now} of {cases.length} reported{changed ? ` (saved: ${saved})` : ""}</span>
        {changed && <>
          <button className="primary" disabled={busy} onClick={apply}>{busy ? "Recomputing…" : "Save & recompute"}</button>
          <button className="ghost" disabled={busy} onClick={() => { setThr(v.overlap_m ?? 0.3); setDur(v.min_duration_s ?? 0.5); }}>Reset</button>
        </>}
      </div>
      {err && <div className="banner error small">{err}</div>}
      {cases.length === 0 && <p className="muted small">No vehicle came within {(project.config.violations as any)?.mask_gate_m ?? 0.5} m of a solid line.</p>}
      <div className="cases">
        {judged.map(({ c, reported, longest, crossed, sp }) => (
          <CaseCard key={`${c.line}-${c.track_id}`} project={project} c={c} fps={fps} thr={thr} dur={dur}
            reported={reported} longest={longest} crossed={crossed} spans={sp} onSeek={onSeek}
            name={res.vehicles.find((x) => x.track_id === c.track_id)?.name} />
        ))}
      </div>
      <p className="footnote">Yellow: the vehicle's segmentation mask, green: the box estimate (its class's usual width along its direction of travel),
        red: the line. A mask includes the mirrors (~0.15 m each side); seen at an angle it also holds the side facing the camera. The curve is the median of
        {" "}{(project.config.violations as any)?.mask_smooth ?? 7} masks, so single-frame glitches don't count. A lane change (the centre crossing) is always reported.</p>
    </div>
  );
}

function CaseCard({ project, c, fps, thr, dur, reported, longest, crossed, spans: sp, onSeek, name }: {
  project: Project; c: SolidCase; fps: number; thr: number; dur: number; reported: boolean; longest: number; crossed: boolean;
  spans: [number, number][]; onSeek?: (t: number) => void; name?: string;
}) {
  const [frame, setFrame] = useState<number | null>(c.best_frame);
  const [strip, setStrip] = useState(false);
  const W = 260, H = 64, pad = 4;
  const t0 = c.t[0] ?? 0, t1 = c.t[c.t.length - 1] ?? 1;
  const vmax = Math.max(0.5, thr * 1.4, ...c.over_m);
  const x = (t: number) => pad + ((t - t0) / Math.max(1e-6, t1 - t0)) * (W - 2 * pad);
  const y = (v: number) => H - pad - (v / vmax) * (H - 2 * pad);
  const path = c.t.map((t, i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(c.over_m[i]).toFixed(1)}`).join("");
  const img = (f: number) => `/api/projects/${project.id}/solid-case.jpg?tid=${c.track_id}&frame=${f}&line=${c.line}`;
  const stripFrames = useMemo(() => {
    const idx = c.t.map((_, i) => i).filter((i) => c.over_m[i] > 0);
    const pool = idx.length >= 4 ? idx : c.t.map((_, i) => i);
    const n = Math.min(6, pool.length);
    return Array.from({ length: n }, (_, k) => Math.round(c.t[pool[Math.round((k * (pool.length - 1)) / Math.max(1, n - 1))]] * fps));
  }, [c, fps]);
  const pick = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = t0 + ((e.clientX - r.left) / r.width) * (t1 - t0);
    let best = 0;
    c.t.forEach((tt, i) => { if (Math.abs(tt - t) < Math.abs(c.t[best] - t)) best = i; });
    setFrame(Math.round(c.t[best] * fps));
    onSeek?.(c.t[best] - 0.5);
  };
  return (
    <div className={`case${reported ? " reported" : ""}`}>
      <div className="case-head">
        <b>{name || `#${c.track_id} ${c.class}`}</b>
        <span className="muted small">{name ? `#${c.track_id} ${c.class} · ` : ""}{c.name}</span>
        <span className="grow" />
        <span className={`tag ${c.source === "mask" ? "" : "warn"}`} title={c.source === "mask" ? "Measured by its segmentation masks" : "Measured by the box estimate only (no masks)"}>
          {c.source === "mask" ? "masks" : "box estimate"}</span>
        <span className={`tag ${reported ? "bad" : "ok"}`}>{reported ? (crossed ? "reported: lane change" : "reported") : "not reported"}</span>
      </div>
      {frame !== null && <img className="case-img" src={img(frame)} alt={`Vehicle ${c.track_id} at frame ${frame}`} loading="lazy" />}
      <svg viewBox={`0 0 ${W} ${H}`} className="case-graph" onClick={pick} role="img"
        aria-label={`Over the line: up to ${c.max_over_m.toFixed(2)} m`}>
        {sp.map(([a, b], i) => <rect key={i} x={x(a)} y={pad} width={Math.max(1, x(b) - x(a))} height={H - 2 * pad}
          className={b - a >= dur ? "span-on" : "span-short"} />)}
        <line x1={pad} x2={W - pad} y1={y(thr)} y2={y(thr)} className="thr" />
        <line x1={pad} x2={W - pad} y1={y(0)} y2={y(0)} className="zero" />
        <path d={path} className="curve" />
        {c.crossed_t.map((t, i) => <line key={`c${i}`} x1={x(t)} x2={x(t)} y1={pad} y2={H - pad} className="cross" />)}
        {frame !== null && <line x1={x(frame / fps)} x2={x(frame / fps)} y1={pad} y2={H - pad} className="now" />}
      </svg>
      <div className="case-facts small">
        <span>up to <b>{c.max_over_m.toFixed(2)} m</b> over</span>
        <span>over {thr.toFixed(2)} m for {longest.toFixed(1)} s{longest > 0 && longest < dur ? ` (< ${dur.toFixed(1)} s)` : ""}</span>
        <span className="grow" />
        <button className="link" onClick={() => setStrip(!strip)}>{strip ? "hide frames" : "frames"}</button>
      </div>
      {strip && (
        <div className="case-strip">
          {stripFrames.map((f) => (
            <img key={f} src={img(f)} alt={`frame ${f}`} loading="lazy" className={f === frame ? "on" : ""}
              onClick={() => { setFrame(f); onSeek?.(f / fps - 0.5); }} />
          ))}
        </div>
      )}
    </div>
  );
}
