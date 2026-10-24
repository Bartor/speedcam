import { useState } from "react";
import { api, fmt, Project, useJob } from "../api";
import ProgressBar from "../components/ProgressBar";
import "../components/library.css";

interface Props { project: Project; onChanged: (p: Project) => void; onDone: () => void }

type Field = { section: string; key: string; label: string; type: "num" | "text" | "bool" | "select"; options?: string[]; hint?: string };

const FIELDS: Field[] = [
  { section: "detection", key: "model", label: "YOLO model", type: "text", hint: "auto = picks the best of COCO, top-down and VisDrone per clip and fills in with the runner-up (any angle, ~2x GPU time) · yolo11x.pt = COCO, oblique views · yolo11x-obb.pt = DOTA, straight down" },
  { section: "detection", key: "conf", label: "Confidence", type: "num" },
  { section: "detection", key: "imgsz", label: "Inference size", type: "num" },
  { section: "detection", key: "half", label: "Fast mode (FP16)", type: "bool", hint: "~45% faster, but results aren't identical on reruns and speeds can shift by ~0.2 km/h. Off = FP32, repeatable." },
  { section: "detection", key: "batch", label: "Frames per GPU batch", type: "num", hint: "Detection only; frames are still detected independently and tracked in order. Lower it if the GPU runs out of memory." },
  { section: "detection", key: "tile", label: "Tile size (0 = off)", type: "num", hint: "Tile 4K frames when vehicles are small, e.g. 1024" },
  { section: "stabilization", key: "model", label: "Motion model", type: "select", options: ["similarity", "affine", "homography"] },
  { section: "stabilization", key: "features", label: "Registration", type: "select", options: ["klt", "orb", "sift"] },
  { section: "tracking", key: "lost_buffer_s", label: "Track buffer (s)", type: "num" },
  { section: "speed", key: "smooth_window_s", label: "Smoothing window (s)", type: "num" },
  { section: "speed", key: "max_accel", label: "Max accel (m/s²)", type: "num" },
  { section: "speed", key: "max_residual_m", label: "Max fit residual (m)", type: "num" },
  { section: "speed", key: "max_disagreement", label: "Max arc/line disagreement", type: "num" },
];

export default function ProcessScreen({ project, onChanged, onDone }: Props) {
  const cfg = project.config as any;
  const [jobId, setJobId] = useState<string | null>(project.jobs.find((j) => j.kind === "analyze")?.id ?? null);
  const [err, setErr] = useState<string | null>(null);
  const job = useJob(jobId, (j) => { api.project(project.id).then(onChanged); if (j.status === "done") setTimeout(onDone, 800); });
  const running = job.status === "running" || job.status === "queued" || job.status === "connecting";
  const [pvJobId, setPvJobId] = useState<string | null>(project.jobs.find((j) => j.kind === "preview")?.id ?? null);
  const pv = useJob(pvJobId);
  const pvRunning = ["running", "queued", "connecting"].includes(pv.status);
  const [pvFrame, setPvFrame] = useState(String(cfg.calibration.reference_frame));
  const [pvSeconds, setPvSeconds] = useState("3");
  const startPreview = async () => {
    setErr(null);
    try { setPvJobId((await api.preview(project.id, parseInt(pvFrame) || 0, parseFloat(pvSeconds) || 3)).id); }
    catch (e) { setErr(String(e)); }
  };
  const showPv = !!pvJobId && (pvRunning || !running);
  const live = showPv ? pv : job;

  const setStab = async (enabled: boolean) => {
    try { onChanged(await api.patch(project.id, { stabilization: { enabled } })); } catch (e) { setErr(String(e)); }
  };

  const setField = async (f: Field, raw: string | boolean) => {
    const v = f.type === "num" ? parseFloat(raw as string) : raw;
    if (f.type === "num" && !Number.isFinite(v)) return;
    try {
      onChanged(await api.patch(project.id, { [f.section]: { [f.key]: v } }));
    } catch (e) { setErr(String(e)); }
  };

  const setRange = async (key: "start_frame" | "end_frame", raw: string) => {
    const v = raw === "" ? (key === "end_frame" ? null : 0) : parseInt(raw);
    onChanged(await api.patch(project.id, { [key]: v }));
  };

  const start = async () => {
    setErr(null);
    try { setJobId((await api.analyze(project.id)).id); } catch (e) { setErr(String(e)); }
  };

  const p = job.progress;
  const eta = p.fps && p.total ? (p.total - p.done) / p.fps : null;
  return (
    <div className="page two-col">
      <section className="panel grow">
        <h2>Analysis</h2>
        <div className="row">
          <span>capture_fps <b>{cfg.capture_fps ?? "not set"}</b></span>
          <span>· frames {project.info.nb_frames}</span>
          <span>· calibration {project.calibration_result.error ? <span className="bad-text">incomplete</span> : `${fmt(project.calibration_result.cm_per_px, 3)} cm/px, gap ${fmt(project.calibration_result.line_gap_m, 2)} m`}</span>
        </div>
        {project.timing_warnings.map((w) => <div key={w} className="banner info small">{w}</div>)}
        {(() => {
          const c = cfg.calibration, r = project.calibration_result;
          const geometry = (c.paths ?? []).some((p: any) => p.points.length >= 2) || (c.line_a && c.line_b) ||
            (c.solid_lines ?? []).some((l: any) => l.points.length >= 2) || (c.red_lights ?? []).some((s: any) => s.stop_lines.length);
          const why = r.error || r.fit_ok === false ? "there's no usable calibration" : !geometry ? "nothing to measure is drawn (no path, lines A/B, stop line or solid line)" : null;
          return why && <div className="banner info small"><b>Tracking only:</b> {why}, so the analysis detects and tracks vehicles without measuring them. Add those later and press Recompute on the results: no need to analyse again.</div>;
        })()}
        <div className="row">
          {!running ? <button className="primary" onClick={start}>{project.status.results_json ? "Re-run analysis" : "Start analysis"}</button>
            : <button className="danger" onClick={() => jobId && api.cancel(jobId)}>Cancel</button>}
          {project.status.results_json && !running && <button onClick={onDone}>View results →</button>}
        </div>
        <div className="camera-mode">
          <span className="muted small">Camera</span>
          <div className="seg-ctl">
            <button className={cfg.stabilization.enabled ? "active" : ""} disabled={running}
              onClick={() => setStab(true)} title="Register every frame to the reference frame (KLT); needed when the drone sways or drifts">
              Hovering: stabilize
            </button>
            <button className={!cfg.stabilization.enabled ? "active" : ""} disabled={running}
              onClick={() => setStab(false)} title="Treat every frame as aligned with the reference frame; faster">
              Still: skip stabilization
            </button>
          </div>
          <span className="muted small">{cfg.stabilization.enabled
            ? "Every frame is aligned to the reference frame."
            : "Frames are assumed to line up with the reference frame. Any camera drift becomes position (and speed) error."}</span>
        </div>
        <div className="row small">
          <button disabled={pvRunning || running} onClick={startPreview}>Preview tracking</button>
          <span>from frame</span>
          <input type="number" className="narrow" value={pvFrame} onChange={(e) => setPvFrame(e.target.value)} />
          <span>for</span>
          <input type="number" className="narrow" value={pvSeconds} onChange={(e) => setPvSeconds(e.target.value)} />
          <span>s</span>
          <span className="muted">· a short detect + stabilize + track run to check the settings; results are not saved</span>
        </div>
        {err && <div className="banner error">{err}</div>}
        {showPv && (
          <>
            <ProgressBar label={`preview: ${pv.progress.stage ?? pv.status}`} done={pv.progress.done} total={pv.progress.total}
              extra={pv.progress.fps ? `· ${pv.progress.fps.toFixed(1)} fps` : ""} />
            {pv.end?.status === "error" && <div className="banner error"><b>{pv.end.error}</b></div>}
            {pv.end?.status === "done" && (
              <div className="banner ok small">
                Preview: {pv.end.result.detections_per_frame?.toFixed(1)} detections/frame · {pv.end.result.tracks} tracks
                ({pv.end.result.tracks_crossing_both} crossed both lines, {pv.end.result.tracks_crossing_a_only} only A,
                {" "}{pv.end.result.tracks_crossing_b_only} only B) · {pv.end.result.registration_failures} registration failures
                <DriftVerdict drift={pv.end.result.camera_drift} stabilized={cfg.stabilization.enabled} onSkip={() => setStab(false)} />
                {pv.end.result.vehicles?.length > 0 && <> · speeds: {pv.end.result.vehicles.map((v: any) => `#${v.track_id} ${v.speed_kmh.toFixed(1)}${v.valid ? "" : "?"}`).join(", ")} km/h</>}
              </div>
            )}
          </>
        )}
        {live.preview && <img className="preview" src={live.preview} alt="tracking preview" />}
        {jobId && !showPv && (
          <>
            <ProgressBar label={`${p.stage ?? job.status}`} done={p.done} total={p.total}
              extra={p.fps ? `· ${p.fps.toFixed(1)} fps${eta !== null ? ` · ETA ${Math.round(eta)} s` : ""}${p.tracks ? ` · ${p.tracks} tracks` : ""}` : ""} />
            {job.end?.status === "error" && <div className="banner error"><b>{job.end.error}</b><pre className="tb">{job.end.traceback}</pre></div>}
            {job.end?.status === "cancelled" && <div className="banner warn">Cancelled.</div>}
            {job.end?.status === "done" && <div className="banner ok">Done: {job.end.result?.n_valid} valid of {job.end.result?.n_vehicles} measured vehicles.</div>}
            <div className="log">
              {job.logs.map((l, i) => <div key={i} className={l.level === "warning" ? "warn-text" : ""}>{l.message}</div>)}
            </div>
          </>
        )}
      </section>
      <section className="panel side">
        <h3>Settings</h3>
        <div className="row wrap small">
          <span className="muted">Model:</span>
          {["auto", "visdrone", "yolo11x.pt", "yolo11x-obb.pt"].map((m) => (
            <button key={m} className={`chip ${cfg.detection.model === m ? "active" : ""}`}
              title={m === "auto" ? "Picks COCO, top-down or VisDrone per clip (top-down when filmed straight down) and fills in with the runner-up; any angle"
                : m === "visdrone" ? "yolo11x trained on VisDrone drone footage: best on oblique views (more distant cars, faster); weaker straight down" : undefined}
              onClick={() => setField(FIELDS[0], m)}>{m === "auto" ? "Auto (any angle)" : m === "visdrone" ? "VisDrone (trained)" : m}</button>
          ))}
        </div>
        <div className="form-grid compact">
          <label>Start frame</label>
          <input type="number" defaultValue={cfg.start_frame} onBlur={(e) => setRange("start_frame", e.target.value)} />
          <label>End frame</label>
          <input type="number" defaultValue={cfg.end_frame ?? ""} placeholder="end" onBlur={(e) => setRange("end_frame", e.target.value)} />
          {FIELDS.map((f) => {
            const val = cfg[f.section][f.key];
            return [
              <label key={`${f.key}-l`} title={f.hint}>{f.label}</label>,
              f.type === "bool" ? <input key={f.key} type="checkbox" checked={!!val} onChange={(e) => setField(f, e.target.checked)} />
                : f.type === "select" ? <select key={f.key} value={val} onChange={(e) => setField(f, e.target.value)}>{f.options!.map((o) => <option key={o}>{o}</option>)}</select>
                  : <input key={`${f.key}-${val}`} type={f.type === "num" ? "number" : "text"} step="any" defaultValue={val} title={f.hint}
                    onBlur={(e) => e.target.value !== String(val) && setField(f, e.target.value)} />,
            ];
          })}
        </div>
        <p className="muted small">Model weights are downloaded by Ultralytics on first use. Changing speed thresholds only needs “Recompute” on the results page.</p>
      </section>
    </div>
  );
}

/** After a preview run: is the camera still enough to skip stabilization? */
function DriftVerdict({ drift, stabilized, onSkip }: {
  drift?: { max_px: number; p95_px: number; max_m: number } | null; stabilized: boolean; onSkip: () => void;
}) {
  if (!stabilized) return <div className="muted">Camera drift not measured (stabilization is off). Switch to “Hovering” and preview again to measure it.</div>;
  if (!drift) return null;
  const cm = drift.max_m * 100;
  const still = drift.max_px < 0.5;
  return (
    <div className={still ? "" : "warn-text"}>
      Camera drift during preview: max {drift.max_px.toFixed(2)} px (≈ {cm.toFixed(1)} cm on the ground), p95 {drift.p95_px.toFixed(2)} px.{" "}
      {still ? <>The camera is effectively still: skipping stabilization is safe here. <button className="chip" onClick={onSkip}>Skip stabilization</button></>
        : drift.max_px < 2 ? "Small drift: skipping stabilization would add up to that much position error. Keep it on for the most accurate speeds."
          : "The camera moves: keep stabilization on."}
      {" "}<span className="muted">(Measured over the preview only; check a longer stretch if the drone was gusting.)</span>
    </div>
  );
}
