import { useEffect, useState } from "react";
import RotatedVideo, { VideoControls } from "../components/RotatedVideo";
import Timeline from "../components/Timeline";
import Help, { Advanced } from "../components/Help";
import { api, fmt, Project, useJob, useVideoFrame } from "../api";
import ProgressBar from "../components/ProgressBar";
import ActivityPanel from "../components/ActivityPanel";
import "../components/library.css";

interface Props { project: Project; onChanged: (p: Project) => void; onDone: () => void; onSetup: () => void }

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
];

export default function AnalyzeScreen({ project, onChanged, onDone, onSetup }: Props) {
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
  const nb = project.info.nb_frames;
  const fps = cfg.capture_fps || project.info.container_fps;
  const parts: [number, number][] | null = cfg.parts ?? null;
  const covered = parts ? parts.reduce((a, [x, y]) => a + (y - x), 0) : nb;
  const [suggested, setSuggested] = useState<[number, number][]>([]);
  useEffect(() => { api.getScan(project.id).then((s) => setSuggested(s.suggested)).catch(() => {}); }, [project.id, project.jobs.length, cfg.scan]);
  const setParts = async (pp: [number, number][] | null) => {
    try { onChanged(await api.patch(project.id, { parts: pp })); } catch (e) { setErr(String(e)); }
  };
  const usesActivity = !!parts && suggested.length > 0 && JSON.stringify(parts) === JSON.stringify(suggested);
  const c = cfg.calibration, r = project.calibration_result;
  const geometry = (c.paths ?? []).some((pp: any) => pp.points.length >= 2) || (c.line_a && c.line_b) ||
    (c.solid_lines ?? []).some((l: any) => l.points.length >= 2) || (c.red_lights ?? []).some((s: any) => s.stop_lines.length);
  const trackingOnly = r.error || r.fit_ok === false ? "there's no usable calibration" : !geometry ? "nothing to measure is set up (no path, lines A/B, stop line or solid line)" : null;

  return (
    <div className="analyze">
      <div className="an-main">
        <section className="panel">
          <div className="panel-head">
            <h3>Analyze<Help topic="process.analysis" /></h3>
            <span className="tag">{fmt(fps, 2)} fps real</span>
            <span className="tag" title="Scale, from Setup">{r.error ? "no calibration" : `${fmt(r.cm_per_px, 3)} cm/px`}</span>
            <span className="tag" title="From Setup → Limits & tolerances">limit {cfg.render?.speed_limit_kmh ?? 50} km/h</span>
            <span className="grow" />
            <button className="ghost" onClick={onSetup}>← Setup</button>
          </div>
          {project.timing_warnings.map((w) => <div key={w} className="banner info small">{w}</div>)}
          {trackingOnly && <div className="banner info small"><b>Tracking only:</b> {trackingOnly}, so the analysis detects and tracks vehicles without
            measuring them. Set them up later and press Recompute in Review: no need to analyse again.</div>}
          <ClipPreview project={project} onChanged={onChanged} />
        </section>
        {err && <div className="banner error">{err}</div>}
        {showPv && (
          <section className="panel">
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
            {pv.preview && <img className="preview" src={pv.preview} alt="tracking preview" />}
          </section>
        )}
        {jobId && !showPv && (
          <section className="panel">
            <ProgressBar label={`${p.stage ?? job.status}`} done={p.done} total={p.total}
              extra={p.fps ? `· ${p.fps.toFixed(1)} fps${eta !== null ? ` · ETA ${Math.round(eta)} s` : ""}${p.tracks ? ` · ${p.tracks} tracks` : ""}` : ""} />
            {job.end?.status === "error" && <div className="banner error"><b>{job.end.error}</b><pre className="tb">{job.end.traceback}</pre></div>}
            {job.end?.status === "cancelled" && <div className="banner warn">Cancelled.</div>}
            {job.end?.status === "done" && <div className="banner ok">Done: {job.end.result?.n_valid} valid of {job.end.result?.n_vehicles} measured vehicles.</div>}
            {job.preview && <img className="preview" src={job.preview} alt="tracking preview" />}
            <div className="log">
              {job.logs.map((l, i) => <div key={i} className={l.level === "warning" ? "warn-text" : ""}>{l.message}</div>)}
            </div>
          </section>
        )}
      </div>

      <aside className="an-side">
        <section className="panel">
          <ActivityPanel project={project} onChanged={onChanged} disabled={running} />
        </section>

        <section className="panel">
          <h3><span className="step-badge">2</span>Choose what to analyse</h3>
          <p className="parts-summary">{parts ? <>Analysing <b>{parts.length} part{parts.length === 1 ? "" : "s"}</b>, {clockOf(covered / fps)} of {clockOf(nb / fps)}
            {" "}({Math.round((100 * covered) / nb)} %){usesActivity ? ": the parts with vehicles" : ""}</> : <>Analysing <b>the whole clip</b>, {clockOf(nb / fps)}</>}</p>
          <div className="row wrap">
            <button className={usesActivity ? "chip active" : "chip"} disabled={running || !suggested.length} onClick={() => setParts(suggested)}
              title={suggested.length ? "The parts with moving vehicles found by the scan" : "Find the activity first (step 1)"}>Only the activity</button>
            <button className={!parts ? "chip active" : "chip"} disabled={running} onClick={() => setParts(null)}>The whole clip</button>
          </div>
          <p className="muted small">Or pick parts on the timeline's bottom lane: drag over blue to cut a stretch out, elsewhere to add one; I / O set
            in and out points at the playhead. Only these parts are decoded and analysed (and by default rendered); tracking starts fresh in each.</p>
        </section>

        <section className="panel">
          <h3><span className="step-badge">3</span>Check and run</h3>
          <div className="camera-mode">
            <span className="muted small">Camera<Help topic="process.camera" /></span>
            <div className="seg-ctl">
              <button className={cfg.stabilization.enabled ? "active" : ""} disabled={running}
                onClick={() => setStab(true)} title="Register every frame to the reference frame (KLT); needed when the drone sways or drifts">Hovering</button>
              <button className={!cfg.stabilization.enabled ? "active" : ""} disabled={running}
                onClick={() => setStab(false)} title="Treat every frame as aligned with the reference frame; faster">Still</button>
            </div>
          </div>
          <div className="row wrap small">
            <span className="muted">Detector</span>
            {["auto", "visdrone", "yolo11x.pt", "yolo11x-obb.pt"].map((m) => (
              <button key={m} className={`chip ${cfg.detection.model === m ? "active" : ""}`} disabled={running}
                title={m === "auto" ? "Picks COCO, top-down or VisDrone per clip (top-down when filmed straight down) and fills in with the runner-up; any angle"
                  : m === "visdrone" ? "yolo11x trained on VisDrone drone footage: best on oblique views (more distant cars, faster); weaker straight down"
                    : m === "yolo11x-obb.pt" ? "Oriented boxes (DOTA), straight-down footage: the box is the vehicle's outline" : "COCO: oblique views"}
                onClick={() => setField(FIELDS[0], m)}>{m === "auto" ? "Auto" : m === "visdrone" ? "VisDrone" : m === "yolo11x-obb.pt" ? "Top-down (OBB)" : "COCO"}</button>
            ))}
          </div>
          <div className="row small wrap">
            <button disabled={pvRunning || running} onClick={startPreview}>Preview tracking</button><Help topic="process.preview" />
            <span>from frame</span>
            <input type="number" className="narrow" value={pvFrame} onChange={(e) => setPvFrame(e.target.value)} />
            <span>for</span>
            <input type="number" className="narrow" value={pvSeconds} onChange={(e) => setPvSeconds(e.target.value)} />
            <span>s</span>
          </div>
          <p className="muted small">A few seconds of detection, stabilization and tracking to check the settings; nothing is saved.</p>
          <Advanced id="process">
            <div className="form-grid compact">
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
              <label>Start frame</label>
              <input type="number" defaultValue={cfg.start_frame} onBlur={(e) => setRange("start_frame", e.target.value)} />
              <label>End frame</label>
              <input type="number" defaultValue={cfg.end_frame ?? ""} placeholder="end" onBlur={(e) => setRange("end_frame", e.target.value)} />
            </div>
            <p className="muted small">Detector, stabilization and tracking. The defaults suit most footage. Model weights download on first use.</p>
          </Advanced>
          <div className="row">
            {!running ? <button className="primary grow lg" onClick={start}>{project.status.results_json ? "Analyse again" : "Start analysis"}</button>
              : <button className="danger grow" onClick={() => jobId && api.cancel(jobId)}>Cancel</button>}
            {project.status.results_json && !running && <button onClick={onDone}>Review →</button>}
          </div>
        </section>
      </aside>
    </div>
  );
}

const clockOf = (s: number) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`;
};

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

/** The clip, scrubbable: a small player and the timeline (parts to analyse, activity). */
function ClipPreview({ project, onChanged }: { project: Project; onChanged: (p: Project) => void }) {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const { frame } = useVideoFrame(video, project.info.container_fps);
  const fps = project.config.capture_fps || project.info.container_fps;
  const rot = project.config.rotation ?? 0;
  const seek = (f: number) => {
    if (!video) return;
    video.pause();
    video.currentTime = (Math.max(0, Math.min(project.info.nb_frames - 1, f)) + 0.5) / project.info.container_fps;
  };
  if (!project.status.proxy_mp4) return <p className="muted small">The clip preview appears when the proxy is built.</p>;
  return (
    <div className="clip-preview">
      <div ref={setBox} className={`clip-video${rot ? " rotated" : ""}`}>
        <RotatedVideo rotation={rot} videoRef={setVideo} src={api.proxyUrl(project.id)} muted playsInline preload="auto"
          onClick={(e) => { const v = e.currentTarget; v.paused ? v.play() : v.pause(); }} />
      </div>
      <VideoControls video={video} containerFps={project.info.container_fps} fullscreenEl={box}
        realTime={(t) => (t * project.info.container_fps) / fps} />
      <Timeline project={project} frame={frame} onSeek={seek} onProjectChange={onChanged} />
    </div>
  );
}
