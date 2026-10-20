import { useEffect, useState } from "react";
import RotateButtons from "../components/RotateButtons";
import { api, fmt, JobRecord, Project, timeAgo, useJob } from "../api";
import ProgressBar from "../components/ProgressBar";
import "../components/library.css";
import ProjectLibrary from "../components/ProjectLibrary";


interface Props {
  project: Project | null;
  onProject: (p: Project) => void;
  onChanged: (p: Project) => void;
  onNext: () => void;
}

export default function ImportScreen({ project, onProject, onChanged, onNext }: Props) {
  return project ? (
    <ProjectSetup project={project} onChanged={onChanged} onNext={onNext} />
  ) : (
    <Picker onProject={onProject} />
  );
}

function Picker({ onProject }: { onProject: (p: Project) => void }) {
  const [dir, setDir] = useState<Awaited<ReturnType<typeof api.fs>> | null>(null);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const browse = (p?: string) => api.fs(p).then(setDir).catch((e) => setErr(String(e)));
  useEffect(() => { browse(localStorage.getItem("speedcam.lastDir") ?? undefined); }, []);

  // a video that already has a project opens that project (one project per video)
  const selected = dir?.entries.find((e) => e.path === path);
  const owner = selected?.projects?.[0];
  const create = async () => {
    setBusy(true);
    setErr(null);
    try {
      onProject(await api.create(path.trim().replace(/^"|"$/g, "")));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page two-col import-page">
      <ProjectLibrary />
      <section className="panel side new-project">
        <h2>New project</h2>
        <p className="muted">The video is read from disk in place, nothing is uploaded. A 1080p H.264 proxy is built for scrubbing.</p>
        <div className="row">
          <input className="grow" placeholder="C:\path\to\DJI_0001.MP4" value={path}
            onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} />
          <button className="primary" disabled={!path || busy} onClick={create}>{busy ? "Probing…" : owner ? "Open project" : "Create project"}</button>
        </div>
        {owner && <p className="small muted">This video already has a project, <b>{owner.name}</b>: it opens instead of making another.</p>}
        {err && <div className="banner error">{err}</div>}
        {dir && (
          <div className="browser">
            <div className="row small">
              <button onClick={() => browse(dir.parent)}>↑ Up</button>
              <code className="grow ellipsis">{dir.path}</code>
            </div>
            <ul>
              {dir.entries.map((e) => (
                <li key={e.path} className={e.path === path ? "sel" : ""}
                  onClick={() => {
                    if (e.dir) browse(e.path);
                    else { setPath(e.path); localStorage.setItem("speedcam.lastDir", dir.path); }
                  }}
                  onDoubleClick={() => !e.dir && create()}>
                  <span className="grow ellipsis">{e.dir ? "📁" : "🎞"} {e.name}</span>
                  {!e.dir && e.projects && e.projects.length > 0 && (
                    <span className="tag ok" title={`Has a project: ${e.projects.map((x) => x.name).join(", ")}. Opening the video opens it.`}>
                      {e.projects.length > 1 ? `${e.projects.length} projects` : "project"}
                    </span>
                  )}
                  {!e.dir && <span className="muted">{((e.size ?? 0) / 1e6).toFixed(0)} MB</span>}
                </li>
              ))}
              {dir.entries.length === 0 && <li className="muted">No folders or videos here</li>}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}

function ProjectSetup({ project, onChanged, onNext }: { project: Project; onChanged: (p: Project) => void; onNext: () => void }) {
  const info = project.info;
  const cfg = project.config;
  const [fps, setFps] = useState(String(cfg.capture_fps ?? info.suggested_capture_fps));
  const [dur, setDur] = useState(cfg.real_duration_s ? String(cfg.real_duration_s) : "");
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState(project.name);
  const [relinkPath, setRelinkPath] = useState("");
  const [relinkMsg, setRelinkMsg] = useState<string[] | null>(null);
  const [history, setHistory] = useState<JobRecord[]>([]);
  useEffect(() => { api.jobs(project.id).then(setHistory).catch(() => {}); }, [project.id, project.status]);
  const rename = async () => {
    if (name.trim() && name !== project.name) onChanged(await api.patch(project.id, { name: name.trim() }));
  };
  const relink = async () => {
    try {
      const r = await api.relink(project.id, relinkPath.trim().replace(/^"|"$/g, ""));
      setRelinkMsg(r.relink_warnings);
      onChanged(r);
    } catch (e) { setErr(String(e)); }
  };
  const proxyJobId = project.proxy_job ?? project.jobs.find((j) => j.kind === "proxy")?.id;
  const proxy = useJob(proxyJobId, () => api.project(project.id).then(onChanged));

  const save = async () => {
    const f = parseFloat(fps);
    if (!(f > 0)) return setErr("capture_fps must be a positive number");
    try {
      onChanged(await api.patch(project.id, { capture_fps: f, real_duration_s: dur ? parseFloat(dur) : null }));
      setErr(null);
      return true;
    } catch (e) {
      setErr(String(e));
    }
  };

  const f = parseFloat(fps);
  const real = f > 0 ? info.nb_frames / f : NaN;
  return (
    <div className="page two-col">
      <section className="panel grow">
        <input className="title-input" value={name} onChange={(e) => setName(e.target.value)}
          onBlur={rename} onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} title="Rename project" />
        {!project.source_exists && (
          <div className="banner error">
            <b>The source video is missing</b> (moved, renamed or on a drive that isn't connected): <code>{info.path}</code>
            <div className="row">
              <input className="grow" placeholder="New path to the same video" value={relinkPath} onChange={(e) => setRelinkPath(e.target.value)} />
              <button className="primary" disabled={!relinkPath} onClick={relink}>Relink</button>
            </div>
          </div>
        )}
        {relinkMsg && (relinkMsg.length
          ? <div className="banner warn">Relinked, but this file differs from the original: {relinkMsg.join("; ")}. Calibration and results may not match.</div>
          : <div className="banner ok">Relinked: same resolution and frame count.</div>)}
        <table className="kv">
          <tbody>
            <tr><td>Source</td><td><code>{info.path}</code></td></tr>
            <tr><td>Resolution</td><td>{info.width} × {info.height}</td></tr>
            <tr><td>Codec</td><td>{info.codec} · {info.pix_fmt} · {info.bit_depth}-bit {info.color_transfer && `· ${info.color_transfer}`}</td></tr>
            <tr><td>Container fps</td><td>{fmt(info.container_fps, 3)}</td></tr>
            <tr><td>Frames</td><td>{info.nb_frames}</td></tr>
            <tr><td>Container duration</td><td>{fmt(info.duration_s, 3)} s</td></tr>
            {info.srt_fps && <tr><td>Sidecar SRT rate</td><td>{fmt(info.srt_fps, 2)} fps</td></tr>}
          </tbody>
        </table>
        {info.warnings.map((w) => <div key={w} className={`banner ${info.slowmo_suspect ? "warn" : "info"}`}>{w}</div>)}

        <h3>Timing</h3>
        <p className="muted">Real time is computed as <code>frame / capture_fps</code>. The container rate is not trusted: DJI stores 100/120 fps clips as slow motion.</p>
        <div className="form-grid">
          <label>capture_fps (real recording rate)</label>
          <div className="row">
            <input type="number" step="any" value={fps} onChange={(e) => setFps(e.target.value)} />
            {[25, 30, 50, 60, 100, 120].map((v) => <button key={v} className="chip" onClick={() => setFps(String(v))}>{v}</button>)}
          </div>
          <label title="Only how the video is shown and rendered: calibration and results are unaffected">View rotation</label>
          <div><RotateButtons project={project} onChanged={onChanged} /></div>
          <label>Known real duration (s, optional check)</label>
          <input type="number" step="any" value={dur} onChange={(e) => setDur(e.target.value)} placeholder="e.g. from flight log" />
          <label>Resulting real duration</label>
          <div>{fmt(real, 3)} s {Number.isFinite(real) && Math.abs(f - info.container_fps) / info.container_fps > 0.02 &&
            <span className="tag warn">slow-motion ×{fmt(f / info.container_fps, 2)}</span>}</div>
        </div>
        {project.timing_warnings.map((w) => <div key={w} className="banner warn">{w}</div>)}
        {err && <div className="banner error">{err}</div>}
        <div className="row end">
          <button onClick={save}>Save</button>
          <button className="primary" onClick={async () => { if (await save()) onNext(); }}>Confirm and calibrate →</button>
        </div>
      </section>
      <section className="panel side">
        <h3>Browser proxy</h3>
        {project.status.proxy_mp4 ? (
          <p>Proxy ready ✓</p>
        ) : proxy.status === "error" ? (
          <>
            <div className="banner error">{proxy.end?.error}</div>
            <button onClick={() => api.rebuildProxy(project.id).then(() => api.project(project.id).then(onChanged))}>Retry</button>
          </>
        ) : (
          <ProgressBar label={proxy.progress.stage ?? "queued"} done={proxy.progress.done} total={proxy.progress.total} />
        )}
        <p className="muted small">Calibration clicks use full-resolution stills decoded from the original, not the proxy.</p>
        <h3>History</h3>
        {history.length === 0 && <p className="muted small">No jobs yet.</p>}
        <ul className="job-history">
          {history.map((j) => (
            <li key={j.id} title={j.error ?? ""}>
              <span className={`status ${j.status === "done" ? "ok" : j.status === "error" ? "bad" : "warn"}`}>
                {j.status === "done" ? "✓" : j.status === "error" ? "✕" : "•"}
              </span>
              <span className="grow">{j.kind}{j.status !== "done" ? ` (${j.status})` : ""}
                {j.kind === "analyze" && j.result?.n_vehicles !== undefined && <span className="muted"> · {j.result.n_valid}/{j.result.n_vehicles} valid</span>}
              </span>
              <span className="muted">{timeAgo(j.finished ?? j.started ?? j.created)}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
