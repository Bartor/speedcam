import { useEffect, useState } from "react";
import { api, fmt, Project, useJob } from "../api";
import ProgressBar from "../components/ProgressBar";

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
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof api.projects>>>([]);
  const [dir, setDir] = useState<Awaited<ReturnType<typeof api.fs>> | null>(null);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.projects().then(setRecent).catch(() => {}); }, []);
  const browse = (p?: string) => api.fs(p).then(setDir).catch((e) => setErr(String(e)));
  useEffect(() => { browse(localStorage.getItem("speedcam.lastDir") ?? undefined); }, []);

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
    <div className="page two-col">
      <section className="panel grow">
        <h2>New project</h2>
        <p className="muted">The video is read from disk in place, nothing is uploaded. A 1080p H.264 proxy is built for scrubbing.</p>
        <div className="row">
          <input className="grow" placeholder="C:\path\to\DJI_0001.MP4" value={path}
            onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} />
          <button className="primary" disabled={!path || busy} onClick={create}>{busy ? "Probing…" : "Create project"}</button>
        </div>
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
                  <span>{e.dir ? "📁" : "🎞"} {e.name}</span>
                  {!e.dir && <span className="muted">{((e.size ?? 0) / 1e6).toFixed(0)} MB</span>}
                </li>
              ))}
              {dir.entries.length === 0 && <li className="muted">No folders or videos here</li>}
            </ul>
          </div>
        )}
      </section>
      <section className="panel side">
        <h2>Recent projects</h2>
        {recent.length === 0 && <p className="muted">None yet.</p>}
        <ul className="list">
          {recent.map((r) => (
            <li key={r.id} onClick={() => (window.location.hash = `#/p/${r.id}/calibrate`)}>
              <div>{r.name}</div>
              <div className="muted small ellipsis">{r.source}</div>
            </li>
          ))}
        </ul>
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
        <h2>{project.name}</h2>
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
      </section>
    </div>
  );
}
