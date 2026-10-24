import { useEffect, useMemo, useState } from "react";
import { screenOf } from "../steps";
import { api, fmt, LibraryProject, timeAgo } from "../api";
import { ActiveJob, pct, useActiveJobs } from "./JobsIndicator";

/** All projects in the data folder, newest-opened first; click to resume where you left off. */
export default function ProjectLibrary() {
  const [items, setItems] = useState<LibraryProject[] | null>(null);
  const [q, setQ] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const { jobs } = useActiveJobs();
  const activeCount = jobs.length;
  // refresh rows (stats, status) whenever a job finishes
  useEffect(() => { load(); }, [activeCount]);

  const load = () => api.projects().then(setItems).catch((e) => setErr(String(e)));
  useEffect(() => { load(); }, []);

  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (items ?? []).filter((p) => !s || p.name.toLowerCase().includes(s) || p.source.toLowerCase().includes(s));
  }, [items, q]);

  const remove = async (p: LibraryProject) => {
    if (!confirm(`Delete project "${p.name}"?\n\nThis removes its calibration, results, proxy and renders. The source video is not touched.`)) return;
    try { await api.remove(p.id); load(); } catch (e) { setErr(String(e)); }
  };

  return (
    <section className="panel library">
      <div className="row">
        <h2 className="grow">Projects {items && <span className="muted small">({items.length})</span>}</h2>
        {items && items.length > 4 && <input className="lib-search" placeholder="Filter by name or file" value={q} onChange={(e) => setQ(e.target.value)} />}
      </div>
      {err && <div className="banner error">{err}</div>}
      {items === null && <p className="muted">Loading…</p>}
      {items?.length === 0 && <p className="muted">No projects yet. Pick a video on the left to create one.</p>}
      <div className="lib-grid">
        {shown.map((p) => <Card key={p.id} p={p} job={jobs.find((j) => j.project_id === p.id)} onDelete={() => remove(p)}
          twins={(items ?? []).filter((x) => x.id !== p.id && sameFile(x.source, p.source)).length} />)}
      </div>
    </section>
  );
}

/** Same video file, whatever the case or slashes (Windows paths). */
const sameFile = (a: string, b: string) => a.replace(/\//g, "\\").toLowerCase() === b.replace(/\//g, "\\").toLowerCase();

function Card({ p, job, onDelete, twins }: { p: LibraryProject; job?: ActiveJob; onDelete: () => void; twins: number }) {
  const [thumbOk, setThumbOk] = useState(true);
  const screen = p.last_screen ? screenOf(p.last_screen) : p.has_results ? "review" : p.calibrated ? "analyze" : "setup";
  const open = () => (window.location.hash = `#/p/${p.id}/${screen}`);
  const file = p.source.split(/[\\/]/).pop();
  return (
    <div className="lib-card" role="button" tabIndex={0} onClick={open} onKeyDown={(e) => e.key === "Enter" && open()}>
      <div className="lib-thumb">
        {thumbOk && p.source_exists ? <img src={api.thumbUrl(p.id)} alt="" loading="lazy" onError={() => setThumbOk(false)} />
          : <span className="muted small">{p.source_exists ? "no preview" : "video missing"}</span>}
        {job && (
          <div className="lib-job" title={job.progress.stage ?? job.status}>
            <span className="small">{job.status === "running" ? `${job.kind} ${pct(job).toFixed(0)}%` : `${job.kind} queued`}</span>
            <span className="jobs-ind-bar wide"><span style={{ width: `${pct(job)}%` }} /></span>
          </div>
        )}
        {p.has_results && !job ? (
          <span className="lib-badge">{p.n_valid}/{p.n_vehicles} valid</span>
        ) : null}
      </div>
      <div className="lib-body">
        <div className="lib-title ellipsis" title={p.name}>{p.name}</div>
        <div className="muted small ellipsis" title={p.source}>{file}</div>
        <div className="small lib-meta">
          {p.width}×{p.height}{p.capture_fps ? ` · ${fmt(p.capture_fps, 2)} fps` : ""}{p.duration_s ? ` · ${fmt(p.duration_s, 1)} s` : ""}
        </div>
        {p.has_results ? (
          <div className="small" title="Of each vehicle's max speed">max: mean <b>{fmt(p.mean_kmh)}</b> · p85 <b>{fmt(p.p85_kmh)}</b> km/h</div>
        ) : (
          <div className="small muted">{p.calibrated ? "calibrated, not analysed" : p.capture_fps ? "needs calibration" : "needs capture fps"}</div>
        )}
        <div className="row small lib-foot">
          {!p.source_exists && <span className="tag bad" title="Open the project to point it at the moved file">video missing</span>}
          {p.has_render ? <span className="tag">rendered</span> : null}
          {twins > 0 && <span className="tag warn" title="Other projects use the same video. New ones aren't made any more; delete the ones you don't need.">same video ×{twins + 1}</span>}
          <span className="muted grow">opened {timeAgo(p.opened)}</span>
          <button className="x" title="Delete project" onClick={(e) => { e.stopPropagation(); onDelete(); }}>✕</button>
        </div>
      </div>
    </div>
  );
}
