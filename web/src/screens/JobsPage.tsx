import { useEffect, useMemo, useState } from "react";
import Help from "../components/Help";
import { api, JobHistoryRow } from "../api";
import "../components/library.css";

const KIND: Record<string, string> = {
  analyze: "Analysis", render: "Render", recompute: "Recompute", preview: "Preview", proxy: "Proxy", scan: "Activity scan",
};
const SCREEN: Record<string, string> = { scan: "process", analyze: "results", render: "export", recompute: "results", preview: "process", proxy: "import" };

export const fmtBytes = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${Math.round(n / 1e3)} kB` : `${n} B`;
const fmtDur = (s: number) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`
  : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s.toFixed(s < 10 ? 1 : 0)} s`);
const day = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "long", day: "numeric" });
const clock = (t: number) => new Date(t * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

/** What the job came to, in a few words. */
function summary(j: JobHistoryRow): string {
  const r = j.result ?? {};
  if (j.status === "error") return j.error ?? "failed";
  if (j.status !== "done") return "";
  if (j.kind === "analyze" || j.kind === "recompute") return r.n_vehicles !== undefined ? `${r.n_valid} valid of ${r.n_vehicles} vehicles` : "";
  if (j.kind === "render") return r.width ? `${r.width}×${r.height} · ${r.output_fps ? Math.round(r.output_fps) : "?"} fps${r.playback_speed && r.playback_speed !== 1 ? ` · ${r.playback_speed}× speed` : ""}` : "";
  if (j.kind === "scan") return r.samples !== undefined ? `vehicles in ${r.with_vehicles} of ${r.samples} samples (${r.mode})` : "";
  if (j.kind === "preview") return r.tracks !== undefined ? `${r.tracks} tracks · ${r.detections_per_frame?.toFixed?.(1) ?? "?"} detections/frame` : "";
  return "";
}

/** Every job across projects, newest first, with the files each one produced. */
export default function JobsPage() {
  const [data, setData] = useState<{ jobs: JobHistoryRow[]; artifacts_bytes: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [project, setProject] = useState("all");
  const [kind, setKind] = useState("all");
  const [withFiles, setWithFiles] = useState(false);

  const load = () => api.jobHistory().then((d) => { setData(d); setErr(null); }).catch((e) => setErr(String(e)));
  const busy = !!data?.jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    load();
    const t = window.setInterval(load, busy ? 2000 : 10000);
    return () => window.clearInterval(t);
  }, [busy]);

  const projects = useMemo(() => {
    const m = new Map<string, string>();
    data?.jobs.forEach((j) => m.set(j.project_id, j.project_name ?? j.project_id));
    return [...m.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [data]);
  const rows = (data?.jobs ?? []).filter((j) => (project === "all" || j.project_id === project) &&
    (kind === "all" || j.kind === kind) && (!withFiles || j.files.length > 0));

  const removeFiles = async (j: JobHistoryRow) => {
    const size = j.files.reduce((a, f) => a + f.size, 0);
    if (!confirm(`Delete the files kept for this ${KIND[j.kind]?.toLowerCase() ?? j.kind} (${fmtBytes(size)})?\n\nThe job stays in the list; the project's current results and render are not touched.`)) return;
    try { await api.deleteJobFiles(j.id); load(); } catch (e) { setErr(String(e)); }
  };

  let lastDay = "";
  return (
    <div className="page jobs-page">
      <section className="panel">
        <div className="row wrap">
          <h2 className="grow">Jobs<Help topic="jobs" /> {data && <span className="muted small">({rows.length}{rows.length !== data.jobs.length ? ` of ${data.jobs.length}` : ""})</span>}</h2>
          <select value={project} onChange={(e) => setProject(e.target.value)} aria-label="Project">
            <option value="all">All projects</option>
            {projects.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
          </select>
          <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Kind">
            <option value="all">All kinds</option>
            {Object.entries(KIND).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
          <label className="small"><input type="checkbox" checked={withFiles} onChange={(e) => setWithFiles(e.target.checked)} /> with files only</label>
        </div>
        <p className="muted small">
          Each finished analysis, recompute and render keeps its own copy of what it produced, so earlier runs stay
          downloadable after the project's results or video are replaced.
          {data ? ` Kept files take ${fmtBytes(data.artifacts_bytes)}.` : ""} Jobs from before this page existed have no files.
        </p>
        {err && <div className="banner error">{err}</div>}
        {!data && !err && <p className="muted">Loading…</p>}
        {data && rows.length === 0 && <p className="empty-note">No jobs{data.jobs.length ? " match the filter" : " yet"}.</p>}
        <div className="jobs-list">
          {rows.map((j) => {
            const d = day(j.created);
            const head = d !== lastDay ? <h3 className="jobs-day">{d}</h3> : null;
            lastDay = d;
            const dur = j.started && j.finished ? j.finished - j.started : j.started && j.status === "running" ? Date.now() / 1000 - j.started : null;
            const p = j.progress;
            const pc = p?.total ? Math.min(100, (100 * (p.done ?? 0)) / p.total) : null;
            return (
              <div key={j.id}>
                {head}
                <div className={`job-row status-${j.status}`}>
                  <span className="job-time">{clock(j.created)}</span>
                  <span className={`job-kind kind-${j.kind}`}>{KIND[j.kind] ?? j.kind}</span>
                  <a className="job-project ellipsis" href={j.project_name ? `#/p/${j.project_id}/${SCREEN[j.kind] ?? "import"}` : undefined}
                    title={j.project_name ? "Open the project" : "Project deleted"}>{j.project_name ?? `${j.project_id} (deleted)`}</a>
                  <span className={`job-status ${j.status}`}>
                    {j.status === "running" ? `running${pc !== null ? ` ${pc.toFixed(0)}%` : ""}` : j.status}
                    {j.status === "running" && p?.stage ? <span className="muted"> · {p.stage}</span> : null}
                  </span>
                  <span className="job-dur muted">{dur !== null ? fmtDur(dur) : ""}</span>
                  <span className={`job-summary ellipsis ${j.status === "error" ? "bad-text" : "muted"}`} title={summary(j)}>{summary(j)}</span>
                  <span className="job-files">
                    {j.files.map((f) => (
                      <a key={f.name} className="file-chip" href={api.jobFileUrl(j.id, f.name)} title={`Download ${f.name} (${fmtBytes(f.size)})`}>
                        <span className="dl-ico">↓</span>{f.name}<span className="muted"> {fmtBytes(f.size)}</span>
                      </a>
                    ))}
                    {j.files.length > 0 && <button className="x" title="Delete this job's kept files" onClick={() => removeFiles(j)}>✕</button>}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
