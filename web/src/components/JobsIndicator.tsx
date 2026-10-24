import { useEffect, useState } from "react";

export interface ActiveJob {
  id: string; project_id: string; project_name: string; kind: string; status: string;
  created: number; started: number | null; progress: { stage?: string; done?: number; total?: number; fps?: number };
}

type Listener = (jobs: ActiveJob[], slots: number) => void;
const listeners = new Set<Listener>();
let latest: { jobs: ActiveJob[]; slots: number } = { jobs: [], slots: 1 };
let timer: number | undefined;

/** One poller for the whole app: fast while something runs, slow when idle. */
async function poll() {
  try {
    const r = await fetch("/api/jobs/active");
    if (r.ok) latest = await r.json();
  } catch { /* server restarting */ }
  listeners.forEach((l) => l(latest.jobs, latest.slots));
  updateTitle(latest.jobs);
  timer = window.setTimeout(poll, latest.jobs.length ? 1000 : 4000);
}

export function useActiveJobs(): { jobs: ActiveJob[]; slots: number } {
  const [state, setState] = useState(latest);
  useEffect(() => {
    const l: Listener = (jobs, slots) => setState({ jobs, slots });
    listeners.add(l);
    if (timer === undefined) poll();
    return () => { listeners.delete(l); };
  }, []);
  return state;
}

export const pct = (j: ActiveJob) =>
  j.progress.total ? Math.min(100, (100 * (j.progress.done ?? 0)) / j.progress.total) : 0;

/** Browser tab title shows overall progress, so it can be watched from another tab. */
function updateTitle(jobs: ActiveJob[]) {
  const running = jobs.filter((j) => j.status === "running");
  if (!running.length) { document.title = "speedcam"; return; }
  const p = running.reduce((a, j) => a + pct(j), 0) / running.length;
  document.title = `${Math.floor(p)}% · ${running.length} job${running.length > 1 ? "s" : ""} · speedcam`;
}

const LABEL: Record<string, string> = { proxy: "proxy", analyze: "analysis", render: "render", preview: "preview", scan: "activity scan" };

/** Top-bar summary of everything running or queued, across projects. */
export default function JobsIndicator() {
  const { jobs, slots } = useActiveJobs();
  const [open, setOpen] = useState(false);
  if (!jobs.length) return null;
  const running = jobs.filter((j) => j.status === "running");
  const queued = jobs.length - running.length;
  const avg = running.length ? running.reduce((a, j) => a + pct(j), 0) / running.length : 0;
  return (
    <div className="jobs-ind" onMouseLeave={() => setOpen(false)}>
      <button className="jobs-ind-btn" onClick={() => setOpen(!open)} title="Running and queued jobs">
        <span className="jobs-ind-dot" />
        {running.length} running{queued ? ` · ${queued} queued` : ""}
        <span className="jobs-ind-bar"><span style={{ width: `${avg}%` }} /></span>
      </button>
      {open && (
        <div className="jobs-ind-menu">
          <div className="muted small">{slots} jobs at a time · one analysis on the GPU · one job per project</div>
          {jobs.map((j) => (
            <a key={j.id} className="jobs-ind-row" href={`#/p/${j.project_id}/${j.kind === "render" ? "export" : j.kind === "proxy" ? "import" : "process"}`}>
              <div className="row small">
                <span className="grow ellipsis">{j.project_name}</span>
                <span className="muted">{LABEL[j.kind] ?? j.kind}</span>
              </div>
              {j.status === "running" ? (
                <div className="jobs-ind-bar wide"><span style={{ width: `${pct(j)}%` }} /></div>
              ) : <div className="muted small">queued</div>}
              <div className="muted small">
                {j.progress.stage ?? j.status}
                {j.progress.total ? ` · ${pct(j).toFixed(0)}%` : ""}{j.progress.fps ? ` · ${j.progress.fps.toFixed(1)} fps` : ""}
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
