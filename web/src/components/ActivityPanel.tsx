import { useEffect, useRef, useState } from "react";
import Help from "./Help";
import { api, Project, ScanResult, useJob } from "../api";
import ProgressBar from "./ProgressBar";

const clock = (s: number) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`;
};

/** Activity scan: a timeline of vehicles in the video, and the parts to analyse (the rest is skipped). */
export default function ActivityPanel({ project, onChanged, disabled }: { project: Project; onChanged: (p: Project) => void; disabled: boolean }) {
  const [data, setData] = useState<ScanResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(project.jobs.find((j) => j.kind === "scan")?.id ?? null);
  const load = () => api.getScan(project.id).then(setData).catch((e) => setErr(String(e)));
  const job = useJob(jobId, () => { load(); api.project(project.id).then(onChanged); });
  const running = ["running", "queued", "connecting"].includes(job.status);
  useEffect(() => { load(); }, [project.id]);

  const sc = data?.scan;
  const parts = project.config.parts ?? null;
  const fps = sc?.container_fps ?? project.info.container_fps;
  const total = sc?.nb_frames ?? project.info.nb_frames;
  const s = (project.config as any).scan ?? {};
  const covered = (pp: [number, number][]) => pp.reduce((a, [x, y]) => a + (y - x), 0);

  const start = async () => {
    setErr(null);
    try { setJobId((await api.scan(project.id)).id); } catch (e) { setErr(String(e)); }
  };
  const setParts = async (pp: [number, number][] | null) => {
    try { onChanged(await api.patch(project.id, { parts: pp })); } catch (e) { setErr(String(e)); }
  };
  const setScan = async (patch: Record<string, number>) => {
    try {
      const p = await api.patch(project.id, { scan: patch });
      const d = await api.getScan(project.id);
      setData(d);
      // parts already chosen follow the new padding/merging
      onChanged(parts ? await api.patch(project.id, { parts: d.suggested }) : p);
    } catch (e) { setErr(String(e)); }
  };

  return (
    <div className="activity">
      <div className="row">
        <h3 className="grow">Activity<Help topic="process.activity" /></h3>
        <button disabled={running || disabled} onClick={start}
          title="A quick pass over the whole video at low resolution (about one frame a second) that finds where vehicles are">
          {sc ? "Scan again" : "Scan for vehicles"}
        </button>
      </div>
      {!sc && !running && (
        <p className="muted small">For long videos (CCTV, long hovers): find the parts with vehicles first, then analyse and
          render only those. The scan reads about one frame a second ({clock(total / fps)} of video) with a small detector.</p>
      )}
      {running && <ProgressBar label={job.progress.stage ?? job.status} done={job.progress.done} total={job.progress.total}
        extra={job.progress.fps ? `· ${job.progress.fps.toFixed(1)} samples/s` : ""} />}
      {job.end?.status === "error" && <div className="banner error">{job.end.error}</div>}
      {err && <div className="banner error">{err}</div>}
      {sc && (
        <>
          <Timeline sc={sc} suggested={data!.suggested} parts={parts} />
          <div className="row small muted">
            <span>0:00</span><span className="grow" />
            <span>{(sc.moving ?? sc.counts).filter((c) => c > 0).length} of {sc.counts.length} samples with moving vehicles
              {sc.moving ? <> (bright; dim: all vehicles, parked too)</> : null} ·
              read {sc.mode === "keyframes" ? "keyframes" : "1 frame/s"} in {sc.elapsed_s.toFixed(0)} s</span>
            <span className="grow" /><span>{clock(total / fps)}</span>
          </div>
          <label className="item-row">
            <input type="checkbox" checked={!!parts} disabled={disabled || !data!.suggested.length}
              onChange={(e) => setParts(e.target.checked ? data!.suggested : null)} />
            <span>Analyse only the parts with vehicles: <b>{data!.suggested.length}</b> part{data!.suggested.length === 1 ? "" : "s"},
              {" "}{clock(covered(data!.suggested) / fps)} of {clock(total / fps)} ({Math.round((100 * covered(data!.suggested)) / total)} %)</span>
          </label>
          <div className="item-row small">
            <span className="muted" title="Added before and after each part with vehicles, so vehicles entering and leaving are tracked whole">Padding</span>
            <input key={`p${s.pad_s}`} type="number" step="0.5" min={0} className="narrow" defaultValue={s.pad_s ?? 2}
              onBlur={(e) => setScan({ pad_s: Math.max(0, parseFloat(e.target.value) || 0) })} />
            <span className="muted">s · join parts closer than</span>
            <input key={`m${s.merge_s}`} type="number" step="1" min={0} className="narrow" defaultValue={s.merge_s ?? 5}
              onBlur={(e) => setScan({ merge_s: Math.max(0, parseFloat(e.target.value) || 0) })} />
            <span className="muted">s</span>
          </div>
          {parts && (
            <p className="muted small">Only these parts are decoded and analysed, and by default rendered. Tracking starts fresh in each
              part; frame numbers and times stay those of the whole video.{project.config.stabilization.enabled
                ? " Stabilization re-finds the reference frame after each jump." : ""}</p>
          )}
        </>
      )}
    </div>
  );
}

function Timeline({ sc, suggested, parts }: { sc: NonNullable<ScanResult["scan"]>; suggested: [number, number][]; parts: [number, number][] | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const x = (f: number) => (f / sc.nb_frames) * W;
    g.fillStyle = "rgba(255,255,255,.04)"; g.fillRect(0, 0, W, H);
    (parts ?? suggested).forEach(([a, b]) => {
      g.fillStyle = parts ? "rgba(57,135,229,.30)" : "rgba(57,135,229,.14)";
      g.fillRect(x(a), 0, Math.max(1, x(b) - x(a)), H);
    });
    const peak = Math.max(1, ...sc.counts);
    const bw = Math.max(1, W / Math.max(1, sc.frames.length));
    const bar = (f: number, n: number, color: string) => {
      if (!n) return;
      const h = 3 + (H - 6) * Math.min(1, n / peak);
      g.fillStyle = color;
      g.fillRect(x(f) - bw / 2, H - h, Math.max(1, bw * 0.8), h);
    };
    sc.frames.forEach((f, k) => bar(f, sc.counts[k], "rgba(233,238,245,.22)"));  // all vehicles, parked too
    sc.frames.forEach((f, k) => bar(f, (sc.moving ?? sc.counts)[k], "#e9eef5"));  // moving ones
  });
  return <canvas ref={ref} className="activity-timeline" title="Vehicles per sample over the whole video (bright: moving, dim: all, parked too); blue: the parts to analyse" />;
}
