import { useEffect, useMemo, useState } from "react";
import "../components/speed-profile.css";
import { api, fmt, margin, marginBreakdown, Project, Results, Stats, useJob, useVideoFrame, Vehicle } from "../api";
import MarginPanel from "../components/MarginPanel";
import ValidationPanel from "../components/ValidationPanel";
import Histogram from "../components/Histogram";
import ProgressBar from "../components/ProgressBar";
import TrackOverlay, { OverlayData } from "../components/TrackOverlay";
import SpeedProfile from "../components/SpeedProfile";

interface Props { project: Project; onChanged: (p: Project) => void }

type SortKey = keyof Vehicle;
const COLS: { key: SortKey; label: string; d?: number; title?: string }[] = [
  { key: "track_id", label: "#" },
  { key: "class", label: "Class" },
  { key: "direction", label: "Dir" },
  { key: "t_entry_s", label: "Entry s", d: 2 },
  { key: "dt_s", label: "Δt s", d: 3 },
  { key: "path_length_m", label: "Path m", d: 2 },
  { key: "speed_max_kmh", label: "Max", d: 1, title: "Highest momentary speed between the lines (0.5 s window), ± its 95 % margin of error (hover a value for the breakdown)" },
  { key: "speed_kmh", label: "Avg", d: 1, title: "Average A→B: arc-length regression, ± its 95 % margin of error" },
  { key: "speed_line_kmh", label: "Lines", d: 1, title: "Line-crossing audit speed" },
  { key: "disagreement_pct", label: "Δ %", d: 2 },
  { key: "fit_residual_m", label: "Resid m", d: 3 },
  { key: "accel_mps2", label: "a m/s²", d: 2 },
];

export default function ResultsScreen({ project, onChanged }: Props) {
  const [res, setRes] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dir, setDir] = useState<"all" | "A->B" | "B->A">("all");
  const [validOnly, setValidOnly] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "t_entry_s", asc: true });
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [source, setSource] = useState<"render" | "proxy">("proxy");
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlay, setOverlay] = useState<OverlayData | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [metric, setMetric] = useState<"max" | "avg">("max");
  const [renderJob, setRenderJob] = useState<string | null>(project.jobs.find((j) => j.kind === "render")?.id ?? null);
  const rs = project.config.render as any;
  const [rset, setRset] = useState({
    mode: rs.mode, output_fps: rs.output_fps ?? "", playback_speed: rs.playback_speed, lut: rs.lut ?? "",
    speed_limit_kmh: rs.speed_limit_kmh,
    panel: rs.panel ?? "right", panel_layout: rs.panel_layout ?? "overlay", panel_sort: rs.panel_sort ?? "latest",
    panel_rows: rs.panel_rows ?? 7, title: rs.title ?? "", subtitle: rs.subtitle ?? "", footer: rs.footer ?? "",
    show_hud: !!rs.show_hud, show_margin: rs.show_margin ?? true,
    scale: rs.scale === 0.5 && project.info.height > 1080 ? +(1080 / project.info.height).toFixed(4) : rs.scale,
  });

  const load = () => {
    api.overlay(project.id).then(setOverlay).catch(() => setOverlay(null));
    return api.results(project.id).then((r) => { setRes(r); setErr(null); }).catch((e) => setErr(String(e)));
  };
  useEffect(() => { load(); }, [project.id]);
  const rjob = useJob(renderJob, () => { load(); api.project(project.id).then(onChanged); });

  const useRender = source === "render" && !!res?.render;
  const fps = res?.capture_fps ?? project.config.capture_fps ?? project.info.container_fps;
  // proxy frames are 1:1 with the source; render time = real time / playback_speed
  const { frame, mediaTime } = useVideoFrame(video, useRender ? (res!.render!.output_fps) : project.info.container_fps);
  const tNow = useRender ? mediaTime * res!.render!.playback_speed : frame / fps;

  const seekTo = (v: Vehicle) => {
    if (!video) return;
    const t = Math.max(0, v.t_entry_s - 0.3);
    video.currentTime = useRender ? t / res!.render!.playback_speed : (t * fps + 0.5) / project.info.container_fps;
    video.pause();
  };

  const rows = useMemo(() => {
    if (!res) return [];
    const r = res.vehicles.filter((v) => (dir === "all" || v.direction === dir) && (!validOnly || v.valid));
    return [...r].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      const c = typeof x === "number" ? (x as number) - (y as number) : String(x).localeCompare(String(y));
      return sort.asc ? c : -c;
    });
  }, [res, dir, validOnly, sort]);

  const useMax = metric === "max" && !!res?.stats_max;
  const stats: Stats | undefined = res
    ? (useMax ? (dir === "all" ? res.stats_max : res.by_direction_max?.[dir]) : (dir === "all" ? res.stats : res.by_direction[dir]))
    : undefined;
  const validSpeeds = res ? res.vehicles.filter((v) => v.valid && (dir === "all" || v.direction === dir))
    .map((v) => (useMax ? v.speed_max_kmh : v.speed_kmh)) : [];
  // profile shown: the clicked vehicle, else the fastest valid one
  const fastest = res?.vehicles.filter((v) => v.valid).sort((a, b) => b.speed_max_kmh - a.speed_max_kmh)[0];
  const shown = res?.vehicles.find((v) => v.track_id === selected) ?? fastest;
  const limit = +rset.speed_limit_kmh || 50;

  const startRender = async () => {
    try {
      const body: Record<string, unknown> = {
        mode: rset.mode, playback_speed: +rset.playback_speed, scale: +rset.scale, speed_limit_kmh: +rset.speed_limit_kmh,
        output_fps: rset.output_fps === "" ? null : +rset.output_fps, lut: rset.lut || null,
        panel: rset.panel, panel_layout: rset.panel_layout, panel_sort: rset.panel_sort, panel_rows: +rset.panel_rows,
        title: rset.title, subtitle: rset.subtitle, footer: rset.footer, show_hud: rset.show_hud,
        show_margin: rset.show_margin,
      };
      setRenderJob((await api.render(project.id, body)).id);
    } catch (e) { setErr(String(e)); }
  };

  const recompute = async () => {
    try { setRes(await api.recompute(project.id)); api.overlay(project.id).then(setOverlay); } catch (e) { setErr(String(e)); }
  };

  if (err && !res) return <div className="page"><div className="banner warn">{err}</div></div>;
  if (!res) return <div className="page muted">Loading…</div>;
  const renderRunning = ["running", "queued", "connecting"].includes(rjob.status);
  const reqN = stats?.required_n?.p85;

  return (
    <div className="results">
      <div className="res-left">
        <div className="video-wrap">
          <video key={useRender ? "r" : "p"} ref={setVideo} controls muted playsInline preload="auto"
            src={useRender ? `${api.fileUrl(project.id, "render.mp4")}?v=${res.render!.file}${rjob.end?.id ?? ""}` : api.proxyUrl(project.id)} />
          {!useRender && showOverlay && overlay && <TrackOverlay data={overlay} video={video} frame={frame} />}
        </div>
        <div className="video-bar small">
          <span className="time-readout">t = <b>{tNow.toFixed(3)}</b> s real</span>
          <span className="grow" />
          {!useRender && <label className="small"><input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> tracking overlay</label>}
          {res.render && (
            <div className="seg-ctl">
              <button className={!useRender ? "active" : ""} onClick={() => setSource("proxy")}>Proxy + overlay</button>
              <button className={useRender ? "active" : ""} onClick={() => setSource("render")}>Rendered MP4</button>
            </div>
          )}
        </div>

        {shown && (
          <section className="panel">
            <div className="panel-head">
              <h3>Speed along the section</h3>
              <span className="tag">#{shown.track_id} {shown.class} · {shown.direction.replace("->", " → ")}</span>
              {selected === null && <span className="muted small">fastest vehicle · click a row to pick another</span>}
            </div>
            <SpeedProfile v={shown} limit={limit} tNow={tNow} />
          </section>
        )}

        <section className="panel">
          <div className="panel-head">
            <h3>Vehicles</h3>
            <span className="tag">{rows.length}{rows.length !== res.vehicles.length ? ` of ${res.vehicles.length}` : ""}</span>
            <span className="grow" />
            <div className="seg-ctl">
              {(["all", "A->B", "B->A"] as const).map((d) => <button key={d} className={dir === d ? "active" : ""} onClick={() => setDir(d)}>{d === "all" ? "All" : d.replace("->", " → ")}</button>)}
            </div>
            <label className="small"><input type="checkbox" checked={validOnly} onChange={(e) => setValidOnly(e.target.checked)} /> valid only</label>
          </div>
          <div className="table-wrap">
            <table className="vehicles">
              <thead>
                <tr>
                  {COLS.map((c) => (
                    <th key={c.key} title={c.title} aria-sort={sort.key === c.key ? (sort.asc ? "ascending" : "descending") : undefined}
                      className={`${c.d !== undefined || c.key === "track_id" ? "num" : ""}${sort.key === c.key ? " sorted" : ""}`}
                      onClick={() => setSort({ key: c.key, asc: sort.key === c.key ? !sort.asc : true })}>
                      {c.label}{sort.key === c.key && <span className="sort">{sort.asc ? "▲" : "▼"}</span>}
                    </th>
                  ))}
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((v) => {
                  const active = tNow >= v.t_entry_s - 0.5 && tNow <= v.t_exit_s + 0.5;
                  return (
                    <tr key={v.track_id} className={`${active ? "active" : ""} ${v.valid ? "" : "invalid"} ${shown?.track_id === v.track_id ? "selected" : ""}`}
                      onClick={() => { setSelected(v.track_id); seekTo(v); }}>
                      {COLS.map((c) => {
                        const x = v[c.key];
                        const mk = c.key === "speed_max_kmh" ? "max" : c.key === "speed_kmh" ? "avg" : null;
                        const pm = mk && margin(v, mk);
                        return <td key={c.key} title={mk ? marginBreakdown(v, mk) : undefined}
                          className={c.key === "speed_max_kmh" ? `num strong${v.valid && (x as number) > limit ? " bad-text" : ""}` : typeof x === "number" ? "num" : ""}>
                          {typeof x === "number" && c.d !== undefined ? x.toFixed(c.d) : String(x).replace("->", "→")}
                          {pm && <span className="pm">{pm}</span>}
                          {mk === "max" && v.u_over_target && <span className="u-flag"> ⚠</span>}
                        </td>;
                      })}
                      <td className="status-cell">{v.valid ? <span className="status ok">✓ valid</span> : <span className="status bad" title={v.flags}>✕ {v.flags}</span>}
                        {v.valid && v.flags && <span className="status warn" title="Flag that does not invalidate">⚠ {v.flags}</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {res.uncertainty && <p className="footnote">± = 95 % margin of error per vehicle (calibration and noise; "+a −b" also allows the uncorrected height-parallax bias).
              {" "}⚠ = margin over the {res.uncertainty.target_pct} % target. Hover a speed for its breakdown.</p>}
            {rows.length === 0 && (
              <div className="banner warn small">
                No vehicles {res.vehicles.length ? "match the filter" : "crossed both lines"}.
                {res.vehicles.length === 0 && res.run.tracks !== undefined && <> Detector: {res.run.detections_per_frame?.toFixed(1)} detections/frame,
                  {" "}{res.run.tracks} tracks ({res.run.tracks_long_enough} long enough): {res.run.tracks_crossing_a_only} crossed only A,
                  {" "}{res.run.tracks_crossing_b_only} only B, {res.run.tracks_crossing_neither} neither.
                  {res.run.detections_per_frame < 1 ? " Very few detections: try another model or tiling (Process → Preview tracking)." :
                    " Check that lines A and B span every lane the vehicles use (watch the overlay)."}</>}
              </div>
            )}
          </div>
        </section>

        <ValidationPanel projectId={project.id} res={res} selected={res.vehicles.find((v) => v.track_id === selected)} tNow={tNow} />
      </div>

      <aside className="res-side">
        <section className="panel">
          <div className="panel-head">
            <h3>Speed statistics</h3>
            {dir !== "all" && <span className="tag">{dir.replace("->", " → ")}</span>}
            <span className="grow" />
            {res.stats_max && (
              <div className="seg-ctl" title="Statistics of each vehicle's max momentary speed, or of its A→B average">
                <button className={useMax ? "active" : ""} onClick={() => setMetric("max")}>Max</button>
                <button className={!useMax ? "active" : ""} onClick={() => setMetric("avg")}>Average</button>
              </div>
            )}
          </div>
          {res.calibration_check?.status === "suspect" && (
            <div className="banner warn small calib-check"><b>Check the calibration.</b> {res.calibration_check.message}</div>
          )}
          <div className="tiles">
            <Tile label="Valid / measured" value={`${stats?.n_valid ?? 0} / ${dir === "all" ? res.stats.n_total : res.vehicles.filter((v) => v.direction === dir).length}`} />
            <Tile label={`Required n (p85, ±${stats?.permitted_error_kmh} km/h)`} value={reqN !== undefined ? String(reqN) : "–"}
              note={reqN !== undefined ? ((stats?.n_valid ?? 0) >= reqN ? "✓ sample sufficient" : `✕ need ${reqN - (stats?.n_valid ?? 0)} more`) : "needs ≥2 vehicles"}
              tone={reqN !== undefined ? ((stats?.n_valid ?? 0) >= reqN ? "ok" : "bad") : undefined} />
            <Tile label="Mean" unit="km/h" value={fmt(stats?.mean)} note={stats?.mean_ci ? `CI ${fmt(stats.mean_ci[0])}–${fmt(stats.mean_ci[1])}` : ""} />
            <Tile label="Standard deviation σ" unit="km/h" value={fmt(stats?.sd, 2)} />
            <Tile label="85th percentile" unit="km/h" value={fmt(stats?.p85)} note={stats?.p85_ci ? `CI ${fmt(stats.p85_ci[0])}–${fmt(stats.p85_ci[1])}` : ""} />
            <Tile label="95th percentile" unit="km/h" value={fmt(stats?.p95)} note={stats?.p95_ci ? `CI ${fmt(stats.p95_ci[0])}–${fmt(stats.p95_ci[1])}` : ""} />
          </div>
          <p className="footnote">{Math.round((stats?.confidence ?? 0.95) * 100)} % intervals: t-interval for the mean, bootstrap for percentiles. Required n uses the ITE spot-speed formula with the σ measured so far.</p>
          <Histogram values={validSpeeds} mean={stats?.mean} p85={stats?.p85} />
          <p className="footnote">{useMax ? "Per vehicle: highest momentary speed between the lines." : "Per vehicle: average speed from line A to line B."}</p>
        </section>

        <MarginPanel project={project} res={res} onProject={onChanged}
          onRecomputed={(r) => { setRes(r); api.overlay(project.id).then(setOverlay); }} />

        <section className="panel">
          <h3>Export</h3>
          <div className="export-grid">
            <a className="button" href={api.fileUrl(project.id, "results.csv")}><span className="dl-ico">↓</span>results.csv</a>
            <a className="button" href={api.fileUrl(project.id, "results.json")}><span className="dl-ico">↓</span>results.json</a>
            {res.render && <a className="button" href={api.fileUrl(project.id, "render.mp4")}><span className="dl-ico">↓</span>render.mp4</a>}
          </div>
          <div className="row">
            <button onClick={recompute} title="Re-run speed estimation from stored tracks (after editing lines, calibration or thresholds)">Recompute speeds</button>
          </div>
        </section>

        <section className="panel">
          <h3>Render annotated video</h3>
          <div className="form-grid compact">
            <label>Mode</label>
            <select value={rset.mode} onChange={(e) => setRset({ ...rset, mode: e.target.value })}>
              <option value="stabilized">stabilized (lines fixed)</option><option value="original">original frames</option>
            </select>
            <label>Output fps</label>
            <select value={String(rset.output_fps)} onChange={(e) => setRset({ ...rset, output_fps: e.target.value })}>
              <option value="">native ({fps} fps)</option><option value="60">60</option><option value="30">30</option>
            </select>
            <label>Playback speed</label>
            <select value={String(rset.playback_speed)} onChange={(e) => setRset({ ...rset, playback_speed: +e.target.value })}>
              <option value="1">1× real time</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option>
            </select>
            <label>Scale</label>
            <input type="number" step="any" value={rset.scale} onChange={(e) => setRset({ ...rset, scale: e.target.value })} />
            <label>Speed limit (colour)</label>
            <input type="number" value={rset.speed_limit_kmh} onChange={(e) => setRset({ ...rset, speed_limit_kmh: e.target.value })} />
            <label title="D-Log M → Rec.709 .cube, applied before drawing">LUT (.cube path)</label>
            <input type="text" value={rset.lut} placeholder="optional" onChange={(e) => setRset({ ...rset, lut: e.target.value })} />
          </div>
          <h4 className="subhead">Presentation</h4>
          <div className="form-grid compact">
            <label title="Lists every vehicle's max speed as it crosses the second line, with totals">Vehicle panel</label>
            <select value={rset.panel} onChange={(e) => setRset({ ...rset, panel: e.target.value })}>
              <option value="right">right side</option><option value="left">left side</option><option value="none">off</option>
            </select>
            {rset.panel !== "none" && <>
              <label title="Overlay keeps the video's 16:9 frame; extend adds the panel beside it (wider video)">Panel layout</label>
              <select value={rset.panel_layout} onChange={(e) => setRset({ ...rset, panel_layout: e.target.value })}>
                <option value="overlay">over the video (16:9)</option><option value="extend">beside the video (wider)</option>
              </select>
              <label>List order</label>
              <select value={rset.panel_sort} onChange={(e) => setRset({ ...rset, panel_sort: e.target.value })}>
                <option value="latest">latest first</option><option value="fastest">fastest first</option>
              </select>
              <label>Rows</label>
              <input type="number" min={1} max={20} value={rset.panel_rows} onChange={(e) => setRset({ ...rset, panel_rows: e.target.value })} />
              <label title="Each row shows the max speed's 95 % margin of error">Margin of error</label>
              <input type="checkbox" checked={rset.show_margin} onChange={(e) => setRset({ ...rset, show_margin: e.target.checked })} />
              <label>Title</label>
              <input type="text" value={rset.title} placeholder="e.g. street, city" onChange={(e) => setRset({ ...rset, title: e.target.value })} />
              <label>Subtitle</label>
              <input type="text" value={rset.subtitle} placeholder="e.g. date · time" onChange={(e) => setRset({ ...rset, subtitle: e.target.value })} />
              <label>Footer</label>
              <input type="text" value={rset.footer} placeholder="e.g. source, handle" onChange={(e) => setRset({ ...rset, footer: e.target.value })} />
            </>}
            <label title="Small time / frame readout in the corner">Technical readout</label>
            <input type="checkbox" checked={rset.show_hud} onChange={(e) => setRset({ ...rset, show_hud: e.target.checked })} />
          </div>
          <div className="row end">
            {renderRunning ? <button className="danger" onClick={() => renderJob && api.cancel(renderJob)}>Cancel render</button>
              : <button className="primary" onClick={startRender}>{res.render ? "Re-render" : "Render"}</button>}
          </div>
          {renderJob && rjob.status !== "done" && (
            <ProgressBar label={rjob.progress.stage ?? rjob.status} done={rjob.progress.done} total={rjob.progress.total}
              extra={rjob.progress.fps ? `· ${rjob.progress.fps.toFixed(1)} fps` : ""} />
          )}
          {rjob.end?.status === "error" && <div className="banner error">{rjob.end.error}</div>}
        </section>
        {res.timing_warnings.map((w) => <div key={w} className="banner info small">{w}</div>)}
        {err && <div className="banner error">{err}</div>}
      </aside>
    </div>
  );
}

function Tile({ label, value, unit, note, tone }: { label: string; value: string; unit?: string; note?: string; tone?: "ok" | "bad" }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}{unit && value !== "–" && <span className="tile-unit">{unit}</span>}</div>
      {note && <div className={`tile-note ${tone ? `${tone}-text` : ""}`}>{note}</div>}
    </div>
  );
}
