import { useEffect, useMemo, useState } from "react";
import Timeline from "../components/Timeline";
import Help from "../components/Help";
import "../components/speed-profile.css";
import { api, fmt, margin, marginBreakdown, MergeCandidates, Project, Results, Stats, useVideoFrame, Vehicle } from "../api";
import ValidationPanel from "../components/ValidationPanel";
import Histogram from "../components/Histogram";
import TrackOverlay, { OverlayData } from "../components/TrackOverlay";
import SpeedProfile, { vehicleEvents } from "../components/SpeedProfile";
import ViolationsPanel from "../components/ViolationsPanel";
import RotatedVideo, { VideoControls } from "../components/RotatedVideo";
import { SortBy, SplitsPanel, VehicleCard, VehicleList } from "../components/Vehicles";

interface Props { project: Project; onChanged: (p: Project) => void; onExport: () => void; onSetup: () => void }

type SortKey = keyof Vehicle;
const COLS: { key: SortKey; label: string; d?: number; title?: string }[] = [
  { key: "track_id", label: "#" },
  { key: "name", label: "Name" },
  { key: "class", label: "Class" },
  { key: "path", label: "Path" },
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

/** Review, as a dashboard: the clip with the vehicle list beside it, the selected vehicle's speed
 * along the section and its card below, then the statistics, violations and margins. Rendering
 * and downloads are on the Export screen. */
type Tab = "speeds" | "violations" | "splits" | "table" | "validation";

export default function ReviewScreen({ project, onChanged, onExport, onSetup }: Props) {
  const [res, setRes] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dir, setDir] = useState<"all" | "A->B" | "B->A">("all");
  const [road, setRoad] = useState<string>("all");
  const [validOnly, setValidOnly] = useState(false);
  const [sortBy, setSortBy] = useState<SortBy>("time");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "t_entry_s", asc: true });
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [wrapEl, setWrapEl] = useState<HTMLDivElement | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlay, setOverlay] = useState<OverlayData | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [metric, setMetric] = useState<"max" | "avg">("max");
  const [cands, setCands] = useState<MergeCandidates | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>(() => { try { return (localStorage.getItem("speedcam.review.tab") as Tab) || "speeds"; } catch { return "speeds"; } });
  const pickTab = (k: Tab) => { setTab(k); try { localStorage.setItem("speedcam.review.tab", k); } catch { /* private mode */ } };

  const extras = () => {
    api.overlay(project.id).then(setOverlay).catch(() => setOverlay(null));
    api.mergeCandidates(project.id).then(setCands).catch(() => setCands(null));
  };
  const load = () => {
    extras();
    return api.results(project.id).then((r) => { setRes(r); setErr(null); }).catch((e) => setErr(String(e)));
  };
  useEffect(() => { load(); }, [project.id]);  // eslint-disable-line react-hooks/exhaustive-deps
  // new tracks or merges: the suggestions are stale until they're found again
  const gotResults = (r: Results) => { setRes(r); setCands(null); extras(); api.project(project.id).then(onChanged).catch(() => {}); };

  const rotation = project.config.rotation ?? 0;
  const fps = res?.capture_fps ?? project.config.capture_fps ?? project.info.container_fps;
  const { frame } = useVideoFrame(video, project.info.container_fps);
  const tNow = frame / fps;
  const seekTime = (t: number) => {
    if (!video) return;
    video.currentTime = (Math.max(0, t) * fps + 0.5) / project.info.container_fps;
    video.pause();
  };
  const pick = (v: Vehicle) => { setSelected(v.track_id); seekTime(v.t_entry_s - 0.3); };

  const rows = useMemo(() => {
    if (!res) return [];
    const r = res.vehicles.filter((v) => (dir === "all" || v.direction === dir) && (!validOnly || v.valid) &&
      (road === "all" || v.path === road));
    const by = sortBy === "max" ? (a: Vehicle, b: Vehicle) => b.speed_max_kmh - a.speed_max_kmh
      : sortBy === "name" ? (a: Vehicle, b: Vehicle) => (a.name || "~").localeCompare(b.name || "~") || a.t_entry_s - b.t_entry_s
        : (a: Vehicle, b: Vehicle) => a.t_entry_s - b.t_entry_s;
    return [...r].sort(by);
  }, [res, dir, validOnly, road, sortBy]);
  const tableRows = useMemo(() => [...rows].sort((a, b) => {
    const x = a[sort.key], y = b[sort.key];
    const c = typeof x === "number" ? (x as number) - (y as number) : String(x ?? "").localeCompare(String(y ?? ""));
    return sort.asc ? c : -c;
  }), [rows, sort]);

  const useMax = metric === "max" && !!res?.stats_max;
  const roads = Object.keys(res?.by_path ?? {});
  const cols = COLS.filter((c) => (c.key !== "path" || roads.length > 0) && (c.key !== "name" || res?.vehicles.some((v) => v.name)));
  const stats: Stats | undefined = res
    ? (road !== "all" && res.by_path?.[road]
      ? (useMax ? res.by_path[road].max : res.by_path[road].avg)
      : useMax ? (dir === "all" ? res.stats_max : res.by_direction_max?.[dir]) : (dir === "all" ? res.stats : res.by_direction[dir]))
    : undefined;
  const validSpeeds = res ? res.vehicles.filter((v) => v.valid && (useMax || v.avg_valid !== false) && (road !== "all" ? v.path === road : dir === "all" || v.direction === dir))
    .map((v) => (useMax ? v.speed_max_kmh : v.speed_kmh)) : [];
  // shown: the picked vehicle, else the fastest valid one
  const fastest = res?.vehicles.filter((v) => v.valid).sort((a, b) => b.speed_max_kmh - a.speed_max_kmh)[0];
  const shown = res?.vehicles.find((v) => v.track_id === selected) ?? fastest ?? res?.vehicles[0];
  const limit = +(project.config.render as any).speed_limit_kmh || 50;

  const recompute = async () => {
    setBusy("Recomputing speeds…");
    try { gotResults(await api.recompute(project.id)); } catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const merge = async (ids: number[]) => {
    setBusy("Merging and measuring again…");
    try {
      const r = await api.mergeVehicles(project.id, ids);
      gotResults(r);
      const kept = r.vehicles.find((v) => ids.some((i) => v.merged?.includes(i)));
      if (kept) setSelected(kept.track_id);
    } catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const split = async (tid: number) => {
    setBusy("Splitting and measuring again…");
    try { gotResults(await api.splitVehicle(project.id, tid)); } catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const named = async (tid: number, name: string) => {
    try {
      await api.nameVehicle(project.id, tid, name);
      setRes((r) => r && { ...r, vehicles: r.vehicles.map((v) => (v.track_id === tid ? { ...v, name } : v)) });
      setCands((c) => c && { ...c, tracks: { ...c.tracks, ...(c.tracks[String(tid)] ? { [String(tid)]: { ...c.tracks[String(tid)], name } } : {}) } });
    } catch (e) { setErr(String(e)); }
  };

  if (err && !res) return <div className="page"><div className="banner warn">{err}</div></div>;
  if (!res) return <div className="page muted">Loading…</div>;
  const reqN = stats?.required_n?.p85;
  const nValid = res.vehicles.filter((v) => v.valid).length;
  const nOver = res.vehicles.filter((v) => v.valid && v.speed_max_kmh > limit).length;
  const nViol = (res.violations?.n_red_light ?? 0) + (res.violations?.n_solid_line ?? 0);
  const nSplits = cands?.pairs.filter((p) => p.likely).length;
  const tabs: { key: Tab; label: string; count?: number; bad?: boolean }[] = [
    { key: "speeds", label: "Speeds" },
    ...(res.violations ? [{ key: "violations" as Tab, label: "Violations", count: nViol, bad: nViol > 0 }] : []),
    { key: "splits", label: "Possibly split", count: nSplits },
    { key: "table", label: "All measurements", count: res.vehicles.length },
    { key: "validation", label: "Validation run" },
  ];
  if (!tabs.some((x) => x.key === tab)) setTimeout(() => setTab("speeds"), 0);
  const noPathOrLines = !(project.config.calibration.paths ?? []).some((p) => p.points.length >= 2) && !(project.config.calibration.line_a && project.config.calibration.line_b);

  const empty = (
    <div className="banner warn small">
      {res.vehicles.length === 0 && res.calibration?.method === "none" ? (
        <>Tracking only: {res.run.tracks ?? 0} vehicles were tracked, but there's no usable calibration, so nothing was measured. Calibrate and draw a path or lines in Setup, then press Recompute.</>
      ) : res.vehicles.length === 0 && noPathOrLines ? (
        <>No speeds measured: {res.run.tracks ?? 0} vehicles were tracked, but there's no path or lines A/B to measure them on. Draw them in Setup, then press Recompute.</>
      ) : <>No vehicles {res.vehicles.length ? "match the filter" : "crossed both lines"}.</>}
      {res.vehicles.length === 0 && res.calibration?.method !== "none" && !noPathOrLines && res.run.tracks !== undefined && <> Detector: {res.run.detections_per_frame?.toFixed(1)} detections/frame,
        {" "}{res.run.tracks} tracks ({res.run.tracks_long_enough} long enough): {res.run.tracks_crossing_a_only} crossed only A,
        {" "}{res.run.tracks_crossing_b_only} only B, {res.run.tracks_crossing_neither} neither.
        {res.run.detections_per_frame < 1 ? " Very few detections: try another model or tiling (Analyze → Preview tracking)." :
          " Check that lines A and B span every lane the vehicles use (watch the overlay)."}</>}
    </div>
  );

  return (
    <div className="dash">
      <div className="dash-head">
        <h2>Review</h2>
        <span className="tag">{nValid} valid of {res.vehicles.length}</span>
        {res.vehicles.length > 0 && <span className={`tag ${nOver ? "bad" : ""}`}>{nOver} over {limit} km/h</span>}
        {res.stats_max?.p85 !== undefined && <span className="tag">p85 {fmt(res.stats_max.p85)} km/h</span>}
        {res.violations && <span className={`tag ${nViol ? "bad" : ""}`}>{nViol} violation{nViol === 1 ? "" : "s"}</span>}
        <span className="grow" />
        <button onClick={recompute} disabled={!!busy} title="Re-run speeds and violation checks from stored tracks (after editing lines, lights, calibration or thresholds). Lights added or moved since the analysis are read from the video, which takes a minute.">Recompute</button>
        <button className="primary" onClick={onExport} title="Render the annotated video, download the data">Export →</button>
      </div>
      {res.stale?.recompute && (
        <div className="banner info small row">
          <span className="grow">The setup changed since these results were computed (scale, paths, limits, rules or tolerances).</span>
          <button className="primary" disabled={!!busy} onClick={recompute}>Recompute</button>
        </div>
      )}
      {res.stale?.analysis && (
        <div className="banner info small">Analysis settings (detector, camera, the parts analysed) changed since this analysis: they apply when you analyse again.</div>
      )}
      {res.calibration_check?.status === "suspect" && (
        <div className="banner warn small calib-check"><b>Check the calibration.</b> {res.calibration_check.message}</div>
      )}
      {res.timing_warnings.map((w) => <div key={w} className="banner info small">{w}</div>)}
      {err && <div className="banner error">{err}</div>}

      <div className="dash-top">
        <div className="dash-player">
          <div ref={setWrapEl} className={`video-wrap${rotation ? " rotated" : ""}`}>
            <RotatedVideo rotation={rotation} videoRef={setVideo} muted playsInline preload="auto" src={api.proxyUrl(project.id)}
              onClick={(e) => { const v = e.currentTarget; v.paused ? v.play() : v.pause(); }} />
            {showOverlay && overlay && <TrackOverlay data={overlay} video={video} frame={frame} rotation={rotation} />}
          </div>
          <VideoControls video={video} fullscreenEl={wrapEl} containerFps={project.info.container_fps}
            realTime={(t) => (t * project.info.container_fps) / fps} />
          <Timeline project={project} frame={frame} editable={false}
            onSeek={(f) => { if (video) { video.pause(); video.currentTime = (f + 0.5) / project.info.container_fps; } }}
            vehicles={res.vehicles.map((v) => ({ id: v.track_id, f0: Math.round(v.t_entry_s * fps), f1: Math.round(v.t_exit_s * fps), valid: v.valid }))}
            selectedVehicle={shown?.track_id ?? null} onPickVehicle={setSelected} />
          <div className="video-bar small">
            <label className="small"><input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> tracking overlay</label>
          </div>
        </div>
        <VehicleList projectId={project.id} res={res} rows={rows} selected={shown?.track_id ?? null} tNow={tNow} limit={limit} onPick={pick}
          dir={dir} setDir={setDir} road={road} setRoad={setRoad} roads={roads} validOnly={validOnly} setValidOnly={setValidOnly}
          sortBy={sortBy} setSortBy={setSortBy} empty={empty} />
      </div>

      {shown && (
        <div className="dash-mid">
          <section className="panel">
            <div className="panel-head">
              <h3>Speed along the section<Help topic="results.profile" /></h3>
              <span className="tag">{shown.name || `#${shown.track_id} ${shown.class}`} · {shown.direction.replace("->", " → ")}</span>
              {selected === null && <span className="muted small">fastest vehicle · pick another in the list or on the timeline</span>}
            </div>
            <SpeedProfile v={shown} limit={limit} tNow={tNow} events={vehicleEvents(res, shown.track_id)} onSeek={seekTime} />
          </section>
          <VehicleCard projectId={project.id} res={res} v={shown} cands={cands} busy={busy}
            onNamed={named} onMerge={merge} onSplit={split} onSeek={seekTime} />
        </div>
      )}

      <section className="panel review-tabs">
        <div className="rt-bar" role="tablist" aria-label="More about the results">
          {tabs.map((tb) => (
            <button key={tb.key} role="tab" aria-selected={tab === tb.key} className={`rt-tab${tab === tb.key ? " active" : ""}`} onClick={() => pickTab(tb.key)}>
              {tb.label}{tb.count !== undefined && <span className={`tag${tb.bad ? " bad" : ""}`}>{tb.count}</span>}
            </button>
          ))}
          <span className="grow" />
          {tab === "speeds" && res.stats_max && (
            <div className="seg-ctl" title="Statistics of each vehicle's max momentary speed, or of its A→B average">
              <button className={useMax ? "active" : ""} onClick={() => setMetric("max")}>Max</button>
              <button className={!useMax ? "active" : ""} onClick={() => setMetric("avg")}>Average</button>
            </div>
          )}
        </div>

        {tab === "speeds" && (
          <div className="rt-speeds">
            <div>
              <div className="stat-row">
                <Stat label="Valid / measured" value={`${stats?.n_valid ?? 0} / ${road !== "all" ? res.vehicles.filter((v) => v.path === road).length
                  : dir === "all" ? res.stats.n_total : res.vehicles.filter((v) => v.direction === dir).length}`} />
                <Stat label="Mean" unit="km/h" value={fmt(stats?.mean)} note={stats?.mean_ci ? `${fmt(stats.mean_ci[0])}–${fmt(stats.mean_ci[1])}` : ""} />
                <Stat label="85th percentile" unit="km/h" value={fmt(stats?.p85)} note={stats?.p85_ci ? `${fmt(stats.p85_ci[0])}–${fmt(stats.p85_ci[1])}` : ""} />
                <Stat label="95th percentile" unit="km/h" value={fmt(stats?.p95)} note={stats?.p95_ci ? `${fmt(stats.p95_ci[0])}–${fmt(stats.p95_ci[1])}` : ""} />
                <Stat label="Standard deviation" unit="km/h" value={fmt(stats?.sd, 2)} />
                <Stat label={`Needed for p85 ±${stats?.permitted_error_kmh}`} value={reqN !== undefined ? String(reqN) : "–"}
                  note={reqN !== undefined ? ((stats?.n_valid ?? 0) >= reqN ? "✓ enough" : `${reqN - (stats?.n_valid ?? 0)} more`) : "needs ≥2"}
                  tone={reqN !== undefined ? ((stats?.n_valid ?? 0) >= reqN ? "ok" : "bad") : undefined} />
              </div>
              <p className="footnote">{road !== "all" ? `${road} · ` : dir !== "all" ? `${dir.replace("->", " → ")} · ` : ""}
                {useMax ? "Per vehicle: its highest momentary speed, from cleanly tracked frames." : "Per vehicle: its average A→B, where conclusive."}
                {" "}{Math.round((stats?.confidence ?? 0.95) * 100)} % intervals (t for the mean, bootstrap for percentiles); "needed" is the ITE spot-speed sample size.</p>
              <MarginSummary res={res} onSetup={onSetup} />
            </div>
            <Histogram values={validSpeeds} mean={stats?.mean} p85={stats?.p85} />
          </div>
        )}

        {tab === "violations" && res.violations && (
          <ViolationsPanel res={res} tNow={tNow} selected={selected} onSeek={seekTime}
            onPick={(tid, t) => { setSelected(tid); seekTime(t - 1.5); }}
            hasSolid={(project.config.calibration.solid_lines ?? []).length > 0}
            solidMode={(project.config.violations as any)?.solid_mode ?? "wheels"}
            minDuration={(project.config.violations as any)?.min_duration_s ?? 0.5} />
        )}

        {tab === "splits" && (!cands ? <p className="muted small">Looking for vehicles the tracker split in two…</p>
          : nSplits ? <SplitsPanel cands={cands} busy={busy} onMerge={merge} onSeek={seekTime} />
            : <p className="muted small">No vehicle looks split in two: no track starts where another one ended, heading the same way at the same speed.
              Pick a vehicle to merge one by its number.</p>)}

        {tab === "table" && (
          <div className="table-wrap">
            <table className="vehicles">
              <thead>
                <tr>
                  {cols.map((c) => (
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
                {tableRows.map((v) => {
                  const active = tNow >= v.t_entry_s - 0.5 && tNow <= v.t_exit_s + 0.5;
                  return (
                    <tr key={v.track_id} className={`${active ? "active" : ""} ${v.valid ? "" : "invalid"} ${shown?.track_id === v.track_id ? "selected" : ""}`}
                      onClick={() => pick(v)}>
                      {cols.map((c) => {
                        const x = v[c.key];
                        const mk = c.key === "speed_max_kmh" ? "max" : c.key === "speed_kmh" ? "avg" : null;
                        const inconclusive = c.key === "speed_kmh" && v.avg_valid === false;
                        const pm = mk && !inconclusive && margin(v, mk);
                        if (inconclusive) return <td key={c.key} className="num muted" title={`Average inconclusive (${v.flags}): a detection gap or position jump in the stretch. The max only uses cleanly tracked frames.`}>–</td>;
                        return <td key={c.key} title={mk ? marginBreakdown(v, mk) : undefined}
                          className={c.key === "speed_max_kmh" ? `num strong${v.valid && (x as number) > limit ? " bad-text" : ""}` : typeof x === "number" ? "num" : ""}>
                          {typeof x === "number" && c.d !== undefined ? x.toFixed(c.d) : String(x ?? "").replace("->", "→")}
                          {pm && <span className="pm">({pm})</span>}
                        </td>;
                      })}
                      <td className="status-cell">{v.valid ? <span className="status ok">✓ valid</span> : <span className="status bad" title={v.flags}>✕ {v.flags}</span>}
                        {v.valid && v.avg_valid === false && <span className="status warn" title={`${v.flags}: the average over the stretch is inconclusive; the max comes from the ${Math.round(v.max_clean_pct ?? 0)} % of it tracked cleanly`}>avg inconclusive</span>}
                        {v.valid && v.avg_valid !== false && v.flags && <span className="status warn" title="Flag that does not invalidate">⚠ {v.flags}</span>}
                        {(v.merged?.length ?? 0) > 1 && <span className="status" title="Measured as one vehicle from these tracks">⧉ {v.merged!.join("+")}</span>}
                        {v.violations?.map((x) => <span key={x} className="status bad" title="Violation">⛔ {x}</span>)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {res.uncertainty && <p className="footnote">Every column, sortable, with the list's filter. (± %) = 95 % margin of error per vehicle (calibration and noise; "+a −b" also allows the uncorrected height-parallax bias). Hover a speed for its breakdown.</p>}
          </div>
        )}

        {tab === "validation" && (
          <ValidationPanel projectId={project.id} res={res} selected={res.vehicles.find((v) => v.track_id === selected)} tNow={tNow} />
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, unit, note, tone }: { label: string; value: string; unit?: string; note?: string; tone?: "ok" | "bad" }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}{unit && value !== "–" && <span className="stat-unit">{unit}</span>}</div>
      {note && <div className={`stat-note ${tone ? `${tone}-text` : ""}`}>{note}</div>}
    </div>
  );
}

/** What the speeds' margins of error cover; the inputs are in Setup. */
function MarginSummary({ res, onSetup }: { res: Results; onSetup: () => void }) {
  const u = res.uncertainty;
  if (!u) return <div className="banner info small">These results predate the margin of error: press Recompute to add it.</div>;
  return (
    <div className="margin-sum small">
      <span title="Median 95 % margin of error of the valid vehicles' max speeds">Margin of error <b>±{fmt(u.median_u_pct, 1)} %</b> typical</span>
      <span title="How well the redundant check distances agree, relative to the declared tolerances">
        {u.fit_ratio === null ? <span className="warn-text">no check distance</span> : <>check fit {fmt(u.fit_ratio, 2)}{u.fit_inflation > 1 ? " (widened)" : ""}</>}</span>
      <span className={u.parallax === "bounded" ? "" : "warn-text"} title="Vehicle tops are closer to the camera than the road, so speeds read slightly high. Bounded once the drone height is known.">
        height parallax {u.parallax === "bounded" ? "bounded" : "not bounded (no drone height)"}</span>
      <button className="link" onClick={onSetup}>tolerances in Setup →</button>
    </div>
  );
}
