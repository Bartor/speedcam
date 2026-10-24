import { useEffect, useRef, useState } from "react";
import Timeline from "../components/Timeline";
import Help, { Advanced } from "../components/Help";
import { SettingNum } from "../components/Settings";
import SolidCases from "../components/SolidCases";
import { api, Calibration, CalibrationResult, fmt, LIGHT_COLORS, MotionPath, PATH_COLORS, PHASE_COLORS, Project, RedLightSetup, Results, SOLID_COLOR, SolidLine, StopDirection, useDebounced, useVideoFrame } from "../api";
import FrameCanvas, { rectDims, Tool } from "../components/FrameCanvas";
import RotatedVideo, { VideoControls } from "../components/RotatedVideo";
import RotateButtons from "../components/RotateButtons";

interface Props { project: Project; onChanged: (p: Project) => void; onNext: () => void }

type ToolGroup = "nav" | "cal" | "lines" | "rules" | "area";
const TOOLS: { key: Tool; label: string; hint: string; group: ToolGroup; color?: string; line?: boolean }[] = [
  { key: "move", group: "nav", label: "Move", hint: "Drag handles to adjust; drag empty space to pan." },
  { key: "polygon", group: "cal", color: "#b98cff", label: "Polygon", hint: "Click 4+ flat ground points (close with Enter, double-click or the first point), then type measured sides and diagonals. Corrects perspective: best for oblique views." },
  { key: "rect", group: "cal", color: "#ff8c3a", label: "Rectangle", hint: "Click 4 ground corners of a known rectangle in order around it (e.g. lane width × marking spacing), then type both edge lengths. Gives a homography: required for oblique views." },
  { key: "segment", group: "cal", color: "#f5c518", line: true, label: "Distance", hint: "Click two ground points, then type the measured distance (m)." },
  { key: "point", group: "cal", color: "#ff8c3a", label: "Control pt", hint: "Click a ground point with known world coordinates (m). 4+ points give a full homography." },
  { key: "path", group: "lines", color: "#00c8ff", line: true, label: "Path", hint: "Click points along a road in the direction of travel (A → B); Enter or double-click to finish. Draw one per road (e.g. at an intersection). Vehicles inside the corridor are measured along the curve; replaces lines A/B." },
  { key: "lineA", group: "lines", color: "#00c8ff", line: true, label: "Line A", hint: "Click two points across the road for measurement line A." },
  { key: "lineB", group: "lines", color: "#ff3cc8", line: true, label: "Line B", hint: "Click two points across the road for measurement line B." },
  { key: "light", group: "rules", color: "#ff5a5a", label: "Light", hint: "Click the middle of a traffic light's lamps; set the circle's radius in the side panel so it covers the lit lamps. Add every light showing the same signal to one setup." },
  { key: "stopline", group: "rules", color: "#ff5a5a", line: true, label: "Stop line", hint: "Click points along the stop line (2 or more), Enter or double-click to finish. Crossing it while the setup's light is red counts as running the red." },
  { key: "solid", group: "rules", color: SOLID_COLOR, line: true, label: "Solid line", hint: "Click points along a solid road marking, Enter or double-click to finish. Any vehicle crossing it is reported. Works best straight down." },
  { key: "roi", group: "area", color: "#4cd97b", label: "ROI", hint: "Click polygon corners; close with Enter, double-click or by clicking the first point." },
];
const GROUPS: { key: ToolGroup; label?: string }[] = [
  { key: "nav" }, { key: "cal", label: "Calibration" }, { key: "lines", label: "Measure" }, { key: "rules", label: "Violations" }, { key: "area", label: "Area" },
];

const GROUP_HELP: Partial<Record<ToolGroup, string>> = {
  cal: "calibrate.calibration", lines: "calibrate.measure", rules: "calibrate.solid", area: "calibrate.area",
};

/** Setup, in the order it's done: each stage has its own tools and side panel; the frame shows everything. */
type Stage = "scale" | "measure" | "limits" | "rules";
const STAGES: { key: Stage; label: string; groups: ToolGroup[]; title: string }[] = [
  { key: "scale", label: "Scale", groups: ["nav", "cal"], title: "How many metres a pixel is: measured distances on the ground, and how well they're known" },
  { key: "measure", label: "Measure", groups: ["nav", "lines", "area"], title: "Where speeds are measured: paths of motion or lines A and B, and the area to watch" },
  { key: "limits", label: "Limits & tolerances", groups: ["nav"], title: "The speed limit, the statistics, and when a measurement counts" },
  { key: "rules", label: "Rules", groups: ["nav", "rules"], title: "Red lights and solid lines (optional)" },
];

export default function SetupScreen({ project, onChanged, onNext }: Props) {
  const info = project.info;
  const [cal, setCal] = useState<Calibration>(project.config.calibration);
  const [result, setResult] = useState<CalibrationResult>(project.calibration_result);
  const [saving, setSaving] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [tool, setTool] = useState<Tool>("move");
  const [stage, setStage] = useState<Stage>(() => {
    const c = project.config.calibration, r = project.calibration_result;
    if (r.error || r.fit_ok === false) return "scale";
    return (c.paths ?? []).some((pp) => pp.points.length >= 2) || (c.line_a && c.line_b) ? "limits" : "measure";
  });
  const stageDef = STAGES.find((s) => s.key === stage)!;
  const goStage = (s: Stage) => {
    setStage(s);
    if (!STAGES.find((x) => x.key === s)!.groups.includes(TOOLS.find((x) => x.key === tool)!.group)) setTool("move");
  };
  const [focusSeg, setFocusSeg] = useState<number | null>(null);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const { frame } = useVideoFrame(video, info.container_fps);
  const [scrub, setScrub] = useState(cal.reference_frame);
  const segInputs = useRef<Record<number, HTMLInputElement | null>>({});
  const rectW = useRef<HTMLInputElement | null>(null);
  const rd = rectDims(cal);
  const polyInputs = useRef<Record<number, HTMLInputElement | null>>({});
  const [diag, setDiag] = useState<[number, number]>([0, 2]);
  const poly = cal.polygon;
  const polyN = poly?.points.length ?? 0;
  const polyMeasured = poly?.distances.filter((d) => d.meters > 0).length ?? 0;
  const setPolyDist = (k: number, meters: number | null) => {
    const c = structuredClone(cal);
    if (meters === null) c.polygon!.distances.splice(k, 1);
    else c.polygon!.distances[k].meters = meters;
    update(c);
  };
  const addDiagonal = () => {
    const [i, j] = diag;
    if (!poly || i === j || poly.distances.some((d) => (d.i === i && d.j === j) || (d.i === j && d.j === i))) return;
    const c = structuredClone(cal);
    c.polygon!.distances.push({ i: Math.min(i, j), j: Math.max(i, j), meters: 0 });
    update(c);
    setTimeout(() => polyInputs.current[c.polygon!.distances.length - 1]?.focus(), 0);
  };
  const setRect = (w: number, l: number) => {
    const c = structuredClone(cal);
    c.points = c.points.map((p, i) => ({ ...p, world: ([[0, 0], [w, 0], [w, l], [0, l]] as [number, number][])[i] }));
    update(c);
  };

  const save = useDebounced(async (c: Calibration) => {
    setSaving("saving");
    try {
      const rd = rectDims(c);
      const clean = {
        ...c,
        segments: c.segments.filter((s) => s.meters > 0),
        points: rd && (rd.w <= 0 || rd.l <= 0) ? [] : c.points,
        polygon: c.polygon ? { ...c.polygon, distances: c.polygon.distances.filter((d) => d.meters > 0) } : null,
      };
      const r = await api.putCalibration(project.id, clean);
      setResult(r.result);
      setSaving("saved");
      onChanged(await api.project(project.id));
    } catch {
      setSaving("error");
    }
  }, 350);

  const update = (c: Calibration) => { setCal(c); save(c); };

  useEffect(() => { if (focusSeg !== null) segInputs.current[focusSeg]?.focus(); }, [focusSeg, cal.segments.length]);
  useEffect(() => { setScrub(frame); }, [frame]);

  const seek = (n: number) => {
    n = Math.max(0, Math.min(info.nb_frames - 1, Math.round(n)));
    setScrub(n);
    if (video) { video.pause(); video.currentTime = (n + 0.5) / info.container_fps; }
  };

  const setReference = (n: number) => {
    const has = cal.segments.length || cal.points.length || cal.line_a || cal.line_b || cal.roi || cal.paths?.length;
    if (has && n !== cal.reference_frame &&
      !confirm("Changing the reference frame: existing points were clicked on the old frame. Keep them anyway? (The drone may have drifted.)")) return;
    update({ ...cal, reference_frame: n });
  };

  const segRes = (i: number) => result.residuals?.filter((r) => r.kind === "segment")[cal.segments.slice(0, i + 1).filter((s) => s.meters > 0).length - 1];
  const paths = cal.paths ?? [];
  const hasPath = paths.some((p) => p.points.length >= 2);
  const setPath = (k: number, patch: Partial<MotionPath> | null) => {
    const c = structuredClone(cal);
    if (patch === null) c.paths!.splice(k, 1);
    else c.paths![k] = { ...c.paths![k], ...patch };
    update(c);
  };
  const setups = cal.red_lights ?? [];
  const solids = cal.solid_lines ?? [];
  const [activeSetup, setActiveSetup] = useState(0);
  const setSetup = (s: number, patch: Partial<RedLightSetup> | null) => {
    const c = structuredClone(cal);
    if (patch === null) { c.red_lights!.splice(s, 1); setActiveSetup(Math.max(0, Math.min(activeSetup, c.red_lights!.length - 1))); }
    else c.red_lights![s] = { ...c.red_lights![s], ...patch };
    update(c);
  };
  const addSetup = () => {
    const c = structuredClone(cal);
    c.red_lights = [...setups, { name: `Light ${setups.length + 1}`, lights: [], stop_lines: [], state_source: "colour", red_intervals: [] }];
    update(c);
    setActiveSetup(c.red_lights.length - 1);
    setTool("light");
  };
  const setSolid = (k: number, patch: Partial<SolidLine> | null) => {
    const c = structuredClone(cal);
    if (patch === null) c.solid_lines!.splice(k, 1);
    else c.solid_lines![k] = { ...c.solid_lines![k], ...patch };
    update(c);
  };
  const vcfg = (project.config.violations ?? {}) as Record<string, any>;
  const setViolationSetting = async (patch: Record<string, any>) => onChanged(await api.patch(project.id, { violations: patch }));
  // what each setup's lights read on the scrubbed frame
  const [lightRead, setLightRead] = useState<{ frame: number; stabilized: boolean; setups: { state: string; counts: [number, number, number][] }[] } | null>(null);
  const lightsKey = JSON.stringify(setups.map((s) => s.lights));
  useEffect(() => {
    if (!setups.some((s) => s.lights.length)) { setLightRead(null); return; }
    const t = setTimeout(() => { api.readLights(project.id, scrub).then(setLightRead).catch(() => setLightRead(null)); }, 400);
    return () => clearTimeout(t);
  }, [scrub, lightsKey, saving === "saved"]);  // eslint-disable-line react-hooks/exhaustive-deps
  const hasRules = setups.some((s) => s.stop_lines.some((l) => l.points.length >= 2)) || solids.some((s) => s.points.length >= 2);
  const ready = !!((hasPath || (cal.line_a && cal.line_b) || hasRules) && !result.error && result.fit_ok !== false);
  const hint = TOOLS.find((t) => t.key === tool)!.hint;
  const stageDone: Record<Stage, boolean> = {
    scale: !result.error && result.fit_ok !== false,
    measure: hasPath || !!(cal.line_a && cal.line_b),
    limits: true,
    rules: hasRules,
  };
  const next = STAGES[STAGES.findIndex((s) => s.key === stage) + 1];

  return (
    <div className="calibrate">
      <div className="cal-main">
        <div className="toolbar" role="toolbar" aria-label="Calibration tools">
          {GROUPS.filter((g) => stageDef.groups.includes(g.key)).map((g) => (
            <div key={g.key} className="tool-group">
              {g.label && <span className="tool-group-label">{g.label}{GROUP_HELP[g.key] && <Help topic={GROUP_HELP[g.key]!} />}</span>}
              <div className="tool-btns">
                {TOOLS.filter((t) => t.group === g.key).map((t) => (
                  <button key={t.key} className={tool === t.key ? "tool active" : "tool"} aria-pressed={tool === t.key}
                    title={t.hint} onClick={() => setTool(t.key)}>
                    {t.color && <span className={t.line ? "tool-dot line" : "tool-dot"} style={{ background: t.color }} />}
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="hintbar">
          <span className="hint" title={hint}>{hint}</span>
          <span className="keys"><kbd>Wheel</kbd> zoom · <kbd>Right-drag</kbd> pan · <kbd>Esc</kbd> cancel</span>
          <RotateButtons project={project} onChanged={onChanged} />
        </div>
        <FrameCanvas src={api.frameUrl(project.id, cal.reference_frame)} cal={cal} tool={tool} onChange={update}
          onSegmentAdded={(i) => { setFocusSeg(i); setTool("move"); }}
          onPolygonAdded={() => { setTool("move"); setTimeout(() => polyInputs.current[0]?.focus(), 0); }}
          onRectAdded={() => { setTool("move"); setTimeout(() => rectW.current?.focus(), 0); }} lineGapM={result.line_gap_m}
          pathsDraw={result.paths} onPathAdded={() => setTool("move")}
          activeSetup={activeSetup} onRuleAdded={() => { if (tool !== "light") setTool("move"); }}
          rotation={project.config.rotation ?? 0}
          selected={focusSeg !== null ? { kind: "seg", i: focusSeg } : null} />
        <div className="scrubber">
          {project.status.proxy_mp4 ? (
            <RotatedVideo rotation={project.config.rotation ?? 0} videoRef={setVideo} src={api.proxyUrl(project.id)} muted playsInline preload="auto" className="scrub-video"
              onClick={(e) => { const v = e.currentTarget; v.paused ? v.play() : v.pause(); }} />
          ) : <div className="scrub-video placeholder">proxy building…</div>}
          <div className="grow scrub-body">
            <VideoControls video={video} containerFps={info.container_fps}
              realTime={(t) => (t * info.container_fps) / (project.config.capture_fps || info.container_fps)} />
            <Timeline project={project} frame={scrub} onSeek={seek} onProjectChange={onChanged} />
            <div className="row small">
              <span className="muted">Frame</span>
              <input type="number" className="narrow" value={scrub} onChange={(e) => seek(+e.target.value)} />
              <span className="muted frame-readout">/ {info.nb_frames - 1} · t = {fmt(scrub / (project.config.capture_fps || info.container_fps), 3)} s real</span>
              <span className="grow" />
              <span className="ref-pill" title="Frame the calibration is drawn on">Reference {cal.reference_frame}</span>
              <button className="primary" disabled={scrub === cal.reference_frame} onClick={() => setReference(scrub)}>Use frame {scrub} as reference</button>
            </div>
          </div>
        </div>
      </div>

      <aside className="cal-side">
        <nav className="stage-tabs" aria-label="Setup stages">
          {STAGES.map((s, i) => (
            <button key={s.key} className={`stage-tab${stage === s.key ? " active" : ""}`} title={s.title} onClick={() => goStage(s.key)}>
              <span className={`stage-num${stageDone[s.key] ? " done" : ""}`}>{stageDone[s.key] && stage !== s.key ? "✓" : i + 1}</span>{s.label}
            </button>
          ))}
        </nav>
        <div className="cal-side-scroll">
        {stage === "scale" && <>
        <section className="panel">
          <div className="row"><h3 className="grow">Calibration<Help topic="calibrate.calibration" /></h3><span className={`tag ${saving === "error" ? "bad" : saving === "saved" ? "ok" : ""}`}>{saving === "saving" ? "saving…" : saving === "saved" ? "saved" : saving === "error" ? "save failed" : ""}</span></div>
          {result.error ? <div className="banner warn">{result.error}</div> : (
            <table className="kv">
              <tbody>
                <tr><td>Model</td><td>{result.method}</td></tr>
                <tr><td>Scale</td><td>{fmt(result.cm_per_px, 3)} cm/px</td></tr>
                <tr><td>Residual RMS</td><td>{fmt((result.rms_m ?? 0) * 100, 1)} cm</td></tr>
                <tr><td>Max rel. error</td><td className={(result.max_rel_error ?? 0) > 0.01 ? "bad-text" : ""}>{fmt((result.max_rel_error ?? 0) * 100, 2)} %</td></tr>
                {result.paths && result.paths.length > 0 ? (
                  <tr><td>Measured along</td><td>{result.paths.length === 1 ? `${result.paths[0].measured_m.toFixed(1)} m of path`
                    : `${result.paths.length} paths · ${result.paths.map((q) => q.measured_m.toFixed(0)).join(" + ")} m`}</td></tr>
                ) : (
                  <tr><td>Line gap A–B</td><td>{result.line_gap_m ? `${result.line_gap_m.toFixed(2)} m` : "–"}</td></tr>
                )}
              </tbody>
            </table>
          )}
          {result.warnings?.map((w, i) => (
            <div key={w} className={`banner small ${i === 0 && result.fit_ok === false ? "error" : "info"}`}>{w}</div>
          ))}
          <p className="muted small">A 20 cm error over a 20 m baseline is already 1 %. Use tape or orthophoto distances, ideally several.</p>
        </section>

        {poly && polyN > 0 && (
          <section className="panel">
            <div className="row"><h3 className="grow">Polygon ({polyN} points)</h3>
              <button className="x" title="Remove polygon" onClick={() => update({ ...cal, polygon: null })}>✕</button></div>
            <p className={`small ${polyMeasured >= 2 * polyN - 3 ? "muted" : "warn-text"}`}>
              {polyMeasured} of {2 * polyN - 3} distances needed to fix the shape
              {polyMeasured > 2 * polyN - 3 ? ` (+${polyMeasured - (2 * polyN - 3)} as checks)` : polyMeasured === 2 * polyN - 3 ? " · add one more as a check" : " · measure sides and diagonals"}
            </p>
            {poly.distances.map((d, k) => {
              const r = result.residuals?.find((x) => x.kind === "polygon" && x.i === d.i && x.j === d.j);
              const side = Math.abs(d.i - d.j) === 1 || Math.abs(d.i - d.j) === polyN - 1;
              return (
                <div key={`${d.i}-${d.j}`} className="item-row">
                  <span className="swatch" style={{ background: "#b98cff", opacity: side ? 1 : 0.5 }} />
                  <span className="grow">{d.i + 1}–{d.j + 1} <span className="muted small">{side ? "side" : "diagonal"}</span></span>
                  <input ref={(el) => (polyInputs.current[k] = el)} type="number" step="any" className="narrow" value={d.meters || ""} placeholder="m"
                    onChange={(e) => setPolyDist(k, parseFloat(e.target.value) || 0)} />
                  <span className="unit">m</span>
                  {r && <span className={`small resid ${Math.abs(r.rel_error ?? 0) > 0.01 ? "bad-text" : "muted"}`} title="Residual">{(r.error_m * 100).toFixed(1)} cm</span>}
                  <button className="x" onClick={() => setPolyDist(k, null)}>✕</button>
                </div>
              );
            })}
            <div className="item-row small">
              <span>Add diagonal</span>
              {[0, 1].map((k) => (
                <select key={k} value={diag[k]} onChange={(e) => setDiag(k ? [diag[0], +e.target.value] : [+e.target.value, diag[1]])}>
                  {poly.points.map((_, i) => <option key={i} value={i}>{i + 1}</option>)}
                </select>
              ))}
              <button onClick={addDiagonal}>Add</button>
            </div>
            <p className="muted small">Assumes the polygon lies on flat ground. Diagonals that cross the area you measure in give the best perspective correction.</p>
          </section>
        )}

        <section className="panel">
          <h3>Distances ({cal.segments.length})</h3>
          {cal.segments.length === 0 && <p className="empty-note">None yet. Use the Distance tool to add a measured ground distance.</p>}
          {cal.segments.map((s, i) => {
            const r = s.meters > 0 ? segRes(i) : undefined;
            return (
              <div key={i} className="item-row">
                <span className="swatch" style={{ background: "#f5c518" }} />
                <span>S{i + 1}</span>
                <span className="muted small grow">{fmt(Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]), 1)} px</span>
                <input ref={(el) => (segInputs.current[i] = el)} type="number" step="any" className="narrow"
                  value={s.meters || ""} placeholder="m"
                  onFocus={() => setFocusSeg(i)}
                  onChange={(e) => { const c = structuredClone(cal); c.segments[i].meters = parseFloat(e.target.value) || 0; update(c); }} />
                <span className="unit">m</span>
                {r && <span className={`small resid ${Math.abs(r.rel_error ?? 0) > 0.01 ? "bad-text" : "muted"}`} title="Residual">{(r.error_m * 100).toFixed(1)} cm</span>}
                <button className="x" onClick={() => { const c = structuredClone(cal); c.segments.splice(i, 1); update(c); setFocusSeg(null); }}>✕</button>
              </div>
            );
          })}
        </section>

        <section className="panel">
          <h3>{rd ? "Rectangle" : `Control points (${cal.points.length})`}</h3>
          {rd && (
            <>
              <div className="item-row">
                <span className="swatch" style={{ background: "#ff8c3a" }} />
                <span className="grow">Edge 1→2</span>
                <input ref={rectW} type="number" step="any" className="narrow" value={rd.w || ""} placeholder="m"
                  onChange={(e) => setRect(parseFloat(e.target.value) || 0, rd.l)} /><span className="unit">m</span>
              </div>
              <div className="item-row">
                <span className="swatch" style={{ background: "#ff8c3a" }} />
                <span className="grow">Edge 2→3</span>
                <input type="number" step="any" className="narrow" value={rd.l || ""} placeholder="m"
                  onChange={(e) => setRect(rd.w, parseFloat(e.target.value) || 0)} /><span className="unit">m</span>
                <button className="x" onClick={() => update({ ...cal, points: [] })}>✕</button>
              </div>
              <p className="muted small">Distance segments become independent checks of the homography: add one or two along the road.</p>
            </>
          )}
          {!rd && cal.points.length === 0 && <p className="empty-note">Optional. World coordinates in metres (e.g. from an orthophoto). ≥4 points → homography.</p>}
          {!rd && cal.points.map((p, i) => {
            const r = result.residuals?.find((x) => x.kind === "point" && x.index === i);
            return (
              <div key={i} className="item-row">
                <span className="grow">P{i + 1}</span>
                {[0, 1].map((k) => (
                  <input key={k} type="number" step="any" className="narrow" value={p.world[k]}
                    onChange={(e) => { const c = structuredClone(cal); c.points[i].world[k] = parseFloat(e.target.value) || 0; update(c); }} />
                ))}
                {r && <span className="muted small resid" title="Residual">{(r.error_m * 100).toFixed(1)} cm</span>}
                <button className="x" onClick={() => { const c = structuredClone(cal); c.points.splice(i, 1); update(c); }}>✕</button>
              </div>
            );
          })}
        </section>

        <TolerancesPanel project={project} onChanged={onChanged} />
        </>}
        {stage === "measure" && <>
        <section className="panel">
          <h3>Paths of motion<Help topic="calibrate.measure" /> {paths.length > 0 && <span className="muted small">({paths.length})</span>}</h3>
          {paths.map((pp, k) => {
            const dr = result.paths?.[k];
            return (
              <div key={k} className="path-item">
                <div className="item-row">
                  <span className="swatch" style={{ background: PATH_COLORS[k % PATH_COLORS.length] }} />
                  <input className="grow path-name" value={pp.name} onChange={(e) => setPath(k, { name: e.target.value })} title="Name, e.g. the road" />
                  <button className="x" title="Remove this path" onClick={() => setPath(k, null)}>✕</button>
                </div>
                <div className="item-row small">
                  <span className="grow muted">{dr ? `${dr.measured_m.toFixed(1)} m measured${dr.length_m - dr.measured_m > 1 ? ` of ${dr.length_m.toFixed(0)} m` : ""}` : `${pp.points.length} points`}</span>
                  <span title="Vehicles further than this from the path (either side) are ignored">± </span>
                  <input type="number" step="0.5" min={0.5} className="narrow" value={pp.corridor_m}
                    onChange={(e) => setPath(k, { corridor_m: Math.max(0.5, parseFloat(e.target.value) || 6) })} />
                  <span className="muted">m</span>
                  <button className="ghost" title="Swap which end is A" onClick={() => setPath(k, { points: [...pp.points].reverse() })}>A ↔ B</button>
                </div>
              </div>
            );
          })}
          {hasPath ? (
            <>
              <label className="small item-row">
                <input type="checkbox" checked={cal.path_clip_to_calibration !== false}
                  onChange={(e) => update({ ...cal, path_clip_to_calibration: e.target.checked })} />
                <span>Only measure inside the calibrated area (recommended)</span>
              </label>
              <p className="muted small">Each vehicle is measured on the path it covers most, over all of it inside the corridor, corrected for its lane on bends. Lines A/B are ignored while paths exist. Dashed = outside the calibrated area.</p>
            </>
          ) : (
            <p className="muted small">Instead of lines A and B, draw one or more paths with the <b>Path</b> tool, e.g. one per road of an intersection. Speeds are measured along the whole curve.</p>
          )}
        </section>

        <section className="panel">
          <h3>Measurement lines & ROI<Help topic="calibrate.area" /></h3>
          {(["line_a", "line_b"] as const).map((k) => (
            <div key={k} className="item-row">
              <span className="swatch" style={{ background: k === "line_a" ? "#00c8ff" : "#ff3cc8" }} />
              <span className="grow">Line {k === "line_a" ? "A" : "B"}</span>
              <span className={`small ${cal[k] ? "ok-text" : "muted"}`}>{cal[k] ? "✓ set" : "not set"}</span>
              {cal[k] && <button className="x" onClick={() => update({ ...cal, [k]: null })}>✕</button>}
            </div>
          ))}
          <div className="item-row">
            <span className="swatch" style={{ background: "#4cd97b" }} />
            <span className="grow">ROI</span>
            <span className="muted small">{cal.roi ? `${cal.roi.length} corners` : "whole frame"}</span>
            {cal.roi && <button className="x" onClick={() => update({ ...cal, roi: null })}>✕</button>}
          </div>
        </section>

        </>}
        {stage === "limits" && <LimitsPanel project={project} onChanged={onChanged} />}
        {stage === "rules" && <>
        <section className="panel">
          <h3>Red lights<Help topic="calibrate.redlights" /> {setups.length > 0 && <span className="muted small">({setups.length})</span>}</h3>
          {setups.length === 0 && <p className="muted small">Mark a traffic light with the <b>Light</b> tool and its stop line with <b>Stop line</b>. A vehicle whose front reaches the stop line while the light is red is reported. Add one setup per signal group.</p>}
          {setups.map((r, s) => {
            const color = LIGHT_COLORS[s % LIGHT_COLORS.length];
            const read = lightRead?.setups[s];
            return (
              <div key={s} className={`path-item rule-setup${s === activeSetup ? " active" : ""}`} onClick={() => setActiveSetup(s)}>
                <div className="item-row">
                  <span className="swatch" style={{ background: color }} />
                  <input className="grow path-name" value={r.name} onChange={(e) => setSetup(s, { name: e.target.value })} title="Name, e.g. the approach" />
                  <button className="x" title="Remove this setup" onClick={(e) => { e.stopPropagation(); setSetup(s, null); }}>✕</button>
                </div>
                <div className="seg-ctl small">
                  <button className={r.state_source === "colour" ? "active" : ""} onClick={() => setSetup(s, { state_source: "colour" })}
                    title="Read red/amber/green from the light's colour on every frame">From the video</button>
                  <button className={r.state_source === "manual" ? "active" : ""} onClick={() => setSetup(s, { state_source: "manual" })}
                    title="Type when the light was red, e.g. when it isn't visible in the shot">Typed red times</button>
                </div>
                {r.state_source === "colour" ? (
                  <>
                    {r.lights.length === 0 && <p className="warn-text small">No light marked: pick the <b>Light</b> tool and click the lamps.</p>}
                    {r.lights.map((lt, i) => (
                      <div key={i} className="item-row small">
                        <span className="grow muted">Light {i + 1}</span>
                        {read?.counts[i] && <LightCounts counts={read.counts[i]} />}
                        <span title="Radius of the circle whose colour is read (px of the original video)">r</span>
                        <input type="number" min={2} step={1} className="narrow" value={lt.radius_px}
                          onChange={(e) => { const c = structuredClone(r); c.lights[i].radius_px = Math.max(2, parseFloat(e.target.value) || 12); setSetup(s, c); }} />
                        <span className="muted">px</span>
                        <button className="x" onClick={() => { const c = structuredClone(r); c.lights.splice(i, 1); setSetup(s, c); }}>✕</button>
                      </div>
                    ))}
                    {read && r.lights.length > 0 && (
                      <div className="item-row small">
                        <span className="muted grow">At frame {lightRead!.frame}{lightRead!.stabilized ? "" : " (as drawn)"}</span>
                        <span className="phase-chip" style={{ background: PHASE_COLORS[read.state] ?? PHASE_COLORS.unknown }}>{read.state}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="item-row small">
                    <span className="muted" title="Seconds of real time, e.g. 12.0-18.5, 40-52">Red</span>
                    <input key={JSON.stringify(r.red_intervals)} className="grow" placeholder="e.g. 12.0-18.5, 40-52"
                      defaultValue={r.red_intervals.map(([a, b]) => `${a}-${b}`).join(", ")}
                      onBlur={(e) => setSetup(s, { red_intervals: parseIntervals(e.target.value) })} />
                    <span className="muted">s</span>
                  </div>
                )}
                {r.stop_lines.length === 0 && <p className="warn-text small">No stop line: pick the <b>Stop line</b> tool.</p>}
                {r.stop_lines.map((sl, l) => (
                  <div key={l} className="item-row small">
                    <span className="grow muted">Stop line {l + 1}</span>
                    <select value={sl.direction} title="Which crossings count; the arrow on the line shows it"
                      onChange={(e) => { const c = structuredClone(r); c.stop_lines[l].direction = e.target.value as StopDirection; setSetup(s, c); }}>
                      <option value="both">both ways</option>
                      <option value="forward">along the arrow</option>
                      <option value="reverse">against the arrow</option>
                    </select>
                    {sl.direction !== "both" && <button className="ghost" title="Flip the direction"
                      onClick={() => { const c = structuredClone(r); c.stop_lines[l].direction = sl.direction === "forward" ? "reverse" : "forward"; setSetup(s, c); }}>⇅</button>}
                    <button className="x" onClick={() => { const c = structuredClone(r); c.stop_lines.splice(l, 1); setSetup(s, c); }}>✕</button>
                  </div>
                ))}
              </div>
            );
          })}
          <div className="row small">
            <button onClick={addSetup}>+ Add red-light setup</button>
            {setups.length > 0 && <span className="muted">New lights and stop lines go to <b>{setups[activeSetup]?.name}</b></span>}
          </div>
          {setups.length > 0 && (
            <div className="item-row small">
              <label className="item-row"><input type="checkbox" checked={!!vcfg.amber_is_red}
                onChange={(e) => setViolationSetting({ amber_is_red: e.target.checked })} /><span>Amber counts as red</span></label>
              <span className="grow" />
              <span title="A vehicle entering this soon after the change to red isn't reported">Grace</span>
              <input type="number" step="0.1" min={0} className="narrow" defaultValue={vcfg.red_grace_s ?? 0}
                onBlur={(e) => setViolationSetting({ red_grace_s: Math.max(0, parseFloat(e.target.value) || 0) })} />
              <span className="muted">s</span>
            </div>
          )}
        </section>

        <section className="panel">
          <h3>Solid lines<Help topic="calibrate.solid" /> {solids.length > 0 && <span className="muted small">({solids.length})</span>}</h3>
          {solids.length === 0 && <p className="muted small">Draw a solid road marking with the <b>Solid line</b> tool. Every vehicle that crosses it, anywhere along it, is reported.</p>}
          {solids.map((sl, k) => (
            <div key={k} className="item-row">
              <span className="swatch" style={{ background: SOLID_COLOR }} />
              <input className="grow path-name" value={sl.name} onChange={(e) => setSolid(k, { name: e.target.value })} />
              <span className="muted small">{sl.points.length} pts</span>
              <button className="x" onClick={() => setSolid(k, null)}>✕</button>
            </div>
          ))}
          {solids.length > 0 && (
            <>
              <div className="seg-ctl small">
                <button className={(vcfg.solid_mode ?? "wheels") === "wheels" ? "active" : ""} onClick={() => setViolationSetting({ solid_mode: "wheels" })}
                  title="Any part of the vehicle gets over the line (driving on it, cutting a corner), and every lane change">Touches the line</button>
                <button className={vcfg.solid_mode === "centre" ? "active" : ""} onClick={() => setViolationSetting({ solid_mode: "centre" })}
                  title="Only when the vehicle's centre goes from one side to the other: a lane change across the line">Crosses it (lane change)</button>
              </div>
              {(vcfg.solid_mode ?? "wheels") === "wheels" && (
                <div className="item-row small">
                  <span title="How far the vehicle's outline must get over the line. Lower catches lighter touches but also tracking noise (normal driving reached 0.29 m on a straight-down clip).">Over by ≥</span>
                  <input key={`o${vcfg.overlap_m}`} type="number" step="0.05" min={0.05} className="narrow" defaultValue={vcfg.overlap_m ?? 0.3}
                    onBlur={(e) => setViolationSetting({ overlap_m: Math.max(0.05, parseFloat(e.target.value) || 0.3) })} />
                  <span className="muted">m for ≥</span>
                  <input key={`d${vcfg.min_duration_s}`} type="number" step="0.1" min={0} className="narrow" defaultValue={vcfg.min_duration_s ?? 0.5}
                    onBlur={(e) => setViolationSetting({ min_duration_s: Math.max(0, parseFloat(e.target.value) || 0) })} />
                  <span className="muted">s</span>
                </div>
              )}
              {(vcfg.solid_mode ?? "wheels") === "wheels" && (
                <div className="item-row small">
                  <label className="item-row" title="Near a line, measure each vehicle by its segmentation mask (its own shape) instead of its box: ~30 ms of GPU per vehicle and frame, only near the line">
                    <input type="checkbox" checked={vcfg.solid_masks ?? true} onChange={(e) => setViolationSetting({ solid_masks: e.target.checked })} />
                    <span>Outline vehicles near the line</span></label>
                  <span className="grow" />
                  <span title="Only vehicles whose box estimate comes this close to a line are outlined">within</span>
                  <input key={`g${vcfg.mask_gate_m}`} type="number" step="0.1" min={0} className="narrow" defaultValue={vcfg.mask_gate_m ?? 0.5}
                    onBlur={(e) => setViolationSetting({ mask_gate_m: Math.max(0, parseFloat(e.target.value) || 0.5) })} />
                  <span className="muted">m</span>
                </div>
              )}
              <p className="muted small">{(vcfg.solid_mode ?? "wheels") === "wheels"
                ? "Near the line a vehicle is measured by its segmentation mask; elsewhere (or without masks) by an outline of its class's width inside its box. Check and tune the threshold on the cases below. "
                : ""}A lane change only counts once the vehicle is {vcfg.hysteresis_m ?? 0.5} m clear of the line on both sides, so tracking wobble along a line isn't reported. Changes apply after Recompute.</p>
            </>
          )}
        </section>
        {solids.length > 0 && project.status.results_json && <SetupCases project={project} onChanged={onChanged} />}

        </>}
        </div>
        <div className="cal-foot">
          <div className="row">
            {next && <button className="grow" onClick={() => goStage(next.key)}>Next: {next.label}</button>}
            <button className="primary grow lg" onClick={onNext}>Continue to Analyze →</button>
          </div>
          {!ready && <p className="muted small">{result.fit_ok === false ? "The calibration doesn't fit its check distances (see above), so nothing would be measured: " : result.error ? "No calibration yet, so nothing will be measured: " : "Nothing to measure yet (no path, lines A/B, stop line or solid line): "}
            the analysis only detects and tracks the vehicles. Set them up later, then press Recompute in Review.</p>}
        </div>
      </aside>
    </div>
  );
}

/** The solid-line cases of the last analysis, to tune the threshold on (see SolidCases). */
function SetupCases({ project, onChanged }: { project: Project; onChanged: (p: Project) => void }) {
  const [res, setRes] = useState<Results | null>(null);
  useEffect(() => { api.results(project.id).then(setRes).catch(() => setRes(null)); }, [project.id]);
  if (!res?.violations) return null;
  return <section className="panel"><SolidCases project={project} res={res} onResults={setRes} onChanged={onChanged} /></section>;
}

/** How well the calibration inputs are known: they make each speed's margin of error. */
function TolerancesPanel({ project, onChanged }: { project: Project; onChanged: (p: Project) => void }) {
  const us = (project.config as any).uncertainty ?? {};
  const set = async (patch: Record<string, unknown>) => onChanged(await api.patch(project.id, { uncertainty: patch }));
  return (
    <section className="panel">
      <h3>Tolerances<Help topic="setup.tolerances" /></h3>
      <div className="form-grid compact">
        <label title="Standard uncertainty of each typed distance: tape ~2 cm, orthophoto ~10 cm">Distances known to ±</label>
        <SettingNum value={Math.round((us.distance_tolerance_m ?? 0.05) * 1000) / 10} unit="cm" step={0.5} min={0}
          onCommit={(v) => set({ distance_tolerance_m: (v ?? 5) / 100 })} />
        <label title="Standard uncertainty of each clicked calibration point">Clicks precise to ±</label>
        <SettingNum value={us.click_tolerance_px ?? 0.5} unit="px" step={0.1} min={0} onCommit={(v) => set({ click_tolerance_px: v ?? 0.5 })} />
        <label title="Drone height above the road, from the flight log. Bounds the height-parallax bias (vehicle tops are closer to the camera, so speeds read slightly high).">Drone height</label>
        <SettingNum value={us.drone_height_m ?? null} unit="m" step={1} min={0} placeholder="unknown" onCommit={(v) => set({ drone_height_m: v })} />
      </div>
      <p className="muted small">They make each speed's margin of error (95 %), Monte Carlo over these tolerances. Without a drone height the height-parallax
        bias isn't in the margin.</p>
    </section>
  );
}

/** The speed limit, the statistics and the speed checks: one place for "when does a measurement count". */
function LimitsPanel({ project, onChanged }: { project: Project; onChanged: (p: Project) => void }) {
  const c = project.config as any;
  const patch = async (section: string, p: Record<string, unknown>) => onChanged(await api.patch(project.id, { [section]: p }));
  return (
    <>
      <section className="panel">
        <h3>Speed limit<Help topic="setup.limit" /></h3>
        <div className="form-grid compact">
          <label>Speed limit</label>
          <SettingNum value={c.render.speed_limit_kmh ?? 50} unit="km/h" step={5} min={5} onCommit={(v) => patch("render", { speed_limit_kmh: v ?? 50 })} />
        </div>
        <p className="muted small">Counts vehicles over the limit in Review, draws the limit on the speed graphs and colours speeds in the video
          (green up to it, amber up to 10 km/h over, red beyond).</p>
      </section>
      <section className="panel">
        <h3>Statistics<Help topic="results.stats" /></h3>
        <div className="form-grid compact">
          <label title="How close the 85th percentile should be known: sets how many vehicles are needed (ITE spot-speed formula)">p85 known to ±</label>
          <SettingNum value={c.stats.permitted_error_kmh ?? 2} unit="km/h" step={0.5} min={0.1} onCommit={(v) => patch("stats", { permitted_error_kmh: v ?? 2 })} />
          <label>Confidence</label>
          <select value={String(c.stats.confidence ?? 0.95)} onChange={(e) => patch("stats", { confidence: +e.target.value })}>
            <option value="0.9">90 %</option><option value="0.95">95 %</option><option value="0.99">99 %</option>
          </select>
        </div>
      </section>
      <section className="panel">
        <h3>When a measurement counts<Help topic="setup.checks" /></h3>
        <div className="form-grid compact">
          <label title="A detection gap longer than this makes the average inconclusive (the max still stands)">Detection gap up to</label>
          <SettingNum value={c.speed.max_gap_s ?? 0.3} unit="s" step={0.05} min={0} onCommit={(v) => patch("speed", { max_gap_s: v ?? 0.3 })} />
          <label title="Less cleanly tracked time than this between the lines: no max">Clean tracking at least</label>
          <SettingNum value={c.speed.min_clean_s ?? 0.5} unit="s" step={0.1} min={0} onCommit={(v) => patch("speed", { min_clean_s: v ?? 0.5 })} />
          <label title="Momentary speed (max, graphs) is measured over this window. Shorter follows changes faster but is noisier.">Momentary speed over</label>
          <SettingNum value={c.speed.momentary_window_s ?? 0.5} unit="s" step={0.05} min={0.1} onCommit={(v) => patch("speed", { momentary_window_s: v ?? 0.5 })} />
        </div>
        <Advanced id="setup-checks">
          <div className="form-grid compact">
            <label title="Accelerations above this flag the vehicle (a tracking jump, not real driving)">Max acceleration</label>
            <SettingNum value={c.speed.max_accel ?? 8} unit="m/s²" step={0.5} min={0} onCommit={(v) => patch("speed", { max_accel: v ?? 8 })} />
            <label title="Max deviation from a local constant-acceleration fit: an ID switch or a box jump is a metre or more">Max position jump</label>
            <SettingNum value={c.speed.max_residual_m ?? 0.5} unit="m" step={0.1} min={0} onCommit={(v) => patch("speed", { max_residual_m: v ?? 0.5 })} />
            <label>Smoothing window</label>
            <SettingNum value={c.speed.smooth_window_s ?? 0.25} unit="s" step={0.05} min={0} onCommit={(v) => patch("speed", { smooth_window_s: v ?? 0.25 })} />
            <label title="Lines mode: the line-crossing speed may differ from the path speed by this share">Line / path disagreement</label>
            <SettingNum value={c.speed.max_disagreement ?? 0.02} step={0.005} min={0} onCommit={(v) => patch("speed", { max_disagreement: v ?? 0.02 })} />
          </div>
        </Advanced>
        <p className="muted small">After an analysis, changes here apply with Recompute in Review; no new analysis is needed.</p>
      </section>
    </>
  );
}

/** "12.0-18.5, 40-52" -> [[12, 18.5], [40, 52]] */
function parseIntervals(text: string): [number, number][] {
  return text.split(/[,;]/).map((part) => part.trim().match(/^(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => [parseFloat(m[1]), parseFloat(m[2])] as [number, number])
    .filter(([a, b]) => b > a);
}

/** Lit pixels per colour inside a light's circle on the current frame. */
function LightCounts({ counts }: { counts: [number, number, number] }) {
  const names = ["red", "amber", "green"];
  return (
    <span className="light-counts" title="Bright pixels of each colour inside the circle on this frame">
      {counts.map((n, i) => <span key={i} style={{ color: n > 0 ? PHASE_COLORS[names[i]] : undefined }} className={n > 0 ? "" : "muted"}>{n}</span>)}
    </span>
  );
}
