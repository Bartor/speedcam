import { useEffect, useRef, useState } from "react";
import { api, Calibration, CalibrationResult, fmt, Project, useDebounced, useVideoFrame } from "../api";
import FrameCanvas, { rectDims, Tool } from "../components/FrameCanvas";

interface Props { project: Project; onChanged: (p: Project) => void; onNext: () => void }

const TOOLS: { key: Tool; label: string; hint: string }[] = [
  { key: "move", label: "Move", hint: "Drag handles to adjust; drag empty space to pan." },
  { key: "polygon", label: "Polygon", hint: "Click 4+ flat ground points (close with Enter, double-click or the first point), then type measured sides and diagonals. Corrects perspective: best for oblique views." },
  { key: "rect", label: "Rectangle", hint: "Click 4 ground corners of a known rectangle in order around it (e.g. lane width × marking spacing), then type both edge lengths. Gives a homography: required for oblique views." },
  { key: "segment", label: "Distance", hint: "Click two ground points, then type the measured distance (m)." },
  { key: "point", label: "Control pt", hint: "Click a ground point with known world coordinates (m). 4+ points give a full homography." },
  { key: "lineA", label: "Line A", hint: "Click two points across the road for measurement line A." },
  { key: "lineB", label: "Line B", hint: "Click two points across the road for measurement line B." },
  { key: "roi", label: "ROI", hint: "Click polygon corners; close with Enter, double-click or by clicking the first point." },
];

export default function CalibrateScreen({ project, onChanged, onNext }: Props) {
  const info = project.info;
  const [cal, setCal] = useState<Calibration>(project.config.calibration);
  const [result, setResult] = useState<CalibrationResult>(project.calibration_result);
  const [saving, setSaving] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [tool, setTool] = useState<Tool>("move");
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
    const has = cal.segments.length || cal.points.length || cal.line_a || cal.line_b || cal.roi;
    if (has && n !== cal.reference_frame &&
      !confirm("Changing the reference frame: existing points were clicked on the old frame. Keep them anyway? (The drone may have drifted.)")) return;
    update({ ...cal, reference_frame: n });
  };

  const segRes = (i: number) => result.residuals?.filter((r) => r.kind === "segment")[cal.segments.slice(0, i + 1).filter((s) => s.meters > 0).length - 1];
  const ready = !!(cal.line_a && cal.line_b && !result.error && result.fit_ok !== false);
  const hint = TOOLS.find((t) => t.key === tool)!.hint;

  return (
    <div className="calibrate">
      <div className="cal-main">
        <div className="toolbar">
          {TOOLS.map((t) => (
            <button key={t.key} className={tool === t.key ? "tool active" : "tool"} onClick={() => setTool(t.key)}>{t.label}</button>
          ))}
          <span className="muted small grow hint" title={hint}>{hint} Wheel zooms, right-drag pans, Esc cancels.</span>
        </div>
        <FrameCanvas src={api.frameUrl(project.id, cal.reference_frame)} cal={cal} tool={tool} onChange={update}
          onSegmentAdded={(i) => { setFocusSeg(i); setTool("move"); }}
          onPolygonAdded={() => { setTool("move"); setTimeout(() => polyInputs.current[0]?.focus(), 0); }}
          onRectAdded={() => { setTool("move"); setTimeout(() => rectW.current?.focus(), 0); }} lineGapM={result.line_gap_m}
          selected={focusSeg !== null ? { kind: "seg", i: focusSeg } : null} />
        <div className="scrubber">
          {project.status.proxy_mp4 ? (
            <video ref={setVideo} src={api.proxyUrl(project.id)} muted playsInline preload="auto" className="scrub-video"
              onClick={(e) => { const v = e.currentTarget; v.paused ? v.play() : v.pause(); }} />
          ) : <div className="scrub-video placeholder">proxy building…</div>}
          <div className="grow">
            <input type="range" min={0} max={info.nb_frames - 1} value={scrub} className="wide"
              onChange={(e) => seek(+e.target.value)} />
            <div className="row small">
              <button onClick={() => seek(scrub - 1)}>◀ 1</button>
              <button onClick={() => seek(scrub + 1)}>1 ▶</button>
              <span>frame</span>
              <input type="number" className="narrow" value={scrub} onChange={(e) => seek(+e.target.value)} />
              <span className="muted">/ {info.nb_frames - 1} · t = {fmt(scrub / (project.config.capture_fps || info.container_fps), 3)} s real</span>
              <span className="grow" />
              <span className="muted">reference: frame {cal.reference_frame}</span>
              <button className="primary" disabled={scrub === cal.reference_frame} onClick={() => setReference(scrub)}>Use frame {scrub} as reference</button>
            </div>
          </div>
        </div>
      </div>

      <aside className="cal-side">
        <section className="panel">
          <div className="row"><h3 className="grow">Calibration</h3><span className={`tag ${saving === "error" ? "bad" : ""}`}>{saving === "saving" ? "saving…" : saving === "saved" ? "saved" : saving === "error" ? "save failed" : ""}</span></div>
          {result.error ? <div className="banner warn">{result.error}</div> : (
            <table className="kv">
              <tbody>
                <tr><td>Model</td><td>{result.method}</td></tr>
                <tr><td>Scale</td><td>{fmt(result.cm_per_px, 3)} cm/px</td></tr>
                <tr><td>Residual RMS</td><td>{fmt((result.rms_m ?? 0) * 100, 1)} cm</td></tr>
                <tr><td>Max rel. error</td><td className={(result.max_rel_error ?? 0) > 0.01 ? "bad-text" : ""}>{fmt((result.max_rel_error ?? 0) * 100, 2)} %</td></tr>
                <tr><td>Line gap A–B</td><td>{result.line_gap_m ? `${result.line_gap_m.toFixed(2)} m` : "–"}</td></tr>
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
                  <span className="small">m</span>
                  {r && <span className={`small ${Math.abs(r.rel_error ?? 0) > 0.01 ? "bad-text" : "muted"}`}>{(r.error_m * 100).toFixed(1)} cm</span>}
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
          {cal.segments.map((s, i) => {
            const r = s.meters > 0 ? segRes(i) : undefined;
            return (
              <div key={i} className="item-row">
                <span className="swatch" style={{ background: "#f5c518" }} />
                <span>S{i + 1}</span>
                <span className="muted small">{fmt(Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]), 1)} px</span>
                <input ref={(el) => (segInputs.current[i] = el)} type="number" step="any" className="narrow"
                  value={s.meters || ""} placeholder="m"
                  onFocus={() => setFocusSeg(i)}
                  onChange={(e) => { const c = structuredClone(cal); c.segments[i].meters = parseFloat(e.target.value) || 0; update(c); }} />
                <span className="small">m</span>
                {r && <span className={`small ${Math.abs(r.rel_error ?? 0) > 0.01 ? "bad-text" : "muted"}`}>{(r.error_m * 100).toFixed(1)} cm</span>}
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
                  onChange={(e) => setRect(parseFloat(e.target.value) || 0, rd.l)} /><span className="small">m</span>
              </div>
              <div className="item-row">
                <span className="swatch" style={{ background: "#ff8c3a" }} />
                <span className="grow">Edge 2→3</span>
                <input type="number" step="any" className="narrow" value={rd.l || ""} placeholder="m"
                  onChange={(e) => setRect(rd.w, parseFloat(e.target.value) || 0)} /><span className="small">m</span>
                <button className="x" onClick={() => update({ ...cal, points: [] })}>✕</button>
              </div>
              <p className="muted small">Distance segments become independent checks of the homography: add one or two along the road.</p>
            </>
          )}
          {!rd && cal.points.length === 0 && <p className="muted small">Optional. World coordinates in metres (e.g. from an orthophoto). ≥4 points → homography.</p>}
          {!rd && cal.points.map((p, i) => {
            const r = result.residuals?.find((x) => x.kind === "point" && x.index === i);
            return (
              <div key={i} className="item-row">
                <span>P{i + 1}</span>
                {[0, 1].map((k) => (
                  <input key={k} type="number" step="any" className="narrow" value={p.world[k]}
                    onChange={(e) => { const c = structuredClone(cal); c.points[i].world[k] = parseFloat(e.target.value) || 0; update(c); }} />
                ))}
                {r && <span className="muted small">{(r.error_m * 100).toFixed(1)} cm</span>}
                <button className="x" onClick={() => { const c = structuredClone(cal); c.points.splice(i, 1); update(c); }}>✕</button>
              </div>
            );
          })}
        </section>

        <section className="panel">
          <h3>Measurement lines & ROI</h3>
          {(["line_a", "line_b"] as const).map((k) => (
            <div key={k} className="item-row">
              <span className="swatch" style={{ background: k === "line_a" ? "#00c8ff" : "#ff3cc8" }} />
              <span className="grow">Line {k === "line_a" ? "A" : "B"}</span>
              <span className="muted small">{cal[k] ? "set" : "not set"}</span>
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

        <button className="primary wide" disabled={!ready} onClick={onNext}>Continue to processing →</button>
        {!ready && <p className="muted small">{result.fit_ok === false ? "The calibration does not fit its check distances (see above)." : "Needs a calibration (rectangle, distance or control points) and both lines."}</p>}
      </aside>
    </div>
  );
}
