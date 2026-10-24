import { useState } from "react";
import Help from "./Help";
import { PHASE_COLORS, RedLightEvent, Results, SolidLineEvent } from "../api";

interface Props {
  res: Results;
  tNow: number;
  selected: number | null;
  onPick: (trackId: number, t: number) => void;
  onSeek: (t: number) => void;
  hasSolid: boolean;
  solidMode: string;
  minDuration?: number;
}

type Row = (RedLightEvent | SolidLineEvent) & { verdict: "violation" | "ok" | "unknown" };

/** Red-light running and solid-line crossings: light phase timelines and the list of events. */
export default function ViolationsPanel({ res, tNow, selected, onPick, onSeek, hasSolid, solidMode, minDuration = 0.5 }: Props) {
  const v = res.violations!;
  const [all, setAll] = useState(false);
  const duration = res.nb_frames / res.capture_fps;
  const rows: Row[] = [
    ...v.red_light.map((e) => ({ ...e, verdict: (e.violation === null ? "unknown" : e.violation ? "violation" : "ok") as Row["verdict"] })),
    ...v.solid_line.map((e) => ({ ...e, verdict: "violation" as const })),
  ].filter((e) => all || e.verdict !== "ok").sort((a, b) => a.t_s - b.t_s);
  const passedOk = v.red_light.filter((e) => e.violation === false).length;

  return (
    <section className="panel violations">
      <div className="panel-head">
        <h3>Violations<Help topic="results.violations" /></h3>
        {v.lights.length > 0 && <span className={`tag ${v.n_red_light ? "bad" : ""}`}>{v.n_red_light} red-light</span>}
        {hasSolid && <span className={`tag ${v.n_solid_line ? "bad" : ""}`}>{v.n_solid_line} solid-line</span>}
        <span className="grow" />
        {passedOk > 0 && (
          <label className="small"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> also {passedOk} stop-line crossing{passedOk === 1 ? "" : "s"} when not red</label>
        )}
      </div>
      {v.warnings.map((w) => <div key={w} className="banner warn small">{w}</div>)}

      {v.lights.map((lt) => (
        <div key={lt.name} className="phase-row">
          <span className="phase-name" title={lt.source === "colour" ? `Read from the video${lt.read_share != null ? `; colour seen on ${Math.round(lt.read_share * 100)} % of frames` : ""}` : "Typed red intervals"}>
            {lt.name}<span className="muted small"> · {lt.source === "colour" ? "video" : "typed"}</span>
          </span>
          <div className="phase-bar" onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            onSeek(((e.clientX - r.left) / r.width) * duration);
          }}>
            {lt.phases.map((p, i) => (
              <span key={i} className="phase-seg" title={`${p.state} ${p.t0.toFixed(1)}–${p.t1.toFixed(1)} s`}
                style={{ left: `${(p.t0 / duration) * 100}%`, width: `${((p.t1 - p.t0) / duration) * 100}%`, background: PHASE_COLORS[p.state] ?? PHASE_COLORS.unknown }} />
            ))}
            {v.red_light.filter((e) => e.name === lt.name && e.violation).map((e, i) => (
              <span key={`e${i}`} className="phase-event" style={{ left: `${(e.t_s / duration) * 100}%` }} title={`#${e.track_id} at ${e.t_s.toFixed(2)} s`} />
            ))}
            <span className="phase-now" style={{ left: `${Math.min(100, (tNow / duration) * 100)}%` }} />
          </div>
        </div>
      ))}

      {rows.length === 0 ? (
        <p className="empty-note">{v.red_light.length + v.solid_line.length === 0 ? "No vehicle crossed a stop line or solid line." : "No violations."}</p>
      ) : (
        <div className="table-wrap">
          <table className="vehicles">
            <thead>
              <tr><th className="num">Time s</th><th className="num">#</th><th>Class</th><th>Rule</th><th>What happened</th><th className="num">km/h</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              {rows.map((e, i) => {
                const active = Math.abs(tNow - e.t_s) < 1;
                return (
                  <tr key={i} className={`${active ? "active" : ""} ${selected === e.track_id ? "selected" : ""}`} onClick={() => onPick(e.track_id, e.t_s)}>
                    <td className="num">{e.t_s.toFixed(2)}</td>
                    <td className="num">{e.track_id}</td>
                    <td>{e.class}</td>
                    <td>{e.type === "red_light" ? `${e.name} · stop line ${e.line}` : e.name}</td>
                    <td>{describe(e)}</td>
                    <td className="num">{e.speed_kmh != null ? e.speed_kmh.toFixed(0) : "–"}</td>
                    <td>{e.verdict === "violation" ? <span className="status bad">{e.type === "red_light" ? "✕ ran the red" : e.how === "straddled" ? "✕ on the line" : "✕ crossed"}</span>
                      : e.verdict === "unknown" ? <span className="status warn" title="The light's phase isn't known at that moment">? phase unknown</span>
                        : <span className="status ok">✓ ok</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {(v.solid_near ?? []).length > 0 && !(v.solid_cases ?? []).length && (
        <div className="near-misses">
          <div className="subhead">Over a solid line, but under the threshold</div>
          {v.solid_near!.map((e) => (
            <button key={`${e.track_id}-${e.name}`} className="near-row small" onClick={() => onPick(e.track_id, e.t_s)}>
              <span>#{e.track_id} {e.class} · {e.name}</span>
              <span className="muted">{e.over_m.toFixed(2)} m over for {e.for_s} s, at {e.t_s.toFixed(1)} s</span>
            </button>
          ))}
          <p className="footnote">Reported from {v.overlap_m ?? "–"} m over for {minDuration} s (Setup → Rules). The outline comes from the
            detection box and the class's usual width, so expect ± 0.1-0.2 m; in a view at an angle the box also holds the vehicle's side.</p>
        </div>
      )}
      <p className="footnote">Red light: the moment the front of the vehicle's box reached the stop line, and the light's phase then.
        Solid line: {solidMode === "wheels"
          ? `the vehicle's outline got over the line by ${v.overlap_m ?? 0.3} m for ${minDuration} s, or its centre crossed it (a lane change, at least 0.5 m clear on both sides)`
          : "its centre crossed the line, at least 0.5 m clear on both sides"}. Click a row to watch it.</p>
    </section>
  );
}

function describe(e: Row): string {
  if (e.type === "red_light") {
    const phase = e.light === "unknown" ? "phase unknown" : e.light;
    const since = e.phase_for_s != null ? ` for ${e.phase_for_s.toFixed(1)} s` : "";
    return `entered on ${phase}${e.light === "red" || e.light === "amber" ? since : ""}`;
  }
  if (e.how === "straddled") return `drove over the line for ${e.duration_s?.toFixed(1)} s (up to ${e.max_over_m?.toFixed(1)} m)`;
  return e.duration_s != null ? `crossed, over the line ${e.duration_s.toFixed(1)} s` : "crossed the line";
}
