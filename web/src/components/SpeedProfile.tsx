import { useEffect, useRef, useState } from "react";
import { fmt, margin, marginBreakdown, Results, Vehicle } from "../api";
import "./speed-profile.css";

/** A violation of this vehicle, marked on its profile (times in real seconds). */
export interface ProfileEvent { t: number; t1?: number | null; label: string }

/** The vehicle's violations from the results, as profile markers. */
export function vehicleEvents(res: Results, tid: number): ProfileEvent[] {
  const v = res.violations;
  if (!v) return [];
  return [
    ...v.red_light.filter((e) => e.track_id === tid && e.violation).map((e) => ({ t: e.t_s, label: `Red light · ${e.name}` })),
    ...v.solid_line.filter((e) => e.track_id === tid).map((e) => ({
      t: e.t_s, t1: e.duration_s ? e.t_s + e.duration_s : null,
      label: `${e.how === "straddled" ? "On the solid line" : "Crossed the solid line"} · ${e.name}`,
    })),
  ].sort((a, b) => a.t - b.t);
}

interface Props { v: Vehicle; limit: number; tNow?: number; events?: ProfileEvent[]; onSeek?: (t: number) => void }

/** Momentary speed along the A→B section for one vehicle: max/min marked, average and
 *  speed-limit reference lines, ±2σ noise band, violations, hover readout, playback cursor.
 *  Drawn at the size of its box, so text stays text-sized on a wide screen. */
export default function SpeedProfile({ v, limit, tNow, events = [], onSeek }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(280, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const p = v.profile;
  if (!p || p.v.length < 2) return <div ref={box} className="muted small">No speed profile for this vehicle.</div>;

  const H = 240, m = { l: 40, r: 14, t: events.length > 1 ? 40 : events.length ? 28 : 18, b: 26 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sig = v.speed_sigma_kmh ?? 0;
  const vmin = Math.min(...p.v, limit, v.speed_kmh) - Math.max(2, 2 * sig);
  const vmax = Math.max(...p.v, limit, v.speed_kmh) + Math.max(2, 2 * sig);
  const lo = Math.floor(vmin / 5) * 5, hi = Math.ceil(vmax / 5) * 5;
  const dEnd = p.d[p.d.length - 1] || 1;
  const x = (d: number) => m.l + (d / dEnd) * iw;
  const y = (s: number) => m.t + ih - ((s - lo) / (hi - lo)) * ih;

  const line = p.d.map((d, i) => `${i ? "L" : "M"}${x(d).toFixed(1)},${y(p.v[i]).toFixed(1)}`).join("");
  const band = sig > 0
    ? p.d.map((d, i) => `${i ? "L" : "M"}${x(d).toFixed(1)},${y(p.v[i] + 2 * sig).toFixed(1)}`).join("") +
      [...p.d].reverse().map((d, j) => `L${x(d).toFixed(1)},${y(p.v[p.v.length - 1 - j] - 2 * sig).toFixed(1)}`).join("") + "Z"
    : "";
  const iMax = p.v.indexOf(Math.max(...p.v));
  const iMin = p.v.indexOf(Math.min(...p.v));
  const yTicks: number[] = [];
  for (let s = lo; s <= hi + 1e-9; s += (hi - lo) > 40 ? 10 : 5) yTicks.push(s);
  const xStep = [1, 2, 5, 10, 20, 50, 100].find((s) => (iw / dEnd) * s >= 60) ?? 200;
  const xTicks: number[] = [];
  for (let d = 0; d <= dEnd + 1e-9; d += xStep) xTicks.push(d);

  // time since the first line -> the profile sample (clamped to the section)
  const at = (tAbs: number) => {
    const tr = tAbs - v.t_entry_s;
    let i = 0;
    while (i < p.t.length - 1 && p.t[i + 1] <= tr) i++;
    if (tr <= p.t[0]) return { d: p.d[0], s: p.v[0], out: tr < p.t[0] - 0.05 ? "before" : "" };
    if (i >= p.t.length - 1) return { d: p.d[p.d.length - 1], s: p.v[p.v.length - 1], out: tr > p.t[p.t.length - 1] + 0.05 ? "after" : "" };
    const k = (tr - p.t[i]) / Math.max(1e-9, p.t[i + 1] - p.t[i]);
    return { d: p.d[i] + k * (p.d[i + 1] - p.d[i]), s: p.v[i] + k * (p.v[i + 1] - p.v[i]), out: "" };
  };

  // playback cursor: time since the first line, if the vehicle is between the lines right now
  const tRel = tNow !== undefined ? tNow - v.t_entry_s : null;
  const cursorI = tRel !== null && tRel >= 0 && tRel <= v.dt_s
    ? p.t.reduce((best, t, i) => (Math.abs(t - tRel) < Math.abs(p.t[best] - tRel) ? i : best), 0) : null;

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const d = ((e.clientX - r.left) / r.width) * dEnd;
    let best = 0;
    p.d.forEach((dd, i) => { if (Math.abs(dd - d) < Math.abs(p.d[best] - d)) best = i; });
    setHover(best);
  };
  const hi_ = hover ?? cursorI;
  const above = v.speed_max_kmh > limit;

  return (
    <div className="sprof" ref={box}>
      <div className="sprof-head">
        <div className="sprof-kpi" title={marginBreakdown(v, "max")}>
          <span className="sprof-kpi-label">Max</span>
          <span className={`sprof-kpi-val ${above ? "bad-text" : ""}`}>{fmt(v.speed_max_kmh)}</span>
          <span className="sprof-kpi-unit">km/h {margin(v, "max") ?? (sig ? `±${(2 * sig).toFixed(1)} noise` : "")}</span>
        </div>
        {v.avg_valid === false ? (
          <div className="sprof-kpi" title={`A detection gap or position jump in the stretch (${v.flags}) makes the average inconclusive. The max only uses cleanly tracked frames.`}>
            <span className="sprof-kpi-label">Average A→B</span><span className="sprof-kpi-val">–</span>
            <span className="sprof-kpi-unit">inconclusive</span>
          </div>
        ) : (
          <div className="sprof-kpi" title={marginBreakdown(v, "avg")}>
            <span className="sprof-kpi-label">Average A→B</span><span className="sprof-kpi-val">{fmt(v.speed_kmh)}</span>
            <span className="sprof-kpi-unit">km/h {margin(v, "avg") ?? ""}</span>
          </div>
        )}
        <div className="sprof-kpi"><span className="sprof-kpi-label">Min</span><span className="sprof-kpi-val">{fmt(v.speed_min_kmh)}</span><span className="sprof-kpi-unit">km/h</span></div>
        <div className="sprof-kpi"><span className="sprof-kpi-label">Max at</span><span className="sprof-kpi-val">{fmt(v.d_max_m)}</span><span className="sprof-kpi-unit">m after the first line</span></div>
        {events.length > 0 && (
          <div className="sprof-kpi"><span className="sprof-kpi-label">Violations</span><span className="sprof-kpi-val bad-text">{events.length}</span><span className="sprof-kpi-unit">marked on the graph</span></div>
        )}
      </div>
      <div className="sprof-plot">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img"
          aria-label={`Speed along the section for vehicle ${v.track_id}: max ${fmt(v.speed_max_kmh)} km/h, average ${fmt(v.speed_kmh)} km/h`}>
          {yTicks.map((s) => (
            <g key={s}>
              <line x1={m.l} x2={W - m.r} y1={y(s)} y2={y(s)} className="grid" />
              <text x={m.l - 6} y={y(s) + 4} className="axis" textAnchor="end">{s}</text>
            </g>
          ))}
          {xTicks.map((d) => <text key={d} x={x(d)} y={H - 8} className="axis" textAnchor="middle">{d} m</text>)}
          {band && <path d={band} className="band" />}
          {events.map((e, k) => {
            if (!e.t1) return null;
            const a = at(e.t), b = at(e.t1);
            return <rect key={`b${k}`} x={x(a.d)} y={m.t} width={Math.max(2, x(b.d) - x(a.d))} height={ih} className="viol-band" />;
          })}
          <line x1={m.l} x2={W - m.r} y1={y(limit)} y2={y(limit)} className="limit" />
          <text x={W - m.r} y={y(limit) - 5} className="ref-label limit-label" textAnchor="end">limit {limit}</text>
          {v.avg_valid !== false && <>
            <line x1={m.l} x2={W - m.r} y1={y(v.speed_kmh)} y2={y(v.speed_kmh)} className="avg" />
            <text x={m.l + 4} y={y(v.speed_kmh) - 5} className="ref-label">avg {fmt(v.speed_kmh)}</text>
          </>}
          <path d={line} className="series" />
          <circle cx={x(p.d[iMin])} cy={y(p.v[iMin])} r={4} className="mark min" />
          <circle cx={x(p.d[iMax])} cy={y(p.v[iMax])} r={5.5} className={`mark max ${above ? "over" : ""}`} />
          <text x={x(p.d[iMax])} y={y(p.v[iMax]) - 10} className="max-label"
            textAnchor={x(p.d[iMax]) > W - 80 ? "end" : x(p.d[iMax]) < m.l + 60 ? "start" : "middle"}>max {fmt(p.v[iMax])}</text>
          {events.map((e, k) => {
            const a = at(e.t);
            const ex = x(a.d), ey = y(a.s);
            const right = ex > W - 200;
            const label = `⛔ ${e.label}${a.out ? ` (${a.out} the section, ${e.t.toFixed(1)} s)` : ` · ${a.s.toFixed(1)} km/h`}`;
            return (
              <g key={`e${k}`} className="viol" onClick={() => onSeek?.(e.t - 1)} style={{ cursor: onSeek ? "pointer" : undefined }}>
                <title>{`${e.label} at ${e.t.toFixed(2)} s${a.out ? ` (${a.out} the measured section)` : `, ${a.s.toFixed(1)} km/h, ${a.d.toFixed(1)} m after the first line`}`}</title>
                <line x1={ex} x2={ex} y1={m.t - 4} y2={m.t + ih} className="viol-line" />
                <path d={`M${ex},${ey - 7} L${ex + 6},${ey} L${ex},${ey + 7} L${ex - 6},${ey} Z`} className="viol-mark" />
                <text x={ex + (right ? -5 : 5)} y={m.t - 8 - (k % 2) * 12} className="viol-label" textAnchor={right ? "end" : "start"}>{label}</text>
              </g>
            );
          })}
          {hi_ !== null && (
            <g pointerEvents="none">
              <line x1={x(p.d[hi_])} x2={x(p.d[hi_])} y1={m.t} y2={m.t + ih} className={hover !== null ? "cross" : "cursor"} />
              <circle cx={x(p.d[hi_])} cy={y(p.v[hi_])} r={4.5} className="mark hover" />
            </g>
          )}
          <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)}
            onClick={() => hover !== null && onSeek?.(v.t_entry_s + p.t[hover])} style={{ cursor: onSeek ? "pointer" : undefined }} />
        </svg>
        {hi_ !== null && (
          <div className="sprof-tip" style={{ left: x(p.d[hi_]), top: y(p.v[hi_]) }}>
            <b>{p.v[hi_].toFixed(1)} km/h</b> · {p.d[hi_].toFixed(1)} m · +{p.t[hi_].toFixed(2)} s
          </div>
        )}
      </div>
      <div className="sprof-foot small muted">
        Momentary speed (0.5 s window) along the section · shaded = ±2σ measurement noise · dashed = average A→B · red = speed limit
        {events.length > 0 && " · ⛔ = violation (click to watch)"} · click the graph to jump the video there
      </div>
    </div>
  );
}
