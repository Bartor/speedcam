import { useState } from "react";
import { fmt, Vehicle } from "../api";
import "./speed-profile.css";

interface Props { v: Vehicle; limit: number; tNow?: number }

/** Momentary speed along the A→B section for one vehicle: max/min marked, average and
 *  speed-limit reference lines, ±2σ noise band, hover readout, playback cursor. */
export default function SpeedProfile({ v, limit, tNow }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const p = v.profile;
  if (!p || p.v.length < 2) return <div className="muted small">No speed profile for this vehicle.</div>;

  const W = 640, H = 220, m = { l: 40, r: 14, t: 18, b: 30 };
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
  const xStep = dEnd > 60 ? 20 : dEnd > 25 ? 10 : 5;
  const xTicks: number[] = [];
  for (let d = 0; d <= dEnd + 1e-9; d += xStep) xTicks.push(d);

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
    <div className="sprof">
      <div className="sprof-head">
        <div className="sprof-kpi">
          <span className="sprof-kpi-label">Max</span>
          <span className={`sprof-kpi-val ${above ? "bad-text" : ""}`}>{fmt(v.speed_max_kmh)}</span>
          <span className="sprof-kpi-unit">km/h{sig ? ` ±${(2 * sig).toFixed(1)}` : ""}</span>
        </div>
        <div className="sprof-kpi"><span className="sprof-kpi-label">Average A→B</span><span className="sprof-kpi-val">{fmt(v.speed_kmh)}</span><span className="sprof-kpi-unit">km/h</span></div>
        <div className="sprof-kpi"><span className="sprof-kpi-label">Min</span><span className="sprof-kpi-val">{fmt(v.speed_min_kmh)}</span><span className="sprof-kpi-unit">km/h</span></div>
        <div className="sprof-kpi"><span className="sprof-kpi-label">Max at</span><span className="sprof-kpi-val">{fmt(v.d_max_m)}</span><span className="sprof-kpi-unit">m after the first line</span></div>
      </div>
      <div className="sprof-plot">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label={`Speed along the section for vehicle ${v.track_id}: max ${fmt(v.speed_max_kmh)} km/h, average ${fmt(v.speed_kmh)} km/h`}>
          {yTicks.map((s) => (
            <g key={s}>
              <line x1={m.l} x2={W - m.r} y1={y(s)} y2={y(s)} className="grid" />
              <text x={m.l - 6} y={y(s) + 4} className="axis" textAnchor="end">{s}</text>
            </g>
          ))}
          {xTicks.map((d) => <text key={d} x={x(d)} y={H - 10} className="axis" textAnchor="middle">{d} m</text>)}
          {band && <path d={band} className="band" />}
          <line x1={m.l} x2={W - m.r} y1={y(limit)} y2={y(limit)} className="limit" />
          <text x={W - m.r} y={y(limit) - 5} className="ref-label limit-label" textAnchor="end">limit {limit}</text>
          <line x1={m.l} x2={W - m.r} y1={y(v.speed_kmh)} y2={y(v.speed_kmh)} className="avg" />
          <text x={m.l + 4} y={y(v.speed_kmh) - 5} className="ref-label">avg {fmt(v.speed_kmh)}</text>
          <path d={line} className="series" />
          <circle cx={x(p.d[iMin])} cy={y(p.v[iMin])} r={4} className="mark min" />
          <circle cx={x(p.d[iMax])} cy={y(p.v[iMax])} r={5.5} className={`mark max ${above ? "over" : ""}`} />
          <text x={x(p.d[iMax])} y={y(p.v[iMax]) - 10} className="max-label"
            textAnchor={x(p.d[iMax]) > W - 80 ? "end" : x(p.d[iMax]) < m.l + 60 ? "start" : "middle"}>max {fmt(p.v[iMax])}</text>
          {hi_ !== null && (
            <g pointerEvents="none">
              <line x1={x(p.d[hi_])} x2={x(p.d[hi_])} y1={m.t} y2={m.t + ih} className={hover !== null ? "cross" : "cursor"} />
              <circle cx={x(p.d[hi_])} cy={y(p.v[hi_])} r={4.5} className="mark hover" />
            </g>
          )}
          <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
        {hi_ !== null && (
          <div className="sprof-tip" style={{ left: `${(x(p.d[hi_]) / W) * 100}%`, top: `${(y(p.v[hi_]) / H) * 100}%` }}>
            <b>{p.v[hi_].toFixed(1)} km/h</b> · {p.d[hi_].toFixed(1)} m · +{p.t[hi_].toFixed(2)} s
          </div>
        )}
      </div>
      <div className="sprof-foot small muted">
        Momentary speed (0.5 s window) along the section · shaded = ±2σ measurement noise · dashed = average A→B · red = speed limit
      </div>
    </div>
  );
}
