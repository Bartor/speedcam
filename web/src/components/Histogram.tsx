import { useState } from "react";

interface Props { values: number[]; mean?: number; p85?: number; binWidth?: number }

/** Speed histogram: single series, recessive axes, mean/p85 reference lines, per-bar hover + focus tooltip. */
export default function Histogram({ values, mean, p85, binWidth = 2 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  if (values.length === 0) return <div className="muted small hist">No valid measurements yet.</div>;
  const lo = Math.floor(Math.min(...values) / binWidth) * binWidth;
  const hi = Math.max(lo + binWidth, Math.ceil((Math.max(...values) + 1e-9) / binWidth) * binWidth);
  const n = Math.round((hi - lo) / binWidth);
  const counts = new Array(n).fill(0);
  values.forEach((v) => counts[Math.min(n - 1, Math.floor((v - lo) / binWidth))]++);
  const maxC = Math.max(...counts);

  const W = 460, H = 200, m = { l: 32, r: 10, t: 22, b: 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (v: number) => m.l + ((v - lo) / (hi - lo)) * iw;
  const y = (c: number) => m.t + ih - (c / maxC) * ih;
  const bw = iw / n;
  const yTicks = Array.from(new Set([0, Math.ceil(maxC / 2), maxC]));
  const xStep = Math.max(binWidth, Math.ceil((hi - lo) / 8 / binWidth) * binWidth);
  const xTicks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += xStep) xTicks.push(v);

  // bars touch their neighbours through a 2px surface gap and never exceed 24px
  const barW = Math.max(1, Math.min(24, bw - 2));
  const bar = (i: number) => {
    const x0 = m.l + i * bw + (bw - barW) / 2, w = barW, y0 = y(counts[i]), h = m.t + ih - y0;
    const r = Math.min(4, w / 2, h);
    // rounded top, square baseline
    return `M${x0},${m.t + ih} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + w - r} Q${x0 + w},${y0} ${x0 + w},${y0 + r} V${m.t + ih} Z`;
  };
  // reference lines; labels anchor away from the plot edges and merge when they would collide
  const refs = [
    ...(mean !== undefined ? [{ name: "mean", v: mean, dashed: true }] : []),
    ...(p85 !== undefined ? [{ name: "p85", v: p85, dashed: false }] : []),
  ];
  const anchorAt = (px: number): "start" | "middle" | "end" => (px > W - m.r - 50 ? "end" : px < m.l + 50 ? "start" : "middle");
  const refLabels = refs.length === 2 && Math.abs(x(refs[0].v) - x(refs[1].v)) < 90
    ? [{ text: refs.map((r) => `${r.name} ${r.v.toFixed(1)}`).join(" · "), x: (x(refs[0].v) + x(refs[1].v)) / 2 }]
    : refs.map((r) => ({ text: `${r.name} ${r.v.toFixed(1)}`, x: x(r.v) }));
  const labelsPlaced = refLabels.map((l) => ({ ...l, anchor: anchorAt(l.x) }));
  const range = (i: number) => `${lo + i * binWidth}–${lo + (i + 1) * binWidth} km/h`;

  return (
    <div className="hist">
      <div className="hist-plot">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Histogram of valid vehicle speeds">
        {yTicks.map((c) => (
          <g key={c}>
            {c > 0 && <line x1={m.l} x2={W - m.r} y1={y(c)} y2={y(c)} className="grid" />}
            <text x={m.l - 6} y={y(c) + 4} className="axis" textAnchor="end">{c}</text>
          </g>
        ))}
        {counts.map((c, i) => c > 0 && (
          <path key={i} d={bar(i)} className={hover === null ? "bar" : hover === i ? "bar hover" : "bar dim"} />
        ))}
        <line x1={m.l} x2={W - m.r} y1={m.t + ih + .5} y2={m.t + ih + .5} className="baseline" />
        {counts.map((c, i) => (
          <rect key={`h${i}`} x={m.l + i * bw} y={m.t} width={bw} height={ih} fill="transparent" className="hit"
            tabIndex={c > 0 ? 0 : -1} aria-label={`${c} vehicles at ${range(i)}`}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(i)} onBlur={() => setHover(null)} />
        ))}
        {xTicks.map((v) => <text key={v} x={x(v)} y={H - 8} className="axis" textAnchor="middle">{v}</text>)}
        {refs.map((r) => (
          <line key={r.name} x1={x(r.v)} x2={x(r.v)} y1={m.t} y2={m.t + ih} className={r.dashed ? "ref dashed" : "ref"} pointerEvents="none" />
        ))}
        {labelsPlaced.map((l) => (
          <text key={l.text} x={l.x} y={m.t - 6} className="ref-label" textAnchor={l.anchor} pointerEvents="none">{l.text}</text>
        ))}
      </svg>
      {hover !== null && (
        <div className="hist-tip" role="status"
          style={{ left: `${((m.l + (hover + 0.5) * bw) / W) * 100}%`, top: `${(Math.max(m.t, y(counts[hover])) / H) * 100}%` }}>
          <b>{counts[hover]}</b>{counts[hover] === 1 ? "vehicle" : "vehicles"} · {range(hover)}
        </div>
      )}
      </div>
      <div className="hist-foot small muted">km/h · {binWidth} km/h bins · dashed line = mean, solid = p85</div>
    </div>
  );
}
