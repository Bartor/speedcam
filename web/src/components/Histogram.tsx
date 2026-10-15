import { useState } from "react";

interface Props { values: number[]; mean?: number; p85?: number; binWidth?: number }

/** Speed histogram: single series, recessive axes, mean/p85 reference lines, per-bar hover. */
export default function Histogram({ values, mean, p85, binWidth = 2 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  if (values.length === 0) return <div className="muted small">No valid measurements yet.</div>;
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

  const bar = (i: number) => {
    const x0 = m.l + i * bw + 1, w = Math.max(1, bw - 2), y0 = y(counts[i]), h = m.t + ih - y0;
    const r = Math.min(4, w / 2, h);
    // rounded top, square baseline
    return `M${x0},${m.t + ih} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + w - r} Q${x0 + w},${y0} ${x0 + w},${y0 + r} V${m.t + ih} Z`;
  };

  return (
    <div className="hist">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Histogram of valid vehicle speeds">
        {yTicks.map((c) => (
          <g key={c}>
            <line x1={m.l} x2={W - m.r} y1={y(c)} y2={y(c)} className="grid" />
            <text x={m.l - 6} y={y(c) + 4} className="axis" textAnchor="end">{c}</text>
          </g>
        ))}
        {counts.map((c, i) => c > 0 && (
          <path key={i} d={bar(i)} className={hover === i ? "bar hover" : "bar"} />
        ))}
        {counts.map((_, i) => (
          <rect key={`h${i}`} x={m.l + i * bw} y={m.t} width={bw} height={ih} fill="transparent"
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
        {xTicks.map((v) => <text key={v} x={x(v)} y={H - 8} className="axis" textAnchor="middle">{v}</text>)}
        {mean !== undefined && (
          <g pointerEvents="none">
            <line x1={x(mean)} x2={x(mean)} y1={m.t} y2={m.t + ih} className="ref dashed" />
            <text x={x(mean)} y={m.t - 6} className="ref-label" textAnchor="middle">mean {mean.toFixed(1)}</text>
          </g>
        )}
        {p85 !== undefined && (
          <g pointerEvents="none">
            <line x1={x(p85)} x2={x(p85)} y1={m.t} y2={m.t + ih} className="ref" />
            <text x={x(p85)} y={m.t - 6} className="ref-label" textAnchor={p85 > (lo + hi) / 2 ? "end" : "start"} dx={p85 > (lo + hi) / 2 ? -3 : 3}>p85 {p85.toFixed(1)}</text>
          </g>
        )}
      </svg>
      <div className="hist-foot small muted">
        {hover !== null
          ? <span><b className="text">{counts[hover]}</b> vehicles at {lo + hover * binWidth}–{lo + (hover + 1) * binWidth} km/h</span>
          : <span>km/h · {binWidth} km/h bins · hover a bar for its count</span>}
      </div>
    </div>
  );
}
