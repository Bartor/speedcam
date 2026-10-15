import { useCallback, useEffect, useRef, useState } from "react";
import { Calibration, Pt } from "../api";

export type Tool = "move" | "polygon" | "rect" | "segment" | "lineA" | "lineB" | "roi" | "point";

/** Tools that collect an open-ended list of corners, closed with Enter / double-click / first point. */
const MIN_CORNERS: Partial<Record<Tool, number>> = { roi: 3, polygon: 4 };

/** Four control points with world coords (0,0) (W,0) (W,L) (0,L): a rectangle calibration. */
export function rectDims(cal: Calibration): { w: number; l: number } | null {
  const p = cal.points;
  if (p.length !== 4) return null;
  const [a, b, c, d] = p.map((x) => x.world);
  if (a[0] !== 0 || a[1] !== 0 || b[1] !== 0 || c[0] !== b[0] || d[0] !== 0 || d[1] !== c[1]) return null;
  return { w: b[0], l: c[1] };
}

type Handle =
  | { kind: "seg"; i: number; end: "a" | "b" }
  | { kind: "line"; which: "line_a" | "line_b"; end: 0 | 1 }
  | { kind: "roi"; i: number }
  | { kind: "poly"; i: number }
  | { kind: "pt"; i: number };

interface Props {
  src: string;
  cal: Calibration;
  tool: Tool;
  onChange: (cal: Calibration) => void;
  onSegmentAdded?: (index: number) => void;
  onRectAdded?: () => void;
  onPolygonAdded?: () => void;
  lineGapM?: number | null;
  selected?: { kind: string; i: number } | null;
}

const COLORS = { poly: "#b98cff", seg: "#f5c518", a: "#00c8ff", b: "#ff3cc8", roi: "#4cd97b", pt: "#ff8c3a", pending: "#ff5050" };
const NEEDED: Partial<Record<Tool, number>> = { rect: 4, segment: 2, lineA: 2, lineB: 2, point: 1 };

export default function FrameCanvas({ src, cal, tool, onChange, onSegmentAdded, onRectAdded, onPolygonAdded, lineGapM, selected }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const view = useRef({ zoom: 1, ox: 0, oy: 0 });
  const [, force] = useState(0);
  const redraw = useCallback(() => force((n) => n + 1), []);
  const [loaded, setLoaded] = useState(false);
  const [pending, setPending] = useState<Pt[]>([]);
  const mouse = useRef<{ sx: number; sy: number; inside: boolean }>({ sx: 0, sy: 0, inside: false });
  const drag = useRef<{ type: "pan"; sx: number; sy: number; ox: number; oy: number } | { type: "handle"; h: Handle } | null>(null);

  const fit = useCallback(() => {
    const img = imgRef.current, wrap = wrapRef.current;
    if (!img || !wrap) return;
    const z = Math.min(wrap.clientWidth / img.naturalWidth, wrap.clientHeight / img.naturalHeight);
    view.current = { zoom: z, ox: (wrap.clientWidth - img.naturalWidth * z) / 2, oy: (wrap.clientHeight - img.naturalHeight * z) / 2 };
    redraw();
  }, [redraw]);

  useEffect(() => {
    setLoaded(false);
    const img = new Image();
    img.onload = () => {
      const first = !imgRef.current;
      imgRef.current = img;
      setLoaded(true);
      if (first) fit(); else redraw();
    };
    img.src = src;
  }, [src, fit, redraw]);

  useEffect(() => setPending([]), [tool]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => redraw());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [redraw]);

  // image px (OpenCV convention: pixel centres at integers) <-> screen px
  const toScreen = (p: Pt): Pt => {
    const v = view.current;
    return [(p[0] + 0.5) * v.zoom + v.ox, (p[1] + 0.5) * v.zoom + v.oy];
  };
  const toImage = (sx: number, sy: number): Pt => {
    const v = view.current;
    return [(sx - v.ox) / v.zoom - 0.5, (sy - v.oy) / v.zoom - 0.5];
  };

  const handles = (): { h: Handle; p: Pt }[] => {
    const out: { h: Handle; p: Pt }[] = [];
    cal.segments.forEach((s, i) => { out.push({ h: { kind: "seg", i, end: "a" }, p: s.a }, { h: { kind: "seg", i, end: "b" }, p: s.b }); });
    (["line_a", "line_b"] as const).forEach((w) => {
      const l = cal[w];
      if (l) out.push({ h: { kind: "line", which: w, end: 0 }, p: l[0] }, { h: { kind: "line", which: w, end: 1 }, p: l[1] });
    });
    cal.roi?.forEach((p, i) => out.push({ h: { kind: "roi", i }, p }));
    cal.polygon?.points.forEach((p, i) => out.push({ h: { kind: "poly", i }, p }));
    cal.points.forEach((c, i) => out.push({ h: { kind: "pt", i }, p: c.px }));
    return out;
  };

  const setHandle = (h: Handle, p: Pt): Calibration => {
    const c: Calibration = JSON.parse(JSON.stringify(cal));
    if (h.kind === "seg") c.segments[h.i][h.end] = p;
    else if (h.kind === "line") c[h.which]![h.end] = p;
    else if (h.kind === "roi") c.roi![h.i] = p;
    else if (h.kind === "poly") c.polygon!.points[h.i] = p;
    else c.points[h.i].px = p;
    return c;
  };

  const hit = (sx: number, sy: number) => {
    let best: { h: Handle; d: number } | null = null;
    for (const { h, p } of handles()) {
      const [x, y] = toScreen(p);
      const d = Math.hypot(x - sx, y - sy);
      if (d < 9 && (!best || d < best.d)) best = { h, d };
    }
    return best?.h ?? null;
  };

  const commit = (pts: Pt[]) => {
    const c: Calibration = JSON.parse(JSON.stringify(cal));
    if (tool === "rect") {
      c.points = pts.map((px) => ({ px, world: [0, 0] as [number, number] }));
      onChange(c);
      onRectAdded?.();
    } else if (tool === "segment") {
      c.segments.push({ a: pts[0], b: pts[1], meters: 0 });
      onChange(c);
      onSegmentAdded?.(c.segments.length - 1);
    } else if (tool === "lineA") { c.line_a = [pts[0], pts[1]]; onChange(c); }
    else if (tool === "lineB") { c.line_b = [pts[0], pts[1]]; onChange(c); }
    else if (tool === "point") { c.points.push({ px: pts[0], world: [0, 0] }); onChange(c); }
    else if (tool === "roi" || tool === "polygon") {
      // a double-click adds two near-identical points before closing: drop them
      const clean = pts.filter((p, i) => i === 0 || Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 2);
      if (clean.length >= MIN_CORNERS[tool]!) {
        if (tool === "roi") c.roi = clean;
        else c.polygon = { points: clean, distances: clean.map((_, i) => ({ i, j: (i + 1) % clean.length, meters: 0 })) };
        onChange(c);
        if (tool === "polygon") onPolygonAdded?.();
      }
    }
    setPending([]);
  };

  const local = (e: React.MouseEvent | React.WheelEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const [sx, sy] = local(e);
    const h = e.button === 0 ? hit(sx, sy) : null;
    if (e.button === 1 || e.button === 2 || (e.button === 0 && tool === "move" && !h)) {
      drag.current = { type: "pan", sx, sy, ox: view.current.ox, oy: view.current.oy };
      return;
    }
    if (e.button !== 0) return;
    const corners = MIN_CORNERS[tool];
    if (h && (tool === "move" || pending.length === 0) && !corners) {
      drag.current = { type: "handle", h };
      return;
    }
    const p = toImage(sx, sy);
    if (corners) {
      if (pending.length >= corners) {
        const [fx, fy] = toScreen(pending[0]);
        if (Math.hypot(fx - sx, fy - sy) < 10) return commit(pending);
      }
      setPending([...pending, p]);
      return;
    }
    const next = [...pending, p];
    if (next.length >= (NEEDED[tool] ?? 99)) commit(next);
    else setPending(next);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const [sx, sy] = local(e);
    mouse.current = { sx, sy, inside: true };
    const d = drag.current;
    if (d?.type === "pan") {
      view.current.ox = d.ox + sx - d.sx;
      view.current.oy = d.oy + sy - d.sy;
    } else if (d?.type === "handle") {
      onChange(setHandle(d.h, toImage(sx, sy)));
    }
    redraw();
  };

  const onWheel = (e: React.WheelEvent) => {
    const [sx, sy] = local(e);
    const v = view.current;
    const f = Math.exp(-e.deltaY * 0.0015);
    const z = Math.min(64, Math.max(0.02, v.zoom * f));
    v.ox = sx - ((sx - v.ox) * z) / v.zoom;
    v.oy = sy - ((sy - v.oy) * z) / v.zoom;
    v.zoom = z;
    redraw();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "Escape") setPending([]);
      if (e.key === "Enter" && MIN_CORNERS[tool] && pending.length >= MIN_CORNERS[tool]!) commit(pending);
      if (e.key === "f") fit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ---- drawing ----
  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current, img = imgRef.current;
    if (!canvas || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);
    if (!img) return;
    const v = view.current;
    ctx.imageSmoothingEnabled = v.zoom < 2;
    ctx.drawImage(img, v.ox, v.oy, img.naturalWidth * v.zoom, img.naturalHeight * v.zoom);

    const line = (a: Pt, b: Pt, color: string, w = 2, dash: number[] = []) => {
      const [x0, y0] = toScreen(a), [x1, y1] = toScreen(b);
      ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
    };
    const dot = (p: Pt, color: string, r = 4) => {
      const [x, y] = toScreen(p);
      ctx.fillStyle = color; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    };
    const label = (p: Pt, text: string, color: string) => {
      const [x, y] = toScreen(p);
      ctx.font = "600 12px system-ui, sans-serif";
      const w = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(0,0,0,.75)"; ctx.fillRect(x + 6, y - 20, w + 8, 18);
      ctx.fillStyle = color; ctx.fillText(text, x + 10, y - 7);
    };

    if (cal.roi && cal.roi.length >= 3) {
      ctx.beginPath();
      cal.roi.forEach((p, i) => { const [x, y] = toScreen(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fillStyle = "rgba(76,217,123,.08)"; ctx.fill();
      ctx.strokeStyle = COLORS.roi; ctx.lineWidth = 1.5; ctx.stroke();
      cal.roi.forEach((p) => dot(p, COLORS.roi, 3));
    }
    cal.segments.forEach((s, i) => {
      const sel = selected?.kind === "seg" && selected.i === i;
      line(s.a, s.b, COLORS.seg, sel ? 3 : 2);
      dot(s.a, COLORS.seg); dot(s.b, COLORS.seg);
      label([(s.a[0] + s.b[0]) / 2, (s.a[1] + s.b[1]) / 2], s.meters > 0 ? `${s.meters} m` : `S${i + 1}: enter m`, COLORS.seg);
    });
    if (cal.polygon && cal.polygon.points.length) {
      const P = cal.polygon.points;
      const n = P.length;
      ctx.beginPath();
      P.forEach((p, i) => { const [x, y] = toScreen(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fillStyle = "rgba(180,120,255,.10)"; ctx.fill();
      cal.polygon.distances.forEach((d) => {
        if (!P[d.i] || !P[d.j]) return;
        const side = Math.abs(d.i - d.j) === 1 || Math.abs(d.i - d.j) === n - 1;
        line(P[d.i], P[d.j], COLORS.poly, side ? 2 : 1.5, side ? [] : [6, 4]);
        const mid: Pt = [(P[d.i][0] + P[d.j][0]) / 2, (P[d.i][1] + P[d.j][1]) / 2];
        label(mid, d.meters > 0 ? `${d.meters} m` : `${d.i + 1}–${d.j + 1}: m?`, COLORS.poly);
      });
      P.forEach((p, i) => { dot(p, COLORS.poly, 5); label(p, `${i + 1}`, COLORS.poly); });
    }
    const rd = rectDims(cal);
    if (rd) {
      ctx.beginPath();
      cal.points.forEach((c, i) => { const [x, y] = toScreen(c.px); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fillStyle = "rgba(255,140,58,.10)"; ctx.fill();
      ctx.strokeStyle = COLORS.pt; ctx.lineWidth = 2; ctx.stroke();
      const mid = (i: number, j: number): Pt => [(cal.points[i].px[0] + cal.points[j].px[0]) / 2, (cal.points[i].px[1] + cal.points[j].px[1]) / 2];
      label(mid(0, 1), rd.w > 0 ? `${rd.w} m` : "edge 1→2: enter m", COLORS.pt);
      label(mid(1, 2), rd.l > 0 ? `${rd.l} m` : "edge 2→3: enter m", COLORS.pt);
      cal.points.forEach((c, i) => { dot(c.px, COLORS.pt, 5); label(c.px, `${i + 1}`, COLORS.pt); });
    } else {
      cal.points.forEach((c, i) => { dot(c.px, COLORS.pt, 5); label(c.px, `P${i + 1} (${c.world[0]}, ${c.world[1]})`, COLORS.pt); });
    }
    ([["line_a", "A", COLORS.a], ["line_b", "B", COLORS.b]] as const).forEach(([k, name, color]) => {
      const l = cal[k];
      if (!l) return;
      line(l[0], l[1], color, 2.5);
      dot(l[0], color); dot(l[1], color);
      label(l[0], name === "A" && lineGapM ? `A · gap ${lineGapM.toFixed(2)} m` : name, color);
    });

    // in-progress clicks + rubber band
    const m = mouse.current;
    const cursor = toImage(m.sx, m.sy);
    if (pending.length) {
      const color = COLORS.pending;
      pending.forEach((p, i) => { dot(p, color); if (i) line(pending[i - 1], p, color, 1.5); });
      if (m.inside) line(pending[pending.length - 1], cursor, color, 1, [5, 4]);
      if ((MIN_CORNERS[tool] || tool === "rect") && pending.length >= 2 && m.inside) line(cursor, pending[0], color, 1, [2, 4]);
    }

    // loupe for precise placement
    if (tool !== "move" && m.inside && !drag.current) {
      const R = 80, mag = Math.max(v.zoom * 4, 3);
      const src = (R / mag);
      const cx = cursor[0] + 0.5, cy = cursor[1] + 0.5;
      const lx = m.sx > W - 2 * R - 40 && m.sy < 2 * R + 40 ? 12 : W - 2 * R - 12, ly = 12;
      ctx.save();
      ctx.beginPath(); ctx.rect(lx, ly, 2 * R, 2 * R); ctx.clip();
      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = "#000"; ctx.fillRect(lx, ly, 2 * R, 2 * R);
      ctx.drawImage(img, cx - src, cy - src, 2 * src, 2 * src, lx, ly, 2 * R, 2 * R);
      ctx.strokeStyle = "rgba(255,80,80,.9)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(lx + R, ly); ctx.lineTo(lx + R, ly + 2 * R); ctx.moveTo(lx, ly + R); ctx.lineTo(lx + 2 * R, ly + R); ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = "#888"; ctx.strokeRect(lx, ly, 2 * R, 2 * R);
      ctx.fillStyle = "#ddd"; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`${cursor[0].toFixed(1)}, ${cursor[1].toFixed(1)} px`, lx + 4, ly + 2 * R + 14);
    }
  });

  return (
    <div ref={wrapRef} className="frame-canvas" onContextMenu={(e) => e.preventDefault()}>
      <canvas ref={canvasRef}
        style={{ cursor: tool === "move" ? (drag.current ? "grabbing" : "grab") : "crosshair" }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={() => { drag.current = null; redraw(); }}
        onMouseLeave={() => { drag.current = null; mouse.current.inside = false; redraw(); }}
        onDoubleClick={() => MIN_CORNERS[tool] && pending.length >= MIN_CORNERS[tool]! && commit(pending)}
        onWheel={onWheel} />
      {!loaded && <div className="canvas-loading">Decoding full-resolution frame…</div>}
      <button className="fit-btn" onClick={fit} title="Fit (F)">Fit</button>
    </div>
  );
}
