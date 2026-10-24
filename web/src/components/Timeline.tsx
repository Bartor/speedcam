import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Project, rotSize, ScanResult } from "../api";
import { useActiveJobs } from "./JobsIndicator";
import { typing } from "./RotatedVideo";

/** Every shortcut on a screen with a timeline (the player's and the timeline's own). */
export const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["K / L", "Pause / play (L again: faster)"],
  ["← / →", "Previous / next frame"],
  ["Shift + ← / →", "One second back / forward"],
  ["Home / End", "Start / end of the clip"],
  ["Wheel", "Scroll along the timeline (when zoomed in)"],
  ["Alt or Ctrl + wheel", "Zoom in / out at the cursor"],
  ["Shift + wheel", "Move the playhead frame by frame"],
  ["+ / −", "Zoom in / out at the playhead"],
  ["\\", "Fit the whole clip"],
  ["I / O", "In point / out point: adds that range (to analyse, or on Export to render)"],
  ["Drag on the parts lane", "Cut out (starting over blue) or add back (elsewhere)"],
  ["?", "Show these shortcuts"],
];

export interface TimelineVehicle { id: number; f0: number; f1: number; valid: boolean; label?: string }

interface Props {
  project: Project;
  frame: number; // playhead, in video frames (proxy frame n == source frame n)
  onSeek: (frame: number) => void;
  onProjectChange?: (p: Project) => void; // parts edits are saved and reported here
  editable?: boolean; // the parts lane can be edited
  vehicles?: TimelineVehicle[];
  selectedVehicle?: number | null;
  onPickVehicle?: (id: number) => void;
  /** the parts lane edits these ranges instead of the analysed parts (Export: what to render) */
  ranges?: Range[] | null;
  onRanges?: (r: Range[] | null) => void;
  verb?: string; // what the parts are for: "Analyse", "Render"
}

export type Range = [number, number];

// lane layout (px)
const RULER = 18, FILM = 46, ACT = 26, PARTS = 20, VEH = 16, GAP = 2;
const TICKS_S = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

const clock = (s: number, fine = false) => {
  const neg = s < 0 ? "-" : "";
  s = Math.abs(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  const sec = fine ? x.toFixed(1).padStart(4, "0") : String(Math.floor(x)).padStart(2, "0");
  return h ? `${neg}${h}:${String(m).padStart(2, "0")}:${sec}` : `${neg}${m}:${sec}`;
};

/** Sorted, merged, clipped to [0, n). */
function norm(rs: Range[], n: number): Range[] {
  const s = rs.map(([a, b]) => [Math.max(0, Math.min(a, b)), Math.min(n, Math.max(a, b))] as Range)
    .filter(([a, b]) => b > a).sort((p, q) => p[0] - q[0]);
  const out: Range[] = [];
  for (const r of s) {
    const last = out[out.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else out.push([r[0], r[1]]);
  }
  return out;
}
const add = (rs: Range[], r: Range, n: number) => norm([...rs, r], n);
function cut(rs: Range[], [a, b]: Range, n: number): Range[] {
  const out: Range[] = [];
  for (const [x, y] of rs) {
    if (y <= a || x >= b) out.push([x, y]);
    else {
      if (x < a) out.push([x, a]);
      if (y > b) out.push([b, y]);
    }
  }
  return norm(out, n);
}
const inside = (rs: Range[], f: number) => rs.some(([a, b]) => f >= a && f < b);

/** A Premiere-like timeline of the clip: ruler, frame thumbnails, the activity scan, the parts
 * to analyse (drag to add or cut out, drag edges, I / O keys) and, on the results, the vehicles.
 * Wheel zooms around the cursor, Shift+wheel pans. Frames are video frames; times are real
 * seconds (frame / capture fps). */
export default function Timeline({ project, frame, onSeek, onProjectChange, editable = true, vehicles, selectedVehicle, onPickVehicle,
  ranges, onRanges, verb = "Analyse" }: Props) {
  const nb = Math.max(1, project.info.nb_frames);
  const fps = project.config.capture_fps || project.info.container_fps;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [W, setW] = useState(800);
  const [view, setView] = useState<[number, number]>([0, nb]);
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [sprite, setSprite] = useState<{ img: HTMLImageElement; count: number; tw: number } | null>(null);
  const [hover, setHover] = useState<{ x: number; f: number } | null>(null);
  const [inPoint, setInPoint] = useState<number | null>(null);
  const saved = onRanges ? ranges ?? null : project.config.parts ?? null;
  const [parts, setParts] = useState<Range[]>(saved ? norm(saved as Range[], nb) : [[0, nb]]);
  const drag = useRef<null | { kind: "scrub" } | { kind: "paint"; a: number; b: number; mode: "add" | "cut" } | { kind: "edge"; i: number; side: 0 | 1 }>(null);
  const [, redraw] = useState(0);
  const tick = () => redraw((n) => n + 1);

  // keep in step with the saved parts (another screen, the activity panel)
  const savedKey = JSON.stringify(saved);
  useEffect(() => { if (!drag.current) setParts(saved ? norm(saved as Range[], nb) : [[0, nb]]); }, [savedKey, nb]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // scan: load, and reload when a scan job of this project finishes
  const { jobs } = useActiveJobs();
  const scanning = jobs.some((j) => j.project_id === project.id && j.kind === "scan");
  useEffect(() => { if (!scanning) api.getScan(project.id).then(setScan).catch(() => setScan(null)); }, [project.id, scanning]);

  // filmstrip: enough thumbnails to fill the width at the widest zoom
  const [vw, vh] = rotSize(project.info.width, project.info.height, project.config.rotation ?? 0);
  const tw = Math.max(16, Math.round((FILM * vw) / vh));
  const count = Math.max(8, Math.min(160, Math.ceil(W / tw) * 2));
  useEffect(() => {
    if (!project.status.proxy_mp4) return;
    const img = new Image();
    img.onload = () => setSprite({ img, count, tw: img.naturalWidth / count });
    img.src = `/api/projects/${project.id}/filmstrip.jpg?count=${count}&height=${FILM * 2}&r=${project.config.rotation ?? 0}`;
  }, [project.id, count, project.status.proxy_mp4, project.config.rotation]);

  const [v0, v1] = view;
  const x = useCallback((f: number) => ((f - v0) / (v1 - v0)) * W, [v0, v1, W]);
  const fAt = useCallback((px: number) => Math.round(Math.max(0, Math.min(nb - 1, v0 + (px / W) * (v1 - v0)))), [v0, v1, W, nb]);
  const lanes = useMemo(() => {
    let y = 0;
    const ruler = [y, RULER]; y += RULER + GAP;
    const film = [y, FILM]; y += FILM + GAP;
    const act = [y, ACT]; y += ACT + GAP;
    const prt = [y, PARTS]; y += PARTS + GAP;
    const veh = vehicles ? [y, VEH] : null; if (veh) y += VEH + GAP;
    return { ruler, film, act, parts: prt, veh, H: y };
  }, [vehicles]);

  const save = useRef<number | undefined>(undefined);
  const commit = (next: Range[]) => {
    const n = norm(next, nb);
    if (!n.length) { tick(); return; }  // cutting out everything would leave nothing to analyse
    setParts(n);
    if (onRanges) {
      onRanges(n.length === 1 && n[0][0] <= 0 && n[0][1] >= nb ? null : n);
      return;
    }
    window.clearTimeout(save.current);
    save.current = window.setTimeout(async () => {
      const all = n.length === 1 && n[0][0] <= 0 && n[0][1] >= nb;
      try { const p = await api.patch(project.id, { parts: all ? null : n }); onProjectChange?.(p); } catch { /* shown on reload */ }
    }, 300);
  };

  // ---- drawing
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const H = lanes.H;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; c.style.width = `${W}px`; c.style.height = `${H}px`; }
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const css = getComputedStyle(c);
    const accent = css.getPropertyValue("--accent").trim() || "#3987e5";
    const lane = (y: number, h: number) => { g.fillStyle = "rgba(255,255,255,.035)"; g.fillRect(0, y, W, h); };

    // ruler
    const [ry, rh] = lanes.ruler;
    const secPerPx = (v1 - v0) / fps / W;
    const step = TICKS_S.find((s) => s / secPerPx >= 80) ?? 3600;
    g.fillStyle = "rgba(255,255,255,.5)"; g.font = "10px system-ui, sans-serif"; g.textBaseline = "top";
    for (let s = Math.ceil((v0 / fps) / step) * step; s * fps <= v1; s += step) {
      const px = x(s * fps);
      g.fillRect(px, ry + rh - 6, 1, 6);
      g.fillText(clock(s, step < 1), px + 3, ry + 1);
      const minor = step / 5;
      for (let k = 1; k < 5; k++) { const mx = x((s + k * minor) * fps); if (mx < W) g.fillRect(mx, ry + rh - 3, 1, 3); }
    }

    // filmstrip
    const [fy, fh] = lanes.film;
    lane(fy, fh);
    if (sprite) {
      const slot = Math.max(8, (sprite.tw * fh) / sprite.img.naturalHeight);
      for (let px = 0; px < W; px += slot) {
        const f = fAt(px + slot / 2);
        const k = Math.round((f / Math.max(1, nb - 1)) * (sprite.count - 1));
        g.drawImage(sprite.img, k * sprite.tw, 0, sprite.tw, sprite.img.naturalHeight, px, fy, slot - 1, fh);
      }
    }

    // activity
    const [ay, ah] = lanes.act;
    lane(ay, ah);
    const sc = scan?.scan;
    if (sc && sc.frames.length) {
      const peak = Math.max(1, ...sc.counts);
      const bw = Math.max(1.5, x(sc.frames[1] ?? 1) - x(sc.frames[0] ?? 0) - 1);
      const bars = (vals: number[], color: string) => {
        g.fillStyle = color;
        sc.frames.forEach((f, k) => {
          const v = vals[k];
          if (!v) return;
          const px = x(f);
          if (px < -bw || px > W + bw) return;
          const h = 2 + (ah - 3) * Math.min(1, v / peak);
          g.fillRect(px - bw / 2, ay + ah - h, bw, h);
        });
      };
      bars(sc.counts, "rgba(233,238,245,.22)");
      bars(sc.moving ?? sc.counts, "rgba(233,238,245,.85)");
    } else {
      g.fillStyle = "rgba(255,255,255,.35)"; g.font = "11px system-ui, sans-serif"; g.textBaseline = "middle";
      g.fillText(scanning ? "Scanning for vehicles…" : "Activity: not scanned yet (Scan for vehicles, above)", 8, ay + ah / 2);
    }

    // parts: analysed (accent) vs cut out (hatched); cut-out time is also dimmed above
    const [py, ph] = lanes.parts;
    let shown = parts;
    const d = drag.current;
    if (d?.kind === "paint") shown = d.mode === "add" ? add(parts, [d.a, d.b], nb) : cut(parts, [Math.min(d.a, d.b), Math.max(d.a, d.b)], nb);
    g.fillStyle = "rgba(0,0,0,.35)"; g.fillRect(0, py, W, ph);
    g.save(); g.beginPath(); g.rect(0, py, W, ph); g.clip();
    g.strokeStyle = "rgba(255,255,255,.08)";
    for (let k = -ph; k < W; k += 7) { g.beginPath(); g.moveTo(k, py + ph); g.lineTo(k + ph, py); g.stroke(); }
    g.restore();
    const allIn = shown.length === 1 && shown[0][0] <= 0 && shown[0][1] >= nb;
    shown.forEach(([a, b]) => {
      const xa = Math.max(0, x(a)), xb = Math.min(W, x(b));
      if (xb <= xa) return;
      g.fillStyle = accent; g.globalAlpha = allIn ? 0.45 : 0.85; g.fillRect(xa, py + 2, xb - xa, ph - 4); g.globalAlpha = 1;
      if (!allIn) { g.fillStyle = "#fff"; g.fillRect(xa, py + 2, 2, ph - 4); g.fillRect(xb - 2, py + 2, 2, ph - 4); }
    });
    // dim the cut-out time over the picture lanes
    const gaps: Range[] = [];
    let prev = 0;
    for (const [a, b] of shown) { if (a > prev) gaps.push([prev, a]); prev = Math.max(prev, b); }
    if (prev < nb) gaps.push([prev, nb]);
    g.fillStyle = "rgba(0,0,0,.55)";
    gaps.forEach(([a, b]) => {
      const xa = Math.max(0, x(a)), xb = Math.min(W, x(b));
      if (xb > xa) g.fillRect(xa, lanes.film[0], xb - xa, lanes.act[0] + ACT - lanes.film[0]);
    });
    g.fillStyle = "rgba(255,255,255,.8)"; g.font = "600 10px system-ui, sans-serif"; g.textBaseline = "middle";
    if (allIn) g.fillText(`${verb}: whole clip`, 6, py + ph / 2);

    // vehicles
    if (lanes.veh && vehicles) {
      const [vy, vhh] = lanes.veh;
      lane(vy, vhh);
      vehicles.forEach((v) => {
        const xa = x(v.f0), xb = Math.max(xa + 2, x(v.f1));
        if (xb < 0 || xa > W) return;
        g.fillStyle = v.id === selectedVehicle ? "#ffd166" : v.valid ? "rgba(46,204,113,.8)" : "rgba(200,0,200,.7)";
        g.fillRect(xa, vy + 3, xb - xa, vhh - 6);
      });
    }

    // in point and playhead
    if (inPoint !== null) {
      g.fillStyle = "#ffd166"; const ix = x(inPoint);
      g.fillRect(ix, 0, 2, H);
      g.fillText("I", ix + 4, ry + rh / 2);
    }
    const hx = x(frame);
    g.fillStyle = "#ff4d4f"; g.fillRect(hx - 1, 0, 2, H);
    g.beginPath(); g.moveTo(hx - 5, 0); g.lineTo(hx + 5, 0); g.lineTo(hx, 7); g.closePath(); g.fill();
  });

  // ---- interaction
  const laneAt = (y: number) => {
    const inL = (l: number[] | null) => !!l && y >= l[0] && y < l[0] + l[1];
    if (inL(lanes.parts)) return "parts";
    if (inL(lanes.veh)) return "veh";
    return "scrub";
  };
  const local = (e: React.PointerEvent | PointerEvent | WheelEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const [px, py] = local(e);
    const f = fAt(px);
    const where = laneAt(py);
    try { (e.target as HTMLElement).setPointerCapture(e.pointerId); } catch { /* synthetic or already released */ }
    if (where === "parts" && editable) {
      const allIn = parts.length === 1 && parts[0][0] <= 0 && parts[0][1] >= nb;
      if (!allIn) {
        for (let i = 0; i < parts.length; i++) {
          for (const side of [0, 1] as const) if (Math.abs(x(parts[i][side]) - px) < 6) { drag.current = { kind: "edge", i, side }; return; }
        }
      }
      drag.current = { kind: "paint", a: f, b: f, mode: inside(parts, f) ? "cut" : "add" };
      tick();
      return;
    }
    if (where === "veh" && vehicles) {
      const v = vehicles.find((q) => px >= x(q.f0) - 3 && px <= Math.max(x(q.f0) + 2, x(q.f1)) + 3);
      if (v) { onPickVehicle?.(v.id); onSeek(v.f0); return; }
    }
    drag.current = { kind: "scrub" };
    onSeek(f);
  };
  const onMove = (e: React.PointerEvent) => {
    const [px] = local(e);
    const f = fAt(px);
    setHover({ x: px, f });
    const d = drag.current;
    if (!d) return;
    if (d.kind === "scrub") onSeek(f);
    else if (d.kind === "paint") { d.b = f; tick(); }
    else if (d.kind === "edge") {
      const next = parts.map((r) => [...r] as Range);
      next[d.i][d.side] = f;
      setParts(norm(next, nb));
    }
  };
  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === "paint") {
      if (Math.abs(x(d.b) - x(d.a)) < 3) { onSeek(d.a); tick(); return; }  // a click: just seek
      const r: Range = [Math.min(d.a, d.b), Math.max(d.a, d.b)];
      commit(d.mode === "add" ? add(parts, r, nb) : cut(parts, r, nb));
    } else if (d.kind === "edge") commit(parts);
  };

  // wheel, as in Premiere: scroll along the timeline; Alt or Ctrl zooms at the cursor; Shift
  // moves the playhead frame by frame. With the whole clip in view the wheel scrolls the page.
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const seekRef = useRef(onSeek);
  seekRef.current = onSeek;
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const onWheel = (e: WheelEvent) => {
      const [px] = local(e);
      const dy = e.deltaY || e.deltaX;  // Shift+wheel arrives as deltaX in some browsers
      if (e.shiftKey) {
        e.preventDefault();
        seekRef.current(Math.max(0, Math.min(nb - 1, frameRef.current + (dy > 0 ? 1 : -1))));
        return;
      }
      if (e.altKey || e.ctrlKey || e.metaKey) {
        e.preventDefault();  // also stops the browser's own Ctrl+wheel page zoom
        setView(([a, b]) => {
          const span = b - a;
          const z = Math.exp(e.deltaY * 0.0015);
          const ns = Math.max(Math.min(nb, fps * 2), Math.min(nb, span * z));
          const anchor = a + (px / W) * span;
          const na = Math.max(0, Math.min(nb - ns, anchor - ((anchor - a) / span) * ns));
          return [na, na + ns];
        });
        return;
      }
      setView(([a, b]) => {
        const span = b - a;
        if (span >= nb - 0.5) return [a, b];  // nothing to scroll: leave the wheel to the page
        e.preventDefault();
        const d = ((Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / W) * span;
        const na = Math.max(0, Math.min(nb - span, a + d));
        return [na, na + span];
      });
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [W, nb, fps]);

  // keep the playhead in view while playing
  useEffect(() => {
    const span = v1 - v0;
    if (frame < v0 || frame > v1) {
      const na = Math.max(0, Math.min(nb - span, frame - span * 0.1));
      setView([na, na + span]);
    }
  }, [frame]);  // eslint-disable-line react-hooks/exhaustive-deps

  const zoom = (k: number) => setView(([a, b]) => {
    const span = Math.max(Math.min(nb, fps * 2), Math.min(nb, (b - a) * k));
    const c = Math.max(span / 2, Math.min(nb - span / 2, frameRef.current));
    return [c - span / 2, c + span / 2];
  });
  const [showKeys, setShowKeys] = useState(false);
  // I / O, zoom, fit and ? anywhere on the page (not while typing); Space, arrows, Home/End, K/L
  // belong to the player controls
  const keyState = useRef({ frame, inPoint, parts, editable });
  keyState.current = { frame, inPoint, parts, editable };
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      const s = keyState.current;
      if (e.key === "?") { setShowKeys((v) => !v); e.preventDefault(); }
      else if (e.key === "Escape") setShowKeys(false);
      else if (e.key === "+" || e.key === "=") zoom(0.5);
      else if (e.key === "-" || e.key === "_") zoom(2);
      else if (e.key === "\\") setView([0, nb]);
      else if (s.editable && (e.key === "i" || e.key === "I")) setInPoint(s.frame);
      else if (s.editable && (e.key === "o" || e.key === "O") && s.inPoint !== null) {
        const r: Range = [Math.min(s.inPoint, s.frame), Math.max(s.inPoint, s.frame) + 1];
        const allIn = s.parts.length === 1 && s.parts[0][0] <= 0 && s.parts[0][1] >= nb;
        commit(allIn ? [r] : add(s.parts, r, nb));  // the first in/out range replaces "whole clip"
        setInPoint(null);
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [nb]);  // eslint-disable-line react-hooks/exhaustive-deps

  const suggested = scan?.suggested ?? [];
  const allIn = parts.length === 1 && parts[0][0] <= 0 && parts[0][1] >= nb;
  const covered = parts.reduce((s, [a, b]) => s + b - a, 0);

  // hover preview (from the filmstrip)
  let preview: React.ReactNode = null;
  if (hover && sprite) {
    const k = Math.round((hover.f / Math.max(1, nb - 1)) * (sprite.count - 1));
    const ph = 90, pw = Math.round((sprite.tw * ph) / sprite.img.naturalHeight);
    const left = Math.max(0, Math.min(W - pw, hover.x - pw / 2));
    preview = (
      <div className="tl-preview" style={{ left, width: pw }}>
        <div style={{ width: pw, height: ph, backgroundImage: `url(${sprite.img.src})`, backgroundSize: `${pw * sprite.count}px ${ph}px`, backgroundPosition: `-${k * pw}px 0` }} />
        <span>{clock(hover.f / fps, true)} · frame {hover.f}</span>
      </div>
    );
  }

  return (
    <div className="timeline">
      <div className="tl-bar small">
        <span className="tl-time">{clock(frame / fps, true)}</span>
        <span className="muted">/ {clock(nb / fps)} · frame {frame}</span>
        <span className="grow" />
        {editable && <>
          <span className="muted" title="Drag on the bottom lane: over blue to cut out, elsewhere to add. Drag a part's edge to adjust. I / O set in and out points at the playhead.">
            {allIn ? `${verb}: the whole clip` : `${verb}: ${parts.length} part${parts.length === 1 ? "" : "s"}, ${clock(covered / fps)} of ${clock(nb / fps)}`}
          </span>
          {!onRanges && suggested.length > 0 && <button className="chip" onClick={() => commit(suggested as Range[])} title="The parts with moving vehicles found by the activity scan">Use activity</button>}
          {!allIn && <button className="chip" onClick={() => commit([[0, nb]])} title={`${verb} the whole clip`}>{verb} all</button>}
        </>}
        <button className="chip" onClick={() => zoom(0.5)} title="Zoom in (wheel)">+</button>
        <button className="chip" onClick={() => zoom(2)} title="Zoom out (wheel)">−</button>
        <button className="chip" onClick={() => setView([0, nb])} title="Show the whole clip (\)">Fit</button>
        <button className="chip" onClick={() => setShowKeys(!showKeys)} title="Keyboard and mouse shortcuts (?)">⌨ Shortcuts</button>
      </div>
      {showKeys && (
        <div className="tl-keys" role="dialog" aria-label="Shortcuts">
          {SHORTCUTS.map(([k, what]) => <div key={k}><kbd>{k}</kbd><span>{what}</span></div>)}
        </div>
      )}
      <div ref={wrap} className="tl-canvas" onPointerLeave={() => setHover(null)}
        aria-label="Timeline: click or drag to seek">
        <canvas ref={canvas} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} />
        {preview}
      </div>
    </div>
  );
}
