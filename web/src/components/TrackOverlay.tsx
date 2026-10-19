import { useEffect, useMemo, useRef } from "react";
import { PATH_COLORS, Pt } from "../api";

export interface OverlayData {
  width: number; height: number; capture_fps: number; container_fps: number;
  line_a: [Pt, Pt] | null; line_b: [Pt, Pt] | null; roi: Pt[] | null; line_gap_m: number | null;
  paths?: { name: string; measured_m: number; px: number[]; trusted_px: number[] | null; corridor_px: [number[], number[]] }[];
  speed_limit_kmh: number; trail_frames: number;
  T_frames: number[]; T_inv: number[][];
  det_frame: number[]; det_tid: number[]; det_corners: number[];
  tracks: Record<string, {
    f0: number; trail: number[]; prov: (number | null)[] | null; cls: string;
    t_first: number | null; t_second: number | null; speed: number | null; speed_max?: number | null;
    dt: number | null; valid: boolean | null;
  }>;
}

const LINE_A = "#00c8ff", LINE_B = "#ff3cc8", GREY = "#aaaaaa";

function speedColor(v: number, limit: number) {
  return v <= limit ? "#3cc83c" : v <= limit + 10 ? "#ffa500" : "#e62828";
}

/** Draws boxes, trails, lines and speed labels over a <video> showing the original framing (proxy). */
export default function TrackOverlay({ data, video, frame }: { data: OverlayData; video: HTMLVideoElement | null; frame: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const byFrame = useMemo(() => {
    const m = new Map<number, number[]>();
    data.det_frame.forEach((f, i) => { const a = m.get(f); if (a) a.push(i); else m.set(f, [i]); });
    return m;
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video) return;
    const W = video.clientWidth, H = video.clientHeight, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // letterboxed content rect of the video element (object-fit: contain)
    const s = Math.min(W / data.width, H / data.height);
    const ox = (W - data.width * s) / 2, oy = (H - data.height * s) / 2;

    // reference px -> frame px for this frame (nearest stored transform)
    let lo = 0, hi = data.T_frames.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (data.T_frames[mid] < frame) lo = mid + 1; else hi = mid; }
    const M = data.T_inv[lo] ?? [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const P = (x: number, y: number): Pt => {
      const w = M[6] * x + M[7] * y + M[8];
      return [((M[0] * x + M[1] * y + M[2]) / w) * s + ox, ((M[3] * x + M[4] * y + M[5]) / w) * s + oy];
    };
    const fs = Math.max(11, Math.round(13 * (H / 600)));
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const label = (x: number, y: number, text: string, color: string) => {
      const w = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(0,0,0,.75)"; ctx.fillRect(x - 3, y - fs - 3, w + 6, fs + 6);
      ctx.fillStyle = color; ctx.fillText(text, x, y);
    };

    if (data.roi && data.roi.length >= 3) {
      ctx.beginPath();
      data.roi.forEach(([x, y], i) => { const [a, b] = P(x, y); if (i) ctx.lineTo(a, b); else ctx.moveTo(a, b); });
      ctx.closePath(); ctx.strokeStyle = "rgba(160,160,160,.7)"; ctx.lineWidth = 1; ctx.stroke();
    }
    const many = (data.paths?.length ?? 0) > 1;
    data.paths?.forEach((pp, k) => {
      // flat [x, y, x, y, ...] in reference px
      const poly = (flat: number[], color: string, w: number, dash: number[] = []) => {
        ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath();
        for (let i = 0; i < flat.length; i += 2) { const p = P(flat[i], flat[i + 1]); if (i) ctx.lineTo(...p); else ctx.moveTo(...p); }
        ctx.stroke(); ctx.setLineDash([]);
      };
      const color = PATH_COLORS[k % PATH_COLORS.length];
      pp.corridor_px.forEach((e) => poly(e, "rgba(255,255,255,.45)", 1, [6, 5]));
      poly(pp.px, color, 1.5, [4, 6]);
      const m = pp.trusted_px;
      if (!m || m.length < 4) return;
      poly(m, color, 2.5);
      const a = P(m[0], m[1]), b = P(m[m.length - 2], m[m.length - 1]);
      label(a[0] + 6, a[1] - 6, many ? `A · ${pp.name}` : "A", color);
      label(b[0] + 6, b[1] - 6, `B  ${pp.measured_m.toFixed(1)} m`, color);
    });
    ([["A", data.line_a, LINE_A], ["B", data.line_b, LINE_B]] as const).forEach(([name, l, color]) => {
      if (!l || (data.paths?.length ?? 0) > 0) return;
      const [a, b] = [P(...l[0]), P(...l[1])];
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      label(a[0] + 6, a[1] - 6, name === "A" && data.line_gap_m ? `A  ${data.line_gap_m.toFixed(1)} m` : name, color);
    });

    const t = frame / data.capture_fps;
    for (const i of byFrame.get(frame) ?? []) {
      const tid = data.det_tid[i];
      const tr = data.tracks[String(tid)];
      const c = data.det_corners.slice(i * 8, i * 8 + 8);
      let color = "#ffffff", text = `#${tid}${tr ? ` ${tr.cls}` : ""}`;
      if (tr && tr.speed !== null && tr.t_first !== null && tr.t_second !== null) {
        if (t >= tr.t_second) {
          const vmax = tr.speed_max ?? tr.speed;
          color = tr.valid ? speedColor(vmax, data.speed_limit_kmh) : "#c800c8";
          text = `max ${vmax.toFixed(1)} · avg ${tr.speed.toFixed(1)} km/h${tr.valid ? "" : " ?"}`;
        } else if (t > tr.t_first) {
          const v = tr.prov?.[frame - tr.f0];
          color = GREY;
          text = v != null ? `${Math.round(v)} km/h` : `#${tid} …`;
        }
      }
      const pts = [0, 1, 2, 3].map((k) => P(c[2 * k], c[2 * k + 1]));
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); pts.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath(); ctx.stroke();
      if (tr) {
        const k = frame - tr.f0, from = Math.max(0, k - data.trail_frames);
        if (k > from && 2 * k + 1 < tr.trail.length) {
          ctx.beginPath();
          for (let j = from; j <= k; j++) {
            const p = P(tr.trail[2 * j], tr.trail[2 * j + 1]);
            if (j === from) ctx.moveTo(...p); else ctx.lineTo(...p);
          }
          ctx.stroke();
        }
      }
      // top-left of the bounding rectangle: picking the "highest corner" flips between the
      // two near-equal top corners from frame to frame and makes labels jump
      label(Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])) - 8, text, color);
    }
  });

  return <canvas ref={canvasRef} className="track-overlay" />;
}
