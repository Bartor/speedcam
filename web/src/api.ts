import { useEffect, useRef, useState } from "react";

export type Pt = [number, number];

export interface Segment { a: Pt; b: Pt; meters: number }
export interface ControlPoint { px: Pt; world: Pt }
export interface PolygonDistance { i: number; j: number; meters: number }
export interface Polygon { points: Pt[]; distances: PolygonDistance[] }
export interface Calibration {
  polygon?: Polygon | null;
  reference_frame: number;
  segments: Segment[];
  points: ControlPoint[];
  line_a: [Pt, Pt] | null;
  line_b: [Pt, Pt] | null;
  roi: Pt[] | null;
}

export interface VideoInfo {
  path: string; width: number; height: number; container_fps: number; nb_frames: number;
  duration_s: number; codec: string; pix_fmt: string; bit_depth: number; color_transfer: string;
  is_dji: boolean; srt_fps: number | null; suggested_capture_fps: number; slowmo_suspect: boolean;
  warnings: string[];
}

export interface CalibrationResult {
  error?: string;
  method?: string; m_per_px?: number; cm_per_px?: number; rms_m?: number; max_rel_error?: number;
  line_gap_m?: number | null; warnings?: string[]; fit_ok?: boolean;
  residuals?: { kind: string; index: number; i?: number; j?: number; error_m: number; measured_m?: number; given_m?: number; rel_error?: number }[];
}

export interface ProjectConfig {
  source: string; capture_fps: number | null; real_duration_s: number | null;
  start_frame: number; end_frame: number | null;
  calibration: Calibration;
  detection: Record<string, any>; tracking: Record<string, any>; stabilization: Record<string, any>;
  speed: Record<string, any>; render: Record<string, any>; stats: Record<string, any>;
}

export interface Job {
  id: string; project_id: string; kind: string; status: string; progress: Record<string, any>;
  result: any; error: string | null; log: { t: number; level: string; message: string }[];
  started: number | null;
}

export interface Project {
  id: string; name: string; created: number; info: VideoInfo; config: ProjectConfig;
  status: Record<string, boolean>; jobs: Job[]; timing_warnings: string[]; source_exists: boolean;
  calibration_result: CalibrationResult; proxy_job?: string;
}

export interface LibraryProject {
  id: string; name: string; source: string; created: number; opened: number | null; updated: number | null;
  last_screen: string | null; source_exists: boolean; width: number; height: number;
  capture_fps: number | null; duration_s: number | null; has_proxy: number; has_results: number; has_render: number;
  calibrated: number; n_vehicles: number | null; n_valid: number | null; mean_kmh: number | null; p85_kmh: number | null;
  analyzed: number | null;
}

export interface JobRecord {
  id: string; kind: string; status: string; created: number; started: number | null; finished: number | null;
  error: string | null; result: Record<string, any> | null;
}

export interface Vehicle {
  track_id: number; class: string; direction: string; t_entry_s: number; t_exit_s: number;
  frame_entry: number; frame_exit: number; dt_s: number; path_length_m: number; chord_m: number;
  speed_kmh: number; speed_line_kmh: number; disagreement_pct: number; fit_residual_m: number;
  accel_mps2: number; max_gap_s: number; n_points: number; mean_conf: number; valid: boolean; flags: string;
  speed_max_kmh: number; speed_min_kmh: number; speed_sigma_kmh: number | null; t_max_s: number | null;
  d_max_m: number | null; speed_trend_pct: number | null;
  profile: { t: number[]; d: number[]; v: number[] } | null;
  // margin of error (k=2, ~95 %); *_u_lower adds the one-sided uncorrected parallax bias below the value
  speed_avg_se_kmh?: number | null;
  speed_max_u_kmh?: number; speed_max_u_lower_kmh?: number; speed_avg_u_kmh?: number; speed_avg_u_lower_kmh?: number;
  u_pct?: number; u_cal_pct?: number; u_parallax_pct?: number | null; u_over_target?: boolean;
  uncertainty?: Record<"max" | "avg", { calibration: number; noise: number; parallax_bias: number }>;
}

export interface UncertaintySummary {
  k: number; mc_samples: number; line_gap_rel_pct: number | null; fit_ratio: number | null; fit_dof: number;
  fit_inflation: number; parallax: "bounded" | "unknown"; target_pct: number; n_over_target: number;
  median_u_pct: number | null;
  settings: { distance_tolerance_m: number; click_tolerance_px: number; drone_height_m: number | null; target_pct: number };
}

export interface Reference {
  speed_kmh: number; t_s?: number | null; track_id?: number | null; direction?: "A->B" | "B->A" | null;
  metric: "avg" | "max"; tolerance_kmh: number; source: string; note: string;
}

export interface ValidationRow {
  index: number; track_id: number | null; note: string; measured_kmh?: number; error_kmh?: number; error_pct?: number;
  valid?: boolean; u_kmh?: number; u_lower_kmh?: number; bound_low_kmh?: number; bound_high_kmh?: number; inside?: boolean;
}

export interface Validation {
  references: Reference[]; updated: number;
  report: {
    rows: ValidationRow[];
    summary: {
      n_references: number; n_matched: number; expected_coverage: number; verdict: string; message: string;
      bias_kmh?: number; bias_pct?: number; sd_kmh?: number; sd_pct?: number; rms_pct?: number; max_abs_pct?: number;
      n_inside?: number; n_with_margin?: number; coverage?: number; z_rms?: number; p_coverage?: number;
    };
  };
}

export interface Stats {
  n_valid: number; n_total: number; mean?: number; sd?: number; mean_ci?: [number, number];
  p50?: number; p85?: number; p85_ci?: [number, number]; p95?: number; p95_ci?: [number, number];
  min?: number; max?: number; required_n?: Record<string, number>; confidence: number; permitted_error_kmh: number;
}

export interface Results {
  capture_fps: number; container_fps: number; nb_frames: number; vehicles: Vehicle[]; stats: Stats;
  by_direction: Record<string, Stats>; timing_warnings: string[];
  stats_max?: Stats; by_direction_max?: Record<string, Stats>;
  uncertainty?: UncertaintySummary | null;
  calibration_check?: { n: number; status: "ok" | "suspect" | "insufficient"; mean_pct?: number; sd_pct?: number; message?: string };
  run: Record<string, any>; calibration: CalibrationResult;
  render: { file: string; output_fps: number; playback_speed: number; capture_fps: number; mode: string } | null;
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  fs: (path?: string) => req<{ path: string; parent: string; entries: { name: string; path: string; dir: boolean; size?: number }[] }>(
    "GET", `/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  projects: () => req<LibraryProject[]>("GET", "/api/projects"),
  opened: (id: string, screen: string) => req<{ ok: boolean }>("POST", `/api/projects/${id}/opened`, { screen }),
  relink: (id: string, source: string) => req<Project & { relink_warnings: string[] }>("POST", `/api/projects/${id}/relink`, { source }),
  remove: (id: string) => req<{ deleted: string }>("DELETE", `/api/projects/${id}`),
  jobs: (id: string) => req<JobRecord[]>("GET", `/api/projects/${id}/jobs`),
  thumbUrl: (id: string) => `/api/projects/${id}/thumb.jpg`,
  create: (source: string, name?: string) => req<Project>("POST", "/api/projects", { source, name }),
  project: (id: string) => req<Project>("GET", `/api/projects/${id}`),
  patch: (id: string, body: Record<string, unknown>) => req<Project>("PATCH", `/api/projects/${id}`, body),
  putCalibration: (id: string, cal: Calibration) =>
    req<{ saved: boolean; result: CalibrationResult }>("PUT", `/api/projects/${id}/calibration`, cal),
  analyze: (id: string) => req<Job>("POST", `/api/projects/${id}/analyze`),
  preview: (id: string, frame: number, seconds: number) =>
    req<Job>("POST", `/api/projects/${id}/preview?frame=${frame}&seconds=${seconds}`),
  recompute: (id: string) => req<Results>("POST", `/api/projects/${id}/recompute`),
  render: (id: string, settings: Record<string, unknown>) => req<Job>("POST", `/api/projects/${id}/render`, settings),
  overlay: (id: string) => fetch(`/api/projects/${id}/overlay`).then((r) => (r.ok ? r.json() : null)),
  results: (id: string) => req<Results>("GET", `/api/projects/${id}/results`),
  validation: (id: string) => req<Validation>("GET", `/api/projects/${id}/validation`),
  putValidation: (id: string, references: Reference[]) =>
    req<Validation>("PUT", `/api/projects/${id}/validation`, { references }),
  cancel: (jobId: string) => req<Job>("POST", `/api/jobs/${jobId}/cancel`),
  rebuildProxy: (id: string) => req<Job>("POST", `/api/projects/${id}/proxy`),
  frameUrl: (id: string, n: number) => `/api/projects/${id}/frame/${n}`,
  proxyUrl: (id: string) => `/api/projects/${id}/proxy.mp4`,
  fileUrl: (id: string, name: string) => `/api/projects/${id}/files/${name}`,
};

export interface JobState {
  status: string;
  progress: Record<string, any>;
  logs: { level: string; message: string }[];
  preview: string | null;
  end: (Job & { traceback?: string }) | null;
}

/** Subscribe to a job's SSE stream. */
export function useJob(jobId: string | null | undefined, onEnd?: (j: Job) => void): JobState {
  const [state, setState] = useState<JobState>({ status: "idle", progress: {}, logs: [], preview: null, end: null });
  const onEndRef = useRef(onEnd);
  onEndRef.current = onEnd;
  useEffect(() => {
    if (!jobId) return;
    setState({ status: "connecting", progress: {}, logs: [], preview: null, end: null });
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.addEventListener("progress", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, status: d.status, progress: d }));
    });
    es.addEventListener("log", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, logs: [...s.logs, d] }));
    });
    es.addEventListener("preview", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, preview: `data:image/jpeg;base64,${d.jpeg}` }));
    });
    es.addEventListener("end", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, status: d.status, end: d }));
      es.close();
      onEndRef.current?.(d);
    });
    return () => es.close();
  }, [jobId]);
  return state;
}

/** Current frame index of a <video>, via requestVideoFrameCallback (not currentTime arithmetic). */
export function useVideoFrame(video: HTMLVideoElement | null, fps: number): { frame: number; mediaTime: number } {
  const [st, setSt] = useState({ frame: 0, mediaTime: 0 });
  useEffect(() => {
    if (!video) return;
    const hasRvfc = "requestVideoFrameCallback" in video;
    let handle = 0;
    let lastRvfc = 0;
    const cb = (_now: number, meta: VideoFrameCallbackMetadata) => {
      lastRvfc = performance.now();
      setSt({ frame: Math.round(meta.mediaTime * fps), mediaTime: meta.mediaTime });
      handle = video.requestVideoFrameCallback(cb);
    };
    if (hasRvfc) handle = video.requestVideoFrameCallback(cb);
    // Fallback only when frame callbacks are unavailable or stalled (e.g. a hidden tab).
    const onTime = () => {
      if (hasRvfc && performance.now() - lastRvfc < 500) return;
      const t = video.currentTime;
      setSt({ frame: Math.floor(t * fps + 1e-6), mediaTime: t });
    };
    video.addEventListener("seeked", onTime);
    video.addEventListener("timeupdate", onTime);
    return () => {
      if (hasRvfc) video.cancelVideoFrameCallback(handle);
      video.removeEventListener("seeked", onTime);
      video.removeEventListener("timeupdate", onTime);
    };
  }, [video, fps]);
  return st;
}

export function useDebounced<T>(fn: (v: T) => void, ms: number) {
  const t = useRef<number>();
  const f = useRef(fn);
  f.current = fn;
  return (v: T) => {
    window.clearTimeout(t.current);
    t.current = window.setTimeout(() => f.current(v), ms);
  };
}

/** Report the open project + screen to the server, so the library can resume there. */
export function trackOpenedProjects() {
  let last = "";
  const report = () => {
    const m = window.location.hash.match(/^#\/p\/([a-z0-9-]+)(?:\/(\w+))?/);
    if (!m) return;
    const key = `${m[1]}/${m[2] ?? "import"}`;
    if (key === last) return;
    last = key;
    api.opened(m[1], m[2] ?? "import").catch(() => {});
  };
  window.addEventListener("hashchange", report);
  report();
}

export function timeAgo(t: number | null | undefined): string {
  if (!t) return "never";
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)} d ago`;
  return new Date(t * 1000).toLocaleDateString();
}

/** "±1.1", or "+0.5 −1.4" when the uncorrected parallax bias widens the lower side. */
export function margin(v: Vehicle, key: "max" | "avg"): string | null {
  const u = v[`speed_${key}_u_kmh`], lo = v[`speed_${key}_u_lower_kmh`] ?? u;
  if (u === undefined || u === null || lo === undefined) return null;
  return lo - u < 0.05 ? `±${u.toFixed(1)}` : `+${u.toFixed(1)} −${lo.toFixed(1)}`;
}

/** Hover text: how the margin of error is made up. */
export function marginBreakdown(v: Vehicle, key: "max" | "avg"): string {
  const t = v.uncertainty?.[key];
  if (!t) return "No margin of error in these results: press Recompute speeds.";
  const val = key === "max" ? v.speed_max_kmh : v.speed_kmh;
  const pct = (x: number) => `${x.toFixed(2)} km/h (${((100 * x) / val).toFixed(2)} %)`;
  const lines = [
    `${key === "max" ? "Max" : "Average A→B"} ${val.toFixed(1)} km/h, margin ${margin(v, key)} km/h (95 %, k=2)`,
    `• calibration: ±${pct(t.calibration)}`,
    `• ${key === "max" ? "momentary-speed noise" : "regression noise"}: ±${pct(t.noise)}`,
    t.parallax_bias > 0
      ? `• height parallax, not corrected: up to −${pct(t.parallax_bias)} (measured can only be too high)`
      : "• height parallax: not corrected and not bounded (drone height unknown)",
    "Terms are independent and combined root-sum-square.",
  ];
  if (v.u_over_target) lines.push(`Over the accuracy target: the margin is ${fmt(v.u_pct, 1)} % of the speed.`);
  return lines.join("\n");
}

export const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d);
