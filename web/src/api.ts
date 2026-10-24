import { useEffect, useRef, useState } from "react";

export type Pt = [number, number];

/** View rotation, degrees clockwise. Only how frames are shown: every stored coordinate is in the
 * original (unrotated) frame. */
export type Rotation = 0 | 90 | 180 | 270;
/** a pixel of the w x h source frame -> the same pixel in the frame rotated by `rot` */
export function rotPt(p: Pt, rot: number, w: number, h: number): Pt {
  const [x, y] = p;
  return rot === 90 ? [h - 1 - y, x] : rot === 180 ? [w - 1 - x, h - 1 - y] : rot === 270 ? [y, w - 1 - x] : [x, y];
}
export function unrotPt(q: Pt, rot: number, w: number, h: number): Pt {
  const [u, v] = q;
  return rot === 90 ? [v, h - 1 - u] : rot === 180 ? [w - 1 - u, h - 1 - v] : rot === 270 ? [w - 1 - v, u] : [u, v];
}
/** size of the frame after rotation */
export const rotSize = (w: number, h: number, rot: number): [number, number] => (rot === 90 || rot === 270 ? [h, w] : [w, h]);

export interface Segment { a: Pt; b: Pt; meters: number }
export interface ControlPoint { px: Pt; world: Pt }
export interface PolygonDistance { i: number; j: number; meters: number }
export interface Polygon { points: Pt[]; distances: PolygonDistance[] }
export interface MotionPath { name: string; points: Pt[]; corridor_m: number }
export interface PathDraw {
  name: string; length_m: number; measured_m: number;
  px: Pt[] | null; trusted_px: Pt[] | null; corridor_px: [Pt[], Pt[]] | null;
}
/** One colour per path of motion, in order (distinct from the green/amber/red speed colours). */
export const PATH_COLORS = ["#00c8ff", "#b98cff", "#ff3cc8", "#78e6ff", "#ffd278"];
export interface TrafficLight { px: Pt; radius_px: number }
export type StopDirection = "both" | "forward" | "reverse";
export interface StopLine { points: Pt[]; direction: StopDirection }
export interface RedLightSetup {
  name: string; lights: TrafficLight[]; stop_lines: StopLine[];
  state_source: "colour" | "manual"; red_intervals: [number, number][];
}
export interface SolidLine { name: string; points: Pt[] }
/** One colour per red-light setup (stop lines are drawn in the light's phase colour once analysed). */
export const LIGHT_COLORS = ["#ff5a5a", "#ff9f1a", "#f368e0", "#ff7f7f"];
export const SOLID_COLOR = "#f2f2f2";
export interface Calibration {
  red_lights?: RedLightSetup[];
  solid_lines?: SolidLine[];
  polygon?: Polygon | null;
  paths?: MotionPath[];
  path_clip_to_calibration?: boolean;
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
  paths?: PathDraw[];
  residuals?: { kind: string; index: number; i?: number; j?: number; error_m: number; measured_m?: number; given_m?: number; rel_error?: number }[];
}

export interface ProjectConfig {
  source: string; rotation?: Rotation; capture_fps: number | null; real_duration_s: number | null;
  start_frame: number; end_frame: number | null;
  calibration: Calibration;
  detection: Record<string, any>; tracking: Record<string, any>; stabilization: Record<string, any>;
  speed: Record<string, any>; render: Record<string, any>; stats: Record<string, any>;
  violations?: Record<string, any>;
  parts?: [number, number][] | null;
  vehicles?: { names: Record<string, string>; merges: number[][] };
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
  /** false: a gap or jump in the stretch makes the average inconclusive (the max still stands) */
  avg_valid?: boolean; max_clean_pct?: number | null;
  speed_max_kmh: number; speed_min_kmh: number; speed_sigma_kmh: number | null; t_max_s: number | null;
  d_max_m: number | null; speed_trend_pct: number | null;
  profile: { t: number[]; d: number[]; v: number[] } | null;
  path?: string; coverage_pct?: number;
  violations?: string[];
  /** the user's name for it ("" = none); the tracks merged into it (first = this id) */
  name?: string; merged?: number[];
  // margin of error (k=2, ~95 %); *_u_lower adds the one-sided uncorrected parallax bias below the value
  speed_avg_se_kmh?: number | null;
  speed_max_u_kmh?: number; speed_max_u_lower_kmh?: number; speed_avg_u_kmh?: number; speed_avg_u_lower_kmh?: number;
  u_pct?: number; u_cal_pct?: number; u_parallax_pct?: number | null;
  uncertainty?: Record<"max" | "avg", { calibration: number; noise: number; parallax_bias: number }>;
}

export interface UncertaintySummary {
  k: number; mc_samples: number; line_gap_rel_pct: number | null; fit_ratio: number | null; fit_dof: number;
  fit_inflation: number; parallax: "bounded" | "unknown";
  median_u_pct: number | null;
  settings: { distance_tolerance_m: number; click_tolerance_px: number; drone_height_m: number | null };
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

/** Two tracks that look like one vehicle the tracker lost for a moment. */
export interface MergePair {
  a: number; b: number; gap_s: number; miss_m: number; speed_kmh: number; speed_change_kmh: number; likely: boolean;
  t_a_end_s: number; t_b_start_s: number;
}
export interface MergeCandidates {
  pairs: MergePair[];
  tracks: Record<string, { cls: string; t0: number; t1: number; measured: boolean; speed_max_kmh: number | null; name: string }>;
}
export interface VehicleLook { track_id: number; colour: string; rgb: [number, number, number]; suggestion: string; frame: number }

export interface Phase { t0: number; t1: number; state: "red" | "amber" | "green" | "not red" | "unknown" }
export interface RedLightEvent {
  type: "red_light"; name: string; line: number; track_id: number; class: string; t_s: number; t_centre_s: number;
  frame: number; light: string; phase_for_s: number | null; violation: boolean | null; speed_kmh: number | null;
}
export interface SolidLineEvent {
  type: "solid_line"; name: string; track_id: number; class: string; t_s: number; frame: number;
  how: "crossed" | "straddled"; duration_s: number | null; max_over_m: number | null; speed_kmh: number | null;
}
/** A vehicle near a solid line: how far over it was per sample (m, smoothed masks or the box estimate). */
export interface SolidCase {
  line: number; name: string; track_id: number; class: string; source: "mask" | "outline"; step_s: number;
  t: number[]; over_m: number[]; max_over_m: number; closest_m: number; closest_t: number;
  best_frame: number | null; crossed_t: number[]; reported?: boolean;
}
export interface Violations {
  lights: { name: string; source: "colour" | "manual"; read_share: number | null; phases: Phase[] }[];
  red_light: RedLightEvent[]; solid_line: SolidLineEvent[];
  /** vehicles over a solid line by less than the threshold: how far they stayed over for min_duration_s */
  solid_near?: { name: string; track_id: number; class: string; t_s: number; over_m: number; for_s: number }[];
  solid_cases?: SolidCase[];
  masks_missing?: number;
  overlap_m?: number | null;
  min_duration_s?: number;
  n_red_light: number; n_solid_line: number; warnings: string[];
}
export const PHASE_COLORS: Record<string, string> = { red: "#eb4034", amber: "#ffb020", green: "#2ecc71", "not red": "#5b6270", unknown: "#3a3f48" };

export interface JobHistoryRow {
  id: string; project_id: string; project_name: string | null; kind: string; status: string;
  created: number; started: number | null; finished: number | null; error: string | null; result: any;
  progress?: { stage?: string; done?: number; total?: number; fps?: number };
  files: { name: string; size: number }[];
}

export interface ScanResult {
  scan: { frames: number[]; counts: number[]; moving?: number[]; mode: string; nb_frames: number; container_fps: number;
    duration_s: number; elapsed_s: number; model: string } | null;
  suggested: [number, number][];
  parts: [number, number][] | null;
}

export interface Results {
  capture_fps: number; container_fps: number; nb_frames: number; vehicles: Vehicle[]; stats: Stats;
  by_direction: Record<string, Stats>; timing_warnings: string[];
  stats_max?: Stats; by_direction_max?: Record<string, Stats>;
  by_path?: Record<string, { avg: Stats; max: Stats }>;
  uncertainty?: UncertaintySummary | null;
  calibration_check?: { n: number; status: "ok" | "suspect" | "insufficient"; mean_pct?: number; sd_pct?: number; message?: string };
  violations?: Violations | null;
  run: Record<string, any>; calibration: CalibrationResult;
  /** what changed in the setup since these results: a Recompute applies "recompute"; "analysis" needs a new analysis */
  stale?: { recompute: boolean; analysis: boolean } | null;
  render: { file: string; output_fps: number; playback_speed: number; capture_fps: number; mode: string;
    width?: number; height?: number; frames?: number; parts?: [number, number][] | null } | null;
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
  fs: (path?: string) => req<{ path: string; parent: string; entries: { name: string; path: string; dir: boolean; size?: number; projects?: { id: string; name: string }[] }[] }>(
    "GET", `/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  projects: () => req<LibraryProject[]>("GET", "/api/projects"),
  opened: (id: string, screen: string) => req<{ ok: boolean }>("POST", `/api/projects/${id}/opened`, { screen }),
  relink: (id: string, source: string) => req<Project & { relink_warnings: string[] }>("POST", `/api/projects/${id}/relink`, { source }),
  remove: (id: string) => req<{ deleted: string }>("DELETE", `/api/projects/${id}`),
  jobs: (id: string) => req<JobRecord[]>("GET", `/api/projects/${id}/jobs`),
  thumbUrl: (id: string) => `/api/projects/${id}/thumb.jpg`,
  create: (source: string, name?: string) => req<Project & { existing?: boolean }>("POST", "/api/projects", { source, name }),
  project: (id: string) => req<Project>("GET", `/api/projects/${id}`),
  patch: (id: string, body: Record<string, unknown>) => req<Project>("PATCH", `/api/projects/${id}`, body),
  scan: (id: string) => req<Job>("POST", `/api/projects/${id}/scan`),
  getScan: (id: string) => req<ScanResult>("GET", `/api/projects/${id}/scan`),
  readLights: (id: string, frame: number) =>
    req<{ frame: number; stabilized: boolean; setups: { name: string; state: string; counts: [number, number, number][] }[] }>(
      "GET", `/api/projects/${id}/lights/${frame}`),
  putCalibration: (id: string, cal: Calibration) =>
    req<{ saved: boolean; result: CalibrationResult }>("PUT", `/api/projects/${id}/calibration`, cal),
  analyze: (id: string) => req<Job>("POST", `/api/projects/${id}/analyze`),
  preview: (id: string, frame: number, seconds: number) =>
    req<Job>("POST", `/api/projects/${id}/preview?frame=${frame}&seconds=${seconds}`),
  recompute: (id: string) => req<Results>("POST", `/api/projects/${id}/recompute`),
  render: (id: string, settings: Record<string, unknown>) => req<Job>("POST", `/api/projects/${id}/render`, settings),
  nameVehicle: (id: string, tid: number, name: string) =>
    req<{ track_id: number; name: string }>("PUT", `/api/projects/${id}/vehicles/${tid}`, { name }),
  mergeVehicles: (id: string, ids: number[]) => req<Results>("POST", `/api/projects/${id}/vehicles/merge`, { ids }),
  /** outline (segment) the vehicles near solid lines that have no mask yet, then recompute */
  outline: (id: string) => req<Job>("POST", `/api/projects/${id}/masks`),
  splitVehicle: (id: string, tid: number) => req<Results>("POST", `/api/projects/${id}/vehicles/${tid}/split`),
  mergeCandidates: (id: string) => req<MergeCandidates>("GET", `/api/projects/${id}/vehicles/candidates`),
  vehicleLook: (id: string, tid: number) => req<VehicleLook>("GET", `/api/projects/${id}/vehicles/${tid}/look`),
  vehicleThumbUrl: (id: string, tid: number, v = "") => `/api/projects/${id}/vehicles/${tid}/thumb.jpg${v ? `?v=${v}` : ""}`,
  /** one output frame as the render would draw it with these (unsaved) settings */
  /** null: superseded by a newer request */
  renderPreview: async (id: string, frame: number, settings: Record<string, unknown>, signal?: AbortSignal): Promise<Blob | null> => {
    const r = await fetch(`/api/projects/${id}/render/preview?frame=${frame}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings), signal,
    });
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* not json */ }
      throw new Error(msg);
    }
    return r.status === 204 ? null : r.blob();
  },
  overlay: (id: string) => fetch(`/api/projects/${id}/overlay`).then((r) => (r.ok ? r.json() : null)),
  results: (id: string) => req<Results>("GET", `/api/projects/${id}/results`),
  validation: (id: string) => req<Validation>("GET", `/api/projects/${id}/validation`),
  putValidation: (id: string, references: Reference[]) =>
    req<Validation>("PUT", `/api/projects/${id}/validation`, { references }),
  cancel: (jobId: string) => req<Job>("POST", `/api/jobs/${jobId}/cancel`),
  jobHistory: () => req<{ jobs: JobHistoryRow[]; artifacts_bytes: number }>("GET", "/api/jobs/history"),
  jobFileUrl: (jobId: string, name: string) => `/api/jobs/${jobId}/files/${encodeURIComponent(name)}`,
  deleteJobFiles: (jobId: string) => req<{ deleted: boolean }>("DELETE", `/api/jobs/${jobId}/files`),
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

/** "±1.8 %", or "+1.8 −3.1 %" when the uncorrected parallax bias widens the lower side. */
export function margin(v: Vehicle, key: "max" | "avg"): string | null {
  const u = v[`speed_${key}_u_kmh`], lo = v[`speed_${key}_u_lower_kmh`] ?? u;
  const val = key === "max" ? v.speed_max_kmh : v.speed_kmh;
  if (u === undefined || u === null || lo === undefined || !(val > 0)) return null;
  const p = (x: number) => ((100 * x) / val).toFixed(1);
  return p(lo) === p(u) ? `±${p(u)} %` : `+${p(u)} −${p(lo)} %`;
}

/** Hover text: how the margin of error is made up. */
export function marginBreakdown(v: Vehicle, key: "max" | "avg"): string {
  const t = v.uncertainty?.[key];
  if (!t) return "No margin of error in these results: press Recompute speeds.";
  const val = key === "max" ? v.speed_max_kmh : v.speed_kmh;
  const pct = (x: number) => `${x.toFixed(2)} km/h (${((100 * x) / val).toFixed(2)} %)`;
  const lines = [
    `${key === "max" ? "Max" : "Average A→B"} ${val.toFixed(1)} km/h, margin ${margin(v, key)} (95 %, k=2)`,
    `• calibration: ±${pct(t.calibration)}`,
    `• ${key === "max" ? "momentary-speed noise" : "regression noise"}: ±${pct(t.noise)}`,
    t.parallax_bias > 0
      ? `• height parallax, not corrected: up to −${pct(t.parallax_bias)} (measured can only be too high)`
      : "• height parallax: not corrected and not bounded (drone height unknown)",
    "Terms are independent and combined root-sum-square.",
  ];
  return lines.join("\n");
}

export const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d);
