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
  status: Record<string, boolean>; jobs: Job[]; timing_warnings: string[];
  calibration_result: CalibrationResult; proxy_job?: string;
}

export interface Vehicle {
  track_id: number; class: string; direction: string; t_entry_s: number; t_exit_s: number;
  frame_entry: number; frame_exit: number; dt_s: number; path_length_m: number; chord_m: number;
  speed_kmh: number; speed_line_kmh: number; disagreement_pct: number; fit_residual_m: number;
  accel_mps2: number; max_gap_s: number; n_points: number; mean_conf: number; valid: boolean; flags: string;
}

export interface Stats {
  n_valid: number; n_total: number; mean?: number; sd?: number; mean_ci?: [number, number];
  p50?: number; p85?: number; p85_ci?: [number, number]; p95?: number; p95_ci?: [number, number];
  min?: number; max?: number; required_n?: Record<string, number>; confidence: number; permitted_error_kmh: number;
}

export interface Results {
  capture_fps: number; container_fps: number; nb_frames: number; vehicles: Vehicle[]; stats: Stats;
  by_direction: Record<string, Stats>; timing_warnings: string[];
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
  projects: () => req<{ id: string; name: string; created: number; source: string }[]>("GET", "/api/projects"),
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

export const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d);
