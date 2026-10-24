import { useEffect, useRef, useState } from "react";

type Props = { rotation: number; videoRef: (v: HTMLVideoElement | null) => void } & React.VideoHTMLAttributes<HTMLVideoElement>;

/** A <video> shown rotated clockwise by `rotation` degrees, filling its box. Unrotated it is a
 * plain <video>. Native controls would turn with the picture, so callers show VideoControls. */
export default function RotatedVideo({ rotation, videoRef, className, ...props }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize([el.clientWidth, el.clientHeight]));
    ro.observe(el);
    return () => ro.disconnect();
  }, [rotation]);
  if (!rotation) return <video ref={videoRef} className={className} {...props} />;
  const swap = rotation === 90 || rotation === 270;
  return (
    <div ref={box} className={`rot-box ${className ?? ""}`}>
      <video ref={videoRef} {...props} controls={false} style={{
        width: swap ? size[1] : size[0], height: swap ? size[0] : size[1],
        transform: `translate(-50%, -50%) rotate(${rotation}deg)`,
      }} />
    </div>
  );
}

const clock = (s: number) => {
  s = Math.max(0, s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  const sec = x.toFixed(2).padStart(5, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};

/** True when a key press is meant for a form field, not the player. */
export const typing = (e: KeyboardEvent) => {
  const el = e.target as HTMLElement | null;
  return !!el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName));
};

const RATES = [0.25, 0.5, 1, 2];

interface ControlsProps {
  video: HTMLVideoElement | null;
  containerFps: number; // frames per second of the video file (proxy == source)
  /** real seconds for a video time (slow-motion clips: frame / capture fps); default identity */
  realTime?: (t: number) => number;
  seekBar?: boolean; // a seek bar, where there's no timeline
  fullscreenEl?: HTMLElement | null; // what goes full screen (the video and its overlay)
  keys?: boolean; // Space / K / L / arrows / Home / End anywhere on the page (default on)
}

/** Our player controls (the browser's own don't step frames and turn with a rotated view):
 * play/pause, frame steps, start/end, speed, real-time readout, full screen. */
export function VideoControls({ video, containerFps, realTime = (t) => t, seekBar = false, fullscreenEl, keys = true }: ControlsProps) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!video) return;
    const on = () => tick((n) => n + 1);
    const evs = ["timeupdate", "play", "pause", "loadedmetadata", "seeked", "ratechange"];
    evs.forEach((e) => video.addEventListener(e, on));
    return () => evs.forEach((e) => video.removeEventListener(e, on));
  }, [video]);

  const frameOf = (v: HTMLVideoElement) => Math.max(0, Math.round(v.currentTime * containerFps - 0.5));
  const toFrame = (v: HTMLVideoElement, f: number) => {
    v.pause();
    const last = Number.isFinite(v.duration) ? Math.floor(v.duration * containerFps) - 1 : f;
    v.currentTime = (Math.max(0, Math.min(last, f)) + 0.5) / containerFps;
  };
  const toggle = (v: HTMLVideoElement) => (v.paused || v.ended ? v.play() : v.pause());

  useEffect(() => {
    if (!video || !keys) return;
    const on = (e: KeyboardEvent) => {
      if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      const sec = Math.round(containerFps);
      if (e.key === " " || e.code === "Space") { toggle(video); e.preventDefault(); }
      else if (e.key === "k" || e.key === "K") video.pause();
      else if (e.key === "l" || e.key === "L") {
        // Premiere: L plays, pressing it again plays faster
        if (!video.paused) video.playbackRate = Math.min(2, video.playbackRate * 2); else video.play();
      }
      else if (e.key === "ArrowRight") { toFrame(video, frameOf(video) + (e.shiftKey ? sec : 1)); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { toFrame(video, frameOf(video) - (e.shiftKey ? sec : 1)); e.preventDefault(); }
      else if (e.key === "Home") { toFrame(video, 0); e.preventDefault(); }
      else if (e.key === "End") { toFrame(video, Number.MAX_SAFE_INTEGER); e.preventDefault(); }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [video, keys, containerFps]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!video) return null;
  const d = Number.isFinite(video.duration) ? video.duration : 0;
  // buttons lose focus after a click, so Space keeps meaning play/pause rather than "press it again"
  const btn = (label: string, title: string, fn: () => void, cls = "ghost") => (
    <button className={`vc-btn ${cls}`} title={title} aria-label={title}
      onClick={(e) => { fn(); (e.currentTarget as HTMLButtonElement).blur(); }}>{label}</button>
  );
  return (
    <div className="video-controls">
      {btn("⏮", "To the start (Home)", () => toFrame(video, 0))}
      {btn("◂", "Previous frame (←, Shift+← one second)", () => toFrame(video, frameOf(video) - 1))}
      {btn(video.paused ? "▶" : "❚❚", video.paused ? "Play (Space)" : "Pause (Space)", () => toggle(video), "vc-play")}
      {btn("▸", "Next frame (→, Shift+→ one second)", () => toFrame(video, frameOf(video) + 1))}
      {btn("⏭", "To the end (End)", () => toFrame(video, Number.MAX_SAFE_INTEGER))}
      <span className="vc-time" title="Real time (frame ÷ capture fps) and frame number">
        {clock(realTime(video.currentTime))} <span className="muted">/ {clock(realTime(d))} · frame {frameOf(video)}</span>
      </span>
      {seekBar ? <input type="range" className="grow" min={0} max={d || 1} step={0.001} value={video.currentTime} aria-label="Seek"
        onChange={(e) => { video.currentTime = +e.target.value; }} /> : <span className="grow" />}
      <select className="vc-rate" value={String(video.playbackRate)} title="Playback speed (L to speed up while playing)"
        onChange={(e) => { video.playbackRate = +e.target.value; e.currentTarget.blur(); }}>
        {RATES.map((r) => <option key={r} value={r}>{r}×</option>)}
      </select>
      {fullscreenEl && btn("⛶", "Full screen", () => {
        if (document.fullscreenElement) document.exitFullscreen(); else fullscreenEl.requestFullscreen?.();
      })}
    </div>
  );
}
