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

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Play/pause and a seek bar for a video whose native controls are hidden. */
export function VideoControls({ video }: { video: HTMLVideoElement | null }) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!video) return;
    const on = () => tick((n) => n + 1);
    const evs = ["timeupdate", "play", "pause", "loadedmetadata", "seeked"];
    evs.forEach((e) => video.addEventListener(e, on));
    return () => evs.forEach((e) => video.removeEventListener(e, on));
  }, [video]);
  if (!video) return null;
  const d = Number.isFinite(video.duration) ? video.duration : 0;
  return (
    <div className="video-controls">
      <button className="ghost" onClick={() => (video.paused ? video.play() : video.pause())} title="Play / pause">{video.paused ? "▶" : "❚❚"}</button>
      <input type="range" className="grow" min={0} max={d || 1} step={0.001} value={video.currentTime} aria-label="Seek"
        onChange={(e) => { video.currentTime = +e.target.value; }} />
      <span className="muted small time">{clock(video.currentTime)} / {clock(d)}</span>
    </div>
  );
}
