import { useEffect, useMemo, useRef, useState } from "react";
import Timeline, { Range } from "../components/Timeline";
import Help from "../components/Help";
import ProgressBar from "../components/ProgressBar";
import RotatedVideo, { VideoControls } from "../components/RotatedVideo";
import { api, Project, Results, rotSize, useJob, useVideoFrame } from "../api";

interface Props { project: Project; onChanged: (p: Project) => void }

type Settings = {
  mode: string; output_fps: number | null; playback_speed: number; scale: number; lut: string; encoder: string; quality: number;
  trail_s: number; label_hold_s: number; label_position: string; only_parts: boolean; ranges: Range[] | null;
  show_lines: boolean; show_rules: boolean; show_boxes: boolean; show_trails: boolean; show_labels: boolean; show_names: boolean;
  show_unmeasured: boolean; show_hud: boolean; show_margin: boolean;
  panel: string; panel_layout: string; panel_width: number; panel_sort: string; panel_rows: number;
  title: string; subtitle: string; footer: string;
};
const KEYS: (keyof Settings)[] = ["mode", "output_fps", "playback_speed", "scale", "lut", "encoder", "quality", "trail_s", "label_hold_s",
  "label_position", "only_parts", "ranges", "show_lines", "show_rules", "show_boxes", "show_trails", "show_labels", "show_names",
  "show_unmeasured", "show_hud", "show_margin", "panel", "panel_layout", "panel_width", "panel_sort", "panel_rows",
  "title", "subtitle", "footer"];
type Preset = "whole" | "parts" | "vehicles" | "violations" | "custom";

const clock = (s: number) => {
  const m = Math.floor(s / 60), x = Math.floor(s - m * 60);
  return `${m}:${String(x).padStart(2, "0")}`;
};
const even = (x: number) => Math.max(2, Math.round(x / 2) * 2);

/** Sorted and merged; neighbours closer than `join` frames become one range. */
function joinRanges(rs: Range[], n: number, join = 0): Range[] {
  const s = rs.map(([a, b]) => [Math.max(0, Math.round(a)), Math.min(n, Math.round(b))] as Range).filter(([a, b]) => b > a)
    .sort((p, q) => p[0] - q[0]);
  const out: Range[] = [];
  for (const r of s) {
    const last = out[out.length - 1];
    if (last && r[0] <= last[1] + join) last[1] = Math.max(last[1], r[1]);
    else out.push([r[0], r[1]]);
  }
  return out;
}

/** The annotated video (what to render, how it looks, its quality) with a live preview of any
 * frame exactly as the render draws it, and the data downloads. */
export default function ExportScreen({ project, onChanged }: Props) {
  const info = project.info;
  const nb = info.nb_frames;
  const rot = project.config.rotation ?? 0;
  const [res, setRes] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [s, setS] = useState<Settings>(() => {
    const r = project.config.render as any;
    const out = Object.fromEntries(KEYS.map((k) => [k, r[k]])) as Settings;
    out.lut = r.lut ?? "";
    out.ranges = r.ranges ?? null;
    // older projects kept 0.5 of 4K: the same as 1080p
    if (r.scale === 0.5 && info.height > 1080) out.scale = +(1080 / info.height).toFixed(4);
    return out;
  });
  const [preset, setPreset] = useState<Preset>(() => (s.ranges ? "custom" : project.config.parts && s.only_parts ? "parts" : "whole"));
  const fps = project.config.capture_fps || info.container_fps;
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setS((o) => ({ ...o, [k]: v }));

  useEffect(() => { api.results(project.id).then(setRes).catch((e) => setErr(String(e))); }, [project.id]);

  // the settings are kept with the project as they change (the render uses the saved ones)
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    const t = window.setTimeout(() => {
      api.patch(project.id, { render: { ...s, lut: s.lut || null } }).then(onChanged).catch((e) => setErr(String(e)));
    }, 700);
    return () => window.clearTimeout(t);
  }, [s]);  // eslint-disable-line react-hooks/exhaustive-deps

  // ---- preview: one frame, as rendered
  const [frame, setFrame] = useState<number | null>(null);
  const [img, setImg] = useState<string | null>(null);
  const [pvBusy, setPvBusy] = useState(false);
  const [pvErr, setPvErr] = useState<string | null>(null);
  useEffect(() => {  // start where there's something to see: the fastest vehicle, just past its label
    if (!res || frame !== null) return;
    const v = res.vehicles.filter((x) => x.valid).sort((a, b) => b.speed_max_kmh - a.speed_max_kmh)[0];
    setFrame(v ? Math.min(nb - 1, Math.round((v.t_exit_s + 0.2) * fps)) : project.config.calibration.reference_frame);
  }, [res]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (frame === null) return;
    const ctl = new AbortController();
    const t = window.setTimeout(async () => {
      setPvBusy(true);
      try {
        const blob = await api.renderPreview(project.id, frame, { ...s, lut: s.lut || null }, ctl.signal);
        if (!blob) return;  // a newer frame was asked for
        const url = URL.createObjectURL(blob);
        setImg((old) => { if (old) URL.revokeObjectURL(old); return url; });
        setPvErr(null);
      } catch (e) {
        if (!ctl.signal.aborted) setPvErr(String(e));
      } finally {
        if (!ctl.signal.aborted) setPvBusy(false);
      }
    }, 300);
    return () => { window.clearTimeout(t); ctl.abort(); };
  }, [frame, s, project.id]);

  // ---- what to render
  const presetRanges = (p: Preset): Range[] | null => {
    if (!res) return null;
    if (p === "vehicles") {
      const r = res.vehicles.filter((v) => v.valid).map((v) => [(v.t_entry_s - 1) * fps, (v.t_exit_s + 1 + s.label_hold_s) * fps] as Range);
      return r.length ? joinRanges(r, nb, 2 * fps) : null;
    }
    if (p === "violations") {
      const vi = res.violations;
      const r = vi ? [...vi.red_light.filter((e) => e.violation), ...vi.solid_line].map((e) => [(e.t_s - 3) * fps, (e.t_s + 3) * fps] as Range) : [];
      return r.length ? joinRanges(r, nb, 2 * fps) : null;
    }
    return null;
  };
  const choose = (p: Preset) => {
    setPreset(p);
    if (p === "whole") setS((o) => ({ ...o, ranges: null, only_parts: false }));
    else if (p === "parts") setS((o) => ({ ...o, ranges: null, only_parts: true }));
    else if (p !== "custom") setS((o) => ({ ...o, ranges: presetRanges(p), only_parts: false }));
  };
  const shownRanges: Range[] | null = s.ranges ?? (s.only_parts && project.config.parts ? (project.config.parts as Range[]) : null);
  const srcFrames = (shownRanges ?? [[0, nb] as Range]).reduce((a, [x, y]) => a + (y - x), 0);
  const outFps = s.output_fps || fps;
  const outSeconds = srcFrames / fps / s.playback_speed;

  // ---- output size
  const sizeOf = (scale: number) => {
    const [w, h] = [even(info.width * scale), even(info.height * scale)];
    const [vw, vh] = rotSize(w, h, rot);
    const pw = s.panel !== "none" && s.panel_layout === "extend" ? even(vw * s.panel_width) : 0;
    return [vw + pw, vh];
  };
  const resOptions = useMemo(() => {
    const opts = [720, 1080, 1440, 2160].filter((h) => h < info.height).map((h) => +(h / info.height).toFixed(4));
    return [...opts, 1];
  }, [info.height]);
  const [ow, oh] = sizeOf(s.scale);

  // ---- render job
  const [renderJob, setRenderJob] = useState<string | null>(project.jobs.find((j) => j.kind === "render")?.id ?? null);
  const rjob = useJob(renderJob, (j) => {
    if (j.status === "done") setView("rendered");
    api.results(project.id).then(setRes).catch(() => {});
    api.project(project.id).then(onChanged).catch(() => {});
  });
  const running = ["running", "queued", "connecting"].includes(rjob.status);
  const startRender = async () => {
    try { setRenderJob((await api.render(project.id, { ...s, lut: s.lut || null })).id); } catch (e) { setErr(String(e)); }
  };
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [wrapEl, setWrapEl] = useState<HTMLDivElement | null>(null);
  const [view, setView] = useState<"preview" | "rendered">("preview");
  const rendered = view === "rendered" && !!res?.render;

  // the rendered video's time <-> frames of the clip: its parts back to back, at playback speed
  const rm = res?.render;
  const rParts: Range[] = rm ? ((rm.parts as Range[] | null | undefined) ?? [[0, nb]]) : [];
  const rRatio = rm ? (rm.playback_speed * (rm.capture_fps || fps)) / rm.output_fps : 1;  // clip frames per video frame
  const rCounts = rParts.map(([a, b]) => (b > a ? Math.floor((b - a - 1) / rRatio) + 1 : 0));
  const clipFrameAt = (videoFrame: number) => {
    let j = videoFrame;
    for (let k = 0; k < rParts.length; k++) {
      if (j < rCounts[k]) return rParts[k][0] + Math.round(j * rRatio);
      j -= rCounts[k];
    }
    return rParts.length ? rParts[rParts.length - 1][1] - 1 : 0;
  };
  const videoTimeOf = (f: number) => {
    let before = 0;
    for (let k = 0; k < rParts.length; k++) {
      const [a, b] = rParts[k];
      if (f < b) return (before + Math.max(0, Math.round((f - a) / rRatio)) + 0.5) / rm!.output_fps;  // in a gap: the next part
      before += rCounts[k];
    }
    return (before - 0.5) / rm!.output_fps;
  };
  const { frame: vFrame } = useVideoFrame(rendered ? video : null, rm?.output_fps ?? 30);
  const playhead = rendered ? clipFrameAt(vFrame) : frame ?? 0;

  const vehicles = res?.vehicles.map((v) => ({ id: v.track_id, f0: Math.round(v.t_entry_s * fps), f1: Math.round(v.t_exit_s * fps), valid: v.valid }));
  const nViol = res?.violations ? res.violations.red_light.filter((e) => e.violation).length + res.violations.solid_line.length : 0;
  const check = (k: keyof Settings, label: string, title?: string) => (
    <label className="check" title={title}><input type="checkbox" checked={!!s[k]} onChange={(e) => set(k, e.target.checked as any)} /> {label}</label>
  );

  return (
    <div className="dash export">
      <div className="dash-head">
        <h2>Export</h2>
        <span className="tag">{ow}×{oh} · {+outFps.toFixed(2)} fps · {clock(outSeconds)}</span>
        <span className="grow" />
        {running ? <button className="danger" onClick={() => renderJob && api.cancel(renderJob)}>Cancel render</button>
          : <button className="primary" onClick={startRender}>{res?.render ? "Render again" : "Render video"}</button>}
      </div>
      {err && <div className="banner error">{err}</div>}
      {renderJob && rjob.status !== "done" && rjob.status !== "idle" && (
        <ProgressBar label={rjob.progress.stage ?? rjob.status} done={rjob.progress.done} total={rjob.progress.total}
          extra={rjob.progress.fps ? `· ${rjob.progress.fps.toFixed(1)} fps` : ""} />
      )}
      {rjob.end?.status === "error" && <div className="banner error">{rjob.end.error}</div>}

      <div className="export-main">
        <div className="export-left">
          <section className="panel">
            <div className="panel-head">
              <div className="seg-ctl" role="tablist" aria-label="Show">
                <button className={!rendered ? "active" : ""} onClick={() => setView("preview")} title="Any frame, drawn as the render will draw it">Preview</button>
                <button className={rendered ? "active" : ""} disabled={!res?.render} onClick={() => setView("rendered")}
                  title={res?.render ? "The last rendered video" : "Nothing rendered yet"}>Rendered</button>
              </div>
              <Help topic="export.preview" />
              {rendered && rm ? <>
                <span className="tag">{rm.width}×{rm.height} · {+rm.output_fps.toFixed(2)} fps{rm.playback_speed !== 1 ? ` · ${rm.playback_speed}×` : ""}</span>
                <span className="muted small">the last render, made with the settings it had then</span>
                <span className="grow" />
                <a className="button" href={api.fileUrl(project.id, "render.mp4")}><span className="dl-ico">↓</span>render.mp4</a>
              </> : <>
                <span className="muted small">frame {frame ?? "–"} · {frame !== null ? (frame / fps).toFixed(2) : "–"} s · as the render will draw it</span>
                <span className="grow" />
                {pvBusy && <span className="muted small">{img ? "updating…" : ""}</span>}
              </>}
            </div>
            {rendered && rm ? (
              <>
                <div ref={setWrapEl} className="video-wrap preview-box" style={{ aspectRatio: `${rm.width ?? ow} / ${rm.height ?? oh}` }}>
                  <RotatedVideo rotation={0} videoRef={setVideo} muted playsInline preload="metadata"
                    onClick={(e) => { const v = e.currentTarget; v.paused ? v.play() : v.pause(); }}
                    src={`${api.fileUrl(project.id, "render.mp4")}?v=${rm.file}${rjob.end?.id ?? ""}`} />
                </div>
                <VideoControls video={video} fullscreenEl={wrapEl} containerFps={rm.output_fps}
                  realTime={(tv) => tv * rm.playback_speed} />
              </>
            ) : (
              <div className={`preview-box${pvBusy ? " busy" : ""}`} style={{ aspectRatio: `${ow} / ${oh}` }}>
                {img && <img src={img} alt="Render preview" />}
                {!img && <div className="preview-wait">{pvErr ? "" : pvBusy || frame === null
                  ? "Preparing the preview: the clip's speeds and violations are worked out once (about 15 s for a 4K clip, longer while a render runs)…"
                  : ""}</div>}
              </div>
            )}
            {pvErr && !rendered && <div className="banner warn small">{pvErr}</div>}
            <Timeline project={project} frame={playhead} vehicles={vehicles} verb="Render"
              onSeek={(f) => { if (rendered && video) { video.pause(); video.currentTime = videoTimeOf(f); } else setFrame(f); }}
              ranges={shownRanges} onRanges={(r) => { setPreset(r ? "custom" : "whole"); setS((o) => ({ ...o, ranges: r, only_parts: false })); }} />
            <p className="footnote">{rendered
              ? "The playhead follows the rendered video through the parts it covers; click the timeline to jump to that moment in it."
              : "Click or drag on the timeline to preview another frame."} Drag on the bottom lane to choose what's rendered:
              over blue to cut out, elsewhere to add; I / O set in and out points.</p>
          </section>
        </div>

        <aside className="export-side">
          <section className="panel">
            <h3>What to render<Help topic="export.range" /></h3>
            <div className="choice-list">
              {([
                ["whole", "The whole clip", `${clock(nb / fps)}`],
                ...(project.config.parts ? [["parts", "The analysed parts", `${project.config.parts.length} part${project.config.parts.length === 1 ? "" : "s"}`]] : []),
                ["vehicles", "Measured vehicles only", `each from 1 s before to just after its label`],
                ...(nViol ? [["violations", "Violations only", `${nViol}, 3 s either side`]] : []),
                ["custom", "Picked on the timeline", s.ranges ? `${s.ranges.length} range${s.ranges.length === 1 ? "" : "s"}` : "drag on the bottom lane"],
              ] as [Preset, string, string][]).map(([k, label, note]) => (
                <label key={k} className={`choice${preset === k ? " on" : ""}`}>
                  <input type="radio" name="range" checked={preset === k} onChange={() => choose(k)} />
                  <span>{label}<span className="muted small"> · {note}</span></span>
                </label>
              ))}
            </div>
            <p className="small muted">Source {clock(srcFrames / fps)} → video {clock(outSeconds)} at {s.playback_speed}× speed.</p>
          </section>

          <section className="panel">
            <h3>Quality<Help topic="export.quality" /></h3>
            <div className="form-grid compact">
              <label>Resolution</label>
              <select value={String(s.scale)} onChange={(e) => set("scale", +e.target.value)}>
                {!resOptions.includes(s.scale) && <option value={String(s.scale)}>{ow}×{oh} (custom)</option>}
                {resOptions.map((k) => { const [w, h] = sizeOf(k); return <option key={k} value={String(k)}>{w}×{h}{k === 1 ? " (original)" : ""}</option>; })}
              </select>
              <label>Frame rate</label>
              <select value={String(s.output_fps ?? "")} onChange={(e) => set("output_fps", e.target.value === "" ? null : +e.target.value)}>
                <option value="">native ({+fps.toFixed(2)} fps)</option><option value="60">60</option><option value="30">30</option><option value="25">25</option>
              </select>
              <label>Playback speed</label>
              <select value={String(s.playback_speed)} onChange={(e) => set("playback_speed", +e.target.value)}>
                <option value="1">1× real time</option><option value="0.5">0.5× slow motion</option><option value="0.25">0.25×</option><option value="0.1">0.1×</option>
              </select>
              <label title="Constant quality: lower is better and bigger">Quality</label>
              <select value={String(s.quality)} onChange={(e) => set("quality", +e.target.value)}>
                {![16, 20, 24, 28].includes(s.quality) && <option value={String(s.quality)}>{s.quality} (custom)</option>}
                <option value="16">high · bigger file</option><option value="20">standard</option><option value="24">smaller file</option><option value="28">smallest · for messaging</option>
              </select>
              <label title="Hardware encoders are much faster; the app falls back to x264 when one isn't available">Encoder</label>
              <select value={s.encoder} onChange={(e) => set("encoder", e.target.value)}>
                <option value="auto">automatic (GPU if available)</option><option value="nvenc">NVIDIA (NVENC)</option>
                <option value="videotoolbox">Apple (VideoToolbox)</option><option value="x264">x264 (CPU)</option>
              </select>
              <label title="Stabilized: frames warped so the lines stay put. Original: the frames as filmed, lines follow the drift">Frames</label>
              <select value={s.mode} onChange={(e) => set("mode", e.target.value)}>
                <option value="stabilized">stabilized (lines fixed)</option><option value="original">as filmed</option>
              </select>
              <label title="A .cube file applied before drawing, e.g. D-Log M → Rec.709">Colour LUT</label>
              <input type="text" value={s.lut} placeholder="optional .cube path" onChange={(e) => set("lut", e.target.value)} />
            </div>
          </section>

          <section className="panel">
            <h3>On the video<Help topic="export.overlay" /></h3>
            <div className="check-grid">
              {check("show_labels", "Speed labels")}
              {check("show_names", "Vehicle names", "Names given on Results, in the labels and the panel")}
              {check("show_boxes", "Vehicle boxes")}
              {check("show_trails", "Trails")}
              {check("show_lines", "Paths / lines A–B")}
              {check("show_rules", "Lights and solid lines")}
              {check("show_unmeasured", "Unmeasured vehicles", "Boxes on vehicles that were tracked but not measured (parked cars etc.)")}
              {check("show_hud", "Time / frame readout")}
            </div>
            <div className="form-grid compact">
              <label title="Pick the side away from neighbouring lanes and lines">Labels</label>
              <select value={s.label_position} onChange={(e) => set("label_position", e.target.value)}>
                <option value="above">above the vehicle</option><option value="below">below the vehicle</option>
                <option value="left">left of the vehicle</option><option value="right">right of the vehicle</option>
              </select>
              <label title="Speeds above it are amber, 10 km/h above it red. Set in Setup → Limits & tolerances.">Speed limit</label>
              <span className="muted">{(project.config.render as any).speed_limit_kmh ?? 50} km/h · set in Setup</span>
              <label title="How long a vehicle keeps its max label after the second line">Label stays</label>
              <select value={String(s.label_hold_s)} onChange={(e) => set("label_hold_s", +e.target.value)}>
                {![1, 2.5, 5, 10].includes(s.label_hold_s) && <option value={String(s.label_hold_s)}>{s.label_hold_s} s</option>}
                <option value="1">1 s</option><option value="2.5">2.5 s</option><option value="5">5 s</option><option value="10">10 s</option>
              </select>
              <label>Trail length</label>
              <select value={String(s.trail_s)} onChange={(e) => set("trail_s", +e.target.value)}>
                {![0.5, 1, 2, 4].includes(s.trail_s) && <option value={String(s.trail_s)}>{s.trail_s} s</option>}
                <option value="0.5">0.5 s</option><option value="1">1 s</option><option value="2">2 s</option><option value="4">4 s</option>
              </select>
            </div>
          </section>

          <section className="panel">
            <h3>Side panel<Help topic="export.panel" /></h3>
            <div className="form-grid compact">
              <label title="Lists every vehicle's max speed as it crosses the second line, with totals">Panel</label>
              <select value={s.panel} onChange={(e) => set("panel", e.target.value)}>
                <option value="right">right side</option><option value="left">left side</option><option value="none">off</option>
              </select>
              {s.panel !== "none" && <>
                <label title="Over the video keeps its size; beside it makes the video wider">Layout</label>
                <select value={s.panel_layout} onChange={(e) => set("panel_layout", e.target.value)}>
                  <option value="overlay">over the video</option><option value="extend">beside the video (wider)</option>
                </select>
                <label>Width</label>
                <select value={String(s.panel_width)} onChange={(e) => set("panel_width", +e.target.value)}>
                  {![0.22, 0.27, 0.33, 0.4].includes(s.panel_width) && <option value={String(s.panel_width)}>{Math.round(s.panel_width * 100)} %</option>}
                  <option value="0.22">narrow</option><option value="0.27">standard</option><option value="0.33">wide</option><option value="0.4">widest</option>
                </select>
                <label>Title</label>
                <input type="text" value={s.title} placeholder="e.g. street, city" onChange={(e) => set("title", e.target.value)} />
                <label>Subtitle</label>
                <input type="text" value={s.subtitle} placeholder="e.g. date · time" onChange={(e) => set("subtitle", e.target.value)} />
                <label>Footer</label>
                <input type="text" value={s.footer} placeholder="e.g. source, handle" onChange={(e) => set("footer", e.target.value)} />
                <label>List order</label>
                <select value={s.panel_sort} onChange={(e) => set("panel_sort", e.target.value)}>
                  <option value="latest">latest first</option><option value="fastest">fastest first</option>
                </select>
                <label>Rows</label>
                <input type="number" min={1} max={20} value={s.panel_rows} onChange={(e) => set("panel_rows", Math.max(1, Math.min(20, +e.target.value || 1)))} />
                <label title="Each row shows the max speed's 95 % margin of error">Margin of error</label>
                <input type="checkbox" checked={s.show_margin} onChange={(e) => set("show_margin", e.target.checked)} />
              </>}
            </div>
          </section>

          <section className="panel">
            <h3>Data</h3>
            <div className="export-grid">
              <a className="button" href={api.fileUrl(project.id, "results.csv")}><span className="dl-ico">↓</span>results.csv</a>
              <a className="button" href={api.fileUrl(project.id, "results.json")}><span className="dl-ico">↓</span>results.json</a>
              {res?.violations && <a className="button" href={api.fileUrl(project.id, "violations.csv")}><span className="dl-ico">↓</span>violations.csv</a>}
              {res?.render && <a className="button" href={api.fileUrl(project.id, "render.mp4")}><span className="dl-ico">↓</span>render.mp4</a>}
            </div>
            <p className="footnote">Earlier renders and results stay downloadable on the Jobs page.</p>
          </section>
        </aside>
      </div>
    </div>
  );
}
