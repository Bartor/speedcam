import { useEffect, useState } from "react";
import Help from "./Help";
import { api, fmt, margin, marginBreakdown, MergeCandidates, MergePair, Results, Vehicle, VehicleLook } from "../api";

const clock = (s: number) => {
  const m = Math.floor(s / 60), x = s - m * 60;
  return `${m}:${x.toFixed(1).padStart(4, "0")}`;
};
const dirText = (d: string) => d.replace("->", " → ");
/** a vehicle's picture; changes when its tracks are merged or split */
export const thumbUrl = (pid: string, v: Pick<Vehicle, "track_id" | "merged">) =>
  api.vehicleThumbUrl(pid, v.track_id, (v.merged ?? []).join("-"));

const FLAG_TEXT: Record<string, string> = {
  gap: "detection gap (average inconclusive)", residual: "position jump (average inconclusive)", accel: "implausible acceleration",
  few_points: "too few detections", disagree: "line and path speeds disagree", multi_cross: "crossed a line more than once",
  no_clean: "too little clean tracking for a max",
};
export const flagsText = (flags: string) => flags.split(";").filter(Boolean).map((f) => FLAG_TEXT[f] ?? f).join(", ");

export type SortBy = "time" | "max" | "name";

interface ListProps {
  projectId: string; res: Results; rows: Vehicle[]; selected: number | null; tNow: number; limit: number;
  onPick: (v: Vehicle) => void;
  dir: string; setDir: (d: "all" | "A->B" | "B->A") => void;
  road: string; setRoad: (r: string) => void; roads: string[];
  validOnly: boolean; setValidOnly: (b: boolean) => void;
  sortBy: SortBy; setSortBy: (s: SortBy) => void;
  empty?: React.ReactNode;
}

/** The measured vehicles beside the player: picture, name, max speed, what's wrong with it. */
export function VehicleList({ projectId, res, rows, selected, tNow, limit, onPick, dir, setDir, road, setRoad, roads,
  validOnly, setValidOnly, sortBy, setSortBy, empty }: ListProps) {
  useEffect(() => {
    if (selected !== null) document.getElementById(`vl-${selected}`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  return (
    <section className="panel dash-list">
      <div className="panel-head">
        <h3>Vehicles<Help topic="results.vehicles" /></h3>
        <span className="tag">{rows.length}{rows.length !== res.vehicles.length ? ` of ${res.vehicles.length}` : ""}</span>
        <span className="grow" />
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortBy)} aria-label="Order" title="Order">
          <option value="time">in order seen</option><option value="max">fastest first</option><option value="name">by name</option>
        </select>
      </div>
      <div className="vl-filters">
        <div className="seg-ctl">
          {(["all", "A->B", "B->A"] as const).map((d) => <button key={d} className={dir === d ? "active" : ""} onClick={() => setDir(d)}>{d === "all" ? "All" : dirText(d)}</button>)}
        </div>
        {roads.length > 0 && (
          <select value={road} onChange={(e) => setRoad(e.target.value)} title="Path of motion (road)">
            <option value="all">All paths</option>
            {roads.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        )}
        <label className="small"><input type="checkbox" checked={validOnly} onChange={(e) => setValidOnly(e.target.checked)} /> valid only</label>
      </div>
      <div className="vl-scroll">
        <div className="vl-inner">
          {rows.map((v) => {
            const active = tNow >= v.t_entry_s - 0.5 && tNow <= v.t_exit_s + 0.5;
            const over = v.valid && v.speed_max_kmh > limit;
            return (
              <button key={v.track_id} id={`vl-${v.track_id}`} onClick={() => onPick(v)}
                className={`vl-row${selected === v.track_id ? " selected" : ""}${active ? " active" : ""}${v.valid ? "" : " invalid"}`}>
                <img className="vl-thumb" loading="lazy" src={thumbUrl(projectId, v)} alt="" />
                <span className="vl-who">
                  <b>{v.name || `#${v.track_id} ${v.class}`}</b>
                  <span>{v.name ? `#${v.track_id} ${v.class} · ` : ""}{dirText(v.direction)}{v.path && roads.length > 1 ? ` · ${v.path}` : ""} · {clock(v.t_entry_s)}</span>
                </span>
                <span className="vl-flags">
                  {(v.merged?.length ?? 0) > 1 && <span title={`Measured as one vehicle from tracks ${v.merged!.map((t) => `#${t}`).join(" + ")}`}>⧉</span>}
                  {(v.violations?.length ?? 0) > 0 && <span className="bad-text" title={v.violations!.join("\n")}>⛔</span>}
                  {!v.valid && <span className="bad-text" title={flagsText(v.flags)}>✕</span>}
                  {v.valid && v.avg_valid === false && <span className="warn-text" title={`Average inconclusive: ${flagsText(v.flags)}. The max stands.`}>~</span>}
                </span>
                <span className="vl-speed" title={marginBreakdown(v, "max")}>
                  <b className={over ? "bad-text" : ""}>{fmt(v.speed_max_kmh)}</b>
                  <small>{v.avg_valid === false ? "avg –" : `avg ${fmt(v.speed_kmh)}`}</small>
                </span>
              </button>
            );
          })}
          {rows.length === 0 && (empty ?? <p className="empty-note">No vehicles match the filter.</p>)}
        </div>
      </div>
    </section>
  );
}

interface CardProps {
  projectId: string; res: Results; v: Vehicle; cands: MergeCandidates | null; busy: string | null;
  onNamed: (tid: number, name: string) => void;
  onMerge: (ids: number[]) => void; onSplit: (tid: number) => void;
  onSeek: (t: number) => void;
}

/** The selected vehicle: its picture and name (with a suggestion from its colour), the facts, and
 * merging with a track the tracker split off from it. */
export function VehicleCard({ projectId, res, v, cands, busy, onNamed, onMerge, onSplit, onSeek }: CardProps) {
  const [name, setName] = useState(v.name ?? "");
  const [look, setLook] = useState<VehicleLook | null>(null);
  const [other, setOther] = useState("");
  useEffect(() => { setName(v.name ?? ""); }, [v.track_id, v.name]);
  useEffect(() => {
    setLook(null);
    let live = true;
    api.vehicleLook(projectId, v.track_id).then((l) => live && setLook(l)).catch(() => {});
    return () => { live = false; };
  }, [projectId, v.track_id, (v.merged ?? []).join("-")]);  // eslint-disable-line react-hooks/exhaustive-deps

  const save = (n: string) => {
    const clean = n.trim().replace(/\s+/g, " ");
    if (clean !== (v.name ?? "")) onNamed(v.track_id, clean);
  };
  const ids = new Set(v.merged?.length ? v.merged : [v.track_id]);
  // likely continuations, and weaker ones only if they're close (a far-off one is another vehicle)
  const pairs = (cands?.pairs ?? []).filter((p) => (ids.has(p.a) || ids.has(p.b)) && (p.likely || p.miss_m <= 5));
  const evs = v.violations ?? [];
  const mergedWith = (v.merged ?? []).filter((t) => t !== v.track_id);

  return (
    <section className="panel vcard">
      <div className="panel-head">
        <h3>Vehicle<Help topic="results.vehicle" /></h3>
        <span className="tag">#{v.track_id} {v.class}</span>
        {!v.valid && <span className="tag bad" title={flagsText(v.flags)}>not valid</span>}
      </div>
      <div className="vcard-head">
        <img className="vcard-pic" src={thumbUrl(projectId, v)} alt={`Vehicle ${v.track_id}`} />
        <div className="vcard-name">
          <input type="text" value={name} placeholder="Name it, e.g. white hatchback" aria-label="Vehicle name" maxLength={60}
            onChange={(e) => setName(e.target.value)} onBlur={(e) => save(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setName(v.name ?? ""); }} />
          {look && (
            <div className="vcard-suggest small">
              <span className="swatch" style={{ background: `rgb(${look.rgb.join(",")})` }} />
              looks {look.colour}
              {look.suggestion !== (v.name ?? "") && (
                <button className="chip" onClick={() => { setName(look.suggestion); onNamed(v.track_id, look.suggestion); }}
                  title="Name it by its colour and class (seen from above: the roof's colour)">Use “{look.suggestion}”</button>
              )}
            </div>
          )}
        </div>
      </div>
      <dl className="vcard-facts">
        <dt>Max</dt><dd title={marginBreakdown(v, "max")}><b className={v.valid ? "" : "muted"}>{fmt(v.speed_max_kmh)} km/h</b> {margin(v, "max")}</dd>
        <dt>Average A→B</dt><dd>{v.avg_valid === false ? <span className="muted">inconclusive</span> : <>{fmt(v.speed_kmh)} km/h {margin(v, "avg")}</>}</dd>
        <dt>Seen</dt><dd>
          <button className="link" onClick={() => onSeek(v.t_entry_s - 0.3)}>{clock(v.t_entry_s)}</button> →{" "}
          <button className="link" onClick={() => onSeek(v.t_exit_s - 0.3)}>{clock(v.t_exit_s)}</button>
          <span className="muted"> · {v.dt_s.toFixed(2)} s over {v.path_length_m.toFixed(1)} m</span></dd>
        <dt>Direction</dt><dd>{dirText(v.direction)}{v.path ? ` · ${v.path}` : ""}</dd>
        {v.flags && <><dt>Notes</dt><dd className={v.valid ? "warn-text" : "bad-text"}>{flagsText(v.flags)}</dd></>}
        {evs.length > 0 && <><dt>Violations</dt><dd className="bad-text">{evs.join(", ")}</dd></>}
      </dl>

      <div className="subhead">Same vehicle under another number?</div>
      {mergedWith.length > 0 && (
        <div className="vcard-merged small">
          <span>⧉ Measured as one vehicle from tracks {v.merged!.map((t) => `#${t}`).join(" + ")}</span>
          <button className="chip" disabled={!!busy} onClick={() => onSplit(v.track_id)} title="Measure the tracks separately again">Split</button>
        </div>
      )}
      {pairs.map((p) => <PairRow key={`${p.a}-${p.b}`} p={p} cands={cands!} self={ids} busy={busy} onMerge={onMerge} onSeek={onSeek} />)}
      {!cands && <p className="muted small">Looking for tracks that continue this one…</p>}
      {cands && pairs.length === 0 && mergedWith.length === 0 && (
        <p className="muted small">When the tracker loses a vehicle (behind a tree or a bus) it picks it up again under a new number. Tracks that
          continue this one show up here; or merge one by its number.</p>
      )}
      <div className="row small">
        <label htmlFor="merge-other">Merge with track #</label>
        <input id="merge-other" type="number" min={0} value={other} onChange={(e) => setOther(e.target.value)} style={{ width: 80 }} />
        <button disabled={!other || !!busy} onClick={() => { onMerge([v.track_id, +other]); setOther(""); }}>Merge</button>
      </div>
      {busy && <p className="small muted">{busy}</p>}
    </section>
  );
}

function PairRow({ p, cands, self, busy, onMerge, onSeek }: {
  p: MergePair; cands: MergeCandidates; self?: Set<number>; busy: string | null;
  onMerge: (ids: number[]) => void; onSeek: (t: number) => void;
}) {
  const desc = (id: number) => {
    const t = cands.tracks[String(id)];
    if (!t) return `#${id}`;
    return `${t.name || `#${id} ${t.cls}`}${t.measured ? ` (${fmt(t.speed_max_kmh)} km/h)` : " (not measured)"}`;
  };
  const otherFirst = self?.has(p.b);
  const gap = p.gap_s >= 0 ? `${p.gap_s.toFixed(1)} s after` : `${(-p.gap_s).toFixed(1)} s before`;
  return (
    <div className={`pair-row small${p.likely ? " likely" : ""}`}>
      <span>
        {self ? (otherFirst ? <>{desc(p.a)} ends {p.gap_s.toFixed(1)} s before this one starts</> : <>{desc(p.b)} starts {gap} this one ends</>)
          : <>{desc(p.a)} → {desc(p.b)} · starts {gap} it ends</>}
        <span className="muted"> · {p.miss_m.toFixed(1)} m from where it was heading, ~{p.speed_kmh.toFixed(0)} km/h{p.likely ? "" : " · maybe"}</span>
      </span>
      <span className="pair-actions">
        <button className="chip" onClick={() => onSeek(p.t_a_end_s - 1)} title="Watch the hand-over">Watch</button>
        <button className={`chip${p.likely ? " primary" : ""}`} disabled={!!busy} onClick={() => onMerge([p.a, p.b])}
          title={p.likely ? "Measure the two as one vehicle" : "Less likely the same vehicle: watch first"}>Merge</button>
      </span>
    </div>
  );
}

/** Likely splits in the whole clip, for a quick pass over them. */
export function SplitsPanel({ cands, busy, onMerge, onSeek }: {
  cands: MergeCandidates; busy: string | null; onMerge: (ids: number[]) => void; onSeek: (t: number) => void;
}) {
  const [all, setAll] = useState(false);
  const likely = cands.pairs.filter((p) => p.likely);
  if (!likely.length) return null;
  const SHOW = 6;
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Possibly split vehicles<Help topic="results.splits" /></h3>
        <span className="tag">{likely.length}</span>
        <span className="grow" />
        {likely.length > SHOW && <button className="chip" onClick={() => setAll(!all)}>{all ? "Fewer" : `Show all ${likely.length}`}</button>}
      </div>
      <div className={all ? "pairs-scroll" : ""}>
        {(all ? likely : likely.slice(0, SHOW)).map((p) => <PairRow key={`${p.a}-${p.b}`} p={p} cands={cands} busy={busy} onMerge={onMerge} onSeek={onSeek} />)}
      </div>
      {busy && <p className="small muted">{busy}</p>}
    </section>
  );
}
