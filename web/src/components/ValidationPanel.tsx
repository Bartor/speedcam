import { useEffect, useState } from "react";
import Help from "./Help";
import { api, fmt, Reference, Results, Validation, Vehicle } from "../api";

interface Props { projectId: string; res: Results; selected: Vehicle | undefined; tNow: number }

interface Draft { track_id: string; t_s: string; speed_kmh: string; metric: "avg" | "max"; tolerance_kmh: string; source: string; note: string }

const toDraft = (r: Reference): Draft => ({
  track_id: r.track_id != null ? String(r.track_id) : "", t_s: r.t_s != null ? String(r.t_s) : "",
  speed_kmh: String(r.speed_kmh), metric: r.metric, tolerance_kmh: String(r.tolerance_kmh), source: r.source, note: r.note,
});

const num = (s: string) => (s.trim() === "" || !Number.isFinite(+s) ? null : +s);

const VERDICT: Record<string, { tone: string; label: string }> = {
  consistent: { tone: "ok", label: "margin holds" },
  margin_too_small: { tone: "bad", label: "margin too small" },
  bias: { tone: "warn", label: "bias" },
  too_few: { tone: "", label: "too few" },
  none: { tone: "", label: "no matches" },
};

/** Validation run: known reference speeds (GPS logger, cruise control) vs. the measured vehicles. */
export default function ValidationPanel({ projectId, res, selected, tNow }: Props) {
  const [val, setVal] = useState<Validation | null>(null);
  const [rows, setRows] = useState<Draft[]>([]);
  const [dirty, setDirty] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // the report is evaluated against the current results, so reload it after a recompute
  useEffect(() => {
    api.validation(projectId).then((v) => { setVal(v); setRows(v.references.map(toDraft)); setDirty(false); })
      .catch((e) => setErr(String(e)));
  }, [projectId, res]);

  const edit = (i: number, patch: Partial<Draft>) => { setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r))); setDirty(true); };
  const add = (d: Partial<Draft>) => {
    setRows([...rows, { track_id: "", t_s: "", speed_kmh: "", metric: "avg", tolerance_kmh: "0.5", source: "GPS logger", note: "", ...d }]);
    setDirty(true);
  };

  const save = async () => {
    const refs: Reference[] = [];
    for (const r of rows) {
      const speed = num(r.speed_kmh);
      if (speed === null) { setErr("Every reference needs a speed (km/h)."); return; }
      if (num(r.track_id) === null && num(r.t_s) === null) { setErr("Every reference needs a track # or a time."); return; }
      refs.push({ speed_kmh: speed, track_id: num(r.track_id), t_s: num(r.t_s), metric: r.metric,
        tolerance_kmh: num(r.tolerance_kmh) ?? 0.5, source: r.source, note: r.note });
    }
    try {
      const v = await api.putValidation(projectId, refs);
      setVal(v); setRows(v.references.map(toDraft)); setDirty(false); setErr(null);
    } catch (e) { setErr(String(e)); }
  };

  const s = val?.report.summary;
  const verdict = s && !dirty ? VERDICT[s.verdict] : undefined;
  const reportRows = dirty ? [] : val?.report.rows ?? [];

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Validation run<Help topic="results.validation" /></h3>
        {verdict && <span className={`tag ${verdict.tone}`}>{verdict.label}</span>}
        <span className="grow" />
        <button className="ghost" onClick={() => add({ t_s: tNow.toFixed(2) })} title="A reference matched by time: the vehicle between the lines at the current video time">+ at {tNow.toFixed(1)} s</button>
        <button className="ghost" disabled={!selected} onClick={() => selected && add({ track_id: String(selected.track_id) })}>
          + vehicle {selected ? `#${selected.track_id}` : ""}</button>
      </div>
      <p className="footnote">
        Drive (or log) vehicles at known speeds through the section, then enter each reference here: a GPS logger's average
        over A→B, or a cruise-control speed. About 95 % of reference speeds should land inside the declared margin; fewer means
        the margin is too small. Use 20+ runs at several speeds and both directions.
      </p>
      {rows.length > 0 && (
        <div className="table-wrap validation-wrap">
          <table className="vehicles validation">
            <thead>
              <tr>
                <th className="num">Track #</th><th className="num" title="Real time in the clip (s), while the vehicle is between the lines">or time s</th>
                <th className="num">Reference km/h</th><th>Compare</th><th className="num" title="95 % uncertainty of the reference itself">Ref ± km/h</th>
                <th>Source</th><th className="num">Measured</th><th className="num">Error</th><th>Inside ±?</th><th />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const rep = reportRows[i];
                return (
                  <tr key={i}>
                    <td><input className="cell-in" type="number" value={r.track_id} onChange={(e) => edit(i, { track_id: e.target.value })} /></td>
                    <td><input className="cell-in" type="number" step="any" value={r.t_s} onChange={(e) => edit(i, { t_s: e.target.value })} /></td>
                    <td><input className="cell-in" type="number" step="any" value={r.speed_kmh} onChange={(e) => edit(i, { speed_kmh: e.target.value })} /></td>
                    <td><select value={r.metric} onChange={(e) => edit(i, { metric: e.target.value as "avg" | "max" })}>
                      <option value="avg">average A→B</option><option value="max">max</option></select></td>
                    <td><input className="cell-in" type="number" step="any" value={r.tolerance_kmh} onChange={(e) => edit(i, { tolerance_kmh: e.target.value })} /></td>
                    <td><input className="cell-in wide" type="text" value={r.source} onChange={(e) => edit(i, { source: e.target.value })} /></td>
                    <td className="num">{rep?.measured_kmh !== undefined ? <>{fmt(rep.measured_kmh)} <span className="pm">#{rep.track_id}</span></> : "–"}</td>
                    <td className="num">{rep?.error_kmh !== undefined ? `${rep.error_kmh >= 0 ? "+" : ""}${fmt(rep.error_kmh, 2)} (${fmt(rep.error_pct, 1)} %)` : ""}</td>
                    <td title={rep?.bound_low_kmh !== undefined ? `allowed error ${fmt(rep.bound_low_kmh, 2)} … +${fmt(rep.bound_high_kmh, 2)} km/h` : rep?.note}>
                      {rep?.inside === undefined ? <span className="muted small">{rep?.note || (dirty ? "save to check" : "")}</span>
                        : rep.inside ? <span className="status ok">✓ inside</span> : <span className="status bad">✕ outside</span>}
                      {rep?.note && rep.inside !== undefined && <span className="status warn" title={rep.note}> ⚠</span>}
                    </td>
                    <td><button className="x" title="Remove" onClick={() => { setRows(rows.filter((_, j) => j !== i)); setDirty(true); }}>×</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {s && !dirty && s.n_matched > 0 && (
        <div className="tiles">
          <Tile label="Inside the margin" value={s.n_with_margin ? `${s.n_inside} / ${s.n_with_margin}` : "–"}
            note={s.coverage !== undefined ? `${Math.round(100 * s.coverage)} % · expect ~95 %` : "results lack margins: recompute"} />
          <Tile label="Bias (measured − reference)" value={`${(s.bias_pct ?? 0) >= 0 ? "+" : ""}${fmt(s.bias_pct, 2)} %`} note={`${fmt(s.bias_kmh, 2)} km/h`} />
          <Tile label="Spread (SD)" value={s.sd_pct !== undefined ? `${fmt(s.sd_pct, 2)} %` : "–"} note={s.sd_kmh !== undefined ? `${fmt(s.sd_kmh, 2)} km/h` : "needs ≥2"} />
          <Tile label="Worst error" value={`${fmt(s.max_abs_pct, 2)} %`} note={s.z_rms !== undefined ? `errors / declared σ: ${fmt(s.z_rms, 2)} rms` : ""} />
        </div>
      )}
      {s && !dirty && s.n_references > 0 && <div className={`banner small ${s.verdict === "margin_too_small" ? "error" : s.verdict === "bias" ? "warn" : "info"}`}>{s.message}</div>}
      <div className="row end">
        {rows.length === 0 && <span className="muted small grow">No reference speeds yet.</span>}
        <button className="ghost" onClick={() => add({})}>+ reference</button>
        <button className="primary" disabled={!dirty} onClick={save}>Save & check</button>
      </div>
      {err && <div className="banner error small">{err}</div>}
    </section>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}
