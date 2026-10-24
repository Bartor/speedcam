import { useState } from "react";
import Help from "./Help";
import { api, fmt, Project, Results } from "../api";

interface Props { project: Project; res: Results; onRecomputed: (r: Results) => void; onProject: (p: Project) => void }

/** Project-level margin of error: what it covers, and the inputs it is built from. */
export default function MarginPanel({ project, res, onRecomputed, onProject }: Props) {
  const us = (project.config as any).uncertainty ?? {};
  const [f, setF] = useState({
    tol_cm: String(Math.round((us.distance_tolerance_m ?? 0.05) * 1000) / 10),
    px: String(us.click_tolerance_px ?? 0.5),
    height: us.drone_height_m != null ? String(us.drone_height_m) : "",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const u = res.uncertainty;
  const valid = res.vehicles.filter((v) => v.valid);

  const apply = async () => {
    setBusy(true);
    setErr(null);
    try {
      const p = await api.patch(project.id, {
        uncertainty: {
          distance_tolerance_m: +f.tol_cm / 100, click_tolerance_px: +f.px,
          drone_height_m: f.height.trim() === "" ? null : +f.height,
        },
      });
      onProject(p);
      onRecomputed(await api.recompute(project.id));
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Margin of error<Help topic="results.margin" /></h3>
      </div>
      {!u ? (
        <div className="banner info small">These results predate the margin of error. Press <b>Recompute speeds</b> to add it.</div>
      ) : (
        <>
          <table className="kv margin-kv"><tbody>
            <tr><td>Typical margin (median, valid)</td><td className="tnum">±{fmt(u.median_u_pct, 2)} %</td></tr>
            <tr><td>Calibration, A–B distance</td><td className="tnum">±{fmt(u.line_gap_rel_pct, 2)} %</td></tr>
            <tr><td title="How well the redundant check distances agree, relative to the tolerances you declared. Above 1 the calibration term is widened by this factor.">Check-distance fit ratio</td>
              <td className="tnum">{u.fit_ratio === null ? "no check distance" : `${fmt(u.fit_ratio, 2)}${u.fit_inflation > 1 ? " (term widened)" : ""}`}</td></tr>
            <tr><td>Height parallax</td><td>{u.parallax === "bounded" ? "not corrected, bounded" : "not corrected, not bounded"}</td></tr>
          </tbody></table>
          {u.fit_ratio === null && <div className="banner warn small">No redundant distance checks the calibration, so a wrong tape value or a misclick isn't caught and isn't in the margin. Add a check distance.</div>}
          {u.parallax === "unknown" && <div className="banner warn small">Height parallax is not corrected and the drone height is unknown, so its bias (measured speeds too high, ~+0.8 % for a car at 100 m, ~+3 % for a bus at 50 m) is <b>not</b> in the margin. Enter the drone height.</div>}
        </>
      )}
      <h4 className="subhead">Inputs</h4>
      <div className="form-grid compact">
        <label title="Standard uncertainty of each typed calibration distance: tape ~2 cm, orthophoto ~10 cm">Distance tolerance ± cm</label>
        <input type="number" step="any" min={0} value={f.tol_cm} onChange={(e) => setF({ ...f, tol_cm: e.target.value })} />
        <label title="Standard uncertainty of each clicked calibration point">Click precision ± px</label>
        <input type="number" step="any" min={0} value={f.px} onChange={(e) => setF({ ...f, px: e.target.value })} />
        <label title="Drone height above the road, from the flight log. Bounds the height-parallax bias.">Drone height m</label>
        <input type="number" step="any" min={0} value={f.height} placeholder="unknown" onChange={(e) => setF({ ...f, height: e.target.value })} />
      </div>
      <div className="row end">
        <button className="primary" disabled={busy} onClick={apply}>{busy ? "Recomputing…" : "Apply & recompute"}</button>
      </div>
      {err && <div className="banner error small">{err}</div>}
      <p className="footnote">
        Per vehicle, 95 % (k=2): calibration (Monte Carlo over the tolerances above, {u?.mc_samples ?? "–"} refits) and noise,
        combined root-sum-square; the parallax bound widens the lower side only. {valid.length ? "Hover a speed for its breakdown." : ""}
        {" "}Only a validation run against reference speeds proves the margin.
      </p>
    </section>
  );
}
