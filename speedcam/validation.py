"""Validation run: measured speeds vs. known reference speeds (GPS logger, cruise control).

The declared margin of error is a model; only reference speeds can show it is honest. For each
reference the matching measured vehicle is found, and the report gives the bias, the spread and
the coverage: the share of reference speeds inside the declared ± (about 95 % if it's right).
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel
from scipy import stats as st

COVERAGE = 0.95
MIN_N = 5  # fewer references than this can't say much about a 95 % interval


class Reference(BaseModel):
    """A vehicle whose true speed is known. Match it by track id, or by when it passed."""

    speed_kmh: float
    t_s: Optional[float] = None  # real time (s) in the clip while the vehicle is between the lines
    track_id: Optional[int] = None
    direction: Optional[Literal["A->B", "B->A"]] = None
    metric: Literal["avg", "max"] = "avg"  # GPS average over the section, or the peak speed
    tolerance_kmh: float = 0.5  # 95 % uncertainty of the reference itself (GPS ~0.2-0.5, cruise control ~1-2)
    source: str = ""
    note: str = ""


def match(ref: Reference, vehicles: list[dict]) -> tuple[Optional[dict], str]:
    """The measured vehicle a reference refers to, and a note ('' or why the match is doubtful)."""
    if ref.track_id is not None:
        v = next((v for v in vehicles if v["track_id"] == ref.track_id), None)
        return v, "" if v else f"no measured vehicle #{ref.track_id}"
    if ref.t_s is None:
        return None, "give a track # or a time"
    cand = [v for v in vehicles if v["t_entry_s"] - 1.0 <= ref.t_s <= v["t_exit_s"] + 1.0
            and (ref.direction is None or v["direction"] == ref.direction)]
    if not cand:
        return None, f"no vehicle between the lines at {ref.t_s:.1f} s"
    cand.sort(key=lambda v: abs((v["t_entry_s"] + v["t_exit_s"]) / 2 - ref.t_s))
    note = f"{len(cand)} vehicles at that time; nearest picked, give the track # to be sure" if len(cand) > 1 else ""
    return cand[0], note


def evaluate(refs: list[Reference], vehicles: list[dict]) -> dict:
    rows, errs, errs_pct, z, inside = [], [], [], [], []
    for i, ref in enumerate(refs):
        v, note = match(ref, vehicles)
        row: dict = {"index": i, "track_id": v["track_id"] if v else None, "note": note}
        if v is None:
            rows.append(row)
            continue
        key = "speed_max" if ref.metric == "max" else "speed"
        measured = v[f"{key}_kmh"]
        ukey = "speed_max" if ref.metric == "max" else "speed_avg"
        u, u_lo = v.get(f"{ukey}_u_kmh"), v.get(f"{ukey}_u_lower_kmh")
        err = measured - ref.speed_kmh
        row.update(measured_kmh=measured, error_kmh=err, error_pct=100 * err / ref.speed_kmh, valid=v["valid"])
        if not v["valid"]:
            row["note"] = (row["note"] + "; " if row["note"] else "") + f"vehicle flagged invalid ({v['flags']})"
        if u is not None:
            # combined with the reference's own uncertainty; the upper bound of the error also
            # allows the one-sided parallax bias (measured speeds can only be too high from it)
            up = math.hypot(u, ref.tolerance_kmh)
            hi = up + (u_lo - u if u_lo is not None else 0.0)
            ok = -up <= err <= hi
            row.update(u_kmh=u, u_lower_kmh=u_lo, bound_low_kmh=-up, bound_high_kmh=hi, inside=ok)
            inside.append(ok)
            side = hi if err > 0 else up  # the one-sided parallax allowance only applies above
            z.append(err / (side / 2) if side > 0 else 0.0)
        errs.append(err)
        errs_pct.append(row["error_pct"])
        rows.append(row)
    return {"rows": rows, "summary": _summary(errs, errs_pct, z, inside, len(refs))}


def _summary(errs, errs_pct, z, inside, n_refs) -> dict:
    n = len(errs)
    out: dict = {"n_references": n_refs, "n_matched": n, "expected_coverage": COVERAGE}
    if n == 0:
        out["verdict"] = "none"
        out["message"] = "No reference matched a measured vehicle yet."
        return out
    e, ep = np.array(errs), np.array(errs_pct)
    out.update(bias_kmh=float(e.mean()), bias_pct=float(ep.mean()),
               rms_pct=float(np.sqrt((ep ** 2).mean())), max_abs_pct=float(np.abs(ep).max()))
    if n >= 2:
        out["sd_kmh"] = float(e.std(ddof=1))
        out["sd_pct"] = float(ep.std(ddof=1))
    if inside:
        k = int(sum(inside))
        out.update(n_inside=k, n_with_margin=len(inside), coverage=k / len(inside),
                   z_rms=float(np.sqrt(np.mean(np.square(z)))))
        # probability of seeing this few inside if the margin really covered 95 %
        out["p_coverage"] = float(st.binom.cdf(k, len(inside), COVERAGE))
    if n < MIN_N:
        out["verdict"] = "too_few"
        out["message"] = (f"{n} reference{'s' if n != 1 else ''}: too few for a verdict. Use at least {MIN_N}, "
                          "ideally 20+ runs at several speeds and both directions.")
    elif inside and out["p_coverage"] < 0.05:
        out["verdict"] = "margin_too_small"
        out["message"] = (f"Only {out['n_inside']} of {out['n_with_margin']} reference speeds fall inside the "
                          "declared margin; 95 % should. The margin is too small for this setup: re-check the "
                          "calibration distances and tolerances, or widen the declared tolerances.")
    elif n >= 2 and abs(out["bias_pct"]) > 0.5 and abs(out["bias_kmh"]) > 2 * out["sd_kmh"] / math.sqrt(n):
        out["verdict"] = "bias"
        out["message"] = (f"Measured speeds are {abs(out['bias_pct']):.1f}% {'high' if out['bias_kmh'] > 0 else 'low'} "
                          "on average (beyond chance). A shared bias points at the calibration scale or height "
                          "parallax.")
    else:
        out["verdict"] = "consistent"
        out["message"] = ("Reference speeds agree with the measurements within the declared margin."
                          + (" The margin looks generous (errors well inside it)." if inside and out["z_rms"] < 0.5
                             else ""))
    return out
