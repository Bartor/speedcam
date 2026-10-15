"""Speed statistics with confidence intervals and the spot-speed sample-size formula."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

from .config import StatsSettings

# Standard-normal quantile of the percentile (ITE spot-speed formula constant U).
U = {"mean": 0.0, "p85": 1.04, "p95": 1.645}


def required_n(sd: float, permitted_error: float, confidence: float, which: str = "p85") -> int:
    """ITE spot-speed formula n = (S·K/E)² · (2 + U²) / 2; U = 0 gives the mean."""
    K = st.norm.ppf(0.5 + confidence / 2)
    u = U[which]
    return int(math.ceil((sd * K / permitted_error) ** 2 * (2 + u * u) / 2))


def _bootstrap_ci(x: np.ndarray, q: float, confidence: float, n_boot: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    est = np.percentile(samples, q, axis=1)
    a = (1 - confidence) / 2 * 100
    return float(np.percentile(est, a)), float(np.percentile(est, 100 - a))


def summarize(speeds, s: StatsSettings, n_total: int | None = None) -> dict:
    x = np.asarray([v for v in speeds if np.isfinite(v)], float)
    n = len(x)
    out: dict = {"n_valid": n, "n_total": n_total if n_total is not None else n,
                 "confidence": s.confidence, "permitted_error_kmh": s.permitted_error_kmh}
    if n == 0:
        return out
    out.update(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
               p50=float(np.percentile(x, 50)), p85=float(np.percentile(x, 85)),
               p95=float(np.percentile(x, 95)))
    if n >= 2:
        sd = float(x.std(ddof=1))
        tq = st.t.ppf(0.5 + s.confidence / 2, n - 1)
        half = tq * sd / math.sqrt(n)
        out.update(sd=sd, mean_ci=[out["mean"] - half, out["mean"] + half])
        out["p85_ci"] = list(_bootstrap_ci(x, 85, s.confidence))
        out["p95_ci"] = list(_bootstrap_ci(x, 95, s.confidence))
        out["required_n"] = {k: required_n(sd, s.permitted_error_kmh, s.confidence, k) for k in U}
    return out
