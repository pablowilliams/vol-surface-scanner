"""Deform the observed implied vol chain under four named scenarios.

Scenarios:

- vol_spike: multiply every implied vol by 1.30.
- crash_skew: amplify the left wing (k < 0) by a factor that grows in |k|.
- inversion: drag the long end tenors below the short end tenors.
- calm: compress every implied vol towards its ATM level.

For each scenario we re-fit SVI, then measure how many fitted parameters
drift outside of a "normal range" estimated from the baseline chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..svi.fit import SliceFit, fit_surface


@dataclass
class StressReport:
    scenarios: dict[str, dict] = field(default_factory=dict)
    drift_summary: dict[str, int] = field(default_factory=dict)


def _vol_spike(iv: np.ndarray) -> np.ndarray:
    return iv * 1.30


def _crash_skew(iv: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    out = iv.copy()
    for j, k in enumerate(k_grid):
        if k < 0:
            out[:, j] = out[:, j] * (1.0 + 0.6 * abs(k))
    return out


def _inversion(iv: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    out = iv.copy()
    # Lift short tenors and depress long tenors.
    for i, t in enumerate(tenors):
        if t < 0.5:
            out[i, :] = out[i, :] * 1.35
        else:
            out[i, :] = out[i, :] * 0.85
    return out


def _calm(iv: np.ndarray) -> np.ndarray:
    out = iv.copy()
    for i in range(out.shape[0]):
        mu = float(np.mean(out[i, :]))
        out[i, :] = mu + 0.35 * (out[i, :] - mu)
    return out


def _param_range(fits: list[SliceFit]) -> dict[str, tuple[float, float]]:
    a = np.array([f.params.a for f in fits])
    b = np.array([f.params.b for f in fits])
    rho = np.array([f.params.rho for f in fits])
    m = np.array([f.params.m for f in fits])
    sigma = np.array([f.params.sigma for f in fits])
    return {
        "a": (float(a.min()), float(a.max())),
        "b": (float(b.min()), float(b.max())),
        "rho": (float(rho.min()), float(rho.max())),
        "m": (float(m.min()), float(m.max())),
        "sigma": (float(sigma.min()), float(sigma.max())),
    }


def _count_drift(fits: list[SliceFit], ranges: dict[str, tuple[float, float]]) -> int:
    drift = 0
    for f in fits:
        for key in ("a", "b", "rho", "m", "sigma"):
            lo, hi = ranges[key]
            val = float(getattr(f.params, key))
            pad = max(0.05, 0.15 * (hi - lo))
            if val < lo - pad or val > hi + pad:
                drift += 1
    return drift


def run_stress_scenarios(
    k_grid: np.ndarray,
    iv: np.ndarray,
    tenors: np.ndarray,
    svi_cfg: dict,
    baseline_fits: list[SliceFit] | None = None,
) -> StressReport:
    if baseline_fits is None:
        baseline_fits = fit_surface(k_grid, iv, tenors, svi_cfg)
    ranges = _param_range(baseline_fits)

    scenarios = {
        "vol_spike": _vol_spike(iv),
        "crash_skew": _crash_skew(iv, k_grid),
        "inversion": _inversion(iv, tenors),
        "calm": _calm(iv),
    }

    out: dict[str, dict] = {}
    drift: dict[str, int] = {}
    for name, iv_new in scenarios.items():
        fits = fit_surface(k_grid, iv_new, tenors, svi_cfg)
        n_drift = _count_drift(fits, ranges)
        drift[name] = int(n_drift)
        out[name] = {
            "n_slices": len(fits),
            "n_drift": int(n_drift),
            "mean_rmse": float(np.mean([f.rmse for f in fits])),
            "params": [
                {
                    "tenor": float(f.tenor),
                    "a": float(f.params.a),
                    "b": float(f.params.b),
                    "rho": float(f.params.rho),
                    "m": float(f.params.m),
                    "sigma": float(f.params.sigma),
                    "rmse": float(f.rmse),
                }
                for f in fits
            ],
        }

    return StressReport(scenarios=out, drift_summary=drift)
