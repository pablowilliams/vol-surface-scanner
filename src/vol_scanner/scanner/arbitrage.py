"""Static arbitrage scanner for volatility surfaces.

Three tests are performed on the fitted SVI parameter set:

Butterfly: Durrleman's g(k) must be non-negative for every log moneyness.
Calendar: total variance w(k, t) must be non-decreasing in t for every k.
Vertical: call price C(K) must be non-increasing in K, which reduces to
   d1 - d2 machinery applied to the SVI-implied vol slice. We detect
   violations by numerical differentiation of Black-Scholes call prices.

Each violation carries a severity in {low, medium, high} based on the gap
relative to configured bands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from ..svi.fit import SliceFit
from ..svi.parametric import durrleman_g, total_variance


@dataclass
class Violation:
    type: str
    severity: str
    strike: float
    tenor: float
    description: str
    magnitude: float


def _severity(magnitude: float, bands: dict) -> str:
    if magnitude >= bands["high"]:
        return "high"
    if magnitude >= bands["medium"]:
        return "medium"
    return "low"


def butterfly_violations(
    fits: list[SliceFit],
    forward: float,
    cfg: dict,
) -> list[Violation]:
    k_grid = np.linspace(-1.5, 1.5, int(cfg["k_grid_points"]))
    out: list[Violation] = []
    bands = cfg["butterfly"]["severity_bands"]
    tol = float(cfg["butterfly"]["tolerance"])
    for f in fits:
        g = durrleman_g(k_grid, f.params)
        mask = g < -tol
        if not np.any(mask):
            continue
        for j in np.where(mask)[0]:
            magnitude = float(-g[j])
            out.append(
                Violation(
                    type="butterfly",
                    severity=_severity(magnitude, bands),
                    strike=float(forward * np.exp(k_grid[j])),
                    tenor=float(f.tenor),
                    description=(
                        f"Durrleman g(k) equals {g[j]:.2e} at log moneyness "
                        f"{k_grid[j]:.2f}, below the zero floor"
                    ),
                    magnitude=magnitude,
                )
            )
    return out


def calendar_violations(
    fits: list[SliceFit],
    forward: float,
    cfg: dict,
) -> list[Violation]:
    """Detect non-monotonic total variance across tenor."""
    k_grid = np.linspace(-1.2, 1.2, int(cfg["k_grid_points"]))
    tenors = np.array([f.tenor for f in fits])
    order = np.argsort(tenors)
    tenors_sorted = tenors[order]
    fits_sorted = [fits[i] for i in order]
    w_mat = np.stack([total_variance(k_grid, f.params) for f in fits_sorted], axis=0)
    bands = cfg["calendar"]["severity_bands"]
    tol = float(cfg["calendar"]["tolerance"])
    out: list[Violation] = []
    for j, k in enumerate(k_grid):
        for i in range(1, len(tenors_sorted)):
            diff = w_mat[i, j] - w_mat[i - 1, j]
            if diff < -tol:
                magnitude = float(-diff)
                out.append(
                    Violation(
                        type="calendar",
                        severity=_severity(magnitude, bands),
                        strike=float(forward * np.exp(k)),
                        tenor=float(tenors_sorted[i]),
                        description=(
                            f"Total variance decreases from {w_mat[i-1, j]:.4f} at "
                            f"t equals {tenors_sorted[i-1]:.2f} to {w_mat[i, j]:.4f} "
                            f"at t equals {tenors_sorted[i]:.2f}"
                        ),
                        magnitude=magnitude,
                    )
                )
    return out


def _bs_call(forward: float, strike: float, vol: float, t: float) -> float:
    if vol <= 0 or t <= 0:
        return max(forward - strike, 0.0)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return float(forward * norm.cdf(d1) - strike * norm.cdf(d2))


def vertical_violations(
    fits: list[SliceFit],
    forward: float,
    cfg: dict,
) -> list[Violation]:
    k_grid = np.linspace(-1.2, 1.2, int(cfg["k_grid_points"]))
    strikes = forward * np.exp(k_grid)
    bands = cfg["vertical"]["severity_bands"]
    tol = float(cfg["vertical"]["tolerance"])
    out: list[Violation] = []
    for f in fits:
        w = total_variance(k_grid, f.params)
        iv = np.sqrt(np.clip(w, 1e-10, None) / max(f.tenor, 1e-6))
        prices = np.array(
            [_bs_call(forward, strikes[j], iv[j], f.tenor) for j in range(strikes.size)]
        )
        diff = np.diff(prices)
        mask = diff > tol
        for j in np.where(mask)[0]:
            magnitude = float(diff[j])
            out.append(
                Violation(
                    type="vertical",
                    severity=_severity(magnitude, bands),
                    strike=float(strikes[j + 1]),
                    tenor=float(f.tenor),
                    description=(
                        f"Call price increases by {diff[j]:.4f} between strikes "
                        f"{strikes[j]:.2f} and {strikes[j+1]:.2f}"
                    ),
                    magnitude=magnitude,
                )
            )
    return out


def scan_surface(
    fits: list[SliceFit],
    forward: float,
    cfg: dict,
) -> dict:
    butterflies = butterfly_violations(fits, forward, cfg)
    calendars = calendar_violations(fits, forward, cfg)
    verticals = vertical_violations(fits, forward, cfg)
    all_v = butterflies + calendars + verticals
    records = [asdict(v) for v in all_v]
    counts = {
        "butterfly": len(butterflies),
        "calendar": len(calendars),
        "vertical": len(verticals),
    }
    return {"violations": records, "counts": counts}
