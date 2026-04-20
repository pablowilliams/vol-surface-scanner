"""Rolling 30 day backtest of violation count versus realised move magnitude.

The backtest is run on a synthetic 30 day rolling chain. For each historical
day t, we fit the SVI surface, scan for arbitrage violations, and then look
forward five business days to compute the absolute return of the forward
price; the correlation between violation count and realised move magnitude
is reported at lags 1 through 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.synthetic import CALM_PARAMS, STRESS_PARAMS, svi_total_variance
from ..scanner.arbitrage import scan_surface
from ..svi.fit import fit_surface


@dataclass
class BacktestResult:
    dates: list[str] = field(default_factory=list)
    forward_path: list[float] = field(default_factory=list)
    violation_counts: list[int] = field(default_factory=list)
    realised_moves: list[float] = field(default_factory=list)
    lead_lag: dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    n_days: int = 0


def _simulate_forward(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = 0.0
    sigma = 0.015
    rets = rng.normal(mu, sigma, n)
    regime_shocks = rng.uniform(size=n) < 0.10
    rets[regime_shocks] = rng.normal(0.0, 0.05, int(regime_shocks.sum()))
    return np.exp(np.cumsum(rets)) * 100.0


def rolling_backtest(
    svi_cfg: dict,
    scanner_cfg: dict,
    data_cfg: dict,
    n_days: int = 30,
    horizon: int = 5,
    seed: int = 2026,
) -> BacktestResult:
    """Run a synthetic rolling backtest.

    For each day we perturb the SVI mixture probability with a small cyclical
    term plus a random walk, then build an implied vol chain, fit it, scan it,
    and record the number of violations.
    """
    rng = np.random.default_rng(seed)
    n_strikes = int(data_cfg["n_strikes"])
    n_tenors = int(data_cfg["n_tenors"])
    k_grid = np.linspace(
        float(data_cfg["moneyness_min"]),
        float(data_cfg["moneyness_max"]),
        n_strikes,
    )
    tenors = np.linspace(
        float(data_cfg["tenor_min_years"]),
        float(data_cfg["tenor_max_years"]),
        n_tenors,
    )
    forward_path = _simulate_forward(n_days + horizon, seed)

    violation_counts: list[int] = []
    for day in range(n_days):
        stress_p = 0.25 + 0.15 * np.sin(2 * np.pi * day / 12.0) + 0.10 * rng.normal()
        stress_p = float(np.clip(stress_p, 0.05, 0.9))
        iv = np.zeros((n_tenors, n_strikes))
        for i, t in enumerate(tenors):
            params = STRESS_PARAMS if rng.random() < stress_p else CALM_PARAMS
            w = svi_total_variance(k_grid, params)
            noise = rng.normal(0.0, float(data_cfg["noise_sigma_iv"]), n_strikes)
            iv[i, :] = np.clip(np.sqrt(np.clip(w, 1e-8, None) / max(t, 1e-4)) + noise, 1e-3, None)
        fits = fit_surface(k_grid, iv, tenors, svi_cfg)
        scan = scan_surface(fits, float(forward_path[day]), scanner_cfg)
        violation_counts.append(int(len(scan["violations"])))

    realised_moves = [
        float(abs(np.log(forward_path[min(day + horizon, n_days + horizon - 1)] / forward_path[day])))
        for day in range(n_days)
    ]

    # Lead lag: cor(v_t, |r_{t+h}|) for h in 1..horizon.
    lead_lag: dict[int, float] = {}
    v = np.array(violation_counts, dtype=float)
    for h in range(1, horizon + 1):
        future = np.array(
            [abs(np.log(forward_path[min(t + h, n_days + horizon - 1)] / forward_path[t])) for t in range(n_days)]
        )
        if np.std(v) < 1e-9 or np.std(future) < 1e-9:
            lead_lag[h] = 0.0
        else:
            lead_lag[h] = float(np.corrcoef(v, future)[0, 1])

    # Hit rate: fraction of days where a top quartile violation count precedes
    # a top quartile realised move.
    if v.size > 4:
        v_thr = np.quantile(v, 0.75)
        r_thr = np.quantile(np.array(realised_moves), 0.75)
        hits = ((v >= v_thr) & (np.array(realised_moves) >= r_thr)).sum()
        hit_rate = float(hits) / max(1, int((v >= v_thr).sum()))
    else:
        hit_rate = 0.0

    return BacktestResult(
        dates=[f"t+{i:02d}" for i in range(n_days)],
        forward_path=[float(x) for x in forward_path[:n_days]],
        violation_counts=violation_counts,
        realised_moves=realised_moves,
        lead_lag=lead_lag,
        hit_rate=hit_rate,
        n_days=n_days,
    )
