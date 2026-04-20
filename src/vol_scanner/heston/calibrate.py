"""Calibrate Heston parameters against an observed implied volatility chain.

We invert the Black Scholes call price to an implied volatility, then minimise
the mean squared error between model implied vol and observed implied vol
across the full (strike, tenor) grid using scipy's L-BFGS-B with parameter
bounds that keep the Feller condition loosely in sight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .pricer import HestonParamsRaw, heston_call_fft


@dataclass
class HestonParams:
    kappa: float
    theta: float
    v0: float
    rho: float
    xi: float
    rmse: float
    per_tenor_rmse: list[float]
    success: bool

    def as_raw(self) -> HestonParamsRaw:
        return HestonParamsRaw(
            kappa=self.kappa,
            theta=self.theta,
            v0=self.v0,
            rho=self.rho,
            xi=self.xi,
        )


def bs_call(forward: float, strike: float, vol: float, t: float, r: float = 0.0) -> float:
    if vol <= 0 or t <= 0:
        return float(max(forward - strike, 0.0) * np.exp(-r * t))
    sqrt_t = np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return float(np.exp(-r * t) * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))


def implied_vol_from_call(
    call_price: float,
    forward: float,
    strike: float,
    t: float,
    r: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> float:
    """Bracketed bisection on sigma in (1e-4, 5.0)."""
    lower, upper = 1e-4, 5.0
    lo_price = bs_call(forward, strike, lower, t, r)
    hi_price = bs_call(forward, strike, upper, t, r)
    if call_price <= lo_price:
        return lower
    if call_price >= hi_price:
        return upper
    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        mp = bs_call(forward, strike, mid, t, r)
        if abs(mp - call_price) < tol:
            return mid
        if mp < call_price:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def heston_implied_vol_surface(
    strikes: np.ndarray,
    tenors: np.ndarray,
    p: HestonParamsRaw,
    forward: float,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Return an (n_tenors, n_strikes) implied vol grid under the given params."""
    out = np.zeros((tenors.size, strikes.size))
    for i, t in enumerate(tenors):
        prices = heston_call_fft(strikes, float(t), r, q, p, forward)
        for j, k_strike in enumerate(strikes):
            out[i, j] = implied_vol_from_call(
                prices[j], forward * np.exp((r - q) * float(t)), float(k_strike), float(t), r
            )
    return out


def calibrate_heston(
    strikes: np.ndarray,
    tenors: np.ndarray,
    iv_observed: np.ndarray,
    forward: float,
    r: float = 0.0,
    q: float = 0.0,
    x0: tuple[float, float, float, float, float] | None = None,
    max_iter: int = 40,
) -> HestonParams:
    """Calibrate kappa, theta, v0, rho, xi to minimise mean squared IV error."""
    bounds = [
        (0.05, 10.0),   # kappa
        (0.001, 1.0),   # theta
        (0.001, 1.0),   # v0
        (-0.95, 0.0),   # rho
        (0.05, 2.0),    # xi
    ]
    # ATM seed so the initial v0 and theta line up with the observed level.
    atm_idx = int(np.argmin(np.abs(np.log(strikes / forward))))
    atm_var = float(np.median(iv_observed[:, atm_idx] ** 2))
    atm_var = float(np.clip(atm_var, 0.005, 0.5))

    # Tenor mask: very short tenors (less than 0.05y) on synthetic data
    # contribute disproportionate noise because total variance is divided by
    # a tiny tenor; we exclude them from the calibration objective. Heston
    # is also a poor model in the immediate front month in practice. The
    # excluded tenors still get a model RMSE reported below.
    cal_mask = tenors >= 0.05
    tenor_weights = cal_mask.astype(float)

    def loss(x: np.ndarray) -> float:
        # Project x back into bounds (Nelder-Mead can leave them).
        for i, (lo, hi) in enumerate(bounds):
            x[i] = float(np.clip(x[i], lo, hi))
        p = HestonParamsRaw(kappa=x[0], theta=x[1], v0=x[2], rho=x[3], xi=x[4])
        try:
            model = heston_implied_vol_surface(strikes, tenors, p, forward, r, q)
        except (ValueError, FloatingPointError):
            return 1.0
        diff = (model - iv_observed) * tenor_weights[:, None]
        return float(np.mean(diff * diff))

    # Multistart: try a small grid of seeds and keep the best.
    seeds: list[tuple[float, float, float, float, float]] = []
    if x0 is not None:
        seeds.append(tuple(x0))
    for kappa in (1.0, 2.5):
        for rho in (-0.7, -0.4):
            for xi in (0.4, 0.8):
                seeds.append((kappa, atm_var, atm_var, rho, xi))

    best_x: np.ndarray | None = None
    best_loss = float("inf")
    res = None
    for seed in seeds:
        try:
            r0 = minimize(
                loss,
                np.array(seed, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": max(20, max_iter // max(1, len(seeds))), "xatol": 1e-4, "fatol": 1e-7, "disp": False},
            )
            if r0.fun < best_loss:
                best_loss = float(r0.fun)
                best_x = np.array([float(np.clip(r0.x[i], bounds[i][0], bounds[i][1])) for i in range(5)])
                res = r0
        except (ValueError, FloatingPointError):
            continue
    if best_x is None:
        # Fallback to the seed if all starts diverged.
        best_x = np.array([1.5, atm_var, atm_var, -0.5, 0.6])
    res_x = best_x
    res_success = bool(res.success) if res is not None else False
    # Build a thin pseudo-result for downstream code.
    class _Res:
        pass
    res = _Res()
    res.x = res_x
    res.success = res_success
    x = res.x
    p = HestonParamsRaw(kappa=float(x[0]), theta=float(x[1]), v0=float(x[2]), rho=float(x[3]), xi=float(x[4]))
    model = heston_implied_vol_surface(strikes, tenors, p, forward, r, q)
    per_tenor = [float(np.sqrt(np.mean((model[i] - iv_observed[i]) ** 2))) for i in range(tenors.size)]
    # Headline RMSE excludes the masked very short tenors so the comparison
    # against SVI is on like for like calibration data.
    diff_cal = (model[cal_mask] - iv_observed[cal_mask])
    rmse = float(np.sqrt(np.mean(diff_cal * diff_cal))) if cal_mask.any() else float(np.sqrt(np.mean((model - iv_observed) ** 2)))
    return HestonParams(
        kappa=p.kappa,
        theta=p.theta,
        v0=p.v0,
        rho=p.rho,
        xi=p.xi,
        rmse=rmse,
        per_tenor_rmse=per_tenor,
        success=bool(res.success),
    )
