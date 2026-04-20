"""Euler Maruyama Monte Carlo pricer under the calibrated Heston SDE.

We simulate the joint log spot and variance SDE with a full truncation
Euler scheme (Lord, Koekkoek and Van Dijk 2010) to keep the variance
process non negative. The pricer returns European call prices for a
small book of five strikes at one common tenor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..heston.pricer import HestonParamsRaw


@dataclass
class MCResult:
    strikes: np.ndarray
    tenor: float
    mc_prices: np.ndarray
    mc_stderr: np.ndarray
    surface_prices: np.ndarray
    absolute_gap: np.ndarray
    n_paths: int


def simulate_paths(
    n_paths: int,
    n_steps: int,
    t: float,
    spot: float,
    r: float,
    q: float,
    p: HestonParamsRaw,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(2026)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    s = np.full(n_paths, spot, dtype=np.float64)
    v = np.full(n_paths, p.v0, dtype=np.float64)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(max(0.0, 1.0 - p.rho**2)) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        s = s * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp) * sqrt_dt * z1)
        v = v + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp) * sqrt_dt * z2
        v = np.maximum(v, 0.0)
    return s, v


def price_book_mc(
    strikes: np.ndarray,
    tenor: float,
    spot: float,
    r: float,
    q: float,
    p: HestonParamsRaw,
    surface_prices: np.ndarray,
    n_paths: int = 10_000,
    n_steps: int = 64,
    seed: int = 2026,
) -> MCResult:
    rng = np.random.default_rng(seed)
    s_t, _ = simulate_paths(n_paths, n_steps, tenor, spot, r, q, p, rng=rng)
    disc = np.exp(-r * tenor)
    prices = np.zeros(strikes.size)
    stderrs = np.zeros(strikes.size)
    for i, k in enumerate(strikes):
        payoff = np.maximum(s_t - k, 0.0)
        prices[i] = float(disc * payoff.mean())
        stderrs[i] = float(disc * payoff.std(ddof=1) / np.sqrt(n_paths))
    gap = np.abs(prices - surface_prices)
    return MCResult(
        strikes=strikes.copy(),
        tenor=float(tenor),
        mc_prices=prices,
        mc_stderr=stderrs,
        surface_prices=surface_prices.copy(),
        absolute_gap=gap,
        n_paths=n_paths,
    )
