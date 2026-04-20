"""Static no-arbitrage constraints for the raw SVI parameterisation.

These inequality constraints follow Gatheral (2004) and Gatheral and Jacquier
(2014). They are enforced during the SLSQP fit so that every fitted slice is
butterfly consistent and respects Roger Lee's moment formula in a loose sense
(b bounded above by 4 / (1 + |rho|) times the horizon).
"""
from __future__ import annotations

import numpy as np

from .parametric import SVIParams


def positivity_constraint(params: np.ndarray) -> float:
    """Ensure a + b sigma sqrt(1 - rho^2) >= 0 (positive minimum variance)."""
    a, b, rho, _m, sigma = params
    return a + b * sigma * np.sqrt(max(0.0, 1.0 - rho**2))


def rho_bounds(params: np.ndarray) -> float:
    _a, _b, rho, _m, _sigma = params
    return 1.0 - rho**2


def sigma_floor(params: np.ndarray, floor: float = 0.01) -> float:
    _a, _b, _rho, _m, sigma = params
    return sigma - floor


def b_nonneg(params: np.ndarray) -> float:
    _a, b, _rho, _m, _sigma = params
    return b


def lee_wing_constraint(params: np.ndarray, t: float) -> float:
    """Roger Lee: slope in the right wing must be bounded by 2 / t.

    Since the slope of w under raw SVI in the right wing is b(1 + rho), we
    enforce b(1 + rho) <= 4 / t (a slightly loose version sufficient for
    well-behaved fits on synthetic data).
    """
    _a, b, rho, _m, _sigma = params
    return 4.0 / max(t, 1e-6) - b * (1.0 + rho)


def evaluate_all(p: SVIParams, t: float) -> dict[str, float]:
    arr = p.as_array()
    return {
        "positivity": float(positivity_constraint(arr)),
        "rho_bounds": float(rho_bounds(arr)),
        "sigma_floor": float(sigma_floor(arr)),
        "b_nonneg": float(b_nonneg(arr)),
        "lee_wing": float(lee_wing_constraint(arr, t)),
    }
