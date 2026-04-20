"""Raw SVI parameterisation utilities.

The raw SVI total variance function is

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))

with k the log moneyness. Implied volatility for a slice of tenor t is then
sigma_imp(k, t) = sqrt(w(k) / t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> SVIParams:
        return cls(a=float(x[0]), b=float(x[1]), rho=float(x[2]), m=float(x[3]), sigma=float(x[4]))


def total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def implied_vol(k: np.ndarray, t: float, p: SVIParams) -> np.ndarray:
    w = total_variance(k, p)
    w = np.clip(w, 1e-10, None)
    return np.sqrt(w / max(t, 1e-6))


def w_derivatives(k: np.ndarray, p: SVIParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return w(k), w'(k), w''(k) in closed form."""
    dk = k - p.m
    r = np.sqrt(dk**2 + p.sigma**2)
    w = p.a + p.b * (p.rho * dk + r)
    wp = p.b * (p.rho + dk / r)
    wpp = p.b * (p.sigma**2) / (r**3)
    return w, wp, wpp


def durrleman_g(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Durrleman function g(k), non-negativity is the butterfly no-arb test."""
    w, wp, wpp = w_derivatives(k, p)
    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = (wp**2) / 4.0 * (1.0 / w + 0.25)
    return term1 - term2 + wpp / 2.0
