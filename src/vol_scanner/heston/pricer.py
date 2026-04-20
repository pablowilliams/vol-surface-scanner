"""Heston (1993) characteristic function and Carr Madan FFT call pricer.

We implement the characteristic function of log spot under the Heston model
and use the Carr and Madan (1999) fast Fourier transform inversion to price
European calls in O(N log N) across a log strike grid. The characteristic
function is taken in its so called "little Heston trap" form which is
analytically continuous and numerically stable, see Albrecher et al. (2007).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParamsRaw:
    """Raw Heston parameters.

    kappa: mean reversion speed.
    theta: long run variance.
    v0: initial instantaneous variance.
    rho: correlation between spot and variance Brownians.
    xi: vol of vol (sometimes written sigma_v).
    """

    kappa: float
    theta: float
    v0: float
    rho: float
    xi: float

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.v0, self.rho, self.xi])


def heston_char_fn(
    u: np.ndarray,
    t: float,
    r: float,
    q: float,
    p: HestonParamsRaw,
    s0: float,
) -> np.ndarray:
    """Heston characteristic function of log(S_T) under the little trap form."""
    kappa, theta, v0, rho, xi = p.kappa, p.theta, p.v0, p.rho, p.xi
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + (xi**2) * (iu + u * u))
    g2 = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    expdt = np.exp(-d * t)
    D = (kappa - rho * xi * iu - d) / (xi**2) * (
        (1.0 - expdt) / (1.0 - g2 * expdt)
    )
    C = (r - q) * iu * t + (kappa * theta / (xi**2)) * (
        (kappa - rho * xi * iu - d) * t
        - 2.0 * np.log(np.clip((1.0 - g2 * expdt) / (1.0 - g2), 1e-30, None))
    )
    return np.exp(C + D * v0 + iu * np.log(max(s0, 1e-12)))


def heston_call_fft(
    strikes: np.ndarray,
    t: float,
    r: float,
    q: float,
    p: HestonParamsRaw,
    s0: float,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    """Price European calls via Carr and Madan FFT on a log strike grid.

    We damp the payoff with exp(alpha * log K), compute the FFT of the damped
    call transform, then invert and linearly interpolate onto the requested
    strike vector.
    """
    lam = 2.0 * np.pi / (n * eta)
    b = n * lam / 2.0
    u = np.arange(n) * eta
    k_grid = -b + lam * np.arange(n)

    phi = heston_char_fn(u - 1j * (alpha + 1), t, r, q, p, s0)
    num = np.exp(-r * t) * phi
    denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
    psi = num / denom

    # Simpson weights for the first and last nodes.
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    x = np.exp(1j * b * u) * psi * eta * w
    y = np.fft.fft(x).real
    call_prices = np.exp(-alpha * k_grid) / np.pi * y

    log_strikes_target = np.log(np.clip(strikes, 1e-12, None))
    prices = np.interp(log_strikes_target, k_grid, call_prices)
    # Enforce lower bound C >= max(F e^{-r t} - K e^{-r t}, 0).
    forward = s0 * np.exp((r - q) * t)
    intrinsic = np.exp(-r * t) * np.maximum(forward - strikes, 0.0)
    return np.maximum(prices, intrinsic)
