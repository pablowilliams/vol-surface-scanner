"""Finite difference Greeks on the fitted SVI plus residual surface.

We compute delta, gamma, vega, theta, vanna and volga on a strike by tenor
grid by bumping the Black Scholes inputs and pricing vanilla calls under the
fitted implied vol, which is the SVI slice value plus the neural residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GreeksSurface:
    strikes: np.ndarray
    tenors: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray

    def to_payload(self) -> dict:
        return {
            "strikes": self.strikes.tolist(),
            "tenors": self.tenors.tolist(),
            "delta": self.delta.tolist(),
            "gamma": self.gamma.tolist(),
            "vega": self.vega.tolist(),
            "theta": self.theta.tolist(),
            "vanna": self.vanna.tolist(),
            "volga": self.volga.tolist(),
        }


def _bs_call(s: float, k: float, vol: float, t: float, r: float) -> float:
    if vol <= 0 or t <= 0:
        return float(max(s - k * np.exp(-r * t), 0.0))
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return float(s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))


def compute_greeks(
    strikes: np.ndarray,
    tenors: np.ndarray,
    iv_surface: np.ndarray,
    spot: float,
    r: float = 0.0,
    bump_s: float = 0.01,
    bump_v: float = 0.01,
    bump_t: float = 1e-3,
) -> GreeksSurface:
    n_t, n_k = iv_surface.shape
    delta = np.zeros_like(iv_surface)
    gamma = np.zeros_like(iv_surface)
    vega = np.zeros_like(iv_surface)
    theta = np.zeros_like(iv_surface)
    vanna = np.zeros_like(iv_surface)
    volga = np.zeros_like(iv_surface)

    ds = spot * bump_s
    dv = bump_v
    dt = bump_t

    for i in range(n_t):
        t = float(tenors[i])
        for j in range(n_k):
            k = float(strikes[j])
            v = float(iv_surface[i, j])
            c = _bs_call(spot, k, v, t, r)
            c_sp = _bs_call(spot + ds, k, v, t, r)
            c_sm = _bs_call(spot - ds, k, v, t, r)
            c_vp = _bs_call(spot, k, v + dv, t, r)
            c_vm = _bs_call(spot, k, max(v - dv, 1e-6), t, r)
            c_tp = _bs_call(spot, k, v, max(t - dt, 1e-6), r)
            c_sp_vp = _bs_call(spot + ds, k, v + dv, t, r)
            c_sp_vm = _bs_call(spot + ds, k, max(v - dv, 1e-6), t, r)
            c_sm_vp = _bs_call(spot - ds, k, v + dv, t, r)
            c_sm_vm = _bs_call(spot - ds, k, max(v - dv, 1e-6), t, r)

            delta[i, j] = (c_sp - c_sm) / (2.0 * ds)
            gamma[i, j] = (c_sp - 2.0 * c + c_sm) / (ds * ds)
            vega[i, j] = (c_vp - c_vm) / (2.0 * dv)
            theta[i, j] = -(c - c_tp) / dt
            vanna[i, j] = (c_sp_vp - c_sp_vm - c_sm_vp + c_sm_vm) / (4.0 * ds * dv)
            volga[i, j] = (c_vp - 2.0 * c + c_vm) / (dv * dv)

    return GreeksSurface(
        strikes=strikes.copy(),
        tenors=tenors.copy(),
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        vanna=vanna,
        volga=volga,
    )
