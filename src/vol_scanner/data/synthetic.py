"""Synthetic option chain generator driven by a two-regime SVI mixture.

The generator is fully deterministic under a fixed seed so that the pipeline
and test suite can reproduce results exactly. Two SVI regimes are combined,
the calm regime dominating normal trading and the stressed regime occasionally
contaminating the surface with a steeper skew and richer term structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CALM_PARAMS = {
    "a": 0.010,
    "b": 0.040,
    "rho": -0.25,
    "m": 0.00,
    "sigma": 0.30,
}

STRESS_PARAMS = {
    "a": 0.030,
    "b": 0.110,
    "rho": -0.55,
    "m": -0.05,
    "sigma": 0.20,
}


def svi_total_variance(k: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return total variance w(k) under the raw SVI parameterisation."""
    a = params["a"]
    b = params["b"]
    rho = params["rho"]
    m = params["m"]
    sigma = params["sigma"]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


@dataclass
class SyntheticOptionChain:
    strikes: np.ndarray
    tenors: np.ndarray
    log_moneyness: np.ndarray
    implied_vol: np.ndarray
    regime_mask: np.ndarray
    forward: float
    atm_vol: np.ndarray
    meta: dict = field(default_factory=dict)


def generate_chain(config: dict) -> SyntheticOptionChain:
    """Generate a deterministic option chain."""
    rng = np.random.default_rng(config["seed"])

    n_strikes = int(config["n_strikes"])
    n_tenors = int(config["n_tenors"])
    forward = float(config["forward_price"])

    tenors = np.linspace(
        float(config["tenor_min_years"]),
        float(config["tenor_max_years"]),
        n_tenors,
    )
    k_grid = np.linspace(
        float(config["moneyness_min"]),
        float(config["moneyness_max"]),
        n_strikes,
    )

    iv = np.zeros((n_tenors, n_strikes))
    regime_mask = np.zeros(n_tenors, dtype=int)
    stress_p = float(config["stress_mixture_probability"])
    noise = float(config["noise_sigma_iv"])

    for i, t in enumerate(tenors):
        draw = rng.random()
        if draw < stress_p:
            params = STRESS_PARAMS
            regime_mask[i] = 1
        else:
            params = CALM_PARAMS
            regime_mask[i] = 0
        w = svi_total_variance(k_grid, params)
        sigma_imp = np.sqrt(np.clip(w, 1e-8, None) / max(t, 1e-4))
        sigma_imp = sigma_imp + rng.normal(0.0, noise, size=n_strikes)
        sigma_imp = np.clip(sigma_imp, 1e-3, None)
        iv[i, :] = sigma_imp

    strikes = forward * np.exp(k_grid)
    atm_vol = np.array(
        [float(np.interp(0.0, k_grid, iv[i, :])) for i in range(n_tenors)]
    )

    return SyntheticOptionChain(
        strikes=strikes,
        tenors=tenors,
        log_moneyness=k_grid,
        implied_vol=iv,
        regime_mask=regime_mask,
        forward=forward,
        atm_vol=atm_vol,
        meta={
            "seed": int(config["seed"]),
            "n_strikes": n_strikes,
            "n_tenors": n_tenors,
            "noise_sigma_iv": noise,
        },
    )


def sample_flat_points(
    chain: SyntheticOptionChain,
    n_samples: int,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Produce flat (k, t, iv, atm_vol) samples for the neural residual.

    We resample with light jitter around the grid nodes so the MLP sees more
    than just the 250 points of the raw chain.
    """
    rng = np.random.default_rng(seed)
    kk, tt = np.meshgrid(chain.log_moneyness, chain.tenors)
    flat_k = kk.ravel()
    flat_t = tt.ravel()
    flat_iv = chain.implied_vol.ravel()
    atm_flat = np.tile(chain.atm_vol[:, None], (1, chain.log_moneyness.size)).ravel()

    if n_samples <= flat_k.size:
        idx = rng.choice(flat_k.size, size=n_samples, replace=False)
    else:
        idx = rng.integers(0, flat_k.size, size=n_samples)

    k_sample = flat_k[idx] + rng.normal(0.0, 0.005, size=n_samples)
    t_sample = flat_t[idx] + rng.normal(0.0, 0.002, size=n_samples)
    t_sample = np.clip(t_sample, 1e-3, None)
    iv_sample = flat_iv[idx] + rng.normal(0.0, 0.001, size=n_samples)
    iv_sample = np.clip(iv_sample, 1e-3, None)

    return {
        "k": k_sample,
        "t": t_sample,
        "atm_vol": atm_flat[idx],
        "iv": iv_sample,
    }
