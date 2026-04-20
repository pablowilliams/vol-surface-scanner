"""Compute ATM vol, ATM skew and ATM kurtosis per tenor slice.

For raw SVI we have
    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))
and the implied volatility is sigma_imp(k) = sqrt(w(k) / t). We compute the
first and second derivative of implied vol with respect to log moneyness at
k = 0 analytically from the chain rule, so no finite difference blur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..svi.fit import SliceFit
from ..svi.parametric import w_derivatives


@dataclass
class TermStructure:
    tenors: np.ndarray
    atm_vol: np.ndarray
    atm_skew: np.ndarray
    atm_kurtosis: np.ndarray

    def to_payload(self) -> dict:
        return {
            "tenors": self.tenors.tolist(),
            "atm_vol": self.atm_vol.tolist(),
            "atm_skew": self.atm_skew.tolist(),
            "atm_kurtosis": self.atm_kurtosis.tolist(),
        }


def _atm_from_fit(fit: SliceFit) -> tuple[float, float, float]:
    t = max(fit.tenor, 1e-6)
    k0 = np.array([0.0])
    w, wp, wpp = w_derivatives(k0, fit.params)
    w0 = float(max(w[0], 1e-10))
    wp0 = float(wp[0])
    wpp0 = float(wpp[0])
    sigma = float(np.sqrt(w0 / t))
    # d sigma / dk = wp / (2 t sigma)
    skew = wp0 / (2.0 * t * sigma)
    # d2 sigma / dk2 = wpp / (2 t sigma) - wp^2 / (4 t^2 sigma^3)
    kurt = wpp0 / (2.0 * t * sigma) - (wp0**2) / (4.0 * (t**2) * (sigma**3))
    return sigma, float(skew), float(kurt)


def decompose_term_structure(fits: list[SliceFit]) -> TermStructure:
    order = np.argsort([f.tenor for f in fits])
    fits_sorted = [fits[i] for i in order]
    tenors = np.array([f.tenor for f in fits_sorted])
    atm_vol = np.zeros(tenors.size)
    atm_skew = np.zeros(tenors.size)
    atm_kurt = np.zeros(tenors.size)
    for i, f in enumerate(fits_sorted):
        s, sk, ku = _atm_from_fit(f)
        atm_vol[i] = s
        atm_skew[i] = sk
        atm_kurt[i] = ku
    return TermStructure(
        tenors=tenors,
        atm_vol=atm_vol,
        atm_skew=atm_skew,
        atm_kurtosis=atm_kurt,
    )
