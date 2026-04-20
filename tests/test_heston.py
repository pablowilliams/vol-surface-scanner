"""Unit tests for the Heston pricer and calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.heston.calibrate import (
    bs_call,
    calibrate_heston,
    heston_implied_vol_surface,
    implied_vol_from_call,
)
from vol_scanner.heston.pricer import HestonParamsRaw, heston_call_fft


def test_bs_call_inversion_round_trip() -> None:
    forward = 100.0
    vol = 0.25
    t = 0.5
    for k in (90.0, 100.0, 110.0):
        price = bs_call(forward, k, vol, t)
        v = implied_vol_from_call(price, forward, k, t)
        assert abs(v - vol) < 1e-3


def test_heston_call_fft_positive_and_monotone() -> None:
    p = HestonParamsRaw(kappa=2.0, theta=0.04, v0=0.04, rho=-0.5, xi=0.5)
    strikes = np.linspace(80, 120, 9)
    prices = heston_call_fft(strikes, t=0.5, r=0.0, q=0.0, p=p, s0=100.0)
    assert np.all(prices >= 0.0)
    # Calls strictly decreasing in strike for a positive rate / dividend.
    assert np.all(np.diff(prices) <= 1e-6)


def test_heston_implied_vol_surface_shape() -> None:
    p = HestonParamsRaw(kappa=2.0, theta=0.04, v0=0.04, rho=-0.5, xi=0.5)
    strikes = np.linspace(85, 115, 7)
    tenors = np.array([0.25, 0.5, 1.0])
    iv = heston_implied_vol_surface(strikes, tenors, p, forward=100.0)
    assert iv.shape == (3, 7)
    assert np.all(iv > 0.05) and np.all(iv < 1.5)


def test_calibrate_heston_recovers_seed() -> None:
    p = HestonParamsRaw(kappa=2.0, theta=0.04, v0=0.04, rho=-0.5, xi=0.5)
    strikes = np.linspace(85, 115, 9)
    tenors = np.array([0.25, 0.5, 1.0, 2.0])
    iv = heston_implied_vol_surface(strikes, tenors, p, forward=100.0)
    calibrated = calibrate_heston(strikes, tenors, iv, forward=100.0, max_iter=80)
    assert calibrated.rmse < 0.02
