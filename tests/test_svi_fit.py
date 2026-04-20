"""The SVI fit should recover known parameters within tolerance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.data.io import load_yaml
from vol_scanner.svi.fit import fit_slice
from vol_scanner.svi.parametric import SVIParams, total_variance


def test_fit_recovers_synthetic_slice() -> None:
    cfg = load_yaml(ROOT / "configs" / "svi.yaml")
    true_params = SVIParams(a=0.02, b=0.08, rho=-0.3, m=0.0, sigma=0.3)
    k = np.linspace(-1.0, 1.0, 30)
    t = 0.5
    w = total_variance(k, true_params)
    iv = np.sqrt(np.clip(w, 1e-10, None) / t)

    fit = fit_slice(k, iv, t, cfg)
    assert fit.success or fit.rmse < 1e-3
    assert abs(fit.params.rho - true_params.rho) < 0.15
    assert abs(fit.params.b - true_params.b) < 0.05
    assert fit.rmse < 5e-3
