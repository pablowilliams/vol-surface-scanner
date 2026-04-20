"""Static no-arbitrage constraint evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.svi.constraints import evaluate_all
from vol_scanner.svi.parametric import SVIParams


def test_calm_params_pass_constraints() -> None:
    p = SVIParams(a=0.02, b=0.08, rho=-0.3, m=0.0, sigma=0.3)
    res = evaluate_all(p, t=0.5)
    assert res["positivity"] >= 0
    assert res["rho_bounds"] >= 0
    assert res["sigma_floor"] >= 0
    assert res["b_nonneg"] >= 0
    assert res["lee_wing"] >= 0


def test_bad_rho_fails() -> None:
    p = SVIParams(a=0.02, b=0.08, rho=1.5, m=0.0, sigma=0.3)
    res = evaluate_all(p, t=0.5)
    assert res["rho_bounds"] < 0
