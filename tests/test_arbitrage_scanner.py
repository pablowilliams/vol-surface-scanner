"""Scanner should detect injected violations."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.data.io import load_yaml
from vol_scanner.scanner.arbitrage import (
    butterfly_violations,
    calendar_violations,
)
from vol_scanner.svi.fit import SliceFit
from vol_scanner.svi.parametric import SVIParams


def _make_fit(a: float, b: float, rho: float, m: float, sigma: float, t: float) -> SliceFit:
    return SliceFit(
        tenor=t,
        params=SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma),
        rmse=0.0,
        iterations=0,
        success=True,
    )


def test_calendar_violation_detected() -> None:
    cfg = load_yaml(ROOT / "configs" / "scanner.yaml")
    high = _make_fit(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.3, t=0.25)
    low = _make_fit(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.3, t=0.5)
    violations = calendar_violations([high, low], forward=100.0, cfg=cfg)
    assert len(violations) > 0
    assert all(v.type == "calendar" for v in violations)


def test_butterfly_violation_detected() -> None:
    cfg = load_yaml(ROOT / "configs" / "scanner.yaml")
    # Extreme parameters produce negative Durrleman g at some k.
    bad = _make_fit(a=-0.02, b=1.5, rho=0.95, m=0.0, sigma=0.05, t=0.5)
    violations = butterfly_violations([bad], forward=100.0, cfg=cfg)
    assert len(violations) > 0
    assert all(v.type == "butterfly" for v in violations)


def test_healthy_slice_no_violations() -> None:
    cfg = load_yaml(ROOT / "configs" / "scanner.yaml")
    good = _make_fit(a=0.02, b=0.08, rho=-0.3, m=0.0, sigma=0.3, t=0.5)
    calendar = calendar_violations([good], forward=100.0, cfg=cfg)
    butter = butterfly_violations([good], forward=100.0, cfg=cfg)
    assert calendar == []
    assert butter == []
