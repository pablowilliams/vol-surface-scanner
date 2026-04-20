"""Determinism of the synthetic chain generator under a fixed seed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.data.io import load_yaml
from vol_scanner.data.synthetic import generate_chain


def test_same_seed_same_surface() -> None:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    a = generate_chain(cfg)
    b = generate_chain(cfg)
    np.testing.assert_allclose(a.implied_vol, b.implied_vol)
    np.testing.assert_allclose(a.strikes, b.strikes)
    np.testing.assert_allclose(a.tenors, b.tenors)


def test_different_seed_changes_surface() -> None:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    cfg2 = dict(cfg)
    cfg2["seed"] = 7
    a = generate_chain(cfg)
    b = generate_chain(cfg2)
    assert not np.allclose(a.implied_vol, b.implied_vol)
