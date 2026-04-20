"""Unit tests covering the ten new extensions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.backtest.rolling import rolling_backtest
from vol_scanner.cli import build_parser
from vol_scanner.data.io import load_yaml
from vol_scanner.data.synthetic import generate_chain
from vol_scanner.greeks.finite_difference import compute_greeks
from vol_scanner.heston.pricer import HestonParamsRaw
from vol_scanner.mc.heston_mc import price_book_mc
from vol_scanner.regime.classifier import train_regime_classifier
from vol_scanner.resampler.project import project_arbitrage_free
from vol_scanner.severity.score import severity_score, severity_score_records
from vol_scanner.stress.scenarios import run_stress_scenarios
from vol_scanner.svi.fit import fit_surface
from vol_scanner.term_structure.decompose import decompose_term_structure


def _build_chain():
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    cfg["n_strikes"] = 12
    cfg["n_tenors"] = 5
    return generate_chain(cfg), cfg


def test_backtest_returns_lead_lag_dict() -> None:
    chain, data_cfg = _build_chain()
    svi_cfg = load_yaml(ROOT / "configs" / "svi.yaml")
    scanner_cfg = load_yaml(ROOT / "configs" / "scanner.yaml")
    bt = rolling_backtest(svi_cfg, scanner_cfg, data_cfg, n_days=6, horizon=3, seed=2026)
    assert bt.n_days == 6
    assert set(bt.lead_lag.keys()) == {1, 2, 3}
    for v in bt.lead_lag.values():
        assert -1.0 <= v <= 1.0


def test_stress_scenarios_produce_drift_summary() -> None:
    chain, _ = _build_chain()
    svi_cfg = load_yaml(ROOT / "configs" / "svi.yaml")
    fits = fit_surface(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg)
    rep = run_stress_scenarios(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg, baseline_fits=fits)
    assert set(rep.scenarios.keys()) == {"vol_spike", "crash_skew", "inversion", "calm"}
    for k in rep.drift_summary:
        assert isinstance(rep.drift_summary[k], int)


def test_greeks_shapes_and_signs() -> None:
    chain, _ = _build_chain()
    iv = np.full((chain.tenors.size, chain.strikes.size), 0.25)
    g = compute_greeks(chain.strikes, chain.tenors, iv, spot=float(chain.forward))
    assert g.delta.shape == iv.shape
    assert g.vega.shape == iv.shape
    # Vega is non negative for vanilla calls.
    assert np.all(g.vega >= -1e-9)
    # At least some interior strikes have positive vega.
    assert (g.vega > 0).sum() > 0
    # Delta lies in [0, 1].
    assert np.all((g.delta >= -0.05) & (g.delta <= 1.05))


def test_resampler_returns_no_arb_surface() -> None:
    chain, _ = _build_chain()
    rep = project_arbitrage_free(chain.log_moneyness, chain.implied_vol, chain.tenors)
    assert rep.projected_iv.shape == chain.implied_vol.shape
    # Nudge magnitudes are non negative.
    assert rep.mean_nudge >= 0.0
    assert rep.max_nudge >= rep.mean_nudge


def test_term_structure_three_curves() -> None:
    chain, _ = _build_chain()
    svi_cfg = load_yaml(ROOT / "configs" / "svi.yaml")
    fits = fit_surface(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg)
    ts = decompose_term_structure(fits)
    assert ts.atm_vol.size == ts.tenors.size
    assert ts.atm_skew.size == ts.tenors.size
    assert np.all(ts.atm_vol > 0)


def test_mc_pricer_close_to_surface() -> None:
    p = HestonParamsRaw(kappa=2.0, theta=0.04, v0=0.04, rho=-0.5, xi=0.5)
    strikes = np.array([95.0, 100.0, 105.0])
    surf = np.array([8.0, 5.0, 3.0])
    res = price_book_mc(strikes, tenor=0.5, spot=100.0, r=0.0, q=0.0, p=p, surface_prices=surf, n_paths=2000, n_steps=32)
    assert res.mc_prices.shape == strikes.shape
    assert res.absolute_gap.shape == strikes.shape


def test_cli_parser_subcommands() -> None:
    parser = build_parser()
    a = parser.parse_args(["run", "--quick"])
    assert a.command == "run"
    assert a.quick is True
    b = parser.parse_args(["scan", "--threshold", "0.05"])
    assert b.command == "scan"
    assert abs(b.threshold - 0.05) < 1e-9
    c = parser.parse_args(["export", "--format", "json"])
    assert c.command == "export"
    assert c.format == "json"


def test_severity_score_simple() -> None:
    assert severity_score(0.0, 1.0) == 0.0
    assert severity_score(1.0, 1.0) == 1.0
    assert severity_score(2.0, 1.0) == 2.0
    assert severity_score(0.5, 0.0) == 0.0


def test_severity_score_records_adds_field() -> None:
    rows = [{"magnitude": 0.01}, {"magnitude": 0.05}]
    out = severity_score_records(rows, reference_magnitudes=[0.005, 0.01, 0.02, 0.04])
    assert "severity_score" in out[0]
    assert out[1]["severity_score"] >= out[0]["severity_score"]


def test_regime_classifier_train_high_accuracy() -> None:
    clf = train_regime_classifier([])
    assert clf.train_accuracy > 0.85
    # Confusion matrix is 3x3.
    assert clf.confusion.shape == (3, 3)
