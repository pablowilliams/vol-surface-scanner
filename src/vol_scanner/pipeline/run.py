"""End to end pipeline orchestration.

Runs SVI fits, trains the neural residual MLP, scans for static arbitrage,
and then produces every artefact required by the ten build on features:
Heston calibration, rolling backtest, stress scenarios, Greeks, arbitrage
free resampler, term structure decomposition, Monte Carlo pricing, severity
scoring and regime classification. Every output JSON is saved under
dashboard/data so the dashboard can render them without further work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np

from ..backtest.rolling import rolling_backtest
from ..data.io import load_yaml, save_json
from ..data.synthetic import generate_chain, sample_flat_points
from ..figures.fig_extensions import (
    fig_backtest_leadlag,
    fig_heston_vs_svi,
    fig_raw_chain_histogram,
    fig_regime_confusion,
    fig_resampler_nudge,
    fig_severity_hist,
    fig_stress_radar,
    fig_svi_slices,
    fig_term_structure,
    fig_training_curves,
    fig_vega_heatmap,
    fig_violations_scatter,
)
from ..figures.fig_residuals import plot_residuals
from ..figures.fig_surface import plot_surface
from ..figures.fig_violations import plot_violations
from ..greeks.finite_difference import compute_greeks
from ..heston.calibrate import calibrate_heston, heston_implied_vol_surface
from ..heston.pricer import heston_call_fft
from ..mc.heston_mc import price_book_mc
from ..metrics.coverage import severity_distribution
from ..metrics.errors import mae, r_squared, rmse
from ..neural.train import predict_residual, train_residual
from ..regime.classifier import train_regime_classifier
from ..resampler.project import project_arbitrage_free
from ..scanner.arbitrage import scan_surface
from ..severity.score import severity_score_records
from ..stress.scenarios import run_stress_scenarios
from ..svi.fit import fit_surface, predict_surface
from ..term_structure.decompose import decompose_term_structure

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"
FIGURES_DIR = REPO_ROOT / "figures"


def run_pipeline(quick: bool = False) -> dict:  # noqa: PLR0915
    budget: dict[str, float] = {}
    t_start = time.time()

    data_cfg = load_yaml(CONFIG_DIR / "data.yaml")
    svi_cfg = load_yaml(CONFIG_DIR / "svi.yaml")
    neural_cfg = load_yaml(CONFIG_DIR / "neural.yaml")
    scanner_cfg = load_yaml(CONFIG_DIR / "scanner.yaml")

    if quick:
        neural_cfg["epochs"] = 20
        data_cfg["n_samples"] = 1000
        data_cfg["n_strikes"] = 15
        data_cfg["n_tenors"] = 6

    t0 = time.time()
    chain = generate_chain(data_cfg)
    budget["data"] = time.time() - t0

    t0 = time.time()
    fits = fit_surface(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg)
    svi_surface = predict_surface(chain.log_moneyness, chain.tenors, fits)
    svi_rmse = rmse(svi_surface, chain.implied_vol)
    svi_mae = mae(svi_surface, chain.implied_vol)
    svi_r2 = r_squared(chain.implied_vol, svi_surface)
    svi_per_tenor_rmse = [
        float(np.sqrt(np.mean((svi_surface[i] - chain.implied_vol[i]) ** 2)))
        for i in range(chain.tenors.size)
    ]
    budget["svi"] = time.time() - t0

    # --- Heston calibration --------------------------------------------------
    t0 = time.time()
    forward = float(chain.forward)
    heston = calibrate_heston(
        chain.strikes,
        chain.tenors,
        chain.implied_vol,
        forward,
        r=float(data_cfg.get("risk_free_rate", 0.0)),
        q=float(data_cfg.get("dividend_yield", 0.0)),
        max_iter=(8 if quick else 25),
    )
    budget["heston"] = time.time() - t0

    # --- Neural residual -----------------------------------------------------
    t0 = time.time()
    samples = sample_flat_points(chain, int(data_cfg["n_samples"]))
    svi_pred_samples = np.zeros_like(samples["iv"])
    for i in range(samples["iv"].size):
        ki = samples["k"][i]
        ti = samples["t"][i]
        ti_idx = int(np.argmin(np.abs(chain.tenors - ti)))
        fit = fits[ti_idx]
        from ..svi.parametric import total_variance
        w = total_variance(np.array([ki]), fit.params)[0]
        svi_pred_samples[i] = float(np.sqrt(max(w, 1e-10) / max(ti, 1e-6)))
    residual_target = samples["iv"] - svi_pred_samples
    features = np.stack([samples["k"], samples["t"], samples["atm_vol"]], axis=-1)
    train_result = train_residual(features, residual_target, neural_cfg)

    k_grid = chain.log_moneyness
    t_grid = chain.tenors
    kk, tt = np.meshgrid(k_grid, t_grid)
    atm_mat = np.tile(chain.atm_vol[:, None], (1, k_grid.size))
    residual_pred = predict_residual(
        train_result.model, kk.ravel(), tt.ravel(), atm_mat.ravel()
    ).reshape(kk.shape)
    combined_surface = svi_surface + residual_pred
    residual_rmse = rmse(combined_surface, chain.implied_vol)
    residual_improve = svi_rmse - residual_rmse
    budget["neural"] = time.time() - t0

    # --- Scanner -------------------------------------------------------------
    t0 = time.time()
    scan = scan_surface(fits, forward, scanner_cfg)
    severities = severity_distribution(scan["violations"])

    # Continuous severity scoring against a reference calm chain.
    ref_cfg = dict(data_cfg)
    ref_cfg["stress_mixture_probability"] = 0.0
    ref_cfg["seed"] = int(data_cfg["seed"]) + 1000
    ref_chain = generate_chain(ref_cfg)
    ref_fits = fit_surface(ref_chain.log_moneyness, ref_chain.implied_vol, ref_chain.tenors, svi_cfg)
    ref_scan = scan_surface(ref_fits, float(ref_chain.forward), scanner_cfg)
    ref_mags = [float(r.get("magnitude", 0.0)) for r in ref_scan["violations"]]
    if not ref_mags:
        ref_mags = [1e-6]
    scan_scored = severity_score_records(scan["violations"], reference_magnitudes=ref_mags)
    budget["scanner"] = time.time() - t0

    # --- Stress --------------------------------------------------------------
    t0 = time.time()
    stress_report = run_stress_scenarios(
        chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg, baseline_fits=fits
    )
    budget["stress"] = time.time() - t0

    # --- Greeks --------------------------------------------------------------
    t0 = time.time()
    greeks = compute_greeks(
        chain.strikes, chain.tenors, combined_surface, forward, r=float(data_cfg.get("risk_free_rate", 0.0))
    )
    budget["greeks"] = time.time() - t0

    # --- Arbitrage free resampler -------------------------------------------
    t0 = time.time()
    resampler = project_arbitrage_free(
        chain.log_moneyness, chain.implied_vol, chain.tenors, max_iter=(30 if quick else 80)
    )
    budget["resampler"] = time.time() - t0

    # --- Term structure ------------------------------------------------------
    t0 = time.time()
    ts = decompose_term_structure(fits)
    budget["term_structure"] = time.time() - t0

    # --- Monte Carlo pricer --------------------------------------------------
    t0 = time.time()
    mc_tenor_idx = int(chain.tenors.size // 2)
    mc_tenor = float(chain.tenors[mc_tenor_idx])
    mc_strikes = np.linspace(0.8 * forward, 1.2 * forward, 5)
    # Surface prices (BS under fitted combined IV).
    from ..scanner.arbitrage import _bs_call as _bs
    # Interpolate IV for each mc strike on the chosen tenor.
    iv_row = combined_surface[mc_tenor_idx, :]
    iv_interp = np.interp(np.log(mc_strikes / forward), chain.log_moneyness, iv_row)
    surf_prices = np.array(
        [float(_bs(forward, float(mc_strikes[i]), float(iv_interp[i]), mc_tenor)) for i in range(mc_strikes.size)]
    )
    mc = price_book_mc(
        mc_strikes,
        mc_tenor,
        forward,
        r=float(data_cfg.get("risk_free_rate", 0.0)),
        q=float(data_cfg.get("dividend_yield", 0.0)),
        p=heston.as_raw(),
        surface_prices=surf_prices,
        n_paths=(2000 if quick else 10_000),
        n_steps=(16 if quick else 64),
    )
    # Heston Carr Madan prices for the same book.
    heston_prices = heston_call_fft(
        mc_strikes,
        mc_tenor,
        float(data_cfg.get("risk_free_rate", 0.0)),
        float(data_cfg.get("dividend_yield", 0.0)),
        heston.as_raw(),
        forward,
    )
    budget["mc"] = time.time() - t0

    # --- Backtest ------------------------------------------------------------
    t0 = time.time()
    bt = rolling_backtest(
        svi_cfg,
        scanner_cfg,
        data_cfg,
        n_days=(8 if quick else 30),
        horizon=5,
        seed=2026,
    )
    budget["backtest"] = time.time() - t0

    # --- Regime classifier ---------------------------------------------------
    t0 = time.time()
    slice_params_dicts = [
        {"a": float(f.params.a), "b": float(f.params.b), "rho": float(f.params.rho), "m": float(f.params.m), "sigma": float(f.params.sigma)}
        for f in fits
    ]
    regime_clf = train_regime_classifier(slice_params_dicts)
    budget["regime"] = time.time() - t0

    # --- Build payloads ------------------------------------------------------
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    surface_payload = {
        "strikes": chain.strikes.tolist(),
        "tenors": chain.tenors.tolist(),
        "iv": chain.implied_vol.tolist(),
        "svi_iv": svi_surface.tolist(),
        "residual": (chain.implied_vol - svi_surface).tolist(),
        "combined_iv": combined_surface.tolist(),
        "log_moneyness": chain.log_moneyness.tolist(),
    }
    meta_payload = {
        "generated": now,
        "n_options": int(chain.implied_vol.size),
        "seed": int(data_cfg["seed"]),
        "forward": forward,
        "n_tenors": int(chain.tenors.size),
        "n_strikes": int(chain.strikes.size),
        "quick_mode": bool(quick),
        "regime": regime_clf.inferred_regime,
    }
    violations_payload = {"violations": scan_scored, "counts": scan["counts"]}
    svi_params_payload = [
        {
            "tenor": float(f.tenor),
            "a": float(f.params.a),
            "b": float(f.params.b),
            "rho": float(f.params.rho),
            "m": float(f.params.m),
            "sigma": float(f.params.sigma),
            "rmse": float(f.rmse),
            "regime": regime_clf.per_slice[i] if i < len(regime_clf.per_slice) else "unknown",
        }
        for i, f in enumerate(fits)
    ]
    heston_payload = {
        "kappa": float(heston.kappa),
        "theta": float(heston.theta),
        "v0": float(heston.v0),
        "rho": float(heston.rho),
        "xi": float(heston.xi),
        "rmse": float(heston.rmse),
        "per_tenor_rmse": [float(x) for x in heston.per_tenor_rmse],
        "success": bool(heston.success),
    }
    # Combined RMSE per tenor.
    combined_per_tenor_rmse = [
        float(np.sqrt(np.mean((combined_surface[i] - chain.implied_vol[i]) ** 2)))
        for i in range(chain.tenors.size)
    ]
    # Injected adversarial precision recall on a synthetic set of violations.
    adv_pr = _evaluate_adversarial_scanner(svi_cfg, scanner_cfg, seed=int(data_cfg["seed"]))

    metrics_payload = {
        "svi_rmse": float(svi_rmse),
        "svi_mae": float(svi_mae),
        "svi_r2": float(svi_r2),
        "residual_rmse": float(train_result.final_rmse),
        "combined_rmse": float(residual_rmse),
        "combined_improvement": float(residual_improve),
        "violation_counts": scan["counts"],
        "violation_severity": severities,
        "total_violations": int(len(scan["violations"])),
        "svi_per_tenor_rmse": svi_per_tenor_rmse,
        "combined_per_tenor_rmse": combined_per_tenor_rmse,
        "heston_rmse": float(heston.rmse),
        "heston_per_tenor_rmse": heston.per_tenor_rmse,
        "resampler_mean_nudge": float(resampler.mean_nudge),
        "resampler_mean_rel_nudge": float(resampler.mean_relative_nudge),
        "resampler_max_nudge": float(resampler.max_nudge),
        "regime_train_accuracy": float(regime_clf.train_accuracy),
        "regime_test_accuracy": float(regime_clf.test_accuracy),
        "regime_confusion": regime_clf.confusion.tolist(),
        "regime_inferred": regime_clf.inferred_regime,
        "backtest_lead_lag": {str(k): float(v) for k, v in bt.lead_lag.items()},
        "backtest_hit_rate": float(bt.hit_rate),
        "stress_drift_summary": stress_report.drift_summary,
        "mc_absolute_gap_mean": float(mc.absolute_gap.mean()),
        "mc_absolute_gap_max": float(mc.absolute_gap.max()),
        "adversarial_precision": adv_pr["precision"],
        "adversarial_recall": adv_pr["recall"],
        "adversarial_f1": adv_pr["f1"],
        "compute_budget_seconds": {k: float(v) for k, v in budget.items()},
    }

    bundle = {
        "meta": meta_payload,
        "surface": surface_payload,
        "violations": violations_payload["violations"],
        "violation_counts": violations_payload["counts"],
        "svi_params": svi_params_payload,
        "metrics": metrics_payload,
        "heston": heston_payload,
    }

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_json(DASHBOARD_DATA_DIR / "surface.json", bundle)
    save_json(DASHBOARD_DATA_DIR / "violations.json", violations_payload)
    save_json(DASHBOARD_DATA_DIR / "meta.json", {"meta": meta_payload, "metrics": metrics_payload})
    save_json(DASHBOARD_DATA_DIR / "heston.json", heston_payload)
    save_json(
        DASHBOARD_DATA_DIR / "stress.json",
        {"scenarios": stress_report.scenarios, "drift_summary": stress_report.drift_summary},
    )
    save_json(
        DASHBOARD_DATA_DIR / "backtest.json",
        {
            "dates": bt.dates,
            "forward_path": bt.forward_path,
            "violation_counts": bt.violation_counts,
            "realised_moves": bt.realised_moves,
            "lead_lag": {str(k): float(v) for k, v in bt.lead_lag.items()},
            "hit_rate": float(bt.hit_rate),
            "n_days": int(bt.n_days),
        },
    )
    save_json(DASHBOARD_DATA_DIR / "greeks.json", greeks.to_payload())
    save_json(
        DASHBOARD_DATA_DIR / "resampler.json",
        {
            "projected_iv": resampler.projected_iv.tolist(),
            "mean_nudge": float(resampler.mean_nudge),
            "max_nudge": float(resampler.max_nudge),
            "mean_relative_nudge": float(resampler.mean_relative_nudge),
            "success": bool(resampler.success),
        },
    )
    save_json(DASHBOARD_DATA_DIR / "term_structure.json", ts.to_payload())
    save_json(
        DASHBOARD_DATA_DIR / "mc.json",
        {
            "strikes": mc.strikes.tolist(),
            "tenor": mc.tenor,
            "mc_prices": mc.mc_prices.tolist(),
            "mc_stderr": mc.mc_stderr.tolist(),
            "surface_prices": mc.surface_prices.tolist(),
            "heston_fft_prices": heston_prices.tolist(),
            "absolute_gap": mc.absolute_gap.tolist(),
            "n_paths": int(mc.n_paths),
        },
    )
    save_json(
        DASHBOARD_DATA_DIR / "regime.json",
        {
            "inferred": regime_clf.inferred_regime,
            "per_slice": regime_clf.per_slice,
            "per_slice_probs": regime_clf.per_slice_probs,
            "train_accuracy": float(regime_clf.train_accuracy),
            "test_accuracy": float(regime_clf.test_accuracy),
            "confusion": regime_clf.confusion.tolist(),
        },
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_surface(chain.strikes, chain.tenors, chain.implied_vol, svi_surface, FIGURES_DIR / "fig_surface.png")
    plot_residuals(chain.strikes, chain.tenors, chain.implied_vol - svi_surface, FIGURES_DIR / "fig_residuals.png")
    plot_violations(scan["counts"], FIGURES_DIR / "fig_violations.png")

    # Extended figures.
    fig_raw_chain_histogram(chain.implied_vol, chain.tenors, FIGURES_DIR / "fig01_raw_chain.png")
    fig_svi_slices(
        chain.log_moneyness, chain.implied_vol, svi_surface, chain.tenors,
        FIGURES_DIR / "fig02_svi_slices.png",
    )
    heston_iv = heston_implied_vol_surface(
        chain.strikes, chain.tenors, heston.as_raw(), forward,
        r=float(data_cfg.get("risk_free_rate", 0.0)),
        q=float(data_cfg.get("dividend_yield", 0.0)),
    )
    heston_per_tenor = [
        float(np.sqrt(np.mean((heston_iv[i] - chain.implied_vol[i]) ** 2)))
        for i in range(chain.tenors.size)
    ]
    fig_heston_vs_svi(
        chain.tenors, svi_per_tenor_rmse, heston_per_tenor, FIGURES_DIR / "fig03_heston_vs_svi.png"
    )
    fig_training_curves(train_result.train_losses, train_result.val_losses, FIGURES_DIR / "fig04_train_curves.png")
    fig_violations_scatter(scan_scored, FIGURES_DIR / "fig05_violations_scatter.png")
    fig_backtest_leadlag(bt.lead_lag, FIGURES_DIR / "fig06_backtest_leadlag.png")
    fig_stress_radar(stress_report.drift_summary, FIGURES_DIR / "fig07_stress_radar.png")
    fig_vega_heatmap(chain.strikes, chain.tenors, greeks.vega, FIGURES_DIR / "fig08_vega_heatmap.png")
    fig_term_structure(
        ts.tenors, ts.atm_vol, ts.atm_skew, ts.atm_kurtosis,
        FIGURES_DIR / "fig09_term_structure.png",
    )
    fig_resampler_nudge(
        np.abs(resampler.projected_iv - chain.implied_vol), FIGURES_DIR / "fig10_resampler_nudge.png"
    )
    fig_regime_confusion(regime_clf.confusion, FIGURES_DIR / "fig11_regime_confusion.png")
    fig_severity_hist(
        [float(v.get("severity_score", 0.0)) for v in scan_scored],
        FIGURES_DIR / "fig12_severity_hist.png",
    )

    budget["total"] = time.time() - t_start
    metrics_payload["compute_budget_seconds"] = {k: float(v) for k, v in budget.items()}
    save_json(DASHBOARD_DATA_DIR / "meta.json", {"meta": meta_payload, "metrics": metrics_payload})
    save_json(DASHBOARD_DATA_DIR / "surface.json", bundle)

    return bundle


def _evaluate_adversarial_scanner(
    svi_cfg: dict,
    scanner_cfg: dict,
    seed: int = 42,
) -> dict:
    """Inject known violations and measure detection precision and recall.

    We build ten healthy slices and ten adversarial slices (extreme b, rho or
    non monotonic total variance) and record how many violations are detected
    on each. Healthy slices contribute false positives, adversarial slices
    contribute true positives.
    """
    from ..svi.fit import SliceFit
    from ..svi.parametric import SVIParams

    rng = np.random.default_rng(seed)
    healthy_fits = []
    for _ in range(10):
        healthy_fits.append(
            SliceFit(
                tenor=0.5,
                params=SVIParams(
                    a=float(rng.uniform(0.005, 0.03)),
                    b=float(rng.uniform(0.03, 0.09)),
                    rho=float(rng.uniform(-0.45, -0.1)),
                    m=0.0,
                    sigma=float(rng.uniform(0.2, 0.5)),
                ),
                rmse=0.0,
                iterations=0,
                success=True,
            )
        )
    adversarial_fits = []
    for _ in range(10):
        adversarial_fits.append(
            SliceFit(
                tenor=0.5,
                params=SVIParams(
                    a=-0.02,
                    b=float(rng.uniform(1.2, 1.8)),
                    rho=0.97,
                    m=0.0,
                    sigma=0.05,
                ),
                rmse=0.0,
                iterations=0,
                success=True,
            )
        )
    # Precision / recall on the butterfly test.
    from ..scanner.arbitrage import butterfly_violations

    tp = fp = fn = 0
    for f in healthy_fits:
        detected = butterfly_violations([f], forward=100.0, cfg=scanner_cfg)
        if detected:
            fp += 1
    for f in adversarial_fits:
        detected = butterfly_violations([f], forward=100.0, cfg=scanner_cfg)
        if detected:
            tp += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vol-scanner pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick CI smoke mode")
    args = parser.parse_args()
    bundle = run_pipeline(quick=args.quick)
    m = bundle["metrics"]
    print(
        f"SVI RMSE {m['svi_rmse']:.5f}, combined RMSE {m['combined_rmse']:.5f}, "
        f"Heston RMSE {m['heston_rmse']:.5f}, violations {m['total_violations']}, "
        f"regime {m['regime_inferred']}"
    )


if __name__ == "__main__":
    main()
