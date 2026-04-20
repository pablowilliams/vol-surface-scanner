"""End to end pipeline orchestration."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from ..data.io import load_yaml, save_json
from ..data.synthetic import generate_chain, sample_flat_points
from ..figures.fig_residuals import plot_residuals
from ..figures.fig_surface import plot_surface
from ..figures.fig_violations import plot_violations
from ..metrics.coverage import severity_distribution
from ..metrics.errors import mae, r_squared, rmse
from ..neural.train import predict_residual, train_residual
from ..scanner.arbitrage import scan_surface
from ..svi.fit import fit_surface, predict_surface

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"
FIGURES_DIR = REPO_ROOT / "figures"


def run_pipeline(quick: bool = False) -> dict:
    data_cfg = load_yaml(CONFIG_DIR / "data.yaml")
    svi_cfg = load_yaml(CONFIG_DIR / "svi.yaml")
    neural_cfg = load_yaml(CONFIG_DIR / "neural.yaml")
    scanner_cfg = load_yaml(CONFIG_DIR / "scanner.yaml")

    if quick:
        neural_cfg["epochs"] = 20
        data_cfg["n_samples"] = 1000
        data_cfg["n_strikes"] = 15
        data_cfg["n_tenors"] = 6

    chain = generate_chain(data_cfg)
    fits = fit_surface(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg)
    svi_surface = predict_surface(chain.log_moneyness, chain.tenors, fits)

    svi_rmse = rmse(svi_surface, chain.implied_vol)
    svi_mae = mae(svi_surface, chain.implied_vol)
    svi_r2 = r_squared(chain.implied_vol, svi_surface)

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

    scan = scan_surface(fits, chain.forward, scanner_cfg)
    severities = severity_distribution(scan["violations"])

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
        "forward": float(chain.forward),
        "n_tenors": int(chain.tenors.size),
        "n_strikes": int(chain.strikes.size),
        "quick_mode": bool(quick),
    }
    violations_payload = {"violations": scan["violations"], "counts": scan["counts"]}
    svi_params_payload = [
        {
            "tenor": float(f.tenor),
            "a": float(f.params.a),
            "b": float(f.params.b),
            "rho": float(f.params.rho),
            "m": float(f.params.m),
            "sigma": float(f.params.sigma),
            "rmse": float(f.rmse),
        }
        for f in fits
    ]
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
    }

    bundle = {
        "meta": meta_payload,
        "surface": surface_payload,
        "violations": violations_payload["violations"],
        "violation_counts": violations_payload["counts"],
        "svi_params": svi_params_payload,
        "metrics": metrics_payload,
    }

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_json(DASHBOARD_DATA_DIR / "surface.json", bundle)
    save_json(DASHBOARD_DATA_DIR / "violations.json", violations_payload)
    save_json(DASHBOARD_DATA_DIR / "meta.json", {"meta": meta_payload, "metrics": metrics_payload})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_surface(chain.strikes, chain.tenors, chain.implied_vol, svi_surface, FIGURES_DIR / "fig_surface.png")
    plot_residuals(chain.strikes, chain.tenors, chain.implied_vol - svi_surface, FIGURES_DIR / "fig_residuals.png")
    plot_violations(scan["counts"], FIGURES_DIR / "fig_violations.png")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vol-scanner pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick CI smoke mode")
    args = parser.parse_args()
    bundle = run_pipeline(quick=args.quick)
    m = bundle["metrics"]
    print(
        f"SVI RMSE {m['svi_rmse']:.5f}, combined RMSE {m['combined_rmse']:.5f}, "
        f"violations {m['total_violations']}"
    )


if __name__ == "__main__":
    main()
