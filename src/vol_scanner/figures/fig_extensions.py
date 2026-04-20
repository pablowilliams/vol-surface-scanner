"""Figure helpers for the ten build on features.

Each function saves a PNG into the given path and returns the path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fig_raw_chain_histogram(iv: np.ndarray, tenors: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for i, t in enumerate(tenors):
        ax.hist(iv[i, :], bins=18, alpha=0.45, label=f"t={float(t):.2f} y")
    ax.set_xlabel("Implied volatility")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of implied volatility by tenor")
    ax.legend(ncol=2, fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_svi_slices(
    k_grid: np.ndarray,
    iv_true: np.ndarray,
    iv_svi: np.ndarray,
    tenors: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    n = min(4, tenors.size)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), dpi=150)
    axes = axes.flatten()
    idxs = np.linspace(0, tenors.size - 1, n).astype(int)
    for ax, i in zip(axes, idxs, strict=False):
        ax.plot(k_grid, iv_true[i, :], "o", ms=4, alpha=0.7, label="observed")
        ax.plot(k_grid, iv_svi[i, :], "-", lw=1.8, label="SVI")
        ax.set_title(f"t = {float(tenors[i]):.3f} y")
        ax.set_xlabel("log moneyness")
        ax.set_ylabel("IV")
        ax.legend(fontsize=8)
    fig.suptitle("SVI fit per tenor slice")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_heston_vs_svi(
    tenors: np.ndarray,
    svi_rmse: list[float],
    heston_rmse: list[float],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(tenors, svi_rmse, "-o", label="SVI", color="#1F5FA8")
    ax.plot(tenors, heston_rmse, "-s", label="Heston", color="#C26C3C")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("RMSE (IV)")
    ax.set_title("Fit quality by tenor, Heston vs SVI")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(train_losses, label="train", lw=1.2)
    ax.plot(val_losses, label="val", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title("Residual MLP training curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_violations_scatter(violations: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    if violations:
        strikes = [v["strike"] for v in violations]
        tenors = [v["tenor"] for v in violations]
        severities = [float(v.get("severity_score", v.get("magnitude", 0.0))) for v in violations]
        sc = ax.scatter(strikes, tenors, c=severities, cmap="magma", s=28, alpha=0.7, edgecolor="none")
        fig.colorbar(sc, ax=ax, label="severity score")
    else:
        ax.text(0.5, 0.5, "no violations", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tenor (years)")
    ax.set_title("Arbitrage violations by strike and tenor")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_backtest_leadlag(lead_lag: dict[int, float], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    lags = sorted(lead_lag.keys())
    vals = [lead_lag[h] for h in lags]
    ax.bar(lags, vals, color="#1F5FA8")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Lead horizon (business days)")
    ax.set_ylabel("Corr(violations, |log return|)")
    ax.set_title("Lead lag correlation, scanner vs realised moves")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_stress_radar(drift_summary: dict[str, int], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    scenarios = list(drift_summary.keys())
    values = [float(drift_summary[s]) for s in scenarios]
    if not scenarios:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    angles = np.linspace(0, 2 * np.pi, len(scenarios), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw={"polar": True}, dpi=150)
    ax.plot(angles, values, color="#C26C3C", lw=1.6)
    ax.fill(angles, values, color="#C26C3C", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_title("SVI parameter drift under stress scenarios")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_vega_heatmap(
    strikes: np.ndarray,
    tenors: np.ndarray,
    vega: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    im = ax.imshow(
        vega,
        aspect="auto",
        origin="lower",
        extent=(float(strikes.min()), float(strikes.max()), float(tenors.min()), float(tenors.max())),
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="vega")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tenor (years)")
    ax.set_title("Vega surface on the fitted IV grid")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_term_structure(
    tenors: np.ndarray,
    atm_vol: np.ndarray,
    atm_skew: np.ndarray,
    atm_kurt: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
    axes[0].plot(tenors, atm_vol, "-o", color="#1F5FA8")
    axes[0].set_title("ATM vol")
    axes[1].plot(tenors, atm_skew, "-o", color="#C26C3C")
    axes[1].set_title("ATM skew (d sigma / dk)")
    axes[2].plot(tenors, atm_kurt, "-o", color="#3C8C5A")
    axes[2].set_title("ATM kurtosis (d2 sigma / dk2)")
    for ax in axes:
        ax.set_xlabel("Tenor (years)")
        ax.grid(alpha=0.25)
    fig.suptitle("Term structure decomposition")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_resampler_nudge(nudges: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    ax.hist(nudges.flatten(), bins=30, color="#1F5FA8", alpha=0.8)
    ax.set_xlabel("Nudge magnitude (IV units)")
    ax.set_ylabel("Count")
    ax.set_title("Arbitrage free resampler nudge distribution")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_regime_confusion(cm: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    labels = ["calm", "stressed", "crash"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Regime classifier confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_severity_hist(severity_scores: list[float], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=150)
    if severity_scores:
        ax.hist(severity_scores, bins=25, color="#C26C3C", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "no violations", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Continuous severity score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of violation severity scores")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
