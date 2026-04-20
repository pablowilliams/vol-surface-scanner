"""Residual heatmap figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals(
    strikes: np.ndarray,
    tenors: np.ndarray,
    residual: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    vmax = float(np.max(np.abs(residual))) + 1e-9
    im = ax.imshow(
        residual,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=(float(strikes.min()), float(strikes.max()), float(tenors.min()), float(tenors.max())),
    )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tenor (years)")
    ax.set_title("SVI residual surface")
    fig.colorbar(im, ax=ax, label="IV residual")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
