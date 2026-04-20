"""3D matplotlib rendering of the fitted volatility surface."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_surface(
    strikes: np.ndarray,
    tenors: np.ndarray,
    iv_true: np.ndarray,
    iv_svi: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 5), dpi=150)
    ax = fig.add_subplot(121, projection="3d")
    K, T = np.meshgrid(strikes, tenors)
    ax.plot_surface(K, T, iv_true, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Implied vol")
    ax.set_title("Synthetic market surface")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(K, T, iv_svi, cmap="plasma", alpha=0.85)
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Tenor (years)")
    ax2.set_zlabel("Implied vol")
    ax2.set_title("SVI fitted surface")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
