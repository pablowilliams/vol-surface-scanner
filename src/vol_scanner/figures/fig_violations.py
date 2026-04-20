"""Violation funnel bar chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_violations(counts: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    kinds = ["butterfly", "calendar", "vertical"]
    values = [int(counts.get(k, 0)) for k in kinds]
    bars = ax.bar(kinds, values, color=["#38BDF8", "#F59E0B", "#F472B6"])
    for bar, v in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(v), ha="center")
    ax.set_ylabel("Number of violations flagged")
    ax.set_title("Arbitrage violation funnel by type")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
