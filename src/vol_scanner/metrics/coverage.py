"""Coverage metrics for arbitrage scan output."""
from __future__ import annotations

from collections import Counter


def severity_distribution(records: list[dict]) -> dict:
    c = Counter(r["severity"] for r in records)
    return {"high": int(c.get("high", 0)), "medium": int(c.get("medium", 0)), "low": int(c.get("low", 0))}


def violation_rate(records: list[dict], n_probes: int) -> float:
    if n_probes <= 0:
        return 0.0
    return float(len(records)) / float(n_probes)
