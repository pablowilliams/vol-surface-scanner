"""Continuous severity score for each violation record.

The score is the violation magnitude rescaled by a reference magnitude,
taken as the 95th percentile of the distribution of magnitudes on a
healthy reference chain. Scores above 1.0 indicate violations that are
worse than the upper tail of the reference chain, scores below 1.0 are
comparable to what a healthy chain produces via numerical noise.
"""
from __future__ import annotations

import numpy as np


def severity_score(magnitude: float, reference_p95: float) -> float:
    if reference_p95 <= 0:
        return 0.0
    return float(max(0.0, magnitude) / reference_p95)


def severity_score_records(
    violations: list[dict],
    reference_magnitudes: list[float] | None = None,
) -> list[dict]:
    if reference_magnitudes is None or len(reference_magnitudes) == 0:
        mags = [float(v.get("magnitude", 0.0)) for v in violations]
        ref_p95 = float(np.quantile(mags, 0.95)) if mags else 1.0
    else:
        ref_p95 = float(np.quantile(np.array(reference_magnitudes, dtype=float), 0.95))
    ref_p95 = max(ref_p95, 1e-9)
    out: list[dict] = []
    for v in violations:
        mag = float(v.get("magnitude", 0.0))
        v2 = dict(v)
        v2["severity_score"] = severity_score(mag, ref_p95)
        out.append(v2)
    return out
