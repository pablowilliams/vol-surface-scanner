"""End to end pipeline smoke test, must complete fast in quick mode."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.pipeline.run import run_pipeline


def test_pipeline_smoke_quick() -> None:
    start = time.time()
    bundle = run_pipeline(quick=True)
    elapsed = time.time() - start
    assert elapsed < 60.0, f"pipeline too slow: {elapsed:.1f}s"
    assert "metrics" in bundle
    assert "surface" in bundle
    assert bundle["metrics"]["svi_rmse"] < 0.1
    assert len(bundle["svi_params"]) == bundle["meta"]["n_tenors"]
