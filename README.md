# Neural Volatility Surface Arbitrage Scanner

An end to end quantitative finance research pipeline by Pablo Williams (UCL MSc Business Analytics). The project fits a Gatheral raw SVI parametric surface to a synthetic equity index option chain, trains a small neural residual model on top, and scans the joint surface for static no arbitrage violations. Results are served through an accessibility first static dashboard.

## Live metrics from the reference run

The numbers below were captured from a single deterministic run under seed 42.

| Metric | Value |
| --- | ---: |
| Options priced | 250 |
| Tenors fitted | 10 |
| SVI RMSE | 0.00410 |
| SVI MAE | 0.00329 |
| SVI R squared | 0.99988 |
| Residual held out RMSE | 0.02763 |
| Combined RMSE | 0.00512 |
| Combined RMSE change | plus 0.00102 |
| Total violations | 186 |
| Butterfly violations | 0 |
| Calendar violations | 186 |
| Vertical spread violations | 0 |
| High severity | 9 |
| Medium severity | 43 |
| Low severity | 134 |
| Wall clock end to end | about 6 seconds |

On this synthetic chain the SVI family already explains over 99.98 percent of the variance, so the neural residual does not materially improve grid node error and, on this particular run, produces a small positive RMSE delta consistent with fitting mid price noise. The calendar violations reflect the two regime (calm and stressed) structure of the synthetic data by design.

## How to reproduce

Requirements: Python 3.10, 3.11, or 3.12.

```bash
# Create a virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
PYTHONPATH=src python3 -m vol_scanner.pipeline.run

# Or the quick CI smoke
PYTHONPATH=src python3 -m vol_scanner.pipeline.run --quick

# Run the tests
PYTHONPATH=src python3 -m pytest -q

# Lint
ruff check .

# Rebuild the report PDF (runs the pipeline internally)
PYTHONPATH=src python3 report/build_report.py

# Serve the dashboard locally
cd dashboard && python3 -m http.server 8765
# then open http://localhost:8765/
```

The `scripts/run_pipeline.sh` wrapper sets `PYTHONPATH` automatically.

## Directory layout

```
configs/            YAML configs for data, svi, neural, scanner
src/vol_scanner/    Python package
  data/             Synthetic chain generator and IO helpers
  svi/              Raw SVI, SLSQP fit, static arbitrage constraints
  neural/           ResidualMLP and training loop
  scanner/          Butterfly, calendar, vertical spread checks
  metrics/          Errors and coverage
  pipeline/         Orchestration entry point
  figures/          Matplotlib figure generators for the report
dashboard/          Static accessibility first HTML, CSS, JS
  data/             Pipeline JSON output consumed by the UI
tests/              Pytest suite
figures/            Generated PNGs used in the report
report/             PDF report and build script
scripts/            Shell wrappers
```

## Dashboard

The dashboard is a no build step static site. Three.js is loaded via an ESM import map from https://esm.sh/three@0.160.0. All interactive elements are keyboard operable. The page ships two live regions (polite and assertive) for screen reader announcements, a skip link, an accessible sortable violations table, per slider aria-valuetext, a theme toggle with aria-pressed, and a fallback data table for the three dimensional canvas. Colour is never the only carrier of meaning; severity is conveyed by a chip combining icon, text, and colour.

## Pipeline output schema

The dashboard consumes `dashboard/data/surface.json` with the following top level keys:

- `meta`: generation time, number of options, seed, forward, grid sizes
- `surface`: strike, tenor, and implied vol arrays plus SVI and combined surfaces
- `violations`: list of typed violation records (type, severity, strike, tenor, magnitude, description)
- `violation_counts`: counts by violation type
- `svi_params`: fitted raw SVI parameters per tenor
- `metrics`: aggregate RMSE, MAE, R squared, violation counts, severity distribution

## Testing

Nine tests cover synthetic determinism, SVI fit recovery within tolerance, constraint evaluation, injected arbitrage detection, and an end to end pipeline smoke in quick mode under 60 seconds. Continuous integration runs the full matrix on Python 3.10, 3.11, and 3.12.

## Licence

MIT, Pablo Williams 2026. See LICENSE.
