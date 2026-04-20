"""Command line interface for vol-scanner.

Usage:
    vol-scanner run --quick
    vol-scanner scan --threshold 0.05
    vol-scanner export --format json [--out path]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data.io import load_json, load_yaml, save_json
from .pipeline.run import run_pipeline
from .scanner.arbitrage import scan_surface
from .svi.fit import fit_surface

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DASHBOARD_DIR = REPO_ROOT / "dashboard" / "data"


def cmd_run(args: argparse.Namespace) -> int:
    bundle = run_pipeline(quick=bool(args.quick))
    m = bundle["metrics"]
    print(
        f"SVI RMSE {m['svi_rmse']:.5f}, combined RMSE {m['combined_rmse']:.5f}, "
        f"violations {m['total_violations']}"
    )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Re scan the stored bundle with a custom magnitude threshold.

    The scanner is rerun on the cached SVI parameter set so the user can tighten
    or loosen bands without re training the residual MLP.
    """
    from .data.synthetic import generate_chain

    threshold = float(args.threshold)
    data_cfg = load_yaml(CONFIG_DIR / "data.yaml")
    svi_cfg = load_yaml(CONFIG_DIR / "svi.yaml")
    scanner_cfg = load_yaml(CONFIG_DIR / "scanner.yaml")
    for key in ("butterfly", "calendar", "vertical"):
        scanner_cfg[key]["severity_bands"]["low"] = threshold / 10.0
        scanner_cfg[key]["severity_bands"]["medium"] = threshold
        scanner_cfg[key]["severity_bands"]["high"] = threshold * 10.0

    chain = generate_chain(data_cfg)
    fits = fit_surface(chain.log_moneyness, chain.implied_vol, chain.tenors, svi_cfg)
    result = scan_surface(fits, chain.forward, scanner_cfg)
    out = {
        "threshold": threshold,
        "counts": result["counts"],
        "total": len(result["violations"]),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    fmt = str(args.format).lower()
    if fmt != "json":
        print(f"unsupported format: {fmt}", file=sys.stderr)
        return 2
    src = DASHBOARD_DIR / "surface.json"
    if not src.exists():
        print("No cached surface. Run vol-scanner run first.", file=sys.stderr)
        return 3
    bundle = load_json(src)
    out_path = Path(args.out) if args.out else REPO_ROOT / "vol-scanner-export.json"
    save_json(out_path, bundle)
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vol-scanner", description="Vol Surface Scanner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline")
    p_run.add_argument("--quick", action="store_true", help="Quick smoke mode")
    p_run.set_defaults(func=cmd_run)

    p_scan = sub.add_parser("scan", help="Rescan with a custom threshold")
    p_scan.add_argument("--threshold", type=float, default=0.05, help="Magnitude threshold")
    p_scan.set_defaults(func=cmd_scan)

    p_export = sub.add_parser("export", help="Export the cached bundle")
    p_export.add_argument("--format", default="json", help="json (only format)")
    p_export.add_argument("--out", default=None, help="Destination path")
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
