#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
python3 -m vol_scanner.pipeline.run "$@"
