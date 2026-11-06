#!/usr/bin/env bash
# Rerun every experiment of the paper (seeds 42, 0, 1), then rebuild the tables and figures.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 42 0 1; do TGRAD_SEED=$s python scripts/01_elliptic.py > results/log_elliptic_seed$s.txt 2>&1; done
for s in 42 0 1; do TGRAD_SEED=$s python scripts/02_scaling.py > results/log_scaling_seed$s.txt 2>&1; done
python scripts/06_latency.py
python scripts/03_benchmark_full.py
python scripts/04_summary.py
python scripts/05_detector_figure.py
