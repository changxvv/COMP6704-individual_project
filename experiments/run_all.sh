#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/figures" "${REPO_ROOT}/opentuner.db"

cd "${REPO_ROOT}"

echo "[run_all] Running experiments (all, 3 reps)..."
uv run python experiments/run_experiments.py --experiment=all --reps=3

echo "[run_all] Generating plots..."
uv run python experiments/plot_results.py

echo "[run_all] Done. Results are in experiments/results, figures in experiments/figures."
