#!/usr/bin/env bash
# CPU-only same-data depth-consistency audit; requires full-data seeds 42--46.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 baseline/.venv-baseline/bin/python \
  experiments/preflight_inverse_llrd_full_seeds.py
