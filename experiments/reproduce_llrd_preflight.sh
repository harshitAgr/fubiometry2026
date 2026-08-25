#!/usr/bin/env bash
# CPU-only checkpoint drift falsification; launches no training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 baseline/.venv-baseline/bin/python experiments/preflight_llrd.py
