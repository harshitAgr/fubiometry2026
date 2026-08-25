#!/usr/bin/env bash
# Reproduce the Lever-3 Phase-2 geo_v1 5-fold CV (adopted 2026-06-18; task-mean MRE 27.79 / paramMAE 22.05).
# Canonical from-scratch run: trains all 5 folds fresh. (The recorded run reused the matched-probe
# geo_v1 fold-0 and was completed via an idempotent resume after an interruption — identical config.)
#
# Prereqs (see TRAINING.md §2-§3): data/ prepared + leak-free folds built (k=5, guard=2, seed=0).
# Environment: BASELINE venv (torch cu130 + albumentations 2.0.8). Seed 42. mem_frac 0.30 (shared GPU).
# Compare vs the photo_v1 baseline in experiments/results/aug_photo_v1/cv_summary.json (task-mean 32.47).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
TTA="--method soft --tta scale --scales 0.92,1.08"
mkdir -p submission/aug_geo_v1 experiments/results/aug_geo_v1
for K in 0 1 2 3 4; do
  echo "=== $(date) geo_v1 fold $K (20 epochs, seed 42, mem_frac 0.30) ==="
  $PY experiments/run_config.py --fold "$K" --epochs 20 --aug geo_v1 --mem-frac 0.30
  $PY experiments/infer_tta.py --checkpoint "runs/cvfold$K/best_model.pth" \
    --split-csv "data/_cvfold${K}_val.csv" --out "submission/aug_geo_v1/cvfold$K" $TTA \
    --gt "data/_cvfold${K}_gt.csv" --results-json "experiments/results/aug_geo_v1/cvfold$K.json"
done
uv run python experiments/aggregate_cv.py \
  --results-glob 'experiments/results/aug_geo_v1/cvfold*.json' \
  --out experiments/results/aug_geo_v1/cv_summary.json
echo "=== $(date) DONE — task-mean MRE = mean of the 9 per-task means in cv_summary.json (expect ~27.8) ==="
