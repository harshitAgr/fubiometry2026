#!/usr/bin/env bash
# Reproduce the Lever-3 photo_v1 5-fold CV (adopted 2026-06-18; task-mean MRE 32.47 / paramMAE 25.19).
#
# Canonical from-scratch run: trains all 5 folds fresh. (The recorded 2026-06-18 run reused the
# fold-0 checkpoint from the matched probe — identical config — to save one training; this script
# does fold 0 fresh too, so it is fully self-contained.)
#
# Prereqs (see TRAINING.md §2-§3):
#   - data/ prepared:           uv run python scripts/prepare_data.py
#   - leak-free folds built:    group-aware k=5, guard=2, seed=0 -> data/folds/folds.csv
#                               (asserts AOP adjacency leak == 0; exact command in TRAINING.md §3)
# Environment: BASELINE venv (torch cu130 + albumentations 2.0.8). Seed 42. mem_frac 0.30 (shared GPU).
# Compare against the adopted Lever-1 baseline in experiments/results/decode_tta/cv_summary.json (33.73).
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
PY=baseline/.venv-baseline/bin/python
TTA="--method soft --tta scale --scales 0.92,1.08"   # the adopted decode
mkdir -p submission/aug_photo_v1 experiments/results/aug_photo_v1

for K in 0 1 2 3 4; do
  echo "=== $(date) photo_v1 fold $K (20 epochs, seed 42, mem_frac 0.30) ==="
  $PY experiments/run_config.py --fold "$K" --epochs 20 --aug photo_v1 --mem-frac 0.30
  $PY experiments/infer_tta.py --checkpoint "runs/cvfold$K/best_model.pth" \
    --split-csv "data/_cvfold${K}_val.csv" --out "submission/aug_photo_v1/cvfold$K" $TTA \
    --gt "data/_cvfold${K}_gt.csv" --results-json "experiments/results/aug_photo_v1/cvfold$K.json"
done

uv run python experiments/aggregate_cv.py \
  --results-glob 'experiments/results/aug_photo_v1/cvfold*.json' \
  --out experiments/results/aug_photo_v1/cv_summary.json

echo "=== $(date) DONE — task-mean MRE = mean of the 9 per-task means in cv_summary.json (expect ~32.5) ==="
