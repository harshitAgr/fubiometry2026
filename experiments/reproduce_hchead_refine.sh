#!/usr/bin/env bash
# Checkpoint-initialized HC-head-only refinement.
# Fixed after the fold-0 screen: 5 epochs, head LR 1e-4, geo_v1_hcsmall, window-9 decode.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=baseline/.venv-baseline/bin/python
GPU=${FUB_GPU:-1}
RESULTS=experiments/results/geo_cosine40_vitb_hchead
mkdir -p "$RESULTS" submission/hchead

for K in 0 1 2 3 4; do
  BASE="runs/fold${K}_vitb/best_model.pth"
  RUN="runs/fold${K}_hchead_lr1e4"
  if [[ ! -f "$RUN/best_model.pth" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/run_config.py \
      --fold "$K" --epochs 5 --aug geo_v1_hcsmall \
      --encoder dinov2_vitb --input-size 518 --folds-csv data/folds/folds.csv \
      --init-checkpoint "$BASE" --train-task HC --head-lr 1e-4 \
      --run-name "fold${K}_hchead_lr1e4" --mem-frac 0.28
  fi

  "$PY" experiments/audit_head_refinement.py \
    --before "$BASE" --after "$RUN/best_model.pth" --task HC \
    --out "$RESULTS/cvfold${K}_state_audit.json"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/infer_tta.py \
    --checkpoint "$RUN/best_model.pth" --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_cvfold${K}_val.csv" --gt "data/_cvfold${K}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 --window 9 \
    --out "submission/hchead/cvfold${K}" \
    --results-json "$RESULTS/cvfold${K}.json"
done

uv run python experiments/aggregate_cv.py \
  --results-glob "$RESULTS/cvfold[0-4].json" \
  --out "$RESULTS/cv_summary.json"
