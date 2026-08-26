#!/usr/bin/env bash
# ADOPTED CV baseline (2026-06-20): geo_v1 + LinearWarmup(3) + CosineAnnealingLR, 40 epochs, scored
# soft + scale-TTA. 5-fold task-mean MRE 25.48 (-2.31 vs the prior geo_v1 const+20ep 27.79; also beats
# USFM@224 26.75 -> spine = DINOv2-geo, no USFM). The LR schedule (previously MISSING — constant 2e-5)
# is the dominant lever; matched fold-0 const-vs-cosine confirmed -6.2px.
#
# NOTE: `set -e` is deliberate. The first run of this 5-fold (no set -e) trained all folds fine but every
# fold's scoring hit a transient infer_tta CLI error, and the job STILL printed DONE with an empty results
# dir. set -e makes a scoring failure abort loudly instead of faking success.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
TTA="--method soft --tta scale --scales 0.92,1.08"
mkdir -p experiments/results/geo_cosine40 submission/geo_cosine40
for K in 0 1 2 3 4; do
  echo "=== fold $K train (geo_v1, warmup3+cosine, 40ep) $(date) ==="
  $PY experiments/run_config.py --fold "$K" --epochs 40 --aug geo_v1 --warmup 3 --cosine --mem-frac 0.30
  echo "=== fold $K score (soft+scale-TTA) $(date) ==="
  $PY experiments/infer_tta.py --checkpoint "runs/cvfold$K/best_model.pth" \
    --split-csv "data/_cvfold${K}_val.csv" --gt "data/_cvfold${K}_gt.csv" $TTA \
    --out "submission/geo_cosine40/cvfold$K" --results-json "experiments/results/geo_cosine40/cvfold$K.json"
done
uv run python experiments/aggregate_cv.py --results-glob 'experiments/results/geo_cosine40/cvfold*.json' \
  --out experiments/results/geo_cosine40/cv_summary.json
echo "=== DONE — task-mean MRE (expect ~25.5) = mean of the 9 per-task means in cv_summary.json $(date) ==="
