#!/usr/bin/env bash
# Five-fold continued-DINO HC-small + HC-head specialists, followed by the exact
# window-9 Docker route comparison. No official validation, Docker mutation, or submission.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=baseline/.venv-baseline/bin/python
ENC=runs/dino_ssl_vitb/encoder.pth
HCS_RESULTS=experiments/results/dino_ssl_vitb_hcsmall_5fold
HCH_RESULTS=experiments/results/dino_ssl_vitb_hchead_5fold
HCS_PRED=submission/dino_ssl_vitb_hcsmall_5fold
HCH_PRED=submission/dino_ssl_vitb_hchead_5fold
ROUTE_RESULTS=experiments/results/dino_ssl_vitb_full_route
mkdir -p "$HCS_RESULTS" "$HCH_RESULTS" "$HCS_PRED" "$HCH_PRED" "$ROUTE_RESULTS" logs
test -s "$ENC"

for gpu in 0 1; do
  active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader)"
  [[ -z "$active" ]] || { echo "GPU $gpu occupied: $active" >&2; exit 4; }
done

run_complete() {
  local run="$1" epochs="$2"
  [[ -s "runs/$run/best_model.pth" && -s "runs/$run/training_manifest.json" ]] || return 1
  [[ "$(wc -l <"runs/$run/metrics.jsonl")" -eq "$epochs" ]]
}

run_fold() {
  local gpu="$1" fold="$2"
  local base="runs/fold${fold}_dino_vitb/best_model.pth"
  local hch="fold${fold}_dino_vitb_hchead"
  local hcs="fold${fold}_dino_vitb_hcsmall"
  test -s "$base"

  # The HC-head branch is cheap and must change only the HC decoder tensors.
  if ! run_complete "$hch" 5; then
    [[ ! -e "runs/$hch/best_model.pth" ]] || { echo "partial $hch" >&2; return 5; }
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/run_config.py \
      --fold "$fold" --epochs 5 --aug geo_v1_hcsmall \
      --encoder dinov2_vitb --input-size 518 --folds-csv data/folds/folds.csv \
      --init-checkpoint "$base" --train-task HC --head-lr 1e-4 \
      --run-name "$hch" --mem-frac 0.28
  fi
  "$PY" experiments/audit_head_refinement.py \
    --before "$base" --after "runs/$hch/best_model.pth" --task HC \
    --out "$HCH_RESULTS/cvfold${fold}_state_audit.json"
  test "$(jq -r '.n_changed_tensors' "$HCH_RESULTS/cvfold${fold}_state_audit.json")" = 14
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/infer_tta.py \
    --checkpoint "runs/$hch/best_model.pth" --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_cvfold${fold}_val.csv" --gt "data/_cvfold${fold}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 --window 9 \
    --out "$HCH_PRED/cvfold${fold}" --results-json "$HCH_RESULTS/cvfold${fold}.json"

  # HC-small is a full multi-task fold run with the continued-DINO initialization.
  if ! run_complete "$hcs" 40; then
    [[ ! -e "runs/$hcs/best_model.pth" ]] || { echo "partial $hcs" >&2; return 5; }
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/run_config.py \
      --fold "$fold" --epochs 40 --aug geo_v1_hcsmall --warmup 3 --cosine \
      --encoder dinov2_vitb --encoder-init "$ENC" --input-size 518 \
      --folds-csv data/folds/folds.csv --run-name "$hcs" --mem-frac 0.45
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/infer_tta.py \
    --checkpoint "runs/$hcs/best_model.pth" --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_cvfold${fold}_val.csv" --gt "data/_cvfold${fold}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 --window 9 \
    --out "$HCS_PRED/cvfold${fold}" --results-json "$HCS_RESULTS/cvfold${fold}.json"
}

worker() {
  local gpu="$1"; shift
  for fold in "$@"; do
    echo "=== $(date -u) GPU $gpu fold $fold ==="
    run_fold "$gpu" "$fold"
  done
}

worker 0 0 2 4 >logs/dino_vitb_specialists_gpu0.log 2>&1 & p0=$!
worker 1 1 3 >logs/dino_vitb_specialists_gpu1.log 2>&1 & p1=$!
status=0
wait "$p0" || status=$?
wait "$p1" || status=$?
(( status == 0 )) || { echo "specialist worker failed: $status" >&2; exit "$status"; }

uv run python experiments/aggregate_cv.py \
  --results-glob "$HCS_RESULTS/cvfold[0-4].json" --out "$HCS_RESULTS/cv_summary.json"
uv run python experiments/aggregate_cv.py \
  --results-glob "$HCH_RESULTS/cvfold[0-4].json" --out "$HCH_RESULTS/cv_summary.json"

"$PY" experiments/analyze_dino_hybrid_route.py \
  --treatment-dir submission/dino_ssl_vitb_5fold_window9 \
  --treatment-hcsmall-dir "$HCS_PRED" \
  --treatment-hchead-dir "$HCH_PRED" \
  --out-dir "$ROUTE_RESULTS"

echo "ALL-NEW CONTINUED-DINO FAMILY OOF ROUTE COMPLETE. DOCKER UNTOUCHED."
