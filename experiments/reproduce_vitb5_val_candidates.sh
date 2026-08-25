#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/baseline/.venv-baseline/bin/python"
OUT="$ROOT/submission/vitb5_val_candidates"
RESULTS="$ROOT/experiments/results/vitb5_val_candidates"
GPU="${FUB_GPU:?set FUB_GPU to a verified-free physical GPU index}"
BASE="$ROOT/runs/vitb_full_corr/best_model.pth,$ROOT/runs/vitb_full_corr_s43/best_model.pth,$ROOT/runs/vitb_full_corr_s44/best_model.pth"
ALL="$BASE,$ROOT/runs/vitb_full_corr_s45/best_model.pth,$ROOT/runs/vitb_full_corr_s46/best_model.pth"

if [[ -e "$OUT" || -e "$RESULTS" ]]; then
  echo "Refusing to overwrite existing ViT-B 3/4/5 candidate artifacts." >&2
  exit 2
fi
for checkpoint in \
  "$ROOT/runs/vitb_full_corr/best_model.pth" \
  "$ROOT/runs/vitb_full_corr_s43/best_model.pth" \
  "$ROOT/runs/vitb_full_corr_s44/best_model.pth" \
  "$ROOT/runs/vitb_full_corr_s45/best_model.pth" \
  "$ROOT/runs/vitb_full_corr_s46/best_model.pth"; do
  test -s "$checkpoint"
done
test -s "$ROOT/submission/v15/regression_predictions.json"
test -s "$ROOT/experiments/results/inverse_llrd_full_seeds/report.json"

GPU_PROCESSES="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader)"
if [[ -n "$GPU_PROCESSES" ]]; then
  echo "Refusing to launch on occupied GPU $GPU:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
COMMON=(
  --data-root data/val --method soft --tta scale --scales 0.92,1.08 --window 9
  --encoder dinov2_vitb --input-size 518 --heatmap-size 64 --mem-frac 0.40
)

uv run python experiments/audit_vitb5_candidates.py \
  --snapshot-only \
  --reference submission/v15/regression_predictions.json \
  --checkpoints "$ALL" \
  --checkpoint-provenance experiments/results/inverse_llrd_full_seeds/report.json \
  --out "$RESULTS/preflight_manifest.json"

"$PY" experiments/infer_ensemble.py --checkpoints "$BASE" \
  "${COMMON[@]}" --out "$OUT/ensemble3"
uv run python experiments/audit_vitb5_candidates.py \
  --control-only \
  --reference submission/v15/regression_predictions.json \
  --ensemble3 "$OUT/ensemble3/regression_predictions.json" \
  --checkpoints "$ALL" \
  --checkpoint-provenance experiments/results/inverse_llrd_full_seeds/report.json \
  --preflight-manifest "$RESULTS/preflight_manifest.json" \
  --out "$RESULTS/control_equivalence.json"
"$PY" experiments/infer_ensemble.py \
  --checkpoints "$BASE,$ROOT/runs/vitb_full_corr_s45/best_model.pth" \
  "${COMMON[@]}" --out "$OUT/ensemble4"
"$PY" experiments/infer_ensemble.py \
  --checkpoints "$BASE,$ROOT/runs/vitb_full_corr_s45/best_model.pth,$ROOT/runs/vitb_full_corr_s46/best_model.pth" \
  "${COMMON[@]}" --out "$OUT/ensemble5"
"$PY" experiments/infer_ensemble.py \
  --checkpoints "$ROOT/runs/vitb_full_corr_s45/best_model.pth" \
  "${COMMON[@]}" --out "$OUT/member45"
"$PY" experiments/infer_ensemble.py \
  --checkpoints "$ROOT/runs/vitb_full_corr_s46/best_model.pth" \
  "${COMMON[@]}" --out "$OUT/member46"

uv run python experiments/audit_vitb5_candidates.py \
  --reference submission/v15/regression_predictions.json \
  --ensemble3 "$OUT/ensemble3/regression_predictions.json" \
  --ensemble4 "$OUT/ensemble4/regression_predictions.json" \
  --ensemble5 "$OUT/ensemble5/regression_predictions.json" \
  --member45 "$OUT/member45/regression_predictions.json" \
  --member46 "$OUT/member46/regression_predictions.json" \
  --checkpoints "$ALL" \
  --checkpoint-provenance experiments/results/inverse_llrd_full_seeds/report.json \
  --preflight-manifest "$RESULTS/preflight_manifest.json" \
  --out "$RESULTS/prediction_audit.json"
