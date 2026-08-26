#!/usr/bin/env bash
# Pragmatic post-drop family hedge: B5 + one H42 + one base-initialized R42.
# Candidate artifact only. This script never scores, zips, or submits.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
GPU="${FUB_GPU:?set FUB_GPU to a verified-free physical GPU index}"
RESULTS=experiments/results/full_family_postdrop_seed42
PRED=submission/full_family_postdrop_seed42
BASE=runs/vitb_full_corr/best_model.pth
H_RUN=runs/vitb_full_hcsmall_corr
R_RUN=runs/vitb_full_hchead_corr

if [[ "${1:-}" != "--detached-worker" ]]; then
  mkdir -p logs scratch_tmp
  setsid -f nohup env FUB_GPU="$GPU" "$0" --detached-worker \
    </dev/null >logs/full_family_seed42.log 2>&1
  echo "Launched detached seed-42 family worker; monitor logs/full_family_seed42.log"
  exit 0
fi
shift
SID="$(ps -o sid= -p $$ | tr -d ' ')"
if [[ "$PPID" -ne 1 || "$SID" != "$$" ]]; then
  echo "Family worker must run detached with PPID=1 and SID=PID." >&2
  exit 2
fi
exec 9>scratch_tmp/full_family.lock
flock -n 9 || { echo "Another full-family worker holds the shared lock." >&2; exit 2; }

require_empty_gpu() {
  local active
  active="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader)"
  [[ -z "$active" ]] || { echo "Refusing occupied GPU $GPU: $active" >&2; exit 2; }
}
require_empty_gpu

mkdir -p "$RESULTS"
if [[ -s "$RESULTS/preflight.json" ]]; then
  uv run python experiments/preflight_full_family.py \
    --mode pragmatic_seed42 --verify-existing --out "$RESULTS/preflight.json"
else
  uv run python experiments/preflight_full_family.py \
    --mode pragmatic_seed42 --out "$RESULTS/preflight.json"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

H_MANIFEST="$RESULTS/hcsmall_seed42_training.json"
R_MANIFEST="$RESULTS/hchead_seed42_training.json"
H_RECEIPT="$RESULTS/hcsmall_seed42_launch.json"
R_RECEIPT="$RESULTS/hchead_seed42_launch.json"
if [[ -s "$H_RECEIPT" ]]; then
  "$PY" experiments/audit_full_family_training.py --prepare-launch --verify \
    --family hcsmall --seed 42 --checkpoint "$H_RUN/best_model.pth" \
    --preflight "$RESULTS/preflight.json" --receipt "$H_RECEIPT"
else
  "$PY" experiments/audit_full_family_training.py --prepare-launch \
    --family hcsmall --seed 42 --checkpoint "$H_RUN/best_model.pth" \
    --preflight "$RESULTS/preflight.json" --receipt "$H_RECEIPT"
fi
if [[ -s "$H_RUN/best_model.pth" ]]; then
  VERIFY=()
  [[ -s "$H_MANIFEST" ]] && VERIFY=(--verify)
  "$PY" experiments/audit_full_family_training.py "${VERIFY[@]}" \
      --family hcsmall --seed 42 --checkpoint "$H_RUN/best_model.pth" \
      --metrics "$H_RUN/metrics.jsonl" --preflight "$RESULTS/preflight.json" \
      --receipt "$H_RECEIPT" --out "$H_MANIFEST"
else
  [[ ! -e "$H_MANIFEST" ]] || { echo "H42 manifest exists without checkpoint." >&2; exit 2; }
  require_empty_gpu
  "$PY" experiments/run_config.py \
    --full-data --run-name vitb_full_hcsmall_corr \
    --aug geo_v1_hcsmall --warmup 3 --cosine --epochs 40 \
    --encoder dinov2_vitb --input-size 518 --seed 42 --mem-frac 0.45
  "$PY" experiments/audit_full_family_training.py \
    --family hcsmall --seed 42 --checkpoint "$H_RUN/best_model.pth" \
    --metrics "$H_RUN/metrics.jsonl" --preflight "$RESULTS/preflight.json" \
    --receipt "$H_RECEIPT" --out "$H_MANIFEST"
fi
if [[ -s "$R_RECEIPT" ]]; then
  "$PY" experiments/audit_full_family_training.py --prepare-launch --verify \
    --family hchead --seed 42 --checkpoint "$R_RUN/best_model.pth" \
    --preflight "$RESULTS/preflight.json" --receipt "$R_RECEIPT"
else
  "$PY" experiments/audit_full_family_training.py --prepare-launch \
    --family hchead --seed 42 --checkpoint "$R_RUN/best_model.pth" \
    --preflight "$RESULTS/preflight.json" --receipt "$R_RECEIPT"
fi
if [[ -s "$R_RUN/best_model.pth" ]]; then
  VERIFY=()
  [[ -s "$R_MANIFEST" ]] && VERIFY=(--verify)
  "$PY" experiments/audit_full_family_training.py "${VERIFY[@]}" \
      --family hchead --seed 42 --checkpoint "$R_RUN/best_model.pth" \
      --metrics "$R_RUN/metrics.jsonl" --preflight "$RESULTS/preflight.json" \
      --base "$BASE" --receipt "$R_RECEIPT" --out "$R_MANIFEST"
else
  [[ ! -e "$R_MANIFEST" ]] || { echo "R42 manifest exists without checkpoint." >&2; exit 2; }
  require_empty_gpu
  "$PY" experiments/run_config.py \
    --full-data --run-name vitb_full_hchead_corr \
    --aug geo_v1_hcsmall --epochs 5 \
    --encoder dinov2_vitb --input-size 518 --seed 42 --mem-frac 0.28 \
    --init-checkpoint "$BASE" --train-task HC --head-lr 1e-4
  "$PY" experiments/audit_full_family_training.py \
    --family hchead --seed 42 --checkpoint "$R_RUN/best_model.pth" \
    --metrics "$R_RUN/metrics.jsonl" --preflight "$RESULTS/preflight.json" \
    --base "$BASE" --receipt "$R_RECEIPT" --out "$R_MANIFEST"
fi
"$PY" experiments/audit_head_refinement.py \
  --before "$BASE" --after "$R_RUN/best_model.pth" --task HC \
  --out "$RESULTS/hchead_state_audit.json"
test "$(jq -r '.n_changed_tensors' "$RESULTS/hchead_state_audit.json")" = 14

COMMON=(--data-root data/val --method soft --tta scale --scales 0.92,1.08 --window 9
  --encoder dinov2_vitb --input-size 518 --heatmap-size 64 --mem-frac 0.40)
if [[ -e "$PRED" ]]; then
  [[ -s "$PRED/regression_predictions.json" && -s "$PRED/candidate_audit.json" ]] || {
    echo "Existing seed-42 prediction directory is incomplete." >&2; exit 2; }
  "$PY" experiments/full_family_candidate.py --verify \
    --mode pragmatic_seed42 --preflight-manifest "$RESULTS/preflight.json" \
    --base submission/vitb5_val_candidates/ensemble5/regression_predictions.json \
    --hcsmall "$PRED/hcsmall_family/regression_predictions.json" \
    --hchead "$PRED/hchead_family/regression_predictions.json" \
    --out "$PRED/regression_predictions.json" --audit-out "$PRED/candidate_audit.json" \
    --training-manifest "$H_MANIFEST" --training-manifest "$R_MANIFEST"
  if [[ ! -e "$RESULTS/candidate_audit.json" ]]; then
    cp "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json"
  else
    cmp -s "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json" || {
      echo "Retained seed-42 candidate audits conflict." >&2; exit 2; }
  fi
  echo "Seed-42 family candidate already complete."
  exit 0
fi
STAGE="$(mktemp -d scratch_tmp/full_family_seed42_infer.XXXXXX)"
require_empty_gpu
"$PY" experiments/infer_ensemble.py --checkpoints "$H_RUN/best_model.pth" \
  "${COMMON[@]}" --out "$STAGE/hcsmall_family"
require_empty_gpu
"$PY" experiments/infer_ensemble.py --checkpoints "$R_RUN/best_model.pth" \
  "${COMMON[@]}" --out "$STAGE/hchead_family"
"$PY" experiments/full_family_candidate.py \
  --mode pragmatic_seed42 --preflight-manifest "$RESULTS/preflight.json" \
  --base submission/vitb5_val_candidates/ensemble5/regression_predictions.json \
  --hcsmall "$STAGE/hcsmall_family/regression_predictions.json" \
  --hchead "$STAGE/hchead_family/regression_predictions.json" \
  --out "$STAGE/regression_predictions.json" --audit-out "$STAGE/candidate_audit.json" \
  --training-manifest "$H_MANIFEST" --training-manifest "$R_MANIFEST"
mv "$STAGE" "$PRED"
cp "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json"
