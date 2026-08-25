#!/usr/bin/env bash
# Faithful five-seed post-drop realization of base/HC-small/HC-head uniform3.
# Reuses valid seed-42 artifacts if present. Never scores, zips, or submits.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
GPU="${FUB_GPU:?set FUB_GPU to a verified-free physical GPU index}"
RESULTS=experiments/results/full_family_postdrop_symmetric
PRED=submission/full_family_postdrop_symmetric

if [[ "${1:-}" != "--detached-worker" ]]; then
  mkdir -p logs scratch_tmp
  setsid -f nohup env FUB_GPU="$GPU" "$0" --detached-worker \
    </dev/null >logs/full_family_symmetric.log 2>&1
  echo "Launched detached symmetric family worker; monitor logs/full_family_symmetric.log"
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
    --mode symmetric_five_seed --verify-existing --out "$RESULTS/preflight.json"
else
  uv run python experiments/preflight_full_family.py \
    --mode symmetric_five_seed --out "$RESULTS/preflight.json"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

base_run() { [[ "$1" = 42 ]] && echo vitb_full_corr || echo "vitb_full_corr_s$1"; }
family_run() { local stem="$1" seed="$2"; [[ "$seed" = 42 ]] && echo "$stem" || echo "${stem}_s${seed}"; }

H_CKPTS=()
R_CKPTS=()
TRAINING_MANIFEST_ARGS=()
PRAGMATIC_SEED42=(
  experiments/results/full_family_postdrop_seed42/preflight.json
  experiments/results/full_family_postdrop_seed42/hcsmall_seed42_launch.json
  experiments/results/full_family_postdrop_seed42/hcsmall_seed42_training.json
  experiments/results/full_family_postdrop_seed42/hchead_seed42_launch.json
  experiments/results/full_family_postdrop_seed42/hchead_seed42_training.json
)
PRAGMATIC_PRESENT=0
for artifact in "${PRAGMATIC_SEED42[@]}"; do [[ -s "$artifact" ]] && ((PRAGMATIC_PRESENT += 1)); done
if [[ "$PRAGMATIC_PRESENT" -ne 0 && "$PRAGMATIC_PRESENT" -ne "${#PRAGMATIC_SEED42[@]}" ]]; then
  echo "Partial pragmatic seed-42 provenance exists; finish/repair it before symmetric expansion." >&2
  exit 2
fi
for seed in 42 43 44 45 46; do
  base="runs/$(base_run "$seed")/best_model.pth"
  hrun="$(family_run vitb_full_hcsmall_corr "$seed")"
  rrun="$(family_run vitb_full_hchead_corr "$seed")"
  hmanifest="$RESULTS/hcsmall_seed${seed}_training.json"
  rmanifest="$RESULTS/hchead_seed${seed}_training.json"
  hreceipt="$RESULTS/hcsmall_seed${seed}_launch.json"
  rreceipt="$RESULTS/hchead_seed${seed}_launch.json"
  hpreflight="$RESULTS/preflight.json"
  rpreflight="$RESULTS/preflight.json"
  if [[ "$seed" = 42 && "$PRAGMATIC_PRESENT" -eq "${#PRAGMATIC_SEED42[@]}" ]]; then
    hmanifest=experiments/results/full_family_postdrop_seed42/hcsmall_seed42_training.json
    rmanifest=experiments/results/full_family_postdrop_seed42/hchead_seed42_training.json
    hpreflight=experiments/results/full_family_postdrop_seed42/preflight.json
    rpreflight="$hpreflight"
    hreceipt=experiments/results/full_family_postdrop_seed42/hcsmall_seed42_launch.json
    rreceipt=experiments/results/full_family_postdrop_seed42/hchead_seed42_launch.json
  fi
  if [[ -s "$hreceipt" ]]; then
    "$PY" experiments/audit_full_family_training.py --prepare-launch --verify \
      --family hcsmall --seed "$seed" --checkpoint "runs/$hrun/best_model.pth" \
      --preflight "$hpreflight" --receipt "$hreceipt"
  else
    "$PY" experiments/audit_full_family_training.py --prepare-launch \
      --family hcsmall --seed "$seed" --checkpoint "runs/$hrun/best_model.pth" \
      --preflight "$hpreflight" --receipt "$hreceipt"
  fi
  if [[ -s "runs/$hrun/best_model.pth" ]]; then
    VERIFY=()
    [[ -s "$hmanifest" ]] && VERIFY=(--verify)
    "$PY" experiments/audit_full_family_training.py "${VERIFY[@]}" \
        --family hcsmall --seed "$seed" --checkpoint "runs/$hrun/best_model.pth" \
        --metrics "runs/$hrun/metrics.jsonl" --preflight "$hpreflight" \
        --receipt "$hreceipt" --out "$hmanifest"
  else
    [[ ! -e "$hmanifest" ]] || { echo "H$seed manifest exists without checkpoint." >&2; exit 2; }
    require_empty_gpu
    "$PY" experiments/run_config.py \
      --full-data --run-name "$hrun" \
      --aug geo_v1_hcsmall --warmup 3 --cosine --epochs 40 \
      --encoder dinov2_vitb --input-size 518 --seed "$seed" --mem-frac 0.45
    "$PY" experiments/audit_full_family_training.py \
      --family hcsmall --seed "$seed" --checkpoint "runs/$hrun/best_model.pth" \
      --metrics "runs/$hrun/metrics.jsonl" --preflight "$hpreflight" \
      --receipt "$hreceipt" --out "$hmanifest"
  fi
  if [[ -s "$rreceipt" ]]; then
    "$PY" experiments/audit_full_family_training.py --prepare-launch --verify \
      --family hchead --seed "$seed" --checkpoint "runs/$rrun/best_model.pth" \
      --preflight "$rpreflight" --receipt "$rreceipt"
  else
    "$PY" experiments/audit_full_family_training.py --prepare-launch \
      --family hchead --seed "$seed" --checkpoint "runs/$rrun/best_model.pth" \
      --preflight "$rpreflight" --receipt "$rreceipt"
  fi
  if [[ -s "runs/$rrun/best_model.pth" ]]; then
    VERIFY=()
    [[ -s "$rmanifest" ]] && VERIFY=(--verify)
    "$PY" experiments/audit_full_family_training.py "${VERIFY[@]}" \
        --family hchead --seed "$seed" --checkpoint "runs/$rrun/best_model.pth" \
        --metrics "runs/$rrun/metrics.jsonl" --preflight "$rpreflight" \
        --base "$base" --receipt "$rreceipt" --out "$rmanifest"
  else
    [[ ! -e "$rmanifest" ]] || { echo "R$seed manifest exists without checkpoint." >&2; exit 2; }
    require_empty_gpu
    "$PY" experiments/run_config.py \
      --full-data --run-name "$rrun" \
      --aug geo_v1_hcsmall --epochs 5 \
      --encoder dinov2_vitb --input-size 518 --seed "$seed" --mem-frac 0.28 \
      --init-checkpoint "$base" --train-task HC --head-lr 1e-4
    "$PY" experiments/audit_full_family_training.py \
      --family hchead --seed "$seed" --checkpoint "runs/$rrun/best_model.pth" \
      --metrics "runs/$rrun/metrics.jsonl" --preflight "$rpreflight" \
      --base "$base" --receipt "$rreceipt" --out "$rmanifest"
  fi
  "$PY" experiments/audit_head_refinement.py \
    --before "$base" --after "runs/$rrun/best_model.pth" --task HC \
    --out "$RESULTS/hchead_seed${seed}_state_audit.json"
  test "$(jq -r '.n_changed_tensors' "$RESULTS/hchead_seed${seed}_state_audit.json")" = 14
  H_CKPTS+=("runs/$hrun/best_model.pth")
  R_CKPTS+=("runs/$rrun/best_model.pth")
  TRAINING_MANIFEST_ARGS+=(--training-manifest "$hmanifest" --training-manifest "$rmanifest")
done

join_by_comma() { local IFS=,; echo "$*"; }
COMMON=(--data-root data/val --method soft --tta scale --scales 0.92,1.08 --window 9
  --encoder dinov2_vitb --input-size 518 --heatmap-size 64 --mem-frac 0.40)
if [[ -e "$PRED" ]]; then
  [[ -s "$PRED/regression_predictions.json" && -s "$PRED/candidate_audit.json" ]] || {
    echo "Existing symmetric prediction directory is incomplete." >&2; exit 2; }
  "$PY" experiments/full_family_candidate.py --verify \
    --mode symmetric_five_seed --preflight-manifest "$RESULTS/preflight.json" \
    --base submission/vitb5_val_candidates/ensemble5/regression_predictions.json \
    --hcsmall "$PRED/hcsmall_family/regression_predictions.json" \
    --hchead "$PRED/hchead_family/regression_predictions.json" \
    --out "$PRED/regression_predictions.json" --audit-out "$PRED/candidate_audit.json" \
    "${TRAINING_MANIFEST_ARGS[@]}"
  if [[ ! -e "$RESULTS/candidate_audit.json" ]]; then
    cp "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json"
  else
    cmp -s "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json" || {
      echo "Retained symmetric candidate audits conflict." >&2; exit 2; }
  fi
  echo "Symmetric family candidate already complete."
  exit 0
fi
STAGE="$(mktemp -d scratch_tmp/full_family_symmetric_infer.XXXXXX)"
require_empty_gpu
"$PY" experiments/infer_ensemble.py --checkpoints "$(join_by_comma "${H_CKPTS[@]}")" \
  "${COMMON[@]}" --out "$STAGE/hcsmall_family"
require_empty_gpu
"$PY" experiments/infer_ensemble.py --checkpoints "$(join_by_comma "${R_CKPTS[@]}")" \
  "${COMMON[@]}" --out "$STAGE/hchead_family"
"$PY" experiments/full_family_candidate.py \
  --mode symmetric_five_seed --preflight-manifest "$RESULTS/preflight.json" \
  --base submission/vitb5_val_candidates/ensemble5/regression_predictions.json \
  --hcsmall "$STAGE/hcsmall_family/regression_predictions.json" \
  --hchead "$STAGE/hchead_family/regression_predictions.json" \
  --out "$STAGE/regression_predictions.json" --audit-out "$STAGE/candidate_audit.json" \
  "${TRAINING_MANIFEST_ARGS[@]}"
mv "$STAGE" "$PRED"
cp "$PRED/candidate_audit.json" "$RESULTS/candidate_audit.json"
