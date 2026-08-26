#!/usr/bin/env bash
# Conditional full-data continued-DINO family training. Checkpoints only: NO val inference,
# Docker copy/build, packaging, or submission. Must receive an explicit passing decision JSON;
# both the conservative and expected-score policies preserve their policy in that artifact.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
DECISION="${1:-experiments/results/dino_ssl_vitb_5fold/decision.json}"
ENC=runs/dino_ssl_vitb/encoder.pth
OUT=experiments/results/dino_ssl_vitb_full_family
mkdir -p "$OUT" logs

test "$(jq -r '.deploy' "$DECISION")" = true || {
  echo "continued-DINO deployment gate did not pass; refusing full-data training" >&2
  exit 3
}
test "$(jq -r '.policy' "$DECISION")" = strict \
  || test "$(jq -r '.policy' "$DECISION")" = expected_score \
  || { echo "decision policy is absent or invalid" >&2; exit 3; }
test -s "$ENC"
cp "$DECISION" "$OUT/cv_decision.json"
for gpu in 0 1; do
  active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader)"
  [[ -z "$active" ]] || { echo "GPU $gpu is occupied: $active" >&2; exit 4; }
done

run_complete() {
  local run="$1" epochs="$2"
  [[ -s "runs/$run/best_model.pth" && -s "runs/$run/training_manifest.json" ]] || return 1
  [[ "$(wc -l <"runs/$run/metrics.jsonl")" -eq "$epochs" ]]
}

train_base() {
  local gpu="$1" seed="$2" run="vitb_full_dino_corr"
  [[ "$seed" -eq 42 ]] || run="${run}_s${seed}"
  if run_complete "$run" 40; then echo "SKIP complete $run"; return; fi
  [[ ! -e "runs/$run/best_model.pth" ]] || { echo "partial $run" >&2; return 5; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/run_config.py \
    --full-data --run-name "$run" --aug geo_v1 --warmup 3 --cosine --epochs 40 \
    --encoder dinov2_vitb --encoder-init "$ENC" --input-size 518 \
    --seed "$seed" --mem-frac 0.45
}

train_hcsmall() {
  local gpu="$1" run=vitb_full_dino_hcsmall_corr
  if run_complete "$run" 40; then echo "SKIP complete $run"; return; fi
  [[ ! -e "runs/$run/best_model.pth" ]] || { echo "partial $run" >&2; return 5; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/run_config.py \
    --full-data --run-name "$run" --aug geo_v1_hcsmall --warmup 3 --cosine --epochs 40 \
    --encoder dinov2_vitb --encoder-init "$ENC" --input-size 518 \
    --seed 42 --mem-frac 0.45
}

train_hchead() {
  local gpu="$1" run=vitb_full_dino_hchead_corr base=runs/vitb_full_dino_corr/best_model.pth
  test -s "$base"
  if run_complete "$run" 5; then echo "SKIP complete $run"; return; fi
  [[ ! -e "runs/$run/best_model.pth" ]] || { echo "partial $run" >&2; return 5; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/run_config.py \
    --full-data --run-name "$run" --aug geo_v1_hcsmall --epochs 5 \
    --encoder dinov2_vitb --input-size 518 --seed 42 --mem-frac 0.28 \
    --init-checkpoint "$base" --train-task HC --head-lr 1e-4
}

(
  train_base 0 42
  train_hchead 0
  train_base 0 44
  train_base 0 46
) >logs/dino_vitb_full_gpu0.log 2>&1 &
gpu0_pid=$!
(
  train_base 1 43
  train_base 1 45
  train_hcsmall 1
) >logs/dino_vitb_full_gpu1.log 2>&1 &
gpu1_pid=$!

status=0
wait "$gpu0_pid" || status=$?
wait "$gpu1_pid" || status=$?
(( status == 0 )) || { echo "full-data worker failed; status=$status" >&2; exit "$status"; }

"$PY" experiments/audit_dino_vitb_full_family.py --out "$OUT/audit.json"
echo "FULL FAMILY COMPLETE AND AUDITED. STOPPING BEFORE DOCKER AS REQUIRED."
