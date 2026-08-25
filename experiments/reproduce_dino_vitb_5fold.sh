#!/usr/bin/env bash
# Continued-DINO ViT-B/14 — challenge-unlabelled-pool pretrain followed by five-fold CV.
#
# Adoption comparison: geo_cosine40_vitb post-drop re-score, task-mean MRE 23.00. Validation folds,
# supervised recipe, seed, decoder and TTA match. Caveat: the adopted checkpoints were trained
# before 25 mirrored fetal-femur frames were dropped, whereas this arm uses today's corrected
# 6,727-image eligible training pool. A borderline result is therefore not causally attributable
# to DINO continuation; only a decisive, sign-consistent win clears the adoption gate.
#
# The five folds are partitioned across two GPUs. Fold-specific checkpoints, predictions, and
# data/_cvfoldK scratch CSVs are disjoint between workers; never give the same K to both workers.
# Run fully detached (see TRAINING.md §7):
#   setsid nohup bash experiments/reproduce_dino_vitb_5fold.sh >logs/dino_vitb_5fold.log 2>&1 </dev/null &
# To reuse an already completed pretrain after an orchestrator restart, add --skip-pretrain.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
SSL_RUN=runs/dino_ssl_vitb
OUT=dino_ssl_vitb_5fold
mkdir -p "$SSL_RUN" "experiments/results/$OUT" "submission/$OUT" logs

if [[ "${1:-}" != "--skip-pretrain" ]]; then
  echo "=== $(date -u) [dino-vitb] continue DINO on current 24,467-image challenge manifest ==="
  # ViT-S used batch 24 / lr 5e-5. Batch 16 and lr 3.333e-5 preserve linear LR scaling.
  CUDA_VISIBLE_DEVICES=0 $PY experiments/dino_pretrain.py \
    --encoder dinov2_vitb --epochs 3 --batch-size 16 --lr 3.333333333e-5 \
    --out "$SSL_RUN" --mem-frac 0.45 --log-every 50 \
    --results-json "experiments/results/$OUT/pretrain_log.json"
else
  test -s "$SSL_RUN/encoder.pth"
  $PY - <<'PY'
import json
with open('experiments/results/dino_ssl_vitb_5fold/pretrain_log.json') as handle:
    log = json.load(handle)
assert log['args']['encoder'] == 'dinov2_vitb', log['args']['encoder']
assert log['steps_run'] == log['total_steps'], (log.get('steps_run'), log.get('total_steps'))
assert log['final_loss'] is not None
print(f"validated completed pretrain: {log['steps_run']} steps, final EMA loss {log['final_loss']:.4f}")
PY
  echo "=== $(date -u) [dino-vitb] reusing completed $SSL_RUN/encoder.pth ==="
fi

sha256sum "$SSL_RUN/encoder.pth" | tee "experiments/results/$OUT/encoder.sha256"

run_folds() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  for K in "$@"; do
    echo "=== $(date -u) [dino-vitb gpu=$gpu] fold $K train ==="
    $PY experiments/run_config.py --fold "$K" --epochs 40 --aug geo_v1 --warmup 3 --cosine \
      --mem-frac 0.45 --encoder dinov2_vitb --input-size 518 \
      --encoder-init "$SSL_RUN/encoder.pth" --folds-csv data/folds/folds.csv \
      --run-name "fold${K}_dino_vitb"

    echo "=== $(date -u) [dino-vitb gpu=$gpu] fold $K score ==="
    $PY experiments/infer_tta.py --checkpoint "runs/fold${K}_dino_vitb/best_model.pth" \
      --encoder dinov2_vitb --input-size 518 \
      --split-csv "data/_cvfold${K}_val.csv" --gt "data/_cvfold${K}_gt.csv" \
      --method soft --tta scale --scales 0.92,1.08 \
      --out "submission/$OUT/cvfold$K" \
      --results-json "experiments/results/$OUT/cvfold$K.json"
  done
}

echo "=== $(date -u) [dino-vitb] parallel folds: GPU0={0,2,4}; GPU1={1,3} ==="
run_folds 0 0 2 4 >logs/dino_vitb_folds_gpu0.log 2>&1 &
gpu0_pid=$!
run_folds 1 1 3 >logs/dino_vitb_folds_gpu1.log 2>&1 &
gpu1_pid=$!
worker_status=0
wait "$gpu0_pid" || worker_status=$?
wait "$gpu1_pid" || worker_status=$?
if (( worker_status != 0 )); then
  echo "one or more GPU fold workers failed; inspect logs/dino_vitb_folds_gpu{0,1}.log" >&2
  exit "$worker_status"
fi

uv run python experiments/aggregate_cv.py \
  --results-glob "experiments/results/$OUT/cvfold*.json" \
  --out "experiments/results/$OUT/cv_summary.json"

echo "=== $(date -u) DONE — compare continued-DINO against adopted post-drop baseline ==="
$PY - <<'PY'
import json
import numpy as np

tasks = ['A4C','AOP','FA','FUGC','HC','IVC','PLAX','PSAX','fetal_femur']
def load(path):
    with open(path) as handle:
        return json.load(handle)
def task_mean(result):
    return float(np.mean([result['per_task'][task]['mre'] for task in tasks]))

deltas = []
print(f"{'fold':>5}{'control':>11}{'cont-DINO':>12}{'delta':>10}")
for fold in range(5):
    control = load(f'experiments/results/geo_cosine40_vitb/cvfold{fold}_postdrop.json')
    treatment = load(f'experiments/results/dino_ssl_vitb_5fold/cvfold{fold}.json')
    a, b = task_mean(control), task_mean(treatment)
    deltas.append(b - a)
    print(f"{fold:>5}{a:11.3f}{b:12.3f}{b-a:+10.3f}")
print(f"mean paired delta {np.mean(deltas):+.3f}; "
      f"sample SD {np.std(deltas, ddof=1):.3f}; signs "
      f"{sum(d < 0 for d in deltas)}/5 improved")
PY
