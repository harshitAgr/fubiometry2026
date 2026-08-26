#!/usr/bin/env bash
# Train the N-seed ENSEMBLE (the test-phase deliverable).
#
# Recipe per member = the adopted spine + the per-task FUGC precision lever:
#   geo_v1 aug + LinearWarmup(3)+Cosine + 40 epochs, FUGC@128 (per-task heatmap), DINOv2 ViT-S@518,
#   100% data (no held-out fold). One model per seed; heatmap-space averaging at inference
#   (experiments/infer_ensemble.py). The inert AOP probe-angle aug is dropped (it never moved OOD AOP).
#
# Variance reduction is the point: it denoises the noisy tiny tasks (IVC/PSAX/HC = the biggest
# non-AOP headroom) and compounds the FUGC@128 win. Run with the baseline venv. ~3-4h/model.
#
#   nohup bash experiments/train_ensemble.sh > logs/train_ensemble.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
mkdir -p logs runs
SEEDS=(42 43 44)  # 3-seed ensemble (1/√3≈0.58 of single-model noise; 5 was diminishing returns)
for s in "${SEEDS[@]}"; do
  if [ -f "runs/ens_s${s}/best_model.pth" ]; then
    echo "=== [$(date)] SKIP seed=$s (runs/ens_s${s}/best_model.pth exists) ==="
    continue
  fi
  echo "=== [$(date)] training ensemble member seed=$s ==="
  $PY experiments/run_config.py \
    --full-data --run-name "ens_s${s}" \
    --aug geo_v1 --warmup 3 --cosine --epochs 40 \
    --fugc-heatmap-size 128 --seed "$s" --mem-frac 0.5 \
    2>&1 | tee "logs/ens_s${s}.log"
  echo "=== [$(date)] DONE seed=$s -> runs/ens_s${s}/best_model.pth ==="
done
echo "=== [$(date)] ALL ENSEMBLE MEMBERS DONE ==="
