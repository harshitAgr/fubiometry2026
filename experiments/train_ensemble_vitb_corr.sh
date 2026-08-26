#!/usr/bin/env bash
# TEST-PHASE DELIVERABLE — ViT-B 3-seed ensemble on the POST-FEMUR-DROP data.
# MANDATORY retrain: the current best v12 (827004) trained on the 25 mirrored/flipped fetal_femur
# images the organizers told participants to disregard (2026-07-16 email). The shipped model must
# not train on them.
#
# ⚠️ HEADER CORRECTED 2026-08-01: the original version of this script (2026-07-17) also cited the
# A4C/PSAX left-right endpoint swap as a reason to retrain. The organizers RETRACTED that swap on
# 2026-07-18 ("the original CSV was entirely correct") and cardiac was reverted to the original raw
# (commit 45f186c). So the ONLY data delta vs v12 is femur 727->702. Recipe is otherwise identical
# to v12 (runs/vitb_full{,_s43,_s44}):
#   full-data (no held-out fold), geo_v1 + warmup3 + cosine + 40ep, UNIFORM heatmap 64,
#   DINOv2 ViT-B/14@518, seeds 42/43/44. Inference = infer_ensemble.py (heatmap-avg + scale-TTA).
#
# ⚠️ `--fugc-heatmap-size 128` REMOVED 2026-08-01. This script previously passed it while claiming
# "same recipe as v12" -- it is NOT. Proof: epoch-1 per-task train loss of runs/vitb_full (the
# deployed v12 member) matches a known uniform-64 run (runs/fold0_vitb) to ~2e-5 on all 9 tasks,
# but a FUGC@128 run differs on FUGC alone by 1.9e-3 (~80x). Independently, the container path
# reproduces v12's val predictions to ~3e-5 px on FUGC ONLY when decoding at 64 (2.77 px off at
# 128). FUGC@128 belongs to the ViT-S v8/v9 lineage (experiments/train_ensemble.sh); it was
# dropped when the ViT-B spine was built, and the older "geo_cosine40 + FUGC@128 + 3-seed
# ensemble" description of the ViT-B deliverable is wrong on that point. Keeping the flag would
# change TWO variables vs v12 (femur drop + FUGC grid) and ship a config with no ViT-B evidence
# behind it -- the 2026-07-08 per-task-res 5-fold found FUGC "flat (+0.03, no headroom)" at ViT-B.
# The drop rides in via folds.csv (6743 rows, post-drop): --full-data trains on all fold>=0 rows.
# ~3-4h/seed. Idempotent (skips a seed whose checkpoint already exists).
#
# LAUNCH (detached; SEEDS is overridable so the 3 members can be split across both GPUs):
#   CUDA_VISIBLE_DEVICES=0 SEEDS="42 44" setsid nohup bash experiments/train_ensemble_vitb_corr.sh \
#     > logs/ens_vitb_corr_gpu0.log 2>&1 < /dev/null &
#   CUDA_VISIBLE_DEVICES=1 SEEDS="43"    setsid nohup bash experiments/train_ensemble_vitb_corr.sh \
#     > logs/ens_vitb_corr_gpu1.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY=baseline/.venv-baseline/bin/python
mkdir -p logs runs
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
for s in "${SEEDS[@]}"; do
  RUN="vitb_full_corr"; [ "$s" -ne 42 ] && RUN="vitb_full_corr_s${s}"
  if [ -f "runs/${RUN}/best_model.pth" ]; then
    echo "=== [$(date)] SKIP seed=$s (runs/${RUN}/best_model.pth exists) ==="; continue
  fi
  echo "=== [$(date)] training ViT-B ensemble member seed=$s -> runs/${RUN} (CUDA=$CUDA_VISIBLE_DEVICES) ==="
  $PY experiments/run_config.py \
    --full-data --run-name "${RUN}" \
    --aug geo_v1 --warmup 3 --cosine --epochs 40 \
    --encoder dinov2_vitb --input-size 518 \
    --seed "$s" --mem-frac 0.45 \
    2>&1 | tee "logs/${RUN}.log"
  echo "=== [$(date)] DONE seed=$s -> runs/${RUN}/best_model.pth ==="
done
echo "=== [$(date)] ALL CORRECTED ViT-B ENSEMBLE MEMBERS DONE ==="
echo "Next: build submission via infer_ensemble.py over runs/vitb_full_corr{,_s43,_s44} (see reproduce_hc_scale_norm.sh for the ensemble inference invocation)."
