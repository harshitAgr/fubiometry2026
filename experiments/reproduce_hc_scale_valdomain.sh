#!/usr/bin/env bash
# HC ellipse-scale correction, fitted OUT OF SAMPLE on the validation domain.
#
#   1. hc_valdomain_harness.py  - match challenge val images to pixel-identical public
#                                 FETAL_PLANES_DB images; gate the endpoint-order rule
#                                 against our own HC train GT; score deployed artifacts.
#   2. build_fp_head_probe.py   - assemble 1,484 FP head images that are patient-disjoint
#                                 from every val-matched image (and unseen in training).
#   3. infer_ensemble.py        - deployed 5-seed ViT-B ensemble over that fitting set.
#   4. fit_hc_scale.py          - fit one centroid-preserving scale there, FREEZE it, and
#                                 apply it to the 151 val-matched images.
#
# Inference only: no training, no fold scratch, no submission. EVALUATION ONLY --
# never train on the matched images (it would destroy the harness and inflate the
# hidden test score unmeasurably).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
GPU="${GPU:-1}"
BASE="runs/vitb_full_corr/best_model.pth,runs/vitb_full_corr_s43/best_model.pth,runs/vitb_full_corr_s44/best_model.pth,runs/vitb_full_corr_s45/best_model.pth,runs/vitb_full_corr_s46/best_model.pth"

$PY experiments/hc_valdomain_harness.py
$PY experiments/build_fp_head_probe.py
CUDA_VISIBLE_DEVICES="$GPU" $PY experiments/infer_ensemble.py \
  --checkpoints "$BASE" \
  --data-root data/_fpheads --split-csv data/_fpheads/split.csv \
  --method soft --tta scale --scales 0.92,1.08 --window 9 \
  --encoder dinov2_vitb --input-size 518 --heatmap-size 64 --mem-frac 0.40 \
  --out submission/fp_head_probe
$PY experiments/fit_hc_scale.py
