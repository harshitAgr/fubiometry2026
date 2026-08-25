#!/usr/bin/env bash
# HC SMALL-HEAD ZOOM-OUT AUGMENTATION LEVER — DINOv2 ViT-B, paired 5-fold vs geo_cosine40_vitb.
#
# Lever: an HC-ONLY aggressive zoom-out / scale-jitter augmentation (geo_v1_hcsmall) that
# synthesizes small heads at TRAIN time (Affine scale=(0.5,1.05), black-padded border), applied
# ONLY to the HC task via KeypointAugDataset.task_transforms. Every other task keeps base geo_v1
# unchanged. Motivation: HC is tail-dominated by small/faint heads; a synthetic wide-FOV test
# recovered -3.36px, so the remaining lever is training-side small-head exposure.
#
# Recipe = the adopted geo_cosine40_vitb spine (geo_v1 + warmup3 + cosine, 40ep, ViT-B@518,
# soft decode + scale-TTA {0.92,1.08}) with --aug geo_v1_hcsmall swapped in. Uses the SAME
# adopted folds.csv, so this is a clean paired A/B vs experiments/results/geo_cosine40_vitb/.
#
# Determinism: seed 42 (run_config default) + the documented folds.csv build. ~2 hr/fold on the
# Blackwell GPU. Pin to GPU 0; --mem-frac 0.85 (owns GPU 0).
#
# NOTE the NAMESPACED split/gt csvs: data/_hcsmall_cvfold{K}_{val,gt}.csv — used instead of the
# shared data/_cvfold{K}_*.csv so a co-running sibling job can't clobber them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
mkdir -p experiments/results/geo_cosine40_vitb_hcsmall submission/hcsmall

for K in 0 1 2 3 4; do
  echo "=== HCSMALL ViT-B fold $K train (geo_v1_hcsmall, warmup3+cosine, 40ep) $(date) ==="
  CUDA_VISIBLE_DEVICES=0 $PY experiments/run_config.py \
    --fold "$K" --epochs 40 --aug geo_v1_hcsmall --warmup 3 --cosine --mem-frac 0.85 \
    --encoder dinov2_vitb --input-size 518 --folds-csv data/folds/folds.csv \
    --run-name fold${K}_hcsmall
  echo "=== HCSMALL ViT-B fold $K score $(date) ==="
  CUDA_VISIBLE_DEVICES=0 $PY experiments/infer_tta.py \
    --checkpoint "runs/fold${K}_hcsmall/best_model.pth" \
    --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_hcsmall_cvfold${K}_val.csv" --gt "data/_hcsmall_cvfold${K}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 \
    --out "submission/hcsmall/cvfold$K" \
    --results-json "experiments/results/geo_cosine40_vitb_hcsmall/cvfold$K.json"
done

echo "=== aggregate paired deltas vs geo_cosine40_vitb $(date) ==="
.venv/bin/python experiments/aggregate_cv.py \
  --results-glob 'experiments/results/geo_cosine40_vitb_hcsmall/cvfold*.json' \
  --out experiments/results/geo_cosine40_vitb_hcsmall/cv_summary.json

# Paired per-fold task-mean + per-task deltas vs the adopted ViT-B baseline.
.venv/bin/python - <<'PY'
import json, os, numpy as np
TASKS=['A4C','AOP','FA','FUGC','HC','IVC','PLAX','PSAX','fetal_femur']
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def tm(d):
    pt=d.get('per_task',{}); vs=[pt[t]['mre'] for t in TASKS if t in pt]
    return float(np.mean(vs)) if vs else float('nan')
d=[]
print(f"{'fold':>5}{'base':>9}{'hcsmall':>9}{'delta':>8}{'HC_base':>9}{'HC_hcs':>9}{'HC_d':>8}")
for K in range(5):
    a=load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json')
    b=load(f'experiments/results/geo_cosine40_vitb_hcsmall/cvfold{K}.json')
    if not(a and b): print(f"{K:>5}  missing"); continue
    ta,tb=tm(a),tm(b); ha,hb=a['per_task']['HC']['mre'],b['per_task']['HC']['mre']
    d.append(tb-ta); print(f"{K:>5}{ta:9.3f}{tb:9.3f}{tb-ta:+8.3f}{ha:9.3f}{hb:9.3f}{hb-ha:+8.3f}")
if d:
    print(f"\ntask-mean delta: {np.mean(d):+.3f} +/- {np.std(d,ddof=1) if len(d)>1 else 0:.3f} (n={len(d)}); adopted 24.14")
    A={t:[] for t in TASKS}; B={t:[] for t in TASKS}
    for K in range(5):
        a=load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json')
        b=load(f'experiments/results/geo_cosine40_vitb_hcsmall/cvfold{K}.json')
        if a and b:
            for t in TASKS: A[t].append(a['per_task'][t]['mre']); B[t].append(b['per_task'][t]['mre'])
    print(f"{'task':13}{'base':>9}{'hcsmall':>9}{'delta':>8}")
    for t in TASKS: print(f"{t:13}{np.mean(A[t]):9.3f}{np.mean(B[t]):9.3f}{np.mean(B[t])-np.mean(A[t]):+8.3f}")
PY
echo "=== DONE $(date) ==="
