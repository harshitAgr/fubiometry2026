#!/usr/bin/env bash
# ENCODER-CAPACITY GATE — DINOv2 ViT-B, full 5-fold, NO external data. Uses the adopted
# folds.csv (identical val folds to geo_cosine40), so this is a clean paired ViT-S->ViT-B
# A/B on the same data. External data was evaluated (femur-only 5-fold, task-mean -0.56
# within +/-0.98 noise) and NOT adopted; this isolates the encoder lever alone.
# Recipe = geo_cosine40 (geo_v1 + warmup3 + cosine, 40ep, soft+scale-TTA). ~2 hr/fold.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
mkdir -p experiments/results/geo_cosine40_vitb submission/geo_cosine40_vitb
for K in 0 1 2 3 4; do
  echo "=== ViT-B fold $K train (geo_v1, warmup3+cosine, 40ep, no-ext) $(date) ==="
  $PY experiments/run_config.py --fold "$K" --epochs 40 --aug geo_v1 --warmup 3 --cosine \
    --mem-frac 0.45 --encoder dinov2_vitb --input-size 518 \
    --folds-csv data/folds/folds.csv --run-name fold${K}_vitb
  echo "=== ViT-B fold $K score $(date) ==="
  $PY experiments/infer_tta.py --checkpoint "runs/fold${K}_vitb/best_model.pth" \
    --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_cvfold${K}_val.csv" --gt "data/_cvfold${K}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 \
    --out "submission/geo_cosine40_vitb/cvfold$K" \
    --results-json "experiments/results/geo_cosine40_vitb/cvfold$K.json"
done
echo "=== DONE — PAIRED ViT-S vs ViT-B per fold $(date) ==="
$PY - <<'PY'
import json, os, numpy as np
def load(p): return json.load(open(p)) if os.path.exists(p) else None
TASKS=['A4C','AOP','FA','FUGC','HC','IVC','PLAX','PSAX','fetal_femur']
def tm(d):
    pt=d.get('per_task',{}); vs=[pt[t]['mre'] for t in TASKS if t in pt]
    return float(np.mean(vs)) if vs else float('nan')
d=[]
print(f"{'fold':>5}{'ViT-S':>10}{'ViT-B':>10}{'delta':>9}")
for K in range(5):
    a=load(f'experiments/results/geo_cosine40/cvfold{K}.json'); b=load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json')
    if not(a and b): print(f"{K:>5}  missing"); continue
    ta,tb=tm(a),tm(b); d.append(tb-ta); print(f"{K:>5}{ta:10.3f}{tb:10.3f}{tb-ta:+9.3f}")
if d:
    print(f"\ntask-mean delta: {np.mean(d):+.3f} +/- {np.std(d,ddof=1) if len(d)>1 else 0:.3f} (n={len(d)}); adopted 25.48")
    A={t:[] for t in TASKS}; B={t:[] for t in TASKS}
    for K in range(5):
        a=load(f'experiments/results/geo_cosine40/cvfold{K}.json'); b=load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json')
        if a and b:
            for t in TASKS:
                A[t].append(a['per_task'][t]['mre']); B[t].append(b['per_task'][t]['mre'])
    print(f"{'task':13}{'ViT-S':>9}{'ViT-B':>9}{'delta':>8}")
    for t in TASKS: print(f"{t:13}{np.mean(A[t]):9.3f}{np.mean(B[t]):9.3f}{np.mean(B[t])-np.mean(A[t]):+8.3f}")
PY
