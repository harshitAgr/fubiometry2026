#!/usr/bin/env bash
# RESTATE THE ADOPTED CV BASELINE on the post-femur-drop folds — INFERENCE ONLY, no training.
#
# WHY: the adopted "5-fold task-mean 24.14" (geo_cosine40_vitb, 2026-07-03) was scored on folds
# whose femur VAL data contained the 25 organizer-flagged mirrored images, scored against
# wrong-orientation GT. Re-scoring the SAME fold-0 checkpoint on the post-drop split moved femur
# 39.819 -> 26.435 (-13.38 px) and task-mean 27.843 -> 26.356 with IDENTICAL weights ⇒ the 24.14
# reference is a measurement artifact, not a model property, and is invalid for comparison
# against anything run on today's folds.
#
# This re-scores the five surviving runs/fold{0..4}_vitb checkpoints against the post-drop splits
# to produce a valid reference. Inference only (~10 min/fold) — the checkpoints are unchanged, so
# the train-side effect of the drop is NOT captured here (those models did train on ~20 of the 25
# mirrored images). This isolates the VAL-side artifact exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-baseline/bin/python
CFG=geo_cosine40_vitb
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Regenerate the shared scratch splits from the CURRENT folds.csv (post-drop).
uv run python experiments/make_cv_splits.py --folds 0 1 2 3 4

for K in 0 1 2 3 4; do
  OUT="experiments/results/$CFG/cvfold${K}_postdrop.json"
  if [ -f "$OUT" ]; then echo "=== fold $K already scored ($OUT) — skipping ==="; continue; fi
  echo "=== ViT-B fold $K RE-SCORE on post-drop split $(date) ==="
  $PY experiments/infer_tta.py --checkpoint "runs/fold${K}_vitb/best_model.pth" \
    --encoder dinov2_vitb --input-size 518 \
    --split-csv "data/_cvfold${K}_val.csv" --gt "data/_cvfold${K}_gt.csv" \
    --method soft --tta scale --scales 0.92,1.08 \
    --out "submission/${CFG}_postdrop/cvfold$K" \
    --results-json "$OUT"
done

echo "=== RESTATED BASELINE $(date) ==="
$PY - <<'PY'
import json, os, numpy as np
TASKS = ['A4C','AOP','FA','FUGC','HC','IVC','PLAX','PSAX','fetal_femur']
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def tm(d):
    pt = d.get('per_task', {}); vs = [pt[t]['mre'] for t in TASKS if t in pt]
    return float(np.mean(vs)) if vs else float('nan')

pre, post = [], []
print(f"{'fold':>5}{'PRE-drop':>10}{'POST-drop':>11}{'delta':>9}")
for K in range(5):
    a = load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json')
    b = load(f'experiments/results/geo_cosine40_vitb/cvfold{K}_postdrop.json')
    if not (a and b):
        print(f"{K:>5}  missing"); continue
    ta, tb = tm(a), tm(b); pre.append(ta); post.append(tb)
    print(f"{K:>5}{ta:10.3f}{tb:11.3f}{tb-ta:+9.3f}")

if pre:
    print(f"\n5-fold task-mean: PRE {np.mean(pre):.3f} (the stale 'adopted 24.14' family) "
          f"-> POST {np.mean(post):.3f}   delta {np.mean(post)-np.mean(pre):+.3f}")
    print(f"\n{'task':13}{'PRE':>9}{'POST':>9}{'delta':>8}")
    for t in TASKS:
        A = [load(f'experiments/results/geo_cosine40_vitb/cvfold{K}.json') for K in range(5)]
        B = [load(f'experiments/results/geo_cosine40_vitb/cvfold{K}_postdrop.json') for K in range(5)]
        pa = [d['per_task'][t]['mre'] for d in A if d and t in d.get('per_task', {})]
        pb = [d['per_task'][t]['mre'] for d in B if d and t in d.get('per_task', {})]
        if pa and pb:
            n = min(len(pa), len(pb))
            print(f"{t:13}{np.mean(pa[:n]):9.3f}{np.mean(pb[:n]):9.3f}"
                  f"{np.mean(pb[:n])-np.mean(pa[:n]):+8.3f}")
    print("\nNOTE: same weights in both columns — every delta here is MEASUREMENT (val-list) "
          "change, not model improvement. Only femur should move; any other task moving "
          "indicates a bug.")
PY
