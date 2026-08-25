#!/usr/bin/env bash
# Reproduce the label-geometry invariant sweep (2026-08-07).
#
# Detects which exact geometric invariants the 9 tasks' TRAINING LABELS satisfy,
# measures how far our 5-fold out-of-fold predictions sit off them, and reports the
# paired per-fold MRE / approximate-parameter effect of projecting onto them.
#
# CPU only, seconds to run. Reads no validation ground truth. Trains nothing,
# scores no submission, writes no model.
#
# Result: experiments/results/label_geometry/sweep.json
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python experiments/label_geometry_sweep.py "$@"
uv run python - <<'PY'
import json
d = json.load(open('experiments/results/label_geometry/sweep.json'))
print(f"\n{'task':4} {'projection':18} {'dMRE':>9} {'folds':>6} {'corrected CI95':>22} {'dParam':>9}")
for t, r in d['tasks'].items():
    for res in r.get('projection_results', []):
        ci = res['delta_mre_ci95_corrected']
        print(f"{t:4} {res['projection']:18} {res['delta_mre_mean']:+9.4f} "
              f"{res['delta_mre_folds_improved']}/{res['n_folds']:<4} "
              f"[{ci[0]:+7.4f},{ci[1]:+7.4f}] {res['delta_param_mean']:+9.4f}")
PY
