"""Stack per-fold CV result JSONs into a de-leaked baseline summary.

Reads experiments/results/cvfold*.json (each has scalar per-task mre/param_mae for that
fold), supplies per-task group counts from data/folds/folds.csv, and runs
experiments.aggregate.aggregate to produce per-task mean +/- corrected-resampled-t CI +
pooled cardiac. This is the reusable form of the F6 aggregation.

Run: uv run python experiments/aggregate_cv.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from experiments.aggregate import aggregate  # noqa: E402

N_TRAIN, N_TEST = 5400, 1350  # ~4/5 vs 1/5 of the ~6750 kept labeled images


def load_per_fold(results_glob: str, folds_csv: str):
    folds = [json.load(open(p)) for p in sorted(glob.glob(results_glob))]
    if not folds:
        raise FileNotFoundError(f"no fold results match {results_glob}")
    tasks = list(folds[0]["per_task"].keys())
    gc = pd.read_csv(folds_csv)
    gc = gc[gc.fold >= 0].groupby("task_id")["group"].nunique().to_dict()
    return {
        t: {"mre": [f["per_task"][t]["mre"] for f in folds],
            "param_mae": [f["per_task"][t]["param_mae"] for f in folds],
            "groups": gc.get(t)}
        for t in tasks
    }, len(folds)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob",
                    default=os.path.join(PROJ, "experiments", "results", "cvfold*.json"))
    ap.add_argument("--folds-csv", default=os.path.join(PROJ, "data", "folds", "folds.csv"))
    ap.add_argument("--out", default=os.path.join(PROJ, "experiments", "results", "cv_summary.json"))
    args = ap.parse_args()

    per_fold, n = load_per_fold(args.results_glob, args.folds_csv)
    res = aggregate(per_fold, n_train=N_TRAIN, n_test=N_TEST)
    print(f"=== CV ({n} folds) — per-task MRE mean [95% CI], groups ===")
    for t in sorted(res["per_task"]):
        m = res["per_task"][t]["mre"]
        p = res["per_task"][t]["param_mae"]
        print(f"  {t:12s} MRE {m['mean']:6.2f} [{m['ci'][0]:5.1f},{m['ci'][1]:5.1f}]  "
              f"paramMAE {p['mean']:7.2f}  groups={res['per_task'][t]['groups']}")
    pc = res["pooled"]["cardiac"]["mre"]
    print(f"  {'POOLED CARDIAC':12s} MRE {pc['mean']:6.2f} [{pc['ci'][0]:.1f},{pc['ci'][1]:.1f}]")
    print(f"  task-mean MRE: {np.mean([res['per_task'][t]['mre']['mean'] for t in res['per_task']]):.2f}px")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
