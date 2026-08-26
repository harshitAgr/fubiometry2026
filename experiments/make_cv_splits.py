#!/usr/bin/env python3
"""Generate the per-fold CV val/GT split CSVs WITHOUT training.

``run_config.py`` writes ``data/_cvfold{K}_val.csv`` + ``data/_cvfold{K}_gt.csv`` only as a side
effect of a training run (lines 544-566). That makes it impossible to re-score an existing
checkpoint against a fold it didn't just train — which is exactly what's needed to restate a
baseline after the fold data changes. This reproduces those two files verbatim from folds.csv.

Logic mirrors run_config.py exactly:
    val_paths = sorted(set(folds[folds.fold == K].image_path))
    gt        = concat(data/csv/*.csv) filtered to val_paths

(The GT concat intentionally globs ALL task CSVs including ``*_ext*.csv``, matching run_config;
external rows can never match because they are absent from folds.csv.)

⚠️ These are SHARED SCRATCH paths — a training run for a different fold will overwrite them. Never
run this while a fold-based training job is active.

Usage:
    uv run python experiments/make_cv_splits.py --folds 1 2 3 4
    uv run python experiments/make_cv_splits.py --folds 0 --verify-only
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_split(fold: int, folds_csv: str, verify_only: bool = False) -> tuple[int, int]:
    folds = pd.read_csv(folds_csv)
    val_paths = sorted(set(folds[folds.fold == fold]["image_path"]))
    gt = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(PROJ, "data", "csv", "*.csv"))],
                   ignore_index=True)
    gt = gt[gt.image_path.isin(set(val_paths))]
    assert len(gt) == len(val_paths), f"fold {fold}: GT rows {len(gt)} != val paths {len(val_paths)}"

    split_p = os.path.join(PROJ, "data", f"_cvfold{fold}_val.csv")
    gt_p = os.path.join(PROJ, "data", f"_cvfold{fold}_gt.csv")
    split_df = pd.DataFrame({"image_path": val_paths})

    if verify_only:
        if os.path.exists(split_p):
            on_disk = pd.read_csv(split_p)
            match = sorted(on_disk.image_path.tolist()) == val_paths
            print(f"  fold {fold}: on-disk split {'MATCHES' if match else 'DIFFERS'} "
                  f"({len(on_disk)} rows vs {len(val_paths)} generated)")
        else:
            print(f"  fold {fold}: no on-disk split to compare")
        return len(val_paths), len(gt)

    split_df.to_csv(split_p, index=False)
    gt.to_csv(gt_p, index=False)
    print(f"  fold {fold}: wrote {len(val_paths)} val paths -> {os.path.basename(split_p)}, "
          f"{len(gt)} GT rows -> {os.path.basename(gt_p)}")
    return len(val_paths), len(gt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--folds-csv", default=os.path.join(PROJ, "data", "folds", "folds.csv"))
    ap.add_argument("--verify-only", action="store_true",
                    help="compare against the existing on-disk split instead of writing")
    args = ap.parse_args()
    print(f"folds-csv: {args.folds_csv}")
    for k in args.folds:
        make_split(k, args.folds_csv, args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
