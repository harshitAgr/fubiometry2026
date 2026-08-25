#!/usr/bin/env python3
"""Rebuild the canonical leak-free CV fold file (data/folds/folds.csv) — CORE data only.

WHY THIS EXISTS (2026-07-26): the fold-build snippet previously documented in our notes used a
bare ``glob('data/csv/*.csv')``. That was correct until the external-data lever landed
``FA_train_ext.csv`` (811), ``FA_train_ext_reor.csv`` (811) and ``fetal_femur_train_ext.csv``
(355) into the SAME directory. The bare glob now sweeps in those 1977 ``ext_``-prefixed rows and
**reshuffles the FA and fetal_femur fold assignments**, so it no longer reproduces the adopted
folds.csv — a silent reproducibility break, since data/folds/ is gitignored.

The external rows are deliberately kept on disk (the external-femur re-test is an approved,
not-yet-run lever and consumes them via data/folds/folds_femur_only_corr.csv with fold == -2
train-only). They must simply never enter the CORE fold build.

Usage:
    uv run python experiments/make_folds_core.py            # verify only (no write)
    uv run python experiments/make_folds_core.py --write    # rebuild data/folds/folds.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.folds import aop_adjacent_leak, make_folds  # noqa: E402

EXPECT_ROWS = 6743
EXPECT_PER_TASK = {
    "A4C": 108, "AOP": 4000, "FA": 500, "FUGC": 260, "HC": 999,
    "IVC": 38, "PLAX": 87, "PSAX": 49, "fetal_femur": 702,
}


def core_csvs() -> list[str]:
    """Task CSVs that make up the CORE (challenge-released) training set.

    Excludes any ``*_ext*.csv`` (external-data lever artifacts).
    """
    return sorted(
        p for p in glob.glob(os.path.join(PROJ, "data", "csv", "*.csv"))
        if "_ext" not in os.path.basename(p)
    )


def dropped_femur_rows() -> pd.DataFrame:
    """The 25 organizer-flagged mirrored femur rows, recovered from the RAW femur CSV.

    ``prepare_data.py`` bakes these out of ``data/csv/fetal_femur_train.csv`` (727 -> 702), so they
    must be read back from the raw drive CSV to reconstruct the pre-drop fold structure.
    """
    sys.path.insert(0, os.path.join(PROJ, "scripts"))
    from prepare_data import DROP_IMAGES  # noqa: PLC0415 — single source of truth for the 25

    drop = DROP_IMAGES["fetal_femur"]
    raw = pd.read_csv(os.path.join(PROJ, "data", "drive_raw", "fetal_femur", "labeled",
                                   "Reg-Two_3.fetal_femur.csv"))
    hit = raw[raw.image_path.map(lambda p: os.path.basename(str(p)) in drop)].copy()
    assert len(hit) == len(drop) == 25, f"recovered {len(hit)} of {len(drop)} dropped femur rows"
    hit["image_path"] = hit.image_path.map(lambda p: "fetal_femur/" + os.path.basename(str(p)))
    hit["task_id"] = "fetal_femur"
    return hit[["image_path", "task_id"]]


def build() -> pd.DataFrame:
    """Rebuild folds.csv.

    CRITICAL RECIPE DETAIL (verified 2026-07-26): folds are built on the **PRE-drop** row set and
    the 25 flagged femur rows are removed AFTERWARDS — NOT by running make_folds on the reduced
    set. This is deliberate: it keeps every surviving row's fold assignment identical to the
    pre-drop folds, which is what preserves paired comparability with the pre-drop ViT-S/ViT-B
    baselines. Running make_folds on the reduced 702-row femur set instead reshuffles femur
    (only 167/702 assignments survive) and silently breaks that pairing.
    """
    paths = core_csvs()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)[["image_path", "task_id"]]
    n_ext = df.image_path.str.contains("/ext_").sum()
    assert n_ext == 0, f"external rows leaked into the core fold build: {n_ext}"

    dropped = dropped_femur_rows()
    pre = pd.concat([df, dropped], ignore_index=True)
    assert len(pre) == EXPECT_ROWS + 25, f"pre-drop row count {len(pre)} != {EXPECT_ROWS + 25}"

    folds = make_folds(pre, k=5, guard=2, seed=0)
    return folds[~folds.image_path.isin(set(dropped.image_path))].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write data/folds/folds.csv")
    args = ap.parse_args()

    print("core CSVs:")
    for p in core_csvs():
        print(f"  {os.path.basename(p)}")

    out = build()
    leak = aop_adjacent_leak(out)
    print(f"\nrows={len(out)} (expect {EXPECT_ROWS})   AOP adjacency leak={leak}")
    assert leak == 0.0, f"AOP adjacency leak != 0: {leak}"
    assert len(out) == EXPECT_ROWS, f"row count {len(out)} != {EXPECT_ROWS}"
    got = out.groupby("task_id").size().to_dict()
    assert got == EXPECT_PER_TASK, f"per-task counts differ: {got}"

    target = os.path.join(PROJ, "data", "folds", "folds.csv")
    if os.path.exists(target):
        cur = pd.read_csv(target)
        m = cur.merge(out, on="image_path", suffixes=("_cur", "_new"))
        if len(m) != len(cur):
            print(f"MISMATCH: only {len(m)}/{len(cur)} image_paths in common")
            return 1
        same = int((m.fold_cur == m.fold_new).sum())
        print(f"reproduces existing folds.csv: {same}/{len(m)} identical fold assignments")
        if same != len(m):
            print("MISMATCH — the existing folds.csv was NOT built from core CSVs this way.")
            return 1
    else:
        print("no existing folds.csv to compare against")

    if args.write:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        out.to_csv(target, index=False)
        print(f"wrote {target}")
    else:
        print("(verify-only; pass --write to rebuild)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
