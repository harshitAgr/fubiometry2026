"""Group-aware K-fold. Non-AOP tasks: GroupKFold by group_key, round-robin within each task
so every task spans all folds. AOP: contiguous frame-index blocks with a guard band dropped at
block boundaries (fold = -1), which provably removes the +/-1-frame adjacency leak."""
from __future__ import annotations
import hashlib
import warnings
import numpy as np
import pandas as pd
from experiments.groups import group_key, aop_frame_index


def _stable_seed(seed: int, salt: str = "") -> int:
    return int(hashlib.md5(f"{seed}-{salt}".encode()).hexdigest(), 16) % (2**32)


def make_folds(df: pd.DataFrame, k: int = 5, guard: int = 2, seed: int = 0) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1
    df["group"] = ""

    aop_mask = df["task_id"] == "AOP"
    if aop_mask.any():
        aop = df[aop_mask].copy()
        aop["_fi"] = aop["image_path"].map(aop_frame_index)
        ordered = aop.sort_values("_fi").index.to_numpy()
        n = len(ordered)
        bounds = np.linspace(0, n, k + 1).astype(int)
        for f in range(k):
            block = ordered[bounds[f]:bounds[f + 1]]
            df.loc[block, "fold"] = f
            df.loc[block, "group"] = f"AOP:block{f}"
        for b in bounds[1:-1]:
            df.loc[ordered[max(0, b - guard):b + guard], "fold"] = -1

    for tid, sub in df[~aop_mask].groupby("task_id"):
        groups: dict[str, list[int]] = {}
        for ridx, row in sub.iterrows():
            groups.setdefault(group_key(tid, row["image_path"]), []).append(ridx)
        gkeys = sorted(groups)
        if len(gkeys) < k:
            warnings.warn(
                f"task {tid}: {len(gkeys)} groups < k={k} folds; it will use only "
                f"{len(gkeys)} folds (no group is ever split to fill a fold)."
            )
        rng = np.random.default_rng(_stable_seed(seed, tid))
        rng.shuffle(gkeys)
        for i, gk in enumerate(gkeys):
            for ridx in groups[gk]:
                df.loc[ridx, "fold"] = i % k
                df.loc[ridx, "group"] = gk
    return df


def aop_adjacent_leak(df: pd.DataFrame) -> float:
    """Fraction of kept AOP frames whose +1-frame neighbor sits in a DIFFERENT fold."""
    aop = df[(df["task_id"] == "AOP") & (df["fold"] >= 0)].copy()
    if aop.empty:
        return 0.0
    fold_of = {aop_frame_index(p): f for p, f in zip(aop["image_path"], aop["fold"])}
    bad = sum(1 for fi, f in fold_of.items() if (fi + 1) in fold_of and fold_of[fi + 1] != f)
    return bad / len(fold_of)
