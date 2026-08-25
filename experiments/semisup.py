"""Pure-numpy/pandas helpers for the SSL mean-teacher trainer (Lever 2B).

No torch import: tested under the project .venv. The torch driver (experiments/ssl_train.py,
baseline venv) converts heatmaps to numpy at the boundary for gating, and uses the schedule /
sampler / leakage helpers here. Pinned hyperparameters live here as the single source of truth.
"""
from __future__ import annotations
import numpy as np

# --- pinned hyperparameters (single source of truth) -------------------------------------------
LEAK_THRESH = 2          # phash Hamming threshold for per-fold leakage exclusion (matches Lever 2A)
EPOCHS = 20              # same budget as the de-leaked baseline checkpoints (clean A/B)
EMA_ALPHA_START = 0.99
EMA_ALPHA_END = 0.999
LAMBDA_MAX = 1.0
SCALE_RANGE = (0.85, 1.15)       # strong-aug geometric scale (Lever-1 OOD axis)
UNLABELED_BATCH = 4
PROM_FLOOR = {"AOP": 0.20, "_default": 0.05}   # per-task prominence floor (AOP guard); tune on probe


def task_floor(floors: dict, task_id: str) -> float:
    """Resolve a per-task prominence floor, falling back to floors['_default']."""
    return float(floors.get(task_id, floors["_default"]))


def prominence_weight(heatmaps, floor: float):
    """Per-landmark consistency weight from a [K,H,W] post-sigmoid teacher heatmap.

    weight_k = (max_k - median_k) if that prominence >= floor else 0.0, clipped to [0,1].
    Mirrors decode's prominence guard; below-floor landmarks contribute nothing to consistency."""
    hm = np.asarray(heatmaps, dtype=np.float64)
    K = hm.shape[0]
    flat = hm.reshape(K, -1)
    prom = flat.max(axis=1) - np.median(flat, axis=1)
    w = np.where(prom >= floor, prom, 0.0)
    return np.clip(w, 0.0, 1.0)


def lambda_ramp(step: int, ramp_steps: int, lam_max: float = LAMBDA_MAX) -> float:
    """Sigmoid consistency-weight ramp: ~0 at step 0, ~lam_max/2 at the ramp midpoint, clamped
    to lam_max for step >= ramp_steps. Mean-teacher style (keeps SSL from dominating early)."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return float(lam_max)
    # logistic centred on ramp_steps/2; k chosen so the endpoints are ~clamped (>=~5 sigma).
    x = (step - ramp_steps / 2.0) / (ramp_steps / 10.0)
    return float(lam_max / (1.0 + np.exp(-x)))


def ema_alpha(step: int, ramp_steps: int,
              start: float = EMA_ALPHA_START, end: float = EMA_ALPHA_END) -> float:
    """Linear EMA decay ramp start->end over ramp_steps, then held at end. Slower teacher
    smoothing early (when the student moves fast), tighter later."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return float(end)
    return float(start + (end - start) * (step / ramp_steps))


import pandas as pd
from experiments.ssl_pool import exclude_near


def normalize_labeled_path(p: str) -> str:
    """_labeled_phash.csv uses 'data/images/<task>/<file>'; folds.csv uses '<task>/<file>'.
    Strip the 'data/images/' prefix so the two join on the same key."""
    prefix = "data/images/"
    return p[len(prefix):] if p.startswith(prefix) else p


def fold_val_phashes(folds: pd.DataFrame, labeled: pd.DataFrame, fold: int) -> list[int]:
    """Phashes of the labeled images held out in fold K (the val fold). These are the reference
    set the pool must NOT leak against when training fold K."""
    val_paths = set(folds.loc[folds["fold"] == fold, "image_path"].astype(str))
    lab = labeled.copy()
    lab["_key"] = lab["image_path"].astype(str).map(normalize_labeled_path)
    return lab.loc[lab["_key"].isin(val_paths), "phash"].astype("int64").tolist()


def count_near(pool_phashes, ref_phashes, thresh: int = LEAK_THRESH) -> int:
    """Number of pool phashes within `thresh` Hamming of ANY ref phash (the leakage count).
    Computed as len(pool) - len(kept) using the exact exclude_near (no bucketing)."""
    pool = [int(x) for x in pool_phashes]
    refs = [int(x) for x in ref_phashes]
    kept = exclude_near(pool, refs, thresh=thresh)
    return len(pool) - len(kept)


def filter_pool_for_fold(pool: pd.DataFrame, folds: pd.DataFrame, labeled: pd.DataFrame,
                         fold: int, thresh: int = LEAK_THRESH) -> pd.DataFrame:
    """Drop pool rows whose phash is within `thresh` Hamming of ANY fold-K-val labeled phash.
    Reuses the exact experiments.ssl_pool.exclude_near (NOT the old bucketed assumption)."""
    refs = fold_val_phashes(folds, labeled, fold)
    pool = pool.reset_index(drop=True)
    keep_idx = exclude_near(pool["phash"].astype("int64").tolist(), refs, thresh=thresh)
    return pool.iloc[keep_idx].reset_index(drop=True)


import random


def indices_by_task(manifest: pd.DataFrame) -> dict:
    """Map task_id -> list of integer row indices into `manifest` (positional, 0-based)."""
    out: dict = {}
    for i, tid in enumerate(manifest["task_id"].tolist()):
        out.setdefault(tid, []).append(i)
    return out


def task_balanced_batches(manifest: pd.DataFrame, batch_size: int, steps: int):
    """Produce `steps` task-uniform batches of positional indices into `manifest`.

    Each batch: pick a task uniformly at random (so A4C's volume can't swamp AOP/cardiac), then
    fill `batch_size` indices from that task, reshuffling + cycling that task's pool whenever it is
    exhausted (a task whose pool is smaller than batch_size repeats indices to fill the batch —
    handled by the while-loop so the batch is ALWAYS exactly batch_size). Mirrors
    KeypointUniformSampler's uniform-over-tasks logic. Tasks absent from the manifest (e.g.
    fetal_femur) are never chosen. Determinism is controlled by the caller seeding `random`."""
    by_task = {t: ix[:] for t, ix in indices_by_task(manifest).items()}
    tasks = list(by_task)
    for t in tasks:
        random.shuffle(by_task[t])
    cursors = {t: 0 for t in tasks}
    batches = []
    for _ in range(steps):
        t = random.choice(tasks)
        ix = by_task[t]
        batch: list = []
        while len(batch) < batch_size:
            if cursors[t] >= len(ix):           # exhausted -> reshuffle + restart this task's pool
                random.shuffle(ix)
                cursors[t] = 0
            take = min(batch_size - len(batch), len(ix) - cursors[t])
            batch.extend(ix[cursors[t]:cursors[t] + take])
            cursors[t] += take
        batches.append(batch)
    return batches
