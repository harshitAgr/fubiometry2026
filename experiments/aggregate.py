"""Aggregate per-fold metrics into per-task mean +/- Nadeau-Bengio corrected-resampled-t CIs,
plus a pooled-cardiac selection unit. The correction accounts for the dependence between CV
folds (train sets overlap), which a naive SE ignores and which matters at tiny n."""
from __future__ import annotations
import numpy as np
from scipy import stats

CARDIAC = ("A4C", "PLAX", "PSAX", "IVC")


def corrected_resampled_t_ci(scores, n_train: int, n_test: int, alpha: float = 0.05):
    s = np.asarray(scores, float)
    j = len(s)
    mean = float(s.mean())
    if j < 2:
        return mean, mean, mean
    var = s.var(ddof=1)
    corrected_var = var * (1.0 / j + n_test / max(n_train, 1))  # Nadeau & Bengio (2003)
    half = float(stats.t.ppf(1 - alpha / 2, j - 1) * np.sqrt(corrected_var))
    return mean, mean - half, mean + half


def _stat(values, n_train, n_test):
    m, lo, hi = corrected_resampled_t_ci(values, n_train, n_test)
    return {"mean": m, "ci": [lo, hi]}


def aggregate(per_fold: dict, n_train: int, n_test: int) -> dict:
    per_task = {}
    for tid, d in per_fold.items():
        per_task[tid] = {
            "mre": _stat(d["mre"], n_train, n_test),
            "param_mae": _stat(d["param_mae"], n_train, n_test),
            "groups": d.get("groups"),
        }
    pooled = {}
    card = [t for t in CARDIAC if t in per_fold]
    if card:
        pooled["cardiac"] = {
            "mre": _stat(np.concatenate([per_fold[t]["mre"] for t in card]), n_train, n_test),
            "param_mae": _stat(np.concatenate([per_fold[t]["param_mae"] for t in card]), n_train, n_test),
        }
    return {"per_task": per_task, "pooled": pooled}
