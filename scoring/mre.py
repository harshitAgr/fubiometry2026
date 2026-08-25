"""Mean Radial Error — reconstructed to match baseline/evaluate.py semantics.

NOTE (spec): the official scorer is not public; confirm/adjust against the
per-task feedback in the M7 submission panel. spacing=(sx,sy) mm/px converts to mm.
"""
from __future__ import annotations
import numpy as np


def mean_radial_error(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"pred/gt must be equal (K,2); got {pred.shape} vs {gt.shape}")
    diff = pred - gt
    if spacing is not None:
        diff = diff * np.asarray(spacing, float)[None, :]
    return float(np.mean(np.linalg.norm(diff, axis=1)))
