"""Shared rendering for the experiment notebooks: per-task metric tables (pure, pandas) and
landmark overlays (matplotlib, imported lazily so the table path stays project-.venv-testable)."""
from __future__ import annotations
import pandas as pd


def metric_table(summary: dict, baseline: dict | None = None) -> pd.DataFrame:
    """summary/baseline are cv_summary-style dicts: per_task[t]['mre'/'param_mae']['mean']."""
    pt = summary["per_task"]
    rows = []
    for t in sorted(pt):
        r = {"task": t, "MRE": pt[t]["mre"]["mean"], "paramMAE": pt[t]["param_mae"]["mean"]}
        if baseline is not None and t in baseline["per_task"]:
            b = baseline["per_task"][t]
            r["dMRE"] = r["MRE"] - b["mre"]["mean"]
            r["dparamMAE"] = r["paramMAE"] - b["param_mae"]["mean"]
        rows.append(r)
    return pd.DataFrame(rows)


def overlay_landmarks(image, pred_pts, gt_pts=None, ax=None, title=None):
    """image: HxW(x3) array; pred_pts/gt_pts: [(x,y), ...] in pixels. Returns the matplotlib ax."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    if gt_pts:
        gx, gy = zip(*gt_pts)
        ax.scatter(gx, gy, c="lime", s=18, marker="o", label="GT")
    if pred_pts:
        px, py = zip(*pred_pts)
        ax.scatter(px, py, c="red", s=18, marker="x", label="pred")
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off"); ax.legend(loc="lower right", fontsize=6)
    return ax
