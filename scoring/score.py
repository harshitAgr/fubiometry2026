"""Combined local scorer: per-task MRE + derived-parameter MAE.

Ground-truth coordinates in the FU_Biometry CSVs are in ORIGINAL-IMAGE PIXELS — the
baseline's dataset.py divides them by image size to normalize, which proves the CSV
values are pixels. That is the SAME space as a submission's `predicted_points_pixels`,
so MRE and parameters are computed directly in pixel space (no normalize/denormalize).

LIMITATION (spec): the official per-parameter IQR/tolerance normalization and the
ChallengeR domain weighting are NOT public. This computes per-task raw MRE and raw
param-MAE (an ESTIMATE only); the result carries `param_mae_is_estimate=True` and `notes`.
Callers must treat `avg_param_mae` as an estimate to calibrate against official feedback,
NOT the official metric. Missing/invalid cases are counted (`n_missing`, top-level
`total_missing`) but are NOT folded into the averages — a submission with total_missing>0
is OPTIMISTICALLY scored, so gate on total_missing before trusting the numbers.
"""
from __future__ import annotations
import numpy as np
from scoring import schema, mre as mre_mod, derive, gt as gt_mod

_NOTES = ("param_mae is a raw estimate (official IQR/tolerance normalization + ChallengeR "
          "aggregation are not public); missing cases are NOT penalized in the averages — "
          "check total_missing before trusting the numbers.")


def score_submission(pred_path, gt_csv_path) -> dict:
    records = schema.load_submission(pred_path)
    schema.validate_submission(records)
    gt = gt_mod.load_gt(gt_csv_path)  # pixel coords, {(image_path, task_id): (K,2)}
    by_key = {(r["image_path"], r["task_id"]): r for r in records}

    per_task: dict[str, dict] = {}
    for (img, tid), gt_px in gt.items():
        d = per_task.setdefault(tid, {"mre": [], "param_ae": [], "n_missing": 0,
                                      "n_param_errors": 0})
        rec = by_key.get((img, tid))
        if rec is None:
            d["n_missing"] += 1
            continue
        gt_px = np.asarray(gt_px, float)
        pred_px = schema.points_to_array(rec["predicted_points_pixels"])
        if pred_px.shape != gt_px.shape:
            d["n_missing"] += 1
            continue
        d["mre"].append(mre_mod.mean_radial_error(pred_px, gt_px))
        # Parameter derivation can fail on degenerate predictions (e.g. coincident
        # points -> undefined angle). Never let one bad case crash the whole run.
        try:
            gt_params = derive.derive_parameters(tid, gt_px)
            pred_params = derive.derive_parameters(tid, pred_px)
            for name, gval in gt_params.items():
                d["param_ae"].append(abs(pred_params[name] - gval))
        except (ValueError, KeyError, IndexError):
            d["n_param_errors"] += 1

    out: dict = {"per_task": {}}
    mre_means, pmae_means = [], []
    total_missing = 0
    total_param_errors = 0
    for tid, d in per_task.items():
        m = float(np.mean(d["mre"])) if d["mre"] else float("nan")
        p = float(np.mean(d["param_ae"])) if d["param_ae"] else float("nan")
        out["per_task"][tid] = {"mre": m, "param_mae": p,
                                "n_missing": d["n_missing"],
                                "n_param_errors": d["n_param_errors"]}
        total_missing += d["n_missing"]
        total_param_errors += d["n_param_errors"]
        if d["mre"]:
            mre_means.append(m)
        if d["param_ae"]:
            pmae_means.append(p)
    out["avg_mre"] = float(np.mean(mre_means)) if mre_means else float("nan")
    out["avg_param_mae"] = float(np.mean(pmae_means)) if pmae_means else float("nan")
    out["total_missing"] = total_missing
    out["total_param_errors"] = total_param_errors
    out["param_mae_is_estimate"] = True
    out["notes"] = _NOTES
    return out
