#!/usr/bin/env python3
"""Preregistered CPU-only measurement: centroid-preserving IVC diameter LENGTH
calibration, fitted leave-one-fold-out (LOFO).

Motivation (the originating diagnostic notebook is not part of the public release): our
predicted IVC diameter carries essentially no signal about the
true (GT) diameter -- near-zero correlation, and a param-MAE *worse* than simply predicting a
training constant. The preregistered lever: rescale the predicted 2-point IVC segment about its
own centroid so its length matches a constant fitted LEAVE-ONE-FOLD-OUT, across a small fixed
panel of ungated and gated variants (no post-hoc additions).

CPU-only, deterministic, no GPU, no training, no submission. Reuses the existing label-geometry
infra (experiments/label_geometry_sweep.py) for GT/OOF loading, the approximate parameter-MAE
formula, and the Nadeau-Bengio corrected CI.

SANITY-BLOCK CAVEAT (found while building this script, see `sanity.reproduction_check` in the
output): the notebook's cited numbers (corr -0.106, slope -0.141, our MAE 12.80) were traced to
`submission/cvfold*/regression_predictions.json` -- the PRE-ViT-B lineage (2026-06-15), reused via
a stale `oof` variable carried over from an earlier, unrelated notebook cell (the panel-bias
check) -- NOT the deployed-route post-drop ViT-B lineage that `experiments/label_geometry_sweep.py`
(and this script) load via `load_oof`/`load_oof_by_fold`. The GT-only numbers (the constant-predictor
MAE, which does not depend on any OOF lineage) reproduce exactly (8.06). The OOF-dependent numbers
do not reproduce their sign/exact value on the deployed lineage, but the qualitative finding --
near-zero correlation, worse than a constant -- holds on BOTH lineages. This script measures and
gates the lever on the DEPLOYED lineage, since that is what any shipped calibration sits on top of.

Writes experiments/results/ivc_length_calibration/result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.label_geometry_sweep import (  # noqa: E402
    OOF_FILES, load_gt, load_oof, load_oof_by_fold, param_abs_err, corrected_ci,
)
from scoring import mre as mre_mod  # noqa: E402

TASK = "IVC"
FOLDS_CSV = os.path.join(ROOT, "data/folds/folds.csv")
N_FOLDS = 5
VERTICAL_UNIT = np.array([0.0, 1.0])

# The exact notebook-cited diagnosis numbers this script's sanity block is checked against.
NOTEBOOK_DIAGNOSIS = {
    "corr": -0.106,
    "slope": -0.141,
    "gt_mean": 26.23,
    "gt_sd": 11.44,
    "pred_mean": 25.89,
    "pred_sd": 15.27,
    "mae_ours": 12.80,
    "global_best_const_mae": 8.06,
    "source": ("IVC length diagnostic, computed over "
               "submission/cvfold*/regression_predictions.json "
               "(the pre-ViT-B lineage, dated 2026-06-15), NOT the deployed post-drop ViT-B "
               "lineage (experiments/label_geometry_sweep.OOF_FILES) this script uses."),
}


# --------------------------------------------------------------------- loading


def load_fold_map(task):
    """{filename: fold_int} from data/folds/folds.csv."""
    out = {}
    with open(FOLDS_CSV) as f:
        for row in csv.DictReader(f):
            if row["task_id"] == task:
                out[row["image_path"].split("/")[-1]] = int(row["fold"])
    return out


def diameter(pts):
    return float(np.linalg.norm(pts[0] - pts[1]))


# ------------------------------------------------------------------- geometry


def rescale(pts, new_len):
    """Centroid-preserving rescale of a 2-point segment to length new_len.

    c = (p0+p1)/2, u = (p0-p1)/||p0-p1||, new points = c +- (new_len/2) * u.
    Falls back to the vertical unit vector if the source segment is (numerically)
    collapsed to a point; the caller is told so it can count the occurrence.
    """
    p0, p1 = pts[0], pts[1]
    d = p0 - p1
    norm = float(np.linalg.norm(d))
    degenerate = norm < 1e-9
    u = VERTICAL_UNIT if degenerate else d / norm
    c = (p0 + p1) / 2.0
    half = new_len / 2.0
    return np.array([c + half * u, c - half * u]), degenerate


# --------------------------------------------------------------- variant panel


def apply_clamp_always(pts, target_len):
    new_pts, degenerate = rescale(pts, target_len)
    return new_pts, True, degenerate  # (new_points, fired, used_degenerate_fallback)


def apply_gated(pts, target_len, lo, hi):
    d = diameter(pts)
    if lo <= d <= hi:
        return pts.copy(), False, False
    new_pts, degenerate = rescale(pts, target_len)
    return new_pts, True, degenerate


def apply_shrink_half(pts, target_len):
    d = diameter(pts)
    new_len = 0.5 * d + 0.5 * target_len
    new_pts, degenerate = rescale(pts, new_len)
    return new_pts, True, degenerate  # ungated: "fires" (changes) every prediction


def make_variant_fn(name):
    if name == "clamp_always_mean":
        return lambda pts, lofo: apply_clamp_always(pts, lofo["mean"])
    if name == "clamp_always_median":
        return lambda pts, lofo: apply_clamp_always(pts, lofo["median"])
    if name == "gated_minmax_median":
        return lambda pts, lofo: apply_gated(pts, lofo["median"], lofo["min"], lofo["max"])
    if name == "gated_p10p90_median":
        return lambda pts, lofo: apply_gated(pts, lofo["median"], lofo["p10"], lofo["p90"])
    if name == "shrink_half_median":
        return lambda pts, lofo: apply_shrink_half(pts, lofo["median"])
    raise ValueError(name)


VARIANTS = [
    "clamp_always_mean", "clamp_always_median",
    "gated_minmax_median", "gated_p10p90_median",
    "shrink_half_median",
]
GATED = {"gated_minmax_median", "gated_p10p90_median"}


# ------------------------------------------------------------------- fitting


def fit_lofo(gt, fold_map, held_out_fold):
    """LOFO diameter stats using ONLY GT rows whose fold != held_out_fold."""
    diam = [diameter(gt[img]) for img, fold in fold_map.items()
            if fold != held_out_fold and img in gt]
    arr = np.asarray(diam, float)
    return {
        "n": len(diam),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# -------------------------------------------------------------------- sanity


def sanity_block(gt, oof):
    keys = sorted(set(gt) & set(oof))
    n = len(keys)
    g = np.asarray([diameter(gt[k]) for k in keys], float)
    p = np.asarray([diameter(oof[k]) for k in keys], float)
    mx, my = float(g.mean()), float(p.mean())
    cov = float(np.mean((g - mx) * (p - my)))
    varg = float(np.mean((g - mx) ** 2))
    varp = float(np.mean((p - my) ** 2))
    slope = cov / varg
    corr = cov / (varg ** 0.5 * varp ** 0.5)
    sdg = float(g.std(ddof=1))
    sdp = float(p.std(ddof=1))
    mae_ours = float(np.mean(np.abs(p - g)))
    global_mean_mae = float(np.mean(np.abs(mx - g)))
    global_median_mae = float(np.mean(np.abs(np.median(g) - g)))
    global_best_const_mae = min(global_mean_mae, global_median_mae)

    deployed = {
        "n": n,
        "corr_pred_vs_gt_diameter": round(corr, 4),
        "slope_pred_on_gt": round(slope, 4),
        "gt_mean": round(mx, 4), "gt_sd": round(sdg, 4),
        "pred_mean": round(my, 4), "pred_sd": round(sdp, 4),
        "dispersion_ratio_pred_over_gt": round(sdp / sdg, 4),
        "mae_ours": round(mae_ours, 4),
        "global_mean_const_mae": round(global_mean_mae, 4),
        "global_median_const_mae": round(global_median_mae, 4),
        "global_best_const_mae": round(global_best_const_mae, 4),
        "worse_than_best_constant_by": round(mae_ours - global_best_const_mae, 4),
    }

    def close(a, b, tol):
        return abs(a - b) <= tol

    check = {
        "global_best_const_mae_matches (tol 0.02)":
            close(global_best_const_mae, NOTEBOOK_DIAGNOSIS["global_best_const_mae"], 0.02),
        "mae_ours_approx_matches (tol 1.0)":
            close(mae_ours, NOTEBOOK_DIAGNOSIS["mae_ours"], 1.0),
        "corr_sign_matches": (corr < 0) == (NOTEBOOK_DIAGNOSIS["corr"] < 0),
        "corr_magnitude_approx_matches (tol 0.05)":
            close(abs(corr), abs(NOTEBOOK_DIAGNOSIS["corr"]), 0.05),
    }
    check["overall"] = (
        "PARTIAL -- GT-only constant-predictor number reproduces exactly; the OOF-dependent "
        "corr/slope/MAE do not reproduce on the deployed lineage (root cause: the notebook's "
        "Test 3 reused a stale pre-ViT-B `oof` variable, see module docstring / "
        "NOTEBOOK_DIAGNOSIS['source']). The qualitative finding -- near-zero correlation, worse "
        "than a constant -- holds under BOTH lineages, so the lever proceeds on the deployed one."
    )

    return {
        "notebook_diagnosis_reference_numbers": NOTEBOOK_DIAGNOSIS,
        "deployed_lineage": deployed,
        "reproduction_check": check,
    }


# ------------------------------------------------------------------- endpoints


def evaluate_variant(name, gt, folds_oof, lofo_by_fold):
    fn = make_variant_fn(name)
    per_fold = []
    per_image = []  # for global improved/worsened counts
    for k, d in enumerate(folds_oof):
        keys = sorted(set(d) & set(gt))
        if not keys:
            continue
        lofo = lofo_by_fold[k]
        mre_raw, mre_var, param_raw, param_var = [], [], [], []
        n_fired, n_degenerate = 0, 0
        for key in keys:
            g, p = gt[key], d[key]
            new_p, fired, degenerate = fn(p, lofo)
            n_fired += int(fired)
            n_degenerate += int(degenerate)
            rm = mre_mod.mean_radial_error(p, g)
            vm = mre_mod.mean_radial_error(new_p, g)
            rp = param_abs_err(TASK, p, g)
            vp = param_abs_err(TASK, new_p, g)
            mre_raw.append(rm); mre_var.append(vm)
            param_raw.append(rp); param_var.append(vp)
            per_image.append({
                "image": key, "fold": k, "fired": fired,
                "mre_delta": vm - rm, "param_delta": vp - rp,
            })
        per_fold.append({
            "fold": k, "n": len(keys), "n_fired": n_fired, "n_degenerate_fallback": n_degenerate,
            "mre_raw": statistics.mean(mre_raw), "mre_variant": statistics.mean(mre_var),
            "param_raw": statistics.mean(param_raw), "param_variant": statistics.mean(param_var),
        })

    dm = [f["mre_variant"] - f["mre_raw"] for f in per_fold]
    dp = [f["param_variant"] - f["param_raw"] for f in per_fold]
    mre_ci = corrected_ci(dm)
    param_ci = corrected_ci(dp)

    n_img_mre_improved = sum(1 for r in per_image if r["mre_delta"] < 0)
    n_img_mre_worsened = sum(1 for r in per_image if r["mre_delta"] > 0)
    n_img_param_improved = sum(1 for r in per_image if r["param_delta"] < 0)
    n_img_param_worsened = sum(1 for r in per_image if r["param_delta"] > 0)

    return {
        "variant": name,
        "gated": name in GATED,
        "per_fold": per_fold,
        "fire_counts_per_fold": {f["fold"]: f["n_fired"] for f in per_fold},
        "degenerate_fallback_counts_per_fold": {f["fold"]: f["n_degenerate_fallback"] for f in per_fold},
        "total_fires": sum(f["n_fired"] for f in per_fold),
        "total_degenerate_fallbacks": sum(f["n_degenerate_fallback"] for f in per_fold),
        "delta_mre_per_fold": dm,
        "delta_mre_mean": statistics.mean(dm),
        "delta_mre_folds_improved": sum(1 for x in dm if x < 0),
        "delta_mre_ci95_corrected": mre_ci,
        "delta_param_per_fold": dp,
        "delta_param_mean": statistics.mean(dp),
        "delta_param_folds_improved": sum(1 for x in dp if x < 0),
        "delta_param_ci95_corrected": param_ci,
        "n_folds": len(per_fold),
        "per_image_mre_improved": n_img_mre_improved,
        "per_image_mre_worsened": n_img_mre_worsened,
        "per_image_param_improved": n_img_param_improved,
        "per_image_param_worsened": n_img_param_worsened,
        "n_images_total": len(per_image),
    }


# -------------------------------------------------------------------- decision


def decide(result):
    """Mechanical preregistered adoption gate, applied identically to every variant."""
    param_improves = result["delta_param_mean"] < 0
    param_folds_ok = result["delta_param_folds_improved"] >= 4
    ci = result["delta_mre_ci95_corrected"]
    mre_not_sig_worse = (ci is None) or (ci[0] <= 0)  # CI does not exclude zero on the harmful side
    shippable = bool(param_improves and param_folds_ok and mre_not_sig_worse)
    return {
        "variant": result["variant"],
        "gated": result["gated"],
        "param_mae_improves": param_improves,
        "param_folds_favorable": result["delta_param_folds_improved"],
        "param_folds_gate_met (>=4/5)": param_folds_ok,
        "mre_ci95_corrected": ci,
        "mre_not_significantly_worse": mre_not_sig_worse,
        "SHIPPABLE_CANDIDATE": shippable,
    }


def summarize_decision(decisions):
    shippable = [d for d in decisions if d["SHIPPABLE_CANDIDATE"]]
    if not shippable:
        return {"verdict": "NO SHIPPABLE CANDIDATE", "shippable_candidates": [], "recommended": None}
    gated_pass = [d for d in shippable if d["gated"]]
    if gated_pass:
        # "most conservative" = fewest fired corrections among the passing gated variants,
        # computed from the per-variant results looked up by name below (see main()).
        recommended_pool = gated_pass
        distinct_note = None
    else:
        recommended_pool = shippable
        distinct_note = ("Only UNGATED variants pass the adoption gate -- reporting distinctly, "
                         "as instructed, rather than silently recommending an ungated rescale.")
    return {
        "verdict": "SHIPPABLE CANDIDATE(S) FOUND",
        "shippable_candidates": [d["variant"] for d in shippable],
        "gated_candidates": [d["variant"] for d in gated_pass],
        "ungated_only": distinct_note is not None,
        "note": distinct_note,
        "recommended_pool": [d["variant"] for d in recommended_pool],
    }


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/ivc_length_calibration/result.json")
    args = ap.parse_args()

    gt = load_gt(TASK)
    oof = load_oof(TASK)
    folds_oof = load_oof_by_fold(TASK)
    fold_map = load_fold_map(TASK)

    print(f"[sanity] n_gt={len(gt)} n_oof={len(oof)} n_common={len(set(gt) & set(oof))}")
    sanity = sanity_block(gt, oof)
    dep = sanity["deployed_lineage"]
    print(f"[sanity] deployed lineage: corr={dep['corr_pred_vs_gt_diameter']:+.4f} "
          f"slope={dep['slope_pred_on_gt']:+.4f} mae_ours={dep['mae_ours']:.2f} "
          f"global_best_const_mae={dep['global_best_const_mae']:.2f} "
          f"(notebook: corr={NOTEBOOK_DIAGNOSIS['corr']:+.3f} "
          f"slope={NOTEBOOK_DIAGNOSIS['slope']:+.3f} mae_ours={NOTEBOOK_DIAGNOSIS['mae_ours']:.2f} "
          f"global_best_const_mae={NOTEBOOK_DIAGNOSIS['global_best_const_mae']:.2f})")
    for k, v in sanity["reproduction_check"].items():
        print(f"    check: {k} = {v}")

    lofo_by_fold = [fit_lofo(gt, fold_map, k) for k in range(N_FOLDS)]
    print("\n[lofo fits]")
    for k, s in enumerate(lofo_by_fold):
        print(f"  fold {k}: n={s['n']:2d} mean={s['mean']:.2f} median={s['median']:.2f} "
              f"min={s['min']:.2f} max={s['max']:.2f} p10={s['p10']:.2f} p90={s['p90']:.2f}")

    variant_results = {}
    decisions = []
    print("\n[variants]")
    for name in VARIANTS:
        res = evaluate_variant(name, gt, folds_oof, lofo_by_fold)
        variant_results[name] = res
        dec = decide(res)
        decisions.append(dec)
        print(f"  {name:24} dMRE={res['delta_mre_mean']:+.4f} ({res['delta_mre_folds_improved']}/5) "
              f"CI={res['delta_mre_ci95_corrected']}  "
              f"dParamMAE={res['delta_param_mean']:+.4f} ({res['delta_param_folds_improved']}/5)  "
              f"fires={res['total_fires']}/{res['n_images_total']}  "
              f"degenerate_fallbacks={res['total_degenerate_fallbacks']}  "
              f"SHIPPABLE={dec['SHIPPABLE_CANDIDATE']}")

    decision_summary = summarize_decision(decisions)
    # Break "most conservative" ties among passing gated variants by fewest total fires.
    if decision_summary.get("gated_candidates"):
        decision_summary["gated_candidates_by_conservatism"] = sorted(
            decision_summary["gated_candidates"],
            key=lambda n: variant_results[n]["total_fires"])
        decision_summary["recommended"] = decision_summary["gated_candidates_by_conservatism"][0]
    elif decision_summary["shippable_candidates"]:
        decision_summary["recommended"] = None  # ungated-only case; report distinctly, do not auto-pick
    print(f"\n[decision] {json.dumps(decision_summary, indent=1)}")

    report = {
        "description": __doc__.strip().split("\n\n")[0],
        "task": TASK,
        "oof_files": OOF_FILES,
        "n_gt_rows": len(gt),
        "sanity": sanity,
        "lofo_fits": {str(k): s for k, s in enumerate(lofo_by_fold)},
        "variants": variant_results,
        "decisions": decisions,
        "decision_summary": decision_summary,
        "param_mae_is_estimate": True,
    }

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
