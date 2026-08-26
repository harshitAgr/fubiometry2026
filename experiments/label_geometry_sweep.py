#!/usr/bin/env python3
"""Sweep all 9 tasks for EXACT label-geometry invariants, then measure what
projecting our predictions onto them buys.

Motivation (2026-08-07, FA diagnosis): the FA label turned out to be an exactly
axis-aligned concentric ellipse in 500/500 rows, i.e. its 8 coordinates encode only
4 free parameters. The valid-label set is therefore a linear subspace of R^8 and
orthogonal projection onto it is NON-EXPANSIVE -- it cannot increase the squared
distance to any ground truth lying in the subspace. That gave a free -0.87 px of FA
MRE. Nobody has ever checked the other eight tasks.

This script:
  1. DETECTS invariants from the training labels themselves (does not assume them):
     per-pair axis alignment, pair-pair perpendicularity / parallelism, concentricity,
     shared endpoints, and the ordering convention.
  2. MEASURES how far our 5-fold out-of-fold predictions sit off each invariant.
  3. PROJECTS onto the invariant manifold where one exists and reports the paired
     per-fold change in MRE and in the (approximate) derived parameter MAE.

CPU only. Reads no validation ground truth. Trains nothing, submits nothing.

Writes experiments/results/label_geometry/sweep.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import statistics
from collections import defaultdict

import numpy as np

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scoring import derive, mre as mre_mod  # noqa: E402
from experiments.geometry_project import (  # noqa: E402
    project_fa as _canon_fa, project_hc as _canon_hc)


def _flatten(p):
    return [float(v) for xy in np.asarray(p, float) for v in xy]

TASKS = ["A4C", "AOP", "FA", "fetal_femur", "FUGC", "HC", "IVC", "PLAX", "PSAX"]

# The adopted post-drop ViT-B lineage, one file per CV fold (fold k holds out its own
# validation split, so the union is a genuine out-of-fold set over the training data).
OOF_FILES = [
    "submission/vitb_postdrop_fold0/regression_predictions.json",
    "submission/geo_cosine40_vitb_postdrop/cvfold1/regression_predictions.json",
    "submission/geo_cosine40_vitb_postdrop/cvfold2/regression_predictions.json",
    "submission/geo_cosine40_vitb_postdrop/cvfold3/regression_predictions.json",
    "submission/geo_cosine40_vitb_postdrop/cvfold4/regression_predictions.json",
]

# GT rounding budget: labels are stored as integers, so an invariant that holds
# exactly in the underlying annotation shows up as <= half a pixel of slack per
# coordinate (hence <= sqrt(2)/2 for a midpoint).
ROUND_TOL = math.sqrt(2) / 2 + 1e-9
ANGLE_TOL_DEG = 1.0


# ---------------------------------------------------------------- data loading


def load_gt(task):
    """{filename: (K,2) array} from the CORE task csv (never the _ext variants)."""
    path = os.path.join(ROOT, "data/csv", f"{task}_train.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            k = int(row["num_classes"])
            pts = [ast.literal_eval(row[f"point_{i}_xy"]) for i in range(1, k + 1)]
            out[row["image_path"].split("/")[-1]] = np.asarray(pts, float)
    return out


def load_oof(task):
    out = {}
    for rel in OOF_FILES:
        p = os.path.join(ROOT, rel)
        for r in json.load(open(p)):
            if r["task_id"] != task:
                continue
            v = r["predicted_points_pixels"]
            out[r["image_path"].split("/")[-1]] = np.asarray(v, float).reshape(-1, 2)
    return out


def load_oof_by_fold(task):
    """Same as load_oof but keyed by fold index, for paired per-fold statistics."""
    per_fold = []
    for rel in OOF_FILES:
        d = {}
        for r in json.load(open(os.path.join(ROOT, rel))):
            if r["task_id"] == task:
                v = r["predicted_points_pixels"]
                d[r["image_path"].split("/")[-1]] = np.asarray(v, float).reshape(-1, 2)
        per_fold.append(d)
    return per_fold


# ------------------------------------------------------------ invariant detect


def pair_vectors(pts):
    """Consecutive landmark pairs (0,1), (2,3), ... -> (vector, midpoint)."""
    n = len(pts) // 2
    return [(pts[2 * i + 1] - pts[2 * i], (pts[2 * i] + pts[2 * i + 1]) / 2) for i in range(n)]


def detect(task, gt):
    """Which exact geometric invariants hold across EVERY training label?"""
    rows = list(gt.values())
    n_pairs = len(rows[0]) // 2
    rep = {"n": len(rows), "n_landmarks": int(len(rows[0])), "n_pairs": n_pairs}

    vertical, horizontal, above = [], [], []
    for i in range(n_pairs):
        v = [r[2 * i + 1] - r[2 * i] for r in rows]
        vertical.append(sum(1 for d in v if abs(d[0]) <= 0.5) / len(v))
        horizontal.append(sum(1 for d in v if abs(d[1]) <= 0.5) / len(v))
        above.append(sum(1 for d in v if d[1] > 0) / len(v))
    rep["pair_vertical_frac"] = [round(x, 4) for x in vertical]
    rep["pair_horizontal_frac"] = [round(x, 4) for x in horizontal]
    rep["pair_first_is_above_frac"] = [round(x, 4) for x in above]

    # pair-pair relationships (only meaningful for a handful of pairs)
    perp, para, conc = {}, {}, {}
    if n_pairs >= 2:
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                angs, gaps = [], []
                for r in rows:
                    u = r[2 * i + 1] - r[2 * i]
                    w = r[2 * j + 1] - r[2 * j]
                    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
                    if nu < 1e-9 or nw < 1e-9:
                        continue
                    c = float(np.clip(np.dot(u, w) / (nu * nw), -1, 1))
                    angs.append(math.degrees(math.acos(abs(c))))  # 0=parallel, 90=perp
                    ci = (r[2 * i] + r[2 * i + 1]) / 2
                    cj = (r[2 * j] + r[2 * j + 1]) / 2
                    gaps.append(float(np.linalg.norm(ci - cj)))
                if not angs:
                    continue
                key = f"{i}-{j}"
                if all(a >= 90 - ANGLE_TOL_DEG for a in angs):
                    perp[key] = round(max(90 - a for a in angs), 4)
                if all(a <= ANGLE_TOL_DEG for a in angs):
                    para[key] = round(max(angs), 4)
                if all(g <= ROUND_TOL for g in gaps):
                    conc[key] = round(max(gaps), 4)
    rep["perpendicular_pairs"] = perp          # value = worst deviation from 90 deg
    rep["parallel_pairs"] = para
    rep["concentric_pairs"] = conc             # value = worst centre gap (px)

    shared = []
    for i in range(len(rows[0])):
        for j in range(i + 1, len(rows[0])):
            if all(np.linalg.norm(r[i] - r[j]) <= ROUND_TOL for r in rows):
                shared.append(f"{i}={j}")
    rep["coincident_landmarks"] = shared
    return rep


# ----------------------------------------------------------------- projections


def project_fa(p):
    """FA: {x0=x1, y2=y3, the two diameters concentric}. Linear -> non-expansive.

    Closed form: the centre is the mean of all four points, and each pair keeps its
    own along-axis extent.
    """
    c = p.mean(axis=0)
    t = (p[0, 1] - p[1, 1]) / 2.0
    w = (p[2, 0] - p[3, 0]) / 2.0
    return np.array([[c[0], c[1] + t], [c[0], c[1] - t],
                     [c[0] + w, c[1]], [c[0] - w, c[1]]])


def project_concentric(p):
    """Both pairs share a centre; orientations and lengths untouched. Linear."""
    c = p.mean(axis=0)
    u = (p[0] - p[1]) / 2.0
    v = (p[2] - p[3]) / 2.0
    return np.array([c + u, c - u, c + v, c - v])


def project_orthoconcentric(p):
    """Concentric AND the two diameters exactly perpendicular (the HC label form).

    Centre is separable and lands on the 4-point mean. With A, B the half-vectors of
    the two pairs, minimising |U-A|^2 + |V-B|^2 over U perp V is solved by
    phi = arg(|A|^2 e^{2i alpha} - |B|^2 e^{2i beta}) / 2, then projecting A and B onto
    the resulting orthogonal frame. Non-linear (the manifold is a variety, not a
    subspace), so the non-expansiveness guarantee does NOT carry over -- it is
    measured empirically below.
    """
    c = p.mean(axis=0)
    A = (p[0] - p[1]) / 2.0
    B = (p[2] - p[3]) / 2.0
    na, nb = float(np.linalg.norm(A)), float(np.linalg.norm(B))
    if na < 1e-9 or nb < 1e-9:
        return project_concentric(p)
    alpha, beta = math.atan2(A[1], A[0]), math.atan2(B[1], B[0])
    z = (na ** 2) * complex(math.cos(2 * alpha), math.sin(2 * alpha)) \
        - (nb ** 2) * complex(math.cos(2 * beta), math.sin(2 * beta))
    phi = 0.5 * math.atan2(z.imag, z.real)
    a_hat = np.array([math.cos(phi), math.sin(phi)])
    b_hat = np.array([-math.sin(phi), math.cos(phi)])
    U = float(np.dot(A, a_hat)) * a_hat
    V = float(np.dot(B, b_hat)) * b_hat
    return np.array([c + U, c - U, c + V, c - V])


def project_ortho_keeplen(p):
    """Concentric + perpendicular, each diameter keeping its ORIGINAL length.

    Delegates to the canonical shippable implementation so the measurement here and the
    code that would run at inference can never diverge. See experiments/geometry_project.py
    for the derivation and for why length preservation is what makes it parameter-neutral.
    """
    return np.asarray(_canon_hc(_flatten(p)), float).reshape(4, 2)


def project_fa_keeplen(p):
    """FA subspace, each diameter keeping its ORIGINAL length. See project_ortho_keeplen."""
    return np.asarray(_canon_fa(_flatten(p)), float).reshape(4, 2)


PROJECTIONS = {
    "concentric": project_concentric,
    "ortho+concentric": project_orthoconcentric,
    "ortho-keeplen": project_ortho_keeplen,
    "FA subspace": project_fa,
    "FA-keeplen": project_fa_keeplen,
}


# -------------------------------------------------------------------- scoring


def param_abs_err(task, pred, gt):
    """Mean |param error| over the task's parameter list (APPROXIMATE formulas)."""
    dp = derive.derive_parameters(task, pred)
    dg = derive.derive_parameters(task, gt)
    if not dp:
        return None
    return float(np.mean([abs(dp[k] - dg[k]) for k in dp]))


def evaluate(task, which, gt, folds):
    fn = PROJECTIONS[which]
    per_fold = []
    for d in folds:
        keys = sorted(set(d) & set(gt))
        if not keys:
            continue
        raw_m, prj_m, raw_p, prj_p, worse = [], [], [], [], 0
        for k in keys:
            g, p = gt[k], d[k]
            q = fn(p)
            rm = mre_mod.mean_radial_error(p, g)
            pm = mre_mod.mean_radial_error(q, g)
            raw_m.append(rm); prj_m.append(pm)
            worse += rm < pm
            a, b = param_abs_err(task, p, g), param_abs_err(task, q, g)
            if a is not None:
                raw_p.append(a); prj_p.append(b)
        per_fold.append({
            "n": len(keys),
            "mre_raw": statistics.mean(raw_m), "mre_proj": statistics.mean(prj_m),
            "param_raw": statistics.mean(raw_p) if raw_p else None,
            "param_proj": statistics.mean(prj_p) if prj_p else None,
            "frac_images_worse": worse / len(keys),
        })
    if not per_fold:
        return None
    dm = [f["mre_proj"] - f["mre_raw"] for f in per_fold]
    dp = [f["param_proj"] - f["param_raw"] for f in per_fold if f["param_raw"] is not None]
    return {
        "projection": which,
        "per_fold": per_fold,
        "delta_mre_mean": statistics.mean(dm),
        "delta_mre_folds_improved": sum(1 for x in dm if x < 0),
        "delta_mre_ci95_corrected": corrected_ci(dm),
        "delta_param_mean": statistics.mean(dp) if dp else None,
        "delta_param_folds_improved": sum(1 for x in dp if x < 0) if dp else None,
        "delta_param_ci95_corrected": corrected_ci(dp) if dp else None,
        "n_folds": len(per_fold),
    }


def corrected_ci(deltas, n_test_over_train=1 / 4):
    """Nadeau-Bengio corrected-resampled-t 95% CI on a paired k-fold delta.

    The repo's adoption gate uses this rather than the naive paired t, which is
    anticonservative for cross-validation because the folds share training data.
    """
    k = len(deltas)
    if k < 2:
        return None
    m = statistics.mean(deltas)
    var = statistics.variance(deltas)
    se = math.sqrt(var * (1.0 / k + n_test_over_train))
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(k, 2.776)
    return [m - tcrit * se, m + tcrit * se]


def violation_stats(gt, oof, rep):
    """How far predictions sit off the invariants the labels satisfy exactly."""
    keys = sorted(set(gt) & set(oof))
    if not keys or rep["n_pairs"] < 2:
        return {}
    out = {}
    for tag, store in (("perpendicular", "perpendicular_pairs"), ("concentric", "concentric_pairs")):
        for key in rep[store]:
            i, j = (int(x) for x in key.split("-"))
            vals = []
            for k in keys:
                p = oof[k]
                if tag == "concentric":
                    ci = (p[2 * i] + p[2 * i + 1]) / 2
                    cj = (p[2 * j] + p[2 * j + 1]) / 2
                    vals.append(float(np.linalg.norm(ci - cj)))
                else:
                    u = p[2 * i + 1] - p[2 * i]
                    w = p[2 * j + 1] - p[2 * j]
                    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
                    if nu < 1e-9 or nw < 1e-9:
                        continue
                    c = float(np.clip(np.dot(u, w) / (nu * nw), -1, 1))
                    vals.append(abs(90 - math.degrees(math.acos(abs(c)))))
            vals.sort()
            out[f"{tag}_{key}"] = {
                "median": round(statistics.median(vals), 4),
                "p90": round(vals[int(0.9 * (len(vals) - 1))], 4),
                "max": round(vals[-1], 4),
                "unit": "px" if tag == "concentric" else "deg",
            }
    # axis alignment, where the labels demand it
    for i, frac in enumerate(rep["pair_vertical_frac"]):
        if frac == 1.0:
            v = sorted(abs(oof[k][2 * i, 0] - oof[k][2 * i + 1, 0]) for k in keys)
            out[f"pair{i}_x_spread"] = {"median": round(statistics.median(v), 4),
                                        "p90": round(v[int(0.9 * (len(v) - 1))], 4),
                                        "max": round(v[-1], 4), "unit": "px"}
    for i, frac in enumerate(rep["pair_horizontal_frac"]):
        if frac == 1.0:
            v = sorted(abs(oof[k][2 * i, 1] - oof[k][2 * i + 1, 1]) for k in keys)
            out[f"pair{i}_y_spread"] = {"median": round(statistics.median(v), 4),
                                        "p90": round(v[int(0.9 * (len(v) - 1))], 4),
                                        "max": round(v[-1], 4), "unit": "px"}
    return out


def self_check(gt):
    """A projection must be the identity on labels that already satisfy it."""
    out = {}
    for name, fn in PROJECTIONS.items():
        worst = 0.0
        for g in gt.values():
            if len(g) != 4:
                continue
            worst = max(worst, float(np.abs(fn(g) - g).max()))
        out[name] = round(worst, 6)
    return out


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/label_geometry/sweep.json")
    args = ap.parse_args()

    report = {
        "description": "Exact label-geometry invariants per task, prediction violations, "
                       "and the paired 5-fold effect of projecting onto them.",
        "oof_files": OOF_FILES,
        "gt_rounding_tolerance_px": round(ROUND_TOL, 6),
        "angle_tolerance_deg": ANGLE_TOL_DEG,
        "param_mae_is_estimate": True,
        "tasks": {},
    }

    for task in TASKS:
        gt = load_gt(task)
        oof = load_oof(task)
        folds = load_oof_by_fold(task)
        rep = detect(task, gt)
        rep["oof_images"] = len(set(gt) & set(oof))
        rep["violations"] = violation_stats(gt, oof, rep)

        # pick the projections this task's own labels justify
        applicable = []
        if rep["n_landmarks"] == 4:
            if rep["concentric_pairs"]:
                applicable.append("concentric")
                if rep["perpendicular_pairs"]:
                    applicable += ["ortho+concentric", "ortho-keeplen"]
            if (rep["pair_vertical_frac"][:1] == [1.0]
                    and rep["pair_horizontal_frac"][1:2] == [1.0]
                    and rep["concentric_pairs"]):
                applicable += ["FA subspace", "FA-keeplen"]
        rep["projections_applicable"] = applicable
        if applicable:
            rep["identity_on_labels_max_abs_px"] = self_check(gt)
            rep["projection_results"] = [evaluate(task, w, gt, folds) for w in applicable]
        report["tasks"][task] = rep
        print(f"[{task:12}] n={rep['n']:4} pairs={rep['n_pairs']:2} "
              f"perp={list(rep['perpendicular_pairs'])} conc={list(rep['concentric_pairs'])} "
              f"applicable={applicable}")

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
