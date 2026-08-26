#!/usr/bin/env python3
"""Build a validation candidate = base submission + the label-geometry projection.

Pure post-processing on an existing prediction JSON: no model, no GPU, no re-inference.
Only FA and HC coordinates change; the other seven tasks must come out byte-identical,
which makes the resulting official comparison a clean single-variable A/B.

Follows experiments/build_hc_scale_candidate.py (the v21/v22 builder) so the audit
record has the same shape.

    uv run python experiments/build_geometry_projection_candidate.py \
        --base submission/v22/regression_predictions.json --out submission/v23

Writes <out>/regression_predictions.json and <out>/candidate_audit.json. Does NOT zip
and does NOT submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.geometry_project import PROJECTED_TASKS, project  # noqa: E402


def _pairs(flat):
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


def _diams(flat):
    p = _pairs(flat)
    return math.dist(p[0], p[1]), math.dist(p[2], p[3])


def _centre_gap(flat):
    p = _pairs(flat)
    c1 = ((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2)
    c2 = ((p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2)
    return math.dist(c1, c2)


def _angle(flat):
    p = _pairs(flat)
    ux, uy = p[1][0] - p[0][0], p[1][1] - p[0][1]
    vx, vy = p[3][0] - p[2][0], p[3][1] - p[2][1]
    c = (ux * vx + uy * vy) / (math.hypot(ux, uy) * math.hypot(vx, vy))
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(c)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submission/v22/regression_predictions.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = json.load(open(a.base))
    audit = {
        "base": a.base,
        "base_md5": hashlib.md5(open(a.base, "rb").read()).hexdigest(),
        "projection": "length-preserving label-geometry projection (experiments/geometry_project.py)",
        "projected_tasks": sorted(PROJECTED_TASKS),
        "records_base": len(base),
    }

    out = []
    moved, len_err, gap_before, gap_after, ang_before, ang_after = [], [], [], [], [], []
    untouched_identical = True
    for r in base:
        t = r["task_id"]
        src = list(r["predicted_points_pixels"])
        if t not in PROJECTED_TASKS:
            out.append(r)
            continue
        dst = project(src, t)
        d0b, d1b = _diams(src)
        d0a, d1a = _diams(dst)
        len_err.append(max(abs(d0a - d0b), abs(d1a - d1b)))
        moved.append(max(abs(x - y) for x, y in zip(src, dst)))
        gap_before.append(_centre_gap(src)); gap_after.append(_centre_gap(dst))
        ang_before.append(abs(90.0 - _angle(src))); ang_after.append(abs(90.0 - _angle(dst)))
        rr = dict(r)
        rr["predicted_points_pixels"] = dst
        out.append(rr)

    ref = {(r["task_id"], r["image_path"]): r["predicted_points_pixels"] for r in base}
    for r in out:
        if r["task_id"] in PROJECTED_TASKS:
            continue
        if r["predicted_points_pixels"] != ref[(r["task_id"], r["image_path"])]:
            untouched_identical = False

    by_task = {}
    for r in base:
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + 1

    finite = all(math.isfinite(v) for r in out for v in r["predicted_points_pixels"])
    keys_identical = ([(r["task_id"], r["image_path"]) for r in base]
                      == [(r["task_id"], r["image_path"]) for r in out])

    audit.update({
        "records": len(out),
        "records_by_task": by_task,
        "records_projected": len(moved),
        "other_tasks_byte_identical": untouched_identical,
        "keys_identical": keys_identical,
        "all_finite": finite,
        "max_diameter_length_change_px": max(len_err) if len_err else 0.0,
        "max_landmark_move_px": max(moved) if moved else 0.0,
        "mean_landmark_move_px": (sum(moved) / len(moved)) if moved else 0.0,
        "centre_gap_px": {"mean_before": sum(gap_before) / len(gap_before),
                          "max_after": max(gap_after)},
        "perpendicularity_deviation_deg": {"mean_before": sum(ang_before) / len(ang_before),
                                           "max_after": max(ang_after)},
    })
    audit["PASS"] = bool(
        untouched_identical and keys_identical and finite
        and audit["records"] == audit["records_base"]
        and audit["max_diameter_length_change_px"] < 1e-6      # parameter neutrality
        and audit["centre_gap_px"]["max_after"] < 1e-6         # constraint achieved
        and audit["perpendicularity_deviation_deg"]["max_after"] < 1e-6
    )

    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, "regression_predictions.json")
    json.dump(out, open(pj, "w"))
    audit["out_md5"] = hashlib.md5(open(pj, "rb").read()).hexdigest()
    json.dump(audit, open(os.path.join(a.out, "candidate_audit.json"), "w"), indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
