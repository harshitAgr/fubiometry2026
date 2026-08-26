#!/usr/bin/env python3
"""Compose the FINAL-TEST ship reference and cross-check the container's post-processing.

The container ships a lever set that was never itself submitted:

    v19 raw family route
      -> gated HC ellipse scale s=0.975   (v21 applied this UNGATED; on validation the gate
                                           fires 215/215, so v21 == the gated result here)
      -> FA/HC label-geometry projection  (v23's lever)
      -> gated IVC length calibration     (v24's lever)

v24 (882078) is the same chain at HC scale s=0.950. That gives a strong, NON-tautological
check on the two newly vendored container modules, because of which tasks each lever owns:

    * the 6 untouched tasks   -> must be byte-identical across v21 / v24 / ship
    * FA   (projection only)  -> ship FA  must equal v24 FA  exactly
    * IVC  (gate only)        -> ship IVC must equal v24 IVC exactly
    * HC   (all three)        -> must DIFFER from v24, by exactly the 0.975/0.950 scale ratio

So if docker/geometry_project.py or docker/ivc_calibrate.py deviated at all from the code
that produced the officially-scored v24, the FA/IVC equalities below would fail.

    uv run python experiments/build_ship_candidate.py

Writes submission/ship_v29/{regression_predictions.json,candidate_audit.json}. Does NOT zip,
does NOT submit, does NOT touch docker/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docker"))

# The EXACT modules the container imports -- this is the point of the exercise.
from docker.geometry_project import project                       # noqa: E402
from docker.hc_scale import HC_SCALE, apply_hc_scale              # noqa: E402
from docker.ivc_calibrate import BAND_HI, BAND_LO, TARGET_LEN, apply_ivc_calibration  # noqa: E402

BASE = os.path.join(ROOT, "submission/v21/regression_predictions.json")   # v19 + s=0.975
V22 = os.path.join(ROOT, "submission/v22/regression_predictions.json")    # v19 + s=0.950
V24 = os.path.join(ROOT, "submission/v24/regression_predictions.json")    # v22 + proj + IVC
OUT = os.path.join(ROOT, "submission/ship_v29")
VAL_META = os.path.join(ROOT, "data/val/csv/val_metadata.csv")


def key(r):
    return (r["task_id"], r["image_path"])


def load(p):
    return {key(r): r for r in json.load(open(p))}


def maxdiff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def native_hw(keys):
    """(width, height) per key, read from the actual image file.

    The validation CSVs carry only image_path/task_id/num_classes, and the container reads the
    native size from the decoded image (`img.shape[:2]`) rather than from metadata -- so read
    it the same way here. Only HC rows need it (the scale gate); everything else is inert.
    """
    from PIL import Image
    out = {}
    for task_id, rel in keys:
        p = os.path.join(ROOT, "data/val/images", rel)
        with Image.open(p) as im:
            out[(task_id, rel)] = im.size            # PIL size is (width, height)
    return out


def main():
    base, v22, v24 = load(BASE), load(V22), load(V24)
    assert set(base) == set(v24) == set(v22), "key sets differ between artifacts"
    hw = native_hw(list(base))

    ship, stats = [], {"scaled": 0, "projected": 0, "calibrated": 0}
    fired_ivc, gate_skipped_hc = [], []
    for k, r in base.items():
        t = r["task_id"]
        w0, h0 = hw[k]
        px = list(r["predicted_points_pixels"])

        # NOTE: base is already s=0.975 applied UNGATED (that is what v21 is). Re-running the
        # gated function here would double-apply, so instead we ASSERT the gate would have
        # fired on this row -- i.e. that gated and ungated coincide on validation, which is
        # what makes v21 a valid stand-in for the gated pipeline.
        if t == "HC":
            probe = apply_hc_scale([1000.0, 1000.0, 1000.0, 900.0, 1050.0, 950.0, 950.0, 950.0],
                                   t, w0, h0)
            if probe == [1000.0, 1000.0, 1000.0, 900.0, 1050.0, 950.0, 950.0, 950.0]:
                gate_skipped_hc.append(r["image_path"])
            else:
                stats["scaled"] += 1

        proj = project(px, t)
        if proj != px:
            stats["projected"] += 1
        final = apply_ivc_calibration(proj, t)
        if final != proj:
            stats["calibrated"] += 1
            fired_ivc.append((r["image_path"],
                              round(math.dist((px[0], px[1]), (px[2], px[3])), 2)))

        rr = dict(r)
        rr["predicted_points_pixels"] = final
        ship.append(rr)

    S = {key(r): r for r in ship}

    # ---- cross-checks against the officially scored v24 -------------------------------
    checks, worst = {}, {}
    for t, expect_equal in (("FA", True), ("IVC", True), ("HC", False),
                            ("A4C", True), ("AOP", True), ("FUGC", True),
                            ("PLAX", True), ("PSAX", True), ("fetal_femur", True)):
        ks = [k for k in S if k[0] == t]
        d = max(maxdiff(S[k]["predicted_points_pixels"], v24[k]["predicted_points_pixels"])
                for k in ks)
        worst[t] = d
        if expect_equal:
            checks[f"{t}_matches_v24"] = d < 1e-9
        else:
            # HC is the ONE task the two artifacts must disagree on (s=0.975 vs s=0.950).
            checks[f"{t}_differs_from_v24_as_expected"] = d > 1e-6

    # HC must differ from v24 by exactly the scale ratio about the shared centroid.
    ratios = []
    for k in (k for k in S if k[0] == "HC"):
        a, b = S[k]["predicted_points_pixels"], v24[k]["predicted_points_pixels"]
        ca = (sum(a[0::2]) / 4.0, sum(a[1::2]) / 4.0)
        cb = (sum(b[0::2]) / 4.0, sum(b[1::2]) / 4.0)
        ra = math.dist((a[0], a[1]), ca)
        rb = math.dist((b[0], b[1]), cb)
        if rb > 1e-6:
            ratios.append(ra / rb)
    expected_ratio = HC_SCALE / 0.950
    checks["HC_scale_ratio_vs_v24_exact"] = bool(
        ratios and max(abs(x - expected_ratio) for x in ratios) < 1e-9)

    # Parameter neutrality of the projection: FA/HC diameters unchanged from the base.
    len_err = 0.0
    for k, r in base.items():
        if k[0] not in ("FA", "HC"):
            continue
        p, q = r["predicted_points_pixels"], S[k]["predicted_points_pixels"]
        for i in (0, 2):
            len_err = max(len_err, abs(math.dist((p[2 * i], p[2 * i + 1]),
                                                 (p[2 * i + 2], p[2 * i + 3]))
                                       - math.dist((q[2 * i], q[2 * i + 1]),
                                                   (q[2 * i + 2], q[2 * i + 3]))))
    checks["projection_preserves_diameters"] = len_err < 1e-6

    by_task = {}
    for r in ship:
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + 1
    checks["records_619"] = len(ship) == 619
    checks["nine_tasks"] = len(by_task) == 9
    checks["all_finite"] = all(math.isfinite(v) for r in ship
                               for v in r["predicted_points_pixels"])
    checks["keys_identical_to_base"] = [key(r) for r in ship] == [k for k in base]
    checks["hc_gate_fired_on_all_val_hc"] = not gate_skipped_hc
    checks["ivc_gate_fired_3"] = stats["calibrated"] == 3

    audit = {
        "purpose": "final-test container ship reference",
        "chain": [f"v19 family route -> gated HC ellipse scale s={HC_SCALE}",
                  "FA/HC length-preserving label-geometry projection",
                  f"gated IVC length calibration band=[{BAND_LO}, {BAND_HI}] "
                  f"target={TARGET_LEN}"],
        "base": os.path.relpath(BASE, ROOT),
        "base_md5": hashlib.md5(open(BASE, "rb").read()).hexdigest(),
        "cross_check_against": os.path.relpath(V24, ROOT),
        "note": ("v24 is the same chain at HC scale 0.950; FA and IVC are untouched by the HC "
                 "scale, so their exact equality with v24 validates the vendored container "
                 "modules against an officially scored artifact."),
        "records_by_task": by_task,
        "counts": stats,
        "ivc_gate_fired": fired_ivc,
        "hc_gate_skipped_images": gate_skipped_hc,
        "worst_abs_coord_diff_vs_v24_px": {k: round(v, 12) for k, v in worst.items()},
        "hc_scale_ratio_vs_v24": {"expected": expected_ratio,
                                  "observed_max_dev": (max(abs(x - expected_ratio)
                                                           for x in ratios)
                                                       if ratios else None)},
        "max_diameter_length_change_px": len_err,
        "checks": checks,
    }
    audit["PASS"] = all(checks.values())

    os.makedirs(OUT, exist_ok=True)
    pj = os.path.join(OUT, "regression_predictions.json")
    json.dump(ship, open(pj, "w"))
    audit["out_md5"] = hashlib.md5(open(pj, "rb").read()).hexdigest()
    json.dump(audit, open(os.path.join(OUT, "candidate_audit.json"), "w"), indent=2)
    print(json.dumps(audit, indent=2))
    return 0 if audit["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
