"""Numerically diff the CONTAINER inference path against the validated research path.

The container's `model.py` is a rewrite of `experiments/infer_ensemble.py` (no albumentations, no
pandas, images batched within a task, one row guaranteed per metadata row). This script proves the
rewrite is faithful by running it over the official validation images with the SAME checkpoints
that produced a known-good reference submission, then comparing pixel coordinates.

Reference default = `submission/ship_dino/regression_predictions.json`, independently built by
`experiments/infer_ensemble.py` plus `experiments/build_dino_ship_candidate.py`: five full-data
continued-DINO base seeds, the continued-DINO hcsmall/hchead route, size-gated HC scale s=0.975,
FA/HC geometry projection, and gated IVC length calibration. The research driver and vendored
Docker path instantiate, execute, and average the models differently, so this is a meaningful
packaging-equivalence check rather than a self-comparison.

Usage (baseline venv, from repo root):
  baseline/.venv-baseline/bin/python docker/verify_against_reference.py --per-task 5
  baseline/.venv-baseline/bin/python docker/verify_against_reference.py --all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER_DIR = os.path.join(PROJ, "docker")

# 5 continued-DINO base seeds + the two continued-DINO specialists. These must stay in step
# with docker/weights/ and MEMBER_GROUPS.
DEFAULT_CKPTS = {
    "best_model.pth": "runs/vitb_full_dino_corr/best_model.pth",
    "model_s43.pth": "runs/vitb_full_dino_corr_s43/best_model.pth",
    "model_s44.pth": "runs/vitb_full_dino_corr_s44/best_model.pth",
    "model_s45.pth": "runs/vitb_full_dino_corr_s45/best_model.pth",
    "model_s46.pth": "runs/vitb_full_dino_corr_s46/best_model.pth",
    "model_hcsmall.pth": "runs/vitb_full_dino_hcsmall_corr/best_model.pth",
    "model_hchead.pth": "runs/vitb_full_dino_hchead_corr/best_model.pth",
}


def build_workdir(work, val_dir, per_task, ckpts):
    """Create a container-shaped {work}/images,{work}/csv plus an /app-shaped checkpoint dir."""
    images = os.path.join(work, "images")
    csv_dir = os.path.join(work, "csv")
    app = os.path.join(work, "app")
    for d in (images, csv_dir, app):
        os.makedirs(d, exist_ok=True)

    for name, rel in ckpts.items():
        src = os.path.join(PROJ, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"checkpoint missing: {src}")
        link = os.path.join(app, name)
        # The work directory is deliberately reusable, but the candidate checkpoint set can
        # change. Never retain a stale symlink from an earlier Docker candidate.
        if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(src):
            os.unlink(link)
        if not os.path.lexists(link):
            os.symlink(src, link)

    rows = []
    for csv_path in sorted(glob.glob(os.path.join(val_dir, "csv", "*_val.csv"))):
        recs = list(csv.DictReader(open(csv_path)))
        if per_task:
            recs = recs[:per_task]
        for r in recs:
            task = r["task_id"]
            src_dir = os.path.join(val_dir, "images", task)
            link = os.path.join(images, task)
            if not os.path.lexists(link):
                os.symlink(src_dir, link)
            img = cv2.imread(os.path.join(src_dir, os.path.basename(r["image_path"])))
            if img is None:
                raise FileNotFoundError(f"cannot read {r['image_path']}")
            h, w = img.shape[:2]
            rows.append({
                "image_path": r["image_path"], "task_name": "Regression", "task_id": task,
                "num_classes": r["num_classes"], "height": h, "width": w,
            })

    meta = os.path.join(csv_dir, "test_metadata.csv")
    with open(meta, "w", newline="") as f:
        wtr = csv.DictWriter(
            f, fieldnames=["image_path", "task_name", "task_id", "num_classes", "height", "width"])
        wtr.writeheader()
        wtr.writerows(rows)
    return app, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default=os.path.join(PROJ, "data", "val"))
    ap.add_argument("--reference",
                    default=os.path.join(PROJ, "submission", "ship_dino",
                                         "regression_predictions.json"))
    ap.add_argument("--work", default=os.path.join(PROJ, "scratch_tmp", "fub_container_verify"))
    ap.add_argument("--per-task", type=int, default=5,
                    help="images per task (0 = all)")
    ap.add_argument("--all", action="store_true", help="shorthand for --per-task 0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max allowed per-coordinate pixel difference")
    args = ap.parse_args()
    per_task = 0 if args.all else args.per_task

    app, n = build_workdir(args.work, args.val_dir, per_task, DEFAULT_CKPTS)
    print(f"[verify] work={args.work}  rows={n}")

    os.environ["FUB_APP_DIR"] = app
    sys.path.insert(0, DOCKER_DIR)
    from model import Model                                   # noqa: E402

    out_dir = os.path.join(args.work, "out")
    Model().predict(data_root=args.work, output_dir=out_dir, batch_size=args.batch_size)

    got = {(r["task_id"], r["image_path"]): r["predicted_points_pixels"]
           for r in json.load(open(os.path.join(out_dir, "regression_predictions.json")))}
    ref = {(r["task_id"], r["image_path"]): r["predicted_points_pixels"]
           for r in json.load(open(args.reference))}

    shared = sorted(set(got) & set(ref))
    missing = sorted(set(got) - set(ref))
    if not shared:
        print("[verify] FAIL: no overlapping keys with the reference")
        return 1
    if missing:
        print(f"[verify] WARNING: {len(missing)} produced keys absent from reference "
              f"(e.g. {missing[:3]})")

    per_task_max, worst = {}, (0.0, None)
    for k in shared:
        a, b = np.asarray(got[k], float), np.asarray(ref[k], float)
        if a.shape != b.shape:
            print(f"[verify] FAIL: shape mismatch at {k}: {a.shape} vs {b.shape}")
            return 1
        d = float(np.abs(a - b).max())
        per_task_max[k[0]] = max(per_task_max.get(k[0], 0.0), d)
        if d > worst[0]:
            worst = (d, k)

    print(f"[verify] compared {len(shared)} images")
    for t in sorted(per_task_max):
        print(f"    {t:14s} max|dpx| = {per_task_max[t]:.6g}")
    print(f"[verify] worst: {worst[0]:.6g} px at {worst[1]}")
    ok = worst[0] <= args.tol
    print(f"[verify] {'PASS' if ok else 'FAIL'} (tolerance {args.tol} px)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
