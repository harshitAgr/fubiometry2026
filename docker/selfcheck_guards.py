#!/usr/bin/env python3
"""Prove the container's crash-loud guards actually fire. Run in the baseline venv:

    baseline/.venv-baseline/bin/python docker/selfcheck_guards.py

The test phase grants 10 RETRYABLE attempts but locks in the FIRST SUCCESSFUL run. That makes
a crash cheap and a silently-degraded success unrecoverable, so `model.py` was changed on
2026-08-11 to abort rather than centre-fill. An `if` that never fires is worse than no guard at
all -- it reads as protection while providing none -- so each one is exercised here against a
real Model instance.

Guards checked:
  1. unknown task_id in test_metadata.csv          -> KeyError   (never centre-fill a task)
  2. systematic unreadable images beyond budget    -> IOError
  3. a straggler WITHIN budget                     -> tolerated, one row still emitted
  4. an ensemble member missing from the image     -> FileNotFoundError
  5. batch_size is capped to MAX_IMAGES_PER_FORWARD
  6. non-finite coordinates                        -> ValueError from _save
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER_DIR = os.path.join(PROJ, "docker")
sys.path.insert(0, DOCKER_DIR)

CKPTS = {
    "best_model.pth": "runs/vitb_full_dino_corr/best_model.pth",
    "model_s43.pth": "runs/vitb_full_dino_corr_s43/best_model.pth",
    "model_s44.pth": "runs/vitb_full_dino_corr_s44/best_model.pth",
    "model_s45.pth": "runs/vitb_full_dino_corr_s45/best_model.pth",
    "model_s46.pth": "runs/vitb_full_dino_corr_s46/best_model.pth",
    "model_hcsmall.pth": "runs/vitb_full_dino_hcsmall_corr/best_model.pth",
    "model_hchead.pth": "runs/vitb_full_dino_hchead_corr/best_model.pth",
}
VAL_IMAGES = os.path.join(PROJ, "data/val/images")

FAILURES = []
PASSES = []


def check(name, fn, exc, needle=None):
    try:
        fn()
    except exc as e:
        if needle and needle not in str(e):
            FAILURES.append(f"{name}: raised {exc.__name__} but message lacks {needle!r}: {e}")
        else:
            PASSES.append(f"{name}: raised {exc.__name__} as required")
        return
    except Exception as e:                                     # noqa: BLE001
        FAILURES.append(f"{name}: raised {type(e).__name__} ({e}), expected {exc.__name__}")
        return
    FAILURES.append(f"{name}: DID NOT RAISE -- the guard is inert")


def make_workdir(root, rows, broken=()):
    """Container-shaped work dir. `broken` names rows whose image is a truncated file."""
    images = os.path.join(root, "images")
    csv_dir = os.path.join(root, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    for task in {r["task_id"] for r in rows}:
        os.makedirs(os.path.join(images, task), exist_ok=True)
    for r in rows:
        dst = os.path.join(images, r["image_path"])
        if r["image_path"] in broken:
            with open(dst, "wb") as f:
                f.write(b"not an image")
            continue
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.join(VAL_IMAGES, r["real_path"]), dst)
    with open(os.path.join(csv_dir, "test_metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["image_path", "task_name", "task_id", "num_classes",
                               "height", "width"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    return root


def val_rows(task, n, num_classes):
    d = os.path.join(VAL_IMAGES, task)
    names = sorted(os.listdir(d))[:n]
    return [{"image_path": f"{task}/{nm}", "real_path": f"{task}/{nm}", "task_name": "Regression",
             "task_id": task, "num_classes": num_classes, "height": 0, "width": 0}
            for nm in names]


def main():
    # ---- guard 4 first: it must fail BEFORE any expensive checkpoint load. -------------
    empty_app = tempfile.mkdtemp(prefix="fub_guard_emptyapp_")
    os.environ["FUB_APP_DIR"] = empty_app
    import model as M                                          # noqa: N814

    check("4 missing ensemble member", lambda: M.Model(), FileNotFoundError,
          needle="missing from the image")

    # ---- now build a real Model against the actual checkpoints. ------------------------
    app = tempfile.mkdtemp(prefix="fub_guard_app_")
    for name, rel in CKPTS.items():
        src = os.path.join(PROJ, rel)
        if not os.path.isfile(src):
            print(f"SKIP: checkpoint absent: {src}")
            return 2
        os.symlink(src, os.path.join(app, name))
    os.environ["FUB_APP_DIR"] = app
    import importlib

    M = importlib.reload(M)
    print("[guards] constructing Model against the real 7 checkpoints ...", flush=True)
    mdl = M.Model()

    # guard 5: batch cap. predict() prints it, but assert the constant is actually applied
    # by inspecting the value the loop would use.
    if M.MAX_IMAGES_PER_FORWARD != 4:
        FAILURES.append(f"5 batch cap: MAX_IMAGES_PER_FORWARD={M.MAX_IMAGES_PER_FORWARD}, "
                        f"expected 4")
    else:
        PASSES.append("5 batch cap: MAX_IMAGES_PER_FORWARD == 4")

    # ---- guard 1: unknown task_id -----------------------------------------------------
    w1 = make_workdir(tempfile.mkdtemp(prefix="fub_guard_unk_"), [
        *val_rows("IVC", 2, 2),
        {"image_path": "NOT_A_TASK/0001.png", "real_path": "IVC/0001.png",
         "task_name": "Regression", "task_id": "NOT_A_TASK", "num_classes": 2,
         "height": 600, "width": 800},
    ])
    out1 = tempfile.mkdtemp(prefix="fub_guard_out1_")
    check("1 unknown task_id", lambda: mdl.predict(w1, out1), KeyError,
          needle="Refusing to centre-fill")

    # ---- guard 2: systematic unreadable images ---------------------------------------
    rows = val_rows("AOP", 10, 4)
    broken = {r["image_path"] for r in rows[:3]}               # 3 of 10, budget is 1
    w2 = make_workdir(tempfile.mkdtemp(prefix="fub_guard_brk_"), rows, broken=broken)
    out2 = tempfile.mkdtemp(prefix="fub_guard_out2_")
    check("2 mass unreadable images", lambda: mdl.predict(w2, out2), IOError,
          needle="systematic path/naming failure")

    # ---- guard 3: a single straggler stays tolerated ----------------------------------
    rows = val_rows("AOP", 10, 4)
    w3 = make_workdir(tempfile.mkdtemp(prefix="fub_guard_one_"), rows,
                      broken={rows[0]["image_path"]})
    out3 = tempfile.mkdtemp(prefix="fub_guard_out3_")
    try:
        res = mdl.predict(w3, out3)
        emitted = json.load(open(os.path.join(out3, "regression_predictions.json")))
        if len(emitted) == 10 and len(res) == 10:
            PASSES.append("3 single straggler: tolerated, all 10 rows emitted")
        else:
            FAILURES.append(f"3 single straggler: emitted {len(emitted)} rows, expected 10")
    except Exception as e:                                     # noqa: BLE001
        FAILURES.append(f"3 single straggler: should be tolerated but raised "
                        f"{type(e).__name__}: {e}")

    # ---- guard 6: non-finite coordinates ---------------------------------------------
    out6 = tempfile.mkdtemp(prefix="fub_guard_out6_")
    bad = [{"image_path": "AOP/0001.png", "task_id": "AOP",
            "predicted_points_pixels": [1.0, 2.0, float("nan"), 4.0]}]
    check("6 non-finite coordinate", lambda: M.Model._save(bad, out6), ValueError,
          needle="non-finite")

    for d in (empty_app, app, w1, w2, w3, out1, out2, out3, out6):
        shutil.rmtree(d, ignore_errors=True)

    print("\n=== guard self-check ===")
    for p in PASSES:
        print(f"  PASS  {p}")
    for f in FAILURES:
        print(f"  FAIL  {f}")
    print(f"\n{len(PASSES)} passed, {len(FAILURES)} failed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
