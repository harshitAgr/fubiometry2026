"""Prepare VALIDATION data for inference.

Extracts val images and builds the CSVs the baseline's InferenceDataset reads
(columns: image_path, task_id, task_name, num_classes). Inference needs no GT points.

FUGC uses the FUGC folder INSIDE val_data.zip (val_data/FUGC/0001.png..; 50 imgs, 544x336 =
the official scored set). The separate FUGC_val.zip (FUGA_val/, 20 raw 1136x735 imgs) is NOT the
scored val — it matched 0 of the 50 official FUGC GT (proven by sub 809881) — so it is NOT used.

Output (a self-contained inference data_root):
  data/val/csv/<task>_val.csv
  data/val/images/<task>/<file>

Run: uv run python scripts/prepare_val.py
"""
from __future__ import annotations
import os
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/drive_raw")
VAL_ROOT = Path("data/val")
IMG_OUT = VAL_ROOT / "images"
CSV_OUT = VAL_ROOT / "csv"
IMG_EXT = (".png", ".jpg", ".jpeg")


def num_classes_map() -> dict[str, int]:
    nc = {}
    for f in Path("data/csv").glob("*_train.csv"):
        d = pd.read_csv(f, nrows=1)
        nc[str(d["task_id"].iloc[0])] = int(d["num_classes"].iloc[0])
    return nc


def _extract(zip_rel: str, task_of, files_by_task: dict[str, list[str]]):
    with zipfile.ZipFile(RAW / zip_rel) as z:
        for n in z.namelist():
            if n.endswith("/") or not n.lower().endswith(IMG_EXT):
                continue
            task = task_of(n)
            if task is None:
                continue
            base = os.path.basename(n)
            out = IMG_OUT / task
            out.mkdir(parents=True, exist_ok=True)
            (out / base).write_bytes(z.read(n))
            files_by_task.setdefault(task, []).append(base)


def main():
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    nc = num_classes_map()
    files_by_task: dict[str, list[str]] = {}

    # val_data.zip: val_data/<TASK>/<file> — includes the REAL FUGC (50 cropped imgs = scored set).
    def task_of_valdata(n: str):
        parts = n.split("/")
        if len(parts) < 3 or not parts[1]:
            return None
        return parts[1]

    _extract("val_data.zip", task_of_valdata, files_by_task)
    # NB: FUGC_val.zip (FUGA_val, 20 raw 1136x735 imgs) is the WRONG set (0/50 GT match) — NOT used.

    total = 0
    for task, files in sorted(files_by_task.items()):
        if task not in nc:
            raise ValueError(f"val task {task!r} has no train num_classes")
        rows = [{"image_path": f"{task}/{b}", "task_id": task,
                 "task_name": "Regression", "num_classes": nc[task]}
                for b in sorted(files)]
        pd.DataFrame(rows).to_csv(CSV_OUT / f"{task}_val.csv", index=False)
        total += len(files)
        print(f"  {task:12s} val_imgs={len(files):4d} num_classes={nc[task]}")
    print(f"Done. {total} val images across {len(files_by_task)} tasks -> {VAL_ROOT}")


if __name__ == "__main__":
    main()
