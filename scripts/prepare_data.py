"""Prepare the FU_Biometry TRAIN data into the layout the baseline expects.

Reads the raw per-task download under data/drive_raw/<TASK>/labeled/ and produces:
  data/images/<task>/<basename>            (flattened labeled images)
  data/csv/<task>_train.csv                (normalized: image_path, task_name,
                                            task_id, num_classes, point_1_xy..N)

Cardiac CSVs (A4C/PLAX/PSAX/IVC) use named anatomical *_xy columns; we map them to
point_{i}_xy IN COLUMN ORDER (this order also defines the parameter->landmark mapping
used by scoring/param_specs.py). The other tasks already use point_{i}_xy.

Run: uv run python scripts/prepare_data.py
"""
from __future__ import annotations
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/drive_raw")
IMG_OUT = Path("data/images")
CSV_OUT = Path("data/csv")

# task_id -> (labeled_zip, csv_relpath, csv_encoding)
# CARDIAC LABEL SAGA (2026-07-17 -> 2026-07-18): organizers first shipped a "corrected" newcsv
# that SWAPPED the left-right endpoint pairs, then RETRACTED it the next day -- they had been
# misled by participant feedback; the ORIGINAL CSVs were entirely correct. The confirmed
# official standard is VERTICAL: within every odd-even pair (1-2,3-4,...,15-16) the FIRST
# landmark is ABOVE the second. Verified: the original raw (below) is 99-100% compliant;
# yesterday's swap violated it (78-83%). We briefly ingested the bogus swap and started CV +
# ensemble on it, then killed both and REVERTED A4C/PSAX to the original raw here. The
# re-posted Drive newcsv (new ids A4C=1_In7-tuplTGesotYmns9YFLCiZubMJjU,
# PSAX=1YXoe2dq3_n2yrE-K6I8I8mIqf-BV9hmn) was verified BYTE-IDENTICAL to this original raw
# (diff 0/108 A4C, 0/49 PSAX). => cardiac labels == what v12 trained on (no net change).
TASKS = {
    "A4C": ("A4C/labeled/A4C_labeled.zip", "A4C/labeled/A4C_train.csv", "gb18030"),
    "AOP": ("AOP/labeled/AOP.zip", "AOP/labeled/key_points_xy.csv", "utf-8"),
    "FA": ("FA/labeled/FA.zip", "FA/labeled/FA_train_new.csv", "utf-8"),
    "FUGC": ("FUGC/labeled/FUGC.zip", "FUGC/labeled/FUGC.csv", "utf-8"),
    "HC": ("HC/labeled/HC.zip", "HC/labeled/HC.csv", "utf-8"),
    "IVC": ("IVC/labeled/IVC.zip", "IVC/labeled/train.csv", "utf-8"),
    "PLAX": ("PLAX/labeled/PLAX.zip", "PLAX/labeled/train.csv", "utf-8"),
    "PSAX": ("PSAK/labeled/PSAX.zip", "PSAK/labeled/train.csv", "utf-8"),
    "fetal_femur": ("fetal_femur/labeled/fetal_femur.zip",
                    "fetal_femur/labeled/Reg-Two_3.fetal_femur.csv", "utf-8"),
}

IMG_EXT = (".png", ".jpg", ".jpeg")

# Organizer-flagged bad images to DISREGARD (message 2026-07-16, actioned 2026-07-17):
# 25 fetal_femur images are a horizontal mirror/flip of the original acquisition
# (non-standard orientation -> violate the landmark convention). Organizers: "please
# disregard these 25 problematic images. All images in the test set have been verified
# and maintain standard orientation." Dropped from BOTH training and CV eval. (A4C/PSAX
# received corrected labels via newcsv instead — see TASKS above.)
DROP_IMAGES: dict[str, set[str]] = {
    "fetal_femur": {
        "Patient00757_Plane5_1_of_1.png", "Patient00863_Plane5_1_of_2.png",
        "Patient00863_Plane5_2_of_2.png", "Patient01025_Plane5_1_of_1.png",
        "Patient01035_Plane5_2_of_4.png", "Patient01035_Plane5_4_of_4.png",
        "Patient01130_Plane5_2_of_4.png", "Patient01221_Plane5_2_of_2.png",
        "Patient01246_Plane5_1_of_2.png", "Patient01248_Plane5_1_of_1.png",
        "Patient01249_Plane5_1_of_1.png", "Patient01301_Plane5_1_of_2.png",
        "Patient01301_Plane5_2_of_2.png", "Patient01304_Plane5_2_of_2.png",
        "Patient01475_Plane5_1_of_1.png", "Patient01476_Plane5_1_of_1.png",
        "Patient01477_Plane5_1_of_2.png", "Patient01478_Plane5_1_of_1.png",
        "Patient01480_Plane5_1_of_1.png", "Patient01481_Plane5_1_of_1.png",
        "Patient01605_Plane5_2_of_2.png", "Patient01606_Plane5_2_of_2.png",
        "Patient01607_Plane5_1_of_2.png", "Patient01608_Plane5_1_of_1.png",
        "Patient01609_Plane5_1_of_1.png",
    },
}


def extract_images(task: str, zip_rel: str) -> set[str]:
    out = IMG_OUT / task
    out.mkdir(parents=True, exist_ok=True)
    names = set()
    with zipfile.ZipFile(RAW / zip_rel) as z:
        for n in z.namelist():
            if n.endswith("/") or not n.lower().endswith(IMG_EXT):
                continue
            base = os.path.basename(n)
            (out / base).write_bytes(z.read(n))
            names.add(base)
    return names


def normalize_csv(task: str, csv_rel: str, enc: str, img_basenames: set[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW / csv_rel, encoding=enc)
    point_cols = [c for c in df.columns if str(c).endswith("_xy")]
    if not point_cols:
        raise ValueError(f"{task}: no *_xy point columns in {csv_rel}")

    # Validate point cells parse to [x, y].
    sample = df.iloc[0]
    for c in point_cols:
        v = sample[c]
        xy = json.loads(v) if isinstance(v, str) else v
        if len(xy) != 2:
            raise ValueError(f"{task}: column {c} is not an [x,y] pair: {v!r}")

    out = pd.DataFrame()
    out["image_path"] = df["image_path"].map(lambda p: f"{task}/{os.path.basename(str(p))}")
    out["task_name"] = "Regression"
    out["task_id"] = task
    out["num_classes"] = len(point_cols)
    for i, c in enumerate(point_cols, start=1):
        out[f"point_{i}_xy"] = df[c]

    # Drop organizer-flagged bad images (see DROP_IMAGES).
    drop = DROP_IMAGES.get(task, set())
    if drop:
        before = len(out)
        out = out[~out["image_path"].map(lambda p: os.path.basename(p) in drop)].reset_index(drop=True)
        print(f"  {task:12s} dropped {before - len(out)} organizer-flagged bad image(s)")

    # How many CSV rows resolve to an extracted image?
    resolved = out["image_path"].map(lambda p: os.path.basename(p) in img_basenames).sum()
    print(f"  {task:12s} rows={len(out):5d} points={len(point_cols):2d} "
          f"img_extracted={len(img_basenames):5d} resolved={resolved:5d}"
          + ("" if resolved == len(out) else "  <-- WARNING: unresolved image_paths"))
    return out


def main():
    import sys
    only = set(sys.argv[1:])  # optional: regenerate only these task_ids (e.g. "A4C PSAX")
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    print("Preparing train data ->", CSV_OUT, "and", IMG_OUT,
          (f"(only: {sorted(only)})" if only else ""))
    n = 0
    for task, (zip_rel, csv_rel, enc) in TASKS.items():
        if only and task not in only:
            continue
        basenames = extract_images(task, zip_rel)
        out = normalize_csv(task, csv_rel, enc, basenames)
        out.to_csv(CSV_OUT / f"{task}_train.csv", index=False)
        n += 1
    print("Done. Wrote", n, "normalized CSVs.")


if __name__ == "__main__":
    main()
