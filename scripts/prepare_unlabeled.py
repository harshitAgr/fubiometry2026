"""Extract per-task unlabeled zips -> data/unlabeled/<task>/, content-dedup (perceptual ahash),
drop any image that perceptually matches a val/labeled image, write a phash manifest.

Two phases (so threshold tuning never re-hashes):
  phase 1 (slow, once): decode+ahash every unlabeled/labeled/val image in parallel -> caches:
      data/unlabeled/_pool_hashes.csv     (task_id, zip, member, phash)
      data/unlabeled/_labeled_phash.csv   (task_id, image_path, phash)   [also for Lever 2B]
      data/unlabeled/_val_phash.csv       (phash)
  phase 2 (fast, re-tunable): dedup(thresh) + exclude_near(thresh) per task -> extract kept
      images to data/unlabeled/<task>/ + write data/unlabeled/manifest.csv (task_id, image_path, phash)

Run (project venv):
  uv run python scripts/prepare_unlabeled.py --phase hash               # phase 1 (all tasks)
  uv run python scripts/prepare_unlabeled.py --phase materialize --thresh 4   # phase 2, tunable
  uv run python scripts/prepare_unlabeled.py --phase both --tasks PLAX --thresh 4   # smoke one task
Caches are reused if present (delete to force re-hash).
"""
import argparse
import glob
import io
import os
import shutil
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated frames present in the unlabeled zips

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from experiments import ssl_pool as P  # noqa: E402

RAW = os.path.join(PROJ, "data", "drive_raw")
OUT = os.path.join(PROJ, "data", "unlabeled")
TASK_DIR = {"A4C": "A4C", "AOP": "AOP", "FA": "FA", "FUGC": "FUGC", "HC": "HC",
            "IVC": "IVC", "PLAX": "PLAX", "PSAX": "PSAK", "fetal_femur": "fetal_femur"}
POOL_CACHE = os.path.join(OUT, "_pool_hashes.csv")
LAB_CACHE = os.path.join(OUT, "_labeled_phash.csv")
VAL_CACHE = os.path.join(OUT, "_val_phash.csv")


def _ahash_bytes(b: bytes):
    """Return the 64-bit ahash, or None if the image can't be decoded (corrupt/truncated)."""
    try:
        im = Image.open(io.BytesIO(b)).convert("L").resize((8, 8))
        return P.ahash(np.asarray(im, dtype=np.float64))
    except Exception:
        return None


def _hash_zip_chunk(args):
    """Worker: open one zip, hash a chunk of its members. Returns list of (task, zip, member, phash);
    corrupt/undecodable images are skipped (None)."""
    task, zpath, members = args
    out = []
    with zipfile.ZipFile(zpath) as zf:
        for m in members:
            h = _ahash_bytes(zf.read(m))
            if h is not None:
                out.append((task, os.path.basename(zpath), m, h))
    return out


def _hash_path(path: str):
    with open(path, "rb") as f:
        return _ahash_bytes(f.read())


def phase_hash(sel, workers):
    os.makedirs(OUT, exist_ok=True)
    # pool: build (task, zip, member-chunk) jobs across workers (parallelizes within big zips)
    jobs = []
    for task, d in TASK_DIR.items():
        if task not in sel:
            continue
        for z in sorted(glob.glob(os.path.join(RAW, d, "unlabeled", "*.zip"))):
            with zipfile.ZipFile(z) as zf:
                members = [n for n in zf.namelist()
                           if n.lower().endswith((".png", ".jpg", ".jpeg")) and not n.endswith("/")]
            for i in range(0, len(members), 2000):
                jobs.append((task, z, members[i:i + 2000]))
    rows = []
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_hash_zip_chunk, jobs):
            rows.extend(res)
    new = pd.DataFrame(rows, columns=["task_id", "zip", "member", "phash"])
    if sel != set(TASK_DIR) and os.path.exists(POOL_CACHE):
        # partial re-hash: keep other tasks' cached rows, replace only the selected tasks
        old = pd.read_csv(POOL_CACHE)
        new = pd.concat([old[~old.task_id.isin(sel)], new], ignore_index=True)
    new.to_csv(POOL_CACHE, index=False)
    print(f"phase hash: wrote {len(new)} pool rows ({len(rows)} from this run) -> {POOL_CACHE}")
    # labeled + val (small; serial is fine); skip any undecodable image
    lab = []
    for p in glob.glob(os.path.join(PROJ, "data", "images", "*", "*")):
        if p.lower().endswith((".png", ".jpg", ".jpeg")):
            h = _hash_path(p)
            if h is not None:
                lab.append({"task_id": os.path.basename(os.path.dirname(p)),
                            "image_path": os.path.relpath(p, PROJ), "phash": h})
    pd.DataFrame(lab).to_csv(LAB_CACHE, index=False)
    val = []
    for p in glob.glob(os.path.join(PROJ, "data", "val", "images", "*", "*")):
        if p.lower().endswith((".png", ".jpg", ".jpeg")):
            h = _hash_path(p)
            if h is not None:
                val.append({"phash": h})
    pd.DataFrame(val).to_csv(VAL_CACHE, index=False)
    print(f"phase hash: labeled={len(lab)} val={len(val)}")


def phase_materialize(sel, thresh):
    pool = pd.read_csv(POOL_CACHE)
    lab = pd.read_csv(LAB_CACHE)
    val = pd.read_csv(VAL_CACHE)
    ref = list(set(lab["phash"]).union(set(val["phash"])))
    manifest = []
    for task in TASK_DIR:
        if task not in sel:
            continue
        sub = pool[pool.task_id == task].reset_index(drop=True)
        if sub.empty:
            print(f"WARNING: {task} not in pool cache, skipping")
            continue
        hashes = sub["phash"].tolist()
        keep = P.dedup(hashes, thresh=thresh)
        keep = [keep[j] for j in P.exclude_near([hashes[i] for i in keep], ref, thresh=thresh)]
        tdir = os.path.join(OUT, task)
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)
        os.makedirs(tdir)
        # re-open zips to extract only kept members
        by_zip = {}
        for i in keep:
            r = sub.iloc[i]
            by_zip.setdefault(r["zip"], []).append((i, r["member"]))
        for zname, items in by_zip.items():
            zpath = glob.glob(os.path.join(RAW, TASK_DIR[task], "unlabeled", zname))[0]
            with zipfile.ZipFile(zpath) as zf:
                for i, m in items:
                    fn = f"{task}_{i:06d}.png"
                    Image.open(io.BytesIO(zf.read(m))).convert("RGB").save(os.path.join(tdir, fn))
                    manifest.append({"task_id": task, "image_path": f"{task}/{fn}",
                                     "phash": int(sub.iloc[i]["phash"])})
        print(f"{task:12s} pool={len(sub):6d} -> kept={sum(1 for x in manifest if x['task_id']==task):6d}")
    pd.DataFrame(manifest).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print(f"TOTAL kept: {len(manifest)} -> data/unlabeled/manifest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["hash", "materialize", "both"], default="both")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--thresh", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    sel = set(t for t in args.tasks.split(",") if t) if args.tasks else set(TASK_DIR)
    if args.phase in ("hash", "both"):
        phase_hash(sel, args.workers)
    if args.phase in ("materialize", "both"):
        phase_materialize(sel, args.thresh)


if __name__ == "__main__":
    main()
