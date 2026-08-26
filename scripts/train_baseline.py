"""Launch the official baseline training (good-citizen GPU sharing).

Runs the cloned baseline's train.py against the prepared data/ with:
  - a per-process GPU memory cap so we never OOM a co-running job on a shared GPU,
  - a run dir (runs/baseline_v1) holding a `data` symlink (train.py hardcodes
    DATA_ROOT_PATH="data") and the output best_model.pth.

Run from the project root with the BASELINE venv:
  baseline/.venv-baseline/bin/python scripts/train_baseline.py
"""
import os
import runpy
import sys

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJ, "baseline", "baseline")
RUN = os.path.join(PROJ, "runs", "baseline_v1")

os.makedirs(RUN, exist_ok=True)
link = os.path.join(RUN, "data")
if not os.path.lexists(link):
    os.symlink(os.path.join(PROJ, "data"), link)

# Cap our share of the shared GPU (~27 GB of 98) so a co-running job is never OOM'd.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.28, 0)
    print(f"GPU mem fraction capped at 0.28; device={torch.cuda.get_device_name(0)}")

sys.path.insert(0, BASE)
os.chdir(RUN)                 # DATA_ROOT_PATH="data" -> runs/baseline_v1/data symlink
sys.argv = ["train.py"]       # defaults: 40 epochs, per-task val_split 0.2, seed 42
print(f"Running baseline train.py from {RUN} ...")
runpy.run_path(os.path.join(BASE, "train.py"), run_name="__main__")
