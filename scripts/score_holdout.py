"""Score the trained baseline on the SAME held-out split train.py used (seed 42),
using our local scorer (MRE + estimated param-MAE). No train leakage.

Run with the BASELINE venv (needs torch/timm/cv2 + numpy/pandas; scoring needs only the
latter two):
  baseline/.venv-baseline/bin/python scripts/score_holdout.py
"""
import json
import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJ, "baseline", "baseline")
RUN = os.path.join(PROJ, "runs", "baseline_v1")
sys.path.insert(0, BASE)
sys.path.insert(0, PROJ)

import torch  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

from dataset import KeypointDataset            # noqa: E402
from train import _stratified_split_indices    # noqa: E402
from model import Model                         # noqa: E402
from scoring import score as scorer             # noqa: E402

# 1. Rebuild train.py's dataframe + reproduce its seed-42 per-task val split.
ds = KeypointDataset(data_root=os.path.join(PROJ, "data"))
df = ds.dataframe.reset_index(drop=True)
_, val_idx = _stratified_split_indices(df, val_split=0.2, seed=42)
val_df = df.iloc[val_idx].reset_index(drop=True)
holdout_csv = os.path.join(PROJ, "data", "holdout_split.csv")
val_df[["image_path"]].to_csv(holdout_csv, index=False)
print(f"Holdout: {len(val_df)} imgs across {val_df['task_id'].nunique()} tasks")

# 2. Predict the holdout (best_model.pth lives in RUN; chdir so model.py finds it).
out_dir = os.path.join(PROJ, "submission", "holdout")
os.chdir(RUN)
Model().predict(data_root=os.path.join(PROJ, "data"), output_dir=out_dir,
                batch_size=8, split_csv=holdout_csv)
os.chdir(PROJ)

# 3. Score the holdout predictions against the holdout GT (pixels) directly.
#    val_df holds exactly the held-out rows with their GT point columns, so
#    total_missing should be ~0 (everything was predicted).
gt_csv = os.path.join(PROJ, "data", "holdout_gt.csv")
val_df.to_csv(gt_csv, index=False)
pred_json = os.path.join(out_dir, "regression_predictions.json")
res = scorer.score_submission(pred_json, gt_csv)
print("\n=== LOCAL HOLDOUT SCORE (seed-42 split) ===")
for tid in sorted(res["per_task"]):
    d = res["per_task"][tid]
    print(f"  {tid:12s} MRE={d['mre']:8.3f}px  paramMAE(est)={d['param_mae']:10.3f}  "
          f"missing={d['n_missing']} param_err={d['n_param_errors']}")
print(f"  AVG MRE={res['avg_mre']:.3f}px  AVG paramMAE(est)={res['avg_param_mae']:.3f}  "
      f"total_missing={res['total_missing']}")
json.dump(res, open(os.path.join(PROJ, "logs", "holdout_score.json"), "w"), indent=2)
