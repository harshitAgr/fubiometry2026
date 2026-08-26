"""Load ground-truth landmark CSVs and handle normalized<->pixel conversion."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

POINT_COL_RE = re.compile(r"^point_(\d+)_xy$")

def load_gt(csv_path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Return {(image_path, task_id): (K,2) points AS STORED IN THE CSV}.

    For FU_Biometry these are ORIGINAL-IMAGE PIXEL coords (the baseline normalizes them at
    train time); the scorer compares them directly to predicted_points_pixels. `denormalize`
    is a separate helper for callers that hold normalized coords."""
    df = pd.read_csv(csv_path)
    pt_cols = sorted([c for c in df.columns if POINT_COL_RE.match(c)],
                     key=lambda c: int(POINT_COL_RE.match(c).group(1)))
    out: dict[tuple[str, str], np.ndarray] = {}
    for _, row in df.iterrows():
        pts = []
        for c in pt_cols:
            v = row[c]
            if pd.isna(v):
                continue
            xy = json.loads(v) if isinstance(v, str) else v
            pts.append([float(xy[0]), float(xy[1])])
        out[(row["image_path"], row["task_id"])] = np.asarray(pts, float)
    return out

def denormalize(points_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(points_norm, float) * np.array([width, height], float)[None, :]
