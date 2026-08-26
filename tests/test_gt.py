import json
import numpy as np
import pandas as pd
import pytest
from scoring import gt

def _csv(tmp_path):
    df = pd.DataFrame([
        {"task_name": "A4C", "task_id": "A4C", "image_path": "A4C/a.jpg",
         "num_classes": 2, "point_0_xy": json.dumps([0.1, 0.2]),
         "point_1_xy": json.dumps([0.3, 0.4])},
    ])
    p = tmp_path / "gt.csv"; df.to_csv(p, index=False); return p

def test_load_gt_parses_points_normalized(tmp_path):
    rows = gt.load_gt(_csv(tmp_path))
    key = ("A4C/a.jpg", "A4C")
    assert key in rows
    assert np.allclose(rows[key], [[0.1, 0.2], [0.3, 0.4]])

def test_denormalize(tmp_path):
    pts_norm = np.array([[0.5, 0.5]])
    assert np.allclose(gt.denormalize(pts_norm, width=100, height=200), [[50, 100]])
