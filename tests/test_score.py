# tests/test_score.py
import json
import numpy as np
import pandas as pd
import pytest
from scoring import score, param_specs


def _setup(tmp_path):
    # one A4C image, 2 landmarks in PIXEL coords; one derived distance param.
    gt = pd.DataFrame([{"task_name": "A4C", "task_id": "A4C", "image_path": "A4C/a.jpg",
                        "num_classes": 2, "point_1_xy": json.dumps([0.0, 0.0]),
                        "point_2_xy": json.dumps([30.0, 40.0])}])
    gtp = tmp_path / "gt.csv"; gt.to_csv(gtp, index=False)
    pred = [{"image_path": "A4C/a.jpg", "task_id": "A4C",
             "predicted_points_normalized": [0.0, 0.0, 0.3, 0.4],
             "predicted_points_pixels": [0.0, 0.0, 30.0, 40.0]}]
    predp = tmp_path / "pred.json"; predp.write_text(json.dumps(pred))
    return gtp, predp


def test_perfect_prediction_scores_zero(tmp_path, monkeypatch):
    monkeypatch.setitem(param_specs.PARAM_SPECS, "A4C",
                        [param_specs.ParamSpec("d", "distance", (0, 1))])
    gtp, predp = _setup(tmp_path)
    res = score.score_submission(predp, gtp)
    assert res["per_task"]["A4C"]["mre"] == pytest.approx(0.0)
    assert res["per_task"]["A4C"]["param_mae"] == pytest.approx(0.0)


def test_imperfect_prediction_mre_in_pixels(tmp_path):
    # pred off by (3,4) on the second point -> per-point distances 0 and 5 -> MRE 2.5
    gt = pd.DataFrame([{"task_name": "A4C", "task_id": "A4C", "image_path": "A4C/a.jpg",
                        "num_classes": 2, "point_1_xy": json.dumps([0.0, 0.0]),
                        "point_2_xy": json.dumps([30.0, 40.0])}])
    gtp = tmp_path / "gt.csv"; gt.to_csv(gtp, index=False)
    pred = [{"image_path": "A4C/a.jpg", "task_id": "A4C",
             "predicted_points_normalized": [0.0, 0.0, 0.33, 0.44],
             "predicted_points_pixels": [0.0, 0.0, 33.0, 44.0]}]
    predp = tmp_path / "pred.json"; predp.write_text(json.dumps(pred))
    res = score.score_submission(predp, gtp)
    assert res["per_task"]["A4C"]["mre"] == pytest.approx(2.5)


def test_missing_case_is_flagged(tmp_path):
    gtp, _ = _setup(tmp_path)
    empty = tmp_path / "empty.json"; empty.write_text("[]")
    res = score.score_submission(empty, gtp)
    assert res["per_task"]["A4C"]["n_missing"] == 1


def test_degenerate_predicted_angle_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setitem(param_specs.PARAM_SPECS, "AOP",
                        [param_specs.ParamSpec("aop", "angle", (0, 1, 2))])
    gt = pd.DataFrame([{"task_name": "AOP", "task_id": "AOP", "image_path": "AOP/a.jpg",
                        "num_classes": 3,
                        "point_1_xy": json.dumps([10.0, 0.0]),
                        "point_2_xy": json.dumps([0.0, 0.0]),
                        "point_3_xy": json.dumps([0.0, 10.0])}])
    gtp = tmp_path / "gt.csv"; gt.to_csv(gtp, index=False)
    # all predicted points coincide -> angle_deg raises -> must be caught
    pred = [{"image_path": "AOP/a.jpg", "task_id": "AOP",
             "predicted_points_normalized": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
             "predicted_points_pixels": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]}]
    predp = tmp_path / "pred.json"; predp.write_text(json.dumps(pred))
    res = score.score_submission(predp, gtp)  # must NOT raise
    assert res["per_task"]["AOP"]["n_param_errors"] == 1
    assert not np.isnan(res["per_task"]["AOP"]["mre"])  # MRE still computed


def test_shape_mismatch_counts_missing(tmp_path):
    gtp, _ = _setup(tmp_path)  # A4C gt has 2 points
    pred = [{"image_path": "A4C/a.jpg", "task_id": "A4C",
             "predicted_points_normalized": [0.0, 0.0],
             "predicted_points_pixels": [0.0, 0.0]}]  # only 1 point
    predp = tmp_path / "pred.json"; predp.write_text(json.dumps(pred))
    res = score.score_submission(predp, gtp)
    assert res["per_task"]["A4C"]["n_missing"] == 1


def test_total_missing_and_estimate_flag_surfaced(tmp_path):
    gtp, _ = _setup(tmp_path)
    empty = tmp_path / "empty.json"; empty.write_text("[]")
    res = score.score_submission(empty, gtp)
    assert res["total_missing"] == 1
    assert res["param_mae_is_estimate"] is True
