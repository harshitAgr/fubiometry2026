import numpy as np, pytest
from experiments import aggregate as A

def test_corrected_ci_wider_than_naive():
    scores = [10.0, 12.0, 11.0, 9.0, 13.0]  # 5 fold scores
    mean, lo, hi = A.corrected_resampled_t_ci(scores, n_train=80, n_test=20)
    naive_half = 1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores))
    corrected_half = (hi - lo) / 2
    assert mean == pytest.approx(11.0)
    assert corrected_half > naive_half  # Nadeau-Bengio correction inflates the interval

def test_aggregate_reports_per_task_and_pooled_cardiac():
    per_fold = {
        "IVC":  {"mre": [60, 64, 62], "param_mae": [30, 32, 31], "groups": 38},
        "A4C":  {"mre": [40, 42, 41], "param_mae": [20, 22, 21], "groups": 85},
        "AOP":  {"mre": [70, 72, 71], "param_mae": [50, 52, 51], "groups": 0},
    }
    res = A.aggregate(per_fold, n_train=80, n_test=20)
    assert "IVC" in res["per_task"] and "ci" in res["per_task"]["IVC"]["mre"]
    assert "cardiac" in res["pooled"]            # pooled-cardiac selection unit present
    assert res["per_task"]["IVC"]["groups"] == 38
