import json
import pandas as pd
from experiments import aggregate_cv as AC


def test_load_per_fold_reads_glob(tmp_path):
    rdir = tmp_path / "res"; rdir.mkdir()
    for k in range(2):
        json.dump({"per_task": {"AOP": {"mre": 20.0 + k, "param_mae": 5.0 + k}}},
                  open(rdir / f"cvfold{k}.json", "w"))
    folds = tmp_path / "folds.csv"
    pd.DataFrame({"task_id": ["AOP", "AOP"], "group": [0, 1], "fold": [0, 1]}).to_csv(folds, index=False)
    per_fold, n = AC.load_per_fold(str(rdir / "cvfold*.json"), str(folds))
    assert n == 2
    assert per_fold["AOP"]["mre"] == [20.0, 21.0]
    assert per_fold["AOP"]["groups"] == 2
