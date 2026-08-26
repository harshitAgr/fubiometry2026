from experiments import reporting as R


def test_metric_table_builds_per_task_rows_with_deltas():
    base = {"per_task": {"AOP": {"mre": {"mean": 20.0}, "param_mae": {"mean": 5.0}}}}
    new = {"per_task": {"AOP": {"mre": {"mean": 18.0}, "param_mae": {"mean": 4.0}}}}
    df = R.metric_table(new, baseline=base)
    row = df[df["task"] == "AOP"].iloc[0]
    assert abs(row["MRE"] - 18.0) < 1e-9
    assert abs(row["dMRE"] - (-2.0)) < 1e-9
    assert abs(row["dparamMAE"] - (-1.0)) < 1e-9


def test_metric_table_without_baseline_has_no_delta_columns():
    new = {"per_task": {"AOP": {"mre": {"mean": 18.0}, "param_mae": {"mean": 4.0}}}}
    df = R.metric_table(new)
    assert "dMRE" not in df.columns and "MRE" in df.columns
