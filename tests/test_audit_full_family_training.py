import json

import pytest

from experiments.audit_full_family_training import expected_run, load_metrics


TASKS = ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"]


def write_metrics(path, family, mutate=None):
    epochs = 40 if family == "hcsmall" else 5
    for epoch in range(1, epochs + 1):
        if family == "hchead":
            lr = 1e-4
            tasks = ["HC"]
        else:
            if epoch == 1:
                lr = 6.66668e-6
            elif epoch == 2:
                lr = 1.333334e-5
            else:
                lr = max(0.0, 2e-5 * (40 - epoch) / 38)
            tasks = TASKS
        record = {
            "epoch": epoch,
            "train_loss": 0.1 / epoch,
            "per_task_train_loss": {task: 0.1 for task in tasks},
            "lr": lr,
            "val_avg_mre": None,
            "val_per_task_mre": None,
        }
        if mutate:
            mutate(record, epoch)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def test_expected_run_paths_are_frozen():
    assert expected_run("hcsmall", 42) == "vitb_full_hcsmall_corr"
    assert expected_run("hcsmall", 46) == "vitb_full_hcsmall_corr_s46"
    assert expected_run("hchead", 43) == "vitb_full_hchead_corr_s43"


@pytest.mark.parametrize("family", ["hcsmall", "hchead"])
def test_metrics_enforce_epochs_tasks_and_schedule(tmp_path, family):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, family)
    result = load_metrics(path, family)
    assert result["epochs"] == (40 if family == "hcsmall" else 5)


def test_metrics_reject_wrong_task_scope(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, "hchead", lambda record, _: record["per_task_train_loss"].update(FA=0.2))
    with pytest.raises(ValueError, match="wrong per-task"):
        load_metrics(path, "hchead")


def test_metrics_reject_nonconstant_refinement_lr(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, "hchead", lambda record, epoch: record.update(lr=1e-5) if epoch == 3 else None)
    with pytest.raises(ValueError, match="constant"):
        load_metrics(path, "hchead")
