import csv

import pytest

from experiments.preflight_full_family import SOURCE_PATHS, persist_report, validate_folds


def protocol(rows=3, train=2, dropped=1):
    return {
        "data": {
            "expected_rows": rows,
            "expected_train_rows": train,
            "expected_guard_drop_rows": dropped,
            "allowed_train_folds": [0, 1, 2, 3, 4],
            "train_task_counts": {"FA": train},
        }
    }


def write_folds(path, values):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "task_id", "fold"])
        writer.writeheader()
        for index, fold in enumerate(values):
            writer.writerow({"image_path": f"T/{index}.png", "task_id": "FA", "fold": fold})


def test_validate_folds_counts_and_hashes(tmp_path, monkeypatch):
    path = tmp_path / "folds.csv"
    write_folds(path, [0, 4, -1])
    import experiments.preflight_full_family as module
    monkeypatch.setattr(module, "PROJ", tmp_path)
    result = validate_folds(path, protocol())
    assert result["rows"] == 3
    assert result["train_rows"] == 2
    assert result["guard_drop_rows"] == 1
    assert len(result["sha256"]) == 64


def test_validate_folds_rejects_duplicates_and_unexpected_labels(tmp_path, monkeypatch):
    import experiments.preflight_full_family as module
    monkeypatch.setattr(module, "PROJ", tmp_path)
    path = tmp_path / "folds.csv"
    write_folds(path, [0, -2, -1])
    with pytest.raises(ValueError, match="unexpected"):
        validate_folds(path, protocol())
    path.write_text(
        "image_path,task_id,fold\nT/a.png,FA,0\nT/a.png,FA,1\nT/b.png,FA,-1\n")
    with pytest.raises(ValueError, match="duplicate"):
        validate_folds(path, protocol())


def test_source_closure_contains_training_and_inference_dependencies():
    required = {
        "experiments/kp_aug_dataset.py",
        "baseline/baseline/dataset.py",
        "experiments/fugc_scale.py",
        "experiments/hc_scale_norm.py",
        "experiments/audit_vitb5_candidates.py",
        "experiments/audit_full_family_training.py",
    }
    assert required <= set(SOURCE_PATHS)


@pytest.mark.parametrize("name", [
    "experiments/reproduce_full_family_seed42.sh",
    "experiments/reproduce_full_family_symmetric.sh",
])
def test_long_launcher_self_detaches_and_verifies_reused_checkpoints(name):
    text = open(name).read()
    assert "setsid -f nohup" in text
    assert '"$PPID" -ne 1' in text
    assert '"$SID" != "$$"' in text
    assert "VERIFY=(--verify)" in text
    assert "--prepare-launch --verify" in text
    assert "scratch_tmp/full_family.lock" in text
    assert "_launch.json" in text
    assert "--verify-existing" in text
    assert "preflight_manifest.json" not in text  # chain is centralized in the Python preflight


def test_retained_preflight_path_is_actually_verified(tmp_path):
    path = tmp_path / "preflight.json"
    report = {"passed": True, "hash": "a" * 64}
    persist_report(report, path)
    persist_report(report, path, verify_existing=True)
    with pytest.raises(ValueError, match="differs"):
        persist_report({"passed": True, "hash": "b" * 64}, path, verify_existing=True)
    with pytest.raises(FileNotFoundError, match="missing"):
        persist_report(report, tmp_path / "absent.json", verify_existing=True)
