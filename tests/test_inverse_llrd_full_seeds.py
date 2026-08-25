import csv
import json

import pytest

torch = pytest.importorskip("torch")

from experiments.preflight_inverse_llrd_full_seeds import (
    EXPECT_PER_TASK,
    EXPECTED_GUARD_PER_TASK,
    EXPECTED_TRAIN_PER_TASK,
    evaluate_inverse_gate,
    validate_folds,
    validate_metrics,
    validate_text_log,
)
from experiments.preflight_llrd import analyze_layer_updates


def _states(top_signs=(1, -1, 1, -1, 1), top_scale=0.5):
    initial = {
        "encoder.backbone.cls_token": torch.ones(2),
        "encoder.backbone.pos_embed": torch.ones(2),
        "encoder.backbone.patch_embed.weight": torch.ones(2),
        "encoder.backbone.norm.weight": torch.ones(2),
    }
    for block in range(12):
        initial[f"encoder.backbone.blocks.{block}.weight"] = torch.ones(2)
    checkpoints = []
    for fit in range(5):
        state = {}
        for key, value in initial.items():
            if ".blocks." in key:
                block = int(key.split(".blocks.")[1].split(".")[0])
                if block < 6:
                    direction = torch.tensor([1.0, 0.0])
                    scale = 1.0
                else:
                    direction = torch.tensor([1.0, 0.0]) * top_signs[fit]
                    scale = top_scale
            else:
                direction = torch.tensor([1.0, 0.0])
                scale = 1.0
            state[key] = value + direction * scale * (1 + 0.01 * fit)
        checkpoints.append(state)
    return initial, checkpoints


def _geometry(bottom_cosine=0.5, top_cosines=None, bottom_drift=1.0, top_drift=0.25):
    if top_cosines is None:
        top_cosines = [0.0] * 6
    blocks = []
    for index in range(12):
        cosine = bottom_cosine if index < 6 else top_cosines[index - 6]
        drift = bottom_drift if index < 6 else top_drift
        blocks.append({
            "raw_median_relative_drift": drift,
            "raw_median_pairwise_update_cosine": cosine,
            "debiased_median_relative_drift": drift,
            "debiased_median_pairwise_update_cosine": cosine,
        })
    return {"blocks": blocks}


def test_inverse_gate_passes_material_inconsistent_top_updates():
    initial, checkpoints = _states()
    inverse = evaluate_inverse_gate(analyze_layer_updates(initial, checkpoints))
    assert inverse["gate_passed"]
    for kind in ("raw", "debiased"):
        summary = inverse["summary"][kind]
        assert summary["bottom_minus_top_consistency_gap"] >= 0.10
        assert summary["top_to_bottom_drift_ratio"] >= 0.25
        assert summary["top_blocks_below_bottom_consistency_median"] == 6


def test_inverse_gate_rejects_consistent_top_updates():
    initial, checkpoints = _states(top_signs=(1, 1, 1, 1, 1))
    inverse = evaluate_inverse_gate(analyze_layer_updates(initial, checkpoints))
    assert not inverse["gate_passed"]
    for kind in ("raw", "debiased"):
        assert inverse["summary"][kind]["bottom_minus_top_consistency_gap"] < 0.10


def test_inverse_gate_rejects_tiny_top_drift():
    initial, checkpoints = _states(top_scale=0.1)
    inverse = evaluate_inverse_gate(analyze_layer_updates(initial, checkpoints))
    assert not inverse["gate_passed"]
    for kind in ("raw", "debiased"):
        assert inverse["summary"][kind]["top_to_bottom_drift_ratio"] < 0.25


def test_inverse_gate_accepts_exact_boundaries_and_exactly_five_blocks():
    inverse = evaluate_inverse_gate(_geometry(
        bottom_cosine=0.2,
        top_cosines=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
        bottom_drift=1.0,
        top_drift=0.25,
    ))
    assert inverse["gate_passed"]
    for kind in ("raw", "debiased"):
        summary = inverse["summary"][kind]
        assert summary["bottom_minus_top_consistency_gap"] == pytest.approx(0.10)
        assert summary["top_to_bottom_drift_ratio"] == pytest.approx(0.25)
        assert summary["top_blocks_below_bottom_consistency_median"] == 5


def test_inverse_gate_rejects_four_strictly_lower_blocks():
    inverse = evaluate_inverse_gate(_geometry(
        bottom_cosine=0.5,
        top_cosines=[-1.0, -1.0, -1.0, -1.0, 0.5, 0.5],
    ))
    assert inverse["summary"]["raw"]["bottom_minus_top_consistency_gap"] >= 0.10
    assert inverse["summary"]["raw"]["top_blocks_below_bottom_consistency_median"] == 4
    assert not inverse["gate_passed"]


def test_inverse_gate_requires_raw_and_debiased_predicates():
    geometry = _geometry()
    for row in geometry["blocks"][6:]:
        row["debiased_median_pairwise_update_cosine"] = 0.5
    inverse = evaluate_inverse_gate(geometry)
    assert all(value for key, value in inverse["gate"].items() if key.startswith("raw_"))
    assert not all(value for key, value in inverse["gate"].items()
                   if key.startswith("debiased_"))
    assert not inverse["gate_passed"]


def test_inverse_gate_rejects_invalid_debiased_geometry():
    geometry = _geometry()
    geometry["blocks"][6]["debiased_median_pairwise_update_cosine"] = None
    inverse = evaluate_inverse_gate(geometry)
    assert not inverse["summary"]["debiased"]["geometry_valid"]
    assert not inverse["gate_passed"]


def test_inverse_gate_rejects_wrong_block_count():
    with pytest.raises(ValueError, match="expected 12 blocks"):
        evaluate_inverse_gate({"blocks": _geometry()["blocks"][:-1]})


def _metric_rows():
    tasks = ["A4C", "AOP", "FA", "fetal_femur", "FUGC", "HC", "IVC", "PLAX", "PSAX"]
    return [{
        "epoch": epoch,
        "train_loss": 1.0 / epoch,
        "per_task_train_loss": {task: 1.0 / epoch for task in tasks},
        "lr": 0.0,
        "val_avg_mre": None,
        "val_per_task_mre": None,
    } for epoch in range(1, 41)]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_metrics_accepts_exact_full_run(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(path, _metric_rows())
    report = validate_metrics(path)
    assert report["epochs"] == list(range(1, 41))


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(3), "expected epochs exactly"),
    (lambda rows: rows[0]["per_task_train_loss"].pop("HC"), "invalid per-task losses"),
    (lambda rows: rows[0].update(train_loss=float("nan")), "non-finite metric"),
    (lambda rows: rows[0].update(val_avg_mre=1.0), "without validation metrics"),
])
def test_validate_metrics_rejects_malformed_runs(tmp_path, mutation, match):
    rows = _metric_rows()
    mutation(rows)
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=match):
        validate_metrics(path)


def test_validate_text_log_requires_all_evidence(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("alpha\nbeta\n")
    report = validate_text_log(path, ["alpha", "beta"])
    assert report["required_evidence"] == ["alpha", "beta"]
    with pytest.raises(ValueError, match="missing required log evidence"):
        validate_text_log(path, ["alpha", "gamma"])


def _write_folds(path, *, corrupt_counts=False):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "task_id", "fold", "group"])
        writer.writeheader()
        index = 0
        for task, count in EXPECTED_TRAIN_PER_TASK.items():
            for task_index in range(count):
                written_task = "A4C" if corrupt_counts and task == "HC" and task_index == 0 else task
                writer.writerow({
                    "image_path": f"{written_task}/{index}.png",
                    "task_id": written_task,
                    "fold": 0,
                    "group": f"{written_task}:{index}",
                })
                index += 1
        for task, count in EXPECTED_GUARD_PER_TASK.items():
            for _ in range(count):
                writer.writerow({
                    "image_path": f"{task}/{index}.png",
                    "task_id": task,
                    "fold": -1,
                    "group": f"{task}:{index}",
                })
                index += 1


def test_validate_folds_accepts_exact_canonical_counts(tmp_path):
    path = tmp_path / "folds.csv"
    _write_folds(path)
    report = validate_folds(path)
    assert report["all_per_task_counts"] == dict(sorted(EXPECT_PER_TASK.items()))
    assert report["trained_per_task_counts"] == dict(sorted(EXPECTED_TRAIN_PER_TASK.items()))
    assert report["guard_per_task_counts"] == dict(sorted(EXPECTED_GUARD_PER_TASK.items()))


def test_validate_folds_rejects_noncanonical_task_distribution(tmp_path):
    path = tmp_path / "folds.csv"
    _write_folds(path, corrupt_counts=True)
    with pytest.raises(ValueError, match="per-task counts differ"):
        validate_folds(path)
