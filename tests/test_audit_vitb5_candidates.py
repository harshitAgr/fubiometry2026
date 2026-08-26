"""Synthetic tests for compliant ensemble prediction audits."""

from pathlib import Path

import pytest

from experiments.audit_vitb5_candidates import (  # noqa: E402
    audit,
    audit_control,
    keyed,
    mean_landmark_shift,
    snapshot_val_data,
)


def record(x, task="FA", image="x"):
    return {"image_path": image, "task_id": task,
            "predicted_points_normalized": [x / 100.0, 0.0],
            "predicted_points_pixels": [x, 0.0]}


def test_shift_and_full_audit_are_exact_and_do_not_claim_accuracy():
    reference = [record(3.0)]
    ensemble3 = [record(3.0)]
    ensemble4 = [record(4.0)]
    ensemble5 = [record(4.5)]
    shift = mean_landmark_shift(ensemble3, ensemble4)
    assert shift["overall_mean_px"] == pytest.approx(1.0)
    member45 = [record(7.0)]
    member46 = [record(9.0)]
    result = audit(reference, ensemble3, ensemble4, ensemble5, member45, member46)
    assert result["passed"]
    assert not result["selection_made"]
    assert not result["accuracy_claimed"]


def test_duplicate_or_mismatched_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        mean_landmark_shift([record(1.0), record(2.0)], [record(1.0)])
    with pytest.raises(ValueError, match="key sets"):
        mean_landmark_shift([record(1.0)], [record(1.0, image="y")])


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(predicted_points_pixels=[]), "coordinate lengths"),
    (lambda row: row.update(predicted_points_pixels=[1.0]), "coordinate lengths"),
    (lambda row: row.update(predicted_points_normalized=[]), "coordinate lengths"),
    (lambda row: row.update(predicted_points_pixels=[float("nan"), 0.0]), "finite"),
])
def test_schema_rejects_empty_odd_mismatched_or_nonfinite_coordinates(mutation, match):
    row = record(1.0)
    mutation(row)
    with pytest.raises(ValueError, match=match):
        keyed([row])


def test_control_rejects_coordinate_shape_mismatch_before_equivalence():
    reference = [record(1.0)]
    malformed = [record(1.0)]
    malformed[0]["predicted_points_pixels"] += [2.0, 3.0]
    malformed[0]["predicted_points_normalized"] += [0.02, 0.03]
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_control(reference, malformed)


def test_window9_reproducer_uses_the_matching_v15_control():
    script = Path("experiments/reproduce_vitb5_val_candidates.sh").read_text()
    assert "--window 9" in script
    assert "submission/v15/regression_predictions.json" in script
    assert "submission/v14/regression_predictions.json" not in script


def test_validation_snapshot_binds_every_file(tmp_path, monkeypatch):
    import experiments.audit_vitb5_candidates as module
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"pixels")
    (tmp_path / "task.csv").write_text("image_path\nT/a.png\n")
    monkeypatch.setattr(module, "EXPECTED_VAL_FILES", 2)
    snapshot = snapshot_val_data(tmp_path)
    assert set(snapshot) == {"images/a.png", "task.csv"}
    assert snapshot["images/a.png"]["bytes"] == 6
    assert len(snapshot["images/a.png"]["sha256"]) == 64
