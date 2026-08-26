"""Mechanical preflight for the post-drop full-data family hedge.

This snapshots the exact protocol, training/inference sources, five compliant base
members, fold membership, cached CV evidence, and the five-member base prediction.
It does not load a GPU, train, score, select, zip, or submit anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from experiments.audit_vitb5_candidates import verify_manifest  # noqa: E402
from experiments.audit_vitb5_candidates import snapshot_val_data  # noqa: E402

PROTOCOL = PROJ / "experiments" / "full_family_protocol.json"
SOURCE_PATHS = (
    "experiments/full_family_protocol.json",
    "experiments/preflight_full_family.py",
    "experiments/full_family_candidate.py",
    "experiments/audit_full_family_training.py",
    "experiments/audit_vitb5_candidates.py",
    "experiments/reproduce_full_family_seed42.sh",
    "experiments/reproduce_full_family_symmetric.sh",
    "experiments/run_config.py",
    "experiments/augment.py",
    "experiments/encoders.py",
    "experiments/per_task_model.py",
    "experiments/kp_aug_dataset.py",
    "experiments/infer_ensemble.py",
    "experiments/infer_tta.py",
    "experiments/decode.py",
    "experiments/fugc_scale.py",
    "experiments/hc_scale_norm.py",
    "experiments/audit_head_refinement.py",
    "baseline/baseline/model.py",
    "baseline/baseline/model_factory.py",
    "baseline/baseline/dataset.py",
    "baseline/baseline/utils.py",
    "tests/test_full_family_candidate.py",
    "tests/test_preflight_full_family.py",
    "tests/test_audit_full_family_training.py",
    "tests/test_audit_vitb5_candidates.py",
)
EXPECTED_BASE_PREDICTION = (
    PROJ / "submission" / "vitb5_val_candidates" / "ensemble5"
    / "regression_predictions.json"
)
INVERSE_REPORT = PROJ / "experiments" / "results" / "inverse_llrd_full_seeds" / "report.json"
CONVERGENCE_DIR = PROJ / "experiments" / "results" / "vitb5_val_candidates"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not stable:
        raise RuntimeError(f"input changed while hashing: {path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def snapshot_training_inputs(folds_path: Path) -> dict:
    folds_path = folds_path.resolve()
    with open(folds_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["fold"]) in (0, 1, 2, 3, 4)]
    image_paths = sorted(PROJ / "data" / "images" / row["image_path"] for row in selected)
    if len(image_paths) != 6727 or len(set(image_paths)) != 6727:
        raise ValueError("expected 6,727 unique selected training images")
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} selected training images; first={missing[0]}")
    csv_paths = sorted((PROJ / "data" / "csv").glob("*.csv"))
    if len(csv_paths) != 12:
        raise ValueError(f"expected 12 dataset CSV inputs, got {len(csv_paths)}")
    return {
        "folds_csv": file_record(folds_path, PROJ),
        "dataset_csv": [file_record(path, PROJ) for path in csv_paths],
        "selected_images": [file_record(path, PROJ) for path in image_paths],
        "selected_image_count": len(image_paths),
        "dataset_csv_count": len(csv_paths),
    }


def validate_folds(path: Path, protocol: dict) -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = protocol["data"]
    if len(rows) != expected["expected_rows"]:
        raise ValueError(f"expected {expected['expected_rows']} fold rows, got {len(rows)}")
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("folds.csv contains duplicate image_path values")
    folds = [int(row["fold"]) for row in rows]
    train = sum(fold in expected["allowed_train_folds"] for fold in folds)
    dropped = sum(fold == -1 for fold in folds)
    unexpected = sorted(set(folds) - set(expected["allowed_train_folds"]) - {-1})
    if unexpected:
        raise ValueError(f"unexpected fold labels: {unexpected}")
    if train != expected["expected_train_rows"] or dropped != expected["expected_guard_drop_rows"]:
        raise ValueError(
            f"expected {expected['expected_train_rows']} train/"
            f"{expected['expected_guard_drop_rows']} drop rows, got {train}/{dropped}")
    task_counts = Counter(row["task_id"] for row, fold in zip(rows, folds)
                          if fold in expected["allowed_train_folds"])
    if dict(sorted(task_counts.items())) != expected["train_task_counts"]:
        raise ValueError(
            f"training task counts differ: {dict(sorted(task_counts.items()))}")
    return {
        "path": str(path.relative_to(PROJ)),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "train_rows": train,
        "guard_drop_rows": dropped,
        "train_task_counts": dict(sorted(task_counts.items())),
    }


def validate_metrics(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [record.get("epoch") for record in records]
    if epochs != list(range(1, 41)):
        raise ValueError(f"{path}: expected exact epochs 1..40, got {epochs}")
    for record in records:
        loss = record.get("train_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError(f"{path}: non-finite train loss")
        if record.get("val_avg_mre") is not None:
            raise ValueError(f"{path}: full-data record unexpectedly contains validation")
    return {"path": str(path.relative_to(PROJ)), "sha256": file_sha256(path), "epochs": epochs}


def snapshot(mode: str) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if mode not in protocol["realizations"]:
        raise ValueError(f"unknown realization {mode}")
    source_before = {path: file_sha256(PROJ / path) for path in SOURCE_PATHS}
    folds_path = PROJ / protocol["data"]["folds_csv"]
    folds = validate_folds(folds_path, protocol)
    training_inputs = snapshot_training_inputs(folds_path)
    base_checkpoints = {}
    base_metrics = {}
    for seed, relative in protocol["base_seed_paths"].items():
        checkpoint = PROJ / relative
        if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000_000:
            raise FileNotFoundError(f"missing/incomplete base checkpoint: {checkpoint}")
        base_checkpoints[relative] = file_sha256(checkpoint)
        metrics = checkpoint.parent / "metrics.jsonl"
        base_metrics[str(metrics.relative_to(PROJ))] = validate_metrics(metrics)
    convergence_manifest_path = CONVERGENCE_DIR / "preflight_manifest.json"
    convergence_audit_path = CONVERGENCE_DIR / "prediction_audit.json"
    control_audit_path = CONVERGENCE_DIR / "control_equivalence.json"
    required_chain = (INVERSE_REPORT, convergence_manifest_path, convergence_audit_path,
                      control_audit_path, EXPECTED_BASE_PREDICTION)
    missing_chain = [str(path.relative_to(PROJ)) for path in required_chain if not path.is_file()]
    if missing_chain:
        raise FileNotFoundError(
            "canonical inverse/convergence chain is incomplete: " + ", ".join(missing_chain))
    base_paths = [PROJ / protocol["base_seed_paths"][str(seed)] for seed in (42, 43, 44, 45, 46)]
    # snapshot_manifest records reference/provenance paths verbatim, so they must be spelled
    # exactly as reproduce_vitb5_val_candidates.sh recorded them: PROJ-relative, not absolute.
    convergence_manifest = verify_manifest(
        convergence_manifest_path,
        Path("submission/v15/regression_predictions.json"),
        base_paths,
        INVERSE_REPORT.relative_to(PROJ),
    )
    convergence_audit = json.loads(convergence_audit_path.read_text())
    control_audit = json.loads(control_audit_path.read_text())
    if convergence_audit.get("passed") is not True:
        raise ValueError("five-seed prediction audit did not pass")
    if control_audit.get("passed") is not True:
        raise ValueError("three-seed control equivalence audit did not pass")
    if convergence_audit.get("preflight_manifest") != convergence_manifest:
        raise ValueError("prediction audit embeds a different convergence preflight")
    ensemble5_hash = convergence_audit.get("prediction_sha256", {}).get("ensemble5")
    if ensemble5_hash != file_sha256(EXPECTED_BASE_PREDICTION):
        raise ValueError("ensemble5 prediction differs from canonical prediction audit")
    evidence_path = PROJ / protocol["evidence"]["cached_cv_report"]
    evidence = json.loads(evidence_path.read_text())["summary"]
    def exact_float(actual, expected_value):
        return math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-15)

    checks = {
        "mre_delta": math.isclose(
            evidence["paired_mean_mre_delta"], protocol["evidence"]["mre_delta"],
            rel_tol=0.0, abs_tol=1e-15),
        "mre_folds": evidence["favorable_mre_folds"] == protocol["evidence"]["favorable_mre_folds"],
        "mre_ci": (
            len(evidence["paired_mre_delta_corrected_ci95"]) == 2
            and len(protocol["evidence"]["mre_corrected_ci95"]) == 2
            and all(exact_float(actual, expected_value) for actual, expected_value in zip(
                evidence["paired_mre_delta_corrected_ci95"], protocol["evidence"]["mre_corrected_ci95"]))
        ),
        "param_delta": exact_float(
            evidence["paired_mean_param_mae_delta"],
            protocol["evidence"]["parameter_mae_delta"]),
        "param_folds": evidence["favorable_param_mae_folds"] == protocol["evidence"]["favorable_parameter_folds"],
        "param_ci": (
            len(evidence["paired_param_mae_delta_corrected_ci95"]) == 2
            and len(protocol["evidence"]["parameter_corrected_ci95"]) == 2
            and all(exact_float(actual, expected_value) for actual, expected_value in zip(
                evidence["paired_param_mae_delta_corrected_ci95"],
                protocol["evidence"]["parameter_corrected_ci95"]))
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"cached CV evidence differs from frozen protocol: {checks}")
    source_after = {path: file_sha256(PROJ / path) for path in SOURCE_PATHS}
    if source_after != source_before:
        raise RuntimeError("family sources changed during preflight")
    return {
        "passed": True,
        "realization": mode,
        "selection_made": False,
        "accuracy_claimed": False,
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_before,
        "folds": folds,
        "training_inputs": training_inputs,
        "base_checkpoint_sha256": base_checkpoints,
        "base_metrics": base_metrics,
        "base_prediction_path": str(EXPECTED_BASE_PREDICTION.relative_to(PROJ)),
        "base_prediction_sha256": file_sha256(EXPECTED_BASE_PREDICTION),
        "inverse_provenance_path": str(INVERSE_REPORT.relative_to(PROJ)),
        "inverse_provenance_sha256": file_sha256(INVERSE_REPORT),
        "convergence_preflight_path": str(convergence_manifest_path.relative_to(PROJ)),
        "convergence_preflight_sha256": file_sha256(convergence_manifest_path),
        "convergence_prediction_audit_path": str(convergence_audit_path.relative_to(PROJ)),
        "convergence_prediction_audit_sha256": file_sha256(convergence_audit_path),
        "convergence_control_audit_path": str(control_audit_path.relative_to(PROJ)),
        "convergence_control_audit_sha256": file_sha256(control_audit_path),
        "validation_file_sha256": convergence_manifest["validation_file_sha256"],
        "cached_cv_report": str(evidence_path.relative_to(PROJ)),
        "cached_cv_report_sha256": file_sha256(evidence_path),
        "cached_evidence_checks": checks,
        "interpretation": protocol["realizations"][mode]["interpretation"],
        "submission_authorized": False,
    }


def persist_report(report: dict, output: Path, *, verify_existing: bool = False) -> None:
    if output.exists():
        if not verify_existing:
            raise FileExistsError(f"refusing to overwrite {output}")
        if json.loads(output.read_text()) != report:
            raise ValueError("retained preflight differs from the current exact snapshot")
        return
    if verify_existing:
        raise FileNotFoundError(f"cannot verify missing retained preflight {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pragmatic_seed42", "symmetric_five_seed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    report = snapshot(args.mode)
    persist_report(report, args.out, verify_existing=args.verify_existing)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
