"""Audit prediction-space convergence of compliant 3/4/5-member ensembles.

This deliberately does not score or select a candidate: official validation GT is
unavailable locally.  It verifies the regenerated three-member control exactly and
quantifies the incremental prediction shifts introduced by seeds 45 and 46.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
EXPECTED_TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}
EXPECTED_RECORDS = 619
SOURCE_PATHS = (
    "experiments/audit_vitb5_candidates.py",
    "experiments/infer_ensemble.py",
    "experiments/reproduce_vitb5_val_candidates.sh",
    "experiments/decode.py",
    "experiments/infer_tta.py",
    "experiments/encoders.py",
    "experiments/per_task_model.py",
    "experiments/fugc_scale.py",
    "experiments/hc_scale_norm.py",
    "baseline/baseline/model.py",
    "baseline/baseline/model_factory.py",
    "baseline/baseline/utils.py",
    "tests/test_audit_vitb5_candidates.py",
)
VAL_DATA_ROOT = PROJ / "data" / "val"
EXPECTED_VAL_FILES = 628  # 619 images + one metadata CSV for each of nine tasks


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_val_data(root=VAL_DATA_ROOT):
    files = sorted(path for path in Path(root).rglob("*") if path.is_file())
    if len(files) != EXPECTED_VAL_FILES:
        raise ValueError(
            f"expected {EXPECTED_VAL_FILES} validation files, got {len(files)} under {root}")
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    }


def coordinate_equivalence(actual, reference, atol=1e-12, pixel_atol=5e-9):
    """Pure-NumPy equivalence check; keep this audit runnable in the project venv."""
    left = keyed(actual, label="actual control")
    right = keyed(reference, label="reference control")
    missing = sorted(set(right) - set(left))
    extra = sorted(set(left) - set(right))
    max_normalized = 0.0
    max_pixels = 0.0
    length_mismatches = []
    if not missing and not extra:
        for key in sorted(left):
            normalized_a = np.asarray(
                left[key]["predicted_points_normalized"], dtype=np.float64)
            normalized_b = np.asarray(
                right[key]["predicted_points_normalized"], dtype=np.float64)
            pixels_a = np.asarray(left[key]["predicted_points_pixels"], dtype=np.float64)
            pixels_b = np.asarray(right[key]["predicted_points_pixels"], dtype=np.float64)
            if normalized_a.shape != normalized_b.shape or pixels_a.shape != pixels_b.shape:
                length_mismatches.append(key)
                continue
            max_normalized = max(
                max_normalized, float(np.max(np.abs(normalized_a - normalized_b))))
            max_pixels = max(max_pixels, float(np.max(np.abs(pixels_a - pixels_b))))
    passed = (not missing and not extra and not length_mismatches
              and max_normalized <= atol and max_pixels <= pixel_atol)
    return {
        "passed": passed,
        "atol": atol,
        "pixel_atol": pixel_atol,
        "max_normalized_coordinate_abs_difference": max_normalized,
        "max_pixel_coordinate_abs_difference": max_pixels,
        "missing_keys": missing,
        "extra_keys": extra,
        "length_mismatch_keys": length_mismatches,
    }


def keyed(records, *, label="records", strict_reference=False):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label}: expected a nonempty record list")
    result = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label}[{index}]: expected an object")
        image_path, task_id = record.get("image_path"), record.get("task_id")
        if not isinstance(image_path, str) or not image_path or not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label}[{index}]: invalid image_path/task_id")
        key = (image_path, task_id)
        if key in result:
            raise ValueError(f"{label}: prediction file contains duplicate image/task keys")
        pixels = record.get("predicted_points_pixels")
        normalized = record.get("predicted_points_normalized")
        if (not isinstance(pixels, list) or not pixels or len(pixels) % 2
                or not isinstance(normalized, list) or len(normalized) != len(pixels)):
            raise ValueError(f"{label}[{index}]: invalid coordinate lengths")
        if not all(not isinstance(value, bool) and isinstance(value, (int, float))
                   and math.isfinite(value)
                   for value in pixels + normalized):
            raise ValueError(f"{label}[{index}]: coordinates must be finite numbers")
        result[key] = record
    if len(result) != len(records):
        raise ValueError(f"{label}: prediction file contains duplicate image/task keys")
    if strict_reference:
        tasks = {key[1] for key in result}
        if len(records) != EXPECTED_RECORDS or tasks != EXPECTED_TASKS:
            raise ValueError(
                f"{label}: expected {EXPECTED_RECORDS} records and tasks "
                f"{sorted(EXPECTED_TASKS)}, got {len(records)} and {sorted(tasks)}")
    return result


def mean_landmark_shift(records_a, records_b):
    left = keyed(records_a, label="left predictions")
    right = keyed(records_b, label="right predictions")
    if set(left) != set(right):
        raise ValueError("prediction key sets differ")
    by_task = {}
    all_distances = []
    for key in sorted(left):
        a = np.asarray(left[key]["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(right[key]["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        if a.shape != b.shape:
            raise ValueError(f"landmark shape mismatch for {key}")
        distances = np.linalg.norm(a - b, axis=1)
        by_task.setdefault(key[1], []).extend(distances.tolist())
        all_distances.extend(distances.tolist())
    return {
        "overall_mean_px": float(np.mean(all_distances)),
        "overall_p95_px": float(np.percentile(all_distances, 95)),
        "overall_max_px": float(np.max(all_distances)),
        "per_task_mean_px": {
            task: float(np.mean(values)) for task, values in sorted(by_task.items())
        },
    }


def audit_control(reference, ensemble3, *, strict_reference=False):
    expected = keyed(reference, label="reference", strict_reference=strict_reference)
    control = keyed(ensemble3, label="ensemble3")
    if set(expected) != set(control):
        raise ValueError("reference and ensemble3 prediction key sets differ")
    for key in expected:
        if len(expected[key]["predicted_points_pixels"]) != len(
                control[key]["predicted_points_pixels"]):
            raise ValueError(f"coordinate shape mismatch for {key}")
    control_equivalence = coordinate_equivalence(ensemble3, reference)
    if not control_equivalence["passed"]:
        raise ValueError(f"regenerated ensemble3 differs from v15 window-9 control: {control_equivalence}")
    return control_equivalence


def audit(reference, ensemble3, ensemble4, ensemble5, member45, member46,
          *, strict_reference=False):
    control_equivalence = audit_control(
        reference, ensemble3, strict_reference=strict_reference)
    anchor = keyed(reference, label="reference", strict_reference=strict_reference)
    for label, records in (
            ("ensemble4", ensemble4), ("ensemble5", ensemble5),
            ("member45", member45), ("member46", member46)):
        candidate = keyed(records, label=label)
        if set(candidate) != set(anchor):
            raise ValueError(f"{label}: prediction key sets differ from reference")
        for key in anchor:
            if len(candidate[key]["predicted_points_pixels"]) != len(
                    anchor[key]["predicted_points_pixels"]):
                raise ValueError(f"{label}: landmark shape mismatch for {key}")
    return {
        "passed": True,
        "selection_made": False,
        "accuracy_claimed": False,
        "control_equivalence": control_equivalence,
        "ensemble_shifts": {
            "3_to_4": mean_landmark_shift(ensemble3, ensemble4),
            "4_to_5": mean_landmark_shift(ensemble4, ensemble5),
            "3_to_5": mean_landmark_shift(ensemble3, ensemble5),
        },
        "new_member_diversity": {
            "seed45_vs_ensemble3": mean_landmark_shift(member45, ensemble3),
            "seed46_vs_ensemble4": mean_landmark_shift(member46, ensemble4),
        },
        "new_member_measurement": "direct single-checkpoint inference; no nonlinear reconstruction",
        "interpretation": (
            "Prediction-space convergence only; without local GT this audit cannot choose "
            "between 3, 4, and 5 members or establish an accuracy gain."
        ),
    }


def verify_checkpoint_provenance(report_path, checkpoints):
    report = json.loads(Path(report_path).read_text())
    recorded = report.get("checkpoint_sha256", {})
    actual = {str(Path(path).resolve().relative_to(PROJ)): file_sha256(path)
              for path in checkpoints}
    if actual != {path: recorded.get(path) for path in actual}:
        raise ValueError("checkpoint hashes differ from inverse-audit provenance")
    return actual


def snapshot_manifest(reference, checkpoints, checkpoint_provenance):
    return {
        "reference_path": str(Path(reference)),
        "reference_sha256": file_sha256(reference),
        "checkpoint_sha256": verify_checkpoint_provenance(
            checkpoint_provenance, checkpoints),
        "checkpoint_provenance_path": str(Path(checkpoint_provenance)),
        "checkpoint_provenance_sha256": file_sha256(checkpoint_provenance),
        "source_sha256": {path: file_sha256(PROJ / path) for path in SOURCE_PATHS},
        "validation_data_root": str(VAL_DATA_ROOT.relative_to(PROJ)),
        "validation_file_sha256": snapshot_val_data(),
    }


def verify_manifest(manifest_path, reference, checkpoints, checkpoint_provenance):
    expected = json.loads(Path(manifest_path).read_text())
    actual = snapshot_manifest(reference, checkpoints, checkpoint_provenance)
    if actual != expected:
        raise ValueError("candidate inputs or audited sources changed after preflight snapshot")
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ensemble3")
    parser.add_argument("--ensemble4")
    parser.add_argument("--ensemble5")
    parser.add_argument("--member45")
    parser.add_argument("--member46")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    parser.add_argument("--preflight-manifest")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoints = [path for path in args.checkpoints.split(",") if path]
    if len(checkpoints) != 5:
        parser.error("--checkpoints must list the five canonical members")
    if args.snapshot_only:
        if args.control_only or args.preflight_manifest:
            parser.error("snapshot mode cannot be combined with control/verification mode")
        result = snapshot_manifest(args.reference, checkpoints, args.checkpoint_provenance)
    else:
        if not args.preflight_manifest:
            parser.error("--preflight-manifest is required outside snapshot mode")
        manifest = verify_manifest(
            args.preflight_manifest, args.reference, checkpoints, args.checkpoint_provenance)
        if not args.ensemble3:
            parser.error("--ensemble3 is required")
        with open(args.reference) as handle:
            reference = json.load(handle)
        with open(args.ensemble3) as handle:
            ensemble3 = json.load(handle)
        if args.control_only:
            equivalence = audit_control(reference, ensemble3, strict_reference=True)
            result = {"passed": True, "control_equivalence": equivalence,
                      "preflight_manifest": manifest}
        else:
            required = (args.ensemble4, args.ensemble5, args.member45, args.member46)
            if any(path is None for path in required):
                parser.error("final audit requires ensemble4/ensemble5/member45/member46")
            records = [reference, ensemble3]
            for path in required:
                with open(path) as handle:
                    records.append(json.load(handle))
            result = audit(*records, strict_reference=True)
            result["preflight_manifest"] = manifest
            result["prediction_sha256"] = {
                "reference": file_sha256(args.reference),
                "ensemble3": file_sha256(args.ensemble3),
                "ensemble4": file_sha256(args.ensemble4),
                "ensemble5": file_sha256(args.ensemble5),
                "member45": file_sha256(args.member45),
                "member46": file_sha256(args.member46),
            }
    source_hashes_after = {path: file_sha256(PROJ / path) for path in SOURCE_PATHS}
    expected_sources = result.get("source_sha256", result.get("preflight_manifest", {}).get("source_sha256"))
    if source_hashes_after != expected_sources:
        raise RuntimeError("audited candidate sources changed during audit")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
