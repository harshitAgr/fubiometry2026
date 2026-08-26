"""Build an audited coordinate-space full-data family candidate.

Within-family heatmap averaging is performed upstream by infer_ensemble.py. This
module performs only the frozen cross-family rules: exact base passthrough for IVC;
an equal three-family coordinate mean for the symmetric realization; and the
preregistered B5/H42/R42 formulas for the asymmetric hedge. It never scores,
selects, zips, or submits the output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from experiments.preflight_full_family import (  # noqa: E402
    SOURCE_PATHS,
    file_sha256,
    snapshot_training_inputs,
)
from experiments.audit_vitb5_candidates import snapshot_val_data  # noqa: E402

EXPECTED_TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}
EXPECTED_RECORDS = 619
EXPECTED_TASK_COUNTS = {
    "A4C": 20, "AOP": 60, "FA": 188, "FUGC": 20, "HC": 215,
    "IVC": 10, "PLAX": 26, "PSAX": 18, "fetal_femur": 62,
}
EXPECTED_LANDMARKS = {
    "A4C": 16, "AOP": 4, "FA": 4, "FUGC": 2, "HC": 4,
    "IVC": 2, "PLAX": 22, "PSAX": 4, "fetal_femur": 2,
}
COORDINATE_FIELDS = ("predicted_points_pixels", "predicted_points_normalized")


def keyed(records: list[dict], *, label: str, strict: bool = False) -> dict[tuple[str, str], dict]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label}: expected a nonempty record list")
    result = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label}[{index}]: expected an object")
        key = (record.get("image_path"), record.get("task_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError(f"{label}[{index}]: invalid key")
        if key in result:
            raise ValueError(f"{label}: duplicate key {key}")
        for field in COORDINATE_FIELDS:
            values = record.get(field)
            if not isinstance(values, list) or not values or len(values) % 2:
                raise ValueError(f"{label}[{index}]: invalid {field}")
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in values):
                raise ValueError(f"{label}[{index}]: non-finite {field}")
        if len(record[COORDINATE_FIELDS[0]]) != len(record[COORDINATE_FIELDS[1]]):
            raise ValueError(f"{label}[{index}]: coordinate field lengths differ")
        result[key] = record
    if strict:
        tasks = {key[1] for key in result}
        counts = {task: sum(key[1] == task for key in result) for task in sorted(tasks)}
        landmarks = {
            task: {len(record["predicted_points_pixels"]) // 2
                   for key, record in result.items() if key[1] == task}
            for task in tasks
        }
        expected_landmarks = {task: {count} for task, count in EXPECTED_LANDMARKS.items()}
        if (len(result) != EXPECTED_RECORDS or tasks != EXPECTED_TASKS
                or counts != EXPECTED_TASK_COUNTS or landmarks != expected_landmarks):
            raise ValueError(
                f"{label}: strict schema mismatch: records={len(result)}, tasks={sorted(tasks)}, "
                f"counts={counts}, landmarks={landmarks}")
        max_consistency = 0.0
        for key, record in result.items():
            image_path = PROJ / "data" / "val" / "images" / record["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"{label}: missing validation image {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            normalized = np.asarray(record["predicted_points_normalized"], dtype=np.float64).reshape(-1, 2)
            pixels = np.asarray(record["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
            residual = float(np.max(np.abs(pixels - normalized * np.asarray([width, height]))))
            max_consistency = max(max_consistency, residual)
        if max_consistency > 1e-6:
            raise ValueError(
                f"{label}: normalized/pixel coordinates disagree by {max_consistency:.3g}px")
    return result


def blend_family_records(base: list[dict], hcsmall: list[dict], hchead: list[dict],
                         *, mode: str = "symmetric_five_seed",
                         strict: bool = False) -> tuple[list[dict], dict]:
    if mode not in {"pragmatic_seed42", "symmetric_five_seed"}:
        raise ValueError(f"unknown realization mode {mode}")
    families = {
        "base": keyed(base, label="base", strict=strict),
        "hcsmall": keyed(hcsmall, label="hcsmall", strict=strict),
        "hchead": keyed(hchead, label="hchead", strict=strict),
    }
    keys = set(families["base"])
    for name, records in families.items():
        if set(records) != keys:
            raise ValueError(f"{name}: prediction keys differ from base")
        for key in keys:
            shapes = {tuple(np.asarray(records[key][field]).shape) for field in COORDINATE_FIELDS}
            if len(shapes) != 1:
                raise ValueError(f"{name}: coordinate field shape mismatch for {key}")
            base_shapes = {tuple(np.asarray(families["base"][key][field]).shape)
                           for field in COORDINATE_FIELDS}
            if shapes != base_shapes:
                raise ValueError(f"{name}: landmark shape differs from base for {key}")

    symmetric_non_hc_identity = None
    if mode == "symmetric_five_seed":
        mismatches = []
        for key in sorted(keys):
            if key[1] == "HC":
                continue
            for field in COORDINATE_FIELDS:
                if not np.array_equal(
                        np.asarray(families["base"][key][field]),
                        np.asarray(families["hchead"][key][field])):
                    mismatches.append((key, field))
        if mismatches:
            raise ValueError(
                f"symmetric hchead family diverges from base outside HC: {mismatches[:5]}")
        symmetric_non_hc_identity = True

    output = []
    ivc_exact = True
    max_formula_residual = 0.0
    shifts = {task: [] for task in EXPECTED_TASKS}
    for base_record in base:  # preserve canonical base ordering and metadata
        key = (base_record["image_path"], base_record["task_id"])
        if key[1] == "IVC":
            record = dict(base_record)
            ivc_exact &= record == base_record
        else:
            record = dict(base_record)
            for field in COORDINATE_FIELDS:
                base_array = np.asarray(families["base"][key][field], dtype=np.float64)
                small_array = np.asarray(families["hcsmall"][key][field], dtype=np.float64)
                head_array = np.asarray(families["hchead"][key][field], dtype=np.float64)
                if mode == "pragmatic_seed42" and key[1] != "HC":
                    # R42's non-HC tensors equal B42, not the lower-variance B5 family.
                    # The frozen pragmatic extrapolation therefore duplicates B5 here.
                    expected = (2.0 * base_array + small_array) / 3.0
                else:
                    expected = (base_array + small_array + head_array) / 3.0
                record[field] = expected.tolist()
                actual = np.asarray(record[field], dtype=np.float64)
                max_formula_residual = max(
                    max_formula_residual, float(np.max(np.abs(actual - expected))))
            base_px = np.asarray(base_record["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
            out_px = np.asarray(record["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
            shifts[key[1]].extend(np.linalg.norm(out_px - base_px, axis=1).tolist())
        output.append(record)
    audit = {
        "passed": bool(ivc_exact and max_formula_residual == 0.0),
        "records": len(output),
        "tasks": sorted({record["task_id"] for record in output}),
        "realization": mode,
        "routes": {
            "IVC": "exact base passthrough",
            "HC": "(base + hcsmall + hchead) / 3",
            "other_tasks": (
                "(2 * base + hcsmall) / 3" if mode == "pragmatic_seed42"
                else "(base + hcsmall + hchead) / 3"
            ),
        },
        "ivc_route": "exact_base_passthrough",
        "ivc_record_values_exact": ivc_exact,
        "max_coordinate_formula_residual": max_formula_residual,
        "symmetric_hchead_equals_base_outside_hc": symmetric_non_hc_identity,
        "per_task_mean_shift_from_base_px": {
            task: (float(np.mean(values)) if values else 0.0)
            for task, values in sorted(shifts.items())
        },
        "accuracy_claimed": False,
        "selection_made": False,
        "submission_authorized": False,
    }
    return output, audit


def verify_preflight(path: Path, mode: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("passed") is not True or report.get("realization") != mode:
        raise ValueError("preflight report does not authorize this realization")
    actual_sources = {source: file_sha256(PROJ / source) for source in SOURCE_PATHS}
    if actual_sources != report.get("source_sha256"):
        raise ValueError("audited family sources changed after preflight")
    current_training = snapshot_training_inputs(PROJ / report["folds"]["path"])
    if current_training != report.get("training_inputs"):
        raise ValueError("training inputs changed after family preflight")
    if snapshot_val_data() != report.get("validation_file_sha256"):
        raise ValueError("validation inputs changed after family preflight")
    return report


def verify_training_manifests(paths: list[Path], mode: str) -> dict:
    from experiments.audit_full_family_training import build_report

    expected_keys = (
        {("hcsmall", 42), ("hchead", 42)}
        if mode == "pragmatic_seed42" else
        {(family, seed) for family in ("hcsmall", "hchead") for seed in (42, 43, 44, 45, 46)}
    )
    if len(paths) != len(expected_keys):
        raise ValueError(f"expected {len(expected_keys)} training manifests for {mode}")
    result = {}
    seen = set()
    loaded = []
    for path in paths:
        manifest = json.loads(path.read_text())
        key = (manifest.get("family"), manifest.get("seed"))
        if manifest.get("passed") is not True or key in seen:
            raise ValueError(f"invalid/duplicate training manifest {path}")
        seen.add(key)
        loaded.append((path, manifest, key))
    if seen != expected_keys:
        raise ValueError(f"training manifest set differs: expected={expected_keys}, got={seen}")
    for path, manifest, key in loaded:
        checkpoint = PROJ / manifest["checkpoint_path"]
        metrics = PROJ / manifest["metrics_path"]
        preflight = PROJ / manifest["preflight_path"]
        receipt = PROJ / manifest["launch_receipt_path"]
        base = (PROJ / manifest["base_checkpoint_path"] if key[0] == "hchead" else None)
        actual = build_report(
            key[0], key[1], checkpoint, metrics, preflight, base, receipt)
        if actual != manifest:
            raise ValueError(f"training manifest does not rederive exactly: {path}")
        result[str(path)] = {
            "manifest_sha256": file_sha256(path),
            "family": key[0],
            "seed": key[1],
            "checkpoint_path": manifest["checkpoint_path"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pragmatic_seed42", "symmetric_five_seed"), required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hcsmall", type=Path, required=True)
    parser.add_argument("--hchead", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, action="append", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and (args.out.exists() or args.audit_out.exists()):
        raise FileExistsError("refusing to overwrite candidate or audit")
    preflight = verify_preflight(args.preflight_manifest, args.mode)
    training = verify_training_manifests(args.training_manifest, args.mode)
    records = [json.loads(path.read_text()) for path in (args.base, args.hcsmall, args.hchead)]
    output, audit = blend_family_records(*records, mode=args.mode, strict=True)
    input_hashes = {
        "base": file_sha256(args.base),
        "hcsmall": file_sha256(args.hcsmall),
        "hchead": file_sha256(args.hchead),
    }
    def expected_audit(output_hash: str) -> dict:
        result = dict(audit)
        result.update({
            "preflight_manifest": str(args.preflight_manifest),
            "preflight_manifest_sha256": file_sha256(args.preflight_manifest),
            "input_sha256": input_hashes,
            "training_manifests": training,
            "output_sha256": output_hash,
            "preflight_base_prediction_sha256": preflight["base_prediction_sha256"],
        })
        return result

    if args.verify:
        actual_output = json.loads(args.out.read_text())
        retained_audit = json.loads(args.audit_out.read_text())
        if actual_output != output:
            raise ValueError("retained candidate differs from frozen family formula")
        expected = expected_audit(file_sha256(args.out))
        if retained_audit != expected:
            raise ValueError("retained candidate audit does not exactly rederive")
        if input_hashes["base"] != preflight["base_prediction_sha256"]:
            raise ValueError("retained base-family prediction differs from preflight snapshot")
        print(json.dumps({"passed": True, "verified_existing": True}, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output) + "\n")
    audit = expected_audit(file_sha256(args.out))
    if audit["input_sha256"]["base"] != preflight["base_prediction_sha256"]:
        args.out.unlink()
        raise ValueError("base-family prediction differs from preflight snapshot")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
