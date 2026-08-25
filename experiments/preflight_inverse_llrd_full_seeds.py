"""Same-data falsification gate for the post-hoc inverse-LLRD hypothesis.

This is CPU-only and must be run only after full-data seeds 42--46 exist.  The
fixed gate was preregistered in commit 3fe7973 before seeds 45/46 were inspected.
Passing can justify a controlled inverse-schedule experiment; it is not evidence
that inverse LLRD improves validation performance.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from experiments.encoders import build_encoder
from experiments.make_folds_core import EXPECT_PER_TASK
from experiments.preflight_llrd import (
    analyze_layer_updates,
    file_sha256,
    load_encoder_checkpoint,
    state_sha256,
)


RUN_SPECS = (
    (42, "vitb_full_corr", "ens_vitb_corr_gpu0.log"),
    (43, "vitb_full_corr_s43", "ens_vitb_corr_gpu1.log"),
    (44, "vitb_full_corr_s44", "ens_vitb_corr_gpu0.log"),
    (45, "vitb_full_corr_s45", "ens_vitb_corr_gpu1_s45_s46.log"),
    (46, "vitb_full_corr_s46", "ens_vitb_corr_gpu1_s45_s46.log"),
)
OUT = PROJ / "experiments" / "results" / "inverse_llrd_full_seeds" / "report.json"
TASKS = {"A4C", "AOP", "FA", "fetal_femur", "FUGC", "HC", "IVC", "PLAX", "PSAX"}
EXPECTED_TRAIN_PER_TASK = {**EXPECT_PER_TASK, "AOP": EXPECT_PER_TASK["AOP"] - 16}
EXPECTED_GUARD_PER_TASK = {"AOP": 16}
SOURCE_PATHS = (
    "experiments/preflight_inverse_llrd_full_seeds.py",
    "experiments/preflight_llrd.py",
    "experiments/reproduce_inverse_llrd_full_seeds.sh",
    "experiments/train_ensemble_vitb_corr.sh",
    "experiments/run_config.py",
    "experiments/encoders.py",
    "experiments/make_folds_core.py",
    "tests/test_inverse_llrd_full_seeds.py",
    "baseline/baseline/model_factory.py",
)


def path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJ))
    except ValueError:
        return str(path.resolve())


def validate_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [row.get("epoch") for row in rows]
    if epochs != list(range(1, 41)):
        raise ValueError(f"{path}: expected epochs exactly 1..40, got {epochs}")
    for row in rows:
        losses = row.get("per_task_train_loss")
        if not isinstance(losses, dict) or set(losses) != TASKS:
            raise ValueError(f"{path}: invalid per-task losses at epoch {row.get('epoch')}")
        numeric = [row.get("train_loss"), row.get("lr"), *losses.values()]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in numeric):
            raise ValueError(f"{path}: non-finite metric at epoch {row.get('epoch')}")
        if row.get("val_avg_mre") is not None or row.get("val_per_task_mre") is not None:
            raise ValueError(f"{path}: expected a full-data run without validation metrics")
    return {
        "path": path_label(path),
        "sha256": file_sha256(path),
        "epochs": epochs,
        "task_ids": sorted(TASKS),
    }


def validate_text_log(path: Path, required: list[str]) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"{path}: missing required log evidence: {missing}")
    return {
        "path": path_label(path),
        "sha256": file_sha256(path),
        "required_evidence": required,
    }


def validate_folds(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6743:
        raise ValueError(f"{path}: expected 6743 rows, got {len(rows)}")
    if set(rows[0]) != {"image_path", "task_id", "fold", "group"}:
        raise ValueError(f"{path}: unexpected columns {list(rows[0])}")
    train_rows = [row for row in rows if int(row["fold"]) >= 0]
    dropped_rows = [row for row in rows if int(row["fold"]) == -1]
    if len(train_rows) != 6727 or len(dropped_rows) != 16:
        raise ValueError(
            f"{path}: expected 6727 train/16 guard-drop rows, got "
            f"{len(train_rows)}/{len(dropped_rows)}")
    all_counts = dict(Counter(row["task_id"] for row in rows))
    train_counts = dict(Counter(row["task_id"] for row in train_rows))
    guard_counts = dict(Counter(row["task_id"] for row in dropped_rows))
    if all_counts != EXPECT_PER_TASK:
        raise ValueError(f"{path}: full per-task counts differ: {all_counts}")
    if train_counts != EXPECTED_TRAIN_PER_TASK:
        raise ValueError(f"{path}: trained per-task counts differ: {train_counts}")
    if guard_counts != EXPECTED_GUARD_PER_TASK:
        raise ValueError(f"{path}: guard per-task counts differ: {guard_counts}")
    return {
        "path": path_label(path),
        "sha256": file_sha256(path),
        "total_rows": len(rows),
        "fold_nonnegative_training_rows": len(train_rows),
        "fold_minus_one_guard_rows": len(dropped_rows),
        "all_per_task_counts": dict(sorted(all_counts.items())),
        "trained_per_task_counts": dict(sorted(train_counts.items())),
        "guard_per_task_counts": dict(sorted(guard_counts.items())),
    }


def evaluate_inverse_gate(geometry: dict[str, Any]) -> dict[str, Any]:
    """Apply only the fixed same-data inverse-LLRD mechanism gates."""
    blocks = geometry["blocks"]
    if len(blocks) != 12:
        raise ValueError(f"expected 12 blocks, got {len(blocks)}")
    summaries: dict[str, Any] = {}
    gate: dict[str, bool] = {}
    for kind in ("raw", "debiased"):
        drift_key = f"{kind}_median_relative_drift"
        cosine_key = f"{kind}_median_pairwise_update_cosine"
        bottom_drift = float(np.median([row[drift_key] for row in blocks[:6]]))
        top_drift = float(np.median([row[drift_key] for row in blocks[6:]]))
        bottom_cosines = [row[cosine_key] for row in blocks[:6]]
        top_cosines = [row[cosine_key] for row in blocks[6:]]
        geometry_valid = all(value is not None for value in bottom_cosines + top_cosines)
        if not geometry_valid:
            bottom_consistency = None
            top_consistency = None
            consistency_gap = None
            top_below_bottom_median = 0
        else:
            bottom_consistency = float(np.median(bottom_cosines))
            top_consistency = float(np.median(top_cosines))
            consistency_gap = bottom_consistency - top_consistency
            top_below_bottom_median = sum(
                value < bottom_consistency for value in top_cosines)
        top_to_bottom_drift = top_drift / bottom_drift if bottom_drift > 0 else 0.0
        summaries[kind] = {
            "bottom6_median_relative_drift": bottom_drift,
            "top6_median_relative_drift": top_drift,
            "top_to_bottom_drift_ratio": top_to_bottom_drift,
            "bottom6_median_update_consistency": bottom_consistency,
            "top6_median_update_consistency": top_consistency,
            "bottom_minus_top_consistency_gap": consistency_gap,
            "top_blocks_below_bottom_consistency_median": top_below_bottom_median,
            "geometry_valid": geometry_valid,
        }
        gate[f"{kind}_bottom_minus_top_consistency_gap_at_least_0_10"] = bool(
            geometry_valid and consistency_gap is not None and consistency_gap >= 0.10)
        gate[f"{kind}_top_to_bottom_drift_ratio_at_least_0_25"] = bool(
            top_to_bottom_drift >= 0.25)
        gate[f"{kind}_at_least_5_of_6_top_blocks_below_bottom_median"] = bool(
            geometry_valid and top_below_bottom_median >= 5)
    return {
        "definitions": {
            "bottom_blocks": "ViT blocks 0--5",
            "top_blocks": "ViT blocks 6--11",
            "consistency_gap": "bottom-six median consistency minus top-six median consistency",
            "block_count": "top blocks with consistency strictly below the bottom-block median",
        },
        "summary": summaries,
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def main() -> None:
    torch.set_num_threads(1)
    # Freeze audit-time source bytes before reading any experimental artifact.
    # This prevents a concurrent workspace edit from being attested as code that
    # the already-running interpreter actually executed.
    source_hashes = {relative: file_sha256(PROJ / relative) for relative in SOURCE_PATHS}
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite inverse-LLRD audit: {args.out}")

    checkpoint_paths = [
        (PROJ / "runs" / run_name / "best_model.pth").resolve()
        for _, run_name, _ in RUN_SPECS
    ]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError("canonical checkpoint paths are not unique")
    training_runs = []
    for seed, run_name, outer_log_name in RUN_SPECS:
        checkpoint = (PROJ / "runs" / run_name / "best_model.pth").resolve()
        metrics = (PROJ / "runs" / run_name / "metrics.jsonl").resolve()
        run_log = (PROJ / "logs" / f"{run_name}.log").resolve()
        outer_log = (PROJ / "logs" / outer_log_name).resolve()
        for path in (checkpoint, metrics, run_log, outer_log):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics_evidence = validate_metrics(metrics)
        run_log_evidence = validate_text_log(run_log, [
            "[full-data] training on ALL 6727 images (no held-out val)",
            "fold -1 epoch 40/40 done",
            f"runs/{run_name}/best_model.pth",
        ])
        outer_log_evidence = validate_text_log(outer_log, [
            f"training ViT-B ensemble member seed={seed} -> runs/{run_name}",
            f"DONE seed={seed} -> runs/{run_name}/best_model.pth",
            "ALL CORRECTED ViT-B ENSEMBLE MEMBERS DONE",
        ])
        training_runs.append({
            "seed": seed,
            "run_name": run_name,
            "checkpoint_path": str(checkpoint.relative_to(PROJ)),
            "metrics": metrics_evidence,
            "run_log": run_log_evidence,
            "outer_launch_log": outer_log_evidence,
        })
    folds_evidence = validate_folds(PROJ / "data" / "folds" / "folds.csv")

    checkpoint_hashes = [file_sha256(path) for path in checkpoint_paths]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("canonical checkpoints contain duplicate file hashes")

    encoder = build_encoder("dinov2_vitb", input_size=518)
    initial = {f"encoder.{key}": value.detach().cpu()
               for key, value in encoder.state_dict().items()}
    del encoder
    gc.collect()
    checkpoints = []
    for path in checkpoint_paths:
        checkpoints.append(load_encoder_checkpoint(path))
        gc.collect()
    checkpoint_encoder_hashes = [state_sha256(state) for state in checkpoints]
    if len(set(checkpoint_encoder_hashes)) != len(checkpoint_encoder_hashes):
        raise ValueError("canonical checkpoints contain duplicate encoder-state hashes")

    geometry = analyze_layer_updates(initial, checkpoints)
    # These are the already-rejected standard-LLRD gates from the shared geometry
    # helper, not the decision target of this independently preregistered audit.
    geometry.pop("gate")
    geometry.pop("gate_passed")
    inverse = evaluate_inverse_gate(geometry)
    relative_paths = [str(path.relative_to(PROJ)) for path in checkpoint_paths]
    report = {
        "protocol": "preregistered same-data full-seed inverse-LLRD mechanism falsification",
        "preregistration_commit": "3fe7973",
        "decision_scope": (
            "pass may only justify a fixed inverse-schedule controlled experiment; "
            "failure closes depth scheduling"
        ),
        "seed_order": [seed for seed, _, _ in RUN_SPECS],
        "internal_geometry_label_mapping": {
            f"fold{index}": f"seed{seed}"
            for index, (seed, _, _) in enumerate(RUN_SPECS)
        },
        "pretrained_encoder": "timm vit_base_patch14_dinov2.lvd142m",
        "pretrained_encoder_state_sha256": state_sha256(initial),
        "checkpoint_order": relative_paths,
        "checkpoint_sha256": {
            relative: digest
            for relative, digest in zip(relative_paths, checkpoint_hashes)
        },
        "checkpoint_encoder_state_sha256": {
            relative: digest
            for relative, digest in zip(relative_paths, checkpoint_encoder_hashes)
        },
        "training_provenance": {
            "runs": training_runs,
            "folds": folds_evidence,
        },
        "geometry": geometry,
        "inverse_hypothesis": inverse,
        "source_sha256_snapshotted_before_artifact_reads": source_hashes,
        "versions": {
            "torch": torch.__version__,
            "timm": __import__("timm").__version__,
            "numpy": np.__version__,
        },
    }
    report["passed"] = inverse["gate_passed"]
    source_hashes_after = {relative: file_sha256(PROJ / relative) for relative in SOURCE_PATHS}
    if source_hashes_after != source_hashes:
        raise RuntimeError("audited source files changed during inverse-LLRD preflight")
    report["source_sha256_verified_before_write"] = source_hashes_after
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
