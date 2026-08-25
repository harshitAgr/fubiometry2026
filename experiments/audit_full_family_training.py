"""Create or verify fail-closed provenance for full-family training runs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from experiments.preflight_full_family import (  # noqa: E402
    SOURCE_PATHS,
    file_sha256,
    snapshot_training_inputs,
)

TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}


def expected_run(family: str, seed: int) -> str:
    stem = {"hcsmall": "vitb_full_hcsmall_corr", "hchead": "vitb_full_hchead_corr"}[family]
    return stem if seed == 42 else f"{stem}_s{seed}"


def launch_receipt(family: str, seed: int, preflight: Path) -> dict:
    if seed not in (42, 43, 44, 45, 46):
        raise ValueError("seed must be one of 42..46")
    preflight_report = json.loads(preflight.read_text())
    if preflight_report.get("passed") is not True:
        raise ValueError("family preflight did not pass")
    current_sources = {path: file_sha256(PROJ / path) for path in SOURCE_PATHS}
    if current_sources != preflight_report.get("source_sha256"):
        raise ValueError("family sources differ from preflight snapshot")
    current_training_inputs = snapshot_training_inputs(PROJ / preflight_report["folds"]["path"])
    if current_training_inputs != preflight_report.get("training_inputs"):
        raise ValueError("training data differs from family preflight snapshot")
    return {
        "family": family,
        "seed": seed,
        "run_name": expected_run(family, seed),
        "preflight_path": str(preflight.relative_to(PROJ)),
        "preflight_sha256": file_sha256(preflight),
        "source_sha256": current_sources,
        "training_inputs_verified": True,
        "checkpoint_absent_before_first_launch": True,
        "recipe": (
            "fresh ViT-B, geo_v1_hcsmall, warmup3+cosine40, HM64, seed-specific"
            if family == "hcsmall" else
            "corresponding base checkpoint, HC head only, geo_v1_hcsmall, 5ep, constant 1e-4"
        ),
    }


def load_metrics(path: Path, family: str) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected_epochs = 40 if family == "hcsmall" else 5
    if [record.get("epoch") for record in records] != list(range(1, expected_epochs + 1)):
        raise ValueError(f"{path}: incomplete/noncanonical epoch sequence")
    expected_tasks = TASKS if family == "hcsmall" else {"HC"}
    lrs = []
    for record in records:
        if record.get("val_avg_mre") is not None or record.get("val_per_task_mre") is not None:
            raise ValueError(f"{path}: full-data metrics unexpectedly contain validation")
        loss = record.get("train_loss")
        lr = record.get("lr")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (loss, lr)):
            raise ValueError(f"{path}: non-finite loss/LR")
        per_task = record.get("per_task_train_loss")
        if not isinstance(per_task, dict) or set(per_task) != expected_tasks:
            raise ValueError(f"{path}: wrong per-task loss keys")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in per_task.values()):
            raise ValueError(f"{path}: non-finite per-task loss")
        lrs.append(float(lr))
    if family == "hchead":
        if any(not math.isclose(lr, 1e-4, rel_tol=0.0, abs_tol=1e-12) for lr in lrs):
            raise ValueError("HC-head refinement did not use constant 1e-4 LR")
    else:
        if not (math.isclose(lrs[0], 6.66668e-6, rel_tol=0.0, abs_tol=1e-10)
                and math.isclose(lrs[1], 1.333334e-5, rel_tol=0.0, abs_tol=1e-10)
                and max(lrs) <= 2.000001e-5
                and all(a + 1e-15 >= b for a, b in zip(lrs[2:], lrs[3:]))
                and lrs[-1] <= 1e-7):
            raise ValueError("HC-small LR history is not warmup3+cosine40 at encoder LR 2e-5")
    return {
        "sha256": file_sha256(path),
        "epochs": expected_epochs,
        "task_loss_keys": sorted(expected_tasks),
        "lr_first": lrs[0],
        "lr_last": lrs[-1],
    }


def state_scope(base: Path, refined: Path) -> dict:
    import torch

    before = torch.load(base, map_location="cpu", weights_only=True)
    after = torch.load(refined, map_location="cpu", weights_only=True)
    if before.keys() != after.keys():
        raise ValueError("base/refined state keys differ")
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    unexpected = [key for key in changed if not key.startswith("heads.HC.")]
    if len(changed) != 14 or unexpected:
        raise ValueError(
            f"expected exactly 14 HC-head tensors, got {len(changed)}; unexpected={unexpected}")
    return {"changed_tensors": changed, "n_changed_tensors": len(changed)}


def build_report(family: str, seed: int, checkpoint: Path, metrics: Path,
                 preflight: Path, base: Path | None, receipt: Path) -> dict:
    if seed not in (42, 43, 44, 45, 46):
        raise ValueError("seed must be one of 42..46")
    expected_checkpoint = PROJ / "runs" / expected_run(family, seed) / "best_model.pth"
    expected_metrics = expected_checkpoint.parent / "metrics.jsonl"
    if checkpoint.resolve() != expected_checkpoint.resolve() or metrics.resolve() != expected_metrics.resolve():
        raise ValueError("checkpoint/metrics path does not match frozen family+seed mapping")
    preflight_report = json.loads(preflight.read_text())
    if preflight_report.get("passed") is not True:
        raise ValueError("family preflight did not pass")
    current_sources = {path: file_sha256(PROJ / path) for path in SOURCE_PATHS}
    if current_sources != preflight_report.get("source_sha256"):
        raise ValueError("family sources differ from preflight snapshot")
    current_training_inputs = snapshot_training_inputs(PROJ / preflight_report["folds"]["path"])
    if current_training_inputs != preflight_report.get("training_inputs"):
        raise ValueError("training data differs from family preflight snapshot")
    expected_receipt = launch_receipt(family, seed, preflight)
    if json.loads(receipt.read_text()) != expected_receipt:
        raise ValueError("launch receipt differs from frozen family launch")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000_000:
        raise FileNotFoundError("family checkpoint is missing or incomplete")
    report = {
        "passed": True,
        "family": family,
        "seed": seed,
        "run_name": expected_run(family, seed),
        "checkpoint_path": str(checkpoint.relative_to(PROJ)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "metrics_path": str(metrics.relative_to(PROJ)),
        "metrics": load_metrics(metrics, family),
        "preflight_path": str(preflight.relative_to(PROJ)),
        "preflight_sha256": file_sha256(preflight),
        "launch_receipt_path": str(receipt.relative_to(PROJ)),
        "launch_receipt_sha256": file_sha256(receipt),
        "source_sha256": current_sources,
        "folds_csv_sha256": preflight_report["folds"]["sha256"],
        "train_rows": preflight_report["folds"]["train_rows"],
        "training_inputs_verified": True,
        "recipe": (
            "fresh ViT-B, geo_v1_hcsmall, warmup3+cosine40, HM64, seed-specific"
            if family == "hcsmall" else
            "corresponding base checkpoint, HC head only, geo_v1_hcsmall, 5ep, constant 1e-4"
        ),
        "submission_authorized": False,
    }
    if family == "hchead":
        if base is None:
            raise ValueError("HC-head audit requires corresponding base checkpoint")
        protocol = json.loads((PROJ / "experiments" / "full_family_protocol.json").read_text())
        expected_base = PROJ / protocol["base_seed_paths"][str(seed)]
        if base.resolve() != expected_base.resolve():
            raise ValueError("HC-head base path does not match corresponding frozen seed")
        if file_sha256(base) != preflight_report["base_checkpoint_sha256"][str(base.relative_to(PROJ))]:
            raise ValueError("HC-head base checkpoint differs from preflight")
        report["base_checkpoint_path"] = str(base.relative_to(PROJ))
        report["base_checkpoint_sha256"] = file_sha256(base)
        report["state_scope"] = state_scope(base, checkpoint)
    elif base is not None:
        raise ValueError("HC-small audit must not receive a base checkpoint")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("hcsmall", "hchead"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prepare-launch", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    # Callers may spell these relative to the repo root, but every path is recorded
    # PROJ-relative; canonicalize once here so relative_to(PROJ) below always holds.
    for name in ("checkpoint", "metrics", "preflight", "base", "out", "receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.prepare_launch:
        expected = launch_receipt(args.family, args.seed, args.preflight)
        expected_checkpoint = PROJ / "runs" / expected_run(args.family, args.seed) / "best_model.pth"
        if args.verify:
            if json.loads(args.receipt.read_text()) != expected:
                raise ValueError("retained launch receipt differs from current frozen launch")
        else:
            if args.receipt.exists() or expected_checkpoint.exists():
                raise FileExistsError("refusing launch receipt for an existing receipt/checkpoint")
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(json.dumps(expected, indent=2, sort_keys=True))
        return
    if args.metrics is None or args.out is None:
        parser.error("--metrics and --out are required outside --prepare-launch")
    actual = build_report(
        args.family, args.seed, args.checkpoint, args.metrics, args.preflight, args.base,
        args.receipt)
    if args.verify:
        expected = json.loads(args.out.read_text())
        if actual != expected:
            raise ValueError("family run differs from its frozen training manifest")
        print(json.dumps(actual, indent=2, sort_keys=True))
        return
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    print(json.dumps(actual, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
