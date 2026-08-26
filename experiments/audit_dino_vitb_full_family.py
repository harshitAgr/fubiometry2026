"""Audit continued-DINO full-data checkpoints without running validation or Docker inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}
ENCODER_INIT = Path("runs/dino_ssl_vitb/encoder.pth")
EXPECTED = {
    "base_s42": ("runs/vitb_full_dino_corr", 42, 40, 6727, TASKS),
    "base_s43": ("runs/vitb_full_dino_corr_s43", 43, 40, 6727, TASKS),
    "base_s44": ("runs/vitb_full_dino_corr_s44", 44, 40, 6727, TASKS),
    "base_s45": ("runs/vitb_full_dino_corr_s45", 45, 40, 6727, TASKS),
    "base_s46": ("runs/vitb_full_dino_corr_s46", 46, 40, 6727, TASKS),
    "hcsmall_s42": ("runs/vitb_full_dino_hcsmall_corr", 42, 40, 6727, TASKS),
    "hchead_s42": ("runs/vitb_full_dino_hchead_corr", 42, 5, 999, {"HC"}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(role: str, expected_encoder_sha: str, base42: Path) -> tuple[dict, dict]:
    run_rel, seed, epochs, train_examples, task_keys = EXPECTED[role]
    run = PROJ / run_rel
    checkpoint = run / "best_model.pth"
    metrics_path = run / "metrics.jsonl"
    manifest_path = run / "training_manifest.json"
    for path in (checkpoint, metrics_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["mode"] != "full_data" or manifest["full_data"] is not True:
        raise ValueError(f"{role}: not a full-data run")
    if manifest["seed"] != seed or manifest["epochs"] != epochs:
        raise ValueError(f"{role}: seed/epoch mismatch")
    if manifest["train_examples"] != train_examples:
        raise ValueError(f"{role}: expected {train_examples} training examples")
    if manifest["encoder"] != "dinov2_vitb" or manifest["input_size"] != 518:
        raise ValueError(f"{role}: wrong encoder/input size")
    if manifest["heatmap_size"] != [64, 64]:
        raise ValueError(f"{role}: heatmap size is not uniform 64")
    if role == "hchead_s42":
        if manifest["train_task"] != "HC" or manifest["head_lr"] != 1e-4:
            raise ValueError("HC-head refinement scope/learning rate mismatch")
        if manifest["init_checkpoint"] != str(base42.relative_to(PROJ)):
            raise ValueError("HC-head did not initialize from continued-DINO base seed 42")
        if manifest["init_checkpoint_sha256"] != sha256(base42):
            raise ValueError("HC-head base checkpoint hash mismatch")
    else:
        if manifest["encoder_init"] != str(ENCODER_INIT):
            raise ValueError(f"{role}: continued-DINO encoder path missing")
        if manifest["encoder_init_sha256"] != expected_encoder_sha:
            raise ValueError(f"{role}: continued-DINO encoder hash mismatch")
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if [record["epoch"] for record in metrics] != list(range(1, epochs + 1)):
        raise ValueError(f"{role}: incomplete epoch sequence")
    for record in metrics:
        if record["val_avg_mre"] is not None or record["val_per_task_mre"] is not None:
            raise ValueError(f"{role}: validation was unexpectedly run")
        if set(record["per_task_train_loss"]) != task_keys:
            raise ValueError(f"{role}: task-loss keys mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    nonfinite = [name for name, value in state.items()
                 if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"{role}: non-finite checkpoint tensors: {nonfinite[:5]}")
    return ({
        "role": role,
        "run": run_rel,
        "seed": seed,
        "epochs": epochs,
        "train_examples": train_examples,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "metrics_sha256": sha256(metrics_path),
        "manifest_sha256": sha256(manifest_path),
        "validation_run": False,
    }, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/dino_ssl_vitb_full_family/audit.json"))
    args = parser.parse_args()
    encoder_path = PROJ / ENCODER_INIT
    encoder_sha = sha256(encoder_path)
    base42 = PROJ / EXPECTED["base_s42"][0] / "best_model.pth"
    reports = []
    comparison_states = {}
    reference_keys = None
    for role in EXPECTED:
        report, state = audit_run(role, encoder_sha, base42)
        if reference_keys is None:
            reference_keys = set(state)
        elif set(state) != reference_keys:
            raise ValueError(f"{role}: checkpoint state keys differ from base")
        reports.append(report)
        if role in {"base_s42", "hchead_s42"}:
            comparison_states[role] = state
    before, after = comparison_states["base_s42"], comparison_states["hchead_s42"]
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if len(changed) != 14 or any(not name.startswith("heads.HC.") for name in changed):
        raise ValueError(f"HC-head refinement changed unexpected tensors: {changed}")
    checkpoint_hashes = [report["checkpoint_sha256"] for report in reports]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("full-family checkpoints are not all distinct")
    report = {
        "passed": True,
        "protocol": "continued_dino_vitb_full_family_no_val_v1",
        "encoder_init": str(ENCODER_INIT),
        "encoder_init_sha256": encoder_sha,
        "runs": reports,
        "hchead_changed_tensors": changed,
        "docker_modified": False,
        "docker_built": False,
        "validation_inference_run": False,
        "submission_made": False,
    }
    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
