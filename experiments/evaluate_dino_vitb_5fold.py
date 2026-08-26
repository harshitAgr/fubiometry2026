"""Fail-closed deployment gate for the continued-DINO ViT-B five-fold experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.aggregate import corrected_resampled_t_ci

TASKS = ("A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur")
N_TRAIN = 5400
N_TEST = 1350


def deployment_decision(deltas, treatment_mean: float, baseline: float = 23.0,
                        policy: str = "strict") -> dict:
    deltas = [float(value) for value in deltas]
    if len(deltas) != 5 or not np.isfinite(deltas).all() or not np.isfinite(treatment_mean):
        raise ValueError("deployment decision requires five finite paired deltas and a finite mean")
    mean, low, high = corrected_resampled_t_ci(deltas, N_TRAIN, N_TEST)
    improved = sum(value < 0.0 for value in deltas)
    if policy not in {"strict", "expected_score"}:
        raise ValueError(f"unknown deployment policy: {policy}")
    checks = {
        "treatment_below_23": treatment_mean < baseline,
        "at_least_4_of_5_folds_improved": improved >= 4,
        "corrected_ci_upper_below_zero": high < 0.0,
    }
    strict_deploy = all(checks.values())
    deploy = strict_deploy if policy == "strict" else (
        checks["treatment_below_23"] and checks["at_least_4_of_5_folds_improved"]
    )
    return {
        "policy": policy,
        "deploy": deploy,
        "strict_deploy": strict_deploy,
        "checks": checks,
        "paired_delta_mean": mean,
        "paired_delta_corrected_ci95": [low, high],
        "improved_folds": improved,
    }


def _load_score(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    if set(result.get("per_task", {})) != set(TASKS):
        raise ValueError(f"{path}: expected exactly nine task scores")
    if result.get("total_missing", 0) != 0:
        raise ValueError(f"{path}: missing predictions are not allowed")
    task_mean = float(np.mean([result["per_task"][task]["mre"] for task in TASKS]))
    if not np.isclose(task_mean, result.get("avg_mre"), atol=1e-9, rtol=0.0):
        raise ValueError(f"{path}: avg_mre differs from the nine-task mean")
    return {"raw": result, "task_mean": task_mean}


def evaluate(control_dir: Path, treatment_dir: Path, policy: str = "strict") -> dict:
    folds = []
    task_deltas = {task: [] for task in TASKS}
    for fold in range(5):
        control_path = control_dir / f"cvfold{fold}_postdrop.json"
        treatment_path = treatment_dir / f"cvfold{fold}.json"
        control = _load_score(control_path)
        treatment = _load_score(treatment_path)
        delta = treatment["task_mean"] - control["task_mean"]
        folds.append({
            "fold": fold,
            "control_mre": control["task_mean"],
            "treatment_mre": treatment["task_mean"],
            "delta": delta,
        })
        for task in TASKS:
            task_deltas[task].append(
                treatment["raw"]["per_task"][task]["mre"]
                - control["raw"]["per_task"][task]["mre"]
            )
    control_mean = float(np.mean([fold["control_mre"] for fold in folds]))
    treatment_mean = float(np.mean([fold["treatment_mre"] for fold in folds]))
    gate = deployment_decision([fold["delta"] for fold in folds], treatment_mean,
                               baseline=control_mean, policy=policy)
    return {
        "protocol": "continued_dino_vitb_paired_5fold_deployment_gate_v2",
        "baseline": "geo_cosine40_vitb post-femur-drop re-score",
        "control_mean_mre": control_mean,
        "treatment_mean_mre": treatment_mean,
        "folds": folds,
        "per_task_mean_delta": {
            task: float(np.mean(values)) for task, values in task_deltas.items()
        },
        **gate,
        "decision_basis": (
            "The expected-score policy selects the lower five-fold mean when at least four folds "
            "improve; it does not claim that the corrected confidence interval excludes zero."
            if policy == "expected_score" else
            "The strict policy requires the corrected confidence interval to exclude zero."
        ),
        "caveat": (
            "Control weights trained before the invalid femur frames were dropped, so part of the "
            "observed gain may be due to the corrected supervised training pool."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path,
                        default=Path("experiments/results/geo_cosine40_vitb"))
    parser.add_argument("--treatment-dir", type=Path,
                        default=Path("experiments/results/dino_ssl_vitb_5fold"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/dino_ssl_vitb_5fold/decision.json"))
    parser.add_argument("--policy", choices=("strict", "expected_score"), default="strict")
    args = parser.parse_args()
    report = evaluate(args.control_dir, args.treatment_dir, policy=args.policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
