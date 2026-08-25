"""Compare old and continued-DINO Docker routes on five OOF folds.

This does not use official validation data. Checkpoint-family prediction directories are explicit,
so the treatment may replace only the base or the complete base/HC-small/HC-head family. Both arms
always use the same route mode and identical Docker post-processing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from docker.geometry_project import project  # noqa: E402
from docker.hc_scale import apply_hc_scale  # noqa: E402
from docker.ivc_calibrate import apply_ivc_calibration  # noqa: E402
from experiments.aggregate import corrected_resampled_t_ci  # noqa: E402
from scoring.gt import load_gt  # noqa: E402
from scoring.score import score_submission  # noqa: E402

TASKS = ("A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur")
FIELDS = ("predicted_points_pixels", "predicted_points_normalized")


def keyed(path: Path) -> dict[tuple[str, str], dict]:
    rows = json.loads(path.read_text())
    result = {(row["image_path"], row["task_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction key in {path}")
    return result


def build_route(base_path: Path, hcsmall_path: Path, hchead_path: Path,
                gt_path: Path, route_mode: str = "pragmatic") -> tuple[list[dict], dict]:
    if route_mode not in {"pragmatic", "hc_only", "base_only"}:
        raise ValueError(f"unknown route mode: {route_mode}")
    families = {
        "base": keyed(base_path),
        "hcsmall": keyed(hcsmall_path),
        "hchead": keyed(hchead_path),
    }
    required = set(load_gt(gt_path))
    for name, records in families.items():
        if not required.issubset(records):
            raise ValueError(
                f"{name}: key mismatch: missing={len(required-set(records))}, "
                f"extra={len(set(records)-required)}")

    output = []
    counts = {"hc_scaled": 0, "projected": 0, "ivc_calibrated": 0}
    for key in sorted(required):
        base = families["base"][key]
        task = key[1]
        if task == "IVC" or route_mode == "base_only":
            pixels = np.asarray(base["predicted_points_pixels"], dtype=float)
        elif task == "HC":
            pixels = np.mean([
                np.asarray(families[name][key]["predicted_points_pixels"], dtype=float)
                for name in ("base", "hcsmall", "hchead")
            ], axis=0)
        elif route_mode == "pragmatic":
            pixels = (
                2.0 * np.asarray(base["predicted_points_pixels"], dtype=float)
                + np.asarray(families["hcsmall"][key]["predicted_points_pixels"], dtype=float)
            ) / 3.0
        else:
            pixels = np.asarray(base["predicted_points_pixels"], dtype=float)

        image_path = PROJ / "data" / "images" / base["image_path"]
        with Image.open(image_path) as image:
            width, height = image.size
        before = pixels.tolist()
        scaled = apply_hc_scale(before, task, width, height)
        counts["hc_scaled"] += scaled != before
        projected = project(scaled, task)
        counts["projected"] += projected != scaled
        final = apply_ivc_calibration(projected, task)
        counts["ivc_calibrated"] += final != projected
        final_array = np.asarray(final, dtype=float).reshape(-1, 2)
        if not np.isfinite(final_array).all():
            raise ValueError(f"non-finite route output at {key}")
        row = dict(base)
        row["predicted_points_pixels"] = final_array.reshape(-1).tolist()
        row["predicted_points_normalized"] = (
            final_array / np.asarray([width, height], dtype=float)
        ).reshape(-1).tolist()
        output.append(row)
    return output, counts


def task_mean(result: dict, metric: str) -> float:
    return float(np.mean([result["per_task"][task][metric] for task in TASKS]))


def ci(values: list[float]) -> list[float]:
    _, low, high = corrected_resampled_t_ci(values, n_train=5400, n_test=1350)
    return [float(low), float(high)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment-dir", type=Path,
                        default=Path("submission/dino_ssl_vitb_5fold_window9"))
    parser.add_argument("--control-hcsmall-dir", type=Path,
                        default=Path("submission/hcsmall"))
    parser.add_argument("--control-hchead-dir", type=Path,
                        default=Path("submission/hchead"))
    parser.add_argument("--treatment-hcsmall-dir", type=Path,
                        default=Path("submission/hcsmall"))
    parser.add_argument("--treatment-hchead-dir", type=Path,
                        default=Path("submission/hchead"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("experiments/results/dino_ssl_vitb_hybrid_route"))
    parser.add_argument("--route-mode", choices=("pragmatic", "hc_only", "base_only"),
                        default="pragmatic")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = PROJ / "scratch_tmp" / "dino_ssl_vitb_hybrid_route"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    per_task_delta = {task: {"mre": [], "param_mae": []} for task in TASKS}
    for fold in range(5):
        gt = PROJ / f"data/_cvfold{fold}_gt.csv"
        control_rows, control_counts = build_route(
            PROJ / f"scratch_tmp/vitb_window9/fold{fold}/regression_predictions.json",
            PROJ / args.control_hcsmall_dir / f"cvfold{fold}/regression_predictions.json",
            PROJ / args.control_hchead_dir / f"cvfold{fold}/regression_predictions.json",
            gt, route_mode=args.route_mode)
        treatment_rows, treatment_counts = build_route(
            PROJ / args.treatment_dir / f"cvfold{fold}/regression_predictions.json",
            PROJ / args.treatment_hcsmall_dir / f"cvfold{fold}/regression_predictions.json",
            PROJ / args.treatment_hchead_dir / f"cvfold{fold}/regression_predictions.json",
            gt, route_mode=args.route_mode)
        control_path = prediction_dir / f"control_cvfold{fold}.json"
        treatment_path = prediction_dir / f"treatment_cvfold{fold}.json"
        control_path.write_text(json.dumps(control_rows) + "\n")
        treatment_path.write_text(json.dumps(treatment_rows) + "\n")
        control = score_submission(control_path, gt)
        treatment = score_submission(treatment_path, gt)
        if control["total_missing"] or treatment["total_missing"]:
            raise ValueError(f"fold {fold}: missing predictions")
        record = {"fold": fold, "postprocess_counts": {
            "control": control_counts, "treatment": treatment_counts}}
        for metric in ("mre", "param_mae"):
            c = task_mean(control, metric)
            t = task_mean(treatment, metric)
            record[f"control_{metric}"] = c
            record[f"treatment_{metric}"] = t
            record[f"delta_{metric}"] = t - c
            for task in TASKS:
                per_task_delta[task][metric].append(
                    treatment["per_task"][task][metric] - control["per_task"][task][metric])
        folds.append(record)
        (args.out_dir / f"control_cvfold{fold}.json").write_text(json.dumps(control, indent=2) + "\n")
        (args.out_dir / f"treatment_cvfold{fold}.json").write_text(
            json.dumps(treatment, indent=2) + "\n")

    summary = {
        "protocol": (
            "OOF old-vs-continued-DINO Docker route; same route mode and post-processing; "
            f"route_mode={args.route_mode}"
        ),
        "prediction_sources": {
            "control": {
                "base": "scratch_tmp/vitb_window9",
                "hcsmall": str(args.control_hcsmall_dir),
                "hchead": str(args.control_hchead_dir),
            },
            "treatment": {
                "base": str(args.treatment_dir),
                "hcsmall": str(args.treatment_hcsmall_dir),
                "hchead": str(args.treatment_hchead_dir),
            },
        },
        "official_validation_used": False,
        "folds": folds,
        "summary": {},
        "per_task_mean_delta": {
            task: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for task, metrics in per_task_delta.items()
        },
    }
    for metric in ("mre", "param_mae"):
        delta = [fold[f"delta_{metric}"] for fold in folds]
        summary["summary"][metric] = {
            "control_mean": float(np.mean([fold[f"control_{metric}"] for fold in folds])),
            "treatment_mean": float(np.mean([fold[f"treatment_{metric}"] for fold in folds])),
            "paired_mean_delta": float(np.mean(delta)),
            "favorable_folds": int(sum(value < 0 for value in delta)),
            "corrected_ci95": ci(delta),
        }
    out = args.out_dir / "paired_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
