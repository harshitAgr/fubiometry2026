#!/usr/bin/env python3
"""Build the independent validation reference for the continued-DINO Docker candidate.

Inputs are produced by ``experiments/infer_ensemble.py``. This script applies the frozen
coordinate-space family route and the same three adopted post-processing functions as Docker.
It writes a reference and audit only; it does not build, zip, push, or submit anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from docker.geometry_project import project  # noqa: E402
from docker.hc_scale import apply_hc_scale  # noqa: E402
from docker.ivc_calibrate import apply_ivc_calibration  # noqa: E402
from experiments.full_family_candidate import blend_family_records  # noqa: E402

INPUTS = {
    "base": PROJ / "submission/dino_ship_reference/base/regression_predictions.json",
    "hcsmall": PROJ / "submission/dino_ship_reference/hcsmall/regression_predictions.json",
    "hchead": PROJ / "submission/dino_ship_reference/hchead/regression_predictions.json",
}
OUT = PROJ / "submission/ship_dino"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    loaded = {name: json.loads(path.read_text()) for name, path in INPUTS.items()}
    routed, route_audit = blend_family_records(
        loaded["base"], loaded["hcsmall"], loaded["hchead"],
        mode="pragmatic_seed42", strict=True,
    )
    if route_audit.get("passed") is not True:
        raise ValueError("family route audit failed")

    output = []
    counts = {"hc_scaled": 0, "geometry_projected": 0, "ivc_calibrated": 0}
    for record in routed:
        image_path = PROJ / "data/val/images" / record["image_path"]
        with Image.open(image_path) as image:
            width, height = image.size
        task = record["task_id"]
        px = list(record["predicted_points_pixels"])
        scaled = apply_hc_scale(px, task, width, height)
        counts["hc_scaled"] += int(scaled != px)
        projected = project(scaled, task)
        counts["geometry_projected"] += int(projected != scaled)
        final = apply_ivc_calibration(projected, task)
        counts["ivc_calibrated"] += int(final != projected)
        if not all(math.isfinite(value) for value in final):
            raise ValueError(f"non-finite output for {task}/{record['image_path']}")
        output.append({
            "image_path": record["image_path"],
            "task_id": task,
            "predicted_points_pixels": final,
        })

    by_task = {}
    for record in output:
        by_task[record["task_id"]] = by_task.get(record["task_id"], 0) + 1
    checks = {
        "route_audit_passed": route_audit.get("passed") is True,
        "records_619": len(output) == 619,
        "nine_tasks": len(by_task) == 9,
        "unique_keys": len({(r["task_id"], r["image_path"]) for r in output}) == len(output),
        "all_finite": all(math.isfinite(v) for r in output
                           for v in r["predicted_points_pixels"]),
    }
    if not all(checks.values()):
        raise ValueError(f"candidate checks failed: {checks}")

    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "regression_predictions.json"
    prediction_path.write_text(json.dumps(output, allow_nan=False) + "\n")
    audit = {
        "passed": True,
        "purpose": "continued-DINO Docker numerical reference; not a submission",
        "input_sha256": {name: sha256(path) for name, path in INPUTS.items()},
        "prediction_sha256": sha256(prediction_path),
        "records_by_task": dict(sorted(by_task.items())),
        "postprocess_counts": counts,
        "route": route_audit,
        "checks": checks,
        "submission_authorized": False,
    }
    (OUT / "candidate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
