#!/usr/bin/env python3
"""Stage the audited continued-DINO full-data family into ``docker/weights``.

Every source hash is pinned below. Destination replacement is atomic, and the generated manifest
records both the prior and staged hashes so the previous candidate remains identifiable.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "experiments/results/dino_ssl_vitb_docker/staging_manifest.json"

MEMBERS = {
    "best_model.pth": (
        "runs/vitb_full_dino_corr/best_model.pth",
        "264c79797d380d5472f2546f3aa5345350b704e0ee043f64601becb222c23308",
    ),
    "model_s43.pth": (
        "runs/vitb_full_dino_corr_s43/best_model.pth",
        "4f6b0d47b0cda0cb9d8b727456c7fae738281c2dd645fee84327e7fe9b527b6a",
    ),
    "model_s44.pth": (
        "runs/vitb_full_dino_corr_s44/best_model.pth",
        "eb8bb9c875c05e0d437b4bfca75d2258c1ca1f60d68db5c3f7bb507cac940113",
    ),
    "model_s45.pth": (
        "runs/vitb_full_dino_corr_s45/best_model.pth",
        "acefb9f648ebb0ebb045fbb8063e2f6683ba26bd8f385d3ae141068caf69a215",
    ),
    "model_s46.pth": (
        "runs/vitb_full_dino_corr_s46/best_model.pth",
        "fe491fad70df96324d4048966be96771b0282d61fb26a02b6186d1d149b1a4cd",
    ),
    "model_hcsmall.pth": (
        "runs/vitb_full_dino_hcsmall_corr/best_model.pth",
        "79c3c273b5e4f7a5d30f12dcbd39faaf8024347e4bfc630bdc0daba49109bfc2",
    ),
    "model_hchead.pth": (
        "runs/vitb_full_dino_hchead_corr/best_model.pth",
        "2a5d43069c5d4ba9455ab6fd8c142b3dc15e1c7a5ec773ba055111f9e1a20240",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    weights = PROJ / "docker/weights"
    weights.mkdir(parents=True, exist_ok=True)
    checked = {}
    for name, (rel, expected) in MEMBERS.items():
        source = PROJ / rel
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"{rel}: expected {expected}, got {actual}")
        dest = weights / name
        checked[name] = {
            "source": rel,
            "source_sha256": actual,
            "previous_destination_sha256": sha256(dest) if dest.is_file() else None,
        }

    for name, (rel, expected) in MEMBERS.items():
        source = PROJ / rel
        dest = weights / name
        temp = weights / f".{name}.stage-{os.getpid()}"
        if temp.exists():
            temp.unlink()
        os.link(source, temp)
        os.replace(temp, dest)
        actual = sha256(dest)
        if actual != expected:
            raise RuntimeError(f"post-stage hash mismatch for {dest}: {actual}")
        checked[name]["destination"] = str(dest.relative_to(PROJ))
        checked[name]["destination_sha256"] = actual

    report = {
        "passed": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "family": "continued_dinov2_vitb_full_data",
        "members": checked,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"staged {len(checked)} audited members; manifest={OUT.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
