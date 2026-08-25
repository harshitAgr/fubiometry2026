"""Checkpoint-only falsification gate for layer-wise encoder learning-rate decay.

Compare five trained ViT-B fold encoders with their common DINOv2 initialization.
LLRD is justified only if lower blocks both moved materially and moved less
consistently across folds than upper blocks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from experiments.encoders import build_encoder


DEFAULT_CHECKPOINTS = tuple(PROJ / "runs" / f"fold{fold}_vitb" / "best_model.pth"
                            for fold in range(5))
OUT = PROJ / "experiments" / "results" / "llrd_preflight" / "report.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_encoder_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path}: checkpoint is not a state dict")
    encoder = {
        key: value.detach().cpu()
        for key, value in state.items()
        if key.startswith("encoder.") and isinstance(value, torch.Tensor)
    }
    if not encoder:
        raise ValueError(f"{path}: no encoder tensors")
    return encoder


def _selected_keys(initial: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(key for key in initial if any(key.startswith(prefix) for prefix in prefixes))


def _layer_stats(
    initial: dict[str, torch.Tensor],
    checkpoints: list[dict[str, torch.Tensor]],
    name: str,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    keys = _selected_keys(initial, prefixes)
    if not keys:
        raise ValueError(f"no initial tensors for {name}: {prefixes}")
    n_fold = len(checkpoints)
    init_norm_sq = 0.0
    update_norm_sq = np.zeros(n_fold, dtype=np.float64)
    update_init_dots = np.zeros(n_fold, dtype=np.float64)
    pair_dots = np.zeros((n_fold, n_fold), dtype=np.float64)
    tensor_count = 0
    parameter_count = 0
    for key in keys:
        init = initial[key]
        if not init.is_floating_point():
            continue
        if not torch.isfinite(init).all():
            raise ValueError(f"non-finite initial tensor {key}")
        deltas = []
        for fold, state in enumerate(checkpoints):
            if key not in state or state[key].shape != init.shape:
                raise ValueError(f"fold {fold}: missing/shape mismatch for {key}")
            if not torch.isfinite(state[key]).all():
                raise ValueError(f"fold {fold}: non-finite tensor {key}")
            deltas.append((state[key].to(torch.float64) - init.to(torch.float64)).reshape(-1))
        init64 = init.to(torch.float64).reshape(-1)
        init_norm_sq += float(torch.dot(init64, init64))
        for left in range(n_fold):
            update_norm_sq[left] += float(torch.dot(deltas[left], deltas[left]))
            update_init_dots[left] += float(torch.dot(deltas[left], init64))
            for right in range(left + 1, n_fold):
                pair_dots[left, right] += float(torch.dot(deltas[left], deltas[right]))
        tensor_count += 1
        parameter_count += init.numel()
    if not tensor_count or init_norm_sq <= 0 or np.any(update_norm_sq <= 0):
        raise ValueError(f"degenerate update geometry for {name}")
    raw_relative = np.sqrt(update_norm_sq / init_norm_sq)
    debiased_norm_sq = update_norm_sq - update_init_dots ** 2 / init_norm_sq
    numerical_floor = np.finfo(np.float64).eps * np.maximum(update_norm_sq, init_norm_sq)
    if np.any(debiased_norm_sq < -32 * numerical_floor):
        raise ValueError(f"invalid radial-debias geometry for {name}: {debiased_norm_sq}")
    debiased_norm_sq = np.maximum(debiased_norm_sq, 0.0)
    debiased_relative = np.sqrt(debiased_norm_sq / init_norm_sq)
    debiased_geometry_valid = bool(np.all(debiased_relative > 1e-12))
    raw_pairwise = {}
    debiased_pairwise = {}
    for left in range(n_fold):
        for right in range(left + 1, n_fold):
            pair = f"fold{left}__fold{right}"
            raw_pairwise[pair] = float(
                pair_dots[left, right]
                / np.sqrt(update_norm_sq[left] * update_norm_sq[right]))
            denominator = np.sqrt(debiased_norm_sq[left] * debiased_norm_sq[right])
            if denominator > 0 and debiased_geometry_valid:
                debiased_dot = (
                    pair_dots[left, right]
                    - update_init_dots[left] * update_init_dots[right] / init_norm_sq)
                debiased_pairwise[pair] = float(debiased_dot / denominator)
            else:
                debiased_pairwise[pair] = None
    raw_cosines = list(raw_pairwise.values())
    debiased_cosines = [value for value in debiased_pairwise.values() if value is not None]
    if not np.isfinite(raw_relative).all() or not np.isfinite(raw_cosines).all():
        raise ValueError(f"non-finite raw update statistics for {name}")
    if debiased_cosines and not np.isfinite(debiased_cosines).all():
        raise ValueError(f"non-finite debiased update statistics for {name}")
    return {
        "name": name,
        "prefixes": list(prefixes),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "raw_relative_drift_by_fold": {
            f"fold{fold}": float(value) for fold, value in enumerate(raw_relative)},
        "raw_median_relative_drift": float(np.median(raw_relative)),
        "raw_pairwise_update_cosines": raw_pairwise,
        "raw_median_pairwise_update_cosine": float(np.median(raw_cosines)),
        "radial_projection_fraction_by_fold": {
            f"fold{fold}": float(update_init_dots[fold] ** 2
                                 / (init_norm_sq * update_norm_sq[fold]))
            for fold in range(n_fold)
        },
        "debiased_relative_drift_by_fold": {
            f"fold{fold}": float(value) for fold, value in enumerate(debiased_relative)},
        "debiased_median_relative_drift": float(np.median(debiased_relative)),
        "debiased_geometry_valid": debiased_geometry_valid,
        "debiased_pairwise_update_cosines": debiased_pairwise,
        "debiased_median_pairwise_update_cosine": (
            float(np.median(debiased_cosines))
            if len(debiased_cosines) == n_fold * (n_fold - 1) // 2 else None),
    }


def analyze_layer_updates(
    initial: dict[str, torch.Tensor],
    checkpoints: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if len(checkpoints) != 5:
        raise ValueError(f"expected five checkpoints, got {len(checkpoints)}")
    initial_keys = set(initial)
    for fold, state in enumerate(checkpoints):
        if set(state) != initial_keys:
            missing = sorted(initial_keys - set(state))
            extra = sorted(set(state) - initial_keys)
            raise ValueError(
                f"fold {fold}: encoder key mismatch; missing={missing[:3]}, extra={extra[:3]}")

    group_specs = [
        ("stem", ("encoder.backbone.cls_token", "encoder.backbone.pos_embed",
                  "encoder.backbone.patch_embed.")),
        *[(f"block{index}", (f"encoder.backbone.blocks.{index}.",))
          for index in range(12)],
        ("final_norm", ("encoder.backbone.norm.",)),
    ]
    assignments = {key: [] for key in initial_keys}
    for name, prefixes in group_specs:
        for key in _selected_keys(initial, prefixes):
            assignments[key].append(name)
    bad_assignments = {key: value for key, value in assignments.items() if len(value) != 1}
    if bad_assignments:
        raise ValueError(f"encoder keys must map to exactly one layer group: {bad_assignments}")

    groups = {
        name: _layer_stats(initial, checkpoints, name, prefixes)
        for name, prefixes in group_specs
    }
    blocks = [
        groups[f"block{index}"]
        for index in range(12)
    ]
    bottom = blocks[:6]
    top = blocks[6:]

    def half_summary(prefix: str) -> dict[str, Any]:
        drift_key = f"{prefix}_median_relative_drift"
        cosine_key = f"{prefix}_median_pairwise_update_cosine"
        bottom_drift = float(np.median([row[drift_key] for row in bottom]))
        top_drift = float(np.median([row[drift_key] for row in top]))
        drift_ratio = bottom_drift / top_drift if top_drift > 0 else 0.0
        bottom_cosines = [row[cosine_key] for row in bottom]
        top_cosines = [row[cosine_key] for row in top]
        geometry_valid = all(value is not None for value in bottom_cosines + top_cosines)
        bottom_consistency = (float(np.median(bottom_cosines)) if geometry_valid else None)
        top_consistency = (float(np.median(top_cosines)) if geometry_valid else None)
        gap = (top_consistency - bottom_consistency if geometry_valid else None)
        return {
            "bottom6_median_relative_drift": bottom_drift,
            "top6_median_relative_drift": top_drift,
            "bottom_to_top_drift_ratio": drift_ratio,
            "bottom6_median_update_consistency": bottom_consistency,
            "top6_median_update_consistency": top_consistency,
            "top_minus_bottom_consistency_gap": gap,
            "geometry_valid": geometry_valid,
        }

    raw_summary = half_summary("raw")
    debiased_summary = half_summary("debiased")
    gate = {
        "raw_bottom_to_top_drift_ratio_at_least_0_25": (
            raw_summary["bottom_to_top_drift_ratio"] >= 0.25),
        "raw_top_minus_bottom_consistency_gap_at_least_0_10": (
            raw_summary["top_minus_bottom_consistency_gap"] is not None
            and raw_summary["top_minus_bottom_consistency_gap"] >= 0.10),
        "debiased_bottom_to_top_drift_ratio_at_least_0_25": (
            debiased_summary["bottom_to_top_drift_ratio"] >= 0.25),
        "debiased_top_minus_bottom_consistency_gap_at_least_0_10": (
            debiased_summary["top_minus_bottom_consistency_gap"] is not None
            and debiased_summary["top_minus_bottom_consistency_gap"] >= 0.10),
    }
    return {
        "definitions": {
            "raw_relative_drift": "L2(trained-init)/L2(init), aggregated over all tensors in block",
            "debiased_update": "trained-init projected orthogonal to the initialization vector",
            "block_consistency": "median of 10 pairwise cross-fold update-direction cosines",
            "half_summary": "median across six block statistics",
        },
        "coverage": {
            "encoder_key_count": len(initial_keys),
            "analyzed_key_count": sum(len(_selected_keys(initial, prefixes))
                                      for _, prefixes in group_specs),
            "encoder_parameter_count": sum(value.numel() for value in initial.values()),
            "analyzed_parameter_count": sum(row["parameter_count"] for row in groups.values()),
            "every_key_in_exactly_one_group": True,
        },
        "stem": groups["stem"],
        "blocks": blocks,
        "final_norm": groups["final_norm"],
        "summary": {"raw": raw_summary, "debiased": debiased_summary},
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--checkpoints", nargs=5, type=Path, default=DEFAULT_CHECKPOINTS)
    args = parser.parse_args()
    checkpoint_paths = [path.resolve() for path in args.checkpoints]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    encoder = build_encoder("dinov2_vitb", input_size=518)
    initial = {f"encoder.{key}": value.detach().cpu()
               for key, value in encoder.state_dict().items()}
    del encoder
    gc.collect()
    checkpoints = []
    for path in checkpoint_paths:
        checkpoints.append(load_encoder_checkpoint(path))
        gc.collect()
    report = analyze_layer_updates(initial, checkpoints)
    report.update({
        "protocol": "preregistered checkpoint-only LLRD mechanism falsification",
        "pretrained_encoder": "timm vit_base_patch14_dinov2.lvd142m",
        "pretrained_encoder_state_sha256": state_sha256(initial),
        "checkpoint_order": [str(path.relative_to(PROJ)) for path in checkpoint_paths],
        "checkpoint_sha256": {str(path.relative_to(PROJ)): file_sha256(path)
                              for path in checkpoint_paths},
        "checkpoint_encoder_state_sha256": {
            str(path.relative_to(PROJ)): state_sha256(state)
            for path, state in zip(checkpoint_paths, checkpoints)
        },
        "source_sha256": {
            "experiments/preflight_llrd.py": file_sha256(Path(__file__).resolve()),
            "experiments/reproduce_llrd_preflight.sh": file_sha256(
                PROJ / "experiments" / "reproduce_llrd_preflight.sh"),
            "experiments/reproduce_vitb_5fold.sh": file_sha256(
                PROJ / "experiments" / "reproduce_vitb_5fold.sh"),
            "experiments/run_config.py": file_sha256(PROJ / "experiments" / "run_config.py"),
            "experiments/encoders.py": file_sha256(PROJ / "experiments" / "encoders.py"),
            "baseline/baseline/model_factory.py": file_sha256(
                PROJ / "baseline" / "baseline" / "model_factory.py"),
        },
        "versions": {
            "torch": torch.__version__,
            "timm": __import__("timm").__version__,
            "numpy": np.__version__,
        },
        "if_passed_fixed_treatment": {
            "top_block_and_final_norm_lr": 2e-5,
            "per_depth_decay_toward_input": 0.75,
            "stem_lr_cls_position_and_patch_embedding": 2e-5 * 0.75 ** 12,
            "head_lr": 1e-3,
            "sweep": "none",
        },
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
