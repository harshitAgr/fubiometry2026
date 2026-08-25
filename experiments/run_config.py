"""Train the baseline on an explicit train/val fold and score the held-out fold.

Reuses the baseline's model/dataset/utils but runs OUR loop on explicit fold indices (no
leaky internal split). Trains a fixed number of epochs and saves the FINAL-epoch weights
(no val-based best-epoch selection; the file is named best_model.pth only for the baseline
loader). Supports a reduced epoch count for screening, then predicts + scores the held-out
fold with scoring.score.

Run with the BASELINE venv:
  baseline/.venv-baseline/bin/python experiments/run_config.py --fold 0 --epochs 15
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
sys.path.insert(0, PROJ)
from dataset import KeypointDataset, KeypointUniformSampler  # noqa: E402
from model_factory import MultiTaskModelFactory               # noqa: E402
from utils import keypoint_collate_fn, set_seed                # noqa: E402
from model import Model                                        # noqa: E402
from scoring import score as scorer                            # noqa: E402
from experiments.augment import (build_train_transform, build_geo_fallback, GEO_PACKS,  # noqa: E402
                                  build_aop_task_transforms, build_hc_smallhead_transform)  # noqa: E402
from experiments.kp_aug_dataset import KeypointAugDataset                    # noqa: E402
from experiments.encoders import build_encoder                               # noqa: E402
from experiments.per_task_model import build_model                           # noqa: E402
from experiments.param_loss import combined_loss as param_combined_loss      # noqa: E402
from experiments.adaptive_wing import adaptive_wing_loss                     # noqa: E402
from experiments.model_ema import StateDictEMA                               # noqa: E402
from experiments.marginal_loss import mse_with_marginal_kl                    # noqa: E402

HM = (64, 64)
# Default input size for the DINOv2 ViT-S/14 incumbent (kept as module constant
# for backward-compat with callers that don't pass --input-size)
INPUT = 518


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable without GPU/model)
# ---------------------------------------------------------------------------

def make_lr_schedule(optimizer, epochs, warmup, cosine):
    """Build and return (scheduler, is_sequential) for the training loop.

    Supports four combinations:
      - warmup=0, cosine=False  -> no scheduler (constant LR)
      - warmup=0, cosine=True   -> CosineAnnealingLR(T_max=epochs)
      - warmup>0, cosine=False  -> LinearLR warmup only
      - warmup>0, cosine=True   -> LinearLR warmup then CosineAnnealingLR

    Returns the scheduler (or None) so the caller calls sched.step() once per epoch.
    The warmup phase occupies epoch indices [0, warmup-1]; cosine occupies
    [warmup, epochs-1].  Each component receives the correct T_max/total_iters.
    """
    if warmup <= 0 and not cosine:
        return None
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup)
    if not cosine:
        return warmup_sched
    # cosine after warmup; remaining epochs = epochs - warmup
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


def lr_values_for_schedule(base_lr, epochs, warmup, cosine):
    """Return the list of LRs (one per epoch) for a given schedule config.

    Used in unit tests (and optionally for logging); does NOT touch a real model.
    Simulates the scheduler by creating a minimal single-param-group optimizer
    without referencing actual tensors, so this helper works in the project venv
    (no torch tensors needed — only the scheduler arithmetic is exercised).
    """
    # Build a minimal fake param group so torch schedulers can read/write .lr
    class _DummyParamGroup(dict):
        pass
    pg = _DummyParamGroup(lr=base_lr, params=[])

    class _DummyOpt:
        """Mimics the interface torch schedulers expect (param_groups, state_dict stub)."""
        def __init__(self, pg):
            self.param_groups = [pg]
            self.state_dict = lambda: {}  # satisfies SequentialLR internal checks

    opt = _DummyOpt(pg)
    sched = make_lr_schedule(opt, epochs, warmup, cosine)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
    return lrs


def select_best_by_val(history):
    """Return the record from *history* with the minimum val_avg_mre.

    history: list of dicts each containing at least "val_avg_mre" (float or None).
    Returns None if history is empty or all val_avg_mre are None/nan.
    """
    best = None
    for record in history:
        v = record.get("val_avg_mre")
        if v is None:
            continue
        try:
            if np.isnan(v):
                continue
        except TypeError:
            continue
        if best is None or v < best["val_avg_mre"]:
            best = record
    return best


def build_metrics_record(epoch, train_loss, per_task_train_loss, lr,
                         val_avg_mre=None, val_per_task_mre=None):
    """Assemble one line for metrics.jsonl.

    All fields are always present; val fields are None when not computed.
    """
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "per_task_train_loss": per_task_train_loss,   # dict {task_id: float}
        "lr": lr,
        "val_avg_mre": val_avg_mre,
        "val_per_task_mre": val_per_task_mre,         # dict {task_id: float} or None
    }


def configure_trainable_scope(model, freeze_encoder=False, train_task=None,
                              train_fusion_only=False, unfreeze_last_blocks=0):
    """Set ``requires_grad`` for the requested training scope and return trainable parameters.

    ``train_task`` is the strict checkpoint-refinement mode: the encoder and every other head are
    frozen, leaving only the named task head trainable. With no task restriction, this preserves
    the historical ``freeze_encoder`` behavior.
    """
    if train_task is not None and train_fusion_only:
        raise ValueError("--train-task and --train-fusion-only are mutually exclusive")
    if unfreeze_last_blocks < 0:
        raise ValueError("--unfreeze-last-blocks must be non-negative")
    if unfreeze_last_blocks and train_task is None:
        raise ValueError("--unfreeze-last-blocks requires --train-task")
    if train_fusion_only:
        fusion = getattr(model.encoder, "fusion", None)
        if fusion is None:
            raise ValueError("--train-fusion-only requires a fusion encoder")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in fusion.parameters():
            parameter.requires_grad = True
    elif train_task is not None:
        if train_task not in model.heads:
            raise ValueError(f"unknown --train-task {train_task!r}; available: {sorted(model.heads)}")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.heads[train_task].parameters():
            parameter.requires_grad = True
        if unfreeze_last_blocks:
            backbone = getattr(model.encoder, "backbone", None)
            blocks = getattr(backbone, "blocks", None)
            if blocks is None or unfreeze_last_blocks > len(blocks):
                raise ValueError(f"encoder does not expose {unfreeze_last_blocks} final blocks")
            for block in blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            # DINOv2 applies this norm after the final block; let it co-adapt with the branch.
            norm = getattr(backbone, "norm", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad = True
    elif freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


# ---------------------------------------------------------------------------
# AOP per-landmark loss weighting (precision lever — pure, unit-testable)
# ---------------------------------------------------------------------------

def weighted_heatmap_mse(pred, target, weights):
    """Per-landmark-weighted MSE over heatmap channels (AOP precision lever).

    pred, target: [B, K, H, W] tensors (pred already sigmoid-activated).
    weights:      length-K per-landmark weights. Normalized to mean 1 internally, so
        UNIFORM weights reproduce ``F.mse_loss(pred, target)`` EXACTLY and the loss
        magnitude (hence the LR / warmup-cosine schedule) is preserved under reweighting.

    Diagnostic (geo_cosine40 5-fold CV): AOP p3 (LM4) mean error 21.4px = 2.1x the
    vertex -> the dominant AOP-MRE source. Upweighting p3 reallocates the shared
    encoder's focus toward the bottleneck landmark.
    """
    per_ch = ((pred - target) ** 2).mean(dim=(0, 2, 3))   # [K] mean SE per landmark
    w = torch.as_tensor(weights, dtype=per_ch.dtype, device=per_ch.device)
    w = w / w.mean()
    return (per_ch * w).mean()


# ---------------------------------------------------------------------------
# FA wall-variance reshaping lever (2026-08-07 FA diagnosis: unbiased size variance
# concentrated in the along-axis error of the lateral-wall landmarks p2/p3).
# ---------------------------------------------------------------------------

def parse_fa_aniso_sigma(spec):
    """Parse ``--fa-aniso-sigma "SX,SY"`` into a ``(sx, sy)`` float tuple; ``None`` passes through.

    Pure string parsing (no torch) so it is unit-testable without a GPU/model.
    """
    if spec is None:
        return None
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--fa-aniso-sigma must be 'SX,SY', got {spec!r}")
    try:
        sx, sy = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"--fa-aniso-sigma must be 'SX,SY' floats, got {spec!r}") from exc
    if not (sx > 0 and sy > 0):
        raise ValueError(f"--fa-aniso-sigma values must be positive, got {spec!r}")
    return (sx, sy)


def validate_fa_config(train_task, aug, fa_wall_weight=1.0, fa_aniso_sigma=None):
    """Scope guard for the FA-head-refinement lever.

    Both new FA knobs are checkpoint-refinement-only (the same machinery as the HC-head
    refinement): they require ``--train-task FA`` so their effect is confined to a single
    frozen-encoder head-refinement run, never a general multi-task training config.
    ``fa_aniso_sigma`` additionally requires an aug pack this repo can render it under:
    a geometric pack (KeypointAugDataset already renders FA per-task) or ``photo_v1``
    (routed through KeypointAugDataset with the proven-equivalent keypoint-aware photo_v1
    transform -- see ``build_geo_fallback`` and ``tests/test_kp_aug.py::
    test_geo_p0_matches_baseline_heatmap``). Any other aug value is rejected rather than
    silently mis-rendered.
    """
    fa_active = fa_wall_weight != 1.0 or fa_aniso_sigma is not None
    if fa_active and train_task != "FA":
        raise ValueError("--fa-wall-weight / --fa-aniso-sigma require --train-task FA")
    if fa_aniso_sigma is not None and aug != "photo_v1" and aug not in GEO_PACKS:
        raise ValueError(
            f"--fa-aniso-sigma is only supported with --aug photo_v1 or a geometric pack "
            f"({sorted(GEO_PACKS)}); got --aug {aug!r}")


# ---------------------------------------------------------------------------
# Per-task loss dispatch (extracted for direct unit-testing of the branch logic
# used inside train_fold's inner loop -- both the AOP lever and the IVC/HC param-loss
# probe are gated per-task_id here; every task NOT named goes through plain F.mse_loss).
# ---------------------------------------------------------------------------

def select_task_loss(tid, pred, hm, aop_p3_weight=1.0, param_loss_beta=0.0,
                      param_loss_temperature=1.0, heatmap_loss="mse",
                      awing_alpha=2.1, awing_omega=14.0, awing_epsilon=1.0,
                      awing_theta=0.5, marginal_kl_beta=None, fa_wall_weight=1.0):
    """Return the training loss for one task_id's batch (pred, hm already sliced to that task).

    Dispatch (mirrors the training loop in train_fold, kept in exact sync with it):
      - marginal_kl_beta is not None -> MSE + beta * axis-marginal KL.  Beta zero
        deliberately traverses this new path but returns exact plain MSE.
      - heatmap_loss == "adaptive_wing" -> core Adaptive Wing loss for every task
      - tid == "AOP" and aop_p3_weight != 1.0  -> weighted_heatmap_mse (Lever A, REJECTED but kept)
      - tid == "FA" and fa_wall_weight != 1.0  -> weighted_heatmap_mse([1,1,W,W]) on p2/p3
        (the lateral-wall landmarks; FA head-refinement lever, 2026-08-07)
      - tid in {"IVC","HC"} and param_loss_beta != 0.0 -> param_combined_loss (this probe)
      - otherwise (ALL other tasks, and IVC/HC/AOP/FA when their lever knob is at its
        default/off value) -> plain F.mse_loss(pred, hm), BYTE-IDENTICAL to the pre-lever
        baseline.

    Do-no-harm: with aop_p3_weight=1.0, fa_wall_weight=1.0 (both defaults) AND
    param_loss_beta=0.0 (default), every task_id takes the plain F.mse_loss branch -- this
    function then behaves identically to a version of train_fold with none of these
    levers' code present at all.
    """
    if marginal_kl_beta is not None:
        return mse_with_marginal_kl(pred, hm, marginal_kl_beta)
    if heatmap_loss == "adaptive_wing":
        return adaptive_wing_loss(pred, hm, alpha=awing_alpha, omega=awing_omega,
                                  epsilon=awing_epsilon, theta=awing_theta)
    if heatmap_loss != "mse":
        raise ValueError(f"unknown heatmap loss: {heatmap_loss!r}")
    if tid == "AOP" and aop_p3_weight != 1.0:
        return weighted_heatmap_mse(pred, hm, [1.0, 1.0, 1.0, aop_p3_weight])
    if tid == "FA" and fa_wall_weight != 1.0:
        return weighted_heatmap_mse(pred, hm, [1.0, 1.0, fa_wall_weight, fa_wall_weight])
    if tid in ("IVC", "HC") and param_loss_beta != 0.0:
        return param_combined_loss(pred, hm, tid, alpha=1.0, beta=param_loss_beta,
                                    temperature=param_loss_temperature)
    return F.mse_loss(pred, hm)


def validate_loss_config(heatmap_loss="mse", aop_p3_weight=1.0, param_loss_beta=0.0,
                         marginal_kl_betas=None, fa_wall_weight=1.0):
    """Reject mixed loss levers so an Adaptive Wing result has one causal change."""
    if heatmap_loss not in {"mse", "adaptive_wing"}:
        raise ValueError(f"unknown heatmap loss: {heatmap_loss!r}")
    if heatmap_loss == "adaptive_wing" and (
            aop_p3_weight != 1.0 or param_loss_beta != 0.0 or fa_wall_weight != 1.0):
        raise ValueError("--heatmap-loss adaptive_wing cannot be combined with "
                         "--aop-p3-weight, --param-loss-beta, or --fa-wall-weight")
    if marginal_kl_betas is not None:
        if (heatmap_loss != "mse" or aop_p3_weight != 1.0 or param_loss_beta != 0.0
                or fa_wall_weight != 1.0):
            raise ValueError("--marginal-kl-betas cannot be combined with another loss lever")
        if not isinstance(marginal_kl_betas, dict):
            raise ValueError("marginal KL betas must be a task-to-beta mapping")
        for task_id, beta in marginal_kl_betas.items():
            if not isinstance(task_id, str) or not isinstance(beta, (int, float)):
                raise ValueError("marginal KL betas must map task strings to numbers")
            if not np.isfinite(beta) or beta < 0:
                raise ValueError(f"invalid marginal KL beta for {task_id}: {beta!r}")


def load_marginal_kl_betas(path):
    """Load the fixed task-to-beta mapping emitted by the calibration preflight."""
    with open(path) as handle:
        payload = json.load(handle)
    if "passed" in payload and not payload["passed"]:
        raise ValueError(f"refusing failed marginal KL calibration: {path}")
    betas = payload.get("betas", payload)
    validate_loss_config(marginal_kl_betas=betas)
    return {task_id: float(beta) for task_id, beta in betas.items()}


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Val-eval helper
# ---------------------------------------------------------------------------

def _run_val_eval(checkpoint_path, ckpt_dir, fold, encoder, input_size, mem_frac,
                  heatmap_size=HM, model_variant="base"):
    """Run infer_tta predict + score on the held-out fold, return (avg_mre, per_task_mre).

    Uses method="soft", tta="none" for speed. Returns (None, None) on any failure.
    """
    from experiments import infer_tta
    split_csv = os.path.join(PROJ, "data", f"_cvfold{fold}_val.csv")
    gt_csv = os.path.join(PROJ, "data", f"_cvfold{fold}_gt.csv")
    if not (os.path.exists(split_csv) and os.path.exists(gt_csv)):
        print(f"  [val-eval] split/gt csv not found for fold {fold} — skipping")
        return None, None
    out_dir = os.path.join(ckpt_dir, "_val_tmp")
    try:
        pred_json = infer_tta.predict(
            checkpoint=checkpoint_path,
            data_root=os.path.join(PROJ, "data"),
            split_csv=split_csv,
            out_dir=out_dir,
            method="soft",
            tta="none",
            scales=(0.92, 1.08),
            window=7,
            mem_frac=mem_frac,
            encoder_name=encoder,
            input_size=input_size,
            heatmap_size=heatmap_size,
            model_variant=model_variant,
        )
        res = scorer.score_submission(pred_json, gt_csv)
        avg_mre = res.get("avg_mre")
        per_task = {tid: v["mre"] for tid, v in res.get("per_task", {}).items()}
        return avg_mre, per_task
    except Exception as exc:
        print(f"  [val-eval] ERROR: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Train-dataset construction (extracted from train_fold for direct unit-testing of the
# aug-pack -> dataset-class routing, including the FA anisotropic-target lever's dispatch).
# ---------------------------------------------------------------------------

def build_train_dataset(aug, input_size, seed, heatmap_size, sigma, fa_aniso_sigma=None,
                        data_root=None):
    """Build the training Dataset for the given aug pack (mirrors train_fold's historical
    inline branch exactly; callers must have already run validate_fa_config).

    - aug in GEO_PACKS       -> KeypointAugDataset with the pack's own (possibly per-task)
      keypoint-aware transforms. fa_aniso_sigma flows straight through -- KeypointAugDataset's
      renderer is already per-task.
    - aug == "photo_v1" AND fa_aniso_sigma is not None -> routes through KeypointAugDataset
      using build_geo_fallback (photo_v1's own ops, made keypoint-aware) as BOTH transforms and
      fallback_transforms. photo_v1 has no geometric component, so this is proven byte-identical
      to the plain KeypointDataset+photo_v1 path for every task's image/label (see
      tests/test_kp_aug.py::test_geo_p0_matches_baseline_heatmap); only the FA heatmap RENDERING
      changes (isotropic -> anisotropic on p2/p3).
    - otherwise (fa_aniso_sigma is None, the default) -> plain KeypointDataset, BYTE-IDENTICAL
      to every pre-lever caller.
    """
    data_root = data_root or os.path.join(PROJ, "data")
    if aug in GEO_PACKS:
        # Composite geo packs run geo_v1 as the base for ALL tasks, plus an optional
        # single-task override transform:
        #   aop_robust_v1  -> AOP: probe-angle Perspective + wider Affine
        #   geo_v1_hcsmall -> HC:  aggressive zoom-out (synthesizes small heads)
        # Every other task keeps base geo_v1 unchanged; reject-sampling falls back to
        # build_geo_fallback (photometric-only, valid geometry).
        base_aug = "geo_v1" if aug in ("aop_robust_v1", "geo_v1_hcsmall") else aug
        task_tfm = None
        task_fallback = None
        if aug == "aop_robust_v1":
            task_tfm = {"AOP": build_aop_task_transforms(input_size, seed=seed)}
            task_fallback = {"AOP": build_geo_fallback(input_size, seed=seed)}
        elif aug == "geo_v1_hcsmall":
            task_tfm = {"HC": build_hc_smallhead_transform(input_size, seed=seed)}
            task_fallback = {"HC": build_geo_fallback(input_size, seed=seed)}
        return KeypointAugDataset(
            data_root=data_root,
            transforms=build_train_transform(base_aug, input_size, seed=seed),
            fallback_transforms=build_geo_fallback(input_size, seed=seed),
            heatmap_size=heatmap_size, sigma=sigma, input_size=input_size,
            task_transforms=task_tfm,
            task_fallback_transforms=task_fallback,
            fa_aniso_sigma=fa_aniso_sigma)
    if fa_aniso_sigma is not None:
        assert aug == "photo_v1", aug  # validate_fa_config guarantees this
        return KeypointAugDataset(
            data_root=data_root,
            transforms=build_geo_fallback(input_size, seed=seed),
            fallback_transforms=build_geo_fallback(input_size, seed=seed),
            heatmap_size=heatmap_size, sigma=sigma, input_size=input_size,
            fa_aniso_sigma=fa_aniso_sigma)
    return KeypointDataset(
        data_root=data_root,
        transforms=build_train_transform(aug, input_size, seed=seed),
        heatmap_size=heatmap_size, sigma=sigma)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fold(folds_csv, fold, epochs=15, seed=42, mem_frac=0.30, aug="none",
               encoder="dinov2_vits", input_size=518, freeze_encoder=False,
               encoder_init=None, init_checkpoint=None, train_task=None,
               full_data=False, ckpt_name=None,
               save_epochs=None, cosine=False, warmup=0, val_every=0, heatmap_size=HM, sigma=1.8,
               aop_p3_weight=1.0, param_loss_beta=0.0, param_loss_temperature=1.0,
               head_lr=1e-3, heatmap_loss="mse", awing_alpha=2.1,
               awing_omega=14.0, awing_epsilon=1.0, awing_theta=0.5,
               train_fusion_only=False, fusion_lr=1e-4, ema_decay=0.0,
               ema_update_every=1, unfreeze_last_blocks=0,
               encoder_refine_lr=2e-6, marginal_kl_betas=None,
               exclude_train_paths_file=None, model_variant="base",
               coordse_coord_beta=1.0, fa_wall_weight=1.0, fa_aniso_sigma=None):
    validate_loss_config(heatmap_loss, aop_p3_weight, param_loss_beta,
                         marginal_kl_betas=marginal_kl_betas, fa_wall_weight=fa_wall_weight)
    validate_fa_config(train_task, aug, fa_wall_weight=fa_wall_weight,
                       fa_aniso_sigma=fa_aniso_sigma)
    marginal_mode = marginal_kl_betas is not None
    if marginal_mode and (not freeze_encoder or init_checkpoint is None):
        raise ValueError("marginal KL continuation requires --freeze-encoder and --init-checkpoint")
    if ema_decay != 0.0 and not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be 0 (off) or strictly between 0 and 1")
    if ema_update_every <= 0:
        raise ValueError("ema_update_every must be positive")
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_train_dataset(aug, input_size, seed, heatmap_size, sigma,
                             fa_aniso_sigma=fa_aniso_sigma)
    folds = pd.read_csv(folds_csv)
    val_paths = set() if full_data else set(folds[folds.fold == fold]["image_path"])
    # keep = everything except the guard-dropped band (fold == -1). fold == -2 marks
    # TRAIN-ONLY rows (ingested external data): kept in train, never a val fold, so the
    # held-out val distribution stays bit-identical to the adopted no-external CV baseline.
    keep = set(folds[folds.fold != -1]["image_path"])
    paths = ds.dataframe["image_path"]
    is_val = paths.isin(val_paths).to_numpy()
    train_idx = np.where(~is_val & paths.isin(keep).to_numpy())[0]
    if exclude_train_paths_file is not None:
        excluded = {
            line.strip() for line in open(exclude_train_paths_file, encoding="utf-8")
            if line.strip() and not line.lstrip().startswith("#")
        }
        excluded_mask = paths.isin(excluded).to_numpy()
        train_idx = train_idx[~excluded_mask[train_idx]]
        print(f"[exclude-train-paths] removed {int(excluded_mask.sum())} rows from training")
    if full_data:
        print(f"[full-data] training on ALL {len(train_idx)} images (no held-out val)")

    cfgs, seen = [], set()
    for _, r in ds.dataframe.iloc[train_idx].iterrows():
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            cfgs.append({"task_id": r["task_id"], "task_name": "Regression",
                         "num_classes": int(r["num_classes"])})
    if marginal_mode:
        expected_tasks = {config["task_id"] for config in cfgs}
        supplied_tasks = set(marginal_kl_betas)
        if supplied_tasks != expected_tasks:
            raise ValueError("marginal KL beta task mismatch: "
                             f"missing={sorted(expected_tasks - supplied_tasks)}, "
                             f"extra={sorted(supplied_tasks - expected_tasks)}")
    enc = build_encoder(encoder, input_size, encoder_init=encoder_init)
    model = build_model(cfgs, heatmap_size, enc, variant=model_variant).to(dev)
    if (train_task is not None or train_fusion_only) and init_checkpoint is None:
        raise ValueError("checkpoint refinement requires --init-checkpoint")
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location=dev, weights_only=True)
        if train_fusion_only:
            info = model.load_state_dict(state, strict=False)
            expected_missing = {"encoder.fusion.weight"}
            if set(info.missing_keys) != expected_missing or info.unexpected_keys:
                raise RuntimeError("fusion checkpoint migration failed: "
                                   f"missing={info.missing_keys}, unexpected={info.unexpected_keys}")
        else:
            model.load_state_dict(state, strict=True)
        print(f"[init] loaded full model checkpoint: {init_checkpoint}")
    trainable = configure_trainable_scope(model, freeze_encoder=freeze_encoder,
                                          train_task=train_task,
                                          train_fusion_only=train_fusion_only,
                                          unfreeze_last_blocks=unfreeze_last_blocks)
    if train_task is not None:
        task_mask = ds.dataframe.iloc[train_idx]["task_id"].to_numpy() == train_task
        train_idx = train_idx[task_mask]
        print(f"[train-task] {train_task}: {len(train_idx)} training images; "
              f"{sum(parameter.numel() for parameter in trainable):,} trainable parameters")
    if not trainable:
        raise ValueError("training scope contains no trainable parameters")
    sub = torch.utils.data.Subset(ds, train_idx.tolist())
    sub.dataframe = ds.dataframe.iloc[train_idx].reset_index(drop=True)
    loader = torch.utils.data.DataLoader(sub, batch_sampler=KeypointUniformSampler(sub, 4),
                                         num_workers=4, collate_fn=keypoint_collate_fn)
    groups = []
    if train_fusion_only:
        groups.append({"params": trainable, "lr": fusion_lr})
        print(f"[fusion-only] {sum(parameter.numel() for parameter in trainable):,} trainable "
              f"parameters at lr={fusion_lr:.2e}; backbone and all heads frozen")
    elif train_task is not None:
        head_parameters = list(model.heads[train_task].parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        encoder_parameters = [parameter for parameter in trainable
                              if id(parameter) not in head_ids]
        groups.append({"params": head_parameters, "lr": head_lr})
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": encoder_refine_lr})
            print(f"[train-task] last {unfreeze_last_blocks} encoder block(s) + final norm: "
                  f"{sum(parameter.numel() for parameter in encoder_parameters):,} parameters "
                  f"at lr={encoder_refine_lr:.2e}")
    else:
        if not freeze_encoder:
            groups.append({"params": model.encoder.parameters(), "lr": 2e-5})
        for h in model.heads.values():
            groups.append({"params": h.parameters(), "lr": head_lr})
    opt = torch.optim.AdamW(groups)
    ema = StateDictEMA(model, ema_decay) if ema_decay else None
    optimizer_steps = 0
    last_ema_step = 0
    sched = make_lr_schedule(opt, epochs, warmup, cosine)
    ckpt_dir = os.path.join(PROJ, "runs", ckpt_name or f"cvfold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_at = set(save_epochs or [])

    metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")
    metrics_fh = open(metrics_path, "w")  # truncate/create fresh each train run

    # For best-by-val checkpoint
    best_val_mre = None
    val_history = []
    task_optimizer_steps = {config["task_id"]: 0 for config in cfgs}
    optimizer_task_sequence = hashlib.sha256()
    optimizer_target_sequence = hashlib.sha256()

    for ep in range(epochs):
        model.train()
        if marginal_mode:
            # The encoder is frozen for this continuation.  requires_grad=False alone
            # does not disable train-time stochasticity, so keep its state and features fixed.
            model.encoder.eval()
        if train_fusion_only:
            # requires_grad=False does not freeze BatchNorm running statistics.  Keep every
            # historical module in inference mode so only fusion.weight can change.
            model.encoder.backbone.eval()
            model.heads.eval()
            model.encoder.fusion.train()
        ep_loss, nb = 0.0, 0
        # Per-task loss accumulators: {task_id: [list of loss values]}
        task_losses: dict = {}
        for b in loader:
            if marginal_mode:
                for task_id, label in zip(b["task_id"], b["label"]):
                    optimizer_target_sequence.update((task_id + "\0").encode())
                    optimizer_target_sequence.update(
                        label.detach().cpu().contiguous().numpy().tobytes())
            imgs = b["image"].to(dev)
            batch_tasks = sorted(set(b["task_id"]))
            if marginal_mode and len(batch_tasks) != 1:
                raise RuntimeError(f"marginal KL requires task-pure batches, got {batch_tasks}")
            for tid in batch_tasks:
                ix = [i for i, t in enumerate(b["task_id"]) if t == tid]
                hm = torch.stack([b["heatmap"][i] for i in ix], 0).to(dev)
                raw_pred = model(imgs[ix], task_id=tid)
                pred = torch.sigmoid(raw_pred)
                loss = select_task_loss(tid, pred, hm, aop_p3_weight=aop_p3_weight,
                                         param_loss_beta=param_loss_beta,
                                         param_loss_temperature=param_loss_temperature,
                                         heatmap_loss=heatmap_loss,
                                         awing_alpha=awing_alpha, awing_omega=awing_omega,
                                         awing_epsilon=awing_epsilon,
                                         awing_theta=awing_theta,
                                         marginal_kl_beta=(marginal_kl_betas[tid]
                                                           if marginal_mode else None),
                                         fa_wall_weight=fa_wall_weight)
                if model_variant == "coordse":
                    coords, _ = model.dsnt_modules[tid](raw_pred)
                    target_coords = torch.stack([b["label"][i] for i in ix], 0).to(dev)
                    target_coords = target_coords.reshape(-1, hm.shape[1], 2)
                    target_coords = target_coords.mul(2.0).sub(1.0)
                    loss = loss + coordse_coord_beta * F.mse_loss(coords, target_coords)
                opt.zero_grad()
                loss.backward()
                opt.step()
                optimizer_steps += 1
                task_optimizer_steps[tid] += 1
                optimizer_task_sequence.update((tid + "\n").encode())
                if ema is not None:
                    # N=1 is exact per-step EMA. Larger N is an optional sampled approximation;
                    # exponentiating decay preserves the requested per-step time constant.
                    if optimizer_steps % ema_update_every == 0:
                        elapsed = optimizer_steps - last_ema_step
                        ema.update(model, decay=ema_decay ** elapsed)
                        last_ema_step = optimizer_steps
                loss_val = float(loss.item())
                ep_loss += loss_val; nb += 1
                if tid not in task_losses:
                    task_losses[tid] = []
                task_losses[tid].append(loss_val)
        if sched is not None:
            sched.step()

        mean_loss = ep_loss / max(nb, 1)
        per_task_train_loss = {tid: float(np.mean(v)) for tid, v in task_losses.items()}
        current_lr = opt.param_groups[0]["lr"]

        # Bring EMA exactly to this epoch boundary before any staged checkpoint is saved.
        if ema is not None and optimizer_steps > last_ema_step:
            elapsed = optimizer_steps - last_ema_step
            ema.update(model, decay=ema_decay ** elapsed)
            last_ema_step = optimizer_steps

        # Build per-task loss string for log line
        task_loss_str = " ".join(f"{tid}={v:.4f}" for tid, v in sorted(per_task_train_loss.items()))
        print(f"fold {fold} epoch {ep + 1}/{epochs} done | mean_loss {mean_loss:.5f} "
              f"| lr {current_lr:.2e} | per_task: [{task_loss_str}]")

        if (ep + 1) in save_at:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ckpt_ep{ep + 1}.pth"))
            print(f"  [save] {ckpt_dir}/ckpt_ep{ep + 1}.pth")
            if ema is not None:
                torch.save(ema.state_dict(), os.path.join(ckpt_dir, f"ema_ep{ep + 1}.pth"))
                print(f"  [save] {ckpt_dir}/ema_ep{ep + 1}.pth")

        # Val eval
        val_avg_mre, val_per_task_mre = None, None
        do_val = (val_every > 0) and (not full_data) and ((ep + 1) % val_every == 0)
        if do_val:
            # Save current weights to a temp checkpoint for predict
            tmp_ckpt = os.path.join(ckpt_dir, "_val_tmp_model.pth")
            torch.save(model.state_dict(), tmp_ckpt)
            val_avg_mre, val_per_task_mre = _run_val_eval(
                tmp_ckpt, ckpt_dir, fold, encoder, input_size, mem_frac, heatmap_size,
                model_variant=model_variant)
            try:
                os.remove(tmp_ckpt)
            except OSError:
                pass
            if val_avg_mre is not None:
                print(f"  [val-eval] epoch {ep + 1} val_avg_mre={val_avg_mre:.4f}")
                # Update best-by-val checkpoint
                if best_val_mre is None or val_avg_mre < best_val_mre:
                    best_val_mre = val_avg_mre
                    # Save best-by-val from the current model state directly
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_by_val.pth"))
                    print(f"  [val-eval] new best_by_val.pth (mre={best_val_mre:.4f})")

        record = build_metrics_record(
            epoch=ep + 1,
            train_loss=mean_loss,
            per_task_train_loss=per_task_train_loss,
            lr=current_lr,
            val_avg_mre=val_avg_mre,
            val_per_task_mre=val_per_task_mre,
        )
        val_history.append(record)
        metrics_fh.write(json.dumps(record) + "\n")
        metrics_fh.flush()

    metrics_fh.close()
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(ckpt_dir, "ema_model.pth"))
    if marginal_mode:
        manifest = {
            "mode": "marginal_kl_head_only_continuation",
            "seed": seed,
            "epochs": epochs,
            "head_lr": head_lr,
            "fold": fold,
            "full_data": full_data,
            "aug": aug,
            "encoder": encoder,
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "sigma": sigma,
            "batch_size": 4,
            "num_workers": 4,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "cosine": cosine,
            "warmup": warmup,
            "ema_decay": ema_decay,
            "val_every": val_every,
            "init_checkpoint": init_checkpoint,
            "init_checkpoint_sha256": _file_sha256(init_checkpoint),
            "folds_csv": folds_csv,
            "folds_csv_sha256": _file_sha256(folds_csv),
            "train_examples": len(train_idx),
            "marginal_kl_betas": marginal_kl_betas,
            "task_optimizer_steps": task_optimizer_steps,
            "optimizer_task_sequence_sha256": optimizer_task_sequence.hexdigest(),
            "optimizer_target_sequence_sha256": optimizer_target_sequence.hexdigest(),
            "encoder_eval": True,
        }
        with open(os.path.join(ckpt_dir, "training_manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    else:
        # General provenance for ordinary CV/full-data runs. In particular this records the exact
        # continued-pretraining encoder hash; without it, a final checkpoint cannot prove which
        # initialization it actually used after supervised fine-tuning overwrites the encoder.
        manifest = {
            "mode": "full_data" if full_data else "cross_validation",
            "seed": seed,
            "epochs": epochs,
            "head_lr": head_lr,
            "fold": fold,
            "full_data": full_data,
            "aug": aug,
            "encoder": encoder,
            "encoder_init": encoder_init,
            "encoder_init_sha256": _file_sha256(encoder_init) if encoder_init else None,
            "init_checkpoint": init_checkpoint,
            "init_checkpoint_sha256": _file_sha256(init_checkpoint) if init_checkpoint else None,
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "sigma": sigma,
            "batch_size": 4,
            "num_workers": 4,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "cosine": cosine,
            "warmup": warmup,
            "ema_decay": ema_decay,
            "val_every": val_every,
            "folds_csv": folds_csv,
            "folds_csv_sha256": _file_sha256(folds_csv),
            "train_examples": len(train_idx),
            "train_task": train_task,
            "freeze_encoder": freeze_encoder,
            "unfreeze_last_blocks": unfreeze_last_blocks,
            "encoder_refine_lr": encoder_refine_lr,
            "model_variant": model_variant,
            "task_optimizer_steps": task_optimizer_steps,
            "optimizer_task_sequence_sha256": optimizer_task_sequence.hexdigest(),
            "optimizer_target_sequence_sha256": optimizer_target_sequence.hexdigest(),
            "metrics_sha256": _file_sha256(os.path.join(ckpt_dir, "metrics.jsonl")),
        }
        with open(os.path.join(ckpt_dir, "training_manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return ckpt_dir, sorted(val_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-csv", default=os.path.join(PROJ, "data", "folds", "folds.csv"))
    ap.add_argument("--fold", type=int, default=None,
                    help="CV fold to hold out and score (omit when using --full-data)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--aug", default="none",
                    choices=["none", "photo_v1", "geo_v1", "aop_robust_v1", "geo_v1_hcsmall"])
    ap.add_argument("--mem-frac", type=float, default=0.30)
    ap.add_argument("--encoder", default="dinov2_vits",
                    choices=["dinov2_vits", "dinov2_vitb", "dinov2_vitb_fuse4", "dinov2_vitl", "beit_imagenet", "usfm_beit", "dinov3_vits", "dinov3_vitb"],
                    help="Encoder backbone to use (default: dinov2_vits)")
    ap.add_argument("--model-variant", default="base", choices=["base", "coordse"],
                    help="Task-head/decoder variant; coordse adds coordinate-grid/SE heads and DSNT.")
    ap.add_argument("--coordse-coord-beta", type=float, default=1.0,
                    help="Auxiliary DSNT coordinate-loss weight for --model-variant coordse.")
    ap.add_argument("--input-size", type=int, default=518,
                    help="Square input image size in pixels (default: 518 for DINOv2 ViT-S/14)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Freeze encoder parameters (only train heads)")
    ap.add_argument("--encoder-init", default=None,
                    help="Path to a matching custom DINOv2 backbone state_dict to init the encoder "
                         "(Axis A: continued-DINO weights; dinov2_vits or dinov2_vitb)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="Load a complete trained model state_dict before optimization. Required "
                         "for --train-task checkpoint refinement.")
    ap.add_argument("--train-task", default=None,
                    help="Strict task-head-only refinement: filter training samples to this task "
                         "and freeze the encoder plus every other head. Requires --init-checkpoint.")
    ap.add_argument("--unfreeze-last-blocks", type=int, default=0,
                    help="With --train-task, additionally refine this many final transformer blocks "
                         "plus the final encoder norm. The resulting checkpoint is a task-specific "
                         "branch and must be routed only for that task.")
    ap.add_argument("--encoder-refine-lr", type=float, default=2e-6,
                    help="Encoder LR for --unfreeze-last-blocks (default 2e-6).")
    ap.add_argument("--head-lr", type=float, default=1e-3,
                    help="Learning rate for task heads (default 1e-3, preserving historical runs). "
                         "Checkpoint refinement should normally use a smaller value such as 1e-4.")
    ap.add_argument("--train-fusion-only", action="store_true",
                    help="Load a non-fusion checkpoint into dinov2_vitb_fuse4, freeze every "
                         "historical tensor, and train only its zero-init residual fusion adapter.")
    ap.add_argument("--fusion-lr", type=float, default=1e-4,
                    help="Learning rate for --train-fusion-only (default: 1e-4).")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="Track and save an independently scored model-state EMA. 0 disables it; "
                         "0.9999 is the fixed screening value.")
    ap.add_argument("--ema-update-every", type=int, default=1,
                    help="Sample the raw state every N optimizer steps for EMA (default 1, exact); "
                         "the applied decay is exponentiated to preserve the per-step time constant.")
    ap.add_argument("--full-data", action="store_true",
                    help="Train on ALL fold>=0 images (no held-out val/scoring) for the "
                         "final/deployment model; requires --run-name.")
    ap.add_argument("--exclude-train-paths-file", default=None,
                    help="Text file of image_path values to exclude from training only.")
    ap.add_argument("--run-name", default=None,
                    help="Checkpoint subdir under runs/ (required with --full-data; also usable in fold mode).")
    ap.add_argument("--save-epochs", default=None,
                    help="Comma-sep epochs at which to ALSO save runs/<name>/ckpt_ep<N>.pth (convergence probe).")
    ap.add_argument("--cosine", action="store_true",
                    help="Use CosineAnnealingLR (T_max=epochs) instead of constant LR.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="Number of epochs for linear LR warmup (default: 0 = off). "
                         "Composable with --cosine: warmup then cosine annealing.")
    ap.add_argument("--val-every", type=int, default=0,
                    help="Run held-out val eval every N epochs (default: 0 = off). "
                         "FOLD mode only (skipped in --full-data). Writes metrics.jsonl "
                         "with val_avg_mre + per_task_mre; saves best_by_val.pth.")
    ap.add_argument("--heatmap-size", type=int, default=64,
                    help="Square output heatmap resolution (default 64). Finer (e.g. 128) = a "
                         "precision lever (the ViT-S@518 head decodes a native 148 grid). For "
                         "!=64 the Model().predict argmax proxy is skipped (it hardcodes 64) -> "
                         "score with infer_tta.py --heatmap-size N.")
    ap.add_argument("--sigma", type=float, default=1.8,
                    help="Gaussian target sigma in heatmap CELLS (default 1.8). Smaller = sharper "
                         "peak = more precise localisation (helps precise 2-pt tasks, can hurt "
                         "multi-landmark). Probe lever to disentangle the HM=128 FUGC win.")
    ap.add_argument("--fugc-heatmap-size", type=int, default=None,
                    help="Per-task override: FUGC output heatmap resolution (default = --heatmap-size). "
                         "e.g. 128 gives FUGC a finer grid while other tasks stay at --heatmap-size "
                         "(assumed 64); builds a per-task model. The precise 2-pt FUGC benefits; the "
                         "multi-landmark tasks are unaffected (they stay at 64).")
    ap.add_argument("--femur-heatmap-size", type=int, default=None,
                    help="Per-task override: fetal_femur output heatmap resolution (default = "
                         "--heatmap-size). Like --fugc-heatmap-size and COMBINABLE with it. femur is "
                         "the other precise 2-pt task (HM=128 probe: femur -2.1); multi-landmark tasks "
                         "stay at --heatmap-size. Omit (the default) to leave the FUGC-only path "
                         "byte-identical.")
    ap.add_argument("--aop-p3-weight", type=float, default=1.0,
                    help="AOP-only per-landmark loss weight on p3 (LM4, the weak ray endpoint). "
                         "1.0 = uniform (current behaviour, byte-identical). >1 upweights p3 via a "
                         "normalized weighted MSE [1,1,1,W]; other tasks AND other AOP landmarks "
                         "are unaffected. Diagnostic: p3 is 2.1x worse than the vertex = the "
                         "dominant AOP-MRE source. Lever A.")
    ap.add_argument("--aop-heatmap-size", type=int, default=None,
                    help="Per-task override: AOP output heatmap resolution (default = --heatmap-size). "
                         "Like --fugc-heatmap-size; builds a per-task model with AOP on a finer grid "
                         "(precision probe B). Other tasks stay at --heatmap-size. Score with "
                         "infer_tta / --val-every (per-task model skips the argmax proxy).")
    ap.add_argument("--fugc-sigma", type=float, default=None,
                    help="Per-task override: FUGC Gaussian target sigma in heatmap CELLS (default = "
                         "--sigma). Use WITH --fugc-heatmap-size to keep the PHYSICAL peak width "
                         "constant on the finer grid, e.g. --fugc-heatmap-size 128 --fugc-sigma 3.6 "
                         "(= 1.8 * 128/64). Removes the HM=128 unscaled-sharpness confound; a "
                         "training-target-only quantity (NOT needed at inference). Other tasks use "
                         "--sigma unchanged; omit to leave the scalar path byte-identical.")
    ap.add_argument("--femur-sigma", type=float, default=None,
                    help="Per-task override: fetal_femur Gaussian target sigma in heatmap CELLS "
                         "(default = --sigma). Like --fugc-sigma and COMBINABLE with it; pair with "
                         "--femur-heatmap-size to hold the physical peak width fixed.")
    ap.add_argument("--aop-sigma", type=float, default=None,
                    help="Per-task override: AOP Gaussian target sigma in heatmap CELLS (default = "
                         "--sigma). Like --fugc-sigma; pair with --aop-heatmap-size to hold the "
                         "physical peak width fixed on AOP's finer grid.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed (default 42). Vary across ensemble members for diversity.")
    ap.add_argument("--param-loss-beta", type=float, default=0.0,
                    help="IVC/HC-only differentiable downstream-parameter loss weight (probe "
                         "lever, distinct from --aop-p3-weight). L = 1.0*heatmap_mse + beta*"
                         "param_loss, where param_loss = MSE(derived_param(pred), "
                         "derived_param(gt)) via a differentiable soft-argmax decode (IVC: "
                         "distance; HC: ellipse circumference). 0.0 (default) = OFF, "
                         "byte-identical to plain F.mse_loss for ALL tasks including IVC/HC. "
                         "Other 7 tasks are NEVER affected regardless of this value.")
    ap.add_argument("--param-loss-temperature", type=float, default=1.0,
                    help="Softmax temperature for the param-loss soft-argmax decode (only used "
                         "when --param-loss-beta != 0). Lower = peakier/closer to hard-argmax.")
    ap.add_argument("--heatmap-loss", choices=["mse", "adaptive_wing"], default="mse",
                    help="Global heatmap-regression objective. Default mse preserves the adopted "
                         "training path; adaptive_wing uses the core ICCV 2019 Eq. (3) only.")
    ap.add_argument("--awing-alpha", type=float, default=2.1)
    ap.add_argument("--awing-omega", type=float, default=14.0)
    ap.add_argument("--awing-epsilon", type=float, default=1.0)
    ap.add_argument("--awing-theta", type=float, default=0.5)
    ap.add_argument("--marginal-kl-betas", default=None,
                    help="JSON file containing fixed per-task marginal-KL betas. This is a "
                         "head-only checkpoint continuation: requires --freeze-encoder and "
                         "--init-checkpoint, asserts task-pure batches, and keeps the encoder "
                         "in eval mode. A file of all-zero betas is the exact matched control.")
    ap.add_argument("--fa-wall-weight", type=float, default=1.0,
                    help="FA-only per-landmark loss weight on p2/p3 (the lateral-wall landmarks). "
                         "1.0 = uniform (current behaviour, byte-identical). >1 upweights p2/p3 via "
                         "a normalized weighted MSE [1,1,W,W]; other tasks AND p0/p1 are unaffected. "
                         "Diagnosis (2026-08-07): FA param-MAE error is unbiased size variance "
                         "concentrated in the along-axis (x) error of p2/p3. Requires --train-task FA.")
    ap.add_argument("--fa-aniso-sigma", default=None,
                    help="FA-only per-landmark ANISOTROPIC Gaussian target 'SX,SY' (heatmap CELLS) "
                         "for p2/p3 (the lateral-wall landmarks); p0/p1 stay isotropic at --sigma. "
                         "None (default) = OFF, byte-identical isotropic rendering for every task. "
                         "Requires --train-task FA and --aug photo_v1 or a geometric pack.")
    args = ap.parse_args()
    marginal_kl_betas = None
    fa_aniso_sigma = None
    try:
        if args.marginal_kl_betas is not None:
            marginal_kl_betas = load_marginal_kl_betas(args.marginal_kl_betas)
        fa_aniso_sigma = parse_fa_aniso_sigma(args.fa_aniso_sigma)
        validate_loss_config(args.heatmap_loss, args.aop_p3_weight, args.param_loss_beta,
                             marginal_kl_betas=marginal_kl_betas, fa_wall_weight=args.fa_wall_weight)
        validate_fa_config(args.train_task, args.aug, fa_wall_weight=args.fa_wall_weight,
                           fa_aniso_sigma=fa_aniso_sigma)
        # Validate Adaptive Wing hyperparameters before any dataset/model/GPU setup.
        if args.heatmap_loss == "adaptive_wing":
            adaptive_wing_loss(torch.zeros(1), torch.zeros(1), alpha=args.awing_alpha,
                               omega=args.awing_omega, epsilon=args.awing_epsilon,
                               theta=args.awing_theta)
    except ValueError as exc:
        ap.error(str(exc))
    hm = (args.heatmap_size, args.heatmap_size)
    per_task_hm = {}
    if args.fugc_heatmap_size and args.fugc_heatmap_size != args.heatmap_size:
        per_task_hm["FUGC"] = (args.fugc_heatmap_size, args.fugc_heatmap_size)
    if args.femur_heatmap_size and args.femur_heatmap_size != args.heatmap_size:
        per_task_hm["fetal_femur"] = (args.femur_heatmap_size, args.femur_heatmap_size)
    if args.aop_heatmap_size and args.aop_heatmap_size != args.heatmap_size:
        per_task_hm["AOP"] = (args.aop_heatmap_size, args.aop_heatmap_size)
    if per_task_hm:
        hm = per_task_hm  # listed tasks -> finer grid; all others -> (heatmap_size) via hm_for default
    per_task = isinstance(hm, dict)
    # Per-task sigma (training-target-only): scale sigma with the finer grid to hold the physical
    # peak width constant. Only listed tasks differ; all others resolve to --sigma via sigma_for.
    per_task_sigma = {}
    if args.fugc_sigma is not None and args.fugc_sigma != args.sigma:
        per_task_sigma["FUGC"] = args.fugc_sigma
    if args.femur_sigma is not None and args.femur_sigma != args.sigma:
        per_task_sigma["fetal_femur"] = args.femur_sigma
    if args.aop_sigma is not None and args.aop_sigma != args.sigma:
        per_task_sigma["AOP"] = args.aop_sigma
    sigma = per_task_sigma if per_task_sigma else args.sigma  # dict -> per-task; else scalar (byte-identical)
    save_ep = [int(x) for x in args.save_epochs.split(",")] if args.save_epochs else None

    if args.full_data:
        if not args.run_name:
            ap.error("--full-data requires --run-name")
        run, _ = train_fold(args.folds_csv, fold=-1, epochs=args.epochs, seed=args.seed,
                            mem_frac=args.mem_frac, aug=args.aug,
                            encoder=args.encoder, input_size=args.input_size,
                            freeze_encoder=args.freeze_encoder, encoder_init=args.encoder_init,
                            init_checkpoint=args.init_checkpoint, train_task=args.train_task,
                            full_data=True, ckpt_name=args.run_name,
                            save_epochs=save_ep, cosine=args.cosine,
                            warmup=args.warmup, val_every=0, heatmap_size=hm, sigma=sigma,
                            aop_p3_weight=args.aop_p3_weight,
                            param_loss_beta=args.param_loss_beta,
                            param_loss_temperature=args.param_loss_temperature,
                            head_lr=args.head_lr, heatmap_loss=args.heatmap_loss,
                            awing_alpha=args.awing_alpha, awing_omega=args.awing_omega,
                            awing_epsilon=args.awing_epsilon, awing_theta=args.awing_theta,
                            train_fusion_only=args.train_fusion_only,
                            fusion_lr=args.fusion_lr, ema_decay=args.ema_decay,
                            ema_update_every=args.ema_update_every,
                            unfreeze_last_blocks=args.unfreeze_last_blocks,
                            encoder_refine_lr=args.encoder_refine_lr,
                            marginal_kl_betas=marginal_kl_betas,
                            exclude_train_paths_file=args.exclude_train_paths_file,
                            model_variant=args.model_variant,
                            coordse_coord_beta=args.coordse_coord_beta,
                            fa_wall_weight=args.fa_wall_weight,
                            fa_aniso_sigma=fa_aniso_sigma)
        print(f"[run_config] full-data model trained on all fold>=0 images -> "
              f"{run}/best_model.pth (no held-out val/scoring; predict the official val "
              "set with scripts/predict_val.py --encoder ... --input-size ...).")
        return

    if args.fold is None:
        ap.error("--fold is required unless --full-data is set")

    run, val_paths = train_fold(args.folds_csv, args.fold, args.epochs, seed=args.seed,
                                mem_frac=args.mem_frac, aug=args.aug,
                                encoder=args.encoder, input_size=args.input_size,
                                freeze_encoder=args.freeze_encoder,
                                encoder_init=args.encoder_init,
                                init_checkpoint=args.init_checkpoint, train_task=args.train_task,
                                ckpt_name=args.run_name,
                                save_epochs=save_ep, cosine=args.cosine,
                                warmup=args.warmup, val_every=args.val_every, heatmap_size=hm, sigma=sigma,
                                aop_p3_weight=args.aop_p3_weight,
                                param_loss_beta=args.param_loss_beta,
                                param_loss_temperature=args.param_loss_temperature,
                                head_lr=args.head_lr, heatmap_loss=args.heatmap_loss,
                                awing_alpha=args.awing_alpha, awing_omega=args.awing_omega,
                                awing_epsilon=args.awing_epsilon, awing_theta=args.awing_theta,
                                train_fusion_only=args.train_fusion_only,
                                fusion_lr=args.fusion_lr, ema_decay=args.ema_decay,
                                ema_update_every=args.ema_update_every,
                                unfreeze_last_blocks=args.unfreeze_last_blocks,
                                encoder_refine_lr=args.encoder_refine_lr,
                                marginal_kl_betas=marginal_kl_betas,
                                exclude_train_paths_file=args.exclude_train_paths_file,
                                model_variant=args.model_variant,
                                coordse_coord_beta=args.coordse_coord_beta,
                                fa_wall_weight=args.fa_wall_weight,
                                fa_aniso_sigma=fa_aniso_sigma)
    split = os.path.join(PROJ, "data", f"_cvfold{args.fold}_val.csv")
    pd.DataFrame({"image_path": val_paths}).to_csv(split, index=False)
    out = os.path.join(PROJ, "submission", f"cvfold{args.fold}")

    if args.encoder == "dinov2_vits" and args.heatmap_size == 64 and not per_task:
        # Default path: use the baseline Model().predict (hardcodes ViT-S/518 + argmax decode).
        # This is a quick score proxy; production scoring uses infer_tta.py.
        os.chdir(run)  # model.py loads best_model.pth from CWD
        Model().predict(data_root=os.path.join(PROJ, "data"), output_dir=out, batch_size=8,
                        split_csv=split)
        os.chdir(PROJ)
    else:
        # Non-default encoder: skip the argmax proxy — Model() hardcodes ViT-S/518 and would
        # produce wrong predictions. Use infer_tta.py for scoring (run separately).
        print(f"[run_config] encoder={args.encoder!r}: skipping Model().predict argmax proxy. "
              "Run experiments/infer_tta.py --encoder ... --input-size ... to score this fold.")
        os.makedirs(out, exist_ok=True)

    gt = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(PROJ, "data/csv/*.csv"))],
                   ignore_index=True)
    gt = gt[gt.image_path.isin(set(val_paths))]
    gtp = os.path.join(PROJ, "data", f"_cvfold{args.fold}_gt.csv")
    gt.to_csv(gtp, index=False)

    if args.encoder == "dinov2_vits" and args.heatmap_size == 64 and not per_task:
        res = scorer.score_submission(os.path.join(out, "regression_predictions.json"), gtp)
        os.makedirs(os.path.join(PROJ, "experiments", "results"), exist_ok=True)
        json.dump(res, open(os.path.join(PROJ, "experiments", "results", f"cvfold{args.fold}.json"), "w"), indent=2)
        print(json.dumps({k: res[k] for k in ("avg_mre", "avg_param_mae", "total_missing")}, indent=2))


if __name__ == "__main__":
    main()
