"""Axis A (DINO formulation) — continued DINO self-distillation pretraining of timm ViT-S/14 or
ViT-B/14 DINOv2 on the 24k in-domain US pool, then hand off the adapted encoder to fine-tuning via
experiments/encoders.py's --encoder-init (unchanged mechanism).

CORRECTS the prior Axis-A attempt's objective mismatch: DINOv2's own pretraining recipe is DINO
self-distillation (+ iBOT patch MIM), not masked-image-modeling (MAE) — continuing THAT objective
is the more faithful "continued pretrain" of this exact backbone. (The internal SSL literature
review that motivated Axis A is not part of the public release.)

HONEST SCOPE (stated explicitly, not buried): CLS-token multi-crop DINO loss ONLY. No iBOT
patch-level loss, no KoLeo regularizer. This is an accepted simplification given the compliance-
hedge framing (genuine, defensible, non-broken SSL run on our own unlabeled pool — not a scoring
lever that needs the full DINOv2 recipe to move the needle).

Adaptation: PARTIAL UNFREEZE of the last --unfreeze-blocks transformer blocks (+ final norm) only
(see experiments/dino_model.py docstring for the risk rationale vs LoRA/full-FT).

VALIDATE-BEFORE-SCALING: pass --steps to cap a micro-run; check loss trend + the frozen-feature
drift control (experiments/dino_feature_drift.py) BEFORE committing the full pretrain budget —
this is the check that would have caught Axis A's MAE-decoder-clobbers-encoder bug even earlier,
and the one that guards against a repeat of the 1.3x-drift-vs-0.18-noise-floor finding.

Run with the BASELINE venv:
  CUDA_VISIBLE_DEVICES=0 baseline/.venv-baseline/bin/python experiments/dino_pretrain.py \
      --epochs 3 --batch-size 24 --out runs/dino_ssl --mem-frac 0.30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from experiments.dino_model import DinoStudentTeacher          # noqa: E402
from experiments.dino_augment import build_multicrop_transform  # noqa: E402
from experiments.dino_loss import cosine_momentum_schedule, dino_loss, update_center  # noqa: E402
from utils import set_seed                                      # noqa: E402


ENCODER_MODELS = {
    "dinov2_vits": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb": "vit_base_patch14_dinov2.lvd142m",
}


class UnlabeledPoolMultiCrop(Dataset):
    """The 24k unlabeled US pool, multi-crop augmented per DINO."""

    def __init__(self, manifest_csv: str, pool_root: str, n_global: int, n_local: int,
                global_size: int, local_size: int):
        self.df = pd.read_csv(manifest_csv)
        self.root = pool_root
        self.tfm = build_multicrop_transform(n_global, n_local, global_size, local_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        rel = self.df.iloc[idx]["image_path"]
        path = os.path.join(self.root, rel)
        img = cv2.imread(path)
        if img is None:                          # skip unreadable -> next (rare, mirrors mae_pretrain)
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tfm(img)                     # list[Tensor], global crops first


def multicrop_collate(batch):
    """list[sample] where sample = list[n_crops Tensors] -> list[n_crops Tensor[B,3,H,W]]."""
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(n_crops)]


def _cosine_lr(step, total, base_lr, warmup, min_lr=1e-6):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(PROJ, "data", "unlabeled", "manifest.csv"))
    ap.add_argument("--pool-root", default=os.path.join(PROJ, "data", "unlabeled"))
    ap.add_argument("--global-size", type=int, default=518)
    ap.add_argument("--local-size", type=int, default=168)
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=4)
    ap.add_argument("--unfreeze-blocks", type=int, default=2,
                    help="partial-unfreeze: only the last N transformer blocks (+final norm) train")
    ap.add_argument("--encoder", choices=sorted(ENCODER_MODELS), default="dinov2_vits",
                    help="DINOv2 backbone to continue (default: dinov2_vits)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help="cap total optimizer steps (0 = no cap; "
                    "use a small value for the validate-before-scaling micro-run)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--center-momentum", type=float, default=0.9)
    ap.add_argument("--teacher-momentum-base", type=float, default=0.996)
    ap.add_argument("--head-out-dim", type=int, default=4096)
    ap.add_argument("--head-hidden-dim", type=int, default=2048)
    ap.add_argument("--head-bottleneck-dim", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--mem-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(PROJ, "runs", "dino_ssl"))
    ap.add_argument("--results-json",
                    default=os.path.join(PROJ, "experiments", "results", "dino_ssl", "pretrain_log.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UnlabeledPoolMultiCrop(args.manifest, args.pool_root, args.n_global, args.n_local,
                               args.global_size, args.local_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=False,
                        collate_fn=multicrop_collate)
    print(f"[dino] pool={len(ds)} images, global={args.global_size}x{args.n_global}, "
          f"local={args.local_size}x{args.n_local}, bs={args.batch_size}, steps/epoch={len(loader)}")

    model = DinoStudentTeacher(
        input_size=args.global_size, unfreeze_blocks=args.unfreeze_blocks,
        encoder_name=ENCODER_MODELS[args.encoder],
        head_out_dim=args.head_out_dim, head_hidden_dim=args.head_hidden_dim,
        head_bottleneck_dim=args.head_bottleneck_dim, pretrained=True,
    ).to(dev)
    n_trainable = sum(p.numel() for p in model.student_encoder.trainable_parameters())
    n_total = sum(p.numel() for p in model.student_encoder.parameters())
    print(f"[dino] encoder trainable params: {n_trainable}/{n_total} "
          f"({100*n_trainable/n_total:.1f}%, last {args.unfreeze_blocks} blocks)")

    trainable = model.student_encoder.trainable_parameters() + list(model.student_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = args.steps if args.steps > 0 else args.epochs * len(loader)
    center = torch.zeros(args.head_out_dim, device=dev)
    log = {"args": vars(args), "total_steps": total_steps, "loss": [], "step": [], "lr": [],
          "momentum": []}
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)

    model.train()
    step = 0
    t0 = time.time()
    ema_loss = None
    stop = False
    for ep in range(args.epochs):
        for crops in loader:
            crops = [c.to(dev, non_blocking=True) for c in crops]
            lr = _cosine_lr(step, total_steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            momentum = cosine_momentum_schedule(step, total_steps, args.teacher_momentum_base)

            with torch.amp.autocast("cuda", enabled=use_amp):
                student_out = torch.stack([model.student_forward(c) for c in crops], dim=0)
                with torch.no_grad():
                    teacher_out = torch.stack(
                        [model.teacher_forward(c) for c in crops[:args.n_global]], dim=0
                    )
                loss = dino_loss(student_out, teacher_out, args.student_temp, args.teacher_temp,
                                 center, args.n_global)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 3.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                center = update_center(center, teacher_out.float(), args.center_momentum)
                model.ema_update(momentum)

            lv = loss.item()
            ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"[dino] step {step}/{total_steps} ep {ep} loss {lv:.4f} ema {ema_loss:.4f} "
                      f"lr {lr:.2e} momentum {momentum:.5f} ({dt:.0f}s)")
                log["step"].append(step)
                log["loss"].append(lv)
                log["lr"].append(lr)
                log["momentum"].append(momentum)
                json.dump(log, open(args.results_json, "w"), indent=2)
            step += 1
            if args.steps > 0 and step >= args.steps:
                stop = True
                break
        if stop:
            break

    enc_path = os.path.join(args.out, "encoder.pth")
    torch.save(model.encoder_state_dict(), enc_path)
    full_path = os.path.join(args.out, "full_dino.pth")
    torch.save(model.state_dict(), full_path)
    log["final_loss"] = ema_loss
    log["wall_seconds"] = time.time() - t0
    log["steps_run"] = step
    log["encoder_path"] = enc_path
    json.dump(log, open(args.results_json, "w"), indent=2)
    print(f"[dino] DONE — {step} steps, ema_loss {ema_loss:.4f}, {log['wall_seconds']:.0f}s; "
          f"saved encoder -> {enc_path}")
    print(f"[dino] log -> {args.results_json}")


if __name__ == "__main__":
    main()
