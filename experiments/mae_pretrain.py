"""Axis A — MAE continued-pretraining of timm ViT-S/14 DINOv2 on the 24k US pool.

Reads data/unlabeled/manifest.csv, builds a Dataset that mirrors the fine-tune preprocessing
(cv2 -> RGB -> resize to input_size -> ImageNet Normalize -> CHW), and trains the
MaskedAutoencoderViT (encoder init from DINOv2). Logs recon loss; saves the ViT-S backbone
state_dict (decoder discarded) as the fine-tune init.

VALIDATE-BEFORE-SCALING: pass --steps to cap a micro-run and confirm the recon loss drops
meaningfully (not flat/NaN) BEFORE committing the full pretraining budget.

Run with the BASELINE venv:
  baseline/.venv-baseline/bin/python experiments/mae_pretrain.py --epochs 3 --batch-size 32 \
      --out runs/axis_a_mae --mem-frac 0.30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from experiments.mae_model import MaskedAutoencoderViT  # noqa: E402
from utils import set_seed                               # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class UnlabeledPool(Dataset):
    """The 24k unlabeled US pool. Preprocessing mirrors the fine-tune dataset exactly so the
    encoder sees the same input distribution at pretrain and fine-tune time."""

    def __init__(self, manifest_csv: str, pool_root: str, input_size: int):
        self.df = pd.read_csv(manifest_csv)
        self.root = pool_root
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rel = self.df.iloc[idx]["image_path"]
        path = os.path.join(self.root, rel)
        img = cv2.imread(path)
        if img is None:                          # skip unreadable -> next (rare)
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1))   # CHW


def _cosine_lr(step, total, base_lr, warmup, min_lr=1e-6):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(PROJ, "data", "unlabeled", "manifest.csv"))
    ap.add_argument("--pool-root", default=os.path.join(PROJ, "data", "unlabeled"))
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help="cap total optimizer steps (0 = no cap; "
                    "use a small value for the validate-before-scaling micro-run)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--mem-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(PROJ, "runs", "axis_a_mae"))
    ap.add_argument("--results-json",
                    default=os.path.join(PROJ, "experiments", "results", "axis_a", "pretrain_log.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UnlabeledPool(args.manifest, args.pool_root, args.input_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    print(f"[mae] pool={len(ds)} images, input={args.input_size}, bs={args.batch_size}, "
          f"steps/epoch={len(loader)}")

    model = MaskedAutoencoderViT(
        input_size=args.input_size, mask_ratio=args.mask_ratio, pretrained=True,
    ).to(dev)
    print(f"[mae] grid={model.grid}x{model.grid}={model.num_patches} patches, "
          f"mask_ratio={args.mask_ratio} -> ~{int(model.num_patches*args.mask_ratio)} masked/img")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = args.steps if args.steps > 0 else args.epochs * len(loader)
    log = {"args": vars(args), "total_steps": total_steps, "loss": [], "step": [], "lr": []}
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)

    model.train()
    step = 0
    t0 = time.time()
    ema_loss = None
    stop = False
    for ep in range(args.epochs):
        for imgs in loader:
            imgs = imgs.to(dev, non_blocking=True)
            lr = _cosine_lr(step, total_steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _, _ = model(imgs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            lv = loss.item()
            ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"[mae] step {step}/{total_steps} ep {ep} loss {lv:.4f} "
                      f"ema {ema_loss:.4f} lr {lr:.2e} ({dt:.0f}s)")
                log["step"].append(step)
                log["loss"].append(lv)
                log["lr"].append(lr)
                json.dump(log, open(args.results_json, "w"), indent=2)
            step += 1
            if args.steps > 0 and step >= args.steps:
                stop = True
                break
        if stop:
            break

    # save the ViT-S backbone (decoder discarded) — the fine-tune init
    enc_path = os.path.join(args.out, "encoder.pth")
    torch.save(model.encoder_state_dict(), enc_path)
    # also save the FULL MAE (encoder+decoder) for the reconstruction sanity montage
    full_path = os.path.join(args.out, "full_mae.pth")
    torch.save(model.state_dict(), full_path)
    log["final_loss"] = ema_loss
    log["wall_seconds"] = time.time() - t0
    log["steps_run"] = step
    log["encoder_path"] = enc_path
    json.dump(log, open(args.results_json, "w"), indent=2)
    print(f"[mae] DONE — {step} steps, ema_loss {ema_loss:.4f}, "
          f"{log['wall_seconds']:.0f}s; saved encoder -> {enc_path}")
    print(f"[mae] log -> {args.results_json}")


if __name__ == "__main__":
    main()
