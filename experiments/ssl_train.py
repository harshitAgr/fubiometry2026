# experiments/ssl_train.py
"""Mean-teacher SSL fold trainer (Lever 2B): the de-leaked supervised fold loop
(experiments/run_config.py) + an EMA teacher + a confidence-gated consistency loss on the 2A
unlabeled pool (task-balanced, per-fold phash-leakage-filtered). Same 20-epoch budget as the
baseline checkpoints (clean A/B). Saves runs/ssl_cvfold{K}/best_model.pth, then scores the
held-out fold with experiments.infer_tta.predict(method='soft', tta='scale') -> the adopted
soft+scale-TTA decode (apples-to-apples vs the 33.73 baseline).

Run with the BASELINE venv:
  baseline/.venv-baseline/bin/python experiments/ssl_train.py --fold 0 --epochs 20 \
      --lambda-max 1.0 --consistency-steps 200
"""
import argparse
import copy
import glob
import json
import os
import random
import sys

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
sys.path.insert(0, PROJ)
from dataset import KeypointDataset, KeypointUniformSampler  # noqa: E402
from model_factory import MultiTaskModelFactory               # noqa: E402
from utils import keypoint_collate_fn, set_seed                # noqa: E402
import cv2                                                     # noqa: E402
from experiments import semisup as S                           # noqa: E402
from experiments import decode as D                            # noqa: E402
from experiments import infer_tta                              # noqa: E402
from scoring import score as scorer                            # noqa: E402

INPUT, HM = 518, (64, 64)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _build_model(train_df, dev):
    """Build the full multi-task model over the tasks present in the training split (matches
    run_config's head construction)."""
    cfgs, seen = [], set()
    for _, r in train_df.iterrows():
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            cfgs.append({"task_id": r["task_id"], "task_name": "Regression",
                         "num_classes": int(r["num_classes"])})
    return MultiTaskModelFactory("vit_small_patch14_dinov2.lvd142m", "pretrained", cfgs, HM).to(dev)


@torch.no_grad()
def _ema_update(teacher, student, alpha):
    """theta_t <- alpha*theta_t + (1-alpha)*theta_s for params AND buffers (BN stats)."""
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(alpha).add_(sp.detach(), alpha=1.0 - alpha)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        if tb.dtype.is_floating_point:
            tb.mul_(alpha).add_(sb.detach(), alpha=1.0 - alpha)
        else:
            tb.copy_(sb)


class UnlabeledPool(torch.utils.data.Dataset):
    """Loads pool images (data/unlabeled/<task>/<file>) as 518x518 uint8 RGB (weak/canonical
    frame). Strong augmentation (scale+photometric) is applied in the training loop so the exact
    geometric inverse (decode.warp_affine) is shared with the heatmap un-warp."""

    def __init__(self, pool_df, data_root):
        self.df = pool_df.reset_index(drop=True)
        self.root = os.path.join(data_root, "unlabeled")
        self.resize = A.Compose([A.Resize(INPUT, INPUT)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.root, r["image_path"])
        img = cv2.imread(p)
        if img is None:
            return self.__getitem__((i + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize(image=img)["image"]            # 518x518x3 uint8
        return {"image": img, "task_id": r["task_id"]}


def _to_tensor(img_uint8):
    return A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])(image=img_uint8)["image"]


def _consistency_step(student, teacher, batch_imgs_uint8, task_id, dev, floors):
    """One consistency step for a task-uniform unlabeled batch (all same task_id).

    teacher: weak (canonical) view -> soft targets + per-landmark prominence gating weights.
    student: strong view (scale s + photometric); the loss is computed in the student frame by
    warping the DETACHED teacher target forward (decode.warp_affine(inverse=False)) so grad flows
    through the student. Returns the gated weighted-MSE consistency loss (a tensor)."""
    n = len(batch_imgs_uint8)
    s = float(np.random.uniform(*S.SCALE_RANGE))

    weak_t, strong_t = [], []
    for img in batch_imgs_uint8:                                 # img: 518x518x3 uint8 (canonical)
        weak_t.append(_to_tensor(img))
        # photometric (numpy, geometry-preserving): brightness/contrast/gamma
        f = img.astype(np.float64) / 255.0
        f = np.clip((f - 0.5) * float(np.random.uniform(0.8, 1.2)) + 0.5
                    + float(np.random.uniform(-0.1, 0.1)), 0, 1)      # contrast + brightness
        f = np.power(f, float(np.random.uniform(0.8, 1.25)))         # gamma
        strong = np.stack([D.warp_affine(f[..., c], s, inverse=False) for c in range(3)], axis=-1)
        strong = np.clip(strong * 255.0, 0, 255).astype(np.uint8)
        strong_t.append(_to_tensor(strong))
    weak = torch.stack(weak_t, 0).to(dev)
    strong = torch.stack(strong_t, 0).to(dev)

    with torch.no_grad():
        t_hm = torch.sigmoid(teacher(weak, task_id=task_id))         # [n,K,64,64] canonical
    s_hm = torch.sigmoid(student(strong, task_id=task_id))           # [n,K,64,64] strong frame
    # Align in the STUDENT (strong) frame so grad flows through s_hm: warp the DETACHED teacher
    # target forward (inverse=False) into the strong frame. (Un-warping the student via numpy would
    # break autograd — the student must stay a torch tensor.) decode.warp_affine is the exact,
    # round-trip-tested operator; the teacher is already no-grad so warping it in numpy is fine.
    t_np = t_hm.cpu().numpy()
    t_in_strong = np.stack([[D.warp_affine(t_np[b, k], s, inverse=False) for k in range(t_np.shape[1])]
                            for b in range(n)], axis=0)
    t_target = torch.from_numpy(t_in_strong).to(dev).float()         # teacher target in student frame
    # per-landmark gating weight from the teacher's canonical map (prominence), per sample
    floor_vec = []
    for b in range(n):
        w = S.prominence_weight(t_np[b], floor=S.task_floor(floors, task_id))   # [K]
        floor_vec.append(w)
    w = torch.from_numpy(np.stack(floor_vec, 0)).to(dev).float()     # [n,K]
    se = (s_hm - t_target) ** 2                                       # [n,K,64,64], grad via s_hm
    per_lm = se.mean(dim=(2, 3))                                      # [n,K]
    denom = w.sum().clamp_min(1e-6)
    return (w * per_lm).sum() / denom


def train_fold(folds_csv, fold, seed=42, mem_frac=0.28):
    """Set up student + EMA teacher, supervised loader, and the leak-filtered unlabeled pool for the
    fold; returns the components. The epoch loop + lambda/EMA ramps live in main() (loop hyperparams
    are not train_fold args)."""
    set_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- supervised data (identical to run_config) ---
    tfm = A.Compose([A.Resize(INPUT, INPUT), A.Normalize(MEAN, STD), ToTensorV2()])
    ds = KeypointDataset(data_root=os.path.join(PROJ, "data"), transforms=tfm, heatmap_size=HM, sigma=1.8)
    folds = pd.read_csv(folds_csv)
    val_paths = set(folds[folds.fold == fold]["image_path"])
    keep = set(folds[folds.fold >= 0]["image_path"])
    paths = ds.dataframe["image_path"]
    is_val = paths.isin(val_paths).to_numpy()
    train_idx = np.where(~is_val & paths.isin(keep).to_numpy())[0]
    train_df = ds.dataframe.iloc[train_idx].reset_index(drop=True)

    student = _build_model(train_df, dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    sub = torch.utils.data.Subset(ds, train_idx.tolist())
    sub.dataframe = train_df
    sup_loader = torch.utils.data.DataLoader(sub, batch_sampler=KeypointUniformSampler(sub, 4),
                                             num_workers=4, collate_fn=keypoint_collate_fn)

    # --- unlabeled pool: per-fold leakage filter (exact phash exclude_near) ---
    manifest = pd.read_csv(os.path.join(PROJ, "data", "unlabeled", "manifest.csv"))
    labeled = pd.read_csv(os.path.join(PROJ, "data", "unlabeled", "_labeled_phash.csv"))
    pool = S.filter_pool_for_fold(manifest, folds, labeled, fold, thresh=S.LEAK_THRESH)
    leak = S.count_near(pool["phash"].tolist(), S.fold_val_phashes(folds, labeled, fold), thresh=S.LEAK_THRESH)
    assert leak == 0, f"fold {fold}: {leak} pool images leak against fold-{fold} val (must be 0)"
    print(f"fold {fold}: pool {len(manifest)} -> {len(pool)} after leak filter (leak={leak})")
    unl_ds = UnlabeledPool(pool, os.path.join(PROJ, "data"))

    groups = [{"params": student.encoder.parameters(), "lr": 2e-5}]
    for h in student.heads.values():
        groups.append({"params": h.parameters(), "lr": 1e-3})
    opt = torch.optim.AdamW(groups)

    return student, teacher, dev, sup_loader, unl_ds, pool, opt, sorted(val_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-csv", default=os.path.join(PROJ, "data", "folds", "folds.csv"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=S.EPOCHS)
    ap.add_argument("--lambda-max", type=float, default=S.LAMBDA_MAX)
    ap.add_argument("--consistency-steps", type=int, default=200)
    ap.add_argument("--ema-steps", type=int, default=200)
    ap.add_argument("--mem-frac", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    (student, teacher, dev, sup_loader, unl_ds, pool, opt, val_paths) = train_fold(
        args.folds_csv, args.fold, seed=args.seed, mem_frac=args.mem_frac)

    floors = S.PROM_FLOOR
    gstep = 0
    loss_log = []                       # (gstep, supervised, consistency, lambda) for the notebook
    for ep in range(args.epochs):
        student.train(); teacher.eval()   # teacher in eval: BN uses EMA running stats, not batch stats
        # one task-uniform unlabeled batch plan per epoch, length = supervised steps this epoch
        n_steps = len(sup_loader)
        unl_batches = S.task_balanced_batches(pool, batch_size=S.UNLABELED_BATCH, steps=n_steps)
        ub_iter = iter(unl_batches)
        for b in sup_loader:
            imgs = b["image"].to(dev)
            sup_loss_val = 0.0
            for tid in sorted(set(b["task_id"])):
                ix = [i for i, t in enumerate(b["task_id"]) if t == tid]
                hm = torch.stack([b["heatmap"][i] for i in ix], 0).to(dev)
                sup = F.mse_loss(torch.sigmoid(student(imgs[ix], task_id=tid)), hm)
                # consistency on one unlabeled batch (its own task)
                ub = next(ub_iter)
                ut = pool.iloc[ub[0]]["task_id"]
                ub_imgs = [unl_ds[j]["image"] for j in ub]
                cons = _consistency_step(student, teacher, ub_imgs, ut, dev, floors)
                lam = S.lambda_ramp(gstep, args.consistency_steps, args.lambda_max)
                total = sup + lam * cons
                opt.zero_grad(); total.backward(); opt.step()
                _ema_update(teacher, student, S.ema_alpha(gstep, args.ema_steps))
                loss_log.append((gstep, float(sup.item()), float(cons.item()), float(lam)))
                sup_loss_val = float(sup.item()); gstep += 1
        print(f"fold {args.fold} epoch {ep + 1}/{args.epochs} done (last sup={sup_loss_val:.4f})")

    ckpt_dir = os.path.join(PROJ, "runs", f"ssl_cvfold{args.fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
    # persist the loss log for the notebook (gitignored under runs/)
    pd.DataFrame(loss_log, columns=["step", "supervised", "consistency", "lambda"]).to_csv(
        os.path.join(ckpt_dir, "loss_log.csv"), index=False)

    # held-out fold split + GT (identical to run_config)
    split = os.path.join(PROJ, "data", f"_ssl_cvfold{args.fold}_val.csv")
    pd.DataFrame({"image_path": val_paths}).to_csv(split, index=False)
    gt = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(PROJ, "data/csv/*.csv"))],
                   ignore_index=True)
    gt = gt[gt.image_path.isin(set(val_paths))]
    gtp = os.path.join(PROJ, "data", f"_ssl_cvfold{args.fold}_gt.csv")
    gt.to_csv(gtp, index=False)

    # score with the ADOPTED soft + scale-TTA decode (apples-to-apples vs the 33.73 baseline)
    out = os.path.join(PROJ, "submission", f"ssl_cvfold{args.fold}")
    pred = infer_tta.predict(os.path.join(ckpt_dir, "best_model.pth"),
                             data_root=os.path.join(PROJ, "data"), split_csv=split, out_dir=out,
                             method="soft", tta="scale", scales=(0.92, 1.08), window=7,
                             mem_frac=args.mem_frac)
    res = scorer.score_submission(pred, gtp)
    rdir = os.path.join(PROJ, "experiments", "results", "ssl")
    os.makedirs(rdir, exist_ok=True)
    json.dump(res, open(os.path.join(rdir, f"cvfold{args.fold}.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("avg_mre", "avg_param_mae", "total_missing")}, indent=2))


if __name__ == "__main__":
    main()
