# experiments/infer_tta.py
"""Re-decode existing checkpoints with sub-pixel decode + optional light geometric TTA.

Inference-only. Reuses the baseline InferenceDataset (path resolution, original_size, the
Regression-task filter) and MultiTaskModelFactory; swaps the decode and adds heatmap-space TTA.
Writes the standard regression_predictions.json and (with --gt) the score JSON used by the CV
aggregator. Shared by the CV path and the official-val path.

Run with the BASELINE venv:
  baseline/.venv-baseline/bin/python experiments/infer_tta.py \
      --checkpoint runs/cvfold0/best_model.pth --split-csv data/_cvfold0_val.csv \
      --out submission/screen_fold0/soft --method soft --tta none \
      --gt data/_cvfold0_gt.csv --results-json experiments/results/screen_fold0/soft.json
"""
import argparse
import glob
import json
import os
import sys

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
sys.path.insert(0, PROJ)
from model import InferenceDataset                 # noqa: E402
from model_factory import MultiTaskModelFactory    # noqa: E402
from experiments import decode as D                # noqa: E402
from experiments.encoders import build_encoder     # noqa: E402
from experiments.per_task_model import build_model  # noqa: E402
from scoring import score as scorer                # noqa: E402
from experiments.geometry_project import project as geom_project  # noqa: E402

INPUT, HM = 518, (64, 64)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _full_task_configs(data_root):
    """Build configs for ALL tasks in data/csv so heads match the checkpoint exactly."""
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(data_root, "csv", "*.csv"))],
                   ignore_index=True)
    cfgs, seen = [], set()
    for _, r in df.sort_values("task_id").iterrows():
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            cfgs.append({"task_id": r["task_id"], "task_name": "Regression",
                         "num_classes": int(r["num_classes"])})
    return cfgs


def _to_tensor(img_uint8, mean=MEAN, std=STD):
    """HWC uint8 RGB -> CHW float tensor with the given normalization."""
    return A.Compose([A.Normalize(mean, std), ToTensorV2()])(image=img_uint8)["image"]


def build_view_batch(base_uint8, view_scales, mean=MEAN, std=STD):
    """Build the exact deployed normalized view stack from one resized RGB image."""
    tensors = []
    for scale in view_scales:
        if scale == 1.0:
            image = base_uint8
        else:
            image = np.stack([
                D.warp_affine(base_uint8[..., channel].astype(np.float64), scale, inverse=False)
                for channel in range(3)
            ], axis=-1)
            image = np.clip(image, 0, 255).astype(np.uint8)
        tensors.append(_to_tensor(image, mean=mean, std=std))
    return torch.stack(tensors, dim=0)


def predict(checkpoint, data_root, split_csv, out_dir, method, tta, scales, window,
            mem_frac=0.28, encoder_name="dinov2_vits", input_size=None, encoder_init=None,
            heatmap_size=HM, temperature=1.0, model_variant="base", geometry_project=False):
    # Resolve input size: use provided value, else fall back to module default (518)
    if input_size is None:
        input_size = INPUT
    if model_variant == "coordse" and tta != "none":
        raise ValueError("coordse variant currently requires --tta none; coordinate TTA needs explicit inverse mapping")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = _full_task_configs(data_root)
    # encoder_init is a no-op at score time (the fine-tuned checkpoint already carries the adapted
    # encoder weights via load_state_dict below); threaded for symmetry/safety so the architecture
    # is rebuilt identically and a reviewer cannot mis-wire it.
    enc = build_encoder(encoder_name, input_size, encoder_init=encoder_init)
    # Determine normalization: use encoder's norm_mean/norm_std if present, else ImageNet default
    norm_mean = getattr(enc, "norm_mean", MEAN)
    norm_std = getattr(enc, "norm_std", STD)
    model = build_model(cfgs, heatmap_size, enc, variant=model_variant).to(dev)
    model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    model.eval()

    ds = InferenceDataset(data_root=data_root, transforms=A.Compose([A.Resize(input_size, input_size)]),
                          split_csv=split_csv)
    view_scales = [1.0] if tta == "none" else [1.0] + list(scales)
    results = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            base = item["image"]                      # 518x518x3 uint8 RGB (A.Resize only)
            batch = build_view_batch(base, view_scales, mean=norm_mean, std=norm_std).to(dev)
            raw_tensor = model(batch, task_id=item["task_id"])  # [V,K,64,64] logits
            if model_variant == "coordse":
                # The pulled method has a task-specific learned temperature. Decode each
                # view in the model's coordinate space, then average coordinates after
                # undoing geometric TTA; no sigmoid/local-peak decoder is involved.
                coords = model.dsnt_modules[item["task_id"]](raw_tensor)[0]
                coords = coords.detach().cpu().numpy()
                coords = coords[0]
                coords = coords.reshape(-1).tolist()
                H0, W0 = item["original_size"]
                px = []
                for j in range(0, len(coords), 2):
                    px += [(coords[j] + 1.0) * 0.5 * W0,
                           (coords[j + 1] + 1.0) * 0.5 * H0]
                if geometry_project:
                    px = geom_project(px, item["task_id"])
                    coords_n = [px[j] / (W0 if j % 2 == 0 else H0) for j in range(len(px))]
                else:
                    coords_n = [(v + 1.0) * 0.5 for v in coords]
                results.append({"image_path": item["image_path"], "task_id": item["task_id"],
                                "predicted_points_normalized": coords_n,
                                "predicted_points_pixels": px})
                continue
            raw = raw_tensor.cpu().numpy()
            hm = 1.0 / (1.0 + np.exp(-raw))
            canon = []
            for vi, s in enumerate(view_scales):
                h = raw[vi] if method == "dsnt" else hm[vi]
                if s != 1.0:
                    h = np.stack([D.warp_affine(h[c], s, inverse=True) for c in range(h.shape[0])], 0)
                canon.append(h)
            avg = D.average_heatmaps(canon)                            # [K,64,64]
            if method == "dsnt":
                coords = D.decode_dsnt(avg[None], temperature=temperature)[0].tolist()
            else:
                coords = D.decode_subpixel(avg[None], method=method, window=window)[0].tolist()
            H0, W0 = item["original_size"]
            px = []
            for j in range(0, len(coords), 2):
                px += [coords[j] * W0, coords[j + 1] * H0]
            if geometry_project:
                px = geom_project(px, item["task_id"])
                coords = [px[j] / (W0 if j % 2 == 0 else H0) for j in range(len(px))]
            results.append({"image_path": item["image_path"], "task_id": item["task_id"],
                            "predicted_points_normalized": coords, "predicted_points_pixels": px})

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "regression_predictions.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {json_path} ({len(results)} samples)")
    return json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=os.path.join(PROJ, "data"))
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="soft",
                    choices=["argmax", "soft", "parabolic", "log_parabolic", "dsnt"])
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="DSNT logit temperature divisor (only used with --method dsnt).")
    ap.add_argument("--model-variant", default="base", choices=["base", "coordse"],
                    help="Model head/decoder variant; coordse requires a matching checkpoint.")
    ap.add_argument("--tta", default="none", choices=["none", "scale"])
    ap.add_argument("--scales", default="0.92,1.08")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--geometry-project", action="store_true",
                    help="snap FA/HC landmarks onto the exact label geometry, "
                         "length-preserving (see experiments/geometry_project.py). "
                         "CV: task-mean MRE -0.1455, parameter +0.0000.")
    ap.add_argument("--gt", default=None, help="optional GT csv -> also score")
    ap.add_argument("--results-json", default=None, help="where to write the score dict")
    ap.add_argument("--encoder", default="dinov2_vits",
                    choices=["dinov2_vits", "dinov2_vitb", "dinov2_vitb_fuse4", "dinov2_vitl", "beit_imagenet", "usfm_beit", "dinov3_vits", "dinov3_vitb"],
                    help="Encoder backbone (must match the trained checkpoint; default: dinov2_vits)")
    ap.add_argument("--input-size", type=int, default=None,
                    help="Square input size in pixels (default: 518 for dinov2_vits, 224 for BEiT)")
    ap.add_argument("--encoder-init", default=None,
                    help="Path to a matching custom DINOv2 backbone state_dict (Axis A); no-op at score time "
                         "since the FT checkpoint carries the weights — accepted for symmetry")
    ap.add_argument("--heatmap-size", type=int, default=64,
                    help="Square heatmap resolution the checkpoint was trained with (default 64).")
    ap.add_argument("--fugc-heatmap-size", type=int, default=None,
                    help="If set (and != --heatmap-size), build a per-task model with FUGC at this "
                         "resolution (others at --heatmap-size) — must match how it was trained.")
    ap.add_argument("--femur-heatmap-size", type=int, default=None,
                    help="If set (and != --heatmap-size), put fetal_femur at this resolution too "
                         "(combinable with --fugc-heatmap-size) — must match how it was trained.")
    args = ap.parse_args()

    hm = (args.heatmap_size, args.heatmap_size)
    per_task_hm = {}
    if args.fugc_heatmap_size and args.fugc_heatmap_size != args.heatmap_size:
        per_task_hm["FUGC"] = (args.fugc_heatmap_size, args.fugc_heatmap_size)
    if args.femur_heatmap_size and args.femur_heatmap_size != args.heatmap_size:
        per_task_hm["fetal_femur"] = (args.femur_heatmap_size, args.femur_heatmap_size)
    if per_task_hm:
        hm = per_task_hm
    scales = tuple(float(x) for x in args.scales.split(",") if x)
    pred = predict(args.checkpoint, args.data_root, args.split_csv, args.out,
                   args.method, args.tta, scales, args.window,
                   encoder_name=args.encoder, input_size=args.input_size,
                   encoder_init=args.encoder_init,
                   heatmap_size=hm, temperature=args.temperature,
                   model_variant=args.model_variant,
                   geometry_project=args.geometry_project)
    if args.gt:
        res = scorer.score_submission(pred, args.gt)
        if args.results_json:
            os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump(res, f, indent=2)
        print(json.dumps({k: res[k] for k in ("avg_mre", "avg_param_mae", "total_missing")}, indent=2))


if __name__ == "__main__":
    main()
