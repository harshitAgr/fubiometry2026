# experiments/infer_ensemble.py
"""N-seed ENSEMBLE inference: heatmap-space averaging across N checkpoints (× scale-TTA views),
then a single sub-pixel decode. The test-phase deliverable's inference path.

Why heatmap-space (not coord-space) averaging: the decode is non-linear (soft-argmax over a 7x7
window), and averaging confident peaks BEFORE decode is the standard, variance-reducing heatmap
ensemble — it denoises the noisy tiny tasks (IVC/PSAX/HC) and is robust to a single member's
single-landmark blowup. Each member is built with the SAME per-task heatmap size it was trained
with (e.g. {"FUGC": (128,128)}), so the forward emits the matching grid; decode is grid-agnostic.

Reuses infer_tta's dataset/decode/scoring helpers verbatim; only the model loop differs (a LIST of
checkpoints averaged per image). Writes the standard regression_predictions.json and (with --gt)
the score JSON used by the CV aggregator.

Run with the BASELINE venv:
  baseline/.venv-baseline/bin/python experiments/infer_ensemble.py \
      --checkpoints "runs/ens_s42/best_model.pth,runs/ens_s43/best_model.pth,..." \
      --split-csv data/_cvfold0_val.csv --out submission/ens/fold0 \
      --tta scale --fugc-heatmap-size 128 \
      --gt data/_cvfold0_gt.csv --results-json experiments/results/ens/fold0.json
"""
import argparse
import glob
import json
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
sys.path.insert(0, PROJ)
from model import InferenceDataset                 # noqa: E402
from experiments import decode as D                # noqa: E402
from experiments import fugc_scale as FSN           # noqa: E402
from experiments import hc_scale_norm as HCN        # noqa: E402
from experiments.encoders import build_encoder     # noqa: E402
from experiments.per_task_model import build_model  # noqa: E402
from experiments.infer_tta import _full_task_configs, _to_tensor, INPUT, HM  # noqa: E402
from scoring import score as scorer                # noqa: E402

HC_SCALE_NORM_TASK = "HC"
FUGC_SCALE_NORM_TASK = "FUGC"


def _load_models(checkpoints, cfgs, heatmap_size, encoder_name, input_size, encoder_init, dev):
    """Build + load each checkpoint into its own model (own encoder instance). Returns
    (models, norm_mean, norm_std). All members share architecture/normalization."""
    models = []
    norm_mean, norm_std = None, None
    for ckpt in checkpoints:
        enc = build_encoder(encoder_name, input_size, encoder_init=encoder_init)
        norm_mean = getattr(enc, "norm_mean", (0.485, 0.456, 0.406))
        norm_std = getattr(enc, "norm_std", (0.229, 0.224, 0.225))
        m = build_model(cfgs, heatmap_size, enc).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        m.eval()
        models.append(m)
    return models, norm_mean, norm_std


def _ensemble_decode(img_uint8, task_id, models, view_scales, norm_mean, norm_std, dev,
                      method, window):
    """One ensemble forward pass (all members x all TTA scale-views, heatmap-space average,
    single sub-pixel decode) on an already-square-resized uint8 RGB image. Returns normalized
    (x,y interleaved) coords. Factored out so pass-1 (full image) and pass-2 (HC scale-norm crop)
    of the gated HC lever run through byte-identical code."""
    tens = []
    for s in view_scales:
        if s == 1.0:
            img = img_uint8
        else:
            img = np.stack([D.warp_affine(img_uint8[..., c].astype(np.float64), s, inverse=False)
                            for c in range(3)], axis=-1)
            img = np.clip(img, 0, 255).astype(np.uint8)
        tens.append(_to_tensor(img, mean=norm_mean, std=norm_std))
    batch = torch.stack(tens, 0).to(dev)                       # [V,3,H,H]
    canon = []                                                 # N*V canonicalized heatmaps
    for m in models:
        hm = torch.sigmoid(m(batch, task_id=task_id)).cpu().numpy()  # [V,K,h,w]
        for vi, s in enumerate(view_scales):
            h = hm[vi]
            if s != 1.0:
                h = np.stack([D.warp_affine(h[c], s, inverse=True) for c in range(h.shape[0])], 0)
            canon.append(h)
    avg = D.average_heatmaps(canon)                            # [K,h,w] mean over N*V
    return D.decode_subpixel(avg[None], method=method, window=window)[0].tolist()


def _hc_scale_norm_pass(ds, item, px, input_size, models, view_scales, norm_mean, norm_std,
                         dev, method, window, margin=HCN.DEFAULT_MARGIN):
    """Gated two-pass HC scale-norm. `px` is the pass-1 predicted pixel coords (x,y interleaved,
    original-image space). Returns (px, triggered, cropped) where `px` is either the pass-1
    coords unchanged (no-op / fallback) or the pass-2 remapped coords; `triggered` is True iff
    crop_frac < 1.0 (the image is larger than train scale); `cropped` is True iff pass 2 actually
    ran (triggered AND a valid crop box was found)."""
    H0, W0 = item["original_size"]
    frac = HCN.crop_frac(W0, H0)
    if frac >= 1.0:
        return px, False, False                                # do-no-harm no-op (train-sized+)
    pts = list(zip(px[0::2], px[1::2]))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    box = HCN.compute_crop_box(W0, H0, cx, cy, frac, points=pts, margin=margin)
    if box is None:
        return px, True, False                                  # can't contain pass-1 pts -> fallback
    x0i, y0i = int(round(box[0])), int(round(box[1]))
    x1i, y1i = int(round(box[2])), int(round(box[3]))
    x1i = min(max(x1i, x0i + 1), W0)
    y1i = min(max(y1i, y0i + 1), H0)
    x0i, y0i = max(0, x0i), max(0, y0i)
    abs_path = ds._resolve_image_path(item["image_path"])
    orig = cv2.imread(abs_path)
    if orig is None:
        return px, True, False                                  # can't re-read -> fallback
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    crop = orig[y0i:y1i, x0i:x1i]
    crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    coords2 = _ensemble_decode(crop, item["task_id"], models, view_scales, norm_mean, norm_std,
                               dev, method, window)
    px2 = HCN.map_crop_to_original_px(coords2, (x0i, y0i, x1i, y1i))
    return px2, True, True


def _fugc_scale_norm_pass(ds, item, input_size, models, view_scales, norm_mean, norm_std,
                          dev, method, window):
    """Geometry-gated FUGC crop pass.

    Returns ``(coords, px, triggered, cropped)``. The first two values are ``None`` unless a
    crop prediction completed successfully, allowing the caller to retain the byte-identical
    full-frame prediction on non-triggered inputs or image-read failures.
    """
    H0, W0 = item["original_size"]
    train_wh = FSN.TRAIN_DIMS[FUGC_SCALE_NORM_TASK]
    if not FSN.needs_scale_norm((W0, H0), train_wh):
        return None, None, False, False

    abs_path = ds._resolve_image_path(item["image_path"])
    orig = cv2.imread(abs_path) if abs_path else None
    if orig is None:
        return None, None, True, False
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    box = FSN.scale_norm_crop_box(H0, W0, train_wh)
    crop = orig[box[1]:box[3], box[0]:box[2]]
    if crop.size == 0:
        return None, None, True, False
    crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    crop_coords = _ensemble_decode(crop, item["task_id"], models, view_scales, norm_mean,
                                   norm_std, dev, method, window)
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    crop_px = [(crop_coords[j] * crop_w, crop_coords[j + 1] * crop_h)
               for j in range(0, len(crop_coords), 2)]
    coords = FSN.crop_pred_to_orig_norm(crop_px, box, W0, H0)
    px = []
    for j in range(0, len(coords), 2):
        px.extend((coords[j] * W0, coords[j + 1] * H0))
    return coords, px, True, True


def predict(checkpoints, data_root, split_csv, out_dir, method, tta, scales, window,
            mem_frac=0.28, encoder_name="dinov2_vits", input_size=None, encoder_init=None,
            heatmap_size=HM, hc_scale_norm=False, fugc_scale_norm=False):
    if input_size is None:
        input_size = INPUT
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = _full_task_configs(data_root)
    models, norm_mean, norm_std = _load_models(
        checkpoints, cfgs, heatmap_size, encoder_name, input_size, encoder_init, dev)
    print(f"[ensemble] loaded {len(models)} members")

    ds = InferenceDataset(data_root=data_root,
                          transforms=A.Compose([A.Resize(input_size, input_size)]),
                          split_csv=split_csv)
    view_scales = [1.0] if tta == "none" else [1.0] + list(scales)
    results = []
    n_hc_triggered = n_hc_cropped = 0
    n_fugc_triggered = n_fugc_cropped = 0
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            base = item["image"]                      # HxWx3 uint8 RGB (A.Resize only)
            coords = _ensemble_decode(base, item["task_id"], models, view_scales, norm_mean,
                                      norm_std, dev, method, window)
            H0, W0 = item["original_size"]
            px = []
            for j in range(0, len(coords), 2):
                px += [coords[j] * W0, coords[j + 1] * H0]
            if hc_scale_norm and item["task_id"] == HC_SCALE_NORM_TASK:
                new_px, triggered, cropped = _hc_scale_norm_pass(
                    ds, item, px, input_size, models, view_scales, norm_mean, norm_std, dev,
                    method, window)
                n_hc_triggered += int(triggered)
                n_hc_cropped += int(cropped)
                if cropped:
                    # only rebuild px/coords when pass 2 actually ran -- keeps the no-op path
                    # (frac>=1.0, or a fallback where new_px IS px unchanged) byte-identical to
                    # pass-1 (no coords->px->coords round-trip float noise).
                    px = new_px
                    coords = []
                    for j in range(0, len(px), 2):
                        coords += [px[j] / W0, px[j + 1] / H0]
            if fugc_scale_norm and item["task_id"] == FUGC_SCALE_NORM_TASK:
                new_coords, new_px, triggered, cropped = _fugc_scale_norm_pass(
                    ds, item, input_size, models, view_scales, norm_mean, norm_std, dev,
                    method, window)
                n_fugc_triggered += int(triggered)
                n_fugc_cropped += int(cropped)
                if cropped:
                    coords, px = new_coords, new_px
            results.append({"image_path": item["image_path"], "task_id": item["task_id"],
                            "predicted_points_normalized": coords, "predicted_points_pixels": px})

    if hc_scale_norm:
        print(f"[hc_scale_norm] HC images: gate triggered {n_hc_triggered}, "
              f"pass-2 crop applied {n_hc_cropped}")
    if fugc_scale_norm:
        print(f"[fugc_scale_norm] FUGC images: gate triggered {n_fugc_triggered}, "
              f"crop applied {n_fugc_cropped}")

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "regression_predictions.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {json_path} ({len(results)} samples)")
    return json_path


def _resolve_checkpoints(spec):
    """spec is a comma-sep list of paths and/or globs -> sorted, de-duplicated existing files."""
    paths = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        hits = sorted(glob.glob(tok))
        paths.extend(hits if hits else [tok])
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True,
                    help="Comma-separated checkpoint paths and/or globs "
                         "(e.g. 'runs/ens_s*/best_model.pth').")
    ap.add_argument("--data-root", default=os.path.join(PROJ, "data"))
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="soft",
                    choices=["argmax", "soft", "parabolic", "log_parabolic"])
    ap.add_argument("--tta", default="scale", choices=["none", "scale"])
    ap.add_argument("--scales", default="0.92,1.08")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--gt", default=None, help="optional GT csv -> also score")
    ap.add_argument("--results-json", default=None, help="where to write the score dict")
    ap.add_argument("--encoder", default="dinov2_vits",
                    choices=["dinov2_vits", "dinov2_vitb", "dinov2_vitl", "beit_imagenet", "usfm_beit", "dinov3_vits"])
    ap.add_argument("--mem-frac", type=float, default=0.28,
                    help="Per-process GPU memory cap (shared GPU). Inference is light even for N "
                         "models; raise if loading many large members.")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--encoder-init", default=None)
    ap.add_argument("--heatmap-size", type=int, default=64,
                    help="Base square heatmap resolution the members were trained with (default 64).")
    ap.add_argument("--fugc-heatmap-size", type=int, default=None,
                    help="If set (and != --heatmap-size), build a per-task model with FUGC at this "
                         "resolution (others at --heatmap-size) — must match how the members were trained.")
    ap.add_argument("--femur-heatmap-size", type=int, default=None,
                    help="If set (and != --heatmap-size), put fetal_femur at this resolution too "
                         "(combinable with --fugc-heatmap-size) — must match how the members were trained.")
    ap.add_argument("--hc-scale-norm", action="store_true",
                    help="Gated two-pass HC scale-norm (default OFF -> byte-identical to no flag). "
                         "For HC images larger than train scale (crop_frac()<1.0), re-crop around "
                         "the pass-1 head centroid to train-equivalent FOV and re-predict; falls "
                         "back to the pass-1 prediction unchanged if the crop can't contain the "
                         "pass-1 points or the image is already train-sized-or-smaller "
                         "(see experiments/hc_scale_norm.py).")
    ap.add_argument("--fugc-scale-norm", action="store_true",
                    help="Geometry-gated FUGC scale normalization. Wide-FOV FUGC images are "
                         "center-cropped toward the 544x336 training field of view, predicted "
                         "again, and mapped back; normal-sized images are exact no-ops.")
    args = ap.parse_args()

    checkpoints = _resolve_checkpoints(args.checkpoints)
    if not checkpoints:
        ap.error(f"no checkpoints matched: {args.checkpoints!r}")
    missing = [c for c in checkpoints if not os.path.exists(c)]
    if missing:
        ap.error(f"checkpoint(s) not found: {missing}")
    print(f"[ensemble] members: {checkpoints}")

    hm = (args.heatmap_size, args.heatmap_size)
    per_task_hm = {}
    if args.fugc_heatmap_size and args.fugc_heatmap_size != args.heatmap_size:
        per_task_hm["FUGC"] = (args.fugc_heatmap_size, args.fugc_heatmap_size)
    if args.femur_heatmap_size and args.femur_heatmap_size != args.heatmap_size:
        per_task_hm["fetal_femur"] = (args.femur_heatmap_size, args.femur_heatmap_size)
    if per_task_hm:
        hm = per_task_hm

    scales = tuple(float(x) for x in args.scales.split(",") if x)
    pred = predict(checkpoints, args.data_root, args.split_csv, args.out,
                   args.method, args.tta, scales, args.window,
                   mem_frac=args.mem_frac,
                   encoder_name=args.encoder, input_size=args.input_size,
                   encoder_init=args.encoder_init, heatmap_size=hm,
                   hc_scale_norm=args.hc_scale_norm,
                   fugc_scale_norm=args.fugc_scale_norm)
    if args.gt:
        res = scorer.score_submission(pred, args.gt)
        if args.results_json:
            os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump(res, f, indent=2)
        print(json.dumps({k: res[k] for k in ("avg_mre", "avg_param_mae", "total_missing")}, indent=2))


if __name__ == "__main__":
    main()
