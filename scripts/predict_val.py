"""Run the adopted Lever-1 inference (sub-pixel soft decode + scale-TTA) on the prepared
validation set -> submission JSON.

Uses experiments.infer_tta.predict (the same driver validated on the CV folds) with the adopted
config: method="soft" (7x7 windowed centroid) + heatmap-space scale-TTA {0.92,1.0,1.08}, no
flip/photometric. Writes regression_predictions.json (the verified flat-array submission schema).
GPU memory is capped inside predict() for shared-GPU use.

Run from the project root with the BASELINE venv:
  baseline/.venv-baseline/bin/python scripts/predict_val.py [run_name]
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

ap = argparse.ArgumentParser(description="Predict the official val set -> submission JSON.")
ap.add_argument("run_name", nargs="?", default="baseline_v1",
                help="runs/<run_name>/best_model.pth -> submission/<run_name>/")
ap.add_argument("--encoder", default="dinov2_vits",
                choices=["dinov2_vits", "dinov2_vitb", "dinov2_vitl", "beit_imagenet", "usfm_beit", "dinov3_vits"],
                help="Encoder backbone the checkpoint was trained with (default: dinov2_vits).")
ap.add_argument("--input-size", type=int, default=None,
                help="Square input size; default None -> 518 (DINOv2). Set 224 for USFM@224.")
args = ap.parse_args()
run_name = args.run_name
RUN = os.path.join(PROJ, "runs", run_name)
OUT = os.path.join(PROJ, "submission", run_name)

ckpt = os.path.join(RUN, "best_model.pth")
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"No best_model.pth in {RUN} (train first).")

from experiments.infer_tta import predict  # noqa: E402

predict(checkpoint=ckpt, data_root=os.path.join(PROJ, "data", "val"), split_csv=None,
        out_dir=OUT, method="soft", tta="scale", scales=(0.92, 1.08), window=7,
        encoder_name=args.encoder, input_size=args.input_size)
print(f"Wrote predictions to {OUT}/regression_predictions.json")
