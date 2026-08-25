"""Self-contained model architecture for the FU_Biometry test-phase container.

Vendored from `baseline/baseline/model_factory.py` (HeatmapHead, DINOv2Backbone) plus
`experiments/per_task_model.py` (per-task heatmap resolution). Vendored rather than imported
because the container has no `baseline/` tree, and because the module MUST NOT be named
`model.py` -- the organizer entry script does `from model import Model`, so that name is
reserved for our inference wrapper.

Two deliberate deviations from the training-time code, both required by the offline container:

1. `pretrained=False`. Training used timm `pretrained=True`, which downloads DINOv2 weights from
   the HF hub -- a hard failure under `--network none`. It is safe to drop: the fine-tuned
   checkpoint carries ALL 174 encoder keys (verified against runs/vitb_full/best_model.pth), so
   `load_state_dict(..., strict=True)` fully determines the encoder. Nothing is left at random
   init, and no weights need to be COPY'd into the image.

2. `TASK_SPEC` is a fixed 9-task literal instead of globbing `{data_root}/csv/*.csv`. The training
   code derived heads from whatever CSVs were present; in the container that would make the head
   set (and therefore state_dict key match) depend on the organizers' test metadata. A test set
   that omitted a task, or reported a different num_classes, would produce a shape/key mismatch on
   load. The spec below is pinned to the checkpoint itself.

`num_classes` == number of landmarks == head output channels; the emitted coordinate list is
`num_classes * 2` long (x,y interleaved).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pinned to the checkpoint. Verified against runs/vitb_full/best_model.pth head shapes AND
# against data/csv/*_train.csv num_classes -- the two agree exactly.
TASK_SPEC = {
    "A4C": 16,
    "AOP": 4,
    "FA": 4,
    "FUGC": 2,
    "HC": 4,
    "IVC": 2,
    "PLAX": 22,
    "PSAX": 4,
    "fetal_femur": 2,
}

# UNIFORM 64 for all 9 tasks.
#
# ⚠️ Do NOT "restore" FUGC@128 here. Some of our older notes describe the deliverable as
# "geo_cosine40 + FUGC@128 + 3-seed ensemble", but that is wrong for the ViT-B spine: FUGC@128
# belongs to the ViT-S v8/v9 lineage and was dropped when ViT-B was built. Two independent proofs (2026-08-01):
#   1. This container path reproduces v12's official-val predictions to ~3e-5 px on FUGC when
#      decoding at 64, but is 2.77 px off at 128 (8/9 other tasks match at ~1e-4 either way).
#   2. Epoch-1 per-task train loss of runs/vitb_full (deployed v12 member) matches a known
#      uniform-64 run on all 9 tasks to ~2e-5; a FUGC@128 run differs on FUGC alone by ~1.9e-3.
# Decoding FUGC at 128 against these checkpoints is a train/inference mismatch AND breaks
# equivalence with the 23.69/29.56 result. The per-task-res 5-fold (2026-07-08) also found FUGC
# "flat (+0.03, no headroom)" at ViT-B, so there is nothing to gain.
HEATMAP_SIZES = {}
DEFAULT_HM = (64, 64)

ENCODER_NAME = "vit_base_patch14_dinov2.lvd142m"   # DINOv2 ViT-B/14, 768-ch
INPUT_SIZE = 518                                    # 518/14 -> 37x37 token grid

# ImageNet normalization (the DINOv2 backbone exposes no norm_mean/norm_std, so infer_ensemble
# falls back to these -- replicated here so the container matches the validated path).
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def hm_for(task_id):
    """Per-task output heatmap size."""
    return tuple(HEATMAP_SIZES.get(task_id, DEFAULT_HM))


class HeatmapHead(nn.Module):
    """Light decoder mapping DINOv2 feature maps to keypoint heatmaps. Verbatim from the baseline
    (module layout fixes the state_dict keys -- do not restructure)."""

    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden = max(in_channels // 2, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, num_points, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        x = self.decoder(x)
        if x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DINOv2Backbone(nn.Module):
    """Returns the last patch feature map as [B, C, H, W]. Verbatim from the baseline except the
    default `pretrained=False` (see module docstring)."""

    def __init__(self, model_name: str = ENCODER_NAME, pretrained: bool = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
        self.out_channels = int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                patch_tokens = feats["x_prenorm"][:, 1:, :]
            else:
                raise RuntimeError("Unsupported forward_features output from DINOv2 backbone.")
        elif isinstance(feats, torch.Tensor) and feats.dim() == 3:
            patch_tokens = feats[:, 1:, :]
        else:
            raise RuntimeError("Unexpected feature type from DINOv2 backbone.")

        bsz, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square; input size may be incompatible.")
        return patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)


class PerTaskHeatmapModel(nn.Module):
    """Shared DINOv2 encoder + per-task heatmap heads, with per-task output resolution.

    State_dict layout is `encoder.backbone.*` and `heads.<task_id>.decoder.*`, matching the
    training-time PerTaskHeatmapModel(MultiTaskModelFactory) exactly.
    """

    def __init__(self, task_spec=None):
        super().__init__()
        task_spec = TASK_SPEC if task_spec is None else task_spec
        self.encoder = DINOv2Backbone(ENCODER_NAME, pretrained=False)
        self.heads = nn.ModuleDict()
        for task_id, num_points in task_spec.items():
            self.heads[task_id] = HeatmapHead(self.encoder.out_channels, int(num_points))

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")
        return self.heads[task_id](self.encoder(x), out_size=hm_for(task_id))


def load_member(ckpt_path, device):
    """Build the architecture and load one ensemble member, strictly."""
    model = PerTaskHeatmapModel()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)   # strict: a silent key drift must fail loudly
    model.to(device).eval()
    return model
