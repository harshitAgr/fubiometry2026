"""DINO multi-crop augmentation (2 global + N local crops per image), unsupervised — no keypoint
co-transform needed (unlike experiments/augment.py's KeypointAugDataset packs, which exist because
supervised fine-tuning needs landmark-consistent geometry). Reuses the same photometric building
blocks (experiments.augment._photo_v1) for domain-appropriate speckle/gamma/blur, composed here
with RandomResizedCrop instead of the fine-tune's fixed-Resize + conformal-Affine.

Crop sizes are multiples of 14 (patch-14 backbone) so every crop yields a clean square patch grid;
global crops see most of the frame (scale close to 1) so the teacher gets whole-anatomy context,
local crops are small high-magnification views (DINO's local-to-global consistency: the student's
local view must predict the teacher's global view of the SAME underlying image).
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from experiments.augment import MEAN, STD, _photo_v1

GLOBAL_SIZE = 518   # 37x37 grid @ patch 14 — matches the fine-tune input, so continued-pretrain
LOCAL_SIZE = 168    # sees the same scale regime the encoder will be fine-tuned/scored at; 168/14=12
GLOBAL_SCALE = (0.4, 1.0)
LOCAL_SCALE = (0.05, 0.4)


def _base_compose(size: int, scale: tuple) -> A.Compose:
    assert size % 14 == 0, f"crop size {size} must be a multiple of 14 (patch-14 backbone)"
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=scale, ratio=(0.9, 1.1)),
        *_photo_v1(),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def build_multicrop_transform(n_global: int = 2, n_local: int = 4,
                              global_size: int = GLOBAL_SIZE, local_size: int = LOCAL_SIZE,
                              global_scale: tuple = GLOBAL_SCALE, local_scale: tuple = LOCAL_SCALE):
    """Returns a callable image[H,W,3] uint8-or-float -> list[Tensor] of length n_global+n_local,
    global crops first (order matters: dino_loss.dino_loss's n_global slicing assumes this)."""
    global_tfm = _base_compose(global_size, global_scale)
    local_tfm = _base_compose(local_size, local_scale) if n_local > 0 else None

    def apply(image):
        crops = [global_tfm(image=image)["image"] for _ in range(n_global)]
        crops += [local_tfm(image=image)["image"] for _ in range(n_local)]
        return crops

    return apply
