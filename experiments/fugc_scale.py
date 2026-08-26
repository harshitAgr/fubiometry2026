"""Pure geometry for gated FUGC field-of-view scale normalization.

The current raw FUGC validation images are substantially larger than the cropped training
images.  The inference driver can restore approximately the training field of view with a
center crop, but only when both image dimensions prove that the input is genuinely wider.
This module contains no model dependencies so the gate and coordinate mapping are easy to
regression-test.
"""
from __future__ import annotations


SCALE_RATIO = 1.25
CROP_MARGIN = 1.05
TRAIN_DIMS = {"FUGC": (544, 336)}  # median training (width, height)


def needs_scale_norm(img_wh, train_wh, ratio: float = SCALE_RATIO) -> bool:
    """Return whether an image is wider-FOV than training in both dimensions."""
    (width, height), (train_width, train_height) = img_wh, train_wh
    return width / train_width > ratio and height / train_height > ratio


def scale_norm_crop_box(height: int, width: int, train_wh, margin: float = CROP_MARGIN):
    """Return a centered, in-bounds crop approximately matching the training field of view."""
    train_width, train_height = train_wh
    crop_width = min(width, round(train_width * margin))
    crop_height = min(height, round(train_height * margin))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return x0, y0, x0 + crop_width, y0 + crop_height


def crop_pred_to_orig_norm(pred_px_crop, box, width: int, height: int):
    """Map crop-frame pixel coordinates to normalized original-image coordinates."""
    x0, y0 = box[0], box[1]
    out = []
    for x, y in pred_px_crop:
        out.extend(((x0 + x) / width, (y0 + y) / height))
    return out
