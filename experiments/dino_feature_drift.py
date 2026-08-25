"""Frozen-feature drift gate for continued-pretrain sanity runs (the Axis A lesson:
MAE's fixed 3-epoch adaptation shifted CLS features 1.3x on a fixed batch vs only 0.18
for a matched-magnitude random weight perturbation control -- i.e. the drift was COORDINATED
(a real, if premature, directional shift) not noise, which is exactly what SSL pretraining SHOULD
look like; the gate is to make sure a training run isn't WORSE than a same-magnitude random
perturbation would be (that would mean the optimizer found a harmful direction, not a useful one)
and isn't so large it looks like an outright destructive re-init (catastrophic, order(s) of
magnitude beyond the noise floor).

Metric: 1 - mean cosine_similarity(pre_features, post_features) over a fixed batch of real images,
CLS token, encoder in eval() mode both times (no dropout/stochastic-depth noise in the comparison
itself). "shift" here always means this drift metric.
"""
from __future__ import annotations

import copy

import torch


@torch.no_grad()
def extract_cls_features(encoder, images: torch.Tensor) -> torch.Tensor:
    """encoder: anything with .forward(images) -> [B, D] (DinoViTEncoder's contract) or a timm
    backbone exposing forward_features (CLS at token 0); images: [B,3,H,W]. Returns [B, D], eval-mode."""
    was_training = encoder.training
    encoder.eval()
    if hasattr(encoder, "forward_features"):
        out = encoder.forward_features(images)[:, 0, :]
    else:
        out = encoder(images)
    encoder.train(was_training)
    return out


def feature_drift(pre_features: torch.Tensor, post_features: torch.Tensor) -> float:
    """1 - mean per-sample cosine similarity. 0 = identical features, 1 = orthogonal, 2 = opposite."""
    cos = torch.nn.functional.cosine_similarity(pre_features, post_features, dim=-1)
    return float((1.0 - cos).mean().item())


def random_perturbation_control(encoder, images: torch.Tensor, magnitude: float,
                                seed: int = 0) -> float:
    """Perturb every trainable-in-spirit weight tensor of `encoder` by additive Gaussian noise of
    `magnitude` std (matching the reference's "matched random 0.012 perturbation" methodology),
    measure the resulting feature drift, then RESTORE the original weights in-place.

    Returns the noise-floor drift value: a training run's drift should be the same order of
    magnitude as this (coordinated-but-not-destructive) or larger (a real, directional adaptation);
    a drift far below this floor would suggest training barely moved the encoder at all, and a
    drift wildly larger (order(s) of magnitude) suggests destructive/runaway change, not adaptation.
    """
    pre = extract_cls_features(encoder, images)
    original_state = copy.deepcopy(encoder.state_dict())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in encoder.parameters():
            noise = torch.randn(p.shape, generator=gen).to(p.device, p.dtype) * magnitude
            p.add_(noise)
    post = extract_cls_features(encoder, images)
    encoder.load_state_dict(original_state)
    return feature_drift(pre, post)
