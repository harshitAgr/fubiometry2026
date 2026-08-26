"""Axis-marginal likelihood regularization for landmark heatmaps.

The existing model predicts independent 2-D heatmaps.  This auxiliary objective compares
their normalized x/y marginals with the marginals of the Gaussian training targets.  It is
deliberately an auxiliary loss, not a SimCC decoder: inference and output coordinates stay
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_marginals(heatmaps: torch.Tensor, *, eps: float = 1e-8):
    """Return normalized ``(p_x, p_y)`` for ``[B,K,H,W]`` non-negative heatmaps.

    ``p_x`` has shape ``[B,K,W]`` and sums over image rows (height). ``p_y`` has
    shape ``[B,K,H]`` and sums over columns (width).  All-zero heatmaps remain
    all-zero rather than producing NaNs; ordinary sigmoid predictions and Gaussian
    targets always have positive mass.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"expected [B,K,H,W], got shape {tuple(heatmaps.shape)}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not heatmaps.is_floating_point():
        raise TypeError("heatmaps must be floating-point")
    if torch.any(heatmaps < 0):
        raise ValueError("heatmaps must be non-negative")

    mass = heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    normalized = heatmaps / mass
    p_x = normalized.sum(dim=-2)
    p_y = normalized.sum(dim=-1)
    return p_x, p_y


def _forward_kl(target_prob: torch.Tensor, pred_prob: torch.Tensor, eps: float) -> torch.Tensor:
    """Elementwise-safe KL(target || prediction), reduced over the final axis."""
    # xlogy makes the q==0 contribution exactly zero. The exp/log round trip also
    # gives an exact numerical zero when target_prob == pred_prob. With eps=1e-8
    # the ratio is bounded to a safe float32 range.
    log_ratio = (torch.log(target_prob.clamp_min(eps))
                 - torch.log(pred_prob.clamp_min(eps)))
    return torch.special.xlogy(target_prob, torch.exp(log_ratio)).sum(dim=-1)


def marginal_kl_loss(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8,
                     reduction: str = "mean") -> torch.Tensor:
    """Compute ``0.5 * (KL(q_x||p_x) + KL(q_y||p_y))``.

    ``pred`` is the sigmoid-activated model output and ``target`` is the Gaussian
    heatmap target.  The unreduced result has shape ``[B,K]``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    p_x, p_y = heatmap_marginals(pred.float(), eps=eps)
    q_x, q_y = heatmap_marginals(target.float(), eps=eps)
    loss = 0.5 * (_forward_kl(q_x, p_x, eps) + _forward_kl(q_y, p_y, eps))
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def mse_with_marginal_kl(pred: torch.Tensor, target: torch.Tensor, beta: float,
                         *, eps: float = 1e-8) -> torch.Tensor:
    """MSE plus fixed-weight marginal KL, with an exact beta-zero baseline path."""
    if beta < 0:
        raise ValueError("marginal KL beta must be non-negative")
    mse = F.mse_loss(pred, target)
    if beta == 0.0:
        return mse
    return mse + beta * marginal_kl_loss(pred, target, eps=eps)
