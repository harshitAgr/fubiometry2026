"""Differentiable downstream parameter loss (IVC distance / HC ellipse circumference).

Companion to the AOP per-landmark loss weighting lever (`run_config.weighted_heatmap_mse`), but a
DIFFERENT mechanism: that lever reweights per-landmark heatmap MSE independently (still
landmark-independent, just unequal weight) and was REJECTED because AOP's p3 error is
ambiguity-bound, not focus-bound. This module instead COUPLES the landmarks that jointly define a
physical measurement — it adds a loss term equal to the error of the DERIVED parameter (IVC
diameter distance; HC ellipse perimeter) computed directly from predicted coordinates, targeting a
correlated/coupled error mode (diagnosed for IVC as run-to-run derived-diameter VARIANCE, and
for HC as ellipse-endpoint-consistency failures) rather than
a single-landmark localization problem.

Requires torch -> run tests with the BASELINE venv (see tests/test_param_loss.py header).

Coordinate space: everything here operates in NORMALIZED heatmap-space coordinates
(x,y each in [0,1], matching `experiments/decode.py`'s convention: x_norm = x_cell/(W-1)) — the
same space the Gaussian heatmap targets are built in (`kp_aug_dataset._gen_heatmaps_per_task`).
This requires no additional per-sample pixel-scale plumbing through the training loop/collate
function; the param-loss magnitude is consequently in normalized units, not pixels, which is why
alpha/beta must be calibrated empirically (see reproduce_param_loss_probe.sh) rather than assumed
to be commensurate with MRE-px intuitions.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Differentiable soft-argmax (the new primitive)
# ---------------------------------------------------------------------------

def soft_argmax_coords(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """[B, K, H, W] heatmaps -> [B, K, 2] normalized (x, y) expected-value coordinates.

    Spatial softmax (over the flattened H*W, with `temperature`) then an expected-value
    coordinate readout over a fixed coordinate grid normalized to [0, 1] via /(W-1), /(H-1) —
    matching `decode.py.decode_subpixel`'s normalization convention exactly (so a soft-argmax
    coordinate and a `decode_subpixel` coordinate on the same heatmap are directly comparable).

    `temperature` divides the logits before softmax (lower -> peakier/closer to hard-argmax,
    higher -> smoother/more averaged). heatmaps here are already POST-SIGMOID activations in
    [0, 1] (matching how `pred` is used elsewhere in run_config.py's training loop), so we take
    log(heatmap + eps) as the softmax logit -- this makes a single dominant peak in the sigmoid
    map dominate the softmax, while temperature=1.0 with an eps=1e-6 floor keeps flat/near-zero
    maps numerically stable (uniform softmax -> centroid of the grid, a benign fallback that
    mirrors decode.py's own flat-map guard).
    """
    if heatmaps.dim() != 4:
        raise ValueError(f"expected [B,K,H,W], got shape {tuple(heatmaps.shape)}")
    B, K, H, W = heatmaps.shape
    eps = 1e-6
    logits = torch.log(heatmaps.clamp(min=0.0) + eps) / temperature
    flat = logits.reshape(B, K, H * W)
    probs = torch.softmax(flat, dim=-1).reshape(B, K, H, W)

    device, dtype = heatmaps.device, heatmaps.dtype
    ys = torch.arange(H, device=device, dtype=dtype) / max(H - 1, 1)
    xs = torch.arange(W, device=device, dtype=dtype) / max(W - 1, 1)
    # expected value under probs
    x_coord = torch.einsum("bkhw,w->bk", probs, xs)
    y_coord = torch.einsum("bkhw,h->bk", probs, ys)
    return torch.stack([x_coord, y_coord], dim=-1)  # [B, K, 2]


# ---------------------------------------------------------------------------
# Torch port of scoring.geometry.ellipse_perimeter (Ramanujan II) — differentiable
# ---------------------------------------------------------------------------

def ellipse_perimeter_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ramanujan II ellipse-perimeter approximation, elementwise, differentiable.

    Direct torch port of `scoring.geometry.ellipse_perimeter`: h = (a-b)^2 / (a+b)^2 (0 when
    a+b==0), perimeter = pi*(a+b)*(1 + 3h/(10 + sqrt(4-3h))). Must match the numpy original to
    floating-point precision on the same inputs — see
    tests/test_param_loss.py::test_ellipse_perimeter_matches_numpy.
    """
    denom = a + b
    safe_denom_sq = (denom * denom).clamp(min=1e-30)  # avoid 0/0 in the unused branch of where()
    h = torch.where(denom != 0, ((a - b) ** 2) / safe_denom_sq, torch.zeros_like(a))
    return torch.pi * denom * (1 + 3 * h / (10 + torch.sqrt((4 - 3 * h).clamp(min=0.0))))


# ---------------------------------------------------------------------------
# Param-derivation from soft-argmax coords (mirrors scoring.derive for IVC/HC only)
# ---------------------------------------------------------------------------

def ivc_diameter(coords: torch.Tensor) -> torch.Tensor:
    """coords: [B, 2, 2] (2 landmarks, x/y) -> [B] euclidean distance (IVC param, PARAM_SPECS (0,1))."""
    if coords.shape[-2:] != (2, 2):
        raise ValueError(f"expected [B,2,2], got {tuple(coords.shape)}")
    diff = coords[:, 0, :] - coords[:, 1, :]
    return torch.linalg.norm(diff, dim=-1)


def hc_circumference(coords: torch.Tensor) -> torch.Tensor:
    """coords: [B, 4, 2] (4 landmarks, x/y) -> [B] ellipse perimeter (HC param, PARAM_SPECS (0,1,2,3)).

    Mirrors scoring.derive.derive_from_specs's ellipse_perimeter branch: a = |p0-p1|/2 (minor
    axis half-length), b = |p2-p3|/2 (major axis half-length).
    """
    if coords.shape[-2:] != (4, 2):
        raise ValueError(f"expected [B,4,2], got {tuple(coords.shape)}")
    a = torch.linalg.norm(coords[:, 0, :] - coords[:, 1, :], dim=-1) / 2.0
    b = torch.linalg.norm(coords[:, 2, :] - coords[:, 3, :], dim=-1) / 2.0
    return ellipse_perimeter_torch(a, b)


# ---------------------------------------------------------------------------
# Param losses (predicted heatmaps + GT heatmaps -> scalar loss, via soft-argmax on BOTH)
# ---------------------------------------------------------------------------

def _decode_gt_coords(gt_heatmaps: torch.Tensor) -> torch.Tensor:
    """GT coords from GT heatmaps via the SAME soft-argmax readout (keeps pred/GT in an
    identical coordinate convention; GT Gaussian targets are near-one-hot so this recovers the
    GT landmark location to sub-cell precision, consistent with how the GT heatmap was built
    from a normalized coordinate in kp_aug_dataset._gen_heatmaps_per_task)."""
    return soft_argmax_coords(gt_heatmaps)


def ivc_param_loss(pred_heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """Mean-squared error between the predicted and GT IVC diameter (differentiable).

    pred_heatmaps, gt_heatmaps: [B, 2, H, W] (K=2 landmarks; IVC PARAM_SPECS (0,1)).
    """
    pred_coords = soft_argmax_coords(pred_heatmaps, temperature)
    gt_coords = _decode_gt_coords(gt_heatmaps)
    pred_d = ivc_diameter(pred_coords)
    gt_d = ivc_diameter(gt_coords)
    return torch.mean((pred_d - gt_d) ** 2)


def hc_param_loss(pred_heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """Mean-squared error between the predicted and GT HC ellipse circumference (differentiable).

    pred_heatmaps, gt_heatmaps: [B, 4, H, W] (K=4 landmarks; HC PARAM_SPECS (0,1,2,3)).
    """
    pred_coords = soft_argmax_coords(pred_heatmaps, temperature)
    gt_coords = _decode_gt_coords(gt_heatmaps)
    pred_c = hc_circumference(pred_coords)
    gt_c = hc_circumference(gt_coords)
    return torch.mean((pred_c - gt_c) ** 2)


# ---------------------------------------------------------------------------
# Combined training loss: alpha * heatmap_mse + beta * param_loss
# ---------------------------------------------------------------------------

def combined_loss(pred_heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor, task_id: str,
                   alpha: float = 1.0, beta: float = 0.0,
                   temperature: float = 1.0) -> torch.Tensor:
    """alpha * F.mse_loss(pred, gt) + beta * param_loss(pred, gt) for task_id in {"IVC", "HC"}.

    DO-NO-HARM: beta=0.0 must reproduce `alpha * F.mse_loss(pred_heatmaps, gt_heatmaps)` EXACTLY
    (see tests/test_param_loss.py::test_beta_zero_is_do_no_harm) — the param term is computed
    only when beta != 0, so a beta=0 call never touches soft_argmax_coords/ellipse math at all,
    matching this repo's established do-no-harm pattern (weighted_heatmap_mse with W=1).
    Callers outside {"IVC","HC"} should not call this at all (run_config gates on task_id
    before dispatching here) -- unsupported task_ids raise, they don't silently no-op.
    """
    import torch.nn.functional as F

    heatmap_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
    if beta == 0.0:
        return alpha * heatmap_loss
    if task_id == "IVC":
        param_loss = ivc_param_loss(pred_heatmaps, gt_heatmaps, temperature)
    elif task_id == "HC":
        param_loss = hc_param_loss(pred_heatmaps, gt_heatmaps, temperature)
    else:
        raise ValueError(f"combined_loss only supports task_id in {{'IVC','HC'}}, got {task_id!r}")
    return alpha * heatmap_loss + beta * param_loss
