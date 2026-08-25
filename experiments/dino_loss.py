"""DINO self-distillation loss + EMA math (student/teacher, CLS-token, multi-crop).

Pure tensor math, no model/data dependencies -> unit-testable in either venv (only needs torch).
Simplification vs full DINOv2 (stated honestly, not hidden): CLS-token DINO loss only — no iBOT
patch-level loss, no KoLeo regularizer. See experiments/dino_pretrain.py module docstring.

Loss (Caron et al. 2021, DINO): student sees ALL crops (global+local), teacher sees ONLY global
crops (never local -> the asymmetry that drives local-to-global mimicry, DINO Sec 3). Both project
CLS features through the same-architecture head; teacher's output is centered (subtract a running
mean, prevents one dimension from dominating -> collapse mode 1) and sharpened with a LOW
temperature (teacher_temp < student_temp -> sharper target than prediction -> collapse mode 2,
"uniform output", is disfavoured because cross-entropy against a peaked target is minimized at a
peaked student prediction, not a uniform one). Cross-entropy is standard softmax(target) . -log
softmax(pred); we skip same-view (student_i vs teacher_i) pairs per the original recipe (a view
predicting itself is a trivial/free minimum, not a training signal).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dino_loss(student_out, teacher_out, student_temp: float, teacher_temp: float,
              center: torch.Tensor, n_global: int):
    """Cross-entropy of every (teacher_global, student_any) pair, skipping same-view pairs.

    Args:
        student_out: [n_student_crops, B, D] logits, one per crop (all crops, global+local).
        teacher_out: [n_global, B, D] logits, GLOBAL crops only.
        student_temp, teacher_temp: softmax temperatures (teacher_temp should be << student_temp).
        center: [D] running center subtracted from teacher logits before sharpening.
        n_global: number of teacher (global) crops; student crops [0:n_global] are the same
            global views in the same order, so pair (t=i, s=i) is the disallowed same-view pair.

    Returns: scalar mean cross-entropy over all valid (teacher_view, student_view) pairs.
    """
    student_logp = F.log_softmax(student_out / student_temp, dim=-1)  # [Ns,B,D]
    teacher_p = F.softmax((teacher_out - center) / teacher_temp, dim=-1).detach()  # [Ng,B,D]

    total = student_out.new_zeros(())
    n_terms = 0
    n_student = student_out.shape[0]
    for t in range(n_global):
        for s in range(n_student):
            if s == t:  # same-view pair is a free/trivial minimum, not a learning signal
                continue
            ce = -(teacher_p[t] * student_logp[s]).sum(dim=-1).mean()
            total = total + ce
            n_terms += 1
    return total / max(n_terms, 1)


def update_center(center: torch.Tensor, teacher_out: torch.Tensor, momentum: float) -> torch.Tensor:
    """EMA update of the teacher-output center (Caron et al. Alg.1): center <- m*center + (1-m)*batch_mean.

    teacher_out: [n_global, B, D] raw (pre-softmax) teacher logits for this step.
    Batch mean is over crops AND batch (every teacher logit contributes equally).
    """
    batch_mean = teacher_out.detach().reshape(-1, teacher_out.shape[-1]).mean(dim=0)
    return momentum * center + (1.0 - momentum) * batch_mean


def ema_update_(teacher_params, student_params, momentum: float) -> None:
    """In-place EMA blend: teacher_p <- momentum*teacher_p + (1-momentum)*student_p, per-tensor.

    Operates on any two equal-length sequences of same-shape tensors (works for both
    model.parameters() and a state_dict's .values() — buffers like BN running stats are handled
    the same way by the caller passing state_dict values instead of parameters).
    """
    with torch.no_grad():
        for t_p, s_p in zip(teacher_params, student_params):
            t_p.mul_(momentum).add_(s_p, alpha=1.0 - momentum)


def cosine_momentum_schedule(step: int, total_steps: int, base: float, final: float = 1.0) -> float:
    """Cosine ramp of the teacher EMA momentum from `base` to `final` over training (DINO Sec 3)."""
    if total_steps <= 1:
        return final
    import math
    prog = min(max(step, 0), total_steps) / total_steps
    return final - 0.5 * (final - base) * (1 + math.cos(math.pi * prog))
