"""Adaptive Wing heatmap-regression loss.

Implements Eq. (3) from Wang et al., ICCV 2019.  This is deliberately only the
core loss: the paper's weighted loss map is a separate experimental lever.
Inputs are expected to be sigmoid predictions and Gaussian targets in [0, 1].
"""

import torch


def adaptive_wing_loss(pred, target, *, alpha=2.1, omega=14.0, epsilon=1.0,
                       theta=0.5, reduction="mean"):
    """Return the elementwise Adaptive Wing loss with the requested reduction.

    The default hyperparameters are those used in the paper and its reference
    implementations.  ``alpha > 1`` keeps ``alpha - target`` positive for
    targets in [0, 1].
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1 for targets in [0, 1]")
    if omega <= 0.0:
        raise ValueError("omega must be > 0")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if theta <= 0.0:
        raise ValueError("theta must be > 0")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction!r}")

    delta = torch.abs(target - pred)
    exponent = alpha - target
    theta_over_epsilon = theta / epsilon
    theta_power = torch.pow(theta_over_epsilon, exponent)

    # The linear branch is tangent to the logarithmic branch at delta=theta.
    coefficient = (
        omega
        * (1.0 / (1.0 + theta_power))
        * exponent
        * torch.pow(theta_over_epsilon, exponent - 1.0)
        / epsilon
    )
    intercept = theta * coefficient - omega * torch.log1p(theta_power)

    nonlinear = omega * torch.log1p(torch.pow(delta / epsilon, exponent))
    linear = coefficient * delta - intercept
    loss = torch.where(delta < theta, nonlinear, linear)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
