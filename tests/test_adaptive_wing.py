"""Numerical and dispatch tests for the core Adaptive Wing loss."""

import math
import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.adaptive_wing import adaptive_wing_loss  # noqa: E402
from experiments.run_config import select_task_loss, validate_loss_config  # noqa: E402


def _scalar_reference(pred, target, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
    delta = abs(target - pred)
    exponent = alpha - target
    if delta < theta:
        return omega * math.log1p((delta / epsilon) ** exponent)
    ratio = theta / epsilon
    coefficient = (omega / (1.0 + ratio ** exponent) * exponent
                   * ratio ** (exponent - 1.0) / epsilon)
    intercept = theta * coefficient - omega * math.log1p(ratio ** exponent)
    return coefficient * delta - intercept


def test_matches_independent_scalar_reference_on_both_branches():
    pred = torch.tensor([0.0, 0.1, 0.4, 0.9, 1.0], dtype=torch.float64)
    target = torch.tensor([0.0, 0.8, 0.2, 0.3, 1.0], dtype=torch.float64)
    got = adaptive_wing_loss(pred, target, reduction="none")
    expected = torch.tensor([_scalar_reference(p, t) for p, t in zip(pred, target)],
                            dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("target", [0.0, 0.25, 0.75, 1.0])
def test_value_and_derivative_are_continuous_at_theta(target):
    eps = 1e-7
    x = torch.tensor([target + 0.5 - eps, target + 0.5 + eps],
                     dtype=torch.float64, requires_grad=True)
    y = torch.full_like(x, target)
    losses = adaptive_wing_loss(x, y, reduction="none")
    grads = torch.autograd.grad(losses.sum(), x)[0]
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-5)


def test_zero_error_has_zero_loss_and_finite_zero_gradient():
    pred = torch.linspace(0.0, 1.0, 17, dtype=torch.float64, requires_grad=True)
    loss = adaptive_wing_loss(pred, pred.detach())
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad, torch.zeros_like(pred.grad))


def test_realistic_heatmaps_have_finite_loss_and_gradients():
    grid_y, grid_x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    target = torch.exp(-((grid_x - 31.2) ** 2 + (grid_y - 28.7) ** 2) / (2 * 1.8 ** 2))
    target = target.repeat(2, 4, 1, 1)
    logits = torch.randn_like(target, requires_grad=True)
    loss = adaptive_wing_loss(torch.sigmoid(logits), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("tid", ["A4C", "PLAX", "PSAX", "FUGC", "fetal_femur",
                                  "FA", "AOP", "IVC", "HC"])
def test_default_dispatch_remains_byte_identical_mse(tid):
    pred = torch.rand(2, 4, 8, 8)
    target = torch.rand_like(pred)
    assert torch.equal(select_task_loss(tid, pred, target), F.mse_loss(pred, target))


def test_adaptive_wing_dispatches_globally():
    pred = torch.rand(2, 4, 8, 8)
    target = torch.rand_like(pred)
    expected = adaptive_wing_loss(pred, target)
    for tid in ("AOP", "IVC", "HC", "FUGC"):
        torch.testing.assert_close(
            select_task_loss(tid, pred, target, heatmap_loss="adaptive_wing"), expected)


def test_rejects_mixed_experimental_loss_levers():
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_loss_config("adaptive_wing", aop_p3_weight=2.0)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_loss_config("adaptive_wing", param_loss_beta=0.1)


def test_marginal_kl_rejects_other_loss_levers():
    betas = {"HC": 0.01}
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_loss_config("adaptive_wing", marginal_kl_betas=betas)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_loss_config("mse", aop_p3_weight=2.0, marginal_kl_betas=betas)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_loss_config("mse", param_loss_beta=0.1, marginal_kl_betas=betas)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 1.0}, {"omega": 0.0}, {"epsilon": 0.0}, {"theta": 0.0},
])
def test_invalid_hyperparameters_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        adaptive_wing_loss(torch.zeros(1), torch.zeros(1), **kwargs)
