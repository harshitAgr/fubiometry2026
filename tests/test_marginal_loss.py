"""Tests for the axis-marginal heatmap auxiliary objective."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.marginal_loss import (  # noqa: E402
    heatmap_marginals,
    marginal_kl_loss,
    mse_with_marginal_kl,
)


def test_axis_convention_and_normalization():
    hm = torch.zeros(1, 1, 3, 4)
    hm[0, 0, 2, 1] = 3.0
    px, py = heatmap_marginals(hm)
    assert px.shape == (1, 1, 4)
    assert py.shape == (1, 1, 3)
    assert torch.equal(px, torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]))
    assert torch.equal(py, torch.tensor([[[0.0, 0.0, 1.0]]]))
    assert torch.equal(px.sum(-1), torch.ones(1, 1))
    assert torch.equal(py.sum(-1), torch.ones(1, 1))


def test_perfect_prediction_has_zero_kl():
    g = torch.Generator().manual_seed(7)
    target = torch.rand(2, 4, 17, 13, generator=g).clamp_min(1e-5)
    got = marginal_kl_loss(target, target, reduction="none")
    assert torch.equal(got, torch.zeros_like(got))


def test_marginal_kl_detects_axis_shift():
    target = torch.full((1, 1, 9, 9), 1e-4)
    target[0, 0, 4, 2] = 1.0
    shifted = torch.full_like(target, 1e-4)
    shifted[0, 0, 4, 6] = 1.0
    assert marginal_kl_loss(shifted, target) > 0.5


def test_gaussian_target_marginal_mean_recovers_interior_coordinate_and_exposes_border_bias():
    height = width = 64
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    def gaussian(x, y):
        return torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.8 ** 2))[None, None]

    interior_x, interior_y = 21.3, 42.7
    px, py = heatmap_marginals(gaussian(interior_x, interior_y))
    recovered_x = (px * torch.arange(width)).sum()
    recovered_y = (py * torch.arange(height)).sum()
    assert abs(float(recovered_x) - interior_x) < 1e-4
    assert abs(float(recovered_y) - interior_y) < 1e-4

    border_px, _ = heatmap_marginals(gaussian(0.0, interior_y))
    border_mean_x = (border_px * torch.arange(width)).sum()
    assert 0.9 < float(border_mean_x) < 1.2  # expected truncation bias, audited on real targets


def test_beta_zero_is_exact_mse_and_exact_optimizer_update():
    g = torch.Generator().manual_seed(11)
    x = torch.rand(2, 3, 7, 7, generator=g)
    target = torch.rand(2, 2, 7, 7, generator=g)
    first = torch.nn.Conv2d(3, 2, 1)
    second = torch.nn.Conv2d(3, 2, 1)
    second.load_state_dict(first.state_dict())
    opt_first = torch.optim.AdamW(first.parameters(), lr=1e-3)
    opt_second = torch.optim.AdamW(second.parameters(), lr=1e-3)

    pred_first = torch.sigmoid(first(x))
    pred_second = torch.sigmoid(second(x))
    loss_first = F.mse_loss(pred_first, target)
    loss_second = mse_with_marginal_kl(pred_second, target, beta=0.0)
    assert torch.equal(loss_first, loss_second)
    loss_first.backward()
    loss_second.backward()
    for p_first, p_second in zip(first.parameters(), second.parameters()):
        assert torch.equal(p_first.grad, p_second.grad)
    opt_first.step()
    opt_second.step()
    for p_first, p_second in zip(first.parameters(), second.parameters()):
        assert torch.equal(p_first, p_second)


@pytest.mark.parametrize("fill", [-100.0, 0.0, 100.0])
def test_flat_or_saturated_inputs_have_finite_loss_and_gradients(fill):
    logits = torch.full((2, 3, 8, 8), fill, requires_grad=True)
    pred = torch.sigmoid(logits)
    target = torch.zeros_like(pred)
    target[:, :, 0, 0] = 1.0
    loss = marginal_kl_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        heatmap_marginals(torch.ones(2, 3, 4))
    with pytest.raises(ValueError, match="non-negative"):
        heatmap_marginals(-torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        marginal_kl_loss(torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        mse_with_marginal_kl(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), -0.1)
