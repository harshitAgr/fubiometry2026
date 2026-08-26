"""Unit tests for the AOP per-landmark weighted-MSE loss helper (Lever A).

Requires torch -> run with the BASELINE venv (NOT the project .venv):
  baseline/.venv-baseline/bin/python -m pytest tests/test_aop_loss.py -q
  (or run this file directly: baseline/.venv-baseline/bin/python tests/test_aop_loss.py)
"""
import os
import sys

import pytest

pytest.importorskip("torch")  # torch-only test -> skipped under the project .venv; run via baseline venv

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_config import weighted_heatmap_mse  # noqa: E402


def _rand(seed=0):
    torch.manual_seed(seed)
    return torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8)


def test_uniform_weights_equals_plain_mse():
    """[1,1,1,1] must reproduce F.mse_loss EXACTLY (do-no-harm guarantee)."""
    pred, tgt = _rand()
    got = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 1.0])
    exp = F.mse_loss(pred, tgt)
    assert torch.allclose(got, exp, atol=1e-7), (got.item(), exp.item())


def test_equal_weights_any_scale_equals_mse():
    """Normalization to mean 1 => any constant weight vector == plain MSE."""
    pred, tgt = _rand()
    exp = F.mse_loss(pred, tgt)
    for c in (2.0, 0.5, 7.0):
        got = weighted_heatmap_mse(pred, tgt, [c, c, c, c])
        assert torch.allclose(got, exp, atol=1e-7), (c, got.item(), exp.item())


def test_weighted_value_matches_hand_computation():
    """[1,1,1,3] -> normalized [.667,.667,.667,2.0] applied to per-channel mean SE."""
    pred, tgt = _rand()
    per_ch = ((pred - tgt) ** 2).mean(dim=(0, 2, 3))   # [K]
    w = torch.tensor([1.0, 1.0, 1.0, 3.0]); w = w / w.mean()
    exp = (per_ch * w).mean()
    got = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 3.0])
    assert torch.allclose(got, exp, atol=1e-7), (got.item(), exp.item())


def test_upweighting_p3_raises_loss_when_p3_is_the_worst():
    """If p3 carries the largest error, upweighting it must increase the loss vs uniform."""
    pred = torch.zeros(1, 4, 4, 4)
    tgt = torch.zeros(1, 4, 4, 4)
    tgt[0, 3] = 1.0  # only the p3 channel has error
    uniform = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 1.0])
    upweighted = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 3.0])
    assert upweighted > uniform, (uniform.item(), upweighted.item())


def test_upweighting_p3_lowers_loss_when_p3_is_the_best():
    """Sanity on direction: if p3 is the BEST channel, upweighting it lowers the loss."""
    pred = torch.zeros(1, 4, 4, 4)
    tgt = torch.zeros(1, 4, 4, 4)
    tgt[0, 0] = 1.0  # error is on p0, not p3
    uniform = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 1.0])
    upweighted = weighted_heatmap_mse(pred, tgt, [1.0, 1.0, 1.0, 3.0])
    assert upweighted < uniform, (uniform.item(), upweighted.item())


if __name__ == "__main__":
    test_uniform_weights_equals_plain_mse()
    test_equal_weights_any_scale_equals_mse()
    test_weighted_value_matches_hand_computation()
    test_upweighting_p3_raises_loss_when_p3_is_the_worst()
    test_upweighting_p3_lowers_loss_when_p3_is_the_best()
    print("OK: all 5 AOP weighted-MSE tests passed")
