"""Unit tests for the FA-head wall-variance-reshaping loss lever (2026-08-07 FA diagnosis):
--fa-wall-weight (weighted MSE on p2/p3) and the --fa-aniso-sigma parser + scope guard.

Requires torch -> run with the BASELINE venv (NOT the project .venv):
  baseline/.venv-baseline/bin/python -m pytest tests/test_fa_head_loss.py -q
"""
import os
import sys

import pytest

pytest.importorskip("torch")  # torch-only test -> skipped under the project .venv; run via baseline venv

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_config import (  # noqa: E402
    select_task_loss, validate_loss_config, validate_fa_config, parse_fa_aniso_sigma,
    weighted_heatmap_mse,
)


def _pred_hm(K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, K, 16, 16, generator=g)
    hm = torch.rand(2, K, 16, 16, generator=g)
    return pred, hm


# ---------------------------------------------------------------------------
# parse_fa_aniso_sigma -- pure string parsing
# ---------------------------------------------------------------------------

def test_parse_fa_aniso_sigma_none_passes_through():
    assert parse_fa_aniso_sigma(None) is None


def test_parse_fa_aniso_sigma_valid():
    assert parse_fa_aniso_sigma("1.2,2.7") == (1.2, 2.7)
    assert parse_fa_aniso_sigma("3,4") == (3.0, 4.0)


@pytest.mark.parametrize("spec", ["1.2", "1.2,2.7,3.0", "a,b", "", "1.2,-2.7", "-1.2,2.7", "0,2.7"])
def test_parse_fa_aniso_sigma_rejects_malformed(spec):
    with pytest.raises(ValueError):
        parse_fa_aniso_sigma(spec)


# ---------------------------------------------------------------------------
# select_task_loss FA branch -- do-no-harm defaults + weighted-MSE wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["A4C", "PLAX", "PSAX", "FUGC", "fetal_femur", "AOP", "IVC", "HC"])
def test_fa_wall_weight_does_not_affect_other_tasks(tid):
    """fa_wall_weight only gates on tid=='FA'; every other task ignores it entirely."""
    pred, hm = _pred_hm(K=4)
    got = select_task_loss(tid, pred, hm, fa_wall_weight=5.0)
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected), tid


def test_fa_default_weight_is_plain_mse():
    """fa_wall_weight=1.0 (default) -> FA takes the plain F.mse_loss branch, byte-identical."""
    pred, hm = _pred_hm(K=4, seed=1)
    got = select_task_loss("FA", pred, hm)  # default fa_wall_weight=1.0
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected)


def test_fa_nonzero_wall_weight_matches_hand_computation():
    pred, hm = _pred_hm(K=4, seed=2)
    got = select_task_loss("FA", pred, hm, fa_wall_weight=3.0)
    expected = weighted_heatmap_mse(pred, hm, [1.0, 1.0, 3.0, 3.0])
    assert torch.allclose(got, expected)


def test_fa_wall_weight_upweights_p2_p3_only():
    """[1,1,3,3] normalization behaviour: uniform error across channels -> weight is a no-op
    (mean-1 normalization), but error concentrated on p2/p3 raises the loss vs uniform weights."""
    pred = torch.zeros(1, 4, 4, 4)
    tgt = torch.zeros(1, 4, 4, 4)
    tgt[0, 2] = 1.0  # error only on p2 (a wall landmark)
    uniform = select_task_loss("FA", pred, tgt, fa_wall_weight=1.0)
    upweighted = select_task_loss("FA", pred, tgt, fa_wall_weight=3.0)
    assert upweighted > uniform, (uniform.item(), upweighted.item())


def test_fa_wall_weight_uniform_scale_equals_plain_mse():
    """Any constant [c,c,c,c] normalizes to mean 1 -> exactly plain MSE (do-no-harm identity
    shared with the AOP lever's weighted_heatmap_mse)."""
    pred, hm = _pred_hm(K=4, seed=3)
    exp = F.mse_loss(pred, hm)
    got = weighted_heatmap_mse(pred, hm, [2.0, 2.0, 2.0, 2.0])
    assert torch.allclose(got, exp, atol=1e-7)


def test_fa_wall_weight_and_aop_p3_weight_are_independent():
    """The two per-landmark weighting levers gate on different task_ids and don't interact."""
    pred, hm = _pred_hm(K=4, seed=4)
    fa_only = select_task_loss("FA", pred, hm, fa_wall_weight=3.0, aop_p3_weight=5.0)
    fa_expected = weighted_heatmap_mse(pred, hm, [1.0, 1.0, 3.0, 3.0])
    assert torch.allclose(fa_only, fa_expected)
    aop_only = select_task_loss("AOP", pred, hm, fa_wall_weight=3.0, aop_p3_weight=5.0)
    aop_expected = weighted_heatmap_mse(pred, hm, [1.0, 1.0, 1.0, 5.0])
    assert torch.allclose(aop_only, aop_expected)


# ---------------------------------------------------------------------------
# validate_loss_config -- fa_wall_weight participates in the mutual-exclusion checks
# ---------------------------------------------------------------------------

def test_validate_loss_config_defaults_pass():
    validate_loss_config()  # must not raise


def test_validate_loss_config_rejects_fa_wall_weight_with_adaptive_wing():
    with pytest.raises(ValueError):
        validate_loss_config(heatmap_loss="adaptive_wing", fa_wall_weight=3.0)


def test_validate_loss_config_rejects_fa_wall_weight_with_marginal_kl():
    with pytest.raises(ValueError):
        validate_loss_config(marginal_kl_betas={"FA": 0.0}, fa_wall_weight=3.0)


def test_validate_loss_config_accepts_fa_wall_weight_alone():
    validate_loss_config(fa_wall_weight=3.0)  # must not raise


# ---------------------------------------------------------------------------
# validate_fa_config -- scope guard (train-task + aug combination)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["A4C", "AOP", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur", None])
def test_fa_wall_weight_requires_train_task_fa(tid):
    with pytest.raises(ValueError):
        validate_fa_config(tid, "photo_v1", fa_wall_weight=3.0)


@pytest.mark.parametrize("tid", ["A4C", "AOP", "HC", None])
def test_fa_aniso_sigma_requires_train_task_fa(tid):
    with pytest.raises(ValueError):
        validate_fa_config(tid, "photo_v1", fa_aniso_sigma=(1.2, 2.7))


def test_fa_config_defaults_never_raise_regardless_of_train_task():
    for tid in ("A4C", "AOP", "FA", None, "HC"):
        for aug in ("none", "photo_v1", "geo_v1"):
            validate_fa_config(tid, aug)  # both knobs at default -> always fine


def test_fa_wall_weight_with_train_task_fa_any_aug_ok():
    for aug in ("none", "photo_v1", "geo_v1", "geo_v1_hcsmall"):
        validate_fa_config("FA", aug, fa_wall_weight=3.0)  # loss-only, doesn't touch the dataset


@pytest.mark.parametrize("aug", ["photo_v1", "geo_v1", "aop_robust_v1", "fugc_scale_v1", "geo_v1_hcsmall"])
def test_fa_aniso_sigma_supported_augs_ok(aug):
    validate_fa_config("FA", aug, fa_aniso_sigma=(1.2, 2.7))  # must not raise


@pytest.mark.parametrize("aug", ["none"])
def test_fa_aniso_sigma_rejects_unsupported_aug(aug):
    with pytest.raises(ValueError):
        validate_fa_config("FA", aug, fa_aniso_sigma=(1.2, 2.7))


if __name__ == "__main__":
    print("run via: baseline/.venv-baseline/bin/python -m pytest tests/test_fa_head_loss.py -q")
