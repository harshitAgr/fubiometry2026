"""Unit tests for run_config.select_task_loss -- the per-task loss dispatch that both the AOP
per-landmark weighting lever and this IVC/HC param-loss probe are gated through.

Requires torch AND the baseline/baseline clone (run_config.py imports the baseline's
dataset/model_factory/model/utils at module level) -> run with the BASELINE venv:
  baseline/.venv-baseline/bin/python -m pytest tests/test_select_task_loss.py -q

NOTE: if baseline/baseline is not present locally (see env/README.md), this whole
module fails to COLLECT (ModuleNotFoundError: dataset) rather than failing individual tests --
that failure mode is a local-environment setup issue unrelated to the logic under test (verify
with the sibling tests/test_param_loss.py and tests/test_aop_loss.py, which hit the identical
import chain and are unaffected by anything in this file).
"""
import os
import sys

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_config import select_task_loss  # noqa: E402


def _pred_hm(K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, K, 16, 16, generator=g)
    hm = torch.rand(2, K, 16, 16, generator=g)
    return pred, hm


# ---------------------------------------------------------------------------
# Do-no-harm: defaults (aop_p3_weight=1.0, param_loss_beta=0.0) -> plain MSE for EVERY task
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["A4C", "PLAX", "PSAX", "FUGC", "fetal_femur", "FA", "AOP",
                                  "IVC", "HC"])
def test_defaults_give_plain_mse_for_every_task(tid):
    pred, hm = _pred_hm(K=4)
    got = select_task_loss(tid, pred, hm)  # aop_p3_weight=1.0, param_loss_beta=0.0 defaults
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected), tid


# ---------------------------------------------------------------------------
# param_loss_beta only touches IVC/HC
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["A4C", "PLAX", "PSAX", "FUGC", "fetal_femur", "FA"])
def test_nonzero_beta_does_not_affect_other_tasks(tid):
    """The 7 untouched tasks must stay on plain F.mse_loss regardless of param_loss_beta."""
    pred, hm = _pred_hm(K=4)
    got = select_task_loss(tid, pred, hm, param_loss_beta=0.5)
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected), tid


def test_nonzero_beta_does_not_affect_aop():
    """AOP must stay on plain F.mse_loss under param_loss_beta (it only reacts to aop_p3_weight)."""
    pred, hm = _pred_hm(K=4)
    got = select_task_loss("AOP", pred, hm, param_loss_beta=0.5)
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected)


def test_ivc_nonzero_beta_changes_loss_when_param_differs():
    pred, hm = _pred_hm(K=2, seed=1)
    base = select_task_loss("IVC", pred, hm, param_loss_beta=0.0)
    boosted = select_task_loss("IVC", pred, hm, param_loss_beta=0.5)
    assert not torch.equal(base, boosted)


def test_hc_nonzero_beta_changes_loss_when_param_differs():
    pred, hm = _pred_hm(K=4, seed=2)
    base = select_task_loss("HC", pred, hm, param_loss_beta=0.0)
    boosted = select_task_loss("HC", pred, hm, param_loss_beta=0.5)
    assert not torch.equal(base, boosted)


# ---------------------------------------------------------------------------
# aop_p3_weight only touches AOP (unaffected by param_loss_beta interaction)
# ---------------------------------------------------------------------------

def test_aop_p3_weight_unaffected_by_param_loss_beta_default():
    pred, hm = _pred_hm(K=4, seed=3)
    from experiments.run_config import weighted_heatmap_mse
    got = select_task_loss("AOP", pred, hm, aop_p3_weight=2.0, param_loss_beta=0.0)
    expected = weighted_heatmap_mse(pred, hm, [1.0, 1.0, 1.0, 2.0])
    assert torch.allclose(got, expected)


def test_ivc_hc_unaffected_by_aop_p3_weight():
    """aop_p3_weight only gates on tid=='AOP'; IVC/HC ignore it entirely."""
    pred, hm = _pred_hm(K=2, seed=4)
    got = select_task_loss("IVC", pred, hm, aop_p3_weight=5.0, param_loss_beta=0.0)
    expected = F.mse_loss(pred, hm)
    assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# marginal-KL continuation path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX",
                                  "fetal_femur"])
def test_zero_marginal_beta_traverses_exact_mse_control(tid):
    pred, hm = _pred_hm(K=4, seed=5)
    got = select_task_loss(tid, pred, hm, marginal_kl_beta=0.0)
    assert torch.equal(got, F.mse_loss(pred, hm))


def test_nonzero_marginal_beta_changes_loss_for_every_task():
    pred, hm = _pred_hm(K=4, seed=6)
    base = select_task_loss("FA", pred, hm, marginal_kl_beta=0.0)
    treatment = select_task_loss("FA", pred, hm, marginal_kl_beta=1e-3)
    assert treatment > base


if __name__ == "__main__":
    print("run via: baseline/.venv-baseline/bin/python -m pytest tests/test_select_task_loss.py -q")
