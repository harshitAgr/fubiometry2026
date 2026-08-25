"""Unit tests for the differentiable downstream parameter loss (IVC distance / HC circumference).

Requires torch -> run with the BASELINE venv (NOT the project .venv):
  baseline/.venv-baseline/bin/python -m pytest tests/test_param_loss.py -q
"""
import os
import sys

import pytest

pytest.importorskip("torch")  # torch-only test -> skipped under the project .venv; run via baseline venv

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.param_loss import (  # noqa: E402
    combined_loss,
    ellipse_perimeter_torch,
    hc_circumference,
    hc_param_loss,
    ivc_diameter,
    ivc_param_loss,
    soft_argmax_coords,
)
from scoring.geometry import ellipse_perimeter  # noqa: E402


def _gaussian_hm(H, W, cy, cx, sigma=1.8):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def _cell_to_norm(c, size):
    return c / max(size - 1, 1)


# ---------------------------------------------------------------------------
# soft_argmax_coords sanity cases
# ---------------------------------------------------------------------------

def test_soft_argmax_all_mass_one_cell_decodes_to_that_cell():
    """A heatmap with all mass at one cell must decode to that cell's normalized coordinate."""
    H, W = 16, 16
    hm = torch.zeros(1, 1, H, W)
    hm[0, 0, 10, 4] = 1.0  # (iy=10, ix=4)
    coords = soft_argmax_coords(hm)  # [1,1,2] -> (x,y)
    x, y = coords[0, 0, 0].item(), coords[0, 0, 1].item()
    assert abs(x - _cell_to_norm(4, W)) < 1e-4
    assert abs(y - _cell_to_norm(10, H)) < 1e-4


def test_soft_argmax_recovers_subcell_gaussian_peak():
    """A Gaussian peak between cells should decode close to the true sub-cell location."""
    H, W = 64, 64
    hm = torch.from_numpy(_gaussian_hm(H, W, cy=30.4, cx=40.7))[None, None]
    coords = soft_argmax_coords(hm, temperature=0.05)
    x, y = coords[0, 0, 0].item() * (W - 1), coords[0, 0, 1].item() * (H - 1)
    assert abs(x - 40.7) < 1.0
    assert abs(y - 30.4) < 1.0


def test_soft_argmax_symmetric_bimodal_decodes_to_midpoint():
    """Two equal-mass peaks -> expected value lands at their midpoint (sanity on the softmax
    expectation semantics, distinguishing it from an argmax which would pick one peak)."""
    H, W = 32, 32
    hm = torch.from_numpy(_gaussian_hm(H, W, 10, 10) + _gaussian_hm(H, W, 10, 20))[None, None]
    coords = soft_argmax_coords(hm, temperature=1.0)
    x = coords[0, 0, 0].item() * (W - 1)
    assert abs(x - 15.0) < 1.5


def test_soft_argmax_shape():
    hm = torch.rand(3, 4, 8, 8)
    coords = soft_argmax_coords(hm)
    assert coords.shape == (3, 4, 2)


def test_soft_argmax_is_differentiable():
    hm = torch.rand(2, 2, 8, 8, requires_grad=True)
    coords = soft_argmax_coords(hm)
    loss = coords.sum()
    loss.backward()
    assert hm.grad is not None
    assert torch.isfinite(hm.grad).all()


# ---------------------------------------------------------------------------
# ellipse_perimeter_torch must match scoring.geometry.ellipse_perimeter exactly
# ---------------------------------------------------------------------------

def test_ellipse_perimeter_matches_numpy():
    """float64 tensors -> must match the numpy (float64) original to floating-point precision."""
    cases = [(40.0, 25.0), (1.0, 1.0), (100.0, 1.0), (0.5, 0.5), (73.2, 12.9)]
    for a, b in cases:
        expected = ellipse_perimeter(a, b)
        got = ellipse_perimeter_torch(torch.tensor(a, dtype=torch.float64),
                                       torch.tensor(b, dtype=torch.float64)).item()
        assert abs(got - expected) < 1e-9, (a, b, got, expected)


def test_ellipse_perimeter_torch_batched_matches_numpy_elementwise():
    a = torch.tensor([40.0, 1.0, 100.0, 0.5], dtype=torch.float64)
    b = torch.tensor([25.0, 1.0, 1.0, 0.5], dtype=torch.float64)
    got = ellipse_perimeter_torch(a, b)
    for i in range(len(a)):
        expected = ellipse_perimeter(a[i].item(), b[i].item())
        assert abs(got[i].item() - expected) < 1e-9


def test_ellipse_perimeter_matches_numpy_float32():
    """Training runs in float32 (the model's default dtype) -> confirm the port still matches
    the numpy reference to float32 precision (~1e-5 relative), not just float64."""
    cases = [(40.0, 25.0), (1.0, 1.0), (100.0, 1.0), (0.5, 0.5), (73.2, 12.9)]
    for a, b in cases:
        expected = ellipse_perimeter(a, b)
        got = ellipse_perimeter_torch(torch.tensor(a, dtype=torch.float32),
                                       torch.tensor(b, dtype=torch.float32)).item()
        assert abs(got - expected) < 1e-3, (a, b, got, expected)


def test_ellipse_perimeter_zero_sum_no_nan():
    """a=b=0 hits the numpy (a+b) falsy branch (h=0) -- the torch port must not NaN/div-by-zero."""
    got = ellipse_perimeter_torch(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(got)
    assert abs(got.item() - ellipse_perimeter(0.0, 0.0)) < 1e-9


# ---------------------------------------------------------------------------
# IVC distance loss
# ---------------------------------------------------------------------------

def test_ivc_diameter_matches_euclidean():
    coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # [1,2,2] -> distance 5
    d = ivc_diameter(coords)
    assert torch.allclose(d, torch.tensor([5.0]))


def test_ivc_param_loss_zero_when_pred_equals_gt():
    H, W = 32, 32
    gt = torch.stack([
        torch.from_numpy(_gaussian_hm(H, W, 10, 10)),
        torch.from_numpy(_gaussian_hm(H, W, 20, 20)),
    ])[None]
    loss = ivc_param_loss(gt.clone(), gt.clone(), temperature=0.05)
    assert loss.item() < 1e-6


def test_ivc_param_loss_positive_when_pred_diameter_differs():
    H, W = 32, 32
    gt = torch.stack([
        torch.from_numpy(_gaussian_hm(H, W, 10, 10)),
        torch.from_numpy(_gaussian_hm(H, W, 20, 10)),
    ])[None]
    # pred: same p0, p1 shifted further away -> larger diameter
    pred = torch.stack([
        torch.from_numpy(_gaussian_hm(H, W, 10, 10)),
        torch.from_numpy(_gaussian_hm(H, W, 30, 10)),
    ])[None]
    loss = ivc_param_loss(pred, gt, temperature=0.05)
    assert loss.item() > 0.01


def test_ivc_param_loss_is_differentiable():
    H, W = 16, 16
    pred = torch.rand(2, 2, H, W, requires_grad=True)
    gt = torch.rand(2, 2, H, W)
    loss = ivc_param_loss(pred, gt)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# HC circumference loss
# ---------------------------------------------------------------------------

def test_hc_circumference_matches_derive():
    from scoring.derive import derive_from_specs
    from scoring.param_specs import PARAM_SPECS
    pts = np.array([[10.0, 0.0], [10.0, 40.0], [0.0, 20.0], [30.0, 20.0]])
    expected = derive_from_specs(PARAM_SPECS["HC"], pts)["head_circumference"]
    coords = torch.tensor(pts, dtype=torch.float64)[None]
    got = hc_circumference(coords).item()
    assert abs(got - expected) < 1e-9


def test_hc_param_loss_zero_when_pred_equals_gt():
    H, W = 32, 32
    centers = [(5, 16), (27, 16), (16, 5), (16, 27)]
    gt = torch.stack([torch.from_numpy(_gaussian_hm(H, W, cy, cx)) for cy, cx in centers])[None]
    loss = hc_param_loss(gt.clone(), gt.clone(), temperature=0.05)
    assert loss.item() < 1e-4


def test_hc_param_loss_is_differentiable():
    H, W = 16, 16
    pred = torch.rand(2, 4, H, W, requires_grad=True)
    gt = torch.rand(2, 4, H, W)
    loss = hc_param_loss(pred, gt)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# combined_loss: beta=0 do-no-harm equivalence (the load-bearing guarantee)
# ---------------------------------------------------------------------------

def test_beta_zero_is_do_no_harm_ivc():
    """beta=0 must reproduce alpha * F.mse_loss EXACTLY, for any alpha, IVC or HC."""
    torch.manual_seed(0)
    pred = torch.rand(3, 2, 16, 16)
    gt = torch.rand(3, 2, 16, 16)
    for alpha in (1.0, 0.5, 2.0):
        got = combined_loss(pred, gt, "IVC", alpha=alpha, beta=0.0)
        expected = alpha * F.mse_loss(pred, gt)
        assert torch.allclose(got, expected, atol=1e-12), (alpha, got.item(), expected.item())


def test_beta_zero_is_do_no_harm_hc():
    torch.manual_seed(1)
    pred = torch.rand(3, 4, 16, 16)
    gt = torch.rand(3, 4, 16, 16)
    for alpha in (1.0, 0.5, 2.0):
        got = combined_loss(pred, gt, "HC", alpha=alpha, beta=0.0)
        expected = alpha * F.mse_loss(pred, gt)
        assert torch.allclose(got, expected, atol=1e-12), (alpha, got.item(), expected.item())


def test_beta_zero_never_touches_param_math_even_for_unsupported_task():
    """beta=0 must short-circuit BEFORE the task_id dispatch -- so it works (and is a pure
    no-op) even for a task_id combined_loss doesn't otherwise support, proving the do-no-harm
    path never evaluates soft_argmax/ellipse code for the other 7 tasks."""
    pred = torch.rand(2, 8, 8, 8)
    gt = torch.rand(2, 8, 8, 8)
    got = combined_loss(pred, gt, "A4C", alpha=1.0, beta=0.0)
    expected = F.mse_loss(pred, gt)
    assert torch.allclose(got, expected, atol=1e-12)


def test_beta_zero_alpha_one_matches_plain_mse_loss_call_used_in_run_config():
    """Direct equivalence check against the exact call run_config.py's training loop makes
    today (F.mse_loss(pred, hm)) for the untouched-task control path."""
    torch.manual_seed(2)
    pred = torch.sigmoid(torch.rand(4, 2, 64, 64))
    hm = torch.rand(4, 2, 64, 64)
    assert torch.equal(combined_loss(pred, hm, "IVC", alpha=1.0, beta=0.0), F.mse_loss(pred, hm))


def test_combined_loss_unsupported_task_with_nonzero_beta_raises():
    pred = torch.rand(2, 8, 8, 8)
    gt = torch.rand(2, 8, 8, 8)
    with pytest.raises(ValueError):
        combined_loss(pred, gt, "A4C", alpha=1.0, beta=0.1)


def test_combined_loss_beta_nonzero_adds_param_term():
    """With beta>0, combined_loss must differ from the pure heatmap MSE when the derived
    parameter differs between pred and gt (i.e. the param term has a real, nonzero effect)."""
    H, W = 32, 32
    gt = torch.stack([
        torch.from_numpy(_gaussian_hm(H, W, 10, 10)),
        torch.from_numpy(_gaussian_hm(H, W, 20, 10)),
    ])[None]
    pred = torch.stack([
        torch.from_numpy(_gaussian_hm(H, W, 10, 10)),
        torch.from_numpy(_gaussian_hm(H, W, 30, 10)),
    ])[None]
    base = combined_loss(pred, gt, "IVC", alpha=1.0, beta=0.0)
    boosted = combined_loss(pred, gt, "IVC", alpha=1.0, beta=1.0)
    assert boosted.item() > base.item()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"OK: all {len(tests)} param_loss tests passed")
