import numpy as np
import pytest
from scoring import mre

def test_zero_error_when_identical():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert mre.mean_radial_error(pts, pts) == pytest.approx(0.0)

def test_mean_of_per_point_distances():
    pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    gt = np.array([[3.0, 4.0], [6.0, 8.0]])  # distances 5 and 10
    assert mre.mean_radial_error(pred, gt) == pytest.approx(7.5)

def test_spacing_scales_to_mm():
    pred = np.array([[0.0, 0.0]])
    gt = np.array([[10.0, 0.0]])  # 10 px
    assert mre.mean_radial_error(pred, gt, spacing=(0.5, 0.5)) == pytest.approx(5.0)

def test_mismatched_shape_raises():
    with pytest.raises(ValueError):
        mre.mean_radial_error(np.zeros((2, 2)), np.zeros((3, 2)))
