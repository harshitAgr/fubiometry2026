import numpy as np
import pytest
from scoring import geometry as g


def test_euclidean_345():
    assert g.euclidean((0, 0), (3, 4)) == pytest.approx(5.0)


def test_angle_right_angle():
    # vertex at origin, rays to (1,0) and (0,1) -> 90 degrees
    assert g.angle_deg((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_angle_straight():
    assert g.angle_deg((1, 0), (0, 0), (-1, 0)) == pytest.approx(180.0)


def test_ellipse_perimeter_circle():
    # a == b == r -> perimeter == 2*pi*r (Ramanujan exact for a circle)
    assert g.ellipse_perimeter(5.0, 5.0) == pytest.approx(2 * np.pi * 5.0, rel=1e-6)
