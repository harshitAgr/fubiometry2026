import numpy as np
import pytest
from scoring import derive, param_specs


def test_distance_param():
    spec = [param_specs.ParamSpec(name="d", kind="distance", indices=(0, 1))]
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = derive.derive_from_specs(spec, pts)
    assert out["d"] == pytest.approx(5.0)


def test_angle_param_aop_like():
    # indices = (a, vertex, b)
    spec = [param_specs.ParamSpec(name="aop", kind="angle", indices=(0, 1, 2))]
    pts = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    out = derive.derive_from_specs(spec, pts)
    assert out["aop"] == pytest.approx(90.0)


def test_ellipse_param_from_axis_endpoints():
    # indices = (major_p1, major_p2, minor_p1, minor_p2); a,b are half the axis lengths
    spec = [param_specs.ParamSpec(name="hc", kind="ellipse_perimeter", indices=(0, 1, 2, 3))]
    pts = np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, -5.0], [0.0, 5.0]])
    out = derive.derive_from_specs(spec, pts)
    # a=10, b=5
    from scoring.geometry import ellipse_perimeter
    assert out["hc"] == pytest.approx(ellipse_perimeter(10.0, 5.0))


def test_all_nine_task_ids_have_specs():
    expected = {"A4C", "AOP", "FA", "fetal_femur", "FUGC", "HC", "IVC", "PLAX", "PSAX"}
    assert expected.issubset(set(param_specs.PARAM_SPECS.keys()))
    for tid in expected:
        assert isinstance(param_specs.PARAM_SPECS[tid], list)


def test_param_specs_populated_with_valid_indices():
    # Landmark counts per task (verified from the prepared CSV num_classes, 2026-06-15).
    n_pts = {"A4C": 16, "AOP": 4, "FA": 4, "fetal_femur": 2, "FUGC": 2,
             "HC": 4, "IVC": 2, "PLAX": 22, "PSAX": 4}
    # Expected parameter counts (cardiac = num_pts/2 distances; obstetric per design).
    n_params = {"A4C": 8, "PLAX": 11, "PSAX": 2, "IVC": 1, "FUGC": 1,
                "fetal_femur": 1, "HC": 1, "FA": 1, "AOP": 2}
    for tid, n in n_pts.items():
        specs = param_specs.PARAM_SPECS[tid]
        assert len(specs) == n_params[tid], f"{tid}: expected {n_params[tid]} params, got {len(specs)}"
        for s in specs:
            assert s.indices, f"{tid}/{s.name}: empty indices"
            assert max(s.indices) < n, f"{tid}/{s.name}: index {max(s.indices)} >= {n} landmarks"


def test_derive_parameters_on_a4c_consecutive_pairs():
    # 16 landmarks; param k = distance(point_{2k}, point_{2k+1}) in 0-indexed pairs.
    pts = np.zeros((16, 2), dtype=float)
    pts[1] = [3.0, 4.0]  # LV_ud endpoint -> distance 5 from origin
    out = derive.derive_parameters("A4C", pts)
    assert len(out) == 8
    assert out["LV_ud"] == pytest.approx(5.0)
