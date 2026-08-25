import pytest

from experiments.evaluate_dino_vitb_5fold import deployment_decision


def test_decisive_consistent_win_deploys():
    report = deployment_decision([-1.2, -0.9, -1.0, -0.8, -1.1], treatment_mean=22.0)
    assert report["deploy"] is True
    assert all(report["checks"].values())


def test_ci_crossing_zero_blocks_deployment():
    report = deployment_decision([-1.0, -0.8, -0.5, 0.9, 0.8], treatment_mean=22.9)
    assert report["deploy"] is False
    assert report["checks"]["corrected_ci_upper_below_zero"] is False


def test_expected_score_policy_accepts_consistent_lower_mean_despite_wide_ci():
    report = deployment_decision(
        [-0.52, -1.24, -0.53, 0.79, -0.02], treatment_mean=22.694,
        baseline=23.0, policy="expected_score",
    )
    assert report["deploy"] is True
    assert report["strict_deploy"] is False
    assert report["policy"] == "expected_score"


def test_only_three_favorable_folds_blocks_deployment():
    report = deployment_decision([-2.0, -2.0, -2.0, 0.01, 0.01], treatment_mean=22.0)
    assert report["deploy"] is False
    assert report["checks"]["at_least_4_of_5_folds_improved"] is False


def test_mean_above_baseline_blocks_deployment():
    report = deployment_decision([-1.0] * 5, treatment_mean=23.01)
    assert report["deploy"] is False
    assert report["checks"]["treatment_below_23"] is False


def test_malformed_delta_count_fails_closed():
    with pytest.raises(ValueError):
        deployment_decision([-1.0] * 4, treatment_mean=22.0)


def test_unknown_policy_fails_closed():
    with pytest.raises(ValueError):
        deployment_decision([-1.0] * 5, treatment_mean=22.0, policy="optimistic")
