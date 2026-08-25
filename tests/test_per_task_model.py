"""Unit tests for the per-task heatmap-size resolver.

hm_for itself is pure, but per_task_model imports the baseline MultiTaskModelFactory (torch) at
module load, so this test runs in the baseline venv and is skipped in the torch-free project .venv.
"""
import pytest

pytest.importorskip("torch")  # per_task_model -> model_factory needs torch
from experiments.per_task_model import hm_for, DEFAULT_HM, sigma_for, DEFAULT_SIGMA  # noqa: E402


def test_hm_for_uniform_tuple_returns_same_for_any_task():
    hm = (64, 64)
    assert hm_for(hm, "FUGC") == (64, 64)
    assert hm_for(hm, "AOP") == (64, 64)
    assert hm_for(hm, "anything") == (64, 64)


def test_hm_for_dict_fugc_and_femur_finer_others_default():
    # both precise 2-pt tasks at 128, everything else at the (64,64) default
    hm = {"FUGC": (128, 128), "fetal_femur": (128, 128)}
    assert hm_for(hm, "FUGC") == (128, 128)
    assert hm_for(hm, "fetal_femur") == (128, 128)
    assert DEFAULT_HM == (64, 64)
    for other in ("AOP", "PLAX", "HC", "A4C", "FA", "IVC", "PSAX"):
        assert hm_for(hm, other) == (64, 64)


def test_hm_for_dict_fugc_only_leaves_femur_at_default():
    # the RUNNING ensemble's recipe: FUGC@128 only -> femur must stay 64 (consistency guard)
    hm = {"FUGC": (128, 128)}
    assert hm_for(hm, "FUGC") == (128, 128)
    assert hm_for(hm, "fetal_femur") == (64, 64)


def test_sigma_for_scalar_returns_same_for_any_task():
    # scalar sigma -> uniform (byte-identical to the pre-per-task-sigma scalar path)
    assert sigma_for(1.8, "FUGC") == 1.8
    assert sigma_for(1.8, "AOP") == 1.8
    assert sigma_for(2.2, "anything") == 2.2


def test_sigma_for_dict_scales_precise_tasks_others_default():
    # FUGC/femur at 128 grid -> sigma scaled 1.8*128/64=3.6 to hold physical width; others -> 1.8
    sig = {"FUGC": 3.6, "fetal_femur": 3.6}
    assert sigma_for(sig, "FUGC") == 3.6
    assert sigma_for(sig, "fetal_femur") == 3.6
    assert DEFAULT_SIGMA == 1.8
    for other in ("AOP", "PLAX", "HC", "A4C", "FA", "IVC", "PSAX"):
        assert sigma_for(sig, other) == 1.8


def test_sigma_for_dict_returns_float():
    # dict path always yields a float (guards against int/np-scalar leaking into the exp())
    assert isinstance(sigma_for({"FUGC": 3.6}, "FUGC"), float)
    assert isinstance(sigma_for({"FUGC": 3.6}, "AOP"), float)
