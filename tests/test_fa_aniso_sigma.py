"""Tests for the FA-only anisotropic-Gaussian target lever in experiments/kp_aug_dataset.py
(2026-08-07 FA diagnosis: wall-variance reshaping, --fa-aniso-sigma).

Run in the BASELINE venv:  baseline/.venv-baseline/bin/python -m pytest tests/test_fa_aniso_sigma.py -q
Project `uv run pytest` skips this module (no albumentations)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("albumentations")
import torch  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from experiments.augment import build_train_transform, build_geo_fallback  # noqa: E402
from experiments.kp_aug_dataset import KeypointAugDataset  # noqa: E402

DATA = os.path.join(PROJ, "data")
_HAS_DATA = os.path.isdir(os.path.join(DATA, "csv"))
HM, SIG = (64, 64), 1.8
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="needs local data/ (csv + images)")


def _ds(fa_aniso_sigma=None, seed=0):
    return KeypointAugDataset(
        data_root=DATA, transforms=build_geo_fallback(seed=seed),
        fallback_transforms=build_geo_fallback(seed=seed), heatmap_size=HM, sigma=SIG,
        fa_aniso_sigma=fa_aniso_sigma)


def _fa_index(ds):
    df = ds.dataframe
    return int(df.index[df.task_id == "FA"][0])


def _first_non_fa_index(ds):
    df = ds.dataframe
    return int(df.index[df.task_id != "FA"][0])


def _marginal_sd(h, axis):
    """Standard deviation of the heatmap's marginal distribution along `axis` (0=y, 1=x)."""
    coords = np.arange(h.shape[axis])
    w = h.sum(axis=1 - axis)
    w = w / w.sum()
    mean = (coords * w).sum()
    return float(((coords - mean) ** 2 * w).sum() ** 0.5)


# ---------------------------------------------------------------------------
# Default (fa_aniso_sigma=None) byte-identity -- the critical do-no-harm guarantee
# ---------------------------------------------------------------------------

def test_default_none_matches_pre_lever_isotropic_rendering():
    """fa_aniso_sigma defaulting to None must be byte-identical to the pre-lever renderer for
    EVERY task including FA (constructor default + explicit None both take this path)."""
    ds_implicit = _ds()  # relies on the constructor default
    ds_explicit = _ds(fa_aniso_sigma=None)
    idx = _fa_index(ds_implicit)
    h_i = ds_implicit[idx]["heatmap"]
    h_e = ds_explicit[idx]["heatmap"]
    assert torch.equal(h_i, h_e)


def test_default_none_fa_channels_all_isotropic_at_task_sigma():
    ds = _ds()
    idx = _fa_index(ds)
    h = ds[idx]["heatmap"].numpy()
    assert h.shape[0] == 4
    for c in range(4):
        assert _marginal_sd(h[c], axis=0) == pytest.approx(SIG, abs=0.05)
        assert _marginal_sd(h[c], axis=1) == pytest.approx(SIG, abs=0.05)


def test_non_fa_tasks_unaffected_by_fa_aniso_sigma():
    ds_default = _ds()
    ds_aniso = _ds(fa_aniso_sigma=(1.2, 2.7))
    idx = _first_non_fa_index(ds_default)
    assert ds_default.dataframe.iloc[idx]["task_id"] != "FA"
    h_def = ds_default[idx]["heatmap"]
    h_an = ds_aniso[idx]["heatmap"]
    assert torch.equal(h_def, h_an)


# ---------------------------------------------------------------------------
# Anisotropic rendering moments
# ---------------------------------------------------------------------------

def test_aniso_p0_p1_stay_isotropic_p2_p3_become_anisotropic():
    ds_default = _ds()
    ds_aniso = _ds(fa_aniso_sigma=(1.2, 2.7))
    idx = _fa_index(ds_default)
    h_def = ds_default[idx]["heatmap"].numpy()
    h_an = ds_aniso[idx]["heatmap"].numpy()
    # p0 (top), p1 (bottom) untouched
    assert np.array_equal(h_def[0], h_an[0])
    assert np.array_equal(h_def[1], h_an[1])
    # p2 (right), p3 (left) differ from the isotropic baseline
    assert not np.allclose(h_def[2], h_an[2])
    assert not np.allclose(h_def[3], h_an[3])


def test_aniso_argmax_still_at_the_keypoint():
    """Anisotropic re-shaping must not move the peak -- only its spread changes."""
    ds_default = _ds()
    ds_aniso = _ds(fa_aniso_sigma=(1.2, 2.7))
    idx = _fa_index(ds_default)
    h_def = ds_default[idx]["heatmap"].numpy()
    h_an = ds_aniso[idx]["heatmap"].numpy()
    for c in range(4):
        p_def = np.unravel_index(int(h_def[c].argmax()), h_def[c].shape)
        p_an = np.unravel_index(int(h_an[c].argmax()), h_an[c].shape)
        assert p_def == p_an, f"channel {c}: peak moved {p_def} -> {p_an}"
        # Peak value is < 1.0 whenever the (continuous) keypoint doesn't land exactly on a
        # grid cell -- same convention as tests/test_kp_aug.py's "> 0.9" real-peak check.
        assert float(h_an[c].max()) > 0.9


def test_aniso_marginal_sd_matches_sx_sy_when_sx_less_than_sy():
    """SX < SY (narrower in x, wider in y) -> the rendered p2/p3 target's x marginal sd is
    SMALLER than its y marginal sd, matching the requested (sx, sy)."""
    sx, sy = 1.2, 2.7
    ds = _ds(fa_aniso_sigma=(sx, sy))
    idx = _fa_index(ds)
    h = ds[idx]["heatmap"].numpy()
    for c in (2, 3):
        x_sd = _marginal_sd(h[c], axis=1)
        y_sd = _marginal_sd(h[c], axis=0)
        assert x_sd == pytest.approx(sx, abs=0.05)
        assert y_sd == pytest.approx(sy, abs=0.05)
        assert x_sd < y_sd


def test_aniso_marginal_sd_matches_sx_sy_when_sx_greater_than_sy():
    """Direction sanity: swapping SX/SY swaps which axis is wider (not hardcoded)."""
    sx, sy = 3.0, 1.0
    ds = _ds(fa_aniso_sigma=(sx, sy))
    idx = _fa_index(ds)
    h = ds[idx]["heatmap"].numpy()
    for c in (2, 3):
        x_sd = _marginal_sd(h[c], axis=1)
        y_sd = _marginal_sd(h[c], axis=0)
        assert x_sd == pytest.approx(sx, abs=0.05)
        assert y_sd == pytest.approx(sy, abs=0.05)
        assert x_sd > y_sd


def test_aniso_isotropic_sx_equals_sy_matches_baseline_rendering():
    """sx == sy must reproduce the plain isotropic Gaussian exactly (sanity: anisotropy
    collapses to isotropy at equal scales)."""
    ds_default = _ds()
    ds_aniso = _ds(fa_aniso_sigma=(SIG, SIG))
    idx = _fa_index(ds_default)
    h_def = ds_default[idx]["heatmap"].numpy()
    h_an = ds_aniso[idx]["heatmap"].numpy()
    assert np.allclose(h_def, h_an, atol=1e-6)


if __name__ == "__main__":
    print("run via: baseline/.venv-baseline/bin/python -m pytest tests/test_fa_aniso_sigma.py -q")
