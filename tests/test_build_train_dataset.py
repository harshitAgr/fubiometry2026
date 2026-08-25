"""Tests for experiments.run_config.build_train_dataset -- the aug-pack -> Dataset-class
routing extracted from train_fold, including the FA anisotropic-target lever's dispatch
(--aug photo_v1 + --fa-aniso-sigma routes through KeypointAugDataset; everything else is
byte-identical to the pre-lever routing).

Run in the BASELINE venv:
  baseline/.venv-baseline/bin/python -m pytest tests/test_build_train_dataset.py -q
"""
import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("albumentations")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from dataset import KeypointDataset  # noqa: E402
from experiments.augment import build_train_transform  # noqa: E402
from experiments.kp_aug_dataset import KeypointAugDataset  # noqa: E402
from experiments.run_config import build_train_dataset  # noqa: E402

DATA = os.path.join(PROJ, "data")
_HAS_DATA = os.path.isdir(os.path.join(DATA, "csv"))
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="needs local data/ (csv + images)")
HM, SIG = (64, 64), 1.8


def _fa_index(ds):
    df = ds.dataframe
    return int(df.index[df.task_id == "FA"][0])


# ---------------------------------------------------------------------------
# Default (fa_aniso_sigma=None) routing is unchanged: photo_v1 -> KeypointDataset,
# a geo pack -> KeypointAugDataset. No behavior change for any existing caller.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aug", ["none", "photo_v1"])
def test_non_geo_packs_route_to_plain_keypoint_dataset_by_default(aug):
    ds = build_train_dataset(aug, 518, seed=7, heatmap_size=HM, sigma=SIG, data_root=DATA)
    assert type(ds) is KeypointDataset


@pytest.mark.parametrize("aug", ["geo_v1", "aop_robust_v1", "fugc_scale_v1", "geo_v1_hcsmall"])
def test_geo_packs_route_to_keypoint_aug_dataset(aug):
    ds = build_train_dataset(aug, 518, seed=7, heatmap_size=HM, sigma=SIG, data_root=DATA)
    assert isinstance(ds, KeypointAugDataset)


def test_photo_v1_default_byte_identical_to_direct_keypoint_dataset():
    got = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG, data_root=DATA)
    expected = KeypointDataset(data_root=DATA, transforms=build_train_transform("photo_v1", seed=3),
                               heatmap_size=HM, sigma=SIG)
    for i in (0, 12, 123):
        assert torch.allclose(got[i]["heatmap"], expected[i]["heatmap"], atol=1e-5)
        assert torch.allclose(got[i]["label"], expected[i]["label"], atol=1e-5)


# ---------------------------------------------------------------------------
# fa_aniso_sigma routing: photo_v1 reroutes through KeypointAugDataset; a geo pack keeps
# using KeypointAugDataset (fa_aniso_sigma flows straight into its existing constructor arg).
# ---------------------------------------------------------------------------

def test_photo_v1_with_fa_aniso_sigma_routes_to_keypoint_aug_dataset():
    ds = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                             fa_aniso_sigma=(1.2, 2.7), data_root=DATA)
    assert isinstance(ds, KeypointAugDataset)
    assert ds.fa_aniso_sigma == (1.2, 2.7)


def test_photo_v1_with_fa_aniso_sigma_non_fa_rows_stay_byte_identical():
    """Only the FA heatmap RENDERING changes; every other task's image/label/heatmap is
    byte-identical to the default (fa_aniso_sigma=None) photo_v1 routing."""
    default_ds = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                                     data_root=DATA)
    aniso_ds = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                                   fa_aniso_sigma=(1.2, 2.7), data_root=DATA)
    df = default_ds.dataframe
    non_fa_idx = [i for i in (0, 12, 123, 200) if df.iloc[i]["task_id"] != "FA"]
    assert non_fa_idx  # sanity: the sampled indices really are non-FA
    for i in non_fa_idx:
        a, b = default_ds[i], aniso_ds[i]
        assert torch.allclose(a["heatmap"], b["heatmap"], atol=1e-6)
        assert torch.allclose(a["label"], b["label"], atol=1e-6)
        assert torch.allclose(a["image"], b["image"], atol=1e-6)


def test_photo_v1_with_fa_aniso_sigma_fa_row_differs_only_on_wall_channels():
    default_ds = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                                     data_root=DATA)
    aniso_ds = build_train_dataset("photo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                                   fa_aniso_sigma=(1.2, 2.7), data_root=DATA)
    idx = _fa_index(default_ds)
    a = default_ds[idx]["heatmap"].numpy()
    b = aniso_ds[idx]["heatmap"].numpy()
    # KeypointDataset vs KeypointAugDataset compute the normalized label via a differently
    # ordered (but mathematically equivalent) float32 expression, so isotropic channels match
    # only up to float32 rounding (~1e-6), not bit-exactly; see test_photo_v1_default_byte_...
    # for the same tolerance on the plain (non-aniso) comparison.
    assert np.allclose(a[0], b[0], atol=1e-5)
    assert np.allclose(a[1], b[1], atol=1e-5)
    assert not np.allclose(a[2], b[2])
    assert not np.allclose(a[3], b[3])


def test_geo_pack_with_fa_aniso_sigma_flows_through_unchanged_constructor_path():
    ds = build_train_dataset("geo_v1", 518, seed=3, heatmap_size=HM, sigma=SIG,
                             fa_aniso_sigma=(1.2, 2.7), data_root=DATA)
    assert ds.fa_aniso_sigma == (1.2, 2.7)


def test_none_aug_with_fa_aniso_sigma_raises():
    """build_train_dataset trusts validate_fa_config's guarantee (assert aug == 'photo_v1'
    for the reroute branch); passing an unsupported aug should never silently mis-render."""
    with pytest.raises(AssertionError):
        build_train_dataset("none", 518, seed=3, heatmap_size=HM, sigma=SIG,
                            fa_aniso_sigma=(1.2, 2.7), data_root=DATA)


if __name__ == "__main__":
    print("run via: baseline/.venv-baseline/bin/python -m pytest tests/test_build_train_dataset.py -q")
