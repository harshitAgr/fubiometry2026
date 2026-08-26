"""Tests for experiments/kp_aug_dataset.py (Lever 3 Phase-2 geometric, keypoint-aware).
Run in the BASELINE venv:  baseline/.venv-baseline/bin/python -m pytest tests/test_kp_aug.py -W error -v
Project `uv run pytest` skips this module (no albumentations)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("albumentations")
import albumentations as A  # noqa: E402
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


def _ds(transforms, fallback):
    return KeypointAugDataset(data_root=DATA, transforms=transforms,
                              fallback_transforms=fallback, heatmap_size=HM, sigma=SIG)


def test_item_shapes_and_count_preserved():
    ds = _ds(build_train_transform("geo_v1", seed=0), build_geo_fallback(seed=0))
    for i in (0, 5, 50, 200):
        item = ds[i]
        n = int(ds.dataframe.iloc[i]["num_classes"])
        assert tuple(item["image"].shape) == (3, 518, 518)
        assert tuple(item["heatmap"].shape) == (n, 64, 64)
        assert item["label"].shape == (2 * n,)
        lab = item["label"].numpy()
        assert (lab >= 0).all() and (lab <= 1).all()
        assert np.isfinite(item["heatmap"].numpy()).all()
        assert float(item["heatmap"].max()) > 0.9   # a real Gaussian peak in-frame


def test_extreme_affine_falls_back_to_valid_target():
    # An always-out-of-frame Affine MUST force the fallback every time -> verify fallback fires.
    # translate_percent=1.5 shifts every keypoint ~775px off a 518px canvas — guaranteed OOF.
    import cv2
    from albumentations.pytorch import ToTensorV2
    extreme = A.Compose(
        [A.Affine(translate_percent={"x": (1.5, 1.5), "y": (1.5, 1.5)},
                  border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
         A.Resize(518, 518), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, check_each_transform=True),
        seed=0)
    ds_extreme = _ds(extreme, build_geo_fallback(seed=0))
    ds_fallback_only = _ds(build_geo_fallback(seed=0), build_geo_fallback(seed=0))
    idx = 0
    item_extreme = ds_extreme[idx]
    item_fallback = ds_fallback_only[idx]
    n = int(ds_extreme.dataframe.iloc[idx]["num_classes"])
    # Shape + validity
    assert tuple(item_extreme["heatmap"].shape) == (n, 64, 64)
    lab = item_extreme["label"].numpy()
    assert (lab >= 0).all() and (lab <= 1).all()
    assert float(item_extreme["heatmap"].max()) > 0.9   # valid in-frame peak => fallback fired
    # Fallback identity: extreme-transform dataset must produce exactly the fallback-only heatmap
    assert torch.allclose(item_extreme["heatmap"], item_fallback["heatmap"], atol=1e-5), (
        "Fallback did not fire or produced a different result than fallback_only dataset"
    )


def test_geo_p0_matches_baseline_heatmap():
    # geo with no geometry beyond Resize (the fallback) must reproduce the baseline image-only
    # KeypointDataset heatmap target (normalize-by-518 == baseline normalize-by-original).
    from dataset import KeypointDataset
    base = KeypointDataset(data_root=DATA, transforms=build_train_transform("photo_v1", seed=3),
                           heatmap_size=HM, sigma=SIG)
    geo0 = _ds(build_geo_fallback(seed=3), build_geo_fallback(seed=3))
    for i in (0, 12, 123):
        assert torch.allclose(base[i]["heatmap"], geo0[i]["heatmap"], atol=1e-5)


def test_per_task_sigma_scales_with_grid_holding_physical_width():
    """Per-task sigma lever: FUGC/femur on a 128 grid with sigma scaled to 3.6 must keep the
    PHYSICAL Gaussian width ~constant vs the 64/1.8 baseline (the fix for the HM=128 confound),
    while an UNSCALED 128/1.8 target is ~2x sharper. Other tasks stay at 64/1.8 (byte-identical)."""
    hm = {"FUGC": (128, 128), "fetal_femur": (128, 128)}
    sig = {"FUGC": 3.6, "fetal_femur": 3.6}
    ds = KeypointAugDataset(data_root=DATA, transforms=build_geo_fallback(518, seed=0),
                            fallback_transforms=build_geo_fallback(518, seed=0),
                            heatmap_size=hm, sigma=sig, input_size=518)
    ds_conf = KeypointAugDataset(data_root=DATA, transforms=build_geo_fallback(518, seed=0),
                                 fallback_transforms=build_geo_fallback(518, seed=0),
                                 heatmap_size=hm, sigma=1.8, input_size=518)  # unscaled = confound
    df = ds.dataframe

    def eff_sigma_cells(h):  # peak-normalized Gaussian mass = 2*pi*sigma^2
        return float(((h.sum() / h.max()) / (2.0 * np.pi)) ** 0.5)

    # FUGC/femur: 128 grid, effective sigma ~3.6 cells -> physical std ~ 3.6/127 ~ 64-grid 1.8/63
    for task in ("FUGC", "fetal_femur"):
        i = int(df.index[df.task_id == task][0])
        h = ds[i]["heatmap"].numpy()
        assert h.shape[1:] == (128, 128)
        assert eff_sigma_cells(h[0]) == pytest.approx(3.6, abs=0.15)
        assert (3.6 / 127) == pytest.approx(1.8 / 63, rel=0.02)  # physical width held constant
    # Non-overridden tasks stay at 64/1.8 (byte-identical scalar path)
    for task in ("AOP", "A4C"):
        i = int(df.index[df.task_id == task][0])
        h = ds[i]["heatmap"].numpy()
        assert h.shape[1:] == (64, 64)
        assert eff_sigma_cells(h[0]) == pytest.approx(1.8, abs=0.15)
    # Confound guard: same 128 grid with UNSCALED sigma 1.8 is ~2x sharper (half the cell-width)
    j = int(df.index[df.task_id == "FUGC"][0])
    assert eff_sigma_cells(ds_conf[j]["heatmap"].numpy()[0]) == pytest.approx(1.8, abs=0.15)


def test_heatmap_peak_invariant_to_input_size():
    """Regression (Lever-2 USFM probe failure): KeypointAugDataset must normalize keypoints
    by the ACTUAL input_size, not a hardcoded 518. A resize to 224 vs 518 preserves the
    RELATIVE keypoint position (k/sz == original_kp/original_size, the sz cancels), so the
    64x64 heatmap peak cell must be identical across input sizes. With the old `/ 518` bug,
    the 224 target was mis-scaled ~0.43x -> the model learned garbage (~280px MRE vs ~30)."""
    def mk(sz):
        return KeypointAugDataset(data_root=DATA, transforms=build_geo_fallback(sz, seed=0),
                                  fallback_transforms=build_geo_fallback(sz, seed=0),
                                  heatmap_size=HM, sigma=SIG, input_size=sz)
    ds224, ds518 = mk(224), mk(518)
    for i in (0, 12, 123):
        h224, h518 = ds224[i]["heatmap"].numpy(), ds518[i]["heatmap"].numpy()
        for c in range(h224.shape[0]):
            p224 = np.unravel_index(int(h224[c].argmax()), h224[c].shape)
            p518 = np.unravel_index(int(h518[c].argmax()), h518[c].shape)
            assert p224 == p518, f"sample {i} ch {c}: peak moved {p224} vs {p518} — input_size not honored"
