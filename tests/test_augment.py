"""Tests for experiments/augment.py (Lever 3 Phase-1 photometric augmentation).

Runs in the BASELINE venv (albumentations 2.0.8 + cv2 + torch live there):
    baseline/.venv-baseline/bin/python -m pytest tests/test_augment.py -W error -v
The project `uv run pytest` lacks albumentations and SKIPS this module via importorskip.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("albumentations")  # skip whole module if albu absent (project venv)
import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from experiments.augment import build_train_transform, MEAN, STD, PACKS  # noqa: E402

INPUT = 518
DATA = os.path.join(PROJ, "data")
_HAS_DATA = os.path.isdir(os.path.join(DATA, "csv"))


def _img():
    rng = np.random.default_rng(0)
    return (rng.random((INPUT, INPUT, 3)) * 255).astype(np.uint8)


def test_rejects_non_2x_version(monkeypatch):
    monkeypatch.setattr(A, "__version__", "1.4.0")
    with pytest.raises(RuntimeError, match="2.0"):
        build_train_transform("photo_v1")


def test_unknown_pack_raises():
    with pytest.raises(ValueError, match="unknown aug pack"):
        build_train_transform("bogus")


def test_none_equals_current_transform():
    img = _img()
    current = A.Compose([A.Resize(INPUT, INPUT), A.Normalize(MEAN, STD), ToTensorV2()])
    a = current(image=img)["image"].numpy()
    b = build_train_transform("none")(image=img)["image"].numpy()
    assert np.array_equal(a, b)


def test_photo_v1_runs_on_uint8():
    out = build_train_transform("photo_v1", seed=0)(image=_img())["image"]
    assert tuple(out.shape) == (3, INPUT, INPUT)
    assert out.dtype.is_floating_point
    assert bool(np.isfinite(out.numpy()).all())


def test_photo_v1_composition_is_exact():
    names = [type(t).__name__ for t in build_train_transform("photo_v1").transforms]
    assert names == [
        "RandomBrightnessContrast", "RandomGamma", "MultiplicativeNoise",
        "GaussNoise", "GaussianBlur", "Resize", "Normalize", "ToTensorV2",
    ]
    # guard against accidental reintroduction of removed/deferred transforms
    for banned in ("HorizontalFlip", "VerticalFlip", "CLAHE", "Downscale", "ImageCompression",
                   "Affine", "ElasticTransform"):
        assert banned not in names


def test_gaussian_blur_sigma_is_safe():
    # sigma_limit=0 divides-by-zero in albu 2.0.8; forcing the pack's blur (p=1.0) under -W error
    # would raise if the sigma were degenerate.
    blur = A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.3, 0.9), p=1.0)
    out = blur(image=_img())["image"]
    assert out.shape == (INPUT, INPUT, 3)


def test_photo_v1_is_reproducible_with_seed():
    img = _img()
    a = build_train_transform("photo_v1", seed=123)(image=img)["image"].numpy()
    b = build_train_transform("photo_v1", seed=123)(image=img)["image"].numpy()
    assert np.array_equal(a, b)


@pytest.mark.skipif(not _HAS_DATA, reason="needs local data/ (csv + images)")
def test_heatmap_target_invariant_under_image_aug():
    """Photometric aug changes pixels but NOT the heatmap target — the Phase-1 safety property."""
    import torch  # noqa: E402  (baseline venv)
    from dataset import KeypointDataset  # noqa: E402

    forced = A.Compose([A.RandomBrightnessContrast(0.5, 0.5, p=1.0),
                        A.Resize(INPUT, INPUT), A.Normalize(MEAN, STD), ToTensorV2()])
    plain = build_train_transform("none")
    ds_aug = KeypointDataset(data_root=DATA, transforms=forced, heatmap_size=(64, 64), sigma=1.8)
    ds_none = KeypointDataset(data_root=DATA, transforms=plain, heatmap_size=(64, 64), sigma=1.8)
    a, n = ds_aug[0], ds_none[0]
    assert not torch.equal(a["image"], n["image"])        # aug altered the pixels
    assert torch.equal(a["heatmap"], n["heatmap"])         # ...but the target is identical
    assert torch.equal(a["label"], n["label"])


@pytest.mark.skipif(not _HAS_DATA, reason="needs local data/ (csv + images)")
def test_photo_v1_flows_through_dataset():
    from dataset import KeypointDataset  # noqa: E402
    ds = KeypointDataset(data_root=DATA, transforms=build_train_transform("photo_v1", seed=0),
                         heatmap_size=(64, 64), sigma=1.8)
    item = ds[0]
    assert tuple(item["image"].shape) == (3, INPUT, INPUT)
    assert item["heatmap"].shape[1:] == (64, 64)


def test_geo_v1_composition_and_guards():
    import cv2
    from experiments.augment import build_geo_fallback  # noqa: F401 (import smoke)
    tfm = build_train_transform("geo_v1")
    names = [type(t).__name__ for t in tfm.transforms]
    assert names == ["RandomBrightnessContrast", "RandomGamma", "MultiplicativeNoise",
                     "GaussNoise", "GaussianBlur", "Affine", "Resize", "Normalize", "ToTensorV2"]
    for banned in ("ElasticTransform", "HorizontalFlip", "VerticalFlip", "CLAHE"):
        assert banned not in names
    kp_proc = tfm.processors.get("keypoints")
    assert kp_proc is not None and kp_proc.params.remove_invisible is False
    aff = next(t for t in tfm.transforms if type(t).__name__ == "Affine")
    assert aff.border_mode == cv2.BORDER_CONSTANT          # 0 — NOT reflect
    assert aff.interpolation == cv2.INTER_LINEAR
    assert aff.fill == 0
    assert aff.keep_ratio is True
    # shear stored as {'x': (0.0, 0.0), 'y': (0.0, 0.0)}
    sx = aff.shear["x"] if isinstance(aff.shear, dict) else aff.shear
    assert tuple(sx) == (0.0, 0.0) or sx == 0.0


def test_geo_v1_preserves_keypoint_count():
    tfm = build_train_transform("geo_v1", seed=0)
    img = _img()
    kps = [(10.0, 10.0), (200.0, 300.0), (500.0, 500.0), (260.0, 260.0)]
    for _ in range(30):  # many draws — reflect/remove_invisible bugs would change the count
        out = tfm(image=img, keypoints=kps)
        assert len(out["keypoints"]) == len(kps)


def test_geo_fallback_equals_photo_v1_pixels():
    from experiments.augment import build_geo_fallback
    img = _img()
    a = build_geo_fallback(seed=7)(image=img, keypoints=[(5.0, 5.0)])["image"].numpy()
    b = build_train_transform("photo_v1", seed=7)(image=img)["image"].numpy()
    assert np.array_equal(a, b)   # keypoint_params doesn't alter pixel RNG -> geo p=0 == photo_v1


def test_geo_v1_no_cval_kwarg():
    # 1.x idiom `cval=` is silently ignored in 2.0.8; -W error (the run flag) turns its UserWarning
    # into an error. This test just asserts the pack builds+runs clean under the suite's -W error.
    out = build_train_transform("geo_v1", seed=1)(image=_img(), keypoints=[(1.0, 1.0)])
    assert tuple(out["image"].shape) == (3, INPUT, INPUT)


# --- HC small-head zoom-out override (geo_v1_hcsmall) -------------------------------------------

def test_geo_v1_hcsmall_registered():
    from experiments.augment import PACKS, GEO_PACKS
    assert "geo_v1_hcsmall" in PACKS
    assert "geo_v1_hcsmall" in GEO_PACKS


def test_hc_smallhead_composition_and_guards():
    import cv2
    from experiments.augment import build_hc_smallhead_transform
    tfm = build_hc_smallhead_transform(INPUT)
    names = [type(t).__name__ for t in tfm.transforms]
    # photo_v1 stack + one conformal zoom-out Affine + Resize/Normalize/ToTensor (mirrors geo_v1)
    assert names == ["RandomBrightnessContrast", "RandomGamma", "MultiplicativeNoise",
                     "GaussNoise", "GaussianBlur", "Affine", "Resize", "Normalize", "ToTensorV2"]
    for banned in ("ElasticTransform", "HorizontalFlip", "VerticalFlip", "CLAHE"):
        assert banned not in names
    kp_proc = tfm.processors.get("keypoints")
    assert kp_proc is not None and kp_proc.params.remove_invisible is False
    aff = next(t for t in tfm.transforms if type(t).__name__ == "Affine")
    assert aff.border_mode == cv2.BORDER_CONSTANT          # 0 — NOT reflect (breaks num_points)
    assert aff.fill == 0
    assert aff.keep_ratio is True                          # conformal -> HC ellipse param-safe
    sx = aff.shear["x"] if isinstance(aff.shear, dict) else aff.shear
    assert tuple(sx) == (0.0, 0.0) or sx == 0.0            # no shear -> conformal
    # the defining property of the lever: the scale range reaches well BELOW 1.0 (zoom-out /
    # small-head synthesis), unlike geo_v1's mild (0.88, 1.12).
    lo = aff.scale["x"][0] if isinstance(aff.scale, dict) else aff.scale[0]
    assert lo <= 0.6


def test_hc_smallhead_preserves_keypoint_count():
    from experiments.augment import build_hc_smallhead_transform
    tfm = build_hc_smallhead_transform(INPUT, seed=0)
    img = _img()
    kps = [(10.0, 10.0), (200.0, 300.0), (500.0, 500.0), (260.0, 260.0)]  # 4 HC ellipse pts
    for _ in range(30):
        out = tfm(image=img, keypoints=kps)
        assert len(out["keypoints"]) == len(kps)
