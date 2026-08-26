"""Training-time augmentation packs for the per-fold trainer (Lever 3, Phase 1: photometric).

PURE factory: build_train_transform(pack) -> albumentations.Compose. Photometric-only packs are
coord-safe — they alter pixels only, so the Gaussian heatmap target (built from CSV coords the
transform never sees in baseline/baseline/dataset.py) is unchanged. Geometric augs are deferred to
Phase 2 (need keypoint co-transformation). Written for albumentations 2.0.8 (the baseline venv);
1.x idioms (e.g. GaussNoise var_limit, GaussianBlur sigma_limit=0) silently misbehave or divide by
zero, so the major version is asserted. Imported by experiments/run_config.py (baseline venv).
"""
from typing import Optional

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PACKS = ("none", "photo_v1", "geo_v1", "aop_robust_v1", "fugc_scale_v1", "geo_v1_hcsmall")
GEO_PACKS = frozenset({"geo_v1", "aop_robust_v1", "fugc_scale_v1", "geo_v1_hcsmall"})


def _photo_v1():
    """Photometric domain-randomization for cross-scanner OOD robustness (ultrasound-appropriate).

    Speckle (MultiplicativeNoise) is core — US is multiplicative-speckle dominated. Both noises are
    per_channel=False (input is grayscale-in-RGB; per-channel noise would fabricate chroma). Blur is
    mild + low-p to protect the sub-pixel heatmap peak. No flip (ordered landmarks); no CLAHE
    (distorts localization cues); no Downscale (fights inference scale-TTA).
    """
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), per_channel=False, p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.3, 0.9), p=0.15),
    ]


def _geo_v1_affine():
    """Conformal Affine (keep_ratio + no shear -> preserves angles/ratios = param-MAE safe).
    BORDER_CONSTANT is mandatory: reflect multiplies keypoints and breaks the fixed num_points."""
    return A.Affine(
        scale=(0.88, 1.12),
        rotate=(-12, 12),
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        shear=0.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        keep_ratio=True,
        fit_output=False,
        rotate_method="largest_box",
        p=0.5,
    )


def _fugc_scale_affine():
    """Wide ZOOM-OUT Affine for FUGC scale-robustness (fixes the val FOV bug).

    The 20 FUGC val imgs are wider-FOV: the anatomy fills ~half the frame vs train where it fills
    it. Square-squashing to 518 then makes the val anatomy ~2x too small -> the model can't find it
    and emits a confident constant prior (GT-confirmed: tight 0.82px -> wide-FOV 71px -> crop-back
    0.65px). This Affine simulates that wide FOV at TRAIN time (scale DOWN to 0.4 = anatomy shrunk
    into a black frame), so the encoder learns to localise small-in-frame anatomy. Zoom-out keeps
    landmarks in-frame (they move toward centre) -> reject-sampling never fires. Conformal
    (keep_ratio + no shear) so Cervical-Length distance stays param-safe; BORDER_CONSTANT mandatory.
    """
    return A.Affine(
        scale=(0.4, 1.1),
        rotate=(-12, 12),
        translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
        shear=0.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        keep_ratio=True,
        fit_output=False,
        rotate_method="largest_box",
        p=0.85,
    )


def build_fugc_scale_transforms(input: int = 518, seed: Optional[int] = None) -> A.Compose:
    """FUGC scale/FOV-robustness override transform (keypoint-aware): photo_v1 + wide zoom-out Affine.
    Applied ONLY to FUGC samples via KeypointAugDataset.task_transforms in the multi-task trainer."""
    return A.Compose(
        [*_photo_v1(), _fugc_scale_affine(), A.Resize(input, input), A.Normalize(MEAN, STD), ToTensorV2()],
        keypoint_params=_kp_params(), seed=seed,
    )


def _kp_params():
    return A.KeypointParams(format="xy", remove_invisible=False, check_each_transform=True)


def build_train_transform(pack: str = "none", input: int = 518,
                          seed: Optional[int] = None) -> A.Compose:
    if not A.__version__.startswith("2.0"):
        raise RuntimeError(
            f"augment.py is written for albumentations 2.0.x; found {A.__version__}. "
            "1.x idioms (e.g. GaussNoise var_limit, Affine cval) silently misbehave."
        )
    if pack not in PACKS:
        raise ValueError(f"unknown aug pack {pack!r}; expected one of {PACKS}")
    if pack == "geo_v1":
        return A.Compose(
            [*_photo_v1(), _geo_v1_affine(), A.Resize(input, input), A.Normalize(MEAN, STD), ToTensorV2()],
            keypoint_params=_kp_params(), seed=seed,
        )
    if pack == "fugc_scale_v1":
        return A.Compose(
            [*_photo_v1(), _fugc_scale_affine(), A.Resize(input, input), A.Normalize(MEAN, STD), ToTensorV2()],
            keypoint_params=_kp_params(), seed=seed,
        )
    photo = _photo_v1() if pack == "photo_v1" else []
    return A.Compose([*photo, A.Resize(input, input), A.Normalize(MEAN, STD), ToTensorV2()], seed=seed)


def build_geo_fallback(input: int = 518, seed: Optional[int] = None) -> A.Compose:
    """Keypoint-aware photometric-only Compose — the reject-sampling fallback (valid geometry)."""
    return A.Compose(
        [*_photo_v1(), A.Resize(input, input), A.Normalize(MEAN, STD), ToTensorV2()],
        keypoint_params=_kp_params(), seed=seed,
    )


def _aop_probe_angle_affine():
    """Wider Affine for AOP: ±30° rotation (vs geo_v1's ±12°) to simulate fan-angle variation.

    Same conformal constraints as geo_v1 (keep_ratio, no shear) so the angle ratio param-safe.
    BORDER_CONSTANT mandatory — same reason as geo_v1.
    """
    return A.Affine(
        scale=(0.85, 1.15),
        rotate=(-30, 30),
        translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
        shear=0.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        keep_ratio=True,
        fit_output=False,
        rotate_method="largest_box",
        p=0.7,
    )


def _aop_perspective():
    """Perspective warp simulating transperineal probe tilt / viewpoint shift.

    scale=(0.05, 0.12) gives mild-to-moderate viewpoint change without extreme cropping.
    BORDER_CONSTANT + keep_size so the output is still input×input.
    """
    return A.Perspective(
        scale=(0.05, 0.12),
        keep_size=True,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        p=0.4,
    )


def _aop_optical_distortion():
    """Mild optical distortion simulating lens / scanner-geometry variation."""
    return A.OpticalDistortion(
        distort_limit=(-0.1, 0.1),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=0.2,
    )


def build_aop_task_transforms(input: int = 518, seed: Optional[int] = None) -> A.Compose:
    """AOP probe-angle override transform (keypoint-aware).

    Stack: photo_v1 + wider Affine (±30°) + Perspective + OpticalDistortion.
    All geometric ops use BORDER_CONSTANT + keep_size — identical to geo_v1's contract.
    Applied ONLY to AOP samples via KeypointAugDataset.task_transforms.
    """
    return A.Compose(
        [
            *_photo_v1(),
            _aop_probe_angle_affine(),
            _aop_perspective(),
            _aop_optical_distortion(),
            A.Resize(input, input),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ],
        keypoint_params=_kp_params(),
        seed=seed,
    )


def _hc_smallhead_affine():
    """Aggressive ZOOM-OUT Affine for HC small-head robustness (training-side small-head lever).

    HC (head circumference) is tail-dominated: the worst-10% of HC images = 44% of HC error and the
    failures are SMALL / faint heads (error correlates −0.30 with head size) = genuine small-head OOD
    mislocalization. Inference-only fixes (ellipse-decode, multi-view TTA) were both REJECTED; but a
    synthetic wide-FOV test recovered −3.36px (zoom-out = larger FOV = smaller head un-suppresses the
    error). So the lever is TRAINING-side: synthesize small heads by zooming OUT so the head occupies a
    smaller fraction of the frame with a black-padded border — the same mechanism the synthetic recovery
    exploited. scale=(0.5, 1.05) skews hard toward zoom-out (down to 2× smaller head) while keeping
    near-native scale in reach; conformal (keep_ratio + no shear) so the HC ELLIPSE ratio/angle stays
    param-MAE safe; BORDER_CONSTANT mandatory (reflect multiplies keypoints, breaks fixed num_points).
    Zoom-out keeps the 4 ellipse landmarks in-frame (they move toward centre) so reject-sampling rarely
    fires. Wider translate than geo_v1 so the shrunken head isn't always centred.
    """
    return A.Affine(
        scale=(0.5, 1.05),
        rotate=(-12, 12),
        translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
        shear=0.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        keep_ratio=True,
        fit_output=False,
        rotate_method="largest_box",
        p=0.85,
    )


def build_hc_smallhead_transform(input: int = 518, seed: Optional[int] = None) -> A.Compose:
    """HC small-head override transform (keypoint-aware): photo_v1 + aggressive zoom-out Affine.

    Same conventions as build_aop_task_transforms / build_fugc_scale_transforms — applied ONLY to HC
    samples via KeypointAugDataset.task_transforms in the multi-task trainer; all other tasks keep the
    base geo_v1 pipeline unchanged. Reject-sampling fallback = build_geo_fallback (photometric-only).
    """
    return A.Compose(
        [*_photo_v1(), _hc_smallhead_affine(), A.Resize(input, input),
         A.Normalize(MEAN, STD), ToTensorV2()],
        keypoint_params=_kp_params(), seed=seed,
    )
