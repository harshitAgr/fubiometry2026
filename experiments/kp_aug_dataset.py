"""Keypoint-aware dataset for Lever 3 Phase 2 (geometric augmentation).

Subclasses the baseline KeypointDataset (NOT edited) and overrides __getitem__ to co-transform
landmarks through a keypoint-aware albumentations Compose, with reject-sampling for out-of-frame
landmarks (fallback to a photometric-only Compose), normalize-by-INPUT, and heatmap rebuild.
Train-only: the held-out fold is scored by the separate infer_tta path. Run with the baseline venv.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
sys.path.insert(0, PROJ)
from dataset import KeypointDataset  # noqa: E402
from experiments.per_task_model import hm_for, sigma_for  # noqa: E402


class KeypointAugDataset(KeypointDataset):
    """Geometric-aware KeypointDataset. `transforms` and `fallback_transforms` MUST be keypoint-aware
    Composes (A.KeypointParams(format='xy', remove_invisible=False)). Reject-samples the (random)
    geometric transform until all landmarks land in [0, INPUT); else uses fallback_transforms.

    Per-task override: `task_transforms` and `task_fallback_transforms` are optional dicts mapping
    task_id strings to keypoint-aware A.Compose objects. When a sample's task_id is present in
    `task_transforms`, that Compose is used instead of the base `transforms`; the corresponding
    entry in `task_fallback_transforms` (or the base `fallback_transforms` if absent) is used for
    reject-sampling fallback. Non-overridden tasks use the base transforms unchanged.
    """

    INPUT = 518
    MAX_TRIES = 10
    # FA landmark order is p0=top, p1=bottom (vertical/diameter pair), p2=right, p3=left
    # (horizontal/diameter pair) -- see scoring/param_specs.py's ellipse_perimeter(0,1,2,3).
    # `fa_aniso_sigma` renders ONLY these two wall channels anisotropically.
    FA_ANISO_CHANNELS = (2, 3)

    def __init__(self, *args, fallback_transforms=None, input_size=INPUT,
                 task_transforms=None, task_fallback_transforms=None,
                 fa_aniso_sigma=None, **kwargs):
        super().__init__(*args, **kwargs)
        if fallback_transforms is None:
            raise ValueError("KeypointAugDataset requires fallback_transforms (keypoint-aware)")
        self.fallback_transforms = fallback_transforms
        # normalize keypoints + in-frame check by the ACTUAL input size (the transform's
        # resize target), not the hardcoded 518 — else targets mis-scale at other resolutions.
        self.input_size = int(input_size)
        # Per-task overrides (None -> no per-task behaviour; empty dict -> same as None).
        self.task_transforms: dict = dict(task_transforms) if task_transforms else {}
        self.task_fallback_transforms: dict = dict(task_fallback_transforms) if task_fallback_transforms else {}
        # FA-only anisotropic target lever (default None -> _gen_heatmaps_per_task is
        # byte-identical to the pre-lever isotropic renderer for EVERY task, FA included).
        # (sx, sy) in heatmap-grid CELLS, applied only to FA_ANISO_CHANNELS.
        self.fa_aniso_sigma = tuple(float(v) for v in fa_aniso_sigma) if fa_aniso_sigma is not None else None

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record["task_id"]
        path = self._resolve_image_path(record["image_path"])
        if path is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.imread(path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        n = int(record["num_classes"])
        kps = []
        for i in range(1, n + 1):
            col = f"point_{i}_xy"
            if col in record and pd.notna(record[col]):
                xy = json.loads(record[col])
                kps.append((float(xy[0]), float(xy[1])))
            else:
                kps.append((0.0, 0.0))

        # Select the transform to use: per-task override takes priority over base.
        active_transforms = self.task_transforms.get(task_id, self.transforms)
        active_fallback = self.task_fallback_transforms.get(task_id, self.fallback_transforms)

        out, k = None, None
        for _ in range(self.MAX_TRIES):
            cand = active_transforms(image=image, keypoints=kps)
            ck = np.asarray(cand["keypoints"], dtype=np.float32)
            if ck.shape[0] == n and (ck >= 0).all() and (ck < self.input_size).all():  # in [0,input_size)
                out, k = cand, ck
                break
        if out is None:
            out = active_fallback(image=image, keypoints=kps)
            k = np.asarray(out["keypoints"], dtype=np.float32)

        label = np.empty(2 * n, dtype=np.float32)
        label[0::2] = k[:, 0] / self.input_size
        label[1::2] = k[:, 1] / self.input_size
        label = np.clip(label, 0.0, 1.0)   # safety no-op when in-frame
        heatmaps = self._gen_heatmaps_per_task(label, n, task_id)
        return {
            "image": out["image"],
            "label": torch.from_numpy(label).float(),
            "heatmap": torch.from_numpy(heatmaps).float(),
            "task_id": task_id,
        }

    def _gen_heatmaps_per_task(self, norm_coords, num_points, task_id):
        """Like the baseline _generate_heatmaps but BOTH the target size AND the Gaussian sigma are
        PER-TASK (hm_for(self.heatmap_size, task_id) / sigma_for(self.sigma, task_id)); identical to
        the baseline when heatmap_size is a uniform tuple AND sigma is a scalar. Scaling sigma with
        the finer grid keeps the PHYSICAL peak width constant (removes the HM=128 sharpness confound).

        FA wall-variance lever: when `self.fa_aniso_sigma` is set AND task_id == "FA", channels
        FA_ANISO_CHANNELS (p2/p3, the lateral-wall landmarks) render as an ANISOTROPIC Gaussian
        exp(-(dx^2/(2*sx^2) + dy^2/(2*sy^2))) instead of the isotropic exp(-(dx^2+dy^2)/(2*sig^2));
        channels p0/p1 are untouched (still isotropic at `sig`). With fa_aniso_sigma=None (the
        default for every caller that doesn't pass it), this method is BYTE-IDENTICAL to before."""
        hh, hw = hm_for(self.heatmap_size, task_id)
        sig = sigma_for(self.sigma, task_id)
        yy, xx = np.meshgrid(np.arange(hh), np.arange(hw), indexing="ij")
        out = np.zeros((num_points, hh, hw), dtype=np.float32)
        aniso = self.fa_aniso_sigma if (task_id == "FA" and self.fa_aniso_sigma is not None) else None
        if aniso is not None and num_points != 4:
            raise ValueError(
                f"fa_aniso_sigma requires FA to have exactly 4 landmarks, got {num_points}")
        for i in range(num_points):
            x = min(max(float(norm_coords[2 * i]), 0.0), 1.0) * (hw - 1)
            y = min(max(float(norm_coords[2 * i + 1]), 0.0), 1.0) * (hh - 1)
            if aniso is not None and i in self.FA_ANISO_CHANNELS:
                sx, sy = aniso
                out[i] = np.exp(-((xx - x) ** 2 / (2.0 * sx * sx) + (yy - y) ** 2 / (2.0 * sy * sy)))
            else:
                out[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sig * sig))
        return out
