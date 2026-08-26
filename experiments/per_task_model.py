"""Per-task heatmap-resolution wrapper around the baseline MultiTaskModelFactory.

The HM=128 probe showed finer heatmaps help the PRECISE 2-landmark tasks (FUGC −2.5, fetal_femur
−2.1) but hurt the multi-landmark ones (sigma confound + harder learning) — so a GLOBAL finer grid
is a net negative. The fix is PER-TASK resolution: the precise 2-pt tasks (FUGC, fetal_femur) get a
finer grid, everything else stays @64. The baseline HeatmapHead already accepts a per-call
`out_size`, so this only needs `forward` to look the size up per task; the dataset must build
matching per-task targets (see kp_aug_dataset). The size map is fully task-agnostic — pass any
{task_id: (h,w)} dict (run_config exposes --fugc-heatmap-size / --femur-heatmap-size).

`heatmap_size` may be either a tuple (h,w) [uniform, the baseline behaviour] or a dict
{task_id: (h,w)} with a (64,64) default for unlisted tasks.
"""
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "baseline", "baseline"))
from model_factory import MultiTaskModelFactory, HeatmapHead  # noqa: E402

DEFAULT_HM = (64, 64)
DEFAULT_SIGMA = 1.8


class SEBlock(nn.Module):
    """Lightweight channel attention used by the coordinate-grid refinement head."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], x.shape[1], 1, 1)


class DSNTLayer(nn.Module):
    """Task-specific learnable-temperature spatial expectation."""

    def __init__(self, heatmap_size):
        super().__init__()
        h, w = heatmap_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("grid_x", xx.reshape(1, 1, h, w), persistent=False)
        self.register_buffer("grid_y", yy.reshape(1, 1, h, w), persistent=False)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        b, k, h, w = logits.shape
        z = (logits * self.temperature).reshape(b, k, -1)
        weights = torch.softmax(z, dim=-1).reshape(b, k, h, w)
        x = (weights * self.grid_x).sum(dim=(2, 3))
        y = (weights * self.grid_y).sum(dim=(2, 3))
        coords = torch.stack((x, y), dim=2)
        return coords, weights


class CoordSEHeatmapHead(nn.Module):
    """Coordinate-grid + SE refinement head (rejected probe; not part of the submitted system)."""

    def __init__(self, in_channels, num_points):
        super().__init__()
        hidden = max(in_channels // 2, 256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_channels + 2, hidden, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = nn.Conv2d(hidden + 2, hidden // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden // 2)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(p=0.05)
        self.refine_conv1 = nn.Conv2d(hidden // 2 + 2, hidden // 2, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden // 2)
        self.act3 = nn.GELU()
        self.refine_conv2 = nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden // 2)
        self.se = SEBlock(hidden // 2)
        self.act4 = nn.GELU()
        self.final_proj = nn.Conv2d(hidden // 2, num_points, 1)

    @staticmethod
    def _grid(x):
        b, _, h, w = x.shape
        y = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xg = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        return torch.cat((x, xg.expand(b, 1, h, w), y.expand(b, 1, h, w)), dim=1)

    def forward(self, x, out_size):
        x = self._grid(self.upsample1(x))
        x = self.dropout1(self.act1(self.bn1(self.conv1(x))))
        x = self._grid(self.upsample2(x))
        x = self.dropout2(self.act2(self.bn2(self.conv2(x))))
        residual = self.act3(self.bn3(self.refine_conv1(self._grid(x))))
        residual = self.bn4(self.refine_conv2(residual))
        x = self.act4(self.se(x + residual))
        x = self.final_proj(x)
        if x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


def hm_for(heatmap_size, task_id):
    """Resolve the per-task heatmap size (dict -> lookup w/ default; tuple -> uniform)."""
    if isinstance(heatmap_size, dict):
        return tuple(heatmap_size.get(task_id, DEFAULT_HM))
    return heatmap_size


def sigma_for(sigma, task_id):
    """Resolve the per-task Gaussian target sigma (dict -> lookup w/ default; scalar -> uniform).

    Analogous to hm_for. When a task uses a FINER per-task heatmap grid, its sigma (measured in
    heatmap CELLS) must scale proportionally to keep the PHYSICAL peak width constant — otherwise a
    2x finer grid at the same cell-sigma yields a physically ~2x sharper target. That unscaled
    sharpening was the confound in the earlier global HM=128 probe (FUGC/femur improved but a net
    +2.76 global regression). Callers build a {task_id: sigma} dict for the overridden tasks;
    unlisted tasks fall back to DEFAULT_SIGMA (== the base --sigma default 1.8), mirroring hm_for's
    DEFAULT_HM assumption that the base grid is 64.
    """
    if isinstance(sigma, dict):
        return float(sigma.get(task_id, DEFAULT_SIGMA))
    return sigma


class PerTaskHeatmapModel(MultiTaskModelFactory):
    """MultiTaskModelFactory with per-task output heatmap resolution."""

    def forward(self, x, task_id):
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")
        features = self.encoder(x)
        return self.heads[task_id](features, out_size=hm_for(self.heatmap_size, task_id))


def build_model(cfgs, heatmap_size, encoder, variant="base"):
    """Return a PerTaskHeatmapModel if heatmap_size is per-task (dict), else the plain factory.

    The vendored (gitignored) MultiTaskModelFactory.__init__ has no way to accept a pre-built
    encoder -- it always constructs its own DINOv2Backbone internally. Rather than patching that
    vendored file in place (fragile: baseline/ isn't tracked, so any such patch is lost on a fresh
    clone/machine -- exactly what happened here), let it build its own throwaway (384-channel)
    encoder and heads, then swap the attribute post-construction. If the real encoder has a
    DIFFERENT channel count (e.g. USFM/BEiT at 768 vs the throwaway DINOv2's 384), the heads sized
    for the throwaway encoder are wrong -- rebuild them at the correct in_channels, preserving each
    task's num_points (read off the throwaway head's own final conv layer).
    """
    if variant not in {"base", "coordse"}:
        raise ValueError(f"unknown model variant: {variant!r}")
    cls = PerTaskHeatmapModel if isinstance(heatmap_size, dict) else MultiTaskModelFactory
    model = cls("vit_small_patch14_dinov2.lvd142m", "pretrained", cfgs, heatmap_size)
    factory_channels = model.encoder.out_channels
    model.encoder = encoder
    if variant == "coordse":
        model.heads = nn.ModuleDict()
        model.dsnt_modules = nn.ModuleDict()
        for config in cfgs:
            task_id = config["task_id"]
            num_points = int(config["num_classes"])
            model.heads[task_id] = CoordSEHeatmapHead(encoder.out_channels, num_points)
            model.dsnt_modules[task_id] = DSNTLayer(hm_for(heatmap_size, task_id))
        model.model_variant = variant
    elif encoder.out_channels != factory_channels:
        for task_id, head in model.heads.items():
            num_points = head.decoder[-1].out_channels
            model.heads[task_id] = HeatmapHead(in_channels=encoder.out_channels, num_points=num_points)
        model.model_variant = variant
    else:
        model.model_variant = variant
    return model
