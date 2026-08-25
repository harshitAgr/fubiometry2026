"""Pluggable encoder factory for the FU Biometry multi-task model.

Encoder names:
  "dinov2_vits"   — ViT-S/14 DINOv2 (384-ch, input 518, 37×37 grid)
  "dinov2_vitb"   — ViT-B/14 DINOv2 (768-ch, input 518, 37×37 grid — capacity probe)
  "dinov2_vitb_fuse4" — ViT-B/14 with zero-init residual fusion of the last four blocks
  "dinov2_vitl"   — ViT-L/14 DINOv2 (1024-ch, input 518, 37×37 grid — capacity probe, 3.5x ViT-B)
  "beit_imagenet" — BEiT-B/16 ImageNet (768-ch, input 224, 14×14 grid)
  "usfm_beit"     — BEiT-B/16 USFM US-domain SSL (768-ch, input 224, 14×14 grid)
  "dinov3_vits"   — ViT-S/16 DINOv3 lvd1689m (384-ch, patch-16, n_prefix=5; 512 or 592)
  "dinov3_vitb"   — ViT-B/16 DINOv3 lvd1689m (768-ch, patch-16, n_prefix=5; 512 or 592)

Each returned module has:
  .forward(x: [B,3,H,W]) -> [B,C,G,G]  (CLS/prefix stripped, square patch grid)
  .out_channels: int
"""
import importlib
import math
import os
import sys

import torch
import torch.nn as nn

# Ensure baseline package is importable when this module is imported stand-alone
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_baseline_path = os.path.join(PROJ, "baseline", "baseline")
if _baseline_path not in sys.path:
    sys.path.insert(0, _baseline_path)

from model_factory import DINOv2Backbone  # noqa: E402 — reuse verbatim


class ChannelwiseResidualFusion(nn.Module):
    """Per-channel mixture of three aligned feature maps, initialized to zero."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, channels))

    def forward(self, features) -> torch.Tensor:
        if len(features) != 3:
            raise RuntimeError(f"expected three residual feature maps, got {len(features)}")
        stacked = torch.stack(features, dim=1)  # [B,3,C,H,W]
        return (stacked * self.weight[None, :, :, None, None]).sum(dim=1)


class DINOv2Last4FusionBackbone(DINOv2Backbone):
    """DINOv2 ViT-B with an identity-initialized residual last-four-layer adapter.

    The last normalized feature map remains the spine.  A zero-initialized 1x1
    per-channel mixture may add information from the preceding three normalized maps:

        output[c] = final[c] + sum_i fusion.weight[i,c] * block[i,c]

    At construction, ``fusion`` is exactly zero, so output is tensor-identical
    to the ordinary final-layer DINOv2Backbone.  Keeping ``backbone`` at the same
    attribute path also lets an adopted checkpoint load every historical tensor;
    only ``fusion.weight`` is new.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__("vit_base_patch14_dinov2.lvd142m", pretrained=pretrained)
        channels = self.out_channels
        self.fusion = ChannelwiseResidualFusion(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_intermediate_layers(
            x, n=4, reshape=True, return_prefix_tokens=False, norm=True)
        if len(features) != 4:
            raise RuntimeError(f"expected four DINOv2 feature maps, got {len(features)}")
        return features[-1] + self.fusion(features[:-1])


def _tokens_to_map(tokens: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """[B, T, C] -> [B, C, G, G], stripping n_prefix non-patch (CLS/register) tokens.

    Asserts the remaining patch count is a perfect square.
    """
    patch = tokens[:, n_prefix:, :]
    b, n, c = patch.shape
    g = int(round(math.sqrt(n)))
    assert g * g == n, f"non-square patch grid: n={n}, after stripping n_prefix={n_prefix}"
    return patch.transpose(1, 2).reshape(b, c, g, g)


# ---------------------------------------------------------------------------
# BEiT-ImageNet control backbone
# ---------------------------------------------------------------------------

class BeitImageNet(nn.Module):
    """timm beit_base_patch16_224 ImageNet weights — matched-arch control for USFM.

    Uses 1 CLS prefix token; native input 224 (14×14 patch grid).
    If pretrained weights are unavailable (offline), construction raises; callers
    should guard with pytest.mark.skipif or a try/except as appropriate.

    NOTE: we do NOT override use_shared_rel_pos_bias/use_rel_pos_bias — the
    in22k_ft_in22k_in1k model has its own defaults and timm's checkpoint loader
    will error if we try to override the bias configuration post-hoc.
    """

    def __init__(self, input_size: int = 224):
        super().__init__()
        timm = importlib.import_module("timm")
        # BEiT-B/16 ImageNet — use model's native bias config; global_pool='token' to
        # get [B, 1+G*G, 768] from forward_features (with CLS at position 0)
        self.backbone = timm.create_model(
            "beit_base_patch16_224.in22k_ft_in22k_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="token",
            img_size=input_size,
        )
        self.out_channels = int(self.backbone.num_features)  # 768
        self.n_prefix = 1  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)  # [B, 1+G*G, 768]
        return _tokens_to_map(tokens, self.n_prefix)


# ---------------------------------------------------------------------------
# USFM backbone (BEiT-B/16 US-domain SSL)
# ---------------------------------------------------------------------------

class USFMBackbone(nn.Module):
    """BEiT-B/16 USFM ultrasound foundation model encoder.

    Loads from assets/FMweight/USFM_latest.pth (gitignored, 343 MB).
    Key-coverage assertion: after dropping mask_token and
    rel_pos_bias.relative_position_index, strict=False load yields
    0 missing and 0 unexpected keys at native 224 resolution.

    Attributes stored for test assertions:
      .missing_keys   — list of missing keys after load (must be [])
      .unexpected_keys — list of unexpected keys after load (must be [])
    """

    WEIGHTS_PATH = os.path.join(PROJ, "assets", "FMweight", "USFM_latest.pth")

    # Keys present in USFM checkpoint that are not part of the timm BEiT backbone
    _KNOWN_DROP = frozenset({"mask_token", "rel_pos_bias.relative_position_index"})

    _BIAS_SUFFIX = "relative_position_bias_table"

    def __init__(self, input_size: int = 224):
        super().__init__()
        timm = importlib.import_module("timm")
        # BEiT-style ViT-B/16 (shared rel-pos-bias, LayerScale, q/v-bias)
        self.backbone = timm.create_model(
            "beit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            use_shared_rel_pos_bias=True,
            use_rel_pos_bias=False,
            global_pool="token",
            img_size=input_size,
        )

        if not os.path.exists(self.WEIGHTS_PATH):
            raise FileNotFoundError(
                f"USFM weights not found at {self.WEIGHTS_PATH}. "
                "Download from https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/"
            )

        sd = torch.load(self.WEIGHTS_PATH, map_location="cpu", weights_only=False)

        # Checkpoint is a flat state_dict: no 'state_dict'/'model' prefix nesting -- the
        # top-level dict IS the state_dict (verified by inspecting USFM_latest.pth).
        for key in ("state_dict", "model", "module"):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]

        # Drop keys that are not part of the backbone (MIM-only / buffer)
        sd = {
            k: v for k, v in sd.items()
            if hasattr(v, "shape") and k not in self._KNOWN_DROP
        }

        # BEiT's shared relative-position-bias table is tied to the pretraining grid
        # (14×14 @224). At any other input size (e.g. 512 → 32×32) the table must be
        # interpolated to the target window before load. We mirror timm's OWN BEiT
        # checkpoint loader (timm.models.beit.checkpoint_filter_fn): for each
        # *.relative_position_bias_table key, find the owning RelativePositionBias module
        # and resize via timm.layers.resize_rel_pos_bias_table to that module's
        # window_size / table shape. No hand-rolled interpolation. (No-op at 224.)
        sd = self._resize_rel_pos_bias(sd)

        info = self.backbone.load_state_dict(sd, strict=False)
        self.missing_keys = list(info.missing_keys)
        self.unexpected_keys = list(info.unexpected_keys)

        if self.missing_keys:
            raise RuntimeError(
                f"USFM key-coverage FAILED — missing keys: {self.missing_keys[:8]}"
            )

        self.out_channels = 768
        self.n_prefix = 1  # CLS token only; mask_token is MIM-only and dropped at inference

        # Normalization: USFM training transform unresolved in repo README.
        # Use ImageNet norm as the baseline; Task-5 smoke will probe [0,1] if needed.
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

    def _resize_rel_pos_bias(self, sd: dict) -> dict:
        """Resize any *.relative_position_bias_table in `sd` to the built model's grid.

        Mirrors timm.models.beit.checkpoint_filter_fn exactly: for each table key, locate
        the owning RelativePositionBias submodule (key minus the trailing
        '.relative_position_bias_table') and call timm.layers.resize_rel_pos_bias_table with
        that module's window_size and target table shape. timm's helper internally handles
        the BEiT extra-token rows (the 3 non-spatial cls↔patch bias entries) and uses
        geometric-progression sampling — so we get timm's own interpolation, not a hand-roll.

        No-op when shapes already match (e.g. input_size == 224).
        """
        resize_fn = importlib.import_module("timm.layers").resize_rel_pos_bias_table
        out = {}
        suffix = "." + self._BIAS_SUFFIX
        for k, v in sd.items():
            if k.endswith(suffix):
                module = self.backbone.get_submodule(k[: -len(suffix)])
                target = module.relative_position_bias_table
                if tuple(v.shape) != tuple(target.shape) or module.window_size[0] != module.window_size[1]:
                    v = resize_fn(
                        v,
                        new_window_size=module.window_size,
                        new_bias_shape=target.shape,
                    )
            out[k] = v
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)  # [B, 1+G*G, 768]
        return _tokens_to_map(tokens, self.n_prefix)


# ---------------------------------------------------------------------------
# DINOv3 backbone (ViT-S/16, web SSL — lvd1689m)
# ---------------------------------------------------------------------------

class DINOv3Backbone(nn.Module):
    """DINOv3 ViT-{S,B}/16 (timm `vit_{small,base}_patch16_dinov3.lvd1689m`).

    embed_dim matches the corresponding DINOv2 spine (ViT-S → 384, ViT-B → 768) so the heatmap
    head is unchanged, n_prefix=5 (1 CLS + 4 register tokens), patch-16 with RoPE position encoding
    (so off-native input sizes work; input must be divisible by 16). ImageNet normalization.

    Weights are HF-gated under Meta's DINOv3 License (access granted on this machine; cached).
    """

    def __init__(self, input_size: int = 512, model_name: str = "vit_small_patch16_dinov3",
                 out_channels: int = 384):
        super().__init__()
        assert input_size % 16 == 0, (
            f"DINOv3 is patch-16; input_size must be divisible by 16, got {input_size} "
            f"(the DINOv2 spine's 518 is INVALID here — use 512 or 592)."
        )
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=input_size,
        )
        self.out_channels = out_channels
        self.n_prefix = 5  # 1 CLS + 4 register tokens (verified: forward_features = [B, 5+G*G, C])
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)  # [B, 5+G*G, C]
        return _tokens_to_map(tokens, self.n_prefix)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_encoder_init(backbone: nn.Module, encoder_init: str) -> None:
    """Load a custom DINOv2 backbone state_dict (e.g. continued-SSL, Axis A) into `backbone`.

    The saved file is the matching timm backbone state_dict (decoder discarded). We drop any non-
    backbone keys (mask_token / decoder.* — a no-op safety since the saved file is encoder-only)
    and require strict coverage of the backbone: 0 missing AND 0 unexpected keys. Incomplete
    threading is the documented trap (the input_size bug) — fail loud rather than silently load a
    mismatched encoder.
    """
    if not os.path.exists(encoder_init):
        raise FileNotFoundError(f"--encoder-init checkpoint not found: {encoder_init}")
    sd = torch.load(encoder_init, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module", "encoder"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    sd = {k: v for k, v in sd.items()
          if hasattr(v, "shape") and not k.startswith(("decoder", "mask_token"))}
    info = backbone.load_state_dict(sd, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            f"encoder-init coverage FAILED — missing={info.missing_keys[:8]} "
            f"unexpected={info.unexpected_keys[:8]}"
        )


def build_encoder(name: str, input_size: int, encoder_init: str | None = None) -> nn.Module:
    """Build and return an encoder module by name.

    Args:
        name:        One of "dinov2_vits", "beit_imagenet", "usfm_beit".
        input_size:  Spatial input size in pixels (square).
        encoder_init: Optional path to a custom backbone state_dict to load AFTER building the
                     pretrained encoder (Axis A: continued-DINO weights). Supported for
                     "dinov2_vits" and "dinov2_vitb". At fine-tune-SCORE time it is unnecessary
                     (the fine-tuned checkpoint already carries these weights) but is accepted for
                     symmetry/safety; when set it is applied before the FT state_dict overwrites it.

    Returns:
        nn.Module with .forward(x) -> [B,C,G,G] and .out_channels attribute.
    """
    if name == "dinov2_vits":
        enc = DINOv2Backbone("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        if encoder_init:
            _load_encoder_init(enc.backbone, encoder_init)
        return enc

    if name == "dinov2_vitb":
        enc = DINOv2Backbone("vit_base_patch14_dinov2.lvd142m", pretrained=True)
        if encoder_init:
            _load_encoder_init(enc.backbone, encoder_init)
        return enc

    if encoder_init:
        raise ValueError(
            "--encoder-init is only supported for 'dinov2_vits' and 'dinov2_vitb', "
            f"not {name!r}."
        )

    if name == "beit_imagenet":
        return BeitImageNet(input_size)

    if name == "usfm_beit":
        return USFMBackbone(input_size)

    if name == "dinov3_vits":
        return DINOv3Backbone(input_size)

    if name == "dinov3_vitb":
        # DINOv3 ViT-B/16 (768-ch == DINOv2 ViT-B spine head; patch-16, n_prefix=5, lvd1689m web SSL).
        return DINOv3Backbone(input_size, model_name="vit_base_patch16_dinov3", out_channels=768)

    if name == "dinov2_vitb_fuse4":
        return DINOv2Last4FusionBackbone(pretrained=True)

    if name == "dinov2_vitl":
        # DINOv2 ViT-L/14 (1024-ch, input 518, 37×37 grid — same patch/grid as ViT-S/B, ~3.5x
        # ViT-B params). Continues the CAPACITY axis (the only encoder axis that has ever won
        # here: ViT-S -> ViT-B was -1.35 task-mean, 5/5 folds and 9/9 tasks negative), as opposed
        # to the PRETRAINING axis (USFM tie, DINOv3 ViT-S tie, DINOv3 ViT-B decisive loss).
        return DINOv2Backbone("vit_large_patch14_dinov2.lvd142m", pretrained=True)

    raise ValueError(
        f"Unknown encoder name: {name!r}. Valid choices: 'dinov2_vits', 'dinov2_vitb', "
        f"'dinov2_vitb_fuse4', "
        f"'dinov2_vitl', 'beit_imagenet', 'usfm_beit', 'dinov3_vits', 'dinov3_vitb'."
    )
