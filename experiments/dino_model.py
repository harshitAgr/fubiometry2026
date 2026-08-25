"""DINO student/teacher wrapping a project ViT-S/14 or ViT-B/14 DINOv2 encoder.

Corrects Axis A's MAE mismatch: DINOv2's own pretraining objective is DINO self-distillation (+
iBOT patch MIM, dropped here — see module-level scope note in dino_pretrain.py). Continuing THAT
objective (not masked-reconstruction) on the 24k in-domain pool is the more faithful "continued
pretrain" of this exact backbone.

Adaptation strategy: PARTIAL UNFREEZE of only the last `unfreeze_blocks` transformer blocks (+
final norm) + the new DINO head. Chosen over full-FT given Axis A's measured 1.3x feature drift on
24k images (vs ~0.18 noise-floor) — freezing the first N-k blocks makes it structurally impossible
for early/mid-layer general features to drift, bounding the blast radius to exactly the unfrozen
tail while still letting the encoder specialize its late-layer semantics to ultrasound. Chosen over
LoRA for simplicity/no-new-dependency (repo convention: minimal deps) — partial-unfreeze is
strictly easier to get right and to audit (freeze boolean per block vs a low-rank re-parameterization).

Head: MLP -> L2-normalize -> weight-normalized Linear (Caron et al. 2021 DINOHead), simplified:
we use nn.utils.parametrizations.weight_norm instead of hand-rolling the frozen-norm variant DINO
uses for the last layer; functionally the same weight-norm decomposition, just letting both g and v
train (DINO freezes g at 1 for early stability -- an optional refinement we skip for scope).
"""
from __future__ import annotations

import importlib

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    """CLS-token projection head: MLP -> L2-normalize -> weight-normalized Linear.

    out_dim is the "number of prototypes" (Sinkhorn-free softmax targets), bottleneck < out_dim
    is standard DINO practice (compress before expanding into the prototype space).
    """

    def __init__(self, in_dim: int, out_dim: int = 4096, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DinoViTEncoder(nn.Module):
    """timm DINOv2 backbone -> CLS token, with the last `unfreeze_blocks` blocks (+ final
    norm) trainable and everything else (patch_embed, pos_embed, cls_token, earlier blocks) frozen.

    unfreeze_blocks=0 freezes the whole backbone (head-only training); unfreeze_blocks>=depth
    unfreezes everything (full-FT, NOT the adopted default here).
    """

    def __init__(self, input_size: int = 518, unfreeze_blocks: int = 2,
                 encoder_name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True):
        super().__init__()
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, img_size=input_size,
            dynamic_img_size=True,  # DINO multi-crop feeds global (input_size) AND local
            # (smaller) crops through this SAME backbone -- a fixed img_size bakes an exact-match
            # assertion into patch_embed that only the global crops satisfy. dynamic_img_size
            # interpolates the position embedding per-forward instead of asserting a fixed grid.
        )
        self.embed_dim = int(self.backbone.embed_dim)
        depth = len(self.backbone.blocks)
        self.unfreeze_blocks = max(0, min(unfreeze_blocks, depth))
        self._set_trainable(self.unfreeze_blocks)

    def _set_trainable(self, unfreeze_blocks: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        depth = len(self.backbone.blocks)
        for blk in self.backbone.blocks[depth - unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        if unfreeze_blocks > 0:
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, embed_dim] CLS token (post final norm)."""
        tokens = self.backbone.forward_features(x)
        return tokens[:, 0, :]


class DinoStudentTeacher(nn.Module):
    """Student (trainable per unfreeze_blocks) + teacher (EMA, no-grad, always eval()).

    Teacher is a structurally separate module (deep-copied backbone + head) so its BatchNorm/
    LayerNorm/stochastic-depth behaviour never sees train-mode statistics -- matching the fix
    already made in experiments/ssl_train.py's mean-teacher (teacher.eval() always, for stable
    targets). Teacher params carry requires_grad=False; they are updated only via the EMA math in
    dino_loss.ema_update_, never by backward().
    """

    def __init__(self, input_size: int = 518, unfreeze_blocks: int = 2,
                 encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
                 head_out_dim: int = 4096, head_hidden_dim: int = 2048,
                 head_bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.student_encoder = DinoViTEncoder(input_size, unfreeze_blocks, encoder_name, pretrained)
        self.student_head = DINOHead(self.student_encoder.embed_dim, head_out_dim,
                                     head_hidden_dim, head_bottleneck_dim)
        self.teacher_encoder = DinoViTEncoder(input_size, unfreeze_blocks, encoder_name, pretrained)
        self.teacher_head = DINOHead(self.teacher_encoder.embed_dim, head_out_dim,
                                     head_hidden_dim, head_bottleneck_dim)
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.teacher_encoder.eval()
        self.teacher_head.eval()

    def train(self, mode: bool = True):
        """Override so .train() never flips the teacher to train mode (must stay .eval() always
        for stable LayerNorm/dropout/stochastic-depth statistics -- an EMA target that jitters
        with train-mode noise is a moving, noisy target and destabilizes the student)."""
        self.student_encoder.train(mode)
        self.student_head.train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self

    def student_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_head(self.student_encoder(x))

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.teacher_encoder(x))

    def ema_update(self, momentum: float) -> None:
        from experiments.dino_loss import ema_update_
        ema_update_(list(self.teacher_encoder.parameters()), list(self.student_encoder.parameters()), momentum)
        ema_update_(list(self.teacher_head.parameters()), list(self.student_head.parameters()), momentum)

    def encoder_state_dict(self) -> dict:
        """The timm DINOv2 backbone state_dict (head discarded) — the fine-tune init. Saves the
        TEACHER's backbone: DINO's own recipe (Caron et al. 2021, Sec 3.4) reports the EMA teacher
        as the better feature extractor for downstream transfer (it's an ensemble-like average over
        recent student states, so its features are less noisy than any single student snapshot).

        Frozen tensors are copied from the student when exporting. Although teacher and student are
        mathematically identical there, thousands of ``m*t + (1-m)*s`` float32 EMA operations cause
        measurable roundoff drift if frozen teacher tensors are exported directly. Restoring them
        preserves the partial-unfreeze invariant exactly while retaining EMA teacher values for the
        trainable tail and final norm.
        """
        teacher = self.teacher_encoder.backbone.state_dict()
        student = self.student_encoder.backbone.state_dict()
        trainable = {
            name for name, parameter in self.student_encoder.backbone.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value if name in trainable else student[name]
            for name, value in teacher.items()
        }
