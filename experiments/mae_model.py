"""MAE-style masked image modeling over the timm ViT-S/14 DINOv2 encoder.

Axis A — in-domain continued pretraining. We adapt facebookresearch/mae's MaskedAutoencoderViT
to use the project encoder (timm `vit_small_patch14_dinov2.lvd142m`, init from DINOv2 weights) so
the SAME ViT-S backbone we fine-tune is the one we continue-pretrain. The decoder is lightweight and
DISCARDED after pretraining; only the encoder backbone state_dict is saved as the fine-tune init.

Correctness invariants (the SSL must be done right — asserted in tests):
  * masking keeps exactly round((1-r)*N) patches per sample; `mask` is binary, 1==masked, sums to
    N - kept; `ids_restore` un-shuffles encoder outputs back to canonical patch order.
  * the encoder sees ONLY visible patches (+ CLS) — this is what makes MAE efficient AND forces the
    representation to be predictive, not a denoiser over the full grid.
  * the loss is the per-patch-normalized MSE averaged over MASKED patches only (norm_pix_loss),
    matching MAE; it is a finite scalar with gradient flowing to the encoder.

Run with the BASELINE venv (needs timm/torch). Patch 14 → input must be a multiple of 14.
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# A minimal Transformer decoder block (standard pre-norm MHSA + MLP).
# We don't reuse timm Blocks for the decoder to keep it dependency-light and obviously correct.
# ---------------------------------------------------------------------------
class _DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    """MAE over the timm ViT-S/14 DINOv2 encoder.

    Args:
        input_size:        square input px; must be a multiple of patch (14). 518 for the pipeline.
        mask_ratio:        fraction of patches masked (default 0.75 per MAE).
        decoder_embed_dim: decoder width (192 — lighter than MAE-Base, ViT-S is small).
        decoder_depth:     decoder Transformer depth (4).
        decoder_heads:     decoder attention heads (3).
        norm_pix_loss:     per-patch normalize the target pixels before MSE (MAE default True).
        pretrained:        init the encoder from DINOv2 weights (True for continued pretraining).
        encoder_name:      timm model name (default the project ViT-S/14 DINOv2).
    """

    def __init__(
        self,
        input_size: int = 518,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 3,
        norm_pix_loss: bool = True,
        pretrained: bool = True,
        encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        timm = importlib.import_module("timm")
        self.encoder_name = encoder_name
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)

        # ---- encoder: the timm ViT-S/14 DINOv2 backbone (the one we fine-tune) ----
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, img_size=input_size
        )
        self.embed_dim = int(self.encoder.embed_dim)
        ph, pw = self.encoder.patch_embed.patch_size
        assert ph == pw, "non-square patch not supported"
        self.patch_size = int(ph)
        assert input_size % self.patch_size == 0, (
            f"input_size {input_size} not divisible by patch {self.patch_size}"
        )
        self.grid = input_size // self.patch_size
        self.num_patches = self.grid * self.grid
        # timm DINOv2 ViT-S: 1 CLS prefix token, 0 register tokens.
        self.num_prefix = int(getattr(self.encoder, "num_prefix_tokens", 1))

        # ---- decoder (lightweight; discarded after pretraining) ----
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # learnable decoder pos-embed for the full patch grid (no CLS in the decoder path)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [_DecoderBlock(decoder_embed_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size * self.patch_size * 3, bias=True
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        # IMPORTANT: init ONLY the decoder modules. Using self.apply(...) here would recurse into
        # self.encoder and re-initialize the pretrained DINOv2 Linear/LayerNorm layers, destroying
        # the init (the whole point of *continued* pretraining). Apply per decoder module instead.
        self.decoder_embed.apply(self._init_decoder_weights)
        for blk in self.decoder_blocks:
            blk.apply(self._init_decoder_weights)
        self.decoder_norm.apply(self._init_decoder_weights)
        self.decoder_pred.apply(self._init_decoder_weights)

    @staticmethod
    def _init_decoder_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------ patches
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, N, p*p*3] (row-major patch order)."""
        p = self.patch_size
        b, c, h, w = imgs.shape
        assert c == 3 and h == w and h % p == 0
        g = h // p
        x = imgs.reshape(b, 3, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [b, g(rows), g(cols), p, p, 3]
        x = x.reshape(b, g * g, p * p * 3)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, p*p*3] -> [B,3,H,W]."""
        p = self.patch_size
        g = self.grid
        b = x.shape[0]
        x = x.reshape(b, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4)  # [b,3,g,p,g,p]
        return x.reshape(b, 3, g * p, g * p)

    # ------------------------------------------------------------------ masking
    def random_masking(self, x: torch.Tensor):
        """Per-sample random masking of patch tokens.

        Args:  x: [B, N, D] patch embeddings (no prefix).
        Returns: x_kept [B, n_keep, D], mask [B, N] (1==masked), ids_restore [B, N].
        """
        b, n, d = x.shape
        n_keep = int(round(n * (1.0 - self.mask_ratio)))
        n_keep = max(1, min(n, n_keep))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)        # ascending: first n_keep are kept
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)         # un-shuffle to canonical order
        return x_kept, mask, ids_restore

    # ------------------------------------------------------------------ encoder
    def forward_encoder(self, imgs: torch.Tensor):
        """Embed patches, drop masked, prepend CLS+pos, run the timm blocks on visible tokens.

        Returns: latent [B, 1+n_keep, D] (CLS at 0), mask [B,N], ids_restore [B,N].
        """
        enc = self.encoder
        x = enc.patch_embed(imgs)                          # [B, N, D]
        pos = enc.pos_embed                                # [1, num_prefix+N, D]
        # add patch pos-embed (skip the prefix slots) BEFORE masking, like MAE
        x = x + pos[:, self.num_prefix:, :]
        x_kept, mask, ids_restore = self.random_masking(x)

        # prepend CLS (+ its pos-embed). Use the first prefix slot as CLS.
        cls = enc.cls_token + pos[:, :1, :]
        cls = cls.expand(x_kept.shape[0], -1, -1)
        x = torch.cat([cls, x_kept], dim=1)                # [B, 1+n_keep, D]
        x = enc.norm_pre(x) if hasattr(enc, "norm_pre") else x
        for blk in enc.blocks:
            x = blk(x)
        x = enc.norm(x)
        return x, mask, ids_restore

    # ------------------------------------------------------------------ decoder
    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """latent [B,1+n_keep,D] -> pred [B, N, p*p*3] over ALL patches (CLS dropped)."""
        x = self.decoder_embed(latent)                     # [B, 1+n_keep, Dd]
        b = x.shape[0]
        n = ids_restore.shape[1]
        n_keep = x.shape[1] - 1                            # minus CLS
        # append mask tokens to fill N, then un-shuffle to canonical patch order
        mask_tokens = self.mask_token.expand(b, n - n_keep, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # drop CLS, [B, N, Dd]
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        x = x_ + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)                        # [B, N, p*p*3]

    # ------------------------------------------------------------------ loss
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-patch-normalized MSE, averaged over MASKED patches only (MAE)."""
        target = self.patchify(imgs)                       # [B, N, p*p*3]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                            # [B, N] per-patch MSE
        denom = mask.sum()
        if denom <= 0:
            return loss.mean() * 0.0
        return (loss * mask).sum() / denom                 # mean over masked patches

    def forward(self, imgs: torch.Tensor):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------ save
    def encoder_state_dict(self) -> dict:
        """The timm ViT-S backbone state_dict (decoder discarded) — the fine-tune init."""
        return self.encoder.state_dict()
