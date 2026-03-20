"""
Segmentation Decoder (C) and Domain Discriminator (Dψ)
=======================================================
Decoder:       Z*_0 (N×d_L) → dense segmentation logits (K×H×W)   [Eq. 14-15]
Discriminator: Z (N×d_L)    → domain probability scalar            [Eq. 20-21]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Up-sampling block with skip connections
# ─────────────────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    """Transposed-conv upsampling + skip connection fusion."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Adjust spatial size if needed
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDecoder(nn.Module):
    """
    Decodes refined graph embeddings Z*_0 into dense segmentation maps.

    Pipeline:
      1. Each node embedding is assigned back to its spatial patch.
      2. A UNet-style decoder with skip connections up-samples to H×W.
      3. A 1×1 Conv head produces K-class logits.

    Args:
        node_dim    : d_L – graph node embedding dimension
        enc_channels: tuple of encoder channel widths (shallow → deep)
        num_classes : K
        img_size    : original image size H (=W)
        num_nodes   : N
    """

    def __init__(
        self,
        node_dim:     int = 256,
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256),
        num_classes:  int = 4,
        img_size:     int = 256,
        num_nodes:    int = 256,
    ):
        super().__init__()
        import math
        self.sqrt_N  = int(math.isqrt(num_nodes))
        self.img_size = img_size

        # Initial projection from node_dim → deepest enc channel
        deep_ch = enc_channels[-1]
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, deep_ch),
            nn.ReLU(inplace=True),
        )

        # Build up-blocks (deep → shallow, mirroring the encoder)
        rev_ch = list(reversed(enc_channels))   # [256, 128, 64, 32]
        self.up_blocks = nn.ModuleList()
        for i in range(len(rev_ch) - 1):
            in_ch   = rev_ch[i]
            skip_ch = rev_ch[i + 1]
            out_ch  = rev_ch[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))

        # Final head
        self.head = nn.Conv2d(rev_ch[-1], num_classes, 1)

    def _nodes_to_spatial(self, Z_star: torch.Tensor) -> torch.Tensor:
        """
        Map (B, N, d_L) → (B, deep_ch, sqrt_N, sqrt_N) spatial feature map.
        """
        B, N, _ = Z_star.shape
        h = self.node_proj(Z_star)            # (B, N, deep_ch)
        sN = self.sqrt_N
        h = h.permute(0, 2, 1)               # (B, deep_ch, N)
        h = h.view(B, -1, sN, sN)            # (B, deep_ch, sN, sN)
        return h

    def forward(
        self,
        Z_star: torch.Tensor,
        skip_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            Z_star     : (B, N, d_L)  refined graph embeddings
            skip_feats : list of encoder skip features (shallow → deep)

        Returns:
            logits : (B, K, H, W)
        """
        x = self._nodes_to_spatial(Z_star)    # (B, deep_ch, sN, sN)

        # Reverse skip features: deep → shallow
        skips = list(reversed(skip_feats))    # [deepest_skip, ..., shallowest]

        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[i])

        # Final upsample to img_size if residual mismatch
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        return self.head(x)                   # (B, K, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Domain Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    Latent-space adversarial discriminator Dψ: R^{N×d_L} → [0,1].

    Operates on globally aggregated (mean-pooled) graph embeddings to ensure
    permutation invariance over node ordering.  Trained with standard binary
    cross-entropy:
        L_D = -E[log D(Z_s)] - E[log(1 - D(Z_t))]

    Args:
        node_dim : d_L – per-node embedding dimension
    """

    def __init__(self, node_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z : graph embeddings  (B, N, d_L)
        Returns:
            score : domain probability  (B, 1)
        """
        agg = Z.mean(dim=1)      # (B, d_L)  – permutation-invariant pooling
        return self.net(agg)     # (B, 1)
