"""
Wavelet Frequency Encoder (W)
==============================
Extracts a compact multi-scale frequency summary f ∈ R^{d_f} from the CNN
feature map F.  This vector captures appearance characteristics (contrast,
texture, noise) that are disentangled from anatomical graph topology.

Implementation uses learnable 2-D Haar-style wavelet filter banks
(Eq. 8 in the paper):   f = W(F),  f ∈ R^{d_f}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWaveletBank(nn.Module):
    """
    One scale of a learnable 2-D wavelet decomposition.
    Produces approximation (LL) and three detail sub-bands (LH, HL, HH)
    via a pair of learnable filter-banks.

    Args:
        channels : number of feature channels C
    """

    def __init__(self, channels: int):
        super().__init__()
        # Low-pass and high-pass kernels initialised as Haar wavelets
        lp = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 4.0
        hp = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 4.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 4.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 4.0

        # Shape: (C, 1, 2, 2) – applied channel-wise (depthwise conv)
        def make_kernel(k):
            return k.unsqueeze(0).unsqueeze(0).expand(channels, 1, 2, 2).clone()

        self.register_parameter('w_ll', nn.Parameter(make_kernel(lp)))
        self.register_parameter('w_lh', nn.Parameter(make_kernel(lh)))
        self.register_parameter('w_hl', nn.Parameter(make_kernel(hl)))
        self.register_parameter('w_hh', nn.Parameter(make_kernel(hp)))
        self.C = channels

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, C, H, W)
        Returns:
            tuple of four sub-band tensors each of shape (B, C, H/2, W/2)
        """
        conv = lambda w: F.conv2d(x, w, stride=2, padding=0, groups=self.C)
        return conv(self.w_ll), conv(self.w_lh), conv(self.w_hl), conv(self.w_hh)


class WaveletEncoder(nn.Module):
    """
    Multi-scale wavelet encoder producing frequency conditioning vector f.

    Architecture:
      • 2 scales of learnable wavelet decomposition
      • Global average-pool of all detail sub-bands
      • MLP projection to d_f dimensions

    Args:
        feat_channels : C – input feature channels
        freq_dim      : d_f – output frequency embedding dimension
        num_scales    : number of wavelet decomposition scales (default 2)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        freq_dim: int = 128,
        num_scales: int = 2,
    ):
        super().__init__()
        self.banks = nn.ModuleList([
            LearnableWaveletBank(feat_channels)
            for _ in range(num_scales)
        ])
        # 3 detail sub-bands per scale → 3 × num_scales × C channels after GAP
        agg_dim = 3 * num_scales * feat_channels
        self.proj = nn.Sequential(
            nn.Linear(agg_dim, freq_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(freq_dim * 2, freq_dim),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F : (B, C, H', W')
        Returns:
            f : (B, d_f)   frequency conditioning vector
        """
        x = F
        details = []
        for bank in self.banks:
            ll, lh, hl, hh = bank(x)
            # Collect detail sub-bands; discard approximation (used next scale)
            for sb in (lh, hl, hh):
                # Global average pool over spatial dims → (B, C)
                details.append(sb.mean(dim=[2, 3]))
            x = ll   # pass approximation to next scale

        agg = torch.cat(details, dim=1)   # (B, 3*scales*C)
        f = self.proj(agg)                # (B, d_f)
        return f
