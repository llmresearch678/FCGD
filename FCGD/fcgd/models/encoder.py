"""
Convolutional Encoder (Ec)
==========================
ResNet-style or plain CNN backbone producing dense feature maps F ∈ R^{C×H'×W'}.
Skip-connection features are cached for the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ConvBlock(nn.Module):
    """Conv → BN → ReLU (×2) residual block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class ConvEncoder(nn.Module):
    """
    Hierarchical convolutional encoder.

    For an input of shape (B, in_channels, H, W) and channels=(32,64,128,256),
    the output feature map F has shape (B, 256, H/16, W/16).
    Skip features from each stage are cached and returned by get_skip_features().

    Args:
        in_channels : number of image input channels
        channels    : tuple of output channels per stage (each stage halves H, W)
    """

    def __init__(self, in_channels: int = 1, channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        stages = []
        prev = in_channels
        for ch in channels:
            stages.append(
                nn.Sequential(
                    ConvBlock(prev, ch),
                    nn.MaxPool2d(2, 2),
                )
            )
            prev = ch
        self.stages = nn.ModuleList(stages)
        self._skip_feats: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, H, W)
        Returns:
            F : (B, channels[-1], H/2^n, W/2^n)
        """
        self._skip_feats = []
        for stage in self.stages:
            # Store pre-pool feature for skip connection
            feat = stage[0](x)          # ConvBlock output
            self._skip_feats.append(feat)
            x = stage[1](feat)          # MaxPool
        return x

    def get_skip_features(self) -> List[torch.Tensor]:
        """Return cached skip-connection features (list from shallow→deep)."""
        return list(self._skip_feats)
