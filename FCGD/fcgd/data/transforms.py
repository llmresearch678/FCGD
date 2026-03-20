"""
Data Augmentation Transforms
=============================
Domain-robust augmentation pipeline for medical image slices.
All transforms operate on single-channel 2D tensors (1, H, W).
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math
from typing import Tuple, Optional


class RandomFlip:
    """Random horizontal and/or vertical flip."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if random.random() < self.p:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() < self.p:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        return (img, mask) if mask is not None else img


class RandomRotate:
    """Random rotation within ±angle degrees."""
    def __init__(self, angle: float = 15.0, p: float = 0.5):
        self.angle = angle
        self.p = p

    def __call__(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if random.random() < self.p:
            deg = random.uniform(-self.angle, self.angle)
            img = TF.rotate(img, deg, interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.rotate(
                    mask.unsqueeze(0).float(), deg,
                    interpolation=TF.InterpolationMode.NEAREST
                ).squeeze(0).long()
        return (img, mask) if mask is not None else img


class RandomElasticDeformation:
    """
    Lightweight random elastic deformation via grid perturbation.
    Suitable for simulating anatomical shape variation.
    """
    def __init__(self, alpha: float = 20.0, sigma: float = 5.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def _gaussian_smooth(self, t: torch.Tensor, sigma: float) -> torch.Tensor:
        k = max(3, int(4 * sigma) | 1)   # odd kernel size
        kernel = torch.arange(k, dtype=torch.float32) - k // 2
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,N)
        t = F.conv1d(t, kernel.view(1, 1, -1), padding=k // 2)
        return t.squeeze()

    def __call__(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if random.random() > self.p:
            return (img, mask) if mask is not None else img

        _, H, W = img.shape
        dx = self._gaussian_smooth(
            torch.rand(H, W) * 2 - 1, self.sigma) * self.alpha
        dy = self._gaussian_smooth(
            torch.rand(H, W) * 2 - 1, self.sigma) * self.alpha

        # Build sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid_x = (grid_x + dx / W).clamp(-1, 1)
        grid_y = (grid_y + dy / H).clamp(-1, 1)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

        out_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True,
                                 mode='bilinear', padding_mode='border').squeeze(0)
        if mask is not None:
            out_msk = F.grid_sample(
                mask.unsqueeze(0).unsqueeze(0).float(), grid,
                align_corners=True, mode='nearest', padding_mode='border'
            ).squeeze().long()
            return out_img, out_msk
        return out_img


class RandomGammaCorrection:
    """Simulate scanner-specific gamma/intensity differences."""
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.4):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, img: torch.Tensor):
        if random.random() < self.p:
            gamma = random.uniform(self.lo, self.hi)
            img = img.clamp(0).pow(gamma)
        return img


class GaussianNoise:
    """Additive Gaussian noise to simulate MRI/CT acquisition noise."""
    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor):
        if random.random() < self.p:
            img = img + torch.randn_like(img) * self.std
        return img


class MedicalImageAugmentation:
    """
    Full training augmentation pipeline for medical images.
    Applied only to source-domain labelled data.
    """

    def __init__(self):
        self.flip    = RandomFlip(p=0.5)
        self.rotate  = RandomRotate(angle=15, p=0.5)
        self.elastic = RandomElasticDeformation(alpha=20, sigma=5, p=0.3)
        self.gamma   = RandomGammaCorrection(gamma_range=(0.7, 1.5), p=0.4)
        self.noise   = GaussianNoise(std=0.02, p=0.3)

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        img, mask = self.flip(img, mask)
        img, mask = self.rotate(img, mask)
        img, mask = self.elastic(img, mask)
        img = self.gamma(img)
        img = self.noise(img)
        if mask is not None:
            return img, mask
        return img
