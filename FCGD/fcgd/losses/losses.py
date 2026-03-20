"""
Loss Functions
==============
Implements all training objectives from Eq. 16–22:

  L_total = L_sup + λ · L_str + µ · L_G

  L_sup   : supervised segmentation loss (CE + Dice)         [Eq. 16-18]
  L_str   : structural consistency regularisation            [Eq. 19]
  L_D     : discriminator binary cross-entropy               [Eq. 20]
  L_G     : generator (fool-discriminator) loss              [Eq. 21]
  L_diff  : diffusion denoising score-matching               [Eq. 13]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-wise cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Pixel-wise cross-entropy  (Eq. 17).

    Args:
        pred   : logits (B, K, H, W)
        target : one-hot or integer labels (B, H, W) or (B, K, H, W)
    """
    if target.dim() == 4:                      # one-hot → integer labels
        target = target.argmax(dim=1)
    return F.cross_entropy(pred, target.long())


# ─────────────────────────────────────────────────────────────────────────────
# Soft Dice loss
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(
    pred:    torch.Tensor,
    target:  torch.Tensor,
    smooth:  float = 1e-5,
) -> torch.Tensor:
    """
    Soft Dice loss  (Eq. 18).

    Args:
        pred   : logits (B, K, H, W)
        target : one-hot (B, K, H, W) or integer (B, H, W)
        smooth : ε smoothing constant
    """
    K = pred.shape[1]
    prob = torch.softmax(pred, dim=1)           # (B, K, H, W)

    if target.dim() == 3:                       # integer → one-hot
        target = F.one_hot(target.long(), K).permute(0, 3, 1, 2).float()

    # Flatten spatial dims
    prob_f   = prob.view(prob.shape[0], K, -1)     # (B, K, HW)
    target_f = target.view(target.shape[0], K, -1) # (B, K, HW)

    intersection = (prob_f * target_f).sum(dim=-1)         # (B, K)
    union        = prob_f.sum(dim=-1) + target_f.sum(dim=-1)  # (B, K)
    dice_per_cls = (2 * intersection + smooth) / (union + smooth)  # (B, K)
    return 1 - dice_per_cls.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Supervised segmentation loss  L_sup = L_CE + L_Dice
# ─────────────────────────────────────────────────────────────────────────────

def supervised_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Eq. 16: L_sup = L_CE + L_Dice."""
    return cross_entropy_loss(pred, target) + dice_loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Structural consistency regularisation  L_str
# ─────────────────────────────────────────────────────────────────────────────

def structural_consistency_loss(
    Z_s: torch.Tensor,
    Z_t: torch.Tensor,
) -> torch.Tensor:
    """
    Eq. 19: L_str = || E[Z_s] - E[Z_t] ||_F^2

    Aligns first-order statistics of source and target embeddings in GNN space.

    Args:
        Z_s : source GNN embeddings  (B, N, d_L)
        Z_t : target GNN embeddings  (B, N, d_L)
    """
    mean_s = Z_s.mean(dim=0)   # (N, d_L)
    mean_t = Z_t.mean(dim=0)   # (N, d_L)
    return (mean_s - mean_t).pow(2).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial domain alignment losses
# ─────────────────────────────────────────────────────────────────────────────

def discriminator_loss(
    d_source: torch.Tensor,
    d_target: torch.Tensor,
) -> torch.Tensor:
    """
    Eq. 20: L_D = -E[log D(Z_s)] - E[log(1 - D(Z_t))]

    Args:
        d_source : discriminator output for source  (B, 1)
        d_target : discriminator output for target  (B, 1)
    """
    loss_s = F.binary_cross_entropy(d_source, torch.ones_like(d_source))
    loss_t = F.binary_cross_entropy(d_target, torch.zeros_like(d_target))
    return (loss_s + loss_t) * 0.5


def generator_loss(d_target: torch.Tensor) -> torch.Tensor:
    """
    Eq. 21: L_G = -E[log D(Z_t)]  (fool the discriminator)

    Args:
        d_target : discriminator output for target  (B, 1)
    """
    return F.binary_cross_entropy(d_target, torch.ones_like(d_target))


# ─────────────────────────────────────────────────────────────────────────────
# Combined training loss  L_total
# ─────────────────────────────────────────────────────────────────────────────

class FCGDLoss(nn.Module):
    """
    Composite loss for the FCGD segmentation network.

    L_total = L_sup + λ · L_str + µ · L_G   (Eq. 22)

    Args:
        lambda_str : λ – structural consistency weight (default 0.1)
        mu_adv     : µ – adversarial alignment weight  (default 0.01)
    """

    def __init__(self, lambda_str: float = 0.1, mu_adv: float = 0.01):
        super().__init__()
        self.lam = lambda_str
        self.mu  = mu_adv

    def segmentation_network_loss(
        self,
        pred_s:   torch.Tensor,
        label_s:  torch.Tensor,
        Z_s:      torch.Tensor,
        Z_t:      torch.Tensor,
        d_t:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Full generator-side loss.

        Args:
            pred_s  : source segmentation logits  (B, K, H, W)
            label_s : source ground-truth labels  (B, H, W) or (B, K, H, W)
            Z_s     : source GNN embeddings       (B, N, d_L)
            Z_t     : target GNN embeddings       (B, N, d_L)
            d_t     : discriminator score target  (B, 1)
        """
        l_sup = supervised_loss(pred_s, label_s)
        l_str = structural_consistency_loss(Z_s, Z_t)
        l_g   = generator_loss(d_t)
        return l_sup + self.lam * l_str + self.mu * l_g

    def forward(
        self,
        pred_s:   torch.Tensor,
        label_s:  torch.Tensor,
        Z_s:      torch.Tensor,
        Z_t:      torch.Tensor,
        d_t:      torch.Tensor,
    ) -> dict:
        """Returns dict of individual and total loss values."""
        l_sup = supervised_loss(pred_s, label_s)
        l_str = structural_consistency_loss(Z_s, Z_t)
        l_g   = generator_loss(d_t)
        total = l_sup + self.lam * l_str + self.mu * l_g

        return {
            'total':    total,
            'sup':      l_sup,
            'str':      l_str,
            'gen':      l_g,
        }
