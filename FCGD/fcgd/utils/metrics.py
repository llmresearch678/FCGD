"""
Evaluation Metrics
==================
Dice Similarity Coefficient (DSC) and Average Symmetric Surface Distance (ASD)
as used throughout the FCGD paper.

  DSC (↑, %)  : volumetric overlap measure
  ASD (↓, mm) : boundary localisation error

Both metrics are computed per structure and averaged.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import binary_erosion, label as scipy_label
from scipy.spatial.distance import directed_hausdorff


# ─────────────────────────────────────────────────────────────────────────────
# Per-class Dice Similarity Coefficient
# ─────────────────────────────────────────────────────────────────────────────

def dice_per_class(
    pred:   torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Compute Dice score for each class.

    Args:
        pred        : predicted class map (B, H, W) or (H, W), integer labels
        target      : ground-truth labels (same shape)
        num_classes : K (including background class 0)
        smooth      : ε for numerical stability

    Returns:
        dices : (K,) Dice score per class (background included at index 0)
    """
    pred   = pred.view(-1).long()
    target = target.view(-1).long()

    dices = []
    for cls in range(num_classes):
        p = (pred   == cls).float()
        t = (target == cls).float()
        inter = (p * t).sum()
        dices.append((2 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return torch.stack(dices)


def mean_dice(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    num_classes: int,
    ignore_bg:   bool = True,
) -> float:
    """
    Mean Dice over foreground classes (ignores class 0 if ignore_bg=True).

    Returns a float in [0, 1].
    """
    dices = dice_per_class(pred, target, num_classes)
    if ignore_bg:
        dices = dices[1:]
    return dices.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Average Symmetric Surface Distance
# ─────────────────────────────────────────────────────────────────────────────

def _surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Return coordinates of surface voxels (eroded boundary)."""
    eroded  = binary_erosion(binary_mask)
    surface = binary_mask ^ eroded
    pts     = np.argwhere(surface)
    return pts


def asd_per_class(
    pred:        np.ndarray,
    target:      np.ndarray,
    num_classes: int,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """
    Compute Average Symmetric Surface Distance (mm) per class.

    Args:
        pred          : (H, W) integer label map (numpy)
        target        : (H, W) ground-truth label map (numpy)
        num_classes   : K
        voxel_spacing : physical spacing (mm per pixel), (sy, sx)

    Returns:
        asds : (K,) ASD per class (class 0 = background, usually skipped)
    """
    sx, sy = voxel_spacing
    asds = np.zeros(num_classes)

    for cls in range(1, num_classes):  # skip background
        p_bin = (pred   == cls).astype(bool)
        t_bin = (target == cls).astype(bool)

        if p_bin.sum() == 0 and t_bin.sum() == 0:
            asds[cls] = 0.0
            continue
        if p_bin.sum() == 0 or t_bin.sum() == 0:
            asds[cls] = np.nan  # will be reported as N/A
            continue

        p_pts = _surface_points(p_bin).astype(float)
        t_pts = _surface_points(t_bin).astype(float)

        # Scale coordinates by voxel spacing
        p_pts[:, 0] *= sy
        p_pts[:, 1] *= sx
        t_pts[:, 0] *= sy
        t_pts[:, 1] *= sx

        d_pt = directed_hausdorff(p_pts, t_pts)[0]
        d_tp = directed_hausdorff(t_pts, p_pts)[0]
        asds[cls] = (d_pt + d_tp) / 2.0

    return asds


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated evaluation over a full test set
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationEvaluator:
    """
    Accumulates per-slice predictions and computes final DSC and ASD.

    Usage:
        evaluator = SegmentationEvaluator(num_classes=5, class_names=['MYO','LAC','LVC','AA'])
        for pred, target in test_loader:
            evaluator.update(pred, target)
        results = evaluator.compute()
        evaluator.print_results(results)
    """

    def __init__(
        self,
        num_classes:   int,
        class_names:   Optional[List[str]] = None,
        ignore_bg:     bool = True,
        voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    ):
        self.K             = num_classes
        self.names         = class_names or [f'cls_{k}' for k in range(1, num_classes)]
        self.ignore_bg     = ignore_bg
        self.spacing       = voxel_spacing
        self._dices: List  = []
        self._asds:  List  = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred   : (B, H, W) predicted integer labels
            target : (B, H, W) ground-truth labels
        """
        pred   = pred.cpu()
        target = target.cpu()
        for b in range(pred.shape[0]):
            d = dice_per_class(pred[b], target[b], self.K)
            self._dices.append(d.numpy())
            a = asd_per_class(pred[b].numpy(), target[b].numpy(), self.K, self.spacing)
            self._asds.append(a)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Return per-class and mean DSC (%) and ASD (mm)."""
        all_d = np.stack(self._dices)           # (N_slices, K)
        all_a = np.stack(self._asds)            # (N_slices, K)

        results = {}
        for i, name in enumerate(self.names, start=1):
            d_vals = all_d[:, i]
            a_vals = all_a[:, i]
            results[name] = {
                'DSC_mean': float(d_vals.mean() * 100),
                'DSC_std':  float(d_vals.std() * 100),
                'ASD_mean': float(np.nanmean(a_vals)),
                'ASD_std':  float(np.nanstd(a_vals)),
            }

        # Overall mean (ignoring background)
        fg_d = all_d[:, 1:].mean(axis=1)
        fg_a = np.nanmean(all_a[:, 1:], axis=1)
        results['Mean'] = {
            'DSC_mean': float(fg_d.mean() * 100),
            'DSC_std':  float(fg_d.std() * 100),
            'ASD_mean': float(np.nanmean(fg_a)),
            'ASD_std':  float(np.nanstd(fg_a)),
        }
        return results

    def print_results(self, results: Optional[Dict] = None):
        if results is None:
            results = self.compute()
        print(f"\n{'Structure':<12} {'DSC (%)':<18} {'ASD (mm)':<18}")
        print("─" * 50)
        for name, vals in results.items():
            d_str = f"{vals['DSC_mean']:.2f} ± {vals['DSC_std']:.2f}"
            a_str = f"{vals['ASD_mean']:.2f} ± {vals['ASD_std']:.2f}"
            print(f"{name:<12} {d_str:<18} {a_str:<18}")

    def reset(self):
        self._dices.clear()
        self._asds.clear()
