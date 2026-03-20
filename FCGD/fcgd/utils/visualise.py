"""
Visualisation Utilities
========================
Tools for plotting segmentation predictions, uncertainty maps,
and uncertainty calibration curves (ECE, AURC).
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Dict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes matching the paper (Figure 2)
# ─────────────────────────────────────────────────────────────────────────────

MMWHS_COLOURS = {
    0: (0.00, 0.00, 0.00),   # background – black
    1: (0.80, 0.10, 0.10),   # MYO – red
    2: (0.10, 0.70, 0.10),   # LAC – green
    3: (1.00, 0.85, 0.00),   # LVC – yellow
    4: (0.10, 0.30, 0.85),   # AA  – blue
}

CHAOS_COLOURS = {
    0: (0.00, 0.00, 0.00),   # background
    1: (0.95, 0.60, 0.10),   # liver
    2: (0.10, 0.70, 0.80),   # right kidney
    3: (0.80, 0.10, 0.70),   # left kidney
    4: (0.20, 0.80, 0.20),   # spleen
}


def label_to_rgb(label_map: np.ndarray, palette: Dict[int, Tuple]) -> np.ndarray:
    """Convert integer label map (H, W) → RGB (H, W, 3)."""
    H, W = label_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls, colour in palette.items():
        mask = label_map == cls
        rgb[mask] = colour
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Single sample visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_prediction(
    image:      np.ndarray,
    gt_label:   np.ndarray,
    pred_label: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    palette:    Optional[Dict] = None,
    save_path:  Optional[str]  = None,
    title:      str = '',
) -> plt.Figure:
    """
    Plot: image | GT | prediction | (uncertainty map).

    Args:
        image       : (H, W) grayscale image, normalised [0,1]
        gt_label    : (H, W) ground-truth integer labels
        pred_label  : (H, W) predicted integer labels
        uncertainty : (H, W) per-pixel uncertainty U (optional)
        palette     : colour dict {class_int: (R,G,B)}
        save_path   : if given, save figure to this path
        title       : figure title

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = MMWHS_COLOURS

    n_cols = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input Image')

    axes[1].imshow(label_to_rgb(gt_label,   palette))
    axes[1].set_title('Ground Truth')

    axes[2].imshow(label_to_rgb(pred_label, palette))
    axes[2].set_title('FCGD Prediction')

    if uncertainty is not None:
        im = axes[3].imshow(uncertainty, cmap='hot', vmin=0)
        axes[3].set_title('Uncertainty Map U')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-method comparison (mirrors Figure 2 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_comparison(
    image:          np.ndarray,
    gt_label:       np.ndarray,
    method_preds:   Dict[str, np.ndarray],
    palette:        Optional[Dict] = None,
    save_path:      Optional[str]  = None,
) -> plt.Figure:
    """
    Render a qualitative comparison row: image | GT | method_1 | … | method_N

    Args:
        method_preds : ordered dict {method_name: pred_label_map}
    """
    if palette is None:
        palette = MMWHS_COLOURS

    methods = list(method_preds.items())
    n_cols  = 2 + len(methods)   # image + GT + N methods

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Image', fontsize=9)

    axes[1].imshow(label_to_rgb(gt_label, palette))
    axes[1].set_title('GT', fontsize=9)

    for i, (name, pred) in enumerate(methods, start=2):
        axes[i].imshow(label_to_rgb(pred, palette))
        axes[i].set_title(name, fontsize=8)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Calibration curves (ECE & AURC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    n_bins:      int = 15,
) -> float:
    """
    Expected Calibration Error over n_bins confidence bins.

    Args:
        confidences : per-pixel max softmax probability   (N,)
        accuracies  : per-pixel correctness (0/1)         (N,)
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    N        = len(confidences)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = accuracies[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    return float(ece)


def plot_calibration_curve(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    n_bins:      int = 15,
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """Reliability diagram for calibration assessment."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confs, bin_accs = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_confs.append(confidences[mask].mean())
        bin_accs.append(accuracies[mask].mean())

    ece = compute_ece(confidences, accuracies, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.bar(bin_confs, bin_accs, width=1 / n_bins, alpha=0.6,
           color='steelblue', edgecolor='navy', label='FCGD')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram  (ECE = {ece:.3f})')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
