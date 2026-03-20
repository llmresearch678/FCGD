"""
infer.py – Single-Image Inference with Uncertainty Estimation
==============================================================
Runs FCGD on a single 2D medical image slice and saves:
  • Predicted segmentation overlay
  • Pixel-wise uncertainty heatmap
  • Class probability maps

Usage:
    python scripts/infer.py \
        --ckpt  outputs/fcgd_mmwhs_mri2ct/ckpt_best.pth \
        --image path/to/slice.png \
        --config configs/mmwhs_mri2ct.yaml \
        --out   outputs/inference \
        --mc    10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcgd.models.fcgd import FCGD
from fcgd.utils.visualise import visualise_prediction, MMWHS_COLOURS, label_to_rgb


def load_image(path: str, img_size: int = 256) -> torch.Tensor:
    """Load a grayscale image → (1,1,H,W) normalised tensor."""
    img = np.array(Image.open(path).convert('L').resize((img_size, img_size)),
                   dtype=np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


def main():
    parser = argparse.ArgumentParser(description='FCGD single-image inference')
    parser.add_argument('--ckpt',   required=True)
    parser.add_argument('--image',  required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out',    default='outputs/inference')
    parser.add_argument('--mc',     type=int, default=10,
                        help='Monte-Carlo samples for uncertainty estimation')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Uncertainty threshold τ_U for review flagging')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg['model']
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    model = FCGD(
        in_channels   = m_cfg.get('in_channels', 1),
        num_classes   = m_cfg.get('num_classes', 5),
        img_size      = m_cfg.get('img_size', 256),
        enc_channels  = tuple(m_cfg.get('enc_channels', [32, 64, 128, 256])),
        num_nodes     = m_cfg.get('num_nodes', 256),
        gnn_layers    = m_cfg.get('gnn_layers', 3),
        gnn_hidden_dim= m_cfg.get('gnn_hidden_dim', 256),
        freq_dim      = m_cfg.get('freq_dim', 128),
        diff_timesteps= m_cfg.get('diff_timesteps', 10),
        mc_samples    = args.mc,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"✓ Model loaded from {args.ckpt}")

    # ── Load image ────────────────────────────────────────────────────────────
    img_t = load_image(args.image, m_cfg.get('img_size', 256)).to(device)
    print(f"✓ Image loaded: {args.image}  shape={img_t.shape}")

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        y_bar, U = model.predict(img_t, mc_samples=args.mc)

    pred  = y_bar.argmax(dim=1)[0].cpu().numpy()    # (H, W)
    probs = y_bar[0].cpu().numpy()                   # (K, H, W)
    unc   = U[0].cpu().numpy()                       # (H, W)
    img_np = img_t[0, 0].cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # ── 1. Segmentation + uncertainty overlay ─────────────────────────────────
    fig = visualise_prediction(
        image=img_np, gt_label=pred, pred_label=pred,
        uncertainty=unc,
        save_path=str(out_dir / 'prediction.png'),
        title=f'FCGD Prediction  (MC={args.mc})',
    )
    plt.close(fig)
    print(f"✓ Saved prediction → {out_dir / 'prediction.png'}")

    # ── 2. Uncertainty review mask ────────────────────────────────────────────
    review_mask = (unc > args.threshold).astype(np.uint8) * 255
    Image.fromarray(review_mask).save(str(out_dir / 'review_mask.png'))
    pct = 100 * (unc > args.threshold).mean()
    print(f"✓ Review mask saved  ({pct:.1f}% pixels flagged, τ={args.threshold})")

    # ── 3. Per-class probability maps ─────────────────────────────────────────
    fig2, axes = plt.subplots(1, probs.shape[0], figsize=(3.5 * probs.shape[0], 3.5))
    class_names = cfg['dataset'].get('class_names', [f'cls_{k}' for k in range(probs.shape[0])])
    for k, ax in enumerate(axes):
        im = ax.imshow(probs[k], cmap='viridis', vmin=0, vmax=1)
        name = 'BG' if k == 0 else (class_names[k-1] if k-1 < len(class_names) else f'cls_{k}')
        ax.set_title(name, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Per-class Probability Maps', fontsize=11)
    plt.tight_layout()
    fig2.savefig(str(out_dir / 'prob_maps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Probability maps saved → {out_dir / 'prob_maps.png'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Max uncertainty : {unc.max():.4f}")
    print(f"  Mean uncertainty: {unc.mean():.4f}")
    print(f"  Predicted classes present: {sorted(np.unique(pred).tolist())}")
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
