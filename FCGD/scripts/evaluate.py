"""
evaluate.py – FCGD Evaluation Script
======================================
Runs inference on a test set and reports per-structure DSC and ASD,
matching Table 1/2/3 of the paper.

Usage:
    python scripts/evaluate.py \
        --config  configs/mmwhs_mri2ct.yaml \
        --ckpt    outputs/fcgd_mmwhs_mri2ct/ckpt_best.pth \
        --split   test \
        --mc      10 \
        --save_vis outputs/vis
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcgd.models.fcgd import FCGD
from fcgd.data.datasets import build_dataloaders
from fcgd.utils.metrics import SegmentationEvaluator
from fcgd.utils.visualise import visualise_prediction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Evaluate FCGD')
    parser.add_argument('--config',   required=True,  help='YAML config path')
    parser.add_argument('--ckpt',     required=True,  help='Checkpoint path')
    parser.add_argument('--split',    default='test', help='Data split to evaluate')
    parser.add_argument('--mc',       type=int, default=10,
                        help='Monte-Carlo inference samples M')
    parser.add_argument('--save_vis', default=None,
                        help='Directory to save visualisation images')
    parser.add_argument('--max_vis',  type=int, default=20,
                        help='Maximum number of visualisations to save')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    m_cfg = cfg['model']
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
    logger.info(f"Loaded checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch', '?')})")

    # ── DataLoader ────────────────────────────────────────────────────────────
    ds_cfg = cfg['dataset']
    _, tgt_loader = build_dataloaders(
        dataset_name    = ds_cfg['name'],
        root            = ds_cfg['root'],
        source_modality = ds_cfg['source_modality'],
        target_modality = ds_cfg['target_modality'],
        batch_size      = 1,
        num_workers     = 2,
        img_size        = ds_cfg.get('img_size', 256),
    )

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = SegmentationEvaluator(
        num_classes = m_cfg.get('num_classes', 5),
        class_names = ds_cfg.get('class_names', None),
    )

    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    vis_count = 0
    logger.info("Running inference …")

    with torch.no_grad():
        for i, batch in enumerate(tgt_loader):
            imgs   = batch[0].to(device)
            labels = batch[1].cpu() if len(batch) > 1 else None

            y_bar, U = model.predict(imgs, mc_samples=args.mc)
            pred = y_bar.argmax(dim=1).cpu()      # (B, H, W)

            if labels is not None:
                evaluator.update(pred, labels)

            # ── Save visualisations ──────────────────────────────────────────
            if args.save_vis and vis_count < args.max_vis and labels is not None:
                img_np   = imgs[0, 0].cpu().numpy()
                lbl_np   = labels[0].numpy()
                pred_np  = pred[0].numpy()
                unc_np   = U[0].cpu().numpy()

                # Normalise image to [0,1] for display
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                save_p = str(Path(args.save_vis) / f'sample_{i:04d}.png')
                visualise_prediction(
                    image=img_np, gt_label=lbl_np,
                    pred_label=pred_np, uncertainty=unc_np,
                    save_path=save_p,
                    title=f'Sample {i} | MC={args.mc}',
                )
                vis_count += 1

    # ── Print final results ───────────────────────────────────────────────────
    logger.info("\n" + "═" * 55)
    logger.info(f"  FCGD Evaluation: {ds_cfg['name'].upper()}  "
                f"{ds_cfg['source_modality']} → {ds_cfg['target_modality']}")
    logger.info("═" * 55)
    results = evaluator.compute()
    evaluator.print_results(results)

    mean = results['Mean']
    logger.info(f"\n  Mean DSC : {mean['DSC_mean']:.2f} ± {mean['DSC_std']:.2f} %")
    logger.info(f"  Mean ASD : {mean['ASD_mean']:.2f} ± {mean['ASD_std']:.2f} mm")
    if args.save_vis:
        logger.info(f"\n  Saved {vis_count} visualisation images → {args.save_vis}")


if __name__ == '__main__':
    main()
