"""
train.py – FCGD Training Entry Point
======================================
Usage:
    python scripts/train.py --config configs/mmwhs_mri2ct.yaml
    python scripts/train.py --config configs/chaos_mri2ct.yaml --resume outputs/ckpt_ep050.pth
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcgd.models.fcgd import FCGD
from fcgd.data.datasets import build_dataloaders
from fcgd.utils.trainer import FCGDTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train FCGD')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed from config')
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed or cfg.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    output_dir = cfg.get('output', {}).get('dir', 'outputs/fcgd')
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {cfg.get('experiment', {}).get('name', 'fcgd')}")

    # ── Build DataLoaders ────────────────────────────────────────────────────
    ds_cfg = cfg['dataset']
    logger.info(f"Building dataloaders for {ds_cfg['name']} "
                f"({ds_cfg['source_modality']} → {ds_cfg['target_modality']}) …")
    src_loader, tgt_loader = build_dataloaders(
        dataset_name    = ds_cfg['name'],
        root            = ds_cfg['root'],
        source_modality = ds_cfg['source_modality'],
        target_modality = ds_cfg['target_modality'],
        batch_size      = ds_cfg.get('batch_size', 4),
        num_workers     = ds_cfg.get('num_workers', 4),
        img_size        = ds_cfg.get('img_size', 256),
    )
    logger.info(f"Source batches: {len(src_loader)}  |  Target batches: {len(tgt_loader)}")

    # ── Build Model ──────────────────────────────────────────────────────────
    m_cfg = cfg['model']
    logger.info("Instantiating FCGD model …")
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
        mc_samples    = m_cfg.get('mc_samples', 10),
        lambda_str    = cfg['training'].get('lambda_str', 0.1),
        mu_adv        = cfg['training'].get('mu_adv', 0.01),
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    train_cfg = {**cfg['training'],
                 'num_classes':  m_cfg.get('num_classes', 5),
                 'class_names':  ds_cfg.get('class_names', None),
                 'mc_samples':   m_cfg.get('mc_samples', 10)}

    trainer = FCGDTrainer(
        model      = model,
        src_loader = src_loader,
        tgt_loader = tgt_loader,
        val_loader = src_loader,   # use source as proxy val; swap with held-out target
        cfg        = train_cfg,
        output_dir = output_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Pre-train diffusion ───────────────────────────────────────────────────
    pretrain_epochs = cfg['training'].get('pretrain_diff_epochs', 10)
    if pretrain_epochs > 0 and not args.resume:
        trainer.pretrain_diffusion(n_epochs=pretrain_epochs)

    # ── Main training ─────────────────────────────────────────────────────────
    trainer.train()
    logger.info("Training complete.")


if __name__ == '__main__':
    main()
