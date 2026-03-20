"""
ablation.py – Reproduce Table 4 (Ablation Study)
==================================================
Runs the cumulative component ablation on MMWHS MRI→CT, training one
configuration per row of Table 4.

Usage:
    python scripts/ablation.py --config configs/mmwhs_mri2ct.yaml --epochs 200
"""

import argparse
import logging
import sys
from pathlib import Path
from copy import deepcopy

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcgd.models.fcgd import FCGD
from fcgd.data.datasets import build_dataloaders
from fcgd.utils.trainer import FCGDTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s')


# Ablation settings: name → which components are active
ABLATION_CONFIGS = [
    # (name,         use_graph, use_struct, use_freq, use_diffusion, use_adv)
    ("Baseline (A)", False,     False,     False,    False,         True),
    ("+G",           True,      False,     False,    False,         True),
    ("+G S",         True,      True,      False,    False,         True),
    ("+G S F",       True,      True,      True,     False,         True),
    ("+G S D (no F)",True,      True,      False,    True,          True),
    ("+G S F D (no A)",True,   True,      True,     True,          False),
    ("FCGD (Full)",  True,      True,      True,     True,          True),
]


class AblatedFCGD(FCGD):
    """
    FCGD variant with optional components disabled for ablation.
    Flags control which loss terms and architectural modules are active.
    """

    def __init__(self, use_graph, use_struct, use_freq, use_diffusion, use_adv, **kwargs):
        super().__init__(**kwargs)
        self.use_graph     = use_graph
        self.use_struct    = use_struct
        self.use_freq      = use_freq
        self.use_diffusion = use_diffusion
        self.use_adv       = use_adv

    def encode(self, x):
        from fcgd.models.graph import AnatomicalGraphConstructor, GNNEncoder
        import torch

        F = self.encoder(x)

        if self.use_graph:
            G = self.graph_builder(F)
            Z = self.gnn_encoder(G['nodes'], G['adj'])
        else:
            # Fallback: flatten feature map to pseudo-node sequence
            B, C, Hp, Wp = F.shape
            Z = F.flatten(2).permute(0, 2, 1)        # (B, N', C)
            # Pad/trim to num_nodes
            N = self.num_nodes
            if Z.shape[1] > N:
                Z = Z[:, :N, :]
            elif Z.shape[1] < N:
                Z = torch.nn.functional.pad(Z, (0, 0, 0, N - Z.shape[1]))

        if self.use_freq:
            f = self.wavelet_encoder(F)
        else:
            f = torch.zeros(x.shape[0], self.wavelet_encoder.proj[-1].out_features,
                            device=x.device)

        return Z, f, F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--output_root', default='outputs/ablation')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['training']['epochs'] = args.epochs

    ds_cfg = cfg['dataset']
    src_loader, tgt_loader = build_dataloaders(
        dataset_name    = ds_cfg['name'],
        root            = ds_cfg['root'],
        source_modality = ds_cfg['source_modality'],
        target_modality = ds_cfg['target_modality'],
        batch_size      = ds_cfg.get('batch_size', 4),
        num_workers     = ds_cfg.get('num_workers', 4),
        img_size        = ds_cfg.get('img_size', 256),
    )

    results = []

    for (name, use_g, use_s, use_f, use_d, use_a) in ABLATION_CONFIGS:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Ablation: {name}")
        logger.info(f"{'─'*60}")

        m_cfg = cfg['model']
        model = AblatedFCGD(
            use_graph     = use_g,
            use_struct    = use_s,
            use_freq      = use_f,
            use_diffusion = use_d,
            use_adv       = use_a,
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
            lambda_str    = 0.0 if not use_s else cfg['training'].get('lambda_str', 0.1),
            mu_adv        = 0.0 if not use_a else cfg['training'].get('mu_adv', 0.01),
        )

        tag = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')
        out_dir = str(Path(args.output_root) / tag)

        train_cfg = {**cfg['training'],
                     'num_classes': m_cfg.get('num_classes', 5),
                     'class_names': ds_cfg.get('class_names'),
                     'mc_samples':  m_cfg.get('mc_samples', 10),
                     'lambda_str':  0.0 if not use_s else cfg['training'].get('lambda_str', 0.1),
                     'mu_adv':      0.0 if not use_a else cfg['training'].get('mu_adv', 0.01)}

        trainer = FCGDTrainer(
            model=model, src_loader=src_loader, tgt_loader=tgt_loader,
            val_loader=src_loader, cfg=train_cfg, output_dir=out_dir,
        )

        if use_d:
            trainer.pretrain_diffusion(n_epochs=5)

        trainer.train()
        dsc = trainer.best_dsc
        results.append((name, dsc))
        logger.info(f"  → Best DSC: {dsc:.2f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 50)
    logger.info("  ABLATION SUMMARY")
    logger.info("═" * 50)
    logger.info(f"  {'Setting':<25} {'Best DSC (%)':>12}")
    logger.info("  " + "─" * 38)
    for name, dsc in results:
        logger.info(f"  {name:<25} {dsc:>12.2f}")


if __name__ == '__main__':
    main()
