"""
FCGD Training Engine
=====================
Implements the alternating optimisation described in Section "Training Objectives":
  1. Pre-train diffusion model on source embeddings  (L_diff)
  2. Joint training loop:
       a. Update discriminator  (L_D)
       b. Update segmentation network  (L_total = L_sup + λ·L_str + µ·L_G)
       c. Optionally fine-tune diffusion end-to-end
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from fcgd.models.fcgd import FCGD
from fcgd.losses.losses import FCGDLoss, discriminator_loss
from fcgd.utils.metrics import SegmentationEvaluator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient reversal (GRL) – not strictly required but useful for stabilisation
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x

    @staticmethod
    def backward(ctx, grad):
        alpha, = ctx.saved_tensors
        return -alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FCGDTrainer:
    """
    Full FCGD training coordinator.

    Args:
        model          : FCGD model instance
        src_loader     : labelled source DataLoader
        tgt_loader     : unlabelled target DataLoader
        val_loader     : validation DataLoader (optional)
        cfg            : configuration dict (see configs/)
        output_dir     : directory for checkpoints and logs
    """

    def __init__(
        self,
        model:      FCGD,
        src_loader: DataLoader,
        tgt_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg:        Dict[str, Any],
        output_dir: str = 'outputs',
    ):
        self.model      = model
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.val_loader = val_loader
        self.cfg        = cfg
        self.out_dir    = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.model.to(self.device)

        # ── Optimisers ──────────────────────────────────────────────────────
        seg_params  = (
            list(model.encoder.parameters())
            + list(model.graph_builder.parameters())
            + list(model.gnn_encoder.parameters())
            + list(model.wavelet_encoder.parameters())
            + list(model.decoder.parameters())
        )
        self.opt_seg  = torch.optim.Adam(
            seg_params,
            lr=cfg.get('lr_seg', 1e-4),
            weight_decay=cfg.get('wd', 1e-5),
        )
        self.opt_disc = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=cfg.get('lr_disc', 1e-4),
            betas=(0.5, 0.999),
        )
        self.opt_diff = torch.optim.Adam(
            model.diffusion.parameters(),
            lr=cfg.get('lr_diff', 5e-5),
        )

        # ── Schedulers ───────────────────────────────────────────────────────
        total_steps = cfg.get('epochs', 200) * min(len(src_loader), len(tgt_loader))
        self.sched_seg = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_seg, T_max=total_steps, eta_min=1e-6)

        # ── Loss function ────────────────────────────────────────────────────
        self.criterion = FCGDLoss(
            lambda_str=cfg.get('lambda_str', 0.1),
            mu_adv=cfg.get('mu_adv', 0.01),
        )

        # ── Mixed precision ──────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=cfg.get('amp', True))

        # ── Metrics ──────────────────────────────────────────────────────────
        self.evaluator = SegmentationEvaluator(
            num_classes=cfg.get('num_classes', 5),
            class_names=cfg.get('class_names', None),
        )

        self.best_dsc  = 0.0
        self.global_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-training: diffusion on source embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def pretrain_diffusion(self, n_epochs: int = 10):
        """Pre-train the latent diffusion model on source-domain embeddings."""
        logger.info(f"Pre-training diffusion for {n_epochs} epochs …")
        self.model.train()

        for ep in range(n_epochs):
            ep_loss = 0.0
            for batch in self.src_loader:
                imgs = batch[0].to(self.device)
                with autocast(enabled=self.cfg.get('amp', True)):
                    Z, f, _ = self.model.encode(imgs)
                    loss = self.model.diffusion.compute_loss(Z, f)

                self.opt_diff.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt_diff)
                self.scaler.update()
                ep_loss += loss.item()

            logger.info(f"  Diffusion pre-train epoch {ep+1}/{n_epochs}  "
                        f"loss={ep_loss/len(self.src_loader):.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        epochs     = self.cfg.get('epochs', 200)
        val_every  = self.cfg.get('val_every', 5)
        save_every = self.cfg.get('save_every', 10)

        logger.info(f"Starting FCGD training for {epochs} epochs …")
        tgt_iter = iter(self.tgt_loader)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_metrics = {k: 0.0 for k in ('loss_total', 'loss_sup',
                                               'loss_str', 'loss_disc')}
            t0 = time.time()

            for batch_s in self.src_loader:
                # ── Load one batch from each domain ──────────────────────────
                try:
                    batch_t = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(self.tgt_loader)
                    batch_t  = next(tgt_iter)

                xs, ys = batch_s[0].to(self.device), batch_s[1].to(self.device)
                xt     = batch_t[0].to(self.device)

                # ── Step 1: Update discriminator ─────────────────────────────
                self.opt_disc.zero_grad()
                with autocast(enabled=self.cfg.get('amp', True)):
                    fwd = self.model(xs, xt)
                    loss_D = discriminator_loss(
                        fwd['d_s'].detach(),
                        fwd['d_t'].detach(),
                    )
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.opt_disc)
                self.scaler.update()

                # ── Step 2: Update segmentation network ──────────────────────
                self.opt_seg.zero_grad()
                with autocast(enabled=self.cfg.get('amp', True)):
                    fwd = self.model(xs, xt)
                    losses = self.criterion(
                        pred_s=fwd['pred_s'],
                        label_s=ys,
                        Z_s=fwd['Z_s'],
                        Z_t=fwd['Z_t'],
                        d_t=fwd['d_t'],
                    )
                self.scaler.scale(losses['total']).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.get('grad_clip', 1.0))
                self.scaler.step(self.opt_seg)
                self.scaler.update()
                self.sched_seg.step()

                # ── (Optional) fine-tune diffusion ───────────────────────────
                if self.cfg.get('finetune_diff', False):
                    self.opt_diff.zero_grad()
                    with autocast(enabled=self.cfg.get('amp', True)):
                        Z, f, _ = self.model.encode(xs)
                        diff_loss = self.model.diffusion.compute_loss(Z, f)
                    self.scaler.scale(diff_loss).backward()
                    self.scaler.step(self.opt_diff)
                    self.scaler.update()

                for k in epoch_metrics:
                    epoch_metrics[k] += losses.get(k.split('_', 1)[-1], 0.0).item() \
                        if isinstance(losses.get(k.split('_', 1)[-1], 0.0), torch.Tensor) \
                        else losses.get(k.split('_', 1)[-1], 0.0)
                epoch_metrics['loss_disc'] += loss_D.item()
                self.global_step += 1

            n_batches = len(self.src_loader)
            elapsed   = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"total={epoch_metrics['loss_total']/n_batches:.4f} "
                f"sup={epoch_metrics['loss_sup']/n_batches:.4f} "
                f"str={epoch_metrics['loss_str']/n_batches:.4f} "
                f"disc={epoch_metrics['loss_disc']/n_batches:.4f} "
                f"| {elapsed:.1f}s"
            )

            # ── Validation ───────────────────────────────────────────────────
            if self.val_loader and epoch % val_every == 0:
                dsc = self.validate()
                if dsc > self.best_dsc:
                    self.best_dsc = dsc
                    self.save_checkpoint(epoch, tag='best')
                    logger.info(f"  ↑ New best DSC: {dsc:.2f}%")

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        self.evaluator.reset()
        num_classes = self.cfg.get('num_classes', 5)

        for batch in self.val_loader:
            imgs, lbls = batch[0].to(self.device), batch[1]
            y_bar, _ = self.model.predict(imgs, mc_samples=self.cfg.get('mc_samples', 10))
            pred = y_bar.argmax(dim=1).cpu()   # (B, H, W)
            self.evaluator.update(pred, lbls)

        results = self.evaluator.compute()
        self.evaluator.print_results(results)
        self.model.train()
        return results['Mean']['DSC_mean']

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, tag: str = ''):
        fname = f"ckpt_ep{epoch:03d}{'_' + tag if tag else ''}.pth"
        torch.save({
            'epoch':       epoch,
            'model':       self.model.state_dict(),
            'opt_seg':     self.opt_seg.state_dict(),
            'opt_disc':    self.opt_disc.state_dict(),
            'opt_diff':    self.opt_diff.state_dict(),
            'best_dsc':    self.best_dsc,
            'global_step': self.global_step,
            'cfg':         self.cfg,
        }, self.out_dir / fname)
        logger.info(f"  Saved checkpoint: {fname}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.opt_seg.load_state_dict(ckpt['opt_seg'])
        self.opt_disc.load_state_dict(ckpt['opt_disc'])
        self.opt_diff.load_state_dict(ckpt['opt_diff'])
        self.best_dsc    = ckpt.get('best_dsc', 0.0)
        self.global_step = ckpt.get('global_step', 0)
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
