"""
Frequency-Conditioned Graph Diffusion (FCGD)
============================================
Main model class integrating all five modules:
  1. Convolutional Encoder (Ec)
  2. Anatomical Graph Construction
  3. GNN Encoder (Eg)
  4. Frequency-Conditioned Latent Diffusion (Dφ)
  5. Segmentation Decoder (C)

Reference:
  Usmani et al., "Robust Unsupervised Domain Adaptation for Medical Image
  Segmentation via Frequency-Conditioned Graph Diffusion", Scientific Reports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .encoder import ConvEncoder
from .graph import AnatomicalGraphConstructor, GNNEncoder
from .wavelet import WaveletEncoder
from .diffusion import FrequencyConditionedDiffusion
from .decoder import SegmentationDecoder
from .discriminator import DomainDiscriminator


class FCGD(nn.Module):
    """
    Full FCGD model for unsupervised domain adaptive medical image segmentation.

    Args:
        in_channels    : number of input image channels (1 for MRI/CT grayscale)
        num_classes    : number of segmentation classes K
        img_size       : spatial size of the input image (H=W assumed)
        enc_channels   : list of channel widths in the convolutional encoder
        num_nodes      : number of graph nodes N  (default 256)
        gnn_layers     : number of GNN propagation layers L (default 3)
        gnn_hidden_dim : hidden dimensionality d_L of GNN embeddings
        freq_dim       : dimensionality d_f of the frequency conditioning vector
        diff_timesteps : total diffusion timesteps T (default 10)
        mc_samples     : number of Monte-Carlo inference samples M (default 10)
        lambda_str     : weight λ for structural consistency loss
        mu_adv         : weight µ for adversarial loss
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 256,
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256),
        num_nodes: int = 256,
        gnn_layers: int = 3,
        gnn_hidden_dim: int = 256,
        freq_dim: int = 128,
        diff_timesteps: int = 10,
        mc_samples: int = 10,
        lambda_str: float = 0.1,
        mu_adv: float = 0.01,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.mc_samples = mc_samples
        self.lambda_str = lambda_str
        self.mu_adv = mu_adv

        # ── Module 1: Convolutional Encoder ──────────────────────────────────
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            channels=enc_channels,
        )
        feat_channels = enc_channels[-1]  # C in the paper
        feat_h = img_size // (2 ** len(enc_channels))  # H'
        feat_w = img_size // (2 ** len(enc_channels))  # W'

        # ── Module 2 & 3: Graph Construction + GNN Encoder ───────────────────
        self.graph_builder = AnatomicalGraphConstructor(
            num_nodes=num_nodes,
            feat_channels=feat_channels,
            edge_threshold=0.5,
        )
        self.gnn_encoder = GNNEncoder(
            in_dim=feat_channels,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
        )

        # ── Module 4a: Wavelet / Frequency Encoder ───────────────────────────
        self.wavelet_encoder = WaveletEncoder(
            feat_channels=feat_channels,
            freq_dim=freq_dim,
        )

        # ── Module 4b: Frequency-Conditioned Latent Diffusion ─────────────────
        self.diffusion = FrequencyConditionedDiffusion(
            node_dim=gnn_hidden_dim,
            freq_dim=freq_dim,
            timesteps=diff_timesteps,
        )

        # ── Module 5: Segmentation Decoder ───────────────────────────────────
        self.decoder = SegmentationDecoder(
            node_dim=gnn_hidden_dim,
            enc_channels=enc_channels,
            num_classes=num_classes,
            img_size=img_size,
            num_nodes=num_nodes,
        )

        # ── Domain Discriminator ──────────────────────────────────────────────
        self.discriminator = DomainDiscriminator(node_dim=gnn_hidden_dim)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward helpers
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run convolutional encoder → graph → GNN → frequency encoder.

        Returns:
            Z    : GNN latent embeddings  (B, N, d_L)
            f    : frequency vector       (B, d_f)
            F    : raw feature map        (B, C, H', W')
        """
        F = self.encoder(x)                         # (B, C, H', W')
        G = self.graph_builder(F)                   # dict with nodes, adj
        Z = self.gnn_encoder(G['nodes'], G['adj'])  # (B, N, d_L)
        f = self.wavelet_encoder(F)                 # (B, d_f)
        return Z, f, F

    def decode(self, Z_star: torch.Tensor, enc_feats) -> torch.Tensor:
        """Map refined graph embeddings Z* → dense segmentation logits."""
        return self.decoder(Z_star, enc_feats)      # (B, K, H, W)

    # ─────────────────────────────────────────────────────────────────────────
    # Training forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        xs: torch.Tensor,
        xt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward pass.

        Args:
            xs : source-domain images  (B, C_in, H, W)
            xt : target-domain images  (B, C_in, H, W)

        Returns:
            dict with keys:
              'pred_s'   – source segmentation logits  (B, K, H, W)
              'pred_t'   – target segmentation logits  (B, K, H, W)
              'Z_s'      – source GNN embeddings       (B, N, d_L)
              'Z_t'      – target GNN embeddings       (B, N, d_L)
              'd_s'      – discriminator scores source (B, 1)
              'd_t'      – discriminator scores target (B, 1)
        """
        # ── Source domain ────────────────────────────────────────────────────
        Z_s, f_s, F_s = self.encode(xs)
        Z_star_s = self.diffusion.denoise(Z_s, f_s)
        enc_feats_s = self.encoder.get_skip_features()
        pred_s = self.decode(Z_star_s, enc_feats_s)  # (B, K, H, W)

        # ── Target domain ────────────────────────────────────────────────────
        Z_t, f_t, F_t = self.encode(xt)
        Z_star_t = self.diffusion.denoise(Z_t, f_t)
        enc_feats_t = self.encoder.get_skip_features()
        pred_t = self.decode(Z_star_t, enc_feats_t)  # (B, K, H, W)

        # ── Discriminator scores ─────────────────────────────────────────────
        d_s = self.discriminator(Z_s)
        d_t = self.discriminator(Z_t)

        return dict(
            pred_s=pred_s,
            pred_t=pred_t,
            Z_s=Z_s,
            Z_t=Z_t,
            d_s=d_s,
            d_t=d_t,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Inference with uncertainty estimation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mc_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uncertainty-aware inference via Monte-Carlo diffusion sampling.

        Args:
            x          : input image  (B, C_in, H, W)
            mc_samples : override default M

        Returns:
            y_bar : ensemble mean prediction  (B, K, H, W)
            U     : pixel-wise uncertainty    (B, H, W)
        """
        M = mc_samples or self.mc_samples
        Z, f, _ = self.encode(x)
        enc_feats = self.encoder.get_skip_features()

        preds = []
        for _ in range(M):
            Z_star = self.diffusion.denoise(Z, f)
            logits = self.decode(Z_star, enc_feats)
            preds.append(torch.softmax(logits, dim=1))   # (B, K, H, W)

        preds = torch.stack(preds, dim=0)                # (M, B, K, H, W)
        y_bar = preds.mean(dim=0)                        # (B, K, H, W)
        # Pixel-wise variance averaged over classes
        U = preds.var(dim=0).mean(dim=1)                 # (B, H, W)
        return y_bar, U
