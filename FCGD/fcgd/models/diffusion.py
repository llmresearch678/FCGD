"""
Frequency-Conditioned Latent Diffusion (Dφ)
============================================
Implements the forward diffusion process (Eq. 9–10) and the frequency-
conditioned reverse denoising process (Eq. 11–13) over the GNN graph
embeddings Z ∈ R^{N×d_L}.

Key design choices
  • Cosine noise schedule (Nichol & Dhariwal, 2021)
  • Feature-wise affine modulation (FiLM) by the frequency vector f
  • Lightweight Transformer denoiser operating on node sequences
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule utilities
# ─────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine variance schedule from Nichol & Dhariwal (2021).
    Returns β_1 … β_T  (shape: (T,)).
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0, 0.999).float()


# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning layer
# ─────────────────────────────────────────────────────────────────────────────

class FiLMCondition(nn.Module):
    """
    Feature-wise Linear Modulation: scale and shift Z via f.
    γ, β = MLP(f) → Z_out = γ · Z + β
    """

    def __init__(self, node_dim: int, freq_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, node_dim * 2),
            nn.SiLU(),
            nn.Linear(node_dim * 2, node_dim * 2),
        )

    def forward(self, Z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z : (B, N, d_L)
            f : (B, d_f)
        Returns:
            Z_modulated : (B, N, d_L)
        """
        params = self.mlp(f)                           # (B, 2*d_L)
        gamma, beta = params.chunk(2, dim=-1)          # each (B, d_L)
        gamma = gamma.unsqueeze(1)                     # (B, 1, d_L)
        beta  = beta.unsqueeze(1)                      # (B, 1, d_L)
        return (1 + gamma) * Z + beta


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : integer timestep tensor (B,)
        Returns:
            emb : (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)
        return self.proj(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Denoiser network εφ(Z_t, f, t)
# ─────────────────────────────────────────────────────────────────────────────

class DenoiserBlock(nn.Module):
    """One Transformer block with FiLM conditioning."""

    def __init__(self, node_dim: int, freq_dim: int, nhead: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(node_dim)
        self.attn  = nn.MultiheadAttention(node_dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(node_dim)
        self.ff    = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Linear(node_dim * 4, node_dim),
        )
        self.film = FiLMCondition(node_dim, freq_dim)

    def forward(self, Z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        Z = self.film(Z, f)
        Z2, _ = self.attn(self.norm1(Z), self.norm1(Z), self.norm1(Z))
        Z = Z + Z2
        Z = Z + self.ff(self.norm2(Z))
        return Z


class NoisePredictor(nn.Module):
    """
    εφ(Z_t, f, t) – predicts the noise added at diffusion step t.

    Architecture:
      Input projection → time injection → stack of DenoiserBlocks → output projection
    """

    def __init__(
        self,
        node_dim: int = 256,
        freq_dim: int = 128,
        time_dim: int = 128,
        depth:    int = 4,
        nhead:    int = 4,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        cond_dim = freq_dim + time_dim

        self.in_proj  = nn.Linear(node_dim, node_dim)
        self.blocks   = nn.ModuleList([
            DenoiserBlock(node_dim, cond_dim, nhead)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(node_dim, node_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        Z_t: torch.Tensor,
        f:   torch.Tensor,
        t:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Z_t : noisy embeddings    (B, N, d_L)
            f   : frequency vector    (B, d_f)
            t   : diffusion timestep  (B,)
        Returns:
            eps_pred : predicted noise (B, N, d_L)
        """
        t_emb = self.time_emb(t)                      # (B, time_dim)
        cond  = torch.cat([f, t_emb], dim=-1)         # (B, cond_dim)

        Z = self.in_proj(Z_t)
        for block in self.blocks:
            Z = block(Z, cond)
        return self.out_proj(Z)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Conditioned Diffusion
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyConditionedDiffusion(nn.Module):
    """
    Latent diffusion model over GNN graph embeddings.

    Training:  compute_loss()  → Eq. 13 (simplified denoising score matching)
    Inference: denoise()       → run reverse chain from t=T to t=0

    Args:
        node_dim   : d_L – dimensionality of graph embeddings
        freq_dim   : d_f – dimensionality of frequency conditioning vector
        timesteps  : T   – total diffusion steps (default 10)
    """

    def __init__(
        self,
        node_dim:  int = 256,
        freq_dim:  int = 128,
        timesteps: int = 10,
    ):
        super().__init__()
        self.T = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_bar = alphas.cumprod(dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.register_buffer('betas',       betas)
        self.register_buffer('alphas',      alphas)
        self.register_buffer('alphas_bar',  alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_alphas_bar', alphas_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_bar', (1.0 - alphas_bar).sqrt())

        self.denoiser = NoisePredictor(
            node_dim=node_dim,
            freq_dim=freq_dim,
            time_dim=max(node_dim // 2, 64),
        )

    # ── Forward process (Eq. 10) ──────────────────────────────────────────

    def q_sample(self, Z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Sample Z_t from the forward process: Z_t = √ᾱ_t · Z_0 + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(Z0)
        sqrt_ab  = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return sqrt_ab * Z0 + sqrt_1ab * noise, noise

    # ── Training loss (Eq. 13) ────────────────────────────────────────────

    def compute_loss(self, Z0: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Simplified denoising score-matching objective.

        L_diff = E_{t, Z0, ε} [ || ε - εφ(√ᾱ_t Z0 + √(1-ᾱ_t) ε, f, t) ||² ]
        """
        B = Z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=Z0.device)
        noise = torch.randn_like(Z0)
        Zt, _ = self.q_sample(Z0, t, noise)
        eps_pred = self.denoiser(Zt, f, t)
        return F.mse_loss(eps_pred, noise)

    # ── Reverse process (Eq. 11–12) ──────────────────────────────────────

    @torch.no_grad()
    def denoise(self, Z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Run the full reverse chain from t=T to t=0.

        Args:
            Z : GNN embeddings Z_0  (B, N, d_L)
            f : frequency vector    (B, d_f)
        Returns:
            Z_star : refined embeddings Z*_0  (B, N, d_L)
        """
        B = Z.shape[0]
        device = Z.device
        Zt = torch.randn_like(Z)    # start from pure noise

        for t_idx in reversed(range(self.T)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            beta_t      = self.betas[t].view(-1, 1, 1)
            alpha_t     = self.alphas[t].view(-1, 1, 1)
            alpha_bar_t = self.alphas_bar[t].view(-1, 1, 1)

            eps_pred = self.denoiser(Zt, f, t)

            # Posterior mean (Eq. 12)
            mu = (1 / alpha_t.sqrt()) * (
                Zt - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(Zt)
                sigma = beta_t.sqrt()
                Zt = mu + sigma * noise
            else:
                Zt = mu

        return Zt
