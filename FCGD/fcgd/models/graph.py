"""
Anatomical Graph Construction & GNN Encoder
============================================
Converts convolutional feature maps F into an anatomical graph G=(V,E,A),
then encodes topological structure with an L-layer graph convolution network
following the formulation of Jiang et al. (CVPR 2019).

Graph construction
  • Node vi : average-pooled feature over spatial patch Ωi  (Eq. 3)
  • Edge Aij: exponentiated cosine similarity               (Eq. 4)

GNN propagation  (Eq. 5–6)
  z_i^{l+1} = σ( Σ_{j∈N(i)} Ã_ij · z_j^l · W^l )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

class AnatomicalGraphConstructor(nn.Module):
    """
    Builds a sparse weighted anatomical graph from a dense feature map.

    Args:
        num_nodes      : N – number of graph nodes (spatial patches)
        feat_channels  : C – feature channels
        edge_threshold : τ – keep only edges with A_ij > τ
    """

    def __init__(
        self,
        num_nodes: int = 256,
        feat_channels: int = 256,
        edge_threshold: float = 0.5,
    ):
        super().__init__()
        self.N = num_nodes
        self.C = feat_channels
        self.tau = edge_threshold
        self._sqrt_N = int(math.isqrt(num_nodes))
        assert self._sqrt_N ** 2 == num_nodes, \
            f"num_nodes must be a perfect square, got {num_nodes}"

    def forward(self, F: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            F : convolutional feature map  (B, C, H', W')

        Returns:
            dict:
              'nodes' : node feature matrix  (B, N, C)   [h_i  in Eq. 3]
              'adj'   : normalised adjacency (B, N, N)   [Ã_ij in Eq. 6]
        """
        B, C, Hp, Wp = F.shape

        # ── Node construction (Eq. 3) ─────────────────────────────────────
        # Adaptive average-pool to (sqrt_N × sqrt_N) grid
        sN = self._sqrt_N
        H_nodes = F.new_zeros(B, self.N, C)
        pooled = F.reshape(B, C, sN, Hp // sN, sN, Wp // sN)
        pooled = pooled.mean(dim=[3, 5])                # (B, C, sN, sN)
        H_nodes = pooled.flatten(2).permute(0, 2, 1)   # (B, N, C)

        # ── Edge construction (Eq. 4) ─────────────────────────────────────
        # Cosine similarity → exponentiated → threshold
        H_norm = F.normalize(H_nodes, p=2, dim=-1)     # (B, N, C)
        cos_sim = torch.bmm(H_norm, H_norm.transpose(1, 2))  # (B, N, N)
        A = torch.exp(cos_sim)                          # A_ij = exp(cos)

        # Apply threshold τ (sparse mask)
        A = A * (A > self.tau).float()

        # ── Symmetric normalisation (Eq. 6) ──────────────────────────────
        D = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)     # (B, N, 1)
        D_inv_sqrt = D.pow(-0.5)                             # (B, N, 1)
        A_hat = D_inv_sqrt * A * D_inv_sqrt.transpose(1, 2) # (B, N, N)

        return {'nodes': H_nodes, 'adj': A_hat}


# ─────────────────────────────────────────────────────────────────────────────
# GNN Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    """Single graph convolutional layer: Z^{l+1} = σ(Ã · Z^l · W^l)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, Z: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z     : (B, N, d_in)
            A_hat : (B, N, N)   symmetrically normalised adjacency
        Returns:
            Z_out : (B, N, d_out)
        """
        # Neighbourhood aggregation
        AZ = torch.bmm(A_hat, Z)            # (B, N, d_in)
        out = self.W(AZ)                    # (B, N, d_out)
        # BN over the node dimension
        B, N, D = out.shape
        out = self.bn(out.view(B * N, D)).view(B, N, D)
        return self.act(out)


class GNNEncoder(nn.Module):
    """
    L-layer GNN encoder producing topology-aware latent embeddings Z.

    Args:
        in_dim     : input node feature dimension (= feat_channels C)
        hidden_dim : intermediate layer width
        out_dim    : output dimension d_L
        num_layers : number of propagation layers L
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GNNLayer(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes : (B, N, in_dim)
            adj   : (B, N, N)

        Returns:
            Z : (B, N, out_dim)   – topology-aware latent embeddings
        """
        Z = nodes
        for layer in self.layers:
            Z = layer(Z, adj)
        return Z                   # (B, N, d_L)
