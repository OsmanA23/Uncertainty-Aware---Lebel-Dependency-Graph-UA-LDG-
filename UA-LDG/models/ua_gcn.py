"""
models/ua_gcn.py
────────────────
Uncertainty-Aware Graph Convolutional Network.

Implements:
  1. Two GCN layers with the uncertainty-modulated adjacency matrix Â.
  2. An uncertainty-conditioned attention gate that suppresses messages
     from high-variance (unreliable) edges.

Forward inputs:
  H0      : (L, d_h)  initial node representations
  A_hat   : (L, L)    symmetrically normalised, uncertainty-modulated adjacency
  var_A   : (L, L)    edge variance matrix (for attention gate)

Forward output:
  h_star : (L, d_h)   final gated node representations
  gates  : (L, L) or None — per-edge attention gate values (None when use_gate=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Single GCN layer ─────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Standard graph convolutional layer:
        H' = σ( Â · H · W )

    Args:
        in_dim  (int) : Input node feature dimension.
        out_dim (int) : Output node feature dimension.
        dropout (float) : Dropout probability applied to node features.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H     : (L, in_dim)
            A_hat : (L, L)  normalised adjacency

        Returns:
            H_out : (L, out_dim)
        """
        # Message passing:  A_hat @ H @ W  +  residual
        agg   = A_hat @ H                        # (L, in_dim)  neighbour aggregation
        H_out = self.linear(agg)                 # (L, out_dim)
        H_out = H_out + H                        # residual keeps node's own identity
        H_out = self.norm(H_out)
        H_out = F.leaky_relu(H_out, negative_slope=0.2)
        H_out = self.dropout(H_out)
        return H_out


# ── Attention gate ────────────────────────────────────────────────────────────

class UncertaintyAttentionGate(nn.Module):
    """
    Computes per-edge attention weights conditioned on edge variance.

    g_ij = sigmoid( w_g^T [ h_i || h_j || Var(A_ij) ] )

    Low-confidence edges (high variance) will receive gate values close to 0,
    suppressing their contribution to the final node representation.

    Args:
        node_dim (int) : Dimension of node representations.
    """

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        # Input: concat of two node vectors + 1 scalar variance
        self.gate_fc = nn.Linear(node_dim * 2 + 1, 1, bias=True)

    def forward(
        self,
        H: torch.Tensor,        # (L, d_h)
        var_A: torch.Tensor,    # (L, L)
        A_hat: torch.Tensor,    # (L, L)  normalised adjacency
    ) -> tuple:
        """
        Args:
            H     : (L, d_h)  node representations from GCN
            var_A : (L, L)    edge variances
            A_hat : (L, L)    normalised adjacency (used for gated aggregation)

        Returns:
            h_star : (L, d_h)  gated node representations
            gates  : (L, L)    per-edge attention gate values in (0, 1)
        """
        L, d = H.shape

        # Build pairwise input tensor  (L, L, 2*d + 1)
        H_i = H.unsqueeze(1).expand(-1, L, -1)     # (L, L, d)
        H_j = H.unsqueeze(0).expand(L, -1, -1)     # (L, L, d)
        V   = var_A.unsqueeze(-1)                   # (L, L, 1)

        gate_input = torch.cat([H_i, H_j, V], dim=-1)  # (L, L, 2d+1)
        gates = torch.sigmoid(self.gate_fc(gate_input)).squeeze(-1)  # (L, L)

        # Gated aggregation:  h*_i = h_i + Σ_j  g_ij · Â_ij · h_j
        gated_adj = gates * A_hat                   # (L, L)
        h_agg     = gated_adj @ H                   # (L, d)
        h_star    = H + h_agg                       # residual connection
        return h_star, gates


# ── Full UA-GCN ───────────────────────────────────────────────────────────────

class UncertaintyAwareGCN(nn.Module):
    """
    Two-layer GCN followed by an uncertainty-conditioned attention gate.

    Args:
        node_dim    (int)   : Node feature dimension (d_h).
        num_layers  (int)   : Number of GCN layers (default 2).
        dropout     (float) : GCN layer dropout.
        use_gate    (bool)  : Whether to apply the attention gate.
    """

    def __init__(
        self,
        node_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate

        # GCN layers (all same dimension for simplicity)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(node_dim, node_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        if use_gate:
            self.att_gate = UncertaintyAttentionGate(node_dim)

    def forward(
        self,
        H0: torch.Tensor,        # (L, d_h)
        A_hat: torch.Tensor,     # (L, L)  normalised adjacency
        var_A: torch.Tensor,     # (L, L)  edge variances
    ) -> tuple:
        """
        Args:
            H0    : (L, d_h)  initial node features
            A_hat : (L, L)    uncertainty-modulated, sym-normalised adjacency
            var_A : (L, L)    edge variance matrix

        Returns:
            h_star : (L, d_h)  final gated node representations
            gates  : (L, L) or None — attention gate values (None if use_gate=False)
        """
        H = H0
        for layer in self.gcn_layers:
            H = layer(H, A_hat)              # (L, d_h)

        if self.use_gate:
            H, gates = self.att_gate(H, var_A, A_hat)
        else:
            gates = None

        return H, gates
