"""
models/edl_head.py
──────────────────
Evidential Multi-Label Prediction Head.

For each pathology l, maps the gated node representation h*_l to:
    - evidence_alpha e_α >= 0  and  evidence_beta e_β >= 0  (via softplus)
    - Beta parameters (α_l, β_l) — both learned from node representations
    - predicted probability p_hat_l  = α_l / (α_l + β_l)
    - vacuity score       u_l        = 2 / (α_l + β_l)

Both α and β are learned from node representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Two-branch MLP producing learnable α and β for each pathology.

    Args:
        node_dim    (int) : Input dimension (d_h = 512).
        hidden_dim  (int) : MLP hidden dimension.
        num_labels  (int) : Number of pathology classes (12).
    """

    def __init__(
        self,
        node_dim: int = 512,
        hidden_dim: int = 256,
        num_labels: int = 12,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.mlp_alpha = nn.Linear(hidden_dim, 1)   # evidence FOR presence
        self.mlp_beta  = nn.Linear(hidden_dim, 1)   # evidence FOR absence

    def forward(self, h_star: torch.Tensor):
        """
        Args:
            h_star : (B, L, d_h)  batched gated node representations

        Returns:
            p_hat  : (B, L)  predicted disease probabilities in (0,1)
            u      : (B, L)  vacuity scores in (0,1]  — higher = less certain
            alpha  : (B, L)  Beta α parameters
            beta   : (B, L)  Beta β parameters
        """
        h = self.encoder(h_star)                                  # (B, L, hidden_dim)

        alpha = F.softplus(self.mlp_alpha(h).squeeze(-1)) + 1.0  # (B, L) >= 1
        beta  = F.softplus(self.mlp_beta(h).squeeze(-1))  + 1.0  # (B, L) >= 1

        # Predicted probability (mean of Beta)
        p_hat = alpha / (alpha + beta)                            # (B, L) ∈ (0,1)

        # Vacuity: uncertainty about the prediction (inverse of total evidence)
        u = 2.0 / (alpha + beta)                                  # (B, L) ∈ (0,1]

        return p_hat, u, alpha, beta

