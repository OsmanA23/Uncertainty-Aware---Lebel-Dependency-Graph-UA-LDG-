"""
models/edl_head.py
──────────────────
Evidential Multi-Label Prediction Head.

For each pathology l, maps the gated node representation h*_l to:
    - evidence_alpha e_α >= 0  and  evidence_beta e_β >= 0  (via softplus)
    - Beta parameters (α_l, β_l) — both learned from node representations
    - predicted probability p_hat_l  = α_l / (α_l + β_l)
    - vacuity score       u_l        = 2 / (α_l + β_l)

Both α and β are learned (previously β was fixed at 2.0, which caused
systematic underconfidence and poor ECE on imbalanced data).
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

    @staticmethod
    def compute_edl_loss(
        y: torch.Tensor,        # (B, L)  ground-truth binary labels
        alpha: torch.Tensor,    # (B, L)
        beta:  torch.Tensor,    # (B, L)
        annealing_step: int = 1,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """
        EDL KL-divergence regulariser.

        Encourages the model to output low evidence (uninformative Beta(1,1))
        for incorrectly labelled pathologies.

        The annealing coefficient λ_t ramps from 0 → 1 over training epochs,
        allowing the model to first learn the classification task before
        being penalised for overconfident wrong predictions.

        Args:
            y              : (B, L) binary ground-truth
            alpha, beta    : (B, L) predicted Beta parameters
            annealing_step : epoch at which λ_t reaches 1
            current_epoch  : current training epoch

        Returns:
            edl_loss : scalar
        """
        # Annealing coefficient
        lam_t = min(1.0, current_epoch / max(annealing_step, 1))

        # Remove ground-truth evidence before computing KL
        # α̃ = (1-y)·α + y·1  →  "what would we predict without the true class?"
        alpha_tilde = (1.0 - y) * alpha + y       * torch.ones_like(alpha)
        beta_tilde  =       y   * beta  + (1.0 - y) * torch.ones_like(beta)

        # KL[ Beta(α̃, β̃) || Beta(1,1) ] = log B(1,1)/B(α̃,β̃) + ...
        # For Beta(α,β) vs Beta(1,1):
        #   KL = log[Γ(α+β)/(Γ(α)Γ(β))] + (α-1)ψ(α) + (β-1)ψ(β) - (α+β-2)ψ(α+β)
        #   where ψ is the digamma function
        kl = _beta_kl_uniform(alpha_tilde, beta_tilde)  # (B, L)
        return lam_t * kl.mean()


# ── KL divergence helper ──────────────────────────────────────────────────────

def _beta_kl_uniform(alpha: torch.Tensor, beta_: torch.Tensor) -> torch.Tensor:
    """
    KL[ Beta(α, β) || Beta(1, 1) ]

    Closed-form using the digamma (ψ) function.
    Beta(1,1) is the uniform distribution over [0,1].
    """
    # log B(1,1) = log Γ(1)Γ(1)/Γ(2) = log(1·1/1) = 0
    # log B(α,β) = log Γ(α) + log Γ(β) - log Γ(α+β)
    log_beta_ab = (
        torch.lgamma(alpha)
        + torch.lgamma(beta_)
        - torch.lgamma(alpha + beta_)
    )
    psi_alpha    = torch.digamma(alpha)
    psi_beta     = torch.digamma(beta_)
    psi_alpha_beta = torch.digamma(alpha + beta_)

    kl = (
        -log_beta_ab
        + (alpha - 1.0) * psi_alpha
        + (beta_ - 1.0) * psi_beta
        - (alpha + beta_ - 2.0) * psi_alpha_beta
    )
    return kl
