"""
utils/losses.py
───────────────
Composite training loss for UA-LDG:

    L = L_cls + λ_edl · L_edl

Where:
    L_cls : Asymmetric Binary Cross-Entropy (handles class imbalance)
    L_edl : EDL KL regulariser (penalises overconfident wrong predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Asymmetric BCE ────────────────────────────────────────────────────────────

class AsymmetricBCELoss(nn.Module):
    """
    Asymmetric Binary Cross-Entropy Loss.

    Applies different focusing factors to positive and negative samples,
    addressing the severe label imbalance in multi-label CXR datasets.

    L_ASL = -[ y · log(p) + (1-y) · (1-p)^{γ-} · log(1-p) ]

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification",
        ICCV 2021.

    Args:
        gamma_neg (float) : Focusing exponent for negative samples (default 4).
        gamma_pos (float) : Focusing exponent for positive samples (default 0).
        clip      (float) : Probability shift to prevent log(0) [0.0, 0.05].
        reduction (str)   : "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,    # (B, L)  predicted probabilities in (0,1)
        target: torch.Tensor,  # (B, L)  binary ground-truth {0,1}
    ) -> torch.Tensor:

        # Probability shift (clip negative prob away from 1)
        pred_pos = pred
        pred_neg = (pred - self.clip).clamp(min=0.0, max=1.0)

        log_p_pos = torch.log(pred_pos.clamp(min=1e-8))
        log_p_neg = torch.log((1.0 - pred_neg).clamp(min=1e-8))

        # Asymmetric focusing
        # Positive: (1-p)^γ+ · log(p)  — downweights easy positives (high p)
        # Negative: p_m^γ- · log(1-p_m) — downweights easy negatives (low p_m)
        pos_loss = target       * (1.0 - pred_pos) ** self.gamma_pos * log_p_pos
        neg_loss = (1 - target) * pred_neg         ** self.gamma_neg * log_p_neg

        loss = -(pos_loss + neg_loss)                  # (B, L)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── EDL KL regulariser ────────────────────────────────────────────────────────

class EDLRegularisationLoss(nn.Module):
    """
    Evidential Deep Learning KL-divergence regulariser.

    Penalises the model for generating evidence for incorrect labels,
    encouraging the predicted Beta distribution to collapse to Beta(1,1)
    — the uninformative uniform prior — when the label is absent.

    Uses annealing: the penalty grows linearly from 0 to its full weight
    over the first `annealing_epochs` training epochs, preventing the
    regulariser from dominating before the model has learned the basic task.

    Args:
        annealing_epochs (int) : Number of epochs to ramp up to full penalty.
    """

    def __init__(self, annealing_epochs: int = 10) -> None:
        super().__init__()
        self.annealing_epochs = annealing_epochs

    def forward(
        self,
        y: torch.Tensor,          # (B, L)  binary ground-truth
        alpha: torch.Tensor,      # (B, L)  Beta α from EDL head
        beta: torch.Tensor,       # (B, L)  Beta β from EDL head
        current_epoch: int = 0,
    ) -> torch.Tensor:

        lam_t = min(1.0, current_epoch / max(self.annealing_epochs, 1))

        # Remove correct-class evidence before computing KL:
        #   y=1 → don't penalise α (presence evidence is correct), penalise β
        #   y=0 → penalise α (presence evidence is wrong), don't penalise β
        alpha_t = (1.0 - y) * alpha + y * torch.ones_like(alpha)
        beta_t  =       y   * beta  + (1.0 - y) * torch.ones_like(beta)

        kl = _kl_beta_uniform(alpha_t, beta_t)     # (B, L)
        return lam_t * kl.mean()


def _kl_beta_uniform(alpha: torch.Tensor, beta_: torch.Tensor) -> torch.Tensor:
    """KL[ Beta(α, β) || Beta(1, 1) ] — closed form."""
    log_beta_ab = (
        torch.lgamma(alpha)
        + torch.lgamma(beta_)
        - torch.lgamma(alpha + beta_)
    )
    psi_a  = torch.digamma(alpha)
    psi_b  = torch.digamma(beta_)
    psi_ab = torch.digamma(alpha + beta_)
    return (
        -log_beta_ab
        + (alpha - 1.0) * psi_a
        + (beta_ - 1.0) * psi_b
        - (alpha + beta_ - 2.0) * psi_ab
    )


# ── Composite loss ────────────────────────────────────────────────────────────

class UALDGLoss(nn.Module):
    """
    Composite loss for UA-LDG.

        L = L_cls + λ_edl · L_edl

    Args:
        lambda_edl       (float) : Weight for EDL regulariser.
        gamma_neg        (float) : ASL negative focusing exponent.
        gamma_pos        (float) : ASL positive focusing exponent.
        annealing_epochs (int)   : EDL annealing duration.
    """

    def __init__(
        self,
        lambda_edl: float = 0.1,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        annealing_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.lambda_edl = lambda_edl
        self.cls_loss   = AsymmetricBCELoss(gamma_neg=gamma_neg,
                                            gamma_pos=gamma_pos)
        self.edl_loss   = EDLRegularisationLoss(annealing_epochs=annealing_epochs)

    def forward(
        self,
        p_hat: torch.Tensor,   # (B, L)
        y: torch.Tensor,       # (B, L)
        alpha: torch.Tensor,   # (B, L)
        beta: torch.Tensor,    # (B, L)
        current_epoch: int = 0,
    ) -> dict:
        """
        Returns:
            dict with keys: total, cls, edl
        """
        l_cls = self.cls_loss(p_hat, y)
        l_edl = self.edl_loss(y, alpha, beta, current_epoch)
        total = l_cls + self.lambda_edl * l_edl

        return {
            "total": total,
            "cls":   l_cls,
            "edl":   l_edl,
        }
