"""
models/edge_confidence.py
─────────────────────────
Beta-Binomial Edge Weight Model.

Each co-occurrence edge weight is treated as an unknown probability p_ij.
Using a flat Beta(1, 1) prior and observed co-occurrence counts as the
likelihood, the posterior is:

    p_ij | data  ~  Beta(1 + n_ij,  1 + n_i - n_ij)

where n_ij is the number of patients where both disease i and disease j
are present, and n_i is the total number of patients with disease i.

During training:  sample stochastically from the posterior (rsample gives
                  a differentiable reparameterised sample so gradients flow).
During inference: use the posterior mean  alpha / (alpha + beta).

Edges with fewer than COOCCURRENCE_THRESHOLD co-occurrences are masked out
entirely (graph_mask = 0) — they do not exist in the graph.

This class has NO learnable parameters. Everything is derived directly from
the training data co-occurrence counts and stored as fixed buffers.
"""

import numpy as np
import torch
import torch.nn as nn


class BetaBinomialEdgeWeights(nn.Module):
    """
    Beta-Binomial posterior edge weight model.

    Args:
        n_ij       : (L, L) co-occurrence count matrix
        n_i        : (L,)   marginal disease count vector
        graph_mask : (L, L) binary matrix — 1 where edge meets threshold
    """

    def __init__(
        self,
        n_ij:       np.ndarray,
        n_i:        np.ndarray,
        graph_mask: np.ndarray,
    ) -> None:
        super().__init__()

        # Posterior parameters from co-occurrence counts.
        # Prior: Beta(1, 1) — flat, no assumptions before seeing data.
        # After n_ij successes in n_i trials:
        #   alpha = 1 + n_ij
        #   beta  = 1 + n_i - n_ij
        alpha = (1.0 + n_ij).astype(np.float32)                 # (L, L)
        beta  = (1.0 + n_i[:, None] - n_ij).astype(np.float32) # (L, L)

        alpha = np.clip(alpha, 1e-6, None)
        beta  = np.clip(beta,  1e-6, None)

        self.register_buffer("alpha",      torch.tensor(alpha))
        self.register_buffer("beta_param", torch.tensor(beta))
        self.register_buffer("graph_mask", torch.tensor(
            graph_mask.astype(np.float32)))

    def forward(self, training: bool = True) -> torch.Tensor:
        """
        Returns the symmetrically normalised adjacency matrix A_hat.

        During training:  stochastic sample via rsample() (differentiable)
        During inference: deterministic posterior mean

        Args:
            training : True during training, False at test/eval time

        Returns:
            A_hat : (L, L) symmetrically normalised adjacency
        """
        if training:
            dist     = torch.distributions.Beta(self.alpha, self.beta_param)
            A_sample = dist.rsample()
        else:
            A_sample = self.alpha / (self.alpha + self.beta_param)

        A_masked = A_sample * self.graph_mask
        return self._sym_normalise(A_masked)

    def get_posterior_mean(self) -> torch.Tensor:
        """
        Posterior mean edge weight matrix (masked).
        Use for the edge confidence heatmap figure.

        E[p_ij] = alpha / (alpha + beta) = (1 + n_ij) / (2 + n_i)
        """
        mean = self.alpha / (self.alpha + self.beta_param)
        return mean * self.graph_mask

    def get_posterior_variance(self) -> torch.Tensor:
        """
        Posterior variance for each edge (masked).
        Passed to the GCN attention gate and shown in the variance heatmap.

        High variance = uncertain edge (few observations)
        Low variance  = confident edge (many observations)

        Var[p_ij] = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
        """
        S   = self.alpha + self.beta_param
        var = (self.alpha * self.beta_param) / (S * S * (S + 1.0))
        return var * self.graph_mask

    def get_edge_summary(self) -> dict:
        """Summary statistics for logging. Replaces get_source_weights()."""
        mean = self.get_posterior_mean()
        var  = self.get_posterior_variance()
        mask = self.graph_mask.bool()
        return {
            "n_edges":           int(mask.sum().item()),
            "mean_edge_weight":  float(mean[mask].mean().item()) if mask.any() else 0.0,
            "mean_variance":     float(var[mask].mean().item())  if mask.any() else 0.0,
        }

    @staticmethod
    def _sym_normalise(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Symmetric normalisation: D^{-1/2} A D^{-1/2}"""
        deg          = adj.sum(dim=1)
        deg_inv_sqrt = (deg + eps).pow(-0.5)
        D            = torch.diag(deg_inv_sqrt)
        return D @ adj @ D


class FixedEdgeWeights(nn.Module):
    """
    Fixed co-occurrence edge weights for ablation study.

    Uses the plain MLE conditional probability p_ij = n_ij / n_i as a
    deterministic, fixed adjacency — no Beta-Binomial posterior, no variance,
    no stochastic sampling.  Intended to isolate the contribution of
    Beta-Binomial uncertainty modulation vs. any graph signal at all.

    Args:
        n_ij       : (L, L) co-occurrence count matrix
        n_i        : (L,)   marginal disease count vector
        graph_mask : (L, L) binary edge existence mask
    """

    def __init__(
        self,
        n_ij:       np.ndarray,
        n_i:        np.ndarray,
        graph_mask: np.ndarray,
    ) -> None:
        super().__init__()

        n_i_safe = np.where(n_i > 0, n_i, 1).astype(np.float32)
        p_ij     = (n_ij / n_i_safe[:, None]).astype(np.float32)
        p_ij     = np.clip(p_ij, 0.0, 1.0)

        self.register_buffer("p_ij",       torch.tensor(p_ij))
        self.register_buffer("graph_mask", torch.tensor(
            graph_mask.astype(np.float32)))

    def forward(self, training: bool = True) -> torch.Tensor:
        A_masked = self.p_ij * self.graph_mask
        return BetaBinomialEdgeWeights._sym_normalise(A_masked)

    def get_posterior_mean(self) -> torch.Tensor:
        return self.p_ij * self.graph_mask

    def get_posterior_variance(self) -> torch.Tensor:
        return torch.zeros_like(self.p_ij)

    def get_edge_summary(self) -> dict:
        mask = self.graph_mask.bool()
        return {
            "n_edges":          int(mask.sum().item()),
            "mean_edge_weight": float(self.p_ij[mask].mean().item()) if mask.any() else 0.0,
            "mean_variance":    0.0,
        }
