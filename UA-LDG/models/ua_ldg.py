"""
models/ua_ldg.py
────────────────
UA-LDG: Uncertainty-Aware Label Dependency Graph.

Assembles all four components into a single nn.Module:
    1. CXRBackbone                — visual encoder
    2. BetaBinomialEdgeWeights    — posterior edge weight model
    3. UncertaintyAwareGCN        — message passing with uncertainty modulation
    4. EvidentialHead             — calibrated multi-label predictions

Forward input  : images (B, 3, H, W)
Forward output : dict with p_hat, u, alpha, beta, var_A, gates, edge_stats
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CXRBackbone
from .edge_confidence import BetaBinomialEdgeWeights, FixedEdgeWeights
from .ua_gcn import UncertaintyAwareGCN
from .edl_head import EvidentialHead


class UALDG(nn.Module):
    """
    Full UA-LDG model.

    Args:
        num_labels         (int)         : Number of pathology classes (12).
        n_ij               (np.ndarray)  : (L, L) co-occurrence count matrix.
        n_i                (np.ndarray)  : (L,)   marginal disease counts.
        graph_mask         (np.ndarray)  : (L, L) binary edge existence mask.
        label_embed_matrix (np.ndarray)  : (L, d_e) pre-trained label embeddings.
        image_embed_dim    (int)         : 1024 for DenseNet-121.
        label_embed_dim    (int)         : 200 for BioWordVec.
        node_hidden_dim    (int)         : d_h = 512.
        gamma              (float)       : Uncertainty suppression strength.
        gcn_layers         (int)         : Number of GCN layers.
        gcn_dropout        (float)       : GCN dropout rate.
        use_att_gate       (bool)        : Whether to use attention gate.
        edl_hidden_dim     (int)         : EDL MLP hidden dimension.
        backbone_pretrained(bool)        : Use pretrained backbone weights.
        backbone_pretrained_source(str)  : "imagenet" | "torchxrayvision".
        backbone_freeze_bn (bool)        : Freeze BN in backbone.
        model_variant      (str)         : "ua_ldg" | "no_graph" | "fixed_gcn"
            ua_ldg     — full model (Beta-Binomial edges + attention gate)
            no_graph   — backbone + label embeddings → EDL head directly (no GCN)
            fixed_gcn  — fixed co-occurrence edges, no attention gate
    """

    def __init__(
        self,
        num_labels: int,
        n_ij: np.ndarray,
        n_i:  np.ndarray,
        graph_mask: np.ndarray,
        label_embed_matrix: np.ndarray,
        image_embed_dim: int = 1024,
        label_embed_dim: int = 200,
        node_hidden_dim: int = 512,
        gamma: float = 5.0,
        gcn_layers: int = 2,
        gcn_dropout: float = 0.3,
        use_att_gate: bool = True,
        edl_hidden_dim: int = 256,
        backbone_pretrained: bool = True,
        backbone_pretrained_source: str = "imagenet",
        backbone_freeze_bn: bool = False,
        model_variant: str = "ua_ldg",
    ) -> None:
        super().__init__()

        assert model_variant in ("ua_ldg", "no_graph", "fixed_gcn"), \
            f"Unknown model_variant: {model_variant}"
        self.num_labels    = num_labels
        self.model_variant = model_variant

        # ── 1. Backbone ───────────────────────────────────────────────────────
        self.backbone = CXRBackbone(
            embed_dim          = image_embed_dim,
            pretrained         = backbone_pretrained,
            pretrained_source  = backbone_pretrained_source,
            freeze_bn          = backbone_freeze_bn,
        )

        # ── 2. Label embeddings (trainable, initialised from BioWordVec) ──────
        self.label_embeddings = nn.Parameter(
            torch.tensor(label_embed_matrix, dtype=torch.float32)
        )  # (L, d_e)

        # ── 3. Node projection:  [img_embed || label_embed] → d_h ─────────────
        proj_in = image_embed_dim + label_embed_dim
        self.node_proj = nn.Sequential(
            nn.Linear(proj_in, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
        )

        # ── 4. Edge weights and GCN (skipped for no_graph variant) ────────────
        if model_variant != "no_graph":
            if model_variant == "fixed_gcn":
                self.edge_weights = FixedEdgeWeights(
                    n_ij       = n_ij,
                    n_i        = n_i,
                    graph_mask = graph_mask,
                )
                use_att_gate = False  # no variance signal to condition on
            else:
                self.edge_weights = BetaBinomialEdgeWeights(
                    n_ij       = n_ij,
                    n_i        = n_i,
                    graph_mask = graph_mask,
                )

            self.ua_gcn = UncertaintyAwareGCN(
                node_dim   = node_hidden_dim,
                num_layers = gcn_layers,
                dropout    = gcn_dropout,
                use_gate   = use_att_gate,
            )

        # ── 5. Evidential prediction head ─────────────────────────────────────
        self.edl_head = EvidentialHead(
            node_dim   = node_hidden_dim,
            hidden_dim = edl_hidden_dim,
            num_labels = num_labels,
        )

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images : (B, 3, H, W)  normalised CXR images

        Returns:
            dict with keys:
              p_hat      (B, L)      predicted disease probabilities
              u          (B, L)      vacuity (uncertainty) scores
              alpha      (B, L)      prediction Beta α
              beta       (B, L)      prediction Beta β
              var_A      (L, L)      posterior edge variance matrix (zeros for no_graph/fixed_gcn)
              gates      (B, L, L)   attention gate values per image (None for no_graph/fixed_gcn)
              edge_stats dict        edge weight summary for logging
        """
        B = images.size(0)
        L = self.num_labels

        # ── Step 1: Visual embedding ──────────────────────────────────────────
        img_embed, _ = self.backbone(images)    # (B, d_img)

        # ── Step 2: Node initialisation ───────────────────────────────────────
        img_expand = img_embed.unsqueeze(1).expand(-1, L, -1)          # (B, L, d_img)
        lbl_expand = self.label_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_e)
        node_input = torch.cat([img_expand, lbl_expand], dim=-1)        # (B, L, d_img+d_e)
        H0         = self.node_proj(node_input)                          # (B, L, d_h)

        if self.model_variant == "no_graph":
            # Skip GCN — feed node initialisation directly to EDL head
            h_star     = H0
            gates      = None
            var_A      = torch.zeros(L, L, device=images.device)
            edge_stats = {"n_edges": 0, "mean_edge_weight": 0.0, "mean_variance": 0.0}
        else:
            # ── Step 3: Edge weights ──────────────────────────────────────────
            A_hat = self.edge_weights(training=self.training)   # (L, L)
            var_A = self.edge_weights.get_posterior_variance()  # (L, L)

            # ── Step 4: GCN message passing ───────────────────────────────────
            h_star_list = []
            gates_list  = []
            for b in range(B):
                h_b, g_b = self.ua_gcn(H0[b], A_hat, var_A)
                h_star_list.append(h_b)
                if g_b is not None:
                    gates_list.append(g_b)
            h_star     = torch.stack(h_star_list, dim=0)        # (B, L, d_h)
            gates      = torch.stack(gates_list, dim=0) if gates_list else None  # (B, L, L)
            edge_stats = self.edge_weights.get_edge_summary()

        # ── Step 5: Evidential prediction ─────────────────────────────────────
        p_hat, u, alpha_y, beta_y = self.edl_head(h_star)  # each (B, L)

        return {
            "p_hat":      p_hat,
            "u":          u,
            "alpha":      alpha_y,
            "beta":       beta_y,
            "var_A":      var_A,
            "gates":      gates,      # (B, L, L) or None
            "edge_stats": edge_stats,
        }

    # ── Convenience ──────────────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, base_lr: float) -> list:
        """Backbone at base_lr/10; all other parameters at base_lr."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)
        other_params    = [p for p in self.parameters()
                           if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": base_lr / 10, "name": "backbone"},
            {"params": other_params,    "lr": base_lr,      "name": "graph_head"},
        ]
