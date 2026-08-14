"""
tests/test_forward_pass.py
──────────────────────────
Tests the full UA-LDG forward pass using synthetic data (no real dataset needed).
Verifies shapes, value ranges, and that all components connect correctly.

Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from models.ua_ldg import UALDG
from models.edge_confidence import BetaBinomialEdgeWeights
from utils.losses import UALDGLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

NUM_LABELS   = 12
BATCH_SIZE   = 4
IMAGE_SIZE   = 224
LABEL_DIM    = 200
NODE_DIM     = 512


def make_synthetic_graph_data(L=NUM_LABELS):
    """Create random but valid graph data arrays."""
    np.random.seed(0)
    n_ij = np.random.randint(0, 5000, (L, L)).astype(np.float32)
    np.fill_diagonal(n_ij, 0)
    n_i  = n_ij.sum(axis=1) + 500
    # Threshold at 100 — ensure at least some edges exist
    graph_mask = (n_ij >= 100).astype(np.float32)
    label_embeds = np.random.randn(L, LABEL_DIM).astype(np.float32)
    return n_ij, n_i, graph_mask, label_embeds


@pytest.fixture(scope="module")
def model():
    n_ij, n_i, graph_mask, label_embeds = make_synthetic_graph_data()
    m = UALDG(
        num_labels          = NUM_LABELS,
        n_ij                = n_ij,
        n_i                 = n_i,
        graph_mask          = graph_mask,
        label_embed_matrix  = label_embeds,
        image_embed_dim     = 1024,
        label_embed_dim     = LABEL_DIM,
        node_hidden_dim     = NODE_DIM,
        backbone_pretrained = False,   # faster for tests
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_batch():
    images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, 2, (BATCH_SIZE, NUM_LABELS)).float()
    return images, labels


# ── Forward pass tests ────────────────────────────────────────────────────────

class TestForwardPass:

    def test_output_keys(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        expected = {"p_hat", "u", "alpha", "beta", "var_A", "edge_stats"}
        assert expected.issubset(out.keys()), \
            f"Missing keys: {expected - out.keys()}"

    def test_p_hat_shape(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        assert out["p_hat"].shape == (BATCH_SIZE, NUM_LABELS)

    def test_p_hat_range(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        p = out["p_hat"]
        assert (p >= 0.0).all() and (p <= 1.0).all(), \
            f"p_hat out of [0,1]: min={p.min():.4f}, max={p.max():.4f}"

    def test_vacuity_range(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        u = out["u"]
        assert (u > 0.0).all() and (u <= 1.0).all(), \
            f"Vacuity out of (0,1]: min={u.min():.4f}, max={u.max():.4f}"

    def test_alpha_beta_positive(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        assert (out["alpha"] > 0).all()
        assert (out["beta"]  > 0).all()

    def test_var_A_shape(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        assert out["var_A"].shape == (NUM_LABELS, NUM_LABELS)

    def test_var_A_nonnegative(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        assert (out["var_A"] >= 0).all()

    def test_edge_stats_keys(self, model, dummy_batch):
        images, _ = dummy_batch
        with torch.no_grad():
            out = model(images)
        stats = out["edge_stats"]
        assert "n_edges" in stats
        assert "mean_edge_weight" in stats
        assert "mean_variance" in stats
        assert stats["n_edges"] >= 0


# ── BetaBinomialEdgeWeights tests ─────────────────────────────────────────────

class TestBetaBinomialEdgeWeights:

    def test_posterior_mean_close_to_mle_with_large_counts(self):
        """With large counts, posterior mean ≈ MLE estimate n_ij / n_i."""
        L    = NUM_LABELS
        n_ij = np.random.randint(1000, 5000, (L, L)).astype(np.float32)
        np.fill_diagonal(n_ij, 0)
        n_i  = n_ij.sum(axis=1) + 1000
        mask = np.ones((L, L), dtype=np.float32)
        np.fill_diagonal(mask, 0)

        ew   = BetaBinomialEdgeWeights(n_ij, n_i, mask)
        mean = ew.get_posterior_mean()
        mle  = torch.tensor(n_ij / (n_i[:, None] + 1e-6)) * torch.tensor(mask)

        assert torch.allclose(mean, mle.float(), atol=0.01), \
            "With large counts posterior mean should be close to MLE"

    def test_sparse_edge_has_higher_variance(self):
        """Sparse edge (n=10) should have much higher variance than dense (n=5000)."""
        L    = NUM_LABELS
        mask = np.ones((L, L), dtype=np.float32)
        np.fill_diagonal(mask, 0)

        ew_sparse = BetaBinomialEdgeWeights(
            np.full((L, L), 10.0), np.full(L, 50.0), mask)
        ew_dense  = BetaBinomialEdgeWeights(
            np.full((L, L), 4000.0), np.full(L, 5000.0), mask)

        var_sparse = ew_sparse.get_posterior_variance().mean().item()
        var_dense  = ew_dense.get_posterior_variance().mean().item()

        assert var_sparse > var_dense * 10, \
            "Sparse edges must have much higher variance than dense edges"

    def test_masked_edges_are_zero(self):
        """Edges below threshold (mask=0) must be zero in mean and variance."""
        L    = NUM_LABELS
        n_ij = np.random.randint(0, 500, (L, L)).astype(np.float32)
        n_i  = n_ij.sum(axis=1) + 100
        mask = np.zeros((L, L), dtype=np.float32)
        mask[0, 1] = 1.0

        ew   = BetaBinomialEdgeWeights(n_ij, n_i, mask)
        mean = ew.get_posterior_mean()
        var  = ew.get_posterior_variance()

        assert mean[0, 1].item() > 0,  "Existing edge should have non-zero mean"
        assert mean[1, 2].item() == 0, "Masked edge should have zero mean"
        assert var[1, 2].item()  == 0, "Masked edge should have zero variance"

    def test_training_samples_are_stochastic(self):
        """Training samples from a sparse edge should differ across calls."""
        L    = NUM_LABELS
        n_ij = np.full((L, L), 10.0)
        n_i  = np.full(L, 50.0)
        mask = np.ones((L, L), dtype=np.float32)
        np.fill_diagonal(mask, 0)

        ew = BetaBinomialEdgeWeights(n_ij, n_i, mask)
        s1 = ew(training=True)
        s2 = ew(training=True)
        assert not torch.allclose(s1, s2, atol=1e-6), \
            "Training samples should be stochastic"

    def test_inference_is_deterministic(self):
        """Inference (posterior mean) should be identical across calls."""
        L    = NUM_LABELS
        n_ij = np.random.randint(100, 1000, (L, L)).astype(np.float32)
        n_i  = n_ij.sum(axis=1) + 500
        mask = (n_ij > 100).astype(np.float32)

        ew = BetaBinomialEdgeWeights(n_ij, n_i, mask)
        i1 = ew(training=False)
        i2 = ew(training=False)
        assert torch.allclose(i1, i2), \
            "Inference should be deterministic (posterior mean)"


# ── Loss tests ────────────────────────────────────────────────────────────────

class TestLoss:

    def test_loss_is_scalar(self, model, dummy_batch):
        images, labels = dummy_batch
        with torch.no_grad():
            out = model(images)
        criterion = UALDGLoss()
        losses = criterion(
            p_hat         = out["p_hat"],
            y             = labels,
            alpha         = out["alpha"],
            beta          = out["beta"],
            current_epoch = 5,
        )
        assert losses["total"].shape == torch.Size([])

    def test_loss_has_two_terms(self, model, dummy_batch):
        images, labels = dummy_batch
        with torch.no_grad():
            out = model(images)
        criterion = UALDGLoss()
        losses = criterion(
            p_hat = out["p_hat"], y = labels,
            alpha = out["alpha"], beta = out["beta"],
        )
        assert set(losses.keys()) == {"total", "cls", "edl"}, \
            f"Expected keys {{total, cls, edl}}, got {set(losses.keys())}"

    def test_loss_backward(self):
        """Gradients must flow through the full model."""
        n_ij, n_i, mask, embeds = make_synthetic_graph_data()
        m = UALDG(
            num_labels=NUM_LABELS, n_ij=n_ij, n_i=n_i,
            graph_mask=mask, label_embed_matrix=embeds,
            backbone_pretrained=False,
        )
        m.train()
        images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, 2, (2, NUM_LABELS)).float()

        out = m(images)
        criterion = UALDGLoss()
        losses = criterion(
            p_hat=out["p_hat"], y=labels,
            alpha=out["alpha"], beta=out["beta"],
        )
        losses["total"].backward()

        # Label embeddings should receive a gradient
        assert m.label_embeddings.grad is not None, \
            "label_embeddings did not receive a gradient"
        assert not torch.isnan(m.label_embeddings.grad).any(), \
            "NaN gradient in label_embeddings"
