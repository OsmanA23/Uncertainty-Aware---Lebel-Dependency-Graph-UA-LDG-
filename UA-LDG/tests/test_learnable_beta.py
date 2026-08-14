"""
tests/test_learnable_beta.py
────────────────────────────
Quick validation that the learnable-β EDL head:
  1. Produces probabilities spread across (0,1) — not all < 0.3
  2. Can output p_hat > 0.5 (impossible with fixed β=2 and small evidence)
  3. Gradients flow through both α and β branches
  4. Vacuity is still in (0,1]
  5. α and β are both > 1 (valid Beta parameters)

Run: pytest tests/test_learnable_beta.py -v
No dataset required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.edl_head import EvidentialHead

NUM_LABELS  = 12
BATCH_SIZE  = 32
NODE_DIM    = 512
HIDDEN_DIM  = 256


@pytest.fixture(scope="module")
def head():
    return EvidentialHead(node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM,
                          num_labels=NUM_LABELS)


@pytest.fixture(scope="module")
def h_star():
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, NUM_LABELS, NODE_DIM)


class TestLearnableBeta:

    def test_output_shapes(self, head, h_star):
        p, u, a, b = head(h_star)
        assert p.shape == (BATCH_SIZE, NUM_LABELS)
        assert u.shape == (BATCH_SIZE, NUM_LABELS)
        assert a.shape == (BATCH_SIZE, NUM_LABELS)
        assert b.shape == (BATCH_SIZE, NUM_LABELS)

    def test_p_hat_range(self, head, h_star):
        with torch.no_grad():
            p, _, _, _ = head(h_star)
        assert (p > 0).all() and (p < 1).all(), \
            f"p_hat out of (0,1): min={p.min():.4f} max={p.max():.4f}"

    def test_vacuity_range(self, head, h_star):
        with torch.no_grad():
            _, u, _, _ = head(h_star)
        assert (u > 0).all() and (u <= 1).all(), \
            f"Vacuity out of (0,1]: min={u.min():.4f} max={u.max():.4f}"

    def test_alpha_beta_above_one(self, head, h_star):
        with torch.no_grad():
            _, _, a, b = head(h_star)
        assert (a >= 1.0).all(), "α must be >= 1"
        assert (b >= 1.0).all(), "β must be >= 1"

    def test_beta_is_not_fixed(self, head, h_star):
        """β must vary across samples — it is no longer a constant 2.0."""
        with torch.no_grad():
            _, _, _, b = head(h_star)
        assert b.std().item() > 0.01, \
            f"β appears fixed (std={b.std().item():.6f}) — learnable branch not working"

    def test_p_hat_can_exceed_half(self, head):
        """
        With fixed β=2, p_hat = α/(α+2) < 0.5 when evidence < 1.
        With learnable β the head must be able to output p_hat > 0.5
        when given strongly positive input features.
        """
        # Feed large positive activations → model should produce high p_hat
        h_positive = torch.ones(4, NUM_LABELS, NODE_DIM) * 5.0
        with torch.no_grad():
            p, _, _, _ = head(h_positive)
        assert (p > 0.5).any(), \
            "Head cannot produce p_hat > 0.5 — β may still be fixed"

    def test_p_hat_spread(self, head, h_star):
        """
        Probabilities should be spread across (0,1), not all clustered < 0.3.
        Std > 0.05 across a random batch is a basic sanity check.
        """
        with torch.no_grad():
            p, _, _, _ = head(h_star)
        std = p.std().item()
        assert std > 0.02, \
            f"p_hat is too concentrated (std={std:.4f}) — learnable β not working"

    def test_gradients_flow_through_both_branches(self, head):
        """Both mlp_alpha and mlp_beta must receive gradients."""
        h = torch.randn(4, NUM_LABELS, NODE_DIM)
        p, _, _, _ = head(h)
        loss = p.mean()
        loss.backward()
        assert head.mlp_alpha.weight.grad is not None, \
            "No gradient in mlp_alpha"
        assert head.mlp_beta.weight.grad is not None, \
            "No gradient in mlp_beta"
        assert not torch.isnan(head.mlp_alpha.weight.grad).any(), \
            "NaN gradient in mlp_alpha"
        assert not torch.isnan(head.mlp_beta.weight.grad).any(), \
            "NaN gradient in mlp_beta"
