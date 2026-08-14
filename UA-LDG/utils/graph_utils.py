"""
utils/graph_utils.py
────────────────────
Utilities for building and analysing the disease co-occurrence graph.

Key functions:
  build_cooccurrence_matrix  : compute n_ij (counts) and p_hat (probs)
  load_or_build_graph_data   : high-level loader used by preprocess.py
  visualise_edge_confidence  : plot the confidence heatmap for a paper figure
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


from utils.dataset import PATHOLOGIES

SHORT_NAMES = [
    "Atel", "Card", "Cons", "Edem", "EnCM", "Frac",
    "LLes", "LOpa", "PlEf", "Pneu", "Pntx", "PlOt",
]


# ── Co-occurrence matrix ──────────────────────────────────────────────────────

def build_cooccurrence_matrix(
    label_matrix: np.ndarray,
    pathologies: list = None,
    eps: float = 1e-6,
) -> tuple:
    """
    Compute disease co-occurrence counts and conditional probabilities.

    Args:
        label_matrix : (N, L)  binary label array
        pathologies  : list of L pathology names (for reference)
        eps          : small constant to avoid division by zero

    Returns:
        n_ij  : (L, L)  co-occurrence counts  n_ij = #{samples where i AND j}
        n_i   : (L,)    marginal counts        n_i  = #{samples where i}
        p_hat : (L, L)  conditional probs      p_ij = n_ij / n_i
    """
    if pathologies is None:
        pathologies = PATHOLOGIES

    L = len(pathologies)
    N = label_matrix.shape[0]

    n_i   = label_matrix.sum(axis=0)                 # (L,)
    n_ij  = label_matrix.T @ label_matrix            # (L, L)  dot product

    # Zero diagonal (a pathology doesn't co-occur with itself)
    np.fill_diagonal(n_ij, 0)

    # Conditional probability: P(j|i) = n_ij / n_i
    p_hat = n_ij / (n_i[:, None] + eps)              # (L, L)
    p_hat = np.clip(p_hat, 0.0, 1.0)

    return n_ij.astype(np.float32), n_i.astype(np.float32), p_hat.astype(np.float32)


# ── Cross-dataset consistency ─────────────────────────────────────────────────

def compute_cross_dataset_probs(
    datasets: dict,
    eps: float = 1e-6,
) -> tuple:
    """
    Compute per-dataset conditional probability matrices and derive
    the cross-dataset consistency score S_cross.

    Args:
        datasets : dict mapping dataset_name -> (N, L) label_matrix

    Returns:
        p_hats   : dict mapping dataset_name -> (L, L) p_hat matrix
        s_cross  : (L, L) cross-dataset consistency score
    """
    p_hats = {}
    for name, labels in datasets.items():
        _, n_i, p = build_cooccurrence_matrix(labels)
        p_hats[name] = p

    # Stack into (K, L, L) and compute CoV
    stack = np.stack(list(p_hats.values()), axis=0)  # (K, L, L)
    mu    = stack.mean(axis=0) + eps
    sigma = stack.std(axis=0)
    cov   = sigma / mu                                # coefficient of variation
    s_cross = np.clip(1.0 - cov, 1e-6, 1.0)

    return p_hats, s_cross.astype(np.float32)


# ── Full graph data loader ────────────────────────────────────────────────────

def load_or_build_graph_data(
    processed_root: str,
    primary_labels: np.ndarray,
    threshold: int = 100,
    force_rebuild: bool = False,
) -> dict:
    """
    Load pre-computed graph data if it exists, otherwise build and save it.

    Builds a Beta-Binomial edge weight model: computes co-occurrence counts
    and applies a threshold to determine which edges exist.

    Args:
        processed_root : directory for processed graph files
        primary_labels : (N, L) binary labels from primary dataset
        threshold      : minimum co-occurrences for an edge to exist
        force_rebuild  : always recompute even if files exist

    Returns:
        dict with keys: n_ij, n_i, graph_mask
    """
    cache_path = os.path.join(processed_root, "graph_data.npz")
    os.makedirs(processed_root, exist_ok=True)

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"[graph_utils] Loading cached graph data from {cache_path}")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    print("[graph_utils] Building graph data from scratch ...")

    n_ij, n_i, _ = build_cooccurrence_matrix(primary_labels)
    graph_mask    = (n_ij >= threshold).astype(np.float32)

    data = {"n_ij": n_ij, "n_i": n_i, "graph_mask": graph_mask}
    np.savez(cache_path, **data)
    print(f"[graph_utils] Saved graph data to {cache_path}")
    return data


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_edge_confidence(
    C: np.ndarray,
    title: str = "Edge Confidence Matrix",
    save_path: str = None,
    cmap: str = "YlOrRd",
) -> None:
    """
    Plot the combined edge confidence matrix as a labelled heatmap.

    Args:
        C         : (L, L) edge confidence matrix
        title     : figure title
        save_path : if provided, save figure to this path
        cmap      : matplotlib colourmap
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        C,
        ax=ax,
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        square=True,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Edge Confidence $C_{ij}$"},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Target pathology $j$", fontsize=11)
    ax.set_ylabel("Source pathology $i$", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[graph_utils] Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualise_variance_matrix(
    var_A: np.ndarray,
    save_path: str = None,
) -> None:
    """Plot edge variance matrix (complement of confidence)."""
    visualise_edge_confidence(
        var_A,
        title="Edge Variance Matrix $\\mathrm{Var}[A_{ij}]$",
        save_path=save_path,
        cmap="Blues",
    )
