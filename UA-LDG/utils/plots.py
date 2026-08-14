"""
utils/plots.py
──────────────
Figure generators for paper-ready evaluation of UA-LDG.

  plot_roc_curves          : 12-class + macro-average ROC curves
  plot_pr_curves           : 12-class precision-recall curves
  plot_calibration_diagram : 3×4 per-class reliability diagram
  plot_vacuity_analysis    : box plots of vacuity on correct vs incorrect predictions

All functions save to disk and do not display interactively.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required on HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

try:
    from utils.dataset import PATHOLOGIES
    from utils.graph_utils import SHORT_NAMES
except ImportError:
    from dataset import PATHOLOGIES
    from graph_utils import SHORT_NAMES


# 12 visually distinct colours — one per class
COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "ROC Curves — UA-LDG (Test Set)",
) -> None:
    """
    One ROC curve per class + dashed macro-average curve.

    Args:
        y_true    : (N, L) binary labels
        y_pred    : (N, L) predicted probabilities
        save_path : output PNG path
        title     : figure title
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    mean_fpr = np.linspace(0, 1, 300)
    interp_tprs = []

    for i, (path, color) in enumerate(zip(PATHOLOGIES, COLORS)):
        if y_true[:, i].sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except Exception:
            continue
        ax.plot(fpr, tpr, color=color, lw=1.3, alpha=0.85,
                label=f"{path} ({auc:.3f})")
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))

    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[[0, -1]] = [0.0, 1.0]
        valid_aucs = [
            roc_auc_score(y_true[:, i], y_pred[:, i])
            for i in range(len(PATHOLOGIES))
            if y_true[:, i].sum() > 0
        ]
        macro_auc = float(np.mean(valid_aucs))
        ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5, ls="--",
                label=f"Macro avg ({macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.25)
    _save(fig, save_path)


# ── Precision-Recall curves ───────────────────────────────────────────────────

def plot_pr_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Precision-Recall Curves — UA-LDG (Test Set)",
) -> None:
    """
    One PR curve per class. Legend shows per-class Average Precision.

    Args:
        y_true    : (N, L)
        y_pred    : (N, L)
        save_path : output PNG path
        title     : figure title
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    for i, (path, color) in enumerate(zip(PATHOLOGIES, COLORS)):
        if y_true[:, i].sum() == 0:
            continue
        try:
            prec, rec, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        except Exception:
            continue
        ax.plot(rec, prec, color=color, lw=1.3, alpha=0.85,
                label=f"{path} (AP={ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    _save(fig, save_path)


# ── Calibration / reliability diagram ────────────────────────────────────────

def plot_calibration_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    num_bins: int = 15,
    title: str = "Calibration (Reliability Diagram) — UA-LDG",
) -> None:
    """
    3×4 subplot grid — one reliability diagram per class.

    Each panel shows predicted confidence on the x-axis vs actual positive
    frequency on the y-axis. The dashed diagonal is perfect calibration.

    Args:
        y_true    : (N, L)
        y_pred    : (N, L)
        save_path : output PNG path
        num_bins  : number of equal-width confidence bins
        title     : figure suptitle
    """
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for i, (path, ax) in enumerate(zip(PATHOLOGIES, axes)):
        yt, yp = y_true[:, i], y_pred[:, i]
        bin_conf, bin_acc, bin_w = [], [], []
        for b in range(num_bins):
            mask = (yp >= bin_edges[b]) & (yp < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            bin_conf.append(float(yp[mask].mean()))
            bin_acc.append(float(yt[mask].mean()))
            bin_w.append(int(mask.sum()))

        if bin_conf:
            bar_w = 1.0 / num_bins * 0.85
            ax.bar(bin_conf, bin_acc, width=bar_w,
                   color="#377eb8", alpha=0.7, label="Actual freq")
            # Gap between perfect and actual → calibration error shading
            for bc, ba in zip(bin_conf, bin_acc):
                ax.fill_betweenx([min(bc, ba), max(bc, ba)],
                                 bc - bar_w / 2, bc + bar_w / 2,
                                 color="#e41a1c", alpha=0.25)

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(path, fontsize=9, pad=4)
        ax.set_xlabel("Confidence", fontsize=7)
        ax.set_ylabel("Frequency",  fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide any unused subplots (shouldn't happen with 12 classes + 3×4)
    for j in range(len(PATHOLOGIES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


# ── Vacuity analysis ──────────────────────────────────────────────────────────

def plot_edge_confidence_heatmaps(
    model,
    save_path: str,
    title: str = "Beta-Binomial Edge Weights — UA-LDG",
) -> None:
    """
    2×2 figure showing the Beta-Binomial edge weight model.

      Top-left  : n_ij          — raw co-occurrence counts
      Top-right : graph_mask    — which edges meet the threshold (binary)
      Bot-left  : E[A_ij]       — posterior mean edge weight
      Bot-right : Var[A_ij]     — posterior variance (uncertainty)

    Args:
        model     : trained UALDG instance
        save_path : output PNG path
        title     : figure suptitle
    """
    import torch
    import seaborn as sns

    model.eval()
    with torch.no_grad():
        mean  = model.edge_weights.get_posterior_mean().cpu().numpy()
        var_A = model.edge_weights.get_posterior_variance().cpu().numpy()
        mask  = model.edge_weights.graph_mask.cpu().numpy()
        alpha = model.edge_weights.alpha.cpu().numpy()
        beta  = model.edge_weights.beta_param.cpu().numpy()
        # Recover n_ij from alpha: alpha = 1 + n_ij  → n_ij = alpha - 1
        n_ij  = (alpha - 1.0) * mask

    matrices = [
        (n_ij,          r"Co-occurrence Counts $n_{ij}$",                       "Blues"),
        (mask,          r"Graph Mask (threshold applied)",                       "Greys"),
        (mean,          r"Posterior Mean $\mathbb{E}[A_{ij}]$",                  "YlOrRd"),
        (var_A * 1e4,   r"Posterior Variance $\mathrm{Var}[A_{ij}]$ ($\times10^{-4}$)", "Purples"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (mat, subtitle, cmap) in zip(axes, matrices):
        try:
            vmax = float(np.nanmax(mat)) if np.nanmax(mat) > 0 else 1.0
            if "Counts" in subtitle:
                fmt = ".0f"
            elif "Variance" in subtitle:
                fmt = ".2f"
            else:
                fmt = ".3f"
            sns.heatmap(
                mat, ax=ax,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                cmap=cmap, vmin=0.0, vmax=vmax,
                square=True, linewidths=0.3, linecolor="white",
                annot=True, fmt=fmt, annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8},
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(subtitle, fontsize=11, pad=8)
        ax.set_xlabel("Target $j$", fontsize=9)
        ax.set_ylabel("Source $i$", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


def plot_gate_heatmap(
    mean_gates: np.ndarray,
    var_A: np.ndarray,
    graph_mask: np.ndarray,
    save_path: str,
    title: str = "Attention Gate Analysis — UA-LDG",
) -> None:
    """
    Three-panel figure visualising learned edge trust.

      Left   : Mean attention gate values  mean(g_ij) across the test set.
               High value → model trusts this edge; low value → suppressed.
      Centre : Posterior variance Var[A_ij] (×10^{-4}).
               High value → statistically unreliable edge (few co-occurrences).
      Right  : Scatter — Var[A_ij] vs mean g_ij for every active edge.
               A negative trend confirms the design intent: uncertain edges
               receive lower gate values.

    Only active edges (graph_mask == 1) are shown / included in the scatter.

    Args:
        mean_gates  : (L, L) mean gate matrix averaged over test set
        var_A       : (L, L) posterior variance matrix (raw, not scaled)
        graph_mask  : (L, L) binary mask of active edges
        save_path   : output PNG path
        title       : figure suptitle
    """
    import seaborn as sns
    from scipy import stats as scipy_stats

    mask = graph_mask.astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── Left: mean gate heatmap ────────────────────────────────────────────────
    gate_display = np.where(mask, mean_gates, np.nan)
    sns.heatmap(
        gate_display, ax=axes[0],
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        square=True, linewidths=0.3, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "label": "Mean gate g_ij"},
        mask=~mask,
    )
    axes[0].set_title(r"Mean Attention Gate $\bar{g}_{ij}$", fontsize=12, pad=8)
    axes[0].set_xlabel("Target $j$", fontsize=9)
    axes[0].set_ylabel("Source $i$", fontsize=9)
    axes[0].tick_params(labelsize=7)

    # ── Centre: posterior variance heatmap ────────────────────────────────────
    var_display = np.where(mask, var_A * 1e4, np.nan)
    vmax_var = float(np.nanmax(var_display)) if np.nanmax(var_display) > 0 else 1.0
    sns.heatmap(
        var_display, ax=axes[1],
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        cmap="Purples", vmin=0.0, vmax=vmax_var,
        square=True, linewidths=0.3, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "label": r"Var[$A_{ij}$] (×10$^{-4}$)"},
        mask=~mask,
    )
    axes[1].set_title(
        r"Posterior Variance $\mathrm{Var}[A_{ij}]$ ($\times10^{-4}$)",
        fontsize=12, pad=8,
    )
    axes[1].set_xlabel("Target $j$", fontsize=9)
    axes[1].set_ylabel("Source $i$", fontsize=9)
    axes[1].tick_params(labelsize=7)

    # ── Right: scatter Var vs gate (active edges only) ─────────────────────────
    ax = axes[2]
    x = var_A[mask] * 1e4          # posterior variance (scaled)
    y = mean_gates[mask]           # mean gate value
    ax.scatter(x, y, alpha=0.6, s=40, color="#377eb8", edgecolors="white", linewidths=0.5)

    # Regression line + Pearson r
    if len(x) > 2:
        slope, intercept, r, p_val, _ = scipy_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="#e41a1c", lw=1.8,
                label=f"r = {r:.3f}  (p={p_val:.3g})")
        ax.legend(fontsize=9)

    ax.set_xlabel(r"Posterior Variance $\mathrm{Var}[A_{ij}]$ ($\times10^{-4}$)", fontsize=10)
    ax.set_ylabel(r"Mean Gate Value $\bar{g}_{ij}$", fontsize=10)
    ax.set_title("Edge Uncertainty vs. Learned Trust", fontsize=12, pad=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


def plot_vacuity_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vacuity: np.ndarray,
    save_path: str,
    title: str = "Vacuity: Correct vs Incorrect Predictions — UA-LDG",
) -> None:
    """
    Side-by-side box plots of vacuity scores split by prediction correctness.

    Green boxes = correct predictions, red boxes = incorrect predictions.
    A well-calibrated uncertainty model should show higher vacuity on
    incorrect predictions.

    Args:
        y_true    : (N, L)
        y_pred    : (N, L)
        vacuity   : (N, L)  u from EDL head
        save_path : output PNG path
        title     : figure title
    """
    data_correct, data_wrong, labels = [], [], []

    for i, path in enumerate(PATHOLOGIES):
        y_bin = (y_pred[:, i] >= 0.5).astype(int)
        is_correct = (y_bin == y_true[:, i].astype(int))
        vc = vacuity[is_correct,  i]
        vw = vacuity[~is_correct, i]
        if len(vc) > 1 and len(vw) > 1:
            data_correct.append(vc)
            data_wrong.append(vw)
            labels.append(path.replace(" ", "\n"))

    if not labels:
        print("[plots] Warning: not enough data for vacuity plot — skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 6))
    x = np.arange(len(labels))
    w = 0.35

    def _bp(data, positions, color):
        return ax.boxplot(
            data, positions=positions, widths=w * 0.9,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.72),
            medianprops=dict(color="black", lw=1.8),
            whiskerprops=dict(lw=1),
            capprops=dict(lw=1),
            flierprops=dict(marker=".", ms=2, alpha=0.3),
            showfliers=True,
        )

    _bp(data_correct, x - w / 2, "#4daf4a")
    _bp(data_wrong,   x + w / 2, "#e41a1c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Vacuity (uncertainty score)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color="#4daf4a", alpha=0.72, label="Correct predictions"),
        mpatches.Patch(color="#e41a1c", alpha=0.72, label="Incorrect predictions"),
    ], fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, save_path)
