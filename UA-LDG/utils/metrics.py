"""
utils/metrics.py
────────────────
Evaluation metrics for UA-LDG.

Lightweight functions (used every epoch in the training loop):
  compute_auc, compute_map, compute_f1, compute_ece,
  compute_uncertainty_auroc, compute_all_metrics, print_metrics_summary

Full evaluation functions (used once at the end of training):
  find_optimal_thresholds, compute_bootstrap_ci,
  compute_per_class_precision_recall_f1, compute_per_class_ece,
  compute_per_class_uncertainty, compute_full_evaluation, print_results_table
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, roc_curve,
    precision_score, recall_score,
)

try:
    from utils.dataset import PATHOLOGIES
except ImportError:
    from dataset import PATHOLOGIES


# ── AUC-ROC ──────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Per-class and mean AUC-ROC.

    Args:
        y_true : (N, L) binary ground-truth
        y_pred : (N, L) predicted probabilities

    Returns:
        dict with per-class AUCs and "mean_auc"
    """
    results = {}
    aucs = []
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            results[path] = float("nan")
            continue
        try:
            auc = roc_auc_score(col[valid], y_pred[valid, i])
        except Exception:
            auc = float("nan")
        results[path] = auc
        aucs.append(auc)

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    results["mean_auc"] = np.mean(valid_aucs) if valid_aucs else float("nan")
    return results


# ── Mean Average Precision ────────────────────────────────────────────────────

def compute_map(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class Average Precision and mean AP."""
    results = {}
    aps = []
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            results[path] = float("nan")
            continue
        try:
            ap = average_precision_score(col[valid], y_pred[valid, i])
        except Exception:
            ap = float("nan")
        results[path] = ap
        aps.append(ap)

    valid_aps = [a for a in aps if not np.isnan(a)]
    results["mAP"] = np.mean(valid_aps) if valid_aps else float("nan")
    return results


# ── F1 ───────────────────────────────────────────────────────────────────────

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5) -> dict:
    """Per-class and macro F1 at a given threshold."""
    y_bin = (y_pred >= threshold).astype(int)
    results = {}
    f1s = []
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            results[path] = float("nan")
            continue
        f1 = f1_score(col[valid], y_bin[valid, i], zero_division=0)
        results[path] = f1
        f1s.append(f1)

    valid_f1s = [f for f in f1s if not np.isnan(f)]
    results["macro_f1"] = np.mean(valid_f1s) if valid_f1s else float("nan")
    return results


# ── Expected Calibration Error ────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray,
                num_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) over all labels and samples.

    Lower ECE means predictions are better calibrated:
    a predicted probability of 0.7 should correspond to ~70% actual
    frequency of positive labels.

    Args:
        y_true   : (N, L) binary ground-truth (flattened over labels)
        y_pred   : (N, L) predicted probabilities
        num_bins : number of equal-width bins in [0, 1]

    Returns:
        ece : scalar float
    """
    flat_true   = y_true.flatten()
    flat_pred   = y_pred.flatten()
    valid       = ~np.isnan(flat_true)
    y_true_flat = flat_true[valid]
    y_pred_flat = flat_pred[valid]

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(y_true_flat)

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask   = (y_pred_flat >= lo) & (y_pred_flat < hi)
        if mask.sum() == 0:
            continue
        acc_b  = y_true_flat[mask].mean()       # fraction of true positives
        conf_b = y_pred_flat[mask].mean()        # average predicted prob
        ece   += (mask.sum() / total) * abs(acc_b - conf_b)

    return float(ece)


# ── Uncertainty-AUROC ─────────────────────────────────────────────────────────

def compute_uncertainty_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vacuity: np.ndarray,
) -> float:
    """
    Uncertainty-AUROC: measures whether the vacuity score can distinguish
    correct from incorrect predictions.

    A correct prediction is where (y_pred >= 0.5) == y_true.
    We compute AUROC of vacuity as a detector of incorrect predictions.

    Higher score = vacuity is a useful signal of prediction correctness.

    Args:
        y_true   : (N, L)
        y_pred   : (N, L)
        vacuity  : (N, L)  u_l from EDL head (higher = more uncertain)

    Returns:
        auroc : scalar float
    """
    valid        = ~np.isnan(y_true.flatten())
    y_true_valid = y_true.flatten()[valid]
    y_pred_valid = y_pred.flatten()[valid]
    vac_flat     = vacuity.flatten()[valid]
    y_pred_bin   = (y_pred_valid >= 0.5).astype(int)
    # 1 = wrong prediction, 0 = correct prediction
    is_wrong     = (y_pred_bin != y_true_valid.astype(int)).astype(int)

    if is_wrong.sum() == 0 or is_wrong.sum() == len(is_wrong):
        return float("nan")

    try:
        return roc_auc_score(is_wrong, vac_flat)
    except Exception:
        return float("nan")


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vacuity: np.ndarray = None,
    threshold: float = 0.5,
    num_bins_ece: int = 15,
) -> dict:
    """
    Compute and return all evaluation metrics in one call.

    Args:
        y_true   : (N, L) binary ground-truth
        y_pred   : (N, L) predicted probabilities ∈ (0,1)
        vacuity  : (N, L) optional vacuity scores from EDL head
        threshold: for F1 binarisation
        num_bins_ece: for ECE computation

    Returns:
        dict with all metric values
    """
    metrics = {}
    metrics.update(compute_auc(y_true, y_pred))
    metrics.update(compute_map(y_true, y_pred))
    metrics.update(compute_f1(y_true, y_pred, threshold=threshold))
    metrics["ece"] = compute_ece(y_true, y_pred, num_bins=num_bins_ece)

    if vacuity is not None:
        metrics["uncertainty_auroc"] = compute_uncertainty_auroc(
            y_true, y_pred, vacuity
        )

    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Pretty-print the key metrics to stdout."""
    print("\n" + "=" * 55)
    print(f"  Mean AUC-ROC : {metrics.get('mean_auc', float('nan')):.4f}")
    print(f"  mAP          : {metrics.get('mAP',      float('nan')):.4f}")
    print(f"  Macro F1     : {metrics.get('macro_f1', float('nan')):.4f}")
    print(f"  ECE          : {metrics.get('ece',      float('nan')):.4f}")
    if "uncertainty_auroc" in metrics:
        print(f"  Uncert-AUROC : {metrics['uncertainty_auroc']:.4f}")
    print("=" * 55)
    print("\n  Per-class AUC:")
    for p in PATHOLOGIES:
        val = metrics.get(p, float("nan"))
        bar = "█" * int(val * 20) if not np.isnan(val) else ""
        print(f"    {p:<22} {val:.4f}  {bar}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Full evaluation — used once after training, not in the per-epoch loop
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Find the per-class threshold that maximises F1 on a held-out set.

    Args:
        y_true : (N, L) binary labels from the validation set
        y_pred : (N, L) predicted probabilities from the validation set

    Returns:
        thresholds : (L,) optimal threshold per class (default 0.3 when no positives)
    """
    L = y_true.shape[1]
    candidates = np.arange(0.05, 0.96, 0.01)
    thresholds = np.full(L, 0.3, dtype=np.float32)
    for i in range(L):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.3
        for t in candidates:
            y_bin = (y_pred[valid, i] >= t).astype(int)
            f1 = f1_score(col[valid], y_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[i] = best_t
    return thresholds


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Per-class AUC-ROC with bootstrap confidence interval.

    Args:
        y_true      : (N, L)
        y_pred      : (N, L)
        n_bootstrap : number of bootstrap samples
        ci          : confidence level (default 0.95 → 95% CI)
        seed        : random seed for reproducibility

    Returns:
        dict mapping class_name → (mean_auc, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    alpha = (1.0 - ci) / 2.0
    results = {}
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            results[path] = (float("nan"), float("nan"), float("nan"))
            continue
        col_v  = col[valid]
        pred_v = y_pred[valid, i]
        N_v    = int(valid.sum())
        boot_aucs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(N_v, N_v, replace=True)
            yt, yp = col_v[idx], pred_v[idx]
            if yt.sum() == 0 or yt.sum() == N_v:
                continue
            try:
                boot_aucs.append(roc_auc_score(yt, yp))
            except Exception:
                continue
        if not boot_aucs:
            results[path] = (float("nan"), float("nan"), float("nan"))
            continue
        arr = np.array(boot_aucs)
        results[path] = (
            float(np.mean(arr)),
            float(np.percentile(arr, alpha * 100)),
            float(np.percentile(arr, (1.0 - alpha) * 100)),
        )
    return results


def compute_per_class_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """
    Per-class precision, recall, F1 evaluated at the given per-class thresholds.

    Args:
        y_true     : (N, L)
        y_pred     : (N, L)
        thresholds : (L,) one threshold per class

    Returns:
        dict mapping class_name → {precision, recall, f1, threshold}
    """
    results = {}
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0 or col[valid].sum() == 0:
            results[path] = {
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "threshold": float(thresholds[i]),
            }
            continue
        y_bin = (y_pred[valid, i] >= thresholds[i]).astype(int)
        results[path] = {
            "precision": float(precision_score(col[valid], y_bin, zero_division=0)),
            "recall":    float(recall_score(col[valid], y_bin, zero_division=0)),
            "f1":        float(f1_score(col[valid], y_bin, zero_division=0)),
            "threshold": float(thresholds[i]),
        }
    return results


def compute_per_class_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int = 15,
) -> dict:
    """
    Expected Calibration Error computed independently for each class.

    Args:
        y_true    : (N, L)
        y_pred    : (N, L)
        num_bins  : number of equal-width bins

    Returns:
        dict mapping class_name → ECE float
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    results = {}
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            results[path] = float("nan")
            continue
        yt, yp = col[valid], y_pred[valid, i]
        ece, total = 0.0, len(yt)
        for b in range(num_bins):
            mask = (yp >= bin_edges[b]) & (yp < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / total) * abs(yt[mask].mean() - yp[mask].mean())
        results[path] = float(ece)
    return results


def compute_per_class_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vacuity: np.ndarray,
) -> dict:
    """
    Per-class uncertainty analysis.

    For each class computes:
      - uncertainty_auroc       : AUROC of vacuity as a detector of wrong predictions
      - mean_vacuity_correct    : average vacuity on correctly predicted samples
      - mean_vacuity_incorrect  : average vacuity on incorrectly predicted samples

    Args:
        y_true   : (N, L)
        y_pred   : (N, L)
        vacuity  : (N, L)  u from EDL head — higher means more uncertain

    Returns:
        dict mapping class_name → {uncertainty_auroc, mean_vacuity_correct,
                                    mean_vacuity_incorrect}
    """
    results = {}
    for i, path in enumerate(PATHOLOGIES):
        col   = y_true[:, i]
        valid = ~np.isnan(col)
        yt    = col[valid]
        yp    = y_pred[valid, i]
        vac   = vacuity[valid, i]
        y_bin = (yp >= 0.5).astype(int)
        is_wrong = (y_bin != yt.astype(int)).astype(int)
        correct_mask = is_wrong == 0
        results[path] = {
            "mean_vacuity_correct":   float(vac[correct_mask].mean())   if correct_mask.sum() > 0  else float("nan"),
            "mean_vacuity_incorrect": float(vac[~correct_mask].mean())  if (~correct_mask).sum() > 0 else float("nan"),
            "uncertainty_auroc":      float("nan"),
        }
        if 0 < is_wrong.sum() < len(is_wrong):
            try:
                results[path]["uncertainty_auroc"] = float(roc_auc_score(is_wrong, vac))
            except Exception:
                pass
    return results


def compute_full_evaluation(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    vacuity_test: np.ndarray,
    y_true_val: np.ndarray = None,
    y_pred_val: np.ndarray = None,
    n_bootstrap: int = 1000,
    num_bins_ece: int = 15,
) -> dict:
    """
    Comprehensive test-set evaluation for paper reporting.

    Thresholds are tuned on the validation set when val predictions are provided;
    otherwise a fixed threshold of 0.3 is used for all classes.

    Args:
        y_true_test  : (N_test, L)
        y_pred_test  : (N_test, L)
        vacuity_test : (N_test, L)
        y_true_val   : (N_val,  L) optional — used for threshold tuning
        y_pred_val   : (N_val,  L) optional
        n_bootstrap  : bootstrap resamples for AUC CI
        num_bins_ece : bins for ECE computation

    Returns:
        Comprehensive dict with all metric values and metadata.
    """
    results = {}

    # Optimal thresholds
    if y_true_val is not None and y_pred_val is not None:
        opt_thr = find_optimal_thresholds(y_true_val, y_pred_val)
        results["threshold_source"] = "val_optimised"
    else:
        opt_thr = np.full(len(PATHOLOGIES), 0.3, dtype=np.float32)
        results["threshold_source"] = "fixed_0.3"
    results["optimal_thresholds"] = opt_thr

    # AUC with bootstrap CI
    auc_ci = compute_bootstrap_ci(y_true_test, y_pred_test, n_bootstrap=n_bootstrap)
    results["auc_ci"] = auc_ci
    results["mean_auc"] = float(np.nanmean([v[0] for v in auc_ci.values()]))

    # Per-class AP
    ap_dict = compute_map(y_true_test, y_pred_test)
    results["per_class_ap"] = {p: ap_dict[p] for p in PATHOLOGIES}
    results["mAP"] = ap_dict["mAP"]

    # Per-class precision / recall / F1 at optimal thresholds
    prf = compute_per_class_precision_recall_f1(y_true_test, y_pred_test, opt_thr)
    results["per_class_prf"] = prf
    valid_f1s  = [v["f1"]        for v in prf.values() if not np.isnan(v["f1"])]
    valid_prec = [v["precision"] for v in prf.values() if not np.isnan(v["precision"])]
    valid_rec  = [v["recall"]    for v in prf.values() if not np.isnan(v["recall"])]
    results["macro_f1"]        = float(np.mean(valid_f1s))  if valid_f1s  else float("nan")
    results["macro_precision"] = float(np.mean(valid_prec)) if valid_prec else float("nan")
    results["macro_recall"]    = float(np.mean(valid_rec))  if valid_rec  else float("nan")

    # Calibration
    results["ece_global"]    = compute_ece(y_true_test, y_pred_test, num_bins=num_bins_ece)
    results["per_class_ece"] = compute_per_class_ece(y_true_test, y_pred_test, num_bins=num_bins_ece)

    # Uncertainty
    if vacuity_test is not None:
        results["uncertainty_auroc_global"] = compute_uncertainty_auroc(
            y_true_test, y_pred_test, vacuity_test
        )
        results["per_class_uncertainty"] = compute_per_class_uncertainty(
            y_true_test, y_pred_test, vacuity_test
        )

    return results


def print_results_table(eval_results: dict) -> None:
    """
    Print a paper-ready per-class results table to stdout.

    Columns: Pathology | AUC | 95% CI | AP | F1 | Prec | Recall | ECE | U-AUC | Thr
    Followed by a vacuity analysis table.
    """
    W = 114
    sep = "─" * W

    def _f(v):
        return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "  N/A"

    print(f"\n{'═' * W}")
    print(f"  FINAL TEST RESULTS — UA-LDG")
    print(f"{'═' * W}")
    print(f"  Mean AUC-ROC   : {_f(eval_results.get('mean_auc', float('nan')))}")
    print(f"  mAP            : {_f(eval_results.get('mAP', float('nan')))}")
    print(f"  Macro F1       : {_f(eval_results.get('macro_f1', float('nan')))}")
    print(f"  Macro Precision: {_f(eval_results.get('macro_precision', float('nan')))}")
    print(f"  Macro Recall   : {_f(eval_results.get('macro_recall', float('nan')))}")
    print(f"  ECE (global)   : {_f(eval_results.get('ece_global', float('nan')))}")
    if "uncertainty_auroc_global" in eval_results:
        print(f"  Uncert-AUROC   : {_f(eval_results['uncertainty_auroc_global'])}")
    print(f"  Threshold src  : {eval_results.get('threshold_source', 'N/A')}")

    auc_ci      = eval_results.get("auc_ci",              {})
    per_ap      = eval_results.get("per_class_ap",        {})
    per_prf     = eval_results.get("per_class_prf",       {})
    per_ece     = eval_results.get("per_class_ece",       {})
    per_unc     = eval_results.get("per_class_uncertainty",{})

    print(f"\n{sep}")
    hdr = (f"  {'Pathology':<30} {'AUC':>6} {'95% CI':>14} "
           f"{'AP':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} "
           f"{'ECE':>6} {'U-AUC':>6} {'Thr':>5}")
    print(hdr)
    print(sep)

    for path in PATHOLOGIES:
        am, al, ah = auc_ci.get(path, (float("nan"),) * 3)
        ap   = per_ap.get(path, float("nan"))
        prf  = per_prf.get(path, {})
        ece  = per_ece.get(path, float("nan"))
        unc  = per_unc.get(path, {}).get("uncertainty_auroc", float("nan")) if per_unc else float("nan")
        f1   = prf.get("f1",        float("nan"))
        prec = prf.get("precision", float("nan"))
        rec  = prf.get("recall",    float("nan"))
        thr  = prf.get("threshold", float("nan"))
        ci_s = f"[{al:.3f},{ah:.3f}]" if not np.isnan(al) else "    N/A    "
        print(f"  {path:<30} {_f(am):>6} {ci_s:>14} "
              f"{_f(ap):>6} {_f(f1):>6} {_f(prec):>6} {_f(rec):>6} "
              f"{_f(ece):>6} {_f(unc):>6} {thr:>5.2f}")

    print(sep)

    # Vacuity analysis
    if per_unc:
        print(f"\n  Vacuity Analysis  (higher vacuity → model is more uncertain)")
        print(f"  {'Pathology':<30} {'Vac_correct':>12} {'Vac_wrong':>12} {'Δ (wrong−correct)':>18}")
        print(f"  {'─' * 76}")
        for path in PATHOLOGIES:
            s = per_unc.get(path, {})
            vc = s.get("mean_vacuity_correct",   float("nan"))
            vw = s.get("mean_vacuity_incorrect", float("nan"))
            delta = vw - vc if not (np.isnan(vc) or np.isnan(vw)) else float("nan")
            print(f"  {path:<30} {_f(vc):>12} {_f(vw):>12} {_f(delta):>18}")

    print(f"\n{'═' * W}\n")
