"""
scripts/train.py
────────────────
Main training script for UA-LDG.

Primary dataset: MIMIC-CXR-JPG v2.0.0
  - 377,110 JPG images across 227,827 studies
  - 12 CheXpert labels (MIMIC-CXR-JPG native labels)
  - Official train / validate / test splits from mimic-cxr-2.0.0-split.csv.gz
  - Frontal views only (PA + AP) filtered via mimic-cxr-2.0.0-metadata.csv.gz
  - Uncertain labels (-1.0) handled by uncertain_policy in config

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --config configs/bridges2.yaml
    python scripts/train.py --config configs/default.yaml training.lr=5e-5
    python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/best_model.pt
    python scripts/train.py --config configs/default.yaml --eval_only --resume outputs/checkpoints/best_model.pt
"""

import argparse
import os
import sys
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from models.ua_ldg import UALDG
from utils.dataset import (
    MIMICCXRDataset, ChestXray14Dataset, CheXpertDataset,
    build_dataloader, inspect_dataset, PATHOLOGIES,
)
from utils.losses import UALDGLoss
from utils.metrics import (
    compute_all_metrics, print_metrics_summary,
    compute_full_evaluation, print_results_table,
)
from utils.plots import (
    plot_roc_curves, plot_pr_curves,
    plot_calibration_diagram, plot_vacuity_analysis,
    plot_edge_confidence_heatmaps,
)
from utils.logger import TrainingLogger


# ── Seeds ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimiser, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimiser":  optimiser.state_dict(),
        "metrics":    metrics,
    }, path)


def load_checkpoint(path, model, optimiser=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimiser and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_datasets(cfg, logger):
    """
    Build train / val / test datasets based on cfg.data.primary_dataset.

    Supported primary datasets:
      mimic_cxr   : MIMIC-CXR-JPG v2.0.0 (default — you have this)
      chestxray14 : NIH ChestX-ray14
      chexpert    : Stanford CheXpert
    """
    primary = cfg.data.primary_dataset
    logger.info(f"[train] Primary dataset: {primary}")

    if primary == "mimic_cxr":
        mc = cfg.data.mimic_cxr
        validate_images = mc.get("validate_images", False)
        logger.info(f"[train] validate_images = {validate_images}")

        train_dataset = MIMICCXRDataset(
            mimic_root       = mc.root,
            split            = "train",
            split_csv        = mc.split_csv,
            label_csv        = mc.label_csv,
            metadata_csv     = mc.metadata_csv,
            uncertain_policy = mc.uncertain_policy,
            frontal_only     = mc.frontal_only,
            image_size       = cfg.data.image_size,
            validate_images  = validate_images,
        )
        val_dataset = MIMICCXRDataset(
            mimic_root       = mc.root,
            split            = "val",
            split_csv        = mc.split_csv,
            label_csv        = mc.label_csv,
            metadata_csv     = mc.metadata_csv,
            uncertain_policy = mc.uncertain_policy,
            frontal_only     = mc.frontal_only,
            image_size       = cfg.data.image_size,
            validate_images  = mc.get("validate_images", False),
        )
        test_dataset = MIMICCXRDataset(
            mimic_root       = mc.root,
            split            = "test",
            split_csv        = mc.split_csv,
            label_csv        = mc.label_csv,
            metadata_csv     = mc.metadata_csv,
            uncertain_policy = mc.uncertain_policy,
            frontal_only     = mc.frontal_only,
            image_size       = cfg.data.image_size,
            validate_images  = mc.get("validate_images", False),
        )

    elif primary == "chestxray14":
        cx = cfg.data.chestxray14
        train_dataset = ChestXray14Dataset(
            image_dir  = cx.image_dir,
            label_file = cx.label_file,
            split_file = cx.train_val_list,
            split      = "train",
            image_size = cfg.data.image_size,
        )
        val_dataset = ChestXray14Dataset(
            image_dir  = cx.image_dir,
            label_file = cx.label_file,
            split_file = cx.train_val_list,
            split      = "val",
            image_size = cfg.data.image_size,
        )
        test_dataset = ChestXray14Dataset(
            image_dir  = cx.image_dir,
            label_file = cx.label_file,
            split_file = cx.test_list,
            split      = "test",
            image_size = cfg.data.image_size,
        )

    elif primary == "chexpert":
        cp = cfg.data.chexpert
        train_dataset = CheXpertDataset(
            csv_file         = cp.train_csv,
            data_root        = cp.image_dir,
            split            = "train",
            uncertain_policy = "negative",
            image_size       = cfg.data.image_size,
        )
        val_dataset = CheXpertDataset(
            csv_file         = cp.valid_csv,
            data_root        = cp.image_dir,
            split            = "val",
            uncertain_policy = "negative",
            image_size       = cfg.data.image_size,
        )
        test_dataset = val_dataset   # CheXpert has no public test labels

    else:
        raise ValueError(
            f"Unknown primary_dataset: '{primary}'. "
            "Choose from: mimic_cxr | chestxray14 | chexpert"
        )

    inspect_dataset(train_dataset, f"{primary} [train]")
    return train_dataset, val_dataset, test_dataset


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimiser, scaler,
    device, epoch, logger, log_every=50,
):
    model.train()
    total_losses = {"total": 0.0, "cls": 0.0, "edl": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for step, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.clamp(0.0, 1.0)

        optimiser.zero_grad()

        with autocast("cuda", enabled=(scaler is not None)):
            out    = model(images)
            losses = criterion(
                p_hat         = out["p_hat"],
                y             = labels,
                alpha         = out["alpha"],
                beta          = out["beta"],
                current_epoch = epoch,
            )

        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

        global_step = (epoch - 1) * len(loader) + step
        if step % log_every == 0:
            logger.log_scalars("train_loss", losses, global_step)
            logger.log_scalars("edge_stats", out["edge_stats"], global_step)

        pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_vacuity = [], [], []

    for images, labels, _ in tqdm(loader, desc="  [val]", leave=False):
        images = images.to(device, non_blocking=True)
        out    = model(images)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(out["p_hat"].cpu().numpy())
        all_vacuity.append(out["u"].cpu().numpy())

    y_true  = np.concatenate(all_labels,  axis=0)
    y_pred  = np.concatenate(all_preds,   axis=0)
    vacuity = np.concatenate(all_vacuity, axis=0)

    # Clamp labels for metric computation (remove -1 sentinels)
    y_true = y_true.clip(0, 1)
    return compute_all_metrics(y_true, y_pred, vacuity=vacuity)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run inference over a dataloader and return raw numpy arrays.
    Used for the final evaluation only (threshold tuning + full metrics).

    Returns:
        y_true   : (N, L) ground-truth labels clipped to [0, 1]
        y_pred   : (N, L) predicted probabilities
        vacuity  : (N, L) EDL vacuity scores
    """
    model.eval()
    all_labels, all_preds, all_vacuity = [], [], []
    for images, labels, _ in tqdm(loader, desc="  [collect]", leave=False):
        images = images.to(device, non_blocking=True)
        out    = model(images)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(out["p_hat"].cpu().numpy())
        all_vacuity.append(out["u"].cpu().numpy())
    y_true  = np.concatenate(all_labels,  axis=0).clip(0, 1)
    y_pred  = np.concatenate(all_preds,   axis=0)
    vacuity = np.concatenate(all_vacuity, axis=0)
    return y_true, y_pred, vacuity


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train UA-LDG")
    parser.add_argument("--config", type=str, action="append",
                        default=[], dest="configs",
                        help="Config file(s). Can be passed multiple times; "
                             "later files override earlier ones.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("overrides", nargs="*",
                        help="Dotlist overrides e.g. training.lr=5e-5")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Support multiple --config flags (later overrides earlier)
    if not args.configs:
        args.configs = ["configs/default.yaml"]
    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in args.configs])
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    set_seed(cfg.training.seed)

    device = torch.device(
        cfg.hardware.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[train] Device: {device}")

    # ── Logger ────────────────────────────────────────────────────────────────
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    logger = TrainingLogger(
        log_dir        = cfg.logging.log_dir,
        use_tensorboard= cfg.logging.use_tensorboard,
        use_wandb      = cfg.logging.use_wandb,
        wandb_project  = cfg.logging.wandb_project,
    )
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Graph data ────────────────────────────────────────────────────────────
    graph_path = os.path.join(cfg.data.processed_root, "graph_data.npz")
    embed_path = os.path.join(cfg.data.processed_root, "label_embeddings.npy")
    for p, label in [(graph_path, "graph_data.npz"),
                     (embed_path, "label_embeddings.npy")]:
        if not os.path.exists(p):
            logger.info(f"[train] ERROR: {label} not found at {p}")
            logger.info("  Run: python scripts/preprocess.py --config configs/default.yaml")
            sys.exit(1)

    graph_data       = np.load(graph_path)
    label_embeddings = np.load(embed_path)
    logger.info(f"[train] Graph data: n_ij{graph_data['n_ij'].shape} | "
                f"edges={int(graph_data['graph_mask'].sum())} | "
                f"Label embeddings: {label_embeddings.shape}")

    # ── Datasets & dataloaders ────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_datasets(cfg, logger)

    train_loader = build_dataloader(train_ds, "train",
                                    cfg.training.batch_size,
                                    cfg.data.num_workers,
                                    cfg.data.pin_memory)
    val_loader   = build_dataloader(val_ds, "val",
                                    cfg.training.batch_size,
                                    cfg.data.num_workers,
                                    cfg.data.pin_memory)
    test_loader  = build_dataloader(test_ds, "test",
                                    cfg.training.batch_size,
                                    cfg.data.num_workers,
                                    cfg.data.pin_memory)

    logger.info(f"[train] Sizes — train:{len(train_ds):,} | "
                f"val:{len(val_ds):,} | test:{len(test_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_variant = cfg.model.get("model_variant", "ua_ldg")
    logger.info(f"[train] Building UA-LDG model (variant={model_variant}) ...")
    model = UALDG(
        num_labels                 = len(PATHOLOGIES),
        n_ij                       = graph_data["n_ij"],
        n_i                        = graph_data["n_i"],
        graph_mask                 = graph_data["graph_mask"],
        label_embed_matrix         = label_embeddings,
        image_embed_dim            = cfg.model.image_embed_dim,
        label_embed_dim            = cfg.model.label_embed_dim,
        node_hidden_dim            = cfg.model.node_hidden_dim,
        gamma                      = cfg.model.gamma,
        gcn_layers                 = cfg.model.gcn_layers,
        gcn_dropout                = cfg.model.gcn_dropout,
        use_att_gate               = cfg.model.use_attention_gate,
        edl_hidden_dim             = cfg.model.edl_hidden_dim,
        backbone_pretrained        = cfg.model.backbone_pretrained,
        backbone_pretrained_source = cfg.model.get("backbone_pretrained_source", "imagenet"),
        backbone_freeze_bn         = cfg.model.backbone_freeze_bn,
        model_variant              = model_variant,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[train] Trainable parameters: {n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    param_groups = model.get_param_groups(cfg.training.lr)
    optimiser    = optim.AdamW(param_groups,
                               weight_decay=cfg.training.weight_decay)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max",
        patience = cfg.training.lr_patience,
        factor   = cfg.training.lr_factor,
        min_lr   = cfg.training.min_lr,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = UALDGLoss(
        lambda_edl = cfg.training.lambda_edl,
        gamma_neg  = cfg.training.asym_gamma_neg,
        gamma_pos  = cfg.training.asym_gamma_pos,
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    scaler = GradScaler("cuda") if (
        cfg.hardware.mixed_precision and device.type == "cuda"
    ) else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch      = 1
    best_auc         = 0.0
    patience_counter = 0

    if args.resume:
        start_epoch, best_metrics = load_checkpoint(args.resume, model, optimiser)
        best_auc    = best_metrics.get("mean_auc", 0.0)
        start_epoch += 1
        logger.info(f"[train] Resumed from epoch {start_epoch-1}, AUC={best_auc:.4f}")

    if args.eval_only:
        logger.info("[train] Eval-only mode on test set ...")
        metrics = validate(model, test_loader, device)
        print_metrics_summary(metrics)
        logger.close()
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    logger.info(f"[train] Starting training — {cfg.training.epochs} epochs ...")

    for epoch in range(start_epoch, cfg.training.epochs + 1):

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimiser, scaler,
            device, epoch, logger, cfg.logging.log_every,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step(val_metrics["mean_auc"])

        logger.log_epoch(epoch, train_losses, val_metrics)
        logger.log_scalars("val", val_metrics, step=epoch * len(train_loader))

        if epoch % cfg.training.save_every == 0:
            ckpt = os.path.join(cfg.training.save_dir,
                                f"checkpoint_epoch{epoch:03d}.pt")
            save_checkpoint(model, optimiser, epoch, val_metrics, ckpt)

        current_auc = val_metrics.get("mean_auc", 0.0)
        if current_auc > best_auc:
            best_auc         = current_auc
            patience_counter = 0
            best_path = os.path.join(cfg.training.save_dir, "best_model.pt")
            save_checkpoint(model, optimiser, epoch, val_metrics, best_path)
            logger.info(f"  ★ New best AUC={best_auc:.4f} → {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(f"[train] Early stopping at epoch {epoch}")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\n[train] Loading best checkpoint for final evaluation ...")
    best_path = os.path.join(cfg.training.save_dir, "best_model.pt")
    load_checkpoint(best_path, model)

    # Collect val predictions for per-class threshold optimisation
    logger.info("[train] Collecting val predictions (threshold tuning) ...")
    y_true_val, y_pred_val, _ = collect_predictions(model, val_loader, device)

    logger.info("[train] Collecting test predictions ...")
    y_true_test, y_pred_test, vacuity_test = collect_predictions(model, test_loader, device)

    # Full evaluation (bootstrap CI = 1000 samples)
    logger.info("[train] Computing comprehensive metrics (n_bootstrap=1000) ...")
    eval_results = compute_full_evaluation(
        y_true_test, y_pred_test, vacuity_test,
        y_true_val   = y_true_val,
        y_pred_val   = y_pred_val,
        n_bootstrap  = 1000,
        num_bins_ece = cfg.evaluation.num_bins_ece,
    )

    # Edge weight summary (only for variants that have a graph)
    if hasattr(model, 'edge_weights'):
        stats = model.edge_weights.get_edge_summary()
        logger.info(
            f"[train] Edge summary: "
            f"n_edges={stats['n_edges']}  "
            f"mean_weight={stats['mean_edge_weight']:.4f}  "
            f"mean_var={stats['mean_variance']:.4f}"
        )
    else:
        logger.info("[train] Edge summary: no_graph variant — no graph edges")

    # Paper-ready table
    print_results_table(eval_results)

    # Results directory derived from save_dir so ablation variants don't collide
    results_dir = os.path.join(os.path.dirname(cfg.training.save_dir), "results")
    fig_dir     = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_roc_curves(
        y_true_test, y_pred_test,
        os.path.join(fig_dir, "roc_curves.png"),
    )
    plot_pr_curves(
        y_true_test, y_pred_test,
        os.path.join(fig_dir, "pr_curves.png"),
    )
    plot_calibration_diagram(
        y_true_test, y_pred_test,
        os.path.join(fig_dir, "calibration.png"),
        num_bins = cfg.evaluation.num_bins_ece,
    )
    plot_vacuity_analysis(
        y_true_test, y_pred_test, vacuity_test,
        os.path.join(fig_dir, "vacuity_analysis.png"),
    )
    # Edge confidence heatmap only for Beta-Binomial variant (requires alpha/beta buffers)
    if hasattr(model, 'edge_weights') and hasattr(model.edge_weights, 'alpha'):
        plot_edge_confidence_heatmaps(
            model,
            os.path.join(fig_dir, "edge_confidence.png"),
        )

    # Save raw predictions for post-hoc analysis
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        os.path.join(results_dir, "predictions.npz"),
        y_true             = y_true_test,
        y_pred             = y_pred_test,
        vacuity            = vacuity_test,
        optimal_thresholds = eval_results["optimal_thresholds"],
    )
    logger.info(f"[train] Raw predictions saved → {results_dir}/predictions.npz")

    scalar_metrics = {
        k: np.array(v)
        for k, v in eval_results.items()
        if isinstance(v, (int, float, np.ndarray))
    }
    np.savez(os.path.join(results_dir, "test_metrics.npz"), **scalar_metrics)
    logger.info(f"[train] Scalar metrics saved  → {results_dir}/test_metrics.npz")

    logger.log_scalars("test", {
        "mean_auc":  eval_results["mean_auc"],
        "mAP":       eval_results["mAP"],
        "macro_f1":  eval_results["macro_f1"],
        "ece":       eval_results["ece_global"],
    }, step=cfg.training.epochs)

    logger.info(f"[train] All results saved to {results_dir}/")
    logger.close()


if __name__ == "__main__":
    main()
