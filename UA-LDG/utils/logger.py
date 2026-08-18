"""
utils/logger.py
───────────────
Training logger that writes to:
  - Console (tqdm-compatible)
  - A plain text log file
  - TensorBoard (optional)
  - Weights & Biases (optional)
"""

import os
import sys
import time
import logging
from typing import Optional

import torch


class TrainingLogger:
    """
    Unified logger for UA-LDG training.

    Args:
        log_dir         : directory for log files
        experiment_name : name prefix for log files
        use_tensorboard : enable TensorBoard logging
        use_wandb       : enable W&B logging
        wandb_project   : W&B project name
    """

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        experiment_name: str = "ua_ldg",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "ua_ldg",
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()

        # ── File logger ───────────────────────────────────────────────────────
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        self._file_logger = logging.getLogger(experiment_name)
        self._file_logger.setLevel(logging.INFO)

        if not self._file_logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._file_logger.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("%(message)s"))
            self._file_logger.addHandler(ch)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self.info(f"TensorBoard logging → {tb_dir}")
            except ImportError:
                self.info("TensorBoard not available; skipping.")

        # ── W&B ───────────────────────────────────────────────────────────────
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=wandb_project,
                                            name=experiment_name)
                self.info(f"W&B run: {self.wandb_run.url}")
            except Exception as e:
                self.info(f"W&B init failed: {e}")

    # ── Logging helpers ───────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._file_logger.info(msg)

    def log_scalars(self, tag: str, values: dict, step: int) -> None:
        """Log a dict of scalar values with a shared tag prefix."""
        for k, v in values.items():
            key = f"{tag}/{k}"
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            if self.tb_writer:
                self.tb_writer.add_scalar(key, val, global_step=step)
            if self.wandb_run:
                self.wandb_run.log({key: val}, step=step)

    def log_epoch(self, epoch: int, train_losses: dict,
                  val_metrics: dict = None) -> None:
        """Pretty-print and log epoch summary."""
        elapsed = (time.time() - self.start_time) / 60
        msg = (
            f"\n{'─'*60}\n"
            f"  Epoch {epoch:3d} | Elapsed {elapsed:.1f} min\n"
            f"  Train → total:{train_losses.get('total',0):.4f}  "
            f"cls:{train_losses.get('cls',0):.4f}  "
            f"edl:{train_losses.get('edl',0):.4f}"
        )
        if val_metrics:
            msg += (
                f"\n  Val   → AUC:{val_metrics.get('mean_auc',0):.4f}  "
                f"mAP:{val_metrics.get('mAP',0):.4f}  "
                f"F1:{val_metrics.get('macro_f1',0):.4f}  "
                f"ECE:{val_metrics.get('ece',0):.4f}"
            )
        msg += f"\n{'─'*60}"
        self.info(msg)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
