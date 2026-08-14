"""
scripts/extract_gates.py
────────────────────────
Extract and visualise attention gate values from the trained UA-LDG model.

Runs one inference pass over the test set (no gradients), collects the per-image
(L, L) gate matrix g_ij from the UncertaintyAttentionGate, and accumulates the
running mean across all 3,403 test images.

Outputs (saved to outputs/results/gates/):
  mean_gates.npy        — (12, 12) mean gate matrix
  gate_heatmap.png      — 3-panel figure:
                            left   = mean g_ij heatmap
                            centre = posterior variance Var[A_ij] (×10^-4)
                            right  = scatter Var vs gate (with Pearson r)

Usage:
    python scripts/extract_gates.py \\
        --config configs/default.yaml \\
        --config configs/bridges2.yaml \\
        --resume outputs/checkpoints/best_model.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from models.ua_ldg import UALDG
from utils.dataset import MIMICCXRDataset, build_dataloader, PATHOLOGIES
from utils.plots import plot_gate_heatmap


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(config_paths: list, overrides: list) -> OmegaConf:
    cfg = OmegaConf.create({})
    for path in config_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    for ov in overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([ov]))
    return cfg


def load_graph_data(cfg) -> tuple:
    processed  = cfg.data.processed_root
    graph_data = np.load(os.path.join(processed, "graph_data.npz"))
    label_emb  = np.load(os.path.join(processed, "label_embeddings.npy"))
    return (
        graph_data["n_ij"],
        graph_data["n_i"],
        graph_data["graph_mask"],
        label_emb,
    )


def build_model(cfg, n_ij, n_i, graph_mask, label_emb) -> UALDG:
    mc = cfg.model
    model = UALDG(
        num_labels                = len(PATHOLOGIES),
        n_ij                      = n_ij,
        n_i                       = n_i,
        graph_mask                = graph_mask,
        label_embed_matrix        = label_emb,
        image_embed_dim           = mc.image_embed_dim,
        label_embed_dim           = mc.label_embed_dim,
        node_hidden_dim           = mc.node_hidden_dim,
        gamma                     = mc.gamma,
        gcn_layers                = mc.gcn_layers,
        gcn_dropout               = mc.gcn_dropout,
        use_att_gate              = mc.use_attention_gate,
        edl_hidden_dim            = mc.edl_hidden_dim,
        backbone_pretrained       = mc.backbone_pretrained,
        backbone_pretrained_source= mc.backbone_pretrained_source,
        backbone_freeze_bn        = mc.backbone_freeze_bn,
        model_variant             = "ua_ldg",
    )
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  action="append", default=[], dest="configs")
    parser.add_argument("--resume",  required=True, help="Path to best_model.pt")
    parser.add_argument("--out_dir", default="outputs/results/gates")
    args, overrides = parser.parse_known_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.configs, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_gates] Device: {device}")

    # ── Graph data + model ────────────────────────────────────────────────────
    n_ij, n_i, graph_mask, label_emb = load_graph_data(cfg)
    model = build_model(cfg, n_ij, n_i, graph_mask, label_emb)

    ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"[extract_gates] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Test dataset ──────────────────────────────────────────────────────────
    mc = cfg.data.mimic_cxr
    test_ds = MIMICCXRDataset(
        mimic_root       = mc.root,
        split            = "test",
        split_csv        = mc.split_csv,
        label_csv        = mc.label_csv,
        metadata_csv     = mc.metadata_csv,
        uncertain_policy = mc.uncertain_policy,
        frontal_only     = mc.frontal_only,
        image_size       = cfg.data.image_size,
        validate_images  = False,
    )
    loader = build_dataloader(test_ds, "test",
                              batch_size     = 64,
                              num_workers    = cfg.data.num_workers,
                              pin_memory     = cfg.data.pin_memory)
    print(f"[extract_gates] Test set: {len(test_ds)} images")

    # ── Accumulate gate values ────────────────────────────────────────────────
    L = len(PATHOLOGIES)
    gate_sum   = np.zeros((L, L), dtype=np.float64)
    gate_count = 0

    with torch.no_grad():
        for images, _labels, _meta in tqdm(loader, desc="Extracting gates"):
            images = images.to(device, non_blocking=True)
            out    = model(images)

            if out["gates"] is None:
                raise RuntimeError(
                    "gates is None — model may not have use_att_gate=True"
                )

            # gates: (B, L, L)  values in (0, 1)
            gate_np    = out["gates"].cpu().numpy()   # (B, L, L)
            gate_sum  += gate_np.sum(axis=0)
            gate_count += gate_np.shape[0]

    mean_gates = gate_sum / gate_count    # (L, L)
    print(f"[extract_gates] Averaged gates over {gate_count} images")
    print(f"  Gate range: [{mean_gates.min():.4f}, {mean_gates.max():.4f}]")
    print(f"  Active edge mean gate : {mean_gates[graph_mask.astype(bool)].mean():.4f}")
    print(f"  Inactive edge mean gate: {mean_gates[~graph_mask.astype(bool)].mean():.4f}")

    # ── Posterior variance (fixed — same for all images) ──────────────────────
    var_A = (model.edge_weights.get_posterior_variance()
             .cpu().numpy())                          # (L, L)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "mean_gates.npy"), mean_gates)
    np.save(os.path.join(args.out_dir, "var_A.npy"),      var_A)
    print(f"[extract_gates] Saved mean_gates.npy and var_A.npy → {args.out_dir}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "gate_heatmap.png")
    plot_gate_heatmap(
        mean_gates = mean_gates,
        var_A      = var_A,
        graph_mask = graph_mask,
        save_path  = save_path,
    )
    print(f"[extract_gates] Saved gate_heatmap.png → {args.out_dir}")

    # ── Print top-5 most trusted and least trusted active edges ───────────────
    mask = graph_mask.astype(bool)
    edges = [(mean_gates[i, j], var_A[i, j] * 1e4, PATHOLOGIES[i], PATHOLOGIES[j])
             for i in range(L) for j in range(L) if mask[i, j] and i != j]
    edges.sort(key=lambda x: -x[0])

    print("\n  Top-5 most trusted edges (high gate):")
    print(f"  {'Source':<30} {'Target':<30} {'Gate':>6}  {'Var×1e4':>8}")
    print("  " + "-" * 78)
    for g, v, src, tgt in edges[:5]:
        print(f"  {src:<30} {tgt:<30} {g:.4f}  {v:.4f}")

    print("\n  Bottom-5 least trusted edges (low gate):")
    print(f"  {'Source':<30} {'Target':<30} {'Gate':>6}  {'Var×1e4':>8}")
    print("  " + "-" * 78)
    for g, v, src, tgt in edges[-5:]:
        print(f"  {src:<30} {tgt:<30} {g:.4f}  {v:.4f}")


if __name__ == "__main__":
    main()
