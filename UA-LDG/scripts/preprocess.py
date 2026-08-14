"""
scripts/preprocess.py
─────────────────────
One-time preprocessing for UA-LDG using MIMIC-CXR-JPG as primary dataset.

What it does:
  1. Loads MIMIC-CXR-JPG training split labels from the official CSVs
  2. Computes co-occurrence counts n_ij and marginal counts n_i
  3. Applies COOCCURRENCE_THRESHOLD to produce graph_mask
  4. Saves graph_data.npz  (n_ij, n_i, graph_mask)
  5. Builds / loads BioWordVec label embeddings

Usage:
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --config configs/default.yaml \\
                                 --config configs/bridges2.yaml
    python scripts/preprocess.py --config configs/default.yaml --force_rebuild
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from utils.dataset import (
    PATHOLOGIES, FRONTAL_VIEWS, COOCCURRENCE_THRESHOLD,
    MIMIC_CHEXPERT_TO_STD, _apply_uncertainty_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UA-LDG preprocessing")
    parser.add_argument("--config", type=str, action="append",
                        default=[], dest="configs",
                        help="Config file(s). Can be passed multiple times; "
                             "later files override earlier ones.")
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


def load_mimic_labels_from_csv(mimic_root, split="train",
                                uncertain_policy="negative",
                                frontal_only=True):
    """Load MIMIC-CXR-JPG labels from CSVs — no image I/O needed."""

    def find_csv(root, *names):
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find any of {names} under {root}")

    split_csv    = find_csv(mimic_root,
                            "mimic-cxr-2.0.0-split.csv.gz",
                            "mimic-cxr-2.0.0-split.csv")
    label_csv    = find_csv(mimic_root,
                            "mimic-cxr-2.0.0-chexpert.csv.gz",
                            "mimic-cxr-2.0.0-chexpert.csv")
    metadata_csv = find_csv(mimic_root,
                            "mimic-cxr-2.0.0-metadata.csv.gz",
                            "mimic-cxr-2.0.0-metadata.csv")

    df_split  = pd.read_csv(split_csv)
    df_labels = pd.read_csv(label_csv)
    df_meta   = pd.read_csv(metadata_csv, usecols=["dicom_id", "ViewPosition"])

    df_split["split"] = df_split["split"].replace("validate", "val")
    df_split = df_split[df_split["split"] == split].copy()

    if frontal_only:
        frontal_ids = df_meta[df_meta["ViewPosition"].isin(FRONTAL_VIEWS)]["dicom_id"]
        df_split = df_split[df_split["dicom_id"].isin(frontal_ids)].copy()

    df = df_split.merge(df_labels, on=["subject_id", "study_id"], how="left")

    label_matrix = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)
    for mimic_col, std_col in MIMIC_CHEXPERT_TO_STD.items():
        if std_col is None or mimic_col not in df.columns:
            continue
        std_idx = PATHOLOGIES.index(std_col)
        label_matrix[:, std_idx] = df[mimic_col].apply(
            lambda v: _apply_uncertainty_policy(v, uncertain_policy)
        ).values

    print(f"  [{split}] {len(df):,} samples (frontal_only={frontal_only})")
    return label_matrix


def build_label_embeddings(embed_path, processed_root, force_rebuild=False):
    out_path = os.path.join(processed_root, "label_embeddings.npy")
    if os.path.exists(out_path) and not force_rebuild:
        print(f"[preprocess] Loaded label embeddings from {out_path}")
        return np.load(out_path)

    embed_dim  = 200
    embeddings = np.zeros((len(PATHOLOGIES), embed_dim), dtype=np.float32)

    if os.path.exists(embed_path):
        try:
            from gensim.models import KeyedVectors
            print(f"[preprocess] Loading BioWordVec from {embed_path} ...")
            wv = KeyedVectors.load_word2vec_format(embed_path, binary=True)
            for i, path in enumerate(PATHOLOGIES):
                words = path.lower().replace("_", " ").split()
                vecs  = [wv[w] for w in words if w in wv]
                embeddings[i] = np.mean(vecs, axis=0) if vecs else \
                    np.random.randn(embed_dim).astype(np.float32)
        except Exception as e:
            print(f"  BioWordVec failed ({e}), using random embeddings.")
            np.random.seed(42)
            embeddings = np.random.randn(len(PATHOLOGIES), embed_dim).astype(np.float32)
    else:
        print(f"[preprocess] BioWordVec not found. Using random embeddings.")
        np.random.seed(42)
        embeddings = np.random.randn(len(PATHOLOGIES), embed_dim).astype(np.float32)

    np.save(out_path, embeddings)
    print(f"[preprocess] Saved label embeddings → {out_path}")
    return embeddings


def main():
    args = parse_args()
    if not args.configs:
        args.configs = ["configs/default.yaml"]
    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in args.configs])

    os.makedirs(cfg.data.processed_root, exist_ok=True)
    os.makedirs(cfg.data.splits_root,    exist_ok=True)

    graph_path = os.path.join(cfg.data.processed_root, "graph_data.npz")

    if os.path.exists(graph_path) and not args.force_rebuild:
        print(f"[preprocess] Graph data exists at {graph_path}")
        print("  Use --force_rebuild to recompute.")
    else:
        mimic_root = cfg.data.mimic_cxr.get(
            "root",
            os.path.dirname(cfg.data.mimic_cxr.image_dir)
        )

        # ── Step 1: Load MIMIC training labels ────────────────────────────────
        print("\n[preprocess] Step 1: MIMIC-CXR-JPG training labels ...")
        if not os.path.exists(mimic_root):
            print(f"  ERROR: mimic_root not found: {mimic_root}")
            print("  Update data.mimic_cxr.root in configs/bridges2.yaml")
            sys.exit(1)

        primary_labels = load_mimic_labels_from_csv(
            mimic_root, split="train",
            uncertain_policy="negative", frontal_only=True,
        )

        # ── Step 2: Co-occurrence counts ──────────────────────────────────────
        print("\n[preprocess] Step 2: Co-occurrence counts ...")
        n_ij = (primary_labels.T @ primary_labels).astype(np.float32)  # (L, L)
        np.fill_diagonal(n_ij, 0)
        n_i  = primary_labels.sum(axis=0).astype(np.float32)            # (L,)
        print(f"  Label matrix: {primary_labels.shape}")
        print(f"  n_ij range: [{n_ij.min():.0f}, {n_ij.max():.0f}]")

        # ── Step 3: Apply threshold to create graph structure ─────────────────
        threshold = cfg.model.get("cooccurrence_threshold", COOCCURRENCE_THRESHOLD)
        print(f"\n[preprocess] Step 3: Applying threshold ({threshold} min co-occurrences) ...")
        graph_mask = (n_ij >= threshold).astype(np.float32)

        n_edges    = int(graph_mask.sum())
        n_possible = len(PATHOLOGIES) * (len(PATHOLOGIES) - 1)
        n_isolated = int((graph_mask.sum(axis=1) == 0).sum())
        print(f"  Possible edges : {n_possible}")
        print(f"  Edges retained : {n_edges}  ({n_edges/n_possible:.1%})")
        print(f"  Isolated nodes : {n_isolated}")

        print("\n  Existing edges:")
        for i in range(len(PATHOLOGIES)):
            for j in range(i + 1, len(PATHOLOGIES)):
                if graph_mask[i, j] == 1:
                    rate = n_ij[i, j] / (n_i[i] + 1e-6)
                    print(f"    {PATHOLOGIES[i]:<30} ↔ "
                          f"{PATHOLOGIES[j]:<30} "
                          f"n={int(n_ij[i,j]):>6}  rate={rate:.3f}")

        # ── Step 4: Save ──────────────────────────────────────────────────────
        np.savez(graph_path,
                 n_ij=n_ij, n_i=n_i, graph_mask=graph_mask)
        print(f"\n[preprocess] Saved graph data → {graph_path}")
        print(f"  Keys: n_ij {n_ij.shape}, n_i {n_i.shape}, "
              f"graph_mask {graph_mask.shape}")

    # ── Step 5: Label embeddings ───────────────────────────────────────────────
    print("\n[preprocess] Step 5: Label embeddings ...")
    build_label_embeddings(
        embed_path     = cfg.model.label_embed_path,
        processed_root = cfg.data.processed_root,
        force_rebuild  = args.force_rebuild,
    )

    print("\n[preprocess] ✓ Done. Run training with:")
    print("    python scripts/train.py --config configs/default.yaml\n")


if __name__ == "__main__":
    main()
