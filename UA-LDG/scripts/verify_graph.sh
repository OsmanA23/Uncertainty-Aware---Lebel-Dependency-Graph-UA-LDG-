#!/usr/bin/env bash
# =============================================================
#  scripts/verify_graph.sh
#  Quick sanity check on the preprocessed graph data.
#  Run from the project root:
#    bash scripts/verify_graph.sh
# =============================================================

python3 - << 'EOF'
import numpy as np

path = "data/processed/graph_data.npz"

try:
    d = np.load(path)
except FileNotFoundError:
    print(f"ERROR: Could not find {path}")
    print("Have you run: python scripts/preprocess.py ?")
    raise SystemExit(1)

print("\n-- Graph Data Verification ---------------------------")
print(f"  n_ij shape       : {d['n_ij'].shape}   (expect (12, 12))")
print(f"  n_i  shape       : {d['n_i'].shape}    (expect (12,))")
print(f"  graph_mask shape : {d['graph_mask'].shape}  (expect (12, 12))")
print(f"  Edges            : {int(d['graph_mask'].sum())}")
print(f"  Isolated nodes   : {int((d['graph_mask'].sum(1) == 0).sum())}  (want 0)")
print(f"  Max co-occurrence: {int(d['n_ij'].max()):,}  (expect thousands)")
print(f"  Min n_i          : {int(d['n_i'].min()):,}")
print(f"  Max n_i          : {int(d['n_i'].max()):,}")
print("------------------------------------------------------\n")
EOF
