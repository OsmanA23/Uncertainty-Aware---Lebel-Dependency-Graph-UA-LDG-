#!/bin/bash
# =============================================================================
#  scripts/sbatch_setup_preprocess.sh
#  UA-LDG: Complete setup + preprocessing job (CPU, RM-shared partition)
#
#  This script:
#    1. Sets up the environment (venv + dependencies)
#    2. Runs preprocessing (graph building + embeddings)
#
#  Reads MIMIC-CXR-JPG CSVs, builds co-occurrence graph, computes
#  S_stat / S_cross / S_ont, saves graph_data.npz + label_embeddings.npy.
#
#  Submit: sbatch scripts/sbatch_setup_preprocess.sh
#  Monitor: squeue -u $USER
#           tail -f outputs/logs/preprocess_JOBID.out
# =============================================================================

#SBATCH --job-name=ua_ldg_setup_preprocess
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/preprocess_%j.out
#SBATCH --error=outputs/logs/preprocess_%j.err
#SBATCH -A cis250046p

set -euo pipefail

# ── Print job info ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Setup + Preprocessing Job"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  CPUs    : $SLURM_CPUS_PER_TASK"
echo "  Start   : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Load modules ──────────────────────────────────────────────────────────────
echo ""
echo "── Loading modules ────────────────────────────────────────────"
module purge
module load anaconda3/2024.10-1
module load cuda

echo "  Python : $(python3 --version)"
echo "  GCC    : $(gcc --version | head -1)"
echo ""

# ── Setup environment ─────────────────────────────────────────────────────────
echo "── Setting up environment ─────────────────────────────────────"

VENV_DIR="/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
else
    echo "  Using existing virtual environment at $VENV_DIR"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
echo "  ✓ Activated: $VIRTUAL_ENV"
export PYTHONNOUSERSITE=1
echo "  ✓ PYTHONNOUSERSITE=1 (ignore user site-packages)"
echo ""

# ── Install dependencies ──────────────────────────────────────────────────────
echo "── Installing/verifying dependencies ──────────────────────────"

pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"

# Pin numpy to <2 to avoid incompatibilities with binary wheels built for NumPy 1.x
echo "  Pinning numpy to <2 for binary compatibility..."
pip install 'numpy<2' --quiet
echo "  ✓ numpy pinned to <2"

# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch with CUDA support... (2-3 minutes)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo "  ✓ PyTorch installed"
fi

# Install remaining dependencies (for preprocessing we need pandas, gensim, etc.)
echo "  Installing other dependencies..."
pip install \
    numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.2.0 pandas>=2.0.0 \
    Pillow>=9.5.0 gensim>=4.3.0 pyyaml>=6.0 omegaconf>=2.3.0 \
    tqdm>=4.65.0 networkx>=3.1 pytest>=7.3.0 \
    --quiet
echo "  ✓ All dependencies installed"
echo ""

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p outputs/logs outputs/checkpoints outputs/results

# ── Run preprocessing ──────────────────────────────────────────────────────────
echo "── Starting preprocessing ─────────────────────────────────────"
echo "  Python : $(python3 --version)"
echo ""

python3 scripts/preprocess.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    --force_rebuild

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Preprocessing completed at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Next step: sbatch scripts/sbatch_setup_train.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
