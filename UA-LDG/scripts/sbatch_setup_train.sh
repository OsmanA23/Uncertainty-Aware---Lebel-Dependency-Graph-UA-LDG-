#!/bin/bash
# =============================================================================
#  scripts/sbatch_setup_train.sh
#  UA-LDG: Complete setup + training job (GPU-shared, 2×V100 32GB)
#
#  This script:
#    1. Sets up the environment (venv + dependencies)
#    2. Runs training on the configured dataset
#
#  Primary dataset: MIMIC-CXR-JPG v2.0.0 (configured in configs/bridges2.yaml)
#  Expected runtime: ~20 hours total (including setup)
#
#  Submit: sbatch scripts/sbatch_setup_train.sh
#  Monitor: squeue -u $USER
#           tail -f outputs/logs/train_JOBID.out
# =============================================================================

#SBATCH --job-name=ua_ldg_setup_train
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100-32:2             # 2× V100 32GB
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH -A cis250046p

set -euo pipefail

# ── Print job info ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Setup + Training Job"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPUs       : $SLURM_GPUS_ON_NODE"
echo "  CPUs/task  : $SLURM_CPUS_PER_TASK"
echo "  Start      : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Load modules ──────────────────────────────────────────────────────────────
echo ""
echo "── Loading modules ────────────────────────────────────────────"
module purge
module load anaconda3/2024.10-1
module load cuda

echo "  Python : $(python3 --version)"
echo "  CUDA   : $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
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

# Detect PyTorch + CUDA versions
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null | sed 's/+.*//')
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null | sed 's/\.//')

echo "  PyTorch version: $TORCH_VERSION"
echo "  CUDA version   : $CUDA_VERSION"
echo ""

# Install PyTorch Geometric if not present
if python3 -c "import torch_geometric" 2>/dev/null; then
    echo "  ✓ PyTorch Geometric already installed: $(python3 -c 'import torch_geometric; print(torch_geometric.__version__)')"
else
    echo "  Installing PyTorch Geometric... (5-10 minutes)"
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"
    pip install torch-geometric --quiet
    pip install torch-scatter -f "$PYG_URL" --quiet
    pip install torch-sparse -f "$PYG_URL" --quiet
    pip install torch-cluster -f "$PYG_URL" --quiet
    echo "  ✓ PyTorch Geometric installed"
fi

# Install remaining dependencies
echo "  Installing other dependencies..."
pip install \
    numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.2.0 pandas>=2.0.0 \
    Pillow>=9.5.0 'opencv-python-headless<4.12' gensim>=4.3.0 \
    pyyaml>=6.0 omegaconf>=2.3.0 tqdm>=4.65.0 tensorboard>=2.13.0 \
    matplotlib>=3.7.0 seaborn>=0.12.0 networkx>=3.1 pytest>=7.3.0 \
    torchxrayvision>=1.0.0 \
    --quiet
echo "  ✓ All dependencies installed"
echo ""

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p outputs/logs outputs/checkpoints outputs/results

# ── Run training ──────────────────────────────────────────────────────────────
echo "── Starting training ──────────────────────────────────────────"
echo "  Python  : $(python3 --version)"
echo "  PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA    : $(python3 -c 'import torch; print(torch.version.cuda)')"
python3 -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

python3 scripts/train.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Training completed at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
