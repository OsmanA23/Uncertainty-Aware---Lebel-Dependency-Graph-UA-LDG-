#!/bin/bash
# =============================================================================
#  scripts/slurm_extract_gates.sh
#  UA-LDG: Extract attention gate values from the trained model.
#
#  Runs one inference pass over the 3,403 test images, accumulates the mean
#  (12×12) gate matrix, and generates gate_heatmap.png.
#
#  Output: outputs/results/gates/
#    mean_gates.npy   — (12,12) mean gate matrix
#    var_A.npy        — (12,12) posterior variance matrix
#    gate_heatmap.png — 3-panel figure (gate | variance | scatter)
#
#  Submit: sbatch scripts/slurm_extract_gates.sh
# =============================================================================

#SBATCH --job-name=ua_ldg_gates
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/extract_gates_%j.out
#SBATCH --error=outputs/logs/extract_gates_%j.err
#SBATCH -A cis250046p

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Gate Extraction"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1
module load cuda

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"

echo "Python  : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

mkdir -p outputs/logs outputs/results/gates

python3 scripts/extract_gates.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    --resume outputs/checkpoints/best_model.pt \
    --out_dir outputs/results/gates \
    data.num_workers=$SLURM_CPUS_PER_TASK

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Gate extraction complete at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
