#!/bin/bash
# =============================================================================
#  scripts/slurm_preprocess.sh
#  UA-LDG: One-time preprocessing job (CPU, RM-shared partition)
#
#  Reads MIMIC-CXR-JPG CSVs, builds co-occurrence graph, computes
#  S_stat / S_cross / S_ont, saves graph_data.npz + label_embeddings.npy.
#  No GPU or image I/O required — runs on RM-shared to save GPU SUs.
#
#  Before submitting:
#    1. Confirm setup_bridges2.sh has been run
#    2. Confirm data exists under /ocean/projects/cis250046p/oali2/DeepLearningProject/Dataset/new_pipeline/mimic_jpg
#
#  Submit: sbatch scripts/slurm_preprocess.sh
# =============================================================================

#SBATCH --job-name=ua_ldg_preprocess
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/preprocess_%j.out
#SBATCH --error=outputs/logs/preprocess_%j.err
#SBATCH -A cis250046p

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Preprocessing Job"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1
module load cuda

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"
echo "Python : $(python3 --version)"

mkdir -p outputs/logs outputs/checkpoints outputs/results

# Run preprocessing — reads from configs/bridges2.yaml for Bridges-2 paths
python3 scripts/preprocess.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml

echo ""
echo "  Preprocessing complete at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
