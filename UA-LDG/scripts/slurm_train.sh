#!/bin/bash
# =============================================================================
#  scripts/slurm_train.sh
#  UA-LDG: Main training job (GPU-shared, 2×V100 32GB)
#
#  Primary dataset: MIMIC-CXR-JPG v2.0.0 (configured in configs/bridges2.yaml)
#  Expected runtime: ~18 hours for 50 epochs on 2×V100
#
#  Before submitting:
#    1. Confirm slurm_preprocess.sh has completed successfully
#    2. Optionally change to H100-shared for faster training (see below)
#
#  Submit: sbatch scripts/slurm_train.sh
#  Monitor: squeue -u $USER
#           tail -f outputs/logs/train_JOBID.out
# =============================================================================

#SBATCH --job-name=ua_ldg_train
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100-32:2             # 2× V100 32GB
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH -A cis250046p

# ── To use H100 instead (faster, ~3× speedup): ───────────────────────────────
##SBATCH --partition=H100-shared
##SBATCH --gres=gpu:h100-80:2

# ── Email notifications (optional): ──────────────────────────────────────────
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=YOUR_EMAIL@university.edu

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Training Job"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPUs       : $SLURM_GPUS_ON_NODE"
echo "  CPUs/task  : $SLURM_CPUS_PER_TASK"
echo "  Dataset    : MIMIC-CXR-JPG v2.0.0"
echo "  Start      : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1
module load cuda

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"

echo "Python  : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python3 -c 'import torch; print(torch.version.cuda)')"
python3 -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outputs/logs outputs/checkpoints outputs/results

echo "── Starting training ──────────────────────────────"

python3 scripts/train.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    data.num_workers=$SLURM_CPUS_PER_TASK

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training complete at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
